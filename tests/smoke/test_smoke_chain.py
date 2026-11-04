"""Calibration -> run -> evaluation chain on synthetic data.

Rendered boards are calibrated by av-perception-lab-calibrate; the
distorted scene video is processed with camera.calibration with and without
undistortion; av-perception-lab-evaluate compares the runs with references
built from the true geometry (box selectors in original video pixels).
Errors stay within the calibration error and the implied focal recovers the
focal the fake DA3 perceives.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml
from smoke_scene import SCENE, render_labels, true_box
from test_smoke_pipeline import SEG

from av_perception_lab.camera import load_calibration
from av_perception_lab.core.types import CameraIntrinsics
from av_perception_lab.tools import calibrate, evaluate

UNCERTAINTY_M = 0.05


def _write_references(
    path: Path, video: str, labels: np.ndarray, k: CameraIntrinsics
) -> Path:
    """References of B (person) and C (truck) over the whole clip: box selectors on the
    distorted frame `labels` (original video pixels), z distances, and B once
    more as a range measured to the centre of its ideal (pinhole) box with
    the true K, so the evaluation converts range to z."""
    x0, y0, x1, y1 = true_box(render_labels(SCENE, 0), SCENE, "person")
    u, v = ((x0 + x1) / 2 - k.cx) / k.fx, ((y0 + y1) / 2 - k.cy) / k.fy
    ray = math.sqrt(1.0 + u * u + v * v)
    frames = [0, SCENE.n_frames - 1]

    def ref(id_: str, cls: str, distance: float, quantity: str) -> dict:
        return {
            "id": id_,
            "select": {"box": list(true_box(labels, SCENE, cls))},
            "frames": frames,
            "distance_m": distance,
            "quantity": quantity,
            "uncertainty_m": UNCERTAINTY_M,
        }

    references = [
        ref("b", "person", SCENE.cls("person").depth_m, "z"),
        ref("c", "truck", SCENE.cls("truck").depth_m, "z"),
        ref("b_range", "person", SCENE.cls("person").depth_m * ray, "range"),
    ]
    path.write_text(yaml.safe_dump({"video": video, "references": references}))
    return path


def test_calibrate_run_evaluate(board_images, run_cli, scene_videos, tmp_path):
    cal = tmp_path / "estimated.yaml"
    pattern = str(board_images / "*.png")
    assert calibrate.main(["calibrate", "--images", pattern, "--out", str(cal)]) == 0
    estimated = load_calibration(cal)
    assert estimated.image_size == (960, 540)
    assert estimated.intrinsics.fx == pytest.approx(725.0, rel=5e-3)

    common = [
        f"detectors=[{json.dumps(SEG)}]",
        "distance.region=mask",
        "camera.focal_px=null",
    ]
    video = scene_videos.distorted
    # Tags keep the run names apart: the evaluation keys results by folder name.
    plain = run_cli(
        video,
        [*common, f"camera.calibration={{path: {cal}}}", "output.tag=plain"],
        perceived=scene_videos.true_k,
    )
    undistorted = run_cli(
        video,
        [
            *common,
            f"camera.calibration={{path: {cal}, undistort: true}}",
            "output.tag=undistorted",
        ],
        perceived=scene_videos.true_undistorted_k,
    )
    refs = _write_references(
        tmp_path / "refs.yaml",
        video.name,
        scene_videos.distorted_labels[0],
        scene_videos.true_k,
    )
    out = tmp_path / "eval"
    runs = [str(plain), str(undistorted)]
    assert evaluate.main(["--refs", str(refs), *runs, "--out", str(out)]) == 0

    report = json.loads((out / "eval.json").read_text())
    assert report["warnings"] == []
    perceived = {
        plain.name: scene_videos.true_k,
        undistorted.name: scene_videos.true_undistorted_k,
    }
    for name, k in perceived.items():
        (overall,) = (
            r
            for r in report["overall"]
            if (r["run"], r["source"], r["estimator"]) == (name, "seg", "distance_m")
        )
        # Every reference matched (IoU >= 0.3, also after undistorting the
        # selector boxes with the recorded K) and within the calibration error.
        assert overall["n_refs"] == 3 and overall["median_abs_rel"] < 0.01
        coverage = [c for c in report["coverage"] if c["run"] == name]
        assert len(coverage) == 3
        assert all(c["not_detected"] == 0 for c in coverage)
        assert all(c["detected_valid"] == SCENE.n_frames for c in coverage)
        # DA3METRIC distance scales with the focal: the evaluation recovers the
        # focal the network perceived, whatever the calibration error.
        focal = report["implied_focal"][name]["seg"]
        assert focal == pytest.approx((k.fx + k.fy) / 2, rel=2e-3)
