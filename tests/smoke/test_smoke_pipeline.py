"""Pairwise configuration matrix of the whole pipeline through the CLI.

Nine cases cover every pair of levels of aggregation x region x refiner x
semantic exclusion x tracking+smoothing x calibration+undistort
(`test_cases_cover_every_pair` guards it). Each case checks the exit code,
the run contract (tests/run_outputs.py) and every distance against the
analytic value of the scene. `test_all_off_is_task_0002` pins the
detector-only behaviour: neutral stages leave the task_0001/0002 columns
(REQUIRED_COLUMNS) byte-identical.
"""

import itertools
import json
import subprocess
import sys
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pytest
from run_outputs import RunOutputs, check_run_outputs
from smoke_scene import SCENE, expected_box_distance

from det_depth_distance.evaluation.run_data import REQUIRED_COLUMNS

REPO = Path(__file__).resolve().parents[2]
LABEL_MAP = str(REPO / "configs" / "labels" / "coco_driving.yaml")
# The two detectors: a segmenter and a box-only detector whose boxes are one
# pixel larger and whose truck is a "bus" (fused with the truck through the
# label map, as task_0003 fuses truck x bus).
SEG = {"type": "scene_det", "name": "seg", "masks": True, "score": 0.9}
BOX = {
    "type": "scene_det",
    "name": "box",
    "pad_px": 1,
    "score": 0.8,
    "vehicle_label": "bus",
}
# Raw label of a row -> scene object class.
OBJECT_OF_RAW = {"car": "car", "person": "person", "truck": "truck", "bus": "truck"}
REL_TOL = 1e-4  # float32 rounding of to_metres (8.000000954 for 8 m)
EUCLIDEAN_REL_TOL = 1e-3


@dataclass(frozen=True, slots=True)
class SmokeCase:
    """One run of the matrix: the six pairwise factors, then secondary ones."""

    id: str
    aggregation: Literal["concat", "nms", "wbf"]
    two_detectors: bool  # seg + box (with the label map unless concat), else box
    region: Literal["box", "mask", "both"]
    refiner: Literal["missing", "all"] | None  # scene_sam targets
    exclude_road: bool  # semantic segmenter "scene" excluding scene.road
    smoothing: Literal["kalman", "ema", "median"] | None  # with the iou tracker
    undistort: bool  # distorted video + true calibration with undistort
    distance_type: Literal["z", "euclidean"] = "z"
    exclude_sky: bool = True
    box_shrink: float = 0.1
    stride: int = 1
    save_depth: bool = False
    layout: Literal["overlay", "side-by-side"] = "overlay"
    crop: bool = False  # depth.process_res_method=upper_bound_crop

    @property
    def fused(self) -> bool:
        return self.aggregation != "concat"

    def overrides(self, calibration: Path) -> list[str]:
        """--set values of this case on top of the base config (smoke conftest)."""
        out = []
        if self.two_detectors:
            dets = [SEG, BOX]
            if self.fused:
                dets = [{**d, "labels": LABEL_MAP} for d in dets]
            out.append(f"detectors={json.dumps(dets)}")
        out.append(f"aggregation={{type: {self.aggregation}}}")
        out.append(f"distance.region={self.region}")
        if self.refiner is not None:
            refiner = {"type": "scene_sam", "name": "sam", "targets": self.refiner}
            out.append(f"refiners=[{json.dumps({**refiner, 'max_boxes': 2})}]")
        if self.exclude_road:
            out += [
                "semantic=[{name: scene, type: scene_seg}]",
                "distance.exclude_semantic=[scene.road]",
            ]
        if self.smoothing is not None:
            out += ["tracking={type: iou}", f"smoothing.filter={self.smoothing}"]
        if self.undistort:
            out += [
                "camera.focal_px=null",
                f"camera.calibration={{path: {calibration}, undistort: true}}",
            ]
        out += [
            f"distance.distance_type={self.distance_type}",
            f"distance.exclude_sky={str(self.exclude_sky).lower()}",
            f"distance.box_shrink={self.box_shrink}",
            f"source.stride={self.stride}",
            f"output.save_depth={str(self.save_depth).lower()}",
            f"render.layout={self.layout}",
        ]
        if self.crop:
            out.append("depth.process_res_method=upper_bound_crop")
        return out


CASES: tuple[SmokeCase, ...] = (
    SmokeCase("c01_all_off", "concat", False, "box", None, False, None, False),
    SmokeCase(
        "c02",
        "concat",
        True,
        "mask",
        "missing",
        True,
        "kalman",
        True,
        layout="side-by-side",
    ),
    SmokeCase("c03", "concat", True, "both", None, False, "ema", False, box_shrink=0.0),
    SmokeCase("c04", "nms", True, "box", None, True, "median", True, exclude_sky=False),
    SmokeCase(
        "c05", "nms", True, "mask", "all", True, None, False, crop=True, stride=2
    ),
    SmokeCase("c06", "nms", True, "both", "missing", False, None, True),
    SmokeCase(
        "c07",
        "wbf",
        True,
        "box",
        "missing",
        True,
        None,
        True,
        distance_type="euclidean",
        save_depth=True,
    ),
    SmokeCase(
        "c08",
        "wbf",
        True,
        "mask",
        None,
        False,
        "kalman",
        False,
        distance_type="euclidean",
    ),
    SmokeCase(
        "c09_all_on",
        "wbf",
        True,
        "both",
        "missing",
        True,
        "kalman",
        True,
        save_depth=True,
        layout="side-by-side",
    ),
)
# The factors every pair of levels of which some case covers.
PAIRWISE_FACTORS: dict[str, Callable[[SmokeCase], object]] = {
    "aggregation": lambda c: c.aggregation,
    "region": lambda c: c.region,
    "refiner": lambda c: c.refiner is not None,
    "exclude_road": lambda c: c.exclude_road,
    "tracking": lambda c: c.smoothing is not None,
    "undistort": lambda c: c.undistort,
}
LEVELS = {
    "aggregation": ("concat", "nms", "wbf"),
    "region": ("box", "mask", "both"),
}


def test_cases_cover_every_pair():
    for (a, fa), (b, fb) in itertools.combinations(PAIRWISE_FACTORS.items(), 2):
        seen = {(fa(c), fb(c)) for c in CASES}
        wanted = set(
            itertools.product(
                LEVELS.get(a, (False, True)), LEVELS.get(b, (False, True))
            )
        )
        assert seen == wanted, (a, b, wanted - seen)


def _frame_ray_factors(
    window: tuple[int, int, int, int], k: Mapping[str, float]
) -> np.ndarray:
    """z -> euclidean range factor at the centres of a depth-map window (the
    pinhole formula written out here, independent of distance.ray_factors)."""
    c0, r0, c1, r1 = window
    u = (np.arange(c0, c1) + 0.5 - k["cx"]) / k["fx"]
    v = (np.arange(r0, r1) + 0.5 - k["cy"]) / k["fy"]
    return np.sqrt(1.0 + u[None, :] ** 2 + v[:, None] ** 2)


def _check_distance(
    value: str, z: float, row: Mapping[str, str], case: SmokeCase, k: Mapping[str, Any]
) -> None:
    """`value` is z (distance type z) or, for euclidean, z times a ray factor
    of the row's depth-map window (exactly the median one for a solid box)."""
    d = float(value)
    if case.distance_type == "z":
        assert d == pytest.approx(z, rel=REL_TOL), (row, z)
        return
    window = tuple(int(row[c]) for c in ("map_c0", "map_r0", "map_c1", "map_r1"))
    factors = _frame_ray_factors(window, k)
    assert z * factors.min() * (1 - REL_TOL) <= d <= z * factors.max() * (1 + REL_TOL)
    if row["used_mask"] == "0" and OBJECT_OF_RAW[row["raw_label"]] == "car":
        # Car A is solid: every pixel of its shrunk box is the car plane.
        assert d == pytest.approx(z * np.median(factors), rel=EUCLIDEAN_REL_TOL)


def _check_rows(out: RunOutputs, case: SmokeCase) -> None:
    rows, metrics = out.rows, out.metrics
    n_frames = SCENE.n_frames // case.stride
    assert metrics["frames_processed"] == n_frames
    assert metrics["boxes_without_distance"] == 0
    per_frame = 3 if case.fused or not case.two_detectors else 6
    frames = [int(r["frame"]) for r in rows]
    assert sorted(set(frames)) == list(range(0, SCENE.n_frames, case.stride))
    assert all(frames.count(f) == per_frame for f in set(frames))

    excluded = {"road"} if case.exclude_road else set()
    k = metrics["focal_map_intrinsics"]
    for row in rows:
        cls = OBJECT_OF_RAW[row["raw_label"]]
        z_object = SCENE.cls(cls).depth_m
        z_box = expected_box_distance(
            SCENE,
            cls,
            exclude_sky=case.exclude_sky,
            excluded=excluded,
            box_shrink=case.box_shrink,
        )
        # Labels: canonical through the map with fusion, raw otherwise.
        expected_label = (
            ("vehicle" if cls != "person" else "person")
            if case.fused
            else row["raw_label"]
        )
        assert row["class_name"] == expected_label
        # Masks: the segmenter's, the refiner's ("sam") or none.
        has_detector_mask = case.fused or row["source"] == "seg"
        if case.refiner == "all" or (
            case.refiner == "missing" and not has_detector_mask
        ):
            assert row["mask_source"] == "sam" and row["mask_score"] == "0.95"
        else:
            assert row["mask_source"] == ("seg" if has_detector_mask else "")
        used_mask = case.region != "box" and row["mask_source"] != ""
        assert row["used_mask"] == str(int(used_mask)), row
        _check_distance(
            row["distance_m"], z_object if used_mask else z_box, row, case, k
        )
        if case.region == "both" and used_mask:
            _check_distance(row["box_distance_m"], z_box, row, case, k)
        else:
            assert row["box_distance_m"] == ""


def _check_metrics(out: RunOutputs, case: SmokeCase) -> None:
    metrics, rows = out.metrics, out.rows
    n_sources = "2" if case.fused else "1"
    assert metrics["detections_by_n_sources"] == {n_sources: len(rows)}
    assert ("agreement" in metrics) == case.two_detectors
    assert ("region_comparison" in metrics) == (case.region == "both")
    used = sum(r["used_mask"] == "1" for r in rows)
    assert metrics["distance_regions"] == {"mask": used, "box": len(rows) - used}
    if case.refiner is not None:
        assert set(metrics["refined"]["sam"]) == {"targets", "filled"}
    else:
        assert metrics["refined"] == {}
    if case.exclude_road:
        scene = metrics["semantic"]["maps"]["scene"]
        assert scene["exclude"] == ["road"] and 0 < scene["excluded_frac_mean"] < 1
    if case.undistort:
        camera = metrics["camera"]
        assert camera["scaled_from"] == [960, 540]
        assert camera["lens_check"]["status"] == "unavailable"
        assert camera["undistort"]["alpha"] == 0.0
        assert metrics["video"]["transform"] is not None
        assert metrics["focal_source"].startswith("calibration")
    else:
        assert metrics["camera"] == {} and metrics["video"]["transform"] is None


def _check_tracks(out: RunOutputs, case: SmokeCase) -> None:
    rows, metrics = out.rows, out.metrics
    smooth_columns = ("distance_smooth_m", "distance_smooth_std_m", "rel_speed_mps")
    if case.smoothing is None:
        assert all(r["track_id"] == "" for r in rows)
        assert all(r[c] == "" for r in rows for c in smooth_columns)
        return
    by_object: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_object[(OBJECT_OF_RAW[row["raw_label"]], row["source"])].append(row)
    ids = {key: {r["track_id"] for r in group} for key, group in by_object.items()}
    assert all(len(v) == 1 and "" not in v for v in ids.values()), ids  # one id each
    assert metrics["tracks"]["n_tracks"] == len(by_object) == (3 if case.fused else 6)
    for group in by_object.values():
        raw = [float(r["distance_m"]) for r in group]
        constant = max(raw) - min(raw) < 1e-6
        for r in group:
            # A constant measurement is a fixed point of every filter.
            if constant:
                assert float(r["distance_smooth_m"]) == pytest.approx(raw[0], abs=1e-4)
            kalman = case.smoothing == "kalman"
            assert (r["distance_smooth_std_m"] != "") == kalman
            assert (r["rel_speed_mps"] != "") == kalman
            if kalman and constant:
                assert abs(float(r["rel_speed_mps"])) < 1e-6


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_pipeline_case(case, run_cli, scene_videos, true_calibration):
    video = scene_videos.distorted if case.undistort else scene_videos.pinhole
    perceived = (
        scene_videos.true_undistorted_k if case.undistort else scene_videos.true_k
    )
    run_dir = run_cli(video, case.overrides(true_calibration), perceived=perceived)
    out = check_run_outputs(run_dir)
    _check_rows(out, case)
    _check_metrics(out, case)
    _check_tracks(out, case)


def _columns(
    rows: Sequence[Mapping[str, str]], names: Sequence[str]
) -> list[list[str]]:
    return [[r[c] for c in names] for r in rows]


def test_all_off_is_task_0002(run_cli, scene_videos, true_calibration):
    (case,) = (c for c in CASES if c.id == "c01_all_off")
    off = check_run_outputs(
        run_cli(
            scene_videos.pinhole,
            case.overrides(true_calibration),
            perceived=scene_videos.true_k,
        )
    )
    for r in off.rows:
        assert r["used_mask"] == "0" and r["mask_source"] == ""
        empty = (
            "mask_px",
            "mask_erode",
            "box_distance_m",
            "mask_score",
            "track_id",
            "distance_smooth_m",
            "distance_smooth_std_m",
            "rel_speed_mps",
            "rel_speed_std_mps",
        )
        assert all(r[c] == "" for c in empty), r
        assert (r["n_sources"], r["sources"]) == ("1", r["source"])
        assert r["raw_labels"] == r["raw_label"]
        assert r["source_scores"] == r["confidence"]
    m = off.metrics
    assert not {"agreement", "region_comparison", "semantic", "tracks"} & set(m)
    components = m["components"]
    assert components["refiners"] == [] and components["semantic"] == []
    assert components["tracker"] is None and m["pipeline"]["smoothing"] is None
    assert m["camera"] == {} and m["video"]["transform"] is None
    assert m["frame_intrinsics"] == {
        "fx": 725 / 3,
        "fy": 725 / 3,
        "cx": 160.0,
        "cy": 90.0,
    }

    # Neutral stages: region "both" without masks, a tracker and a smoother.
    neutral = check_run_outputs(
        run_cli(
            scene_videos.pinhole,
            [
                *case.overrides(true_calibration),
                "distance.region=both",
                "tracking={type: iou}",
                "smoothing.filter=kalman",
            ],
            perceived=scene_videos.true_k,
        )
    )
    assert _columns(neutral.rows, REQUIRED_COLUMNS) == _columns(
        off.rows, REQUIRED_COLUMNS
    )


def test_python_m_entry_point():
    out = subprocess.run(
        [sys.executable, "-m", "det_depth_distance", "--list-models"],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr
    for family in ("detector:", "depth:", "refiner:", "semantic:", "tracker:"):
        assert family in out.stdout
    assert "scene_det" not in out.stdout  # the doubles never leave the test process
