"""Fixtures of the end-to-end smoke tests (marked `smoke` by the collection
hook below, like tests/gpu/conftest.py marks `gpu`).

`run_cli` runs the real entry point in process, `cli.main([VIDEO,
"--config", base.yaml, "--set", ...])`, with the scene doubles installed;
every call gets its own output.out_root because make_run_dir names folders
with a one-second timestamp. Videos and outputs live in tmp_path only.
"""

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
import yaml
from run_outputs import single_run_dir
from smoke_fakes import DepthOracle, install_scene_fakes
from smoke_scene import SCENE, scene_frames, write_scene_video

from det_depth_distance import cli
from det_depth_distance.camera import Calibration, Undistorter, save_calibration
from det_depth_distance.core.types import CameraIntrinsics

SMOKE_DIR = Path(__file__).parent
REPO = SMOKE_DIR.parents[1]

# Base run: one box-only detector, the real DA3Depth on a 154x84 map, the
# focal of the true camera at 320x180; cases change it with --set only.
BASE_CONFIG: dict[str, Any] = {
    "device": "cpu",
    "detectors": [
        {
            "type": "scene_det",
            "name": "box",
            "pad_px": 1,
            "score": 0.8,
            "vehicle_label": "bus",
        }
    ],
    "depth": {"type": "da3", "model": "da3metric-large", "process_res": 154},
    "camera": {"focal_px": 725.0 / 3, "estimate": None},
}


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    # The hook receives the whole session's items, hence the path filter.
    for item in items:
        if SMOKE_DIR in item.path.parents:
            item.add_marker(pytest.mark.smoke)


@dataclass(frozen=True, slots=True)
class SceneVideos:
    """The scene seen by a pinhole camera and by the distorting true camera."""

    pinhole: Path
    distorted: Path
    distorted_labels: tuple[np.ndarray, ...]  # ground truth for reference boxes
    true_k: CameraIntrinsics  # 320x180, continuous convention
    true_undistorted_k: CameraIntrinsics  # Undistorter(true, alpha=0).output


def _true_calibration(camera) -> Calibration:
    """The synthetic camera (960x540) as a project Calibration."""
    return Calibration.from_opencv(camera.k, camera.dist, camera.size)


@pytest.fixture
def scene_videos(tmp_path, synthetic_camera, distort_image) -> SceneVideos:
    """Both scene videos in tmp_path; the distorted one renders the label image
    through the true lens at 320x180 (exactly 1/3 of the calibrated size)."""
    true = _true_calibration(synthetic_camera).scaled_to(*SCENE.size)
    k, dist = true.camera_matrix(), true.dist_array()

    def lens(labels: np.ndarray) -> np.ndarray:
        # Corner pixels whose ray falls outside the ideal frame see the road.
        return distort_image(
            labels,
            k,
            dist,
            interpolation=cv2.INTER_NEAREST,
            border_value=SCENE.index("road"),
        )

    distorted = scene_frames(SCENE, lens)
    return SceneVideos(
        pinhole=write_scene_video(SCENE, tmp_path / "scene.mp4", scene_frames(SCENE)),
        distorted=write_scene_video(SCENE, tmp_path / "scene_lens.mp4", distorted),
        distorted_labels=tuple(distorted),
        true_k=true.intrinsics,
        true_undistorted_k=Undistorter(true, alpha=0.0).output.intrinsics,
    )


@pytest.fixture
def true_calibration(tmp_path, synthetic_camera) -> Path:
    """Calibration file of the true camera (960x540, as a calibration of the
    recording mode would be)."""
    path = tmp_path / "true_camera.yaml"
    save_calibration(_true_calibration(synthetic_camera), path)
    return path


@pytest.fixture
def scene_runtime(monkeypatch, scene_videos) -> DepthOracle:
    """Scene doubles registered and torch seams patched for this test; the
    returned oracle's `perceived` focal starts as the true pinhole K."""
    oracle = DepthOracle(SCENE, scene_videos.true_k)
    install_scene_fakes(monkeypatch, oracle)
    return oracle


@pytest.fixture
def run_cli(tmp_path, scene_runtime) -> Callable[..., Path]:
    """Factory: run_cli(video, overrides=(), *, perceived) -> run folder.

    Runs `cli.main` with the video as positional argument, BASE_CONFIG as
    --config and every override as `--set`, asserts exit code 0 and returns
    the run folder. `perceived` is the frame focal the fake DA3 sees in this
    video (the true K, or the undistorted one).
    """
    config = tmp_path / "base.yaml"
    config.write_text(yaml.safe_dump(BASE_CONFIG))
    counter = itertools.count()

    def run(
        video: Path, overrides: Sequence[str] = (), *, perceived: CameraIntrinsics
    ) -> Path:
        out_root = tmp_path / f"run{next(counter)}"
        scene_runtime.perceived = perceived
        argv = [str(video), "--config", str(config)]
        for override in (*overrides, f"output.out_root={out_root}"):
            argv += ["--set", override]
        assert cli.main(argv) == 0
        return single_run_dir(out_root)

    return run
