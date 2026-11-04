"""Shared test helpers. The `make_instances` fixture returns a factory that
builds contract-valid Instances (dtypes, raw_label = label, one source) from
boxes and optional fields; request the fixture and call what it returns,
`def test_x(make_instances): inst = make_instances(...)`, so tests of every
task construct instances the same way.

task_0007 adds `synthetic_camera` (a known camera and chessboard with
poses that keep the board in view), `render_board` (factory: the distorted
grayscale image of the board at a pose) and `fake_run_dir` (factory: a run
folder with detections.csv and metrics.json, `fake_run_dir(rows, metrics)`).

task_0008 adds `distort_image` (factory: an ideal pinhole image seen through
a distorting lens, shared by the boards and the smoke scene) and moves
`board_images` (a folder of rendered board views) here from
test_tools_calibrate.py, since the calibrate -> run -> evaluate smoke uses
it too.
"""

import csv
import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from av_perception_lab.core.types import Instances
from av_perception_lab.io.sinks import CSV_FIELDS


def _make(
    xyxy: Sequence[Sequence[float]],
    *,
    score: Sequence[float] | None = None,
    labels: Sequence[str] | None = None,
    source: str = "det",
    class_id: Sequence[int] | None = None,
    masks: np.ndarray | None = None,
    **extra,
) -> Instances:
    boxes = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
    n = len(boxes)
    labels = tuple(labels) if labels is not None else ("car",) * n
    return Instances(
        boxes,
        np.asarray(score if score is not None else [0.9] * n, dtype=np.float32),
        np.asarray(class_id if class_id is not None else range(n), dtype=np.int64),
        labels,
        labels,
        (source,) * n,
        masks,
        **extra,
    )


@pytest.fixture
def make_instances():
    """Factory of contract-valid Instances (see the module docstring)."""
    return _make


@dataclass(frozen=True)
class SyntheticCamera:
    """Pinhole camera with radial distortion (OpenCV convention) and a 9x6
    chessboard of 25 mm squares."""

    size: tuple[int, int] = (960, 540)
    k: np.ndarray = field(
        default_factory=lambda: np.array(
            [[725.0, 0.0, 479.5], [0.0, 725.0, 269.5], [0.0, 0.0, 1.0]]
        )
    )
    dist: np.ndarray = field(
        default_factory=lambda: np.array([[0.10, -0.20, 0.0, 0.0, 0.0]])
    )
    cols: int = 9
    rows: int = 6
    square_m: float = 0.025

    def object_points(self) -> np.ndarray:
        grid = np.mgrid[0 : self.cols, 0 : self.rows].T.reshape(-1, 2)
        obj = np.zeros((len(grid), 3), np.float32)
        obj[:, :2] = grid * self.square_m
        return obj

    def project(self, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        """(K, 2) distorted image points of the inner corners, OpenCV convention."""
        pts, _ = cv2.projectPoints(self.object_points(), rvec, tvec, self.k, self.dist)
        return pts.reshape(-1, 2)

    def poses(self, n: int, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
        """`n` tilted poses (up to ~35 deg) spread over the frame, board fully visible
        with a 30 px margin."""
        rng = np.random.default_rng(seed)
        w, h = self.size
        centre = np.array([(self.cols - 1) / 2, (self.rows - 1) / 2, 0]) * self.square_m
        out = []
        while len(out) < n:
            rvec = rng.uniform(-0.6, 0.6, 3) * np.array([1.0, 1.0, 0.5])
            z = rng.uniform(0.35, 0.6)
            target = np.array(
                [
                    rng.uniform(-0.35, 0.35) * w / 725 * z,
                    rng.uniform(-0.3, 0.3) * h / 725 * z,
                    z,
                ]
            )
            rot, _ = cv2.Rodrigues(rvec)
            tvec = target - rot @ centre  # board centre at `target`
            pts = self.project(rvec, tvec)
            if (pts.min(0) > 30).all() and (pts.max(0) < np.array([w, h]) - 30).all():
                out.append((rvec, tvec))
        return out


@pytest.fixture
def synthetic_camera() -> SyntheticCamera:
    return SyntheticCamera()


def _render(cam: SyntheticCamera, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Board rendered through the pinhole camera, then distorted with remap."""
    sq_px = 40  # texture pixels per square
    margin = sq_px  # white border around the board, one square wide
    nx, ny = cam.cols + 1, cam.rows + 1
    tex = np.full((ny * sq_px + 2 * margin, nx * sq_px + 2 * margin), 255, np.uint8)
    for j, i in itertools.product(range(ny), range(nx)):
        if (i + j) % 2 == 0:
            y0, x0 = margin + j * sq_px, margin + i * sq_px
            tex[y0 : y0 + sq_px, x0 : x0 + sq_px] = 0
    # Texture pixel u (centre at u in OpenCV terms) -> board X in metres; inner
    # corner 0 sits at the continuous edge margin + sq_px, i.e. u = that - 0.5.
    scale = cam.square_m / sq_px
    offset = -(margin + sq_px - 0.5) * scale
    tex_to_board = np.array([[scale, 0, offset], [0, scale, offset], [0, 0, 1]])
    rot, _ = cv2.Rodrigues(rvec)
    board_to_img = cam.k @ np.column_stack([rot[:, 0], rot[:, 1], tvec])
    w, h = cam.size
    ideal = cv2.warpPerspective(
        tex,
        board_to_img @ tex_to_board,
        (w, h),
        flags=cv2.INTER_AREA,
        borderValue=255,
    )
    return _distort(ideal, cam.k, cam.dist)


def _distort(
    ideal: np.ndarray,
    k: np.ndarray,
    dist: np.ndarray,
    *,
    interpolation: int = cv2.INTER_LINEAR,
    border_value: int = 255,
) -> np.ndarray:
    """Apply the lens distortion of (k, dist) (OpenCV convention) to an ideal
    pinhole image: every output pixel samples the ideal image where its ray
    lands. INTER_NEAREST keeps label images exact."""
    h, w = ideal.shape[:2]
    grid = np.stack(np.meshgrid(np.arange(w), np.arange(h)), -1).reshape(-1, 1, 2)
    undist = cv2.undistortPointsIter(
        grid.astype(np.float64),
        k,
        dist,
        None,
        k,
        (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-9),
    ).reshape(h, w, 2)
    return cv2.remap(
        ideal,
        undist[..., 0].astype(np.float32),
        undist[..., 1].astype(np.float32),
        interpolation,
        borderValue=border_value,
    )


@pytest.fixture
def render_board(synthetic_camera):
    """Factory: render_board(rvec, tvec) -> (H, W) uint8 distorted board image."""
    return lambda rvec, tvec: _render(synthetic_camera, rvec, tvec)


@pytest.fixture
def distort_image():
    """Factory: distort_image(ideal, k, dist, *, interpolation, border_value) ->
    the image seen through the lens (see `_distort`)."""
    return _distort


@pytest.fixture
def board_images(synthetic_camera, render_board, tmp_path):
    """Folder of 14 rendered board views (view_NN.png) of the synthetic camera."""
    folder = tmp_path / "views"
    folder.mkdir()
    for i, (rvec, tvec) in enumerate(synthetic_camera.poses(14, seed=5)):
        cv2.imwrite(str(folder / f"view_{i:02d}.png"), render_board(rvec, tvec))
    return folder


# metrics.json of a fake run: the keys evaluation.run_data reads, values of a
# 1920x1080 30 fps DA3METRIC run without calibration.
FAKE_METRICS: dict[str, Any] = {
    "video": {
        "path": "_testbed/data/IMG_4100.MOV",
        "width": 1920,
        "height": 1080,
        "fps": 30.0,
    },
    "frames_processed": 10,
    "components": {
        "detectors": [{"type": "rfdetr", "name": "rfdetr-large"}],
        "depth": {"type": "da3", "name": "da3metric-large", "kind": "metric_focal"},
        "aggregator": {"type": "concat", "options": {}},
        "tracker": None,
    },
    "focal_frame_px": 1400.0,
    "focal_source": "config (camera.focal_px)",
    "frame_intrinsics": {"fx": 1400.0, "fy": 1400.0, "cx": 960.0, "cy": 540.0},
    "camera": {},
    "config": {
        "source": {
            "video": "_testbed/data/IMG_4100.MOV",
            "start": 0.0,
            "duration": 0.0,
            "max_frames": 0,
            "stride": 1,
        },
        "distance": {"stat": "median", "distance_type": "z"},
    },
}


@pytest.fixture
def fake_run_dir(tmp_path):
    """Factory: fake_run_dir(rows, metrics=None) -> run folder. `rows` are CSV
    rows by column name (missing columns empty); `metrics` replaces top-level
    keys of FAKE_METRICS."""
    counter = itertools.count()

    def make(
        rows: Sequence[Mapping[str, Any]], metrics: Mapping[str, Any] | None = None
    ) -> Path:
        run = tmp_path / f"run{next(counter)}"
        run.mkdir()
        with open(run / "detections.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        (run / "metrics.json").write_text(
            json.dumps({**FAKE_METRICS, **(metrics or {})})
        )
        return run

    return make
