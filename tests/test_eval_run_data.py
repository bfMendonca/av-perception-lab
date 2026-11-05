import json

import numpy as np
import pytest

from av_perception_lab.camera import Calibration
from av_perception_lab.core.options import ConfigError
from av_perception_lab.core.types import CameraIntrinsics
from av_perception_lab.evaluation.run_data import ESTIMATORS, load_run

ROWS = [
    {"frame": 3, "t": 0.1, "class_name": "car", "x0": 1, "y0": 2, "x1": 3, "y1": 4,
     "distance_m": 12.5, "median": 12.5, "source": "rfdetr-large", "raw_label": "car",
     "track_id": 1, "distance_smooth_m": 12.4, "motion_model": "vehicle",
     "motion_speed_mps": 2.5},
    {"frame": 4, "t": 0.1333, "class_name": "person", "x0": 1, "y0": 2, "x1": 3, "y1": 4,
     "median": 30.0, "source": "rfdetr-large", "raw_label": "person", "track_id": 2},
]  # fmt: skip


def test_columns_and_properties(fake_run_dir):
    run = load_run(fake_run_dir(ROWS))
    assert len(run) == 2 and run.name == "run0"
    assert run.columns["frame"].dtype == np.float64
    assert np.isnan(run.columns["distance_m"][1])  # empty cell = n/a
    assert run.columns["class_name"].tolist() == ["car", "person"]
    # task_0009: the motion model is text, the motion numbers are floats.
    assert run.columns["motion_model"].tolist() == ["vehicle", ""]
    assert run.columns["motion_speed_mps"][0] == 2.5
    assert run.video_name == "IMG_4100.MOV" and run.distance_type == "z"
    assert run.depth_kind == "metric_focal" and run.focal_frame_px == 1400.0
    assert run.frame_intrinsics == CameraIntrinsics(1400.0, 1400.0, 960.0, 540.0)
    assert run.undistort_calibration is None and run.has_tracks
    assert run.sources == ("rfdetr-large",)
    # box_distance_m is in the header but empty (region "mask"): not available.
    assert run.available_estimators() == (
        "distance_m",
        "median",
        "distance_smooth_m",
    )
    assert set(ESTIMATORS) >= set(run.available_estimators())


def test_processed_frames_follow_the_source_settings(fake_run_dir):
    metrics = {
        "frames_processed": 4,
        "config": {
            "source": {"start": 2.0, "stride": 3, "max_frames": 4, "duration": 0.0},
            "distance": {"distance_type": "euclidean"},
        },
    }
    run = load_run(fake_run_dir(ROWS, metrics))
    frames = run.processed_frames()
    assert frames.tolist() == [60, 63, 66, 69]  # round(2.0 * 30) + 3k
    assert run.frame_times(frames) == pytest.approx([2.0, 2.1, 2.2, 2.3])
    assert run.distance_type == "euclidean"


def test_fusion_runs_form_one_source_group(fake_run_dir):
    components = {
        "detectors": [{"name": "rfdetr-large"}, {"name": "yolo26l"}],
        "depth": {"kind": "metric"},
        "aggregator": {"type": "wbf"},
    }
    run = load_run(fake_run_dir(ROWS, {"components": components}))
    assert run.sources == ("wbf",)
    assert run.row_sources().tolist() == ["wbf", "wbf"]
    concat = load_run(
        fake_run_dir(
            ROWS, {"components": {**components, "aggregator": {"type": "concat"}}}
        )
    )
    assert concat.sources == ("rfdetr-large", "yolo26l")


def test_undistorted_run(fake_run_dir):
    k = np.array([[1400.0, 0.0, 959.5], [0.0, 1401.0, 539.5], [0.0, 0.0, 1.0]])
    cal = Calibration.from_opencv(k, np.array([0.1, -0.2, 0.0, 0.0]), (1920, 1080))
    camera = {
        # As camera.build_camera records it: the path is informative only.
        "calibration": {"path": "configs/calibration/c.yaml", **cal.describe()},
        "undistort": {"alpha": 0.5},
    }
    run = load_run(fake_run_dir(ROWS, {"camera": camera, "frame_intrinsics": None}))
    assert run.is_undistorted
    rebuilt, alpha = run.undistort_calibration
    assert alpha == 0.5 and rebuilt.image_size == cal.image_size
    assert rebuilt.intrinsics == cal.intrinsics and rebuilt.dist == cal.dist
    assert run.frame_intrinsics is None
    broken = load_run(
        fake_run_dir(ROWS, {"camera": {**camera, "calibration": {"path": "c.yaml"}}})
    )
    with pytest.raises(ConfigError, match="camera.calibration/undistort unusable"):
        _ = broken.undistort_calibration


def test_mcap_run_uses_the_recorded_frame_times(fake_run_dir):
    # An MCAP source describes itself without "fps" (variable rate): the
    # nominal rate counts frames, and frame times come from the CSV `t`
    # (interpolated between frames with rows, extrapolated at the nominal rate).
    rows = [
        {**ROWS[0], "frame": 0, "t": 0.0},
        {**ROWS[1], "frame": 2, "t": 0.25},
        {**ROWS[0], "frame": 3, "t": 0.31},
    ]
    metrics = {
        "video": {
            "path": "/data/drive.mcap",
            "topic": "/cam/image_raw",
            "nominal_fps": 10.0,
            "time_base": "header",
        },
        "frames_processed": 5,
    }
    run = load_run(fake_run_dir(rows, metrics))
    assert run.video_name == "drive.mcap"
    frames = run.processed_frames()
    assert frames.tolist() == [0, 1, 2, 3, 4]
    assert run.frame_times(frames) == pytest.approx([0.0, 0.125, 0.25, 0.31, 0.41])


def test_mcap_frames_before_the_first_row(fake_run_dir):
    rows = [{**ROWS[0], "frame": 2, "t": 0.3}, {**ROWS[1], "frame": 3, "t": 0.42}]
    metrics = {
        "video": {"path": "/d.mcap", "nominal_fps": 10.0, "time_base": "log"},
        "frames_processed": 4,
    }
    run = load_run(fake_run_dir(rows, metrics))
    frames = run.processed_frames()
    assert run.frame_times(frames) == pytest.approx([0.1, 0.2, 0.3, 0.42])


def test_mcap_run_with_start_is_refused(fake_run_dir):
    config = {
        "source": {"start": 1.5, "stride": 1, "max_frames": 0, "duration": 0.0},
        "distance": {"distance_type": "z"},
    }
    metrics = {
        "video": {"path": "/d.mcap", "nominal_fps": 10.0, "time_base": "header"},
        "config": config,
    }
    run = fake_run_dir(ROWS, metrics)
    with pytest.raises(ConfigError, match="MCAP runs with source.start > 0") as exc:
        load_run(run)
    assert run.name in str(exc.value)


def test_load_errors(fake_run_dir, tmp_path):
    with pytest.raises(ConfigError, match="detections.csv not found"):
        load_run(tmp_path)
    run = fake_run_dir(ROWS)
    (run / "detections.csv").write_text("frame,t,class_name\n1,0.0,car\n")
    with pytest.raises(ConfigError, match="missing columns class_id, confidence"):
        load_run(run)
    old = fake_run_dir(ROWS)
    metrics = json.loads((old / "metrics.json").read_text())
    del metrics["components"]
    (old / "metrics.json").write_text(json.dumps(metrics))
    with pytest.raises(ConfigError, match="missing components.detectors"):
        load_run(old)
