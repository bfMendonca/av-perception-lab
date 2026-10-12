import numpy as np
import pytest

from det_depth_distance.cli import (
    build_parser,
    latency_stats,
    main,
    make_run_dir,
    process_frame,
)
from det_depth_distance.depth import DepthFrame
from det_depth_distance.detector import Detections
from det_depth_distance.distance import CameraIntrinsics, DistanceConfig, MapGeometry


class FakeDetector:
    def __call__(self, rgb):
        return Detections(
            np.array([[0, 0, 100, 100], [0, 0, 20, 100]], dtype=np.float32),
            np.array([0.9, 0.6], dtype=np.float32),
            np.array([3, 1]),
            ("car", "person"),
        )


class FakeDepth:
    geometry = MapGeometry(100, 100, 100, 100, 1.0, 1.0)

    def __call__(self, rgb):
        depth = np.full((100, 100), 40.0, dtype=np.float32)
        depth[:, 20:80] = 10.0
        sky = np.zeros((100, 100), dtype=bool)
        sky[:, :20] = True  # the "person" box is all sky
        return DepthFrame(depth, CameraIntrinsics(50.0, 50.0, 50.0, 50.0), sky)


def test_process_frame_excludes_sky():
    cfg = DistanceConfig(box_shrink=0.0)
    frame = np.zeros((100, 100, 3), np.uint8)
    result = process_frame(frame, FakeDetector(), FakeDepth(), cfg, exclude_sky=True)
    assert len(result.stats) == 2
    assert result.stats[0].distance == pytest.approx(10.0)
    assert result.stats[1].distance is None
    assert set(result.timings_ms) == {"det", "depth", "dist"}

    result = process_frame(frame, FakeDetector(), FakeDepth(), cfg, exclude_sky=False)
    assert result.stats[1].distance == pytest.approx(40.0)


def test_parser_defaults_and_validation():
    args = build_parser().parse_args(["clip.mov"])
    assert (args.det_model, args.depth_model, args.distance_stat) == (
        "large",
        "da3metric-large",
        "median",
    )
    assert args.exclude_sky and args.layout == "overlay" and args.max_depth == 30.0
    assert (
        build_parser()
        .parse_args(["clip.mov", "--no-exclude-sky", "--distance-stat", "p30"])
        .distance_stat
        == "p30"
    )
    with pytest.raises(SystemExit):
        build_parser().parse_args(["clip.mov", "--distance-stat", "p100"])
    with pytest.raises(SystemExit):
        build_parser().parse_args(["clip.mov", "--det-model", "base"])


@pytest.mark.parametrize(
    "bad",
    [
        ["--stride", "0"],
        ["--max-frames", "-1"],
        ["--start", "-1"],
        ["--focal-px", "-1400"],
        ["--focal-frames", "0"],
        ["--process-res", "10"],
        ["--max-depth", "0"],
    ],
)
def test_main_rejects_invalid_numbers_before_loading_models(tmp_path, bad):
    video = tmp_path / "clip.mov"
    video.touch()
    with pytest.raises(SystemExit) as exc:
        main([str(video), *bad])
    assert exc.value.code == 2


def test_latency_stats():
    assert latency_stats([]) == {}
    s = latency_stats([1.0, 2.0, 3.0])
    assert s["mean"] == 2.0 and s["min"] == 1.0 and s["max"] == 3.0


def test_make_run_dir(tmp_path):
    run_dir = make_run_dir(
        tmp_path / "out", ["IMG_4021", "rfdetr-large", "", "da3metric-large"]
    )
    assert run_dir.is_dir()
    assert run_dir.name.endswith("_IMG_4021_rfdetr-large_da3metric-large")
