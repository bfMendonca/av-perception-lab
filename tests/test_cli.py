import subprocess
import sys

import numpy as np
import pytest
import yaml

from det_depth_distance import cli
from det_depth_distance.cli import build_parser, main
from det_depth_distance.io import source as source_module
from det_depth_distance.io.video import VideoInfo


@pytest.fixture
def config(tmp_path):
    video = tmp_path / "clip.mov"
    video.touch()
    path = tmp_path / "run.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "source": {"video": str(video)},
                "detectors": [{"type": "rfdetr"}],
                "depth": {"type": "da3"},
            }
        )
    )
    return path


def test_parser():
    args = build_parser().parse_args(
        ["clip.mov", "--config", "run.yaml", "--set", "a=1", "--set", "b.c=2"]
    )
    assert str(args.video) == "clip.mov" and str(args.config) == "run.yaml"
    assert args.overrides == ["a=1", "b.c=2"] and not args.list_models
    assert build_parser().parse_args(["--list-models"]).video is None
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--det-model", "large"])  # old flags are gone


def test_list_models(capsys):
    assert main(["--list-models"]) == 0
    out = capsys.readouterr().out
    for name in (
        "rfdetr:",
        "yolo:",
        "da3:",
        "concat:",
        "nms: iou_thr=0.5, class_agnostic=False, weights={}",
        "wbf: iou_thr=0.55, class_agnostic=False, weights={}, conf_type='avg'",
        "weights (required)",
        "variant='large'",
        "refiner:",
        "sam2: targets='missing', max_boxes=32",
        "model='facebook/sam2.1-hiera-large'",
        "ultralytics_sam:",
        "semantic:",
        "segformer:",
        "twinlitenet:",
        "tracker:",
        (
            "iou: iou_thr=0.3, max_age_s=0.5, motion='constant_velocity', "
            "velocity_alpha=0.5, class_aware=False"
        ),
        "motion:",
        "cv: init_speed_std=5.0, min_speed_for_heading=1.0, accel_std=2.0",
        "ctrv: ",
        "warmup_s=0.5",
        "ego_motion:",
        "csv: path (required), time_offset_s=0.0",
    ):
        assert name in out


def test_config_is_required():
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2


def test_configuration_errors_exit_with_code_2(tmp_path, config, capsys, monkeypatch):
    def no_models(*args, **kwargs):
        raise AssertionError("a model was about to be loaded")

    monkeypatch.setattr(cli.Pipeline, "setup", no_models)
    monkeypatch.setattr(cli.Pipeline, "load", no_models)
    # The camera is composed from the probed video before the Pipeline is built;
    # the touched clip.mov is not a video, so the probe is faked.
    monkeypatch.setattr(
        source_module,
        "probe",
        lambda path: VideoInfo(path, 64, 48, 10.0, "10/1", 10, False, 0),
    )
    assert main(["--config", str(tmp_path / "nope.yaml")]) == 2
    assert main(["--config", str(config), str(tmp_path / "missing.mov")]) == 2
    assert "video not found" in capsys.readouterr().err
    # Ported --process-res 10: validated by DA3Depth.Options, before any load.
    assert main(["--config", str(config), "--set", "depth.process_res=10"]) == 2
    assert "depth: process_res must be >= 14" in capsys.readouterr().err
    assert main(["--config", str(config), "--set", "source.stride=0"]) == 2
    # task_0005: refiner, segmenter and exclusion errors, all before any load.
    for override, message in [
        ("refiners=[{type: sam_hq, score: object}]", "refiners[0].score: sam_hq has"),
        (
            "refiners=[{type: sam2}, {type: sam2}]",
            "duplicate name 'sam2.1-hiera-large'",
        ),
        ("semantic=[{type: segformer}]", "semantic[0].name is required"),
        ("semantic=[{name: s, type: segformer, size: [1, 1]}]", "semantic[0]: size"),
        ("distance.exclude_semantic=[scene.road]", "no semantic segmenter configured"),
        # task_0006: tracking and smoothing errors, before any load.
        ("smoothing.filter=kalman", "smoothing: needs tracking"),
        ("tracking={type: iou, iou_thr: 1.5}", "tracking: iou_thr must be in (0, 1)"),
        ("smoothing.window_s=0", "smoothing: window_s must be > 0"),
        # task_0007: calibration errors, before any load.
        ("camera.calibration={path: nope.yaml}", "calibration file not found"),
        (
            "camera.calibration={path: c.yaml, alpha: 2}",
            "camera.calibration: alpha must be in [0, 1]",
        ),
    ]:
        assert main(["--config", str(config), "--set", override]) == 2
        assert message in capsys.readouterr().err


def test_help_without_torch():
    code = (
        "import sys; from det_depth_distance.cli import main\n"
        "try:\n    main(['--help'])\nexcept SystemExit:\n    pass\n"
        "assert 'torch' not in sys.modules"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr
    assert "--config" in out.stdout


def test_run_calls_on_frame_per_frame(tmp_path, monkeypatch):
    """on_frame sees every composed frame, indices 0..n-1 (fake scene adapters)."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent / "smoke"))
    from smoke_fakes import DepthOracle, install_scene_fakes
    from smoke_scene import SCENE, scene_frames, write_scene_video

    from det_depth_distance.config import RunConfig, config_from_mapping
    from det_depth_distance.core.types import CameraIntrinsics

    focal = 725.0 / 3
    install_scene_fakes(
        monkeypatch, DepthOracle(SCENE, CameraIntrinsics(focal, focal, 160.0, 90.0))
    )
    video = write_scene_video(SCENE, tmp_path / "scene.mp4", scene_frames(SCENE))
    cfg: RunConfig = config_from_mapping(
        {
            "device": "cpu",
            "source": {"video": str(video)},
            "detectors": [{"type": "scene_det", "name": "box"}],
            "depth": {"type": "da3", "model": "da3metric-large", "process_res": 154},
            "camera": {"focal_px": focal, "estimate": None},
            "output": {"out_root": str(tmp_path / "out")},
        }
    )
    seen = []
    run_dir = cli.run(cfg, lambda i, image: seen.append((i, image.shape, image.dtype)))
    assert [i for i, _, _ in seen] == list(range(SCENE.n_frames))
    assert {(s, d) for _, s, d in seen} == {((180, 320, 3), np.dtype("uint8"))}
    assert (run_dir / "video.mp4").is_file()
