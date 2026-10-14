import subprocess
import sys

import pytest
import yaml

from det_depth_distance import cli
from det_depth_distance.cli import build_parser, main


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
