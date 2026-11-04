import subprocess
import sys

import numpy as np
import pytest
import yaml

from av_perception_lab import cli
from av_perception_lab.cli import build_parser, main
from av_perception_lab.io import source as source_module
from av_perception_lab.io.video import VideoInfo


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
        "import sys; from av_perception_lab.cli import main\n"
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

    from av_perception_lab.config import RunConfig, config_from_mapping
    from av_perception_lab.core.types import CameraIntrinsics

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


# --------------------------------------------------------------- --prepare (task_0020)


def _prepare_config(tmp_path, detector_type: str):
    path = tmp_path / "prep.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                # The video is never probed by --prepare: it does not exist.
                "source": {"video": str(tmp_path / "absent.mov")},
                "detectors": [{"type": detector_type, "name": "fake"}],
                "depth": {"type": "da3"},
            }
        )
    )
    return path


def _fake_detector(asset):
    from av_perception_lab.core.components import Detector, DetectorOptions
    from av_perception_lab.core.requirements import Requirements

    class _NeedsWeights(Detector):
        Options = DetectorOptions

        @classmethod
        def requirements(cls, options):
            return Requirements(assets=(asset,))

        @property
        def class_names(self):
            return ["car"]

        def _load(self, ctx):
            raise AssertionError("a model was loaded by --prepare")

        def _predict(self, rgb):
            raise AssertionError("unreachable")

    _NeedsWeights.type_name = "needs_weights"
    return _NeedsWeights


def test_cli_prepare_exit_code_and_table(tmp_path, monkeypatch, capsys):
    from av_perception_lab.core import registry
    from av_perception_lab.core.provision import sources
    from av_perception_lab.core.requirements import UrlAsset

    def no_models(*args, **kwargs):
        raise AssertionError("a model was about to be loaded")

    monkeypatch.setattr(cli.Pipeline, "load", no_models)
    monkeypatch.setattr(cli, "run", no_models)
    root = tmp_path / "weights"
    monkeypatch.setenv("AVPL_WEIGHTS_ROOT", str(root))
    registry.load_adapters()
    asset = UrlAsset(
        key="ckpt", target="fake/ckpt.bin", url="https://example.invalid/c"
    )
    monkeypatch.setitem(
        registry._REGISTRY["detector"], "needs_weights", _fake_detector(asset)
    )
    cfg = _prepare_config(tmp_path, "needs_weights")

    # Offline download: failed, exit code 1, readable reason in the table notes.
    def offline(*args, **kwargs):
        raise sources.urllib.error.URLError("no network")

    def hf_offline(self, asset, root, progress, cancel):
        raise sources.ProvisionError(f"offline: cannot download {asset.repo}")

    monkeypatch.setattr(sources, "_open", offline)
    monkeypatch.setattr(sources.HuggingFaceSource, "prepare", hf_offline)
    # Packages are reported, not installed: pretend the infer stack is there.
    monkeypatch.setattr("av_perception_lab.core.requirements._top_level_present", bool)
    assert main(["--config", str(cfg), "--prepare"]) == 1
    out = capsys.readouterr().out
    assert "[prepare] detectors[0] fake (needs_weights)" in out
    assert "component" in out and "packages" in out
    table = {line.split()[0]: line.split() for line in out.splitlines() if line}
    assert table["detectors[0]"][2:4] == ["needs_weights", "failed"]
    # DA3 and the focal model of camera.estimate declare their HF checkpoints.
    assert table["depth"][2:4] == ["da3", "failed"]
    assert table["camera.estimate"][1:4] == ["da3-large-1.1", "da3", "failed"]
    assert "offline or unreachable" in out
    assert "offline: cannot download depth-anything/DA3METRIC-LARGE" in out

    # Once the files are there every component is ready: exit code 0, no download.
    (root / "fake").mkdir(parents=True, exist_ok=True)
    (root / "fake/ckpt.bin").write_bytes(b"x")
    for repo in ("depth-anything/DA3METRIC-LARGE", "depth-anything/DA3-LARGE-1.1"):
        snap = root / "hf" / f"models--{repo.replace('/', '--')}" / "snapshots/abc"
        snap.mkdir(parents=True)
        (snap / "config.json").write_text("{}")
        with (snap / "model.safetensors").open("wb") as f:
            f.truncate(2_000_000_000)  # sparse: above the declared size
    assert main(["--config", str(cfg), "--prepare"]) == 0
    out = capsys.readouterr().out
    assert f"weights root: {root}" in out
    table = {line.split()[0]: line.split() for line in out.splitlines() if line}
    assert table["detectors[0]"][2:6] == ["needs_weights", "ready", "ready", "ready"]


def test_cli_prepare_config_errors_exit_with_code_2(tmp_path, capsys):
    cfg = _prepare_config(tmp_path, "no_such_detector")
    assert main(["--config", str(cfg), "--prepare"]) == 2
    assert "unknown detector 'no_such_detector'" in capsys.readouterr().err


def test_cli_existing_invocations_unchanged(tmp_path, config, monkeypatch):
    args = build_parser().parse_args(["--config", "run.yaml"])
    assert args.prepare is False
    calls = []
    monkeypatch.setattr(cli, "run", lambda cfg: calls.append(cfg))
    monkeypatch.setattr(
        cli, "prepare_command", lambda cfg: pytest.fail("prepare without --prepare")
    )
    assert main(["--config", str(config)]) == 0
    assert len(calls) == 1 and calls[0].depth["type"] == "da3"
    assert main(["--list-models"]) == 0


def test_cli_prepare_ctrl_c_exits_130_and_keeps_part(tmp_path, monkeypatch, capsys):
    from av_perception_lab.core import registry
    from av_perception_lab.core.provision import sources
    from av_perception_lab.core.requirements import UrlAsset

    root = tmp_path / "weights"
    monkeypatch.setenv("AVPL_WEIGHTS_ROOT", str(root))
    registry.load_adapters()
    asset = UrlAsset(
        key="ckpt", target="fake/ckpt.bin", url="https://example.invalid/c"
    )
    monkeypatch.setitem(
        registry._REGISTRY["detector"], "needs_weights", _fake_detector(asset)
    )
    part = root / "fake/ckpt.bin.part"

    def interrupted(self, asset, root, progress, cancel):
        part.parent.mkdir(parents=True, exist_ok=True)
        part.write_bytes(b"half")
        raise KeyboardInterrupt

    monkeypatch.setattr(sources.UrlSource, "prepare", interrupted)
    cfg = _prepare_config(tmp_path, "needs_weights")
    assert main(["--config", str(cfg), "--prepare"]) == 130
    err = capsys.readouterr().err
    assert "cancelled" in err and "Traceback" not in err
    assert part.read_bytes() == b"half"


# ------------------------------------------------- run gate (task_0022 fix round 1)


def _gated_config(tmp_path, monkeypatch):
    """A config whose only declared requirement is the fake detector's file:
    DA3 points its cache outside the weights root (nothing declared) and the
    focal is given; packages count as installed."""
    from av_perception_lab.core import registry
    from av_perception_lab.core.requirements import UrlAsset

    root = tmp_path / "weights"
    monkeypatch.setenv("AVPL_WEIGHTS_ROOT", str(root))
    monkeypatch.setattr("av_perception_lab.core.requirements._top_level_present", bool)
    registry.load_adapters()
    asset = UrlAsset(
        key="ckpt", target="fake/ckpt.bin", url="https://example.invalid/c"
    )
    monkeypatch.setitem(
        registry._REGISTRY["detector"], "needs_weights", _fake_detector(asset)
    )
    monkeypatch.setattr(
        source_module,
        "probe",
        lambda path: VideoInfo(path, 64, 48, 10.0, "10/1", 10, False, 0),
    )
    video = tmp_path / "clip.mov"
    video.touch()
    path = tmp_path / "gated.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "source": {"video": str(video)},
                "detectors": [{"type": "needs_weights", "name": "fake"}],
                "depth": {"type": "da3", "hf_cache": str(tmp_path / "elsewhere")},
                "camera": {"focal_px": 100.0, "estimate": None},
                "output": {"out_root": str(tmp_path / "out")},
            }
        )
    )
    return root, path


def test_run_refuses_not_ready_models_before_any_load(tmp_path, monkeypatch, capsys):
    root, path = _gated_config(tmp_path, monkeypatch)
    loads = []
    monkeypatch.setattr(cli.Pipeline, "load", lambda *a, **k: loads.append(a))
    monkeypatch.setattr(
        cli.Pipeline, "resolve_intrinsics", lambda *a, **k: loads.append(a)
    )
    assert main(["--config", str(path)]) == 2
    err = capsys.readouterr().err
    assert loads == []  # stopped before any model was loaded
    assert "1 model(s) not ready" in err
    assert "detectors[0] fake (needs_weights): ckpt: fake/ckpt.bin not found" in err
    assert "--prepare" in err
    assert str(root) not in err  # root-relative
    assert not (tmp_path / "out").exists()  # no run folder either


def test_run_gate_passes_once_prepared(tmp_path, monkeypatch, capsys):
    from av_perception_lab.config import load_config

    root, path = _gated_config(tmp_path, monkeypatch)
    cfg = load_config(path, [], None)
    with pytest.raises(cli.ConfigError, match="--prepare"):
        cli.require_ready(cfg)
    (root / "fake").mkdir(parents=True)
    (root / "fake/ckpt.bin").write_bytes(b"x")
    cli.require_ready(cfg)  # ready: no error
    # An explicit DA3 cache outside the root is not tracked (ready by design),
    # and --prepare itself is not gated.
    assert main(["--config", str(path), "--prepare"]) == 0
    assert "needs_weights  ready" in capsys.readouterr().out


@pytest.mark.parametrize("bad", ["/", "~", "parent-of-out"])
def test_cli_refuses_an_unsafe_weights_root(tmp_path, monkeypatch, capsys, bad):
    cfg = _prepare_config(tmp_path, "rfdetr")
    root = {"/": "/", "~": "~", "parent-of-out": str(tmp_path)}[bad]
    monkeypatch.setenv("AVPL_WEIGHTS_ROOT", root)
    over = ["--set", f"output.out_root={tmp_path / 'outputs'}"]
    monkeypatch.setattr(cli, "run", lambda cfg: pytest.fail("ran"))
    monkeypatch.setattr(cli, "prepare_command", lambda *a, **k: pytest.fail("ran"))
    assert main(["--config", str(cfg), "--prepare", *over]) == 2
    err = capsys.readouterr().err
    assert "weights root" in err and "use a dedicated folder" in err


def test_cli_prepare_reports_a_user_provided_file_without_failing(
    tmp_path, monkeypatch, capsys
):
    from av_perception_lab.core import registry
    from av_perception_lab.core.requirements import UserProvided

    root = tmp_path / "weights"
    monkeypatch.setenv("AVPL_WEIGHTS_ROOT", str(root))
    registry.load_adapters()
    asset = UserProvided(
        key="weights", target="m/large.pth", option="weights", hint="get large.pth"
    )
    monkeypatch.setitem(
        registry._REGISTRY["detector"], "needs_weights", _fake_detector(asset)
    )
    monkeypatch.setattr("av_perception_lab.core.requirements._top_level_present", bool)
    cfg = _prepare_config(tmp_path, "needs_weights")
    data = yaml.safe_load(cfg.read_text())
    data["depth"]["hf_cache"] = str(tmp_path / "elsewhere")
    data["camera"] = {"estimate": None}
    cfg.write_text(yaml.safe_dump(data))
    assert main(["--config", str(cfg), "--prepare"]) == 1  # not ready: non-zero
    out = capsys.readouterr().out
    table = {line.split()[0]: line.split() for line in out.splitlines() if line}
    assert table["detectors[0]"][2:4] == ["needs_weights", "missing"]
    assert "provide this file: get large.pth" in out
