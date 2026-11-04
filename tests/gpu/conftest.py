"""GPU test fixtures (run on the GPU machine: `uv run --extra infer pytest -m gpu -v`).

Every test under tests/gpu/ is marked `gpu` by the collection hook below, so
the local `uv run pytest` (addopts "-m 'not gpu'") deselects them without a
marker on each test. torch is checked inside an autouse fixture, not at
import: a module-level skip would prevent the hook from registering. Weights
and caches come from environment variables, with the default paths below;
missing assets skip the test that needs them (nothing is downloaded; RF-DETR
checkpoints are looked up under the weights root, `weights/rfdetr/`, task_0021).

Mask refiners and semantic segmenters (task_0005) read the project's HF cache
($AVPL_WEIGHTS_HF, default weights/hf); presence is checked with
huggingface_hub.try_to_load_from_cache, so a missing checkpoint skips with the
download command. REFINER_CASES lists every registered refiner type: a new
box-prompted model adds its case here to run the shared contract test.

End-to-end runs (the task_0001 regression, the task_0008 GPU smoke) go
through the `run_cli_subprocess` fixture, a fresh interpreter per run, so frameworks
imported by other tests of the session cannot influence them;
`require_assets` skips a run whose weights are missing, with the hint;
`require_ready` skips an example-config run until every model it declares is
ready under the weights root (run `--prepare` once on the host, task_0021).
"""

import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from av_perception_lab.config import SourceConfig
from av_perception_lab.core import registry
from av_perception_lab.core.components import LoadContext
from av_perception_lab.core.types import CameraIntrinsics
from av_perception_lab.io.source import VideoSource

GPU_DIR = Path(__file__).parent
REPO = GPU_DIR.parents[1]
VIDEO = REPO / "_testbed" / "data" / "IMG_4021.MOV"
YOLO_WEIGHTS = Path(
    os.environ.get("AVPL_YOLO_WEIGHTS", "~/dev/ultralytics/_testbed/weights")
).expanduser()
HF_CACHE = Path(
    os.environ.get("AVPL_HF_CACHE", "~/dev/depth-anything-3/_testbed/weights/hf")
).expanduser()
WEIGHTS_HF = Path(
    os.environ.get("AVPL_WEIGHTS_HF", str(REPO / "weights" / "hf"))
).expanduser()
ULTRALYTICS_SAM = YOLO_WEIGHTS / "mobile_sam.pt"
TWINLITENET_REPO = Path(
    os.environ.get("AVPL_TWINLITENET_REPO", "~/dev/TwinLiteNetPlus")
).expanduser()
SEGFORMER_REPO = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
DEVICE = "cuda:0"


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    # The hook receives the whole session's items, hence the path filter.
    for item in items:
        if GPU_DIR in item.path.parents:
            item.add_marker(pytest.mark.gpu)


@pytest.fixture(autouse=True, scope="session")
def _cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")


@pytest.fixture(scope="session")
def video():
    """The IMG_4021 test clip (not versioned; lives in _testbed/data)."""
    if not VIDEO.is_file():
        pytest.skip(f"missing test video {VIDEO}")
    return VIDEO


@pytest.fixture(scope="session")
def source(video):
    return VideoSource(SourceConfig(video=video, max_frames=1))


@pytest.fixture(scope="session")
def frame0(source):
    """Frame 0 of IMG_4021 (upright RGB)."""
    return next(iter(source))


@pytest.fixture
def load_ctx(source):
    """Frame-size context with a fixed 1390 px focal (DA3METRIC needs one)."""
    w, h = source.frame_size
    return LoadContext(DEVICE, (w, h), CameraIntrinsics.centered(1390.0, w, h))


def _yolo(weights: str) -> dict:
    # 0.25 is ultralytics' default conf; the nano models score frame 0 below 0.5.
    return {"type": "yolo", "weights": str(YOLO_WEIGHTS / weights), "threshold": 0.25}


# Keys are the detectors' default names.
DETECTOR_SPECS = {
    "rfdetr-nano": ({"type": "rfdetr", "variant": "nano"}, "coco"),
    "yolo26n": (_yolo("yolo26n.pt"), "coco"),
    "yolo26n-objv1-150": (_yolo("yolo26n-objv1-150.pt"), "objects365"),
    # Instance segmentation (task_0004). The objv1-seg vocabulary is expected to
    # be Objects365 like objv1-150; this entry is what confirms it.
    "rfdetr-seg-nano": ({"type": "rfdetr", "variant": "seg-nano"}, "coco"),
    "yolo26n-seg": (_yolo("yolo26n-seg.pt"), "coco"),
    "yolo26n-objv1-seg": (_yolo("yolo26n-objv1-seg.pt"), "objects365"),
}
SEGMENTERS = frozenset({"rfdetr-seg-nano", "yolo26n-seg", "yolo26n-objv1-seg"})


def _missing_weights(spec: dict) -> Path | None:
    """The weights file a spec needs when it is not on disk, else None (RF-DETR
    variants: their checkpoint under the weights root, fetched by --prepare)."""
    if "weights" in spec:
        path = Path(spec["weights"])
    elif spec.get("type") == "rfdetr":
        from av_perception_lab.adapters.detectors.rfdetr import checkpoint_path

        path = checkpoint_path(spec.get("variant", "large"))
    else:
        return None
    return None if path.is_file() else path


@pytest.fixture(params=list(DETECTOR_SPECS))
def detector_spec(request):
    """(detector spec, expected vocabulary) for each adapter/vocabulary pair."""
    spec, vocabulary = DETECTOR_SPECS[request.param]
    missing = _missing_weights(spec)
    if missing is not None:
        pytest.skip(f"missing weights {missing}")
    return spec, vocabulary


@pytest.fixture
def detector_case(detector_spec):
    """(unloaded detector, expected vocabulary) for each adapter/vocabulary pair."""
    spec, vocabulary = detector_spec
    return registry.build("detector", spec, path="detector"), vocabulary


@pytest.fixture(params=sorted(SEGMENTERS))
def segmenter(request):
    """Unloaded instance-segmentation detector (RF-DETR Seg, YOLO26-seg)."""
    spec, _ = DETECTOR_SPECS[request.param]
    missing = _missing_weights(spec)
    if missing is not None:
        pytest.skip(f"missing weights {missing}")
    return registry.build("detector", spec, path="detector")


@pytest.fixture(scope="session")
def segmenter_names():
    """Default names of the DETECTOR_SPECS entries that return masks."""
    return SEGMENTERS


@pytest.fixture(scope="session")
def yolo_weights():
    """Directory of the YOLO26 checkpoints ($AVPL_YOLO_WEIGHTS)."""
    return YOLO_WEIGHTS


@pytest.fixture(scope="session")
def hf_cache():
    """Hugging Face cache holding the DA3 checkpoints ($AVPL_HF_CACHE)."""
    if not HF_CACHE.is_dir():
        pytest.skip(f"missing HF cache {HF_CACHE}")
    return HF_CACHE


@pytest.fixture
def da3_metric(hf_cache):
    """DA3METRIC-LARGE from the local HF cache (not loaded)."""
    spec = {"type": "da3", "model": "da3metric-large", "hf_cache": str(hf_cache)}
    return registry.build("depth", spec, path="depth")


# Registry type -> candidate specs, the first one on disk is used (sam/sam_hq:
# huge if downloaded, else base). Every refiner type must have a case.
REFINER_CASES: dict[str, list[dict]] = {
    "sam2": [{"type": "sam2", "model": "facebook/sam2.1-hiera-large"}],
    "edgetam": [{"type": "edgetam", "model": "yonigozlan/EdgeTAM-hf"}],
    "sam_hq": [
        {"type": "sam_hq", "model": "syscv-community/sam-hq-vit-huge"},
        {"type": "sam_hq", "model": "syscv-community/sam-hq-vit-base"},
    ],
    "sam": [
        {"type": "sam", "model": "facebook/sam-vit-huge"},
        {"type": "sam", "model": "facebook/sam-vit-base"},
    ],
    "sam3_tracker": [{"type": "sam3_tracker", "model": "facebook/sam3"}],
    "ultralytics_sam": [{"type": "ultralytics_sam", "weights": str(ULTRALYTICS_SAM)}],
}


def _hf_cached(repo: str, filename: str) -> bool:
    from huggingface_hub import try_to_load_from_cache

    path = try_to_load_from_cache(repo, filename, cache_dir=str(WEIGHTS_HF))
    return isinstance(path, str)


def _available_spec(type_name: str) -> dict:
    """The first candidate of REFINER_CASES[type_name] on disk, else skip."""
    from av_perception_lab.adapters._hf_hub import download_command

    hints = []
    for spec in REFINER_CASES[type_name]:
        if "weights" in spec:
            if Path(spec["weights"]).is_file():
                return spec
            hints.append(f"missing {spec['weights']}")
        elif _hf_cached(spec["model"], "model.safetensors"):
            return {**spec, "hf_cache": str(WEIGHTS_HF)}
        else:
            hints.append(download_command(spec["model"], WEIGHTS_HF))
    pytest.skip("no weights for " + type_name + ": " + " | ".join(hints))


@pytest.fixture(scope="session")
def frame_ctx(source):
    """Session-wide LoadContext (module-scoped fixtures cannot use load_ctx)."""
    w, h = source.frame_size
    return LoadContext(DEVICE, (w, h), CameraIntrinsics.centered(1390.0, w, h))


@pytest.fixture(scope="module")
def rfdetr_boxes(frame0, frame_ctx):
    """RF-DETR nano instances (boxes only) of frame 0."""
    missing = _missing_weights({"type": "rfdetr", "variant": "nano"})
    if missing is not None:
        pytest.skip(f"missing weights {missing} (run --prepare)")
    detector = registry.build(
        "detector", {"type": "rfdetr", "variant": "nano"}, path="d"
    )
    detector.load(frame_ctx)
    try:
        return detector(frame0)
    finally:
        detector.close()


@pytest.fixture(scope="module", params=list(REFINER_CASES))
def refiner(request, frame_ctx):
    """Every refiner type in turn, loaded once per module (one model on the GPU
    at a time), built through the registry like the CLI does."""
    spec = _available_spec(request.param)
    component = registry.build("refiner", spec, path="refiners[0]")
    component.load(frame_ctx)
    yield component
    component.close()


@pytest.fixture(scope="session")
def weights_hf():
    """Project HF cache of the refiners and SegFormer ($AVPL_WEIGHTS_HF)."""
    return WEIGHTS_HF


@pytest.fixture
def segformer():
    """SegFormer b5 Cityscapes (not loaded)."""
    if not _hf_cached(SEGFORMER_REPO, "pytorch_model.bin"):
        from av_perception_lab.adapters._hf_hub import download_command

        pytest.skip(download_command(SEGFORMER_REPO, WEIGHTS_HF, "pytorch_model.bin"))
    spec = {
        "type": "segformer",
        "name": "scene",
        "model": SEGFORMER_REPO,
        "hf_cache": str(WEIGHTS_HF),
    }
    return registry.build("semantic", spec, path="semantic[0]")


@pytest.fixture
def twinlitenet():
    """TwinLiteNet+ large from the local clone ($AVPL_TWINLITENET_REPO; not loaded)."""
    if not (TWINLITENET_REPO / "pretrained" / "large.pth").is_file():
        pytest.skip(f"missing TwinLiteNetPlus clone/weights in {TWINLITENET_REPO}")
    spec = {
        "type": "twinlitenet",
        "name": "lanes",
        "repo": str(TWINLITENET_REPO),
        "variant": "large",
    }
    return registry.build("semantic", spec, path="semantic[0]")


def _run_cli_subprocess(
    argv: Sequence[str], *, cwd: Path, forbidden_modules: Sequence[str] = ()
) -> subprocess.CompletedProcess[str]:
    """Run `python -c main(argv)` in a fresh interpreter in `cwd`; assert exit
    code 0 (stderr tail in the message) and that none of `forbidden_modules`
    was imported."""
    guards = "".join(
        f"assert {m!r} not in sys.modules, '{m} imported by this run'\n"
        for m in forbidden_modules
    )
    code = (
        "import sys\n"
        "from av_perception_lab.cli import main\n"
        f"code = main({list(argv)!r})\n"
        f"{guards}"
        "sys.exit(code)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr[-3000:]
    return out


@pytest.fixture(scope="session")
def run_cli_subprocess() -> Callable[..., subprocess.CompletedProcess[str]]:
    """Factory: run_cli_subprocess(argv, *, cwd, forbidden_modules=()) (see
    `_run_cli_subprocess`). A fixture rather than an import: test modules
    cannot import a conftest reliably (several conftest.py share the name)."""
    return _run_cli_subprocess


@pytest.fixture
def require_assets() -> Callable[[Mapping[str, Any]], None]:
    """Factory that skips the test when an asset is missing: a detector spec
    (_missing_weights), a refiner type (_available_spec), an HF file
    (_hf_cached) or a local path; the skip reason carries the download hint.

    Usage: require_assets({"detector": spec} | {"refiner": "sam2"} |
    {"hf": (repo, filename)} | {"path": path}).
    """
    from av_perception_lab.adapters._hf_hub import download_command

    def require(asset: Mapping[str, Any]) -> None:
        if "detector" in asset:
            missing = _missing_weights(asset["detector"])
            if missing is not None:
                pytest.skip(f"missing weights {missing}")
        elif "refiner" in asset:
            _available_spec(asset["refiner"])
        elif "hf" in asset:
            repo, filename = asset["hf"]
            if not _hf_cached(repo, filename):
                pytest.skip(download_command(repo, WEIGHTS_HF, filename))
        elif "path" in asset:
            path = Path(asset["path"]).expanduser()
            if not path.exists():
                pytest.skip(f"missing {path}")
        else:
            raise ValueError(f"unknown asset kind: {dict(asset)}")

    return require


@pytest.fixture
def require_ready(monkeypatch) -> Callable[[str, Sequence[str]], None]:
    """Factory that skips the test until every model of an example config
    (with `--set` overrides) is ready under the weights root, naming
    `--prepare`: require_ready("configs/x.yaml", ("k=v", ...))."""
    from av_perception_lab.cli import config_requirements
    from av_perception_lab.config import load_config
    from av_perception_lab.core.requirements import check

    def require(config: str, overrides: Sequence[str] = ()) -> None:
        monkeypatch.chdir(REPO)
        cfg = load_config(Path(config), list(overrides))
        missing = []
        for path, name, _, req in config_requirements(cfg):
            status = check(req) if not isinstance(req, str) else None
            if status is None or status.state != "ready":
                reason = req if status is None else status.message
                missing.append(f"{path} {name}: {reason}")
        if missing:
            pytest.skip(
                f"not ready (run av-perception-lab --config {config} --prepare): "
                + "; ".join(missing)
            )

    return require
