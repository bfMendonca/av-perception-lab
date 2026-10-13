"""GPU test fixtures (run on gpu-host: `uv run --extra infer pytest -m gpu -v`).

Every test under tests/gpu/ is marked `gpu` by the collection hook below, so
the local `uv run pytest` (addopts "-m 'not gpu'") deselects them without a
marker on each test. torch is checked inside an autouse fixture, not at
import: a module-level skip would prevent the hook from registering. Weights
and caches come from environment variables, with gpu-host's paths as defaults;
missing assets skip the test that needs them (nothing is downloaded).
"""

import os
from pathlib import Path

import pytest

from det_depth_distance.config import SourceConfig
from det_depth_distance.core import registry
from det_depth_distance.core.components import LoadContext
from det_depth_distance.core.types import CameraIntrinsics
from det_depth_distance.io.source import VideoSource

GPU_DIR = Path(__file__).parent
REPO = GPU_DIR.parents[1]
VIDEO = REPO / "_testbed" / "data" / "IMG_4021.MOV"
YOLO_WEIGHTS = Path(
    os.environ.get("DDD_YOLO_WEIGHTS", "~/dev/ultralytics/_testbed/weights")
).expanduser()
HF_CACHE = Path(
    os.environ.get("DDD_HF_CACHE", "~/dev/depth-anything-3/_testbed/weights/hf")
).expanduser()
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


DETECTOR_SPECS = {
    "rfdetr-nano": ({"type": "rfdetr", "variant": "nano"}, "coco"),
    "yolo26n": (_yolo("yolo26n.pt"), "coco"),
    "yolo26n-objv1-150": (_yolo("yolo26n-objv1-150.pt"), "objects365"),
}


@pytest.fixture(params=list(DETECTOR_SPECS))
def detector_case(request):
    """(unloaded detector, expected vocabulary) for each adapter/vocabulary pair."""
    spec, vocabulary = DETECTOR_SPECS[request.param]
    if "weights" in spec and not Path(spec["weights"]).is_file():
        pytest.skip(f"missing weights {spec['weights']}")
    return registry.build("detector", spec, path="detector"), vocabulary


@pytest.fixture(scope="session")
def yolo_weights():
    """Directory of the YOLO26 checkpoints ($DDD_YOLO_WEIGHTS)."""
    return YOLO_WEIGHTS


@pytest.fixture(scope="session")
def hf_cache():
    """Hugging Face cache holding the DA3 checkpoints ($DDD_HF_CACHE)."""
    if not HF_CACHE.is_dir():
        pytest.skip(f"missing HF cache {HF_CACHE}")
    return HF_CACHE


@pytest.fixture
def da3_metric(hf_cache):
    """DA3METRIC-LARGE from the local HF cache (not loaded)."""
    spec = {"type": "da3", "model": "da3metric-large", "hf_cache": str(hf_cache)}
    return registry.build("depth", spec, path="depth")
