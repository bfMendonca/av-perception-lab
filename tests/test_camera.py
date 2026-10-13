import numpy as np
import pytest

from det_depth_distance import camera
from det_depth_distance.camera import (
    EstimatedIntrinsics,
    FixedIntrinsics,
    build_intrinsics_provider,
)
from det_depth_distance.config import CameraConfig, EstimateConfig
from det_depth_distance.core.options import ConfigError
from det_depth_distance.core.types import CameraIntrinsics
from det_depth_distance.io.source import FrameSource


class StillSource(FrameSource):
    frame_size = (1920, 1080)
    name = "still"

    def __iter__(self):
        return iter(())

    def sample(self, n):
        return [np.zeros((1080, 1920, 3), np.uint8)] * n

    def describe(self):
        return {}


def test_fixed():
    result = FixedIntrinsics(1400.0).resolve(StillSource(), device="cpu")
    assert result.intrinsics == CameraIntrinsics(1400.0, 1400.0, 960.0, 540.0)
    assert result.source == "config (camera.focal_px)"
    with pytest.raises(ConfigError):
        FixedIntrinsics(0.0)


def test_estimated(monkeypatch):
    calls = {}

    def fake_estimate(frames, model, **kwargs):
        calls.update(n=len(frames), model=model, **kwargs)
        return 1390.5

    monkeypatch.setattr(camera, "estimate_focal_frame_px", fake_estimate)
    cfg = EstimateConfig(frames=4, process_res=280)
    result = EstimatedIntrinsics(cfg).resolve(StillSource(), device="cuda:1")
    assert result.intrinsics == CameraIntrinsics.centered(1390.5, 1920, 1080)
    assert result.source == "estimated by da3-large-1.1 (median of 4 frames)"
    assert calls == {
        "n": 4,
        "model": "da3-large-1.1",
        "device": "cuda:1",
        "process_res": 280,
        "process_res_method": "upper_bound_resize",
        "cache_dir": None,
    }


def test_precedence():
    assert isinstance(
        build_intrinsics_provider(CameraConfig(focal_px=1000.0)), FixedIntrinsics
    )
    assert isinstance(build_intrinsics_provider(CameraConfig()), EstimatedIntrinsics)
    assert build_intrinsics_provider(CameraConfig(estimate=None)) is None
