"""Camera intrinsics providers (frame-resolution pinhole K).

The focal length belongs to the camera, not to the depth model: providers
return intrinsics at frame resolution and depth adapters rescale them to
their map with `MapGeometry.scale_intrinsics`. Depth models that predict K
per frame declare `needs_intrinsics == False` and need no provider.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from det_depth_distance.adapters.depth.da3 import estimate_focal_frame_px
from det_depth_distance.config import CameraConfig, EstimateConfig
from det_depth_distance.core.options import ConfigError
from det_depth_distance.core.types import CameraIntrinsics
from det_depth_distance.io.source import FrameSource


@dataclass(frozen=True, slots=True)
class IntrinsicsResult:
    """Resolved intrinsics and where they came from."""

    intrinsics: CameraIntrinsics  # frame resolution
    source: str  # metrics.json["focal_source"]


class IntrinsicsProvider(ABC):
    """Resolves the camera intrinsics of a frame source."""

    @abstractmethod
    def resolve(self, source: FrameSource, *, device: str) -> IntrinsicsResult: ...


class FixedIntrinsics(IntrinsicsProvider):
    """A known focal length, principal point at the frame centre."""

    def __init__(self, focal_px: float) -> None:
        if focal_px <= 0:
            raise ConfigError(f"camera.focal_px must be > 0, got {focal_px}")
        self.focal_px = focal_px

    def resolve(self, source: FrameSource, *, device: str) -> IntrinsicsResult:
        return IntrinsicsResult(
            CameraIntrinsics.centered(self.focal_px, *source.frame_size),
            "config (camera.focal_px)",
        )


class EstimatedIntrinsics(IntrinsicsProvider):
    """Focal estimated by an any-view DA3 model over frames sampled from the source.

    The estimation model is loaded, used and released inside `resolve`.
    """

    def __init__(self, cfg: EstimateConfig) -> None:
        self.cfg = cfg

    def resolve(self, source: FrameSource, *, device: str) -> IntrinsicsResult:
        frames = source.sample(self.cfg.frames)
        focal = estimate_focal_frame_px(
            frames,
            self.cfg.model,
            device=device,
            process_res=self.cfg.process_res,
            process_res_method=self.cfg.process_res_method,
            cache_dir=str(self.cfg.hf_cache) if self.cfg.hf_cache else None,
        )
        return IntrinsicsResult(
            CameraIntrinsics.centered(focal, *source.frame_size),
            f"estimated by {self.cfg.model} (median of {len(frames)} frames)",
        )


def build_intrinsics_provider(cfg: CameraConfig) -> IntrinsicsProvider | None:
    """`camera.focal_px` takes precedence over `camera.estimate`; None if neither is set."""
    if cfg.focal_px is not None:
        return FixedIntrinsics(cfg.focal_px)
    if cfg.estimate is not None:
        return EstimatedIntrinsics(cfg.estimate)
    return None
