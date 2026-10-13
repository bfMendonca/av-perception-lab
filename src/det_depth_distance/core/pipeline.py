"""Model-agnostic per-frame pipeline: detectors -> aggregation -> depth -> distance.

Depends only on the abstract components; timings are measured here around
each stage so components stay stateless per call. `describe()` keeps
component descriptions ("components") apart from pipeline policy
("pipeline"). Instance masks are carried but not yet used for distance
(boxes only).

Import direction: this module may import configuration dataclasses from
`config`, `temporal`, `camera` and `io.source`; those modules never import
`core.pipeline`. `core.types` and `core.components` stay contract-only.
"""

import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Self, TypeVar

from det_depth_distance.camera import IntrinsicsProvider
from det_depth_distance.core.aggregate import Aggregator
from det_depth_distance.core.components import DepthEstimator, Detector, LoadContext
from det_depth_distance.core.options import ConfigError
from det_depth_distance.core.types import (
    CameraIntrinsics,
    DepthResult,
    Frame,
    FrameResult,
    Instances,
)
from det_depth_distance.distance import BoxDepthStats, DistanceConfig, box_depth_stats
from det_depth_distance.io.source import FrameSource

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class SetupInfo:
    """Camera facts resolved before the models are loaded."""

    frame_intrinsics: CameraIntrinsics | None  # frame resolution
    focal_frame_px: float | None
    focal_source: str


def _timed(fn: Callable[..., T], *args: Any) -> tuple[T, float]:
    """(fn(*args), elapsed ms). Models return numpy arrays, so each call ends
    with a device sync and wall-clock time is the stage's GPU time."""
    t0 = time.perf_counter()
    out = fn(*args)
    return out, (time.perf_counter() - t0) * 1e3


class Pipeline:
    """Runs N detectors, the aggregator, the depth model and the distance
    statistics on each frame; orchestrates, never decodes or writes."""

    def __init__(
        self,
        detectors: Sequence[Detector],
        depth: DepthEstimator,
        aggregator: Aggregator,
        distance: DistanceConfig,
        *,
        exclude_sky: bool = True,
        intrinsics: IntrinsicsProvider | None = None,
        device: str = "cuda:0",
    ) -> None:
        """Raises ConfigError for no detectors or duplicate detector names."""
        if not detectors:
            raise ConfigError("detectors: at least one detector is required")
        duplicates = [n for n, c in Counter(d.name for d in detectors).items() if c > 1]
        if duplicates:
            raise ConfigError(
                f"detectors: duplicate name {duplicates[0]!r}; give each a distinct 'name'"
            )
        self.detectors = tuple(detectors)
        self.depth = depth
        self.aggregator = aggregator
        self.distance = distance
        self.exclude_sky = exclude_sky
        self.intrinsics = intrinsics
        self.device = device
        aggregator.bind_sources([d.name for d in self.detectors])

    def resolve_intrinsics(self, source: FrameSource) -> SetupInfo:
        """Frame intrinsics, only when the depth model needs them (may run and
        release a focal-estimation model).

        Raises:
            ConfigError: the depth model needs intrinsics and no provider is configured.
        """
        if not self.depth.needs_intrinsics:
            return SetupInfo(None, None, "predicted per frame by the depth model")
        if self.intrinsics is None:
            raise ConfigError(
                f"{self.depth.name} needs camera intrinsics: "
                "set camera.focal_px or camera.estimate"
            )
        resolved = self.intrinsics.resolve(source, device=self.device)
        return SetupInfo(resolved.intrinsics, resolved.intrinsics.fx, resolved.source)

    def load(self, source: FrameSource, info: SetupInfo) -> None:
        """Load every detector, then the depth model, for `source`'s frame size."""
        ctx = LoadContext(self.device, source.frame_size, info.frame_intrinsics)
        for detector in self.detectors:
            detector.load(ctx)
        self.depth.load(ctx)

    def setup(self, source: FrameSource) -> SetupInfo:
        """Resolve camera intrinsics if the depth model needs them, then load all models.

        Intrinsics come first so a focal-estimation model is released before the
        main models are loaded (two DA3 models are never resident).

        Raises:
            ConfigError: the depth model needs intrinsics and no provider is configured.
        """
        info = self.resolve_intrinsics(source)
        self.load(source, info)
        return info

    def process(self, frame: Frame) -> FrameResult:
        """Instances, depth and per-instance distance statistics of one frame.

        Timings (ms): "det" (sum), "det.<name>" per detector, "aggregate",
        "depth", "dist".
        """
        parts: list[Instances] = []
        det_ms: dict[str, float] = {}
        for detector in self.detectors:
            instances, det_ms[f"det.{detector.name}"] = _timed(detector, frame)
            parts.append(instances)
        instances, aggregate_ms = _timed(self.aggregator, parts)
        depth, depth_ms = _timed(self.depth, frame)
        stats, dist_ms = _timed(self._distances, instances, depth)
        timings = {
            "det": sum(det_ms.values()),
            **det_ms,
            "aggregate": aggregate_ms,
            "depth": depth_ms,
            "dist": dist_ms,
        }
        return FrameResult(frame, instances, stats, depth, timings)

    def _distances(
        self, instances: Instances, depth: DepthResult
    ) -> tuple[BoxDepthStats, ...]:
        # Sky pixels are not eligible when asked and the model has a sky head.
        mask = ~depth.sky if self.exclude_sky and depth.sky is not None else None
        return tuple(
            box_depth_stats(
                depth.depth_m,
                box,
                depth.geometry,
                self.distance,
                intrinsics=depth.intrinsics,
                mask=mask,
            )
            for box in instances.xyxy
        )

    def close(self) -> None:
        """Release every model."""
        for detector in self.detectors:
            detector.close()
        self.depth.close()

    def describe(self) -> dict[str, Any]:
        """{"components": component descriptions, "pipeline": pipeline policy}."""
        return {
            "components": {
                "detectors": [d.describe() for d in self.detectors],
                "depth": self.depth.describe(),
                "aggregator": self.aggregator.describe(),
            },
            "pipeline": {"exclude_sky": self.exclude_sky},
        }

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
