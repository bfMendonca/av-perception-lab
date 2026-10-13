"""Model-agnostic per-frame pipeline.

Stages: detectors -> (agreement) -> aggregation -> mask refiners -> depth ->
semantic segmenters (in config order) -> pixel eligibility (not sky, not in
the union of excluded "<segmenter>.<class>" labels) -> per-instance distance.

Depends only on the abstract components; timings are measured here around
each stage so components stay stateless per call. `describe()` keeps
component descriptions ("components") apart from pipeline policy
("pipeline"). Distance uses each instance's mask or box according to the
`region` policy (see distance.instance_depth_stats). Without refiners and
semantic segmenters every output is the one of the detector-only pipeline.

Import direction: this module may import configuration dataclasses from
`config`, `temporal`, `camera` and `io.source`; those modules never import
`core.pipeline`. `core.types` and `core.components` stay contract-only.
"""

import re
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any, Self, TypeVar, get_args

import numpy as np

from det_depth_distance.camera import IntrinsicsProvider
from det_depth_distance.config import AgreementConfig
from det_depth_distance.core.aggregate import Aggregator, FusionAggregator
from det_depth_distance.core.agreement import frame_agreement
from det_depth_distance.core.components import (
    Component,
    DepthEstimator,
    Detector,
    LoadContext,
    MaskRefiner,
    SemanticSegmenter,
)
from det_depth_distance.core.labels import SEMANTIC_NAME_PATTERN, group_semantic_labels
from det_depth_distance.core.options import ConfigError
from det_depth_distance.core.types import (
    CameraIntrinsics,
    DepthResult,
    Frame,
    FrameAgreement,
    FrameResult,
    Instances,
    SemanticMap,
)
from det_depth_distance.distance import (
    BoxDepthStats,
    DistanceConfig,
    RegionPolicy,
    instance_depth_stats,
    map_eligibility,
)
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
    """Runs N detectors, the aggregator, the mask refiners, the depth model, the
    semantic segmenters and the distance statistics on each frame;
    orchestrates, never decodes or writes."""

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
        agreement: AgreementConfig | None = None,
        region: RegionPolicy = "mask",
        refiners: Sequence[MaskRefiner] = (),
        semantic: Sequence[SemanticSegmenter] = (),
        exclude_semantic: str | Sequence[str] = (),
    ) -> None:
        """Raises ConfigError for no detectors, duplicate component names, an
        invalid semantic segmenter name, an unknown `region` policy, aggregator
        settings naming unknown detectors (`bind_sources`) or an
        `exclude_semantic` label whose prefix names no semantic segmenter.

        `agreement` takes effect only when enabled and with two or more detectors.
        `refiners` run in order (each sees the previous one's masks);
        `exclude_semantic` holds "<segmenter>.<class>" labels, as a sequence or a
        comma-separated string (classes are checked by `load`).
        """
        if not detectors:
            raise ConfigError("detectors: at least one detector is required")
        bad = [
            s.name for s in semantic if not re.fullmatch(SEMANTIC_NAME_PATTERN, s.name)
        ]
        if bad:
            raise ConfigError(
                f"semantic: name {bad[0]!r} must match {SEMANTIC_NAME_PATTERN}"
            )
        _check_unique_names(
            {"detectors": detectors, "refiners": refiners, "semantic": semantic}
        )
        if region not in get_args(RegionPolicy):
            raise ConfigError(
                f"distance.region: {region!r} is not one of {list(get_args(RegionPolicy))}"
            )
        try:
            self._exclude = group_semantic_labels(
                exclude_semantic, [s.name for s in semantic]
            )
        except ConfigError as exc:
            raise ConfigError(f"distance.exclude_semantic: {exc}") from None
        # Excluded label ids per segmenter (only those excluding something), set by load().
        self._exclude_ids: dict[str, np.ndarray] = {}
        self.detectors = tuple(detectors)
        self.refiners = tuple(refiners)
        self.semantic = tuple(semantic)
        self.depth = depth
        self.aggregator = aggregator
        self.distance = distance
        self.region = region
        self.exclude_sky = exclude_sky
        self.intrinsics = intrinsics
        self.device = device
        names = [d.name for d in self.detectors]
        aggregator.bind_sources(names)
        self.agreement = (
            agreement
            if agreement is not None and agreement.enabled and len(names) > 1
            else None
        )

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
        """Load every detector, the depth model, the refiners and the semantic
        segmenters (in that order) for `source`'s frame size, then resolve the
        excluded semantic classes against each segmenter's class names.

        Raises:
            ConfigError: an excluded class is not a class of its segmenter.
        """
        ctx = LoadContext(self.device, source.frame_size, info.frame_intrinsics)
        for detector in self.detectors:
            detector.load(ctx)
        self._warn_disjoint_labels()
        self.depth.load(ctx)
        for component in (*self.refiners, *self.semantic):
            component.load(ctx)
        self._exclude_ids = {}
        for segmenter in self.semantic:
            classes = self._exclude[segmenter.name]
            if not classes:
                continue
            names = segmenter.class_names
            unknown = [c for c in classes if c not in names]
            if unknown:
                raise ConfigError(
                    f"distance.exclude_semantic: {segmenter.name}.{unknown[0]}: "
                    f"unknown class; {segmenter.name} classes: {', '.join(names)}"
                )
            self._exclude_ids[segmenter.name] = np.array(
                [names.index(c) for c in classes], dtype=np.int32
            )

    def _warn_disjoint_labels(self) -> None:
        # Fusing by label merges nothing across two detectors whose label sets
        # are disjoint (e.g. a COCO and an Objects365 model without label maps).
        aggregator = self.aggregator
        if (
            not isinstance(aggregator, FusionAggregator)
            or aggregator.options.class_agnostic
        ):
            return
        for a, b in combinations(self.detectors, 2):
            if not a.output_labels & b.output_labels:
                print(
                    f"[warn] {aggregator.type_name} fuses by label, but {a.name} and "
                    f"{b.name} share no label: set label maps (labels) or "
                    "aggregation.class_agnostic",
                    file=sys.stderr,
                )

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
        """Instances, depth, semantic maps and per-instance distance statistics
        of one frame.

        Timings (ms): "det" (sum), "det.<name>" per detector, "agree" (when
        agreement is computed), "aggregate", "refine" (sum) and "refine.<name>"
        (only with refiners), "depth", "semantic" (sum) and "semantic.<name>"
        (only with semantic segmenters), "dist" (eligibility included).
        """
        parts: list[Instances] = []
        det_ms: dict[str, float] = {}
        for detector in self.detectors:
            instances, det_ms[f"det.{detector.name}"] = _timed(detector, frame)
            parts.append(instances)
        timings = {"det": sum(det_ms.values()), **det_ms}
        agreement = None
        if self.agreement is not None:
            agreement, timings["agree"] = _timed(self._agreement_of, parts)
        instances, timings["aggregate"] = _timed(self.aggregator, parts)
        if self.refiners:
            refine_ms: dict[str, float] = {}
            for refiner in self.refiners:
                instances, refine_ms[f"refine.{refiner.name}"] = _timed(
                    refiner, frame, instances
                )
            timings |= {"refine": sum(refine_ms.values()), **refine_ms}
        depth, timings["depth"] = _timed(self.depth, frame)
        maps: dict[str, SemanticMap] = {}
        if self.semantic:
            semantic_ms: dict[str, float] = {}
            for segmenter in self.semantic:
                name = segmenter.name
                maps[name], semantic_ms[f"semantic.{name}"] = _timed(segmenter, frame)
            timings |= {"semantic": sum(semantic_ms.values()), **semantic_ms}
        (stats, box_stats), timings["dist"] = _timed(
            self._distances, instances, depth, maps
        )
        return FrameResult(
            frame, instances, stats, depth, timings, agreement, box_stats, maps
        )

    def _agreement_of(self, parts: Sequence[Instances]) -> FrameAgreement:
        return frame_agreement(
            parts,
            [d.name for d in self.detectors],
            iou_thr=self.agreement.iou_thr,
            class_aware=self.agreement.class_aware,
        )

    def _distances(
        self, instances: Instances, depth: DepthResult, maps: dict[str, SemanticMap]
    ) -> tuple[tuple[BoxDepthStats, ...], tuple[BoxDepthStats | None, ...] | None]:
        """(stats, box_stats) of every instance; box_stats only with region "both"."""
        # Sky pixels are not eligible when asked and the model has a sky head;
        # neither are pixels of an excluded class of any semantic map.
        mask = map_eligibility(
            depth.geometry,
            sky=depth.sky if self.exclude_sky else None,
            exclusions=[(maps[n].labels, ids) for n, ids in self._exclude_ids.items()],
        )
        masks = instances.masks
        pairs = [
            instance_depth_stats(
                depth.depth_m,
                box,
                depth.geometry,
                self.distance,
                region=self.region,
                instance_mask=None if masks is None else masks[i],
                intrinsics=depth.intrinsics,
                mask=mask,
            )
            for i, box in enumerate(instances.xyxy)
        ]
        stats = tuple(primary for primary, _ in pairs)
        box_stats = tuple(alt for _, alt in pairs) if self.region == "both" else None
        return stats, box_stats

    def close(self) -> None:
        """Release every model."""
        for component in (*self.detectors, self.depth, *self.refiners, *self.semantic):
            component.close()

    def describe(self) -> dict[str, Any]:
        """{"components": component descriptions, "pipeline": pipeline policy}."""
        return {
            "components": {
                "detectors": [d.describe() for d in self.detectors],
                "depth": self.depth.describe(),
                "aggregator": self.aggregator.describe(),
                "refiners": [r.describe() for r in self.refiners],
                "semantic": [s.describe() for s in self.semantic],
            },
            "pipeline": {
                "exclude_sky": self.exclude_sky,
                "agreement": asdict(self.agreement) if self.agreement else None,
                "region": self.region,
                "mask_erode": self.distance.mask_erode,
                # segmenter name -> excluded classes, only maps excluding something
                "exclude_semantic": {
                    name: list(classes)
                    for name, classes in self._exclude.items()
                    if classes
                },
            },
        }

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _check_unique_names(families: dict[str, Sequence[Component]]) -> None:
    """Component names key the outputs (CSV source/mask_source, timings, HUD), so
    they must be unique across detectors, refiners and semantic segmenters.

    Raises:
        ConfigError: a repeated name, reported under the family of its second use.
    """
    seen: set[str] = set()
    for family, components in families.items():
        for component in components:
            if component.name in seen:
                raise ConfigError(
                    f"{family}: duplicate name {component.name!r}; "
                    "give each component a distinct 'name'"
                )
            seen.add(component.name)
