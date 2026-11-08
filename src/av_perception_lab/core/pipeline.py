"""Model-agnostic per-frame pipeline.

Stages: detectors -> (agreement) -> aggregation -> mask refiners ->
tracker -> depth -> semantic segmenters (in config order) -> pixel
eligibility (not sky, not in the union of excluded "<segmenter>.<class>"
labels) -> per-instance distance -> per-track temporal smoothing ->
per-track motion estimation.

Depends only on the abstract components; timings are measured here around
each stage so components stay stateless per call. `describe()` keeps
component descriptions ("components") apart from pipeline policy
("pipeline"). Distance uses each instance's mask or box according to the
`region` policy (see distance.instance_depth_stats). Without refiners,
semantic segmenters, tracker, smoothing and motion every output is the one of
the detector-only pipeline.

Import direction: this module may import configuration dataclasses from
`config`, `temporal`, `motion`, `camera` and `io.source`; those modules never import
`core.pipeline`. `core.types` and `core.components` stay contract-only.
"""

import re
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from itertools import combinations
from typing import Any, Self, TypeVar, get_args

import numpy as np

from av_perception_lab.camera import IntrinsicsProvider
from av_perception_lab.config import AgreementConfig
from av_perception_lab.core.aggregate import Aggregator, FusionAggregator
from av_perception_lab.core.agreement import frame_agreement
from av_perception_lab.core.components import (
    Component,
    DepthEstimator,
    Detector,
    LoadContext,
    MaskRefiner,
    SemanticSegmenter,
    Tracker,
)
from av_perception_lab.core.labels import SEMANTIC_NAME_PATTERN, group_semantic_labels
from av_perception_lab.core.options import ConfigError
from av_perception_lab.core.types import (
    CameraIntrinsics,
    DepthResult,
    Frame,
    FrameAgreement,
    FrameResult,
    Instances,
    SemanticMap,
)
from av_perception_lab.distance import (
    BoxDepthStats,
    DistanceConfig,
    RegionPolicy,
    instance_depth_stats,
    map_eligibility,
)
from av_perception_lab.io.source import FrameSource
from av_perception_lab.motion import MotionEstimator
from av_perception_lab.temporal import SmoothingConfig, TrackSmoother

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class SetupInfo:
    """Camera facts resolved before the models are loaded."""

    frame_intrinsics: CameraIntrinsics | None  # frame resolution
    focal_frame_px: float | None
    focal_source: str
    # IntrinsicsResult.details of the provider (e.g. the calibration cross-check).
    intrinsics_details: Mapping[str, Any] = field(default_factory=dict)


def _timed(fn: Callable[..., T], *args: Any) -> tuple[T, float]:
    """(fn(*args), elapsed ms). Models return numpy arrays, so each call ends
    with a device sync and wall-clock time is the stage's GPU time."""
    t0 = time.perf_counter()
    out = fn(*args)
    return out, (time.perf_counter() - t0) * 1e3


class Pipeline:
    """Runs N detectors, the aggregator, the mask refiners, the tracker, the depth
    model, the semantic segmenters, the distance statistics, the temporal
    smoothing and the motion estimation on each frame; orchestrates, never
    decodes or writes."""

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
        tracker: Tracker | None = None,
        smoothing: SmoothingConfig | None = None,
        motion: MotionEstimator | None = None,
    ) -> None:
        """Raises ConfigError for no detectors, duplicate component names, an
        invalid semantic segmenter name, an unknown `region` policy, aggregator
        settings naming unknown detectors (`bind_sources`), an
        `exclude_semantic` label whose prefix names no semantic segmenter, or
        `smoothing` or `motion` without a `tracker` (both filter per track id).

        `agreement` takes effect only when enabled and with two or more detectors.
        `refiners` run in order (each sees the previous one's masks);
        `exclude_semantic` holds "<segmenter>.<class>" labels, as a sequence or a
        comma-separated string (classes are checked by `load`). The `tracker`
        runs after the refiners; `smoothing` filters each track's distance and
        `motion` (built by the CLI, motion.build_motion) its position on the plane.
        """
        if not detectors:
            raise ConfigError("detectors: at least one detector is required")
        if smoothing is not None and tracker is None:
            raise ConfigError("smoothing: needs tracking (set the tracking section)")
        if motion is not None and tracker is None:
            raise ConfigError("motion: needs tracking (set the tracking section)")
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
        self.tracker = tracker
        self.smoothing = smoothing
        self._smoother: TrackSmoother | None = None  # created by load()
        self.motion = motion
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
                "set camera.calibration, camera.focal_px or camera.estimate"
            )
        resolved = self.intrinsics.resolve(source, device=self.device)
        return SetupInfo(
            resolved.intrinsics,
            resolved.intrinsics.fx,
            resolved.source,
            resolved.details,
        )

    def load(self, source: FrameSource, info: SetupInfo) -> None:
        """Load every detector, the depth model, the refiners, the semantic
        segmenters and the tracker (in that order) for `source`'s frame size,
        start a fresh track smoother and motion estimator, then resolve the excluded semantic
        classes against each segmenter's class names.

        Raises:
            ConfigError: an excluded class is not a class of its segmenter.
        """
        ctx = LoadContext(self.device, source.frame_size, info.frame_intrinsics)
        for detector in self.detectors:
            detector.load(ctx)
        self._warn_disjoint_labels()
        self._warn_unknown_motion_classes()
        self.depth.load(ctx)
        for component in (*self.refiners, *self.semantic, *self._trackers):
            component.load(ctx)
        self._smoother = (
            TrackSmoother(self.smoothing) if self.smoothing is not None else None
        )
        if self.motion is not None:
            self.motion.reset()
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

    def semantic_classes(self) -> dict[str, tuple[str, ...]]:
        """Class names of every semantic segmenter by name (valid after `load`)."""
        return {s.name: tuple(s.class_names) for s in self.semantic}

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

    def _warn_unknown_motion_classes(self) -> None:
        # A motion.classes key that no detector emits never selects its model
        # (e.g. "vehicle" with a COCO detector and no label map).
        if self.motion is None:
            return
        labels = frozenset().union(*(d.output_labels for d in self.detectors))
        for label in self.motion.cfg.classes:
            if label not in labels:
                print(
                    f"[warn] motion.classes.{label}: no detector emits this label "
                    "(set label maps (labels) or fix the class name)",
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
        (only with refiners), "track" (only with a tracker), "depth", "semantic"
        (sum) and "semantic.<name>" (only with semantic segmenters), "dist"
        (eligibility included), "smooth" (only with smoothing), "motion" (only
        with motion).
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
        if self.tracker is not None:
            instances, timings["track"] = _timed(self.tracker, frame, instances)
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
        temporal = None
        if self._smoother is not None:
            # Primary statistics only: with region "both" box_stats stay raw.
            temporal, timings["smooth"] = _timed(
                self._smoother.update, frame.t, instances.track_id, stats
            )
        motion = None
        if self.motion is not None:
            motion, timings["motion"] = _timed(
                self.motion.update, frame.t, instances, stats, depth
            )
        return FrameResult(
            frame,
            instances,
            stats,
            depth,
            timings,
            agreement,
            box_stats,
            maps,
            temporal=temporal,
            motion=motion,
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

    @property
    def _trackers(self) -> tuple[Tracker, ...]:
        return () if self.tracker is None else (self.tracker,)

    def close(self) -> None:
        """Release every model and the tracker's state."""
        for component in (
            *self.detectors,
            self.depth,
            *self.refiners,
            *self.semantic,
            *self._trackers,
        ):
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
                "tracker": (
                    self.tracker.describe() if self.tracker is not None else None
                ),
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
                "smoothing": (
                    asdict(self.smoothing) if self.smoothing is not None else None
                ),
                "motion": self.motion.describe() if self.motion is not None else None,
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
