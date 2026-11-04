"""Combination of the instances of several detectors into one list.

Aggregators are not Components (nothing to load, no name). `concat` (the
default) keeps everything. Fusion aggregators (nms, wbf) merge instances of
the same object, from different detectors and also duplicates from one
detector, grouped by canonical label unless class_agnostic. The fused
instance keeps every member in `sources` (detector, raw label, original
score); class_id and raw_label come from its highest-weighted member, the
mask from the highest-weighted member that has a non-empty mask (has_mask),
so a box-only detector winning the cluster does not drop a segmenter's mask.
Scores of different models are not calibrated: per-detector `weights` only
rescale them for ranking and averaging.
"""

import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar, Literal

import numpy as np

from av_perception_lab.core.boxes import label_groups, nms_clusters, wbf_clusters
from av_perception_lab.core.options import ConfigError
from av_perception_lab.core.registry import register
from av_perception_lab.core.types import Instances, Provenance


class Aggregator(ABC):
    """Merges per-detector instances (in detector order) into one Instances."""

    type_name: ClassVar[str]  # set by @register
    Options: ClassVar[type]

    def __init__(self, options: Any) -> None:
        self.options = options

    @abstractmethod
    def __call__(self, parts: Sequence[Instances]) -> Instances: ...

    def bind_sources(self, names: Sequence[str]) -> None:
        """Receive the detector names, in order, once at pipeline construction.

        No-op here; fusing aggregators override it to validate per-source settings.
        """

    def describe(self) -> dict[str, Any]:
        """JSON-ready description for metrics.json."""
        return {"type": self.type_name, "options": asdict(self.options)}


@register("aggregator", "concat")
class Concat(Aggregator):
    """Keep every instance of every detector, tagged by `source`. Scores of
    different models are not comparable, so nothing is merged or suppressed."""

    @dataclass(frozen=True, slots=True)
    class Options:
        pass

    def __call__(self, parts: Sequence[Instances]) -> Instances:
        return Instances.concat(parts)


@dataclass(frozen=True, slots=True, kw_only=True)
class FusionOptions:
    """Options shared by the fusion aggregators (same names, so
    `--set aggregation.type=nms` works on a wbf section)."""

    iou_thr: float = 0.5  # same-object overlap, in (0, 1)
    class_agnostic: bool = False  # True: fuse across labels (label of the best member)
    # detector name -> weight > 0 multiplying its scores (absent = 1)
    weights: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 < self.iou_thr < 1.0:
            raise ValueError(f"iou_thr must be in (0, 1), got {self.iou_thr}")
        bad = [name for name, w in self.weights.items() if w <= 0]
        if bad:
            raise ValueError(
                f"weights must be > 0, got {bad[0]!r}: {self.weights[bad[0]]}"
            )


class FusionAggregator(Aggregator):
    """Template method: concat -> weights -> label groups -> `_fuse` -> fused
    instances carrying their members in `sources`."""

    options: FusionOptions

    def __init__(self, options: FusionOptions) -> None:
        super().__init__(options)
        # Weight of every bound detector, in detector order; empty until
        # bind_sources (direct use, e.g. tests: every weight is 1).
        self._weights: dict[str, float] = {}

    def bind_sources(self, names: Sequence[str]) -> None:
        """Check `weights` against the detector names and fix every detector's weight.

        Raises:
            ConfigError: a weight names no configured detector.
        """
        unknown = [n for n in self.options.weights if n not in names]
        if unknown:
            raise ConfigError(
                f"aggregation.weights: unknown detector {unknown[0]!r}; "
                f"detectors: {', '.join(names)}"
            )
        self._weights = {n: self.options.weights.get(n, 1.0) for n in names}

    def __call__(self, parts: Sequence[Instances]) -> Instances:
        inst = Instances.concat(parts)
        if not len(inst):
            return Instances.empty()
        weight = np.array([self._weights.get(s, 1.0) for s in inst.source])
        wscore = inst.score.astype(np.float64) * weight
        groups = label_groups(inst.label, class_aware=not self.options.class_agnostic)
        model_weights = (
            np.array(list(self._weights.values()))
            if self._weights
            else np.ones(len(parts))
        )
        clusters, xyxy, score = self._fuse(inst, wscore, groups, model_weights)
        return self._build(inst, clusters, xyxy, score)

    @abstractmethod
    def _fuse(
        self,
        inst: Instances,
        wscore: np.ndarray,
        groups: np.ndarray,
        model_weights: np.ndarray,
    ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
        """(clusters of indices into `inst`, best weighted member first; fused
        boxes (K, 4); fused scores (K,)) from the weighted scores, the label
        groups and the weights of all T detectors (T = len(model_weights))."""

    def _build(
        self,
        inst: Instances,
        clusters: list[np.ndarray],
        xyxy: np.ndarray,
        score: np.ndarray,
    ) -> Instances:
        """One instance per cluster: per-instance fields of its representative
        (first member), the fused box and score, the mask of the first member
        that has one (else the representative's) and every member in `sources`.
        With masks, `mask_source` names that mask's member ("" when no member has
        one) and `mask_score` is NaN (a detector mask), so `mask_sources()` does
        not credit a box-only representative with another detector's mask."""
        out = inst.select(np.array([c[0] for c in clusters], dtype=np.intp))
        masks, mask_source, mask_score = out.masks, None, None
        if inst.masks is not None:
            has = inst.has_mask()
            owner = [next((k for k in c if has[k]), c[0]) for c in clusters]
            masks = inst.masks[np.array(owner, dtype=np.intp)]
            mask_source = tuple(inst.source[k] if has[k] else "" for k in owner)
            mask_score = np.full(len(clusters), np.nan, dtype=np.float32)
        sources = tuple(
            tuple(
                Provenance(inst.source[k], inst.raw_label[k], float(inst.score[k]))
                for k in c
            )
            for c in clusters
        )
        return dataclasses.replace(
            out,
            xyxy=np.asarray(xyxy, dtype=np.float32).reshape(-1, 4),
            score=np.asarray(score, dtype=np.float32),
            masks=masks,
            sources=sources,
            mask_source=mask_source,
            mask_score=mask_score,
        )

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "weights": dict(self._weights)}


@register("aggregator", "nms")
class Nms(FusionAggregator):
    """Non-maximum suppression over the union of all detectors' instances.
    The best weighted instance of each cluster survives with its own box and
    original score; suppressed ones are recorded in `sources`."""

    Options = FusionOptions  # nothing beyond the common options

    def _fuse(
        self,
        inst: Instances,
        wscore: np.ndarray,
        groups: np.ndarray,
        model_weights: np.ndarray,
    ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
        clusters = nms_clusters(inst.xyxy, wscore, groups, self.options.iou_thr)
        keepers = np.array([c[0] for c in clusters], dtype=np.intp)
        return clusters, inst.xyxy[keepers], inst.score[keepers]


@register("aggregator", "wbf")
class Wbf(FusionAggregator):
    """Weighted Boxes Fusion: each cluster becomes one box, the score-weighted
    mean of its members. conf_type="avg" averages weighted scores and rescales
    by min(T, n) / sum(weights) (T = number of detectors), so an instance seen by
    one of two equally weighted detectors gets half its score, as in the
    reference implementation, capped at 1 (unlike the reference, which can
    exceed 1 with unequal weights); "max" takes the best weighted score /
    max(weights)."""

    @dataclass(frozen=True, slots=True, kw_only=True)
    class Options(FusionOptions):
        iou_thr: float = 0.55  # default of the reference implementation
        conf_type: Literal["avg", "max"] = "avg"
        skip_box_thr: float = 0.0  # members whose original score is below are dropped

    def _fuse(
        self,
        inst: Instances,
        wscore: np.ndarray,
        groups: np.ndarray,
        model_weights: np.ndarray,
    ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
        keep = np.flatnonzero(inst.score >= self.options.skip_box_thr)
        clusters, xyxy = wbf_clusters(
            inst.xyxy[keep], wscore[keep], groups[keep], self.options.iou_thr
        )
        clusters = [keep[c] for c in clusters]
        if self.options.conf_type == "avg":
            n_models, total = len(model_weights), model_weights.sum()
            # The reference's rescale exceeds 1 when a detector weighted above
            # the mean contributes several boxes to one cluster (n > T); the
            # score is a confidence (CSV `confidence`, label percent), so cap it.
            score = [
                min(1.0, wscore[c].mean() * min(n_models, len(c)) / total)
                for c in clusters
            ]
        else:
            score = [wscore[c].max() / model_weights.max() for c in clusters]
        return clusters, xyxy, np.asarray(score, dtype=np.float64)
