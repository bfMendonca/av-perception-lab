"""Abstract pipeline components.

Construction only validates options (cheap, no framework imports), so every
configuration error surfaces before any model is loaded; `load()` does the
heavy work and `close()` releases GPU memory. Concrete adapters translate
their framework (colour order, letterbox, units, class names) and return the
shared contracts of `core.types`; the base classes apply the common policy
(class filter, label mapping, target selection, contract validation) as
template methods. Families: detectors, depth models, mask refiners (fill
instance masks from boxes), semantic segmenters (per-pixel class maps) and
trackers (stable instance ids across frames).

Layered options rule: option dataclasses are `frozen=True, slots=True,
kw_only=True`; a subclass whose `__post_init__` adds checks calls the base
explicitly, `DetectorOptions.__post_init__(self)`, never a bare
`super().__post_init__()` (slots=True recreates the class, so the zero-arg
super() cell points at the original class and raises TypeError). A subclass
without its own checks simply inherits the base `__post_init__`.
"""

import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np

from av_perception_lab.core import class_cache
from av_perception_lab.core.labels import (
    LabelMap,
    identify_vocabulary,
    resolve_classes,
)
from av_perception_lab.core.options import ConfigError
from av_perception_lab.core.requirements import Requirements
from av_perception_lab.core.types import (
    CameraIntrinsics,
    ContractError,
    DepthResult,
    Frame,
    Instances,
    MapGeometry,
    RefinedMasks,
    SemanticMap,
)


@dataclass(frozen=True, slots=True)
class LoadContext:
    """What a component may need to load: device, frame size and camera."""

    device: str
    frame_size: tuple[int, int]  # (W, H)
    intrinsics: CameraIntrinsics | None  # frame resolution; None if nobody needs it


@dataclass(frozen=True, slots=True)
class ClassCatalog:
    """The classes a component can emit, known before loading it.

    `source`: "static" (constant of the architecture), "metadata" (read from
    the checkpoint's config files, no framework import) or "cache" (recorded
    by an earlier load)."""

    names: tuple[str, ...]
    source: Literal["static", "metadata", "cache"]


class Component(ABC):
    """A configurable model wrapper with an explicit load/close life cycle."""

    # registry family: "detector", "depth", "refiner", "semantic", "tracker"
    kind: ClassVar[str]
    type_name: ClassVar[str]  # set by @register
    Options: ClassVar[type]  # dataclass of the adapter's options

    def __init__(self, options: Any, *, name: str | None = None) -> None:
        self.options = options
        self.name = name or self.default_name()
        self._loaded = False

    @classmethod
    def requirements(cls, options: Any) -> Requirements:
        """What this class needs to run, from already-validated options (never
        from `load()`); the default declares nothing."""
        return Requirements()

    @classmethod
    def declared_classes(cls, options: Any) -> ClassCatalog | None:
        """Classes this component will emit, without loading it (no framework
        import, never raises); None when they are not knowable beforehand."""
        return None

    def default_name(self) -> str:
        """Name used in outputs when the spec gives none (e.g. "rfdetr-large")."""
        return self.type_name

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self, ctx: LoadContext) -> None:
        """Load the model (idempotent); `close()` allows loading again."""
        if self._loaded:
            return
        self._load(ctx)
        self._loaded = True
        self._after_load()

    def close(self) -> None:
        """Release the model (GPU memory); no-op when not loaded."""
        if self._loaded:
            self._close()
            self._loaded = False

    def describe(self) -> dict[str, Any]:
        """JSON-ready description for metrics.json: type, name, options + adapter facts."""
        options = {
            k: str(v) if isinstance(v, Path) else v
            for k, v in asdict(self.options).items()
        }
        return {
            "type": self.type_name,
            "name": self.name,
            "options": options,
            **self._describe(),
        }

    @abstractmethod
    def _load(self, ctx: LoadContext) -> None: ...

    def _after_load(self) -> None:
        """Hook run once after `_load` (base-class policy that needs the model)."""

    def _close(self) -> None:
        """Drop references to the model; the default keeps nothing to release."""

    def _describe(self) -> dict[str, Any]:
        return {}

    def _require_loaded(self) -> None:
        if not self._loaded:
            raise ContractError(f"{self.name} used before load()")


@dataclass(frozen=True, slots=True, kw_only=True)
class DetectorOptions:
    """Options shared by every detector adapter (kw_only: adapters add required fields)."""

    threshold: float = 0.5  # minimum score, in (0, 1]
    classes: str | tuple[str, ...] = "driving"  # "all", a preset, or class names
    labels: Path | None = None  # LabelMap YAML (raw -> canonical labels); None = raw

    def __post_init__(self) -> None:
        if not 0.0 < self.threshold <= 1.0:
            raise ValueError(f"threshold must be in (0, 1], got {self.threshold}")


class Detector(Component):
    """Template method: the adapter only translates its framework in `_predict`;
    class filter, label mapping and validation belong to this base, so every
    detector delivers the same contract."""

    kind = "detector"

    def __init__(self, options: DetectorOptions, *, name: str | None = None) -> None:
        """Raises ConfigError for an unreadable or invalid label map (before any load)."""
        super().__init__(options, name=name)
        self._keep: frozenset[str] | None = None
        self._vocabulary = "custom"
        self.label_map: LabelMap | None = None
        if options.labels is not None:
            try:
                self.label_map = LabelMap.load(options.labels)
            except ConfigError as exc:
                raise ConfigError(f"labels: {exc}") from None

    @classmethod
    def weights_file(cls, options: Any) -> Path | None:
        """The checkpoint file whose classes the load-time cache records
        (importable without frameworks); None when there is no such file."""
        return None

    @classmethod
    def declared_classes(cls, options: Any) -> ClassCatalog | None:
        """Classes recorded by an earlier load of the same weights file."""
        try:
            weights = cls.weights_file(options)
            names = None if weights is None else class_cache.read(weights)
        except (ConfigError, OSError):
            return None
        return None if names is None else ClassCatalog(names, "cache")

    @property
    @abstractmethod
    def class_names(self) -> Sequence[str]:
        """Class names of the loaded model, indexed by class id."""

    @property
    def vocabulary(self) -> str:
        """ "coco", "objects365" or "custom", identified from `class_names` at load."""
        self._require_loaded()
        return self._vocabulary

    def _after_load(self) -> None:
        # Class errors surface at load time, before the first frame.
        vocabulary = identify_vocabulary(self.class_names)
        self._vocabulary = vocabulary.name if vocabulary else "custom"
        self._keep = resolve_classes(self.options.classes, self.class_names)
        if self.label_map is not None:
            try:
                self.label_map.check(self.class_names, self._vocabulary)
            except ConfigError as exc:
                raise ConfigError(f"{self.name}.labels: {exc}") from None
        weights = self.weights_file(self.options)
        if weights is not None:
            class_cache.write(weights, list(self.class_names))

    @property
    def output_labels(self) -> frozenset[str]:
        """Labels this detector can emit: the classes kept by the filter, mapped
        by the label map (valid after load)."""
        self._require_loaded()
        raw = tuple(self.class_names if self._keep is None else self._keep)
        if self.label_map is None:
            return frozenset(raw)
        labels, keep = self.label_map.apply(raw)
        return frozenset(label for label, k in zip(labels, keep, strict=True) if k)

    def describe(self) -> dict[str, Any]:
        labels = self.label_map.describe() if self.label_map else None
        return {**super().describe(), "vocabulary": self._vocabulary, "labels": labels}

    def __call__(self, frame: Frame) -> Instances:
        """Detect instances in `frame`.

        Runs the adapter's `_predict` on the RGB array, keeps only the configured
        classes (by raw label), maps labels, and validates the result against the
        frame size.

        Raises:
            ContractError: not loaded, or the adapter broke the contract.
        """
        self._require_loaded()
        instances = self._predict(frame.rgb)
        if self._keep is not None:
            keep = np.array([n in self._keep for n in instances.raw_label], dtype=bool)
            instances = instances.select(keep)
        instances = self._map_labels(instances)
        instances.validate(frame.width, frame.height)
        return instances

    @abstractmethod
    def _predict(self, rgb: np.ndarray) -> Instances:
        """Raw model output for an (H, W, 3) uint8 RGB frame, boxes in frame pixels."""

    def _instances(
        self,
        xyxy: np.ndarray,
        score: np.ndarray,
        class_id: np.ndarray,
        names: Sequence[str],
        masks: np.ndarray | None = None,
    ) -> Instances:
        """Contract dtypes for adapter outputs: label = raw_label = names, source = self.name."""
        names = tuple(names)
        return Instances(
            np.asarray(xyxy, dtype=np.float32).reshape(-1, 4),
            np.asarray(score, dtype=np.float32),
            np.asarray(class_id, dtype=np.int64),
            names,
            names,
            (self.name,) * len(names),
            masks,
        )

    def _map_labels(self, instances: Instances) -> Instances:
        """Raw -> canonical labels through the label map (identity without one);
        drops unmapped instances when the map says `unmapped: drop`."""
        if self.label_map is None:
            return instances
        labels, keep = self.label_map.apply(instances.raw_label)
        instances = instances.with_labels(labels)
        return instances if keep.all() else instances.select(keep)


class DepthEstimator(Component):
    """Metric depth per frame; the base checks the map against the adapter's geometry."""

    kind = "depth"

    def __init__(self, options: Any, *, name: str | None = None) -> None:
        super().__init__(options, name=name)
        self._geometry: MapGeometry | None = None  # set by the adapter's _load

    @property
    @abstractmethod
    def needs_intrinsics(self) -> bool:
        """Whether this model needs camera intrinsics to output metres.

        Must be answerable before `load()`: the pipeline resolves intrinsics (and may
        load and release a focal-estimation model) before loading the main models.
        """

    @property
    def geometry(self) -> MapGeometry:
        """Frame -> depth-map mapping (valid after load)."""
        self._require_loaded()
        return self._geometry

    def __call__(self, frame: Frame) -> DepthResult:
        """Metric depth of `frame`.

        Raises:
            ContractError: not loaded, or the map's shape/dtype does not match
                the geometry (e.g. a broken replica of the model's resizing).
        """
        self._require_loaded()
        result = self._predict(frame.rgb)
        expected = (result.geometry.map_h, result.geometry.map_w)
        if result.depth_m.shape != expected or result.depth_m.dtype != np.float32:
            raise ContractError(
                f"{self.name}: depth {result.depth_m.shape} {result.depth_m.dtype} "
                f"!= geometry {expected} float32"
            )
        if result.sky is not None and (
            result.sky.shape != expected or result.sky.dtype != bool
        ):
            raise ContractError(f"{self.name}: sky mask must be bool {expected}")
        return result

    @abstractmethod
    def _predict(self, rgb: np.ndarray) -> DepthResult:
        """Depth of an (H, W, 3) uint8 RGB frame at the model's map resolution."""


@dataclass(frozen=True, slots=True, kw_only=True)
class RefinerOptions:
    """Options shared by every mask refiner (kw_only: adapters add required fields)."""

    # "missing": only instances without a non-empty mask in their box (keeps
    # native masks of segmentation detectors); "all": every instance.
    targets: Literal["missing", "all"] = "missing"


class MaskRefiner(Component):
    """Fills instance masks from the boxes of any detector (e.g. SAM 2.1).

    Template method: the base selects the target instances (`targets`:
    "missing" = instances without a non-empty mask, "all" = every instance),
    runs the adapter's `_refine` on them only, merges the result and validates
    it. A refined mask that comes back empty never replaces an existing one.
    This is the only refiner contract the pipeline knows; box-prompted models
    (SAM family) derive from adapters.refiners.box_prompt.BoxPromptRefiner.
    """

    kind = "refiner"
    options: RefinerOptions

    def targets(self, instances: Instances) -> np.ndarray:
        """(N,) bool: the instances this refiner refines under `options.targets`."""
        if self.options.targets == "all":
            return np.ones(len(instances), dtype=bool)
        return ~instances.has_mask()

    def __call__(self, frame: Frame, instances: Instances) -> Instances:
        """Return `instances` with masks for the targets, same length and order.

        `mask_source` names the producer of each mask ("" = none) and `mask_score`
        holds the refiner's predicted quality (NaN when the mask came from the
        detector). Instances that are not refined, or whose refined mask is
        empty, keep their mask, source and score (so chained refiners compose).
        `_refine` is not called when there is no target.

        Raises:
            ContractError: not loaded, or the adapter broke the RefinedMasks contract.
        """
        self._require_loaded()
        n = len(instances)
        sources = list(instances.mask_sources())
        scores = (
            np.full(n, np.nan, dtype=np.float32)
            if instances.mask_score is None
            else instances.mask_score.copy()
        )
        masks = instances.masks
        selected = self.targets(instances)
        if selected.any():
            index = np.flatnonzero(selected)
            refined = self._refine(frame.rgb, instances.select(index))
            refined.validate(len(index), frame.width, frame.height)
            masks = (
                np.zeros((n, frame.height, frame.width), dtype=bool)
                if masks is None
                else masks.copy()
            )
            for j, i in enumerate(index):
                if refined.masks[j].any():
                    masks[i] = refined.masks[j]
                    sources[i] = self.name
                    scores[i] = refined.score[j]
        out = dataclasses.replace(
            instances, masks=masks, mask_source=tuple(sources), mask_score=scores
        )
        out.validate(frame.width, frame.height)
        return out

    @abstractmethod
    def _refine(self, rgb: np.ndarray, instances: Instances) -> RefinedMasks:
        """Masks and quality scores for the target `instances` (K >= 1) of an
        (H, W, 3) uint8 RGB frame. Receives Instances, not only boxes, so future
        refiners may also use labels."""


class SemanticSegmenter(Component):
    """Per-pixel class labels for a frame, used to make pixels ineligible for
    distance (e.g. road, vegetation) and as a context layer in the video.

    `class_names` is known after load() so excluded names are validated before
    the first frame. Several segmenters may run on the same frame; each keeps
    its own map and vocabulary (maps are never merged) and is addressed by its
    component name, e.g. "scene.sky" or "lanes.lane".
    """

    kind = "semantic"

    @property
    @abstractmethod
    def class_names(self) -> tuple[str, ...]:
        """Class names indexed by label value (valid after load)."""

    def __call__(self, frame: Frame) -> SemanticMap:
        """Semantic map of `frame`.

        Raises:
            ContractError: not loaded, or the map does not match the frame size or
                this segmenter's `class_names`.
        """
        self._require_loaded()
        out = self._segment(frame.rgb)
        out.validate(frame.width, frame.height)
        if tuple(out.class_names) != tuple(self.class_names):
            raise ContractError(f"{self.name}: map class_names differ from the model's")
        return out

    @abstractmethod
    def _segment(self, rgb: np.ndarray) -> SemanticMap:
        """Labels of an (H, W, 3) uint8 RGB frame at frame resolution."""


@dataclass(frozen=True, slots=True, kw_only=True)
class TrackerOptions:
    """Options shared by every tracker (none yet; kw_only: adapters add their own).

    It has no `__post_init__`; if it gets one, subclasses call it as
    `TrackerOptions.__post_init__(self)` (see the layered options rule above).
    """


class Tracker(Component):
    """Assigns a stable `track_id` to every instance across frames.

    Stateful per run: `load()` resets it, `close()` drops the state, `reset()`
    starts over (ids restart at 1). Template method: the adapter's `_assign`
    returns one id per instance and the base checks the contract - same
    instances in the same order, int64 ids >= 1, unique within the frame.
    Instances are never dropped; short-lived tracks are filtered in reports,
    not in the stream. Receives the whole Frame so appearance-based trackers
    can be added without changing the pipeline.
    """

    kind = "tracker"

    def reset(self) -> None:
        """Forget every track; the next id is 1 again."""
        self._reset()

    def __call__(self, frame: Frame, instances: Instances) -> Instances:
        """Return `instances` (same length and order) with `track_id` set.

        Frames must come in time order (`frame.t` non-decreasing) within a run.

        Raises:
            ContractError: not loaded, or the adapter returned ids of the wrong
                shape or dtype, ids < 1 or an id repeated within the frame.
        """
        self._require_loaded()
        ids = self._assign(frame, instances)
        out = dataclasses.replace(instances, track_id=ids)
        out.validate(frame.width, frame.height)
        if len(np.unique(ids)) != len(ids):
            raise ContractError(f"{self.name}: repeated track id within a frame")
        return out

    def _load(self, ctx: LoadContext) -> None:
        # Nothing to load for geometric trackers; appearance-based ones (ReID)
        # override this to load their model, and call reset() too.
        self.reset()

    def _close(self) -> None:
        self._reset()

    @abstractmethod
    def _reset(self) -> None:
        """Drop every track and restart the id counter at 1."""

    @abstractmethod
    def _assign(self, frame: Frame, instances: Instances) -> np.ndarray:
        """(N,) int64 track id of every instance, in order."""
