"""Abstract pipeline components.

Construction only validates options (cheap, no framework imports), so every
configuration error surfaces before any model is loaded; `load()` does the
heavy work and `close()` releases GPU memory. Concrete adapters translate
their framework (colour order, letterbox, units, class names) and return the
shared contracts of `core.types`; the base classes apply the common policy
(class filter, label mapping, contract validation) as template methods.

Layered options rule: option dataclasses are `frozen=True, slots=True,
kw_only=True`; a subclass whose `__post_init__` adds checks calls the base
explicitly, `DetectorOptions.__post_init__(self)`, never a bare
`super().__post_init__()` (slots=True recreates the class, so the zero-arg
super() cell points at the original class and raises TypeError). A subclass
without its own checks simply inherits the base `__post_init__`.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from det_depth_distance.core.labels import (
    LabelMap,
    identify_vocabulary,
    resolve_classes,
)
from det_depth_distance.core.options import ConfigError
from det_depth_distance.core.types import (
    CameraIntrinsics,
    ContractError,
    DepthResult,
    Frame,
    Instances,
    MapGeometry,
)


@dataclass(frozen=True, slots=True)
class LoadContext:
    """What a component may need to load: device, frame size and camera."""

    device: str
    frame_size: tuple[int, int]  # (W, H)
    intrinsics: CameraIntrinsics | None  # frame resolution; None if nobody needs it


class Component(ABC):
    """A configurable model wrapper with an explicit load/close life cycle."""

    kind: ClassVar[str]  # registry family: "detector", "depth"
    type_name: ClassVar[str]  # set by @register
    Options: ClassVar[type]  # dataclass of the adapter's options

    def __init__(self, options: Any, *, name: str | None = None) -> None:
        self.options = options
        self.name = name or self.default_name()
        self._loaded = False

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
