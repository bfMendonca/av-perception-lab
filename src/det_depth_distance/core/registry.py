"""Name -> class registry of adapters, filled by the `@register` decorator.

`load_adapters()` imports the adapter modules listed in ADAPTER_MODULES;
those modules import their frameworks only inside `_load`, so listing and
validating components works in the light environment (no torch). A spec's
optional "name" key is accepted only for Component kinds; aggregators have
no name, so "name" in an aggregator spec is a ConfigError.

Adding an adapter = a new module with `@register(kind, type_name)` and an
entry in ADAPTER_MODULES; pipeline, config and CLI do not change. Besides
the components, the families "aggregator", "motion" (motion models) and
"ego_motion" (ego-motion providers) hold registered classes that are not
Components.

Pipeline stages are described, for user interfaces, by `Stage` descriptors in
a second registry (`register_stage` / `stages`): the YAML key, title, adapter
family or config dataclass, cardinality and order of each stage. The built-in
stages are registered at the bottom of `config.py` (this module cannot import
it); a UI renders whatever is registered, so a new stage needs no UI change.
"""

import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, TypeVar

from det_depth_distance.core.components import Component
from det_depth_distance.core.options import ConfigError, from_mapping

T = TypeVar("T")

ADAPTER_MODULES: tuple[str, ...] = (
    "det_depth_distance.core.aggregate",
    "det_depth_distance.core.tracking",
    "det_depth_distance.motion.models",
    "det_depth_distance.motion.ego",
    "det_depth_distance.adapters.detectors.rfdetr",
    "det_depth_distance.adapters.detectors.ultralytics_yolo",
    "det_depth_distance.adapters.depth.da3",
    "det_depth_distance.adapters.refiners.hf_sam",
    "det_depth_distance.adapters.refiners.ultralytics_sam",
    "det_depth_distance.adapters.semantic.segformer",
    "det_depth_distance.adapters.semantic.twinlitenet",
)

_REGISTRY: dict[str, dict[str, type]] = {}


def register(kind: str, type_name: str) -> Callable[[type[T]], type[T]]:
    """Class decorator registering an adapter as `type_name` of family `kind`.

    Raises:
        ValueError: `type_name` is already registered for `kind`.
    """

    def decorator(cls: type[T]) -> type[T]:
        family = _REGISTRY.setdefault(kind, {})
        if type_name in family:
            raise ValueError(f"{kind} {type_name!r} is already registered")
        cls.type_name = type_name
        family[type_name] = cls
        return cls

    return decorator


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage:
    """UI-facing descriptor of one pipeline stage (one top-level YAML key).

    `kind` (an adapter family: the stage holds registered specs
    `{type, **options}`) or `config_cls` (a plain options dataclass) describes
    the stage's content.
    """

    yaml_key: str
    title: str
    order: int  # position in the pipeline lane
    kind: str | None = None  # adapter family of the spec(s)
    config_cls: type | None = None  # dataclass of a plain config section
    multiple: bool = False  # a list of specs instead of one
    optional: bool = False  # the stage can be absent (null)
    general: bool = False  # shown in "General settings", not in the lane


_STAGES: dict[str, Stage] = {}


def register_stage(stage: Stage) -> None:
    """Register a pipeline stage descriptor.

    Raises:
        ValueError: `stage.yaml_key` is already registered.
    """
    if stage.yaml_key in _STAGES:
        raise ValueError(f"stage {stage.yaml_key!r} is already registered")
    _STAGES[stage.yaml_key] = stage


def stages() -> tuple[Stage, ...]:
    """Registered stages sorted by `order` (registration order breaks ties)."""
    return tuple(sorted(_STAGES.values(), key=lambda s: s.order))


def load_adapters() -> None:
    """Import every adapter module so its classes register themselves."""
    for module in ADAPTER_MODULES:
        importlib.import_module(module)


def available(kind: str | None = None) -> dict[str, dict[str, type]]:
    """Registered classes by family (only `kind` when given), for --list-models."""
    load_adapters()
    if kind is None:
        return {k: dict(v) for k, v in _REGISTRY.items()}
    return {kind: dict(_REGISTRY.get(kind, {}))}


def build(kind: str, spec: Mapping[str, Any], *, path: str) -> Any:
    """Instantiate a registered class from a spec `{"type", "name"?, **options}`.

    Only validates options; nothing is loaded.

    Raises:
        ConfigError: missing/unknown type, "name" for a non-Component kind, or
            invalid options (message prefixed with `path`).
    """
    load_adapters()
    if not isinstance(spec, Mapping):
        raise ConfigError(f"{path}: expected a mapping with a 'type' key")
    options = dict(spec)
    type_name = options.pop("type", None)
    family = _REGISTRY.get(kind, {})
    if type_name not in family:
        raise ConfigError(
            f"{path}.type: unknown {kind} {type_name!r}; available: {', '.join(family)}"
        )
    cls = family[type_name]
    name = options.pop("name", None)
    if name is not None and not issubclass(cls, Component):
        raise ConfigError(f"{path}.name: {kind}s have no name")
    if name is not None and not isinstance(name, str):
        raise ConfigError(f"{path}.name: expected a string, got {name!r}")
    opts = from_mapping(cls.Options, options, path)
    try:
        return cls(opts, name=name) if issubclass(cls, Component) else cls(opts)
    except ConfigError as exc:
        raise ConfigError(f"{path}.{exc}") from None
