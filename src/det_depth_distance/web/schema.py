"""Options schema, module catalog and validation for the web interface.

`options_schema` describes an options dataclass for a form generator by
walking its type hints, mirroring the annotations `core.options.from_mapping`
accepts. UI hints are dataclass field metadata
(`field(default=..., metadata={"ui": {...}})`, keys label/unit/min/max/step/
advanced/help, all optional) and travel unchanged as the field's `hint`.
`module_catalog` is a pure function of the registries (adapters, stages).
`validate_config` returns every problem it can find as dotted-path errors.
"""

import dataclasses
import re
import types
import typing
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Union

from det_depth_distance import config
from det_depth_distance.core import registry
from det_depth_distance.core.options import ConfigError
from det_depth_distance.web import stages

CATALOG_VERSION = 1

_SCALARS: dict[Any, str] = {bool: "bool", int: "int", float: "float", str: "str"}
_PATH_PREFIX = re.compile(r"^([A-Za-z_][\w.\[\]]*): (.*)$", re.DOTALL)


def _plain(value: Any) -> Any:
    """JSON-ready copy of a default value."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: _plain(getattr(value, f.name)) for f in dataclasses.fields(value)
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    return value


def _type_schema(tp: Any) -> dict[str, Any]:
    """Schema of one annotation: `type`, `nullable` and, by type, `choices`,
    `items`, `fields` or `variants`."""
    if tp is Any:
        return {"type": "any", "nullable": True}
    origin, args = typing.get_origin(tp), typing.get_args(tp)
    if origin in (Union, types.UnionType):
        nullable = type(None) in args
        members = [_type_schema(a) for a in args if a is not type(None)]
        if len(members) == 1:
            return {**members[0], "nullable": nullable or members[0]["nullable"]}
        return {"type": "any", "nullable": nullable, "variants": members}
    if origin is Literal:
        return {"type": "enum", "nullable": False, "choices": list(args)}
    if dataclasses.is_dataclass(tp):
        return {"type": "object", "nullable": False, **options_schema(tp)}
    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return {"type": "list", "nullable": False, "items": _type_schema(args[0])}
        members = [_type_schema(a) for a in args]
        same = all(m == members[0] for m in members)
        return {
            "type": "list",
            "nullable": False,
            "items": members[0] if same and members else {"type": "any"},
            "length": len(args),
        }
    if origin in (dict, Mapping):
        return {"type": "map", "nullable": False, "items": _type_schema(args[1])}
    if tp is Path:
        return {"type": "path", "nullable": False}
    if tp in _SCALARS:
        return {"type": _SCALARS[tp], "nullable": False}
    raise TypeError(f"options_schema does not support the annotation {tp!r}")


def options_schema(cls: type) -> dict[str, Any]:
    """`{"fields": [Field]}` of an options dataclass.

    Field = {name, type, default, required, nullable, hint, choices?, items?,
    fields?}; `type` is one of bool, int, float, str, path, enum, list, map,
    object, any.

    Raises:
        TypeError: an annotation `from_mapping` does not support either.
    """
    hints = typing.get_type_hints(cls)
    out = []
    for f in dataclasses.fields(cls):
        if not f.init:
            continue
        if f.default is not dataclasses.MISSING:
            default, required = _plain(f.default), False
        elif f.default_factory is not dataclasses.MISSING:
            default, required = _plain(f.default_factory()), False
        else:
            default, required = None, True
        out.append(
            {
                "name": f.name,
                **_type_schema(hints[f.name]),
                "default": default,
                "required": required,
                "hint": dict(f.metadata.get("ui", {})),
            }
        )
    return {"fields": out}


def module_catalog() -> dict[str, Any]:
    """Everything the UI needs to render any pipeline: stages, the options of
    every registered adapter by family, and the options of the config sections."""
    stage_list = stages.stage_list()
    modules = {
        kind: {
            type_name: {
                "schema": options_schema(cls.Options),
                "doc": (cls.__doc__ or "").strip().split("\n\n")[0].replace("\n", " "),
            }
            for type_name, cls in classes.items()
        }
        for kind, classes in registry.available().items()
    }
    sections = {
        s.yaml_key: {"schema": options_schema(s.config_cls)}
        for s in registry.stages()
        if s.config_cls is not None
    }
    return {
        "catalog_version": CATALOG_VERSION,
        "stages": stage_list,
        "modules": modules,
        "sections": sections,
    }


def _split_error(exc: ConfigError) -> dict[str, str]:
    """`{"path", "message"}` from a ConfigError message `path: message`."""
    match = _PATH_PREFIX.match(str(exc))
    if match is None:
        return {"path": "", "message": str(exc)}
    return {"path": match.group(1), "message": match.group(2)}


def validate_config(data: dict[str, Any]) -> list[dict[str, str]]:
    """Problems of a run configuration (YAML layout) as `{"path", "message"}`.

    Checks the shape with the same code as `load_config`, then every adapter
    spec of the registered stages independently, so one bad spec does not hide
    the others. Empty = valid.
    """
    errors: list[ConfigError] = []
    cfg = None
    try:
        cfg = config.config_from_mapping(_copy(data))
    except ConfigError as exc:
        errors.append(exc)
    for stage in registry.stages():
        if stage.kind is None or not isinstance(data, Mapping):
            continue
        value = data.get(stage.yaml_key)
        if value is None:
            continue
        specs = (
            list(enumerate(value))
            if stage.multiple and isinstance(value, list)
            else None
        )
        if stage.multiple and specs is None:
            continue  # the shape error comes from the section check above
        for i, spec in specs if specs is not None else [(None, value)]:
            path = stage.yaml_key if i is None else f"{stage.yaml_key}[{i}]"
            try:
                registry.build(stage.kind, spec, path=path)
            except ConfigError as exc:
                errors.append(exc)
    if cfg is not None:
        errors.extend(config.motion_spec_errors(cfg))
    seen, out = set(), []
    for exc in errors:
        item = _split_error(exc)
        key = (item["path"], item["message"])
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _copy(value: Any) -> Any:
    """Deep copy of nested dicts/lists (`config_from_mapping` mutates)."""
    if isinstance(value, dict):
        return {k: _copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_copy(v) for v in value]
    return value
