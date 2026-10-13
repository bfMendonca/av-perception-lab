"""Typed conversion of YAML mappings into option/config dataclasses.

`from_mapping` checks the *shape* of user input against a dataclass's type
hints (unknown or missing keys, wrong types, Literal choices, tuple sizes) and
reports errors with the dotted path of the offending key
(`detectors[1].treshold: unknown option`). Domain rules stay in each
dataclass's `__post_init__`; their ValueError is re-raised as ConfigError with
the section path. Shared by `config` (run sections) and `core.registry`
(adapter options), which keeps those two modules free of an import cycle.
"""

import dataclasses
import types
import typing
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, TypeVar, Union

T = TypeVar("T")


class ConfigError(ValueError):
    """Invalid user configuration (the CLI reports it and exits with code 2)."""


def _key(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def _type_name(tp: Any) -> str:
    return getattr(tp, "__name__", None) or str(tp).replace("typing.", "")


def _convert(tp: Any, value: Any, path: str) -> Any:
    """Convert one YAML value to the annotated type `tp` (see `from_mapping`)."""
    if tp is Any:
        return value
    origin, args = typing.get_origin(tp), typing.get_args(tp)
    if origin in (Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        errors = []
        for arg in (a for a in args if a is not type(None)):
            try:
                return _convert(arg, value, path)
            except ConfigError as exc:
                errors.append(exc)
        if len(errors) == 1:
            raise errors[0]
        raise ConfigError(f"{path}: expected {_type_name(tp)}, got {value!r}")
    if origin is Literal:
        if value in args:
            return value
        raise ConfigError(f"{path}: {value!r} is not one of {list(args)}")
    if dataclasses.is_dataclass(tp):
        return from_mapping(tp, value, path)
    if origin is tuple:
        if not isinstance(value, list | tuple):
            raise ConfigError(f"{path}: expected a list, got {value!r}")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(
                _convert(args[0], v, f"{path}[{i}]") for i, v in enumerate(value)
            )
        if len(value) != len(args):
            raise ConfigError(f"{path}: expected {len(args)} items, got {len(value)}")
        return tuple(
            _convert(a, v, f"{path}[{i}]") for i, (a, v) in enumerate(zip(args, value))
        )
    if origin in (dict, Mapping):
        if not isinstance(value, Mapping):
            raise ConfigError(f"{path}: expected a mapping, got {value!r}")
        out = {}
        for k, v in value.items():
            if not isinstance(k, str):
                raise ConfigError(f"{path}: keys must be strings, got {k!r}")
            out[k] = _convert(args[1], v, _key(path, k))
        return out
    if tp is bool:
        if isinstance(value, bool):
            return value
    elif tp is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    elif tp is float:
        # int -> float is the only coercion (YAML "30" for 30.0); bools are not numbers.
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
    elif tp is str:
        if isinstance(value, str):
            return value
    elif tp is Path:
        if isinstance(value, str | Path):
            return Path(value).expanduser()
    else:
        raise TypeError(f"from_mapping does not support the annotation {tp!r}")
    raise ConfigError(f"{path}: expected {_type_name(tp)}, got {value!r}")


def from_mapping(cls: type[T], data: Mapping[str, Any] | None, path: str) -> T:
    """Build dataclass `cls` from a YAML mapping (None = all defaults).

    Supported annotations: bool, int, float (int accepted), str, Path (from
    str, `~` expanded), Literal[...], nested dataclasses, Optional/unions of
    those (tried in order), tuple[X, ...], fixed-size tuple[X, Y], dict[str, X]
    / Mapping[str, X], and Any (value passed through).

    Raises:
        ConfigError: unknown or missing key, wrong type, or a ValueError from
            `cls.__post_init__`; the message starts with the dotted path.
    """
    if data is None:
        data = {}
    if not isinstance(data, Mapping):
        raise ConfigError(f"{path or 'config'}: expected a mapping, got {data!r}")
    fields = {f.name: f for f in dataclasses.fields(cls) if f.init}
    unknown = [k for k in data if k not in fields]
    if unknown:
        raise ConfigError(
            f"{_key(path, str(unknown[0]))}: unknown option; valid: {', '.join(fields)}"
        )
    hints = typing.get_type_hints(cls)
    kwargs = {}
    for name, f in fields.items():
        if name in data:
            kwargs[name] = _convert(hints[name], data[name], _key(path, name))
        elif (
            f.default is dataclasses.MISSING
            and f.default_factory is dataclasses.MISSING
        ):
            raise ConfigError(f"{_key(path, name)}: missing required option")
    try:
        return cls(**kwargs)
    except ValueError as exc:
        raise ConfigError(f"{path}: {exc}" if path else str(exc)) from None
