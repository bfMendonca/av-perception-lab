from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pytest

from av_perception_lab.core.options import ConfigError, from_mapping


@dataclass(frozen=True, slots=True)
class Inner:
    k: int = 1

    def __post_init__(self) -> None:
        if self.k < 0:
            raise ValueError("k must be >= 0")


@dataclass(frozen=True, slots=True)
class Outer:
    name: str
    flag: bool = False
    count: int = 0
    ratio: float = 0.5
    mode: Literal["a", "b"] = "a"
    path: Path | None = None
    inner: Inner = field(default_factory=Inner)
    maybe: Inner | None = None
    classes: str | tuple[str, ...] = "all"
    items: tuple[Inner, ...] = ()
    pairs: tuple[tuple[float, float], ...] = ()
    size: tuple[int, int] = (0, 0)
    weights: dict[str, float] = field(default_factory=dict)
    specs: tuple[Mapping[str, Any], ...] = ()


def test_defaults_and_primitives():
    out = from_mapping(Outer, {"name": "x", "flag": True, "count": 3, "ratio": 2}, "")
    assert out == Outer("x", flag=True, count=3, ratio=2.0)
    assert isinstance(out.ratio, float)


def test_unknown_and_missing_keys_report_dotted_path():
    with pytest.raises(ConfigError, match=r"^sec\.nme: unknown option; valid: name"):
        from_mapping(Outer, {"nme": "x"}, "sec")
    with pytest.raises(ConfigError, match=r"^sec\.name: missing required option"):
        from_mapping(Outer, {}, "sec")
    with pytest.raises(ConfigError, match=r"^sec\.items\[1\]\.j: unknown option"):
        from_mapping(Outer, {"name": "x", "items": [{"k": 1}, {"j": 2}]}, "sec")


@pytest.mark.parametrize(
    ("key", "value"),
    [("flag", 1), ("count", 1.5), ("count", True), ("ratio", False), ("name", 3)],
)
def test_wrong_primitive_types(key, value):
    data = {"name": "x", key: value}
    with pytest.raises(ConfigError, match=rf"^{key}: expected"):
        from_mapping(Outer, data, "")


def test_literal():
    assert from_mapping(Outer, {"name": "x", "mode": "b"}, "").mode == "b"
    with pytest.raises(ConfigError, match=r"mode: 'c' is not one of \['a', 'b'\]"):
        from_mapping(Outer, {"name": "x", "mode": "c"}, "")


def test_path_expands_user():
    out = from_mapping(Outer, {"name": "x", "path": "~/dev/w.pt"}, "")
    assert out.path == Path.home() / "dev" / "w.pt"


def test_nested_and_optional_dataclasses():
    out = from_mapping(Outer, {"name": "x", "inner": {"k": 4}, "maybe": None}, "")
    assert out.inner == Inner(4) and out.maybe is None
    assert from_mapping(Outer, {"name": "x", "maybe": {}}, "").maybe == Inner()


def test_str_or_tuple():
    assert from_mapping(Outer, {"name": "x", "classes": "car,bus"}, "").classes == (
        "car,bus"
    )
    out = from_mapping(Outer, {"name": "x", "classes": ["car", "bus"]}, "")
    assert out.classes == ("car", "bus")
    with pytest.raises(ConfigError, match="classes"):
        from_mapping(Outer, {"name": "x", "classes": 3}, "")


def test_variadic_nested_and_fixed_tuples():
    out = from_mapping(
        Outer,
        {"name": "x", "items": [{"k": 2}], "pairs": [[1, 2.5]], "size": [3, 4]},
        "",
    )
    assert out.items == (Inner(2),)
    assert out.pairs == ((1.0, 2.5),)
    assert out.size == (3, 4)
    with pytest.raises(ConfigError, match=r"^size: expected 2 items, got 3"):
        from_mapping(Outer, {"name": "x", "size": [1, 2, 3]}, "")
    with pytest.raises(ConfigError, match=r"^pairs\[0\]\[1\]: expected float"):
        from_mapping(Outer, {"name": "x", "pairs": [[1, "a"]]}, "")


def test_mappings():
    out = from_mapping(
        Outer,
        {"name": "x", "weights": {"a": 1, "b": 0.5}, "specs": [{"type": "t"}]},
        "",
    )
    assert out.weights == {"a": 1.0, "b": 0.5}
    assert isinstance(out.weights["a"], float)
    assert out.specs == ({"type": "t"},)
    with pytest.raises(ConfigError, match=r"^weights\.a: expected float"):
        from_mapping(Outer, {"name": "x", "weights": {"a": "heavy"}}, "")
    with pytest.raises(ConfigError, match=r"^specs\[0\]: expected a mapping"):
        from_mapping(Outer, {"name": "x", "specs": ["t"]}, "")


def test_post_init_value_error_becomes_config_error_with_path():
    with pytest.raises(ConfigError, match=r"^sec\.inner: k must be >= 0"):
        from_mapping(Outer, {"name": "x", "inner": {"k": -1}}, "sec")


def test_none_and_non_mapping_input():
    assert from_mapping(Inner, None, "inner") == Inner()
    with pytest.raises(ConfigError, match=r"^inner: expected a mapping"):
        from_mapping(Inner, [1], "inner")
