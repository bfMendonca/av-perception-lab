import dataclasses
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest
import yaml

pytest.importorskip("fastapi")

from det_depth_distance.web import schema


def _field(sch, name):
    return next(f for f in sch["fields"] if f["name"] == name)


def test_catalog_lists_registered_adapters():
    catalog = schema.module_catalog()
    assert "rfdetr" in catalog["modules"]["detector"]
    assert "wbf" in catalog["modules"]["aggregator"]
    assert "schema" in catalog["modules"]["detector"]["rfdetr"]


def test_literal_becomes_enum_with_choices():
    @dataclass
    class Opts:
        layout: Literal["a", "b"] = "a"

    f = _field(schema.options_schema(Opts), "layout")
    assert f["type"] == "enum" and f["choices"] == ["a", "b"]
    assert f["default"] == "a" and f["required"] is False


def test_optional_field_is_nullable():
    @dataclass
    class Opts:
        focal: float | None = None
        plain: int = 1
        need: str = dataclasses.field(default="x")

    sch = schema.options_schema(Opts)
    assert _field(sch, "focal")["nullable"] is True
    assert _field(sch, "focal")["type"] == "float"
    assert _field(sch, "plain")["nullable"] is False


def test_required_field_and_path_type():
    @dataclass
    class Opts:
        path: Path

    f = _field(schema.options_schema(Opts), "path")
    assert f["required"] is True and f["type"] == "path" and f["default"] is None


def test_nested_dataclass_becomes_object_with_fields():
    catalog = schema.module_catalog()
    estimate = _field(catalog["sections"]["camera"]["schema"], "estimate")
    assert estimate["type"] == "object" and estimate["nullable"] is True
    names = [f["name"] for f in estimate["fields"]]
    assert "frames" in names
    assert estimate["default"]["frames"] == 16


def test_containers_and_unions():
    @dataclass
    class Opts:
        a: tuple[str, ...] = ()
        b: tuple[float, float] = (1.0, 2.0)
        c: dict[str, int] = dataclasses.field(default_factory=dict)
        d: str | tuple[str, ...] = ()

    sch = schema.options_schema(Opts)
    assert _field(sch, "a")["items"]["type"] == "str"
    assert _field(sch, "b")["length"] == 2
    assert _field(sch, "c")["type"] == "map"
    assert _field(sch, "d")["type"] == "any"
    assert len(_field(sch, "d")["variants"]) == 2


def test_unsupported_annotation_raises():
    @dataclass
    class Opts:
        x: set[int] = dataclasses.field(default_factory=set)

    with pytest.raises(TypeError, match="does not support"):
        schema.options_schema(Opts)


def test_smoothing_section_exposes_accel_std_float():
    catalog = schema.module_catalog()
    f = _field(catalog["sections"]["smoothing"]["schema"], "accel_std")
    assert f["type"] == "float" and f["default"] == 2.0
    assert f["hint"]["unit"] == "m/s²"


def test_tracker_fields_carry_ui_hints():
    catalog = schema.module_catalog()
    sch = catalog["modules"]["tracker"]["iou"]["schema"]
    assert _field(sch, "iou_thr")["hint"]["max"] == 0.95


def test_validate_reports_dotted_path(recommended_yaml):
    data = yaml.safe_load(recommended_yaml)
    data["detectors"][0]["treshold"] = 0.5
    errors = schema.validate_config(data)
    assert any(e["path"] == "detectors[0].treshold" for e in errors)


def test_validate_collects_independent_errors(recommended_yaml):
    data = yaml.safe_load(recommended_yaml)
    data["detectors"][0]["type"] = "nope"
    data["depth"]["bogus"] = 1
    paths = {e["path"] for e in schema.validate_config(data)}
    assert {"detectors[0].type", "depth.bogus"} <= paths


def test_validate_ok_for_recommended_config(recommended_yaml):
    assert schema.validate_config(yaml.safe_load(recommended_yaml)) == []


def test_validate_does_not_mutate_input(recommended_yaml):
    data = yaml.safe_load(recommended_yaml)
    before = yaml.safe_dump(data)
    schema.validate_config(data)
    assert yaml.safe_dump(data) == before


def test_server_imports_without_torch():
    code = (
        "import sys\n"
        "import det_depth_distance.web.app as app\n"
        "from det_depth_distance.web import schema\n"
        "assert schema.module_catalog()['modules']['detector']\n"
        "cfg = {'detectors': [{'type': 'rfdetr'}], 'depth': {'type': 'da3'}}\n"
        "assert schema.validate_config(cfg) == [], schema.validate_config(cfg)\n"
        "assert schema.validate_config({'detectors': [{'type': 'x'}]})\n"
        "assert 'torch' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-I", "-c", code], check=True)


def test_distance_section_includes_lifted_run_config_fields():
    catalog = schema.module_catalog()
    sch = catalog["sections"]["distance"]["schema"]
    assert _field(sch, "box_shrink")["type"] == "float"
    assert _field(sch, "exclude_sky")["type"] == "bool"
    region = _field(sch, "region")
    assert region["type"] == "enum" and "mask" in region["choices"]
    assert _field(sch, "exclude_semantic")["type"] == "any"
    assert [f["name"] for f in sch["fields"]].count("region") == 1
