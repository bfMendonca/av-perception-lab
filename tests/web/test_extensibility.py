"""Proof of the future-proof requirement: models and stages registered next to
the pipeline code appear in the catalog with no change under web/."""

import ast
import dataclasses
import re
from dataclasses import dataclass, field
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from av_perception_lab.core import registry
from av_perception_lab.core.components import Component
from av_perception_lab.web import schema

WEB_SRC = Path(schema.__file__).parent


class FakeDetector(Component):
    """A detector the web package has never heard of."""

    @dataclass(frozen=True, slots=True, kw_only=True)
    class Options:
        zoom: float = field(
            default=1.5, metadata={"ui": {"min": 0, "max": 10, "advanced": True}}
        )
        plain: int = 3


def test_new_adapter_appears_in_catalog_without_web_changes(monkeypatch):
    registry.load_adapters()
    monkeypatch.setitem(registry._REGISTRY["detector"], "fake", FakeDetector)
    entry = schema.module_catalog()["modules"]["detector"]["fake"]
    assert [f["name"] for f in entry["schema"]["fields"]] == ["zoom", "plain"]
    assert entry["doc"] == "A detector the web package has never heard of."


def test_new_stage_appears_in_catalog_in_order(monkeypatch):
    monkeypatch.setattr(registry, "_STAGES", dict(registry._STAGES))
    before = [s["yaml_key"] for s in schema.module_catalog()["stages"]]
    registry.register_stage(
        registry.Stage(
            yaml_key="fake_stage",
            title="Fake",
            order=55,
            config_cls=FakeDetector.Options,
        )
    )
    catalog = schema.module_catalog()
    keys = [s["yaml_key"] for s in catalog["stages"]]
    orders = {s["yaml_key"]: s["order"] for s in catalog["stages"]}
    i = keys.index("fake_stage")
    assert orders[keys[i - 1]] < 55 < orders[keys[i + 1]]
    assert [k for k in keys if k != "fake_stage"] == before
    assert "fake_stage" in catalog["sections"]


def test_stage_group_flows_to_catalog(monkeypatch):
    monkeypatch.setattr(registry, "_STAGES", dict(registry._STAGES))
    registry.register_stage(
        registry.Stage(
            yaml_key="fake_grouped",
            title="Fake",
            order=56,
            config_cls=FakeDetector.Options,
            group="Fake group",
        )
    )
    stages = {s["yaml_key"]: s for s in schema.module_catalog()["stages"]}
    assert stages["fake_grouped"]["group"] == "Fake group"
    assert stages["depth"]["group"] is None


def test_duplicate_stage_key_rejected():
    with pytest.raises(ValueError, match="already registered"):
        registry.register_stage(registry.Stage(yaml_key="depth", title="x", order=1))


def test_ui_metadata_flows_to_schema_hint():
    f = next(
        f
        for f in schema.options_schema(FakeDetector.Options)["fields"]
        if f["name"] == "zoom"
    )
    assert f["hint"] == {"min": 0, "max": 10, "advanced": True}


def test_field_without_ui_metadata_has_empty_hint():
    f = next(
        f
        for f in schema.options_schema(FakeDetector.Options)["fields"]
        if f["name"] == "plain"
    )
    assert f["hint"] == {}


def test_catalog_carries_version():
    assert schema.module_catalog()["catalog_version"] == 1


def test_builtin_stages_cover_the_run_config():
    from av_perception_lab.config import RunConfig

    keys = {s["yaml_key"] for s in schema.module_catalog()["stages"]}
    fields = {f.name for f in dataclasses.fields(RunConfig)}
    assert keys <= fields


def _string_literals(path: Path) -> set[str]:
    """String constants of a module, docstrings excluded."""
    tree = ast.parse(path.read_text())
    docstrings = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (
            isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef)
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
        ):
            docstrings.add(id(body[0].value))
    return {
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant)
        and isinstance(n.value, str)
        and id(n) not in docstrings
    }


def test_web_package_has_no_hardcoded_adapter_names():
    names = {
        type_name for classes in registry.available().values() for type_name in classes
    }
    assert "rfdetr" in names
    for path in WEB_SRC.glob("*.py"):
        clash = _string_literals(path) & names
        assert not clash, f"{path.name} hard-codes adapter names {sorted(clash)}"


# The run contract of cli.run (config.RunConfig input/output), not stages.
RUN_CONTRACT_KEYS = {"source", "output"}
FRONT_END_SRC = WEB_SRC.parents[2] / "web" / "src"
_TS_STRING = re.compile(r"""(["'`])([A-Za-z0-9_.\-]+)\1""")


def test_front_end_has_no_hardcoded_stage_or_model_names():
    names = {t for classes in registry.available().values() for t in classes}
    keys = {s.yaml_key for s in registry.stages()} - RUN_CONTRACT_KEYS
    assert "rfdetr" in names and "detectors" in keys
    sources = [
        p
        for p in FRONT_END_SRC.rglob("*.ts*")
        if ".test." not in p.name and "test" not in p.relative_to(FRONT_END_SRC).parts
    ]
    assert sources, f"no front-end sources under {FRONT_END_SRC}"
    for path in sources:
        literals = {m.group(2) for m in _TS_STRING.finditer(path.read_text())}
        clash = literals & (names | keys)
        assert not clash, f"{path.name} hard-codes {sorted(clash)}"
