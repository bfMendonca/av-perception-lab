"""Every registered detector/semantic adapter honours the `declared_classes`
contract: None or a non-empty catalog of non-empty names, never raises, and
never imports a framework."""

import sys

import pytest

from av_perception_lab.core import registry
from av_perception_lab.core.components import ClassCatalog
from av_perception_lab.core.options import ConfigError

registry.load_adapters()
CASES = [
    (kind, name)
    for kind in ("detector", "semantic")
    for name in sorted(registry.available(kind).get(kind, {}))
]
FRAMEWORKS = ("torch", "rfdetr", "ultralytics", "transformers", "timm")

# Options with required fields (e.g. YOLO `weights`) get a made-up value: the
# declaration must be None (nothing recorded) or valid, never an exception.
FILLERS = {"weights": "no-such-weights.pt", "model": "acme/none"}


def _options(cls):
    import dataclasses

    fields = {
        f.name: FILLERS.get(f.name, "x")
        for f in dataclasses.fields(cls.Options)
        if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
    }
    try:
        return cls.Options(**fields)
    except (ConfigError, TypeError, ValueError):
        pytest.skip("options need values this generic test cannot invent")


def test_registry_is_not_empty():
    assert CASES


@pytest.mark.parametrize(("kind", "name"), CASES)
def test_declared_classes_contract(kind, name, tmp_path, monkeypatch):
    from av_perception_lab.core import requirements as rq

    monkeypatch.setattr(rq, "_explicit_root", tmp_path / "weights")
    cls = registry.available(kind)[kind][name]
    before = {m for m in FRAMEWORKS if m in sys.modules}
    got = cls.declared_classes(_options(cls))
    assert got is None or (
        isinstance(got, ClassCatalog)
        and isinstance(got.names, tuple)
        and got.names
        and all(isinstance(n, str) and n for n in got.names)
    )
    assert {m for m in FRAMEWORKS if m in sys.modules} == before
