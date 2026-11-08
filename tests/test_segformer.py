"""SegFormer adapter options and class-name helper without transformers."""

import json
import sys

import pytest

from av_perception_lab.adapters.semantic.segformer import (
    SegformerSegmenter,
    _id2label_names,
)
from av_perception_lab.core import registry
from av_perception_lab.core.components import ClassCatalog
from av_perception_lab.core.options import ConfigError
from av_perception_lab.core.types import ContractError


def test_options_defaults():
    options = SegformerSegmenter.Options()
    assert options.model == "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
    assert options.size == (1024, 1024) and options.hf_cache is None


def test_built_through_the_registry():
    spec = {"type": "segformer", "name": "scene", "size": [1024, 1792]}
    seg = registry.build("semantic", spec, path="semantic[0]")
    assert seg.name == "scene" and seg.options.size == (1024, 1792)
    assert not seg.is_loaded
    assert seg.default_name() == "segformer-b5-finetuned-cityscapes-1024-1024"
    with pytest.raises(ConfigError, match=r"^semantic\[0\]\.colour: unknown option"):
        registry.build(
            "semantic", {"type": "segformer", "colour": 1}, path="semantic[0]"
        )
    with pytest.raises(ConfigError, match=r"^semantic\[0\]: size must be multiples"):
        registry.build(
            "semantic", {"type": "segformer", "size": [1000, 1024]}, path="semantic[0]"
        )


def test_id2label_names():
    assert _id2label_names({"2": "c", "0": "a", "1": "b"}) == ("a", "b", "c")
    assert _id2label_names({0: "road", 1: "sky"}) == ("road", "sky")
    with pytest.raises(ContractError, match="not 0..C-1"):
        _id2label_names({"0": "a", "2": "c"})
    assert "transformers" not in sys.modules


def _write_config(cache, id2label, repo="acme/seg"):
    snap = cache / ("models--" + repo.replace("/", "--")) / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text(json.dumps({"id2label": id2label}))


def _options(cache, model="acme/seg"):
    return SegformerSegmenter.Options(model=model, hf_cache=cache)


def test_segformer_declared_classes_from_config_json(tmp_path):
    _write_config(tmp_path, {"0": "a", "1": "b", "2": "c"})
    got = SegformerSegmenter.declared_classes(_options(tmp_path))
    assert got == ClassCatalog(("a", "b", "c"), "metadata")
    assert "transformers" not in sys.modules


def test_segformer_declared_classes_missing_file_is_none(tmp_path):
    assert SegformerSegmenter.declared_classes(_options(tmp_path)) is None


def test_segformer_declared_classes_invalid_ids_is_none(tmp_path):
    _write_config(tmp_path, {"0": "a", "2": "c"})
    assert SegformerSegmenter.declared_classes(_options(tmp_path)) is None


def test_segformer_declared_classes_invalid_json_is_none(tmp_path):
    _write_config(tmp_path, {"0": "a"})
    (next(tmp_path.rglob("config.json"))).write_text("{not json")
    assert SegformerSegmenter.declared_classes(_options(tmp_path)) is None


def test_segformer_declared_classes_local_directory(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"id2label": {"0": "x"}}))
    got = SegformerSegmenter.declared_classes(_options(None, str(tmp_path)))
    assert got == ClassCatalog(("x",), "metadata")
