"""SegFormer adapter options and class-name helper without transformers."""

import sys

import pytest

from det_depth_distance.adapters.semantic.segformer import (
    SegformerSegmenter,
    _id2label_names,
)
from det_depth_distance.core import registry
from det_depth_distance.core.options import ConfigError
from det_depth_distance.core.types import ContractError


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
