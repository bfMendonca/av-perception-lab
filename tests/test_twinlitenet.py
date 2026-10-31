"""TwinLiteNet+ adapter helpers (letterbox, head merge, import guard) without torch."""

import sys
import types

import numpy as np
import pytest

from det_depth_distance.adapters.semantic.twinlitenet import (
    TWINLITENET_CLASSES,
    _import_network,
    _letterbox,
    _merge_heads,
)
from det_depth_distance.core import registry
from det_depth_distance.core.components import LoadContext
from det_depth_distance.core.options import ConfigError


def test_letterbox_1080p_matches_the_demo():
    image, scale, pad = _letterbox(np.zeros((1080, 1920, 3), np.uint8), 640)
    # 1920x1080 -> 640x360, padded to 384 rows (12 + 12), demoDataset.letterbox_for_img.
    assert image.shape == (384, 640, 3) and pad == (0, 12)
    assert scale == pytest.approx(1 / 3)
    assert (image[:12] == 114).all() and (image[12:372] == 0).all()


def test_letterbox_non_integer_ratio():
    image, scale, pad = _letterbox(np.zeros((1080, 1440, 3), np.uint8), 640)
    assert image.shape[1] % 32 == 0 and image.shape[0] % 32 == 0
    assert scale == pytest.approx(640 / 1440) and 1 / scale != int(1 / scale)
    assert image.shape == (480, 640, 3) and pad == (0, 0)


def test_merge_heads_lane_wins():
    drivable = np.array([[0, 1], [1, 0]])
    lane = np.array([[0, 1], [0, 1]])
    labels = _merge_heads(drivable, lane)
    assert labels.dtype == np.int32 and labels.tolist() == [[0, 2], [1, 2]]
    assert TWINLITENET_CLASSES[2] == "lane"


def test_options_and_default_name():
    seg = registry.build(
        "semantic", {"type": "twinlitenet", "name": "lanes"}, path="semantic[0]"
    )
    assert seg.name == "lanes" and seg.class_names == TWINLITENET_CLASSES
    assert seg.options.repo is None  # the pinned clone under the weights root
    assert seg.default_name() == "twinlitenet-large"
    with pytest.raises(ConfigError, match=r"^semantic\[0\]: imgsz must be"):
        registry.build(
            "semantic", {"type": "twinlitenet", "imgsz": 100}, path="semantic[0]"
        )


def test_missing_clone_or_weights(tmp_path):
    ctx = LoadContext("cpu", (100, 50), None)
    seg = registry.build(
        "semantic", {"type": "twinlitenet", "repo": str(tmp_path)}, path="semantic[0]"
    )
    with pytest.raises(ConfigError, match="^repo: no TwinLiteNetPlus clone"):
        seg.load(ctx)
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "model.py").write_text("")
    with pytest.raises(ConfigError, match=r"^weights: file not found: .*large\.pth"):
        seg.load(ctx)
    assert "torch" not in sys.modules


def test_import_refuses_a_foreign_model_module(tmp_path, monkeypatch):
    foreign = types.ModuleType("model")
    foreign.__path__ = [str(tmp_path / "elsewhere")]
    monkeypatch.setitem(sys.modules, "model", foreign)
    with pytest.raises(ConfigError, match="another top-level module 'model'"):
        _import_network(tmp_path)


def test_import_from_the_clone_only_while_importing(tmp_path, monkeypatch):
    package = tmp_path / "model"
    package.mkdir()
    (package / "config.py").write_text("variants = ('large',)\n")
    (package / "model.py").write_text(
        "from model import config as cfg\nclass TwinLiteNetPlus:\n    cfg = cfg\n"
    )
    names = ("model", "model.model", "model.config")
    for name in names:
        monkeypatch.delitem(sys.modules, name, raising=False)
    try:
        network = _import_network(tmp_path)
        assert network.__name__ == "TwinLiteNetPlus"
        assert network.cfg.variants == ("large",)
        assert str(tmp_path) not in sys.path
        assert _import_network(tmp_path) is network  # same clone: reused
    finally:
        for name in names:  # the fake clone's modules must not leak into other tests
            sys.modules.pop(name, None)
