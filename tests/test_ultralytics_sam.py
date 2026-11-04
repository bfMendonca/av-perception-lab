"""Ultralytics SAM refiner helpers and the shared Ultralytics environment
policy, without ultralytics or torch."""

import os
import sys

import numpy as np
import pytest

from av_perception_lab.adapters._ultralytics_env import apply_ultralytics_env
from av_perception_lab.adapters.refiners.ultralytics_sam import (
    _UltralyticsFrame,
    split_flat,
)
from av_perception_lab.core import registry
from av_perception_lab.core.components import LoadContext
from av_perception_lab.core.options import ConfigError
from av_perception_lab.core.types import ContractError
from av_perception_lab.runtime import REPO

ENV = ("YOLO_OFFLINE", "YOLO_AUTOINSTALL", "YOLO_CONFIG_DIR")


def _build(**options):
    spec = {"type": "ultralytics_sam", **options}
    return registry.build("refiner", spec, path="refiners[0]")


@pytest.mark.parametrize("c", [1, 3])
def test_split_flat(c):
    k = 2
    masks = np.arange(k * c * 2 * 3).reshape(k * c, 2, 3) % 2 == 0
    scores = np.arange(k * c, dtype=np.float32)
    m, s = split_flat(masks, scores, k)
    assert m.shape == (k, c, 2, 3) and s.shape == (k, c)
    # Box-major: box 1's candidates follow box 0's.
    np.testing.assert_array_equal(m[1, 0], masks[c])
    assert s[1].tolist() == list(range(c, 2 * c))
    with pytest.raises(ContractError, match="for 2 boxes"):
        split_flat(masks[:-1], scores[:-1], k)


@pytest.mark.parametrize("weights", ["mobile_sam.pt", "sam2.1_l.pt", "x/sam_b.pt"])
def test_known_weights_names(weights):
    refiner = _build(weights=weights)
    assert refiner.name == weights.split("/")[-1][:-3] and not refiner.is_loaded
    assert refiner.options.imgsz == 1024 and refiner.has_object_score is False


@pytest.mark.parametrize("weights", ["sam3.pt", "foo.pt"])
def test_unknown_weights_names_are_refused(weights):
    with pytest.raises(
        ConfigError, match=r"^refiners\[0\]\.weights: .*not an Ultralytics"
    ):
        _build(weights=weights)


@pytest.mark.parametrize(
    ("options", "match"),
    [
        # F1: a base field checked through an Options with its own __post_init__.
        ({"max_boxes": 0}, "max_boxes must be >= 1"),
        ({"imgsz": 33}, "imgsz must be a positive multiple of 32"),
        ({"score": "object"}, "score: ultralytics_sam has no object score"),
    ],
)
def test_options_are_validated(options, match):
    with pytest.raises(ConfigError, match=rf"^refiners\[0\].*{match}"):
        _build(weights="mobile_sam.pt", **options)


def test_missing_weights_fail_before_importing_ultralytics(tmp_path):
    refiner = _build(weights=str(tmp_path / "mobile_sam.pt"))
    with pytest.raises(ConfigError, match=r"(?s)^weights: file not found: .*curl -fL"):
        refiner.load(LoadContext("cpu", (100, 50), None))
    assert "ultralytics" not in sys.modules and not refiner.is_loaded


class _Tensor:
    """The few torch.Tensor methods `_decode` calls, over a numpy array."""

    def __init__(self, array):
        self.array = array

    def __getitem__(self, index):
        return _Tensor(self.array[index])

    def float(self):
        return _Tensor(self.array.astype(np.float32))

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class _ScalingPredictor:
    """Mimics Predictor.inference_features on CPU: torch.as_tensor shares memory
    with a float32 array and `_prepare_prompts` scales the boxes in place."""

    def inference_features(self, features, src_shape, bboxes, multimask_output):
        bboxes *= 0.5
        k, (h, w) = len(bboxes), src_shape
        masks = np.ones((k, h, w), dtype=bool)
        boxes = np.column_stack([bboxes, np.full(k, 0.9), np.arange(k)])
        return _Tensor(masks), _Tensor(boxes)


def test_decode_does_not_scale_the_callers_boxes():
    refiner = _build(weights="mobile_sam.pt")
    refiner._predictor = _ScalingPredictor()
    xyxy = np.array([[0, 0, 4, 2], [2, 1, 6, 3]], dtype=np.float32)
    decoded = refiner._decode(_UltralyticsFrame(None, (4, 8), xyxy), slice(0, 2))
    np.testing.assert_array_equal(xyxy, [[0, 0, 4, 2], [2, 1, 6, 3]])
    assert decoded.masks.shape == (2, 1, 4, 8) and decoded.iou.shape == (2, 1)
    decoded.validate(2, 8, 4)


def test_apply_ultralytics_env(monkeypatch):
    for name in ENV:
        monkeypatch.delenv(name, raising=False)
    apply_ultralytics_env()
    expected = {
        "YOLO_OFFLINE": "1",
        "YOLO_AUTOINSTALL": "False",
        "YOLO_CONFIG_DIR": str(REPO / "_testbed" / "config"),
    }
    assert {n: os.environ[n] for n in ENV} == expected
    apply_ultralytics_env()  # idempotent
    assert {n: os.environ[n] for n in ENV} == expected
    # Values already in the environment win (setdefault).
    monkeypatch.setenv("YOLO_CONFIG_DIR", "/elsewhere")
    apply_ultralytics_env()
    assert os.environ["YOLO_CONFIG_DIR"] == "/elsewhere"
    assert "ultralytics" not in sys.modules
