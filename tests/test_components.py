import json
from dataclasses import dataclass

import numpy as np
import pytest

from det_depth_distance.core import registry
from det_depth_distance.core.components import (
    Detector,
    DetectorOptions,
    LoadContext,
)
from det_depth_distance.core.labels import COCO
from det_depth_distance.core.options import ConfigError
from det_depth_distance.core.types import ContractError, Frame, Instances

CTX = LoadContext("cpu", (100, 50), None)
FRAME = Frame(0, 0.0, np.zeros((50, 100, 3), np.uint8))


class FakeDetector(Detector):
    """Returns a car, a person and a dog (COCO names) on every frame."""

    type_name = "fake"
    Options = DetectorOptions

    def __init__(self, options=None, *, name=None, boxes=None):
        super().__init__(options or DetectorOptions(), name=name)
        self.boxes = boxes if boxes is not None else [[0, 0, 10, 10]] * 3
        self.loads = 0

    def _load(self, ctx):
        self.loads += 1

    @property
    def class_names(self):
        return [*COCO.presets["driving"], "dog"]

    def _predict(self, rgb):
        return self._instances(
            self.boxes, [0.9, 0.8, 0.7], [2, 0, 16], ["car", "person", "dog"]
        )


def test_filter_labels_and_source():
    det = FakeDetector(name="cam")
    det.load(CTX)
    out = det(FRAME)
    assert out.raw_label == ("car", "person") == out.label
    assert out.source == ("cam", "cam")
    assert det.vocabulary == "coco"

    every = FakeDetector(DetectorOptions(classes="all"))
    every.load(CTX)
    assert len(every(FRAME)) == 3


def test_class_errors_surface_at_load():
    det = FakeDetector(DetectorOptions(classes=("car", "tractor")))
    with pytest.raises(ConfigError, match="tractor"):
        det.load(CTX)


def test_broken_adapter_output_is_a_contract_error():
    det = FakeDetector(boxes=[[0, 0, 200, 10]] * 3)  # outside the 100x50 frame
    det.load(CTX)
    with pytest.raises(ContractError, match="outside"):
        det(FRAME)


def test_use_before_load_and_reload():
    det = FakeDetector()
    with pytest.raises(ContractError, match="before load"):
        det(FRAME)
    det.load(CTX)
    det.load(CTX)  # idempotent
    assert det.loads == 1 and det.is_loaded
    det.close()
    assert not det.is_loaded
    det.load(CTX)
    assert det.loads == 2


def test_describe_is_json_ready():
    det = FakeDetector(name="cam")
    det.load(CTX)
    desc = json.loads(json.dumps(det.describe()))
    assert desc["type"] == "fake" and desc["name"] == "cam"
    assert desc["options"]["threshold"] == 0.5 and desc["vocabulary"] == "coco"


def test_default_name_is_type_name():
    assert FakeDetector().name == "fake"


def test_labels_option_is_refused_until_task_0003(tmp_path):
    with pytest.raises(ConfigError, match="labels"):
        FakeDetector(DetectorOptions(labels=tmp_path / "map.yaml"))


@pytest.mark.parametrize("threshold", [0, 1.5])
def test_detector_options_threshold(threshold):
    with pytest.raises(ValueError, match="threshold"):
        DetectorOptions(threshold=threshold)


@dataclass(frozen=True, slots=True, kw_only=True)
class RequiredOptions(DetectorOptions):
    """An adapter's Options with a required field and its own __post_init__."""

    weights: str
    size: int = 32

    def __post_init__(self) -> None:
        DetectorOptions.__post_init__(self)  # never bare super() in slots dataclasses
        if self.size % 32:
            raise ValueError("size must be a multiple of 32")


class RequiredDetector(FakeDetector):
    type_name = "required-fake"
    Options = RequiredOptions


def test_layered_options_run_the_base_validation():
    assert RequiredOptions(weights="w").threshold == 0.5
    with pytest.raises(TypeError):
        RequiredOptions()  # weights is required (kw_only allows it after defaults)
    with pytest.raises(ValueError, match="threshold"):
        RequiredOptions(weights="w", threshold=0)
    with pytest.raises(ValueError, match="size"):
        RequiredOptions(weights="w", size=10)


def test_layered_options_errors_through_registry(monkeypatch):
    registry.load_adapters()
    monkeypatch.setitem(
        registry._REGISTRY["detector"], "required-fake", RequiredDetector
    )
    spec = {"type": "required-fake", "weights": "w", "threshold": 0}
    with pytest.raises(ConfigError, match=r"^detectors\[0\]: threshold must be"):
        registry.build("detector", spec, path="detectors[0]")


def test_instances_helper_dtypes():
    det = FakeDetector(name="x")
    inst = det._instances([[0, 0, 1, 1]], [0.5], [3], ["car"])
    assert isinstance(inst, Instances)
    assert (inst.xyxy.dtype, inst.score.dtype, inst.class_id.dtype) == (
        np.float32,
        np.float32,
        np.int64,
    )
    inst.validate(10, 10)
