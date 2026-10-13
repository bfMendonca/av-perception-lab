import json
from dataclasses import dataclass

import numpy as np
import pytest
import yaml

from det_depth_distance.core import registry
from det_depth_distance.core.components import (
    Detector,
    DetectorOptions,
    LoadContext,
    MaskRefiner,
    RefinerOptions,
    SemanticSegmenter,
)
from det_depth_distance.core.labels import COCO
from det_depth_distance.core.options import ConfigError
from det_depth_distance.core.types import (
    ContractError,
    Frame,
    Instances,
    RefinedMasks,
    SemanticMap,
)

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


def _label_map(tmp_path, **data):
    path = tmp_path / "map.yaml"
    path.write_text(yaml.safe_dump({"canonical": {"vehicle": ["car"]}, **data}))
    return path


def test_label_map_maps_raw_labels(tmp_path):
    det = FakeDetector(DetectorOptions(classes="all", labels=_label_map(tmp_path)))
    det.load(CTX)
    out = det(FRAME)
    assert out.label == ("vehicle", "person", "dog")
    assert out.raw_label == ("car", "person", "dog")
    expected = set(COCO.presets["driving"]) - {"car"} | {"vehicle", "dog"}
    assert det.output_labels == expected
    desc = json.loads(json.dumps(det.describe()))
    assert desc["labels"]["canonical"] == {"vehicle": ["car"]}
    assert FakeDetector().describe()["labels"] is None


def test_label_map_drop_after_class_filter(tmp_path):
    path = _label_map(tmp_path, unmapped="drop")
    det = FakeDetector(DetectorOptions(labels=path))  # classes: driving drops "dog"
    det.load(CTX)
    out = det(FRAME)
    assert out.label == ("vehicle",) and out.source == ("fake",)
    assert det.output_labels == {"vehicle"}


def test_label_map_errors(tmp_path):
    with pytest.raises(ConfigError, match=r"^labels: .*label map not found"):
        FakeDetector(DetectorOptions(labels=tmp_path / "nope.yaml"))
    # Raw names and vocabulary are only checked against the loaded model.
    det = FakeDetector(
        DetectorOptions(labels=_label_map(tmp_path, vocabulary="objects365"))
    )
    with pytest.raises(ConfigError, match=r"^fake\.labels: map: written for"):
        det.load(CTX)
    det = FakeDetector(
        DetectorOptions(labels=_label_map(tmp_path, canonical={"vehicle": ["suv"]}))
    )
    with pytest.raises(ConfigError, match="'suv'"):
        det.load(CTX)


def test_label_map_errors_through_registry(tmp_path, monkeypatch):
    registry.load_adapters()
    monkeypatch.setitem(registry._REGISTRY["detector"], "fake", FakeDetector)
    path = tmp_path / "map.yaml"
    path.write_text("canonical: {vehicle: [car], truck: [car]}")
    spec = {"type": "fake", "labels": str(path)}
    with pytest.raises(
        ConfigError, match=r"^detectors\[1\]\.labels: .*'car' mapped to both"
    ):
        registry.build("detector", spec, path="detectors[1]")


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


def box_masks(xyxy: np.ndarray, width: int, height: int) -> np.ndarray:
    """(N, H, W) bool masks filling each box's pixel window."""
    masks = np.zeros((len(xyxy), height, width), dtype=bool)
    for i, (x0, y0, x1, y1) in enumerate(np.asarray(xyxy, dtype=int)):
        masks[i, y0:y1, x0:x1] = True
    return masks


class FakeRefiner(MaskRefiner):
    """Mask = the box rectangle, score 0.9; `empty` indexes (into the targets)
    get an empty mask; records the targets of every call."""

    type_name = "fake-refiner"
    Options = RefinerOptions

    def __init__(self, options=None, *, name="sam", empty=(), broken=False):
        super().__init__(options or RefinerOptions(), name=name)
        self.empty, self.broken = set(empty), broken
        self.calls: list[Instances] = []

    def _load(self, ctx):
        self.size = ctx.frame_size

    def _refine(self, rgb, instances):
        self.calls.append(instances)
        h, w = rgb.shape[:2]
        masks = box_masks(instances.xyxy, w, h)
        for j in self.empty:
            masks[j] = False
        if self.broken:
            masks = masks[:, :-1]
        return RefinedMasks(masks, np.full(len(instances), 0.9, np.float32))


class FakeSemantic(SemanticSegmenter):
    """`labels` (H, W) int32 on every frame, classes `names`."""

    type_name = "fake-semantic"

    @dataclass(frozen=True, slots=True)
    class Options:
        pass

    def __init__(self, labels, names, *, name="scene", log=None, map_names=None):
        super().__init__(self.Options(), name=name)
        self.labels, self.names, self.log = labels, tuple(names), log
        self.map_names = tuple(map_names or names)

    def _load(self, ctx):
        if self.log is not None:
            self.log.append(f"load {self.name}")

    def _close(self):
        if self.log is not None:
            self.log.append(f"close {self.name}")

    @property
    def class_names(self):
        return self.names

    def _segment(self, rgb):
        if self.log is not None:
            self.log.append(f"semantic {self.name}")
        return SemanticMap(self.labels, self.map_names)


@pytest.fixture
def mixed(make_instances):
    """Three boxes in the 100x50 FRAME: a native mask on the first only."""
    xyxy = [[0, 0, 10, 10], [20, 0, 40, 20], [50, 10, 60, 50]]
    masks = np.zeros((3, 50, 100), dtype=bool)
    masks[0, 2:6, 2:6] = True
    return make_instances(xyxy, source="seg", masks=masks)


def test_refiner_missing_keeps_native_masks(mixed):
    refiner = FakeRefiner()
    refiner.load(CTX)
    out = refiner(FRAME, mixed)
    assert len(refiner.calls) == 1 and len(refiner.calls[0]) == 2  # targets only
    np.testing.assert_array_equal(out.masks[0], mixed.masks[0])
    np.testing.assert_array_equal(out.masks[1:], box_masks(mixed.xyxy[1:], 100, 50))
    assert out.mask_source == ("seg", "sam", "sam")
    assert (
        np.isnan(out.mask_score[0])
        and out.mask_score[1:].tolist() == [pytest.approx(0.9)] * 2
    )
    # Same N, order and other fields.
    np.testing.assert_array_equal(out.xyxy, mixed.xyxy)
    assert out.source == mixed.source and out.label == mixed.label


def test_refiner_all_replaces_but_empty_never_overwrites(mixed):
    refiner = FakeRefiner(RefinerOptions(targets="all"), empty={0, 2})
    refiner.load(CTX)
    out = refiner(FRAME, mixed)
    assert len(refiner.calls[0]) == 3
    np.testing.assert_array_equal(out.masks[0], mixed.masks[0])  # empty: kept
    assert not out.masks[2].any()  # empty refined mask, no mask before
    assert out.mask_source == ("seg", "sam", "")
    assert out.mask_sources() == out.mask_source


def test_refiner_without_targets_is_not_called(make_instances):
    masks = box_masks(np.array([[0, 0, 10, 10]]), 100, 50)
    inst = make_instances([[0, 0, 10, 10]], source="seg", masks=masks)
    refiner = FakeRefiner()
    refiner.load(CTX)
    out = refiner(FRAME, inst)
    assert refiner.calls == []
    assert out.mask_source == ("seg",) and np.isnan(out.mask_score).all()
    empty = refiner(FRAME, Instances.empty())
    assert refiner.calls == [] and len(empty) == 0 and empty.mask_source == ()


def test_refiner_fills_box_only_instances(make_instances):
    inst = make_instances([[0, 0, 10, 10], [20, 0, 40, 20]], source="det")
    refiner = FakeRefiner()
    refiner.load(CTX)
    out = refiner(FRAME, inst)
    assert out.masks.shape == (2, 50, 100) and out.has_mask().all()
    assert out.mask_sources() == ("sam", "sam")


def test_chained_refiners_keep_previous_scores(make_instances):
    inst = make_instances([[0, 0, 10, 10]], source="det")
    first, second = FakeRefiner(name="a"), FakeRefiner(name="b")
    first.load(CTX)
    second.load(CTX)
    out = second(FRAME, first(FRAME, inst))
    assert second.calls == []  # "missing": the first refiner filled it
    assert out.mask_source == ("a",) and out.mask_score.tolist() == [pytest.approx(0.9)]


def test_broken_refiner_and_use_before_load(mixed):
    refiner = FakeRefiner(broken=True)
    with pytest.raises(ContractError, match="before load"):
        refiner(FRAME, mixed)
    refiner.load(CTX)
    with pytest.raises(ContractError, match="masks: shape"):
        refiner(FRAME, mixed)


def test_semantic_segmenter_contract():
    labels = np.zeros((50, 100), np.int32)
    seg = FakeSemantic(labels, ["road", "sky"])
    with pytest.raises(ContractError, match="before load"):
        seg(FRAME)
    seg.load(CTX)
    out = seg(FRAME)
    assert out.class_names == ("road", "sky") and out.labels is labels
    bad = FakeSemantic(labels[:10], ["road"])
    bad.load(CTX)
    with pytest.raises(ContractError, match="shape"):
        bad(FRAME)
    other = FakeSemantic(labels, ["road", "sky"], map_names=["a", "b"])
    other.load(CTX)
    with pytest.raises(ContractError, match="class_names differ"):
        other(FRAME)
