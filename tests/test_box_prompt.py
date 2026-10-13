"""BoxPromptRefiner (the framework-agnostic base of the SAM-family refiners)
with a fake box-prompted model: batching, single frame preparation, mask
choice, object score, and the MaskRefiner contract (no torch)."""

import math

import numpy as np
import pytest
from test_components import box_masks

from det_depth_distance.adapters.refiners.box_prompt import (
    BoxPromptOptions,
    BoxPromptRefiner,
    DecodedMasks,
    box_batches,
    select_masks,
)
from det_depth_distance.core import registry
from det_depth_distance.core.components import LoadContext
from det_depth_distance.core.options import ConfigError
from det_depth_distance.core.types import ContractError, Frame

CTX = LoadContext("cpu", (100, 50), None)
FRAME = Frame(0, 0.0, np.zeros((50, 100, 3), np.uint8))
BOXES = [[0, 0, 10, 10], [20, 0, 40, 20], [50, 10, 60, 50]]


class FakeBoxModel(BoxPromptRefiner):
    """Not registered. Candidate c of a box is its rectangle shrunk by 1 + c px;
    `iou` (N, C) and `logits` (N,) are indexed by target position."""

    type_name = "fake-box"
    Options = BoxPromptOptions

    def __init__(self, options=None, *, c=1, iou=None, logits=None, name="box"):
        super().__init__(options or BoxPromptOptions(), name=name)
        self.c, self.iou, self.logits = c, iou, logits
        self.prepared = self.decoded = 0

    def _load(self, ctx):
        pass

    def _prepare(self, rgb, xyxy):
        self.prepared += 1
        return rgb.shape[:2], xyxy

    def _decode(self, state, batch):
        self.decoded += 1
        (h, w), xyxy = state
        boxes = xyxy[batch]
        masks = np.stack(
            [
                box_masks(boxes + [1 + c, 1 + c, -1 - c, -1 - c], w, h)
                for c in range(self.c)
            ],
            axis=1,
        )
        n = len(xyxy)
        iou = self.iou if self.iou is not None else np.full((n, self.c), 0.8)
        logit = None
        if self.has_object_score:
            logit = (self.logits if self.logits is not None else np.ones(n))[batch]
            logit = np.asarray(logit, np.float32)
        return DecodedMasks(masks, np.asarray(iou, np.float32)[batch], logit)


class FakeObjectModel(FakeBoxModel):
    has_object_score = True


@pytest.fixture
def inst(make_instances):
    return make_instances(BOXES, source="det")


def test_box_batches():
    for n, count in [(0, 0), (1, 1), (32, 1), (33, 2)]:
        batches = box_batches(n, 32)
        assert len(batches) == count
        covered = [i for b in batches for i in range(b.start, b.stop)]
        assert covered == list(range(n))
    assert box_batches(3, 1) == [slice(0, 1), slice(1, 2), slice(2, 3)]


def test_batch_size_does_not_change_the_result(inst):
    out = []
    for max_boxes in (1, 32):
        refiner = FakeBoxModel(BoxPromptOptions(max_boxes=max_boxes), c=3)
        refiner.load(CTX)
        out.append(refiner(FRAME, inst))
        assert refiner.prepared == 1  # one frame preparation, reused by every batch
        assert refiner.decoded == (3 if max_boxes == 1 else 1)
    np.testing.assert_array_equal(out[0].masks, out[1].masks)
    np.testing.assert_array_equal(out[0].mask_score, out[1].mask_score)


def test_no_target_no_preparation(make_instances):
    masks = box_masks(np.array(BOXES), 100, 50)
    native = make_instances(BOXES, source="seg", masks=masks)
    refiner = FakeBoxModel()
    refiner.load(CTX)
    out = refiner(FRAME, native)
    assert (refiner.prepared, refiner.decoded) == (0, 0)
    assert out.mask_source == ("seg",) * 3


def _decoded(iou, logit=None, c=3, k=2):
    masks = np.zeros((k, c, 4, 4), bool)
    for j in range(k):
        for i in range(c):
            masks[j, i, i, j] = True  # candidate i of box j: pixel (i, j)
    return DecodedMasks(
        masks,
        np.asarray(iou, np.float32),
        None if logit is None else np.asarray(logit, np.float32),
    )


def test_select_masks_by_predicted_iou():
    decoded = _decoded([[0.2, 0.9, 0.5], [0.7, 0.7, 0.1]])
    masks, scores = select_masks(decoded, score="iou", min_object_score=None)
    assert masks.shape == (2, 4, 4) and masks.dtype == bool
    assert masks[0, 1, 0] and masks[1, 0, 1]  # best IoU; tie -> first index
    assert scores.dtype == np.float32 and scores.tolist() == pytest.approx([0.9, 0.7])
    single = _decoded([[0.3], [0.4]], c=1)
    masks, scores = select_masks(single, score="iou", min_object_score=None)
    assert masks[0, 0, 0] and scores.tolist() == pytest.approx([0.3, 0.4])


def test_select_masks_object_score():
    decoded = _decoded([[0.9], [0.8]], logit=[2.0, -1.0], c=1)
    masks, scores = select_masks(decoded, score="object", min_object_score=None)
    assert scores.tolist() == pytest.approx([1 / (1 + math.exp(-2)), 1 / (1 + math.e)])
    masks, scores = select_masks(decoded, score="iou", min_object_score=0.5)
    assert masks[0].any() and not masks[1].any()  # sigmoid(-1) < 0.5: emptied
    assert scores.tolist() == pytest.approx([0.9, 0.8])


def test_min_object_score_keeps_the_previous_mask(make_instances):
    native = np.zeros((3, 50, 100), bool)
    native[2, 20:30, 52:58] = True
    inst = make_instances(BOXES, source="seg", masks=native)
    refiner = FakeObjectModel(
        BoxPromptOptions(targets="all", min_object_score=0.5, score="object"),
        logits=[3.0, 3.0, -3.0],
    )
    refiner.load(CTX)
    out = refiner(FRAME, inst)
    np.testing.assert_array_equal(out.masks[2], native[2])  # emptied -> native kept
    assert out.mask_source == ("box", "box", "seg")
    assert out.mask_score[0] == pytest.approx(1 / (1 + math.exp(-3)))
    assert np.isnan(out.mask_score[2])


def test_decoded_masks_validate():
    ok = _decoded([[0.1] * 3] * 2)
    ok.validate(2, 4, 4)
    with pytest.raises(ContractError, match="not \\(3, C, H, W\\)"):
        ok.validate(3, 4, 4)
    with pytest.raises(ContractError, match="size"):
        ok.validate(2, 5, 4)
    with pytest.raises(ContractError, match="expected bool"):
        DecodedMasks(ok.masks.astype(np.uint8), ok.iou).validate(2, 4, 4)
    with pytest.raises(ContractError, match="iou: shape"):
        DecodedMasks(ok.masks, ok.iou[:, :1]).validate(2, 4, 4)
    with pytest.raises(ContractError, match="object_logit: expected float32"):
        DecodedMasks(ok.masks, ok.iou, np.zeros(2)).validate(2, 4, 4)


def test_object_logit_must_match_has_object_score(inst):
    class Liar(FakeBoxModel):
        def _decode(self, state, batch):
            d = super()._decode(state, batch)
            return DecodedMasks(d.masks, d.iou, np.zeros(len(d.iou), np.float32))

    refiner = Liar()
    refiner.load(CTX)
    with pytest.raises(ContractError, match="object_logit must be given iff"):
        refiner(FRAME, inst)


def test_object_options_need_the_signal():
    with pytest.raises(ConfigError, match="^score: fake-box has no object score"):
        FakeBoxModel(BoxPromptOptions(score="object"))
    with pytest.raises(ConfigError, match="^min_object_score: fake-box has no"):
        FakeBoxModel(BoxPromptOptions(min_object_score=0.5))
    FakeObjectModel(BoxPromptOptions(score="object", min_object_score=0.5))


@pytest.mark.parametrize(
    ("option", "match"),
    [
        ({"max_boxes": 0}, "max_boxes must be >= 1"),
        ({"min_object_score": 1.5}, r"min_object_score must be in \[0, 1\]"),
        ({"mask_threshold": float("nan")}, "mask_threshold must be finite"),
        ({"targets": "some"}, r"targets: 'some' is not one of"),
    ],
)
def test_options_are_validated_through_the_registry(option, match):
    with pytest.raises(ConfigError, match=rf"^refiners\[0\].*{match}"):
        registry.build("refiner", {"type": "edgetam", **option}, path="refiners[0]")


# LSP: the MaskRefiner contract cases of test_components pass with this base too.
def test_contract_missing_keeps_native_masks(make_instances):
    native = np.zeros((3, 50, 100), bool)
    native[0, 2:6, 2:6] = True
    inst = make_instances(BOXES, source="seg", masks=native)
    refiner = FakeBoxModel()
    refiner.load(CTX)
    out = refiner(FRAME, inst)
    np.testing.assert_array_equal(out.masks[0], native[0])
    shrunk = box_masks(np.array(BOXES[1:]) + [1, 1, -1, -1], 100, 50)
    np.testing.assert_array_equal(out.masks[1:], shrunk)
    assert out.mask_source == ("seg", "box", "box")
    np.testing.assert_array_equal(out.xyxy, inst.xyxy)
    assert out.label == inst.label and len(out) == 3


def test_contract_box_only_instances_and_use_before_load(inst):
    refiner = FakeBoxModel(c=3, iou=np.tile([0.1, 0.3, 0.2], (3, 1)))
    with pytest.raises(ContractError, match="before load"):
        refiner(FRAME, inst)
    refiner.load(CTX)
    out = refiner(FRAME, inst)
    # Candidate 1 (IoU 0.3): the box shrunk by 2 px.
    np.testing.assert_array_equal(
        out.masks, box_masks(np.array(BOXES) + [2, 2, -2, -2], 100, 50)
    )
    assert out.mask_score.tolist() == pytest.approx([0.3] * 3)
    assert out.has_mask().all()
