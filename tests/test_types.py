from dataclasses import dataclass, field

import numpy as np
import pytest

from det_depth_distance.adapters.depth.da3 import (
    da3_map_geometry,
    intrinsics_from_focal,
)
from det_depth_distance.core.types import (
    CameraIntrinsics,
    ContractError,
    Frame,
    Instances,
    check_array,
    per_instance,
)


def test_select_by_mask_and_indices(make_instances):
    inst = make_instances(
        np.arange(12).reshape(3, 4),
        labels=["car", "person", "car"],
        score=[0.9, 0.8, 0.7],
    )
    kept = inst.select(np.array([True, False, True]))
    assert len(kept) == 2
    assert kept.label == ("car", "car") and kept.raw_label == ("car", "car")
    assert kept.xyxy[1].tolist() == [8, 9, 10, 11]
    assert inst.select(np.array([2, 0])).score.tolist() == pytest.approx([0.7, 0.9])
    assert len(inst.select(np.zeros(3, dtype=bool))) == 0


def test_concat_keeps_order_and_source(make_instances):
    a = make_instances([[0, 0, 10, 10]], source="a")
    b = make_instances([[1, 1, 5, 5], [2, 2, 6, 6]], source="b")
    assert len(Instances.concat([])) == 0
    assert Instances.concat([a]).source == ("a",)
    both = Instances.concat([a, b])
    assert both.source == ("a", "b", "b")
    assert both.xyxy[:, 0].tolist() == [0, 1, 2]
    assert both.masks is None


def test_concat_fills_missing_masks_with_zeros(make_instances):
    masks = np.ones((1, 20, 30), dtype=bool)
    with_mask = make_instances([[0, 0, 10, 10]], masks=masks)
    without = make_instances([[0, 0, 5, 5], [1, 1, 6, 6]])
    empty = make_instances(np.zeros((0, 4)), masks=np.zeros((0, 20, 30), bool))
    out = Instances.concat([without, with_mask, empty])
    assert out.masks.shape == (3, 20, 30) and out.masks.dtype == bool
    assert not out.masks[:2].any() and out.masks[2].all()
    out.validate(30, 20)


@dataclass(frozen=True, slots=True)
class Tracked(Instances):
    """Instances plus a later-task style optional field with its own checker."""

    track_id: np.ndarray | None = field(
        default=None, metadata=per_instance(check_array(np.int64))
    )


def _tracked(inst, track_id=None):
    fields = ("xyxy", "score", "class_id", "label", "raw_label", "source")
    return Tracked(*(getattr(inst, f) for f in fields), track_id=track_id)


def test_concat_field_without_fill_is_none_when_partly_missing(make_instances):
    a = _tracked(make_instances([[0, 0, 1, 1]]), np.array([7]))
    b = _tracked(make_instances([[0, 0, 2, 2]]))
    assert Instances.concat([a, a]).track_id.tolist() == [7, 7]
    assert Instances.concat([a, b]).track_id is None
    assert Instances.concat([b, b]).track_id is None


def test_has_mask_uses_box_window(make_instances):
    masks = np.zeros((3, 20, 20), dtype=bool)
    masks[1, 15, 15] = True  # only outside its box
    masks[2, 5, 5] = True  # pixel centre (5.5, 5.5) inside [0, 10]
    inst = make_instances([[0, 0, 10, 10]] * 3, masks=masks)
    assert inst.has_mask().tolist() == [False, False, True]
    assert make_instances([[0, 0, 10, 10]]).has_mask().tolist() == [False]


@pytest.mark.parametrize(
    "box", [[0, 0, 101, 10], [-1, 0, 10, 10], [10, 0, 5, 10], [0, 0, np.nan, 10]]
)
def test_validate_rejects_bad_boxes(make_instances, box):
    with pytest.raises(ContractError):
        make_instances([box]).validate(100, 50)


def test_validate_rejects_dtype_and_mask_shape(make_instances):
    inst = make_instances([[0, 0, 10, 10]])
    inst.validate(100, 50)
    bad = Instances(
        inst.xyxy,
        inst.score.astype(np.float64),
        inst.class_id,
        inst.label,
        inst.raw_label,
        inst.source,
    )
    with pytest.raises(ContractError, match="score"):
        bad.validate(100, 50)
    with pytest.raises(ContractError, match="masks"):
        make_instances([[0, 0, 10, 10]], masks=np.zeros((1, 10, 10), bool)).validate(
            100, 50
        )
    with pytest.raises(ContractError, match="label"):
        Instances(inst.xyxy, inst.score, inst.class_id, (), ("car",), ("d",)).validate(
            100, 50
        )


def test_new_field_is_validated_by_its_own_checker(make_instances):
    base = make_instances([[0, 0, 10, 10]])
    _tracked(base, np.array([3])).validate(100, 50)
    _tracked(base).validate(100, 50)  # optional field left None
    with pytest.raises(ContractError, match="track_id"):
        _tracked(base, np.array([3.0])).validate(100, 50)
    assert _tracked(base, np.array([3])).select(np.array([True])).track_id.tolist() == [
        3
    ]


def test_frame_size():
    frame = Frame(0, 0.0, np.zeros((48, 64, 3), np.uint8))
    assert (frame.width, frame.height) == (64, 48)


@pytest.mark.parametrize("method", ["upper_bound_resize", "upper_bound_crop"])
def test_scale_intrinsics_matches_intrinsics_from_focal(method):
    g = da3_map_geometry(1920, 1080, 504, method)
    k = g.scale_intrinsics(CameraIntrinsics.centered(1390.4, 1920, 1080))
    assert k == intrinsics_from_focal(1390.4, g)  # bit-identical
    assert k.fx == 1390.4 * g.sx and k.cy == 540 * g.sy + g.oy
