import dataclasses
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
    DepthResult,
    Frame,
    FrameResult,
    Instances,
    MapGeometry,
    Provenance,
    RefinedMasks,
    SemanticMap,
    best_per_source,
    check_array,
    per_instance,
)
from det_depth_distance.distance import DistanceConfig, box_depth_stats


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


def _fused(inst, sources):
    return dataclasses.replace(inst, sources=sources)


def test_sources_travel_with_select_and_concat(make_instances):
    p, q = Provenance("a", "car", 0.9), Provenance("b", "suv", 0.7)
    fused = _fused(make_instances([[0, 0, 1, 1], [0, 0, 2, 2]]), ((p, q), (q,)))
    assert fused.select(np.array([1])).sources == ((q,),)
    assert Instances.concat([fused, fused]).sources == ((p, q), (q,)) * 2
    # No concat_fill: concatenating with unfused instances drops the field.
    assert Instances.concat([fused, make_instances([[0, 0, 1, 1]])]).sources is None


def test_provenance_synthesises_single_members(make_instances):
    inst = make_instances([[0, 0, 1, 1]], score=[0.5], labels=["suv"], source="y")
    assert inst.provenance() == ((Provenance("y", "suv", 0.5),),)
    fused = _fused(inst, ((Provenance("a", "car", 0.9),),))
    assert fused.provenance() is fused.sources


def test_best_per_source():
    members = (
        Provenance("a", "car", 0.6),
        Provenance("b", "suv", 0.8),
        Provenance("a", "truck", 0.7),
        Provenance("c", "van", 0.8),
    )
    assert best_per_source(members) == (
        Provenance("b", "suv", 0.8),
        Provenance("c", "van", 0.8),  # tie: order of members
        Provenance("a", "truck", 0.7),
    )


@pytest.mark.parametrize(
    "sources",
    [
        (),  # wrong length
        ((),),  # empty member tuple
        (("a",),),  # not Provenance
        [(Provenance("a", "car", 0.9),)],  # not a tuple
    ],
)
def test_validate_rejects_bad_sources(make_instances, sources):
    with pytest.raises(ContractError, match="sources"):
        _fused(make_instances([[0, 0, 1, 1]]), sources).validate(10, 10)


def test_segmenter_masks_travel_with_select(make_instances):
    # A Seg adapter's output: one mask per instance, frame resolution.
    masks = np.zeros((3, 20, 30), dtype=bool)
    for i in range(3):
        masks[i, i : i + 2, i : i + 2] = True
    inst = make_instances([[0, 0, 10, 10]] * 3, masks=masks)
    inst.validate(30, 20)
    kept = inst.select(np.array([True, False, True]))
    kept.validate(30, 20)
    np.testing.assert_array_equal(kept.masks, masks[[0, 2]])


def test_frame_result_box_stats_is_optional(make_instances):
    geometry = MapGeometry(10, 10, 10, 10, 1.0, 1.0)
    depth = DepthResult(
        np.full((10, 10), 5.0, np.float32),
        CameraIntrinsics.centered(10.0, 10, 10),
        None,
        geometry,
    )
    inst = make_instances([[0, 0, 10, 10], [0, 0, 5, 5]])
    stats = tuple(
        box_depth_stats(depth.depth_m, box, geometry, DistanceConfig())
        for box in inst.xyxy
    )
    frame = Frame(0, 0.0, np.zeros((10, 10, 3), np.uint8))
    plain = FrameResult(frame, inst, stats, depth, {})
    assert plain.box_stats is None and plain.agreement is None
    both = FrameResult(frame, inst, stats, depth, {}, box_stats=(stats[0], None))
    assert both.box_stats == (stats[0], None)


def _two_with_one_mask(make_instances):
    """Two 10x10 instances in a 20x20 frame; only the first has a mask in its box."""
    masks = np.zeros((2, 20, 20), dtype=bool)
    masks[0, 2:5, 2:5] = True
    return make_instances([[0, 0, 10, 10], [10, 10, 20, 20]], source="seg", masks=masks)


def test_mask_fields_are_validated_and_selected(make_instances):
    inst = _two_with_one_mask(make_instances)
    refined = dataclasses.replace(
        inst,
        mask_source=("seg", "sam"),
        mask_score=np.array([np.nan, 0.8], np.float32),
    )
    refined.validate(20, 20)
    kept = refined.select(np.array([1]))
    assert kept.mask_source == ("sam",) and kept.mask_score.tolist() == [
        pytest.approx(0.8)
    ]
    with pytest.raises(ContractError, match="mask_source"):
        dataclasses.replace(refined, mask_source=("seg",)).validate(20, 20)
    with pytest.raises(ContractError, match="mask_score: expected float32"):
        dataclasses.replace(refined, mask_score=np.zeros(2)).validate(20, 20)
    assert inst.mask_source is None and inst.mask_score is None  # optional


def test_mask_sources_synthesised(make_instances):
    assert _two_with_one_mask(make_instances).mask_sources() == ("seg", "")
    assert make_instances([[0, 0, 1, 1]], source="det").mask_sources() == ("",)
    inst = dataclasses.replace(
        _two_with_one_mask(make_instances), mask_source=("seg", "sam")
    )
    assert inst.mask_sources() == ("seg", "sam")  # set by a refiner: used as is


def test_refined_masks_validate():
    ok = RefinedMasks(np.zeros((2, 4, 6), bool), np.zeros(2, np.float32))
    ok.validate(2, 6, 4)
    with pytest.raises(ContractError, match="masks: shape"):
        ok.validate(2, 4, 6)
    with pytest.raises(ContractError, match="masks: expected bool"):
        RefinedMasks(np.zeros((2, 4, 6), np.uint8), ok.score).validate(2, 6, 4)
    with pytest.raises(ContractError, match="score: shape"):
        RefinedMasks(ok.masks, np.zeros(3, np.float32)).validate(2, 6, 4)


def test_semantic_map_validate_and_fractions():
    labels = np.zeros((4, 5), np.int32)
    labels[:, :2] = 2
    m = SemanticMap(labels, ("road", "sky", "lane"))
    m.validate(5, 4)
    assert m.fractions() == {"road": 0.6, "sky": 0.0, "lane": 0.4}
    assert sum(m.fractions().values()) == pytest.approx(1.0)
    with pytest.raises(ContractError, match="outside"):
        SemanticMap(labels + 1, m.class_names).validate(5, 4)
    with pytest.raises(ContractError, match="outside"):
        SemanticMap(labels - 1, m.class_names).validate(5, 4)
    with pytest.raises(ContractError, match="expected int32"):
        SemanticMap(labels.astype(np.int64), m.class_names).validate(5, 4)
    with pytest.raises(ContractError, match="shape"):
        m.validate(4, 5)


def test_frame_result_semantic_defaults_to_an_unshared_dict(make_instances):
    geometry = MapGeometry(10, 10, 10, 10, 1.0, 1.0)
    depth = DepthResult(
        np.ones((10, 10), np.float32), CameraIntrinsics(1, 1, 5, 5), None, geometry
    )
    frame = Frame(0, 0.0, np.zeros((10, 10, 3), np.uint8))
    a = FrameResult(frame, Instances.empty(), (), depth, {})
    b = FrameResult(frame, Instances.empty(), (), depth, {})
    assert a.semantic == {} and a.semantic is not b.semantic
