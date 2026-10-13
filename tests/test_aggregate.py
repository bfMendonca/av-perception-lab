import numpy as np
import pytest

from det_depth_distance.core import registry
from det_depth_distance.core.aggregate import Concat, Nms, Wbf
from det_depth_distance.core.options import ConfigError
from det_depth_distance.core.types import Provenance


def test_concat_preserves_order_and_source(make_instances):
    a = make_instances([[0, 0, 1, 1]], source="a", labels=["car"])
    b = make_instances([[0, 0, 2, 2], [0, 0, 3, 3]], source="b", labels=["suv", "van"])
    out = Concat(Concat.Options())([a, b])
    assert out.source == ("a", "b", "b")
    assert out.label == ("car", "suv", "van")
    assert Concat(Concat.Options()).describe() == {"type": "concat", "options": {}}


def test_bind_sources_is_a_noop():
    concat = Concat(Concat.Options())
    assert concat.bind_sources(["a", "b"]) is None


# Fusion aggregators on synthetic boxes in a 100x100 frame.
CAR = [10, 10, 40, 40]
CAR_SHIFTED = [12, 10, 42, 40]  # IoU 28/32 = 0.875 with CAR
FAR = [60, 60, 90, 90]


def _nms(**options):
    return Nms(Nms.Options(**options))


def _wbf(**options):
    return Wbf(Wbf.Options(**options))


def test_two_detectors_same_object_fuse_with_members(make_instances):
    a = make_instances([CAR], source="a", labels=["vehicle"], score=[0.9])
    b = make_instances([CAR_SHIFTED, FAR], source="b", labels=["vehicle"] * 2)
    for agg in (_nms(), _wbf()):
        agg.bind_sources(["a", "b"])
        out = agg([a, b])
        out.validate(100, 100)
        assert len(out) == 2
        assert out.sources[0] == (
            Provenance("a", "vehicle", pytest.approx(0.9)),
            Provenance("b", "vehicle", pytest.approx(0.9)),
        )
        assert [len(m) for m in out.sources] == [2, 1]
        assert out.source[0] == "a"  # representative: first in score order


def test_labels_group_unless_class_agnostic(make_instances):
    a = make_instances([CAR], source="a", labels=["person"])
    b = make_instances([CAR_SHIFTED], source="b", labels=["vehicle"], score=[0.8])
    assert len(_nms()([a, b])) == 2
    out = _nms(class_agnostic=True)([a, b])
    assert len(out) == 1 and out.label == ("person",)  # label of the representative


def test_duplicates_within_one_detector_are_fused(make_instances):
    # Fact 4: one detector boxes the same object as car and truck, both -> vehicle.
    raw = make_instances([CAR, CAR], source="a", labels=["car", "truck"])
    one = raw.with_labels(["vehicle", "vehicle"])
    out = _nms()([one])
    assert len(out) == 1
    assert [m.raw_label for m in out.sources[0]] == ["car", "truck"]


def test_weight_changes_the_nms_winner(make_instances):
    a = make_instances([CAR], source="a", score=[0.9])
    b = make_instances([CAR_SHIFTED], source="b", score=[0.7])
    agg = _nms(weights={"b": 2.0})
    assert agg([a, b]).source == ("a",)  # unbound: weights not applied yet
    agg.bind_sources(["a", "b"])
    out = agg([a, b])
    assert out.source == ("b",)
    # NMS keeps the keeper's own box and original score.
    assert out.xyxy[0].tolist() == CAR_SHIFTED and out.score[0] == pytest.approx(0.7)
    assert agg.describe()["weights"] == {"a": 1.0, "b": 2.0}


def test_wbf_avg_score_by_hand(make_instances):
    a = make_instances([CAR], source="a", score=[0.9])
    b = make_instances([CAR_SHIFTED, FAR], source="b", score=[0.7, 0.6])
    agg = _wbf()
    agg.bind_sources(["a", "b"])
    out = agg([a, b])
    assert out.score.tolist() == pytest.approx([0.8, 0.3])  # T = 2: lone box halved
    x0 = (0.9 * 10 + 0.7 * 12) / 1.6
    assert out.xyxy[0].tolist() == pytest.approx([x0, 10, x0 + 30, 40])
    single = _wbf()
    single.bind_sources(["a"])
    assert single([a]).score.tolist() == pytest.approx([0.9])  # T = 1: unchanged


def test_wbf_avg_score_is_capped_at_one(make_instances):
    # The reference rescale gives (0.9 * 2) * min(2, 2) / (2 + 1) = 1.2 when the
    # heavier detector boxes the same object twice; a confidence stays <= 1.
    a = make_instances([CAR, CAR_SHIFTED], source="a", score=[0.9, 0.9])
    b = make_instances([FAR], source="b", score=[0.6])
    agg = _wbf(weights={"a": 2.0})
    agg.bind_sources(["a", "b"])
    out = agg([a, b])
    out.validate(100, 100)
    assert out.score.tolist() == pytest.approx([1.0, 0.2])  # lone b: 0.6 * 1 / 3


def test_wbf_max_and_skip_box_thr(make_instances):
    a = make_instances([CAR], source="a", score=[0.9])
    b = make_instances([CAR_SHIFTED], source="b", score=[0.7])
    agg = _wbf(conf_type="max", weights={"a": 2.0})
    agg.bind_sources(["a", "b"])
    assert agg([a, b]).score.tolist() == pytest.approx([0.9])  # (0.9 * 2) / 2
    skip = _wbf(skip_box_thr=0.8)
    out = skip([a, b])
    assert len(out) == 1 and len(out.sources[0]) == 1
    assert len(_wbf(skip_box_thr=0.95)([a, b])) == 0


def test_empty_parts(make_instances):
    empty = make_instances(np.zeros((0, 4)))
    for agg in (_nms(), _wbf()):
        out = agg([empty, empty])
        assert len(out) == 0
        out.validate(100, 100)
        assert len(agg([])) == 0


def _mask(filled=False):
    mask = np.zeros((1, 100, 100), bool)
    if filled:
        mask[0, 20:30, 20:30] = True
    return mask


def test_mask_of_the_best_member_with_a_mask(make_instances):
    seg = make_instances([CAR_SHIFTED], source="seg", score=[0.7], masks=_mask(True))
    box_only = make_instances([CAR], source="box", score=[0.9])
    out = _nms()([box_only, seg])  # the box-only detector wins the cluster
    assert out.source == ("box",) and out.masks[0].any()
    own = np.zeros((1, 100, 100), bool)
    own[0, 12:15, 12:15] = True  # differs from seg's: the representative's is kept
    both = make_instances([CAR], source="box", score=[0.9], masks=own)
    out = _nms()([both, seg])
    assert np.array_equal(out.masks[0], both.masks[0])
    none = make_instances([CAR], source="x", score=[0.9], masks=_mask())
    out = _nms()([none, make_instances([CAR_SHIFTED], source="y", masks=_mask())])
    assert out.masks.shape == (1, 100, 100) and not out.masks.any()
    out.validate(100, 100)


def test_bind_sources_rejects_unknown_detector():
    agg = _wbf(weights={"yolo": 2.0})
    with pytest.raises(ConfigError, match="unknown detector 'yolo'; detectors: a, b"):
        agg.bind_sources(["a", "b"])


@pytest.mark.parametrize(
    ("spec", "match"),
    [
        ({"type": "nms", "iou_thr": 1.0}, "iou_thr"),
        ({"type": "wbf", "iou_thr": 0}, "iou_thr"),
        ({"type": "wbf", "weights": {"a": 0}}, "weights must be > 0"),
        ({"type": "nms", "conf_type": "max"}, "unknown option"),
    ],
)
def test_fusion_options_errors_through_registry(spec, match):
    with pytest.raises(ConfigError, match=rf"^aggregation.*{match}"):
        registry.build("aggregator", spec, path="aggregation")


def test_fusion_options_from_yaml():
    agg = registry.build(
        "aggregator",
        {"type": "wbf", "weights": {"a": 2}, "class_agnostic": True},
        path="aggregation",
    )
    assert agg.options.weights == {"a": 2.0} and agg.options.iou_thr == 0.55
    assert (
        registry.build("aggregator", {"type": "nms"}, path="x").options.iou_thr == 0.5
    )


@pytest.mark.parametrize("agg", [_nms, _wbf], ids=["nms", "wbf"])
def test_fused_mask_names_its_member(make_instances, agg):
    # Box-only `a` wins the cluster, the mask comes from `b`: mask_sources() must
    # name `b`, not the representative.
    a = make_instances([CAR], source="a", score=[0.9])
    b = make_instances([CAR_SHIFTED], source="b", score=[0.7], masks=_mask(True))
    fusion = agg()
    fusion.bind_sources(["a", "b"])
    out = fusion([a, b])
    out.validate(100, 100)
    assert len(out) == 1 and out.source == ("a",) and out.masks[0].any()
    assert out.mask_source == ("b",) and out.mask_sources() == ("b",)
    assert out.mask_score.dtype == np.float32 and np.isnan(out.mask_score).all()
    # No member with a mask in the cluster: "" (and the masks stay all-False).
    none = make_instances([CAR_SHIFTED], source="b", masks=_mask())
    out = fusion([a, none])
    assert out.mask_source == ("",) and not out.masks.any()
    # No masks at all: the fields stay unset.
    out = fusion([a, make_instances([CAR_SHIFTED], source="b")])
    assert out.masks is None and out.mask_source is None and out.mask_score is None
    assert out.mask_sources() == ("",)
