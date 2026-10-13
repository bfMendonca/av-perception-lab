import json

import pytest

from det_depth_distance.core.agreement import AgreementStats, frame_agreement
from det_depth_distance.core.types import Instances

A_BOXES = [[0, 0, 10, 10], [20, 0, 30, 10], [40, 0, 50, 10]]
B_BOXES = [[1, 0, 11, 10], [20, 0, 30, 10], [70, 0, 80, 10]]


def test_two_detectors_matched_and_exclusive(make_instances):
    a = make_instances(A_BOXES, source="a", labels=["vehicle"] * 3)
    b = make_instances(B_BOXES, source="b", labels=["vehicle"] * 3)
    fa = frame_agreement([a, b], ["a", "b"], iou_thr=0.5, class_aware=True)
    assert fa.instances == {"a": 3, "b": 3} and fa.duplicates == {"a": 0, "b": 0}
    (pair,) = fa.pairs
    assert (pair.a, pair.b, pair.matched, pair.only_a, pair.only_b) == (
        "a",
        "b",
        2,
        1,
        1,
    )
    assert pair.iou_sum == pytest.approx(90 / 110 + 1.0)


def test_internal_duplicate_is_counted_not_exclusive(make_instances):
    # The second detector sees the first object twice (e.g. car and suv -> vehicle).
    a = make_instances(A_BOXES[:1], source="a", labels=["vehicle"])
    b = make_instances(
        [[0, 0, 10, 10], [0, 0, 10, 9]], source="b", labels=["vehicle"] * 2
    )
    fa = frame_agreement([a, b], ["a", "b"], iou_thr=0.5, class_aware=True)
    assert fa.instances["b"] == 1 and fa.duplicates["b"] == 1
    assert (fa.pairs[0].matched, fa.pairs[0].only_b) == (1, 0)


def test_class_aware_does_not_match_different_labels(make_instances):
    a = make_instances(A_BOXES[:1], source="a", labels=["person"])
    b = make_instances(A_BOXES[:1], source="b", labels=["vehicle"])
    aware = frame_agreement([a, b], ["a", "b"], iou_thr=0.5, class_aware=True)
    assert (aware.pairs[0].matched, aware.pairs[0].only_a) == (0, 1)
    blind = frame_agreement([a, b], ["a", "b"], iou_thr=0.5, class_aware=False)
    assert blind.pairs[0].matched == 1


def test_three_detectors_and_empty_part(make_instances):
    a = make_instances(A_BOXES, source="a")
    b = make_instances(B_BOXES, source="b")
    fa = frame_agreement(
        [a, b, Instances.empty()], ["a", "b", "c"], iou_thr=0.5, class_aware=True
    )
    assert [(p.a, p.b) for p in fa.pairs] == [("a", "b"), ("a", "c"), ("b", "c")]
    assert (fa.pairs[1].matched, fa.pairs[1].only_a, fa.pairs[1].only_b) == (0, 3, 0)
    assert fa.instances["c"] == 0 and fa.pairs[1].iou_sum == 0.0


def test_summary_totals_and_mean_iou(make_instances):
    a = make_instances(A_BOXES, source="a")
    b = make_instances(B_BOXES, source="b")
    stats = AgreementStats(0.5, True)
    assert len(stats) == 0
    frames = {
        3: frame_agreement([a, b], ["a", "b"], iou_thr=0.5, class_aware=True),
        4: frame_agreement([a, a], ["a", "b"], iou_thr=0.5, class_aware=True),
    }
    for index, fa in frames.items():
        stats.add(index, fa)
    summary = json.loads(json.dumps(stats.summary()))
    assert len(stats) == 2
    assert (summary["iou_thr"], summary["class_aware"]) == (0.5, True)
    assert summary["detectors"] == {
        "a": {"instances": 6, "duplicates": 0},
        "b": {"instances": 6, "duplicates": 0},
    }
    (pair,) = summary["pairs"]
    assert (pair["matched"], pair["only_a"], pair["only_b"]) == (5, 1, 1)
    assert pair["mean_iou"] == pytest.approx((90 / 110 + 1.0 + 3.0) / 5)
    per_frame = summary["per_frame"]
    assert per_frame["frame"] == [3, 4]
    assert per_frame["a|b"] == [[2, 1, 1], [3, 0, 0]]
    assert sum(row[0] for row in per_frame["a|b"]) == pair["matched"]


def test_summary_without_matches_has_no_mean_iou(make_instances):
    stats = AgreementStats(0.5, True)
    a = make_instances(A_BOXES[:1], source="a")
    stats.add(
        0,
        frame_agreement(
            [a, Instances.empty()], ["a", "b"], iou_thr=0.5, class_aware=True
        ),
    )
    assert stats.summary()["pairs"][0]["mean_iou"] is None
