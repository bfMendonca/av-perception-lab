import numpy as np
import pytest

from av_perception_lab.core.boxes import (
    greedy_match,
    iou_matrix,
    label_groups,
    nms_clusters,
    wbf_clusters,
)


def _clusters(clusters):
    return [c.tolist() for c in clusters]


def test_iou_matrix():
    a = np.array([[0, 0, 10, 10], [20, 20, 30, 30], [5, 5, 5, 5]], np.float32)
    b = np.array([[0, 0, 10, 10], [2, 2, 7, 7], [100, 100, 110, 110]])
    iou = iou_matrix(a, b)
    assert iou.shape == (3, 3) and iou.dtype == np.float64
    assert iou[0, 0] == 1.0  # identical
    assert iou[0, 1] == pytest.approx(25 / 100)  # contained: smaller / larger area
    assert iou[0, 2] == 0.0 and iou[1, 1] == 0.0  # disjoint
    assert iou[2].tolist() == [0.0, 0.0, 0.0]  # zero-area box
    assert iou_matrix(a[2], a[2]).tolist() == [[0.0]]  # zero union, no division error
    assert iou_matrix(np.zeros((0, 4)), b).shape == (0, 3)
    assert iou_matrix(a, np.zeros((0, 4))).shape == (3, 0)


def test_label_groups():
    groups = label_groups(["vehicle", "person", "vehicle"], class_aware=True)
    assert groups.dtype == np.int64
    assert groups[0] == groups[2] != groups[1]
    assert label_groups(["vehicle", "person"], class_aware=False).tolist() == [0, 0]
    assert label_groups([], class_aware=True).shape == (0,)


def test_greedy_match_is_one_to_one_by_decreasing_iou():
    iou = np.array([[0.9, 0.8], [0.85, 0.2], [0.0, 0.6]])
    rows, cols = greedy_match(iou, 0.5)
    assert sorted(zip(rows.tolist(), cols.tolist())) == [(0, 0), (2, 1)]
    rows, cols = greedy_match(np.array([[0.7, 0.7]]), 0.5)  # tie: first column
    assert (rows.tolist(), cols.tolist()) == ([0], [0])
    rows, cols = greedy_match(np.zeros((0, 3)), 0.5)
    assert rows.dtype == np.intp and len(rows) == len(cols) == 0


def test_nms_clusters():
    xyxy = np.array([[0, 0, 10, 10], [1, 0, 11, 10], [50, 50, 60, 60]], np.float32)
    score = np.array([0.6, 0.9, 0.8])
    same = np.zeros(3, np.int64)
    assert _clusters(nms_clusters(xyxy, score, same, 0.5)) == [[1, 0], [2]]
    # Different groups are not merged; one group (class agnostic) is.
    groups = np.array([0, 1, 0])
    assert _clusters(nms_clusters(xyxy, score, groups, 0.5)) == [[1], [2], [0]]
    assert nms_clusters(np.zeros((0, 4)), np.zeros(0), np.zeros(0, np.int64), 0.5) == []


def test_nms_chain_is_not_transitive():
    # B overlaps A and C, A does not overlap C: C stays its own cluster.
    xyxy = np.array([[0, 0, 10, 10], [4, 0, 14, 10], [8, 0, 18, 10]], np.float32)
    score = np.array([0.9, 0.8, 0.7])
    clusters = nms_clusters(xyxy, score, np.zeros(3, np.int64), 0.4)
    assert _clusters(clusters) == [[0, 1], [2]]


def test_wbf_fused_box_by_hand():
    xyxy = np.array([[0, 0, 10, 10], [2, 0, 12, 10], [4, 0, 14, 10]], np.float32)
    score = np.array([0.9, 0.6, 0.3])
    clusters, fused = wbf_clusters(xyxy[:2], score[:2], np.zeros(2, np.int64), 0.5)
    assert _clusters(clusters) == [[0, 1]]
    x0 = (0.9 * 0 + 0.6 * 2) / 1.5
    assert fused[0].tolist() == pytest.approx([x0, 0, x0 + 10, 10])
    clusters, fused = wbf_clusters(xyxy, score, np.zeros(3, np.int64), 0.5)
    assert _clusters(clusters) == [[0, 1, 2]]
    x0 = (0.6 * 2 + 0.3 * 4) / 1.8
    assert fused[0, 0] == pytest.approx(x0)


def test_wbf_matches_against_the_fused_box():
    # C overlaps A by IoU 0.25 < 0.4 but the fused box of A and B ([2, 0, 12, 10])
    # by 6/14 >= 0.4, so it joins their cluster.
    xyxy = np.array([[0, 0, 10, 10], [4, 0, 14, 10], [6, 0, 16, 10]], np.float32)
    score = np.array([0.5, 0.5, 0.4])
    groups = np.zeros(3, np.int64)
    assert iou_matrix(xyxy[0], xyxy[2])[0, 0] == pytest.approx(0.25)
    clusters, fused = wbf_clusters(xyxy, score, groups, 0.4)
    assert _clusters(clusters) == [[0, 1, 2]]
    x0 = (0.5 * 0 + 0.5 * 4 + 0.4 * 6) / 1.4
    assert fused[0].tolist() == pytest.approx([x0, 0, x0 + 10, 10])
    # Same boxes through NMS: C only overlaps the suppressed B, so it stays apart.
    assert _clusters(nms_clusters(xyxy, score, groups, 0.4)) == [[0, 1], [2]]


def test_wbf_respects_groups_and_empty_input():
    xyxy = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], np.float32)
    clusters, fused = wbf_clusters(xyxy, np.array([0.9, 0.8]), np.array([0, 1]), 0.5)
    assert _clusters(clusters) == [[0], [1]]
    assert fused.tolist() == xyxy.tolist()
    clusters, fused = wbf_clusters(
        np.zeros((0, 4)), np.zeros(0), np.zeros(0, np.int64), 0.5
    )
    assert clusters == [] and fused.shape == (0, 4)
