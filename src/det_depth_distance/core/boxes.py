"""Axis-aligned box geometry and grouping (numpy only).

Boxes are xyxy with continuous edges in frame pixels. IoU and score-weighted
averages are scale invariant, so no normalisation is needed (unlike the
reference WBF implementation, which expects [0, 1] coordinates). Shared by
the fusion aggregators, the cross-detector agreement metrics and, later,
IoU-based tracking. Matching uses `iou >= iou_thr` everywhere.

Clusters are index arrays into the input boxes; boxes are visited by
decreasing score (stable sort, so ties keep input order), and each cluster
lists its members in that order, the first being its representative.
"""

from collections.abc import Sequence

import numpy as np


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of (Na, 4) and (Nb, 4) xyxy boxes -> (Na, Nb) float64.

    A pair whose union has zero area (two degenerate boxes) has IoU 0.
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    top_left = np.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(bottom_right - top_left, 0.0, None).prod(axis=-1)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def label_groups(labels: Sequence[str], *, class_aware: bool) -> np.ndarray:
    """(N,) int64 group id per label: equal labels share an id when
    `class_aware`, otherwise every box is in group 0."""
    if not class_aware:
        return np.zeros(len(labels), dtype=np.int64)
    _, inverse = np.unique(np.asarray(labels, dtype=str), return_inverse=True)
    return inverse.astype(np.int64).reshape(-1)


def greedy_match(iou: np.ndarray, iou_thr: float) -> tuple[np.ndarray, np.ndarray]:
    """One-to-one pairs (rows, cols) with `iou >= iou_thr`, accepted by
    decreasing IoU (ties in row-major order).

    Greedy rather than optimal assignment: IoUs of the same object across
    detectors are bimodal (about 0 or >= 0.7, task_0003 Fact 2), where both
    agree, and it keeps scipy out of the light environment.
    """
    rows, cols = np.nonzero(iou >= iou_thr)
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    out_rows, out_cols = [], []
    for k in np.argsort(-iou[rows, cols], kind="stable"):
        r, c = int(rows[k]), int(cols[k])
        if r in used_rows or c in used_cols:
            continue
        used_rows.add(r)
        used_cols.add(c)
        out_rows.append(r)
        out_cols.append(c)
    return np.asarray(out_rows, dtype=np.intp), np.asarray(out_cols, dtype=np.intp)


def nms_clusters(
    xyxy: np.ndarray, score: np.ndarray, groups: np.ndarray, iou_thr: float
) -> list[np.ndarray]:
    """Clusters of non-maximum suppression.

    The highest-scoring unassigned box keeps its cluster and absorbs every
    unassigned box of the same group overlapping *it* by `iou_thr` (a box
    that only overlaps a suppressed one starts its own cluster). Clusters
    come by decreasing keeper score, keeper first.
    """
    order = np.argsort(-np.asarray(score), kind="stable")
    iou = iou_matrix(xyxy, xyxy)
    assigned = np.zeros(len(order), dtype=bool)
    clusters = []
    for i in order:
        if assigned[i]:
            continue
        candidates = order[~assigned[order]]
        hit = (
            (groups[candidates] == groups[i])
            & (iou[i, candidates] >= iou_thr)
            & (candidates != i)  # explicit: a zero-area box has IoU 0 with itself
        )
        cluster = np.concatenate(([i], candidates[hit])).astype(np.intp)
        assigned[cluster] = True
        clusters.append(cluster)
    return clusters


def wbf_clusters(
    xyxy: np.ndarray, score: np.ndarray, groups: np.ndarray, iou_thr: float
) -> tuple[list[np.ndarray], np.ndarray]:
    """Cluster boxes as Weighted Boxes Fusion does and return the fused boxes.

    Boxes are visited by decreasing score; each joins the same-group cluster
    whose current fused box overlaps it most (IoU >= iou_thr) or starts a new
    one, and the cluster's fused box becomes the score-weighted mean of its
    members. Ported from ZFTurbo/Weighted-Boxes-Fusion
    (ensemble_boxes_wbf.py, MIT licence), keeping cluster membership, which the
    original does not return. The original requires IoU > iou_thr; here it is
    >= like every other match in this module.

    Returns:
        (clusters as in `nms_clusters`, fused boxes (K, 4) float64).
    """
    xyxy = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4)
    score = np.asarray(score, dtype=np.float64)
    members: list[list[int]] = []
    fused: list[np.ndarray] = []
    fused_groups: list[int] = []
    for i in np.argsort(-score, kind="stable"):
        best = -1
        if fused:
            iou = iou_matrix(xyxy[i], np.stack(fused))[0]
            iou[np.asarray(fused_groups) != groups[i]] = -1.0
            j = int(np.argmax(iou))
            if iou[j] >= iou_thr:
                best = j
        if best < 0:
            members.append([int(i)])
            fused.append(xyxy[i].copy())
            fused_groups.append(groups[i])
            continue
        members[best].append(int(i))
        w = score[members[best]]
        fused[best] = (w[:, None] * xyxy[members[best]]).sum(axis=0) / w.sum()
    clusters = [np.asarray(m, dtype=np.intp) for m in members]
    boxes = np.stack(fused) if fused else np.zeros((0, 4), dtype=np.float64)
    return clusters, boxes
