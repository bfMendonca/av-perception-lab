"""Cross-detector agreement metrics, computed per frame before aggregation.

Each detector's instances are first deduplicated (NMS within the detector,
same grouping), then every pair of detectors is matched one-to-one by IoU
(greedy, highest IoU first). Counts are matched / only-a / only-b per pair,
independent of the aggregation strategy, so they also work with concat.
"""

from collections import defaultdict
from collections.abc import Sequence
from itertools import combinations
from typing import Any

import numpy as np

from av_perception_lab.core.boxes import (
    greedy_match,
    iou_matrix,
    label_groups,
    nms_clusters,
)
from av_perception_lab.core.types import FrameAgreement, Instances, PairAgreement


def frame_agreement(
    parts: Sequence[Instances],
    names: Sequence[str],
    *,
    iou_thr: float,
    class_aware: bool,
) -> FrameAgreement:
    """Agreement between the detectors of one frame.

    Args:
        parts: one Instances per detector (possibly empty), aligned with `names`.
        names: detector names, in detector order.
        iou_thr: IoU for both the deduplication and the cross-detector match.
        class_aware: only instances with the same (canonical) label match.
    """
    keepers: list[Instances] = []
    instances, duplicates = {}, {}
    for name, part in zip(names, parts, strict=True):
        groups = label_groups(part.label, class_aware=class_aware)
        clusters = nms_clusters(part.xyxy, part.score, groups, iou_thr)
        keepers.append(part.select(np.array([c[0] for c in clusters], dtype=np.intp)))
        instances[name] = len(clusters)
        duplicates[name] = len(part) - len(clusters)
    pairs = []
    for (a, inst_a), (b, inst_b) in combinations(zip(names, keepers), 2):
        iou = iou_matrix(inst_a.xyxy, inst_b.xyxy)
        if class_aware:
            labels_a = np.asarray(inst_a.label, dtype=str)
            same = labels_a[:, None] == np.asarray(inst_b.label, dtype=str)[None]
            iou = np.where(same, iou, 0.0)
        rows, cols = greedy_match(iou, iou_thr)
        pairs.append(
            PairAgreement(
                a,
                b,
                matched=len(rows),
                only_a=len(inst_a) - len(rows),
                only_b=len(inst_b) - len(rows),
                iou_sum=float(iou[rows, cols].sum()),
            )
        )
    return FrameAgreement(instances, duplicates, tuple(pairs))


class AgreementStats:
    """Accumulates per-frame agreement into the summary of metrics.json."""

    def __init__(self, iou_thr: float, class_aware: bool) -> None:
        self.iou_thr = iou_thr
        self.class_aware = class_aware
        self._frames: list[int] = []
        self._detectors: dict[str, dict[str, int]] = {}
        self._pairs: dict[tuple[str, str], dict[str, float]] = {}
        self._series: dict[str, list[list[int]]] = defaultdict(list)

    def __len__(self) -> int:
        """Number of frames added."""
        return len(self._frames)

    def add(self, frame_index: int, fa: FrameAgreement) -> None:
        """Add the agreement of the frame `frame_index`."""
        self._frames.append(frame_index)
        for name, n in fa.instances.items():
            totals = self._detectors.setdefault(name, {"instances": 0, "duplicates": 0})
            totals["instances"] += n
            totals["duplicates"] += fa.duplicates[name]
        for p in fa.pairs:
            totals = self._pairs.setdefault(
                (p.a, p.b), {"matched": 0, "only_a": 0, "only_b": 0, "iou_sum": 0.0}
            )
            totals["matched"] += p.matched
            totals["only_a"] += p.only_a
            totals["only_b"] += p.only_b
            totals["iou_sum"] += p.iou_sum
            self._series[f"{p.a}|{p.b}"].append([p.matched, p.only_a, p.only_b])

    def summary(self) -> dict[str, Any]:
        """JSON summary for metrics.json:
        {"iou_thr", "class_aware",
         "detectors": {name: {"instances", "duplicates"}},
         "pairs": [{"a", "b", "matched", "only_a", "only_b", "mean_iou"}],
         "per_frame": {"frame": [...], "<a>|<b>": [[matched, only_a, only_b], ...]}}
        """
        pairs = []
        for (a, b), t in self._pairs.items():
            matched = int(t["matched"])
            pairs.append(
                {
                    "a": a,
                    "b": b,
                    "matched": matched,
                    "only_a": int(t["only_a"]),
                    "only_b": int(t["only_b"]),
                    "mean_iou": t["iou_sum"] / matched if matched else None,
                }
            )
        return {
            "iou_thr": self.iou_thr,
            "class_aware": self.class_aware,
            "detectors": self._detectors,
            "pairs": pairs,
            "per_frame": {"frame": self._frames, **self._series},
        }
