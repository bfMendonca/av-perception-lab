"""IoU tracker and per-track run summary (numpy only).

Association is greedy one-to-one on IoU (core.boxes) between each track's
predicted box and the frame's instances. Boxes are predicted with a
constant velocity in pixels per second, so association survives a frame
stride (replaying the task_0001 detections it holds up to stride 5 at
30 fps and breaks at 10). All times are seconds from Frame.t, never frame counts.
Instances only match tracks of the same group: same `source` for unfused
instances (so two detectors' copies of one object under concat keep their
own ids), one shared group for fused instances; labels join the group only
with class_aware, because detectors flip car/truck on the same object.

Not ultralytics' ByteTrack/BoT-SORT: they need `lap`, import torch and
ultralytics' global settings, count time in calls, ship as AGPL-3.0, and
ByteTrack's low-score stage is idle at our detector thresholds.
"""

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from det_depth_distance.core.boxes import greedy_match, iou_matrix, label_groups
from det_depth_distance.core.components import Tracker, TrackerOptions
from det_depth_distance.core.registry import register
from det_depth_distance.core.types import (
    FRAME_TIME_EPS_S,
    Frame,
    FrameResult,
    Instances,
)

Motion = Literal["none", "constant_velocity"]


def association_keys(instances: Instances, *, class_aware: bool) -> list[str]:
    """Association group key of every instance: "*" for fused instances (output
    of a fusion aggregator, whatever its member count in this frame, so the key
    does not change when a detector misses the object), else its `source`;
    "|<label>" is appended when `class_aware`."""
    if instances.sources is not None:
        keys = ["*"] * len(instances)
    else:
        keys = list(instances.source)
    if class_aware:
        keys = [f"{k}|{label}" for k, label in zip(keys, instances.label, strict=True)]
    return keys


@dataclass(slots=True)
class _Track:
    """Mutable state of one live track."""

    id: int
    key: str  # association group key (see association_keys)
    box: np.ndarray  # (4,) float64, last observed box, frame px
    velocity: np.ndarray  # (4,) float64, px/s per box edge
    last_t: float  # s, time of the last match
    hits: int = 1  # matched observations


@register("tracker", "iou")
class IouTracker(Tracker):
    """Greedy IoU tracker with constant-velocity box prediction.

    A track is born from every unmatched instance and dies after `max_age_s`
    without a match; a reappearing object then gets a new id.
    """

    @dataclass(frozen=True, slots=True, kw_only=True)
    class Options(TrackerOptions):
        iou_thr: float = 0.3  # minimum IoU of a match, in (0, 1)
        max_age_s: float = 0.5  # s without a match before a track dies, > 0
        motion: Motion = "constant_velocity"  # "none": predicted box = last box
        # Weight of the newest velocity measurement, in (0, 1]; the first
        # match sets the velocity directly.
        velocity_alpha: float = 0.5
        class_aware: bool = False  # match only equal labels

        def __post_init__(self) -> None:
            if not 0.0 < self.iou_thr < 1.0:
                raise ValueError(f"iou_thr must be in (0, 1), got {self.iou_thr}")
            if self.max_age_s <= 0:
                raise ValueError(f"max_age_s must be > 0, got {self.max_age_s}")
            if not 0.0 < self.velocity_alpha <= 1.0:
                raise ValueError(
                    f"velocity_alpha must be in (0, 1], got {self.velocity_alpha}"
                )

    options: Options

    def __init__(self, options: Options, *, name: str | None = None) -> None:
        super().__init__(options, name=name)
        self._tracks: list[_Track] = []
        self._next_id = 1

    def _reset(self) -> None:
        self._tracks = []
        self._next_id = 1

    def _assign(self, frame: Frame, instances: Instances) -> np.ndarray:
        """Return one track id per instance (same order).

        Expires stale tracks, predicts their boxes at frame.t, matches by IoU
        within association groups (iou >= iou_thr, highest IoU first), updates
        matched tracks (box, velocity EMA, last time) and opens tracks for the
        rest.
        """
        t, opts = frame.t, self.options
        max_age = opts.max_age_s + FRAME_TIME_EPS_S  # inclusive, rounding-proof
        self._tracks = [tr for tr in self._tracks if t - tr.last_t <= max_age]
        keys = association_keys(instances, class_aware=opts.class_aware)
        boxes = instances.xyxy.astype(np.float64)
        ids = np.zeros(len(instances), dtype=np.int64)
        if self._tracks and len(instances):
            iou = iou_matrix(self._predict(t), boxes)
            groups = label_groups(
                [tr.key for tr in self._tracks] + keys, class_aware=True
            )
            n_tracks = len(self._tracks)
            iou[groups[:n_tracks, None] != groups[None, n_tracks:]] = 0.0
            for r, c in zip(*greedy_match(iou, opts.iou_thr), strict=True):
                track = self._tracks[r]
                self._update(track, boxes[c], t)
                ids[c] = track.id
        for c in np.flatnonzero(ids == 0):
            self._tracks.append(
                _Track(self._next_id, keys[c], boxes[c].copy(), np.zeros(4), t)
            )
            ids[c] = self._next_id
            self._next_id += 1
        return ids

    def _predict(self, t: float) -> np.ndarray:
        """(T, 4) predicted box of every live track at time `t`."""
        boxes = np.stack([tr.box for tr in self._tracks])
        if self.options.motion == "none":
            return boxes
        dt = np.array([t - tr.last_t for tr in self._tracks])
        return boxes + np.stack([tr.velocity for tr in self._tracks]) * dt[:, None]

    def _update(self, track: _Track, box: np.ndarray, t: float) -> None:
        dt = t - track.last_t
        if dt > 0:
            velocity = (box - track.box) / dt
            alpha = 1.0 if track.hits == 1 else self.options.velocity_alpha
            track.velocity = alpha * velocity + (1.0 - alpha) * track.velocity
        track.box = box.copy()
        track.last_t = t
        track.hits += 1


@dataclass(slots=True)
class _TrackRecord:
    """Observations of one track, in frame order (TrackSummary)."""

    order: list[int] = field(default_factory=list)  # ordinal of the processed frame
    frame: list[int] = field(default_factory=list)
    t: list[float] = field(default_factory=list)
    label: list[str] = field(default_factory=list)
    sources: set[str] = field(default_factory=set)
    distance: list[float] = field(default_factory=list)  # NaN = n/a
    smooth: list[float] = field(default_factory=list)  # NaN = none
    speed: list[float] = field(default_factory=list)  # NaN = none


def _jitter(order: np.ndarray, values: np.ndarray) -> float | None:
    """Std of the differences between consecutive processed frames where both
    values exist; None with fewer than two differences."""
    consecutive = np.diff(order) == 1
    diffs = np.diff(values)[consecutive]
    diffs = diffs[np.isfinite(diffs)]
    return float(diffs.std()) if len(diffs) >= 2 else None


class TrackSummary:
    """Accumulates per-frame track ids and estimates into metrics.json["tracks"].

    Same pattern as core.agreement.AgreementStats: pure, fed one FrameResult
    per processed frame (frames without track ids only advance the frame
    count, so coverage and gaps are measured in processed frames).
    """

    def __init__(self, *, min_duration_s: float = 1.0) -> None:
        self.min_duration_s = min_duration_s
        self._n_frames = 0
        self._tracks: dict[int, _TrackRecord] = {}

    def add(self, result: FrameResult) -> None:
        """Add one processed frame."""
        order = self._n_frames
        self._n_frames += 1
        inst = result.instances
        if inst.track_id is None:
            return
        temporal = result.temporal or (None,) * len(inst)
        for i, (tid, members, stats, est) in enumerate(
            zip(inst.track_id, inst.provenance(), result.stats, temporal, strict=True)
        ):
            rec = self._tracks.setdefault(int(tid), _TrackRecord())
            rec.order.append(order)
            rec.frame.append(result.frame.index)
            rec.t.append(result.frame.t)
            rec.label.append(inst.label[i])
            rec.sources.update(m.source for m in members)
            rec.distance.append(math.nan if stats.distance is None else stats.distance)
            rec.smooth.append(math.nan if est is None else est.distance)
            rec.speed.append(math.nan if est is None else est.rel_speed)

    def summary(self) -> dict[str, Any]:
        """JSON summary for metrics.json["tracks"]:
        {"n_tracks", "short_tracks" (shorter than min_duration_s), "min_duration_s",
         "per_track": [{"id", "label" (most frequent), "labels" {label: n},
           "label_changes", "sources", "first_frame", "last_frame", "first_t",
           "duration_s", "n_frames", "coverage" (n_frames / frames spanned),
           "gaps", "max_gap_s",
           "distance_m" {"min", "median", "max"},
           "jitter_raw_m", "jitter_smooth_m" (std of frame-to-frame differences),
           "rel_speed_mps" {"p5", "median", "p95"} (kalman only)}]}
        Frames are processed frames (stride aware); "max_gap_s" is the longest
        time between two consecutive observations around a gap (the quantity
        compared with the tracker's max_age_s), 0 without gaps. Statistics
        without data are null.
        """
        per_track = []
        short = 0
        for tid, rec in self._tracks.items():
            duration = rec.t[-1] - rec.t[0]
            if duration < self.min_duration_s:
                short += 1
                continue
            per_track.append(_track_summary(tid, rec, duration))
        return {
            "n_tracks": len(self._tracks),
            "short_tracks": short,
            "min_duration_s": self.min_duration_s,
            "per_track": per_track,
        }


def _track_summary(tid: int, rec: _TrackRecord, duration: float) -> dict[str, Any]:
    order = np.asarray(rec.order)
    t = np.asarray(rec.t)
    distance = np.asarray(rec.distance)
    speed = np.asarray(rec.speed)
    steps = np.diff(order)
    gap = steps > 1
    labels = Counter(rec.label)
    valid = distance[np.isfinite(distance)]
    speed = speed[np.isfinite(speed)]
    return {
        "id": tid,
        "label": labels.most_common(1)[0][0],
        "labels": dict(labels.most_common()),
        "label_changes": sum(a != b for a, b in zip(rec.label, rec.label[1:])),
        "sources": sorted(rec.sources),
        "first_frame": rec.frame[0],
        "last_frame": rec.frame[-1],
        "first_t": rec.t[0],
        "duration_s": duration,
        "n_frames": len(order),
        "coverage": len(order) / (int(order[-1] - order[0]) + 1),
        "gaps": int(gap.sum()),
        "max_gap_s": float(np.diff(t)[gap].max()) if gap.any() else 0.0,
        "distance_m": (
            {
                "min": float(valid.min()),
                "median": float(np.median(valid)),
                "max": float(valid.max()),
            }
            if len(valid)
            else None
        ),
        "jitter_raw_m": _jitter(order, distance),
        "jitter_smooth_m": _jitter(order, np.asarray(rec.smooth)),
        "rel_speed_mps": (
            {
                "p5": float(np.percentile(speed, 5)),
                "median": float(np.median(speed)),
                "p95": float(np.percentile(speed, 95)),
            }
            if len(speed)
            else None
        ),
    }
