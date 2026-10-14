"""Reference x detection matching, one match per reference, processed frame
and source group (detector, or the fused output).

Reference selectors are in pixels of the original video; when the run
undistorted its frames, `points_transform` (Undistorter.undistort_points)
moves them into the run's undistorted pixels first.
"""

from collections.abc import Callable
from dataclasses import dataclass
from math import floor

import numpy as np

from det_depth_distance.core.boxes import iou_matrix
from det_depth_distance.core.options import ConfigError
from det_depth_distance.core.types import CameraIntrinsics
from det_depth_distance.distance import ray_factors
from det_depth_distance.evaluation.references import Quantity, Reference, ReferenceSet
from det_depth_distance.evaluation.run_data import RunData

PointsTransform = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True, slots=True)
class Match:
    """The instance matched to a reference in one processed frame."""

    ref_id: str
    frame: int
    t: float
    source: str  # source group (RunData.sources)
    row: int | None  # detections.csv row; None = not detected
    iou: float  # box selectors; 0 for point/track_id/not detected
    reference_m: float  # reference distance in the run's quantity (z or euclidean)
    uncertainty_m: float


def reference_to_run_quantity(
    distance: float,
    quantity: Quantity,
    run_type: str,
    center_xy: tuple[float, float],
    intrinsics: CameraIntrinsics | None,
) -> tuple[float, bool]:
    """Express a reference distance in the run's distance type.

    "range" equals the run's "euclidean"; between range and z the factor is
    the ray factor of the frame pixel holding `center_xy` (reuses
    distance.ray_factors with frame-resolution K). Returns (distance,
    converted); without intrinsics nothing is converted.
    """
    same = (quantity == "range") == (run_type == "euclidean")
    if same or intrinsics is None:
        return distance, False
    c, r = floor(center_xy[0]), floor(center_xy[1])
    factor = float(ray_factors((c, r, c + 1, r + 1), intrinsics)[0, 0])
    return (distance / factor if quantity == "range" else distance * factor), True


def _selector_box(
    ref: Reference, transform: PointsTransform | None
) -> np.ndarray | None:
    """Reference box in the run's pixels: the bounding box of its transformed
    corners (undistortion bends straight edges)."""
    x0, y0, x1, y1 = ref.select.box
    if transform is None:
        return np.array([x0, y0, x1, y1])
    pts = transform(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]))
    return np.concatenate([pts.min(0), pts.max(0)])


def _pick(
    ref: Reference,
    cand: np.ndarray,
    boxes: np.ndarray,
    track_ids: np.ndarray | None,
    ref_box: np.ndarray | None,
    ref_point: np.ndarray | None,
    iou_min: float,
) -> tuple[int | None, float]:
    """(row, iou) of the matched candidate row, (None, 0) if none qualifies."""
    if not len(cand):
        return None, 0.0
    b = boxes[cand]
    if ref_box is not None:
        iou = iou_matrix(ref_box[None], b)[0]
        best = int(np.argmax(iou))
        return (
            (int(cand[best]), float(iou[best])) if iou[best] >= iou_min else (None, 0.0)
        )
    if ref_point is not None:
        x, y = ref_point
        inside = (b[:, 0] <= x) & (x <= b[:, 2]) & (b[:, 1] <= y) & (y <= b[:, 3])
        if not inside.any():
            return None, 0.0
        idx = np.flatnonzero(inside)
        area = (b[idx, 2] - b[idx, 0]) * (b[idx, 3] - b[idx, 1])
        centre = (b[idx, :2] + b[idx, 2:]) / 2
        dist = np.hypot(centre[:, 0] - x, centre[:, 1] - y)
        best = idx[np.lexsort((dist, area))[0]]  # smallest box, then nearest centre
        return int(cand[best]), 0.0
    hit = np.flatnonzero(track_ids[cand] == ref.select.track_id)
    return (int(cand[hit[0]]), 0.0) if len(hit) else (None, 0.0)


def match_references(
    run: RunData,
    refs: ReferenceSet,
    *,
    iou_min: float = 0.3,
    points_transform: PointsTransform | None = None,
) -> list[Match]:
    """Match every reference to one instance per processed frame and detector.

    Box selectors take the highest-IoU instance (>= iou_min), point selectors
    the smallest box containing the point, track selectors the instance with
    that id. Unmatched frames are kept (row None) so coverage counts misses.
    Range references are converted to z when the run reports z, using the
    run's frame intrinsics at the matched box centre.

    Raises:
        ConfigError: a track_id selector on a run without tracking, or a
            series that does not cover a matched frame.
    """
    col = run.columns
    boxes = np.column_stack([col["x0"], col["y0"], col["x1"], col["y1"]])
    labels = col["class_name"]
    row_sources = run.row_sources()
    track_ids = col.get("track_id")
    frame_rows: dict[int, np.ndarray] = {}
    if len(run):
        order = np.argsort(col["frame"], kind="stable")
        frames_sorted = col["frame"][order].astype(int)
        cuts = np.flatnonzero(np.diff(frames_sorted)) + 1
        for chunk in np.split(order, cuts):
            frame_rows[int(col["frame"][chunk[0]])] = chunk
    empty = np.zeros(0, dtype=int)
    frames = run.processed_frames()
    times = run.frame_times(frames)
    k = run.frame_intrinsics
    matches: list[Match] = []
    for ref in refs.references:
        sel = ref.select
        if sel.track_id is not None and not run.has_tracks:
            raise ConfigError(
                f"reference {ref.id}: selects track_id {sel.track_id}, "
                f"but run {run.name} has no tracking"
            )
        ref_box = _selector_box(ref, points_transform) if sel.box is not None else None
        ref_point = None
        if sel.point is not None:
            pt = np.array([sel.point], dtype=np.float64)
            ref_point = (points_transform(pt) if points_transform else pt)[0]
        for frame, t in zip(frames.tolist(), times.tolist()):
            if not ref.covers(frame, t):
                continue
            try:
                distance = ref.distance_at(t)
            except ValueError as exc:
                raise ConfigError(f"reference {ref.id}: {exc}") from None
            rows = frame_rows.get(frame, empty)
            if sel.label is not None:
                rows = rows[labels[rows] == sel.label]
            for source in run.sources:
                cand = rows[row_sources[rows] == source]
                row, iou = _pick(
                    ref, cand, boxes, track_ids, ref_box, ref_point, iou_min
                )
                value, scale = distance, 1.0
                if row is not None:
                    centre = (
                        (boxes[row, 0] + boxes[row, 2]) / 2,
                        (boxes[row, 1] + boxes[row, 3]) / 2,
                    )
                    value, _ = reference_to_run_quantity(
                        distance, ref.quantity, run.distance_type, centre, k
                    )
                    scale = value / distance
                matches.append(
                    Match(
                        ref.id,
                        frame,
                        t,
                        source,
                        row,
                        iou,
                        value,
                        refs.uncertainty(ref, distance) * scale,
                    )
                )
    return matches
