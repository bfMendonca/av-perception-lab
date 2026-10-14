"""Error statistics of estimated vs reference distances (pure numpy).

Errors are estimate minus reference (positive = estimated too far). At
level "reference" each reference contributes one error (the median over
its frames), because consecutive frames are strongly correlated; level
"frame" weights every frame. Estimates marked n/a are excluded from errors
and reported in coverage instead.
"""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from typing import Any, Literal

import numpy as np

from det_depth_distance.evaluation.matching import Match
from det_depth_distance.evaluation.run_data import INLIER_ESTIMATORS, RunData

# Reference distance bin edges (m); the last bin is open (">= 50").
DEFAULT_BINS: tuple[float, ...] = (0.0, 5.0, 10.0, 20.0, 30.0, 50.0)
Level = Literal["reference", "frame"]


@dataclass(frozen=True, slots=True)
class ErrorStats:
    """Signed/absolute/relative error summary of one group; NaN when n == 0."""

    n: int  # errors (frames or references, by level)
    n_refs: int  # distinct references
    reference_median_m: float  # median reference distance
    bias_m: float  # mean signed error
    median_signed_m: float
    mae_m: float
    median_abs_m: float
    p90_abs_m: float
    rmse_m: float
    mean_rel: float  # mean of error / reference
    mean_abs_rel: float
    median_abs_rel: float
    p90_abs_rel: float
    within_2u: float  # fraction with |error| <= 2 * uncertainty
    scale_median: float  # median of estimate / reference

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def error_stats(
    estimate: np.ndarray,
    reference: np.ndarray,
    uncertainty: np.ndarray | None = None,
    ref_ids: Sequence[str] | None = None,
) -> ErrorStats:
    """Statistics of `estimate - reference` (1-D arrays of finite values).

    `within_2u` is NaN without `uncertainty`; `n_refs` counts distinct
    `ref_ids` (each value is its own reference without them).
    """
    est = np.asarray(estimate, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    n = len(est)
    n_refs = len(set(ref_ids)) if ref_ids is not None else n
    if n == 0:
        nan = float("nan")
        values = [nan] * (len(fields(ErrorStats)) - 2)
        return ErrorStats(0, n_refs, *values)
    e = est - ref
    rel = e / ref
    within = (
        float(np.mean(np.abs(e) <= 2 * np.asarray(uncertainty)))
        if uncertainty is not None
        else float("nan")
    )
    return ErrorStats(
        n=n,
        n_refs=n_refs,
        reference_median_m=float(np.median(ref)),
        bias_m=float(e.mean()),
        median_signed_m=float(np.median(e)),
        mae_m=float(np.abs(e).mean()),
        median_abs_m=float(np.median(np.abs(e))),
        p90_abs_m=float(np.percentile(np.abs(e), 90)),
        rmse_m=float(np.sqrt(np.mean(e**2))),
        mean_rel=float(rel.mean()),
        mean_abs_rel=float(np.abs(rel).mean()),
        median_abs_rel=float(np.median(np.abs(rel))),
        p90_abs_rel=float(np.percentile(np.abs(rel), 90)),
        within_2u=within,
        scale_median=float(np.median(est / ref)),
    )


@dataclass(frozen=True, slots=True)
class Sample:
    """One estimator's value for one matched instance."""

    run: str
    ref_id: str
    frame: int
    source: str
    label: str  # class_name of the matched instance
    region: str  # "mask" | "box" (used_mask)
    mask_source: str  # "" without a mask
    track_id: int | None
    estimator: str
    estimate: float  # NaN = n/a
    reference: float
    uncertainty: float


def _text(run: RunData, column: str, row: int) -> str:
    values = run.columns.get(column)
    return "" if values is None else str(values[row])


def build_samples(
    run: RunData, matches: Sequence[Match], estimators: Sequence[str]
) -> list[Sample]:
    """One Sample per detected match and estimator present in the CSV.

    Inlier statistics (median, mean, p10...) count only where distance_m has a
    value (the region had min_valid_px valid pixels); other estimators carry
    their own validity (distance_smooth_m is predicted through n/a frames).
    """
    col = run.columns
    present = [e for e in estimators if e in col]
    used_mask = col.get("used_mask")
    track_ids = col.get("track_id")
    out = []
    for m in matches:
        if m.row is None:
            continue
        r = m.row
        valid_distance = np.isfinite(col["distance_m"][r])
        track = track_ids[r] if track_ids is not None else np.nan
        common = {
            "run": run.name,
            "ref_id": m.ref_id,
            "frame": m.frame,
            "source": m.source,
            "label": _text(run, "class_name", r),
            "region": "mask" if used_mask is not None and used_mask[r] == 1 else "box",
            "mask_source": _text(run, "mask_source", r),
            "track_id": int(track) if np.isfinite(track) else None,
            "reference": m.reference_m,
            "uncertainty": m.uncertainty_m,
        }
        for est in present:
            value = float(col[est][r])
            if est in INLIER_ESTIMATORS and not valid_distance:
                value = float("nan")
            out.append(Sample(estimator=est, estimate=value, **common))
    return out


def bin_label(distance: float, bins: Sequence[float]) -> str:
    """Label of the reference distance bin holding `distance` ("10-20", ">=50")."""
    i = int(np.searchsorted(bins, distance, side="right")) - 1
    if i >= len(bins) - 1:
        return f">={bins[-1]:g}"
    return f"{bins[max(i, 0)]:g}-{bins[max(i, 0) + 1]:g}"


def summarize(
    samples: Sequence[Sample],
    *,
    by: Sequence[str],
    level: Level = "reference",
    bins: Sequence[float] = DEFAULT_BINS,
) -> list[dict[str, Any]]:
    """Group samples by `by` (any Sample field plus "bin", the reference
    distance bin) and compute ErrorStats per group.

    n/a estimates are skipped. At level "reference" each reference of a group
    becomes one error: the median signed error over its frames, against the
    median reference distance. Rows are sorted by the group key.
    """
    groups: dict[tuple, list[Sample]] = defaultdict(list)
    for s in samples:
        if not np.isfinite(s.estimate):
            continue
        key = tuple(
            bin_label(s.reference, bins) if name == "bin" else getattr(s, name)
            for name in by
        )
        groups[key].append(s)
    rows = []
    for key in sorted(groups, key=lambda k: tuple(str(v) for v in k)):
        group = groups[key]
        if level == "reference":
            per_ref: dict[str, list[Sample]] = defaultdict(list)
            for s in group:
                per_ref[s.ref_id].append(s)
            ids = list(per_ref)
            ref = np.array([np.median([s.reference for s in per_ref[i]]) for i in ids])
            err = np.array(
                [np.median([s.estimate - s.reference for s in per_ref[i]]) for i in ids]
            )
            unc = np.array(
                [np.median([s.uncertainty for s in per_ref[i]]) for i in ids]
            )
            stats = error_stats(ref + err, ref, unc, ids)
        else:
            stats = error_stats(
                np.array([s.estimate for s in group]),
                np.array([s.reference for s in group]),
                np.array([s.uncertainty for s in group]),
                [s.ref_id for s in group],
            )
        rows.append({**dict(zip(by, key, strict=True)), **stats.to_dict()})
    return rows


def coverage_summary(matches: Sequence[Match], run: RunData) -> list[dict[str, Any]]:
    """Per (reference, source): processed frames in range, detections with a
    valid distance_m, detections with "n/a", misses, valid values per
    estimator, and distinct track ids of the matched instances (None without
    tracking; more than one means an id switch on that object)."""
    col = run.columns
    estimators = run.available_estimators()
    track_ids = col.get("track_id") if run.has_tracks else None
    groups: dict[tuple[str, str], list[Match]] = defaultdict(list)
    for m in matches:
        groups[(m.ref_id, m.source)].append(m)
    out = []
    for (ref_id, source), group in groups.items():
        rows = [m.row for m in group if m.row is not None]
        valid = int(np.isfinite(col["distance_m"][rows]).sum()) if rows else 0
        tracks = None
        if track_ids is not None:
            ids = track_ids[rows] if rows else np.zeros(0)
            tracks = len(set(ids[np.isfinite(ids)].astype(int).tolist()))
        out.append(
            {
                "ref_id": ref_id,
                "source": source,
                "frames": len(group),
                "detected_valid": valid,
                "detected_na": len(rows) - valid,
                "not_detected": len(group) - len(rows),
                "valid_by_estimator": {
                    e: int(np.isfinite(col[e][rows]).sum()) if rows else 0
                    for e in estimators
                },
                "track_ids": tracks,
            }
        )
    return out


def implied_focal(run: RunData, samples: Sequence[Sample]) -> float | None:
    """Focal (frame px) that would cancel the scale error of distance_m:
    focal * median over references of median(reference / estimate), where
    focal is (fx + fy) / 2 of the frame intrinsics (what DA3METRIC scales by;
    fx != fy with a calibration), else focal_frame_px.

    Only for "metric_focal" depth models (DA3METRIC: distance proportional to
    the focal); None otherwise or without valid samples.
    """
    k = run.frame_intrinsics
    focal = (k.fx + k.fy) / 2 if k is not None else run.focal_frame_px
    if run.depth_kind != "metric_focal" or focal is None:
        return None
    ratios: dict[str, list[float]] = defaultdict(list)
    for s in samples:
        if (
            s.run == run.name
            and s.estimator == "distance_m"
            and np.isfinite(s.estimate)
        ):
            ratios[s.ref_id].append(s.reference / s.estimate)
    if not ratios:
        return None
    per_ref = [float(np.median(v)) for v in ratios.values()]
    return focal * float(np.median(per_ref))
