"""Per-box object distance from a metric depth map (pure numpy).

Maps a detector box (frame pixels) onto the depth map through `MapGeometry`
(per-axis affine: Depth Anything 3 resizes anisotropically and may
centre-crop), shrinks it, drops invalid / masked pixels, rejects background
outliers and reduces the remaining depths to one distance with a robust
statistic. No torch, OpenCV or PIL imports: this module must stay importable
and unit-testable on a machine without the GPU stack.
"""

import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

OutlierMethod = Literal["none", "iqr", "mad", "sigma"]
DistanceType = Literal["z", "euclidean"]
DEFAULT_OUTLIER_K: dict[str, float] = {
    "none": 0.0,
    "iqr": 1.5,
    "mad": 3.5,
    "sigma": 3.0,
}

# Column order of BoxDepthStats.to_row(); also the CSV header for these fields.
STATS_ROW_FIELDS: tuple[str, ...] = (
    "map_c0",
    "map_r0",
    "map_c1",
    "map_r1",
    "n_box_px",
    "n_valid",
    "n_inliers",
    "inlier_frac",
    "mean",
    "median",
    "std",
    "var",
    "p10",
    "p25",
    "p75",
    "p90",
    "min",
    "max",
    "distance_m",
)

_PERCENTILE_RE = re.compile(r"p(\d+(?:\.\d+)?)")
_TRIM_FRACTION = 0.1  # per side, for the "trimmed" statistic
_MODE_BIN_M = 0.25
_MODE_MIN_PEAK = 0.2  # a histogram peak counts if it holds >= 20 % of the tallest bin


@dataclass(frozen=True, slots=True)
class MapGeometry:
    """Per-axis affine from frame pixels to depth-map pixels.

    `x_map = x * sx + ox`, `y_map = y * sy + oy`, applied to continuous box
    edges. sx != sy in general (DA3 rounds each side to a multiple of 14), and
    ox/oy are negative crop offsets for the *_crop processing methods.
    """

    frame_w: int
    frame_h: int
    map_w: int
    map_h: int
    sx: float
    sy: float
    ox: float = 0.0
    oy: float = 0.0

    def frame_to_map(self, xyxy: Sequence[float]) -> tuple[float, float, float, float]:
        """Map box edges (x0, y0, x1, y1) from frame to depth-map coordinates."""
        x0, y0, x1, y1 = xyxy
        return (
            x0 * self.sx + self.ox,
            y0 * self.sy + self.oy,
            x1 * self.sx + self.ox,
            y1 * self.sy + self.oy,
        )

    def map_to_frame(self, xyxy: Sequence[float]) -> tuple[float, float, float, float]:
        """Inverse of `frame_to_map`."""
        x0, y0, x1, y1 = xyxy
        return (
            (x0 - self.ox) / self.sx,
            (y0 - self.oy) / self.sy,
            (x1 - self.ox) / self.sx,
            (y1 - self.oy) / self.sy,
        )


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    """Pinhole intrinsics in pixels, expressed at the depth-map resolution."""

    fx: float
    fy: float
    cx: float
    cy: float


def parse_stat(stat: str) -> str:
    """Validate a distance statistic name (median, mean, trimmed, mode or pXX, 0 < XX < 100).

    Used as the argparse `type=` of --distance-stat, so it returns the value unchanged.
    """
    if stat in _STATS:
        return stat
    match = _PERCENTILE_RE.fullmatch(stat)
    if match and 0 < float(match.group(1)) < 100:
        return stat
    raise ValueError(
        f"invalid distance statistic {stat!r}: use {', '.join(_STATS)} or pXX with 0 < XX < 100"
    )


@dataclass(frozen=True, slots=True)
class DistanceConfig:
    """Parameters of `box_depth_stats`, validated on construction."""

    box_shrink: float = 0.1
    outlier: OutlierMethod = "iqr"
    outlier_k: float | None = None  # None -> DEFAULT_OUTLIER_K[outlier]
    stat: str = "median"
    distance_type: DistanceType = "z"
    min_valid_px: int = 20
    max_range: float = 0.0  # 0 = no limit

    def __post_init__(self) -> None:
        if not 0.0 <= self.box_shrink < 0.5:
            raise ValueError(f"box_shrink must be in [0, 0.5), got {self.box_shrink}")
        if self.outlier not in DEFAULT_OUTLIER_K:
            raise ValueError(f"unknown outlier method {self.outlier!r}")
        if self.distance_type not in ("z", "euclidean"):
            raise ValueError(f"unknown distance type {self.distance_type!r}")
        if self.min_valid_px < 0 or self.max_range < 0:
            raise ValueError("min_valid_px and max_range must be >= 0")
        parse_stat(self.stat)

    @property
    def k(self) -> float:
        """Outlier threshold, falling back to the method's default."""
        return (
            DEFAULT_OUTLIER_K[self.outlier]
            if self.outlier_k is None
            else self.outlier_k
        )


@dataclass(frozen=True, slots=True)
class BoxDepthStats:
    """Depth statistics of one box (metres). `distance` is None when too few inliers remain."""

    map_box: tuple[int, int, int, int]  # (c0, r0, c1, r1), half-open, depth-map pixels
    n_box_px: int
    n_valid: int
    n_inliers: int
    inlier_frac: float
    mean: float
    median: float
    std: float
    var: float
    p10: float
    p25: float
    p75: float
    p90: float
    min: float
    max: float
    distance: float | None

    def to_row(self) -> dict[str, float | int | None]:
        """Flat dict keyed by STATS_ROW_FIELDS, for CSV output."""
        values = (
            *self.map_box,
            self.n_box_px,
            self.n_valid,
            self.n_inliers,
            self.inlier_frac,
            self.mean,
            self.median,
            self.std,
            self.var,
            self.p10,
            self.p25,
            self.p75,
            self.p90,
            self.min,
            self.max,
            self.distance,
        )
        return dict(zip(STATS_ROW_FIELDS, values, strict=True))


def box_to_map_pixels(
    xyxy: Sequence[float], geom: MapGeometry, shrink: float
) -> tuple[int, int, int, int]:
    """Depth-map pixel range (c0, r0, c1, r1), half-open, covered by a shrunk box.

    The box is mapped to the depth map, shrunk by `shrink` of its width/height
    on each side, and a pixel is included when its centre lies inside. The
    result is clipped to the map and may be empty (c1 <= c0 or r1 <= r0).
    """
    x0, y0, x1, y1 = geom.frame_to_map(xyxy)
    dx, dy = (x1 - x0) * shrink, (y1 - y0) * shrink
    x0, x1, y0, y1 = x0 + dx, x1 - dx, y0 + dy, y1 - dy
    c0 = min(max(math.ceil(x0 - 0.5), 0), geom.map_w)
    r0 = min(max(math.ceil(y0 - 0.5), 0), geom.map_h)
    c1 = min(max(math.floor(x1 - 0.5) + 1, c0), geom.map_w)
    r1 = min(max(math.floor(y1 - 0.5) + 1, r0), geom.map_h)
    return c0, r0, c1, r1


def _keep_all(values: np.ndarray, k: float) -> np.ndarray:
    return np.ones(values.shape, dtype=bool)


def _iqr_inliers(values: np.ndarray, k: float) -> np.ndarray:
    q1, q3 = np.percentile(values, [25, 75])
    spread = k * (q3 - q1)
    return (values >= q1 - spread) & (values <= q3 + spread)


def _mad_inliers(values: np.ndarray, k: float) -> np.ndarray:
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    # MAD == 0: more than half the pixels share one depth, nothing to call an outlier.
    if mad == 0:
        return np.ones(values.shape, dtype=bool)
    return np.abs(0.6745 * (values - median) / mad) <= k


def _sigma_inliers(values: np.ndarray, k: float) -> np.ndarray:
    return np.abs(values - values.mean()) <= k * values.std()


_OUTLIER_FILTERS: dict[str, Callable[[np.ndarray, float], np.ndarray]] = {
    "none": _keep_all,
    "iqr": _iqr_inliers,
    "mad": _mad_inliers,
    "sigma": _sigma_inliers,
}


def inlier_mask(values: np.ndarray, method: OutlierMethod, k: float) -> np.ndarray:
    """Return a boolean mask of values kept by `method` with threshold `k`.

    iqr: [Q1 - k*IQR, Q3 + k*IQR]; mad: |0.6745 (x - median) / MAD| <= k
    (keeps everything when MAD == 0); sigma: |x - mean| <= k*std; none: all.
    """
    if values.size == 0:
        return np.zeros(0, dtype=bool)
    return _OUTLIER_FILTERS[method](values, k)


def _trimmed_mean(values: np.ndarray) -> float:
    ordered = np.sort(values)
    cut = int(ordered.size * _TRIM_FRACTION)
    return float(ordered[cut : ordered.size - cut].mean())


def _nearest_mode(values: np.ndarray) -> float:
    # Histogram peaks: the nearest sufficiently tall one is the object in front;
    # farther peaks are background (road, vegetation) leaking into the box.
    lo, hi = float(values.min()), float(values.max())
    edges = np.arange(lo, hi + _MODE_BIN_M, _MODE_BIN_M)
    if edges.size < 2:
        return float(values.mean())
    counts, edges = np.histogram(values, bins=edges)
    padded = np.concatenate(([-1], counts, [-1]))
    is_peak = (counts >= padded[:-2]) & (counts >= padded[2:])
    is_peak &= counts >= _MODE_MIN_PEAK * counts.max()
    i = int(np.flatnonzero(is_peak)[0])
    near = values[
        (values >= edges[max(i - 1, 0)]) & (values <= edges[min(i + 2, edges.size - 1)])
    ]
    return float(near.mean())


_STATS: dict[str, Callable[[np.ndarray], float]] = {
    "median": lambda v: float(np.median(v)),
    "mean": lambda v: float(v.mean()),
    "trimmed": _trimmed_mean,
    "mode": _nearest_mode,
}


def robust_distance(values: np.ndarray, stat: str) -> float:
    """Reduce inlier depths to one distance: median, mean, trimmed (10 % per
    side), mode (nearest histogram peak, 0.25 m bins) or pXX percentile."""
    if stat in _STATS:
        return _STATS[stat](values)
    return float(np.percentile(values, float(parse_stat(stat)[1:])))


def ray_factors(
    map_box: tuple[int, int, int, int], intrinsics: CameraIntrinsics
) -> np.ndarray:
    """Per-pixel factor turning z-depth into euclidean range for a map box:
    sqrt(1 + ((u - cx) / fx)**2 + ((v - cy) / fy)**2) at pixel centres."""
    c0, r0, c1, r1 = map_box
    u = (np.arange(c0, c1) + 0.5 - intrinsics.cx) / intrinsics.fx
    v = (np.arange(r0, r1) + 0.5 - intrinsics.cy) / intrinsics.fy
    return np.sqrt(1.0 + u[None, :] ** 2 + v[:, None] ** 2)


def box_depth_stats(
    depth_m: np.ndarray,
    xyxy: Sequence[float],
    geom: MapGeometry,
    cfg: DistanceConfig,
    *,
    intrinsics: CameraIntrinsics | None = None,
    mask: np.ndarray | None = None,
) -> BoxDepthStats:
    """Estimate the camera-to-object distance for one detection box.

    Args:
        depth_m: (map_h, map_w) depth in metres along the optical axis.
        xyxy: box edges in frame pixels, as returned by the detector.
        geom: frame -> depth-map mapping for this video.
        cfg: shrink, outlier, statistic and validity parameters.
        intrinsics: depth-map-resolution intrinsics; required when
            cfg.distance_type == "euclidean".
        mask: optional boolean (map_h, map_w) eligibility mask (e.g. not-sky,
            instance segmentation).

    Returns:
        BoxDepthStats; `distance` is None when fewer than cfg.min_valid_px
        inliers remain (rendered as "n/a").

    Raises:
        ValueError: euclidean distance requested without intrinsics, or
            mask/depth shape mismatch.
    """
    if mask is not None and mask.shape != depth_m.shape:
        raise ValueError(f"mask shape {mask.shape} != depth shape {depth_m.shape}")
    if cfg.distance_type == "euclidean" and intrinsics is None:
        raise ValueError("euclidean distance needs camera intrinsics")

    map_box = box_to_map_pixels(xyxy, geom, cfg.box_shrink)
    c0, r0, c1, r1 = map_box
    patch = depth_m[r0:r1, c0:c1].astype(np.float64)
    valid = np.isfinite(patch) & (patch > 0)
    if cfg.max_range > 0:
        valid &= patch <= cfg.max_range
    if mask is not None:
        valid &= mask[r0:r1, c0:c1]
    if cfg.distance_type == "euclidean":
        patch = patch * ray_factors(map_box, intrinsics)

    values = patch[valid]
    inliers = values[inlier_mask(values, cfg.outlier, cfg.k)]
    if inliers.size:
        p10, p25, p75, p90 = (
            float(p) for p in np.percentile(inliers, [10, 25, 75, 90])
        )
        summary = (
            float(inliers.mean()),
            float(np.median(inliers)),
            float(inliers.std()),
            float(inliers.var()),
            p10,
            p25,
            p75,
            p90,
            float(inliers.min()),
            float(inliers.max()),
        )
    else:
        summary = (math.nan,) * 10
    enough = inliers.size > 0 and inliers.size >= cfg.min_valid_px
    return BoxDepthStats(
        map_box,
        int(patch.size),
        int(values.size),
        int(inliers.size),
        inliers.size / values.size if values.size else math.nan,
        *summary,
        distance=robust_distance(inliers, cfg.stat) if enough else None,
    )
