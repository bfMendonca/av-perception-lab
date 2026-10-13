"""Per-instance object distance from a metric depth map (pure numpy).

The sampled region is either the detector box or the instance mask. A box is
mapped onto the depth map through `MapGeometry` (per-axis affine: Depth
Anything 3 resizes anisotropically and may centre-crop) and shrunk; a mask
(frame resolution) is resampled by nearest neighbour inside the unshrunk box
window and eroded in depth-map pixels, because depth edges mix object and
background. Both then drop invalid / ineligible pixels, reject outliers and
reduce the remaining depths with the same robust statistic. No torch, OpenCV
or PIL imports: this module must stay importable and unit-testable on a
machine without the GPU stack.
"""

import dataclasses
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

# Contract types live in core.types; imported here (and so still importable from
# this module) because the distance math is expressed with them.
from det_depth_distance.core.types import CameraIntrinsics, MapGeometry

OutlierMethod = Literal["none", "iqr", "mad", "sigma"]
DistanceType = Literal["z", "euclidean"]
# Region sampled for each instance (see instance_depth_stats).
RegionPolicy = Literal["mask", "box", "both"]
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
# Column order of BoxDepthStats.mask_row(), appended after the other CSV columns.
MASK_ROW_FIELDS: tuple[str, ...] = ("used_mask", "mask_px", "mask_erode")

_PERCENTILE_RE = re.compile(r"p(\d+(?:\.\d+)?)")
_TRIM_FRACTION = 0.1  # per side, for the "trimmed" statistic
_MODE_BIN_M = 0.25
_MODE_MIN_PEAK = 0.2  # a histogram peak counts if it holds >= 20 % of the tallest bin


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
    """Parameters of `box_depth_stats` and `mask_depth_stats`, validated on construction."""

    box_shrink: float = 0.1
    outlier: OutlierMethod = "iqr"
    outlier_k: float | None = None  # None -> DEFAULT_OUTLIER_K[outlier]
    stat: str = "median"
    distance_type: DistanceType = "z"
    min_valid_px: int = 20
    max_range: float = 0.0  # 0 = no limit
    mask_erode: int = 1  # instance-mask erosion, depth-map pixels

    def __post_init__(self) -> None:
        if not 0.0 <= self.box_shrink < 0.5:
            raise ValueError(f"box_shrink must be in [0, 0.5), got {self.box_shrink}")
        if self.outlier not in DEFAULT_OUTLIER_K:
            raise ValueError(f"unknown outlier method {self.outlier!r}")
        if self.distance_type not in ("z", "euclidean"):
            raise ValueError(f"unknown distance type {self.distance_type!r}")
        if self.min_valid_px < 0 or self.max_range < 0:
            raise ValueError("min_valid_px and max_range must be >= 0")
        if self.mask_erode < 0:
            raise ValueError(f"mask_erode must be >= 0, got {self.mask_erode}")
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
    """Depth statistics of one region (box or instance mask), in metres.

    `distance` is None when too few inliers remain; `mask_px` / `mask_erode`
    are set only when the region was the instance mask (`used_mask`).
    """

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
    mask_px: int | None = None  # depth-map pixels of the eroded mask region
    mask_erode: int | None = None  # erosion actually applied (map pixels)

    @property
    def used_mask(self) -> bool:
        """Whether the statistics were computed over the instance mask."""
        return self.mask_px is not None

    def mask_row(self) -> dict[str, int | None]:
        """Flat dict keyed by MASK_ROW_FIELDS (used_mask as 0/1), for CSV output."""
        values = (int(self.used_mask), self.mask_px, self.mask_erode)
        return dict(zip(MASK_ROW_FIELDS, values, strict=True))

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


def resample_mask(
    frame_mask: np.ndarray, geom: MapGeometry, map_box: tuple[int, int, int, int]
) -> np.ndarray:
    """Sample a frame-resolution 2-D array (mask or labels) at the centres of a
    depth-map window.

    Map pixel (r, c) reads the frame pixel under its centre,
    (floor((c + 0.5 - ox) / sx), floor((r + 0.5 - oy) / sy)), clipped to the
    frame: nearest neighbour with separate x/y scales and crop offsets. Mask
    edges are half-open, so a box edge falling exactly on a pixel centre is
    excluded (box_to_map_pixels includes it). Any dtype works and is preserved
    (boolean masks, integer class labels).

    Returns:
        (r1 - r0, c1 - c0) array of the input dtype for map_box = (c0, r0, c1, r1).
    """
    c0, r0, c1, r1 = map_box
    h, w = frame_mask.shape
    xs = np.floor((np.arange(c0, c1) + 0.5 - geom.ox) / geom.sx).astype(np.intp)
    ys = np.floor((np.arange(r0, r1) + 0.5 - geom.oy) / geom.sy).astype(np.intp)
    return frame_mask[np.ix_(np.clip(ys, 0, h - 1), np.clip(xs, 0, w - 1))]


def erode_mask(mask: np.ndarray, iterations: int) -> np.ndarray:
    """Binary erosion with a 3x3 square, `iterations` times (cv2.erode
    equivalent with a zero border). Pixels outside the array count as
    background, so regions touching the window edge erode there too."""
    out = mask.astype(bool, copy=True)
    for _ in range(iterations):
        # The 3x3 minimum is separable: a horizontal then a vertical 3-tap AND.
        padded = np.pad(out, 1)
        rows = padded[:, :-2] & padded[:, 1:-1] & padded[:, 2:]
        out = rows[:-2] & rows[1:-1] & rows[2:]
    return out


def _check_inputs(
    depth_m: np.ndarray,
    cfg: DistanceConfig,
    intrinsics: CameraIntrinsics | None,
    mask: np.ndarray | None,
) -> None:
    if mask is not None and mask.shape != depth_m.shape:
        raise ValueError(f"mask shape {mask.shape} != depth shape {depth_m.shape}")
    if cfg.distance_type == "euclidean" and intrinsics is None:
        raise ValueError("euclidean distance needs camera intrinsics")


def _region_stats(
    depth_m: np.ndarray,
    map_box: tuple[int, int, int, int],
    cfg: DistanceConfig,
    *,
    region: np.ndarray | None,
    intrinsics: CameraIntrinsics | None,
    mask: np.ndarray | None,
) -> BoxDepthStats:
    """Statistics over the pixels of `map_box` that are valid, eligible (`mask`)
    and inside `region` (window-shaped bool; None = the whole window)."""
    c0, r0, c1, r1 = map_box
    patch = depth_m[r0:r1, c0:c1].astype(np.float64)
    valid = np.isfinite(patch) & (patch > 0)
    if cfg.max_range > 0:
        valid &= patch <= cfg.max_range
    if mask is not None:
        valid &= mask[r0:r1, c0:c1]
    if region is not None:
        valid &= region
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
        mask: optional boolean (map_h, map_w) eligibility mask (e.g. not-sky);
            instance masks go through `mask_depth_stats` instead.

    Returns:
        BoxDepthStats; `distance` is None when fewer than cfg.min_valid_px
        inliers remain (rendered as "n/a").

    Raises:
        ValueError: euclidean distance requested without intrinsics, or
            mask/depth shape mismatch.
    """
    _check_inputs(depth_m, cfg, intrinsics, mask)
    map_box = box_to_map_pixels(xyxy, geom, cfg.box_shrink)
    return _region_stats(
        depth_m, map_box, cfg, region=None, intrinsics=intrinsics, mask=mask
    )


def mask_depth_stats(
    depth_m: np.ndarray,
    xyxy: Sequence[float],
    instance_mask: np.ndarray,
    geom: MapGeometry,
    cfg: DistanceConfig,
    *,
    intrinsics: CameraIntrinsics | None = None,
    mask: np.ndarray | None = None,
) -> BoxDepthStats | None:
    """Depth statistics of one instance over its mask instead of its box.

    The window is the unshrunk box on the depth map (mask pixels outside the box
    are ignored); the resampled mask is eroded by up to cfg.mask_erode map
    pixels, backing off while fewer than max(cfg.min_valid_px, 1) pixels would
    remain so thin objects keep a region. Validity, eligibility (`mask`),
    outlier rejection and the statistic are those of `box_depth_stats`.

    Args:
        instance_mask: boolean (frame_h, frame_w) mask of the instance.
        (other arguments as in `box_depth_stats`)

    Returns:
        BoxDepthStats with mask_px / mask_erode set, or None when the mask has
        no pixel inside the window (caller falls back to the box).

    Raises:
        ValueError: as `box_depth_stats`, or an instance mask that is not
            frame-sized.
    """
    _check_inputs(depth_m, cfg, intrinsics, mask)
    if instance_mask.shape != (geom.frame_h, geom.frame_w):
        raise ValueError(
            f"instance mask shape {instance_mask.shape} != frame "
            f"{(geom.frame_h, geom.frame_w)}"
        )
    map_box = box_to_map_pixels(xyxy, geom, 0.0)
    region = resample_mask(instance_mask, geom, map_box)
    if not region.any():
        return None
    # Erosion only removes pixels, so the region shrinks with every step: stop
    # at the first step that would leave too few pixels for a distance.
    need, applied, eroded = max(cfg.min_valid_px, 1), 0, region
    for k in range(1, cfg.mask_erode + 1):
        eroded = erode_mask(eroded, 1)
        if np.count_nonzero(eroded) < need:
            break
        region, applied = eroded, k
    stats = _region_stats(
        depth_m, map_box, cfg, region=region, intrinsics=intrinsics, mask=mask
    )
    return dataclasses.replace(
        stats, mask_px=int(np.count_nonzero(region)), mask_erode=applied
    )


def instance_depth_stats(
    depth_m: np.ndarray,
    xyxy: Sequence[float],
    geom: MapGeometry,
    cfg: DistanceConfig,
    *,
    region: RegionPolicy,
    instance_mask: np.ndarray | None = None,
    intrinsics: CameraIntrinsics | None = None,
    mask: np.ndarray | None = None,
) -> tuple[BoxDepthStats, BoxDepthStats | None]:
    """Apply the distance region policy to one instance.

    "box": box statistics. "mask": mask statistics when the instance has a
    non-empty mask in its box, else box statistics. "both": as "mask", plus the
    box statistics as second element when the mask was used (for comparison).

    Returns:
        (primary, box_alternative_or_None).
    """
    if region != "box" and instance_mask is not None:
        by_mask = mask_depth_stats(
            depth_m,
            xyxy,
            instance_mask,
            geom,
            cfg,
            intrinsics=intrinsics,
            mask=mask,
        )
        if by_mask is not None:
            if region == "both":
                return by_mask, box_depth_stats(
                    depth_m, xyxy, geom, cfg, intrinsics=intrinsics, mask=mask
                )
            return by_mask, None
    by_box = box_depth_stats(depth_m, xyxy, geom, cfg, intrinsics=intrinsics, mask=mask)
    return by_box, None
