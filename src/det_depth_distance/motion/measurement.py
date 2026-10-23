"""Instance -> position on the plane, and the inverse projection.

The box reference point (continuous frame pixels) is mapped to the depth map
with MapGeometry and back-projected with the depth-map intrinsics. With
distance_type "euclidean" the range is turned into z with the ray factor at
the centre of the sampled region (stats.map_box). The covariance propagates
the distance std (max(std_scale * std, rel_floor * d), see
temporal.distance_sigma) and the reference point's pixel std through the
Jacobian of x = a * z.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from det_depth_distance.core.types import CameraIntrinsics, MapGeometry
from det_depth_distance.distance import BoxDepthStats, DistanceType
from det_depth_distance.motion.config import MeasurementConfig, ReferencePoint
from det_depth_distance.temporal import distance_sigma

# Projected points nearer than this (or behind the camera) are not drawn.
MIN_PROJECT_Z_M: float = 1.0


@dataclass(frozen=True, slots=True)
class PlaneMeasurement:
    """Position of one instance on the camera plane."""

    xz: np.ndarray  # (2,) m: x right, z along the optical axis
    cov: np.ndarray  # (2, 2) m^2
    y: float  # m, camera y (+ down) of the reference point
    ref_px: tuple[float, float]  # the reference point, frame px


def reference_pixel(
    xyxy: Sequence[float], point: ReferencePoint
) -> tuple[float, float]:
    """Reference point of a box in frame pixels: middle of the bottom edge or centre."""
    x0, y0, x1, y1 = (float(v) for v in xyxy)
    u = (x0 + x1) / 2
    return (u, y1) if point == "bottom_center" else (u, (y0 + y1) / 2)


def measure(
    xyxy: Sequence[float],
    stats: BoxDepthStats,
    intrinsics: CameraIntrinsics,
    geometry: MapGeometry,
    cfg: MeasurementConfig,
    distance_type: DistanceType,
) -> PlaneMeasurement | None:
    """Position (x, z) and covariance of one instance, or None without a
    distance. `intrinsics` are at depth-map resolution; `xyxy` in frame pixels.
    """
    if stats.distance is None:
        return None
    ref = reference_pixel(xyxy, cfg.reference_point)
    ((u, v),) = geometry.frame_to_map_points(np.array([ref]))
    k = intrinsics
    a, b = (u - k.cx) / k.fx, (v - k.cy) / k.fy
    factor = 1.0
    if distance_type == "euclidean":
        # The range is a statistic of the sampled region: undo the ray factor
        # at its centre (the reference point may lie outside it).
        c0, r0, c1, r1 = stats.map_box
        ac = ((c0 + c1) / 2 - k.cx) / k.fx
        bc = ((r0 + r1) / 2 - k.cy) / k.fy
        factor = math.sqrt(1.0 + ac**2 + bc**2)
    z = stats.distance / factor
    sigma_z = (
        distance_sigma(
            stats.distance,
            stats.std,
            std_scale=cfg.std_scale,
            rel_floor=cfg.rel_floor,
        )
        / factor
    )
    sigma_u = cfg.pixel_std * geometry.sx  # depth-map px
    # (x, z) = (a(u) * z, z) as a function of (z, u)
    jac = np.array([[a, z / k.fx], [1.0, 0.0]])
    cov = jac @ np.diag([sigma_z**2, sigma_u**2]) @ jac.T
    return PlaneMeasurement(np.array([a * z, z]), cov, b * z, ref)


def project_points(
    xyz: np.ndarray, intrinsics: CameraIntrinsics, geometry: MapGeometry
) -> np.ndarray:
    """Inverse of `measure`'s back-projection: (K, 3) camera points -> (K, 2)
    frame pixels; rows with z < MIN_PROJECT_Z_M are NaN (behind or too close
    to the camera)."""
    xyz = np.asarray(xyz, dtype=np.float64)
    k = intrinsics
    out = np.full((len(xyz), 2), np.nan)
    ok = xyz[:, 2] >= MIN_PROJECT_Z_M
    x, y, z = xyz[ok].T
    uv = np.stack([k.fx * x / z + k.cx, k.fy * y / z + k.cy], axis=1)
    out[ok] = geometry.map_to_frame_points(uv)
    return out
