"""Plane measurement of task_0009 (motion/measurement.py): reference point,
back-projection through MapGeometry and the depth-map intrinsics, z vs
euclidean distance, covariance propagation (against Monte Carlo) and the
inverse projection."""

import math

import numpy as np
import pytest

from av_perception_lab.core.types import CameraIntrinsics, MapGeometry
from av_perception_lab.distance import BoxDepthStats
from av_perception_lab.motion.config import MeasurementConfig
from av_perception_lab.motion.measurement import (
    MIN_PROJECT_Z_M,
    measure,
    project_points,
    reference_pixel,
)

# 1920x1080 frame on a 504x280 map with a crop offset, as DA3 *_crop.
GEOMETRY = MapGeometry(1920, 1080, 504, 280, 504 / 1920, 280 / 1080, -2.0, -1.0)
K_MAP = CameraIntrinsics(380.0, 375.0, 250.0, 139.0)
BOX = (900.0, 400.0, 1100.0, 700.0)


def stats(distance=20.0, std=0.4, map_box=(230, 100, 290, 180)) -> BoxDepthStats:
    nan = math.nan
    return BoxDepthStats(
        map_box, 100, 100, 100, 1.0, nan, nan, std, nan, nan, nan, nan, nan, nan, nan,
        distance,
    )  # fmt: skip


def test_reference_pixel():
    assert reference_pixel(BOX, "bottom_center") == (1000.0, 700.0)
    assert reference_pixel(BOX, "center") == (1000.0, 550.0)


def test_measure_round_trips_through_project_points():
    m = measure(BOX, stats(), K_MAP, GEOMETRY, MeasurementConfig(), "z")
    assert m.xz[1] == pytest.approx(20.0)
    assert m.ref_px == (1000.0, 700.0)
    xyz = np.array([[m.xz[0], m.y, m.xz[1]]])
    np.testing.assert_allclose(project_points(xyz, K_MAP, GEOMETRY), [m.ref_px])
    # The map pixel of the reference point back-projects onto the ray (x/z = a).
    ((u, _),) = GEOMETRY.frame_to_map_points([m.ref_px])
    assert m.xz[0] / m.xz[1] == pytest.approx((u - K_MAP.cx) / K_MAP.fx)


def test_euclidean_distance_is_turned_into_z_at_the_sampled_centre():
    box = stats(map_box=(400, 200, 460, 260))  # off-axis region
    z = measure(BOX, box, K_MAP, GEOMETRY, MeasurementConfig(), "z")
    e = measure(BOX, box, K_MAP, GEOMETRY, MeasurementConfig(), "euclidean")
    ac = (430 - K_MAP.cx) / K_MAP.fx
    bc = (230 - K_MAP.cy) / K_MAP.fy
    factor = np.sqrt(1 + ac**2 + bc**2)
    assert e.xz[1] == pytest.approx(20.0 / factor)
    np.testing.assert_allclose(e.xz, z.xz / factor)
    np.testing.assert_allclose(e.cov, z.cov / factor**2, rtol=1e-6)


def test_no_distance_no_measurement():
    assert (
        measure(BOX, stats(distance=None), K_MAP, GEOMETRY, MeasurementConfig(), "z")
        is None
    )


def test_covariance_matches_monte_carlo():
    cfg = MeasurementConfig(pixel_std=4.0)
    m = measure(BOX, stats(std=0.6), K_MAP, GEOMETRY, cfg, "z")
    sigma_d = max(cfg.std_scale * 0.6, cfg.rel_floor * 20.0)
    rng = np.random.default_rng(0)
    n = 200_000
    d = 20.0 + rng.normal(0, sigma_d, n)
    u_frame = 1000.0 + rng.normal(0, cfg.pixel_std, n)
    u_map = u_frame * GEOMETRY.sx + GEOMETRY.ox
    x = (u_map - K_MAP.cx) / K_MAP.fx * d
    samples = np.stack([x, d], axis=1)
    np.testing.assert_allclose(np.cov(samples.T), m.cov, rtol=0.02, atol=1e-6)


def test_distance_sigma_floor_without_std():
    m = measure(BOX, stats(std=float("nan")), K_MAP, GEOMETRY, MeasurementConfig(), "z")
    assert np.sqrt(m.cov[1, 1]) == pytest.approx(MeasurementConfig().rel_floor * 20.0)


def test_project_points_cuts_near_and_behind():
    xyz = np.array(
        [[0.0, 1.0, 10.0], [0.0, 1.0, MIN_PROJECT_Z_M / 2], [1.0, 0.0, -5.0]]
    )
    px = project_points(xyz, K_MAP, GEOMETRY)
    assert np.isfinite(px[0]).all() and np.isnan(px[1:]).all()


def test_map_geometry_points_are_the_box_affine():
    pts = np.array([[10.0, 20.0], [1900.0, 1070.0]])
    on_map = GEOMETRY.frame_to_map_points(pts)
    box = GEOMETRY.frame_to_map((10.0, 20.0, 1900.0, 1070.0))
    np.testing.assert_allclose(on_map.ravel(), box)
    np.testing.assert_allclose(GEOMETRY.map_to_frame_points(on_map), pts)
