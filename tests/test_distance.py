import math

import numpy as np
import pytest

from det_depth_distance.distance import (
    STATS_ROW_FIELDS,
    CameraIntrinsics,
    DistanceConfig,
    MapGeometry,
    box_depth_stats,
    box_to_map_pixels,
    inlier_mask,
    parse_stat,
    ray_factors,
    robust_distance,
)

# 1920x1080 frame, DA3 @504 upper_bound_resize -> 504x280; upper_bound_crop -> 504x284 cropped to 280.
RESIZE = MapGeometry(1920, 1080, 504, 280, 504 / 1920, 280 / 1080)
CROP = MapGeometry(1920, 1080, 504, 280, 504 / 1920, 284 / 1080, 0.0, -2.0)
IDENTITY = MapGeometry(100, 100, 100, 100, 1.0, 1.0)


def object_on_background(fg: float = 10.0, bg: float = 40.0) -> np.ndarray:
    """100x100 map at `bg` m with a 60 x 100 (60 %) object at `fg` m in columns 20..80."""
    depth = np.full((100, 100), bg, dtype=np.float32)
    depth[:, 20:80] = fg
    return depth


class TestGeometry:
    def test_full_frame_box_covers_whole_map(self):
        assert box_to_map_pixels((0, 0, 1920, 1080), RESIZE, 0.0) == (0, 0, 504, 280)
        assert box_to_map_pixels((0, 0, 1920, 1080), CROP, 0.0) == (0, 0, 504, 280)

    def test_central_box_maps_to_expected_pixels(self):
        # 960 * 0.2625 = 252; 540 * 280/1080 = 140
        assert box_to_map_pixels((960, 540, 1000, 600), RESIZE, 0.0)[:2] == (252, 140)
        # crop shifts rows up by 2: 540 * 284/1080 - 2 = 140
        assert box_to_map_pixels((960, 540, 1000, 600), CROP, 0.0)[:2] == (252, 140)

    def test_round_trip(self):
        box = (100.0, 200.0, 300.0, 400.0)
        assert CROP.map_to_frame(CROP.frame_to_map(box)) == pytest.approx(box)

    def test_shrink_drops_edges(self):
        assert box_to_map_pixels((0, 0, 100, 100), IDENTITY, 0.1) == (10, 10, 90, 90)

    def test_box_outside_map_is_empty(self):
        c0, r0, c1, r1 = box_to_map_pixels((150, 150, 200, 200), IDENTITY, 0.0)
        assert c1 <= c0 and r1 <= r0


class TestStatistics:
    def test_median_rejects_background(self):
        stats = box_depth_stats(
            object_on_background(),
            (0, 0, 100, 100),
            IDENTITY,
            DistanceConfig(box_shrink=0.0),
        )
        assert stats.distance == pytest.approx(10.0, abs=0.01)

    def test_mean_without_outlier_rejection_is_biased(self):
        cfg = DistanceConfig(box_shrink=0.0, outlier="none", stat="mean")
        stats = box_depth_stats(object_on_background(), (0, 0, 100, 100), IDENTITY, cfg)
        assert 10.0 < stats.distance < 40.0

    def test_iqr_removes_far_tail(self):
        values = np.concatenate([np.full(95, 10.0), np.full(5, 40.0)])
        assert inlier_mask(values, "iqr", 1.5).sum() == 95

    def test_mad_zero_keeps_everything(self):
        values = np.concatenate([np.full(80, 10.0), np.full(20, 40.0)])
        assert inlier_mask(values, "mad", 3.5).all()

    def test_sigma_and_none(self):
        values = np.concatenate([np.full(999, 10.0), [1000.0]])
        assert inlier_mask(values, "sigma", 3.0).sum() == 999
        assert inlier_mask(values, "none", 0.0).all()

    def test_percentile_trimmed_and_mode(self):
        values = np.concatenate([np.full(40, 10.0), np.full(60, 40.0)])
        assert robust_distance(values, "p30") == pytest.approx(10.0)
        # mode returns the nearest peak, not the tallest one
        assert robust_distance(values, "mode") == pytest.approx(10.0)
        assert robust_distance(np.arange(1.0, 11.0), "trimmed") == pytest.approx(5.5)

    def test_max_range_and_mask(self):
        depth = object_on_background()
        cfg = DistanceConfig(
            box_shrink=0.0, outlier="none", stat="mean", max_range=20.0
        )
        assert box_depth_stats(
            depth, (0, 0, 100, 100), IDENTITY, cfg
        ).distance == pytest.approx(10.0)
        mask = np.zeros_like(depth, dtype=bool)
        mask[:, :20] = True  # background only
        cfg = DistanceConfig(box_shrink=0.0, outlier="none", stat="mean")
        stats = box_depth_stats(depth, (0, 0, 100, 100), IDENTITY, cfg, mask=mask)
        assert stats.distance == pytest.approx(40.0)
        assert stats.n_valid == 2000

    def test_invalid_pixels_are_ignored(self):
        depth = object_on_background()
        depth[:, :20] = np.nan
        depth[:, 80:] = 0.0
        cfg = DistanceConfig(box_shrink=0.0, outlier="none", stat="mean")
        assert box_depth_stats(
            depth, (0, 0, 100, 100), IDENTITY, cfg
        ).distance == pytest.approx(10.0)


class TestEuclidean:
    def test_on_axis_equals_z_off_axis_is_larger(self):
        k = CameraIntrinsics(fx=50.0, fy=50.0, cx=50.0, cy=50.0)
        assert ray_factors((49, 49, 51, 51), k) == pytest.approx(
            np.full((2, 2), math.sqrt(1 + 2 * 0.01**2))
        )
        depth = np.full((100, 100), 10.0, dtype=np.float32)
        cfg = DistanceConfig(box_shrink=0.0, distance_type="euclidean")
        corner = box_depth_stats(depth, (0, 0, 20, 20), IDENTITY, cfg, intrinsics=k)
        assert corner.distance > 10.0

    def test_euclidean_needs_intrinsics(self):
        cfg = DistanceConfig(distance_type="euclidean")
        with pytest.raises(ValueError):
            box_depth_stats(object_on_background(), (0, 0, 10, 10), IDENTITY, cfg)


class TestEdgeCases:
    def test_empty_box_gives_no_distance_and_serialisable_row(self):
        stats = box_depth_stats(
            object_on_background(), (150, 150, 200, 200), IDENTITY, DistanceConfig()
        )
        assert stats.distance is None
        row = stats.to_row()
        assert tuple(row) == STATS_ROW_FIELDS
        assert math.isnan(row["median"]) and row["distance_m"] is None

    def test_too_few_inliers(self):
        cfg = DistanceConfig(box_shrink=0.0, min_valid_px=50)
        assert (
            box_depth_stats(
                object_on_background(), (0, 0, 5, 5), IDENTITY, cfg
            ).distance
            is None
        )

    def test_mask_shape_mismatch(self):
        with pytest.raises(ValueError):
            box_depth_stats(
                object_on_background(),
                (0, 0, 10, 10),
                IDENTITY,
                DistanceConfig(),
                mask=np.ones((10, 10), dtype=bool),
            )

    @pytest.mark.parametrize("stat", ["p0", "p100", "foo", "p"])
    def test_parse_stat_rejects(self, stat):
        with pytest.raises(ValueError):
            parse_stat(stat)

    def test_config_validation(self):
        with pytest.raises(ValueError):
            DistanceConfig(box_shrink=0.5)
        assert DistanceConfig(outlier="mad").k == 3.5
        assert DistanceConfig(outlier="mad", outlier_k=2.0).k == 2.0
