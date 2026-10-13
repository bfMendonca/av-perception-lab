import math

import cv2
import numpy as np
import pytest

from det_depth_distance.core.types import Instances
from det_depth_distance.distance import (
    MASK_ROW_FIELDS,
    STATS_ROW_FIELDS,
    CameraIntrinsics,
    DistanceConfig,
    MapGeometry,
    box_depth_stats,
    box_to_map_pixels,
    erode_mask,
    inlier_mask,
    instance_depth_stats,
    map_eligibility,
    mask_depth_stats,
    parse_stat,
    ray_factors,
    resample_mask,
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


FULL_MAP = (0, 0, 504, 280)


def frame_mask(*slices: tuple[slice, slice], shape=(1080, 1920)) -> np.ndarray:
    """Boolean frame-resolution mask, True on the union of (rows, cols) slices."""
    mask = np.zeros(shape, dtype=bool)
    for rows, cols in slices:
        mask[rows, cols] = True
    return mask


def window_indicator(geom: MapGeometry, box) -> np.ndarray:
    """(map_h, map_w) bool: the pixels of box_to_map_pixels(box, geom, 0.0)."""
    c0, r0, c1, r1 = box_to_map_pixels(box, geom, 0.0)
    out = np.zeros((geom.map_h, geom.map_w), dtype=bool)
    out[r0:r1, c0:c1] = True
    return out


def _has_tie(edges: np.ndarray, scale: float, offset: float) -> bool:
    mapped = edges * scale + offset
    return bool(np.any(np.abs(mapped - np.floor(mapped) - 0.5) < 1e-6))


class TestMaskResample:
    @pytest.mark.parametrize("geom", [RESIZE, CROP], ids=["resize", "crop"])
    def test_rectangle_reproduces_box_window(self, geom):
        rng = np.random.default_rng(0)
        checked = 0
        while checked < 50:
            x0, x1 = np.sort(rng.integers(0, 1921, 2))
            y0, y1 = np.sort(rng.integers(0, 1081, 2))
            xs, ys = np.array([x0, x1]), np.array([y0, y1])
            if _has_tie(xs, geom.sx, geom.ox) or _has_tie(ys, geom.sy, geom.oy):
                continue  # ties follow the half-open rule (next test)
            mask = frame_mask((slice(y0, y1), slice(x0, x1)))
            box = (float(x0), float(y0), float(x1), float(y1))
            got = resample_mask(mask, geom, FULL_MAP)
            np.testing.assert_array_equal(got, window_indicator(geom, box))
            checked += 1

    def test_tie_is_half_open(self):
        # x1 = 600 -> 600 * 0.2625 = 157.5, the centre of map column 157: the box
        # window includes the column, the resampled mask does not.
        box = (400.0, 400.0, 600.0, 600.0)
        assert box_to_map_pixels(box, RESIZE, 0.0)[2] == 158
        got = resample_mask(
            frame_mask((slice(400, 600), slice(400, 600))), RESIZE, FULL_MAP
        )
        assert np.flatnonzero(got.any(axis=0))[-1] == 156

    def test_anisotropic_square(self):
        # 100 * 0.2625 = 26.25 columns, 100 * 0.2593 = 25.9 rows -> 26 x 26 centres
        square = frame_mask((slice(400, 500), slice(800, 900)))
        got = resample_mask(square, RESIZE, FULL_MAP)
        assert got.sum(axis=0).max() == 26 and got.sum(axis=1).max() == 26
        assert got.sum() == 26 * 26

    def test_crop_shifts_rows(self):
        uncropped = MapGeometry(1920, 1080, 504, 284, CROP.sx, CROP.sy)
        mask = frame_mask((slice(300, 700), slice(500, 900)))
        cropped = resample_mask(mask, CROP, FULL_MAP)
        full = resample_mask(mask, uncropped, (0, 0, 504, 284))
        np.testing.assert_array_equal(cropped, full[2:282])

    def test_empty_window(self):
        mask = frame_mask((slice(0, 10), slice(0, 10)))
        assert resample_mask(mask, RESIZE, (10, 5, 10, 9)).shape == (4, 0)
        assert resample_mask(mask, RESIZE, (5, 5, 9, 5)).shape == (0, 4)

    def test_border_is_clipped(self):
        # The last map column centre maps past the frame (x = 105): it reads column 99.
        geom = MapGeometry(100, 100, 10, 10, 0.1, 0.1, -1.0, 0.0)
        mask = frame_mask((slice(0, 100), slice(99, 100)), shape=(100, 100))
        got = resample_mask(mask, geom, (0, 0, 10, 10))
        assert got[:, -1].all() and not got[:, :-1].any()

    def test_any_dtype_is_preserved(self):
        labels = np.arange(100 * 100, dtype=np.int32).reshape(100, 100)
        got = resample_mask(labels, IDENTITY, (10, 20, 30, 25))
        assert got.dtype == np.int32
        np.testing.assert_array_equal(got, labels[20:25, 10:30])


class TestErode:
    @pytest.mark.parametrize("k", [1, 2, 3])
    def test_matches_cv2_erode_with_zero_border(self, k):
        mask = np.random.default_rng(k).random((40, 60)) < 0.7
        expected = cv2.erode(
            mask.astype(np.uint8),
            np.ones((3, 3), np.uint8),
            iterations=k,
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        np.testing.assert_array_equal(erode_mask(mask, k), expected.astype(bool))

    def test_zero_iterations_is_a_copy(self):
        mask = np.random.default_rng(0).random((10, 10)) < 0.5
        out = erode_mask(mask, 0)
        np.testing.assert_array_equal(out, mask)
        assert out is not mask

    def test_window_edge_erodes(self):
        out = erode_mask(np.ones((5, 5), dtype=bool), 1)
        assert out.sum() == 9 and out[1:4, 1:4].all()


def diagonal_pole(width_px: float = 24.0) -> np.ndarray:
    """Frame mask of a pole from (800, 200) to (1000, 900), `width_px` thick."""
    ys, xs = np.mgrid[0:1080, 0:1920] + 0.5
    (ax, ay), (bx, by) = (800.0, 200.0), (1000.0, 900.0)
    dist = np.abs((by - ay) * xs - (bx - ax) * ys + bx * ay - by * ax) / math.hypot(
        bx - ax, by - ay
    )
    return (dist <= width_px / 2) & (ys >= ay) & (ys <= by)


class TestMaskStats:
    def test_mask_is_right_where_the_box_is_wrong(self):
        # A thin diagonal pole at 8 m fills ~12 % of its box over a 30 m background.
        pole = diagonal_pole()
        box = (800.0, 200.0, 1000.0, 900.0)
        depth = np.where(resample_mask(pole, RESIZE, FULL_MAP), 8.0, 30.0)
        depth = depth.astype(np.float32)
        cfg = DistanceConfig()
        by_box = box_depth_stats(depth, box, RESIZE, cfg)
        by_mask = mask_depth_stats(depth, box, pole, RESIZE, cfg)
        assert by_box.distance == pytest.approx(30.0)
        assert by_mask.distance == pytest.approx(8.0)
        assert by_mask.used_mask and by_mask.mask_erode == 1

    def test_l_shaped_object(self):
        # L of 1900 px (19 % of the box) at 5 m over a 40 m background.
        depth = np.full((100, 100), 40.0, dtype=np.float32)
        shape = frame_mask(
            (slice(0, 100), slice(0, 10)),
            (slice(90, 100), slice(0, 100)),
            shape=(100, 100),
        )
        depth[shape] = 5.0
        cfg = DistanceConfig(box_shrink=0.0)
        box = (0, 0, 100, 100)
        assert box_depth_stats(depth, box, IDENTITY, cfg).distance == pytest.approx(
            40.0
        )
        stats = mask_depth_stats(depth, box, shape, IDENTITY, cfg)
        assert stats.distance == pytest.approx(5.0)

    def test_mask_pixels_outside_the_box_are_ignored(self):
        depth = np.full((100, 100), 20.0, dtype=np.float32)
        depth[:, 60:] = 3.0
        mask = frame_mask((slice(0, 100), slice(0, 100)), shape=(100, 100))
        cfg = DistanceConfig(mask_erode=0)
        stats = mask_depth_stats(depth, (0, 0, 50, 100), mask, IDENTITY, cfg)
        assert stats.distance == pytest.approx(20.0)
        assert stats.map_box == (0, 0, 50, 100) and stats.n_box_px == 5000
        assert stats.mask_px == 5000

    def test_eligibility_mask_is_combined(self):
        depth = np.full((100, 100), 10.0, dtype=np.float32)
        instance = frame_mask((slice(0, 100), slice(20, 40)), shape=(100, 100))
        not_sky = np.ones((100, 100), dtype=bool)
        not_sky[:50] = False  # top half of the object is sky
        cfg = DistanceConfig(mask_erode=0)
        box = (0, 0, 100, 100)
        stats = mask_depth_stats(depth, box, instance, IDENTITY, cfg)
        assert stats.n_valid == 2000
        stats = mask_depth_stats(depth, box, instance, IDENTITY, cfg, mask=not_sky)
        assert stats.n_valid == 1000 and stats.mask_px == 2000

    def test_euclidean_uses_the_window_ray_factors(self):
        depth = np.full((100, 100), 10.0, dtype=np.float32)
        k = CameraIntrinsics(fx=50.0, fy=50.0, cx=50.0, cy=50.0)
        instance = frame_mask((slice(5, 15), slice(5, 15)), shape=(100, 100))
        cfg = DistanceConfig(distance_type="euclidean", outlier="none", mask_erode=0)
        box = (0, 0, 40, 40)
        stats = mask_depth_stats(depth, box, instance, IDENTITY, cfg, intrinsics=k)
        expected = np.median(10.0 * ray_factors((0, 0, 40, 40), k)[5:15, 5:15])
        assert stats.distance == pytest.approx(expected)
        with pytest.raises(ValueError):
            mask_depth_stats(depth, box, instance, IDENTITY, cfg)

    def test_empty_mask_gives_none(self):
        depth = object_on_background()
        empty = np.zeros((100, 100), dtype=bool)
        assert (
            mask_depth_stats(depth, (0, 0, 50, 50), empty, IDENTITY, DistanceConfig())
            is None
        )
        outside = frame_mask((slice(60, 70), slice(60, 70)), shape=(100, 100))
        assert (
            mask_depth_stats(depth, (0, 0, 50, 50), outside, IDENTITY, DistanceConfig())
            is None
        )

    def test_instance_mask_must_be_frame_sized(self):
        with pytest.raises(ValueError, match="instance mask shape"):
            mask_depth_stats(
                object_on_background(),
                (0, 0, 50, 50),
                np.ones((50, 50), dtype=bool),
                IDENTITY,
                DistanceConfig(),
            )

    @pytest.mark.parametrize(
        ("width", "erode", "applied", "mask_px"),
        [
            (2, 1, 0, 200),  # a 2 px pole erodes to nothing: no erosion
            (10, 2, 2, 6 * 96),  # wide enough: full erosion (borders erode too)
            (5, 3, 2, 1 * 96),  # 3 would empty it: back off to 2
        ],
    )
    def test_erosion_backs_off_for_thin_objects(self, width, erode, applied, mask_px):
        depth = np.full((100, 100), 10.0, dtype=np.float32)
        pole = frame_mask((slice(0, 100), slice(50, 50 + width)), shape=(100, 100))
        cfg = DistanceConfig(mask_erode=erode, min_valid_px=20)
        stats = mask_depth_stats(depth, (40, 0, 60, 100), pole, IDENTITY, cfg)
        assert (stats.mask_erode, stats.mask_px) == (applied, mask_px)
        assert stats.n_box_px == 20 * 100

    def test_mask_row_and_config(self):
        depth = object_on_background()
        box_stats = box_depth_stats(depth, (0, 0, 50, 50), IDENTITY, DistanceConfig())
        assert not box_stats.used_mask
        assert box_stats.mask_row() == {
            "used_mask": 0,
            "mask_px": None,
            "mask_erode": None,
        }
        assert tuple(box_stats.mask_row()) == MASK_ROW_FIELDS
        assert tuple(box_stats.to_row()) == STATS_ROW_FIELDS  # unchanged
        mask = frame_mask((slice(0, 50), slice(20, 50)), shape=(100, 100))
        stats = mask_depth_stats(
            depth, (0, 0, 50, 50), mask, IDENTITY, DistanceConfig()
        )
        assert stats.mask_row() == {"used_mask": 1, "mask_px": 48 * 28, "mask_erode": 1}
        assert DistanceConfig().mask_erode == 1
        with pytest.raises(ValueError, match="mask_erode"):
            DistanceConfig(mask_erode=-1)


class TestRegionPolicy:
    BOX = (0, 0, 100, 100)

    @pytest.mark.parametrize("region", ["mask", "box", "both"])
    def test_without_mask_equals_box_stats(self, region):
        depth = object_on_background()
        cfg = DistanceConfig()
        expected = box_depth_stats(depth, self.BOX, IDENTITY, cfg)
        for instance_mask in (None, np.zeros((100, 100), dtype=bool)):
            primary, alt = instance_depth_stats(
                depth,
                self.BOX,
                IDENTITY,
                cfg,
                region=region,
                instance_mask=instance_mask,
            )
            assert primary == expected and alt is None
            assert not primary.used_mask

    def test_policies_with_a_mask(self):
        depth = object_on_background()
        cfg = DistanceConfig()
        background = frame_mask((slice(0, 100), slice(85, 100)), shape=(100, 100))
        by_box = box_depth_stats(depth, self.BOX, IDENTITY, cfg)
        args = (depth, self.BOX, IDENTITY, cfg)
        primary, alt = instance_depth_stats(
            *args, region="box", instance_mask=background
        )
        assert primary == by_box and alt is None
        primary, alt = instance_depth_stats(
            *args, region="mask", instance_mask=background
        )
        assert primary.used_mask and primary.distance == pytest.approx(40.0)
        assert alt is None
        primary, alt = instance_depth_stats(
            *args, region="both", instance_mask=background
        )
        assert primary.distance == pytest.approx(40.0) and alt == by_box

    def test_mask_missing_every_map_pixel_centre_falls_back_to_the_box(self):
        # Frame columns/rows 2..3 lie between the centres of map pixels 0 and 1
        # (x = 1.9 and 5.7, y = 1.9 and 5.8) at 504x280: has_mask() is True (the
        # pixels are inside the box) but the resampled mask is empty.
        masks = frame_mask((slice(2, 4), slice(2, 4)))[None]
        inst = Instances(
            np.array([[0, 0, 20, 20]], np.float32),
            np.array([0.9], np.float32),
            np.array([0], np.int64),
            ("pole",),
            ("pole",),
            ("det",),
            masks,
        )
        assert inst.has_mask().tolist() == [True]
        depth = np.full((280, 504), 10.0, dtype=np.float32)
        primary, alt = instance_depth_stats(
            depth,
            inst.xyxy[0],
            RESIZE,
            DistanceConfig(),
            region="both",
            instance_mask=masks[0],
        )
        assert not primary.used_mask and alt is None


def _left_half_road() -> np.ndarray:
    """1920x1080 labels: 0 (road) on the left half, 1 elsewhere."""
    labels = np.ones((1080, 1920), np.int32)
    labels[:, :960] = 0
    return labels


def _sky_top_rows(n: int = 20) -> np.ndarray:
    sky = np.zeros((280, 504), dtype=bool)
    sky[:n] = True
    return sky


def test_map_eligibility_without_semantic_is_the_sky_rule():
    sky = _sky_top_rows()
    out = map_eligibility(RESIZE, sky=sky)
    assert out.dtype == bool and np.array_equal(out, ~sky)
    assert map_eligibility(RESIZE) is None
    # Layers that exclude nothing are ignored entirely.
    empty = [(_left_half_road(), ()), (_left_half_road(), np.array([], np.int32))]
    assert np.array_equal(map_eligibility(RESIZE, sky=sky, exclusions=empty), ~sky)
    assert map_eligibility(RESIZE, exclusions=empty) is None


def test_map_eligibility_excludes_classes_on_the_map_grid():
    labels = _left_half_road()
    out = map_eligibility(RESIZE, exclusions=[(labels, [0])])
    # Map column c samples frame x = (c + 0.5) * 1920 / 504: c < 252 is the left half.
    assert not out[:, :252].any() and out[:, 252:].all()
    # Rows: the crop offset shifts the sampling; same rule as resample_mask.
    top = np.zeros((1080, 1920), np.int32)
    top[:300] = 3
    crop = map_eligibility(CROP, exclusions=[(top, [3])])
    window = (0, 0, CROP.map_w, CROP.map_h)
    assert np.array_equal(crop, ~resample_mask(top == 3, CROP, window))
    resize = map_eligibility(RESIZE, exclusions=[(top, [3])])
    assert (~crop).sum(axis=0)[0] < (~resize).sum(axis=0)[0]  # -2 rows offset
    # With the sky mask: AND of both criteria.
    sky = _sky_top_rows()
    both = map_eligibility(RESIZE, sky=sky, exclusions=[(labels, [0])])
    assert np.array_equal(both, out & ~sky)


def test_map_eligibility_union_of_layers():
    left = _left_half_road()  # road id 0 on the left half
    lanes = np.zeros((1080, 1920), np.int32)
    lanes[:360] = 1  # "lane" id 1 on the top third (id 1 means "other" in `left`)
    a = map_eligibility(RESIZE, exclusions=[(left, [0])])
    b = map_eligibility(RESIZE, exclusions=[(lanes, [1])])
    union = map_eligibility(RESIZE, exclusions=[(left, [0]), (lanes, [1])])
    assert np.array_equal(union, a & b)  # overlapping pixels counted once
    swapped = map_eligibility(RESIZE, exclusions=[(lanes, [1]), (left, [0])])
    assert np.array_equal(union, swapped)
    # Ids belong to their own layer: excluding id 1 of `lanes` keeps id 1 of `left`.
    assert np.array_equal(
        map_eligibility(RESIZE, exclusions=[(left, []), (lanes, [1])]), b
    )


def test_map_eligibility_commutes_with_resampling():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 5, size=(1080, 1920)).astype(np.int32)
    window = (0, 0, CROP.map_w, CROP.map_h)
    after = map_eligibility(CROP, exclusions=[(labels, [1, 3])])
    before = ~resample_mask(np.isin(labels, [1, 3]), CROP, window)
    assert np.array_equal(after, before)


def test_map_eligibility_checks_shapes():
    with pytest.raises(ValueError, match="labels shape"):
        map_eligibility(RESIZE, exclusions=[(np.zeros((10, 10), np.int32), [0])])
    with pytest.raises(ValueError, match="sky shape"):
        map_eligibility(RESIZE, sky=np.zeros((10, 10), bool))
