import cv2
import numpy as np
import pytest

from det_depth_distance.calibration import (
    CalibrationOptions,
    TargetSpec,
    View,
    calibrate_views,
    coverage,
    detect_views,
    focal_px_from_35mm,
    make_target,
    select_views,
)
from det_depth_distance.camera import load_calibration, save_calibration
from det_depth_distance.core.options import ConfigError


def point_views(cam, n, *, noise=0.0, seed=0):
    """Exact projections of the board (plus Gaussian noise), one View per pose."""
    rng = np.random.default_rng(seed)
    views = []
    for i, (rvec, tvec) in enumerate(cam.poses(n, seed=seed)):
        pts = cam.project(rvec, tvec) + rng.normal(0.0, noise, (cam.cols * cam.rows, 2))
        views.append(View(i, cam.object_points(), pts.astype(np.float32)))
    return views


def k_errors(report, cam):
    k = report.calibration.camera_matrix()
    rel_f = abs(k[0, 0] - cam.k[0, 0]) / cam.k[0, 0]
    rel_fy = abs(k[1, 1] - cam.k[1, 1]) / cam.k[1, 1]
    dc = np.abs(k[:2, 2] - cam.k[:2, 2]).max()
    return max(rel_f, rel_fy), dc


def test_noise_free_points_recover_the_camera(synthetic_camera):
    cam = synthetic_camera
    report = calibrate_views(point_views(cam, 20), cam.size)
    rel_f, dc = k_errors(report, cam)
    assert rel_f < 1e-6 and dc < 1e-3
    assert report.calibration.dist == pytest.approx((0.1, -0.2, 0.0, 0.0), abs=1e-6)
    assert report.rejected_views == () and len(report.used_views) == 20
    assert set(report.calibration.std) == {
        "fx",
        "fy",
        "cx",
        "cy",
        "k1",
        "k2",
        "p1",
        "p2",
    }


def test_noisy_points(synthetic_camera):
    cam = synthetic_camera
    report = calibrate_views(point_views(cam, 20, noise=0.1, seed=1), cam.size)
    rel_f, dc = k_errors(report, cam)
    assert rel_f <= 3e-3 and dc <= 3.0
    assert 0.10 <= report.calibration.rms_px <= 0.20
    meta = report.calibration.meta
    assert meta["n_views"] == 20 and meta["coverage"] == report.coverage
    assert meta["per_view_rms"]["max"] == pytest.approx(max(report.per_view_rms))


def test_rendered_chessboard_images(synthetic_camera, render_board, tmp_path):
    cam = synthetic_camera
    images = [render_board(r, t) for r, t in cam.poses(12, seed=2)]
    blank = np.full_like(images[0], 255)
    views, examined = detect_views(
        enumerate([*images, blank]), make_target(TargetSpec())
    )
    assert (len(views), examined) == (12, 13)  # the blank image is counted, not kept
    report = calibrate_views(views, cam.size, CalibrationOptions(min_views=12))
    rel_f, _ = k_errors(report, cam)
    assert rel_f <= 5e-3
    # The fitted model corrects the image like the true one, to half a pixel.
    w, h = cam.size
    grid = np.stack(
        np.meshgrid(np.linspace(0, w - 1, 17), np.linspace(0, h - 1, 9)), -1
    )
    grid = grid.reshape(-1, 1, 2)
    cal = report.calibration
    fitted = cv2.undistortPoints(grid, cal.camera_matrix(), cal.dist_array(), P=cam.k)
    true = cv2.undistortPoints(grid, cam.k, cam.dist, P=cam.k)
    assert np.abs(fitted - true).max() <= 0.5
    # Written and read back like any calibration file.
    save_calibration(cal, tmp_path / "cal.yaml")
    assert load_calibration(tmp_path / "cal.yaml").intrinsics == pytest.approx(
        cal.intrinsics
    )


def test_outlier_view_is_rejected(synthetic_camera):
    cam = synthetic_camera
    views = point_views(cam, 16, noise=0.1, seed=4)
    bad = views[5]
    # Every other corner off by 5 px (a wrong detection).
    shift = np.where(np.arange(len(bad.image_points))[:, None] % 2, [5.0, -5.0], 0.0)
    corrupt = (bad.image_points + shift).astype(np.float32)
    views[5] = View(bad.index, bad.object_points, corrupt)
    report = calibrate_views(views, cam.size)
    assert report.rejected_views == (5,) and 5 not in report.used_views
    rel_f, dc = k_errors(report, cam)
    assert rel_f <= 3e-3 and dc <= 3.0


def test_too_few_views(synthetic_camera):
    cam = synthetic_camera
    with pytest.raises(ConfigError, match="5 views with the target, at least 12"):
        calibrate_views(point_views(cam, 5), cam.size)
    with pytest.raises(ConfigError, match="max_views must be >= min_views"):
        CalibrationOptions(min_views=12, max_views=10)


def test_coverage_by_hand():
    obj = np.zeros((2, 3), np.float32)
    # 100 x 100 image, 2 x 2 grid: corners in the top-left and bottom-right cells.
    view = View(0, obj, np.array([[10.0, 10.0], [80.0, 70.0]], np.float32))
    assert coverage([view], (100, 100), (2, 2)) == 0.5
    other = View(1, obj, np.array([[90.0, 5.0], [99.4, 99.4]], np.float32))
    assert coverage([view, other], (100, 100), (2, 2)) == 0.75


def test_select_views_spreads_over_time():
    views = [View(i, np.zeros((1, 3)), np.zeros((1, 2))) for i in range(10)]
    assert [v.index for v in select_views(views, 4)] == [0, 3, 6, 9]
    assert len(select_views(views, 20)) == 10


def test_charuco_detection_on_a_warped_board():
    spec = TargetSpec(type="charuco", cols=7, rows=5, square_mm=30.0, marker_mm=22.0)
    target = make_target(spec)
    board = target.image(px_per_mm=4.0, margin_mm=10.0)
    h, w = board.shape
    assert (w, h) == (7 * 30 * 4 + 80, 5 * 30 * 4 + 80)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[60, 40], [w + 20, 90], [w - 30, h + 60], [30, h - 10]])
    image = cv2.warpPerspective(
        board,
        cv2.getPerspectiveTransform(src, dst),
        (w + 120, h + 120),
        borderValue=255,
    )
    view = target.detect(np.dstack([image] * 3), 7)
    assert view is not None and view.index == 7 and len(view.object_points) >= 6
    assert np.all(view.object_points[:, 2] == 0)
    # Board points (m) and detected pixels are related by one homography.
    hom, _ = cv2.findHomography(view.object_points[:, :2], view.image_points)
    mapped = cv2.perspectiveTransform(view.object_points[None, :, :2], hom)[0]
    assert np.abs(mapped - view.image_points).max() < 1.0
    assert target.detect(np.full((200, 200), 255, np.uint8), 0) is None


def test_charuco_collinear_corners_are_not_a_view():
    # Only one row of inner corners visible (a board cut by the frame edge):
    # no homography exists and calibrateCamera would abort the whole fit.
    target = make_target(
        TargetSpec(type="charuco", cols=8, rows=5, square_mm=30.0, marker_mm=22.0)
    )
    image = target.image(px_per_mm=4.0, margin_mm=10.0)  # 120 px squares, 40 px margin
    strip = np.full_like(image, 255)
    strip[160:400] = image[160:400]  # square rows 1-2: one row of 7 inner corners
    _, ids, _, _ = target._detector.detectBoard(strip)
    assert ids is not None and len(ids) >= 6  # enough corners, but collinear
    assert target.detect(strip, 0) is None
    assert target.detect(image, 0) is not None


def test_chessboard_print_size():
    image = make_target(TargetSpec()).image(px_per_mm=2.0, margin_mm=5.0)
    assert image.shape == (7 * 25 * 2 + 20, 10 * 25 * 2 + 20)
    assert image[0, 0] == 255 and image[10, 10] == 0  # margin, then a black square


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"cols": 1}, "cols and rows must be >= 2"),
        ({"square_mm": 0}, "square_mm must be > 0"),
        ({"type": "charuco"}, "0 < marker_mm < square_mm"),
        ({"type": "charuco", "marker_mm": 30.0}, "0 < marker_mm < square_mm"),
        ({"type": "charuco", "marker_mm": 18.0, "dictionary": "DICT_X"}, "dictionary"),
        ({"marker_mm": 18.0}, "only used by charuco"),
    ],
)
def test_target_spec_errors(kwargs, match):
    with pytest.raises(ConfigError, match=match):
        TargetSpec(**kwargs)


def test_focal_from_35mm_equivalent():
    # task_0007 Fact 2: 24 mm-eq. at 1920x1080 is 1221.95 px (diagonal convention).
    assert focal_px_from_35mm(24, 1920, 1080) == pytest.approx(1221.95, abs=0.01)
