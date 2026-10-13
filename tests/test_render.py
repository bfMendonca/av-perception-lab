import cv2
import numpy as np
import pytest

from det_depth_distance.render import (
    CLASS_COLORS,
    PALETTE,
    BoxAnnotation,
    Renderer,
    color_for,
    colorize_metric,
    format_label,
    metric_legend,
)


def test_format_label():
    assert format_label("truck", 0.94, 12.34) == "truck 94% · 12.3 m"
    assert format_label("truck", 0.94, 12.34, 0.42) == "truck 94% · 12.3 m ± 0.4"
    assert format_label("car", 0.5, None, 1.0) == "car 50% · n/a"


def test_colorize_metric_is_rgb_near_warm():
    depth = np.array([[0.5, 29.0]], dtype=np.float32)
    rgb = colorize_metric(depth, 30.0, cv2.COLORMAP_TURBO)
    near, far = rgb[0, 0].astype(int), rgb[0, 1].astype(int)
    assert near[0] > near[2]  # red dominates near
    assert far[2] > far[0]  # blue dominates far


def test_legend_even_width():
    assert metric_legend(1080, 30.0, cv2.COLORMAP_TURBO).shape[1] % 2 == 0
    assert metric_legend(721, 15.0, cv2.COLORMAP_TURBO).shape[1] % 2 == 0
    # max_depth beyond the tick table must not raise StopIteration
    assert metric_legend(540, 1000.0, cv2.COLORMAP_TURBO).shape[0] == 540


BOX = BoxAnnotation(
    (10.0, 20.0, 60.0, 70.0), "car", 0.9, "car 90% · 8.0 m", (15.0, 25.0, 55.0, 65.0)
)


def test_overlay_render():
    r = Renderer(160, 120, title="test")
    out = r.render(np.zeros((120, 160, 3), np.uint8), [BOX], ["line"])
    assert out.shape == (120, 160, 3) and r.output_size() == (160, 120)
    assert out.any()


def test_side_by_side_render():
    r = Renderer(160, 120, title="test", layout="side-by-side", show_hud=False)
    width, height = r.output_size()
    assert width % 2 == 0 and width > 2 * 160
    out = r.render(
        np.zeros((120, 160, 3), np.uint8), [BOX], [], np.full((28, 42), 5.0, np.float32)
    )
    assert out.shape == (height, width, 3)
    with pytest.raises(ValueError):
        r.render(np.zeros((120, 160, 3), np.uint8), [BOX], [])


def test_bad_cmap():
    with pytest.raises(ValueError):
        Renderer(160, 120, title="x", cmap="nope")


def test_color_for():
    assert color_for("car") == CLASS_COLORS["car"] == color_for("Car")
    assert color_for("pickup truck") == CLASS_COLORS["pickup truck"]
    # Unknown keys (e.g. detector names) get a stable palette colour.
    assert color_for("rfdetr-large") in PALETTE
    assert color_for("rfdetr-large") == color_for("rfdetr-large")


def test_color_key_changes_colour_and_hud_count():
    by_class = BoxAnnotation((10.0, 20.0, 60.0, 70.0), "car", 0.9, "car")
    by_source = BoxAnnotation(
        (10.0, 20.0, 60.0, 70.0), "car", 0.9, "car", color_key="yolo26l-objv1-150"
    )
    assert by_class.key == "car" and by_source.key == "yolo26l-objv1-150"
    r = Renderer(160, 120, title="test", show_hud=False)
    frame = np.zeros((120, 160, 3), np.uint8)
    assert not np.array_equal(
        r.render(frame, [by_class], []), r.render(frame, [by_source], [])
    )
    r = Renderer(160, 120, title="test")
    counts = []
    r._hud = lambda d, lines, c: counts.append(c)
    r.render(frame, [by_source, by_source], ["line"])
    assert counts == [{"yolo26l-objv1-150": 2}]
