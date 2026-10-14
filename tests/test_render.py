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
    assert format_label("vehicle", 0.91, 12.34, n_sources=1) == "vehicle 91% · 12.3 m"
    assert (
        format_label("vehicle", 0.91, 12.34, n_sources=2) == "vehicle 91% ×2 · 12.3 m"
    )
    assert format_label("vehicle", 0.91, None, 0.4, 2) == "vehicle 91% ×2 · n/a"


def test_format_label_track_id_and_speed():
    assert format_label("truck", 0.94, 12.34, track_id=12) == "#12 truck 94% · 12.3 m"
    assert (
        format_label("truck", 0.94, 12.34, 0.42, 2, track_id=3, rel_speed=-1.23)
        == "#3 truck 94% ×2 · 12.3 m ± 0.4 · -1.2 m/s"
    )
    assert format_label("car", 0.5, 8.0, rel_speed=0.5) == "car 50% · 8.0 m · +0.5 m/s"
    assert format_label("car", 0.5, 8.0, rel_speed=float("nan")) == "car 50% · 8.0 m"
    assert format_label("car", 0.5, None, track_id=1) == "#1 car 50% · n/a"


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
    assert color_for("vehicle") == CLASS_COLORS["car"]  # canonical label of the maps
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


def _masked(mask_alpha: float, layout: str = "overlay"):
    mask = np.zeros((120, 160), dtype=bool)
    mask[30:60, 20:40] = True
    box = BoxAnnotation((10.0, 20.0, 60.0, 70.0), "car", 0.9, "car", mask=mask)
    plain = BoxAnnotation((10.0, 20.0, 60.0, 70.0), "car", 0.9, "car")
    r = Renderer(
        160, 120, title="t", layout=layout, show_hud=False, mask_alpha=mask_alpha
    )
    return r, mask, box, plain


def test_mask_tints_only_its_pixels():
    r, mask, box, plain = _masked(0.35)
    frame = np.full((120, 160, 3), 100, np.uint8)
    changed = (r.render(frame, [box], []) != r.render(frame, [plain], [])).any(axis=2)
    assert changed.any() and not (changed & ~mask).any()
    assert np.array_equal(frame, np.full((120, 160, 3), 100, np.uint8))  # not in place


def test_mask_alpha_zero_draws_no_mask():
    r, _, box, plain = _masked(0.0)
    frame = np.full((120, 160, 3), 100, np.uint8)
    assert np.array_equal(r.render(frame, [box], []), r.render(frame, [plain], []))


def test_mask_also_tints_the_depth_panel():
    r, mask, box, plain = _masked(0.35, "side-by-side")
    frame = np.full((120, 160, 3), 100, np.uint8)
    depth = np.full((28, 42), 5.0, np.float32)
    with_mask = r.render(frame, [box], [], depth)
    without = r.render(frame, [plain], [], depth)
    panel = (with_mask != without).any(axis=2)[:, 160:320]
    assert panel.any() and not (panel & ~mask).any()


def test_mask_alpha_range():
    with pytest.raises(ValueError, match="mask_alpha"):
        Renderer(160, 120, title="x", mask_alpha=1.5)


# task_0005: semantic layers (labels, class_names) under masks and boxes.
SCENE = ("background", "road", "car")  # "car" shares the box colour
LANES = ("background", "lane", "road")


def _layers():
    scene = np.zeros((120, 160), np.int32)
    scene[:, :80] = 1  # road on the left half
    lanes = np.zeros((120, 160), np.int32)
    lanes[:40] = 1  # lane on the top third
    lanes[100:, 100:] = 2  # road in the lanes map too
    return (scene, SCENE), (lanes, LANES)


def _plain_renderer(**kwargs):
    return Renderer(160, 120, title="t", show_hud=False, **kwargs)


FRAME = np.full((120, 160, 3), 100, np.uint8)


def test_semantic_layer_tints_only_non_background():
    r = _plain_renderer()
    scene, _ = _layers()
    out = r.render(FRAME, [], [], semantic=[scene])
    changed = (out != FRAME).any(axis=2)
    assert np.array_equal(changed, scene[0] != 0)
    assert np.array_equal(
        out[0, 0], np.rint(0.6 * 100 + 0.4 * np.array(color_for("road")))
    )
    assert np.array_equal(FRAME, np.full((120, 160, 3), 100, np.uint8))  # not in place


def test_semantic_off_or_empty_is_the_plain_render():
    plain = _plain_renderer().render(FRAME, [BOX], [])
    assert np.array_equal(
        _plain_renderer().render(FRAME, [BOX], [], semantic=()), plain
    )
    off = _plain_renderer(semantic_alpha=0.0)
    assert np.array_equal(off.render(FRAME, [BOX], [], semantic=list(_layers())), plain)
    with pytest.raises(ValueError, match="semantic_alpha"):
        _plain_renderer(semantic_alpha=1.5)


def test_semantic_layers_compose_in_order():
    r = _plain_renderer()
    scene, lanes = _layers()
    both = r.render(FRAME, [], [], semantic=[scene, lanes])
    swapped = r.render(FRAME, [], [], semantic=[lanes, scene])
    overlap = (scene[0] != 0) & (lanes[0] != 0)
    differs = (both != swapped).any(axis=2)
    assert differs.any() and not (differs & ~overlap).any()  # only overlaps differ
    only_scene = (scene[0] != 0) & ~overlap
    alone = r.render(FRAME, [], [], semantic=[scene])
    assert np.array_equal(both[only_scene], alone[only_scene])
    # The same class keeps one colour in both maps ("road" left and bottom right).
    road_only = r.render(FRAME, [], [], semantic=[lanes])
    assert np.array_equal(road_only[110, 150], alone[0, 0])


def test_semantic_layer_is_under_the_boxes_and_not_on_the_depth_panel():
    r = _plain_renderer(layout="side-by-side")
    depth = np.full((28, 42), 5.0, np.float32)
    scene, _ = _layers()
    with_layer = r.render(FRAME, [BOX], [], depth, semantic=[scene])
    without = r.render(FRAME, [BOX], [], depth)
    assert np.array_equal(with_layer[:, 160:], without[:, 160:])
    # Box outline pixels are drawn over the layer: identical where the box edge is.
    assert np.array_equal(with_layer[21, 15:55], without[21, 15:55])
