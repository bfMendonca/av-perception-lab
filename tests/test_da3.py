import numpy as np
import pytest

from det_depth_distance.adapters.depth.da3 import (
    classify_depth_model,
    da3_map_geometry,
    intrinsics_from_focal,
    peek_model_name,
    resolve_depth_model,
    to_metres,
)
from det_depth_distance.core.types import CameraIntrinsics


@pytest.mark.parametrize(
    ("res", "method", "expected"),
    [
        (504, "upper_bound_resize", (504, 280, 0.0, 0.0)),
        (504, "upper_bound_crop", (504, 280, 0.0, -2.0)),
        (1008, "upper_bound_resize", (1008, 574, 0.0, 0.0)),
        (504, "lower_bound_resize", (896, 504, 0.0, 0.0)),
        (504, "lower_bound_crop", (896, 504, -0.0, 0.0)),
    ],
)
def test_map_geometry_1080p(res, method, expected):
    g = da3_map_geometry(1920, 1080, res, method)
    assert (g.map_w, g.map_h, g.ox, g.oy) == expected


def test_crop_geometry_scale_uses_resized_size():
    g = da3_map_geometry(1920, 1080, 504, "upper_bound_crop")
    assert g.sx == pytest.approx(504 / 1920)
    assert g.sy == pytest.approx(284 / 1080)


def test_unknown_method():
    with pytest.raises(ValueError):
        da3_map_geometry(1920, 1080, 504, "bilinear")


def test_intrinsics_from_focal():
    g = da3_map_geometry(1920, 1080, 504, "upper_bound_resize")
    k = intrinsics_from_focal(1400.0, g)
    assert k.fx == pytest.approx(1400 * 504 / 1920)
    assert k.fy == pytest.approx(1400 * 280 / 1080)
    assert (k.cx, k.cy) == pytest.approx((252.0, 140.0))


def test_to_metres():
    raw = np.full((2, 2), 3.0, dtype=np.float32)
    k = CameraIntrinsics(fx=300.0, fy=600.0, cx=0.0, cy=0.0)
    # 3 * mean(300, 600) / 300
    assert to_metres(raw, "metric_focal", k) == pytest.approx(np.full((2, 2), 4.5))
    assert to_metres(raw, "metric", None) is not None
    with pytest.raises(ValueError):
        to_metres(raw, "metric_focal", None)
    with pytest.raises(ValueError):
        to_metres(raw, "relative", k)


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        ("da3metric-large", "metric_focal"),
        ("depth-anything/DA3METRIC-LARGE", "metric_focal"),
        ("da3nested-giant-large", "metric"),
        ("depth-anything/DA3NESTED-GIANT-LARGE-1.1", "metric"),
        ("da3-large", "relative"),
        ("da3mono-large", "relative"),
    ],
)
def test_classify_depth_model(name, kind):
    assert classify_depth_model(name) == kind


def test_resolve_depth_model():
    assert resolve_depth_model("DA3METRIC-large") == "depth-anything/DA3METRIC-LARGE"
    assert resolve_depth_model("/models/custom") == "/models/custom"


def test_peek_model_name_reads_local_config(tmp_path):
    # Same layout as the HF snapshots: {"model_name": ..., "config": {...}}
    (tmp_path / "config.json").write_text(
        '{"model_name": "da3metric-large", "config": {}}'
    )
    assert classify_depth_model(peek_model_name(str(tmp_path))) == "metric_focal"
    assert peek_model_name("depth-anything/DA3-LARGE") == "depth-anything/DA3-LARGE"
    assert peek_model_name(str(tmp_path / "missing")) == str(tmp_path / "missing")
