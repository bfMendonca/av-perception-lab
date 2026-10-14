import math

import numpy as np
import pytest

from det_depth_distance.core.options import ConfigError
from det_depth_distance.core.types import CameraIntrinsics
from det_depth_distance.evaluation.matching import (
    match_references,
    reference_to_run_quantity,
)
from det_depth_distance.evaluation.references import (
    Instrument,
    Reference,
    ReferenceSet,
    Selector,
)
from det_depth_distance.evaluation.run_data import load_run

TWO_DETECTORS = {
    "components": {
        "detectors": [{"name": "rfdetr"}, {"name": "yolo"}],
        "depth": {"kind": "metric_focal"},
        "aggregator": {"type": "concat"},
    }
}
CAR = (900.0, 500.0, 1020.0, 580.0)  # centred on the principal point (960, 540)


def row(frame, box, source="rfdetr", label="car", distance=10.0, **extra):
    x0, y0, x1, y1 = box
    return {
        "frame": frame,
        "t": frame / 30,
        "class_name": label,
        "x0": x0,
        "y0": y0,
        "x1": x1,
        "y1": y1,
        "distance_m": distance,
        "source": source,
        "raw_label": label,
        **extra,
    }


def refs(*references, video="IMG_4100.MOV"):
    return ReferenceSet(
        video,
        tuple(references),
        {"laser": Instrument(uncertainty_m=0.01, uncertainty_rel=0.01)},
    )


def ref(select, frames=(2, 4), distance_m=10.0, **fields):
    return Reference(
        "r", select, frames=frames, distance_m=distance_m, instrument="laser", **fields
    )


@pytest.fixture
def run(fake_run_dir):
    rows = []
    for f in range(10):
        rows += [
            row(f, CAR, track_id=1),
            row(f, (100, 100, 400, 300), label="truck", track_id=2),
            row(f, (905, 505, 1025, 585), source="yolo"),
        ]
    rows.append(row(3, (950, 530, 970, 550), source="yolo", label="car"))  # small box
    return load_run(fake_run_dir(rows, TWO_DETECTORS))


def by_source(matches):
    out = {}
    for m in matches:
        out.setdefault(m.source, []).append(m)
    return out


def test_box_selector_takes_the_best_iou_per_detector(run):
    matches = match_references(run, refs(ref(Selector(box=CAR))))
    groups = by_source(matches)
    assert [m.frame for m in groups["rfdetr"]] == [2, 3, 4]
    assert all(m.iou == pytest.approx(1.0) for m in groups["rfdetr"])
    assert all(0.8 < m.iou < 1.0 for m in groups["yolo"])
    assert run.columns["source"][groups["yolo"][0].row] == "yolo"
    # Boxes near the principal point, z run: range ~= z.
    assert all(m.reference_m == pytest.approx(10.0, rel=1e-4) for m in matches)
    assert matches[0].uncertainty_m == pytest.approx(math.hypot(0.01, 0.1))
    strict = match_references(
        run, refs(ref(Selector(box=(880, 480, 1000, 560)))), iou_min=0.9
    )
    assert all(m.row is None and m.iou == 0.0 for m in strict)  # not detected


def test_point_label_and_track_selectors(run):
    point = match_references(run, refs(ref(Selector(point=(960, 540)))))
    yolo = by_source(point)["yolo"]
    # Frame 3 has a smaller yolo box containing the point: it wins.
    assert (
        run.columns["x0"][yolo[1].row] == 950 and run.columns["x0"][yolo[0].row] == 905
    )
    truck = match_references(run, refs(ref(Selector(point=(200, 200), label="truck"))))
    assert [m.row is not None for m in truck] == [True, False] * 3
    assert not any(
        m.row is not None
        for m in match_references(
            run, refs(ref(Selector(point=(200, 200), label="car")))
        )
    )
    tracked = match_references(run, refs(ref(Selector(track_id=2), frames=(0, 9))))
    rows = [m.row for m in tracked if m.source == "rfdetr"]
    assert len(rows) == 10 and set(run.columns["track_id"][rows]) == {2.0}


def test_time_range_and_series(run):
    series = ref(
        Selector(box=CAR),
        frames=None,
        t=(0.1, 0.2),
        distance_m=None,
        series=((0.0, 9.0), (0.3, 12.0)),
    )
    matches = [m for m in match_references(run, refs(series)) if m.source == "rfdetr"]
    assert [m.frame for m in matches] == [3, 4, 5, 6]  # t = 0.1 ... 0.2 inclusive
    assert matches[0].reference_m == pytest.approx(10.0)  # 9 + (0.1 / 0.3) * 3
    short = ref(Selector(box=CAR), distance_m=None, series=((0.0, 9.0), (0.1, 10.0)))
    with pytest.raises(
        ConfigError, match="reference r: t=0.133 s is outside the series"
    ):
        match_references(run, refs(short))


def test_off_axis_range_becomes_z(run):
    left = (100.0, 100.0, 400.0, 300.0)  # the truck, far from the principal point
    m = match_references(run, refs(ref(Selector(box=left))))[0]
    k = run.frame_intrinsics
    factor = math.sqrt(1 + ((250 - k.cx) / k.fx) ** 2 + ((200 - k.cy) / k.fy) ** 2)
    assert m.reference_m == pytest.approx(10.0 / factor, rel=1e-3) and factor > 1
    assert m.uncertainty_m == pytest.approx(math.hypot(0.01, 0.1) / factor, rel=1e-3)
    as_z = match_references(run, refs(ref(Selector(box=left), quantity="z")))[0]
    assert as_z.reference_m == 10.0


def test_reference_to_run_quantity():
    k = CameraIntrinsics.centered(1000.0, 2000, 1000)
    assert reference_to_run_quantity(10.0, "range", "z", (1000.0, 500.0), k)[0] == (
        pytest.approx(10.0 / math.sqrt(1 + (0.5 / 1000) ** 2 * 2))
    )
    corner = (1999.5, 500.5)  # pixel (1999, 500): u = 0.9995, v = 0.0005
    z, converted = reference_to_run_quantity(10.0, "range", "z", corner, k)
    assert converted and z == pytest.approx(10.0 / math.hypot(1, 0.9995), rel=1e-6)
    r, _ = reference_to_run_quantity(z, "z", "euclidean", corner, k)
    assert r == pytest.approx(10.0)
    assert reference_to_run_quantity(10.0, "range", "euclidean", corner, k) == (
        10.0,
        False,
    )
    assert reference_to_run_quantity(10.0, "range", "z", corner, None) == (10.0, False)


def test_points_transform_moves_reference_pixels(run):
    shifted = tuple(np.array(CAR) - 50)  # annotated on the "original" video
    plain = match_references(run, refs(ref(Selector(box=shifted))), iou_min=0.5)
    assert all(m.row is None for m in plain)
    moved = match_references(
        run,
        refs(ref(Selector(box=shifted))),
        iou_min=0.5,
        points_transform=lambda xy: xy + 50,
    )
    assert all(m.row is not None for m in moved)


def test_track_selector_needs_tracking(fake_run_dir):
    untracked = load_run(fake_run_dir([row(2, CAR)]))
    with pytest.raises(ConfigError, match="has no tracking"):
        match_references(untracked, refs(ref(Selector(track_id=1))))
