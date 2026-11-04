import math
from pathlib import Path

import pytest
import yaml

from av_perception_lab.core.options import ConfigError
from av_perception_lab.evaluation.references import (
    Reference,
    Selector,
    load_references,
)

EXAMPLE = (
    Path(__file__).parents[1] / "configs" / "references" / "example_references.yaml"
)
LASER = {"description": "laser", "uncertainty_m": 0.003, "uncertainty_rel": 0.001}


def write(tmp_path, references, **top):
    data = {"video": "IMG_4100.MOV", "instruments": {"laser": LASER}, **top}
    data["references"] = references
    path = tmp_path / "refs.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def ref(**fields):
    return {
        "id": "r01",
        "frames": [10, 20],
        "select": {"box": [0, 0, 10, 10]},
        "distance_m": 12.0,
        "instrument": "laser",
        **fields,
    }


def test_example_file_is_valid():
    refs = load_references(EXAMPLE)
    assert refs.video == "IMG_4100.MOV" and refs.version == 1
    assert [r.id for r in refs.references] == ["r01", "r02", "r03"]
    r01, r02, r03 = refs.references
    assert r01.select == Selector(box=(812.0, 430.0, 1104.0, 655.0), label="car")
    assert r02.select.point == (1500.0, 610.0) and r02.quantity == "z"
    assert r03.select.track_id == 3 and r03.series[1] == (42.0, 21.3)
    assert refs.uncertainty(r01, 12.4) == pytest.approx(math.hypot(0.003, 0.0124))
    assert refs.uncertainty(r03, 20.0) == 0.05  # the reference's own value wins


def test_covers_and_distance_at():
    by_frame = Reference("a", Selector(point=(1, 1)), frames=(10, 20), distance_m=5.0)
    assert by_frame.covers(10, 99.0) and by_frame.covers(20, 0.0)
    assert not by_frame.covers(21, 0.0)
    assert by_frame.distance_at(123.0) == 5.0
    series = Reference(
        "b",
        Selector(track_id=1),
        t=(1.0, 3.0),
        series=((1.0, 10.0), (3.0, 20.0)),
        uncertainty_m=0.1,
    )
    assert series.covers(0, 1.0 - 1e-9) and not series.covers(0, 3.1)
    assert series.distance_at(2.0) == pytest.approx(15.0)
    with pytest.raises(ValueError, match="outside the series"):
        series.distance_at(3.5)


@pytest.mark.parametrize(
    ("references", "match"),
    [
        (
            [ref(), ref(id="r02", select={"box": [0, 0, 1, 1], "point": [1, 1]})],
            r"references\[1\]\.select: exactly one of box, point, track_id",
        ),
        ([ref(select={"box": [5, 0, 1, 1]})], r"references\[0\]\.select: box must"),
        ([ref(select={"box": [0, 0, 1]})], r"references\[0\]\.select\.box: expected 4"),
        ([ref(t=[1.0, 2.0])], r"references\[0\]: exactly one of frames, t"),
        ([ref(frames=[20, 10])], r"references\[0\]: frames must be"),
        ([ref(series=[[0.0, 1.0]])], r"exactly one of distance_m, series"),
        ([ref(distance_m=0)], "distance_m must be > 0"),
        (
            [ref(distance_m=None, frames=None, t=[0.0, 5.0], series=[[0, 1], [4, 2]])],
            "series must cover the t range",
        ),
        (
            [ref(distance_m=None, series=[[1, 1], [1, 2]])],
            "series times must be strictly increasing",
        ),
        ([ref(), ref()], r"references\[1\]\.id: duplicate id 'r01'"),
        ([ref(instrument="tape")], r"references\[0\]\.instrument: unknown instrument"),
        ([ref(instrument=None)], r"references\[0\]: set instrument or uncertainty_m"),
        ([ref(quantity="depth")], r"references\[0\]\.quantity: 'depth' is not one of"),
        ([ref(colour="red")], r"references\[0\]\.colour: unknown option"),
        ([], "at least one reference is required"),
    ],
)
def test_validation_errors(tmp_path, references, match):
    path = write(tmp_path, references)
    with pytest.raises(ConfigError, match=match) as exc:
        load_references(path)
    assert str(exc.value).startswith(f"{path}: ")


def test_file_errors(tmp_path):
    with pytest.raises(ConfigError, match="references not found"):
        load_references(tmp_path / "missing.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("video: [unclosed\n")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_references(bad)
    no_video = tmp_path / "novideo.yaml"
    no_video.write_text(yaml.safe_dump({"references": [ref()]}))
    with pytest.raises(ConfigError, match=r"video: missing required option"):
        load_references(no_video)
    with pytest.raises(ConfigError, match="instruments.laser: uncertainty_m"):
        load_references(
            write(tmp_path, [ref()], instruments={"laser": {"uncertainty_m": -1}})
        )
