import dataclasses
import json
import sys

import pytest

from av_perception_lab.core.options import ConfigError
from av_perception_lab.evaluation.evaluate import EvalOptions, evaluate
from av_perception_lab.evaluation.references import (
    Instrument,
    Reference,
    ReferenceSet,
    Selector,
)
from av_perception_lab.evaluation.report import render_markdown, write_report
from av_perception_lab.evaluation.run_data import load_run

TWO_DETECTORS = {
    "components": {
        "detectors": [{"name": "rfdetr"}, {"name": "yolo"}],
        "depth": {"kind": "metric_focal"},
        "aggregator": {"type": "concat"},
    }
}
NEAR = (900.0, 500.0, 1020.0, 580.0)
FAR = (1500.0, 600.0, 1540.0, 630.0)


def row(frame, box, source, distance, label="car"):
    x0, y0, x1, y1 = box
    return {"frame": frame, "t": frame / 30, "class_name": label, "x0": x0, "y0": y0,
            "x1": x1, "y1": y1, "distance_m": distance, "median": distance,
            "source": source, "raw_label": label}  # fmt: skip


@pytest.fixture
def runs(fake_run_dir):
    """Two runs of the same video: the second 10 % farther (e.g. another focal)."""
    out = []
    for scale in (1.0, 1.1):
        rows = []
        for f in range(10):
            rows += [
                row(f, NEAR, "rfdetr", 8.0 * scale),
                row(f, NEAR, "yolo", 8.4 * scale),
                row(f, FAR, "rfdetr", 30.0 * scale if f % 2 else "", "person"),
            ]
        out.append(load_run(fake_run_dir(rows, TWO_DETECTORS)))
    return out


@pytest.fixture
def refs():
    return ReferenceSet(
        "IMG_4100.MOV",
        (
            Reference(
                "near",
                Selector(box=NEAR),
                frames=(0, 9),
                distance_m=8.0,
                instrument="laser",
                quantity="z",
            ),
            Reference(
                "far",
                Selector(point=(1520, 615), label="person"),
                frames=(0, 9),
                distance_m=30.0,
                instrument="laser",
                quantity="z",
            ),
            Reference(
                "late",
                Selector(box=NEAR),
                frames=(100, 120),
                distance_m=8.0,
                instrument="laser",
            ),
            Reference(
                "moving",
                Selector(track_id=4),
                frames=(0, 9),
                distance_m=9.0,
                instrument="laser",
            ),
        ),
        {"laser": Instrument(uncertainty_m=0.01)},
    )


def overall(result, run, source, estimator="distance_m"):
    (out,) = [
        r
        for r in result.overall
        if (r["run"], r["source"], r["estimator"]) == (run, source, estimator)
    ]
    return out


def test_evaluate_two_runs_side_by_side(runs, refs):
    result = evaluate(refs, runs)
    assert overall(result, "run0", "rfdetr")["median_abs_rel"] == pytest.approx(0.0)
    assert overall(result, "run1", "rfdetr")["scale_median"] == pytest.approx(1.1)
    assert overall(result, "run0", "yolo")["bias_m"] == pytest.approx(0.4)
    assert overall(result, "run0", "yolo")["n_refs"] == 1  # yolo has no person box
    assert {r["estimator"] for r in result.overall} == {"distance_m", "median"}
    focal = result.implied_focal
    assert focal["run0"]["rfdetr"] == pytest.approx(1400.0)
    assert focal["run1"]["rfdetr"] == pytest.approx(1400.0 / 1.1)
    assert focal["run0"]["yolo"] == pytest.approx(1400.0 / 1.05)
    bins = {(r["run"], r["source"], r["bin"]) for r in result.tables["bin"]}
    assert ("run0", "rfdetr", "20-30") not in bins and (
        "run0",
        "rfdetr",
        "30-50",
    ) in bins
    far = [c for c in result.coverage if (c["run"], c["ref_id"]) == ("run0", "far")]
    rfdetr_far = next(c for c in far if c["source"] == "rfdetr")
    assert (rfdetr_far["detected_valid"], rfdetr_far["detected_na"]) == (5, 5)
    warnings = "\n".join(result.warnings)
    assert "reference late is outside the processed frames" in warnings
    assert "reference moving selects a track id, skipped (no tracking)" in warnings
    json.dumps(result.to_dict(), allow_nan=True)


def test_evaluate_checks_the_video_and_undistortion(runs, refs, fake_run_dir):
    other = ReferenceSet("IMG_9999.MOV", refs.references, refs.instruments)
    with pytest.raises(ConfigError, match="the references are for IMG_9999.MOV"):
        evaluate(other, runs)
    forced = evaluate(other, runs, EvalOptions(force=True))
    assert any("IMG_9999.MOV" in w for w in forced.warnings)
    undistorted = load_run(
        fake_run_dir(
            [],
            {"camera": {"calibration": {"path": "c.yaml"}, "undistort": {"alpha": 0}}},
        )
    )
    with pytest.raises(ConfigError, match="need its undistortion transform"):
        evaluate(refs, [undistorted])


def test_evaluate_refuses_runs_with_the_same_name(runs, refs, tmp_path):
    # Folders of different out_roots can share a name (same second, same parts);
    # their results would silently merge (task_0008 smoke).
    twin = dataclasses.replace(runs[1], run_dir=tmp_path / "other" / runs[0].name)
    with pytest.raises(ConfigError, match=f"two runs are named {runs[0].name}"):
        evaluate(refs, [runs[0], twin])


def test_range_references_without_intrinsics_warn(fake_run_dir, refs):
    run = load_run(
        fake_run_dir([row(0, NEAR, "rfdetr", 8.0)], {"frame_intrinsics": None})
    )
    result = evaluate(refs, [run])
    assert any("no frame intrinsics" in w for w in result.warnings)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"estimators": ("mode",)}, "estimators: unknown"),
        ({"level": "track"}, "level"),
        ({"bins": (0.0, 10.0, 5.0)}, "strictly increasing"),
        ({"iou_min": 0.0}, r"iou_min must be in \(0, 1\]"),
    ],
)
def test_options_errors(kwargs, match):
    with pytest.raises(ConfigError, match=match):
        EvalOptions(**kwargs)


def test_markdown_and_files(runs, refs, tmp_path, monkeypatch):
    result = evaluate(refs, runs)
    text = render_markdown(result)
    for heading in (
        "# Evaluation against references of IMG_4100.MOV",
        "## Overall",
        "## By reference distance (m) (distance_m)",
        "## By region (mask / box) (distance_m)",
        "## Coverage",
        "## Implied focal",
        "## Warnings",
    ):
        assert heading in text
    assert "| run1 | rfdetr | distance_m |" in text and "10.0 %" in text
    paths = write_report(result, tmp_path / "eval")
    assert [p.name for p in paths] == ["eval.json", "eval.md"]
    data = json.loads(paths[0].read_text())  # strict JSON: NaN written as null
    assert data["references"]["video"] == "IMG_4100.MOV"
    assert data["options"]["level"] == "reference"
    monkeypatch.setitem(sys.modules, "matplotlib", None)  # as in the light env
    with pytest.raises(ConfigError, match="uv sync --extra eval"):
        write_report(result, tmp_path / "plots", plot=True)
    assert not (tmp_path / "plots").exists()  # nothing written before the check


def test_plots(runs, refs, tmp_path):
    pytest.importorskip("matplotlib")
    paths = write_report(evaluate(refs, runs), tmp_path, plot=True)
    names = [p.name for p in paths]
    assert names[2:] == ["eval_error_vs_distance.png", "eval_abs_rel_by_bin.png"]
    assert all(p.stat().st_size > 0 for p in paths)
