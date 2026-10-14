import math
from dataclasses import replace

import numpy as np
import pytest

from det_depth_distance.evaluation.matching import Match
from det_depth_distance.evaluation.run_data import load_run
from det_depth_distance.evaluation.stats import (
    Sample,
    bin_label,
    build_samples,
    coverage_summary,
    error_stats,
    implied_focal,
    summarize,
)


def test_error_stats_by_hand():
    est = np.array([11.0, 9.0, 22.0, 40.0])
    ref = np.array([10.0, 10.0, 20.0, 50.0])  # errors 1, -1, 2, -10
    unc = np.array([0.6, 0.4, 0.5, 6.0])
    s = error_stats(est, ref, unc, ["a", "a", "b", "c"])
    assert (s.n, s.n_refs) == (4, 3)
    assert s.bias_m == pytest.approx(-2.0)
    assert s.median_signed_m == pytest.approx(0.0)
    assert s.mae_m == pytest.approx(3.5)
    assert s.median_abs_m == pytest.approx(1.5)
    assert s.p90_abs_m == pytest.approx(np.percentile([1, 1, 2, 10], 90))
    assert s.rmse_m == pytest.approx(math.sqrt((1 + 1 + 4 + 100) / 4))
    assert s.mean_rel == pytest.approx((0.1 - 0.1 + 0.1 - 0.2) / 4)
    assert s.mean_abs_rel == pytest.approx(0.125)
    assert s.median_abs_rel == pytest.approx(0.1)
    assert s.within_2u == pytest.approx(0.5)  # only |1| <= 1.2 and |-10| <= 12
    assert s.scale_median == pytest.approx(np.median([1.1, 0.9, 1.1, 0.8]))
    assert s.reference_median_m == 15.0


def test_error_stats_empty_and_without_uncertainty():
    empty = error_stats(np.zeros(0), np.zeros(0))
    assert empty.n == 0 and math.isnan(empty.bias_m) and math.isnan(empty.within_2u)
    assert math.isnan(error_stats(np.ones(2), np.ones(2)).within_2u)
    assert error_stats(np.ones(2), np.ones(2)).n_refs == 2


def sample(ref_id, estimate, reference, frame=0, estimator="distance_m", **kw):
    fields = {
        "run": "run0",
        "source": "rfdetr",
        "label": "car",
        "region": "box",
        "mask_source": "",
        "track_id": None,
        "uncertainty": 0.1,
        **kw,
    }
    return Sample(ref_id=ref_id, frame=frame, estimator=estimator, estimate=estimate,
                  reference=reference, **fields)  # fmt: skip


def test_reference_level_weights_each_reference_once():
    # r1: 100 frames, error +1 m at 10 m; r2: one frame, error -4 m at 40 m.
    samples = [sample("r1", 11.0, 10.0, frame=f) for f in range(100)]
    samples += [sample("r2", 36.0, 40.0), sample("r2", float("nan"), 40.0, frame=1)]
    (by_ref,) = summarize(samples, by=("run",))
    assert (by_ref["n"], by_ref["n_refs"]) == (2, 2)
    assert by_ref["bias_m"] == pytest.approx(-1.5)
    (by_frame,) = summarize(samples, by=("run",), level="frame")
    assert by_frame["n"] == 101  # the n/a frame is skipped
    assert by_frame["bias_m"] == pytest.approx((100 - 4) / 101)


def test_grouping_by_bin_region_and_estimator():
    samples = [
        sample("a", 4.5, 4.0),
        sample("b", 12.0, 10.0, region="mask", mask_source="sam2"),
        sample("c", 66.0, 60.0),
        sample("c", 63.0, 60.0, estimator="distance_smooth_m"),
    ]
    rows = summarize(samples, by=("estimator", "bin"))
    keys = [(r["estimator"], r["bin"]) for r in rows]
    assert keys == [
        ("distance_m", "0-5"),
        ("distance_m", "10-20"),
        ("distance_m", ">=50"),
        ("distance_smooth_m", ">=50"),
    ]
    regions = {r["region"]: r["n"] for r in summarize(samples, by=("region",))}
    assert regions == {"box": 2, "mask": 1}  # per reference: c counts once
    assert (
        bin_label(5.0, (0, 5, 10)) == "5-10" and bin_label(10.0, (0, 5, 10)) == ">=10"
    )
    assert bin_label(2.0, (0.0, 2.5)) == "0-2.5"


def match(ref_id, frame, row, reference=10.0, source="rfdetr"):
    return Match(ref_id, frame, frame / 30, source, row, 0.0, reference, 0.1)


@pytest.fixture
def run(fake_run_dir):
    def r(frame, distance, smooth, track, used_mask=0, box=None):
        return {
            "frame": frame, "class_name": "car", "x0": 0, "y0": 0, "x1": 10, "y1": 10,
            "distance_m": distance, "median": 11.0, "p90": 13.0, "source": "rfdetr",
            "raw_label": "car", "track_id": track, "distance_smooth_m": smooth,
            "used_mask": used_mask, "box_distance_m": box,
            "mask_source": "sam2" if used_mask else "",
        }  # fmt: skip

    rows = [r(0, 11.0, 11.0, 1), r(1, "", 10.8, 1), r(2, 12.1, 11.2, 4, 1, 12.6)]
    return load_run(fake_run_dir(rows))


def test_build_samples(run):
    matches = [match("r1", f, f) for f in range(3)] + [match("r1", 3, None)]
    estimators = ("distance_m", "median", "distance_smooth_m", "box_distance_m", "min")
    samples = build_samples(run, matches, estimators)
    by = {(s.frame, s.estimator): s for s in samples}
    assert len(samples) == 3 * 5  # misses give no samples; every CSV estimator
    assert math.isnan(by[(1, "distance_m")].estimate)
    assert math.isnan(by[(1, "median")].estimate)  # inlier stat without distance_m
    assert by[(1, "distance_smooth_m")].estimate == 10.8  # predicted through n/a
    assert by[(0, "median")].estimate == 11.0 and math.isnan(by[(0, "min")].estimate)
    assert by[(2, "box_distance_m")].estimate == 12.6
    assert (by[(2, "distance_m")].region, by[(2, "distance_m")].mask_source) == (
        "mask",
        "sam2",
    )
    assert by[(0, "distance_m")].track_id == 1 and by[(0, "distance_m")].label == "car"


def test_coverage_and_id_switches(run):
    matches = [match("r1", f, f) for f in range(3)] + [match("r1", 3, None)]
    (cov,) = coverage_summary(matches, run)
    assert cov == {
        "ref_id": "r1",
        "source": "rfdetr",
        "frames": 4,
        "detected_valid": 2,
        "detected_na": 1,
        "not_detected": 1,
        "valid_by_estimator": {
            "distance_m": 2,
            "median": 3,
            "p90": 3,
            "box_distance_m": 1,
            "distance_smooth_m": 3,
        },
        "track_ids": 2,  # ids 1 and 4: one switch
    }


def test_coverage_without_tracking(fake_run_dir):
    rows = [{"frame": 0, "class_name": "car", "x0": 0, "y0": 0, "x1": 1, "y1": 1,
             "distance_m": 5.0, "source": "rfdetr", "raw_label": "car"}]  # fmt: skip
    (cov,) = coverage_summary([match("r1", 0, 0)], load_run(fake_run_dir(rows)))
    assert cov["track_ids"] is None  # unknown, not zero


def test_implied_focal(run, fake_run_dir):
    samples = [
        sample("r1", 11.0, 10.0, run="run0"),
        sample("r2", 22.0, 20.0, run="run0"),
    ]
    samples.append(sample("r3", 50.0, 20.0, run="run0", estimator="median"))  # ignored
    assert implied_focal(run, samples) == pytest.approx(1400.0 / 1.1)
    assert implied_focal(run, []) is None
    # DA3METRIC scales by (fx + fy) / 2: a calibrated K with fx != fy uses the mean.
    k = {"fx": 1390.0, "fy": 1410.0, "cx": 960.0, "cy": 540.0}
    calibrated = load_run(
        fake_run_dir([], {"frame_intrinsics": k, "focal_frame_px": 1390.0})
    )
    mine = [replace(s, run=calibrated.name) for s in samples]
    assert implied_focal(calibrated, mine) == pytest.approx(1400.0 / 1.1)
    nested = load_run(
        fake_run_dir(
            [],
            {
                "components": {
                    "detectors": [{"name": "rfdetr"}],
                    "depth": {"kind": "metric"},
                    "aggregator": {"type": "concat"},
                },
                "focal_frame_px": None,
            },
        )
    )
    assert implied_focal(nested, samples) is None
