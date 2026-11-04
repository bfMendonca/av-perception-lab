"""Run summaries, config diff, distance series and artifacts (no server)."""

import json

import pytest

pytest.importorskip("fastapi")

from run_folders import NEW_HEADER, OLD_HEADER, make_run, write_csv

from av_perception_lab.web import compare
from av_perception_lab.web.compare import (
    NoRunFiles,
    config_diff,
    distance_series,
    list_artifacts,
    run_summary,
)


def test_config_diff_lists_changed_leaf_paths():
    a = {
        "smoothing": {"accel_std": 2.0, "filter": "kalman"},
        "detectors": [{"type": "rfdetr", "threshold": 0.5}],
        "source": {"start": 0.0},
    }
    b = {
        "smoothing": {"accel_std": 3.5, "filter": "kalman"},
        "detectors": [{"type": "rfdetr", "threshold": 0.4}],
        "source": {"start": 0.0},
    }
    diff = config_diff(a, b)
    assert [(d["path"], d["a"], d["b"]) for d in diff] == [
        ("smoothing.accel_std", 2.0, 3.5),
        ("detectors[0].threshold", 0.5, 0.4),
    ]
    assert all(d["change"] == "changed" for d in diff)
    assert config_diff(a, a) == []


def test_config_diff_handles_added_and_removed_keys():
    a = {"tracking": {"type": "iou"}, "detectors": [{"type": "x"}], "gone": None}
    b = {
        "tracking": {"type": "iou", "max_age_s": 0.5},
        "detectors": [{"type": "x"}, {"type": "y", "threshold": 0.3}],
        "motion": {"models": {"cv": {"type": "cv"}}},
    }
    diff = {d["path"]: d for d in config_diff(a, b)}
    assert set(diff) == {"tracking.max_age_s", "detectors[1]", "motion", "gone"}
    assert diff["tracking.max_age_s"] == {
        "path": "tracking.max_age_s",
        "a": None,
        "b": 0.5,
        "change": "added",
    }
    assert diff["detectors[1]"]["b"] == {"type": "y", "threshold": 0.3}
    assert diff["motion"]["change"] == "added"
    # A key whose value is null is still "present": removing it is a change.
    assert (diff["gone"]["change"], diff["gone"]["a"]) == ("removed", None)
    # A type change (mapping -> scalar) is one leaf, not a cascade.
    (only,) = config_diff({"x": {"y": 1}}, {"x": 3})
    assert (only["path"], only["a"], only["b"]) == ("x", {"y": 1}, 3)


def test_summary_jitter_known_series(tmp_path):
    out = tmp_path / "out"
    # Track 1: 10, 11, 13 -> |d| = 1, 2 -> median 1.5.
    # Track 2: 20, 20.5 then a gap (frame 3 missing), then 30 -> only the
    # consecutive pair counts: 0.5. Median over tracks: (1.5 + 0.5) / 2 = 1.0.
    rows = [
        [0, 0.0, "car", 10.0, "d", "car", 1, 10.0],
        [1, 0.1, "car", 11.0, "d", "car", 1, 10.6],
        [2, 0.2, "car", 13.0, "d", "car", 1, 11.4],
        [0, 0.0, "car", 20.0, "d", "car", 2, 20.0],
        [1, 0.1, "car", 20.5, "d", "car", 2, 20.2],
        [4, 0.4, "car", 30.0, "d", "car", 2, 21.0],
        [4, 0.4, "car", "", "d", "car", "", ""],  # no distance, no track
    ]
    metrics = {
        "frames_processed": 5,
        "end_to_end_fps": 2.5,
        "config": {"source": {"start": 0.0, "stride": 1}},
        "video": {"fps": 10.0},
    }
    make_run(out, "r1", NEW_HEADER, rows, metrics)
    s = run_summary(out, "r1")
    assert s["jitter_m"] == pytest.approx(1.0)
    # smoothed: track 1 |d| = 0.6, 0.8 -> 0.7; track 2: 0.2 -> median 0.45
    assert s["jitter_smooth_m"] == pytest.approx(0.45)
    assert s["median_distance_m"] == pytest.approx(16.5)  # of the 6 finite rows
    assert (s["fps"], s["frames"], s["boxes"], s["tracks"]) == (2.5, 5, 7, 2)


def test_summary_tolerates_old_run_without_smoothed_columns(tmp_path):
    out = tmp_path / "out"
    rows = [[0, 0.0, "car", 10.0, "d", "car"], [1, 0.1, "car", 12.0, "d", "car"]]
    make_run(out, "old", OLD_HEADER, rows)  # no metrics.json, no config.yaml
    s = run_summary(out, "old")
    assert s["median_distance_m"] == pytest.approx(11.0)
    assert (s["jitter_m"], s["jitter_smooth_m"], s["tracks"]) == (None, None, None)
    assert (s["frames"], s["boxes"], s["fps"]) == (2, 2, None)
    assert s["notes"]  # tells why jitter is missing
    series = distance_series(out, "old", None)
    assert series["track"] is None and series["smoothed"] is None
    assert series["t"] == [0.0, 0.1] and series["raw"] == [10.0, 12.0]
    assert series["video_t"] == pytest.approx([0.0, 0.1])
    # a folder without any detections.csv still summarises (all unknown)
    (out / "empty").mkdir()
    (out / "empty" / "video.mp4").write_bytes(b"")
    assert run_summary(out, "empty")["boxes"] is None


def test_series_picks_longest_track_maps_video_time_and_downsamples(tmp_path):
    out = tmp_path / "out"
    rows = [[f, f / 10, "car", 10 + f % 7, "d", "car", 5, 10.0] for f in range(4000)]
    rows += [[0, 0.0, "car", 50.0, "d", "car", 9, 50.0]]
    metrics = {
        "frames_processed": 4000,
        # start 2 s at 10 fps -> first source frame 20 (frames below are source
        # indices from 20 on), stride 1.
        "config": {"source": {"start": 2.0, "stride": 1}},
        "video": {"fps": 10.0},
    }
    rows = [[r[0] + 20, *r[1:]] for r in rows]
    make_run(out, "long", NEW_HEADER, rows, metrics)
    s = distance_series(out, "long", None, max_points=500)
    assert s["track"] == 5 and s["points"] == 4000 and s["downsampled"]
    assert len(s["t"]) <= 500 and len(s["t"]) == len(s["raw"]) == len(s["video_t"])
    assert s["video_t"][0] == pytest.approx(0.0)  # source frame 20 -> 0 s
    assert min(s["raw"]) == 10 and max(s["raw"]) == 16  # extremes kept
    assert [t["id"] for t in s["tracks"]] == [5, 9]
    assert distance_series(out, "long", 9)["raw"] == [50.0]
    missing = distance_series(out, "long", 77)
    assert missing["found"] is False and missing["t"] == []


def test_artifacts_lists_unknown_files_in_run_dir(tmp_path):
    out = tmp_path / "out"
    run = make_run(out, "r", OLD_HEADER, [[0, 0.0, "car", 1.0, "d", "car"]])
    (run / "video.mp4").write_bytes(b"\0" * 10)
    (run / "brand_new_sink.parquet").write_bytes(b"PAR1")
    (run / "trajectories.jsonl").write_text("{}\n")
    (run / "plots").mkdir()
    (run / "plots" / "err.png").write_bytes(b"\x89PNG")
    (run / ".hidden").write_text("x")
    arts = {a["name"]: a for a in list_artifacts(out, "r")}
    assert set(arts) == {
        "detections.csv",
        "video.mp4",
        "brand_new_sink.parquet",
        "trajectories.jsonl",
        "plots/err.png",
    }
    assert arts["video.mp4"]["size"] == 10
    assert arts["video.mp4"]["media_type"] == "video/mp4"
    assert arts["video.mp4"]["role"] == "video"
    assert arts["detections.csv"]["role"] == "detections"
    assert arts["plots/err.png"]["media_type"] == "image/png"
    assert arts["trajectories.jsonl"]["media_type"] == "application/x-ndjson"
    assert arts["brand_new_sink.parquet"]["role"] is None
    assert arts["plots/err.png"]["url"] == "/api/runs/r/files/plots/err.png"


def test_job_entries_have_no_files(tmp_path):
    out = tmp_path / "out"
    jobs = out / ".web_jobs"
    jobs.mkdir(parents=True)
    (jobs / "abc123.json").write_text(
        json.dumps({"id": "abc123", "status": "failed", "error": "boom"})
    )
    for call in (run_summary, list_artifacts):
        with pytest.raises(NoRunFiles, match="no files"):
            call(out, "job-abc123")
    with pytest.raises(NoRunFiles):
        distance_series(out, "job-abc123", None)


def test_columns_are_cached_until_the_csv_changes(tmp_path, monkeypatch):
    out = tmp_path / "out"
    run = make_run(out, "r", OLD_HEADER, [[0, 0.0, "car", 1.0, "d", "car"]])
    calls = []
    real = compare.read_columns
    monkeypatch.setattr(
        compare, "read_columns", lambda *a, **k: calls.append(1) or real(*a, **k)
    )
    compare._columns.cache_clear()
    run_summary(out, "r")
    run_summary(out, "r")
    assert len(calls) == 1
    write_csv(
        run / "detections.csv",
        OLD_HEADER,
        [[0, 0.0, "car", 1.0, "d", "car"], [1, 0.1, "car", 3.0, "d", "car"]],
    )
    assert run_summary(out, "r")["boxes"] == 2
    assert len(calls) == 2
