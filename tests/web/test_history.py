"""History scan: tolerant of old folders, incomplete runs, id traversal."""

import json
import os

import pytest

pytest.importorskip("fastapi")

from av_perception_lab.web.history import delete_run, load_run, run_size, scan_runs

METRICS = {
    "video": {"path": "data/IMG_1.MOV"},
    "frames_processed": 5,
    "wall_time_s": 3.5,
    "components": {
        "detectors": [{"name": "rfdetr-large"}],
        "depth": {"name": "da3metric-large"},
    },
}


def test_scan_tolerates_missing_metrics_and_config(tmp_path):
    old = tmp_path / "20250101_101010_old"
    old.mkdir()
    (old / "detections.csv").write_text("frame,t\n0,0.0\n0,0.0\n1,0.1\n")
    (tmp_path / "notes").mkdir()  # not a run folder
    (tmp_path / "stray.txt").write_text("x")
    (rec,) = scan_runs(tmp_path)
    assert rec.id == "20250101_101010_old"
    assert rec.created == "2025-01-01T10:10:10"
    assert (rec.video, rec.detectors, rec.depth, rec.wall_s) == (None, [], None, None)
    assert rec.frames == 2
    assert (rec.status, rec.complete, rec.has_config) == ("incomplete", False, False)
    assert scan_runs(tmp_path / "missing") == []


def test_scan_reads_finished_run_and_web_status(tmp_path):
    done = tmp_path / "20260102_000000_a"
    done.mkdir()
    (done / "metrics.json").write_text(json.dumps(METRICS))
    (done / "config.yaml").write_text("x: 1\n")
    legacy = load_run(tmp_path, done.name)
    assert (legacy.status, legacy.complete, legacy.has_config) == ("done", True, True)
    assert (legacy.video, legacy.detectors, legacy.depth) == (
        "IMG_1.MOV",
        ["rfdetr-large"],
        "da3metric-large",
    )
    assert (legacy.frames, legacy.wall_s) == (5, 3.5)
    (done / "web_run.json").write_text(json.dumps({"status": "cancelled"}))
    assert load_run(tmp_path, done.name).complete is False


def test_scan_marks_incomplete_run(tmp_path):
    run = tmp_path / "20260103_000000_b"
    run.mkdir()
    (run / "detections.csv").write_text("frame\n0\n")
    (run / "web_run.json").write_text(json.dumps({"status": "running"}))
    (rec,) = scan_runs(tmp_path)
    assert (rec.status, rec.complete) == ("running", False)
    (run / "web_run.json").write_text("{broken")  # unreadable: fall back
    (rec,) = scan_runs(tmp_path)
    assert (rec.status, rec.complete) == ("incomplete", False)


def test_scan_is_newest_first(tmp_path):
    for name in ("20260101_000000_a", "20260102_000000_a"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "metrics.json").write_text(json.dumps(METRICS))
    assert next(r.id for r in scan_runs(tmp_path)) == "20260102_000000_a"


def test_run_id_traversal_rejected(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (tmp_path / "secret").mkdir()
    (out / "link").symlink_to(tmp_path / "secret")
    for bad in ("../secret", "..", ".", "", "a/b", str(tmp_path / "secret"), "link"):
        with pytest.raises(ValueError):
            load_run(out, bad)
    with pytest.raises(FileNotFoundError):
        load_run(out, "nope")
    assert os.path.isdir(tmp_path / "secret")


def test_run_size_sums_regular_files(tmp_path):
    run = tmp_path / "20250101_101010_a"
    run.mkdir()
    (run / "detections.csv").write_bytes(b"x" * 10)
    (run / "metrics.json").write_bytes(b"y" * 5)
    assert run_size(tmp_path, run.name) == 15


def test_run_size_ignores_symlinks(tmp_path):
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"z" * 1000)
    out = tmp_path / "out"
    run = out / "20250101_101010_a"
    run.mkdir(parents=True)
    (run / "detections.csv").write_bytes(b"x" * 10)
    (run / "link.bin").symlink_to(outside)
    assert run_size(out, run.name) == 10


def test_delete_run_removes_folder_not_symlink_target(tmp_path):
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"z" * 1000)
    out = tmp_path / "out"
    run = out / "20250101_101010_a"
    run.mkdir(parents=True)
    (run / "detections.csv").write_text("frame\n0\n")
    (run / "link.bin").symlink_to(outside)
    delete_run(out, run.name)
    assert not run.exists()
    assert outside.exists()


def test_delete_job_record(tmp_path):
    jobs = tmp_path / ".web_jobs"
    jobs.mkdir()
    path = jobs / "abc123.json"
    path.write_text(
        json.dumps({"id": "abc123", "status": "failed", "created": "2025-01-01"})
    )
    assert run_size(tmp_path, "job-abc123") == path.stat().st_size
    delete_run(tmp_path, "job-abc123")
    assert not path.exists()
    assert scan_runs(tmp_path) == []


@pytest.mark.parametrize("bad", ["../x", ".web_jobs", "a/b", ""])
def test_run_ops_reject_bad_ids(tmp_path, bad):
    with pytest.raises(ValueError):
        run_size(tmp_path, bad)
    with pytest.raises(ValueError):
        delete_run(tmp_path, bad)


def test_run_ops_unknown_run(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_size(tmp_path, "nope")
    with pytest.raises(FileNotFoundError):
        delete_run(tmp_path, "job-nope")


def test_delete_run_also_drops_failed_job_record_of_the_folder(tmp_path):
    folder = tmp_path / "20250101_101010_a"
    folder.mkdir()
    (folder / "web_run.json").write_text("{}")
    jobs = tmp_path / ".web_jobs"
    jobs.mkdir()
    rec = jobs / "abc123.json"
    rec.write_text(
        json.dumps(
            {
                "id": "abc123",
                "status": "failed",
                "created": "2025-01-01",
                "run_dir": str(folder),
            }
        )
    )
    other = jobs / "other.json"
    other.write_text(json.dumps({"id": "other", "status": "failed"}))
    delete_run(tmp_path, folder.name)
    assert not rec.exists() and other.exists()
    assert [r.id for r in scan_runs(tmp_path)] == ["job-other"]


def test_run_ops_refuse_folder_without_marker(tmp_path):
    folder = tmp_path / "library"
    folder.mkdir()
    (folder / "x.yaml").write_text("a: 1\n")
    with pytest.raises(FileNotFoundError):
        run_size(tmp_path, "library")
    with pytest.raises(FileNotFoundError):
        delete_run(tmp_path, "library")
    assert folder.exists()
