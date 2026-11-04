"""Compare, series, artifacts, reopen and rerun routes (fake worker)."""

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from conftest import LOCAL
from fastapi.testclient import TestClient
from run_folders import NEW_HEADER, OLD_HEADER, make_run

from av_perception_lab.web.app import create_app

FAKE = Path(__file__).with_name("fake_worker.py")
CONFIG = {
    "detectors": [{"type": "rfdetr"}],
    "depth": {"type": "da3"},
    "smoothing": {"filter": "kalman", "accel_std": 2.0},
}
CONFIG_A = {**CONFIG, "output": {"tag": "web-a"}}  # every web job gets a tag


@pytest.fixture
def api(web_paths):
    app = create_app(web_paths, lambda p: [sys.executable, str(FAKE), str(p)])
    with TestClient(base_url=LOCAL, app=app) as client:
        yield client


def two_runs(out: Path) -> None:
    rows = [
        [0, 0.0, "car", 10.0, "d", "car", 1, 10.0],
        [1, 0.1, "car", 11.0, "d", "car", 1, 10.5],
    ]
    metrics = {"frames_processed": 2, "end_to_end_fps": 2.0}
    make_run(out, "20260101_000000_a.v1", NEW_HEADER, rows, metrics, CONFIG_A)
    faster = {
        **CONFIG,
        "smoothing": {"filter": "kalman", "accel_std": 3.5},
        "output": {"tag": "web-b"},
    }
    metrics_b = {"frames_processed": 2, "end_to_end_fps": 2.5}
    rows_b = [[*r[:3], r[3] + 1, *r[4:]] for r in rows]
    make_run(out, "20260101_000001_b", NEW_HEADER, rows_b, metrics_b, faster)


def test_compare_two_runs(api, web_paths):
    two_runs(web_paths.out_root)
    resp = api.get(
        "/api/compare", params={"a": "20260101_000000_a.v1", "b": "20260101_000001_b"}
    )
    assert resp.status_code == 200
    body = resp.json()
    diff = {d["path"]: d["server_set"] for d in body["config_diff"]}
    assert diff == {"smoothing.accel_std": False, "output.tag": True}
    assert body["summary_a"]["median_distance_m"] == pytest.approx(10.5)
    assert body["deltas"]["median_distance_m"]["abs"] == pytest.approx(1.0)
    assert body["deltas"]["fps"]["rel"] == pytest.approx(0.25)
    by_key = {m["key"]: m for m in body["metrics"]}
    assert by_key["jitter_m"]["better"] == "lower"
    one = api.get("/api/runs/20260101_000001_b/summary").json()
    assert one["summary"]["fps"] == 2.5 and one["metrics"] == body["metrics"]


def test_compare_missing_run_404(api, web_paths):
    two_runs(web_paths.out_root)
    resp = api.get("/api/compare", params={"a": "20260101_000000_a.v1", "b": "nope"})
    assert resp.status_code == 404 and "nope" in resp.json()["detail"]
    assert api.get("/api/compare", params={"a": "x"}).status_code == 422
    # a failed job with no folder: readable 404, never a 500
    jobs = web_paths.out_root / ".web_jobs"
    jobs.mkdir()
    (jobs / "abc.json").write_text(json.dumps({"id": "abc", "status": "failed"}))
    for url in (
        "/api/compare?a=job-abc&b=20260101_000001_b",
        "/api/runs/job-abc/series",
        "/api/runs/job-abc/summary",
        "/api/runs/job-abc/artifacts",
        "/api/runs/job-abc/files/detections.csv",
    ):
        resp = api.get(url)
        assert resp.status_code == 404, url
        assert "no files" in resp.json()["detail"]


def test_compare_rejects_traversal_ids(api, web_paths):
    two_runs(web_paths.out_root)
    secret = web_paths.out_root.parent / "secret"
    make_run(secret.parent, "secret", OLD_HEADER, [[0, 0.0, "car", 1, "d", "car"]])
    (web_paths.out_root / "link").symlink_to(secret)
    ok = "20260101_000001_b"
    for bad in ("..", "../secret", str(secret), "link", "job-../x", "a/b"):
        assert api.get("/api/compare", params={"a": bad, "b": ok}).status_code == 404
        assert api.get("/api/compare", params={"a": ok, "b": bad}).status_code == 404
    for rel in ("..%2f..%2fsecret%2fdetections.csv", "%2e%2e/secret/detections.csv"):
        assert api.get(f"/api/runs/{ok}/files/{rel}").status_code == 404
    (web_paths.out_root / ok / ".hidden").write_text("x")
    assert api.get(f"/api/runs/{ok}/files/.hidden").status_code == 404


def test_compare_ignores_columns_missing_in_old_runs(api, web_paths):
    two_runs(web_paths.out_root)
    out = web_paths.out_root
    make_run(out, "20250101_000000_old", OLD_HEADER, [[0, 0.0, "car", 9.0, "d", "car"]])
    resp = api.get(
        "/api/compare", params={"a": "20250101_000000_old", "b": "20260101_000001_b"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["summary_a"]["jitter_m"] is None
    assert "jitter_m" not in body["deltas"]  # only metrics both runs have
    assert body["deltas"]["median_distance_m"]["abs"] == pytest.approx(2.5)
    assert body["config_diff"] is None  # the old run recorded no configuration
    assert body["config_origin"] == {"a": None, "b": "config.yaml"}
    series = api.get("/api/runs/20250101_000000_old/series").json()
    assert series["smoothed"] is None and series["raw"] == [9.0]


def test_series_route_and_track_param(api, web_paths):
    two_runs(web_paths.out_root)
    body = api.get("/api/runs/20260101_000000_a.v1/series", params={"track": 1}).json()
    assert body["track"] == 1 and body["raw"] == [10.0, 11.0]
    assert body["smoothed"] == [10.0, 10.5]
    assert api.get("/api/runs/20260101_000000_a.v1/series?track=x").status_code == 422


def test_artifacts_lists_unknown_files_in_run_dir(api, web_paths):
    two_runs(web_paths.out_root)
    run = web_paths.out_root / "20260101_000001_b"
    (run / "brand_new_sink.bin").write_bytes(b"\1\2\3")
    arts = api.get("/api/runs/20260101_000001_b/artifacts").json()
    names = {a["name"]: a for a in arts}
    assert {
        "detections.csv",
        "metrics.json",
        "config.yaml",
        "brand_new_sink.bin",
    } <= set(names)
    assert names["metrics.json"]["role"] == "metrics"
    resp = api.get(names["brand_new_sink.bin"]["url"])
    assert resp.status_code == 200 and resp.content == b"\1\2\3"
    resp = api.get("/api/runs/20260101_000001_b/metrics")
    assert resp.json()["end_to_end_fps"] == 2.5


def test_reopen_returns_effective_config(api, web_paths):
    two_runs(web_paths.out_root)
    run = web_paths.out_root / "20260101_000000_a.v1"
    body = api.post(f"/api/runs/{run.name}/reopen").json()
    assert body["config"] == CONFIG_A and body["extras"] == {}
    assert body["origin"] == "config.yaml" and body["file_name"].endswith(".yaml")
    # a CLI run (no config.yaml): the configuration recorded in metrics.json
    (run / "config.yaml").unlink()
    metrics = json.loads((run / "metrics.json").read_text())
    (run / "metrics.json").write_text(
        json.dumps({**metrics, "config": {**CONFIG, "x_extra": 1}})
    )
    body = api.post(f"/api/runs/{run.name}/reopen").json()
    assert body["origin"] == "metrics.json" and body["extras"] == {"x_extra": 1}
    (run / "metrics.json").unlink()
    resp = api.post(f"/api/runs/{run.name}/reopen")
    assert resp.status_code == 404 and "configuration" in resp.json()["detail"]
    # a failed job without a folder reopens with the config it was submitted with
    jobs = web_paths.out_root / ".web_jobs"
    jobs.mkdir()
    (jobs / "j1.json").write_text(
        json.dumps({"id": "j1", "status": "failed", "config": CONFIG})
    )
    assert api.post("/api/runs/job-j1/reopen").json()["config"] == CONFIG
    # traversal never reaches a folder (normalised away: 405, or refused: 404)
    for bad in ("..%2fx", "%2e%2e", "link%2f..%2f..", "job-..%2fx"):
        assert api.post(f"/api/runs/{bad}/reopen").status_code in (404, 405)


def test_rerun_queues_same_config_and_video(api, web_paths, video_factory):
    clip = video_factory("v.mp4")
    video = api.post(
        "/api/videos", files={"file": ("v.mp4", clip.read_bytes(), "video/mp4")}
    ).json()
    two_runs(web_paths.out_root)
    run = web_paths.out_root / "20260101_000000_a.v1"
    (run / "web_run.json").write_text(
        json.dumps({"status": "done", "video_id": video["id"]})
    )
    resp = api.post(f"/api/runs/{run.name}/rerun")
    assert resp.status_code == 200, resp.text
    job = api.app.state.jobs.get(resp.json()["job_id"])
    assert job.video_id == video["id"]
    assert job.config["smoothing"] == CONFIG["smoothing"]
    assert job.config["detectors"] == CONFIG["detectors"]
    assert job.config["source"]["video"] == video["original"]
    assert job.config["output"]["tag"] == f"web-{job.id}"  # a fresh, unique tag


def test_rerun_without_library_video_checks_the_import_roots(
    api, web_paths, video_factory
):
    two_runs(web_paths.out_root)
    run = web_paths.out_root / "20260101_000001_b"
    outside = video_factory("outside.mp4")  # tmp/src: not an import root
    cfg = {**CONFIG, "source": {"video": str(outside)}}
    (run / "config.yaml").write_text(json.dumps(cfg))
    resp = api.post(f"/api/runs/{run.name}/rerun")
    assert resp.status_code == 409 and "import" in resp.json()["detail"]
    inside = video_factory(web_paths.data_roots[0] / "in.mp4")
    (run / "config.yaml").write_text(
        json.dumps({**CONFIG, "source": {"video": str(inside)}})
    )
    resp = api.post(f"/api/runs/{run.name}/rerun")
    assert resp.status_code == 200
    job = api.app.state.jobs.get(resp.json()["job_id"])
    assert job.video_id is None and job.config["source"]["video"] == str(inside)
    # a job entry without any configuration cannot be rerun
    jobs = web_paths.out_root / ".web_jobs"
    jobs.mkdir(exist_ok=True)  # the rerun above wrote its own job record
    (jobs / "j2.json").write_text(json.dumps({"id": "j2", "status": "failed"}))
    assert api.post("/api/runs/job-j2/rerun").status_code == 404
