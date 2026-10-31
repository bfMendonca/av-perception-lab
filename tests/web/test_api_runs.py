"""Run endpoints: validation, queue, SSE, history files (fake worker)."""

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from det_depth_distance.web import gpu
from det_depth_distance.web.app import create_app

FAKE = Path(__file__).with_name("fake_worker.py")
VALID = {
    "detectors": [{"type": "rfdetr"}],
    "depth": {"type": "da3"},
}


@pytest.fixture
def api(web_paths):
    # A context-managed client keeps one event loop alive (queue runner, SSE).
    app = create_app(web_paths, lambda p: [sys.executable, str(FAKE), str(p)])
    with TestClient(app) as client:
        yield client


def events_of(api, job_id) -> list[dict]:
    with api.stream("GET", f"/api/jobs/{job_id}/events") as resp:
        assert resp.headers["content-type"].startswith("text/event-stream")
        return [
            json.loads(line[len("data: ") :])
            for line in resp.iter_lines()
            if line.startswith("data: ")
        ]


def test_post_runs_invalid_config_returns_422_with_paths(api):
    resp = api.post(
        "/api/runs",
        json={"config": {"detectors": [{"type": "rfdetr", "threshold": "x"}]}},
    )
    assert resp.status_code == 422
    errors = resp.json()["errors"]
    assert any(e["path"].startswith("detectors[0]") for e in errors)
    resp = api.post("/api/runs", json={"config": VALID, "video_id": "../x"})
    assert resp.status_code == 422
    assert resp.json()["errors"][0]["path"] == "video_id"


def test_start_past_end_surfaces_readable_error(web_paths):
    # The worker reports the ConfigError message; the job exposes it as `error`.
    script = web_paths.out_root.parent / "past_end.py"
    script.write_text(
        "import json,sys\n"
        "print(json.dumps({'event':'error','message':'no frames decoded from v.mp4 "
        "(is source.start past the end of the video?)'}),flush=True)\n"
        "sys.exit(2)\n"
    )
    app = create_app(web_paths, lambda p: [sys.executable, str(script)])
    with TestClient(app) as api:
        job_id = api.post("/api/runs", json={"config": VALID}).json()["job_id"]
        events = events_of(api, job_id)
        job = api.get(f"/api/jobs/{job_id}").json()
    assert events[-1]["event"] == "end" and events[-1]["status"] == "failed"
    assert job["status"] == "failed"
    assert job["error"] == (
        "no frames decoded from v.mp4 (is source.start past the end of the video?)"
    )


def test_run_lifecycle_history_and_files(api, web_paths):
    job_id = api.post("/api/runs", json={"config": VALID, "video_id": "abc"}).json()[
        "job_id"
    ]
    events = events_of(api, job_id)
    assert events[-1] == {"event": "end", "status": "done", "error": None}
    job = api.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "done" and job["frames"] == 2
    runs = api.get("/api/runs").json()
    assert [r["id"] for r in runs] == [job["run_id"]]
    assert runs[0]["status"] == "done" and runs[0]["complete"] is False  # no metrics
    assert api.get(f"/api/runs/{job['run_id']}").json()["id"] == job["run_id"]
    run_dir = web_paths.out_root / job["run_id"]
    (run_dir / "detections.csv").write_text("frame\n0\n")
    resp = api.get(f"/api/runs/{job['run_id']}/detections")
    assert resp.status_code == 200 and resp.text == "frame\n0\n"
    assert api.get(f"/api/runs/{job['run_id']}/video").status_code == 404
    assert api.get(f"/api/runs/{job['run_id']}/other").status_code == 404


def test_cancel_endpoint_and_unknown_job(api):
    job_id = api.post(
        "/api/runs", json={"config": {**VALID, "output": {"tag": "sleep"}}}
    ).json()["job_id"]
    second = api.post("/api/runs", json={"config": VALID}).json()["job_id"]
    assert api.get(f"/api/jobs/{second}").json()["position"] == 0
    assert api.delete(f"/api/jobs/{job_id}").json()["status"] == "cancelled"
    assert events_of(api, job_id)[-1]["status"] == "cancelled"
    assert events_of(api, second)[-1]["status"] == "done"
    assert api.get("/api/jobs/nope").status_code == 404
    assert api.delete("/api/jobs/nope").status_code == 404
    assert api.get("/api/jobs/nope/events").status_code == 404


def test_run_id_traversal_is_404(api, web_paths):
    web_paths.out_root.mkdir(parents=True)
    (web_paths.out_root.parent / "secret.csv").write_text("x")
    for rid in ("..", "%2e%2e", "..%2fsecret.csv", "x%2f..%2f..%2fsecret.csv"):
        assert api.get(f"/api/runs/{rid}").status_code == 404
        assert api.get(f"/api/runs/{rid}/detections").status_code == 404


def test_gpu_probe(api, monkeypatch):
    monkeypatch.setattr(gpu, "free_memory_mib", lambda: 4321)
    assert api.get("/api/gpu").json() == {"free_mib": 4321}
    monkeypatch.setattr(gpu, "free_memory_mib", lambda: None)
    assert api.get("/api/gpu").json() == {"free_mib": None}


def test_free_memory_probe_subprocess(monkeypatch):
    monkeypatch.setattr(gpu, "_PROBE", "print('noise'); print(2048)")
    assert gpu.free_memory_mib() == 2048
    monkeypatch.setattr(gpu, "_PROBE", "raise SystemExit(1)")
    assert gpu.free_memory_mib() is None
    monkeypatch.setattr(gpu, "_PROBE", "print('not a number')")
    assert gpu.free_memory_mib() is None
