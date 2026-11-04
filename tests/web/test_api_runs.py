"""Run endpoints: validation, queue, SSE, history files (fake worker)."""

import json
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from conftest import LOCAL
from fastapi.testclient import TestClient

from av_perception_lab.web import gpu
from av_perception_lab.web.app import create_app

FAKE = Path(__file__).with_name("fake_worker.py")
VALID = {
    "detectors": [{"type": "rfdetr"}],
    "depth": {"type": "da3"},
}


@pytest.fixture
def api(web_paths):
    # A context-managed client keeps one event loop alive (queue runner, SSE).
    app = create_app(web_paths, lambda p: [sys.executable, str(FAKE), str(p)])
    with TestClient(base_url=LOCAL, app=app) as client:
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
    with TestClient(base_url=LOCAL, app=app) as api:
        job_id = api.post("/api/runs", json={"config": VALID}).json()["job_id"]
        events = events_of(api, job_id)
        job = api.get(f"/api/jobs/{job_id}").json()
    assert events[-1]["event"] == "end" and events[-1]["status"] == "failed"
    assert job["status"] == "failed"
    assert job["error"] == (
        "no frames decoded from v.mp4 (is source.start past the end of the video?)"
    )


def test_run_lifecycle_history_and_files(api, web_paths, video_factory):
    clip = video_factory("v.mp4")
    video_id = api.post(
        "/api/videos", files={"file": ("v.mp4", clip.read_bytes(), "video/mp4")}
    ).json()["id"]
    job_id = api.post("/api/runs", json={"config": VALID, "video_id": video_id}).json()[
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


def test_cancel_endpoint_and_unknown_job(web_paths):
    # The first worker started sleeps (60 s), the second is an ok run.
    modes = iter(["sleep", "ok"])
    app = create_app(
        web_paths, lambda p: [sys.executable, str(FAKE), "--mode", next(modes), str(p)]
    )
    with TestClient(base_url=LOCAL, app=app) as api:
        _cancel_scenario(api)


def _cancel_scenario(api):
    job_id = api.post("/api/runs", json={"config": VALID}).json()["job_id"]
    second = api.post("/api/runs", json={"config": VALID}).json()["job_id"]
    assert api.get(f"/api/jobs/{second}").json()["position"] == 0
    job = api.app.state.jobs.get(job_id)
    deadline = time.monotonic() + 15
    while job.frames < 1:  # the worker really is in its sleep
        assert time.monotonic() < deadline, "the worker never started"
        time.sleep(0.02)
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


def _make_run(web_paths, name="run_a"):
    from run_folders import NEW_HEADER, make_run

    return make_run(web_paths.out_root, name, NEW_HEADER, [])


def test_size_route(api, web_paths):
    run = _make_run(web_paths)
    resp = api.get("/api/runs/run_a/size")
    assert resp.status_code == 200
    expected = sum(p.stat().st_size for p in run.rglob("*") if p.is_file())
    assert resp.json() == {"bytes": expected}
    assert api.get("/api/runs/nope/size").status_code == 404
    assert api.get("/api/runs/..%2Fx/size").status_code == 404


def test_delete_route_removes_run(api, web_paths):
    run = _make_run(web_paths)
    assert api.delete("/api/runs/run_a").status_code == 204
    assert not run.exists()
    assert api.get("/api/runs/run_a").status_code == 404
    assert "run_a" not in [r["id"] for r in api.get("/api/runs").json()]


def test_delete_traversal_404(api, web_paths):
    _make_run(web_paths)
    sibling = web_paths.out_root.parent / "keep.txt"
    sibling.write_text("x")
    (web_paths.out_root / ".web_jobs").mkdir(exist_ok=True)
    for rid in (".web_jobs", "%2e%2e", "job-%2e%2e"):
        assert api.delete(f"/api/runs/{rid}").status_code == 404
    # The client collapses a literal `..`: the request never reaches the route.
    for rid in ("..%2Fx", "x%2f..%2f..%2fkeep.txt"):
        assert api.delete(f"/api/runs/{rid}").status_code in (404, 405)
    assert sibling.exists() and (web_paths.out_root / ".web_jobs").exists()


def test_delete_active_run_409(web_paths):
    app = create_app(
        web_paths, lambda p: [sys.executable, str(FAKE), "--mode", "sleep", str(p)]
    )
    with TestClient(base_url=LOCAL, app=app) as api:
        job_id = api.post("/api/runs", json={"config": VALID}).json()["job_id"]
        job = api.app.state.jobs.get(job_id)
        deadline = time.monotonic() + 15
        while job.frames < 1:
            assert time.monotonic() < deadline, "the worker never started"
            time.sleep(0.02)
        resp = api.delete(f"/api/runs/{Path(job.run_dir).name}")
        assert resp.status_code == 409
        assert resp.json()["detail"] == "run is still queued or running"
        assert Path(job.run_dir).exists()
        api.delete(f"/api/jobs/{job_id}")


def test_delete_job_entry(web_paths):
    app = create_app(
        web_paths,
        lambda p: [sys.executable, str(FAKE), "--mode", "crash_early", str(p)],
    )
    with TestClient(base_url=LOCAL, app=app) as api:
        job_id = api.post("/api/runs", json={"config": VALID}).json()["job_id"]
        events_of(api, job_id)
        entry = f"job-{job_id}"
        assert entry in [r["id"] for r in api.get("/api/runs").json()]
        assert api.delete(f"/api/runs/{entry}").status_code == 204
        assert entry not in [r["id"] for r in api.get("/api/runs").json()]


def test_delete_route_refuses_non_run_folder(api, web_paths):
    folder = web_paths.out_root / "library"
    folder.mkdir(parents=True)
    (folder / "x.yaml").write_text("a: 1\n")
    assert api.delete("/api/runs/library").status_code == 404
    assert api.get("/api/runs/library/size").status_code == 404
    assert folder.exists()


def test_delete_route_reports_os_error(api, web_paths, monkeypatch):
    _make_run(web_paths)

    def boom(*_a, **_k):
        raise PermissionError("denied")

    monkeypatch.setattr("av_perception_lab.web.history.delete_run", boom)
    resp = api.delete("/api/runs/run_a")
    assert resp.status_code == 500
    assert resp.json()["detail"].startswith("could not delete: ")
