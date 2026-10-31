"""Model status and prepare queue API (task_0022): fake components and
sources, no network, no model loaded."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from conftest import LOCAL, REPO
from fastapi.testclient import TestClient
from model_fakes import (  # noqa: F401 - fake_models is a fixture
    CODE_URL,
    fake_models,
    gate_config,
    wait_prepare,
)

from det_depth_distance.web import security
from det_depth_distance.web.app import create_app
from det_depth_distance.web.paths import WebPaths

FAKE_WORKER = Path(__file__).with_name("fake_worker.py")


@pytest.fixture
def api(web_paths, fake_models):  # noqa: F811
    app = create_app(web_paths, lambda p: [sys.executable, str(FAKE_WORKER), str(p)])
    with TestClient(base_url=LOCAL, app=app) as client:
        yield client


def events_of(api, job_id) -> list[dict]:
    with api.stream("GET", f"/api/models/prepare/{job_id}/events") as resp:
        assert resp.headers["content-type"].startswith("text/event-stream")
        return [
            json.loads(line[len("data: ") :])
            for line in resp.iter_lines()
            if line.startswith("data: ")
        ]


def status_of(api, cfg) -> dict:
    resp = api.post("/api/models/status", json={"config": cfg})
    assert resp.status_code == 200, resp.text
    return resp.json()


def by_key(report) -> dict:
    return {(m["stage"], m["index"]): m for m in report["models"]}


def wait_for(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached")
        time.sleep(0.01)


# -- status ---------------------------------------------------------------------


_FRESH_SCRIPT = r"""
import json, sys
from pathlib import Path
import yaml
from fastapi.testclient import TestClient
from det_depth_distance.web.app import create_app
from det_depth_distance.web.paths import WebPaths

tmp = Path(sys.argv[1])
app = create_app(WebPaths(tmp / "out", tmp / "lib", (tmp / "data",)))
out = {}
with TestClient(base_url="http://127.0.0.1", app=app) as client:
    for path in sorted(Path("configs").glob("*.yaml")):
        cfg = yaml.safe_load(path.read_text())
        resp = client.post("/api/models/status", json={"config": cfg})
        out[path.name] = [resp.status_code, resp.json()]
print(json.dumps({"torch": "torch" in sys.modules, "configs": out}))
"""


def test_status_for_fresh_config_lists_missing_models_without_torch(tmp_path):
    # A fresh clone: empty weights root, every shipped config, in a process
    # that never imports torch.
    env = {**os.environ, "DDD_WEIGHTS_ROOT": str(tmp_path / "weights")}
    out = subprocess.run(
        [sys.executable, "-c", _FRESH_SCRIPT, str(tmp_path)],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    )
    result = json.loads(out.stdout.strip().splitlines()[-1])
    assert result["torch"] is False
    assert len(result["configs"]) == len(list((REPO / "configs").glob("*.yaml")))
    for name, (code, report) in result["configs"].items():
        assert code == 200, (name, report)
        rows = by_key(report)
        assert rows[("depth", None)]["state"] == "missing", name
        assert rows[("depth", None)]["layers"]["assets"] == "missing", name
        assert rows[("detectors", 0)]["state"] != "ready", name
        assert report["total_size_bytes"] > 0, name
        for row in report["models"]:
            assert set(row) == {
                "stage",
                "index",
                "type",
                "name",
                "state",
                "layers",
                "missing",
                "size_bytes",
                "message",
            }
            # Absolute paths of the server never reach the client.
            assert str(tmp_path) not in (row["message"] or ""), (name, row)


def test_status_ready_after_fake_asset_present(api, fake_models, tmp_path):  # noqa: F811
    cfg = gate_config(tmp_path, "a", "b")
    report = status_of(api, cfg)
    rows = by_key(report)
    assert [m["stage"] for m in report["models"]] == ["detectors", "detectors", "depth"]
    assert rows[("detectors", 0)]["state"] == "missing"
    assert rows[("detectors", 0)]["missing"] == ["weights"]
    assert rows[("detectors", 0)]["size_bytes"] == 8
    assert rows[("depth", None)]["state"] == "ready"  # packages only
    assert report["total_size_bytes"] == 16
    # The message names the weights root, never its absolute path.
    message = rows[("detectors", 0)]["message"]
    assert "<weights root>/gate/a.bin" in message
    assert str(fake_models.root) not in message
    (fake_models.root / "gate").mkdir(parents=True)
    (fake_models.root / "gate/a.bin").write_bytes(b"x")
    rows = by_key(status_of(api, cfg))
    assert rows[("detectors", 0)]["state"] == "ready"
    assert rows[("detectors", 1)]["state"] == "missing"
    # An invalid config is a 422 with dotted-path errors, like validate.
    bad = {**cfg, "detectors": [{"type": "gate", "blob": 3}]}
    resp = api.post("/api/models/status", json={"config": bad})
    assert resp.status_code == 422
    assert resp.json()["errors"][0]["path"].startswith("detectors[0]")


# -- prepare queue --------------------------------------------------------------


def test_prepare_all_runs_one_model_at_a_time(api, fake_models, tmp_path):  # noqa: F811
    cfg = gate_config(tmp_path, "a", "b")
    resp = api.post("/api/models/prepare", json={"config": cfg, "models": "all"})
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["id"]
    events = events_of(api, job_id)
    assert events[0]["event"] == "queued" and events[1]["event"] == "running"
    assert events[-1] == {"event": "end", "status": "done", "error": None}
    models = [
        (e["stage"], e["index"], e["state"]) for e in events if e["event"] == "model"
    ]
    assert models == [
        ("detectors", 0, "preparing"),
        ("detectors", 0, "ready"),
        ("detectors", 1, "preparing"),
        ("detectors", 1, "ready"),
        ("depth", None, "preparing"),
        ("depth", None, "ready"),
    ]
    progress = [e for e in events if e["event"] == "progress"]
    assert progress and {e["item"] for e in progress} == {"weights"}
    assert progress[-1]["done"] == progress[-1]["total"] == 8
    assert fake_models.source.peak == 1
    assert [t for t, _ in fake_models.source.calls] == ["gate/a.bin", "gate/b.bin"]
    job = api.get(f"/api/models/prepare/{job_id}").json()
    assert job["status"] == "done"
    assert [m["state"] for m in job["models"]] == ["ready", "ready", "ready"]
    assert all(m["state"] == "ready" for m in status_of(api, cfg)["models"])
    # A Prepare of ready models is a no-op: no source is called.
    again = api.post(
        "/api/models/prepare",
        json={"config": cfg, "models": [{"stage": "detectors", "index": 1}]},
    ).json()
    assert wait_prepare(api, again["id"])["status"] == "done"
    assert len(fake_models.source.calls) == 2


def test_second_prepare_waits_for_first(api, fake_models, tmp_path):  # noqa: F811
    cfg = gate_config(tmp_path, ("a", "block"), "b")
    first = api.post(
        "/api/models/prepare",
        json={"config": cfg, "models": [{"stage": "detectors", "index": 0}]},
    ).json()
    assert fake_models.source.started.wait(5)
    second = api.post(
        "/api/models/prepare",
        json={"config": cfg, "models": [{"stage": "detectors", "index": 1}]},
    ).json()
    assert second["status"] == "queued" and second["position"] == 0
    view = api.get(f"/api/models/prepare/{first['id']}").json()
    assert view["status"] == "running"
    assert view["models"][0]["state"] == "preparing"
    # The model being downloaded reads "preparing" in the status too.
    rows = by_key(status_of(api, cfg))
    assert rows[("detectors", 0)]["state"] == "preparing"
    assert rows[("detectors", 1)]["state"] == "missing"
    time.sleep(0.1)
    assert [t for t, _ in fake_models.source.calls] == ["gate/a.bin"]
    fake_models.source.release.set()
    assert wait_prepare(api, first["id"])["status"] == "done"
    assert wait_prepare(api, second["id"])["status"] == "done"
    assert [t for t, _ in fake_models.source.calls] == ["gate/a.bin", "gate/b.bin"]
    assert fake_models.source.peak == 1


def test_cancel_prepare_frees_queue_and_leaves_failed_not_ready(
    api,
    fake_models,  # noqa: F811
    tmp_path,
):
    cfg = gate_config(tmp_path, ("a", "block"), "b")
    first = api.post(
        "/api/models/prepare", json={"config": cfg, "models": "all"}
    ).json()
    assert fake_models.source.started.wait(5)
    second = api.post(
        "/api/models/prepare",
        json={"config": cfg, "models": [{"stage": "detectors", "index": 1}]},
    ).json()
    resp = api.delete(f"/api/models/prepare/{first['id']}")
    assert resp.status_code == 200 and resp.json()["cancel_requested"] is True
    job = wait_prepare(api, first["id"])
    assert job["status"] == "cancelled"
    states = [m["state"] for m in job["models"]]
    assert states[0] == "missing"  # what the engine left: not ready
    assert job["models"][0]["message"].startswith("cancelled")
    assert states[1:] == ["cancelled", "cancelled"]  # never started
    events = events_of(api, first["id"])
    assert events[-1]["event"] == "end" and events[-1]["status"] == "cancelled"
    # The queue is free: the next Prepare runs; the cancelled model is not ready
    # (its partial file never counts).
    assert wait_prepare(api, second["id"])["status"] == "done"
    rows = by_key(status_of(api, cfg))
    assert rows[("detectors", 0)]["state"] == "missing"
    assert (fake_models.root / "gate/a.bin.part").is_file()
    assert rows[("detectors", 1)]["state"] == "ready"
    # Cancelling a queued job ends it at once; an ended job stays as it was.
    fake_models.source.reset()
    blocker = api.post(
        "/api/models/prepare",
        json={"config": cfg, "models": [{"stage": "detectors", "index": 0}]},
    ).json()
    assert fake_models.source.started.wait(5)
    queued = api.post(
        "/api/models/prepare", json={"config": cfg, "models": "all"}
    ).json()
    assert (
        api.delete(f"/api/models/prepare/{queued['id']}").json()["status"]
        == "cancelled"
    )
    fake_models.source.release.set()
    assert wait_prepare(api, blocker["id"])["status"] == "done"
    assert api.delete(f"/api/models/prepare/{blocker['id']}").json()["status"] == "done"
    assert api.delete("/api/models/prepare/nope").status_code == 404


def test_failed_prepare_ends_failed_with_readable_reason(api, fake_models, tmp_path):  # noqa: F811
    cfg = gate_config(tmp_path, ("a", "fail"), "b")
    job_id = api.post(
        "/api/models/prepare", json={"config": cfg, "models": "all"}
    ).json()["id"]
    job = wait_prepare(api, job_id)
    assert job["status"] == "failed"
    assert "offline or unreachable" in job["error"]
    assert job["error"].startswith("detectors[0] ga:")
    assert [m["state"] for m in job["models"]] == ["failed", "ready", "ready"]
    assert job["models"][0]["message"] == (
        "weights: offline or unreachable: example.invalid"
    )
    # The queue keeps working after a failure.
    ok = api.post(
        "/api/models/prepare",
        json={"config": cfg, "models": [{"stage": "detectors", "index": 1}]},
    ).json()
    assert wait_prepare(api, ok["id"])["status"] == "done"


def test_prepare_unknown_model_key_is_422(api, fake_models, tmp_path):  # noqa: F811
    cfg = gate_config(tmp_path, "a")
    for models in (
        [{"stage": "detectors", "index": 5}],
        [{"stage": "depth", "index": 0}],
        [{"stage": "refiners", "index": 0}],
        [],
        "some",
    ):
        resp = api.post("/api/models/prepare", json={"config": cfg, "models": models})
        assert resp.status_code == 422, (models, resp.text)
    resp = api.post(
        "/api/models/prepare",
        json={"config": cfg, "models": [{"stage": "detectors", "index": 5}]},
    )
    assert "detectors[5] is not a model of this config" in resp.json()["detail"]
    # An invalid config is refused before anything is queued.
    bad = {**cfg, "depth": {"type": "nope"}}
    resp = api.post("/api/models/prepare", json={"config": bad, "models": "all"})
    assert resp.status_code == 422 and resp.json()["errors"][0]["path"] == "depth.type"
    assert api.get("/api/models/prepare/nope").status_code == 404
    assert fake_models.source.calls == []


def test_prepare_never_uses_client_supplied_urls(
    api,
    fake_models,  # noqa: F811
    tmp_path,
    monkeypatch,
):
    cfg = gate_config(tmp_path, "a")
    evil = "https://evil.example/x.bin"
    # Fields beyond the config and the model keys are refused.
    for body in (
        {"config": cfg, "models": "all", "url": evil},
        {"config": cfg, "models": "all", "target": "/etc/x"},
        {"config": cfg, "models": [{"stage": "detectors", "index": 0, "url": evil}]},
        {"config": cfg, "models": [{"stage": "detectors", "index": 0, "path": "/x"}]},
    ):
        resp = api.post("/api/models/prepare", json=body)
        assert resp.status_code == 422, body
    resp = api.post("/api/models/status", json={"config": cfg, "url": evil})
    assert resp.status_code == 422
    assert fake_models.source.calls == []
    # The source gets the URL and target of the adapter's declaration.
    job_id = api.post(
        "/api/models/prepare", json={"config": cfg, "models": "all"}
    ).json()["id"]
    assert wait_prepare(api, job_id)["status"] == "done"
    assert fake_models.source.calls == [("gate/a.bin", CODE_URL)]
    # A YOLO `weights` that looks like a URL is a path outside the weights
    # root: nothing is declared, so nothing is downloaded.
    from det_depth_distance.core.provision import sources

    def no_network(*args, **kwargs):
        raise AssertionError("a download was attempted")

    monkeypatch.setattr(sources, "_open", no_network)
    yolo = {
        **cfg,
        "detectors": [{"type": "yolo", "weights": "https://evil.example/yolo26n.pt"}],
    }
    rows = by_key(status_of(api, yolo))
    assert rows[("detectors", 0)]["layers"]["assets"] == "ready"  # none declared
    job_id = api.post(
        "/api/models/prepare", json={"config": yolo, "models": "all"}
    ).json()["id"]
    assert wait_prepare(api, job_id)["status"] == "done"


def test_prepare_does_not_block_gpu_job_queue(api, fake_models, tmp_path):  # noqa: F811
    cfg = gate_config(tmp_path, ("a", "block"))
    prep = api.post("/api/models/prepare", json={"config": cfg, "models": "all"}).json()
    assert fake_models.source.started.wait(5)
    # A run (fake worker; the gate is lifted by the autouse fixture) is
    # queued and finishes while the download is still blocked.
    run = api.post(
        "/api/runs",
        json={"config": {"detectors": [{"type": "rfdetr"}], "depth": {"type": "da3"}}},
    )
    assert run.status_code == 200, run.text
    job_id = run.json()["job_id"]
    deadline = time.monotonic() + 10
    while api.get(f"/api/jobs/{job_id}").json()["status"] not in ("done", "failed"):
        assert time.monotonic() < deadline
        time.sleep(0.02)
    assert api.get(f"/api/jobs/{job_id}").json()["status"] == "done"
    assert api.get(f"/api/models/prepare/{prep['id']}").json()["status"] == "running"
    # The event loop stays responsive while the engine runs in its thread.
    assert api.get("/api/catalog").status_code == 200
    fake_models.source.release.set()
    assert wait_prepare(api, prep["id"])["status"] == "done"


# -- weights root ---------------------------------------------------------------


def test_weights_root_containment_enforced(api, fake_models, web_paths, tmp_path):  # noqa: F811
    # A target that would leave the weights root fails, nothing is written.
    cfg = gate_config(tmp_path, "../../escape")
    rows = by_key(status_of(api, cfg))
    assert rows[("detectors", 0)]["state"] == "failed"
    assert rows[("detectors", 0)]["layers"]["assets"] == "invalid"
    job_id = api.post(
        "/api/models/prepare", json={"config": cfg, "models": "all"}
    ).json()["id"]
    job = wait_prepare(api, job_id)
    assert job["status"] == "failed" and "unsafe path" in job["error"]
    assert fake_models.source.calls == []
    assert not (tmp_path / "escape.bin").exists()
    # The weights root may not overlap a folder clients write.
    for bad in (
        web_paths.library_root,
        web_paths.library_root / "w",
        web_paths.out_root / "weights",
        tmp_path,  # contains both
    ):
        assert security.weights_root_problem(bad, web_paths) is not None, bad
    assert security.weights_root_problem(tmp_path / "weights", web_paths) is None


def test_create_app_and_server_refuse_overlapping_weights_root(monkeypatch, tmp_path):
    from det_depth_distance.core import requirements as rq
    from det_depth_distance.web import server

    paths = WebPaths(tmp_path / "out", tmp_path / "lib", ())
    monkeypatch.setattr(rq, "_explicit_root", tmp_path / "lib" / "weights")
    with pytest.raises(ValueError, match="overlaps"):
        create_app(paths)
    monkeypatch.setattr(rq, "_explicit_root", None)
    ran = []
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: ran.append(a))
    argv = [
        "--out-root",
        str(tmp_path / "out"),
        "--library-root",
        str(tmp_path / "lib"),
    ]
    with pytest.raises(SystemExit) as exc:
        server.main([*argv, "--weights-root", str(tmp_path / "out")])
    assert exc.value.code == 2 and not ran
    # DDD_WEIGHTS_ROOT is honoured (and checked) when the flag is absent.
    monkeypatch.setattr(rq, "_explicit_root", None)
    monkeypatch.setenv("DDD_WEIGHTS_ROOT", str(tmp_path / "lib"))
    with pytest.raises(SystemExit):
        server.main(argv)
    assert not ran
    monkeypatch.setenv("DDD_WEIGHTS_ROOT", str(tmp_path / "w-env"))
    server.main(argv)
    assert len(ran) == 1
    server.main([*argv, "--weights-root", str(tmp_path / "w-flag")])
    assert rq.weights_root() == tmp_path / "w-flag"
    assert len(ran) == 2
