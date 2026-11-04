"""Model status and prepare queue API (task_0022): fake components and
sources, no network, no model loaded."""

import json
import os
import subprocess
import sys
import threading
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

from av_perception_lab.web import security
from av_perception_lab.web.app import create_app
from av_perception_lab.web.paths import WebPaths

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
from av_perception_lab.web.app import create_app
from av_perception_lab.web.paths import WebPaths

tmp = Path(sys.argv[1])
app = create_app(WebPaths(tmp / "out", tmp / "lib", (tmp / "data",)))
out = {}
with TestClient(base_url="http://127.0.0.1", app=app) as client:
    for path in sorted(Path("configs").glob("*.yaml")):
        if path.name == "presets.yaml":  # the presets manifest, not a run config
            continue
        cfg = yaml.safe_load(path.read_text())
        resp = client.post("/api/models/status", json={"config": cfg})
        out[path.name] = [resp.status_code, resp.json()]
print(json.dumps({"torch": "torch" in sys.modules, "configs": out}))
"""


def test_status_for_fresh_config_lists_missing_models_without_torch(tmp_path):
    # A fresh clone: empty weights root, every shipped config, in a process
    # that never imports torch.
    env = {**os.environ, "AVPL_WEIGHTS_ROOT": str(tmp_path / "weights")}
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
    shipped = [p for p in (REPO / "configs").glob("*.yaml") if p.name != "presets.yaml"]
    assert len(result["configs"]) == len(shipped)
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
                "source",
                "license",
                "target",
                "reason",
                "preparable",
                "packages",
            }
            # Absolute paths of the server never reach the client.
            for key in ("message", "source", "license", "target"):
                assert str(tmp_path) not in (row[key] or ""), (name, key, row)


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
    from av_perception_lab.core.provision import sources

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
    from av_perception_lab.core import requirements as rq
    from av_perception_lab.web import server

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
    # AVPL_WEIGHTS_ROOT is honoured (and checked) when the flag is absent.
    monkeypatch.setattr(rq, "_explicit_root", None)
    monkeypatch.setenv("AVPL_WEIGHTS_ROOT", str(tmp_path / "lib"))
    with pytest.raises(SystemExit):
        server.main(argv)
    assert not ran
    monkeypatch.setenv("AVPL_WEIGHTS_ROOT", str(tmp_path / "w-env"))
    server.main(argv)
    assert len(ran) == 1
    server.main([*argv, "--weights-root", str(tmp_path / "w-flag")])
    assert rq.weights_root() == tmp_path / "w-flag"
    assert len(ran) == 2


# -- fix round 1: unknown sources (I1) and redaction (I2) -----------------------

UNKNOWN_CASES = {
    "depth": (
        {"depth": {"type": "da3", "model": "someone/da3metric-fork"}},
        ("depth", None),
    ),
    "hf_sam": (
        {"refiners": [{"type": "sam2", "model": "someone/sam-fork"}]},
        ("refiners", 0),
    ),
    "segformer": (
        {"semantic": [{"type": "segformer", "name": "s", "model": "someone/seg-fork"}]},
        ("semantic", 0),
    ),
}


@pytest.fixture
def hf_calls(monkeypatch):
    """Hugging Face source stand-in: records the repo, downloads nothing."""
    from av_perception_lab.core.provision import sources

    calls = []

    def fake(self, asset, root, progress, cancel):
        calls.append(asset.repo)
        raise sources.ProvisionError(f"offline: cannot download {asset.repo}")

    monkeypatch.setattr(sources.HuggingFaceSource, "prepare", fake)
    return calls


@pytest.mark.parametrize("case", sorted(UNKNOWN_CASES))
def test_unknown_hf_repo_is_refused_by_the_web_and_preparable_by_the_cli(
    case,
    api,
    fake_models,  # noqa: F811
    tmp_path,
    hf_calls,
):
    from av_perception_lab.core.provision import prepare
    from av_perception_lab.web import models

    extra, (stage, index) = UNKNOWN_CASES[case]
    cfg = {**gate_config(tmp_path, "a"), **extra}
    (fake_models.root / "gate").mkdir(parents=True)
    (fake_models.root / "gate/a.bin").write_bytes(b"x")
    row = by_key(status_of(api, cfg))[(stage, index)]
    assert row["state"] == "missing"
    assert row["message"] == "unknown source: prepare it from the CLI (--prepare)"
    # Named explicitly: 422; nothing queued.
    resp = api.post(
        "/api/models/prepare",
        json={"config": cfg, "models": [{"stage": stage, "index": index}]},
    )
    assert resp.status_code == 422
    assert (
        "unknown source: prepare it from the CLI (--prepare)" in resp.json()["detail"]
    )
    # "all": the known models are prepared, the unknown one is reported.
    job = api.post("/api/models/prepare", json={"config": cfg, "models": "all"}).json()
    assert [(s["stage"], s["index"]) for s in job["skipped"]] == [(stage, index)]
    assert job["skipped"][0]["message"].startswith("unknown source")
    assert (stage, index) not in {(m["stage"], m["index"]) for m in job["models"]}
    assert wait_prepare(api, job["id"])["status"] == "done"
    assert hf_calls == []  # the web never asked the hub for it
    # The engine fails closed (the web's call) and never asks the hub...
    rows = {r.key: r for r in models.config_rows(cfg)}
    closed = prepare(
        rows[(stage, index)].req, fake_models.root, lambda *a: None, threading.Event()
    )
    assert closed.state == "failed" and "unknown source" in closed.message
    assert hf_calls == []
    # ...while the CLI path (allow_unknown) prepares any repo the user typed.
    status = prepare(
        rows[(stage, index)].req,
        fake_models.root,
        lambda *a: None,
        threading.Event(),
        allow_unknown=True,
    )
    assert hf_calls == [extra_repo(extra)]
    assert status.state == "failed"


def extra_repo(extra) -> str:
    spec = next(iter(extra.values()))
    spec = spec[0] if isinstance(spec, list) else spec
    return spec["model"]


def test_known_hf_repo_is_prepared_by_the_web(api, fake_models, tmp_path, hf_calls):  # noqa: F811
    cfg = {
        **gate_config(tmp_path, "a"),
        "refiners": [{"type": "sam2"}],  # facebook/sam2.1-hiera-large (HF_FACTS)
    }
    row = by_key(status_of(api, cfg))[("refiners", 0)]
    assert row["state"] == "missing" and not row["message"].startswith("unknown")
    job = api.post(
        "/api/models/prepare",
        json={"config": cfg, "models": [{"stage": "refiners", "index": 0}]},
    ).json()
    done = wait_prepare(api, job["id"])
    assert hf_calls == ["facebook/sam2.1-hiera-large"]
    assert done["status"] == "failed" and "offline" in done["error"]
    assert job["skipped"] == []


def test_redact_relative_root_and_sibling_folders(tmp_path, monkeypatch):
    from av_perception_lab.web.models import redact

    monkeypatch.chdir(tmp_path)
    abs_root = tmp_path / "weights"
    rel = Path("weights")  # `--weights-root weights`, as in the README
    assert redact(f"weights: {abs_root}/ultralytics/x.pt not found", rel) == (
        "weights: <weights root>/ultralytics/x.pt not found"
    )
    assert redact("weights: offline or unreachable", rel) == (
        "weights: offline or unreachable"
    )
    text = (
        f"a {abs_root}-old/x, b {abs_root}/y, c '{abs_root}', d {abs_root}. "
        f"e {abs_root}.bak f {tmp_path}/weightsX"
    )
    assert redact(text, abs_root) == (
        f"a {abs_root}-old/x, b <weights root>/y, c '<weights root>', "
        f"d <weights root>. e {abs_root}.bak f {tmp_path}/weightsX"
    )
    # A symlinked root is redacted in both spellings.
    link = tmp_path / "link"
    link.symlink_to(abs_root, target_is_directory=True)
    abs_root.mkdir()
    assert redact(f"{link}/a {abs_root}/b", link) == "<weights root>/a <weights root>/b"


# -- display fields and actions (task_0023 fix round) ------------------------------


def test_status_rows_carry_source_license_target_and_action(api, fake_models, tmp_path):  # noqa: F811
    cfg = {
        **gate_config(tmp_path, "a"),
        "refiners": [{"type": "sam2"}, {"type": "sam2", "model": "someone/else"}],
    }
    rows = by_key(status_of(api, cfg))
    gate = rows[("detectors", 0)]
    assert gate["source"] == "GateAsset"  # unregistered kind: the class name
    assert gate["target"] == "gate/a.bin"
    assert gate["license"] is None
    assert gate["reason"] is None and gate["preparable"] is True
    known = rows[("refiners", 0)]
    assert known["source"] == "Hugging Face · facebook/sam2.1-hiera-large"
    assert known["license"]  # declared by the adapter's table
    assert known["preparable"] is True
    unknown = rows[("refiners", 1)]
    assert unknown["reason"] == "unknown_source" and unknown["preparable"] is False
    assert unknown["source"] == "Hugging Face · someone/else"
    ready = rows[("depth", None)]
    assert ready["state"] == "ready" and ready["preparable"] is False
    root = str(fake_models.root.resolve())
    for row in rows.values():
        for key in ("source", "license", "target", "message"):
            value = row[key] or ""
            assert root not in value and str(tmp_path) not in value
            assert not value.startswith("/")


def test_declaration_view_redacts_and_classifies(tmp_path):
    from av_perception_lab.core import requirements as rq
    from av_perception_lab.web import models

    root = tmp_path / "weights"
    req = rq.Requirements(
        assets=(
            rq.UserProvided(
                key="weights",
                target="x/custom.pt",
                option="weights",
                hint=f"provide this file at {root / 'x/custom.pt'}",
                license="custom",
            ),
            rq.UrlAsset(
                key="w2",
                target="x/w2.pt",
                url="https://example.invalid/a/b.pt",
                size_bytes=1,
            ),
        ),
        code=(
            rq.CodeRepo(
                name="tool", url="https://example.invalid/tool.git", commit="a" * 40
            ),
        ),
    )
    view = models.declaration_view(req, root)
    assert view["source"] == (
        "provide this file at <weights root>/x/custom.pt; example.invalid; "
        "git · example.invalid/tool.git"
    )
    assert view["target"] == "x/custom.pt; x/w2.pt; _code/tool"
    assert view["license"] == "custom"
    packages = {"state": "missing", "missing": ["package:nopkg_xyz"]}
    assert models.with_action(dict(packages), False)["reason"] == "packages_only"
    assert models.with_action(dict(packages), False)["preparable"] is False
    assert (
        models.with_action({"state": "missing", "missing": ["weights"]}, False)[
            "preparable"
        ]
        is True
    )
    assert (
        models.with_action({"state": "conflict", "missing": ["weights"]}, False)[
            "preparable"
        ]
        is False
    )


def test_prepare_listing_returns_recent_jobs(api, fake_models, tmp_path):  # noqa: F811
    assert api.get("/api/models/prepare").json() == []
    cfg = gate_config(tmp_path, ("a", "block"))
    job = api.post("/api/models/prepare", json={"config": cfg, "models": "all"}).json()
    assert fake_models.source.started.wait(5)
    listed = api.get("/api/models/prepare").json()
    assert [j["id"] for j in listed] == [job["id"]]
    assert listed[0]["status"] == "running"
    assert listed[0]["models"][0]["stage"] == "detectors"
    rows = by_key(status_of(api, cfg))
    assert rows[("detectors", 0)]["state"] == "preparing"
    assert rows[("detectors", 0)]["preparable"] is False
    api.delete(f"/api/models/prepare/{job['id']}")
    assert wait_prepare(api, job["id"])["status"] == "cancelled"


# --- final wave (task_0026) -------------------------------------------------


def test_user_provided_only_model_is_not_preparable_and_skipped_by_all(
    api,
    fake_models,  # noqa: F811
    tmp_path,
):
    """A model whose only missing file is a manual download is never offered
    as Prepare: reason `user_provided`, the row's message carries the hint,
    Prepare all leaves it out (listed in `skipped`), naming it is a 422."""
    root = fake_models.root
    (root / "gate").mkdir(parents=True)
    (root / "gate/m.bin").write_bytes(b"x" * 8)  # its downloadable file is there
    cfg = gate_config(tmp_path, "a", {"blob": "m", "manual": "big"})
    rows = by_key(status_of(api, cfg))
    manual = rows[("detectors", 1)]
    assert manual["state"] == "missing" and manual["preparable"] is False
    assert manual["reason"] == "user_provided"
    assert "manual download: save big.pth by hand" in manual["message"]
    assert rows[("detectors", 0)]["preparable"] is True
    job = api.post("/api/models/prepare", json={"config": cfg, "models": "all"}).json()
    assert [(s["stage"], s["index"]) for s in job["skipped"]] == [("detectors", 1)]
    assert "save big.pth by hand" in job["skipped"][0]["message"]
    chosen = [(m["stage"], m["index"]) for m in job["models"]]
    assert ("detectors", 0) in chosen and ("detectors", 1) not in chosen
    assert wait_prepare(api, job["id"])["status"] == "done"
    named = api.post(
        "/api/models/prepare",
        json={"config": cfg, "models": [{"stage": "detectors", "index": 1}]},
    )
    assert named.status_code == 422
    assert "save big.pth by hand" in named.json()["detail"]
    assert ("gate/big.pth", CODE_URL) not in fake_models.source.calls


def test_prepare_that_fetches_all_it_can_ends_done_with_the_manual_hint(
    api,
    fake_models,  # noqa: F811
    tmp_path,
):
    cfg = gate_config(tmp_path, {"blob": "m", "manual": "big"})
    row = by_key(status_of(api, cfg))[("detectors", 0)]
    assert row["preparable"] is True and row["reason"] is None  # gate/m.bin first
    job = api.post("/api/models/prepare", json={"config": cfg, "models": "all"}).json()
    job = wait_prepare(api, job["id"])
    assert job["status"] == "done", job  # no "Prepare failed" for a manual step
    (model,) = [m for m in job["models"] if m["stage"] == "detectors"]
    assert model["state"] == "missing"
    assert "provide this file: manual download: save big.pth" in model["message"]
    after = by_key(status_of(api, cfg))[("detectors", 0)]
    assert after["reason"] == "user_provided" and after["preparable"] is False


def test_engine_rechecks_unknown_sources_when_the_job_runs(
    api,
    fake_models,  # noqa: F811
    tmp_path,
):
    """Ready (so queued) at submit, missing when the job reaches it: the web
    engine still never fetches a source the adapter does not vouch for."""
    root = fake_models.root
    (root / "gate").mkdir(parents=True)
    (root / "gate/u.bin").write_bytes(b"x" * 8)
    cfg = gate_config(tmp_path, ("a", "block"), {"blob": "u", "known": False})
    job = api.post("/api/models/prepare", json={"config": cfg, "models": "all"}).json()
    assert job["skipped"] == []
    assert [m["name"] for m in job["models"][:2]] == ["ga", "gu"]
    assert fake_models.source.started.wait(5)
    (root / "gate/u.bin").unlink()
    fake_models.source.release.set()
    job = wait_prepare(api, job["id"])
    assert job["status"] == "failed"
    unknown = job["models"][1]
    assert unknown["state"] == "failed" and "unknown source" in unknown["message"]
    assert [t for t, _ in fake_models.source.calls] == ["gate/a.bin"]
    assert not (root / "gate/u.bin").exists()


def test_web_cannot_prepare_hostile_weights_or_hf_cache_targets(
    api,
    fake_models,  # noqa: F811
    tmp_path,
    hf_calls,
    monkeypatch,
):
    """A release name or a known repo at a client-chosen place inside the
    weights root is not vouched for: shown as unknown source, refused, and
    nothing is written."""
    from av_perception_lab.core.provision import sources

    def no_network(*args, **kwargs):
        raise AssertionError("the web downloaded a client-chosen target")

    monkeypatch.setattr(sources, "_open", no_network)
    root = fake_models.root
    hostile = {
        "detectors": [
            {
                "type": "yolo",
                "name": "evil",
                "weights": str(root / "_code/TwinLiteNetPlus/yolo26n.pt"),
            }
        ],
        "depth": {"type": "da3", "hf_cache": str(root / "_code/cache")},
        "camera": {"estimate": None},
    }
    rows = by_key(status_of(api, hostile))
    for key in (("detectors", 0), ("depth", None)):
        assert rows[key]["reason"] == "unknown_source", rows[key]
        assert rows[key]["preparable"] is False
        resp = api.post(
            "/api/models/prepare",
            json={"config": hostile, "models": [{"stage": key[0], "index": key[1]}]},
        )
        assert resp.status_code == 422 and "unknown source" in resp.json()["detail"]
    resp = api.post("/api/models/prepare", json={"config": hostile, "models": "all"})
    assert resp.status_code == 422
    assert hf_calls == []
    assert not (root / "_code").exists()


def test_size_mismatched_file_survives_a_web_prepare(
    api,
    fake_models,  # noqa: F811
    tmp_path,
    monkeypatch,
):
    from av_perception_lab.core.provision import sources

    monkeypatch.setattr(
        sources, "_open", lambda *a, **k: pytest.fail("downloaded over a user file")
    )
    root = fake_models.root
    mine = root / "ultralytics/yolo26n.pt"
    mine.parent.mkdir(parents=True)
    mine.write_bytes(b"my fine-tuned checkpoint")
    cfg = {
        "detectors": [{"type": "yolo", "name": "mine", "weights": "yolo26n.pt"}],
        "depth": {"type": "da3", "hf_cache": str(tmp_path / "elsewhere")},
        "camera": {"estimate": None},
    }
    row = by_key(status_of(api, cfg))[("detectors", 0)]
    assert row["state"] == "failed" and row["preparable"] is False
    assert "exists with a different size at ultralytics/yolo26n.pt" in row["message"]
    job = api.post("/api/models/prepare", json={"config": cfg, "models": "all"}).json()
    job = wait_prepare(api, job["id"])
    assert job["status"] == "failed" and "move or delete it" in job["error"]
    assert mine.read_bytes() == b"my fine-tuned checkpoint"


def test_prepare_422_replies_are_redacted(
    api,
    fake_models,  # noqa: F811
    tmp_path,
    monkeypatch,
):
    from av_perception_lab.core.options import ConfigError
    from av_perception_lab.web import models

    root = fake_models.root

    def leaky(data):
        raise ConfigError(
            f"model: x is not in {root}/hf (Authorization: Bearer abcdefghijkl)"
        )

    monkeypatch.setattr(models, "config_rows", leaky)
    resp = api.post(
        "/api/models/prepare",
        json={"config": gate_config(tmp_path, "a"), "models": "all"},
    )
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert str(root) not in detail and "<weights root>/hf" in detail
    assert "abcdefghijkl" not in detail


@pytest.mark.parametrize("bad", ["/", "home", "repo-parent"])
def test_server_refuses_an_unsafe_weights_root(monkeypatch, tmp_path, capsys, bad):
    from av_perception_lab.core import requirements as rq
    from av_perception_lab.web import server

    root = {"/": Path("/"), "home": Path.home(), "repo-parent": REPO.parent}[bad]
    monkeypatch.setattr(rq, "_explicit_root", None)
    ran = []
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: ran.append(a))
    argv = ["--out-root", str(tmp_path / "out"), "--library-root", str(tmp_path / "l")]
    with pytest.raises(SystemExit) as exc:
        server.main([*argv, "--weights-root", str(root)])
    assert exc.value.code == 2 and not ran
    assert "use a dedicated folder" in capsys.readouterr().err
    monkeypatch.setattr(rq, "_explicit_root", root)
    with pytest.raises(ValueError, match="dedicated folder"):
        create_app(WebPaths(tmp_path / "out", tmp_path / "l", ()))
