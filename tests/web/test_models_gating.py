"""Run gating on model readiness and the path check of shipped configs
(task_0022)."""

import sys
import time
from pathlib import Path

import pytest
import yaml

pytest.importorskip("fastapi")

from conftest import LOCAL, REPO
from fastapi.testclient import TestClient
from model_fakes import fake_models, gate_config, wait_prepare  # noqa: F401

from det_depth_distance.core import requirements as rq
from det_depth_distance.web import schema, security
from det_depth_distance.web.app import create_app
from det_depth_distance.web.paths import WebPaths

FAKE_WORKER = Path(__file__).with_name("fake_worker.py")
SHIPPED = sorted(
    p for p in (REPO / "configs").glob("*.yaml") if p.name != "presets.yaml"
)


@pytest.fixture
def models_ready():
    """Overrides the autouse fixture of conftest: the real 409 gate."""


@pytest.fixture
def api(web_paths, fake_models):  # noqa: F811
    app = create_app(web_paths, lambda p: [sys.executable, str(FAKE_WORKER), str(p)])
    with TestClient(base_url=LOCAL, app=app) as client:
        yield client


def test_submit_run_409_when_model_not_ready(api, fake_models, tmp_path):  # noqa: F811
    cfg = gate_config(tmp_path, "a", "b")
    (fake_models.root / "gate").mkdir(parents=True)
    (fake_models.root / "gate/a.bin").write_bytes(b"x")
    resp = api.post("/api/runs", json={"config": cfg})
    assert resp.status_code == 409, resp.text
    pending = resp.json()["models_not_ready"]
    assert [(m["stage"], m["index"], m["state"]) for m in pending] == [
        ("detectors", 1, "missing")
    ]
    assert pending[0]["missing"] == ["weights"]
    assert str(fake_models.root) not in pending[0]["message"]
    assert api.get("/api/jobs").json() == []  # nothing was queued
    # An invalid config is still a 422 (validation comes first).
    bad = {**cfg, "detectors": [{"type": "gate", "blob": 3}]}
    assert api.post("/api/runs", json={"config": bad}).status_code == 422


def test_submit_run_409_while_preparing(api, fake_models, tmp_path):  # noqa: F811
    cfg = gate_config(tmp_path, ("a", "block"))
    job = api.post("/api/models/prepare", json={"config": cfg, "models": "all"}).json()
    assert fake_models.source.started.wait(5)
    resp = api.post("/api/runs", json={"config": cfg})
    assert resp.status_code == 409
    assert resp.json()["models_not_ready"][0]["state"] == "preparing"
    fake_models.source.release.set()
    assert wait_prepare(api, job["id"])["status"] == "done"
    assert api.post("/api/runs", json={"config": cfg}).status_code == 200


def test_submit_run_ok_when_all_ready(api, fake_models, tmp_path):  # noqa: F811
    cfg = gate_config(tmp_path, "a", "b")
    job = api.post("/api/models/prepare", json={"config": cfg, "models": "all"}).json()
    assert wait_prepare(api, job["id"])["status"] == "done"
    resp = api.post("/api/runs", json={"config": cfg})
    assert resp.status_code == 200, resp.text
    assert "job_id" in resp.json()


def test_rerun_is_gated_too(api, fake_models, web_paths, tmp_path):  # noqa: F811
    from run_folders import NEW_HEADER, make_run

    video = web_paths.data_roots[0] / "v.mp4"  # an import folder
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"")
    cfg = {**gate_config(tmp_path, "a"), "source": {"video": str(video)}}
    run = make_run(
        web_paths.out_root, "20260101_000000_v_ga", NEW_HEADER, [], config=cfg
    )
    resp = api.post(f"/api/runs/{run.name}/rerun")
    assert resp.status_code == 409, resp.text
    assert resp.json()["models_not_ready"][0]["stage"] == "detectors"
    job = api.post("/api/models/prepare", json={"config": cfg, "models": "all"}).json()
    assert wait_prepare(api, job["id"])["status"] == "done"
    assert api.post(f"/api/runs/{run.name}/rerun").status_code == 200


@pytest.mark.parametrize("path", SHIPPED, ids=lambda p: p.name)
def test_shipped_configs_pass_asset_root_check(path, tmp_path, monkeypatch):
    # The weights root counts as an asset root, and YOLO / Ultralytics SAM
    # bare weight names resolve under it as their adapter opens them.
    root = tmp_path / "weights"
    monkeypatch.setattr(rq, "_explicit_root", root)
    paths = WebPaths(tmp_path / "out", tmp_path / "lib", (), (REPO / "configs",))
    cfg = yaml.safe_load(path.read_text())
    monkeypatch.chdir(REPO)  # relative YAML paths start at the repo root
    assert security.path_problems(cfg, paths) == []
    found = dict(schema.config_paths(cfg))
    for dotted, value in found.items():
        if dotted.endswith(".weights") and value.suffix == ".pt":
            assert value == root / "ultralytics" / value.name, dotted


def test_ultralytics_weights_outside_asset_roots_still_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(rq, "_explicit_root", tmp_path / "weights")
    paths = WebPaths(tmp_path / "out", tmp_path / "lib", (), (REPO / "configs",))
    cfg = {
        "detectors": [{"type": "yolo", "weights": "/opt/models/yolo26n.pt"}],
        "depth": {"type": "da3"},
        "refiners": [{"type": "ultralytics_sam", "weights": "mobile_sam.pt"}],
    }
    problems = security.path_problems(cfg, paths)
    assert [p["path"] for p in problems] == ["detectors[0].weights"]
    assert "--asset-root" in problems[0]["message"]


def test_submit_run_409_for_unknown_hf_repo(api, fake_models, tmp_path):  # noqa: F811
    cfg = {
        **gate_config(tmp_path, "a"),
        "semantic": [{"type": "segformer", "name": "s", "model": "someone/seg-fork"}],
    }
    (fake_models.root / "gate").mkdir(parents=True)
    (fake_models.root / "gate/a.bin").write_bytes(b"x")
    resp = api.post("/api/runs", json={"config": cfg})
    assert resp.status_code == 409
    [row] = resp.json()["models_not_ready"]
    assert (row["stage"], row["index"]) == ("semantic", 0)
    assert row["message"] == "unknown source: prepare it from the CLI (--prepare)"


def test_run_worker_checks_the_same_weights_root_as_the_409_gate(
    web_paths,
    fake_models,  # noqa: F811
    tmp_path,
    monkeypatch,
):
    """The server's root is not the default (and the server's own env names
    another): a run that passes the 409 gate sees the same root in the worker
    (DDD_WEIGHTS_ROOT), so the CLI gate finds the same prepared files."""
    monkeypatch.setenv("DDD_WEIGHTS_ROOT", str(tmp_path / "stale"))
    seen = tmp_path / "seen.txt"
    code = (
        "import sys; from pathlib import Path; "
        "from det_depth_distance.core.requirements import weights_root; "
        "r = weights_root(); "
        f"Path({str(seen)!r}).write_text(f'{{r}}|{{(r / \"gate/a.bin\").is_file()}}')"
    )
    app = create_app(web_paths, lambda p: [sys.executable, "-c", code, str(p)])
    with TestClient(base_url=LOCAL, app=app) as api:
        cfg = gate_config(tmp_path, "a")
        job = api.post("/api/models/prepare", json={"config": cfg, "models": "all"})
        assert wait_prepare(api, job.json()["id"])["status"] == "done"
        resp = api.post("/api/runs", json={"config": cfg})
        assert resp.status_code == 200, resp.text
        deadline = time.monotonic() + 10
        while not seen.exists():
            assert time.monotonic() < deadline, "the worker never ran"
            time.sleep(0.05)
    root, ready = seen.read_text().split("|")
    assert root == str(fake_models.root.resolve()) and ready == "True"
