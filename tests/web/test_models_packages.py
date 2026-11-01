"""Packages layer in the web API (task_0024): conflict state, Install
packages jobs, base stack command. Every `uv` call goes to the fake of
`tests/fake_uv.py`; no network, no real install, no model loaded."""

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tests/fake_uv.py

from conftest import LOCAL
from fake_uv import fake_uv, snapshot  # noqa: F401 - fake_uv is a fixture
from fastapi.testclient import TestClient
from model_fakes import fake_models, wait_prepare  # noqa: F401

from det_depth_distance.core import registry
from det_depth_distance.core import requirements as rq
from det_depth_distance.core.components import Detector, DetectorOptions
from det_depth_distance.core.requirements import Requirements
from det_depth_distance.web.app import create_app

FAKE_WORKER = Path(__file__).with_name("fake_worker.py")
REAL_PRESENT = rq._top_level_present
STACK = {"torch", "depth_anything_3"}  # DA3's packages count as installed


class PkgDetector(Detector):
    """A detector whose only requirement is its packages."""

    @dataclass(frozen=True, slots=True, kw_only=True)
    class Options(DetectorOptions):
        extra: str = "model-xyz"

    @classmethod
    def requirements(cls, options: Options) -> Requirements:
        package = "xyzlib" if options.extra == "model-xyz" else "fakebase_missing"
        return Requirements(packages=(package,), extra=options.extra)

    @property
    def class_names(self) -> list[str]:
        return ["car"]

    def _load(self, ctx) -> None:
        raise AssertionError("a model was loaded")

    def _predict(self, rgb):
        raise AssertionError("unreachable")


PkgDetector.type_name = "pkg"


def pkg_config(tmp_path: Path, extra: str = "model-xyz") -> dict:
    return {
        "detectors": [{"type": "pkg", "name": "p", "extra": extra}],
        "depth": {"type": "da3", "hf_cache": str(tmp_path / "elsewhere")},
        "camera": {"estimate": None},
    }


@pytest.fixture
def api(web_paths, fake_models, fake_uv, monkeypatch):  # noqa: F811
    monkeypatch.setattr(
        rq, "_top_level_present", lambda n: n in STACK or REAL_PRESENT(n)
    )
    registry.load_adapters()
    monkeypatch.setitem(registry._REGISTRY["detector"], "pkg", PkgDetector)
    app = create_app(web_paths, lambda p: [sys.executable, str(FAKE_WORKER), str(p)])
    with TestClient(base_url=LOCAL, app=app) as client:
        yield client


def status(api, cfg) -> dict:
    resp = api.post("/api/models/status", json={"config": cfg})
    assert resp.status_code == 200, resp.text
    return resp.json()


def detector_row(report) -> dict:
    return next(m for m in report["models"] if m["stage"] == "detectors")


def install(api, cfg):
    return api.post(
        "/api/models/prepare",
        json={
            "models": [{"stage": "detectors", "index": 0}],
            "config": cfg,
            "packages": True,
        },
    )


def events_of(api, job_id) -> list[dict]:
    with api.stream("GET", f"/api/models/prepare/{job_id}/events") as resp:
        return [
            json.loads(line[len("data: ") :])
            for line in resp.iter_lines()
            if line.startswith("data: ")
        ]


def test_conflict_state_reaches_the_api_and_ui_contract(api, fake_uv, tmp_path):  # noqa: F811
    fake_uv.mode("conflict")
    before = snapshot(fake_uv.envdir)
    report = status(api, pkg_config(tmp_path))
    row = detector_row(report)
    assert row["state"] == "conflict"
    assert "incompatible" in row["message"]
    assert "docs/model_environments.md" in row["message"]
    assert row["packages"]["action"] == "conflict"
    assert row["packages"]["extra"] == "model-xyz"
    assert row["preparable"] is False
    assert row["reason"] == "packages_only"
    assert str(tmp_path) not in json.dumps(report)  # project-relative, redacted
    assert report["restart_required"] is False
    # A config in conflict is not ready: a run would be refused.
    assert row["state"] != "ready"
    # Installing it from the web: the job's fresh dry-run fails, nothing changes.
    resp = install(api, pkg_config(tmp_path))
    assert resp.status_code == 200, resp.text
    job = wait_prepare(api, resp.json()["id"])
    assert job["status"] == "failed"
    assert job["models"][0]["state"] == "conflict"
    assert job["models"][0]["installed"] is False
    assert snapshot(fake_uv.envdir) == before
    assert all("--dry-run" in c["args"] for c in fake_uv.calls())


def test_clean_dry_run_offers_install_packages(api, fake_uv, tmp_path):  # noqa: F811
    row = detector_row(status(api, pkg_config(tmp_path)))
    assert row["state"] == "missing"
    assert row["packages"] == {
        "action": "install",
        "extra": "model-xyz",
        "command": "uv --color never sync --locked --inexact "
        "--extra infer --extra model-xyz --extra web",
        "message": "would install xyzlib==1.0",
    }
    assert row["preparable"] is False  # Prepare (weights) cannot fix it
    (call,) = fake_uv.calls()
    assert "--dry-run" in call["args"]


def test_install_packages_job_streams_output_and_requires_restart(
    api,
    fake_uv,  # noqa: F811
    tmp_path,
):
    resp = install(api, pkg_config(tmp_path))
    assert resp.status_code == 200, resp.text
    assert resp.json()["kind"] == "packages"
    job = wait_prepare(api, resp.json()["id"])
    assert job["status"] == "done", job
    model = job["models"][0]
    assert model["installed"] is True
    assert "restart the server" in model["message"]
    lines = [e["line"] for e in events_of(api, job["id"]) if e["event"] == "output"]
    assert lines[0].startswith("$ uv --color never sync --locked --inexact")
    assert " + xyzlib==1.0" in lines
    real = [c for c in fake_uv.calls() if "--dry-run" not in c["args"]]
    assert len(real) == 1
    args = real[0]["args"]
    extras = [args[i + 1] for i, a in enumerate(args) if a == "--extra"]
    assert extras == ["infer", "model-xyz", "web"]
    report = status(api, pkg_config(tmp_path))
    assert report["restart_required"] is True
    assert detector_row(report)["state"] == "ready"


def test_base_stack_missing_shows_command_and_is_never_installed(
    api,
    fake_uv,  # noqa: F811
    tmp_path,
):
    cfg = pkg_config(tmp_path, extra="infer")
    row = detector_row(status(api, cfg))
    assert row["state"] == "missing"
    assert row["packages"]["action"] == "base"
    assert row["packages"]["command"] == "uv sync --extra infer --extra web"
    assert row["preparable"] is False
    resp = install(api, cfg)
    assert resp.status_code == 422
    assert "never installed by the server" in resp.json()["detail"]
    assert "uv sync --extra infer --extra web" in resp.json()["detail"]
    assert fake_uv.calls() == []


def test_install_packages_names_its_models(api, fake_uv, tmp_path):  # noqa: F811
    resp = api.post(
        "/api/models/prepare",
        json={"models": "all", "config": pkg_config(tmp_path), "packages": True},
    )
    assert resp.status_code == 422
    assert fake_uv.calls() == []


def _gone(pid: int) -> bool:
    try:
        text = Path(f"/proc/{pid}/status").read_text()
    except FileNotFoundError:
        return True
    return "\nState:\tZ" in text


def test_cancel_install_kills_uv_and_frees_the_queue(api, fake_uv, tmp_path):  # noqa: F811
    fake_uv.mode("hang")
    job_id = install(api, pkg_config(tmp_path)).json()["id"]
    deadline = time.monotonic() + 10
    while not (fake_uv.pids.exists() and fake_uv.pids.read_text()):
        assert time.monotonic() < deadline, "the fake uv did not start"
        time.sleep(0.02)
    pids = [int(p) for p in fake_uv.pids.read_text().split()]
    assert api.delete(f"/api/models/prepare/{job_id}").status_code == 200
    job = wait_prepare(api, job_id)
    assert job["status"] == "cancelled"
    deadline = time.monotonic() + 5
    while not all(_gone(p) for p in pids):
        assert time.monotonic() < deadline, "uv's process group survived"
        time.sleep(0.05)
    # The queue is free: the next job runs.
    fake_uv.mode("ok")
    nxt = wait_prepare(api, install(api, pkg_config(tmp_path)).json()["id"])
    assert nxt["status"] == "done"
