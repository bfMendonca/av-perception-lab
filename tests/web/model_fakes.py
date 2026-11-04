"""Fake components and sources of the model-readiness API tests (task_0022).

No network and no model: `GateDetector` (registered as detector `gate` by the
`fake_models` fixture) declares one `GateAsset` per `blob` option, and
`GateSource` writes it (mode ok), blocks until released or cancelled (mode
block) or fails with a readable reason (mode fail). The DA3 depth of
`gate_config` uses an `hf_cache` outside the weights root (nothing declared
but packages) and `camera.estimate` is off, so the gate assets are the only
files a config needs.
"""

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from av_perception_lab.core import registry
from av_perception_lab.core import requirements as rq
from av_perception_lab.core.components import Detector, DetectorOptions
from av_perception_lab.core.provision import (
    ProvisionCancelled,
    ProvisionError,
    engine,
)
from av_perception_lab.core.requirements import Asset, AssetState, Requirements

CODE_URL = "https://example.invalid/fixed-in-code"


@dataclass(frozen=True, kw_only=True)
class GateAsset(Asset):
    url: str = CODE_URL  # fixed by the adapter's code, never by a request
    mode: str = "ok"

    def present(self, root: Path) -> AssetState:
        path = self.path(root)
        if path.is_file():
            return AssetState("ready")
        return AssetState("missing", f"{self.key}: {path} not found")


class GateSource:
    lock = threading.Lock()
    calls: ClassVar[list[tuple[str, str]]] = []
    active = 0
    peak = 0
    started = threading.Event()
    release = threading.Event()

    @classmethod
    def reset(cls) -> None:
        cls.calls = []
        cls.active = cls.peak = 0
        cls.started = threading.Event()
        cls.release = threading.Event()

    def prepare(self, asset, root, progress, cancel) -> None:
        cls = GateSource
        with cls.lock:
            cls.calls.append((asset.target, asset.url))
            cls.active += 1
            cls.peak = max(cls.peak, cls.active)
        try:
            path = asset.path(root)
            path.parent.mkdir(parents=True, exist_ok=True)
            if asset.mode == "fail":
                raise ProvisionError("offline or unreachable: example.invalid")
            if asset.mode == "block":
                cls.started.set()
                while not cls.release.wait(0.01):
                    if cancel.is_set():
                        path.with_name(path.name + ".part").write_bytes(b"half")
                        raise ProvisionCancelled()
            for done in range(0, 8, 2):
                progress(done, 8)
                time.sleep(0.005)
            path.write_bytes(b"x" * 8)
            progress(8, 8)
        finally:
            with cls.lock:
                cls.active -= 1


class GateDetector(Detector):
    """A detector whose only requirement is one gate asset."""

    @dataclass(frozen=True, slots=True, kw_only=True)
    class Options(DetectorOptions):
        blob: str = "a"
        mode: str = "ok"
        manual: str = ""  # also a user-provided file gate/<manual>.pth
        known: bool = True  # the source is vouched for (False: typed by a user)

    @classmethod
    def requirements(cls, options: Options) -> Requirements:
        asset = GateAsset(
            key="weights",
            target=f"gate/{options.blob}.bin",
            mode=options.mode,
            size_bytes=8,
            known=options.known,  # in a real adapter: fixed in its code
        )
        assets: tuple[Asset, ...] = (asset,)
        if options.manual:
            assets += (
                rq.UserProvided(
                    key="manual",
                    target=f"gate/{options.manual}.pth",
                    option="manual",
                    hint=f"manual download: save {options.manual}.pth by hand",
                ),
            )
        return Requirements(assets=assets)

    @property
    def class_names(self) -> list[str]:
        return ["car"]

    def _load(self, ctx) -> None:
        raise AssertionError("a model was loaded")

    def _predict(self, rgb):
        raise AssertionError("unreachable")


GateDetector.type_name = "gate"


def gate_config(
    tmp_path: Path, *blobs: str | tuple[str, str] | dict[str, Any]
) -> dict[str, Any]:
    """A valid config with one gate detector per blob (`(blob, mode)` too, or
    a dict of extra options with its `blob`)."""
    detectors = []
    for blob in blobs:
        if isinstance(blob, dict):
            extra = dict(blob)
            name, mode = extra.pop("blob"), extra.pop("mode", "ok")
        else:
            name, mode = blob if isinstance(blob, tuple) else (blob, "ok")
            extra = {}
        detectors.append(
            {"type": "gate", "name": f"g{name}", "blob": name, "mode": mode, **extra}
        )
    return {
        "detectors": detectors,
        "depth": {"type": "da3", "hf_cache": str(tmp_path / "elsewhere")},
        "camera": {"estimate": None},
    }


@pytest.fixture
def fake_models(monkeypatch, tmp_path):
    """Weights root under tmp, detector `gate` and its source registered for
    this test only, every declared package treated as installed."""
    root = tmp_path / "weights"
    monkeypatch.setattr(rq, "_explicit_root", root)
    registry.load_adapters()
    monkeypatch.setitem(registry._REGISTRY["detector"], "gate", GateDetector)
    monkeypatch.setattr(engine, "_SOURCES", dict(engine._SOURCES))
    engine.register_source(GateAsset)(GateSource)
    monkeypatch.setattr(rq, "_top_level_present", bool)
    GateSource.reset()
    yield SimpleNamespace(root=root, source=GateSource)
    GateSource.release.set()  # never leave a worker thread blocked


def wait_prepare(api, job_id: str, timeout: float = 10.0) -> dict[str, Any]:
    """Poll a prepare job until it ends."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = api.get(f"/api/models/prepare/{job_id}").json()
        if job["status"] in ("done", "failed", "cancelled"):
            return job
        time.sleep(0.02)
    raise AssertionError(f"prepare {job_id} did not end: {job}")
