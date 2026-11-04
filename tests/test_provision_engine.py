"""Provisioning engine: registry of sources and `prepare` (task_0020)."""

import functools
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pytest

from av_perception_lab.core.provision import (
    ProvisionError,
    engine,
    prepare,
    register_source,
    source_for,
)
from av_perception_lab.core.requirements import (
    Asset,
    AssetState,
    CodeRepo,
    Requirements,
    UrlAsset,
)

# These tests exercise the engine and the sources as the CLI runs them (any
# source); the web policy (`allow_unknown=False`) has its own tests.
prepare = functools.partial(prepare, allow_unknown=True)


@dataclass(frozen=True, kw_only=True)
class _BlobAsset(Asset):
    """A new asset kind (not registered with any source by default)."""

    content: bytes = b"blob"

    def present(self, root: Path) -> AssetState:
        path = self.path(root)
        return AssetState("ready") if path.is_file() else AssetState("missing", "nope")


class _BlobSource:
    calls: ClassVar[list[str]] = []

    def prepare(self, asset, root, progress, cancel) -> None:
        _BlobSource.calls.append(asset.key)
        path = asset.path(root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(asset.content)
        progress(len(asset.content), len(asset.content))


@pytest.fixture
def sources(monkeypatch):
    """A private copy of the source registry for this test."""
    monkeypatch.setattr(engine, "_SOURCES", dict(engine._SOURCES))
    _BlobSource.calls = []
    return engine._SOURCES


def _none(label, done, total) -> None:
    pass


def test_register_source_adds_new_kind_without_engine_edit(tmp_path, sources):
    register_source(_BlobAsset)(_BlobSource)
    asset = _BlobAsset(key="blob", target="b/blob.bin")
    seen = []
    status = prepare(
        Requirements(assets=(asset,)),
        tmp_path,
        lambda *a: seen.append(a),
        threading.Event(),
    )
    assert status.state == "ready" and status.message is None
    assert (tmp_path / "b/blob.bin").read_bytes() == b"blob"
    assert seen == [("blob", 0, None), ("blob", 4, 4)]
    assert isinstance(source_for(asset), _BlobSource)


def test_register_source_twice_raises(sources):
    register_source(_BlobAsset)(_BlobSource)
    with pytest.raises(ValueError, match="already registered"):
        register_source(_BlobAsset)(_BlobSource)
    with pytest.raises(ValueError):
        register_source(UrlAsset)(_BlobSource)  # built-ins are protected too


def test_subclass_of_a_kind_uses_the_parent_source(sources):
    register_source(_BlobAsset)(_BlobSource)

    @dataclass(frozen=True, kw_only=True)
    class _Child(_BlobAsset):
        pass

    assert isinstance(source_for(_Child(key="c", target="c")), _BlobSource)


def test_unknown_asset_kind_without_source_fails_clearly(tmp_path, sources):
    status = prepare(
        Requirements(assets=(_BlobAsset(key="blob", target="b/blob.bin"),)),
        tmp_path,
        _none,
        threading.Event(),
    )
    assert status.state == "failed"
    assert "blob: no source registered for asset kind" in (status.message or "")
    assert "register_source" in (status.message or "")


def test_prepare_is_noop_when_ready(tmp_path, sources):
    register_source(_BlobAsset)(_BlobSource)
    req = Requirements(assets=(_BlobAsset(key="blob", target="b/blob.bin"),))
    assert prepare(req, tmp_path, _none, threading.Event()).state == "ready"
    calls = []
    again = prepare(req, tmp_path, lambda *a: calls.append(a), threading.Event())
    assert again.state == "ready"
    assert _BlobSource.calls == ["blob"] and calls == []
    # Empty requirements are ready without any source.
    assert prepare(Requirements(), tmp_path, _none, threading.Event()).state == "ready"


def test_failure_of_one_asset_does_not_stop_the_others(tmp_path, sources):
    class _Failing:
        def prepare(self, asset, root, progress, cancel):
            raise ProvisionError("server said no")

    @dataclass(frozen=True, kw_only=True)
    class _Bad(_BlobAsset):
        pass

    register_source(_BlobAsset)(_BlobSource)
    register_source(_Bad)(_Failing)
    req = Requirements(
        assets=(_Bad(key="bad", target="x"), _BlobAsset(key="ok", target="y"))
    )
    status = prepare(req, tmp_path, _none, threading.Event())
    assert status.state == "failed"
    assert status.message == "bad: server said no"
    assert (tmp_path / "y").is_file()


def test_unexpected_os_error_is_reported_not_raised(tmp_path, sources):
    class _DiskFull:
        def prepare(self, asset, root, progress, cancel):
            raise OSError(28, "No space left on device")

    register_source(_BlobAsset)(_DiskFull)
    status = prepare(
        Requirements(assets=(_BlobAsset(key="blob", target="b"),)),
        tmp_path,
        _none,
        threading.Event(),
    )
    assert status.state == "failed"
    assert "No space left on device" in (status.message or "")


def test_cancel_before_start_runs_no_source(tmp_path, sources):
    register_source(_BlobAsset)(_BlobSource)
    cancel = threading.Event()
    cancel.set()
    status = prepare(
        Requirements(assets=(_BlobAsset(key="blob", target="b"),)),
        tmp_path,
        _none,
        cancel,
    )
    assert status.state == "missing"
    assert (status.message or "").startswith("cancelled")
    assert _BlobSource.calls == []


def test_packages_are_reported_not_installed(tmp_path, sources):
    req = Requirements(packages=("surely_not_a_real_package_xyz",), extra="model-x")
    status = prepare(req, tmp_path, _none, threading.Event())
    assert status.state == "missing"
    assert "extra 'model-x'" in (status.message or "")


def test_default_root_is_weights_root(tmp_path, sources, monkeypatch):
    register_source(_BlobAsset)(_BlobSource)
    monkeypatch.setenv("AVPL_WEIGHTS_ROOT", str(tmp_path))
    req = Requirements(assets=(_BlobAsset(key="blob", target="b/x"),))
    assert prepare(req, None, _none, threading.Event()).state == "ready"
    assert (tmp_path / "b/x").is_file()


def test_importing_the_engine_does_not_import_hub_or_torch():
    code = (
        "import sys; import av_perception_lab.core.provision\n"
        "assert 'torch' not in sys.modules\n"
        "assert 'huggingface_hub' not in sys.modules"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr


def test_web_policy_never_fetches_an_unknown_source(tmp_path, sources):
    """`allow_unknown=False` (the default, the web's call): an item whose
    source the adapter does not vouch for is a failure and its source never
    runs; a known one is fetched; the CLI (`allow_unknown=True`) fetches both."""
    register_source(_BlobAsset)(_BlobSource)
    unknown = _BlobAsset(key="typed", target="b/typed.bin")
    known = _BlobAsset(key="fixed", target="b/fixed.bin", known=True)
    req = Requirements(assets=(unknown, known))
    status = engine.prepare(req, tmp_path, _none, threading.Event())
    assert status.state == "failed"
    assert "typed: unknown source: prepare it from the CLI (--prepare)" in (
        status.message
    )
    assert _BlobSource.calls == ["fixed"]
    assert not (tmp_path / "b/typed.bin").exists()
    cli = engine.prepare(req, tmp_path, _none, threading.Event(), allow_unknown=True)
    assert cli.state == "ready" and _BlobSource.calls == ["fixed", "typed"]


@dataclass(frozen=True, kw_only=True)
class _ManualAsset(Asset):
    downloadable: ClassVar[bool] = False

    def present(self, root: Path) -> AssetState:
        path = self.path(root)
        return AssetState("ready") if path.is_file() else AssetState("missing", "x")


class _ManualSource:
    def prepare(self, asset, root, progress, cancel) -> None:
        raise ProvisionError(f"provide this file: put {asset.target} there")


def test_non_downloadable_item_is_reported_not_failed(tmp_path, sources):
    """A step that was never possible is not a failure: the model stays
    missing with the source's "provide this file" hint, the others are
    fetched; without a registered source it is the readable registry error."""
    register_source(_BlobAsset)(_BlobSource)
    manual = _ManualAsset(key="manual", target="m/file.pth")
    other = _BlobAsset(key="fixed", target="b/fixed.bin", known=True)
    req = Requirements(assets=(manual, other))
    status = engine.prepare(req, tmp_path, _none, threading.Event())
    assert status.state == "failed"  # no source registered for the manual kind
    assert "no source registered" in status.message
    register_source(_ManualAsset)(_ManualSource)
    status = engine.prepare(req, tmp_path, _none, threading.Event())
    assert status.state == "missing"
    assert status.message == "manual: provide this file: put m/file.pth there"
    assert (tmp_path / "b/fixed.bin").is_file()


# --- clone before the files that go inside it (task_0028) --------------------


@dataclass(frozen=True, kw_only=True)
class _FakeRepo(CodeRepo):
    """A code repo whose clone is a folder with a `.git` marker file."""

    def present(self, root: Path) -> AssetState:
        ok = (self.path(root) / ".git").is_file()
        return AssetState("ready") if ok else AssetState("missing", "not cloned")


class _FakeGit:
    fail = False
    seen_folder_before: ClassVar[list[bool]] = []

    def prepare(self, repo, root, progress, cancel) -> None:
        path = repo.path(root)
        # A folder created before the clone would make `git clone` refuse it.
        _FakeGit.seen_folder_before.append(path.exists())
        if _FakeGit.fail:
            raise ProvisionError("git clone failed: offline")
        path.mkdir(parents=True)
        (path / ".git").write_text("x")


@pytest.mark.parametrize("fail", [False, True])
def test_file_inside_a_clone_waits_for_the_clone(tmp_path, sources, fail):
    register_source(_BlobAsset)(_BlobSource)
    register_source(_FakeRepo)(_FakeGit)
    _FakeGit.fail, _FakeGit.seen_folder_before = fail, []
    repo = _FakeRepo(name="Tool", url="https://h/t.git", commit="a" * 40)
    inside = _BlobAsset(key="weights", target="_code/Tool/pretrained/w.pth")
    outside = _BlobAsset(key="other", target="elsewhere/o.bin")
    # Assets listed first: the engine still runs the code layer first.
    req = Requirements(assets=(inside, outside), code=(repo,))
    status = prepare(req, tmp_path, _none, threading.Event())
    assert _FakeGit.seen_folder_before == [False]
    assert (tmp_path / "elsewhere/o.bin").is_file()
    if fail:
        assert status.state == "failed"
        assert "code:Tool: git clone failed: offline" in status.message
        assert "weights: not downloaded: it goes inside code:Tool" in status.message
        assert not (tmp_path / "_code/Tool").exists()  # the next clone can run
        assert _BlobSource.calls == ["other"]
    else:
        assert status.state == "ready", status.message
        assert (tmp_path / "_code/Tool/pretrained/w.pth").is_file()
        assert _BlobSource.calls == ["weights", "other"]
