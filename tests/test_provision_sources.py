"""Built-in provisioning sources, without the real network (task_0020).

A local `http.server` thread serves files with or without `Range` support, a
404 and a slow stream; git sources clone local repositories; the Hugging Face
source runs against a monkeypatched `huggingface_hub.snapshot_download`.
"""

import hashlib
import http.server
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from det_depth_distance.core.provision import (
    ProvisionCancelled,
    ProvisionError,
    prepare,
)
from det_depth_distance.core.provision import sources as src
from det_depth_distance.core.requirements import (
    CodeRepo,
    HuggingFaceAsset,
    Requirements,
    UrlAsset,
    UserProvided,
    check,
)

DATA = bytes(range(256)) * 2048  # 512 KiB, several CHUNKs
SHA = hashlib.sha256(DATA).hexdigest()


class _Server(http.server.ThreadingHTTPServer):
    daemon_threads = True
    range_support = True
    ranges: list[str | None]
    slow_stop: threading.Event


class _Handler(http.server.BaseHTTPRequestHandler):
    server: _Server

    def log_message(self, *args) -> None:  # keep pytest output clean
        pass

    def do_GET(self) -> None:
        self.server.ranges.append(self.headers.get("Range"))
        if self.path == "/slow":
            self.send_response(200)
            self.send_header("Content-Length", str(1 << 30))
            self.end_headers()
            while not self.server.slow_stop.is_set():
                try:
                    self.wfile.write(b"x" * 4096)
                    self.wfile.flush()
                except OSError:
                    return
                time.sleep(0.01)
            return
        if self.path != "/file.bin":
            self.send_error(404)
            return
        start = 0
        header = self.headers.get("Range")
        if header and self.server.range_support:
            start = int(header.removeprefix("bytes=").split("-")[0])
            self.send_response(206)
            self.send_header(
                "Content-Range", f"bytes {start}-{len(DATA) - 1}/{len(DATA)}"
            )
        else:
            self.send_response(200)
        body = DATA[start:]
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def server():
    srv = _Server(("127.0.0.1", 0), _Handler)
    srv.ranges = []
    srv.slow_stop = threading.Event()
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.slow_stop.set()
    srv.shutdown()
    srv.server_close()


def _url(srv: _Server, path: str = "/file.bin") -> str:
    return f"http://127.0.0.1:{srv.server_address[1]}{path}"


def _asset(srv: _Server, path: str = "/file.bin", **kw) -> UrlAsset:
    return UrlAsset(
        key="w", target="fam/model/w.bin", url=_url(srv, path), **({"sha256": SHA} | kw)
    )


def _no_progress(label: str, done: int, total: int | None) -> None:
    pass


def test_url_source_downloads_atomically_and_verifies_sha(tmp_path, server):
    asset = _asset(server)
    seen: list[tuple[int, int | None]] = []

    def progress(done, total):
        # The target never appears before the download is complete.
        assert not (tmp_path / "fam/model/w.bin").exists()
        seen.append((done, total))

    src.UrlSource().prepare(asset, tmp_path, progress, threading.Event())
    target = tmp_path / "fam/model/w.bin"
    assert target.read_bytes() == DATA
    assert not (tmp_path / "fam/model/w.bin.part").exists()
    assert seen[-1] == (len(DATA), len(DATA))
    assert check(Requirements(assets=(asset,)), tmp_path).state == "ready"


@pytest.mark.parametrize("range_support", [True, False])
def test_url_source_resumes_from_part_file(tmp_path, server, range_support):
    server.range_support = range_support
    part = tmp_path / "fam/model/w.bin.part"
    part.parent.mkdir(parents=True)
    half = len(DATA) // 2
    part.write_bytes(DATA[:half])
    asset = _asset(server)
    assert check(Requirements(assets=(asset,)), tmp_path).state == "missing"
    status = prepare(
        Requirements(assets=(asset,)), tmp_path, _no_progress, threading.Event()
    )
    assert status.state == "ready", status.message
    assert server.ranges == [f"bytes={half}-"]
    # With Range support only the tail was sent; without it the file restarted.
    assert (tmp_path / "fam/model/w.bin").read_bytes() == DATA
    assert not part.exists()


def test_url_sha_mismatch_ends_failed_and_removes_part(tmp_path, server):
    asset = _asset(server, sha256="0" * 64)
    status = prepare(
        Requirements(assets=(asset,)), tmp_path, _no_progress, threading.Event()
    )
    assert status.state == "failed"
    assert "sha256 mismatch" in (status.message or "")
    assert not (tmp_path / "fam/model/w.bin.part").exists()
    assert not (tmp_path / "fam/model/w.bin").exists()


def test_stale_part_with_wrong_bytes_is_restarted_not_accepted(tmp_path, server):
    part = tmp_path / "fam/model/w.bin.part"
    part.parent.mkdir(parents=True)
    part.write_bytes(b"\xff" * 1000)  # a resumed prefix that is not the file
    req = Requirements(assets=(_asset(server),))
    first = prepare(req, tmp_path, _no_progress, threading.Event())
    assert first.state == "failed" and "sha256 mismatch" in (first.message or "")
    assert not part.exists()
    second = prepare(req, tmp_path, _no_progress, threading.Event())
    assert second.state == "ready", second.message
    assert server.ranges == ["bytes=1000-", None]


def test_cancel_mid_download_leaves_no_ready_file(tmp_path, server):
    asset = _asset(server)
    cancel = threading.Event()
    calls = []

    def progress(label, done, total):
        calls.append(done)
        if done > 0:
            cancel.set()  # after the first chunk

    req = Requirements(assets=(asset,))
    status = prepare(req, tmp_path, progress, cancel)
    assert status.state == "missing"
    assert (status.message or "").startswith("cancelled")
    assert not (tmp_path / "fam/model/w.bin").exists()
    part = tmp_path / "fam/model/w.bin.part"
    assert 0 < part.stat().st_size < len(DATA)
    assert check(req, tmp_path).layers["assets"] == "partial"
    # A second Prepare resumes the .part and completes.
    assert prepare(req, tmp_path, _no_progress, threading.Event()).state == "ready"
    assert (tmp_path / "fam/model/w.bin").read_bytes() == DATA


def test_cancel_frees_the_caller_on_a_slow_stream(tmp_path, server):
    asset = _asset(server, "/slow", sha256=None)
    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    started = time.monotonic()
    with pytest.raises(ProvisionCancelled):
        src.UrlSource().prepare(asset, tmp_path, lambda d, t: None, cancel)
    assert time.monotonic() - started < 5
    assert not (tmp_path / "fam/model/w.bin").exists()


def test_http_404_is_readable(tmp_path, server):
    status = prepare(
        Requirements(assets=(_asset(server, "/nope.bin"),)),
        tmp_path,
        _no_progress,
        threading.Event(),
    )
    assert status.state == "failed"
    assert "not found (HTTP 404)" in (status.message or "")


def test_offline_failure_is_readable(tmp_path):
    with socket.socket() as sock:  # a port nobody listens on
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    asset = UrlAsset(key="w", target="w.bin", url=f"http://127.0.0.1:{port}/w.bin")
    status = prepare(
        Requirements(assets=(asset,)), tmp_path, _no_progress, threading.Event()
    )
    assert status.state == "failed"
    assert "offline or unreachable" in (status.message or "")
    assert not (tmp_path / "w.bin").exists()


def test_url_target_escape_is_refused_before_any_request(tmp_path, server):
    root = tmp_path / "root"
    root.mkdir()
    (root / "link").symlink_to(tmp_path)  # leads outside the root
    for target in ("../w.bin", "link/w.bin"):
        asset = UrlAsset(key="w", target=target, url=_url(server))
        status = prepare(
            Requirements(assets=(asset,)), root, _no_progress, threading.Event()
        )
        assert status.state == "failed"
    assert server.ranges == []
    assert not (tmp_path / "w.bin").exists() and not (tmp_path / "w.bin.part").exists()


def test_user_provided_never_downloads(tmp_path, server, monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("a user-provided asset triggered a download")

    monkeypatch.setattr(src.urllib.request, "urlopen", no_network)
    asset = UserProvided(
        key="ckpt", target="u/ckpt.pt", option="weights", hint="Put the checkpoint"
    )
    status = prepare(
        Requirements(assets=(asset,)), tmp_path, _no_progress, threading.Event()
    )
    assert status.state == "failed"
    assert "provide this file: Put the checkpoint" in (status.message or "")
    assert server.ranges == []


# --------------------------------------------------------------- git


def _git(cwd: Path, *args: str) -> str:
    env = {
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "HOME": str(cwd),
    }
    out = subprocess.run(
        ["git", *args], cwd=cwd, env=env, check=True, capture_output=True, text=True
    )
    return out.stdout.strip()


@pytest.fixture
def origin(tmp_path):
    """A local repository with two commits; returns (path, first, second)."""
    repo = tmp_path / "origin"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "a.txt").write_text("1")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-q", "-m", "one")
    first = _git(repo, "rev-parse", "HEAD")
    (repo / "a.txt").write_text("2")
    _git(repo, "commit", "-q", "-am", "two")
    return repo, first, _git(repo, "rev-parse", "HEAD")


def test_git_source_clones_pinned_commit(tmp_path, origin):
    repo, first, _ = origin
    root = tmp_path / "weights"
    code = CodeRepo(name="tool", url=str(repo), commit=first)
    req = Requirements(code=(code,))
    status = prepare(req, root, _no_progress, threading.Event())
    assert status.state == "ready", status.message
    clone = root / "_code" / "tool"
    assert (clone / "a.txt").read_text() == "1"
    assert _git(clone, "rev-parse", "HEAD") == first
    assert not (root / "_code" / "tool.part").exists()
    # A second Prepare is a no-op.
    assert prepare(req, root, _no_progress, threading.Event()).state == "ready"


def test_git_source_moves_existing_clone_to_pin_without_deleting(tmp_path, origin):
    repo, first, second = origin
    root = tmp_path / "weights"
    clone = root / "_code" / "tool"
    clone.parent.mkdir(parents=True)
    _git(clone.parent, "clone", "-q", str(repo), "tool")
    (clone / "notes.txt").write_text("mine")  # untracked: must survive
    assert _git(clone, "rev-parse", "HEAD") == second
    code = CodeRepo(name="tool", url=str(repo), commit=first)
    status = prepare(Requirements(code=(code,)), root, _no_progress, threading.Event())
    assert status.state == "ready", status.message
    assert _git(clone, "rev-parse", "HEAD") == first
    assert (clone / "notes.txt").read_text() == "mine"


def test_git_source_leaves_a_foreign_folder_alone(tmp_path, origin):
    repo, first, _ = origin
    root = tmp_path / "weights"
    folder = root / "_code" / "tool"
    folder.mkdir(parents=True)
    (folder / "keep.txt").write_text("x")
    code = CodeRepo(name="tool", url=str(repo), commit=first)
    status = prepare(Requirements(code=(code,)), root, _no_progress, threading.Event())
    assert status.state == "failed"
    assert "not a git clone" in (status.message or "")
    assert (folder / "keep.txt").read_text() == "x"


def test_git_unknown_commit_fails_and_cleans_temporary_clone(tmp_path, origin):
    repo, _, _ = origin
    root = tmp_path / "weights"
    code = CodeRepo(name="tool", url=str(repo), commit="f" * 40)
    status = prepare(Requirements(code=(code,)), root, _no_progress, threading.Event())
    assert status.state == "failed"
    assert "code:tool" in (status.message or "")
    assert not (root / "_code" / "tool").exists()
    assert not (root / "_code" / "tool.part").exists()


def test_git_cancel_kills_the_clone(tmp_path, origin, monkeypatch):
    repo, first, _ = origin
    cancel = threading.Event()
    cancel.set()  # set before the clone: git is killed at the first poll
    monkeypatch.setattr(src, "GIT_POLL_S", 0.0001)
    code = CodeRepo(name="tool", url=str(repo), commit=first)
    root = tmp_path / "weights"
    status = prepare(Requirements(code=(code,)), root, _no_progress, cancel)
    assert (status.message or "").startswith("cancelled")
    assert not (root / "_code" / "tool").exists()


# --------------------------------------------------------------- Hugging Face


def _hf_asset(**kw) -> HuggingFaceAsset:
    return HuggingFaceAsset(
        key="sam", target="hf", repo="org/model", **({"files": ("config.json",)} | kw)
    )


def test_hf_source_passes_allow_patterns_and_cache_dir(tmp_path, monkeypatch):
    import huggingface_hub

    calls = []

    def fake_snapshot_download(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        repo_dir = Path(kwargs["cache_dir"]) / "models--org--model"
        snap = repo_dir / "snapshots" / "abc"
        snap.mkdir(parents=True)
        (repo_dir / "refs").mkdir()
        (repo_dir / "refs" / kwargs["revision"]).write_text("abc")
        (snap / "config.json").write_text("{}")
        (snap / "model.safetensors").write_bytes(b"x" * 10)
        bar = kwargs["tqdm_class"](total=10, name="huggingface_hub.snapshot_download")
        bar.update(10)
        return str(snap)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    asset = _hf_asset(patterns=("*.safetensors",), revision="main")
    seen = []
    status = prepare(
        Requirements(assets=(asset,)),
        tmp_path,
        lambda label, done, total: seen.append((label, done, total)),
        threading.Event(),
    )
    assert status.state == "ready", status.message
    ((repo, kwargs),) = calls
    assert repo == "org/model"
    assert kwargs["allow_patterns"] == ["config.json", "*.safetensors"]
    assert Path(kwargs["cache_dir"]) == (tmp_path / "hf").resolve()
    assert kwargs["revision"] == "main"
    assert ("sam", 10, 10) in seen


def test_hf_cancel_through_progress_bar(tmp_path, monkeypatch):
    import huggingface_hub

    cancel = threading.Event()

    def fake_snapshot_download(repo_id, **kwargs):
        cancel.set()
        kwargs["tqdm_class"](total=10, name="huggingface_hub.snapshot_download").update(
            1
        )
        raise AssertionError("not cancelled")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    status = prepare(
        Requirements(assets=(_hf_asset(),)), tmp_path, _no_progress, cancel
    )
    assert (status.message or "").startswith("cancelled")


def _hub_error(name: str, message: str = "boom", status: int = 404):
    from huggingface_hub import errors

    cls = getattr(errors, name)
    if not issubclass(cls, errors.HfHubHTTPError):
        return cls(message)
    # HTTP errors need a response of the hub's HTTP client (httpx or a fork).
    http = sys.modules[errors.HfHubHTTPError.__mro__[1].__module__.split(".")[0]]
    request = http.Request("GET", "https://huggingface.co/org/model")
    return cls(message, response=http.Response(status, request=request))


def test_gated_error_message_names_licence_and_hf_token(tmp_path, monkeypatch):
    import huggingface_hub

    def gated(repo_id, **kwargs):
        raise _hub_error("GatedRepoError", "401 Client Error", 401)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", gated)
    status = prepare(
        Requirements(assets=(_hf_asset(gated=True),)),
        tmp_path,
        _no_progress,
        threading.Event(),
    )
    assert status.state == "failed"
    message = status.message or ""
    assert "gated" in message and "accept the licence" in message
    assert "https://huggingface.co/org/model" in message and "HF_TOKEN" in message


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        ("LocalEntryNotFoundError", "offline or unreachable"),
        ("RepositoryNotFoundError", "not found (HTTP 404)"),
        ("RevisionNotFoundError", "not found (HTTP 404)"),
    ],
)
def test_hf_errors_are_readable(tmp_path, monkeypatch, error, expected):
    import huggingface_hub

    def failing(repo_id, **kwargs):
        raise _hub_error(error)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", failing)
    status = prepare(
        Requirements(assets=(_hf_asset(),)), tmp_path, _no_progress, threading.Event()
    )
    assert status.state == "failed"
    assert expected in (status.message or "")


def test_hf_download_that_misses_declared_files_is_failed(tmp_path, monkeypatch):
    import huggingface_hub

    def empty(repo_id, **kwargs):
        snap = Path(kwargs["cache_dir"]) / "models--org--model" / "snapshots" / "abc"
        snap.mkdir(parents=True)
        return str(snap)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", empty)
    status = prepare(
        Requirements(assets=(_hf_asset(),)), tmp_path, _no_progress, threading.Event()
    )
    assert status.state == "failed"
    assert "still not ready" in (status.message or "")


def test_source_error_types_are_exported():
    assert issubclass(ProvisionError, Exception)
    assert not issubclass(ProvisionCancelled, ProvisionError)
