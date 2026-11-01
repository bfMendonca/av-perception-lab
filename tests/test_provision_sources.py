"""Built-in provisioning sources, without the real network (task_0020).

A local `http.server` thread serves files with or without `Range` support, a
404 and a slow stream; git sources clone local repositories; the Hugging Face
source runs against a monkeypatched `huggingface_hub.snapshot_download`.
"""

import functools
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

# These tests exercise the engine and the sources as the CLI runs them (any
# source); the web policy (`allow_unknown=False`) has its own tests.
prepare = functools.partial(prepare, allow_unknown=True)

DATA = bytes(range(256)) * 2048  # 512 KiB, several CHUNKs
SHA = hashlib.sha256(DATA).hexdigest()


ETAG = '"v1"'


class _Server(http.server.ThreadingHTTPServer):
    daemon_threads = True
    range_support = True  # honour Range (and If-Range) on /file.bin
    range_mode = "ok"  # "ok", "bad_start" (206 from 0) or "416"
    short_once = False  # /short: first answer cut in half
    etag = ETAG
    ranges: list[str | None]
    if_ranges: list[str | None]
    slow_stop: threading.Event


class _Handler(http.server.BaseHTTPRequestHandler):
    server: _Server

    def log_message(self, *args) -> None:  # keep pytest output clean
        pass

    def _redirect(self, location: str) -> None:
        self.send_response(302)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        srv = self.server
        srv.ranges.append(self.headers.get("Range"))
        srv.if_ranges.append(self.headers.get("If-Range"))
        if self.path == "/slow":
            self.send_response(200)
            self.send_header("Content-Length", str(1 << 30))
            self.end_headers()
            while not srv.slow_stop.is_set():
                try:
                    self.wfile.write(b"x" * 4096)
                    self.wfile.flush()
                except OSError:
                    return
                time.sleep(0.01)
            return
        if self.path == "/to-file":
            return self._redirect("file:///etc/hostname")
        if self.path == "/to-ftp":
            return self._redirect("ftp://127.0.0.1/file.bin")
        if self.path == "/to-ok":
            return self._redirect("/file.bin")
        if self.path == "/nolength":  # HTTP/1.0: the body ends when the socket closes
            self.send_response(200)
            self.end_headers()
            self.wfile.write(DATA)
            return
        if self.path not in ("/file.bin", "/short"):
            self.send_error(404)
            return
        start = 0
        header = self.headers.get("Range")
        if_range = self.headers.get("If-Range")
        if header and srv.range_support and if_range in (None, srv.etag):
            if srv.range_mode == "416":
                self.send_error(416)
                return
            start = int(header.removeprefix("bytes=").split("-")[0])
            sent = 0 if srv.range_mode == "bad_start" else start
            self.send_response(206)
            self.send_header(
                "Content-Range", f"bytes {sent}-{len(DATA) - 1}/{len(DATA)}"
            )
            start = sent
        else:
            self.send_response(200)
        body = DATA[start:]
        self.send_header("Content-Length", str(len(body)))
        self.send_header("ETag", srv.etag)
        self.end_headers()
        if self.path == "/short" and srv.short_once:
            srv.short_once = False
            self.wfile.write(body[: len(body) // 2])  # then the socket closes
            return
        self.wfile.write(body)


@pytest.fixture
def server():
    srv = _Server(("127.0.0.1", 0), _Handler)
    srv.ranges = []
    srv.if_ranges = []
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


def _half_part(tmp_path: Path, etag: str | None = ETAG) -> tuple[Path, int]:
    """A `.part` holding the first half of DATA, with its validator sidecar."""
    part = tmp_path / "fam/model/w.bin.part"
    part.parent.mkdir(parents=True, exist_ok=True)
    half = len(DATA) // 2
    part.write_bytes(DATA[:half])
    if etag is not None:
        (tmp_path / "fam/model/w.bin.part.meta").write_text(etag)
    return part, half


def _prepare(asset: UrlAsset, root: Path):
    return prepare(Requirements(assets=(asset,)), root, _no_progress, threading.Event())


@pytest.mark.parametrize("range_support", [True, False])
def test_url_source_resumes_from_part_file(tmp_path, server, range_support):
    server.range_support = range_support
    part, half = _half_part(tmp_path)
    asset = _asset(server)
    assert check(Requirements(assets=(asset,)), tmp_path).state == "missing"
    status = _prepare(asset, tmp_path)
    assert status.state == "ready", status.message
    assert server.ranges == [f"bytes={half}-"]
    assert server.if_ranges == [ETAG]
    # With Range support only the tail was sent; without it the file restarted.
    assert (tmp_path / "fam/model/w.bin").read_bytes() == DATA
    assert not part.exists()
    assert not (tmp_path / "fam/model/w.bin.part.meta").exists()


def test_resume_with_changed_remote_restarts(tmp_path, server):
    _, half = _half_part(tmp_path, etag='"v0"')  # the remote is now "v1"
    status = _prepare(_asset(server), tmp_path)
    assert status.state == "ready", status.message
    assert server.ranges == [f"bytes={half}-"] and server.if_ranges == ['"v0"']
    assert (tmp_path / "fam/model/w.bin").read_bytes() == DATA


def test_part_without_validator_or_integrity_is_not_resumed(tmp_path, server):
    _half_part(tmp_path, etag=None)  # no sidecar: never trusted
    assert _prepare(_asset(server), tmp_path).state == "ready"
    _half_part(tmp_path / "b")  # sidecar, but nothing to verify the result
    asset = _asset(server, sha256=None, size_bytes=None)
    assert _prepare(asset, tmp_path / "b").state == "ready"
    assert server.ranges == [None, None]


@pytest.mark.parametrize("mode", ["bad_start", "416"])
def test_wrong_range_answer_restarts_from_zero(tmp_path, server, mode):
    server.range_mode = mode
    _, half = _half_part(tmp_path)
    status = _prepare(_asset(server), tmp_path)
    assert status.state == "ready", status.message
    assert server.ranges == [f"bytes={half}-", None]
    assert (tmp_path / "fam/model/w.bin").read_bytes() == DATA


def test_body_shorter_than_content_length_is_kept_and_resumed(tmp_path, server):
    server.short_once = True
    asset = _asset(server, "/short")
    first = _prepare(asset, tmp_path)
    assert first.state == "failed" and "interrupted" in (first.message or "")
    assert not (tmp_path / "fam/model/w.bin").exists()
    part = tmp_path / "fam/model/w.bin.part"
    assert part.stat().st_size == len(DATA) // 2
    second = _prepare(asset, tmp_path)
    assert second.state == "ready", second.message
    assert server.ranges == [None, f"bytes={len(DATA) // 2}-"]
    assert (tmp_path / "fam/model/w.bin").read_bytes() == DATA


def test_short_body_without_content_length_or_sha_is_never_ready(tmp_path, server):
    server.short_once = True
    asset = _asset(server, "/short", sha256=None)  # Content-Length still sent
    assert "interrupted" in (_prepare(asset, tmp_path).message or "")
    assert not (tmp_path / "fam/model/w.bin").exists()


def test_unverifiable_download_fails_and_leaves_no_file(tmp_path, server):
    asset = _asset(server, "/nolength", sha256=None)
    status = _prepare(asset, tmp_path)
    assert status.state == "failed"
    assert src.UNVERIFIABLE in (status.message or "")
    assert not (tmp_path / "fam/model/w.bin").exists()
    assert not (tmp_path / "fam/model/w.bin.part").exists()
    # A declared size (or sha256) makes the same answer acceptable.
    sized = _asset(server, "/nolength", sha256=None, size_bytes=len(DATA))
    assert _prepare(sized, tmp_path).state == "ready"


@pytest.mark.parametrize("path", ["/to-file", "/to-ftp"])
def test_redirect_outside_http_is_refused(tmp_path, server, path):
    status = _prepare(_asset(server, path), tmp_path)
    assert status.state == "failed"
    assert "refused redirect" in (status.message or "")
    assert not (tmp_path / "fam/model/w.bin").exists()


def test_http_redirect_is_followed(tmp_path, server):
    assert _prepare(_asset(server, "/to-ok"), tmp_path).state == "ready"
    assert server.ranges == [None, None]


def test_https_to_http_redirect_is_refused():
    import email.message
    import io

    request = src.urllib.request.Request("https://example.org/w.bin")
    headers = email.message.Message()
    headers["Location"] = "http://example.org/w.bin"
    with pytest.raises(ProvisionError, match="downgrade"):
        src._SafeRedirect().http_error_302(request, io.BytesIO(), 302, "Found", headers)


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
    (tmp_path / "fam/model/w.bin.part.meta").write_text(ETAG)
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

    monkeypatch.setattr(src, "_open", no_network)
    asset = UserProvided(
        key="ckpt", target="u/ckpt.pt", option="weights", hint="Put the checkpoint"
    )
    status = prepare(
        Requirements(assets=(asset,)), tmp_path, _no_progress, threading.Event()
    )
    # Never possible to download: reported, not a failure (the model stays
    # missing with the hint).
    assert status.state == "missing"
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


def _fake_git(tmp_path: Path, body: str) -> str:
    """An executable standing in for git (POSIX shell)."""
    script = tmp_path / "fake-git"
    script.write_text("#!/bin/sh\n" + body)
    script.chmod(0o755)
    return str(script)


def test_git_cancel_kills_the_clone(tmp_path, monkeypatch):
    # The stand-in creates the clone folder (last argument) and then hangs, so
    # the cancel lands while "git clone" runs; exec keeps one process to kill.
    monkeypatch.setattr(
        src,
        "GIT",
        _fake_git(
            tmp_path, 'for a; do last=$a; done\nmkdir -p "$last"\nexec sleep 30\n'
        ),
    )
    root = tmp_path / "weights"
    cancel = threading.Event()
    clone_dir = root / "_code" / "tool.part"

    def progress(done, total):
        threading.Timer(0.3, cancel.set).start()

    code = CodeRepo(name="tool", url="https://example.invalid/x.git", commit="a" * 40)
    started = time.monotonic()
    with pytest.raises(ProvisionCancelled):
        src.GitSource().prepare(code, root, progress, cancel)
    assert time.monotonic() - started < 10
    assert not clone_dir.exists()
    assert not (root / "_code" / "tool").exists()


def test_git_command_times_out(tmp_path, monkeypatch):
    monkeypatch.setattr(src, "GIT", _fake_git(tmp_path, "exec sleep 30\n"))
    monkeypatch.setattr(src, "GIT_TIMEOUT_S", 0.3)
    started = time.monotonic()
    with pytest.raises(ProvisionError, match="timed out after 0.3 s"):
        src._git(["status"], threading.Event(), "git status")
    assert time.monotonic() - started < 10


@pytest.mark.parametrize("pin", ["-x", "--upload-pack=touch", "main"])
def test_git_pin_that_is_not_a_hex_commit_is_refused(pin):
    with pytest.raises(ValueError, match="hex"):
        CodeRepo(name="tool", url="x", commit=pin)


def test_git_arguments_end_options_before_the_commit(tmp_path, origin, monkeypatch):
    repo, first, _ = origin
    calls = []
    real = src._git

    def spy(args, cancel, what):
        calls.append(args)
        return real(args, cancel, what)

    monkeypatch.setattr(src, "_git", spy)
    code = CodeRepo(name="tool", url=str(repo), commit=first)
    root = tmp_path / "weights"
    status = prepare(Requirements(code=(code,)), root, _no_progress, threading.Event())
    assert status.state == "ready", status.message
    assert calls
    for args in calls:
        if first in " ".join(args):
            commit_at = next(i for i, a in enumerate(args) if first in a)
            assert "--end-of-options" in args[:commit_at], args
    assert any("switch" in args for args in calls)


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


def test_existing_target_is_never_replaced(tmp_path, server):
    """A file at the target that is not the declared one (another size) is
    the user's: Prepare refuses, nothing is requested, nothing changes."""
    target = tmp_path / "fam/model/w.bin"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"fine-tuned")
    asset = _asset(server, size_bytes=len(DATA))
    with pytest.raises(ProvisionError, match="never replaced: move or delete it"):
        src.UrlSource().prepare(asset, tmp_path, _no_progress, threading.Event())
    status = _prepare(asset, tmp_path)
    assert status.state == "failed"
    assert "exists with a different size at fam/model/w.bin" in status.message
    assert target.read_bytes() == b"fine-tuned"
    assert server.ranges == []
    assert not target.with_name("w.bin.part").exists()


@pytest.mark.parametrize("path", ["/nolength", "/file.bin"])
def test_body_larger_than_expected_stops_and_removes_the_part(tmp_path, server, path):
    # Declared 1000 bytes; the server sends 512 KiB (with or without a length).
    asset = _asset(server, path, sha256=None, size_bytes=1000)
    status = _prepare(asset, tmp_path)
    assert status.state == "failed"
    assert "larger than expected (more than 1000 bytes)" in status.message
    assert not (tmp_path / "fam/model/w.bin.part").exists()
    assert not (tmp_path / "fam/model/w.bin").exists()


def test_free_disk_space_is_checked_before_writing(tmp_path, server, monkeypatch):
    real = src.shutil.disk_usage
    monkeypatch.setattr(src.shutil, "disk_usage", lambda p: real(p)._replace(free=1000))
    status = _prepare(_asset(server, size_bytes=len(DATA)), tmp_path)
    assert status.state == "failed"
    assert "not enough free disk space" in status.message
    part = tmp_path / "fam/model/w.bin.part"
    assert not part.exists() or part.stat().st_size == 0
