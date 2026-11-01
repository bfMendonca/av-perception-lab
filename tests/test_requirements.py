"""Requirements declaration and the torch-free status check (task_0019)."""

import hashlib
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from det_depth_distance.core import requirements as rq
from det_depth_distance.core.requirements import (
    Asset,
    AssetState,
    CodeRepo,
    HuggingFaceAsset,
    Requirements,
    UrlAsset,
    UserProvided,
    check,
)


def _write(path: Path, data: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _asset(**kw) -> UrlAsset:
    base = {"key": "w", "target": "m/w.bin", "size_bytes": None, "license": None}
    return UrlAsset(url="https://example.invalid/w.bin", **{**base, **kw})


def test_empty_requirements_are_ready(tmp_path):
    status = check(Requirements(), tmp_path)
    assert status.state == "ready"
    assert status.missing == ()
    assert status.message is None


def test_missing_asset_file_is_missing_with_layer_detail(tmp_path):
    status = check(Requirements(assets=(_asset(size_bytes=10),)), tmp_path)
    assert status.state == "missing"
    assert status.layers["assets"] == "missing"
    assert status.missing == ("w",)
    assert status.size_bytes == 10


def test_present_asset_is_ready(tmp_path):
    _write(tmp_path / "m/w.bin", b"12345")
    status = check(Requirements(assets=(_asset(size_bytes=5),)), tmp_path)
    assert status.state == "ready"
    assert status.layers["assets"] == "ready"


def test_existing_file_with_another_size_or_sha_is_never_ready_nor_partial(tmp_path):
    # An existing file that is not the declared one is the user's (a
    # fine-tuned checkpoint under a release name): "invalid", so nothing ever
    # downloads over it, with a message that says what to do.
    f = _write(tmp_path / "m/w.bin", b"123")
    sized = check(Requirements(assets=(_asset(size_bytes=5),)), tmp_path)
    assert sized.state == "failed" and sized.layers["assets"] == "invalid"
    assert "exists with a different size at m/w.bin" in sized.message
    assert "move or delete it, or point the config at another file" in sized.message
    good = hashlib.sha256(b"123").hexdigest()
    assert check(Requirements(assets=(_asset(sha256=good),)), tmp_path).state == (
        "ready"
    )
    f.write_bytes(b"12")  # another content
    bad = check(Requirements(assets=(_asset(sha256=good),)), tmp_path)
    assert bad.state == "failed"
    assert bad.layers["assets"] == "invalid"
    assert "exists with a different sha256" in bad.message
    (tmp_path / "d/w.bin").mkdir(parents=True)  # a folder at the target
    folder = check(Requirements(assets=(_asset(target="d/w.bin"),)), tmp_path)
    assert folder.layers["assets"] == "invalid" and "not a file" in folder.message


def test_part_file_is_never_ready(tmp_path):
    # A complete target is judged on its own; a leftover .part is not ready.
    _write(tmp_path / "m/w.bin", b"12345")
    _write(tmp_path / "m/w.bin.part", b"1")
    status = check(Requirements(assets=(_asset(size_bytes=5),)), tmp_path)
    assert status.state == "ready"
    only_part = tmp_path / "o"
    _write(only_part / "w.bin.part", b"1")
    status = check(Requirements(assets=(_asset(target="o/w.bin"),)), tmp_path)
    assert status.layers["assets"] == "partial"


def test_user_provided_missing_reports_hint_and_never_downloadable(tmp_path):
    a = UserProvided(
        key="ckpt",
        target="u/ckpt.pt",
        size_bytes=None,
        license=None,
        option="weights",
        hint="Put the checkpoint at weights/u/ckpt.pt",
    )
    assert a.downloadable is False
    assert _asset().downloadable is True
    status = check(Requirements(assets=(a,)), tmp_path)
    assert status.state == "missing"
    assert "Put the checkpoint" in (status.message or "")
    _write(tmp_path / "u/ckpt.pt")
    assert check(Requirements(assets=(a,)), tmp_path).state == "ready"


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


def test_code_repo_ready_only_when_commit_matches_head(tmp_path):
    repo = tmp_path / "_code" / "tool"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q")
    (repo / "a.txt").write_text("1")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-q", "-m", "one")
    first = _git(repo, "rev-parse", "HEAD")
    req = Requirements(code=(CodeRepo(name="tool", url="x", commit=first),))
    assert check(req, tmp_path).state == "ready"
    (repo / "a.txt").write_text("2")
    _git(repo, "commit", "-q", "-am", "two")
    status = check(req, tmp_path)
    assert status.state == "missing"
    assert status.layers["code"] == "partial"
    absent = Requirements(code=(CodeRepo(name="nope", url="x", commit=first),))
    assert check(absent, tmp_path).layers["code"] == "missing"


def test_package_layer_uses_find_spec_without_importing(tmp_path):
    top = "colorsys"
    sys.modules.pop(top, None)
    ok = check(Requirements(packages=(top, "no_such_pkg_zz"), extra="infer"), tmp_path)
    assert top not in sys.modules
    assert ok.layers["packages"] == "missing"
    assert ok.missing == ("package:no_such_pkg_zz",)
    assert "infer" in (ok.message or "")
    assert check(Requirements(packages=(top,)), tmp_path).state == "ready"


def test_check_does_not_import_torch(tmp_path):
    code = (
        "import sys\n"
        "from pathlib import Path\n"
        "from det_depth_distance.core.requirements import *\n"
        "from det_depth_distance.core.requirements import check, Requirements\n"
        f"check(Requirements(packages=('torch','rfdetr')), Path({str(tmp_path)!r}))\n"
        "assert 'torch' not in sys.modules, 'torch imported'\n"
    )
    out = subprocess.run(
        [sys.executable, "-I", "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr


def test_asset_target_escape_is_rejected(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "link").symlink_to(outside)
    for target in ("../x", "/etc/passwd", "link/x", "a/../../x"):
        asset = _asset(target=target)
        with pytest.raises(rq.UnsafePathError):
            asset.path(root)
        status = check(Requirements(assets=(asset,)), root)
        assert status.state == "failed"
        assert status.layers["assets"] == "invalid"
    assert _asset(target="ok/w.bin").path(root) == root.resolve() / "ok/w.bin"


def test_register_asset_kind_makes_new_kind_checkable(tmp_path):
    @rq.register_asset_kind("test-marker-kind")
    @dataclass(frozen=True, kw_only=True)
    class MarkerAsset(Asset):
        def present(self, root: Path) -> AssetState:
            ok = (self.path(root) / "MARK").exists()
            return AssetState("ready" if ok else "missing", "marker")

    try:
        asset = MarkerAsset(key="m", target="mk", size_bytes=None, license=None)
        assert rq.asset_kinds()["test-marker-kind"] is MarkerAsset
        assert check(Requirements(assets=(asset,)), tmp_path).state == "missing"
        _write(tmp_path / "mk/MARK")
        assert check(Requirements(assets=(asset,)), tmp_path).state == "ready"
        with pytest.raises(ValueError):
            rq.register_asset_kind("test-marker-kind")(MarkerAsset)
    finally:
        rq._ASSET_KINDS.pop("test-marker-kind", None)


def _hf_snapshot(root: Path, repo: str, rev: str = "abc") -> Path:
    base = root / "hf" / ("models--" + repo.replace("/", "--"))
    (base / "refs").mkdir(parents=True, exist_ok=True)
    (base / "refs" / "main").write_text(rev)
    return base / "snapshots" / rev


def test_huggingface_asset_reads_cache_layout(tmp_path):
    hf = HuggingFaceAsset(
        key="sam",
        target="hf",
        size_bytes=None,
        license=None,
        repo="org/name",
        files=("config.json", "model.safetensors"),
    )
    assert check(Requirements(assets=(hf,)), tmp_path).state == "missing"
    snap = _hf_snapshot(tmp_path, "org/name")
    _write(snap / "config.json")
    assert check(Requirements(assets=(hf,)), tmp_path).layers["assets"] == "partial"
    _write(snap / "model.safetensors", b"abc")
    assert check(Requirements(assets=(hf,)), tmp_path).state == "ready"
    incomplete = snap.parent.parent / "blobs" / "deadbeef.incomplete"
    _write(incomplete)
    assert check(Requirements(assets=(hf,)), tmp_path).layers["assets"] == "partial"
    incomplete.unlink()
    pat = HuggingFaceAsset(
        key="p",
        target="hf",
        size_bytes=100,
        license=None,
        repo="org/name",
        patterns=("*.safetensors",),
    )
    assert check(Requirements(assets=(pat,)), tmp_path).layers["assets"] == "partial"


def test_weights_root_precedence(tmp_path, monkeypatch):
    monkeypatch.setattr(rq, "_explicit_root", None)
    monkeypatch.delenv("DDD_WEIGHTS_ROOT", raising=False)
    assert rq.weights_root().name == "weights"
    monkeypatch.setenv("DDD_WEIGHTS_ROOT", str(tmp_path / "env"))
    assert rq.weights_root() == tmp_path / "env"
    rq.set_weights_root(tmp_path / "set")
    assert rq.weights_root() == tmp_path / "set"
    rq.set_weights_root(None)
    assert rq.weights_root() == tmp_path / "env"


def test_code_repo_name_must_be_one_plain_component():
    for bad in ("", ".", "..", "../evil", "a/b", "/abs", "a\\b", "x\0y"):
        with pytest.raises(rq.UnsafePathError):
            CodeRepo(name=bad, url="x", commit="a" * 40)
    CodeRepo(name="ok-1.2_x", url="x", commit="a" * 40)


def test_code_repo_commit_must_be_a_hex_pin():
    for bad in ("main", "v1.0", "-x", "--upload-pack=touch", "abc", "a" * 65, "g" * 40):
        with pytest.raises(rq.UnsafePathError, match="hex"):
            CodeRepo(name="ok", url="x", commit=bad)
    CodeRepo(name="ok", url="x", commit="90f1b86")
    CodeRepo(name="ok", url="x", commit="A" * 64)


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file modes")
def test_unreadable_paths_do_not_raise(tmp_path):
    locked = tmp_path / "locked"
    _write(locked / "w.bin", b"x")
    locked.chmod(0)
    try:
        status = check(Requirements(assets=(_asset(target="locked/w.bin"),)), tmp_path)
    finally:
        locked.chmod(0o755)
    assert status.state == "missing"
    assert status.layers["assets"] in ("partial", "missing")
    f = _write(tmp_path / "m/w.bin", b"x")
    f.chmod(0)
    try:
        status = check(Requirements(assets=(_asset(sha256="0" * 64),)), tmp_path)
    finally:
        f.chmod(0o644)
    assert status.layers["assets"] == "partial"
    assert "cannot inspect" in (status.message or "")


def test_symlink_loop_and_nul_byte_do_not_raise(tmp_path):
    (tmp_path / "a").symlink_to(tmp_path / "b")
    (tmp_path / "b").symlink_to(tmp_path / "a")
    loop = check(Requirements(assets=(_asset(target="a/x"),)), tmp_path)
    assert loop.state in ("missing", "failed")
    nul = check(Requirements(assets=(_asset(target="x\0y"),)), tmp_path)
    assert nul.state == "failed"
    assert nul.layers["assets"] == "invalid"


# --- Hugging Face cache layout as the hub writes it: snapshot files are
# symlinks into blobs/<hash>.


def _hf(**kw) -> HuggingFaceAsset:
    base = {"key": "hf", "target": "hf", "repo": "org/name"}
    return HuggingFaceAsset(**{**base, **kw})


def _hf_link(root: Path, rev: str, name: str, data: bytes | None = b"x") -> Path:
    """Snapshot `name` as a symlink to a blob (dangling when data is None)."""
    base = root / "hf" / "models--org--name"
    blob = base / "blobs" / hashlib.sha1(f"{rev}/{name}".encode()).hexdigest()
    if data is not None:
        _write(blob, data)
    link = base / "snapshots" / rev / name
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(os.path.relpath(blob, link.parent))
    return link


def _assets(tmp_path, asset) -> str:
    return check(Requirements(assets=(asset,)), tmp_path).layers["assets"]


def test_hf_symlinked_snapshot_with_nested_file_is_ready(tmp_path):
    _hf_link(tmp_path, "r1", "config.json")
    _hf_link(tmp_path, "r1", "sub/dir/model.safetensors", b"abcdef")
    asset = _hf(files=("config.json", "sub/dir/model.safetensors"), size_bytes=7)
    assert _assets(tmp_path, asset) == "ready"
    assert _assets(tmp_path, _hf(patterns=("sub/*/*.safetensors",))) == "ready"
    assert _assets(
        tmp_path, _hf(files=("sub/dir/model.safetensors",), size_bytes=99)
    ) == ("partial")


def test_hf_dangling_blob_symlink_is_not_present(tmp_path):
    _hf_link(tmp_path, "r1", "config.json")
    _hf_link(tmp_path, "r1", "model.safetensors", data=None)
    asset = _hf(files=("config.json", "model.safetensors"))
    status = check(Requirements(assets=(asset,)), tmp_path)
    assert status.layers["assets"] == "partial"
    assert "model.safetensors" in (status.message or "")


def test_hf_incomplete_blob_is_partial(tmp_path):
    _hf_link(tmp_path, "r1", "config.json")
    blobs = tmp_path / "hf" / "models--org--name" / "blobs"
    _write(blobs / "abc123.incomplete", b"half")
    assert _assets(tmp_path, _hf(files=("config.json",))) == "partial"


def test_hf_explicit_revision_via_ref_and_commit_name(tmp_path):
    base = tmp_path / "hf" / "models--org--name"
    _hf_link(tmp_path, "c0ffee", "config.json")
    _write(base / "refs" / "main", b"c0ffee\n")
    assert _assets(tmp_path, _hf(files=("config.json",), revision="main")) == "ready"
    assert _assets(tmp_path, _hf(files=("config.json",), revision="c0ffee")) == "ready"
    assert _assets(tmp_path, _hf(files=("config.json",), revision="v9")) == "missing"


def test_hf_ref_pointing_to_missing_snapshot_is_missing(tmp_path):
    base = tmp_path / "hf" / "models--org--name"
    _hf_link(tmp_path, "old", "config.json")
    _write(base / "refs" / "main", b"new")
    assert _assets(tmp_path, _hf(files=("config.json",), revision="main")) == "missing"


def test_hf_pinned_revision_ignores_stale_snapshot_next_to_partial_one(tmp_path):
    base = tmp_path / "hf" / "models--org--name"
    _hf_link(tmp_path, "old", "config.json")
    _hf_link(tmp_path, "old", "model.safetensors")
    _hf_link(tmp_path, "new", "config.json")  # newer snapshot lacks the weights
    _write(base / "refs" / "main", b"new")
    files = ("config.json", "model.safetensors")
    assert _assets(tmp_path, _hf(files=files, revision="main")) == "partial"
    assert _assets(tmp_path, _hf(files=files, revision="old")) == "ready"


def test_hf_revision_and_ref_are_confined(tmp_path):
    for bad in ("../..", "../x", "/etc", "a/b", "..", ""):
        with pytest.raises(rq.UnsafePathError):
            _hf(revision=bad)
    _hf(revision="refs-pr_1.0")
    outside = tmp_path / "hf" / "models--org--name" / "snapshots"
    _hf_link(tmp_path, "r1", "config.json")
    base = outside.parent
    _write(base / "refs" / "main", b"../../../../etc")
    status = check(
        Requirements(assets=(_hf(files=("passwd",), revision="main"),)), tmp_path
    )
    assert status.state == "failed"
    assert status.layers["assets"] == "invalid"
    # a snapshot that is a symlink leaving the cache is rejected too
    evil = tmp_path / "elsewhere"
    _write(evil / "config.json")
    (outside / "evil").symlink_to(evil)
    status = check(
        Requirements(assets=(_hf(files=("config.json",), revision="evil"),)), tmp_path
    )
    assert status.layers["assets"] == "invalid"


def test_describe_source_per_kind():
    from det_depth_distance.core import requirements as rq

    assert (
        rq.UrlAsset(
            key="w", target="a", url="https://host.example/x.pt"
        ).describe_source()
        == "host.example"
    )
    assert (
        rq.HuggingFaceAsset(key="w", target="hf", repo="org/m").describe_source()
        == "Hugging Face · org/m"
    )
    assert (
        rq.UserProvided(
            key="w", target="a", option="weights", hint="provide it"
        ).describe_source()
        == "provide it"
    )
    assert (
        rq.CodeRepo(name="t", url="https://h/t.git", commit="b" * 40).describe_source()
        == "git · h/t.git"
    )
    # host and path only: never the URL's credentials or query
    leaky = rq.CodeRepo(
        name="t", url="https://user:s3cret@h:8443/o/t.git?token=abc", commit="b" * 40
    )
    assert leaky.describe_source() == "git · h:8443/o/t.git"

    @dataclass(frozen=True, kw_only=True)
    class Bare(rq.Asset):
        def present(self, root):
            return rq.AssetState("ready")

    assert Bare(key="k", target="t").describe_source() == "Bare"


def test_redact_root_boundaries_include_markup_and_template_characters(tmp_path):
    root = tmp_path / "weights"
    for wrapped in (f"`{root}`", f"<{root}>", f"{{{root}}}", f"`{root}/x.pt`"):
        out = rq.redact_root(f"see {wrapped} now", root, "<R>")
        assert str(root) not in out, wrapped
    assert rq.redact_root(f"`{root}-old`", root, "<R>") == f"`{root}-old`"


def test_weights_root_problem_refuses_unsafe_roots(tmp_path, monkeypatch):
    project = Path(rq.__file__).resolve().parents[3]
    assert rq.weights_root_problem(Path("/")) is not None
    assert "home folder" in rq.weights_root_problem(Path.home())
    assert "project folder" in rq.weights_root_problem(project)
    assert "project folder" in rq.weights_root_problem(project.parent)
    venv = tmp_path / "venv"
    monkeypatch.setattr(sys, "prefix", str(venv))
    assert "Python environment" in rq.weights_root_problem(tmp_path)
    out = tmp_path / "x" / "outputs"
    assert "outputs" in rq.weights_root_problem(tmp_path / "x", (out,))
    # Dedicated folders are fine, inside the project too (the default).
    assert rq.weights_root_problem(tmp_path / "weights", (out,)) is None
    assert rq.weights_root_problem(project / "weights") is None


# --- Google Drive kind (task_0028) ------------------------------------------


def test_google_drive_asset_present_follows_the_url_rules(tmp_path):
    data = b"drive weights"
    sha = hashlib.sha256(data).hexdigest()
    asset = rq.GoogleDriveAsset(
        key="weights",
        target="m/w.pth",
        file_id="1N7jNsa8P1dM-UevqPRsx6heUpN4L00FJ",
        size_bytes=len(data),
        sha256=sha,
    )
    assert rq.asset_kinds()["gdrive"] is rq.GoogleDriveAsset
    assert asset.downloadable and asset.known is False  # fails closed
    assert asset.describe_source() == (
        "Google Drive file 1N7jNsa8P1dM-UevqPRsx6heUpN4L00FJ"
    )
    assert asset.present(tmp_path).state == "missing"
    path = tmp_path / "m/w.pth"
    path.parent.mkdir()
    path.with_name("w.pth.part").write_bytes(data)
    assert asset.present(tmp_path).state == "partial"  # a .part is never ready
    path.write_bytes(b"x" * len(data))  # same size, another file: the user's
    state = asset.present(tmp_path)
    assert state.state == "invalid" and "different sha256" in state.detail
    path.write_bytes(b"short")
    assert "different size" in asset.present(tmp_path).detail
    path.write_bytes(data)
    assert asset.present(tmp_path).state == "ready"


@pytest.mark.parametrize(
    "file_id",
    [
        "",
        "short",
        "1N7jNsa8P1dM?x=1",
        "a/b/cccccccccc",
        "https://x.y/zzzzzzzz",
        "-" * 9,
    ],
)
def test_google_drive_file_id_is_validated(file_id):
    with pytest.raises(rq.UnsafePathError, match="Google Drive file id"):
        rq.GoogleDriveAsset(key="w", target="w.pth", file_id=file_id)
