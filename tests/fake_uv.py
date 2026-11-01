"""A fake `uv` executable for the packages-layer tests (task_0024).

The real `uv sync` would rewrite the shared environment, so no test ever runs
it: `fake_uv` writes a Python script that records every call (arguments, cwd,
`UV_PROJECT_ENVIRONMENT`) to a JSON-lines log and behaves per `mode`:

- `ok`: a dry-run plans `+ xyzlib==1.0`; an install writes the `xyzlib`
  package into the fake site folder and a marker into the environment dir;
- `conflict`: prints a resolver error (naming the project's absolute path)
  and exits 1; *without* `--dry-run` it would also change the environment,
  so a test can tell a dry-run from an install;
- `replace`: a clean dry-run that would replace `torch`;
- `fail-install`: clean dry-run, failing install;
- `hang`: clean dry-run; the install spawns a child, writes both pids and
  sleeps (for the cancel test);
- `slow`: a clean dry-run that takes 0.6 s (single-flight test);
- `hang-dry`: the dry-run spawns a grandchild holding the output pipe and
  sleeps (timeout test).

The project has a `pyproject.toml` with extras `infer` (the base stack:
distribution `fakebase-dist`, import name `fakebase`, whose import raises),
`web` (`fakeweb-dist`), `eval` (not installed) and `model-xyz` (`xyzlib`, not
installed). The fake site folder is put on `sys.path`.
"""

import json
import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from det_depth_distance.core.provision import packages

SCRIPT = r"""
import json, os, subprocess, sys, time
from pathlib import Path

args = sys.argv[1:]
with open(os.environ["FAKE_UV_LOG"], "a") as f:
    f.write(json.dumps({
        "args": args,
        "cwd": os.getcwd(),
        "env": os.environ.get("UV_PROJECT_ENVIRONMENT"),
    }) + "\n")
mode = os.environ.get("FAKE_UV_MODE", "ok")
dry = "--dry-run" in args
envdir = Path(os.environ["UV_PROJECT_ENVIRONMENT"])


def install():
    pkg = Path(os.environ["FAKE_UV_SITE"]) / "xyzlib"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("")
    (envdir / "installed.txt").write_text("xyzlib")


def clean_plan():
    print("Resolved 3 packages in 1ms")
    print("Would install 1 package")
    print(" + xyzlib==1.0")


if mode == "conflict":
    if not dry:
        install()
    print("Resolved 3 packages in 1ms", file=sys.stderr)
    print("  × No solution found when resolving dependencies:", file=sys.stderr)
    print(
        "  ╰─▶ Because model-xyz depends on torch==2.9.0 and infer "
        f"depends on torch==2.8.0 (from {os.getcwd()}/pyproject.toml), extras "
        "infer and model-xyz are incompatible.",
        file=sys.stderr,
    )
    sys.exit(1)
if mode == "slow" and dry:
    time.sleep(0.6)
    clean_plan()
    sys.exit(0)
if mode == "hang-dry" and dry:
    # a grandchild that keeps uv's output pipe open, like a stuck git/build
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    Path(os.environ["FAKE_UV_PIDS"]).write_text(f"{os.getpid()} {child.pid}")
    time.sleep(120)
if mode == "replace" and dry:
    print("Would uninstall 1 package")
    print(" - torch==2.8.0")
    print(" + torch==2.9.0")
    sys.exit(0)
if dry:
    clean_plan()
    sys.exit(0)
if mode == "fail-install":
    print("error: Failed to download `xyzlib==1.0`: network unreachable", file=sys.stderr)
    sys.exit(2)
if mode == "hang":
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    Path(os.environ["FAKE_UV_PIDS"]).write_text(f"{os.getpid()} {child.pid}")
    print("Downloading xyzlib", flush=True)
    time.sleep(120)
    sys.exit(0)
print("Resolved 3 packages in 1ms")
print("Installed 1 package in 2ms")
print(" + xyzlib==1.0")
install()
"""

PYPROJECT = """\
[project]
name = "fake-project"
version = "0"

[project.optional-dependencies]
infer = ["fakebase-dist==1.0"]
web = ["fakeweb-dist; python_version >= '3'"]
eval = ["fakeeval-dist>=1"]
model-xyz = ["xyzlib==1.0"]
"""


@dataclass
class FakeUv:
    uv: str
    log: Path
    project: Path
    pyproject: Path
    site: Path
    envdir: Path
    pids: Path
    monkeypatch: pytest.MonkeyPatch

    def mode(self, name: str) -> None:
        self.monkeypatch.setenv("FAKE_UV_MODE", name)
        packages.clear_cache()

    def calls(self) -> list[dict]:
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text().splitlines()]


def _dist(site: Path, name: str) -> None:
    info = site / f"{name.replace('-', '_')}-1.0.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n"
    )


@pytest.fixture
def fake_uv(tmp_path, monkeypatch) -> FakeUv:
    base = tmp_path / "fakeuv"
    project = base / "project"
    project.mkdir(parents=True)
    pyproject = project / "pyproject.toml"
    pyproject.write_text(PYPROJECT)
    site = base / "site"
    site.mkdir()
    _dist(site, "fakebase-dist")
    _dist(site, "fakeweb-dist")
    (site / "fakebase").mkdir()
    (site / "fakebase" / "__init__.py").write_text(
        "raise RuntimeError('fakebase must never be imported')\n"
    )
    envdir = base / "env"
    (envdir / "lib").mkdir(parents=True)
    (envdir / "lib" / "existing.txt").write_text("base stack")
    script = base / "uv"
    script.write_text(f"#!{sys.executable}\n{SCRIPT}")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    log = base / "calls.jsonl"
    pids = base / "pids.txt"
    monkeypatch.syspath_prepend(str(site))
    monkeypatch.setenv("FAKE_UV_LOG", str(log))
    monkeypatch.setenv("FAKE_UV_SITE", str(site))
    monkeypatch.setenv("FAKE_UV_PIDS", str(pids))
    monkeypatch.setenv("FAKE_UV_MODE", "ok")
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", str(envdir))
    monkeypatch.setenv("DDD_UV", str(script))
    monkeypatch.setattr(packages, "default_pyproject", lambda: pyproject)
    packages.clear_cache()
    yield FakeUv(str(script), log, project, pyproject, site, envdir, pids, monkeypatch)
    packages.clear_cache()
    sys.modules.pop("xyzlib", None)


def snapshot(folder: Path) -> dict[str, tuple[int, int, bytes]]:
    """Every file under `folder` with its size, mtime and content."""
    out = {}
    for path in sorted(folder.rglob("*")):
        st = path.stat()
        content = path.read_bytes() if path.is_file() else b""
        out[os.path.relpath(path, folder)] = (st.st_size, st.st_mtime_ns, content)
    return out
