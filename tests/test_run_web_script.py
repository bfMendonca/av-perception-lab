"""utils/run_web.sh checked with fake uv, node and npm: the dry run prints what
would run, the extras never drop an installed GPU stack, the front-end is built
only when missing, and missing or old tools stop with a clear message."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "utils" / "run_web.sh"
BASH = shutil.which("bash")

pytestmark = pytest.mark.skipif(BASH is None, reason="needs bash")


def _fake(path: Path, body: str) -> None:
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(0o755)


@pytest.fixture
def sandbox(tmp_path):
    """A copy of the script in a fake checkout, with fake tools on PATH."""
    root = tmp_path / "repo"
    (root / "utils").mkdir(parents=True)
    (root / "web").mkdir()
    shutil.copy(SCRIPT, root / "utils" / "run_web.sh")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls.txt"
    for tool in ("uv", "npm"):
        _fake(bin_dir / tool, f'echo "{tool} $@" >> "{calls}"')
    _fake(bin_dir / "node", 'echo "v22.12.0"')
    return root, bin_dir, calls


def _run(sandbox, *args, env_dir=None, path=None):
    root, bin_dir, _ = sandbox
    env = {
        "PATH": path if path is not None else f"{bin_dir}:/usr/bin:/bin",
        "UV_PROJECT_ENVIRONMENT": str(env_dir or root / "no-env"),
    }
    return subprocess.run(
        [BASH, str(root / "utils" / "run_web.sh"), *args],
        capture_output=True,
        check=False,
        text=True,
        env=env,
        cwd="/",
    )


def _with_dist(root: Path) -> None:
    (root / "web" / "dist").mkdir()
    (root / "web" / "dist" / "index.html").write_text("<html></html>")


def _with_torch(tmp_path: Path) -> Path:
    env = tmp_path / "env"
    (env / "lib" / "python3.11" / "site-packages" / "torch").mkdir(parents=True)
    return env


def test_syntax():
    assert subprocess.run([BASH, "-n", str(SCRIPT)], check=False).returncode == 0


def test_help_lists_the_options():
    proc = subprocess.run(
        [BASH, str(SCRIPT), "--help"], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0
    for option in ("--gpu", "--rebuild", "--dry-run"):
        assert option in proc.stdout


def test_dry_run_builds_the_front_end_when_dist_is_missing(sandbox):
    proc = _run(sandbox, "--dry-run")
    assert proc.returncode == 0, proc.stderr
    assert "+ (cd web && npm ci)" in proc.stdout
    assert "+ (cd web && npm run build)" in proc.stdout
    assert "+ uv run --extra web av-perception-lab-web" in proc.stdout
    assert "infer" not in proc.stdout.split("+ uv run")[1]


def test_no_build_when_dist_exists(sandbox):
    _with_dist(sandbox[0])
    proc = _run(sandbox, "--dry-run")
    assert proc.returncode == 0, proc.stderr
    assert "npm" not in proc.stdout


def test_rebuild_forces_the_build(sandbox):
    _with_dist(sandbox[0])
    proc = _run(sandbox, "--dry-run", "--rebuild")
    assert "+ (cd web && npm run build)" in proc.stdout


def test_installed_gpu_stack_is_kept(sandbox, tmp_path):
    _with_dist(sandbox[0])
    proc = _run(sandbox, "--dry-run", env_dir=_with_torch(tmp_path))
    assert "keeping it" in proc.stdout
    assert "+ uv run --extra infer --extra web av-perception-lab-web" in proc.stdout


def test_gpu_flag_selects_the_infer_extra_and_says_what_it_installs(sandbox):
    _with_dist(sandbox[0])
    proc = _run(sandbox, "--dry-run", "--gpu")
    assert "about 9 GB" in proc.stdout
    assert "--extra infer --extra web" in proc.stdout


def test_server_options_pass_through_and_show_in_the_url(sandbox):
    _with_dist(sandbox[0])
    proc = _run(sandbox, "--dry-run", "--port", "8010", "--host", "0.0.0.0")
    assert "http://0.0.0.0:8010" in proc.stdout
    assert "av-perception-lab-web --port 8010 --host 0.0.0.0" in proc.stdout


def test_dry_run_executes_nothing(sandbox):
    _with_dist(sandbox[0])
    _run(sandbox, "--dry-run", "--rebuild")
    assert not sandbox[2].exists()


def test_real_run_calls_uv_with_the_extras(sandbox):
    _with_dist(sandbox[0])
    proc = _run(sandbox, "--port", "8010")
    assert proc.returncode == 0, proc.stderr
    assert sandbox[2].read_text().strip() == (
        "uv run --extra web av-perception-lab-web --port 8010"
    )


def test_old_node_stops_when_a_build_is_needed(sandbox):
    _fake(sandbox[1] / "node", 'echo "v20.11.0"')
    proc = _run(sandbox, "--dry-run")
    assert proc.returncode == 1
    assert "22.12" in proc.stderr and "v20.11.0" in proc.stderr


def test_old_node_is_fine_when_nothing_is_built(sandbox):
    _fake(sandbox[1] / "node", 'echo "v20.11.0"')
    _with_dist(sandbox[0])
    assert _run(sandbox, "--dry-run").returncode == 0


def _lean_path(sandbox, tmp_path, without):
    """A PATH with only dirname and the fake tools except `without`, so a real
    uv or node on the machine cannot be found."""
    lean = tmp_path / "lean"
    lean.mkdir()
    (lean / "dirname").symlink_to(shutil.which("dirname"))
    for tool in ("uv", "npm", "node"):
        if tool != without:
            (lean / tool).symlink_to(sandbox[1] / tool)
    return str(lean)


def test_missing_node_stops_when_a_build_is_needed(sandbox, tmp_path):
    proc = _run(sandbox, "--dry-run", path=_lean_path(sandbox, tmp_path, "node"))
    assert proc.returncode == 1
    assert "Node.js" in proc.stderr


def test_missing_uv_stops_with_the_install_link(sandbox, tmp_path):
    proc = _run(sandbox, "--dry-run", path=_lean_path(sandbox, tmp_path, "uv"))
    assert proc.returncode == 1
    assert "docs.astral.sh/uv" in proc.stderr


def test_script_is_executable():
    assert os.access(SCRIPT, os.X_OK)
