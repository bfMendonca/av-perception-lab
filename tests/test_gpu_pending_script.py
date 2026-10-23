"""utils/gpu_pending.sh checked locally: syntax, step order, a dry run that
prints every command and executes, creates and downloads nothing, and real
runs of a copy of the script against a fake uv (statuses, abort, download
gating); the run with the real uv only happens on the GPU host."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "utils" / "gpu_pending.sh"
STEP_IDS = (
    "01-sync",
    "02-env",
    "03-unit",
    "04-regression",
    "05-gpu",
    "06-plots",
    "07-inspect",
    "08-nocal",
    "09-fusion",
    "10-track",
    "11-track-variants",
    "12-dl-0004",
    "13-vocab-seg",
    "14-gpu-seg",
    "15-boxmask",
    "16-dl-0005",
    "17-gpu-refiners",
    "18-refiner-runs",
    "19-calibrated",
    "20-motion",
)
needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _script(
    *args: str,
    uv: str = "uv-must-not-run",
    script: Path = SCRIPT,
    home: str = "/nonexistent",
) -> subprocess.CompletedProcess[str]:
    # UV points at a missing binary: a dry run that executed uv would fail.
    return subprocess.run(
        ["bash", str(script), *args],
        check=False,
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "HOME": home, "UV": uv},
        stdin=subprocess.DEVNULL,
    )


# A uv that records its arguments and fails when they contain $FAIL_ON.
FAKE_UV = """#!/usr/bin/env bash
echo "$*" >> "$(dirname "$0")/calls.log"
case " $* " in *"FAIL_ON"*) exit 7 ;; esac
"""


@pytest.fixture
def fake_repo(tmp_path):
    """Factory: (script copy in a scratch repo root, fake uv failing on
    `fail_on`, its calls log). The copy writes its logs under the scratch
    root, never under the real repository."""

    def make(fail_on: str) -> tuple[Path, Path, Path]:
        root = tmp_path / "repo"
        (root / "utils").mkdir(parents=True)
        (root / "weights" / "hf").mkdir(parents=True)
        script = root / "utils" / "gpu_pending.sh"
        shutil.copy(SCRIPT, script)
        uv = tmp_path / "bin" / "uv"
        uv.parent.mkdir()
        uv.write_text(FAKE_UV.replace("FAIL_ON", fail_on))
        uv.chmod(0o755)
        return script, uv, uv.parent / "calls.log"

    return make


def _summary(script: Path) -> list[list[str]]:
    (log,) = (script.parents[1] / "_testbed" / "outputs").glob("gpu_pending_*[0-9]")
    lines = (log / "summary.tsv").read_text().splitlines()
    return [line.split("\t") for line in lines[1:]]


@needs_bash
def test_syntax():
    out = subprocess.run(
        ["bash", "-n", str(SCRIPT)], check=False, capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr


@needs_bash
def test_list_is_in_dependency_order():
    out = _script("list")
    assert out.returncode == 0, out.stderr
    ids = tuple(line.split()[0] for line in out.stdout.splitlines())
    assert ids == STEP_IDS
    # Every listed step has its function.
    text = SCRIPT.read_text()
    assert all(f"step_{i.replace('-', '_')}()" in text for i in ids)


@needs_bash
def test_dry_run_executes_and_downloads_nothing():
    outputs = REPO / "_testbed" / "outputs"
    before = set(outputs.glob("gpu_pending_*")) if outputs.is_dir() else set()
    out = _script("run", "--dry-run", "--allow-downloads", "--sam-hq", "base")
    assert out.returncode == 0, out.stderr
    after = set(outputs.glob("gpu_pending_*")) if outputs.is_dir() else set()
    assert after == before  # no log folder in a dry run
    text = out.stdout
    assert re.findall(r"^=== (\S+)", text, re.MULTILINE) == list(STEP_IDS)
    assert "+ uv-must-not-run sync --extra infer" in text
    assert "--basetemp=" in text and "_pytest/" in text
    # A dry run never prompts and only prints the downloads (HOME has
    # no weights, so the steps needing them skip).
    assert "[s/N]" not in text
    assert "snapshot_download" in text and "sam-hq-vit-base" in text
    assert "MANUAL: task_0004 downloads are never run" in text
    assert (
        "SKIP: missing" in text
        and "uv-must-not-run: command not found" not in out.stderr
    )


@needs_bash
def test_dry_run_selection_and_bad_options():
    out = _script("run", "--dry-run", "--from", "17", "--skip-long")
    ids = re.findall(r"^=== (\S+)", out.stdout, re.MULTILINE)
    assert ids == ["17-gpu-refiners", "18-refiner-runs", "19-calibrated", "20-motion"]
    out = _script("run", "--dry-run", "--only", "10-track", "--skip-long")
    assert re.findall(r"^=== (\S+)", out.stdout, re.MULTILINE) == ["10-track"]
    assert "SKIP: --skip-long" in out.stdout
    out = _script("run", "--dry-run", "--only", "16")
    assert "SKIP: downloads need --allow-downloads" in out.stdout
    assert _script("run", "--sam-hq", "large").returncode == 2
    assert _script("run", "--bogus").returncode == 2
    assert _script().returncode == 1


@needs_bash
def test_run_reports_a_failure_in_the_middle_of_a_step(fake_repo):
    # 07-inspect runs two commands: the first fails, the second passes. The
    # step must be FAIL (not hidden by the last command) and both must run.
    script, uv, calls = fake_repo("IMG_4021")
    out = _script("run", "--only", "07", uv=str(uv), script=script)
    assert out.returncode == 1, out.stderr
    assert _summary(script) == [["07-inspect", "0007a.3", "FAIL", "7", "0"]]
    assert len(calls.read_text().splitlines()) == 2
    assert "FAILED (exit 7)" in out.stdout  # step output (stderr too) is teed


@needs_bash
def test_run_aborts_when_an_abort_step_fails(fake_repo):
    script, uv, calls = fake_repo("sync")
    out = _script("run", uv=str(uv), script=script)
    assert out.returncode == 1
    assert "aborting: 01-sync failed" in out.stderr
    assert [row[:3] for row in _summary(script)] == [["01-sync", "0002-0007", "FAIL"]]
    assert calls.read_text().splitlines() == ["sync --extra infer"]


@needs_bash
def test_run_without_a_tty_downloads_nothing(fake_repo, tmp_path):
    script, uv, calls = fake_repo("never")
    out = _script(
        "run",
        "--only",
        "16-dl-0005",
        "--allow-downloads",
        "--sam-hq",
        "base",
        "--with-optional",
        uv=str(uv),
        script=script,
        home=str(tmp_path / "home"),
    )
    assert out.returncode == 0, out.stderr
    # Every item is declined: uv is never even started.
    assert "no TTY" in out.stdout and not calls.exists()
    assert [row[:3] for row in _summary(script)] == [["16-dl-0005", "0005.3", "PASS"]]


@needs_bash
def test_skip_status_and_unknown_steps(fake_repo, tmp_path):
    script, uv, _ = fake_repo("never")
    out = _script("run", "--only", "13", uv=str(uv), script=script, home=str(tmp_path))
    assert out.returncode == 0, out.stderr
    assert [row[:3] for row in _summary(script)] == [["13-vocab-seg", "0004.3", "SKIP"]]
    for bad in (("--from", "5"), ("--only", "99-nope")):
        out = _script("run", "--dry-run", *bad, uv=str(uv), script=script)
        assert out.returncode == 2 and "unknown step" in out.stderr
