"""Packages layer (task_0024): detect, dry-run, install — with a fake `uv`.

No test runs the real `uv` (it would rewrite the shared environment): every
call goes to the script of `fake_uv`, which records its arguments.
"""

import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml
from fake_uv import FakeUv, fake_uv, snapshot  # noqa: F401 - fake_uv is a fixture

from det_depth_distance import cli
from det_depth_distance.cli import main
from det_depth_distance.core import registry
from det_depth_distance.core import requirements as rq
from det_depth_distance.core.components import Detector, DetectorOptions
from det_depth_distance.core.provision import (
    ProvisionCancelled,
    ProvisionError,
    install_packages,
    packages,
    with_package_plan,
)
from det_depth_distance.core.provision.packages import PackagePlan
from det_depth_distance.core.requirements import Requirements

MODEL = Requirements(packages=("xyzlib",), extra="model-xyz")
IMPORTS = {"infer": ("fakebase",)}


def _flags(call: dict) -> list[str]:
    args = call["args"]
    return [args[i + 1] for i, a in enumerate(args) if a == "--extra"]


# -- detection -------------------------------------------------------------------


def test_detect_extras_uses_import_names_without_importing(fake_uv):  # noqa: F811
    detected = packages.detect_extras(fake_uv.pyproject, IMPORTS)
    # infer: distribution installed and `fakebase` found; web: distribution
    # with a marker; eval / model-xyz: distributions missing.
    assert detected == {"infer", "web"}
    assert "fakebase" not in sys.modules  # found by find_spec, never imported
    # An import name the components declare but nothing provides: not detected.
    assert packages.detect_extras(fake_uv.pyproject, {"web": ("no_such_mod_x",)}) == {
        "infer"
    }
    assert fake_uv.calls() == []  # detection never calls uv


def test_declared_extras_reads_distribution_names(fake_uv):  # noqa: F811
    assert packages.declared_extras(fake_uv.pyproject) == {
        "infer": ("fakebase-dist",),
        "web": ("fakeweb-dist",),
        "eval": ("fakeeval-dist",),
        "model-xyz": ("xyzlib",),
    }


def test_repo_extras_are_declared():
    extras = packages.declared_extras()
    assert {"infer", "web", "eval"} <= set(extras)
    assert "torch" in extras["infer"]


# -- dry-run ---------------------------------------------------------------------


def test_dry_run_ok_offers_install(fake_uv):  # noqa: F811
    plan = packages.plan_packages(MODEL, imports=IMPORTS)
    assert plan.action == "install"
    assert plan.extra == "model-xyz"
    assert plan.extras == ("infer", "model-xyz", "web")
    assert plan.message == "would install xyzlib==1.0"
    assert plan.command == (
        "uv --color never sync --locked --inexact "
        "--extra infer --extra model-xyz --extra web"
    )
    (call,) = fake_uv.calls()
    assert "--dry-run" in call["args"]
    assert _flags(call) == ["infer", "model-xyz", "web"]
    assert call["cwd"] == str(fake_uv.project)
    # Cached: the status route asks again without another uv call.
    assert packages.plan_packages(MODEL, imports=IMPORTS) == plan
    assert len(fake_uv.calls()) == 1


def test_dry_run_conflict_reports_reason_and_changes_nothing(fake_uv):  # noqa: F811
    fake_uv.mode("conflict")
    before = snapshot(fake_uv.envdir)
    lock_before = snapshot(fake_uv.project)
    result = packages.dry_run("model-xyz", {"infer", "web"}, fake_uv.uv)
    assert result.ok is False
    assert "No solution found when resolving dependencies" in result.message
    assert "extras infer and model-xyz are incompatible" in result.message
    assert "Resolved 3 packages" not in result.message  # progress noise dropped
    # Project-relative: the absolute project folder never appears.
    assert str(fake_uv.project) not in result.message
    assert "(from ./pyproject.toml)" in result.message
    plan = packages.plan_packages(MODEL, imports=IMPORTS, fresh=True)
    assert plan.action == "conflict"
    assert "incompatible" in plan.message
    assert "docs/model_environments.md" in plan.message
    # Nothing changed: the fake would have written files without --dry-run.
    assert snapshot(fake_uv.envdir) == before
    assert snapshot(fake_uv.project) == lock_before
    assert all("--dry-run" in c["args"] for c in fake_uv.calls())
    assert not (fake_uv.site / "xyzlib").exists()


def test_dry_run_that_replaces_packages_is_a_conflict(fake_uv):  # noqa: F811
    fake_uv.mode("replace")
    result = packages.dry_run("model-xyz", {"infer"}, fake_uv.uv)
    assert result.ok is False
    assert "torch==2.8.0" in result.message and "untouched" in result.message
    plan = packages.plan_packages(MODEL, imports=IMPORTS)
    assert plan.action == "conflict"


def test_dry_run_without_uv_is_a_readable_failure(fake_uv, tmp_path):  # noqa: F811
    result = packages.dry_run("model-xyz", {"infer"}, str(tmp_path / "no-uv"))
    assert result.ok is False and "uv is not available" in result.message


def test_uv_missing_is_an_error_not_a_conflict_and_never_cached(
    fake_uv,  # noqa: F811
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("DDD_UV", str(tmp_path / "no-uv"))
    plan = packages.plan_packages(MODEL, imports=IMPORTS)
    assert plan.action == "error"
    assert "DDD_UV" in plan.message and "conflict" not in plan.message
    status = with_package_plan(rq.check(MODEL, tmp_path), plan)
    assert status.state == "failed"
    # uv becomes available: the next check is not shadowed by the failure.
    monkeypatch.setenv("DDD_UV", fake_uv.uv)
    assert packages.plan_packages(MODEL, imports=IMPORTS).action == "install"


def test_dry_run_timeout_is_not_cached_and_kills_a_hanging_grandchild(
    fake_uv,  # noqa: F811
    monkeypatch,
):
    fake_uv.mode("hang-dry")
    # Long enough for the fake uv to start its grandchild and write the pids
    # file under load (1.0 s was flaky), still well under the 10 s bound.
    monkeypatch.setattr(packages, "DRY_RUN_TIMEOUT_S", 3.0)
    start = time.monotonic()
    plan = packages.plan_packages(MODEL, imports=IMPORTS)
    assert (
        time.monotonic() - start < 10
    )  # returned although a grandchild holds the pipe
    assert plan.action == "error" and "timed out" in plan.message
    uv_pid, child_pid = map(int, fake_uv.pids.read_text().split())
    deadline = time.monotonic() + 5
    while not (_gone(uv_pid) and _gone(child_pid)):
        assert time.monotonic() < deadline, "the dry-run's process group survived"
        time.sleep(0.05)
    fake_uv.monkeypatch.setenv(
        "FAKE_UV_MODE", "ok"
    )  # no cache clear: must not be cached
    assert packages.plan_packages(MODEL, imports=IMPORTS).action == "install"


def test_dry_run_cancel_stops_uv(fake_uv):  # noqa: F811
    fake_uv.mode("hang-dry")
    cancel = threading.Event()
    timer = threading.Timer(0.5, cancel.set)
    timer.start()
    result = packages.dry_run("model-xyz", {"infer"}, fake_uv.uv, cancel=cancel)
    timer.join()
    assert result.ok is False and result.kind == "cancelled"


def test_concurrent_checks_share_one_dry_run(fake_uv):  # noqa: F811
    fake_uv.mode("slow")
    results = []
    threads = [
        threading.Thread(
            target=lambda: results.append(
                packages.cached_dry_run(
                    "model-xyz", {"infer", "web"}, fake_uv.uv, fake_uv.pyproject
                )
            )
        )
        for _ in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert len(results) == 4 and all(r.ok for r in results)
    assert len(fake_uv.calls()) == 1  # single flight


def test_scrub_removes_credentials():
    text = (
        "error: Failed to fetch https://alice:s3cr3t@pypi.example.com/simple/x "
        "and https://__token__@host/x?token=abc123&page=2 and "
        "https://h/x?access_token=zzz; Authorization: Bearer abcdefghijkl0123 "
        "hf_ABCDEFGHIJKLMNOPQRSTUV ghp_ABCDEFGHIJKLMNOPQRSTUV"
    )
    out = packages.scrub(text)
    for secret in (
        "alice",
        "s3cr3t",
        "__token__",
        "abc123",
        "zzz",
        "abcdefghijkl0123",
        "hf_ABCD",
        "ghp_ABCD",
    ):
        assert secret not in out, secret
    assert "https://***@pypi.example.com/simple/x" in out
    assert "page=2" in out
    # uv output goes through it (dry-run reasons and install lines).
    assert "s3cr3t" not in packages.relative(text, Path("/nowhere"))


def test_resolver_reason_keeps_the_error_part():
    text = (
        "Resolved 10 packages in 2ms\n\x1b[31merror\x1b[0m: Extras `a` and `b` "
        "are incompatible with the declared conflicts\n"
    )
    assert packages.resolver_reason(text) == (
        "error: Extras `a` and `b` are incompatible with the declared conflicts"
    )
    assert len(packages.resolver_reason("error: " + "x" * 5000)) == (
        packages.MESSAGE_LIMIT
    )


# -- install ---------------------------------------------------------------------


def test_install_command_includes_every_detected_extra(fake_uv, tmp_path):  # noqa: F811
    lines: list[str] = []
    outcome = install_packages(
        MODEL, lines.append, threading.Event(), root=tmp_path / "weights"
    )
    assert outcome.installed is True
    assert outcome.status.layers["packages"] == "ready"
    dry, real = fake_uv.calls()
    assert "--dry-run" in dry["args"] and "--dry-run" not in real["args"]
    assert _flags(real) == ["infer", "model-xyz", "web"]
    assert real["args"][:5] == ["--color", "never", "sync", "--locked", "--inexact"]
    # The environment uv changes is the one that was inspected.
    assert real["env"] == str(fake_uv.envdir)
    assert lines[0].startswith("$ uv --color never sync --locked --inexact")
    assert " + xyzlib==1.0" in lines
    assert (fake_uv.envdir / "installed.txt").exists()


def test_install_never_runs_bare_sync(fake_uv, tmp_path):  # noqa: F811
    # Every path that reaches uv names the new extra, keeps what is installed
    # (--inexact) and never rewrites the lock (--locked).
    fake_uv.mode("conflict")  # dry-run only
    install_packages(MODEL, lambda line: None, threading.Event(), root=tmp_path)
    fake_uv.mode("fail-install")  # dry-run, failing install
    install_packages(MODEL, lambda line: None, threading.Event(), root=tmp_path)
    packages.dry_run("model-xyz", set(), fake_uv.uv)
    fake_uv.mode("ok")  # dry-run, install
    install_packages(MODEL, lambda line: None, threading.Event(), root=tmp_path)
    calls = fake_uv.calls()
    assert len(calls) == 6
    for call in calls:
        args = call["args"]
        assert "sync" in args
        assert "--inexact" in args and "--locked" in args
        assert "model-xyz" in _flags(call)
        assert not {"--no-extra", "--only-group", "--no-default-groups"} & set(args)
    assert packages.sync_command("model-xyz", set(), "uv", dry_run=False) == [
        "uv",
        "--color",
        "never",
        "sync",
        "--locked",
        "--inexact",
        "--extra",
        "model-xyz",
    ]
    with pytest.raises(ValueError, match="invalid extra"):
        packages.sync_command("--no-extra", {"infer"}, "uv", dry_run=True)


def test_failed_install_is_a_readable_reason(fake_uv, tmp_path):  # noqa: F811
    fake_uv.mode("fail-install")
    outcome = install_packages(
        MODEL, lambda line: None, threading.Event(), root=tmp_path
    )
    assert outcome.installed is False and outcome.status.state == "failed"
    assert "uv sync failed" in outcome.status.message
    assert "network unreachable" in outcome.status.message


def test_base_stack_missing_is_reported_with_exact_command_not_installed(
    fake_uv,  # noqa: F811
    tmp_path,
    monkeypatch,
):
    base = Requirements(packages=("fakebase_missing",), extra="infer")
    plan = packages.plan_packages(base)
    assert plan.action == "base"
    assert plan.command == "uv sync --extra infer --extra web"
    outcome = install_packages(
        base, lambda line: None, threading.Event(), root=tmp_path
    )
    assert outcome.installed is False and outcome.plan.action == "base"
    with pytest.raises(ProvisionError, match="never installed"):
        packages.install("infer", {"infer"}, fake_uv.uv)
    # A per-model extra on top of a missing base stack: the base command, no uv.
    fake_uv.pyproject.write_text(
        fake_uv.pyproject.read_text().replace("fakebase-dist==1.0", "absent-dist")
    )
    plan = packages.plan_packages(MODEL, imports=IMPORTS)
    assert plan.action == "base" and plan.command == packages.BASE_COMMAND
    with pytest.raises(ProvisionError, match="base stack is not installed"):
        packages.install("model-xyz", {"web"}, fake_uv.uv)
    assert fake_uv.calls() == []  # uv never ran


def test_undeclared_extra_is_invalid(fake_uv):  # noqa: F811
    req = Requirements(packages=("xyzlib",), extra="model-nope")
    plan = packages.plan_packages(req, imports=IMPORTS)
    assert plan.action == "invalid" and "not declared" in plan.message
    with pytest.raises(ProvisionError, match="not declared"):
        packages.install("model-nope", {"infer"}, fake_uv.uv)
    assert fake_uv.calls() == []


def _gone(pid: int) -> bool:
    try:
        status = Path(f"/proc/{pid}/status").read_text()
    except FileNotFoundError:
        return True
    return "\nState:\tZ" in status  # a zombie is dead, waiting to be reaped


def test_cancel_kills_the_process_group(fake_uv):  # noqa: F811
    fake_uv.mode("hang")
    cancel = threading.Event()
    errors: list[BaseException] = []

    def work():
        try:
            packages.install("model-xyz", {"infer"}, fake_uv.uv, cancel=cancel)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    thread = threading.Thread(target=work)
    thread.start()
    deadline = time.monotonic() + 10
    while not fake_uv.pids.exists() or not fake_uv.pids.read_text():
        assert time.monotonic() < deadline, "the fake uv did not start"
        time.sleep(0.02)
    uv_pid, child_pid = map(int, fake_uv.pids.read_text().split())
    cancel.set()
    thread.join(10)
    assert not thread.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], ProvisionCancelled)
    deadline = time.monotonic() + 5
    while not (_gone(uv_pid) and _gone(child_pid)):
        assert time.monotonic() < deadline, "uv's process group survived the cancel"
        time.sleep(0.05)


def test_install_refuses_a_plan_other_than_the_confirmed_one(fake_uv, tmp_path):  # noqa: F811
    shown = packages.plan_packages(MODEL, imports=IMPORTS)
    assert shown.action == "install"
    other = PackagePlan(
        "install", "model-xyz", "uv sync --extra other", None, ("other",)
    )
    outcome = install_packages(
        MODEL,
        lambda line: None,
        threading.Event(),
        root=tmp_path,
        imports=IMPORTS,
        expected=other,
    )
    assert outcome.installed is False and outcome.status.state == "failed"
    assert (
        "environment changed since the install was confirmed" in outcome.status.message
    )
    base = PackagePlan("base", "infer", packages.BASE_COMMAND)
    outcome = install_packages(
        MODEL,
        lambda line: None,
        threading.Event(),
        root=tmp_path,
        imports=IMPORTS,
        expected=base,
    )
    assert outcome.installed is False
    assert all("--dry-run" in c["args"] for c in fake_uv.calls())  # nothing installed
    # The confirmed plan, computed with the same import names, runs.
    outcome = install_packages(
        MODEL,
        lambda line: None,
        threading.Event(),
        root=tmp_path,
        imports=IMPORTS,
        expected=shown,
    )
    assert outcome.installed is True


def test_engine_maps_conflict_state(fake_uv, tmp_path):  # noqa: F811
    fake_uv.mode("conflict")
    outcome = install_packages(
        MODEL, lambda line: None, threading.Event(), root=tmp_path
    )
    assert outcome.installed is False
    assert outcome.status.state == "conflict"
    assert "incompatible" in outcome.status.message
    assert all("--dry-run" in c["args"] for c in fake_uv.calls())


# -- CLI -------------------------------------------------------------------------


def _model_detector():
    class _NeedsPackages(Detector):
        @dataclass(frozen=True, slots=True, kw_only=True)
        class Options(DetectorOptions):
            extra: str = "model-xyz"

        @classmethod
        def requirements(cls, options):
            package = "xyzlib" if options.extra == "model-xyz" else "fakebase_missing"
            return Requirements(packages=(package,), extra=options.extra)

        @property
        def class_names(self):
            return ["car"]

        def _load(self, ctx):
            raise AssertionError("a model was loaded")

        def _predict(self, rgb):
            raise AssertionError("unreachable")

    _NeedsPackages.type_name = "needs_packages"
    return _NeedsPackages


@pytest.fixture
def packages_config(tmp_path, monkeypatch, fake_uv):  # noqa: F811
    """A config whose only unmet requirement is the fake detector's packages
    (DA3's cache outside the weights root; its packages count as present)."""
    monkeypatch.setenv("DDD_WEIGHTS_ROOT", str(tmp_path / "weights"))
    real = rq._top_level_present
    stack = {"torch", "depth_anything_3"}
    monkeypatch.setattr(rq, "_top_level_present", lambda n: n in stack or real(n))
    registry.load_adapters()
    monkeypatch.setitem(
        registry._REGISTRY["detector"], "needs_packages", _model_detector()
    )
    monkeypatch.setattr(cli.Pipeline, "load", lambda *a, **k: pytest.fail("loaded"))
    monkeypatch.setattr(
        cli.Pipeline, "resolve_intrinsics", lambda *a, **k: pytest.fail("loaded")
    )
    from det_depth_distance.io import source as source_module
    from det_depth_distance.io.video import VideoInfo

    monkeypatch.setattr(
        source_module,
        "probe",
        lambda path: VideoInfo(path, 64, 48, 10.0, "10/1", 10, False, 0),
    )
    video = tmp_path / "clip.mov"
    video.touch()

    def write(extra: str) -> Path:
        path = tmp_path / f"{extra}.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "source": {"video": str(video)},
                    "detectors": [
                        {"type": "needs_packages", "name": "fake", "extra": extra}
                    ],
                    "depth": {"type": "da3", "hf_cache": str(tmp_path / "elsewhere")},
                    "camera": {"focal_px": 100.0, "estimate": None},
                    "output": {"out_root": str(tmp_path / "out")},
                }
            )
        )
        return path

    return write


def test_cli_gate_names_uv_sync_for_missing_base_packages(packages_config, capsys):
    assert main(["--config", str(packages_config("infer"))]) == 2
    err = capsys.readouterr().err
    assert "package fakebase_missing is not installed" in err
    assert "`uv sync --extra infer`" in err
    assert "--prepare` first" not in err  # --prepare does not install packages


def test_cli_gate_names_install_packages_for_a_model_extra(packages_config, capsys):
    assert main(["--config", str(packages_config("model-xyz"))]) == 2
    err = capsys.readouterr().err
    assert "--prepare --install-packages" in err


def test_cli_prepare_reports_packages_and_installs_nothing(
    packages_config,
    fake_uv,  # noqa: F811
    capsys,
):
    path = packages_config("model-xyz")
    assert main(["--config", str(path), "--prepare"]) == 1
    out = capsys.readouterr().out
    assert "can be installed (uv dry-run clean, would install xyzlib==1.0)" in out
    assert "--install-packages" in out
    assert all("--dry-run" in c["args"] for c in fake_uv.calls())
    assert not (fake_uv.envdir / "installed.txt").exists()

    fake_uv.mode("conflict")
    assert main(["--config", str(path), "--prepare"]) == 1
    out = capsys.readouterr().out
    table = {line.split()[0]: line.split() for line in out.splitlines() if line}
    assert table["detectors[0]"][2:4] == ["needs_packages", "conflict"]
    assert "incompatible" in out and "docs/model_environments.md" in out

    base = packages_config("infer")
    assert main(["--config", str(base), "--prepare", "--install-packages"]) == 1
    out = capsys.readouterr().out
    assert "`uv sync --extra infer`" in out and "never installs it" in out
    assert all("--dry-run" in c["args"] for c in fake_uv.calls())


def test_cli_install_packages_runs_dry_run_then_install(
    packages_config,
    fake_uv,  # noqa: F811
    capsys,
):
    path = packages_config("model-xyz")
    assert main(["--config", str(path), "--prepare", "--install-packages"]) == 0
    out = capsys.readouterr().out
    assert "[uv]  + xyzlib==1.0" in out
    assert "extra 'model-xyz' installed" in out
    real = [c for c in fake_uv.calls() if "--dry-run" not in c["args"]]
    assert len(real) == 1 and _flags(real[0]) == ["infer", "model-xyz", "web"]
    assert os.path.exists(fake_uv.envdir / "installed.txt")


def test_install_packages_flag_needs_prepare(tmp_path, capsys):
    with pytest.raises(SystemExit):
        main(["--config", str(tmp_path / "x.yaml"), "--install-packages"])
    assert "--install-packages is used with --prepare" in capsys.readouterr().err


def test_dry_run_plan_messages_are_relative_and_scrubbed(fake_uv):  # noqa: F811
    fake_uv.mode("path-plan")
    result = packages.dry_run(
        "model-xyz", {"infer"}, fake_uv.uv, pyproject=fake_uv.pyproject
    )
    assert result.ok, result
    assert str(fake_uv.project) not in result.message
    assert "./vendor/xyzlib" in result.message
    assert str(Path.home()) not in result.message and "~/wheels" in result.message
    assert "s3cr3t" not in result.message and "alice" not in result.message
    fake_uv.mode("replace")
    replaced = packages.dry_run(
        "model-xyz", {"infer"}, fake_uv.uv, pyproject=fake_uv.pyproject
    )
    assert replaced.kind == "replace" and "would remove or replace" in replaced.message


def test_scrub_leaves_prose_alone_and_catches_more_token_shapes():
    prose = (
        "basic requirements are met; token count 12345678; the bearer of bad "
        "news; Basic authentication docs"
    )
    assert packages.scrub(prose) == prose
    text = (
        "Authorization: Basic dXNlcjpwYXNzd29yZA== and authorization=Token "
        "abcdefgh12345678 and Bearer abcdefghijkl0123 and "
        "github_pat_11ABCDEFG0123456789_abcdefghijklmnop"
    )
    out = packages.scrub(text)
    for secret in ("dXNlcjpw", "abcdefgh12345678", "abcdefghijkl0123", "11ABCDEFG"):
        assert secret not in out, secret
    assert "Authorization: Basic ***" in out and "Bearer ***" in out


def test_transient_dry_run_failure_names_the_reason(fake_uv, tmp_path):  # noqa: F811
    shown = packages.plan_packages(MODEL, imports=IMPORTS)
    assert shown.action == "install"
    outcome = install_packages(
        MODEL,
        lambda line: None,
        threading.Event(),
        root=tmp_path,
        imports=IMPORTS,
        expected=shown,
        uv=str(tmp_path / "no-such-uv"),
    )
    assert outcome.installed is False and outcome.status.state == "failed"
    assert "uv is not available" in outcome.status.message
    assert "(now: error)" not in outcome.status.message
