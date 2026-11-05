"""Command-line entry point: av-perception-lab --config run.yaml [VIDEO] [--set k=v]...

Composition root: builds every component from the YAML (no model loaded
yet, so configuration errors exit with code 2 in seconds), probes the video,
composes the camera (calibration, lens check, undistortion: camera.build_camera),
then runs source -> pipeline -> sinks. `--list-models` prints the registered adapters
and their options. `--prepare` builds the components, obtains their declared
requirements (weights, pinned code) under the weights root, reports the
packages layer (base stack command, clean dry-run or conflict; installs a
per-model extra only with `--install-packages`), prints a status table and
exits (0 only when every component is ready). torch is imported only through
runtime.DeviceMonitor.

Writes <output.out_root>/<timestamp>_<video>_<detectors>_<depth>[_tag]/ with
video.mp4, detections.csv, metrics.json, trajectories.jsonl (with motion)
and optionally depth.npz (see io.sinks).
"""

import argparse
import dataclasses
import sys
import threading
import time
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np

from av_perception_lab.adapters.depth.da3 import da3_requirements
from av_perception_lab.camera import build_camera, focal_estimate_config
from av_perception_lab.config import RunConfig, load_config
from av_perception_lab.core import registry
from av_perception_lab.core.options import ConfigError
from av_perception_lab.core.pipeline import Pipeline
from av_perception_lab.core.provision import (
    install_packages,
    prepare,
    with_package_plan,
)
from av_perception_lab.core.provision import packages as pkgs
from av_perception_lab.core.requirements import (
    ModelStatus,
    Requirements,
    check,
    redact_root,
    weights_root,
    weights_root_problem,
)
from av_perception_lab.io.sinks import (
    CsvSink,
    MetricsSink,
    NpzSink,
    RunContext,
    TrajectorySink,
    VideoSink,
    make_run_dir,
)
from av_perception_lab.io.source import FrameSource, McapSource, VideoSource
from av_perception_lab.motion import build_motion
from av_perception_lab.runtime import DeviceMonitor


def build_parser() -> argparse.ArgumentParser:
    """CLI arguments: the YAML config, an optional video, overrides, the model list."""
    p = argparse.ArgumentParser(
        prog="av-perception-lab",
        description="Run the configured detectors and metric depth model on every "
        "frame of a video and render each box with its estimated distance to the camera.",
    )
    p.add_argument(
        "video", nargs="?", type=Path, help="input video; overrides source.video"
    )
    p.add_argument("--config", type=Path, help="run configuration YAML (see configs/)")
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config value, e.g. source.max_frames=60 or "
        "detectors.0.threshold=0.3 (YAML value; repeatable)",
    )
    p.add_argument(
        "--list-models",
        action="store_true",
        help="list the registered detectors, depth models, aggregators, trackers, "
        "mask refiners, semantic segmenters, motion models and ego-motion "
        "providers with their options",
    )
    p.add_argument(
        "--prepare",
        action="store_true",
        help="download the weights and code every configured model needs "
        "(under AVPL_WEIGHTS_ROOT, default weights/), print their status and exit; "
        "exit code 0 only when all are ready; Python packages are reported "
        "(command, dry-run), not installed",
    )
    p.add_argument(
        "--install-packages",
        action="store_true",
        help="with --prepare: install a model's own pyproject extra when its "
        "uv dry-run is clean (every installed extra is kept; the base stack "
        f"is never installed: `{pkgs.CLI_BASE_COMMAND}`)",
    )
    return p


def _options_text(options_cls: type) -> str:
    parts = []
    for f in dataclasses.fields(options_cls):
        if f.default is not dataclasses.MISSING:
            parts.append(f"{f.name}={f.default!r}")
        elif f.default_factory is not dataclasses.MISSING:
            parts.append(f"{f.name}={f.default_factory()!r}")
        else:
            parts.append(f"{f.name} (required)")
    return ", ".join(parts) or "(no options)"


def list_models() -> str:
    """Registered adapters by family, one line each with option defaults."""
    lines = []
    for kind, classes in registry.available().items():
        lines.append(f"{kind}:")
        lines += [
            f"  {name}: {_options_text(cls.Options)}" for name, cls in classes.items()
        ]
    return "\n".join(lines)


def run(
    cfg: RunConfig, on_frame: Callable[[int, np.ndarray], None] | None = None
) -> Path:
    """Process the configured video and return the run folder.

    `on_frame(index, image)` is called with each composed (RGB) video frame
    just before it is written (progress and previews); None changes nothing.

    Raises:
        ConfigError: invalid component specs, a model that is not ready
            (`require_ready`), unusable calibration, missing intrinsics, or
            no frames.
    """
    # Every component is built (options validated) before the video is probed
    # and before any model is loaded; the camera (calibration file, lens check)
    # needs the probed video, so it comes next, still before any model.
    detectors = [
        registry.build("detector", spec, path=f"detectors[{i}]")
        for i, spec in enumerate(cfg.detectors)
    ]
    depth = registry.build("depth", cfg.depth, path="depth")
    aggregator = registry.build("aggregator", cfg.aggregation, path="aggregation")
    refiners = [
        registry.build("refiner", spec, path=f"refiners[{i}]")
        for i, spec in enumerate(cfg.refiners)
    ]
    semantic = [
        registry.build("semantic", spec, path=f"semantic[{i}]")
        for i, spec in enumerate(cfg.semantic)
    ]
    tracker = (
        registry.build("tracker", cfg.tracking, path="tracking")
        if cfg.tracking is not None
        else None
    )
    motion = (
        build_motion(cfg.motion, distance_type=cfg.distance.distance_type)
        if cfg.motion is not None
        else None
    )
    input_source: FrameSource = (
        McapSource(cfg.source)
        if cfg.source.mcap is not None
        else VideoSource(cfg.source)
    )
    camera = build_camera(cfg.camera, input_source)
    source = camera.source
    pipeline = Pipeline(
        detectors,
        depth,
        aggregator,
        cfg.distance,
        exclude_sky=cfg.exclude_sky,
        intrinsics=camera.intrinsics,
        device=cfg.device,
        agreement=cfg.agreement,
        region=cfg.region,
        refiners=refiners,
        semantic=semantic,
        exclude_semantic=cfg.exclude_semantic,
        tracker=tracker,
        smoothing=cfg.smoothing,
        motion=motion,
    )
    # Every configuration error comes first; then every declared model must be
    # present before any of them (the focal model included) is loaded.
    require_ready(cfg)
    monitor = DeviceMonitor(cfg.device)
    with pipeline, ExitStack() as sinks_stack:
        info = pipeline.resolve_intrinsics(source)
        if info.focal_frame_px:
            w, h = source.frame_size
            print(
                f"[focal] {info.focal_frame_px:.1f} px at {w}x{h} ({info.focal_source})"
            )
        # Baseline after the focal model is released, before the main models load.
        monitor.start()
        pipeline.load(source, info)
        names = [d.name for d in detectors]
        run_dir = make_run_dir(
            cfg.output.out_root, [source.name, *names, depth.name, cfg.output.tag]
        )
        g = depth.geometry
        print(
            f"[run] {source.frame_size[0]}x{source.frame_size[1]}"
            f" @ {source.nominal_fps:.2f} fps"
            f" | {', '.join(names)} | {depth.name} ({g.map_w}x{g.map_h}) -> {run_dir}",
            flush=True,
        )
        video = VideoSink(cfg.render, cfg.distance, on_frame)
        metrics = MetricsSink(
            monitor, lambda: {"render": video.render_ms}, describe=pipeline.describe
        )
        sinks = [CsvSink(), video, *([NpzSink()] if cfg.output.save_depth else [])]
        if motion is not None:
            sinks.append(TrajectorySink())
        sinks.append(metrics)
        ctx = RunContext(
            run_dir, source, info, pipeline.describe(), cfg, camera.details
        )
        for sink in sinks:
            sink.open(ctx)
            sinks_stack.callback(sink.close)
        t_start = time.perf_counter()
        for frame in source:
            result = pipeline.process(frame)
            for sink in sinks:
                sink.write(result)
            if metrics.n_frames % 100 == 0:
                elapsed = time.perf_counter() - t_start
                print(f"  {metrics.n_frames} frames in {elapsed:.1f} s", flush=True)
        if metrics.n_frames == 0:
            raise ConfigError(
                f"no frames decoded from {source.input_name} "
                "(is source.start past the end of the video?)"
            )
    wall = time.perf_counter() - t_start
    print(
        f"[done] {metrics.n_frames} frames, {metrics.n_boxes} boxes "
        f"({metrics.n_without_distance} n/a) in {wall:.1f} s -> {run_dir}"
    )
    return run_dir


def config_components(cfg: RunConfig) -> list[tuple[str, Any]]:
    """`(path, component)` of every Component the config declares (options
    validated, nothing loaded); raises ConfigError like `run`."""
    found: list[tuple[str, Any]] = [
        (f"detectors[{i}]", registry.build("detector", spec, path=f"detectors[{i}]"))
        for i, spec in enumerate(cfg.detectors)
    ]
    found.append(("depth", registry.build("depth", cfg.depth, path="depth")))
    found += [
        (f"refiners[{i}]", registry.build("refiner", spec, path=f"refiners[{i}]"))
        for i, spec in enumerate(cfg.refiners)
    ]
    found += [
        (f"semantic[{i}]", registry.build("semantic", spec, path=f"semantic[{i}]"))
        for i, spec in enumerate(cfg.semantic)
    ]
    if cfg.tracking is not None:
        found.append(
            ("tracking", registry.build("tracker", cfg.tracking, path="tracking"))
        )
    return found


def config_requirements(
    cfg: RunConfig,
) -> list[tuple[str, str, str, Requirements | str]]:
    """`(path, name, type, requirements)` of every model the config would load:
    each component, plus the focal model of `camera.estimate` when the depth
    model needs intrinsics and the focal is estimated. `requirements` is the
    error text when the declaration is unsafe (UnsafePathError). Nothing is
    loaded; raises ConfigError like `run`."""
    rows: list[tuple[str, str, str, Requirements | str]] = []
    depth_needs_intrinsics = False
    for path, component in config_components(cfg):
        if path == "depth":
            depth_needs_intrinsics = component.needs_intrinsics
        try:
            req: Requirements | str = type(component).requirements(component.options)
        except ValueError as exc:  # unsafe declaration (UnsafePathError)
            req = str(exc)
        rows.append((path, component.name, component.type_name, req))
    estimate = focal_estimate_config(cfg.camera)
    if depth_needs_intrinsics and estimate is not None:
        try:
            req = da3_requirements(estimate.model, estimate.hf_cache)
        except ValueError as exc:
            req = str(exc)
        rows.append(("camera.estimate", estimate.model, "da3", req))
    return rows


def require_ready(cfg: RunConfig, root: Path | None = None) -> None:
    """Raise ConfigError naming `--prepare` when a model of `cfg` is not ready
    under the weights root (what is missing, root-relative). Files at explicit
    paths outside the root are not declared, so they are not checked here (the
    adapter checks them when it loads)."""
    root = weights_root() if root is None else Path(root)
    problems = []
    files = base = extras = False
    for path, name, type_name, req in config_requirements(cfg):
        if isinstance(req, str):
            problems.append(f"{path} {name} ({type_name}): {req}")
            files = True
            continue
        status = check(req, root)
        if status.state != "ready":
            why = status.message or ", ".join(status.missing) or status.state
            problems.append(f"{path} {name} ({type_name}): {why}")
            pkg = [m for m in status.missing if m.startswith("package:")]
            files = files or len(pkg) < len(status.missing)
            if pkg and (req.extra is None or req.extra in pkgs.BASE_EXTRAS):
                base = True
            elif pkg:
                extras = True
    if problems:
        details = "\n  ".join(redact_root(p, root) for p in problems)
        hints = []
        if files or not (base or extras):
            hints.append("run `av-perception-lab --config <yaml> --prepare` first")
        if base:
            hints.append(
                "install the base Python stack once: "
                f"`{pkgs.CLI_BASE_COMMAND}` (add `--extra web` for the web "
                "server); --prepare does not install packages"
            )
        if extras:
            hints.append(
                "install the model's packages: `av-perception-lab --config "
                "<yaml> --prepare --install-packages` (a uv dry-run runs first)"
            )
        raise ConfigError(
            f"{len(problems)} model(s) not ready under the weights root:\n  "
            f"{details}\n" + "\n".join(hints)
        )


def _progress_printer(min_interval_s: float = 1.0):
    """progress(label, done, total) printing at most once per second per item."""
    last: dict[str, float] = {}

    def show(label: str, done: int, total: int | None) -> None:
        now = time.monotonic()
        final = total is not None and done >= total
        if label in last and not final and now - last[label] < min_interval_s:
            return
        last[label] = now
        mb = done / 1e6
        size = f"{mb:.1f}/{total / 1e6:.1f} MB" if total else f"{mb:.1f} MB"
        print(f"  [prepare] {label}: {size}", flush=True)

    return show


def _packages_note(label: str, plan: pkgs.PackagePlan) -> str | None:
    """What `--prepare` says about a packages layer it did not install."""
    if plan.action == "base":
        return (
            f"{label}: install the base Python stack once: "
            f"`{pkgs.CLI_BASE_COMMAND}` (add `--extra web` for the web server); "
            "--prepare never installs it"
        )
    if plan.action == "install":
        return (
            f"{label}: extra {plan.extra!r} can be installed (uv dry-run clean, "
            f"{plan.message}): run again with --install-packages to run "
            f"`{plan.command}`"
        )
    return None  # ready, or conflict/invalid/error: the status message says it


def prepare_command(
    cfg: RunConfig, root: Path | None = None, install: bool = False
) -> int:
    """`--prepare`: prepare every component of `cfg` and print a status table.

    The packages layer is reported (`packages.plan_packages`); with `install`
    a per-model extra whose fresh dry-run is clean is installed
    (`install_packages`); the base stack never is.

    Returns 0 when every component is ready, 1 otherwise.
    """
    root = weights_root() if root is None else root
    cancel = threading.Event()
    rows: list[tuple[str, str, ModelStatus | str]] = []
    package_notes: list[str] = []
    entries = config_requirements(cfg)
    # The same config-wide import names for the plan shown and the plan run.
    imports = pkgs.imports_of(r for *_, r in entries if not isinstance(r, str))
    for path, name, type_name, req in entries:
        label = f"{path} {name}"
        if isinstance(req, str):
            rows.append((label, type_name, f"failed: {req}"))
            continue
        print(f"[prepare] {label} ({type_name})", flush=True)
        # The CLI user is local and trusted: sources typed in the YAML too.
        status = prepare(req, root, _progress_printer(), cancel, allow_unknown=True)
        if status.layers.get("packages", "ready") != "ready":
            plan = pkgs.plan_packages(req, imports=imports)
            if install and plan.action == "install":
                print(
                    f"[packages] {label}: installing extra {plan.extra!r}", flush=True
                )
                outcome = install_packages(
                    req,
                    lambda line: print(f"  [uv] {line}", flush=True),
                    cancel,
                    root=root,
                    imports=imports,
                    expected=plan,
                )
                status, plan = outcome.status, outcome.plan
                note = (
                    f"{label}: extra {plan.extra!r} installed"
                    if outcome.installed
                    else _packages_note(label, plan)
                )
            else:
                status = with_package_plan(status, plan)
                note = _packages_note(label, plan)
            if note and note not in package_notes:
                package_notes.append(note)
        rows.append((label, type_name, status))
    print(f"\nweights root: {root}")
    header = ("component", "type", "state", "assets", "code", "packages")
    table = [header]
    notes = []
    for label, type_name, status in rows:
        if isinstance(status, str):
            table.append((label, type_name, "failed", "-", "-", "-"))
            notes.append(f"{label}: {status}")
            continue
        layers = status.layers
        table.append(
            (
                label,
                type_name,
                status.state,
                layers.get("assets", "-"),
                layers.get("code", "-"),
                layers.get("packages", "-"),
            )
        )
        if status.message:
            notes.append(f"{label}: {status.message}")
    widths = [max(len(r[i]) for r in table) for i in range(len(header))]
    for row in table:
        print("  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)).rstrip())
    for note in notes + package_notes:
        print(f"  - {note}")
    ready = all(not isinstance(s, str) and s.state == "ready" for _, _, s in rows)
    return 0 if ready else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Console entry point (`av-perception-lab`, `python -m av_perception_lab`)."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_models:
        print(list_models())
        return 0
    if args.config is None:
        parser.error("--config is required (or use --list-models)")
    if args.install_packages and not args.prepare:
        parser.error("--install-packages is used with --prepare")
    try:
        cfg = load_config(args.config, args.overrides, args.video)
        problem = weights_root_problem(weights_root(), (cfg.output.out_root,))
        if problem is not None:
            raise ConfigError(problem)
        if args.prepare:
            try:
                return prepare_command(cfg, install=args.install_packages)
            except KeyboardInterrupt:
                print(
                    "\ncancelled: partial downloads are kept and resumed by the "
                    "next --prepare",
                    file=sys.stderr,
                )
                return 130
        run(cfg)
    # User/configuration errors only; bugs keep their traceback.
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0
