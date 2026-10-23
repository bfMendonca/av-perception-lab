"""Command-line entry point: det-depth-distance --config run.yaml [VIDEO] [--set k=v]...

Composition root: builds every component from the YAML (no model loaded
yet, so configuration errors exit with code 2 in seconds), probes the video,
composes the camera (calibration, lens check, undistortion: camera.build_camera),
then runs source -> pipeline -> sinks. `--list-models` prints the registered adapters
and their options. torch is imported only through runtime.DeviceMonitor.

Writes <output.out_root>/<timestamp>_<video>_<detectors>_<depth>[_tag]/ with
video.mp4, detections.csv, metrics.json, trajectories.jsonl (with motion)
and optionally depth.npz (see io.sinks).
"""

import argparse
import dataclasses
import sys
import time
from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path

from det_depth_distance.camera import build_camera
from det_depth_distance.config import RunConfig, load_config
from det_depth_distance.core import registry
from det_depth_distance.core.options import ConfigError
from det_depth_distance.core.pipeline import Pipeline
from det_depth_distance.io.sinks import (
    CsvSink,
    MetricsSink,
    NpzSink,
    RunContext,
    TrajectorySink,
    VideoSink,
    make_run_dir,
)
from det_depth_distance.io.source import VideoSource
from det_depth_distance.motion import build_motion
from det_depth_distance.runtime import DeviceMonitor


def build_parser() -> argparse.ArgumentParser:
    """CLI arguments: the YAML config, an optional video, overrides, the model list."""
    p = argparse.ArgumentParser(
        prog="det-depth-distance",
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


def run(cfg: RunConfig) -> Path:
    """Process the configured video and return the run folder.

    Raises:
        ConfigError: invalid component specs, unusable calibration, missing
            intrinsics, or no frames.
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
    camera = build_camera(cfg.camera, VideoSource(cfg.source))
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
            f"[run] {source.info.width}x{source.info.height} @ {source.info.fps:.2f} fps"
            f" | {', '.join(names)} | {depth.name} ({g.map_w}x{g.map_h}) -> {run_dir}",
            flush=True,
        )
        video = VideoSink(cfg.render, cfg.distance)
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
                f"no frames decoded from {source.info.path} "
                "(is source.start past the end of the video?)"
            )
    wall = time.perf_counter() - t_start
    print(
        f"[done] {metrics.n_frames} frames, {metrics.n_boxes} boxes "
        f"({metrics.n_without_distance} n/a) in {wall:.1f} s -> {run_dir}"
    )
    return run_dir


def main(argv: Sequence[str] | None = None) -> int:
    """Console entry point (`det-depth-distance`, `python -m det_depth_distance`)."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_models:
        print(list_models())
        return 0
    if args.config is None:
        parser.error("--config is required (or use --list-models)")
    try:
        run(load_config(args.config, args.overrides, args.video))
    # User/configuration errors only; bugs keep their traceback.
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0
