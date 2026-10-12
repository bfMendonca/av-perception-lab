"""Command-line entry point: RF-DETR + DA3 per frame -> per-box distance.

Writes <out-root>/<timestamp>_<video>_<det>_<depth>[_tag]/ with video.mp4,
detections.csv (one row per kept box with all BoxDepthStats fields),
metrics.json (timings per stage, peak VRAM, focal and its source, versions,
args) and optionally depth.npz: `depth` (N, map_h, map_w) float16 metres,
`frame_index` (N,), `intrinsics` (N, 4) fx, fy, cx, cy at map resolution, and
`geometry` with its `geometry_fields` names (MapGeometry). torch is imported
only inside `run()`.
"""

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Protocol

import numpy as np

from det_depth_distance.depth import (
    DEPTH_MODELS,
    PROCESS_RES_METHODS,
    DepthEstimator,
    DepthFrame,
    classify_depth_model,
    estimate_focal_frame_px,
    peek_model_name,
    resolve_depth_model,
)
from det_depth_distance.detector import (
    DET_MODELS,
    DRIVING_CLASSES,
    Detections,
    Detector,
)
from det_depth_distance.distance import (
    DEFAULT_OUTLIER_K,
    STATS_ROW_FIELDS,
    BoxDepthStats,
    DistanceConfig,
    MapGeometry,
    box_depth_stats,
    parse_stat,
)
from det_depth_distance.render import BoxAnnotation, Renderer, format_label
from det_depth_distance.video_io import (
    VideoWriter,
    output_fps,
    probe,
    read_frames,
    sample_frames,
)

REPO = Path(__file__).resolve().parents[2]
CSV_FIELDS: tuple[str, ...] = (
    "frame",
    "t",
    "class_name",
    "class_id",
    "confidence",
    "x0",
    "y0",
    "x1",
    "y1",
    *STATS_ROW_FIELDS,
)


class DetectorLike(Protocol):
    def __call__(self, rgb: np.ndarray) -> Detections: ...


class DepthLike(Protocol):
    geometry: MapGeometry

    def __call__(self, rgb: np.ndarray) -> DepthFrame: ...


@dataclass(frozen=True, slots=True)
class FrameResult:
    """Everything computed for one frame."""

    detections: Detections
    stats: tuple[BoxDepthStats, ...]  # aligned with detections
    depth: DepthFrame
    timings_ms: dict[str, float]  # "det", "depth", "dist"


def _stat_arg(value: str) -> str:
    try:
        return parse_stat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def build_parser() -> argparse.ArgumentParser:
    """CLI arguments, grouped by pipeline stage."""
    p = argparse.ArgumentParser(
        prog="det-depth-distance",
        description="Run RF-DETR detection and Depth Anything 3 metric depth on every frame of a "
        "video and render each box with its estimated distance to the camera.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    io = p.add_argument_group("input / output")
    io.add_argument("video", type=Path, help="input video file")
    io.add_argument(
        "--out-root",
        type=Path,
        default=Path("_testbed/outputs"),
        help="parent folder of the timestamped run folder",
    )
    io.add_argument("--start", type=float, default=0.0, help="start time (s)")
    io.add_argument(
        "--duration", type=float, default=0.0, help="segment length (s, 0 = to the end)"
    )
    io.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="stop after N processed frames (0 = all)",
    )
    io.add_argument("--stride", type=int, default=1, help="process every N-th frame")
    io.add_argument("--tag", default="", help="suffix of the run folder name")

    det = p.add_argument_group("detector (RF-DETR)")
    det.add_argument(
        "--det-model",
        choices=list(DET_MODELS),
        default="large",
        help="COCO-pretrained variant",
    )
    det.add_argument(
        "--det-weights", help="fine-tuned checkpoint (.pth); overrides --det-model"
    )
    det.add_argument(
        "--det-threshold", type=float, default=0.5, help="minimum confidence"
    )
    det.add_argument(
        "--classes",
        default="driving",
        help=f"'driving' ({', '.join(DRIVING_CLASSES)}), 'all', or comma-separated names",
    )
    det.add_argument(
        "--det-optimize",
        action="store_true",
        help="TorchScript + FP16 (model.inference)",
    )

    dep = p.add_argument_group("depth (Depth Anything 3)")
    dep.add_argument(
        "--depth-model",
        default="da3metric-large",
        help=f"alias ({', '.join(DEPTH_MODELS)}), HF repo id or local dir; must be metric",
    )
    dep.add_argument(
        "--process-res", type=int, default=504, help="DA3 processing resolution"
    )
    dep.add_argument(
        "--process-res-method",
        choices=PROCESS_RES_METHODS,
        default="upper_bound_resize",
    )
    dep.add_argument(
        "--focal-px",
        type=float,
        default=0.0,
        help="[DA3METRIC] focal length in pixels of the INPUT video (0 = estimate)",
    )
    dep.add_argument(
        "--focal-model",
        default="da3-large-1.1",
        help="[DA3METRIC] any-view model that estimates the focal when --focal-px is 0",
    )
    dep.add_argument(
        "--focal-frames", type=int, default=16, help="frames used for focal estimation"
    )
    dep.add_argument(
        "--hf-cache",
        help="Hugging Face cache dir for DA3 weights (default: HF_HUB_CACHE)",
    )

    dist = p.add_argument_group("distance")
    dist.add_argument(
        "--box-shrink",
        type=float,
        default=0.1,
        help="fraction of the box width/height dropped on each side",
    )
    dist.add_argument(
        "--outlier",
        choices=list(DEFAULT_OUTLIER_K),
        default="iqr",
        help="outlier rejection",
    )
    dist.add_argument(
        "--outlier-k",
        type=float,
        help="outlier threshold (default per method: iqr 1.5, mad 3.5, sigma 3)",
    )
    dist.add_argument(
        "--distance-stat",
        type=_stat_arg,
        default="median",
        help="median, mean, trimmed, mode (nearest histogram peak) or pXX percentile",
    )
    dist.add_argument(
        "--distance-type",
        choices=["z", "euclidean"],
        default="z",
        help="z = depth along the optical axis; euclidean = straight-line range",
    )
    dist.add_argument(
        "--min-valid-px", type=int, default=20, help="fewer inliers -> n/a"
    )
    dist.add_argument(
        "--max-range",
        type=float,
        default=0.0,
        help="ignore depths beyond this (m, 0 = none)",
    )
    dist.add_argument(
        "--exclude-sky",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="drop pixels the depth model marks as sky",
    )

    ren = p.add_argument_group("rendering")
    ren.add_argument(
        "--layout",
        choices=["overlay", "side-by-side"],
        default="overlay",
        help="side-by-side adds the colourised depth map and a legend",
    )
    ren.add_argument(
        "--show-stats", action="store_true", help="add ± std to the labels"
    )
    ren.add_argument(
        "--draw-inner-box",
        action="store_true",
        help="outline the sampled (shrunk) region",
    )
    ren.add_argument(
        "--max-depth",
        type=float,
        default=30.0,
        help="far end of the depth colour scale (m)",
    )
    ren.add_argument("--cmap", default="turbo", help="OpenCV colormap name")
    ren.add_argument(
        "--crf", type=int, default=20, help="libx264 quality (lower = better)"
    )
    ren.add_argument(
        "--no-hud", action="store_true", help="boxes and labels only, no info panel"
    )
    ren.add_argument("--font", type=Path, help="TTF/TTC font for labels")

    ex = p.add_argument_group("execution")
    ex.add_argument("--device", default="cuda:0")
    ex.add_argument(
        "--save-depth",
        action="store_true",
        help="also write depth.npz (float16, all frames held in RAM)",
    )
    return p


def process_frame(
    rgb: np.ndarray,
    detector: DetectorLike,
    depth: DepthLike,
    cfg: DistanceConfig,
    *,
    exclude_sky: bool,
) -> FrameResult:
    """Run detector and depth on one RGB frame and compute stats for every box.

    Pure orchestration over `DetectorLike` / `DepthLike`, so it is unit-testable
    with fakes. When `exclude_sky` and the depth model provides a sky mask, sky
    pixels are not eligible.
    """
    # Both models return numpy arrays, so each call ends with a device sync and
    # wall-clock timings are per-stage GPU timings.
    t0 = time.perf_counter()
    detections = detector(rgb)
    t1 = time.perf_counter()
    frame_depth = depth(rgb)
    t2 = time.perf_counter()
    mask = ~frame_depth.sky if exclude_sky and frame_depth.sky is not None else None
    stats = tuple(
        box_depth_stats(
            frame_depth.depth_m,
            box,
            depth.geometry,
            cfg,
            intrinsics=frame_depth.intrinsics,
            mask=mask,
        )
        for box in detections.xyxy
    )
    t3 = time.perf_counter()
    timings = {
        "det": (t1 - t0) * 1e3,
        "depth": (t2 - t1) * 1e3,
        "dist": (t3 - t2) * 1e3,
    }
    return FrameResult(detections, stats, frame_depth, timings)


def make_run_dir(out_root: Path, parts: Sequence[str]) -> Path:
    """Create <out_root>/<YYYYmmdd_HHMMSS>_<non-empty parts joined by _>."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_root / "_".join([stamp, *(part for part in parts if part)])
    run_dir.mkdir(parents=True)
    return run_dir


def latency_stats(values: Sequence[float]) -> dict[str, float]:
    """mean / p50 / p95 / min / max of a latency series ({} when empty)."""
    if not values:
        return {}
    a = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _version(dist: str) -> str | None:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def run(args: argparse.Namespace) -> Path:
    """Process the whole video and return the run folder."""
    import torch

    info = probe(args.video)
    cfg = DistanceConfig(
        box_shrink=args.box_shrink,
        outlier=args.outlier,
        outlier_k=args.outlier_k,
        stat=args.distance_stat,
        distance_type=args.distance_type,
        min_valid_px=args.min_valid_px,
        max_range=args.max_range,
    )
    segment = {"start": args.start, "duration": args.duration}

    # Estimate the focal before loading the main models, so two DA3 models are never resident.
    focal_px, focal_source = (args.focal_px or None), "user (--focal-px)"
    # Kind from the checkpoint's own model_name when it is a local dir (dir names may say nothing).
    depth_kind = classify_depth_model(
        peek_model_name(resolve_depth_model(args.depth_model))
    )
    if not focal_px and depth_kind == "metric_focal":
        frames = sample_frames(info, args.focal_frames, **segment)
        focal_px = estimate_focal_frame_px(
            frames,
            args.focal_model,
            device=args.device,
            process_res=args.process_res,
            process_res_method=args.process_res_method,
            cache_dir=args.hf_cache,
        )
        focal_source = (
            f"estimated by {args.focal_model} (median of {len(frames)} frames)"
        )
        print(
            f"[focal] {focal_px:.1f} px at {info.width}x{info.height} ({focal_source})",
            flush=True,
        )

    # Initialise CUDA before querying memory stats (nothing has touched it with --focal-px).
    torch.cuda.set_device(torch.device(args.device))
    torch.cuda.reset_peak_memory_stats(args.device)
    free0, _ = torch.cuda.mem_get_info(args.device)
    detector = Detector(
        args.det_model,
        weights=args.det_weights,
        device=args.device,
        threshold=args.det_threshold,
        classes=args.classes,
        optimize=args.det_optimize,
    )
    depth = DepthEstimator(
        args.depth_model,
        frame_size=(info.width, info.height),
        device=args.device,
        process_res=args.process_res,
        process_res_method=args.process_res_method,
        focal_frame_px=focal_px,
        cache_dir=args.hf_cache,
    )
    if depth.kind == "metric":
        focal_px, focal_source = None, "predicted per frame by the depth model"

    depth_name = depth.repo_id.rstrip("/").split("/")[-1].lower()
    run_dir = make_run_dir(
        args.out_root, [info.path.stem, detector.name, depth_name, args.tag]
    )
    renderer = Renderer(
        info.width,
        info.height,
        title=f"{detector.name} + {depth.model_name}",
        layout=args.layout,
        show_hud=not args.no_hud,
        max_depth=args.max_depth,
        cmap=args.cmap,
        font_path=args.font,
    )
    whole_video = (
        not (args.start or args.duration or args.max_frames) and args.stride == 1
    )
    audio_from = info.path if whole_video and info.has_audio else None
    side_by_side = args.layout == "side-by-side"
    print(
        f"[run] {info.width}x{info.height} @ {info.fps:.2f} fps | {detector.name} | "
        f"{depth.model_name} ({depth.geometry.map_w}x{depth.geometry.map_h}) -> {run_dir}",
        flush=True,
    )

    timings: dict[str, list[float]] = {"det": [], "depth": [], "dist": [], "render": []}
    totals: Counter = Counter()
    n_frames = n_boxes = n_na = 0
    saved_depth: list[np.ndarray] = []
    saved_index: list[int] = []
    saved_intrinsics: list[tuple[float, float, float, float]] = []
    t_start = time.perf_counter()
    with (
        VideoWriter(
            run_dir / "video.mp4",
            *renderer.output_size(),
            output_fps(info, args.stride),
            crf=args.crf,
            audio_from=audio_from,
        ) as writer,
        open(run_dir / "detections.csv", "w", newline="") as fh,
    ):
        rows = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        rows.writeheader()
        for frame in read_frames(
            info, max_frames=args.max_frames, stride=args.stride, **segment
        ):
            result = process_frame(
                frame.rgb, detector, depth, cfg, exclude_sky=args.exclude_sky
            )
            t0 = time.perf_counter()
            dets = result.detections
            annotations = []
            for i, stats in enumerate(result.stats):
                box = tuple(float(v) for v in dets.xyxy[i])
                name, score = dets.class_name[i], float(dets.confidence[i])
                rows.writerow(
                    {
                        "frame": frame.index,
                        "t": round(frame.t, 4),
                        "class_name": name,
                        "class_id": int(dets.class_id[i]),
                        "confidence": round(score, 4),
                        "x0": box[0],
                        "y0": box[1],
                        "x1": box[2],
                        "y1": box[3],
                        **stats.to_row(),
                    }
                )
                spread = (
                    stats.std
                    if args.show_stats and stats.distance is not None
                    else None
                )
                inner = (
                    depth.geometry.map_to_frame(stats.map_box)
                    if args.draw_inner_box
                    else None
                )
                annotations.append(
                    BoxAnnotation(
                        box,
                        name,
                        score,
                        format_label(name, score, stats.distance, spread),
                        inner,
                    )
                )
                n_na += stats.distance is None
            hud = [
                f"det {result.timings_ms['det']:5.1f} ms  ·  depth {result.timings_ms['depth']:5.1f} ms",
                f"frame {frame.index + 1}/{info.n_frames or '?'}  ·  conf ≥ {args.det_threshold:.2f}",
                f"{cfg.stat} · {cfg.outlier} · shrink {cfg.box_shrink:g} · {cfg.distance_type}",
                f"focal {focal_px:.0f} px" if focal_px else "focal: model-predicted",
            ]
            writer.write(
                renderer.render(
                    frame.rgb,
                    annotations,
                    hud,
                    result.depth.depth_m if side_by_side else None,
                )
            )
            timings["render"].append((time.perf_counter() - t0) * 1e3)
            for stage, ms in result.timings_ms.items():
                timings[stage].append(ms)
            totals.update(dets.class_name)
            n_boxes += len(dets)
            n_frames += 1
            if args.save_depth:
                saved_depth.append(result.depth.depth_m.astype(np.float16))
                saved_index.append(frame.index)
                k = result.depth.intrinsics
                saved_intrinsics.append((k.fx, k.fy, k.cx, k.cy))
            if n_frames % 100 == 0:
                print(
                    f"  {n_frames} frames  depth p50 {np.median(timings['depth']):.1f} ms",
                    flush=True,
                )
    wall = time.perf_counter() - t_start
    free1, _ = torch.cuda.mem_get_info(args.device)
    if n_frames == 0:
        raise ValueError(
            f"no frames decoded from {info.path} (is --start past the end of the video?)"
        )

    if args.save_depth:
        geometry = asdict(depth.geometry)
        np.savez_compressed(
            run_dir / "depth.npz",
            depth=np.stack(saved_depth),
            frame_index=np.asarray(saved_index),
            # per frame: NESTED models predict K for every frame
            intrinsics=np.asarray(saved_intrinsics, dtype=np.float32),
            geometry=np.asarray(list(geometry.values()), dtype=np.float64),
            geometry_fields=np.asarray(list(geometry)),
        )
    focal_map = asdict(depth.fixed_intrinsics) if depth.fixed_intrinsics else None
    metrics = {
        "video": {**asdict(info), "path": str(info.path)},
        "run_dir": str(run_dir),
        "frames_processed": n_frames,
        "detector": {
            "name": detector.name,
            "resolution": detector.resolution,
            "threshold": args.det_threshold,
            "classes": args.classes,
        },
        "depth": {
            "repo_id": depth.repo_id,
            "model_name": depth.model_name,
            "kind": depth.kind,
            "process_res": args.process_res,
            "process_res_method": args.process_res_method,
            "geometry": asdict(depth.geometry),
        },
        "focal_frame_px": focal_px,
        "focal_map_intrinsics": focal_map,
        "focal_source": focal_source,
        "distance": asdict(cfg) | {"k": cfg.k, "exclude_sky": args.exclude_sky},
        "timings_ms": {
            stage: latency_stats(values) for stage, values in timings.items()
        },
        "wall_time_s": wall,
        "end_to_end_fps": n_frames / wall if wall else None,
        "peak_vram_torch_allocated_gb": torch.cuda.max_memory_allocated(args.device)
        / 1e9,
        "device_mem_used_delta_gb_approx": (free0 - free1) / 1e9,
        "detections_total": dict(totals.most_common()),
        "boxes_total": n_boxes,
        "boxes_without_distance": n_na,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "rfdetr": _version("rfdetr"),
            "depth-anything-3": _version("depth-anything-3"),
            "gpu": torch.cuda.get_device_name(args.device),
        },
        "repo_commit": _git_commit(),
        "host": platform.node(),
        "args": {k: str(v) for k, v in vars(args).items()},
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False)
    )
    print(
        f"[done] {n_frames} frames, {n_boxes} boxes ({n_na} n/a) in {wall:.1f} s -> {run_dir}"
    )
    return run_dir


def main(argv: Sequence[str] | None = None) -> int:
    """Console entry point (`det-depth-distance`, `python -m det_depth_distance`)."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.video.is_file():
        parser.error(f"video not found: {args.video}")
    checks = (
        (args.stride >= 1, "--stride must be >= 1"),
        (args.max_frames >= 0, "--max-frames must be >= 0 (0 = all)"),
        (args.start >= 0 and args.duration >= 0, "--start/--duration must be >= 0"),
        (args.focal_px >= 0, "--focal-px must be >= 0 (0 = estimate)"),
        (args.focal_frames >= 1, "--focal-frames must be >= 1"),
        (args.process_res >= 14, "--process-res must be >= 14 (DA3 patch size)"),
        (args.max_depth > 0, "--max-depth must be > 0"),
    )
    for is_valid, message in checks:
        if not is_valid:
            parser.error(message)
    try:
        run(args)
    except ValueError as exc:  # configuration errors raised by the pipeline components
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0
