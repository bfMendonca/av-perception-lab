"""Run outputs, one sink per artefact: detections.csv (one row per
instance: frame, box, label, all BoxDepthStats fields, then source and
raw_label), video.mp4 (annotated; colour by source when several detectors
are concatenated), depth.npz (optional) and metrics.json (component
descriptions under "components", pipeline policy under "pipeline", focal and
its source, timings, GPU memory, versions, cross-detector agreement,
resolved config).

detections.csv also ends with n_sources, sources, source_scores and
raw_labels: one entry per contributing detector (its best member),
"|"-separated, in decreasing score; for unfused instances n_sources is 1.

After those, detections.csv ends with used_mask, mask_px (depth-map pixels
of the eroded mask region), mask_erode (erosion actually applied) and
box_distance_m (only with region "both" when the mask was used).
metrics.json adds distance_regions and, with "both", region_comparison
(mask minus box).

depth.npz holds `depth` (N, map_h, map_w) float16 metres, `frame_index`
(N,), `intrinsics` (N, 4) fx, fy, cx, cy at map resolution, and `geometry`
with its `geometry_fields` names (MapGeometry).
"""

import csv
import json
import platform
import time
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from det_depth_distance.config import RenderConfig, RunConfig
from det_depth_distance.core.agreement import AgreementStats
from det_depth_distance.core.pipeline import SetupInfo
from det_depth_distance.core.types import FrameResult, MapGeometry, best_per_source
from det_depth_distance.distance import (
    MASK_ROW_FIELDS,
    STATS_ROW_FIELDS,
    DistanceConfig,
)
from det_depth_distance.io.source import VideoSource
from det_depth_distance.io.video import VideoWriter
from det_depth_distance.render import BoxAnnotation, Renderer, format_label
from det_depth_distance.runtime import DeviceMonitor, git_commit, package_version

# Columns of task_0001 in the same order, then the new ones at the end so
# readers that index by position keep working.
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
    "source",
    "raw_label",
    "n_sources",
    "sources",
    "source_scores",
    "raw_labels",
    *MASK_ROW_FIELDS,
    "box_distance_m",
)


@dataclass(frozen=True, slots=True)
class RunContext:
    """What every sink may need to open its artefact."""

    run_dir: Path
    source: VideoSource
    setup: SetupInfo
    # Pipeline.describe(): {"components": ..., "pipeline": ...}
    pipeline: dict[str, Any]
    config: RunConfig

    @property
    def detector_names(self) -> list[str]:
        return [d["name"] for d in self.pipeline["components"]["detectors"]]


class Sink(ABC):
    """Consumer of FrameResults: open once, write per frame, close once."""

    def open(self, ctx: RunContext) -> None:
        self.ctx = ctx

    @abstractmethod
    def write(self, result: FrameResult) -> None: ...

    def close(self) -> None:
        """Flush and finalise the artefact."""


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


class CsvSink(Sink):
    """detections.csv: one row per instance with all BoxDepthStats fields."""

    def open(self, ctx: RunContext) -> None:
        super().open(ctx)
        # Kept open across write() calls; closed in close().
        self._fh = open(ctx.run_dir / "detections.csv", "w", newline="")  # noqa: SIM115
        self._rows = csv.DictWriter(self._fh, fieldnames=CSV_FIELDS)
        self._rows.writeheader()

    def write(self, result: FrameResult) -> None:
        frame, inst = result.frame, result.instances
        for i, (stats, members) in enumerate(
            zip(result.stats, inst.provenance(), strict=True)
        ):
            best = best_per_source(members)
            box = result.box_stats[i] if result.box_stats is not None else None
            x0, y0, x1, y1 = (float(v) for v in inst.xyxy[i])
            self._rows.writerow(
                {
                    "frame": frame.index,
                    "t": round(frame.t, 4),
                    "class_name": inst.label[i],
                    "class_id": int(inst.class_id[i]),
                    "confidence": round(float(inst.score[i]), 4),
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    **stats.to_row(),
                    "source": inst.source[i],
                    "raw_label": inst.raw_label[i],
                    "n_sources": len(best),
                    "sources": "|".join(m.source for m in best),
                    # same representation as `confidence`
                    "source_scores": "|".join(str(round(m.score, 4)) for m in best),
                    "raw_labels": "|".join(m.raw_label for m in best),
                    **stats.mask_row(),
                    "box_distance_m": box.distance if box is not None else None,
                }
            )

    def close(self) -> None:
        self._fh.close()


class VideoSink(Sink):
    """video.mp4: boxes, instance masks, labels with distance, HUD and optional
    depth panel.

    With several detectors, boxes and HUD counts are coloured by `source`,
    unless the instances are fused (then by canonical label); fused labels
    show the number of agreeing detectors (×N). The distance HUD line names the
    region policy and erosion only in frames where some instance used its mask.
    """

    def __init__(self, cfg: RenderConfig, distance: DistanceConfig) -> None:
        self.cfg = cfg
        self.distance = distance
        self.render_ms: list[float] = []  # read by MetricsSink

    def open(self, ctx: RunContext) -> None:
        super().open(ctx)
        cfg, source = self.cfg, ctx.source
        detectors = ctx.pipeline["components"]["detectors"]
        depth_name = ctx.pipeline["components"]["depth"]["name"]
        self._by_source = len(detectors) > 1
        self._renderer = Renderer(
            *source.frame_size,
            title=" + ".join([*ctx.detector_names, depth_name]),
            layout=cfg.layout,
            show_hud=cfg.hud,
            max_depth=cfg.max_depth,
            cmap=cfg.cmap,
            font_path=cfg.font,
            mask_alpha=cfg.mask_alpha,
        )
        policy = ctx.pipeline["pipeline"]
        self._region = f" · region {policy['region']} · erode {policy['mask_erode']}"
        thresholds = [(d["name"], d["options"]["threshold"]) for d in detectors]
        if self._by_source:
            self._conf = "conf " + ", ".join(f"{n} ≥ {t:.2f}" for n, t in thresholds)
        else:
            self._conf = f"conf ≥ {thresholds[0][1]:.2f}"
        aggregator = ctx.pipeline["components"]["aggregator"]
        self._fusion = None
        if aggregator["type"] != "concat":
            iou_thr = aggregator["options"].get("iou_thr")
            self._fusion = aggregator["type"] + (
                f" · IoU ≥ {iou_thr:g}" if iou_thr is not None else ""
            )
        audio = source.is_whole_video and source.info.has_audio
        self._writer = VideoWriter(
            ctx.run_dir / "video.mp4",
            *self._renderer.output_size(),
            source.output_fps(),
            crf=cfg.crf,
            audio_from=source.info.path if audio else None,
        )

    def write(self, result: FrameResult) -> None:
        t0 = time.perf_counter()
        inst, d, focal = result.instances, self.distance, self.ctx.setup.focal_frame_px
        by_source = self._by_source and inst.sources is None
        masks = inst.masks if self.cfg.mask_alpha > 0 else None
        annotations = []
        for i, (stats, members) in enumerate(
            zip(result.stats, inst.provenance(), strict=True)
        ):
            name, score = inst.label[i], float(inst.score[i])
            n_sources = len(best_per_source(members))
            spread = (
                stats.std
                if self.cfg.show_stats and stats.distance is not None
                else None
            )
            inner = (
                result.depth.geometry.map_to_frame(stats.map_box)
                if self.cfg.draw_inner_box
                else None
            )
            annotations.append(
                BoxAnnotation(
                    tuple(float(v) for v in inst.xyxy[i]),
                    name,
                    score,
                    format_label(name, score, stats.distance, spread, n_sources),
                    inner,
                    color_key=inst.source[i] if by_source else None,
                    mask=None if masks is None else masks[i],
                )
            )
        ms, n_total = result.timings_ms, self.ctx.source.info.n_frames
        hud = [
            f"det {ms['det']:5.1f} ms  ·  depth {ms['depth']:5.1f} ms",
            f"frame {result.frame.index + 1}/{n_total or '?'}  ·  {self._conf}",
            f"{d.stat} · {d.outlier} · shrink {d.box_shrink:g} · {d.distance_type}"
            + (self._region if any(s.used_mask for s in result.stats) else ""),
            f"focal {focal:.0f} px" if focal else "focal: model-predicted",
        ]
        if self._fusion:
            hud.append(self._fusion)
        if result.agreement is not None and len(result.agreement.pairs) == 1:
            p = result.agreement.pairs[0]
            hud.append(
                f"agree {p.matched} · only {p.a} {p.only_a} · only {p.b} {p.only_b}"
            )
        side_by_side = self.cfg.layout == "side-by-side"
        self._writer.write(
            self._renderer.render(
                result.frame.rgb,
                annotations,
                hud,
                result.depth.depth_m if side_by_side else None,
            )
        )
        self.render_ms.append((time.perf_counter() - t0) * 1e3)

    def close(self) -> None:
        self._writer.close()


class NpzSink(Sink):
    """depth.npz: every depth map as float16 (all frames held in RAM until close)."""

    def __init__(self) -> None:
        self._depth: list[np.ndarray] = []
        self._index: list[int] = []
        self._intrinsics: list[tuple[float, float, float, float]] = []
        self._geometry: MapGeometry | None = None

    def write(self, result: FrameResult) -> None:
        self._depth.append(result.depth.depth_m.astype(np.float16))
        self._index.append(result.frame.index)
        k = result.depth.intrinsics
        self._intrinsics.append((k.fx, k.fy, k.cx, k.cy))
        self._geometry = result.depth.geometry

    def close(self) -> None:
        if self._geometry is None:  # no frames: nothing to save
            return
        geometry = asdict(self._geometry)
        np.savez_compressed(
            self.ctx.run_dir / "depth.npz",
            depth=np.stack(self._depth),
            frame_index=np.asarray(self._index),
            # per frame: NESTED models predict K for every frame
            intrinsics=np.asarray(self._intrinsics, dtype=np.float32),
            geometry=np.asarray(list(geometry.values()), dtype=np.float64),
            geometry_fields=np.asarray(list(geometry)),
        )


class MetricsSink(Sink):
    """metrics.json: run summary, written at close.

    `extra_timings` returns per-frame timing series measured outside the
    pipeline (e.g. {"render": VideoSink.render_ms}).
    """

    def __init__(
        self,
        monitor: DeviceMonitor,
        extra_timings: Callable[[], dict[str, list[float]]],
    ) -> None:
        self.monitor = monitor
        self.extra_timings = extra_timings
        self.n_frames = 0
        self._timings: dict[str, list[float]] = defaultdict(list)
        self._labels: Counter = Counter()
        self._sources: Counter = Counter()
        self._n_sources: Counter = Counter()
        self._regions: Counter = Counter()
        # (label, mask distance, box distance) of instances measured both ways
        self._region_pairs: list[tuple[str, float, float]] = []
        self.n_boxes = self.n_without_distance = 0

    def open(self, ctx: RunContext) -> None:
        super().open(ctx)
        agreement = ctx.config.agreement
        self._agreement = AgreementStats(agreement.iou_thr, agreement.class_aware)
        self._t_start = time.perf_counter()

    def write(self, result: FrameResult) -> None:
        for stage, ms in result.timings_ms.items():
            self._timings[stage].append(ms)
        self._labels.update(result.instances.label)
        self._sources.update(result.instances.source)
        self._n_sources.update(
            len(best_per_source(m)) for m in result.instances.provenance()
        )
        if result.agreement is not None:
            self._agreement.add(result.frame.index, result.agreement)
        self._regions.update("mask" if s.used_mask else "box" for s in result.stats)
        if result.box_stats is not None:
            self._region_pairs += [
                (label, s.distance, b.distance)
                for label, s, b in zip(
                    result.instances.label, result.stats, result.box_stats, strict=True
                )
                if b is not None and s.distance is not None and b.distance is not None
            ]
        self.n_boxes += len(result.instances)
        self.n_without_distance += sum(s.distance is None for s in result.stats)
        self.n_frames += 1

    def close(self) -> None:
        if self.n_frames == 0:  # failed run (the CLI reports it): nothing to summarise
            return
        wall = time.perf_counter() - self._t_start
        ctx, cfg = self.ctx, self.ctx.config
        components = ctx.pipeline["components"]
        device = self.monitor.summary()
        timings = {**self._timings, **self.extra_timings()}
        metrics = {
            "video": ctx.source.describe(),
            "run_dir": str(ctx.run_dir),
            "frames_processed": self.n_frames,
            "components": components,
            "pipeline": ctx.pipeline["pipeline"],
            "focal_frame_px": ctx.setup.focal_frame_px,
            "focal_map_intrinsics": components["depth"].get("fixed_intrinsics"),
            "focal_source": ctx.setup.focal_source,
            "distance": asdict(cfg.distance)
            | {
                "k": cfg.distance.k,
                "exclude_sky": cfg.exclude_sky,
                "region": cfg.region,
            },
            "timings_ms": {
                stage: latency_stats(values) for stage, values in timings.items()
            },
            "wall_time_s": wall,
            "end_to_end_fps": self.n_frames / wall if wall else None,
            "peak_vram_torch_allocated_gb": device["peak_vram_torch_allocated_gb"],
            "device_mem_used_delta_gb_approx": device[
                "device_mem_used_delta_gb_approx"
            ],
            "detections_total": dict(self._labels.most_common()),
            "detections_by_source": dict(self._sources.most_common()),
            "detections_by_n_sources": dict(sorted(self._n_sources.items())),
            "boxes_total": self.n_boxes,
            "boxes_without_distance": self.n_without_distance,
            "distance_regions": {
                "mask": self._regions["mask"],
                "box": self._regions["box"],
            },
            "versions": {
                "python": platform.python_version(),
                "torch": device["torch"],
                "cuda": device["cuda"],
                "rfdetr": package_version("rfdetr"),
                "depth-anything-3": package_version("depth-anything-3"),
                "ultralytics": package_version("ultralytics"),
                "gpu": device["gpu"],
            },
            "repo_commit": git_commit(),
            "host": platform.node(),
            "config": cfg.to_dict(),
        }
        if self._agreement:  # some frame had agreement (>= 2 detectors, enabled)
            metrics["agreement"] = self._agreement.summary()
        if cfg.region == "both":
            metrics["region_comparison"] = _region_comparison(self._region_pairs)
        (ctx.run_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False)
        )


def _difference_summary(mask_m: np.ndarray, box_m: np.ndarray) -> dict[str, Any]:
    diff = mask_m - box_m
    return {
        "n": int(diff.size),
        "median_signed_m": float(np.median(diff)),
        "median_abs_m": float(np.median(np.abs(diff))),
        "p90_abs_m": float(np.percentile(np.abs(diff), 90)),
        "median_rel": float(np.median(diff / box_m)),
    }


def _region_comparison(pairs: Sequence[tuple[str, float, float]]) -> dict[str, Any]:
    """Mask-minus-box distance differences over all instances and per label.

    Args:
        pairs: (label, mask distance, box distance) of the instances whose
            distance exists both ways.

    Returns:
        {"n", "median_signed_m", "median_abs_m", "p90_abs_m", "median_rel",
        "by_label": {label: same keys}}; only {"n": 0} without pairs.
        median_rel is relative to the box distance; a negative median_signed_m
        means the mask places objects nearer than the box did.
    """
    if not pairs:
        return {"n": 0}
    labels = np.asarray([p[0] for p in pairs])
    mask_m = np.asarray([p[1] for p in pairs], dtype=np.float64)
    box_m = np.asarray([p[2] for p in pairs], dtype=np.float64)
    by_label = {
        str(label): _difference_summary(mask_m[labels == label], box_m[labels == label])
        for label in sorted(set(labels.tolist()))
    }
    return {**_difference_summary(mask_m, box_m), "by_label": by_label}
