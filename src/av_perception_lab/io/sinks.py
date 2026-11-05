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

detections.csv ends with mask_source (producer of the instance mask, "" if
none) and mask_score (refiner quality, empty for detector masks).
metrics.json adds mask_sources, refined (targets and filled masks per
refiner) and, with semantic segmenters, semantic (per map: excluded
classes, mean class fractions, excluded fraction; plus the excluded
fraction of the union of all maps). No semantic CSV column. With
save_depth, depth.npz adds semantic_labels_<name> (labels sampled on the
depth-map grid) and semantic_classes_<name> per segmenter.

detections.csv ends with track_id, distance_smooth_m, distance_smooth_std_m,
rel_speed_mps and rel_speed_std_mps (empty without tracking/smoothing; std
and speed only with the kalman filter); distance_m stays the raw per-frame
value. metrics.json adds "tracks" (per-track summary) with a tracker.

detections.csv ends with the motion_* columns (MOTION_FIELDS; empty without
motion; heading NaN below the model's min speed, yaw rate only with ctrv,
abs speed only with ego-motion). With motion, trajectories.jsonl holds one
line per frame and instance with its predicted trajectory, and metrics.json
tracks.per_track[] gains "motion".

metrics.json also holds frame_intrinsics (the frame-resolution K given to the
depth model, null when it predicts K per frame) and camera (calibration file,
rescaling, lens check, undistortion and the focal cross-check; {} without
camera.calibration). With undistortion every box is in undistorted pixels.
"""

import csv
import json
import math
import platform
import time
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from av_perception_lab.config import RenderConfig, RunConfig
from av_perception_lab.core.agreement import AgreementStats
from av_perception_lab.core.pipeline import SetupInfo
from av_perception_lab.core.tracking import TrackSummary
from av_perception_lab.core.types import (
    DepthResult,
    FrameResult,
    Instances,
    MapGeometry,
    MotionEstimate,
    TrackEstimate,
    best_per_source,
)
from av_perception_lab.distance import (
    MASK_ROW_FIELDS,
    STATS_ROW_FIELDS,
    DistanceConfig,
    resample_mask,
)
from av_perception_lab.io.source import FrameSource, McapSource, VideoSource
from av_perception_lab.io.video import VideoWriter
from av_perception_lab.motion.measurement import project_points, reference_pixel
from av_perception_lab.render import (
    BevTrack,
    BoxAnnotation,
    PathAnnotation,
    Renderer,
    format_label,
)
from av_perception_lab.runtime import DeviceMonitor, git_commit, package_version

# Smoothing columns of a row, from TrackEstimate (distance, distance_std,
# rel_speed, rel_speed_std).
ESTIMATE_FIELDS: tuple[str, ...] = (
    "distance_smooth_m",
    "distance_smooth_std_m",
    "rel_speed_mps",
    "rel_speed_std_mps",
)
# Motion columns of a row, from MotionEstimate (model ... n_updates).
MOTION_FIELDS: tuple[str, ...] = (
    "motion_model",
    "motion_x_m",
    "motion_z_m",
    "motion_x_std_m",
    "motion_z_std_m",
    "motion_vx_mps",
    "motion_vz_mps",
    "motion_speed_mps",
    "motion_speed_std_mps",
    "motion_heading_deg",
    "motion_heading_std_deg",
    "motion_yaw_rate_dps",
    "motion_yaw_rate_std_dps",
    "motion_abs_speed_mps",
    "motion_nis",
    "motion_n_updates",
)
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
    "mask_source",
    "mask_score",
    "track_id",
    *ESTIMATE_FIELDS,
    *MOTION_FIELDS,
)


@dataclass(frozen=True, slots=True)
class RunContext:
    """What every sink may need to open its artefact."""

    run_dir: Path
    source: FrameSource
    setup: SetupInfo
    # Pipeline.describe(): {"components": ..., "pipeline": ...}
    pipeline: dict[str, Any]
    config: RunConfig
    camera: Mapping[str, Any] = field(default_factory=dict)  # CameraSetup.details

    @property
    def detector_names(self) -> list[str]:
        return [d["name"] for d in self.pipeline["components"]["detectors"]]

    @property
    def excluded_classes(self) -> dict[str, list[str]]:
        """Semantic segmenter name -> excluded class names (only maps excluding some)."""
        return self.pipeline["pipeline"]["exclude_semantic"]


def _output_fps(source: FrameSource) -> str:
    """Constant frame rate ("num/den") of the rendered video. An MCAP renders
    at its nominal rate for now (per-frame times come with the VFR writer)."""
    if isinstance(source, VideoSource | McapSource):
        return source.output_fps()
    fps = Fraction(source.nominal_fps).limit_denominator(1001)
    return f"{fps.numerator}/{fps.denominator}"


def refine_counts(
    instances: Instances, refiners: Sequence[dict[str, Any]]
) -> dict[str, tuple[int, int]]:
    """(targets, filled masks) of every refiner in one frame, from the final
    `mask_sources()` and the refiners' describe() (in pipeline order).

    "filled" counts the masks a refiner produced; its targets are every
    instance under `targets: all`, else the instances whose final mask came
    from it, from a later refiner or from nobody (a refined mask that came back
    empty leaves "").
    """
    sources = instances.mask_sources()
    counts = {}
    for i, refiner in enumerate(refiners):
        name = refiner["name"]
        if refiner["options"]["targets"] == "all":
            targets = len(sources)
        else:
            later = {"", *(r["name"] for r in refiners[i:])}
            targets = sum(s in later for s in sources)
        counts[name] = (targets, sources.count(name))
    return counts


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


def _csv_float(value: float | None) -> float | None:
    """4-decimal CSV value; None (empty cell) for None and NaN."""
    return None if value is None or not math.isfinite(value) else round(value, 4)


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
        mask_sources, mask_scores = inst.mask_sources(), inst.mask_score
        track_ids = inst.track_id
        temporal = result.temporal or (None,) * len(inst)
        motion = result.motion or (None,) * len(inst)
        for i, (stats, members, est, mot) in enumerate(
            zip(result.stats, inst.provenance(), temporal, motion, strict=True)
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
                    "mask_source": mask_sources[i],
                    "mask_score": (
                        None
                        if mask_scores is None or np.isnan(mask_scores[i])
                        else round(float(mask_scores[i]), 4)
                    ),
                    "track_id": None if track_ids is None else int(track_ids[i]),
                    **_estimate_row(est),
                    **_motion_row(mot),
                }
            )

    def close(self) -> None:
        self._fh.close()


def _estimate_row(est: TrackEstimate | None) -> dict[str, float | None]:
    """ESTIMATE_FIELDS of one CSV row (all empty without an estimate)."""
    values = (
        (None,) * len(ESTIMATE_FIELDS)
        if est is None
        else (est.distance, est.distance_std, est.rel_speed, est.rel_speed_std)
    )
    return {
        name: _csv_float(v) for name, v in zip(ESTIMATE_FIELDS, values, strict=True)
    }


def _motion_row(est: MotionEstimate | None) -> dict[str, str | float | int | None]:
    """MOTION_FIELDS of one CSV row (all empty without an estimate)."""
    if est is None:
        return dict.fromkeys(MOTION_FIELDS)
    values = (
        est.x,
        est.z,
        est.x_std,
        est.z_std,
        est.vx,
        est.vz,
        est.speed,
        est.speed_std,
        est.heading_deg,
        est.heading_std_deg,
        est.yaw_rate_dps,
        est.yaw_rate_std_dps,
        est.abs_speed,
        est.nis,
    )
    floats = zip(MOTION_FIELDS[1:-1], values, strict=True)
    return {
        "motion_model": est.model,
        **{name: _csv_float(v) for name, v in floats},
        "motion_n_updates": est.n_updates,
    }


def _path_annotation(
    est: MotionEstimate,
    ref_px: tuple[float, float],
    depth: DepthResult,
    n_sigma: float,
) -> PathAnnotation | None:
    """The predicted trajectory in frame pixels, from the box reference point;
    cut at the first point that cannot be projected (None if that is the
    first). Radii: n_sigma half-axes of each point's uncertainty projected on
    the ground: x std * fx / z across, z std * fy * |y| / z^2 along (the
    ground recedes toward the horizon)."""
    traj = est.trajectory
    if traj is None or not len(traj.t):
        return None
    x, z = traj.xz.T
    px = project_points(
        np.stack([x, np.full_like(x, traj.y), z], axis=1),
        depth.intrinsics,
        depth.geometry,
    )
    invalid = ~np.isfinite(px).all(axis=1)
    n = int(np.argmax(invalid)) if invalid.any() else len(px)
    if n == 0:
        return None
    k, g = depth.intrinsics, depth.geometry
    std = np.sqrt(np.diagonal(traj.cov[:n], axis1=1, axis2=2).clip(0.0))
    z = z[:n]
    radii = n_sigma * np.stack(
        [
            std[:, 0] * (k.fx / g.sx) / z,
            std[:, 1] * (k.fy / g.sy) * abs(traj.y) / z**2,
        ],
        axis=1,
    )
    return PathAnnotation(np.vstack([ref_px, px[:n]]), np.vstack([[0.0, 0.0], radii]))


class VideoSink(Sink):
    """video.mp4: semantic layers, boxes, instance masks, labels with distance,
    HUD and optional depth panel.

    With several detectors, boxes and HUD counts are coloured by `source`,
    unless the instances are fused (then by canonical label); fused labels
    show the number of agreeing detectors (×N). The distance HUD line names the
    region policy and erosion only in frames where some instance used its mask.
    Refiners add a HUD line each ("refine <name> · filled/targets"), semantic
    segmenters one each ("semantic <name> · excl <classes>"); the maps named
    by `render.semantic_overlay` (all when None) are drawn in config order.
    With a tracker, labels start with "#<track id>" and the last HUD line
    describes tracking (ids = tracks opened so far) and smoothing; with
    smoothing, labels show the smoothed distance (`render.show_smoothed`) and
    the relative speed (`render.show_speed`, kalman only). With motion, labels
    end with the speed on the plane (`render.show_motion_speed`), each box
    gets its predicted trajectory (`render.show_trajectory`) and, with
    `render.bev_inset`, the frame panel a top-view inset.
    """

    def __init__(
        self,
        cfg: RenderConfig,
        distance: DistanceConfig,
        on_frame: Callable[[int, np.ndarray], None] | None = None,
    ) -> None:
        self.cfg = cfg
        self.distance = distance
        self.on_frame = on_frame  # called with (frame index, composed image)
        self.render_ms: list[float] = []  # read by MetricsSink

    def open(self, ctx: RunContext) -> None:
        super().open(ctx)
        cfg, source = self.cfg, ctx.source
        motion = ctx.config.motion
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
            semantic_alpha=cfg.semantic_alpha,
            trajectory_alpha=cfg.trajectory_alpha,
            bev_size_px=round(cfg.bev_size * source.frame_size[1])
            if cfg.bev_inset
            else 0,
            bev_range_m=cfg.bev_range_m,
            bev_width_m=cfg.bev_width_m,
            n_sigma=motion.prediction.n_sigma if motion is not None else 2.0,
        )
        self._motion = motion
        self._refiners = ctx.pipeline["components"]["refiners"]
        self._semantic_hud = [
            f"semantic {s['name']} · excl "
            + (", ".join(ctx.excluded_classes.get(s["name"], [])) or "-")
            for s in ctx.pipeline["components"]["semantic"]
        ]
        policy = ctx.pipeline["pipeline"]
        self._region = f" · region {policy['region']} · erode {policy['mask_erode']}"
        self._track_hud = _track_hud(
            ctx.pipeline["components"]["tracker"], policy["smoothing"], policy["motion"]
        )
        self._max_track_id = 0
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
        self._writer = VideoWriter(
            ctx.run_dir / "video.mp4",
            *self._renderer.output_size(),
            _output_fps(source),
            crf=cfg.crf,
            audio_from=source.audio_from(),
        )

    def write(self, result: FrameResult) -> None:
        t0 = time.perf_counter()
        inst, d, focal = result.instances, self.distance, self.ctx.setup.focal_frame_px
        by_source = self._by_source and inst.sources is None
        masks = inst.masks if self.cfg.mask_alpha > 0 else None
        temporal = result.temporal or (None,) * len(inst)
        motion = result.motion or (None,) * len(inst)
        annotations, bev = [], []
        for i, (stats, members, est, mot) in enumerate(
            zip(result.stats, inst.provenance(), temporal, motion, strict=True)
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
            distance = (
                est.distance
                if est is not None and self.cfg.show_smoothed
                else stats.distance
            )
            label = format_label(
                name,
                score,
                distance,
                spread,
                n_sources,
                track_id=None if inst.track_id is None else int(inst.track_id[i]),
                rel_speed=(
                    est.rel_speed if est is not None and self.cfg.show_speed else None
                ),
                motion_speed=(
                    mot.speed
                    if mot is not None and self.cfg.show_motion_speed
                    else None
                ),
            )
            xyxy = tuple(float(v) for v in inst.xyxy[i])
            path = None
            if mot is not None and self.cfg.show_trajectory:
                path = _path_annotation(
                    mot,
                    reference_pixel(xyxy, self._motion.measurement.reference_point),
                    result.depth,
                    self._motion.prediction.n_sigma,
                )
            box = BoxAnnotation(
                xyxy,
                name,
                score,
                label,
                inner,
                color_key=inst.source[i] if by_source else None,
                mask=None if masks is None else masks[i],
                path=path,
            )
            annotations.append(box)
            if mot is not None and mot.trajectory is not None:
                bev.append(
                    BevTrack(
                        box.key,
                        f"#{int(inst.track_id[i])}",
                        (mot.x, mot.z),
                        mot.trajectory.xz,
                        mot.trajectory.cov,
                    )
                )
        ms, n_total = result.timings_ms, self.ctx.source.n_frames_total
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
        hud += [
            f"refine {name} · {filled}/{targets}"
            for name, (targets, filled) in refine_counts(inst, self._refiners).items()
        ]
        hud += self._semantic_hud
        if self._track_hud is not None:
            if inst.track_id is not None and len(inst):
                self._max_track_id = max(self._max_track_id, int(inst.track_id.max()))
            hud.append(f"{self._track_hud} · ids {self._max_track_id}")
        overlay = self.cfg.semantic_overlay
        layers = (
            [
                (m.labels, m.class_names)
                for name, m in result.semantic.items()
                if overlay is None or name in overlay
            ]
            if self.cfg.semantic_alpha > 0
            else []
        )
        side_by_side = self.cfg.layout == "side-by-side"
        image = self._renderer.render(
            result.frame.rgb,
            annotations,
            hud,
            result.depth.depth_m if side_by_side else None,
            semantic=layers,
            bev=bev,
        )
        if self.on_frame is not None:
            self.on_frame(result.frame.index, image)
        self._writer.write(image)
        self.render_ms.append((time.perf_counter() - t0) * 1e3)

    def close(self) -> None:
        self._writer.close()


def _track_hud(
    tracker: dict[str, Any] | None,
    smoothing: dict[str, Any] | None,
    motion: dict[str, Any] | None = None,
) -> str | None:
    """Static part of the tracking HUD line, e.g. "track iou · IoU ≥ 0.3 · age
    0.5 s · kalman q 2 · motion cv/ctrv · 3 s"; None without a tracker."""
    if tracker is None:
        return None
    options = tracker["options"]
    text = f"track {tracker['type']}"
    if "iou_thr" in options:
        text += f" · IoU ≥ {options['iou_thr']:g}"
    if "max_age_s" in options:
        text += f" · age {options['max_age_s']:g} s"
    if smoothing is not None:
        text += f" · {smoothing['filter']}"
        if smoothing["filter"] == "kalman":
            text += f" q {smoothing['accel_std']:g}"
    if motion is not None:
        types = dict.fromkeys(m["type"] for m in motion["models"].values())
        text += f" · motion {'/'.join(types)} · {motion['prediction']['horizon_s']:g} s"
    return text


class TrajectorySink(Sink):
    """trajectories.jsonl: one JSON line per frame x instance with a motion
    estimate: {"frame", "t", "track_id", "label", "model", "x", "z", "vx",
    "vz", "y", "t_ahead" [K], "xz" [K][2], "cov" [K][2][2]} in the current
    camera frame (metres, seconds; 4 decimals)."""

    def open(self, ctx: RunContext) -> None:
        super().open(ctx)
        # Kept open across write() calls; closed in close().
        self._fh = open(ctx.run_dir / "trajectories.jsonl", "w")  # noqa: SIM115

    def write(self, result: FrameResult) -> None:
        if result.motion is None:
            return
        inst = result.instances
        for i, est in enumerate(result.motion):
            if est is None or est.trajectory is None:
                continue
            traj = est.trajectory
            line = {
                "frame": result.frame.index,
                "t": round(result.frame.t, 4),
                "track_id": int(inst.track_id[i]),
                "label": inst.label[i],
                "model": est.model,
                **{k: round(getattr(est, k), 4) for k in ("x", "z", "vx", "vz")},
                "y": round(traj.y, 4),
                "t_ahead": np.round(traj.t, 4).tolist(),
                "xz": np.round(traj.xz, 4).tolist(),
                "cov": np.round(traj.cov, 4).tolist(),
            }
            self._fh.write(json.dumps(line) + "\n")

    def close(self) -> None:
        self._fh.close()


class NpzSink(Sink):
    """depth.npz: every depth map as float16 (all frames held in RAM until close),
    plus each semantic map's labels sampled on the depth-map grid (the grid and
    sampling of the distance eligibility), uint8 up to 256 classes."""

    def __init__(self) -> None:
        self._depth: list[np.ndarray] = []
        self._index: list[int] = []
        self._intrinsics: list[tuple[float, float, float, float]] = []
        self._geometry: MapGeometry | None = None
        self._labels: dict[str, list[np.ndarray]] = defaultdict(list)
        self._classes: dict[str, tuple[str, ...]] = {}

    def write(self, result: FrameResult) -> None:
        self._depth.append(result.depth.depth_m.astype(np.float16))
        self._index.append(result.frame.index)
        k = result.depth.intrinsics
        self._intrinsics.append((k.fx, k.fy, k.cx, k.cy))
        g = self._geometry = result.depth.geometry
        for name, m in result.semantic.items():
            dtype = np.uint8 if len(m.class_names) <= 256 else np.uint16
            grid = resample_mask(m.labels, g, (0, 0, g.map_w, g.map_h))
            self._labels[name].append(grid.astype(dtype))
            self._classes[name] = m.class_names

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
            **{f"semantic_labels_{n}": np.stack(v) for n, v in self._labels.items()},
            **{
                f"semantic_classes_{n}": np.asarray(c) for n, c in self._classes.items()
            },
        )


class MetricsSink(Sink):
    """metrics.json: run summary, written at close.

    `extra_timings` returns per-frame timing series measured outside the
    pipeline (e.g. {"render": VideoSink.render_ms}). `describe`, when given
    (Pipeline.describe), is read again at close for the "pipeline" policy,
    whose counters are only final after the last frame (e.g.
    motion.ego_motion.clamped_frames). With a tracker it adds "tracks"
    (core.tracking.TrackSummary, with per-track "motion" when motion runs).
    """

    def __init__(
        self,
        monitor: DeviceMonitor,
        extra_timings: Callable[[], dict[str, list[float]]],
        *,
        describe: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self.monitor = monitor
        self.extra_timings = extra_timings
        self.describe = describe
        self.n_frames = 0
        self._timings: dict[str, list[float]] = defaultdict(list)
        self._labels: Counter = Counter()
        self._sources: Counter = Counter()
        self._n_sources: Counter = Counter()
        self._regions: Counter = Counter()
        # (label, mask distance, box distance) of instances measured both ways
        self._region_pairs: list[tuple[str, float, float]] = []
        self._mask_sources: Counter = Counter()
        self._refined: dict[str, Counter] = defaultdict(Counter)
        # semantic: per map, per-frame class fractions and excluded fraction; per
        # frame, the excluded fraction of the union of all maps
        self._class_fracs: dict[str, list[dict[str, float]]] = defaultdict(list)
        self._excluded_fracs: dict[str, list[float]] = defaultdict(list)
        self._union_fracs: list[float] = []
        self._excluded_ids: dict[str, np.ndarray] | None = None  # at the first frame
        self.n_boxes = self.n_without_distance = 0

    def open(self, ctx: RunContext) -> None:
        super().open(ctx)
        agreement = ctx.config.agreement
        self._agreement = AgreementStats(agreement.iou_thr, agreement.class_aware)
        has_tracker = ctx.pipeline["components"]["tracker"] is not None
        has_motion = ctx.pipeline["pipeline"]["motion"] is not None
        self._tracks = TrackSummary(motion=has_motion) if has_tracker else None
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
        self._mask_sources.update(s for s in result.instances.mask_sources() if s)
        refiners = self.ctx.pipeline["components"]["refiners"]
        for name, (targets, filled) in refine_counts(
            result.instances, refiners
        ).items():
            self._refined[name].update(targets=targets, filled=filled)
        if result.semantic:
            self._add_semantic(result)
        if self._tracks is not None:
            self._tracks.add(result)
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

    def _add_semantic(self, result: FrameResult) -> None:
        excluded = self.ctx.excluded_classes
        if self._excluded_ids is None:  # class ids are fixed once the models load
            self._excluded_ids = {
                name: np.array(
                    [m.class_names.index(c) for c in excluded.get(name, [])], np.int32
                )
                for name, m in result.semantic.items()
            }
        union = None
        for name, m in result.semantic.items():
            fractions = m.fractions()
            self._class_fracs[name].append(fractions)
            self._excluded_fracs[name].append(
                sum(fractions[c] for c in excluded.get(name, []))
            )
            ids = self._excluded_ids[name]
            if ids.size:
                layer = np.isin(m.labels, ids)
                union = layer if union is None else union | layer
        self._union_fracs.append(0.0 if union is None else float(union.mean()))

    def _semantic_summary(self) -> dict[str, Any]:
        """Means over frames: per map its excluded classes, class fractions and
        excluded fraction; on top, the excluded fraction of the union of maps
        (maps overlap, e.g. road and lane, so the per-map values do not add up)."""
        maps = {}
        for name, frames in self._class_fracs.items():
            maps[name] = {
                "exclude": self.ctx.excluded_classes.get(name, []),
                "class_frac_mean": {
                    c: float(np.mean([f[c] for f in frames])) for c in frames[0]
                },
                "excluded_frac_mean": float(np.mean(self._excluded_fracs[name])),
            }
        return {"maps": maps, "excluded_frac_mean": float(np.mean(self._union_fracs))}

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
            "pipeline": (self.describe() if self.describe else ctx.pipeline)[
                "pipeline"
            ],
            "focal_frame_px": ctx.setup.focal_frame_px,
            "focal_map_intrinsics": components["depth"].get("fixed_intrinsics"),
            "focal_source": ctx.setup.focal_source,
            "frame_intrinsics": (
                asdict(ctx.setup.frame_intrinsics)
                if ctx.setup.frame_intrinsics is not None
                else None
            ),
            "camera": {**ctx.camera, **ctx.setup.intrinsics_details},
            "distance": asdict(cfg.distance)
            | {
                "k": cfg.distance.k,
                "exclude_sky": cfg.exclude_sky,
                "region": cfg.region,
                "exclude_semantic": list(cfg.exclude_semantic),
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
            # instances with a mask, by producer (detector, refiner)
            "mask_sources": dict(self._mask_sources.most_common()),
            "refined": {name: dict(c) for name, c in self._refined.items()},
            "versions": {
                "python": platform.python_version(),
                "torch": device["torch"],
                "cuda": device["cuda"],
                "rfdetr": package_version("rfdetr"),
                "depth-anything-3": package_version("depth-anything-3"),
                "ultralytics": package_version("ultralytics"),
                "transformers": package_version("transformers"),
                "timm": package_version("timm"),
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
        if self._class_fracs:
            metrics["semantic"] = self._semantic_summary()
        if self._tracks is not None:
            metrics["tracks"] = self._tracks.summary()
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
