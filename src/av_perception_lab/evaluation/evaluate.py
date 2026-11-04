"""Evaluation orchestration: references x runs -> error tables (no I/O).

Runs are compared side by side; every table row carries its run, source
group and estimator. Loading files and building undistortion transforms is
left to the caller (tools.evaluate), so this module only computes.
"""

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from av_perception_lab.core.options import ConfigError
from av_perception_lab.evaluation.matching import PointsTransform, match_references
from av_perception_lab.evaluation.references import ReferenceSet
from av_perception_lab.evaluation.run_data import ESTIMATORS, RunData
from av_perception_lab.evaluation.stats import (
    DEFAULT_BINS,
    Level,
    Sample,
    build_samples,
    coverage_summary,
    implied_focal,
    summarize,
)

# Grouping of the overall table, and the extra dimension of each detail table.
OVERALL_BY: tuple[str, ...] = ("run", "source", "estimator")
TABLE_DIMENSIONS: tuple[str, ...] = ("bin", "label", "region", "mask_source", "ref_id")


@dataclass(frozen=True, slots=True)
class EvalOptions:
    """Evaluation settings. Built from CLI arguments, so it raises ConfigError
    directly."""

    estimators: tuple[str, ...] = ESTIMATORS  # those absent from a CSV are skipped
    level: Level = "reference"
    bins: tuple[float, ...] = DEFAULT_BINS
    iou_min: float = 0.3
    force: bool = False  # evaluate runs of another video than the references'

    def __post_init__(self) -> None:
        unknown = [e for e in self.estimators if e not in ESTIMATORS]
        if unknown or not self.estimators:
            raise ConfigError(
                f"estimators: unknown {unknown}; choose from {', '.join(ESTIMATORS)}"
            )
        if self.level not in ("reference", "frame"):
            raise ConfigError(f"level: {self.level!r} is not reference or frame")
        if not self.bins or any(b <= a for a, b in itertools.pairwise(self.bins)):
            raise ConfigError("bins: edges must be strictly increasing")
        if not 0.0 < self.iou_min <= 1.0:
            raise ConfigError("iou_min must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Everything the report shows (eval.json is `to_dict()`)."""

    references: ReferenceSet
    runs: tuple[
        dict[str, Any], ...
    ]  # name, video, focal and its source, distance settings, camera
    overall: list[dict[str, Any]]  # run x source x estimator
    tables: Mapping[str, list[dict[str, Any]]]  # TABLE_DIMENSIONS -> rows
    coverage: list[dict[str, Any]]  # run x reference x source
    # run name -> source group -> focal (px); detectors have their own scale error
    implied_focal: Mapping[str, Mapping[str, float | None]]
    warnings: tuple[str, ...]
    options: EvalOptions = field(default_factory=EvalOptions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "references": {
                "video": self.references.video,
                "n_references": len(self.references.references),
            },
            "options": asdict(self.options),
            "runs": list(self.runs),
            "overall": self.overall,
            "tables": dict(self.tables),
            "coverage": self.coverage,
            "implied_focal": dict(self.implied_focal),
            "warnings": list(self.warnings),
        }


def _run_info(run: RunData) -> dict[str, Any]:
    distance = run.metrics["config"]["distance"]
    camera = run.metrics.get("camera") or {}
    return {
        "name": run.name,
        "video": run.video_name,
        "depth_kind": run.depth_kind,
        "focal_frame_px": run.focal_frame_px,
        "focal_source": run.metrics.get("focal_source"),
        "distance_type": run.distance_type,
        "stat": distance.get("stat"),
        "region": distance.get("region"),
        "sources": list(run.sources),
        "calibration": (camera.get("calibration") or {}).get("path"),
        "undistort": bool(camera.get("undistort")),
    }


def evaluate(
    refs: ReferenceSet,
    runs: Sequence[RunData],
    options: EvalOptions | None = None,
    *,
    points_transforms: Mapping[str, PointsTransform] | None = None,
) -> EvaluationResult:
    """Match `refs` in every run and summarise the errors.

    `points_transforms` maps a run name to the transform of original-video
    pixels into its undistorted pixels (runs with camera.calibration.undistort);
    `options` defaults to EvalOptions().
    References selecting a track id are skipped, with a warning, in runs
    without tracking.

    Raises:
        ConfigError: two runs with the same folder name (every result is keyed
            by it), a run of another video than the references' (unless
            options.force), or an undistorted run without its transform.
    """
    names = [run.name for run in runs]
    repeated = next((n for n in names if names.count(n) > 1), None)
    if repeated is not None:
        raise ConfigError(
            f"two runs are named {repeated}: results are keyed by the run folder "
            "name, rename one folder (e.g. set output.tag)"
        )
    options = options or EvalOptions()
    transforms = points_transforms or {}
    warnings: list[str] = []
    samples: list[Sample] = []
    coverage: list[dict[str, Any]] = []
    focal: dict[str, dict[str, float | None]] = {}
    for run in runs:
        if run.video_name != refs.video:
            message = (
                f"run {run.name} processed {run.video_name}, "
                f"the references are for {refs.video}"
            )
            if not options.force:
                raise ConfigError(f"{message} (use --force to evaluate anyway)")
            warnings.append(message)
        if run.is_undistorted and run.name not in transforms:
            raise ConfigError(
                f"run {run.name} undistorted its frames: the reference pixels need "
                "its undistortion transform"
            )
        usable = [
            r for r in refs.references if r.select.track_id is None or run.has_tracks
        ]
        for r in refs.references:
            if r not in usable:
                warnings.append(
                    f"run {run.name}: reference {r.id} selects a track id, skipped (no tracking)"
                )
        if not usable:
            continue
        run_refs = replace(refs, references=tuple(usable))
        matches = match_references(
            run,
            run_refs,
            iou_min=options.iou_min,
            points_transform=transforms.get(run.name),
        )
        covered = {m.ref_id for m in matches}
        warnings += [
            f"run {run.name}: reference {r.id} is outside the processed frames"
            for r in usable
            if r.id not in covered
        ]
        if run.frame_intrinsics is None and any(
            (r.quantity == "range") != (run.distance_type == "euclidean")
            for r in usable
        ):
            warnings.append(
                f"run {run.name}: no frame intrinsics, references compared without "
                f"converting their quantity to {run.distance_type}"
            )
        estimators = [e for e in options.estimators if e in run.available_estimators()]
        run_samples = build_samples(run, matches, estimators)
        samples += run_samples
        coverage += [{"run": run.name, **row} for row in coverage_summary(matches, run)]
        focal[run.name] = {
            src: implied_focal(run, [s for s in run_samples if s.source == src])
            for src in run.sources
        }
    summary = {"level": options.level, "bins": options.bins}
    return EvaluationResult(
        references=refs,
        runs=tuple(_run_info(run) for run in runs),
        overall=summarize(samples, by=OVERALL_BY, **summary),
        tables={
            dim: summarize(samples, by=(*OVERALL_BY, dim), **summary)
            for dim in TABLE_DIMENSIONS
        },
        coverage=coverage,
        implied_focal=focal,
        warnings=tuple(warnings),
        options=options,
    )
