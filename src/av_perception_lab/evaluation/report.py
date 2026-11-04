"""Evaluation report files: eval.json, eval.md and optional PNG figures.

matplotlib is imported only when plots are requested (optional "eval"
extra), so the light environment never needs it.
"""

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from av_perception_lab.core.options import ConfigError
from av_perception_lab.evaluation.evaluate import EvaluationResult

# Columns of the error tables in eval.md (ErrorStats fields; *_rel shown in %).
_STAT_COLUMNS: tuple[str, ...] = (
    "n",
    "n_refs",
    "reference_median_m",
    "bias_m",
    "median_abs_m",
    "p90_abs_m",
    "rmse_m",
    "mean_rel",
    "median_abs_rel",
    "p90_abs_rel",
    "within_2u",
    "scale_median",
)
# Detail tables show the configured distance only; eval.json has every estimator.
_DETAIL_ESTIMATOR = "distance_m"
_TABLE_TITLES: dict[str, str] = {
    "bin": "By reference distance (m)",
    "label": "By class",
    "region": "By region (mask / box)",
    "mask_source": "By mask producer",
    "ref_id": "By reference",
}


def _cell(name: str, value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if math.isnan(value):
            return "-"
        if name.endswith("_rel") or name == "within_2u":
            return f"{100 * value:.1f} %"
        return f"{value:.3f}"
    return str(value) if value != "" else "(none)"


def _table(rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> list[str]:
    if not rows:
        return ["(no data)"]
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "---|" * len(columns),
    ]
    lines += [
        "| " + " | ".join(_cell(c, r.get(c)) for c in columns) + " |" for r in rows
    ]
    return lines


def render_markdown(result: EvaluationResult) -> str:
    """eval.md: runs, overall errors per run/source/estimator, detail tables of
    distance_m (distance bin, class, region, mask producer, reference),
    coverage, implied focal and warnings."""
    opts = result.options
    level = (
        "one error per reference (median over its frames)"
        if opts.level == "reference"
        else "one error per frame"
    )
    out = [
        f"# Evaluation against references of {result.references.video}",
        "",
        (
            f"{len(result.references.references)} references; level: {level}; "
            "error = estimate - reference (positive = too far)."
        ),
        "",
        "## Runs",
        "",
        *_table(
            result.runs,
            (
                "name",
                "video",
                "depth_kind",
                "focal_frame_px",
                "focal_source",
                "distance_type",
                "stat",
                "calibration",
                "undistort",
            ),
        ),
        "",
        "## Overall",
        "",
        *_table(result.overall, ("run", "source", "estimator", *_STAT_COLUMNS)),
    ]
    for dim, rows in result.tables.items():
        detail = [r for r in rows if r["estimator"] == _DETAIL_ESTIMATOR]
        out += [
            "",
            f"## {_TABLE_TITLES[dim]} ({_DETAIL_ESTIMATOR})",
            "",
            *_table(detail, ("run", "source", dim, *_STAT_COLUMNS)),
        ]
    out += [
        "",
        "## Coverage",
        "",
        *_table(
            result.coverage,
            (
                "run",
                "ref_id",
                "source",
                "frames",
                "detected_valid",
                "detected_na",
                "not_detected",
                "track_ids",
            ),
        ),
        "",
        "## Implied focal (metric_focal models)",
        "",
        *_table(
            [
                {"run": run, "source": source, "implied_focal_px": f}
                for run, by_source in result.implied_focal.items()
                for source, f in by_source.items()
            ],
            ("run", "source", "implied_focal_px"),
        ),
    ]
    if result.warnings:
        out += ["", "## Warnings", "", *[f"- {w}" for w in result.warnings]]
    return "\n".join(out) + "\n"


def _plot(result: EvaluationResult, out_dir: Path, plt: Any) -> list[Path]:
    """Signed error vs reference distance (per reference) and median |rel| per bin."""
    paths = []
    per_ref = [
        r for r in result.tables["ref_id"] if r["estimator"] == _DETAIL_ESTIMATOR
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    for key in sorted({(r["run"], r["source"]) for r in per_ref}):
        rows = [r for r in per_ref if (r["run"], r["source"]) == key]
        ax.plot(
            [r["reference_median_m"] for r in rows],
            [r["median_signed_m"] for r in rows],
            "o",
            label=" / ".join(key),
        )
    ax.axhline(0.0, color="grey", linewidth=0.8)
    ax.set_xlabel("reference distance (m)")
    ax.set_ylabel("signed error, estimate - reference (m)")
    ax.legend(fontsize="small")
    paths.append(out_dir / "eval_error_vs_distance.png")
    fig.savefig(paths[-1], dpi=120, bbox_inches="tight")
    plt.close(fig)

    by_bin = [r for r in result.tables["bin"] if r["estimator"] == _DETAIL_ESTIMATOR]
    bins = sorted(
        {r["bin"] for r in by_bin}, key=lambda b: float(b.lstrip(">=").split("-")[0])
    )
    series = sorted({(r["run"], r["source"]) for r in by_bin})
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.8 / max(len(series), 1)
    for i, key in enumerate(series):
        values = {
            r["bin"]: r["median_abs_rel"]
            for r in by_bin
            if (r["run"], r["source"]) == key
        }
        ax.bar(
            [j + i * width for j in range(len(bins))],
            [100 * values.get(b, math.nan) for b in bins],
            width,
            label=" / ".join(key),
        )
    ax.set_xticks([j + width * (len(series) - 1) / 2 for j in range(len(bins))], bins)
    ax.set_xlabel("reference distance bin (m)")
    ax.set_ylabel("median |relative error| (%)")
    ax.legend(fontsize="small")
    paths.append(out_dir / "eval_abs_rel_by_bin.png")
    fig.savefig(paths[-1], dpi=120, bbox_inches="tight")
    plt.close(fig)
    return paths


def write_report(
    result: EvaluationResult, out_dir: Path, *, plot: bool = False
) -> list[Path]:
    """Write eval.json (everything, machine-readable), eval.md (tables: overall
    per run/detector/estimator, by distance bin, label, region, coverage,
    implied focal, warnings) and, with plot=True, PNG figures.

    Raises:
        ConfigError: plot requested without matplotlib (uv sync --extra eval).
    """
    plt = None
    if plot:
        try:
            import matplotlib

            matplotlib.use("Agg")  # files only, no display
            import matplotlib.pyplot as plt
        except ImportError:
            raise ConfigError("--plot needs matplotlib: uv sync --extra eval") from None
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / "eval.json", out_dir / "eval.md"]
    # NaN (empty groups) is not JSON: written as null.
    paths[0].write_text(
        json.dumps(_json_ready(result.to_dict()), indent=2, ensure_ascii=False)
    )
    paths[1].write_text(render_markdown(result))
    if plt is not None:
        paths += _plot(result, out_dir, plt)
    return paths


def _json_ready(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: _json_ready(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(v) for v in value]
    return value
