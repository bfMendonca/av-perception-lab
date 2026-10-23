"""Configuration of the motion stage (YAML section "motion"; stdlib only).

Model and ego-motion specs stay mappings `{type, **options}`, validated by
the registry when `build_motion` builds them (as RunConfig.tracking); this
module only checks what crosses keys (classes and default name models).
Imported by `config` and `core.pipeline`, never the reverse.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

# Point of the box measured on the plane: the middle of its bottom edge (the
# ground contact of vehicles and people) or its centre.
ReferencePoint = Literal["bottom_center", "center"]


@dataclass(frozen=True, slots=True)
class MeasurementConfig:
    """How a box becomes a position (x, z) and its covariance.

    Defaults from the task_0001 tracks (docs/task_0009.md, Passo 0): the
    reference point jitters 0.7-1.6 px frame to frame at 1920x1080 (0.25 px
    robust); a 2 % floor, twice the smoothing's, makes the mean NIS of the
    straight-driving video ~2 (the far-range DA3 noise is heavy-tailed).
    """

    reference_point: ReferencePoint = "bottom_center"
    # sigma_d = max(std_scale * box std, rel_floor * d), as SmoothingConfig
    std_scale: float = 0.5
    rel_floor: float = 0.02
    pixel_std: float = 2.0  # frame px, lateral std of the reference point

    def __post_init__(self) -> None:
        if self.std_scale < 0 or self.rel_floor < 0:
            raise ValueError("std_scale and rel_floor must be >= 0")
        if self.std_scale == 0 and self.rel_floor == 0:
            raise ValueError("std_scale and rel_floor cannot both be 0")
        if self.pixel_std <= 0:
            raise ValueError(f"pixel_std must be > 0, got {self.pixel_std}")


@dataclass(frozen=True, slots=True)
class PredictionConfig:
    """Trajectory prediction: a point every step_s up to horizon_s seconds."""

    horizon_s: float = 3.0
    step_s: float = 0.25
    n_sigma: float = 2.0  # drawn uncertainty (video and BEV ellipses)

    def __post_init__(self) -> None:
        if self.horizon_s <= 0 or self.n_sigma <= 0:
            raise ValueError("horizon_s and n_sigma must be > 0")
        if not 0 < self.step_s <= self.horizon_s:
            raise ValueError(f"step_s must be in (0, horizon_s], got {self.step_s}")


@dataclass(frozen=True, slots=True)
class CameraMount:
    """Camera pose in the vehicle frame (x forward, y left); used only with
    ego-motion."""

    x_m: float = 0.0
    y_m: float = 0.0
    yaw_deg: float = 0.0  # optical axis vs vehicle forward, + = to the left


@dataclass(frozen=True, slots=True)
class MotionConfig:
    """Per-track motion estimation (YAML section "motion"); needs tracking
    (checked by the Pipeline). Times in seconds."""

    # model name -> spec {type, **options} (registry family "motion")
    models: Mapping[str, Mapping[str, Any]] = field(
        default_factory=lambda: {"cv": {"type": "cv"}}
    )
    classes: Mapping[str, str] = field(default_factory=dict)  # label -> model name
    default: str | None = "cv"  # model of unmapped classes; None = no estimate
    reset_after_s: float = 1.0  # state dropped after this long without a measurement
    # The filter restarts above it (id switch). chi2(2) 99.9 % is 13.8, but the
    # depth noise is heavy-tailed: 30 cut the restarts 3-6x on the task_0001
    # tracks while still catching jumps of several metres.
    nis_gate: float = 30.0
    measurement: MeasurementConfig = field(default_factory=MeasurementConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    # spec {type, **options} (registry family "ego_motion"); None = camera frame
    ego_motion: Mapping[str, Any] | None = None
    camera_mount: CameraMount = field(default_factory=CameraMount)

    def __post_init__(self) -> None:
        if not self.models:
            raise ValueError("models: at least one model is required")
        for label, name in self.classes.items():
            if name not in self.models:
                raise ValueError(
                    f"classes.{label}: unknown model {name!r}; "
                    f"models: {', '.join(self.models)}"
                )
        if self.default is not None and self.default not in self.models:
            raise ValueError(
                f"default: unknown model {self.default!r}; "
                f"models: {', '.join(self.models)}"
            )
        if self.reset_after_s <= 0 or self.nis_gate <= 0:
            raise ValueError("reset_after_s and nis_gate must be > 0")
