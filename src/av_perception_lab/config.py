"""YAML run configuration: load, apply --set overrides, validate shape.

Sections are dataclasses built with `from_mapping` (unknown keys are errors,
reported with their dotted path); domain rules live in each dataclass's
__post_init__. Component specs (detectors, depth, aggregation, refiners,
semantic, tracking, motion models and ego-motion) are validated by their
adapter's Options when built through the registry; RunConfig only checks what
crosses sections (semantic segmenter names, the namespaces of excluded
semantic labels, the drawn maps, the BEV inset needing motion).
Pipeline-policy keys written in the `distance` section
(PIPELINE_DISTANCE_KEYS: exclude_sky, region, exclude_semantic) are split
off into RunConfig fields before DistanceConfig is built.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import cv2
import yaml

from av_perception_lab.adapters.depth.da3 import ProcessResMethod
from av_perception_lab.core import registry
from av_perception_lab.core.labels import SEMANTIC_NAME_PATTERN, group_semantic_labels
from av_perception_lab.core.options import ConfigError, from_mapping
from av_perception_lab.core.registry import Stage, register_stage
from av_perception_lab.distance import DistanceConfig, RegionPolicy
from av_perception_lab.motion.config import MotionConfig
from av_perception_lab.temporal import SmoothingConfig


@dataclass(frozen=True, slots=True)
class SourceConfig:
    """Input (a video, or an image topic of a ROS 2 MCAP) and the processed segment."""

    video: Path | None = None
    mcap: Path | None = None  # exclusive with video; needs topic
    topic: str | None = None  # image topic of the MCAP
    start: float = 0.0  # s
    duration: float = 0.0  # s, 0 = to the end
    max_frames: int = 0  # 0 = all
    stride: int = 1  # process every N-th frame (MCAP: every N-th message)

    def __post_init__(self) -> None:
        if self.video is not None and self.mcap is not None:
            raise ValueError("video and mcap are mutually exclusive")
        if self.mcap is not None and not self.topic:
            raise ValueError("mcap needs a topic (the image topic to read)")
        if self.stride < 1:
            raise ValueError("stride must be >= 1")
        if self.max_frames < 0:
            raise ValueError("max_frames must be >= 0 (0 = all)")
        if self.start < 0 or self.duration < 0:
            raise ValueError("start/duration must be >= 0")


@dataclass(frozen=True, slots=True)
class EstimateConfig:
    """Focal estimation with an any-view DA3 model (median over sampled frames)."""

    model: str = "da3-large-1.1"
    frames: int = 16
    process_res: int = 504
    process_res_method: ProcessResMethod = "upper_bound_resize"
    hf_cache: Path | None = None

    def __post_init__(self) -> None:
        if self.frames < 1:
            raise ValueError("frames must be >= 1")
        if self.process_res < 14:
            raise ValueError("process_res must be >= 14 (DA3 patch size)")


# What a camera mismatch between the calibration file and the video does.
LensCheck = Literal["error", "warn", "off"]


@dataclass(frozen=True, slots=True)
class CalibrationConfig:
    """A camera calibration file (camera.calibration; see camera.build_camera).

    The file's existence is checked by build_camera, before any model loads,
    so configurations can be validated without it.
    """

    # OpenCV FileStorage YAML/JSON (camera.load_calibration); exclusive with
    # from_source (intrinsics from the MCAP's CameraInfo).
    path: Path | None = None
    from_source: bool = False
    undistort: bool = False  # remove lens distortion from every frame
    alpha: float = 0.0  # getOptimalNewCameraMatrix, [0, 1]; only with undistort
    check_lens: LensCheck = "error"  # recording camera vs the video's metadata
    cross_check: bool = (
        False  # also estimate the focal (camera.estimate) and report both
    )

    def __post_init__(self) -> None:
        if self.from_source and self.path is not None:
            raise ValueError("from_source and path are exclusive: set only one")
        if not self.from_source and self.path is None:
            raise ValueError("set path or from_source")
        if self.from_source and self.undistort:
            raise ValueError(
                "undistort is not available with from_source: use the rectify section"
            )
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")


@dataclass(frozen=True, slots=True)
class RectifyConfig:
    """Lens-distortion removal with the MCAP's CameraInfo (first stage after
    the source, so every later stage sees rectified pixels)."""

    enabled: bool = False
    alpha: float = 0.0  # getOptimalNewCameraMatrix, [0, 1]

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")


@dataclass(frozen=True, slots=True)
class CameraConfig:
    """Camera intrinsics, by precedence: a calibration file, a known focal or its
    estimation."""

    focal_px: float | None = None  # focal length in pixels of the input video
    estimate: EstimateConfig | None = field(default_factory=EstimateConfig)
    calibration: CalibrationConfig | None = None  # null = focal_px / estimate

    def __post_init__(self) -> None:
        if self.focal_px is not None and self.focal_px <= 0:
            raise ValueError("focal_px must be > 0 (null = estimate)")
        if self.calibration is not None and self.focal_px is not None:
            raise ValueError("calibration and focal_px are exclusive: set only one")
        if (
            self.calibration is not None
            and self.calibration.cross_check
            and self.estimate is None
        ):
            raise ValueError("calibration.cross_check needs the estimate section")


@dataclass(frozen=True, slots=True)
class RenderConfig:
    """Annotated video rendering."""

    layout: Literal["overlay", "side-by-side"] = "overlay"
    show_stats: bool = False  # add ± std to the labels
    draw_inner_box: bool = False  # outline the sampled (shrunk) region
    max_depth: float = 30.0  # far end of the depth colour scale (m)
    cmap: str = "turbo"  # OpenCV colormap name
    crf: int = 20  # libx264 quality (lower = better)
    hud: bool = True
    font: Path | None = None
    mask_alpha: float = 0.35  # instance-mask tint opacity, [0, 1]; 0 = no masks
    # semantic layer tint opacity, [0, 1]; 0 = no semantic layer drawn
    semantic_alpha: float = 0.4
    # Names of the semantic maps drawn in the video (None = all), in the order
    # of `semantic`; checked against the segmenter names by RunConfig.
    semantic_overlay: tuple[str, ...] | None = None
    # Allowlist of "<map>.<class>" entries painted in the semantic layer (None =
    # every class, [] = none); render-only, never affects distance. A bare YAML
    # string is accepted as a one-item list (like distance.exclude_semantic).
    # The map prefix is checked by RunConfig, class names once the segmenters load.
    semantic_classes: str | tuple[str, ...] | None = None
    show_smoothed: bool = True  # label shows the smoothed distance (raw stays in CSV)
    show_speed: bool = True  # label shows the relative speed (kalman smoothing)
    # With motion: the predicted trajectory (polyline fading out with n-sigma
    # ground ellipses), the motion speed at the end of the label and a top-view inset.
    show_trajectory: bool = True
    trajectory_alpha: float = 0.8  # opacity of the first segment, [0, 1]
    show_motion_speed: bool = True
    bev_inset: bool = False  # needs motion
    bev_size: float = 0.35  # inset side, fraction of the frame height, (0, 1]
    bev_range_m: float = 40.0  # forward range of the inset
    bev_width_m: float = 20.0  # lateral span of the inset

    def __post_init__(self) -> None:
        if isinstance(self.semantic_classes, str):
            object.__setattr__(self, "semantic_classes", (self.semantic_classes,))
        if self.max_depth <= 0:
            raise ValueError("max_depth must be > 0")
        if not 0.0 <= self.trajectory_alpha <= 1.0:
            raise ValueError(
                f"trajectory_alpha must be in [0, 1], got {self.trajectory_alpha}"
            )
        if not 0.0 < self.bev_size <= 1.0:
            raise ValueError(f"bev_size must be in (0, 1], got {self.bev_size}")
        if self.bev_range_m <= 0 or self.bev_width_m <= 0:
            raise ValueError("bev_range_m and bev_width_m must be > 0")
        if not 0.0 <= self.mask_alpha <= 1.0:
            raise ValueError(f"mask_alpha must be in [0, 1], got {self.mask_alpha}")
        if not 0.0 <= self.semantic_alpha <= 1.0:
            raise ValueError(
                f"semantic_alpha must be in [0, 1], got {self.semantic_alpha}"
            )
        if not hasattr(cv2, f"COLORMAP_{self.cmap.upper()}"):
            raise ValueError(f"unknown OpenCV colormap {self.cmap!r}")


@dataclass(frozen=True, slots=True)
class AgreementConfig:
    """Cross-detector agreement metrics (core.agreement); only computed with
    two or more detectors."""

    enabled: bool = True
    iou_thr: float = 0.5  # same-object overlap, in (0, 1)
    class_aware: bool = True  # match only equal (canonical) labels

    def __post_init__(self) -> None:
        if not 0.0 < self.iou_thr < 1.0:
            raise ValueError(f"iou_thr must be in (0, 1), got {self.iou_thr}")


@dataclass(frozen=True, slots=True)
class OutputConfig:
    """Where the run folder goes and what it holds."""

    out_root: Path = Path("_testbed/outputs")
    tag: str = ""  # suffix of the run folder name
    save_depth: bool = False  # also write depth.npz (float16, all frames in RAM)


# Pipeline-policy keys written in the YAML `distance` section that are not
# DistanceConfig fields; load_config moves them to the RunConfig field of the
# same name.
PIPELINE_DISTANCE_KEYS: tuple[str, ...] = ("exclude_sky", "region", "exclude_semantic")


@dataclass(frozen=True, slots=True)
class RunConfig:
    """The whole run, as described by the YAML file."""

    detectors: tuple[Mapping[str, Any], ...]  # specs, validated by registry.build
    depth: Mapping[str, Any]
    source: SourceConfig = field(default_factory=SourceConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    rectify: RectifyConfig = field(default_factory=RectifyConfig)
    # Flat spec {type, **options}, validated by registry.build("aggregator", ...).
    aggregation: Mapping[str, Any] = field(default_factory=lambda: {"type": "concat"})
    # Mask refiner specs {type, name?, **options}, applied in order.
    refiners: tuple[Mapping[str, Any], ...] = ()
    # semantic: list of segmenter specs ({name, type, **options}); every map is
    # kept separately in FrameResult.semantic under its required, unique name.
    semantic: tuple[Mapping[str, Any], ...] = ()
    agreement: AgreementConfig = field(default_factory=AgreementConfig)
    # Flat tracker spec {type, name?, **options}, validated by
    # registry.build("tracker", ...); None = no tracking.
    tracking: Mapping[str, Any] | None = None
    # Per-track distance smoothing; needs tracking (checked by the Pipeline).
    smoothing: SmoothingConfig | None = None
    # Per-track motion estimation; needs tracking (checked by the Pipeline).
    motion: MotionConfig | None = None
    distance: DistanceConfig = field(default_factory=DistanceConfig)
    exclude_sky: bool = True  # from distance.exclude_sky (PIPELINE_DISTANCE_KEYS)
    region: RegionPolicy = "mask"  # from distance.region (PIPELINE_DISTANCE_KEYS)
    # From distance.exclude_semantic: "<segmenter>.<class>" labels, a YAML list
    # or a comma-separated string; normalised to a tuple by __post_init__.
    exclude_semantic: str | tuple[str, ...] = ()
    render: RenderConfig = field(default_factory=RenderConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    device: str = "cuda:0"

    def __post_init__(self) -> None:
        if not self.detectors:
            raise ValueError("detectors: at least one detector is required")
        names: list[str] = []
        for i, spec in enumerate(self.semantic):
            name = spec.get("name")
            if name is None:
                raise ValueError(f"semantic[{i}].name is required")
            if not isinstance(name, str) or not re.fullmatch(
                SEMANTIC_NAME_PATTERN, name
            ):
                raise ValueError(
                    f"semantic[{i}].name: {name!r} must match {SEMANTIC_NAME_PATTERN}"
                )
            if name in names:
                raise ValueError(f"semantic[{i}].name: duplicate name {name!r}")
            names.append(name)
        try:
            groups = group_semantic_labels(self.exclude_semantic, names)
        except ConfigError as exc:
            raise ValueError(f"distance.exclude_semantic: {exc}") from None
        # Canonical form (frozen dataclass), whatever the YAML form was.
        canonical = tuple(f"{n}.{c}" for n, classes in groups.items() for c in classes)
        object.__setattr__(self, "exclude_semantic", canonical)
        overlay = self.render.semantic_overlay or ()
        unknown = [n for n in overlay if n not in names]
        if unknown:
            raise ValueError(
                f"render.semantic_overlay: unknown semantic map {unknown[0]!r}; "
                f"configured: {', '.join(names) or 'none'}"
            )
        for entry in self.render.semantic_classes or ():
            prefix, dot, cls_name = entry.partition(".")
            if not dot or not cls_name:
                raise ValueError(
                    f"render.semantic_classes: {entry!r} must be "
                    "'<map>.<class>' (e.g. scene.road)"
                )
            if prefix not in names:
                raise ValueError(
                    f"render.semantic_classes: unknown semantic map {prefix!r}; "
                    f"configured: {', '.join(names) or 'none'}"
                )
        if self.source.mcap is None:
            cal = self.camera.calibration
            if cal is not None and cal.from_source:
                raise ValueError(
                    "camera.calibration.from_source: needs source.mcap "
                    "(a video has no CameraInfo)"
                )
            if self.rectify.enabled:
                raise ValueError(
                    "rectify.enabled: needs source.mcap (a video has no CameraInfo)"
                )
        if self.rectify.enabled:
            # rectify takes K from the CameraInfo: another intrinsics source
            # would be dropped (from_source is the same source, so it is allowed).
            if self.camera.focal_px is not None:
                raise ValueError(
                    "rectify.enabled: camera.focal_px is exclusive with it "
                    "(rectify uses the CameraInfo K); remove camera.focal_px"
                )
            cal = self.camera.calibration
            if cal is not None and cal.path is not None:
                raise ValueError(
                    "rectify.enabled: camera.calibration.path is exclusive with it "
                    "(rectify uses the CameraInfo); remove camera.calibration"
                )
        if self.render.bev_inset and self.motion is None:
            raise ValueError("render.bev_inset: needs the motion section")

    def to_dict(self) -> dict[str, Any]:
        """Resolved configuration in YAML layout (for metrics.json["config"])."""
        data = _plain(asdict(self))
        for key in PIPELINE_DISTANCE_KEYS:
            data["distance"][key] = data.pop(key)
        return data


def _plain(value: Any) -> Any:
    """JSON/YAML-ready copy: Path -> str, tuple -> list, mappings -> dict."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    return value


def _index(node: list, key: str, override: str) -> int:
    try:
        i = int(key)
    except ValueError:
        raise ConfigError(f"--set {override}: {key!r} is not a list index") from None
    if not -len(node) <= i < len(node):
        raise ConfigError(f"--set {override}: index {i} out of range")
    return i


def apply_overrides(data: dict[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    """Apply "dotted.key=value" overrides; list items by index ("detectors.0.threshold=0.3").

    Values are parsed with yaml.safe_load, so numbers, booleans and null work;
    a mapping or list value replaces the whole section or list
    ("aggregation={type: concat}"). Missing intermediate mappings are created
    (sections may be omitted from the YAML; misspelt keys are still rejected
    when the config is validated).

    Raises:
        ConfigError: malformed override, unparsable value, list index out of
            range, or a path that descends into a scalar.
    """
    for override in overrides:
        dotted, sep, text = override.partition("=")
        if not sep or not dotted:
            raise ConfigError(f"--set {override}: expected KEY=VALUE")
        try:
            value = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ConfigError(f"--set {override}: invalid YAML value ({exc})") from None
        *parents, last = dotted.split(".")
        node: Any = data
        for key in parents:
            if isinstance(node, list):
                node = node[_index(node, key, override)]
            elif isinstance(node, dict):
                if node.get(key) is None:
                    node[key] = {}
                node = node[key]
            else:
                raise ConfigError(f"--set {override}: {key!r} is inside a scalar")
        if isinstance(node, list):
            node[_index(node, last, override)] = value
        elif isinstance(node, dict):
            node[last] = value
        else:
            raise ConfigError(f"--set {override}: {last!r} is inside a scalar")
    return data


def load_config(
    path: Path, overrides: Sequence[str] = (), video: Path | None = None
) -> RunConfig:
    """Read the YAML at `path`, apply overrides and the positional video, validate.

    Order: safe_load -> apply_overrides -> video -> split of
    PIPELINE_DISTANCE_KEYS -> from_mapping(RunConfig).

    Raises:
        ConfigError: unreadable file, invalid YAML, or invalid configuration.
    """
    if not path.is_file():
        raise ConfigError(f"config not found: {path}")
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML ({exc})") from None
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: the top level must be a mapping")
    apply_overrides(data, overrides)
    if video is not None:
        source = data.get("source") or {}
        if not isinstance(source, dict):
            raise ConfigError("source: expected a mapping")
        data["source"] = {**source, "video": str(video)}
    return config_from_mapping(data)


def config_from_mapping(data: dict[str, Any]) -> RunConfig:
    """Split the PIPELINE_DISTANCE_KEYS off `data["distance"]` and validate.

    `data` is the parsed YAML mapping (modified in place: the split keys move
    to the top level, as RunConfig fields).

    Raises:
        ConfigError: invalid configuration (dotted path in the message).
    """
    distance = data.get("distance") or {}
    if not isinstance(distance, dict):
        raise ConfigError("distance: expected a mapping")
    for key in PIPELINE_DISTANCE_KEYS:
        if key in data:
            raise ConfigError(f"{key}: write it under the distance section")
        if key in distance:
            data[key] = distance.pop(key)
    data["distance"] = distance
    return from_mapping(RunConfig, data, "")


def motion_spec_errors(cfg: RunConfig) -> list[ConfigError]:
    """Errors of the motion model and ego-motion specs nested in `cfg.motion`.

    The same `registry.build` calls as `motion.build_motion`, without opening
    the ego-motion data; used to validate a configuration without running it.
    """
    errors: list[ConfigError] = []
    if cfg.motion is None:
        return errors
    for name, spec in cfg.motion.models.items():
        try:
            registry.build("motion", spec, path=f"motion.models.{name}")
        except ConfigError as exc:
            errors.append(exc)
    if cfg.motion.ego_motion is not None:
        try:
            registry.build(
                "ego_motion", cfg.motion.ego_motion, path="motion.ego_motion"
            )
        except ConfigError as exc:
            errors.append(exc)
    return errors


# UI-facing pipeline stages (core.registry.Stage), in lane order; the gaps in
# `order` leave room for stages registered later.
for _stage in (
    Stage(yaml_key="source", title="Source", order=10, config_cls=SourceConfig),
    Stage(
        yaml_key="rectify",
        title="Rectification",
        order=15,
        config_cls=RectifyConfig,
    ),
    Stage(
        yaml_key="detectors",
        title="Detectors",
        order=20,
        kind="detector",
        multiple=True,
    ),
    Stage(yaml_key="aggregation", title="Aggregation", order=30, kind="aggregator"),
    Stage(
        yaml_key="refiners",
        title="Mask refiners",
        order=40,
        kind="refiner",
        multiple=True,
        optional=True,
    ),
    Stage(
        yaml_key="semantic",
        title="Semantic segmentation",
        order=75,
        kind="semantic",
        multiple=True,
        optional=True,
    ),
    Stage(yaml_key="depth", title="Depth", order=50, kind="depth"),
    Stage(
        yaml_key="tracking",
        title="Tracking",
        order=60,
        kind="tracker",
        optional=True,
        group="Tracking & filter",
    ),
    Stage(
        yaml_key="smoothing",
        title="Smoothing",
        order=70,
        config_cls=SmoothingConfig,
        optional=True,
        group="Tracking & filter",
    ),
    Stage(
        yaml_key="motion",
        title="Motion",
        order=80,
        config_cls=MotionConfig,
        optional=True,
    ),
    Stage(
        yaml_key="camera",
        title="Camera",
        order=110,
        config_cls=CameraConfig,
        general=True,
    ),
    Stage(
        yaml_key="distance",
        title="Distance",
        order=120,
        config_cls=DistanceConfig,
        general=True,
        lifted_from=RunConfig,
        lifted=PIPELINE_DISTANCE_KEYS,
    ),
    Stage(
        yaml_key="agreement",
        title="Agreement",
        order=130,
        config_cls=AgreementConfig,
        general=True,
    ),
    Stage(
        yaml_key="render",
        title="Render",
        order=140,
        config_cls=RenderConfig,
        general=True,
    ),
    Stage(
        yaml_key="output",
        title="Output",
        order=150,
        config_cls=OutputConfig,
        general=True,
    ),
):
    register_stage(_stage)
del _stage
