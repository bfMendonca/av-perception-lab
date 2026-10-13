"""YAML run configuration: load, apply --set overrides, validate shape.

Sections are dataclasses built with `from_mapping` (unknown keys are errors,
reported with their dotted path); domain rules live in each dataclass's
__post_init__. Component specs (detectors, depth, aggregation) are validated
by their adapter's Options when built through the registry. Pipeline-policy
keys written in the `distance` section (PIPELINE_DISTANCE_KEYS, e.g.
exclude_sky) are split off into RunConfig fields before DistanceConfig is
built.
"""

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import cv2
import yaml

from det_depth_distance.adapters.depth.da3 import ProcessResMethod
from det_depth_distance.core.options import ConfigError, from_mapping
from det_depth_distance.distance import DistanceConfig


@dataclass(frozen=True, slots=True)
class SourceConfig:
    """Input video and the processed segment."""

    video: Path | None = None
    start: float = 0.0  # s
    duration: float = 0.0  # s, 0 = to the end
    max_frames: int = 0  # 0 = all
    stride: int = 1  # process every N-th frame

    def __post_init__(self) -> None:
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


@dataclass(frozen=True, slots=True)
class CameraConfig:
    """Camera intrinsics: a known focal (precedence) or its estimation."""

    focal_px: float | None = None  # focal length in pixels of the input video
    estimate: EstimateConfig | None = field(default_factory=EstimateConfig)

    def __post_init__(self) -> None:
        if self.focal_px is not None and self.focal_px <= 0:
            raise ValueError("focal_px must be > 0 (null = estimate)")


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

    def __post_init__(self) -> None:
        if self.max_depth <= 0:
            raise ValueError("max_depth must be > 0")
        if not hasattr(cv2, f"COLORMAP_{self.cmap.upper()}"):
            raise ValueError(f"unknown OpenCV colormap {self.cmap!r}")


@dataclass(frozen=True, slots=True)
class OutputConfig:
    """Where the run folder goes and what it holds."""

    out_root: Path = Path("_testbed/outputs")
    tag: str = ""  # suffix of the run folder name
    save_depth: bool = False  # also write depth.npz (float16, all frames in RAM)


# Pipeline-policy keys written in the YAML `distance` section that are not
# DistanceConfig fields; load_config moves them to the RunConfig field of the
# same name.
PIPELINE_DISTANCE_KEYS: tuple[str, ...] = ("exclude_sky",)


@dataclass(frozen=True, slots=True)
class RunConfig:
    """The whole run, as described by the YAML file."""

    detectors: tuple[Mapping[str, Any], ...]  # specs, validated by registry.build
    depth: Mapping[str, Any]
    source: SourceConfig = field(default_factory=SourceConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    # Flat spec {type, **options}, validated by registry.build("aggregator", ...).
    aggregation: Mapping[str, Any] = field(default_factory=lambda: {"type": "concat"})
    distance: DistanceConfig = field(default_factory=DistanceConfig)
    exclude_sky: bool = True  # from distance.exclude_sky (PIPELINE_DISTANCE_KEYS)
    render: RenderConfig = field(default_factory=RenderConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    device: str = "cuda:0"

    def __post_init__(self) -> None:
        if not self.detectors:
            raise ValueError("detectors: at least one detector is required")

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
