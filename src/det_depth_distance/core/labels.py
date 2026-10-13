"""Class vocabularies and class-filter presets.

A vocabulary is identified from the loaded model's own class names (names
come from the checkpoint; e.g. the Objects365 YOLO26 checkpoint uses lower
case names that differ from ultralytics' Objects365.yaml). Presets such as
"driving" are defined per vocabulary.

`LabelMap` harmonises vocabularies: a per-detector YAML maps raw class names
to canonical ones (e.g. Objects365 "pickup truck", "suv" and COCO "truck" ->
"vehicle") so that instances of different models can be compared and fused.
"""

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, get_args

import numpy as np
import yaml

from det_depth_distance.core.options import ConfigError


@dataclass(frozen=True, slots=True)
class Vocabulary:
    """A known label set, recognised by its presets' names."""

    name: str
    # preset -> exact class names of the checkpoint
    presets: Mapping[str, tuple[str, ...]]

    def fits(self, class_names: Collection[str], preset: str = "driving") -> bool:
        """Whether every name of `preset` is a class of the model."""
        return set(self.presets[preset]) <= set(class_names)


# COCO has no generic "traffic sign" class (only "stop sign").
COCO = Vocabulary(
    "coco",
    {
        "driving": (
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "bus",
            "truck",
            "traffic light",
            "stop sign",
        )
    },
)
# Names as spelled by the yolo26*-objv1-150.pt checkpoints (lower case).
OBJECTS365 = Vocabulary(
    "objects365",
    {
        "driving": (
            "person",
            "bicycle",
            "car",
            "suv",
            "van",
            "pickup truck",
            "sports car",
            "truck",
            "bus",
            "motorcycle",
            "tricycle",
            "machinery vehicle",
            "fire truck",
            "ambulance",
            "traffic light",
            "traffic sign",
            "stop sign",
            "crosswalk sign",
            "traffic cone",
            "street lights",
            "fire hydrant",
        )
    },
)
# Most specific first: the Objects365 preset contains "suv", absent from COCO.
VOCABULARIES: tuple[Vocabulary, ...] = (OBJECTS365, COCO)


def identify_vocabulary(class_names: Collection[str]) -> Vocabulary | None:
    """First known vocabulary whose "driving" preset is contained in `class_names`."""
    return next((v for v in VOCABULARIES if v.fits(class_names)), None)


def resolve_classes(
    spec: str | Sequence[str], class_names: Sequence[str]
) -> frozenset[str] | None:
    """Class filter from the `classes` option: "all" (None = keep everything),
    a preset name of the model's vocabulary (e.g. "driving"), a YAML list or a
    comma-separated string of class names.

    Raises:
        ConfigError: preset undefined for this model's vocabulary, or a name
            that is not a class of the loaded model (names are case-sensitive).
    """
    if spec == "all":
        return None
    if isinstance(spec, str):
        if spec in {p for v in VOCABULARIES for p in v.presets}:
            vocabulary = identify_vocabulary(class_names)
            if vocabulary is None or spec not in vocabulary.presets:
                raise ConfigError(
                    f"preset {spec!r} undefined for this model's classes; list class names"
                )
            return frozenset(vocabulary.presets[spec])
        wanted = {name.strip() for name in spec.split(",") if name.strip()}
    else:
        wanted = set(spec)
    unknown = wanted - set(class_names)
    if unknown:
        raise ConfigError(
            f"classes not known to the model: {sorted(unknown)} (use classes: all)"
        )
    return frozenset(wanted)


UnmappedPolicy = Literal["keep", "drop"]


@dataclass(frozen=True, slots=True)
class LabelMap:
    """Raw -> canonical class-name mapping for one detector, loaded from YAML.

    Schema:
        vocabulary: coco | objects365   # optional; checked against the loaded model
        unmapped: keep | drop           # labels not listed: keep the raw name or drop the instance
        canonical: {canonical_name: [raw_name, ...]}

    Raw names are validated against the model's own class names at load time,
    so a typo or a map written for another vocabulary fails before inference.
    """

    name: str  # file stem, shown in describe() / metrics.json
    mapping: Mapping[str, str]  # raw -> canonical
    unmapped: UnmappedPolicy = "keep"
    vocabulary: str | None = None  # a VOCABULARIES name; None = not checked

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, name: str, path: str) -> "LabelMap":
        """Validate the YAML schema (free-form class-name keys, hence not
        `from_mapping`) and invert `canonical` into raw -> canonical.

        Raises:
            ConfigError: unknown key, bad vocabulary/unmapped value, empty or
                malformed `canonical`, or a raw name mapped to two canonical
                names; the message starts with `path`.
        """
        if not isinstance(data, Mapping):
            raise ConfigError(f"{path}: expected a mapping, got {data!r}")
        keys = ("vocabulary", "unmapped", "canonical")
        unknown = [k for k in data if k not in keys]
        if unknown:
            raise ConfigError(
                f"{path}.{unknown[0]}: unknown key; valid: {', '.join(keys)}"
            )
        vocabulary = data.get("vocabulary")
        known = [v.name for v in VOCABULARIES]
        if vocabulary is not None and vocabulary not in known:
            raise ConfigError(
                f"{path}.vocabulary: {vocabulary!r} is not one of {known}"
            )
        unmapped = data.get("unmapped", "keep")
        if unmapped not in get_args(UnmappedPolicy):
            raise ConfigError(
                f"{path}.unmapped: {unmapped!r} is not one of "
                f"{list(get_args(UnmappedPolicy))}"
            )
        canonical = data.get("canonical")
        if not isinstance(canonical, Mapping) or not canonical:
            raise ConfigError(f"{path}.canonical: expected a non-empty mapping")
        mapping: dict[str, str] = {}
        for target, raws in canonical.items():
            if (
                not isinstance(target, str)
                or not isinstance(raws, list)
                or not raws
                or not all(isinstance(r, str) for r in raws)
            ):
                raise ConfigError(
                    f"{path}.canonical.{target}: expected a non-empty list of names"
                )
            for raw in raws:
                if mapping.setdefault(raw, target) != target:
                    raise ConfigError(
                        f"{path}: {raw!r} mapped to both {mapping[raw]!r} and {target!r}"
                    )
        return cls(name, mapping, unmapped, vocabulary)

    @classmethod
    def load(cls, path: Path) -> "LabelMap":
        """Read and validate the label map YAML at `path`.

        Raises:
            ConfigError: missing file, invalid YAML or schema (see `from_dict`).
        """
        if not path.is_file():
            raise ConfigError(f"{path}: label map not found")
        try:
            data = yaml.safe_load(path.read_text())
        except yaml.YAMLError as exc:
            raise ConfigError(f"{path}: invalid YAML ({exc})") from None
        return cls.from_dict(data, name=path.stem, path=str(path))

    @property
    def canonical(self) -> frozenset[str]:
        """Canonical names this map produces (unmapped raw names not included)."""
        return frozenset(self.mapping.values())

    def check(self, class_names: Collection[str], vocabulary: str) -> None:
        """Validate the map against a loaded model (same strict policy as
        `resolve_classes`).

        Raises:
            ConfigError: the declared vocabulary differs from the model's (a COCO
                map on an Objects365 model, where "truck" means something else),
                or a raw name is not a class of the model.
        """
        if self.vocabulary is not None and self.vocabulary != vocabulary:
            raise ConfigError(
                f"{self.name}: written for {self.vocabulary!r}, "
                f"the model's vocabulary is {vocabulary!r}"
            )
        unknown = set(self.mapping) - set(class_names)
        if unknown:
            raise ConfigError(
                f"{self.name}: raw names not known to the model: {sorted(unknown)}"
            )

    def apply(self, raw_labels: Sequence[str]) -> tuple[tuple[str, ...], np.ndarray]:
        """(canonical labels, keep (N,) bool); keep is False only for unmapped
        labels under `unmapped: drop` (their label stays the raw name)."""
        labels = tuple(self.mapping.get(raw, raw) for raw in raw_labels)
        keep = np.array(
            [self.unmapped == "keep" or raw in self.mapping for raw in raw_labels],
            dtype=bool,
        )
        return labels, keep

    def describe(self) -> dict[str, Any]:
        """JSON-ready form for metrics.json, in the YAML layout."""
        canonical: dict[str, list[str]] = {}
        for raw, target in self.mapping.items():
            canonical.setdefault(target, []).append(raw)
        return {
            "name": self.name,
            "vocabulary": self.vocabulary,
            "unmapped": self.unmapped,
            "canonical": canonical,
        }
