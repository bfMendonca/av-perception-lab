"""Class vocabularies and class-filter presets.

A vocabulary is identified from the loaded model's own class names (names
come from the checkpoint; e.g. the Objects365 YOLO26 checkpoint uses lower
case names that differ from ultralytics' Objects365.yaml). Presets such as
"driving" are defined per vocabulary. Base for the cross-model LabelMap.
"""

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

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
