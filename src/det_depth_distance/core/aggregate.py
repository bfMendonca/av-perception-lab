"""Combination of the instances of several detectors into one list.

Aggregators are not Components (nothing to load, no name). Cross-model
fusion (NMS/WBF) arrives as further registered subclasses.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, ClassVar

from det_depth_distance.core.registry import register
from det_depth_distance.core.types import Instances


class Aggregator(ABC):
    """Merges per-detector instances (in detector order) into one Instances."""

    type_name: ClassVar[str]  # set by @register
    Options: ClassVar[type]

    def __init__(self, options: Any) -> None:
        self.options = options

    @abstractmethod
    def __call__(self, parts: Sequence[Instances]) -> Instances: ...

    def bind_sources(self, names: Sequence[str]) -> None:
        """Receive the detector names, in order, once at pipeline construction.

        No-op here; fusing aggregators override it to validate per-source settings.
        """

    def describe(self) -> dict[str, Any]:
        """JSON-ready description for metrics.json."""
        return {"type": self.type_name, "options": asdict(self.options)}


@register("aggregator", "concat")
class Concat(Aggregator):
    """Keep every instance of every detector, tagged by `source`. Scores of
    different models are not comparable, so nothing is merged or suppressed."""

    @dataclass(frozen=True, slots=True)
    class Options:
        pass

    def __call__(self, parts: Sequence[Instances]) -> Instances:
        return Instances.concat(parts)
