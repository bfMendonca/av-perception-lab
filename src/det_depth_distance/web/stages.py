"""Stage descriptors for the catalog, read from `core.registry`.

There is no table here: stages are registered next to the pipeline code
(`config.py`), so a new stage shows up without touching this package.
"""

from typing import Any

from det_depth_distance import config  # noqa: F401  (registers the built-in stages)
from det_depth_distance.core import registry


def stage_list() -> list[dict[str, Any]]:
    """The registered stages in lane order, as JSON-ready mappings."""
    return [
        {
            "yaml_key": s.yaml_key,
            "title": s.title,
            "kind": s.kind,
            "multiple": s.multiple,
            "optional": s.optional,
            "order": s.order,
            "general": s.general,
            "has_section": s.config_cls is not None,
        }
        for s in registry.stages()
    ]
