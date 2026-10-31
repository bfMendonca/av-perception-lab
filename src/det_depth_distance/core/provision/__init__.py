"""Provisioning: obtain what a `Requirements` declares (see `engine`).

Importing the package registers the built-in sources (`sources`): Hugging
Face, URL, git clone and user-provided.
"""

from det_depth_distance.core.provision import sources as _sources  # noqa: F401
from det_depth_distance.core.provision.engine import (
    ItemProgress,
    Progress,
    ProvisionCancelled,
    ProvisionError,
    Source,
    prepare,
    register_source,
    source_for,
)

__all__ = [
    "ItemProgress",
    "Progress",
    "ProvisionCancelled",
    "ProvisionError",
    "Source",
    "prepare",
    "register_source",
    "source_for",
]
