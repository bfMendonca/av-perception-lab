"""Provisioning: obtain what a `Requirements` declares (see `engine`).

Importing the package registers the built-in sources (`sources`): Hugging
Face, URL, git clone and user-provided. `packages` handles the packages
layer (detect, dry-run, install; never the base stack).
"""

from av_perception_lab.core.provision import sources as _sources  # noqa: F401
from av_perception_lab.core.provision.engine import (
    ItemProgress,
    PackagesOutcome,
    Progress,
    ProvisionCancelled,
    ProvisionError,
    Source,
    install_packages,
    prepare,
    register_source,
    source_for,
    with_package_plan,
)

__all__ = [
    "ItemProgress",
    "PackagesOutcome",
    "Progress",
    "ProvisionCancelled",
    "ProvisionError",
    "Source",
    "install_packages",
    "prepare",
    "register_source",
    "source_for",
    "with_package_plan",
]
