"""Process environment for every Ultralytics-based adapter (YOLO, SAM).

Ultralytics reads YOLO_OFFLINE, YOLO_AUTOINSTALL and YOLO_CONFIG_DIR when it
is imported, so each adapter module calls `apply_ultralytics_env()` at its
own import, before the lazy `import ultralytics` in `_load`: otherwise it may
pip-install missing packages, send usage events, and write settings.json to
the user's config dir. Values already in the environment win (setdefault).
"""

import os

from det_depth_distance.runtime import REPO


def apply_ultralytics_env() -> None:
    """Set (setdefault) the offline, no-autoinstall, repo-local settings policy."""
    # No DNS probe, usage events or version check.
    os.environ.setdefault("YOLO_OFFLINE", "1")
    os.environ.setdefault("YOLO_AUTOINSTALL", "False")  # never pip-install at run time
    os.environ.setdefault("YOLO_CONFIG_DIR", str(REPO / "_testbed" / "config"))
