"""Ultralytics weight files: location under the weights root and known assets.

YOLO and Ultralytics SAM share one convention: a bare file name (no directory
part, e.g. `yolo26l.pt`) resolves under `<weights root>/ultralytics/`; a path
with a directory part keeps its old meaning (absolute, `~` expanded, or
relative to the working directory, like every other YAML path). Files named in
`ULTRALYTICS_ASSETS` (the `ultralytics/assets` v8.4.0 release, the table
`ultralytics.utils.downloads.GITHUB_ASSETS_NAMES` of 8.4.175; sizes read with
HTTP HEAD in task_0021) are declared as `UrlAsset`; any other file is
`UserProvided`. Pure stdlib (ultralytics is imported by the adapters' `_load`).
"""

from pathlib import Path

from det_depth_distance.adapters._weights import under_root
from det_depth_distance.core.requirements import (
    Asset,
    UrlAsset,
    UserProvided,
    weights_root,
)

ULTRALYTICS_DIR = "ultralytics"  # <weights root>/ultralytics/<file>
ASSETS_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0"
YOLO_LICENSE = "AGPL-3.0"
SAM_LICENSE = "Apache-2.0"  # upstream SAM / SAM 2 / MobileSAM weights

# file name -> (bytes, licence)
ULTRALYTICS_ASSETS: dict[str, tuple[int, str]] = {
    **{
        name: (size, YOLO_LICENSE)
        for name, size in {
            "yolo26n.pt": 5_544_453,
            "yolo26s.pt": 20_422_725,
            "yolo26m.pt": 44_255_705,
            "yolo26l.pt": 53_211_173,
            "yolo26x.pt": 118_667_365,
            "yolo26n-seg.pt": 6_719_965,
            "yolo26s-seg.pt": 23_467_933,
            "yolo26m-seg.pt": 54_750_385,
            "yolo26l-seg.pt": 63_700_037,
            "yolo26x-seg.pt": 142_129_861,
            "yolo26n-objv1-150.pt": 5_975_045,
            "yolo26s-objv1-150.pt": 20_872_197,
            "yolo26m-objv1-150.pt": 45_141_337,
            "yolo26l-objv1-150.pt": 54_105_317,
            "yolo26x-objv1-150.pt": 120_006_117,
            "yolo26n-objv1-seg.pt": 7_185_309,
            "yolo26s-objv1-seg.pt": 24_002_461,
            "yolo26m-objv1-seg.pt": 55_804_721,
            "yolo26l-objv1-seg.pt": 64_758_405,
            "yolo26x-objv1-seg.pt": 143_707_397,
        }.items()
    },
    **{
        name: (size, SAM_LICENSE)
        for name, size in {
            "mobile_sam.pt": 40_728_226,
            "sam_b.pt": 375_042_383,
            "sam_l.pt": 1_249_524_607,
            "sam2_t.pt": 78_064_050,
            "sam2_s.pt": 92_278_178,
            "sam2_b.pt": 161_894_082,
            "sam2_l.pt": 449_203_114,
            "sam2.1_t.pt": 78_105_722,
            "sam2.1_s.pt": 92_319_866,
            "sam2.1_b.pt": 161_935_802,
            "sam2.1_l.pt": 449_239_354,
        }.items()
    },
}


def _is_bare_name(weights: Path) -> bool:
    return not weights.is_absolute() and len(weights.parts) == 1


def check_weights(weights: Path) -> None:
    """ValueError for a `weights` that is not a file name (empty, `.`, `..`)."""
    weights = Path(weights).expanduser()
    if not weights.parts or weights.name in ("", ".", ".."):
        raise ValueError(f"weights: {str(weights)!r} is not a file name")


def resolve_weights(weights: Path) -> Path:
    """Where a `weights` option points: a bare file name -> under
    `<weights root>/ultralytics/`; any other path is returned unchanged
    (`~` expanded; a relative one is opened from the working directory)."""
    check_weights(weights)
    weights = Path(weights).expanduser()
    if _is_bare_name(weights):
        return weights_root() / ULTRALYTICS_DIR / weights
    return weights


def weights_asset(weights: Path, hint: str) -> Asset | None:
    """`UrlAsset` for a known release file name, `UserProvided` for any other
    file inside the weights root, None for a file outside it."""
    path = resolve_weights(weights)
    target = under_root(path)
    if target is None:
        return None
    known = ULTRALYTICS_ASSETS.get(path.name)
    if known is None:
        return UserProvided(
            key="weights", target=target, option="weights", hint=f"{hint} at {path}"
        )
    size, licence = known
    return UrlAsset(
        key="weights",
        target=target,
        url=f"{ASSETS_URL}/{path.name}",
        size_bytes=size,
        license=licence,
    )
