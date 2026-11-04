"""Config presets: built-in setups from `configs/` plus the user's own.

Built-ins are the `configs/*.yaml` files named in the manifest
`configs/presets.yaml` (display order, name, description); they are read-only.
User presets are `<user_root>/<slug>.yaml` with the typed name in a first-line
`# name:` comment. A preset holds the pipeline only: `source` and `output` are
stripped on save and on read. Ids are `builtin:<stem>` and `user:<slug>`; the
client's text only ever reaches the disk through a strict slug and
`web.paths.safe_join`. No HTTP here (the routes live in `web.app`).
"""

import logging
import os
import re
import tempfile
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from det_depth_distance.web.paths import safe_join

logger = logging.getLogger(__name__)

# Sections the server owns (set at run time) and a preset never carries.
EXCLUDED_KEYS: tuple[str, ...] = ("source", "output")
BUILTIN_DIR = Path(__file__).resolve().parents[3] / "configs"
MANIFEST = "presets.yaml"
MAX_NAME_CHARS = 80
MAX_SLUG_CHARS = 60
_NAME_COMMENT = "# name:"
_ID_PART = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


@dataclass(frozen=True, slots=True)
class PresetInfo:
    id: str
    name: str
    description: str
    kind: Literal["builtin", "user"]


class PresetNotFound(LookupError):
    """No preset with that id."""


class PresetExists(Exception):
    """A user preset with that slug exists and `overwrite` was not set."""


class PresetReadOnly(Exception):
    """Built-in presets cannot be changed or deleted."""


class PresetInvalid(ValueError):
    """Bad name, bad id or unreadable file; the message is shown to the user."""


class PresetReserved(PresetInvalid):
    """The name is that of a built-in preset."""


def slugify(name: str) -> str:
    """File-name slug of a preset name: ascii, lowercase, `-` between runs."""
    if len(name) > MAX_NAME_CHARS:
        raise PresetInvalid(
            f"The name is too long (at most {MAX_NAME_CHARS} characters)."
        )
    # C*: controls; Zl/Zp: U+2028/U+2029 are line breaks for PyYAML and would
    # split the `# name:` comment.
    if any(
        unicodedata.category(c).startswith("C")
        or unicodedata.category(c) in ("Zl", "Zp")
        for c in name
    ):
        raise PresetInvalid("The name has control characters.")
    folded = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    slug = (
        re.sub(r"[^a-z0-9]+", "-", folded.lower())
        .strip("-")[:MAX_SLUG_CHARS]
        .strip("-")
    )
    if not slug:
        raise PresetInvalid("The name needs at least one ASCII letter or digit.")
    return slug


def _strip(data: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in data.items() if k not in EXCLUDED_KEYS}


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (yaml.YAMLError, UnicodeDecodeError, OSError) as exc:
        raise PresetInvalid(f"{path.name}: cannot be read as YAML ({exc}).") from exc
    if not isinstance(data, dict):
        raise PresetInvalid(f"{path.name}: the preset must be a YAML mapping.")
    return data


def _manifest(builtin_dir: Path) -> list[dict[str, str]] | None:
    """The valid manifest entries, or None when the manifest is absent or bad."""
    path = builtin_dir / MANIFEST
    if not path.is_file():
        return None
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        entries = raw["presets"]
        if not isinstance(entries, list):
            raise TypeError("'presets' must be a list")
        return [
            {
                "file": str(e["file"]),
                "name": str(e.get("name") or Path(str(e["file"])).stem),
                "description": str(e.get("description") or ""),
            }
            for e in entries
        ]
    except (yaml.YAMLError, OSError, KeyError, TypeError, AttributeError) as exc:
        logger.warning(
            "Preset manifest %s is malformed (%s); listing file stems.", path, exc
        )
        return None


def _builtin_stems(builtin_dir: Path) -> list[str]:
    return sorted(
        p.stem for p in builtin_dir.glob("*.yaml") if p.stem != Path(MANIFEST).stem
    )


def _list_builtin(builtin_dir: Path) -> list[PresetInfo]:
    entries = _manifest(builtin_dir)
    if entries is None:
        return [
            PresetInfo(f"builtin:{stem}", stem, "", "builtin")
            for stem in _builtin_stems(builtin_dir)
        ]
    out: list[PresetInfo] = []
    for e in entries:
        stem = Path(e["file"]).stem
        if (builtin_dir / f"{stem}.yaml").is_file() and stem != Path(MANIFEST).stem:
            out.append(
                PresetInfo(f"builtin:{stem}", e["name"], e["description"], "builtin")
            )
        else:
            logger.warning("Preset manifest lists %s, which does not exist.", e["file"])
    return out


def _file_name(path: Path) -> str:
    """The `# name:` of a user file, or its slug."""
    try:
        with path.open(encoding="utf-8") as fh:
            first = fh.readline()
    except (OSError, UnicodeDecodeError):
        return path.stem
    if first.startswith(_NAME_COMMENT):
        return first[len(_NAME_COMMENT) :].strip() or path.stem
    return path.stem


def _list_user(user_root: Path | None) -> list[PresetInfo]:
    if user_root is None or not user_root.is_dir():
        return []
    infos = [
        PresetInfo(f"user:{p.stem}", _file_name(p), "", "user")
        for p in user_root.glob("*.yaml")
        if p.is_file() and not p.is_symlink() and _ID_PART.fullmatch(p.stem)
    ]
    return sorted(infos, key=lambda i: (i.name.casefold(), i.id))


def list_presets(builtin_dir: Path, user_root: Path | None) -> list[PresetInfo]:
    """Built-ins in manifest order, then user presets sorted by name."""
    return _list_builtin(builtin_dir) + _list_user(user_root)


def _split_id(preset_id: str) -> tuple[str, str]:
    kind, sep, part = preset_id.partition(":")
    if not sep or kind not in ("builtin", "user") or not _ID_PART.fullmatch(part):
        raise PresetNotFound(f"No preset {preset_id!r}.")
    return kind, part


def _user_path(user_root: Path | None, slug: str) -> Path:
    """The file of a user preset; symlinks and escapes count as not found."""
    if user_root is None:
        raise PresetNotFound(f"No preset 'user:{slug}'.")
    try:
        path = safe_join(user_root, f"{slug}.yaml")
    except ValueError as exc:
        raise PresetNotFound(f"No preset 'user:{slug}'.") from exc
    if (user_root / f"{slug}.yaml").is_symlink() or not path.is_file():
        raise PresetNotFound(f"No preset 'user:{slug}'.")
    return path


def read_preset(
    preset_id: str, builtin_dir: Path, user_root: Path | None
) -> dict[str, Any]:
    """The preset's YAML mapping without `source` and `output`."""
    kind, part = _split_id(preset_id)
    if kind == "builtin":
        if part == Path(MANIFEST).stem:
            raise PresetNotFound(f"No preset {preset_id!r}.")
        try:
            path = safe_join(builtin_dir, f"{part}.yaml")
        except ValueError as exc:
            raise PresetNotFound(f"No preset {preset_id!r}.") from exc
        if not path.is_file():
            raise PresetNotFound(f"No preset {preset_id!r}.")
    else:
        path = _user_path(user_root, part)
    return _strip(_load_mapping(path))


def _reserved_slugs(builtin_dir: Path) -> set[str]:
    slugs = set(_builtin_stems(builtin_dir))
    for info in _list_builtin(builtin_dir):
        try:
            slugs.add(slugify(info.name))
        except PresetInvalid:
            pass
    return slugs


def save_preset(
    user_root: Path,
    builtin_dir: Path,
    name: str,
    data: Mapping[str, Any],
    overwrite: bool,
) -> PresetInfo:
    """Write a user preset (pipeline part of `data`) as `<slug>.yaml`."""
    name = name.strip()
    slug = slugify(name)
    if slug in _reserved_slugs(builtin_dir):
        raise PresetReserved(
            f"{name!r} is the name of a built-in preset; choose another name."
        )
    user_root.mkdir(parents=True, exist_ok=True)
    try:
        path = safe_join(user_root, f"{slug}.yaml")
    except ValueError as exc:
        raise PresetInvalid(str(exc)) from exc
    if (user_root / f"{slug}.yaml").is_symlink():
        raise PresetInvalid(f"{slug}.yaml is a symbolic link.")
    text = f"{_NAME_COMMENT} {name}\n" + yaml.safe_dump(
        _strip(data), sort_keys=False, allow_unicode=True
    )
    if overwrite:
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{slug}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(text)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
    else:
        try:
            with path.open("x", encoding="utf-8") as fh:
                fh.write(text)
        except FileExistsError as exc:
            raise PresetExists(f"A preset named {name!r} already exists.") from exc
    return PresetInfo(f"user:{slug}", name, "", "user")


def delete_preset(preset_id: str, user_root: Path | None) -> None:
    """Remove a user preset; built-ins are read-only."""
    kind, part = _split_id(preset_id)
    if kind == "builtin":
        raise PresetReadOnly("Built-in presets cannot be deleted.")
    path = _user_path(user_root, part)
    try:
        path.unlink()
    except FileNotFoundError:  # a concurrent delete won the race
        raise PresetNotFound(f"No preset {preset_id!r}.") from None
