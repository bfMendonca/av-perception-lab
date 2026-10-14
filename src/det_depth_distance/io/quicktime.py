"""Minimal QuickTime/MP4 metadata reader (stdlib only).

Reads the `keys` + `ilst` pairs of moov/meta and of every meta nested in
moov's trak/udta/mdia containers. iPhone videos keep the camera description
(lens model, 35 mm equivalent focal length) in the track-level meta, which
ffprobe does not report. Only the moov atom is read (mdat is skipped with
seek). Anything unreadable yields {}: metadata is advisory, never required.
"""

import struct
from pathlib import Path

APPLE_KEY_PREFIX = "com.apple.quicktime."
# QuickTime key -> normalised name used by camera_metadata and calibration files.
CAMERA_KEYS: dict[str, str] = {
    APPLE_KEY_PREFIX + key: name
    for key, name in (
        ("make", "make"),
        ("model", "model"),
        ("software", "software"),
        ("camera.lens_model", "lens_model"),
        ("camera.focal_length.35mm_equivalent", "focal_length_35mm"),
        ("camera.lens_irisfnumber", "lens_f_number"),
    )
}

# Containers searched for meta atoms (moov/meta, moov/trak/meta, moov/udta/meta...).
_CONTAINERS = frozenset({b"moov", b"trak", b"udta", b"mdia"})
# The moov of a phone video is tens of kB; refuse absurd sizes from corrupt headers.
_MAX_MOOV_BYTES = 64 << 20

Value = str | int | float


class _Malformed(Exception):
    """An atom runs past its parent or a payload is too short."""


def _atoms(buf: bytes, start: int, end: int):
    """(type, payload start, payload end) of the atoms in buf[start:end]."""
    off = start
    while off + 8 <= end:
        size, kind = struct.unpack_from(">I4s", buf, off)
        header = 8
        if size == 1:  # 64-bit size follows the type
            if off + 16 > end:
                raise _Malformed
            (size,) = struct.unpack_from(">Q", buf, off + 8)
            header = 16
        elif size == 0:  # extends to the end of the parent
            size = end - off
        if size < header or off + size > end:
            raise _Malformed
        yield kind, off + header, off + size
        off += size


def _decode(type_code: int, value: bytes) -> Value | None:
    """Well-known QuickTime data types; others are ignored (None)."""
    if type_code == 1:
        return value.decode("utf-8", errors="replace")
    if type_code in (21, 22) and len(value) in (1, 2, 4, 8):
        return int.from_bytes(value, "big", signed=type_code == 21)
    if type_code == 23 and len(value) == 4:
        return struct.unpack(">f", value)[0]
    if type_code == 24 and len(value) == 8:
        return struct.unpack(">d", value)[0]
    return None


def _read_meta(buf: bytes, start: int, end: int, out: dict[str, Value]) -> None:
    # QuickTime meta has no version/flags; the ISO (MP4) form has 4 bytes of them.
    if buf[start + 4 : start + 8] != b"hdlr" and buf[start + 8 : start + 12] == b"hdlr":
        start += 4
    keys: list[str] = []
    items: list[tuple[int, int, int]] = []
    for kind, p0, p1 in _atoms(buf, start, end):
        if kind == b"keys":
            if p1 - p0 < 8:
                raise _Malformed
            (count,) = struct.unpack_from(">I", buf, p0 + 4)  # after version/flags
            off = p0 + 8
            for _ in range(count):
                if off + 8 > p1:
                    raise _Malformed
                (size,) = struct.unpack_from(
                    ">I", buf, off
                )  # includes size + namespace
                if size < 8 or off + size > p1:
                    raise _Malformed
                keys.append(buf[off + 8 : off + size].decode("utf-8", errors="replace"))
                off += size
        elif kind == b"ilst":
            for index, i0, i1 in _atoms(buf, p0, p1):
                items.append((int.from_bytes(index, "big"), i0, i1))
    for index, i0, i1 in items:
        if not 1 <= index <= len(keys):  # 1-based index into keys
            continue
        for kind, d0, d1 in _atoms(buf, i0, i1):
            if kind != b"data" or d1 - d0 < 8:
                continue
            # 1 byte type set + 3 bytes type code, then 4 bytes locale.
            type_code = int.from_bytes(buf[d0 + 1 : d0 + 4], "big")
            value = _decode(type_code, buf[d0 + 8 : d1])
            if value is not None:
                out[keys[index - 1]] = value
            break


def _walk(buf: bytes, start: int, end: int, out: dict[str, Value]) -> None:
    for kind, p0, p1 in _atoms(buf, start, end):
        if kind == b"meta":
            _read_meta(buf, p0, p1, out)
        elif kind in _CONTAINERS:
            _walk(buf, p0, p1, out)


def _read_moov(path: Path) -> bytes | None:
    """Payload of the moov atom, found by seeking over the top-level atoms."""
    with open(path, "rb") as fh:
        while True:
            header = fh.read(8)
            if len(header) < 8:
                return None
            size, kind = struct.unpack(">I4s", header)
            header_len = 8
            if size == 1:
                large = fh.read(8)
                if len(large) < 8:
                    return None
                (size,) = struct.unpack(">Q", large)
                header_len = 16
            if kind == b"moov":
                if size == 0:  # last atom: to the end of the file
                    return fh.read(_MAX_MOOV_BYTES)
                if not header_len <= size <= _MAX_MOOV_BYTES:
                    return None
                body = fh.read(size - header_len)
                # A truncated moov yields nothing rather than a part of it.
                return body if len(body) == size - header_len else None
            if size == 0 or size < header_len:
                return None
            fh.seek(size - header_len, 1)


def read_quicktime_metadata(path: Path) -> dict[str, Value]:
    """Every mdta key of moov/meta and of the metas nested in moov's containers.

    Later atoms win on duplicate keys (track metas come after moov/meta in
    iPhone files, but the keys read here do not collide). Returns {} for a
    file that is not QuickTime/MP4, has no moov, or is truncated or corrupt.
    """
    try:
        moov = _read_moov(Path(path))
        if moov is None:
            return {}
        out: dict[str, Value] = {}
        _walk(moov, 0, len(moov), out)
        return out
    except (OSError, _Malformed, struct.error):
        return {}


def camera_metadata(path: Path) -> dict[str, Value]:
    """Camera description of a recording, normalised: make, model, software,
    lens_model, focal_length_35mm, lens_f_number (only the keys present)."""
    raw = read_quicktime_metadata(path)
    return {name: raw[key] for key, name in CAMERA_KEYS.items() if key in raw}
