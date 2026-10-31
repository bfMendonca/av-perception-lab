"""Streaming multipart reader for large uploads (videos of hundreds of MB).

Starlette's form parser spools each file to the system temp folder; this
reader writes each wanted file part straight to a folder of the caller's
choice (the library's disk) as the request body arrives, hashing it on the
way, so nothing is held in memory and nothing is copied twice.
"""

import hashlib
from collections.abc import AsyncIterator, Collection
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

from python_multipart.exceptions import MultipartParseError
from python_multipart.multipart import MultipartParser, parse_options_header

MAX_UPLOAD_BYTES = 8 << 30  # a raw 12-minute H.264 is ~700 MB


class UploadError(ValueError):
    """The request is not a well-formed multipart upload (or was cut short)."""


@dataclass
class UploadedFile:
    """One file part written to disk."""

    field: str
    filename: str
    path: Path
    size: int = 0
    sha256: str = ""
    _hash: Any = field(default_factory=hashlib.sha256, repr=False)


async def receive_files(
    content_type: str,
    body: AsyncIterator[bytes],
    dest: Path,
    fields: Collection[str],
    max_bytes: int = MAX_UPLOAD_BYTES,
) -> dict[str, UploadedFile]:
    """Write the file parts named in `fields` under `dest` (`<field>.part`).

    Other parts are skipped; a repeated field keeps its first file.

    Raises:
        UploadError: not multipart, malformed, larger than `max_bytes`, or the
            body ended before the closing boundary.
    """
    ctype, params = parse_options_header(content_type or "")
    boundary = params.get(b"boundary")
    if ctype != b"multipart/form-data" or not boundary:
        raise UploadError("expected a multipart/form-data upload")

    files: dict[str, UploadedFile] = {}
    state: dict[str, Any] = {"headers": {}, "name": b"", "value": b""}
    current: list[UploadedFile | None] = [None]
    handle: list[IO[bytes] | None] = [None]
    total = [0]
    ended = [False]

    def on_part_begin() -> None:
        state["headers"] = {}

    def on_header_field(data: bytes, start: int, end: int) -> None:
        state["name"] += data[start:end]

    def on_header_value(data: bytes, start: int, end: int) -> None:
        state["value"] += data[start:end]

    def on_header_end() -> None:
        state["headers"][state["name"].lower()] = state["value"]
        state["name"], state["value"] = b"", b""

    def on_headers_finished() -> None:
        disposition = state["headers"].get(b"content-disposition", b"")
        _, opts = parse_options_header(disposition)
        name = opts.get(b"name", b"").decode("utf-8", "replace")
        filename = opts.get(b"filename")
        if name in fields and filename is not None and name not in files:
            part = UploadedFile(
                name, filename.decode("utf-8", "replace"), dest / f"{name}.part"
            )
            files[name] = part
            current[0] = part
            handle[0] = open(part.path, "wb")  # noqa: SIM115 - closed in on_part_end

    def on_part_data(data: bytes, start: int, end: int) -> None:
        total[0] += end - start
        if total[0] > max_bytes:
            raise UploadError(f"upload larger than {max_bytes >> 20} MiB")
        part, fh = current[0], handle[0]
        if part is not None and fh is not None:
            chunk = data[start:end]
            fh.write(chunk)
            part._hash.update(chunk)
            part.size += len(chunk)

    def on_part_end() -> None:
        part, fh = current[0], handle[0]
        if part is not None and fh is not None:
            fh.close()
            part.sha256 = part._hash.hexdigest()
        current[0], handle[0] = None, None

    def on_end() -> None:
        ended[0] = True

    parser = MultipartParser(
        boundary,
        {
            "on_part_begin": on_part_begin,
            "on_header_field": on_header_field,
            "on_header_value": on_header_value,
            "on_header_end": on_header_end,
            "on_headers_finished": on_headers_finished,
            "on_part_data": on_part_data,
            "on_part_end": on_part_end,
            "on_end": on_end,
        },
    )
    try:
        async for chunk in body:
            if chunk:
                parser.write(chunk)
        parser.finalize()
    except MultipartParseError as exc:
        raise UploadError(f"malformed multipart upload: {exc}") from None
    finally:
        if handle[0] is not None:
            handle[0].close()
    if not ended[0]:
        raise UploadError("the upload was cut short")
    return files
