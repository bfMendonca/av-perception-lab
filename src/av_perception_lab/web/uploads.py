"""Streaming multipart reader for large uploads (videos of hundreds of MB).

Starlette's form parser spools each file to the system temp folder; this
reader writes each wanted file part straight to a folder of the caller's
choice (the library's disk) as the request body arrives, hashing it on the
way, so nothing is held in memory and nothing is copied twice. Header
blocks, the number of parts and each field's size are bounded, so a
malformed request cannot grow the server's memory either.
"""

import hashlib
from collections.abc import AsyncIterator, Collection, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

from python_multipart.exceptions import MultipartParseError
from python_multipart.multipart import MultipartParser, parse_options_header

MAX_UPLOAD_BYTES = 8 << 30  # a raw 12-minute H.264 is ~700 MB
MAX_HEADER_BYTES = 8 << 10  # one part header (name + value)
MAX_HEADERS = 16  # headers of one part
MAX_PARTS = 8  # parts of one upload (the UI sends at most two)


class UploadError(ValueError):
    """The request is not a well-formed multipart upload (or was cut short);
    `status` is the HTTP status to answer (413 when a limit was passed)."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


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
    field_limits: Mapping[str, int] | None = None,
) -> dict[str, UploadedFile]:
    """Write the file parts named in `fields` under `dest` (`<field>.part`).

    Other parts are skipped; a repeated field keeps its first file.
    `max_bytes` bounds the data of all parts together, `field_limits` the
    data of one field.

    Raises:
        UploadError: not multipart, malformed, a header longer than
            `MAX_HEADER_BYTES`, more than `MAX_HEADERS` headers in a part or
            `MAX_PARTS` parts, larger than `max_bytes` or a field's limit
            (status 413), or the body ended before the closing boundary.
    """
    limits = dict(field_limits or {})
    ctype, params = parse_options_header(content_type or "")
    boundary = params.get(b"boundary")
    if ctype != b"multipart/form-data" or not boundary:
        raise UploadError("expected a multipart/form-data upload")

    files: dict[str, UploadedFile] = {}
    state: dict[str, Any] = {"headers": {}, "name": b"", "value": b""}
    current: list[UploadedFile | None] = [None]
    handle: list[IO[bytes] | None] = [None]
    total = [0]
    parts = [0]
    ended = [False]

    def on_part_begin() -> None:
        parts[0] += 1
        if parts[0] > MAX_PARTS:
            raise UploadError(f"more than {MAX_PARTS} parts in the upload")
        state["headers"] = {}
        state["name"], state["value"] = b"", b""

    def check_header() -> None:
        if len(state["name"]) + len(state["value"]) > MAX_HEADER_BYTES:
            raise UploadError(
                f"a part header is longer than {MAX_HEADER_BYTES >> 10} KiB"
            )

    def on_header_field(data: bytes, start: int, end: int) -> None:
        state["name"] += data[start:end]
        check_header()

    def on_header_value(data: bytes, start: int, end: int) -> None:
        state["value"] += data[start:end]
        check_header()

    def on_header_end() -> None:
        state["headers"][state["name"].lower()] = state["value"]
        state["name"], state["value"] = b"", b""
        if len(state["headers"]) > MAX_HEADERS:
            raise UploadError(f"a part has more than {MAX_HEADERS} headers")

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
            raise UploadError(f"upload larger than {max_bytes >> 20} MiB", 413)
        part, fh = current[0], handle[0]
        if part is not None and fh is not None:
            limit = limits.get(part.field)
            if limit is not None and part.size + end - start > limit:
                raise UploadError(
                    f"the {part.field!r} part is larger than {limit >> 20} MiB", 413
                )
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
