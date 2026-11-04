import shutil
import struct
import subprocess

import pytest

from av_perception_lab.io.quicktime import camera_metadata, read_quicktime_metadata

LENS = "iPhone 14 Pro Max back triple camera 6.86mm f/1.78"


def atom(kind: bytes, payload: bytes, *, large: bool = False) -> bytes:
    if large:  # 64-bit size after the type
        return struct.pack(">I4sQ", 1, kind, 16 + len(payload)) + payload
    return struct.pack(">I4s", 8 + len(payload), kind) + payload


def data(type_code: int, value: bytes) -> bytes:
    return atom(b"data", struct.pack(">I", type_code) + b"\0" * 4 + value)


def meta(items: dict[str, tuple[int, bytes]], *, iso: bool = False) -> bytes:
    """QuickTime mdta meta (hdlr, keys, ilst); `iso` adds version/flags."""
    keys = b"".join(struct.pack(">I", 8 + len(k)) + b"mdta" + k.encode() for k in items)
    ilst = b"".join(
        atom(struct.pack(">I", i), data(code, value))
        for i, (code, value) in enumerate(items.values(), start=1)
    )
    body = (
        atom(b"hdlr", b"\0" * 8 + b"mdta" + b"\0" * 13)
        + atom(b"keys", b"\0" * 4 + struct.pack(">I", len(items)) + keys)
        + atom(b"ilst", ilst)
    )
    return atom(b"meta", (b"\0" * 4 if iso else b"") + body)


def iphone_moov() -> bytes:
    """moov with the file-level and the track-level metas of an iPhone video."""
    top = meta(
        {
            "com.apple.quicktime.make": (1, b"Apple"),
            "com.apple.quicktime.model": (1, b"iPhone 14 Pro Max"),
            "com.apple.quicktime.software": (1, b"27.0.1"),
        }
    )
    track = meta(
        {
            "com.apple.quicktime.camera.lens_model": (1, LENS.encode()),
            "com.apple.quicktime.camera.focal_length.35mm_equivalent": (
                21,
                struct.pack(">i", 24),
            ),
            "com.apple.quicktime.camera.lens_irisfnumber": (1, b"F1.78"),
            "com.apple.quicktime.apple-maker-note.74": (22, struct.pack(">H", 2)),
            "com.example.float": (23, struct.pack(">f", 1.5)),
            "com.example.unknown": (99, b"xx"),
        }
    )
    trak = atom(b"trak", atom(b"tkhd", b"\0" * 84) + track)
    return atom(b"moov", atom(b"mvhd", b"\0" * 100) + top + trak)


EXPECTED = {
    "make": "Apple",
    "model": "iPhone 14 Pro Max",
    "software": "27.0.1",
    "lens_model": LENS,
    "focal_length_35mm": 24,
    "lens_f_number": "F1.78",
}


def write(tmp_path, *atoms: bytes):
    path = tmp_path / "clip.mov"
    path.write_bytes(b"".join(atoms))
    return path


def test_track_and_file_metas_after_a_large_mdat(tmp_path):
    ftyp = atom(b"ftyp", b"qt  " + b"\0" * 4)
    path = write(tmp_path, ftyp, atom(b"mdat", b"\0" * (1 << 20)), iphone_moov())
    raw = read_quicktime_metadata(path)
    assert raw["com.apple.quicktime.apple-maker-note.74"] == 2
    assert raw["com.example.float"] == pytest.approx(1.5)
    assert "com.example.unknown" not in raw  # unknown data types are skipped
    assert camera_metadata(path) == EXPECTED


def test_moov_first_and_64_bit_sizes(tmp_path):
    moov = iphone_moov()
    path = write(
        tmp_path,
        atom(b"moov", moov[8:], large=True),
        atom(b"mdat", b"\0" * 64, large=True),
    )
    assert camera_metadata(path) == EXPECTED


def test_iso_meta_with_version_flags(tmp_path):
    moov = atom(b"moov", meta({"com.apple.quicktime.model": (1, b"X")}, iso=True))
    assert camera_metadata(write(tmp_path, moov)) == {"model": "X"}


def test_unreadable_files_give_nothing(tmp_path):
    text = tmp_path / "notes.txt"
    text.write_text("hello, this is not a QuickTime file at all")
    assert read_quicktime_metadata(text) == {}
    assert read_quicktime_metadata(tmp_path / "missing.mov") == {}
    moov = iphone_moov()
    truncated = write(tmp_path, atom(b"ftyp", b"qt  "), moov[: len(moov) // 2])
    assert read_quicktime_metadata(truncated) == {}
    no_moov = write(tmp_path, atom(b"ftyp", b"qt  "), atom(b"mdat", b"\0" * 16))
    assert read_quicktime_metadata(no_moov) == {}


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_ffmpeg_written_udta_meta(tmp_path):
    # ffmpeg writes mdta keys in moov/udta/meta: a real muxer's layout.
    path = tmp_path / "tagged.mov"
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=10",
            "-t", "0.5", "-pix_fmt", "yuv420p", "-movflags", "use_metadata_tags",
            "-metadata", "com.apple.quicktime.model=iPhone 14 Pro Max",
            "-metadata", f"com.apple.quicktime.camera.lens_model={LENS}",
            str(path),
        ],
        check=True,
    )  # fmt: skip
    assert camera_metadata(path) == {"model": "iPhone 14 Pro Max", "lens_model": LENS}
