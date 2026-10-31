"""Video library: probing, dedupe, rejection, imports and `.h264` remux.

Every clip is a tiny synthetic video (a few KB) made by the conftest factory.
"""

import io
import json
import re
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from det_depth_distance.web import library
from det_depth_distance.web.library import InvalidVideo, Library, OutsideRoots


@pytest.fixture
def lib(web_paths):
    lib = Library(web_paths.library_root, web_paths.data_roots)
    yield lib
    lib.close()


def files_under(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file()] if root.exists() else []


def packet_times(path: Path) -> list[float]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries"]
        + ["packet=pts_time", "-of", "json", str(path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return sorted(float(p["pts_time"]) for p in json.loads(out)["packets"])


def write_meta(path: Path, times_ms: list[float]) -> Path:
    # Same layout as rpicam-vid's --metadata output (keys in varying order).
    t0 = 56216714000
    items = [
        f'{{\n    "ExposureTime": 1993,\n    "SensorTimestamp": {t0 + int(t * 1e6)},'
        f'\n    "FrameWallClock": {1793354829505088 + i}\n}}'
        for i, t in enumerate(times_ms)
    ]
    path.write_text("[\n" + ",\n".join(items) + "\n]\n")
    return path


def test_probe_reads_fps_size_duration(video_factory):
    info = library.probe(video_factory("p.mp4", seconds=2, fps=10, size="64x48"))
    assert (info.width, info.height) == (64, 48)
    assert info.fps == pytest.approx(10)
    assert info.duration_s == pytest.approx(2.0, abs=0.05)


def test_add_dedupes_by_hash(lib, video_factory):
    clip = video_factory("clip.mp4")
    with open(clip, "rb") as fh:
        first = lib.add(fh, "clip.mp4")
    with open(clip, "rb") as fh:
        again = lib.add(fh, "renamed.mp4")
    assert re.fullmatch(r"[0-9a-f]{12}", first.id)
    assert again.id == first.id
    assert [r.id for r in lib.list()] == [first.id]
    assert first.name == "clip.mp4"
    assert first.size_bytes == clip.stat().st_size
    assert first.original.is_file() and first.original.read_bytes() == clip.read_bytes()
    assert first.thumb.is_file() and first.thumb.read_bytes()[:2] == b"\xff\xd8"
    assert lib.get(first.id).id == first.id


def test_add_rejects_text_file_and_leaves_no_entry(lib, web_paths):
    with pytest.raises(InvalidVideo):
        lib.add(io.BytesIO(b"just some notes\n" * 10), "notes.mp4")
    assert lib.list() == []
    assert files_under(web_paths.library_root) == []


@pytest.mark.parametrize("faststart", [False, True])
def test_add_rejects_truncated_video(lib, web_paths, video_factory, faststart):
    data = video_factory("t.mp4", seconds=3, faststart=faststart).read_bytes()
    with pytest.raises(InvalidVideo):
        lib.add(io.BytesIO(data[: len(data) // 2]), "t.mp4")
    assert lib.list() == []
    assert files_under(web_paths.library_root) == []


def test_get_rejects_unknown_and_malformed_ids(lib):
    for bad in ("0123456789ab", "../library", "", "ABCDEF012345", "/etc/passwd"):
        with pytest.raises(KeyError):
            lib.get(bad)


def test_import_inside_data_root_references_the_file(lib, web_paths, video_factory):
    root = web_paths.data_roots[0]
    root.mkdir(parents=True)
    clip = video_factory(root / "sub" / "a.mp4")
    record = lib.add_from_path(clip)
    assert record.original == clip.resolve()
    assert record.name == "a.mp4"
    assert files_under(web_paths.library_root / record.id) != []
    assert not (web_paths.library_root / record.id / "original.mp4").exists()


def test_import_outside_data_roots_rejected(lib, web_paths, tmp_path, video_factory):
    web_paths.data_roots[0].mkdir(parents=True)
    outside = video_factory(tmp_path / "elsewhere" / "x.mp4")
    for candidate in (
        outside,  # absolute, outside
        web_paths.data_roots[0] / ".." / "elsewhere" / "x.mp4",  # ../ escape
        Path("elsewhere/x.mp4"),  # relative: never resolved against a root
    ):
        with pytest.raises(OutsideRoots):
            lib.add_from_path(candidate)
    assert lib.list() == []
    assert files_under(web_paths.library_root) == []


def test_import_without_data_roots_rejected(web_paths, tmp_path, video_factory):
    lib = Library(web_paths.library_root, ())
    try:
        with pytest.raises(OutsideRoots):
            lib.add_from_path(video_factory(tmp_path / "x.mp4"))
    finally:
        lib.close()


def test_import_symlink_escape_rejected(lib, web_paths, tmp_path, video_factory):
    root = web_paths.data_roots[0]
    root.mkdir(parents=True)
    outside = video_factory(tmp_path / "elsewhere" / "x.mp4")
    (root / "link.mp4").symlink_to(outside)
    (root / "dirlink").symlink_to(outside.parent, target_is_directory=True)
    for candidate in (root / "link.mp4", root / "dirlink" / "x.mp4"):
        with pytest.raises(OutsideRoots):
            lib.add_from_path(candidate)
    assert lib.list() == []


def test_import_missing_file_is_not_found(lib, web_paths):
    web_paths.data_roots[0].mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        lib.add_from_path(web_paths.data_roots[0] / "nope.mp4")


def test_h264_with_meta_is_remuxed_with_sensor_timestamps(
    lib, web_paths, video_factory
):
    root = web_paths.data_roots[0]
    root.mkdir(parents=True)
    raw = video_factory(root / "0000.h264", seconds=6 / 10, fmt="h264")
    # 5 timestamps for 6 frames: the frame without one is dropped (remux_vfr.py).
    times_ms = [0, 40, 100, 130, 200]
    write_meta(root / "0000.meta.json", times_ms)
    record = lib.add_from_path(raw)
    assert record.original.suffix == ".mkv"
    assert record.original.is_relative_to(web_paths.library_root.resolve())
    assert record.name == "0000.h264"
    assert packet_times(record.original) == pytest.approx(
        [t / 1000 for t in times_ms], abs=1e-3
    )
    assert 0.2 <= record.duration_s <= 0.4  # last pts + that frame's duration


def test_h264_remux_without_pyav_falls_back_to_ffmpeg(
    lib, web_paths, monkeypatch, video_factory
):
    monkeypatch.setattr(library, "_import_av", lambda: None)
    root = web_paths.data_roots[0]
    root.mkdir(parents=True)
    raw = video_factory(root / "0001.h264", seconds=6 / 10, fmt="h264")
    write_meta(root / "0001.meta.json", [0, 40, 100, 130, 200])
    record = lib.add_from_path(raw)
    times = packet_times(record.original)
    # The CLI cannot set per-packet times: frames kept, mean rate preserved.
    assert len(times) == 5
    assert times[-1] == pytest.approx(0.2, abs=0.01)


def test_h264_without_meta_rejected(lib, web_paths, video_factory):
    root = web_paths.data_roots[0]
    root.mkdir(parents=True)
    raw = video_factory(root / "lonely.h264", seconds=4 / 10, fmt="h264")
    with pytest.raises(InvalidVideo, match="meta.json"):
        lib.add_from_path(raw)
    assert files_under(web_paths.library_root) == []


def test_preview_is_produced_in_background(lib, video_factory):
    with open(video_factory("v.mp4"), "rb") as fh:
        record = lib.add(fh, "v.mp4")
    lib.wait_previews(timeout=60)
    ready = lib.get(record.id)
    assert ready.preview is not None and ready.preview.is_file()
    assert ready.preview_status == "ready"
    assert library.probe(ready.preview).duration_s == pytest.approx(2.0, abs=0.1)


def test_record_lists_preview_none_until_ready(web_paths, video_factory, monkeypatch):
    gate = __import__("threading").Event()
    real = library._make_preview
    monkeypatch.setattr(
        library, "_make_preview", lambda src, dst: (gate.wait(30), real(src, dst))
    )
    lib = Library(web_paths.library_root)
    try:
        with open(video_factory("w.mp4"), "rb") as fh:
            record = lib.add(fh, "w.mp4")
        assert record.preview is None and record.preview_status == "pending"
        assert lib.get(record.id).preview is None
        gate.set()
        lib.wait_previews(timeout=60)
        assert lib.get(record.id).preview is not None
    finally:
        gate.set()
        lib.close()


def test_library_reloads_records_from_disk(web_paths, video_factory):
    lib = Library(web_paths.library_root)
    with open(video_factory("r.mp4"), "rb") as fh:
        record = lib.add(fh, "r.mp4")
    lib.wait_previews(timeout=60)
    lib.close()
    again = Library(web_paths.library_root)
    try:
        assert [r.id for r in again.list()] == [record.id]
        assert again.get(record.id).to_dict()["width"] == 64
    finally:
        again.close()


def test_stale_staging_is_removed_on_start(web_paths):
    stale = web_paths.library_root / ".incoming" / "tmpabc"
    stale.mkdir(parents=True)
    (stale / "upload.mp4").write_bytes(b"x" * 100)
    Library(web_paths.library_root).close()
    assert not stale.exists()
