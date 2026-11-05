import shutil
from pathlib import Path

import numpy as np
import pytest

from av_perception_lab.io.video import (
    TimestampedVideoWriter,
    VideoWriter,
    _parse_probe,
    output_fps,
    probe,
    read_frames,
    sample_frames,
)

PAYLOAD = {
    "streams": [
        {
            "codec_type": "video",
            "width": 1920,
            "height": 1080,
            "r_frame_rate": "30000/1001",
            "nb_frames": "1860",
            "side_data_list": [{"rotation": -90}],
        },
        {"codec_type": "audio", "r_frame_rate": "0/0"},
        {"codec_type": "data", "r_frame_rate": "0/0"},
    ]
}


def test_parse_probe_rotation_swaps_size():
    info = _parse_probe(Path("x.mov"), PAYLOAD)
    assert (info.width, info.height, info.rotation) == (1080, 1920, -90)
    assert info.n_frames == 1860 and info.has_audio
    assert info.fps == pytest.approx(29.97, abs=0.01)


def test_parse_probe_missing_frames_and_no_video():
    payload = {
        "streams": [
            {
                "codec_type": "video",
                "width": 64,
                "height": 48,
                "r_frame_rate": "25/1",
                "nb_frames": "N/A",
            }
        ]
    }
    info = _parse_probe(Path("x.mp4"), payload)
    assert info.n_frames == 0 and not info.has_audio and info.rotation == 0
    with pytest.raises(ValueError):
        _parse_probe(Path("x.mp4"), {"streams": [{"codec_type": "audio"}]})


def test_output_fps():
    info = _parse_probe(Path("x.mov"), PAYLOAD)
    assert output_fps(info, 1) == "30000/1001"
    assert output_fps(info, 2) == "15000/1001"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg/ffprobe")
def test_writer_reader_round_trip(tmp_path):
    path = tmp_path / "clip.mp4"
    with VideoWriter(path, 64, 48, "10/1", crf=0) as writer:
        for i in range(10):
            writer.write(np.full((48, 64, 3), i * 20, dtype=np.uint8))
    info = probe(path)
    assert (info.width, info.height, info.n_frames) == (64, 48, 10)

    frames = list(read_frames(info, stride=3, max_frames=3))
    assert [f.index for f in frames] == [0, 3, 6]
    assert frames[1].t == pytest.approx(0.3)
    assert frames[1].rgb.shape == (48, 64, 3) and frames[1].rgb.flags.writeable
    assert abs(int(frames[1].rgb.mean()) - 60) <= 3
    assert len(sample_frames(info, 4)) == 4
    assert list(read_frames(info, start=50.0)) == []  # segment past the end


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_read_frames_reports_decoder_failure(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video" * 100)
    info = _parse_probe(bad, PAYLOAD)
    with pytest.raises(RuntimeError, match="decoder failed"):
        list(read_frames(info))


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_writer_rejects_wrong_size(tmp_path):
    with VideoWriter(tmp_path / "x.mp4", 64, 48, "10/1") as writer:
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
        with pytest.raises(ValueError):
            writer.write(np.zeros((10, 10, 3), dtype=np.uint8))


def _read_times(path: Path) -> tuple[list[float], list[float]]:
    """(packet PTS, decoded frame PTS) of the first video stream, in seconds."""
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        packets = sorted(
            float(p.pts * p.time_base)
            for p in container.demux(stream)
            if p.pts is not None
        )
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        frames = [float(f.pts * f.time_base) for f in container.decode(stream)]
    return packets, frames


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="needs ffprobe")
def test_vfr_video_keeps_timestamps(tmp_path):
    path = tmp_path / "vfr.mp4"
    times = [0.0, 0.10, 0.35]
    with TimestampedVideoWriter(path, 64, 48) as writer:
        for i, t in enumerate(times):
            writer.write(np.full((48, 64, 3), i * 60, dtype=np.uint8), t)
    packets, frames = _read_times(path)
    assert packets == pytest.approx(times, abs=1e-3)
    assert frames == pytest.approx(times, abs=1e-3)
    assert frames[0] == 0.0
    info = probe(path)
    assert (info.width, info.height, info.n_frames) == (64, 48, 3)
    # moov before mdat (+faststart), like VideoWriter
    head = path.read_bytes()
    assert head.index(b"moov") < head.index(b"mdat")


def test_vfr_video_irregular_and_offset_times(tmp_path):
    # Times need not start at 0 and gaps may be uneven (dropped messages).
    path = tmp_path / "vfr.mp4"
    times = [0.5, 0.51, 0.9, 1.7, 1.71, 3.0]
    with TimestampedVideoWriter(path, 32, 32) as writer:
        for t in times:
            writer.write(np.zeros((32, 32, 3), np.uint8), t)
    packets, frames = _read_times(path)
    assert frames == pytest.approx(times, abs=1e-3)
    assert packets == pytest.approx(times, abs=1e-3)


def test_vfr_video_single_frame(tmp_path):
    path = tmp_path / "one.mp4"
    with TimestampedVideoWriter(path, 32, 16) as writer:
        writer.write(np.zeros((16, 32, 3), np.uint8), 0.0)
    assert _read_times(path)[1] == [0.0]


def test_vfr_writer_rejects_wrong_size(tmp_path):
    with TimestampedVideoWriter(tmp_path / "x.mp4", 64, 48) as writer:
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8), 0.0)
        with pytest.raises(ValueError, match="frame shape"):
            writer.write(np.zeros((10, 10, 3), dtype=np.uint8), 0.1)


@pytest.mark.parametrize("t", [0.1, 0.05, 0.1 + 1e-7, float("nan"), float("inf")])
def test_vfr_writer_rejects_non_increasing_times(tmp_path, t):
    # 0.1 + 1e-7 s rounds to the same 90 kHz tick as 0.1 s.
    with TimestampedVideoWriter(tmp_path / "x.mp4", 32, 32) as writer:
        writer.write(np.zeros((32, 32, 3), np.uint8), 0.1)
        with pytest.raises(ValueError, match="t ="):
            writer.write(np.zeros((32, 32, 3), np.uint8), t)


@pytest.mark.parametrize("size", [(33, 32), (32, 33)])
def test_vfr_writer_rejects_odd_size(tmp_path, size):
    # yuv420p needs even sizes; fail clearly instead of cropping.
    with pytest.raises(ValueError, match="even"):
        TimestampedVideoWriter(tmp_path / "x.mp4", *size)


def test_vfr_writer_close_is_idempotent_and_closes_on_error(tmp_path):
    path = tmp_path / "x.mp4"
    writer = TimestampedVideoWriter(path, 32, 32)
    with pytest.raises(RuntimeError, match="boom"), writer:
        writer.write(np.zeros((32, 32, 3), np.uint8), 0.0)
        writer.write(np.zeros((32, 32, 3), np.uint8), 0.2)
        raise RuntimeError("boom")
    writer.close()  # second close: no-op
    assert _read_times(path)[1] == pytest.approx([0.0, 0.2], abs=1e-3)
    with pytest.raises(ValueError, match="closed"):
        writer.write(np.zeros((32, 32, 3), np.uint8), 0.3)


def test_vfr_writer_without_frames(tmp_path):
    # Closing an unused writer must not raise (e.g. a run with no frames).
    TimestampedVideoWriter(tmp_path / "x.mp4", 32, 32).close()
