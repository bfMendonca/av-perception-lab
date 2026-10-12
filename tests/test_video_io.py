import shutil
from pathlib import Path

import numpy as np
import pytest

from det_depth_distance.video_io import (
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
