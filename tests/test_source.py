import shutil

import numpy as np
import pytest

from det_depth_distance.config import SourceConfig
from det_depth_distance.core.options import ConfigError
from det_depth_distance.core.types import ContractError, Frame
from det_depth_distance.io.source import FrameTransform, VideoSource
from det_depth_distance.io.video import VideoWriter


def test_missing_video_is_a_config_error(tmp_path):
    with pytest.raises(ConfigError, match="no input video"):
        VideoSource(SourceConfig())
    with pytest.raises(ConfigError, match="not found"):
        VideoSource(SourceConfig(video=tmp_path / "nope.mov"))


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg/ffprobe")
def test_video_source(tmp_path):
    path = tmp_path / "clip.mp4"
    with VideoWriter(path, 64, 48, "10/1", crf=0) as writer:
        for i in range(10):
            writer.write(np.full((48, 64, 3), i * 20, dtype=np.uint8))
    source = VideoSource(SourceConfig(video=path, stride=2, max_frames=3))
    assert source.frame_size == (64, 48) and source.name == "clip"
    frames = list(source)
    assert all(isinstance(f, Frame) for f in frames)
    assert [f.index for f in frames] == [0, 2, 4]
    assert len(source.sample(4)) == 4
    assert source.output_fps() == "5/1"
    assert not source.is_whole_video
    assert VideoSource(SourceConfig(video=path)).is_whole_video
    assert source.describe()["path"] == str(path)


class Invert(FrameTransform):
    """Spy transform: counts calls, inverts the frame."""

    def __init__(self):
        self.calls = 0

    def __call__(self, rgb):
        self.calls += 1
        return 255 - rgb

    def describe(self):
        return {"type": "invert"}


class Crop(Invert):
    def __call__(self, rgb):
        return rgb[1:]


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "clip.mp4"
    with VideoWriter(path, 64, 48, "10/1", crf=0) as writer:
        for i in range(10):
            writer.write(np.full((48, 64, 3), i * 20, dtype=np.uint8))
    return path


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg/ffprobe")
def test_transform_applies_to_frames_and_samples(clip):
    plain = VideoSource(SourceConfig(video=clip, max_frames=3))
    assert plain.transform is None and plain.describe()["transform"] is None
    assert plain.camera_metadata() == {}  # no QuickTime camera keys, no error
    spy = Invert()
    source = plain.with_transform(spy)
    assert source.info is plain.info and plain.transform is None  # copy, no new probe
    frames, raw = list(source), list(plain)
    assert spy.calls == 3
    assert all(
        np.array_equal(f.rgb, 255 - r.rgb) and f.index == r.index and f.t == r.t
        for f, r in zip(frames, raw, strict=True)
    )
    assert len(source.sample(2)) == 2 and spy.calls == 5
    assert source.describe()["transform"] == {"type": "invert"}
    with pytest.raises(ContractError, match=r"Crop returned \(47, 64, 3\)"):
        list(VideoSource(SourceConfig(video=clip), transform=Crop()))
