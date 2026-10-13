import shutil

import numpy as np
import pytest

from det_depth_distance.config import SourceConfig
from det_depth_distance.core.options import ConfigError
from det_depth_distance.core.types import Frame
from det_depth_distance.io.source import VideoSource
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
