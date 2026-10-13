"""Frame sources. VideoSource decodes upright RGB frames with ffmpeg pipes
over a time segment, with stride and frame cap."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import asdict
from typing import Any

import numpy as np

from det_depth_distance.config import SourceConfig
from det_depth_distance.core.options import ConfigError
from det_depth_distance.core.types import Frame
from det_depth_distance.io.video import (
    VideoInfo,
    output_fps,
    probe,
    read_frames,
    sample_frames,
)


class FrameSource(ABC):
    """Where frames come from."""

    @property
    @abstractmethod
    def frame_size(self) -> tuple[int, int]:
        """(W, H) of every frame."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short name, part of the run folder name."""

    @abstractmethod
    def __iter__(self) -> Iterator[Frame]: ...

    @abstractmethod
    def sample(self, n: int) -> list[np.ndarray]:
        """Up to `n` RGB frames spread over the selection (for focal estimation)."""

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """JSON-ready description for metrics.json."""


class VideoSource(FrameSource):
    """A segment of a video file (probed at construction)."""

    def __init__(self, cfg: SourceConfig) -> None:
        if cfg.video is None:
            raise ConfigError("source.video: no input video (YAML or VIDEO argument)")
        if not cfg.video.is_file():
            raise ConfigError(f"source.video: video not found: {cfg.video}")
        self.cfg = cfg
        self.info: VideoInfo = probe(cfg.video)

    @property
    def frame_size(self) -> tuple[int, int]:
        return self.info.width, self.info.height

    @property
    def name(self) -> str:
        return self.info.path.stem

    def __iter__(self) -> Iterator[Frame]:
        return read_frames(
            self.info,
            start=self.cfg.start,
            duration=self.cfg.duration,
            max_frames=self.cfg.max_frames,
            stride=self.cfg.stride,
        )

    def sample(self, n: int) -> list[np.ndarray]:
        return sample_frames(
            self.info, n, start=self.cfg.start, duration=self.cfg.duration
        )

    def output_fps(self) -> str:
        """Exact frame rate ("num/den") of the processed frames."""
        return output_fps(self.info, self.cfg.stride)

    @property
    def is_whole_video(self) -> bool:
        """Every frame, unchanged timing: only then may the source audio be copied."""
        c = self.cfg
        return not (c.start or c.duration or c.max_frames) and c.stride == 1

    def describe(self) -> dict[str, Any]:
        return {**asdict(self.info), "path": str(self.info.path)}
