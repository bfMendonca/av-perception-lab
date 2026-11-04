"""Frame sources. VideoSource decodes upright RGB frames with ffmpeg pipes
over a time segment, with stride and frame cap, and optionally applies a
FrameTransform (e.g. lens undistortion) to every frame it yields."""

import copy
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import asdict
from typing import Any

import numpy as np

from av_perception_lab.config import SourceConfig
from av_perception_lab.core.options import ConfigError
from av_perception_lab.core.types import ContractError, Frame
from av_perception_lab.io.quicktime import camera_metadata
from av_perception_lab.io.video import (
    VideoInfo,
    output_fps,
    probe,
    read_frames,
    sample_frames,
)


class FrameTransform(ABC):
    """Per-frame image transform applied by a source after decoding (e.g. lens
    undistortion). Must keep the frame shape and dtype."""

    @abstractmethod
    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        """(H, W, 3) uint8 -> transformed frame of the same shape and dtype."""

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """JSON-ready description for metrics.json."""


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

    def camera_metadata(self) -> Mapping[str, Any]:
        """Recording camera description (io.quicktime.camera_metadata keys); {} if unknown."""
        return {}


class VideoSource(FrameSource):
    """A segment of a video file (probed at construction), optionally transformed."""

    def __init__(
        self, cfg: SourceConfig, *, transform: FrameTransform | None = None
    ) -> None:
        if cfg.video is None:
            raise ConfigError("source.video: no input video (YAML or VIDEO argument)")
        if not cfg.video.is_file():
            raise ConfigError(f"source.video: video not found: {cfg.video}")
        self.cfg = cfg
        self.info: VideoInfo = probe(cfg.video)
        self.transform = transform
        self._camera: Mapping[str, Any] | None = None  # camera_metadata() cache

    def with_transform(self, transform: FrameTransform) -> "VideoSource":
        """The same source with `transform` applied to every frame (no new probe)."""
        other = copy.copy(self)
        other.transform = transform
        return other

    def _apply(self, rgb: np.ndarray) -> np.ndarray:
        if self.transform is None:
            return rgb
        out = self.transform(rgb)
        if out.shape != rgb.shape or out.dtype != rgb.dtype:
            raise ContractError(
                f"{type(self.transform).__name__} returned {out.shape} {out.dtype}, "
                f"expected {rgb.shape} {rgb.dtype}"
            )
        return out

    @property
    def frame_size(self) -> tuple[int, int]:
        return self.info.width, self.info.height

    @property
    def name(self) -> str:
        return self.info.path.stem

    def __iter__(self) -> Iterator[Frame]:
        frames = read_frames(
            self.info,
            start=self.cfg.start,
            duration=self.cfg.duration,
            max_frames=self.cfg.max_frames,
            stride=self.cfg.stride,
        )
        if self.transform is None:
            return frames
        return (Frame(f.index, f.t, self._apply(f.rgb)) for f in frames)

    def sample(self, n: int) -> list[np.ndarray]:
        # Transformed like the processed frames: focal estimation sees the same pixels.
        frames = sample_frames(
            self.info, n, start=self.cfg.start, duration=self.cfg.duration
        )
        return [self._apply(rgb) for rgb in frames]

    def camera_metadata(self) -> Mapping[str, Any]:
        """QuickTime camera keys of the file (track-level meta included), cached."""
        if self._camera is None:
            self._camera = camera_metadata(self.info.path)
        return self._camera

    def output_fps(self) -> str:
        """Exact frame rate ("num/den") of the processed frames."""
        return output_fps(self.info, self.cfg.stride)

    @property
    def is_whole_video(self) -> bool:
        """Every frame, unchanged timing: only then may the source audio be copied."""
        c = self.cfg
        return not (c.start or c.duration or c.max_frames) and c.stride == 1

    def describe(self) -> dict[str, Any]:
        transform = self.transform.describe() if self.transform is not None else None
        return {
            **asdict(self.info),
            "path": str(self.info.path),
            "transform": transform,
        }
