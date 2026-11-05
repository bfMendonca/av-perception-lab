"""Frame sources. VideoSource decodes upright RGB frames with ffmpeg pipes
over a time segment, with stride and frame cap; McapSource reads an image
topic of a ROS 2 MCAP with the message times. Both optionally apply a
FrameTransform (e.g. lens undistortion) to every frame they yield."""

import copy
import itertools
import statistics
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from av_perception_lab.config import SourceConfig
from av_perception_lab.core.options import ConfigError
from av_perception_lab.core.types import ContractError, Frame
from av_perception_lab.io.mcap_reader import (
    CameraInfoMsg,
    ImageTopic,
    McapError,
    McapInfo,
    iter_images,
    probe_mcap,
    read_camera_info,
)
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


def _apply(transform: FrameTransform | None, rgb: np.ndarray) -> np.ndarray:
    """`transform(rgb)`, checked to keep the shape and dtype."""
    if transform is None:
        return rgb
    out = transform(rgb)
    if out.shape != rgb.shape or out.dtype != rgb.dtype:
        raise ContractError(
            f"{type(transform).__name__} returned {out.shape} {out.dtype}, "
            f"expected {rgb.shape} {rgb.dtype}"
        )
    return out


class FrameSource(ABC):
    """Where frames come from."""

    # Frame.t are the real capture times (variable rate), not index / fps.
    frame_times_known: bool = False

    @property
    @abstractmethod
    def frame_size(self) -> tuple[int, int]:
        """(W, H) of every frame."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short name, part of the run folder name."""

    @property
    @abstractmethod
    def input_name(self) -> str:
        """The input file, for messages."""

    @property
    @abstractmethod
    def nominal_fps(self) -> float:
        """Frame rate of the input (informational for variable-rate inputs)."""

    @property
    @abstractmethod
    def n_frames_total(self) -> int | None:
        """Frames (messages) in the whole input; None if unknown."""

    @abstractmethod
    def __iter__(self) -> Iterator[Frame]: ...

    @abstractmethod
    def sample(self, n: int) -> list[np.ndarray]:
        """Up to `n` RGB frames spread over the selection (for focal estimation)."""

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """JSON-ready description for metrics.json."""

    @abstractmethod
    def with_transform(self, transform: FrameTransform) -> "FrameSource":
        """The same source with `transform` applied to every frame (no new probe)."""

    def camera_metadata(self) -> Mapping[str, Any]:
        """Recording camera description (io.quicktime.camera_metadata keys); {} if unknown."""
        return {}

    def audio_from(self) -> Path | None:
        """File whose audio the rendered video may copy (timing unchanged); None if none."""
        return None

    def camera_info(self) -> CameraInfoMsg | None:
        """The input's ROS `CameraInfo` for the frames; None if there is none."""
        return None


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
        return _apply(self.transform, rgb)

    @property
    def frame_size(self) -> tuple[int, int]:
        return self.info.width, self.info.height

    @property
    def name(self) -> str:
        return self.info.path.stem

    @property
    def input_name(self) -> str:
        return str(self.info.path)

    @property
    def nominal_fps(self) -> float:
        return self.info.fps

    @property
    def n_frames_total(self) -> int | None:
        return self.info.n_frames or None

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

    def audio_from(self) -> Path | None:
        return self.info.path if self.is_whole_video and self.info.has_audio else None

    def describe(self) -> dict[str, Any]:
        transform = self.transform.describe() if self.transform is not None else None
        return {
            **asdict(self.info),
            "path": str(self.info.path),
            "transform": transform,
        }


# Messages inspected to choose the time base (and the nominal rate).
TIME_BASE_WINDOW = 50


class McapSource(FrameSource):
    """An image topic of a ROS 2 MCAP (probed at construction), optionally
    transformed.

    `Frame.t` is seconds from the first selected message, on the time base
    chosen from the first TIME_BASE_WINDOW messages: `header.stamp` when it is
    set (> 0) and non-decreasing there, else the MCAP log time. Messages whose
    time does not increase are dropped (counted in `describe()`).
    `start`/`duration` select by message time from the first message,
    `stride` counts messages and `Frame.index` is the message index (after
    dropping) from the start of the topic, like the frame index of a video.
    """

    frame_times_known = True

    def __init__(
        self, cfg: SourceConfig, *, transform: FrameTransform | None = None
    ) -> None:
        if cfg.mcap is None:
            raise ConfigError("source.mcap: no input MCAP")
        if not cfg.topic:
            raise ConfigError("source.topic: required with source.mcap")
        if not cfg.mcap.is_file():
            raise ConfigError(f"source.mcap: MCAP not found: {cfg.mcap}")
        try:
            info: McapInfo = probe_mcap(cfg.mcap)
        except McapError as exc:
            raise ConfigError(f"source.mcap: {exc}") from exc
        topic = next((i for i in info.images if i.topic.name == cfg.topic), None)
        if topic is None:
            available = ", ".join(i.topic.name for i in info.images) or "none"
            raise ConfigError(
                f"source.topic: {cfg.topic!r} is not an image topic of {cfg.mcap} "
                f"(image topics: {available})"
            )
        if not topic.supported:
            raise ConfigError(
                f"source.topic: {cfg.topic!r} cannot be read: "
                f"{topic.unsupported_reason}"
            )
        self.cfg = cfg
        self.info = info
        self.topic: ImageTopic = topic
        self.transform = transform
        self._head: tuple[str, float] | None = None  # (time base, nominal fps)
        self._duplicates = 0  # dropped by the last iteration (not by sample)

    def with_transform(self, transform: FrameTransform) -> "McapSource":
        """The same source with `transform` applied to every frame (no new probe)."""
        other = copy.copy(self)
        other.transform = transform
        return other

    @property
    def frame_size(self) -> tuple[int, int]:
        return self.topic.width, self.topic.height

    @property
    def name(self) -> str:
        return self.info.path.stem + self.topic.topic.name.replace("/", "_")

    @property
    def input_name(self) -> str:
        return str(self.info.path)

    @property
    def n_frames_total(self) -> int | None:
        return self.topic.topic.count

    def output_fps(self) -> str:
        """Nominal rate ("num/den") of the processed messages (with stride)."""
        fps = Fraction(self.nominal_fps / self.cfg.stride).limit_denominator(1001)
        return f"{fps.numerator}/{fps.denominator}"

    def _messages(self):
        return iter_images(self.info.path, self.topic.topic.name)

    def _read_head(self) -> tuple[str, float]:
        """Time base and median message rate over the first messages (cached)."""
        if self._head is None:
            head = []
            try:
                for msg in self._messages():
                    head.append((msg.log_ns, msg.header_ns))
                    if len(head) >= TIME_BASE_WINDOW:
                        break
            except McapError as exc:
                raise ConfigError(f"source.mcap: {exc}") from exc
            stamps = [h for _, h in head]
            ok = all(h > 0 for h in stamps) and all(
                b >= a for a, b in itertools.pairwise(stamps)
            )
            base = "header" if ok else "log"
            times = stamps if ok else [log for log, _ in head]
            steps = [b - a for a, b in itertools.pairwise(times) if b > a]
            fps = 1e9 / statistics.median(steps) if steps else self.topic.topic.rate_hz
            self._head = (base, fps)
        return self._head

    @property
    def time_base(self) -> str:
        """ "header" (header.stamp) or "log" (MCAP log time)."""
        return self._read_head()[0]

    @property
    def nominal_fps(self) -> float:
        """Median message rate over the first messages."""
        return self._read_head()[1]

    def _timed(self, *, count: bool) -> Iterator[tuple[int, int, np.ndarray]]:
        """(index, time ns, rgb) of every message with an increasing time;
        size-checked. With `count`, the dropped messages go to self._duplicates."""
        use_header = self.time_base == "header"
        w, h = self.frame_size
        if count:
            self._duplicates = 0
        last = None
        index = 0
        for msg in self._messages():
            ns = msg.header_ns if use_header else msg.log_ns
            if last is not None and ns <= last:
                if count:
                    self._duplicates += 1
                continue
            last = ns
            got_h, got_w = msg.rgb.shape[:2]
            if (got_w, got_h) != (w, h):
                raise ContractError(
                    f"{self.input_name} {self.topic.topic.name}: message {index} is "
                    f"{got_w}x{got_h}, the first one is {w}x{h}"
                )
            yield index, ns, msg.rgb
            index += 1

    def _select(
        self, *, stride: int, max_frames: int, count: bool = False
    ) -> Iterator[tuple[int, float, np.ndarray]]:
        """(index, seconds from the first selected message, rgb) of the selection."""
        start_ns = round(self.cfg.start * 1e9)
        end_ns = round((self.cfg.start + self.cfg.duration) * 1e9)
        first = t0 = None
        selected = kept = 0
        for index, ns, rgb in self._timed(count=count):
            first = ns if first is None else first
            if ns - first < start_ns:
                continue
            if self.cfg.duration and ns - first >= end_ns:
                return
            t0 = ns if t0 is None else t0
            if selected % stride == 0:
                yield index, (ns - t0) / 1e9, rgb
                kept += 1
                if max_frames and kept >= max_frames:
                    return
            selected += 1

    def __iter__(self) -> Iterator[Frame]:
        try:
            for index, t, rgb in self._select(
                stride=self.cfg.stride, max_frames=self.cfg.max_frames, count=True
            ):
                yield Frame(index, t, _apply(self.transform, rgb))
        except McapError as exc:
            raise ConfigError(f"source.mcap: {exc}") from exc

    def sample(self, n: int) -> list[np.ndarray]:
        # Like sample_frames: the selection length is estimated from the
        # probed count and rate, so no extra pass over the file.
        rate, total = self.topic.topic.rate_hz, self.topic.topic.count
        if self.cfg.duration:
            count = int(self.cfg.duration * rate)
        else:
            count = max(total - round(self.cfg.start * rate), 0)
        stride = max(1, count // n) if count else 1
        try:
            return [
                _apply(self.transform, rgb)
                for _, _, rgb in self._select(stride=stride, max_frames=n)
            ]
        except McapError as exc:
            raise ConfigError(f"source.mcap: {exc}") from exc

    def camera_info(self) -> CameraInfoMsg | None:
        """First message of the paired `CameraInfo` topic; None if not paired."""
        if self.topic.camera_info_topic is None:
            return None
        try:
            return read_camera_info(self.info.path, self.topic.camera_info_topic)
        except McapError as exc:
            raise ConfigError(f"source.mcap: {exc}") from exc

    def describe(self) -> dict[str, Any]:
        t = self.topic
        return {
            "path": str(self.info.path),
            "topic": t.topic.name,
            "schema": t.topic.schema,
            "encoding": t.encoding,
            "width": t.width,
            "height": t.height,
            "count": t.topic.count,
            "rate": t.topic.rate_hz,
            "nominal_fps": self.nominal_fps,
            "time_base": self.time_base,
            "duplicates_dropped": self._duplicates,
            "camera_info_topic": t.camera_info_topic,
            "transform": self.transform.describe() if self.transform else None,
        }
