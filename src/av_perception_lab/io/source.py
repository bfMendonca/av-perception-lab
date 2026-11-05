"""Frame sources. VideoSource decodes upright RGB frames with ffmpeg pipes
over a time segment, with stride and frame cap; McapSource reads an image
topic of a ROS 2 MCAP with the message times. Both optionally apply a
FrameTransform (e.g. lens undistortion) to every frame they yield."""

import contextlib
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
    RawImage,
    iter_raw_images,
    probe_mcap,
    read_camera_info,
    read_images_at,
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

    def warnings(self) -> list[str]:
        """Problems found while reading the frames (after an iteration), for
        the end of the run; [] if none."""
        return []


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


class _Drops:
    """Messages dropped because their time did not increase."""

    def __init__(self) -> None:
        self.count = 0
        self.regression_at: int | None = None  # first message whose time goes back
        self.after_regression = 0  # dropped from that message on

    def add(self, position: int, backward: bool) -> None:
        """Message `position` (from the start of the topic) dropped; `backward`
        when its time is before the last kept one (not a repeat)."""
        self.count += 1
        if backward and self.regression_at is None:
            self.regression_at = position
        if self.regression_at is not None:
            self.after_regression += 1


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
    Only the selected messages are decoded: the times come from the raw
    message bytes, so dropped, skipped (`start`, `stride`) and unsampled
    messages are read but never go through the image decoder.
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
        self._drops = _Drops()  # dropped by the last iteration (not by sample)

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

    def _messages(self) -> Iterator[RawImage]:
        return iter_raw_images(self.info.path, self.topic.topic.name)

    def _read_head(self) -> tuple[str, float]:
        """Time base and median message rate over the first messages (cached;
        only the stamps are read, no image is decoded)."""
        if self._head is None:
            try:
                with contextlib.closing(self._messages()) as messages:
                    head = [
                        (msg.log_ns, msg.header_ns)
                        for msg in itertools.islice(messages, TIME_BASE_WINDOW)
                    ]
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

    def _timed(self, drops: _Drops) -> Iterator[tuple[int, int, int, RawImage]]:
        """(index, time ns, log-time rank, message) of every message with an
        increasing time, not decoded; the dropped ones are counted in `drops`.
        The rank counts the earlier messages with the same log time."""
        use_header = self.time_base == "header"
        last = prev_log = None
        index = rank = 0
        for position, msg in enumerate(self._messages()):
            rank = rank + 1 if msg.log_ns == prev_log else 0
            prev_log = msg.log_ns
            ns = msg.header_ns if use_header else msg.log_ns
            if last is not None and ns <= last:
                drops.add(position, ns < last)
                continue
            last = ns
            yield index, ns, rank, msg
            index += 1

    def _select(
        self, *, stride: int, max_frames: int, drops: _Drops
    ) -> Iterator[tuple[int, float, int, RawImage]]:
        """(index, seconds from the first selected message, log-time rank,
        message) of the selection; nothing is decoded."""
        start_ns = round(self.cfg.start * 1e9)
        end_ns = round((self.cfg.start + self.cfg.duration) * 1e9)
        first = t0 = None
        selected = kept = 0
        for index, ns, rank, msg in self._timed(drops):
            first = ns if first is None else first
            if ns - first < start_ns:
                continue
            if self.cfg.duration and ns - first >= end_ns:
                return
            t0 = ns if t0 is None else t0
            if selected % stride == 0:
                yield index, (ns - t0) / 1e9, rank, msg
                kept += 1
                if max_frames and kept >= max_frames:
                    return
            selected += 1

    def _decode(self, index: int, msg: RawImage) -> np.ndarray:
        """The message's RGB frame, size-checked and transformed."""
        rgb = msg.decode()
        w, h = self.frame_size
        got_h, got_w = rgb.shape[:2]
        if (got_w, got_h) != (w, h):
            raise ContractError(
                f"{self.input_name} {self.topic.topic.name}: message {index} is "
                f"{got_w}x{got_h}, the first one is {w}x{h}"
            )
        return _apply(self.transform, rgb)

    def __iter__(self) -> Iterator[Frame]:
        self._drops = drops = _Drops()
        try:
            for index, t, _, msg in self._select(
                stride=self.cfg.stride, max_frames=self.cfg.max_frames, drops=drops
            ):
                yield Frame(index, t, self._decode(index, msg))
        except McapError as exc:
            raise ConfigError(f"source.mcap: {exc}") from exc

    def sample(self, n: int) -> list[np.ndarray]:
        # Spread over the start/duration selection (stride and max_frames
        # ignored, like sample_frames): a pass over the stamps finds it, then
        # only the n sampled messages are read again and decoded.
        try:
            window = [
                (index, msg.log_ns, rank)
                for index, _, rank, msg in self._select(
                    stride=1, max_frames=0, drops=_Drops()
                )
            ]
            stride = max(1, len(window) // n)
            picks = window[::stride][:n]
            messages = read_images_at(
                self.info.path,
                self.topic.topic.name,
                [(log_ns, rank) for _, log_ns, rank in picks],
            )
            return [
                self._decode(index, msg)
                for (index, _, _), msg in zip(picks, messages, strict=True)
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

    def warnings(self) -> list[str]:
        """The messages dropped by the last iteration (time not increasing)."""
        drops, total = self._drops, self.topic.topic.count
        if not drops.count:
            return []
        base = "header.stamp" if self.time_base == "header" else "MCAP log time"
        text = (
            f"source.mcap: {drops.count} of the {total} messages of "
            f"{self.topic.topic.name} ({100 * drops.count / max(total, 1):.1f} %) "
            f"were dropped because their time ({base}) did not increase"
        )
        if drops.regression_at is not None:
            text += (
                f"; the time goes back at message {drops.regression_at} and "
                f"{drops.after_regression} of the drops follow it (a stamp reset, "
                "e.g. restarted sim time or concatenated recordings?)"
            )
        return [text]

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
            "duplicates_dropped": self._drops.count,
            "stamp_regression_at": self._drops.regression_at,
            "dropped_after_regression": self._drops.after_regression,
            "camera_info_topic": t.camera_info_topic,
            "transform": self.transform.describe() if self.transform else None,
        }
