"""ROS 2 MCAP reader: probes image topics, pairs them with `CameraInfo`, and
decodes `Image` / `CompressedImage` messages to upright RGB uint8 frames.

The MCAP libraries are imported only inside the functions that use them, so
the module imports in the light environment. Every library failure (missing
or corrupt file, truncated recording, bad message) surfaces as `McapError`.
"""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

IMAGE_SCHEMA = "sensor_msgs/msg/Image"
COMPRESSED_SCHEMA = "sensor_msgs/msg/CompressedImage"
CAMERA_INFO_SCHEMA = "sensor_msgs/msg/CameraInfo"

# encoding -> (channels per pixel, OpenCV conversion to RGB or None)
_RAW_ENCODINGS: dict[str, tuple[int, int | None]] = {
    "rgb8": (3, None),
    "bgr8": (3, cv2.COLOR_BGR2RGB),
    "rgba8": (4, cv2.COLOR_RGBA2RGB),
    "bgra8": (4, cv2.COLOR_BGRA2RGB),
    "mono8": (1, cv2.COLOR_GRAY2RGB),
}


class McapError(ValueError):
    """The MCAP file or one of its messages cannot be read."""


@dataclass(frozen=True, slots=True)
class TopicInfo:
    name: str
    schema: str
    count: int
    first_ns: int  # MCAP log time of the first / last message
    last_ns: int
    rate_hz: float  # (count - 1) / (last - first); 0.0 with fewer than 2


@dataclass(frozen=True, slots=True)
class ImageTopic:
    """An image topic; size and encoding come from its first message."""

    topic: TopicInfo
    width: int
    height: int
    encoding: str  # raw encoding, or the compressed format ("jpeg", "png")
    camera_info_topic: str | None


@dataclass(frozen=True, slots=True)
class McapInfo:
    path: Path
    duration_s: float  # first to last message over the image topics
    images: tuple[ImageTopic, ...]
    camera_info_topics: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CameraInfoMsg:
    """Plain copy of a `sensor_msgs/msg/CameraInfo`."""

    width: int
    height: int
    k: tuple[float, ...]
    d: tuple[float, ...]
    model: str
    p: tuple[float, ...]
    roi: tuple[int, int, int, int]  # x_offset, y_offset, height, width
    binning: int  # binning_x (0 and 1 both mean none)


@dataclass(frozen=True, slots=True)
class ImageMessage:
    log_ns: int  # MCAP log time
    header_ns: int  # header.stamp
    rgb: np.ndarray  # (H, W, 3) uint8


@contextmanager
def _guard(path: Path, what: str = "read"):
    """Turn any library failure into McapError."""
    try:
        yield
    except McapError:
        raise
    except Exception as exc:
        raise McapError(f"cannot {what} MCAP {path}: {exc}") from exc


@contextmanager
def _stream(path: Path):
    """The file opened for reading, library failures turned into McapError."""
    with _guard(path), Path(path).open("rb") as stream:
        yield stream


@contextmanager
def _open(path: Path):
    from mcap.reader import make_reader

    with _stream(path) as stream:
        yield make_reader(stream)


def _stamp_ns(stamp: Any) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _messages(path: Path, topics: Sequence[str] | None = None) -> Iterator[Any]:
    """Yield (schema, channel, message, decoded) in log-time order."""
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory

    with _stream(path) as stream:
        reader = make_reader(stream, decoder_factories=[DecoderFactory()])
        yield from reader.iter_decoded_messages(topics=topics)


def _summary_counts(path: Path) -> dict[str, tuple[str, int]] | None:
    """topic -> (schema, count) from the statistics record, None if absent."""
    with _open(path) as reader:
        summary = reader.get_summary()
        if summary is None or summary.statistics is None:
            return None
        counts = summary.statistics.channel_message_counts
        out: dict[str, tuple[str, int]] = {}
        for cid, channel in summary.channels.items():
            schema = summary.schemas.get(channel.schema_id)
            out[channel.topic] = (
                schema.name if schema else "",
                int(counts.get(cid, 0)),
            )
        return out


def _scan(path: Path) -> dict[str, list]:
    """One pass over the file: topic -> [schema, count, first_ns, last_ns]."""
    from mcap.reader import make_reader

    found: dict[str, list] = {}
    with _stream(path) as stream:
        reader = make_reader(stream)
        for schema, channel, message in reader.iter_messages(log_time_order=False):
            row = found.setdefault(
                channel.topic,
                [
                    schema.name if schema else "",
                    0,
                    message.log_time,
                    message.log_time,
                ],
            )
            row[1] += 1
            row[2] = min(row[2], message.log_time)
            row[3] = max(row[3], message.log_time)
    return found


def _topic_times(path: Path, topic: str) -> tuple[int, int]:
    """First and last log time of a topic, via the index when there is one."""
    with _open(path) as reader:
        first = next(reader.iter_messages(topics=[topic]), None)
        last = next(reader.iter_messages(topics=[topic], reverse=True), None)
    if first is None or last is None:
        raise McapError(f"topic {topic} has no messages in {path}")
    return first[2].log_time, last[2].log_time


def _rate(count: int, first_ns: int, last_ns: int) -> float:
    if count < 2 or last_ns <= first_ns:
        return 0.0
    return (count - 1) / ((last_ns - first_ns) / 1e9)


def compressed_codec(fmt: str) -> str:
    """Codec of a `CompressedImage.format`: plain (`jpeg`, `png`) or the
    image_transport form (`rgb8; jpeg compressed bgr8`, whose first part is
    the original encoding, not the codec)."""
    text = fmt.strip().lower()
    for codec, names in (("jpeg", ("jpeg", "jpg")), ("png", ("png",))):
        if any(name in text for name in names):
            return codec
    return text.split(";")[0].strip()


def _first_decoded(path: Path, topic: str) -> tuple[str, Any]:
    for schema, _channel, _message, decoded in _messages(path, [topic]):
        return schema.name, decoded
    raise McapError(f"topic {topic} has no messages in {path}")


def probe_mcap(path: Path) -> McapInfo:
    """Image topics (with size, encoding, count, rate and paired `CameraInfo`)
    and `CameraInfo` topics of an MCAP. Counts come from the summary section
    when there is one, otherwise from a single scan."""
    path = Path(path)
    wanted = {IMAGE_SCHEMA, COMPRESSED_SCHEMA, CAMERA_INFO_SCHEMA}
    with _guard(path):
        counts = _summary_counts(path)
        if counts is not None:
            topics = {
                name: (schema, count)
                for name, (schema, count) in counts.items()
                if schema in wanted and count > 0
            }
            times = {name: _topic_times(path, name) for name in topics}
            rows = {n: [s, c, *times[n]] for n, (s, c) in topics.items()}
        else:
            rows = {n: r for n, r in _scan(path).items() if r[0] in wanted}

        info_topics = tuple(
            sorted(n for n, r in rows.items() if r[0] == CAMERA_INFO_SCHEMA)
        )
        images = []
        for name in sorted(n for n, r in rows.items() if r[0] != CAMERA_INFO_SCHEMA):
            schema, count, first_ns, last_ns = rows[name]
            _, msg = _first_decoded(path, name)
            if schema == COMPRESSED_SCHEMA:
                frame = _decode_compressed(msg)
                height, width = frame.shape[:2]
                encoding = compressed_codec(str(msg.format))
            else:
                width, height, encoding = (
                    int(msg.width),
                    int(msg.height),
                    str(msg.encoding),
                )
            images.append(
                ImageTopic(
                    topic=TopicInfo(
                        name,
                        schema,
                        count,
                        first_ns,
                        last_ns,
                        _rate(count, first_ns, last_ns),
                    ),
                    width=width,
                    height=height,
                    encoding=encoding,
                    camera_info_topic=pair_camera_info(name, info_topics),
                )
            )
    if images:
        start = min(i.topic.first_ns for i in images)
        end = max(i.topic.last_ns for i in images)
        duration = (end - start) / 1e9
    else:
        duration = 0.0
    return McapInfo(path, duration, tuple(images), info_topics)


def _namespace(topic: str) -> list[str]:
    parts = [p for p in topic.split("/") if p]
    if parts and parts[-1] == "compressed":
        parts.pop()
    return parts[:-1]  # drop the leaf (image_raw / camera_info)


def pair_camera_info(image_topic: str, camera_info_topics: Sequence[str]) -> str | None:
    """The `CameraInfo` topic sharing the longest namespace prefix with the
    image topic (`/cam/image_raw/compressed` lives in `/cam`), None when no
    candidate shares a namespace component (root-level topics pair together)."""
    image_ns = _namespace(image_topic)
    best: tuple[int, int] | None = None
    best_topic = None
    for candidate in camera_info_topics:
        ns = _namespace(candidate)
        common = 0
        for a, b in zip(image_ns, ns, strict=False):
            if a != b:
                break
            common += 1
        if common == 0 and (image_ns or ns):
            continue
        score = (common, -(len(image_ns) - common) - (len(ns) - common))
        if best is None or score > best:
            best, best_topic = score, candidate
    return best_topic


def read_camera_info(path: Path, topic: str) -> CameraInfoMsg:
    """First `CameraInfo` message of the topic."""
    path = Path(path)
    with _guard(path):
        schema, msg = _first_decoded(path, topic)
        if schema != CAMERA_INFO_SCHEMA:
            raise McapError(f"topic {topic} is {schema}, not a CameraInfo")
        roi = msg.roi
        return CameraInfoMsg(
            width=int(msg.width),
            height=int(msg.height),
            k=tuple(float(v) for v in msg.k),
            d=tuple(float(v) for v in msg.d),
            model=str(msg.distortion_model),
            p=tuple(float(v) for v in msg.p),
            roi=(
                int(roi.x_offset),
                int(roi.y_offset),
                int(roi.height),
                int(roi.width),
            ),
            binning=int(msg.binning_x),
        )


def _decode_compressed(msg: Any) -> np.ndarray:
    buf = np.frombuffer(bytes(msg.data), np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise McapError(f"cannot decode compressed image (format {msg.format!r})")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def decode_image(msg: Any, schema: str) -> np.ndarray:
    """A decoded `Image` / `CompressedImage` message as (H, W, 3) uint8 RGB.
    Raw rows are read with `step`, so row padding does not shear the frame."""
    if schema == COMPRESSED_SCHEMA:
        return _decode_compressed(msg)
    if schema != IMAGE_SCHEMA:
        raise McapError(f"unsupported image schema {schema}")
    encoding = str(msg.encoding)
    if encoding not in _RAW_ENCODINGS:
        raise McapError(
            f"unsupported image encoding {encoding!r}; "
            f"supported: {', '.join(_RAW_ENCODINGS)}, jpeg, png"
        )
    channels, convert = _RAW_ENCODINGS[encoding]
    height, width, step = int(msg.height), int(msg.width), int(msg.step)
    data = np.frombuffer(bytes(msg.data), np.uint8)
    if step < width * channels or data.size < step * height:
        raise McapError(
            f"image data too short: {data.size} bytes for {height}x{width} "
            f"{encoding} with step {step}"
        )
    rows = data[: step * height].reshape(height, step)[:, : width * channels]
    arr = np.ascontiguousarray(rows.reshape(height, width, channels))
    return arr if convert is None else cv2.cvtColor(arr, convert)


def iter_images(path: Path, topic: str) -> Iterator[ImageMessage]:
    """Decode every message of an image topic, in log-time order."""
    path = Path(path)
    seen = False
    for schema, _channel, message, decoded in _messages(path, [topic]):
        seen = True
        with _guard(path, "decode a message of"):
            rgb = decode_image(decoded, schema.name)
            header_ns = _stamp_ns(decoded.header.stamp)
        yield ImageMessage(message.log_time, header_ns, rgb)
    if not seen:
        raise McapError(f"topic {topic} has no messages in {path}")
