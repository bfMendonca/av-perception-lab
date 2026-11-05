"""Writes small ROS 2 MCAP files for tests (no ROS, no binary file in git)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from mcap.writer import IndexType
from mcap.writer import Writer as McapWriter
from mcap_ros2.writer import Writer

_HEADER = """\
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
"""

_IMAGE = (
    "std_msgs/Header header\n"
    "uint32 height\nuint32 width\nstring encoding\nuint8 is_bigendian\n"
    "uint32 step\nuint8[] data\n"
    "================================================================================\n"
    "MSG: std_msgs/Header\n" + _HEADER
)
_COMPRESSED = (
    "std_msgs/Header header\nstring format\nuint8[] data\n"
    "================================================================================\n"
    "MSG: std_msgs/Header\n" + _HEADER
)
_CAMERA_INFO = (
    "std_msgs/Header header\n"
    "uint32 height\nuint32 width\nstring distortion_model\nfloat64[] d\n"
    "float64[9] k\nfloat64[9] r\nfloat64[12] p\n"
    "uint32 binning_x\nuint32 binning_y\nsensor_msgs/RegionOfInterest roi\n"
    "================================================================================\n"
    "MSG: std_msgs/Header\n"
    + _HEADER
    + "================================================================================\n"
    "MSG: sensor_msgs/RegionOfInterest\n"
    "uint32 x_offset\nuint32 y_offset\nuint32 height\nuint32 width\nbool do_rectify\n"
)

# (channels per pixel, RGB -> encoded) for the raw encodings.
_RAW = {
    "rgb8": (3, lambda a: a),
    "bgr8": (3, lambda a: a[..., ::-1]),
    "rgba8": (4, lambda a: np.concatenate([a, np.full_like(a[..., :1], 255)], axis=-1)),
    "bgra8": (
        4,
        lambda a: np.concatenate(
            [a[..., ::-1], np.full_like(a[..., :1], 255)], axis=-1
        ),
    ),
    "mono8": (1, lambda a: cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)[..., None]),
}


@dataclass(frozen=True, slots=True, kw_only=True)
class ImageSpec:
    """One image topic: RGB uint8 frames written at `times_ns` (log time)."""

    topic: str
    frames: Sequence[np.ndarray]
    times_ns: Sequence[int]
    encoding: str = "rgb8"
    compressed: str | None = None  # "jpeg" or "png"
    format: str | None = None  # CompressedImage.format (default: `compressed`)
    data_prefix: bytes = b""  # before the compressed bytes (compressedDepth header)
    data_patch: Callable[[bytes], bytes] | None = None  # rewrites the encoded bytes
    header_ns: Sequence[int] | None = None  # None: stamp = time
    step_pad: int = 0  # extra bytes per row


@dataclass(frozen=True, slots=True, kw_only=True)
class CameraInfoSpec:
    topic: str
    width: int
    height: int
    k: Sequence[float]
    d: Sequence[float] = ()
    model: str = "plumb_bob"
    p: Sequence[float] | None = None
    roi: tuple[int, int, int, int] = (0, 0, 0, 0)  # x_offset, y_offset, height, width
    binning: int = 0


def _stamp(ns: int) -> dict:
    return {"sec": ns // 1_000_000_000, "nanosec": ns % 1_000_000_000}


def _new_writer(path: Path, indexed: bool) -> tuple[Writer, object]:
    stream = path.open("wb")
    if indexed:
        return Writer(stream), stream
    writer = Writer.__new__(Writer)
    writer._writer = McapWriter(
        stream,
        index_types=IndexType.NONE,
        use_chunking=False,
        use_statistics=False,
        use_summary_offsets=False,
        repeat_channels=False,
        repeat_schemas=False,
    )
    writer._encoders = {}
    writer._channel_ids = {}
    writer._writer.start(profile="ros2", library="av-perception-lab-tests")
    writer._finished = False
    return writer, stream


def _image_message(spec: ImageSpec, frame: np.ndarray, stamp_ns: int) -> dict:
    header = {"stamp": _stamp(stamp_ns), "frame_id": "camera"}
    if spec.compressed:
        ext = {"jpeg": ".jpg", "png": ".png"}[spec.compressed]
        ok, buf = cv2.imencode(ext, np.ascontiguousarray(frame[..., ::-1]))
        assert ok
        fmt = spec.format if spec.format is not None else spec.compressed
        data = spec.data_prefix + buf.tobytes()
        if spec.data_patch is not None:
            data = spec.data_patch(data)
        return {"header": header, "format": fmt, "data": data}
    if spec.encoding in _RAW:
        channels, convert = _RAW[spec.encoding]
        arr = np.ascontiguousarray(convert(frame), dtype=np.uint8)
        height, width = arr.shape[:2]
        rows = arr.reshape(height, width * channels)
    else:  # any other encoding (16UC1, 32FC1...): the frame's own bytes
        arr = np.ascontiguousarray(frame)
        height, width = arr.shape[:2]
        rows = arr.view(np.uint8).reshape(height, -1)
    if spec.step_pad:
        pad = np.zeros((height, spec.step_pad), np.uint8)
        rows = np.concatenate([rows, pad], axis=1)
    return {
        "header": header,
        "height": height,
        "width": width,
        "encoding": spec.encoding,
        "is_bigendian": 0,
        "step": rows.shape[1],
        "data": rows.tobytes(),
    }


def write_mcap(
    path: Path,
    *,
    images: Sequence[ImageSpec] = (),
    camera_infos: Sequence[CameraInfoSpec] = (),
    indexed: bool = True,
) -> Path:
    """Write an MCAP with the given image and CameraInfo topics.

    CameraInfo messages are written once, at time 0. `indexed=False` writes
    without chunks, indexes or a summary section.
    """
    writer, stream = _new_writer(path, indexed)
    try:
        image_schema = writer.register_msgdef("sensor_msgs/msg/Image", _IMAGE)
        comp_schema = writer.register_msgdef(
            "sensor_msgs/msg/CompressedImage", _COMPRESSED
        )
        info_schema = writer.register_msgdef("sensor_msgs/msg/CameraInfo", _CAMERA_INFO)
        events: list[tuple[int, str, object, dict]] = []
        for info in camera_infos:
            p = list(info.p) if info.p is not None else [
                info.k[0], 0.0, info.k[2], 0.0,
                0.0, info.k[4], info.k[5], 0.0,
                0.0, 0.0, 1.0, 0.0,
            ]  # fmt: skip
            x, y, h, w = info.roi
            msg = {
                "header": {"stamp": _stamp(0), "frame_id": "camera"},
                "height": info.height,
                "width": info.width,
                "distortion_model": info.model,
                "d": [float(v) for v in info.d],
                "k": [float(v) for v in info.k],
                "r": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                "p": [float(v) for v in p],
                "binning_x": info.binning,
                "binning_y": info.binning,
                "roi": {
                    "x_offset": x, "y_offset": y, "height": h, "width": w,
                    "do_rectify": False,
                },
            }  # fmt: skip
            events.append((0, info.topic, info_schema, msg))
        for spec in images:
            stamps = spec.header_ns if spec.header_ns is not None else spec.times_ns
            schema = comp_schema if spec.compressed else image_schema
            for frame, t, s in zip(spec.frames, spec.times_ns, stamps, strict=True):
                events.append((t, spec.topic, schema, _image_message(spec, frame, s)))
        events.sort(key=lambda e: e[0])
        for t, topic, schema, msg in events:
            writer.write_message(topic, schema, msg, log_time=t, publish_time=t)
    finally:
        writer.finish()
        stream.close()
    return path
