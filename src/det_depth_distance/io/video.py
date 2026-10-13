"""ffprobe / ffmpeg pipe helpers (system binaries, no OpenCV video I/O).

Decoding yields upright RGB frames (ffmpeg auto-rotates, so `VideoInfo`
reports post-rotation size) from the first video stream only; iPhone MOVs
carry several data streams. Encoding is libx264 yuv420p, copying the source
audio only when the whole video is processed unchanged.
"""

import json
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Self

import numpy as np

from det_depth_distance.core.types import Frame


@dataclass(frozen=True, slots=True)
class VideoInfo:
    """Metadata of the first video stream."""

    path: Path
    width: int  # after rotation
    height: int
    fps: float
    fps_str: str  # r_frame_rate, e.g. "30/1"
    n_frames: int  # 0 = unknown
    has_audio: bool
    rotation: int  # degrees


def _parse_probe(path: Path, payload: dict[str, Any]) -> VideoInfo:
    streams = payload.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise ValueError(f"no video stream in {path}")
    rotation = 0
    for side_data in video.get("side_data_list", []):
        if "rotation" in side_data:
            rotation = int(float(side_data["rotation"]))
    if not rotation and "rotate" in video.get("tags", {}):
        rotation = int(float(video["tags"]["rotate"]))
    width, height = int(video["width"]), int(video["height"])
    if abs(rotation) % 180 == 90:
        width, height = height, width
    nb_frames = str(video.get("nb_frames", ""))
    return VideoInfo(
        path=Path(path),
        width=width,
        height=height,
        fps=float(Fraction(video["r_frame_rate"])),
        fps_str=video["r_frame_rate"],
        n_frames=int(nb_frames) if nb_frames.isdigit() else 0,
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
        rotation=rotation,
    )


def probe(path: Path) -> VideoInfo:
    """Read video metadata with ffprobe."""
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            (
                "stream=codec_type,width,height,r_frame_rate,nb_frames"
                ":stream_side_data=rotation:stream_tags=rotate"
            ),
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return _parse_probe(Path(path), json.loads(out))


def read_frames(
    info: VideoInfo,
    *,
    start: float = 0.0,
    duration: float = 0.0,
    max_frames: int = 0,
    stride: int = 1,
) -> Iterator[Frame]:
    """Decode RGB frames from `start` (s) for `duration` (s, 0 = to the end).

    Keeps every `stride`-th frame and stops after `max_frames` kept frames
    (0 = no cap). Stride is applied after decoding: simple, and exact for any
    stride without ffmpeg select filters. A segment past the end of the video
    yields nothing.

    Raises:
        RuntimeError: the decoder exited with a non-zero code (unreadable input).
    """
    cmd = ["ffmpeg", "-v", "error"]
    if start:
        cmd += ["-ss", f"{start}"]
    if duration:
        cmd += ["-t", f"{duration}"]
    cmd += [
        "-i",
        str(info.path),
        "-map",
        "0:v:0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    frame_bytes = info.width * info.height * 3
    first = round(start * info.fps)
    decoded = kept = 0
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            if decoded % stride == 0:
                # bytearray -> writable array (torch warns on read-only numpy buffers)
                rgb = np.frombuffer(bytearray(buf), np.uint8).reshape(
                    info.height, info.width, 3
                )
                yield Frame(first + decoded, start + decoded / info.fps, rgb)
                kept += 1
                if max_frames and kept >= max_frames:
                    return
            decoded += 1
        # Natural end of the stream: only here is the exit code meaningful (an
        # early stop kills the decoder below). Without this check an unreadable
        # file would look like an empty video.
        if proc.wait() != 0:
            raise RuntimeError(
                f"ffmpeg decoder failed with code {proc.returncode} on {info.path}"
            )
    finally:
        # SIGKILL before closing the pipe: on an early exit (max_frames, sampling)
        # SIGTERM can hang (ffmpeg blocks writing to the full pipe), and closing
        # first makes it print "Broken pipe" muxer errors. It only reads input.
        proc.kill()
        proc.wait()
        proc.stdout.close()


def sample_frames(
    info: VideoInfo, n: int, *, start: float = 0.0, duration: float = 0.0
) -> list[np.ndarray]:
    """Up to `n` frames evenly spread over the selected segment (first `n` if its length is unknown)."""
    if duration:
        count = int(duration * info.fps)
    else:
        count = max(info.n_frames - round(start * info.fps), 0)
    stride = max(1, count // n) if count else 1
    frames = read_frames(
        info, start=start, duration=duration, max_frames=n, stride=stride
    )
    return [frame.rgb for frame in frames]


def output_fps(info: VideoInfo, stride: int) -> str:
    """Exact output frame rate ("num/den") when keeping every `stride`-th frame."""
    fps = Fraction(info.fps_str) / stride
    return f"{fps.numerator}/{fps.denominator}"


class VideoWriter:
    """H.264 encoder fed with RGB frames through an ffmpeg pipe (context manager).

    `audio_from`: copy (re-encode to AAC) the audio of this file; the caller
    decides when that is valid (whole video, no stride).
    """

    def __init__(
        self,
        path: Path,
        width: int,
        height: int,
        fps: str,
        *,
        crf: int = 20,
        audio_from: Path | None = None,
    ) -> None:
        self.width, self.height = width, height
        cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            fps,
            "-i",
            "-",
        ]
        if audio_from is not None:
            cmd += [
                "-i",
                str(audio_from),
                "-map",
                "0:v",
                "-map",
                "1:a",
                "-c:a",
                "aac",
                "-b:a",
                "160k",
                "-shortest",
            ]
        cmd += [
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, rgb: np.ndarray) -> None:
        """Append one (height, width, 3) uint8 frame."""
        if rgb.shape != (self.height, self.width, 3):
            raise ValueError(
                f"frame shape {rgb.shape} != {(self.height, self.width, 3)}"
            )
        self._proc.stdin.write(np.ascontiguousarray(rgb, dtype=np.uint8).tobytes())

    def close(self) -> None:
        """Flush and finalise the file.

        Raises:
            RuntimeError: ffmpeg exited with a non-zero code.
        """
        if self._proc.stdin and not self._proc.stdin.closed:
            self._proc.stdin.close()
        if self._proc.wait() != 0:
            raise RuntimeError(
                f"ffmpeg encoder failed with code {self._proc.returncode}"
            )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
