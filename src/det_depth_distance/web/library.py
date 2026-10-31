"""Video library of the web server: uploads, server-path imports, previews.

One folder per video, `library_root/<id>/`, where `id` is the first 12 hex
digits of the content's sha256 (re-uploads dedupe):

- `record.json`: name, size, probe (width, height, fps, duration), where the
  original is and the state of the preview;
- `original.<ext>` for uploads, or `original.mkv` for a converted `.h264`;
  a server-path import references the source file instead (no copy);
- `thumb.jpg` and, once the background thread has made it, `preview.mp4`
  (H.264 MP4 the browser plays; VFR `.mkv` and HEVC do not play there).

An entry is assembled under `library_root/.incoming/` and renamed into place
only when complete, so a rejected input (not a video, truncated) leaves
nothing behind. Raw Raspberry Pi `.h264` streams need their sibling
`.meta.json`: they are remuxed to a VFR `.mkv` with the per-frame
`SensorTimestamp`s (the algorithm of `data/s000032/remux_vfr.py`), with
PyAV when it is importable, else with the ffmpeg CLI at the mean rate of
those timestamps (an approximation: per-frame jitter is lost).

Server-path imports must resolve (symlinks followed) inside one of the
allowed `data_roots`. Only `ffprobe`/`ffmpeg` (and PyAV) are used: no torch.
"""

import concurrent.futures
import contextlib
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any, BinaryIO

CHUNK = 1 << 20
ID_LEN = 12
_ID = re.compile(rf"^[0-9a-f]{{{ID_LEN}}}$")
INCOMING = ".incoming"
RECORD = "record.json"
THUMB = "thumb.jpg"
PREVIEW = "preview.mp4"
RAW_SUFFIXES = (".h264", ".264")
PREVIEW_MAX_SIDE = 1280
THUMB_WIDTH = 320
_SENSOR_TS = re.compile(r'"SensorTimestamp":\s*(\d+)\s*[,}\n]')  # remux_vfr.py
PROBE_TIMEOUT_S = 60.0
DECODE_TIMEOUT_S = 120.0
REMUX_TIMEOUT_S = 900.0


class InvalidVideo(ValueError):
    """The input is not a readable video (or is truncated)."""


class OutsideRoots(PermissionError):
    """A server path outside the folders videos may be imported from."""


@dataclass(frozen=True, slots=True)
class ProbeInfo:
    """What the library needs of a video: upright size, mean rate, length."""

    width: int
    height: int
    fps: float
    duration_s: float


@dataclass(frozen=True, slots=True)
class VideoRecord:
    """One library entry (`preview` is None until the MP4 is ready)."""

    id: str
    name: str
    width: int
    height: int
    fps: float
    duration_s: float
    size_bytes: int
    original: Path
    preview: Path | None
    thumb: Path
    preview_status: str = "pending"  # pending|ready|failed
    created: str = ""
    origin: str = "upload"  # upload|import

    def to_dict(self) -> dict[str, Any]:
        """JSON view, with the URLs of the file routes."""
        data = asdict(self)
        data.update(
            original=str(self.original),
            preview=str(self.preview) if self.preview else None,
            thumb=str(self.thumb),
            urls={
                k: f"/api/videos/{self.id}/{k}" for k in ("file", "preview", "thumb")
            },
        )
        return data


# -- ffprobe / ffmpeg ----------------------------------------------------------


def _run(cmd: list[str], timeout: float) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(cmd, capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        raise InvalidVideo(f"{cmd[0]} timed out after {timeout:.0f} s") from None
    except OSError as exc:
        raise InvalidVideo(f"cannot run {cmd[0]}: {exc}") from None


def _last_error(proc: subprocess.CompletedProcess[bytes]) -> str:
    """Last stderr line of ffmpeg/ffprobe, without the `[demuxer @ 0x..]`
    prefix and server paths (the message goes back to the client)."""
    lines = proc.stderr.decode(errors="replace").strip().splitlines()
    if not lines:
        return f"exit code {proc.returncode}"
    line = re.sub(r"^(\[[^\]]*\]\s*)+", "", lines[-1])
    return re.sub(r"(?:^|(?<=\s))(?:/[^\s:/]+)+/?:?\s*", "", line).strip() or line


def _rate(text: str | None) -> float:
    try:
        value = float(Fraction(text or "0/0"))
    except (ValueError, ZeroDivisionError):
        return 0.0
    return value if value > 0 else 0.0


def probe(path: Path) -> ProbeInfo:
    """Size (after rotation), mean frame rate and duration of the first video
    stream, checking that its first and last frames decode.

    Raises:
        InvalidVideo: no video stream, no duration, or undecodable/truncated.
    """
    proc = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            (
                "stream=width,height,avg_frame_rate,r_frame_rate,duration"
                ":stream_side_data=rotation:stream_tags=rotate:format=duration"
            ),
            "-of",
            "json",
            str(path),
        ],
        PROBE_TIMEOUT_S,
    )
    if proc.returncode != 0:
        raise InvalidVideo(f"not a readable video: {_last_error(proc)}")
    try:
        payload = json.loads(proc.stdout)
    except ValueError:
        raise InvalidVideo("not a readable video (ffprobe output)") from None
    streams = payload.get("streams") or []
    if not streams:
        raise InvalidVideo("no video stream in the file")
    video = streams[0]
    width, height = int(video.get("width") or 0), int(video.get("height") or 0)
    if width <= 0 or height <= 0:
        raise InvalidVideo("the video stream has no frame size")
    rotation = 0
    for side in video.get("side_data_list") or []:
        if "rotation" in side:
            rotation = int(float(side["rotation"]))
    if not rotation and "rotate" in (video.get("tags") or {}):
        rotation = int(float(video["tags"]["rotate"]))
    if abs(rotation) % 180 == 90:
        width, height = height, width
    fps = _rate(video.get("avg_frame_rate")) or _rate(video.get("r_frame_rate"))
    duration = 0.0
    for text in ((payload.get("format") or {}).get("duration"), video.get("duration")):
        with contextlib.suppress(TypeError, ValueError):
            duration = duration or max(float(text), 0.0)
    if duration <= 0 or fps <= 0:
        raise InvalidVideo("could not read the duration or frame rate of the video")
    info = ProbeInfo(width, height, fps, duration)
    _check_decodes(path, 0.0, "first")
    _check_decodes(path, max(0.0, duration - min(1.0, duration / 2)), "last")
    return info


def _check_decodes(path: Path, at: float, which: str) -> None:
    """One frame at `at` seconds decodes (a truncated file fails near its end)."""
    proc = _run(
        ["ffmpeg", "-v", "error", "-nostdin", "-xerror", "-ss", f"{at:.3f}"]
        + ["-i", str(path), "-map", "0:v:0", "-frames:v", "1"]
        + ["-f", "rawvideo", "-pix_fmt", "gray", "-s", "8x8", "pipe:1"],
        DECODE_TIMEOUT_S,
    )
    if proc.returncode != 0 or len(proc.stdout) != 64:
        raise InvalidVideo(
            f"the {which} frames do not decode (truncated file?): {_last_error(proc)}"
        )


def _thumbnail(src: Path, dst: Path, at: float) -> None:
    proc = _run(
        ["ffmpeg", "-v", "error", "-nostdin", "-y", "-ss", f"{at:.3f}", "-i", str(src)]
        + ["-map", "0:v:0", "-frames:v", "1", "-q:v", "4"]
        + ["-vf", f"scale='min({THUMB_WIDTH},iw)':-2", str(dst)],
        DECODE_TIMEOUT_S,
    )
    if proc.returncode != 0 or not dst.is_file() or dst.stat().st_size == 0:
        raise InvalidVideo(f"could not decode a frame: {_last_error(proc)}")


_PREVIEW_PROCS: set[subprocess.Popen[bytes]] = set()
_PREVIEW_LOCK = threading.Lock()


def _make_preview(src: Path, dst: Path) -> None:
    """Browser-playable H.264 MP4 (longest side <= PREVIEW_MAX_SIDE, source
    timestamps kept so segment times match the original), atomically.

    Raises:
        RuntimeError: ffmpeg failed.
    """
    side = PREVIEW_MAX_SIDE
    scale = (
        f"scale='if(gte(iw,ih),trunc(min({side},iw)/2)*2,-2)'"
        f":'if(gte(iw,ih),-2,trunc(min({side},ih)/2)*2)'"
    )
    tmp = dst.with_name(f".{dst.stem}.tmp.mp4")
    cmd = ["nice", "-n", "10"] if shutil.which("nice") else []
    cmd += ["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(src)]
    cmd += ["-map", "0:v:0", "-an", "-sn", "-dn", "-vf", scale]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-g", "30"]
    cmd += ["-pix_fmt", "yuv420p", "-fps_mode", "passthrough"]
    cmd += ["-movflags", "+faststart", str(tmp)]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    with _PREVIEW_LOCK:
        _PREVIEW_PROCS.add(proc)
    try:
        _, err = proc.communicate()
    finally:
        with _PREVIEW_LOCK:
            _PREVIEW_PROCS.discard(proc)
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        lines = err.decode(errors="replace").strip().splitlines()
        raise RuntimeError(lines[-1] if lines else f"ffmpeg exit {proc.returncode}")
    os.replace(tmp, dst)


# -- .h264 remux ---------------------------------------------------------------


def _import_av() -> Any:
    """PyAV, or None when it is not installed."""
    try:
        import av
    except ImportError:
        return None
    return av


def sensor_timestamps_us(meta: Path) -> list[int]:
    """Per-frame `SensorTimestamp`s of a rpicam-vid `.meta.json`, in us from the
    first (parsed with the regex of remux_vfr.py: the file may be cut short)."""
    text = meta.read_text(errors="replace")
    ts = [int(x) for x in _SENSOR_TS.findall(text)]
    return [(t - ts[0]) // 1000 for t in ts] if ts else []


def remux_h264(src: Path, meta: Path, dst: Path) -> None:
    """Raw H.264 + `.meta.json` -> VFR Matroska with the sensor timestamps;
    frames past the last timestamp are dropped.

    Raises:
        InvalidVideo: no timestamps, or the stream cannot be read.
    """
    pts_us = sensor_timestamps_us(meta)
    if not pts_us:
        raise InvalidVideo(f"{meta.name} has no SensorTimestamp entries")
    av = _import_av()
    if av is None:
        _remux_h264_cli(src, pts_us, dst)
        return
    try:
        _remux_h264_av(av, src, pts_us, dst)
    except (av.FFmpegError, ValueError, IndexError) as exc:
        dst.unlink(missing_ok=True)
        raise InvalidVideo(f"cannot read the H.264 stream: {exc}") from None


def _remux_h264_av(av: Any, src: Path, pts_us: list[int], dst: Path) -> None:
    base = Fraction(1, 1_000_000)
    inp = av.open(str(src), format="h264")
    try:
        out = av.open(str(dst), "w", format="matroska")
        try:
            istream = inp.streams.video[0]
            ostream = out.add_stream_from_template(istream)
            ostream.time_base = base
            n = 0
            for packet in inp.demux(istream):
                if packet.size == 0:
                    continue
                if n >= len(pts_us):
                    break
                packet.pts = packet.dts = pts_us[n]
                packet.time_base = base
                packet.stream = ostream
                out.mux(packet)
                n += 1
        finally:
            out.close()
    finally:
        inp.close()
    if n == 0:
        raise InvalidVideo("the H.264 stream has no frames")


def _remux_h264_cli(src: Path, pts_us: list[int], dst: Path) -> None:
    """Fallback without PyAV: the CLI cannot set per-packet times, so the
    frames that have a timestamp are muxed at the timestamps' mean rate."""
    proc = _run(
        ["ffprobe", "-v", "error", "-f", "h264", "-count_packets", "-select_streams"]
        + ["v:0", "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0"]
        + [str(src)],
        REMUX_TIMEOUT_S,
    )
    try:
        packets = int(proc.stdout.decode().strip().split(",")[0])
    except ValueError:
        packets = 0
    n = min(packets, len(pts_us))
    if proc.returncode != 0 or n == 0:
        raise InvalidVideo(f"cannot read the H.264 stream: {_last_error(proc)}")
    step_s = (pts_us[n - 1] - pts_us[0]) / max(n - 1, 1) / 1e6 or 1 / 25
    ts = f"round(N*{step_s:.9f}/TB)"  # N = packet index (no B-frames, as rpicam)
    proc = _run(
        ["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "h264", "-i", str(src)]
        + ["-map", "0:v:0", "-frames:v", str(n), "-c", "copy"]
        + ["-bsf:v", f"setts=pts={ts}:dts={ts}", str(dst)],
        REMUX_TIMEOUT_S,
    )
    if proc.returncode != 0:
        dst.unlink(missing_ok=True)
        raise InvalidVideo(f"cannot remux the H.264 stream: {_last_error(proc)}")


# -- helpers -------------------------------------------------------------------


def _copy_hashing(stream: BinaryIO, dst: Path) -> str:
    digest = hashlib.sha256()
    with open(dst, "wb") as out:
        while chunk := stream.read(CHUNK):
            digest.update(chunk)
            out.write(chunk)
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    """sha256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def display_name(filename: str) -> str:
    """Base name of a client file name (no directories), at most 200 chars."""
    name = re.split(r"[\\/]", filename or "")[-1].strip()
    name = "".join(ch for ch in name if ch.isprintable())
    return name[:200] or "video"


def _suffix(filename: str) -> str:
    suffix = Path(display_name(filename)).suffix.lower()
    return suffix if re.fullmatch(r"\.[a-z0-9]{1,8}", suffix) else ""


def _is_raw(filename: str) -> bool:
    return _suffix(filename) in RAW_SUFFIXES


def _write_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, path)


def _within(path: Path, roots: Iterable[Path]) -> bool:
    return any(path == r or path.is_relative_to(r) for r in roots)


# -- library -------------------------------------------------------------------


class Library:
    """The video library under `root`; imports allowed from `data_roots`."""

    def __init__(self, root: Path, data_roots: Iterable[Path] = ()) -> None:
        self.root = Path(root)
        self.data_roots = tuple(Path(r) for r in data_roots)
        self._lock = threading.Lock()
        self._closed = False
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ddd-preview"
        )
        self._futures: set[concurrent.futures.Future[None]] = set()
        # Uploads cut by a previous shutdown leave staging folders behind.
        shutil.rmtree(self.root / INCOMING, ignore_errors=True)
        for record in self.list():  # resume previews interrupted by a restart
            if record.preview_status == "pending":
                self._schedule_preview(record.id)

    # -- queries --------------------------------------------------------------

    def allowed_roots(self) -> list[Path]:
        """The import folders that exist, resolved."""
        return [r.resolve() for r in self.data_roots if r.is_dir()]

    def list(self) -> list[VideoRecord]:
        """Every entry, newest first."""
        if not self.root.is_dir():
            return []
        records = [
            r
            for p in self.root.iterdir()
            if _ID.fullmatch(p.name) and (r := self._load(p.name)) is not None
        ]
        return sorted(records, key=lambda r: r.created, reverse=True)

    def get(self, video_id: str) -> VideoRecord:
        """The entry `video_id`.

        Raises:
            KeyError: not a library id (malformed or unknown).
        """
        record = self._load(video_id) if _ID.fullmatch(video_id or "") else None
        if record is None:
            raise KeyError(video_id)
        return record

    # -- adding ---------------------------------------------------------------

    def staging_dir(self) -> Path:
        """A new empty folder on the library's disk for an incoming upload;
        the caller removes it (whatever `add_staged` did not move away)."""
        incoming = self.root / INCOMING
        incoming.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(dir=incoming))

    def add(
        self, stream: BinaryIO, filename: str, meta: BinaryIO | None = None
    ) -> VideoRecord:
        """Add a video read from `stream` (in chunks); `meta` is the
        `.meta.json` of a raw `.h264`.

        Raises:
            InvalidVideo: not a video, truncated, or `.h264` without meta.
        """
        stage = self.staging_dir()
        try:
            src = stage / f"upload{_suffix(filename)}"
            digest = _copy_hashing(stream, src)
            meta_path = None
            if meta is not None:
                meta_path = stage / "upload.meta.json"
                _copy_hashing(meta, meta_path)
            return self.add_staged(stage, src, filename, digest, meta_path)
        finally:
            shutil.rmtree(stage, ignore_errors=True)

    def add_staged(
        self,
        stage: Path,
        src: Path,
        filename: str,
        sha256: str,
        meta: Path | None = None,
    ) -> VideoRecord:
        """Add an upload already written to `src` inside `stage`
        (`staging_dir()`), whose content hashes to `sha256`.

        Raises:
            InvalidVideo: as in `add`.
        """
        video_id = sha256[:ID_LEN]
        if (existing := self._load(video_id)) is not None:
            return existing
        work = stage / "entry"
        work.mkdir()
        if _is_raw(filename):
            if meta is None:
                raise InvalidVideo(
                    f"{display_name(filename)} is a raw H.264 stream: add its "
                    ".meta.json (sensor timestamps) to the same upload"
                )
            original = work / "original.mkv"
            remux_h264(src, meta, original)
        else:
            original = work / f"original{_suffix(filename)}"
            os.replace(src, original)
        return self._finish(video_id, work, original, None, filename, "upload")

    def add_from_path(self, path: Path) -> VideoRecord:
        """Add a file of the server by reference (a `.h264` with its sibling
        `.meta.json` is converted into the library).

        Raises:
            OutsideRoots: relative path, or it resolves (through `..` or a
                symlink) outside every allowed root.
            FileNotFoundError: no such file (inside the roots).
            InvalidVideo: as in `add`.
        """
        path = Path(path)
        if not path.is_absolute():
            raise OutsideRoots(f"{path}: give an absolute path on the server")
        roots = self.allowed_roots()
        if not roots:
            raise OutsideRoots("importing server files is disabled (no --data-root)")
        real = path.resolve()
        if not _within(real, roots):
            raise OutsideRoots(f"{path} is outside the import folders")
        if not real.is_file():
            raise FileNotFoundError(f"{path}: no such file")
        video_id = file_sha256(real)[:ID_LEN]
        if (existing := self._load(video_id)) is not None:
            return existing
        stage = self.staging_dir()
        try:
            work = stage / "entry"
            work.mkdir()
            if _is_raw(real.name):
                meta = real.with_name(f"{real.stem}.meta.json").resolve()
                if not (_within(meta, roots) and meta.is_file()):
                    raise InvalidVideo(
                        f"{real.name} is a raw H.264 stream without "
                        f"{real.stem}.meta.json next to it"
                    )
                original = work / "original.mkv"
                remux_h264(real, meta, original)
                return self._finish(video_id, work, original, None, path.name, "import")
            return self._finish(video_id, work, real, real, path.name, "import")
        finally:
            shutil.rmtree(stage, ignore_errors=True)

    # -- previews -------------------------------------------------------------

    def wait_previews(self, timeout: float | None = None) -> None:
        """Block until the scheduled previews are done (tests, shutdown)."""
        concurrent.futures.wait(list(self._futures), timeout=timeout)

    def close(self) -> None:
        """Stop the preview thread (a running ffmpeg is killed; its entry stays
        `pending` and resumes on the next start)."""
        self._closed = True
        self._pool.shutdown(wait=False, cancel_futures=True)
        with _PREVIEW_LOCK:
            for proc in list(_PREVIEW_PROCS):
                with contextlib.suppress(OSError):
                    proc.kill()

    # -- internals ------------------------------------------------------------

    def _finish(
        self,
        video_id: str,
        work: Path,
        original: Path,
        reference: Path | None,
        filename: str,
        origin: str,
    ) -> VideoRecord:
        """Probe, thumbnail and move `work` into place as entry `video_id`."""
        info = probe(original)
        _thumbnail(original, work / THUMB, min(1.0, info.duration_s / 3))
        record = {
            "id": video_id,
            "name": display_name(filename),
            "width": info.width,
            "height": info.height,
            "fps": info.fps,
            "duration_s": info.duration_s,
            "size_bytes": original.stat().st_size,
            # inside the entry: a name relative to it; an import: absolute path
            "original": str(reference) if reference else original.name,
            "preview_status": "pending",
            "created": datetime.now().astimezone().isoformat(timespec="seconds"),
            "origin": origin,
        }
        _write_json(work / RECORD, record)
        target = self.root / video_id
        with self._lock:
            if target.exists():  # the same content arrived meanwhile
                existing = self._load(video_id)
                if existing is not None:
                    return existing
                shutil.rmtree(target, ignore_errors=True)
            os.rename(work, target)
        self._schedule_preview(video_id)
        loaded = self._load(video_id)
        assert loaded is not None
        return loaded

    def _load(self, video_id: str) -> VideoRecord | None:
        folder = self.root / video_id
        try:
            data = json.loads((folder / RECORD).read_text())
            original = Path(data["original"])
            preview = folder / PREVIEW
            status = str(data.get("preview_status", "pending"))
            return VideoRecord(
                id=video_id,
                name=str(data["name"]),
                width=int(data["width"]),
                height=int(data["height"]),
                fps=float(data["fps"]),
                duration_s=float(data["duration_s"]),
                size_bytes=int(data["size_bytes"]),
                original=original if original.is_absolute() else folder / original,
                preview=preview if status == "ready" and preview.is_file() else None,
                thumb=folder / THUMB,
                preview_status=status,
                created=str(data.get("created", "")),
                origin=str(data.get("origin", "upload")),
            )
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _schedule_preview(self, video_id: str) -> None:
        if self._closed:
            return
        future = self._pool.submit(self._preview_job, video_id)
        self._futures.add(future)
        future.add_done_callback(self._futures.discard)

    def _preview_job(self, video_id: str) -> None:
        record = self._load(video_id)
        if record is None or self._closed:
            return
        folder = self.root / video_id
        try:
            _make_preview(record.original, folder / PREVIEW)
            status, error = "ready", None
        except Exception as exc:  # noqa: BLE001 - recorded on the entry
            if self._closed:
                return  # killed by close(): resume on the next start
            status, error = "failed", str(exc)
        with self._lock:
            path = folder / RECORD
            with contextlib.suppress(OSError, ValueError):
                data = json.loads(path.read_text())
                data["preview_status"] = status
                data["preview_error"] = error
                _write_json(path, data)
