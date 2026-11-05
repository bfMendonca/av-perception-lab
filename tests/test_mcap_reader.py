import cv2
import numpy as np
import pytest
from mcap_fixture import CameraInfoSpec, ImageSpec, write_mcap

from av_perception_lab.io.mcap_reader import (
    McapError,
    decode_image,
    iter_images,
    pair_camera_info,
    probe_mcap,
    read_camera_info,
)

K = [100, 0, 8, 0, 100, 6, 0, 0, 1]


def _frames(n=3, h=12, w=16, seed=0):
    rng = np.random.default_rng(seed)
    # smooth content so JPEG error stays small
    base = rng.integers(0, 255, (h // 4, w // 4, 3), dtype=np.uint8)
    return [cv2.resize(base, (w, h), interpolation=cv2.INTER_LINEAR) for _ in range(n)]


def test_probe_lists_image_topics_with_resolution_and_count(tmp_path):
    path = write_mcap(
        tmp_path / "a.mcap",
        images=[
            ImageSpec(
                topic="/cam/image_raw",
                frames=_frames(3),
                times_ns=[0, 500_000_000, 1_000_000_000],
            ),
            ImageSpec(
                topic="/cam2/image_raw/compressed",
                frames=_frames(2, 8, 10),
                times_ns=[0, 1_000_000_000],
                compressed="png",
            ),
        ],
        camera_infos=[
            CameraInfoSpec(topic="/cam/camera_info", width=16, height=12, k=K)
        ],
    )
    info = probe_mcap(path)
    assert info.path == path
    assert [i.topic.name for i in info.images] == [
        "/cam/image_raw",
        "/cam2/image_raw/compressed",
    ]
    a, b = info.images
    assert (a.width, a.height, a.topic.count) == (16, 12, 3)
    assert a.topic.rate_hz == pytest.approx(2.0)
    assert a.encoding == "rgb8"
    assert a.camera_info_topic == "/cam/camera_info"
    assert (b.width, b.height, b.topic.count) == (10, 8, 2)
    assert b.topic.rate_hz == pytest.approx(1.0)
    assert b.camera_info_topic is None
    assert info.camera_info_topics == ("/cam/camera_info",)
    assert info.duration_s == pytest.approx(1.0)


def test_pairing():
    assert (
        pair_camera_info("/cam/image_raw", ["/cam/camera_info"]) == "/cam/camera_info"
    )
    assert (
        pair_camera_info("/cam/image_raw/compressed", ["/cam/camera_info"])
        == "/cam/camera_info"
    )
    assert pair_camera_info("/cam/image_raw", ["/other/camera_info"]) is None
    assert pair_camera_info("/cam/image_raw", []) is None
    assert (
        pair_camera_info(
            "/a/cam/image_raw", ["/b/cam/camera_info", "/a/cam/camera_info"]
        )
        == "/a/cam/camera_info"
    )
    assert (
        pair_camera_info(
            "/cam/left/image_raw", ["/cam/camera_info", "/cam/left/camera_info"]
        )
        == "/cam/left/camera_info"
    )


def _gray(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


@pytest.mark.parametrize(
    "encoding,compressed",
    [
        ("rgb8", None),
        ("bgr8", None),
        ("rgba8", None),
        ("bgra8", None),
        ("mono8", None),
        ("rgb8", "jpeg"),
        ("rgb8", "png"),
    ],
)
def test_decode_encodings(tmp_path, encoding, compressed):
    frames = _frames(2)
    path = write_mcap(
        tmp_path / "a.mcap",
        images=[
            ImageSpec(
                topic="/c",
                frames=frames,
                times_ns=[10, 20],
                encoding=encoding,
                compressed=compressed,
            )
        ],
    )
    out = list(iter_images(path, "/c"))
    assert [m.log_ns for m in out] == [10, 20]
    for msg, src in zip(out, frames, strict=True):
        assert msg.rgb.dtype == np.uint8 and msg.rgb.shape == src.shape
        if encoding == "mono8":
            expected = np.repeat(_gray(src)[..., None], 3, axis=-1)
        else:
            expected = src
        diff = np.abs(msg.rgb.astype(int) - expected.astype(int))
        if compressed == "jpeg":
            assert diff.mean() <= 6
        else:
            assert diff.max() == 0


def test_unknown_encoding_names_it():
    class Msg:
        height, width, step, encoding, data = 1, 1, 2, "16UC1", b"\x00\x00"

    with pytest.raises(McapError, match="16UC1"):
        decode_image(Msg(), "sensor_msgs/msg/Image")


def test_step_padding_decodes_without_shear(tmp_path):
    frames = _frames(1, 6, 7)
    for enc in ("rgb8", "mono8"):
        path = write_mcap(
            tmp_path / f"{enc}.mcap",
            images=[
                ImageSpec(
                    topic="/c", frames=frames, times_ns=[1], encoding=enc, step_pad=5
                )
            ],
        )
        (msg,) = list(iter_images(path, "/c"))
        expected = (
            frames[0]
            if enc == "rgb8"
            else np.repeat(_gray(frames[0])[..., None], 3, axis=-1)
        )
        assert np.array_equal(msg.rgb, expected)


def test_header_ns_differs_from_log_ns(tmp_path):
    path = write_mcap(
        tmp_path / "a.mcap",
        images=[
            ImageSpec(
                topic="/c",
                frames=_frames(1),
                times_ns=[5_000],
                header_ns=[1_500_000_007],
            )
        ],
    )
    (msg,) = list(iter_images(path, "/c"))
    assert (msg.log_ns, msg.header_ns) == (5_000, 1_500_000_007)


def test_probe_without_summary_scans(tmp_path):
    path = write_mcap(
        tmp_path / "a.mcap",
        images=[
            ImageSpec(
                topic="/c",
                frames=_frames(4),
                times_ns=[0, 100_000_000, 300_000_000, 600_000_000],
            )
        ],
        camera_infos=[CameraInfoSpec(topic="/camera_info", width=16, height=12, k=K)],
        indexed=False,
    )
    info = probe_mcap(path)
    (img,) = info.images
    assert img.topic.count == 4
    assert (img.topic.first_ns, img.topic.last_ns) == (0, 600_000_000)
    assert img.topic.rate_hz == pytest.approx(3 / 0.6)
    assert (img.width, img.height) == (16, 12)
    assert info.duration_s == pytest.approx(0.6)
    assert len(list(iter_images(path, "/c"))) == 4


def test_single_message_rate_is_zero(tmp_path):
    path = write_mcap(
        tmp_path / "a.mcap",
        images=[ImageSpec(topic="/c", frames=_frames(1), times_ns=[7])],
    )
    assert probe_mcap(path).images[0].topic.rate_hz == 0.0


@pytest.mark.parametrize("indexed", [True, False])
def test_truncated_file_raises_mcap_error(tmp_path, indexed):
    path = write_mcap(
        tmp_path / "a.mcap",
        images=[ImageSpec(topic="/c", frames=_frames(5), times_ns=[0, 1, 2, 3, 4])],
        indexed=indexed,
    )
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])
    with pytest.raises(McapError):
        probe_mcap(path)
    with pytest.raises(McapError):
        list(iter_images(path, "/c"))


def test_not_an_mcap_and_missing_topic(tmp_path):
    bad = tmp_path / "bad.mcap"
    bad.write_bytes(b"not an mcap at all")
    with pytest.raises(McapError):
        probe_mcap(bad)
    good = write_mcap(
        tmp_path / "a.mcap",
        images=[ImageSpec(topic="/c", frames=_frames(1), times_ns=[1])],
    )
    with pytest.raises(McapError, match="/nope"):
        list(iter_images(good, "/nope"))
    with pytest.raises(McapError):
        probe_mcap(tmp_path / "missing.mcap")


def test_read_camera_info_first_message(tmp_path):
    path = write_mcap(
        tmp_path / "a.mcap",
        camera_infos=[
            CameraInfoSpec(
                topic="/cam/camera_info",
                width=16,
                height=12,
                k=K,
                d=[0.1, -0.2, 0.0, 0.0, 0.0],
                model="plumb_bob",
                roi=(1, 2, 3, 4),
                binning=2,
            )
        ],
    )
    info = read_camera_info(path, "/cam/camera_info")
    assert (info.width, info.height) == (16, 12)
    assert info.k == tuple(float(v) for v in K)
    assert info.d == (0.1, -0.2, 0.0, 0.0, 0.0)
    assert info.model == "plumb_bob"
    assert len(info.p) == 12
    assert info.roi == (1, 2, 3, 4)
    assert info.binning == 2
    with pytest.raises(McapError, match="/other"):
        read_camera_info(path, "/other")


@pytest.mark.parametrize(
    ("compressed", "fmt", "expected"),
    [
        ("jpeg", "rgb8; jpeg compressed bgr8", "jpeg"),
        ("png", "bgr8; png compressed bgr8", "png"),
        ("jpeg", "JPEG", "jpeg"),
        ("png", "png", "png"),
    ],
)
def test_probe_reports_codec_of_image_transport_format(
    tmp_path, compressed, fmt, expected
):
    path = write_mcap(
        tmp_path / "a.mcap",
        images=[
            ImageSpec(
                topic="/c/image_raw/compressed",
                frames=_frames(1),
                times_ns=[1],
                compressed=compressed,
                format=fmt,
            )
        ],
    )
    assert probe_mcap(path).images[0].encoding == expected


def test_probe_flags_unsupported_topics_without_raising(tmp_path):
    depth = [np.full((12, 16), 1000, np.uint16)] * 2
    path = write_mcap(
        tmp_path / "a.mcap",
        images=[
            ImageSpec(topic="/cam/color/image_raw", frames=_frames(2), times_ns=[0, 1]),
            ImageSpec(
                topic="/cam/aligned_depth/image_raw",
                frames=depth,
                times_ns=[0, 1],
                encoding="16UC1",
            ),
            ImageSpec(
                topic="/cam/depth/image_raw/compressedDepth",
                frames=_frames(2),
                times_ns=[0, 1],
                compressed="png",
                format="16UC1; compressedDepth png",
                data_prefix=bytes(12),
            ),
            ImageSpec(
                topic="/cam/broken/compressed",
                frames=_frames(1),
                times_ns=[0],
                compressed="jpeg",
                data_prefix=b"garbage",
            ),
        ],
    )
    info = probe_mcap(path)
    topics = {i.topic.name: i for i in info.images}
    color = topics["/cam/color/image_raw"]
    assert color.supported and color.unsupported_reason is None
    raw16 = topics["/cam/aligned_depth/image_raw"]
    assert not raw16.supported and "16UC1" in raw16.unsupported_reason
    assert (raw16.width, raw16.height, raw16.encoding) == (16, 12, "16UC1")
    comp_depth = topics["/cam/depth/image_raw/compressedDepth"]
    assert (
        not comp_depth.supported and "compressedDepth" in comp_depth.unsupported_reason
    )
    assert comp_depth.encoding == "compressedDepth"  # not "png"
    broken = topics["/cam/broken/compressed"]
    assert not broken.supported and "decode" in broken.unsupported_reason
    # the reader still names the encoding when an unsupported topic is read
    with pytest.raises(McapError, match="16UC1"):
        list(iter_images(path, "/cam/aligned_depth/image_raw"))
    with pytest.raises(McapError, match="compressedDepth"):
        list(iter_images(path, "/cam/depth/image_raw/compressedDepth"))


def test_cdr_stamp_matches_the_decoder_both_byte_orders():
    from av_perception_lab.io.mcap_reader import cdr_stamp_ns

    le = bytes([0, 1, 0, 0]) + (-6).to_bytes(4, "little", signed=True)
    le += (999_999_999).to_bytes(4, "little") + b"rest"
    be = bytes([0, 0, 0, 0]) + (7).to_bytes(4, "big", signed=True)
    be += (123).to_bytes(4, "big")
    assert cdr_stamp_ns(le) == -6 * 1_000_000_000 + 999_999_999
    assert cdr_stamp_ns(be) == 7_000_000_123
    assert cdr_stamp_ns(bytes([0, 7, 0, 0]) + bytes(8)) is None  # not plain CDR
    assert cdr_stamp_ns(bytes([0, 1, 0, 0, 1])) is None  # too short


@pytest.mark.parametrize("compressed", [None, "png"])
def test_raw_images_read_stamps_without_decoding(tmp_path, monkeypatch, compressed):
    from av_perception_lab.io import mcap_reader

    stamps = [7_123_456_789, 1_000_000_000, 3]
    path = write_mcap(
        tmp_path / "a.mcap",
        images=[
            ImageSpec(
                topic="/c",
                frames=_frames(3),
                times_ns=[1, 2, 3],
                header_ns=stamps,
                compressed=compressed,
            )
        ],
    )
    calls = []
    real = mcap_reader.decode_image
    monkeypatch.setattr(
        mcap_reader, "decode_image", lambda *a: calls.append(1) or real(*a)
    )
    raw = list(mcap_reader.iter_raw_images(path, "/c"))
    assert [(m.log_ns, m.header_ns) for m in raw] == list(zip([1, 2, 3], stamps))
    assert calls == []
    assert np.array_equal(raw[1].decode(), list(iter_images(path, "/c"))[1].rgb)
    assert len(calls) == 4  # raw[1].decode() + the three of iter_images


@pytest.mark.parametrize("indexed", [True, False])
def test_read_images_at_fetches_only_the_targets(tmp_path, monkeypatch, indexed):
    from av_perception_lab.io import mcap_reader

    frames = [np.full((4, 4, 3), 10 * i, np.uint8) for i in range(6)]
    times = [0, 10, 10, 20, 30, 40]  # a log-time tie: rank tells them apart
    path = write_mcap(
        tmp_path / "a.mcap",
        images=[ImageSpec(topic="/c", frames=frames, times_ns=times)],
        indexed=indexed,
    )
    calls = []
    real = mcap_reader.decode_image
    monkeypatch.setattr(
        mcap_reader, "decode_image", lambda *a: calls.append(1) or real(*a)
    )
    got = mcap_reader.read_images_at(path, "/c", [(10, 1), (30, 0)])
    assert [int(m.decode()[0, 0, 0]) // 10 for m in got] == [2, 4]
    assert len(calls) == 2
    with pytest.raises(McapError, match="no message"):
        mcap_reader.read_images_at(path, "/c", [(25, 0)])


def _huge_png(data: bytes, w: int = 60000, h: int = 60000) -> bytes:
    """A PNG whose IHDR declares w x h (the pixels stay tiny)."""
    return data[:16] + w.to_bytes(4, "big") + h.to_bytes(4, "big") + data[24:]


def _huge_jpeg(data: bytes, w: int = 60000, h: int = 60000) -> bytes:
    """A JPEG whose SOF segment declares w x h."""
    i = 2
    while True:
        marker, length = data[i + 1], int.from_bytes(data[i + 2 : i + 4], "big")
        if marker in (0xC0, 0xC1, 0xC2):
            dims = h.to_bytes(2, "big") + w.to_bytes(2, "big")
            return data[: i + 5] + dims + data[i + 9 :]
        i += 2 + length


class _Compressed:
    def __init__(self, data, fmt):
        self.data, self.format = data, fmt


@pytest.mark.parametrize(("ext", "craft"), [(".png", _huge_png), (".jpg", _huge_jpeg)])
def test_compressed_pixel_cap_is_checked_before_decoding(monkeypatch, ext, craft):
    from av_perception_lab.io import mcap_reader

    _, buf = cv2.imencode(ext, np.zeros((6, 8, 3), np.uint8))
    data = buf.tobytes()
    assert mcap_reader.compressed_size(data) == (8, 6)
    assert mcap_reader.compressed_size(craft(data, 9, 7)) == (9, 7)
    schema = "sensor_msgs/msg/CompressedImage"
    assert decode_image(_Compressed(data, ext[1:]), schema).shape == (6, 8, 3)

    def no_imdecode(*_):
        raise AssertionError("imdecode called on an oversized image")

    monkeypatch.setattr(mcap_reader.cv2, "imdecode", no_imdecode)
    with pytest.raises(McapError, match=r"60000x60000 .*above the cap"):
        decode_image(_Compressed(craft(data), ext[1:]), schema)
    with pytest.raises(McapError, match="not a PNG or JPEG"):
        decode_image(_Compressed(b"GIF89a" + bytes(20), "png"), schema)


def test_raw_pixel_cap():
    from av_perception_lab.io.mcap_reader import MAX_IMAGE_PIXELS

    class Msg:
        height, width, step, encoding, data = 60000, 60000, 180000, "rgb8", b"\x00"

    assert 60000 * 60000 > MAX_IMAGE_PIXELS >= 8192 * 8192
    with pytest.raises(McapError, match=r"60000x60000 .*above the cap"):
        decode_image(Msg(), "sensor_msgs/msg/Image")


def test_probe_flags_an_oversized_compressed_topic(tmp_path):
    path = write_mcap(
        tmp_path / "a.mcap",
        images=[
            ImageSpec(
                topic="/c",
                frames=_frames(1),
                times_ns=[1],
                compressed="png",
                data_patch=_huge_png,
            )
        ],
    )
    (topic,) = probe_mcap(path).images
    assert not topic.supported and "above the cap" in topic.unsupported_reason
