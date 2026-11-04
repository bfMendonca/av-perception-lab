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
