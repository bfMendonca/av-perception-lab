import itertools

import numpy as np
import pytest
from mcap_fixture import CameraInfoSpec, ImageSpec, write_mcap

from av_perception_lab.config import SourceConfig
from av_perception_lab.core.options import ConfigError
from av_perception_lab.core.types import ContractError, Frame
from av_perception_lab.io.mcap_reader import CameraInfoMsg
from av_perception_lab.io.source import FrameTransform, McapSource

TOPIC = "/cam/image_raw"
BASE_NS = 5_000_000_000  # recordings rarely start at 0
K = [100, 0, 8, 0, 100, 6, 0, 0, 1]


def _frames(n, h=12, w=16):
    """Frame i is filled with the value 10 * i (identifies it after decoding)."""
    return [np.full((h, w, 3), 10 * i, np.uint8) for i in range(n)]


def _ns(seconds):
    return [BASE_NS + round(s * 1e9) for s in seconds]


def _record(tmp_path, seconds, *, header=None, frames=None, **kwargs):
    frames = frames if frames is not None else _frames(len(seconds))
    return write_mcap(
        tmp_path / "rec.mcap",
        images=[
            ImageSpec(
                topic=TOPIC,
                frames=frames,
                times_ns=_ns(seconds),
                header_ns=header,
            )
        ],
        **kwargs,
    )


def _source(path, **cfg):
    return McapSource(SourceConfig(mcap=path, topic=TOPIC, **cfg))


def _ids(frames):
    return [int(f.rgb[0, 0, 0]) // 10 for f in frames]


def test_frames_use_message_times(tmp_path):
    path = _record(
        tmp_path,
        [0.0, 0.10, 0.35],
        camera_infos=[
            CameraInfoSpec(topic="/cam/camera_info", width=16, height=12, k=K)
        ],
    )
    source = _source(path)
    frames = list(source)
    assert all(isinstance(f, Frame) for f in frames)
    assert [f.index for f in frames] == [0, 1, 2]
    assert [f.t for f in frames] == pytest.approx([0.0, 0.10, 0.35])
    assert _ids(frames) == [0, 1, 2]
    assert source.frame_size == (16, 12)
    assert source.name == "rec_cam_image_raw"
    assert source.input_name == str(path)
    assert source.frame_times_known is True
    assert source.n_frames_total == 3
    assert source.nominal_fps == pytest.approx(1 / 0.175)  # median interval
    assert source.audio_from() is None
    info = source.camera_info()
    assert isinstance(info, CameraInfoMsg) and info.k == tuple(map(float, K))
    d = source.describe()
    assert d["path"] == str(path) and d["topic"] == TOPIC
    assert d["schema"] == "sensor_msgs/msg/Image" and d["count"] == 3
    assert d["time_base"] == "header" and d["duplicates_dropped"] == 0
    assert d["camera_info_topic"] == "/cam/camera_info" and d["transform"] is None


def test_stride_start_duration_max_frames_count_messages(tmp_path):
    path = _record(tmp_path, [i / 10 for i in range(10)])
    selected = list(_source(path, start=0.25, duration=0.5, stride=2))
    assert [f.index for f in selected] == [3, 5, 7]
    assert _ids(selected) == [3, 5, 7]
    assert [f.t for f in selected] == pytest.approx([0.0, 0.2, 0.4])
    capped = list(_source(path, start=0.25, duration=0.5, stride=2, max_frames=2))
    assert [f.index for f in capped] == [3, 5]
    assert [f.index for f in _source(path, stride=3)] == [0, 3, 6, 9]
    assert list(_source(path, start=5.0)) == []
    assert _source(path).camera_info() is None  # no CameraInfo topic


def test_header_stamp_zero_falls_back_to_log_time(tmp_path):
    path = _record(tmp_path, [0.0, 0.2, 0.3], header=[0, 0, 0])
    source = _source(path)
    assert [f.t for f in source] == pytest.approx([0.0, 0.2, 0.3])
    assert source.describe()["time_base"] == "log"


def test_header_stamp_is_used_when_set_and_monotonic(tmp_path):
    # header stamps differ from the log times (e.g. capture vs. record time)
    path = _record(
        tmp_path, [0.0, 0.1, 0.2], header=[7_000_000_000, 7_050_000_000, 7_300_000_000]
    )
    source = _source(path)
    assert [f.t for f in source] == pytest.approx([0.0, 0.05, 0.3])
    assert source.describe()["time_base"] == "header"


def test_decreasing_header_falls_back_to_log_time(tmp_path):
    path = _record(tmp_path, [0.0, 0.1, 0.2], header=_ns([0.3, 0.2, 0.1]))
    source = _source(path)
    assert [f.t for f in source] == pytest.approx([0.0, 0.1, 0.2])
    assert source.describe()["time_base"] == "log"


def test_duplicate_timestamps_are_dropped_and_counted(tmp_path):
    path = _record(tmp_path, [0.0, 0.1, 0.1, 0.2])
    source = _source(path)
    frames = list(source)
    ts = [f.t for f in frames]
    assert all(b > a for a, b in itertools.pairwise(ts))
    assert ts == pytest.approx([0.0, 0.1, 0.2])
    assert _ids(frames) == [0, 1, 3]  # the second message at 0.1 s is dropped
    assert [f.index for f in frames] == [0, 1, 2]
    assert source.describe()["duplicates_dropped"] == 1


def test_size_change_raises_contract_error(tmp_path):
    frames = [np.zeros((12, 16, 3), np.uint8), np.zeros((10, 16, 3), np.uint8)]
    path = _record(tmp_path, [0.0, 0.1], frames=frames)
    source = _source(path)
    with pytest.raises(ContractError, match="16x10"):
        list(source)


def test_missing_topic_lists_image_topics(tmp_path):
    path = write_mcap(
        tmp_path / "rec.mcap",
        images=[
            ImageSpec(topic="/a/image_raw", frames=_frames(1), times_ns=[0]),
            ImageSpec(topic="/b/image_raw", frames=_frames(1), times_ns=[0]),
        ],
    )
    with pytest.raises(ConfigError, match=r"/a/image_raw, /b/image_raw"):
        _source(path)
    with pytest.raises(ConfigError, match="not found"):
        _source(tmp_path / "nope.mcap")
    bad = tmp_path / "bad.mcap"
    bad.write_bytes(b"not an mcap file")
    with pytest.raises(ConfigError, match="bad.mcap"):
        _source(bad)


def test_sample_spreads_over_selection(tmp_path):
    path = _record(tmp_path, [i / 10 for i in range(10)])
    samples = _source(path, max_frames=2, stride=5).sample(3)
    assert [int(rgb[0, 0, 0]) // 10 for rgb in samples] == [0, 3, 6]
    late = _source(path, start=0.5).sample(3)
    assert [int(rgb[0, 0, 0]) // 10 for rgb in late] == [5, 6, 7]


class Invert(FrameTransform):
    def __init__(self):
        self.calls = 0

    def __call__(self, rgb):
        self.calls += 1
        return 255 - rgb

    def describe(self):
        return {"type": "invert"}


class Crop(Invert):
    def __call__(self, rgb):
        return rgb[1:]


def test_transform_applied_and_shape_checked(tmp_path):
    path = _record(tmp_path, [0.0, 0.1, 0.2])
    plain = _source(path)
    spy = Invert()
    source = plain.with_transform(spy)
    assert plain.transform is None and source.transform is spy
    frames, raw = list(source), list(plain)
    assert spy.calls == 3
    assert all(
        np.array_equal(f.rgb, 255 - r.rgb) and f.index == r.index and f.t == r.t
        for f, r in zip(frames, raw, strict=True)
    )
    assert len(source.sample(2)) == 2 and spy.calls == 5
    assert source.describe()["transform"] == {"type": "invert"}
    with pytest.raises(ContractError, match=r"Crop returned \(11, 16, 3\)"):
        list(McapSource(SourceConfig(mcap=path, topic=TOPIC), transform=Crop()))


def test_source_config_exclusivity(tmp_path):
    with pytest.raises(ValueError, match="mutually exclusive"):
        SourceConfig(video=tmp_path / "a.mp4", mcap=tmp_path / "a.mcap", topic=TOPIC)
    with pytest.raises(ValueError, match="topic"):
        SourceConfig(mcap=tmp_path / "a.mcap")
    cfg = SourceConfig(mcap=tmp_path / "a.mcap", topic=TOPIC)
    assert cfg.video is None and cfg.topic == TOPIC
    with pytest.raises(ConfigError, match="no input MCAP"):
        McapSource(SourceConfig())


def test_unsupported_topic_is_refused_naming_the_encoding(tmp_path):
    depth = [np.full((12, 16), 1000, np.uint16)]
    path = write_mcap(
        tmp_path / "rec.mcap",
        images=[ImageSpec(topic=TOPIC, frames=depth, times_ns=[0], encoding="16UC1")],
    )
    with pytest.raises(ConfigError, match=r"source.topic: .*16UC1"):
        _source(path)


@pytest.fixture
def decode_calls(monkeypatch):
    """Counts the image decodes (cvtColor / imdecode) of the MCAP reader."""
    from av_perception_lab.io import mcap_reader

    calls = []
    real = mcap_reader.decode_image
    monkeypatch.setattr(
        mcap_reader, "decode_image", lambda *a: calls.append(1) or real(*a)
    )
    return calls


@pytest.mark.parametrize("indexed", [True, False])
def test_discarded_messages_are_never_decoded(tmp_path, decode_calls, indexed):
    seconds = [i / 10 for i in range(20)]
    seconds[7] = seconds[6]  # a duplicate: dropped, never decoded
    path = _record(tmp_path, seconds, indexed=indexed)
    source = _source(path, start=0.25, stride=3)
    assert source.time_base == "header" and decode_calls == []
    frames = list(source)
    assert [f.index for f in frames] == [3, 6, 9, 12, 15, 18]
    assert len(decode_calls) == len(frames)
    decode_calls.clear()
    assert len(_source(path, max_frames=2).sample(4)) == 4
    assert len(decode_calls) == 4
    decode_calls.clear()
    assert [f.index for f in _source(path, max_frames=2, stride=2)] == [0, 2]
    assert len(decode_calls) == 2


def test_sample_spreads_over_exact_selection(tmp_path, decode_calls):
    path = _record(tmp_path, [i / 10 for i in range(10)], indexed=False)
    samples = _source(path, start=0.2, duration=0.6).sample(3)
    assert [int(rgb[0, 0, 0]) // 10 for rgb in samples] == [2, 4, 6]
    assert len(decode_calls) == 3


def test_header_stamp_reset_after_the_head_window_is_reported(tmp_path):
    # header stamps restart (sim time / concatenated bags) after message 60,
    # past the TIME_BASE_WINDOW messages that chose the header time base
    seconds = [i / 10 for i in range(100)]
    header = _ns(seconds[:60]) + _ns([s / 100 for s in range(40)])
    path = _record(tmp_path, seconds, header=header)
    source = _source(path)
    assert len(list(source)) == 60
    d = source.describe()
    assert d["time_base"] == "header" and d["duplicates_dropped"] == 40
    assert d["stamp_regression_at"] == 60 and d["dropped_after_regression"] == 40
    (warning,) = source.warnings()
    assert "40 of the 100 messages" in warning and "40.0 %" in warning
    assert "header.stamp" in warning and "message 60" in warning
    assert "reset" in warning


def test_no_warning_without_drops_and_repeats_are_not_regressions(tmp_path):
    clean = _source(_record(tmp_path, [0.0, 0.1, 0.2]))
    list(clean)
    assert clean.warnings() == []
    assert clean.describe()["stamp_regression_at"] is None
    repeat = _source(
        write_mcap(
            tmp_path / "dup.mcap",
            images=[
                ImageSpec(topic=TOPIC, frames=_frames(3), times_ns=_ns([0, 0.1, 0.1]))
            ],
        )
    )
    list(repeat)
    d = repeat.describe()
    assert d["duplicates_dropped"] == 1 and d["stamp_regression_at"] is None
    (warning,) = repeat.warnings()
    assert "1 of the 3 messages" in warning and "reset" not in warning
