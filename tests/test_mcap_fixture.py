import numpy as np
from mcap.reader import make_reader
from mcap_fixture import CameraInfoSpec, ImageSpec, write_mcap


def test_roundtrip(tmp_path):
    frames = [np.full((4, 6, 3), v, np.uint8) for v in (10, 200)]
    path = write_mcap(
        tmp_path / "a.mcap",
        images=[
            ImageSpec(topic="/cam/image_raw", frames=frames, times_ns=[1_000, 2_000])
        ],
        camera_infos=[
            CameraInfoSpec(
                topic="/cam/camera_info",
                width=6,
                height=4,
                k=[5, 0, 3, 0, 5, 2, 0, 0, 1],
            )
        ],
    )
    with path.open("rb") as f:
        reader = make_reader(f)
        msgs = [
            (s.name, c.topic)
            for s, c, _ in reader.iter_messages(topics=["/cam/image_raw"])
        ]
    assert msgs == [("sensor_msgs/msg/Image", "/cam/image_raw")] * 2


def test_variants_write(tmp_path):
    frames = [np.zeros((4, 6, 3), np.uint8)]
    for i, kw in enumerate(
        [
            {"encoding": "bgra8"},
            {"encoding": "mono8", "step_pad": 3},
            {"compressed": "png"},
        ]
    ):
        spec = ImageSpec(topic="/c", frames=frames, times_ns=[1], **kw)
        for indexed in (True, False):
            p = write_mcap(
                tmp_path / f"{i}{indexed}.mcap", images=[spec], indexed=indexed
            )
            with p.open("rb") as f:
                assert len(list(make_reader(f).iter_messages())) == 1
