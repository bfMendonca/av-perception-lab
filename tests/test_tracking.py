import json
import math

import numpy as np
import pytest

from det_depth_distance.core import registry
from det_depth_distance.core.components import LoadContext
from det_depth_distance.core.options import ConfigError
from det_depth_distance.core.tracking import (
    IouTracker,
    TrackSummary,
    association_keys,
)
from det_depth_distance.core.types import (
    CameraIntrinsics,
    DepthResult,
    Frame,
    FrameResult,
    Instances,
    MapGeometry,
    Provenance,
    TrackEstimate,
)
from det_depth_distance.distance import BoxDepthStats

FPS = 30.0
W, H = 1920, 1080
RGB = np.zeros((H, W, 3), np.uint8)  # shared: the IoU tracker never reads pixels
CTX = LoadContext("cpu", (W, H), None)


def frame(k: int, fps: float = FPS) -> Frame:
    return Frame(k, k / fps, RGB)


def tracker(**options) -> IouTracker:
    t = IouTracker(IouTracker.Options(**options))
    t.load(CTX)
    return t


def jittered(rng, box):
    return np.asarray(box, dtype=np.float64) + rng.uniform(-2, 2, size=4)


def track_ids(t: IouTracker, frames: list[tuple[Frame, Instances]]) -> list[list[int]]:
    """Run the tracker on (frame, instances) pairs; ids per frame."""
    out = []
    for f, inst in frames:
        result = t(f, inst)
        result.validate(f.width, f.height)
        np.testing.assert_array_equal(result.xyxy, inst.xyxy)  # same order
        out.append(result.track_id.tolist())
    return out


def test_two_jittered_objects_keep_two_ids(make_instances):
    rng = np.random.default_rng(0)
    a, b = [100, 100, 300, 250], [900, 500, 1100, 700]
    frames = [
        (frame(k), make_instances([jittered(rng, a), jittered(rng, b)]))
        for k in range(60)
    ]
    ids = track_ids(tracker(), frames)
    assert ids == [[1, 2]] * 60


@pytest.mark.parametrize(("absent", "same_id"), [(10, True), (20, False)])
def test_occlusion_shorter_than_max_age_keeps_the_id(make_instances, absent, same_id):
    # 10 frames missing = 0.37 s between observations (< max_age_s 0.5 s);
    # 20 frames = 0.70 s: the track died, the object gets a new id.
    box = [[500, 400, 700, 600]]
    frames = [(frame(k), make_instances(box)) for k in range(10)]
    frames += [(frame(k), Instances.empty()) for k in range(10, 10 + absent)]
    frames += [(frame(k), make_instances(box)) for k in range(10 + absent, 30 + absent)]
    ids = track_ids(tracker(), frames)
    assert all(i == [] for i in ids[10 : 10 + absent])  # empty frames: no ids
    assert {i[0] for i in ids if i} == ({1} if same_id else {1, 2})


def test_gap_of_exactly_max_age_keeps_the_id(make_instances):
    # Seen at frame 16, back at 31: 15 frames = max_age_s 0.5 s, but
    # 31/30 - 16/30 is 0.5000000000000002 in floating point. The limit is
    # inclusive whatever the frame index.
    box = [[500, 400, 700, 600]]
    frames = [(frame(k), make_instances(box)) for k in range(17)]
    frames += [(frame(k), Instances.empty()) for k in range(17, 31)]
    frames += [(frame(k), make_instances(box)) for k in range(31, 35)]
    assert (31 / 30 - 16 / 30) > 0.5
    assert {i[0] for i in track_ids(tracker(), frames) if i} == {1}


@pytest.mark.parametrize(("class_aware", "n_ids"), [(False, 1), (True, 2)])
def test_label_flips_break_tracks_only_when_class_aware(
    make_instances, class_aware, n_ids
):
    box = [[500, 400, 700, 600]]
    frames = [
        (frame(k), make_instances(box, labels=["car" if k // 3 % 2 else "truck"]))
        for k in range(30)
    ]
    ids = track_ids(tracker(class_aware=class_aware), frames)
    assert len({i[0] for i in ids}) == n_ids


def test_concat_copies_keep_their_own_ids_and_fused_ones_share(make_instances):
    rng = np.random.default_rng(1)
    box = [400, 300, 600, 500]
    concat = []
    fused = []
    members = (Provenance("a", "car", 0.9), Provenance("b", "car", 0.8))
    for k in range(60):
        xa, xb = jittered(rng, box), jittered(rng, box)
        part_a = make_instances([xa], source="a")
        part_b = make_instances([xb], source="b")
        # Concat order alternates, so ids cannot follow the position.
        parts = [part_a, part_b] if k % 2 else [part_b, part_a]
        concat.append((frame(k), Instances.concat(parts)))
        # Fused: one instance whose representative alternates between detectors.
        fused.append(
            (frame(k), make_instances([xa], source="ab"[k % 2], sources=(members,)))
        )
    t = tracker()
    by_source: dict[str, set[int]] = {"a": set(), "b": set()}
    for f, inst in concat:
        out = t(f, inst)
        for source, tid in zip(out.source, out.track_id.tolist(), strict=True):
            by_source[source].add(tid)
    assert by_source == {"a": {2}, "b": {1}}  # b came first in frame 0
    assert {i[0] for i in track_ids(tracker(), fused)} == {1}


def _accelerating(stride: int, make_instances) -> list[tuple[Frame, Instances]]:
    # 160 px wide box, speed 6 -> 26 px/frame over 80 frames, sampled every
    # `stride` frames: at the end it jumps 130 px between samples (IoU 0.10
    # with the last box, 0.67 at the start).
    speed = 6 + 0.25 * np.arange(81)
    x = np.concatenate(([100.0], 100.0 + np.cumsum(speed)))
    return [
        (frame(k), make_instances([[x[k], 400, x[k] + 160, 560]]))
        for k in range(0, 81, stride)
    ]


def test_constant_velocity_survives_a_stride(make_instances):
    # task_0006 Fact 5: without a motion model objects fragment at stride >= 3.
    frames = _accelerating(5, make_instances)
    assert {i[0] for i in track_ids(tracker(), frames)} == {1}
    assert len({i[0] for i in track_ids(tracker(motion="none"), frames)}) > 1


def test_prediction_uses_seconds_not_frames(make_instances):
    # Same box positions at 30 and 10 fps (stride 3): the predicted box at the
    # next frame is the same, because velocity is px/s and dt comes from Frame.t.
    t30, t10 = tracker(), tracker()
    for k, x in enumerate([100.0, 130.0]):
        inst = make_instances([[x, 400, x + 160, 560]])
        t30(Frame(k, k / 30, RGB), inst)
        t10(Frame(k, k / 10, RGB), inst)
    np.testing.assert_allclose(t30._predict(2 / 30), [[160, 400, 320, 560]])
    np.testing.assert_allclose(t10._predict(2 / 10), [[160, 400, 320, 560]])


def test_different_sources_never_match(make_instances):
    box = [[500, 400, 700, 600]]
    t = tracker()
    assert t(frame(0), make_instances(box, source="a")).track_id.tolist() == [1]
    assert t(frame(1), make_instances(box, source="b")).track_id.tolist() == [2]


def test_greedy_match_is_one_to_one(make_instances):
    # Two detections overlapping the same track: one keeps the id, the other opens one.
    t = tracker()
    t(frame(0), make_instances([[500, 400, 700, 600]]))
    out = t(frame(1), make_instances([[505, 400, 705, 600], [500, 400, 700, 600]]))
    assert out.track_id.tolist() == [2, 1]  # the exact box (highest IoU) wins


def test_reset_load_close_and_determinism(make_instances):
    rng = np.random.default_rng(2)
    frames = [
        (
            frame(k),
            make_instances(
                [
                    jittered(rng, [100, 100, 300, 250]),
                    jittered(rng, [900, 500, 1100, 700]),
                ]
            ),
        )
        for k in range(20)
    ]
    t = tracker()
    first = track_ids(t, frames)
    t.reset()
    assert track_ids(t, frames) == first  # ids restart at 1
    t.close()
    t.load(CTX)
    assert track_ids(t, frames) == first
    assert track_ids(tracker(), frames) == first  # deterministic


def test_describe_is_json_ready():
    desc = tracker(iou_thr=0.4).describe()
    json.dumps(desc)
    assert (desc["type"], desc["name"]) == ("iou", "iou")
    assert desc["options"] == {
        "iou_thr": 0.4,
        "max_age_s": 0.5,
        "motion": "constant_velocity",
        "velocity_alpha": 0.5,
        "class_aware": False,
    }


@pytest.mark.parametrize(
    ("spec", "match"),
    [
        ({"iou_thr": 0.0}, r"^tracking: iou_thr must be in \(0, 1\)"),
        ({"max_age_s": 0}, r"^tracking: max_age_s must be > 0"),
        ({"velocity_alpha": 1.5}, r"^tracking: velocity_alpha must be in \(0, 1\]"),
        ({"motion": "linear"}, r"^tracking\.motion: 'linear' is not one of"),
        ({"max_age": 1}, r"^tracking\.max_age: unknown option"),
    ],
)
def test_options_are_validated_by_the_registry(spec, match):
    with pytest.raises(ConfigError, match=match):
        registry.build("tracker", {"type": "iou", **spec}, path="tracking")


def test_association_keys(make_instances):
    plain = make_instances([[0, 0, 1, 1]] * 2, labels=["car", "truck"], source="a")
    assert association_keys(plain, class_aware=False) == ["a", "a"]
    assert association_keys(plain, class_aware=True) == ["a|car", "a|truck"]
    members = ((Provenance("a", "car", 0.9),), (Provenance("b", "car", 0.9),))
    fused = make_instances([[0, 0, 1, 1]] * 2, sources=members)
    assert association_keys(fused, class_aware=False) == ["*", "*"]
    assert association_keys(fused, class_aware=True) == ["*|car", "*|car"]


# TrackSummary ----------------------------------------------------------------

GEOMETRY = MapGeometry(W, H, W, H, 1.0, 1.0)
DEPTH = DepthResult(
    np.zeros((1, 1), np.float32), CameraIntrinsics(1.0, 1.0, 0.5, 0.5), None, GEOMETRY
)


def _stats(distance: float | None) -> BoxDepthStats:
    nan = math.nan
    return BoxDepthStats(
        (0, 0, 1, 1), 1, 1, 1, 1.0, nan, nan, 0.1, nan, nan, nan, nan, nan, nan, nan,
        distance,
    )  # fmt: skip


def _result(make_instances, k, tracks):
    """Frame k at 10 fps with one instance per (track id, label, distance, estimate)."""
    if tracks is None:  # a frame processed without track ids
        return FrameResult(Frame(k, k / 10, RGB), Instances.empty(), (), DEPTH, {})
    ids, labels, distances, estimates = zip(*tracks, strict=True)
    inst = make_instances(
        [[0, 0, 10, 10]] * len(ids),
        labels=labels,
        track_id=np.asarray(ids, dtype=np.int64),
    )
    return FrameResult(
        Frame(k, k / 10, RGB),
        inst,
        tuple(_stats(d) for d in distances),
        DEPTH,
        {},
        temporal=estimates,
    )


def test_track_summary(make_instances):
    # Track 1: frames 0-4 and 7-14 (one gap), truck at frame 2 and 10-14, no
    # distance at frame 3, smoothed estimate with -1 m/s. Track 2: frames 0-3
    # (0.3 s, short). Frame 15 has no track ids; it only counts as processed.
    def distance(k):
        return None if k == 3 else 20.0 - 0.1 * k + (0.05 if k % 2 else 0.0)

    seen = [*range(5), *range(7, 15)]
    summary = TrackSummary(min_duration_s=1.0)
    for k in range(16):
        tracks = []
        if k in seen:
            label = "truck" if k == 2 or k >= 10 else "car"
            tracks.append((1, label, distance(k), TrackEstimate(20 - k, 0, -1, 0, k)))
        if k < 4:
            tracks.append((2, "person", 5.0, None))
        summary.add(_result(make_instances, k, tracks or None))
    out = summary.summary()
    json.dumps(out)
    assert (out["n_tracks"], out["short_tracks"], out["min_duration_s"]) == (2, 1, 1.0)
    (track,) = out["per_track"]
    valid = [distance(k) for k in seen if k != 3]
    raw_diffs = [distance(k + 1) - distance(k) for k in (0, 1, 7, 8, 9, 10, 11, 12, 13)]
    expected = {
        "id": 1,
        "label": "car",
        "labels": {"car": 7, "truck": 6},
        "label_changes": 3,
        "sources": ["det"],
        "first_frame": 0,
        "last_frame": 14,
        "first_t": 0.0,
        "duration_s": pytest.approx(1.4),
        "n_frames": 13,
        "coverage": pytest.approx(13 / 15),
        "gaps": 1,
        "max_gap_s": pytest.approx(0.3),
        "distance_m": {
            "min": pytest.approx(min(valid)),
            "median": pytest.approx(float(np.median(valid))),
            "max": pytest.approx(max(valid)),
        },
        "jitter_raw_m": pytest.approx(float(np.std(raw_diffs))),
        "jitter_smooth_m": pytest.approx(0.0),  # smoothed: exactly -1 per frame
        "rel_speed_mps": {"p5": -1.0, "median": -1.0, "p95": -1.0},
    }
    assert track == expected


def test_track_summary_without_estimates_or_ids(make_instances):
    summary = TrackSummary(min_duration_s=0.0)
    for k in range(3):
        summary.add(_result(make_instances, k, [(4, "car", None, None)]))
    summary.add(_result(make_instances, 3, None))
    (track,) = summary.summary()["per_track"]
    assert track["distance_m"] is None and track["rel_speed_mps"] is None
    assert track["jitter_raw_m"] is None and track["jitter_smooth_m"] is None
    assert (track["gaps"], track["max_gap_s"], track["coverage"]) == (0, 0.0, 1.0)
    assert TrackSummary().summary() == {
        "n_tracks": 0,
        "short_tracks": 0,
        "min_duration_s": 1.0,
        "per_track": [],
    }
