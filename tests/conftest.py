"""Shared test helpers. The `make_instances` fixture returns a factory that
builds contract-valid Instances (dtypes, raw_label = label, one source) from
boxes and optional fields; request the fixture and call what it returns,
`def test_x(make_instances): inst = make_instances(...)`, so tests of every
task construct instances the same way."""

from collections.abc import Sequence

import numpy as np
import pytest

from det_depth_distance.core.types import Instances


def _make(
    xyxy: Sequence[Sequence[float]],
    *,
    score: Sequence[float] | None = None,
    labels: Sequence[str] | None = None,
    source: str = "det",
    class_id: Sequence[int] | None = None,
    masks: np.ndarray | None = None,
    **extra,
) -> Instances:
    boxes = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
    n = len(boxes)
    labels = tuple(labels) if labels is not None else ("car",) * n
    return Instances(
        boxes,
        np.asarray(score if score is not None else [0.9] * n, dtype=np.float32),
        np.asarray(class_id if class_id is not None else range(n), dtype=np.int64),
        labels,
        labels,
        (source,) * n,
        masks,
        **extra,
    )


@pytest.fixture
def make_instances():
    """Factory of contract-valid Instances (see the module docstring)."""
    return _make
