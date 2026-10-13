from det_depth_distance.core.aggregate import Concat


def test_concat_preserves_order_and_source(make_instances):
    a = make_instances([[0, 0, 1, 1]], source="a", labels=["car"])
    b = make_instances([[0, 0, 2, 2], [0, 0, 3, 3]], source="b", labels=["suv", "van"])
    out = Concat(Concat.Options())([a, b])
    assert out.source == ("a", "b", "b")
    assert out.label == ("car", "suv", "van")
    assert Concat(Concat.Options()).describe() == {"type": "concat", "options": {}}


def test_bind_sources_is_a_noop():
    concat = Concat(Concat.Options())
    assert concat.bind_sources(["a", "b"]) is None
