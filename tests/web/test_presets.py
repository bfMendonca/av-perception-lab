"""Preset storage: built-in manifest, user presets, slugs and traversal."""

from pathlib import Path

import pytest
import yaml

from det_depth_distance.web import presets
from det_depth_distance.web.presets import (
    PresetExists,
    PresetInvalid,
    PresetNotFound,
    PresetReadOnly,
    delete_preset,
    list_presets,
    read_preset,
    save_preset,
    slugify,
)

REPO_CONFIGS = Path(__file__).resolve().parents[2] / "configs"

PIPELINE = {
    "device": "cuda",
    "detectors": [{"name": "rfdetr"}],
    "depth": {"name": "da3"},
    "source": {"video": "a.mp4"},
    "output": {"out_root": "runs"},
}


@pytest.fixture
def builtin(tmp_path: Path) -> Path:
    d = tmp_path / "configs"
    d.mkdir()
    for stem in ("alpha", "beta"):
        (d / f"{stem}.yaml").write_text(yaml.safe_dump(PIPELINE))
    manifest = {
        "presets": [
            {"file": "beta.yaml", "name": "Beta One", "description": "second file"},
            {"file": "alpha.yaml", "name": "Alpha Two", "description": "first file"},
        ]
    }
    (d / "presets.yaml").write_text(yaml.safe_dump(manifest))
    return d


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / "user"


def test_slugify_cases():
    assert slugify("RF-DETR + DA3") == "rf-detr-da3"
    assert slugify("Minha Câmera 4K") == "minha-camera-4k"
    for bad in ("!!!", "   ", "", "a\nb", "x" * 81):
        with pytest.raises(PresetInvalid):
            slugify(bad)


def test_list_builtin_follows_manifest_order_and_names(builtin, root):
    infos = list_presets(builtin, root)
    assert [(i.id, i.name, i.description, i.kind) for i in infos] == [
        ("builtin:beta", "Beta One", "second file", "builtin"),
        ("builtin:alpha", "Alpha Two", "first file", "builtin"),
    ]


def test_list_without_manifest_falls_back_to_stems(builtin, root):
    (builtin / "presets.yaml").unlink()
    assert [i.id for i in list_presets(builtin, root)] == [
        "builtin:alpha",
        "builtin:beta",
    ]
    (builtin / "presets.yaml").write_text("presets: [1, 2")  # malformed
    ids = [i.id for i in list_presets(builtin, root)]
    assert ids == ["builtin:alpha", "builtin:beta"]


def test_list_ignores_manifest_entry_whose_file_is_missing(builtin, root):
    (builtin / "beta.yaml").unlink()
    assert [i.id for i in list_presets(builtin, root)] == ["builtin:alpha"]


def test_read_builtin_strips_source_and_output(builtin, root):
    data = read_preset("builtin:alpha", builtin, root)
    assert set(data) == {"device", "detectors", "depth"}
    with pytest.raises(PresetNotFound):
        read_preset("builtin:presets", builtin, root)
    with pytest.raises(PresetNotFound):
        read_preset("builtin:nope", builtin, root)


def test_save_then_list_and_read_roundtrip(builtin, root):
    info = save_preset(root, builtin, "Minha Câmera 4K", PIPELINE, overwrite=False)
    assert info.id == "user:minha-camera-4k"
    assert info.kind == "user"
    assert info.name == "Minha Câmera 4K"
    path = root / "minha-camera-4k.yaml"
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# name: Minha Câmera 4K\n")
    saved = yaml.safe_load(text)
    assert "source" not in saved and "output" not in saved
    users = [i for i in list_presets(builtin, root) if i.kind == "user"]
    assert users == [info]
    assert read_preset(info.id, builtin, root) == {
        "device": "cuda",
        "detectors": [{"name": "rfdetr"}],
        "depth": {"name": "da3"},
    }


def test_user_presets_sorted_by_name_after_builtins(builtin, root):
    save_preset(root, builtin, "zeta", PIPELINE, overwrite=False)
    save_preset(root, builtin, "Mu", PIPELINE, overwrite=False)
    infos = list_presets(builtin, root)
    assert [i.kind for i in infos] == ["builtin", "builtin", "user", "user"]
    assert [i.name for i in infos[2:]] == ["Mu", "zeta"]


def test_save_creates_missing_root(builtin, tmp_path):
    deep = tmp_path / "a" / "b"
    save_preset(deep, builtin, "x", PIPELINE, overwrite=False)
    assert (deep / "x.yaml").is_file()


def test_save_existing_without_overwrite_raises_exists(builtin, root):
    save_preset(root, builtin, "x", PIPELINE, overwrite=False)
    with pytest.raises(PresetExists):
        save_preset(root, builtin, "X", PIPELINE, overwrite=False)


def test_save_with_overwrite_replaces(builtin, root):
    save_preset(root, builtin, "x", PIPELINE, overwrite=False)
    new = {"device": "cpu", "detectors": []}
    save_preset(root, builtin, "X", new, overwrite=True)
    assert read_preset("user:x", builtin, root) == new
    assert (root / "x.yaml").read_text().startswith("# name: X\n")
    assert [p.name for p in root.iterdir()] == ["x.yaml"]


def test_save_rejects_builtin_stem_and_builtin_name_slug(builtin, root):
    with pytest.raises(PresetInvalid):
        save_preset(root, builtin, "alpha", PIPELINE, overwrite=False)
    with pytest.raises(PresetInvalid):
        save_preset(root, builtin, "Alpha Two", PIPELINE, overwrite=True)
    assert not root.exists() or not list(root.iterdir())


def test_invalid_user_file_is_listed_and_read_raises_invalid_without_hiding_others(
    builtin, root
):
    save_preset(root, builtin, "good", PIPELINE, overwrite=False)
    (root / "bad.yaml").write_text("# name: Bad one\n: : [\n")
    (root / "list.yaml").write_text("- 1\n- 2\n")
    infos = list_presets(builtin, root)
    ids = [i.id for i in infos]
    assert {"user:good", "user:bad", "user:list"} <= set(ids)
    assert next(i for i in infos if i.id == "user:bad").name == "Bad one"
    with pytest.raises(PresetInvalid):
        read_preset("user:bad", builtin, root)
    with pytest.raises(PresetInvalid):
        read_preset("user:list", builtin, root)
    assert "detectors" in read_preset("user:good", builtin, root)


def test_ids_with_traversal_are_rejected(builtin, root, tmp_path):
    root.mkdir()
    outside = tmp_path / "outside.yaml"
    outside.write_text("detectors: []\n")
    (root / "evil.yaml").symlink_to(outside)
    for pid in (
        "user:../x",
        "user:/etc/passwd",
        "user:a/b",
        "user:evil",
        "builtin:../x",
    ):
        with pytest.raises((PresetNotFound, PresetInvalid)):
            read_preset(pid, builtin, root)
    with pytest.raises((PresetNotFound, PresetInvalid)):
        delete_preset("user:evil", root)
    assert outside.exists()
    # the listing does not follow the symlink either
    assert "user:evil" not in [i.id for i in list_presets(builtin, root)]


def test_delete_user_removes_file(builtin, root):
    save_preset(root, builtin, "x", PIPELINE, overwrite=False)
    delete_preset("user:x", root)
    assert not (root / "x.yaml").exists()


def test_delete_builtin_raises_read_only(builtin, root):
    with pytest.raises(PresetReadOnly):
        delete_preset("builtin:alpha", root)
    assert (builtin / "alpha.yaml").exists()


def test_delete_missing_raises_not_found(root):
    with pytest.raises(PresetNotFound):
        delete_preset("user:nope", root)
    with pytest.raises(PresetNotFound):
        delete_preset("user:nope", None)


def test_repo_manifest_lists_every_config_and_every_file_exists():
    manifest = yaml.safe_load(
        (REPO_CONFIGS / "presets.yaml").read_text(encoding="utf-8")
    )
    files = {e["file"] for e in manifest["presets"]}
    on_disk = {p.name for p in REPO_CONFIGS.glob("*.yaml")} - {"presets.yaml"}
    assert files == on_disk
    assert presets.BUILTIN_DIR == REPO_CONFIGS
    infos = list_presets(REPO_CONFIGS, None)
    assert len(infos) == len(files)
    for info in infos:
        assert info.name and info.description
        data = read_preset(info.id, REPO_CONFIGS, None)
        assert "detectors" in data
        assert "source" not in data and "output" not in data


@pytest.mark.parametrize("sep", [" ", " "])
def test_slugify_rejects_unicode_line_separators(sep):
    with pytest.raises(PresetInvalid):
        slugify(f"a{sep}b")


def test_delete_racing_unlink_raises_not_found(builtin, root, monkeypatch):
    save_preset(root, builtin, "x", PIPELINE, overwrite=False)
    from det_depth_distance.web import presets as mod

    real = mod._user_path

    def resolve_then_vanish(user_root, slug):
        path = real(user_root, slug)
        path.unlink()  # a concurrent delete wins between resolution and unlink
        return path

    monkeypatch.setattr(mod, "_user_path", resolve_then_vanish)
    with pytest.raises(PresetNotFound):
        delete_preset("user:x", root)


def test_slugify_non_ascii_name_message_says_ascii():
    with pytest.raises(PresetInvalid, match="ASCII letter or digit"):
        slugify("Пресет")
