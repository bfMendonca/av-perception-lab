"""TwinLiteNet+ adapter: drivable area and lane lines (BDD100K, MIT).

The network is imported from a local clone (`repo`), which is not an
installable package: its code imports `model.config`, so the clone is put on
sys.path only while importing (the top-level name `model` then stays in
sys.modules; another clone in the same process is refused). Pre/post-
processing is ported from the repo's demo: letterbox to `imgsz` (multiple
of 32), /255, then the padding is cropped and the two 2-channel heads are
resized to the exact frame size (the demo's integer scale factor only works
when frame/input is an integer ratio). Frames are already RGB, the order the
demo feeds after its BGR -> RGB flip. Labels: 0 background, 1 drivable,
2 lane (lane wins). torch is imported inside the hooks only.

Location: `repo` defaults to the pinned clone `<weights root>/_code/
TwinLiteNetPlus` (declared as a `CodeRepo`, cloned by `--prepare`); any other
clone (e.g. `~/dev/TwinLiteNetPlus`) is an explicit override and is not
declared when it is outside the weights root. The pretrained files are not in
the git repository (upstream serves them from Google Drive, not a v1 source),
so `pretrained/<variant>.pth` is declared `UserProvided` with the download
hint.
"""

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import cv2
import numpy as np

from det_depth_distance.adapters._weights import INFER_EXTRA, under_root
from det_depth_distance.core.components import LoadContext, SemanticSegmenter
from det_depth_distance.core.options import ConfigError
from det_depth_distance.core.registry import register
from det_depth_distance.core.requirements import (
    CODE_DIR,
    CodeRepo,
    Requirements,
    UserProvided,
    weights_root,
)
from det_depth_distance.core.types import SemanticMap

TWINLITENET_CLASSES: tuple[str, ...] = ("background", "drivable", "lane")
_PAD_VALUE = (114, 114, 114)  # demoDataset.letterbox_for_img default

TWINLITENET_REPO = CodeRepo(
    name="TwinLiteNetPlus",
    url="https://github.com/chequanghuy/TwinLiteNetPlus",
    commit="90f1b8695ae311d5123b05f8534b2e11e42499d2",
    known=True,  # fixed here, cloned to its fixed folder
)
# Google Drive downloads of the upstream README (line 87 of the clone): the
# folder of every variant, and the file id of `small`.
_DRIVE_FOLDER = (
    "https://drive.google.com/drive/folders/1EqBzUw0b17aEumZmWYrGZmbx_XJqU-vz"
)
_SMALL_URL = (
    "https://drive.usercontent.google.com/download?"
    "id=1N7jNsa8P1dM-UevqPRsx6heUpN4L00FJ&export=download&confirm=t"
)


def default_repo() -> Path:
    """The pinned clone under the weights root."""
    return weights_root() / CODE_DIR / TWINLITENET_REPO.name


def _weights_hint(variant: str, path: Path) -> str:
    if variant == "small":
        return f'manual download (Google Drive): curl -fL -o {path} "{_SMALL_URL}"'
    return (
        f"manual download (Google Drive, not fetched by --prepare): get "
        f"{variant}.pth from {_DRIVE_FOLDER} and save it as {path}"
    )


@register("semantic", "twinlitenet")
class TwinLiteNetSegmenter(SemanticSegmenter):
    """TwinLiteNetPlus (nano/small/medium/large) from a local clone."""

    @dataclass(frozen=True, slots=True)
    class Options:
        # None = the pinned clone <weights root>/_code/TwinLiteNetPlus; "~" is
        # expanded here too (from_mapping expands it only in YAML values).
        repo: Path | None = None
        variant: Literal["nano", "small", "medium", "large"] = "large"
        weights: Path | None = None  # default: <repo>/pretrained/<variant>.pth
        imgsz: int = 640  # longest letterbox side, multiple of 32
        half: bool = False  # FP16 inference (the demo's default)

        def __post_init__(self) -> None:
            if self.imgsz < 32 or self.imgsz % 32:
                raise ValueError(
                    f"imgsz must be a positive multiple of 32, got {self.imgsz}"
                )

    def __init__(self, options: Options, *, name: str | None = None) -> None:
        super().__init__(options, name=name)
        self._net = None
        self._device = ""

    @classmethod
    def requirements(cls, options: Options) -> Requirements:
        """The pinned clone (default repo only) and the pretrained file, each
        declared when it lies inside the weights root."""
        repo = _repo(options)
        weights = _weights(options, repo)
        code = (TWINLITENET_REPO,) if options.repo is None else ()
        target = under_root(weights)
        assets = (
            ()
            if target is None
            else (
                UserProvided(
                    key="weights",
                    target=target,
                    option="weights",
                    hint=_weights_hint(options.variant, weights),
                ),
            )
        )
        return Requirements(
            assets=assets, code=code, packages=("torch",), extra=INFER_EXTRA
        )

    def default_name(self) -> str:
        """ "twinlitenet-<variant>" (programmatic use: configs require a name)."""
        return f"twinlitenet-{self.options.variant}"

    def _weights(self, repo: Path) -> Path:
        return _weights(self.options, repo)

    def _load(self, ctx: LoadContext) -> None:
        repo = _repo(self.options)
        if not (repo / "model" / "model.py").is_file():
            hint = (
                "; run `det-depth-distance --config <yaml> --prepare`"
                if self.options.repo is None
                else ""
            )
            raise ConfigError(f"repo: no TwinLiteNetPlus clone at {repo}{hint}")
        weights = self._weights(repo)
        if not weights.is_file():
            raise ConfigError(
                f"weights: file not found: {weights}; "
                f"{_weights_hint(self.options.variant, weights)}"
            )
        import torch

        net = _import_network(repo)(SimpleNamespace(config=self.options.variant))
        # A plain state_dict (loads with torch 2.8's weights_only default).
        net.load_state_dict(torch.load(weights, map_location=ctx.device))
        net = net.to(ctx.device).eval()
        self._net = net.half() if self.options.half else net
        self._device = ctx.device

    @property
    def class_names(self) -> tuple[str, ...]:
        return TWINLITENET_CLASSES

    def _segment(self, rgb: np.ndarray) -> SemanticMap:
        import torch
        import torch.nn.functional as F

        h, w = rgb.shape[:2]
        image, scale, (left, top) = _letterbox(rgb, self.options.imgsz)
        new_h, new_w = round(h * scale), round(w * scale)
        x = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))[None]
        x = x.to(self._device)
        x = (x.half() if self.options.half else x.float()) / 255.0
        heads = []
        with torch.inference_mode():
            for out in self._net(x):  # (drivable, lane), each (1, 2, H_in, W_in)
                crop = out[:, :, top : top + new_h, left : left + new_w]
                # Exact frame size (the demo's int(1/ratio) scale is exact only
                # for integer ratios, e.g. 1920 -> 640).
                logits = F.interpolate(crop, size=(h, w), mode="bilinear")
                heads.append(torch.max(logits, 1)[1][0].cpu().numpy())
        return SemanticMap(_merge_heads(*heads), TWINLITENET_CLASSES)

    def _close(self) -> None:
        import torch

        self._net = None
        torch.cuda.empty_cache()

    def _describe(self) -> dict[str, Any]:
        return {"weights_file": str(self._weights(_repo(self.options)))}


def _repo(options: Any) -> Path:
    return default_repo() if options.repo is None else options.repo.expanduser()


def _weights(options: Any, repo: Path) -> Path:
    """`weights`, default `<repo>/pretrained/<variant>.pth`."""
    if options.weights is None:
        return repo / "pretrained" / f"{options.variant}.pth"
    return options.weights.expanduser()


def _letterbox(
    rgb: np.ndarray, imgsz: int
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """demoDataset.letterbox_for_img(img, imgsz, auto=True): resize the longest
    side to `imgsz` (INTER_AREA), then pad each side to a multiple of 32.

    Returns:
        (padded image, scale, (pad_left, pad_top)).
    """
    h, w = rgb.shape[:2]
    scale = min(imgsz / h, imgsz / w)
    new_w, new_h = round(w * scale), round(h * scale)
    dw, dh = (imgsz - new_w) % 32 / 2, (imgsz - new_h) % 32 / 2
    image = rgb
    if (new_w, new_h) != (w, h):
        image = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=_PAD_VALUE
    )
    return image, scale, (left, top)


def _merge_heads(drivable: np.ndarray, lane: np.ndarray) -> np.ndarray:
    """(H, W) int32 labels from the two binary heads: 0 background, 1 drivable,
    2 lane (a lane pixel wins over drivable area)."""
    labels = (drivable == 1).astype(np.int32)
    labels[lane == 1] = 2
    return labels


def _import_network(repo: Path) -> type:
    """TwinLiteNetPlus class of the clone at `repo` (its top-level package
    `model` is importable only with the clone on sys.path).

    Raises:
        ConfigError: a different top-level module `model` is already imported.
    """
    package = (repo / "model").resolve()
    loaded = sys.modules.get("model")
    if loaded is not None and package not in {
        Path(p).resolve() for p in getattr(loaded, "__path__", [])
    }:
        raise ConfigError(
            f"repo: another top-level module 'model' is already imported "
            f"({getattr(loaded, '__file__', None) or list(getattr(loaded, '__path__', []))}); "
            f"cannot import TwinLiteNetPlus from {repo}"
        )
    sys.path.insert(0, str(repo))
    try:
        module = importlib.import_module("model.model")
    finally:
        sys.path.remove(str(repo))
    return module.TwinLiteNetPlus
