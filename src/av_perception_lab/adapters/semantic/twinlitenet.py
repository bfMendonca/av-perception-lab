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
the git repository: upstream serves them from a public Google Drive folder, so
`pretrained/<variant>.pth` is a `GoogleDriveAsset` (file id, size and sha256 in
`TWINLITENET_WEIGHTS`, verified in task_0028) that Prepare downloads into the
clone once the clone is ready. It is `known` (web-preparable) only at the
canonical place `<weights root>/_code/TwinLiteNetPlus/pretrained/<variant>.pth`
of the pinned clone; a `<variant>.pth` elsewhere in the root is prepared by the
CLI only, and any other file name in the root is `UserProvided`.
"""

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import cv2
import numpy as np

from av_perception_lab.adapters._weights import INFER_EXTRA, under_root
from av_perception_lab.core.components import (
    ClassCatalog,
    LoadContext,
    SemanticSegmenter,
)
from av_perception_lab.core.options import ConfigError
from av_perception_lab.core.registry import register
from av_perception_lab.core.requirements import (
    CODE_DIR,
    CodeRepo,
    GoogleDriveAsset,
    Requirements,
    UserProvided,
    weights_root,
)
from av_perception_lab.core.types import ContractError, SemanticMap

TWINLITENET_CLASSES: tuple[str, ...] = ("background", "drivable", "lane")
_PAD_VALUE = (114, 114, 114)  # demoDataset.letterbox_for_img default

TWINLITENET_REPO = CodeRepo(
    name="TwinLiteNetPlus",
    url="https://github.com/chequanghuy/TwinLiteNetPlus",
    commit="90f1b8695ae311d5123b05f8534b2e11e42499d2",
    known=True,  # fixed here, cloned to its fixed folder
)
# The upstream README's "Pre-trained Model" link (line 87 of the clone): the
# public Drive folder of every variant (shown for a manual download).
DRIVE_FOLDER = (
    "https://drive.google.com/drive/folders/1EqBzUw0b17aEumZmWYrGZmbx_XJqU-vz"
)
# The repository is MIT (LICENSE of the pinned clone, (c) 2025 Che Quang Huy);
# the Drive files carry no licence of their own: the repository's is assumed.
WEIGHTS_LICENSE = "MIT"
# variant -> (Drive file id, bytes, sha256): ids from the folder listing,
# sizes from Content-Length, sha256 of the downloaded files, equal to the
# author's local copies (task_0028, 2026-11-01).
TWINLITENET_WEIGHTS: dict[str, tuple[str, int, str]] = {
    "nano": (
        "1SvD03WZOq8eN4X0bMFNZyzj4upZM1QVl",
        217_395,
        "4b76d7fe57b86a7f0583f8a07fcd767a60177fc9b054386161e1c3e12a492129",
    ),
    "small": (
        "1N7jNsa8P1dM-UevqPRsx6heUpN4L00FJ",
        594_777,
        "0dd3bd7356a5b70e1d21ae60cd09b55641b48bde738393177f94daff17ab6360",
    ),
    "medium": (
        "121z9XUh7_lgze8i6nS6Ne2HQ7ZG5_uG9",
        2_054_463,
        "04a7959946f687482256d1a499ee4e0ab5d523e9c1760be41e7e797dc65066ff",
    ),
    "large": (
        "1H8P-GrOUBOaVs5LEqXBfz0dC9gguUUio",
        7_959_943,
        "5605d1a8ce762c1b6f7d23f4c2e7ec9384bfd6e6887b040b43e1f8902a817e22",
    ),
}


def default_repo() -> Path:
    """The pinned clone under the weights root."""
    return weights_root() / CODE_DIR / TWINLITENET_REPO.name


def canonical_weights_target(variant: str) -> str:
    """Target (relative to the weights root) of `<variant>.pth` in the
    pinned clone: the only place where the Drive asset is `known`."""
    return f"{CODE_DIR}/{TWINLITENET_REPO.name}/pretrained/{variant}.pth"


def _weights_hint(variant: str, path: Path) -> str:
    return (
        f"run `av-perception-lab --config <yaml> --prepare` (downloads "
        f"{variant}.pth from Google Drive), or get {variant}.pth from "
        f"{DRIVE_FOLDER} and save it as {path}"
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
        declared when it lies inside the weights root: `<variant>.pth` is the
        Google Drive file of `TWINLITENET_WEIGHTS` (known only at its
        canonical place in the pinned clone), any other file is
        `UserProvided`."""
        repo = _repo(options)
        weights = _weights(options, repo)
        code = (TWINLITENET_REPO,) if options.repo is None else ()
        target = under_root(weights)
        assets: tuple = ()
        if target is not None and weights.name == f"{options.variant}.pth":
            file_id, size, sha256 = TWINLITENET_WEIGHTS[options.variant]
            assets = (
                GoogleDriveAsset(
                    key="weights",
                    target=target,
                    file_id=file_id,
                    size_bytes=size,
                    sha256=sha256,
                    license=WEIGHTS_LICENSE,
                    manual=DRIVE_FOLDER,
                    known=options.repo is None
                    and target == canonical_weights_target(options.variant),
                ),
            )
        elif target is not None:
            assets = (
                UserProvided(
                    key="weights",
                    target=target,
                    option="weights",
                    hint=f"provide your TwinLiteNet+ {options.variant} weights at "
                    f"{weights} (or use the published {options.variant}.pth: "
                    f"{DRIVE_FOLDER})",
                ),
            )
        return Requirements(
            assets=assets, code=code, packages=("torch",), extra=INFER_EXTRA
        )

    @classmethod
    def declared_classes(cls, options: Options) -> ClassCatalog | None:
        """The architecture's fixed classes (checked against the head in `_load`)."""
        return ClassCatalog(TWINLITENET_CLASSES, "static")

    def default_name(self) -> str:
        """ "twinlitenet-<variant>" (programmatic use: configs require a name)."""
        return f"twinlitenet-{self.options.variant}"

    def _weights(self, repo: Path) -> Path:
        return _weights(self.options, repo)

    def _load(self, ctx: LoadContext) -> None:
        repo = _repo(self.options)
        if not (repo / "model" / "model.py").is_file():
            hint = (
                "; run `av-perception-lab --config <yaml> --prepare`"
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
        _check_heads(net)
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


def _check_heads(net: Any) -> None:
    """The static class list holds only if each head (`out_da`, `out_ll`) ends
    in a 2-channel conv (background + one class, merged into 3 labels)."""
    for head in ("out_da", "out_ll"):
        convs = [
            m.out_channels
            for m in getattr(net, head).modules()
            if hasattr(m, "out_channels")
        ]
        if not convs or convs[-1] != 2:
            raise ContractError(
                f"TwinLiteNet+ head {head} has {convs[-1] if convs else '?'} output "
                f"channels, expected 2: the declared classes {TWINLITENET_CLASSES} "
                f"(3 classes) do not match this network"
            )
