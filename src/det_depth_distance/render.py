"""Annotated-frame rendering: translucent semantic layers, translucent instance
masks, boxes, label pills, HUD and an optional side-by-side metric depth panel
with a legend in metres.

Ported from rf-detr's infer_video.py Renderer and depth-anything-3's
run_test.py colorize_metric/metric_legend. Everything is RGB; OpenCV
colormaps (BGR) are converted on output. The depth panel is resized to the
exact frame size so the same box and mask coordinates apply to both panels.
"""

import math
import zlib
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# One distinct, readable colour per driving class of the COCO and Objects365
# presets (RGB); any other key (class or detector name) gets a stable PALETTE colour.
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "person": (255, 99, 132),
    "bicycle": (255, 159, 64),
    "car": (54, 162, 235),
    "vehicle": (54, 162, 235),  # canonical label of the driving label maps
    "suv": (90, 130, 255),
    "van": (120, 200, 255),
    "sports car": (0, 200, 230),
    "motorcycle": (255, 205, 86),
    "tricycle": (240, 230, 140),
    "bus": (153, 102, 255),
    "truck": (75, 192, 192),
    "pickup truck": (60, 220, 170),
    "machinery vehicle": (190, 160, 110),
    "fire truck": (255, 120, 90),
    "ambulance": (250, 250, 250),
    "traffic light": (46, 204, 113),
    "traffic sign": (255, 140, 0),
    "stop sign": (231, 76, 60),
    "crosswalk sign": (240, 100, 200),
    "traffic cone": (255, 120, 20),
    "street lights": (180, 180, 255),
    "fire hydrant": (220, 60, 110),
}
PALETTE: list[tuple[int, int, int]] = [
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 212),
    (0, 128, 128),
    (220, 190, 255),
]

# Semantic class never drawn (e.g. TwinLiteNet+'s label 0).
SEMANTIC_BACKGROUND: str = "background"

FONT_CANDIDATES: list[str] = [
    "/usr/share/fonts/truetype/inter/Inter-SemiBold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-M.ttf",
]


def color_for(key: str) -> tuple[int, int, int]:
    """Colour of a class or detector name: CLASS_COLORS, else a stable PALETTE entry.

    crc32 (not hash()) keeps colours identical across processes and runs.
    """
    return (
        CLASS_COLORS.get(key.lower())
        or PALETTE[zlib.crc32(key.encode()) % len(PALETTE)]
    )


def load_font(
    path: Path | None, size: int
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """`path` if given, else the first available system font, else PIL's default."""
    for candidate in ([path] if path else []) + [Path(p) for p in FONT_CANDIDATES]:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default(size)


def colorize_metric(depth_m: np.ndarray, max_depth: float, cmap: int) -> np.ndarray:
    """Linear metres on a fixed 0..max_depth scale (near = warm) -> RGB uint8.

    Beyond max_depth (and NaN) gets the far colour.
    """
    norm = 1.0 - np.clip(np.nan_to_num(depth_m, nan=max_depth) / max_depth, 0, 1)
    bgr = cv2.applyColorMap((norm * 255).astype(np.uint8), cmap)
    return np.ascontiguousarray(bgr[..., ::-1])


def metric_legend(height: int, max_depth: float, cmap: int) -> np.ndarray:
    """Vertical colour bar (max_depth at the top, 0 m at the bottom) with ticks in metres.

    RGB uint8, even width; sizes scale with `height` (the original assumed 540 px).
    """
    scale = height / 540
    width = max(2, round(110 * scale) // 2 * 2)
    legend = np.full((height, width, 3), 30, np.uint8)
    top, bottom = round(50 * scale), height - round(20 * scale)
    x0, x1 = round(12 * scale), round(40 * scale)
    ramp = np.linspace(max_depth, 0, bottom - top)[:, None].repeat(x1 - x0, 1)
    legend[top:bottom, x0:x1] = colorize_metric(ramp, max_depth, cmap)
    font_scale, thickness = 0.5 * scale, max(1, round(scale))
    # At most ~8 ticks; beyond the table (max_depth > 800 m) fall back to an even split.
    step = next(
        (s for s in (0.5, 1, 2, 2.5, 5, 10, 20, 25, 50, 100) if max_depth / s <= 8),
        max_depth / 8,
    )
    for value in np.arange(0, max_depth + 1e-6, step):
        y = round(bottom - 1 - value / max_depth * (bottom - top - 1))
        cv2.line(
            legend, (x1, y), (x1 + round(6 * scale), y), (255, 255, 255), thickness
        )
        text = f"{value:g}" + ("+" if value >= max_depth else "")
        cv2.putText(
            legend,
            text,
            (x1 + round(10 * scale), y + round(5 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    cv2.putText(
        legend,
        "metres",
        (round(8 * scale), round(30 * scale)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55 * scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return legend


def format_label(
    class_name: str,
    score: float,
    distance: float | None,
    spread: float | None = None,
    n_sources: int = 1,
    *,
    track_id: int | None = None,
    rel_speed: float | None = None,
) -> str:
    """Box label, e.g. "#12 truck 94% · 12.3 m ± 0.4 · -1.2 m/s": optional track
    id prefix, agreeing detectors (×N) after the score, spread after the
    distance and relative speed (negative = approaching) at the end, only when
    finite. Without a distance: "truck 94% · n/a"."""
    head = f"{class_name} {score:.0%}" + (f" ×{n_sources}" if n_sources > 1 else "")
    if track_id is not None:
        head = f"#{track_id} {head}"
    if distance is None:
        text = f"{head} · n/a"
    else:
        text = f"{head} · {distance:.1f} m"
        if spread is not None:
            text += f" ± {spread:.1f}"
    if rel_speed is not None and math.isfinite(rel_speed):
        text += f" · {rel_speed:+.1f} m/s"
    return text


@dataclass(frozen=True, slots=True)
class BoxAnnotation:
    """What to draw for one box."""

    xyxy: tuple[float, float, float, float]  # frame pixels
    class_name: str
    score: float  # drawing order (strongest on top)
    label: str
    # sampled (shrunk) region, frame pixels
    inner_xyxy: tuple[float, float, float, float] | None = None
    # colour and HUD-count key (e.g. the detector name); None -> class_name
    color_key: str | None = None
    # (H, W) bool instance mask at frame resolution, tinted with the box colour
    mask: np.ndarray | None = None

    @property
    def key(self) -> str:
        return self.color_key or self.class_name


class Renderer:
    """Draws boxes, label pills and a translucent HUD with PIL (anti-aliased text,
    alpha blending), optionally next to a colourised depth panel and legend."""

    def __init__(
        self,
        frame_w: int,
        frame_h: int,
        *,
        title: str,
        layout: Literal["overlay", "side-by-side"] = "overlay",
        show_hud: bool = True,
        max_depth: float = 30.0,
        cmap: str = "turbo",
        font_path: Path | None = None,
        mask_alpha: float = 0.35,
        semantic_alpha: float = 0.4,
    ) -> None:
        if layout not in ("overlay", "side-by-side"):
            raise ValueError(f"unknown layout {layout!r}")
        if not 0.0 <= mask_alpha <= 1.0:
            raise ValueError(f"mask_alpha must be in [0, 1], got {mask_alpha}")
        if not 0.0 <= semantic_alpha <= 1.0:
            raise ValueError(f"semantic_alpha must be in [0, 1], got {semantic_alpha}")
        try:
            self._cmap = getattr(cv2, f"COLORMAP_{cmap.upper()}")
        except AttributeError:
            raise ValueError(f"unknown OpenCV colormap {cmap!r}") from None
        self.frame_w, self.frame_h = frame_w, frame_h
        self.layout = layout
        self.max_depth = max_depth
        self.title = title
        self.show_hud = show_hud
        self.mask_alpha = mask_alpha
        self.semantic_alpha = semantic_alpha
        self._luts: dict[tuple[str, ...], np.ndarray] = {}  # by class_names
        scale = frame_h / 1080
        self.box_width = max(2, round(3 * scale))
        self.font = load_font(font_path, max(14, round(22 * scale)))
        self.hud_title_font = load_font(font_path, max(16, round(26 * scale)))
        self.pad = max(4, round(7 * scale))
        self.radius = max(4, round(8 * scale))
        self._legend = (
            metric_legend(frame_h, max_depth, self._cmap)
            if layout == "side-by-side"
            else None
        )

    def output_size(self) -> tuple[int, int]:
        """(width, height) of the rendered frames."""
        if self._legend is None:
            return self.frame_w, self.frame_h
        return 2 * self.frame_w + self._legend.shape[1], self.frame_h

    def render(
        self,
        frame_rgb: np.ndarray,
        boxes: Sequence[BoxAnnotation],
        hud_lines: Sequence[str],
        depth_m: np.ndarray | None = None,
        *,
        semantic: Sequence[tuple[np.ndarray, Sequence[str]]] = (),
    ) -> np.ndarray:
        """Annotated frame of size `output_size()`; `depth_m` is required for side-by-side.

        `semantic` holds layers (labels (H, W), class_names) in drawing order (the
        last one on top), drawn on the frame panel only, under masks and boxes.
        """
        image = frame_rgb
        if self.semantic_alpha > 0:
            for labels, class_names in semantic:
                image = self._blend_semantic(image, labels, tuple(class_names))
        left = self._draw(image, boxes, hud_lines if self.show_hud else None)
        if self._legend is None:
            return left
        if depth_m is None:
            raise ValueError("side-by-side layout needs the depth map")
        panel = cv2.resize(
            colorize_metric(depth_m, self.max_depth, self._cmap),
            (self.frame_w, self.frame_h),
            interpolation=cv2.INTER_LINEAR,
        )
        return np.hstack([left, self._draw(panel, boxes, None), self._legend])

    def _draw(
        self,
        image_rgb: np.ndarray,
        boxes: Sequence[BoxAnnotation],
        hud_lines: Sequence[str] | None,
    ) -> np.ndarray:
        base = Image.fromarray(self._blend_masks(image_rgb, boxes)).convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        # low-confidence boxes first so the strongest end up on top
        for box in sorted(boxes, key=lambda b: b.score):
            x1, y1, x2, y2 = box.xyxy
            color = color_for(box.key)
            d.rounded_rectangle(
                [x1, y1, x2, y2],
                radius=self.radius // 2,
                outline=color + (255,),
                width=self.box_width,
                fill=color + (28,),
            )
            if box.inner_xyxy is not None:
                d.rectangle(list(box.inner_xyxy), outline=(255, 255, 255, 200), width=1)
            tw, th = d.textbbox((0, 0), box.label, font=self.font)[2:]
            label_h = th + 2 * self.pad
            ly = y1 - label_h - 2 if y1 - label_h - 2 > 0 else y1 + 2
            lx = min(max(0, x1), base.width - tw - 2 * self.pad)
            d.rounded_rectangle(
                [lx, ly, lx + tw + 2 * self.pad, ly + label_h],
                radius=self.radius,
                fill=color + (225,),
            )
            d.text(
                (lx + self.pad, ly + self.pad - 1),
                box.label,
                font=self.font,
                fill=(15, 15, 20, 255),
            )
        if hud_lines is not None:
            self._hud(d, hud_lines, Counter(b.key for b in boxes))
        return np.asarray(Image.alpha_composite(base, overlay).convert("RGB"))

    def _blend_masks(
        self, image_rgb: np.ndarray, boxes: Sequence[BoxAnnotation]
    ) -> np.ndarray:
        """Tint every non-empty instance mask with its box colour at `mask_alpha`,
        weakest first so the strongest instance ends on top. Works on the mask's
        bounding rectangle only; used for the frame and the depth panel alike."""
        out = image_rgb
        if self.mask_alpha == 0:
            return out
        for box in sorted(boxes, key=lambda b: b.score):
            if box.mask is None:
                continue
            rows = np.flatnonzero(box.mask.any(axis=1))
            if not rows.size:
                continue
            cols = np.flatnonzero(box.mask.any(axis=0))
            if out is image_rgb:  # copy once, only when something is drawn
                out = image_rgb.copy()
            window = out[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
            inside = box.mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
            tint = self.mask_alpha * np.asarray(color_for(box.key), np.float32)
            window[inside] = np.rint(
                (1 - self.mask_alpha) * window[inside] + tint
            ).astype(np.uint8)
        return out

    def _semantic_lut(self, class_names: tuple[str, ...]) -> np.ndarray:
        """(C, 4) uint8 lookup table: RGB colour of each class (color_for) and its
        alpha (semantic_alpha * 255; 0 for SEMANTIC_BACKGROUND), cached per tuple."""
        lut = self._luts.get(class_names)
        if lut is None:
            alpha = round(self.semantic_alpha * 255)
            lut = np.array(
                [
                    (*color_for(name), 0 if name == SEMANTIC_BACKGROUND else alpha)
                    for name in class_names
                ],
                dtype=np.uint8,
            ).reshape(-1, 4)
            self._luts[class_names] = lut
        return lut

    def _blend_semantic(
        self, image_rgb: np.ndarray, labels: np.ndarray, class_names: tuple[str, ...]
    ) -> np.ndarray:
        """Tint every non-background pixel with its class colour (color_for) at
        `semantic_alpha`, using a per-class lookup table. Drawn first, under the
        instance masks and boxes, on the frame panel only. With several maps the
        layers are applied in order, each over the previous one; colours follow
        the class name, so a class shared by two maps keeps one colour."""
        rgba = self._semantic_lut(class_names)[labels]
        drawn = rgba[..., 3] > 0
        if not drawn.any():
            return image_rgb
        out = image_rgb.copy()
        alpha = rgba[drawn, 3:].astype(np.float32) / 255
        out[drawn] = np.rint((1 - alpha) * out[drawn] + alpha * rgba[drawn, :3]).astype(
            np.uint8
        )
        return out

    def _hud(
        self, d: ImageDraw.ImageDraw, lines: Sequence[str], counts: Counter
    ) -> None:
        m = p = 2 * self.pad
        title_h = d.textbbox((0, 0), self.title, font=self.hud_title_font)[3]
        line_h = d.textbbox((0, 0), "Ag", font=self.font)[3] + self.pad
        count_items = sorted(counts.items(), key=lambda kv: -kv[1])
        width = (
            max(
                [d.textbbox((0, 0), s, font=self.font)[2] for s in lines]
                + [d.textbbox((0, 0), self.title, font=self.hud_title_font)[2]]
                + [
                    d.textbbox((0, 0), f"{n}  {c}", font=self.font)[2] + 3 * self.pad
                    for n, c in count_items
                ]
            )
            + 2 * p
        )
        height = (
            p
            + title_h
            + self.pad * 2
            + line_h * (len(lines) + len(count_items))
            + (self.pad if count_items else 0)
            + p
        )
        d.rounded_rectangle(
            [m, m, m + width, m + height],
            radius=2 * self.radius,
            fill=(12, 14, 20, 170),
        )
        y = m + p
        d.text(
            (m + p, y), self.title, font=self.hud_title_font, fill=(255, 255, 255, 255)
        )
        y += title_h + self.pad * 2
        for s in lines:
            d.text((m + p, y), s, font=self.font, fill=(210, 215, 225, 255))
            y += line_h
        if count_items:
            y += self.pad
            dot = line_h // 3
            for n, c in count_items:
                color = color_for(n)
                cy = y + line_h // 2 - self.pad // 2
                d.ellipse(
                    [m + p, cy - dot // 2, m + p + dot, cy + dot // 2],
                    fill=color + (255,),
                )
                d.text(
                    (m + p + dot + self.pad * 2, y),
                    f"{n}  {c}",
                    font=self.font,
                    fill=(255, 255, 255, 255),
                )
                y += line_h
