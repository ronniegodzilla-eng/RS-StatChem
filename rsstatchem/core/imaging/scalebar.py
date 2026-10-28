"""Analyze > Tools > Scale Bar..., Calibration Bar... and Grid... (geometry, overlay ROIs and
burned-in rendering; no Qt).

The scale bar length is in calibrated units (default: about 1/10 of the image width rounded
to 1, 2 or 5 x 10^n like ImageJ), with thickness, font size, colour, background, location
(Upper Right, Lower Right, Lower Left, Upper Left, At Selection), bold, hide text, serif font,
overlay or burn-in, label all slices. The calibration bar shows the LUT with value labels;
the grid draws lines, crosses, points or circles with a given area per point.

Written after ImageJ 1.54 ij/plugin/ScaleBar.java, CalibrationBar.java and Grid.java (public
domain, NIH).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from rsstatchem.core.imaging.roi import shapes as S

LOCATIONS = ("Upper Right", "Lower Right", "Lower Left", "Upper Left", "At Selection")
COLORS = {"White": (255, 255, 255), "Black": (0, 0, 0), "Light Gray": (192, 192, 192),
          "Gray": (128, 128, 128), "Dark Gray": (64, 64, 64), "Red": (255, 0, 0),
          "Green": (0, 255, 0), "Blue": (0, 0, 255), "Yellow": (255, 255, 0),
          "Cyan": (0, 255, 255), "Magenta": (255, 0, 255), "Orange": (255, 200, 0),
          "Pink": (255, 175, 175), "None": None}


def hex_color(name):
    c = COLORS.get(name, None) if not str(name).startswith("#") else name
    if c is None:
        return None
    if isinstance(c, str):
        return c
    return "#%02x%02x%02x" % c


def default_bar_width(image) -> float:
    """ImageJ's initial scale bar width: ~1/10 of the image in calibrated units, rounded."""
    cal = image.calibration
    width = image.width * cal.pixel_width / 10.0
    if width <= 0:
        return 1.0
    mag = 10 ** math.floor(math.log10(width))
    for m in (1, 2, 5, 10):
        if width <= m * mag * 1.5:
            return float(m * mag)
    return float(10 * mag)


@dataclass
class ScaleBarSpec:
    width: float = 0.0             # calibrated units (0 = default)
    height: int = 4                # thickness in pixels
    font_size: int = 14
    color: str = "White"
    background: str = "None"
    location: str = "Lower Right"
    bold: bool = False
    hide: bool = False             # hide text
    serif: bool = False
    overlay: bool = True
    label_all: bool = False
    vertical: bool = False


def scale_bar_geometry(image, spec: ScaleBarSpec, roi=None):
    """(bar rect (x, y, w, h), text position (x, y baseline), label text)."""
    cal = image.calibration
    bw_units = spec.width or default_bar_width(image)
    bar_px = int(round(bw_units / cal.pixel_width)) if cal.pixel_width else int(bw_units)
    bar_px = max(bar_px, 1)
    thick = max(int(spec.height), 1)
    unit = cal.units if cal.scaled() else "pixels"
    num = f"{bw_units:g}"
    label = f"{num} {unit}" if not spec.hide else ""
    margin = max(4, image.width // 40)
    text_h = 0 if spec.hide else spec.font_size
    total_h = thick + (text_h + 2 if text_h else 0)
    W, H = image.width, image.height
    loc = spec.location
    if loc == "At Selection" and roi is not None:
        x, y = int(roi.bounds()[0]), int(roi.bounds()[1])
    elif loc == "Upper Left":
        x, y = margin, margin
    elif loc == "Upper Right":
        x, y = W - margin - bar_px, margin
    elif loc == "Lower Left":
        x, y = margin, H - margin - total_h
    else:
        x, y = W - margin - bar_px, H - margin - total_h
    tx = x + bar_px // 2
    ty = y + thick + text_h + 1
    return (x, y, bar_px, thick), (tx, ty), label


def scale_bar_rois(image, spec: ScaleBarSpec, roi=None) -> list:
    (x, y, w, h), (tx, ty), label = scale_bar_geometry(image, spec, roi)
    color = hex_color(spec.color) or "#ffffff"
    out = []
    bg = hex_color(spec.background)
    if bg:
        pad = 3
        th = spec.font_size + 2 if label else 0
        out.append(S.RectRoi(x - pad, y - pad, w + 2 * pad, h + th + 2 * pad, fill_color=bg,
                             name="|SB| background"))
    bar = S.RectRoi(x, y, w, h, fill_color=color, stroke_color=color, name="|SB| bar")
    out.append(bar)
    if label:
        tw = len(label) * spec.font_size * 0.6
        out.append(S.TextRoi(tx - tw / 2, ty - spec.font_size, label,
                             font="Serif" if spec.serif else "SansSerif",
                             font_size=spec.font_size, stroke_color=color, name="|SB| text"))
    return out


def burn_text(rgb_or_gray: np.ndarray, text: str, x: int, y_baseline: int, size: int, color,
              bold: bool = False, serif: bool = False, center: bool = True) -> None:
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.load_default(size=size)
    except TypeError:  # older Pillow
        font = ImageFont.load_default()
    arr = rgb_or_gray
    rgb = arr.ndim == 3
    img = Image.fromarray(arr if rgb else arr.astype(np.uint8))
    d = ImageDraw.Draw(img)
    bbox = d.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    px = x - tw // 2 if center else x
    py = y_baseline - th - bbox[1]
    fill = tuple(color) if rgb else int(sum(color) / 3)
    d.text((px, py), text, fill=fill, font=font, stroke_width=1 if bold else 0,
           stroke_fill=fill if bold else None)
    arr[...] = np.asarray(img)


def burn_scale_bar(plane: np.ndarray, image, spec: ScaleBarSpec, roi=None) -> None:
    (x, y, w, h), (tx, ty), label = scale_bar_geometry(image, spec, roi)
    color = COLORS.get(spec.color) or (255, 255, 255)
    bg = COLORS.get(spec.background)
    rgb = plane.ndim == 3
    val = np.array(color, np.uint8) if rgb else int(round(sum(color) / 3))
    if bg is not None:
        pad = 3
        th = spec.font_size + 2 if label else 0
        bval = np.array(bg, np.uint8) if rgb else int(round(sum(bg) / 3))
        plane[max(y - pad, 0):y + h + th + pad, max(x - pad, 0):x + w + pad] = bval
    plane[max(y, 0):y + h, max(x, 0):x + w] = val
    if label:
        burn_text(plane, label, tx, ty, spec.font_size, color, spec.bold, spec.serif)


# ------------------------------------------------------------------------------ calibration bar
@dataclass
class CalibrationBarSpec:
    location: str = "Upper Right"
    fill: str = "White"
    label_color: str = "Black"
    n_labels: int = 5
    decimals: int = 0
    font_size: int = 12
    zoom: float = 1.0
    bold: bool = False
    overlay: bool = False
    show_unit: bool = False


def calibration_bar_image(image, spec: CalibrationBarSpec) -> np.ndarray:
    """RGB copy of the displayed image with the LUT bar and value labels burned in."""
    from rsstatchem.core.imaging.lut import render

    rgb = render(image).copy()
    lut = image.lut.as_array()
    vmin, vmax = image.display_range
    cal = image.calibration
    z = spec.zoom
    bar_w, bar_h = int(12 * z), int(128 * z)
    tw = int((spec.font_size * 0.6) * (6 + spec.decimals) * z)
    box_w, box_h = bar_w + tw + int(16 * z), bar_h + int(spec.font_size * z) + int(12 * z)
    W, H = image.width, image.height
    loc = spec.location
    x0 = W - box_w - 5 if "Right" in loc else 5
    y0 = 5 if "Upper" in loc else H - box_h - 5
    if loc == "At Selection" and image.roi is not None:
        x0, y0 = int(image.roi.bounds()[0]), int(image.roi.bounds()[1])
    fill = COLORS.get(spec.fill)
    if fill is not None:
        rgb[max(y0, 0):y0 + box_h, max(x0, 0):x0 + box_w] = fill
    bx, by = x0 + int(6 * z), y0 + int(spec.font_size * z / 2) + int(4 * z)
    for i in range(bar_h):
        idx = int(255 - i * 255 / max(bar_h - 1, 1))
        rgb[by + i:by + i + 1, bx:bx + bar_w] = lut[idx]
    col = COLORS.get(spec.label_color) or (0, 0, 0)
    n = max(spec.n_labels, 0)
    for k in range(n):
        frac = k / (n - 1) if n > 1 else 0.0
        raw = vmax - frac * (vmax - vmin)
        v = cal.cvalue(raw) if cal.calibrated() else raw
        text = f"{v:.{spec.decimals}f}"
        ty = by + int(frac * (bar_h - 1)) + int(spec.font_size * z / 2)
        burn_text(rgb, text, bx + bar_w + int(4 * z), ty, int(spec.font_size * z), col,
                  spec.bold, center=False)
    return rgb


def calibration_bar_rois(image, spec: CalibrationBarSpec) -> list:
    """Overlay version: filled rectangles per LUT step plus text labels."""
    lut = image.lut.as_array()
    vmin, vmax = image.display_range
    z = spec.zoom
    bar_w, bar_h = int(12 * z), int(128 * z)
    W = image.width
    x0 = W - bar_w - int(60 * z) if "Right" in spec.location else 5
    y0 = 5 if "Upper" in spec.location else image.height - bar_h - 25
    out = []
    steps = min(64, bar_h)
    seg = bar_h / steps
    for i in range(steps):
        idx = int(255 - i * 255 / max(steps - 1, 1))
        c = "#%02x%02x%02x" % tuple(int(v) for v in lut[idx])
        out.append(S.RectRoi(x0, y0 + i * seg, bar_w, math.ceil(seg), fill_color=c,
                             name="|CB|"))
    n = max(spec.n_labels, 0)
    col = hex_color(spec.label_color) or "#000000"
    for k in range(n):
        frac = k / (n - 1) if n > 1 else 0.0
        v = vmax - frac * (vmax - vmin)
        out.append(S.TextRoi(x0 + bar_w + 4, y0 + frac * bar_h - spec.font_size / 2,
                             f"{v:.{spec.decimals}f}", font_size=spec.font_size,
                             stroke_color=col, name="|CB|"))
    return out


# ------------------------------------------------------------------------------ grid
GRID_TYPES = ("Lines", "Dotted Lines", "Crosses", "Points", "Circles", "None")


def grid_rois(image, grid_type: str = "Lines", area_per_point: float = 0.0,
              color: str = "Cyan", center: bool = False, random_offset: bool = False,
              seed: int | None = None) -> list:
    """Grid overlay ROIs (``ij.plugin.Grid``); area per point in calibrated units^2."""
    if grid_type == "None":
        return []
    cal = image.calibration
    W, H = image.width, image.height
    if area_per_point <= 0:
        area_per_point = (W * H / 81.0) * cal.pixel_width * cal.pixel_height
    tile = math.sqrt(area_per_point / (cal.pixel_width * cal.pixel_height))
    tile = max(tile, 2.0)
    rng = np.random.default_rng(seed)
    xoff = rng.uniform(0, tile) if random_offset else (tile / 2 if center else tile / 2)
    yoff = rng.uniform(0, tile) if random_offset else (tile / 2 if center else tile / 2)
    xs = np.arange(xoff, W, tile)
    ys = np.arange(yoff, H, tile)
    c = hex_color(color) or "#00ffff"
    out = []
    if grid_type in ("Lines", "Dotted Lines"):
        for x in xs:
            out.append(S.LineRoi(x, 0, x, H, stroke_color=c, name="grid"))
        for y in ys:
            out.append(S.LineRoi(0, y, W, y, stroke_color=c, name="grid"))
    elif grid_type == "Crosses":
        arm = tile / 10
        for x in xs:
            for y in ys:
                out.append(S.LineRoi(x - arm, y, x + arm, y, stroke_color=c, name="grid"))
                out.append(S.LineRoi(x, y - arm, x, y + arm, stroke_color=c, name="grid"))
    elif grid_type == "Points":
        gx, gy = np.meshgrid(xs, ys)
        out.append(S.PointRoi(gx.ravel(), gy.ravel(), stroke_color=c, name="grid"))
    elif grid_type == "Circles":
        r = tile / 10
        for x in xs:
            for y in ys:
                out.append(S.OvalRoi(x - r, y - r, 2 * r, 2 * r, stroke_color=c, name="grid"))
    return out
