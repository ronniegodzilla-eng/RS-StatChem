"""ImageJ's built-in lookup tables (Image > Lookup Tables) with their exact 256-entry tables.

Ported from ImageJ 1.54 ij/plugin/LutLoader.java (public domain, NIH; W. Rasband): Fire and Ice
are 32-entry tables expanded with LutLoader.interpolate (truncating), Spectrum is
``Color.getHSBColor(i/255f, 1f, 1f)``, 3-3-2 RGB and Red/Green are bit formulas, the primary
colours are ramps. Importing this module registers every table in
:data:`rsstatchem.core.imaging.lut.NAMED_LUTS` (so ``named_lut("Fire")`` works); a few common Fiji
tables made from published colour maps ("16 colors", "Thermal"-like "mpl-*") are added as
extras and are not part of the ImageJ parity set.
"""
from __future__ import annotations

import numpy as np

from rsstatchem.core.imaging.lut import LUT, register_lut

_FIRE = (
    [0, 0, 1, 25, 49, 73, 98, 122, 146, 162, 173, 184, 195, 207, 217, 229, 240, 252, 255, 255, 255,
     255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 35, 57, 79, 101, 117, 133, 147, 161, 175, 190, 205,
     219, 234, 248, 255, 255, 255, 255],
    [0, 61, 96, 130, 165, 192, 220, 227, 210, 181, 151, 122, 93, 64, 35, 5, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 35, 98, 160, 223, 255])
_ICE = (
    [0, 0, 0, 0, 0, 0, 19, 29, 50, 48, 79, 112, 134, 158, 186, 201, 217, 229, 242, 250, 250, 250,
     250, 251, 250, 250, 250, 250, 251, 251, 243, 230],
    [156, 165, 176, 184, 190, 196, 193, 184, 171, 162, 146, 125, 107, 93, 81, 87, 92, 97, 95, 93,
     93, 90, 85, 69, 64, 54, 47, 35, 19, 0, 4, 0],
    [140, 147, 158, 166, 170, 176, 209, 220, 234, 225, 236, 246, 250, 251, 250, 250, 245, 230, 230,
     222, 202, 180, 163, 142, 123, 114, 106, 94, 84, 64, 26, 27])

# Image > Lookup Tables, in menu order (labels as in ImageJ)
IMAGEJ_LUTS = ("Fire", "Grays", "Ice", "Spectrum", "3-3-2 RGB", "Red", "Green", "Blue", "Cyan",
               "Magenta", "Yellow", "Red/Green")


def interpolate(table: list[list[int]], n_colors: int) -> np.ndarray:
    """LutLoader.interpolate: expand ``n_colors`` entries to 256 (truncating, as ImageJ)."""
    src = np.asarray(table, dtype=np.float64)[:, :n_colors]
    scale = n_colors / 256.0
    i = np.arange(256)
    i1 = (i * scale).astype(np.int64)
    i2 = np.minimum(i1 + 1, n_colors - 1)
    fraction = i * scale - i1
    out = (1.0 - fraction) * src[:, i1] + fraction * src[:, i2]
    return out.astype(np.int64).astype(np.uint8)        # (byte)(double) truncates


def _from_rows(rgb: np.ndarray, name: str) -> LUT:
    return LUT(rgb[0], rgb[1], rgb[2], name)


def fire() -> LUT:
    return _from_rows(interpolate(_FIRE, 32), "Fire")


def ice() -> LUT:
    return _from_rows(interpolate(_ICE, 32), "Ice")


def grays() -> LUT:
    return LUT.grays()


def java_hsb_to_rgb(hue, saturation, brightness):
    """java.awt.Color.HSBtoRGB in float32 arithmetic (vectorised); returns (r, g, b) int arrays."""
    f32 = np.float32
    hue = np.asarray(hue, dtype=f32)
    s = np.broadcast_to(np.asarray(saturation, dtype=f32), hue.shape)
    b = np.broadcast_to(np.asarray(brightness, dtype=f32), hue.shape)
    with np.errstate(invalid="ignore", over="ignore"):
        h = (hue - np.floor(hue).astype(f32)) * f32(6.0)
        fr = h - np.floor(h).astype(f32)
        p = b * (f32(1.0) - s)
        q = b * (f32(1.0) - s * fr)
        t = b * (f32(1.0) - (s * (f32(1.0) - fr)))

        def to_int(v):
            return np.nan_to_num(v * f32(255.0) + f32(0.5)).astype(np.int64)

        bi, pi, qi, ti = to_int(b), to_int(p), to_int(q), to_int(t)
    sector = np.nan_to_num(h).astype(np.int64)
    r = np.select([sector == 0, sector == 1, sector == 2, sector == 3, sector == 4, sector == 5],
                  [bi, qi, pi, pi, ti, bi], 0)
    g = np.select([sector == 0, sector == 1, sector == 2, sector == 3, sector == 4, sector == 5],
                  [ti, bi, bi, qi, pi, pi], 0)
    bb = np.select([sector == 0, sector == 1, sector == 2, sector == 3, sector == 4, sector == 5],
                   [pi, pi, ti, bi, bi, qi], 0)
    gray = s == 0
    r = np.where(gray, bi, r)
    g = np.where(gray, bi, g)
    bb = np.where(gray, bi, bb)
    return r & 0xFF, g & 0xFF, bb & 0xFF


def spectrum() -> LUT:
    hue = np.arange(256, dtype=np.float32) / np.float32(255.0)
    r, g, b = java_hsb_to_rgb(hue, 1.0, 1.0)
    return LUT(r, g, b, "Spectrum")


def rgb332() -> LUT:
    i = np.arange(256)
    return LUT(i & 0xE0, (i << 3) & 0xE0, (i << 6) & 0xC0, "3-3-2 RGB")


def primary(color: int, name: str) -> LUT:
    """LutLoader.primaryColor: bit 4 = red, 2 = green, 1 = blue."""
    ramp = np.arange(256)
    zero = np.zeros(256, dtype=int)
    return LUT(ramp if color & 4 else zero, ramp if color & 2 else zero,
               ramp if color & 1 else zero, name)


def red_green() -> LUT:
    i = np.arange(256)
    r = np.where(i < 128, (i * 2) & 0xFF, 0)
    g = np.where(i < 128, 0, (i * 2) & 0xFF)
    return LUT(r, g, np.zeros(256, dtype=int), "Red/Green")


def _sixteen_colors() -> LUT:
    """Fiji's "16 colors" table (published palette), expanded to 256 entries."""
    base = [(0, 0, 0), (1, 1, 171), (1, 1, 224), (0, 110, 255), (0, 171, 255), (1, 224, 255),
            (1, 255, 255), (1, 255, 171), (1, 255, 0), (171, 255, 1), (224, 255, 0),
            (255, 255, 1), (255, 192, 0), (255, 111, 1), (255, 0, 0), (255, 255, 255)]
    rgb = np.repeat(np.asarray(base, dtype=np.uint8), 16, axis=0)
    return LUT.from_array(rgb, "16 colors")


def _matplotlib(name: str, label: str):
    def factory() -> LUT:
        from matplotlib import colormaps
        cmap = colormaps[name]
        rgb = np.round(cmap(np.linspace(0.0, 1.0, 256))[:, :3] * 255.0).astype(np.uint8)
        return LUT.from_array(rgb, label)
    return factory


BUILTIN = {
    "Fire": fire,
    "Grays": grays,
    "Ice": ice,
    "Spectrum": spectrum,
    "3-3-2 RGB": rgb332,
    "Red": lambda: primary(4, "Red"),
    "Green": lambda: primary(2, "Green"),
    "Blue": lambda: primary(1, "Blue"),
    "Cyan": lambda: primary(3, "Cyan"),
    "Magenta": lambda: primary(5, "Magenta"),
    "Yellow": lambda: primary(6, "Yellow"),
    "Red/Green": red_green,
}
EXTRAS = {
    "16 colors": _sixteen_colors,
    "mpl-viridis": _matplotlib("viridis", "mpl-viridis"),
    "mpl-magma": _matplotlib("magma", "mpl-magma"),
    "mpl-inferno": _matplotlib("inferno", "mpl-inferno"),
    "mpl-plasma": _matplotlib("plasma", "mpl-plasma"),
}


def register_all() -> None:
    for name, factory in {**BUILTIN, **EXTRAS}.items():
        register_lut(name, factory)


register_all()
