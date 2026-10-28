"""Lookup tables and display rendering (ImageJ ``IndexColorModel`` / ``LUT`` semantics).

:class:`LUT` holds 256 RGB entries. :data:`NAMED_LUTS` maps names to factories; Phase 0 provides
"Grays" and the primary colours; WP-D adds Fire, Ice, Spectrum, 3-3-2 RGB, ... from
``core/imaging/lut_tables.py`` with :func:`register_lut` (its own file, so this one stays fixed).

:func:`to_8bit` maps a raw plane to 0..255 display indices exactly as ImageJ's
``create8BitImage`` / ``setMinAndMax`` do for 8/16/32-bit data; :func:`render` turns a whole
:class:`~rsstatchem.core.imaging.image.ImageDoc` position into an RGB uint8 array (composite
channels blended by sum, as ImageJ's ``CompositeImage`` in COMPOSITE mode).
Ported in part from ImageJ 1.54 ij/process/{ByteProcessor,ShortProcessor,FloatProcessor}.java
(public domain, NIH).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(eq=False)
class LUT:
    red: np.ndarray      # uint8[256]
    green: np.ndarray
    blue: np.ndarray
    name: str = ""

    def __post_init__(self) -> None:
        for attr in ("red", "green", "blue"):
            arr = np.asarray(getattr(self, attr), dtype=np.uint8).reshape(256)
            setattr(self, attr, arr)

    @classmethod
    def grays(cls) -> "LUT":
        ramp = np.arange(256, dtype=np.uint8)
        return cls(ramp, ramp.copy(), ramp.copy(), "Grays")

    @classmethod
    def from_color(cls, r: int, g: int, b: int, name: str = "") -> "LUT":
        """Black-to-colour ramp (ImageJ ``LUT.createLutFromColor``)."""
        ramp = np.arange(256, dtype=float) / 255.0
        return cls(np.round(ramp * r).astype(np.uint8), np.round(ramp * g).astype(np.uint8),
                   np.round(ramp * b).astype(np.uint8), name)

    @classmethod
    def from_array(cls, rgb: np.ndarray, name: str = "") -> "LUT":
        """From an (256, 3) array (e.g. a GIF palette, padded to 256 entries)."""
        rgb = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3)
        if len(rgb) < 256:
            rgb = np.vstack([rgb, np.zeros((256 - len(rgb), 3), np.uint8)])
        return cls(rgb[:256, 0], rgb[:256, 1], rgb[:256, 2], name)

    def as_array(self) -> np.ndarray:
        """(256, 3) uint8 table."""
        return np.stack([self.red, self.green, self.blue], axis=1)

    def inverted(self) -> "LUT":
        return LUT(self.red[::-1].copy(), self.green[::-1].copy(), self.blue[::-1].copy(),
                   self.name)

    def is_inverted(self) -> bool:
        """ImageJ ``isInvertedLut``: first entry white, last entry black (grayscale-like)."""
        return bool(self.red[0] == 255 and self.green[0] == 255 and self.blue[0] == 255 and
                    self.red[255] == 0 and self.green[255] == 0 and self.blue[255] == 0)

    def is_grayscale(self) -> bool:
        ramp = np.arange(256)
        same = np.array_equal(self.red, self.green) and np.array_equal(self.red, self.blue)
        return bool(same and (np.array_equal(self.red, ramp) or
                              np.array_equal(self.red, ramp[::-1])))

    def copy(self) -> "LUT":
        return LUT(self.red.copy(), self.green.copy(), self.blue.copy(), self.name)

    def __eq__(self, other) -> bool:
        return isinstance(other, LUT) and np.array_equal(self.as_array(), other.as_array())


NAMED_LUTS: dict[str, Callable[[], LUT]] = {
    "Grays": LUT.grays,
    "Red": lambda: LUT.from_color(255, 0, 0, "Red"),
    "Green": lambda: LUT.from_color(0, 255, 0, "Green"),
    "Blue": lambda: LUT.from_color(0, 0, 255, "Blue"),
    "Cyan": lambda: LUT.from_color(0, 255, 255, "Cyan"),
    "Magenta": lambda: LUT.from_color(255, 0, 255, "Magenta"),
    "Yellow": lambda: LUT.from_color(255, 255, 0, "Yellow"),
}
# ImageJ's default channel colours of a composite image (CompositeImage.setup)
CHANNEL_COLORS = ("Red", "Green", "Blue", "Grays", "Cyan", "Magenta", "Yellow")


def register_lut(name: str, factory: Callable[[], LUT]) -> None:
    NAMED_LUTS[name] = factory


def named_lut(name: str) -> LUT:
    return NAMED_LUTS[name]()


def to_8bit(plane: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Display indices (uint8) of a 2-D raw plane for the display range ``vmin..vmax``."""
    a = np.asarray(plane)
    if a.dtype == np.uint8:
        lo, hi = int(vmin), int(vmax)
        if lo == 0 and hi == 255:
            return a
        i = np.arange(256, dtype=np.int64)
        span = max(hi - lo, 1)
        idx = np.clip((256 * (i - lo)) // span, 0, 255)  # ByteProcessor.setMinAndMax
        idx = np.where(i < lo, 0, np.where(i > hi, 255, idx))
        return idx.astype(np.uint8)[a]
    if a.dtype == np.uint16:
        scale = 256.0 / (vmax - vmin + 1)  # ShortProcessor.create8BitImage
        v = np.maximum(a.astype(np.float64) - vmin, 0)
        return np.minimum((v * scale + 0.5).astype(np.int64), 255).astype(np.uint8)
    f = a.astype(np.float32)
    span = (vmax - vmin) if vmax > vmin else 1.0
    scale = np.float32(255.0 / span)  # FloatProcessor.create8BitImage
    v = np.maximum(f - np.float32(vmin), np.float32(0))
    v = np.nan_to_num(v * scale + np.float32(0.5), nan=0.0, posinf=255.0)
    return np.minimum(v.astype(np.int64), 255).astype(np.uint8)


def apply_lut(indices: np.ndarray, lut: LUT) -> np.ndarray:
    """RGB uint8 (H, W, 3) of display indices through ``lut``."""
    return lut.as_array()[indices]


def render(doc, c: int | None = None, z: int | None = None, t: int | None = None) -> np.ndarray:
    """RGB uint8 (H, W, 3) of the image at position (c, z, t) as ImageJ displays it."""
    pc, pz, pt = doc.position
    c = pc if c is None else c
    z = pz if z is None else z
    t = pt if t is None else t
    if doc.is_rgb:
        rgb = doc.plane(0, z, t)
        lo, hi = doc.display_ranges[0]
        if (lo, hi) == (0, 255):
            return np.ascontiguousarray(rgb)
        return np.stack([to_8bit(rgb[..., k], lo, hi) for k in range(3)], axis=-1)
    if doc.n_channels > 1 and doc.composite_mode == "composite":
        acc = np.zeros((doc.height, doc.width, 3), dtype=np.int32)
        for ch in range(doc.n_channels):
            if not doc.active_channels[ch]:
                continue
            lo, hi = doc.display_ranges[ch]
            acc += apply_lut(to_8bit(doc.plane(ch, z, t), lo, hi), doc.luts[ch])
        return np.minimum(acc, 255).astype(np.uint8)
    lo, hi = doc.display_ranges[c]
    lut = doc.luts[c] if doc.composite_mode != "grayscale" else LUT.grays()
    return apply_lut(to_8bit(doc.plane(c, z, t), lo, hi), lut)
