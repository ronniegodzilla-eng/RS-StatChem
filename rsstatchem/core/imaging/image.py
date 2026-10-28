"""The image model: :class:`ImageDoc` (ImageJ ``ImagePlus`` + ``ImageStack`` + display state).

Pixel layout (fixed contract):

* grayscale and 8-bit colour: ``pixels.shape == (T, Z, C, Y, X)``, dtype ``uint8`` (GRAY8,
  COLOR_256), ``uint16`` (GRAY16) or ``float32`` (GRAY32);
* RGB: ``pixels.shape == (T, Z, C, Y, X, 3)``, dtype ``uint8`` (C is normally 1).

Positions are 0-based ``(c, z, t)``. ImageJ's 1-based stack index in czt order is
``stack_index(c, z, t) + 1``; slice labels are stored per stack index. Display state is per
channel (``display_ranges[c]``, ``luts[c]``); ``threshold`` is the ImageJ threshold of the image
(raw pixel values) or None; ``roi`` the current selection, ``overlay`` the non-destructive
overlay. Operations that change pixels call :meth:`snapshot` first (single-level undo, as in
ImageJ) and set ``changed``.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from rsstatchem.core.imaging.calibration import Calibration
from rsstatchem.core.imaging.lut import CHANNEL_COLORS, LUT, named_lut

# ImagePlus.getType() constants
GRAY8, GRAY16, GRAY32, COLOR_256, COLOR_RGB = 0, 1, 2, 3, 4
TYPE_NAMES = {GRAY8: "8-bit", GRAY16: "16-bit", GRAY32: "32-bit", COLOR_256: "8-bit Color",
              COLOR_RGB: "RGB"}
BIT_DEPTH = {GRAY8: 8, GRAY16: 16, GRAY32: 32, COLOR_256: 8, COLOR_RGB: 24}
DTYPES = {GRAY8: np.uint8, GRAY16: np.uint16, GRAY32: np.float32, COLOR_256: np.uint8,
          COLOR_RGB: np.uint8}
BYTES_PER_PIXEL = {GRAY8: 1, GRAY16: 2, GRAY32: 4, COLOR_256: 1, COLOR_RGB: 4}
COMPOSITE_MODES = ("composite", "color", "grayscale")


class Threshold(NamedTuple):
    """ImageJ threshold in raw pixel values; ``mode`` is how it is shown (Threshold window):
    "red" (RED_LUT), "bw" (BLACK_AND_WHITE_LUT), "over_under" or "none" (NO_LUT_UPDATE)."""
    lower: float
    upper: float
    mode: str = "red"


@dataclass
class _State:
    pixels: np.ndarray
    type: int
    calibration: Calibration
    luts: list
    display_ranges: list
    slice_labels: list
    roi: object
    overlay: object
    threshold: Threshold | None


class ImageDoc:
    """One open image (2-D to 5-D) with everything ImageJ keeps per ``ImagePlus``."""

    def __init__(self, pixels: np.ndarray, type: int | None = None, title: str = "Untitled",
                 calibration: Calibration | None = None, path: str | None = None):
        pixels = np.asarray(pixels)
        rgb = pixels.ndim == 6
        if pixels.ndim not in (5, 6) or (rgb and pixels.shape[-1] != 3):
            raise ValueError("pixels must be TZCYX or TZCYXS (S=3); use ImageDoc.from_array")
        if type is None:
            type = COLOR_RGB if rgb else _type_for_dtype(pixels.dtype)
        if (type == COLOR_RGB) != rgb:
            raise ValueError("RGB images need a trailing axis of 3 samples")
        self.pixels = np.ascontiguousarray(pixels, dtype=DTYPES[type])
        if not self.pixels.flags.writeable:   # e.g. np.asarray(PIL image)
            self.pixels = self.pixels.copy()
        self.type = type
        self.title = title
        self.path = path
        self.calibration = calibration or Calibration()
        nc = self.n_channels
        self.luts: list[LUT] = [LUT.grays() for _ in range(nc)]
        if nc > 1 and type != COLOR_RGB:
            self.luts = [named_lut(CHANNEL_COLORS[i % len(CHANNEL_COLORS)]) for i in range(nc)]
        self.composite_mode = "composite" if nc > 1 else "color"
        self.active_channels = [True] * nc
        self.display_ranges: list[tuple[float, float]] = [(0.0, 255.0)] * nc
        self.slice_labels: list[str | None] = [None] * self.stack_size
        self.properties: dict[str, str] = {}   # ImageJ "image properties" (key -> string)
        self.info: str = ""                     # ImageJ "Info" property (Show Info text)
        self.roi = None                         # rsstatchem.core.imaging.roi.Roi | None
        self.overlay = None                     # rsstatchem.core.imaging.roi.Overlay | None
        self.threshold: Threshold | None = None
        self.position: tuple[int, int, int] = (0, 0, 0)   # (c, z, t), 0-based
        self.changed = False
        self.hyperstack = False                 # opened/declared as a hyperstack
        self.id = _next_id()
        self._undo: _State | None = None
        self.undo_label = ""
        self.reset_display_range()

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_array(cls, array, axes: str = "YX", title: str = "Untitled",
                   type: int | None = None, **kw) -> "ImageDoc":
        """Create from an array with axes drawn from "TZCYXS" in any order (S = RGB samples).

        bool -> 8-bit 0/255; int16 -> 16-bit with signed calibration (raw = v + 32768);
        float64/int32 -> 32-bit; a trailing S axis of 3 uint8 samples -> RGB."""
        a = np.asarray(array)
        axes = axes.upper()
        if len(axes) != a.ndim or set(axes) - set("TZCYXS"):
            raise ValueError(f"axes {axes!r} do not match an array of shape {a.shape}")
        order = [axes.index(ax) for ax in "TZCYXS" if ax in axes]
        a = np.transpose(a, order)
        present = "".join(ax for ax in "TZCYXS" if ax in axes)
        for i, ax in enumerate("TZCYX"):
            if ax not in present:
                a = np.expand_dims(a, i)
        cal = kw.pop("calibration", None) or Calibration()
        if a.dtype == bool:
            a = a.astype(np.uint8) * 255
        elif a.dtype == np.int16:
            a = (a.astype(np.int32) + 32768).astype(np.uint16)
            cal.set_signed16()
        elif a.dtype in (np.float64, np.float16, np.int32, np.uint32, np.int64, np.uint64):
            a = a.astype(np.float32)
        elif a.dtype == np.int8:
            a = a.astype(np.float32)
        if a.ndim == 6 and type is None:
            type = COLOR_RGB
        return cls(a, type=type, title=title, calibration=cal, **kw)

    @classmethod
    def new(cls, width: int, height: int, type: int = GRAY8, channels: int = 1,
            slices: int = 1, frames: int = 1, fill: float = 0, title: str = "Untitled"
            ) -> "ImageDoc":
        shape = (frames, slices, channels, height, width) + ((3,) if type == COLOR_RGB else ())
        return cls(np.full(shape, fill, dtype=DTYPES[type]), type=type, title=title)

    # ------------------------------------------------------------------ geometry
    @property
    def n_frames(self) -> int:
        return self.pixels.shape[0]

    @property
    def n_slices(self) -> int:
        return self.pixels.shape[1]

    @property
    def n_channels(self) -> int:
        return self.pixels.shape[2]

    @property
    def height(self) -> int:
        return self.pixels.shape[3]

    @property
    def width(self) -> int:
        return self.pixels.shape[4]

    @property
    def dims(self) -> tuple[int, int, int]:
        """(channels, slices, frames) like ImageJ ``getNChannels/NSlices/NFrames``."""
        return self.n_channels, self.n_slices, self.n_frames

    @property
    def stack_size(self) -> int:
        return self.n_channels * self.n_slices * self.n_frames

    @property
    def bit_depth(self) -> int:
        return BIT_DEPTH[self.type]

    @property
    def is_rgb(self) -> bool:
        return self.type == COLOR_RGB

    @property
    def is_stack(self) -> bool:
        return self.stack_size > 1

    @property
    def is_composite(self) -> bool:
        return self.n_channels > 1 and not self.is_rgb

    @property
    def type_name(self) -> str:
        return TYPE_NAMES[self.type]

    # ------------------------------------------------------------------ planes
    def stack_index(self, c: int, z: int, t: int) -> int:
        """0-based index in ImageJ stack order (czt)."""
        return c + z * self.n_channels + t * self.n_channels * self.n_slices

    def position_of(self, index: int) -> tuple[int, int, int]:
        """(c, z, t) of a 0-based stack index."""
        nc, nz = self.n_channels, self.n_slices
        return index % nc, (index // nc) % nz, index // (nc * nz)

    def plane(self, c: int | None = None, z: int | None = None, t: int | None = None
              ) -> np.ndarray:
        """View (Y, X) or (Y, X, 3) of one plane; defaults to the current position."""
        pc, pz, pt = self.position
        return self.pixels[pt if t is None else t, pz if z is None else z,
                           pc if c is None else c]

    def set_plane(self, data: np.ndarray, c: int | None = None, z: int | None = None,
                  t: int | None = None) -> None:
        pc, pz, pt = self.position
        self.pixels[pt if t is None else t, pz if z is None else z,
                    pc if c is None else c] = data

    def plane_at(self, index: int) -> np.ndarray:
        return self.plane(*self.position_of(index))

    def iter_planes(self, all_planes: bool = True, channel: int | None = None):
        """Yield (c, z, t) of every plane (or only the current one); ``channel`` restricts
        to one channel (ImageJ processes the current channel of a hyperstack)."""
        if not all_planes:
            yield self.position
            return
        for index in range(self.stack_size):
            c, z, t = self.position_of(index)
            if channel is None or c == channel:
                yield c, z, t

    def set_position(self, c: int = 0, z: int = 0, t: int = 0) -> None:
        self.position = (min(max(c, 0), self.n_channels - 1), min(max(z, 0), self.n_slices - 1),
                         min(max(t, 0), self.n_frames - 1))

    @property
    def current_index(self) -> int:
        return self.stack_index(*self.position)

    # ------------------------------------------------------------------ values & display
    def raw_value(self, x: int, y: int, c=None, z=None, t=None):
        """Raw pixel value (float, or an (r, g, b) tuple for RGB)."""
        v = self.plane(c, z, t)[y, x]
        return tuple(int(k) for k in v) if self.is_rgb else float(v)

    def value(self, x: int, y: int, c=None, z=None, t=None):
        """Calibrated pixel value (RGB returned raw)."""
        v = self.raw_value(x, y, c, z, t)
        return v if self.is_rgb else self.calibration.cvalue(v)

    def plane_min_max(self, c=None, z=None, t=None) -> tuple[float, float]:
        p = self.plane(c, z, t)
        if self.type == GRAY32:
            finite = p[np.isfinite(p)]
            if finite.size == 0:
                return 0.0, 0.0
            return float(finite.min()), float(finite.max())
        return float(p.min()), float(p.max())

    def reset_display_range(self, channel: int | None = None) -> None:
        """ImageJ ``resetDisplayRange``: 0-255 for 8-bit/RGB, plane min/max for 16/32-bit."""
        chans = range(self.n_channels) if channel is None else [channel]
        ranges = list(self.display_ranges)
        for ch in chans:
            if self.bit_depth in (8, 24):
                ranges[ch] = (0.0, 255.0)
            else:
                ranges[ch] = self.plane_min_max(c=ch)
        self.display_ranges = ranges

    def set_display_range(self, vmin: float, vmax: float, channel: int | None = None) -> None:
        ranges = list(self.display_ranges)
        ranges[self.position[0] if channel is None else channel] = (float(vmin), float(vmax))
        self.display_ranges = ranges

    @property
    def display_range(self) -> tuple[float, float]:
        return self.display_ranges[self.position[0]]

    @property
    def lut(self) -> LUT:
        return self.luts[self.position[0]]

    def is_inverted_lut(self) -> bool:
        return not self.is_rgb and self.luts[self.position[0]].is_inverted()

    # ------------------------------------------------------------------ labels
    def slice_label(self, c=None, z=None, t=None) -> str | None:
        pc, pz, pt = self.position
        return self.slice_labels[self.stack_index(pc if c is None else c, pz if z is None else z,
                                                  pt if t is None else t)]

    def set_slice_label(self, label: str | None, index: int | None = None) -> None:
        self.slice_labels[self.current_index if index is None else index] = label

    # ------------------------------------------------------------------ undo
    def _state(self, deep: bool) -> _State:
        cp = copy.deepcopy if deep else (lambda v: v)
        return _State(self.pixels.copy() if deep else self.pixels, self.type,
                      self.calibration.copy(), [lut.copy() for lut in self.luts],
                      list(self.display_ranges), list(self.slice_labels), cp(self.roi),
                      cp(self.overlay), self.threshold)

    def snapshot(self, label: str = "") -> None:
        """Remember the current state for a single-level Edit > Undo."""
        self._undo = self._state(deep=True)
        self.undo_label = label

    def can_undo(self) -> bool:
        return self._undo is not None

    def undo(self) -> bool:
        """Swap with the snapshot (undoing twice redoes, like ImageJ). False if none."""
        if self._undo is None:
            return False
        current = self._state(deep=False)
        s = self._undo
        self.pixels, self.type, self.calibration = s.pixels, s.type, s.calibration
        self.luts, self.display_ranges, self.slice_labels = s.luts, s.display_ranges, \
            s.slice_labels
        self.roi, self.overlay, self.threshold = s.roi, s.overlay, s.threshold
        nc = self.n_channels
        if len(self.active_channels) != nc:
            self.active_channels = [True] * nc
        self.set_position(*self.position)
        self._undo = current
        return True

    def forget_undo(self) -> None:
        self._undo = None

    # ------------------------------------------------------------------ misc
    def duplicate(self, title: str | None = None) -> "ImageDoc":
        """Deep copy without undo history (Image > Duplicate of the whole image)."""
        dup = ImageDoc(self.pixels.copy(), self.type, title or self.title,
                       self.calibration.copy(), None)
        dup.luts = [lut.copy() for lut in self.luts]
        dup.display_ranges = list(self.display_ranges)
        dup.composite_mode = self.composite_mode
        dup.active_channels = list(self.active_channels)
        dup.slice_labels = list(self.slice_labels)
        dup.properties = dict(self.properties)
        dup.info = self.info
        dup.roi = copy.deepcopy(self.roi)
        dup.overlay = copy.deepcopy(self.overlay)
        dup.threshold = self.threshold
        dup.position = self.position
        dup.hyperstack = self.hyperstack
        return dup

    def replace_pixels(self, pixels: np.ndarray, type: int | None = None) -> None:
        """Swap in new pixel data (type conversion, crop, resize, add slice ...); resets
        display ranges and per-plane metadata that no longer fits."""
        old_size, old_nc = self.stack_size, self.n_channels
        tmp = ImageDoc(pixels, type)
        self.pixels, self.type = tmp.pixels, tmp.type
        if self.n_channels != old_nc:
            self.luts = tmp.luts
            self.active_channels = tmp.active_channels
            self.composite_mode = tmp.composite_mode
        if self.stack_size != old_size:
            self.slice_labels = [None] * self.stack_size
        self.display_ranges = [(0.0, 255.0)] * self.n_channels
        self.set_position(*self.position)
        self.reset_display_range()

    def size_bytes(self) -> int:
        return self.width * self.height * self.stack_size * BYTES_PER_PIXEL[self.type]

    def subtitle(self) -> str:
        """The info line above the image, e.g. "256x254 pixels; 8-bit (inverting LUT); 64K"."""
        parts = []
        if self.is_stack:
            c, z, t = self.position
            if self.n_channels > 1 or self.n_frames > 1 or self.hyperstack:
                pos = []
                if self.n_channels > 1:
                    pos.append(f"c:{c + 1}/{self.n_channels}")
                if self.n_slices > 1:
                    pos.append(f"z:{z + 1}/{self.n_slices}")
                if self.n_frames > 1:
                    pos.append(f"t:{t + 1}/{self.n_frames}")
                s = " ".join(pos)
            else:
                s = f"{self.current_index + 1}/{self.stack_size}"
            label = self.slice_label()
            if label:
                s += f" ({label.splitlines()[0][:60]})"
            parts.append(s)
        cal = self.calibration
        if cal.scaled():
            w, h = self.width * cal.pixel_width, self.height * cal.pixel_height
            parts.append(f"{_d2s(w)}x{_d2s(h)} {cal.units} ({self.width}x{self.height})")
        else:
            parts.append(f"{self.width}x{self.height} pixels")
        kind = self.type_name
        if self.is_inverted_lut():
            kind += " (inverting LUT)"
        parts.append(kind)
        size = self.size_bytes() / 1024.0
        if size < 10000:
            parts.append(f"{size:.0f}K")
        elif size < 10000 * 1024:
            parts.append(f"{size / 1024:.1f}MB")
        else:
            parts.append(f"{size / 1024 / 1024:.1f}GB")
        return "; ".join(parts)

    def __repr__(self) -> str:
        c, z, t = self.dims
        return (f"ImageDoc({self.title!r}, {self.width}x{self.height}, c={c} z={z} t={t}, "
                f"{self.type_name})")


def _d2s(v: float) -> str:
    return f"{v:.0f}" if float(v).is_integer() else f"{v:.2f}"


def _type_for_dtype(dtype) -> int:
    dtype = np.dtype(dtype)
    if dtype == np.uint8:
        return GRAY8
    if dtype == np.uint16:
        return GRAY16
    if dtype == np.float32:
        return GRAY32
    raise ValueError(f"unsupported pixel dtype {dtype}; use ImageDoc.from_array")


_ID = [0]


def _next_id() -> int:
    """Unique negative ids like ImageJ's ``ImagePlus.getID()``."""
    _ID[0] -= 1
    return _ID[0]
