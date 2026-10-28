"""Spatial, temporal and value calibration of an image (ImageJ ``ij.measure.Calibration``).

Spatial: pixel width/height/depth in ``unit`` (``"pixel"`` when uncalibrated), origin in pixels,
optional invert-Y. Temporal: ``frame_interval`` in ``time_unit``. Value calibration maps raw
pixel values to calibrated ones with ``function`` + ``coefficients`` (ImageJ CurveFitter /
Calibration constants below); 16-bit signed images are ``STRAIGHT_LINE`` with ``(-32768, 1)``.

Value functions are looked up in :data:`VALUE_FUNCTIONS`; WP-C (Analyze > Calibrate) adds more
with :func:`register_value_function` without editing this file.
Ported in part from ImageJ 1.54 ij/measure/Calibration.java (public domain, NIH; W. Rasband).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np

# ij.measure.CurveFitter / Calibration function codes
STRAIGHT_LINE, POLY2, POLY3, POLY4, EXPONENTIAL, POWER, LOG, RODBARD = 0, 1, 2, 3, 4, 5, 6, 7
GAMMA_VARIATE, LOG2, RODBARD2, EXP_WITH_OFFSET, GAUSSIAN, EXP_RECOVERY = 8, 9, 10, 11, 12, 13
NONE, UNCALIBRATED_OD, CUSTOM = 20, 21, 22

ValueFunction = Callable[[np.ndarray, tuple], np.ndarray]


def _poly(x, c):
    out = np.zeros_like(x, dtype=float)
    for k in reversed(c):
        out = out * x + k
    return out


def _power(x, c):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(x == 0, 0.0, c[0] * np.exp(c[1] * np.log(x)))


def _log(x, c):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(x == 0, -1e100, c[0] * np.log(c[1] * x))


def _rodbard(x, c):
    a, b, cc, d = c[:4]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return d + (a - d) / (1.0 + np.power(x / cc, b))


VALUE_FUNCTIONS: dict[int, ValueFunction] = {
    STRAIGHT_LINE: lambda x, c: c[0] + c[1] * x,
    POLY2: _poly, POLY3: _poly, POLY4: _poly,
    EXPONENTIAL: lambda x, c: c[0] * np.exp(c[1] * x),
    POWER: _power,
    LOG: _log,
    RODBARD: _rodbard,
}


def register_value_function(code: int, fn: ValueFunction) -> None:
    """Add or replace the evaluator of a value-calibration function code."""
    VALUE_FUNCTIONS[code] = fn


def _od(v: np.ndarray, inverted_lut: bool) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    with np.errstate(divide="ignore"):
        if inverted_lut:
            v = np.where(v == 255.0, 254.5, v)
            return 0.434294481 * np.log(255.0 / (255.0 - v))
        v = np.where(v == 0.0, 0.5, v)
        return 0.434294481 * np.log(255.0 / v)


@dataclass
class Calibration:
    pixel_width: float = 1.0
    pixel_height: float = 1.0
    pixel_depth: float = 1.0
    unit: str = "pixel"            # spatial unit (x); y/z use the same unit unless set below
    y_unit: str | None = None
    z_unit: str | None = None
    x_origin: float = 0.0
    y_origin: float = 0.0
    z_origin: float = 0.0
    invert_y: bool = False
    frame_interval: float = 0.0
    time_unit: str = "sec"
    fps: float = 0.0
    function: int = NONE
    coefficients: tuple = ()
    value_unit: str = "Gray Value"
    inverted_lut: bool = False     # only used by UNCALIBRATED_OD
    info: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ spatial
    def scaled(self) -> bool:
        """True when a spatial scale is set (ImageJ ``Calibration.scaled()``)."""
        return (self.pixel_width != 1.0 or self.pixel_height != 1.0 or self.pixel_depth != 1.0
                or self.unit not in ("pixel", "pixels", ""))

    @property
    def units(self) -> str:
        """Plural unit as ImageJ shows it ("pixels", "microns", "inches")."""
        u = self.unit
        if u in ("pixel", ""):
            return "pixels"
        if u in ("micron", "µm", "um"):
            return "microns" if u == "micron" else "µm"
        if u == "inch":
            return "inches"
        return u

    def get_x(self, x: float) -> float:
        return (x - self.x_origin) * self.pixel_width

    def get_y(self, y: float, image_height: int = 0) -> float:
        if self.invert_y:
            y = (self.y_origin - y) if self.y_origin != 0 else (image_height - y - 1)
            return y * self.pixel_height
        return (y - self.y_origin) * self.pixel_height

    def get_z(self, z: float) -> float:
        return (z - self.z_origin) * self.pixel_depth

    def raw_x(self, x: float) -> float:
        return x / self.pixel_width + self.x_origin

    def raw_y(self, y: float) -> float:
        return y / self.pixel_height + self.y_origin

    def set_scale(self, distance_pixels: float, known_distance: float, unit: str,
                  aspect_ratio: float = 1.0) -> None:
        """Analyze > Set Scale semantics: known/pixels units per pixel."""
        if distance_pixels <= 0 or known_distance <= 0 or unit in ("pixel", "pixels", ""):
            self.pixel_width = self.pixel_height = self.pixel_depth = 1.0
            self.unit = "pixel"
            return
        self.pixel_width = known_distance / distance_pixels
        self.pixel_height = self.pixel_width * aspect_ratio
        self.pixel_depth = self.pixel_width
        self.unit = unit

    # ------------------------------------------------------------------ values
    def calibrated(self) -> bool:
        return self.function != NONE

    def set_signed16(self) -> None:
        self.function, self.coefficients, self.value_unit = STRAIGHT_LINE, (-32768.0, 1.0), \
            "Gray Value"

    def is_signed16(self) -> bool:
        return self.function == STRAIGHT_LINE and tuple(self.coefficients) == (-32768.0, 1.0)

    def disable_density(self) -> None:
        self.function, self.coefficients, self.value_unit = NONE, (), "Gray Value"

    def cvalue(self, raw):
        """Calibrated value(s) of raw pixel value(s); scalar in, float out."""
        scalar = np.ndim(raw) == 0
        x = np.asarray(raw, dtype=float)
        if self.function == NONE:
            out = x
        elif self.function == UNCALIBRATED_OD:
            out = _od(x, self.inverted_lut)
        else:
            fn = VALUE_FUNCTIONS.get(self.function)
            if fn is None:
                raise NotImplementedError(f"value calibration function {self.function}")
            out = np.asarray(fn(x, tuple(self.coefficients)), dtype=float)
        return float(out) if scalar else out

    def ctable(self, bit_depth: int) -> np.ndarray | None:
        """Lookup table raw -> calibrated for 8/16-bit images (None when uncalibrated)."""
        if self.function == NONE or bit_depth not in (8, 16):
            return None
        return self.cvalue(np.arange(256 if bit_depth == 8 else 65536, dtype=float))

    def raw_value(self, value: float, bit_depth: int) -> float:
        """Raw value giving the calibrated ``value`` (nearest table entry, like ImageJ)."""
        table = self.ctable(bit_depth)
        if table is None:
            return value
        finite = np.where(np.isfinite(table), table, np.inf)
        return float(np.argmin(np.abs(finite - value)))

    def copy(self) -> "Calibration":
        return replace(self, coefficients=tuple(self.coefficients), info=dict(self.info))

    def __eq__(self, other) -> bool:  # NaN-safe enough for tests
        if not isinstance(other, Calibration):
            return NotImplemented
        return all(getattr(self, f) == getattr(other, f) or
                   (isinstance(getattr(self, f), float) and math.isnan(getattr(self, f))
                    and math.isnan(getattr(other, f)))
                   for f in self.__dataclass_fields__)
