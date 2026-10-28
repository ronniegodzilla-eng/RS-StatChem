"""Shared helpers for the Process-menu command modules (``dialogs/process_*.py``). **WP-B.**

No Qt at import time: these helpers run headless (macros, selftest). They reproduce what
ImageJ's ``PlugInFilterRunner`` does around a filter:

* the filter sees the whole plane (so neighbourhood filters read pixels outside a selection),
* only pixels inside an *area* selection are written back (rectangle = its pixel block, other
  shapes = ``roi.get_mask``; ImageJ's ``reset(mask)``); line/point selections are ignored,
* RGB images are processed channel by channel (``ColorProcessor`` splits into three bytes),
* "Process all N slices" iterates every plane of the stack in ImageJ (czt) order.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from rsstatchem.core.imaging.commands import CommandError
from rsstatchem.core.imaging.image import COLOR_RGB, GRAY8, GRAY16, GRAY32, ImageDoc

ALL_TYPES = (GRAY8, GRAY16, GRAY32, COLOR_RGB)
GRAY_TYPES = (GRAY8, GRAY16, GRAY32)


def roi_mask(image: ImageDoc) -> np.ndarray | None:
    """Bool (H, W) mask of the image's area selection, or None (no/non-area selection)."""
    roi = image.roi
    if roi is None or not roi.is_area():
        return None
    mask = roi.get_mask((image.height, image.width))
    if mask.all():
        return None
    return mask


def roi_rect(image: ImageDoc) -> tuple[int, int, int, int]:
    """(x, y, w, h) integer bounds of the area selection clipped to the image, else the whole
    image (ImageJ ``ip.getRoi()``)."""
    roi = image.roi
    if roi is None or not roi.is_area():
        return 0, 0, image.width, image.height
    x, y, w, h = roi.bounds()
    x0, y0 = max(int(round(x)), 0), max(int(round(y)), 0)
    x1, y1 = min(int(round(x + w)), image.width), min(int(round(y + h)), image.height)
    if x1 <= x0 or y1 <= y0:
        return 0, 0, image.width, image.height
    return x0, y0, x1 - x0, y1 - y0


def write_back(image: ImageDoc, new: np.ndarray, c: int, z: int, t: int,
               mask: np.ndarray | None) -> None:
    """Store ``new`` (same shape/dtype as the plane) inside ``mask`` (or everywhere)."""
    plane = image.plane(c, z, t)
    new = np.asarray(new)
    if new.dtype != plane.dtype:
        new = new.astype(plane.dtype)
    if mask is None:
        plane[...] = new
    else:
        plane[mask] = new[mask]


def apply_filter(image: ImageDoc, fn: Callable[[np.ndarray], np.ndarray], stack: bool = False,
                 rgb: str = "channels", use_roi: bool = True, progress=None) -> None:
    """Run ``fn(plane) -> plane`` on the current plane (or all with ``stack``).

    ``rgb``: "channels" applies ``fn`` to each uint8 channel of an RGB plane; "whole" passes
    the (H, W, 3) array. ``use_roi`` restricts writing to the area selection."""
    mask = roi_mask(image) if use_roi else None
    planes = list(image.iter_planes(stack))
    for i, (c, z, t) in enumerate(planes):
        plane = image.plane(c, z, t)
        if image.type == COLOR_RGB and rgb == "channels":
            out = np.empty_like(plane)
            for k in range(3):
                out[..., k] = fn(np.ascontiguousarray(plane[..., k]))
        else:
            out = fn(plane.copy())
        write_back(image, out, c, z, t, mask)
        if progress is not None and len(planes) > 1:
            progress((i + 1) / len(planes))


def require_binary(image: ImageDoc, label: str = "") -> None:
    """ImageJ's "8-bit binary image (0 and 255) required" check (current plane)."""
    if image.type != GRAY8:
        raise CommandError("8-bit binary image (0 and 255) required.")
    vals = np.unique(image.plane())
    if not set(vals.tolist()) <= {0, 255}:
        raise CommandError("8-bit binary image (0 and 255) required.")


def session_progress(session):
    """A progress callback for core loops (None when no session)."""
    return getattr(session, "progress", None)


def new_image_like(image: ImageDoc, pixels: np.ndarray, title: str, type: int | None = None
                   ) -> ImageDoc:
    """A new ImageDoc with the calibration of ``image`` (ImageJ copies the calibration)."""
    doc = ImageDoc(pixels, type, title, image.calibration.copy())
    return doc
