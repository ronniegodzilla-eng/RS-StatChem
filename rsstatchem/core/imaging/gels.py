"""Analyze > Gels (``ij.plugin.GelAnalyzer``): lanes as equal rectangles, lane profiles
(average across the lane, optionally uncalibrated OD), peak areas between baseline segments
and the "Label Peaks" percentages.

Workflow as in ImageJ: Select First Lane (rectangle), Select Next Lane (the same size moved to
the next lane), Plot Lanes (stacked profiles), then close peaks with straight lines and click
inside them with the wand (in the plot window) or use :func:`auto_peaks`; Label Peaks writes
area and percent of total.

Ported in part from ImageJ 1.54 ij/plugin/GelAnalyzer.java (public domain, NIH).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from rsstatchem.core.imaging.results import ResultsTable
from rsstatchem.core.imaging.roi import geometry as G
from rsstatchem.core.imaging.roi import shapes as S


@dataclass
class GelOptions:
    vertical: bool = True          # "Vertical lanes" (else horizontal; guessed from 1st lane)
    uncalibrated_od: bool = True
    label_percent: bool = True
    invert_peaks: bool = False
    plot_width: int = 0
    plot_height: int = 0


@dataclass
class GelState:
    image_id: int | None = None
    lanes: list = field(default_factory=list)       # RectRoi per lane
    vertical: bool | None = None
    profiles: list = field(default_factory=list)
    peaks: list = field(default_factory=list)       # (lane, start, end, area)

    def reset(self):
        self.image_id = None
        self.lanes.clear()
        self.vertical = None
        self.profiles.clear()
        self.peaks.clear()


def select_first_lane(state: GelState, image, roi, options: GelOptions):
    if not isinstance(roi, S.RectRoi):
        raise ValueError("Rectangular selection required.")
    state.reset()
    state.image_id = image.id
    x, y, w, h = G.int_bounds(roi)
    state.vertical = h >= w if options.vertical is None else (h >= w)
    lane = S.RectRoi(x, y, w, h, name="Lane 1")
    state.lanes.append(lane)
    return lane


def select_next_lane(state: GelState, image, roi):
    if not state.lanes:
        raise ValueError('You must first use the "Select First Lane" command.')
    if state.image_id is not None and image.id != state.image_id:
        raise ValueError("The lanes must be on the same image.")
    first = state.lanes[0]
    if not isinstance(roi, S.RectRoi):
        raise ValueError("Rectangular selection required.")
    x, y, _, _ = G.int_bounds(roi)
    _, _, w, h = G.int_bounds(first)
    if state.vertical:
        y = int(first.y)
    else:
        x = int(first.x)
    lane = S.RectRoi(x, y, w, h, name=f"Lane {len(state.lanes) + 1}")
    state.lanes.append(lane)
    return lane


def lane_profile(image, lane, vertical: bool, uncalibrated_od: bool) -> np.ndarray:
    from rsstatchem.core.imaging.profile import value_plane

    vp = value_plane(image, not uncalibrated_od)
    if uncalibrated_od and image.bit_depth == 8:
        inv = image.is_inverted_lut()
        v = vp.copy()
        if inv:
            v = np.where(v == 255.0, 254.5, v)
            vp = 0.434294481 * np.log(255.0 / (255.0 - v))
        else:
            v = np.where(v == 0.0, 0.5, v)
            vp = 0.434294481 * np.log(255.0 / v)
    x, y, w, h = G.int_bounds(lane)
    block = vp[max(y, 0):y + h, max(x, 0):x + w]
    return block.mean(axis=1) if vertical else block.mean(axis=0)


def plot_lanes(state: GelState, image, options: GelOptions) -> list[np.ndarray]:
    if not state.lanes:
        raise ValueError('You must first use the "Select First Lane" command.')
    state.profiles = [lane_profile(image, ln, state.vertical, options.uncalibrated_od)
                      for ln in state.lanes]
    if options.invert_peaks:
        state.profiles = [p.max() - p for p in state.profiles]
    return state.profiles


def peak_area(profile: np.ndarray, start: int, end: int) -> float:
    """Area between the profile and the straight baseline from ``start`` to ``end``."""
    start, end = sorted((int(start), int(end)))
    seg = profile[start:end + 1].astype(float)
    if len(seg) < 2:
        return 0.0
    base = np.linspace(seg[0], seg[-1], len(seg))
    return float(np.trapezoid(np.clip(seg - base, 0, None)))


def auto_peaks(profile: np.ndarray, min_prominence: float | None = None):
    """Peaks bounded by the neighbouring minima: [(start, end, area)]."""
    from scipy.signal import find_peaks

    p = np.asarray(profile, float)
    if min_prominence is None:
        min_prominence = 0.05 * (p.max() - p.min() or 1.0)
    peaks, props = find_peaks(p, prominence=min_prominence)
    out = []
    for k, pk in enumerate(peaks):
        lb, rb = int(props["left_bases"][k]), int(props["right_bases"][k])
        out.append((lb, rb, peak_area(p, lb, rb)))
    return out


def label_peaks(state: GelState, percent: bool = True) -> ResultsTable:
    rt = ResultsTable("Results")
    total = sum(a for *_, a in state.peaks) or float("nan")
    for i, (lane, start, end, area) in enumerate(state.peaks):
        row = {"Area": area}
        if percent:
            row["Percent"] = 100.0 * area / total
        rt.add_row(row, label=f"Lane {lane + 1}")
    return rt


def lane_overlay(state: GelState) -> S.Overlay:
    ov = S.Overlay(show_labels=True)
    for ln in state.lanes:
        r = ln.copy()
        r.stroke_color = "#ffff00"
        ov.add(r)
    return ov


def peak_percentages(areas) -> list[float]:
    total = float(sum(areas))
    return [100.0 * a / total if total else math.nan for a in areas]
