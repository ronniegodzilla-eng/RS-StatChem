"""Image Analysis commands: Analyze menu, Edit > Selection and the Fiji analysis plugins
(Cell Counter, Coloc 2, Analyze Skeleton). **Owned by WP-C.**

``register(registry)`` adds every label of ``menu_layout.labels_of_area("analyze")`` plus
"Cell Counter", "Coloc 2" and "Analyze Skeleton (2D/3D)" (Fiji names; not in ImageJ's menus).
Commands work headlessly (``session.run("Measure")``, macro option strings as ImageJ records
them); window-showing results (histograms, plots, ROI Manager, Cell Counter) open windows when
an ImageJ bar exists for the session, and are kept in ``session._wpc`` otherwise.
``selftest()`` runs a few headless checks (no Qt widgets). See docs/IMAGEJ_CONTRACTS.md.
"""
from __future__ import annotations

import math

import numpy as np

from rsstatchem.core.imaging import commands as C
from rsstatchem.core.imaging import measure as M
from rsstatchem.core.imaging.commands import Command, CommandError
from rsstatchem.core.imaging.image import (COLOR_256, COLOR_RGB, GRAY8, GRAY16, GRAY32,
                                           ImageDoc)
from rsstatchem.core.imaging.roi import geometry as G
from rsstatchem.core.imaging.roi import selection_ops as SO
from rsstatchem.core.imaging.roi import shapes as S

GRAY_TYPES = (GRAY8, GRAY16, GRAY32, COLOR_256)
ALL_TYPES = (GRAY8, GRAY16, GRAY32, COLOR_256, COLOR_RGB)

MEASURE_LABELS = ("Area", "Mean gray value", "Standard deviation", "Modal gray value",
                  "Min & max gray value", "Centroid", "Center of mass", "Perimeter",
                  "Bounding rectangle", "Fit ellipse", "Shape descriptors", "Feret's diameter",
                  "Integrated density", "Median", "Skewness", "Kurtosis", "Area_fraction",
                  "Stack position")
OPTION_LABELS = ("Limit to threshold", "Display label", "Invert Y coordinates",
                 "Scientific notation", "Add to overlay", "NaN empty cells")


# ---------------------------------------------------------------------------------- state
class WpcState:
    """Per-session state of the analysis commands (not persisted)."""

    def __init__(self, session):
        self.previous_roi: dict[int, object] = {}
        self.last_roi = None
        self.plots: list = []
        self.histograms: list = []
        self.gel = None
        self.gel_options = None
        self.global_calibration = None
        self.counters = None
        self.sync = None
        self._roi_seen: dict[int, object] = {}
        session.on("image_updated", self._image_updated)

    def _image_updated(self, doc):
        cur = getattr(doc, "roi", None)
        prev = self._roi_seen.get(doc.id)
        if cur is not prev:
            if prev is not None:
                self.previous_roi[doc.id] = prev.copy()
                self.last_roi = prev.copy()
            self._roi_seen[doc.id] = cur


def state(session) -> WpcState:
    st = getattr(session, "_wpc", None)
    if st is None:
        st = WpcState(session)
        session._wpc = st
    return st


def set_roi(session, image, roi) -> None:
    """Replace the selection, remembering the old one for Restore Selection."""
    st = state(session)
    if image.roi is not None:
        st.previous_roi[image.id] = image.roi.copy()
        st.last_roi = image.roi.copy()
    image.roi = roi
    st._roi_seen[image.id] = roi


def bar_for(session):
    """The ImageJ bar showing ``session`` (None headless)."""
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:  # pragma: no cover
        return None
    app = QApplication.instance()
    if app is None:
        return None
    from rsstatchem.ui.imagej import windows as W
    for b in list(W.BARS):
        if getattr(b, "session", None) is session:
            return b
    for w in app.topLevelWidgets():
        if getattr(w, "session", None) is session and hasattr(w, "run_command"):
            return w
    return None


def _flags(session) -> int:
    return M.flags_from_keys(session.options.measurements)


def _redirect(session, image):
    title = session.options.redirect
    if not title or title == "None":
        return None
    target = session.find_image(title)
    if target is None:
        session.options.redirect = "None"
        raise CommandError(f'Redirect image ("{title}")\nnot found.')
    if target.width != image.width or target.height != image.height:
        raise CommandError(f'Redirect image ("{title}") \nis not the same size as the '
                           f'current image.')
    return target


def _results(session):
    return session.results


def _weighted(session) -> bool:
    return bool(session.options.weighted_rgb)


def _ensure_counter(session) -> None:
    rt = session.results
    rt.precision = -session.options.decimal_places if session.options.measure("scientific") \
        else session.options.decimal_places
    rt.nan_empty_cells = session.options.measure("nan")


# ================================================================================== Analyze
def measure(session, image):
    _ensure_counter(session)
    redirect = _redirect(session, image)
    an = M.Analyzer(image, _flags(session), session.results, session.options.decimal_places,
                    redirect=redirect, weighted_rgb=_weighted(session))
    an.measure()
    return session.results


def particles_fields(session, image):
    cal = image.calibration if image is not None else None
    scaled = cal is not None and cal.scaled()
    unit = (cal.unit if scaled else "pixel") + "^2"
    ex = session.options.extra.get("analyze_particles", {})
    fields = [C.string(f"Size ({unit}):", "0-Infinity", key="size")]
    if scaled:
        fields.append(C.checkbox("Pixel units", ex.get("pixel", False), key="pixel"))
    fields += [C.string("Circularity:", "0.00-1.00", key="circularity"),
               C.choice("Show:", ("Nothing", "Overlay", "Overlay Masks", "Outlines",
                                  "Bare Outlines", "Ellipses", "Masks", "Count Masks"),
                        "Nothing", key="show"),
               C.checkbox_group(("Display results", "Exclude on edges", "Clear results",
                                 "Include holes", "Summarize", "Overlay", "Add to Manager",
                                 "Composite ROIs", "Record starts"),
                                tuple(ex.get(k, False) for k in (
                                    "display", "exclude", "clear", "include", "summarize",
                                    "overlay", "add", "composite", "record")), columns=2),
               C.checkbox("In_situ Show", ex.get("in_situ", False), key="in_situ"),
               C.checkbox("Four connected", False, key="four")]
    return fields


def analyze_particles(session, image, size="0-Infinity", pixel=False, circularity="0.00-1.00",
                      show="Nothing", display=False, exclude=False, clear=False,
                      include=False, summarize=False, overlay=False, add=False,
                      composite=False, record=False, in_situ=False, four=False, stack=False,
                      **_):
    from rsstatchem.core.imaging import particles as P

    if image.type == COLOR_RGB:
        raise CommandError("RGB images must be thresholded using\n"
                           "Image>Adjust>Color Threshold.")
    session.options.extra["analyze_particles"] = dict(
        size=size, pixel=pixel, circularity=circularity, show=show, display=display,
        exclude=exclude, clear=clear, include=include, summarize=summarize, overlay=overlay,
        add=add, composite=composite, record=record, in_situ=in_situ)
    cal = image.calibration
    unit2 = 1.0 if (pixel or not cal.scaled()) else cal.pixel_width * cal.pixel_height
    lo, hi = P.parse_range(size, math.inf)
    min_size, max_size = lo / unit2, hi / unit2
    if max_size < min_size:
        max_size = math.inf
    cmin, cmax = P.parse_range(circularity, 1.0)
    cmin = max(cmin, 0.0)
    if cmin > cmax and cmax == 1.0:
        cmin = 0.0
    if cmin > cmax:
        cmin = cmax
    if cmin == 1.0 and cmax == 1.0:
        cmin = 0.0
    opt = P.ParticleOptions(min_size=min_size, max_size=max_size, min_circularity=cmin,
                            max_circularity=cmax, show=show, display=display, exclude=exclude,
                            clear=clear, include_holes=include or four, summarize=summarize,
                            overlay=overlay, add_to_manager=add, composite=composite,
                            record_starts=record, in_situ=in_situ, four_connected=four,
                            stack=stack)
    if summarize:
        session.options.measurements = tuple(dict.fromkeys(
            tuple(session.options.measurements) + ("area",)))
    _ensure_counter(session)
    rt = session.results if display else M.ResultsTable.results("Results")
    if display:
        if clear:
            rt.reset()
        M.clear_summary(rt)
    rt.precision = session.results.precision
    title = "Summary" if image.stack_size == 1 or not stack else f"Summary of {image.title}"
    summary = session.tables.get(title) if summarize else None
    try:
        res = P.analyze_particles(image, _flags(session), opt, rt=rt, summary=summary,
                                  black_background=session.options.black_background,
                                  redirect=_redirect(session, image),
                                  precision=session.options.decimal_places,
                                  progress=session.progress)
    except ValueError as exc:
        raise CommandError(str(exc)) from None
    out = []
    if display:
        out.append(rt)
    if summarize and res.summary is not None:
        res.summary.title = title
        out.append(res.summary)
    if add and res.rois:
        if clear:
            session.roi_manager.clear()
        session.roi_manager.extend(res.rois)
        _manager_changed(session, image)
    if res.output is not None:
        out.append(res.output)
    if display and rt.size == 0 and image.threshold is not None:
        lo_t, hi_t = image.threshold[0], image.threshold[1]
        session.log(f"No particles detected. The threshold ({lo_t:g}-{hi_t:g}) may not be "
                    f"correct.")
    return out or None


def summarize(session, image):
    rt = session.results
    M.summarize(rt)
    return rt


def distribution_fields(session, image):
    heads = [h for h in session.results.headings if h != "Label"] or ["Area"]
    ex = session.options.extra.get("distribution", {})
    default = ex.get("parameter", "Area")
    return [C.choice("Parameter:", heads, default if default in heads else heads[0],
                     key="parameter"),
            C.checkbox("Automatic binning", ex.get("automatic", True), key="automatic"),
            C.number("or specify bins:", ex.get("or", 10), 0, key="or", integer=True),
            C.string("and range:", ex.get("and", "0-0"), key="and")]


def distribution(session, image, parameter="Area", automatic=True, or_=None, and_="0-0",
                 **kw):
    from rsstatchem.core.imaging.histogram import HistogramData

    nbins = int(kw.get("or", or_ if or_ is not None else 10))
    rng = kw.get("and", and_)
    rt = session.results
    if rt.size == 0 or not rt.column_exists(parameter):
        raise CommandError(f'No "{parameter}" column in the Results table')
    session.options.extra["distribution"] = {"parameter": parameter, "automatic": automatic,
                                            "or": nbins, "and": rng}
    v = rt.column(parameter)
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise CommandError("No data")
    lo, hi = (float(t) for t in (str(rng).split("-") + ["0"])[:2]) if "-" in str(rng) \
        else (0.0, 0.0)
    if lo == 0 and hi == 0:
        lo, hi = float(v.min()), float(v.max())
    if automatic:
        sd = float(v.std(ddof=1)) if v.size > 1 else 0.0
        bw = 3.49 * sd * v.size ** (-1 / 3) if sd > 0 else 1.0
        nbins = max(1, int(math.floor((hi - lo) / bw + 0.5))) if bw > 0 else 10
    nbins = max(nbins, 1)
    width = (hi - lo) / nbins if hi > lo else 1.0
    idx = np.clip(((v - lo) / width).astype(np.int64), 0, nbins - 1)
    sel = (v >= lo) & (v <= hi)
    counts = np.bincount(idx[sel], minlength=nbins)
    k = int(np.argmax(counts))
    data = HistogramData(counts=counts, bin_starts=lo + np.arange(nbins) * width,
                         bin_width=width, n=int(sel.sum()), mean=float(v.mean()),
                         std_dev=float(v.std(ddof=1)) if v.size > 1 else 0.0,
                         min=float(v.min()), max=float(v.max()),
                         mode=float(lo + (k + 0.5) * width), mode_count=int(counts[k]),
                         n_bins=nbins, hist_min=lo, hist_max=hi,
                         title=f"{parameter} Distribution", value_unit=parameter,
                         integer_bins=False)
    _show_histogram(session, None, data, {"kind": "distribution"})
    return None


def label(session, image):
    from rsstatchem.core.imaging.particles import draw_label, draw_polygon

    roi = image.roi
    if roi is None:
        raise CommandError("Selection required")
    n = session.results.size
    if n == 0:
        raise CommandError('"Label" requires a measurement (Analyze > Measure)')
    image.snapshot("Label")
    fg = _foreground_value(session, image)
    plane = image.plane()
    x, y, w, h = G.int_bounds(roi)
    if not isinstance(roi, S.PointRoi):
        xs, ys = G.float_polygon(roi)
        draw_polygon(plane, np.floor(xs), np.floor(ys), fg, closed=roi.is_area())
    text = str(n)
    draw_label(plane, text, x + w // 2 - 3 * len(text), y + h // 2 + 4, fg)
    image.changed = True


def _foreground_value(session, image):
    from rsstatchem.core.imaging.roi.selection_ops import _rgb
    rgb = _rgb(session.options.foreground, (255, 255, 255))
    if image.is_rgb:
        return np.array(rgb, np.uint8)
    gray = sum(rgb) / 3.0
    if image.type == GRAY16:
        return int(round(gray / 255.0 * 65535))
    if image.type == GRAY32:
        lo, hi = image.display_range
        return lo + (hi - lo) * gray / 255.0
    v = int(round(gray))
    return 255 - v if image.is_inverted_lut() else v


def clear_results(session, image):
    session.results.reset()
    session.show_table(session.results)


def set_measurements_fields(session, image):
    o = session.options
    keys = [C.macro_key(lb) for lb in MEASURE_LABELS]
    okeys = [C.macro_key(lb) for lb in OPTION_LABELS]
    titles = ["None"] + [im.title for im in session.images]
    red = o.redirect if o.redirect in titles else "None"
    return [C.checkbox_group(MEASURE_LABELS, tuple(o.measure(k) for k in keys), columns=2),
            C.checkbox_group(OPTION_LABELS, tuple(o.measure(k) for k in okeys), columns=2),
            C.choice("Redirect to:", titles, red, key="redirect"),
            C.number("Decimal places (0-9):", o.decimal_places, 0, key="decimal",
                     integer=True)]


def set_measurements(session, image, redirect="None", decimal=3, **kw):
    keys = [C.macro_key(lb) for lb in MEASURE_LABELS + OPTION_LABELS]
    chosen = tuple(k for k in keys if kw.get(k))
    o = session.options
    o.measurements = chosen
    o.redirect = redirect or "None"
    o.decimal_places = int(min(max(int(decimal), 0), 9))
    _ensure_counter(session)
    if session.results.size:
        session.show_table(session.results)


def set_scale_fields(session, image):
    cal = image.calibration
    distance = 0.0
    known = 0.0
    aspect = 1.0
    unit = cal.unit if cal.scaled() else "pixel"
    roi = image.roi
    if roi is not None and isinstance(roi, S.LineRoi):
        distance = round(roi.length(), 2)
        known = round(roi.length(cal.pixel_width, cal.pixel_height), 4) if cal.scaled() else 0
    elif cal.scaled():
        distance = 1.0
        known = cal.pixel_width
        aspect = cal.pixel_height / cal.pixel_width if cal.pixel_width else 1.0
    return [C.number("Distance in pixels:", distance, 2, key="distance"),
            C.number("Known distance:", known, 4, key="known"),
            C.number("Pixel aspect ratio:", aspect, 1, key="pixel"),
            C.string("Unit of length:", unit, key="unit"),
            C.checkbox("Global", state(session).global_calibration is not None, key="global")]


def set_scale(session, image, distance=0.0, known=0.0, pixel=1.0, unit="pixel",
              global_=False, **kw):
    is_global = kw.get("global", global_)
    cal = image.calibration
    if unit in ("um", "micron", "microns", "u"):
        unit = "µm"
    cal.set_scale(float(distance), float(known), unit, float(pixel or 1.0))
    st = state(session)
    if is_global:
        st.global_calibration = cal.copy()
        for im in session.images:
            if im is not image:
                c2 = im.calibration
                c2.pixel_width, c2.pixel_height = cal.pixel_width, cal.pixel_height
                c2.pixel_depth, c2.unit = cal.pixel_depth, cal.unit
                session.image_updated(im)
    else:
        st.global_calibration = None


def calibrate_fields(session, image):
    from rsstatchem.core.imaging import calibrate as CB

    cal = image.calibration
    funcs = CB.function_list()
    if cal.function == 20:
        default = "None"
    elif cal.function == 21:
        default = CB.UNCALIBRATED_OD
    elif cal.function == 0 and tuple(cal.coefficients)[:2] in ((255.0, -1.0), (65535.0, -1.0)):
        default = CB.INVERTER
    elif 0 <= cal.function < len(CB.FIT_LIST):
        default = CB.FIT_LIST[cal.function]
    else:
        default = "None"
    rt = session.results
    means = ""
    if rt.size and rt.column_exists("Mean"):
        means = " ".join(f"{v:.2f}" for v in rt.column("Mean")[:20])
    return [C.choice("Function:", funcs, default, key="function"),
            C.string("Unit:", cal.value_unit, key="unit"),
            C.string("Measured values:", means, key="text1"),
            C.string("Known values:", "", key="text2"),
            C.checkbox("Global calibration", False, key="global"),
            C.checkbox("Show plot", False, key="show")]


def calibrate_cmd(session, image, function="None", unit="Gray Value", text1="", text2="",
                  show=False, **kw):
    from rsstatchem.core.imaging import calibrate as CB
    from rsstatchem.core.imaging.profile import PlotData

    if function == "-":
        return None
    try:
        out = CB.calibrate(image, function, unit, text1, text2)
    except ValueError as exc:
        raise CommandError(str(exc)) from None
    if kw.get("global"):
        for im in session.images:
            if im is not image and im.bit_depth == image.bit_depth:
                im.calibration.function = image.calibration.function
                im.calibration.coefficients = image.calibration.coefficients
                im.calibration.value_unit = image.calibration.value_unit
    if image.bit_depth != 8:
        image.reset_display_range()
    if out is not None and show:
        p, r2 = out
        code = CB.FIT_LIST.index(function)
        x = CB.parse_column(text1)
        y = CB.parse_column(text2)
        xx = np.linspace(float(x.min()), float(x.max()), 200)
        plot = PlotData("Calibration Function", "pixel value", unit)
        plot.add(xx, CB.f(code, p, xx), label="fit")
        plot.add(x, y, style="circle", label="standards")
        plot.labels.append((float(x.min()), float(y.max()),
                            f"{CB.FORMULAS[code]}  R^2={r2:.4f}"))
        _show_plot(session, plot)
    return None


def histogram_fields(session, image):
    if image is None:
        return []
    fields = []
    if image.type in (GRAY16, GRAY32):
        lo, hi = image.plane_min_max()
        ex = session.options.extra.get("histogram", {})
        fields += [C.number("Bins:", ex.get("bins", 256), 0, key="bins", integer=True),
                   C.checkbox("Use pixel value range", False, key="use"),
                   C.number("X_min:", lo, 2 if image.type == GRAY32 else 0, key="x_min"),
                   C.number("X_max:", hi, 2 if image.type == GRAY32 else 0, key="x_max"),
                   C.string("Y_max:", "Auto", key="y_max")]
    if image.stack_size > 1:
        fields.append(C.checkbox("Stack histogram", False, key="stack"))
    return fields


def histogram_cmd(session, image, bins=256, use=False, x_min=0.0, x_max=0.0, y_max="Auto",
                  stack=False, **_):
    from rsstatchem.core.imaging.histogram import histogram as H

    if image.type in (GRAY16, GRAY32):
        session.options.extra["histogram"] = {"bins": int(bins)}
        lo, hi = (0.0, 0.0) if use else (float(x_min), float(x_max))
        if image.type == GRAY16 and int(bins) == 256 and not use and \
                (lo, hi) == image.plane_min_max():
            lo = hi = 0.0
        data = H(image, int(bins), lo, hi, stack=stack, weighted_rgb=_weighted(session))
    else:
        data = H(image, 256, stack=stack, weighted_rgb=_weighted(session))
    ymax = None
    try:
        ymax = float(y_max)
    except (TypeError, ValueError):
        ymax = None
    _show_histogram(session, image, data, {"kind": "image", "image_id": image.id,
                                           "bins": int(bins), "stack": stack, "y_max": ymax})
    return None


def plot_profile(session, image, **_):
    from rsstatchem.core.imaging.profile import profile_plot

    roi = image.roi
    if roi is None or not (roi.is_line() or isinstance(roi, (S.RectRoi, S.RotatedRectRoi))):
        raise CommandError("Line or rectangular selection required.")
    vertical = bool(session.options.extra.get("vertical_profile", False))
    try:
        plot = profile_plot(image, roi, vertical=vertical)
    except ValueError as exc:
        raise CommandError(str(exc)) from None
    _show_plot(session, plot)
    return None


def surface_fields(session, image):
    ex = session.options.extra.get("surface", {})
    return [C.slider("Polygon Multiplier (10-200%):", 10, 200, ex.get("polygon", 100),
                     key="polygon"),
            C.checkbox("Draw_Wireframe", ex.get("draw_wireframe", False), key="draw_wireframe"),
            C.checkbox("Shade", ex.get("shade", True), key="shade"),
            C.checkbox("Draw_Axis", ex.get("draw_axis", True), key="draw_axis"),
            C.checkbox("Source Background is Lighter", ex.get("source", False), key="source"),
            C.checkbox("Fill Plot Background with Black", ex.get("fill", False), key="fill"),
            C.checkbox("One Polygon Per Line", ex.get("one", False), key="one"),
            C.checkbox("Smooth", ex.get("smooth", True), key="smooth")]


def surface_plot(session, image, polygon=100, draw_wireframe=False, shade=True,
                 draw_axis=True, source=False, fill=False, one=False, smooth=True, **_):
    from rsstatchem.core.imaging.surface import surface_plot as SP

    session.options.extra["surface"] = dict(polygon=polygon, draw_wireframe=draw_wireframe,
                                            shade=shade, draw_axis=draw_axis, source=source,
                                            fill=fill, one=one, smooth=smooth)
    rgb = SP(image, int(polygon), draw_wireframe, shade, draw_axis, source, fill, one, smooth)
    title = image.title
    i = title.rfind(".")
    return ImageDoc.from_array(rgb, "YXS", "Surface Plot of " + (title[:i] if i > 0 else title))


# ---------------------------------------------------------------------------------- gels
def _gel(session):
    from rsstatchem.core.imaging import gels as GL
    st = state(session)
    if st.gel is None:
        st.gel = GL.GelState()
        ex = session.options.extra.get("gel", {})
        st.gel_options = GL.GelOptions(vertical=ex.get("vertical", True),
                                       uncalibrated_od=ex.get("uncalibrated", True),
                                       label_percent=ex.get("label", True),
                                       invert_peaks=ex.get("invert", False))
    return st.gel, st.gel_options


def gel_first(session, image):
    from rsstatchem.core.imaging import gels as GL
    gel, opt = _gel(session)
    try:
        lane = GL.select_first_lane(gel, image, image.roi, opt)
    except ValueError as exc:
        raise CommandError(str(exc)) from None
    _gel_overlay(image, gel)
    session.status(f"Lane 1 selected ({'vertical' if gel.vertical else 'horizontal'} lanes)")
    return lane and None


def gel_next(session, image):
    from rsstatchem.core.imaging import gels as GL
    gel, _ = _gel(session)
    try:
        lane = GL.select_next_lane(gel, image, image.roi)
    except ValueError as exc:
        raise CommandError(str(exc)) from None
    image.roi = lane.copy()
    _gel_overlay(image, gel)
    session.status(f"Lane {len(gel.lanes)} selected")


def _gel_overlay(image, gel):
    from rsstatchem.core.imaging import gels as GL
    ov = GL.lane_overlay(gel)
    keep = [r for r in (image.overlay or []) if not str(r.name).startswith("Lane ")]
    if image.overlay is None:
        image.overlay = ov
    else:
        image.overlay.rois = keep + list(ov.rois)


def gel_plot(session, image):
    from rsstatchem.core.imaging import gels as GL
    from rsstatchem.core.imaging.profile import PlotData
    gel, opt = _gel(session)
    if gel.lanes and image.roi is not None and isinstance(image.roi, S.RectRoi) and \
            G.int_bounds(image.roi) != G.int_bounds(gel.lanes[-1]):
        try:
            GL.select_next_lane(gel, image, image.roi)
        except ValueError:
            pass
    try:
        profiles = GL.plot_lanes(gel, image, opt)
    except ValueError as exc:
        raise CommandError(str(exc)) from None
    plot = PlotData("Plots of " + image.title, "Distance (pixels)",
                    "Uncalibrated OD" if opt.uncalibrated_od else image.calibration.value_unit,
                    {"kind": "gel", "image_id": image.id})
    gel.peaks.clear()
    offset = 0.0
    for k, p in enumerate(profiles):
        plot.add(np.arange(len(p)), p + offset, label=f"Lane {k + 1}")
        for start, end, area in GL.auto_peaks(p):
            gel.peaks.append((k, start, end, area))
        offset += float(p.max() - p.min()) * 1.1 if len(p) else 0
    _show_plot(session, plot)


def gel_replot(session, image):
    gel_plot(session, image)


def gel_reset(session, image):
    gel, _ = _gel(session)
    gel.reset()
    if image is not None and image.overlay is not None:
        image.overlay.rois = [r for r in image.overlay if not str(r.name).startswith("Lane ")]


def gel_label(session, image):
    from rsstatchem.core.imaging import gels as GL
    gel, opt = _gel(session)
    if not gel.peaks:
        raise CommandError("No peaks: use Plot Lanes first")
    rt = GL.label_peaks(gel, opt.label_percent)
    rt.title = "Results"
    session.results.reset()
    for r in range(rt.size):
        row = {h: rt.get_value(h, r) for h in rt.headings if h != "Label"}
        session.results.add_row(row, label=rt.get_label(r))
    return session.results


def gel_options_fields(session, image):
    ex = session.options.extra.get("gel", {})
    return [C.checkbox("Vertical lanes", ex.get("vertical", True), key="vertical"),
            C.checkbox("Uncalibrated OD", ex.get("uncalibrated", True), key="uncalibrated"),
            C.checkbox("Label with percentages", ex.get("label", True), key="label"),
            C.checkbox("Invert peaks", ex.get("invert", False), key="invert")]


def gel_options(session, image, vertical=True, uncalibrated=True, label=True, invert=False):
    session.options.extra["gel"] = dict(vertical=vertical, uncalibrated=uncalibrated,
                                        label=label, invert=invert)
    st = state(session)
    st.gel = None
    _gel(session)


# ---------------------------------------------------------------------------------- tools
def save_xy_fields(session, image):
    return [C.number("Background value:", float(image.plane()[0, 0].mean())
                     if image is not None else 0.0, 0, key="background"),
            C.checkbox("Invert y coordinates", False, key="invert"),
            C.checkbox("Suppress log output", False, key="suppress"),
            Field_file("Save as:", "", key="save")]


def Field_file(label, default="", key=None):  # noqa: N802 - mirrors C.* constructors
    return C.Field("file", label, default, key)


def save_xy(session, image, background=0.0, invert=False, suppress=False, save=""):
    from rsstatchem.core.imaging.fractal import save_xy_coordinates
    if not save:
        raise CommandError("No file name (save=...)")
    n = save_xy_coordinates(save, image, background, invert)
    if not suppress:
        session.log(f"{n} pixels saved to {save}")


def fractal_fields(session, image):
    return [C.string("Box sizes:", "2,3,4,6,8,12,16,32,64", key="box"),
            C.checkbox("Black background", session.options.black_background, key="black")]


def fractal(session, image, box="2,3,4,6,8,12,16,32,64", black=False):
    from rsstatchem.core.imaging.fractal import box_count_table
    from rsstatchem.core.imaging.profile import PlotData
    sizes = [int(float(s)) for s in str(box).replace(" ", ",").split(",") if s.strip()]
    rt, d = box_count_table(image, sizes, black)
    res = session.results
    res.reset()
    row = {"D": d}
    for k in range(rt.size):
        row[f"C{rt.get_value('size', k):g}"] = rt.get_value("count", k)
    res.add_row(row, label=image.title)
    plot = PlotData("Plot", "log (box size)", "log (count)")
    s = np.log(rt.column("size"))
    c = np.log(np.maximum(rt.column("count"), 1))
    plot.add(s, c, style="circle")
    if len(s) > 1:
        slope, icpt = np.polyfit(s, c, 1)
        plot.add(s, slope * s + icpt)
    plot.labels.append((float(s.min()), float(c.min()), f"D={d:.4f}"))
    _show_plot(session, plot)
    return res


def line_graph(session, image):
    from rsstatchem.core.imaging.fractal import analyze_line_graph
    from rsstatchem.core.imaging.profile import PlotData
    rt = analyze_line_graph(image, session.options.black_background)
    plot = PlotData("Line Graph", "X", "Y")
    if rt.size:
        plot.add(rt.column("X"), rt.column("Y"))
    _show_plot(session, plot)
    rt.title = "Results"
    session.results = rt
    return rt


def curve_fit_fields(session, image):
    from rsstatchem.core.imaging import calibrate as CB
    rt = session.results
    xs = ys = ""
    if rt.size and len([h for h in rt.headings if h != "Label"]) >= 2:
        heads = [h for h in rt.headings if h != "Label"]
        xs = " ".join(f"{v:g}" for v in rt.column(heads[0]))
        ys = " ".join(f"{v:g}" for v in rt.column(heads[1]))
    return [C.choice("Function:", CB.FIT_LIST, "Straight Line", key="function"),
            C.string("X values:", xs or "0 1 2 3 4 5", key="x"),
            C.string("Y values:", ys or "0 0.9 4.2 8.8 16.5 25.3", key="y")]


def curve_fit(session, image, function="Straight Line", x="", y=""):
    from rsstatchem.core.imaging import calibrate as CB
    from rsstatchem.core.imaging.profile import PlotData
    xv, yv = CB.parse_column(x), CB.parse_column(y)
    if len(xv) != len(yv) or len(xv) < 2:
        raise CommandError("X and Y must have the same number (>= 2) of values")
    code = CB.FIT_LIST.index(function)
    try:
        p, r2 = CB.fit(code, xv, yv)
    except ValueError as exc:
        raise CommandError(str(exc)) from None
    names = "abcdefghi"
    session.log(f"Formula: {CB.FORMULAS[code]}")
    for k, v in enumerate(p):
        session.log(f"{names[k]} = {v:.6g}")
    session.log(f"R^2 = {r2:.6f}")
    plot = PlotData(f"y = {function}", "X", "Y")
    xx = np.linspace(float(xv.min()), float(xv.max()), 200)
    plot.add(xv, yv, style="circle")
    plot.add(xx, CB.f(code, p, xx))
    plot.labels.append((float(xv.min()), float(yv.max()),
                        f"{CB.FORMULAS[code]}  R^2={r2:.4f}"))
    _show_plot(session, plot)
    return None


def scale_bar_fields(session, image):
    from rsstatchem.core.imaging import scalebar as SB
    ex = session.options.extra.get("scale_bar", {})
    unit = image.calibration.units if image is not None else "pixels"
    return [C.number(f"Width in {unit}:", ex.get("width", SB.default_bar_width(image))
                     if image is not None else 1.0, 2, key="width"),
            C.number("Height in pixels:", ex.get("height", 4), 0, key="height", integer=True),
            C.number("Font size:", ex.get("font", 14), 0, key="font", integer=True),
            C.choice("Color:", tuple(k for k in SB.COLORS if k != "None"),
                     ex.get("color", "White"), key="color"),
            C.choice("Background:", tuple(SB.COLORS), ex.get("background", "None"),
                     key="background"),
            C.choice("Location:", SB.LOCATIONS, ex.get("location", "Lower Right"),
                     key="location"),
            C.checkbox("Horizontal", True, key="horizontal"),
            C.checkbox("Bold Text", ex.get("bold", False), key="bold"),
            C.checkbox("Hide Text", ex.get("hide", False), key="hide"),
            C.checkbox("Serif Font", ex.get("serif", False), key="serif"),
            C.checkbox("Overlay", ex.get("overlay", True), key="overlay"),
            C.checkbox("Label all slices", ex.get("label", False), key="label")]


def scale_bar(session, image, width=0.0, height=4, font=14, color="White", background="None",
              location="Lower Right", horizontal=True, bold=False, hide=False, serif=False,
              overlay=False, label=False):
    from rsstatchem.core.imaging import scalebar as SB
    session.options.extra["scale_bar"] = dict(width=width, height=height, font=font,
                                              color=color, background=background,
                                              location=location, bold=bold, hide=hide,
                                              serif=serif, overlay=overlay, label=label)
    spec = SB.ScaleBarSpec(float(width), int(height), int(font), color, background, location,
                           bold, hide, serif, overlay, label)
    if overlay:
        rois = SB.scale_bar_rois(image, spec, image.roi)
        if image.overlay is None:
            image.overlay = S.Overlay()
        image.overlay.rois = [r for r in image.overlay if not str(r.name).startswith("|SB|")]
        for r in rois:
            image.overlay.add(r)
        return None
    image.snapshot("Scale Bar")
    planes = range(image.stack_size) if label else [image.current_index]
    for i in planes:
        p = image.plane_at(i)
        if image.is_rgb or p.dtype == np.uint8:
            SB.burn_scale_bar(p, image, spec, image.roi)
        else:
            tmp = np.zeros(p.shape, np.uint8)
            SB.burn_scale_bar(tmp, image, spec, image.roi)
            p[tmp > 0] = np.asarray(p.max(), p.dtype)
    image.changed = True
    return None


def calibration_bar_fields(session, image):
    from rsstatchem.core.imaging import scalebar as SB
    ex = session.options.extra.get("calibration_bar", {})
    return [C.choice("Location:", ("Upper Right", "Lower Right", "Lower Left", "Upper Left",
                                   "At Selection"), ex.get("location", "Upper Right"),
                     key="location"),
            C.choice("Fill color:", tuple(SB.COLORS), ex.get("fill", "White"), key="fill"),
            C.choice("Label color:", tuple(k for k in SB.COLORS if k != "None"),
                     ex.get("label", "Black"), key="label"),
            C.number("Number of labels:", ex.get("number", 5), 0, key="number", integer=True),
            C.number("Decimal places:", ex.get("decimal", 0), 0, key="decimal", integer=True),
            C.number("Font size:", ex.get("font", 12), 0, key="font", integer=True),
            C.number("Zoom factor:", ex.get("zoom", 1.0), 1, key="zoom"),
            C.checkbox("Bold text", ex.get("bold", False), key="bold"),
            C.checkbox("Overlay", ex.get("overlay", False), key="overlay")]


def calibration_bar(session, image, location="Upper Right", fill="White", label="Black",
                    number=5, decimal=0, font=12, zoom=1.0, bold=False, overlay=False):
    from rsstatchem.core.imaging import scalebar as SB
    session.options.extra["calibration_bar"] = dict(location=location, fill=fill, label=label,
                                                    number=number, decimal=decimal, font=font,
                                                    zoom=zoom, bold=bold, overlay=overlay)
    spec = SB.CalibrationBarSpec(location, fill, label, int(number), int(decimal), int(font),
                                 float(zoom), bold, overlay)
    if overlay:
        if image.overlay is None:
            image.overlay = S.Overlay()
        image.overlay.rois = [r for r in image.overlay if str(r.name) != "|CB|"]
        for r in SB.calibration_bar_rois(image, spec):
            image.overlay.add(r)
        return None
    rgb = SB.calibration_bar_image(image, spec)
    title = image.title
    i = title.rfind(".")
    doc = ImageDoc.from_array(rgb, "YXS", (title[:i] if i > 0 else title) + " with bar")
    doc.calibration = image.calibration.copy()
    doc.calibration.disable_density()
    return doc


def grid_fields(session, image):
    from rsstatchem.core.imaging import scalebar as SB
    ex = session.options.extra.get("grid", {})
    cal = image.calibration if image is not None else None
    unit = (cal.unit if cal is not None and cal.scaled() else "pixels") + "^2"
    default_area = 0.0
    if image is not None:
        default_area = round(image.width * image.height / 81.0 * cal.pixel_width *
                             cal.pixel_height, 2)
    return [C.choice("Grid type:", SB.GRID_TYPES, ex.get("grid", "Lines"), key="grid"),
            C.number(f"Area per point ({unit}):", ex.get("area", default_area), 2,
                     key="area"),
            C.choice("Color:", tuple(k for k in SB.COLORS if k != "None"),
                     ex.get("color", "Cyan"), key="color"),
            C.checkbox("Center grid on image", ex.get("center", False), key="center"),
            C.checkbox("Random offset", ex.get("random", False), key="random")]


def grid(session, image, grid="Lines", area=0.0, color="Cyan", center=False, random=False):
    from rsstatchem.core.imaging import scalebar as SB
    session.options.extra["grid"] = dict(grid=grid, area=area, color=color, center=center,
                                         random=random)
    rois = SB.grid_rois(image, grid, float(area), color, center, random)
    if image.overlay is None:
        image.overlay = S.Overlay()
    image.overlay.rois = [r for r in image.overlay if r.name != "grid"] + rois


# ================================================================================== Selection
def _need_roi(image, what="This command requires a selection."):
    if image.roi is None:
        raise CommandError(what)
    return image.roi


def _sel(fn):
    """Wrap a selection op: snapshot (Edit > Undo), SelectionError -> CommandError."""
    def run(session, image, **kw):
        try:
            image.snapshot("Selection")
            return fn(session, image, **kw)
        except SO.SelectionError as exc:
            raise CommandError(str(exc)) from None
    run.__name__ = fn.__name__
    run.__doc__ = fn.__doc__
    return run


def select_all(session, image):
    set_roi(session, image, SO.select_all(image))


def select_none(session, image):
    set_roi(session, image, None)


def restore_selection(session, image):
    st = state(session)
    roi = st.previous_roi.get(image.id) or st.last_roi
    if roi is None:
        return
    cur = image.roi
    image.roi = roi.copy()
    if cur is not None:
        st.previous_roi[image.id] = cur.copy()
    st._roi_seen[image.id] = image.roi


@_sel
def fit_spline(session, image):
    set_roi(session, image, SO.fit_spline(_need_roi(image)))


@_sel
def fit_circle(session, image):
    set_roi(session, image, SO.fit_circle(image, _need_roi(image)))


@_sel
def fit_ellipse(session, image):
    set_roi(session, image, SO.fit_ellipse(image, _need_roi(image)))


@_sel
def fit_rectangle(session, image):
    set_roi(session, image, SO.fit_rectangle(_need_roi(image)))


@_sel
def interpolate(session, image, interval=1.0, smooth=False, adjust=False):
    set_roi(session, image, SO.interpolate(_need_roi(image), float(interval), smooth, adjust))


@_sel
def convex_hull(session, image):
    set_roi(session, image, SO.convex_hull(_need_roi(image)))


@_sel
def make_inverse(session, image):
    set_roi(session, image, SO.make_inverse(image, image.roi))


def create_selection(session, image):
    try:
        roi = SO.create_selection(image, session.options.black_background)
    except SO.SelectionError as exc:
        raise CommandError(str(exc)) from None
    if roi is None:
        raise CommandError("No thresholded pixels")
    set_roi(session, image, roi)


def create_mask(session, image):
    existing = session.find_image("Mask")
    try:
        doc = SO.create_mask(image, image.roi, session.options.black_background,
                             existing=existing if (existing is not None and image.roi is not None)
                             else None)
    except SO.SelectionError as exc:
        raise CommandError(str(exc)) from None
    if doc is existing:
        session.image_updated(doc)
        return None
    return doc


def properties_fields(session, image):
    roi = image.roi if image is not None else None
    if roi is None:
        return [C.message("This command requires a selection.")]
    pos = roi.position
    return [C.string("Name:", roi.name or "", key="name"),
            C.string("Position:", "none" if pos == (0, 0, 0) else
                     (str(pos[1]) if pos[0] == 0 and pos[2] == 0 else f"{pos[0]},{pos[1]},"
                      f"{pos[2]}"), key="position"),
            C.string("Stroke color:", roi.stroke_color or "", key="stroke"),
            C.number("Width:", roi.stroke_width or 0, 1, key="width"),
            C.string("Fill color:", roi.fill_color or "none", key="fill"),
            C.checkbox("List coordinates", False, key="list")]


def _parse_position(text):
    t = str(text).strip().lower()
    if t in ("", "none", "0"):
        return (0, 0, 0)
    parts = [int(float(p)) for p in t.replace(" ", ",").split(",") if p]
    if len(parts) == 1:
        return (0, parts[0], 0)
    parts += [0] * (3 - len(parts))
    return tuple(parts[:3])


def _parse_color(text):
    from rsstatchem.core.imaging.scalebar import hex_color
    t = str(text).strip()
    if t.lower() in ("", "none"):
        return None
    if t.startswith("#"):
        return t
    c = hex_color(t.title()) or hex_color(t.capitalize())
    return c or t


def properties(session, image, name="", position="none", stroke="", width=0.0, fill="none",
               list=False):  # noqa: A002 - macro key
    roi = _need_roi(image)
    roi.name = name
    roi.position = _parse_position(position)
    roi.stroke_color = _parse_color(stroke)
    roi.stroke_width = float(width)
    roi.fill_color = _parse_color(fill)
    if list:
        xs, ys = G.float_polygon(roi)
        rt = M.ResultsTable("XY_Coordinates")
        for x, y in zip(xs, ys):
            rt.add_row({"X": x, "Y": y})
        return rt
    return None


@_sel
def scale_sel(session, image, x=1.0, y=1.0, centered=False):
    set_roi(session, image, SO.scale(_need_roi(image), float(x), float(y), centered))


@_sel
def rotate_sel(session, image, rotate=False, angle=15.0, **_):
    roi = _need_roi(image)
    if rotate:
        out = SO.rotate(roi, float(angle), image.width / 2.0, image.height / 2.0)
    else:
        out = SO.rotate(roi, float(angle))
    set_roi(session, image, out)


@_sel
def translate_sel(session, image, x=0.0, y=0.0):
    set_roi(session, image, SO.translate(_need_roi(image), float(x), float(y)))


@_sel
def enlarge(session, image, enlarge=1.0, pixel=False):
    n = float(enlarge)
    cal = image.calibration
    if cal.scaled() and not pixel:
        n = n / cal.pixel_width
    set_roi(session, image, SO.enlarge(image, _need_roi(image), round(n) if
                                       isinstance(image.roi, (S.RectRoi, S.OvalRoi)) else n))


@_sel
def make_band(session, image, band=15.0):
    n = float(band)
    cal = image.calibration
    if cal.scaled():
        n = n / cal.pixel_width
    set_roi(session, image, SO.make_band(image, _need_roi(image), n))


def specify_fields(session, image):
    roi = image.roi if image is not None else None
    x, y, w, h = (G.int_bounds(roi) if roi is not None else
                  (0, 0, image.width // 2 if image else 100, image.height // 2 if image else 100))
    return [C.number("Width:", w, 0, key="width"), C.number("Height:", h, 0, key="height"),
            C.number("X coordinate:", x, 0, key="x"), C.number("Y coordinate:", y, 0, key="y"),
            C.number("Slice:", image.current_index + 1 if image else 1, 0, key="slice",
                     integer=True),
            C.checkbox("Oval", isinstance(roi, S.OvalRoi), key="oval"),
            C.checkbox("Constrain (square/circle)", False, key="constrain"),
            C.checkbox("Centered", False, key="centered"),
            C.checkbox("Scaled units", False, key="scaled")]


@_sel
def specify(session, image, width=100.0, height=100.0, x=0.0, y=0.0, slice=1, oval=False,
            constrain=False, centered=False, scaled=False):  # noqa: A002
    if constrain:
        height = width
    roi = SO.specify(image, float(width), float(height), float(x), float(y), oval, centered,
                     scaled)
    if image.stack_size > 1 and 1 <= int(slice) <= image.stack_size:
        image.set_position(*image.position_of(int(slice) - 1))
    set_roi(session, image, roi)


def straighten_fields(session, image):
    title = image.title.rsplit(".", 1)[0] + "-1" if image else "Untitled"
    lw = 20
    if image is not None and isinstance(image.roi, S.LineRoi) and image.roi.width > 1:
        lw = int(image.roi.width)
    fields = [C.string("Title:", title, key="title"),
              C.number("Line Width:", lw, 0, key="line", integer=True, units="pixels")]
    if image is not None and image.stack_size > 1:
        fields.append(C.checkbox("Process Entire Stack", False, key="process"))
    return fields


def straighten(session, image, title="", line=20, process=False):
    roi = image.roi
    if roi is None or not roi.is_line():
        raise CommandError("Line selection required")
    try:
        doc = SO.straighten(image, roi, int(line), process)
    except SO.SelectionError as exc:
        raise CommandError(str(exc)) from None
    if title:
        doc.title = title
    if not doc.is_rgb:
        doc.luts = [image.lut.copy()]
        doc.reset_display_range()
    return doc


@_sel
def to_bounding_box(session, image):
    set_roi(session, image, SO.to_bounding_box(_need_roi(image)))


@_sel
def line_to_area(session, image):
    set_roi(session, image, SO.line_to_area(_need_roi(image)))


@_sel
def area_to_line(session, image):
    set_roi(session, image, SO.area_to_line(_need_roi(image)))


def image_to_selection_fields(session, image):
    titles = [im.title for im in session.images if im is not image] or ["(none)"]
    return [C.choice("Image:", titles, titles[0], key="image"),
            C.number("Opacity (0-100%):", 100, 0, key="opacity"),
            C.checkbox("Zero transparent", False, key="zero")]


def image_to_selection(session, img, opacity=100, zero=False, **kw):
    """An image selection (ImageJ ImageRoi): a rectangle carrying the rendered source image
    (``roi.image_data``) and its opacity, placed at the top-left corner."""
    from rsstatchem.core.imaging.lut import render

    src = session.find_image(kw.get("image"))
    if src is None:
        raise CommandError("Image not found")
    roi = S.RectRoi(0, 0, src.width, src.height, name=src.title)
    roi.properties["image_roi"] = src.title
    roi.properties["opacity"] = str(opacity)
    roi.properties["zero_transparent"] = str(bool(zero))
    roi.image_data = render(src)
    set_roi(session, img, roi)


def add_to_manager(session, image):
    roi = _need_roi(image)
    r = roi.copy()
    if image.stack_size > 1 and r.position == (0, 0, 0):
        if image.hyperstack or image.is_composite:
            c, z, t = image.position
            r.position = (c + 1, z + 1, t + 1)
        else:
            r.position = (0, image.current_index + 1, 0)
    from rsstatchem.ui.imagej.frames.roi_manager import unique_name
    r.name = unique_name(session, r, image)
    session.roi_manager.append(r)
    _manager_changed(session, image, open_window=True)


def _manager_changed(session, image=None, open_window=False):
    session.emit("roi_manager", image)
    bar = bar_for(session)
    if bar is not None:
        from rsstatchem.ui.imagej.frames.roi_manager import manager_window
        win = manager_window(bar, create=open_window or bool(session.roi_manager))
        if win is not None:
            win.refresh()
            if open_window:
                win.show()


# ================================================================================== Fiji
def coloc_fields(session, image):
    titles = [im.title for im in session.images] or ["(none)"]
    t1 = titles[0]
    t2 = titles[1] if len(titles) > 1 else titles[0]
    return [C.choice("Channel_1", titles, t1, key="channel_1"),
            C.choice("Channel_2", titles, t2, key="channel_2"),
            C.choice("ROI_or_mask", ["<None>", "ROI(s) in channel 1", "ROI(s) in channel 2"] +
                     titles, "<None>", key="roi_or_mask"),
            C.choice("Threshold_regression", ("Costes", "Bisection"), "Bisection",
                     key="threshold_regression"),
            C.checkbox("Display_Images_in_Result", False, key="display_images_in_result"),
            C.checkbox("Li_Histogram_Channel_1", False, key="li_histogram_channel_1"),
            C.checkbox("Li_Histogram_Channel_2", False, key="li_histogram_channel_2"),
            C.checkbox("Li_ICQ", True, key="li_icq"),
            C.checkbox("Spearman's_Rank_Correlation", True,
                       key="spearman's_rank_correlation"),
            C.checkbox("Manders'_Correlation", True, key="manders'_correlation"),
            C.checkbox("Kendall's_Tau_Rank_Correlation", False,
                       key="kendall's_tau_rank_correlation"),
            C.checkbox("2D_Instensity_Histogram", True, key="2d_instensity_histogram"),
            C.checkbox("Costes'_Significance_Test", True, key="costes'_significance_test"),
            C.number("PSF", 3, 0, key="psf", integer=True),
            C.number("Costes_randomisations", 10, 0, key="costes_randomisations",
                     integer=True)]


def coloc2(session, image, **kw):
    from rsstatchem.core.imaging.fiji import coloc as CL
    a = session.find_image(kw.get("channel_1"))
    b = session.find_image(kw.get("channel_2"))
    if a is None or b is None:
        raise CommandError("Coloc 2 needs two open images (Channel_1, Channel_2)")
    if a.width != b.width or a.height != b.height:
        raise CommandError("Both images must have the same size.")
    if a.is_rgb or b.is_rgb:
        raise CommandError("Coloc 2 works on 8, 16 or 32-bit images (split RGB channels "
                           "first)")
    choice = kw.get("roi_or_mask", "<None>")
    mask = None
    if choice == "ROI(s) in channel 1" and a.roi is not None and a.roi.is_area():
        mask = a.roi.get_mask((a.height, a.width))
    elif choice == "ROI(s) in channel 2" and b.roi is not None and b.roi.is_area():
        mask = b.roi.get_mask((b.height, b.width))
    elif choice not in ("<None>", "ROI(s) in channel 1", "ROI(s) in channel 2"):
        mimg = session.find_image(choice)
        if mimg is not None:
            mask = mimg.plane() > 0
    pa = a.plane()
    pb = b.plane()
    res = CL.coloc(pa, pb, mask, psf=int(kw.get("psf", 3)),
                   randomizations=int(kw.get("costes_randomisations", 10))
                   if kw.get("costes'_significance_test", True) else 0,
                   li=kw.get("li_icq", True), spearman=kw.get("spearman's_rank_correlation",
                                                               True),
                   kendall=kw.get("kendall's_tau_rank_correlation", False),
                   manders_=kw.get("manders'_correlation", True))
    rt = res.table("Coloc 2")
    session.log(f"Coloc 2 version 3 (RS-StatChem) for {a.title} and {b.title}")
    for line in res.text().splitlines():
        session.log(line)
    out = [rt]
    if kw.get("2d_instensity_histogram", True):
        hist = CL.histogram_image(res)
        doc = ImageDoc.from_array(hist, "YX", f"2D Histogram of {a.title} and {b.title}")
        from rsstatchem.core.imaging.lut import named_lut
        try:
            doc.luts = [named_lut("Fire")]
        except (KeyError, ValueError):
            pass
        out.append(doc)
    return out


def skeleton_fields(session, image):
    return [C.choice("Prune cycle method:", ("none",), "none", key="prune"),
            C.checkbox("Prune ends", False, key="prune_ends"),
            C.checkbox("Calculate largest shortest path", False, key="calculate"),
            C.checkbox("Show detailed info", False, key="show"),
            C.checkbox("Display labeled skeletons", False, key="display")]


def analyze_skeleton(session, image, prune="none", prune_ends=False, calculate=False,
                     show=False, display=False):
    from rsstatchem.core.imaging.fiji import skeleton_analysis as SK
    if image.bit_depth != 8 or image.is_rgb:
        raise CommandError("8-bit binary (black and white only) image required.")
    vol = image.pixels[image.position[2], :, image.position[0]]
    vals = np.unique(vol)
    if not np.all(np.isin(vals, (0, 255))):
        raise CommandError("8-bit binary (black and white only) image required.")
    skel = vol == 255          # skeleton pixels are 255 whatever the LUT, as in Fiji
    if prune_ends:
        skel = _prune_ends(skel)
    cal = image.calibration
    res = SK.analyze_skeleton(skel, (cal.pixel_depth, cal.pixel_height, cal.pixel_width))
    rt = res.results_table()
    rt.title = "Results"
    session.results = rt
    out = [rt]
    if show:
        bt = res.branch_table()
        out.append(bt)
    tagged = res.tagged
    axes = "YX" if tagged.ndim == 2 else "ZYX"
    doc = ImageDoc.from_array(tagged, axes, "Tagged skeleton")
    doc.calibration = cal.copy()
    out.append(doc)
    if display:
        from scipy import ndimage
        lab, _ = ndimage.label(skel if skel.ndim == 3 else skel,
                               structure=np.ones((3,) * skel.ndim))
        out.append(ImageDoc.from_array(lab.astype(np.float32).squeeze(),
                                       "YX" if lab.squeeze().ndim == 2 else "ZYX",
                                       "Labeled skeletons"))
    return out


def _prune_ends(skel):
    from scipy import ndimage
    """Remove every branch that ends in an end point (junction voxels are kept)."""
    s = skel.copy()
    fp = np.ones((3,) * s.ndim, int)
    nb0 = ndimage.convolve(s.astype(int), fp, mode="constant") - s
    junction = s & (nb0 > 2)
    for _ in range(s.size):
        nb = ndimage.convolve(s.astype(int), fp, mode="constant") - s
        ends = s & (nb <= 1) & ~junction
        if not ends.any():
            break
        s &= ~ends
    return s


# ================================================================================== UI hooks
def _show_plot(session, plot):
    st = state(session)
    st.plots.append(plot)
    del st.plots[:-20]
    bar = bar_for(session)
    if bar is not None:
        from rsstatchem.ui.imagej.frames.plot_window import show_plot
        show_plot(bar, plot)


def _show_histogram(session, image, data, source):
    st = state(session)
    st.histograms.append(data)
    del st.histograms[:-20]
    bar = bar_for(session)
    if bar is not None:
        from rsstatchem.ui.imagej.frames.histogram_window import show_histogram
        show_histogram(bar, image, data, source)


def _ui_roi_manager(bar, _cmd):
    from rsstatchem.ui.imagej.frames.roi_manager import manager_window
    win = manager_window(bar, create=True)
    win.show()
    win.raise_()
    return win


def _ui_cell_counter(bar, _cmd):
    from rsstatchem.ui.imagej.frames.point_counter import counter_window
    win = counter_window(bar, create=True)
    win.show()
    win.raise_()
    return win


def _ui_sync(bar, _cmd):
    from rsstatchem.ui.imagej.frames.plot_window import sync_window
    win = sync_window(bar)
    win.show()
    win.raise_()
    return win


def _cell_counter_headless(session, image, **_):
    """Macro form of the Cell Counter: count the multi-point counters of the selection."""
    roi = image.roi
    if not isinstance(roi, S.PointRoi):
        raise CommandError("Multi-point selection (counters) required")
    counters = np.asarray(roi.counters if roi.counters is not None else
                          np.zeros(len(roi.xs)), int)
    rt = M.ResultsTable("Counts")
    for c in sorted(set(counters.tolist())) or [0]:
        rt.add_row({"Type": c + 1, "Count": int((counters == c).sum())})
    return rt


# ================================================================================== register
def _reg(registry, label, func, fields=(), **kw):
    registry.register(Command(label, func, fields if callable(fields) else list(fields),
                              **kw), replace=True)


def register(registry) -> None:
    """Add every Analyze menu command, Edit > Selection and the Fiji analysis plugins."""
    from rsstatchem.ui.imagej import windows as W

    try:
        from rsstatchem.ui.imagej.frames.results_window import ResultsWindow
        W.TABLE_WINDOW_FACTORY = ResultsWindow
    except ImportError:  # pragma: no cover - Qt missing
        pass
    N = dict(undoable=False)
    _reg(registry, "Measure", measure, types=ALL_TYPES, **N)
    _reg(registry, "Analyze Particles...", analyze_particles, particles_fields,
         supports_stack=True, types=GRAY_TYPES, **N)
    _reg(registry, "Summarize", summarize, needs_image=False, **N)
    _reg(registry, "Distribution...", distribution, distribution_fields, needs_image=False,
         **N)
    _reg(registry, "Label", label, roi="required", undoable=False)
    _reg(registry, "Clear Results", clear_results, needs_image=False, **N)
    _reg(registry, "Set Measurements...", set_measurements, set_measurements_fields,
         needs_image=False, **N)
    _reg(registry, "Set Scale...", set_scale, set_scale_fields, **N)
    _reg(registry, "Calibrate...", calibrate_cmd, calibrate_fields, types=GRAY_TYPES, **N)
    _reg(registry, "Histogram", histogram_cmd, histogram_fields, **N)
    _reg(registry, "Plot Profile", plot_profile, roi="required", **N)
    _reg(registry, "Surface Plot...", surface_plot, surface_fields, **N)
    _reg(registry, "Select First Lane", gel_first, roi="required", **N)
    _reg(registry, "Select Next Lane", gel_next, roi="required", **N)
    _reg(registry, "Plot Lanes", gel_plot, **N)
    _reg(registry, "Re-plot Lanes", gel_replot, **N)
    _reg(registry, "Reset", gel_reset, needs_image=False, **N)
    _reg(registry, "Label Peaks", gel_label, needs_image=False, **N)
    _reg(registry, "Gel Analyzer Options...", gel_options, gel_options_fields,
         needs_image=False, **N)
    _reg(registry, "Save XY Coordinates...", save_xy, save_xy_fields, **N)
    _reg(registry, "Fractal Box Count...", fractal, fractal_fields, types=(GRAY8, COLOR_256),
         **N)
    _reg(registry, "Analyze Line Graph", line_graph, types=(GRAY8, COLOR_256), **N)
    _reg(registry, "Curve Fitting...", curve_fit, curve_fit_fields, needs_image=False, **N)
    registry.register(Command("ROI Manager...", None, needs_image=False, undoable=False,
                              recordable=True, ui_action=_ui_roi_manager), replace=True)
    _reg(registry, "Scale Bar...", scale_bar, scale_bar_fields, **N)
    _reg(registry, "Calibration Bar...", calibration_bar, calibration_bar_fields,
         types=GRAY_TYPES, **N)
    registry.register(Command("Synchronize Windows", None, needs_image=False, undoable=False,
                              ui_action=_ui_sync), replace=True)
    _reg(registry, "Grid...", grid, grid_fields, **N)
    # Edit > Selection
    _reg(registry, "Select All", select_all, **N)
    _reg(registry, "Select None", select_none, **N)
    _reg(registry, "Restore Selection", restore_selection, **N)
    _reg(registry, "Fit Spline", fit_spline, roi="required", **N)
    _reg(registry, "Fit Circle", fit_circle, roi="required", **N)
    _reg(registry, "Fit Ellipse", fit_ellipse, roi="required", **N)
    _reg(registry, "Fit Rectangle", fit_rectangle, roi="area", **N)
    _reg(registry, "Interpolate", interpolate,
         [C.number("Interval:", 1.0, 1, units="pixel", key="interval"),
          C.checkbox("Smooth", False, key="smooth"),
          C.checkbox("Adjust interval to match", False, key="adjust")], roi="required", **N)
    _reg(registry, "Convex Hull", convex_hull, roi="required", **N)
    _reg(registry, "Make Inverse", make_inverse, **N)
    _reg(registry, "Create Selection", create_selection, **N)
    _reg(registry, "Create Mask", create_mask, **N)
    _reg(registry, "Properties... ", properties, properties_fields, roi="required", **N)
    _reg(registry, "Scale... ", scale_sel,
         [C.number("X scale factor:", 1.0, 2, key="x"),
          C.number("Y scale factor:", 1.0, 2, key="y"),
          C.checkbox("Centered", False, key="centered")], roi="required", **N)
    _reg(registry, "Rotate...", rotate_sel,
         [C.checkbox("Rotate around image center", False, key="rotate"),
          C.number("Angle:", 15.0, 1, units="degrees", key="angle")], roi="required", **N)
    _reg(registry, "Translate... ", translate_sel,
         [C.number("X offset:", 0.0, 1, units="pixels", key="x"),
          C.number("Y offset:", 0.0, 1, units="pixels", key="y")], roi="required", **N)
    _reg(registry, "Enlarge...", enlarge,
         [C.number("Enlarge by", 15.0, 1, key="enlarge"),
          C.checkbox("Pixel units", True, key="pixel")], roi="area", **N)
    _reg(registry, "Make Band...", make_band, [C.number("Band Size:", 15.0, 1, key="band")],
         roi="area", **N)
    _reg(registry, "Specify...", specify, specify_fields, **N)
    _reg(registry, "Straighten...", straighten, straighten_fields, roi="line", **N)
    _reg(registry, "To Bounding Box", to_bounding_box, roi="required", **N)
    _reg(registry, "Line to Area", line_to_area, roi="line", **N)
    _reg(registry, "Area to Line", area_to_line, roi="area", **N)
    _reg(registry, "Image to Selection...", image_to_selection, image_to_selection_fields, **N)
    _reg(registry, "Add to Manager", add_to_manager, roi="required", **N)
    # Fiji analysis plugins (not in ImageJ 1.54's menus; found with Find Commands)
    registry.register(Command("Cell Counter", _cell_counter_headless, needs_image=True,
                              undoable=False, ui_action=_ui_cell_counter), replace=True)
    _reg(registry, "Coloc 2", coloc2, coloc_fields, **N)
    _reg(registry, "Analyze Skeleton (2D/3D)", analyze_skeleton, skeleton_fields,
         types=(GRAY8,), **N)


FIJI_COMMANDS = {"Cell Counter": "Plugins>Analyze", "Coloc 2": "Analyze>Colocalization",
                 "Analyze Skeleton (2D/3D)": "Analyze>Skeleton"}


def selftest() -> None:
    """Headless checks: Measure, Analyze Particles and ROI masks on a synthetic image."""
    from rsstatchem.core.imaging.image import Threshold
    from rsstatchem.core.imaging.roi.mask import polygon_mask
    from rsstatchem.core.imaging.session import Session

    s = Session()
    register(s.registry)
    a = np.zeros((32, 32), np.uint8)
    a[4:10, 4:10] = 200
    a[15:25, 12:20] = 150
    doc = ImageDoc.from_array(a, "YX", "selftest")
    s.add_image(doc)
    doc.threshold = Threshold(100, 255)
    s.options.measurements = ("area", "mean", "perimeter", "display")
    s.run("Analyze Particles...", "display clear summarize")
    assert s.results.size == 2 and s.results.get_value("Area", 1) == 80
    assert s.tables["Summary"].get_value("Count", 0) == 2
    doc.roi = S.RectRoi(2, 2, 10, 10)
    s.run("Measure")
    assert abs(s.results.get_value("Mean", 2) - 72.0) < 1e-9
    assert polygon_mask([1, 5, 5, 1], [1, 1, 5, 5], (8, 8)).sum() == 16
    s.run("Convex Hull")
    assert doc.roi.type == S.POLYGON
