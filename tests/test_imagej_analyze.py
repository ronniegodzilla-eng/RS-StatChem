"""WP-C: Analyze menu (Measure, Analyze Particles, Summarize, Set Scale, Calibrate, Histogram,
Plot Profile, Surface Plot, Gels, Tools), Results window, ROI Manager, plot/histogram windows,
Cell Counter, Coloc 2 and Analyze Skeleton; parity with ImageJ 1.54p goldens."""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest
from ij_golden import blobs, golden_csv, input_path, thresholds

from rsstatchem.core.imaging import measure as M
from rsstatchem.core.imaging import particles as P
from rsstatchem.core.imaging.image import GRAY16, GRAY32, ImageDoc, Threshold
from rsstatchem.core.imaging.io import open_image
from rsstatchem.core.imaging.roi import shapes as S
from rsstatchem.core.imaging.session import Session
from rsstatchem.ui.imagej import commands_analyze as CA

ALL_KEYS = ("area mean standard modal min centroid center perimeter bounding fit shape "
            "feret's integrated median skewness kurtosis area_fraction stack display").split()
ALL = M.flags_from_keys(ALL_KEYS)
LIMIT_KEYS = "area mean standard min centroid integrated area_fraction limit display".split()
SHOTS = Path(os.environ.get(
    "IJ_C_SHOTS", "/private/tmp/claude-501/-Users-ronisaputra-Aplikasi-statistik/"
                  "a8ca05d2-7214-4d46-8f3d-8e0d4372bfcd/scratchpad/ij_c"))


def _compare(golden, ours, rel=1e-6):
    for h in golden.headings:
        if h == "Label":
            assert [golden.get_label(r) for r in range(golden.size)] == \
                [ours.get_label(r) for r in range(ours.size)]
            continue
        assert ours.column_exists(h), h
        g, o = golden.column(h), ours.column(h)
        both_nan = np.isnan(g) & np.isnan(o)
        assert np.all(both_nan | (np.abs(g - o) <= rel * np.maximum(np.abs(g), 1))), \
            (h, g[~both_nan][:3], o[~both_nan][:3])


ROIS = {"whole": None, "rect": S.RectRoi(20, 30, 80, 60), "oval": S.OvalRoi(100, 50, 71, 90),
        "polygon": S.PolygonRoi([10, 120, 90, 20], [10, 30, 140, 100]),
        "line": S.LineRoi(10, 20, 200, 180)}


@pytest.mark.parametrize("pass_", ["uncalibrated", "calibrated", "limit"])
def test_measure_matches_imagej(pass_):
    im = blobs()
    keys = ALL_KEYS
    if pass_ == "calibrated":
        im.calibration.set_scale(10, 2.5, "um")
    if pass_ == "limit":
        keys = LIMIT_KEYS
        im.threshold = Threshold(117, 255)   # the macro's setAutoThreshold saw the polygon
    for name, roi in ROIS.items():
        if name == "line" and pass_ != "uncalibrated":
            continue
        im.roi = roi
        an = M.Analyzer(im, M.flags_from_keys(keys), precision=9)
        an.measure()
        golden = golden_csv("measure", f"{pass_}_{name}.csv")
        assert set(an.rt.headings) == set(golden.headings), name
        _compare(golden, an.rt)


VARIANTS = {"default": {}, "exclude": {"exclude": True}, "include": {"include_holes": True},
            "size": {"min_size": 100}, "circ": {"min_circularity": 0.85}}


@pytest.mark.parametrize("stem", ["blobs", "particles8"])
@pytest.mark.parametrize("variant", list(VARIANTS))
def test_analyze_particles_matches_imagej(stem, variant):
    im = open_image(input_path("blobs.gif" if stem == "blobs" else "particles8.tif"))
    im.threshold = Threshold(*thresholds()[("blobs.gif", "Default", False)]) \
        if stem == "blobs" else Threshold(128, 255)
    opt = P.ParticleOptions(display=True, clear=True, summarize=True, **VARIANTS[variant])
    res = P.analyze_particles(im, ALL, opt, precision=9)
    golden = golden_csv("particles", f"{stem}_{variant}.csv")
    assert res.results.size == golden.size
    _compare(golden, res.results)
    gs = golden_csv("particles", f"{stem}_{variant}_summary.csv")
    for h in gs.headings:
        gv, ov = gs.get_value(h, 0), res.summary.get_value(h, 0)
        if isinstance(gv, str):
            assert gv == ov
        else:
            assert ov == pytest.approx(gv, rel=1e-6, nan_ok=True), h


def test_blobs_particle_counts():
    im = blobs()
    im.threshold = Threshold(126, 255)
    counts = {}
    for name, kw in VARIANTS.items():
        res = P.analyze_particles(im, M.AREA, P.ParticleOptions(display=True, **kw))
        counts[name] = (res.count, float(res.results.column("Area").sum()))
    assert counts["default"] == (64, 22243.0)
    assert counts["exclude"][0] == 46 and counts["size"][0] == 52 and counts["circ"][0] == 43


@pytest.fixture
def session():
    s = Session()
    CA.register(s.registry)
    im = blobs()
    s.add_image(im)
    im.threshold = Threshold(126, 255)
    return s


def test_measure_command_and_set_measurements(session):
    s = session
    im = s.active
    s.run("Set Measurements...", "area mean min centroid perimeter display redirect=None "
                                 "decimal=4")
    assert s.options.measurements == ("area", "mean", "min", "centroid", "perimeter", "display")
    im.roi = S.RectRoi(20, 30, 80, 60)
    s.run("Measure")
    rt = s.results
    assert rt.size == 1 and rt.get_label(0) == "blobs.gif" and rt.get_value("Area", 0) == 4800
    assert rt.format_value("Mean", 0) == "103.8233"
    s.run("Measure")
    s.run("Summarize")
    assert rt.size == 6 and rt.get_label(5) == "Max"
    s.run("Measure")                        # the summary rows are replaced
    assert rt.size == 3
    s.run("Set Measurements...", "area mean invert scientific nan decimal=3")
    im.roi = S.PointRoi([10], [20])
    s.run("Measure")
    assert rt.get_value("Y", rt.size - 1) == 233        # inverted y: 254 - 20 - 1
    s.run("Clear Results")
    assert rt.size == 0
    # redirect to a second image
    other = ImageDoc.from_array(np.full((254, 256), 7, np.uint8), "YX", "seven")
    s.add_image(other)
    s.set_active(im)
    s.run("Set Measurements...", "mean redirect=seven decimal=3")
    im.roi = S.RectRoi(0, 0, 10, 10)
    s.run("Measure")
    assert s.results.get_value("Mean", 0) == 7
    s.run("Set Measurements...", "area mean add decimal=3")
    s.run("Measure")
    assert im.overlay is not None and len(im.overlay) == 1


def test_measure_line_point_angle(session):
    s = session
    im = s.active
    s.run("Set Measurements...", "area mean decimal=3")
    im.roi = S.LineRoi(10, 20, 200, 180)
    s.run("Measure")
    assert s.results.get_value("Length", 0) == pytest.approx(248.394846967)
    assert s.results.get_value("Angle", 0) == pytest.approx(-40.100907546)
    assert s.results.get_value("Mean", 0) == pytest.approx(74.506826347)
    im.roi = S.PolygonRoi([0, 10, 10], [0, 0, -10], type=S.ANGLE)
    s.run("Measure")
    assert s.results.get_value("Angle", 1) == pytest.approx(90)
    im.roi = S.PointRoi([5, 6], [7, 8], counters=np.array([0, 1]))
    s.run("Measure")
    assert s.results.size == 4 and s.results.get_value("Counter", 3) == 1


def test_analyze_particles_command_outputs(session):
    s = session
    im = s.active
    s.options.measurements = ("area", "mean", "display")
    out = s.run("Analyze Particles...", "size=0-Infinity show=[Count Masks] display clear "
                                        "summarize add record")
    rt = s.results
    assert rt.size == 64 and rt.column_exists("XStart")
    count_mask = [o for o in out if isinstance(o, ImageDoc)][0]
    assert count_mask.title == "Count Masks of blobs.gif" and count_mask.type == GRAY16
    assert count_mask.plane().max() == 64
    assert (count_mask.plane() > 0).sum() == 22243
    assert len(s.roi_manager) == 64 and s.tables["Summary"].get_value("Count", 0) == 64
    s.set_active(im)
    masks = s.run("Analyze Particles...", "size=100-Infinity show=Masks exclude clear")
    m = [o for o in masks if isinstance(o, ImageDoc)][0]
    assert m.title == "Mask of blobs.gif" and m.is_inverted_lut()
    s.set_active(im)
    s.run("Analyze Particles...", "show=Outlines display clear")
    outl = s.images[-1]
    assert outl.title == "Drawing of blobs.gif" and (outl.plane() == 0).sum() > 1000
    s.set_active(im)
    s.run("Analyze Particles...", "show=[Overlay Masks] display clear")
    assert im.overlay is not None and len(im.overlay) == 64
    assert im.overlay.rois[0].fill_color is not None
    s.run("Analyze Particles...", "show=Ellipses display clear")
    s.set_active(im)
    s.run("Analyze Particles...", "display clear composite include")
    assert s.results.size == 64
    # calibrated sizes and circularity strings
    im.calibration.set_scale(2, 1, "um")
    s.run("Analyze Particles...", "size=25-Infinity circularity=0.85-1.00 display clear")
    assert 0 < s.results.size < 64
    # no threshold on a non-binary image is an error
    im.threshold = None
    from rsstatchem.core.imaging.commands import CommandError
    with pytest.raises(CommandError):
        s.run("Analyze Particles...", "display clear")


def test_particles_binary_without_threshold_and_stack():
    a = np.zeros((3, 40, 40), np.uint8)
    a[:, 5:10, 5:10] = 255
    a[1, 20:30, 20:30] = 255
    im = ImageDoc.from_array(a, "ZYX", "stack")
    s = Session()
    CA.register(s.registry)
    s.add_image(im)
    s.options.black_background = True
    s.options.measurements = ("area", "stack")
    s.run("Analyze Particles...", "display clear summarize stack")
    assert s.results.size == 4
    assert s.results.column("Slice").tolist() == [1, 2, 2, 3]
    summ = s.tables["Summary of stack"]
    assert summ.column("Count").tolist() == [1, 2, 1]


def test_set_scale_calibrate_histogram_profile(session):
    s = session
    im = s.active
    im.roi = S.LineRoi(0, 0, 100, 0)
    s.run("Set Scale...", "distance=100 known=25 unit=um")
    assert im.calibration.pixel_width == 0.25 and im.calibration.unit == "µm"
    s.run("Set Scale...", "distance=0 known=0 unit=pixel")
    assert not im.calibration.scaled()
    s.run("Calibrate...", "function=[Straight Line] unit=OD text1=[0 255] text2=[0 2.55]")
    assert im.calibration.cvalue(100) == pytest.approx(1.0)
    s.run("Calibrate...", "function=[Uncalibrated OD]")
    assert im.calibration.function == 21
    s.run("Calibrate...", "function=[Pixel Inverter]")
    assert im.calibration.cvalue(0) == 255
    s.run("Calibrate...", "function=[2nd Degree Polynomial] text1=[0 50 100 200] "
                          "text2=[1 3.5 11 41]")
    assert im.calibration.cvalue(50) == pytest.approx(3.5, abs=1e-6)
    s.run("Calibrate...", "function=Rodbard text1=[10 50 100 150 200 250] "
                          "text2=[0.1 0.5 1.4 2.1 2.5 2.7]")
    assert im.calibration.function == 7
    s.run("Calibrate...", "function=None")
    im.roi = None
    s.run("Histogram")
    data = CA.state(s).histograms[-1]
    assert data.n == 65024 and data.counts.sum() == 65024 and data.mode == 48
    assert "Mean: 103.269" in data.stats_text()
    im.roi = S.RectRoi(10, 10, 30, 5)
    s.run("Plot Profile")
    plot = CA.state(s).plots[-1]
    assert len(plot.series[0]["y"]) == 30
    assert plot.series[0]["y"][0] == pytest.approx(im.plane()[10:15, 10].mean())
    im.roi = S.LineRoi(10, 20, 200, 180)
    s.run("Plot Profile")
    prof = CA.state(s).plots[-1].series[0]["y"]
    assert len(prof) == 249 and prof.min() == pytest.approx(17.148803711)
    # 16- and 32-bit histograms with bins and range
    f = ImageDoc.from_array(np.linspace(0, 1, 100, dtype=np.float32).reshape(10, 10), "YX")
    s.add_image(f)
    s.run("Histogram", "bins=10 x_min=0 x_max=1 y_max=Auto")
    d = CA.state(s).histograms[-1]
    assert d.n_bins == 10 and d.counts.sum() == 100 and d.bin_width == pytest.approx(0.1)
    assert f.type == GRAY32


def test_distribution_label_surface_tools(session, tmp_path):
    s = session
    im = s.active
    s.options.measurements = ("area", "mean", "display")
    s.run("Analyze Particles...", "display clear")
    s.run("Distribution...", "parameter=Area or=8 and=0-0")
    d = CA.state(s).histograms[-1]
    assert d.n_bins == 8 and d.counts.sum() == 64
    im.roi = S.RectRoi(10, 10, 30, 30)
    s.run("Label")
    assert im.changed
    surf = s.run("Surface Plot...", "polygon=50 shade draw_axis smooth")
    assert surf.is_rgb and surf.width > 100
    s.set_active(im)                         # new images become active, as in ImageJ
    s.run("Scale Bar...", "width=50 height=4 font=14 color=White background=None "
                          "location=[Lower Right] overlay")
    assert any(r.name.startswith("|SB|") for r in im.overlay)
    s.run("Grid...", "grid=Crosses area=1000 color=Cyan")
    assert any(r.name == "grid" for r in im.overlay)
    cb = s.run("Calibration Bar...", "location=[Upper Right] fill=White label=Black number=5 "
                                     "decimal=0 font=12 zoom=1")
    assert cb.is_rgb
    s.set_active(im)
    s.run("Fractal Box Count...", "box=2,4,8,16,32 black")
    assert 1.0 < s.results.get_value("D", 0) < 2.2
    path = tmp_path / "xy.txt"
    s.set_active(im)
    s.run("Save XY Coordinates...", f"background=0 save=[{path}]")
    assert path.read_text().count("\n") > 1000
    s.run("Curve Fitting...", "function=[Straight Line] x=[0 1 2 3] y=[1 3 5 7]")
    assert any("b = 2" in line for line in s.log_lines)


def test_gels_workflow():
    a = np.full((100, 60), 200, np.uint8)
    a[20:30, 5:20] = 50
    a[60:70, 5:20] = 100
    a[40:50, 30:45] = 60
    im = ImageDoc.from_array(a, "YX", "gel")
    s = Session()
    CA.register(s.registry)
    s.add_image(im)
    im.roi = S.RectRoi(5, 0, 15, 100)
    s.run("Select First Lane")
    im.roi = S.RectRoi(30, 0, 15, 100)
    s.run("Select Next Lane")
    s.run("Plot Lanes")
    gel = CA.state(s).gel
    assert len(gel.lanes) == 2 and len(gel.profiles[0]) == 100
    assert len([p for p in gel.peaks if p[0] == 0]) == 2
    rt = s.run("Label Peaks")
    assert rt.size == 3 and abs(rt.column("Percent").sum() - 100) < 1e-6
    s.run("Reset")
    assert not CA.state(s).gel.lanes


def test_selection_commands_and_restore(session):
    s = session
    im = s.active
    s.run("Select All")
    assert im.roi.bounds() == (0, 0, 256, 254)
    im.roi = S.RectRoi(10, 10, 50, 40)
    s.run("Enlarge...", "enlarge=5 pixel")
    assert im.roi.bounds() == (5, 5, 60, 50)
    s.run("Restore Selection")
    assert im.roi.bounds() == (10, 10, 50, 40)
    s.run("Rotate...", "angle=90")
    s.run("Make Inverse")
    assert isinstance(im.roi, S.CompositeRoi)
    s.run("Select None")
    assert im.roi is None
    s.run("Create Selection")
    assert im.roi.get_mask((254, 256)).sum() == (im.plane() >= 126).sum()
    s.run("Fit Ellipse")
    assert isinstance(im.roi, S.EllipseRoi)
    s.run("To Bounding Box")
    assert isinstance(im.roi, S.RectRoi)
    s.run("Specify...", "width=20 height=20 x=50 y=50 oval centered")
    assert isinstance(im.roi, S.OvalRoi) and im.roi.x == 40
    s.run("Properties... ", "name=cell position=none stroke=red width=2 fill=none")
    assert im.roi.name == "cell" and im.roi.stroke_color == "#ff0000"
    s.run("Scale... ", "x=2 y=2 centered")
    s.run("Add to Manager")
    assert s.roi_manager[-1].name == "cell"
    mask = s.run("Create Mask")
    assert mask.title == "Mask"
    s.set_active(im)
    im.roi = S.LineRoi(10, 50, 100, 50)
    st = s.run("Straighten...", "title=s line=10")
    assert st.height == 10
    s.set_active(im)
    im.roi = S.LineRoi(10, 50, 100, 50)
    s.run("Line to Area")
    assert im.roi.is_area()
    s.run("Area to Line")
    assert im.roi.is_line()
    im.roi = S.PolygonRoi([10, 60, 40], [10, 20, 60])
    s.run("Interpolate", "interval=2 smooth")
    s.run("Fit Spline")
    s.run("Convex Hull")
    assert im.roi.type == S.POLYGON
    im.roi = S.RectRoi(0, 0, 10, 10)
    s.run("Make Band...", "band=5")
    assert im.roi.get_mask((254, 256)).sum() > 100
    other = ImageDoc.new(20, 20, fill=128, title="small")
    s.add_image(other)
    s.set_active(im)
    s.run("Image to Selection...", "image=small opacity=50")
    assert im.roi.image_data.shape == (20, 20, 3)
    assert im.can_undo()


def test_roi_manager_model(session, tmp_path):
    from rsstatchem.ui.imagej.frames import roi_manager as RM
    s = session
    im = s.active
    for r in (S.RectRoi(10, 10, 30, 30), S.OvalRoi(50, 50, 40, 30),
              S.PolygonRoi([100, 150, 120], [100, 110, 160])):
        im.roi = r
        RM.add(s, im)
    assert [r.name for r in s.roi_manager][0] == "0025-0025"
    s.options.measurements = ("area", "mean")
    rt = RM.measure(s, im)
    assert rt.size == 3 and rt.get_value("Area", 0) == 900
    mm = RM.multi_measure(s, im)
    assert mm.size == 1 and mm.column_exists("Area(0025-0025)")
    comb = RM.combine(s, [0, 1], "or")
    assert comb.get_mask((254, 256)).sum() == 900 + S.OvalRoi(50, 50, 40, 30).get_mask(
        (254, 256)).sum()
    assert RM.combine(s, [0, 1], "and").get_mask((254, 256)).sum() == 0
    s.roi_manager.append(comb)
    RM.split(s, 3)
    assert len(s.roi_manager) == 6
    RM.rename(s, 0, "zz")
    RM.sort(s)
    assert s.roi_manager[-1].name == "zz"
    RM.translate(s, 5, 5, [0])
    RM.set_properties(s, [0], stroke="#00ff00", width=2)
    assert s.roi_manager[0].stroke_color == "#00ff00"
    p = RM.save(s, tmp_path / "RoiSet.zip")
    n = len(s.roi_manager)
    RM.delete(s, None)
    assert not s.roi_manager
    RM.open_rois(s, p)
    assert len(s.roi_manager) == n
    flat = RM.flatten(s, im, labels=True)
    assert flat.is_rgb
    RM.show_all(im, s.roi_manager, labels=True)
    assert len(im.overlay) == n
    RM.show_all(im, s.roi_manager, show=False)
    assert len(im.overlay) == 0
    RM.fill(s, im, [0])
    assert RM.list_table(s).size == n


def test_coloc2():
    rng = np.random.default_rng(1)
    base = rng.random((64, 64)) * 200
    c1 = (base + rng.normal(0, 5, base.shape)).clip(0, 255).astype(np.uint8)
    c2 = (base * 0.8 + rng.normal(0, 5, base.shape)).clip(0, 255).astype(np.uint8)
    s = Session()
    CA.register(s.registry)
    a = ImageDoc.from_array(c1, "YX", "c1")
    b = ImageDoc.from_array(c2, "YX", "c2")
    s.add_image(a)
    s.add_image(b)
    out = s.run("Coloc 2", "channel_1=c1 channel_2=c2 roi_or_mask=<None> "
                           "threshold_regression=Bisection li_icq spearman's_rank_correlation "
                           "manders'_correlation 2d_instensity_histogram "
                           "costes'_significance_test psf=3 costes_randomisations=5")
    rt = out[0]
    vals = {rt.get_label(i): rt.get_value("Value", i) for i in range(rt.size)}
    assert vals["Pearson's R value (no threshold)"] > 0.95
    assert vals["Pearson's R value (below threshold)"] <= 0.05
    assert vals["Li's ICQ value"] > 0.3 and vals["Costes P-Value"] == 1.0
    assert 0 < vals["Manders' tM1 (Above autothreshold)"] <= 1
    from rsstatchem.core.imaging.fiji.coloc import li_icq, manders, _pearson
    x = np.arange(10.0)
    assert _pearson(x, 2 * x + 1) == pytest.approx(1.0)
    assert li_icq(x, x) == pytest.approx(0.5) and manders(x, x) == (1.0, 1.0)
    assert out[1].width == 256


def test_analyze_skeleton():
    from rsstatchem.core.imaging.fiji.skeleton_analysis import analyze_skeleton
    a = np.zeros((30, 30), bool)
    a[15, 5:26] = True          # horizontal line
    a[5:15, 15] = True          # vertical branch meeting it: a "T"
    res = analyze_skeleton(a)
    row = res.rows[0]
    assert row["# Branches"] == 3 and row["# Junctions"] == 1
    assert row["# End-point voxels"] == 3 and row["# Triple points"] == 1
    # the junction is the cluster (15,14),(15,15),(15,16),(14,15); branches end on it
    assert row["Maximum Branch Length"] == pytest.approx(9.0)
    assert row["# Junction voxels"] == 4
    assert res.tagged[15, 5] == 30 and res.tagged[15, 15] == 70 and res.tagged[15, 10] == 127
    ring = np.zeros((20, 20), bool)                     # an 8-connected diamond loop
    for k in range(-6, 7):
        ring[10 + k, 10 + (6 - abs(k))] = ring[10 + k, 10 - (6 - abs(k))] = True
    r2 = analyze_skeleton(ring)
    assert r2.rows[0]["# Branches"] == 1 and r2.rows[0]["# Junctions"] == 0
    s = Session()
    CA.register(s.registry)
    img = ImageDoc.from_array(np.where(a, 255, 0).astype(np.uint8), "YX", "skel")
    s.add_image(img)
    out = s.run("Analyze Skeleton (2D/3D)", "prune=none show")
    assert out[0].get_value("# Branches", 0) == 3 and out[1].size == 3
    assert out[2].title == "Tagged skeleton"


def test_cell_counter_model(tmp_path):
    from rsstatchem.ui.imagej.frames.point_counter import CounterModel
    m = CounterModel()
    m.add_marker(10, 10)
    m.current = 2
    m.add_marker(20, 20)
    m.add_marker(30, 30, slice_=2)
    assert m.counts() == [1, 0, 2, 0, 0, 0, 0, 0] and m.counts(1) == [1, 0, 1, 0, 0, 0, 0, 0]
    assert m.delete_nearest(21, 21) == 1
    roi = m.to_roi()
    assert roi.counters.tolist() == [0, 2] and roi.positions.tolist() == [1, 2]
    m.export_csv(tmp_path / "c.csv")
    m2 = CounterModel()
    m2.import_csv(tmp_path / "c.csv")
    assert m2.markers == m.markers
    rt = m.results(2)
    assert rt.size == 3 and rt.get_label(2) == "Total"


# ---------------------------------------------------------------------------------- UI
@pytest.fixture
def bar(qapp):
    from rsstatchem.ui.imagej.bar import ImageJBar
    b = ImageJBar(None)
    b.message_hook = lambda title, text: b._messages.append(text)
    b._messages = []
    return b


def _shot(widget, name):
    try:
        SHOTS.mkdir(parents=True, exist_ok=True)
        widget.grab().save(str(SHOTS / name))
    except OSError:
        pass


def test_ui_windows(bar):
    from rsstatchem.ui.imagej import windows as W
    from rsstatchem.ui.imagej.frames.histogram_window import HistogramWindow
    from rsstatchem.ui.imagej.frames.plot_window import PlotWindow
    from rsstatchem.ui.imagej.frames.results_window import ResultsWindow
    from rsstatchem.ui.imagej.frames.roi_manager import manager_window

    s = bar.session
    assert bar.actions_by_label["Measure"].isEnabled()
    assert bar.actions_by_label["Select All"].isEnabled()
    assert all(bar.actions_by_label[lb].isEnabled() for lb in
               __import__("rsstatchem.core.imaging.menu_layout",
                          fromlist=["x"]).labels_of_area("analyze"))
    im = bar.open_path(str(input_path("blobs.gif")))
    im.threshold = Threshold(126, 255)
    s.options.measurements = tuple(ALL_KEYS)
    bar.run_command("Analyze Particles...", "display clear summarize")
    res = [w for w in W.WINDOWS if isinstance(w, ResultsWindow)]
    titles = {w.windowTitle() for w in res}
    assert {"Results", "Summary"} <= titles
    rw = next(w for w in res if w.windowTitle() == "Results")
    assert rw.model.rowCount() == 64 and rw.model.columnCount() == len(rw.table.headings)
    assert rw.copy().startswith(" \tLabel\tArea")
    rw.resize(900, 420)
    _shot(rw, "results_window.png")
    im.roi = S.LineRoi(10, 20, 200, 180)
    bar.run_command("Plot Profile")
    pw = [w for w in W.WINDOWS if isinstance(w, PlotWindow)][-1]
    assert pw.list_values().size == 249
    assert pw.copy_data(False).count("\n") == 250          # header + 249 rows
    pw.live.setChecked(True)
    im.roi = S.LineRoi(10, 20, 100, 20)
    s.image_updated(im)
    assert len(pw.plot.series[0]["y"]) == 91
    _shot(pw, "plot_profile.png")
    im.roi = None
    bar.run_command("Histogram")
    hw = [w for w in W.WINDOWS if isinstance(w, HistogramWindow)][-1]
    assert "N: 65024" in hw.stats.text() and hw.list_values().size == 256
    hw.log_box.setChecked(True)
    _shot(hw, "histogram.png")
    # ROI manager window
    im.roi = S.RectRoi(10, 10, 40, 40)
    bar.run_command("Add to Manager", "")
    mw = manager_window(bar, create=False)
    assert mw is not None and mw.list.count() == 1
    im.roi = S.OvalRoi(60, 60, 40, 40)
    mw.add()
    mw.list.item(0).setSelected(True)
    assert im.roi.bounds() == (10, 10, 40, 40)
    mw.show_all_box.setChecked(True)
    mw.labels_box.setChecked(True)
    assert len(im.overlay) == 2
    mw.list.clearSelection()
    mw.measure()
    assert s.results.size == 66                          # appended to the 64 particles
    _shot(mw, "roi_manager.png")
    from rsstatchem.ui.imagej.windows import window_for
    iw = window_for(im)
    iw.canvas.refresh()
    _shot(iw, "blobs_overlay.png")
    # Cell counter
    bar.run_command("Cell Counter")
    from rsstatchem.ui.imagej.frames.point_counter import counter_window
    cw = counter_window(bar, create=False)
    cw.click(20, 30)
    cw.model.current = 1
    cw.click(40, 50)
    assert isinstance(im.roi, S.PointRoi) and cw.model.counts()[:2] == [1, 1]
    assert cw.results().size == 1
    _shot(cw, "cell_counter.png")
    # results window menus work without dialogs
    rw2 = next(w for w in W.WINDOWS if isinstance(w, ResultsWindow) and
               w.windowTitle() == "Results")
    rw2.sort_dialog("Area", ascending=False)
    assert rw2.table.get_value("Area", 0) >= rw2.table.get_value("Area", 1)
    rw2.decimals_dialog(2)
    assert rw2.table.precision == 2
    dup = rw2.duplicate("Copy of Results")
    assert s.tables["Copy of Results"] is dup
    bar.run_command("Synchronize Windows")
    assert not bar._messages, bar._messages
    bar.close()
    assert W.WINDOWS == []


def test_results_window_save(qapp, tmp_path):
    from rsstatchem.core.imaging.results import ResultsTable
    from rsstatchem.ui.imagej.frames.results_window import ResultsWindow
    rt = ResultsTable.results()
    rt.add_row({"Area": 3, "Mean": 1.5}, label="a")
    win = ResultsWindow(None, rt)
    for ext in (".csv", ".xls", ".txt"):
        p = win.save_as(str(tmp_path / f"r{ext}"))
        assert Path(p).read_text().splitlines()[0].replace("\t", ",") == " ,Label,Area,Mean"
    win.rename("Other")
    assert win.windowTitle() == "Other"
    win.close()
