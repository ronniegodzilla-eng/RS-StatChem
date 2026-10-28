"""Image Analysis Phase 0 contracts: image model, calibration, LUT rendering, ROI masks,
results table, option strings / registry / recorder, menu layout, the ImageJ bar and golden
data (values from the real ImageJ 1.54p, see scripts/imagej_golden.py)."""
from __future__ import annotations

import numpy as np
import pytest

from ij_golden import golden_csv, golden_images, input_path, thresholds
from rsstatchem.core.imaging import commands as C
from rsstatchem.core.imaging.calibration import Calibration
from rsstatchem.core.imaging.image import COLOR_RGB, GRAY8, GRAY16, GRAY32, ImageDoc, Threshold
from rsstatchem.core.imaging.lut import LUT, render, to_8bit
from rsstatchem.core.imaging.menu_layout import (AREAS, find, iter_items, labels_of_area,
                                                 owner_of, path_of, qt_shortcut)
from rsstatchem.core.imaging.options import Options
from rsstatchem.core.imaging.results import ResultsTable, d2s
from rsstatchem.core.imaging.roi import (CompositeRoi, LineRoi, OvalRoi, Overlay, PointRoi,
                                         PolygonRoi, RectRoi)
from rsstatchem.core.imaging.session import Session


# ---------------------------------------------------------------------------------- image
def test_image_layout_and_positions():
    a = np.arange(2 * 3 * 4 * 5 * 6, dtype=np.uint16).reshape(2, 3, 4, 5, 6)   # T Z C Y X
    doc = ImageDoc.from_array(a, "TZCYX", "hs")
    assert doc.dims == (4, 3, 2) and (doc.width, doc.height) == (6, 5)
    assert doc.type == GRAY16 and doc.bit_depth == 16 and doc.stack_size == 24
    for index in range(doc.stack_size):
        assert doc.stack_index(*doc.position_of(index)) == index
    c, z, t = 1, 2, 1
    assert np.array_equal(doc.plane(c, z, t), a[t, z, c])
    assert doc.stack_index(c, z, t) == c + z * 4 + t * 12
    # axis order is free: YX + C first
    b = ImageDoc.from_array(np.zeros((3, 5, 6), np.float64), "CYX")
    assert b.type == GRAY32 and b.pixels.dtype == np.float32 and b.n_channels == 3
    assert b.composite_mode == "composite" and b.luts[0] == LUT.from_color(255, 0, 0)


def test_image_kinds_and_signed16():
    rgb = ImageDoc.from_array(np.zeros((4, 5, 3), np.uint8), "YXS")
    assert rgb.type == COLOR_RGB and rgb.pixels.shape == (1, 1, 1, 4, 5, 3)
    assert ImageDoc.from_array(np.ones((2, 2), bool)).raw_value(0, 0) == 255.0
    s = ImageDoc.from_array(np.array([[-5, 7]], np.int16))
    assert s.type == GRAY16 and s.raw_value(0, 0) == 32763.0
    assert s.calibration.is_signed16() and s.value(0, 0) == -5.0 and s.value(1, 0) == 7.0


def test_display_range_undo_and_subtitle():
    doc = ImageDoc.from_array(np.array([[0, 1000], [10, 20]], np.uint16), title="a.tif")
    assert doc.display_range == (0.0, 1000.0)
    doc.snapshot("Invert")
    doc.pixels[...] = 5
    doc.threshold = Threshold(10, 20)
    assert doc.undo() and doc.raw_value(1, 0) == 1000.0 and doc.threshold is None
    assert doc.undo() and doc.raw_value(1, 0) == 5.0       # undo twice = redo, as ImageJ
    blobs = ImageDoc.from_array(np.zeros((254, 256), np.uint8), title="blobs.gif")
    blobs.luts = [LUT.grays().inverted()]
    assert blobs.subtitle() == "256x254 pixels; 8-bit (inverting LUT); 64K"
    blobs.calibration.set_scale(10, 1, "inch")
    assert blobs.subtitle().startswith("25.60x25.40 inches (256x254)")
    dup = blobs.duplicate("copy")
    dup.pixels[0, 0, 0, 0, 0] = 9
    assert blobs.raw_value(0, 0) == 0 and dup.luts[0].is_inverted()


def test_calibration_functions():
    cal = Calibration()
    assert not cal.scaled() and cal.units == "pixels"
    cal.set_scale(10, 2.5, "um")
    assert cal.pixel_width == 0.25 and cal.scaled() and cal.get_x(4) == 1.0
    cal.function, cal.coefficients = 0, (1.0, 2.0)     # straight line
    assert cal.cvalue(3) == 7.0 and cal.raw_value(7.0, 8) == 3.0
    cal.function = 21                                  # uncalibrated OD
    assert cal.cvalue(255) == pytest.approx(0.0)
    assert cal.cvalue(0) == pytest.approx(np.log10(255 / 0.5))


def test_lut_to_8bit_matches_imagej_formulas():
    v = np.array([[0, 100, 200, 300]], np.uint16)
    # ShortProcessor.create8BitImage: (v - min) * 256/(max-min+1) + 0.5
    assert to_8bit(v, 100, 299).tolist() == [[0, 0, 128, 255]]
    b = np.array([[0, 50, 100, 150, 255]], np.uint8)
    out = to_8bit(b, 50, 150)
    assert out[0, 0] == 0 and out[0, 4] == 255 and out[0, 2] == 128
    assert to_8bit(b, 0, 255) is b
    f = np.array([[0.0, 0.5, 1.0, np.nan]], np.float32)
    assert to_8bit(f, 0.0, 1.0).tolist() == [[0, 128, 255, 0]]
    comp = ImageDoc.from_array(np.full((3, 2, 4), 255, np.uint8), "CYX")
    assert render(comp)[0, 0].tolist() == [255, 255, 255]   # red + green + blue
    comp.active_channels[2] = False
    assert render(comp)[0, 0].tolist() == [255, 255, 0]


# ---------------------------------------------------------------------------------- ROIs
def test_roi_masks_contract():
    assert RectRoi(2, 3, 4, 5).get_mask((20, 20)).sum() == 20
    assert RectRoi(-2, -2, 4, 4).get_mask((10, 10)).sum() == 4
    assert PolygonRoi([0, 4, 4, 0], [0, 0, 4, 4]).get_mask((10, 10)).sum() == 16
    assert LineRoi(0, 0, 5, 5).get_mask((10, 10)).sum() == 0
    assert not PointRoi([1], [1]).is_area()
    comp = CompositeRoi(parts=[("or", RectRoi(0, 0, 4, 4)), ("not", RectRoi(1, 1, 2, 2))])
    assert comp.get_mask((8, 8)).sum() == 12 and comp.bounds() == (0, 0, 4, 4)
    assert RectRoi(0, 0, 3, 3).contains(2, 2) and not RectRoi(0, 0, 3, 3).contains(3, 3)
    assert LineRoi(0, 0, 3, 4).length() == 5.0
    ov = Overlay()
    ov.add(RectRoi(1, 1, 2, 2, name="a"))
    assert len(ov) == 1 and ov.copy().rois[0].name == "a"


def test_roi_masks_against_imagej_goldens():
    """Rectangles and ovals of the Phase 0 rasteriser equal ImageJ's masks; polygons are
    WP-C's to make exact (PolygonFiller port), so only near-equality is required here."""
    import csv

    from ij_golden import golden_path
    masks = golden_images("masks")
    with open(golden_path("masks", "roi_specs.csv"), newline="") as fh:
        specs = list(csv.DictReader(fh))
    assert len(specs) == 12
    for row in specs:
        kind, *nums = row["spec"].split(",")
        v = [float(n) for n in nums]
        roi = {"rect": lambda: RectRoi(*v), "oval": lambda: OvalRoi(*v),
               "poly": lambda: PolygonRoi(v[0::2], v[1::2])}[kind]()
        ours = roi.get_mask((64, 64))
        ij = masks[row["name"]] > 0
        if kind in ("rect", "oval"):
            assert np.array_equal(ours, ij), row
        elif row["name"] != "roi_11":          # roi_11 is self-intersecting
            assert (ours != ij).sum() <= 4, row


# ---------------------------------------------------------------------------------- results
def test_results_table_imagej_semantics(tmp_path):
    assert d2s(2.5, 0) == "3" and d2s(1.23456, 3) == "1.235" and d2s(0.0001, 3) == "1.000E-4"
    assert d2s(float("nan"), 3) == "NaN" and d2s(1.5e12, 3) == "1.500E12"
    rt = ResultsTable.results()
    rt.add_row({"Mean": 2.5, "Foo": "a,b", "Area": 10}, label="blobs.gif")
    rt.add_value("Bar", 1.0)
    rt.add_row({"Area": 3})
    assert rt.headings == ["Label", "Area", "Mean", "Foo", "Bar"]   # standard columns first
    assert rt.get_value("Mean", 1) == 0.0                            # ImageJ fills with 0
    text = rt.to_text()
    assert text.splitlines()[0] == " ,Label,Area,Mean,Foo,Bar"
    assert text.splitlines()[1] == '1,blobs.gif,10,2.500,"a,b",1'
    rt.save(tmp_path / "r.csv")
    back = ResultsTable.read(tmp_path / "r.csv")
    assert back.headings == rt.headings and back.get_value("Foo", 0) == "a,b"
    plain = ResultsTable("T")
    plain.add_row({"Z": 1, "Area": 2})
    assert plain.headings == ["Z", "Area"]
    s = rt.summarize()
    assert s.get_label(0) == "Mean" and s.get_value("Area", 0) == 6.5
    assert list(rt.to_dataframe().columns) == rt.headings


# ---------------------------------------------------------------------------------- commands
def test_option_strings_like_imagej():
    assert C.macro_key("Sigma (Radius):") == "sigma"
    assert C.macro_key("Exclude on edges") == "exclude"
    assert C.parse_options("sigma=2 stack title=[my image] size=0-Infinity") == {
        "sigma": "2", "stack": True, "title": "my image", "size": "0-Infinity"}
    fields = [C.number("Sigma (Radius):", 2.0, 2), C.checkbox("Scaled Units (µm)", True),
              C.choice("Method:", ["Default", "Otsu"]), C.string("Title:", "a b"),
              C.checkbox_group(["Display results", "Clear results"], [True, False]),
              C.message("note")]
    vals = C.values_from_options(fields, "sigma=1.5 method=Otsu display")
    assert vals["sigma"] == 1.5 and vals["scaled"] is False and vals["method"] == "Otsu"
    assert vals["title"] == "a b" and vals["display"] is True and vals["clear"] is False
    assert C.format_options(fields, {**vals, "scaled": True}) == \
        "sigma=1.50 scaled method=Otsu title=[a b] display"
    assert C.format_number(2.0) == "2" and C.format_number(0.125, 1) == "0.125"
    with pytest.raises(C.CommandError):
        C.values_from_options(fields, "method=Bogus")
    with pytest.raises(C.CommandError):
        C.values_from_options(fields, "sigma=abc")


def test_registry_run_records_and_undoes():
    s = Session()
    seen = {}

    def blur(session, image, sigma, stack=False):
        seen.update(sigma=sigma, stack=stack)
        image.pixels[...] = 7

    s.registry.add("Gaussian Blur...", blur, [C.number("Sigma (Radius):", 2.0, 2)],
                   supports_stack=True, types=(GRAY8, GRAY16, GRAY32))
    assert s.registry.get("Gaussian Blur...").owner == "process"
    with pytest.raises(C.CommandError, match="no images"):
        s.run("Gaussian Blur...", "sigma=1")
    doc = s.add_image(ImageDoc.from_array(np.zeros((3, 4, 4), np.uint8), "ZYX", "st"))
    s.recorder = C.Recorder()
    events = []
    s.on("image_updated", events.append)
    s.run("Gaussian Blur...", "sigma=3 stack")
    assert seen == {"sigma": 3.0, "stack": True} and events == [doc] and doc.changed
    assert s.recorder.lines == ['run("Gaussian Blur...", "sigma=3 stack");']
    assert s.last_command == ("Gaussian Blur...", "sigma=3 stack")
    assert doc.undo() and doc.pixels.max() == 0
    assert C.python_line("Gaussian Blur...", "sigma=3 stack") == \
        "ij.run('Gaussian Blur...', sigma=3, stack=True)"
    s.registry.add("Make Table", lambda ss, im: ResultsTable("T"), needs_image=False)
    tables = []
    s.on("table_updated", tables.append)
    s.run("Make Table")
    assert tables and s.tables["T"] is tables[0]
    with pytest.raises(ValueError):
        s.registry.add("Make Table", lambda ss, im: None)
    s.registry.add("Needs RGB", lambda ss, im: None, types=(COLOR_RGB,))
    with pytest.raises(C.CommandError, match="RGB"):
        s.run("Needs RGB")
    assert [c.label for c in s.registry.find("blur")] == ["Gaussian Blur..."]


def test_session_titles_and_options_roundtrip():
    s = Session()
    a = s.add_image(ImageDoc.new(4, 4, title="blobs.gif"))
    b = s.add_image(ImageDoc.new(4, 4, title="blobs.gif"))
    assert b.title == "blobs-1.gif" and s.active is b and s.find_image("blobs.gif") is a
    s.close_image(b)
    assert s.active is a
    o = Options(black_background=True, measurements=("area", "mean"), decimal_places=5)
    back = Options.from_dict(o.to_dict())
    assert back == o and back.measure("mean") and not back.measure("min")
    assert Options.from_dict({"black_background": "true", "bogus": 1}).black_background


# ---------------------------------------------------------------------------------- menus
def test_menu_layout_is_imagej_154():
    labels = [item.label for _, item in iter_items()]
    assert len(labels) == len(set(labels)) > 380
    assert path_of("Gaussian Blur...") == ("Process", "Filters")
    assert path_of("Start Animation [\\]") == ("Image", "Stacks", "Animation")
    assert find("Image...").shortcut == "n" and qt_shortcut(find("Close All")) == "Ctrl+Shift+W"
    assert find("Properties... ").shortcut == "y" and path_of("Properties... ")[0] == "Edit"
    assert owner_of("Threshold...") == "process" and owner_of("Fit Ellipse") == "analyze"
    assert owner_of("Repeat Command") == "plugins" and owner_of("Measure") == "analyze"
    assert owner_of("8-bit") == "image" and owner_of("Open...") == "file"
    assert owner_of("Tile") == "shell" and owner_of("not a command") is None
    for area in AREAS:
        assert labels_of_area(area), area
    assert "Compile and Run..." not in labels


# ---------------------------------------------------------------------------------- golden
def test_golden_thresholds_and_particles():
    th = thresholds()
    assert len(th) == 4 * (1 + 17 * 2)
    assert th[("blobs.gif", "Default", False)] == (126.0, 255.0)
    assert th[("blobs.gif", "Default", True)] == (0.0, 125.0)
    summary = golden_csv("particles", "blobs_default_summary.csv")
    assert summary.get_value("Count", 0) == 64 and summary.get_value("Total Area", 0) == 22243
    rows = golden_csv("particles", "blobs_default.csv")
    assert len(rows) == 64 and rows.column("Area").sum() == 22243
    assert golden_csv("particles", "blobs_exclude_summary.csv").get_value("Count", 0) == 46
    assert golden_csv("particles", "blobs_circ_summary.csv").get_value("Count", 0) == 43
    rect = golden_csv("measure", "uncalibrated_rect.csv")
    assert rect.get_value("Area", 0) == 4800 and rect.get_value("Width", 0) == 80
    img = golden_images("filters")
    assert img["input_synth8"].shape == (64, 64) and img["synth8_gauss_2"].dtype == np.uint8
    assert img["synth32_median_2"].dtype == np.float32
    assert input_path("blobs.gif").exists()


def test_blobs_opens_like_imagej():
    from ij_golden import blobs
    doc = blobs()
    assert (doc.width, doc.height, doc.type) == (256, 254, GRAY8) and doc.is_inverted_lut()
    # Measure of the whole image: ImageJ's mean of the raw pixel values
    whole = golden_csv("measure", "uncalibrated_whole.csv")
    assert doc.pixels.mean() == pytest.approx(whole.get_value("Mean", 0), rel=1e-9)


# ---------------------------------------------------------------------------------- UI
@pytest.fixture
def ctx(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("RSSTATCHEM_HOME", str(tmp_path))
    from rsstatchem.app import AppContext

    context = AppContext(qapp)
    monkeypatch.setattr(context.app, "quit", lambda: None)
    return context


def test_bar_opens_image_and_runs_commands(ctx):
    from rsstatchem.ui.imagej.windows import BARS, WINDOWS, image_windows
    from rsstatchem.ui.modules import open_module

    bar = open_module(ctx, "image", str(input_path("blobs.gif")))
    assert bar.ctx is ctx and bar in ctx.windows and BARS == [bar]
    assert open_module(ctx, "image") is bar                 # one ImageJ bar per app
    titles = [a.text() for a in bar.menuBar().actions()]
    assert titles == ["File", "Edit", "Image", "Process", "Analyze", "Plugins", "Window", "Help"]
    acts = bar.actions_by_label
    assert acts["Tile"].isEnabled() and not acts["Gaussian Blur..."].isEnabled()
    assert acts["Image..."].shortcut().toString() == "Ctrl+N"
    win = image_windows()[0]
    assert win.windowTitle() == "blobs.gif" and "inverting LUT" in win.info.text()
    win.canvas.zoom_in()
    assert win.windowTitle() == "blobs.gif (150%)" and win.canvas.magnification == 1.5
    assert win.status_text(10, 10).startswith("x=10, y=10, value=")
    # a command registered later becomes enabled and runs through the generic path
    bar.session.registry.add("Invert", lambda s, im: im.pixels.__ixor__(255),
                             types=(GRAY8,), replace=True)   # WP-A registers Invert
    bar.refresh_menus()
    assert acts["Invert"].isEnabled()
    before = int(win.doc.pixels[0, 0, 0, 0, 0])
    bar.run_command("Invert")
    assert int(win.doc.pixels[0, 0, 0, 0, 0]) == 255 - before
    messages = []
    bar.message_hook = lambda title, text: messages.append(text)
    bar.run_command("Not There")
    assert messages and "Unrecognized" in messages[0]
    bar.menus["Window"].aboutToShow.emit()
    assert "blobs.gif" in [a.text() for a in bar.menus["Window"].actions()]
    bar.close()
    assert WINDOWS == [] and BARS == [] and bar not in ctx.windows


def test_generic_dialog_values(qapp):
    from rsstatchem.ui.imagej.dialogs.generic import GenericDialog

    fields = [C.number("Sigma (Radius):", 2.0, 2), C.checkbox("Stack", False),
              C.choice("Method:", ["Default", "Otsu"]), C.string("Title:", "x"),
              C.radio("Mode:", ["A", "B"], "B"), C.slider("Level:", 0, 10, 3),
              C.checkbox_group(["Display results", "Clear results"], [True, False])]
    dlg = GenericDialog("Test", fields)
    assert dlg.values() == {"sigma": 2.0, "stack": False, "method": "Default", "title": "x",
                            "mode": "B", "level": 3.0, "display": True, "clear": False,
                            "group": (True, False)}
    dlg.set_value("sigma", 4.5)
    dlg.set_value("method", "Otsu")
    vals = dlg.values()
    assert vals["sigma"] == 4.5 and vals["method"] == "Otsu"
    assert C.format_options(fields, vals) == \
        "sigma=4.50 method=Otsu title=x mode=B level=3 display"
    previews = []
    dlg2 = GenericDialog("P", fields[:1], preview=previews.append)
    dlg2.preview_box.setChecked(True)
    assert previews and previews[-1]["sigma"] == 2.0


def test_module_selftest():
    pytest.importorskip("tifffile")
    pytest.importorskip("roifile")
    pytest.importorskip("pydicom")
    from rsstatchem.ui.imagej import selftest
    selftest()
