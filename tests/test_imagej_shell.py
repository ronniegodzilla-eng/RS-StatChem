"""Image Analysis shell (WP-A): tools on the canvas, zoom/scroll, stack sliders, status line,
single-key shortcuts, File/Edit/Plugins commands, Edit pixel operations against ImageJ
1.54p goldens (case ``edit``), the macro interpreter, the recorder, Find Commands, projects."""
from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent
from PySide6.QtWidgets import QApplication

from ij_golden import golden_images, input_path
from rsstatchem.core.imaging.image import COLOR_RGB, GRAY8, GRAY16, ImageDoc
from rsstatchem.core.imaging.roi import shapes as S
from rsstatchem.core.imaging.session import Session

pytest.importorskip("tifffile")


@pytest.fixture
def bar(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("RSSTATCHEM_HOME", str(tmp_path))
    from rsstatchem.ui.imagej.bar import ImageJBar
    from rsstatchem.ui.imagej.windows import BARS
    b = ImageJBar(None)
    BARS[:] = [b]
    b.messages = []
    b.message_hook = lambda title, text: b.messages.append((title, text))
    b.question_hook = lambda title, text: "no"
    b.show()
    yield b
    b.close()


def _window(bar, doc):
    from rsstatchem.ui.imagej.windows import window_for
    return window_for(doc)


def _new(bar, arr, axes="YX", title="img"):
    doc = ImageDoc.from_array(arr, axes, title)
    bar.session.add_image(doc)
    return doc, _window(bar, doc)


def mouse(canvas, kind, x, y, button=Qt.MouseButton.LeftButton,
          mods=Qt.KeyboardModifier.NoModifier):
    """Send a mouse event at image coordinates (x, y)."""
    sx, sy = canvas.image_to_screen(x, y)
    types = {"press": QMouseEvent.Type.MouseButtonPress,
             "move": QMouseEvent.Type.MouseMove,
             "release": QMouseEvent.Type.MouseButtonRelease,
             "double": QMouseEvent.Type.MouseButtonDblClick}
    buttons = button if kind in ("press", "move", "double") else Qt.MouseButton.NoButton
    ev = QMouseEvent(types[kind], QPointF(sx, sy), canvas.mapToGlobal(QPointF(sx, sy)),
                     button if kind != "move" else Qt.MouseButton.NoButton, buttons, mods)
    QApplication.sendEvent(canvas, ev)


def drag(canvas, a, b, mods=Qt.KeyboardModifier.NoModifier, steps=4):
    mouse(canvas, "press", *a, mods=mods)
    for i in range(1, steps + 1):
        f = i / steps
        mouse(canvas, "move", a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, mods=mods)
    mouse(canvas, "release", *b, mods=mods)


def key(widget, k, text="", mods=Qt.KeyboardModifier.NoModifier):
    QApplication.sendEvent(widget, QKeyEvent(QKeyEvent.Type.KeyPress, k, mods, text))
    QApplication.sendEvent(widget, QKeyEvent(QKeyEvent.Type.KeyRelease, k, mods, text))


# ---------------------------------------------------------------------------------- tools
def test_rectangle_oval_and_composite_tools(bar):
    from rsstatchem.ui.imagej import toolbar as T
    doc, win = _new(bar, np.zeros((60, 80), np.uint8))
    cv = win.canvas
    bar.toolbar.set_tool(T.RECTANGLE)
    drag(cv, (10.2, 12.7), (40.4, 32.1))
    assert isinstance(doc.roi, S.RectRoi) and doc.roi.bounds() == (10, 12, 30, 20)
    # drag inside moves, handle resizes
    drag(cv, (20.5, 20.5), (25.5, 22.5))
    assert doc.roi.bounds() == (15, 14, 30, 20)
    drag(cv, (45, 34), (50, 40))                         # bottom-right handle
    assert doc.roi.bounds() == (15, 14, 35, 26)
    # shift = square
    drag(cv, (60, 2), (70, 30), mods=Qt.KeyboardModifier.ShiftModifier)
    assert isinstance(doc.roi, S.CompositeRoi)            # shift with a selection adds
    mouse(cv, "press", 75, 55)
    mouse(cv, "release", 75, 55)
    assert doc.roi is None                                # click outside removes it
    drag(cv, (5, 5), (15, 25), mods=Qt.KeyboardModifier.ShiftModifier)
    assert doc.roi.bounds() == (5, 5, 20, 20)             # square
    bar.toolbar.set_tool(T.OVAL)
    doc.roi = None
    drag(cv, (10, 10), (30, 20))
    assert isinstance(doc.roi, S.OvalRoi) and doc.roi.bounds() == (10, 10, 20, 10)
    drag(cv, (40, 10), (60, 30), mods=Qt.KeyboardModifier.AltModifier)  # alt-drag off the oval
    assert doc.roi is not None
    bar.toolbar.set_variant(T.RECTANGLE, "Rounded Rectangle")
    doc.roi = None
    drag(cv, (10, 10), (40, 40))
    assert doc.roi.corner_diameter > 0
    bar.toolbar.set_variant(T.OVAL, "Elliptical selections")
    drag(cv, (50, 50), (70, 50))
    assert isinstance(doc.roi, S.EllipseRoi)


def test_polygon_line_angle_point_tools(bar):
    from rsstatchem.ui.imagej import toolbar as T
    doc, win = _new(bar, np.zeros((60, 80), np.uint8))
    cv = win.canvas
    bar.toolbar.set_tool(T.POLYGON)
    for p in ((10, 10), (50, 14), (40, 55)):
        mouse(cv, "press", *p)
        mouse(cv, "release", *p)
    mouse(cv, "double", 40, 55)
    assert isinstance(doc.roi, S.PolygonRoi) and doc.roi.type == S.POLYGON
    assert doc.roi.xs.tolist() == [10, 50, 40]
    bar.toolbar.set_tool(T.LINE)
    doc.roi = None
    drag(cv, (5, 5), (60, 40))
    assert isinstance(doc.roi, S.LineRoi) and (doc.roi.x2, doc.roi.y2) == (60, 40)
    drag(cv, (5, 50), (60, 54), mods=Qt.KeyboardModifier.ShiftModifier)
    assert doc.roi.y1 == doc.roi.y2                       # shift: horizontal
    bar.toolbar.set_variant(T.LINE, "Segmented Line")
    assert bar.toolbar.current == T.POLYLINE
    for p in ((5, 5), (20, 40), (50, 10)):
        mouse(cv, "press", *p)
        mouse(cv, "release", *p)
    key(cv, Qt.Key.Key_Return)
    assert doc.roi.type == S.POLYLINE and len(doc.roi.xs) == 3
    bar.toolbar.set_variant(T.LINE, "Arrow tool")
    doc.roi = None
    drag(cv, (5, 5), (30, 30))
    assert doc.roi.arrow
    bar.toolbar.set_tool(T.ANGLE)
    for p in ((10, 40), (30, 10), (50, 40)):
        mouse(cv, "press", *p)
        mouse(cv, "release", *p)
    assert doc.roi.type == S.ANGLE
    bar.toolbar.set_tool(T.FREEROI)
    drag(cv, (10, 10), (30, 30), steps=10)
    assert doc.roi is None or doc.roi.type in (S.FREEROI,)
    bar.toolbar.set_tool(T.POINT)
    bar.toolbar.set_variant(T.POINT, "Multi-point Tool")
    for p in ((5, 5), (25, 26), (45, 46)):
        mouse(cv, "press", *p)
        mouse(cv, "release", *p)
    assert isinstance(doc.roi, S.PointRoi) and len(doc.roi.xs) == 3
    mouse(cv, "press", 25, 26, mods=Qt.KeyboardModifier.AltModifier)
    mouse(cv, "release", 25, 26, mods=Qt.KeyboardModifier.AltModifier)
    assert len(doc.roi.xs) == 2                           # alt-click deletes a point


def test_wand_text_magnifier_hand_picker(bar):
    from rsstatchem.ui.imagej import toolbar as T
    a = np.zeros((50, 60), np.uint8)
    a[10:20, 10:30] = 200
    a[14:16, 14:16] = 0                                    # hole: legacy wand ignores it
    doc, win = _new(bar, a)
    cv = win.canvas
    bar.toolbar.set_tool(T.WAND)
    mouse(cv, "press", 12, 12)
    mouse(cv, "release", 12, 12)
    assert doc.roi.type == S.TRACED_ROI and doc.roi.bounds() == (10, 10, 20, 10)
    assert doc.roi.get_mask((50, 60)).sum() == 200
    bar.toolbar.set_tool(T.TEXT)
    mouse(cv, "press", 30, 30)
    mouse(cv, "release", 30, 30)
    for ch in "Hi":
        key(cv, Qt.Key.Key_A, ch)
    assert isinstance(doc.roi, S.TextRoi) and doc.roi.text == "Hi"
    key(cv, Qt.Key.Key_Escape)
    bar.toolbar.set_tool(T.MAGNIFIER)
    mouse(cv, "press", 20, 20)
    mouse(cv, "release", 20, 20)
    assert cv.magnification == 1.5 and win.windowTitle().endswith("(150%)")
    mouse(cv, "press", 20, 20, button=Qt.MouseButton.RightButton)
    assert cv.magnification == 1.0
    bar.toolbar.set_tool(T.DROPPER)
    mouse(cv, "press", 12, 12)
    assert bar.session.options.foreground == "#c8c8c8"
    mouse(cv, "press", 1, 1, mods=Qt.KeyboardModifier.AltModifier)
    assert bar.session.options.background == "#000000"


def test_zoom_levels_scroll_sliders_and_status(bar):
    from rsstatchem.ui.imagej.canvas import ZOOM_LEVELS, higher_zoom, lower_zoom
    assert len(ZOOM_LEVELS) == 23 and higher_zoom(1) == 1.5 and lower_zoom(1) == 0.75
    doc, win = _new(bar, np.arange(4 * 30 * 40, dtype=np.uint16).reshape(4, 30, 40), "ZYX",
                    "stack.tif")
    cv = win.canvas
    for _ in range(3):
        cv.zoom_in((20, 15))
    assert cv.magnification == 3.0 and win.windowTitle() == "stack.tif (300%)"
    cv.zoom_out()
    assert cv.magnification == 2.0
    assert win.sliders["stack"].isVisibleTo(win) and not win.sliders["z"].isVisibleTo(win)
    win.slider.setValue(2)
    assert doc.current_index == 2 and doc.subtitle().startswith("3/4")
    win.step_slice(1)
    assert doc.current_index == 3
    assert win.status_text(5, 1) == f"x=5, y=1, z=3, value={int(doc.plane()[1, 5])}"
    hs, hw = _new(bar, np.zeros((2, 3, 2, 8, 9), np.uint8), "TZCYX", "hs")
    assert all(hw.sliders[k].isVisibleTo(hw) for k in "czt")
    hw.sliders["t"].bar.setValue(1)
    hw.sliders["c"].bar.setValue(1)
    assert hs.position == (1, 0, 1)
    rgb, rw = _new(bar, np.full((4, 5, 3), (255, 0, 10), np.uint8), "YXS", "rgb")
    assert rw.status_text(1, 1) == "x=1, y=1, value=255,000,010 (#ff000a)"
    f32, fw = _new(bar, np.array([[1.5, 2.0]], np.float32), "YX", "f")
    assert fw.status_text(0, 0).endswith("value=1.5000") and \
        fw.status_text(1, 0).endswith("value=2.0")
    f32.calibration.set_scale(2, 1, "mm")
    assert fw.status_text(1, 0).startswith("x=0.50 (1), y=0 (0)")


def test_arrow_keys_shortcuts_and_popup(bar):
    from rsstatchem.ui.imagej import toolbar as T
    doc, win = _new(bar, np.zeros((40, 40), np.uint8))
    cv = win.canvas
    doc.roi = S.RectRoi(5, 5, 10, 10)
    key(cv, Qt.Key.Key_Right)
    key(cv, Qt.Key.Key_Down)
    assert doc.roi.bounds() == (6, 6, 10, 10)
    key(cv, Qt.Key.Key_Right, mods=Qt.KeyboardModifier.AltModifier)
    assert doc.roi.width == 11
    doc.roi = None
    key(cv, Qt.Key.Key_Up)
    assert cv.magnification == 1.5
    # single-key shortcut: "f" = Fill (foreground white)
    doc.roi = S.RectRoi(0, 0, 2, 2)
    key(cv, Qt.Key.Key_F, "f")
    assert doc.pixels.sum() == 4 * 255
    bar.session.options.require_control_key = True
    doc.roi = S.RectRoi(4, 4, 2, 2)
    key(cv, Qt.Key.Key_F, "f")
    assert doc.pixels.sum() == 4 * 255
    menu = bar.image_popup(win, cv.mapToGlobal(cv.rect().center()))
    assert any(a.text() == "Record..." for a in menu.actions())
    bar.toolbar.set_tool("multipoint")
    assert bar.toolbar.current == T.POINT and bar.toolbar.multi_point
    bar.toolbar.set_tool("polyline")
    assert bar.toolbar.current == T.POLYLINE and bar.toolbar.tool_name() == "polyline"


# ---------------------------------------------------------------------------------- edit ops
EDIT_CASES = [
    ("fill8_oval", "synth8.tif", lambda: S.OvalRoi(10, 10, 30, 20), ["Fill"],
     {"foreground": "#c8c8c8"}, None),
    ("clear8", "synth8.tif", lambda: S.RectRoi(5, 5, 20, 15), ["Clear"],
     {"background": "#1e1e1e"}, None),
    ("outside8", "synth8.tif", lambda: S.RectRoi(5, 5, 20, 15), ["Clear", "Clear Outside"],
     {"background": "#1e1e1e"}, None),
    ("invert8", "synth8.tif", lambda: None, ["Invert"], {}, None),
    ("invert8_oval", "synth8.tif", lambda: S.OvalRoi(10, 10, 30, 20), ["Invert"], {}, None),
    ("invert16", "synth16.tif", lambda: None, ["Invert"], {}, None),
    ("invert16_rect", "synth16.tif", lambda: S.RectRoi(5, 5, 30, 30), ["Invert"], {},
     (2000, 30000)),
    ("invert32", "synth32.tif", lambda: None, ["Invert"], {}, (-1, 2)),
    ("fill16", "synth16.tif", lambda: S.RectRoi(5, 5, 20, 10), ["Fill"],
     {"foreground": "#808080"}, (2000, 30000)),
    ("fill32", "synth32.tif", lambda: S.RectRoi(5, 5, 20, 10), ["Fill"],
     {"foreground": "#808080"}, (-1, 2)),
    ("fill_blobs_white", "blobs.gif", lambda: S.RectRoi(5, 5, 20, 10), ["Fill"],
     {"foreground": "#ffffff"}, None),
] + [(f"draw_{kind}_w{w}", "synth8.tif", roi, ["Draw"], {"line_width": w}, None)
     for w in (1, 2, 3)
     for kind, roi in (("rect", lambda: S.RectRoi(10, 12, 30, 20)),
                       ("poly", lambda: S.PolygonRoi([10, 50, 40, 12], [10, 14, 55, 40])),
                       ("line", lambda w=w: S.LineRoi(5, 5, 60, 40, width=w,
                                                      stroke_width=w if w > 1 else 0)))]


@pytest.mark.parametrize("name,src,roi,cmds,opts,rng", EDIT_CASES, ids=[c[0] for c in EDIT_CASES])
def test_edit_operations_match_imagej(name, src, roi, cmds, opts, rng):
    from rsstatchem.core.imaging.io import open_image
    from rsstatchem.ui.imagej import commands_edit as E
    s = Session()
    E.register(s.registry)
    for k, v in opts.items():
        setattr(s.options, k, v)
    im = open_image(input_path(src))
    s.add_image(im)
    if rng:
        im.set_display_range(*rng)
    im.roi = roi()
    for c in cmds:
        s.run(c, "slice")
    golden = golden_images("edit")[name]
    if im.type == 2:
        assert np.allclose(im.plane(), golden, atol=1e-6)
    else:
        assert np.array_equal(im.plane(), golden)


@pytest.mark.parametrize("mode", ["Copy", "Add", "Difference", "Transparent-zero", "Blend",
                                  "Subtract", "Max"])
def test_paste_modes_match_imagej(mode):
    from rsstatchem.core.imaging.io import open_image
    from rsstatchem.ui.imagej import commands_edit as E
    s = Session()
    E.register(s.registry)
    im = open_image(input_path("synth8.tif"))
    s.add_image(im)
    im.roi = S.RectRoi(0, 0, 20, 20)
    s.run("Copy")
    im.roi = S.RectRoi(30, 30, 20, 20)
    s.run("Paste Control...", f"transfer={mode}")
    s.run("Paste")
    assert np.array_equal(im.plane(), golden_images("edit")["paste_" + mode.replace("-", "_")])
    s.run("Undo")
    assert np.array_equal(im.plane(), open_image(input_path("synth8.tif")).plane())


def test_undo_cut_and_clipboard_images(bar):
    doc, win = _new(bar, np.arange(100, dtype=np.uint8).reshape(10, 10), title="a")
    doc.roi = S.RectRoi(2, 2, 3, 3)
    bar.run_command("Cut")
    assert doc.pixels[0, 0, 0, 3, 3] == 0 and bar.session.clipboard.width == 3
    bar.run_command("Undo")
    assert doc.pixels[0, 0, 0, 3, 3] == 33
    bar.run_command("Internal Clipboard")
    clip = bar.session.active
    assert clip.title == "Clipboard" and clip.plane().tolist()[0] == [22, 23, 24]
    bar.run_command("Copy to System")
    clip_rgb = QApplication.clipboard().image()
    assert clip_rgb.width() == 3
    bar.run_command("System Clipboard")
    assert bar.session.active.width == 3


def test_options_dialog_commands(bar):
    s = bar.session
    s.run("Colors...", "foreground=red background=blue selection=cyan")
    assert (s.options.foreground, s.options.background, s.options.selection) == \
        ("#ff0000", "#0000ff", "#00ffff")
    assert bar.toolbar.foreground == "#ff0000"
    s.run("Conversions...", "weighted")
    assert s.options.weighted_rgb and not s.options.scale_conversions
    s.run("Input/Output...", "jpeg=70 gif=-1 file=.txt intel save_column save_row")
    assert s.options.jpeg_quality == 70 and s.options.intel_byte_order and \
        s.options.results_extension == ".txt"
    s.run("Line Width...", "line=3")
    assert s.options.line_width == 3
    s.run("Misc...", "divide=NaN require")
    assert np.isnan(s.options.divide_by_zero) and s.options.require_control_key
    s.run("Wand Tool...", "mode=8-connected tolerance=5")
    assert s.options.wand_mode == "8-connected" and s.options.wand_tolerance == 5
    s.run("Point Tool...", "type=Dot color=red size=Large auto-measure add counter=2")
    assert s.options.point_type == "Dot" and s.options.point_auto_measure and \
        s.options.extra["point_counter"] == 2
    s.run("Rounded Rect Tool...", "stroke=1 corner=15 color=yellow")
    assert s.options.rounded_rect_diameter == 15
    for label in ("Appearance...", "Arrow Tool...", "DICOM...", "Fonts...", "Plots...",
                  "Roi Defaults...", "Startup..."):
        s.run(label, "")
    s.run("Reset... ")
    assert s.options.line_width == 1.0 and not s.options.weighted_rgb
    s.run("Fresh Start")
    assert s.options.black_background


# ---------------------------------------------------------------------------------- File
def test_file_new_save_open_revert_close(bar, tmp_path):
    s = bar.session
    s.run("Image...", "name=n type=8-bit fill=Black width=20 height=10 slices=3")
    doc = s.active
    assert (doc.width, doc.height, doc.n_slices, int(doc.pixels.max())) == (20, 10, 3, 0)
    s.run("Hyperstack...", "title=h type=16-bit display=Color width=8 height=6 channels=2 "
          "slices=3 frames=2 label")
    hs = s.active
    assert hs.dims == (2, 3, 2) and hs.composite_mode == "color" and \
        hs.slice_labels[0] == "c:1/2 z:1/3 t:1/2"
    s.set_active(doc)
    doc.pixels[...] = 7
    out = tmp_path / "n.tif"
    s.run("Tiff...", f"save=[{out}]")
    assert out.exists() and doc.title == "n.tif" and not doc.changed
    doc.pixels[...] = 9
    s.run("Revert")
    assert int(doc.pixels.max()) == 7
    for label, ext in (("PNG...", ".png"), ("Jpeg...", ".jpg"), ("Gif...", ".gif"),
                       ("ZIP...", ".zip"), ("Raw Data...", ".raw"), ("Text Image...", ".txt"),
                       ("BMP...", ".bmp"), ("PGM...", ".pgm"), ("FITS...", ".fits"),
                       ("LUT...", ".lut")):
        s.run(label, f"save=[{tmp_path / ('x' + ext)}]")
        assert (tmp_path / ("x" + ext)).exists(), label
    doc.roi = S.RectRoi(1, 1, 5, 5)
    s.run("Selection...", f"save=[{tmp_path / 'r.roi'}]")
    s.run("XY Coordinates...", f"save=[{tmp_path / 'xy.txt'}]")
    assert (tmp_path / "xy.txt").read_text().splitlines()[0] == "1\t1"
    s.results.add_row({"Area": 3})
    s.run("Results...", f"save=[{tmp_path / 'res.csv'}]")
    assert (tmp_path / "res.csv").exists()
    seq = tmp_path / "seq"
    s.run("Image Sequence... ", f"format=PNG name=p start=1 digits=3 save=[{seq}]")
    assert sorted(p.name for p in seq.iterdir()) == ["p001.png", "p002.png", "p003.png"]
    s.run("Image Sequence...", f"dir=[{seq}] sort")
    assert s.active.n_slices == 3 and s.active.slice_labels[0] == "p001.png"
    s.set_active(hs)
    s.run("AVI... ", f"compression=PNG frame=5 save=[{tmp_path / 'm.avi'}]")
    s.run("AVI...", f"open=[{tmp_path / 'm.avi'}]")
    assert s.active.n_slices == 6
    s.run("Raw...", f"open=[{tmp_path / 'x.raw'}] image=8-bit width=20 height=10 number=3")
    assert s.active.n_slices == 3 and int(s.active.pixels.max()) == 7
    s.run("Text Image... ", f"open=[{tmp_path / 'x.txt'}]")
    assert s.active.type == 2
    n = len(s.images)
    s.run("Close")
    assert len(s.images) == n - 1
    s.run("Close All")
    assert s.images == []


def test_open_dialog_recent_and_open_next(bar, tmp_path):
    from rsstatchem.core.imaging.io import save_image
    for i in (1, 2, 10):
        save_image(ImageDoc.from_array(np.full((4, 4), i, np.uint8), title=f"f{i}"),
                   tmp_path / f"f{i}.tif")
    bar.file_hook = lambda kind, title, default: str(tmp_path / "f2.tif")
    bar.run_command("Open...")
    doc = bar.session.active
    assert doc.title == "f2.tif"
    bar.run_command("Open Next")
    assert bar.session.active.title == "f10.tif" and doc not in bar.session.images
    bar.file_hook = lambda kind, title, default: str(tmp_path / "saved.tif")
    bar.session.active.title = "renamed"
    bar.run_command("Tiff...")
    assert (tmp_path / "saved.tif").exists()


def test_print_and_page_setup_to_pdf(qapp, tmp_path):
    from PySide6.QtPrintSupport import QPrinter

    from rsstatchem.ui.imagej.commands_file import print_image
    doc = ImageDoc.from_array(np.arange(64 * 32, dtype=np.uint16).reshape(32, 64))
    doc.overlay = S.Overlay([S.RectRoi(2, 2, 10, 10)])
    printer = QPrinter()
    printer.setOutputFormat(QPrinter.OutputFormat.PdfFormat)
    printer.setOutputFileName(str(tmp_path / "p.pdf"))
    print_image(doc, printer, header=True)
    assert (tmp_path / "p.pdf").read_bytes()[:4] == b"%PDF"


def test_project_save_and_open_through_bar(bar, tmp_path):
    from rsstatchem.core.imaging.project import save_project
    doc, win = _new(bar, np.arange(12, dtype=np.uint8).reshape(3, 4), title="p.tif")
    bar.session.roi_manager.append(S.RectRoi(0, 0, 1, 1, name="r"))
    path = save_project(bar.session, tmp_path / "proj.rsij")
    bar.run_command("Close All")
    bar.open_path(str(path))
    assert [d.title for d in bar.session.images] == ["p.tif"]
    assert bar.session.roi_manager[-1].name == "r"


def test_close_prompts_for_changed_images(bar, tmp_path):
    doc, win = _new(bar, np.zeros((5, 5), np.uint8), title="c")
    doc.changed = True
    bar.question_hook = lambda title, text: "cancel"
    win.close()
    assert doc in bar.session.images
    bar.question_hook = lambda title, text: "yes"
    bar.file_hook = lambda kind, title, default: str(tmp_path / "c.tif")
    win.close()
    assert doc not in bar.session.images and (tmp_path / "c.tif").exists()


# ---------------------------------------------------------------------------------- Plugins
def test_recorder_records_commands_selections_and_files(bar, tmp_path):
    from rsstatchem.ui.imagej import toolbar as T
    bar.run_command("Record...")
    rec = bar.session.recorder
    assert rec is not None and rec.language == "Macro"
    p = tmp_path / "a.tif"
    from rsstatchem.core.imaging.io import save_image
    save_image(ImageDoc.from_array(np.zeros((20, 30), np.uint8)), p)
    bar.open_path(str(p))
    win = _window(bar, bar.session.active)
    bar.toolbar.set_tool(T.RECTANGLE)
    drag(win.canvas, (2, 3), (12, 13))
    bar.run_command("Invert", "")
    bar.toolbar.set_tool(T.POLYGON)
    for q in ((20, 15), (28, 15), (24, 19)):
        mouse(win.canvas, "press", *q)
        mouse(win.canvas, "release", *q)
    mouse(win.canvas, "double", 24, 19)
    assert rec.lines == [f'open("{p.as_posix()}");', "makeRectangle(2, 3, 10, 10);",
                         'run("Invert");', "makePolygon(20, 15, 28, 15, 24, 19);"]
    from rsstatchem.ui.imagej.frames.recorder import RecorderWindow
    from rsstatchem.ui.imagej.windows import WINDOWS
    rw = next(w for w in WINDOWS if isinstance(w, RecorderWindow))
    assert rw.text.toPlainText().splitlines()[1] == "makeRectangle(2, 3, 10, 10);"
    rw.language.setCurrentText("Python")
    bar.run_command("Invert", "")
    assert rec.lines[-1] == "ij.run('Invert')"
    editor = rw.create_macro()
    assert "makePolygon" in editor.contents()
    rw.close()
    assert bar.session.recorder is None


def test_recorded_macro_replays(bar):
    """What the recorder writes runs again through the macro interpreter."""
    bar.run_command("Record...")
    bar.session.run("Image...", "name=r type=8-bit fill=Ramp width=40 height=20 slices=1")
    bar.session.active.roi = S.RectRoi(5, 5, 10, 5)
    bar.record_roi(bar.session.active.roi)
    bar.session.run("Fill", "slice")
    code = bar.session.recorder.text()
    expected = bar.session.active.pixels.copy()
    bar.run_command("Close All")
    bar.run_macro_text(code)
    assert np.array_equal(bar.session.active.pixels, expected)


def test_macro_language_subset():
    from rsstatchem.core.imaging.macro import MacroError, n2s, parse, run_macro
    from rsstatchem.ui.imagej.bar import register_area_commands
    s = Session()
    register_area_commands(s.registry)
    run_macro(s, r'''
        var g = 5;
        function sq(x) { return x * x; }
        newImage("a", "8-bit ramp", 64, 32, 3);
        getDimensions(w, h, c, z, t);
        print("dims " + w + "x" + h + " z=" + z + " " + (sq(3) + g) + " " + 1/3);
        s = "Hello";
        if (s == "hello" && s.length == 5) print("case-insensitive " + toUpperCase(s));
        a = newArray(3, 1, 2); Array.sort(a); print(String.join(a));
        sum = 0; for (i = 0; i < a.length; i++) { if (i == 1) continue; sum += a[i]; }
        k = 0; while (true) { k++; if (k > 4) break; }
        do { k--; } while (k > 2);
        print(sum + " " + k + " " + (5 ^ 3) + " " + (7 % 3) + " " + d2s(3.14159, 2));
        setSlice(2); makeRectangle(10, 5, 20, 10);
        getSelectionBounds(x, y, sw, sh);
        print(getSliceNumber() + " " + x + "," + y + "," + sw + "," + sh + " " + getValue("Mean"));
        run("Invert", "slice"); print(getValue("Mean"));
        setThreshold(100, 200); getThreshold(lo, hi); print(lo + "-" + hi);
        setResult("Area", 0, 12.5); print(nResults + " " + getResult("Area", 0));
        print(split("a,b,,c", ",").length + " " + replace("abc123", "[0-9]+", "#"));
    ''')
    assert s.log_lines == ["dims 64x32 z=3 14 0.3333", "case-insensitive HELLO", "1,2,3",
                           "4 2 6 1 3.14", "2 10,5,20,10 78", "177", "100-200", "1 12.5",
                           "4 abc#"]
    assert [n2s(v) for v in (1234.56789, 0.1, -0.5, 7.0)] == ["1234.5679", "0.1", "-0.5", "7"]
    with pytest.raises(MacroError, match="line 2"):
        run_macro(s, "x = 1;\ny = undefinedVar + 1;")
    assert [n for n, _ in parse('macro "A" {print(1);} macro "B" {}').macros] == ["A", "B"]


def test_python_scripts_and_install_macros(bar, tmp_path):
    from rsstatchem.core.imaging.macro import run_python
    s = bar.session
    run_python(s, "ij.newImage('p', '16-bit black', 8, 4, 1)\n"
                  "ij.setPixel(1, 1, 300)\n"
                  "ij.run('Invert', slice=True)\n"
                  "w, h, c, z, t = ij.getDimensions()\n"
                  "result = (w, h, ij.getPixel(1, 1))")
    assert s.active.title == "p" and s.active.raw_value(1, 1) == 0.0
    names = bar.install_macros('macro "Hello" { print("hello " + nImages); }\n'
                               'macro "Bye" { print("bye"); }')
    assert names == ["Hello", "Bye"]
    texts = [a.text() for a in bar.menus["Plugins>Macros"].actions()]
    assert "Hello" in texts and "Bye" in texts
    bar.run_installed("Bye")
    assert s.log_lines[-1] == "bye"
    f = tmp_path / "m.ijm"
    f.write_text('print("from file " + getArgument());')
    s.run("Run...", f"macro=[{f}] argument=xyz")
    assert s.log_lines[-1] == "from file xyz"
    bar.run_macro_text("x = ;")
    assert bar.messages and bar.messages[-1][0] == "Macro Error"


def test_command_finder_repeat_shortcuts_and_lists(bar):
    from rsstatchem.ui.imagej.frames.command_finder import CommandFinder, search
    cf = CommandFinder(bar)
    cf.query.setText("gauss blur")
    labels = [cf.table.item(r, 0).text() for r in range(cf.table.rowCount())]
    assert "Gaussian Blur..." in labels
    assert "Invert LUT" in [r[0] for r in search(cf.rows, "invert lut")]
    cf.query.setText("Image...")
    cf.close_when_running.setChecked(False)
    bar.message_hook = lambda *a: None
    doc, win = _new(bar, np.zeros((4, 4), np.uint8), title="r")
    bar.session.run("Invert")
    bar.run_command("Repeat Command")
    assert int(doc.pixels.max()) == 0                    # inverted twice
    s = bar.session
    s.run("Add Shortcut... ", "command=Invert shortcut=F5")
    assert s.options.extra["shortcuts"]["F5"] == "Invert"
    rt = s.run("List Shortcuts")
    assert "Ctrl+N" in [rt.get_value("Shortcut", i) for i in range(len(rt))]
    rt = s.run("List Commands")
    assert len(rt) > 390
    s.run("Remove Shortcut...", "shortcut=F5")
    assert "F5" not in s.options.extra["shortcuts"]
    s.set_active(doc)
    cap = s.run("Capture Image")
    assert cap.type == COLOR_RGB and cap.width == 4
    s.run("Reset...", "undo clipboard")
    assert not doc.can_undo()


def test_menus_complete_for_wp_a_areas(bar):
    from rsstatchem.core.imaging.menu_layout import labels_of_area
    reg = bar.session.registry
    missing = [lb for area in ("file", "edit", "plugins") for lb in labels_of_area(area)
               if lb not in reg]
    assert missing == []
    assert all(bar.actions_by_label[lb].isEnabled()
               for area in ("file", "edit", "plugins") for lb in labels_of_area(area))


def test_color_picker_log_text_windows(bar):
    from rsstatchem.ui.imagej.frames.color_picker import ColorPicker, palette
    cp = ColorPicker(bar)
    cp.pick(palette()[3][0], False)
    assert bar.session.options.foreground == palette()[3][0]
    cp.swap()
    assert bar.session.options.background == palette()[3][0]
    bar.session.log("one")
    bar.session.log("two")
    log = bar.window_manager.log_window
    assert log.contents().splitlines() == ["one", "two"]
    bar.session.log("\\Clear")
    assert log.contents() == ""
    bar.run_command("Macro")
    from rsstatchem.ui.imagej.frames.text_window import TextWindow
    from rsstatchem.ui.imagej.windows import WINDOWS
    ed = [w for w in WINDOWS if isinstance(w, TextWindow) and w.macro][-1]
    ed.text.setPlainText('print("ran");')
    ed.run_macro()
    assert bar.session.log_lines[-1] == "ran"


def test_canvas_draws_overlay_threshold_and_image_roi(bar):
    from rsstatchem.core.imaging.image import Threshold
    from rsstatchem.ui.imagej.canvas import threshold_rgb
    doc, win = _new(bar, np.arange(100, dtype=np.uint8).reshape(10, 10), title="o")
    rgb = threshold_rgb(np.zeros((10, 10, 3), np.uint8), doc.plane(), Threshold(50, 99, "red"))
    assert rgb[9, 9].tolist() == [255, 0, 0] and rgb[0, 0].tolist() == [0, 0, 0]
    doc.threshold = Threshold(50, 99, "red")
    try:
        from rsstatchem.ui.imagej.commands_image import ImageRoi
        img_roi = ImageRoi(x=1, y=1, width=4, height=4,
                           pixels=np.full((4, 4, 3), (0, 255, 0), np.uint8), opacity=1.0)
    except ImportError:
        img_roi = S.RectRoi(1, 1, 4, 4)
    doc.overlay = S.Overlay([S.RectRoi(0, 0, 5, 5, stroke_color="#ff0000"),
                             S.CompositeRoi(parts=[("or", S.RectRoi(2, 2, 6, 6)),
                                                   ("not", S.RectRoi(4, 4, 2, 2))]),
                             img_roi, S.TextRoi(x=1, y=1, text="T")])
    doc.roi = S.PointRoi([8, 9], [8, 8])
    bar.session.image_updated(doc)
    img = win.canvas.grab().toImage()
    assert img.width() >= 10
    c = img.pixelColor(int(win.canvas.image_to_screen(2.5, 2.5)[0]),
                       int(win.canvas.image_to_screen(2.5, 2.5)[1]))
    assert c.isValid()
    if hasattr(img_roi, "pixels"):
        px = img.pixelColor(*(int(v) for v in win.canvas.image_to_screen(3.5, 1.5)))
        assert (px.red(), px.green(), px.blue()) == (0, 255, 0)
