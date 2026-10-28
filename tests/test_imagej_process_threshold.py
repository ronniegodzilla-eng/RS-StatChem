"""WP-B thresholding: AutoThresholder port (+ the 4 clean-room methods), setAutoThreshold
semantics, Make Binary / Convert to Mask, colour thresholding, the Threshold and Color Threshold
windows. Goldens from the real ImageJ 1.54p (tests/data_imagej/golden/{thresholds,masks,
process_threshold}); no Java needed."""
from __future__ import annotations

import csv
import time

import numpy as np
import pytest

from ij_golden import golden_images, golden_path, input_path, thresholds
from rsstatchem.core.imaging import threshold as T
from rsstatchem.core.imaging.image import COLOR_RGB, GRAY8, GRAY16, GRAY32, ImageDoc, Threshold
from rsstatchem.core.imaging.io import open_image
from rsstatchem.core.imaging.roi import RectRoi

pytest.importorskip("tifffile")

_CACHE: dict = {}


def _open(name: str) -> ImageDoc:
    if name not in _CACHE:
        _CACHE[name] = open_image(input_path(name))
    return _CACHE[name].duplicate()


def _gimg() -> dict:
    if "_g" not in _CACHE:
        _CACHE["_g"] = golden_images("process_threshold")
    return _CACHE["_g"]


def _stack(name: str) -> ImageDoc:
    arr = _gimg()[name]
    return ImageDoc.from_array(arr, "ZYX", name)


def _rows(fname: str) -> list[dict]:
    with open(golden_path("process_threshold", fname), newline="") as fh:
        return list(csv.DictReader(fh))


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-6 * max(1.0, abs(b))


# ------------------------------------------------------------------ auto thresholds
def test_all_140_threshold_goldens():
    gold = thresholds()
    assert len(gold) == 140
    bad = []
    for (img, method, dark), (lo, hi) in sorted(gold.items()):
        doc = _open(img)
        got = T.set_auto_threshold(doc, None if method == "legacy" else method, dark)
        if not (_close(got[0], lo) and _close(got[1], hi)):
            bad.append((img, method, dark, (lo, hi), got))
    assert bad == []


def _doc_for_case(case: str) -> ImageDoc:
    if case == "synth8":
        return _open("synth8.tif")
    if case == "particles8":
        return _open("particles8.tif")
    if case == "blobs_rect":
        doc = _open("blobs.gif")
        doc.roi = RectRoi(20, 30, 100, 80)
        return doc
    if case == "synth32_noreset":
        doc = _open("synth32.tif")
        doc.set_display_range(-1, 3)
        return doc
    if case == "synth16_16bit":
        return _open("synth16.tif")
    if case.startswith("stack16"):
        doc = _stack("stack16_input")
        if case.endswith("slice2"):
            doc.set_position(0, 1, 0)
        return doc
    if case.startswith("stack32"):
        return _stack("stack32_input")
    if case.startswith("stack8"):
        return _stack("stack8_input")
    raise KeyError(case)


def test_extra_threshold_goldens():
    """synth8 (17 methods +-dark), bilevel, ROI, no-reset, 16-bit histogram, stacks."""
    bad = []
    rows = _rows("extra_thresholds.csv")
    assert len(rows) == 80
    for row in rows:
        doc = _doc_for_case(row["case"])
        got = T.set_auto_threshold(doc, row["options"])
        lo, hi = float(row["lower"]), float(row["upper"])
        if not (_close(got[0], lo) and _close(got[1], hi)):
            bad.append((row["case"], row["options"], (lo, hi), got))
    # documented deviation: MinError on the 65536-bin "16-bit" histogram (ImageJ's
    # GPL-derived implementation is not reproduced; 256-bin MinError matches everywhere)
    known = {("synth16_16bit", "MinError 16-bit"), ("synth16_16bit", "MinError dark 16-bit")}
    assert {(b[0], b[1]) for b in bad} <= known, bad


def test_methods_list_and_errors():
    assert len(T.METHODS) == 17 and T.METHODS[0] == "Default" and "IJ_IsoData" in T.METHODS
    assert "Huang2" in T.FIJI_METHODS
    with pytest.raises(ValueError):
        T.auto_threshold([1, 2, 3, 4], "Nope")
    # bilevel images: the threshold sits just below the upper level
    assert T.auto_threshold([5] + [0] * 254 + [7], "Otsu") == 254
    # every method returns a bin index for a bimodal histogram
    h = np.zeros(256, int)
    h[40:60] = 100
    h[150:200] = 50
    h[90] = 3
    for m in T.FIJI_METHODS + ("IJ_IsoData",):
        assert 0 <= T.auto_threshold(h, m) < 256, m


def test_clean_room_methods_behave_like_papers():
    """Intermodes / Minimum (Prewitt & Mendelsohn), Percentile (Doyle), MinError
    (Kittler & Illingworth) on a synthetic two-Gaussian histogram."""
    x = np.arange(256)
    h = (1000 * np.exp(-0.5 * ((x - 60) / 10) ** 2) +
         3000 * np.exp(-0.5 * ((x - 180) / 15) ** 2)).astype(int)
    inter = T.auto_threshold(h, "Intermodes")
    mini = T.auto_threshold(h, "Minimum")
    assert 110 <= inter <= 130 and 85 <= mini <= 125
    perc = T.auto_threshold(h, "Percentile")
    frac = h[:perc + 1].sum() / h.sum()
    assert abs(frac - 0.5) < 0.02
    me = T.auto_threshold(h, "MinError")
    assert 80 <= me <= 120


def test_long_histogram_methods_are_fast():
    rng = np.random.default_rng(1)
    plane = np.concatenate([rng.normal(8000, 900, 3000), rng.normal(30000, 2500, 3000)])
    plane = np.clip(plane, 0, 65535).astype(np.uint16).reshape(60, 100)
    t0 = time.perf_counter()
    for m in ("Default", "Otsu", "Li", "MaxEntropy", "RenyiEntropy", "Triangle", "MinError"):
        lo, hi = T.set_auto_threshold(plane, m + " 16-bit")
        assert lo == 0 and 0 < hi < 65535, m
        if m != "Triangle":
            assert 8000 < hi < 30000, m
    assert time.perf_counter() - t0 < 30


# ------------------------------------------------------------------ masks
def _mask_rows() -> dict:
    return {r["name"]: r for r in _rows("mask_luts.csv")}


def _check_mask(doc: ImageDoc, name: str):
    g = _gimg()[name]
    row = _mask_rows()[name]
    assert doc.type == GRAY8
    got = doc.pixels[0, :, 0] if doc.stack_size > 1 else doc.plane()
    assert got.shape == g.shape, name
    assert np.array_equal(got, g), (name, int((got != g).sum()))
    assert doc.is_inverted_lut() == (row["inverted"] == "1"), name
    assert doc.threshold == Threshold(255.0, 255.0, "none")


def test_masks_phase0_goldens():
    g = golden_images("masks")
    for m in ("Default", "Otsu"):
        for bb in (0, 1):
            doc = _open("blobs.gif")
            T.set_auto_threshold(doc, m, mode="red")
            T.convert_to_mask(doc, bool(bb))
            assert np.array_equal(doc.plane(), g[f"blobs_mask_{m}_bb{bb}"])
            assert doc.is_inverted_lut() == (bb == 0)


@pytest.mark.parametrize("stem,file", [("blobs", "blobs.gif"), ("colony", "Cell_Colony.jpg"),
                                       ("synth8", "synth8.tif"), ("synth16", "synth16.tif"),
                                       ("synth32", "synth32.tif")])
@pytest.mark.parametrize("bb", [0, 1])
def test_convert_to_mask_auto(stem, file, bb):
    doc = _open(file)
    T.convert_to_mask(doc, bool(bb))
    _check_mask(doc, f"auto_{stem}_bb{bb}")


def test_convert_to_mask_presets():
    presets = {"huangdark": "Huang dark", "otsu": "Otsu", "triangledark": "Triangle dark",
               "minerror": "MinError", "percentiledark": "Percentile dark"}
    files = {"colony": "Cell_Colony.jpg", "synth16": "synth16.tif", "synth32": "synth32.tif"}
    for stem, file in files.items():
        for key, opts in presets.items():
            for bb in (0, 1):
                doc = _open(file)
                T.set_auto_threshold(doc, opts, mode="red")
                T.convert_to_mask(doc, bool(bb))
                _check_mask(doc, f"preset_{stem}_{key}_bb{bb}")


def test_make_binary_and_set_threshold():
    doc = _open("Cell_Colony.jpg")
    T.set_auto_threshold(doc, "Li dark", mode="red")
    T.make_binary(doc, True)
    _check_mask(doc, "makebinary_colony_li_bb1")
    doc = _open("blobs.gif")
    T.make_binary(doc, False)
    _check_mask(doc, "makebinary_blobs_bb0")
    doc = _open("synth32.tif")
    doc.threshold = Threshold(-0.5, 2.0, "red")
    T.convert_to_mask(doc, False)
    _check_mask(doc, "setthr_synth32_bb0")
    doc = _open("synth16.tif")
    doc.threshold = Threshold(15000, 30000, "red")
    T.convert_to_mask(doc, False)
    _check_mask(doc, "setthr_synth16_bb0")


def test_convert_rgb_to_mask():
    rgb = _gimg()["rgb_input"]
    doc = ImageDoc.from_array(rgb, "YXS", "rgb")
    assert doc.type == COLOR_RGB
    T.convert_to_mask(doc, False)
    _check_mask(doc, "auto_rgb_bb0")


def _check_stack(doc, name):
    g = _gimg()[name]
    row = _mask_rows()[name]
    got = doc.pixels[0, :, 0]
    assert got.shape == g.shape
    assert np.array_equal(got, g), (name, int((got != g).sum()))
    assert doc.is_inverted_lut() == (row["inverted"] == "1"), name


def test_convert_stacks():
    doc = _stack("stack16_input")
    T.convert_to_mask(doc, True, "Otsu", "Dark", calculate=True)
    _check_stack(doc, "stack16_otsu_dark_calc_black")
    doc = _stack("stack16_input")
    T.convert_to_mask(doc, False, "Default", "Default")
    _check_stack(doc, "stack16_legacy")
    doc = _stack("stack16_input")
    T.set_auto_threshold(doc, "Huang dark", mode="red")
    T.convert_to_mask(doc, True)
    _check_stack(doc, "stack16_huangdark_preset_black")
    doc = _stack("stack8_input")
    T.convert_to_mask(doc, False, "Default", "Default", calculate=True)
    _check_stack(doc, "stack8_default_calc")
    doc = _stack("stack8_input")
    T.convert_to_mask(doc, True, "Li", "Light", calculate=True)
    _check_stack(doc, "stack8_li_light_calc_black")
    doc = _stack("stack8_input")
    doc.set_position(0, 1, 0)
    T.set_auto_threshold(doc, "Otsu", mode="red")
    T.convert_to_mask(doc, False, "Otsu", "Light", only_current=True)
    _check_stack(doc, "stack8_only_slice2")
    doc = _stack("stack32_input")
    T.convert_to_mask(doc, False, "Triangle", "Dark", calculate=True)
    _check_stack(doc, "stack32_triangle_dark_calc")
    doc = _stack("stack32_input")
    T.set_auto_threshold(doc, "Otsu dark stack", mode="red")
    T.convert_to_mask(doc, True)
    _check_stack(doc, "stack32_otsustack_preset_black")
    # create new stack keeps the original
    doc = _stack("stack8_input")
    out = T.convert_to_mask(doc, False, "Otsu", "Dark", calculate=True, create_stack=True)
    assert out is not doc and doc.type == GRAY8 and out.title.startswith("MASK_")
    assert set(np.unique(out.pixels)) <= {0, 255} and len(np.unique(doc.pixels)) > 2
    lines = []
    T.convert_to_mask(_stack("stack16_input"), False, "Otsu", "Dark", calculate=True,
                      list_thresholds=True, log=lines.append)
    assert lines[0] == "Thresholding method: Otsu" and len(lines) == 4


# ------------------------------------------------------------------ colour spaces
def test_color_spaces_match_imagej():
    g = _gimg()
    rgb = g["rgb_input"]
    h, s, b = T.color_channels(rgb, "HSB")
    hsb = g["rgb_hsb"]
    assert np.array_equal(h, hsb[0]) and np.array_equal(s, hsb[1]) and np.array_equal(b, hsb[2])
    lab = g["rgb_lab"]
    for k, ch in enumerate(T.color_channels(rgb, "Lab")):
        assert np.array_equal(ch, lab[..., k]), k
    yuv = g["rgb_yuv"]
    for k, ch in enumerate(T.color_channels(rgb, "YUV")):
        assert np.array_equal(ch, yuv[..., k]), k
    r, gg, bb = T.color_channels(rgb, "RGB")
    assert np.array_equal(r, rgb[..., 0])


def test_color_threshold_filters():
    rgb = _gimg()["rgb_input"]
    chans = T.color_channels(rgb, "RGB")
    m = T.color_threshold(chans, "RGB", [(100, 255, True), (0, 255, True), (0, 255, True)])
    assert np.array_equal(m, rgb[..., 0] >= 100)
    m = T.color_threshold(rgb, "RGB", [(100, 255, "stop"), (0, 255, True), (0, 255, True)])
    assert np.array_equal(m, rgb[..., 0] < 100)
    ranges = T.color_auto_ranges(T.color_channels(rgb, "HSB"), "HSB", "Otsu")
    assert ranges[0] == (0, 255) and ranges[1] == (0, 255) and ranges[2][0] == 0
    hsb = T.color_channels(rgb, "HSB")
    sel = np.zeros(rgb.shape[:2], bool)
    sel[10:20, 10:20] = True
    sampled = T.sample_ranges(hsb, sel, "HSB")
    assert sampled[1][0] == int(hsb[1][sel].min()) and sampled[2][1] == int(hsb[2][sel].max())
    out = T.apply_color_mode(rgb, m, "Red")
    assert (out[m] == [255, 0, 0]).all() and np.array_equal(out[~m], rgb[~m])
    bw = T.apply_color_mode(rgb, m, "B&W", black_background=False)
    assert (bw[m] == 0).all() and (bw[~m] == 255).all()


def test_mask_to_roi_is_exact():
    rng = np.random.default_rng(3)
    mask = rng.random((40, 50)) > 0.7
    mask[5:25, 10:30] = True
    roi = T.mask_to_roi(mask)
    assert np.array_equal(roi.get_mask(mask.shape), mask)
    assert T.mask_to_roi(np.zeros((4, 4), bool)) is None


# ------------------------------------------------------------------ commands
@pytest.fixture
def session():
    from rsstatchem.core.imaging.session import Session
    from rsstatchem.ui.imagej.dialogs import process_threshold

    s = Session()
    process_threshold.register(s.registry)
    return s


def test_threshold_commands_headless(session):
    doc = session.add_image(_open("blobs.gif"))
    session.run("Threshold...", "method=Default")
    assert doc.threshold[:2] == (126.0, 255.0)
    session.run("Threshold...", "method=Otsu dark")
    assert doc.threshold[:2] == (0.0, 120.0)
    session.run("Threshold...", "lower=50 upper=90 display=B&W")
    assert doc.threshold == Threshold(50.0, 90.0, "bw")
    session.run("Convert to Mask")
    assert set(np.unique(doc.plane())) <= {0, 255}
    assert doc.undo() and doc.threshold == Threshold(50.0, 90.0, "bw")


def test_convert_to_mask_command_on_stack(session):
    doc = session.add_image(_stack("stack16_input"))
    cmd = session.registry.get("Convert to Mask")
    keys = [f.key for f in cmd.dialog_fields(session, doc)]
    assert keys[:5] == ["method", "background", "calculate", "only", "black"]
    session.run("Convert to Mask", "method=Otsu background=Dark calculate black")
    _check_stack(doc, "stack16_otsu_dark_calc_black")


def test_auto_threshold_fiji_command(session):
    doc = session.add_image(_open("blobs.gif"))
    session.run("Auto Threshold", "method=Otsu white")
    assert doc.type == GRAY8 and set(np.unique(doc.plane())) <= {0, 255}
    doc2 = session.add_image(_open("synth16.tif"))
    session.run("Auto Threshold", "method=Li white set")
    assert doc2.type == GRAY16 and doc2.threshold is not None
    session.set_active(doc)
    doc3 = session.add_image(_open("Cell_Colony.jpg"))
    n = len(session.images)
    session.run("Auto Threshold", "method=[Try all]")
    montage = session.images[-1]
    assert len(session.images) == n + 1 and montage.title.startswith("Montage")
    assert montage.height > doc3.height and montage.width > doc3.width


def test_color_threshold_command_headless(session):
    rgb = _gimg()["rgb_input"]
    doc = session.add_image(ImageDoc.from_array(rgb, "YXS", "rgb"))
    session.run("Color Threshold...",
                "space=RGB min1=100 max1=255 min2=0 max2=255 min3=0 max3=255 select")
    assert doc.roi is not None
    mask = doc.roi.get_mask((doc.height, doc.width))
    assert np.array_equal(mask, rgb[..., 0] >= 100)
    session.run("Color Threshold...", "space=HSB min1=0 max1=255 min2=0 max2=255 "
                "min3=128 max3=255 filtered color=White")
    assert doc.type == COLOR_RGB


def test_module_selftest():
    from rsstatchem.ui.imagej.dialogs import process_threshold
    process_threshold.selftest()


# ------------------------------------------------------------------ windows (offscreen Qt)
@pytest.fixture
def bar(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    from rsstatchem.ui.imagej.bar import ImageJBar
    from rsstatchem.ui.imagej import windows as W

    b = ImageJBar()
    W.BARS.append(b)
    b.message_hook = lambda title, text: None
    yield b
    b.close()


def test_threshold_window(bar):
    from rsstatchem.ui.imagej.frames.threshold_adjuster import ThresholdAdjuster
    from rsstatchem.ui.imagej import windows as W

    doc = bar.session.add_image(_open("blobs.gif"))
    bar.run_command("Threshold...")
    win = ThresholdAdjuster.instance
    assert isinstance(win, ThresholdAdjuster) and win in W.WINDOWS
    assert doc.threshold is not None and doc.threshold[:2] == (126.0, 255.0)
    win.method.setCurrentText("Otsu")
    assert doc.threshold[:2] == (121.0, 255.0)
    win.dark.setChecked(True)
    assert doc.threshold[:2] == (0.0, 120.0)
    win.min_slider.setValue(30)
    win.max_slider.setValue(100)
    assert doc.threshold[:2] == (30.0, 100.0)
    win.display.setCurrentText("B&W")
    assert doc.threshold.mode == "bw"
    win.set_levels(40, 90)
    assert doc.threshold[:2] == (40.0, 90.0)
    win.reset_threshold()
    assert doc.threshold is None
    win.auto()
    assert doc.threshold is not None
    win.apply()
    assert set(np.unique(doc.plane())) <= {0, 255}
    # 16-bit image: sliders work on the 0..255 scale, levels in raw values
    doc16 = bar.session.add_image(_open("synth16.tif"))
    win.dark.setChecked(False)
    win.method.setCurrentText("Default")
    win.auto()
    assert doc16.threshold[:2] == (0.0, 23958.0)
    win.close()
    assert win not in W.WINDOWS


def test_color_threshold_window(bar):
    from rsstatchem.ui.imagej.frames.color_threshold import ColorThresholdWindow

    rgb = _gimg()["rgb_input"]
    doc = bar.session.add_image(ImageDoc.from_array(rgb, "YXS", "rgb"))
    bar.run_command("Color Threshold...")
    win = ColorThresholdWindow.instance
    assert win is not None and win.doc is doc
    win.space.setCurrentText("RGB")
    win.set_range(0, 100, 255)
    win.set_range(1, 0, 255)
    win.set_range(2, 0, 255)
    assert np.array_equal(win.current_mask(), rgb[..., 0] >= 100)
    win.select()
    assert np.array_equal(doc.roi.get_mask(rgb.shape[:2]), rgb[..., 0] >= 100)
    win.filtered()
    assert (doc.plane()[rgb[..., 0] >= 100] == [255, 0, 0]).all()
    win.original()
    assert np.array_equal(doc.plane(), rgb)
    doc.roi = RectRoi(5, 5, 10, 10)
    win.sample()
    assert win.ranges()[0][0] == int(rgb[5:15, 5:15, 0].min())
    win.close()
