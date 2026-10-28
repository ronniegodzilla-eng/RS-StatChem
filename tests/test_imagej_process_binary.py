"""Process > Binary and Process > Find Maxima against the real ImageJ 1.54p (golden case
``process_binary``, see tests/data_imagej/cases/process_binary.ijm). Every comparison is exact
(masks, EDMs in all output types, maxima coordinates and output images)."""
from __future__ import annotations

import csv

import numpy as np
import pytest
from ij_golden import golden_csv, golden_images, golden_path, input_path

from rsstatchem.core.imaging import binary as B
from rsstatchem.core.imaging import maxima as M
from rsstatchem.core.imaging.commands import CommandError
from rsstatchem.core.imaging.image import GRAY8, GRAY16, GRAY32, ImageDoc
from rsstatchem.core.imaging.io import open_image
from rsstatchem.core.imaging.roi import OvalRoi, PointRoi, RectRoi
from rsstatchem.core.imaging.session import Session
from rsstatchem.ui.imagej.dialogs import process_binary

# mask name -> (inverting LUT, black background) as in the macro
MASKS = {"blobsF": (True, False), "blobsT": (False, True), "partF": (False, False),
         "partT": (False, True), "big": (False, True)}
ITERS = [(1, 1), (2, 1), (3, 3), (1, 4)]


@pytest.fixture(scope="module")
def gold():
    return golden_images("process_binary")


def _same(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    if a.dtype == np.float32:
        np.testing.assert_allclose(a, b, rtol=1e-7, atol=0)
    else:
        assert np.array_equal(a, b), f"{int((a != b).sum())} pixels differ"


@pytest.mark.parametrize("name", list(MASKS))
def test_binary_morphology_matches_imagej(gold, name):
    inv, bb = MASKS[name]
    src = gold["in_" + name]
    fg = B.foreground_value(inv, bb)
    bg = 255 - fg
    for i, c in ITERS:
        tag = f"_i{i}c{c}"
        _same(B.erode(src, c, bg, i), gold[f"{name}_erode{tag}"])
        _same(B.dilate(src, c, bg, i), gold[f"{name}_dilate{tag}"])
        _same(B.binary_open(src, c, bg, i), gold[f"{name}_open{tag}"])
        _same(B.binary_close(src, c, bg, i), gold[f"{name}_close{tag}"])
    for i, c in (ITERS[0], ITERS[2]):
        _same(B.erode(src, c, bg, i, pad_edges=True), gold[f"{name}_erodepad_i{i}c{c}"])
        _same(B.binary_open(src, c, bg, i, pad_edges=True), gold[f"{name}_openpad_i{i}c{c}"])
    _same(B.outline(src, inv, bb), gold[f"{name}_outline"])
    _same(B.fill_holes(src, fg, bg), gold[f"{name}_fill"])
    _same(B.skeletonize(src, fg), gold[f"{name}_skel"])
    _same(B.binary_median(src, 3), gold[f"{name}_median_r3"])
    _same(B.binary_median(src, 1.5), gold[f"{name}_median_r1.5"])


@pytest.mark.parametrize("name", list(MASKS))
def test_edm_family_matches_imagej(gold, name):
    inv, bb = MASKS[name]
    src = gold["in_" + name]
    _same(B.watershed(src, inv, bb), gold[f"{name}_watershed"])
    for out in B.OUTPUT_TYPES:
        _same(B.distance_map(src, inv, bb, out), gold[f"{name}_edm_{out}"])
        _same(B.ultimate_points(src, inv, bb, out), gold[f"{name}_uep_{out}"])
        _same(B.voronoi(src, inv, bb, out), gold[f"{name}_voronoi_{out}"])


def test_float_edm_is_exact_euclidean_for_small_distances():
    from scipy import ndimage

    a = np.zeros((40, 50), np.uint8)
    a[5:35, 6:45] = 255
    a[18:22, 20:30] = 0
    edm = B.float_edm(a, 0)
    ref = ndimage.distance_transform_edt(a != 0).astype(np.float32)
    np.testing.assert_allclose(edm, ref, rtol=1e-6)


# ---------------------------------------------------------------------------------- maxima
IMAGES = {"synth8": "synth8.tif", "synth16": "synth16.tif", "synth32": "synth32.tif",
          "blobs": "blobs.gif"}
PROMS = {"synth8": (10, 50), "synth16": (1500, 8000), "synth32": (0.3, 1.5),
         "blobs": (10, 50)}
VARIANTS = {"plain": {}, "strict": {"strict": True}, "exclude": {"exclude": True},
            "light": {"light": True}, "strictexclude": {"strict": True, "exclude": True}}


@pytest.fixture(scope="module")
def golden_points():
    pts = {}
    with open(golden_path("process_binary", "maxima_points.csv")) as fh:
        for row in csv.DictReader(fh):
            lst = pts.setdefault(row["case"], [])
            if row["x"] != "-1":
                lst.append((int(float(row["x"])), int(float(row["y"]))))
    return pts


def _list_points(name):
    t = golden_csv("process_binary", name)
    if not len(t):
        return []
    return list(zip(t.column("X").astype(int).tolist(), t.column("Y").astype(int).tolist()))


@pytest.mark.parametrize("name", list(IMAGES))
def test_find_maxima_matches_imagej(gold, golden_points, name):
    doc = open_image(input_path(IMAGES[name]))
    for p in range(2):
        prom = PROMS[name][p]
        for vname, opt in VARIANTS.items():
            base = f"{name}_p{p}_{vname}"
            minima = doc.is_inverted_lut() != opt.get("light", False)
            vals = M.values_for_maxima(doc.plane(), minima, doc.display_range)
            kw = dict(strict=opt.get("strict", False),
                      exclude_on_edges=opt.get("exclude", False),
                      float_image=doc.type == GRAY32)
            res = M.find_maxima(vals, prom, output=M.POINT_SELECTION, **kw)
            assert res.points == golden_points[base], base
            assert res.points == _list_points(f"maxima_{base}_list.csv"), base
            for out, code in (("single", M.SINGLE_POINTS), ("tol", M.IN_TOLERANCE),
                              ("seg", M.SEGMENTED)):
                img = M.find_maxima(vals, prom, output=code, **kw).image
                _same(img, gold[f"maxima_{base}_{out}"])


def test_find_maxima_threshold_and_selections(gold):
    doc = open_image(input_path("synth8.tif"))
    v = M.values_for_maxima(doc.plane(), False)
    res = M.find_maxima(v, 10, threshold=100, output=M.LIST)
    assert res.points == _list_points("maxima_synth8_above_list.csv")
    _same(M.find_maxima(v, 10, threshold=100, output=M.SEGMENTED).image,
          gold["maxima_synth8_above_seg"])
    rect = (10, 8, 40, 30)
    assert M.find_maxima(v, 10, output=M.LIST, roi_rect=rect).points == \
        _list_points("maxima_synth8_rect_list.csv")
    _same(M.find_maxima(v, 10, output=M.IN_TOLERANCE, roi_rect=rect).image,
          gold["maxima_synth8_rect_tol"])
    mask = OvalRoi(5, 5, 50, 45).get_mask((64, 64))
    assert M.find_maxima(v, 10, output=M.LIST, roi_rect=(5, 5, 50, 45), roi_mask=mask).points \
        == _list_points("maxima_synth8_oval_list.csv")


def test_find_maxima_1d():
    xx = [0, 1, 5, 1, 0, 3, 3, 0, 9, 2]
    assert M.find_maxima_1d(xx, 0.5) == [8, 2, 5]
    assert M.find_maxima_1d(xx, 2.5, edge_mode=1) == [8, 2, 5]
    assert M.find_minima_1d(xx, 0.5) == [7, 4, 0, 9]   # equal depths: Tools.rank order


# ---------------------------------------------------------------------------------- commands
def _session(black_background=False):
    s = Session()
    s.options.black_background = black_background
    process_binary.register(s.registry)
    return s


def _open_mask(s, gold, name):
    inv, bb = MASKS[name]
    s.options.black_background = bb
    doc = ImageDoc.from_array(gold["in_" + name].copy(), "YX", name + ".tif")
    if inv:
        doc.luts = [doc.luts[0].inverted()]
    return s.add_image(doc)


def test_binary_commands_run_like_imagej_macros(gold):
    s = _session()
    doc = _open_mask(s, gold, "blobsF")
    s.run("Options...", "iterations=3 count=3 edm=Overwrite do=Nothing")
    assert (s.options.binary_iterations, s.options.binary_count) == (3, 3)
    assert not s.options.black_background
    s.run("Erode")
    _same(doc.plane(), gold["blobsF_erode_i3c3"])
    assert doc.threshold is not None and doc.threshold.mode == "none" and \
        doc.threshold.lower == 255
    doc.undo()
    _same(doc.plane(), gold["in_blobsF"])
    s.run("Options...", "iterations=1 count=1 edm=Overwrite do=Outline")
    _same(doc.plane(), gold["blobsF_outline"])
    doc.undo()
    s.run("Median", "radius=1.5")
    _same(doc.plane(), gold["blobsF_median_r1.5"])
    doc.undo()
    s.run("Watershed")
    _same(doc.plane(), gold["blobsF_watershed"])
    doc.undo()
    s.run("Distance Map")
    _same(doc.plane(), gold["blobsF_edm_Overwrite"])
    for out in ("8-bit", "16-bit", "32-bit"):
        doc.undo()
        s.run("Options...", f"iterations=1 count=1 edm={out}")
        for label, kind, prefix in (("Distance Map", "edm", "EDM of "),
                                    ("Ultimate Points", "uep", "UEPs of "),
                                    ("Voronoi", "voronoi", "Voronoi of ")):
            s.set_active(doc)
            new = s.run(label)
            assert new.title == prefix + "blobsF"
            assert new.is_inverted_lut()            # black background off
            _same(new.plane(), gold[f"blobsF_{kind}_{out}"])
            s.close_image(new)
    assert doc.type == GRAY8


def test_binary_commands_black_background_and_stack(gold):
    s = _session(True)
    a = gold["in_partT"]
    doc = s.add_image(ImageDoc.from_array(np.stack([a, a]), "ZYX", "stack.tif"))
    s.run("Options...", "iterations=2 count=1 black edm=Overwrite")
    s.run("Dilate", "stack")
    for z in range(2):
        _same(doc.plane(0, z, 0), gold["partT_dilate_i2c1"])
    s.run("Options...", "iterations=1 count=1 black edm=32-bit")
    edm = s.run("Distance Map", "stack")
    assert edm.n_slices == 2 and edm.type == GRAY32 and not edm.is_inverted_lut()


def test_binary_selection_masking_and_errors(gold):
    s = _session(True)
    doc = s.add_image(ImageDoc.from_array(gold["in_partT"].copy(), "YX", "p.tif"))
    doc.roi = RectRoi(10, 10, 60, 50)
    s.run("Options...", "iterations=1 count=1 black")
    s.run("Erode")
    full = B.erode(gold["in_partT"], 1, 0, 1)
    inside = np.zeros(full.shape, bool)
    inside[10:60, 10:70] = True
    assert np.array_equal(doc.plane()[inside], full[inside])
    assert np.array_equal(doc.plane()[~inside], gold["in_partT"][~inside])
    gray = s.add_image(ImageDoc.from_array(np.arange(64, dtype=np.uint8).reshape(8, 8), "YX"))
    with pytest.raises(CommandError):
        s.run("Erode")
    with pytest.raises(CommandError):
        s.run("Distance Map", image=gray)


def test_find_maxima_command_outputs(gold, golden_points):
    s = _session()
    doc = s.add_image(open_image(input_path("blobs.gif")))
    assert doc.is_inverted_lut()
    s.run("Find Maxima...", "prominence=10 output=[Point Selection]")
    assert isinstance(doc.roi, PointRoi)
    assert list(zip(doc.roi.xs.astype(int), doc.roi.ys.astype(int))) == \
        golden_points["blobs_p0_plain"]
    doc.roi = None
    rt = s.run("Find Maxima...", "prominence=10 strict exclude output=List")
    assert list(zip(rt.column("X").astype(int), rt.column("Y").astype(int))) == \
        golden_points["blobs_p0_strictexclude"]
    s.results.reset()
    s.run("Find Maxima...", "prominence=10 output=Count")
    s.run("Find Maxima...", "prominence=50 light output=Count")
    assert list(s.results.column("Count")) == [len(golden_points["blobs_p0_plain"]),
                                               len(golden_points["blobs_p1_light"])]
    out = s.run("Find Maxima...", "prominence=10 exclude output=[Segmented Particles]",
                image=doc)
    assert out.title == "blobs.gif Segmented" and out.is_inverted_lut()
    _same(out.plane(), gold["maxima_blobs_p0_exclude_seg"])
    out = s.run("Find Maxima...", "prominence=10 output=[Single Points]", image=doc)
    assert out.title == "blobs.gif Maxima"
    _same(out.plane(), gold["maxima_blobs_p0_plain_single"])
    assert not doc.changed


def test_find_maxima_command_16bit_and_threshold(gold):
    s = _session()
    d16 = s.add_image(open_image(input_path("synth16.tif")))
    assert d16.type == GRAY16
    out = s.run("Find Maxima...", "prominence=1500 output=[Maxima Within Tolerance]")
    _same(out.plane(), gold["maxima_synth16_p0_plain_tol"])
    d8 = s.add_image(open_image(input_path("synth8.tif")))
    from rsstatchem.core.imaging.image import Threshold
    d8.threshold = Threshold(100, 255, "red")
    rt = s.run("Find Maxima...", "prominence=10 above output=List", image=d8)
    assert list(zip(rt.column("X").astype(int), rt.column("Y").astype(int))) == \
        _list_points("maxima_synth8_above_list.csv")


def test_module_selftest():
    process_binary.selftest()
