"""WP-C: ROI masks (PolygonFiller / OvalRoi ports), geometry, Edit > Selection operations,
the wand and .roi / RoiSet.zip I/O."""
from __future__ import annotations

import csv
import math

import numpy as np
import pytest
from ij_golden import blobs, golden_csv, golden_images, golden_path

from rsstatchem.core.imaging.image import ImageDoc, Threshold
from rsstatchem.core.imaging.roi import geometry as G
from rsstatchem.core.imaging.roi import selection_ops as SO
from rsstatchem.core.imaging.roi import shapes as S
from rsstatchem.core.imaging.roi.mask import oval_mask, polygon_mask
from rsstatchem.core.imaging.roi.wand import Wand, do_wand


def _spec_roi(spec):
    p = spec.split(",")
    kind, v = p[0], [float(t) for t in p[1:]]
    return {"rect": lambda: S.RectRoi(*v), "oval": lambda: S.OvalRoi(*v),
            "poly": lambda: S.PolygonRoi(v[0::2], v[1::2])}[kind]()


def test_masks_equal_imagej_exactly():
    masks = golden_images("masks")
    with open(golden_path("masks", "roi_specs.csv"), newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 12
    for row in rows:
        ours = _spec_roi(row["spec"]).get_mask((64, 64))
        assert np.array_equal(ours, masks[row["name"]] > 0), row["name"]


def test_polygon_filler_conventions():
    # pixel-corner coordinates: a unit square covers exactly one pixel
    assert polygon_mask([2, 3, 3, 2], [2, 2, 3, 3], (5, 5)).sum() == 1
    # left border outside, right border inside after the 0.5 shift
    m = polygon_mask([0.5, 2.5, 2.5, 0.5], [0, 0, 1, 1], (2, 4))
    assert m[0].tolist() == [False, True, True, False]
    # several polygons separated by NaN: even-odd
    xs = [0, 6, 6, 0, np.nan, 2, 4, 4, 2]
    ys = [0, 0, 6, 6, np.nan, 2, 2, 4, 4]
    m = polygon_mask(xs, ys, (6, 6))
    assert m.sum() == 32 and not m[3, 3]
    # partly outside the image
    assert S.PolygonRoi([-5, 5, 5, -5], [-5, -5, 5, 5]).get_mask((10, 10)).sum() == 25
    assert oval_mask(1, 1).sum() == 1


def test_traced_outlines_round_trip_blobs():
    m = golden_images("masks")["blobs_mask_Default_bb0"] > 0
    loops = G.trace_mask_outlines(m)
    acc = np.zeros(m.shape, bool)
    for xs, ys in loops:
        acc ^= polygon_mask(xs, ys, m.shape)
    assert np.array_equal(acc, m)
    roi = SO.mask_to_roi(m)
    assert isinstance(roi, S.CompositeRoi) and roi.get_mask(m.shape).sum() == m.sum()


def test_geometry_bounds_hull_feret_lengths():
    assert G.int_bounds(S.RectRoi(2.5, 3.2, 4.1, 5)) == (2, 3, 5, 5)
    assert G.int_bounds(S.LineRoi(10, 20, 200, 180)) == (10, 20, 191, 161)
    hx, hy = G.convex_hull([0, 4, 4, 0, 2, 2], [0, 0, 4, 4, 2, 1])
    assert len(hx) == 4 and G.polygon_area(hx, hy) == 16
    a = G.feret_values(S.RectRoi(0, 0, 30, 40))
    assert a[0] == pytest.approx(50) and a[2] == pytest.approx(30)
    o = G.feret_values(S.OvalRoi(100, 50, 71, 90))
    assert (o[0], o[1], o[2]) == (90, 90.0, 71)
    # perimeters: rectangle, traced (corners count sqrt 2), oval (Ramanujan), line
    assert G.roi_length(S.RectRoi(0, 0, 20, 15)) == 70
    sq = S.PolygonRoi([0, 1, 1, 0], [0, 0, 1, 1], type=S.TRACED_ROI)
    assert G.roi_length(sq) == pytest.approx(4 - 2 * (2 - math.sqrt(2)))
    assert G.roi_length(S.OvalRoi(100, 50, 71, 90)) == pytest.approx(253.779500416)
    assert G.roi_length(S.LineRoi(10, 20, 200, 180)) == pytest.approx(248.394846967)
    ang = S.PolygonRoi([0, 10, 10], [0, 0, -10], type=S.ANGLE)
    assert G.polygon_angle(ang) == pytest.approx(90.0)


def test_oval_float_polygon_hull_area_matches_imagej_solidity():
    # ImageJ: Solidity 1.003995206 = 5026 / convex hull area 5006 (flattened Ellipse2D)
    hx, hy = G.float_convex_hull(S.OvalRoi(100, 50, 71, 90))
    assert G.int_polygon_area(hx, hy) == 5006


def test_wand_tracing():
    a = np.zeros((20, 20), np.uint8)
    a[5:10, 5:12] = 200
    a[7, 8] = 0                                  # a hole is not excluded by the wand
    w = Wand(a)
    assert w.auto_outline_threshold(7, 6, 100, 255)
    assert (min(w.xpoints), min(w.ypoints), max(w.xpoints), max(w.ypoints)) == (5, 5, 12, 10)
    assert w.auto_outline(0, 0, 0.0, 8) and w.npoints == 4  # background around the box
    im = ImageDoc.from_array(a, "YX", "w")
    im.threshold = Threshold(100, 255)
    roi = do_wand(im, 2, 7)                     # left of the object: finds its border
    assert roi.type == S.TRACED_ROI and roi.get_mask((20, 20)).sum() == 35


def test_selection_operations():
    im = ImageDoc.new(100, 80)
    rect = S.RectRoi(10, 10, 40, 30)
    inv = SO.make_inverse(im, rect)
    assert inv.get_mask((80, 100)).sum() == 8000 - 1200
    e = SO.enlarge(im, rect, 5)
    assert G.int_bounds(e) == (5, 5, 50, 40)
    poly = S.PolygonRoi([20, 60, 40], [20, 25, 60])
    big = SO.enlarge(im, poly, 3)
    assert big.get_mask((80, 100)).sum() > poly.get_mask((80, 100)).sum()
    small = SO.enlarge(im, S.OvalRoi(20, 20, 40, 40), -5)
    assert G.int_bounds(small) == (25, 25, 30, 30)
    band = SO.make_band(im, S.RectRoi(30, 30, 20, 20), 5)
    bm = band.get_mask((80, 100))
    assert not bm[35:45, 35:45].any() and bm[27, 40]
    c = SO.fit_circle(im, S.RectRoi(10, 10, 20, 20))
    assert isinstance(c, S.OvalRoi) and abs(c.width - 22.6) < 1
    pts = S.PolygonRoi(50 + 20 * np.cos(np.linspace(0, 6, 12)),
                       40 + 20 * np.sin(np.linspace(0, 6, 12)), type=S.POLYLINE)
    c2 = SO.fit_circle(im, pts)
    assert (c2.x, c2.y, c2.width) == (30, 20, 40)
    el = SO.fit_ellipse(im, S.OvalRoi(10, 20, 60, 30))
    assert isinstance(el, S.EllipseRoi) and el.aspect_ratio == pytest.approx(0.5, abs=0.02)
    rr = SO.fit_rectangle(S.PolygonRoi([10, 50, 40, 0], [0, 30, 43, 13]))
    assert isinstance(rr, S.RotatedRectRoi) and rr.rect_width == pytest.approx(16.4, abs=0.1)
    assert math.hypot(rr.x2 - rr.x1, rr.y2 - rr.y1) == pytest.approx(50, abs=0.5)
    hull = SO.convex_hull(S.PolygonRoi([0, 10, 5, 10, 0], [0, 0, 5, 10, 10]))
    assert len(hull.xs) == 4
    rot = SO.rotate(S.RectRoi(0, 0, 10, 2), 90)
    x, y, w, h = rot.bounds()
    assert (round(w, 6), round(h, 6)) == (2, 10)
    sc = SO.scale(S.RectRoi(10, 10, 10, 10), 2, 2, centered=True)
    assert (sc.x, sc.y, sc.width) == (5, 5, 20)
    assert SO.to_bounding_box(poly).bounds() == (20, 20, 40, 40)
    la = SO.line_to_area(S.LineRoi(10, 10, 50, 10), 5)
    assert isinstance(la, S.RotatedRectRoi) and la.get_mask((80, 100)).sum() == 200
    pl = SO.line_to_area(S.PolygonRoi([10, 40, 40], [10, 10, 40], type=S.POLYLINE), 3)
    assert pl.is_area() and pl.get_mask((80, 100)).sum() > 150
    al = SO.area_to_line(S.RectRoi(10, 10, 20, 10))
    assert al.type == S.POLYLINE and len(al.xs) == 5
    it = SO.interpolate(S.RectRoi(0, 0, 10, 10), 1.0)
    assert len(it.xs) == 40
    sp = SO.fit_spline(S.PolygonRoi([10, 50, 40, 15], [10, 15, 50, 40]))
    assert sp.spline_fit and len(sp.xs) >= 100
    spec = SO.specify(im, 20, 10, 50, 40, oval=True, centered=True)
    assert isinstance(spec, S.OvalRoi) and (spec.x, spec.y) == (40, 35)


def test_selection_ops_and_profiles_match_imagej():
    """Golden case ``selection``: Enlarge (grow/shrink), Make Band, Convex Hull, Fit Ellipse,
    Make Inverse masks and line/rectangle profiles of ImageJ 1.54p."""
    from rsstatchem.core.imaging.profile import line_profile
    im = blobs()
    g = golden_images("selection")
    poly = S.PolygonRoi([30, 120, 100, 40], [30, 40, 140, 110])
    cases = {"enlarge_polygon": SO.enlarge(im, poly, 7),
             "shrink_polygon": SO.enlarge(im, poly, -6),
             "band_oval": SO.make_band(im, S.OvalRoi(60, 50, 80, 60), 8),
             "convex_hull": SO.convex_hull(S.PolygonRoi([30, 120, 75, 100, 40],
                                                        [30, 40, 60, 140, 110])),
             "fit_ellipse": SO.fit_ellipse(im, poly),
             "inverse_rect": SO.make_inverse(im, S.RectRoi(40, 40, 60, 30))}
    for name, roi in cases.items():
        assert np.array_equal(roi.get_mask((254, 256)), g[name] == 255), name
    for name, roi in (("profile_line", S.LineRoi(10, 20, 200, 180)),
                      ("profile_rect", S.RectRoi(20, 30, 80, 60))):
        ij = golden_csv("selection", f"{name}.csv").column("Y")
        assert np.allclose(line_profile(im, roi), ij, rtol=0, atol=1e-8), name


def test_create_mask_and_selection():
    im = blobs()
    im.threshold = Threshold(126, 255)
    mask = SO.create_mask(im, None)
    assert mask.title == "mask" and int((mask.plane() == 255).sum()) == int(
        (im.plane() >= 126).sum())
    ij = golden_images("masks")["blobs_mask_Default_bb0"]
    assert np.array_equal(mask.plane() == 255, ij == 255)
    roi = SO.create_selection(im)
    assert roi.get_mask((im.height, im.width)).sum() == (im.plane() >= 126).sum()
    m2 = SO.create_mask(im, S.OvalRoi(0, 0, 10, 10))
    assert m2.title == "Mask" and m2.is_inverted_lut() and (m2.plane() == 255).sum() == 80


def test_straighten():
    a = np.tile(np.arange(100, dtype=np.uint8), (60, 1))
    im = ImageDoc.from_array(a, "YX", "ramp")
    out = SO.straighten(im, S.LineRoi(10, 30, 60, 30), 9)
    assert out.height == 9 and out.width >= 50
    assert out.plane()[4, 5] == pytest.approx(15, abs=0.6)


@pytest.mark.parametrize("roi", [
    S.RectRoi(5, 6, 20, 10, name="r", stroke_color="#ff0000"),
    S.RectRoi(5.5, 6.25, 20, 10, corner_diameter=0),
    S.OvalRoi(3, 4, 30, 20, fill_color="#8000ff00"),
    S.PolygonRoi([1, 20, 10], [2, 3, 30], name="poly", position=(0, 3, 0)),
    S.PolygonRoi([1.5, 20.25, 10], [2, 3, 30.75], type=S.FREEROI),
    S.PolygonRoi([0, 4, 4, 0], [0, 0, 4, 4], type=S.TRACED_ROI),
    S.PolygonRoi([1, 20, 30], [2, 3, 10], type=S.POLYLINE),
    S.PolygonRoi([1, 20, 30], [2, 3, 10], type=S.FREELINE),
    S.PolygonRoi([1, 20, 30], [2, 3, 10], type=S.ANGLE),
    S.LineRoi(1, 2, 30, 40, width=5),
    S.PointRoi([1, 5, 9], [2, 6, 10], counters=np.array([0, 1, 1]),
               positions=np.array([1, 1, 2])),
    S.EllipseRoi(10, 10, 40, 30, 0.5),
    S.RotatedRectRoi(10, 10, 40, 30, rect_width=8),
    S.TextRoi(5, 5, "Hello"),
])
def test_roi_file_round_trip(tmp_path, roi):
    pytest.importorskip("roifile")
    from rsstatchem.core.imaging.roi.roi_io import read_roi, write_roi
    p = tmp_path / "a.roi"
    write_roi(p, roi)
    back = read_roi(p)
    assert type(back) is type(roi) and back.type == roi.type
    assert back.name == roi.name and back.stroke_color == roi.stroke_color
    assert back.position == roi.position
    if roi.is_area() and not isinstance(roi, S.TextRoi):
        assert np.array_equal(back.get_mask((50, 50)), roi.get_mask((50, 50)))
    if isinstance(roi, S.PointRoi):
        assert back.counters.tolist() == [0, 1, 1] and back.positions.tolist() == [1, 1, 2]
    if isinstance(roi, S.EllipseRoi):
        assert back.aspect_ratio == pytest.approx(0.5)
    if isinstance(roi, S.LineRoi):
        assert back.width == 5 and (back.x2, back.y2) == (30, 40)


def test_roiset_zip_and_composite(tmp_path):
    pytest.importorskip("roifile")
    from rsstatchem.core.imaging.roi.roi_io import read_rois, write_rois
    comp = S.CompositeRoi([("or", S.RectRoi(0, 0, 20, 20)), ("not", S.RectRoi(5, 5, 5, 5))])
    rois = [S.RectRoi(1, 1, 5, 5, name="a"), S.OvalRoi(2, 2, 9, 9, name="a"), comp]
    write_rois(tmp_path / "RoiSet.zip", rois)
    back = read_rois(tmp_path / "RoiSet.zip")
    assert len(back) == 3 and back[0].name == "a"
    assert back[2].get_mask((30, 30)).sum() == 375


@pytest.mark.parametrize("roi", [
    S.RectRoi(1, 2, 5, 5), S.OvalRoi(1, 2, 5, 5), S.PolygonRoi([1, 6, 3], [2, 2, 7]),
    S.LineRoi(1, 2, 6, 9), S.PointRoi([1, 4], [2, 5]), S.RotatedRectRoi(1, 2, 6, 9, rect_width=2),
    S.EllipseRoi(1, 2, 6, 9, 0.5), S.TextRoi(1, 2, "t"),
    S.CompositeRoi([("or", S.RectRoi(1, 2, 5, 5)), ("or", S.LineRoi(1, 2, 6, 9))])])
def test_translate_shifts_x_and_y(roi):
    """Every ROI type moves by (dx, dy) in both axes (incl. LineRoi y1/y2)."""
    xs0, ys0 = G.float_polygon(roi)
    moved = SO.translate(roi, 10, 20)
    xs1, ys1 = G.float_polygon(moved)
    assert np.allclose(xs1 - xs0, 10) and np.allclose(ys1 - ys0, 20)
    if isinstance(roi, S.LineRoi):
        assert (moved.x1, moved.y1, moved.x2, moved.y2) == (11, 22, 16, 29)


def test_golden_rect_measure_uses_imagej_bounds():
    g = golden_csv("measure", "uncalibrated_rect.csv")
    assert g.get_value("Width", 0) == G.int_bounds(S.RectRoi(20, 30, 80, 60))[2]
