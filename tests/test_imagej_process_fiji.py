"""Fiji-style extras (Auto Local Threshold, CLAHE, MorphoLibJ-lite): behavioural tests.

ij.jar contains no Fiji plugins, so there are no ImageJ goldens for these commands; they are
clean-room rewrites from the papers. The tests compare with independent references: brute-force
neighbourhood loops, scipy.ndimage (BSD) operators, closed-form chamfer distances, iterated
geodesic dilation, and properties of the algorithms (clip limit, identity at slope 1, ...).
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage as ndi

from ij_golden import blobs, input_path
from rsstatchem.core.imaging.commands import CommandError
from rsstatchem.core.imaging.fiji import clahe as CL
from rsstatchem.core.imaging.fiji import local_threshold as LT
from rsstatchem.core.imaging.fiji import morpho as MO
from rsstatchem.core.imaging.image import GRAY16, GRAY32, ImageDoc
from rsstatchem.core.imaging.session import Session
from rsstatchem.ui.imagej.dialogs import process_fiji


def _particles():
    tifffile = pytest.importorskip("tifffile")
    return tifffile.imread(input_path("particles8.tif"))


def _rng_image(h=23, w=31, seed=3):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    return np.clip(rng.normal(100, 30, (h, w)) + 2 * xx, 0, 255).astype(np.uint8)


def _brute(a, fp, fn):
    """fn(values) over the footprint at every pixel, edges repeated."""
    k = fp.shape[0] // 2
    p = np.pad(a.astype(np.float64), k, mode="edge")
    out = np.zeros(a.shape)
    for y in range(a.shape[0]):
        for x in range(a.shape[1]):
            out[y, x] = fn(p[y:y + 2 * k + 1, x:x + 2 * k + 1][fp])
    return out


def _session(*docs):
    s = Session()
    process_fiji.register(s.registry)
    for d in docs:
        s.add_image(d)
    return s


# ---------------------------------------------------------------------------------- local threshold
def test_circular_kernel_is_imagej_rankfilters():
    assert LT.footprint(0.5).sum() == 5           # cross
    assert LT.footprint(1).sum() == 9             # 3x3
    assert LT.footprint(2).sum() == 21            # ImageJ's 21-pixel kernel for r=2
    fp = LT.footprint(15)
    assert fp.shape == (31, 31) and np.array_equal(fp, fp[::-1, ::-1])


@pytest.mark.parametrize("radius", [1, 2.5, 4])
def test_local_statistics_against_brute_force(radius):
    a = _rng_image()
    fp = LT.footprint(radius)
    mean, sd = LT.local_mean_std(a.astype(float), radius)
    np.testing.assert_allclose(mean, _brute(a, fp, np.mean), atol=1e-9)
    np.testing.assert_allclose(sd, _brute(a, fp, np.std), atol=1e-7)
    lo, hi = LT.local_min_max(a, radius)
    assert np.array_equal(lo, _brute(a, fp, np.min))
    assert np.array_equal(hi, _brute(a, fp, np.max))
    assert np.array_equal(LT.local_median(a, radius), _brute(a, fp, np.median))


def test_local_threshold_formulas():
    a = _rng_image()
    r = 3
    fp = LT.footprint(r)
    v = a.astype(float)
    mean, sd, med = _brute(a, fp, np.mean), _brute(a, fp, np.std), _brute(a, fp, np.median)
    lo, hi = _brute(a, fp, np.min), _brute(a, fp, np.max)
    assert np.array_equal(LT.local_threshold(a, "Mean", r, 5), v > mean - 5)
    assert np.array_equal(LT.local_threshold(a, "Median", r), v > med)
    assert np.array_equal(LT.local_threshold(a, "MidGrey", r), v > (lo + hi) / 2)
    assert np.array_equal(LT.local_threshold(a, "Niblack", r), v > mean + 0.2 * sd)
    assert np.array_equal(LT.local_threshold(a, "Niblack", r, -0.3, 4),
                          v > mean - 0.3 * sd - 4)
    assert np.array_equal(LT.local_threshold(a, "Sauvola", r),
                          v > mean * (1 + 0.5 * (sd / 128 - 1)))
    n = v / 255
    mn, sn = _brute(n, fp, np.mean), _brute(n, fp, np.std)
    t = mn * (1 + 2 * np.exp(-10 * mn) + 0.25 * (sn / 0.5 - 1))
    assert np.array_equal(LT.local_threshold(a, "Phansalkar", r), n > t)
    con = LT.local_threshold(a, "Contrast", r)
    assert np.array_equal(con, (hi - v) <= (v - lo))
    for m in LT.METHODS:   # white objects off = complement
        assert np.array_equal(LT.local_threshold(a, m, r, white_objects=False),
                              ~LT.local_threshold(a, m, r))


def test_bernsen_low_contrast_rule():
    a = np.full((9, 9), 140, np.uint8)
    a[:, 5:] = 150                                 # contrast 10 < 15: mid grey decides
    assert LT.local_threshold(a, "Bernsen", 2).all()
    a2 = a - 100                                   # mid grey < 128 -> background
    assert not LT.local_threshold(a2, "Bernsen", 2).any()
    a3 = np.full((9, 9), 100, np.uint8)
    a3[:, 5:] = 200                                # high contrast near the edge: v >= mid
    sel = LT.local_threshold(a3, "Bernsen", 2)
    assert sel[:, 6:].all() and not sel[:, :3].any()


def _otsu_brute(hist):
    total = hist.sum()
    levels = np.arange(hist.size)
    best, arg = -1.0, 0
    for k in range(hist.size):
        w0, w1 = hist[:k + 1].sum(), hist[k + 1:].sum()
        if w0 == 0 or w1 == 0:
            continue
        m0 = (levels[:k + 1] * hist[:k + 1]).sum() / w0
        m1 = (levels[k + 1:] * hist[k + 1:]).sum() / w1
        var = w0 * w1 * (m0 - m1) ** 2 / total ** 2
        if var > best + 1e-12:
            best, arg = var, k
    return arg


def test_local_otsu_histograms_and_threshold():
    a = _rng_image(15, 17)
    r = 2.5
    fp = LT.footprint(r)
    hist = LT.local_histograms(a, r, 0, a.shape[0])
    k = fp.shape[0] // 2
    p = np.pad(a, k, mode="edge")
    rng = np.random.default_rng(0)
    for y, x in rng.integers(0, 15, (12, 2)):
        win = p[y:y + 2 * k + 1, x:x + 2 * k + 1][fp]
        h = np.bincount(win, minlength=256)
        assert np.array_equal(hist[:, y, x], h)
        assert LT.otsu_from_histograms(h[:, None])[0] == _otsu_brute(h)
    t = LT.local_otsu_threshold(a, r)
    assert np.array_equal(LT.local_threshold(a, "Otsu", r), a > t)


def test_auto_local_threshold_command():
    doc = blobs()
    s = _session(doc)
    s.run("Auto Local Threshold", "method=Bernsen radius=15 parameter_1=0 parameter_2=0 white")
    p = doc.plane()
    assert set(np.unique(p).tolist()) == {0, 255}
    assert doc.is_inverted_lut()                   # black background off: inverting LUT
    assert 0.2 < (p == 255).mean() < 0.6
    assert doc.can_undo()
    doc2 = blobs()
    s.add_image(doc2)
    allm = s.run("Auto Local Threshold", "method=[Try all] radius=10 white")
    assert allm.n_slices == 9 and allm.slice_labels == list(LT.METHODS)
    sauv = LT.to_mask(LT.local_threshold(doc2.plane(), "Sauvola", 10))
    assert np.array_equal(allm.plane(0, 8, 0), sauv)
    s.add_image(ImageDoc.from_array(np.zeros((8, 8), np.uint16), "YX", "w16"))
    with pytest.raises(CommandError):
        s.run("Auto Local Threshold", "method=Mean radius=5")


# ---------------------------------------------------------------------------------- CLAHE
def test_clahe_identity_at_slope_one_and_range():
    a = _rng_image(40, 50)
    for fast in (False, True):
        out = CL.clahe(a, blocksize=15, slope=1.0, fast=fast)
        assert out.dtype == np.uint8
        assert np.abs(out.astype(int) - a).max() <= (0 if not fast else 1)
    out = CL.clahe(a, blocksize=15, slope=3.0)
    assert out.min() >= 0 and out.max() <= 255
    assert not np.array_equal(out, a)


def test_clahe_clip_limit_bounds_the_mapping_slope():
    hist = np.zeros(256)
    hist[100:110] = 1000.0                         # very peaked histogram
    hist += 1
    n = np.array(hist.sum())
    for slope in (1.5, 3.0, 6.0):
        clipped = CL.clip_histograms(hist, n, slope)
        assert clipped.sum() == pytest.approx(float(n))
        assert clipped.max() <= slope * n / 256 + 1e-6
        m = CL.mapping(clipped, n)
        assert np.diff(m).max() <= slope + 1e-6     # derivative of the mapping <= slope
        assert np.all(np.diff(m) >= 0)


def test_clahe_accurate_against_per_pixel_reference():
    a = _rng_image(20, 24)
    bs, slope = 9, 2.0
    out = CL.clahe(a, blocksize=bs, slope=slope)
    r = bs // 2
    for y, x in ((0, 0), (10, 12), (19, 23), (5, 20)):
        win = a[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1]
        h = np.bincount(win.ravel(), minlength=256).astype(float)
        n = np.array(float(win.size))
        m = CL.mapping(CL.clip_histograms(h, n, slope), n)
        assert out[y, x] == int(np.floor(m[a[y, x]] + 0.5))


def test_clahe_types_mask_and_command():
    a = _rng_image(30, 30).astype(np.uint16) * 100 + 500
    o16 = CL.clahe(a, blocksize=11)
    assert o16.dtype == np.uint16 and o16.min() >= a.min() and o16.max() <= a.max()
    f = (a / 7.0).astype(np.float32)
    o32 = CL.clahe(f, blocksize=11, fast=True)
    assert o32.dtype == np.float32 and np.isfinite(o32).all()
    rgb = np.stack([_rng_image(20, 20, s) for s in (1, 2, 3)], axis=-1)
    orgb = CL.clahe(rgb, blocksize=9)
    assert orgb.shape == rgb.shape and orgb.dtype == np.uint8
    mask = np.zeros((30, 30), bool)
    mask[:10] = True
    om = CL.clahe(a, blocksize=11, mask=mask)
    assert np.array_equal(om[10:], a[10:]) and not np.array_equal(om[:10], a[:10])
    doc = ImageDoc.from_array(_rng_image(40, 40), "YX", "c.tif")
    before = doc.plane().copy()
    s = _session(doc)
    s.run("Enhance Local Contrast (CLAHE)", "blocksize=15 histogram=256 maximum=3 mask=*None* "
                                            "fast_(less_accurate)")
    assert np.array_equal(doc.plane(), CL.clahe(before, 15, 256, 3.0, None, True))


# ---------------------------------------------------------------------------------- morphology
def test_structuring_elements():
    assert MO.strel("Square", 2).sum() == 25
    assert MO.strel("Diamond", 2).sum() == 13
    assert MO.strel("Line Horizontal", 3).sum() == 7
    assert MO.strel("Line 45 degrees", 2)[0, 4] and MO.strel("Line 135 degrees", 2)[0, 0]
    oc = MO.strel("Octagon", 3)
    assert MO.strel("Diamond", 3).sum() < oc.sum() < 49
    for shape in MO.SHAPES:
        fp = MO.strel(shape, 3)
        assert np.array_equal(fp, fp[::-1, ::-1]) and fp[3, 3]


@pytest.mark.parametrize("shape", ["Disk", "Square", "Line 45 degrees", "Octagon"])
def test_morphological_filters_against_brute_force(shape):
    a = _rng_image(14, 17)
    fp = MO.strel(shape, 2)
    ero, dil = _brute(a, fp, np.min), _brute(a, fp, np.max)
    assert np.array_equal(MO.morphological_filter(a, "Erosion", shape, 2), ero)
    assert np.array_equal(MO.morphological_filter(a, "Dilation", shape, 2), dil)
    opening = MO.morphological_filter(a, "Opening", shape, 2)
    assert np.array_equal(opening, _brute(ero.astype(np.uint8), fp, np.max))
    if not shape.startswith("Line"):   # replicated borders break idempotence of lines
        assert np.array_equal(MO.morphological_filter(opening, "Opening", shape, 2), opening)
    assert np.array_equal(MO.morphological_filter(a, "Gradient", shape, 2), dil - ero)
    assert np.array_equal(MO.morphological_filter(a, "White Top Hat", shape, 2),
                          np.clip(a.astype(int) - opening, 0, 255))
    lap = MO.morphological_filter(a, "Laplacian", shape, 2).astype(int)
    assert np.array_equal(lap, np.clip(dil + ero - 2 * a.astype(int) + 128, 0, 255))
    f32 = MO.morphological_filter(a.astype(np.float32), "Internal Gradient", shape, 2)
    assert f32.dtype == np.float32 and np.array_equal(f32, a - ero)


def _geodesic(marker, mask, conn, erosion=False):
    size = ndi.generate_binary_structure(2, 2 if conn == 8 else 1)
    m = marker.astype(float)
    m = np.maximum(m, mask) if erosion else np.minimum(m, mask)
    while True:
        if erosion:
            n = np.maximum(ndi.grey_erosion(m, footprint=size, mode="nearest"), mask)
        else:
            n = np.minimum(ndi.grey_dilation(m, footprint=size, mode="nearest"), mask)
        if np.array_equal(n, m):
            return m
        m = n


@pytest.mark.parametrize("conn", [4, 8])
def test_reconstruction_equals_iterated_geodesic_dilation(conn):
    b = blobs().plane().astype(float)
    rng = np.random.default_rng(5)
    marker = np.where(rng.random(b.shape) < 0.002, b, 0)
    assert np.array_equal(MO.reconstruct_by_dilation(marker, b, conn),
                          _geodesic(marker, b, conn))
    marker_e = np.where(rng.random(b.shape) < 0.002, b, 255)
    assert np.array_equal(MO.reconstruct_by_erosion(marker_e, b, conn),
                          _geodesic(marker_e, b, conn, erosion=True))
    out = MO.reconstruction(marker.astype(np.uint8), b.astype(np.uint8), "By Dilation", conn)
    assert out.dtype == np.uint8


def _regional_max_brute(a, conn):
    st = ndi.generate_binary_structure(2, 2 if conn == 8 else 1)
    nb = st.copy()
    nb[1, 1] = False
    nbmax = ndi.grey_dilation(a.astype(float), footprint=nb, mode="nearest")
    out = np.zeros(a.shape, bool)
    for v in np.unique(a):
        lab, n = ndi.label(a == v, structure=st)
        higher = ndi.maximum(nbmax > v, lab, np.arange(1, n + 1))
        for i, h in enumerate(np.atleast_1d(higher), 1):
            if not h:
                out |= lab == i
    return out


@pytest.mark.parametrize("conn", [4, 8])
def test_regional_and_extended_extrema(conn):
    a = (_rng_image(18, 21) // 16).astype(np.uint8)
    assert np.array_equal(MO.regional_maxima(a, conn), _regional_max_brute(a, conn))
    assert np.array_equal(MO.regional_minima(a, conn), _regional_max_brute(255 - a, conn))
    yy, xx = np.mgrid[0:40, 0:80]
    two = (100 * np.exp(-((xx - 20) ** 2 + (yy - 20) ** 2) / 50) +
           60 * np.exp(-((xx - 60) ** 2 + (yy - 20) ** 2) / 50)).astype(np.float32)
    count = lambda m: ndi.label(m, structure=np.ones((3, 3)))[1]  # noqa: E731
    assert count(MO.extended_maxima(two, 10, conn)) == 2
    assert count(MO.extended_maxima(two, 70, conn)) == 1
    assert count(MO.extended_minima(-two, 10, conn)) == 2


def test_fill_holes_kill_borders_and_labels_on_particles():
    p = _particles()
    assert np.array_equal(MO.fill_holes(p) > 0, ndi.binary_fill_holes(p > 0))
    for conn in (4, 8):
        st = ndi.generate_binary_structure(2, 2 if conn == 8 else 1)
        lab, n = MO.label_components(p, conn)
        ref, nref = ndi.label(p > 0, structure=st)
        assert n == nref and np.array_equal(lab, ref)
    lab, n = ndi.label(p > 0)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    kept = MO.kill_borders(p) > 0
    assert ndi.label(kept)[1] == n - len(border)
    assert np.array_equal(kept, np.isin(lab, sorted(set(range(1, n + 1)) - border)))
    with pytest.raises(ValueError):
        MO.labels_as(np.arange(300).reshape(15, 20), "8 bits")


def test_chamfer_distances_against_closed_forms():
    p = _particles()
    fg = p > 0
    assert np.array_equal(MO.chamfer_distance(p, "Chessboard (1,1)", False),
                          ndi.distance_transform_cdt(fg, metric="chessboard"))
    assert np.array_equal(MO.chamfer_distance(p, "City-Block (1,2)", False),
                          ndi.distance_transform_cdt(fg, metric="taxicab"))
    np.testing.assert_allclose(MO.chamfer_distance(p, "Euclidean (exact)"),
                               ndi.distance_transform_edt(fg))
    small = np.ones((13, 15), np.uint8)
    small[3, 4] = small[9, 12] = small[6, 7] = 0
    bg = np.argwhere(small == 0)
    yy, xx = np.mgrid[0:13, 0:15]
    dy = np.abs(yy[..., None] - bg[:, 0])
    dx = np.abs(xx[..., None] - bg[:, 1])
    lo, hi = np.minimum(dx, dy), np.maximum(dx, dy)
    ref34 = (4 * lo + 3 * (hi - lo)).min(axis=-1)
    assert np.array_equal(MO.chamfer_distance(small, "Borgefors (3,4)", False), ref34)
    np.testing.assert_allclose(MO.chamfer_distance(small, "Borgefors (3,4)", True), ref34 / 3)
    d5711 = MO.chamfer_distance(small, "Chessknight (5,7,11)", False)
    assert d5711[3, 6] == 10 and d5711[4, 6] == 11 and d5711[5, 5] == 11
    assert d5711[0, 0] == 11 + 7 + 7               # knight (1,2) + two diagonals
    assert MO.convert_distance(ref34 / 3.0, "16 bits").dtype == np.uint16


def test_watersheds_split_touching_discs():
    yy, xx = np.mgrid[0:60, 0:90]
    a = ((((xx - 30) ** 2 + (yy - 30) ** 2) <= 225) |
         (((xx - 56) ** 2 + (yy - 30) ** 2) <= 196)).astype(np.uint8) * 255
    for conn in (4, 8):
        ws = MO.distance_transform_watershed(a, dynamic=1, connectivity=conn)
        assert set(np.unique(ws).tolist()) == {0, 1, 2}
        dam = (a > 0) & (ws == 0)
        assert 0 < dam.sum() < 40 and np.all(np.abs(np.argwhere(dam)[:, 1] - 43) <= 3)
        assert np.all(ws[a == 0] == 0)
    nodam = MO.distance_transform_watershed(a, dynamic=1, dams=False)
    assert np.all(nodam[a > 0] > 0)
    # blobs: every blob deeper than the dynamic gets at least one basin (normalised float
    # distances must keep their plateaus: round-off is snapped before the extrema search);
    # blobs whose maximum distance is <= dynamic vanish in the h-minima transform
    m = blobs().plane() > 125
    ws = MO.distance_transform_watershed(m, dynamic=1)
    lab, n = ndi.label(m)
    depth = ndi.maximum(MO.chamfer_distance(m), lab, np.arange(1, n + 1))
    deep = {i for i, d in enumerate(depth, 1) if d > 1}
    assert n == 64 and len(deep) <= ws.max() <= len(deep) + 6
    assert set(np.unique(lab[ws > 0]).tolist()) == deep
    # marker-controlled on a grey ramp: two markers, the boundary at the ridge
    f = np.abs(np.arange(40) - 20)[None, :].repeat(10, 0).astype(float)
    f = f.max() - f                                 # ridge at x = 20
    mk = np.zeros(f.shape, int)
    mk[5, 2], mk[5, 37] = 1, 2
    lab = MO.watershed(f, mk, connectivity=4, dams=True)
    assert np.all(lab[:, :20] == 1) and np.all(lab[:, 21:] == 2) and np.all(lab[:, 20] == 0)


def test_morpholibj_commands_through_the_registry():
    p = _particles()
    doc = ImageDoc.from_array(p.copy(), "YX", "particles8.tif")
    s = _session(doc)
    lbl = s.run("Connected Components Labeling", "connectivity=8 type=[16 bits]")
    assert lbl.title == "particles8-lbl" and lbl.type == GRAY16
    assert int(lbl.plane().max()) == ndi.label(p > 0, structure=np.ones((3, 3)))[1]
    s.set_active(doc)
    dist = s.run("Chamfer Distance Map", "distances=[Borgefors (3,4)] output=[32 bits] normalize")
    assert dist.type == GRAY32 and dist.title == "particles8-dist"
    s.set_active(doc)
    dtw = s.run("Distance Transform Watershed", "distances=[Borgefors (3,4)] output=[32 bits] "
                                                "normalize dynamic=1 connectivity=4")
    assert dtw.title == "particles8-dist-watershed" and dtw.plane().max() >= 11
    s.set_active(doc)
    ero = s.run("Morphological Filters", "operation=Erosion element=Disk radius=2")
    assert ero.title == "particles8-Erosion"
    assert np.array_equal(ero.plane(), MO.morphological_filter(p, "Erosion", "Disk", 2))
    s.set_active(doc)
    rmax = s.run("Regional Min & Max", "operation=[Regional Maxima] connectivity=4")
    assert set(np.unique(rmax.plane()).tolist()) <= {0, 255} and rmax.is_inverted_lut()
    s.set_active(doc)
    emax = s.run("Extended Min & Max", "operation=[Extended Maxima] dynamic=10 connectivity=4")
    assert emax.title == "particles8-emax"
    s.set_active(doc)
    assert s.run("Fill Holes (Binary/Gray)").title == "particles8-fillHoles"
    s.set_active(doc)
    assert s.run("Kill Borders").title == "particles8-killBorders"
    mk = np.zeros_like(p)
    mk[20, 20] = mk[70, 60] = 255
    marker = s.add_image(ImageDoc.from_array(mk, "YX", "markers"))
    rec = s.run("Morphological Reconstruction",
                "marker=markers mask=particles8.tif type=[By Dilation] connectivity=4")
    assert rec.title == "particles8-rec"
    assert ndi.label(rec.plane() > 0)[1] == 2
    s.set_active(marker)
    inv = s.add_image(ImageDoc.from_array((255 - p).astype(np.uint8), "YX", "inv"))
    ws = s.run("Marker-controlled Watershed",
               "input=inv marker=markers mask=particles8.tif binary calculate")
    assert ws.title == "inv-watershed" and int(ws.plane().max()) == 2
    assert doc.plane().tolist() == p.tolist()      # the source image is never modified
    assert inv is not None and len(s.images) >= 10


def test_fiji_selftest():
    process_fiji.selftest()
