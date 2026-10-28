"""WP-B Process > Smooth/Sharpen/Find Edges, Noise, Shadows, Filters against ImageJ 1.54p.

Goldens: ``tests/data_imagej/golden/process_filters`` (macro ``cases/process_filters.ijm``)
and the phase-0 ``golden/filters``. 8/16-bit outputs must be identical, 32-bit within float32
precision (rel/abs 1e-5).
"""
from __future__ import annotations

import numpy as np
import pytest

from ij_golden import golden_images, input_path
from rsstatchem.core.imaging import filters as F
from rsstatchem.core.imaging.roi import OvalRoi, RectRoi

G = golden_images("process_filters")
G0 = golden_images("filters")

PREFIXES = ("input_nan32", "input_rgbstk", "input_stk8", "input_stk16", "input_stk32",
            "input_rgb", "synth8", "synth16", "synth32", "cell", "blobs")
KERNELS = {
    "k1": "1 2 3 2 1\n2 4 6 4 2\n3 6 9 6 3\n2 4 6 4 2\n1 2 3 2 1\n",
    "k2": "-1 -1 -1 -1 -1\n-1 -1 -1 -1 -1\n-1 -1 24 -1 -1\n-1 -1 -1 -1 -1\n-1 -1 -1 -1 -1\n",
    "k3": "0 1 0 -1 0\n1 2.5 0 -2 -1\n0 1 0 -1 0\n",
}
RECT = (10, 12, 30, 25)
OVAL = (8, 10, 40, 30)


def source(prefix: str) -> np.ndarray:
    if prefix.startswith("synth"):
        return G0["input_" + prefix]
    if prefix in ("cell", "blobs"):
        return G["input_" + prefix]
    return G[prefix]


def split(name: str) -> tuple[str, str]:
    for p in PREFIXES:
        if name.startswith(p + "_"):
            return p, name[len(p) + 1:]
    raise KeyError(name)


def compute_plane(src: np.ndarray, op: str, inverted: bool = False, roi=None):
    parts = op.split("_")
    kind = parts[0]
    if kind == "smooth":
        return F.smooth(src)
    if kind == "sharpen":
        return F.sharpen(src)
    if kind == "edges":
        return F.find_edges(src)
    if kind == "shadow":
        return F.shadow(src, parts[1])
    if kind == "despeckle":
        return F.rank_filter(src, 1, "despeckle")
    if kind == "gauss":
        return F.gaussian_blur(src, float(parts[1]), roi=roi)
    if kind == "unsharp":
        return F.unsharp_mask(src, float(parts[1]), float(parts[2]), roi=roi)
    if kind in ("median", "mean", "min", "max", "variance"):
        return F.rank_filter(src, float(parts[1]), kind)
    if kind == "tophat":
        return F.rank_filter(src, float(parts[1]), "top_hat", light_background="light" in parts,
                             dont_subtract="dont" in parts, inverted_lut=inverted)
    if kind == "outliers":
        return F.rank_filter(src, float(parts[1]), "outliers", threshold=float(parts[2]),
                             which_outliers=parts[3].capitalize(), inverted_lut=inverted)
    if kind == "removenan":
        return F.rank_filter(src, float(parts[1]), "remove_nan")
    if kind == "convolve":
        k, _, _ = F.parse_kernel(KERNELS[parts[1]])
        return F.convolve(src, k, normalize="norm" in parts)
    raise KeyError(op)


def compute(name: str):
    prefix, op = split(name)
    src = source(prefix)
    parts = op.split("_")
    if parts[0].endswith("3d"):
        x, y, z = (float(v) for v in parts[1:4])
        if parts[0] == "gauss3d":
            return F.gaussian_blur_3d(src, x, y, z)
        return F.filter_3d(src, parts[0][:-2], x, y, z)
    if op.endswith("_stack"):
        return np.stack([compute_plane(p, op[:-len("_stack")]) for p in src])
    if "seed" in op:
        sd = 25.0 if parts[0] == "addnoise" else float(parts[0][5:])
        seed = int(parts[1][4:])
        rect = RECT if op.endswith("_rect") else None
        return F.add_noise(src, sd, F.JavaRandom(seed), rect=rect)
    roi = None
    if parts[-1] in ("rect", "oval"):
        shape = parts[-1]
        op = "_".join(parts[:-1])
        r = RectRoi(*RECT) if shape == "rect" else OvalRoi(*OVAL)
        mask = r.get_mask(src.shape[:2])
        x, y, w, h = (int(v) for v in r.bounds())
        out = compute_plane(src, op, roi=(x, y, w, h))
        res = src.copy()
        res[mask] = out[mask]
        return res
    return compute_plane(src, op, inverted=(prefix == "blobs"))


def assert_same(out: np.ndarray, exp: np.ndarray, name: str) -> None:
    assert out.shape == exp.shape and out.dtype == exp.dtype, (name, out.shape, exp.shape,
                                                               out.dtype, exp.dtype)
    if exp.dtype == np.float32:
        np.testing.assert_allclose(out, exp, rtol=1e-5, atol=1e-5, equal_nan=True,
                                   err_msg=name)
    else:
        diff = out.astype(np.int64) - exp.astype(np.int64)
        assert not diff.any(), f"{name}: {np.count_nonzero(diff)} pixels differ " \
                               f"(max {np.abs(diff).max()})"


INPUT_NAMES = {"input_nan32", "input_rgbstk", "input_stk8", "input_stk16", "input_stk32",
               "input_rgb", "input_cell", "input_blobs"}
GOLDEN_NAMES = sorted(k for k in G if k not in INPUT_NAMES and k != "circular_masks")


@pytest.mark.parametrize("name", GOLDEN_NAMES)
def test_process_filters_golden(name):
    assert_same(compute(name), G[name], name)


@pytest.mark.parametrize("name", sorted(k for k in G0 if not k.startswith("input_")))
def test_phase0_filter_goldens(name):
    img, op, par = name.split("_", 2)
    src = G0["input_" + img]
    out = F.gaussian_blur(src, float(par)) if op == "gauss" else F.rank_filter(src, float(par),
                                                                             op)
    assert_same(out, G0[name], name)


def test_circular_masks():
    stack, labels = F.circular_masks_image()
    assert_same(stack, G["circular_masks"], "circular_masks")
    assert labels[0] == "radius=0.5, size=3" and labels[1] == "radius=1.0, size=3"
    assert len(labels) == 99


def test_line_radii_and_footprints():
    assert F.make_line_radii(0.5)[-2:] == [5, 1]
    assert F.make_line_radii(1)[-2:] == [9, 1]
    assert F.circular_footprint(2).sum() == F.make_line_radii(2)[-2]
    assert F.line_radii(1.0) == [(-1, 1), (0, 1), (1, 1)]


def test_java_random_matches_java():
    r = F.JavaRandom(42)
    # java.util.Random(42): nextInt() = -1170105035, nextInt(10) = 0 (after), nextDouble ...
    assert r.next(32) == -1170105035
    r = F.JavaRandom(42)
    assert r.next_int(10) == 0
    assert abs(F.JavaRandom(42).next_gaussian() - 1.1419053154730547) < 1e-15


def test_noise_statistics_and_salt_pepper():
    src = np.full((80, 80), 100, np.uint8)
    out = F.add_noise(src, 25.0)
    d = out.astype(float) - 100
    assert abs(d.mean()) < 2 and 22 < d.std() < 28
    f = F.add_noise(np.zeros((50, 50), np.float32), 10.0)
    assert f.dtype == np.float32 and 8 < f.std() < 12
    sp = F.salt_and_pepper(src, 0.05, F.JavaRandom(1))
    n = 80 * 80 * 0.05
    assert abs((sp == 255).sum() - n / 2) <= 5 and abs((sp == 0).sum() - n / 2) <= 5
    assert set(np.unique(sp)) <= {0, 100, 255}


def test_parse_kernel_square_and_errors():
    k, kw, kh = F.parse_kernel("1 1 1 1 1 1 1 1 1")
    assert k.shape == (3, 3) and kw == kh == 3
    with pytest.raises(ValueError):
        F.parse_kernel("1 2\n3 4\n")
    with pytest.raises(ValueError):
        F.parse_kernel("1 2 3\n4 5\n6 7 8\n")


def test_commands_registered_and_run():
    from rsstatchem.core.imaging.image import ImageDoc
    from rsstatchem.core.imaging.session import Session
    from rsstatchem.ui.imagej.dialogs import process_filters as P

    s = Session()
    P.register(s.registry)
    for label in P.LABELS:
        assert label in s.registry, label
    doc = ImageDoc.from_array(G0["input_synth8"].copy(), "YX", "s8")
    s.add_image(doc)
    s.run("Gaussian Blur...", "sigma=2")
    assert_same(doc.plane(), G0["synth8_gauss_2"], "cmd gauss")
    doc.undo()
    s.run("Median...", "radius=2")
    assert_same(doc.plane(), G0["synth8_median_2"], "cmd median")
    doc.undo()
    doc.roi = RectRoi(*RECT)
    s.run("Mean...", "radius=3")
    assert_same(doc.plane(), G["synth8_mean_3_rect"], "cmd mean rect")
    doc.undo()
    doc.roi = None
    s.run("Convolve...", "text1=[" + KERNELS["k1"] + "] normalize")
    assert_same(doc.plane(), G["synth8_convolve_k1_norm"], "cmd convolve")
    doc.undo()
    s.run("Remove Outliers...", "radius=2 threshold=10 which=Bright")
    assert_same(doc.plane(), G["synth8_outliers_2_10_bright"], "cmd outliers")
    doc.undo()
    s.run("Top Hat...", "radius=5 light don't")
    assert_same(doc.plane(), G["synth8_tophat_5_light_dont"], "cmd tophat")
    doc.undo()
    s.run("Unsharp Mask...", "radius=3 mask=0.9")
    assert_same(doc.plane(), G["synth8_unsharp_3_0.9"], "cmd unsharp")
    doc.undo()
    for d in ("North", "Southwest"):
        s.run(d)
        assert_same(doc.plane(), G[f"synth8_shadow_{d.lower()}"], d)
        doc.undo()
    s.registry.run(s, "Salt and Pepper", "")
    assert ((doc.plane() == 0) | (doc.plane() == 255)).sum() > 100
    doc.undo()
    masks = s.run("Show Circular Masks...")
    assert masks.n_slices == 99 and masks.type == 2
    demo = s.run("Shadows Demo", image=doc)
    assert demo.stack_size == 8
    stk = ImageDoc.from_array(G["input_stk8"].copy(), "ZYX", "stk")
    s.add_image(stk)
    s.run("Median 3D...", "x=2 y=2 z=2")
    assert_same(stk.pixels[0, :, 0], G["input_stk8_median3d_2_2_2"], "cmd median3d")
    stk.undo()
    s.run("Variance 3D...", "x=2 y=2 z=2")
    assert stk.type == 2
    assert_same(stk.pixels[0, :, 0], G["input_stk8_variance3d_2_2_2"], "cmd var3d")
    stk.undo()
    s.run("Gaussian Blur 3D...", "x=2 y=2 z=2")
    assert_same(stk.pixels[0, :, 0], G["input_stk8_gauss3d_2_2_2"], "cmd gauss3d")
    stk.undo()
    s.run("Gaussian Blur...", "sigma=2 stack")
    assert_same(stk.pixels[0, :, 0], G["input_stk8_gauss_2_stack"], "cmd gauss stack")
    rgb = ImageDoc.from_array(G["input_rgb"].copy(), "YXS", "rgb")
    s.add_image(rgb)
    s.run("Sharpen")
    assert_same(rgb.plane(), G["input_rgb_sharpen"], "cmd rgb sharpen")
    nan = ImageDoc.from_array(G["input_nan32"].copy(), "YX", "nan")
    s.add_image(nan)
    s.run("Remove NaNs...", "radius=2")
    assert_same(nan.plane(), G["input_nan32_removenan_2"], "cmd remove nans")


def test_type_checks():
    from rsstatchem.core.imaging.commands import CommandError
    from rsstatchem.core.imaging.image import ImageDoc
    from rsstatchem.core.imaging.session import Session
    from rsstatchem.ui.imagej.dialogs import process_filters as P

    s = Session()
    P.register(s.registry)
    s.add_image(ImageDoc.from_array(G0["input_synth8"].copy(), "YX", "s8"))
    with pytest.raises(CommandError):
        s.run("Remove NaNs...", "radius=2")
    s.add_image(ImageDoc.from_array(G0["input_synth16"].copy(), "YX", "s16"))
    with pytest.raises(CommandError):
        s.run("Salt and Pepper")


def test_selftest():
    from rsstatchem.ui.imagej.dialogs import process_filters as P
    P.selftest()


def test_blobs_input_matches_reader():
    from rsstatchem.core.imaging.io import open_image
    doc = open_image(input_path("blobs.gif"))
    assert np.array_equal(doc.plane(), G["input_blobs"]) and doc.is_inverted_lut()
