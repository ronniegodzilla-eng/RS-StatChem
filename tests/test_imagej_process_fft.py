"""Process > FFT family and Subtract Background against the real ImageJ 1.54p
(goldens ``tests/data_imagej/golden/process_fft`` and ``process_background``) plus the
command layer (option strings, FFT Options outputs, inverse round trips)."""
from __future__ import annotations

import numpy as np
import pytest

from ij_golden import golden_images, input_path
from rsstatchem.core.imaging import background as B
from rsstatchem.core.imaging import fft as F
from rsstatchem.core.imaging.image import GRAY8, GRAY32, ImageDoc
from rsstatchem.core.imaging.io import open_image
from rsstatchem.core.imaging.session import Session

IMAGES = {"synth8": "synth8.tif", "synth16": "synth16.tif", "synth32": "synth32.tif",
          "blobs": "blobs.gif"}


@pytest.fixture(scope="module")
def gfft():
    return golden_images("process_fft")


@pytest.fixture(scope="module")
def gbg():
    return golden_images("process_background")


def _doc(name):
    return open_image(input_path(IMAGES.get(name, name)))


def assert_same(ours, ref):
    ours, ref = np.asarray(ours), np.asarray(ref)
    assert ours.shape == ref.shape
    if ref.dtype == np.float32:
        np.testing.assert_allclose(ours.astype(np.float64), ref.astype(np.float64),
                                   rtol=1e-5, atol=1e-6 * max(1.0, float(np.abs(ref).max())))
    else:
        assert ours.dtype == ref.dtype
        assert np.array_equal(ours, ref), f"{np.count_nonzero(ours != ref)} pixels differ"


# ------------------------------------------------------------------------------------- FFT
@pytest.mark.parametrize("name", list(IMAGES))
def test_fft_and_inverse_match_imagej(gfft, name):
    doc = _doc(name)
    ps, info = F.forward_fft(doc.plane(), doc.bit_depth, doc.lut)
    assert_same(ps, gfft[f"{name}_fft"])
    inv, _ = F.inverse_fft(info)
    assert_same(inv, gfft[f"{name}_ifft"])
    for colour in ("black", "white"):
        inv, _ = F.inverse_fft(info, gfft[f"{name}_fft_{colour}"], 8)
        assert_same(inv, gfft[f"{name}_ifft_{colour}"])


BANDPASS = {"default": (40, 3, "None", 5, True, True),
            "horizontal": (20, 2, "Horizontal", 5, True, True),
            "vertical": (30, 1, "Vertical", 20, True, False),
            "noscale": (40, 3, "None", 5, False, False)}


@pytest.mark.parametrize("name", list(IMAGES))
@pytest.mark.parametrize("variant", list(BANDPASS))
def test_bandpass_matches_imagej(gfft, name, variant):
    doc = _doc(name)
    out, _ = F.bandpass_filter(doc.plane(), doc.bit_depth, *BANDPASS[variant])
    assert_same(out, gfft[f"{name}_bandpass_{variant}"])


def test_bandpass_rect_rgb_and_swap(gfft):
    blobs = _doc("blobs")
    out, _ = F.bandpass_filter(blobs.plane(), 8, rect=(30, 40, 100, 70))
    assert_same(out, gfft["blobs_bandpass_rect"])
    rgb = gfft["rgb_input"]
    out, _ = F.bandpass_filter(rgb, 24)
    assert_same(out, gfft["rgb_bandpass"])
    ps, info = F.forward_fft(rgb, 24)
    assert_same(ps, gfft["rgb_fft"])
    assert_same(F.inverse_fft(info)[0], gfft["rgb_ifft"])
    assert_same(F.swap_quadrants(_doc("synth8").plane()), gfft["synth8_swap"])
    assert_same(F.swap_quadrants(_doc("synth32").plane()), gfft["synth32_swap"])


def test_fd_math_matches_imagej(gfft):
    a, b = _doc("synth8").plane(), _doc("synth32").plane()
    for op in F.FD_OPERATIONS:
        assert_same(F.fd_math(a, b, op, True), gfft[f"fdmath_{op}"])
    assert_same(F.fd_math(a, a, "Correlate", True), gfft["fdmath_autocorrelate"])
    ps, info = F.fd_math(a, b, "Convolve", False)
    assert_same(ps, gfft["fdmath_convolve_ps"])
    assert_same(F.inverse_fft(info, ps, 32)[0], gfft["fdmath_convolve_ps_inverse"])


@pytest.mark.parametrize("name", ["synth8", "synth16", "synth32"])
def test_custom_filter_matches_imagej(gfft, name):
    doc = _doc(name)
    for size in ("128", "64"):
        out = F.custom_filter(doc.plane(), doc.bit_depth, gfft[f"custom_filter{size}"])
        assert_same(out, gfft[f"{name}_custom{size}"])


def test_fht_is_invertible_and_hsb_roundtrip():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(32, 32)).astype(np.float32)
    back = F.fht_transform(F.fht_transform(a), inverse=True)
    np.testing.assert_allclose(back, a, atol=1e-5)
    rgb = rng.integers(0, 256, (16, 16, 3)).astype(np.uint8)
    h, s, v = F.rgb_to_hsb(rgb)
    assert np.array_equal(F.hsb_to_rgb(h, s, v), rgb)


# ------------------------------------------------------------------------------------- background
BG_OPTS = {"r50": dict(radius=50), "r20_light": dict(radius=20, light_background=True),
           "r8": dict(radius=8), "r50_create": dict(radius=50, create_background=True),
           "r20_disable": dict(radius=20, presmooth=False),
           "p50": dict(radius=50, use_paraboloid=True),
           "p10_light": dict(radius=10, light_background=True, use_paraboloid=True),
           "p30_disable": dict(radius=30, use_paraboloid=True, presmooth=False),
           "p15_create": dict(radius=15, create_background=True, use_paraboloid=True),
           "r120": dict(radius=120)}
BG_IMAGES = {"blobs": "blobs.gif", "colony": "Cell_Colony.jpg", "synth8": "synth8.tif",
             "synth16": "synth16.tif", "synth32": "synth32.tif"}


@pytest.mark.parametrize("name", list(BG_IMAGES))
@pytest.mark.parametrize("variant", list(BG_OPTS))
def test_subtract_background_matches_imagej(gbg, name, variant):
    if name == "colony" and variant not in ("r50", "r20_light", "p50", "r120"):
        pytest.skip("the large Cell_Colony image has goldens for 4 variants only")
    doc = open_image(input_path(BG_IMAGES[name]))
    out = B.subtract_background(doc.plane(), inverted_lut=doc.is_inverted_lut(),
                                **BG_OPTS[variant])
    assert_same(out, gbg[f"{name}_bg_{variant}"])


@pytest.mark.parametrize("variant,kw", [
    ("r20", dict(radius=20)), ("r20_sep", dict(radius=20, separate_colors=True)),
    ("p10_light", dict(radius=10, light_background=True, use_paraboloid=True)),
    ("p10_sep_light", dict(radius=10, separate_colors=True, light_background=True,
                           use_paraboloid=True)),
    ("r20_sep_create", dict(radius=20, separate_colors=True, create_background=True))])
def test_subtract_background_rgb_matches_imagej(gbg, variant, kw):
    assert_same(B.subtract_background(gbg["rgb_input"], **kw), gbg[f"rgb_bg_{variant}"])


# ------------------------------------------------------------------------------------- commands
@pytest.fixture
def session():
    from rsstatchem.ui.imagej.dialogs import process_background, process_fft

    s = Session()
    process_fft.register(s.registry)
    process_background.register(s.registry)
    return s


def test_fft_commands(session, gfft):
    blobs = _doc("blobs")
    session.add_image(blobs)
    spec = session.run("FFT")[0]
    assert spec.title == "FFT of blobs.gif" and spec.type == GRAY8
    assert np.array_equal(spec.plane(), gfft["blobs_fft"])
    spec.set_plane(gfft["blobs_fft_white"])
    inv = session.run("Inverse FFT", image=spec)
    assert inv.title == "Inverse FFT of blobs.gif" and inv.is_inverted_lut()
    assert np.array_equal(inv.plane(), gfft["blobs_ifft_white"])
    spec.set_plane(np.zeros_like(spec.plane()) + 7)
    session.run("Redisplay Power Spectrum", image=spec)
    assert np.array_equal(spec.plane(), gfft["blobs_fft"])
    again = session.run("FFT", image=spec)          # FFT of an FFT window = inverse
    assert again.title.startswith("Inverse FFT of")
    session.set_active(blobs)
    session.run("Bandpass Filter...", "filter_large=40 filter_small=3 suppress=None "
                                      "tolerance=5 autoscale saturate display")
    assert np.array_equal(blobs.plane(), gfft["blobs_bandpass_default"])
    assert session.active.title == "Filter" and session.active.type == GRAY32
    assert blobs.can_undo()
    with pytest.raises(Exception):
        session.run("Inverse FFT", image=blobs)


def test_fft_options_outputs_roundtrip(session):
    img = ImageDoc.from_array(np.random.default_rng(3).normal(size=(40, 50))
                              .astype(np.float32), "YX", "f")
    session.add_image(img)
    out = session.run("FFT Options...", "fft raw fast complex do")
    titles = [d.title for d in out]
    assert titles == ["PS of f", "FHT of f", "Complex of f", "FFT of f"]
    fht_win, cplx = out[1], out[2]
    back = session.run("Inverse FFT", image=fht_win)
    assert back.width == 50 and back.height == 40
    np.testing.assert_allclose(back.plane(), img.plane(), atol=1e-4)
    back2 = session.run("Inverse FFT", image=cplx)
    assert back2.n_slices == 2 and back2.slice_labels == ["Real", "Imaginary"]
    np.testing.assert_allclose(back2.plane_at(0), img.plane(), atol=1e-4)
    assert np.abs(back2.plane_at(1)).max() < 1e-3
    session.run("FFT Options...", "fft")


def test_fd_math_custom_swap_circle_commands(session, gfft):
    a, b = _doc("synth8"), _doc("synth32")
    a.title, b.title = "a", "b"
    session.add_image(a)
    session.add_image(b)
    res = session.run("FD Math...", "image1=a operation=Deconvolve image2=b result=R do")
    assert res.title == "R"
    assert_same(res.plane(), gfft["fdmath_Deconvolve"])
    filt = ImageDoc.from_array(gfft["custom_filter128"], "YX", "filt128")
    session.add_image(filt)
    session.run("Custom Filter...", "filter=filt128", image=a)
    assert np.array_equal(a.plane(), gfft["synth8_custom128"])
    session.run("Swap Quadrants", image=b)
    assert_same(b.plane(), gfft["synth32_swap"])
    session.run("Make Circular Selection...", "radius=10", image=b)
    assert (b.roi.x, b.roi.y, b.roi.width, b.roi.height) == (22, 22, 20.0, 20.0)


def test_subtract_background_command(session, gbg):
    blobs = _doc("blobs")
    session.add_image(blobs)
    session.run("Subtract Background...", "rolling=20 light")
    assert np.array_equal(blobs.plane(), gbg["blobs_bg_r20_light"])
    blobs.undo()
    session.run("Subtract Background...", "rolling=50 sliding")
    assert np.array_equal(blobs.plane(), gbg["blobs_bg_p50"])
    stack = ImageDoc.from_array(np.stack([_doc("synth8").plane()] * 2), "ZYX", "st")
    session.add_image(stack)
    session.run("Subtract Background...", "rolling=50 stack")
    for z in range(2):
        assert np.array_equal(stack.plane(0, z, 0), gbg["synth8_bg_r50"])


def test_selftests():
    from rsstatchem.ui.imagej.dialogs import process_background, process_fft
    process_fft.selftest()
    process_background.selftest()
