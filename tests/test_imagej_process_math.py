"""Process > Math, Image Calculator and Enhance Contrast against ImageJ 1.54p goldens
(``tests/data_imagej/cases/process_math.ijm``), plus Batch commands and the macro evaluator."""
from __future__ import annotations

import csv
import math

import numpy as np
import pytest

from ij_golden import golden_images, golden_path, input_path
from rsstatchem.core.imaging import contrast as CE
from rsstatchem.core.imaging import math_ops as M
from rsstatchem.core.imaging.image import ImageDoc
from rsstatchem.core.imaging.roi import OvalRoi, RectRoi
from rsstatchem.core.imaging.session import Session


@pytest.fixture(scope="module")
def gold():
    return golden_images("process_math")


@pytest.fixture(scope="module")
def inputs(gold):
    f = golden_images("filters")
    return {"synth8": f["input_synth8"], "synth16": f["input_synth16"],
            "synth32": f["input_synth32"], "rgb": gold["input_rgb"],
            "rgb2": gold["input_rgb2"], "b8": gold["input_b8"],
            "stack": gold["input_stack"]}


def make_session():
    from rsstatchem.ui.imagej.commands_process import register
    s = Session()
    register(s.registry)
    return s


def doc(arr, title):
    a = np.asarray(arr)
    if a.ndim == 3 and a.shape[-1] == 3 and a.dtype == np.uint8:
        return ImageDoc.from_array(a.copy(), "YXS", title)
    if a.ndim == 3:
        return ImageDoc.from_array(a.copy(), "ZYX", title)
    return ImageDoc.from_array(a.copy(), "YX", title)


def pixels(img):
    """Squeezed pixel array as the goldens store it (stacks: ZYX)."""
    p = img.pixels
    if img.is_rgb:
        return p.reshape((-1,) + p.shape[-3:]).squeeze(0) if img.stack_size == 1 else \
            p.reshape((-1,) + p.shape[-3:])
    p = p.reshape((-1,) + p.shape[-2:])
    return p[0] if p.shape[0] == 1 else p


def assert_same(got, want):
    got, want = np.asarray(got), np.asarray(want)
    assert got.shape == want.shape, (got.shape, want.shape)
    if want.dtype == np.float32:
        assert got.dtype == np.float32
        np.testing.assert_allclose(got, want, rtol=2e-7, atol=0, equal_nan=True)
    else:
        assert got.dtype == want.dtype
        diff = np.argwhere(got != want)
        assert diff.size == 0, f"{len(diff)} pixels differ, first {diff[:3].tolist()}"


MATH = [("Add...", "value=25", "add"), ("Subtract...", "value=25", "sub"),
        ("Multiply...", "value=1.5", "mul"), ("Divide...", "value=3", "div"),
        ("Divide...", "value=0", "div0"), ("AND...", "value=01010101", "and"),
        ("OR...", "value=01010101", "or"), ("XOR...", "value=01010101", "xor"),
        ("Min...", "value=100", "min"), ("Max...", "value=100", "max"),
        ("Gamma...", "value=0.5", "gamma05"), ("Gamma...", "value=2", "gamma2"),
        ("Set...", "value=77", "set"), ("Log", "", "log"), ("Exp", "", "exp"),
        ("Square", "", "sqr"), ("Square Root", "", "sqrt"), ("Reciprocal", "", "recip"),
        ("Abs", "", "abs")]
MACROS = ["v=v*2+x", "v=sin(d/4)*a*20+v", "if (d>20) v=0;", "v=getPixel(x+3,y+2)",
          "v=v-y*3", "a=round(a/0.5); if (a%2==0) v=v/2; else v=v+10;"]
IMAGES = ["synth8", "synth16", "synth32", "rgb"]


@pytest.mark.parametrize("img", IMAGES)
@pytest.mark.parametrize("cmd,opts,name", MATH, ids=[m[2] for m in MATH])
def test_math_ops_match_imagej(gold, inputs, img, cmd, opts, name):
    s = make_session()
    im = s.add_image(doc(inputs[img], img))
    s.run(cmd, opts)
    assert_same(pixels(im), gold[f"{img}_{name}"])


@pytest.mark.parametrize("img", IMAGES)
@pytest.mark.parametrize("k", range(len(MACROS)))
def test_math_macro_matches_imagej(gold, inputs, img, k):
    s = make_session()
    im = s.add_image(doc(inputs[img], img))
    s.run("Macro...", f"code=[{MACROS[k]}]")
    assert_same(pixels(im), gold[f"{img}_macro{k}"])


def test_nan_background_and_16bit_display_range(gold, inputs):
    from rsstatchem.core.imaging.image import Threshold as T
    s = make_session()
    im = s.add_image(doc(inputs["synth32"], "s32"))
    with pytest.raises(Exception):
        s.run("NaN Background")
    im.threshold = T(-1.0, 2.0)
    s.run("NaN Background")
    assert_same(pixels(im), gold["synth32_nan"])
    for k, cmd in enumerate(("Gamma...", "Log", "Exp")):
        im = s.add_image(doc(inputs["synth16"], f"s16_{k}"))
        im.set_display_range(2000, 30000)
        s.run(cmd, "value=0.5" if k == 0 else "")
        assert_same(pixels(im), gold[f"synth16_range{k}"])


def test_math_with_selections_and_stacks(gold, inputs):
    s = make_session()
    im = s.add_image(doc(inputs["synth8"], "a"))
    im.roi = RectRoi(10, 12, 30, 20)
    s.run("Add...", "value=60")
    assert_same(pixels(im), gold["synth8_roi_add"])
    im = s.add_image(doc(inputs["synth8"], "g"))
    im.roi = RectRoi(10, 12, 30, 20)
    s.run("Macro...", "code=[v=getPixel(x+1,y)]")
    assert_same(pixels(im), gold["synth8_roi_getpixel"])
    for img, cmd, opts, name in (("synth16", "Multiply...", "value=1.7", "synth16_roi_mul"),
                                 ("synth32", "Macro...", "code=[v=v*10+x]",
                                  "synth32_roi_macro")):
        im = s.add_image(doc(inputs[img], img))
        im.roi = OvalRoi(8, 6, 40, 30)
        s.run(cmd, opts)
        want = gold[name]
        got = pixels(im)
        # Phase 0 oval masks equal ImageJ's for the golden ovals; compare everything
        assert_same(got, want)
    st = s.add_image(doc(inputs["stack"], "stk"))
    s.run("Add...", "value=10 stack")
    s.run("Macro...", "code=[v=v+z*20] stack")
    assert_same(pixels(st), gold["stack_math"])


CALC_OPS = ["Add", "Subtract", "Multiply", "Divide", "AND", "OR", "XOR", "Min", "Max",
            "Average", "Difference", "Copy", "Transparent-zero"]
LOPS = ["add", "sub", "mul", "div", "and", "or", "xor", "min", "max", "ave", "diff", "copy",
        "zero"]


@pytest.fixture(scope="module")
def calc_session(inputs):
    s = make_session()
    s.add_image(doc(inputs["b8"], "b8"))
    s.add_image(doc(inputs["synth8"], "synth8.tif"))
    s.add_image(doc(inputs["synth16"], "synth16.tif"))
    s.add_image(doc(inputs["synth16"][:, ::-1], "b16"))
    s.add_image(doc(inputs["synth32"], "synth32.tif"))
    s.add_image(doc(inputs["synth32"][:, ::-1], "b32"))
    b32z = doc(inputs["synth32"][:, ::-1], "b32z")
    s.add_image(b32z)
    s.run("Min...", "value=0", image=b32z)
    s.add_image(doc(inputs["rgb"], "rgb"))
    s.add_image(doc(inputs["rgb2"], "rgb2"))
    return s


def calc(s, opts):
    out = s.run("Image Calculator...", opts)
    assert isinstance(out, ImageDoc)
    s.close_image(out)
    return pixels(out)


@pytest.mark.parametrize("k", range(len(CALC_OPS)), ids=LOPS)
def test_image_calculator_ops(gold, calc_session, k):
    s, op, lop = calc_session, CALC_OPS[k], LOPS[k]
    for a, b, name in (("synth8.tif", "b8", "calc8_"), ("synth16.tif", "b16", "calc16_"),
                       ("synth32.tif", "b32", "calc32_"), ("rgb", "rgb2", "calcrgb_")):
        got = calc(s, f"image1={a} operation={op} image2={b} create")
        assert_same(got, gold[name + lop])
    got = calc(s, f"image1=synth8.tif operation={op} image2=b8 create 32-bit")
    assert_same(got, gold["calc8f_" + lop])


def test_image_calculator_special_cases(gold, calc_session, inputs):
    s = calc_session
    assert_same(calc(s, "image1=synth32.tif operation=Divide image2=b32z create"),
                gold["calc32_divzero"])
    s.options.divide_by_zero = 0.0
    try:
        assert_same(calc(s, "image1=synth32.tif operation=Divide image2=b32z create"),
                    gold["calc32_divzero_value"])
    finally:
        s.options.divide_by_zero = math.inf
    for a, op, b, name in (("synth8.tif", "Add", "synth16.tif", "calcmix_8_16"),
                           ("synth16.tif", "Subtract", "synth8.tif", "calcmix_16_8"),
                           ("synth8.tif", "Multiply", "synth32.tif", "calcmix_8_32mul"),
                           ("synth8.tif", "Add", "synth32.tif", "calcmix_8_32add"),
                           ("synth32.tif", "Add", "synth16.tif", "calcmix_32_16"),
                           ("rgb", "Add", "synth8.tif", "calcmix_rgb_8")):
        assert_same(calc(s, f"image1={a} operation={op} image2={b} create"), gold[name])
    assert_same(calc(s, "image1=rgb operation=Add image2=rgb2 create 32-bit"),
                gold["calcrgb_add_float"])
    # in place (image1 changes, undoable)
    s2 = make_session()
    a = s2.add_image(doc(inputs["synth16"], "a16"))
    s2.add_image(doc(inputs["synth16"][:, ::-1], "b16"))
    assert s2.run("Image Calculator...", "image1=a16 operation=Difference image2=b16") is None
    assert_same(pixels(a), gold["calc16_inplace_diff"])
    assert a.changed and a.undo() and np.array_equal(pixels(a), inputs["synth16"])


def test_image_calculator_sizes_and_stacks(gold, inputs):
    tifffile = pytest.importorskip("tifffile")
    p8 = tifffile.imread(input_path("particles8.tif"))
    s = make_session()
    big = s.add_image(doc(p8, "particles8.tif"))
    s.add_image(doc(inputs["synth8"], "synth8.tif"))
    assert_same(calc(s, "image1=particles8.tif operation=Add image2=synth8.tif create"),
                gold["calcsize_create"])
    s.run("Image Calculator...", "image1=particles8.tif operation=Max image2=synth8.tif")
    assert_same(pixels(big), gold["calcsize_inplace"])
    stk = s.add_image(doc(inputs["stack"], "stk"))
    assert_same(calc(s, "image1=stk operation=Add image2=synth8.tif create stack"),
                gold["calcstack_add"])
    s.add_image(doc(inputs["stack"][:, ::-1, :], "stk2"))
    assert_same(calc(s, "image1=stk operation=Subtract image2=stk2 create 32-bit stack"),
                gold["calcstack_sub_float"])
    s.run("Image Calculator...", "image1=stk operation=Average image2=stk2 stack")
    assert_same(pixels(stk), gold["calcstack_inplace_ave"])


def contrast_ranges():
    with open(golden_path("process_math", "contrast_ranges.csv"), newline="") as fh:
        return {r["name"]: (float(r["min"]), float(r["max"])) for r in csv.DictReader(fh)}


def test_enhance_contrast_display_ranges(inputs):
    from rsstatchem.core.imaging.io import open_image
    want = contrast_ranges()
    files = {"synth8.tif": inputs["synth8"], "synth16.tif": inputs["synth16"],
             "synth32.tif": inputs["synth32"]}
    for name, (lo, hi) in want.items():
        stem, _, sat = name.rpartition("_")
        s = make_session()
        if stem in files:
            im = s.add_image(doc(files[stem], stem))
        elif stem in ("blobs.gif", "Cell_Colony.jpg"):
            im = s.add_image(open_image(input_path(stem)))
        elif stem == "synth16_oval":
            im = s.add_image(doc(inputs["synth16"], stem))
            im.roi = OvalRoi(10, 10, 30, 25)
        else:
            im = s.add_image(doc(inputs["synth32"], stem))
            im.roi = RectRoi(5, 5, 30, 40)
        s.run("Enhance Contrast...", f"saturated={sat}")
        got = im.display_range
        assert got[0] == pytest.approx(lo, rel=1e-6, abs=1e-6), name
        assert got[1] == pytest.approx(hi, rel=1e-6, abs=1e-6), name


def test_enhance_contrast_normalize_equalize(gold, inputs):
    from ij_golden import blobs
    srcs = {"synth8": lambda: doc(inputs["synth8"], "s8"),
            "synth16": lambda: doc(inputs["synth16"], "s16"),
            "synth32": lambda: doc(inputs["synth32"], "s32"), "blobs": blobs}
    for name, make in srcs.items():
        for opts, suffix in (("saturated=0.35 normalize", "normalize"),
                             ("saturated=3 normalize", "normalize3"),
                             ("saturated=0.35 equalize", "equalize"),
                             ("saturated=0.35 equalize classic", "equalize_classic")):
            if name == "synth32" and "equalize" in opts:
                continue
            s = make_session()
            im = s.add_image(make())
            s.run("Enhance Contrast...", opts)
            assert_same(pixels(im), gold[f"{name}_{suffix}"])
    s = make_session()
    im = s.add_image(doc(inputs["rgb"], "rgb"))
    s.run("Enhance Contrast...", "saturated=1")
    assert_same(pixels(im), gold["rgb_stretch1"])
    im = s.add_image(doc(inputs["rgb"], "rgb2"))
    s.run("Enhance Contrast...", "saturated=0.35 equalize")
    assert_same(pixels(im), gold["rgb_equalize"])
    for opts, name in (("saturated=0.35 normalize", "synth8_normalize_oval"),
                       ("saturated=0.35 normalize update", "synth8_normalize_oval_all")):
        im = s.add_image(doc(inputs["synth8"], name))
        im.roi = OvalRoi(10, 8, 34, 30)
        s.run("Enhance Contrast...", opts)
        assert_same(pixels(im), gold[name])
    for k, opts in enumerate(("saturated=0.35 process_all", "saturated=0.35 process_all use",
                              "saturated=0.35 equalize process_all",
                              "saturated=0.35 equalize process_all use")):
        im = s.add_image(doc(inputs["stack"], f"stk{k}"))
        s.run("Enhance Contrast...", opts)
        assert_same(pixels(im), gold[f"stack_contrast{k}"])


# ---------------------------------------------------------------------------------- units
def test_java_semantics_helpers():
    assert M.java_int(np.array([-1.7, 1.7, np.nan, 1e20])).tolist() == [-1, 1, 0, 2 ** 31 - 1]
    assert M.java_round(np.array([-2.5, 2.5, -0.5])).tolist() == [-2, 3, 0]
    assert M.wrap_int32(np.array([2 ** 31])).tolist() == [-(2 ** 31)]
    assert M.parse_binary("11110000") == 240
    with pytest.raises(ValueError):
        M.parse_binary("12")
    assert M.calculator_mode("Transparent-zero") == "zero"
    assert M.calculator_mode("xor create") == "xor"


def test_macro_evaluator_is_safe_and_vectorised():
    p = np.arange(16, dtype=np.float32).reshape(4, 4)
    out = M.macro_expression(p, "v = maxOf(v, 5) + (x > 1 ? 100 : 0); if (y == 0) { v = -1; }")
    assert out[0].tolist() == [-1, -1, -1, -1]
    assert out[1].tolist() == [5, 5, 106, 107]
    with pytest.raises(M.MacroError):
        M.macro_expression(p, "v = __import__('os')")
    with pytest.raises(M.MacroError):
        M.macro_expression(p, "v = q + 1")
    # bare expression gets "v=" like ImageJ; compound assignment
    assert M.macro_expression(p, "v*0+3")[0, 0] == 3
    assert M.macro_expression(p, "v+=1; v*=2")[0, 1] == 4


def test_contrast_core_helpers():
    st = CE.histogram_stats(np.array([[0, 10], [20, 255]], np.uint8))
    assert st.pixel_count == 4 and CE.stretch_range(st, 0) == (0.0, 255.0)
    with pytest.raises(ValueError):
        CE.equalize(np.zeros((2, 2), np.float32))


def test_batch_convert_and_macro(tmp_path):
    pytest.importorskip("tifffile")
    import tifffile
    src, dst = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    dst.mkdir()
    for i in range(3):
        tifffile.imwrite(src / f"img{i}.tif", np.full((8, 10), 10 * i, np.uint8),
                         imagej=True)
    s = make_session()
    s.run("Convert...", f"input=[{src}] output=[{dst}] output_format=PNG scale=0.5")
    outs = sorted(p.name for p in dst.iterdir())
    assert outs == ["img0.png", "img1.png", "img2.png"]
    from PIL import Image
    with Image.open(dst / "img2.png") as im:
        assert im.size == (5, 4) and np.asarray(im).max() == 20
    dst2 = tmp_path / "out2"
    dst2.mkdir()
    s.run("Macro... ", f'input=[{src}] output=[{dst2}] output_format=TIFF '
                       f'code=[run("Add...", "value=5");]')
    got = tifffile.imread(dst2 / "img1.tif")
    assert got.max() == 15
    s.run("Measure...", f"input=[{src}]")
    assert len(s.results) == 3 and s.results.get_value("Mean", 2) == 20
    st = s.add_image(ImageDoc.from_array(np.zeros((3, 4, 4), np.uint8), "ZYX", "vs"))
    s.run("Virtual Stack...", f'output=[] code=[run("Add...", "value=7");]')
    assert st.plane_at(2).max() == 7
