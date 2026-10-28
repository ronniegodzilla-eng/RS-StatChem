"""Image Analysis WP-D: Image menu (types, LUTs, adjust, color, stacks, hyperstacks, transforms,
zoom, overlay) - parity with the real ImageJ 1.54p (goldens in tests/data_imagej/golden/image_*,
produced by scripts/imagej_golden.py from cases/image_*.ijm) through the same macro option
strings, plus behaviour tests of the windows."""
from __future__ import annotations

import numpy as np
import pytest

from ij_golden import golden_csv, golden_images, input_path
from rsstatchem.core.imaging import adjust as A
from rsstatchem.core.imaging import color as CL
from rsstatchem.core.imaging import convert as CV
from rsstatchem.core.imaging import stacks as S
from rsstatchem.core.imaging import transform as T
from rsstatchem.core.imaging.image import COLOR_256, COLOR_RGB, GRAY8, GRAY16, GRAY32, ImageDoc
from rsstatchem.core.imaging.io import open_image
from rsstatchem.core.imaging.lut import named_lut
from rsstatchem.core.imaging.menu_layout import labels_of_area
from rsstatchem.core.imaging.session import Session

TYPES = golden_images("image_types")
TRANS = golden_images("image_transform")
STACKS = golden_images("image_stacks")


def same(a, g):
    """Exact equality (NaN == NaN), with a readable failure message."""
    a = np.asarray(a)
    g = np.asarray(g)
    assert a.shape == g.shape, (a.shape, g.shape)
    if a.dtype.kind == "f" or g.dtype.kind == "f":
        d = np.abs(a.astype(float) - g.astype(float))
        d = np.where(np.isnan(a) & np.isnan(g), 0.0, d)
        assert np.nanmax(d) == 0, f"max diff {np.nanmax(d)} at {np.count_nonzero(d)} px"
    else:
        d = np.abs(a.astype(np.int64) - g.astype(np.int64))
        assert d.max() == 0, f"max diff {d.max()} at {np.count_nonzero(d)} px"


@pytest.fixture
def session():
    from rsstatchem.ui.imagej import commands_image
    s = Session()
    commands_image.register(s.registry)
    return s


def opened(session, name):
    doc = open_image(input_path(name))
    session.add_image(doc)
    return doc


def add(session, array, axes="YX", title="img", display=None):
    doc = ImageDoc.from_array(np.array(array, copy=True), axes, title)
    if display is not None:
        doc.set_display_range(*display)
    session.add_image(doc)
    return doc


def rgb_input(session):
    return add(session, TYPES["input_rgb"], "YXS", "input_rgb")


def pixels(doc):
    return doc.pixels.squeeze()


# ---------------------------------------------------------------------------------- registry
def test_all_image_menu_labels_are_registered(session):
    missing = [lb for lb in labels_of_area("image") if lb not in session.registry]
    assert missing == []
    for extra in ("Bleach Correction", "Colour Deconvolution", "KymographBuilder"):
        assert extra in session.registry
    # thresholds stay with WP-B
    assert "Threshold..." not in session.registry


# ---------------------------------------------------------------------------------- LUTs
LUT_NAMES = {"fire": "Fire", "grays": "Grays", "ice": "Ice", "spectrum": "Spectrum",
             "rgb332": "3-3-2 RGB", "red": "Red", "green": "Green", "blue": "Blue",
             "cyan": "Cyan", "magenta": "Magenta", "yellow": "Yellow",
             "redgreen": "Red/Green"}


@pytest.mark.parametrize("key", sorted(LUT_NAMES))
def test_builtin_lut_tables_exact(session, key):
    ramp = add(session, np.arange(256, dtype=np.uint8)[None, :], title="ramp")
    session.run(LUT_NAMES[key])
    session.run("RGB Color")
    same(pixels(ramp), TYPES["lut_" + key][0])


def test_invert_lut_and_named_registry(session):
    ramp = add(session, np.arange(256, dtype=np.uint8)[None, :], title="ramp")
    session.run("Fire")
    session.run("Invert LUT")
    session.run("RGB Color")
    same(pixels(ramp), TYPES["lut_fire_inverted"][0])
    assert named_lut("mpl-viridis").as_array().shape == (256, 3)


def test_apply_lut_8_and_16_bit(session):
    d = opened(session, "synth8.tif")
    d.set_display_range(60, 200)
    session.run("Apply LUT")
    same(pixels(d), TYPES["apply8"])
    d = opened(session, "synth16.tif")
    d.set_display_range(3000, 30000)
    session.run("Apply LUT")
    same(pixels(d), TYPES["apply16"])
    assert d.display_range == (0.0, 65535.0)


# ---------------------------------------------------------------------------------- types
TYPE_CASES = [
    ("s16_8_auto", "synth16.tif", None, "8-bit", True, False),
    ("s16_8_range", "synth16.tif", (3000, 30000), "8-bit", True, False),
    ("s16_8_noscale", "synth16.tif", None, "8-bit", False, False),
    ("s16_32", "synth16.tif", None, "32-bit", True, False),
    ("s16_rgb", "synth16.tif", (3000, 30000), "RGB Color", True, False),
    ("s32_8_auto", "synth32.tif", None, "8-bit", True, False),
    ("s32_8_range", "synth32.tif", (-1, 4), "8-bit", True, False),
    ("s32_16_auto", "synth32.tif", None, "16-bit", True, False),
    ("s32_16_range", "synth32.tif", (-1, 4), "16-bit", True, False),
    ("s32_8_noscale", "synth32.tif", None, "8-bit", False, False),
    ("s32_16_noscale", "synth32.tif", None, "16-bit", False, False),
    ("s32_rgb", "synth32.tif", (-1, 4), "RGB Color", True, False),
    ("blobs_rgb", "blobs.gif", None, "RGB Color", True, False),
    ("s8_8_range", "synth8.tif", (40, 210), "8-bit", True, False),
    ("rgb_8", "rgb", None, "8-bit", True, False),
    ("rgb_8_weighted", "rgb", None, "8-bit", True, True),
    ("rgb_32_weighted", "rgb", None, "32-bit", True, True),
    ("rgb_16", "rgb", None, "16-bit", True, False),
    ("rgb_32", "rgb", None, "32-bit", True, False),
]


@pytest.mark.parametrize("key,name,rng,item,scale,weighted", TYPE_CASES,
                         ids=[c[0] for c in TYPE_CASES])
def test_type_conversions_exact(session, key, name, rng, item, scale, weighted):
    doc = rgb_input(session) if name == "rgb" else opened(session, name)
    if rng:
        doc.set_display_range(*rng)
    session.options.scale_conversions = scale
    session.options.weighted_rgb = weighted
    session.run(item)
    same(pixels(session.active), TYPES[key])


def test_fire_lut_to_rgb_uses_display_range(session):
    d = opened(session, "synth8.tif")
    session.run("Fire")
    d.set_display_range(40, 210)
    session.run("RGB Color")
    same(pixels(d), TYPES["s8_fire_rgb"])


@pytest.mark.parametrize("item,key", [("RGB Stack", "rgb_rgbstack"), ("HSB Stack", "rgb_hsb"),
                                      ("HSB (32-bit)", "rgb_hsb32"), ("Lab Stack", "rgb_lab")])
def test_rgb_to_stacks_and_back(session, item, key):
    rgb_input(session)
    session.run(item)
    doc = session.active
    assert doc.n_channels == 3 and doc.composite_mode == "grayscale"
    same(pixels(doc), TYPES[key])
    session.run("RGB Color")
    same(pixels(session.active), TYPES[key + "_back"])
    assert session.active.type == COLOR_RGB


@pytest.mark.parametrize("n,key", [(256, "rgb_mc256"), (16, "rgb_mc16")])
def test_median_cut_8bit_color(session, n, key):
    doc = rgb_input(session)
    session.run("8-bit Color", f"number={n}")
    assert doc.type == COLOR_256
    same(pixels(doc), TYPES[key])
    session.run("RGB Color")
    same(pixels(doc), TYPES[key + "_rgb"])


def test_median_cut_on_cell_colony(session):
    doc = opened(session, "Cell_Colony.jpg")
    session.run("RGB Color")
    session.run("8-bit Color", "number=7")
    session.run("RGB Color")
    same(pixels(doc), TYPES["colony_mc7_rgb"])


def test_merge_channels_rgb_and_composite_to_rgb(session):
    r = opened(session, "synth8.tif")
    r.title = "r"
    g = add(session, 255 - pixels(r), title="g")
    b = add(session, np.rot90(pixels(r), -1).copy(), title="b")
    session.run("Merge Channels...", "c1=r c2=g c3=b")
    same(pixels(session.active), TYPES["input_rgb"])
    assert len(session.images) == 1
    c1 = add(session, pixels(r), title="c1", display=(20, 180))
    c2 = add(session, 255 - pixels(r), title="c2", display=(50, 255))
    session.run("Merge Channels...", "c1=c1 c2=c2 create")
    comp = session.active
    assert comp.n_channels == 2 and comp.composite_mode == "composite"
    assert comp.luts[0] == named_lut("Red") and comp.luts[1] == named_lut("Green")
    session.run("RGB Color")
    same(pixels(comp), TYPES["composite_rgb"])
    assert c1 not in session.images and c2 not in session.images


def test_unsupported_conversion_message(session):
    from rsstatchem.core.imaging.commands import CommandError
    opened(session, "synth16.tif")
    with pytest.raises(CommandError, match="requires an image of type"):
        session.run("8-bit Color")


# ---------------------------------------------------------------------------------- transforms
def _trans_input(session, name):
    if name == "rgb":
        return add(session, TRANS["input_rgb"], "YXS", "rgb")
    return opened(session, name + ".tif")


IMAGES = ("synth8", "synth16", "synth32", "rgb")
METHODS = ("None", "Bilinear", "Bicubic")


@pytest.mark.parametrize("name", IMAGES)
@pytest.mark.parametrize("meth", METHODS)
def test_scale_size_rotate_translate_exact(session, name, meth):
    """Scale (up / down / averaging / mixed / in place), Size, Rotate (15, -33.3 enlarged) and
    Translate with every interpolation, through ImageJ's own option strings."""
    runs = [
        ("scale_up", "Scale...", f"x=1.7 y=1.3 interpolation={meth} create title=up"),
        ("scale_down", "Scale...", f"x=0.6 y=0.45 interpolation={meth} create title=dn"),
        ("scale_downavg", "Scale...",
         f"x=0.6 y=0.45 interpolation={meth} average create title=dna"),
        ("scale_mix", "Scale...", f"x=0.5 y=1.5 interpolation={meth} average create title=mix"),
        ("scale_inplace", "Scale...", f"x=0.7 y=0.8 interpolation={meth}"),
        ("size", "Size...", f"width=50 height=37 depth=1 average interpolation={meth}"),
        ("rot15", "Rotate... ", f"angle=15 grid=1 interpolation={meth}"),
        ("rotm33_enlarge", "Rotate... ", f"angle=-33.3 grid=1 interpolation={meth} enlarge"),
        ("translate", "Translate...", f"x=3.5 y=-2.25 interpolation={meth}"),
    ]
    for key, label, options in runs:
        session.images.clear()
        session.active = None
        _trans_input(session, name)
        session.run(label, options)
        same(pixels(session.active), TRANS[f"{name}_{key}_{meth}"])


@pytest.mark.parametrize("name", IMAGES)
def test_bin_canvas_translate_int_exact(session, name):
    for b in T.BIN_METHODS:
        session.images.clear()
        session.active = None
        _trans_input(session, name)
        session.run("Bin...", f"x=3 y=2 bin={b}")
        same(pixels(session.active), TRANS[f"{name}_bin_{b}"])
    session.images.clear()
    session.active = None
    _trans_input(session, name)
    session.run("Canvas Size...", "width=80 height=50 position=Bottom-Right zero")
    same(pixels(session.active), TRANS[f"{name}_canvas"])
    session.images.clear()
    session.active = None
    _trans_input(session, name)
    session.run("Translate...", "x=5 y=-3 interpolation=None")
    same(pixels(session.active), TRANS[f"{name}_translate_int"])


def test_flips_rotate90_crop_duplicate(session):
    from rsstatchem.core.imaging.roi import OvalRoi, RectRoi
    a = np.arange(30, dtype=np.uint8).reshape(5, 6)
    doc = add(session, a, title="a.tif")
    session.run("Flip Horizontally")
    same(pixels(doc), a[:, ::-1])
    session.run("Flip Vertically")
    same(pixels(doc), a[::-1, ::-1])
    session.run("Rotate 90 Degrees Right")
    same(pixels(doc), np.rot90(a[::-1, ::-1], -1))
    session.run("Rotate 90 Degrees Left")
    same(pixels(doc), a[::-1, ::-1])
    doc.roi = RectRoi(1, 1, 3, 2)
    dup = session.run("Duplicate...", "title=d")
    assert dup.title == "d" and pixels(dup).shape == (2, 3)
    session.set_active(doc)
    doc.roi = RectRoi(1, 1, 3, 2)
    session.run("Crop")
    assert (doc.width, doc.height) == (3, 2) and doc.roi is None
    assert doc.can_undo()
    doc.undo()
    assert doc.width == 6
    doc.roi = OvalRoi(0, 0, 4, 4)
    session.run("Flip Horizontally")          # masked to the oval like ImageJ
    assert doc.pixels.squeeze()[4, 5] == a[::-1, ::-1][4, 5]


def test_image_results_roundtrip(session):
    a = (np.arange(12, dtype=np.float32).reshape(3, 4) / 2)
    add(session, a, title="f")
    rt = session.run("Image to Results")
    assert rt.headings[:2] == ["X0", "X1"] and rt.get_value("X3", 2) == 5.5
    out = session.run("Results to Image")
    same(pixels(out), a)


# ---------------------------------------------------------------------------------- stacks
KINDS = ("8", "16", "32", "rgb")


def stack_in(session, kind, title=None):
    arr = STACKS["input_stack" + kind]
    doc = add(session, arr, "ZYXS" if kind == "rgb" else "ZYX", title or f"stack{kind}")
    if kind in ("16", "32"):
        doc.set_display_range(*open_image(input_path(f"synth{kind}.tif")).display_range)
    return doc


@pytest.mark.parametrize("kind", KINDS)
def test_z_project_all_methods_exact(session, kind):
    short = dict(zip(S.PROJECTIONS, ("avg", "max", "min", "sum", "sd", "median")))
    for method in S.PROJECTIONS:
        stack_in(session, kind)
        out = session.run("Z Project...", f"start=1 stop=6 projection=[{method}]")
        same(pixels(out), STACKS[f"zp_{kind}_{short[method]}"])
        assert out.title.startswith(S.PREFIX[method])
    stack_in(session, kind)
    out = session.run("Z Project...", "start=2 stop=5 projection=[Max Intensity]")
    same(pixels(out), STACKS[f"zp_{kind}_max_2_5"])
    stack_in(session, kind)
    out = session.run("Grouped Z Project...", "projection=[Average Intensity] group=2")
    same(pixels(out), STACKS[f"gzp_{kind}_avg2"])


@pytest.mark.parametrize("kind", KINDS)
def test_reslice_montage_reduce_bin_exact(session, kind):
    cases = [("reslice_{k}_top", "Reslice [/]...", "output=1.000 start=Top avoid"),
             ("reslice_{k}_left", "Reslice [/]...", "output=1.000 start=Left avoid"),
             ("reslice_{k}_bottom_flip_rotate", "Reslice [/]...",
              "output=1.000 start=Bottom flip rotate avoid"),
             ("montage_{k}", "Make Montage...", "columns=3 rows=2 scale=1"),
             ("montage_{k}_half_border", "Make Montage...",
              "columns=2 rows=2 scale=0.5 first=2 last=6 increment=1 border=2")]
    for key, label, options in cases:
        stack_in(session, kind)
        out = session.run(label, options)
        same(pixels(out), STACKS[key.format(k=kind)])
    doc = stack_in(session, kind)
    session.run("Properties...", "channels=1 slices=6 frames=1 pixel_width=1 pixel_height=1 "
                                 "voxel_depth=2.5")
    assert doc.calibration.pixel_depth == 2.5
    out = session.run("Reslice [/]...", "output=1.000 start=Right")
    same(pixels(out), STACKS[f"reslice_{kind}_right_z2.5"])
    stack_in(session, kind)
    montage = session.run("Make Montage...", "columns=3 rows=2 scale=1")
    session.set_active(montage)
    out = session.run("Montage to Stack...", "columns=3 rows=2 border=0")
    same(pixels(out), STACKS[f"m2s_{kind}"])
    doc = stack_in(session, kind)
    session.run("Reduce...", "reduction=2")
    same(pixels(doc), STACKS[f"reduce_{kind}"])
    doc = stack_in(session, kind)
    session.run("Bin...", "x=2 y=2 z=3 bin=Average")
    same(pixels(doc), STACKS[f"binz_{kind}"])


P3D = [
    ("p3d_8_bright_y", "8", "projection=[Brightest Point] axis=Y-Axis slice=1 initial=0 "
     "total=90 rotation=30 lower=1 upper=255 opacity=0 surface=100 interior=100"),
    ("p3d_8_bright_x_cue", "8", "projection=[Brightest Point] axis=X-Axis slice=1 initial=10 "
     "total=60 rotation=25 lower=1 upper=255 opacity=0 surface=100 interior=50"),
    ("p3d_8_nearest_x_cue", "8", "projection=[Nearest Point] axis=X-Axis slice=1 initial=0 "
     "total=90 rotation=45 lower=100 upper=255 opacity=0 surface=50 interior=100"),
    ("p3d_8_mean_z", "8", "projection=[Mean Value] axis=Z-Axis slice=1 initial=0 total=90 "
     "rotation=30 lower=1 upper=255 opacity=0 surface=100 interior=100"),
    ("p3d_8_bright_z_opacity", "8", "projection=[Brightest Point] axis=Z-Axis slice=1 "
     "initial=0 total=40 rotation=20 lower=50 upper=200 opacity=40 surface=70 interior=60"),
    ("p3d_8_bright_y_interp", "8", "projection=[Brightest Point] axis=Y-Axis slice=3 "
     "initial=0 total=60 rotation=30 lower=1 upper=255 opacity=0 surface=100 interior=100 "
     "interpolate"),
    ("p3d_8_nearest_y_z2", "8", "projection=[Nearest Point] axis=Y-Axis slice=2 initial=0 "
     "total=60 rotation=30 lower=90 upper=255 opacity=0 surface=100 interior=100"),
    ("p3d_16_bright_y", "16", "projection=[Brightest Point] axis=Y-Axis slice=1 initial=0 "
     "total=30 rotation=30 lower=1 upper=255 opacity=0 surface=100 interior=100"),
    ("p3d_rgb_bright_y", "rgb", "projection=[Brightest Point] axis=Y-Axis slice=1 initial=0 "
     "total=30 rotation=30 lower=1 upper=255 opacity=0 surface=100 interior=100"),
]


@pytest.mark.parametrize("key,kind,options", P3D, ids=[p[0] for p in P3D])
def test_3d_project_exact(session, key, kind, options):
    stack_in(session, kind)
    out = session.run("3D Project...", options)
    same(pixels(out), STACKS[key])


def test_stack_tools_exact(session):
    doc = stack_in(session, "8")
    session.run("Reverse")
    same(pixels(doc), STACKS["reverse_8"])
    stack_in(session, "8")
    same(pixels(session.run("Make Substack...", "slices=2-5")), STACKS["substack_8_2_5"])
    stack_in(session, "8")
    same(pixels(session.run("Make Substack...", "slices=1,3,6")), STACKS["substack_8_list"])
    for options, key in (("stack1=a stack2=b", "combine_8"),
                         ("stack1=a stack2=b combine", "combine_8_vertical")):
        stack_in(session, "8", "a")
        stack_in(session, "8", "b")
        session.run("Reverse")
        same(pixels(session.run("Combine...", options)), STACKS[key])
    stack_in(session, "8", "a")
    stack_in(session, "8", "b")
    session.run("Reverse")
    out = session.run("Concatenate...", "  title=cat image1=a image2=b")
    same(pixels(out), STACKS["concat_8"])
    assert out.title == "cat" and out.stack_size == 12


def test_stack_conversions_and_hyperstacks_exact(session):
    doc = stack_in(session, "16")
    doc.set_display_range(3000, 30000)
    session.run("8-bit")
    same(pixels(doc), STACKS["stack16_8_range"])
    doc = stack_in(session, "32")
    session.run("16-bit")
    same(pixels(doc), STACKS["stack32_16"])
    assert doc.display_range == (0.0, 65535.0)
    doc = stack_in(session, "32")
    session.run("8-bit")
    same(pixels(doc), STACKS["stack32_8"])
    doc = stack_in(session, "8")
    session.run("Stack to Hyperstack...",
                "order=xyzct channels=2 slices=3 frames=1 display=Color")
    assert doc.dims == (2, 3, 1)
    same(np.moveaxis(doc.pixels[0], 0, 0), STACKS["hyper_xyzct"])
    doc = stack_in(session, "8")
    session.run("Stack to Hyperstack...",
                "order=xytzc channels=3 slices=1 frames=2 display=Color")
    same(doc.pixels[:, 0], STACKS["hyper_xytzc"])
    session.run("Hyperstack to Stack")
    assert doc.dims == (1, 6, 1)


def test_stack_statistics_exact(session):
    for kind, row in (("8", 0), ("16", 1)):
        stack_in(session, kind)
        session.results.reset()
        rt = session.run("Statistics")
        gold = golden_csv("image_stacks", f"stats_{kind}.csv")
        for col in ("Voxels", "%Volume", "Mean", "StdDev", "Min", "Max", "Mode", "Median"):
            assert rt.get_value(col, 0) == pytest.approx(gold.get_value(col, row), rel=1e-9)


def test_slice_navigation_add_delete_labels(session):
    doc = add(session, np.zeros((3, 4, 5), np.uint8), "ZYX", "s")
    session.run("Next Slice [>]")
    assert doc.current_index == 1
    session.run("Previous Slice [<]")
    session.run("Previous Slice [<]")
    assert doc.current_index == 2
    session.run("Set Slice...", "slice=1")
    session.run("Set Label...", "label=first")
    assert doc.slice_labels[0] == "first"
    session.run("Add Slice")
    assert doc.stack_size == 4 and doc.slice_labels[0] == "first"
    session.run("Delete Slice")
    assert doc.stack_size == 3
    session.run("Remove Slice Labels")
    assert doc.slice_labels == [None, None, None]
    parts = session.run("Stack to Images")
    assert [p.title for p in parts] == ["s-0001", "s-0002", "s-0003"]
    stack = session.run("Images to Stack", "name=Again title=s- use")
    assert stack.stack_size == 3 and stack.slice_labels[1] == "s-0002"


def test_hyperstack_tools(session):
    doc = session.run("New Hyperstack...", "title=HS type=16-bit display=Composite width=20 "
                                           "height=10 channels=2 slices=3 frames=4 label")
    assert doc.dims == (2, 3, 4) and doc.slice_labels[0] == "c:1/2 z:1/3 t:1/4"
    doc.pixels[...] = np.arange(doc.stack_size, dtype=np.uint16).reshape(4, 3, 2, 1, 1)
    sub = session.run("Make Subset...", "channels=2 slices=1-2 frames=1,4")
    assert sub.dims == (1, 2, 2) and sub.pixels[1, 1, 0, 0, 0] == doc.pixels[3, 1, 1, 0, 0]
    session.set_active(doc)
    doc.set_position(1, 2, 3)
    red = session.run("Reduce Dimensionality...", "channels keep")
    assert red.dims == (2, 1, 1) and red.pixels[0, 0, 1, 0, 0] == doc.pixels[3, 2, 1, 0, 0]
    session.set_active(doc)
    zp = session.run("Z Project...", "start=1 stop=3 projection=[Max Intensity] all")
    assert zp.dims == (2, 1, 4)
    assert zp.pixels[2, 0, 1, 0, 0] == doc.pixels[2, :, 1, 0, 0].max()


def test_scale_with_size_and_z(session):
    stack_in(session, "8")
    out = session.run("Scale...", "width=24 height=20 depth=3 interpolation=Bilinear "
                                  "average process create")
    assert (out.width, out.height, out.n_slices) == (24, 20, 3)


# ---------------------------------------------------------------------------------- adjust
def test_auto_contrast_threshold_halving():
    rng = np.random.default_rng(1)
    p = np.clip(rng.normal(100, 20, (64, 64)), 0, 255).astype(np.uint8)
    lo, hi, t = A.auto_range(p)
    assert t == A.AUTO_THRESHOLD and 0 < lo < 100 < hi < 255
    # the threshold is pixelCount/5000 = 0 counts: the first/last bins with any pixel
    assert lo == p.min() and hi == p.max()
    lo2, hi2, t2 = A.auto_range(p, t)      # second click: threshold 4096/2500 = 1 count
    assert t2 == 2500 and lo <= lo2 < hi2 <= hi
    h = np.bincount(p.ravel(), minlength=256)
    assert h[int(lo2)] > 1 and h[int(hi2)] > 1
    big = np.clip(rng.normal(30000, 2000, (300, 300)), 0, 65535).astype(np.uint16)
    lo, hi, _ = A.auto_range(big)
    hist, hmin, bs, vmin, vmax, n = A.histogram256(big)
    assert hmin == big.min() and bs == pytest.approx((big.max() - big.min() + 1) / 256)
    assert vmin < lo < hi < vmax


def test_bc_slider_maths_round_trip():
    b, c = A.sliders_from_range(0, 255, 0, 255)
    assert (b, c) == (128, 128)
    lo, hi = A.range_from_brightness(64, 0, 255, 0, 255)
    assert (lo, hi) == (63.75, 318.75)
    lo, hi = A.range_from_contrast(192, 0, 255, 0, 255)
    assert hi - lo == pytest.approx(127.5)
    assert A.window_level(10, 30) == (20, 20)
    assert A.range_from_window_level(20, 20) == (10, 30)


def test_color_balance_and_rgb_apply():
    rgb = TYPES["input_rgb"]
    out = A.color_balance(rgb, "Red", 50, 200)
    same(out[..., 1:], rgb[..., 1:])
    table = A.rgb_display_table(50, 200)
    same(out[..., 0], table[rgb[..., 0]])
    doc = ImageDoc.from_array(rgb.copy(), "YXS")
    doc.set_display_range(50, 200)
    A.apply_lut(doc)
    same(doc.plane(), np.stack([table[rgb[..., k]] for k in range(3)], axis=-1))


def test_canvas_coordinates_properties_line_width(session):
    doc = add(session, np.ones((4, 6), np.uint8) * 7, title="c")
    session.run("Canvas Size...", "width=10 height=8 position=Center zero")
    assert pixels(doc).shape == (8, 10) and pixels(doc)[2, 2] == 7 and pixels(doc)[0, 0] == 0
    session.run("Coordinates...", "left=0 right=5 top=0 bottom=4 unit=mm")
    cal = doc.calibration
    assert cal.pixel_width == 0.5 and cal.pixel_height == 0.5 and cal.unit == "mm"
    session.run("Properties...", "channels=1 slices=1 frames=1 unit=um pixel_width=0.25 "
                                 "pixel_height=0.25 voxel_depth=1 frame=2 origin=3,4")
    assert cal.pixel_width == 0.25 and cal.frame_interval == 2 and cal.x_origin == 3
    session.run("Line Width... ", "line=3")
    assert session.options.line_width == 3


# ---------------------------------------------------------------------------------- color
def test_split_arrange_composite_stack_to_rgb(session):
    rgb_input(session)
    parts = session.run("Split Channels")
    assert [p.title for p in parts] == ["input_rgb (red)", "input_rgb (green)",
                                        "input_rgb (blue)"]
    same(pixels(parts[1]), TYPES["input_rgb"][..., 1])
    doc = rgb_input(session)
    session.run("Make Composite", "display=Composite")
    assert doc.n_channels == 3 and doc.luts[2] == named_lut("Blue")
    session.run("Arrange Channels...", "new=312")
    same(doc.pixels[0, 0, 0], TYPES["input_rgb"][..., 2])
    assert doc.luts[0] == named_lut("Blue")
    session.run("Invert LUTs")
    assert doc.luts[0].blue[0] == 255
    doc2 = rgb_input(session)
    session.run("Make Composite")
    out = session.run("Stack to RGB")
    same(pixels(out), TYPES["input_rgb"])
    assert doc2 in session.images
    session.set_active(doc2)
    lut_img = session.run("Show LUT")
    assert pixels(lut_img).shape == (32, 256, 3)
    luts = session.run("Display LUTs")
    assert luts.type == COLOR_RGB and luts.height >= 12 * 36


def test_colour_deconvolution_unmixes_pure_stains(session):
    m = CL.stain_matrix(CL.STAINS["H DAB"])
    assert np.allclose(np.linalg.norm(m, axis=1), 1.0)
    conc = np.zeros((4, 4, 3))
    conc[:2, :, 0] = 0.8          # haematoxylin only
    conc[2:, :, 1] = 0.5          # DAB only
    od = conc.reshape(-1, 3) @ m
    rgb = np.clip(np.round(256 * 10 ** (-od) - 1), 0, 255).astype(np.uint8).reshape(4, 4, 3)
    planes, _ = CL.colour_deconvolution(rgb, "H DAB", "32bit_OD")
    assert planes[0][:2].mean() == pytest.approx(0.8, abs=0.02)
    assert abs(planes[1][:2]).max() < 0.02 and planes[1][2:].mean() == pytest.approx(0.5,
                                                                                    abs=0.02)
    back = CL.recompose(planes, m)
    assert np.abs(back.astype(int) - rgb).max() <= 1
    add(session, rgb, "YXS", "ihc")
    outs = session.run("Colour Deconvolution", "vectors=[H DAB] show")
    assert [o.title for o in outs[:3]] == ["ihc-(Colour_1)", "ihc-(Colour_2)",
                                           "ihc-(Colour_3)"]
    assert outs[0].plane()[:2].mean() < outs[0].plane()[2:].mean()


# ---------------------------------------------------------------------------------- fiji extras
def test_bleach_correction_methods(session):
    t = np.arange(10)
    base = np.random.default_rng(3).uniform(50, 150, (16, 16))
    frames = np.stack([base * (0.6 * np.exp(-0.2 * k) + 0.4) for k in t]).astype(np.float32)
    doc = add(session, frames, "TYX", "decay")
    for method in ("Simple Ratio", "Exponential Fit", "Histogram Matching"):
        session.set_active(doc)
        res = session.run("Bleach Correction", f"correction=[{method}] background=0")
        out = res[0] if isinstance(res, list) else res
        means = out.pixels.reshape(10, -1).mean(axis=1)
        assert np.ptp(means) / means[0] < 0.02, method
    from rsstatchem.core.imaging.fiji.bleach import exponential_fit
    _, (a, b, c) = exponential_fit(frames)
    assert b == pytest.approx(0.2, rel=1e-3)


def test_kymograph_tracks_a_moving_spot(session):
    from rsstatchem.core.imaging.roi import LineRoi
    frames = np.zeros((5, 10, 30), np.uint8)
    for k in range(5):
        frames[k, 4:6, 5 + 4 * k] = 200
    doc = add(session, frames, "TYX", "move")
    doc.roi = LineRoi(0, 5, 30, 5, width=3)
    out = session.run("KymographBuilder")
    k = pixels(out)
    assert k.shape[0] == 5
    peaks = k.argmax(axis=1)
    assert list(np.diff(peaks)) == [4, 4, 4, 4]


# ---------------------------------------------------------------------------------- overlay
def test_overlay_commands(session):
    from rsstatchem.core.imaging.roi import OvalRoi, RectRoi
    doc = add(session, np.zeros((20, 30), np.uint8), title="ov")
    doc.roi = RectRoi(2, 3, 5, 4)
    session.run("Add Selection...")
    doc.roi = OvalRoi(10, 5, 8, 8)
    session.run("Add Selection...")
    assert len(doc.overlay) == 2 and doc.roi is None
    session.run("Hide Overlay")
    assert doc.overlay.hidden
    session.run("Toggle Overlay")
    assert not doc.overlay.hidden
    rt = session.run("List Elements")
    assert rt.get_value("Width", 0) == 5
    flat = session.run("Flatten")
    assert flat.type == COLOR_RGB and tuple(pixels(flat)[3, 2]) == (0, 255, 255)
    session.set_active(doc)
    session.run("To ROI Manager")
    assert len(session.roi_manager) == 2 and doc.overlay is None
    session.run("From ROI Manager")
    assert len(doc.overlay) == 2
    session.run("Labels...", "color=red font=12 show")
    assert doc.overlay.show_labels and doc.overlay.label_font_size == 12
    session.run("Overlay Options...", "stroke=red width=2 fill=none set apply")
    assert doc.overlay.rois[0].stroke_color == "#ff0000"
    other = add(session, np.full((4, 4), 255, np.uint8), title="small")
    session.set_active(doc)
    session.run("Add Image...", "image=small x=1 y=1 opacity=100")
    assert len(doc.overlay) == 3
    assert other in session.images
    session.run("Remove Overlay")
    assert doc.overlay is None


def test_label_burns_text_and_overlay(session):
    doc = add(session, np.zeros((3, 30, 60), np.uint8), "ZYX", "lab")
    session.run("Label...", "format=0 starting=0 interval=1 x=5 y=20 font=12 text= "
                            "range=1-3")
    assert all(doc.pixels[0, z, 0].max() > 0 for z in range(3))
    doc2 = add(session, np.zeros((2, 30, 60), np.uint8), "ZYX", "lab2")
    session.run("Label...", "format=00:00 starting=0 interval=65 x=5 y=20 font=12 "
                            "range=1-2 use")
    assert [r.text for r in doc2.overlay] == ["00:00", "01:05"]


def test_show_info_and_zoom_noops(session):
    doc = opened(session, "blobs.gif")
    from rsstatchem.ui.imagej.commands_image import image_info
    text = image_info(doc)
    assert "Title: blobs.gif" in text and "Width:  256 pixels" in text
    session.run("Show Info...")
    assert "Title: blobs.gif" in session.log_lines[-1]
    session.run("In [+]")                   # zoom commands are no-ops in batch mode
    session.run("Original Scale")


# ---------------------------------------------------------------------------------- UI
@pytest.fixture
def bar(qapp):
    from rsstatchem.ui.imagej.bar import ImageJBar
    b = ImageJBar(None)
    b.message_hook = lambda title, text: None
    yield b
    b.close()


def test_image_menu_entries_enabled(bar):
    acts = bar.actions_by_label
    for label in labels_of_area("image"):
        assert acts[label].isEnabled(), label


def test_bc_window_auto_reset_apply(bar):
    from rsstatchem.ui.imagej.frames.bc_adjuster import show_adjuster
    doc = bar.open_path(str(input_path("synth16.tif")))
    win = show_adjuster(bar, "bc")
    assert win.windowTitle() == "B&C" and win.image is doc
    lo0, hi0 = doc.display_range
    win.buttons["Auto"].click()
    lo, hi = doc.display_range
    assert (lo, hi) != (lo0, hi0) and lo0 <= lo < hi <= hi0
    expected = A.auto_range(doc.plane())[:2]
    assert (lo, hi) == expected
    win.sliders["Brightness"].setValue(100)
    assert doc.display_range != (lo, hi)
    win.buttons["Reset"].click()
    assert doc.display_range == (lo0, hi0)
    doc.set_display_range(3000, 30000)
    win.apply()
    assert doc.display_range == (0.0, 65535.0) and doc.can_undo()
    same(doc.pixels.squeeze(), TYPES["apply16"])
    wl = show_adjuster(bar, "wl")
    assert set(wl.sliders) == {"Level", "Window"}


def test_color_balance_window_rgb(bar):
    from rsstatchem.ui.imagej.frames.bc_adjuster import show_adjuster
    doc = ImageDoc.from_array(TYPES["input_rgb"].copy(), "YXS", "rgb")
    bar.session.add_image(doc)
    win = show_adjuster(bar, "balance")
    win.channel.setCurrentText("Red")
    win.set_range(50, 200)
    same(doc.plane()[..., 0], A.rgb_display_table(50, 200)[TYPES["input_rgb"][..., 0]])
    same(doc.plane()[..., 1], TYPES["input_rgb"][..., 1])
    win.buttons["Reset"].click()
    same(doc.plane(), TYPES["input_rgb"])


def test_channels_tool_and_ortho_views(bar):
    from rsstatchem.ui.imagej.frames.channels_tool import show_channels_tool
    from rsstatchem.ui.imagej.frames.ortho_views import show_ortho_views
    doc = ImageDoc.from_array(np.random.default_rng(0).integers(0, 255, (3, 5, 16, 20))
                              .astype(np.uint8), "CZYX", "hs")
    bar.session.add_image(doc)
    tool = show_channels_tool(bar)
    assert len(tool.boxes) == 3
    tool.boxes[1].setChecked(False)
    assert doc.active_channels == [True, False, True]
    tool.mode.setCurrentText("Grayscale")
    assert doc.composite_mode == "grayscale"
    doc.calibration.pixel_depth = 2.0
    ortho = show_ortho_views(bar, doc)
    ortho.set_point(7, 9)
    xz, yz = ortho.views()
    assert xz.shape == (10, 20, 3) and yz.shape == (16, 10, 3)
    ortho.close()


def test_run_command_through_bar_dialog_free(bar):
    bar.open_path(str(input_path("blobs.gif")))
    bar.run_command("Fire", "")
    doc = bar.session.active
    assert doc.lut == named_lut("Fire")
    bar.run_command("Scale...", "x=0.5 y=0.5 interpolation=Bilinear average create")
    assert bar.session.active.width == 128
    bar.run_command("In [+]")
    from rsstatchem.ui.imagej.windows import window_for
    assert window_for(bar.session.active).canvas.magnification == 1.5
