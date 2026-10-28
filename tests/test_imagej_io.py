"""Image Analysis file I/O (WP-A) against the real ImageJ 1.54p.

Two golden directions (scripts/imagej_golden.py):
* ``io_write``: files written by ImageJ -> our readers give ImageJ's pixels/metadata, and our
  writers reproduce ImageJ's bytes (TIFF, FITS, raw, text image, .roi);
* ``io_read``: files written by us (tests/data_imagej/inputs/io, regenerated here and required
  to be identical) -> what ImageJ read from them.
"""
from __future__ import annotations

import csv
import io
import zipfile

import numpy as np
import pytest

from ij_golden import golden_images, golden_path, input_path

pytest.importorskip("tifffile")
pytest.importorskip("roifile")

from rsstatchem.core.imaging.image import COLOR_256, COLOR_RGB, GRAY8, GRAY16, GRAY32, ImageDoc  # noqa: E402
from rsstatchem.core.imaging.io import open_file, open_image, save_image, sniff  # noqa: E402
from rsstatchem.core.imaging.io import roi_codec  # noqa: E402
from rsstatchem.core.imaging.io.tiff import java_double, read_tiff, tiff_bytes  # noqa: E402
from rsstatchem.core.imaging.roi import shapes as S  # noqa: E402

W = "io_write"
R = "io_read"


def gw(name):
    return golden_path(W, name)


def _make_inputs():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "make_io_inputs", input_path("").parent / "make_io_inputs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rows(case, name):
    with open(golden_path(case, name), newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------------- helpers
def test_java_double_strings():
    assert [java_double(v) for v in (2000, -2.0, 0.6666666666666666, 1e-5, 1.5e7, 0.001,
                                     32768.0, 1e7, 123456.789)] == \
        ["2000.0", "-2.0", "0.6666666666666666", "1.0E-5", "1.5E7", "0.001", "32768.0",
         "1.0E7", "123456.789"]


def test_lzw_round_trip_and_pillow_files(tmp_path):
    from PIL import Image

    from rsstatchem.core.imaging.io.lzw import lzw_decode, lzw_encode
    rng = np.random.default_rng(3)
    for n in (0, 1, 7, 600, 20000):
        data = bytes(rng.integers(0, 6, n, dtype=np.uint8))
        assert lzw_decode(lzw_encode(data)) == data
    a8 = (rng.integers(0, 40, (90, 130)) * 3).astype(np.uint8)
    a16 = rng.integers(0, 60000, (40, 50)).astype(np.uint16)
    rgb = rng.integers(0, 255, (30, 20, 3)).astype(np.uint8)
    for name, arr in (("a8", a8), ("a16", a16), ("rgb", rgb)):
        Image.fromarray(arr).save(tmp_path / f"{name}.tif", compression="tiff_lzw")
        doc = open_image(tmp_path / f"{name}.tif")
        assert np.array_equal(doc.plane(), arr)


# ---------------------------------------------------------------------------------- io_write
def test_tiff_read_matches_imagej_metadata():
    meta = {r["file"]: r for r in rows(W, "meta.csv")}
    for name, m in meta.items():
        doc = read_tiff(gw(name))
        assert (doc.width, doc.height) == (int(m["width"]), int(m["height"]))
        assert doc.dims == (int(m["channels"]), int(m["slices"]), int(m["frames"]))
        assert doc.bit_depth == int(m["bitDepth"])
        cal = doc.calibration
        assert cal.pixel_width == pytest.approx(float(m["pw"]), rel=1e-6)
        assert cal.pixel_height == pytest.approx(float(m["ph"]), rel=1e-6)
        assert cal.units == m["unit"] or (cal.unit == m["unit"])
        if not doc.is_rgb:
            assert doc.display_ranges[0] == (float(m["dmin"]), float(m["dmax"]))
        if name != "hyperstack.tiff" and not doc.is_rgb:   # ImageJ's mean is at its position
            assert doc.plane().mean() == pytest.approx(float(m["mean"]), rel=1e-7)
        assert cal.frame_interval == pytest.approx(float(m["finterval"]))
    blobs = read_tiff(gw("blobs_cal.tiff"))
    assert blobs.is_inverted_lut() and blobs.type == GRAY8
    assert blobs.info == "blobs info line 1\nline 2" and blobs.properties == {"Sample": "blobs"}
    hs = read_tiff(gw("hyperstack.tiff"))
    assert hs.composite_mode == "composite" and hs.hyperstack
    assert hs.display_ranges == [(100.0, 50000.0), (2000.0, 60000.0)]
    assert hs.slice_labels[:3] == ["c1z1t1", "c2z1t1", "c1z2t1"]
    assert tuple(hs.luts[0].as_array()[255]) == (255, 0, 255)      # Magenta
    assert tuple(hs.luts[1].as_array()[255]) == (0, 255, 255)      # Cyan
    assert hs.calibration.pixel_depth == 2.0 and hs.calibration.fps == pytest.approx(2 / 3)
    for t in range(4):
        for z in range(3):
            for c in range(2):
                assert hs.pixels[t, z, c, 0, 0] == (c + 1) * 1000 + (z + 1) * 100 + (t + 1) * 10
    ov = read_tiff(gw("overlay.tiff")).overlay
    assert [type(r).__name__ for r in ov] == ["RectRoi", "OvalRoi", "LineRoi"]
    assert [r.stroke_color for r in ov] == ["#00ff00", "#ff00ff", "#ffff00"]
    assert ov.rois[1].stroke_width == 2


@pytest.mark.parametrize("name", ["blobs_cal.tiff", "synth32_out.tiff", "hyperstack.tiff",
                                  "rgb.tiff", "stack8.tiff", "overlay.tiff"])
def test_tiff_writer_is_byte_identical_to_imagej(name):
    raw = gw(name).read_bytes()
    doc = read_tiff(gw(name))
    assert tiff_bytes(doc, little_endian=raw[:2] == b"II") == raw


def test_tiff_writer_16bit_calibrated_matches_except_rounded_resolution():
    raw = gw("synth16_cal.tiff").read_bytes()
    doc = read_tiff(gw("synth16_cal.tiff"))
    doc.calibration.pixel_height = 0.3       # ImageJ's in-memory value (file stores 1/0.3)
    assert tiff_bytes(doc, little_endian=raw[:2] == b"II") == raw


def test_zip_and_other_formats_read_like_imagej():
    blobs = read_tiff(gw("blobs_cal.tiff")).plane()
    z = open_image(gw("blobs_cal.zip"))
    assert np.array_equal(z.plane(), blobs) and z.calibration.pixel_width == 0.25
    for name in ("blobs_out.gif", "blobs_out.png", "blobs_out.bmp"):
        doc = open_image(gw(name))
        assert doc.type == GRAY8 and doc.is_inverted_lut() and np.array_equal(doc.plane(), blobs)
    pgm = open_image(gw("blobs_out.pgm"))
    assert np.array_equal(pgm.plane(), 255 - blobs)       # PGM holds the displayed values
    s16 = read_tiff(gw("synth16_cal.tiff")).plane()
    assert np.array_equal(open_image(gw("synth16_out.png")).plane(), s16)
    rgb = read_tiff(gw("rgb.tiff")).plane()
    for name in ("rgb_out.png", "rgb_out.bmp", "rgb.zip"):
        doc = open_image(gw(name))
        assert doc.type == COLOR_RGB and np.array_equal(doc.plane(), rgb)
    jpg = open_image(gw("blobs_out.jpg"))
    assert jpg.type == GRAY8 and np.abs(jpg.plane().astype(int) - (255 - blobs)).mean() < 3
    stack = read_tiff(gw("stack8.tiff"))
    for name in ("stack8_raw.avi", "stack8_png.avi"):
        avi = open_image(gw(name))
        assert avi.n_slices == 5 and np.array_equal(avi.pixels, stack.pixels)
        assert avi.slice_labels[0] == "0.14 s"


def test_fits_raw_text_writers_match_imagej_bytes():
    from rsstatchem.core.imaging.io.fits import fits_bytes, read_fits
    from rsstatchem.core.imaging.io.raw import RawOptions, raw_bytes, read_raw
    from rsstatchem.core.imaging.io.text_image import read_text_image, text_image_bytes

    for src, stem in (("blobs_cal.tiff", "blobs_out"), ("synth16_cal.tiff", "synth16_out"),
                      ("synth32_out.tiff", "synth32_out"), ("rgb.tiff", "rgb_out")):
        doc = read_tiff(gw(src))
        assert raw_bytes(doc) == gw(stem + ".raw").read_bytes()
        if doc.is_rgb:
            continue
        assert fits_bytes(doc) == gw(stem + ".fits").read_bytes()
        assert text_image_bytes(doc) == gw(stem + ".txt").read_bytes()
        back = read_fits(gw(stem + ".fits"))
        assert np.array_equal(back.plane(), doc.plane()) and back.bit_depth == doc.bit_depth
        txt = read_text_image(gw(stem + ".txt"))
        assert txt.type == GRAY32 and np.allclose(txt.plane(), doc.plane(), atol=5e-4)
    raw = read_raw(gw("synth16_out.raw"), RawOptions("16-bit Unsigned", 64, 64))
    assert np.array_equal(raw.plane(), read_tiff(gw("synth16_cal.tiff")).plane())


def test_lut_and_xy_files():
    from rsstatchem.core.imaging.io.lut_io import lut_bytes, read_lut
    from rsstatchem.core.imaging.io.text_image import xy_text

    lut = read_lut(gw("fire.lut"))
    expected = np.array([[int(r["r"]), int(r["g"]), int(r["b"])] for r in rows(W, "luts.csv")])
    assert np.array_equal(lut.as_array(), expected)
    assert lut_bytes(lut) == gw("fire.lut").read_bytes()
    poly = S.PolygonRoi([10, 50, 40, 12], [10, 14, 55, 40])
    assert xy_text(poly) == gw("polygon_xy.txt").read_text()


def test_roi_files_decode_like_imagej_and_encode_identically():
    golden = rows(W, "rois.csv")
    exact = {"rect", "roundrect", "oval", "polygon", "subpixel", "freehand", "line",
             "polyline", "freeline", "angle", "point", "multipoint", "composite", "text"}
    kinds = {"rect": S.RectRoi, "roundrect": S.RectRoi, "oval": S.OvalRoi, "line": S.LineRoi,
             "arrow": S.LineRoi, "point": S.PointRoi, "multipoint": S.PointRoi,
             "ellipse": S.EllipseRoi, "text": S.TextRoi, "rotrect": S.RotatedRectRoi}
    for row in golden:
        raw = gw(row["name"] + ".roi").read_bytes()
        roi = roi_codec.decode_roi(raw)
        assert roi.name == row["name"]
        assert isinstance(roi, kinds.get(row["name"], S.PolygonRoi))
        if row["name"] not in ("oval", "roundrect", "ellipse", "text", "arrow", "line",
                               "rotrect", "composite"):
            xs, ys = roi.polygon()
            assert np.allclose(xs, [float(v) for v in row["xs"].split()], atol=1e-5)
            assert np.allclose(ys, [float(v) for v in row["ys"].split()], atol=1e-5)
        if row["name"] in exact:
            assert roi_codec.encode_roi(roi) == raw, row["name"]
        else:
            again = roi_codec.decode_roi(roi_codec.encode_roi(roi))
            assert type(again) is type(roi) and again.bounds() == pytest.approx(roi.bounds())
    assert roi_codec.decode_roi(gw("rect.roi").read_bytes()).stroke_color == "#ff0000"
    assert roi_codec.decode_roi(gw("line.roi").read_bytes()).stroke_width == 3
    rois = roi_codec.read_roi_zip(gw("RoiSet.zip"))
    assert [r.name for r in rois] == [r["name"] for r in golden]
    opened = open_file(gw("RoiSet.zip"))
    assert opened.kind == "rois" and len(opened.value) == len(golden)


# ---------------------------------------------------------------------------------- io_read
def test_committed_inputs_equal_current_writers():
    """The files ImageJ read in the io_read golden are exactly what the writers produce."""
    files = _make_inputs().build()
    committed = {p.name for p in (input_path("io")).iterdir()}
    assert set(files) == committed
    for name, data in files.items():
        assert (input_path("io") / name).read_bytes() == data, name


def test_imagej_reads_our_files():
    meta = {r["file"]: r for r in rows(R, "read_meta.csv")}
    imgs = golden_images(R)
    for name, m in meta.items():
        key = "read_" + name.replace(".", "_")
        ours = None if name.endswith((".txt", ".raw")) else open_image(input_path("io") / name)
        if ours is not None:
            assert (ours.width, ours.height) == (int(m["width"]), int(m["height"])), name
            assert ours.dims == (int(m["channels"]), int(m["slices"]), int(m["frames"])), name
            assert ours.bit_depth == int(m["bitDepth"]), name
            arr = imgs[key]
            if ours.is_rgb:
                assert np.array_equal(ours.pixels.reshape(arr.shape), arr), name
            elif ours.stack_size > 1:
                assert np.array_equal(ours.pixels.reshape(arr.shape), arr), name
            else:
                assert np.array_equal(ours.plane(), arr, equal_nan=True), name
            assert ours.calibration.pixel_width == pytest.approx(float(m["pw"])), name
            assert bool(ours.is_inverted_lut()) == (m["inverting"] == "1"), name
            if not ours.is_rgb and ours.n_channels == 1:
                cal = ours.calibration
                lo, hi = ours.display_range
                assert cal.cvalue(lo) == pytest.approx(float(m["dmin"]), abs=1e-6), name
                assert cal.cvalue(hi) == pytest.approx(float(m["dmax"]), abs=1e-6), name
                assert cal.cvalue(100) == pytest.approx(float(m["cal100"])), name
    blobs = meta["rs_blobs.tif"]
    assert blobs["overlay"] == "3" and blobs["seltype"] == "2" and blobs["prop"] == "blobs"
    assert blobs["unit"] == "µm" and blobs["pw"].startswith("0.25")
    assert meta["rs_hs.tif"]["mode"] == "composite" and meta["rs_hs.tif"]["finterval"] == \
        "1.500000000"
    assert meta["rs_s16cal.tif"]["cal100"] == "-50.000000000"      # -100 + 0.5 * 100
    assert meta["rs_signed.tif"]["cal100"] == "-32668.000000000"   # signed 16-bit
    assert meta["rs_s16.dcm"]["dicom_spacing"].strip() == "0.5\\0.5"
    assert meta["rs_s16.dcm"]["cal100"] == "-924.000000000"
    s32 = imgs["read_rs_32_txt"]
    assert np.allclose(s32, open_image(input_path("synth32.tif")).plane(), atol=5e-4)


def test_imagej_reads_our_rois():
    golden = {r["name"]: r for r in rows(R, "read_rois.csv")}
    rois = _make_inputs().rois
    for name, roi in rois().items():
        g = golden[name]
        assert g["roiname"] == name
        xs, ys = roi.polygon()
        if isinstance(roi, (S.RectRoi, S.PolygonRoi, S.PointRoi)) and \
                not isinstance(roi, S.TextRoi) and not getattr(roi, "corner_diameter", 0):
            assert np.allclose(sorted(xs), sorted(float(v) for v in g["xs"].split()), atol=1e-4)
            assert np.allclose(sorted(ys), sorted(float(v) for v in g["ys"].split()), atol=1e-4)
        if roi.is_area() and name not in ("subpixel_oval",):
            mask_area = roi.get_mask((80, 100)).sum()
            # rectangles/composites exactly; ovals/polygons depend on WP-C's mask port
            if isinstance(roi, (S.RectRoi, S.CompositeRoi)) and \
                    not getattr(roi, "corner_diameter", 0):
                assert mask_area == float(g["area"]), name
    assert golden["rect"]["stroke"] == "red" and golden["polygon"]["fill"] == "#4000ff00"
    assert golden["line"]["width"].startswith("3") and golden["oval"]["width"].startswith("2")
    assert (golden["positioned"]["c"], golden["positioned"]["z"], golden["positioned"]["t"]) \
        == ("2", "3", "1")
    assert golden["composite"]["type"] == "9" and golden["composite"]["area"].startswith("1500")
    assert golden["text"]["type"] == "0" and golden["ellipse"]["type"] == "3"
    rs = rows(R, "read_set.csv")
    assert [r["name"] for r in rs] == list(rois())


def test_imagej_reads_our_lut():
    from rsstatchem.core.imaging.io.lut_io import read_lut
    lut = read_lut(input_path("io") / "rs.lut")
    expected = np.array([[int(r["r"]), int(r["g"]), int(r["b"])] for r in rows(R, "read_lut.csv")])
    assert np.array_equal(lut.as_array(), expected)


# ---------------------------------------------------------------------------------- misc
def test_save_image_dispatch_and_round_trips(tmp_path):
    from rsstatchem.core.imaging.lut import LUT
    rng = np.random.default_rng(5)
    g8 = ImageDoc.from_array(rng.integers(0, 255, (20, 30)).astype(np.uint8), title="g8")
    g8.luts = [LUT.grays().inverted()]
    for fmt, ext in (("Tiff", ".tif"), ("ZIP", ".zip"), ("PNG", ".png"), ("Gif", ".gif"),
                     ("BMP", ".bmp")):
        p = save_image(g8, tmp_path / ("a" + ext), fmt)
        back = open_image(p)
        assert np.array_equal(back.plane(), g8.plane()) and back.is_inverted_lut(), fmt
    hs = ImageDoc.from_array(rng.integers(0, 4000, (2, 3, 2, 8, 9)).astype(np.uint16), "TZCYX")
    hs.calibration.set_scale(2, 1, "mm")
    hs.roi = S.OvalRoi(1, 1, 4, 5)
    p = save_image(hs, tmp_path / "hs", "tiff")
    back = open_image(p)
    assert back.dims == hs.dims and np.array_equal(back.pixels, hs.pixels)
    assert back.calibration.unit == "mm" and isinstance(back.roi, S.OvalRoi)
    le = save_image(hs, tmp_path / "le.tif", "tiff", intel=True)
    assert le.read_bytes()[:2] == b"II" and np.array_equal(open_image(le).pixels, hs.pixels)
    rgb = ImageDoc.from_array(rng.integers(0, 255, (3, 6, 7, 3)).astype(np.uint8), "ZYXS")
    for fmt in ("tiff", "avi"):
        back = open_image(save_image(rgb, tmp_path / f"r.{fmt[:3]}", fmt))
        assert np.array_equal(back.pixels, rgb.pixels), fmt
    color = ImageDoc.from_array(np.arange(16, dtype=np.uint8).reshape(4, 4), title="c")
    color.luts = [LUT.from_array(np.stack([np.arange(256), np.zeros(256), np.arange(256)[::-1]],
                                          axis=1))]
    color.type = COLOR_256
    back = open_image(save_image(color, tmp_path / "c.tif", "tiff"))
    assert back.type == COLOR_256 and back.lut == color.lut
    assert sniff(tmp_path / "c.tif") == ".tif" and sniff(tmp_path / "a.png") == ".png"
    with pytest.raises(ValueError):
        save_image(color, tmp_path / "x.qqq", "qqq")


def test_dicom_reader_like_imagej(tmp_path):
    pytest.importorskip("pydicom")
    from rsstatchem.core.imaging.io.dicom import read_dicom, write_dicom
    a = (np.arange(30 * 20).reshape(30, 20) * 7).astype(np.uint16)
    doc = ImageDoc.from_array(a)
    doc.calibration.set_scale(4, 1, "mm")
    write_dicom(doc, tmp_path / "x.dcm")
    back = read_dicom(tmp_path / "x.dcm")
    assert np.array_equal(back.plane(), a) and back.calibration.unit == "mm"
    assert back.calibration.pixel_width == 0.25
    assert "0028,0010  Rows: 30" in back.info
    stack = ImageDoc.from_array(np.zeros((3, 5, 6), np.uint8), "ZYX")
    write_dicom(stack, tmp_path / "s.dcm")
    assert open_image(tmp_path / "s.dcm").n_slices == 3


def test_image_sequence_and_raw(tmp_path):
    from rsstatchem.core.imaging.io.raw import RAW_TYPES, RawOptions, read_raw
    from rsstatchem.core.imaging.io.sequence import open_sequence, save_sequence
    rng = np.random.default_rng(7)
    stack = ImageDoc.from_array(rng.integers(0, 255, (4, 9, 11)).astype(np.uint8), "ZYX", "s")
    files = save_sequence(stack, tmp_path / "seq", "TIFF", name="img", digits=2)
    assert [f.name for f in files] == ["img00.tif", "img01.tif", "img02.tif", "img03.tif"]
    (tmp_path / "seq" / "img10.tif").write_bytes(files[0].read_bytes())
    seq = open_sequence(tmp_path / "seq", sort_numerically=True)
    assert seq.n_slices == 5 and seq.slice_labels[-1] == "img10.tif"
    assert np.array_equal(seq.pixels[0, :4, 0], stack.pixels[0, :, 0])
    sub = open_sequence(tmp_path / "seq", start=2, step=2, count=2)
    assert sub.slice_labels == ["img01.tif", "img03.tif"]
    assert open_sequence(tmp_path / "seq", file_filter="(img0[12])").n_slices == 2
    data = (np.arange(12).reshape(3, 4) - 5).astype(">i2")
    (tmp_path / "x.raw").write_bytes(b"\0" * 10 + data.tobytes())
    doc = read_raw(tmp_path / "x.raw", RawOptions("16-bit Signed", 4, 3, offset=10))
    assert doc.calibration.is_signed16() and doc.value(0, 0) == -5.0
    assert len(RAW_TYPES) == 14
    bits = np.packbits(np.array([[1, 0, 1, 1, 0, 0, 0, 0, 1]], np.uint8), axis=1)
    (tmp_path / "b.raw").write_bytes(bits.tobytes())
    b = read_raw(tmp_path / "b.raw", RawOptions("1-bit Bitmap", 9, 1))
    assert b.plane().tolist() == [[255, 0, 255, 255, 0, 0, 0, 0, 255]]


def test_open_file_kinds(tmp_path):
    from rsstatchem.core.imaging.results import ResultsTable
    rt = ResultsTable.results()
    rt.add_row({"Area": 3, "Mean": 1.5})
    rt.save(tmp_path / "Results.csv")
    assert open_file(tmp_path / "Results.csv").kind == "table"
    (tmp_path / "m.ijm").write_text('print("hi");')
    assert open_file(tmp_path / "m.ijm").kind == "macro"
    (tmp_path / "notes.txt").write_text("hello world\nsecond line\n")
    assert open_file(tmp_path / "notes.txt").kind == "text"
    (tmp_path / "img.txt").write_text("1 2 3\n4 5 6\n")
    opened = open_file(tmp_path / "img.txt")
    assert opened.kind == "image" and opened.value.plane().tolist() == [[1, 2, 3], [4, 5, 6]]
    roi = S.RectRoi(1, 2, 3, 4, name="r")
    roi_codec.write_roi(roi, tmp_path / "r.roi")
    assert open_file(tmp_path / "r.roi").kind == "roi"
    zf = zipfile.ZipFile(io.BytesIO(), "w")
    zf.close()


def test_project_round_trip(tmp_path):
    from rsstatchem.core.imaging.project import load_project, read_project, save_project
    from rsstatchem.core.imaging.session import Session
    s = Session()
    a = ImageDoc.from_array(np.arange(12, dtype=np.uint8).reshape(3, 4), title="a.tif")
    a.roi = S.RectRoi(0, 0, 2, 2)
    b = ImageDoc.from_array(np.zeros((2, 3, 4), np.float32), "CYX", title="b")
    b.composite_mode = "color"
    b.set_position(1, 0, 0)
    s.add_image(a)
    s.add_image(b)
    s.set_active(a)
    s.roi_manager.extend([S.OvalRoi(1, 1, 2, 2, name="o"), S.PointRoi([1], [2], name="p")])
    s.results.add_row({"Area": 1.0 / 3, "Mean": 7})
    s.log("hello")
    p = save_project(s, tmp_path / "proj", window_state={a.id: {"zoom": 2.0}})
    assert p.suffix == ".rsij"
    with zipfile.ZipFile(p) as zf:
        names = zf.namelist()
    assert "manifest.json" in names and "rois/RoiSet.zip" in names and "log.txt" in names
    assert any(n.startswith("images/01_a") for n in names)
    data = read_project(p)
    assert [d.title for d in data["images"]] == ["a.tif", "b"]
    assert data["window_state"][0] == {"zoom": 2.0} and data["active"] == 0
    assert data["images"][1].composite_mode == "color" and data["images"][1].position[0] == 1
    s2 = Session()
    load_project(s2, p)
    assert s2.active.title == "a.tif" and isinstance(s2.active.roi, S.RectRoi)
    assert [r.name for r in s2.roi_manager] == ["o", "p"]
    assert s2.results.get_value("Area", 0) == pytest.approx(1 / 3, rel=1e-12)
    assert s2.log_lines == ["hello"]
