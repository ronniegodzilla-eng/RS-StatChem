"""Write ``tests/data_imagej/inputs/io/``: files produced by RS-StatChem's own writers, which
the golden case ``cases/io_read.ijm`` opens with the real ImageJ 1.54p (scripts/
imagej_golden.py --case io_read) so tests can check ImageJ reads what we write.

    python tests/data_imagej/make_io_inputs.py

``tests/test_imagej_io.py`` regenerates every file in memory and requires it to be identical
to the committed one, so the goldens always describe the current writers.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = HERE / "inputs" / "io"


def _hyperstack():
    from rsstatchem.core.imaging.image import ImageDoc
    from rsstatchem.core.imaging.lut import named_lut

    t, z, c, h, w = 4, 3, 2, 30, 40
    ramp = (np.arange(w, dtype=np.uint16) * 1600)[None, :].repeat(h, 0)
    a = np.zeros((t, z, c, h, w), np.uint16)
    for ti in range(t):
        for zi in range(z):
            for ci in range(c):
                a[ti, zi, ci] = ramp + (ci + 1) * 1000 + (zi + 1) * 100 + (ti + 1) * 10
    doc = ImageDoc.from_array(a, "TZCYX", "rs_hs.tif")
    doc.luts = [named_lut("Magenta"), named_lut("Cyan")]
    doc.display_ranges = [(100.0, 50000.0), (2000.0, 60000.0)]
    doc.composite_mode = "composite"
    doc.hyperstack = True
    for i in range(doc.stack_size):
        ci, zi, ti = doc.position_of(i)
        doc.slice_labels[i] = f"c{ci + 1}z{zi + 1}t{ti + 1}"
    cal = doc.calibration
    cal.pixel_width = cal.pixel_height = 0.5
    cal.pixel_depth = 2.0
    cal.unit = "µm"
    cal.frame_interval = 1.5
    cal.fps = 1 / 1.5
    doc.set_position(1, 1, 2)
    return doc


def rois():
    from rsstatchem.core.imaging.roi import (CompositeRoi, EllipseRoi, LineRoi, OvalRoi,
                                             PointRoi, PolygonRoi, RectRoi, RotatedRectRoi,
                                             TextRoi)
    from rsstatchem.core.imaging.roi import shapes as S

    out = {
        "rect": RectRoi(10, 12, 30, 20, name="rect", stroke_color="#ff0000"),
        "roundrect": RectRoi(12, 14, 40, 30, corner_diameter=10, name="roundrect"),
        "oval": OvalRoi(20, 15, 40, 25, stroke_width=2, name="oval"),
        "subpixel_oval": OvalRoi(20.5, 15.25, 30.5, 20.75, name="subpixel_oval"),
        "polygon": PolygonRoi([10, 50, 40, 12], [10, 14, 55, 40], name="polygon",
                              fill_color="#4000ff00"),
        "subpixel": PolygonRoi([2.5, 40.25, 20.5], [3.5, 10.75, 50.5], name="subpixel"),
        "freehand": PolygonRoi([10, 12, 15, 20, 18, 11], [10, 8, 9, 15, 22, 18],
                               type=S.FREEROI, name="freehand"),
        "traced": PolygonRoi([5, 25, 25, 15, 15, 5], [5, 5, 30, 30, 15, 15],
                             type=S.TRACED_ROI, name="traced"),
        "line": LineRoi(5, 5, 60, 40, width=3, stroke_width=3, name="line"),
        "arrow": LineRoi(10, 10, 50, 50, arrow=True, stroke_width=2, name="arrow"),
        "polyline": PolygonRoi([5, 20, 50], [5, 40, 10], type=S.POLYLINE, name="polyline"),
        "freeline": PolygonRoi([5, 8, 12, 18, 25], [30, 28, 31, 35, 33], type=S.FREELINE,
                               name="freeline"),
        "angle": PolygonRoi([10, 30, 50], [40, 10, 40], type=S.ANGLE, name="angle"),
        "point": PointRoi([12], [13], name="point"),
        "multipoint": PointRoi([5, 25, 45, 65], [6, 26, 46, 66], counters=[0, 1, 1, 2],
                               name="multipoint"),
        "rotrect": RotatedRectRoi(10, 20, 60, 50, rect_width=15, name="rotrect"),
        "ellipse": EllipseRoi(20, 30, 70, 50, aspect_ratio=0.5, name="ellipse"),
        "composite": CompositeRoi(parts=[("or", RectRoi(10, 10, 40, 40)),
                                         ("not", RectRoi(20, 20, 10, 10))], name="composite"),
        "text": TextRoi(x=10, y=10, text="Hello", font="SansSerif", font_size=18,
                        name="text"),
        "positioned": RectRoi(5, 6, 7, 8, name="positioned", position=(2, 3, 1)),
    }
    return out


def build() -> dict[str, bytes]:
    """name -> file bytes (deterministic)."""
    import io
    import tempfile

    from rsstatchem.core.imaging.image import ImageDoc
    from rsstatchem.core.imaging.io import open_image
    from rsstatchem.core.imaging.io.avi import avi_bytes
    from rsstatchem.core.imaging.io.fits import fits_bytes
    from rsstatchem.core.imaging.io.lut_io import lut_bytes
    from rsstatchem.core.imaging.io.raw import raw_bytes
    from rsstatchem.core.imaging.io.roi_codec import encode_roi, write_roi_zip
    from rsstatchem.core.imaging.io.text_image import text_image_bytes
    from rsstatchem.core.imaging.io.tiff import tiff_bytes
    from rsstatchem.core.imaging.lut import LUT
    from rsstatchem.core.imaging.roi import LineRoi, OvalRoi, Overlay, PolygonRoi, RectRoi

    inputs = HERE / "inputs"
    files: dict[str, bytes] = {}
    blobs = open_image(inputs / "blobs.gif")
    blobs.title = "rs_blobs.tif"
    blobs.calibration.set_scale(4, 1, "µm")
    blobs.info = "rs info line 1\nline 2"
    blobs.properties = {"Sample": "blobs"}
    blobs.roi = PolygonRoi([10, 50, 40, 12], [10, 14, 55, 40])
    blobs.overlay = Overlay([RectRoi(10, 12, 30, 20, stroke_color="#00ff00"),
                             OvalRoi(20, 15, 40, 25, stroke_color="#ff00ff", stroke_width=2),
                             LineRoi(5, 5, 60, 40, stroke_color="#ffff00")])
    files["rs_blobs.tif"] = tiff_bytes(blobs)
    files["rs_blobs_le.tif"] = tiff_bytes(blobs, little_endian=True)

    s16 = open_image(inputs / "synth16.tif")
    s16.title = "rs_s16cal.tif"
    s16.info = ""
    s16.calibration.function = 0
    s16.calibration.coefficients = (-100.0, 0.5)
    s16.calibration.value_unit = "OD"
    s16.set_display_range(2000, 30000)
    s16.slice_labels = ["plane one"]
    files["rs_s16cal.tif"] = tiff_bytes(s16)

    sgn = ImageDoc.from_array((np.arange(48 * 32, dtype=np.int32).reshape(32, 48) * 13 - 9000
                               ).astype(np.int16), "YX", "rs_signed.tif")
    files["rs_signed.tif"] = tiff_bytes(sgn)

    s32 = open_image(inputs / "synth32.tif")
    s32.title = "rs_32.tif"
    s32.info = ""
    s32.pixels[0, 0, 0, 0, :3] = (np.nan, np.inf, -np.inf)
    s32.set_display_range(-2, 3)
    files["rs_32.tif"] = tiff_bytes(s32)

    hs = _hyperstack()
    files["rs_hs.tif"] = tiff_bytes(hs)

    rgb = np.zeros((48, 64, 3), np.uint8)
    rgb[4:16, 4:24] = (255, 0, 0)
    rgb[10:30, 30:55] = (10, 200, 30)
    rgb[30:44, 8:38] = (20, 40, 250)
    rgbdoc = ImageDoc.from_array(rgb, "YXS", "rs_rgb.tif")
    files["rs_rgb.tif"] = tiff_bytes(rgbdoc)
    rgbstack = ImageDoc.from_array(np.stack([rgb, rgb[::-1], rgb[:, ::-1]]), "ZYXS",
                                   "rs_rgbstack.tif")
    files["rs_rgbstack.tif"] = tiff_bytes(rgbstack)

    stack = ImageDoc.from_array((np.arange(5)[:, None, None] * 20 +
                                 np.arange(48)[None, None, :] * 4 +
                                 np.zeros((5, 32, 1))).astype(np.uint8), "ZYX", "rs_stack.tif")
    files["rs_stack.tif"] = tiff_bytes(stack)
    files["rs_stack_raw.avi"] = avi_bytes(stack, "Uncompressed", 7.0)
    files["rs_rgb_png.avi"] = avi_bytes(rgbstack, "PNG", 5.0)

    files["rs_blobs.fits"] = fits_bytes(blobs)
    s16plain = open_image(inputs / "synth16.tif")
    files["rs_s16.fits"] = fits_bytes(s16plain)
    files["rs_32.fits"] = fits_bytes(open_image(inputs / "synth32.tif"))

    fire = LUT(np.arange(256)[::-1].astype(np.uint8), (np.arange(256) // 2).astype(np.uint8),
               np.full(256, 77, np.uint8))
    files["rs.lut"] = lut_bytes(fire)
    files["rs_32.txt"] = text_image_bytes(open_image(inputs / "synth32.tif"))
    files["rs_s16_be.raw"] = raw_bytes(s16plain)

    for name, roi in rois().items():
        files[f"rs_{name}.roi"] = encode_roi(roi)
    buf = io.BytesIO()
    write_roi_zip(list(rois().values()), buf)
    files["rs_RoiSet.zip"] = _normalise(".zip", buf.getvalue())

    with tempfile.TemporaryDirectory() as tmp:
        from rsstatchem.core.imaging.io import save_image
        from rsstatchem.core.imaging.io.dicom import write_dicom
        from rsstatchem.core.imaging.io.zip_io import write_zip

        tmp = Path(tmp)
        save_image(s16plain, tmp / "rs_s16.png", "png")
        save_image(blobs, tmp / "rs_blobs.png", "png")
        save_image(blobs, tmp / "rs_blobs.gif", "gif")
        save_image(stack, tmp / "rs_stack.gif", "gif")
        save_image(blobs, tmp / "rs_blobs.bmp", "bmp")
        save_image(blobs, tmp / "rs_blobs.pgm", "pgm")
        save_image(rgbdoc, tmp / "rs_rgb.png", "png")
        write_zip(blobs, tmp / "rs_blobs.zip")
        dcm = open_image(inputs / "synth16.tif")
        dcm.calibration.set_scale(2, 1, "mm")
        dcm.calibration.function = 0
        dcm.calibration.coefficients = (-1024.0, 1.0)
        write_dicom(dcm, tmp / "rs_s16.dcm", uid="1.2.826.0.1.3680043.8.498.20261028")
        for p in sorted(tmp.iterdir()):
            data = p.read_bytes()
            if p.suffix in (".dcm", ".zip"):
                data = _normalise(p.suffix, data)
            files[p.name] = data
    return files


def _normalise(suffix: str, data: bytes) -> bytes:
    """Make DICOM UIDs and zip timestamps and "made by" systems deterministic."""
    import io
    import re
    import zipfile

    if suffix == ".dcm":
        return data
    src = zipfile.ZipFile(io.BytesIO(data))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for info in src.infolist():
            zi = zipfile.ZipInfo(info.filename, date_time=(2026, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.create_system = 3  # ZipInfo writes 0 (MS-DOS) on Windows and 3 (Unix) elsewhere
            zf.writestr(zi, src.read(info))
    return out.getvalue()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    files = build()
    for p in OUT.iterdir():
        if p.name not in files:
            p.unlink()
    for name, data in files.items():
        (OUT / name).write_bytes(data)
    print(f"{len(files)} files written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
