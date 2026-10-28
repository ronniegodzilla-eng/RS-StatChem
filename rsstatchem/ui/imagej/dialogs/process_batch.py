"""Process > Batch: Measure..., Convert..., Macro... and Virtual Stack.... **WP-B.**

Behaviour follows ImageJ 1.54p ij/plugin/BatchProcessor.java, BatchConverter.java and
BatchMeasure.java (public domain, NIH): files of the input folder are processed in
``StringSorter`` order (numeric runs compared as numbers), names starting with "." and .avi
files are skipped, outputs keep the input name with the format's extension; "8-bit TIFF" and
"GIF" convert to 8-bit first. Headless option keys::

    run("Macro... ", "input=[/in] output=[/out] output_format=TIFF file=[] code=[run(\\"Invert\\");]")
    run("Convert...", "input=[/in] output=[/out] output_format=PNG interpolation=Bilinear scale=0.5 average")
    run("Measure...", "input=[/in]")
    run("Virtual Stack...", "output=[/out] output_format=TIFF code=[...]")

Integration points (used when present, otherwise the small fallbacks in this module):

* Measure uses the registered "Measure" command (WP-C) with "Display label"; the fallback
  computes Area/Mean/StdDev/Mode/Min/Max/IntDen/Median for the whole image.
  TODO(WP-C): remove the fallback once ``core/imaging/measure.py`` is merged.
* Saving uses ``rsstatchem.core.imaging.io.save_image(doc, path, format)`` when WP-A provides
  it, else tifffile/Pillow and minimal FITS/raw/text/zip writers below.
* Macros run through ``rsstatchem.core.imaging.macro.run_macro(session, code, image)`` (WP-A)
  when available, else a mini interpreter for ``run("Label", "options");``,
  ``setOption(...)``, ``print(...)`` and simple ``var=value;`` statements.
* Scaling uses ``rsstatchem.core.imaging.transform.resize`` (WP-D) when available, else Pillow.
"""
from __future__ import annotations

import io as _io
import os
import re
import zipfile
from pathlib import Path

import numpy as np

from rsstatchem.core.imaging import commands as C
from rsstatchem.core.imaging import math_ops as M
from rsstatchem.core.imaging.commands import CommandError
from rsstatchem.core.imaging.image import COLOR_RGB, GRAY8, GRAY16, GRAY32, ImageDoc

FORMATS = ("TIFF", "8-bit TIFF", "JPEG", "GIF", "PNG", "PGM", "BMP", "FITS", "Text Image",
           "ZIP", "Raw")
EXTENSIONS = {"TIFF": ".tif", "8-bit TIFF": ".tif", "JPEG": ".jpg", "GIF": ".gif",
              "PNG": ".png", "PGM": ".pgm", "BMP": ".bmp", "FITS": ".fits",
              "Text Image": ".txt", "ZIP": ".zip", "Raw": ".raw"}
INTERPOLATIONS = ("None", "Bilinear", "Bicubic")


# ---------------------------------------------------------------------------------- files
def sort_names(names: list[str]) -> list[str]:
    """``StringSorter.sortNumerically``-like order: digit runs compared as numbers."""
    def key(s):
        return [(0, int(t), t) if t.isdigit() else (1, 0, t.lower())
                for t in re.split(r"(\d+)", s) if t]
    return sorted(names, key=key)


def list_inputs(folder: str, contains: str = "") -> list[Path]:
    d = Path(folder)
    if not d.is_dir():
        raise CommandError(f"Input does not exist or is not a folder\n \n{folder}")
    out = []
    for name in sort_names(os.listdir(d)):
        p = d / name
        if p.is_dir() or name.startswith(".") or name.lower().endswith(".avi") or \
                name == "Thumbs.db":
            continue
        if contains and contains not in name:
            continue
        out.append(p)
    return out


def _open(path: Path) -> ImageDoc | None:
    from rsstatchem.core.imaging.io import open_image
    try:
        return open_image(path)
    except Exception:  # noqa: BLE001 - IJ.openImage returns null; the batch logs and continues
        return None


def output_path(folder: str, name: str, fmt: str) -> Path:
    return Path(folder) / (Path(name).stem + EXTENSIONS[fmt])


# ---------------------------------------------------------------------------------- writing
def to_8bit(doc: ImageDoc) -> ImageDoc:
    """Image > Type > 8-bit (scaled with the display range) / 8-bit Color for RGB."""
    if doc.type == GRAY8:
        return doc
    planes = []
    for k in range(doc.stack_size):
        p = doc.plane_at(k)
        if doc.is_rgb:
            from PIL import Image
            q = Image.fromarray(np.ascontiguousarray(p)).quantize(256)
            planes.append(np.asarray(q, dtype=np.uint8))
            lut = np.asarray(q.getpalette()[:768], np.uint8).reshape(-1, 3)
        else:
            c = doc.position_of(k)[0]
            planes.append(M.to_byte_scaled(p, doc.display_ranges[c]))
    out = ImageDoc(np.stack(planes).reshape(doc.pixels.shape[:5]), GRAY8, doc.title,
                   doc.calibration.copy())
    if doc.is_rgb:
        from rsstatchem.core.imaging.lut import LUT
        out.luts = [LUT.from_array(lut)]
    return out


def save_image(doc: ImageDoc, path, fmt: str = "TIFF") -> Path:
    """Save like ``IJ.saveAs(imp, format, path)`` (extension replaced by the format's)."""
    path = Path(path)
    path = path.with_suffix(EXTENSIONS.get(fmt, path.suffix))
    try:
        from rsstatchem.core.imaging import io as _ijio
        writer = getattr(_ijio, "save_image", None)
    except ImportError:  # pragma: no cover
        writer = None
    if writer is not None:
        writer(doc, str(path), fmt)
        return path
    if fmt in ("8-bit TIFF", "GIF"):
        doc = to_8bit(doc)
    if fmt in ("TIFF", "8-bit TIFF"):
        _write_tiff(doc, path)
    elif fmt == "ZIP":
        buf = _io.BytesIO()
        _write_tiff(doc, buf)
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(path.stem + ".tif", buf.getvalue())
    elif fmt == "Raw":
        data = doc.pixels
        if data.dtype != np.uint8:
            data = data.astype(data.dtype.newbyteorder(">"))
        path.write_bytes(np.ascontiguousarray(data).tobytes())
    elif fmt == "Text Image":
        p = doc.plane()
        if doc.is_rgb:
            p = M.rgb_to_gray(p)
        fmt_str = "%.4f" if p.dtype == np.float32 else "%d"
        np.savetxt(path, p, fmt=fmt_str, delimiter="\t")
    elif fmt == "FITS":
        _write_fits(doc, path)
    else:
        _write_pillow(doc, path, fmt)
    return path


def _write_tiff(doc: ImageDoc, target) -> None:
    import tifffile
    data = doc.pixels
    if doc.is_rgb:
        arr = data.reshape((-1,) + data.shape[-3:])
        arr = arr[0] if arr.shape[0] == 1 else arr
        tifffile.imwrite(target, arr, photometric="rgb")
        return
    arr = data
    axes = "TZCYX"
    lo, hi = doc.display_range
    tifffile.imwrite(target, arr, imagej=True,
                     metadata={"axes": axes, "min": float(lo), "max": float(hi)})


def _write_pillow(doc: ImageDoc, path: Path, fmt: str) -> None:
    from PIL import Image
    p = doc.plane()
    if doc.is_rgb:
        im = Image.fromarray(np.ascontiguousarray(p), "RGB")
    elif fmt == "PNG" and p.dtype == np.uint16:
        im = Image.fromarray(p.astype(np.uint16))
    else:
        b = M.to_byte_scaled(p, doc.display_range)
        if doc.type == GRAY8 and not doc.lut.is_grayscale():
            im = Image.fromarray(b, "P")
            im.putpalette(doc.lut.as_array().ravel().tolist())
            if fmt in ("JPEG", "PGM"):
                im = im.convert("RGB" if fmt == "JPEG" else "L")
        else:
            im = Image.fromarray(b, "L")
    kw = {"quality": 85} if fmt == "JPEG" else {}
    im.save(path, {"PGM": "PPM", "JPEG": "JPEG", "GIF": "GIF", "PNG": "PNG",
                   "BMP": "BMP"}[fmt], **kw)


def _write_fits(doc: ImageDoc, path: Path) -> None:
    p = doc.plane()
    if doc.is_rgb:
        p = M.rgb_to_gray(p)
    bitpix = {np.dtype(np.uint8): 8, np.dtype(np.uint16): 16,
              np.dtype(np.float32): -32}[p.dtype]
    cards = [f"SIMPLE  = {'T':>20}", f"BITPIX  = {bitpix:>20}", f"NAXIS   = {2:>20}",
             f"NAXIS1  = {p.shape[1]:>20}", f"NAXIS2  = {p.shape[0]:>20}"]
    data = p[::-1]   # FITS rows bottom-up (ImageJ flips)
    if bitpix == 16:
        cards += [f"BZERO   = {32768:>20}", f"BSCALE  = {1:>20}"]
        data = (data.astype(np.int32) - 32768).astype(">i2")
    elif bitpix == -32:
        data = data.astype(">f4")
    cards.append("END")
    header = "".join(c.ljust(80) for c in cards).encode("ascii")
    header += b" " * (-len(header) % 2880)
    body = np.ascontiguousarray(data).tobytes()
    body += b"\0" * (-len(body) % 2880)
    path.write_bytes(header + body)


def resize(doc: ImageDoc, width: int, height: int, interpolation: str = "Bilinear",
           average: bool = False) -> ImageDoc:
    """``ip.resize(w, h, averageWhenDownsizing)`` for each plane (WP-D's port if present)."""
    try:
        from rsstatchem.core.imaging.transform import resize as _resize  # type: ignore
        return _resize(doc, width, height, interpolation, average)
    except (ImportError, TypeError):
        pass
    from PIL import Image
    method = {"None": Image.Resampling.NEAREST, "Bilinear": Image.Resampling.BILINEAR,
              "Bicubic": Image.Resampling.BICUBIC}[interpolation]
    if average and (width < doc.width or height < doc.height) and interpolation != "None":
        method = Image.Resampling.BOX
    planes = []
    for k in range(doc.stack_size):
        p = doc.plane_at(k)
        if doc.is_rgb:
            r = np.asarray(Image.fromarray(np.ascontiguousarray(p)).resize((width, height),
                                                                          method))
        else:
            f = Image.fromarray(p.astype(np.float32), "F").resize((width, height), method)
            r = np.asarray(f)
            if p.dtype != np.float32:
                hi = 255 if p.dtype == np.uint8 else 65535
                r = np.clip(np.floor(r + 0.5), 0, hi)
            r = r.astype(p.dtype)
        planes.append(r)
    shape = doc.pixels.shape[:3] + planes[0].shape
    out = ImageDoc(np.stack(planes).reshape(shape), doc.type, doc.title,
                   doc.calibration.copy())
    out.luts = [lut.copy() for lut in doc.luts]
    return out


# ---------------------------------------------------------------------------------- macros
_STMT = re.compile(r"""\s*(?:
    (?P<run>run\s*\(\s*(?P<q1>"(?:[^"\\]|\\.)*")\s*(?:,\s*(?P<q2>"(?:[^"\\]|\\.)*")\s*)?\))
  | (?P<opt>setOption\s*\(\s*"(?P<oname>[^"]*)"\s*,\s*(?P<oval>true|false)\s*\))
  | (?P<print>print\s*\((?P<parg>.*?)\))
  | (?P<assign>[A-Za-z_]\w*\s*=\s*[^;]*)
  | (?P<blank>)
)\s*$""", re.VERBOSE | re.DOTALL)


def _unquote(s: str) -> str:
    return s[1:-1].encode("utf-8").decode("unicode_escape") if s else ""


def _split_statements(code: str) -> list[str]:
    out, cur, quote = [], [], False
    i = 0
    while i < len(code):
        ch = code[i]
        if ch == "\\" and quote and i + 1 < len(code):
            cur.append(code[i:i + 2])
            i += 2
            continue
        if ch == '"':
            quote = not quote
        if (ch in ";\n") and not quote:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
        i += 1
    out.append("".join(cur))
    return [s for s in (t.strip() for t in out) if s and not s.startswith("//")]


def run_macro(session, code: str, image: ImageDoc):
    """Run batch macro ``code`` on ``image``; returns the output image (the last new image a
    command returned, like ``Interpreter.runBatchMacro``) or ``image``."""
    try:
        from rsstatchem.core.imaging import macro as _macro  # type: ignore
        fn = getattr(_macro, "run_macro", None)
    except ImportError:
        fn = None
    if fn is not None:
        result = fn(session, code, image)
        return result if isinstance(result, ImageDoc) else image
    current = image
    for stmt in _split_statements(code):
        m = _STMT.match(stmt)
        if m is None:
            raise CommandError(f"Unsupported macro statement in batch mode:\n{stmt}")
        if m.group("run"):
            label = _unquote(m.group("q1"))
            opts = _unquote(m.group("q2")) if m.group("q2") else ""
            res = session.registry.run(session, label, opts, image=current, record=False)
            if isinstance(res, ImageDoc):
                current = res
                if res in session.images:
                    session.images.remove(res)
        elif m.group("opt"):
            if m.group("oname").lower() == "blackbackground":
                session.options.black_background = m.group("oval") == "true"
        elif m.group("print"):
            arg = m.group("parg").strip()
            session.log(_unquote(arg) if arg.startswith('"') else arg)
    return current


# ---------------------------------------------------------------------------------- measure
def _measure(session, doc: ImageDoc) -> None:
    if session.registry.get("Measure") is not None and \
            session.registry.get("Measure").func is not None:
        saved = session.options.measurements
        if "display" not in saved:
            session.options.measurements = tuple(saved) + ("display",)
        try:
            session.registry.run(session, "Measure", "", image=doc, record=False)
        finally:
            session.options.measurements = saved
        return
    # fallback (whole image; TODO(WP-C): use core/imaging/measure.py)
    p = doc.plane()
    vals = M.rgb_to_gray(p).astype(np.float64) if doc.is_rgb else p.astype(np.float64)
    if doc.type == GRAY32:
        vals = vals[np.isfinite(vals)]
    else:
        vals = doc.calibration.cvalue(vals)
    vals = np.asarray(vals, dtype=np.float64).ravel()
    cal = doc.calibration
    opts = session.options
    rt = session.results
    rt.increment_counter()
    rt.add_label(doc.title)
    n = vals.size
    if opts.measure("area"):
        rt.add_value("Area", n * cal.pixel_width * cal.pixel_height)
    if opts.measure("mean"):
        rt.add_value("Mean", float(vals.mean()) if n else float("nan"))
    if opts.measure("standard"):
        rt.add_value("StdDev", float(vals.std(ddof=1)) if n > 1 else 0.0)
    if opts.measure("modal"):
        u, cnt = np.unique(vals, return_counts=True)
        rt.add_value("Mode", float(u[np.argmax(cnt)]) if n else float("nan"))
    if opts.measure("min"):
        rt.add_value("Min", float(vals.min()) if n else float("nan"))
        rt.add_value("Max", float(vals.max()) if n else float("nan"))
    if opts.measure("integrated"):
        rt.add_value("IntDen", n * cal.pixel_width * cal.pixel_height * float(vals.mean()))
        rt.add_value("RawIntDen", float(vals.sum()))
    if opts.measure("median"):
        rt.add_value("Median", float(np.median(vals)) if n else float("nan"))


# ---------------------------------------------------------------------------------- commands
def batch_measure(session, image, input=""):
    if not input:
        raise CommandError("Please choose an input folder")
    for path in list_inputs(input):
        doc = _open(path)
        if doc is None:
            session.log(f"IJ.openImage() returned null: {path}")
            continue
        _measure(session, doc)
    session.show_table(session.results)
    return None


def batch_convert(session, image, input="", output="", output_format="TIFF",
                  interpolation="Bilinear", scale=1.0, average=False):
    if not input:
        raise CommandError("Please choose an input folder")
    if not output or not Path(output).is_dir():
        raise CommandError(f"Output does not exist or is not a folder\n \n{output}")
    files = list_inputs(input)
    for i, path in enumerate(files):
        doc = _open(path)
        if doc is None:
            session.log(f"IJ.openImage() returned null: {path}")
            continue
        if float(scale) != 1.0:
            w = int(float(scale) * doc.width)
            h = int(float(scale) * doc.height)
            doc = resize(doc, max(w, 1), max(h, 1), interpolation, bool(average))
        save_image(doc, output_path(output, path.name, output_format), output_format)
        session.progress((i + 1) / len(files))
    return None


def batch_macro(session, image, input="", output="", output_format="TIFF", file="", code=""):
    if not input:
        raise CommandError("Please choose an input folder")
    if output and not Path(output).is_dir():
        raise CommandError(f"Output does not exist or is not a folder\n \n{output}")
    if not code.strip():
        raise CommandError("There is no macro code in the text area")
    files = list_inputs(input, file)
    for i, path in enumerate(files):
        doc = _open(path)
        if doc is None:
            session.log(f"openImage() returned null: {path}")
            continue
        out = run_macro(session, f"i={i};" + code, doc)
        if output:
            save_image(out, output_path(output, path.name, output_format), output_format)
        session.progress((i + 1) / len(files))
    return None


def batch_virtual_stack(session, image, output="", output_format="TIFF", code=""):
    if image.stack_size == 1:
        raise CommandError("This command requires a stack or virtual stack.")
    if output and not Path(output).is_dir():
        raise CommandError(f"Output does not exist or is not a folder\n \n{output}")
    if not code.strip():
        raise CommandError("There is no macro code in the text area")
    n = image.stack_size
    results = []
    for k in range(n):
        plane = image.plane_at(k).copy()
        axes = "YXS" if image.is_rgb else "YX"
        tmp = ImageDoc.from_array(plane, axes, f"{k + 1}/{n}")
        tmp.calibration = image.calibration.copy()
        c = image.position_of(k)[0]
        tmp.luts = [image.luts[c].copy()] if not image.is_rgb else tmp.luts
        tmp.display_ranges = [image.display_ranges[c]]
        out = run_macro(session, f"i={k};n={n};" + code, tmp)
        if out is tmp and out.type == image.type and out.plane().shape == plane.shape:
            image.set_plane(out.plane(), *image.position_of(k))
        if output:
            save_image(out, Path(output) / f"{k + 1:05d}", output_format)
        results.append(out)
        session.progress((k + 1) / n)
    image.changed = True
    if image.type in (GRAY16, GRAY32):
        image.reset_display_range()
    session.image_updated(image)
    return None


def register(registry) -> None:
    fmt = C.choice("Output_format:", FORMATS, "TIFF")
    registry.add("Measure...", batch_measure, [C.string("Input:", "", columns=40)],
                 needs_image=False, undoable=False)
    registry.add("Convert...", batch_convert,
                 [C.string("Input:", "", columns=40), C.string("Output:", "", columns=40),
                  fmt, C.choice("Interpolation:", INTERPOLATIONS, "Bilinear"),
                  C.number("Scale factor:", 1.0, 2, key="scale"),
                  C.checkbox("Average when downsizing", False)],
                 needs_image=False, undoable=False)
    registry.add("Macro... ", batch_macro,
                 [C.string("Input:", "", columns=40), C.string("Output:", "", columns=40),
                  C.choice("Output_format:", FORMATS, "TIFF"),
                  C.string("File name contains:", "", key="file"),
                  C.string("Code:", "", columns=60)],
                 needs_image=False, undoable=False)
    registry.add("Virtual Stack...", batch_virtual_stack,
                 [C.string("Output:", "", columns=40),
                  C.choice("Output_format:", FORMATS, "TIFF"), C.string("Code:", "", columns=60)],
                 undoable=False, types=(GRAY8, GRAY16, GRAY32, COLOR_RGB))


def selftest() -> None:
    import tempfile

    from rsstatchem.core.imaging.session import Session

    assert sort_names(["a10.tif", "a2.tif", "a1.tif"]) == ["a1.tif", "a2.tif", "a10.tif"]
    stmts = _split_statements('run("Add...", "value=5"); print("a;b")\nx=1')
    assert stmts == ['run("Add...", "value=5")', 'print("a;b")', "x=1"]
    s = Session()
    from rsstatchem.ui.imagej.dialogs import process_math
    process_math.register(s.registry)
    register(s.registry)
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        doc = ImageDoc.from_array(np.full((4, 6), 9, np.uint8), "YX", "x.tif")
        save_image(doc, d / "x.tif", "TIFF")
        assert run_macro(s, 'run("Add...", "value=1");', doc) is doc
        assert doc.raw_value(0, 0) == 10
        s.run("Measure...", f"input=[{d}]")
        assert len(s.results) == 1
