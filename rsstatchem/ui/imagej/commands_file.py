"""Image Analysis commands: File menu (new, open, samples, import, show folder, close, save,
save as, revert, print, quit). **Owned by WP-A.**

Interactive use opens the usual file dialogs (``bar.get_path``); macros pass paths with the
keys ImageJ uses (``open=``, ``save=``, ``dir=``). The I/O itself lives in
:mod:`rsstatchem.core.imaging.io`. Open Samples downloads the sample images from
imagej.net on demand into a cache folder (nothing is bundled), as ImageJ does.
"""
from __future__ import annotations

import tempfile
import urllib.request
from pathlib import Path

import numpy as np

from rsstatchem.core.imaging import commands as C
from rsstatchem.core.imaging.commands import CommandError
from rsstatchem.core.imaging.image import COLOR_RGB, GRAY8, GRAY16, GRAY32, ImageDoc
from rsstatchem.core.imaging.menu_layout import iter_items

SAMPLE_URL = "https://imagej.net/ij/images/"
TYPES = {"8-bit": GRAY8, "16-bit": GRAY16, "32-bit": GRAY32, "RGB": COLOR_RGB}
FILLS = ("White", "Black", "Ramp", "Noise", "Random")
# File > Save As entries: label -> (ImageJ saveAs format name, extension, file filter)
SAVE_AS = {
    "Tiff...": ("Tiff", ".tif", "TIFF (*.tif *.tiff)"),
    "Gif...": ("Gif", ".gif", "GIF (*.gif)"),
    "Jpeg...": ("Jpeg", ".jpg", "JPEG (*.jpg *.jpeg)"),
    "Text Image...": ("Text Image", ".txt", "Text (*.txt)"),
    "ZIP...": ("ZIP", ".zip", "ZIP (*.zip)"),
    "Raw Data...": ("Raw Data", ".raw", "Raw (*.raw)"),
    "BMP...": ("BMP", ".bmp", "BMP (*.bmp)"),
    "PNG...": ("PNG", ".png", "PNG (*.png)"),
    "PGM...": ("PGM", ".pgm", "PGM (*.pgm)"),
    "FITS...": ("FITS", ".fits", "FITS (*.fits *.fit *.fts)"),
    "LUT...": ("LUT", ".lut", "LUT (*.lut)"),
    "Selection...": ("Selection", ".roi", "ROI (*.roi)"),
    "XY Coordinates...": ("XY Coordinates", ".txt", "Text (*.txt)"),
    "Results...": ("Results", ".csv", "Tables (*.csv *.xls *.tsv *.txt)"),
    "Text...": ("Text", ".txt", "Text (*.txt)"),
}


def sample_files() -> dict[str, str]:
    """Open Samples label -> file name on imagej.net (from the menu layout)."""
    out = {}
    for path, item in iter_items():
        if path[-1:] == ("Open Samples",) and item.ij_class.startswith('URLOpener("'):
            name = item.ij_class[len('URLOpener("'):-2]
            if name != "cache":
                out[item.label] = name
    return out


def sample_cache() -> Path:
    import os
    base = os.environ.get("RSSTATCHEM_HOME")
    d = Path(base) / "imagej_samples" if base else Path(tempfile.gettempdir()) / \
        "rsstatchem_imagej_samples"
    d.mkdir(parents=True, exist_ok=True)
    return d


def download(url: str, dest: Path, timeout: float = 60) -> Path:
    req = urllib.request.Request(url, headers={"User-Agent": "RS-StatChem"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - user's request
        data = resp.read()
    part = dest.with_suffix(dest.suffix + ".part")
    part.write_bytes(data)
    part.replace(dest)
    return dest


def sample_path(name: str) -> Path:
    p = sample_cache() / name
    if not p.exists():
        download(SAMPLE_URL + name, p)
    return p


# ---------------------------------------------------------------------------------- new
def new_image(session, image, name, type, fill, width, height, slices):
    t = TYPES.get(type.split()[0] if " " in type else type, GRAY8)
    for key, val in TYPES.items():
        if type.lower().startswith(key.lower()):
            t = val
    w, h, n = int(width), int(height), max(1, int(slices))
    if w < 1 or h < 1:
        raise CommandError("Width and height must be greater than 0")
    from rsstatchem.core.imaging.macro import _fill_new
    doc = ImageDoc.new(w, h, t, 1, n, 1, title=name or "Untitled")
    fill = fill.lower()
    _fill_new(doc, "random" if fill in ("noise", "random") else fill)
    return doc


def _new_image_fields(session, image):
    e = session.options.extra
    return [C.string("Name:", "Untitled", key="name", columns=12),
            C.choice("Type:", tuple(TYPES), e.get("new.type", "8-bit")),
            C.choice("Fill with:", FILLS, e.get("new.fill", "White"), key="fill"),
            C.number("Width:", e.get("new.width", 512), 0, "pixels", integer=True),
            C.number("Height:", e.get("new.height", 512), 0, "pixels", integer=True),
            C.number("Slices:", 1, 0, integer=True)]


def new_hyperstack(session, image, title, type, display, width, height, channels, slices,
                   frames, label):
    t = TYPES.get(type, GRAY8)
    c, z, f = max(1, int(channels)), max(1, int(slices)), max(1, int(frames))
    doc = ImageDoc.new(int(width), int(height), t, c, z, f, title=title or "HyperStack")
    from rsstatchem.core.imaging.macro import _fill_new
    _fill_new(doc, "ramp")
    doc.hyperstack = True
    mode = display.lower()
    if doc.is_composite and mode in ("composite", "color", "grayscale"):
        doc.composite_mode = mode
    if label:
        for i in range(doc.stack_size):
            ci, zi, ti = doc.position_of(i)
            doc.slice_labels[i] = f"c:{ci + 1}/{c} z:{zi + 1}/{z} t:{ti + 1}/{f}"
    return doc


def _hyperstack_fields(session, image):
    return [C.string("Title:", "HyperStack", columns=12),
            C.choice("Type:", tuple(TYPES), "8-bit"),
            C.choice("Display mode:", ("Composite", "Color", "Grayscale"), "Composite"),
            C.number("Width (pixels):", 400, 0, integer=True, key="width"),
            C.number("Height (pixels):", 300, 0, integer=True, key="height"),
            C.number("Channels (c):", 3, 0, integer=True, key="channels"),
            C.number("Slices (z):", 4, 0, integer=True, key="slices"),
            C.number("Frames (t):", 5, 0, integer=True, key="frames"),
            C.checkbox("Label images", False, key="label")]


# ---------------------------------------------------------------------------------- open
def _open_func(session, image, open=""):
    from rsstatchem.core.imaging.io import open_in_session
    if not open:
        raise CommandError("A file path is required (open=...)")
    try:
        open_in_session(session, open)
    except (OSError, ValueError) as exc:
        raise CommandError(str(exc)) from None


def _ui_open(bar, cmd):
    path = bar.get_path("open", "Open", bar.last_directory())
    if path:
        bar.open_path(path)


def next_file(path: str, backwards: bool = False) -> str | None:
    """Open Next: the following (or previous) image file in the folder of ``path``."""
    from rsstatchem.core.imaging.io.sequence import list_images
    p = Path(path)
    files = list_images(p.parent)
    names = [f.name for f in files]
    if not files:
        return None
    if p.name not in names:
        return str(files[0])
    i = names.index(p.name) + (-1 if backwards else 1)
    return str(files[i % len(files)])


def open_next(session, image, backwards=False):
    if image is None or not image.path:
        raise CommandError("The active image has no file path")
    nxt = next_file(image.path, backwards)
    if nxt is None or Path(nxt) == Path(image.path):
        return None
    from rsstatchem.core.imaging.io import open_image
    doc = open_image(nxt)
    session.close_image(image)
    return doc


def _ui_sample(label: str, name: str):
    def action(bar, cmd):
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication
        bar.show_status(f"Downloading {name}...")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            path = sample_path(name)
        except OSError as exc:
            bar.error(f"Unable to download the sample image\n{SAMPLE_URL}{name}\n \n{exc}",
                      "Open Samples")
            return
        finally:
            QApplication.restoreOverrideCursor()
        doc = bar.open_path(str(path))
        if hasattr(doc, "title"):
            doc.title = bar.session.unique_title(name.removesuffix(".zip"), exclude=doc)
            bar.session.image_updated(doc)
    return action


def _sample_func(name):
    def fn(session, image):
        from rsstatchem.core.imaging.io import open_in_session
        try:
            open_in_session(session, sample_path(name))
        except OSError as exc:
            raise CommandError(f"Unable to download {name}: {exc}") from None
    return fn


def _cache_samples(bar, cmd):
    failed = []
    for label, name in sample_files().items():
        try:
            sample_path(name)
        except OSError:
            failed.append(name)
    bar.show_status("Sample images cached" + (f" ({len(failed)} failed)" if failed else ""))


# ---------------------------------------------------------------------------------- import
def _raw_fields(session, image):
    from rsstatchem.core.imaging.io.raw import RAW_TYPES
    e = session.options.extra
    return [C.string("File:", "", key="open", columns=30),
            C.choice("Image type:", RAW_TYPES, e.get("raw.type", "8-bit"), key="image"),
            C.number("Width:", e.get("raw.width", 64), 0, "pixels", integer=True),
            C.number("Height:", e.get("raw.height", 64), 0, "pixels", integer=True),
            C.number("Offset to first image:", e.get("raw.offset", 0), 0, "bytes",
                     integer=True, key="offset"),
            C.number("Number of images:", e.get("raw.n", 1), 0, integer=True, key="number"),
            C.number("Gap between images:", e.get("raw.gap", 0), 0, "bytes", integer=True,
                     key="gap"),
            C.checkbox("White is zero", e.get("raw.white", False), key="white"),
            C.checkbox("Little-endian byte order", e.get("raw.little", False), key="little-endian"),
            C.checkbox("Open all files in folder", False, key="open_all"),
            C.checkbox("Use virtual stack", False, key="use")]


def import_raw(session, _active, open="", width=64, height=64, offset=0, number=1,
               gap=0, white=False, open_all=False, use=False, **kw):
    """File > Import > Raw... (the "Image type:" key is ``image``, hence ``**kw``)."""
    from rsstatchem.core.imaging.io.raw import RawOptions, read_raw
    itype = kw.get("image", "8-bit")
    little = bool(kw.get("little-endian", False))
    session.options.extra.update({"raw.type": itype, "raw.width": int(width),
                                  "raw.height": int(height), "raw.offset": int(offset),
                                  "raw.n": int(number), "raw.gap": int(gap),
                                  "raw.white": white, "raw.little": little})
    if not open:
        raise CommandError("A file path is required (open=...)")
    opts = RawOptions(itype, int(width), int(height), int(offset), int(number), int(gap),
                      white, little)
    paths = [Path(open)]
    if open_all:
        paths = sorted(p for p in Path(open).parent.iterdir() if p.is_file() and
                       not p.name.startswith("."))
    docs = []
    for p in paths:
        try:
            docs.append(read_raw(p, opts))
        except (ValueError, OSError) as exc:
            if not open_all:
                raise CommandError(str(exc)) from None
    if open_all and len(docs) > 1:
        arr = np.concatenate([d.pixels for d in docs], axis=1)
        out = ImageDoc(arr, docs[0].type, Path(open).parent.name)
        out.calibration = docs[0].calibration
        out.slice_labels = [p.name for p in paths][:out.stack_size]
        out.reset_display_range()
        return out
    return docs[0] if docs else None


def _with_file(label: str, title: str, extra_fields=None):
    """ui_action: choose a file, then (optionally) a dialog, then run with open=path."""
    def action(bar, cmd):
        path = bar.get_path("open", title, bar.last_directory())
        if not path:
            return
        fields = [f for f in cmd.dialog_fields(bar.session, bar.session.active)
                  if f.key != "open"]
        values = {"open": path}
        if fields and any(f.has_value for f in fields):
            from rsstatchem.ui.imagej.dialogs.generic import GenericDialog
            got = GenericDialog(label.rstrip(". ").removesuffix("..."), fields, bar).get()
            if got is None:
                return
            values.update(got)
        try:
            bar.session.registry.run(bar.session, label, values=values)
        except CommandError as exc:
            bar.error(str(exc), label)
    return action


def import_sequence(session, image, dir="", filter="", start=1, count=0, step=1, scale=100.0,
                    sort=True, use=False):
    from rsstatchem.core.imaging.io.sequence import open_sequence
    if not dir:
        raise CommandError("A folder is required (dir=...)")
    try:
        return open_sequence(dir, int(count), int(start), int(step), float(scale), filter,
                             sort, progress=session.progress)
    except (ValueError, OSError) as exc:
        raise CommandError(str(exc)) from None


def _sequence_fields(session, image):
    return [C.Field("directory", "Dir:", "", key="dir"),
            C.string("Filter:", ""), C.number("Start:", 1, 0, integer=True),
            C.number("Count:", 0, 0, integer=True), C.number("Step:", 1, 0, integer=True),
            C.number("Scale:", 100, 0, "%"),
            C.checkbox("Sort names numerically", True, key="sort"),
            C.checkbox("Use virtual stack", False, key="use")]


def _ui_sequence(bar, cmd):
    d = bar.get_path("dir", "Open Image Sequence", bar.last_directory())
    if not d:
        return
    fields = [f for f in _sequence_fields(bar.session, None) if f.key != "dir"]
    from rsstatchem.ui.imagej.dialogs.generic import GenericDialog
    values = GenericDialog("Sequence Options", fields, bar).get() if not \
        bar.file_hook else {f.key: f.default for f in fields}
    if values is None:
        return
    values["dir"] = d
    try:
        bar.session.registry.run(bar.session, "Image Sequence...", values=values)
    except CommandError as exc:
        bar.error(str(exc), "Image Sequence")


def import_lut(session, image, open=""):
    from rsstatchem.core.imaging.io import open_in_session
    if not open:
        raise CommandError("A file path is required (open=...)")
    open_in_session(session, open)


def import_text_image(session, image, open=""):
    from rsstatchem.core.imaging.io.text_image import read_text_image
    if not open:
        raise CommandError("A file path is required (open=...)")
    try:
        return read_text_image(open)
    except (OSError, ValueError) as exc:
        raise CommandError(str(exc)) from None


def import_text_file(session, image, open=""):
    p = Path(open)
    session.emit("text_opened", (p.name, p.read_text(errors="replace"), str(p)))


def import_results(session, image, open=""):
    from rsstatchem.core.imaging.results import ResultsTable
    t = ResultsTable.read(open)
    rt = ResultsTable.results("Results")
    for r in range(len(t)):
        rt.increment_counter()
        for h in t.headings:
            if h == "Label":
                rt.set_label(t.get_label(r), r)
            else:
                rt.set_value(h, r, t.get_value(h, r))
    session.show_table(rt)


def import_table(session, image, open=""):
    from rsstatchem.core.imaging.results import ResultsTable
    session.show_table(ResultsTable.read(open, title=Path(open).name))


def import_url(session, image, url=""):
    from rsstatchem.core.imaging.io import open_in_session
    if not url:
        raise CommandError("URL required")
    name = url.rstrip("/").rsplit("/", 1)[-1] or "download"
    d = Path(tempfile.mkdtemp(prefix="rsij_url_"))
    try:
        p = download(url, d / name)
    except OSError as exc:
        raise CommandError(f"Unable to open {url}\n{exc}") from None
    open_in_session(session, p)


def stack_from_list(session, image, open=""):
    from rsstatchem.core.imaging.io import open_image
    lines = [ln.strip() for ln in Path(open).read_text().splitlines() if ln.strip()]
    base = Path(open).parent
    docs = [open_image(p if Path(p).is_absolute() else base / p) for p in lines]
    if not docs:
        raise CommandError("The list is empty")
    first = docs[0]
    docs = [d for d in docs if (d.width, d.height, d.type) == (first.width, first.height,
                                                               first.type)]
    arr = np.concatenate([d.pixels.reshape((1, -1) + d.pixels.shape[3:]) for d in docs], 1)
    out = ImageDoc(arr, first.type, Path(open).stem)
    out.slice_labels = [Path(p).name for p in lines][:out.stack_size]
    out.luts = [first.lut.copy()]
    out.reset_display_range()
    return out


def import_avi(session, image, open="", first=1, last=0, use=False, convert=False,
               flip=False):
    from rsstatchem.core.imaging.io.avi import read_avi
    doc = read_avi(open, int(first), int(last), convert)
    if flip:
        doc.pixels = np.ascontiguousarray(doc.pixels[:, :, :, ::-1])
    return doc


def import_xy(session, image, open=""):
    from rsstatchem.core.imaging.io.text_image import read_xy
    from rsstatchem.core.imaging.roi import PolygonRoi
    if image is None:
        raise CommandError("There are no images open")
    xs, ys = read_xy(open)
    if len(xs) < 1:
        raise CommandError("No coordinates found")
    cal = image.calibration
    if cal.scaled():
        xs = np.array([cal.raw_x(v) for v in xs])
        ys = np.array([cal.raw_y(v) for v in ys])
    image.roi = PolygonRoi(xs, ys) if len(xs) >= 3 else None
    session.image_updated(image)


def tiff_virtual(session, image, open=""):
    from rsstatchem.core.imaging.io import open_image
    return open_image(open)


# ---------------------------------------------------------------------------------- close/save
def close_cmd(session, image):
    if image is not None:
        session.close_image(image)


def _ui_close(bar, cmd):
    from PySide6.QtWidgets import QApplication
    from rsstatchem.ui.imagej import windows as W
    win = QApplication.activeWindow()
    if win is not None and win in W.WINDOWS and getattr(win, "doc", None) is None:
        win.close()
        return
    doc = bar.session.active
    if doc is None:
        bar.error("There are no images open")
        return
    w = W.window_for(doc)
    if w is not None:
        w.close()
    else:
        bar.session.close_image(doc)
    bar.record_call("close")


def close_all(session, image):
    for im in list(session.images):
        session.close_image(im)


def _ui_close_all(bar, cmd):
    from rsstatchem.ui.imagej import windows as W
    for doc in list(bar.session.images):
        if doc.changed:
            a = bar.ask_save_changes(doc)
            if a == "cancel" or (a == "yes" and not bar.save_image(doc)):
                return
        doc.changed = False
        w = W.window_for(doc)
        if w is not None:
            w.close()
        else:
            bar.session.close_image(doc)
    bar.record_call("close", "*")


def save_cmd(session, image):
    from rsstatchem.core.imaging.io import save_in_session
    if not image.path or Path(image.path).suffix.lower() not in (".tif", ".tiff"):
        raise CommandError("The image has not been saved as TIFF yet (use Save As)")
    save_in_session(session, "tiff", image.path, image=image)


def _ui_save(bar, cmd):
    doc = bar.session.active
    if doc is None:
        bar.error("There are no images open")
        return
    bar.save_image(doc)


def _save_as_func(fmt):
    def fn(session, image, save=""):
        from rsstatchem.core.imaging.io import save_in_session
        if not save:
            raise CommandError("A file path is required (save=...)")
        try:
            save_in_session(session, fmt, save, image=image)
        except (OSError, ValueError) as exc:
            raise CommandError(str(exc)) from None
    return fn


def _ui_save_as(label, fmt, ext, flt):
    def action(bar, cmd):
        from rsstatchem.core.imaging.io import save_in_session
        doc = bar.session.active
        if cmd.needs_image and doc is None:
            bar.error("There are no images open")
            return
        if fmt == "Tiff" and doc is not None:
            stem = Path(doc.title).stem if "." in doc.title else doc.title
            path = bar.get_path("save", "Save as TIFF", str(Path(bar.last_directory()) /
                                                            (stem + ".tif")), flt)
            if path:
                bar.save_image(doc, path)
            return
        if fmt == "Results":
            name = "Results" + (bar.session.options.results_extension or ".csv")
        elif doc is not None:
            stem = Path(doc.title).stem if "." in doc.title else doc.title
            name = stem + ext
        else:
            name = "Log.txt"
        path = bar.get_path("save", f"Save as {fmt}", str(Path(bar.last_directory()) / name),
                            flt)
        if not path:
            return
        try:
            out = save_in_session(bar.session, fmt, path)
        except (OSError, ValueError) as exc:
            bar.error(str(exc), label)
            return
        bar.record_call("saveAs", fmt, Path(out).as_posix())
    return action


def save_sequence_cmd(session, image, format="TIFF", name="", start=0, digits=4, use=False,
                      save=""):
    from rsstatchem.core.imaging.io.sequence import save_sequence
    if not save:
        raise CommandError("A folder is required (save=...)")
    save_sequence(image, save, format, name or Path(image.title).stem, int(start), int(digits),
                  use)


def _save_sequence_fields(session, image):
    from rsstatchem.core.imaging.io.sequence import FORMATS
    return [C.choice("Format:", tuple(FORMATS), "TIFF"),
            C.string("Name:", Path(image.title).stem if image else "", columns=12),
            C.number("Start At:", 0, 0, integer=True, key="start"),
            C.number("Digits (1-8):", 4, 0, integer=True, key="digits"),
            C.checkbox("Use slice labels as file names", False, key="use"),
            C.Field("directory", "Save to:", "", key="save")]


def save_avi(session, image, compression="Uncompressed", frame=7.0, save=""):
    from rsstatchem.core.imaging.io import save_image
    if not save:
        raise CommandError("A file path is required (save=...)")
    save_image(image, save, "avi", compression=compression, fps=float(frame),
               quality=session.options.jpeg_quality)


def _avi_fields(session, image):
    fps = image.calibration.fps if image is not None and image.calibration.fps else 7.0
    return [C.choice("Compression:", ("Uncompressed", "PNG", "JPEG"), "JPEG"),
            C.number("Frame rate:", fps, 4, "fps", key="frame"),
            C.Field("file", "Save:", "", key="save")]


def revert(session, image):
    from rsstatchem.core.imaging.io import open_image
    if not image.path or not Path(image.path).exists():
        raise CommandError("This image has no file to revert to")
    fresh = open_image(image.path)
    image.replace_pixels(fresh.pixels, fresh.type)
    image.calibration = fresh.calibration
    image.luts = [lt.copy() for lt in fresh.luts]
    image.display_ranges = list(fresh.display_ranges)
    image.slice_labels = list(fresh.slice_labels)
    image.roi, image.overlay, image.threshold = fresh.roi, fresh.overlay, None
    image.info, image.properties = fresh.info, dict(fresh.properties)
    image.changed = False
    session.image_updated(image)


# ---------------------------------------------------------------------------------- print
def print_image(doc, printer, scale_to_fit: bool = True, center: bool = True,
                header: bool = False, overlay: bool = True) -> None:
    """Render the displayed image (with its overlay) onto a QPrinter (or any QPagedPaintDevice)."""
    from PySide6.QtCore import QRectF
    from PySide6.QtGui import QImage, QPainter

    from rsstatchem.core.imaging.lut import render
    rgb = np.ascontiguousarray(render(doc))
    h, w = rgb.shape[:2]
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
    p = QPainter(printer)
    page = p.viewport()
    top = 0
    if header:
        p.drawText(0, 12, doc.title)
        top = 20
    pw, ph = page.width(), page.height() - top
    s = min(pw / w, ph / h) if scale_to_fit else 1.0
    tw, th = w * s, h * s
    x0 = (pw - tw) / 2 if center else 0
    y0 = top + ((ph - th) / 2 if center else 0)
    p.drawImage(QRectF(x0, y0, tw, th), qimg)
    if overlay and doc.overlay is not None and not doc.overlay.hidden:
        from PySide6.QtGui import QColor, QPen
        for roi in doc.overlay:
            xs, ys = roi.polygon()
            p.setPen(QPen(QColor(roi.stroke_color or "#00ffff"), 1))
            from PySide6.QtCore import QPointF
            from PySide6.QtGui import QPolygonF
            pts = [QPointF(x0 + x * s, y0 + y * s) for x, y in zip(xs, ys)]
            if roi.is_area():
                p.drawPolygon(QPolygonF(pts))
            else:
                p.drawPolyline(QPolygonF(pts))
    p.end()


def _ui_page_setup(bar, cmd):
    from PySide6.QtPrintSupport import QPageSetupDialog, QPrinter
    printer = getattr(bar, "_printer", None) or QPrinter()
    bar._printer = printer
    if not getattr(bar, "file_hook", None):
        QPageSetupDialog(printer, bar).exec()


def _ui_print(bar, cmd):
    from PySide6.QtPrintSupport import QPrintDialog, QPrinter
    doc = bar.session.active
    if doc is None:
        bar.error("There are no images open")
        return
    printer = getattr(bar, "_printer", None) or QPrinter()
    bar._printer = printer
    if QPrintDialog(printer, bar).exec():
        print_image(doc, printer)


def _ui_quit(bar, cmd):
    bar.close()


# ---------------------------------------------------------------------------------- folders
def _show_folder(kind):
    def action(bar, cmd):
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        import os
        home = Path.home()
        base = Path(os.environ.get("RSSTATCHEM_HOME", home))
        if kind == "Image":
            im = bar.session.active
            if im is None or not im.path:
                bar.error("The active image has no folder")
                return
            path = Path(im.path).parent
        elif kind == "temp":
            path = Path(tempfile.gettempdir())
        elif kind == "Home":
            path = home
        elif kind == "LUTs":
            path = base / "imagej" / "luts"
        elif kind == "Macros":
            path = base / "imagej" / "macros"
        elif kind == "Plugins":
            path = base / "imagej" / "plugins"
        else:
            path = base
        path.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
    return action


def _ui_text_window(bar, cmd):
    from rsstatchem.ui.imagej import windows as W
    from rsstatchem.ui.imagej.frames.text_window import TextWindow
    win = TextWindow("Untitled.txt", "", bar, editable=True)
    W.register(win)
    win.show()
    return win


# ---------------------------------------------------------------------------------- register
def register(registry) -> None:
    from rsstatchem.ui.imagej import commands_edit as E

    add = registry.add
    path_field = [C.string("Open:", "", key="open", columns=30)]
    add("Image...", new_image, _new_image_fields, needs_image=False, undoable=False)
    add("Hyperstack...", new_hyperstack, _hyperstack_fields, needs_image=False,
        undoable=False)
    add("Text Window", None, needs_image=False, undoable=False, recordable=False,
        ui_action=_ui_text_window)
    add("Internal Clipboard", E.internal_clipboard, needs_image=False, undoable=False)
    add("System Clipboard", E.system_clipboard, needs_image=False, undoable=False)
    add("Open...", _open_func, path_field, needs_image=False, undoable=False,
        ui_action=_ui_open)
    add("Open Next", open_next, needs_image=True, undoable=False)
    for label, name in sample_files().items():
        add(label, _sample_func(name), needs_image=False, undoable=False,
            ui_action=_ui_sample(label, name))
    add("Cache Sample Images ", None, needs_image=False, undoable=False,
        ui_action=_cache_samples)
    add("Image Sequence...", import_sequence, _sequence_fields, needs_image=False,
        undoable=False, ui_action=_ui_sequence)
    add("Raw...", import_raw, _raw_fields, needs_image=False, undoable=False,
        ui_action=_with_file("Raw...", "Open Raw"))
    add("LUT... ", import_lut, path_field, needs_image=False, undoable=False,
        ui_action=_with_file("LUT... ", "Open LUT"))
    add("Text Image... ", import_text_image, path_field, needs_image=False, undoable=False,
        ui_action=_with_file("Text Image... ", "Open Text Image"))
    add("Text File... ", import_text_file, path_field, needs_image=False, undoable=False,
        ui_action=_with_file("Text File... ", "Open Text File"))
    add("Results... ", import_results, path_field, needs_image=False, undoable=False,
        ui_action=_with_file("Results... ", "Open Results"))
    add("Table... ", import_table, path_field, needs_image=False, undoable=False,
        ui_action=_with_file("Table... ", "Open Table"))
    add("URL...", import_url, [C.string("URL:", "https://", key="url", columns=40)],
        needs_image=False, undoable=False)
    add("Stack From List...", stack_from_list, path_field, needs_image=False, undoable=False,
        ui_action=_with_file("Stack From List...", "Open Image List"))
    add("TIFF Virtual Stack...", tiff_virtual, path_field, needs_image=False, undoable=False,
        ui_action=_with_file("TIFF Virtual Stack...", "Open TIFF"))
    add("AVI...", import_avi, path_field + [
        C.number("First frame:", 1, 0, integer=True, key="first"),
        C.number("Last frame:", 0, 0, integer=True, key="last"),
        C.checkbox("Use virtual stack", False, key="use"),
        C.checkbox("Convert to grayscale", False, key="convert"),
        C.checkbox("Flip vertical", False, key="flip")], needs_image=False, undoable=False,
        ui_action=_with_file("AVI...", "Open AVI"))
    add("XY Coordinates... ", import_xy, path_field, undoable=False,
        ui_action=_with_file("XY Coordinates... ", "Open XY Coordinates"))
    for kind in ("Image", "Plugins", "Macros", "LUTs", "ImageJ", "temp", "Home"):
        add(kind, None, needs_image=False, undoable=False, recordable=False,
            ui_action=_show_folder(kind))
    add("Close", close_cmd, needs_image=False, undoable=False, ui_action=_ui_close)
    add("Close All", close_all, needs_image=False, undoable=False, ui_action=_ui_close_all)
    add("Save", save_cmd, undoable=False, ui_action=_ui_save)
    save_field = [C.string("Save:", "", key="save", columns=30)]
    for label, (fmt, ext, flt) in SAVE_AS.items():
        needs = fmt not in ("Results", "Text")
        add(label, _save_as_func(fmt), save_field, needs_image=needs, undoable=False,
            ui_action=_ui_save_as(label, fmt, ext, flt), recordable=False)
    add("Image Sequence... ", save_sequence_cmd, _save_sequence_fields, undoable=False)
    add("AVI... ", save_avi, _avi_fields, undoable=False)
    add("Revert", revert)
    add("Page Setup...", None, needs_image=False, undoable=False, recordable=False,
        ui_action=_ui_page_setup)
    add("Print...", None, undoable=False, recordable=False, ui_action=_ui_print)
    add("Quit", None, needs_image=False, undoable=False, recordable=False,
        ui_action=_ui_quit)


def selftest() -> None:
    from rsstatchem.core.imaging.session import Session
    s = Session()
    register(s.registry)
    s.run("Image...", "name=t type=16-bit fill=Ramp width=8 height=4 slices=2")
    im = s.active
    assert im.type == GRAY16 and im.n_slices == 2 and im.raw_value(4, 0) == 32768
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "t.tif"
        s.run("Tiff...", f"save=[{p}]")
        s.run("Close")
        s.run("Open...", f"open=[{p}]")
        assert s.active.n_slices == 2 and s.active.title == "t.tif"
