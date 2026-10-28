"""GIF, PNG, JPEG, BMP and PGM/PPM/PBM through Pillow, with ImageJ's conventions.

Reading (ImageJ ``Opener``): palette images keep their palette as the LUT (an inverting grey
palette stays an 8-bit image with an inverting LUT, a colour palette makes "8-bit Color");
animated GIFs open as stacks; 16-bit PNG/PGM open as 16-bit; RGB JPEGs whose three channels are
equal open as 8-bit (``Opener.convertGrayJpegTo8Bits``).

Writing follows ImageJ 1.54's writers (``GifWriter``, ``PNG_Writer``, ``JpegWriter``,
``BMP_Writer``, ``PNM_Writer``; public domain, NIH): 8-bit images are written with their LUT as
the palette, 16-bit PNGs with the default LUT stay 16-bit, PGM stores the displayed grey values
(an inverting LUT is applied), JPEG stores what is displayed with the Input/Output quality.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from rsstatchem.core.imaging.image import COLOR_256, COLOR_RGB, GRAY8, GRAY16, ImageDoc
from rsstatchem.core.imaging.lut import LUT, render, to_8bit


# ---------------------------------------------------------------------------------- reading
def _palette_lut(im) -> LUT:
    pal = np.asarray(im.getpalette() or [], dtype=np.uint8)
    if pal.size == 0:
        return LUT.grays()
    return LUT.from_array(pal.reshape(-1, 3))


def _doc_from_palette(arr, lut: LUT, title: str, axes: str = "YX") -> ImageDoc:
    doc = ImageDoc.from_array(arr, axes, title)
    doc.luts = [lut]
    if not (lut.is_grayscale() or lut.is_inverted()):
        doc.type = COLOR_256
    return doc


def read_pillow(path) -> ImageDoc:
    """Open GIF/PNG/JPEG/BMP/PNM; multi-frame GIFs become stacks."""
    from PIL import GifImagePlugin, Image, ImageSequence

    title = Path(path).name
    # keep later GIF frames as palette images when they share the palette (Pillow's default
    # converts every frame after the first to RGB)
    GifImagePlugin.LOADING_STRATEGY = \
        GifImagePlugin.LoadingStrategy.RGB_AFTER_DIFFERENT_PALETTE_ONLY
    with Image.open(str(path)) as im:
        n_frames = getattr(im, "n_frames", 1)
        if n_frames > 1 and im.format == "GIF":
            frames = [f.copy() for f in ImageSequence.Iterator(im)]
            if all(f.mode == "L" for f in frames):
                return ImageDoc.from_array(np.stack([np.asarray(f) for f in frames]), "ZYX",
                                           title)
            if all(f.mode == "P" for f in frames):
                pals = [tuple(f.getpalette() or ()) for f in frames]
                if all(p == pals[0] for p in pals):
                    arr = np.stack([np.asarray(f) for f in frames])
                    return _doc_from_palette(arr, _palette_lut(frames[0]), title, "ZYX")
            arr = np.stack([np.asarray(f.convert("RGB")) for f in frames])
            return ImageDoc.from_array(arr, "ZYXS", title)
        return _doc_from_pil(im, title)


def _doc_from_pil(im, title: str) -> ImageDoc:
    mode = im.mode
    if mode == "P":
        if "transparency" in im.info and isinstance(im.info["transparency"], bytes):
            return ImageDoc.from_array(np.asarray(im.convert("RGB")), "YXS", title)
        return _doc_from_palette(np.asarray(im), _palette_lut(im), title)
    if mode in ("L", "1", "LA"):
        return ImageDoc.from_array(np.asarray(im.convert("L")), "YX", title)
    if mode.startswith("I;16"):
        return ImageDoc.from_array(np.asarray(im).astype(np.uint16), "YX", title)
    if mode == "I":
        a = np.asarray(im)
        if a.min() >= 0 and a.max() <= 65535:
            return ImageDoc.from_array(a.astype(np.uint16), "YX", title)
        return ImageDoc.from_array(a.astype(np.float32), "YX", title)
    if mode == "F":
        return ImageDoc.from_array(np.asarray(im, dtype=np.float32), "YX", title)
    rgb = np.asarray(im.convert("RGB"))
    if im.format == "JPEG" and np.array_equal(rgb[..., 0], rgb[..., 1]) and \
            np.array_equal(rgb[..., 0], rgb[..., 2]):
        return ImageDoc.from_array(rgb[..., 0].copy(), "YX", title)
    return ImageDoc.from_array(rgb, "YXS", title)


def read_pil_image(im, title: str = "Untitled") -> ImageDoc:
    """ImageDoc from an in-memory PIL image (system clipboard, URLs)."""
    return _doc_from_pil(im, title)


# ---------------------------------------------------------------------------------- writing
def _index_plane(doc: ImageDoc, plane: np.ndarray) -> np.ndarray:
    lo, hi = doc.display_range
    if plane.dtype == np.uint8 and (lo, hi) == (0.0, 255.0):
        return plane
    return to_8bit(plane, lo, hi)


def _pil_8bit(doc: ImageDoc, plane: np.ndarray):
    """8-bit (palette or grey) PIL image of a non-RGB plane with the image's LUT."""
    from PIL import Image

    idx = np.ascontiguousarray(_index_plane(doc, plane))
    lut = doc.lut
    if np.array_equal(lut.as_array(), LUT.grays().as_array()):
        return Image.fromarray(idx)
    im = Image.frombytes("P", (idx.shape[1], idx.shape[0]), idx.tobytes())
    im.putpalette(lut.as_array().ravel().tolist())
    return im


def _rendered(doc: ImageDoc, index: int | None = None):
    from PIL import Image

    if index is None:
        return Image.fromarray(np.ascontiguousarray(render(doc)))
    c, z, t = doc.position_of(index)
    return Image.fromarray(np.ascontiguousarray(render(doc, c, z, t)))


def write_png(doc: ImageDoc, path) -> None:
    from PIL import Image

    plane = doc.plane()
    if doc.is_rgb:
        Image.fromarray(np.ascontiguousarray(plane)).save(str(path), "PNG")
    elif doc.type == GRAY16 and not doc.is_composite and \
            np.array_equal(doc.lut.as_array(), LUT.grays().as_array()):
        Image.fromarray(np.ascontiguousarray(plane.astype(np.uint16))).save(
            str(path), "PNG")
    elif doc.type in (GRAY8, COLOR_256) and not doc.is_composite:
        _pil_8bit(doc, plane).save(str(path), "PNG")
    else:
        _rendered(doc).save(str(path), "PNG")


def write_gif(doc: ImageDoc, path) -> None:
    """8-bit images with their LUT; stacks as animated GIF; RGB quantized to 256 colours."""
    from PIL import Image

    frames = []
    for i in range(doc.stack_size):
        c, z, t = doc.position_of(i)
        plane = doc.plane(c, z, t)
        if doc.is_rgb:
            frames.append(Image.fromarray(np.ascontiguousarray(plane)).quantize(256))
        else:
            im = _pil_8bit(doc, plane)
            if im.mode == "L":
                im = _gray_palette(im)
            frames.append(im)
    first, rest = frames[0], frames[1:]
    kw = {}
    if rest:
        fps = doc.calibration.fps or 7.0
        kw = {"save_all": True, "append_images": rest, "duration": int(1000 / fps), "loop": 0}
    first.save(str(path), "GIF", optimize=False, **kw)


def _gray_palette(im):
    from PIL import Image

    p = Image.frombytes("P", im.size, im.tobytes())
    p.putpalette(LUT.grays().as_array().ravel().tolist())
    return p


def write_jpeg(doc: ImageDoc, path, quality: int = 85) -> None:
    from PIL import Image

    if not doc.is_rgb and not doc.is_composite and doc.lut.is_grayscale() and \
            not doc.lut.is_inverted():
        Image.fromarray(np.ascontiguousarray(_index_plane(doc, doc.plane()))).save(
            str(path), "JPEG", quality=int(quality))
    else:
        _rendered(doc).save(str(path), "JPEG", quality=int(quality))


def write_bmp(doc: ImageDoc, path) -> None:
    from PIL import Image

    if doc.is_rgb:
        Image.fromarray(np.ascontiguousarray(doc.plane())).save(str(path), "BMP")
    elif doc.is_composite:
        _rendered(doc).save(str(path), "BMP")
    else:
        im = _pil_8bit(doc, doc.plane())
        if im.mode == "L":
            im = _gray_palette(im)
        im.save(str(path), "BMP")


def write_pnm(doc: ImageDoc, path) -> None:
    """PNM_Writer: P6 for RGB, 16-bit P5 for 16-bit, otherwise 8-bit P5 of the grey values
    (inverting LUTs applied)."""
    w, h = doc.width, doc.height
    header = "\n# Written by ImageJ PNM Writer\n"
    if doc.is_rgb:
        data = np.ascontiguousarray(doc.plane()).tobytes()
        Path(path).write_bytes(f"P6{header}{w} {h}\n255\n".encode() + data)
        return
    plane = doc.plane()
    if doc.type == GRAY16:
        mx = max(int(plane.max()), 256)
        data = np.ascontiguousarray(plane, dtype=">u2").tobytes()
        Path(path).write_bytes(f"P5{header}{w} {h}\n{mx}\n".encode() + data)
        return
    if doc.bit_depth == 8 and doc.is_inverted_lut():
        idx = 255 - plane.astype(np.uint8)
    else:
        idx = _index_plane(doc, plane)
    Path(path).write_bytes(f"P5{header}{w} {h}\n255\n".encode() +
                           np.ascontiguousarray(idx, np.uint8).tobytes())


def write_pil_rgb(doc: ImageDoc):
    """PIL RGB image of what is displayed (Copy to System, Print, Flatten helpers)."""
    return _rendered(doc)
