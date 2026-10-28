"""AVI reading and writing without ffmpeg (File > Import > AVI..., File > Save As > AVI...).

Follows ImageJ 1.54 ``ij/plugin/AVI_Reader.java`` / ``filter/AVI_Writer.java`` (public domain,
NIH; M. Schmid et al.): RIFF AVI with uncompressed DIB frames (8-bit with palette, 24/32-bit
BGR, bottom-up or top-down), MJPG ('MJPG', 'JPEG', 'jpeg') and PNG ('png ', 'MPNG') frames
decoded with Pillow; AVI-2 'AVIX' extensions are followed. Writing produces an AVI-1 file
with an idx1 index: "Uncompressed" (8-bit with the LUT as palette, otherwise 24-bit), "PNG" or
"JPEG" frames. Grey frames of an RGB AVI are opened as 8-bit when all frames are grey
(ImageJ's "convert to grayscale" option is applied when ``gray=True``).
"""
from __future__ import annotations

import io
import struct
from pathlib import Path

import numpy as np

from rsstatchem.core.imaging.image import COLOR_256, GRAY8, ImageDoc
from rsstatchem.core.imaging.lut import LUT, render


def _chunks(data: bytes, start: int, end: int):
    pos = start
    while pos + 8 <= end:
        fourcc = data[pos:pos + 4]
        size = struct.unpack_from("<I", data, pos + 4)[0]
        body = pos + 8
        yield fourcc, body, size
        pos = body + size + (size & 1)


def read_avi(path, first: int = 1, last: int = 0, gray: bool = False) -> ImageDoc:
    from PIL import Image

    data = Path(path).read_bytes()
    if data[:4] != b"RIFF" or data[8:12] != b"AVI ":
        raise ValueError("Not an AVI file")
    state = {"bih": None, "palette": None, "frames": [], "usec": 0, "handler": b""}

    def walk(start, end):
        for fourcc, body, size in _chunks(data, start, end):
            if fourcc in (b"RIFF", b"LIST"):
                walk(body + 4, min(body + size, len(data)))
            elif fourcc == b"avih":
                state["usec"] = struct.unpack_from("<I", data, body)[0]
            elif fourcc == b"strh" and data[body:body + 4] == b"vids":
                state["handler"] = data[body + 4:body + 8]
            elif fourcc == b"strf" and state["bih"] is None:
                bih = struct.unpack_from("<IiiHHIIiiII", data, body)
                state["bih"] = bih
                ncol = bih[9] or (256 if bih[4] == 8 else 0)
                if bih[4] <= 8 and size >= 40 + 4 * ncol:
                    pal = np.frombuffer(data, np.uint8, 4 * ncol, body + bih[0]).reshape(-1, 4)
                    state["palette"] = pal[:, [2, 1, 0]]
            elif len(fourcc) == 4 and fourcc[2:4] in (b"db", b"dc") and fourcc[:2].isdigit():
                if fourcc[:2] == b"00" and size > 0:
                    state["frames"].append((body, size))

    walk(12, len(data))
    bih = state["bih"]
    if bih is None:
        raise ValueError("AVI file has no video stream format")
    _, width, height, _, bitcount, compression, *_ = bih
    bottom_up = height > 0
    height = abs(height)
    frames = state["frames"]
    last = len(frames) if last <= 0 else min(last, len(frames))
    frames = frames[max(first, 1) - 1:last]
    comp = struct.pack("<I", compression) if compression > 3 else b""
    planes = []
    for body, size in frames:
        chunk = data[body:body + size]
        if compression in (0, 3) and not comp:
            bpp = bitcount // 8 if bitcount >= 8 else 1
            stride = (width * bitcount // 8 + 3) & ~3
            rows = np.frombuffer(chunk, np.uint8, stride * height).reshape(height, stride)
            if bottom_up:
                rows = rows[::-1]
            if bitcount == 8:
                planes.append(rows[:, :width].copy())
            elif bitcount in (24, 32):
                px = rows[:, :width * bpp].reshape(height, width, bpp)
                planes.append(px[..., [2, 1, 0]].copy())
            else:
                raise ValueError(f"Unsupported AVI bit depth {bitcount}")
        else:
            im = Image.open(io.BytesIO(chunk))
            im.load()
            planes.append(np.asarray(im if im.mode == "L" else im.convert("RGB")))
    if not planes:
        raise ValueError("AVI file contains no frames")
    title = Path(path).name
    arr = np.stack(planes)
    if arr.ndim == 3:
        doc = ImageDoc.from_array(arr, "ZYX", title)
        if state["palette"] is not None:
            lut = LUT.from_array(state["palette"])
            doc.luts = [lut]
            if not (lut.is_grayscale() or lut.is_inverted()):
                doc.type = COLOR_256
    else:
        if gray or (np.array_equal(arr[..., 0], arr[..., 1]) and
                    np.array_equal(arr[..., 0], arr[..., 2])):
            doc = ImageDoc.from_array(arr[..., 0].copy(), "ZYX", title)
        else:
            doc = ImageDoc.from_array(arr, "ZYXS", title)
    if state["usec"]:
        doc.calibration.fps = round(1e6 / state["usec"], 4)
        doc.calibration.frame_interval = state["usec"] / 1e6
        base = max(first, 1)
        doc.slice_labels = [f"{(base + i) * state['usec'] / 1e6:.2f} s"
                            for i in range(doc.stack_size)]
    return doc


def avi_bytes(doc: ImageDoc, compression: str = "Uncompressed", fps: float = 7.0,
              jpeg_quality: int = 90) -> bytes:
    from PIL import Image

    w, h = doc.width, doc.height
    eight_bit = compression == "Uncompressed" and not doc.is_rgb and not doc.is_composite \
        and doc.type in (GRAY8, COLOR_256)
    frames = []
    for i in range(doc.stack_size if not doc.is_composite else doc.n_slices * doc.n_frames):
        if doc.is_composite:
            z, t = i % doc.n_slices, i // doc.n_slices
            rgb = render(doc, 0, z, t)
        else:
            c, z, t = doc.position_of(i)
            rgb = None if eight_bit else render(doc, c, z, t)
        if eight_bit:
            plane = doc.pixels[t, z, c]
            stride = (w + 3) & ~3
            buf = np.zeros((h, stride), np.uint8)
            buf[:, :w] = plane
            frames.append(buf[::-1].tobytes())
        elif compression == "Uncompressed":
            stride = (w * 3 + 3) & ~3
            buf = np.zeros((h, stride), np.uint8)
            buf[:, :w * 3] = rgb[..., [2, 1, 0]].reshape(h, w * 3)
            frames.append(buf[::-1].tobytes())
        else:
            out = io.BytesIO()
            im = Image.fromarray(np.ascontiguousarray(rgb))
            if compression == "PNG":
                im.save(out, "PNG")
            else:
                im.save(out, "JPEG", quality=int(jpeg_quality))
            frames.append(out.getvalue())
    bitcount = 8 if eight_bit else 24
    comp_code = {"Uncompressed": 0, "PNG": 0x20676E70, "JPEG": 0x47504A4D}[compression]
    usec = int(round(1e6 / fps)) if fps > 0 else 142857
    fourcc = b"00db" if comp_code == 0 else b"00dc"
    palette = b""
    if eight_bit:
        lut = doc.lut.as_array()
        palette = b"".join(bytes([b, g, r, 0]) for r, g, b in lut.tolist())
    max_size = max(len(f) for f in frames)
    strf = struct.pack("<IiiHHIIiiII", 40, w, h, 1, bitcount, comp_code,
                       0 if comp_code else len(frames[0]), 0, 0,
                       256 if eight_bit else 0, 0) + palette

    def chunk(cc: bytes, body: bytes) -> bytes:
        return cc + struct.pack("<I", len(body)) + body + (b"\0" if len(body) & 1 else b"")

    def lst(kind: bytes, body: bytes) -> bytes:
        return b"LIST" + struct.pack("<I", len(body) + 4) + kind + body

    avih = struct.pack("<IIIIIIIIIIIIII", usec, max_size * int(fps or 7), 0, 0x10,
                       len(frames), 0, 1, max_size, w, h, 0, 0, 0, 0)
    scale, rate = 1000, int(round(fps * 1000)) or 7000
    strh = b"vids" + (b"DIB " if comp_code == 0 else struct.pack("<I", comp_code)) + \
        struct.pack("<IHHIIIIIIIIhhhh", 0, 0, 0, 0, scale, rate, 0, len(frames), max_size,
                    -1 & 0xFFFFFFFF, 0, 0, 0, w, h)
    hdrl = lst(b"hdrl", chunk(b"avih", avih) +
               lst(b"strl", chunk(b"strh", strh) + chunk(b"strf", strf) +
                   chunk(b"strn", b"ImageJ AVI     \0")))
    movi_body = b""
    index = b""
    offset = 4
    for f in frames:
        c = chunk(fourcc, f)
        index += fourcc + struct.pack("<III", 0x10, offset, len(f))
        movi_body += c
        offset += len(c)
    movi = lst(b"movi", movi_body)
    body = b"AVI " + hdrl + movi + chunk(b"idx1", index)
    return b"RIFF" + struct.pack("<I", len(body)) + body


def write_avi(doc: ImageDoc, path, compression: str = "Uncompressed", fps: float = 7.0,
              jpeg_quality: int = 90) -> None:
    Path(path).write_bytes(avi_bytes(doc, compression, fps, jpeg_quality))
