"""Image > Stacks > 3D Project and the orthogonal-views slicing helper.

:func:`project` is a port of ImageJ 1.54 ij/plugin/Projector.java (public domain, NIH; based on
Pascal code by M. Castle, U. Michigan): nearest-point, brightest-point and mean-value
projections of a volume rotating about the x, y or z axis, with transparency bounds, opacity
and surface/interior depth cueing, in the same integer arithmetic (BIGPOWEROF2 = 8192 fixed
point, truncating Java division, byte wrap-around) so that the projections equal ImageJ's
(goldens ``tests/data_imagej/golden/image_stacks/p3d_*``). The per-pixel loops are vectorised
per slice; z-buffer ties are resolved in ImageJ's scan order.
"""
from __future__ import annotations

import math

import numpy as np

from rsstatchem.core.imaging.image import COLOR_RGB, GRAY8, ImageDoc

AXES = ("X-Axis", "Y-Axis", "Z-Axis")
METHODS = ("Nearest Point", "Brightest Point", "Mean Value")
BIG = 8192


def _jdiv(a, b):
    """Java integer division (truncates toward zero)."""
    a = np.asarray(a, dtype=np.int64)
    q = np.abs(a) // abs(b)
    return np.where((a < 0) != (b < 0), -q, q)


def _first_per_offset(off: np.ndarray, key: np.ndarray, minimize: bool = True):
    """Index (into the candidate arrays, in scan order) of the winning candidate per offset:
    smallest ``key`` (largest if not ``minimize``), the first in scan order among ties."""
    order = np.arange(off.size)
    k = key if minimize else -key
    idx = np.lexsort((order, k, off))
    o_sorted = off[idx]
    first = np.ones(idx.size, dtype=bool)
    first[1:] = o_sorted[1:] != o_sorted[:-1]
    return idx[first]


class _State:
    def __init__(self, projsize, method, opacity, cue_int):
        self.proj = np.zeros(projsize, dtype=np.int64)       # projArray (bytes as ints)
        self.zbuf = np.full(projsize, 32767, dtype=np.int64)
        self.opa = np.zeros(projsize, dtype=np.int64)
        self.bright = np.zeros(projsize, dtype=np.int64)
        self.cuez = np.zeros(projsize, dtype=np.int64)
        self.sum = np.zeros(projsize, dtype=np.int64)
        self.count = np.zeros(projsize, dtype=np.int64)


def _project_points(st: _State, off, znew, pix, p):
    """The per-pixel body shared by the three axis loops, for one slice's candidates."""
    method, opacity = p["method"], p["opacity"]
    cue_surf, cue_int = p["cue_surf"], p["cue_int"]
    zmax, zrange100 = p["zmax"], p["zrange100"]
    opacity_or_nearest = method == 0 or opacity > 0
    opacity_not_nearest = opacity > 0 and method != 0
    if opacity_or_nearest and off.size:
        win = _first_per_offset(off, znew, True)
        o, zn, px = off[win], znew[win], pix[win]
        better = zn < st.zbuf[o]
        o, zn, px = o[better], zn[better], px[better]
        st.zbuf[o] = zn
        if cue_surf < 100:
            v = _jdiv(cue_surf * px, 100) + _jdiv((100 - cue_surf) * px * (zmax - zn), zrange100)
        else:
            v = px
        if opacity_not_nearest:
            st.opa[o] = v & 0xFF
        else:
            st.proj[o] = v & 0xFF
    if method == 2:
        np.add.at(st.sum, off, pix)
        np.add.at(st.count, off, 1)
    elif method == 1 and off.size:
        if cue_int < 100:
            # lexicographic max of (pixel, z); existing entries win exact ties
            key = pix * 100000 + (znew + 50000)
            win = _first_per_offset(off, key, False)
            o, zn, px = off[win], znew[win], pix[win]
            cur_b = st.bright[o]
            better = (px > cur_b) | ((px == cur_b) & (zn > st.cuez[o]))
            o, zn, px = o[better], zn[better], px[better]
            st.bright[o] = px
            st.cuez[o] = zn
            v = _jdiv(cue_int * px, 100) + _jdiv((100 - cue_int) * px * (zmax - zn), zrange100)
            st.proj[o] = v & 0xFF
        else:
            np.maximum.at(st.proj, off, pix)


def _volume_bytes(doc: ImageDoc, channel: int = 0, frame: int = 0) -> np.ndarray:
    """(Z, Y, X) uint8 volume; 16/32-bit planes scaled with the display range
    (ImageProcessor.convertToByte(true))."""
    from rsstatchem.core.imaging.convert import float_to_byte, short_to_byte
    vol = doc.pixels[frame, :, channel]
    if doc.type in (GRAY8,) or vol.dtype == np.uint8:
        return vol
    lo, hi = doc.display_ranges[channel]
    if vol.dtype == np.uint16:
        return np.stack([short_to_byte(p, lo, hi) for p in vol])
    return np.stack([float_to_byte(p, lo, hi) for p in vol])


def _z_scale(vol: np.ndarray, slice_interval: float, rect) -> np.ndarray:
    """Projector.zScale: bilinear interpolation of the xz planes to (int)(n*interval+0.5)."""
    from rsstatchem.core.imaging import transform as T
    x, y, w, h = rect
    d1 = vol.shape[0]
    d2 = int(d1 * slice_interval + 0.5)
    out = np.zeros((d2, h, w), dtype=np.uint8)
    for yy in range(h):
        xz = np.ascontiguousarray(vol[:, y + yy, x:x + w])
        out[:, yy, :] = T.resize_plane(xz, "byte", w, d2, T.BILINEAR)
    return out


def project_volume(vol: np.ndarray, method: int = 1, axis: int = 1, slice_interval: float = 1.0,
                   initial: int = 0, total: int = 360, increment: int = 10, lower: int = 1,
                   upper: int = 255, opacity: int = 0, cue_surf: int = 100, cue_int: int = 50,
                   rect=None, progress=None) -> np.ndarray:
    """Projector.doProjections on an 8-bit volume (Z, Y, X); returns (N, H, W) uint8.

    ``cue_surf``/``cue_int`` are ImageJ's internal depthCueSurf/depthCueInt (100 - the dialog's
    "depth-cueing" percentages)."""
    n_slices, img_h, img_w = vol.shape
    left, top, rw, rh = rect if rect is not None else (0, 0, img_w, img_h)
    right, bottom = left + rw, top + rh
    if increment == 0 and total != 0:
        increment = 5
    neg = increment < 0
    inc = -increment if neg else increment
    if inc == 0:
        n_proj = 1
        angle = 0
    else:
        n_proj, angle = 0, 0
        while angle <= total:
            n_proj += 1
            angle += inc
    if angle > 360:
        n_proj -= 1
    n_proj = max(n_proj, 1)
    width, height = right - left, bottom - top
    xcenter = (left + right) // 2
    ycenter = (top + bottom) // 2
    zcenter = int(n_slices * slice_interval / 2.0 + 0.5)
    depth = n_slices * slice_interval
    if axis == 0:
        projheight = int(math.sqrt(depth * depth + height * height) + 0.5)
        projwidth = width
    elif axis == 1:
        projwidth = int(math.sqrt(depth * depth + width * width) + 0.5)
        projheight = height
    else:
        projwidth = int(math.sqrt(depth * depth + width * width) + 0.5)
        projheight = int(math.sqrt(depth * depth + height * height) + 0.5)
    if projwidth % 2 == 1:
        projwidth += 1
    projsize = projwidth * projheight
    if projwidth <= 0 or projheight <= 0:
        raise ValueError("'projwidth' or 'projheight' <= 0")
    out = np.zeros((n_proj, projheight, projwidth), dtype=np.uint8)
    theta = initial
    vol = vol.astype(np.int64)
    zs_slice = [int((k - 1) * slice_interval + 0.5) - zcenter for k in range(1, n_slices + 1)]
    for n in range(n_proj):
        if progress is not None:
            progress(n / n_proj)
        rad = theta * math.pi / 180.0
        cost = int(BIG * math.cos(rad) + 0.5)
        sint = int(BIG * math.sin(rad) + 0.5)
        st = _State(projsize, method, opacity, cue_int)
        if axis == 0:
            zmax = zcenter + projheight // 2
            zmin = zcenter - projheight // 2
        elif axis == 1:
            zmax = zcenter + projwidth // 2
            zmin = zcenter - projwidth // 2
        else:
            zmax = int((n_slices - 1) * slice_interval + 0.5) - zcenter
            zmin = -zcenter
        params = dict(method=method, opacity=opacity, cue_surf=cue_surf, cue_int=cue_int,
                      zmax=zmax, zrange100=100 * (zmax - zmin))
        jj = np.arange(top, bottom)
        ii = np.arange(left, right)
        for k in range(n_slices):
            pixels = vol[k, top:bottom, left:right]
            z = zs_slice[k]
            if axis == 0:
                ycos = (top - ycenter - 1) * cost + (jj - top + 1) * cost
                ysin = (top - ycenter - 1) * sint + (jj - top + 1) * sint
                ynew = _jdiv(ycos - z * sint, BIG) + ycenter - top
                znew_row = _jdiv(ysin + z * cost, BIG) + zcenter
                offinit = ((projheight - bottom + top) // 2) * projwidth + \
                    (projwidth - right + left) // 2 - 1
                base = offinit + ynew * projwidth           # per row
                raw = base[:, None] + 1 + np.arange(width)[None, :]
                bad = (raw >= projsize) | (raw < 0)
                # once out of range, the offset restarts from 0 and keeps incrementing
                firstbad = np.where(bad.any(axis=1), np.argmax(bad, axis=1), width)
                col = np.arange(width)[None, :]
                off = np.where(col >= firstbad[:, None], col - firstbad[:, None], raw)
                znew = np.repeat(znew_row[:, None], width, axis=1)
            elif axis == 1:
                xcos = (left - xcenter - 1) * cost + (ii - left + 1) * cost
                xsin = (left - xcenter - 1) * sint + (ii - left + 1) * sint
                xnew = _jdiv(xcos + z * sint, BIG) + xcenter - left
                znew_col = _jdiv(z * cost - xsin, BIG) + zcenter
                offinit0 = ((projheight - bottom + top) // 2) * projwidth + \
                    (projwidth - right + left) // 2 - projwidth
                rows = offinit0 + (jj - top + 1) * projwidth
                off = rows[:, None] + xnew[None, :]
                off = np.where((off >= projsize) | (off < 0), 0, off)
                znew = np.repeat(znew_col[None, :], height, axis=0)
            else:
                ycos = (top - ycenter - 1) * cost + (jj - top + 1) * cost
                ysin = (top - ycenter - 1) * sint + (jj - top + 1) * sint
                xcos = (left - xcenter - 1) * cost + (ii - left + 1) * cost
                xsin = (left - xcenter - 1) * sint + (ii - left + 1) * sint
                xnew = _jdiv(xcos[None, :] - ysin[:, None], BIG) + xcenter - left
                ynew = _jdiv(xsin[None, :] + ycos[:, None], BIG) + ycenter - top
                offinit = ((projheight - bottom + top) // 2) * projwidth + \
                    (projwidth - right + left) // 2 - 1
                off = offinit + ynew * projwidth + xnew
                off = np.where((off >= projsize) | (off < 0), 0, off)
                znew = np.full(off.shape, z, dtype=np.int64)
            sel = (pixels <= upper) & (pixels >= lower)
            _project_points(st, off[sel].ravel(), znew[sel].ravel(), pixels[sel].ravel(), params)
        proj = st.proj
        if method == 2:
            nz = st.count != 0
            proj = proj.copy()
            proj[nz] = (st.sum[nz] // st.count[nz]) & 0xFF
        if opacity > 0 and method != 0:
            proj = ((opacity * st.opa + (100 - opacity) * proj) // 100) & 0xFF
        if axis == 2:
            pa = proj.copy()
            zeros = np.nonzero(pa[projwidth:projsize - projwidth] == 0)[0] + projwidth
            for i in zeros.tolist():
                if True:
                    prev, nxt = pa[i - 1], pa[i + 1]
                    above, below = pa[i - projwidth], pa[i + projwidth]
                    if prev and nxt and above and below:
                        pa[i] = (prev + nxt + above + below) // 4
            proj = pa
        out[n] = (proj & 0xFF).astype(np.uint8).reshape(projheight, projwidth)
        theta = (theta + increment) % 360
    return out


def project(doc: ImageDoc, method: str = "Brightest Point", axis: str = "Y-Axis",
            slice_spacing: float | None = None, initial: int = 0, total: int = 360,
            increment: int = 10, lower: int = 1, upper: int = 255, opacity: int = 0,
            surface_cueing: int = 0, interior_cueing: int = 50, interpolate: bool = False,
            rect=None, all_frames: bool = False, progress=None) -> ImageDoc:
    """Image > Stacks > 3D Project with the dialog's values (depth-cueing in percent as in the
    dialog: 0 = off). Works on 8/16/32-bit stacks (scaled to 8-bit with the display range), RGB
    (per channel) and the current frame (or all frames) of every channel of a hyperstack."""
    cal = doc.calibration
    pw = cal.pixel_width or 1.0
    spacing = cal.pixel_depth if slice_spacing is None else slice_spacing
    interval = spacing / pw
    if interval > 100:
        raise ValueError(f"Z spacing ({int(interval)}) is too large.")
    m = METHODS.index(method) if isinstance(method, str) else int(method)
    ax = AXES.index(axis) if isinstance(axis, str) else int(axis)
    kw = dict(method=m, axis=ax, initial=int(initial), total=int(total),
              increment=int(increment), lower=int(lower), upper=int(upper),
              opacity=int(opacity), cue_surf=100 - int(surface_cueing),
              cue_int=100 - int(interior_cueing))
    frames = range(doc.n_frames) if all_frames else [doc.position[2]]

    def one(vol):
        r = rect
        ival = interval
        if interpolate and ival > 1.0:
            box = r if r is not None else (0, 0, vol.shape[2], vol.shape[1])
            vol = _z_scale(vol, ival, box)
            ival = 1.0
            r = None
        return project_volume(vol, slice_interval=ival, rect=r, progress=progress, **kw)

    results = []
    for t in frames:
        for c in range(doc.n_channels):
            if doc.type == COLOR_RGB:
                vol = doc.pixels[t, :, 0]
                chans = [one(np.ascontiguousarray(vol[..., k])) for k in range(3)]
                results.append(np.stack(chans, axis=-1))
            else:
                results.append(one(_volume_bytes(doc, c, t)))
    nproj = results[0].shape[0]
    if doc.type == COLOR_RGB:
        arr = np.stack(results, axis=0)              # (T, N, H, W, 3) with C=1
        arr = arr[:, :, None]
        out = ImageDoc(np.ascontiguousarray(arr), COLOR_RGB,
                       "Projections of " + doc.title.rsplit(".", 1)[0])
    else:
        nc = doc.n_channels
        arr = np.stack(results).reshape((len(frames), nc) + results[0].shape)
        arr = np.moveaxis(arr, 2, 1)                 # (T, N, C, H, W)
        if len(frames) == 1 and nc > 1:
            pass
        out = ImageDoc(np.ascontiguousarray(arr), GRAY8,
                       "Projections of " + doc.title.rsplit(".", 1)[0])
        if doc.type == GRAY8 and nc == 1:
            out.luts = [doc.lut.copy()]
        elif nc > 1:
            out.luts = [lut.copy() for lut in doc.luts]
            out.composite_mode = doc.composite_mode
            out.hyperstack = True
    out.calibration = cal.copy()
    del nproj
    return out


# ---------------------------------------------------------------------------------- ortho views
def orthogonal_slices(doc: ImageDoc, x: int, y: int, c: int | None = None, t: int | None = None
                      ) -> tuple[np.ndarray, np.ndarray]:
    """XZ (rows = z) and YZ (columns = z) planes through (x, y) of the current channel/frame,
    as Image > Stacks > Orthogonal Views shows them (without z scaling)."""
    c = doc.position[0] if c is None else c
    t = doc.position[2] if t is None else t
    vol = doc.pixels[t, :, c]                       # (Z, Y, X[, 3])
    x = min(max(int(x), 0), doc.width - 1)
    y = min(max(int(y), 0), doc.height - 1)
    xz = np.ascontiguousarray(vol[:, y, :])         # (Z, X)
    yz = np.ascontiguousarray(np.swapaxes(vol[:, :, x], 0, 1))   # (Y, Z)
    return xz, yz
