"""WAV audio to a worksheet (time + one column per channel, ``scipy.io.wavfile``) and images
to a matrix (Pillow; grayscale by luminance, or one matrix per RGB channel)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from rsstatchem.core.analysis.importers import ImportedTable, ImportFailed, numeric_column


def read_wav(path: str | Path, max_points: int = 5_000_000) -> ImportedTable:
    from scipy.io import wavfile
    p = Path(path)
    try:
        rate, data = wavfile.read(str(p))
    except Exception as exc:  # noqa: BLE001
        raise ImportFailed(f"Cannot read the WAV file: {exc}") from None
    data = np.asarray(data)
    if data.dtype.kind in "iu":   # scale integers to [-1, 1] like Origin's normalized import
        info = np.iinfo(data.dtype)
        data = (data.astype(float) - (info.max + info.min + 1) / 2) / ((info.max - info.min + 1)
                                                                       / 2)
    data = data.reshape(len(data), -1)[:max_points]
    t = np.arange(len(data)) / float(rate)
    cols = [numeric_column("Time", t, "s", "X")]
    for ch in range(data.shape[1]):
        cols.append(numeric_column(f"Channel {ch + 1}", data[:, ch], "", "Y",
                                   f"{rate} Hz"))
    return ImportedTable(p.stem, cols, {"rate": int(rate), "source_format": "WAV"})


def read_image(path: str | Path, mode: str = "gray") -> ImportedTable:
    """Image → matrix (row 1 = top of the image, as Origin's image matrices)."""
    from PIL import Image
    p = Path(path)
    try:
        with Image.open(p) as im:
            im.load()
            if mode == "gray":
                arr = np.asarray(im.convert("L"), dtype=float)
            else:
                arr = np.asarray(im.convert("RGB"), dtype=float)[:, :, "rgb".index(mode[0])]
    except Exception as exc:  # noqa: BLE001
        raise ImportFailed(f"Cannot read the image: {exc}") from None
    r, c = arr.shape
    return ImportedTable(p.stem, [], {"source_format": "image", "mode": mode},
                         kind="matrix", matrix=arr, mapping=(1.0, float(c), 1.0, float(r)))
