"""Download-at-test-time helper for SPSS reference datasets that may not be committed.

The UCLA OARC data files (hsb2.sav, ...) carry no redistribution licence, so they are never
stored in the repository. A test asks for a file by its published URL; the file is downloaded
once into a cache directory and read with pyreadstat. When the file is not cached and the
network is unavailable the test is skipped (never failed).

Cache directory: ``$RSSTATCHEM_SPSS_REF_CACHE`` if set, else ``~/.cache/rsstatchem/spss_ref``.
Set ``RSSTATCHEM_SPSS_REF_OFFLINE=1`` to never touch the network (cached files are still used).
"""
from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

import pytest

UCLA = "https://stats.oarc.ucla.edu/wp-content/uploads/2016/02/"
UCLA_STAT = "https://stats.oarc.ucla.edu/stat/data/"


def cache_dir() -> Path:
    d = os.environ.get("RSSTATCHEM_SPSS_REF_CACHE")
    p = Path(d) if d else Path.home() / ".cache" / "rsstatchem" / "spss_ref"
    return p


def _cached_name(url: str) -> str:
    base = url.rstrip("/").rsplit("/", 1)[-1] or "data"
    tag = hashlib.sha1(url.encode()).hexdigest()[:8]
    return f"{tag}_{base}"


def fetch(url: str, timeout: float = 30.0) -> Path:
    """Local path of the file at ``url`` (downloaded once); ``pytest.skip`` when unavailable."""
    target = cache_dir() / _cached_name(url)
    if target.exists() and target.stat().st_size > 0:
        return target
    if os.environ.get("RSSTATCHEM_SPSS_REF_OFFLINE"):
        pytest.skip(f"offline and not cached: {url}")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "rsstatchem-tests"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
        if not data:
            raise OSError("empty download")
        tmp = target.with_suffix(target.suffix + ".part")
        tmp.write_bytes(data)
        tmp.replace(target)
    except Exception as exc:  # network down, DNS, HTTP error, read-only home ...
        pytest.skip(f"cannot download {url}: {exc}")
    return target


_FRAMES: dict = {}


def read_sav(url: str, *, lower: bool = True):
    """The SPSS .sav file at ``url`` as a pandas DataFrame (column names lower-cased)."""
    key = (url, lower)
    if key not in _FRAMES:
        pyreadstat = pytest.importorskip("pyreadstat")
        df, _meta = pyreadstat.read_sav(str(fetch(url)))
        if lower:
            df.columns = [str(c).lower() for c in df.columns]
        _FRAMES[key] = df
    return _FRAMES[key].copy()


def hsb2():
    """UCLA hsb2 (200 cases; the 2016/02 upload, byte-identical to /stat/data/hsb2.sav)."""
    return read_sav(UCLA + "hsb2.sav")


def spss_fmt(x: float, decimals: int) -> str:
    """Format like an SPSS pivot table cell: no leading zero ('.597', '-.053')."""
    s = f"{x:.{decimals}f}"
    if s.startswith("0.") and decimals > 0:
        s = s[1:]
    elif s.startswith("-0.") and decimals > 0:
        s = "-" + s[2:]
    if s in ("-.000", "-0", "-.00", "-.0") or (s.startswith("-") and float(s) == 0):
        s = s[1:]
    return s


def assert_printed(value: float, printed: str, msg: str = "") -> None:
    """``value`` rounded to the decimals of the SPSS-printed string equals that string.

    Accepts thousands separators, '$' and trailing footnote letters like '(a)' in ``printed``.
    A difference of one unit in the last printed digit is tolerated only when the value sits
    within 1e-9 of the rounding boundary (half-way cases)."""
    p = printed.strip().replace(",", "").replace("$", "")
    if "(" in p:
        p = p[: p.index("(")]
    p = p.rstrip("*abcdefghijklmnopqrstuvwxyz")
    dec = len(p.split(".")[1]) if "." in p else 0
    got = spss_fmt(float(value), dec)
    want = p if not p.startswith("0.") else p[1:]
    if want.startswith("-0."):
        want = "-" + want[2:]
    if got != want:
        step = 10.0 ** (-dec)
        if abs(abs(float(value) - float(p)) - step / 2) < 1e-9 + 1e-12 * abs(float(p)):
            return
        raise AssertionError(f"{msg}: got {float(value)!r} -> {got!r}, SPSS printed {printed!r}")
