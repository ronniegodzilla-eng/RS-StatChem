"""Shared helpers for the Analysis package: Origin-style number formatting, (x, y) data
extraction from a Dataset, and matplotlib (Agg, object-oriented) figure helpers."""
from __future__ import annotations

import io
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.output import Section

DATA = "#1f1f1f"
FIT = "#C62828"
BAND = "#EF9A9A"
PRED = "#90A4AE"
ACCENT = "#1565C0"
PALETTE = ["#1565C0", "#2E7D32", "#EF6C00", "#6A1B9A", "#00838F", "#AD1457", "#4E342E",
           "#9E9D24", "#283593"]


@dataclass
class AnalysisResult:
    """What a dialog's ``compute()`` returns when it also saves new variables."""
    section: Section
    saved: dict[str, tuple[str, pd.Series]] = field(default_factory=dict)


# ---------------------------------------------------------------------- formatting
def num(value, sig: int = 6) -> str:
    """Origin-style number: 6 significant digits, scientific (``1.23457E-5``) when tiny/huge."""
    if value is None:
        return ""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(v):
        return "--"
    if math.isinf(v):
        return "∞" if v > 0 else "-∞"
    if v == 0:
        return "0"
    a = abs(v)
    if 1e-4 <= a < 1e7:
        text = f"{v:.{sig}g}"
        if "e" in text:
            text = f"{v:.{sig}f}".rstrip("0").rstrip(".")
        return text
    mant, exp = f"{v:.{sig - 1}E}".split("E")
    mant = mant.rstrip("0").rstrip(".")
    return f"{mant}E{int(exp)}"


def pval(p) -> str:
    """Probabilities as Origin prints them (``0.01234``, ``1.2345E-12``)."""
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "--"
    p = float(p)
    if p == 0:
        return "0"
    if p < 1e-4:
        mant, exp = f"{p:.4E}".split("E")
        return f"{mant.rstrip('0').rstrip('.')}E{int(exp)}"
    return f"{p:.5g}"


def intf(v) -> str:
    return str(int(v))


def parse_numbers(text: str) -> list[float]:
    """Comma / semicolon / whitespace separated numbers (decimal comma accepted with ';')."""
    out = []
    if not text:
        return out
    seps = text.replace(";", " ").replace("\t", " ").replace("\n", " ")
    tokens = seps.split(",") if ";" not in text else seps.split()
    for chunk in tokens:
        for tok in chunk.split():
            tok = tok.strip()
            if not tok:
                continue
            try:
                out.append(float(tok if ";" not in text else tok.replace(",", ".")))
            except ValueError as exc:
                raise ValueError(f"'{tok}' is not a number.") from exc
    return out


# ---------------------------------------------------------------------- data
def column(ds: Dataset, name: str | None) -> pd.Series | None:
    if not name:
        return None
    return pd.to_numeric(ds.analysis_series(name), errors="coerce").astype(float)


def var_title(ds: Dataset, name: str | None, default: str = "Row index") -> str:
    if not name:
        return default
    v = ds.var(name)
    return v.label or v.name


def xy_frame(ds: Dataset, xname: str | None, ynames: list[str],
             extra: list[str] | None = None) -> pd.DataFrame:
    """Columns ``x`` (row number 1..n when ``xname`` is None) and the y names, complete cases
    only, index = Dataset row index so results can be saved back."""
    cols = {"x": column(ds, xname) if xname else
            pd.Series(np.arange(1, ds.n_cases + 1, dtype=float), index=ds.df.index)}
    for n in ynames + (extra or []):
        cols[n] = column(ds, n)
    return pd.DataFrame(cols).dropna()


def expand(values: np.ndarray, index, ds: Dataset) -> pd.Series:
    """A result computed on some rows → a Series over all Dataset rows (NaN elsewhere)."""
    return pd.Series(np.asarray(values, float), index=index).reindex(ds.df.index)


# ---------------------------------------------------------------------- plotting
def figure(width: float = 6.0, height: float = 4.0) -> Figure:
    fig = Figure(figsize=(width, height), dpi=120)
    FigureCanvasAgg(fig)
    return fig


def style(ax, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.grid(True, color="#e6e6e6", linewidth=0.6)
    ax.set_axisbelow(True)


def png(fig: Figure) -> bytes:
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    return buf.getvalue()


def scatter(ax, x, y, label="Data", **kw) -> None:
    n = len(x)
    size = 22 if n <= 60 else 10 if n <= 400 else 4
    ax.scatter(x, y, s=size, color=kw.pop("color", DATA), zorder=3, label=label,
               edgecolors="none", **kw)


def sorted_xy(x, y):
    order = np.argsort(x, kind="stable")
    return np.asarray(x, float)[order], np.asarray(y, float)[order]
