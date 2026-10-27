"""Small helpers shared by the survival, missing-data and time-series analyses (WP4).

Figures follow the project rule: matplotlib object API only (never ``pyplot``) rendered to
PNG bytes. Contrast codings reproduce the SPSS ``/CONTRAST`` options of COXREG and LOGISTIC
REGRESSION (Indicator, Simple, Difference, Helmert, Repeated, Polynomial, Deviation).
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from rsstatchem.core.dataset import NUMERIC, Dataset
from rsstatchem.core.output import Section
from rsstatchem.core.stats.common import numeric, value_text

COLORS = ["#1f4e9c", "#c8102e", "#2e8b57", "#d48a00", "#6a3d9a", "#00838f", "#8c564b",
          "#e377c2", "#7f7f7f", "#17becf"]

CONTRASTS = {"indicator": "Indicator", "simple": "Simple", "difference": "Difference",
             "helmert": "Helmert", "repeated": "Repeated", "polynomial": "Polynomial",
             "deviation": "Deviation"}


@dataclass
class Result:
    """An analysis result with SPSS-style saved variables (``BASE`` -> (label, series))."""
    section: Section
    saved: dict = field(default_factory=dict)
    models: list = field(default_factory=list)   # resample.ModelInfo (Bootstrap, pooling)


def figure(width: float = 6.0, height: float = 4.2) -> Figure:
    fig = Figure(figsize=(width, height), dpi=110)
    FigureCanvasAgg(fig)
    return fig


def png(fig: Figure) -> bytes:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", facecolor="white")
    return buf.getvalue()


def sort_key(v):
    return (isinstance(v, str), v)


def raw_series(ds: Dataset, name: str) -> pd.Series:
    """Numeric variables as float (user-missing -> NaN); string variables as text."""
    if ds.var(name).type == NUMERIC:
        return numeric(ds, name)
    return ds.analysis_series(name)


def event_mask(values: pd.Series, spec) -> pd.Series:
    """SPSS "Define Event for Status Variable": ("single", v) | ("range", lo, hi) |
    ("list", [v, ...]). Returns 1.0 for the event, 0.0 for censored, NaN where missing."""
    kind = spec[0]
    s = values
    if kind == "single":
        hit = s == spec[1]
    elif kind == "range":
        hit = (s >= spec[1]) & (s <= spec[2])
    else:
        hit = s.isin(list(spec[1]))
    out = hit.astype(float)
    out[s.isna()] = np.nan
    return out


def describe_event(spec) -> str:
    def f(v):
        return f"{v:g}" if isinstance(v, (int, float)) else str(v)
    if spec[0] == "single":
        return f"({f(spec[1])})"
    if spec[0] == "range":
        return f"({f(spec[1])} THRU {f(spec[2])})"
    return "(" + " ".join(f(v) for v in spec[1]) + ")"


# ---------------------------------------------------------------------- contrast coding
def contrast_matrix(k: int, contrast: str, reference: str = "last") -> np.ndarray:
    """Design coding (k categories x k-1 columns) as SPSS prints in "Categorical Variable
    Codings"."""
    if k < 2:
        return np.zeros((k, 0))
    ref = k - 1 if reference == "last" else 0
    others = [i for i in range(k) if i != ref]
    if contrast == "indicator":
        m = np.zeros((k, k - 1))
        for j, r in enumerate(others):
            m[r, j] = 1.0
        return m
    if contrast == "deviation":
        m = np.zeros((k, k - 1))
        for j, r in enumerate(others):
            m[r, j] = 1.0
        m[ref, :] = -1.0
        return m
    if contrast == "simple":
        m = np.full((k, k - 1), -1.0 / k)
        for j, r in enumerate(others):
            m[r, j] += 1.0
        return m
    contrast_rows = np.zeros((k - 1, k))
    if contrast == "difference":      # each category vs the mean of the previous ones
        for i in range(1, k):
            contrast_rows[i - 1, :i] = -1.0 / i
            contrast_rows[i - 1, i] = 1.0
    elif contrast == "helmert":       # each category vs the mean of the subsequent ones
        for i in range(k - 1):
            contrast_rows[i, i] = 1.0
            contrast_rows[i, i + 1:] = -1.0 / (k - i - 1)
    elif contrast == "repeated":      # each category vs the next
        for i in range(k - 1):
            contrast_rows[i, i], contrast_rows[i, i + 1] = 1.0, -1.0
    elif contrast == "polynomial":
        x = np.arange(1, k + 1, dtype=float)
        v = np.vander(x - x.mean(), k, increasing=True)
        q, _ = np.linalg.qr(v)
        poly = q[:, 1:]
        poly *= np.sign(poly[-1, :])
        return poly
    else:
        raise ValueError(f"Unknown contrast {contrast!r}")
    return contrast_rows.T @ np.linalg.inv(contrast_rows @ contrast_rows.T)


@dataclass
class Coding:
    name: str
    categories: list
    labels: list[str]
    matrix: np.ndarray
    contrast: str
    reference: str

    @property
    def columns(self) -> list[str]:
        return [f"{self.name}({j + 1})" for j in range(self.matrix.shape[1])]

    def encode(self, values: pd.Series) -> pd.DataFrame:
        index = {c: i for i, c in enumerate(self.categories)}
        out = np.full((len(values), self.matrix.shape[1]), np.nan)
        for r, v in enumerate(values.to_numpy(dtype=object)):
            i = index.get(v)
            if i is not None:
                out[r] = self.matrix[i]
        return pd.DataFrame(out, index=values.index, columns=self.columns)


def make_coding(ds: Dataset, name: str, values: pd.Series, contrast: str = "indicator",
                reference: str = "last") -> Coding:
    var = ds.var(name)
    cats = sorted(values.dropna().unique(), key=sort_key)
    return Coding(name, cats, [value_text(var, c) for c in cats],
                  contrast_matrix(len(cats), contrast, reference), contrast, reference)


def fmt_num(x, decimals: int = 3) -> str:
    from rsstatchem.core.stats.common import fmt

    return fmt(x, decimals)
