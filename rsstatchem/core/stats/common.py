"""Helpers shared by all analyses: SPSS-style number formatting and data extraction."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from rsstatchem.core.dataset import NUMERIC, Dataset, Variable, format_number

# SPSS prints ".045" rather than "0.045". Flip to True for "0.045".
LEADING_ZERO = False


def fmt(value, decimals: int = 3) -> str:
    if value is None:
        return ""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(value) or math.isinf(value):
        return ""
    text = format_number(value, decimals)
    if not LEADING_ZERO:
        if text.startswith("0."):
            text = text[1:]
        elif text.startswith("-0."):
            text = "-" + text[2:]
    return text


def fmt_p(p) -> str:
    """Significance as SPSS shows it: three decimals, ``.000`` when p < .0005."""
    return fmt(p, 3)


def fmt_int(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(int(round(float(value))))


def fmt_df(value) -> str:
    """Degrees of freedom: integer when whole (pooled tests), 3 decimals for Welch-type df."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return fmt_int(value) if float(value).is_integer() else fmt(value, 3)


_SUPERSCRIPT = dict(zip("abcdefghijklmnoprstuvwxyz", "ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖʳˢᵗᵘᵛʷˣʸᶻ"))


def sup(letter: str) -> str:
    """Footnote marker as SPSS prints it (superscript letter)."""
    return _SUPERSCRIPT.get(letter, letter)


def stars(p: float) -> str:
    if p is None or math.isnan(p):
        return ""
    return "**" if p < 0.01 else "*" if p < 0.05 else ""


def numeric(ds: Dataset, name: str) -> pd.Series:
    return pd.to_numeric(ds.analysis_series(name), errors="coerce")


def listwise(ds: Dataset, names: list[str]) -> pd.DataFrame:
    """Numeric columns with every case that has a missing value on any of them dropped."""
    return pd.concat({n: numeric(ds, n) for n in names}, axis=1).dropna()


def value_text(var: Variable, value) -> str:
    label = var.label_for(value)
    if label is not None:
        return label
    if var.type == NUMERIC:
        return format_number(float(value), var.decimals)
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    return str(value)


def parse_value(var: Variable, text):
    """A group/category code typed by the user, in the variable's own type."""
    if var.type == NUMERIC:
        return float(str(text).replace(",", "."))
    return str(text)


@dataclass
class Group:
    value: object
    label: str
    data: np.ndarray

    @property
    def n(self) -> int:
        return len(self.data)


def groups_of(ds: Dataset, dependent: str, factor: str, levels: list | None = None,
              frame: pd.DataFrame | None = None) -> list[Group]:
    """Split a numeric dependent by a factor. ``levels`` restricts and orders the groups."""
    fvar = ds.var(factor)
    if frame is None:
        frame = pd.DataFrame({"y": numeric(ds, dependent), "g": ds.analysis_series(factor)})
    frame = frame.dropna()
    present = sorted(frame["g"].unique(), key=lambda v: (isinstance(v, str), v))
    order = present if levels is None else [lv for lv in levels if lv in set(present)]
    return [Group(lv, value_text(fvar, lv), frame.loc[frame["g"] == lv, "y"].to_numpy(float))
            for lv in order]


def t_interval(mean: float, se: float, df: float, level: float = 0.95) -> tuple[float, float]:
    if not se or math.isnan(se) or df <= 0:
        return float("nan"), float("nan")
    q = stats.t.ppf(0.5 + level / 2, df)
    return mean - q * se, mean + q * se


def pct(level: float) -> str:
    return f"{level * 100:g}%"


def percentile(x: np.ndarray, q) -> np.ndarray:
    """SPSS HAVERAGE percentiles (weighted average at (n+1)p)."""
    return np.percentile(np.asarray(x, float), q, method="weibull")


def trimmed_mean(x: np.ndarray, proportion: float = 0.05) -> float:
    """SPSS EXAMINE 5% trimmed mean (fractional weights at the cut points)."""
    x = np.sort(np.asarray(x, float))
    n = len(x)
    if n == 0:
        return float("nan")
    tc = proportion * n
    k1 = int(math.floor(tc)) + 1
    k2 = n - k1 + 1
    if k1 > k2:
        return float(np.median(x))
    if k1 == k2:
        return float(x[k1 - 1])
    inner = x[k1:k2 - 1].sum()
    return float(((k1 - tc) * (x[k1 - 1] + x[k2 - 1]) + inner) / (n - 2 * tc))


def lilliefors(x: np.ndarray) -> tuple[float, float, bool]:
    """K-S with Lilliefors correction, as in SPSS. Returns (D, p, p_is_lower_bound)."""
    from statsmodels.stats.diagnostic import lilliefors as _lf

    x = np.asarray(x, float)
    if len(x) < 4 or np.ptp(x) == 0:
        return float("nan"), float("nan"), False
    d, p = _lf(x, dist="norm", pvalmethod="table")
    lower_bound = p >= 0.2 - 1e-12
    return float(d), float(min(p, 0.2) if lower_bound else p), bool(lower_bound)


def shapiro(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, float)
    if len(x) < 3 or len(x) > 5000 or np.ptp(x) == 0:
        return float("nan"), float("nan")
    w, p = stats.shapiro(x)
    return float(w), float(p)


def levene_rows(samples: list[np.ndarray]) -> list[tuple[str, float, int, int, float]]:
    """Levene's test rows like SPSS: based on mean, median and 5% trimmed mean."""
    samples = [np.asarray(s, float) for s in samples if len(s) > 0]
    k = len(samples)
    n = sum(len(s) for s in samples)
    rows = []
    if k < 2 or n - k < 1:
        return rows
    for title, center in (("Based on Mean", np.mean), ("Based on Median", np.median),
                          ("Based on trimmed mean", trimmed_mean)):
        deviations = [np.abs(s - center(s)) for s in samples]
        if all(np.ptp(d) == 0 for d in deviations) and \
                len({float(d[0]) for d in deviations}) == 1:
            stat, p = float("nan"), float("nan")
        else:
            stat, p = stats.f_oneway(*deviations)
        rows.append((title, float(stat), k - 1, n - k, float(p)))
    return rows
