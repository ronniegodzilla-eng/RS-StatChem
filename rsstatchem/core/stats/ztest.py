"""One-sample Z test for a mean when the population standard deviation is known.

SPSS has no such procedure; this follows Minitab's Stat > Basic Statistics > 1-Sample Z:
the standard error is sigma / sqrt(n) with the KNOWN sigma (the sample standard deviation is
only reported), Z = (mean - test value) / SE, the p-value comes from the standard normal
distribution for the chosen alternative, and the interval for the mean is two-sided for
"mean != test value" or a one-sided bound for a directional alternative. Tables keep the
SPSS look of the other Compare Means procedures. Cohen's d uses the known sigma.
"""
from __future__ import annotations

import math

import pandas as pd
from scipy import stats

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import fmt, fmt_int, fmt_p, numeric, pct

ALTERNATIVES = {"two-sided": "≠", "less": "<", "greater": ">"}


def z_statistics(n: float, mean: float, sigma: float, test_value: float = 0.0,
                 level: float = 0.95, alternative: str = "two-sided") -> dict:
    """SE, Z, p and the interval (or bound) for the mean; ``lower``/``upper`` are None on the
    open side of a one-sided bound."""
    if alternative not in ALTERNATIVES:
        raise ValueError(f"Unknown alternative {alternative!r}.")
    if not sigma > 0:
        raise ValueError("The known standard deviation must be greater than 0.")
    if not n >= 1:
        raise ValueError("The sample size must be at least 1.")
    se = sigma / math.sqrt(n)
    z = (mean - test_value) / se
    if alternative == "two-sided":
        p = 2 * stats.norm.sf(abs(z))
        q = stats.norm.ppf(0.5 + level / 2)
        lower, upper = mean - q * se, mean + q * se
    elif alternative == "greater":
        p = stats.norm.sf(z)
        lower, upper = mean - stats.norm.ppf(level) * se, None
    else:
        p = stats.norm.cdf(z)
        lower, upper = None, mean + stats.norm.ppf(level) * se
    return {"n": n, "mean": mean, "se": se, "z": z, "p": float(p), "lower": lower,
            "upper": upper, "diff": mean - test_value, "d": (mean - test_value) / sigma}


def _tables(section: Section, rows: list[tuple], sigma: float, test_value: float,
            level: float, alternative: str) -> None:
    """``rows``: (label, decimals, n, mean, sample sd or None, statistics dict or None)."""
    symbol = ALTERNATIVES[alternative]
    if alternative == "two-sided":
        ci = f"{pct(level)} Confidence Interval for Mean"
        ci_cols = [(ci, "Lower"), (ci, "Upper")]
    elif alternative == "greater":
        ci_cols = [f"{pct(level)} Lower Bound for Mean"]
    else:
        ci_cols = [f"{pct(level)} Upper Bound for Mean"]
    srows, trows = [], []
    for label, d, n, mean, sd, st in rows:
        if st is None:
            srows.append([label, fmt_int(n)] + [""] * (3 + len(ci_cols)))
            continue
        bounds = [fmt(v, d + 3) for v in (st["lower"], st["upper"]) if v is not None]
        srows.append([label, fmt_int(n), fmt(mean, d + 2),
                      fmt(sd, d + 3) if sd is not None else "", fmt(st["se"], d + 3)] + bounds)
        trows.append([label, fmt(st["z"]), fmt_p(st["p"]), fmt(st["diff"], d + 3),
                      fmt(st["d"])])
    section.table(Table("One-Sample Statistics", ["", "N", "Mean", "Std. Deviation",
                                                  "Std. Error Meanᵃ"] + ci_cols, srows,
                        footnotes=["a. Uses the known standard deviation = "
                                   f"{sigma:g} (Std. Error Mean = {sigma:g} / √N)."]))
    group = f"Test Value = {test_value:g}"
    sided = "Two-Sided p" if alternative == "two-sided" else "One-Sided p"
    section.table(Table("One-Sample Z Test", [""] + [
        (group, "Z\u1d43"), (group, sided), (group, "Mean Difference"), (group, "Cohen's dᵇ")],
        trows, footnotes=[
            f"a. Null hypothesis: mean = {test_value:g}. Alternative hypothesis: mean "
            f"{symbol} {test_value:g}.",
            f"b. Mean difference divided by the known standard deviation ({sigma:g})."]))


def _log(section: Section, what: str, sigma: float, test_value: float, level: float,
         alternative: str) -> None:
    section.log(f"* One-Sample Z Test (known standard deviation).\nZTEST {what}\n"
                f"  /TESTVAL={test_value:g}\n  /SIGMA={sigma:g}\n"
                f"  /ALTERNATIVE={alternative.upper()}\n  /CRITERIA=CI({level:.2f}).")


def one_sample(ds: Dataset, names: list[str], sigma: float, test_value: float = 0.0,
               level: float = 0.95, alternative: str = "two-sided",
               listwise: bool = False) -> Section:
    """Z test of each variable in ``names`` against ``test_value`` with known ``sigma``."""
    z_statistics(1, 0.0, sigma, 0.0, level, alternative)       # validates the arguments
    section = Section("Z-Test")
    _log(section, "VARIABLES=" + " ".join(names), sigma, test_value, level, alternative)
    data = {n: numeric(ds, n) for n in names}
    if listwise:
        frame = pd.DataFrame(data).dropna()
        data = {n: frame[n] for n in names}
    rows = []
    for name in names:
        var = ds.var(name)
        x = data[name].dropna().to_numpy(float)
        n = len(x)
        if n < 1:
            rows.append((var.title, var.decimals, 0, None, None, None))
            continue
        mean = float(x.mean())
        sd = float(x.std(ddof=1)) if n > 1 else None
        rows.append((var.title, var.decimals, n, mean, sd,
                     z_statistics(n, mean, sigma, test_value, level, alternative)))
    _tables(section, rows, sigma, test_value, level, alternative)
    return section


def summary_one_sample(n: float, mean: float, sigma: float, test_value: float = 0.0,
                       level: float = 0.95, alternative: str = "two-sided",
                       label: str = "Sample") -> Section:
    """The same test from summarized data (sample size and sample mean)."""
    st = z_statistics(n, mean, sigma, test_value, level, alternative)
    section = Section("Z-Test")
    _log(section, f"SUMMARY N={n:g} MEAN={mean:g}", sigma, test_value, level, alternative)
    _tables(section, [(label, 2, n, mean, None, st)], sigma, test_value, level, alternative)
    return section
