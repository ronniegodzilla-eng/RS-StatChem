"""Analyze > Forecasting and the time-series transforms (SPSS Forecasting module).

* Define Dates (Data menu) and Create Time Series (Transform menu: DIFF, SDIFF, PMA, MA, RMED,
  CSUM, LAG, LEAD, T4253H smoothing).
* Sequence Charts, Autocorrelations (ACF with Box-Ljung, PACF; standard errors under the
  independence model, as SPSS's default, or Bartlett's approximation), Cross-Correlations.
* Seasonal Decomposition (SPSS SEASON: ratio-to-moving-average / difference, "all points
  equal" or "endpoints weighted by .5"; seasonal factors and saved ERR/SAS/SAF/STC).
* Spectral Analysis (periodogram and smoothed spectral density with Tukey-Hamming, Tukey,
  Parzen, Bartlett, Daniell windows).
* Create Traditional Models: Exponential Smoothing (Simple, Holt, Brown, Damped trend, Simple
  seasonal, Winters' additive and multiplicative; statsmodels ETS state space), ARIMA/SARIMA
  with predictors and transformation (statsmodels ARIMA), and an Expert Modeler that searches
  ETS and ARIMA candidates. Model statistics follow SPSS (stationary R-squared, R-squared, RMSE,
  MAPE, MAE, MaxAPE, MaxAE, normalized BIC, Ljung-Box Q(18)); SPSS's MA sign convention is used
  (MA parameters are printed with the opposite sign to statsmodels / R).
* Apply Traditional Models: re-use stored model specifications/parameters for new forecasts.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import signal, stats

from rsstatchem.core.dataset import NUMERIC, Dataset
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import fmt, fmt_int, fmt_p, numeric
from rsstatchem.core.stats.wp4_util import COLORS, Result, figure, png


def _f(x, d: int = 3) -> str:
    return fmt(x, d)


# =========================================================================== dates
DATE_KINDS = {
    "years": ("Years", [("YEAR_", None)], 1),
    "years_quarters": ("Years, quarters", [("YEAR_", None), ("QUARTER_", 4)], 4),
    "years_months": ("Years, months", [("YEAR_", None), ("MONTH_", 12)], 12),
    "weeks_days": ("Weeks, days", [("WEEK_", None), ("DAY_", 7)], 7),
    "weeks_workdays": ("Weeks, work days(5)", [("WEEK_", None), ("DAY_", 5)], 5),
    "days_hours": ("Days, hours", [("DAY_", None), ("HOUR_", 24)], 24),
    "cases": ("Cases are not dated", [], 1),
}


def periodicity(ds: Dataset) -> int:
    return int(ds.meta.get("dates", {}).get("period", 1) or 1)


def define_dates(ds: Dataset, kind: str, start: list[int]) -> Section:
    """Creates YEAR_ / QUARTER_ / MONTH_ ... and a string DATE_ variable and stores the
    periodicity in ``ds.meta["dates"]`` (SPSS DATE command). Modifies ``ds``."""
    title, parts, period = DATE_KINDS[kind]
    sec = Section("Define Dates")
    for name in ["YEAR_", "QUARTER_", "MONTH_", "WEEK_", "DAY_", "HOUR_", "DATE_"]:
        if ds.has_name(name):
            ds.delete_variables([ds.index_of(name)])
    if kind == "cases":
        ds.meta.pop("dates", None)
        sec.log("DATE NONE.")
        sec.note("", "Date variables removed; cases are not dated.")
        return sec
    n = ds.n_cases
    first = start[0]
    if len(parts) == 1:
        major = first + np.arange(n)
        cols = {parts[0][0]: major}
        labels = [str(int(v)) for v in major]
    else:
        minor0 = start[1] if len(start) > 1 else 1
        base = (minor0 - 1) + np.arange(n)
        if kind == "days_hours":
            base = minor0 + np.arange(n)
            major = first + base // period
            minor = base % period
        else:
            major = first + base // period
            minor = base % period + 1
        cols = {parts[0][0]: major, parts[1][0]: minor}
        if kind == "years_quarters":
            labels = [f"Q{int(b)} {int(a)}" for a, b in zip(major, minor)]
        elif kind == "years_months":
            months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT",
                      "NOV", "DEC"]
            labels = [f"{months[int(b) - 1]} {int(a)}" for a, b in zip(major, minor)]
        else:
            labels = [f"{int(a)} {int(b)}" for a, b in zip(major, minor)]
    for name, values in cols.items():
        var = ds.add_variable(name, values=values.astype(float))
        var.decimals = 0
        var.label = name.rstrip("_").capitalize() + (", period " + str(period)
                                                     if name != parts[0][0] else ", not periodic")
    var = ds.add_variable("DATE_", vtype="string", values=labels)
    var.label = "Date.  Format:  " + title
    ds.meta["dates"] = {"kind": kind, "period": period, "start": list(start)}
    sec.log(f"DATE {' '.join(p[0].rstrip('_') for p in parts)} {' '.join(map(str, start))}.")
    sec.table(Table("Define Dates", ["Variable", "Label"],
                    [[v.name, v.label] for v in ds.variables if v.name in
                     list(cols) + ["DATE_"]],
                    footnotes=[f"Periodicity: {period}"]))
    return sec


def date_labels(ds: Dataset) -> list[str] | None:
    if ds.has_name("DATE_"):
        return [str(v) for v in ds.df[ds.var("DATE_").name].to_numpy(dtype=object)]
    return None


# =========================================================================== create series
FUNCTIONS = {"DIFF": "Difference", "SDIFF": "Seasonal difference", "PMA": "Prior moving average",
             "MA": "Centered moving average", "RMED": "Running medians", "CSUM": "Cumulative sum",
             "LAG": "Lag", "LEAD": "Lead", "T4253H": "Smoothing"}


def _running_median(x: np.ndarray, span: int) -> np.ndarray:
    s = pd.Series(x)
    if span % 2 == 1:
        return s.rolling(span, center=True).median().to_numpy()
    med = s.rolling(span).median().to_numpy()       # even span: average of adjacent medians
    out = np.full(len(x), np.nan)
    h = span // 2
    out[h:len(x) - h + 1] = (med[span - 1:] + np.r_[med[span:], np.nan])[:len(x) - span + 1] / 2
    return out


def t4253h(x: np.ndarray) -> np.ndarray:
    """SPSS T4253H compound smoother: running medians of 4 (recentered by 2), 5 and 3,
    hanning, then the same smoothing applied to the residuals (twicing)."""
    def smooth(v):
        n = len(v)
        a = pd.Series(v).rolling(4).median().to_numpy()
        b = np.full(n, np.nan)
        c = (a[3:] + np.r_[a[4:], np.nan]) / 2
        b[2:2 + len(c)] = c
        b[:2] = v[:2]
        b[-1] = v[-1]
        b = np.where(np.isnan(b), v, b)
        b = pd.Series(b).rolling(5, center=True, min_periods=1).median().to_numpy()
        b = pd.Series(b).rolling(3, center=True, min_periods=1).median().to_numpy()
        h = b.copy()
        h[1:-1] = 0.25 * b[:-2] + 0.5 * b[1:-1] + 0.25 * b[2:]
        return h
    s1 = smooth(np.asarray(x, float))
    return s1 + smooth(np.asarray(x, float) - s1)


def series_function(x: np.ndarray, func: str, order: int = 1, span: int = 3,
                    period: int = 12) -> np.ndarray:
    x = np.asarray(x, float)
    s = pd.Series(x)
    if func == "DIFF":
        out = x.copy()
        for _ in range(order):
            out = np.r_[np.nan, np.diff(out)]
        return out
    if func == "SDIFF":
        out = x.copy()
        for _ in range(order):
            out = np.r_[np.full(period, np.nan), out[period:] - out[:-period]]
        return out
    if func == "PMA":
        return s.rolling(span).mean().shift(1).to_numpy()
    if func == "MA":
        if span % 2 == 1:
            return s.rolling(span, center=True).mean().to_numpy()
        w = np.r_[0.5, np.ones(span - 1), 0.5] / span   # 2 x span centered average
        out = np.convolve(x, w, mode="same")
        h = span // 2
        out[:h] = np.nan
        out[len(x) - h:] = np.nan
        return out
    if func == "RMED":
        return _running_median(x, span)
    if func == "CSUM":
        return np.nancumsum(x) + np.where(np.isnan(x), np.nan, 0)
    if func == "LAG":
        return s.shift(order).to_numpy()
    if func == "LEAD":
        return s.shift(-order).to_numpy()
    if func == "T4253H":
        return t4253h(x)
    raise ValueError(func)


def create_time_series(ds: Dataset, specs: list[tuple]) -> Section:
    """``specs``: (new name, function, variable, order, span, period). Modifies ``ds``."""
    sec = Section("Create Time Series")
    lines = []
    rows = []
    for new, func, var, order, span, period in specs:
        x = numeric(ds, var).to_numpy(float)
        y = np.full(len(x), np.nan)
        valid = np.nonzero(~np.isnan(x))[0]
        if len(valid):
            lo, hi = valid[0], valid[-1] + 1
            y[lo:hi] = series_function(x[lo:hi], func, order, span, period)
        if ds.has_name(new):
            ds.delete_variables([ds.index_of(new)])
        v = ds.add_variable(new, values=y)
        arg = {"DIFF": f"{var} {order}", "SDIFF": f"{var} {order}", "LAG": f"{var} {order}",
               "LEAD": f"{var} {order}", "PMA": f"{var} {span}", "MA": f"{var} {span}",
               "RMED": f"{var} {span}"}.get(func, var)
        v.label = f"{func}({arg})"
        v.decimals = ds.var(var).decimals
        lines.append(f"  /{new}={func}({arg})")
        valid = np.nonzero(~np.isnan(y))[0]
        rows.append([new, fmt_int(valid[0] + 1) if len(valid) else "",
                     fmt_int(valid[-1] + 1) if len(valid) else "", fmt_int(len(valid)),
                     f"{func}({arg})"])
    sec.log("CREATE\n" + "\n".join(lines) + ".")
    sec.table(Table("Created Series", ["Series Name",
                                       ("Case Number of Non-Missing Values", "First"),
                                       ("Case Number of Non-Missing Values", "Last"),
                                       "N of Valid Cases", "Creating Function"], rows))
    return sec


# =========================================================================== transforms
def transform_series(x: np.ndarray, log: bool = False, diff: int = 0, sdiff: int = 0,
                     period: int = 1) -> np.ndarray:
    y = np.asarray(x, float)
    if log:
        with np.errstate(divide="ignore", invalid="ignore"):
            y = np.where(y > 0, np.log(y), np.nan)
    for _ in range(diff):
        y = np.r_[np.nan, np.diff(y)]
    for _ in range(sdiff):
        if period > 1:
            y = np.r_[np.full(period, np.nan), y[period:] - y[:-period]]
    return y


def _trim_missing(y: np.ndarray) -> np.ndarray:
    ok = ~np.isnan(y)
    if not ok.any():
        return y[:0]
    idx = np.nonzero(ok)[0]
    return y[idx[0]:idx[-1] + 1]


def _transform_text(log, diff, sdiff, period):
    parts = []
    if log:
        parts.append("Natural Logarithm")
    if diff:
        parts.append(f"Difference ({diff})")
    if sdiff:
        parts.append(f"Seasonal Difference ({sdiff}, period {period})")
    return ", ".join(parts) or "None"


def sequence_chart(ds: Dataset, names: list[str], time_axis: str | None = None,
                   one_chart: bool = True, log: bool = False, diff: int = 0, sdiff: int = 0,
                   mean_line: bool = False) -> Section:
    period = periodicity(ds)
    sec = Section("Sequence Charts")
    sec.log(f"TSPLOT VARIABLES={' '.join(names)}" +
            (f"\n  /ID={time_axis}" if time_axis else "") +
            ("\n  /LN" if log else "") + (f"\n  /DIFF={diff}" if diff else "") +
            (f"\n  /SDIFF={sdiff}" if sdiff else "") + ".")
    rows = [["Series or Sequence", ", ".join(names)],
            ["Transformation", _transform_text(log, diff, sdiff, period)],
            ["Time Axis Labels", time_axis or ("Date" if date_labels(ds) else "Sequence numbers")]]
    sec.table(Table("Model Description", ["", ""], rows))
    labels = None
    if time_axis:
        labels = [str(v) for v in ds.analysis_series(time_axis).to_numpy(dtype=object)]
    elif date_labels(ds):
        labels = date_labels(ds)
    x = np.arange(1, ds.n_cases + 1)
    groups = [names] if one_chart else [[n] for n in names]
    for grp in groups:
        fig = figure(7.0, 3.8)
        ax = fig.add_subplot(111)
        for i, name in enumerate(grp):
            y = transform_series(numeric(ds, name).to_numpy(float), log, diff, sdiff, period)
            ax.plot(x, y, color=COLORS[i % len(COLORS)], lw=1.2, label=ds.var(name).title)
            if mean_line:
                ax.axhline(np.nanmean(y), color=COLORS[i % len(COLORS)], lw=0.8, ls="--")
        if labels:
            step = max(1, len(x) // 10)
            ax.set_xticks(x[::step])
            ax.set_xticklabels(labels[::step], rotation=45, ha="right", fontsize=7)
        ax.set_xlabel(time_axis or ("Date" if labels else "Sequence number"))
        ax.grid(alpha=0.25)
        if len(grp) > 1:
            ax.legend(frameon=False, fontsize=8)
        else:
            ax.set_ylabel(ds.var(grp[0]).title)
        sec.image(", ".join(grp), png(fig), width=640)
    return sec


# =========================================================================== ACF / PACF / CCF
def acf_values(y: np.ndarray, nlags: int) -> np.ndarray:
    y = np.asarray(y, float)
    y = y - y.mean()
    n = len(y)
    c0 = y @ y / n
    return np.array([(y[:n - k] @ y[k:]) / n / c0 for k in range(1, nlags + 1)])


def acf_table(y: np.ndarray, nlags: int, se_method: str = "independence"):
    """(lag, r, se, Q, df, p) with SPSS standard errors: independence model
    sqrt((n-k)/(n(n+2))) or Bartlett sqrt((1 + 2 sum_{j<k} r_j^2)/n)."""
    n = len(y)
    r = acf_values(y, nlags)
    k = np.arange(1, nlags + 1)
    if se_method == "bartlett":
        se = np.sqrt((1 + 2 * np.r_[0, np.cumsum(r[:-1] ** 2)]) / n)
    else:
        se = np.sqrt((n - k) / (n * (n + 2)))
    q = n * (n + 2) * np.cumsum(r ** 2 / (n - k))
    p = stats.chi2.sf(q, k)
    return k, r, se, q, p


def pacf_values(y: np.ndarray, nlags: int) -> np.ndarray:
    """Durbin-Levinson partial autocorrelations from the sample ACF."""
    r = np.r_[1.0, acf_values(y, nlags)]
    phi = np.zeros((nlags + 1, nlags + 1))
    out = np.zeros(nlags)
    for k in range(1, nlags + 1):
        if k == 1:
            phi[1, 1] = r[1]
        else:
            num = r[k] - phi[k - 1, 1:k] @ r[1:k][::-1]
            den = 1 - phi[k - 1, 1:k] @ r[1:k]
            phi[k, k] = num / den
            phi[k, 1:k] = phi[k - 1, 1:k] - phi[k, k] * phi[k - 1, 1:k][::-1]
        out[k - 1] = phi[k, k]
    return out


def _corr_chart(title, lags, vals, conf, ylabel) -> bytes:
    fig = figure(6.4, 3.6)
    ax = fig.add_subplot(111)
    ax.bar(lags, vals, width=0.35, color="#1f4e9c")
    if conf is not None:
        ax.plot(lags, conf, color="#c8102e", lw=1, drawstyle="steps-mid")
        ax.plot(lags, -conf, color="#c8102e", lw=1, drawstyle="steps-mid")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("Lag Number")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.legend(["Coefficient", "Upper Confidence Limit", "Lower Confidence Limit"][:3 if conf
                                                                                is not None
                                                                                else 1],
              frameon=False, fontsize=7, loc="upper right")
    return png(fig)


def autocorrelations(ds: Dataset, names: list[str], max_lags: int = 16, acf: bool = True,
                     pacf: bool = True, se_method: str = "independence", log: bool = False,
                     diff: int = 0, sdiff: int = 0, periodic_only: bool = False,
                     level: float = 0.95) -> Section:
    period = periodicity(ds)
    sec = Section("Autocorrelations")
    sec.log(f"ACF VARIABLES={' '.join(names)}" + ("\n  /LN" if log else "") +
            (f"\n  /DIFF={diff}" if diff else "") + (f"\n  /SDIFF={sdiff}" if sdiff else "") +
            f"\n  /MXAUTO {max_lags}\n  /SERROR={'MA' if se_method == 'bartlett' else 'IND'}" +
            ("\n  /PACF" if pacf else "") + ".")
    z = stats.norm.ppf(0.5 + level / 2)
    for name in names:
        y = _trim_missing(transform_series(numeric(ds, name).to_numpy(float), log, diff, sdiff,
                                           period))
        if np.isnan(y).any():
            sec.note(name, "The series has embedded missing values; ACF cannot be computed.",
                     "warning")
            continue
        n = len(y)
        lags_n = min(max_lags, n - 2)
        if lags_n < 1:
            continue
        k, r, se, q, p = acf_table(y, lags_n, se_method)
        sel = (k % period == 0) if periodic_only and period > 1 else np.ones(len(k), bool)
        if acf:
            rows = [[fmt_int(kk), _f(rr), _f(ss), _f(qq), fmt_int(kk), fmt_p(pp)]
                    for kk, rr, ss, qq, pp in zip(k[sel], r[sel], se[sel], q[sel], p[sel])]
            sec.table(Table(f"Autocorrelations — Series: {name}",
                            ["Lag", "Autocorrelation", "Std. Errorᵃ",
                             ("Box-Ljung Statistic", "Value"), ("Box-Ljung Statistic", "df"),
                             ("Box-Ljung Statistic", "Sig.ᵇ")], rows,
                            footnotes=["a. The underlying process assumed is "
                                       + ("MA with the order equal to the lag number minus "
                                          "one. The Bartlett approximation is used."
                                          if se_method == "bartlett" else
                                          "independence (white noise)."),
                                       "b. Based on the asymptotic chi-square approximation.",
                                       "Transformation: " +
                                       _transform_text(log, diff, sdiff, period)]))
            sec.image(f"ACF: {name}", _corr_chart(name, k[sel], r[sel], z * se[sel], "ACF"))
        if pacf:
            pr = pacf_values(y, lags_n)
            pse = np.full(lags_n, 1 / math.sqrt(n))
            rows = [[fmt_int(kk), _f(v), _f(s)] for kk, v, s in zip(k[sel], pr[sel], pse[sel])]
            sec.table(Table(f"Partial Autocorrelations — Series: {name}",
                            ["Lag", "Partial Autocorrelation", "Std. Error"], rows))
            sec.image(f"PACF: {name}", _corr_chart(name, k[sel], pr[sel], z * pse[sel],
                                                  "Partial ACF"))
    return sec


def ccf_values(x: np.ndarray, y: np.ndarray, max_lag: int):
    """SPSS CCF: r_xy(k) = c_xy(k) / (s_x s_y), c_xy(k) = sum (x_t - xbar)(y_{t+k} - ybar)/n;
    SE = 1/sqrt(n - |k|)."""
    x = np.asarray(x, float) - np.mean(x)
    y = np.asarray(y, float) - np.mean(y)
    n = len(x)
    den = math.sqrt((x @ x) * (y @ y)) / n
    lags = np.arange(-max_lag, max_lag + 1)
    vals = []
    for k in lags:
        if k >= 0:
            c = x[:n - k] @ y[k:] / n
        else:
            c = x[-k:] @ y[:n + k] / n
        vals.append(c / den)
    return lags, np.array(vals), 1 / np.sqrt(n - np.abs(lags))


def cross_correlations(ds: Dataset, names: list[str], max_lag: int = 7, log: bool = False,
                       diff: int = 0, sdiff: int = 0, level: float = 0.95) -> Section:
    period = periodicity(ds)
    sec = Section("Cross-Correlations")
    sec.log(f"CCF VARIABLES={' '.join(names)}" + ("\n  /LN" if log else "") +
            (f"\n  /DIFF={diff}" if diff else "") + f"\n  /MXCROSS {max_lag}.")
    z = stats.norm.ppf(0.5 + level / 2)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            xa = transform_series(numeric(ds, a).to_numpy(float), log, diff, sdiff, period)
            xb = transform_series(numeric(ds, b).to_numpy(float), log, diff, sdiff, period)
            ok = ~np.isnan(xa) & ~np.isnan(xb)
            lags, r, se = ccf_values(xa[ok], xb[ok], min(max_lag, ok.sum() - 2))
            rows = [[fmt_int(k), _f(v), _f(s)] for k, v, s in zip(lags, r, se)]
            sec.table(Table(f"Cross Correlations — Series Pair: {a} with {b}",
                            ["Lag", "Cross Correlation", "Std. Errorᵃ"], rows,
                            footnotes=["a. Based on the assumption that the series are not "
                                       "cross correlated and that one of the series is white "
                                       "noise."]))
            sec.image(f"CCF: {a} with {b}", _corr_chart(f"{a} with {b}", lags, r, z * se, "CCF"))
    return sec


# =========================================================================== seasonal decomposition
def seasonal_decompose(x: np.ndarray, period: int, model: str = "multiplicative",
                       endpoints: str = "half") -> dict:
    """SPSS SEASON. Moving average of span = period (2 x period centered with endpoints
    weighted by .5 when ``endpoints == "half"`` or the period is even; equal weights
    otherwise), seasonal factors = normalized means of the ratios/differences per season,
    SAS = seasonally adjusted series, STC = smoothed trend-cycle (3x3 moving average of SAS),
    ERR = SAS / STC (or SAS - STC)."""
    x = np.asarray(x, float)
    n = len(x)
    if period % 2 == 0 or endpoints == "half":
        if period % 2 == 0:
            w = np.r_[0.5, np.ones(period - 1), 0.5] / period
        else:
            w = np.ones(period) / period
    else:
        w = np.ones(period) / period
    h = len(w) // 2
    ma = np.full(n, np.nan)
    conv = np.convolve(x, w, mode="valid")
    ma[h:h + len(conv)] = conv
    si = x / ma if model == "multiplicative" else x - ma
    season = np.arange(n) % period
    means = np.array([np.nanmean(si[season == s]) for s in range(period)])
    if model == "multiplicative":
        means = means / means.mean()
    else:
        means = means - means.mean()
    saf = means[season]
    sas = x / saf if model == "multiplicative" else x - saf
    stc = _smooth_3x3(sas)
    err = sas / stc if model == "multiplicative" else sas - stc
    return {"factors": means, "SAF": saf, "SAS": sas, "STC": stc, "ERR": err, "MA": ma}


def _smooth_3x3(y: np.ndarray) -> np.ndarray:
    w = np.array([1, 2, 3, 2, 1]) / 9.0
    n = len(y)
    out = np.convolve(y, w, mode="same")
    if n >= 5:
        out[0] = (2 * y[0] + y[1]) / 3 if n > 1 else y[0]  # shorter averages at the ends
        out[1] = (y[0] + y[1] + y[2]) / 3
        out[-2] = (y[-3] + y[-2] + y[-1]) / 3
        out[-1] = (2 * y[-1] + y[-2]) / 3
    return out


def seasonal_decomposition(ds: Dataset, names: list[str], model: str = "multiplicative",
                           endpoints: str = "half", period: int | None = None,
                           save: bool = True) -> Result:
    period = period or periodicity(ds)
    sec = Section("Seasonal Decomposition")
    sec.log(f"SEASON VARIABLES={' '.join(names)}\n  /MODEL="
            f"{'MULTIPLICATIVE' if model == 'multiplicative' else 'ADDITIVE'}\n  /MA="
            f"{'CENTERED' if endpoints == 'half' else 'EQUAL'}\n  /PERIOD={period}.")
    if period < 2:
        raise ValueError("Seasonal decomposition needs a periodicity of at least 2. Define "
                         "dates (Data > Define Date and Time) or enter the period.")
    sec.table(Table("Model Description", ["", ""],
                    [["Model Name", "MOD_1"],
                     ["Model Type", model.capitalize()],
                     ["Series Name", ", ".join(names)],
                     ["Length of Seasonal Period", str(period)],
                     ["Computing Method of Moving Averages",
                      "Span equal to the periodicity plus 1 and endpoints weighted by .5"
                      if endpoints == "half" or period % 2 == 0 else
                      "Span equal to the periodicity and all points weighted equally"]]))
    saved = {}
    for name in names:
        x = numeric(ds, name).to_numpy(float)
        if np.isnan(x).any():
            raise ValueError(f"{name} has missing values; seasonal decomposition needs a "
                             "complete series.")
        if model == "multiplicative" and (x <= 0).any():
            raise ValueError("The multiplicative model needs positive values.")
        res = seasonal_decompose(x, period, model, endpoints)
        fac = res["factors"]
        rows = [[fmt_int(s + 1), _f(100 * fac[s] if model == "multiplicative" else fac[s], 1
                                     if model == "multiplicative" else 3)]
                for s in range(period)]
        sec.table(Table(f"Seasonal Factors — Series Name: {name}",
                        ["Period", "Seasonal Factor (%)" if model == "multiplicative"
                         else "Seasonal Factor"], rows))
        if save:
            labels = {"ERR": "Error for", "SAS": "Seas adj ser for", "SAF": "Seas factors for",
                      "STC": "Trend-cycle for"}
            for key, lab in labels.items():
                base = key if len(names) == 1 else f"{key}_{name}"[:60]
                saved[base] = (f"{lab} {name} from SEASON, MOD_1 {model[:3].upper()} "
                               f"{'CEN' if endpoints == 'half' else 'EQU'} {period}",
                               pd.Series(res[key], index=ds.df.index), 4)
        fig = figure(7.0, 4.6)
        ax = fig.add_subplot(211)
        ax.plot(x, color="#7f7f7f", lw=1, label=name)
        ax.plot(res["SAS"], color="#1f4e9c", lw=1.2, label="Seasonally adjusted (SAS)")
        ax.plot(res["STC"], color="#c8102e", lw=1.2, label="Trend-cycle (STC)")
        ax.legend(frameon=False, fontsize=7)
        ax.grid(alpha=0.25)
        ax2 = fig.add_subplot(212)
        ax2.plot(res["SAF"], color="#2e8b57", lw=1, label="Seasonal factors (SAF)")
        ax2.plot(res["ERR"], color="#d48a00", lw=1, label="Irregular (ERR)")
        ax2.legend(frameon=False, fontsize=7)
        ax2.grid(alpha=0.25)
        ax2.set_xlabel("Sequence number")
        sec.image(f"Seasonal Decomposition: {name}", png(fig), width=640)
    return Result(sec, saved)


# =========================================================================== spectral
WINDOWS = {"tukey_hamming": "Tukey-Hamming", "tukey": "Tukey", "parzen": "Parzen",
           "bartlett": "Bartlett", "daniell": "Daniell (Unit)", "none": "None"}


def _window_weights(window: str, span: int) -> np.ndarray:
    span = max(int(span), 1)
    if window == "none" or span == 1:
        return np.array([1.0])
    m = np.arange(span) - (span - 1) / 2
    if window == "daniell":
        w = np.ones(span)
    elif window == "bartlett":
        w = 1 - np.abs(m) / ((span + 1) / 2)
    elif window == "tukey":
        w = 0.5 * (1 + np.cos(np.pi * m / ((span + 1) / 2)))
    elif window == "parzen":
        w = signal.windows.parzen(span + 2)[1:-1]
    else:  # Tukey-Hamming
        w = 0.54 + 0.46 * np.cos(np.pi * m / ((span + 1) / 2))
    return w / w.sum()


def spectral(ds: Dataset, names: list[str], window: str = "tukey_hamming", span: int = 5,
             center: bool = True, plot_periodogram: bool = True, plot_density: bool = True,
             by_period: bool = False) -> Section:
    sec = Section("Spectral Analysis")
    sec.log(f"SPECTRA VARIABLES={' '.join(names)}\n  /WINDOW={WINDOWS[window].upper()}({span})"
            f"\n  /{'CENTER' if center else 'NOCENTER'}\n  /PLOT=P S BY "
            f"{'PERIOD' if by_period else 'FREQUENCY'}.")
    sec.table(Table("Model Description", ["", ""],
                    [["Series Name", ", ".join(names)], ["Spectral Window", WINDOWS[window]],
                     ["Span", str(span) if window != "none" else ""],
                     ["Center variables", "Yes" if center else "No"]]))
    for name in names:
        x = numeric(ds, name).to_numpy(float)
        x = x[~np.isnan(x)]
        n = len(x)
        if center:
            x = x - x.mean()
        k = np.arange(0, n // 2 + 1)
        freq = k / n
        t = np.arange(n)
        a = np.array([2 / n * np.sum(x * np.cos(2 * np.pi * f * t)) for f in freq])
        b = np.array([2 / n * np.sum(x * np.sin(2 * np.pi * f * t)) for f in freq])
        a[0] /= 2
        if n % 2 == 0:
            a[-1] /= 2
        per = n / 2 * (a ** 2 + b ** 2)
        w = _window_weights(window, span)
        h = len(w) // 2
        padded = np.r_[per[1:h + 1][::-1], per, per[-h - 1:-1][::-1]] if h else per
        dens = np.convolve(padded, w, mode="valid")[:len(per)] / (4 * np.pi)
        rows = [[fmt_int(i + 1), _f(f, 5), _f(1 / f, 3) if f > 0 else "", _f(bb, 3),
                 _f(aa, 3), _f(pp, 3), _f(dd, 3)]
                for i, (f, aa, bb, pp, dd) in enumerate(zip(freq, a, b, per, dens))]
        sec.table(Table(f"Univariate Statistics — Series: {name}",
                        ["", "Frequency", "Period", "Sine Transform", "Cosine Transform",
                         "Periodogram", "Spectral Density Estimate"], rows))
        for flag, values, title in ((plot_periodogram, per, "Periodogram"),
                                    (plot_density, dens, "Spectral Density")):
            if not flag:
                continue
            fig = figure(6.4, 3.6)
            ax = fig.add_subplot(111)
            xv = freq[1:] if not by_period else 1 / freq[1:]
            ax.plot(xv, values[1:], color="#1f4e9c", lw=1.2)
            ax.set_xlabel("Period" if by_period else "Frequency")
            ax.set_ylabel(title)
            ax.set_title(f"{title} of {name}", fontsize=10)
            ax.grid(alpha=0.25)
            sec.image(f"{title}: {name}", png(fig))
    return sec


# =========================================================================== models
ETS_TYPES = {"simple": "Simple", "holt": "Holt's linear trend", "brown": "Brown's linear trend",
             "damped": "Damped trend", "simple_seasonal": "Simple seasonal",
             "winters_add": "Winters' additive", "winters_mult": "Winters' multiplicative"}


@dataclass
class ModelSpec:
    kind: str                     # ets | arima
    ets: str = "simple"
    order: tuple = (0, 0, 0)
    seasonal: tuple = (0, 0, 0)
    transform: str = "none"       # none | sqrt | log
    constant: bool = True
    predictors: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "ets": self.ets, "order": list(self.order),
                "seasonal": list(self.seasonal), "transform": self.transform,
                "constant": self.constant, "predictors": list(self.predictors)}

    @classmethod
    def from_dict(cls, d: dict) -> "ModelSpec":
        return cls(d["kind"], d.get("ets", "simple"), tuple(d.get("order", (0, 0, 0))),
                   tuple(d.get("seasonal", (0, 0, 0))), d.get("transform", "none"),
                   d.get("constant", True), list(d.get("predictors", [])))

    def label(self, period: int) -> str:
        if self.kind == "ets":
            return ETS_TYPES[self.ets]
        p, d, q = self.order
        text = f"ARIMA({p},{d},{q})"
        if period > 1 and any(self.seasonal):
            text += "({},{},{})".format(*self.seasonal)
        return text


@dataclass
class FittedModel:
    spec: ModelSpec
    series: str
    period: int
    fitted: np.ndarray              # one-step predictions on the original scale
    residuals: np.ndarray           # noise residuals on the (transformed) model scale
    forecast: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    params: list                    # (name, estimate, se, t, p)
    n_params: int
    stats: dict
    raw_params: list
    in_sample_lower: np.ndarray | None = None
    in_sample_upper: np.ndarray | None = None


def _tf(y, transform):
    if transform == "log":
        return np.log(y)
    if transform == "sqrt":
        return np.sqrt(y)
    return y


def _itf(y, transform):
    if transform == "log":
        return np.exp(y)
    if transform == "sqrt":
        return np.square(np.clip(y, 0, None))
    return y


def _fit_ets(y, spec: ModelSpec, period: int, horizon: int, level: float, fixed=None):
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel

    trend = None if spec.ets in ("simple", "simple_seasonal") else "add"
    damped = spec.ets == "damped"
    seasonal = {"simple_seasonal": "add", "winters_add": "add", "winters_mult": "mul"}.get(
        spec.ets)
    error = "mul" if spec.ets == "winters_mult" else "add"
    kw = dict(error=error, trend=trend, damped_trend=damped, seasonal=seasonal,
              seasonal_periods=period if seasonal else None, initialization_method="estimated")
    model = ETSModel(pd.Series(np.asarray(y, float)), **kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if spec.ets == "brown" and fixed is None:
            def sse(lam):
                lam = float(np.clip(lam, 1e-4, 0.9999))
                with model.fix_params({"smoothing_level": lam * (2 - lam),
                                       "smoothing_trend": lam ** 2}):
                    r = model.fit(disp=False)
                return float(np.sum(r.resid ** 2)), r
            from scipy.optimize import minimize_scalar

            best = minimize_scalar(lambda v: sse(v)[0], bounds=(1e-3, 0.999), method="bounded")
            res = sse(best.x)[1]
            res._brown_alpha = float(best.x)
        elif fixed:
            with model.fix_params(fixed):
                res = model.fit(disp=False)
        else:
            res = _multistart(model)
        pred = res.get_prediction(start=len(y), end=len(y) + horizon - 1) if horizon else None
    names = {"smoothing_level": "Alpha (Level)", "smoothing_trend": "Gamma (Trend)",
             "smoothing_seasonal": "Delta (Season)", "damping_trend": "Phi (Trend damping)"}
    params = []
    raw = {}
    pnames = list(res.model.param_names)
    bse = np.asarray(getattr(res, "bse", np.full(len(pnames), np.nan)), float)
    for i, nm in enumerate(pnames):
        raw[nm] = float(res.params[i])
        if nm in names:
            est = float(res.params[i])
            se = float(bse[i]) if i < len(bse) else float("nan")
            if spec.ets == "brown" and nm in ("smoothing_level", "smoothing_trend"):
                if nm == "smoothing_trend":
                    continue
                est = getattr(res, "_brown_alpha", est)
                se = float("nan")
                params.append(("Alpha (Level and Trend)", est, se, float("nan"), float("nan")))
                continue
            t = est / se if se and se == se and se > 0 else float("nan")
            p = 2 * stats.t.sf(abs(t), len(y) - 1) if t == t else float("nan")
            params.append((names[nm], est, se, t, p))
    if horizon:
        sf = pred.summary_frame(alpha=1 - level)
        fc, lo, hi = (sf["mean"].to_numpy(), sf["pi_lower"].to_numpy(),
                      sf["pi_upper"].to_numpy())
    else:
        fc = lo = hi = np.zeros(0)
    k = len([p for p in pnames if p in names])
    fitted = np.asarray(res.fittedvalues, float)
    return fitted, np.asarray(y, float) - fitted, fc, lo, hi, params, k, raw


def _multistart(model):
    """ETS likelihoods are multimodal (e.g. Holt on the airline series): fit from the default
    and from several smoothing-parameter starts and keep the best log likelihood."""
    best = model.fit(disp=False)
    names = list(model.param_names)
    base = np.asarray(model.start_params, float)
    for alpha in (0.2, 0.6, 0.95):
        sp = base.copy()
        for i, nm in enumerate(names):
            if nm == "smoothing_level":
                sp[i] = alpha
            elif nm == "smoothing_trend":
                sp[i] = 0.05 * alpha
            elif nm == "smoothing_seasonal":
                sp[i] = 0.05 * (1 - alpha)
            elif nm == "damping_trend":
                sp[i] = 0.95
        try:
            res = model.fit(start_params=sp, disp=False)
        except Exception:  # noqa: BLE001 - a failed start is skipped
            continue
        if np.isfinite(res.llf) and res.llf > best.llf + 1e-8:
            best = res
    return best


def _arima_trend(spec: ModelSpec):
    d, sd = spec.order[1], spec.seasonal[1]
    if not spec.constant:
        return None
    total = d + sd
    if total == 0:
        return "c"
    return [0] * total + [1]


def _fit_arima(y, spec: ModelSpec, period: int, exog, exog_future, horizon: int,
               level: float, fixed=None):
    from statsmodels.tsa.arima.model import ARIMA

    seasonal = tuple(spec.seasonal) + (period,) if period > 1 and any(spec.seasonal) \
        else (0, 0, 0, 0)
    if exog is not None and spec.predictors:
        y = pd.Series(np.asarray(y, float))
        exog = pd.DataFrame(np.asarray(exog, float), columns=spec.predictors)
        if exog_future is not None:
            exog_future = pd.DataFrame(np.asarray(exog_future, float), columns=spec.predictors,
                                       index=pd.RangeIndex(len(y), len(y) + len(exog_future)))
    model = ARIMA(y, exog=exog, order=tuple(spec.order), seasonal_order=seasonal,
                  trend=_arima_trend(spec))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if fixed:
            res = model.filter(np.asarray(fixed, float))
        else:
            res = model.fit(cov_type="approx")
        fc_obj = res.get_forecast(horizon, exog=exog_future) if horizon else None
    names = res.model.param_names
    est = np.asarray(res.params, float)
    bse = np.asarray(res.bse, float) if not fixed else np.full(len(est), np.nan)
    params = []
    n_eff = len(y)
    for i, nm in enumerate(names):
        if nm == "sigma2":
            continue
        e, s = est[i], bse[i]
        label = nm
        sign = 1.0
        if nm == "const":
            label = "Constant"
        elif nm.startswith("x") and nm[1:].isdigit() and exog is None:
            label = "Constant (drift)" if nm == "x1" else nm
        elif nm.startswith("ar.S.L"):
            label = f"AR, Seasonal Lag {int(nm[6:]) // max(period, 1)}"
        elif nm.startswith("ma.S.L"):
            label = f"MA, Seasonal Lag {int(nm[6:]) // max(period, 1)}"
            sign = -1.0
        elif nm.startswith("ar.L"):
            label = f"AR Lag {nm[4:]}"
        elif nm.startswith("ma.L"):
            label = f"MA Lag {nm[4:]}"
            sign = -1.0
        elif spec.predictors and nm in spec.predictors:
            label = f"{nm} Numerator Lag 0"
        elif nm.startswith("drift") or nm.startswith("trend"):
            label = "Constant (drift)"
        e *= sign
        t = e / s if s == s and s > 0 else float("nan")
        p = 2 * stats.t.sf(abs(t), max(n_eff - len(est), 1)) if t == t else float("nan")
        params.append((label, e, s, t, p))
    k = len(names) - 1
    if horizon:
        sf = fc_obj.summary_frame(alpha=1 - level)
        fc, lo, hi = (sf["mean"].to_numpy(), sf["mean_ci_lower"].to_numpy(),
                      sf["mean_ci_upper"].to_numpy())
    else:
        fc = lo = hi = np.zeros(0)
    fitted = np.array(res.fittedvalues, dtype=float)
    resid = np.array(res.resid, dtype=float)
    burn = spec.order[1] + spec.seasonal[1] * (period if period > 1 else 0)
    fitted[:burn] = np.nan
    resid[:burn] = np.nan
    return fitted, resid, fc, lo, hi, params, k, [float(v) for v in est]


def _fit_stats(y, fitted, resid, k, spec: ModelSpec, period: int, y_model) -> dict:
    ok = ~np.isnan(fitted) & ~np.isnan(y)
    e = y[ok] - fitted[ok]
    n = ok.sum()
    sse = float(e @ e)
    sst = float(np.sum((y[ok] - y[ok].mean()) ** 2))
    if spec.kind == "arima":
        d, sd = spec.order[1], spec.seasonal[1]
    else:
        d = {"simple": 1, "holt": 2, "brown": 2, "damped": 1, "simple_seasonal": 0,
             "winters_add": 1, "winters_mult": 1}[spec.ets]
        sd = 1 if spec.ets in ("simple_seasonal", "winters_add", "winters_mult") else 0
    ys = transform_series(y_model, False, d, sd, period)
    ys_ok = ~np.isnan(ys)
    r_ok = ~np.isnan(resid)
    both = ys_ok & r_ok
    sse_m = float(np.sum(resid[both] ** 2))
    sst_s = float(np.sum((ys[both] - ys[both].mean()) ** 2)) if both.any() else float("nan")
    with np.errstate(divide="ignore", invalid="ignore"):
        ape = np.abs(e / y[ok]) * 100
    mse = sse / n if n else float("nan")
    lb_lags = 18
    r = resid[~np.isnan(resid)]
    q = df = pq = float("nan")
    if len(r) > lb_lags + 1:
        kk = np.arange(1, lb_lags + 1)
        rr = acf_values(r, lb_lags)
        q = float(len(r) * (len(r) + 2) * np.sum(rr ** 2 / (len(r) - kk)))
        n_arma = (sum(spec.order[::2]) + sum(spec.seasonal[::2])) if spec.kind == "arima" \
            else k
        df = max(lb_lags - n_arma, 1)
        pq = float(stats.chi2.sf(q, df))
    return {"Stationary R-squared": 1 - sse_m / sst_s if sst_s and sst_s > 0 else float("nan"),
            "R-squared": 1 - sse / sst if sst > 0 else float("nan"),
            "RMSE": math.sqrt(mse), "MAPE": float(np.nanmean(ape)),
            "MaxAPE": float(np.nanmax(ape)) if len(ape) else float("nan"),
            "MAE": float(np.mean(np.abs(e))), "MaxAE": float(np.max(np.abs(e))),
            "Normalized BIC": math.log(mse) + (k + 1) * math.log(n) / n if mse > 0 else float(
                "nan"), "Q": q, "Q_df": df, "Q_sig": pq, "N": int(n)}


def fit_model(y: np.ndarray, spec: ModelSpec, period: int, horizon: int = 0,
              level: float = 0.95, exog=None, exog_future=None, series: str = "",
              fixed=None) -> FittedModel:
    y = np.asarray(y, float)
    if spec.transform != "none" and (y <= 0).any():
        raise ValueError(f"The {spec.transform} transformation needs positive values.")
    ym = _tf(y, spec.transform)
    if spec.kind == "ets":
        fitted_m, resid, fc, lo, hi, params, k, raw = _fit_ets(ym, spec, period, horizon, level,
                                                               fixed)
    else:
        fitted_m, resid, fc, lo, hi, params, k, raw = _fit_arima(ym, spec, period, exog,
                                                                 exog_future, horizon, level,
                                                                 fixed)
    fitted = _itf(np.asarray(fitted_m, float), spec.transform)
    st = _fit_stats(y, fitted, np.asarray(resid, float), k, spec, period, ym)
    return FittedModel(spec, series, period, fitted, np.asarray(resid, float),
                       _itf(fc, spec.transform), _itf(lo, spec.transform),
                       _itf(hi, spec.transform), params, k, st,
                       raw if isinstance(raw, list) else raw)


def expert_modeler(y: np.ndarray, period: int, exog=None, ets_only: bool = False,
                   arima_only: bool = False) -> ModelSpec:
    """A simple Expert Modeler: the best ETS model (AICc) and the best ARIMA model (AICc over
    p, q <= 2, P, Q <= 1 with d from KPSS tests and D = 1 for strong seasonality, log
    transform when the series is positive and its spread grows with its level); the final
    choice between the two is the smaller normalized BIC (both on the original scale)."""
    from statsmodels.tsa.stattools import kpss

    y = np.asarray(y, float)
    seasonal = period > 1 and len(y) >= 2 * period + 4
    positive = (y > 0).all()
    candidates = []
    if not arima_only and exog is None:
        types = ["simple", "holt", "damped"]
        if seasonal:
            types += ["simple_seasonal", "winters_add"] + (["winters_mult"] if positive else [])
        best = None
        for t in types:
            spec = ModelSpec("ets", t)
            try:
                fm = fit_model(y, spec, period)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    from statsmodels.tsa.exponential_smoothing.ets import ETSModel  # noqa
                score = fm.stats["Normalized BIC"]
            except Exception:  # noqa: BLE001 - a candidate that fails is skipped
                continue
            if best is None or score < best[0]:
                best = (score, spec, fm)
        if best:
            candidates.append(best)
    if not ets_only:
        transform = "none"
        if positive and len(y) > 2 * max(period, 4):
            seg = np.array_split(y, max(4, len(y) // max(period, 4)))
            lv = np.array([s.mean() for s in seg])
            sp = np.array([s.std() for s in seg])
            if np.corrcoef(lv, sp)[0, 1] > 0.7:
                transform = "log"
        ym = _tf(y, transform)
        D = 0
        if seasonal:
            dec = seasonal_decompose(ym, period, "additive")
            rem = ym - dec["SAF"]
            strength = 1 - np.nanvar(dec["ERR"]) / np.nanvar(rem - dec["STC"] + dec["ERR"]) \
                if np.nanvar(rem) > 0 else 0
            season_strength = 1 - np.nanvar(dec["ERR"]) / np.nanvar(ym - dec["STC"])
            D = 1 if season_strength > 0.64 else 0
            _ = strength
        z = transform_series(ym, False, 0, D, period)
        z = z[~np.isnan(z)]
        d = 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            while d < 2 and len(z) > 10 and kpss(z, regression="c", nlags="auto")[1] < 0.05:
                d += 1
                z = np.diff(z)
        best = None
        from statsmodels.tsa.arima.model import ARIMA

        for p in range(3):
            for q in range(3 - p):
                for P in range(2 if seasonal else 1):
                    for Q in range(2 if seasonal else 1):
                        spec = ModelSpec("arima", order=(p, d, q), seasonal=(P, D, Q),
                                         transform=transform, constant=(d + D) <= 1)
                        try:
                            so = (P, D, Q, period) if seasonal else (0, 0, 0, 0)
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                r = ARIMA(ym, exog=exog, order=(p, d, q), seasonal_order=so,
                                          trend=_arima_trend(spec)).fit()
                            score = r.aicc
                        except Exception:  # noqa: BLE001
                            continue
                        if np.isfinite(score) and (best is None or score < best[0]):
                            best = (score, spec)
        if best:
            fm = fit_model(y, best[1], period, exog=exog)
            candidates.append((fm.stats["Normalized BIC"], best[1], fm))
    if not candidates:
        return ModelSpec("ets", "simple")
    return min(candidates, key=lambda c: c[0])[1]


STAT_KEYS = ["Stationary R-squared", "R-squared", "RMSE", "MAPE", "MAE", "MaxAPE", "MaxAE",
             "Normalized BIC"]


def _forecast_labels(ds: Dataset, start: int, horizon: int) -> list[str]:
    info = ds.meta.get("dates")
    if info and info.get("kind") in ("years_months", "years_quarters", "years"):
        period = info["period"]
        y0 = info["start"][0]
        m0 = info["start"][1] if len(info["start"]) > 1 else 1
        out = []
        for i in range(start, start + horizon):
            base = (m0 - 1) + i
            year = y0 + base // period
            minor = base % period + 1
            if info["kind"] == "years_months":
                out.append(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP",
                            "OCT", "NOV", "DEC"][minor - 1] + f" {year}")
            elif info["kind"] == "years_quarters":
                out.append(f"Q{minor} {year}")
            else:
                out.append(str(y0 + i))
        return out
    return [str(i + 1) for i in range(start, start + horizon)]


def forecast_models(ds: Dataset, dependents: list[str], method: str = "expert",
                    spec: ModelSpec | None = None, predictors: list[str] | None = None,
                    horizon: int = 12, level: float = 0.95, statistics: bool = True,
                    param_estimates: bool = True, forecasts: bool = True,
                    fit_plot: bool = True, residual_acf: bool = False,
                    save: tuple = (), store: bool = False, period: int | None = None,
                    title: str = "Time Series Modeler", fixed: dict | None = None
                    ) -> Section:
    """SPSS TSMODEL. ``method``: expert | ets | arima. ``save`` from predicted, lcl, ucl,
    nresidual; saving appends forecast cases to the dataset (modifies ``ds``). With ``store``
    the model specifications/parameters go to ``ds.meta["forecast_models"]``."""
    predictors = list(predictors or [])
    period = period or periodicity(ds)
    sec = Section(title)
    lines = ["TSMODEL", f"  /MODELSUMMARY PRINT=[{'MODELFIT' if statistics else ''}]"]
    models = []
    for dep in dependents:
        y_full = numeric(ds, dep).to_numpy(float)
        valid = np.nonzero(~np.isnan(y_full))[0]
        if not len(valid):
            raise ValueError(f"{dep} has no valid values.")
        first, last = valid[0], valid[-1]
        y = y_full[first:last + 1]
        if np.isnan(y).any():
            y = pd.Series(y).interpolate().to_numpy()
            sec.note(dep, "Embedded missing values were replaced by linear interpolation.",
                     "warning")
        exog = exog_future = None
        if predictors:
            ex = pd.concat({p: numeric(ds, p) for p in predictors}, axis=1).to_numpy(float)
            exog = ex[first:last + 1]
            fut = ex[last + 1:last + 1 + horizon]
            if np.isnan(exog).any():
                raise ValueError("Predictors may not have missing values in the estimation "
                                 "period.")
            if len(fut) < horizon or np.isnan(fut).any():
                horizon_eff = len(fut) if not np.isnan(fut).any() else \
                    int(np.argmax(np.isnan(fut).any(axis=1)))
                sec.note(dep, "Forecasts with predictors need predictor values for the "
                              f"forecast period; {horizon_eff} period(s) available.", "warning")
                fut = fut[:horizon_eff]
            exog_future = fut
        h = horizon if exog_future is None else len(exog_future)
        if method == "expert":
            sp = expert_modeler(y, period, exog)
        elif fixed and dep in fixed:
            sp = ModelSpec.from_dict(fixed[dep]["spec"])
        else:
            sp = spec or ModelSpec("ets", "simple")
        if predictors and sp.kind == "arima":
            sp.predictors = predictors
        fixed_params = fixed[dep]["params"] if fixed and dep in fixed and \
            fixed[dep].get("params") is not None else None
        fm = fit_model(y, sp, period, h, level, exog if sp.kind == "arima" else None,
                       exog_future if sp.kind == "arima" else None, dep, fixed_params)
        models.append((dep, first, last, fm))
        lines.append(f"  /MODEL DEPENDENT={dep}" +
                     (f" INDEPENDENT={' '.join(predictors)}" if predictors else "") +
                     f"\n  /{'EXPERTMODELER' if method == 'expert' else sp.kind.upper()} "
                     f"{sp.label(period)}")
    sec.log("\n".join(lines) + f"\n  /AUXILIARY CILEVEL={level * 100:g} MAXACFLAGS=24.")
    sec.table(Table("Model Description", ["", "", "Model Type"],
                    [[dep, f"Model_{i + 1}", fm.spec.label(period) +
                      (f", {fm.spec.transform} transformation" if fm.spec.transform != "none"
                       else "")] for i, (dep, _, _, fm) in enumerate(models)],
                    row_header_cols=2))
    if statistics:
        rows = []
        for i, (dep, _, _, fm) in enumerate(models):
            st = fm.stats
            rows.append([f"{dep}-Model_{i + 1}", fmt_int(len(fm.spec.predictors))] +
                        [_f(st[k], 3) for k in STAT_KEYS] +
                        [_f(st["Q"], 3), fmt_int(st["Q_df"]) if st["Q_df"] == st["Q_df"]
                         else "", fmt_p(st["Q_sig"]) if st["Q_sig"] == st["Q_sig"] else "", "0"])
        sec.table(Table("Model Statistics", ["Model", "Number of Predictors"] +
                        [("Model Fit statistics", k) for k in STAT_KEYS] +
                        [("Ljung-Box Q(18)", "Statistics"), ("Ljung-Box Q(18)", "DF"),
                         ("Ljung-Box Q(18)", "Sig."), "Number of Outliers"], rows))
    if param_estimates:
        for i, (dep, _, _, fm) in enumerate(models):
            rows = [[f"{dep}-Model_{i + 1}" if j == 0 else "", nm, _f(e, 3), _f(s, 3), _f(t, 3),
                     fmt_p(p) if p == p else ""] for j, (nm, e, s, t, p) in enumerate(fm.params)]
            title_t = "Exponential Smoothing Model Parameters" if fm.spec.kind == "ets" else \
                "ARIMA Model Parameters"
            tname = {"none": "No Transformation", "log": "Natural Logarithm",
                     "sqrt": "Square Root"}[fm.spec.transform]
            foot = [f"Transformation: {tname}"]
            if fm.spec.kind == "arima":
                foot.append(f"Difference: {fm.spec.order[1]}; Seasonal Difference: "
                            f"{fm.spec.seasonal[1]}. MA parameters use the SPSS sign "
                            f"convention (1 - theta B).")
            sec.table(Table(title_t, ["", "", "Estimate", "SE", "t", "Sig."], rows,
                            row_header_cols=2, footnotes=foot))
    if forecasts and horizon:
        for i, (dep, first, last, fm) in enumerate(models):
            if not len(fm.forecast):
                continue
            labels = _forecast_labels(ds, last + 1, len(fm.forecast))
            rows = [["Forecast"] + [_f(v, 2) for v in fm.forecast],
                    ["UCL"] + [_f(v, 2) for v in fm.upper],
                    ["LCL"] + [_f(v, 2) for v in fm.lower]]
            sec.table(Table(f"Forecast — {dep}-Model_{i + 1}", ["Model"] + labels, rows,
                            footnotes=["For each model, forecasts start after the last non-"
                                       "missing in the range of the requested estimation "
                                       "period, and end at the last period for which "
                                       "non-missing values of all the predictors are available "
                                       "or at the end of the requested forecast period, "
                                       "whichever is earlier."]))
    if fit_plot:
        for i, (dep, first, last, fm) in enumerate(models):
            fig = figure(7.0, 3.8)
            ax = fig.add_subplot(111)
            x = np.arange(first + 1, last + 2)
            y = numeric(ds, dep).to_numpy(float)[first:last + 1]
            ax.plot(x, y, color="#1f4e9c", lw=1.2, label="Observed")
            ax.plot(x, fm.fitted, color="#c8102e", lw=1, label="Fit")
            if len(fm.forecast):
                xf = np.arange(last + 2, last + 2 + len(fm.forecast))
                ax.plot(xf, fm.forecast, color="#c8102e", lw=1.2, ls="--", label="Forecast")
                ax.fill_between(xf, fm.lower, fm.upper, color="#c8102e", alpha=0.15,
                                label=f"{level * 100:g}% limits")
            labels = date_labels(ds)
            if labels:
                full = labels + _forecast_labels(ds, len(labels), len(fm.forecast))
                ticks = np.arange(first + 1, last + 2 + len(fm.forecast))
                step = max(1, len(ticks) // 10)
                ax.set_xticks(ticks[::step])
                ax.set_xticklabels([full[t - 1] for t in ticks[::step]], rotation=45,
                                   ha="right", fontsize=7)
            ax.set_title(f"{dep}-Model_{i + 1}", fontsize=10)
            ax.legend(frameon=False, fontsize=7)
            ax.grid(alpha=0.25)
            sec.image(f"{dep}-Model_{i + 1}", png(fig), width=640)
    if residual_acf:
        for i, (dep, _, _, fm) in enumerate(models):
            r = fm.residuals[~np.isnan(fm.residuals)]
            nl = min(24, len(r) - 2)
            if nl < 1:
                continue
            k, rr, se, _, _ = acf_table(r, nl)
            pr = pacf_values(r, nl)
            z = stats.norm.ppf(0.5 + level / 2)
            sec.image(f"Residual ACF: {dep}-Model_{i + 1}",
                      _corr_chart(f"Residual ACF {dep}-Model_{i + 1}", k, rr, z * se, "ACF"))
            sec.image(f"Residual PACF: {dep}-Model_{i + 1}",
                      _corr_chart(f"Residual PACF {dep}-Model_{i + 1}", k, pr,
                                  np.full(nl, z / math.sqrt(len(r))), "PACF"))
    if save:
        extra = max(0, max((last + 1 + len(fm.forecast)) - ds.n_cases
                           for _, _, last, fm in models))
        if extra:
            ds.insert_cases(ds.n_cases, extra)
            _extend_dates(ds, extra)
        created = []
        for i, (dep, first, last, fm) in enumerate(models):
            cols = {}
            n = ds.n_cases
            pred = np.full(n, np.nan)
            pred[first:last + 1] = fm.fitted
            pred[last + 1:last + 1 + len(fm.forecast)] = fm.forecast
            cols["predicted"] = ("Predicted", pred)
            lcl = np.full(n, np.nan)
            ucl = np.full(n, np.nan)
            lcl[last + 1:last + 1 + len(fm.lower)] = fm.lower
            ucl[last + 1:last + 1 + len(fm.upper)] = fm.upper
            cols["lcl"] = ("LCL", lcl)
            cols["ucl"] = ("UCL", ucl)
            nres = np.full(n, np.nan)
            nres[first:last + 1] = fm.residuals
            cols["nresidual"] = ("NResidual", nres)
            for key in save:
                prefix, values = cols[key]
                name = f"{prefix}_{dep}_Model_{i + 1}"[:64]
                k = 1
                base = name
                while ds.has_name(name):
                    k += 1
                    name = f"{base}_{k}"
                var = ds.add_variable(name, values=values)
                var.label = f"{prefix} value from {dep}-Model_{i + 1}"
                var.decimals = max(ds.var(dep).decimals, 2)
                created.append(name)
        sec.note("", "Variables created: " + ", ".join(created) +
                 (f". {extra} forecast case(s) were added." if extra else "."))
    if store:
        stored = ds.meta.setdefault("forecast_models", {})
        for dep, _, _, fm in models:
            stored[dep] = {"spec": fm.spec.to_dict(), "period": period,
                           "params": fm.raw_params if fm.spec.kind == "arima" else
                           {k: v for k, v in fm.raw_params.items()},
                           "predictors": predictors}
        sec.note("", "Model specifications were stored with the dataset for Apply Traditional "
                     "Models.")
    return sec


def _extend_dates(ds: Dataset, extra: int) -> None:
    info = ds.meta.get("dates")
    if not info:
        return
    kind, start = info["kind"], info["start"]
    snapshot = ds.n_cases
    tmp = Dataset(pd.DataFrame({"_x": np.zeros(snapshot)}))
    define_dates(tmp, kind, start)
    for v in tmp.variables:
        if v.name.startswith("_") or not ds.has_name(v.name):
            continue
        col = ds.var(v.name).name
        ds.df[col] = tmp.df[v.name].to_numpy()
