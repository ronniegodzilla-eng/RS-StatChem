"""The rest of the SPSS Compute function library (merged into ``expression.FUNCTIONS``).

Families, with SPSS names and argument order (Command Syntax Reference, "Transformation
expressions"):

* date/time: DATE.DMY/MDY/MOYR/QYR/WKYR/YRDAY, TIME.DAYS/HMS, XDATE.*, DATEDIFF, DATESUM,
  CTIME.*, YRMODA. SPSS date values are seconds since 14 Oct 1582 (proleptic Gregorian),
  computed here with integer civil-calendar arithmetic so no date range limit applies;
* distributions: CDF., PDF., IDF., RV., NCDF., NPDF., SIG. for every SPSS family (SPSS
  parameterisations mapped to scipy: GAMMA/EXP use a rate, NEGBIN counts trials, LNORMAL(a, b)
  has median a and log-SD b, WEIBULL(a, b) has scale a and shape b, …);
* strings: CHAR.INDEX/RINDEX/SUBSTR/LPAD/RPAD, REPLACE, STRUNC, LTRIM/RTRIM with a character,
  INDEX/RINDEX/SUBSTR/LPAD/RPAD (legacy names), STRING/NUMBER with SPSS formats;
* arithmetic/statistical extras: LNGAMMA, CFVAR, RND/TRUNC with a multiple, LAG.
"""
from __future__ import annotations

import math
import re
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import special, stats

from rsstatchem.core import expression as E

DAY = 86400.0
# days between 14 Oct 1582 (SPSS day 0) and 1 Jan 1970
EPOCH_OFFSET_DAYS = 141428
_RNG_HOLDER = {"rng": np.random.default_rng(), "seed": None, "generator": "Mersenne Twister"}


# ------------------------------------------------------------------ plumbing
def _index(args):
    return next((a.index for a in args if isinstance(a, pd.Series)), None)


def _wrap(value, index):
    """ndarray results back to a Series aligned to the data (scalars stay scalars)."""
    if index is None:
        if isinstance(value, np.ndarray):
            value = value.item() if value.size == 1 else value
        return E._clean(float(value)) if isinstance(value, (int, float, np.number)) else value
    out = pd.Series(np.asarray(value, dtype=float).reshape(-1) if np.ndim(value) else
                    np.full(len(index), float(value)), index=index)
    return out.replace([np.inf, -np.inf], np.nan)


def _floats(args, name):
    vals = []
    for a in args:
        if E._is_str(a):
            raise E.ExpressionError(f"{name} expects numeric arguments.")
        vals.append(a.to_numpy(float) if isinstance(a, pd.Series) else
                    (np.nan if a is None else float(a)))
    return vals


def numeric(fn, nmin: int, nmax: int | None = None):
    """A vectorised numeric function of ``nmin``…``nmax`` arguments."""
    nmax = nmin if nmax is None else nmax

    def apply(args, name):
        if not nmin <= len(args) <= nmax:
            want = nmin if nmin == nmax else f"{nmin}–{nmax}"
            raise E.ExpressionError(f"{name} takes {want} arguments.")
        vals = _floats(args, name)
        with np.errstate(all="ignore"):
            out = fn(*vals)
        return _wrap(out, _index(args))
    return apply


def _safe(fn):
    """Evaluate a scipy distribution call; invalid parameters give missing (as in SPSS)."""
    def call(*vals):
        try:
            with np.errstate(all="ignore"):
                out = np.asarray(fn(*vals), dtype=float)
        except (ValueError, OverflowError, ZeroDivisionError):
            out = np.full(np.broadcast(*[np.asarray(v) for v in vals]).shape, np.nan)
        return out
    return call


def _string_arg(x):
    return E._str_series(x)


def _str_out(fn):
    def apply(args, name):
        idx = _index(args)
        if idx is None:
            return fn(*args)
        n = len(idx)
        cols = [a.where(a.notna(), np.nan).tolist() if isinstance(a, pd.Series) else [a] * n
                for a in args]
        return pd.Series([fn(*row) for row in zip(*cols)], index=idx, dtype=object)
    return apply


# ------------------------------------------------------------------ civil calendar
def days_from_civil(y, m, d):
    """Days since 1970-01-01 for proleptic Gregorian dates (vectorised, Hinnant)."""
    y = np.asarray(y, dtype=float)
    m = np.asarray(m, dtype=float)
    d = np.asarray(d, dtype=float)
    y = y - (m <= 2)
    era = np.floor(y / 400)
    yoe = y - era * 400
    mp = (m + 9) % 12
    doy = np.floor((153 * mp + 2) / 5) + d - 1
    doe = yoe * 365 + np.floor(yoe / 4) - np.floor(yoe / 100) + doy
    return era * 146097 + doe - 719468


def civil_from_days(z):
    z = np.asarray(z, dtype=float) + 719468
    era = np.floor(z / 146097)
    doe = z - era * 146097
    yoe = np.floor((doe - np.floor(doe / 1460) + np.floor(doe / 36524)
                    - np.floor(doe / 146096)) / 365)
    y = yoe + era * 400
    doy = doe - (365 * yoe + np.floor(yoe / 4) - np.floor(yoe / 100))
    mp = np.floor((5 * doy + 2) / 153)
    d = doy - np.floor((153 * mp + 2) / 5) + 1
    m = np.where(mp < 10, mp + 3, mp - 9)
    return y + (m <= 2), m, d


def spss_seconds(y, m, d, seconds=0.0):
    return (days_from_civil(y, m, d) + EPOCH_OFFSET_DAYS) * DAY + seconds


def _parts(sec):
    sec = np.asarray(sec, dtype=float)
    days = np.floor(sec / DAY)
    y, m, d = civil_from_days(days - EPOCH_OFFSET_DAYS)
    return y, m, d, days, sec - days * DAY


def _month_length(y, m):
    nxt_y = y + (m == 12)
    nxt_m = np.where(m == 12, 1, m + 1)
    return days_from_civil(nxt_y, nxt_m, 1) - days_from_civil(y, m, 1)


def date_dmy(d, m, y):
    d, m, y = (np.asarray(v, dtype=float) for v in (d, m, y))
    ok = (m >= 1) & (m <= 13) & (d >= 1) & (d <= 31) & (y >= 1) & \
        (np.floor(d) == d) & (np.floor(m) == m)
    y2 = y + np.floor((m - 1) / 12)
    m2 = (m - 1) % 12 + 1
    with np.errstate(invalid="ignore"):
        ok = ok & (d <= _month_length(y2, m2))
    out = (days_from_civil(y2, m2, 1) + d - 1 + EPOCH_OFFSET_DAYS) * DAY
    return np.where(ok, out, np.nan)


def date_from_python(value: datetime) -> float:
    return float(spss_seconds(value.year, value.month, value.day,
                              value.hour * 3600 + value.minute * 60 + value.second))


def to_timestamps(sec: pd.Series) -> pd.Series:
    """SPSS date numbers to pandas datetimes (for date-type variables)."""
    unix = pd.to_numeric(sec, errors="coerce") - EPOCH_OFFSET_DAYS * DAY
    return pd.to_datetime(unix, unit="s", errors="coerce")


def from_timestamps(series: pd.Series) -> pd.Series:
    ns = series.astype("datetime64[ns]")
    unix = (ns - pd.Timestamp("1970-01-01")).dt.total_seconds()
    return unix + EPOCH_OFFSET_DAYS * DAY


_UNITS = {"years": None, "quarters": None, "months": None, "weeks": 7 * DAY, "days": DAY,
          "hours": 3600.0, "minutes": 60.0, "seconds": 1.0}


def _unit(text: str, name: str) -> str:
    u = str(text).strip().lower()
    if u and not u.endswith("s"):
        u += "s"
    if u not in _UNITS:
        raise E.ExpressionError(f"{name}: unknown unit '{text}' (years, quarters, months, "
                                "weeks, days, hours, minutes, seconds).")
    return u


def _datediff(args, name):
    if len(args) != 3 or not isinstance(args[2], str):
        raise E.ExpressionError("DATEDIFF(datetime2, datetime1, \"unit\").")
    unit = _unit(args[2], name)
    a, b = _floats(args[:2], name)
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if _UNITS[unit]:
        out = np.trunc((a - b) / _UNITS[unit])
    else:
        ya, ma, da, _, ta = _parts(a)
        yb, mb, db, _, tb = _parts(b)
        months = (ya - yb) * 12 + (ma - mb)
        later = (da > db) | ((da == db) & (ta >= tb))
        earlier = (da < db) | ((da == db) & (ta <= tb))
        months = np.where((months > 0) & ~later, months - 1, months)
        months = np.where((months < 0) & ~earlier, months + 1, months)
        out = months if unit == "months" else np.trunc(
            months / (3 if unit == "quarters" else 12))
        out = np.where(np.isnan(a) | np.isnan(b), np.nan, out)
    return _wrap(out, _index(args[:2]))


def _datesum(args, name):
    if len(args) not in (3, 4) or not isinstance(args[2], str):
        raise E.ExpressionError("DATESUM(datetime, value, \"unit\"[, \"method\"]).")
    unit = _unit(args[2], name)
    method = str(args[3]).strip().lower() if len(args) == 4 else "closest"
    if method not in ("closest", "rollover"):
        raise E.ExpressionError("DATESUM method must be \"closest\" or \"rollover\".")
    base, n = _floats(args[:2], name)
    base, n = np.asarray(base, dtype=float), np.asarray(n, dtype=float)
    if _UNITS[unit]:
        out = base + n * _UNITS[unit]
    else:
        k = np.trunc(n) * {"months": 1, "quarters": 3, "years": 12}[unit]
        y, m, d, _, t = _parts(base)
        total = y * 12 + (m - 1) + k
        ny, nm = np.floor(total / 12), total % 12 + 1
        length = _month_length(ny, nm)
        if method == "closest":
            nd = np.minimum(d, length)
            out = (days_from_civil(ny, nm, nd) + EPOCH_OFFSET_DAYS) * DAY + t
        else:
            out = (days_from_civil(ny, nm, 1) + d - 1 + EPOCH_OFFSET_DAYS) * DAY + t
    return _wrap(out, _index(args[:2]))


def _xdate(part: str):
    def fn(sec):
        y, m, d, days, t = _parts(sec)
        if part == "year":
            return y
        if part == "month":
            return m
        if part == "mday":
            return d
        if part == "quarter":
            return np.floor((m - 1) / 3) + 1
        jday = days - EPOCH_OFFSET_DAYS - days_from_civil(y, 1, 1) + 1
        if part == "jday":
            return jday
        if part == "week":
            return np.minimum(np.floor((jday - 1) / 7) + 1, 53)
        if part == "wkday":  # 1 = Sunday; 14 Oct 1582 was a Thursday
            return (days + 4) % 7 + 1
        if part == "date":
            return days * DAY
        if part == "tday":
            return days
        if part == "time":
            return t
        if part == "hour":
            return np.floor(t / 3600)
        if part == "minute":
            return np.floor((t % 3600) / 60)
        if part == "second":
            return t % 60
        raise ValueError(part)
    return fn


# ------------------------------------------------------------------ strings
def _char_index(hay, needle, divisor=None):
    if hay is None or needle is None or (isinstance(hay, float) and math.isnan(hay)):
        return float("nan")
    hay, needle = str(hay), str(needle)
    if divisor is None or (isinstance(divisor, float) and math.isnan(divisor)):
        return float(hay.find(needle) + 1) if needle else 0.0
    k = int(divisor)
    if k <= 0 or len(needle) % k:
        return float("nan")
    parts = [needle[i:i + k] for i in range(0, len(needle), k)]
    hits = [hay.find(p) for p in parts if hay.find(p) >= 0]
    return float(min(hits) + 1) if hits else 0.0


def _char_rindex(hay, needle, divisor=None):
    if hay is None or needle is None:
        return float("nan")
    hay, needle = str(hay), str(needle)
    if divisor is None or (isinstance(divisor, float) and math.isnan(divisor)):
        return float(hay.rfind(needle) + 1) if needle else 0.0
    k = int(divisor)
    if k <= 0 or len(needle) % k:
        return float("nan")
    parts = [needle[i:i + k] for i in range(0, len(needle), k)]
    hits = [hay.rfind(p) for p in parts if hay.rfind(p) >= 0]
    return float(max(hits) + 1) if hits else 0.0


def _substr(s, pos, length=None):
    s = "" if s is None or (isinstance(s, float) and math.isnan(s)) else str(s)
    if pos is None or (isinstance(pos, float) and math.isnan(pos)) or pos < 1 or pos > len(s):
        return ""
    start = int(pos) - 1
    if length is None or (isinstance(length, float) and math.isnan(length)):
        return s[start:]
    if length < 0:
        return ""
    return s[start:start + int(length)]


def _pad(left: bool):
    def fn(s, length, char=" "):
        s = "" if s is None or (isinstance(s, float) and math.isnan(s)) else str(s)
        char = (str(char) or " ")[0]
        if length is None or (isinstance(length, float) and math.isnan(length)):
            return s
        n = int(length)
        if len(s) >= n:
            return s
        return s.rjust(n, char) if left else s.ljust(n, char)
    return fn


def _trim(left: bool):
    def fn(s, char=" "):
        s = "" if s is None or (isinstance(s, float) and math.isnan(s)) else str(s)
        char = (str(char) or " ")[0]
        return s.lstrip(char) if left else s.rstrip(char)
    return fn


def _replace(s, old, new, count=None):
    s = "" if s is None or (isinstance(s, float) and math.isnan(s)) else str(s)
    if not old:
        return s
    if count is None or (isinstance(count, float) and math.isnan(count)):
        return s.replace(str(old), str(new))
    return s.replace(str(old), str(new), max(int(count), 0))


def _strunc(s, length):
    s = "" if s is None or (isinstance(s, float) and math.isnan(s)) else str(s)
    return s[:max(int(length), 0)].rstrip()


_FORMAT = re.compile(r"^([A-Z]+?)(\d+)(?:\.(\d+))?$", re.IGNORECASE)
FORMAT_TOKEN = re.compile(
    r"^(F|N|E|COMMA|DOT|DOLLAR|PCT|Z|A|AHEX|DATE|ADATE|EDATE|SDATE|JDATE|QYR|MOYR|WKYR|"
    r"DATETIME|YMDHMS|TIME|DTIME|MTIME|WKDAY|MONTH)\d+(\.\d+)?$", re.IGNORECASE)
_MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


def format_value(v, fmt: str) -> str:
    """A number printed with an SPSS output format (F8.2, N5, COMMA9.2, DATE11, ADATE10…)."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    m = _FORMAT.match(fmt.strip())
    if not m:
        raise E.ExpressionError(f"Unknown format {fmt}.")
    kind, width, dec = m.group(1).upper(), int(m.group(2)), int(m.group(3) or 0)
    v = float(v)
    if kind in ("F", "E", "N", "COMMA", "DOT", "DOLLAR", "PCT", "Z"):
        if kind == "E":
            text = f"{v:.{dec}E}"
        elif kind == "N":
            text = f"{int(round(abs(v))):0{width}d}"
        elif kind in ("COMMA", "DOLLAR"):
            text = ("$" if kind == "DOLLAR" else "") + f"{v:,.{dec}f}"
        elif kind == "DOT":
            text = f"{v:,.{dec}f}".replace(",", "_").replace(".", ",").replace("_", ".")
        elif kind == "PCT":
            text = f"{v:.{dec}f}%"
        else:
            text = f"{v:.{dec}f}"
        return text.rjust(width) if len(text) <= width else "*" * width
    y, mo, d, days, t = (float(np.asarray(x)) for x in _parts(v))
    y, mo, d = int(y), int(mo), int(d)
    hh, mm, ss = int(t // 3600), int(t % 3600 // 60), t % 60
    if kind == "DATE":
        return f"{d:02d}-{_MONTHS[mo - 1]}-{y:04d}" if width >= 11 else \
            f"{d:02d}-{_MONTHS[mo - 1]}-{y % 100:02d}"
    if kind == "ADATE":
        return f"{mo:02d}/{d:02d}/{y:04d}" if width >= 10 else f"{mo:02d}/{d:02d}/{y % 100:02d}"
    if kind == "EDATE":
        return f"{d:02d}.{mo:02d}.{y:04d}" if width >= 10 else f"{d:02d}.{mo:02d}.{y % 100:02d}"
    if kind == "SDATE":
        return f"{y:04d}/{mo:02d}/{d:02d}"
    if kind == "JDATE":
        jday = int(_xdate("jday")(v))
        return f"{y:04d}{jday:03d}"
    if kind == "QYR":
        return f"{(mo - 1) // 3 + 1} Q {y:04d}"
    if kind == "MOYR":
        return f"{_MONTHS[mo - 1]} {y:04d}"
    if kind == "WKYR":
        return f"{int(_xdate('week')(v)):2d} WK {y:04d}"
    if kind in ("DATETIME", "YMDHMS"):
        base = f"{d:02d}-{_MONTHS[mo - 1]}-{y:04d}" if kind == "DATETIME" else \
            f"{y:04d}-{mo:02d}-{d:02d}"
        return f"{base} {hh:02d}:{mm:02d}:{int(ss):02d}"
    if kind in ("TIME", "MTIME"):
        total = v
        h = int(total // 3600)
        return f"{h:d}:{int(total % 3600 // 60):02d}" + \
            (f":{int(total % 60):02d}" if width >= 8 else "")
    if kind == "DTIME":
        dd = int(v // DAY)
        rest = v - dd * DAY
        return f"{dd} {int(rest // 3600):02d}:{int(rest % 3600 // 60):02d}:{int(rest % 60):02d}"
    if kind == "WKDAY":
        return ["SUNDAY", "MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY",
                "SATURDAY"][int(v) - 1][:width] if 1 <= v <= 7 else ""
    if kind == "MONTH":
        return ["JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE", "JULY", "AUGUST",
                "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER"][int(v) - 1][:width] \
            if 1 <= v <= 12 else ""
    raise E.ExpressionError(f"Format {fmt} cannot be used here.")


def parse_value(text, fmt: str | None) -> float:
    """A string read with an SPSS input format (F, COMMA, DOT, DATE, ADATE, EDATE, SDATE…)."""
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return float("nan")
    s = str(text).strip()
    if not s:
        return float("nan")
    kind = (_FORMAT.match(fmt.strip()).group(1).upper() if fmt and _FORMAT.match(fmt.strip())
            else "F")
    try:
        if kind in ("F", "N", "E", "Z"):
            return float(s.replace(",", ".")) if s.count(",") == 1 and "." not in s \
                else float(s)
        if kind in ("COMMA", "DOLLAR"):
            return float(s.replace("$", "").replace(",", ""))
        if kind == "DOT":
            return float(s.replace(".", "").replace(",", "."))
        if kind == "PCT":
            return float(s.rstrip("%"))
        patterns = {"DATE": ["%d-%b-%Y", "%d-%b-%y", "%d %b %Y", "%d/%b/%Y"],
                    "ADATE": ["%m/%d/%Y", "%m/%d/%y"], "EDATE": ["%d.%m.%Y", "%d.%m.%y",
                                                                 "%d/%m/%Y"],
                    "SDATE": ["%Y/%m/%d", "%Y-%m-%d"],
                    "DATETIME": ["%d-%b-%Y %H:%M:%S", "%d-%b-%Y %H:%M"],
                    "YMDHMS": ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"]}
        if kind in patterns:
            for p in patterns[kind]:
                try:
                    return date_from_python(datetime.strptime(s.title() if "%b" in p else s,
                                                              p))
                except ValueError:
                    continue
            return float("nan")
        if kind == "TIME":
            parts = [float(x) for x in s.split(":")]
            parts += [0.0] * (3 - len(parts))
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
    except ValueError:
        return float("nan")
    return float("nan")


def _string_fn(args, name):
    if not 1 <= len(args) <= 2:
        raise E.ExpressionError("STRING(numexpr, format).")
    x = args[0]
    fmt = args[1] if len(args) > 1 else None
    if fmt is None:
        return E._string([x], name)
    if not isinstance(fmt, str):
        return E._string([x, fmt], name)

    def conv(v):
        return format_value(v, fmt).strip()
    return x.map(conv).astype(object) if isinstance(x, pd.Series) else conv(x)


def _number_fn(args, name):
    if not 1 <= len(args) <= 2:
        raise E.ExpressionError("NUMBER(strexpr, format).")
    if len(args) == 1:
        return E._number(args, name)
    x, fmt = args
    if isinstance(x, pd.Series):
        return x.map(lambda v: parse_value(v, str(fmt))).astype(float)
    return parse_value(x, str(fmt))


# ------------------------------------------------------------------ random numbers
def set_seed(seed: int | None = None, generator: str = "Mersenne Twister") -> None:
    """Random Number Generators: a fixed seed makes RV.* reproducible (None = random)."""
    _RNG_HOLDER["rng"] = np.random.default_rng(seed)
    _RNG_HOLDER["seed"] = seed
    _RNG_HOLDER["generator"] = generator
    E._RNG = _RNG_HOLDER["rng"]


def rng() -> np.random.Generator:
    return _RNG_HOLDER["rng"]


def random(draw, nargs: int):
    """RV.* with scalar or per-case parameters."""
    def apply(args, name, n=1, index=None):
        if len(args) != nargs:
            raise E.ExpressionError(f"{name} takes {nargs} arguments.")
        vals = [a.to_numpy(float) if isinstance(a, pd.Series) else float(a) for a in args]
        bad = np.zeros(n, dtype=bool)
        for v in vals:
            bad |= np.isnan(np.broadcast_to(np.asarray(v, dtype=float), (n,)))
        clean = [np.where(np.isnan(v), 1.0, v) if isinstance(v, np.ndarray) else v
                 for v in vals]
        try:
            with np.errstate(all="ignore"):
                out = np.asarray(draw(rng(), *clean, size=n), dtype=float)
        except ValueError:
            out = np.full(n, np.nan)
        out = np.where(bad, np.nan, out)
        return pd.Series(out, index=index)
    apply.random = True
    return apply


# ------------------------------------------------------------------ distribution table
# SPSS family -> (scipy frozen-distribution factory from SPSS parameters, n parameters,
# discrete?, parameter text)
def _negbin_shift(r, p):
    return stats.nbinom(r, p, loc=r)


FAMILIES = {
    "BERNOULLI": (lambda p: stats.bernoulli(p), 1, True, "prob"),
    "BETA": (lambda a, b: stats.beta(a, b), 2, False, "shape1, shape2"),
    "BINOM": (lambda n, p: stats.binom(n, p), 2, True, "n, prob"),
    "CAUCHY": (lambda loc, scale: stats.cauchy(loc, scale), 2, False, "loc, scale"),
    "CHISQ": (lambda df: stats.chi2(df), 1, False, "df"),
    "EXP": (lambda rate: stats.expon(scale=1 / rate), 1, False, "rate"),
    "F": (lambda d1, d2: stats.f(d1, d2), 2, False, "df1, df2"),
    "GAMMA": (lambda shape, rate: stats.gamma(shape, scale=1 / rate), 2, False,
              "shape, rate (mean = shape / rate)"),
    "GEOM": (lambda p: stats.geom(p), 1, True, "prob (trials to first success)"),
    "HALFNRM": (lambda loc, scale: stats.halfnorm(loc, scale), 2, False, "mean, scale"),
    "HYPER": (lambda total, sample, hits: stats.hypergeom(total, hits, sample), 3, True,
              "total, sample, hits"),
    "IGAUSS": (lambda mu, lam: stats.invgauss(mu / lam, scale=lam), 2, False, "mean, shape"),
    "LAPLACE": (lambda loc, scale: stats.laplace(loc, scale), 2, False, "mean, scale"),
    "LOGISTIC": (lambda loc, scale: stats.logistic(loc, scale), 2, False, "mean, scale"),
    "LNORMAL": (lambda a, b: stats.lognorm(b, scale=a), 2, False, "a (median), b (log SD)"),
    "NEGBIN": (_negbin_shift, 2, True, "threshold, prob (trials to threshold successes)"),
    "NORMAL": (lambda m, s: stats.norm(m, s), 2, False, "mean, stddev"),
    "PARETO": (lambda threshold, shape: stats.pareto(shape, scale=threshold), 2, False,
               "threshold, shape"),
    "POISSON": (lambda mean: stats.poisson(mean), 1, True, "mean"),
    "SRANGE": (lambda k, df: stats.studentized_range(k, df), 2, False, "k (means), df"),
    "T": (lambda df: stats.t(df), 1, False, "df"),
    "UNIFORM": (lambda lo, hi: stats.uniform(lo, hi - lo), 2, False, "min, max"),
    "WEIBULL": (lambda a, b: stats.weibull_min(b, scale=a), 2, False, "a (scale), b (shape)"),
}
def _frozen_call(family: str, method: str):
    factory, npar, discrete, _ = FAMILIES[family]

    def fn(q, *params):
        q = np.asarray(q, dtype=float)
        shape = np.broadcast(q, *[np.asarray(p) for p in params]).shape
        out = np.full(shape, np.nan)
        qb = np.broadcast_to(q, shape)
        pb = [np.broadcast_to(np.asarray(p, dtype=float), shape) for p in params]
        # group identical parameter sets (usually one) to build each frozen distribution once
        keys = np.stack([p.reshape(-1) for p in pb], axis=1) if pb else np.zeros((qb.size, 0))
        flat_q = qb.reshape(-1)
        flat_out = out.reshape(-1)
        if keys.shape[1]:
            uniq, inverse = np.unique(keys, axis=0, return_inverse=True)
        else:
            uniq, inverse = np.zeros((1, 0)), np.zeros(flat_q.size, dtype=int)
        for k, row in enumerate(uniq):
            mask = inverse.reshape(-1) == k
            if np.isnan(row).any():
                continue
            try:
                with np.errstate(all="ignore"):
                    dist = factory(*row)
                    values = flat_q[mask]
                    if method == "pdf":
                        res = dist.pmf(values) if discrete else dist.pdf(values)
                    else:
                        res = getattr(dist, method)(values)
            except (ValueError, ZeroDivisionError, OverflowError):
                continue
            flat_out[mask] = res
        if method in ("ppf",):
            flat_out[(flat_q < 0) | (flat_q > 1)] = np.nan
        return flat_out.reshape(shape)
    return fn, npar


def _bvnor_cdf(x, y, r):
    x, y, r = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float),
                                  np.asarray(r, float))
    out = np.full(x.shape, np.nan)
    for i in np.ndindex(x.shape):
        if np.isnan([x[i], y[i], r[i]]).any() or abs(r[i]) > 1:
            continue
        cov = [[1, r[i]], [r[i], 1]]
        out[i] = stats.multivariate_normal(mean=[0, 0], cov=cov,
                                           allow_singular=True).cdf([x[i], y[i]])
    return out


def _bvnor_pdf(x, y, r):
    x, y, r = (np.asarray(v, float) for v in (x, y, r))
    with np.errstate(all="ignore"):
        z = (x ** 2 - 2 * r * x * y + y ** 2) / (1 - r ** 2)
        return np.exp(-z / 2) / (2 * np.pi * np.sqrt(1 - r ** 2))


def _build_distributions() -> dict:
    out = {}
    for fam, (factory, npar, discrete, ptext) in FAMILIES.items():
        cdf, _ = _frozen_call(fam, "cdf")
        pdf, _ = _frozen_call(fam, "pdf")
        out[f"CDF.{fam}"] = (numeric(cdf, npar + 1), "CDF & Noncentral CDF",
                             f"CDF.{fam}(quant, {ptext}) — cumulative probability")
        out[f"PDF.{fam}"] = (numeric(pdf, npar + 1), "PDF & Noncentral PDF",
                             f"PDF.{fam}(quant, {ptext}) — "
                             + ("probability" if discrete else "density"))
        if not discrete:
            ppf, _ = _frozen_call(fam, "ppf")
            out[f"IDF.{fam}"] = (numeric(ppf, npar + 1), "Inverse DF",
                                 f"IDF.{fam}(prob, {ptext}) — quantile")
        if fam != "SRANGE":
            out[f"RV.{fam}"] = (random(_rv_draw(fam), npar), "Random Numbers",
                                f"RV.{fam}({ptext}) — random value")
    out["CDF.BVNOR"] = (numeric(_bvnor_cdf, 3), "CDF & Noncentral CDF",
                        "CDF.BVNOR(quant1, quant2, corr) — bivariate standard normal")
    out["PDF.BVNOR"] = (numeric(_bvnor_pdf, 3), "PDF & Noncentral PDF",
                        "PDF.BVNOR(quant1, quant2, corr) — bivariate normal density")
    nc = {
        "BETA": (lambda q, a, b, lam: _ncbeta(q, a, b, lam, "cdf"),
                 lambda q, a, b, lam: _ncbeta(q, a, b, lam, "pdf"), "shape1, shape2, nc"),
        "CHISQ": (_safe(stats.ncx2.cdf), _safe(stats.ncx2.pdf), "df, nc"),
        "F": (_safe(stats.ncf.cdf), _safe(stats.ncf.pdf), "df1, df2, nc"),
        "T": (_safe(stats.nct.cdf), _safe(stats.nct.pdf), "df, nc"),
    }
    for fam, (cdf, pdf, ptext) in nc.items():
        npar = ptext.count(",") + 1
        out[f"NCDF.{fam}"] = (numeric(cdf, npar + 1), "CDF & Noncentral CDF",
                              f"NCDF.{fam}(quant, {ptext}) — noncentral cumulative prob.")
        out[f"NPDF.{fam}"] = (numeric(pdf, npar + 1), "PDF & Noncentral PDF",
                              f"NPDF.{fam}(quant, {ptext}) — noncentral density")
    out["SIG.CHISQ"] = (numeric(_safe(stats.chi2.sf), 2), "Significance",
                        "SIG.CHISQ(quant, df) — upper-tail probability")
    out["SIG.F"] = (numeric(_safe(stats.f.sf), 3), "Significance",
                    "SIG.F(quant, df1, df2) — upper-tail probability")
    return out


def _ncbeta(q, a, b, lam, what):
    """Noncentral beta as a Poisson mixture of central betas (as SPSS defines it)."""
    q, a, b, lam = np.broadcast_arrays(*(np.asarray(v, float) for v in (q, a, b, lam)))
    total = np.zeros(q.shape)
    with np.errstate(all="ignore"):
        for j in range(400):
            w = stats.poisson.pmf(j, lam / 2)
            term = stats.beta.cdf(q, a + j, b) if what == "cdf" else stats.beta.pdf(q, a + j, b)
            total = total + w * term
            if np.nanmax(np.where(np.isnan(w), 0, w)) < 1e-14 and j > np.nanmax(lam):
                break
    return np.where((lam < 0) | (a <= 0) | (b <= 0), np.nan, total)


def _rv_draw(fam: str):
    draws = {
        "BERNOULLI": lambda g, p, size: g.binomial(1, p, size),
        "BETA": lambda g, a, b, size: g.beta(a, b, size),
        "BINOM": lambda g, n, p, size: g.binomial(np.asarray(n).astype(int), p, size),
        "CAUCHY": lambda g, loc, scale, size: loc + scale * g.standard_cauchy(size),
        "CHISQ": lambda g, df, size: g.chisquare(df, size),
        "EXP": lambda g, rate, size: g.exponential(1 / np.asarray(rate), size),
        "F": lambda g, d1, d2, size: g.f(d1, d2, size),
        "GAMMA": lambda g, shape, rate, size: g.gamma(shape, 1 / np.asarray(rate), size),
        "GEOM": lambda g, p, size: g.geometric(p, size),
        "HALFNRM": lambda g, loc, scale, size: loc + scale * np.abs(g.standard_normal(size)),
        "HYPER": lambda g, total, sample, hits, size: g.hypergeometric(
            np.asarray(hits).astype(int), (np.asarray(total) - np.asarray(hits)).astype(int),
            np.asarray(sample).astype(int), size),
        "IGAUSS": lambda g, mu, lam, size: g.wald(mu, lam, size),
        "LAPLACE": lambda g, loc, scale, size: g.laplace(loc, scale, size),
        "LOGISTIC": lambda g, loc, scale, size: g.logistic(loc, scale, size),
        "LNORMAL": lambda g, a, b, size: g.lognormal(np.log(a), b, size),
        "NEGBIN": lambda g, r, p, size: g.negative_binomial(r, p, size) + np.asarray(r),
        "NORMAL": lambda g, m, s, size: g.normal(m, s, size),
        "PARETO": lambda g, threshold, shape, size: threshold * (1 + g.pareto(shape, size)),
        "POISSON": lambda g, mean, size: g.poisson(mean, size),
        "T": lambda g, df, size: g.standard_t(df, size),
        "UNIFORM": lambda g, lo, hi, size: g.uniform(lo, hi, size),
        "WEIBULL": lambda g, a, b, size: a * g.weibull(b, size),
    }
    return draws[fam]


# ------------------------------------------------------------------ assembled table
def _cfvar(args, name, min_valid=None):
    df = E._frame(args, name)
    need = 2 if min_valid is None else min_valid
    out = df.std(axis=1, ddof=1) / df.mean(axis=1)
    return out.where(df.notna().sum(axis=1) >= need).replace([np.inf, -np.inf], np.nan)


def _rnd(args, name):
    if not 1 <= len(args) <= 3:
        raise E.ExpressionError("RND(numexpr[, mult, fuzzbits]).")
    return E._rnd(args[:2], name)


def _trunc(args, name):
    if not 1 <= len(args) <= 3:
        raise E.ExpressionError("TRUNC(numexpr[, mult, fuzzbits]).")
    x = E._num(args[0], name)
    mult = E._num(args[1], name) if len(args) > 1 else 1.0
    with np.errstate(all="ignore"):
        return E._clean(np.trunc(x / mult) * mult)


def _lag(args, name):
    if not 1 <= len(args) <= 2:
        raise E.ExpressionError("LAG(variable[, ncases]).")
    x = args[0]
    n = int(args[1]) if len(args) > 1 else 1
    if not isinstance(x, pd.Series):
        raise E.ExpressionError("LAG needs a variable.")
    out = x.shift(n)
    return out.where(out.notna(), "") if x.dtype == object else out


EXTRA_FUNCTIONS: dict = {
    # arithmetic
    "LNGAMMA": (numeric(special.gammaln, 1), "Arithmetic", "LNGAMMA(numexpr) — log gamma"),
    "RND": (_rnd, "Arithmetic", "RND(numexpr[, mult]) — round (half away from zero)"),
    "TRUNC": (_trunc, "Arithmetic", "TRUNC(numexpr[, mult]) — truncate toward zero"),
    "CFVAR": (_cfvar, "Statistical", "CFVAR(numexpr, numexpr[, …]) — coefficient of variation"),
    "LAG": (_lag, "Miscellaneous", "LAG(variable[, ncases]) — value of a previous case"),
    # strings
    "CHAR.INDEX": (_str_out(_char_index), "String",
                   "CHAR.INDEX(haystack, needle[, divisor]) — first position (0 = absent)"),
    "CHAR.RINDEX": (_str_out(_char_rindex), "String",
                    "CHAR.RINDEX(haystack, needle[, divisor]) — last position"),
    "CHAR.SUBSTR": (_str_out(_substr), "String",
                    "CHAR.SUBSTR(strexpr, pos[, length]) — part of a string"),
    "CHAR.LPAD": (_str_out(_pad(True)), "String",
                  "CHAR.LPAD(strexpr, length[, char]) — pad on the left"),
    "CHAR.RPAD": (_str_out(_pad(False)), "String",
                  "CHAR.RPAD(strexpr, length[, char]) — pad on the right"),
    "LTRIM": (_str_out(_trim(True)), "String", "LTRIM(strexpr[, char]) — remove leading chars"),
    "RTRIM": (_str_out(_trim(False)), "String",
              "RTRIM(strexpr[, char]) — remove trailing chars"),
    "REPLACE": (_str_out(_replace), "String",
                "REPLACE(strexpr, old, new[, count]) — replace text"),
    "STRUNC": (_str_out(_strunc), "String", "STRUNC(strexpr, length) — truncate"),
    "NTRIM": (_str_out(lambda s: "" if s is None else str(s).rstrip()), "String",
              "NTRIM(variable) — value without trailing blanks"),
    "STRING": (_string_fn, "Conversion", "STRING(numexpr, format) — number to text, e.g. F8.2"),
    "NUMBER": (_number_fn, "Conversion",
               "NUMBER(strexpr[, format]) — text to number, e.g. F8.2, DATE11"),
    # date creation
    "DATE.DMY": (numeric(date_dmy, 3), "Date Creation", "DATE.DMY(day, month, year)"),
    "DATE.MDY": (numeric(lambda m, d, y: date_dmy(d, m, y), 3), "Date Creation",
                 "DATE.MDY(month, day, year)"),
    "DATE.MOYR": (numeric(lambda m, y: date_dmy(np.ones_like(np.asarray(m, float)), m, y), 2),
                  "Date Creation", "DATE.MOYR(month, year)"),
    "DATE.QYR": (numeric(lambda q, y: np.where((np.asarray(q) >= 1) & (np.asarray(q) <= 4),
                                               date_dmy(np.ones_like(np.asarray(q, float)),
                                                        3 * np.asarray(q, float) - 2, y),
                                               np.nan), 2),
                 "Date Creation", "DATE.QYR(quarter, year)"),
    "DATE.WKYR": (numeric(lambda w, y: np.where(
        (np.asarray(w) >= 1) & (np.asarray(w) <= 53),
        date_dmy(np.ones_like(np.asarray(w, float)), np.ones_like(np.asarray(w, float)), y)
        + (np.asarray(w, float) - 1) * 7 * DAY, np.nan), 2),
        "Date Creation", "DATE.WKYR(weeknum, year)"),
    "DATE.YRDAY": (numeric(lambda y, d: np.where(
        (np.asarray(d) >= 1) & (np.asarray(d) <= 366),
        date_dmy(np.ones_like(np.asarray(d, float)), np.ones_like(np.asarray(d, float)), y)
        + (np.asarray(d, float) - 1) * DAY, np.nan), 2),
        "Date Creation", "DATE.YRDAY(year, daynum)"),
    "YRMODA": (numeric(lambda y, m, d: date_dmy(d, m, y) / DAY, 3), "Date Creation",
               "YRMODA(year, month, day) — day number (15 Oct 1582 = 1)"),
    "TIME.DAYS": (numeric(lambda d: np.asarray(d, float) * DAY, 1),
                  "Time Duration Creation", "TIME.DAYS(days) — duration in seconds"),
    "TIME.HMS": (numeric(lambda h, m=0.0, s=0.0: np.asarray(h, float) * 3600
                         + np.asarray(m, float) * 60 + np.asarray(s, float), 1, 3),
                 "Time Duration Creation", "TIME.HMS(hours[, minutes, seconds])"),
    "CTIME.DAYS": (numeric(lambda t: np.asarray(t, float) / DAY, 1),
                   "Time Duration Extraction", "CTIME.DAYS(timevalue) — days"),
    "CTIME.HOURS": (numeric(lambda t: np.asarray(t, float) / 3600, 1),
                    "Time Duration Extraction", "CTIME.HOURS(timevalue) — hours"),
    "CTIME.MINUTES": (numeric(lambda t: np.asarray(t, float) / 60, 1),
                      "Time Duration Extraction", "CTIME.MINUTES(timevalue) — minutes"),
    "CTIME.SECONDS": (numeric(lambda t: np.asarray(t, float), 1),
                      "Time Duration Extraction", "CTIME.SECONDS(timevalue) — seconds"),
    "DATEDIFF": (_datediff, "Date Arithmetic",
                 "DATEDIFF(datetime2, datetime1, \"unit\") — difference in whole units"),
    "DATESUM": (_datesum, "Date Arithmetic",
                "DATESUM(datetime, value, \"unit\"[, \"closest\"|\"rollover\"]) — add time"),
    # legacy names
    "INDEX": (_str_out(_char_index), "String", "INDEX(haystack, needle[, divisor])"),
    "RINDEX": (_str_out(_char_rindex), "String", "RINDEX(haystack, needle[, divisor])"),
    "SUBSTR": (_str_out(_substr), "String", "SUBSTR(strexpr, pos[, length])"),
    "LPAD": (_str_out(_pad(True)), "String", "LPAD(strexpr, length[, char])"),
    "RPAD": (_str_out(_pad(False)), "String", "RPAD(strexpr, length[, char])"),
    "NORMAL": (random(lambda g, s, size: g.normal(0, s, size), 1), "Random Numbers",
               "NORMAL(stddev) — random normal value with mean 0"),
    "UNIFORM": (random(lambda g, hi, size: g.uniform(0, hi, size), 1), "Random Numbers",
                "UNIFORM(max) — random uniform value from 0 to max"),
}
for _part in ("date", "hour", "jday", "mday", "minute", "month", "quarter", "second", "tday",
              "time", "week", "wkday", "year"):
    EXTRA_FUNCTIONS[f"XDATE.{_part.upper()}"] = (
        numeric(_xdate(_part), 1), "Date Extraction",
        f"XDATE.{_part.upper()}(datevalue) — "
        + {"date": "date part (seconds)", "hour": "hour (0–23)", "jday": "day of year",
           "mday": "day of month", "minute": "minute", "month": "month (1–12)",
           "quarter": "quarter (1–4)", "second": "second", "tday": "days since 14 Oct 1582",
           "time": "time of day (seconds)", "week": "week of year (1–53)",
           "wkday": "weekday (1 = Sunday)", "year": "four-digit year"}[_part])
EXTRA_FUNCTIONS.update(_build_distributions())

# System variables (no parentheses): name -> (group, description)
SYSTEM_VARIABLES = {
    "$CASENUM": "Case number (1, 2, …)",
    "$SYSMIS": "System-missing value",
    "$JDATE": "Today as days since 14 Oct 1582",
    "$TIME": "Current date and time (SPSS date value)",
    "$DATE": "Today as text, dd-MMM-yy",
    "$DATE11": "Today as text, dd-MMM-yyyy",
}


def system_value(name: str, index: pd.Index):
    now = datetime.now()
    if name == "$CASENUM":
        return pd.Series(np.arange(1, len(index) + 1, dtype=float), index=index)
    if name == "$SYSMIS":
        return float("nan")
    if name == "$JDATE":
        return float(np.floor(date_from_python(now) / DAY))
    if name == "$TIME":
        return date_from_python(now)
    if name == "$DATE":
        return format_value(date_from_python(now), "DATE9")
    if name == "$DATE11":
        return format_value(date_from_python(now), "DATE11")
    raise E.ExpressionError(f"Unknown system variable {name}.")


def _install() -> None:
    E.FUNCTIONS.update(EXTRA_FUNCTIONS)
    E.FUNCTIONS["VALUELABEL"] = (None, "Miscellaneous",
                                 "VALUELABEL(variable) — value label of the value (text)")
    for name, group in (("CDFNORM", "CDF & Noncentral CDF"), ("PROBIT", "Inverse DF")):
        impl, _, desc = E.FUNCTIONS[name]
        E.FUNCTIONS[name] = (impl, group, desc)
    groups = ["All"] + sorted({g for _, g, _ in E.FUNCTIONS.values()})
    if hasattr(E, "FUNCTION_GROUPS"):
        E.FUNCTION_GROUPS[:] = groups
    else:
        E.FUNCTION_GROUPS = groups


_install()
