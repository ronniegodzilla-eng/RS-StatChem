"""SPSS Complex Samples: analysis plan, sample selection and the design-based procedures
CSFREQUENCIES / CSDESCRIPTIVES / CSTABULATE (Crosstabs) / CSRATIO (Ratios) / CSGLM /
CSLOGISTIC / CSORDINAL / CSCOXREG.

Variance estimation is Taylor-series linearization exactly as the IBM SPSS Statistics
Algorithms chapters "Complex Samples: Covariance Matrix of Total", "CSDESCRIPTIVES",
"CSTABULATE", "CSGLM", "CSLOGISTIC", "CSORDINAL", "CSCOXREG" and "Complex Samples: Model
Testing" document it:

* the covariance of an estimated total of (linearized) values z is
  ``sum_h V_h`` with ``V_h = n_h/(n_h - 1) sum_i (z_hi - zbar_h)(z_hi - zbar_h)'`` over the
  weighted PSU totals z_hi for sampling with replacement, times (1 - f_h) for equal-probability
  sampling without replacement, and the Yates-Grundy-Sen form for unequal-probability sampling
  without replacement (joint inclusion probabilities from Brewer's approximation
  ``pi_ij = pi_i pi_j (c_i + c_j)/2, c_i = (n_h - 1)/(n_h - pi_i)``, because SPSS reads them from
  a CSSELECT file); strata with a single PSU contribute zero; with a without-replacement first
  stage a second stage adds ``sum_hi pi_hi V_hi(2)``;
* ratios, means and proportions are linearized (Woodruff 1971): ``z = (y - R x)/X``;
  domains (subpopulations) set z = 0 outside the domain but keep the full design;
* design effect Deff = V / V_srs with ``V_srs = fpc * N/(n - 1) * sum_i w_i (z_i - Z/N)^2``,
  fpc = 1 - n/N for the (default) without-replacement SRS assumption;
* model parameters (CSGLM, CSLOGISTIC, CSORDINAL, CSCOXREG) use the Binder (1983) sandwich
  ``H^-1 V(sum w_i s_i) H^-1`` with the score contributions s_i (CSCOXREG: score residuals,
  Breslow or Efron ties) and V_srs from ``fpc * N/(n - 1) * sum w_i s_i s_i'``;
* tests: Wald F (Fellegi) ``F = (nu - r + 1)/(r nu) X^2`` on (r, nu - r + 1) df, adjusted F,
  Wald chi-square and adjusted chi-square (Rao-Scott second order); crosstab independence by
  the Rao-Scott F-based second-order adjusted Pearson and likelihood-ratio statistics with
  IBM's Delta matrix; design df = #PSU - #strata.

Verification (tests/test_wp10_complex_samples.py): Lohr's one-stage cluster GPA example as
analysed with the R survey package in the SDaA vignette (total 1130.4, SE 67.167); textbook
stratified formulas; statsmodels cluster-robust GLM/logit; lifelines robust Cox.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize, stats

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import fmt, fmt_int, fmt_p, value_text
from rsstatchem.core.stats.wp4_util import Result, sort_key

METHODS = {"WR": "Sampling with replacement", "EQUAL_WOR": "Equal probability sampling "
           "without replacement", "UNEQUAL_WOR": "Unequal probability sampling without "
           "replacement"}
TESTS = {"F": "Wald F", "ADJF": "Adjusted Wald F", "CHISQUARE": "Wald Chi-Square",
         "ADJCHISQUARE": "Adjusted Wald Chi-Square"}
LINKS = {"logit": "Logit", "probit": "Probit", "cloglog": "Complementary Log-log",
         "nloglog": "Negative Log-log", "cauchit": "Cauchit"}
PLAN_KEY = "csplan"


# ===================================================================== plan
@dataclass
class Stage:
    strata: list = field(default_factory=list)
    clusters: list = field(default_factory=list)
    method: str = "WR"
    fpc_kind: str = "none"        # none | rate | popsize | inclusion (variable or value)
    fpc_value: float | None = None
    fpc_var: str | None = None


@dataclass
class Plan:
    """A Complex Samples analysis plan (what SPSS stores in a .csplan file)."""
    weight: str | None = None
    stages: list = field(default_factory=lambda: [Stage()])
    srs_estimator: str = "WOR"    # assumption for design effects
    df: int | None = None         # user-specified design degrees of freedom

    def to_dict(self) -> dict:
        return {"weight": self.weight, "srs_estimator": self.srs_estimator, "df": self.df,
                "stages": [asdict(s) for s in self.stages]}

    @staticmethod
    def from_dict(d: dict) -> "Plan":
        d = dict(d or {})
        stages = [Stage(**s) for s in d.get("stages") or [{}]]
        return Plan(d.get("weight"), stages, d.get("srs_estimator", "WOR"), d.get("df"))

    def describe(self) -> str:
        out = [f"Weight: {self.weight or '(none: every case weight 1)'}"]
        for k, s in enumerate(self.stages, 1):
            out.append(f"Stage {k}: {METHODS.get(s.method, s.method)}; strata: "
                       f"{', '.join(s.strata) or '(none)'}; clusters: "
                       f"{', '.join(s.clusters) or '(none)'}" +
                       (f"; {s.fpc_kind}: {s.fpc_var or s.fpc_value}" if s.fpc_kind != "none"
                        else ""))
        return "\n".join(out)


def get_plan(ds: Dataset) -> Plan | None:
    d = ds.meta.get(PLAN_KEY)
    return Plan.from_dict(d) if d else None


def set_plan(ds: Dataset, plan: Plan) -> None:
    ds.meta[PLAN_KEY] = plan.to_dict()


def _codes(ds: Dataset, names: list[str], rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Integer group code per case for the combination of ``names`` and a validity mask."""
    if not names:
        return np.zeros(rows.sum(), int), np.ones(rows.sum(), bool)
    frame = pd.DataFrame({n: ds.analysis_series(n)[rows].to_numpy() for n in names})
    valid = ~frame.isna().any(axis=1).to_numpy()
    keys = frame.astype(str).agg("\x1f".join, axis=1).to_numpy()
    _, codes = np.unique(keys, return_inverse=True)
    return codes, valid


# ===================================================================== design
class Design:
    """Design arrays for the cases with valid design information."""

    def __init__(self, ds: Dataset, plan: Plan):
        self.plan = plan
        n0 = ds.n_cases
        rows = np.ones(n0, bool)
        if plan.weight:
            w = pd.to_numeric(ds.analysis_series(plan.weight), errors="coerce").to_numpy(float)
            rows &= ~np.isnan(w) & (w > 0)
        else:
            w = np.ones(n0)
        st1 = plan.stages[0]
        names = list(st1.strata) + list(st1.clusters)
        for s in plan.stages[1:]:
            names += list(s.strata) + list(s.clusters)
        for nm in names:
            rows &= ds.analysis_series(nm).notna().to_numpy()
        self.rows = rows                       # mask over the dataset's cases
        self.n_total = n0
        self.n = int(rows.sum())
        self.w = w[rows]
        self.N = float(self.w.sum())
        self.strata, _ = _codes(ds, st1.strata, rows)
        cl, _ = _codes(ds, list(st1.strata) + list(st1.clusters), rows) if st1.clusters else \
            (np.arange(self.n), None)
        self.psu = cl
        self.method = st1.method
        # stage-1 sampling fraction / inclusion probability per PSU
        self.f = np.zeros(self.n)
        self.pi = np.ones(self.n)
        if st1.method in ("EQUAL_WOR", "UNEQUAL_WOR"):
            self.f, self.pi = self._fraction(ds, st1, rows, self.strata, self.psu)
        self.stage2 = None
        if len(plan.stages) > 1 and st1.method != "WR":
            s2 = plan.stages[1]
            st, _ = _codes(ds, list(st1.strata) + list(st1.clusters) + list(s2.strata), rows)
            if s2.clusters:
                ssu, _ = _codes(ds, list(st1.strata) + list(st1.clusters) + list(s2.strata) +
                                list(s2.clusters), rows)
            else:
                ssu = np.arange(self.n)
            f2 = np.zeros(self.n)
            if s2.method == "EQUAL_WOR":
                f2, _ = self._fraction(ds, s2, rows, st, ssu)
            self.stage2 = (st, ssu, s2.method, f2)
        self.n_strata = len(np.unique(self.strata))
        self.n_psu = len(np.unique(self.psu))
        self.df = plan.df if plan.df else max(self.n_psu - self.n_strata, 1)
        if plan.srs_estimator == "WOR" and self.n < self.N:
            self.fpc_srs = 1 - self.n / self.N
        else:
            self.fpc_srs = 1.0

    @staticmethod
    def _fraction(ds, st: Stage, rows, strata, units):
        n = rows.sum()
        f = np.zeros(n)
        pi = np.ones(n)
        if st.fpc_kind == "none":
            return f, pi
        if st.fpc_var:
            val = pd.to_numeric(ds.analysis_series(st.fpc_var), errors="coerce"
                                ).to_numpy(float)[rows]
        else:
            val = np.full(n, float(st.fpc_value or 0))
        if st.fpc_kind in ("rate", "inclusion"):
            f = np.nan_to_num(val)
            pi = np.where(f > 0, f, 1.0)
            if st.method == "EQUAL_WOR":
                f = np.array([np.nanmean(f[strata == h]) for h in strata])
        elif st.fpc_kind == "popsize":
            for h in np.unique(strata):
                m = strata == h
                nh = len(np.unique(units[m]))
                Nh = np.nanmean(val[m])
                f[m] = nh / Nh if Nh and Nh > 0 else 0.0
            pi = np.where(f > 0, f, 1.0)
        return np.clip(f, 0, 1), pi

    # --------------------------------------------------------------- covariance of totals
    def vcov(self, U: np.ndarray) -> np.ndarray:
        """Design covariance of the estimated total of the weighted scores U (n x k)."""
        U = np.asarray(U, float)
        if U.ndim == 1:
            U = U[:, None]
        k = U.shape[1]
        V = np.zeros((k, k))
        T, psu_stratum, psu_f, psu_pi = self._psu_totals(U)
        for h in np.unique(psu_stratum):
            m = psu_stratum == h
            nh = m.sum()
            if nh < 2:
                continue
            Th = T[m]
            if self.method == "UNEQUAL_WOR":
                p = psu_pi[m]
                c = (nh - 1) / np.maximum(nh - p, 1e-12)
                for i in range(nh):
                    for j in range(i + 1, nh):
                        pij = p[i] * p[j] * (c[i] + c[j]) / 2
                        dif = Th[i] - Th[j]
                        V += (p[i] * p[j] - pij) / pij * np.outer(dif, dif)
                continue
            dev = Th - Th.mean(axis=0)
            fac = nh / (nh - 1)
            if self.method == "EQUAL_WOR":
                fac *= 1 - psu_f[m][0]
            V += fac * dev.T @ dev
        if self.stage2 is not None:
            st, ssu, meth2, f2 = self.stage2
            for g in np.unique(self.psu):
                inpsu = self.psu == g
                pig = self.pi[inpsu][0] if self.method == "UNEQUAL_WOR" else self.f[inpsu][0]
                for h2 in np.unique(st[inpsu]):
                    m = inpsu & (st == h2)
                    units = ssu[m]
                    uu = np.unique(units)
                    if len(uu) < 2:
                        continue
                    Tk = np.array([U[m][units == u].sum(axis=0) for u in uu])
                    dev = Tk - Tk.mean(axis=0)
                    fac = len(uu) / (len(uu) - 1)
                    if meth2 == "EQUAL_WOR":
                        fac *= 1 - f2[m][0]
                    V += pig * fac * dev.T @ dev
        return V

    def _psu_totals(self, U):
        uniq, inv = np.unique(self.psu, return_inverse=True)
        T = np.zeros((len(uniq), U.shape[1]))
        np.add.at(T, inv, U)
        first = np.zeros(len(uniq), int)
        first[inv[::-1]] = np.arange(len(inv))[::-1]
        return T, self.strata[first], self.f[first], self.pi[first]

    def vsrs(self, Z: np.ndarray, center: bool = True) -> np.ndarray:
        """SRS covariance for design effects: fpc * N/(n-1) * sum w (z - Z/N)(z - Z/N)'."""
        Z = np.asarray(Z, float)
        if Z.ndim == 1:
            Z = Z[:, None]
        zbar = (self.w @ Z) / self.N if center else np.zeros(Z.shape[1])
        D = Z - zbar
        return self.fpc_srs * self.N / max(self.n - 1, 1) * (D * self.w[:, None]).T @ D

    def info_table(self, valid: int | None = None) -> Table:
        rows = [["N of Cases", "Valid", fmt_int(self.n if valid is None else valid)],
                ["", "Invalid", fmt_int(self.n_total - (self.n if valid is None else valid))],
                ["", "Total", fmt_int(self.n_total)],
                ["Population Size", "", fmt(self.N, 3)],
                ["Stage 1", "Strata", fmt_int(self.n_strata)],
                ["", "Units", fmt_int(self.n_psu)],
                ["Sampling Design Degrees of Freedom", "", fmt_int(self.df)]]
        return Table("Sample Design Information", ["", "", "N"], rows, row_header_cols=2)


def design_for(ds: Dataset, plan: Plan | dict | None) -> Design:
    if plan is None:
        plan = get_plan(ds)
        if plan is None:
            raise ValueError("No Complex Samples plan: run Analyze > Complex Samples > "
                             "Prepare for Analysis... first.")
    if isinstance(plan, dict):
        plan = Plan.from_dict(plan)
    return Design(ds, plan)


# ===================================================================== estimates
@dataclass
class Est:
    value: float
    se: float
    lo: float
    hi: float
    deff: float
    n: int

    @property
    def cv(self) -> float:
        return self.se / self.value if self.value else float("nan")


def _t(dsg: Design, level: float) -> float:
    return float(stats.t.ppf(0.5 + level / 2, dsg.df))


def total_est(dsg: Design, y: np.ndarray, dom: np.ndarray, level=0.95) -> Est:
    yy = np.where(dom, y, 0.0)
    val = float(dsg.w @ yy)
    U = dsg.w * yy
    v = dsg.vcov(U)[0, 0]
    vs = dsg.vsrs(yy)[0, 0]
    se = math.sqrt(max(v, 0))
    tq = _t(dsg, level)
    return Est(val, se, val - tq * se, val + tq * se, v / vs if vs > 0 else float("nan"),
               int(dom.sum()))


def ratio_est(dsg: Design, y: np.ndarray, x: np.ndarray, dom: np.ndarray, level=0.95,
              logit_ci: bool = False) -> Est:
    yy = np.where(dom, y, 0.0)
    xx = np.where(dom, x, 0.0)
    X = float(dsg.w @ xx)
    if X == 0:
        return Est(float("nan"), float("nan"), float("nan"), float("nan"), float("nan"),
                   int(dom.sum()))
    R = float(dsg.w @ yy) / X
    z = (yy - R * xx) / X
    v = dsg.vcov(dsg.w * z)[0, 0]
    vs = dsg.vsrs(z)[0, 0]
    se = math.sqrt(max(v, 0))
    tq = _t(dsg, level)
    if logit_ci and 0 < R < 1:
        lg = math.log(R / (1 - R))
        h = tq * se / (R * (1 - R))
        lo, hi = 1 / (1 + math.exp(-(lg - h))), 1 / (1 + math.exp(-(lg + h)))
    else:
        lo, hi = R - tq * se, R + tq * se
    return Est(R, se, lo, hi, v / vs if vs > 0 else float("nan"), int(dom.sum()))


def _num(ds: Dataset, name: str, rows: np.ndarray) -> np.ndarray:
    return pd.to_numeric(ds.analysis_series(name), errors="coerce").to_numpy(float)[rows]


def _cats(ds: Dataset, name: str, rows: np.ndarray):
    s = ds.analysis_series(name)[rows]
    vals = sorted(pd.unique(s.dropna()), key=sort_key)
    var = ds.var(name)
    return s.to_numpy(object), vals, [str(value_text(var, v)) for v in vals]


def _domains(ds: Dataset, subpops: list[str], rows: np.ndarray):
    """[(label, mask)] for the whole population and each subpopulation category."""
    n = rows.sum()
    out = [("Total", np.ones(n, bool))]
    for sp in subpops or []:
        arr, vals, labs = _cats(ds, sp, rows)
        for v, lab in zip(vals, labs):
            out.append((f"{ds.var(sp).title} = {lab}", np.array([a == v for a in arr])))
    return out


STAT_COLS = {"se": "Standard Error", "ci": "Confidence Interval", "cv": "Coefficient of "
             "Variation", "count": "Unweighted Count", "deff": "Design Effect",
             "sqrt_deff": "Square Root of Design Effect"}


def _est_cells(e: Est, statistics: tuple, d: int = 3) -> list[str]:
    out = [fmt(e.value, d)]
    if "se" in statistics:
        out.append(fmt(e.se, d))
    if "ci" in statistics:
        out += [fmt(e.lo, d), fmt(e.hi, d)]
    if "cv" in statistics:
        out.append(fmt(e.cv, 3))
    if "count" in statistics:
        out.append(fmt_int(e.n))
    if "deff" in statistics:
        out.append(fmt(e.deff, 3))
    if "sqrt_deff" in statistics:
        out.append(fmt(math.sqrt(e.deff), 3) if e.deff >= 0 else "")
    return out


def _est_cols(statistics: tuple, level: float) -> list:
    cols = ["Estimate"]
    if "se" in statistics:
        cols.append("Standard Error")
    if "ci" in statistics:
        cols += [(f"{level * 100:g}% Confidence Interval", "Lower"),
                 (f"{level * 100:g}% Confidence Interval", "Upper")]
    for k in ("cv", "count", "deff", "sqrt_deff"):
        if k in statistics:
            cols.append(STAT_COLS[k])
    return cols


def _plan_log(dsg: Design, cmd: str) -> str:
    return cmd + "\n  /PLAN = active dataset plan\n" + \
        "\n".join("  " + line for line in dsg.plan.describe().splitlines())


# ===================================================================== CS Descriptives
def cs_descriptives(ds: Dataset, variables: list[str], plan=None, means: bool = True,
                    sums: bool = False, subpops: list[str] | None = None,
                    statistics: tuple = ("se",), level: float = 0.95,
                    listwise: bool = False, test_value: float | None = None) -> Section:
    dsg = design_for(ds, plan)
    sec = Section("Complex Samples Descriptives")
    sec.log(_plan_log(dsg, "CSDESCRIPTIVES /SUMMARY VARIABLES=" + " ".join(variables) +
                      f"\n  /{'MEAN' if means else ''} {'SUM' if sums else ''}"))
    rows_ = dsg.rows
    Y = {v: _num(ds, v, rows_) for v in variables}
    common = np.ones(dsg.n, bool)
    if listwise:
        for v in variables:
            common &= ~np.isnan(Y[v])
    sec.table(dsg.info_table())
    doms = _domains(ds, subpops or [], rows_)
    for kind, flag in (("Mean", means), ("Sum", sums)):
        if not flag:
            continue
        rows = []
        for dlab, dom in doms:
            for v in variables:
                ok = dom & common & ~np.isnan(Y[v])
                y = np.nan_to_num(Y[v])
                e = ratio_est(dsg, y, np.ones(dsg.n), ok, level) if kind == "Mean" else \
                    total_est(dsg, y, ok, level)
                lab = ds.var(v).title
                rows.append(([dlab] if subpops else []) + [lab] + _est_cells(e, statistics))
                if test_value is not None and kind == "Mean":
                    t = (e.value - test_value) / e.se if e.se > 0 else float("nan")
                    rows[-1] += [fmt(t, 3), fmt_int(dsg.df),
                                 fmt_p(2 * stats.t.sf(abs(t), dsg.df)) if np.isfinite(t) else ""]
        cols = (["Subpopulation"] if subpops else []) + [""] + _est_cols(statistics, level)
        if test_value is not None and kind == "Mean":
            cols += [f"t (Hypothesis Test, mean = {test_value:g})", "df", "Sig."]
        sec.table(Table(f"Univariate Statistics: {kind}", cols, rows,
                        row_header_cols=2 if subpops else 1))
    return sec


# ===================================================================== CS Ratios
def cs_ratios(ds: Dataset, numerators: list[str], denominators: list[str], plan=None,
              subpops: list[str] | None = None, statistics: tuple = ("se",),
              level: float = 0.95, population_size: bool = False,
              test_value: float | None = None) -> Section:
    dsg = design_for(ds, plan)
    sec = Section("Complex Samples Ratios")
    sec.log(_plan_log(dsg, "CSDESCRIPTIVES /RATIO NUMERATOR=" + " ".join(numerators) +
                      " DENOMINATOR=" + " ".join(denominators)))
    sec.table(dsg.info_table())
    rows = []
    for dlab, dom in _domains(ds, subpops or [], dsg.rows):
        for nu in numerators:
            for de in denominators:
                y, x = _num(ds, nu, dsg.rows), _num(ds, de, dsg.rows)
                ok = dom & ~np.isnan(y) & ~np.isnan(x)
                e = ratio_est(dsg, np.nan_to_num(y), np.nan_to_num(x), ok, level)
                row = ([dlab] if subpops else []) + [ds.var(nu).title, ds.var(de).title] + \
                    _est_cells(e, statistics, 4)
                if population_size:
                    row.append(fmt(float(dsg.w[ok].sum()), 3))
                if test_value is not None:
                    t = (e.value - test_value) / e.se if e.se > 0 else float("nan")
                    row += [fmt(t, 3), fmt_int(dsg.df),
                            fmt_p(2 * stats.t.sf(abs(t), dsg.df)) if np.isfinite(t) else ""]
                rows.append(row)
    cols = (["Subpopulation"] if subpops else []) + ["Numerator", "Denominator"] + \
        [("Ratio", c) if isinstance(c, str) else c for c in _est_cols(statistics, level)]
    cols = [c if not (isinstance(c, tuple) and c[0] == "Ratio") else
            ("Ratio Estimate" if c[1] == "Estimate" else c[1]) for c in cols]
    if population_size:
        cols.append("Population Size")
    if test_value is not None:
        cols += [f"t (Hypothesis Test, ratio = {test_value:g})", "df", "Sig."]
    sec.table(Table("Ratios", cols, rows, row_header_cols=(3 if subpops else 2)))
    return sec


# ===================================================================== CS Frequencies
def cs_frequencies(ds: Dataset, variables: list[str], plan=None,
                   subpops: list[str] | None = None, population_size: bool = True,
                   table_percent: bool = True, statistics: tuple = ("se",),
                   level: float = 0.95, cumulative: bool = False) -> Section:
    dsg = design_for(ds, plan)
    sec = Section("Complex Samples Frequencies")
    sec.log(_plan_log(dsg, "CSTABULATE /TABLES VARIABLES=" + " ".join(variables) +
                      "\n  /CELLS=" + ("POPSIZE " if population_size else "") +
                      ("TABLEPCT" if table_percent else "")))
    sec.table(dsg.info_table())
    for v in variables:
        arr, vals, labs = _cats(ds, v, dsg.rows)
        valid = np.array([a is not None and not (isinstance(a, float) and math.isnan(a))
                          for a in arr])
        rows = []
        for dlab, dom in _domains(ds, subpops or [], dsg.rows):
            base = dom & valid
            pref = [dlab] if subpops else []
            if population_size:
                for val, lab in zip(vals, labs):
                    ind = np.array([a == val for a in arr], float)
                    e = total_est(dsg, ind, base, level)
                    e.n = int((base & (ind > 0)).sum())
                    rows.append(pref + ["Population Size", lab] + _est_cells(e, statistics))
                e = total_est(dsg, np.ones(dsg.n), base, level)
                rows.append(pref + ["Population Size", "Total"] + _est_cells(e, statistics))
            if table_percent:
                cum = 0.0
                for val, lab in zip(vals, labs):
                    ind = np.array([a == val for a in arr], float)
                    e = ratio_est(dsg, ind, np.ones(dsg.n), base, level, logit_ci=True)
                    e.n = int((base & (ind > 0)).sum())
                    pe = Est(e.value * 100, e.se * 100, e.lo * 100, e.hi * 100, e.deff, e.n)
                    cum += pe.value
                    row = pref + ["% of Total", lab] + _est_cells(pe, statistics, 1)
                    if cumulative:
                        row.append(fmt(cum, 1))
                    rows.append(row)
                tot = Est(100.0, 0.0, 100.0, 100.0, float("nan"), int(base.sum()))
                row = pref + ["% of Total", "Total"] + _est_cells(tot, statistics, 1)
                if cumulative:
                    row.append("")
                rows.append(row)
        cols = (["Subpopulation"] if subpops else []) + ["", ds.var(v).title] + \
            _est_cols(statistics, level) + (["Cumulative %"] if cumulative and table_percent
                                            else [])
        if cumulative and not table_percent:
            pass
        rows = [r + [""] * (len(cols) - len(r)) for r in rows]
        sec.table(Table(ds.var(v).title, cols, rows, row_header_cols=3 if subpops else 2))
    return sec


# ===================================================================== CS Crosstabs
def cs_crosstabs(ds: Dataset, row: str, col: str, plan=None, subpops: list[str] | None = None,
                 cells: tuple = ("popsize", "row"), statistics: tuple = ("se",),
                 level: float = 0.95, tests: bool = True, odds_ratio: bool = False,
                 residuals: bool = False) -> Section:
    """Two-way table with design-based cell estimates and the Rao-Scott adjusted tests of
    independence (IBM CSTABULATE)."""
    dsg = design_for(ds, plan)
    sec = Section("Complex Samples Crosstabs")
    sec.log(_plan_log(dsg, f"CSTABULATE /TABLES VARIABLES={row} BY {col}\n  /CELLS=" +
                      " ".join(c.upper() for c in cells) + ("\n  /TEST=INDEPENDENCE"
                                                            if tests else "")))
    sec.table(dsg.info_table())
    ra, rvals, rlabs = _cats(ds, row, dsg.rows)
    ca, cvals, clabs = _cats(ds, col, dsg.rows)
    rvalid = pd.notna(pd.Series(ra)).to_numpy()
    cvalid = pd.notna(pd.Series(ca)).to_numpy()
    RI = np.column_stack([np.array([a == v for a in ra], float) for v in rvals])
    CI = np.column_stack([np.array([a == v for a in ca], float) for v in cvals])
    stat_names = {"popsize": "Population Size", "row": "% within " + ds.var(row).title,
                  "column": "% within " + ds.var(col).title, "table": "% of Total"}
    for dlab, dom in _domains(ds, subpops or [], dsg.rows):
        base = dom & rvalid & cvalid
        R, C = len(rvals), len(cvals)
        out_rows = []
        for i in range(R + 1):
            for cell in cells:
                for stat in ("est",) + tuple(s for s in statistics if s in ("se", "ci", "cv",
                                                                           "deff", "count")):
                    lab_r = rlabs[i] if i < R else "Total"
                    row_cells = []
                    for j in range(C + 1):
                        ri = RI[:, i] if i < R else np.ones(dsg.n)
                        cj = CI[:, j] if j < C else np.ones(dsg.n)
                        ind = ri * cj
                        if cell == "popsize":
                            e = total_est(dsg, ind, base, level)
                            scale = 1.0
                        else:
                            den = {"row": ri, "column": cj, "table": np.ones(dsg.n)}[cell]
                            e = ratio_est(dsg, ind, den, base, level, logit_ci=True)
                            scale = 100.0
                        e.n = int((base & (ind > 0)).sum())
                        val = {"est": e.value * scale, "se": e.se * scale, "cv": e.cv,
                               "deff": e.deff, "count": e.n}.get(stat)
                        if stat == "ci":
                            row_cells.append(f"{fmt(e.lo * scale, 1)}; {fmt(e.hi * scale, 1)}")
                        elif stat == "count":
                            row_cells.append(fmt_int(val))
                        else:
                            d = 1 if (cell != "popsize" and stat in ("est", "se")) else 3
                            row_cells.append(fmt(val, d))
                    sname = stat_names[cell] if stat == "est" else \
                        f"{stat_names[cell]}: {STAT_COLS.get(stat, stat)}"
                    out_rows.append(([dlab] if subpops else []) + [lab_r, sname] + row_cells)
        cols = (["Subpopulation"] if subpops else []) + [ds.var(row).title, ""] + \
            [(ds.var(col).title, c) for c in clabs] + [(ds.var(col).title, "Total")]
        sec.table(Table(f"{ds.var(row).title} * {ds.var(col).title}", cols, out_rows,
                        row_header_cols=3 if subpops else 2))
        if tests and R > 1 and C > 1:
            res = independence_tests(dsg, RI, CI, base)
            trows = []
            for name, (chi, F, d1, d2, p) in res.items():
                trows.append(([dlab] if subpops else []) +
                             [name, fmt(chi, 3), fmt(F, 3), fmt(d1, 3), fmt(d2, 3), fmt_p(p)])
            sec.table(Table("Tests of Independence", (["Subpopulation"] if subpops else []) +
                            ["", "Chi-Square", "Adjusted F", "df1", "df2", "Sig."], trows,
                            row_header_cols=2 if subpops else 1,
                            footnotes=["Rao-Scott second-order adjusted F (IBM CSTABULATE)."]))
        if odds_ratio and R == 2 and C == 2:
            orr = odds_ratios(dsg, RI, CI, base, level)
            sec.table(Table("Risk Estimate", (["Subpopulation"] if subpops else []) +
                            ["", "Estimate", "Standard Error",
                             (f"{level * 100:g}% Confidence Interval", "Lower"),
                             (f"{level * 100:g}% Confidence Interval", "Upper")],
                            [([dlab] if subpops else []) + [k] + [fmt(x, 3) for x in v]
                             for k, v in orr.items()]))
        if residuals and R > 1 and C > 1:
            P, V, n_ = _table_cov(dsg, RI, CI, base)
            Pm = P.reshape(R, C)
            E_ = np.outer(Pm.sum(1), Pm.sum(0)).ravel()
            M = np.diag(P) - np.outer(P, P)
            AR, AC = _contr(R), _contr(C)
            Xc = np.hstack([np.kron(AR, np.ones((C, 1))), np.kron(np.ones((R, 1)), AC)])
            core = Xc @ np.linalg.pinv(Xc.T @ M @ Xc) @ Xc.T
            I = np.eye(R * C)
            VR = (I - M @ core) @ V @ (I - core @ M)
            sres = (P - E_) / np.sqrt(np.maximum(np.diag(VR), 1e-300))
            rr = [([dlab] if subpops else []) + [rlabs[i]] +
                  [fmt(sres[i * C + j], 3) for j in range(C)] for i in range(R)]
            sec.table(Table("Standardized Residuals", (["Subpopulation"] if subpops else []) +
                            [ds.var(row).title] + [(ds.var(col).title, c) for c in clabs], rr))
    return sec


def _contr(k):
    return np.vstack([np.eye(k - 1), -np.ones((1, k - 1))])


def _table_cov(dsg: Design, RI, CI, base):
    R, C = RI.shape[1], CI.shape[1]
    N = float(dsg.w @ base)
    cellind = np.column_stack([RI[:, i] * CI[:, j] for i in range(R) for j in range(C)])
    cellind = cellind * base[:, None]
    P = (dsg.w @ cellind) / N
    Z = (cellind - P[None, :] * base[:, None]) / N
    V = dsg.vcov(Z * dsg.w[:, None])
    return P, V, int(base.sum())


def independence_tests(dsg: Design, RI, CI, base) -> dict:
    R, C = RI.shape[1], CI.shape[1]
    P, V, n = _table_cov(dsg, RI, CI, base)
    Pm = P.reshape(R, C)
    pr, pc = Pm.sum(1), Pm.sum(0)
    E_ = np.outer(pr, pc)
    with np.errstate(divide="ignore", invalid="ignore"):
        X2 = n * np.nansum(np.where(E_ > 0, (Pm - E_) ** 2 / E_, 0))
        G2 = 2 * n * np.nansum(np.where(Pm > 0, Pm * np.log(Pm / E_), 0))
    Cm = np.kron(_contr(R), _contr(C))
    Dinv = np.diag(np.where(P > 0, 1 / np.where(P > 0, P, 1), 0))
    M = np.diag(P) - np.outer(P, P)
    A = Cm.T @ Dinv @ M @ Dinv @ Cm
    B = Cm.T @ Dinv @ V @ Dinv @ Cm
    Delta = n * np.linalg.pinv(A) @ B
    tr = float(np.trace(Delta))
    tr2 = float(np.trace(Delta @ Delta))
    d = tr ** 2 / tr2 if tr2 > 0 else float("nan")
    out = {}
    for name, chi in (("Pearson", X2), ("Likelihood Ratio", G2)):
        F = chi / tr if tr > 0 else float("nan")
        out[name] = (chi, F, d, d * dsg.df, float(stats.f.sf(F, d, d * dsg.df)))
    return out


def odds_ratios(dsg: Design, RI, CI, base, level) -> dict:
    w = dsg.w
    cells = [[RI[:, i] * CI[:, j] * base for j in range(2)] for i in range(2)]
    Nc = [[float(w @ cells[i][j]) for j in range(2)] for i in range(2)]
    tq = _t(dsg, level)
    out = {}

    def from_log(z, val):
        v = dsg.vcov(w * z)[0, 0]
        se_log = math.sqrt(max(v, 0))
        return (val, val * se_log, val * math.exp(-tq * se_log), val * math.exp(tq * se_log))

    if min(Nc[0][0], Nc[0][1], Nc[1][0], Nc[1][1]) > 0:
        orv = Nc[0][0] * Nc[1][1] / (Nc[0][1] * Nc[1][0])
        z = cells[0][0] / Nc[0][0] + cells[1][1] / Nc[1][1] - cells[0][1] / Nc[0][1] - \
            cells[1][0] / Nc[1][0]
        out["Odds Ratio"] = from_log(z, orv)
        for j, lab in ((0, "Relative Risk (column 1)"), (1, "Relative Risk (column 2)")):
            r1 = Nc[0][0 + j] / (Nc[0][0] + Nc[0][1])
            r2 = Nc[1][0 + j] / (Nc[1][0] + Nc[1][1])
            rr = r1 / r2
            z = cells[0][j] / Nc[0][j] - (cells[0][0] + cells[0][1]) / (Nc[0][0] + Nc[0][1]) - \
                cells[1][j] / Nc[1][j] + (cells[1][0] + cells[1][1]) / (Nc[1][0] + Nc[1][1])
            out[lab] = from_log(z, rr)
    return out


# ===================================================================== models: common
def _model_design(ds: Dataset, rows: np.ndarray, dependent_ok: np.ndarray, factors: list[str],
                  covariates: list[str], terms_text: str | None, intercept: bool = True):
    """Indicator (parameters) and effect-coded (Type III tests) design matrices from the GLM
    engine for the cases in ``rows`` & ``dependent_ok``."""
    from rsstatchem.core.stats import glm_model as gm

    ok = dependent_ok.copy()
    fdata, cdata = {}, {}
    for f in factors:
        s = ds.analysis_series(f)[rows].to_numpy(object)
        ok &= pd.notna(pd.Series(s)).to_numpy()
        fdata[f] = s
    for c in covariates:
        x = _num(ds, c, rows)
        ok &= ~np.isnan(x)
        cdata[c] = x
    finfo = []
    for f in factors:
        s = fdata[f][ok]
        levels = sorted(pd.unique(pd.Series(s)), key=sort_key)
        codes = np.array([levels.index(v) for v in s], int)
        var = ds.var(f)
        finfo.append(gm.FactorInfo(f, levels, [str(value_text(var, v)) for v in levels],
                                   codes))
    covs = {c: cdata[c][ok] for c in covariates}
    if terms_text:
        terms = gm.parse_terms(terms_text, factors, covariates)
    else:
        terms = gm.main_effects(factors, covariates)
    des = gm.Design(finfo, covs, terms, intercept=intercept, n=int(ok.sum()))
    Xi, labels, owner = des.indicator_matrix()
    Xe, eidx = des.effect_matrix()
    alias = gm.aliased_columns(Xi)
    return ok, Xi, labels, owner, alias, Xe, eidx, terms


def _sandwich(dsg: Design, Hinv: np.ndarray, S: np.ndarray, okmask: np.ndarray):
    """Binder covariance and its SRS counterpart from element scores S (unweighted,
    rows = analysis cases ``okmask`` within the design rows)."""
    full = np.zeros((dsg.n, S.shape[1]))
    full[okmask] = S
    V = Hinv @ dsg.vcov(full * dsg.w[:, None]) @ Hinv
    Vs = Hinv @ dsg.vsrs(full, center=False) @ Hinv
    return V, Vs


def wald_tests(L: np.ndarray, b: np.ndarray, V: np.ndarray, Vs: np.ndarray, nu: float,
               kind: str = "F") -> tuple:
    """(statistic, df1, df2, p) for H0: L b = 0 (IBM Complex Samples: Model Testing)."""
    Lb = L @ b
    LVL = L @ V @ L.T
    r = np.linalg.matrix_rank(LVL)
    if r == 0:
        return float("nan"), 0, float("nan"), float("nan")
    X2 = float(Lb @ np.linalg.pinv(LVL) @ Lb)
    if kind == "CHISQUARE":
        return X2, r, float("nan"), float(stats.chi2.sf(X2, r))
    if kind == "F":
        if nu < r:
            return float("nan"), r, float("nan"), float("nan")
        F = (nu - r + 1) / (r * nu) * X2
        return F, r, nu - r + 1, float(stats.f.sf(F, r, nu - r + 1))
    LVsL = L @ Vs @ L.T
    X2s = float(Lb @ np.linalg.pinv(LVsL) @ Lb)
    Delta = np.linalg.pinv(LVsL) @ LVL
    tr, tr2 = float(np.trace(Delta)), float(np.trace(Delta @ Delta))
    d = tr ** 2 / tr2 if tr2 > 0 else float("nan")
    if kind == "ADJCHISQUARE":
        stat = X2s / (tr / d)
        return stat, d, float("nan"), float(stats.chi2.sf(stat, d))
    stat = X2s / tr
    return stat, d, d * nu, float(stats.f.sf(stat, d, d * nu))


def _effects_table(terms, eidx, be, Ve, Vse, nu, test, has_intercept=True, title="Tests of "
                   "Model Effects", first_rows=()) -> Table:
    rows = list(first_rows)
    k = len(be)
    cand = []
    if has_intercept and eidx and eidx[0]:
        cand.append(("(Intercept)", eidx[0]))
    for t, idx in zip(terms, eidx[1:]):
        cand.append((t.label, idx))
    for name, idx in cand:
        if not idx:
            continue
        L = np.zeros((len(idx), k))
        for r_, c in enumerate(idx):
            L[r_, c] = 1
        st, d1, d2, p = wald_tests(L, be, Ve, Vse, nu, test)
        rows.append([name, fmt(st, 3), fmt(d1, 3) if not float(d1).is_integer() else fmt_int(d1),
                     fmt(d2, 3) if np.isfinite(d2) and not float(d2).is_integer() else
                     (fmt_int(d2) if np.isfinite(d2) else ""), fmt_p(p)])
    head = {"F": "Wald F", "ADJF": "Adjusted Wald F", "CHISQUARE": "Wald Chi-Square",
            "ADJCHISQUARE": "Adjusted Wald Chi-Square"}[test]
    return Table(title, ["Source", head, "df1", "df2", "Sig."], rows)


def _param_table(labels, b, V, Vs, alias, nu, level, exp_b=False, title="Parameter "
                 "Estimates", deff=True, prefix_cols=None) -> Table:
    tq = float(stats.t.ppf(0.5 + level / 2, nu))
    rows = []
    for j, lab in enumerate(labels):
        if alias[j]:
            row = [lab, "0ᵃ", "", "", "", "", "", ""] + (["", ""] if deff else []) + \
                (["", "", ""] if exp_b else [])
            rows.append(row)
            continue
        se = math.sqrt(max(V[j, j], 0))
        t = b[j] / se if se > 0 else float("nan")
        row = [lab, fmt(b[j], 3), fmt(se, 3), fmt(b[j] - tq * se, 3), fmt(b[j] + tq * se, 3),
               fmt(t, 3), fmt_int(nu), fmt_p(2 * stats.t.sf(abs(t), nu)) if np.isfinite(t)
               else ""]
        if deff:
            dv = V[j, j] / Vs[j, j] if Vs[j, j] > 0 else float("nan")
            row += [fmt(dv, 3), fmt(math.sqrt(dv), 3) if dv >= 0 else ""]
        if exp_b:
            with np.errstate(over="ignore"):
                row += [fmt(np.exp(b[j]), 3), fmt(np.exp(b[j] - tq * se), 3),
                        fmt(np.exp(b[j] + tq * se), 3)]
        rows.append(row)
    cols = ["Parameter", "Estimate", "Std. Error",
            (f"{level * 100:g}% Confidence Interval", "Lower"),
            (f"{level * 100:g}% Confidence Interval", "Upper"),
            ("Hypothesis Test", "t"), ("Hypothesis Test", "df"), ("Hypothesis Test", "Sig.")]
    if deff:
        cols += ["Design Effect", "Sqrt Design Effect"]
    if exp_b:
        cols += ["Exp(B)", (f"{level * 100:g}% CI for Exp(B)", "Lower"),
                 (f"{level * 100:g}% CI for Exp(B)", "Upper")]
    foot = ["a. Set to zero because this parameter is redundant."] if alias.any() else []
    return Table(title, cols, rows, footnotes=foot)


def _domain_mask(ds, dsg, domain):
    if not domain:
        return np.ones(dsg.n, bool), ""
    var, val = domain
    s = ds.analysis_series(var)[dsg.rows].to_numpy(object)
    return np.array([a == val for a in s]), f" (subpopulation {var} = {val})"


# ===================================================================== CSGLM
def cs_glm(ds: Dataset, dependent: str, factors: list[str] = (), covariates: list[str] = (),
           plan=None, terms: str | None = None, intercept: bool = True, level: float = 0.95,
           test: str = "F", deff: bool = True, domain: tuple | None = None,
           show_tests: bool = True) -> Result:
    dsg = design_for(ds, plan)
    factors, covariates = list(factors), list(covariates)
    sec = Section("Complex Samples General Linear Model")
    sec.log(_plan_log(dsg, f"CSGLM {dependent}" + (f" BY {' '.join(factors)}" if factors
                                                   else "") +
                      (f" WITH {' '.join(covariates)}" if covariates else "") +
                      f"\n  /MODEL {terms or ' '.join(factors + covariates)}" +
                      f"\n  /INTERCEPT INCLUDE={'YES' if intercept else 'NO'}"
                      f"\n  /STATISTICS PARAMETER SE CINTERVAL TTEST{' DEFF' if deff else ''}"
                      f"\n  /TEST TYPE={test}"))
    y_all = _num(ds, dependent, dsg.rows)
    dom, dtext = _domain_mask(ds, dsg, domain)
    ok, Xi, labels, owner, alias, Xe, eidx, tlist = _model_design(
        ds, dsg.rows, ~np.isnan(y_all) & dom, factors, covariates, terms, intercept)
    y = y_all[ok]
    w = dsg.w[ok]
    sec.table(dsg.info_table(int(ok.sum())))

    def fit(X):
        XtW = X.T * w
        H = XtW @ X
        Hinv = np.linalg.pinv(H)
        b = Hinv @ XtW @ y
        e = y - X @ b
        S = X * e[:, None]
        V, Vs = _sandwich(dsg, Hinv, S, ok)
        return b, e, V, Vs

    Xi_ = Xi * ~alias
    b, e, V, Vs = fit(Xi_)
    ybar = float(w @ y / w.sum())
    r2 = 1 - float(w @ e ** 2) / float(w @ (y - ybar) ** 2) if intercept else \
        1 - float(w @ e ** 2) / float(w @ y ** 2)
    sec.table(Table("Model Summary", ["R Square"], [[fmt(r2, 3)]], row_header_cols=0,
                    footnotes=[f"Model: {dependent} = " + " + ".join(
                        ["(Intercept)"] * intercept + [t.label for t in tlist]) + dtext]))
    if show_tests:
        be, _, Ve, Vse = fit(Xe)
        first = []
        if Xe.shape[1] > (1 if intercept else 0):
            idx = list(range(1 if intercept else 0, Xe.shape[1]))
            L = np.zeros((len(idx), Xe.shape[1]))
            for r_, c in enumerate(idx):
                L[r_, c] = 1
            st, d1, d2, p = wald_tests(L, be, Ve, Vse, dsg.df, test)
            first = [["Corrected Model", fmt(st, 3), fmt_int(d1) if float(d1).is_integer()
                      else fmt(d1, 3), fmt_int(d2) if np.isfinite(d2) and float(d2).is_integer()
                      else fmt(d2, 3), fmt_p(p)]]
        sec.table(_effects_table(tlist, eidx, be, Ve, Vse, dsg.df, test, intercept,
                                 first_rows=first))
    sec.table(_param_table(labels, b, V, Vs, alias, dsg.df, level, deff=deff))
    return Result(sec)


# ===================================================================== CSLOGISTIC
def _mlogit(X, Y, w, maxiter=100):
    """Generalized-logit ML (reference = column 0 of the indicator matrix Y)."""
    n, p = X.shape
    J = Y.shape[1]
    B = np.zeros((p, J - 1))
    for _ in range(maxiter):
        eta = np.column_stack([np.zeros(n), X @ B])
        eta -= eta.max(axis=1, keepdims=True)
        P = np.exp(eta)
        P /= P.sum(axis=1, keepdims=True)
        R = (Y - P)[:, 1:]
        g = (X * w[:, None]).T @ R
        H = np.zeros((p * (J - 1), p * (J - 1)))
        for a in range(J - 1):
            for c in range(J - 1):
                wt = w * P[:, a + 1] * ((a == c) - P[:, c + 1])
                H[a * p:(a + 1) * p, c * p:(c + 1) * p] = (X * wt[:, None]).T @ X
        step = np.linalg.pinv(H) @ g.T.ravel()
        B = B + step.reshape(J - 1, p).T
        if np.max(np.abs(step)) < 1e-10:
            break
    eta = np.column_stack([np.zeros(n), X @ B])
    eta -= eta.max(axis=1, keepdims=True)
    P = np.exp(eta)
    P /= P.sum(axis=1, keepdims=True)
    H = np.zeros((p * (J - 1), p * (J - 1)))
    for a in range(J - 1):
        for c in range(J - 1):
            wt = w * P[:, a + 1] * ((a == c) - P[:, c + 1])
            H[a * p:(a + 1) * p, c * p:(c + 1) * p] = (X * wt[:, None]).T @ X
    S = np.hstack([X * (Y - P)[:, a + 1][:, None] for a in range(J - 1)])
    ll = float(w @ np.log(np.maximum((Y * P).sum(axis=1), 1e-300)))
    return B, P, H, S, ll


def cs_logistic(ds: Dataset, dependent: str, factors: list[str] = (),
                covariates: list[str] = (), plan=None, reference: str = "last",
                terms: str | None = None, intercept: bool = True, level: float = 0.95,
                test: str = "F", deff: bool = True, odds: bool = True,
                classification: bool = True, domain: tuple | None = None) -> Result:
    """Binary or multinomial (generalized logit) logistic regression; by default the last
    (highest) category is the reference, as in SPSS."""
    dsg = design_for(ds, plan)
    factors, covariates = list(factors), list(covariates)
    sec = Section("Complex Samples Logistic Regression")
    sec.log(_plan_log(dsg, f"CSLOGISTIC {dependent}({reference.upper()})" +
                      (f" BY {' '.join(factors)}" if factors else "") +
                      (f" WITH {' '.join(covariates)}" if covariates else "") +
                      f"\n  /TEST TYPE={test}"))
    yarr, yvals, ylabs = _cats(ds, dependent, dsg.rows)
    yok = pd.notna(pd.Series(yarr)).to_numpy()
    dom, dtext = _domain_mask(ds, dsg, domain)
    ok, Xi, labels, owner, alias, Xe, eidx, tlist = _model_design(
        ds, dsg.rows, yok & dom, factors, covariates, terms, intercept)
    if len(yvals) < 2:
        sec.note("Warning", "The dependent variable needs at least two categories.", "warning")
        return Result(sec)
    ref = len(yvals) - 1 if reference == "last" else 0
    order = [ref] + [k for k in range(len(yvals)) if k != ref]
    Y = np.column_stack([np.array([a == yvals[k] for a in yarr[ok]], float) for k in order])
    w = dsg.w[ok]
    sec.table(dsg.info_table(int(ok.sum())))
    keepc = ~alias
    B, P, H, S, ll = _mlogit(Xi[:, keepc], Y, w)
    Hinv = np.linalg.pinv(H)
    V, Vs = _sandwich(dsg, Hinv, S, ok)
    p_ = int(keepc.sum())
    J = Y.shape[1]
    # null model log-likelihood
    pbar = (w @ Y) / w.sum()
    ll0 = float(w @ np.log(np.maximum(Y @ pbar, 1e-300)))
    Nw = w.sum()
    cs = 1 - math.exp(2 * (ll0 - ll) / Nw)
    nk = cs / (1 - math.exp(2 * ll0 / Nw))
    mf = 1 - ll / ll0 if ll0 != 0 else float("nan")
    sec.table(Table("Pseudo R Squares", ["", "Value"],
                    [["Cox and Snell", fmt(cs, 3)], ["Nagelkerke", fmt(nk, 3)],
                     ["McFadden", fmt(mf, 3)]],
                    footnotes=[f"Dependent Variable: {dependent} (reference category = "
                               f"{ylabs[ref]}){dtext}"]))
    # Type III tests with the effect-coded design
    Be, _, He, Se, _ = _mlogit(Xe, Y, w)
    Hinve = np.linalg.pinv(He)
    Ve, Vse = _sandwich(dsg, Hinve, Se, ok)
    pe = Xe.shape[1]
    eidx_all = [[c + a * pe for a in range(J - 1) for c in idx] for idx in eidx]
    sec.table(_effects_table(tlist, eidx_all, Be.T.ravel(), Ve, Vse, dsg.df, test, intercept))
    # parameter table: one block per non-reference category
    full_labels, full_b, full_alias = [], [], []
    Vfull = np.zeros((Xi.shape[1] * (J - 1),) * 2)
    Vsfull = np.zeros_like(Vfull)
    kept_idx = np.flatnonzero(keepc)
    pos = []
    for a in range(J - 1):
        cat = ylabs[order[a + 1]]
        for j, lab in enumerate(labels):
            full_labels.append(f"[{dependent}={cat}] {lab}" if J > 2 else lab)
            full_alias.append(alias[j])
            full_b.append(B[list(kept_idx).index(j), a] if keepc[j] else 0.0)
            pos.append((a, j))
    idx_map = {}
    for a in range(J - 1):
        for t, j in enumerate(kept_idx):
            idx_map[(a, j)] = a * p_ + t
    for u, (a1, j1) in enumerate(pos):
        for v_, (a2, j2) in enumerate(pos):
            if (a1, j1) in idx_map and (a2, j2) in idx_map:
                Vfull[u, v_] = V[idx_map[(a1, j1)], idx_map[(a2, j2)]]
                Vsfull[u, v_] = Vs[idx_map[(a1, j1)], idx_map[(a2, j2)]]
    sec.table(_param_table(full_labels, np.array(full_b), Vfull, Vsfull,
                           np.array(full_alias), dsg.df, level, exp_b=odds, deff=deff,
                           title="Parameter Estimates"))
    if classification:
        pred = np.argmax(P, axis=1)
        obs = np.argmax(Y, axis=1)
        cats = [ylabs[k] for k in order]
        ordr = np.argsort(order)
        rows = []
        for i in ordr:
            cnt = [float(w[(obs == i) & (pred == j)].sum()) for j in ordr]
            tot = sum(cnt)
            rows.append([cats[i]] + [fmt(c, 1) for c in cnt] +
                        [fmt(cnt[list(ordr).index(i)] / tot * 100 if tot else float("nan"), 1)])
        overall = float(w[obs == pred].sum()) / Nw * 100
        rows.append(["Overall Percent"] + [fmt(float(w[pred == j].sum()) / Nw * 100, 1)
                                           for j in ordr] + [fmt(overall, 1)])
        sec.table(Table("Classification", ["Observed"] + [("Predicted", cats[j]) for j in ordr]
                        + [("Predicted", "Percent Correct")], rows,
                        footnotes=[f"Dependent Variable: {dependent}; weighted counts."]))
    return Result(sec)


# ===================================================================== CSORDINAL
def _link(name):
    if name == "logit":
        return (lambda x: 1 / (1 + np.exp(-x)), lambda x: np.exp(-x) / (1 + np.exp(-x)) ** 2)
    if name == "probit":
        return stats.norm.cdf, stats.norm.pdf
    if name == "cloglog":
        return (lambda x: 1 - np.exp(-np.exp(x)), lambda x: np.exp(x - np.exp(x)))
    if name == "nloglog":
        return (lambda x: np.exp(-np.exp(-x)), lambda x: np.exp(-x - np.exp(-x)))
    if name == "cauchit":
        return stats.cauchy.cdf, stats.cauchy.pdf
    raise ValueError(name)


def _ordinal_fit(X, yk, J, w, link):
    F, f = _link(link)
    n, p = X.shape
    cum = np.cumsum(np.bincount(yk, w, minlength=J))[:-1] / w.sum()
    theta0 = {"logit": np.log(cum / (1 - cum)), "probit": stats.norm.ppf(cum)}.get(
        link, stats.norm.ppf(cum))

    def unpack(par):
        return par[:J - 1], par[J - 1:]

    def probs(par):
        th, b = unpack(par)
        eta = X @ b
        up = np.where(yk < J - 1, th[np.minimum(yk, J - 2)] - eta, np.inf)
        lo = np.where(yk > 0, th[np.maximum(yk - 1, 0)] - eta, -np.inf)
        Fu = np.where(np.isfinite(up), F(np.where(np.isfinite(up), up, 0)), 1.0)
        Fl = np.where(np.isfinite(lo), F(np.where(np.isfinite(lo), lo, 0)), 0.0)
        fu = np.where(np.isfinite(up), f(np.where(np.isfinite(up), up, 0)), 0.0)
        fl = np.where(np.isfinite(lo), f(np.where(np.isfinite(lo), lo, 0)), 0.0)
        return np.maximum(Fu - Fl, 1e-300), fu, fl

    def scores(par):
        pr, fu, fl = probs(par)
        S = np.zeros((n, J - 1 + p))
        for j in range(J - 1):
            S[:, j] = ((yk == j) * fu - (yk == j + 1) * fl) / pr
        S[:, J - 1:] = -X * ((fu - fl) / pr)[:, None]
        return S

    def nll(par):
        return -float(w @ np.log(probs(par)[0]))

    def grad(par):
        return -(w @ scores(par))

    par0 = np.concatenate([theta0, np.zeros(p)])
    res = optimize.minimize(nll, par0, jac=grad, method="BFGS",
                            options={"gtol": 1e-9, "maxiter": 2000})
    par = res.x
    # observed information by central differences of the analytic weighted score
    k = len(par)
    H = np.zeros((k, k))
    for i in range(k):
        h = 1e-5 * max(1.0, abs(par[i]))
        e = np.zeros(k)
        e[i] = h
        H[:, i] = (grad(par + e) - grad(par - e)) / (2 * h)
    H = (H + H.T) / 2
    return par, H, scores(par), -res.fun


def cs_ordinal(ds: Dataset, dependent: str, factors: list[str] = (),
               covariates: list[str] = (), plan=None, link: str = "logit",
               terms: str | None = None, level: float = 0.95, test: str = "F",
               deff: bool = True, domain: tuple | None = None) -> Result:
    """Cumulative link model P(Y <= j) = F(theta_j - x'beta) as SPSS CSORDINAL (and PLUM)."""
    dsg = design_for(ds, plan)
    factors, covariates = list(factors), list(covariates)
    sec = Section("Complex Samples Ordinal Regression")
    sec.log(_plan_log(dsg, f"CSORDINAL {dependent}" + (f" BY {' '.join(factors)}" if factors
                                                       else "") +
                      (f" WITH {' '.join(covariates)}" if covariates else "") +
                      f"\n  /LINK FUNCTION={link.upper()}\n  /TEST TYPE={test}"))
    yarr, yvals, ylabs = _cats(ds, dependent, dsg.rows)
    yok = pd.notna(pd.Series(yarr)).to_numpy()
    dom, dtext = _domain_mask(ds, dsg, domain)
    ok, Xi, labels, owner, alias, Xe, eidx, tlist = _model_design(
        ds, dsg.rows, yok & dom, factors, covariates, terms, True)
    J = len(yvals)
    if J < 3:
        sec.note("Warning", "Ordinal regression needs at least three response categories.",
                 "warning")
        return Result(sec)
    yk = np.array([yvals.index(a) for a in yarr[ok]], int)
    w = dsg.w[ok]
    sec.table(dsg.info_table(int(ok.sum())))
    keepc = ~alias.copy()
    keepc[0] = False  # thresholds replace the intercept
    X = Xi[:, keepc]
    par, H, S, ll = _ordinal_fit(X, yk, J, w, link)
    Hinv = np.linalg.pinv(H)
    V, Vs = _sandwich(dsg, Hinv, S, ok)
    pbar = np.bincount(yk, w, minlength=J) / w.sum()
    ll0 = float(w @ np.log(pbar[yk]))
    Nw = w.sum()
    cs = 1 - math.exp(2 * (ll0 - ll) / Nw)
    nk = cs / (1 - math.exp(2 * ll0 / Nw))
    sec.table(Table("Pseudo R Squares", ["", "Value"],
                    [["Cox and Snell", fmt(cs, 3)], ["Nagelkerke", fmt(nk, 3)],
                     ["McFadden", fmt(1 - ll / ll0, 3)]],
                    footnotes=[f"Dependent Variable: {dependent}; link function: "
                               f"{LINKS[link]}{dtext}"]))
    # effect tests on the effect-coded design (intercept column dropped)
    Xe2 = Xe[:, 1:]
    pe, He, Se, _ = _ordinal_fit(Xe2, yk, J, w, link)
    Ve, Vse = _sandwich(dsg, np.linalg.pinv(He), Se, ok)
    eidx2 = [[]] + [[J - 1 + c - 1 for c in idx] for idx in eidx[1:]]
    sec.table(_effects_table(tlist, eidx2, pe, Ve, Vse, dsg.df, test, False))
    lab = [f"Threshold [{dependent}={ylabs[j]}]" for j in range(J - 1)]
    full_b, full_alias = list(par[:J - 1]), [False] * (J - 1)
    mapping = list(range(J - 1))
    t = J - 1
    for j in range(1, len(labels)):
        lab.append(f"Location {labels[j]}")
        if keepc[j]:
            full_b.append(par[t])
            full_alias.append(False)
            mapping.append(t)
            t += 1
        else:
            full_b.append(0.0)
            full_alias.append(True)
            mapping.append(None)
    K = len(lab)
    Vf = np.zeros((K, K))
    Vsf = np.zeros((K, K))
    for a in range(K):
        for c in range(K):
            if mapping[a] is not None and mapping[c] is not None:
                Vf[a, c] = V[mapping[a], mapping[c]]
                Vsf[a, c] = Vs[mapping[a], mapping[c]]
    sec.table(_param_table(lab, np.array(full_b), Vf, Vsf, np.array(full_alias), dsg.df,
                           level, exp_b=(link == "logit"), deff=deff))
    return Result(sec)


# ===================================================================== CSCOXREG
def cox_fit(T, E, X, w, ties: str = "efron", maxiter: int = 50):
    """Weighted Cox partial likelihood (Breslow or Efron ties). Returns beta, information
    matrix, score residuals (rows = cases) and the log partial likelihood."""
    n, p = X.shape
    order = np.argsort(T, kind="mergesort")
    T, E, X, w = T[order], E[order], X[order], w[order]
    times = np.unique(T[E > 0])
    beta = np.zeros(p)

    def pieces(beta, want_resid=False):
        eta = X @ beta
        r = np.exp(eta - eta.max())
        scale = math.exp(eta.max())
        ll, U, I = 0.0, np.zeros(p), np.zeros((p, p))
        resid = np.zeros((n, p)) if want_resid else None
        # cumulative compensator terms for the residuals: sum over event times of
        # (wD/d) sum_l [x_i - xbar_l] * r_i / S0_l restricted to the risk set
        for t in times:
            risk = T >= t
            D = (T == t) & (E > 0)
            d = int(D.sum())
            wD = float(w[D].sum())
            wr = w * r
            S0 = wr[risk].sum()
            S1 = (wr[risk][:, None] * X[risk]).sum(axis=0)
            S2 = (X[risk] * wr[risk][:, None]).T @ X[risk]
            S0d = wr[D].sum()
            S1d = (wr[D][:, None] * X[D]).sum(axis=0)
            S2d = (X[D] * wr[D][:, None]).T @ X[D]
            ll += float(w[D] @ eta[D])
            U += (w[D][:, None] * X[D]).sum(axis=0)
            L = d if ties == "efron" else 1
            xbars, s0s = [], []
            for l in range(L):
                fr = l / d if ties == "efron" else 0.0
                s0 = S0 - fr * S0d
                s1 = S1 - fr * S1d
                s2 = S2 - fr * S2d
                xb = s1 / s0
                ll -= (wD / L) * (math.log(s0) + math.log(scale))
                U -= (wD / L) * xb
                I += (wD / L) * (s2 / s0 - np.outer(xb, xb))
                xbars.append(xb)
                s0s.append(s0)
            if want_resid:
                xbar_mean = np.mean(xbars, axis=0)
                resid[D] += X[D] - xbar_mean
                for l in range(L):
                    fr = l / d if ties == "efron" else 0.0
                    fac = np.where(D, 1 - fr, 1.0) * risk
                    resid -= (wD / L) * (fac * r / s0s[l])[:, None] * (X - xbars[l])
        return ll, U, I, resid

    for _ in range(maxiter):
        ll, U, I, _r = pieces(beta)
        step = np.linalg.solve(I, U) if p else np.zeros(0)
        # step halving
        new = beta + step
        ll_new = pieces(new)[0]
        k = 0
        while ll_new < ll - 1e-10 and k < 20:
            step /= 2
            new = beta + step
            ll_new = pieces(new)[0]
            k += 1
        beta = new
        if p == 0 or np.max(np.abs(step)) < 1e-9:
            break
    ll, U, I, resid = pieces(beta, True)
    out = np.zeros_like(resid)
    out[order] = resid
    return beta, I, out, ll


def cs_cox(ds: Dataset, time: str, status: str, event_values: list, factors: list[str] = (),
           covariates: list[str] = (), plan=None, terms: str | None = None,
           ties: str = "efron", level: float = 0.95, test: str = "F", deff: bool = True,
           exp_b: bool = True, domain: tuple | None = None) -> Result:
    dsg = design_for(ds, plan)
    factors, covariates = list(factors), list(covariates)
    sec = Section("Complex Samples Cox Regression")
    sec.log(_plan_log(dsg, f"CSCOXREG {time} BY {' '.join(factors)} WITH "
                      f"{' '.join(covariates)}\n  /VARIABLES STATUS={status}("
                      f"{' '.join(str(v) for v in event_values)})\n  /CRITERIA TIES="
                      f"{ties.upper()}\n  /TEST TYPE={test}"))
    T = _num(ds, time, dsg.rows)
    st = ds.analysis_series(status)[dsg.rows].to_numpy(object)
    stok = pd.notna(pd.Series(st)).to_numpy()
    Ev = np.array([a in set(event_values) for a in st], float)
    dom, dtext = _domain_mask(ds, dsg, domain)
    ok, Xi, labels, owner, alias, Xe, eidx, tlist = _model_design(
        ds, dsg.rows, ~np.isnan(T) & stok & dom, factors, covariates, terms, True)
    keepc = ~alias.copy()
    keepc[0] = False
    w = dsg.w[ok]
    sec.table(dsg.info_table(int(ok.sum())))
    beta, I, R, ll = cox_fit(T[ok], Ev[ok], Xi[:, keepc], w, ties)
    Iinv = np.linalg.pinv(I)
    V, Vs = _sandwich(dsg, Iinv, R, ok)
    ll0 = cox_fit(T[ok], Ev[ok], np.zeros((int(ok.sum()), 0)), w, ties)[3]
    sec.table(Table("Model Fit", ["", "Value"],
                    [["-2 Log Pseudo Likelihood (null model)", fmt(-2 * ll0, 3)],
                     ["-2 Log Pseudo Likelihood (final model)", fmt(-2 * ll, 3)]],
                    footnotes=[f"Ties: {ties.capitalize()}; events: "
                               f"{int(Ev[ok].sum())}{dtext}"]))
    be, Ie, Re, _ = cox_fit(T[ok], Ev[ok], Xe[:, 1:], w, ties)
    Ve, Vse = _sandwich(dsg, np.linalg.pinv(Ie), Re, ok)
    eidx2 = [[]] + [[c - 1 for c in idx] for idx in eidx[1:]]
    sec.table(_effects_table(tlist, eidx2, be, Ve, Vse, dsg.df, test, False))
    lab, full_b, full_alias, mapping = [], [], [], []
    t = 0
    for j in range(1, len(labels)):
        lab.append(labels[j])
        if keepc[j]:
            full_b.append(beta[t])
            full_alias.append(False)
            mapping.append(t)
            t += 1
        else:
            full_b.append(0.0)
            full_alias.append(True)
            mapping.append(None)
    K = len(lab)
    Vf = np.zeros((K, K))
    Vsf = np.zeros((K, K))
    for a in range(K):
        for c in range(K):
            if mapping[a] is not None and mapping[c] is not None:
                Vf[a, c] = V[mapping[a], mapping[c]]
                Vsf[a, c] = Vs[mapping[a], mapping[c]]
    sec.table(_param_table(lab, np.array(full_b), Vf, Vsf, np.array(full_alias), dsg.df,
                           level, exp_b=exp_b, deff=deff))
    return Result(sec)


# ===================================================================== sampling wizard
SAMPLING = {"srswor": "Simple random sampling (without replacement)",
            "srswr": "Simple random sampling (with replacement)",
            "systematic": "Simple systematic", "pps_sys": "PPS systematic",
            "pps_wr": "PPS with replacement", "pps_brewer": "PPS Brewer (two units per "
                                                           "stratum)"}


def select_sample(ds: Dataset, method: str = "srswor", strata: list[str] = (),
                  clusters: list[str] = (), size: float = 10, size_kind: str = "count",
                  measure: str | None = None, seed: int = 2000000,
                  save_plan: bool = True) -> Section:
    """Sampling Wizard (CSSELECT subset): selects clusters/cases per stratum and adds SPSS's
    variables InclusionProbability_1_, SampleWeight_1_, PopulationSize_1_,
    SampleSize_1_ and SampleWeight_Final_ (missing for unselected cases). With replacement,
    the number of hits is stored in SampleHits_1_ (a case drawn twice counts twice)."""
    rng = np.random.default_rng(seed)
    n0 = ds.n_cases
    sec = Section("Complex Samples Selection")
    sec.log(f"CSSELECT /PLAN DESIGN\n  /DESIGN STRATA={' '.join(strata)} "
            f"CLUSTER={' '.join(clusters)}\n  /METHOD TYPE={method.upper()}"
            f"\n  /SIZE {'VALUE' if size_kind == 'count' else 'PROPORTION'}={size:g}"
            f"\n  /CRITERIA SEED={seed}")
    all_rows = np.ones(n0, bool)
    for nm in list(strata) + list(clusters):
        all_rows &= ds.analysis_series(nm).notna().to_numpy()
    st_codes, _ = _codes(ds, list(strata), all_rows)
    un_codes, _ = _codes(ds, list(strata) + list(clusters), all_rows) if clusters else \
        (np.arange(all_rows.sum()), None)
    mos = None
    if method.startswith("pps"):
        if not measure:
            raise ValueError("PPS sampling needs a measure of size variable.")
        mos = pd.to_numeric(ds.analysis_series(measure), errors="coerce"
                            ).to_numpy(float)[all_rows]
    incl = np.full(all_rows.sum(), np.nan)
    hits = np.zeros(all_rows.sum())
    popsize = np.full(all_rows.sum(), np.nan)
    samp = np.full(all_rows.sum(), np.nan)
    summary = []
    for h in np.unique(st_codes):
        m = st_codes == h
        units = np.unique(un_codes[m])
        Nh = len(units)
        nh = int(round(size if size_kind == "count" else size * Nh))
        nh = max(1, min(nh, Nh)) if method in ("srswor", "systematic", "pps_sys") else max(1, nh)
        if method == "pps_brewer":
            nh = 2 if Nh >= 2 else 1
        if mos is not None:
            sz = np.array([np.nansum(mos[m & (un_codes == u)]) for u in units])
            sz = np.where(sz > 0, sz, 0)
        sel_count = np.zeros(Nh)
        pi = np.zeros(Nh)
        if method == "srswor":
            pick = rng.choice(Nh, nh, replace=False)
            sel_count[pick] = 1
            pi[:] = nh / Nh
        elif method == "srswr":
            pick = rng.integers(0, Nh, nh)
            np.add.at(sel_count, pick, 1)
            pi[:] = 1 - (1 - 1 / Nh) ** nh
        elif method == "systematic":
            k = Nh / nh
            start = rng.uniform(0, k)
            pick = np.floor(start + k * np.arange(nh)).astype(int)
            sel_count[np.minimum(pick, Nh - 1)] = 1
            pi[:] = nh / Nh
        elif method == "pps_sys":
            pi = np.minimum(nh * sz / sz.sum(), 1.0)
            cum = np.concatenate([[0], np.cumsum(pi)])
            start = rng.uniform(0, 1)
            pts = start + np.arange(int(round(pi.sum())))
            pick = np.searchsorted(cum, pts, side="right") - 1
            sel_count[np.clip(pick, 0, Nh - 1)] = 1
        elif method == "pps_wr":
            prob = sz / sz.sum()
            pick = rng.choice(Nh, nh, replace=True, p=prob)
            np.add.at(sel_count, pick, 1)
            pi = nh * prob   # expected hits; weight = 1/(n p)
        elif method == "pps_brewer":
            p = sz / sz.sum()
            if Nh >= 2:
                q = p * (1 - p) / (1 - 2 * p)
                first = rng.choice(Nh, p=q / q.sum())
                p2 = p.copy()
                p2[first] = 0
                second = rng.choice(Nh, p=p2 / p2.sum())
                sel_count[[first, second]] = 1
                pi = np.minimum(2 * p, 1.0)
            else:
                sel_count[:] = 1
                pi[:] = 1
        else:
            raise ValueError(method)
        for u_i, u in enumerate(units):
            mm = m & (un_codes == u)
            popsize[mm] = Nh
            samp[mm] = nh
            if sel_count[u_i] > 0:
                incl[mm] = pi[u_i]
                hits[mm] = sel_count[u_i]
        summary.append([str(h + 1), fmt_int(Nh), fmt_int(nh),
                        fmt_int((sel_count > 0).sum()), fmt(pi[sel_count > 0].mean()
                                                            if (sel_count > 0).any() else 0, 4)])
    wgt = np.where(hits > 0, 1 / incl, np.nan)
    if method in ("srswr", "pps_wr"):
        wgt = np.where(hits > 0, hits / (incl if method == "pps_wr" else
                                         samp / np.where(popsize > 0, popsize, 1)), np.nan)
    full = {}
    for nm, arr in (("InclusionProbability_1_", np.where(hits > 0, incl, np.nan)),
                    ("SampleWeight_1_", wgt), ("PopulationSize_1_", popsize),
                    ("SampleSize_1_", samp), ("SampleHits_1_", np.where(hits > 0, hits,
                                                                         np.nan)),
                    ("SampleWeight_Final_", wgt)):
        col = np.full(n0, np.nan)
        col[all_rows] = arr
        full[nm] = col
    for nm, col in full.items():
        if ds.has_name(nm):
            ds.df[nm] = col
        else:
            var = ds.add_variable(nm, values=col)
            var.decimals = 4 if "Probability" in nm else 2
    sec.table(Table("Stage Summary", ["Stratum", "Population Units", "Requested Sample Size",
                                      "Units Selected", "Mean Inclusion Probability"],
                    summary, footnotes=[f"Method: {SAMPLING[method]}; seed {seed}."]))
    if save_plan:
        if method in ("srswr", "pps_wr"):
            stg = Stage(list(strata), list(clusters), "WR")
        elif method in ("srswor", "systematic"):
            stg = Stage(list(strata), list(clusters), "EQUAL_WOR", "popsize", None,
                        "PopulationSize_1_")
        else:
            stg = Stage(list(strata), list(clusters), "UNEQUAL_WOR", "inclusion", None,
                        "InclusionProbability_1_")
        set_plan(ds, Plan("SampleWeight_Final_", [stg]))
        sec.note("Plan", "An analysis plan for the selected sample was stored with the data "
                 "(weight SampleWeight_Final_). Unselected cases have missing weights and are "
                 "excluded from Complex Samples analyses.")
    return sec


def prepare_plan(ds: Dataset, plan: Plan) -> Section:
    """Analysis Preparation Wizard: validate and store the plan; summary like SPSS's plan
    summary."""
    sec = Section("Complex Samples Plan")
    for nm in ([plan.weight] if plan.weight else []) + \
            [v for s in plan.stages for v in list(s.strata) + list(s.clusters) +
             ([s.fpc_var] if s.fpc_var else [])]:
        if not ds.has_name(nm):
            raise ValueError(f"Unknown variable in plan: {nm}")
    set_plan(ds, plan)
    dsg = Design(ds, plan)
    rows = [["Weight Variable", plan.weight or "(none)"],
            ["SRS Estimator", "Sampling without replacement" if plan.srs_estimator == "WOR"
             else "Sampling with replacement"]]
    for k, s in enumerate(plan.stages, 1):
        rows += [[f"Stage {k}: Design Variables: Stratification", ", ".join(s.strata) or
                  "(none)"],
                 [f"Stage {k}: Design Variables: Clustering", ", ".join(s.clusters) or "(none)"],
                 [f"Stage {k}: Estimator Assumptions", METHODS[s.method]]]
        if s.fpc_kind != "none":
            rows.append([f"Stage {k}: " + {"rate": "Sampling rate", "popsize":
                                             "Population size", "inclusion":
                                             "Inclusion probability"}[s.fpc_kind],
                         s.fpc_var or fmt(s.fpc_value or 0, 4)])
    sec.table(Table("Plan Summary", ["", "Value"], rows))
    sec.table(dsg.info_table())
    sec.note("Plan", "The plan is stored with the dataset (saved in .rsd projects).")
    return sec


# ===================================================================== selftest
def selftest() -> None:
    gpa = [3.08, 2.60, 3.44, 3.04, 2.36, 3.04, 3.28, 2.68, 2.00, 2.56, 2.52, 1.88, 3.00,
           2.88, 3.44, 3.64, 2.68, 1.92, 3.28, 3.20]
    ds = Dataset(pd.DataFrame({"suite": np.repeat(np.arange(1, 6), 4).astype(float),
                               "gpa": gpa, "pwt": 20.0}))
    plan = Plan("pwt", [Stage([], ["suite"], "WR")])
    dsg = Design(ds, plan)
    e = total_est(dsg, np.array(gpa), np.ones(20, bool))
    assert abs(e.value - 1130.4) < 1e-9 and abs(e.se - 67.167) < 5e-4
    cs_descriptives(ds, ["gpa"], plan, sums=True, statistics=("se", "ci", "deff"))
    ds.df["grp"] = np.tile([1.0, 2.0], 10)
    ds.df["y2"] = np.array([1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1.0])
    cs_crosstabs(Dataset(ds.df.copy()), "grp", "y2", plan)
    cs_glm(Dataset(ds.df.copy()), "gpa", ["grp"], [], plan)
    cs_logistic(Dataset(ds.df.copy()), "y2", [], ["gpa"], plan)
