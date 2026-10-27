"""WP-3 Statistics: outlier, variance, proportion, distribution, survival, ROC, power,
repeated-measures and nonparametric analyses of the Origin-style module, checked against
NIST StRD certified values, the NIST/SEMATECH Grubbs example, Rorabacher's Dixon table and
reference implementations (scipy, statsmodels, pingouin — the latter only in tests)."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from rsstatchem.core.analysis import distfit as dfit
from rsstatchem.core.analysis import outliers as ol
from rsstatchem.core.analysis import vartests as vt
from rsstatchem.core.analysis import weibull_surv as wb
from rsstatchem.ui.origin import stat_tests as st
from rsstatchem.ui.origin.model import Workbook

DATA = Path(__file__).parent / "data"


def _book(cols, name="Book1") -> Workbook:
    b = Workbook.new(name, n_cols=len(cols))
    ws = b.sheets[0]
    ws.set_rows(max(len(c) for c in cols))
    for j, c in enumerate(cols):
        ws.set_column_values(j, list(c))
    return b


def R(col: str, book: str = "Book1") -> str:
    return f"[{book}]Sheet1!{col}"


def _cell(report, table: str, row: str, col) -> str:
    t = report.table(table)
    assert t is not None, [i.get("title") for i in report.items]
    r = next(r for r in t["rows"] if r[0] == row)
    j = col if isinstance(col, int) else [c if isinstance(c, str) else c[1]
                                          for c in t["columns"]].index(col)
    return r[j]


def _f(report, table, row, col) -> float:
    return float(_cell(report, table, row, col).split()[0].rstrip("*"))


# ====================================================================== outliers
def test_grubbs_nist_sematech_example():
    # NIST/SEMATECH e-Handbook 1.3.5.17.1: uranium isotope data, G = 2.4687,
    # upper one-tailed critical value 2.032 at alpha = 0.05
    x = [199.31, 199.53, 200.19, 200.82, 201.92, 201.95, 202.18, 245.57]
    r = ol.grubbs(x, 0.05, "max")
    assert r.statistic == pytest.approx(2.4687, abs=1e-3)
    assert r.critical == pytest.approx(2.032, abs=5e-4)
    assert r.outlier and r.index == 7
    two = ol.grubbs(x, 0.05, "two")
    assert two.critical > r.critical and two.p == pytest.approx(2 * r.p)
    steps = ol.grubbs_iterative(x, 0.05, "max")
    assert steps[0].outlier and not steps[-1].outlier


def test_grubbs_pvalue_matches_critical_value():
    for n in (5, 10, 30, 100):
        for side in ("two", "max"):
            g = ol.grubbs_critical(n, 0.05, side)
            assert ol.grubbs_pvalue(g, n, side) == pytest.approx(0.05, rel=1e-8)


@pytest.mark.parametrize("n", range(3, 11))
def test_dixon_r10_against_rorabacher_table(n):
    printed = ol.RORABACHER_R10[n]
    exact = [ol.dixon_critical(n, a, "r10") for a in (0.05, 0.025, 0.005)]
    # Rorabacher's printed values come from a limited-precision integration: they agree
    # with the exact ones to 0.002 except n = 4 at 99 % (0.926 vs 0.921, McBane 2006)
    for e, p in zip(exact, printed):
        assert e == pytest.approx(p, abs=0.006)
    assert sum(abs(e - p) <= 0.0025 for e, p in zip(exact, printed)) >= 2


def test_dixon_exact_against_monte_carlo():
    rng = np.random.default_rng(1991)
    for kind, n in (("r10", 5), ("r11", 9), ("r21", 12), ("r22", 16)):
        i, j = ol.DIXON_KINDS[kind]
        s = np.sort(rng.standard_normal((200_000, n)), axis=1)
        r = (s[:, i] - s[:, 0]) / (s[:, n - 1 - j] - s[:, 0])
        mc = np.quantile(r, 0.95)
        assert ol.dixon_critical(n, 0.05, kind) == pytest.approx(mc, abs=0.004)
        q = float(np.quantile(r, 0.5))
        assert ol.dixon_sf(q, n, kind) == pytest.approx(0.5, abs=0.005)
    assert ol.dixon_sf(0.0, 10, "r22") == pytest.approx(1.0, abs=1e-9)


def test_dixon_test_and_table_option():
    x = [0.189, 0.167, 0.187, 0.183, 0.186, 0.182, 0.181, 0.184, 0.181, 0.177]
    r = ol.dixon(x, 0.05, "r10", "min")
    assert r.end == "low" and r.value == 0.167
    assert r.statistic == pytest.approx((0.177 - 0.167) / (0.189 - 0.167))
    t = ol.dixon([1.0, 1.1, 1.2, 1.25, 3.0], 0.05, "r10", "two", critical="table")
    assert t.source == "table" and t.critical == 0.710 and t.outlier
    assert ol.auto_kind(6) == "r10" and ol.auto_kind(9) == "r11" and ol.auto_kind(12) == "r21" \
        and ol.auto_kind(20) == "r22"


def test_outlier_reports_and_recalc():
    v = [10.1, 10.3, 9.9, 10.0, 10.2, 13.5, 10.1, 9.8]
    books = {"Book1": _book([v])}
    rep, text = st.run("grubbs", books, {"ranges": [R("A")], "alpha": 0.05})
    assert _cell(rep, "Grubbs Test", "A", "Row") == "6"
    assert _cell(rep, "Grubbs Test", "A", "Decision at level(5%)") == "Outlier"
    rep2, _ = st.run("dixon", books, {"ranges": [R("A")], "kind": "auto"})
    assert _cell(rep2, "Dixon's Q Test", "A", "Statistic") == "r11"
    books["Book1"].sheets[0].set_column_values(0, [10.1, 10.3, 9.9, 10.0, 10.2, 10.4, 10.1,
                                                   9.8])
    st.run("grubbs", books, rep.op["settings"], report=rep)
    assert _cell(rep, "Grubbs Test", "A", "Decision at level(5%)") == "Not an outlier"


# ====================================================================== NIST StRD
def _nist_anova(name: str):
    lines = (DATA / "nist_anova" / f"{name}.dat").read_text().splitlines()
    cert = {}
    for ln in lines:
        parts = ln.split()
        if ln.startswith("Between"):
            cert["df_b"], cert["ss_b"], cert["ms_b"], cert["f"] = \
                int(parts[-4]), float(parts[-3]), float(parts[-2]), float(parts[-1])
        elif ln.startswith("Within"):
            cert["df_w"], cert["ss_w"], cert["ms_w"] = int(parts[-3]), float(parts[-2]), \
                float(parts[-1])
        elif "Certified R-Squared" in ln:
            cert["r2"] = float(parts[-1])
        elif "Standard Deviation" in ln and "Residual" not in ln:
            cert["sd"] = float(parts[-1])
    start = max(i for i, ln in enumerate(lines) if ln.startswith("Data:")) + 1
    rows = [ln.split() for ln in lines[start:] if ln.strip()]
    return cert, [r[0] for r in rows], np.array([float(r[1]) for r in rows])


@pytest.mark.parametrize("name", ["SiRstv", "SmLs01", "SmLs02", "SmLs03", "SmLs04", "SmLs05",
                                  "SmLs06", "SmLs07", "SmLs08", "SmLs09", "AtmWtAg"])
def test_nist_strd_anova_one_way(name):
    cert, groups, y = _nist_anova(name)
    books = {"Book1": _book([groups, y])}
    rep, _ = st.run("anova1", books, {"form": "indexed", "data": R("B"), "factor": R("A"),
                                      "means": [], "means_plot": False,
                                      "homogeneity": "none"})
    # report values carry Origin's 6 significant digits
    assert _f(rep, "Overall ANOVA", "Model", "DF") == cert["df_b"]
    assert _f(rep, "Overall ANOVA", "Error", "DF") == cert["df_w"]
    assert _f(rep, "Overall ANOVA", "Model", "Sum of Squares") == pytest.approx(cert["ss_b"],
                                                                                rel=1e-5)
    assert _f(rep, "Overall ANOVA", "Error", "Sum of Squares") == pytest.approx(cert["ss_w"],
                                                                                rel=1e-5)
    assert _f(rep, "Overall ANOVA", "Model", "F Value") == pytest.approx(cert["f"], rel=1e-4)
    t = rep.table("Fit Statistics")["rows"][0]
    assert float(t[0]) == pytest.approx(cert["r2"], rel=1e-5)
    assert float(t[2]) == pytest.approx(cert["sd"], rel=1e-5)
    # and the k-sample variance tests run on the same layout without loss of precision
    rep2, _ = st.run("vark", books, {"form": "indexed", "data": R("B"), "factor": R("A")})
    lev = stats.levene(*[y[np.array(groups) == g] for g in dict.fromkeys(groups)],
                       center="mean")
    assert _f(rep2, "Tests for Equal Variance", "Levene (absolute deviations from mean)",
              "Statistic") == pytest.approx(lev.statistic, rel=1e-4)


NIST_UNIV = {   # certified mean, SD (n − 1), lag-1 autocorrelation
    "Lew": (-177.435000000000, 277.332168044316, -0.307304800605679),
    "Lottery": (518.958715596330, 291.699727470969, -0.120948622967393),
    "Mavro": (2.00185600000000, 0.000429123454003053, 0.937989183438248),
    "Michelso": (299.852400000000, 0.0790105478190518, 0.535199668621283),
    "NumAcc1": (10000002.0, 1.0, -0.5),
    "NumAcc2": (1.2, 0.1, -0.999),
    "NumAcc3": (1000000.2, 0.1, -0.999),
    "NumAcc4": (10000000.2, 0.1, -0.999),
}


def _univ(name: str) -> np.ndarray:
    if name == "NumAcc1":
        return np.array([10000001.0, 10000003.0, 10000002.0])
    if name.startswith("NumAcc"):
        base = {"NumAcc2": 1.2, "NumAcc3": 1000000.2, "NumAcc4": 10000000.2}[name]
        return np.array([base] + [base - 0.1, base + 0.1] * 500)
    return np.array((DATA / "nist_univ" / f"{name}.txt").read_text().split(), float)


@pytest.mark.parametrize("name", list(NIST_UNIV))
def test_nist_strd_univariate(name):
    v = _univ(name)
    mean, sd, r1 = NIST_UNIV[name]
    d = st.describe(v)
    assert d["mean"] == pytest.approx(mean, rel=1e-13)
    tol = 1e-8 if name in ("NumAcc3", "NumAcc4") else 1e-12   # 7–8 constant leading digits
    assert d["sd"] == pytest.approx(sd, rel=tol)
    dev = v - v.mean()
    assert float((dev[:-1] * dev[1:]).sum() / (dev * dev).sum()) == pytest.approx(r1,
                                                                                 rel=1e-7)
    rep, _ = st.run("desc", {"Book1": _book([v])}, {"ranges": [R("A")]})
    assert _f(rep, "Descriptive Statistics", "A", "Mean") == pytest.approx(mean, rel=5e-6)
    assert _f(rep, "Descriptive Statistics", "A", "Standard Deviation") == \
        pytest.approx(sd, rel=1e-5)


# ====================================================================== variance tests
def test_variance_tests_against_scipy():
    rng = np.random.default_rng(5)
    a, b, c = rng.normal(0, 1, 15), rng.normal(0, 2, 20), rng.normal(0, 1.5, 12)
    one = vt.chi2_variance_test(a, 2.0, "two")
    chi = (len(a) - 1) * a.var(ddof=1) / 2.0
    assert one["chi2"] == pytest.approx(chi)
    assert one["p"] == pytest.approx(2 * stats.chi2.cdf(chi, len(a) - 1))
    two = vt.f_variance_test(a, b, 1.0, "two")
    f = a.var(ddof=1) / b.var(ddof=1)
    assert two["f"] == pytest.approx(f)
    assert two["p"] == pytest.approx(2 * min(stats.f.cdf(f, 14, 19), stats.f.sf(f, 14, 19)))
    assert two["lcl"] < f < two["ucl"]
    k = vt.k_sample_tests([a, b, c])
    assert k["bartlett"][0] == pytest.approx(stats.bartlett(a, b, c).statistic)
    assert k["levene"][2] == pytest.approx(stats.levene(a, b, c, center="mean").pvalue)
    assert k["brown_forsythe"][2] == pytest.approx(stats.levene(a, b, c).pvalue)
    books = {"Book1": _book([a, b])}
    rep, _ = st.run("var2", books, {"ranges": [R("A"), R("B")], "tail": "lower"})
    assert _f(rep, "F-Test for Equal Variance", "A / B", "Prob") == pytest.approx(
        stats.f.cdf(f, 14, 19), rel=1e-4)


# ====================================================================== proportions
def test_proportion_tests_against_statsmodels():
    from statsmodels.stats.proportion import proportions_ztest
    books = {"Book1": _book([[1, 0, 1, 1, 0, 1, 1, 1, 0, 1]])}
    rep, _ = st.run("prop1", books, {"form": "summary", "x": 18, "n": 30, "p0": 0.4,
                                     "method": "exact", "book": "Book1"})
    # two-sided exact p = 2 × the one-sided tail (SPSS / Origin convention)
    assert _f(rep, "Test Statistics", "Exact Binomial", "Prob") == pytest.approx(
        2 * stats.binom.sf(17, 30, 0.4), rel=1e-4)
    assert 2 * stats.binom.sf(17, 30, 0.4) == pytest.approx(
        2 * stats.binomtest(18, 30, 0.4, alternative="greater").pvalue)
    rep, _ = st.run("prop1", books, {"form": "data", "range": R("A"), "success": "1",
                                     "p0": 0.5, "method": "score", "book": "Book1"})
    assert _cell(rep, "Proportion", "7", 1) == "10"
    z, p = proportions_ztest(7, 10, 0.5, prop_var=0.5)
    assert _f(rep, "Test Statistics", "Score (normal approximation)", "Z") == \
        pytest.approx(z, rel=1e-4)
    rep, _ = st.run("prop2", books, {"form": "summary", "x1": 30, "n1": 80, "x2": 45,
                                     "n2": 90, "method": "wald_h0", "book": "Book1"})
    z, p = proportions_ztest([30, 45], [80, 90])
    assert abs(_f(rep, "Test Statistics", "Pooled z (Wald H0)", "Z")) == \
        pytest.approx(abs(z), rel=1e-4)
    assert _f(rep, "Test Statistics", "Pooled z (Wald H0)", "Prob") == pytest.approx(p,
                                                                                    rel=1e-4)
    fe = stats.fisher_exact([[30, 50], [45, 45]])[1]
    assert _f(rep, "Test Statistics", "Fisher's Exact Test", "Prob") == pytest.approx(fe,
                                                                                     rel=1e-4)


# ====================================================================== distribution fit
def test_distribution_fit_mle_and_se():
    rng = np.random.default_rng(11)
    v = rng.gamma(3.0, 2.0, 300)
    n = len(v)
    f = dfit.fit("normal", v)
    assert f.params[0] == pytest.approx(v.mean())
    assert f.params[1] == pytest.approx(v.std())
    assert f.se[0] == pytest.approx(v.std() / math.sqrt(n), rel=1e-4)
    assert f.se[1] == pytest.approx(v.std() / math.sqrt(2 * n), rel=1e-3)
    e = dfit.fit("exponential", v)
    assert e.params[0] == pytest.approx(v.mean()) and e.se[0] == pytest.approx(
        v.mean() / math.sqrt(n), rel=1e-4)
    g = dfit.fit("gamma", v)
    a, _, scale = stats.gamma.fit(v, floc=0)
    assert g.params == pytest.approx([a, scale], rel=1e-6)
    w = dfit.fit("weibull", v)
    k, _, lam = stats.weibull_min.fit(v, floc=0)
    assert w.params == pytest.approx([k, lam], rel=1e-6)
    assert g.loglik == pytest.approx(np.sum(stats.gamma.logpdf(v, a, scale=scale)))
    assert g.aic < f.aic   # gamma data: gamma beats normal
    assert g.ks == pytest.approx(stats.kstest(v, stats.gamma(a, scale=scale).cdf).statistic)
    ad = stats.anderson(v, "norm")   # scipy's A² uses the n − 1 SD
    assert dfit.anderson_darling(v, stats.norm(v.mean(), v.std(ddof=1))) == pytest.approx(
        ad.statistic, rel=1e-10)
    b = dfit.fit("normal", v[:40], bootstrap=99)
    assert 0 < b.ks_p <= 1 and "bootstrap" in b.p_method


def test_distribution_fit_report():
    rng = np.random.default_rng(3)
    v = rng.lognormal(1.0, 0.4, 80)
    books = {"Book1": _book([v])}
    rep, text = st.run("distfit", books, {"range": R("A"),
                                          "dists": ["normal", "lognormal", "weibull"]})
    gof = rep.table("Goodness of Fit")
    assert gof["rows"][0][0] == "Lognormal"   # ranked first by AIC
    assert len(rep.graphs()) == 2


# ====================================================================== survival
def test_weibull_fit_closed_forms_and_references():
    rng = np.random.default_rng(7)
    t = rng.exponential(5.0, 60)
    c = rng.uniform(1, 15, 60)
    d = (t <= c).astype(float)
    tt = np.minimum(t, c)
    e = wb.weibull_fit(tt, d, exponential=True)
    assert e.scale == pytest.approx(tt.sum() / d.sum())
    assert e.se_scale == pytest.approx(e.scale / math.sqrt(d.sum()))
    # uncensored: same as scipy's weibull_min MLE with loc = 0
    u = rng.weibull(1.8, 120) * 3.0
    f = wb.weibull_fit(u, np.ones_like(u))
    k, _, lam = stats.weibull_min.fit(u, floc=0)
    # scipy's optimizer stops near the maximum; the profile-score root is at least as good
    assert f.shape == pytest.approx(k, rel=1e-4) and f.scale == pytest.approx(lam, rel=1e-4)
    assert f.loglik >= np.sum(stats.weibull_min.logpdf(u, k, scale=lam)) - 1e-9
    # censored: the maximum of the log-likelihood (direct optimisation)
    from scipy.optimize import minimize
    w = wb.weibull_fit(tt, d)
    res = minimize(lambda p: -wb.weibull_loglik(math.exp(p[0]), math.exp(p[1]), tt, d),
                   [0.0, 1.0], method="Nelder-Mead",
                   options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 4000})
    assert w.loglik >= -res.fun - 1e-9 and w.loglik == pytest.approx(-res.fun, abs=1e-6)
    assert [w.shape, w.scale] == pytest.approx(np.exp(res.x), rel=1e-4)
    # observed-information SE against a numerical Hessian of the log-likelihood
    from rsstatchem.core.analysis.distfit import _hessian
    H = _hessian(lambda p: wb.weibull_loglik(p[0], p[1], tt, d), np.array([w.shape, w.scale]))
    se = np.sqrt(np.diag(np.linalg.inv(-H)))
    assert [w.se_shape, w.se_scale] == pytest.approx(se, rel=1e-3)
    assert w.median == pytest.approx(w.scale * math.log(2) ** (1 / w.shape))


def test_weibull_groups_lr_test():
    rng = np.random.default_rng(2)
    t = np.r_[rng.weibull(1.5, 50) * 4, rng.weibull(1.5, 50) * 8]
    g = ["a"] * 50 + ["b"] * 50
    res = wb.weibull_groups(t, np.ones(100), g)
    lr, df, p = res["lr_equal"]
    assert df == 2 and lr > 10 and p < 0.01
    assert res["lr_shape"][1] == 1 and res["lr_shape"][2] > 0.01


def _gehan():
    import csv
    rows = list(csv.DictReader((Path(__file__).parent / "data_wp4" / "gehan.csv").open()))
    return rows


def test_kaplan_meier_report_gehan():
    rows = _gehan()
    keys = list(rows[0])
    tk = next(k for k in keys if k.lower() in ("time", "weeks", "t"))
    ck = next(k for k in keys if k.lower() in ("cens", "status", "event", "relapse"))
    gk = next(k for k in keys if k.lower() in ("treat", "group", "drug", "trt"))
    t = [float(r[tk]) for r in rows]
    c = [r[ck] for r in rows]
    g = [r[gk] for r in rows]
    books = {"Book1": _book([t, c, g])}
    rep, text = st.run("km", books, {"time": R("A"), "censor": R("B"), "censor_value": "0",
                                     "group": R("C")})
    lr = rep.table("Test of Equality over Groups")["rows"][0]
    # Gehan (1965) 6-MP vs placebo: log-rank chi-square 16.79 (1 DF)
    assert float(lr[1]) == pytest.approx(16.79, abs=0.01)
    meds = {r[0]: r[2] for r in rep.table("Quartile Estimates")["rows"]
            if r[1] == "50% (Median)"}
    assert sorted(float(v) for v in meds.values()) == [8.0, 23.0]
    assert len(rep.graphs()) == 1


def test_cox_against_statsmodels_phreg():
    from statsmodels.duration.hazard_regression import PHReg
    rng = np.random.default_rng(4)
    n = 120
    x1, x2 = rng.normal(size=n), rng.binomial(1, 0.4, n).astype(float)
    t = rng.exponential(1 / np.exp(0.7 * x1 - 0.5 * x2))
    c = rng.exponential(2.0, n)
    tt, d = np.round(np.minimum(t, c), 2) + 0.01, (t <= c).astype(int)
    ref = PHReg(tt, np.column_stack([x1, x2]), status=d, ties="breslow").fit()
    books = {"Book1": _book([tt, d, x1, x2])}
    rep, _ = st.run("cox", books, {"time": R("A"), "censor": R("B"), "censor_value": "0",
                                   "covariates": [R("C"), R("D")]})
    for j, col in enumerate("CD"):
        assert _f(rep, "Parameter Estimates", col, "Coefficient (B)") == pytest.approx(
            ref.params[j], rel=1e-5)
        assert _f(rep, "Parameter Estimates", col, "Standard Error") == pytest.approx(
            ref.bse[j], rel=1e-4)
    lrt = rep.table("Testing Global Null Hypothesis: Beta = 0")["rows"][0]
    assert float(lrt[1]) > 10


def test_weibull_report():
    rng = np.random.default_rng(8)
    t = rng.weibull(2.0, 60) * 10
    c = (rng.uniform(size=60) < 0.85).astype(int)
    books = {"Book1": _book([t, c])}
    rep, _ = st.run("weibull", books, {"time": R("A"), "censor": R("B"),
                                       "censor_value": "0"})
    f = wb.weibull_fit(t, c)
    assert _f(rep, "Parameters", "All", "Value") == pytest.approx(f.shape, rel=1e-5)
    assert len(rep.graphs()) == 2


# ====================================================================== ROC
def test_roc_against_sklearn_and_delong():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(9)
    y = rng.binomial(1, 0.4, 150)
    a = y * 1.0 + rng.normal(size=150)
    b = y * 0.5 + rng.normal(size=150)
    books = {"Book1": _book([a, b, y])}
    rep, _ = st.run("roc", books, {"ranges": [R("A"), R("B")], "state": R("C"),
                                   "positive": "1"})
    assert _f(rep, "Area Under the Curve", "A", "Area") == pytest.approx(
        roc_auc_score(y, a), rel=5e-5)
    assert _f(rep, "Area Under the Curve", "B", "Area") == pytest.approx(
        roc_auc_score(y, b), rel=5e-5)
    from rsstatchem.core.stats.roc import auc_details
    se = auc_details(a[y == 1], a[y == 0])["se_delong"]
    assert _f(rep, "Area Under the Curve", "A", "Standard Error") == pytest.approx(se,
                                                                                  rel=1e-4)
    assert rep.table("Pairwise Comparison of ROC Curves (DeLong)") is not None
    rep2, _ = st.run("roc", books, {"ranges": [R("A")], "state": R("C"), "positive": "1",
                                    "direction": "smaller"})
    assert _f(rep2, "Area Under the Curve", "A", "Area") == pytest.approx(
        1 - roc_auc_score(y, a), rel=5e-5)


# ====================================================================== power
def test_power_against_statsmodels():
    from statsmodels.stats.power import FTestAnovaPower, TTestIndPower, TTestPower
    from rsstatchem.ui.origin import stat_power_roc as spr
    d2 = spr.make_design({"procedure": "t2", "mean1": 0.5, "mean2": 0, "sd": 1})
    assert d2.power(64) == pytest.approx(TTestIndPower().power(0.5, 64, 0.05), rel=1e-6)
    d1 = spr.make_design({"procedure": "t1", "mean": 0.4, "null": 0, "sd": 1,
                          "tail": "one"})
    assert d1.power(30) == pytest.approx(
        TTestPower().power(0.4, 30, 0.05, alternative="larger"), rel=1e-6)
    da = spr.make_design({"procedure": "anova1", "means": "0, 0.5, 1", "sd": 1})
    f = da.effect()
    assert da.power(20) == pytest.approx(FTestAnovaPower().power(f, 60, 0.05, 3), rel=1e-6)
    # variances: exact chi-square / F power against Monte Carlo
    rng = np.random.default_rng(13)
    v1 = spr.make_design({"procedure": "var1", "var": 2.0, "var0": 1.0})
    s2 = rng.chisquare(19, 200_000) * 2.0 / 19
    chi = 19 * s2 / 1.0
    mc = np.mean((chi < stats.chi2.ppf(0.025, 19)) | (chi > stats.chi2.ppf(0.975, 19)))
    assert v1.power(20) == pytest.approx(mc, abs=0.005)
    v2 = spr.make_design({"procedure": "var2", "ratio_var": 3.0})
    fr = 3.0 * rng.f(14, 14, 200_000)
    mc = np.mean((fr < stats.f.ppf(0.025, 14, 14)) | (fr > stats.f.ppf(0.975, 14, 14)))
    assert v2.power(15) == pytest.approx(mc, abs=0.005)
    books = {"Book1": _book([[1.0]])}
    rep, _ = st.run("power", books, {"procedure": "t2", "mean1": 0.5, "mean2": 0, "sd": 1,
                                     "powers": "0.8", "book": "Book1"})
    row = rep.table("Sample Size")["rows"][0]
    assert row[1] == "64" and row[2] == "64"   # Cohen: d = 0.5, power 0.8 → 64 per group
    assert len(rep.graphs()) == 1


# ====================================================================== repeated measures
def _long(Y, names=("A",)):
    import pandas as pd
    n, k = Y.shape
    return pd.DataFrame({"subj": np.repeat(np.arange(n), k),
                         "cond": np.tile(np.arange(k), n), "y": Y.ravel()})


def test_rm_anova_one_way_against_pingouin():
    pg = pytest.importorskip("pingouin")
    rng = np.random.default_rng(21)
    n, k = 12, 4
    Y = rng.normal(size=(n, 1)) + rng.normal(size=(n, k)) * np.array([1, 1.5, 2, 3]) + \
        np.array([0, 0.3, 0.8, 1.0])
    books = {"Book1": _book([Y[:, j] for j in range(k)])}
    rep, _ = st.run("rmanova1", books, {"ranges": [R(c) for c in "ABCD"]})
    ref = pg.rm_anova(data=_long(Y), dv="y", within="cond", subject="subj", correction=True,
                      detailed=True)
    row = ref.iloc[0]
    t = rep.table("Tests of Within-Subjects Effects")["rows"]
    sph = next(r for r in t if r[1] == "Sphericity Assumed" and r[0] == "Factor")
    gg = t[t.index(sph) + 1]
    assert float(sph[5]) == pytest.approx(row["F"], rel=3e-5)
    assert float(sph[6]) == pytest.approx(row["p_unc"] if "p_unc" in row else row["p-unc"],
                                          rel=1e-4)
    p_gg = row.get("p_GG_corr", row.get("p-GG-corr"))
    assert float(gg[6]) == pytest.approx(p_gg, rel=1e-4)
    m = rep.table("Mauchly's Test of Sphericity")["rows"][0]
    assert float(m[5]) == pytest.approx(row["eps"], rel=3e-5)
    assert float(m[1]) == pytest.approx(row.get("W_spher", row.get("W-spher")), rel=1e-4)


def test_rm_anova_two_way_within_and_mixed_against_pingouin():
    pg = pytest.importorskip("pingouin")
    import pandas as pd
    rng = np.random.default_rng(22)
    n = 10
    Y = rng.normal(size=(n, 1)) + rng.normal(size=(n, 6)) + np.array([0, .5, 1, .2, .9, 1.6])
    books = {"Book1": _book([Y[:, j] for j in range(6)])}
    rep, _ = st.run("rmanova2", books, {"design": "within", "ranges": [R(c) for c in "ABCDEF"],
                                        "levels_a": 2, "levels_b": 3, "name_a": "P",
                                        "name_b": "Q"})
    long = pd.DataFrame({"subj": np.repeat(np.arange(n), 6),
                         "a": np.tile(np.repeat([0, 1], 3), n),
                         "b": np.tile([0, 1, 2], 2 * n), "y": Y.ravel()})
    ref = pg.rm_anova(data=long, dv="y", within=["a", "b"], subject="subj")
    t = rep.table("Tests of Within-Subjects Effects")["rows"]
    for src, name in (("a", "P"), ("b", "Q"), ("a * b", "P * Q")):
        row = next(r for r in t if r[0] == name)
        assert float(row[5]) == pytest.approx(float(ref.loc[ref.Source == src, "F"].iloc[0]),
                                              rel=3e-5)
    # mixed design: within factor from the columns, between factor from a group column
    g = ["x"] * 5 + ["y"] * 5
    Ym = Y[:, :3] + np.array([[0.0]] * 5 + [[1.0]] * 5)
    books = {"Book1": _book([Ym[:, 0], Ym[:, 1], Ym[:, 2], g])}
    rep, _ = st.run("rmanova2", books, {"design": "mixed", "ranges": [R("A"), R("B"),
                                                                       R("C")],
                                        "group": R("D"), "name_a": "Time"})
    longm = pd.DataFrame({"subj": np.repeat(np.arange(n), 3), "t": np.tile([0, 1, 2], n),
                          "g": np.repeat(g, 3), "y": Ym.ravel()})
    ref = pg.mixed_anova(data=longm, dv="y", within="t", between="g", subject="subj")
    t = rep.table("Tests of Within-Subjects Effects")["rows"]
    assert float(next(r for r in t if r[0] == "Time")[5]) == pytest.approx(
        float(ref.loc[ref.Source == "t", "F"].iloc[0]), rel=3e-5)
    assert float(next(r for r in t if r[0] == "Time * D")[5]) == pytest.approx(
        float(ref.loc[ref.Source == "Interaction", "F"].iloc[0]), rel=3e-5)
    b = rep.table("Tests of Between-Subjects Effects")["rows"][0]
    assert float(b[4]) == pytest.approx(float(ref.loc[ref.Source == "g", "F"].iloc[0]),
                                        rel=3e-5)


def test_three_way_anova_and_manova():
    import pandas as pd
    import statsmodels.formula.api as smf
    from statsmodels.multivariate.manova import MANOVA
    rng = np.random.default_rng(23)
    a = np.repeat(["a1", "a2"], 24)
    b = np.tile(np.repeat(["b1", "b2", "b3"], 4), 4)
    c = np.tile(["c1", "c2"], 24)
    y = rng.normal(size=48) + (a == "a2") * 1.0 + (b == "b3") * 0.7
    books = {"Book1": _book([a, b, c, y])}
    rep, _ = st.run("anova3", books, {"data": R("D"), "factor_a": R("A"), "factor_b": R("B"),
                                      "factor_c": R("C")})
    ref = smf.ols("y ~ C(a) * C(b) * C(c)", data=pd.DataFrame(
        {"a": a, "b": b, "c": c, "y": y})).fit()
    assert _f(rep, "Overall ANOVA", "Error", "Sum of Squares") == pytest.approx(ref.ssr,
                                                                                rel=1e-5)
    assert rep.table("Tukey Test (A)") is not None
    y1 = rng.normal(size=48) + (b == "b2")
    y2 = rng.normal(size=48) + 0.5 * (b == "b3")
    books = {"Book1": _book([y1, y2, b])}
    rep, _ = st.run("manova", books, {"ranges": [R("A"), R("B")], "group": R("C")})
    mv = MANOVA.from_formula("y1 + y2 ~ g", data=pd.DataFrame({"y1": y1, "y2": y2, "g": b}))
    res = mv.mv_test().results["g"]["stat"]
    assert _f(rep, "Multivariate Tests", "Wilks' Lambda", "Value") == pytest.approx(
        res.loc["Wilks' lambda", "Value"], rel=1e-4)
    assert _f(rep, "Multivariate Tests", "Pillai's Trace", "F Value") == pytest.approx(
        res.loc["Pillai's trace", "F Value"], rel=1e-4)


# ====================================================================== nonparametric etc.
def test_nonparametric_additions_against_scipy():
    rng = np.random.default_rng(31)
    a = rng.normal(0.4, 1, 25)
    b = rng.normal(0, 1, 25)
    g = ["p"] * 12 + ["q"] * 13
    books = {"Book1": _book([a, b, g])}
    rep, _ = st.run("wilcoxon1", books, {"ranges": [R("A")], "median": 0.0})
    assert _f(rep, "Test Statistics", "A", "Prob (exact when possible)") == pytest.approx(
        stats.wilcoxon(a).pvalue, rel=1e-4)
    rep, _ = st.run("sign", books, {"range1": R("A"), "range2": R("B")})
    k = int(np.sum(a > b))
    assert _f(rep, "Test Statistics", "A − B", "Exact Prob") == pytest.approx(
        stats.binomtest(k, 25).pvalue, rel=1e-4)
    rep, _ = st.run("ks2", books, {"form": "indexed", "data": R("A"), "factor": R("C")})
    ref = stats.ks_2samp(a[:12], a[12:], method="exact")
    assert _f(rep, "Test Statistics", "p − q", "D") == pytest.approx(ref.statistic, rel=1e-5)
    assert _f(rep, "Test Statistics", "p − q", "Exact Prob") == pytest.approx(ref.pvalue,
                                                                             rel=1e-4)
    rep, _ = st.run("median", books, {"ranges": [R("A"), R("B")]})
    ref = stats.median_test(a, b, ties="below", correction=False)
    assert _f(rep, "Test Statistics", "Chi-Square", "Value") == pytest.approx(ref.statistic,
                                                                             rel=3e-5)


def test_partial_correlation_and_crosstab():
    pg = pytest.importorskip("pingouin")
    import pandas as pd
    rng = np.random.default_rng(41)
    z = rng.normal(size=60)
    x = z + rng.normal(size=60)
    y = z + rng.normal(size=60) + 0.3 * x
    books = {"Book1": _book([x, y, z])}
    rep, _ = st.run("partialcorr", books, {"ranges": [R("A"), R("B")],
                                           "covariates": [R("C")]})
    ref = pg.partial_corr(data=pd.DataFrame({"x": x, "y": y, "z": z}), x="x", y="y",
                          covar="z")
    assert _f(rep, "Partial Correlation Coefficients", "A", "B") == pytest.approx(
        float(ref["r"].iloc[0]), rel=1e-4)
    assert float(_cell(rep, "Partial Correlation P-values", "A", "B")) == pytest.approx(
        float(ref["p_val"].iloc[0] if "p_val" in ref else ref["p-val"].iloc[0]), rel=1e-3)
    r1 = list("aabbbccaab" * 3)
    r2 = list("xyxyyxxyyy" * 3)
    books = {"Book1": _book([r1, r2])}
    rep, _ = st.run("crosstab", books, {"range1": R("A"), "range2": R("B")})
    tab = pd.crosstab(pd.Series(r1), pd.Series(r2)).to_numpy()
    chi, p, _, _ = stats.chi2_contingency(tab, correction=False)
    assert _f(rep, "Chi-Square Tests", "Pearson Chi-Square", "Value") == pytest.approx(chi,
                                                                                      rel=1e-4)


def test_masked_rows_are_excluded(monkeypatch):
    v = [1.0, 2.0, 3.0, 100.0, 4.0]
    book = _book([v])
    ws = book.sheets[0]
    monkeypatch.setattr(ws, "masked_rows", lambda col: {3} if col == 0 else set(),
                        raising=False)
    rep, _ = st.run("desc", {"Book1": book}, {"ranges": [R("A")]})
    assert _f(rep, "Descriptive Statistics", "A", "Mean") == pytest.approx(2.5)


def test_extensions_register_dialog_specs_and_ranges():
    st.load_extensions()
    keys = ["grubbs", "dixon", "distfit", "partialcorr", "crosstab", "var1", "var2", "vark",
            "prop1", "prop2", "anova3", "rmanova1", "rmanova2", "manova", "wilcoxon1", "sign",
            "ks2", "median", "pca", "kmeans", "hca", "lda", "pls", "km", "cox", "weibull",
            "roc", "power"]
    for k in keys:
        assert k in st.REPORT_TESTS and k in st.DIALOG_SPECS
        title, groups, desc, out = st.DIALOG_SPECS[k](["[Book1]Sheet1!A", "[Book1]Sheet1!B"])
        assert title and groups
    srcs = st.range_strings({"time": "[B]S!A", "censor": "[B]S!B", "response": ["[B]S!C"],
                             "covariates": ["[B]S!D"], "book": "B"})
    assert srcs == ["[B]S!A", "[B]S!B", "[B]S!C", "[B]S!D"]
