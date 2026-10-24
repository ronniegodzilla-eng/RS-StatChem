"""Binary/multinomial/ordinal logistic regression, Curve Estimation, ROC and Means checked
against statsmodels / scipy reference implementations and hand-computed SPSS formulas."""
import math

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import special, stats
from statsmodels.miscmodels.ordinal_model import OrderedModel

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.stats import curvefit_spss, logistic, means, roc


def num(text: str) -> float:
    return float(text.rstrip("*ᵃᵇᶜᵈᵉ").replace(",", ""))


def table(section, title, nth=0):
    found = [i for i in section.items if getattr(i, "title", "").startswith(title)]
    return found[nth]


def row(tab, *heads):
    for r in tab.rows:
        if all(h in r for h in heads):
            return r
    raise AssertionError(f"row {heads} not found in {tab.title}")


@pytest.fixture
def data():
    rng = np.random.default_rng(11)
    n = 240
    x1, x2, x3 = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    g = rng.integers(1, 4, n).astype(float)
    eta = -0.2 + 0.9 * x1 - 0.6 * x2 + 0.7 * (g == 2) - 0.5 * (g == 3)
    y = (rng.random(n) < special.expit(eta)).astype(float)
    o = np.digitize(eta + rng.logistic(size=n), [-1.2, 0.0, 1.1]) + 1.0
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3, "g": g, "o": o})
    ds = Dataset(df)
    ds.var("g").value_labels = {1.0: "A", 2.0: "B", 3.0: "C"}
    return ds, df


# ------------------------------------------------------------------ binary logistic
def test_binary_enter_matches_statsmodels(data):
    ds, df = data
    res = logistic.binary_logistic(ds, "y", ["x1", "x2", "g"], {"g": ("indicator", "last")},
                                   ci=True)
    X = pd.DataFrame({"x1": df.x1, "x2": df.x2, "g1": (df.g == 1) * 1.0,
                      "g2": (df.g == 2) * 1.0})
    ref = sm.Logit(df.y, sm.add_constant(X)).fit(disp=0)
    vt = table(res.section, "Variables in the Equation", 1)
    assert num(row(vt, "x1")[2]) == pytest.approx(ref.params["x1"], abs=6e-4)
    assert num(row(vt, "x1")[3]) == pytest.approx(ref.bse["x1"], abs=6e-4)
    assert num(row(vt, "g(1)")[2]) == pytest.approx(ref.params["g1"], abs=6e-4)
    assert num(row(vt, "g(2)")[2]) == pytest.approx(ref.params["g2"], abs=6e-4)
    assert num(row(vt, "Constant")[2]) == pytest.approx(ref.params["const"], abs=6e-4)
    wald_x1 = (ref.params["x1"] / ref.bse["x1"]) ** 2
    assert num(row(vt, "x1")[4]) == pytest.approx(wald_x1, abs=2e-3)
    lo = math.exp(ref.params["x1"] - 1.959964 * ref.bse["x1"])
    assert num(row(vt, "x1")[8]) == pytest.approx(lo, abs=2e-3)
    # overall Wald for the 2-df categorical effect
    cov = ref.cov_params().loc[["g1", "g2"], ["g1", "g2"]].to_numpy()
    b = ref.params[["g1", "g2"]].to_numpy()
    assert num(row(vt, "g")[4]) == pytest.approx(b @ np.linalg.solve(cov, b), abs=2e-3)

    ms = table(res.section, "Model Summary")
    n = len(df)
    ll0 = ref.llnull
    cs = 1 - math.exp(2 * (ll0 - ref.llf) / n)
    nk = cs / (1 - math.exp(2 * ll0 / n))
    assert num(ms.rows[0][1]) == pytest.approx(-2 * ref.llf, abs=1e-3)
    assert num(ms.rows[0][2]) == pytest.approx(cs, abs=6e-4)
    assert num(ms.rows[0][3]) == pytest.approx(nk, abs=6e-4)
    om = table(res.section, "Omnibus Tests")
    assert num(row(om, "Model")[2]) == pytest.approx(ref.llr, abs=1e-3)
    assert num(row(om, "Model")[3]) == 4


def test_block0_score_tests(data):
    ds, df = data
    res = logistic.binary_logistic(ds, "y", ["x1", "x2"])
    nt = table(res.section, "Variables not in the Equation")
    null = sm.GLM(df.y, np.ones((len(df), 1)), family=sm.families.Binomial()).fit()
    s1 = null.score_test(exog_extra=df[["x1"]].to_numpy())[0]
    both = null.score_test(exog_extra=df[["x1", "x2"]].to_numpy())[0]
    assert num(row(nt, "x1")[3]) == pytest.approx(float(np.ravel(s1)[0]), abs=2e-3)
    overall = float(np.ravel(both)[0])
    assert num(row(nt, "Overall Statistics")[3]) == pytest.approx(overall, abs=2e-3)
    v0 = table(res.section, "Variables in the Equation")
    p = df.y.mean()
    assert num(v0.rows[0][2]) == pytest.approx(math.log(p / (1 - p)), abs=6e-4)


def test_deviation_coding_same_fit_and_codings(data):
    ds, _ = data
    ind = logistic.binary_logistic(ds, "y", ["x1", "g"], {"g": ("indicator", "first")})
    dev = logistic.binary_logistic(ds, "y", ["x1", "g"], {"g": ("deviation", "last")})
    m1 = table(ind.section, "Model Summary").rows[0][1]
    m2 = table(dev.section, "Model Summary").rows[0][1]
    assert m1 == m2
    cod = table(dev.section, "Categorical Variables Codings")
    assert row(cod, "C")[3:] == ["-1.000", "-1.000"]
    cod = table(ind.section, "Categorical Variables Codings")
    assert row(cod, "A")[3:] == [".000", ".000"]
    assert row(cod, "B")[3:] == ["1.000", ".000"]


def test_hosmer_lemeshow_hand_computed():
    rng = np.random.default_rng(3)
    p = np.sort(rng.random(100))
    y = (rng.random(100) < p).astype(float)
    chi, df, sig, rows = logistic.hosmer_lemeshow(y, p)
    assert df == 8 and len(rows) == 10 and all(r[0] == 10 for r in rows)
    ref = 0.0
    for k in range(10):
        yy, pp = y[10 * k:10 * k + 10], p[10 * k:10 * k + 10]
        o1, e1 = yy.sum(), pp.sum()
        ref += (o1 - e1) ** 2 / e1 + ((10 - o1) - (10 - e1)) ** 2 / (10 - e1)
    assert chi == pytest.approx(ref)
    assert sig == pytest.approx(stats.chi2.sf(ref, 8))


def test_forward_lr_selects_true_predictors(data):
    ds, df = data
    res = logistic.binary_logistic(ds, "y", ["x3", "x1", "x2", "g"], method="fstep_lr")
    vt = table(res.section, "Variables in the Equation", 1)
    last_step = [r for r in vt.rows if r[0].startswith("Step")][-1][0]
    final = vt.rows[[r[0] for r in vt.rows].index(last_step):]
    names = {r[1] for r in final}
    assert {"x1", "x2"} <= names and "x3" not in names
    mt = table(res.section, "Model if Term Removed")
    # change in -2LL for x1 at the last step equals an LR test computed with statsmodels
    terms = [n for n in ("x1", "x2", "g") if n in names]
    X = pd.concat([df[[t]] if t != "g" else pd.DataFrame(
        {"g1": (df.g == 1) * 1.0, "g2": (df.g == 2) * 1.0}) for t in terms], axis=1)
    full = sm.Logit(df.y, sm.add_constant(X)).fit(disp=0)
    red = sm.Logit(df.y, sm.add_constant(X.drop(columns="x1"))).fit(disp=0)
    x1_rows = [r for r in mt.rows if r[1] == "x1"]
    assert num(x1_rows[-1][3]) == pytest.approx(2 * (full.llf - red.llf), abs=2e-3)


def test_backward_wald_and_save(data):
    ds, df = data
    res = logistic.binary_logistic(ds, "y", ["x1", "x2", "x3"], method="bstep_wald",
                                   save=("prob", "group"))
    assert "PRE" in res.saved and "PGR" in res.saved
    vt = table(res.section, "Variables in the Equation", 1)
    last = [i for i, r in enumerate(vt.rows) if r[0].startswith("Step")][-1]
    names = {r[1] for r in vt.rows[last:]}
    assert "x3" not in names
    ref = sm.Logit(df.y, sm.add_constant(df[["x1", "x2"]])).fit(disp=0)
    assert np.allclose(res.saved["PRE"][1].to_numpy(), ref.predict(), atol=1e-6)
    assert set(np.unique(res.saved["PGR"][1])) <= {0.0, 1.0}


def test_classification_table_counts(data):
    ds, df = data
    res = logistic.binary_logistic(ds, "y", ["x1"], cut=0.4)
    ref = sm.Logit(df.y, sm.add_constant(df[["x1"]])).fit(disp=0)
    pred = ref.predict() > 0.4
    ct = table(res.section, "Classification Table", 1)
    assert int(ct.rows[0][3]) == int(((df.y == 0) & ~pred).sum())
    assert int(ct.rows[1][4]) == int(((df.y == 1) & pred).sum())
    assert ct.footnotes[-1] == "a. The cut value is .400"


# ------------------------------------------------------------------ multinomial
def test_multinomial_matches_mnlogit(data):
    ds, df = data
    res = logistic.multinomial(ds, "o", ["g"], ["x1", "x2"], reference="last")
    X = sm.add_constant(pd.DataFrame({"x1": df.x1, "x2": df.x2, "g1": (df.g == 1) * 1.0,
                                      "g2": (df.g == 2) * 1.0}))
    yref = df.o.map({4.0: 0, 1.0: 1, 2.0: 2, 3.0: 3})
    ref = sm.MNLogit(yref, X).fit(disp=0, method="newton")
    pe = table(res.section, "Parameter Estimates")
    from types import SimpleNamespace

    first = SimpleNamespace(rows=pe.rows[:6], title="category 1 block")
    assert num(row(first, "x1")[2]) == pytest.approx(ref.params.iloc[1, 0], abs=6e-4)
    assert num(row(first, "x1")[3]) == pytest.approx(ref.bse.iloc[1, 0], abs=6e-4)
    assert num(row(first, "Intercept")[2]) == pytest.approx(ref.params.iloc[0, 0], abs=6e-4)
    assert row(first, "[g=3]")[2] == "0ᵇ"
    mfi = table(res.section, "Model Fitting Information")
    assert num(mfi.rows[1][2]) == pytest.approx(2 * (ref.llf - ref.llnull), abs=2e-3)
    assert num(mfi.rows[1][3]) == 12
    ps = table(res.section, "Pseudo R-Square")
    assert num(ps.rows[2][1]) == pytest.approx(1 - ref.llf / ref.llnull, abs=6e-4)
    # LR test of x2: drop it and refit
    red = sm.MNLogit(yref, X.drop(columns="x2")).fit(disp=0, method="newton")
    lrt = table(res.section, "Likelihood Ratio Tests")
    assert num(row(lrt, "x2")[2]) == pytest.approx(2 * (ref.llf - red.llf), abs=2e-3)
    assert pe.footnotes[0] == "a. The reference category is: 4."
    # classification table agrees with argmax of the predicted probabilities
    cls = table(res.section, "Classification")
    pred = np.asarray(ref.predict()).argmax(axis=1)
    overall = 100 * np.mean(pred == yref.to_numpy())
    assert num(cls.rows[-1][-1].rstrip("%")) == pytest.approx(overall, abs=0.06)


def test_multinomial_first_reference_and_full_loglik():
    # Grouped data: the full -2LL includes the multinomial constant over subpopulations.
    df = pd.DataFrame({"y": [1, 1, 2, 2, 3, 1, 2, 3, 3, 3, 1, 2] * 5,
                       "f": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2] * 5}, dtype=float)
    ds = Dataset(df)
    res = logistic.multinomial(ds, "y", ["f"], [], reference="first")
    pe = table(res.section, "Parameter Estimates")
    assert pe.footnotes[0] == "a. The reference category is: 1."
    mfi = table(res.section, "Model Fitting Information")
    counts = df.groupby(["f", "y"]).size().unstack(fill_value=0).to_numpy()
    p = counts / counts.sum(axis=1, keepdims=True)
    kernel = float(np.sum(counts * np.log(np.where(p > 0, p, 1))))
    const = float(np.sum(special.gammaln(counts.sum(axis=1) + 1))
                  - np.sum(special.gammaln(counts + 1)))
    assert num(mfi.rows[1][1]) == pytest.approx(-2 * (kernel + const), abs=2e-3)
    gof = table(res.section, "Goodness-of-Fit")
    assert num(gof.rows[1][1]) == pytest.approx(0.0, abs=2e-3)  # saturated model


# ------------------------------------------------------------------ ordinal
def test_ordinal_matches_orderedmodel(data):
    ds, df = data
    res = logistic.ordinal(ds, "o", ["g"], ["x1", "x2"], link="logit", parallel=True)
    X = pd.DataFrame({"x1": df.x1, "x2": df.x2, "g1": (df.g == 1) * 1.0,
                      "g2": (df.g == 2) * 1.0})
    ref = OrderedModel(df.o.astype(int), X, distr="logit").fit(method="bfgs", disp=0,
                                                              maxiter=5000)
    pe = table(res.section, "Parameter Estimates")
    assert num(row(pe, "x1")[2]) == pytest.approx(ref.params["x1"], abs=1e-3)
    assert num(row(pe, "x1")[3]) == pytest.approx(ref.bse["x1"], abs=2e-3)
    th = ref.model.transform_threshold_params(ref.params)[1:-1]
    assert num(row(pe, "[o = 1]")[2]) == pytest.approx(th[0], abs=1e-3)
    assert num(row(pe, "[o = 3]")[2]) == pytest.approx(th[2], abs=1e-3)
    assert row(pe, "[g=3]")[2] == "0ᵃ"
    counts = df.o.value_counts().to_numpy()
    ll0 = float(np.sum(counts * np.log(counts / counts.sum())))
    mfi = table(res.section, "Model Fitting Information")
    assert num(mfi.rows[1][2]) == pytest.approx(2 * (ref.llf - ll0), abs=2e-3)
    ps = table(res.section, "Pseudo R-Square")
    assert num(ps.rows[2][1]) == pytest.approx(1 - ref.llf / ll0, abs=6e-4)
    tp = table(res.section, "Test of Parallel Lines")
    assert num(tp.rows[1][2]) >= 0 and tp.rows[1][3] == "8"


@pytest.mark.parametrize("link,distr", [("probit", "probit"), ("cloglog", stats.gumbel_l),
                                        ("cauchit", stats.cauchy)])
def test_ordinal_links(data, link, distr):
    ds, df = data
    res = logistic.ordinal(ds, "o", [], ["x1"], link=link)
    ref = OrderedModel(df.o.astype(int), df[["x1"]], distr=distr).fit(method="bfgs", disp=0,
                                                                      maxiter=5000)
    pe = table(res.section, "Parameter Estimates")
    assert num(row(pe, "x1")[2]) == pytest.approx(ref.params["x1"], abs=2e-3)


# ------------------------------------------------------------------ curve estimation
def test_curve_estimation_models():
    rng = np.random.default_rng(5)
    t = np.linspace(1, 10, 40)
    y = 2.5 * 1.3 ** t * np.exp(rng.normal(0, 0.05, 40))
    ds = Dataset(pd.DataFrame({"t": t, "y": y}))
    sec = curvefit_spss.curve_estimation(ds, ["y"], "t", tuple(curvefit_spss.MODELS))
    tab = table(sec, "Model Summary and Parameter Estimates")
    lin = stats.linregress(t, y)
    assert num(row(tab, "Linear")[1]) == pytest.approx(lin.rvalue ** 2, abs=6e-4)
    assert num(row(tab, "Linear")[6]) == pytest.approx(lin.intercept, abs=6e-4)
    lg = stats.linregress(t, np.log(y))
    comp = row(tab, "Compound")
    assert num(comp[1]) == pytest.approx(lg.rvalue ** 2, abs=6e-4)
    assert num(comp[6]) == pytest.approx(math.exp(lg.intercept), abs=6e-3)
    assert num(comp[7]) == pytest.approx(math.exp(lg.slope), abs=6e-4)
    assert num(row(tab, "Growth")[7]) == pytest.approx(lg.slope, abs=6e-4)
    assert num(row(tab, "Exponential")[6]) == pytest.approx(math.exp(lg.intercept), abs=6e-3)
    pw = stats.linregress(np.log(t), np.log(y))
    assert num(row(tab, "Power")[7]) == pytest.approx(pw.slope, abs=6e-4)
    q = np.polyfit(t, y, 2)
    assert num(row(tab, "Quadratic")[8]) == pytest.approx(q[0], abs=6e-4)
    lgs = row(tab, "Logistic")  # u = infinity: ln(1/y) = ln b0 + t ln b1
    assert num(lgs[7]) == pytest.approx(math.exp(-lg.slope), abs=6e-4)
    f = lin.rvalue ** 2 / (1 - lin.rvalue ** 2) * 38
    assert num(row(tab, "Linear")[2]) == pytest.approx(f, rel=1e-3)
    assert any(getattr(i, "kind", "") == "image" for i in sec.items)


def test_curve_estimation_time_and_no_constant():
    y = np.array([3.0, 5.1, 6.9, 9.2, 11.0, 12.8])
    ds = Dataset(pd.DataFrame({"y": y}))
    sec = curvefit_spss.curve_estimation(ds, ["y"], None, ("linear",), constant=False)
    tab = table(sec, "Model Summary and Parameter Estimates")
    t = np.arange(1, 7.0)
    b = float(t @ y / (t @ t))
    r2 = 1 - np.sum((y - b * t) ** 2) / np.sum(y ** 2)
    assert num(tab.rows[0][7]) == pytest.approx(b, abs=6e-4)
    assert num(tab.rows[0][1]) == pytest.approx(r2, abs=6e-4)
    assert tab.footnotes[1] == "The independent variable is Sequence."


# ------------------------------------------------------------------ ROC
def test_roc_auc_equals_mann_whitney():
    rng = np.random.default_rng(9)
    pos = np.round(rng.normal(1, 1, 40), 1)
    neg = np.round(rng.normal(0, 1, 55), 1)
    ds = Dataset(pd.DataFrame({"x": np.r_[pos, neg], "s": np.r_[np.ones(40), np.zeros(55)]}))
    sec = roc.roc(ds, ["x"], "s", 1.0, coordinates=True)
    auc = table(sec, "Area Under the Curve").rows[0]
    u = stats.mannwhitneyu(pos, neg).statistic
    assert num(auc[1]) == pytest.approx(u / (40 * 55), abs=6e-4)
    d = roc.auc_details(pos, neg)
    # DeLong variance from the structural components
    v10 = np.array([np.mean((p > neg) + 0.5 * (p == neg)) for p in pos])
    v01 = np.array([np.mean((pos > q) + 0.5 * (pos == q)) for q in neg])
    se = math.sqrt(v10.var(ddof=1) / 40 + v01.var(ddof=1) / 55)
    assert d["se_delong"] == pytest.approx(se)
    assert num(auc[2]) == pytest.approx(se, abs=6e-4)
    a = d["auc"]
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    hm = math.sqrt((a * (1 - a) + 39 * (q1 - a * a) + 54 * (q2 - a * a)) / (40 * 55))
    assert d["se_hm"] == pytest.approx(hm)
    p_mw = stats.mannwhitneyu(pos, neg, use_continuity=False, method="asymptotic").pvalue
    assert num(auc[3]) == pytest.approx(p_mw, abs=0.01)
    coords = table(sec, "Coordinates of the Curve")
    assert num(coords.rows[0][1]) == pytest.approx(min(pos.min(), neg.min()) - 1)
    assert coords.rows[0][2:] == ["1.000", "1.000"]
    assert coords.rows[-1][2:] == [".000", ".000"]


def test_roc_smaller_positive_mirrors():
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8.0])
    s = np.array([1, 1, 1, 0, 1, 0, 0, 0.0])
    ds = Dataset(pd.DataFrame({"x": x, "s": s}))
    small = table(roc.roc(ds, ["x"], "s", 1.0, larger_positive=False), "Area Under").rows[0]
    large = table(roc.roc(ds, ["x"], "s", 0.0), "Area Under").rows[0]
    assert small[1] == large[1] == ".938"


# ------------------------------------------------------------------ means
def test_means_report_and_anova(data):
    ds, df = data
    sec = means.means(ds, ["x1"], [["g"]], ("mean", "count", "stddev", "median", "semean"),
                      anova=True, linearity=True)
    rep = table(sec, "Report")
    grp = df.groupby("g")["x1"]
    assert num(row(rep, "B")[1]) == pytest.approx(grp.mean()[2.0], abs=6e-4)
    assert int(row(rep, "C")[2]) == grp.size()[3.0]
    assert num(row(rep, "A")[3]) == pytest.approx(grp.std()[1.0], abs=6e-5)
    assert num(row(rep, "Total")[4]) == pytest.approx(df.x1.median(), abs=6e-4)
    f, p = stats.f_oneway(*[v.to_numpy() for _, v in grp])
    an = table(sec, "ANOVA Table")
    assert num(row(an, "(Combined)")[6]) == pytest.approx(f, abs=2e-3)
    ss_b = sum(len(v) * (v.mean() - df.x1.mean()) ** 2 for _, v in grp)
    eta2 = ss_b / ((df.x1 - df.x1.mean()) ** 2).sum()
    ma = table(sec, "Measures of Association")
    assert num(ma.rows[0][4]) == pytest.approx(eta2, abs=6e-4)
    r = np.corrcoef(df.g, df.x1)[0, 1]
    assert num(ma.rows[0][1]) == pytest.approx(r, abs=6e-4)


def test_means_two_layers_totals(data):
    ds, df = data
    sec = means.means(ds, ["x1", "x2"], [["g"], ["y"]])
    rep = table(sec, "Report")
    assert rep.columns[-2:] == ["x1", "x2"]
    sub = df[(df.g == 3.0) & (df.y == 1.0)]
    idx = next(i for i, r in enumerate(rep.rows) if r[0] == "C")
    block = rep.rows[idx:idx + 9]
    mean_row = next(r for r in block if r[1] == "1" and r[2] == "Mean")
    assert num(mean_row[3]) == pytest.approx(sub.x1.mean(), abs=6e-4)
    grand = [r for r in rep.rows if r[0] == "Total"]
    assert grand and grand[0][1] == "0"  # grand total also broken down by layer 2


def test_grouped_median():
    x = np.array([1, 2, 2, 3, 3, 3, 4, 4, 5.0])
    # class 3 spans 2.5-3.5 with 3 cases, 3 cases below: 2.5 + (4.5 - 3)/3
    assert means.grouped_median(x) == pytest.approx(3.0)
    assert means.grouped_median(np.array([1, 1, 2, 2, 2, 3.0])) == pytest.approx(
        1.5 + (3 - 2) / 3)


def test_selftest_runs():
    from rsstatchem.ui.dialogs import logistic as dlg

    dlg.selftest()
    assert "Analyze/Classify" in dlg.MENUS and "Analyze/Regression" in dlg.MENUS
