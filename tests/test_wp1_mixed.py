"""WP1: Linear Mixed Models repeated covariance structures, EM means, Save and custom fixed
models; Variance Components (ANOVA, MINQUE, ML, REML)."""
import math

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from scipy import optimize, stats

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.stats import mixed, varcomp
from rsstatchem.core.stats.glm_model import TermSpec


def num(text: str) -> float:
    return float(text.rstrip("*ᵃᵇᶜᵈ").replace(",", ""))


def table(section, title):
    return next(i for i in section.items if getattr(i, "title", "").startswith(title))


@pytest.fixture
def long_data():
    rng = np.random.default_rng(11)
    ns, nt = 30, 4
    sid = np.repeat(np.arange(ns), nt).astype(float)
    tm = np.tile(np.arange(1, nt + 1), ns).astype(float)
    grp = np.repeat(np.where(np.arange(ns) < 15, 1.0, 2.0), nt)
    e = np.zeros(ns * nt)
    for s in range(ns):  # AR(1) errors within subject
        z = rng.normal(0, 1, nt)
        for j in range(nt):
            e[s * nt + j] = z[j] if j == 0 else 0.6 * e[s * nt + j - 1] + 0.8 * z[j]
    y = 5 + 0.7 * tm + 0.5 * grp + rng.normal(0, 0.8, ns)[sid.astype(int)] * 0 + e
    return pd.DataFrame({"id": sid, "time": tm, "grp": grp, "y": y})


def test_cs_equals_random_intercept(long_data):
    ds = Dataset(long_data)
    ri = mixed.linear_mixed(ds, "y", "id", ["grp", "time"])
    cs = mixed.linear_mixed(ds, "y", "id", ["grp", "time"], random_intercept=False,
                            repeated="time", repeated_type="CS")
    a = {r[0]: r for r in table(ri, "Type III Tests").rows}
    b = {r[0]: r for r in table(cs, "Type III Tests").rows}
    for k in ("grp", "time", "grp * time"):
        assert num(a[k][3]) == pytest.approx(num(b[k][3]), abs=2e-3)
    cov_ri = table(ri, "Estimates of Covariance Parameters").rows
    cov_cs = table(cs, "Estimates of Covariance Parameters").rows
    assert num(cov_cs[0][2]) == pytest.approx(num(cov_ri[0][2]), abs=2e-3)  # offset = s2
    assert num(cov_cs[1][2]) == pytest.approx(num(cov_ri[1][2]), abs=2e-3)  # cov = var(u)
    assert cov_cs[1][1] == "CS covariance"
    md = table(cs, "Model Dimension")
    assert any(r[0] == "Repeated Effects" and r[3] == "Compound Symmetry" for r in md.rows)


def test_un_and_diag_closed_form(long_data):
    """Saturated time means, one group: REML UN = sample covariance (divisor n - 1)."""
    df = long_data
    ds = Dataset(df)
    wide = df.pivot(index="id", columns="time", values="y").to_numpy()
    S = np.cov(wide, rowvar=False)
    sec = mixed.linear_mixed(ds, "y", "id", ["time"], random_intercept=False, repeated="time",
                             repeated_type="UN")
    rows = table(sec, "Estimates of Covariance Parameters").rows
    vals = {r[1]: num(r[2]) for r in rows}
    assert vals["UN (1,1)"] == pytest.approx(S[0, 0], abs=2e-3)
    assert vals["UN (3,2)"] == pytest.approx(S[2, 1], abs=2e-3)
    assert vals["UN (4,4)"] == pytest.approx(S[3, 3], abs=2e-3)
    sec = mixed.linear_mixed(ds, "y", "id", ["time"], random_intercept=False, repeated="time",
                             repeated_type="DIAG")
    rows = table(sec, "Estimates of Covariance Parameters").rows
    np.testing.assert_allclose([num(r[2]) for r in rows], np.diag(S), atol=2e-3)


def test_ar1_against_direct_ml(long_data):
    """ML AR(1) fit equals a brute-force maximisation of the multivariate normal likelihood."""
    df = long_data
    sec = mixed.linear_mixed(Dataset(df), "y", "id", ["grp"], ["time"], random_intercept=False,
                             repeated="time", repeated_type="AR1", method="ML")
    rows = {r[1]: num(r[2]) for r in table(sec, "Estimates of Covariance Parameters").rows}
    y = df.y.to_numpy().reshape(-1, 4)
    X = np.column_stack([np.ones(len(df)), (df.grp == 1).astype(float), df.time]) \
        .reshape(-1, 4, 3)
    lag = np.abs(np.subtract.outer(np.arange(4), np.arange(4)))

    def nll(par):
        b, s2, rho = par[:3], math.exp(par[3]), math.tanh(par[4])
        R = s2 * rho ** lag
        return -sum(stats.multivariate_normal.logpdf(y[i], X[i] @ b, R) for i in range(len(y)))

    res = optimize.minimize(nll, [5, 0, 0.7, 0, 0.5], method="Nelder-Mead",
                            options={"maxiter": 20000, "xatol": 1e-9, "fatol": 1e-10})
    assert rows["AR1 diagonal"] == pytest.approx(math.exp(res.x[3]), abs=3e-3)
    assert rows["AR1 rho"] == pytest.approx(math.tanh(res.x[4]), abs=3e-3)
    ic = table(sec, "Information Criteria").rows
    assert num(ic[0][1]) == pytest.approx(2 * res.fun, abs=5e-3)


def test_emm_and_save(long_data):
    df = long_data
    res = mixed.linear_mixed(Dataset(df), "y", "id", ["grp", "time"], emm=["time"],
                             compare=True, adjust="bonferroni",
                             save=["FIXPRED", "PRED", "RESID"], icc=True)
    est = table(res.section, "Estimates — time")
    means = df.groupby("time").y.mean().to_numpy()
    for i in range(4):
        assert num(est.rows[i][1]) == pytest.approx(means[i], abs=2e-3)
    assert table(res.section, "Pairwise Comparisons — time")
    assert table(res.section, "Intraclass Correlation")
    ref = smf.mixedlm("y ~ C(grp) * C(time)", df, groups=df["id"]).fit(reml=True)
    np.testing.assert_allclose(res.saved["PRED"][1], ref.fittedvalues, atol=2e-3)
    np.testing.assert_allclose(res.saved["FIXPRED"][1],
                               ref.model.exog @ ref.fe_params, atol=2e-3)


def test_custom_fixed_terms(long_data):
    df = long_data
    sec = mixed.linear_mixed(Dataset(df), "y", "id", ["grp"], ["time"],
                             fixed_terms=[TermSpec(("grp",)), TermSpec(covariates=("time",)),
                                          TermSpec(("grp",), ("time",))])
    t3 = {r[0]: r for r in table(sec, "Type III Tests").rows}
    assert "grp * time" in t3
    ref = smf.mixedlm("y ~ C(grp, Sum) * time", df, groups=df["id"]).fit(reml=True)
    est = {r[0]: r for r in table(sec, "Estimates of Fixed Effects").rows}
    assert num(est["time"][1]) == pytest.approx(
        ref.fe_params["time"] - ref.fe_params["C(grp, Sum)[S.1.0]:time"], abs=2e-3)


# ------------------------------------------------------------------ variance components
@pytest.fixture
def oneway_random():
    rng = np.random.default_rng(8)
    k, n = 8, 5
    g = np.repeat(np.arange(1., k + 1), n)
    y = 10 + rng.normal(0, 1.5, k)[g.astype(int) - 1] + rng.normal(0, 1, k * n)
    return pd.DataFrame({"y": y, "batch": g})


def test_varcomp_balanced_methods_agree(oneway_random):
    df = oneway_random
    ds = Dataset(df)
    g = df.groupby("batch").y
    msb = 5 * ((g.mean() - df.y.mean()) ** 2).sum() / 7
    msw = ((df.y - g.transform("mean")) ** 2).sum() / 32
    s_a = (msb - msw) / 5
    for method in ("anova", "minque1", "reml"):
        sec = varcomp.variance_components(ds, "y", [], ["batch"], method=method)
        est = table(sec, "Variance Estimates").rows
        assert num(est[0][1]) == pytest.approx(s_a, abs=2e-3), method
        assert num(est[1][1]) == pytest.approx(msw, abs=2e-3), method
    sec = varcomp.variance_components(ds, "y", [], ["batch"], method="ml")
    ref = smf.mixedlm("y ~ 1", df, groups=df["batch"]).fit(reml=False)
    est = table(sec, "Variance Estimates").rows
    assert num(est[0][1]) == pytest.approx(float(ref.cov_re.iloc[0, 0]), abs=3e-3)
    assert num(est[1][1]) == pytest.approx(ref.scale, abs=3e-3)
    sec = varcomp.variance_components(ds, "y", [], ["batch"], method="minque0")
    assert num(table(sec, "Variance Estimates").rows[1][1]) > 0


def test_varcomp_two_way_anova(oneway_random):
    rng = np.random.default_rng(1)
    a = np.repeat([1., 2., 3.], 12)
    b = np.tile(np.repeat([1., 2., 3., 4.], 3), 3)
    y = a + rng.normal(0, 1, 4)[b.astype(int) - 1] + rng.normal(0, 1, 36)
    df = pd.DataFrame({"y": y, "a": a, "b": b})
    sec = varcomp.variance_components(Dataset(df), "y", ["a"], ["b"], method="anova")
    est = {r[0]: num(r[1]) for r in table(sec, "Variance Estimates").rows}
    import statsmodels.api as sm

    aov = sm.stats.anova_lm(smf.ols("y ~ C(a) * C(b)", df).fit())
    ms = aov.sum_sq / aov.df
    assert est["Var(Error)"] == pytest.approx(ms["Residual"], abs=2e-3)
    assert est["Var(a * b)"] == pytest.approx((ms["C(a):C(b)"] - ms["Residual"]) / 3, abs=2e-3)
    assert est["Var(b)"] == pytest.approx((ms["C(b)"] - ms["C(a):C(b)"]) / 9, abs=2e-3)
    assert table(sec, "Expected Mean Squares")
