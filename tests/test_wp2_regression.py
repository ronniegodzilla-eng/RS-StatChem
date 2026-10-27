"""WP2: Linear Regression (blocks, methods, diagnostics, Save) against statsmodels."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.stats import regression


def num(text: str) -> float:
    text = text.rstrip("ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖʳˢᵗᵘᵛʷˣʸᶻ*")
    return float(text)


def table(sec, title):
    return next(t for t in sec.items if getattr(t, "title", "").startswith(title))


def row(t, label, col=1):
    return next(r for r in t.rows if r[col] == label)


@pytest.fixture
def data():
    rng = np.random.default_rng(11)
    n = 80
    x1, x2, x3 = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    m = rng.normal(size=n)
    y = 2 + 1.2 * x1 - 0.7 * x2 + 0.5 * x1 * m + 0.3 * m + rng.normal(size=n)
    w = rng.uniform(0.5, 2.0, n)
    g = rng.integers(1, 3, n).astype(float)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3, "m": m, "x1m": x1 * m, "w": w,
                       "g": g})
    return Dataset(df)


def test_hierarchical_blocks_r2_change(data):
    res = regression.linear(data, "y", blocks=[("enter", ["x1", "m"]), ("enter", ["x1m"])],
                            statistics=("estimates", "model_fit", "r2_change"), plots=(),
                            assumptions=())
    s = table(res.section, "Model Summary")
    f1 = sm.OLS(data.df.y, sm.add_constant(data.df[["x1", "m"]])).fit()
    f2 = sm.OLS(data.df.y, sm.add_constant(data.df[["x1", "m", "x1m"]])).fit()
    assert math.isclose(num(s.rows[0][2]), f1.rsquared, abs_tol=5e-4)
    assert math.isclose(num(s.rows[1][2]), f2.rsquared, abs_tol=5e-4)
    assert math.isclose(num(s.rows[1][5]), f2.rsquared - f1.rsquared, abs_tol=5e-4)
    f_change, p_change, df_diff = f2.compare_f_test(f1)
    assert math.isclose(num(s.rows[1][6]), f_change, rel_tol=1e-3)
    assert s.rows[1][7] == "1" and s.rows[1][8] == str(int(f2.df_resid))
    # Model 1 lists the block-2 variable as excluded, with its partial correlation
    ex = table(res.section, "Excluded Variables")
    t = f2.tvalues["x1m"]
    assert math.isclose(num(ex.rows[0][3]), t, rel_tol=1e-3)
    assert math.isclose(num(ex.rows[0][5]), t / math.sqrt(t * t + f2.df_resid), abs_tol=5e-4)


def test_coefficients_part_partial_ci_collinearity(data):
    res = regression.linear(data, "y", ["x1", "x2", "x3"],
                            statistics=("estimates", "ci", "part_partial", "collinearity",
                                        "model_fit", "bcov"), plots=(), assumptions=())
    X = sm.add_constant(data.df[["x1", "x2", "x3"]])
    f = sm.OLS(data.df.y, X).fit()
    c = table(res.section, "Coefficients")
    r = row(c, "x2")
    assert math.isclose(num(r[2]), f.params["x2"], abs_tol=5e-4)
    assert math.isclose(num(r[3]), f.bse["x2"], abs_tol=5e-4)
    lo, hi = f.conf_int().loc["x2"]
    assert math.isclose(num(r[7]), lo, abs_tol=5e-4) and math.isclose(num(r[8]), hi,
                                                                       abs_tol=5e-4)
    partial = f.tvalues["x2"] / math.sqrt(f.tvalues["x2"] ** 2 + f.df_resid)
    part = f.tvalues["x2"] * math.sqrt((1 - f.rsquared) / f.df_resid)
    assert math.isclose(num(r[10]), partial, abs_tol=5e-4)
    assert math.isclose(num(r[11]), part, abs_tol=5e-4)
    assert math.isclose(num(r[9]), np.corrcoef(data.df.x2, data.df.y)[0, 1], abs_tol=5e-4)
    # Collinearity diagnostics: eigenvalues of the unit-scaled X'X
    Z = X.to_numpy() / np.linalg.norm(X.to_numpy(), axis=0)
    lam = np.sort(np.linalg.eigvalsh(Z.T @ Z))[::-1]
    d = table(res.section, "Collinearity Diagnostics")
    assert [round(num(rr[2]), 3) for rr in d.rows] == [round(v, 3) for v in lam]
    assert math.isclose(num(d.rows[-1][3]), math.sqrt(lam[0] / lam[-1]), abs_tol=5e-3)
    props = np.array([[num(v) for v in rr[4:]] for rr in d.rows])
    assert np.allclose(props.sum(axis=0), 1, atol=0.03)
    cc = table(res.section, "Coefficient Correlations")
    cov = f.cov_params()
    covrow = next(rr for rr in cc.rows if rr[1] == "Covariances")
    assert math.isclose(num(covrow[3]), cov.loc["x3", "x3"], abs_tol=5e-4)


def test_save_diagnostics_match_statsmodels(data):
    keys = ("pred", "zpred", "adjpred", "sepred", "resid", "zresid", "sresid", "dresid",
            "sdresid", "mahal", "cook", "lever", "dfbeta", "sdfbeta", "dffit", "sdffit",
            "covratio", "mcin", "icin")
    res = regression.linear(data, "y", ["x1", "x2"], save=keys, plots=(), assumptions=(),
                            statistics=("estimates", "model_fit", "casewise", "press"))
    X = sm.add_constant(data.df[["x1", "x2"]])
    f = sm.OLS(data.df.y, X).fit()
    inf = OLSInfluence(f)
    sv = {k: v[1].to_numpy() for k, v in res.saved.items()}
    n = len(data.df)
    assert np.allclose(sv["PRE"], f.fittedvalues)
    assert np.allclose(sv["RES"], f.resid)
    assert np.allclose(sv["ZRE"], f.resid / math.sqrt(f.mse_resid))
    assert np.allclose(sv["SRE"], inf.resid_studentized_internal)
    assert np.allclose(sv["SDR"], inf.resid_studentized_external)
    assert np.allclose(sv["COO"], inf.cooks_distance[0])
    assert np.allclose(sv["LEV"], inf.hat_matrix_diag - 1 / n)
    assert np.allclose(sv["DRE"], f.resid / (1 - inf.hat_matrix_diag))
    assert np.allclose(sv["DFB1"], inf.dfbeta[:, 1])
    assert np.allclose(sv["SDB2"], inf.dfbetas[:, 2])
    assert np.allclose(sv["SDF"], inf.dffits[0])
    assert np.allclose(sv["COV"], inf.cov_ratio)
    Xc = data.df[["x1", "x2"]].to_numpy()
    d = Xc - Xc.mean(axis=0)
    S = np.cov(Xc, rowvar=False)
    mahal = np.einsum("ij,jk,ik->i", d, np.linalg.inv(S), d)
    assert np.allclose(sv["MAH"], mahal)
    pr = f.get_prediction(X).summary_frame(alpha=0.05)
    assert np.allclose(sv["LMCI"], pr.mean_ci_lower) and np.allclose(sv["UICI"], pr.obs_ci_upper)
    assert np.allclose(sv["SEP"], pr.mean_se)
    summary = table(res.section, "Model Summary")
    press = float(np.sum((f.resid / (1 - inf.hat_matrix_diag)) ** 2))
    assert math.isclose(num(summary.rows[0][-1]), press, rel_tol=1e-3)
    rs = table(res.section, "Residuals Statistics")
    assert row(rs, "Cook's Distance", 0)


def test_wls_weight_matches_statsmodels(data):
    res = regression.linear(data, "y", ["x1", "x2"], wls="w", plots=(), assumptions=(),
                            statistics=("estimates", "model_fit"))
    f = sm.WLS(data.df.y, sm.add_constant(data.df[["x1", "x2"]]), weights=data.df.w).fit()
    s = table(res.section, "Model Summary")
    assert math.isclose(num(s.rows[0][2]), f.rsquared, abs_tol=5e-4)
    c = table(res.section, "Coefficients")
    assert math.isclose(num(row(c, "x1")[3]), f.bse["x1"], abs_tol=5e-4)
    assert any("Weighted Least Squares" in n for n in s.footnotes)


def test_no_constant_and_use_f_and_remove(data):
    res = regression.linear(data, "y", ["x1", "x2"], constant=False, plots=(), assumptions=(),
                            statistics=("estimates", "model_fit"))
    f = sm.OLS(data.df.y, data.df[["x1", "x2"]]).fit()
    s = table(res.section, "Model Summary")
    assert math.isclose(num(s.rows[0][2]), f.rsquared, abs_tol=5e-4)
    res = regression.linear(data, "y", blocks=[("enter", ["x1", "x2", "x3"]),
                                               ("remove", ["x3"])], plots=(), assumptions=())
    er = table(res.section, "Variables Entered/Removed")
    assert er.rows[1][2] == "x3" and er.rows[1][3] == "Remove"
    res = regression.linear(data, "y", ["x1", "x2", "x3"], method="stepwise", use_f=True,
                            plots=(), assumptions=())
    er = table(res.section, "Variables Entered/Removed")
    assert "F-to-enter" in er.rows[0][3]
    assert "x3" not in [r[1] for r in er.rows]


def test_backward_and_selection_and_casewise(data):
    res = regression.linear(data, "y", ["x1", "x2", "x3"], method="backward", plots=(),
                            assumptions=())
    er = table(res.section, "Variables Entered/Removed")
    assert er.rows[0][3] == "Enter" and er.rows[1][2] == "x3"
    res = regression.linear(data, "y", ["x1", "x2"], selection=("g", "eq", 1.0), plots=(),
                            assumptions=(), statistics=("estimates", "model_fit",
                                                        "casewise"), casewise_sd=2.0)
    sub = data.df[data.df.g == 1]
    f = sm.OLS(sub.y, sm.add_constant(sub[["x1", "x2"]])).fit()
    s = table(res.section, "Model Summary")
    assert math.isclose(num(s.rows[0][3]), f.rsquared, abs_tol=5e-4)
    assert "(Selected)" in s.columns[1][1]
    cw = table(res.section, "Casewise Diagnostics")
    z = f.resid / math.sqrt(f.mse_resid)
    assert len(cw.rows) == int((np.abs(z) >= 2).sum())


def test_pairwise_and_mean_substitution():
    rng = np.random.default_rng(5)
    n = 60
    df = pd.DataFrame({"y": rng.normal(size=n), "a": rng.normal(size=n),
                       "b": rng.normal(size=n)})
    df.loc[[1, 5, 9], "a"] = np.nan
    df.loc[[2, 7], "b"] = np.nan
    ds = Dataset(df)
    res = regression.linear(ds, "y", ["a", "b"], missing="pairwise", plots=(), assumptions=())
    c = table(res.section, "Coefficients")
    cov = df.cov().to_numpy()
    b = np.linalg.solve(cov[1:, 1:], cov[1:, 0])
    assert math.isclose(num(row(c, "a")[2]), b[0], abs_tol=5e-4)
    res = regression.linear(ds, "y", ["a", "b"], missing="mean", plots=(), assumptions=())
    filled = df.fillna(df.mean())
    f = sm.OLS(filled.y, sm.add_constant(filled[["a", "b"]])).fit()
    c = table(res.section, "Coefficients")
    assert math.isclose(num(row(c, "b")[2]), f.params["b"], abs_tol=5e-4)


def test_plots_and_descriptives(data):
    res = regression.linear(data, "y", ["x1", "x2"], plots=("histogram",),
                            plot_pairs=[("*SRESID", "*ZPRED"), ("DEPENDNT", "*ADJPRED")],
                            partial_plots=True, assumptions=(),
                            statistics=("estimates", "descriptives", "model_fit"))
    assert sum(i.kind == "image" for i in res.section.items) == 1 + 2 + 2
    corr = table(res.section, "Correlations")
    assert math.isclose(num(corr.rows[0][3]), np.corrcoef(data.df.y, data.df.x1)[0, 1],
                        abs_tol=5e-4)
