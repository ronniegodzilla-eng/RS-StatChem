"""WP3: Reliability models and statistics, ICC, Fleiss' kappa, weighted kappa, Bland-Altman.

References: Shrout & Fleiss (1979) Table 2 data (ICC(1,1)=.17, ICC(2,1)=.29, ICC(3,1)=.71,
ICC(1,4)=.44, ICC(2,4)=.62, ICC(3,4)=.91); pingouin (cronbach_alpha, intraclass_corr,
rm_anova); statsmodels (Factor ML, fleiss_kappa, cohens_kappa with weights); Wikipedia /
Fleiss (1971)-style worked example for Fleiss' kappa (kappa = .210); Bland & Altman (1999).
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.stats import agreement as ag
from rsstatchem.core.stats import reliability as R

SHOTS = os.environ.get(
    "WP3_SHOTS", "/private/tmp/claude-501/-Users-ronisaputra-Aplikasi-statistik/"
                 "a8ca05d2-7214-4d46-8f3d-8e0d4372bfcd/scratchpad/wp3")
SHROUT_FLEISS = np.array([[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8], [7, 1, 2, 6],
                          [10, 5, 6, 9], [6, 2, 4, 7]], float)


def table(sec, title):
    return next(t for t in sec.items if getattr(t, "title", "").startswith(title))


@pytest.fixture(scope="module")
def items():
    rng = np.random.default_rng(5)
    n = 90
    true = rng.normal(size=n)
    x = np.column_stack([np.clip(np.round(3 + (0.5 + 0.2 * j) * true
                                          + rng.normal(scale=0.9, size=n)), 1, 5)
                         for j in range(5)])
    return x


@pytest.fixture(scope="module")
def ds(items):
    return Dataset(pd.DataFrame(items, columns=[f"q{j}" for j in range(1, 6)]))


def test_icc_shrout_fleiss_published():
    x = SHROUT_FLEISS
    assert ag.icc(x, "oneway")["single"]["value"] == pytest.approx(0.17, abs=0.005)
    assert ag.icc(x, "oneway")["average"]["value"] == pytest.approx(0.44, abs=0.005)
    assert ag.icc(x, "twoway_random", "absolute")["single"]["value"] == pytest.approx(0.29,
                                                                                       abs=0.005)
    assert ag.icc(x, "twoway_random", "absolute")["average"]["value"] == pytest.approx(
        0.62, abs=0.005)
    assert ag.icc(x, "twoway_mixed", "consistency")["single"]["value"] == pytest.approx(
        0.71, abs=0.005)
    assert ag.icc(x, "twoway_mixed", "consistency")["average"]["value"] == pytest.approx(
        0.91, abs=0.005)


def test_icc_against_pingouin(items):
    import pingouin as pg

    n, k = items.shape
    long = pd.DataFrame([(i, j, items[i, j]) for i in range(n) for j in range(k)],
                        columns=["t", "r", "s"])
    ref = pg.intraclass_corr(long, "t", "r", "s").set_index("Type")
    spec = {"ICC(1,1)": ("oneway", "absolute", "single"),
            "ICC(1,k)": ("oneway", "absolute", "average"),
            "ICC(A,1)": ("twoway_random", "absolute", "single"),
            "ICC(A,k)": ("twoway_random", "absolute", "average"),
            "ICC(C,1)": ("twoway_mixed", "consistency", "single"),
            "ICC(C,k)": ("twoway_mixed", "consistency", "average")}
    for key, (model, kind, which) in spec.items():
        r = ag.icc(items, model, kind)[which]
        assert r["value"] == pytest.approx(ref.loc[key, "ICC"], abs=1e-10), key
        assert r["F"] == pytest.approx(ref.loc[key, "F"], rel=1e-8), key
        assert r["p"] == pytest.approx(ref.loc[key, "pval"], abs=1e-10), key
        lo, hi = ref.loc[key, "CI95"]
        assert r["lo"] == pytest.approx(lo, abs=0.006) and r["hi"] == pytest.approx(hi,
                                                                                     abs=0.006)
    # the mixed absolute-agreement estimator equals the two-way random one (SPSS note a)
    assert ag.icc(items, "twoway_mixed", "absolute")["single"]["value"] == pytest.approx(
        ref.loc["ICC(A,1)", "ICC"])


def test_icc_test_value_formula():
    a = ag.anova_two_way(SHROUT_FLEISS)
    r = ag.icc(SHROUT_FLEISS, "twoway_mixed", "consistency", test_value=0.5)
    f = a["msr"] / a["mse"] * (1 - 0.5) / (1 + 3 * 0.5)    # McGraw & Wong (1996) Table 8
    assert r["single"]["F"] == pytest.approx(f)
    assert r["average"]["F"] == pytest.approx(a["msr"] / a["mse"] * 0.5)


def test_alpha_models(items):
    import pingouin as pg

    alpha = R.cronbach_alpha(items)
    assert alpha == pytest.approx(pg.cronbach_alpha(pd.DataFrame(items))[0])
    lam = R.guttman_lambdas(items)
    assert lam[2] == pytest.approx(alpha)                   # lambda 3 = alpha
    s = np.cov(items, rowvar=False)
    assert lam[0] == pytest.approx(1 - np.trace(s) / s.sum())
    assert lam[1] >= lam[2] and lam[4] >= lam[0]
    sh = R.split_half(items)
    assert (sh["k1"], sh["k2"]) == (3, 2)                    # odd item goes to part 1 (SPSS)
    a, b = items[:, :3].sum(1), items[:, 3:].sum(1)
    r = np.corrcoef(a, b)[0, 1]
    assert sh["r"] == pytest.approx(r)
    assert sh["equal"] == pytest.approx(2 * r / (1 + r))
    assert sh["guttman"] == pytest.approx(
        2 * (1 - (a.var(ddof=1) + b.var(ddof=1)) / items.sum(1).var(ddof=1)))
    assert lam[3] == pytest.approx(sh["guttman"])
    pm = R.parallel_model(items)
    assert pm["reliability"] == pytest.approx(alpha)        # parallel ML reliability = alpha
    assert pm["error"] == pytest.approx(pm["common"] - pm["true"])
    assert R.split_half(items, second=1)["k2"] == 1


def test_omega_matches_ml_factor(items):
    from statsmodels.multivariate.factor import Factor

    fit = R.one_factor_ml(np.cov(items, rowvar=False))
    ref = Factor(corr=np.corrcoef(items, rowvar=False), n_factor=1, method="ml",
                 nobs=len(items)).fit()
    ref_load = np.abs(np.real(ref.loadings.ravel()))
    assert fit["std_loadings"] == pytest.approx(ref_load, abs=2e-4)
    omega = R.mcdonald_omega(items)
    lam, psi = fit["loadings"], fit["uniquenesses"]
    assert omega == pytest.approx(lam.sum() ** 2 / (lam.sum() ** 2 + psi.sum()))
    assert 0 < omega < 1


def test_anova_hotelling_tukey(items):
    import pingouin as pg

    n, k = items.shape
    long = pd.DataFrame([(i, j, items[i, j]) for i in range(n) for j in range(k)],
                        columns=["s", "i", "y"])
    ref = pg.rm_anova(long, dv="y", within="i", subject="s", correction=False)
    a = ag.anova_two_way(items)
    assert a["msc"] / a["mse"] == pytest.approx(ref["F"].iloc[0])
    h = R.hotelling_t2(items)
    d = items[:, :-1] - items[:, 1:]
    md = d.mean(0)
    t2 = n * md @ np.linalg.solve(np.cov(d, rowvar=False), md)
    assert h["t2"] == pytest.approx(t2)
    assert h["df2"] == n - k + 1
    tk = R.tukey_additivity(items)
    # Tukey's one-df nonadditivity = regression of the residuals on the product term
    grand = items.mean()
    ri, cj = items.mean(1) - grand, items.mean(0) - grand
    resid = items - grand - ri[:, None] - cj[None, :]
    prod = np.outer(ri, cj)
    ss = (resid * prod).sum() ** 2 / (prod ** 2).sum()
    assert tk["ss"] == pytest.approx(ss)
    frame = pd.DataFrame(items, columns=[f"v{c}" for c in range(k)])
    sec = R.reliability(Dataset(frame), list(frame.columns), anova="friedman")
    fr = table(sec, "ANOVA with Friedman's Test")
    chi = stats.friedmanchisquare(*items.T).statistic
    assert float(fr.rows[1][5]) == pytest.approx(chi, abs=5e-4)


def test_fleiss_kappa_worked_example():
    from statsmodels.stats.inter_rater import fleiss_kappa

    counts = np.array([[0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0],
                       [2, 2, 8, 1, 1], [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2],
                       [6, 5, 2, 1, 0], [0, 2, 2, 3, 7]], float)
    res = ag.fleiss_kappa(counts)
    assert res["kappa"] == pytest.approx(0.210, abs=5e-4)
    assert res["kappa"] == pytest.approx(fleiss_kappa(counts))
    n, m = 10, 14
    p = counts.sum(0) / (n * m)
    q = 1 - p
    var0 = 2 / (n * m * (m - 1)) * ((p * q).sum() ** 2 - (p * q * (q - p)).sum()) \
        / (p * q).sum() ** 2
    assert res["se"] == pytest.approx(math.sqrt(var0))
    assert res["categories"][0]["se"] == pytest.approx(math.sqrt(2 / (n * m * (m - 1))))


def test_weighted_kappa_against_statsmodels():
    from statsmodels.stats.inter_rater import cohens_kappa

    t = np.array([[11, 3, 1, 0], [2, 9, 4, 1], [0, 3, 12, 2], [1, 0, 3, 8]], float)
    for weighting, wt in (("linear", "linear"), ("quadratic", "quadratic")):
        res = ag.weighted_kappa(t, weighting)
        ref = cohens_kappa(t, wt=wt)
        assert res["kappa"] == pytest.approx(ref.kappa)
        assert res["ase"] == pytest.approx(ref.std_kappa)
        assert res["ase0"] == pytest.approx(ref.std_kappa0)
    plain = ag.weighted_kappa(t, "none")
    assert plain["kappa"] == pytest.approx(cohens_kappa(t).kappa)


def test_bland_altman_stats():
    rng = np.random.default_rng(2)
    a = rng.normal(100, 10, 40)
    b = a + 1.5 + 0.05 * a + rng.normal(0, 2, 40)
    res = ag.bland_altman_stats(a, b)
    d = a - b
    sd = d.std(ddof=1)
    assert res["bias"] == pytest.approx(d.mean())
    assert res["upper"] == pytest.approx(d.mean() + stats.norm.ppf(0.975) * sd)
    assert res["se_loa"] == pytest.approx(sd * math.sqrt(1 / 40 + 1.96 ** 2 / 78), rel=1e-3)
    assert res["p"] == pytest.approx(stats.ttest_1samp(d, 0).pvalue)
    lr = stats.linregress((a + b) / 2, d)
    assert res["slope"] == pytest.approx(lr.slope) and res["p_slope"] == pytest.approx(lr.pvalue)


def test_reliability_procedure_all_models(ds):
    names = ds.names()
    for model in R.MODELS:
        sec = R.reliability(ds, names, True, True, True, True, True, model=model,
                            inter_cov=True, summaries=("means", "variances", "covariances",
                                                       "correlations"),
                            anova="f", tukey=True, hotelling=True,
                            icc={"model": "twoway_mixed", "type": "consistency"},
                            fleiss={"individual": True})
        assert table(sec, "Reliability Statistics").rows
        assert table(sec, "Intraclass Correlation Coefficient").rows[1][0] == "Average Measures"
        assert table(sec, "Summary Item Statistics").rows[0][0] == "Item Means"
        assert table(sec, "Overall Kappa").rows
    sec = R.reliability(ds, names, model="omega")
    assert "McDonald's Omega if Item Deleted" in table(sec, "Item-Total Statistics").columns


def test_reliability_and_agreement_dialogs(qapp, ds, monkeypatch):
    from PySide6.QtWidgets import QDialog

    from rsstatchem.ui.dialogs.agreement import BlandAltmanDialog, WeightedKappaDialog
    from rsstatchem.ui.dialogs.explore import ReliabilityDialog

    os.makedirs(SHOTS, exist_ok=True)
    dlg = ReliabilityDialog(ds)
    for name in ds.names():
        dlg._take(name, dlg._targets[0])
    dlg._update_arrows()
    dlg.grab().save(os.path.join(SHOTS, "reliability.png"))
    monkeypatch.setattr(QDialog, "exec", lambda self: (self.adjustSize(), self.grab().save(
        os.path.join(SHOTS, f"sub_{self.windowTitle().split(':')[0].replace(' ', '_')}.png")),
        1)[-1])
    dlg._statistics()
    dlg.opts.update(icc=True, icc_model="twoway_random", icc_type="absolute", model="split",
                    s_means=True, anova="cochran", fleiss=True)
    sec = dlg.compute()
    assert table(sec, "Intraclass Correlation Coefficient")
    wk = WeightedKappaDialog(ds)
    wk._add_pair(wk._pairs[0], "q1", "q2")
    wk._criteria()
    sec = wk.compute()
    assert table(sec, "Weighted Kappa").rows[0][0] == "q1 - q2"
    wk.grab().save(os.path.join(SHOTS, "weighted_kappa.png"))
    ba = BlandAltmanDialog(ds)
    ba._add_pair(ba._pairs[0], "q1", "q2")
    ba._statistics()
    sec = ba.compute()
    assert table(sec, "Limits of Agreement").rows
    ba.grab().save(os.path.join(SHOTS, "bland_altman.png"))
