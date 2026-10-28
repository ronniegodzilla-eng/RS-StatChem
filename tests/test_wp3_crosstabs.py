"""WP3: Crosstabs (layers, cells, all statistics, exact / Monte Carlo) against references.

References: Agresti (2002) *Categorical Data Analysis* §2.4 (job satisfaction x income:
gamma = .221, ASE = .117, X² = 5.97); scipy (somersd, kendalltau, fisher_exact, pearsonr,
spearmanr); statsmodels (cohens_kappa, mcnemar, SquareTable, Table2x2, StratifiedTable).
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.stats import crosstabs as ct

SHOTS = os.environ.get(
    "WP3_SHOTS", "/private/tmp/claude-501/-Users-ronisaputra-Aplikasi-statistik/"
                 "a8ca05d2-7214-4d46-8f3d-8e0d4372bfcd/scratchpad/wp3")

AGRESTI = np.array([[1, 3, 10, 6], [2, 3, 10, 7], [1, 6, 14, 12], [0, 1, 9, 11]], float)


def expand(n):
    x, y = [], []
    for i in range(n.shape[0]):
        for j in range(n.shape[1]):
            x += [i] * int(n[i, j])
            y += [j] * int(n[i, j])
    return np.array(x, float), np.array(y, float)


def table(sec, title):
    return next(t for t in sec.items if getattr(t, "title", "").startswith(title))


def num(text):
    return float(text.replace("ᵃ", "").replace("ᵇ", "").replace("ᶜ", "").replace("ᵈ", "")
                 .replace("ᵉ", "").strip())


def test_agresti_gamma_and_chisquare():
    o = ct.ordinal_measures(AGRESTI)
    assert o["gamma"].value == pytest.approx(0.221, abs=5e-4)
    assert o["gamma"].ase == pytest.approx(0.117, abs=5e-4)
    chi, _ = ct.pearson_chi2(AGRESTI)
    assert chi == pytest.approx(5.97, abs=5e-3)
    g2, _ = ct.likelihood_ratio(AGRESTI)
    assert g2 == pytest.approx(6.76, abs=5e-3)


def test_ordinal_ase_equal_delta_method():
    n = AGRESTI

    def pq(p):
        c, d = ct.concordance(p)
        return (p * c).sum(), (p * d).sum(), p.sum()

    def dr(p):
        return p.sum() ** 2 - (p.sum(1) ** 2).sum()

    def dc(p):
        return p.sum() ** 2 - (p.sum(0) ** 2).sum()

    fns = {
        "gamma": lambda p: (lambda P, Q, N: (P - Q) / (P + Q))(*pq(p)),
        "btau": lambda p: (lambda P, Q, N: (P - Q) / math.sqrt(dr(p) * dc(p)))(*pq(p)),
        "ctau": lambda p: (lambda P, Q, N: 4 * (P - Q) / (N * N * 3))(*pq(p)),
        "d_R": lambda p: (lambda P, Q, N: (P - Q) / dc(p))(*pq(p)),
        "d_C": lambda p: (lambda P, Q, N: (P - Q) / dr(p))(*pq(p)),
        "d_S": lambda p: (lambda P, Q, N: 2 * (P - Q) / (dr(p) + dc(p)))(*pq(p)),
    }
    o = ct.ordinal_measures(n)
    for key, f in fns.items():
        if key == "d_S":
            continue
        assert o[key].ase == pytest.approx(ct.delta_ase(f, n), rel=1e-5), key
    # symmetric Somers' d: IBM CROSSTABS prints 2 sqrt(Dr Dc)/(Dr + Dc) * ASE1(tau-b), which is
    # close to but not the full delta-method value (WP7: follow IBM)
    drv, dcv = dr(n), dc(n)
    assert o["d_S"].ase == pytest.approx(2 * math.sqrt(drv * dcv) / (drv + dcv)
                                         * o["btau"].ase, rel=1e-12)
    assert o["d_S"].ase == pytest.approx(ct.delta_ase(fns["d_S"], n), rel=1e-2)
    # all ordinal measures share the null-hypothesis T, as SPSS prints them
    assert len({round(o[k].t, 10) for k in fns}) == 1


def test_ordinal_against_scipy():
    x, y = expand(AGRESTI)
    o = ct.ordinal_measures(AGRESTI)
    sd = stats.somersd(AGRESTI)   # rows independent -> column variable dependent
    assert o["d_C"].value == pytest.approx(sd.statistic, abs=1e-10)
    assert o["d_C"].p == pytest.approx(sd.pvalue, abs=1e-8)
    assert o["btau"].value == pytest.approx(stats.kendalltau(x, y).statistic, abs=1e-10)
    assert o["ctau"].value == pytest.approx(stats.kendalltau(x, y, variant="c").statistic,
                                            abs=1e-10)


def test_correlations_and_eta():
    x, y = expand(AGRESTI)
    c = ct.correlations(AGRESTI, [1, 2, 3, 4], [1, 2, 3, 4])
    assert c["pearson"].value == pytest.approx(stats.pearsonr(x, y).statistic, abs=1e-10)
    assert c["pearson"].p == pytest.approx(stats.pearsonr(x, y).pvalue, abs=1e-8)
    assert c["spearman"].value == pytest.approx(stats.spearmanr(x, y).statistic, abs=1e-10)
    e = ct.eta(AGRESTI, [1, 2, 3, 4], [1, 2, 3, 4])
    groups = [x[y == j] for j in range(4)]
    ss_w = sum(((g - g.mean()) ** 2).sum() for g in groups)
    assert e["R"] == pytest.approx(math.sqrt(1 - ss_w / ((x - x.mean()) ** 2).sum()))


def test_nominal_measures_definitions():
    n = np.array([[20, 5, 3], [4, 18, 6], [2, 7, 25]], float)
    m = ct.nominal_measures(n)
    total = n.sum()
    lam_c = (n.max(1).sum() - n.sum(0).max()) / (total - n.sum(0).max())
    assert m["lambda_C"].value == pytest.approx(lam_c)
    # ASE1 formula (SPSS) equals the delta method for asymmetric lambda
    assert m["lambda_C"].ase == pytest.approx(
        ct.delta_ase(lambda p: ct._lambda_value(p, "C"), n), rel=1e-5)
    assert m["lambda_R"].ase == pytest.approx(
        ct.delta_ase(lambda p: ct._lambda_value(p, "R"), n), rel=1e-5)
    # Goodman-Kruskal tau as the proportional reduction of Gini variation
    p = n / total
    c = p.sum(0)
    v = 1 - (c ** 2).sum()
    ev = sum(p[i].sum() * (1 - ((p[i] / p[i].sum()) ** 2).sum()) for i in range(3))
    assert m["tau_C"].value == pytest.approx((v - ev) / v)
    # uncertainty coefficient from entropies
    hr, hc = stats.entropy(p.sum(1)), stats.entropy(c)
    hrc = stats.entropy(p.ravel())
    assert m["uc_C"].value == pytest.approx((hr + hc - hrc) / hc)
    assert m["uc_S"].value == pytest.approx(2 * (hr + hc - hrc) / (hr + hc))
    assert m["uc_C"].p == pytest.approx(ct.likelihood_ratio(n)[1])
    chi = stats.chi2_contingency(n, correction=False)[0]
    assert m["v"].value == pytest.approx(math.sqrt(chi / (total * 2)))
    assert m["cc"].value == pytest.approx(math.sqrt(chi / (chi + total)))


def test_kappa_risk_mcnemar_against_statsmodels():
    from statsmodels.stats.contingency_tables import SquareTable, Table2x2, mcnemar
    from statsmodels.stats.inter_rater import cohens_kappa

    n = np.array([[22, 4, 2], [5, 17, 3], [1, 4, 12]], float)
    k = ct.kappa(n)
    ref = cohens_kappa(n)
    assert k.value == pytest.approx(ref.kappa)
    assert k.ase == pytest.approx(ref.std_kappa)
    assert k.t == pytest.approx(ref.z_value)
    bowker = ct.mcnemar(n)
    sym = SquareTable(n).symmetry()
    assert bowker["value"] == pytest.approx(sym.statistic)
    assert bowker["df"] == sym.df
    t2 = np.array([[30, 12], [5, 23]], float)
    assert ct.mcnemar(t2)["p"] == pytest.approx(mcnemar(t2, exact=True).pvalue)
    r = ct.risk(t2)
    tab = Table2x2(t2)
    assert r["or"][0] == pytest.approx(tab.oddsratio)
    assert r["or"][1:] == pytest.approx(tab.oddsratio_confint())
    assert r["rr1"][0] == pytest.approx(tab.riskratio)
    assert r["rr1"][1:] == pytest.approx(tab.riskratio_confint())


def test_cmh_against_statsmodels():
    from statsmodels.stats.contingency_tables import StratifiedTable

    tabs = [np.array([[12, 8], [6, 14]], float), np.array([[9, 11], [5, 15]], float),
            np.array([[15, 5], [10, 10]], float)]
    res = ct.cmh(tabs, 1.0)
    st = StratifiedTable(tabs)
    assert res["mh"][0] == pytest.approx(st.test_null_odds(correction=True).statistic)
    assert res["breslow"][0] == pytest.approx(st.test_equal_odds().statistic)
    assert res["or"]["estimate"] == pytest.approx(st.oddsratio_pooled)
    assert res["or"]["se"] == pytest.approx(st.logodds_pooled_se)
    # Cochran's statistic uses n^3 in the variance (no n-1)
    a = np.array([t[0, 0] for t in tabs])
    n1 = np.array([t[0].sum() for t in tabs])
    m1 = np.array([t[:, 0].sum() for t in tabs])
    tot = np.array([t.sum() for t in tabs])
    var = (n1 * (tot - n1) * m1 * (tot - m1) / tot ** 3).sum()
    assert res["cochran"][0] == pytest.approx(((a - n1 * m1 / tot).sum()) ** 2 / var)


def test_column_proportion_letters():
    from statsmodels.stats.proportion import proportions_ztest

    n = np.array([[40, 10, 38], [10, 40, 12]], float)
    letters, excluded = ct.column_proportion_letters(n)
    assert letters[0][0] == letters[0][2] != letters[0][1]
    assert not excluded
    _, p = proportions_ztest([40, 10], [50, 50])
    assert p * 3 < 0.05
    # a zero proportion is excluded from the comparisons
    _, excluded = ct.column_proportion_letters(np.array([[5, 0, 7], [5, 9, 3]], float))
    assert (0, 1) in excluded


def test_exact_tests_against_scipy_and_enumeration():
    t2 = np.array([[7, 2], [1, 6]], float)
    ex = ct.exact_tests(t2, "exact")
    assert ex["fisher"]["p"] == pytest.approx(stats.fisher_exact(t2)[1])
    one = min(stats.fisher_exact(t2, alternative="less")[1],
              stats.fisher_exact(t2, alternative="greater")[1])
    assert ex["fisher"]["p1"] == pytest.approx(one)
    # enumeration: count equals brute force and probabilities sum to one
    row, col = np.array([3, 2, 4]), np.array([4, 5])
    tabs = ct.enumerate_tables(row, col)
    brute = sum(1 for a in range(4) for b in range(3) if 0 <= 4 - a - b <= 4)
    assert len(tabs) == brute
    probs = np.exp(ct.table_statistics(tabs)["log_p"])
    assert probs.sum() == pytest.approx(1.0)
    # r x c exact p agrees with a large Monte Carlo run
    rc = np.array([[3, 1, 0, 2], [0, 4, 1, 1], [2, 0, 5, 1]], float)
    exact = ct.exact_tests(rc, "exact")
    mc = ct.exact_tests(rc, "mc", samples=40000, seed=1)
    for key in ("pearson", "lr", "fisher", "lbl"):
        assert exact[key]["method"] == "exact"
        assert abs(exact[key]["p"] - mc[key]["p"]) < 0.015, key
        assert mc[key]["lower"] <= mc[key]["p"] <= mc[key]["upper"]
    # too many tables -> falls back to Monte Carlo
    big = ct.exact_tests(np.full((4, 4), 15.0), "exact", samples=200, limit=1000)
    assert big["pearson"]["method"] == "mc"


@pytest.fixture
def ds():
    rng = np.random.default_rng(11)
    n = 240
    sex = rng.integers(1, 3, n).astype(float)
    smoke = (rng.random(n) < np.where(sex == 1, 0.55, 0.3)).astype(float) + 1
    edu = rng.integers(1, 4, n).astype(float)
    sat = np.clip(edu + rng.integers(-1, 2, n), 1, 4).astype(float)
    region = rng.integers(1, 3, n).astype(float)
    d = Dataset(pd.DataFrame({"sex": sex, "smoke": smoke, "edu": edu, "sat": sat,
                              "region": region}))
    d.var("sex").value_labels = {1.0: "Male", 2.0: "Female"}
    d.var("smoke").value_labels = {1.0: "No", 2.0: "Yes"}
    return d


def test_full_procedure_with_layers(ds):
    sec = ct.crosstabs(ds, ["sex"], ["smoke"], cells=ct.CELLS, layers=[["region"]],
                       statistics=ct.STATISTICS, z_test=True, clustered_bar=True)
    cross = table(sec, "sex * smoke * region Crosstabulation")
    labels = {r[3] for r in cross.rows}
    assert {"Count", "Expected Count", "Residual", "Standardized Residual",
            "Adjusted Residual"} <= labels
    assert any(r[0] == "Total" for r in cross.rows)
    chi = table(sec, "Chi-Square Tests")
    assert any("McNemar" in r[1] for r in chi.rows)
    for title in ("Directional Measures", "Symmetric Measures", "Risk Estimate",
                  "Tests of Homogeneity of the Odds Ratio", "Tests of Conditional Independence",
                  "Mantel-Haenszel Common Odds Ratio Estimate"):
        assert table(sec, title).rows
    assert any(getattr(i, "kind", "") == "image" for i in sec.items)
    # adjusted residual of the first cell of the first layer
    sub = ds.df[ds.df.region == 1]
    obs = pd.crosstab(sub.sex, sub.smoke).to_numpy(float)
    e = np.outer(obs.sum(1), obs.sum(0)) / obs.sum()
    adj = (obs[0, 0] - e[0, 0]) / math.sqrt(
        e[0, 0] * (1 - obs[0].sum() / obs.sum()) * (1 - obs[:, 0].sum() / obs.sum()))
    row = next(r for r in cross.rows if r[3] == "Adjusted Residual")
    assert num(row[4]) == pytest.approx(adj, abs=0.051)


def test_backward_compatible_call_and_exact_columns(ds):
    sec = ct.crosstabs(ds, ["edu"], ["sat"], chisq=True, phi=True,
                       cells=("observed", "expected", "row"))
    obs = pd.crosstab(ds.df.edu, ds.df.sat).to_numpy()
    chi = stats.chi2_contingency(obs, correction=False)[0]
    assert num(table(sec, "Chi-Square Tests").rows[0][1]) == pytest.approx(chi, abs=5e-4)
    sec = ct.crosstabs(ds, ["edu"], ["sat"], exact="mc", mc_samples=2000)
    cols = table(sec, "Chi-Square Tests").columns
    assert any(isinstance(c, tuple) and c[0].startswith("Monte Carlo") for c in cols)


def test_crosstabs_dialog(qapp, ds, monkeypatch):
    from PySide6.QtWidgets import QDialog

    from rsstatchem.ui.dialogs.explore import CrosstabsDialog

    os.makedirs(SHOTS, exist_ok=True)
    shots = []

    def fake_exec(self):
        self.adjustSize()
        path = os.path.join(SHOTS, f"crosstabs_sub{len(shots)}.png")
        self.grab().save(path)
        shots.append(path)
        return 1

    dlg = CrosstabsDialog(ds)
    dlg._take("sex", dlg._targets[0])
    dlg._take("smoke", dlg._targets[1])
    dlg._take("region", dlg._targets[2])
    dlg._update_arrows()
    dlg.resize(720, 460)
    dlg.grab().save(os.path.join(SHOTS, "crosstabs.png"))
    monkeypatch.setattr(QDialog, "exec", fake_exec)
    dlg._statistics()
    dlg._cells()
    dlg._format()
    assert len(shots) == 3
    for key in ct.STATISTICS:
        dlg.opts[key] = True
    dlg.opts["cells"] = list(ct.CELLS)
    dlg.opts["z_test"] = True
    sec = dlg.compute()
    assert table(sec, "Symmetric Measures").rows
    # layer navigation keeps every layer
    dlg._switch_layer(1)
    assert dlg.layer_label.text() == "Layer 2 of 2"
    dlg._take("edu", dlg._targets[2])
    assert dlg.layers() == [["region"], ["edu"]]
