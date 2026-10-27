"""WP2: Distances (PROXIMITIES), Canonical Correlation and ROC Analysis against scipy,
statsmodels and scikit-learn (test-only)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy.spatial import distance as D

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.stats import cancorr, distances, roc_analysis


def num(text):
    return float(text.rstrip("ᵃᵇ"))


def table(sec, title):
    return next(t for t in sec.items if getattr(t, "title", "").startswith(title))


@pytest.fixture
def ds():
    rng = np.random.default_rng(6)
    n = 12
    return Dataset(pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n),
                                 "c": rng.normal(size=n), "p": rng.integers(0, 2, n) * 1.0,
                                 "q": rng.integers(0, 2, n) * 1.0,
                                 "r": rng.integers(0, 2, n) * 1.0}))


@pytest.mark.parametrize("measure,ref", [
    ("euclid", lambda u, v: D.euclidean(u, v)), ("seuclid", lambda u, v: D.sqeuclidean(u, v)),
    ("block", lambda u, v: D.cityblock(u, v)), ("chebychev", lambda u, v: D.chebyshev(u, v)),
    ("minkowski", lambda u, v: D.minkowski(u, v, 3)),
    ("cosine", lambda u, v: 1 - D.cosine(u, v)),
    ("correlation", lambda u, v: 1 - D.correlation(u, v))])
def test_interval_measures_between_cases(ds, measure, ref):
    sec = distances.distances(ds, ["a", "b", "c"], "cases", measure, power=3)
    M = ds.df[["a", "b", "c"]].to_numpy()
    t = table(sec, "Proximity Matrix")
    assert math.isclose(num(t.rows[0][3]), ref(M[0], M[2]), abs_tol=5e-4)
    assert t.footnotes[0].endswith("similarity matrix" if measure in ("cosine", "correlation")
                                   else "dissimilarity matrix")


@pytest.mark.parametrize("measure,ref", [
    ("jaccard", lambda u, v: 1 - D.jaccard(u, v)), ("dice", lambda u, v: 1 - D.dice(u, v)),
    ("rt", lambda u, v: 1 - D.rogerstanimoto(u, v)), ("rr", lambda u, v: 1 - D.russellrao(u, v)),
    ("ss2", lambda u, v: 1 - D.sokalsneath(u, v)), ("sm", lambda u, v: 1 - D.hamming(u, v)),
    ("q", lambda u, v: 1 - D.yule(u, v))])
def test_binary_measures_between_variables(ds, measure, ref):
    sec = distances.distances(ds, ["p", "q", "r"], "variables", measure)
    t = table(sec, "Proximity Matrix")
    u, v = ds.df.p.to_numpy().astype(bool), ds.df.q.to_numpy().astype(bool)
    assert math.isclose(num(t.rows[0][2]), ref(u, v), abs_tol=5e-4)


def test_standardize_and_transforms(ds):
    sec = distances.distances(ds, ["a", "b"], "cases", "euclid", standardize="z",
                              rescale=True)
    t = table(sec, "Proximity Matrix")
    vals = np.array([[num(c) for c in r[1:]] for r in t.rows])
    assert math.isclose(vals.max(), 1.0) and math.isclose(vals.min(), 0.0)
    Z = (ds.df[["a", "b"]] - ds.df[["a", "b"]].mean()) / ds.df[["a", "b"]].std()
    raw = distances.proximity(Z.to_numpy(), "euclid")
    assert math.isclose(vals[0, 1], raw[0, 1] / raw.max(), abs_tol=5e-4)
    chi = distances.proximity(np.array([[3.0, 5, 2], [4, 1, 6]]), "chisq")
    from scipy.stats import chi2_contingency
    assert math.isclose(chi[0, 1], math.sqrt(chi2_contingency(np.array([[3, 5, 2], [4, 1, 6]]),
                                                              correction=False)[0]))


def test_canonical_correlation_matches_statsmodels():
    from statsmodels.multivariate.cancorr import CanCorr

    rng = np.random.default_rng(10)
    n = 100
    a = rng.normal(size=(n, 3))
    b = a[:, :2] @ np.array([[0.7, 0.2], [0.1, 0.5]]) + rng.normal(size=(n, 2))
    df = pd.DataFrame(np.column_stack([a, b]), columns=["a1", "a2", "a3", "b1", "b2"])
    res = cancorr.canonical_correlation(Dataset(df), ["a1", "a2", "a3"], ["b1", "b2"],
                                        save=True)
    ref = CanCorr(df[["b1", "b2"]].to_numpy(), df[["a1", "a2", "a3"]].to_numpy())
    cc = table(res.section, "Canonical Correlations")
    assert np.allclose([num(r[1]) for r in cc.rows], ref.cancorr, atol=5e-4)
    test = ref.corr_test().stats
    assert math.isclose(num(cc.rows[0][3]), test["Wilks' lambda"].iloc[0], abs_tol=5e-4)
    # first Wilks test = the multivariate regression test of set 2 on set 1 (Rao's F, as
    # SPSS MANOVA/SAS CANCORR; statsmodels' CanCorr uses an asymmetric df2)
    from statsmodels.multivariate.manova import MANOVA

    exog = np.column_stack([np.ones(n), df[["a1", "a2", "a3"]].to_numpy()])
    mv = MANOVA(df[["b1", "b2"]].to_numpy(), exog).mv_test(
        hypotheses=[("all", np.eye(4)[1:])])
    wilks = mv.results["all"]["stat"].loc["Wilks' lambda"]
    assert math.isclose(num(cc.rows[0][3]), wilks["Value"], abs_tol=5e-4)
    assert math.isclose(num(cc.rows[0][4]), wilks["F Value"], abs_tol=5e-3)
    assert math.isclose(num(cc.rows[0][7] or "0"), wilks["Pr > F"], abs_tol=5e-4)
    u = res.saved["s1_cv001"][1].to_numpy()
    v = res.saved["s2_cv001"][1].to_numpy()
    assert math.isclose(np.corrcoef(u, v)[0, 1], ref.cancorr[0], abs_tol=1e-6)
    assert table(res.section, "Proportion of Variance Explained")


def _delong_var_loop(p1, n1_, p2, n2_):
    """Reference DeLong et al. (1988) covariance by explicit structural components."""
    def comps(pos, neg):
        v10 = np.array([np.mean([(x > y) + 0.5 * (x == y) for y in neg]) for x in pos])
        v01 = np.array([np.mean([(x > y) + 0.5 * (x == y) for x in pos]) for y in neg])
        return v10, v01
    a10, a01 = comps(p1, n1_)
    b10, b01 = comps(p2, n2_)
    m, n = len(a10), len(a01)
    s10 = np.cov(np.vstack([a10, b10]))
    s01 = np.cov(np.vstack([a01, b01]))
    S = s10 / m + s01 / n
    return S[0, 0] + S[1, 1] - 2 * S[0, 1]


def test_roc_analysis_paired_delong_and_pr():
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(13)
    n = 90
    s = rng.integers(0, 2, n).astype(float)
    a = s * 1.2 + rng.normal(size=n)
    b = s * 0.6 + rng.normal(size=n)
    ds = Dataset(pd.DataFrame({"s": s, "a": a, "b": b, "g": rng.integers(1, 3, n) * 1.0}))
    sec = roc_analysis.roc_analysis(ds, ["a", "b"], "s", 1.0, group="g", coordinates=True)
    pd_ = table(sec, "Paired-Sample Area Difference")
    var = _delong_var_loop(a[s == 1], a[s == 0], b[s == 1], b[s == 0])
    assert math.isclose(num(pd_.rows[0][4]), math.sqrt(var), abs_tol=5e-4)
    pr = table(sec, "Area Under the Precision-Recall Curve")
    assert math.isclose(num(pr.rows[0][1]), average_precision_score(s, a), abs_tol=5e-4)
    assert table(sec, "Independent-Group Area Difference")
    m = table(sec, "Classifier Evaluation Metrics")
    from sklearn.metrics import roc_auc_score
    assert math.isclose(num(m.rows[0][7]), 2 * roc_auc_score(s, a) - 1, abs_tol=5e-4)
