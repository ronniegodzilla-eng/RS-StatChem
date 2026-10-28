"""WP11 Data Preparation: Identify Unusual Cases (IBM DETECTANOMALY formulas), Validation rules
and Validate Data, and Automated Data Preparation."""
from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.stats import adp, anomaly, validation


# ----------------------------------------------------------------------------- anomaly
def test_vdi_matches_ibm_log_likelihood_formula():
    r = np.random.default_rng(0)
    Xc = r.normal(size=(40, 2))
    Xcat = r.integers(0, 3, (40, 1))
    peer = np.where(np.arange(40) < 20, 1, 2)
    m = 6.0
    vdi, st = anomaly.deviation_indices(Xc, Xcat, [3], peer, m)
    s = 7                                  # a case of peer group 1
    h = peer == 1
    Nh = h.sum()
    delta = Xc.var(0, ddof=1) / m
    for k in range(2):
        xh = Xc[h, k]
        merged = np.r_[xh, Xc[s, k]]
        exp = 0.5 * (-Nh * math.log(delta[k] + xh.var()) - math.log(delta[k])
                     + (Nh + 1) * math.log(delta[k] + merged.var()))
        assert vdi[s, k] == pytest.approx(exp)
    cnt = np.bincount(Xcat[h, 0], minlength=3).astype(float)
    cnt2 = cnt.copy()
    cnt2[Xcat[s, 0]] += 1
    ent = lambda c: -sum(p * math.log(p) for p in c / c.sum() if p > 0)  # noqa: E731
    assert vdi[s, 2] == pytest.approx(-Nh * ent(cnt) + (Nh + 1) * ent(cnt2))


def test_anomaly_index_averages_one_per_peer_group():
    r = np.random.default_rng(1)
    gdi = r.gamma(2, 1, 50)
    peer = np.r_[np.ones(30, int), np.full(20, 2)]
    idx, warn = anomaly.anomaly_indices(gdi, peer, 3)
    assert idx[peer == 1].mean() == pytest.approx(1.0)
    assert idx[peer == 2].mean() == pytest.approx(1.0)
    # IBM's 0/0 rule: identical cases in a singleton cluster -> avg size / size
    idx2, _ = anomaly.anomaly_indices(np.r_[np.zeros(1), np.ones(9)], np.r_[1, [2] * 9], 1)
    assert idx2[0] == pytest.approx(5.0)


def _anomaly_data():
    r = np.random.default_rng(2)
    n = 200
    g = r.integers(0, 2, n)
    df = pd.DataFrame({"id": np.arange(1001, 1001 + n, dtype=float),
                       "x": r.normal(g * 6, 1), "y": r.normal(-3 * g, 1),
                       "c": np.where(g == 1, 1.0, 2.0)})
    df.loc[17, "x"] = 25.0          # planted anomaly
    df.loc[33, "y"] = np.nan
    return Dataset(df)


def test_detect_anomaly_finds_planted_case_and_saves():
    ds = _anomaly_data()
    sec = anomaly.detect_anomaly(ds, ["c"], ["x", "y"], "id", include_missing=True,
                                 reasons=2, save={"index": True, "peers": True,
                                                  "reasons": True})
    res = sec.result
    # the planted value and the only case with a missing value (the missing-proportion
    # variable is then nearly constant, so that case is very unusual too) head the list
    assert set(res["anomalies"][:2]) == {17, 33}
    titles = [getattr(i, "title", "") for i in sec.items]
    for t in ("Anomaly Case Index List", "Anomaly Case Peer ID List",
              "Anomaly Case Reason List (Reason 1)", "Scale Variable Norms",
              "Categorical Variable Norms", "Anomaly Index Summary",
              "Reason Summary (Reason 1)"):
        assert t in titles
    reason = next(i for i in sec.items if getattr(i, "title", "") ==
                  "Anomaly Case Reason List (Reason 1)")
    assert next(r for r in reason.rows if r[0] == "18")[1:3] == ["1018", "x"]
    for nm in ("AnomalyIndex", "PeerId", "PeerSize", "PeerPctSize", "ReasonVar_1",
               "ReasonMeasure_2", "ReasonNorm_2"):
        assert ds.has_name(nm)
    assert ds.df["ReasonVar_1"].iloc[17] == "x"
    with pytest.raises(anomaly.AnomalyError):
        anomaly.detect_anomaly(ds, ["c"], ["x"], save={"index": True})
    # excluding missing values drops case 34 from the analysis
    sec2 = anomaly.detect_anomaly(_anomaly_data(), ["c"], ["x", "y"], cutpoint=None)
    assert 33 not in sec2.result["rows"]


# ----------------------------------------------------------------------------- validation
def test_single_variable_rules():
    ds = Dataset(pd.DataFrame({"a": [1.0, 2.5, 7.0, np.nan, 3.0],
                               "s": ["x", "", "y", "z", "x"]}))
    ds.var("a").missing = [3.0]
    rule = {"name": "r", "type": "numeric", "domain": "range", "min": 1, "max": 5,
            "noninteger": False, "user_missing": False, "sysmis": True}
    assert list(validation.violations(ds, "a", rule)) == [False, True, True, False, True]
    rule2 = dict(rule, domain="list", values=[1, 7], user_missing=True, sysmis=False)
    assert list(validation.violations(ds, "a", rule2)) == [False, True, False, True, False]
    rs = {"name": "s", "type": "string", "domain": "list", "values": ["x", "y"],
          "blank": False, "user_missing": True}
    assert list(validation.violations(ds, "s", rs)) == [False, True, False, True, False]


def test_validate_data_report_and_saved_variables():
    df = pd.DataFrame({"id": [1.0, 2, 2, np.nan, 5, np.nan],
                       "age": [25.0, 17, 40, 33, 150, np.nan],
                       "preg": [0.0, 1, 0, 1, 0, np.nan],
                       "sex": [1.0, 1, 2, 2, 1, np.nan]})
    ds = Dataset(df)
    validation.load_predefined(ds, ["0 to 1 Dichotomy", "Nonnegative integer"])
    r = validation.rules(ds)
    r["single"].append({"name": "Adult age", "type": "numeric", "domain": "range", "min": 18,
                        "max": 120, "noninteger": False, "user_missing": True,
                        "sysmis": True})
    r["cross"].append({"name": "Pregnant male", "expression": "preg = 1 AND sex = 1"})
    validation.set_rules(ds, r)
    sec = validation.validate_data(
        ds, ["age", "preg", "sex"], ["id"], basic={}, assignments={
            "age": ["Adult age"], "preg": ["0 to 1 Dichotomy"]},
        cross=["Pregnant male"], save={"empty": True, "incomplete": True, "duplicate": True,
                                       "total": True, "indicators": True})
    res = sec.result
    assert list(res["total"]) == [0, 2, 0, 0, 1, 0]
    assert list(ds.df["DuplicateIDGroup"]) == [0, 1, 1, 0, 0, 0]
    assert list(ds.df["IncompleteID"]) == [0, 0, 0, 1, 0, 1]
    assert list(ds.df["EmptyCase"]) == [0, 0, 0, 0, 0, 1]
    assert ds.has_name("Adult_age_age") and ds.has_name("Pregnant_male")
    titles = [getattr(i, "title", "") for i in sec.items]
    assert {"Rule Descriptions", "Variable Summary", "Cross-Variable Rules", "Case Report",
            "Identifier Checks"} <= set(titles)
    with pytest.raises(validation.ValidationError):
        validation.validate_data(ds, ["age"], save={"total": True})


def test_basic_checks():
    ds = Dataset(pd.DataFrame({"const": [5.0] * 10, "miss": [np.nan] * 8 + [1.0, 2.0],
                               "cat": [1.0] * 10}))
    ds.var("cat").measure = "nominal"
    sec = validation.validate_data(ds, ["const", "miss", "cat"], basic={})
    tab = next(i for i in sec.items if getattr(i, "title", "") == "Variable Checks")
    flagged = {r[0] for r in tab.rows}
    assert flagged == {"const", "miss", "cat"}


# ----------------------------------------------------------------------------- ADP
def test_boxcox_grid_matches_scipy_llf():
    r = np.random.default_rng(3)
    y = np.exp(r.normal(1, 0.5, 300))
    lam, z, c = adp.boxcox_grid(y)
    grid = np.arange(-3, 3.01, 0.5)
    best = grid[int(np.argmax([stats.boxcox_llf(g, y) for g in grid]))]
    assert c == 0 and lam == pytest.approx(best)


def test_prepare_data_steps():
    r = np.random.default_rng(4)
    n = 300
    x = r.normal(50, 10, n)
    x[:3] = [500, -400, 450]
    y = 2 * x + r.normal(0, 5, n)
    df = pd.DataFrame({"x": x, "noise": r.normal(size=n), "few": r.integers(1, 4, n) * 1.0,
                       "mostly": np.r_[np.ones(n - 5), np.zeros(5)],
                       "holes": np.where(r.random(n) < 0.6, np.nan, 1.0),
                       "cat": r.choice(["a", "b", "c", "d"], n, p=[.4, .3, .25, .05]),
                       "when": pd.Timestamp("2024-01-01") + pd.to_timedelta(
                           r.integers(0, 900, n), unit="D"), "y": np.exp(y / 100)})
    ds = Dataset(df)
    ds.var("mostly").measure = "nominal"
    sec = adp.prepare_data(ds, ["x", "noise", "few", "mostly", "holes", "cat", "when"], "y",
                           {"reference": dt.date(2027, 1, 1)})
    res = sec.result
    excluded = dict(res["excluded"])
    assert "holes" in excluded and "mostly" in excluded and "noise" in excluded
    assert ds.has_name("x_transformed") and ds.has_name("y_transformed")
    xt = ds.df["x_transformed"].to_numpy()
    assert abs(xt.mean()) < 1e-9 and np.nanstd(xt, ddof=1) == pytest.approx(1.0)
    assert np.abs(xt).max() < 10        # outliers trimmed before rescaling
    dated = set(res["fields"]) | set(excluded)
    assert {"when_year_transformed", "when_month_transformed"} & dated or \
        {"when_year", "when_month"} & dated
    assert "when_months" in dated or "when_months_transformed" in dated   # 196+ days
    assert ds.var("few_transformed").measure == "ordinal"
    sec2 = adp.prepare_data(ds, ["cat", "few"], None, {"sparse_pct": 10})
    codes = ds.df["cat_transformed"]
    assert codes.nunique() <= 4 and codes.min() == 0
