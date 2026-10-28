"""WP11 Propensity Score Matching (logistic propensity = statsmodels Logit, greedy nearest
neighbour with caliper as MatchIt's nearest method), Case Control Matching, Rake Weights
(iterative proportional fitting) and the Direct Marketing procedures."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.stats import marketing, matching


def _psm_data(n=300, seed=0):
    r = np.random.default_rng(seed)
    x1 = r.normal(size=n)
    x2 = r.uniform(0, 1, n)
    p = 1 / (1 + np.exp(-(-0.8 + 1.0 * x1 + 0.8 * x2)))
    t = (r.random(n) < p).astype(float)
    return Dataset(pd.DataFrame({"t": t, "x1": x1, "x2": x2,
                                 "id": np.arange(1, n + 1, dtype=float) * 10}))


def _greedy_reference(st, sc, tol, order):
    """Independent greedy nearest-neighbour matching without replacement."""
    used = set()
    out = {}
    for i in order:
        best, bd = None, None
        for j in range(len(sc)):
            if j in used:
                continue
            d = abs(sc[j] - st[i])
            if d <= tol + 1e-12 and (bd is None or d < bd):
                best, bd = j, d
        if best is not None:
            used.add(best)
            out[i] = best
    return out


def test_propensity_scores_equal_statsmodels_logit():
    ds = _psm_data()
    sec = matching.propensity_matching(ds, "t", ["x1", "x2"], "ps", 0.05, balance=False)
    X = sm.add_constant(ds.df[["x1", "x2"]].to_numpy())
    ref = sm.Logit(ds.df["t"].to_numpy(), X).fit(disp=0).predict(X)
    assert np.allclose(ds.df["ps"].to_numpy(), ref, atol=1e-6)
    assert sec.items


@pytest.mark.parametrize("order", ["data", "largest", "smallest"])
def test_nearest_neighbour_caliper_matches_reference(order):
    ds = _psm_data(seed=1)
    sec = matching.propensity_matching(ds, "t", ["x1", "x2"], "ps", 0.03, case_id="id",
                                       order=order, exact_first=False, logistic_output=False)
    res = sec.result
    ps = res["ps"]
    st, sc = ps[res["treat_rows"]], ps[res["ctrl_rows"]]
    idx = np.arange(len(st))
    if order == "largest":
        idx = idx[np.argsort(-st, kind="mergesort")]
    elif order == "smallest":
        idx = idx[np.argsort(st, kind="mergesort")]
    ref = _greedy_reference(st, sc, 0.03, idx)
    got = {i: m[0] for i, m in enumerate(res["matches"]) if m}
    assert got == ref
    for i, j in got.items():                         # tolerance respected, IDs written
        assert abs(st[i] - sc[j]) <= 0.03 + 1e-12
        assert ds.df["MatchID"].iloc[res["treat_rows"][i]] == ds.df["id"].iloc[
            res["ctrl_rows"][j]]
    assert len(set(got.values())) == len(got)        # no reuse without replacement


def test_matching_with_replacement_ratio_and_balance():
    ds = _psm_data(seed=2)
    sec = matching.propensity_matching(ds, "t", ["x1", "x2"], "ps", 0.1, replace=True,
                                       ratio=2, eligible_name="nelig")
    res = sec.result
    sizes = [len(m) for m in res["matches"]]
    assert max(sizes) == 2 and np.mean(np.array(sizes) == 2) > 0.8
    ps = res["ps"]
    for i, m in enumerate(res["matches"]):         # with replacement: the nearest controls
        d = np.abs(ps[res["ctrl_rows"]] - ps[res["treat_rows"][i]])
        if len(m) == 2:
            assert sorted(d[m]) == pytest.approx(sorted(np.sort(d)[:2]))
    bal = next(i for i in sec.items if getattr(i, "title", "") == "Balance Diagnostics")
    before, after = float(bal.rows[1][3]), float(bal.rows[1][5])
    assert abs(after) < abs(before)
    t = ds.df["t"] == 1
    x = ds.df["x1"]
    smd = (x[t].mean() - x[~t].mean()) / x[t].std(ddof=1)
    assert before == pytest.approx(smd, abs=1e-4)
    assert ds.df.loc[t, "nelig"].notna().all()


def test_nearest_match_exact_priority_and_logit_scale():
    m = matching.nearest_match(np.array([0.30, 0.31]), np.array([0.31, 0.305]), 0.02,
                               exact_first=True)
    assert m == [[1], [0]]                            # the exact 0.31 pair goes first
    m2 = matching.nearest_match(np.array([0.30, 0.31]), np.array([0.31, 0.305]), 0.02,
                                exact_first=False)
    assert m2 == [[1], [0]] or m2 == [[0], [1]]


def test_case_control_matching_exact_and_fuzzy():
    df = pd.DataFrame({"case": [1, 1, 0, 0, 0, 0, 1.0], "sex": [1, 2, 1, 2, 2, 1, 1.0],
                       "age": [40, 50, 42, 49, 60, 39, 70.0]})
    ds = Dataset(df)
    sec = matching.case_control_matching(ds, "case", ["sex", "age"], {"age": 2},
                                         randomize=False)
    g = ds.df["MatchGroup"].to_numpy()
    assert g[0] == g[2] or g[0] == g[5]               # 40 ↔ 42 or 39 within 2 years
    assert g[1] == g[3]                               # 50 ↔ 49, same sex
    assert np.isnan(g[6])                             # age 70 has no control
    summ = next(i for i in sec.items if getattr(i, "title", "") == "Match Summary")
    assert summ.rows[1] == ["Cases matched", "2"]


def test_rake_one_and_two_dimensions():
    sex = np.array([0, 0, 0, 1])
    w, it, _ = matching.rake([sex], [{0: 50.0, 1: 50.0}])
    assert np.allclose(w, [50 / 3] * 3 + [50])        # closed form for one margin
    r = np.random.default_rng(3)
    a, b = r.integers(0, 3, 500), r.integers(0, 2, 500)
    ta = {0: 100.0, 1: 250.0, 2: 150.0}
    tb = {0: 300.0, 1: 200.0}
    w, it, change = matching.rake([a, b], [ta, tb], tol=1e-12)
    for k, v in ta.items():
        assert w[a == k].sum() == pytest.approx(v)
    for k, v in tb.items():
        assert w[b == k].sum() == pytest.approx(v)
    # raking = IPF of the a×b table: cell weights are proportional within each cell
    for i in range(3):
        for j in range(2):
            cell = w[(a == i) & (b == j)]
            assert np.allclose(cell, cell[0])


def test_rake_weights_procedure():
    r = np.random.default_rng(4)
    ds = Dataset(pd.DataFrame({"sex": r.choice([1.0, 2.0], 200, p=[.7, .3]),
                               "age": r.choice([1.0, 2.0, 3.0], 200)}))
    sec = matching.rake_weights(ds, [("sex", {1.0: 0.5, 2.0: 0.5}),
                                     ("age", {1.0: 0.2, 2.0: 0.3, 3.0: 0.5})])
    w = ds.df["rakewt"]
    assert w.sum() == pytest.approx(200)
    assert w[ds.df["sex"] == 1].sum() == pytest.approx(100, abs=1e-3)
    assert w[ds.df["age"] == 3].sum() == pytest.approx(100, abs=1e-3)
    assert matching.parse_targets("1=0.4; 2=0.6") == {1.0: 0.4, 2.0: 0.6}
    with pytest.raises(matching.MatchError):
        matching.rake_weights(ds, [("sex", {1.0: 1.0})], weight_name="w2")


# ----------------------------------------------------------------------------- marketing
def test_ntile_matches_spss_rank_ntiles():
    v = np.array([5.0, 1, 3, 3, 9, 7, 2, 8])
    r = stats.rankdata(v)
    assert list(marketing.ntile(v, 4)) == list(np.floor(r * 4 / 9) + 1)


def test_rfm_nested_and_independent():
    r = np.random.default_rng(5)
    rec, fr, mo = r.integers(1, 400, 500), r.poisson(5, 500), r.gamma(2, 50, 500)
    sc = marketing.rfm_scores(rec, fr, mo, (5, 5, 5), nested=True)
    assert set(np.unique(sc[:, 0])) == {1, 2, 3, 4, 5}
    # most recent customers (fewest days) get recency score 5
    assert rec[sc[:, 0] == 5].max() <= rec[sc[:, 0] == 1].min()
    assert np.allclose(sc[:, 3], 100 * sc[:, 0] + 10 * sc[:, 1] + sc[:, 2])
    ind = marketing.rfm_scores(rec, fr, mo, (5, 5, 5), nested=False)
    assert np.allclose(ind[:, 1], marketing.ntile(fr, 5))


def test_rfm_from_transactions():
    tr = pd.DataFrame({"cid": [1.0, 1, 2, 3, 3, 3],
                       "date": pd.to_datetime(["2024-01-05", "2024-03-01", "2024-02-10",
                                               "2023-12-01", "2024-02-28", "2024-03-15"]),
                       "amt": [10.0, 20, 5, 1, 2, 3]})
    sec, new = marketing.rfm_transactions(Dataset(tr), "cid", "date", "amt",
                                          reference=dt.date(2024, 4, 1), bins=(2, 2, 2))
    assert list(new.df["Transaction_Count"]) == [2, 1, 3]
    assert list(new.df["Amount"]) == [30, 5, 6]
    assert new.df["Recency_Score"].iloc[2] == 2       # 3 is the most recent customer


def test_control_package_test_matches_statsmodels():
    from statsmodels.stats.proportion import proportions_ztest

    r = np.random.default_rng(6)
    camp = np.repeat([1.0, 2.0, 3.0], 200)
    resp = (r.random(600) < np.repeat([0.10, 0.16, 0.11], 200)).astype(float)
    ds = Dataset(pd.DataFrame({"camp": camp, "resp": resp}))
    sec = marketing.control_package_test(ds, "camp", 1.0, "resp")
    tab = sec.items[1]
    z, p = proportions_ztest([resp[camp == 2].sum(), resp[camp == 1].sum()], [200, 200])
    assert float(tab.rows[1][4]) == pytest.approx(z, abs=1e-3)
    assert float(tab.rows[1][6]) == pytest.approx(min(1, 2 * p), abs=1e-3)


def test_propensity_to_purchase_and_profiles():
    r = np.random.default_rng(7)
    n = 600
    x = r.normal(size=n)
    region = r.choice([1.0, 2.0, 3.0], n)
    y = (r.random(n) < 1 / (1 + np.exp(-(1.5 * x + (region == 3) * 1.0 - 1)))).astype(float)
    ds = Dataset(pd.DataFrame({"y": y, "x": x, "region": region, "zip": r.integers(10, 30, n)
                               * 1000.0}))
    ds.var("region").measure = "nominal"
    res = marketing.propensity_to_purchase(ds, "y", [1.0], ["x", "region"], seed=1)
    assert res.model["auc"] > 0.75 and "Propensity_Score" in res.saved
    # without validation the coefficients are the logistic ML estimates (statsmodels)
    full = marketing.propensity_to_purchase(ds, "y", [1.0], ["x", "region"], validate=False)
    X = sm.add_constant(np.column_stack([x, region == 1, region == 2]).astype(float))
    ref = sm.Logit(y, X).fit(disp=0).params
    assert np.allclose(full.model["beta"], ref, atol=1e-6)
    sec = marketing.prospect_profiles(ds, "y", [1.0], ["x", "region"], min_group=50)
    prof = sec.result
    assert prof[0][0] > prof[-1][0] and sum(p[1] for p in prof) == n
    sec2, new = marketing.postal_code_rates(ds, "y", [1.0], "zip", digits=1)
    assert set(new.df["PostalCode"]) == {"1", "2"}
    assert new.df["Contacts"].sum() == n
