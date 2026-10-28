"""WP11 Orthogonal Design (ORTHOPLAN), Display Design and Conjoint Analysis: every prepared plan
is orthogonal, plan sizes follow IBM's decision rules, CONJOINT part-worths equal OLS with
deviation coding (statsmodels), IBM's SE / importance / simulation formulas."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats

from rsstatchem.core.stats import conjoint as cj


def test_every_library_plan_is_orthogonal():
    for plan in cj.library():
        assert cj.is_orthogonal(np.column_stack(plan.columns)), plan.name


def test_galois_fields_are_fields():
    for q in (4, 8, 9):
        F = cj.GF(q)
        for a in range(1, q):       # every nonzero element has an inverse
            assert 1 in F.mul[a, 1:]
        assert all(F.add[a, 0] == a for a in range(q))


@pytest.mark.parametrize("levels,runs", [
    ([2] * 5, 8), ([2] * 7, 8), ([2] * 11, 12), ([2] * 19, 20), ([3] * 4, 9),
    ([4, 2, 2, 2], 8),           # IBM Table 73-5: 8 rows, one 4-level + four 2-level columns
    ([3, 3, 2, 2, 2], 16),       # IBM L = 3 rule: K = F + 2P < 16 -> Table 73-6 (16 rows)
    ([4, 4, 4, 4, 4], 16), ([5, 5, 5], 25), ([3] * 7, 18), ([7, 7, 3], 49),
    ([8, 2, 2, 2], 16),          # IBM Table 73-9: 16 rows, one 8-level + eight 2-level
    ([9, 3, 3, 3], 27),          # IBM Table 73-12: 27 rows, one 9-level + nine 3-level
])
def test_plan_sizes(levels, runs):
    plan, assign = cj.choose_plan(levels)
    assert plan.runs == runs
    assert all(plan.levels[c] >= k for c, k in zip(assign, levels))


def test_minimum_cards_and_holdouts():
    f = [cj.Factor("a", [1, 2, 3]), cj.Factor("b", [1, 2, 3]), cj.Factor("c", [1, 2])]
    df, name = cj.orthoplan(f, minimum=12, holdout=4, seed=5)
    design = df[df["STATUS_"] == 0]
    assert len(design) >= 12
    codes = np.column_stack([pd.factorize(design[c])[0] for c in "abc"])
    assert cj.is_orthogonal(codes)
    hold = df[df["STATUS_"] == 1]
    assert len(hold) == 4
    used = {tuple(r) for r in design[list("abc")].to_numpy().tolist()}
    assert not any(tuple(r) in used for r in hold[list("abc")].to_numpy().tolist())
    df2, _ = cj.orthoplan(f, minimum=12, holdout=4, seed=5)
    pd.testing.assert_frame_equal(df, df2)            # seeded
    assert list(df["CARD_"]) == list(range(1, len(df) + 1))


def test_generate_and_display_design():
    f = [cj.Factor("price", [1.19, 1.39, 1.59], ["$1.19", "$1.39", "$1.59"], "Price"),
         cj.Factor("seal", [1, 2], ["no", "yes"])]
    ds, sec = cj.generate_design(f, holdout=2, seed=1)
    assert ds.var("seal").value_labels == {1: "no", 2: "yes"}
    assert ds.var("STATUS_").value_labels[1.0] == "Holdout"
    out = cj.display_design(ds, ["price", "seal"], True, True, "Card", "Rate 1-10")
    assert sum(1 for i in out.items if getattr(i, "title", "").startswith("Card")) == ds.n_cases


def _deviation_matrix(plan, factors):
    cols = [np.ones(len(plan))]
    for f in factors:
        lv = sorted(plan[f].unique())
        for v in lv[:-1]:
            cols.append(np.where(plan[f] == v, 1.0, np.where(plan[f] == lv[-1], -1.0, 0.0)))
    return np.column_stack(cols)


def _plan():
    f = [cj.Factor("pack", [1, 2, 3]), cj.Factor("brand", [1, 2, 3]),
         cj.Factor("price", [1.19, 1.39, 1.59]), cj.Factor("seal", [1, 2]),
         cj.Factor("money", [1, 2])]
    df, _ = cj.orthoplan(f, seed=2)
    return df


def test_utilities_equal_ols_with_deviation_coding():
    plan = _plan()
    r = np.random.default_rng(1)
    t = 6
    Y = np.array([plan["pack"].map({1: 2.0, 2: 0.5, 3: -1}).to_numpy()
                  - 3 * plan["price"].to_numpy() + plan["seal"].to_numpy()
                  + r.normal(scale=0.4, size=len(plan)) + 10 for _ in range(t)])
    models = [cj.FactorModel("pack"), cj.FactorModel("brand"),
              cj.FactorModel("price", "linear", "less"), cj.FactorModel("seal"),
              cj.FactorModel("money")]
    sec = cj.conjoint(plan, models, Y, "score", plots=False)
    res = sec.result
    # summary utilities = OLS on the stacked subjects (statsmodels)
    Xd = _deviation_matrix(plan, ["pack", "brand", "seal", "money"])
    X = np.column_stack([Xd[:, :1], Xd[:, 1:3], Xd[:, 3:5],
                         plan["price"] - plan["price"].mean(), Xd[:, 5:]])
    Xs = np.vstack([X] * t)
    fit = sm.OLS(Y.reshape(-1), Xs).fit()
    util = {(f, lv): u for f, lv, u, se in res["utilities"]}
    se = {(f, lv): s for f, lv, u, s in res["utilities"]}
    assert util[("pack", 1)] == pytest.approx(fit.params[1])
    assert util[("pack", 3)] == pytest.approx(-(fit.params[1] + fit.params[2]))
    assert se[("pack", 1)] == pytest.approx(fit.bse[1])
    cov = fit.cov_params()
    assert se[("pack", 3)] == pytest.approx(np.sqrt(cov[1, 1] + cov[2, 2] + 2 * cov[1, 2]))
    b = fit.params[5]
    assert util[("price", 1.39)] == pytest.approx(b * 1.39)
    assert res["constant"] == pytest.approx(fit.params[0] - b * plan["price"].mean())
    assert res["sigma2"] == pytest.approx(fit.scale)
    # importance = mean over subjects of 100 * range / sum of ranges
    imp = res["importance"]
    assert sum(imp.values()) == pytest.approx(100.0)
    corr = next(i for i in sec.items if getattr(i, "title", "").startswith("Correlations"))
    pr = stats.pearsonr(Y.mean(0), res["predicted"])[0]
    assert float(corr.rows[0][1]) == pytest.approx(pr, abs=5e-4)


def test_ideal_factor_and_reversals():
    plan = _plan()
    z = plan["price"].to_numpy()
    y = -((z - 1.39) ** 2) * 50 + plan["seal"].to_numpy()
    models = [cj.FactorModel("price", "ideal"), cj.FactorModel("seal")]
    sec = cj.conjoint(plan, models, np.vstack([y, y + 0.1]), "score", plots=False)
    g1, g2 = sec.result["coefs"]["price"]
    assert g2 == pytest.approx(-50, rel=1e-6) and g1 == pytest.approx(2 * 50 * 1.39, rel=1e-6)
    rev = next(i for i in sec.items if getattr(i, "title", "") == "Number of Reversals")
    assert rev.rows == [["price", "0"]]


def test_rank_and_sequence_conversion_and_simulations():
    plan = _plan()
    n = len(plan)
    score = np.arange(n, 0, -1, dtype=float)              # card 1 best
    ranks = np.arange(1, n + 1, dtype=float)               # rank of each card
    seq = np.arange(1, n + 1, dtype=float)                 # card numbers best -> worst
    s1 = cj.responses_to_scores(score[None], "score", n)
    s2 = cj.responses_to_scores(ranks[None], "rank", n)
    s3 = cj.responses_to_scores(seq[None], "sequence", n)
    assert np.allclose(s1, s2) and np.allclose(s2, s3)
    sims = pd.DataFrame({"pack": [1, 3], "brand": [1, 1], "price": [1.19, 1.59],
                         "seal": [2, 1], "money": [2, 1], "STATUS_": [2.0, 2.0],
                         "CARD_": [n + 1.0, n + 2.0]})
    full = pd.concat([plan, sims], ignore_index=True)
    models = [cj.FactorModel(f) for f in ("pack", "brand", "price", "seal", "money")]
    sec = cj.conjoint(full, models, np.vstack([s1[0], s1[0][::-1]]), "score", plots=False)
    tab = next(i for i in sec.items if getattr(i, "title", "").startswith("Preference"))
    mx = [float(r[2].rstrip("%")) for r in tab.rows]
    assert sum(mx) == pytest.approx(100.0)
