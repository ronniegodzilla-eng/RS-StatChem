"""WP11 Quality Control: SPC constants against the ASTM / Montgomery tables (Introduction to
Statistical Quality Control, Appendix VI), IBM SPCHART limit formulas (equal and unequal
subgroups), the control rules, capability indices and the Pareto chart."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.stats import qc

# n: A2, A3, c4, B3, B4, d2, d3, D3, D4  (Montgomery, Appendix VI)
MONTGOMERY = {
    2: (1.880, 2.659, 0.7979, 0.000, 3.267, 1.128, 0.853, 0.000, 3.267),
    3: (1.023, 1.954, 0.8862, 0.000, 2.568, 1.693, 0.888, 0.000, 2.574),
    4: (0.729, 1.628, 0.9213, 0.000, 2.266, 2.059, 0.880, 0.000, 2.282),
    5: (0.577, 1.427, 0.9400, 0.000, 2.089, 2.326, 0.864, 0.000, 2.114),
    6: (0.483, 1.287, 0.9515, 0.030, 1.970, 2.534, 0.848, 0.000, 2.004),
    7: (0.419, 1.182, 0.9594, 0.118, 1.882, 2.704, 0.833, 0.076, 1.924),
    8: (0.373, 1.099, 0.9650, 0.185, 1.815, 2.847, 0.820, 0.136, 1.864),
    10: (0.308, 0.975, 0.9727, 0.284, 1.716, 3.078, 0.797, 0.223, 1.777),
    25: (0.153, 0.606, 0.9896, 0.565, 1.435, 3.931, 0.708, 0.459, 1.541),
}


@pytest.mark.parametrize("n", sorted(MONTGOMERY))
def test_constants_match_montgomery(n):
    c = qc.constants(n)
    exp = MONTGOMERY[n]
    for key, val, tol in zip(("A2", "A3", "c4", "B3", "B4", "d2", "d3", "D3", "D4"), exp,
                             (6e-4, 6e-4, 6e-5, 6e-4, 6e-4, 6e-4, 6e-4, 6e-4, 6e-4)):
        assert c[key] == pytest.approx(val, abs=tol), key


def _groups(seed=1, k=25, n=5):
    r = np.random.default_rng(seed)
    return [r.normal(74.0, 0.01, n) for _ in range(k)]


def test_xbar_r_equal_sizes_uses_A2_D3_D4():
    g = _groups()
    labels = [str(i) for i in range(len(g))]
    cx, cr = qc.xbar_r(g, labels)
    xbb = np.mean([x.mean() for x in g])
    rbar = np.mean([x.max() - x.min() for x in g])
    A2, D3, D4 = qc.constants(5)["A2"], qc.constants(5)["D3"], qc.constants(5)["D4"]
    assert cx.cl[0] == pytest.approx(xbb)
    assert cx.ucl[0] == pytest.approx(xbb + A2 * rbar)
    assert cx.lcl[0] == pytest.approx(xbb - A2 * rbar)
    assert cr.ucl[0] == pytest.approx(D4 * rbar) and cr.lcl[0] == pytest.approx(D3 * rbar)


def test_xbar_s_equal_and_unequal_sizes():
    g = _groups(2)
    cx, cs = qc.xbar_s(g, [str(i) for i in range(25)])
    sbar = np.mean([x.std(ddof=1) for x in g])
    c = qc.constants(5)
    assert cx.ucl[0] - cx.cl[0] == pytest.approx(c["A3"] * sbar)
    assert cs.ucl[0] == pytest.approx(c["B4"] * sbar)
    # unequal sizes: sigma-hat = mean(S_i / c4(n_i)); limits x̄ ± 3σ̂/√n_i (IBM)
    r = np.random.default_rng(3)
    g2 = [r.normal(10, 2, n) for n in (3, 4, 5, 6, 4, 5, 3, 7)]
    cx2, cs2 = qc.xbar_s(g2, [str(i) for i in range(8)])
    sig = np.mean([x.std(ddof=1) / qc.c4(len(x)) for x in g2])
    n = np.array([len(x) for x in g2])
    grand = np.concatenate(g2).mean()
    assert np.allclose(cx2.ucl, grand + 3 * sig / np.sqrt(n))
    assert np.allclose(cs2.cl, sig * np.array([qc.c4(k) for k in n]))


def test_r_chart_unequal_sizes():
    r = np.random.default_rng(4)
    g = [r.normal(0, 1, n) for n in (2, 5, 3, 8, 4)]
    cx, cr = qc.xbar_r(g, list("abcde"))
    sig = np.mean([(x.max() - x.min()) / qc.d2(len(x)) for x in g])
    for i, x in enumerate(g):
        assert cr.cl[i] == pytest.approx(sig * qc.d2(len(x)))
        assert cr.ucl[i] == pytest.approx(sig * qc.d2(len(x)) + 3 * sig * qc.d3(len(x)))


def test_individuals_moving_range():
    y = np.array([10.0, 12, 11, 13, 9, 10, 14, 12, 11, 10])
    ci, cm = qc.individuals(y, [str(i) for i in range(10)])
    mr = np.abs(np.diff(y)).mean()
    assert ci.ucl[0] == pytest.approx(y.mean() + 2.66 * mr, abs=2e-3 * mr)   # 3/d2(2)
    assert cm.ucl[0] == pytest.approx(3.267 * mr, abs=1e-3 * mr)
    ci3, cm3 = qc.individuals(y, [str(i) for i in range(10)], span=3)
    assert np.isnan(cm3.values[:2]).all()
    assert cm3.values[2] == pytest.approx(2.0)


def test_attribute_charts():
    x = np.array([3.0, 5, 2, 4, 6, 1])
    n = np.array([50.0, 50, 40, 60, 50, 50])
    p = qc.p_chart(x, n, list("abcdef"))
    pbar = x.sum() / n.sum()
    assert p.cl[0] == pytest.approx(pbar)
    assert np.allclose(p.ucl, np.minimum(pbar + 3 * np.sqrt(pbar * (1 - pbar) / n), 1))
    npc = qc.p_chart(x, n, list("abcdef"), np_chart=True)
    assert np.allclose(npc.cl, n * pbar)
    c = qc.c_chart(x, np.ones(6), list("abcdef"))
    assert c.ucl[0] == pytest.approx(x.mean() + 3 * math.sqrt(x.mean()))
    assert c.lcl[0] == 0.0
    u = qc.c_chart(x, n, list("abcdef"), u_chart=True)
    ub = x.sum() / n.sum()
    assert np.allclose(u.ucl, ub + 3 * np.sqrt(ub / n))


def test_control_rules():
    labels = [str(i) for i in range(30)]
    base = np.zeros(30)
    ch = qc.Chart("I", "v", labels, base.copy(), np.zeros(30), np.full(30, 3.0),
                  np.full(30, -3.0))
    v = base.copy()
    v[3] = 3.5                       # beyond UCL
    v[10:12] = [2.5, 2.5]            # 2 of 3 beyond +2 sigma
    v[20:28] = 0.5                   # 8 above the centre line
    ch.values = v
    out = qc.apply_rules(ch, list(qc.RULES))
    assert qc.RULES["ucl"] in out[3]
    assert qc.RULES["r2upper"] in out[11]
    assert qc.RULES["r8upper"] in out[27]
    trend = qc.Chart("I", "v", labels[:8], np.arange(8.0) * 0.1, np.zeros(8), np.full(8, 3.0),
                     np.full(8, -3.0))
    assert qc.RULES["trup"] in qc.apply_rules(trend, ["trup"])[5]
    alt = qc.Chart("I", "v", labels[:14], np.tile([0.1, -0.1], 7), np.zeros(14),
                   np.full(14, 3.0), np.full(14, -3.0))
    assert 13 in qc.apply_rules(alt, ["alternating"])


def test_capability_indices_ibm_formulas():
    r = np.random.default_rng(5)
    x = r.normal(10.2, 0.5, 200)
    sig = 0.45
    cap = qc.capability(x, sig, 8.5, 11.5, 10.0)
    m, s = x.mean(), x.std(ddof=1)
    assert cap["CP"] == pytest.approx(3 / (6 * sig))
    assert cap["CpK"] == pytest.approx(min((m - 8.5) / (3 * sig), (11.5 - m) / (3 * sig)))
    assert cap["CpM"] == pytest.approx(3 / (6 * math.sqrt(sig ** 2 + (m - 10) ** 2)))
    assert cap["K"] == pytest.approx(2 * abs(10 - m) / 3)
    assert cap["PpK"] == pytest.approx(min((m - 8.5) / (3 * s), (11.5 - m) / (3 * s)))
    assert cap["CZout"] == pytest.approx(100 * (1 - qc.stats.norm.cdf((11.5 - m) / sig)
                                                + qc.stats.norm.cdf(-(m - 8.5) / sig)))
    one = qc.capability(x, sig, None, 11.5)
    assert one["CpK"] == pytest.approx((11.5 - m) / (3 * sig)) and one["CP"] is None
    g = [x[i:i + 5] for i in range(0, 200, 5)]
    assert qc.capability_sigma(g, "rbar") == pytest.approx(
        np.mean([(v.max() - v.min()) / qc.d2(5) for v in g]))
    pooled = math.sqrt(sum(((v - v.mean()) ** 2).sum() for v in g) / (len(g) * 4))
    assert qc.capability_sigma(g, "variance") == pytest.approx(pooled)


def test_procedure_and_pareto():
    r = np.random.default_rng(6)
    df = pd.DataFrame({"x": r.normal(5, 1, 100), "sub": np.repeat(np.arange(20), 5) + 1.0,
                       "defect": r.choice(["scratch", "dent", "chip", "stain"], 100,
                                          p=[.5, .25, .15, .1]),
                       "cost": r.uniform(1, 3, 100)})
    ds = Dataset(df)
    sec = qc.control_chart(ds, "xbar_r", ["x"], "sub", rules=["ucl", "lcl"],
                           capability_stats=True, lsl=2, usl=8)
    titles = [getattr(i, "title", "") for i in sec.items]
    assert "Process Statistics" in titles and "Control Limits" in titles
    s2 = qc.control_chart(ds, "xbar_s", samples=["x", "cost"])
    assert len(s2.result[0].values) == 100
    sec = qc.pareto(ds, "defect")
    tab = next(i for i in sec.items if getattr(i, "title", "") == "Pareto Table")
    assert tab.rows[0][0] == "scratch" and tab.rows[-2][3] == "100.0"
    sec = qc.pareto(ds, "defect", "cost", stack=None)
    assert sec.items
