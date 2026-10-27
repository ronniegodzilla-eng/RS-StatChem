"""WP3: Ratio Statistics (SPSS RATIO STATISTICS).

Formulas from the IAAO Standard on Ratio Studies (2013): COD = 100·AAD/median with
AAD = mean|r − median|; PRD = mean ratio / weighted mean (Σ A / Σ S); COV = 100·SD/centre;
PRB = OLS slope of (r − median)/median on log2(0.5·A/median + 0.5·S).
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.stats import ratio as R

SHOTS = ("/private/tmp/claude-501/-Users-ronisaputra-Aplikasi-statistik/"
         "a8ca05d2-7214-4d46-8f3d-8e0d4372bfcd/scratchpad/wp3")

# assessed value (A) and sale price (S) in thousands
A = np.array([90.0, 110, 95, 130, 80, 150, 70, 120, 100, 85])
S = np.array([100.0, 100, 110, 120, 100, 140, 90, 100, 105, 95])


def test_ratio_formulas_by_hand():
    r = A / S
    res = R.ratio_stats(A, S, 0.95, 0.9, 1.1, 10)
    med = np.median(r)
    assert res["median"] == pytest.approx(med)
    assert res["mean"] == pytest.approx(r.mean())
    assert res["wmean"] == pytest.approx(A.sum() / S.sum())
    aad = np.mean(np.abs(r - med))
    assert res["aad"] == pytest.approx(aad)
    assert res["cod"] == pytest.approx(100 * aad / med)
    assert res["prd"] == pytest.approx(r.mean() / (A.sum() / S.sum()))
    assert res["mncov"] == pytest.approx(100 * r.std(ddof=1) / r.mean())
    assert res["mdcov"] == pytest.approx(100 * math.sqrt(((r - med) ** 2).sum() / 9) / med)
    assert res["between"] == pytest.approx(100 * np.mean((r >= 0.9) & (r <= 1.1)))
    assert res["within"] == pytest.approx(100 * np.mean(np.abs(r - med) <= 0.1 * med + 1e-15))
    lo, hi = res["mean_ci"]
    t = stats.t.ppf(0.975, 9) * r.std(ddof=1) / math.sqrt(10)
    assert (lo, hi) == pytest.approx((r.mean() - t, r.mean() + t))


def test_median_interval_and_weighted_mean_ci():
    lo, hi, cov = R.median_ci(np.arange(1.0, 21.0), 0.95)
    assert (lo, hi) == (6.0, 15.0) and cov == pytest.approx(0.9586, abs=1e-4)
    res = R.ratio_stats(A, S)
    w = A.sum() / S.sum()
    se = math.sqrt(((A - w * S) ** 2).sum() / (10 * 9)) / S.mean()
    t = stats.t.ppf(0.975, 9)
    assert res["wmean_ci"] == pytest.approx((w - t * se, w + t * se))


def test_prb_matches_ols():
    r = A / S
    med = np.median(r)
    y = (r - med) / med
    x = np.log(0.5 * A / med + 0.5 * S) / np.log(2)
    fit = sm.OLS(y, sm.add_constant(x)).fit()
    p = R.prb(A, S, med)
    assert p["value"] == pytest.approx(fit.params[1])
    assert p["se"] == pytest.approx(fit.bse[1])
    assert p["ci"] == pytest.approx(tuple(fit.conf_int()[1]))


def test_ratio_section_grouped():
    g = np.array([1.0, 2] * 5)
    a = np.append(A, np.nan)
    ds = Dataset(pd.DataFrame({"a": a, "s": np.append(S, 1.0), "g": np.append(g, 1.0)}))
    sec = R.ratio(ds, "a", "s", "g", "descending", R.STATISTICS, ci=True, low=0.9, high=1.1,
                  within=10)
    cps = next(t for t in sec.items if getattr(t, "title", "") == "Case Processing Summary")
    assert cps.rows[0][1] == "2" and cps.rows[-2][2] == "1"  # descending; 1 excluded
    t = next(t for t in sec.items if getattr(t, "title", "").startswith("Ratio Statistics"))
    assert t.columns[2:] == ["2", "1", "Overall"]
    row = {(a_ or b_): r for a_, b_, *r in [x[:2] + x[2:] for x in t.rows]}
    assert float(row["Mean"][2]) == pytest.approx((A / S).mean(), abs=5e-4)
    assert row["Coefficient of Dispersion"][2].endswith("%")
    assert any(r[0] == "Price Related Bias" for r in t.rows)


def test_ratio_dialog(qapp):
    from rsstatchem.ui.dialogs.distplots import RatioDialog

    os.makedirs(SHOTS, exist_ok=True)
    ds = Dataset(pd.DataFrame({"assessed": A, "price": S, "g": [1.0, 2] * 5}))
    dlg = RatioDialog(ds)
    for key, name in (("num", "assessed"), ("den", "price"), ("group", "g")):
        dlg._take(name, next(t for t in dlg._targets if t.key == key))
    dlg._update_arrows()
    assert dlg.ok_button.isEnabled() and dlg.validate() is None
    dlg.grab().save(os.path.join(SHOTS, "ratio.png"))
    dlg.opts.update(ci=True, stats=list(R.STATISTICS))
    sec = dlg.compute()
    assert any(getattr(t, "title", "") == "Ratio Statistics for assessed / price"
               for t in sec.items)
