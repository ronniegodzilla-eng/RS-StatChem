"""WP3: Frequencies, Descriptives and Explore options.

References: statsmodels RLM with the scale fixed at MAD/.6745 (the SPSS EXAMINE M-estimator
definition: Huber 1.339, Tukey 4.685, Hampel 1.7/3.4/8.5, Andrews 1.339), scipy.stats.levene,
Tukey (1977) hinges, SPSS FREQUENCIES grouped-data percentile definition.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.stats import descriptive as D
from rsstatchem.core.stats import explore as E

SHOTS = os.environ.get(
    "WP3_SHOTS", "/private/tmp/claude-501/-Users-ronisaputra-Aplikasi-statistik/"
                 "a8ca05d2-7214-4d46-8f3d-8e0d4372bfcd/scratchpad/wp3")


def table(sec, title):
    return next(t for t in sec.items if getattr(t, "title", "").startswith(title))


def tables(sec, title):
    return [t for t in sec.items if getattr(t, "title", "") == title]


@pytest.fixture
def ds():
    rng = np.random.default_rng(4)
    n = 120
    g = rng.integers(1, 4, n).astype(float)
    y = rng.gamma(2 + g, 2.0)
    y[:3] = [60.0, 55.0, -5.0]
    score = rng.integers(1, 6, n).astype(float)
    d = Dataset(pd.DataFrame({"y": y, "z": rng.normal(10, 2, n), "g": g, "score": score,
                              "name": [f"c{i}" for i in range(n)]}))
    d.var("g").value_labels = {1.0: "Low", 2.0: "Mid", 3.0: "High"}
    return d


@pytest.mark.parametrize("kind", ["huber", "tukey", "hampel", "andrews"])
def test_m_estimators_match_rlm_fixed_scale(kind):
    import statsmodels.api as sm
    from statsmodels.robust import norms

    rng = np.random.default_rng(1)
    x = np.concatenate([rng.normal(10, 2, 60), [25, 30, -4]])
    norm = {"huber": norms.HuberT(1.339), "tukey": norms.TukeyBiweight(4.685),
            "hampel": norms.Hampel(1.7, 3.4, 8.5), "andrews": norms.AndrewWave(1.339)}[kind]
    scale = np.median(np.abs(x - np.median(x))) / 0.6744897501960817
    res = sm.RLM(x, np.ones((len(x), 1)), M=norm).fit(
        start_params=np.array([np.median(x)]), update_scale=False, conv="coefs", tol=1e-12,
        start_scale=scale, maxiter=500)
    assert E.m_estimator(x, kind) == pytest.approx(res.params[0], abs=1e-6)


def test_hinges_extremes_levene_stem():
    x = np.arange(1, 11, dtype=float)
    assert E.tukey_hinges(x) == (3.0, 5.5, 8.0)
    assert E.tukey_hinges(np.arange(1, 10, dtype=float)) == (3.0, 5.0, 7.0)
    ev = E.extreme_values(np.array([5, 9, 9, 1, 9, 3, 9, 9, 2.0]), np.arange(1, 10))
    assert [v for _, v, _ in ev["high"]] == [9, 9, 9, 9, 9]
    assert ev["high_partial"] is None
    ev = E.extreme_values(np.array([9, 9, 9, 9, 9, 9, 1.0]), np.arange(1, 8))
    assert ev["high_partial"] == 9.0
    rng = np.random.default_rng(3)
    samples = [rng.normal(0, s, 25) for s in (1, 2, 3)]
    rows = {r[0]: r for r in E.levene_examine(samples)}
    assert rows["Based on Mean"][1] == pytest.approx(stats.levene(*samples,
                                                                  center="mean").statistic)
    assert rows["Based on Median"][4] == pytest.approx(stats.levene(*samples,
                                                                    center="median").pvalue)
    assert rows["Based on Median and with adjusted df"][3] < 72
    sl = E.stem_and_leaf(np.concatenate([rng.normal(50, 8, 80), [120.0]]))
    assert sum(f for f, _, _ in sl["lines"]) == 81
    assert sl["lines"][-1][1] == "Extremes"


def test_spread_level_power():
    rng = np.random.default_rng(8)
    # spread proportional to level -> slope 1, power 0 (log transform)
    samples = [m * np.exp(rng.normal(0, 0.3, 400)) for m in (5, 20, 80)]
    sl = E.spread_level(samples)
    assert sl["slope"] == pytest.approx(1.0, abs=0.12)
    assert sl["power"] == pytest.approx(0.0, abs=0.12)


def test_explore_full(ds):
    sec = E.explore(ds, ["y", "z"], ["g"], m_estimators=True, outliers=True, percentiles=True,
                    label_by="name", stem_leaf=True, spread="estimate",
                    boxplot_mode="dependents", histogram=True)
    m = table(sec, "M-Estimators")
    assert len(m.rows) == 6 and len(m.footnotes) == 4
    pct = table(sec, "Percentiles")
    assert pct.rows[0][0] == "Weighted Average(Definition 1)"
    assert pct.rows[1][0] == "Tukey's Hinges"
    ext = table(sec, "Extreme Values")
    assert "name" in ext.columns and ext.rows[0][3] == "1"
    assert table(sec, "Test of Homogeneity of Variance").rows[2][1].startswith(
        "Based on Median and with adjusted df")
    assert any("Stem-and-Leaf" in getattr(t, "title", "") for t in sec.items)
    assert any("Detrended" in getattr(t, "title", "") for t in sec.items)
    sub = ds.df[ds.df.g == 1].y.to_numpy()
    assert float(m.rows[0][2]) == pytest.approx(E.m_estimator(sub, "huber"), abs=1e-3)
    # pairwise keeps more cases than listwise when another dependent is missing
    ds.df.loc[5:15, "z"] = np.nan
    lw = table(E.explore(ds, ["y", "z"], missing="listwise"), "Case Processing Summary")
    pw = table(E.explore(ds, ["y", "z"], missing="pairwise"), "Case Processing Summary")
    assert int(pw.rows[0][1]) > int(lw.rows[0][1])
    ds.df.loc[0:4, "g"] = np.nan
    rep = table(E.explore(ds, ["y"], ["g"], missing="report"), "Case Processing Summary")
    assert any("Missing" in r[1] for r in rep.rows)


def test_grouped_percentiles_and_frequencies(ds):
    # values 1..5 with counts 2,3,5,6,4 (N=20): median at 10 -> class 3 (2.5..3.5), cum 5
    assert D.grouped_percentile([1, 2, 3, 4, 5], [2, 3, 5, 6, 4], 50) == pytest.approx(3.5)
    assert D.grouped_percentile([1, 2, 3, 4, 5], [2, 3, 5, 6, 4], 25) == pytest.approx(2.5)
    spec = D.StatSpec(("quartiles", "median", "mean"),
                      {"percentiles": [10, 90], "cut_points": 5, "order": "dfreq"})
    sec = D.frequencies(ds, ["score", "z"], spec, chart="bar")
    st = table(sec, "Statistics")
    ps = [r[1] for r in st.rows if r[1].replace(".", "").isdigit()]
    assert ps == ["10", "20", "25", "40", "50", "60", "75", "80", "90"]
    x = ds.df.z.to_numpy()
    row10 = next(r for r in st.rows if r[1] == "10")
    assert float(row10[3]) == pytest.approx(np.percentile(x, 10, method="weibull"), abs=0.01)
    freq = table(sec, "score")
    counts = [int(r[2]) for r in freq.rows[:5]]
    assert counts == sorted(counts, reverse=True)
    grouped = D.frequencies(ds, ["score"], D.StatSpec(("median",), {"grouped": True}))
    assert table(grouped, "Statistics").footnotes == ["a. Calculated from grouped data."]
    by_var = D.frequencies(ds, ["score", "g"], ("mean",), organize="variable", limit=3)
    assert len(tables(by_var, "Statistics")) == 2
    assert not tables(by_var, "score")              # 5 categories > limit 3: suppressed
    pct = D.frequencies(ds, ["g"], (), chart="pie", chart_values="percentages")
    assert any(getattr(i, "kind", "") == "image" for i in pct.items)


def test_descriptives_display_order(ds):
    sec = D.descriptives(ds, ["z", "y", "score"], D.StatSpec(("mean",), {"order": "amean"}))
    rows = table(sec, "Descriptive Statistics").rows[:-1]
    means = [float(r[2]) for r in rows]
    assert means == sorted(means)
    sec = D.descriptives(ds, ["z", "y", "score"], ("mean",), order="alpha")
    assert [r[0] for r in table(sec, "Descriptive Statistics").rows[:-1]] == ["score", "y", "z"]


def test_descriptive_dialogs(qapp, ds, monkeypatch):
    from PySide6.QtWidgets import QDialog

    from rsstatchem.ui.dialogs.descriptive import DescriptivesDialog, FrequenciesDialog
    from rsstatchem.ui.dialogs.explore import ExploreDialog

    os.makedirs(SHOTS, exist_ok=True)
    shots = []

    def fake_exec(self):
        self.adjustSize()
        path = os.path.join(SHOTS, self.windowTitle().replace(": ", "_").replace(" ", "_")
                            + ".png")
        self.grab().save(path)
        shots.append(path)
        return 1

    fd = FrequenciesDialog(ds)
    fd._take("score", fd._targets[0])
    fd.grab().save(os.path.join(SHOTS, "frequencies.png"))
    monkeypatch.setattr(QDialog, "exec", fake_exec)
    fd._statistics()
    fd._charts()
    fd._format()
    p = fd.params()
    assert isinstance(p["stats"], D.StatSpec) and "order" in p["stats"].options
    # the Data Editor's call (data_editor.run_frequencies) works with the spec
    D.frequencies(ds, p["names"], p["stats"], p["show_tables"], p["chart"], p["normal_curve"])
    dd = DescriptivesDialog(ds)
    dd._take("y", dd._targets[0])
    dd._options()
    p = dd.params()
    D.descriptives(ds, p["names"], p["stats"])
    ed = ExploreDialog(ds)
    ed._take("y", ed._targets[0])
    ed._take("g", ed._targets[1])
    ed._take("name", ed._targets[2])
    ed._update_arrows()
    ed.grab().save(os.path.join(SHOTS, "explore.png"))
    ed._statistics()
    ed._plots()
    ed._options()
    ed.opts.update(m_estimators=True, outliers=True, percentiles=True, spread="transformed")
    sec = ed.compute()
    assert table(sec, "Extreme Values")
    assert len(shots) == 7
