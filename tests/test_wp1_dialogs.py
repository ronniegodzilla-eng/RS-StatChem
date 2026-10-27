"""WP1 dialogs: build each one headlessly, fill the targets/options and run compute()."""
import numpy as np
import pandas as pd
import pytest

from rsstatchem.core.dataset import Dataset
from rsstatchem.ui import menus


@pytest.fixture
def ds():
    rng = np.random.default_rng(2)
    n = 36
    df = pd.DataFrame({"y": rng.normal(10, 2, n), "y2": rng.normal(5, 1, n),
                       "a": np.repeat([1., 2., 3.], 12), "blk": np.tile([1., 2., 3., 4.], 9),
                       "x": rng.normal(size=n), "w": np.linspace(1, 2, n),
                       "pass": rng.integers(0, 2, n).astype(float),
                       "pass2": rng.integers(0, 2, n).astype(float),
                       "t1": rng.normal(size=n), "t2": rng.normal(size=n) + 1,
                       "t3": rng.normal(size=n) + 2})
    return Dataset(df)


def _fill(dlg, key, *names):
    target = next(t for t in dlg._targets if t.key == key)
    for nm in names:
        assert dlg._take(nm, target)
    dlg._update_arrows()


def _titles(result):
    sec = getattr(result, "section", result)
    return [getattr(i, "title", "") for i in sec.items]


def test_univariate_dialog_custom_model(qapp, ds):
    from rsstatchem.ui.dialogs.compare import UnivariateDialog

    dlg = UnivariateDialog(ds)
    _fill(dlg, "dep", "y")
    _fill(dlg, "factors", "a", "blk")
    _fill(dlg, "wls", "w")
    assert dlg.ok_button.isEnabled()
    dlg.opts.update({"terms": ["a", "blk"], "ss_type": 1, "posthoc": ["duncan", "dunnett"],
                     "posthoc_factors": ["a"], "emm": ["a"], "compare": True,
                     "contrasts": {"a": ["helmert", "last"]}, "save": ["PRE", "COO"],
                     "plots": [["a", "blk", None]], "power": True, "parameters": True})
    assert dlg.validate() is None
    res = dlg.compute()
    titles = _titles(res)
    assert "Tests of Between-Subjects Effects" in titles
    assert any(t.startswith("Homogeneous Subsets") for t in titles)
    assert any(t.startswith("Contrast Results") for t in titles)
    assert set(res.saved) == {"PRE", "COO"}
    tab = next(i for i in res.section.items if getattr(i, "title", "") ==
               "Tests of Between-Subjects Effects")
    assert tab.columns[1] == "Type I Sum of Squares"
    assert "a * blk" not in [r[0] for r in tab.rows]


def test_oneway_dialog(qapp, ds):
    from rsstatchem.ui.dialogs.compare import OneWayAnovaDialog

    dlg = OneWayAnovaDialog(ds)
    _fill(dlg, "deps", "y")
    _fill(dlg, "factor", "a")
    dlg.opts.update({"polynomial": True, "degree": 2, "contrasts": [[1, -1, 0]],
                     "posthoc": ["snk", "t3", "waller"], "effects": True})
    titles = _titles(dlg.compute())
    assert "Contrast Tests" in titles and "Multiple Comparisons" in titles


def test_proportions_dialogs(qapp, ds):
    from rsstatchem.ui.dialogs import proportions as P

    d = P.OneSampleProportionsDialog(ds)
    _fill(d, "vars", "pass")
    d.opts.update({"ci": list(P.pr.ONE_CI), "tests": list(P.pr.ONE_TEST)})
    assert "One-Sample Proportions Tests" in _titles(d.compute())
    d = P.IndependentProportionsDialog(ds)
    _fill(d, "vars", "pass")
    _fill(d, "group", "blk")
    d.opts["group_def"] = {"var": "blk", "groups": (1.0, 2.0), "cut": None}
    assert d.validate() is None
    assert "Independent-Samples Proportions Tests" in _titles(d.compute())
    d = P.PairedProportionsDialog(ds)
    d._add_pair(d._pairs[0], "pass", "pass2")
    assert "Paired-Samples Proportions Tests" in _titles(d.compute())
    s = P.SummaryTTestDialog()
    for k, v in {"n1": "12", "mean1": "5", "sd1": "1", "n2": "15", "mean2": "4", "sd2": "1.5"
                 }.items():
        s.fields[k].setText(v)
    s.accept()
    assert "Independent Samples Test" in _titles(s.compute())


def test_multivariate_and_repeated_dialogs(qapp, ds):
    from rsstatchem.ui.dialogs.glm import MultivariateDialog, RepeatedMeasuresDialog

    d = MultivariateDialog(ds)
    _fill(d, "deps", "y", "y2")
    _fill(d, "factors", "a")
    d.opts.update({"posthoc": ["tukey", "gt2"], "posthoc_factors": ["a"], "power": True,
                   "save": ["RES"], "contrasts": {"a": ["simple", "first"]}})
    res = d.compute()
    assert "Multivariate Tests" in _titles(res) and len(res.saved) == 2
    r = RepeatedMeasuresDialog(ds, factors=[("time", 3)])
    for nm in ("t1", "t2", "t3"):
        r._take(nm, r._within_target())
    _fill(r, "between", "a")
    r.opts.update({"within_contrasts": {"time": "repeated"}, "emm": ["time*a"],
                   "simple": True, "power": True, "posthoc": ["duncan"],
                   "posthoc_factors": ["a"]})
    assert r.validate() is None
    out = r.compute()
    titles = _titles(out)
    assert any("within" in t for t in titles)
    sec = getattr(out, "section", out)
    tc = next(i for i in sec.items if getattr(i, "title", "") ==
              "Tests of Within-Subjects Contrasts")
    assert any("Level 1 vs. Level 2" in row for row in tc.rows)


def test_mixed_and_varcomp_dialogs(qapp):
    from rsstatchem.ui.dialogs.glm import LinearMixedDialog, VarianceComponentsDialog

    rng = np.random.default_rng(4)
    long = pd.DataFrame({"id": np.repeat(np.arange(15.), 3), "t": np.tile([1., 2., 3.], 15),
                         "g": np.repeat([1., 2., 3.], 15),
                         "y": rng.normal(size=45) + np.tile([0., 1., 2.], 15)})
    ds = Dataset(long)
    d = LinearMixedDialog(ds)
    _fill(d, "subject", "id")
    _fill(d, "dep", "y")
    _fill(d, "factors", "t")
    d.opts.update({"rtype": "AR1", "repeated": "t", "intercept": False, "emm": ["t"],
                   "compare": True,
                   "save": ["PRED"]})
    assert d.validate() is None
    res = d.compute()
    assert any(t.startswith("Estimates — t") for t in _titles(res))
    v = VarianceComponentsDialog(ds)
    _fill(v, "dep", "y")
    _fill(v, "random", "g")
    v.opts["method"] = "anova"
    assert "Expected Mean Squaresᵃ,ᵇ" in _titles(v.compute())


def test_menu_entries_registered(qapp):
    from rsstatchem.ui.data_editor import menu_tree

    tree = menu_tree()
    analyze = dict((e[0], e[1]) for e in tree["Analyze"] if isinstance(e, tuple))
    compare = [e[0] for e in analyze["Compare Means"] if isinstance(e, tuple)]
    assert "One-Sample Proportions..." in compare
    assert "Summary Independent-Samples T Test..." in compare
    glm = [e[0] for e in analyze["General Linear Model"] if isinstance(e, tuple)]
    assert "Variance Components..." in glm
    assert not menus.failed_extensions()
