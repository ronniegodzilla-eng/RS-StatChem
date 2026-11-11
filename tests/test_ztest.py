"""One-Sample Z Test (known sigma; Minitab's 1-Sample Z) and its two dialogs."""
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.stats.weightstats import _zstat_generic

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.stats import ztest


def _table(sec, title):
    return next(t for t in sec.items if getattr(t, "title", "") == title)


def _num(text):
    return float(text.rstrip("ᵃᵇ"))


def test_textbook_example_two_sided():
    """n = 36, mean = 52, sigma = 6, H0: mu = 50 -> Z = 2, p = .0455, CI 52 +- 1.96."""
    st = ztest.z_statistics(36, 52.0, 6.0, 50.0)
    assert st["se"] == pytest.approx(1.0) and st["z"] == pytest.approx(2.0)
    assert st["p"] == pytest.approx(0.0455003, abs=1e-7)
    assert (st["lower"], st["upper"]) == pytest.approx((50.040036, 53.959964), abs=1e-6)
    assert st["d"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("alternative, sm", [("two-sided", "two-sided"), ("greater", "larger"),
                                             ("less", "smaller")])
def test_matches_statsmodels_and_one_sided_bounds(alternative, sm):
    rng = np.random.default_rng(11)
    x = rng.normal(101.5, 4.0, 23)
    sigma, mu0, level = 4.0, 100.0, 0.90
    st = ztest.z_statistics(len(x), x.mean(), sigma, mu0, level, alternative)
    z, p = _zstat_generic(x.mean(), 0, sigma / math.sqrt(len(x)), sm, diff=mu0)
    assert st["z"] == pytest.approx(z) and st["p"] == pytest.approx(p)
    se = sigma / math.sqrt(len(x))
    if alternative == "two-sided":
        q = stats.norm.ppf(0.95)
        assert (st["lower"], st["upper"]) == pytest.approx((x.mean() - q * se,
                                                            x.mean() + q * se))
    elif alternative == "greater":       # Minitab: lower bound only
        assert st["upper"] is None
        assert st["lower"] == pytest.approx(x.mean() - stats.norm.ppf(0.90) * se)
    else:
        assert st["lower"] is None
        assert st["upper"] == pytest.approx(x.mean() + stats.norm.ppf(0.90) * se)


def test_tables_from_data_with_missing_values():
    df = pd.DataFrame({"kadar": [4.9, 5.1, 5.0, 5.3, 4.8, np.nan, 5.2, 5.1, 4.7, 5.4],
                       "kosong": [np.nan] * 10})
    ds = Dataset(df)
    ds.var("kadar").label = "Kadar (mg/L)"
    sec = ztest.one_sample(ds, ["kadar", "kosong"], sigma=0.2, test_value=5.0)
    s = _table(sec, "One-Sample Statistics")
    x = df["kadar"].dropna()
    assert s.rows[0][:2] == ["Kadar (mg/L)", "9"]
    assert _num(s.rows[0][2]) == pytest.approx(x.mean(), abs=1e-4)
    assert _num(s.rows[0][3]) == pytest.approx(x.std(ddof=1), abs=1e-5)      # sample SD shown
    assert _num(s.rows[0][4]) == pytest.approx(0.2 / 3, abs=1e-5)            # SE from sigma
    assert s.rows[1][1] == "0" and s.rows[1][2] == ""                        # all missing
    assert "known standard deviation = 0.2" in s.footnotes[0]
    t = _table(sec, "One-Sample Z Test")
    assert len(t.rows) == 1
    z = (x.mean() - 5.0) / (0.2 / 3)
    assert _num(t.rows[0][1]) == pytest.approx(z, abs=1e-3)
    assert t.columns[2][1] == "Two-Sided p" and "mean ≠ 5" in t.footnotes[0]
    one = ztest.one_sample(ds, ["kadar"], 0.2, 5.0, alternative="greater")
    assert _table(one, "One-Sample Statistics").columns[-1] == "95% Lower Bound for Mean"
    assert _table(one, "One-Sample Z Test").columns[2][1] == "One-Sided p"


def test_summary_data_and_argument_checks():
    sec = ztest.summary_one_sample(36, 52, 6, 50, label="Batch A")
    assert _table(sec, "One-Sample Statistics").rows[0][0] == "Batch A"
    assert _table(sec, "One-Sample Z Test").rows[0][1:3] == ["2.000", ".046"]
    for bad in ({"sigma": 0}, {"sigma": -1}, {"n": 0}, {"alternative": "both"}):
        args = {"n": 10, "mean": 1.0, "sigma": 1.0, "alternative": "two-sided", **bad}
        with pytest.raises(ValueError):
            ztest.z_statistics(args["n"], args["mean"], args["sigma"], 0.0, 0.95,
                               args["alternative"])


def test_dialogs_and_menu(qapp):
    from rsstatchem.ui import menus
    from rsstatchem.ui.dialogs import compare as cdlg
    from rsstatchem.ui.dialogs import proportions as pdlg
    from rsstatchem.ui.dialogs.base import VariablePickerDialog

    ds = Dataset(pd.DataFrame({"y": np.linspace(48, 56, 36)}))
    VariablePickerDialog._memory["OneSampleZDialog"] = {"vars": ["y"]}
    dlg = cdlg.OneSampleZDialog(ds)
    try:
        assert "greater than 0" in dlg.validate()             # sigma still empty
        dlg.sigma.setText("6")
        dlg.test_value.setText("50,5")                        # decimal comma accepted
        dlg.alternative.setCurrentIndex(dlg.alternative.findData("greater"))
        assert dlg.validate() is None
        sec = dlg.compute()
        t = _table(sec, "One-Sample Z Test")
        assert t.columns[1][0] == "Test Value = 50.5" and t.columns[2][1] == "One-Sided p"
        assert _num(t.rows[0][1]) == pytest.approx((52 - 50.5) / 1.0, abs=1e-3)
    finally:
        VariablePickerDialog._memory.pop("OneSampleZDialog", None)
        VariablePickerDialog._states.pop("OneSampleZDialog", None)
        dlg.deleteLater()

    s = pdlg.SummaryZTestDialog()
    for key, value in {"n": "36", "mean": "52", "sigma": "6", "test_value": "50"}.items():
        s.fields[key].setText(value)
    s.accept()
    assert _table(s.compute(), "One-Sample Z Test").rows[0][1] == "2.000"
    s.deleteLater()

    order = menus.MENU_ORDER["Analyze/Compare Means"]
    assert order.index("One-Sample Z Test...") == order.index("One-Sample T Test...") + 1
    assert "Summary One-Sample Z Test..." in order
