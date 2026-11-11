"""A scale variable in a factor / category box must give a clear message at once, not a
cryptic error, a multi-gigabyte allocation or a computation that never returns.

Found by scripts/analysis_fuzzer.py (every analysis dialog run on demo data)."""
import time

import numpy as np
import pandas as pd
import pytest

from rsstatchem.core.dataset import Dataset


@pytest.fixture
def ds():
    rng = np.random.default_rng(42)
    n = 80
    x1 = rng.normal(50, 10, n).round(2)
    df = pd.DataFrame({
        "y": (0.6 * x1 + rng.normal(0, 5, n)).round(2), "x1": x1,
        "x2": rng.normal(20, 4, n).round(2), "x3": rng.normal(5, 1, n).round(2),
        "grp": rng.integers(1, 4, n).astype(float), "sex": rng.integers(1, 3, n).astype(float),
        "time": rng.exponential(12, n).round(1) + 0.5})
    return Dataset(df)


def _quick(fn, limit: float = 5.0):
    t0 = time.perf_counter()
    with pytest.raises(ValueError) as err:
        fn()
    assert time.perf_counter() - t0 < limit
    return str(err.value)


def test_glm_refuses_a_model_with_millions_of_parameters(ds):
    from rsstatchem.core.stats import glm_model as gm

    def info(name):
        values = ds.df[name]
        levels = sorted(values.unique())
        codes = values.map({v: i for i, v in enumerate(levels)}).to_numpy()
        return gm.FactorInfo(name, levels, [str(v) for v in levels], codes)

    factors = [info(n) for n in ("x1", "x2", "x3")]
    msg = _quick(lambda: gm.GLMFit(ds.df["y"].to_numpy(), factors, {}))
    assert "parameters" in msg and "x1 (" in msg and "Covariate(s)" in msg
    # an ordinary two-factor model is untouched
    ok = gm.GLMFit(ds.df["y"].to_numpy(), [info("grp"), info("sex")], {})
    assert ok.X.shape[0] == 80
    gm.check_model_size([info("grp"), info("sex")], ok.specs)


def test_loglinear_refuses_a_huge_cross_classification(ds):
    from rsstatchem.core.stats import genlin

    msg = _quick(lambda: genlin.loglinear_logit(ds, ["y", "time"], ["sex", "grp"]))
    assert "cells" in msg and "y (" in msg and "Visual Binning" in msg


def test_ordinal_refuses_a_scale_dependent_but_keeps_many_categories(ds):
    from rsstatchem.core.stats import logistic

    msg = _quick(lambda: logistic.ordinal(ds, "y", ["sex"], ["x1"]))
    assert "distinct values" in msg and "Regression > Linear" in msg
    rng = np.random.default_rng(3)
    x = rng.normal(size=200)
    y = pd.qcut(x + rng.logistic(size=200), 12, labels=False).astype(float) + 1
    res = logistic.ordinal(Dataset(pd.DataFrame({"y": y, "x": x})), "y", [], ["x"])
    assert any(getattr(i, "title", "") == "Parameter Estimates" for i in res.section.items)


def test_bayes_regression_needs_more_cases_than_parameters(ds):
    from rsstatchem.core.stats import bayes

    msg = _quick(lambda: bayes.linear_regression(ds, "y", ["x1", "x2"], ["x3"]))
    assert "parameters" in msg and "valid cases" in msg and "math domain" not in msg
    sec = bayes.linear_regression(ds, "y", ["grp"], ["x1"])
    assert any(getattr(i, "title", "").startswith("ANOVA") for i in sec.items)


def test_parametric_survival_explains_when_no_case_is_left(ds):
    from rsstatchem.core.stats import survival_param as sp

    bad = Dataset(ds.df.assign(time=-1.0))        # no positive time at all
    with pytest.raises(ValueError, match="No valid cases are left"):
        sp.parametric_survival(bad, time="time", status="sex", covariates=["x1"])
