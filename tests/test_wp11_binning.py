"""WP11 Optimal Binning: IBM's worked MDLP example (OPTIMAL BINNING Algorithms, 250 cases,
46 distinct values) and the 7-case acceptance example, plus the procedure output."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.stats import optimal_binning as ob

# value, counts of classes 0..3 (IBM Table "2-Dimensional matrix M")
IBM_M = """-2.6 0 1 0 0;-2.4 0 1 0 0;-2.1 0 2 0 0;-2 0 0 2 0;-1.9 0 0 2 0;-1.8 0 0 3 0;
-1.7 0 0 2 0;-1.6 0 0 3 0;-1.5 0 0 3 0;-1.4 0 0 2 0;-1.3 0 0 4 0;-1.2 0 0 3 0;-1.1 0 0 3 0;
-1 0 0 8 0;-0.9 0 0 6 0;-0.8 0 0 7 0;-0.7 0 0 13 0;-0.6 0 0 8 0;-0.5 0 0 4 0;-0.4 0 0 6 0;
-0.3 0 0 7 0;-0.2 0 0 13 0;-0.1 0 0 14 0;0 0 0 0 14;0.1 0 0 0 6;0.2 0 0 0 10;0.3 0 0 0 14;
0.4 0 0 0 12;0.5 0 0 0 4;0.6 0 0 0 9;0.7 0 0 0 5;0.8 0 0 0 3;0.9 0 0 0 10;1 3 0 0 0;
1.1 8 0 0 0;1.2 5 0 0 0;1.3 7 0 0 0;1.4 2 0 0 0;1.5 2 0 0 0;1.6 3 0 0 0;1.7 3 0 0 0;
1.8 4 0 0 0;1.9 4 0 0 0;2 2 0 0 0;2.1 2 0 0 0;2.3 1 0 0 0"""


def _ibm():
    rows = np.array([[float(t) for t in r.split()] for r in IBM_M.replace("\n", "").split(";")])
    return rows[:, 0], rows[:, 1:]


def test_ibm_matrix_shape():
    D, M = _ibm()
    assert len(D) == 46 and M.sum() == 250


def test_ibm_candidate_entropies_and_cuts():
    D, M = _ibm()
    cand = ob._candidates(M, 0, len(M) - 1)
    assert [D[i] for i in cand] == [-2.1, -0.1, 0.9]
    E = [round(ob.class_info_entropy(M, 0, len(M) - 1, i), 4) for i in cand]
    assert E == [1.4742, 0.5955, 0.9038]           # printed by IBM
    assert sorted(D[i] for i in ob.mdlp(M)) == [-2.1, -0.1, 0.9]


def test_ibm_small_example_rejected():
    gain, thr = ob.mdlp_threshold([1, 0], [0, 6])
    assert gain == pytest.approx(0.5916728, abs=1e-7)
    assert thr == pytest.approx(0.6530774, abs=2e-4)
    assert gain < thr                                  # T = 1 is not accepted


def test_binning_on_raw_cases_and_model_entropy():
    D, M = _ibm()
    x = np.repeat(D, M.sum(1).astype(int))
    y = np.concatenate([np.full(int(r.sum()), int(np.argmax(r))) for r in M])
    r = ob.bin_variable(x, y, 4, merge=False)
    assert r["cuts"] == [-2.1, -0.1, 0.9]
    assert r["ends"] == [-2.0, 0.0, 1.0]               # Lower <= x < Upper
    assert [b.sum() for b in r["bins"]] == [4, 113, 87, 46]
    assert r["entropy"] == pytest.approx(0.0)          # every bin is pure
    assert list(ob.assign(np.array([-3, -2.1, -2.0, 0.95, 5]), r["ends"])) == [1, 1, 2, 3, 4]


def test_sparse_bin_merge():
    # bin 2 has 2 cases between bins of 40 -> ratio 0.05 < 0.1 -> merged
    bins = [np.array([40, 0]), np.array([1, 1]), np.array([0, 40])]
    rows, keep = ob.merge_sparse(bins, 0.10)
    assert len(rows) == 2 and sum(r.sum() for r in rows) == 82 and len(keep) == 1
    rows, keep = ob.merge_sparse(bins, 0.01)
    assert len(rows) == 3


def test_hybrid_prebinning_equal_frequency():
    r = np.random.default_rng(0)
    x = r.normal(size=3000)
    y = (x > 0.5).astype(int)
    res = ob.bin_variable(x, y, 2, prebin=20)
    assert len(res["bins"]) == 2
    assert abs(res["cuts"][0] - 0.5) < 0.25


def test_procedure_saves_binned_variable():
    D, M = _ibm()
    x = np.repeat(D, M.sum(1).astype(int))
    y = np.concatenate([np.full(int(r.sum()), int(np.argmax(r))) for r in M])
    ds = Dataset(pd.DataFrame({"a": x, "c": y.astype(float)}))
    sec = ob.optimal_binning(ds, ["a"], "c", save={"a": "a_bin"}, syntax_rules=True)
    titles = [getattr(i, "title", "") for i in sec.items]
    assert {"Descriptive Statistics", "Model Entropy", "a", "Binned Variables"} <= set(titles)
    assert sorted(ds.df["a_bin"].unique()) == [1.0, 2.0, 3.0, 4.0]
    assert ds.var("a_bin").measure == "ordinal"
    with pytest.raises(ob.BinError):
        ob.optimal_binning(ds, ["a"], "a")
