"""Correspondence Analysis and Multidimensional Scaling (WP5): Greenacre's smoking data (the
SPSS CORRESPONDENCE tutorial / R ca::smoke) and sklearn's SMACOF (test-only reference)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.manifold import smacof as sk_smacof

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.stats import correspondence as ca
from rsstatchem.core.stats import mds


def test_ca_smoke_greenacre():
    r = ca.CAResult(ca.SMOKE, 2, "principal")
    # ca::smoke principal inertias and total inertia; SPSS Summary chi-square 16.442 (.172)
    assert r.inertia == pytest.approx([0.07476, 0.01002, 0.00041], abs=5e-6)
    assert r.total == pytest.approx(0.08519, abs=5e-6)
    assert r.n * r.total == pytest.approx(16.442, abs=5e-4)
    assert r.r == pytest.approx([0.057, 0.093, 0.264, 0.456, 0.130], abs=5e-4)
    # principal row coordinates (dim 1 and 2), Greenacre (2007), signs arbitrary
    assert np.abs(r.F[:, 0]) == pytest.approx([0.0658, 0.2590, 0.3806, 0.2330, 0.2011],
                                              abs=5e-4)
    assert np.abs(r.F[:, 1]) == pytest.approx([0.1937, 0.2433, 0.0107, 0.0577, 0.0789],
                                              abs=5e-4)
    assert np.abs(r.G[:, 0]) == pytest.approx([0.3933, 0.0995, 0.1963, 0.2938], abs=5e-4)
    # contributions (ca summary "ctr" and "cor", per mille)
    assert np.round(1000 * r.point_to_dim()[:, 0]).tolist() == [3, 84, 512, 331, 70]
    assert np.round(1000 * r.dim_to_point()[:, 0]).tolist() == [92, 526, 999, 942, 865]
    # normalizations: symmetrical = standard x sqrt(sv); row principal = F for rows, X for cols
    s = ca.CAResult(ca.SMOKE, 2, "symmetrical")
    assert s.row_scores == pytest.approx(s.X * np.sqrt(s.sv))
    rp = ca.CAResult(ca.SMOKE, 2, "rprincipal")
    assert rp.row_scores == pytest.approx(rp.F) and rp.col_scores == pytest.approx(rp.Y)


def test_ca_procedure_from_case_data():
    rows = np.repeat(np.arange(1, 6), 4)
    cols = np.tile(np.arange(1, 5), 5)
    counts = ca.SMOKE.ravel().astype(int)
    df = pd.DataFrame({"staff": np.repeat(rows, counts).astype(float),
                       "smoke": np.repeat(cols, counts).astype(float)})
    sec = ca.correspondence(Dataset(df), "staff", "smoke", plots=("biplot", "rows", "cols"),
                            row_profiles=True, col_profiles=True, permutation=True)
    summary = next(i for i in sec.items if getattr(i, "title", "") == "Summary")
    assert summary.rows[-1][3] == "16.442" and summary.rows[-1][4] == ".172"
    assert summary.rows[0][5] == ".878"
    # Define Range: dropping category 5 leaves a 4 x 4 table
    sec = ca.correspondence(Dataset(df), "staff", "smoke", row_range=(1, 4), plots=())
    table = sec.items[1]
    assert len(table.rows) == 5
    ca.selftest()


def test_mds_classical_and_smacof_vs_sklearn():
    rng = np.random.default_rng(7)
    pts = rng.normal(size=(12, 3))
    D = mds.distances(pts) + rng.uniform(0, 0.3, (12, 12))
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0)
    exact = mds.distances(rng.normal(size=(9, 2)))
    X = mds.classical(exact, 2)
    assert mds.distances(X) == pytest.approx(exact, abs=1e-8)
    X0 = mds.classical(D, 2)
    ours = mds.smacof(D, 2, "ratio", X0=X0, maxiter=300, eps=1e-12)
    Xs, _ = sk_smacof(D, n_components=2, init=X0, n_init=1, metric=True, max_iter=300,
                      eps=1e-12, normalized_stress=False)
    iu = np.triu_indices(12, 1)
    a, b = mds.distances(ours.X)[iu], mds.distances(Xs)[iu]
    assert a / a.sum() == pytest.approx(b / b.sum(), abs=1e-9)
    ordinal = mds.smacof(D, 2, "ordinal")
    assert ordinal.measures()["nraw"] <= ours.measures()["nraw"] + 1e-9
    assert np.all(np.diff(ordinal.dhat[np.argsort(ordinal.delta, kind="stable")]) > -1e-9)


def test_pava():
    y = np.array([1.0, 3.0, 2.0, 4.0, 3.5, 5.0])
    fit = mds.pava(y)
    assert fit.tolist() == pytest.approx([1.0, 2.5, 2.5, 3.75, 3.75, 5.0])


def test_mds_procedures():
    rng = np.random.default_rng(4)
    pts = rng.normal(size=(8, 2))
    D = mds.distances(pts)
    df = pd.DataFrame(D, columns=[f"s{i}" for i in range(8)])
    ds = Dataset(df)
    sec = mds.proxscal(ds, list(df.columns), level="interval",
                       plots=("common", "transformation", "residuals"))
    fit = next(i for i in sec.items if getattr(i, "title", "") == "Stress and Fit Measures")
    assert float(fit.rows[0][1]) < 1e-4
    hist = next(i for i in sec.items if getattr(i, "title", "") == "Iteration History")
    assert float(hist.rows[0][1]) < 0.01          # starts from the scaled Torgerson solution
    sec = mds.alscal_section(ds, list(df.columns), level="ratio")
    note = next(i for i in sec.items if getattr(i, "kind", "") == "text" and "RSQ =" in i.text)
    assert "RSQ = 1.0000" in note.text or "RSQ = 0.9999" in note.text
    raw = pd.DataFrame(rng.normal(size=(15, 4)), columns=list("abcd"))
    D2, labels = mds.proximities_from_data(Dataset(raw), list("abcd"), "euclidean",
                                           "variables", standardize=True)
    z = (raw - raw.mean()) / raw.std()
    assert D2[0, 1] == pytest.approx(np.sqrt(((z.a - z.b) ** 2).sum()))
    sim = pd.DataFrame([[0, 0, 0], [8, 0, 0], [2, 5, 0]], columns=list("xyz"), dtype=float)
    Ds, _ = mds.proximities_from_matrix(Dataset(sim), list("xyz"), similarities=True)
    assert Ds[0, 1] < Ds[0, 2]
    mds.selftest()
