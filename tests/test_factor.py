"""Factor analysis checked against numpy eigen decomposition, statsmodels Factor / rotations and
independent formulas (KMO via partial correlations, Bartlett's sphericity test)."""
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.multivariate.factor import Factor
from statsmodels.multivariate.factor_rotation import rotate_factors

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.stats import factor
from rsstatchem.io.files import load_project

ITEMS = ["X1", "X2", "X3", "X4", "X5", "X6", "Y1", "Y2", "Y3", "Y4", "Y5"]


def num(text: str) -> float:
    return float(text.rstrip("*ᵃᵇᶜᵈ").replace(",", ""))


def table(section, title):
    return next(i for i in section.items if getattr(i, "title", "").startswith(title))


@pytest.fixture
def two_factor():
    rng = np.random.default_rng(4)
    n = 300
    f = rng.normal(size=(n, 2))
    lam = np.array([[0.8, 0.1], [0.7, 0.2], [0.75, 0.0], [0.1, 0.8], [0.2, 0.7], [0.0, 0.6]])
    X = f @ lam.T + rng.normal(size=(n, 6)) * np.sqrt(1 - (lam ** 2).sum(1))
    return pd.DataFrame(X, columns=[f"v{i + 1}" for i in range(6)])


@pytest.fixture(scope="module")
def kuesioner():
    return load_project("examples/contoh_kuesioner.rsd")


def test_pca_matches_numpy_eigen(two_factor):
    ds = Dataset(two_factor)
    res = factor.factor_analysis(ds, list(two_factor.columns), "pc", criterion="fixed",
                                 n_factors=2)
    R = np.corrcoef(two_factor.to_numpy(), rowvar=False)
    w, V = np.linalg.eigh(R)
    w, V = w[::-1], V[:, ::-1]
    tve = table(res.section, "Total Variance Explained").rows
    for i in range(6):
        assert num(tve[i][1]) == pytest.approx(w[i], abs=1e-3)
        assert num(tve[i][2]) == pytest.approx(w[i] / 6 * 100, abs=1e-3)
    mat = table(res.section, "Component Matrix").rows
    L = np.array([[num(c) for c in r[1:]] for r in mat])
    ref = V[:, :2] * np.sqrt(w[:2])
    assert np.allclose(np.abs(L), np.abs(ref), atol=1e-3)
    comm = table(res.section, "Communalities").rows
    assert num(comm[0][2]) == pytest.approx((ref[0] ** 2).sum(), abs=1e-3)


def test_kmo_and_bartlett_independent_formulas(kuesioner):
    X = kuesioner.df[ITEMS].to_numpy(float)
    n, p = X.shape
    R = np.corrcoef(X, rowvar=False)
    overall, msa, _ = factor.kmo(R)
    # partial correlations from residuals of regressing each pair on all other variables
    partial = np.zeros((p, p))
    Z = (X - X.mean(0)) / X.std(0)
    for i in range(p):
        for j in range(i + 1, p):
            others = [k for k in range(p) if k not in (i, j)]
            A = np.column_stack([np.ones(n), Z[:, others]])
            ri = Z[:, i] - A @ np.linalg.lstsq(A, Z[:, i], rcond=None)[0]
            rj = Z[:, j] - A @ np.linalg.lstsq(A, Z[:, j], rcond=None)[0]
            partial[i, j] = partial[j, i] = np.corrcoef(ri, rj)[0, 1]
    off = ~np.eye(p, dtype=bool)
    r2, q2 = (R ** 2)[off].sum(), (partial ** 2)[off].sum()
    assert overall == pytest.approx(r2 / (r2 + q2), abs=1e-9)
    j0 = ((R[0, 1:] ** 2).sum()) / ((R[0, 1:] ** 2).sum() + (partial[0, 1:] ** 2).sum())
    assert msa[0] == pytest.approx(j0, abs=1e-9)
    chi, df, pval = factor.bartlett_sphericity(R, n)
    assert chi == pytest.approx(-(n - 1 - (2 * p + 5) / 6) * math.log(np.linalg.det(R)))
    assert df == 55 and pval == pytest.approx(stats.chi2.sf(chi, 55))
    res = factor.factor_analysis(kuesioner, ITEMS, anti_image=True)
    kmo_tab = table(res.section, "KMO and Bartlett's Test").rows
    assert num(kmo_tab[0][2]) == pytest.approx(overall, abs=1e-3)
    assert num(kmo_tab[1][2]) == pytest.approx(chi, abs=1e-3)
    ai = table(res.section, "Anti-image Matrices").rows
    corr_block = [r for r in ai if r[0] == "Anti-image Correlation"][0]
    assert num(corr_block[2]) == pytest.approx(msa[0], abs=1e-3)
    assert num(corr_block[3]) == pytest.approx(-partial[0, 1], abs=1e-3)


def test_paf_matches_statsmodels(two_factor):
    R = np.corrcoef(two_factor.to_numpy(), rowvar=False)
    L, it, conv = factor.extract_paf(R, 2, max_iter=2000, tol=1e-12)
    ref = Factor(two_factor.to_numpy(), n_factor=2, method="pa", smc=True).fit(
        maxiter=2000, tol=1e-12)
    assert conv
    assert np.allclose((L ** 2).sum(1), ref.communality, atol=1e-4)


def test_ml_matches_statsmodels(two_factor):
    R = np.corrcoef(two_factor.to_numpy(), rowvar=False)
    L, it, F, conv = factor.extract_ml(R, 2)
    ref = Factor(two_factor.to_numpy(), n_factor=2, method="ml").fit()
    assert np.allclose((L ** 2).sum(1), ref.communality, atol=2e-3)
    res = factor.factor_analysis(Dataset(two_factor), list(two_factor.columns), "ml",
                                 criterion="fixed", n_factors=2)
    gof = table(res.section, "Goodness-of-fit Test").rows[0]
    assert int(gof[1]) == ((6 - 2) ** 2 - 8) // 2


def test_rotations(two_factor):
    R = np.corrcoef(two_factor.to_numpy(), rowvar=False)
    L = factor.extract_pc(R, 2)
    h = np.sqrt((L ** 2).sum(1))
    for method, gamma in (("varimax", 1.0), ("quartimax", 0.0)):
        P, phi, T, _ = factor.rotate(L, method)
        ref, _ = rotate_factors(L / h[:, None], method)
        ref = ref * h[:, None]
        # equal up to column order and sign
        for j in range(2):
            col = P[:, j]
            assert min(np.abs(np.abs(col) - np.abs(ref[:, k])).max() for k in range(2)) < 5e-4
        assert np.allclose((P ** 2).sum(1), (L ** 2).sum(1))
    for method in ("oblimin", "promax"):
        P, phi, T, _ = factor.rotate(L, method)
        assert np.allclose(np.diag(phi), 1, atol=1e-6)
        # the oblique solution reproduces the same common variance
        assert np.allclose(np.diag(P @ phi @ P.T), (L ** 2).sum(1), atol=1e-6)


def test_oblique_tables_and_scores(two_factor):
    ds = Dataset(two_factor)
    res = factor.factor_analysis(ds, list(two_factor.columns), "paf", criterion="fixed",
                                 n_factors=2, rotation="oblimin", save="bartlett",
                                 score_coefficients=True, sort=True, suppress=0.3)
    titles = [getattr(i, "title", "") for i in res.section.items]
    for t in ("Pattern Matrix", "Structure Matrix", "Factor Correlation Matrix",
              "Factor Score Coefficient Matrix"):
        assert any(s.startswith(t) for s in titles)
    pattern = table(res.section, "Pattern Matrix").rows
    assert any(c == "" for r in pattern for c in r[1:])  # suppressed small loadings
    assert set(res.saved) == {"FAC1", "FAC2"}
    assert res.saved["FAC1"][0].startswith("BART factor score")


def test_scores_properties(two_factor):
    ds = Dataset(two_factor)
    cols = list(two_factor.columns)
    pc = factor.factor_analysis(ds, cols, "pc", criterion="fixed", n_factors=2,
                                rotation="varimax", save="regression")
    S = np.column_stack([pc.saved[k][1].to_numpy() for k in ("FAC1", "FAC2")])
    assert np.allclose(np.cov(S, rowvar=False), np.eye(2), atol=1e-8)
    ar = factor.factor_analysis(ds, cols, "paf", criterion="fixed", n_factors=2,
                                rotation="varimax", save="anderson_rubin")
    S = np.column_stack([ar.saved[k][1].to_numpy() for k in ("FAC1", "FAC2")])
    assert np.allclose(np.cov(S, rowvar=False), np.eye(2), atol=1e-6)


def test_scores_respect_missing(kuesioner):
    ds = load_project("examples/contoh_kuesioner.rsd")
    ds.set_value(0, ds.index_of("X1"), "")
    res = factor.factor_analysis(ds, ITEMS, save="regression")
    s = res.saved["FAC1"][1]
    assert math.isnan(s.iloc[0]) and s.notna().sum() == ds.n_cases - 1
    tve = table(res.section, "Total Variance Explained").rows
    assert num(tve[-1][3]) == pytest.approx(100, abs=1e-3)
