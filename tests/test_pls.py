"""PLS-SEM (SmartPLS-style) checked against the plspm package (Google's Python port of the
R package), an independent data-based implementation of the PLS algorithm (seminr style),
statsmodels / pingouin and hand formulas."""
import os
import time

import numpy as np
import pandas as pd
import pytest

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.stats import pls
from rsstatchem.core.stats import pls_engine as E
from rsstatchem.core.stats.pls_model import ModelError, parse_syntax

SYNTAX = pls.SELFTEST_SYNTAX


@pytest.fixture(scope="module")
def df():
    return pls.selftest_data(250, seed=11)


def structure(syntax, columns):
    model = parse_syntax(syntax)
    model.validate(list(columns))
    return E.make_structure(model)


def fit(df, syntax=SYNTAX, scheme="path", consistent=False):
    st = structure(syntax, df.columns)
    X = df[st.indicators].to_numpy(float)
    Z, _, _ = E.standardize(X)
    est = E.estimate((Z.T @ Z / len(Z))[None], st, scheme, 300, 1e-10, consistent)
    return st, Z, {k: v[0] for k, v in est.items()}


def naive_pls(X, st, scheme="path", tol=1e-12):
    """Straightforward data-based PLS algorithm (Lohmöller; seminr / SmartPLS conventions)."""
    X = (X - X.mean(0)) / X.std(0, ddof=1)
    n = len(X)
    W = st.mask.copy()
    W = W / (X @ W).std(0, ddof=1)
    for _ in range(1000):
        Y = X @ W
        R = np.corrcoef(Y.T)
        A = st.D + st.D.T
        if scheme == "centroid":
            Ein = np.sign(R) * A
        elif scheme == "factorial":
            Ein = R * A
        else:
            Ein = np.zeros_like(R)
            for j in range(st.K):
                P, Q = st.preds[j], st.succs[j]
                if len(P):
                    Ein[P, j] = np.linalg.lstsq(Y[:, P], Y[:, j], rcond=None)[0]
                Ein[Q, j] = R[Q, j]
        Zin = Y @ Ein
        Wn = np.zeros_like(W)
        for j, b in enumerate(st.blocks):
            if st.modes[j] == "A":
                Wn[b, j] = X[:, b].T @ Zin[:, j] / (n - 1)
            else:
                Wn[b, j] = np.linalg.lstsq(X[:, b], Zin[:, j], rcond=None)[0]
        Wn = Wn / (X @ Wn).std(0, ddof=1)
        diff = np.abs(Wn - W).sum()
        W = Wn
        if diff < tol:
            break
    return W


# ---------------------------------------------------------------------- reference checks
@pytest.mark.parametrize("scheme", ["path", "centroid", "factorial"])
@pytest.mark.parametrize("op", ["=~", "<~"])
def test_matches_naive_algorithm(df, scheme, op):
    syntax = SYNTAX.replace("M =~", f"M {op}")
    st, Z, est = fit(df, syntax, scheme)
    W = naive_pls(df[st.indicators].to_numpy(float), st, scheme)
    assert np.abs(est["W"] - W).max() < 1e-8
    Y = Z @ W
    R = np.corrcoef(Y.T)
    assert np.allclose(est["R"], R, atol=1e-8)
    for j in st.endogenous:
        P = st.preds[j]
        beta = np.linalg.lstsq(Y[:, P], Y[:, j] - Y[:, j].mean(), rcond=None)[0]
        assert np.allclose(est["paths"][P, j], beta, atol=1e-8)


@pytest.mark.parametrize("scheme", ["path", "centroid", "factorial"])
@pytest.mark.parametrize("mode", ["A", "B"])
def test_matches_plspm(df, scheme, mode):
    plspm_config = pytest.importorskip("plspm.config")
    from plspm.estimator import Estimator
    from plspm.mode import Mode
    from plspm.scheme import Scheme
    from plspm.weights import WeightsCalculatorFactory

    op = "=~" if mode == "A" else "<~"
    st, Z, est = fit(df, SYNTAX.replace("M =~", f"M {op}"), scheme)
    lvs = ["X", "M", "Y"]
    path = pd.DataFrame([[0, 0, 0], [1, 0, 0], [1, 1, 0]], index=lvs, columns=lvs)
    # plspm's scaled=True divides by one pooled SD, so pass z-scores unscaled instead
    cfg = plspm_config.Config(path, scaled=False)
    for lv in lvs:
        m = Mode.B if (lv == "M" and mode == "B") else Mode.A
        cfg.add_lv(lv, m, *[plspm_config.MV(i) for i in st.indicators if i.startswith(lv)])
    data = (df - df.mean()) / df.std()
    sch = {"path": Scheme.PATH, "centroid": Scheme.CENTROID, "factorial": Scheme.FACTORIAL}
    n = len(df)
    calc = WeightsCalculatorFactory(cfg, 500, 1e-12, np.sqrt(n / (n - 1)), sch[scheme])
    _, scores, weights = Estimator(cfg).estimate(calc, cfg.filter(data))
    ref = scores[lvs].to_numpy()
    ref = (ref - ref.mean(0)) / ref.std(0)
    assert np.abs(Z @ est["W"] - ref).max() < 1e-5
    w_ref = weights.loc[st.indicators, "weight"].to_numpy()
    ratio = (est["W"] * st.mask).sum(1) / w_ref
    for blk in st.blocks:  # same weights up to plspm's n/(n-1) scaling of the scores
        assert np.ptp(ratio[blk]) < 1e-5
    b = np.linalg.lstsq(ref[:, :2], ref[:, 2], rcond=None)[0]
    assert est["paths"][0, 2] == pytest.approx(b[0], abs=1e-6)
    assert est["paths"][1, 2] == pytest.approx(b[1], abs=1e-6)


# ---------------------------------------------------------------------- internal consistency
def test_scores_standardized_and_mode_a_weights(df):
    st, Z, est = fit(df)
    Y = Z @ est["W"]
    assert np.allclose(Y.mean(0), 0, atol=1e-10)
    assert np.allclose(Y.std(0), 1, atol=1e-10)
    # Mode A: outer weights proportional to cov(indicator, inner proxy) at convergence
    Ein = E._inner_weights(est["R"][None], st, "path")[0]
    Zin = Y @ Ein
    for j, blk in enumerate(st.blocks):
        cov = Z[:, blk].T @ Zin[:, j] / len(Z)
        ratio = est["W"][blk, j] / cov
        assert np.ptp(ratio) < 1e-8
    # loadings are correlations of indicators with the scores
    C = np.corrcoef(np.column_stack([Z, Y]).T)[:st.p, st.p:]
    assert np.allclose(est["C"], C, atol=1e-10)


def test_single_indicator_and_two_construct_model(df):
    st, Z, est = fit(df, "X =~ X1 + X2 + X3\nY =~ Y1\nY ~ X")
    j = st.names.index("Y")
    k = st.blocks[j][0]
    assert est["loadings"][k, j] == pytest.approx(1.0)
    assert est["W"][k, j] == pytest.approx(1.0)
    Y = Z @ est["W"]
    r = np.corrcoef(Y.T)[0, 1]
    assert est["paths"][0, 1] == pytest.approx(r, abs=1e-12)
    assert est["r2"][1] == pytest.approx(r ** 2, abs=1e-12)


# ---------------------------------------------------------------------- quality criteria
def test_reliability_hand_formulas(df):
    pg = pytest.importorskip("pingouin")
    st, Z, est = fit(df)
    S = Z.T @ Z / len(Z)
    rel = E.reliability(S, est["loadings"], est["rhoA"], st)
    for j, blk in enumerate(st.blocks):
        lam = est["loadings"][blk, j]
        rhoc = lam.sum() ** 2 / (lam.sum() ** 2 + (1 - lam ** 2).sum())
        assert rel["rhoC"][j] == pytest.approx(rhoc)
        assert rel["ave"][j] == pytest.approx(np.mean(lam ** 2))
        alpha = pg.cronbach_alpha(pd.DataFrame(Z[:, blk]))[0]
        assert rel["alpha"][j] == pytest.approx(alpha, abs=1e-10)
        # rho_A, Dijkstra & Henseler (2015) eq. with w scaled to unit score variance
        w = est["W"][blk, j]
        Sjj = S[np.ix_(blk, blk)]
        num = w @ (Sjj - np.diag(np.diag(Sjj))) @ w
        ww = np.outer(w, w)
        den = w @ (ww - np.diag(np.diag(ww))) @ w
        assert rel["rhoA"][j] == pytest.approx((w @ w) ** 2 * num / den, rel=1e-10)


def test_htmt_vif_f2_srmr_hand(df):
    sm = pytest.importorskip("statsmodels.api")
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    st, Z, est = fit(df)
    S = Z.T @ Z / len(Z)
    # HTMT
    bx, bm = st.blocks[0], st.blocks[1]
    hetero = np.abs(S[np.ix_(bx, bm)]).mean()
    mono = [np.abs(S[np.ix_(b, b)][np.triu_indices(len(b), 1)]).mean() for b in (bx, bm)]
    assert E.htmt(S[None], st, [(0, 1)])[0, 0] == pytest.approx(hetero / np.sqrt(np.prod(mono)))
    # outer VIF (within block) vs statsmodels
    vif = E.outer_vif(S, st)
    block = sm.add_constant(Z[:, st.blocks[2]])
    for i, k in enumerate(st.blocks[2]):
        assert vif[k] == pytest.approx(variance_inflation_factor(block, i + 1), rel=1e-8)
    # inner VIF and f² from OLS on the construct scores
    Y = Z @ est["W"]
    ivif = E.inner_vif(est["R"], st)
    r_xm = np.corrcoef(Y[:, 0], Y[:, 1])[0, 1]
    assert ivif[0, 2] == pytest.approx(1 / (1 - r_xm ** 2))
    full = sm.OLS(Y[:, 2], sm.add_constant(Y[:, :2])).fit().rsquared
    no_m = sm.OLS(Y[:, 2], sm.add_constant(Y[:, :1])).fit().rsquared
    f2 = E.f_squared(est["R"][None], est["r2"][None], st)[0]
    assert f2[1, 2] == pytest.approx((full - no_m) / (1 - full), rel=1e-8)
    # SRMR of the saturated model: implied = lambda lambda' * construct correlation
    lam = est["loadings"].sum(1)
    sigma = np.outer(lam, lam) * est["R"][np.ix_(st.owner, st.owner)]
    np.fill_diagonal(sigma, 1)
    low = np.tril_indices(st.p)
    srmr = np.sqrt(np.mean((S - sigma)[low] ** 2))
    fit_sat = E.fit_indices(S, E.implied_corr(S, est["loadings"], est["R"], st), len(Z))
    assert fit_sat["SRMR"] == pytest.approx(srmr)
    assert fit_sat["d_ULS"] == pytest.approx(0.5 * ((S - sigma) ** 2).sum())
    ev = np.linalg.eigvals(np.linalg.solve(S, sigma)).real
    assert fit_sat["d_G"] == pytest.approx(0.5 * np.sum(np.log(ev) ** 2))


def test_structural_fit_differs_when_model_not_saturated(df):
    syntax = SYNTAX.replace("Y ~ X + M", "Y ~ M")
    res = pls.pls_sem(Dataset(df), syntax, bootstrap=False, blindfold=False, notes=False)
    f = res.quality["fit"]
    assert f["estimated"]["SRMR"] > f["saturated"]["SRMR"]
    res2 = pls.pls_sem(Dataset(df), SYNTAX, bootstrap=False, blindfold=False)
    titles = [i.title for i in res2.section.items if getattr(i, "kind", "") == "table"]
    assert {"Total indirect effects", "Specific indirect effects", "Total effects"} <= set(titles)
    f2 = res2.quality["fit"]
    assert f2["estimated"]["SRMR"] == pytest.approx(f2["saturated"]["SRMR"], abs=1e-10)


def test_effects(df):
    st, Z, est = fit(df)
    P = est["paths"]
    tot = E.total_effects(P[None], st)[0]
    assert tot[0, 2] == pytest.approx(P[0, 2] + P[0, 1] * P[1, 2])
    labels = E.stat_labels(st)
    assert [lab[-1] for lab in labels["specific_indirect"]] == ["X -> M -> Y"]


def test_consistent_pls_recovers_true_parameters():
    rng = np.random.default_rng(5)
    n = 20000
    x = rng.normal(size=n)
    y = 0.6 * x + np.sqrt(1 - 0.36) * rng.normal(size=n)
    lam = [0.9, 0.7, 0.6]
    data = {}
    for name, lv in (("X", x), ("Y", y)):
        for i, lam_i in enumerate(lam):
            data[f"{name}{i + 1}"] = lam_i * lv + np.sqrt(1 - lam_i ** 2) * rng.normal(size=n)
    d = pd.DataFrame(data)
    syntax = "X =~ X1 + X2 + X3\nY =~ Y1 + Y2 + Y3\nY ~ X"
    _, _, plain = fit(d, syntax)
    _, _, cons = fit(d, syntax, consistent=True)
    assert plain["paths"][0, 1] < 0.55  # attenuated
    assert cons["paths"][0, 1] == pytest.approx(0.6, abs=0.02)
    assert np.allclose(cons["loadings"].sum(1), lam * 2, atol=0.02)


# ---------------------------------------------------------------------- bootstrap
def test_bootstrap_ci_contains_estimate_and_is_reproducible(df):
    st = structure(SYNTAX, df.columns)
    X = df[st.indicators].to_numpy(float)
    a = E.bootstrap(X, st, 500, seed=42)
    b = E.bootstrap(X, st, 500, seed=42)
    c = E.bootstrap(X, st, 500, seed=7)
    assert a.valid == 500
    for g in ("paths", "loadings", "weights", "specific_indirect", "htmt"):
        for ra, rb in zip(a.table[g], b.table[g]):
            assert ra["lo"] <= ra["O"] <= ra["hi"]
            assert ra["M"] == rb["M"] and ra["lo"] == rb["lo"]
    assert a.table["paths"][0]["M"] != c.table["paths"][0]["M"]
    # the first resample reproduces a direct estimate on the resampled data
    rng = np.random.default_rng(42)
    idx = rng.integers(0, len(X), size=(min(1000, 500), len(X)))[0]
    _, _, est = fit(df.iloc[idx].reset_index(drop=True))
    assert a.draws["paths"][0, 0] == pytest.approx(est["paths"][0, 1], abs=1e-7)


def test_bootstrap_methods_and_tails(df):
    st = structure(SYNTAX, df.columns)
    X = df[st.indicators].to_numpy(float)
    two = E.bootstrap(X, st, 400, seed=1, method="percentile", tails=2)
    one = E.bootstrap(X, st, 400, seed=1, method="percentile", tails=1)
    for r2, r1 in zip(two.table["paths"], one.table["paths"]):
        assert r1["P"] == pytest.approx(r2["P"] / 2)
        assert r1["lo"] >= r2["lo"] and r1["hi"] <= r2["hi"]
    for method in ("bc", "bca"):
        res = E.bootstrap(X, st, 400, seed=1, method=method)
        for r in res.table["paths"]:
            assert r["lo"] < r["O"] < r["hi"]


def test_bootstrap_speed():
    data = pls.selftest_data(200, seed=1)
    for i in range(1, 4):
        data[f"Z{i}"] = data[f"Y{i}"] * 0.5 + np.random.default_rng(i).normal(size=200)
    syntax = SYNTAX + "\nZ =~ Z1 + Z2 + Z3\nZ ~ Y + M"
    st = structure(syntax, data.columns)
    t = time.perf_counter()
    res = E.bootstrap(data[st.indicators].to_numpy(float), st, 5000, seed=1)
    assert time.perf_counter() - t < 20
    assert res.valid == 5000


# ---------------------------------------------------------------------- blindfolding / predict
def test_blindfolding(df):
    st, Z, est = fit(df)
    q = E.blindfolding(Z, st, 7)
    n = len(Z)
    for j, (sso, sse) in q.items():
        assert sso == pytest.approx(n * len(st.blocks[j]))
        q2 = 1 - sse / sso
        assert 0 < q2 < est["r2"][j]
    assert E.blindfolding(Z, st, 7) == q


def test_pls_predict(df):
    st = structure(SYNTAX, df.columns)
    rows = E.pls_predict(df[st.indicators].to_numpy(float), st, folds=10, reps=2, seed=3)
    assert [r["indicator"] for r in rows] == ["M1", "M2", "M3", "Y1", "Y2", "Y3", "Y4"]
    assert all(r["Q2predict"] > 0 for r in rows)
    assert all(r["PLS_MAE"] < r["PLS_RMSE"] for r in rows)


# ---------------------------------------------------------------------- syntax / validation
def test_syntax_and_validation(df):
    m = parse_syntax("A =~ X1 + X2; B <~ M1, M2\nC =~ Y1 + Y2 # comment\nA -> B -> C\nC ~ A")
    assert [c.mode for c in m.constructs] == ["A", "B", "A"]
    assert set(m.paths) == {("A", "B"), ("B", "C"), ("A", "C")}
    again = parse_syntax(m.to_syntax())
    assert set(again.paths) == set(m.paths)
    assert [(c.name, c.mode, c.indicators) for c in again.constructs] == \
        [(c.name, c.mode, c.indicators) for c in m.constructs]
    cols = list(df.columns)
    with pytest.raises(ModelError, match="recursive"):
        parse_syntax("A =~ X1\nB =~ Y1\nB ~ A\nA ~ B").validate(cols)
    with pytest.raises(ModelError, match="not a variable"):
        parse_syntax("A =~ X1 + Q9\nB =~ Y1\nB ~ A").validate(cols)
    with pytest.raises(ModelError, match="not a construct"):
        parse_syntax("A =~ X1\nB =~ Y1\nB ~ A + C").validate(cols)
    with pytest.raises(ModelError, match="not connected"):
        parse_syntax("A =~ X1\nB =~ Y1\nC =~ M1\nB ~ A").validate(cols)
    with pytest.raises(ModelError):
        parse_syntax("A = X1")
    m = parse_syntax("a =~ x1 + x2\nb =~ y1\nB ~ A")
    m.validate(cols)
    assert m.constructs[0].indicators == ["X1", "X2"] and m.paths == [("a", "b")]


# ---------------------------------------------------------------------- full analysis
def test_questionnaire_example_and_saved_scores():
    from rsstatchem.io.files import load_project

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ds = load_project(os.path.join(here, "examples", "contoh_kuesioner.rsd"))
    syntax = "X =~ X1 + X2 + X3\nM =~ X4 + X5 + X6\nY =~ Y1 + Y2 + Y3 + Y4 + Y5\nM ~ X\n" \
             "Y ~ X + M"
    res = pls.pls_sem(ds, syntax, subsamples=500, predict=True, reps=2, save_scores=True,
                      ci="bca")
    titles = [i.title for i in res.section.items if getattr(i, "kind", "") == "table"]
    for t in ("Path coefficients", "Outer loadings", "Outer weights", "R-square", "f-square",
              "Construct reliability and validity",
              "Discriminant validity - Fornell-Larcker criterion",
              "Discriminant validity - Cross loadings",
              "Discriminant validity - Heterotrait-monotrait ratio (HTMT) - Matrix",
              "Collinearity statistics (VIF) - Outer model - List",
              "Collinearity statistics (VIF) - Inner model - List", "Model fit",
              "Bootstrapping - Path coefficients", "Bootstrapping - Specific indirect effects",
              "Bootstrapping - Heterotrait-monotrait ratio (HTMT)",
              "Construct crossvalidated redundancy", "PLSpredict - MV prediction summary",
              "Hypothesis testing summary"):
        assert t in titles, t
    assert any(getattr(i, "kind", "") == "image" for i in res.section.items)
    assert set(res.saved) == {"LV_X", "LV_M", "LV_Y"}
    y = res.saved["LV_Y"][1]
    assert len(y) == ds.n_cases and y.std(ddof=0) == pytest.approx(1)


def test_missing_values(df):
    d = df.copy()
    d.loc[[3, 10, 20], "X1"] = np.nan
    lw = pls.pls_sem(Dataset(d), SYNTAX, bootstrap=False, blindfold=False, save_scores=True)
    mr = pls.pls_sem(Dataset(d), SYNTAX, bootstrap=False, blindfold=False, missing="mean")
    assert lw.n == len(d) - 3 and mr.n == len(d)
    assert np.isnan(lw.saved["LV_X"][1].iloc[3])


def test_errors():
    d = pls.selftest_data(50)
    d["C"] = 1.0
    with pytest.raises(ModelError, match="variance"):
        pls.pls_sem(Dataset(d), "A =~ X1 + C\nB =~ Y1\nB ~ A", bootstrap=False)


def test_selftest():
    from rsstatchem.ui.dialogs import pls as dialog_module

    dialog_module.selftest()
    assert "PLS-SEM..." in [e[0] for e in
                            dialog_module.MENUS["Analyze/Structural Equation Modeling"]]


def test_dialog_offscreen(df, qapp):
    from rsstatchem.ui.dialogs.pls import PLSSEMDialog

    dlg = PLSSEMDialog(Dataset(df))
    dlg.tabs.setCurrentIndex(1)
    dlg.editor.setPlainText(SYNTAX)
    dlg.tabs.setCurrentIndex(0)  # syntax -> builder
    assert dlg.tree.topLevelItemCount() == 3 and dlg.path_list.count() == 3
    dlg.opts.update(subsamples=200, blindfold=False)
    dlg.accept()
    res = dlg.compute()
    assert res.section.title == "PLS-SEM"
    dlg2 = PLSSEMDialog(Dataset(df))  # remembers the last model
    assert dlg2.model.to_syntax() == dlg.model.to_syntax()
    dlg.deleteLater()
    dlg2.deleteLater()
    qapp.processEvents()


# ---------------------------------------------------------------------- moderation
@pytest.fixture(scope="module")
def mod_df():
    rng = np.random.default_rng(2)
    n = 300
    x, w = rng.normal(size=n), rng.normal(size=n)
    y = 0.4 * x + 0.2 * w + 0.25 * x * w + rng.normal(size=n) * 0.8
    d = {}
    for name, lv in (("X", x), ("W", w), ("Y", y)):
        for i in range(3):
            d[f"{name}{i + 1}"] = lv + rng.normal(size=n) * 0.6
    return pd.DataFrame(d)


MOD = "X =~ X1 + X2 + X3\nW =~ W1 + W2 + W3\nY =~ Y1 + Y2 + Y3\nY ~ X + W + X:W"


def manual_two_stage(d):
    _, Z1, e1 = fit(d, MOD.replace(" + X:W", ""))
    Y1 = Z1 @ e1["W"]
    d2 = d.copy()
    d2["XW"] = Y1[:, 0] * Y1[:, 1]
    _, _, e2 = fit(d2, "X =~ X1 + X2 + X3\nW =~ W1 + W2 + W3\nXW =~ XW\n"
                         "Y =~ Y1 + Y2 + Y3\nY ~ X + W + XW")
    return e2["paths"][:3, 3]


def test_two_stage_moderation(mod_df):
    res = pls.pls_sem(Dataset(mod_df), MOD, subsamples=300, seed=5, predict=True)
    st = res.structure
    assert st.names == ["X", "W", "X x W", "Y"]
    est = res.estimates["paths"][:3, 3]
    assert np.allclose(est, manual_two_stage(mod_df), atol=1e-8)
    assert res.boot.table["paths"][2]["P"] < 0.01
    assert any("Simple slope" in i.title for i in res.section.items
               if getattr(i, "kind", "") == "image")
    # bootstrap rebuilds the interaction term from each resample's stage-1 scores
    rng = np.random.default_rng(5)
    idx = rng.integers(0, len(mod_df), size=(300, len(mod_df)))[0]
    first = manual_two_stage(mod_df.iloc[idx].reset_index(drop=True))
    assert np.allclose(res.boot.draws["paths"][0], first, atol=1e-6)
    with pytest.raises(ModelError, match="main effect"):
        pls.pls_sem(Dataset(mod_df), MOD.replace("X + W", "X"), bootstrap=False)
    assert parse_syntax(MOD).to_syntax().endswith("Y ~ X + W + X:W")


def test_dialog_builder_and_subdialogs(mod_df, monkeypatch, qapp):
    from rsstatchem.ui.dialogs import pls as dlg_mod
    from rsstatchem.ui.dialogs.base import SubDialog

    dlg = dlg_mod.PLSSEMDialog(Dataset(mod_df))
    dlg.reset()
    for name, items in (("X", ["X1", "X2", "X3"]), ("W", ["W1", "W2", "W3"]),
                        ("Y", ["Y1", "Y2", "Y3"])):
        monkeypatch.setattr(dlg_mod.QInputDialog, "getText",
                            lambda *a, n=name, **k: (n, True))
        dlg.source.clearSelection()
        for i in range(dlg.source.count()):
            if dlg.source.item(i).data(dlg_mod.NAME_ROLE) in items:
                dlg.source.item(i).setSelected(True)
        dlg._new_construct()
    dlg.from_combo.setCurrentText("X")
    dlg.to_combo.setCurrentText("Y")
    dlg.mod_combo.setCurrentText("W")
    dlg._add_path()
    assert set(dlg.model.paths) == {("X", "Y"), ("W", "Y")}
    assert dlg.model.interactions == [("X", "W", "Y")]
    assert dlg.model.to_syntax() == MOD
    # sub-dialogs build and return their values
    monkeypatch.setattr(SubDialog, "exec", lambda self: 1)
    for fn in (dlg._settings, dlg._bootstrap, dlg._quality, dlg._save):
        fn()
    assert dlg.opts["subsamples"] == 5000 and isinstance(dlg.opts["subsamples"], int)
    dlg.opts.update(subsamples=200, blindfold=False)
    dlg.accept()
    res = dlg.compute()
    assert res.structure.names == ["X", "W", "X x W", "Y"]
    med = dlg_mod.MediationTemplateDialog(Dataset(mod_df))
    assert [c.name for c in med.model().constructs] == ["X", "M", "Y"]
    for d in (dlg, med):
        d.deleteLater()
    qapp.processEvents()
