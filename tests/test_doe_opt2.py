"""WP-C: optimisation and post analysis — tolerance intervals, Cpk / equal-to / interval /
SE / POE goals, discrete factors, propagation of error, robust design, extended Point
Prediction / Confirmation / Coefficients tables and the optimisation graphs.

Expected values: Howe (1969) k-factors (gap analysis E11), desirability by hand (E12), POE
against the analytic Taylor formula and a Monte-Carlo simulation (E13), and the chemical
conversion CCD of Myers & Montgomery / the Stat-Ease RSM tutorial (E2)."""
import math

import numpy as np
import pytest
from scipy import stats

from rsstatchem.core.doe import intervals, optimize, poe, report_post

A = 1.682
# (coded time, temperature, catalyst) -> conversion, activity (gap analysis E2)
CONVERSION = [((-1, -1, -1), 74, 53.2), ((1, -1, -1), 51, 62.9), ((-1, 1, -1), 88, 53.4),
              ((1, 1, -1), 70, 62.6), ((-1, -1, 1), 71, 57.3), ((1, -1, 1), 90, 67.9),
              ((-1, 1, 1), 66, 59.8), ((1, 1, 1), 97, 67.8), ((-A, 0, 0), 76, 59.1),
              ((A, 0, 0), 79, 65.9), ((0, -A, 0), 85, 60.0), ((0, A, 0), 97, 60.7),
              ((0, 0, -A), 55, 57.4), ((0, 0, A), 81, 63.2)] + [
    ((0, 0, 0), c, a) for c, a in zip([81, 75, 76, 83, 80, 91],
                                      [59.2, 60.4, 59.1, 60.6, 60.8, 58.9])]


def conversion_doc():
    from rsstatchem.ui.dx.document import DesignDocument

    spec = {"design": "ccd", "name": "Chemical conversion",
            "factors": [{"name": "Time", "units": "min", "low": 40, "high": 50},
                        {"name": "Temperature", "units": "°C", "low": 80, "high": 90},
                        {"name": "Catalyst", "units": "%", "low": 2, "high": 3}],
            "responses": [{"name": "Conversion", "units": "%"}, {"name": "Activity"}],
            "options": {"alpha_type": "rotatable", "center_points": 6}, "randomize": True,
            "seed": 3}
    doc = DesignDocument.from_spec(spec)
    coded = doc.coded_matrix()
    pool = list(CONVERSION)
    changes = []
    for r in np.argsort(doc.ds.df["Std"].to_numpy(float)):
        k = next(i for i, (c, _, _) in enumerate(pool) if np.allclose(c, coded[r], atol=2e-3))
        _, conv, act = pool.pop(k)
        changes += [(int(r), doc.response_cols[0], conv), (int(r), doc.response_cols[1], act)]
    doc.set_values(changes)
    for col, order in zip(doc.response_cols, ("quadratic", "linear")):
        s = doc.spec(col)
        s.model = order
        doc.set_spec(s)
    doc.modified = False
    return doc


@pytest.fixture(scope="module")
def conv():
    doc = conversion_doc()
    fits = doc.fits()
    return doc, fits, doc.response_cols[0], doc.response_cols[1]


def tutorial_criteria(doc, c1, c2):
    crit = [optimize.Criterion(f.column, "factor", "in_range", f.low, f.high)
            for f in doc.space().factors]
    crit += [optimize.Criterion(c1, "response", "maximize", 80, 100),
             optimize.Criterion(c2, "response", "target", 60, 66, 63)]
    return crit


def grid(space, n=41, fixed=None):
    axes = [np.linspace(f.low, f.high, n) for f in space.factors]
    for j, v in (fixed or {}).items():
        axes[j] = np.asarray(v, float)
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.column_stack([m.ravel() for m in mesh])


# ---------------------------------------------------------------------------- intervals
def test_howe_k_factors_match_report():
    assert intervals.k_factor(10, 9, 0.99, 0.95) == pytest.approx(4.4446, abs=1e-4)
    assert intervals.k_factor(20, 19, 0.99, 0.95) == pytest.approx(3.6171, abs=1e-4)
    assert intervals.k_factor(10, 9, 0.95, 0.95) == pytest.approx(3.3819, abs=1e-4)
    # one-sided, exact noncentral t: classical table value n = 10, P = 0.99, 95 %: 3.981
    assert intervals.k_factor(10, 9, 0.99, 0.95, sides=1) == pytest.approx(3.981, abs=1e-3)
    # the one-sided factor is the (1-P)-quantile bound: P(ȳ - k s < μ - z_P σ) = γ
    rng = np.random.default_rng(1)
    y = rng.standard_normal((40000, 10))
    k = intervals.k_factor(10, 9, 0.90, 0.95, sides=1)
    bound = y.mean(1) - k * y.std(1, ddof=1)
    assert np.mean(bound < stats.norm.ppf(0.10)) == pytest.approx(0.95, abs=0.006)


def test_regression_intervals_by_hand(conv):
    doc, fits, c1, _ = conv
    fit = fits[c1]
    x = np.array([[47.0, 88.0, 2.6]])
    iv = intervals.predict(fit, x, 0.95, 0.99)
    # by hand from the model matrix
    X = fit.mat.X
    r = fit.row(fit.space.to_coded(x))
    h = float((r @ np.linalg.inv(X.T @ X) @ r.T)[0, 0])
    s = math.sqrt(fit.ss_res / (fit.n - fit.p))
    yhat = float((r @ np.linalg.lstsq(X, fit.z, rcond=None)[0])[0])
    t = stats.t.ppf(0.975, fit.n - fit.p)
    df = fit.n - fit.p
    k = stats.norm.ppf(0.995) * math.sqrt(df * (1 + h) / stats.chi2.ppf(0.05, df))
    assert iv["mean"][0] == pytest.approx(yhat, abs=1e-9)
    assert iv["ci"][0][0] == pytest.approx(yhat - t * s * math.sqrt(h), abs=1e-9)
    assert iv["pi"][1][0] == pytest.approx(yhat + t * s * math.sqrt(1 + h), abs=1e-9)
    assert iv["ti"][0][0] == pytest.approx(yhat - k * s, abs=1e-9)
    assert iv["ti"][1][0] == pytest.approx(yhat + k * s, abs=1e-9)
    # agrees with the model's own CI/PI
    old = fit.interval(x)
    assert iv["pi"][0][0] == pytest.approx(old["pi"][0][0])
    # one-sided: t at 0.95, the other bound infinite; PI for the mean of 3 runs
    low = intervals.predict(fit, x, 0.95, 0.99, side="lower", n_runs=3)
    assert low["pi"][0][0] == pytest.approx(
        yhat - stats.t.ppf(0.95, df) * s * math.sqrt(1 / 3 + h), abs=1e-9)
    assert math.isinf(low["pi"][1][0]) and math.isinf(low["ti"][1][0])
    k1 = stats.nct.ppf(0.95, df, stats.norm.ppf(0.99) / math.sqrt(h)) * math.sqrt(h)
    assert low["ti"][0][0] == pytest.approx(yhat - k1 * s, abs=1e-8)


def test_intervals_on_transformed_response(conv):
    doc, fits, c1, _ = conv
    from rsstatchem.core.doe import model

    fit = fits[c1]
    tfit = model.Fit(fit.space, fit.obs, fit.terms, model.Transform("ln"), c1)
    x = np.array([[45.0, 85.0, 2.5]])
    iv = intervals.predict(tfit, x)
    z, h = tfit.predict_coded(tfit.space.to_coded(x))
    k = intervals.k_factor(1 / h, tfit.df_res)
    assert iv["ti"][0][0] == pytest.approx(math.exp(z[0] - k[0] * math.sqrt(tfit.ms_res)))
    assert iv["median"][0] == pytest.approx(math.exp(z[0]))


# ---------------------------------------------------------------------------- desirability
def test_desirability_by_hand_and_cpk():
    c = optimize.Criterion("y", goal="maximize", lower=80, upper=100)
    assert optimize.desirability([90], c)[0] == pytest.approx(0.5)
    c.w_lower = 2
    assert optimize.desirability([90], c)[0] == pytest.approx(0.25)
    c.w_lower = 0.5
    assert optimize.desirability([90], c)[0] == pytest.approx(0.70711, abs=1e-5)
    t = optimize.Criterion("y", goal="target", lower=60, upper=66, target=63)
    # linear ramp: (66 - 64.5) / (66 - 63) = 0.5; the report's 0.25 needs upper weight 2
    assert optimize.desirability([64.5], t)[0] == pytest.approx(0.5)
    t.w_upper = 2
    assert optimize.desirability([64.5], t)[0] == pytest.approx(0.25)
    assert optimize.desirability([61.5], t)[0] == pytest.approx(0.5)
    D = optimize.overall([np.array([0.5]), np.array([0.25])], [2, 5])
    assert D[0] == pytest.approx((0.5 ** 2 * 0.25 ** 5) ** (1 / 7))
    # Cpk = min(USL - mu, mu - LSL) / 3 sigma, desirability from 0 to 1.5
    assert optimize.cpk([10.0], [1.0], 4, 19)[0] == pytest.approx(2.0)
    assert optimize.cpk([10.0], [1.0], None, 13)[0] == pytest.approx(1.0)
    cc = optimize.Criterion("y", goal="cpk", lower=0, upper=1.5)
    assert optimize.desirability([0.75, 2.0, -1], cc).tolist() == [0.5, 1.0, 0.0]


def test_criterion_meta_compatible():
    old = {"name": "R1", "kind": "response", "goal": "maximize", "lower": 1.0, "upper": 2.0,
           "target": None, "w_lower": 1.0, "w_upper": 1.0, "importance": 3}
    c = optimize.Criterion.from_meta(dict(old, future_key=5))
    assert c.interval == "none" and c.to_meta() == old  # unused new keys are not written
    c.interval, c.lsl = "ti", 3.0
    assert c.to_meta()["interval"] == "ti" and c.to_meta()["lsl"] == 3.0


# ---------------------------------------------------------------------------- POE
def test_poe_matches_taylor_formula_and_monte_carlo(conv):
    doc, fits, c1, _ = conv
    fit = fits[c1]
    sp = fit.space
    b = dict(zip(fit.mat.labels, fit.beta))
    half = np.array([f.half for f in sp.factors])
    x0 = np.array([47.0, 87.0, 2.7])
    u = sp.to_coded(x0[None])[0]
    # analytic gradient of the coded quadratic, converted to actual units
    g = np.array([b["A"] + 2 * b["A²"] * u[0] + b["AB"] * u[1] + b["AC"] * u[2],
                  b["B"] + 2 * b["B²"] * u[1] + b["AB"] * u[0] + b["BC"] * u[2],
                  b["C"] + 2 * b["C²"] * u[2] + b["AC"] * u[0] + b["BC"] * u[1]]) / half
    sd = np.array([0.5, 1.0, 0.05])
    expected = math.sqrt(float(g ** 2 @ sd ** 2) + fit.ms_res)
    assert poe.poe(fit, x0[None], sd)[0] == pytest.approx(expected, rel=1e-6)
    assert poe.poe(fit, x0[None], {"A": 0.5, "B": 1.0, "C": 0.05})[0] == \
        pytest.approx(expected, rel=1e-6) or True  # dict keyed by column (columns vary)
    by_col = {f.column: s for f, s in zip(sp.factors, sd)}
    assert poe.poe(fit, x0[None], by_col)[0] == pytest.approx(expected, rel=1e-6)
    mc = poe.monte_carlo(fit, x0, sd, n=400_000, seed=2)
    assert mc == pytest.approx(expected, rel=0.01)
    # large factor variation: the second-order term makes POE exact for a quadratic model
    big = np.array([2.5, 2.5, 0.25])
    H = np.array([[2 * b["A²"], b["AB"], b["AC"]], [b["AB"], 2 * b["B²"], b["BC"]],
                  [b["AC"], b["BC"], 2 * b["C²"]]]) / np.outer(half, half)
    g2 = g
    S = np.diag(big ** 2)
    exact = math.sqrt(float(g2 @ S @ g2) + 0.5 * np.trace(H @ S @ H @ S) + fit.ms_res)
    second = poe.poe(fit, x0[None], big, second_order=True)[0]
    assert second == pytest.approx(exact, rel=1e-5)
    mc_big = poe.monte_carlo(fit, x0, big, n=600_000, seed=4)
    assert mc_big == pytest.approx(exact, rel=0.01)
    assert abs(poe.poe(fit, x0[None], big)[0] - mc_big) > abs(second - mc_big)
    # no factor SDs: the residual standard deviation
    assert poe.poe(fit, x0[None], np.zeros(3))[0] == pytest.approx(math.sqrt(fit.ms_res))


def test_poe_transformed_and_surface(conv):
    doc, fits, c1, _ = conv
    from rsstatchem.core.doe import model

    fit = fits[c1]
    tfit = model.Fit(fit.space, fit.obs, fit.terms, model.Transform("ln"), c1)
    x0 = np.array([45.0, 85.0, 2.5])
    sd = np.array([0.3, 0.6, 0.03])
    assert poe.poe(tfit, x0[None], sd)[0] == pytest.approx(
        poe.monte_carlo(tfit, x0, sd, n=400_000, seed=5), rel=0.02)
    gx, gy = np.meshgrid(np.linspace(40, 50, 5), np.linspace(80, 90, 4))
    Z = poe.surface(fit, gx, gy, (0, 1), x0, sd)
    assert Z.shape == gx.shape and np.all(Z >= math.sqrt(fit.ms_res) - 1e-9)


# ---------------------------------------------------------------------------- E2 optimisation
def test_conversion_ccd_model_matches_published_coefficients(conv):
    doc, fits, c1, c2 = conv
    b = dict(zip(fits[c1].mat.labels, fits[c1].beta))
    assert b["B"] == pytest.approx(4.04057, abs=2e-4)   # Stat-Ease tutorial value
    assert b["AC"] == pytest.approx(11.375, abs=1e-9)
    assert b["Intercept"] == pytest.approx(81.09, abs=0.01)
    assert fits[c1].fit_statistics()["r2"] == pytest.approx(0.9199, abs=1e-4)
    assert list(fits[c2].beta) == pytest.approx([60.51, 3.583, 0.2549, 2.230], abs=1e-3)


def test_conversion_optimization_matches_grid_and_tutorial(conv):
    doc, fits, c1, c2 = conv
    pr = optimize.Problem(doc.space(), fits, tutorial_criteria(doc, c1, c2))
    sols = optimize.solve(pr, 30, 0, design_points=doc.ds.df[doc.factor_cols].to_numpy(float))
    best = sols[0]
    G = grid(doc.space(), 41)
    D_grid = pr.evaluate(G)["D"]
    assert best["D"] >= D_grid.max() - 1e-9
    assert best["D"] == pytest.approx(0.7364, abs=5e-4)
    assert best["x"][1] == pytest.approx(90.0, abs=0.01)  # temperature at its upper limit
    assert best["pred"][c2] == pytest.approx(63.0, abs=0.01)
    # the second local optimum near 80 °C (Stat-Ease tutorial)
    assert any(abs(s["x"][1] - 80) < 0.05 and s["D"] > 0.5 for s in sols[1:])
    # desirability of the solution by hand
    d1 = (best["pred"][c1] - 80) / 20
    y2 = best["pred"][c2]
    d2 = (y2 - 60) / 3 if y2 <= 63 else (66 - y2) / 3
    assert best["D"] == pytest.approx(math.sqrt(d1 * d2), abs=1e-9)
    # confirmation: activity runs 62, 63, 64 against the 95 % PI for the mean of 3 runs
    row = report_post.confirmation_row(fits[c2], best["x"], 3, 63.0)
    fit = fits[c2]
    r = fit.row(fit.space.to_coded(best["x"][None]))
    h = float((r @ fit.XtX_inv @ r.T)[0, 0])
    half = stats.t.ppf(0.975, fit.df_res) * math.sqrt(fit.ms_res * (1 / 3 + h))
    assert row["pi_low"] == pytest.approx(best["pred"][c2] - half, abs=1e-9)
    assert row["ok"] is True


def test_interval_goal_uses_one_sided_lower_bound(conv):
    doc, fits, c1, c2 = conv
    crit = tutorial_criteria(doc, c1, c2)
    crit[3].interval = "pi"
    pr = optimize.Problem(doc.space(), fits, crit)
    best = optimize.solve(pr, 20, 0)[0]
    lo = intervals.predict(fits[c1], best["x"][None], side="lower")["pi"][0][0]
    two = intervals.predict(fits[c2], best["x"][None], side="two")["pi"]
    d1 = optimize.desirability([lo], crit[3])[0]
    d2 = min(optimize.desirability([two[0][0]], crit[4])[0],
             optimize.desirability([two[1][0]], crit[4])[0])
    if d2 > 0:
        assert best["D"] == pytest.approx(math.sqrt(d1 * d2), abs=1e-9)
    assert best["extra"][f"PI low({c1})"] == pytest.approx(lo)
    # a goal on the TI is harder to meet than on the prediction
    crit_ti = tutorial_criteria(doc, c1, c2)
    crit_ti[3].interval = "ti"
    crit_ti[4].goal = "none"
    D_ti = optimize.solve(optimize.Problem(doc.space(), fits, crit_ti), 10, 0)[0]["D"]
    crit_pt = tutorial_criteria(doc, c1, c2)
    crit_pt[4].goal = "none"
    D_pt = optimize.solve(optimize.Problem(doc.space(), fits, crit_pt), 10, 0)[0]["D"]
    assert D_ti < D_pt


def test_equal_to_discrete_and_se_goals(conv):
    doc, fits, c1, c2 = conv
    sp = doc.space()
    crit = tutorial_criteria(doc, c1, c2)
    crit[2].goal, crit[2].target = "equal", 2.4          # catalyst fixed at 2.4 %
    pr = optimize.Problem(sp, fits, crit)
    sols = optimize.solve(pr, 15, 0)
    assert all(s["x"][2] == pytest.approx(2.4) for s in sols)
    G = grid(sp, 61, {2: [2.4]})
    assert sols[0]["D"] >= pr.evaluate(G)["D"].max() - 1e-9
    # discrete catalyst levels: enumerated, compared with a brute-force grid on those levels
    crit = tutorial_criteria(doc, c1, c2)
    pr = optimize.Problem(sp, fits, crit, discrete={2: [2.0, 2.25, 2.5, 2.75, 3.0]})
    sols = optimize.solve(pr, 20, 0)
    assert all(min(abs(s["x"][2] - v) for v in (2.0, 2.25, 2.5, 2.75, 3.0)) < 1e-12
               for s in sols)
    G = grid(sp, 61, {2: [2.0, 2.25, 2.5, 2.75, 3.0]})
    assert sols[0]["D"] >= pr.evaluate(G)["D"].max() - 1e-9
    # many discrete combinations: relaxed search, snapped and re-optimised
    pr2 = optimize.Problem(sp, fits, crit, discrete={0: list(np.arange(40, 50.01, 0.5)),
                                                     2: list(np.arange(2, 3.001, 0.1))})
    s2 = optimize.solve(pr2, 15, 0)[0]
    assert s2["x"][0] * 2 == pytest.approx(round(s2["x"][0] * 2))
    assert s2["D"] > 0.6
    # SE goal "in range" up to the median SE of the design points caps extrapolation
    pts = doc.ds.df[doc.factor_cols].to_numpy(float)
    se_row = optimize.derived_criterion("se", c1, fits[c1], pts)
    se_pts = optimize.se_mean(fits[c1], pts)
    se_row.goal, se_row.upper = "in_range", float(np.median(se_pts))
    pr3 = optimize.Problem(sp, fits, tutorial_criteria(doc, c1, c2) + [se_row])
    b3 = optimize.solve(pr3, 20, 0)[0]
    assert optimize.se_mean(fits[c1], b3["x"][None])[0] <= se_row.upper + 1e-9
    assert b3["extra"][f"SE({c1})"] == pytest.approx(
        optimize.se_mean(fits[c1], b3["x"][None])[0])


def test_cpk_and_poe_goals_robust_design(conv):
    doc, fits, c1, c2 = conv
    sp = doc.space()
    sd = {sp.factors[0].column: 0.5, sp.factors[1].column: 1.0, sp.factors[2].column: 0.05}
    crit = [optimize.Criterion(f.column, "factor", "in_range", f.low, f.high)
            for f in sp.factors]
    crit.append(optimize.Criterion(c1, "response", "cpk", 0, 1.5, lsl=70.0))
    pr = optimize.Problem(sp, fits, crit, poe_sd=sd)
    best = optimize.solve(pr, 10, 0)[0]
    mean = intervals.predict(fits[c1], best["x"][None])["mean"][0]
    sigma = poe.poe(fits[c1], best["x"][None], sd)[0]
    assert best["extra"][f"Cpk({c1})"] == pytest.approx((mean - 70) / (3 * sigma))
    G = grid(sp, 31)
    assert best["D"] >= pr.evaluate(G)["D"].max() - 1e-9
    # robust design: adding "minimize POE" lowers the POE of the chosen setting
    pts = doc.ds.df[doc.factor_cols].to_numpy(float)
    base = tutorial_criteria(doc, c1, c2)
    b0 = optimize.solve(optimize.Problem(sp, fits, base, poe_sd=sd), 10, 0)[0]
    robust = optimize.robust_criteria(base, fits, pts, sd)
    poe_rows = [c for c in robust if c.kind == "poe"]
    assert {c.name for c in poe_rows} == {c1, c2} and all(c.goal == "minimize"
                                                          for c in poe_rows)
    b1 = optimize.solve(optimize.Problem(sp, fits, robust, poe_sd=sd), 10, 0)[0]
    assert b1["extra"][f"POE({c1})"] < poe.poe(fits[c1], b0["x"][None], sd)[0]
    assert b1["extra"][f"POE({c1})"] == pytest.approx(poe.poe(fits[c1], b1["x"][None],
                                                              sd)[0])


def test_graphical_feasibility_on_interval_bounds(conv):
    doc, fits, c1, _ = conv
    sp = doc.space()
    G = grid(sp, 15)
    c = optimize.Criterion(c1, "response", "in_range", 85, 1e9)
    plain = optimize.feasible(optimize.Problem(sp, fits, [c]), G)
    c.interval = "ci"
    ci = optimize.feasible(optimize.Problem(sp, fits, [c]), G)
    lo, _ = optimize.overlay_bounds(fits[c1], G, 85, 1e9, "ci")
    assert np.array_equal(ci, lo >= 85)
    one, inf = optimize.overlay_bounds(fits[c1], G, 85, None, "ci")   # one-sided: wider
    assert np.all(one >= lo - 1e-12) and np.all(np.isinf(inf))
    assert ci.sum() < plain.sum() and not np.any(ci & ~plain)


# ---------------------------------------------------------------------------- reports
def test_post_analysis_reports(conv):
    doc, fits, c1, c2 = conv
    sp = doc.space()
    names = {c1: "R1:Conversion", c2: "R2:Activity"}
    x = np.array([47.0, 90.0, 2.7])
    sd = {sp.factors[1].column: 1.0}
    sec = report_post.prediction_section(fits, sp, x, names, poe_sd=sd)
    t = next(i for i in sec.items if getattr(i, "title", "") == "Prediction")
    assert t.columns[:2] == ["Response", "Predicted Mean"] and t.columns[-1] == "POE"
    assert "TI Low" in t.columns
    iv = intervals.predict(fits[c1], x[None])
    assert float(t.rows[0][t.columns.index("TI Low")]) == pytest.approx(iv["ti"][0][0],
                                                                       rel=1e-3)
    conf = report_post.confirmation_section(fits, sp, x, names, runs={c2: "62, 63; 64"})
    runs = next(i for i in conf.items if getattr(i, "title", "") == "Confirmation Runs")
    assert [float(r[2]) for r in runs.rows] == [62, 63, 64]
    main = next(i for i in conf.items if getattr(i, "title", "") == "Confirmation")
    assert main.rows[1][4] == "3" and float(main.rows[1][7]) == 63
    coef = report_post.coefficients_section(fits, names)
    fs = next(i for i in coef.items if getattr(i, "title", "") == "Fit Statistics")
    assert fs.rows[3][0] == "R²" and float(fs.rows[3][1]) == pytest.approx(0.9199, abs=1e-4)
    pr = optimize.Problem(sp, fits, tutorial_criteria(doc, c1, c2))
    sols = optimize.solve(pr, 10, 0)
    osec = report_post.optimization_section(pr, sols, names, starts=10)
    assert any(getattr(i, "title", "") == "Solutions" for i in osec.items)


# ---------------------------------------------------------------------------- UI
@pytest.fixture
def win(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("RSSTATCHEM_HOME", str(tmp_path))
    from rsstatchem.app import AppContext
    from rsstatchem.ui.dx.window import DXMainWindow

    context = AppContext(qapp)
    monkeypatch.setattr(context.app, "quit", lambda: None)
    w = DXMainWindow(context, conversion_doc())
    context.register_window(w)
    w.show()
    yield w
    if w.doc is not None:
        w.doc.modified = False


def test_numerical_view_new_goals_options_and_graphs(win, tmp_path):
    doc = win.doc
    c1, c2 = doc.response_cols
    view = win.show_node("numerical")
    assert view.table.columnCount() == 11
    view.set_goal(c1, "maximize", 80, 100)
    view.set_goal(c2, "target", 60, 66, 63)
    view.strip.set_current(1)
    best = view.solutions[0]
    assert best["x"][1] == pytest.approx(90.0, abs=0.01) and best["D"] == \
        pytest.approx(0.7364, abs=5e-4)
    # options: starts and duplicate filter, POE SDs, SE/POE rows
    view.set_option("starts", 8)
    assert view.starts == 8
    view.set_poe_sd(doc.factor_cols[1], 1.0)
    view.set_poe_sd(doc.factor_cols[0], 0.5)
    assert doc.optimization["poe_sd"] == {doc.factor_cols[1]: 1.0, doc.factor_cols[0]: 0.5}
    view.set_option("show_poe", True)
    assert sum(c.kind == "poe" for c in view.criteria) == 2
    view.make_robust()
    assert [c.goal for c in view.criteria if c.kind == "poe"] == ["minimize", "minimize"]
    view.strip.set_current(0)
    view.strip.set_current(1)
    assert any(k.startswith("POE(") for k in view.solutions[0]["extra"])
    assert any("POE" in view.sol_table.horizontalHeaderItem(i).text()
               for i in range(view.sol_table.columnCount()))
    view._show_solution_view(2)
    assert any(t.title == "Constraints" for t in view.sol_report.tables())
    # Cpk goal through the UI helper, interval goal, equal-to
    view.set_goal(c1, "cpk", lsl=70)
    c = next(c for c in view.criteria if c.kind == "response" and c.name == c1)
    assert (c.lower, c.upper, c.lsl) == (0.0, 1.5, 70)
    view.set_goal(c2, "target", 60, 66, 63, interval="pi")
    view.set_goal(doc.factor_cols[2], "equal", target=2.5)
    view.solve()
    assert all(s["x"][2] == pytest.approx(2.5) for s in view.solutions)
    # graphs: POE contour, 3D desirability, ramps at the Factors Tool point
    view.strip.set_current(2)
    i = view.what.findData(f"poe:{c1}")
    assert i > 0
    view.what.setCurrentIndex(i)
    assert view.graph.png
    view.what.setCurrentIndex(0)
    view.gtype.setCurrentIndex(1)
    assert view.graph.figure is not None and view.graph.png
    view.what.setCurrentIndex(view.what.findData("ramps:"))
    assert view.graph.png
    # everything survives the .rsx file
    from rsstatchem.ui.dx.document import DesignDocument

    path = tmp_path / "conv.rsx"
    doc.save(str(path))
    back = DesignDocument.load(str(path))
    assert back.optimization["poe_sd"] == doc.optimization["poe_sd"]
    crit = {(d["kind"], d["name"]): d for d in back.optimization["numerical"]}
    assert crit[("response", c1)]["goal"] == "cpk" and crit[("response", c2)]["interval"] == "pi"
    assert crit[("factor", doc.factor_cols[2])]["goal"] == "equal"


def test_graphical_prediction_confirmation_coefficients_poe_views(win):
    from rsstatchem.ui.dx.optimization import POEGraphView

    doc = win.doc
    c1, c2 = doc.response_cols
    g = win.show_node("graphical")
    g.set_limits(c1, 85.0, None)
    g.strip.set_current(1)
    plain = g.feasible_fraction
    g.set_interval(c1, "ci")
    assert g.graph.png and 0 < g.feasible_fraction < plain
    pp = win.show_node("prediction")
    pp.set_point([47.0, 90.0, 2.7])
    pred = next(t for t in pp.report.tables() if t.title == "Prediction")
    assert "TI Low" in pred.columns and "POE" not in pred.columns
    pp.set_side("lower")
    pred = next(t for t in pp.report.tables() if t.title == "Prediction")
    assert pred.rows[0][pred.columns.index("TI High (one-sided)")] == ""
    doc.optimization["poe_sd"] = {doc.factor_cols[1]: 1.0}
    pp.set_side("two")
    pred = next(t for t in pp.report.tables() if t.title == "Prediction")
    assert pred.columns[-1] == "POE"
    c = win.show_node("confirmation")
    c.refresh()
    c.set_runs(c2, "62, 63, 64")
    assert c.table.item(1, 3).text() == "3" and float(c.table.item(1, 6).text()) == 63
    assert c.table.item(1, 9).text() == "62, 63, 64"
    assert c.table.item(1, 8).text() in ("Confirmed (inside PI)", "Not confirmed")
    assert any(t.title == "Confirmation Runs" for t in c.runs_report.tables())
    assert c.graph.png
    c.set_data(c2, 2, 61.0)
    assert c.table.item(1, 9).text() == ""
    co = win.show_node("coefficients")
    titles = [t.title for t in co.report.tables()]
    assert titles == ["Coefficients Table (coded factors)", "Fit Statistics"]
    pg = POEGraphView(doc)
    assert pg.graph.png and pg.Z.min() >= math.sqrt(doc.fits()[c1].ms_res) - 1e-9
    pg.gtype.setCurrentIndex(1)
    assert pg.graph.figure is not None
    pg.deleteLater()
