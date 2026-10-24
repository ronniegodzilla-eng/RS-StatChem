"""DOE / RSM module checked against textbook values (Montgomery, Design and Analysis of
Experiments) and statsmodels / scipy reference implementations."""
import itertools
import json
import math

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.outliers_influence import OLSInfluence

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.doe import designs, model, optimize, report


# ---------------------------------------------------------------------------- data
def montgomery_11_2():
    """Example 11.2: CCD, chemical process yield, coded time (x1) and temperature (x2)."""
    a = 1.414
    x1 = [-1, -1, 1, 1, 0, 0, 0, 0, 0, a, -a, 0, 0]
    x2 = [-1, 1, -1, 1, 0, 0, 0, 0, 0, 0, 0, a, -a]
    y = [76.5, 77.0, 78.0, 79.5, 79.9, 80.3, 80.0, 79.7, 79.8, 78.4, 75.6, 78.5, 77.0]
    df = pd.DataFrame({"Time": 85 + 5 * np.array(x1), "Temp": 175 + 5 * np.array(x2),
                       "Yield": y})
    ds = Dataset(df)
    ds.meta["doe"] = {"kind": "process", "design": "ccd",
                      "factors": [{"name": "Time", "column": "Time", "low": 80, "high": 90},
                                  {"name": "Temp", "column": "Temp", "low": 170,
                                   "high": 180}],
                      "responses": [{"name": "Yield", "column": "Yield", "units": "%"}]}
    return ds, np.array(x1, float), np.array(x2, float), np.array(y)


@pytest.fixture
def mont():
    return montgomery_11_2()


def filtration():
    """Example 6.2: 2^4 filtration rate (standard order)."""
    y = [45, 71, 48, 65, 68, 60, 80, 65, 43, 100, 45, 104, 75, 86, 70, 96]
    res = designs.build_design({"design": "factorial", "randomize": False,
                                "factors": [{"name": n, "low": -1, "high": 1} for n in "ABCD"]})
    ds = res.dataset
    ds.df["R1"] = y
    return ds


# ---------------------------------------------------------------------------- the CCD fit
def test_montgomery_11_2_coefficients(mont):
    ds, *_ = mont
    a = model.analyze(ds, model.AnalysisSpec("Yield"))
    assert a.summary.suggested == "Quadratic"
    eq = {t: c for c, t in a.fit.coded_equation()}
    assert eq[""] == pytest.approx(79.94, abs=0.005)
    assert eq["A"] == pytest.approx(0.995, abs=0.0005)
    assert eq["B"] == pytest.approx(0.515, abs=0.0005)
    assert eq["A²"] == pytest.approx(-1.376, abs=0.0005)
    assert eq["B²"] == pytest.approx(-1.001, abs=0.0005)
    assert eq["AB"] == pytest.approx(0.25, abs=1e-9)
    an = a.fit.anova()
    ss_l, df_l, ss_pe, df_pe = an["lof"]
    assert (df_l, df_pe) == (3, 4)
    assert ss_pe == pytest.approx(0.2120, abs=1e-4)
    assert ss_l == pytest.approx(0.2843, abs=1e-3)


def test_anova_and_diagnostics_match_statsmodels(mont):
    ds, x1, x2, y = mont
    a = model.analyze(ds, model.AnalysisSpec("Yield", model="quadratic"))
    fit = a.fit
    df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
    ols = smf.ols("y ~ x1 + x2 + x1:x2 + I(x1**2) + I(x2**2)", df).fit()
    st = fit.fit_statistics()
    assert st["r2"] == pytest.approx(ols.rsquared, abs=1e-9)
    assert st["adj_r2"] == pytest.approx(ols.rsquared_adj, abs=1e-9)
    assert st["std_dev"] == pytest.approx(math.sqrt(ols.mse_resid), rel=1e-9)
    infl = OLSInfluence(ols)
    press = float(np.sum((ols.resid / (1 - infl.hat_matrix_diag)) ** 2))
    assert st["press"] == pytest.approx(press, rel=1e-9)
    assert st["pred_r2"] == pytest.approx(1 - press / ols.centered_tss, rel=1e-9)
    d = fit.diagnostics()
    assert np.allclose(d["leverage"], infl.hat_matrix_diag)
    assert np.allclose(d["ext_stud"], infl.resid_studentized_external)
    assert np.allclose(d["int_stud"], infl.resid_studentized_internal)
    assert np.allclose(d["cooks"], infl.cooks_distance[0])
    assert np.allclose(d["dffits"], infl.dffits[0])
    typ3 = anova_lm(ols, typ=3)
    partial = {t: ss for _, ss, _, t in fit.anova()["terms"]}
    assert partial[(0,)] == pytest.approx(typ3.loc["x1", "sum_sq"], rel=1e-8)
    assert partial[(1, 1)] == pytest.approx(typ3.loc["I(x2 ** 2)", "sum_sq"], rel=1e-8)
    assert partial[(0, 1)] == pytest.approx(typ3.loc["x1:x2", "sum_sq"], rel=1e-8)
    f, p = fit.model_test()
    assert f == pytest.approx(ols.fvalue, rel=1e-9)
    assert p == pytest.approx(ols.f_pvalue, rel=1e-6)
    se = {c["label"]: c["se"] for c in fit.coefficients()}
    assert se["A"] == pytest.approx(ols.bse["x1"], rel=1e-9)
    # adequate precision from its definition
    s2 = ols.mse_resid
    assert st["adeq"] == pytest.approx(np.ptp(ols.fittedvalues) / math.sqrt(6 * s2 / 13))


def test_actual_equation_equals_fit_in_natural_units(mont):
    ds, x1, x2, y = mont
    a = model.analyze(ds, model.AnalysisSpec("Yield", model="quadratic"))
    (_, eq), = a.fit.actual_equations()
    coef = {t: c for c, t in eq}
    df = pd.DataFrame({"t": 85 + 5 * x1, "T": 175 + 5 * x2, "y": y})
    ols = smf.ols("y ~ t + T + I(t**2) + I(T**2) + t:T", df).fit()
    assert coef[""] == pytest.approx(ols.params["Intercept"], rel=1e-7)
    assert coef["Time"] == pytest.approx(ols.params["t"], rel=1e-7)
    assert coef["Temp²"] == pytest.approx(ols.params["I(T ** 2)"], rel=1e-7)
    assert coef["Time * Temp"] == pytest.approx(ols.params["t:T"], rel=1e-7)
    # Montgomery: -1430.52 + 7.81 t + 13.27 T - 0.055 t² - 0.04 T² + 0.01 tT
    assert coef[""] == pytest.approx(-1430.5, abs=1.0)
    assert coef["Time"] == pytest.approx(7.81, abs=0.01)


def test_fit_summary_sequential_ss(mont):
    ds, *_ = mont
    a = model.analyze(ds, model.AnalysisSpec("Yield"))
    seq = {r[0]: r for r in a.summary.sequential}
    total_cor = np.sum((ds.df["Yield"] - ds.df["Yield"].mean()) ** 2)
    parts = sum(seq[k][1] for k in ("Linear vs Mean", "2FI vs Linear", "Quadratic vs 2FI",
                                    "Cubic vs Quadratic", "Residual"))
    assert parts == pytest.approx(total_cor)
    assert seq["Cubic vs Quadratic"][5] == "Aliased"
    assert seq["Quadratic vs 2FI"][5] == "Suggested"
    assert seq["Quadratic vs 2FI"][4] < 0.0001


def test_prediction_interval_matches_statsmodels(mont):
    ds, x1, x2, y = mont
    a = model.analyze(ds, model.AnalysisSpec("Yield", model="quadratic"))
    iv = a.fit.interval(np.array([[87.0, 176.0]]))
    df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
    ols = smf.ols("y ~ x1 + x2 + x1:x2 + I(x1**2) + I(x2**2)", df).fit()
    pr = ols.get_prediction(pd.DataFrame({"x1": [0.4], "x2": [0.2]})).summary_frame(0.05)
    assert iv["mean"][0] == pytest.approx(pr["mean"][0])
    assert iv["ci"][0][0] == pytest.approx(pr["mean_ci_lower"][0])
    assert iv["pi"][1][0] == pytest.approx(pr["obs_ci_upper"][0])


def test_optimum_is_stationary_point(mont):
    ds, *_ = mont
    a = model.analyze(ds, model.AnalysisSpec("Yield", model="quadratic"))
    crit = optimize.default_criteria({"Yield": a.fit}, a.space)
    crit[-1].lower, crit[-1].upper = 75.0, 81.0
    sols = optimize.solve(optimize.Problem(a.space, {"Yield": a.fit}, crit), starts=10,
                          seed=3)
    x = a.space.to_coded(sols[0]["x"][None, :])[0]
    # Montgomery: stationary point x1 = 0.389, x2 = 0.306, predicted yield 80.21
    assert x[0] == pytest.approx(0.389, abs=0.005)
    assert x[1] == pytest.approx(0.306, abs=0.005)
    assert sols[0]["pred"]["Yield"] == pytest.approx(80.21, abs=0.01)
    assert sols[0]["D"] == pytest.approx((80.21 - 75) / 6, abs=0.003)


# ---------------------------------------------------------------------------- designs
@pytest.mark.parametrize("k", [2, 3, 4])
def test_ccd_rotatable_alpha(k):
    res = designs.central_composite(k)
    assert res["alpha"] == pytest.approx((2 ** k) ** 0.25)
    coded = res["coded"]
    assert len(coded) == 2 ** k + 2 * k + designs.CCD_CENTERS[k]
    # rotatability: equal prediction variance at equal distance from the centre
    X = model.raw_matrix(coded, model.process_terms(k, "quadratic"))
    inv = np.linalg.inv(X.T @ X)
    r = 0.8
    p1 = np.zeros(k)
    p1[0] = r
    p2 = np.zeros(k)
    p2[:2] = r / math.sqrt(2)
    v = [float(model.raw_matrix(p[None], model.process_terms(k, "quadratic"))[0] @ inv @
               model.raw_matrix(p[None], model.process_terms(k, "quadratic"))[0])
         for p in (p1, p2)]
    assert v[0] == pytest.approx(v[1])


def test_ccd_variants():
    assert designs.central_composite(3, ccd_type="face")["alpha"] == 1.0
    assert designs.ccd_alpha("orthogonal", 2, 4, 16) == pytest.approx(math.sqrt(2))
    assert designs.ccd_alpha("practical", 4, 16, 30) == pytest.approx(4 ** 0.25)
    assert designs.ccd_alpha("spherical", 3, 8, 20) == pytest.approx(math.sqrt(3))
    ins = designs.central_composite(2, ccd_type="inscribed")
    assert np.max(np.abs(ins["coded"])) == pytest.approx(1.0)
    half = designs.central_composite(5, generators="E=ABCD")
    assert half["nf"] == 16 and half["alpha"] == pytest.approx(2.0)
    blk = designs.central_composite(2, alpha_type="blocking", blocks=2, center_points=3,
                                    center_axial=3)
    assert blk["alpha"] == pytest.approx(math.sqrt(2))
    assert sorted(set(blk["blocks"])) == [1, 2]


def test_orthogonal_blocking_recovers_model():
    res = designs.build_design({"design": "ccd", "randomize": False,
                                "factors": [{"name": "A"}, {"name": "B"}],
                                "options": {"blocks": 2, "alpha_type": "blocking",
                                            "center_points": 3, "center_axial": 3}})
    ds = res.dataset
    x = ds.df[["A", "B"]].to_numpy()
    ds.df["R1"] = (10 + 2 * x[:, 0] - x[:, 1] + 0.5 * x[:, 0] * x[:, 1] - 1.5 * x[:, 0] ** 2
                   + 5.0 * (ds.df["Block"] == 2))
    a = model.analyze(ds, model.AnalysisSpec("R1", model="quadratic"))
    eq = {t: c for c, t in a.fit.coded_equation()}
    assert eq["A"] == pytest.approx(2) and eq["A²"] == pytest.approx(-1.5)
    assert eq["B²"] == pytest.approx(0, abs=1e-9)
    assert a.fit.anova()["block"][0] > 0


@pytest.mark.parametrize("k,runs", [(3, 17), (4, 29), (5, 46), (6, 54), (7, 62)])
def test_box_behnken_runs(k, runs):
    res = designs.box_behnken(k)
    c = res["coded"]
    assert len(c) == runs
    assert set(np.unique(c)) == {-1.0, 0.0, 1.0}
    assert np.allclose(c.sum(axis=0), 0)
    X = model.raw_matrix(c, model.process_terms(k, "quadratic"))
    assert np.linalg.matrix_rank(X) == X.shape[1]


@pytest.mark.parametrize("k", [2, 3, 4, 5])
def test_two_level_factorial_orthogonal(k):
    X = model.raw_matrix(designs.full_factorial(k),
                         [t for r in range(1, k + 1) for t in itertools.combinations(range(k), r)])
    assert np.allclose(X.T @ X, 2 ** k * np.eye(2 ** k))


def test_fractional_alias_structure():
    coded, group = designs.fractional_factorial(4, "D=ABC")
    assert designs.resolution(group) == 4
    assert [designs.word_str(w) for w, _ in group] == ["ABCD"]
    aliases = designs.alias_structure(4, group)
    for line in ("[A] = A + BCD", "[B] = B + ACD", "[C] = C + ABD", "[D] = D + ABC",
                 "[AB] = AB + CD", "[AC] = AC + BD", "[AD] = AD + BC"):
        assert line in aliases
    assert np.allclose(coded[:, 3], coded[:, 0] * coded[:, 1] * coded[:, 2])
    X = model.raw_matrix(coded, model.process_terms(4, "linear"))
    assert np.allclose(X.T @ X, 8 * np.eye(5))


@pytest.mark.parametrize("k,p,res", [(5, 1, 5), (5, 2, 3), (6, 2, 4), (7, 4, 3), (8, 4, 4),
                                     (9, 5, 3), (11, 7, 3)])
def test_fractional_resolution(k, p, res):
    coded, group = designs.fractional_factorial(k, designs.default_generators(k, p))
    assert len(coded) == 2 ** (k - p)
    assert designs.resolution(group) == res
    X = model.raw_matrix(coded, model.process_terms(k, "linear"))
    assert np.allclose(X.T @ X, len(coded) * np.eye(k + 1))


def test_fractional_blocks_and_generators():
    coded = designs.full_factorial(5)
    words = designs.block_words(5, [], 4, n_base=5)
    assert all(designs.popcount(w) >= 3 for w in words)
    blocks = designs.assign_blocks(coded, words)
    assert np.bincount(blocks)[1:].tolist() == [8, 8, 8, 8]
    X = model.raw_matrix(coded, model.process_terms(5, "2fi"))
    B = np.column_stack([(blocks == b).astype(float) - 0.25 for b in range(1, 5)])
    assert np.allclose(X[:, 1:].T @ B, 0)  # blocks orthogonal to main effects and 2FI
    coded, group = designs.fractional_factorial(5, "E=ABCD")
    words = designs.block_words(5, group, 2, n_base=4)
    assert np.bincount(designs.assign_blocks(coded, words))[1:].tolist() == [8, 8]
    with pytest.raises(ValueError):
        designs.parse_generators("D=AD", 4)


@pytest.mark.parametrize("k", [3, 7, 11, 15, 19, 23, 31, 39])
def test_plackett_burman_orthogonal(k):
    d = designs.plackett_burman(k)
    n = len(d)
    assert n % 4 == 0 and n >= k + 1
    assert np.allclose(d.sum(axis=0), 0)
    assert np.allclose(d.T @ d, n * np.eye(k))


@pytest.mark.parametrize("name", list(designs.TAGUCHI_ARRAYS))
def test_taguchi_arrays_orthogonal(name):
    arr = designs.taguchi_array(name)
    n, levels = designs.TAGUCHI_ARRAYS[name]
    assert arr.shape == (n, len(levels))
    for j, lv in enumerate(levels):
        assert sorted(set(arr[:, j])) == list(range(1, lv + 1))
        assert len(set(np.bincount(arr[:, j])[1:])) == 1
    for i, j in itertools.combinations(range(arr.shape[1]), 2):
        pairs = {}
        for a, b in zip(arr[:, i], arr[:, j]):
            pairs[(a, b)] = pairs.get((a, b), 0) + 1
        assert len(pairs) == levels[i] * levels[j]
        assert len(set(pairs.values())) == 1
    assert np.all(arr[0] == 1)


def test_mixture_designs():
    lat = designs.simplex_lattice(3, 2)
    assert len(lat) == 6 and np.allclose(lat.sum(axis=1), 1)
    assert len(designs.simplex_lattice(4, 3)) == math.comb(6, 3)
    cen = designs.simplex_centroid(4)
    assert len(cen) == 15
    lower, upper = [0.1, 0.1, 0.05], [0.6, 0.5, 0.5]
    verts, _ = designs.extreme_vertices(lower, upper, 1.0)
    # brute force: intersections of pairs of constraint lines in the (x1, x2) plane
    lines = [(1, 0, 0.1), (1, 0, 0.6), (0, 1, 0.1), (0, 1, 0.5), (1, 1, 0.95), (1, 1, 0.5)]
    ref = set()
    for (a1, b1, c1), (a2, b2, c2) in itertools.combinations(lines, 2):
        m = np.array([[a1, b1], [a2, b2]], float)
        if abs(np.linalg.det(m)) < 1e-12:
            continue
        x = np.linalg.solve(m, [c1, c2])
        x3 = 1 - x.sum()
        if 0.1 - 1e-9 <= x[0] <= 0.6 + 1e-9 and 0.1 - 1e-9 <= x[1] <= 0.5 + 1e-9 \
                and 0.05 - 1e-9 <= x3 <= 0.5 + 1e-9:
            ref.add((round(x[0], 9), round(x[1], 9)))
    assert {(round(v[0], 9), round(v[1], 9)) for v in verts} == ref
    pts, types = designs.mixture_candidates(lower, upper, 1.0)
    assert np.allclose(pts.sum(axis=1), 1)
    assert np.all(pts >= np.array(lower) - 1e-9) and np.all(pts <= np.array(upper) + 1e-9)
    assert (types == 3).sum() == len(verts)  # a polygon has as many edges as vertices


def test_d_optimal_simple_cases():
    rng = np.random.default_rng(0)
    cand = np.array([[-1.0], [0.0], [1.0], [0.5], [-0.5]])
    F = model.raw_matrix(cand, model.process_terms(1, "linear"))
    idx = designs.d_optimal(F, 2, rng)
    assert sorted(cand[idx, 0]) == [-1, 1]
    F = model.raw_matrix(cand, model.process_terms(1, "quadratic"))
    assert sorted(cand[designs.d_optimal(F, 3, rng), 0]) == [-1, 0, 1]
    cand = designs.process_candidates(2)
    F = model.raw_matrix(cand, model.process_terms(2, "quadratic"))
    idx = designs.d_optimal(F, 6, rng)
    best = np.linalg.det(F[idx].T @ F[idx])
    for _ in range(200):
        r = rng.choice(len(cand), 6, replace=False)
        assert np.linalg.det(F[r].T @ F[r]) <= best + 1e-9


def test_build_design_dataset_and_meta():
    spec = {"design": "ccd", "seed": 11, "factors": [{"name": "Temp", "units": "C",
                                                      "low": 50, "high": 70},
                                                     {"name": "pH", "low": 4, "high": 6}],
            "responses": [{"name": "Yield", "units": "%"}]}
    r1, r2 = designs.build_design(spec), designs.build_design(spec)
    ds = r1.dataset
    assert ds.names()[:4] == ["Std", "Run", "Block", "Space_Type"]
    assert ds.df["Run"].tolist() == list(range(1, 14))
    assert sorted(ds.df["Std"]) == list(range(1, 14))
    assert ds.df["Std"].tolist() == r2.dataset.df["Std"].tolist()  # seed reproducible
    assert ds.df["Yield"].isna().all()
    assert ds.df["Temp"].max() == pytest.approx(60 + 10 * math.sqrt(2))
    json.dumps(ds.meta["doe"])
    assert ds.meta["doe"]["factors"][0]["low"] == 50
    sp = model.space_from_dataset(ds)
    assert sp.factors[1].name == "pH" and sp.factors[0].units == "C"
    for kind, n in (("bbd", 17), ("threelevel", 27), ("pb", 12), ("lattice", 6),
                    ("centroid", 7)):
        f = [{"name": c} for c in ("X", "Y", "Z")]
        if kind == "pb":
            f = [{"name": c} for c in "ABCDEFGHJK"]
        res = designs.build_design({"design": kind, "factors": f, "seed": 1})
        assert res.meta["runs"] == n, kind
        json.dumps(res.meta)
    res = designs.build_design({"design": "general", "seed": 1, "factors": [
        {"name": "Mat", "levels": ["Steel", "Brass"]}, {"name": "T", "levels": ["15", "70",
                                                                               "125"]}],
        "options": {"replicates": 2}})
    assert res.meta["runs"] == 12
    assert res.dataset.var("Mat").value_labels == {1.0: "Steel", 2.0: "Brass"}
    assert sorted(set(res.dataset.df["T"])) == [15, 70, 125]


# ---------------------------------------------------------------------------- effects etc.
def test_filtration_effects_and_backward():
    ds = filtration()
    a = model.analyze(ds, model.AnalysisSpec("R1"))
    eff = {e["name"]: e["effect"] for e in model.factorial_effects(a.space, a.obs)}
    expected = {"A": 21.625, "B": 3.125, "C": 9.875, "D": 14.625, "AB": 0.125, "AC": -18.125,
                "AD": 16.625, "BC": 2.375, "BD": -0.375, "CD": -1.125, "ABC": 1.875,
                "ABD": 4.125, "ACD": -1.625, "BCD": -2.625, "ABCD": 1.375}
    for k, v in expected.items():
        assert eff[k] == pytest.approx(v)
    b = model.analyze(ds, model.AnalysisSpec("R1", model="2fi", reduction="backward",
                                             alpha_out=0.05))
    assert {model.term_name(t) for t in b.fit.terms} == {"A", "C", "D", "AC", "AD"}
    eq = {t: c for c, t in b.fit.coded_equation()}
    assert eq[""] == pytest.approx(70.0625) and eq["AC"] == pytest.approx(-9.0625)
    assert eq["AD"] == pytest.approx(8.3125)


def test_box_cox():
    rng = np.random.default_rng(5)
    y = rng.lognormal(2, 0.4, 40)
    ds = Dataset(pd.DataFrame({"x": rng.normal(size=40), "y": y}))
    sp = model.space_from_dataset(ds, ["x"])
    obs = model.observations(ds, sp, "y")
    bc = model.box_cox(sp, obs, [])
    lam = stats.boxcox_normmax(y, method="mle")
    assert bc["best"] == pytest.approx(lam, abs=2e-3)
    # drill advance data (Montgomery Example 6.3 / Box-Cox): log is recommended
    drill = [1.68, 1.98, 3.28, 3.44, 4.98, 5.70, 9.97, 9.07, 2.07, 2.44, 4.09, 4.53, 7.77,
             9.43, 11.75, 16.30]
    ds = filtration()
    ds.df["R1"] = drill
    sp = model.space_from_dataset(ds)
    obs = model.observations(ds, sp, "R1")
    bc = model.box_cox(sp, obs, [(1,), (2,), (3,)])
    assert bc["low"] < 0 < bc["high"] < 1
    assert bc["recommended"] == 0 and bc["recommended_name"] == "Log"
    a = model.analyze(ds, model.AnalysisSpec("R1", model="manual", terms=["1", "2", "3"],
                                             transform=model.Transform("ln")))
    direct = smf.ols("np.log(y) ~ b + c + d", pd.DataFrame({
        "y": drill, "b": obs.coded[:, 1], "c": obs.coded[:, 2], "d": obs.coded[:, 3]})).fit()
    assert a.fit.fit_statistics()["r2"] == pytest.approx(direct.rsquared)
    pred = a.fit.predict(obs.actual[:1])
    assert pred[0] == pytest.approx(math.exp(direct.fittedvalues[0]))


def test_desirability_arithmetic():
    c = optimize.Criterion("y", goal="maximize", lower=0, upper=10)
    assert optimize.desirability([5, -1, 12], c).tolist() == [0.5, 0.0, 1.0]
    c.w_lower = 2
    assert optimize.desirability([5], c)[0] == pytest.approx(0.25)
    c = optimize.Criterion("y", goal="minimize", lower=0, upper=10, w_upper=0.5)
    assert optimize.desirability([7.5], c)[0] == pytest.approx(0.5)
    c = optimize.Criterion("y", goal="target", lower=0, upper=10, target=5)
    assert optimize.desirability([2.5, 7.5, 5, 11], c).tolist() == [0.5, 0.5, 1.0, 0.0]
    c.target, c.w_upper = 4, 2
    assert optimize.desirability([7], c)[0] == pytest.approx(0.25)
    c = optimize.Criterion("y", goal="in_range", lower=1, upper=2)
    assert optimize.desirability([0.5, 1.5], c).tolist() == [0, 1]
    D = optimize.overall([np.array([0.5]), np.array([0.8])], [1, 1])
    assert D[0] == pytest.approx(math.sqrt(0.4))
    D = optimize.overall([np.array([0.5]), np.array([0.8])], [1, 3])
    assert D[0] == pytest.approx((0.5 * 0.8 ** 3) ** 0.25)
    assert optimize.overall([np.array([0.0]), np.array([0.8])], [1, 1])[0] == 0


def test_scheffe_mixture_yarn_elongation():
    """Montgomery Example 11.5 (Cornell): yarn elongation, simplex lattice {3, 2}."""
    pts = [((1, 0, 0), [11.0, 12.4]), ((0.5, 0.5, 0), [15.0, 14.8, 16.1]),
           ((0, 1, 0), [8.8, 10.0]), ((0, 0.5, 0.5), [10.0, 9.7, 11.8]),
           ((0, 0, 1), [16.8, 16.0]), ((0.5, 0, 0.5), [17.7, 16.4, 16.6])]
    rows = [(*x, v) for x, vals in pts for v in vals]
    ds = Dataset(pd.DataFrame(rows, columns=["PE", "PS", "PP", "Elong"]))
    ds.meta["doe"] = {"kind": "mixture", "mixture": {"total": 1.0},
                      "factors": [{"name": n, "column": n, "type": "mixture", "low": 0,
                                   "high": 1} for n in ("PE", "PS", "PP")],
                      "responses": [{"name": "Elongation", "column": "Elong"}]}
    a = model.analyze(ds, model.AnalysisSpec("Elong", model="quadratic"))
    eq = {t: c for c, t in a.fit.coded_equation()}
    for term, v in (("A", 11.7), ("B", 9.4), ("C", 16.4), ("AB", 19.0), ("AC", 11.4),
                    ("BC", -9.6)):
        assert eq[term] == pytest.approx(v, abs=0.05)
    ols = smf.ols("Elong ~ PE + PS + PP + PE:PS + PE:PP + PS:PP - 1", ds.df).fit()
    assert a.fit.fit_statistics()["r2"] == pytest.approx(
        1 - ols.ssr / np.sum((ds.df["Elong"] - ds.df["Elong"].mean()) ** 2))
    assert a.fit.anova()["terms"][0][0] == "Linear Mixture"
    (_, actual), = a.fit.actual_equations()
    assert dict((t, c) for c, t in actual)["PE * PS"] == pytest.approx(eq["AB"])
    assert a.summary.sequential[1][0] == "Linear vs Mean"


def test_constrained_mixture_pseudo_components():
    res = designs.build_design({"design": "ev", "seed": 1, "factors": [
        {"name": "W", "low": 0.1, "high": 0.6}, {"name": "X", "low": 0.1, "high": 0.5},
        {"name": "Y", "low": 0.05, "high": 0.5}],
        "options": {"include": ["vertex", "edge", "centroid", "interior"]}})
    ds = res.dataset
    A = ds.df[["W", "X", "Y"]].to_numpy()
    assert np.allclose(A.sum(axis=1), 1)
    ds.df["R1"] = 10 * A[:, 0] + 5 * A[:, 1] + 8 * A[:, 2] + 20 * A[:, 0] * A[:, 1]
    a = model.analyze(ds, model.AnalysisSpec("R1", model="quadratic"))
    (_, actual), = a.fit.actual_equations()
    c = {t: v for v, t in actual}
    assert c["W"] == pytest.approx(10) and c["W * X"] == pytest.approx(20)
    assert c["X * Y"] == pytest.approx(0, abs=1e-8)
    assert a.fit.predict(np.array([[0.3, 0.3, 0.4]]))[0] == pytest.approx(
        3 + 1.5 + 3.2 + 20 * 0.09)


def test_taguchi_sn():
    Y = np.array([[10.0, 12.0, 11.0], [5.0, 5.0, 6.0]])
    assert optimize.sn_ratio(Y, "larger")[0] == pytest.approx(
        -10 * math.log10(np.mean(1 / Y[0] ** 2)))
    assert optimize.sn_ratio(Y, "smaller")[1] == pytest.approx(
        -10 * math.log10(np.mean(Y[1] ** 2)))
    assert optimize.sn_ratio(Y, "nominal")[0] == pytest.approx(
        10 * math.log10(11 ** 2 / 1.0))
    rt = optimize.response_table(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
                                 np.array([1.0, 2.0, 5.0, 8.0]), [2, 2])
    assert rt["means"][0] == [1.5, 6.5] and rt["delta"] == [5.0, 2.0] and rt["rank"] == [1, 2]


def test_general_factorial_categoric_anova():
    rng = np.random.default_rng(2)
    res = designs.build_design({"design": "general", "seed": 3, "factors": [
        {"name": "Mat", "levels": ["a", "b", "c"]}, {"name": "T", "levels": ["15", "70",
                                                                            "125"]}],
        "options": {"replicates": 4}})
    ds = res.dataset
    ds.df["R1"] = rng.normal(100, 5, 36) + 0.2 * ds.df["T"] + 10 * (ds.df["Mat"] == 2)
    a = model.analyze(ds, model.AnalysisSpec("R1", model="2fi"))
    df = ds.df.assign(M=ds.df["Mat"].astype(int).astype(str), Tc=ds.df["T"].astype(str))
    ols = smf.ols("R1 ~ C(M, Sum) * C(Tc, Sum)", df).fit()
    typ3 = anova_lm(ols, typ=3)
    terms = {t: (ss, d) for _, ss, d, t in a.fit.anova()["terms"]}
    assert terms[(0,)][0] == pytest.approx(typ3.loc["C(M, Sum)", "sum_sq"])
    assert terms[(0, 1)] == (pytest.approx(typ3.loc["C(M, Sum):C(Tc, Sum)", "sum_sq"]), 4)
    assert a.fit.fit_statistics()["r2"] == pytest.approx(ols.rsquared)


def test_report_sections_render(mont):
    ds, *_ = mont
    a = model.analyze(ds, model.AnalysisSpec("Yield"))
    sec = report.anova_section(a)
    titles = [getattr(i, "title", "") for i in sec.items]
    assert "ANOVA for Quadratic model" in titles
    anova = next(i for i in sec.items if i.title == "ANOVA for Quadratic model")
    model_row = anova.rows[0]
    assert model_row[0] == "Model" and model_row[2] == "5" and model_row[6] == "significant"
    lof = next(r for r in anova.rows if r[0].strip() == "Lack of Fit")
    assert lof[6] == "not significant"
    ref = report.default_point(a.space)
    for g in ("contour", "surface", "perturbation", "interaction", "one_factor"):
        s = report.graph_section(a, g, [0, 1], ref)
        assert any(i.kind == "image" for i in s.items)
    assert report.pval(0.00001) == "< 0.0001" and report.num(79.93995) == "79.94"


def test_selftest():
    from rsstatchem.ui.dialogs.doe import MENUS, selftest

    selftest()
    assert "RSM" in MENUS
