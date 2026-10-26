"""RSM / DOE statistics added after 0.6.0: categoric factors with numeric ones, REML split-plot
analysis (Kenward-Roger), minimum-run designs, foldovers, Auto Select (forward / stepwise /
AICc / BIC) and DFBETAS — checked against Montgomery, Design and Analysis of Experiments, and
statsmodels."""
import itertools
import math

import numpy as np
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.outliers_influence import OLSInfluence

from rsstatchem.core.doe import designs, model, optimize, report

# Montgomery Table 14.16, paper tensile strength: temperature rows, (replicate, pulp) columns
PULP = np.array([[30, 34, 29, 28, 31, 31, 31, 35, 32], [35, 41, 26, 32, 36, 30, 37, 40, 34],
                 [37, 38, 33, 40, 42, 32, 41, 39, 39], [36, 42, 36, 41, 40, 40, 40, 44, 45]],
                float)
# Montgomery Table 5.1, battery life: material x temperature (15, 70, 125 °F)
BATTERY = {(1, 15): [130, 155, 74, 180], (1, 70): [34, 40, 80, 75], (1, 125): [20, 70, 82, 58],
           (2, 15): [150, 188, 159, 126], (2, 70): [136, 122, 106, 115],
           (2, 125): [25, 70, 58, 45], (3, 15): [138, 110, 168, 160],
           (3, 70): [174, 120, 150, 139], (3, 125): [96, 104, 82, 60]}


def pulp_design():
    res = designs.build_design({
        "design": "general", "seed": 5,
        "factors": [{"name": "Pulp", "levels": ["a1", "a2", "a3"], "changes": "hard"},
                    {"name": "Temp", "levels": ["200", "225", "250", "275"]}],
        "responses": [{"name": "Tensile"}],
        "options": {"replicates": 3, "block_per_replicate": True}})
    ds = res.dataset
    temps = [200.0, 225.0, 250.0, 275.0]
    ds.df["Tensile"] = [PULP[temps.index(t), (int(b) - 1) * 3 + int(m) - 1]
                        for b, m, t in zip(ds.df["Block"], ds.df["Pulp"], ds.df["Temp"])]
    return res, ds


def battery_design():
    res = designs.build_design({
        "design": "threelevel", "seed": 9, "options": {"replicates": 4},
        "factors": [{"name": "Temperature", "low": 15, "high": 125},
                    {"name": "Material", "type": "categoric", "levels": ["T1", "T2", "T3"]}],
        "responses": [{"name": "Life"}]})
    ds = res.dataset
    pools = {k: list(v) for k, v in BATTERY.items()}
    life = np.empty(ds.n_cases)
    for i in np.argsort(ds.df["Std"].to_numpy()):
        life[i] = pools[(int(ds.df.at[i, "Material"]),
                         int(round(ds.df.at[i, "Temperature"])))].pop(0)
    ds.df["Life"] = life
    return res, ds


# ---------------------------------------------------------------------------- split-plot REML
def test_split_plot_reml_reproduces_classical_split_plot_anova():
    res, ds = pulp_design()
    sp = model.space_from_dataset(ds)
    assert sp.split_plot and sp.group_col == "Group" and sp.block_col == "Block"
    a = model.analyze(ds, model.AnalysisSpec("Tensile", model="2fi"))
    fit = a.fit
    assert isinstance(fit, model.SplitPlotFit) and fit.n_groups == 9
    tests = {src: (F, l, m, p) for _t, src, F, l, m, p in fit.term_tests()}
    # Montgomery's sums of squares: pulp 128.39 (2 df), whole-plot error (rep x pulp) 36.28
    # (4 df), temperature 434.08 (3), interaction 75.17 (6); subplot error 20.67 + 50.83 on
    # 6 + 12 df. Kenward-Roger in this balanced design gives the exact split-plot F tests.
    ms_wp, ms_sp = 36.28 / 4, (20.67 + 50.83) / 18
    F, l, m, p = tests["A-Pulp"]
    assert (l, m) == pytest.approx((2, 4), abs=1e-6)
    assert F == pytest.approx(128.39 / 2 / ms_wp, rel=2e-3)          # 7.08
    assert p == pytest.approx(stats.f.sf(F, 2, 4), rel=1e-9)
    F, l, m, p = tests["B-Temp"]
    assert (l, m) == pytest.approx((3, 18), abs=1e-6)
    assert F == pytest.approx(434.08 / 3 / ms_sp, rel=2e-3)          # 36.43
    F, l, m, p = tests["AB"]
    assert (l, m) == pytest.approx((6, 18), abs=1e-6)
    assert F == pytest.approx(75.17 / 6 / ms_sp, rel=2e-3)           # 3.15
    assert fit.s2 == pytest.approx(ms_sp, rel=2e-3)
    assert fit.s2_wp == pytest.approx((ms_wp - ms_sp) / 4, rel=5e-3)  # (MS_wp - MS_e) / 4
    # REML variance components agree with statsmodels MixedLM
    df = ds.df.copy()
    md = smf.mixedlm("Tensile ~ C(Block) + C(Pulp) * C(Temp)", df,
                     groups=df["Group"]).fit(reml=True)
    assert fit.s2 == pytest.approx(md.scale, rel=1e-4)
    assert fit.s2_wp == pytest.approx(float(md.cov_re.iloc[0, 0]), rel=1e-3)
    assert fit.beta[0] == pytest.approx(ds.df["Tensile"].mean())  # effect-coded intercept
    t, _ = report.anova_table(fit, "2FI")
    assert t.title.startswith("ANOVA (REML)") and t.rows[1][0] == "Whole-plot"
    vc = report.variance_components_table(fit)
    assert vc.rows[0][0] == "Group" and vc.rows[1][0] == "Residual"


def test_split_plot_fixed_effects_and_unbalanced_case():
    res, ds = pulp_design()
    fit = model.analyze(ds, model.AnalysisSpec("Tensile", model="2fi")).fit
    # GLS = OLS in this balanced orthogonal design; Kenward-Roger SEs differ by stratum
    ols = model.Fit(fit.space, fit.obs, fit.terms)
    assert np.allclose(fit.beta, ols.beta)
    coefs = {c["label"]: c for c in fit.coefficients()}
    assert coefs["A[1]"]["df"] == pytest.approx(4, abs=1e-6)
    assert coefs["B[1]"]["df"] == pytest.approx(18, abs=1e-6)
    # drop one run: still estimable, error df between the stratum limits
    ds.df.loc[3, "Tensile"] = np.nan
    fit2 = model.analyze(ds, model.AnalysisSpec("Tensile", model="2fi")).fit
    tests = {src: m for _t, src, _F, _l, m, _p in fit2.term_tests()}
    assert 3 < tests["A-Pulp"] < 6 and 15 < tests["B-Temp"] <= 18
    assert np.all(np.isfinite(fit2.diagnostics()["ext_stud"]))


# ---------------------------------------------------------------------------- categoric
def test_numeric_by_categoric_battery_life():
    res, ds = battery_design()
    assert res.meta["runs"] == 36
    sp = model.space_from_dataset(ds)
    assert [f.type for f in sp.factors] == ["numeric", "categoric"]
    spec = model.AnalysisSpec("Life", model="manual", terms=["0", "1", "0.1", "0.0", "0.0.1"])
    a = model.analyze(ds, spec)
    an = a.fit.anova()
    # Montgomery Example 5.1: treatments SS 59416.22 on 8 df, error 18230.75 on 27 df
    assert an["model"][0] == pytest.approx(59416.22, abs=0.01) and an["model"][1] == 8
    assert an["residual"] == pytest.approx((18230.75, 27), abs=0.01)
    # Montgomery Example 5.5: one quadratic in temperature per material
    eqs = dict(a.fit.actual_equations())
    expected = {"T1": (169.38, -2.50, 0.01286), "T2": (159.62, -0.17, -0.00567),
                "T3": (132.76, 0.90, -0.01019)}
    for lab, (b0, b1, b2) in expected.items():
        eq = eqs[f"Material = {lab}"]
        got = [c for c, _ in eq]
        for g, w, tol in zip(got, (b0, b1, b2), (0.01, 0.006, 6e-5)):
            assert g == pytest.approx(w, abs=tol)
    # optimisation enumerates the categoric levels: the longest life is material 2 at 15 °F
    fits = {"Life": a.fit}
    sol = optimize.solve(optimize.Problem(sp, fits, optimize.default_criteria(fits, sp)),
                         starts=8)
    assert sol[0]["x"][1] == 2 and sol[0]["x"][0] == pytest.approx(15, abs=1e-3)
    assert sol[0]["pred"]["Life"] == pytest.approx(155.75, abs=1e-6)


def test_two_level_categoric_coding_and_mixed_designs():
    res = designs.build_design({"design": "factorial", "randomize": False,
                                "factors": [{"name": "A", "low": 0, "high": 10},
                                            {"name": "B", "type": "categoric",
                                             "levels": ["Old", "New"]}]})
    ds = res.dataset
    sp = model.space_from_dataset(ds)
    coded = sp.to_coded(ds.df[["A", "B"]].to_numpy(float))
    cols, suffix = model.term_columns(coded, (1,), sp.categoric, sp.nlevels)
    # first level -1, second +1, one column without a [1] suffix
    assert suffix == [""] and list(cols[:, 0]) == [-1, -1, 1, 1]
    for design, extra, runs in (("ccd", [], 13 * 3), ("bbd", [{"name": "D"}], 17 * 2),
                                ("dopt", [], None)):
        nums = [{"name": "X1", "low": 0, "high": 1}, {"name": "X2", "low": 0, "high": 1}]
        cat = {"name": "C", "type": "categoric",
               "levels": ["p", "q", "r"] if design != "bbd" else ["p", "q"]}
        spec = {"design": design, "factors": nums + extra + [cat], "seed": 1}
        r = designs.build_design(spec)
        if runs:
            assert r.meta["runs"] == runs
        assert designs.design_run_count(spec) == r.meta["runs"]
        lv = r.dataset.df["C"].to_numpy()
        assert set(np.unique(lv)) == set(range(1, len(cat["levels"]) + 1))
    with pytest.raises(ValueError, match="two-level categoric"):
        designs.build_design({"design": "factorial", "factors": [
            {"name": "A"}, {"name": "B", "type": "categoric", "levels": ["x", "y", "z"]}]})


# ---------------------------------------------------------------------------- min-run designs
@pytest.mark.parametrize("k,runs", [(6, 22), (7, 30), (8, 38), (9, 46)])
def test_min_run_resolution_v(k, runs):
    # Stat-Ease: runs = number of 2FI-model terms, raised to an even number (7 -> 30)
    assert designs.mr5_runs(k) == runs
    D = designs.min_run_res5(k, np.random.default_rng(2), starts=2)
    assert D.shape == (runs, k)
    assert np.all(D.sum(axis=0) == 0)  # equireplicated: as many highs as lows
    F = designs.two_fi_matrix(D)
    assert np.linalg.matrix_rank(F) == 1 + k + k * (k - 1) // 2
    again = designs.min_run_res5(k, np.random.default_rng(2), starts=2)
    assert np.array_equal(D, again)  # deterministic for a seed


@pytest.mark.parametrize("k", [5, 6, 7, 9, 12])
def test_min_run_resolution_iv(k):
    for extra in (False, True):
        D = designs.min_run_res4(k, np.random.default_rng(4), extra)
        assert D.shape == (designs.mr4_runs(k, extra), k) == (2 * k + 2 * extra, k)
        assert np.all(D.sum(axis=0) == 0)
        two_fi = np.column_stack([D[:, i] * D[:, j]
                                  for i, j in itertools.combinations(range(k), 2)])
        assert np.abs(D.T @ two_fi).max() == 0  # main effects clear of all 2FIs
        assert np.linalg.matrix_rank(np.column_stack([np.ones(len(D)), D])) == k + 1
    res = designs.build_design({"design": "mr4", "factors": [{"name": c} for c in "ABCDEF"],
                                "seed": 3})
    assert res.meta["runs"] == 14 and res.meta["resolution"] == 4


def test_foldover_restores_resolution_iv():
    gens = designs.default_generators(7, 4)
    coded, group = designs.fractional_factorial(7, gens)
    assert designs.resolution(group) == 3
    full = np.vstack([coded, designs.foldover_runs(coded, "full")])
    two_fi = np.column_stack([full[:, i] * full[:, j]
                              for i, j in itertools.combinations(range(7), 2)])
    assert np.abs(full.T @ two_fi).max() == 0
    assert designs.resolution(designs.folded_group(group, "full")) == 4
    single = designs.foldover_runs(coded, "single", factor=1)
    assert np.array_equal(single[:, 1], -coded[:, 1])
    new_group = designs.folded_group(group, "single", 1)
    assert all(not (w >> 1 & 1) for w, _ in new_group)
    semi = designs.foldover_runs(coded, "semifold", factor=1, keep_factor=3, keep_level=1)
    assert len(semi) == 4 and np.all(semi[:, 3] == 1)


# ---------------------------------------------------------------------------- split-plot designs
def test_split_plot_designs_restrict_randomisation():
    cases = [
        ({"design": "factorial", "factors": [{"name": "a", "changes": "hard"}] +
          [{"name": n} for n in "BCD"], "options": {"groups": 4}}, 16, 4),
        ({"design": "ccd", "factors": [{"name": "a", "changes": "hard"},
                                       {"name": "b", "changes": "hard"},
                                       {"name": "C"}, {"name": "D"}],
          "options": {"alpha_type": "practical", "center_points": 1}}, 37, 9),
        ({"design": "dopt", "factors": [{"name": "a", "changes": "hard"}, {"name": "B"},
                                        {"name": "C"}],
          "options": {"groups": 6, "group_size": 4}}, 24, 6),
    ]
    for spec, runs, groups in cases:
        spec["seed"] = 8
        res = designs.build_design(spec)
        df = res.dataset.df
        assert res.meta["runs"] == runs and res.meta["split_plot"]["groups"] == groups
        for j in res.meta["split_plot"]["hard"]:
            assert df.groupby("Group")[res.meta["factors"][j]["column"]].nunique().max() == 1
        g = df.sort_values("Run")["Group"].to_numpy()
        assert np.sum(g[1:] != g[:-1]) == groups - 1  # each group's runs are consecutive
    # the 2^4 split-plot: the whole plots confound A and the BCD interaction
    res = designs.build_design(cases[0][0])
    assert res.meta["whole_plot_words"] == ["BCD"]
    with pytest.raises(ValueError, match="easy-to-change"):
        designs.build_design({"design": "factorial",
                              "factors": [{"name": "a", "changes": "hard"}]})
    with pytest.raises(ValueError, match="Hard-to-change"):
        designs.build_design({"design": "bbd", "factors": [{"name": "a", "changes": "hard"},
                                                           {"name": "B"}, {"name": "C"}]})


# ---------------------------------------------------------------------------- selection
def montgomery_ccd():
    res = designs.build_design({"design": "ccd", "randomize": False,
                                "factors": [{"name": "Time", "low": 80, "high": 90},
                                            {"name": "Temp", "low": 170, "high": 180}],
                                "options": {"alpha_type": "custom", "alpha": 1.414,
                                            "center_points": 5}, "responses": [{"name": "Y"}]})
    ds = res.dataset
    a = 1.414
    pts = {(-1, -1): 76.5, (-1, 1): 77.0, (1, -1): 78.0, (1, 1): 79.5, (a, 0): 78.4,
           (-a, 0): 75.6, (0, a): 78.5, (0, -a): 77.0}
    centre = iter([79.9, 80.3, 80.0, 79.7, 79.8])
    y = []
    for t, u in zip(ds.df["Time"], ds.df["Temp"]):
        key = (round((t - 85) / 5, 3), round((u - 175) / 5, 3))
        y.append(next(centre) if key == (0, 0) else pts[key])
    ds.df["Y"] = y
    return ds


def test_auto_select_directions_and_criteria():
    ds = montgomery_ccd()
    sp = model.space_from_dataset(ds)
    obs = model.observations(ds, sp, "Y")
    quad = model.model_terms(sp, "quadratic")
    tr = model.Transform()
    # backward by p (alpha out 0.10): only AB (p = 0.1025) goes, as before
    fit, steps = model.auto_select(sp, obs, quad, tr, "backward", "p", alpha_out=0.10)
    assert steps == [("remove", (0, 1), pytest.approx(0.1025, abs=1e-3))]
    old_fit, old_steps = model.backward(sp, obs, quad, tr, 0.10)
    assert old_fit.terms == fit.terms and old_steps[0][0] == (0, 1)
    want = {(0,), (1,), (0, 0), (1, 1)}
    for method, crit in (("forward", "p"), ("stepwise", "p"), ("forward", "aicc"),
                         ("backward", "aicc"), ("forward", "bic")):
        f, st = model.auto_select(sp, obs, quad, tr, method, crit, 0.10, 0.10)
        if crit == "bic":  # the lighter BIC penalty at n = 13 also keeps AB
            assert want <= set(f.terms) <= set(quad)
        else:
            assert set(f.terms) == want, (method, crit)
    # the information criteria match statsmodels' maximum likelihood
    info = fit.information()
    ols = sm.OLS(obs.y, fit.mat.X).fit()
    n, k = fit.n, fit.p + 1
    assert info["m2ll"] == pytest.approx(-2 * ols.llf, rel=1e-10)
    assert info["aicc"] == pytest.approx(-2 * ols.llf + 2 * k + 2 * k * (k + 1) / (n - k - 1))
    assert info["bic"] == pytest.approx(-2 * ols.llf + k * math.log(n))
    # the analysis spec stores the settings (old files without them still load)
    spec = model.AnalysisSpec("Y", model="quadratic", reduction="forward", criterion="aicc")
    a = model.analyze(ds, spec)
    assert set(a.fit.terms) == want and a.steps[0][0] == "add"
    back = model.AnalysisSpec.from_meta("Y", spec.to_meta())
    assert (back.reduction, back.criterion, back.alpha_in) == ("forward", "aicc", 0.10)
    old = model.AnalysisSpec.from_meta("Y", {"model": "quadratic", "reduction": "backward",
                                             "alpha_out": 0.05})
    assert old.criterion == "p" and old.alpha_out == 0.05
    sec = report.anova_section(a)
    assert any(t.title.startswith("Forward Selection") for t in sec.items
               if hasattr(t, "rows"))


def test_forward_selection_adds_hierarchy_parents():
    ds = montgomery_ccd()
    sp = model.space_from_dataset(ds)
    obs = model.observations(ds, sp, "Y")
    obs.y = obs.y + 3.0 * obs.coded[:, 0] * obs.coded[:, 1]  # a strong AB, weak mains
    terms = model.model_terms(sp, "2fi")
    f, st = model.auto_select(sp, obs, terms, model.Transform(), "forward", "p", 0.05)
    assert (0, 1) in f.terms and (0,) in f.terms and (1,) in f.terms


def test_dfbetas_match_statsmodels():
    ds = montgomery_ccd()
    a = model.analyze(ds, model.AnalysisSpec("Y", model="quadratic"))
    d = a.fit.diagnostics()
    infl = OLSInfluence(sm.OLS(a.obs.y, a.fit.mat.X).fit())
    assert np.allclose(d["dfbetas"], infl.dfbetas, atol=1e-9)
    assert np.allclose(d["dffits"], infl.dffits[0], atol=1e-9)
    sec = report.diagnostics_section(a, plots_wanted=("dfbetas",), table=False, term_index=1)
    assert next(i for i in sec.items if hasattr(i, "png")).png[:4] == b"\x89PNG"


def test_figures_for_live_model_graphs():
    ds = montgomery_ccd()
    a = model.analyze(ds, model.AnalysisSpec("Y", model="quadratic"))
    ref = report.default_point(a.space)
    gf = report.graph_figure(a, "surface", [0, 1], ref)
    assert gf.rotatable and gf.figure.axes[0].name == "3d"
    gf = report.graph_figure(a, "contour", [0, 1], ref)
    lines = gf.probe(86.9, 176.5)
    pred = a.fit.predict(np.array([[86.9, 176.5]]))[0]
    assert any(report.num(pred) in s for s in lines)
    assert gf.probe(70.0, 176.0) is None  # outside the plot
    png = report.plots.figure_png(gf.figure)
    assert png[:4] == b"\x89PNG"
    sec = report.graph_section(a, "surface", [0, 1], ref)
    assert next(i for i in sec.items if hasattr(i, "png")).png[:4] == b"\x89PNG"
