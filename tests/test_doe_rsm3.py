"""RSM / DOE additions after 0.7.0: fast minimum-run designs up to 50 factors, split-plot CCD
and D-optimal split-plot designs with categoric factors, split-plot factorials blocked by
confounding, whole-plot / subplot effects and profile-likelihood variance intervals.

Checked against the known maximal determinants of +-1 matrices (OEIS A003432), Stat-Ease's
split-plot tutorials (PCR: 2^(9-1), 3 hard-to-change factors, 8 groups of 32 runs; airfoil:
37 runs in 9 groups), Montgomery's paper-pulp split-plot (Table 14.16) and an independent
dense implementation of the restricted likelihood."""
import itertools
import math
import os
import sys
import time

import numpy as np
import pytest
from scipy import optimize as sopt
from scipy import stats

from rsstatchem.core.doe import designs, model, optimize, report

sys.path.insert(0, os.path.dirname(__file__))
from test_doe_extended import pulp_design  # noqa: E402

HEAVY = pytest.mark.skipif(not os.environ.get("RSSTATCHEM_HEAVY_TESTS"),
                           reason="set RSSTATCHEM_HEAVY_TESTS=1 for the 50-factor timing")


def d_eff(D: np.ndarray) -> float:
    F = designs.two_fi_matrix(D)
    sign, ld = np.linalg.slogdet(F.T @ F)
    return math.exp(ld / F.shape[1]) / len(D) if sign > 0 else 0.0


def check_mr5(D: np.ndarray, k: int) -> None:
    n = designs.mr5_runs(k)
    assert D.shape == (n, k)
    assert set(np.unique(D)) == {-1.0, 1.0}
    assert np.all(D.sum(axis=0) == 0)  # equireplicated
    F = designs.two_fi_matrix(D)
    assert np.linalg.matrix_rank(F) == 1 + k + k * (k - 1) // 2  # resolution V: all 2FIs


# ---------------------------------------------------------------------------- min-run designs
@pytest.mark.parametrize("k,floor", [(6, 0.72), (7, 0.72), (10, 0.68), (15, 0.62),
                                     (20, 0.6)])
def test_mr5_properties_and_efficiency(k, floor):
    D = designs.min_run_res5(k, np.random.default_rng(5))
    check_mr5(D, k)
    assert d_eff(D) >= floor
    # far better than a random balanced design of the same size
    rng = np.random.default_rng(0)
    n = designs.mr5_runs(k)
    rand = np.column_stack([rng.permutation(np.repeat([1.0, -1.0], n // 2)) for _ in range(k)])
    assert d_eff(D) > d_eff(rand) + 0.1
    assert np.array_equal(D, designs.min_run_res5(k, np.random.default_rng(5)))


def test_mr5_forty_factors_quickly():
    t = time.perf_counter()
    D = designs.min_run_res5(40, np.random.default_rng(1))
    elapsed = time.perf_counter() - t
    check_mr5(D, 40)
    assert D.shape == (822, 40) and d_eff(D) > 0.53   # 0.532 before the faster exchange
    assert elapsed < 30  # ~0.3 s on a laptop; loose for busy CI machines


def test_mr5_fifty_factors_efficient_and_fast():
    """50 factors (1276 runs): the orthogonal-array start and the pending low-rank updates
    give a higher D-efficiency than the former random start (0.51) in a fraction of the
    time (~0.5 s idle instead of ~2-3 s); the bound is loose for loaded machines."""
    t = time.perf_counter()
    D = designs.min_run_res5(50, np.random.default_rng(1))
    elapsed = time.perf_counter() - t
    check_mr5(D, 50)
    assert d_eff(D) >= 0.515
    assert elapsed < 20


@HEAVY
def test_mr5_fifty_factors_under_a_second():
    spec = {"design": "mr5", "seed": 1, "factors": [{"name": f"X{i}"} for i in range(50)]}
    times = []
    for _ in range(3):   # the best of three is robust against a busy machine
        t = time.perf_counter()
        res = designs.build_design(spec)
        times.append(time.perf_counter() - t)
    assert res.meta["runs"] == 1276 and res.meta["resolution"] == 5
    D = res.dataset.df[[f["column"] for f in res.meta["factors"]]].to_numpy(float)
    check_mr5(D, 50)
    assert d_eff(D) >= 0.515
    assert min(times) < 1.0  # ~0.5 s on an idle laptop (was ~2-3 s)


def test_mr5_orthogonal_array_start():
    """The starting design: runs of a regular resolution V fraction (no defining word shorter
    than five, so its 2FI model matrix is orthogonal), balanced, and keeping the extra
    three-factor column unaliased."""
    for k, n, runs in ((6, 22, 32), (17, 154, 256), (21, 232, 512), (29, 436, 1024),
                       (38, 742, 2048), (50, 1276, 4096)):
        assert designs.mr5_runs(k) == n
        bits, cols = designs._res5_generators(k, n)
        assert 1 << bits == runs and len(set(cols)) == k and 0 not in cols
        # no 2, 3 or 4 columns sum to zero: the sums of pairs are distinct and no column
        pairs = [a ^ b for a, b in itertools.combinations(cols, 2)]
        assert len(set(pairs)) == len(pairs) and not set(pairs) & set(cols)
        u = np.arange(runs)[:, None]
        X = 1.0 - 2.0 * designs._parity(u & np.array(cols)[None, :])
        F = designs.two_fi_matrix(X)
        assert np.array_equal(F.T @ F, runs * np.eye(F.shape[1]))
    rng = np.random.default_rng(3)
    for k in (7, 12, 31):   # run counts with the extra run (and its 3FI column)
        n = designs.mr5_runs(k)
        assert n == 1 + k + k * (k - 1) // 2 + 1
        D = designs._mr5_start(k, n, True, rng)
        assert D.shape == (n, k) and np.all(D.sum(axis=0) == 0)
        assert set(np.unique(D)) == {-1.0, 1.0}
    # far better than random balanced runs for 50 factors (~0.43 against ~0.36)
    D = designs._mr5_start(50, 1276, False, np.random.default_rng(0))
    rand = np.column_stack([rng.permutation(np.repeat([1.0, -1.0], 638)) for _ in range(50)])
    assert d_eff(D) > 0.41 > 0.38 > d_eff(rand)


def test_mr5_limits_and_run_counts():
    from rsstatchem.ui.doe.wizard import LIMITS

    assert LIMITS["mr5"] == (6, 50) and LIMITS["mr4"] == (5, 50)
    # 1 + k + k(k-1)/2 terms raised to an even number (Stat-Ease: 9 factors -> 46 runs)
    assert [designs.mr5_runs(k) for k in (9, 30, 50)] == [46, 466, 1276]
    with pytest.raises(ValueError, match="up to 50"):
        designs.min_run_res5(51, np.random.default_rng(0))
    with pytest.raises(ValueError, match="up to 50"):
        designs.min_run_res4(51, np.random.default_rng(0))


# OEIS A003432: maximal determinant of an n x n (+1, -1) matrix
MAX_DET = {3: 4, 4: 16, 5: 48, 6: 160, 7: 576, 8: 4096, 9: 14336, 10: 73728, 11: 327680,
           12: 2985984}


@pytest.mark.parametrize("k", sorted(MAX_DET))
def test_mr4_base_reaches_the_maximal_determinant(k):
    X = designs.max_det_pm1(k, np.random.default_rng(1))
    assert round(abs(np.linalg.det(X))) == MAX_DET[k]


@pytest.mark.parametrize("k", [25, 37, 50])
def test_mr4_up_to_fifty_factors(k):
    t = time.perf_counter()
    for extra in (False, True):
        D = designs.min_run_res4(k, np.random.default_rng(2), extra)
        assert D.shape == (2 * k + 2 * extra, k) and np.all(D.sum(axis=0) == 0)
        two_fi = np.column_stack([D[:, i] * D[:, j]
                                  for i, j in itertools.combinations(range(k), 2)])
        assert np.abs(D.T @ two_fi).max() == 0  # main effects clear of every 2FI
        assert np.linalg.matrix_rank(np.column_stack([np.ones(len(D)), D])) == k + 1
    assert time.perf_counter() - t < 10


# ---------------------------------------------------------------------------- split-plot CCD
def lift_like(ds, cols, rng, wp_sd=2.0, sd=0.5):
    g = ds.df["Group"].to_numpy(int)
    x1 = ds.df[cols[0]].to_numpy(float)
    x2 = ds.df[cols[1]].to_numpy(float)
    cat = ds.df[cols[2]].to_numpy(int)
    return (100 + 6 * x1 + 4 * x2 - 3 * x1 ** 2 - 2 * x2 ** 2 + 1.5 * x1 * x2
            + np.where(cat == 2, 5.0, 0.0) + rng.normal(0, wp_sd, g.max() + 1)[g - 1]
            + rng.normal(0, sd, len(g)))


@pytest.mark.parametrize("cat_changes,runs,groups", [("easy", 22, 5), ("hard", 22, 10)])
def test_split_plot_ccd_with_categoric_factor(cat_changes, runs, groups):
    spec = {"design": "ccd", "seed": 3, "options": {"alpha_type": "face", "center_points": 1,
                                                    "replicates": 2},
            "factors": [{"name": "a", "low": -1, "high": 1, "changes": "hard"},
                        {"name": "B", "low": -1, "high": 1},
                        {"name": "C", "type": "categoric", "levels": ["p", "q"],
                         "changes": cat_changes}],
            "responses": [{"name": "Y"}]}
    res = designs.build_design(spec)
    ds = res.dataset
    df = ds.df
    cols = [f["column"] for f in res.meta["factors"]]
    assert res.meta["runs"] == 2 * runs and res.meta["split_plot"]["groups"] == 2 * groups
    assert designs.design_run_count(spec) == res.meta["runs"]
    for j in res.meta["split_plot"]["hard"]:
        assert df.groupby("Group")[cols[j]].nunique().max() == 1
    if cat_changes == "easy":  # every run of a group at both levels of C
        per = df.groupby(["Group", cols[2]]).size().unstack()
        assert (per[1] == per[2]).all()
    g = df.sort_values("Run")["Group"].to_numpy()
    assert np.sum(g[1:] != g[:-1]) == 2 * groups - 1  # restricted randomisation
    df["Y"] = lift_like(ds, cols, np.random.default_rng(1))
    a = model.analyze(ds, model.AnalysisSpec("Y", model="quadratic"))
    fit = a.fit
    assert isinstance(fit, model.SplitPlotFit)
    whole = {r[1] for r in fit.anova()["whole"]}
    assert "A-a" in whole and ("C-C" in whole) == (cat_changes == "hard")
    coef = {c["label"]: c["estimate"] for c in fit.coefficients()}
    assert coef["A"] == pytest.approx(6, abs=1.5) and coef["B"] == pytest.approx(4, abs=0.5)
    eqs = dict(fit.actual_equations())  # one equation per level of C
    assert set(eqs) == {"C = p", "C = q"}
    assert eqs["C = q"][0][0] - eqs["C = p"][0][0] == pytest.approx(5, abs=2.5)
    fits = {"Y": fit}
    sol = optimize.solve(optimize.Problem(a.space, fits,
                                          optimize.default_criteria(fits, a.space)), starts=6)
    assert sol[0]["x"][2] == 2  # the better categoric level


def test_split_plot_ccd_categoric_layouts():
    # only a categoric factor is hard to change: the whole CCD runs in each of its levels
    res = designs.build_design({"design": "ccd", "seed": 1, "options": {"center_points": 3},
                                "factors": [{"name": "A"}, {"name": "B"},
                                            {"name": "C", "type": "categoric",
                                             "levels": ["x", "y", "z"], "changes": "hard"}]})
    assert res.meta["runs"] == 3 * 11 and res.meta["split_plot"]["groups"] == 3
    assert any("replicate" in n for n in res.notes)  # no whole-plot error df otherwise
    # all numeric factors hard: every CCD point is its own whole plot, C varies within
    res = designs.build_design({"design": "ccd", "seed": 1, "options": {"center_points": 2},
                                "factors": [{"name": "a", "changes": "hard"},
                                            {"name": "b", "changes": "hard"},
                                            {"name": "C", "type": "categoric",
                                             "levels": ["x", "y"]}]})
    assert res.meta["runs"] == 2 * 10 and res.meta["split_plot"]["groups"] == 10
    with pytest.raises(ValueError, match="2 numeric"):
        designs.build_design({"design": "ccd", "factors": [
            {"name": "a", "changes": "hard"},
            {"name": "C", "type": "categoric", "levels": ["x", "y"]}]})


def test_d_optimal_split_plot_with_hard_and_easy_categoric_factors():
    spec = {"design": "dopt", "seed": 3,
            "factors": [{"name": "Oven", "type": "categoric", "levels": ["X", "Y", "Z"],
                         "changes": "hard"},
                        {"name": "Temp", "low": 100, "high": 200, "changes": "hard"},
                        {"name": "Time", "low": 1, "high": 5},
                        {"name": "Coat", "type": "categoric", "levels": ["thin", "thick"]}],
            "options": {"groups": 12, "group_size": 4, "model": "quadratic"}}
    res = designs.build_design(spec)
    ds = res.dataset
    df = ds.df
    assert res.meta["runs"] == 48 and res.meta["split_plot"]["groups"] == 12
    for c in ("Oven", "Temp"):
        assert df.groupby("Group")[c].nunique().max() == 1
    assert df.groupby("Group")["Coat"].nunique().max() == 2  # easy categoric varies inside
    assert set(df["Oven"]) == {1.0, 2.0, 3.0}
    rng = np.random.default_rng(0)
    g = df["Group"].to_numpy(int)
    x = (df["Temp"] - 150) / 50
    t = (df["Time"] - 3) / 2
    df["R1"] = (10 + 2 * x - 1.5 * t + x * t + np.array([0, 1.5, -1])[df["Oven"].astype(int) - 1]
                + 0.8 * (df["Coat"] == 2) + rng.normal(0, 1, g.max() + 1)[g - 1]
                + rng.normal(0, 0.5, len(g)))
    a = model.analyze(ds, model.AnalysisSpec("R1", model="quadratic"))
    tests = {r[1]: r for r in a.fit.term_tests()}
    # whole-plot terms get whole-plot error df, subplot terms the within-group df
    assert tests["A-Oven"][4] < 8 and tests["B-Temp"][4] < 8
    assert tests["C-Time"][4] > 20 and tests["D-Coat"][4] > 20
    assert tests["C-Time"][5] < 1e-6 and tests["BC"][5] < 1e-6
    assert len(a.fit.actual_equations()) == 6  # Oven x Coat level combinations


# ---------------------------------------------------------------------------- blocking
def test_split_plot_factorial_blocked_by_confounding():
    spec = {"design": "factorial", "seed": 4, "options": {"groups": 4, "blocks": 2},
            "factors": [{"name": "a", "changes": "hard"}] + [{"name": c} for c in "BCD"]}
    res = designs.build_design(spec)
    df = res.dataset.df
    # 2^4 split-plot, whole plots = a x BCD; the blocks confound a.BCD = ABCD
    assert res.meta["whole_plot_words"] == ["BCD"] and res.meta["block_generators"] == ["ABCD"]
    assert res.meta["blocks"] == 2
    assert df.groupby("Group")["Block"].nunique().max() == 1      # whole plots stay intact
    assert df.groupby("Block")["Group"].nunique().tolist() == [2, 2]
    coded = np.sign(df[["a", "B", "C", "D"]].to_numpy(float) - 0.0)
    blk = np.where(df["Block"] == 1, -1.0, 1.0)
    for r in (1, 2, 3):  # block contrast orthogonal to all main effects, 2FIs and 3FIs
        for t in itertools.combinations(range(4), r):
            assert abs(blk @ np.prod(coded[:, list(t)], axis=1)) < 1e-9
    assert np.allclose(blk, np.prod(coded, axis=1)) or np.allclose(blk, -np.prod(coded, axis=1))
    # randomisation: blocks first, then groups within blocks, then runs
    order = df.sort_values("Run")
    assert np.all(np.diff(order["Block"].to_numpy()) >= 0)
    g = order["Group"].to_numpy()
    assert np.sum(g[1:] != g[:-1]) == 3
    # replicated with one block per replicate: the confounded blocks are kept inside
    res2 = designs.build_design({**spec, "options": {"groups": 4, "blocks": 2,
                                                     "replicates": 2,
                                                     "block_per_replicate": True}})
    assert res2.meta["blocks"] == 4 and res2.meta["runs"] == 32
    with pytest.raises(ValueError, match="whole plots"):
        designs.build_design({**spec, "options": {"groups": 2, "blocks": 4}})


def test_pcr_tutorial_layout_with_blocks():
    # Stat-Ease split-plot tutorial: 9 factors, a b c hard to change, 2^(9-1), 8 groups
    spec = {"design": "factorial", "seed": 1,
            "options": {"fraction": 1, "groups": 8, "blocks": 2},
            "factors": [{"name": n, "changes": "hard"} for n in "abc"]
            + [{"name": n} for n in "DEFGHJ"], "responses": [{"name": "Amp"}]}
    res = designs.build_design(spec)
    df = res.dataset.df
    assert res.meta["runs"] == 256 and res.meta["resolution"] == 9
    assert res.meta["split_plot"]["groups"] == 8
    assert df.groupby("Group").size().tolist() == [32] * 8
    assert res.meta["block_generators"] == ["ABC"]  # the whole-plot 3FI
    sp = model.space_from_dataset(res.dataset)
    rng = np.random.default_rng(7)
    g = df["Group"].to_numpy(int)
    s = {c: np.sign(df[c].to_numpy(float)) for c in "abcDEFGHJ"}
    df["Amp"] = (40 + 4 * s["a"] + 3 * s["D"] + 2 * s["D"] * s["E"]
                 + rng.normal(0, 1.5, 8)[g - 1] + rng.normal(0, 0.5, len(g)))
    obs = model.observations(res.dataset, sp, "Amp")
    eff = model.split_plot_effects(sp, obs)
    # 8 whole plots in 2 blocks: 6 whole-plot contrasts left (a, b, c, ab, ac, bc)
    assert sorted(e["name"] for e in eff["whole"]) == ["A", "AB", "AC", "B", "BC", "C"]
    big = max(eff["sub"], key=lambda e: abs(e["effect"]))
    assert big["name"] == "D" and big["effect"] == pytest.approx(6, abs=0.3)


# ---------------------------------------------------------------------------- effects
def test_whole_plot_and_subplot_effects():
    res = designs.build_design({"design": "factorial", "seed": 4,
                                "options": {"groups": 4, "replicates": 2},
                                "factors": [{"name": "a", "changes": "hard"}]
                                + [{"name": c} for c in "BCD"],
                                "responses": [{"name": "R1"}]})
    ds = res.dataset
    df = ds.df
    rng = np.random.default_rng(1)
    g = df["Group"].to_numpy(int)
    A, B, C = (np.sign(df[c].to_numpy(float)) for c in ("a", "B", "C"))
    df["R1"] = (50 + 5 * A + 4 * B + 3 * B * C + rng.normal(0, 3, g.max() + 1)[g - 1]
                + rng.normal(0, 0.5, len(g)))
    a = model.analyze(ds, model.AnalysisSpec("R1"))
    eff = model.split_plot_effects(a.space, a.obs)
    # whole plots = a x BCD: the whole-plot contrasts are A, BCD and ABCD
    assert sorted(e["name"] for e in eff["whole"]) == ["A", "ABCD", "BCD"]
    assert len(eff["sub"]) == 15 - 3
    assert all(not e["whole_plot"] for e in eff["sub"])
    # effects = 2 x least-squares coefficients (orthogonal design)
    ols = model.Fit(a.space, a.obs, [(0,), (1,), (2,), (1, 2)])
    b = dict(zip(ols.mat.labels, ols.beta))
    e = {x["name"]: x["effect"] for x in eff["whole"] + eff["sub"]}
    assert e["A"] == pytest.approx(2 * b["A"]) and e["BC"] == pytest.approx(2 * b["BC"])
    assert e["B"] == pytest.approx(8, abs=0.8) and e["BC"] == pytest.approx(6, abs=0.8)
    # separate Lenth PSEs: the whole-plot effects carry the larger between-group noise
    assert eff["whole"][0]["pse"] != eff["sub"][0]["pse"]
    sec = report.effects_section(a)
    titles = [it.title for it in sec.items]
    assert "Subplot Effects" in titles and "Whole-plot Effects" in titles
    assert "Half-Normal Plot (Whole-plot Effects)" in titles
    fig = __import__("rsstatchem.core.doe.plots", fromlist=["x"]).half_normal_figure(
        eff["sub"], "t", {"B", "BC"})
    assert {n for _x, _y, n in fig.effect_points} == {x["name"] for x in eff["sub"]}


# ---------------------------------------------------------------------------- variance CI
def dense_reml(fit, s2, s2_wp):
    """-0.5 [log|V| + log|X'V^-1 X| + r'V^-1 r] by dense algebra (independent check)."""
    X, z, Z = fit.mat.X, fit.z, fit.Z
    V = s2 * np.eye(fit.n) + s2_wp * Z @ Z.T
    Vi = np.linalg.inv(V)
    A = X.T @ Vi @ X
    beta = np.linalg.solve(A, X.T @ Vi @ z)
    r = z - X @ beta
    return -0.5 * (np.linalg.slogdet(V)[1] + np.linalg.slogdet(A)[1] + r @ Vi @ r)


def dense_profile(fit, which, v):
    if which == "Group":
        res = sopt.minimize_scalar(lambda u: -dense_reml(fit, math.exp(u), v),
                                   bounds=(math.log(fit.s2) - 10, math.log(fit.s2) + 10),
                                   method="bounded", options={"xatol": 1e-10})
        return -res.fun
    res = sopt.minimize_scalar(lambda u: -dense_reml(fit, v, math.exp(u)),
                               bounds=(-30, math.log(fit.s2 + fit.s2_wp) + 10),
                               method="bounded", options={"xatol": 1e-10})
    return max(-res.fun, dense_reml(fit, v, 0.0))


def test_profile_likelihood_variance_intervals():
    _res, ds = pulp_design()
    fit = model.analyze(ds, model.AnalysisSpec("Tensile", model="2fi")).fit
    for s2, s2wp in ((3.0, 1.0), (4.5, 0.0), (2.0, 7.5)):
        assert fit.reml_loglik(s2, s2wp) == pytest.approx(dense_reml(fit, s2, s2wp),
                                                          rel=1e-10)
    vc = {v["source"]: v for v in fit.variance_components()}
    l_max = dense_reml(fit, fit.s2, fit.s2_wp)
    crit = stats.chi2.ppf(0.95, 1)
    # the interval ends are where the likelihood-ratio statistic reaches 3.84
    for which in ("Group", "Residual"):
        hi = vc[which]["high"]
        assert 2 * (l_max - dense_profile(fit, which, hi)) == pytest.approx(crit, abs=1e-4)
    lo = vc["Residual"]["low"]
    assert 2 * (l_max - dense_profile(fit, "Residual", lo)) == pytest.approx(crit, abs=1e-4)
    # zero group variance is inside the region here: the lower limit is 0, never negative
    assert vc["Group"]["low"] == 0.0
    assert 2 * (l_max - dense_profile(fit, "Group", 0.0)) < crit
    # balanced split-plot: the residual interval is close to the exact chi-square one
    exact = (fit.s2 * 18 / stats.chi2.ppf(0.975, 18), fit.s2 * 18 / stats.chi2.ppf(0.025, 18))
    assert vc["Residual"]["low"] == pytest.approx(exact[0], rel=0.08)
    assert vc["Residual"]["high"] == pytest.approx(exact[1], rel=0.08)
    assert vc["Group"]["low"] <= vc["Group"]["variance"] <= vc["Group"]["high"]
    # a strong whole-plot variance: the lower limit moves above 0
    df = ds.df
    g = df["Group"].to_numpy(int)
    df["Tensile"] = df["Tensile"] + np.random.default_rng(2).normal(0, 6, g.max() + 1)[g - 1]
    fit2 = model.analyze(ds, model.AnalysisSpec("Tensile", model="2fi")).fit
    v2 = {v["source"]: v for v in fit2.variance_components()}["Group"]
    assert 0 < v2["low"] < v2["variance"] < v2["high"]
    l2 = dense_reml(fit2, fit2.s2, fit2.s2_wp)
    assert 2 * (l2 - dense_profile(fit2, "Group", v2["low"])) == pytest.approx(crit, abs=1e-4)
    t = report.variance_components_table(fit2)
    assert any("profile-likelihood" in f for f in t.footnotes)
