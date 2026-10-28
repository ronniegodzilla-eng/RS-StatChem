"""WP-A of the Design-Expert gap analysis: the optimal (custom) engine (D/I/A, point /
coordinate / best exchange, blocks, forced runs, edited models, multiple linear constraints),
constrained regions (McLean-Anderson vertices), combined mixture-process designs, definitive
screening, small builders, augmentation, historical data, layout editing and the evaluation
extras (matrix measures, alias matrix, correlations, FDS score), plus the window wiring.

Expected values: docs/DESIGN_EXPERT_GAP_ANALYSIS.md section 4 (E7, E8, E9, E10)."""
from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd
import pytest

from rsstatchem.core.doe import constraints as cons
from rsstatchem.core.doe import designs, evaluation, model, optimal
from rsstatchem.core.doe import terms_combined as tc

FLARE_L = [0.40, 0.10, 0.10, 0.03]
FLARE_U = [0.60, 0.50, 0.50, 0.08]


def _actual(region, L, total=1.0):
    return np.asarray(L) + region.vertices() * (total - sum(L))


def _rows(a):
    return {tuple(np.round(r, 6)) for r in np.asarray(a)}


# ---------------------------------------------------------------------------- E7 regions
def test_e7_mclean_anderson_vertices_and_centroids():
    reg = cons.mixture_region(FLARE_L, FLARE_U, 1.0)
    V = _actual(reg, FLARE_L)
    expected = {(.4, .1, .47, .03), (.4, .1, .42, .08), (.6, .1, .27, .03),
                (.6, .1, .22, .08), (.4, .47, .1, .03), (.4, .42, .1, .08),
                (.6, .27, .1, .03), (.6, .22, .1, .08)}
    assert _rows(V) == expected
    assert np.allclose(V.mean(axis=0), [0.5, 0.2225, 0.2225, 0.055])
    pts, types = reg.candidates(("vertex", "plane", "centroid"))
    assert len(pts) == 15 and (types == 5).sum() == 6 and (types == 7).sum() == 1
    actual = np.asarray(FLARE_L) + pts * (1 - sum(FLARE_L))
    assert np.allclose(actual.sum(axis=1), 1)


def test_e7_with_a_multiple_linear_constraint():
    c = cons.parse_constraint("A + D <= 0.65", 4)
    assert c == {"coefs": [1.0, 0.0, 0.0, 1.0], "low": None, "high": 0.65,
                 "text": "A + D <= 0.65"}
    reg = cons.mixture_region(FLARE_L, FLARE_U, 1.0, [c])
    V = _actual(reg, FLARE_L)
    assert np.all(V[:, 0] + V[:, 3] <= 0.65 + 1e-9)
    assert np.allclose(V.sum(axis=1), 1)
    # brute force: every vertex of the new region is an intersection of 3 active planes
    assert {(0.57, 0.1, 0.25, 0.08), (0.6, 0.1, 0.25, 0.05)} <= _rows(V)
    assert len(V) == 10
    # the corners cut away by the constraint are gone
    assert (0.6, 0.1, 0.22, 0.08) not in _rows(V)


def test_parse_constraint_forms_and_errors():
    assert cons.parse_constraint("0.2 <= A + 2B <= 1", 3)["coefs"] == [1, 2, 0]
    c = cons.parse_constraint("-1 <= 3A - B", 2)
    assert (c["coefs"], c["low"], c["high"]) == ([3, -1], -1, None)
    c = cons.parse_constraint("A - 2*C >= 0", 3)
    assert (c["coefs"], c["low"]) == ([1, 0, -2], 0)
    c = cons.parse_constraint("1 >= A + B", 2)
    assert (c["low"], c["high"]) == (None, 1)
    for bad in ("A + B", "A + Z <= 1", "2 <= 3", "1 <= A <= 0"):
        with pytest.raises(ValueError):
            cons.parse_constraint(bad, 3)


def test_process_region_constraint_in_actual_units():
    reg = cons.process_region([0, 0], [10, 10], [cons.parse_constraint("A + B <= 15", 2)])
    assert _rows(reg.vertices()) == {(-1, -1), (-1, 1), (0, 1), (1, -1), (1, 0)}
    line = reg.line(np.array([0.5, 0.5]), 0, 5)  # A + B <= 15 -> coded a + b <= 1
    assert line[:, 0].max() == pytest.approx(0.5) and line[:, 0].min() == pytest.approx(-1)
    pts = reg.sample(500, np.random.default_rng(0))
    assert reg.feasible(pts).all() and (pts.sum(axis=1) <= 1 + 1e-9).all()


def test_mixture_cox_direction_keeps_the_sum_and_bounds():
    reg = cons.mixture_region(FLARE_L, FLARE_U, 1.0, [cons.parse_constraint("A + D <= 0.65",
                                                                           4)])
    P = reg.line(reg.centroid(), 1, 9)
    assert len(P) == 9 and np.allclose(P.sum(axis=1), 1) and reg.feasible(P).all()
    # hit-and-run fallback stays inside
    H = reg._hit_and_run(200, np.random.default_rng(1))
    assert reg.feasible(H).all()


# ---------------------------------------------------------------------------- E8 exact designs
E8 = {4: (8, 0.5333, 2.0), 5: (16, 0.4444, 1.6667), 6: (32, 0.3778, 1.4167),
      7: (48, 0.3111, 1.1667)}


@pytest.mark.parametrize("n", [4, 5, 6, 7])
def test_e8_exact_optimal_one_factor_quadratic(n):
    cand = np.array([[-1.0], [0.0], [1.0]])
    f = optimal.model_function(model.process_terms(1, "quadratic"))
    det, ival, aval = E8[n]
    rng = np.random.default_rng(3)
    d = optimal.optimal_design(f, n, candidates=cand, criterion="D", rng=rng, starts=12)
    X = f(d.coded)
    assert np.linalg.det(X.T @ X) == pytest.approx(det)
    i = optimal.optimal_design(f, n, candidates=cand, criterion="I", rng=rng, starts=12)
    assert i.value == pytest.approx(ival, abs=1e-4)
    a = optimal.optimal_design(f, n, candidates=cand, criterion="A", rng=rng, starts=12)
    assert a.value == pytest.approx(aval, abs=1e-4)
    if n in (4, 5):
        assert sorted(i.coded[:, 0]) == [-1] + [0] * (n - 2) + [1]
    if n == 6:
        assert sorted(d.coded[:, 0]) == [-1, -1, 0, 0, 1, 1]
        assert sorted(i.coded[:, 0]) in ([-1, -1, 0, 0, 0, 1], [-1, 0, 0, 0, 1, 1])


def test_e8_moment_matrix_is_exact():
    f = optimal.model_function(model.process_terms(1, "quadratic"))
    W = optimal.moment_matrix(f, cons.process_region([-1], [1]), 1)
    assert np.allclose(W, [[1, 0, 1 / 3], [0, 1 / 3, 0], [1 / 3, 0, 1 / 5]])


def test_coordinate_exchange_matches_point_exchange_for_a_square():
    reg = cons.process_region([-1, -1], [1, 1])
    f = optimal.model_function(model.process_terms(2, "quadratic"))
    vals = {}
    for method in ("point", "coordinate", "best"):
        r = optimal.optimal_design(f, 10, region=reg, criterion="I", method=method,
                                   rng=np.random.default_rng(0))
        vals[method] = r.value
        assert reg.feasible(r.coded).all()
    assert vals["coordinate"] == pytest.approx(vals["point"], rel=1e-3)
    assert vals["best"] <= min(vals["point"], vals["coordinate"]) + 1e-9


def test_blocks_and_forced_runs():
    reg = cons.process_region([-1, -1], [1, 1])
    f = optimal.model_function(model.process_terms(2, "quadratic"))
    forced = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1.0]])
    r = optimal.optimal_design(f, 6, region=reg, criterion="D", forced=forced,
                               forced_blocks=[1, 1, 1, 1], block_sizes=[6],
                               rng=np.random.default_rng(0))
    assert np.allclose(r.coded[:4], forced)
    assert list(r.blocks) == [1] * 4 + [2] * 6
    X = np.hstack([f(r.coded), optimal.block_columns(r.blocks)])
    assert np.linalg.matrix_rank(X) == X.shape[1]
    # two equal blocks of new runs
    r = optimal.optimal_design(f, 12, region=reg, criterion="D", block_sizes=[6, 6],
                               rng=np.random.default_rng(1))
    assert list(r.blocks) == [1] * 6 + [2] * 6


def test_optimal_builder_options_and_run_counts():
    F3 = [{"name": "A", "low": 0, "high": 10}, {"name": "B", "low": 0, "high": 10},
          {"name": "C", "low": 1, "high": 2}]
    base = {"design": "dopt", "factors": F3, "seed": 1}
    for opt in ({"criterion": "I", "exchange": "best"},
                {"criterion": "A", "exchange": "coordinate"},
                {"criterion": "I", "blocks": 2},
                {"criterion": "D", "terms": ["0", "1", "2", "0.1", "0.0"]}):
        spec = dict(base, options=opt)
        res = designs.build_design(spec)
        assert res.meta["runs"] == designs.design_run_count(spec)
        assert res.meta["criterion"] == opt["criterion"]
    spec = dict(base, options={"criterion": "I", "constraints": ["A + B <= 15"],
                               "lof_points": 3, "replicate_points": 2})
    res = designs.build_design(spec)
    df = res.dataset.df
    assert (df["A"] + df["B"] <= 15 + 1e-9).all()
    assert res.meta["constraints"][0]["text"] == "A + B <= 15"
    # forced runs stay in the design
    spec = dict(base, options={"criterion": "D", "forced": [[5, 5, 1.5], [0, 0, 1]],
                               "lof_points": 0, "replicate_points": 0})
    df = designs.build_design(spec).dataset.df
    assert ((df["A"] == 5) & (df["B"] == 5) & (df["C"] == 1.5)).any()
    # imported candidates restrict the points
    cands = [[a, b, c] for a in (0, 10) for b in (0, 10) for c in (1, 2)] + [[5, 5, 1.5]]
    spec = dict(base, options={"criterion": "D", "model": "linear", "candidates": cands,
                               "lof_points": 0, "replicate_points": 0})
    df = designs.build_design(spec).dataset.df
    assert _rows(df[["A", "B", "C"]].to_numpy()) <= _rows(cands)


def test_constrained_optimal_mixture_and_blocks():
    M = [{"name": n, "low": lo, "high": hi} for n, lo, hi in
         zip(("Mg", "Na", "Sr", "Bi"), FLARE_L, FLARE_U)]
    spec = {"design": "mixdopt", "factors": M, "seed": 2,
            "options": {"criterion": "I", "constraints": ["A + D <= 0.65"], "blocks": 2}}
    res = designs.build_design(spec)
    X = res.dataset.df[["Mg", "Na", "Sr", "Bi"]].to_numpy()
    assert np.allclose(X.sum(axis=1), 1)
    assert (X[:, 0] + X[:, 3] <= 0.65 + 1e-9).all()
    assert res.meta["blocks"] == 2
    ev_spec = {"design": "ev", "factors": M, "seed": 2,
               "options": {"include": ["vertex", "plane", "centroid"]}}
    assert designs.build_design(ev_spec).meta["runs"] == 15


# ---------------------------------------------------------------------------- E9 alias matrix
def test_e9_plackett_burman_alias_matrix():
    D = designs.plackett_burman(11)
    assert np.allclose(D.T @ D, 12 * np.eye(11))
    sp = model.Space([model.FactorDef(designs.letter(j), f"x{j}", index=j)
                      for j in range(11)])
    rows, cols, A = evaluation.alias_matrix(sp, D, model.process_terms(11, "linear"),
                                            model.process_terms(11, "2fi")[11:])
    main = A[1:]  # skip the intercept
    vals = np.unique(np.round(np.abs(main[np.abs(main) > 1e-9]), 6))
    assert np.allclose(vals, [1 / 3])
    # every main effect is partially aliased with the 2FIs not containing it
    assert np.count_nonzero(np.abs(main[0]) > 1e-9) == 45
    mm = optimal.matrix_measures(np.hstack([np.ones((12, 1)), D]))
    assert mm["d_eff"] == pytest.approx(100) and mm["a_eff"] == pytest.approx(100)
    assert mm["condition"] == pytest.approx(1)


# ---------------------------------------------------------------------------- E10 DSD
C6 = np.array([[0, 1, 1, 1, 1, 1], [1, 0, 1, -1, -1, 1], [1, 1, 0, 1, -1, -1],
               [1, -1, 1, 0, 1, -1], [1, -1, -1, 1, 0, 1], [1, 1, -1, -1, 1, 0]], float)


def _dsd_checks(D):
    m = D.shape[1]
    me = D
    two = np.column_stack([D[:, i] * D[:, j] for i, j in itertools.combinations(range(m), 2)])
    quad = D ** 2
    assert np.allclose(me.T @ two, 0) and np.allclose(me.T @ quad, 0)
    assert np.allclose(me.T @ me, np.diag(np.diag(me.T @ me)))


def test_e10_definitive_screening_from_the_published_conference_matrix():
    assert np.allclose(C6.T @ C6, 5 * np.eye(6))
    res = designs.definitive_screening(6, C=C6)
    D = res["coded"]
    assert D.shape == (13, 6)
    _dsd_checks(D)


@pytest.mark.parametrize("m,runs", [(4, 9), (5, 13), (6, 13), (7, 17), (8, 17), (10, 21),
                                    (12, 25), (14, 29)])
def test_dsd_run_counts_and_orthogonality(m, runs):
    spec = {"design": "dsd", "factors": [{"name": f"F{j}"} for j in range(m)], "seed": 1}
    res = designs.build_design(spec)
    assert res.meta["runs"] == runs == designs.design_run_count(spec)
    coded = model.space_from_dataset(res.dataset).to_coded(
        res.dataset.df[[f["column"] for f in res.meta["factors"]]].to_numpy(float))
    _dsd_checks(coded)
    assert set(np.unique(coded)) == {-1.0, 0.0, 1.0}


@pytest.mark.parametrize("n", [4, 6, 8, 10, 12, 14, 18, 20, 24, 26, 28, 30, 32])
def test_paley_conference_matrices(n):
    C = designs.conference_matrix(n)
    assert np.allclose(C.T @ C, (n - 1) * np.eye(n)) and np.allclose(np.diag(C), 0)


def test_dsd_with_categoric_factor_and_fake_factors():
    facs = [{"name": c} for c in "ABCDE"] + [{"name": "G", "type": "categoric",
                                             "levels": ["a", "b"]}]
    res = designs.build_design({"design": "dsd", "factors": facs, "seed": 1})
    assert res.meta["runs"] == 14  # 2 x 6 + 2 centre runs (categoric at both levels)
    g = res.dataset.df[res.meta["factors"][-1]["column"]]
    assert set(g) == {1.0, 2.0}
    res = designs.build_design({"design": "dsd", "factors": facs[:5], "seed": 1,
                                "options": {"fake_factors": 2}})
    assert res.meta["runs"] == 17 and res.meta["conference_order"] == 8


# ---------------------------------------------------------------------------- small builders
def test_box_behnken_blocks_are_orthogonal():
    for k, nb, runs in ((4, 3, 27), (5, 2, 46)):
        res = designs.box_behnken(k, blocks=nb)
        D, b = res["coded"], res["blocks"]
        assert len(D) == runs and len(set(b)) == nb
        # orthogonal blocking: the sum of squares of every factor is equal in each block
        for j in range(k):
            ss = [np.sum(D[b == g, j] ** 2) for g in set(b)]
            assert np.allclose(ss, ss[0])
        X = model.raw_matrix(D, model.process_terms(k, "quadratic"))
        B = optimal.block_columns(b)
        assert np.allclose((X[:, 1:] - X[:, 1:].mean(0)).T @ B, 0)
    spec = {"design": "bbd", "factors": [{"name": c} for c in "ABCD"],
            "options": {"blocks": 3}}
    assert designs.build_design(spec).meta["blocks"] == 3
    with pytest.raises(ValueError):
        designs.box_behnken(3, blocks=2)


def test_simplex_screening_one_factor_lhs_distance_userdef():
    pts, types = designs.simplex_screening(4)
    assert len(pts) == 13 and np.allclose(pts.sum(axis=1), 1)
    assert list(np.round(pts[4], 4)) == [0.625, 0.125, 0.125, 0.125]
    res = designs.build_design({"design": "onefactor", "seed": 1, "options": {
        "model": "cubic"}, "factors": [{"name": "T", "low": 10, "high": 50}]})
    t = res.dataset.df["T"]
    assert t.min() == 10 and t.max() == 50 and t.nunique() >= 4
    res = designs.build_design({"design": "lhs", "seed": 3, "options": {"runs": 12},
                                "factors": [{"name": "A"}, {"name": "B"}]})
    x = res.dataset.df["A"].to_numpy()
    assert sorted(np.floor((x + 1) / 2 * 12).astype(int)) == list(range(12))  # one per slice
    res = designs.build_design({"design": "distance", "seed": 3,
                                "options": {"runs": 9, "levels": 3},
                                "factors": [{"name": "A"}, {"name": "B"}]})
    assert _rows(res.dataset.df[["A", "B"]].to_numpy()) == _rows(
        list(itertools.product((-1, 0, 1), repeat=2)))
    res = designs.build_design({"design": "userdef", "seed": 1, "options": {
        "include": ["vertex", "edge", "centroid"], "center_points": 2},
        "factors": [{"name": "A"}, {"name": "B"}, {"name": "C"}]})
    assert res.meta["runs"] == 8 + 12 + 1 + 2


# ---------------------------------------------------------------------------- combined
def test_combined_terms_and_space():
    terms = tc.combined_terms(3, 1, "quadratic", "linear")
    assert len(terms) == 12
    assert (0, 3) in terms and (0, 1, 3) in terms and () not in terms
    assert tc.term_label((0, 1, 3), 3) == "AB·D"
    assert len(tc.combined_terms(3, 2, "linear", "quadratic")) == 3 * 6
    with pytest.raises(ValueError):
        tc.combined_terms(3, 1, "cubic", "linear")


def test_combined_design_build_space_and_evaluation():
    from rsstatchem.ui.dx.document import DesignDocument

    spec = {"design": "combined", "seed": 3, "options": {"mix_model": "quadratic",
                                                         "process_model": "linear"},
            "factors": [{"name": "x1", "type": "mixture"}, {"name": "x2", "type": "mixture"},
                        {"name": "x3", "type": "mixture", "high": 0.6},
                        {"name": "T", "low": 20, "high": 40}]}
    res = designs.build_design(spec)
    assert res.meta["kind"] == "combined" and res.meta["runs"] == designs.design_run_count(spec)
    df = res.dataset.df
    assert np.allclose(df[["x1", "x2", "x3"]].sum(axis=1), 1)
    assert df["x3"].max() <= 0.6 + 1e-9 and set(df["T"]) <= {20.0, 30.0, 40.0}
    doc = DesignDocument.from_spec(spec)
    sp = doc.space()
    assert isinstance(sp, tc.CombinedSpace) and sp.n_mix == 3
    coded = doc.coded_matrix()
    assert np.allclose(coded[:, :3].sum(axis=1), 1) and np.abs(coded[:, 3]).max() == 1
    assert np.allclose(sp.to_actual(coded), df[doc.factor_cols].to_numpy())
    ev = evaluation.evaluate(sp, coded, terms=doc.model_terms("quadratic*linear"))
    assert ev["p"] == 12 and not ev["intercept"] and ev["df"]["Model"] == 11
    frac, se = evaluation.fds_curve(sp, ev, n_points=400)
    assert len(se) == 400 and np.all(np.diff(se) >= 0)
    back = DesignDocument.from_dict(json.loads(json.dumps(doc.to_dict())))
    assert back.combined and isinstance(back.space(), tc.CombinedSpace)


# ---------------------------------------------------------------------------- historical
def test_design_from_data_and_blank_historical_layout():
    from rsstatchem.ui.dx.document import DesignDocument

    frame = pd.DataFrame({"T": [1, 2, 3, 4, 5], "cat": list("abaab"),
                          "y": [3, 4, 5, 6, 8.0]})
    doc = DesignDocument.from_data(frame, ["T", "cat"], ["y"])
    f = doc.factors
    assert (f[0]["low"], f[0]["high"]) == (1, 5) and f[1]["labels"] == ["a", "b"]
    assert doc.response_count(doc.response_cols[0]) == 5
    assert doc.meta["design"] == "historical" and doc.editable_factors
    a = doc.analyze(doc.response_cols[0])
    assert a.fit is not None
    blank = DesignDocument.from_spec({"design": "historical", "options": {"runs": 4},
                                      "factors": [{"name": "A", "low": 0, "high": 10}]})
    assert blank.n_runs == 4 and np.isnan(blank.ds.df["A"]).all()
    blank.set_values([(0, "A", 5.0)])
    assert blank.coded_matrix()[0, 0] == 0


# ---------------------------------------------------------------------------- augmentation
def _factorial_doc():
    from rsstatchem.ui.dx.document import DesignDocument

    return DesignDocument.from_spec({"design": "factorial", "seed": 1, "factors": [
        {"name": "A", "low": 0, "high": 10}, {"name": "B", "low": 5, "high": 7}]})


def test_augment_factorial_to_ccd_and_optimal_augment():
    doc = _factorial_doc()
    assert doc.add_center_points(3) == 3 and doc.n_blocks == 1
    assert doc.add_axial_points(None, 2) == 6
    assert doc.n_blocks == 2 and doc.meta["alpha"] == pytest.approx(2 ** 0.5)
    coded = doc.coded_matrix()
    assert np.isclose(np.abs(coded).max(), 2 ** 0.5)
    ev = evaluation.evaluate(doc.space(), coded, "quadratic",
                             blocks=doc.ds.df["Block"].to_numpy(float))
    assert not ev["aliased"]
    terms = doc.model_terms("cubic")
    n = doc.augment_default_runs(terms)
    before = doc.n_runs
    assert doc.optimal_augment(terms, n, "D", seed=1) == n
    assert doc.n_runs == before + n and doc.n_blocks == 3
    X = model.build_matrix(doc.space(), doc.coded_matrix(), terms,
                           doc.ds.df["Block"].to_numpy(float)).X
    assert np.linalg.matrix_rank(X) == X.shape[1]
    assert [a["kind"] for a in doc.meta["augments"]] == ["center", "axial", "optimal"]


def test_replicates_modify_space_round_and_layout_edits():
    doc = _factorial_doc()
    snap = doc.snapshot()
    assert doc.add_replicates() == 4 and doc.n_blocks == 2
    doc.modify_design_space(0, 0, 20)
    assert doc.coded_matrix()[:, 0].max() == 0  # A = 10 is now the centre
    doc.ds.df.loc[0, "B"] = 5.123
    assert doc.round_factors(decimals=1) == 1 and doc.ds.df.loc[0, "B"] == 5.1
    doc.insert_runs(n=2)
    assert doc.n_runs == 10
    doc.ignored = {9}
    assert doc.delete_runs([0, 1]) == 2
    assert sorted(doc.ds.df["Std"]) == list(range(1, 9)) and doc.ignored == {7}
    col = doc.add_response("Purity", "%")
    assert col in doc.response_cols and len(doc.responses) == 2
    doc.rename_block(2, "Day 2")
    assert doc.block_name(2) == "Day 2"
    doc.restore(snap)
    assert doc.n_runs == 4 and doc.response_cols == ["R1"]


def test_mixture_rounding_keeps_the_total_and_constraints_round_trip():
    from rsstatchem.ui.dx.document import DesignDocument

    doc = DesignDocument.from_spec({"design": "mixdopt", "seed": 2, "factors": [
        {"name": "A"}, {"name": "B"}, {"name": "C"}], "options": {
        "criterion": "D", "constraints": ["A + B <= 0.8"]}})
    doc.ds.df.loc[0, doc.factor_cols] = [0.3333333, 0.3333333, 0.3333334]
    doc.round_factors(decimals=2)
    assert doc.ds.df.loc[:, doc.factor_cols].sum(axis=1).round(10).eq(1).all()
    assert doc.constraints[0]["text"] == "A + B <= 0.8"
    doc.set_constraints(["A + B <= 0.7"])
    assert doc.region().vertices().shape[0] >= 3
    back = DesignDocument.from_dict(json.loads(json.dumps(doc.to_dict())))
    assert back.constraints[0]["text"] == "A + B <= 0.7"


def test_old_rsx_without_new_keys_still_loads():
    from rsstatchem.ui.dx.document import DesignDocument

    doc = _factorial_doc()
    d = json.loads(json.dumps(doc.to_dict()))
    for key in ("criterion", "exchange", "augments", "constraints"):
        d["design"].pop(key, None)
    back = DesignDocument.from_dict(d)
    assert back.constraints == [] and not back.combined and back.n_runs == 4


# ---------------------------------------------------------------------------- evaluation extras
def test_matrix_measures_fds_score_power_and_correlation():
    res = designs.build_design({"design": "ccd", "seed": 1, "factors": [
        {"name": "A"}, {"name": "B"}]})
    sp = model.space_from_dataset(res.dataset)
    coded = sp.to_coded(res.dataset.df[["A", "B"]].to_numpy(float))
    terms = model.process_terms(2, "quadratic")
    mm = evaluation.matrix_measures(sp, coded, terms)
    assert 0 < mm["d_eff"] < 100 and mm["max_leverage"] <= 1 + 1e-12
    assert mm["avg_leverage"] == pytest.approx(6 / 13)
    assert mm["i_value"] > 0
    ev = evaluation.evaluate(sp, coded, terms=terms)
    se = evaluation.fds_values(sp, ev, n_points=1000)
    sc = evaluation.fds_score(se, d=2.0, sigma=1.0, alpha=0.05, df=ev["df"]["Residuals"])
    t = sc["t"]
    assert sc["limit"] == pytest.approx(2.0 / t) and 0 <= sc["fraction"] <= 1
    assert np.mean(se <= sc["limit"]) == pytest.approx(sc["fraction"])
    c = ev["rows"][1]["se"] ** 2
    assert evaluation.power_delta(c, 7, 2.0, 1.0) == pytest.approx(
        evaluation.power(c, 7, 2.0))
    corr = evaluation.correlations(sp, coded, terms)
    assert np.allclose(np.diag(corr["coef"]), 1) and corr["r"].shape == (5, 5)
    G1, G2, s = evaluation.se_grid(sp, ev, 0, 1)
    assert s.shape == G1.shape and s.min() > 0


# ---------------------------------------------------------------------------- UI
@pytest.fixture
def ctx(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("RSSTATCHEM_HOME", str(tmp_path))
    from rsstatchem.app import AppContext

    context = AppContext(qapp)
    monkeypatch.setattr(context.app, "quit", lambda: None)
    return context


def make_window(ctx, doc):
    from rsstatchem.ui.dx.window import DXMainWindow

    w = DXMainWindow(ctx, doc)
    ctx.register_window(w)
    w.show()
    return w


@pytest.mark.parametrize("family,design", [
    ("factorial", "dsd"), ("rsm", "onefactor"), ("rsm", "userdef"), ("rsm", "historical"),
    ("rsm", "lhs"), ("rsm", "distance"), ("mixture", "simscreen"), ("custom", "combined")])
def test_wizard_builds_new_designs(qapp, family, design):
    from rsstatchem.ui.dx.wizard import NewDesignWizard

    wiz = NewDesignWizard(None, family, design)
    assert wiz.design == design
    wiz.finish()
    assert wiz.document is not None, wiz.error.text()
    assert wiz.document.meta["design"] == design


def test_wizard_optimal_engine_options_and_dialogs(qapp):
    from rsstatchem.ui.dx.wizard import NewDesignWizard

    wiz = NewDesignWizard(None, "rsm", "dopt")
    wiz.k_spin.setValue(3)
    assert wiz._opt_value("criterion") == "I" and wiz._opt_value("exchange") == "best"
    dlg = wiz.edit_constraints()
    dlg.set_text("A + B <= 0.5")
    dlg.apply()
    dlg = wiz.edit_model()
    dlg.set_selected([t for t in dlg.terms if model.term_degree(t) == 1])
    dlg.accept()
    dlg = wiz.edit_runs("forced")
    dlg.set_text("0, 0, 0")
    dlg.apply()
    wiz.set_option("opt_blocks", 2)
    opt = wiz.spec()["options"]
    assert opt["terms"] == ["0", "1", "2"] and opt["blocks"] == 2
    assert opt["constraints"][0]["high"] == 0.5 and opt["forced"] == [[0.0, 0.0, 0.0]]
    wiz.finish()
    doc = wiz.document
    assert doc is not None, wiz.error.text()
    assert doc.meta["forced_runs"] == 1 and doc.n_blocks == 2
    df = doc.ds.df
    assert (df[doc.factor_cols[0]] + df[doc.factor_cols[1]] <= 0.5 + 1e-9).all()
    # invalid constraint text keeps the dialog open
    dlg = wiz.edit_constraints()
    dlg.set_text("A + <= 3")
    assert not dlg.check()


def test_wizard_combined_and_bbd_blocks(qapp):
    from rsstatchem.ui.dx.wizard import NewDesignWizard

    wiz = NewDesignWizard(None, "custom", "combined")
    wiz.k_spin.setValue(4)
    spec = wiz.spec()
    assert [f.get("type") for f in spec["factors"]][:2] == ["mixture", "mixture"]
    wiz.finish()
    assert wiz.document.combined
    wiz = NewDesignWizard(None, "rsm", "bbd")
    wiz.k_spin.setValue(4)
    wiz.set_option("blocks", 3)
    wiz.finish()
    assert wiz.document.n_blocks == 3 and wiz.document.n_runs == 27


def test_window_augment_menus_undo_and_nodes(ctx):
    doc = _factorial_doc()
    win = make_window(ctx, doc)
    win._update_tool_actions()
    assert win.a_axial.isEnabled() and win.a_foldover.isEnabled()
    dlg = win._tool(__import__("rsstatchem.ui.dx.window", fromlist=["x"]).CenterPointsDialog)
    dlg.n.setValue(2)
    dlg.apply()
    assert doc.n_runs == 6
    win.augment_axial(1.0, 1)
    assert doc.n_runs == 11 and doc.n_blocks == 2
    win.undo.undo()
    assert doc.n_runs == 6
    win.undo.redo()
    assert doc.n_runs == 11
    col = win.add_response("Purity")
    assert f"analysis:{col}" in win.nav_items
    win.undo.undo()
    assert f"analysis:{col}" not in win.nav_items
    page = win.show_node("design")
    page.view.selectRow(0)
    assert win.delete_selected_runs() == 1 and doc.n_runs == 10
    win.insert_run()
    assert doc.n_runs == 11
    from rsstatchem.ui.dx.window import OptimalAugmentDialog

    dlg = win._tool(OptimalAugmentDialog)
    dlg.order.setCurrentIndex(dlg.order.findData("cubic"))
    n = dlg.runs.value()
    dlg.apply()
    assert doc.n_runs == 11 + n
    ev = win.show_node("evaluation")
    assert "Matrix Measures" in [t.title for t in ev.report.tables()]
    for key in ("se", "corr", "pearson", "alias", "fds"):
        ev.set_graph(key)
        assert ev.graph.png or ev.graph.message
    ev.fds_d.setValue(1.0)
    assert "FDS Score" in [t.title for t in ev.report.tables()]
    summary = win.show_node("summary")
    rows = dict((r[0], r[1]) for r in summary.report.tables()[0].rows if r[0] != "Augmented")
    assert rows["Runs"] == str(doc.n_runs)


def test_window_constraints_node_and_import_data(ctx, tmp_path):
    from rsstatchem.ui.dx.document import DesignDocument

    doc = DesignDocument.from_spec({"design": "dopt", "seed": 1, "factors": [
        {"name": "A", "low": 0, "high": 10}, {"name": "B", "low": 0, "high": 10}],
        "options": {"criterion": "D", "constraints": ["A + B <= 15"]}})
    win = make_window(ctx, doc)
    assert "constraints" in win.nav_items
    page = win.show_node("constraints")
    titles = [t.title for t in page.report.tables()]
    assert titles[0] == "Multiple Linear Constraints" and "Vertices" in titles[1]
    assert page.graph.png
    dlg = page.edit()
    dlg.set_text("A + B <= 12")
    dlg.apply()
    assert doc.constraints[0]["high"] == 12
    assert doc.infeasible_runs()  # runs built for A + B <= 15
    path = tmp_path / "hist.csv"
    pd.DataFrame({"Temp": [50, 60, 70, 55, 65], "Cat": ["x", "y", "x", "y", "x"],
                  "Yield": [70, 75, 72, 74, 78.0]}).to_csv(path, index=False)
    idlg = win.import_data_from(str(path))
    assert idlg.roles["Cat"].currentData() == "categoric"
    assert idlg.roles["Yield"].currentData() == "response"
    idlg.apply()
    assert idlg.document is not None and idlg.document.n_runs == 5
    w2 = win.adopt(idlg.document)
    assert w2 is not win and w2.doc.editable_factors


def test_historical_layout_accepts_factor_values(ctx):
    from PySide6.QtCore import Qt

    from rsstatchem.ui.dx.document import DesignDocument

    doc = DesignDocument.from_spec({"design": "historical", "options": {"runs": 3},
                                    "factors": [{"name": "A", "low": 0, "high": 10},
                                                {"name": "C", "type": "categoric",
                                                 "levels": ["lo", "hi"]}]})
    win = make_window(ctx, doc)
    page = win.show_node("design")
    m = page.model
    ca = next(i for i, (k, c) in enumerate(m.columns) if k == "factor" and c == "A")
    cc = next(i for i, (k, c) in enumerate(m.columns) if k == "factor" and c == "C")
    assert m.flags(m.index(0, ca)) & Qt.ItemFlag.ItemIsEditable
    assert m.setData(m.index(0, ca), "5")
    assert m.setData(m.index(0, cc), "hi")
    row = m.rows[0]
    assert doc.value(row, "A") == 5 and doc.value(row, "C") == 2
    assert not m.setData(m.index(1, cc), "middle")
    assert m.paste_text("7\thi\n3\tlo", 1, ca) == 4


def test_extension_nodes_appear_when_views_exist(ctx, monkeypatch):
    from PySide6.QtWidgets import QLabel

    import rsstatchem.ui.dx.analysis as analysis
    from rsstatchem.ui.dx import window

    class AnalysisSummaryView(QLabel):
        def __init__(self, doc):
            super().__init__("summary")

    monkeypatch.setattr(analysis, "AnalysisSummaryView", AnalysisSummaryView, raising=False)
    win = make_window(ctx, _factorial_doc())
    assert "analysis_summary" in win.nav_items
    assert isinstance(win.show_node("analysis_summary"), AnalysisSummaryView)
    assert window.extension_class("rsstatchem.ui.dx.optimization", "NoSuchView") is None


def test_poe_node_and_column_info_feed_the_optimization(ctx):
    from rsstatchem.ui.dx import optimization
    from rsstatchem.ui.dx.examples import chemical_conversion
    from rsstatchem.ui.dx.window import InfoDialog

    doc = chemical_conversion()
    win = make_window(ctx, doc)
    if getattr(optimization, "POEGraphView", None) is not None:
        assert "poe" in win.nav_items and win.a_poe.isEnabled()
        assert isinstance(win.show_node("poe"), optimization.POEGraphView)
    dlg = InfoDialog(doc, win)
    j, sd_edit, lv_edit = dlg.extra[0]
    sd_edit.setText("0.5")
    lv_edit.setText("40, 45, 50")
    dlg.apply()
    col = doc.factor_cols[j]
    assert doc.optimization["poe_sd"][col] == 0.5
    assert doc.factors[j]["discrete_levels"] == [40.0, 45.0, 50.0]
    if hasattr(optimization, "discrete_levels"):
        assert optimization.discrete_levels(doc)[j] == [40.0, 45.0, 50.0]


def test_chemical_conversion_example_matches_e2():
    from rsstatchem.ui.dx.examples import chemical_conversion

    doc = chemical_conversion()
    sp = doc.space()
    obs = model.observations(doc.ds, sp, doc.response_cols[0])
    fit = model.Fit(sp, obs, model.process_terms(3, "quadratic"))
    b = dict(zip([model.term_name(t) for t in [()] + fit.terms], fit.beta))
    assert b["Intercept"] == pytest.approx(81.091, abs=2e-3)
    assert b["B"] == pytest.approx(4.040, abs=2e-3) and b["AC"] == pytest.approx(11.375)


def test_analysis_summary_node_preferences_and_status_text(ctx):
    from rsstatchem.core.doe import model as mdl
    from rsstatchem.ui.dx import analysis
    from rsstatchem.ui.dx.examples import filtration_rate

    win = make_window(ctx, filtration_rate())
    if getattr(analysis, "AnalysisSummaryView", None) is not None:
        page = win.show_node("analysis_summary")
        assert isinstance(page, analysis.AnalysisSummaryView) and page.report is not None
    col = win.doc.response_cols[0]
    view = win.show_node(f"analysis:{col}")
    if hasattr(view, "strip") and "Effects" in view.strip.labels():
        assert "Effects" in win.statusBar().currentMessage()
        assert "Fit Summary" not in win.statusBar().currentMessage()
    if getattr(mdl, "PREFERENCES", None) is None:
        return
    before = mdl.PREFERENCES.to_meta()
    try:
        dlg = win.edit_preferences()
        dlg.spins["alpha"].setValue(0.10)
        dlg.spins["ti_proportion"].setValue(0.95)
        dlg.apply()
        assert mdl.PREFERENCES.alpha == pytest.approx(0.10)
        assert mdl.PREFERENCES.ti_proportion == pytest.approx(0.95)
        assert float(win.setting("pref_alpha")) == pytest.approx(0.10)
    finally:
        mdl.PREFERENCES.update(before)
