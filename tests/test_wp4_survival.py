"""WP4 survival: Kaplan-Meier, Life Tables and Cox regression against statsmodels, lifelines
and published values (Kleinbaum & Klein leukemia remission data = MASS::gehan; R survival
``lung``)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.duration.survfunc import SurvfuncRight, survdiff

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.stats import survival as sv

DATA = Path(__file__).parent / "data_wp4"


def _table(section, title):
    items = [i for i in section.items if getattr(i, "title", "").startswith(title)]
    exact = [i for i in items if i.title.rstrip("ᵃ") == title]
    return (exact or items)[0]


@pytest.fixture(scope="module")
def gehan():
    g = pd.read_csv(DATA / "gehan.csv")
    g["rx"] = (g["treat"] == "control").astype(float)   # 1 = placebo, 0 = 6-MP
    return g


@pytest.fixture(scope="module")
def lung():
    l_ = pd.read_csv(DATA / "lung.csv").rename(columns={"ph.ecog": "ecog"})
    l_["dead"] = (l_["status"] == 2).astype(float)
    return l_


# ------------------------------------------------------------------ Kaplan-Meier
def test_km_matches_statsmodels_survfunc(lung):
    for sex in (1, 2):
        sub = lung[lung.sex == sex]
        c = sv.km_fit(sub.time, sub.dead)
        ref = SurvfuncRight(sub.time.to_numpy(), sub.dead.to_numpy())
        np.testing.assert_allclose(c.times, ref.surv_times)
        np.testing.assert_allclose(c.surv, ref.surv_prob, rtol=1e-10)
        np.testing.assert_allclose(c.se, ref.surv_prob_se, rtol=1e-8)


def test_km_medians_match_r_survfit(lung):
    # R: survfit(Surv(time, status) ~ sex, lung) -> medians 270 (sex=1) and 426 (sex=2)
    assert sv.km_percentile(sv.km_fit(lung.time[lung.sex == 1], lung.dead[lung.sex == 1]),
                            0.5)[0] == 270
    assert sv.km_percentile(sv.km_fit(lung.time[lung.sex == 2], lung.dead[lung.sex == 2]),
                            0.5)[0] == 426


def test_km_restricted_mean_matches_lifelines(lung):
    lifelines = pytest.importorskip("lifelines")
    from lifelines.utils import restricted_mean_survival_time

    sub = lung[lung.sex == 1]
    c = sv.km_fit(sub.time, sub.dead)
    mean, se, restricted = sv.km_mean(c)
    kmf = lifelines.KaplanMeierFitter().fit(sub.time, sub.dead)
    ref = restricted_mean_survival_time(kmf, t=float(sub.time.max()))
    assert mean == pytest.approx(ref, rel=1e-8)
    assert se > 0
    # R print(survfit, rmean="common") for sex=1 with all data: *rmean 326 (se 22.9) with
    # the curve restricted at max time 1022 for the pooled data; here per-group max time.
    assert not restricted or sub.sort_values("time").dead.iloc[-1] == 0


def test_logrank_family_matches_statsmodels_and_r(lung, gehan):
    chi, df, p = sv.compare_groups(lung.time, lung.dead, lung.sex, weight="logrank")
    assert chi == pytest.approx(10.3267, abs=1e-3) and df == 1   # R survdiff: 10.3
    for w, ref_w in (("logrank", None), ("breslow", "gb"), ("tarone", "tw")):
        chi, _, p = sv.compare_groups(lung.time, lung.dead, lung.sex, weight=w)
        ref = survdiff(lung.time, lung.dead, lung.sex, weight_type=ref_w)
        assert chi == pytest.approx(ref[0], rel=1e-8)
        assert p == pytest.approx(ref[1], rel=1e-6)
    # Kleinbaum & Klein, remission data: log-rank chi-square 16.79
    chi, _, _ = sv.compare_groups(gehan.time, gehan.cens, gehan.rx)
    assert chi == pytest.approx(16.793, abs=1e-3)


def test_stratified_and_k_group_tests(lung):
    sub = lung.dropna(subset=["ecog"])
    sub = sub[sub.ecog < 3]
    for w, ref_w in (("logrank", None), ("breslow", "gb")):
        chi, df, _ = sv.compare_groups(sub.time, sub.dead, sub.ecog, sub.sex, weight=w)
        ref = survdiff(sub.time, sub.dead, sub.ecog, strata=sub.sex, weight_type=ref_w)
        assert df == 2
        assert chi == pytest.approx(ref[0], rel=1e-8)
    chi_t, df_t, _ = sv.compare_groups(sub.time, sub.dead, sub.ecog, trend=True)
    assert df_t == 1 and chi_t > 0


def test_kaplan_meier_section_and_saved(gehan):
    ds = Dataset(gehan[["time", "cens", "rx"]])
    res = sv.kaplan_meier(ds, "time", "cens", ("single", 1.0), "rx",
                          tests=("logrank", "breslow", "tarone"), quartiles=True,
                          plots=("survival", "one_minus", "hazard", "log_survival"),
                          save=("survival", "se", "hazard", "cum_events"))
    rows = _table(res.section, "Overall Comparisons").rows
    assert rows[0][1] == "16.793"
    mm = _table(res.section, "Means and Medians").rows
    assert mm[0][5] == "23.000" and mm[1][5] == "8.000"
    surv = res.saved["SUR"][1]
    c = sv.km_fit(gehan.time[gehan.rx == 1], gehan.cens[gehan.rx == 1])
    m = gehan.rx == 1
    np.testing.assert_allclose(surv[m], c.at(gehan.time[m]))
    assert sum(1 for i in res.section.items if i.kind == "image") == 4
    pw = sv.kaplan_meier(ds, "time", "cens", ("single", 1.0), "rx", tests=("logrank",),
                         compare="pairwise", survival_table=False)
    assert _table(pw.section, "Pairwise Comparisons").rows[0][4] == "16.793"


def test_event_definitions():
    s = pd.Series([1.0, 2.0, 3.0, np.nan, 0.0])
    from rsstatchem.core.stats.wp4_util import event_mask

    np.testing.assert_array_equal(event_mask(s, ("range", 1, 2)).to_numpy()[:3], [1, 1, 0])
    np.testing.assert_array_equal(event_mask(s, ("list", [3, 0])).fillna(-1).to_numpy(),
                                  [0, 0, 1, -1, 1])


# ------------------------------------------------------------------ Life tables
def test_actuarial_formulas_without_censoring():
    t = np.array([0.5, 1.5, 1.7, 2.2, 3.9, 4.1, 5.5, 6.0])
    lt = sv.actuarial(t, np.ones_like(t), 2.0, 6.0)
    # no censoring: the cumulative proportion is the empirical survival at interval ends
    np.testing.assert_allclose(lt["cum"].to_numpy()[:3], [5 / 8, 3 / 8, 1 / 8])
    np.testing.assert_allclose(lt["hazard"].iloc[0], 2 * 3 / (2.0 * (2 * 8 - 3)))
    assert lt["exposed"].iloc[0] == 8


def test_lee_desu_matches_bruteforce(gehan):
    t, d, g = gehan.time.to_numpy(float), gehan.cens.to_numpy(float), gehan.rx.to_numpy()
    n = len(t)
    u = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            gt = (t[i] > t[j] and d[j] == 1) or (t[i] == t[j] and d[i] == 0 and d[j] == 1)
            lt_ = (t[i] < t[j] and d[i] == 1) or (t[i] == t[j] and d[i] == 1 and d[j] == 0)
            u[i] += int(gt) - int(lt_)
    w = sum(u[g == k].sum() ** 2 / (g == k).sum() for k in (0, 1))
    ref = (n - 1) * w / np.sum(u ** 2)
    stat, df, _ = sv.lee_desu(t, d, g)
    assert stat == pytest.approx(ref) and df == 1


def test_life_tables_section(gehan):
    ds = Dataset(gehan[["time", "cens", "rx"]])
    sec = sv.life_tables(ds, "time", "cens", ("single", 1.0), 5, 30, "rx", (0, 1),
                         compare=True, plots=("survival", "hazard", "density"))
    assert _table(sec, "Life Table").rows[0][1] == ".000"
    med = _table(sec, "Median Survival Times").rows
    assert len(med) == 2
    assert float(_table(sec, "Overall Comparisons").rows[0][1]) > 10


# ------------------------------------------------------------------ Cox regression
def test_cox_kleinbaum_klein_remission(gehan):
    """K&K chapter 3, model 1 (Rx only, Breslow ties): b = 1.509, SE = 0.410, HR = 4.523,
    -2 log L = 172.759."""
    ds = Dataset(gehan[["time", "cens", "rx"]])
    res = sv.cox_regression(ds, "time", "cens", ("single", 1.0), ["rx"])
    row = _table(res.section, "Variables in the Equation").rows[0]
    assert row[2:4] == ["1.509", ".410"] and row[7] == "4.523"
    omni = _table(res.section, "Omnibus Tests of Model Coefficients").rows[0]
    assert omni[1] == "172.759"


def test_cox_matches_phreg_breslow_with_categorical_and_strata(lung):
    sub = lung.dropna(subset=["ecog"]).copy()
    ds = Dataset(sub[["time", "dead", "age", "sex", "ecog", "inst"]].fillna(-1))
    res = sv.cox_regression(ds, "time", "dead", ("single", 1.0), ["age", "ecog"],
                            {"ecog": ("indicator", "first")}, strata="sex",
                            correlations=True, baseline=True)
    X = pd.get_dummies(sub.ecog.astype(int), prefix="e", drop_first=True).astype(float)
    X.insert(0, "age", sub.age)
    ref = PHReg(sub.time, X, status=sub.dead, strata=sub.sex, ties="breslow").fit()
    rows = [r for r in _table(res.section, "Variables in the Equation").rows if r[2]]
    np.testing.assert_allclose([float(r[2]) for r in rows], ref.params, atol=6e-4)
    np.testing.assert_allclose([float(r[3]) for r in rows], ref.bse, atol=6e-4)


def test_cox_engine_exact(lung):
    sub = lung.dropna(subset=["ecog"])
    X = np.column_stack([sub.age, sub.sex, sub.ecog])
    data = sv.CoxData(sub.time, sub.dead, X)
    fit = sv.cox_fit(data, [0, 1, 2])
    ref = PHReg(sub.time, X, status=sub.dead, ties="breslow").fit()
    np.testing.assert_allclose(fit.beta, ref.params, rtol=1e-6)
    np.testing.assert_allclose(np.sqrt(np.diag(fit.cov)), ref.bse, rtol=1e-5)
    assert fit.ll == pytest.approx(ref.llf, rel=1e-10)


def test_contrasts_reparametrize_same_model(lung):
    sub = lung.dropna(subset=["ecog"])
    sub = sub[sub.ecog < 3]
    ds = Dataset(sub[["time", "dead", "ecog"]])
    lls = []
    for con in ("indicator", "simple", "difference", "helmert", "repeated", "polynomial",
                "deviation"):
        res = sv.cox_regression(ds, "time", "dead", ("single", 1.0), ["ecog"],
                                {"ecog": (con, "last")})
        lls.append(_table(res.section, "Omnibus Tests of Model Coefficients").rows[0][1])
    assert len(set(lls)) == 1


def test_cox_forward_lr_selects_sex_and_ecog(lung):
    sub = lung.dropna(subset=["ecog"])
    ds = Dataset(sub[["time", "status", "age", "sex", "ecog"]])
    res = sv.cox_regression(ds, "time", "status", ("single", 2.0), ["age", "sex", "ecog"],
                            {"ecog": ("indicator", "first")}, method="fstep_lr")
    last = [i for i in res.section.items if getattr(i, "title", "") ==
            "Variables in the Equation"][-1]
    assert {r[1] for r in last.rows} == {"sex", "ecog", "ecog(1)", "ecog(2)", "ecog(3)"}
    for method in ("fstep_wald", "fstep_cond", "bstep_lr", "bstep_wald", "bstep_cond"):
        out = sv.cox_regression(ds, "time", "status", ("single", 2.0), ["age", "sex"],
                                method=method)
        assert any(getattr(i, "title", "") == "Variables in the Equation" for i in
                   out.section.items)


def test_cox_baseline_residuals_vs_lifelines():
    lifelines = pytest.importorskip("lifelines")
    rng = np.random.default_rng(4)
    n = 150
    x1, x2 = rng.normal(size=n), rng.integers(0, 2, n).astype(float)
    t = rng.exponential(1 / np.exp(0.5 * x1 - 0.7 * x2))
    c = rng.exponential(2.0, n)
    df = pd.DataFrame({"T": np.minimum(t, c), "E": (t <= c).astype(float), "x1": x1, "x2": x2})
    cph = lifelines.CoxPHFitter().fit(df, "T", "E")   # no ties: Efron == Breslow
    ds = Dataset(df)
    res = sv.cox_regression(ds, "T", "E", ("single", 1.0), ["x1", "x2"], baseline=True,
                            save=("survival", "partial", "dfbeta", "xbeta"))
    rows = _table(res.section, "Variables in the Equation").rows
    np.testing.assert_allclose([float(r[2]) for r in rows], cph.params_, atol=6e-4)
    base = _table(res.section, "Survival Table").rows
    ref_s = cph.baseline_survival_["baseline survival"]
    times = np.array([float(r[0]) for r in base])
    s_mean = np.array([float(r[2]) for r in base])
    ref_at = ref_s.reindex(times, method="nearest").to_numpy()
    unique = ~pd.Series(times).duplicated(keep=False).to_numpy()   # printed with 3 decimals
    # SPSS (IBM COXREG Algorithms) uses the Kalbfleisch-Prentice baseline with covariates
    # centred at their means, not lifelines' Breslow estimator: check it independently
    # (no ties: alpha_i = (1 - e_i / sum_R e)^(1 / e_i)) and stay close to Breslow.
    b = cph.params_.to_numpy()
    xc = df[["x1", "x2"]].to_numpy() - df[["x1", "x2"]].to_numpy().mean(0)
    e = np.exp(xc @ b)
    order = np.argsort(df["T"].to_numpy())
    tt, ee, dd = df["T"].to_numpy()[order], e[order], df["E"].to_numpy()[order]
    risk = np.cumsum(ee[::-1])[::-1]
    alpha = np.where(dd == 1, (1 - ee / risk) ** (1 / ee), 1.0)
    kp = pd.Series(np.cumprod(alpha), index=tt)[dd == 1]
    kp_at = kp.reindex(times, method="nearest").to_numpy()
    np.testing.assert_allclose(s_mean[unique], kp_at[unique], atol=6e-4)
    np.testing.assert_allclose(s_mean[unique], ref_at[unique], atol=0.02)
    sch = cph.compute_residuals(df, "schoenfeld")
    pr1 = res.saved["PR1"][1]
    np.testing.assert_allclose(pr1.loc[sch.index], sch["x1"], atol=1e-4)
    # DfBeta = score residuals x covariance (one-step approximation of b - b(-i)); compare
    # with exact leave-one-out refits
    dfb = res.saved["DFB1"][1].to_numpy()
    full = sv.cox_fit(sv.CoxData(df["T"], df["E"], df[["x1", "x2"]].to_numpy()), [0, 1]).beta
    loo = []
    for i in range(40):
        m = np.ones(n, bool)
        m[i] = False
        d2 = sv.CoxData(df["T"][m], df["E"][m], df[["x1", "x2"]].to_numpy()[m])
        loo.append(full[0] - sv.cox_fit(d2, [0, 1]).beta[0])
    assert np.corrcoef(dfb[:40], loo)[0, 1] > 0.98
    assert np.max(np.abs(dfb[:40] - np.array(loo))) < 0.004
    xb = res.saved["XBE"][1]
    assert abs(xb.mean()) < 1e-10


def test_time_dependent_matches_lifelines_counting_process():
    lifelines = pytest.importorskip("lifelines")
    rng = np.random.default_rng(9)
    n = 120
    x = rng.normal(size=n)
    t = np.round(rng.exponential(1 / np.exp(0.6 * x)) * 50, 3) + 0.001
    c = np.round(rng.exponential(60, n), 3) + 0.001
    df = pd.DataFrame({"time": np.minimum(t, c), "d": (t <= c).astype(float), "x": x})
    ds = Dataset(df)
    res = sv.cox_regression(ds, "time", "d", ("single", 1.0), ["x", "T_COV_"],
                            time_dependent="x * LN(T_)")
    rows = _table(res.section, "Variables in the Equation").rows
    # reference: episodes split at every event time, fitted by lifelines (no tied event
    # times, so Efron == Breslow); risk set (start, stop]
    ep = sv.episode_split(df.rename(columns={"time": "t"}), "t", "d")
    ep["tc"] = ep["x"] * np.log(ep["_stop"])
    ep["start"] = np.where(np.isfinite(ep["_start"]), ep["_start"], 0.0)
    ctv = lifelines.CoxTimeVaryingFitter().fit(
        ep[["_case", "start", "_stop", "_event", "x", "tc"]], id_col="_case",
        start_col="start", stop_col="_stop", event_col="_event")
    np.testing.assert_allclose([float(r[2]) for r in rows], ctv.params_, atol=6e-4)
    # a time-fixed covariate expressed through T_COV_ gives the ordinary Cox model
    same = sv.cox_regression(ds, "time", "d", ("single", 1.0), ["T_COV_"],
                             time_dependent="x + 0 * T_")
    plain = sv.cox_regression(ds, "time", "d", ("single", 1.0), ["x"])
    assert _table(same.section, "Variables in the Equation").rows[0][2:4] == \
        _table(plain.section, "Variables in the Equation").rows[0][2:4]
