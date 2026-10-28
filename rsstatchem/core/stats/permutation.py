"""Permutation tests (SPSS 32 permutation extensions for compare means, correlation and
regression).

* Independent samples: group labels permuted; statistic = mean difference, Welch t or pooled t.
* Paired samples: signs of the differences flipped; statistic = mean difference or paired t.
* One-way: group labels permuted; statistic = the ANOVA F.
* Correlation: Y permuted against X; Pearson r or Spearman rho.
* Linear regression: Freedman–Lane permutation of the reduced-model residuals for each
  coefficient's t statistic.

The null distribution is enumerated exactly when the number of distinct permutations does
not exceed the limit (default 100,000); otherwise B random permutations (seeded MT19937) are
drawn and p = (count + 1)/(B + 1). Two-sided p = min(1, 2·min(p_less, p_greater)), with
a relative tolerance of 100·eps on ties — the conventions of ``scipy.stats.permutation_test``,
which the tests reproduce exactly.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
from scipy import stats

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import fmt, fmt_int, value_text
from rsstatchem.core.stats.ml_common import MLError, raw_column, rng, sort_levels
from rsstatchem.core.stats.roc import figure, png, style

PermError = MLError
EPS = np.finfo(float).eps * 100


def p_values(obs: float, null: np.ndarray, exact: bool) -> dict:
    gamma = abs(EPS * obs)
    ge = int((null >= obs - gamma).sum())
    le = int((null <= obs + gamma).sum())
    adj = 0 if exact else 1
    B = len(null)
    pg = (ge + adj) / (B + adj)
    pl = (le + adj) / (B + adj)
    return {"greater": pg, "less": pl, "two-sided": min(1.0, 2 * min(pg, pl)),
            "count_ge": ge, "B": B, "exact": exact}


# ----------------------------------------------------------------------------- statistics
def _mean_diff(x, y):
    return x.mean() - y.mean()


def _welch_t(x, y):
    return stats.ttest_ind(x, y, equal_var=False).statistic


def _pooled_t(x, y):
    return stats.ttest_ind(x, y, equal_var=True).statistic


TWO_SAMPLE = {"mean": ("Mean Difference", _mean_diff), "welch": ("Welch t", _welch_t),
              "pooled": ("t (equal variances)", _pooled_t)}


def independent_null(x: np.ndarray, y: np.ndarray, statistic: str = "mean",
                     n_perm: int = 10000, max_exact: int = 100000,
                     seed: int | None = None) -> tuple[float, np.ndarray, bool]:
    f = TWO_SAMPLE[statistic][1]
    z = np.r_[x, y]
    n, n1 = len(z), len(x)
    obs = float(f(x, y))
    if math.comb(n, n1) <= max_exact:
        null = []
        for idx in itertools.combinations(range(n), n1):
            m = np.zeros(n, bool)
            m[list(idx)] = True
            null.append(f(z[m], z[~m]))
        return obs, np.array(null, float), True
    g = rng(seed)
    null = np.empty(int(n_perm))
    for b in range(int(n_perm)):
        p = g.permutation(n)
        null[b] = f(z[p[:n1]], z[p[n1:]])
    return obs, null, False


def paired_null(d: np.ndarray, statistic: str = "mean", n_perm: int = 10000,
                max_exact: int = 100000, seed: int | None = None):
    def f(v):
        if statistic == "mean":
            return v.mean()
        s = v.std(ddof=1)
        return v.mean() / (s / math.sqrt(len(v))) if s > 0 else 0.0

    n = len(d)
    obs = float(f(d))
    if 2 ** n <= max_exact:
        signs = np.array(list(itertools.product([1.0, -1.0], repeat=n)))
        null = np.array([f(d * s) for s in signs])
        return obs, null, True
    g = rng(seed)
    null = np.array([f(d * np.where(g.random(n) < 0.5, -1.0, 1.0)) for _ in range(int(n_perm))])
    return obs, null, False


def oneway_null(groups: list[np.ndarray], n_perm: int = 10000, seed: int | None = None):
    z = np.concatenate(groups)
    sizes = np.cumsum([len(g) for g in groups])[:-1]
    obs = float(stats.f_oneway(*groups).statistic)
    g = rng(seed)
    null = np.empty(int(n_perm))
    for b in range(int(n_perm)):
        null[b] = stats.f_oneway(*np.split(z[g.permutation(len(z))], sizes)).statistic
    return obs, null, False


def correlation_null(x: np.ndarray, y: np.ndarray, method: str = "pearson",
                     n_perm: int = 10000, max_exact: int = 100000, seed: int | None = None):
    if method == "spearman":
        x, y = stats.rankdata(x), stats.rankdata(y)
    xc = (x - x.mean()) / np.linalg.norm(x - x.mean())
    yc = (y - y.mean()) / np.linalg.norm(y - y.mean())
    obs = float(xc @ yc)
    n = len(x)
    if math.factorial(n) <= max_exact:
        null = np.array([xc @ yc[list(p)] for p in itertools.permutations(range(n))])
        return obs, null, True
    g = rng(seed)
    null = np.array([xc @ yc[g.permutation(n)] for _ in range(int(n_perm))])
    return obs, null, False


def freedman_lane(y: np.ndarray, X: np.ndarray, n_perm: int = 5000,
                  seed: int | None = None) -> list[dict]:
    """t statistics and Freedman–Lane permutation p-values for each column of X (a constant
    is added)."""
    n, k = X.shape
    Xf = np.column_stack([np.ones(n), X])

    def tstats(yy, M):
        beta, *_ = np.linalg.lstsq(M, yy, rcond=None)
        r = yy - M @ beta
        df = len(yy) - M.shape[1]
        s2 = (r @ r) / df
        cov = s2 * np.linalg.pinv(M.T @ M)
        return beta, beta / np.sqrt(np.diag(cov))

    beta, t_obs = tstats(y, Xf)
    g = rng(seed)
    out = []
    for j in range(k):
        Xr = np.delete(Xf, j + 1, axis=1)
        br, *_ = np.linalg.lstsq(Xr, y, rcond=None)
        fit = Xr @ br
        res = y - fit
        null = np.empty(int(n_perm))
        for b in range(int(n_perm)):
            null[b] = tstats(fit + res[g.permutation(n)], Xf)[1][j + 1]
        pv = p_values(float(t_obs[j + 1]), null, False)
        out.append({"b": float(beta[j + 1]), "t": float(t_obs[j + 1]), **pv})
    return out


# ----------------------------------------------------------------------------- output
def _mc_ci(p: float, B: int, level: float = 0.99) -> tuple[float, float]:
    z = stats.norm.ppf(0.5 + level / 2)
    se = math.sqrt(max(p * (1 - p), 0) / B)
    return max(0.0, p - z * se), min(1.0, p + z * se)


def _null_plot(null, obs, title, xlabel) -> bytes:
    fig = figure(5.2, 3.6)
    ax = fig.add_subplot(111)
    ax.hist(null, bins=40, color="#9ecae1", edgecolor="white")
    ax.axvline(obs, color="#c0392b", lw=1.5, label="Observed")
    ax.legend(fontsize=8)
    style(ax, title, xlabel, "Frequency")
    return png(fig)


def _result_rows(label, obs, pv, extra=None):
    row = [label] + (extra or []) + [fmt(obs, 4), fmt_int(pv["B"]),
                                     "Exact" if pv["exact"] else "Monte Carlo",
                                     fmt(pv["two-sided"], 4), fmt(pv["less"], 4),
                                     fmt(pv["greater"], 4)]
    if not pv["exact"]:
        lo, hi = _mc_ci(pv["two-sided"], pv["B"])
        row += [fmt(lo, 4), fmt(hi, 4)]
    else:
        row += ["", ""]
    return row


COLS = ["Statistic", "Permutations", "Method", "Sig. (2-tailed)", "Sig. (less)",
        "Sig. (greater)", ("Monte Carlo 99% CI of Sig.", "Lower"),
        ("Monte Carlo 99% CI of Sig.", "Upper")]


def permutation_procedure(ds: Dataset, test: str, variables: list[str],
                          group: str | None = None, groups: tuple | None = None,
                          pairs: list[tuple[str, str]] | None = None,
                          statistic: str = "mean", method: str = "pearson",
                          n_perm: int = 10000, max_exact: int = 100000,
                          seed: int | None = None, plot: bool = True) -> Section:
    """test in independent | paired | oneway | correlation."""
    section = Section("Permutation Test")
    images: list[tuple[str, bytes]] = []
    footer = [f"Exact enumeration when there are at most {max_exact:,} distinct "
              f"permutations, otherwise {n_perm:,} random permutations"
              + (f" (seed {seed})" if seed is not None else "") + "; Monte Carlo "
              "p = (count + 1)/(B + 1)."]
    if test == "independent":
        if not group or not variables:
            raise PermError("Select test variables and a grouping variable.")
        g = raw_column(ds, group)
        lv = list(groups) if groups else sort_levels(g.dropna().tolist())[:2]
        if len(lv) != 2:
            raise PermError("The grouping variable needs two groups.")
        rows = []
        for v in variables:
            x = raw_column(ds, v)
            a = x[(g == lv[0]) & x.notna()].to_numpy(float)
            b = x[(g == lv[1]) & x.notna()].to_numpy(float)
            if len(a) < 2 or len(b) < 2:
                raise PermError(f"{v}: each group needs at least two cases.")
            obs, null, exact = independent_null(a, b, statistic, n_perm, max_exact, seed)
            pv = p_values(obs, null, exact)
            rows.append(_result_rows(v, obs, pv, [fmt(a.mean(), 4), fmt(b.mean(), 4)]))
            if plot:
                images.append((f"Permutation Distribution: {v}", _null_plot(
                    null, obs, f"Permutation Distribution: {v}", TWO_SAMPLE[statistic][0])))
        gl = [value_text(ds.var(group), x) for x in lv]
        section.log(f"PERMUTATION TEST /INDEPENDENT {' '.join(variables)} BY {group}"
                    f"({lv[0]} {lv[1]})\n  /STATISTIC={statistic.upper()} "
                    f"/SAMPLES={n_perm}.")
        section.table(Table(f"Independent-Samples Permutation Test ({TWO_SAMPLE[statistic][0]})",
                            ["", f"Mean ({gl[0]})", f"Mean ({gl[1]})"] + COLS, rows,
                            footnotes=footer))
    elif test == "paired":
        if not pairs:
            raise PermError("Select at least one pair of variables.")
        rows = []
        for a_, b_ in pairs:
            a, b = raw_column(ds, a_), raw_column(ds, b_)
            ok = a.notna() & b.notna()
            d = (a[ok] - b[ok]).to_numpy(float)
            if len(d) < 2:
                raise PermError("Too few valid pairs.")
            obs, null, exact = paired_null(d, "mean" if statistic == "mean" else "t", n_perm,
                                           max_exact, seed)
            pv = p_values(obs, null, exact)
            rows.append(_result_rows(f"{a_} - {b_}", obs, pv, [fmt_int(len(d))]))
            if plot:
                images.append((f"Permutation Distribution: {a_} - {b_}", _null_plot(
                    null, obs, f"Permutation Distribution: {a_} - {b_}",
                    "Mean Difference" if statistic == "mean" else "Paired t")))
        section.log("PERMUTATION TEST /PAIRED " + " ".join(f"{a}-{b}" for a, b in pairs)
                    + f"\n  /STATISTIC={statistic.upper()} /SAMPLES={n_perm}.")
        section.table(Table("Paired-Samples Permutation Test (sign flips)", ["", "N"] + COLS,
                            rows, footnotes=footer))
    elif test == "oneway":
        if not group or not variables:
            raise PermError("Select dependent variables and a factor.")
        g = raw_column(ds, group)
        rows = []
        for v in variables:
            x = raw_column(ds, v)
            ok = x.notna() & g.notna()
            levels = sort_levels(g[ok].tolist())
            gs = [x[ok & (g == lv)].to_numpy(float) for lv in levels]
            if len(gs) < 2:
                raise PermError("The factor needs at least two groups.")
            obs, null, exact = oneway_null(gs, n_perm, seed)
            pv = p_values(obs, null, exact)
            rows.append([v, fmt(obs, 4), fmt_int(pv["B"]), "Monte Carlo",
                         fmt(pv["greater"], 4)] + [fmt(c, 4) for c in
                                                   _mc_ci(pv["greater"], pv["B"])])
            if plot:
                images.append((f"Permutation Distribution: {v}", _null_plot(
                    null, obs, f"Permutation Distribution: {v}", "F")))
        section.log(f"PERMUTATION TEST /ONEWAY {' '.join(variables)} BY {group}"
                    f"\n  /SAMPLES={n_perm}.")
        section.table(Table("One-Way Permutation Test (F)", [
            "", "F", "Permutations", "Method", "Sig.", ("Monte Carlo 99% CI of Sig.", "Lower"),
            ("Monte Carlo 99% CI of Sig.", "Upper")], rows, footnotes=footer))
    elif test == "correlation":
        if len(variables) < 2:
            raise PermError("Select at least two variables.")
        rows = []
        for i, a_ in enumerate(variables):
            for b_ in variables[i + 1:]:
                a, b = raw_column(ds, a_), raw_column(ds, b_)
                ok = a.notna() & b.notna()
                x, y = a[ok].to_numpy(float), b[ok].to_numpy(float)
                if len(x) < 3:
                    raise PermError("Too few valid cases.")
                obs, null, exact = correlation_null(x, y, method, n_perm, max_exact, seed)
                pv = p_values(obs, null, exact)
                rows.append(_result_rows(f"{a_} with {b_}", obs, pv, [fmt_int(len(x))]))
        section.log(f"PERMUTATION TEST /CORRELATION {' '.join(variables)}"
                    f"\n  /METHOD={method.upper()} /SAMPLES={n_perm}.")
        section.table(Table(f"Correlation Permutation Test "
                            f"({'Pearson r' if method == 'pearson' else 'Spearman rho'})",
                            ["", "N"] + COLS, rows, footnotes=footer))
    else:
        raise PermError(f"Unknown test {test!r}.")
    for title, img in images:
        section.image(title, img, width=440)
    return section


def regression_permutation(ds: Dataset, dep: str, predictors: list[str], n_perm: int = 5000,
                           seed: int | None = None) -> Section:
    if not predictors:
        raise PermError("Select at least one predictor.")
    names = [dep] + list(predictors)
    M = np.column_stack([raw_column(ds, v).to_numpy(float) for v in names])
    ok = np.isfinite(M).all(1)
    if ok.sum() < len(predictors) + 3:
        raise PermError("Too few valid cases.")
    y, X = M[ok, 0], M[ok, 1:]
    res = freedman_lane(y, X, n_perm, seed)
    section = Section("Regression Permutation Test")
    section.log(f"PERMUTATION TEST /REGRESSION {dep} WITH {' '.join(predictors)}"
                f"\n  /METHOD=FREEDMANLANE /SAMPLES={n_perm}.")
    df = len(y) - len(predictors) - 1
    rows = []
    for v, r in zip(predictors, res):
        lo, hi = _mc_ci(r["two-sided"], r["B"])
        rows.append([v, fmt(r["b"], 4), fmt(r["t"], 3),
                     fmt(2 * stats.t.sf(abs(r["t"]), df), 4), fmt(r["two-sided"], 4),
                     fmt(lo, 4), fmt(hi, 4)])
    section.table(Table("Coefficients (Freedman-Lane Permutation)", [
        "", "B", "t", "Sig. (parametric)", "Permutation Sig. (2-tailed)",
        ("Monte Carlo 99% CI of Sig.", "Lower"), ("Monte Carlo 99% CI of Sig.", "Upper")],
        rows, footnotes=[f"Dependent variable: {dep}. N = {len(y)}; {n_perm:,} permutations "
                         "of the reduced-model residuals per coefficient."]))
    section.result = res  # type: ignore[attr-defined]
    return section


def selftest() -> None:
    obs, null, exact = independent_null(np.array([1.0, 2, 3]), np.array([4.0, 5, 6]))
    p = p_values(obs, null, exact)
    assert exact and abs(p["two-sided"] - 0.1) < 1e-12
