"""Post hoc multiple comparisons beyond the six classic ones, as offered by SPSS ONEWAY / GLM.

Equal variances assumed: R-E-G-W F, R-E-G-W Q, S-N-K, Tukey's-b, Duncan, Hochberg's GT2,
Gabriel, Waller-Duncan and Dunnett (two-sided, < control, > control; control first or last).
Equal variances not assumed: Tamhane's T2, Dunnett's T3 and Dunnett's C.

Two kinds of output, as in SPSS:

* pairwise rows for the "Multiple Comparisons" table (LSD, Bonferroni, Sidak, Scheffe, Tukey,
  GT2, Gabriel, Dunnett, T2, T3, Games-Howell, C) — :func:`pairwise`;
* "Homogeneous Subsets" for the range tests (S-N-K, Tukey, Tukey-b, Duncan, Scheffe, REGW F,
  REGW Q, GT2, Gabriel, Waller-Duncan) — :func:`homogeneous_subsets`, using the harmonic mean
  of the group sizes like SPSS, except R-E-G-W F and Q, which use the actual n_i and n_j
  (IBM SPSS Statistics Algorithms, Post Hoc Tests: "For homogeneous subsets, the n_i and n_j
  are used for the R-E-G-W F test and the R-E-G-W Q test").

The studentized maximum modulus distribution (GT2, Gabriel, T3) and Dunnett's multivariate t
(one-factor correlation structure) are evaluated by deterministic quadrature, so results are
reproducible. Everything here is a pure function of (group summaries, error term, options).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import integrate, optimize, special, stats

from rsstatchem.core.output import Table
from rsstatchem.core.stats.common import Group, fmt, fmt_int, fmt_p

# key -> label in the Multiple Comparisons / Homogeneous Subsets tables (SPSS wording)
LABELS = {
    "lsd": "LSD", "bonferroni": "Bonferroni", "sidak": "Sidak", "scheffe": "Scheffe",
    "regwf": "Ryan-Einot-Gabriel-Welsch F", "regwq": "Ryan-Einot-Gabriel-Welsch Range",
    "snk": "Student-Newman-Keuls", "tukey": "Tukey HSD", "tukeyb": "Tukey B",
    "duncan": "Duncan", "gt2": "Hochberg", "gabriel": "Gabriel", "waller": "Waller-Duncan",
    "dunnett": "Dunnett t", "t2": "Tamhane", "t3": "Dunnett T3",
    "games_howell": "Games-Howell", "c": "Dunnett C",
}
# Dialog check-box texts and order (SPSS Post Hoc Multiple Comparisons dialog)
EQUAL = (("lsd", "LSD"), ("bonferroni", "Bonferroni"), ("sidak", "Sidak"),
         ("scheffe", "Scheffe"), ("regwf", "R-E-G-W F"), ("regwq", "R-E-G-W Q"),
         ("snk", "S-N-K"), ("tukey", "Tukey"), ("tukeyb", "Tukey's-b"), ("duncan", "Duncan"),
         ("gt2", "Hochberg's GT2"), ("gabriel", "Gabriel"), ("waller", "Waller-Duncan"),
         ("dunnett", "Dunnett"))
UNEQUAL = (("t2", "Tamhane's T2"), ("t3", "Dunnett's T3"), ("games_howell", "Games-Howell"),
           ("c", "Dunnett's C"))
ORDER = tuple(k for k, _ in EQUAL + UNEQUAL)
PAIRWISE = ("tukey", "scheffe", "lsd", "bonferroni", "sidak", "gt2", "gabriel", "dunnett",
            "t2", "t3", "games_howell", "c")
SUBSETS = ("snk", "tukey", "tukeyb", "duncan", "scheffe", "regwf", "regwq", "gt2", "gabriel",
           "waller")
UNEQUAL_KEYS = ("t2", "t3", "games_howell", "c")


@dataclass
class PostHocOptions:
    """Options that change the computations (besides the method list)."""
    level: float = 0.95
    control: str = "last"          # Dunnett control category: "last" | "first"
    dunnett_test: str = "2-sided"  # "2-sided" | "<" | ">"
    k_ratio: float = 100.0         # Waller-Duncan type I/type II error seriousness ratio


# ------------------------------------------------------------------ distributions
@lru_cache(maxsize=64)
def _s_nodes(df: float, n: int = 96) -> tuple[np.ndarray, np.ndarray]:
    """Quadrature nodes/weights for S = sqrt(chi2_df / df) (E[g(S)] = sum w g(s))."""
    if not math.isfinite(df) or df > 1e6:
        return np.array([1.0]), np.array([1.0])
    u, w = special.roots_legendre(n)
    u = (u + 1) / 2
    w = w / 2
    s = np.sqrt(stats.chi2.ppf(u, df) / df)
    return s, w


def smm_cdf(m: float, k: int, df: float) -> float:
    """CDF of the studentized maximum modulus: P(max_i |Z_i| / S <= m), i = 1..k."""
    if m <= 0:
        return 0.0
    if not math.isfinite(df) or df > 1e6:
        return float((2 * stats.norm.cdf(m) - 1) ** k)

    def integrand(s):
        return (2 * stats.norm.cdf(m * s) - 1) ** k * \
            stats.chi2.pdf(df * s * s, df) * 2 * df * s

    val, _ = integrate.quad(integrand, 0, np.inf, limit=200, epsabs=1e-11)
    return float(min(max(val, 0.0), 1.0))


def smm_sf(m: float, k: int, df: float) -> float:
    return float(min(max(1 - smm_cdf(m, k, df), 0.0), 1.0))


def smm_ppf(level: float, k: int, df: float) -> float:
    lo, hi = 1e-6, 50.0
    return float(optimize.brentq(lambda m: smm_cdf(m, k, df) - level, lo, hi, xtol=1e-10))


_HERM_X, _HERM_W = special.roots_hermitenorm(64)
_HERM_W = _HERM_W / _HERM_W.sum()


def dunnett_cdf(c: float, lams: np.ndarray, df: float, two_sided: bool = True) -> float:
    """P(all |T_i| < c) (two-sided) or P(all T_i < c) for Dunnett's statistics with
    correlations lambda_i * lambda_j, by Gauss-Hermite (normal) x Gauss-Legendre (chi)."""
    lams = np.asarray(lams, float)
    r = np.sqrt(np.clip(1 - lams ** 2, 1e-15, None))
    s, ws = _s_nodes(df)
    w = _HERM_X[:, None, None]          # normal common factor
    cs = c * s[None, :, None]
    lw = lams[None, None, :] * w
    if two_sided:
        prob = stats.norm.cdf((cs - lw) / r) - stats.norm.cdf((-cs - lw) / r)
    else:
        prob = stats.norm.cdf((cs - lw) / r)
    prod = np.prod(prob, axis=2)        # (nherm, ns)
    return float(_HERM_W @ prod @ ws)


def dunnett_crit(level: float, lams: np.ndarray, df: float, two_sided: bool = True) -> float:
    return float(optimize.brentq(lambda c: dunnett_cdf(c, lams, df, two_sided) - level,
                                 1e-6, 50.0, xtol=1e-10))


def studentized_range_sf(q: float, k: int, df: float) -> float:
    if k < 2 or not math.isfinite(q):
        return float("nan")
    return float(stats.studentized_range.sf(q, k, df))


def studentized_range_ppf(level: float, k: int, df: float) -> float:
    return float(stats.studentized_range.ppf(level, k, df))


# ------------------------------------------------------------------ pairwise
def _desc(groups: list[Group]):
    n = np.array([g.n for g in groups], float)
    m = np.array([g.data.mean() if g.n else np.nan for g in groups])
    v = np.array([g.data.var(ddof=1) if g.n > 1 else np.nan for g in groups])
    return n, m, v


def control_index(k: int, control: str) -> int:
    return 0 if control == "first" else k - 1


def pairwise(groups: list[Group], method: str, mse: float, df_error: float,
             opts: PostHocOptions | None = None) -> list[tuple]:
    """Rows (i, j, mean diff, SE, p, lower, upper) for the Multiple Comparisons table.

    ``p`` is NaN for Dunnett's C (SPSS prints no significance for it); one-sided Dunnett
    tests have one infinite confidence bound.
    """
    opts = opts or PostHocOptions()
    level = opts.level
    alpha = 1 - level
    n, m, v = _desc(groups)
    k = len(groups)
    npairs = k * (k - 1) / 2
    out = []
    if method == "dunnett":
        ci = control_index(k, opts.control)
        treat = [i for i in range(k) if i != ci]
        lams = np.sqrt(n[treat] / (n[treat] + n[ci]))
        two = opts.dunnett_test == "2-sided"
        crit = dunnett_crit(level, lams, df_error, two)
        for i in treat:
            diff = m[i] - m[ci]
            se = math.sqrt(mse * (1 / n[i] + 1 / n[ci]))
            t = diff / se
            if two:
                p = 1 - dunnett_cdf(abs(t), lams, df_error, True)
                lo, hi = diff - crit * se, diff + crit * se
            elif opts.dunnett_test == ">":
                p = 1 - dunnett_cdf(t, lams, df_error, False)
                lo, hi = diff - crit * se, math.inf
            else:
                p = 1 - dunnett_cdf(-t, lams, df_error, False)
                lo, hi = -math.inf, diff + crit * se
            out.append((i, ci, diff, se, float(min(max(p, 0.0), 1.0)), lo, hi))
        return out
    if method in ("gt2", "gabriel"):
        kstar = int(npairs)
        crit = smm_ppf(level, kstar, df_error)
    for i in range(k):
        for j in range(k):
            if i == j or n[i] == 0 or n[j] == 0:
                continue
            diff = m[i] - m[j]
            if method in UNEQUAL_KEYS:
                vi, vj = v[i] / n[i], v[j] / n[j]
                se = math.sqrt(vi + vj)
                dfw = (vi + vj) ** 2 / (vi ** 2 / (n[i] - 1) + vj ** 2 / (n[j] - 1))
                t = diff / se if se > 0 else float("nan")
                if method == "t2":
                    pt = 2 * stats.t.sf(abs(t), dfw)
                    p = 1 - (1 - pt) ** npairs
                    a_s = 1 - (1 - alpha) ** (1 / npairs)
                    half = stats.t.ppf(1 - a_s / 2, dfw) * se
                elif method == "t3":
                    p = smm_sf(abs(t), int(npairs), dfw)
                    half = smm_ppf(level, int(npairs), dfw) * se
                elif method == "games_howell":
                    p = studentized_range_sf(abs(t) * math.sqrt(2), k, dfw)
                    half = studentized_range_ppf(level, k, dfw) / math.sqrt(2) * se
                else:  # Dunnett's C
                    qi = studentized_range_ppf(level, k, n[i] - 1)
                    qj = studentized_range_ppf(level, k, n[j] - 1)
                    half = (qi * vi + qj * vj) / math.sqrt(vi + vj) / math.sqrt(2)
                    p = float("nan")
                out.append((i, j, diff, se, float(min(max(p, 0.0), 1.0)) if p == p else p,
                            diff - half, diff + half))
                continue
            se = math.sqrt(mse * (1 / n[i] + 1 / n[j]))
            t = diff / se if se > 0 else float("nan")
            if method == "gt2":
                p, half = smm_sf(abs(t), kstar, df_error), crit * se
            elif method == "gabriel":
                scale = math.sqrt(mse) * (1 / math.sqrt(2 * n[i]) + 1 / math.sqrt(2 * n[j]))
                p, half = smm_sf(abs(diff) / scale, kstar, df_error), crit * scale
            else:
                from rsstatchem.core.stats.anova import posthoc as classic

                return classic(groups, method, mse, df_error, level)
            out.append((i, j, diff, se, float(min(max(p, 0.0), 1.0)), diff - half, diff + half))
    return out


# ------------------------------------------------------------------ homogeneous subsets
def harmonic_n(groups: list[Group]) -> float:
    n = np.array([g.n for g in groups if g.n > 0], float)
    return float(len(n) / np.sum(1 / n)) if len(n) else float("nan")


def _alpha_p(method: str, p: int, k: int, alpha: float) -> float:
    if method == "duncan":
        return 1 - (1 - alpha) ** (p - 1)
    if method in ("regwq", "regwf"):
        return 1 - (1 - alpha) ** (p / k) if p < k - 1 else alpha
    return alpha


def _sig_from_alpha_p(method: str, p_raw: float, p: int, k: int) -> float:
    """Invert alpha_p(alpha) so that the printed Sig. is comparable to alpha."""
    if method == "duncan" and p > 1:
        return 1 - (1 - p_raw) ** (1 / (p - 1))
    if method in ("regwq", "regwf") and p < k - 1:
        return 1 - (1 - p_raw) ** (k / p)
    return p_raw


@lru_cache(maxsize=256)
def waller_t(k_ratio: float, F: float, q: int, f: float) -> float:
    """Waller-Duncan Bayesian critical t, t_B(K, F, q, f), for the k-ratio t test.

    Solves K = I(t) / I(-t) with I(+-t) = int_1^inf sqrt(x-1) g(x) h(+-t b(x)) dx, where
    g(x) = x^(-(q+3)/2) (f + qF/x)^(-(f+q-1)/2), b(x) = sqrt((f+q)(x-1) / (f x + q F)) and
    h(z) = (nu + z^2)/(nu - 1) dt_nu(z) + z pt_nu(z), nu = q + f (Waller & Duncan 1969;
    the same integral representation as R agricolae's ``waller``). F <= 1 means the data
    give no evidence of treatment differences; t_B is then effectively infinite."""
    if not (F > 1) or q < 1 or f < 1:
        return math.inf
    f = min(float(f), 1000.0)
    F = min(float(F), 1e5)  # t_B is flat beyond this; avoids quadrature underflow
    nu = q + f

    def h(z):
        return (nu + z * z) / (nu - 1) * stats.t.pdf(z, nu) + z * stats.t.cdf(z, nu)

    def integral(t):
        def fn(x):
            # g scaled by f^((f+q-1)/2) so the integrals are O(1); the ratio is unchanged
            g = x ** (-(q + 3) / 2) * (1 + q * F / (f * x)) ** (-(f + q - 1) / 2)
            b = math.sqrt((f + q) * (x - 1) / (f * x + q * F))
            return math.sqrt(x - 1) * g * h(t * b)
        c = 1 + q * F / f  # the mass sits around x ~ qF/f: split the range there
        return (integrate.quad(fn, 1, c, limit=400, epsabs=0, epsrel=1e-10)[0] +
                integrate.quad(fn, c, np.inf, limit=400, epsabs=0, epsrel=1e-10)[0])

    def ratio(t):
        den = integral(-t)
        return integral(t) / den if den > 0 else 1e300

    try:
        return float(optimize.brentq(lambda t: ratio(t) - k_ratio, 0.05, 15.0, xtol=1e-6))
    except ValueError:
        return math.inf


def homogeneous_subsets(groups: list[Group], method: str, mse: float, df_error: float,
                        opts: PostHocOptions | None = None, anova_f: float | None = None
                        ) -> tuple[list[int], list[list[int]], list[float]]:
    """(order of groups by ascending mean, subsets as lists of positions in that order,
    Sig. per subset). Uses the harmonic mean group size as SPSS does."""
    opts = opts or PostHocOptions()
    alpha = 1 - opts.level
    groups_ok = [i for i, g in enumerate(groups) if g.n > 0]
    means = np.array([groups[i].data.mean() for i in groups_ok])
    order_local = np.argsort(means, kind="stable")
    order = [groups_ok[i] for i in order_local]
    ms = means[order_local]
    ns_ord = np.array([groups[i].n for i in order], float)
    k = len(ms)
    nh = harmonic_n([groups[i] for i in order])
    se1 = math.sqrt(mse / nh)
    kstar = k * (k - 1) // 2

    wd_t = None
    if method == "waller":
        wd_t = waller_t(opts.k_ratio, anova_f if anova_f is not None else math.nan, k - 1,
                        df_error)

    def test(i: int, j: int) -> tuple[bool, float]:
        """Is the range of positions i..j significant? Returns (significant, Sig.)."""
        p = j - i + 1
        rng = ms[j] - ms[i]
        if p < 2:
            return False, 1.0
        q = rng / se1
        if method == "regwf":
            # IBM FREGW: [sum n x^2 - (sum n x)^2 / sum n] / ((r - 1) s^2) over the r means
            sub, nsub = ms[i:j + 1], ns_ord[i:j + 1]
            F = float(np.sum(nsub * sub ** 2) - np.sum(nsub * sub) ** 2 / nsub.sum()) \
                / (p - 1) / mse
            raw = float(stats.f.sf(F, p - 1, df_error))
            a_p = _alpha_p(method, p, k, alpha)
            return raw < a_p, _sig_from_alpha_p(method, raw, p, k)
        if method == "regwq":
            # IBM QREGW: max over pairs in the range of |x_i - x_j| sqrt(min(n_i, n_j)) / s
            sub, nsub = ms[i:j + 1], ns_ord[i:j + 1]
            qq = max(abs(sub[a] - sub[b]) * math.sqrt(min(nsub[a], nsub[b]))
                     for a in range(p) for b in range(a + 1, p)) / math.sqrt(mse)
            raw = studentized_range_sf(qq, p, df_error)
            a_p = _alpha_p(method, p, k, alpha)
            return raw < a_p, _sig_from_alpha_p(method, raw, p, k)
        if method in ("snk", "duncan"):
            raw = studentized_range_sf(q, p, df_error)
            a_p = _alpha_p(method, p, k, alpha)
            return raw < a_p, _sig_from_alpha_p(method, raw, p, k)
        if method == "tukey":
            raw = studentized_range_sf(q, k, df_error)
            return raw < alpha, raw
        if method == "tukeyb":
            # critical value = mean of q(alpha; k) and q(alpha; p). Solve for the alpha at
            # which it equals q using survival functions only: u = q_k, 2q - u = q_p.
            if p == k:
                sig = studentized_range_sf(q, k, df_error)
            else:
                def g(u):
                    return studentized_range_sf(u, k, df_error) - \
                        studentized_range_sf(2 * q - u, p, df_error)
                try:
                    u = float(optimize.brentq(g, q, 2 * q, xtol=1e-7))
                    sig = studentized_range_sf(u, k, df_error)
                except ValueError:
                    sig = 1.0
            return sig < alpha, sig
        if method == "scheffe":
            F = (q * q / 2) / (k - 1)
            raw = float(stats.f.sf(F, k - 1, df_error))
            return raw < alpha, raw
        if method in ("gt2", "gabriel"):
            raw = smm_sf(q / math.sqrt(2), kstar, df_error)
            return raw < alpha, raw
        if method == "waller":
            msd = wd_t * math.sqrt(2 * mse / nh)
            return rng >= msd, float("nan")
        raise ValueError(method)

    sig = {}
    tested = {}
    for size in range(k, 1, -1):
        for i in range(0, k - size + 1):
            tested[(i, i + size - 1)] = test(i, i + size - 1)
    declared = {}
    for size in range(k, 0, -1):
        for i in range(0, k - size + 1):
            j = i + size - 1
            s_ij = tested.get((i, j), (False, 1.0))[0]
            parents = [(i - 1, j)] if i > 0 else []
            parents += [(i, j + 1)] if j < k - 1 else []
            declared[(i, j)] = s_ij and all(declared[pp] for pp in parents)
    subsets = []
    for size in range(k, 0, -1):
        for i in range(0, k - size + 1):
            j = i + size - 1
            if declared[(i, j)]:
                continue
            if (i > 0 and not declared[(i - 1, j)]) or (j < k - 1 and not declared[(i, j + 1)]):
                continue
            subsets.append((i, j))
    subsets.sort()
    sets = [list(range(i, j + 1)) for i, j in subsets]
    sigs = [tested.get((i, j), (False, 1.0))[1] if j > i else 1.0 for i, j in subsets]
    sig.update({})
    return order, sets, sigs


def subsets_table(dep_title: str, factor_title: str, groups: list[Group],
                  methods: list[str], mse: float, df_error: float,
                  opts: PostHocOptions | None = None, anova_f: float | None = None,
                  decimals: int = 2, glm_note: str | None = None) -> Table | None:
    """SPSS "Homogeneous Subsets" table for one dependent variable."""
    opts = opts or PostHocOptions()
    methods = [m for m in SUBSETS if m in methods]
    if not methods:
        return None
    results = []
    max_sets = 1
    for meth in methods:
        res = homogeneous_subsets(groups, meth, mse, df_error, opts, anova_f)
        results.append((meth, res))
        max_sets = max(max_sets, len(res[1]))
    rows = []
    unequal = len({g.n for g in groups if g.n > 0}) > 1
    nh = harmonic_n(groups)
    marks = {}
    letters = iter("abcdefgh")
    l_nh = next(letters)
    l_uneq = next(letters) if unequal else None
    for meth, (order, sets, sigs) in results:
        mark = l_nh + ("," + l_uneq if l_uneq else "")
        if meth == "waller":
            marks.setdefault("waller", next(letters))
            mark = mark + "," + marks["waller"]
        label = LABELS[meth] + "".join(_sup(c) for c in mark.split(","))
        first = True
        for pos, gi in enumerate(order):
            g = groups[gi]
            cells = [label if first else "", g.label, fmt_int(g.n)]
            for s in range(max_sets):
                cells.append(fmt(g.data.mean(), decimals + 2)
                             if s < len(sets) and pos in sets[s] else "")
            rows.append(cells)
            first = False
        if meth != "waller":  # the Bayesian k-ratio test has no significance level
            sig_cells = [fmt_p(sigs[s]) if s < len(sets) and sigs[s] == sigs[s] else ""
                         for s in range(max_sets)]
            rows.append(["", "Sig.", ""] + sig_cells)
    head = f"Subset for alpha = {1 - opts.level:.2f}"
    cols = ["", factor_title, "N"] + [(head, str(s + 1)) for s in range(max_sets)]
    notes = ["Means for groups in homogeneous subsets are displayed."]
    if glm_note:
        notes.append(glm_note)
    notes.append(f"{l_nh}. Uses Harmonic Mean Sample Size = {fmt(nh)}.")
    if l_uneq:
        notes.append(f"{l_uneq}. The group sizes are unequal. The harmonic mean of the group "
                     "sizes is used. Type I error levels are not guaranteed.")
    if "waller" in marks:
        notes.append(f"{marks['waller']}. Type 1/Type 2 Error Seriousness Ratio = "
                     f"{opts.k_ratio:g}.")
    return Table(dep_title, cols, rows, row_header_cols=3, footnotes=notes)


def _sup(letter: str) -> str:
    from rsstatchem.core.stats.common import sup

    return sup(letter)


def method_label(method: str, opts: PostHocOptions | None = None) -> str:
    opts = opts or PostHocOptions()
    if method == "dunnett":
        side = {"2-sided": "2-sided", "<": "<control", ">": ">control"}[opts.dunnett_test]
        return f"Dunnett t ({side})"
    return LABELS.get(method, method)


__all__ = ["LABELS", "EQUAL", "UNEQUAL", "ORDER", "PAIRWISE", "SUBSETS", "PostHocOptions",
           "pairwise", "homogeneous_subsets", "subsets_table", "smm_cdf", "smm_ppf",
           "dunnett_cdf", "dunnett_crit", "waller_t", "harmonic_n", "method_label"]
