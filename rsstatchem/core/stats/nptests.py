"""Nonparametric test statistics (pure functions), exact / Monte Carlo significance, and the
SPSS *Analyze > Nonparametric Tests > One Sample / Independent Samples / Related Samples*
procedures (NPTESTS) with their "Hypothesis Test Summary" model-viewer style output.

The pure functions take NumPy arrays and return plain dicts; the legacy NPAR TESTS output in
:mod:`rsstatchem.core.stats.nonparametric` is built on them as well.

Exact and Monte Carlo p-values (the SPSS *Exact Tests* option)
---------------------------------------------------------------
Every ``exact_*`` function has the signature ``exact_xxx(<data>, method="exact",
samples=MC_SAMPLES, level=MC_LEVEL, seed=MC_SEED, limit=EXACT_LIMIT)`` and returns a
:class:`PValue`. ``method="exact"`` computes the exact permutation (conditional) p-value when
the reference set is small enough (``limit`` permutations / DP cells) and otherwise falls back
to Monte Carlo; ``method="mc"`` always samples. As in SPSS the Monte Carlo p is the proportion
of sampled tables whose statistic is at least as extreme as the observed one, with the normal
confidence interval ``p ± z·sqrt(p(1-p)/M)`` at ``level`` (SPSS defaults: 10000 samples, 99%
CI, seed 2000000). Two-sided p-values use ``|T - E(T)| >= |t - E(T)|``.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import special, stats

MC_SAMPLES = 10000
MC_LEVEL = 0.99
MC_SEED = 2000000
EXACT_LIMIT = 20000
_EPS = 1e-7


# ============================================================================ p-value object
@dataclass
class PValue:
    p: float
    method: str = "exact"           # "exact" | "monte carlo" | "asymptotic"
    lower: float | None = None      # Monte Carlo CI
    upper: float | None = None
    samples: int = 0
    level: float = MC_LEVEL

    def __float__(self) -> float:
        return float(self.p)


def mc_interval(p: float, m: int, level: float = MC_LEVEL) -> tuple[float, float]:
    """SPSS Monte Carlo confidence interval for a p-value estimated from ``m`` samples."""
    z = stats.norm.ppf(0.5 + level / 2)
    half = z * math.sqrt(max(p * (1 - p), 0.0) / m) if m else float("nan")
    return max(0.0, p - half), min(1.0, p + half)


def _mc(extreme: np.ndarray, samples: int, level: float) -> PValue:
    p = float(np.count_nonzero(extreme)) / samples
    lo, hi = mc_interval(p, samples, level)
    return PValue(p, "monte carlo", lo, hi, samples, level)


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def _chunks(total: int, size: int = 2000):
    done = 0
    while done < total:
        step = min(size, total - done)
        yield step
        done += step


# ============================================================================ helpers
def tie_term(values: np.ndarray) -> float:
    """Σ (t³ - t) over tie groups."""
    _, t = np.unique(np.asarray(values, float), return_counts=True)
    return float(((t.astype(float) ** 3) - t).sum())


def haverage(x: np.ndarray, q) -> np.ndarray:
    return np.percentile(np.asarray(x, float), q, method="weibull")


def _norm2(z: float) -> float:
    return float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else float("nan")


# ============================================================================ one sample
def binomial_test(k: int, n: int, prop: float = 0.5) -> dict:
    """SPSS binomial test: exact two-sided when the test proportion is .5, otherwise the exact
    one-sided p in the direction of the observed proportion."""
    if n == 0:
        return {"k": k, "n": n, "p": float("nan"), "two_sided": True}
    two = abs(prop - 0.5) < 1e-12
    if two:
        p = stats.binomtest(k, n, prop, alternative="two-sided").pvalue
    else:
        p = stats.binomtest(k, n, prop, alternative="less" if k / n < prop else
                            "greater").pvalue
    se = math.sqrt(n * prop * (1 - prop))
    return {"k": k, "n": n, "p": float(p), "two_sided": two, "se": se,
            "z": (k - n * prop) / se if se else float("nan")}


def chisq_gof(observed, expected_weights=None) -> dict:
    obs = np.asarray(observed, float)
    w = np.ones(len(obs)) if expected_weights is None else np.asarray(expected_weights, float)
    exp = obs.sum() * w / w.sum()
    chi = float(((obs - exp) ** 2 / exp).sum()) if len(obs) > 1 else float("nan")
    df = len(obs) - 1
    return {"chi2": chi, "df": df, "p": float(stats.chi2.sf(chi, df)) if df > 0 else np.nan,
            "expected": exp, "small": int((exp < 5).sum())}


def runs_test(x: np.ndarray, value: float) -> dict:
    """Runs above/below ``value`` (cases ``>= value`` count as above). SPSS uses a continuity
    correction when there are fewer than 50 cases."""
    x = np.asarray(x, float)
    above = x >= value
    n1, n2 = int((~above).sum()), int(above.sum())
    n = n1 + n2
    runs = int(1 + np.count_nonzero(above[1:] != above[:-1])) if n else 0
    out = {"value": value, "below": n1, "above": n2, "n": n, "runs": runs,
           "z": float("nan"), "p": float("nan"), "mean": float("nan"), "sd": float("nan")}
    if n1 == 0 or n2 == 0 or n < 3:
        return out
    mu = 2.0 * n1 * n2 / n + 1
    var = 2.0 * n1 * n2 * (2.0 * n1 * n2 - n) / (n * n * (n - 1))
    diff = runs - mu
    if n < 50:
        diff = 0.0 if abs(diff) < 0.5 else diff - 0.5 * math.copysign(1, diff)
    z = diff / math.sqrt(var)
    out.update(z=z, p=_norm2(z), mean=mu, sd=math.sqrt(var))
    return out


def runs_pmf(n1: int, n2: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact null distribution of the number of runs for n1 and n2 symbols."""
    n = n1 + n2
    total = special.comb(n, n1, exact=True)
    rs, ps = [], []
    for r in range(2, n + 1):
        k, odd = divmod(r, 2)
        if odd:
            c = special.comb(n1 - 1, k, exact=True) * special.comb(n2 - 1, k - 1, exact=True) + \
                special.comb(n1 - 1, k - 1, exact=True) * special.comb(n2 - 1, k, exact=True)
        else:
            c = 2 * special.comb(n1 - 1, k - 1, exact=True) * \
                special.comb(n2 - 1, k - 1, exact=True)
        if c:
            rs.append(r)
            ps.append(c / total)
    return np.array(rs, float), np.array(ps, float)


def ks_one_sample_test(x: np.ndarray, dist: str = "normal", params: dict | None = None,
                       lilliefors: bool = True) -> dict:
    """One-sample K-S against Normal / Uniform / Poisson / Exponential with SPSS parameter
    estimates (mean & SD; min & max; mean; mean) unless ``params`` are given."""
    x = np.sort(np.asarray(x, float))
    n = len(x)
    out = {"n": n, "dist": dist}
    if n < 1:
        return out
    params = dict(params or {})
    if dist == "normal":
        params.setdefault("mean", float(x.mean()))
        params.setdefault("sd", float(x.std(ddof=1)) if n > 1 else float("nan"))
        ok = params["sd"] > 0
        cdf = lambda v: stats.norm.cdf(v, params["mean"], params["sd"])  # noqa: E731
    elif dist == "uniform":
        params.setdefault("min", float(x.min()))
        params.setdefault("max", float(x.max()))
        ok = params["max"] > params["min"]
        cdf = lambda v: stats.uniform.cdf(v, params["min"],  # noqa: E731
                                          params["max"] - params["min"])
    elif dist == "exponential":
        params.setdefault("mean", float(x.mean()))
        ok = params["mean"] > 0
        cdf = lambda v: stats.expon.cdf(v, scale=params["mean"])  # noqa: E731
    elif dist == "poisson":
        params.setdefault("mean", float(x.mean()))
        ok = params["mean"] > 0
        cdf = None
    else:
        raise ValueError(dist)
    out["params"] = params
    if not ok or (n > 1 and np.ptp(x) == 0 and dist != "poisson"):
        return out
    if dist == "poisson":
        grid = np.arange(0, max(int(np.ceil(x.max())), 0) + 1, dtype=float)
        s = np.searchsorted(x, grid, side="right") / n
        f = stats.poisson.cdf(grid, params["mean"])
        d_pos = float(np.max(s - f))
        d_neg = float(np.max(np.concatenate([f - s, [0.0]])))
        out["outside"] = int((x < 0).sum() + (x != np.round(x)).sum())
    else:
        values, counts = np.unique(x, return_counts=True)
        upper = np.cumsum(counts) / n
        lower = upper - counts / n
        f = cdf(values)
        d_pos = float(np.max(upper - f))
        d_neg = float(np.max(f - lower))
        if dist == "exponential":
            out["outside"] = int((x < 0).sum())
        elif dist == "uniform":
            out["outside"] = int(((x < params["min"]) | (x > params["max"])).sum())
    d = max(d_pos, d_neg)
    z = math.sqrt(n) * d
    out.update(positive=d_pos, negative=-d_neg, absolute=d, statistic=d, z=z,
               p_kolmogorov=float(stats.kstwobign.sf(z)),
               p_exact=float(stats.kstwo.sf(d, n)) if dist != "poisson" else float("nan"))
    out["p"] = out["p_kolmogorov"]
    if dist == "normal" and lilliefors and "mean" in params:
        from rsstatchem.core.stats.common import lilliefors as _lf

        dl, pl, lb = _lf(x)
        if np.isfinite(pl):
            out.update(p=pl, lower_bound=lb, lilliefors=True)
    return out


def wilcoxon_test(d: np.ndarray) -> dict:
    """Wilcoxon signed-rank on differences (zeros dropped), SPSS normal approximation with
    tie-corrected variance and no continuity correction."""
    d = np.asarray(d, float)
    neg, pos, ties = int((d < 0).sum()), int((d > 0).sum()), int((d == 0).sum())
    nz = d[d != 0]
    ranks = stats.rankdata(np.abs(nz)) if len(nz) else np.array([])
    r_neg, r_pos = float(ranks[nz < 0].sum()), float(ranks[nz > 0].sum())
    n = len(nz)
    out = {"neg": neg, "pos": pos, "ties": ties, "r_neg": r_neg, "r_pos": r_pos, "n": n,
           "total": len(d), "z": float("nan"), "p": float("nan")}
    if n == 0:
        return out
    mean = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0 - tie_term(np.abs(nz)) / 48.0
    t = min(r_neg, r_pos)
    z = (t - mean) / math.sqrt(var) if var > 0 else float("nan")
    zpos = (r_pos - mean) / math.sqrt(var) if var > 0 else float("nan")
    out.update(t=t, mean=mean, se=math.sqrt(var) if var > 0 else np.nan, z=z, z_pos=zpos,
               p=_norm2(z), basis="negative" if r_neg <= r_pos else "positive")
    return out


def sign_test(d: np.ndarray) -> dict:
    d = np.asarray(d, float)
    neg, pos, ties = int((d < 0).sum()), int((d > 0).sum()), int((d == 0).sum())
    n = neg + pos
    out = {"neg": neg, "pos": pos, "ties": ties, "n": n, "total": len(d)}
    if n == 0:
        out.update(p=float("nan"), z=float("nan"), exact=True)
        return out
    if n <= 25:
        out.update(p=float(stats.binomtest(min(neg, pos), n, 0.5).pvalue), z=float("nan"),
                   exact=True)
    else:
        z = (min(neg, pos) + 0.5 - n / 2.0) / (0.5 * math.sqrt(n))
        out.update(p=_norm2(z), z=z, exact=False)
    dev = pos - n / 2.0
    out["z_np"] = math.copysign(max(abs(dev) - 0.5, 0.0), dev) / (0.5 * math.sqrt(n))
    out["p_exact"] = float(stats.binomtest(min(neg, pos), n, 0.5).pvalue)
    return out


# ============================================================================ two independent
def mann_whitney_test(a: np.ndarray, b: np.ndarray) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n2 = len(a), len(b)
    out = {"n1": n1, "n2": n2}
    if n1 == 0 or n2 == 0:
        return out
    allx = np.concatenate([a, b])
    ranks = stats.rankdata(allx)
    r1, r2 = float(ranks[:n1].sum()), float(ranks[n1:].sum())
    u1 = r1 - n1 * (n1 + 1) / 2
    u2 = n1 * n2 - u1
    u, w = (u1, r1) if u1 <= u2 else (u2, r2)
    n = n1 + n2
    var = n1 * n2 / 12.0 * ((n + 1) - tie_term(allx) / (n * (n - 1)))
    se = math.sqrt(var) if var > 0 else float("nan")
    z = (u - n1 * n2 / 2.0) / se if var > 0 else float("nan")
    out.update(r1=r1, r2=r2, u1=u1, u2=u2, u=u, w=w, z=z, p=_norm2(z), se=se,
               mean_u=n1 * n2 / 2.0, r=abs(z) / math.sqrt(n) if np.isfinite(z) else np.nan,
               z2=(u2 - n1 * n2 / 2.0) / se if var > 0 else float("nan"))
    if n1 * n2 <= 400:
        out["p_exact"] = float(stats.mannwhitneyu(a, b, alternative="two-sided",
                                                  method="exact").pvalue)
    return out


def ks_two_sample_test(a: np.ndarray, b: np.ndarray) -> dict:
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    n1, n2 = len(a), len(b)
    out = {"n1": n1, "n2": n2}
    if n1 == 0 or n2 == 0:
        return out
    grid = np.unique(np.concatenate([a, b]))
    f1 = np.searchsorted(a, grid, side="right") / n1
    f2 = np.searchsorted(b, grid, side="right") / n2
    diff = f1 - f2
    d = float(np.max(np.abs(diff)))
    z = d * math.sqrt(n1 * n2 / (n1 + n2))
    out.update(absolute=d, positive=float(max(diff.max(), 0.0)),
               negative=float(min(diff.min(), 0.0)), z=z, p=float(stats.kstwobign.sf(z)))
    return out


def moses_span_pvalue(span: int, m: int, n: int, h: int = 0) -> float:
    """P(trimmed control span <= span) (Siegel & Castellan 1988, eq. 6.13): m control and n
    experimental cases, h outliers trimmed from each end of the control group."""
    total = special.comb(m + n, m, exact=True)
    acc = 0
    for i in range(0, span - m + 2 * h + 1):
        acc += special.comb(i + m - 2 * h - 2, i, exact=True) * \
            special.comb(n + 2 * h + 1 - i, n - i, exact=True)
    return min(1.0, acc / total)


def moses_test(control: np.ndarray, experimental: np.ndarray, h: int | None = None) -> dict:
    """Moses test of extreme reactions; the first group is the control group. By default 5%
    of the control cases (at least one) are trimmed from each end."""
    c, e = np.asarray(control, float), np.asarray(experimental, float)
    m, n = len(c), len(e)
    out = {"m": m, "n": n}
    if m < 2 or n < 1:
        return out
    ranks = stats.rankdata(np.concatenate([c, e]))
    rc = np.sort(ranks[:m])
    span = int(round(rc[-1] - rc[0] + 1))
    if h is None:
        h = max(1, int(0.05 * m))
    h = min(h, (m - 2) // 2)
    trimmed = rc[h:m - h] if h else rc
    tspan = int(round(trimmed[-1] - trimmed[0] + 1))
    out.update(span=span, p_span=moses_span_pvalue(span, m, n, 0), h=h, trimmed=tspan,
               p_trimmed=moses_span_pvalue(tspan, m, n, h))
    return out


def _block_options(a: int, b: int) -> list[tuple[int, int, int]]:
    """(start, end, segments) arrangements of a block with a 0-labels and b 1-labels."""
    if a == 0:
        return [(1, 1, 1)]
    if b == 0:
        return [(0, 0, 1)]
    opts = []
    for start in (0, 1):
        for seg in range(2, a + b + 1):
            n_start = (seg + 1) // 2
            n_other = seg // 2
            need_s, need_o = (a, b) if start == 0 else (b, a)
            if 1 <= n_start <= need_s and 1 <= n_other <= need_o:
                end = start if seg % 2 else 1 - start
                opts.append((start, end, seg))
    return opts


def runs_range(values: np.ndarray, labels: np.ndarray) -> tuple[int, int, int, int]:
    """Minimum and maximum number of runs over all orderings within tied values, plus the
    number of inter-group ties and the cases they involve."""
    order = np.argsort(values, kind="mergesort")
    v, g = values[order], labels[order]
    blocks = []
    n_ties = cases = 0
    for value in np.unique(v):
        sel = g[v == value]
        a, b = int((sel == 0).sum()), int((sel == 1).sum())
        if a and b:
            n_ties += 1
            cases += a + b
        blocks.append(_block_options(a, b))
    results = []
    for best in (min, max):
        state: dict[int, int] = {}
        for opts in blocks:
            new: dict[int, int] = {}
            for start, end, seg in opts:
                if not state:
                    cand = seg
                    new[end] = best(new.get(end, cand), cand) if end in new else cand
                    continue
                for last, total in state.items():
                    cand = total + seg - (1 if last == start else 0)
                    new[end] = best(new[end], cand) if end in new else cand
            state = new
        results.append(best(state.values()))
    return results[0], results[1], n_ties, cases


def wald_wolfowitz_test(a: np.ndarray, b: np.ndarray) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n2 = len(a), len(b)
    out = {"n1": n1, "n2": n2}
    if n1 == 0 or n2 == 0:
        return out
    values = np.concatenate([a, b])
    labels = np.concatenate([np.zeros(n1, int), np.ones(n2, int)])
    rmin, rmax, n_ties, cases = runs_range(values, labels)
    n = n1 + n2
    mu = 2.0 * n1 * n2 / n + 1
    var = 2.0 * n1 * n2 * (2.0 * n1 * n2 - n) / (n * n * (n - 1)) if n > 1 else 0.0
    rs, ps = runs_pmf(n1, n2)

    def detail(r: int) -> dict:
        diff = r - mu
        if n < 50:
            diff = 0.0 if abs(diff) < 0.5 else diff - 0.5 * math.copysign(1, diff)
        z = diff / math.sqrt(var) if var > 0 else float("nan")
        return {"runs": r, "z": z, "p_exact": float(ps[rs <= r + _EPS].sum()),
                "p": float(stats.norm.cdf(z)) if np.isfinite(z) else np.nan}

    out.update(min=detail(rmin), max=detail(rmax), ties=n_ties, tie_cases=cases,
               exact_runs=rmin == rmax)
    return out


def hodges_lehmann_two(a: np.ndarray, b: np.ndarray, level: float = 0.95) -> dict:
    """Hodges-Lehmann estimate of the shift (median of all a_i - b_j) with the
    distribution-free (Moses) confidence interval from the Mann-Whitney null distribution."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return {"estimate": np.nan, "lower": np.nan, "upper": np.nan}
    diffs = np.sort(np.subtract.outer(a, b).ravel())
    m = len(diffs)
    z = stats.norm.ppf(0.5 + level / 2)
    c = int(math.floor(m / 2.0 - z * math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)))
    c = min(max(c, 1), m)
    return {"estimate": float(np.median(diffs)), "lower": float(diffs[c - 1]),
            "upper": float(diffs[m - c])}


def hodges_lehmann_one(d: np.ndarray, level: float = 0.95) -> dict:
    """Median of the Walsh averages with the Wilcoxon-based confidence interval."""
    d = np.asarray(d, float)
    n = len(d)
    if n == 0:
        return {"estimate": np.nan, "lower": np.nan, "upper": np.nan}
    i, j = np.triu_indices(n)
    walsh = np.sort((d[i] + d[j]) / 2.0)
    m = len(walsh)
    z = stats.norm.ppf(0.5 + level / 2)
    c = int(math.floor(n * (n + 1) / 4.0 - z * math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)))
    c = min(max(c, 1), m)
    return {"estimate": float(np.median(walsh)), "lower": float(walsh[c - 1]),
            "upper": float(walsh[m - c])}


# ============================================================================ k independent
def kruskal_test(samples: list[np.ndarray]) -> dict:
    samples = [np.asarray(s, float) for s in samples]
    allx = np.concatenate(samples) if samples else np.array([])
    n = len(allx)
    ranks = stats.rankdata(allx)
    sizes = [len(s) for s in samples]
    idx = np.cumsum([0] + sizes)
    sums = [float(ranks[idx[i]:idx[i + 1]].sum()) for i in range(len(samples))]
    means = [s / k if k else np.nan for s, k in zip(sums, sizes)]
    out = {"n": n, "sizes": sizes, "rank_sums": sums, "mean_ranks": means, "df": len(samples) - 1}
    valid = [k for k in sizes if k]
    if len(valid) < 2 or n < 2:
        out.update(h=np.nan, p=np.nan)
        return out
    h = 12.0 / (n * (n + 1)) * sum(s * s / k for s, k in zip(sums, sizes) if k) - 3 * (n + 1)
    c = 1 - tie_term(allx) / (n ** 3 - n)
    h = h / c if c > 0 else float("nan")
    out.update(h=float(h), p=float(stats.chi2.sf(h, len(valid) - 1)), tie_c=c)
    return out


def dunn_pairwise(samples: list[np.ndarray], labels: list[str]) -> list[dict]:
    """Dunn (1964) pairwise comparisons of mean ranks with the tie correction, as in the SPSS
    NPTESTS Kruskal-Wallis model viewer; p adjusted by Bonferroni (k(k-1)/2 tests)."""
    kw = kruskal_test(samples)
    n = kw["n"]
    allx = np.concatenate([np.asarray(s, float) for s in samples])
    base = n * (n + 1) / 12.0 - tie_term(allx) / (12.0 * (n - 1))
    k = len(samples)
    m = k * (k - 1) / 2
    rows = []
    for i, j in itertools.combinations(range(k), 2):
        ni, nj = kw["sizes"][i], kw["sizes"][j]
        if not ni or not nj:
            continue
        stat = kw["mean_ranks"][i] - kw["mean_ranks"][j]
        se = math.sqrt(base * (1.0 / ni + 1.0 / nj))
        z = stat / se if se else float("nan")
        p = _norm2(z)
        rows.append({"pair": (labels[i], labels[j]), "stat": stat, "se": se, "z": z, "p": p,
                     "adj": min(1.0, p * m)})
    return rows


def median_test(samples: list[np.ndarray], median: float | None = None) -> dict:
    """Median test: cases > pooled median vs <= median per group (scipy ``ties='below'``)."""
    samples = [np.asarray(s, float) for s in samples]
    allx = np.concatenate(samples)
    med = float(np.median(allx)) if median is None else float(median)
    above = np.array([int((s > med).sum()) for s in samples])
    below = np.array([int((s <= med).sum()) for s in samples])
    table = np.vstack([above, below]).astype(float)
    out = {"median": med, "above": above, "below": below, "n": int(table.sum()),
           "df": len(samples) - 1}
    keep = table.sum(axis=0) > 0
    t = table[:, keep]
    if t.shape[1] < 2 or (t.sum(axis=1) == 0).any():
        out.update(chi2=np.nan, p=np.nan)
        return out
    exp = np.outer(t.sum(axis=1), t.sum(axis=0)) / t.sum()
    chi = float(((t - exp) ** 2 / exp).sum())
    out.update(chi2=chi, p=float(stats.chi2.sf(chi, t.shape[1] - 1)), expected=exp,
               small=int((exp < 5).sum()), cells=exp.size)
    if t.shape[1] == 2:
        yates = float(stats.chi2_contingency(t, correction=True)[0])
        out.update(yates=yates, p_yates=float(stats.chi2.sf(yates, 1)),
                   p_exact=float(stats.fisher_exact(t)[1]))
    return out


def _jt_stat(samples: list[np.ndarray]) -> float:
    j = 0.0
    for a, b in itertools.combinations(samples, 2):
        diff = np.subtract.outer(np.asarray(a, float), np.asarray(b, float))
        j += float((diff < 0).sum() + 0.5 * (diff == 0).sum())
    return j


def jonckheere_test(samples: list[np.ndarray]) -> dict:
    """Jonckheere-Terpstra test for ordered alternatives with the tie-corrected variance
    (Hollander & Wolfe 1999, eq. 6.19)."""
    samples = [np.asarray(s, float) for s in samples if len(s)]
    sizes = np.array([len(s) for s in samples], float)
    n = sizes.sum()
    allx = np.concatenate(samples)
    j = _jt_stat(samples)
    mean = (n * n - (sizes ** 2).sum()) / 4.0
    _, t = np.unique(allx, return_counts=True)
    t = t.astype(float)
    var = (n * (n - 1) * (2 * n + 5) - (sizes * (sizes - 1) * (2 * sizes + 5)).sum()
           - (t * (t - 1) * (2 * t + 5)).sum()) / 72.0
    if n > 2:
        var += ((sizes * (sizes - 1) * (sizes - 2)).sum() * (t * (t - 1) * (t - 2)).sum()
                / (36.0 * n * (n - 1) * (n - 2)))
    var += (sizes * (sizes - 1)).sum() * (t * (t - 1)).sum() / (8.0 * n * (n - 1))
    sd = math.sqrt(var) if var > 0 else float("nan")
    z = (j - mean) / sd if var > 0 else float("nan")
    return {"levels": len(samples), "n": int(n), "j": j, "mean": mean, "sd": sd, "z": z,
            "p": _norm2(z)}


# ============================================================================ related samples
def friedman_test(x: np.ndarray) -> dict:
    """Friedman test on an n x k matrix (tie-corrected), with Kendall's W."""
    x = np.asarray(x, float)
    n, k = x.shape
    ranks = np.apply_along_axis(stats.rankdata, 1, x)
    rsum = ranks.sum(axis=0)
    out = {"n": n, "k": k, "mean_ranks": rsum / n, "df": k - 1}
    if n == 0 or k < 2:
        out.update(chi2=np.nan, p=np.nan, w=np.nan)
        return out
    ties = sum(tie_term(row) for row in x)
    denom = n * k * (k + 1) - ties / (k - 1)
    chi = 12.0 * ((rsum - n * (k + 1) / 2.0) ** 2).sum() / denom if denom > 0 else np.nan
    out.update(chi2=float(chi), p=float(stats.chi2.sf(chi, k - 1)),
               w=float(chi / (n * (k - 1))))
    return out


def friedman_pairwise(x: np.ndarray, labels: list[str]) -> list[dict]:
    """Pairwise mean-rank comparisons after Friedman (SE = sqrt(k(k+1)/(6n)))."""
    fr = friedman_test(x)
    n, k = fr["n"], fr["k"]
    se = math.sqrt(k * (k + 1) / (6.0 * n)) if n else float("nan")
    m = k * (k - 1) / 2
    rows = []
    for i, j in itertools.combinations(range(k), 2):
        stat = fr["mean_ranks"][i] - fr["mean_ranks"][j]
        z = stat / se
        p = _norm2(z)
        rows.append({"pair": (labels[i], labels[j]), "stat": float(stat), "se": se, "z": z,
                     "p": p, "adj": min(1.0, p * m)})
    return rows


def cochran_test(x: np.ndarray) -> dict:
    """Cochran's Q on an n x k 0/1 matrix."""
    x = np.asarray(x, float)
    n, k = x.shape
    c = x.sum(axis=0)
    r = x.sum(axis=1)
    total = x.sum()
    denom = k * total - (r ** 2).sum()
    q = (k - 1) * (k * (c ** 2).sum() - total ** 2) / denom if denom else float("nan")
    return {"n": n, "k": k, "q": float(q), "df": k - 1, "col_sums": c,
            "p": float(stats.chi2.sf(q, k - 1)) if np.isfinite(q) else np.nan}


def cochran_pairwise(x: np.ndarray, labels: list[str]) -> list[dict]:
    """Pairwise comparisons after Cochran's Q (difference of success proportions,
    SE = sqrt(2(kΣr - Σr²)/(n²k(k-1))))."""
    x = np.asarray(x, float)
    n, k = x.shape
    r = x.sum(axis=1)
    var = 2.0 * (k * r.sum() - (r ** 2).sum()) / (n * n * k * (k - 1)) if n and k > 1 else 0
    se = math.sqrt(var) if var > 0 else float("nan")
    props = x.mean(axis=0)
    m = k * (k - 1) / 2
    rows = []
    for i, j in itertools.combinations(range(k), 2):
        stat = float(props[i] - props[j])
        z = stat / se if se else float("nan")
        p = _norm2(z)
        rows.append({"pair": (labels[i], labels[j]), "stat": stat, "se": se, "z": z, "p": p,
                     "adj": min(1.0, p * m)})
    return rows


def mcnemar_test(a: np.ndarray, b: np.ndarray) -> dict:
    """McNemar test for two dichotomous related variables (SPSS: binomial when the number of
    changes is at most 25, otherwise chi-square with continuity correction)."""
    a, b = np.asarray(a), np.asarray(b)
    cats = sorted(set(a.tolist()) | set(b.tolist()), key=lambda v: (isinstance(v, str), v))
    out = {"cats": cats, "n": len(a)}
    if len(cats) != 2:
        out["error"] = "not dichotomous"
        return out
    lo, hi = cats
    table = np.array([[np.sum((a == lo) & (b == lo)), np.sum((a == lo) & (b == hi))],
                      [np.sum((a == hi) & (b == lo)), np.sum((a == hi) & (b == hi))]], int)
    nb, nc = int(table[0, 1]), int(table[1, 0])
    out["table"] = table
    changes = nb + nc
    out["changes"] = changes
    if changes == 0:
        out.update(p=1.0, exact=True, chi2=np.nan)
        return out
    p_exact = float(min(1.0, 2 * stats.binom.cdf(min(nb, nc), changes, 0.5)))
    chi = (abs(nb - nc) - 1) ** 2 / changes
    out.update(chi2=chi, p_asymp=float(stats.chi2.sf(chi, 1)), p_exact=p_exact)
    out.update(p=p_exact if changes <= 25 else out["p_asymp"], exact=changes <= 25)
    return out


def marginal_homogeneity_test(a: np.ndarray, b: np.ndarray) -> dict:
    """SPSS marginal homogeneity test (ordinal extension of McNemar): over the off-diagonal
    pairs, MH = Σ x2 (the second variable's scores, as SPSS prints it: IBM Exact Tests manual,
    Fig. 5.8, observed 6.687 = sum of the control doses), E(MH) = Σ (x1 + x2)/2,
    Var(MH) = Σ (x1 - x2)²/4."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    distinct = len(np.unique(np.concatenate([a, b])))
    off = a != b
    x1, x2 = a[off], b[off]
    obs = float(x2.sum())
    mean = float(((x1 + x2) / 2).sum())
    var = float((((x1 - x2) ** 2) / 4).sum())
    sd = math.sqrt(var) if var > 0 else float("nan")
    z = (obs - mean) / sd if var > 0 else float("nan")
    return {"distinct": distinct, "off": int(off.sum()), "obs": obs, "mean": mean, "sd": sd,
            "z": z, "p": _norm2(z)}


# ============================================================================ exact helpers
def _sum_distribution(values: np.ndarray, k: int | None) -> np.ndarray:
    """Counts of subset sums of the integer ``values``: choosing exactly k (k given) or any
    subset (k None). Returns counts indexed by the sum."""
    total = int(values.sum())
    if k is None:
        dp = np.zeros(total + 1)
        dp[0] = 1.0
        for v in values:
            v = int(v)
            new = dp.copy()
            if v:
                new[v:] += dp[:total + 1 - v]
            else:
                new += dp
            dp = new
        return dp
    dp = np.zeros((k + 1, total + 1))
    dp[0, 0] = 1.0
    for v in values:
        v = int(v)
        new = dp.copy()
        new[1:, v:] += dp[:-1, :total + 1 - v]
        dp = new
    return dp[k]


def _two_sided(support: np.ndarray, counts: np.ndarray, observed: float, center: float) -> float:
    mask = np.abs(support - center) >= abs(observed - center) - _EPS
    return float(min(1.0, counts[mask].sum() / counts.sum()))


def exact_mann_whitney(a, b, method="exact", samples=MC_SAMPLES, level=MC_LEVEL,
                       seed=MC_SEED, limit=2 * 10 ** 8) -> PValue:
    """Exact (tie-aware permutation) or Monte Carlo two-sided p for the rank-sum test."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n = len(a), len(a) + len(b)
    ranks = stats.rankdata(np.concatenate([a, b]))
    doubled = np.round(2 * ranks).astype(int)
    obs = float(doubled[:n1].sum())
    center = n1 * (n + 1.0)
    if method == "exact" and n * (n1 + 1) * doubled.sum() <= limit:
        counts = _sum_distribution(doubled, n1)
        return PValue(_two_sided(np.arange(len(counts), dtype=float), counts, obs, center))
    rng = _rng(seed)
    extreme = []
    for m in _chunks(samples):
        perm = rng.permuted(np.tile(doubled, (m, 1)), axis=1)[:, :n1].sum(axis=1)
        extreme.append(np.abs(perm - center) >= abs(obs - center) - _EPS)
    return _mc(np.concatenate(extreme), samples, level)


def exact_wilcoxon(d, method="exact", samples=MC_SAMPLES, level=MC_LEVEL, seed=MC_SEED,
                   limit=10 ** 9) -> PValue:
    d = np.asarray(d, float)
    nz = d[d != 0]
    if len(nz) == 0:
        return PValue(1.0)
    doubled = np.round(2 * stats.rankdata(np.abs(nz))).astype(int)
    obs = float(doubled[nz > 0].sum())
    center = doubled.sum() / 2.0
    if method == "exact" and len(nz) * doubled.sum() <= limit:
        counts = _sum_distribution(doubled, None)
        return PValue(_two_sided(np.arange(len(counts), dtype=float), counts, obs, center))
    rng = _rng(seed)
    extreme = []
    for m in _chunks(samples):
        signs = rng.integers(0, 2, size=(m, len(nz)))
        t = signs @ doubled
        extreme.append(np.abs(t - center) >= abs(obs - center) - _EPS)
    return _mc(np.concatenate(extreme), samples, level)


def exact_sign(d, **_) -> PValue:
    d = np.asarray(d, float)
    neg, pos = int((d < 0).sum()), int((d > 0).sum())
    if neg + pos == 0:
        return PValue(1.0)
    return PValue(float(stats.binomtest(min(neg, pos), neg + pos, 0.5).pvalue))


def exact_binomial(k: int, n: int, prop: float = 0.5, **_) -> PValue:
    return PValue(binomial_test(k, n, prop)["p"])


def exact_mcnemar(b: int, c: int, **_) -> PValue:
    n = b + c
    return PValue(1.0 if n == 0 else float(min(1.0, 2 * stats.binom.cdf(min(b, c), n, .5))))


def exact_runs(x, value: float, **_) -> PValue:
    """Exact two-sided p for the one-sample runs test."""
    x = np.asarray(x, float)
    above = x >= value
    n1, n2 = int((~above).sum()), int(above.sum())
    if n1 == 0 or n2 == 0:
        return PValue(float("nan"))
    runs = int(1 + np.count_nonzero(above[1:] != above[:-1]))
    rs, ps = runs_pmf(n1, n2)
    mu = 2.0 * n1 * n2 / (n1 + n2) + 1
    return PValue(_two_sided(rs, ps, runs, mu))


def _label_assignments(sizes: list[int]):
    n = sum(sizes)

    def rec(remaining: tuple, g: int, labels: np.ndarray):
        if g == len(sizes) - 1:
            labels = labels.copy()
            labels[list(remaining)] = g
            yield labels
            return
        for combo in itertools.combinations(remaining, sizes[g]):
            lab = labels.copy()
            lab[list(combo)] = g
            rest = tuple(i for i in remaining if i not in set(combo))
            yield from rec(rest, g + 1, lab)

    yield from rec(tuple(range(n)), 0, np.zeros(n, int))


def _multinomial(sizes: list[int]) -> float:
    n = sum(sizes)
    out = math.lgamma(n + 1) - sum(math.lgamma(k + 1) for k in sizes)
    return math.exp(out)


def _group_test(samples, stat_fn, method, samples_mc, level, seed, limit,
                two_sided_center=None) -> PValue:
    """Permutation test over group labels. ``stat_fn(values, labels_matrix) -> stats``."""
    samples = [np.asarray(s, float) for s in samples]
    sizes = [len(s) for s in samples]
    values = np.concatenate(samples)
    labels0 = np.repeat(np.arange(len(samples)), sizes)
    obs = float(stat_fn(values, labels0[None, :])[0])

    def extreme(t):
        if two_sided_center is None:
            return t >= obs - _EPS * max(1.0, abs(obs))
        return np.abs(t - two_sided_center) >= abs(obs - two_sided_center) - _EPS

    if method == "exact" and _multinomial(sizes) <= limit:
        lab = np.array(list(_label_assignments(sizes)))
        t = np.concatenate([stat_fn(values, lab[i:i + 2000]) for i in range(0, len(lab), 2000)])
        return PValue(float(np.mean(extreme(t))))
    rng = _rng(seed)
    out = []
    for m in _chunks(samples_mc):
        lab = rng.permuted(np.tile(labels0, (m, 1)), axis=1)
        out.append(extreme(stat_fn(values, lab)))
    return _mc(np.concatenate(out), samples_mc, level)


def _kw_stat(values, labels):
    n = len(values)
    ranks = stats.rankdata(values)
    k = labels.max() + 1
    out = np.zeros(labels.shape[0])
    for g in range(k):
        mask = labels == g
        ng = mask[0].sum()
        if ng:
            out += (mask @ ranks) ** 2 / ng
    c = 1 - tie_term(values) / (n ** 3 - n)
    return (12.0 / (n * (n + 1)) * out - 3 * (n + 1)) / (c if c > 0 else 1)


def exact_kruskal(samples, method="exact", samples_mc=MC_SAMPLES, level=MC_LEVEL,
                  seed=MC_SEED, limit=EXACT_LIMIT) -> PValue:
    return _group_test(samples, _kw_stat, method, samples_mc, level, seed, limit)


def exact_median_test(samples, median: float | None = None, method="exact",
                      samples_mc=MC_SAMPLES, level=MC_LEVEL, seed=MC_SEED,
                      limit=EXACT_LIMIT) -> PValue:
    allx = np.concatenate([np.asarray(s, float) for s in samples])
    med = float(np.median(allx)) if median is None else median
    above = (allx > med).astype(float)
    a_tot = above.sum()
    n = len(allx)

    def stat(values, labels):
        k = labels.max() + 1
        chi = np.zeros(labels.shape[0])
        for g in range(k):
            mask = labels == g
            ng = mask[0].sum()
            if not ng:
                continue
            obs_a = mask @ above
            for o, tot in ((obs_a, a_tot), (ng - obs_a, n - a_tot)):
                e = ng * tot / n
                if e > 0:
                    chi += (o - e) ** 2 / e
        return chi

    return _group_test([np.asarray(s, float) for s in samples], stat, method, samples_mc,
                       level, seed, limit)


def exact_jonckheere(samples, method="exact", samples_mc=MC_SAMPLES, level=MC_LEVEL,
                     seed=MC_SEED, limit=EXACT_LIMIT) -> PValue:
    samples = [np.asarray(s, float) for s in samples]
    values = np.concatenate(samples)
    cmp = (np.less.outer(values, values) + 0.5 * np.equal.outer(values, values)).astype(float)
    sizes = np.array([len(s) for s in samples], float)
    center = (sizes.sum() ** 2 - (sizes ** 2).sum()) / 4.0

    def stat(_values, labels):
        k = labels.max() + 1
        out = np.zeros(labels.shape[0])
        onehot = [(labels == g).astype(float) for g in range(k)]
        for g in range(k):
            left = onehot[g] @ cmp
            for h in range(g + 1, k):
                out += (left * onehot[h]).sum(axis=1)
        return out

    return _group_test(samples, stat, method, samples_mc, level, seed, limit, center)


def exact_ks_2samp(a, b, method="exact", samples=MC_SAMPLES, level=MC_LEVEL, seed=MC_SEED,
                   limit=EXACT_LIMIT) -> PValue:
    a, b = np.asarray(a, float), np.asarray(b, float)
    values = np.concatenate([a, b])
    if method == "exact" and len(np.unique(values)) == len(values):
        return PValue(float(stats.ks_2samp(a, b, method="exact").pvalue))
    n1, n2 = len(a), len(b)
    order = np.argsort(values, kind="mergesort")
    sv = values[order]
    last = np.r_[sv[1:] != sv[:-1], True]

    def stat(_v, labels):
        lab = labels[:, order]
        c1 = np.cumsum(lab == 0, axis=1) / n1
        c2 = np.cumsum(lab == 1, axis=1) / n2
        return np.abs(c1 - c2)[:, last].max(axis=1)

    return _group_test([a, b], stat, method, samples, level, seed, limit)


def _within_rows_test(x, stat_fn, method, samples, level, seed, limit) -> PValue:
    x = np.asarray(x, float)
    obs = float(stat_fn(x[None])[0])
    perms = [sorted(set(itertools.permutations(row))) for row in x]
    count = math.prod(len(p) for p in perms)
    if method == "exact" and count <= limit:
        arr = np.array([np.array(c) for c in itertools.product(*perms)], float)
        t = stat_fn(arr)
        return PValue(float(np.mean(t >= obs - _EPS * max(1.0, abs(obs)))))
    rng = _rng(seed)
    out = []
    for m in _chunks(samples, 1000):
        arr = rng.permuted(np.broadcast_to(x, (m,) + x.shape).copy(), axis=2)
        out.append(stat_fn(arr) >= obs - _EPS * max(1.0, abs(obs)))
    return _mc(np.concatenate(out), samples, level)


def exact_friedman(x, method="exact", samples=MC_SAMPLES, level=MC_LEVEL, seed=MC_SEED,
                   limit=EXACT_LIMIT) -> PValue:
    ranks = np.apply_along_axis(stats.rankdata, 1, np.asarray(x, float))
    return _within_rows_test(ranks, lambda a: (a.sum(axis=1) ** 2).sum(axis=1), method,
                             samples, level, seed, limit)


def exact_kendall_w(x, **kw) -> PValue:
    """Kendall's W is a monotone function of the Friedman statistic: same p-value."""
    return exact_friedman(x, **kw)


def exact_cochran(x, method="exact", samples=MC_SAMPLES, level=MC_LEVEL, seed=MC_SEED,
                  limit=EXACT_LIMIT) -> PValue:
    return _within_rows_test(np.asarray(x, float), lambda a: (a.sum(axis=1) ** 2).sum(axis=1),
                             method, samples, level, seed, limit)


def exact_chisq_gof(observed, weights=None, method="exact", samples=MC_SAMPLES,
                    level=MC_LEVEL, seed=MC_SEED, limit=EXACT_LIMIT) -> PValue:
    """Exact multinomial (or Monte Carlo) p for the chi-square goodness-of-fit statistic."""
    obs = np.asarray(observed, int)
    k, n = len(obs), int(obs.sum())
    w = np.ones(k) if weights is None else np.asarray(weights, float)
    p = w / w.sum()
    exp = n * p
    stat_obs = float(((obs - exp) ** 2 / exp).sum())
    if method == "exact" and special.comb(n + k - 1, k - 1) <= limit:
        tables = []

        def rec(prefix, left, cells):
            if cells == 1:
                tables.append(prefix + [left])
                return
            for v in range(left + 1):
                rec(prefix + [v], left - v, cells - 1)

        rec([], n, k)
        t = np.array(tables, float)
        logp = stats.multinomial.logpmf(t, n, p)
        chi = ((t - exp) ** 2 / exp).sum(axis=1)
        return PValue(float(min(1.0, np.exp(logp)[chi >= stat_obs - _EPS].sum())))
    rng = _rng(seed)
    t = rng.multinomial(n, p, size=samples)
    chi = ((t - exp) ** 2 / exp).sum(axis=1)
    return _mc(chi >= stat_obs - _EPS, samples, level)


EXACT_FUNCTIONS = {
    "mann_whitney": exact_mann_whitney, "wilcoxon": exact_wilcoxon, "sign": exact_sign,
    "binomial": exact_binomial, "mcnemar": exact_mcnemar, "runs": exact_runs,
    "kruskal_wallis": exact_kruskal, "median": exact_median_test,
    "jonckheere": exact_jonckheere, "ks_2samp": exact_ks_2samp, "friedman": exact_friedman,
    "kendall_w": exact_kendall_w, "cochran": exact_cochran, "chisq_gof": exact_chisq_gof,
}


# ============================================================================ NPTESTS procedures
ONE_SAMPLE_TESTS = ("binomial", "chisq", "ks", "wilcoxon", "runs")
INDEPENDENT_TESTS = ("mw", "ks", "ww", "moses", "hl", "kw", "median", "jt")
RELATED_TESTS = ("sign", "wilcoxon", "mcnemar", "marginal", "hl", "friedman", "kendall",
                 "cochran")


@dataclass
class _Hyp:
    null: str
    test: str
    p: float
    exact: bool = False
    note: str = ""


def _code(v) -> str:
    from rsstatchem.core.dataset import format_code

    return format_code(v) if isinstance(v, float) else str(v)


def _fmt2(v: float) -> str:
    return f"{v:.2f}" if v is not None and np.isfinite(v) else ""


def is_categorical(var, ordinal_categorical: bool = True) -> bool:
    """NPTESTS treats nominal (and, for one-sample tests, ordinal) fields as categorical."""
    if var.type != "numeric":
        return True
    return var.measure == "nominal" or (ordinal_categorical and var.measure == "ordinal")


def _restrict(ds, names: list[str], listwise: bool):
    from rsstatchem.core.stats.nonparametric import _view

    return _view(ds, names, listwise)


def _cats(series: pd.Series) -> list:
    return sorted(series.dropna().unique(), key=lambda v: (isinstance(v, str), v))


def _summary_table(title: str, rows: list[tuple[str, str]], notes=None):
    from rsstatchem.core.output import Table

    return Table(title, ["", ""], [[a, b] for a, b in rows], footnotes=list(notes or []))


def _hypothesis_table(hyps: list[_Hyp], alpha: float):
    from rsstatchem.core.output import Table
    from rsstatchem.core.stats.common import fmt, fmt_p

    notes = [f"a. The significance level is {fmt(alpha, 3)}.",
             "b. Asymptotic significance is displayed."]
    letters: dict[str, int] = {}
    if any(h.exact for h in hyps):
        letters["exact"] = len(notes)
        notes.append(f"{chr(ord('a') + len(notes))}. Exact significance is displayed for this "
                     f"test.")
    for h in hyps:
        if h.note and h.note not in letters:
            letters[h.note] = len(notes)
            notes.append(f"{chr(ord('a') + len(notes))}. {h.note}")
    rows = []
    for i, h in enumerate(hyps, start=1):
        decision = ("Reject the null hypothesis." if np.isfinite(h.p) and h.p < alpha else
                    "Retain the null hypothesis." if np.isfinite(h.p) else
                    "Unable to compute.")
        mark = sup_letter(letters["exact"]) if h.exact else ""
        if h.note:
            mark += sup_letter(letters[h.note])
        rows.append([str(i), h.null, h.test, fmt_p(h.p) + mark, decision])
    return Table("Hypothesis Test Summary", ["", "Null Hypothesis", "Test", "Sig.ᵃ ᵇ",
                                             "Decision"], rows, footnotes=notes)


def sup_letter(i: int) -> str:
    from rsstatchem.core.stats.common import sup

    return sup(chr(ord("a") + i))


def _pairwise_table(title: str, rows: list[dict], alpha: float):
    from rsstatchem.core.output import Table
    from rsstatchem.core.stats.common import fmt, fmt_p

    body = [[f"{r['pair'][0]}-{r['pair'][1]}", fmt(r["stat"], 3), fmt(r["se"], 3),
             fmt(r["z"], 3), fmt_p(r["p"]), fmt_p(r["adj"])] for r in rows]
    return Table(title, ["Sample 1-Sample 2", "Test Statistic", "Std. Error",
                         "Std. Test Statistic", "Sig.", "Adj. Sig.ᵃ"], body, footnotes=[
        "Each row tests the null hypothesis that the Sample 1 and Sample 2 distributions are "
        "the same.",
        f"Asymptotic significances (2-sided tests) are displayed. The significance level is "
        f"{fmt(alpha, 3)}.",
        "a. Significance values have been adjusted by the Bonferroni correction for multiple "
        "tests."])


def _ci_table(rows: list[tuple[str, str, dict]], level: float, decimals: int = 2):
    from rsstatchem.core.output import Table
    from rsstatchem.core.stats.common import fmt

    pc = f"{level * 100:g}% Confidence Interval"
    body = [[kind, param, fmt(r["estimate"], decimals + 3), fmt(r["lower"], decimals + 3),
             fmt(r["upper"], decimals + 3)] for kind, param, r in rows]
    return Table("Confidence Interval Summary", ["Confidence Interval Type", "Parameter",
                                                 "Estimate", (pc, "Lower"), (pc, "Upper")],
                 body)


def _assemble(section, hyps, details, alpha):
    if not hyps and not details:
        section.note("Warning", "No test could be performed on the selected fields with the "
                     "chosen settings (check the measurement levels and the number of "
                     "categories).", style="warning")
        return section
    if hyps:
        section.table(_hypothesis_table(hyps, alpha))
    for item in details:
        if isinstance(item, tuple):
            section.image(item[1], item[2])
        else:
            section.table(item)
    return section


def _ks_text(dist: str, prm: dict) -> str:
    return {"normal": f"normal with mean {_fmt2(prm.get('mean'))} and standard deviation "
                      f"{_fmt2(prm.get('sd'))}",
            "uniform": f"uniform with minimum {_fmt2(prm.get('min'))} and maximum "
                       f"{_fmt2(prm.get('max'))}",
            "exponential": f"exponential with mean {_fmt2(prm.get('mean'))}",
            "poisson": f"Poisson with mean {_fmt2(prm.get('mean'))}"}[dist]


def auto_one_sample(var, n_categories: int) -> str:
    if is_categorical(var, True):
        return "binomial" if n_categories == 2 else "chisq"
    return "ks"


def one_sample(ds, fields: list[str], tests: tuple | None = None, alpha: float = 0.05,
               level: float = 0.95, listwise: bool = False, binomial_prop: float = 0.5,
               ks_dists: tuple = ("normal",), wilcoxon_median: float = 0.0,
               runs_cut: str = "median", runs_value: float | None = None,
               charts_on: bool = True):
    """SPSS NPTESTS /ONESAMPLE. ``tests=None`` chooses the test from the measurement level:
    binomial (2 categories), chi-square (more categories) or K-S normal (continuous)."""
    from rsstatchem.core.output import Section
    from rsstatchem.core.stats import charts
    from rsstatchem.core.stats.common import fmt, fmt_int, fmt_p, value_text
    from rsstatchem.core.stats.nonparametric import cut_value

    section = Section("Nonparametric Tests — One Sample")
    spec = "AUTOMATIC" if tests is None else " ".join(t.upper() for t in tests)
    section.log(f"NPTESTS\n  /ONESAMPLE TEST ({' '.join(fields)}) {spec}\n"
                f"  /MISSING SCOPE={'LISTWISE' if listwise else 'ANALYSIS'}\n"
                f"  /CRITERIA ALPHA={alpha:g} CILEVEL={level * 100:g}.")
    ds = _restrict(ds, fields, listwise)
    hyps, details = [], []
    for name in fields:
        var = ds.var(name)
        title = var.title
        s = ds.analysis_series(name).dropna()
        cat = is_categorical(var, True)
        cats = _cats(s)
        chosen = (auto_one_sample(var, len(cats)),) if tests is None else tests
        x = pd.to_numeric(s, errors="coerce").dropna().to_numpy(float) \
            if var.type == "numeric" else None
        for test in chosen:
            if test == "binomial":
                n = len(s)
                if len(cats) == 2:
                    first = s.iloc[0]
                    order = [first] + [c for c in cats if c != first]
                    k = int((s == order[0]).sum())
                    names = [value_text(var, c) for c in order]
                elif not cat and x is not None and len(x):
                    mid = float((x.min() + x.max()) / 2)
                    k = int((x <= mid).sum())
                    names = [f"<={_fmt2(mid)}", f">{_fmt2(mid)}"]
                else:
                    continue
                r = binomial_test(k, n, binomial_prop)
                hyps.append(_Hyp(f"The categories defined by {title} = {names[0]} and "
                                 f"{names[1]} occur with probabilities {binomial_prop:g} and "
                                 f"{1 - binomial_prop:g}.", "One-Sample Binomial Test", r["p"],
                                 exact=True))
                details.append(_summary_table(f"One-Sample Binomial Test Summary ({title})", [
                    ("Total N", fmt_int(n)), ("Test Statistic", fmt_int(k)),
                    ("Standard Error", fmt(r["se"], 3)),
                    ("Standardized Test Statistic", fmt(r["z"], 3)),
                    ("Exact Sig.(2-sided test)" if r["two_sided"] else
                     "Exact Sig.(1-sided test)", fmt_p(r["p"]))]))
                if charts_on:
                    details.append(("image", f"One-Sample Binomial Test ({title})",
                                    charts.bar_chart(names, [k, n - k], title)))
            elif test == "chisq":
                if not cat or len(cats) < 2:
                    continue
                obs = np.array([int((s == c).sum()) for c in cats], float)
                r = chisq_gof(obs)
                hyps.append(_Hyp(f"The categories of {title} occur with equal probabilities.",
                                 "One-Sample Chi-Square Test", r["p"]))
                details.append(_summary_table(f"One-Sample Chi-Square Test Summary ({title})", [
                    ("Total N", fmt_int(obs.sum())), ("Test Statistic", fmt(r["chi2"], 3) + "ᵃ"),
                    ("Degree Of Freedom", fmt_int(r["df"])),
                    ("Asymptotic Sig.(2-sided test)", fmt_p(r["p"]))],
                    [f"a. There are {r['small']} cells ({100 * r['small'] / len(obs):.1f}%) "
                     f"with expected values less than 5. The minimum expected value is "
                     f"{r['expected'].min():.3f}."]))
                if charts_on:
                    details.append(("image", f"One-Sample Chi-Square Test ({title})",
                                    charts.bar_chart([value_text(var, c) for c in cats],
                                                     list(obs), title)))
            elif test == "ks" and not cat and x is not None:
                for dist in ks_dists:
                    r = ks_one_sample_test(x, dist)
                    if "absolute" not in r:
                        continue
                    lill = bool(r.get("lilliefors"))
                    hyps.append(_Hyp(f"The distribution of {title} is "
                                     f"{_ks_text(dist, r['params'])}.",
                                     "One-Sample Kolmogorov-Smirnov Test", r["p"],
                                     note="Lilliefors Corrected" if lill else ""))
                    details.append(_summary_table(
                        f"One-Sample Kolmogorov-Smirnov Test Summary ({title})", [
                            ("Total N", fmt_int(r["n"])),
                            ("Most Extreme Differences: Absolute", fmt(r["absolute"], 3)),
                            ("Most Extreme Differences: Positive", fmt(r["positive"], 3)),
                            ("Most Extreme Differences: Negative", fmt(r["negative"], 3)),
                            ("Test Statistic", fmt(r["statistic"], 3)),
                            ("Asymptotic Sig.(2-sided test)" + ("ᵃ" if lill else ""),
                             fmt_p(r["p"]))],
                        ["a. Lilliefors Corrected"] if lill else []))
                    if charts_on:
                        details.append(("image", f"One-Sample Kolmogorov-Smirnov Test ({title})",
                                        charts.histogram(x, title, title, dist == "normal")))
            elif test == "wilcoxon" and not cat and x is not None:
                r = wilcoxon_test(x - wilcoxon_median)
                if not r["n"]:
                    continue
                hyps.append(_Hyp(f"The median of {title} equals {_fmt2(wilcoxon_median)}.",
                                 "One-Sample Wilcoxon Signed Rank Test", r["p"]))
                details.append(_summary_table(
                    f"One-Sample Wilcoxon Signed Rank Test Summary ({title})", [
                        ("Total N", fmt_int(r["total"])), ("Test Statistic", fmt(r["r_pos"], 3)),
                        ("Standard Error", fmt(r["se"], 3)),
                        ("Standardized Test Statistic", fmt(r["z_pos"], 3)),
                        ("Asymptotic Sig.(2-sided test)", fmt_p(r["p"]))]))
                if charts_on:
                    details.append(("image", f"One-Sample Wilcoxon Signed Rank Test ({title})",
                                    charts.histogram(x, title, title, False)))
            elif test == "runs" and x is not None and len(x) > 2:
                if cat:
                    if len(cats) != 2:
                        continue
                    cut = float(max(cats))
                    cut_text = f"{title}={value_text(var, min(cats))} and " \
                               f"{value_text(var, max(cats))}"
                    r = runs_test(x, cut)
                else:
                    cut, _ = cut_value(x, runs_cut, runs_value)
                    cut_text = f"{title}<={_fmt2(cut)} and >{_fmt2(cut)}"
                    r = runs_test(x, float(np.nextafter(cut, np.inf)))
                hyps.append(_Hyp(f"The sequence of values defined by {cut_text} is random.",
                                 "One-Sample Runs Test", r["p"]))
                details.append(_summary_table(f"One-Sample Runs Test Summary ({title})", [
                    ("Total N", fmt_int(r["n"])), ("Test Statistic", fmt_int(r["runs"])),
                    ("Standard Error", fmt(r["sd"], 3)),
                    ("Standardized Test Statistic", fmt(r["z"], 3)),
                    ("Asymptotic Sig.(2-sided test)", fmt_p(r["p"]))]))
    return _assemble(section, hyps, details, alpha)


def independent_samples(ds, fields: list[str], group: str, tests: tuple | None = None,
                        alpha: float = 0.05, level: float = 0.95, listwise: bool = False,
                        multiple: str = "all", moses_trim: int | None = None,
                        charts_on: bool = True):
    """SPSS NPTESTS /INDEPENDENT. ``tests=None``: Mann-Whitney for 2 groups, Kruskal-Wallis
    for more."""
    from rsstatchem.core.output import Section
    from rsstatchem.core.stats import charts
    from rsstatchem.core.stats.common import fmt, fmt_int, fmt_p, numeric, value_text

    section = Section("Nonparametric Tests — Independent Samples")
    spec = "AUTOMATIC" if tests is None else " ".join(t.upper() for t in tests)
    section.log(f"NPTESTS\n  /INDEPENDENT TEST ({' '.join(fields)}) GROUP ({group}) {spec}\n"
                f"  /MISSING SCOPE={'LISTWISE' if listwise else 'ANALYSIS'}\n"
                f"  /CRITERIA ALPHA={alpha:g} CILEVEL={level * 100:g}.")
    ds = _restrict(ds, fields + [group], listwise)
    gvar = ds.var(group)
    g = ds.analysis_series(group)
    gt = gvar.title
    hyps, details = [], []
    for name in fields:
        var = ds.var(name)
        title = var.title
        frame = pd.DataFrame({"y": numeric(ds, name), "g": g}).dropna()
        cats = _cats(frame["g"])
        samples = [frame.loc[frame["g"] == c, "y"].to_numpy(float) for c in cats]
        labels = [value_text(gvar, c) for c in cats]
        if len(cats) < 2:
            section.note("Warning", f"{title}: the grouping field {gt} has fewer than two "
                         "groups.", style="warning")
            continue
        chosen = (("mw",) if len(cats) == 2 else ("kw",)) if tests is None else tests
        same = f"The distribution of {title} is the same across categories of {gt}."
        n_total = len(frame)
        a, b = samples[0], samples[1]
        for test in chosen:
            if test in ("mw", "ks", "ww", "moses", "hl") and len(cats) != 2:
                section.note("", f"{title}: {gt} has {len(cats)} groups; the "
                             f"{test.upper()} test needs exactly 2 groups and was skipped.")
                continue
            if test == "mw":
                r = mann_whitney_test(a, b)
                exact = "p_exact" in r
                hyps.append(_Hyp(same, "Independent-Samples Mann-Whitney U Test",
                                 r["p_exact"] if exact else r["p"], exact=exact))
                rows = [("Total N", fmt_int(n_total)), ("Mann-Whitney U", fmt(r["u2"], 3)),
                        ("Wilcoxon W", fmt(r["r2"], 3)), ("Test Statistic", fmt(r["u2"], 3)),
                        ("Standard Error", fmt(r["se"], 3)),
                        ("Standardized Test Statistic", fmt(r["z2"], 3)),
                        ("Asymptotic Sig.(2-sided test)", fmt_p(r["p"]))]
                if exact:
                    rows.append(("Exact Sig.(2-sided test)", fmt_p(r["p_exact"])))
                details.append(_summary_table(
                    f"Independent-Samples Mann-Whitney U Test Summary ({title})", rows))
            elif test == "ks":
                r = ks_two_sample_test(a, b)
                hyps.append(_Hyp(same, "Independent-Samples Kolmogorov-Smirnov Test", r["p"]))
                details.append(_summary_table(
                    f"Independent-Samples Kolmogorov-Smirnov Test Summary ({title})", [
                        ("Total N", fmt_int(n_total)),
                        ("Most Extreme Differences: Absolute", fmt(r["absolute"], 3)),
                        ("Most Extreme Differences: Positive", fmt(r["positive"], 3)),
                        ("Most Extreme Differences: Negative", fmt(r["negative"], 3)),
                        ("Test Statistic", fmt(r["z"], 3)),
                        ("Asymptotic Sig.(2-sided test)", fmt_p(r["p"]))]))
            elif test == "ww":
                r = wald_wolfowitz_test(a, b)
                d = r["max"]  # with ties the conservative (maximum possible) number of runs
                hyps.append(_Hyp(same, "Independent-Samples Wald-Wolfowitz Runs Test",
                                 d["p_exact"], exact=True))
                details.append(_summary_table(
                    f"Independent-Samples Wald-Wolfowitz Runs Test Summary ({title})", [
                        ("Total N", fmt_int(n_total)), ("Test Statistic", fmt_int(d["runs"])),
                        ("Standardized Test Statistic", fmt(d["z"], 3)),
                        ("Exact Sig.(1-sided test)", fmt_p(d["p_exact"]))],
                    [] if r["exact_runs"] else [
                        f"There are {r['ties']} inter-group ties involving {r['tie_cases']} "
                        f"records; the number of runs ranges from {r['min']['runs']} to "
                        f"{r['max']['runs']} (the maximum is used)."]))
            elif test == "moses":
                r = moses_test(a, b, moses_trim)
                if "span" not in r:
                    continue
                hyps.append(_Hyp(f"The range of {title} is the same across categories of {gt}.",
                                 "Independent-Samples Moses Test of Extreme Reaction",
                                 r["p_trimmed"], exact=True))
                details.append(_summary_table(
                    f"Independent-Samples Moses Test of Extreme Reaction Summary ({title})", [
                        ("Total N", fmt_int(n_total)),
                        ("Observed Control Group: Test Statistic", fmt_int(r["span"])),
                        ("Observed Control Group: Exact Sig.(1-sided test)",
                         fmt_p(r["p_span"])),
                        ("Trimmed Control Group: Test Statistic", fmt_int(r["trimmed"])),
                        ("Trimmed Control Group: Exact Sig.(1-sided test)",
                         fmt_p(r["p_trimmed"])),
                        ("Outliers Trimmed From Each End", fmt_int(r["h"]))],
                    [f"{labels[0]} is used as the control group."]))
            elif test == "hl":
                r = hodges_lehmann_two(a, b, level)
                details.append(_ci_table([("Independent-Samples Hodges-Lehman Median "
                                           "Difference",
                                           f"Difference between medians of {title} across "
                                           f"categories of {gt} ({labels[0]} - {labels[1]}).",
                                           r)], level, var.decimals))
            elif test == "kw":
                r = kruskal_test(samples)
                hyps.append(_Hyp(same, "Independent-Samples Kruskal-Wallis Test", r["p"]))
                details.append(_summary_table(
                    f"Independent-Samples Kruskal-Wallis Test Summary ({title})", [
                        ("Total N", fmt_int(r["n"])), ("Test Statistic", fmt(r["h"], 3) + "ᵃ"),
                        ("Degree Of Freedom", fmt_int(r["df"])),
                        ("Asymptotic Sig.(2-sided test)", fmt_p(r["p"]))],
                    ["a. The test statistic is adjusted for ties."]))
                if multiple == "all" and len(cats) > 2:
                    details.append(_pairwise_table(f"Pairwise Comparisons of {gt} ({title})",
                                                   dunn_pairwise(samples, labels), alpha))
            elif test == "median":
                r = median_test(samples)
                hyps.append(_Hyp(f"The medians of {title} are the same across categories of "
                                 f"{gt}.", "Independent-Samples Median Test", r["p"]))
                details.append(_summary_table(
                    f"Independent-Samples Median Test Summary ({title})", [
                        ("Total N", fmt_int(r["n"])), ("Median", fmt(r["median"], 3)),
                        ("Test Statistic", fmt(r["chi2"], 3)),
                        ("Degree Of Freedom", fmt_int(r["df"])),
                        ("Asymptotic Sig.(2-sided test)", fmt_p(r["p"]))]))
            elif test == "jt":
                r = jonckheere_test(samples)
                hyps.append(_Hyp(same, "Independent-Samples Jonckheere-Terpstra Test for "
                                       "Ordered Alternatives", r["p"]))
                details.append(_summary_table(
                    f"Independent-Samples Jonckheere-Terpstra Test for Ordered Alternatives "
                    f"Summary ({title})", [
                        ("Total N", fmt_int(r["n"])), ("Test Statistic", fmt(r["j"], 3)),
                        ("Standard Error", fmt(r["sd"], 3)),
                        ("Standardized Test Statistic", fmt(r["z"], 3)),
                        ("Asymptotic Sig.(2-sided test)", fmt_p(r["p"]))],
                    [f"Groups are ordered by the values of {gt}: " + " < ".join(labels) + "."]))
        if charts_on:
            details.append(("image", f"{title} by {gt}",
                            charts.boxplot(samples, labels, f"{title} by {gt}", title)))
    return _assemble(section, hyps, details, alpha)


def auto_related(variables, values: set, k: int) -> str:
    categorical = any(is_categorical(v, False) for v in variables)
    binary = len(values) == 2
    if categorical:
        if k == 2:
            return "mcnemar" if binary else "marginal"
        return "cochran" if binary else "friedman"
    return "wilcoxon" if k == 2 else "friedman"


def related_samples(ds, fields: list[str], tests: tuple | None = None, alpha: float = 0.05,
                    level: float = 0.95, multiple: str = "all", charts_on: bool = True):
    """SPSS NPTESTS /RELATED (records with a missing value in any field are excluded, as the
    related-samples tests need complete records)."""
    from rsstatchem.core.output import Section
    from rsstatchem.core.stats import charts
    from rsstatchem.core.stats.common import fmt, fmt_int, fmt_p

    section = Section("Nonparametric Tests — Related Samples")
    spec = "AUTOMATIC" if tests is None else " ".join(t.upper() for t in tests)
    section.log(f"NPTESTS\n  /RELATED TEST ({' '.join(fields)}) {spec}\n"
                f"  /CRITERIA ALPHA={alpha:g} CILEVEL={level * 100:g}.")
    raw = pd.concat({n: ds.analysis_series(n) for n in fields}, axis=1).dropna()
    titles = [ds.var(n).title for n in fields]
    k = len(fields)
    if len(raw) == 0 or k < 2:
        section.note("Warning", "At least two fields and one complete record are needed.",
                     style="warning")
        return section
    variables = [ds.var(n) for n in fields]
    values = set()
    for n in fields:
        values |= set(raw[n].tolist())
    chosen = (auto_related(variables, values, k),) if tests is None else tests
    numeric_ok = all(v.type == "numeric" for v in variables)
    x = raw.apply(pd.to_numeric, errors="coerce").to_numpy(float) if numeric_ok else None
    n = len(raw)
    pair_text = f"{titles[0]} and {titles[1]}" if k == 2 else ""
    all_text = ", ".join(titles[:-1]) + f" and {titles[-1]}"
    hyps, details = [], []
    rank_pairs_done = False
    for test in chosen:
        if test in ("sign", "wilcoxon", "mcnemar", "marginal", "hl") and k != 2:
            section.note("", f"The {test.upper()} test needs exactly 2 fields and was skipped.")
            continue
        if test in ("sign", "wilcoxon", "marginal", "hl", "friedman", "kendall") and x is None:
            continue
        if test == "sign":
            r = sign_test(x[:, 1] - x[:, 0])
            hyps.append(_Hyp(f"The median of differences between {pair_text} equals 0.",
                             "Related-Samples Sign Test", r["p"], exact=r["exact"]))
            details.append(_summary_table("Related-Samples Sign Test Summary", [
                ("Total N", fmt_int(r["total"])), ("Test Statistic", fmt_int(r["pos"])),
                ("Standard Error", fmt(0.5 * math.sqrt(r["n"]), 3) if r["n"] else ""),
                ("Standardized Test Statistic", fmt(r.get("z_np"), 3)),
                ("Exact Sig.(2-sided test)" if r["exact"] else "Asymptotic Sig.(2-sided test)",
                 fmt_p(r["p"]))]))
        elif test == "wilcoxon":
            d = x[:, 1] - x[:, 0]
            r = wilcoxon_test(d)
            hyps.append(_Hyp(f"The median of differences between {pair_text} equals 0.",
                             "Related-Samples Wilcoxon Signed Rank Test", r["p"]))
            details.append(_summary_table("Related-Samples Wilcoxon Signed Rank Test Summary", [
                ("Total N", fmt_int(r["total"])), ("Test Statistic", fmt(r["r_pos"], 3)),
                ("Standard Error", fmt(r.get("se"), 3)),
                ("Standardized Test Statistic", fmt(r.get("z_pos"), 3)),
                ("Asymptotic Sig.(2-sided test)", fmt_p(r["p"]))]))
            if charts_on:
                details.append(("image", f"Differences ({titles[1]} - {titles[0]})",
                                charts.histogram(d, f"{titles[1]} - {titles[0]}",
                                                 "Difference", False)))
        elif test == "mcnemar":
            r = mcnemar_test(raw.iloc[:, 0].to_numpy(), raw.iloc[:, 1].to_numpy())
            if "error" in r:
                section.note("Warning", "McNemar's test needs two fields with the same two "
                             "values.", style="warning")
                continue
            hyps.append(_Hyp(f"The distributions of different values across {pair_text} are "
                             "equally likely.", "Related-Samples McNemar Change Test", r["p"],
                             exact=r["exact"]))
            details.append(_summary_table("Related-Samples McNemar Change Test Summary", [
                ("Total N", fmt_int(r["n"])), ("Test Statistic", fmt(r.get("chi2"), 3)),
                ("Degree Of Freedom", "1"),
                ("Exact Sig.(2-sided test)" if r["exact"] else "Asymptotic Sig.(2-sided test)",
                 fmt_p(r["p"]))]))
        elif test == "marginal":
            r = marginal_homogeneity_test(x[:, 0], x[:, 1])
            hyps.append(_Hyp(f"The distributions of different values across {pair_text} are "
                             "equally likely.", "Related-Samples Marginal Homogeneity Test",
                             r["p"]))
            details.append(_summary_table("Related-Samples Marginal Homogeneity Test Summary", [
                ("Total N", fmt_int(n)), ("Test Statistic", fmt(r["obs"], 3)),
                ("Standard Error", fmt(r["sd"], 3)),
                ("Standardized Test Statistic", fmt(r["z"], 3)),
                ("Asymptotic Sig.(2-sided test)", fmt_p(r["p"]))]))
        elif test == "hl":
            r = hodges_lehmann_one(x[:, 1] - x[:, 0], level)
            details.append(_ci_table([("Related-Samples Hodges-Lehman Median Difference",
                                       f"Median of the difference between {pair_text} "
                                       f"({titles[1]} - {titles[0]}).", r)], level,
                                     variables[0].decimals))
        elif test in ("friedman", "kendall"):
            r = friedman_test(x)
            if test == "friedman":
                hyps.append(_Hyp(f"The distributions of {all_text} are the same.",
                                 "Related-Samples Friedman's Two-Way Analysis of Variance by "
                                 "Ranks", r["p"]))
                details.append(_summary_table(
                    "Related-Samples Friedman's Two-Way Analysis of Variance by Ranks Summary", [
                        ("Total N", fmt_int(n)), ("Test Statistic", fmt(r["chi2"], 3)),
                        ("Degree Of Freedom", fmt_int(r["df"])),
                        ("Asymptotic Sig.(2-sided test)", fmt_p(r["p"]))]))
            else:
                hyps.append(_Hyp(f"The distributions of {all_text} are the same.",
                                 "Related-Samples Kendall's Coefficient of Concordance", r["p"]))
                details.append(_summary_table(
                    "Related-Samples Kendall's Coefficient of Concordance Summary", [
                        ("Total N", fmt_int(n)),
                        ("Kendall's Coefficient of Concordance", fmt(r["w"], 3)),
                        ("Test Statistic", fmt(r["chi2"], 3)),
                        ("Degree Of Freedom", fmt_int(r["df"])),
                        ("Asymptotic Sig.(2-sided test)", fmt_p(r["p"]))]))
            if multiple == "all" and k > 2 and not rank_pairs_done:
                rank_pairs_done = True
                details.append(_pairwise_table("Pairwise Comparisons (mean ranks)",
                                               friedman_pairwise(x, titles), alpha))
        elif test == "cochran":
            if len(values) != 2:
                section.note("Warning", "Cochran's Q needs dichotomous fields with the same "
                             "two values.", style="warning")
                continue
            success = sorted(values, key=lambda v: (isinstance(v, str), v))[-1]
            b = (raw.to_numpy() == success).astype(float)
            r = cochran_test(b)
            hyps.append(_Hyp(f"The distributions of {all_text} are the same.",
                             "Related-Samples Cochran's Q Test", r["p"]))
            details.append(_summary_table("Related-Samples Cochran's Q Test Summary", [
                ("Total N", fmt_int(n)), ("Test Statistic", fmt(r["q"], 3)),
                ("Degree Of Freedom", fmt_int(r["df"])),
                ("Asymptotic Sig.(2-sided test)", fmt_p(r["p"]))],
                [f"{_code(success)} is treated as a success."]))
            if multiple == "all" and k > 2:
                details.append(_pairwise_table("Pairwise Comparisons (Cochran's Q)",
                                               cochran_pairwise(b, titles), alpha))
    if charts_on and x is not None and k > 2:
        details.append(("image", "Related samples", charts.boxplot(
            [x[:, j] for j in range(k)], titles, "Related Samples", "Value")))
    return _assemble(section, hyps, details, alpha)
