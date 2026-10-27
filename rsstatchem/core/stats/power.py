"""Analyze > Power Analysis, mirroring SPSS Statistics 27-31 (POWER command).

Procedures (each can *estimate sample size* for one or more target powers, or *estimate power*
for one or more sample sizes, two-sided ("nondirectional") or one-sided ("directional")):

* Means: One-Sample T Test, Paired-Samples T Test, Independent-Samples T Test (pooled or
  separate variances, group-size ratio), One-Way ANOVA (overall F test, user contrasts and
  pairwise differences with LSD / Bonferroni / Sidak), all on noncentral t / F distributions as
  in SPSS and G*Power. Precision analysis (sample size for a confidence-interval half-width).
* Proportions: One-Sample Binomial, Related-Samples Binomial (McNemar on the discordant pairs),
  Independent-Samples Binomial (chi-square / pooled z, unpooled z, Fisher's exact test), by
  binomial enumeration (exact) or normal approximation with an optional continuity correction.
* Correlations: Pearson, Spearman (Fisher z with the Bonett-Wright, Fieller-Hartley-Pearson or
  Caruso-Cliff variance) and partial correlation (Fisher z with n - 3 - k), optional bias
  correction.
* Regression: Univariate Linear (fixed predictors, test of a set of predictors via the partial
  R-squared, noncentral F with lambda = f^2 N as in G*Power).

Results agree with G*Power (Faul et al., 2007), ``statsmodels.stats.power`` and ``pwr``. Power
curves (power by sample size, power by effect size) are drawn like SPSS's Plot sub-dialog.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import fmt, fmt_int
from rsstatchem.core.stats.wp4_util import figure, png

MAX_N = 1_000_000


def _f(x, d: int = 3) -> str:
    return fmt(x, d)


def _g(x) -> str:
    """Input values as typed (SPSS echoes the test assumptions without trailing zeros)."""
    if x is None:
        return ""
    x = float(x)
    if x.is_integer() and abs(x) < 1e12:
        return str(int(x))
    text = f"{x:.6g}"
    if text.startswith("0."):
        text = text[1:]
    elif text.startswith("-0."):
        text = "-" + text[2:]
    return text


def _t_power(ncp: float, df: float, alpha: float, sided: str) -> float:
    """Power of a t test with noncentrality ``ncp`` (one-sided in the direction of ncp)."""
    if df <= 0:
        return float("nan")
    if sided == "two":
        tc = stats.t.ppf(1 - alpha / 2, df)
        return float(stats.nct.sf(tc, df, ncp) + stats.nct.cdf(-tc, df, ncp))
    tc = stats.t.ppf(1 - alpha, df)
    return float(stats.nct.sf(tc, df, abs(ncp)))


def _f_power(lam: float, df1: float, df2: float, alpha: float) -> float:
    if df2 <= 0:
        return float("nan")
    fc = stats.f.ppf(1 - alpha, df1, df2)
    return float(stats.ncf.sf(fc, df1, df2, lam)) if lam > 0 else alpha


def _z_power(delta: float, se0: float, se1: float, alpha: float, sided: str) -> float:
    """Normal approximation: test statistic delta/se0 under H0 sd se0, under H1 sd se1."""
    delta = abs(delta)
    if sided == "two":
        z = stats.norm.ppf(1 - alpha / 2)
        return float(stats.norm.cdf((delta - z * se0) / se1) +
                     stats.norm.cdf((-delta - z * se0) / se1))
    z = stats.norm.ppf(1 - alpha)
    return float(stats.norm.cdf((delta - z * se0) / se1))


# =========================================================================== designs
@dataclass
class Design:
    """One power-analysis test. ``power(n)`` takes the design's sample-size unit."""
    alpha: float = 0.05
    sided: str = "two"
    min_n: int = 2
    title: str = ""
    monotone: bool = True

    def power(self, n: int) -> float:  # pragma: no cover - abstract
        raise NotImplementedError

    def sizes(self, n: int) -> list[int]:
        return [n]

    def size_labels(self) -> list[str]:
        return ["N"]

    def assumptions(self) -> list[tuple[str, str]]:
        return []

    def effect(self) -> float:
        return float("nan")

    def footnotes(self) -> list[str]:
        return []

    def with_effect(self, effect: float) -> "Design":
        """Same design with another standardized effect size (for power-by-effect plots)."""
        raise NotImplementedError


@dataclass
class OneSampleT(Design):
    mean: float = 0.5
    null: float = 0.0
    sd: float = 1.0
    title: str = "Test for Mean"

    def effect(self) -> float:
        return (self.mean - self.null) / self.sd

    def power(self, n: int) -> float:
        return _t_power(self.effect() * math.sqrt(n), n - 1, self.alpha, self.sided)

    def assumptions(self):
        return [("Population Mean", _g(self.mean)), ("Null Value", _g(self.null)),
                ("Std. Dev.", _g(self.sd))]

    def footnotes(self):
        return ["Based on noncentral t-distribution."]

    def with_effect(self, effect):
        return OneSampleT(self.alpha, self.sided, self.min_n, self.title, True,
                          self.null + effect * self.sd, self.null, self.sd)


@dataclass
class PairedT(OneSampleT):
    title: str = "Test for Paired-Samples Means"

    def assumptions(self):
        return [("Mean Difference", _g(self.mean - self.null)), ("Std. Dev. of Difference",
                                                                 _g(self.sd))]

    def with_effect(self, effect):
        return PairedT(self.alpha, self.sided, self.min_n, self.title, True,
                       self.null + effect * self.sd, self.null, self.sd)


def paired_sd(sd1: float, sd2: float, rho: float) -> float:
    """SD of differences from the two SDs and their correlation (SPSS paired-t alternative)."""
    return math.sqrt(sd1 ** 2 + sd2 ** 2 - 2 * rho * sd1 * sd2)


@dataclass
class IndependentT(Design):
    mean1: float = 0.5
    mean2: float = 0.0
    sd1: float = 1.0
    sd2: float | None = None      # None = pooled standard deviation sd1
    ratio: float = 1.0            # N2 / N1
    title: str = "Test for Mean Difference"

    def n2(self, n1: int) -> int:
        return max(1, int(math.ceil(self.ratio * n1 - 1e-9)))

    def sizes(self, n):
        return [n, self.n2(n)]

    def size_labels(self):
        return ["N1", "N2"]

    def effect(self) -> float:
        if self.sd2 is None:
            return (self.mean1 - self.mean2) / self.sd1
        return (self.mean1 - self.mean2) / math.sqrt((self.sd1 ** 2 + self.sd2 ** 2) / 2)

    def power(self, n: int) -> float:
        n1, n2 = n, self.n2(n)
        diff = self.mean1 - self.mean2
        if self.sd2 is None:
            ncp = diff / (self.sd1 * math.sqrt(1 / n1 + 1 / n2))
            df = n1 + n2 - 2
        else:
            v1, v2 = self.sd1 ** 2 / n1, self.sd2 ** 2 / n2
            ncp = diff / math.sqrt(v1 + v2)
            df = (v1 + v2) ** 2 / (v1 ** 2 / (n1 - 1) + v2 ** 2 / (n2 - 1)) if n1 > 1 and \
                n2 > 1 else 0
        return _t_power(ncp, df, self.alpha, self.sided)

    def assumptions(self):
        out = [("Mean Difference", _g(self.mean1 - self.mean2))]
        if self.sd2 is None:
            out.append(("Pooled Std. Dev.", _g(self.sd1)))
        else:
            out += [("Std. Dev. 1", _g(self.sd1)), ("Std. Dev. 2", _g(self.sd2))]
        return out

    def footnotes(self):
        notes = ["Based on noncentral t-distribution."]
        if self.sd2 is not None:
            notes.append("Group variances are not assumed to be equal (Welch-Satterthwaite "
                         "degrees of freedom).")
        if self.ratio != 1:
            notes.append(f"Group size ratio N2/N1 = {_g(self.ratio)}.")
        return notes

    def with_effect(self, effect):
        s = self.sd1 if self.sd2 is None else math.sqrt((self.sd1 ** 2 + self.sd2 ** 2) / 2)
        return IndependentT(self.alpha, self.sided, self.min_n, self.title, True,
                            self.mean2 + effect * s, self.mean2, self.sd1, self.sd2, self.ratio)


@dataclass
class OneWayAnova(Design):
    means: tuple = (0.0, 0.5, 1.0)
    sd: float = 1.0
    weights: tuple | None = None   # relative group sizes (integers), None = equal
    title: str = "Overall Test"
    sided: str = "two"

    def _w(self) -> np.ndarray:
        w = np.ones(len(self.means)) if not self.weights else np.asarray(self.weights, float)
        return w

    def group_sizes(self, m: int) -> np.ndarray:
        w = self._w()
        return np.maximum(1, np.ceil(m * w / w.min() - 1e-9)).astype(int)

    def sizes(self, n):
        return [int(self.group_sizes(n).sum())]

    def effect(self) -> float:
        w = self._w() / self._w().sum()
        mu = np.asarray(self.means, float)
        mbar = float((w * mu).sum())
        return float(math.sqrt((w * (mu - mbar) ** 2).sum()) / self.sd)

    def power(self, n: int) -> float:
        ni = self.group_sizes(n)
        mu = np.asarray(self.means, float)
        mbar = float((ni * mu).sum() / ni.sum())
        lam = float((ni * (mu - mbar) ** 2).sum() / self.sd ** 2)
        k = len(mu)
        return _f_power(lam, k - 1, ni.sum() - k, self.alpha)

    def assumptions(self):
        return [("Group Means", ", ".join(_g(m) for m in self.means)),
                ("Pooled Std. Dev.", _g(self.sd))]

    def footnotes(self):
        notes = ["Based on noncentral F-distribution."]
        if self.weights and len(set(self.weights)) > 1:
            notes.append("Group sizes are proportional to " +
                         ":".join(_g(w) for w in self.weights) + ".")
        return notes

    def with_effect(self, effect):
        e0 = self.effect()
        mu = np.asarray(self.means, float)
        mbar = float((self._w() * mu).sum() / self._w().sum())
        scale = effect / e0 if e0 > 0 else 0.0
        return OneWayAnova(self.alpha, self.sided, self.min_n, self.title, True,
                           tuple(mbar + (mu - mbar) * scale), self.sd, self.weights)


@dataclass
class AnovaContrast(OneWayAnova):
    coefficients: tuple = (-1.0, 0.0, 1.0)
    adjust_k: int = 1               # number of comparisons for Bonferroni / Sidak
    adjust: str = "none"            # none (LSD) | bonferroni | sidak
    label: str = "Contrast"

    def _alpha(self) -> float:
        if self.adjust == "bonferroni":
            return self.alpha / self.adjust_k
        if self.adjust == "sidak":
            return 1 - (1 - self.alpha) ** (1 / self.adjust_k)
        return self.alpha

    def effect(self) -> float:
        c = np.asarray(self.coefficients, float)
        return float((c * np.asarray(self.means, float)).sum() / self.sd)

    def power(self, n: int) -> float:
        ni = self.group_sizes(n)
        c = np.asarray(self.coefficients, float)
        est = float((c * np.asarray(self.means, float)).sum())
        se = self.sd * math.sqrt(float((c ** 2 / ni).sum()))
        return _t_power(est / se, ni.sum() - len(ni), self._alpha(), self.sided)

    def assumptions(self):
        return [("Contrast Value", _g(sum(c * m for c, m in zip(self.coefficients,
                                                                    self.means)))),
                ("Pooled Std. Dev.", _g(self.sd))]

    def footnotes(self):
        notes = ["Based on noncentral t-distribution."]
        if self.adjust != "none":
            notes.append(f"{self.adjust.title()} adjustment for {self.adjust_k} comparisons.")
        return notes


@dataclass
class OneSampleBinomial(Design):
    p: float = 0.6
    p0: float = 0.5
    method: str = "exact"           # exact (binomial enumeration) | normal
    continuity: bool = False
    title: str = "Test for One-Sample Proportion"

    def __post_init__(self):
        self.monotone = self.method != "exact"
        self.min_n = 1

    def effect(self) -> float:
        return cohen_h(self.p, self.p0)

    def power(self, n: int) -> float:
        if self.method == "normal":
            delta = abs(self.p - self.p0) - (0.5 / n if self.continuity else 0.0)
            return _z_power(delta, math.sqrt(self.p0 * (1 - self.p0) / n),
                            math.sqrt(self.p * (1 - self.p) / n), self.alpha, self.sided)
        return exact_binomial_power(n, self.p, self.p0, self.alpha, self.sided)

    def assumptions(self):
        return [("Population Proportion", _g(self.p)), ("Null Value", _g(self.p0))]

    def footnotes(self):
        if self.method == "normal":
            return ["Based on the normal approximation" +
                    (" with continuity correction." if self.continuity else ".")]
        return ["Based on binomial enumeration (exact test). Power does not increase "
                "monotonically with N; the smallest N reaching the target is reported."]

    def with_effect(self, effect):
        p = float(np.clip(math.sin(math.asin(math.sqrt(self.p0)) + effect / 2) ** 2, 1e-6,
                          1 - 1e-6))
        return OneSampleBinomial(self.alpha, self.sided, 1, self.title, True, p, self.p0,
                                 self.method, self.continuity)


def cohen_h(p1: float, p2: float) -> float:
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def _binom_reject(n: int, p0: float, alpha: float, sided: str, upper: bool = True):
    """Rejection region (lower cut, upper cut) of the exact binomial test (equal tails)."""
    k = np.arange(n + 1)
    cdf = stats.binom.cdf(k, n, p0)
    sf = stats.binom.sf(k - 1, n, p0)
    a = alpha / 2 if sided == "two" else alpha
    lo = k[cdf <= a]
    hi = k[sf <= a]
    lo_cut = int(lo.max()) if len(lo) else -1
    hi_cut = int(hi.min()) if len(hi) else n + 1
    if sided != "two":
        if upper:
            lo_cut = -1
        else:
            hi_cut = n + 1
    return lo_cut, hi_cut


def exact_binomial_power(n: int, p: float, p0: float, alpha: float, sided: str) -> float:
    lo, hi = _binom_reject(n, p0, alpha, sided, upper=p >= p0)
    out = 0.0
    if lo >= 0:
        out += stats.binom.cdf(lo, n, p)
    if hi <= n:
        out += stats.binom.sf(hi - 1, n, p)
    return float(out)


@dataclass
class RelatedBinomial(Design):
    p12: float = 0.25             # P(first = success, second = failure)
    p21: float = 0.10             # P(first = failure, second = success)
    method: str = "exact"
    continuity: bool = False
    title: str = "Test for Related-Samples Proportions"

    def __post_init__(self):
        self.monotone = self.method != "exact"

    def effect(self) -> float:
        return self.p12 - self.p21

    def power(self, n: int) -> float:
        psi, delta = self.p12 + self.p21, self.p12 - self.p21
        if self.method == "normal":
            d = abs(delta) - (1 / n if self.continuity else 0.0)
            return _z_power(d, math.sqrt(psi / n), math.sqrt((psi - delta ** 2) / n),
                            self.alpha, self.sided)
        m = np.arange(n + 1)
        pm = stats.binom.pmf(m, n, psi)
        q = self.p12 / psi
        pw = np.array([exact_binomial_power(int(k), q, 0.5, self.alpha, self.sided) if k else
                       0.0 for k in m])
        return float((pm * pw).sum())

    def assumptions(self):
        return [("Discordant Proportion (1,2)", _g(self.p12)),
                ("Discordant Proportion (2,1)", _g(self.p21))]

    def footnotes(self):
        if self.method == "normal":
            return ["McNemar test, normal approximation (Connor, 1987)" +
                    (" with continuity correction." if self.continuity else ".")]
        return ["McNemar test by binomial enumeration: exact binomial test on the discordant "
                "pairs, averaged over their binomial distribution."]


@dataclass
class IndependentBinomial(Design):
    p1: float = 0.6
    p2: float = 0.4
    ratio: float = 1.0
    test: str = "chisq"           # chisq (pooled z) | unpooled (Wald z) | fisher
    method: str = "normal"        # normal | exact
    continuity: bool = False
    title: str = "Test for Independent-Samples Proportions"

    def __post_init__(self):
        self.monotone = self.method == "normal" and self.test != "fisher"

    def n2(self, n1: int) -> int:
        return max(1, int(math.ceil(self.ratio * n1 - 1e-9)))

    def sizes(self, n):
        return [n, self.n2(n)]

    def size_labels(self):
        return ["N1", "N2"]

    def effect(self) -> float:
        return cohen_h(self.p1, self.p2)

    def power(self, n: int) -> float:
        n1, n2 = n, self.n2(n)
        if self.test == "fisher" or self.method == "exact":
            return _enumerate_2x2(n1, n2, self.p1, self.p2, self.alpha, self.sided, self.test,
                                  self.continuity)
        delta = abs(self.p1 - self.p2)
        if self.continuity:
            delta -= 0.5 * (1 / n1 + 1 / n2)
        pbar = (n1 * self.p1 + n2 * self.p2) / (n1 + n2)
        se1 = math.sqrt(self.p1 * (1 - self.p1) / n1 + self.p2 * (1 - self.p2) / n2)
        se0 = math.sqrt(pbar * (1 - pbar) * (1 / n1 + 1 / n2)) if self.test == "chisq" else se1
        return _z_power(delta, se0, se1, self.alpha, self.sided)

    def assumptions(self):
        return [("Proportion 1", _g(self.p1)), ("Proportion 2", _g(self.p2))]

    def footnotes(self):
        test = {"chisq": "Pearson chi-square test (pooled variance)",
                "unpooled": "z test with unpooled variance",
                "fisher": "Fisher's exact test"}[self.test]
        how = "binomial enumeration" if (self.method == "exact" or self.test == "fisher") else \
            "the normal approximation"
        notes = [f"{test}; power by {how}" + (" with continuity correction." if
                                                self.continuity else ".")]
        if self.ratio != 1:
            notes.append(f"Group size ratio N2/N1 = {_g(self.ratio)}.")
        return notes

    def with_effect(self, effect):
        p1 = float(np.clip(math.sin(math.asin(math.sqrt(self.p2)) + effect / 2) ** 2, 1e-6,
                           1 - 1e-6))
        return IndependentBinomial(self.alpha, self.sided, self.min_n, self.title, True, p1,
                                   self.p2, self.ratio, self.test, self.method, self.continuity)


def _enumerate_2x2(n1, n2, p1, p2, alpha, sided, test, continuity) -> float:
    """Exact power: sum P(x1) P(x2) over the outcomes the chosen test rejects."""
    x1 = np.arange(n1 + 1)[:, None]
    x2 = np.arange(n2 + 1)[None, :]
    prob = stats.binom.pmf(x1, n1, p1) * stats.binom.pmf(x2, n2, p2)
    if test == "fisher":
        pval = _fisher_pvalues(n1, n2, sided, p1 >= p2)
        return float(prob[pval <= alpha + 1e-12].sum())
    r1, r2 = x1 / n1, x2 / n2
    diff = r1 - r2
    if test == "chisq":
        pbar = (x1 + x2) / (n1 + n2)
        se = np.sqrt(pbar * (1 - pbar) * (1 / n1 + 1 / n2))
    else:
        se = np.sqrt(r1 * (1 - r1) / n1 + r2 * (1 - r2) / n2)
    cc = 0.5 * (1 / n1 + 1 / n2) if continuity else 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0, (np.abs(diff) - cc).clip(0) * np.sign(diff) / se, 0.0)
    if sided == "two":
        reject = np.abs(z) >= stats.norm.ppf(1 - alpha / 2)
    else:
        zc = stats.norm.ppf(1 - alpha)
        reject = z >= zc if p1 >= p2 else z <= -zc
    return float(prob[reject].sum())


def _fisher_pvalues(n1: int, n2: int, sided: str, greater: bool) -> np.ndarray:
    """Fisher exact p-value for every 2x2 outcome (x1 of n1, x2 of n2)."""
    out = np.ones((n1 + 1, n2 + 1))
    for t in range(n1 + n2 + 1):
        lo, hi = max(0, t - n2), min(n1, t)
        k = np.arange(lo, hi + 1)
        pmf = stats.hypergeom.pmf(k, n1 + n2, n1, t)
        if sided == "two":
            pv = np.array([pmf[pmf <= p * (1 + 1e-7)].sum() for p in pmf])
        elif greater:
            pv = np.cumsum(pmf[::-1])[::-1]
        else:
            pv = np.cumsum(pmf)
        out[k, t - k] = np.minimum(pv, 1.0)
    return out


@dataclass
class Correlation(Design):
    rho: float = 0.3
    rho0: float = 0.0
    kind: str = "pearson"          # pearson | spearman | partial
    partial_k: int = 0
    spearman_var: str = "bonett"    # bonett | fieller | caruso
    bias: bool = False
    title: str = "Test for Correlation"

    def __post_init__(self):
        self.min_n = 4 + (self.partial_k if self.kind == "partial" else 0)

    def effect(self) -> float:
        return self.rho

    def _z(self, r: float, n: int) -> float:
        z = math.atanh(r)
        if self.bias:
            z += r / (2 * (n - 1))
        return z

    def _var(self, n: int, z: float) -> float:
        if self.kind == "partial":
            return 1 / (n - 3 - self.partial_k)
        if self.kind == "spearman":
            if self.spearman_var == "fieller":
                return 1.06 / (n - 3)
            if self.spearman_var == "caruso":
                return 1 / (n - 2) + abs(z) / (6 * n + 4 * math.sqrt(n))
            return (1 + self.rho ** 2 / 2) / (n - 3)
        return 1 / (n - 3)

    def power(self, n: int) -> float:
        if n <= self.min_n - 1:
            return float("nan")
        z1, z0 = self._z(self.rho, n), self._z(self.rho0, n)
        se1 = math.sqrt(self._var(n, z1))
        if self.kind == "spearman":
            # Under H0 the variance uses the null correlation.
            r_saved = self.rho
            self.rho = self.rho0
            se0 = math.sqrt(self._var(n, z0))
            self.rho = r_saved
        else:
            se0 = se1
        return _z_power(z1 - z0, se0, se1, self.alpha, self.sided)

    def assumptions(self):
        out = [("Population Correlation", _g(self.rho)), ("Null Value", _g(self.rho0))]
        if self.kind == "partial":
            out.append(("Variables Partialled Out", str(self.partial_k)))
        return out

    def footnotes(self):
        notes = ["Based on Fisher's z-transformation and normal approximation" +
                 (" with bias adjustment." if self.bias else ".")]
        if self.kind == "spearman":
            notes.append({"bonett": "Variance by Bonett and Wright (2000).",
                          "fieller": "Variance by Fieller, Hartley and Pearson (1957).",
                          "caruso": "Variance by Caruso and Cliff (1997)."}[self.spearman_var])
        return notes

    def with_effect(self, effect):
        return Correlation(self.alpha, self.sided, self.min_n, self.title, True,
                           float(np.clip(effect, -0.999, 0.999)), self.rho0, self.kind,
                           self.partial_k, self.spearman_var, self.bias)


@dataclass
class LinearRegression(Design):
    r2_full: float = 0.13
    r2_reduced: float = 0.0
    predictors: int = 5           # total predictors in the full model
    tested: int = 5               # predictors being tested
    title: str = "Test for Linear Regression"

    def __post_init__(self):
        self.min_n = self.predictors + 2

    def partial_r2(self) -> float:
        return (self.r2_full - self.r2_reduced) / (1 - self.r2_reduced)

    def effect(self) -> float:
        return (self.r2_full - self.r2_reduced) / (1 - self.r2_full)

    def power(self, n: int) -> float:
        return _f_power(self.effect() * n, self.tested, n - self.predictors - 1, self.alpha)

    def assumptions(self):
        return [("Total Predictors", str(self.predictors)), ("Test Predictors",
                                                             str(self.tested)),
                ("Partial R²", _f(self.partial_r2(), 3))]

    def footnotes(self):
        return ["Based on noncentral F-distribution (fixed predictors, λ = f²·N)."]

    def with_effect(self, effect):
        r2 = self.r2_reduced + effect * (1 - self.r2_reduced) / (1 + effect)
        return LinearRegression(self.alpha, self.sided, self.min_n, self.title, True, r2,
                                self.r2_reduced, self.predictors, self.tested)


# =========================================================================== solving
def solve_n(design: Design, target: float, start: int | None = None) -> int | None:
    """Smallest sample-size unit with power >= target (None when not reachable)."""
    lo = max(design.min_n, 1)

    def pw(n):
        p = design.power(n)
        return -1.0 if p != p else p

    if pw(lo) >= target:
        return lo
    if design.monotone:
        hi = lo * 2
        while pw(hi) < target:
            lo, hi = hi, hi * 2
            if hi > MAX_N:
                return None
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if pw(mid) >= target:
                hi = mid
            else:
                lo = mid
        return hi
    # Non-monotone (exact) power: start below the normal-approximation answer and scan up.
    approx = solve_n(_normal_twin(design), target) if start is None else start
    n = max(design.min_n, int((approx or 10) * 0.6) - 2)
    limit = max(5000, (approx or 1000) * 4)
    while n <= limit:
        if pw(n) >= target:
            return n
        n += 1
    return None


def _normal_twin(design: Design) -> Design:
    import copy

    twin = copy.copy(design)
    if hasattr(twin, "method"):
        twin.method = "normal"
    if getattr(twin, "test", None) == "fisher":
        twin.test = "chisq"
        twin.continuity = True
    twin.monotone = True
    return twin


@dataclass
class PowerResult:
    rows: list = field(default_factory=list)   # (target or None, n unit, power)


def run_design(design: Design, estimate: str, powers=(0.8,), ns=()) -> PowerResult:
    res = PowerResult()
    if estimate == "n":
        for target in powers:
            n = solve_n(design, float(target))
            res.rows.append((float(target), n, design.power(n) if n else float("nan")))
    else:
        for n in ns:
            n = int(n)
            res.rows.append((None, n, design.power(n)))
    return res


# =========================================================================== precision
def precision_n(kind: str, sd: float, half_width: float, level: float = 0.95,
                ratio: float = 1.0, sd2: float | None = None) -> tuple[list[int], float]:
    """Sample size whose confidence interval half-width is at most ``half_width``.

    kind "one" (one-sample / paired, df n-1) or "two" (independent, df n1+n2-2)."""
    a = 1 - level

    def width(n):
        if kind == "one":
            return stats.t.ppf(1 - a / 2, n - 1) * sd / math.sqrt(n)
        n2 = max(1, int(math.ceil(ratio * n - 1e-9)))
        if sd2 is None:
            return stats.t.ppf(1 - a / 2, n + n2 - 2) * sd * math.sqrt(1 / n + 1 / n2)
        v1, v2 = sd ** 2 / n, sd2 ** 2 / n2
        df = (v1 + v2) ** 2 / (v1 ** 2 / (n - 1) + v2 ** 2 / (n2 - 1))
        return stats.t.ppf(1 - a / 2, df) * math.sqrt(v1 + v2)

    n = 2
    while width(n) > half_width:
        n = n * 2 if width(n * 2) > half_width else n + 1
        if n > MAX_N:
            return [], float("nan")
    lo = max(2, n // 2)
    for m in range(lo, n + 1):
        if width(m) <= half_width:
            n = m
            break
    sizes = [n] if kind == "one" else [n, max(1, int(math.ceil(ratio * n - 1e-9)))]
    return sizes, float(width(n))


# =========================================================================== output
PROCEDURES = {
    "one_t": "One-Sample T Test", "paired_t": "Paired-Samples T Test",
    "ind_t": "Independent-Samples T Test", "anova": "One-Way ANOVA",
    "one_binom": "One-Sample Binomial Test", "rel_binom": "Related-Samples Binomial Test",
    "ind_binom": "Independent-Samples Binomial Test", "pearson": "Pearson Correlation",
    "spearman": "Spearman Correlation", "partial": "Partial Correlation",
    "regression": "Univariate Linear Regression",
}


def power_table(designs: list[Design], results: list[PowerResult], estimate: str) -> Table:
    first = designs[0]
    sided_note = "Two-sided test." if first.sided == "two" else "One-sided test."
    if isinstance(first, (OneWayAnova, LinearRegression)) and not isinstance(first,
                                                                             AnovaContrast):
        sided_note = ""
    notes: list[str] = []
    letters = iter("abcdefghijklmnop")

    def mark(text: str) -> str:
        notes.append(text)
        return next(letters)

    from rsstatchem.core.stats.common import sup

    row_mark = sup(mark(sided_note)) if sided_note else ""
    power_mark = sup(mark(" ".join(first.footnotes())))
    effect_mark = sup(mark(_effect_note(first)))
    size_cols = first.size_labels()
    assume = first.assumptions()
    cols = [""] + size_cols + [f"Actual Power{power_mark}"]
    if estimate == "n":
        cols.append(("Test Assumptions", "Power"))
    cols += [("Test Assumptions", name) for name, _ in assume]
    cols += [("Test Assumptions", f"Effect Size{effect_mark}"), ("Test Assumptions", "Sig.")]
    rows = []
    for d, res in zip(designs, results):
        assume = d.assumptions()
        for target, n, pw in res.rows:
            sizes = d.sizes(n) if n else [None] * len(size_cols)
            row = [f"{d.title}{row_mark}"] + [fmt_int(s) if s else "." for s in sizes]
            row.append(_f(pw, 3) if n else ".")
            if estimate == "n":
                row.append(_g(target))
            row += [v for _, v in assume]
            row += [_f(d.effect(), 3), _g(d.alpha)]
            rows.append(row)
    footnotes = [f"{chr(97 + i)}. {t}" for i, t in enumerate(notes)]
    if any(n is None for r in results for _, n, _ in r.rows):
        footnotes.append("A sample size reaching the target power was not found (N ≤ "
                         f"{MAX_N:,}).")
    return Table("Power Analysis Table", cols, rows, footnotes=footnotes)


def _effect_note(d: Design) -> str:
    if isinstance(d, AnovaContrast):
        return "Effect size is the standardized contrast value."
    if isinstance(d, OneWayAnova):
        return "Effect size is Cohen's f."
    if isinstance(d, IndependentT):
        return "Effect size is Cohen's d (pooled or average standard deviation)."
    if isinstance(d, OneSampleT):
        return "Effect size is Cohen's d."
    if isinstance(d, RelatedBinomial):
        return "Effect size is the difference of the discordant proportions."
    if isinstance(d, (OneSampleBinomial, IndependentBinomial)):
        return "Effect size is Cohen's h."
    if isinstance(d, Correlation):
        return "Effect size is the population correlation."
    return "Effect size is Cohen's f²."


def power_curve_png(design: Design, n_lo: int, n_hi: int, marks=(), title: str = "",
                    xlabel: str = "Sample Size") -> bytes:
    n_lo = max(design.min_n, int(n_lo))
    n_hi = max(n_lo + 2, int(n_hi))
    step = max(1, (n_hi - n_lo) // 300)
    ns = np.arange(n_lo, n_hi + 1, step)
    pw = np.array([design.power(int(n)) for n in ns])
    fig = figure(6.0, 4.0)
    ax = fig.add_subplot(111)
    ax.plot([design.sizes(int(n))[0] if len(design.size_labels()) == 1 else n for n in ns], pw,
            color="#1f4e9c", lw=1.6)
    for target, n in marks:
        if n:
            ax.axhline(target, color="#c8102e", lw=0.8, ls="--")
            x = design.sizes(n)[0] if len(design.size_labels()) == 1 else n
            ax.plot([x], [design.power(n)], "o", color="#c8102e")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(xlabel if len(design.size_labels()) == 1 else "Sample Size (Group 1)")
    ax.set_ylabel("Power")
    ax.set_title(title or "Power by Sample Size", fontsize=10)
    ax.grid(alpha=0.3)
    return png(fig)


def effect_curve_png(design: Design, n: int, lo: float, hi: float, title: str = "") -> bytes:
    xs = np.linspace(lo, hi, 121)
    pw = []
    for e in xs:
        try:
            pw.append(design.with_effect(float(e)).power(n))
        except (ValueError, ZeroDivisionError, NotImplementedError):
            pw.append(np.nan)
    fig = figure(6.0, 4.0)
    ax = fig.add_subplot(111)
    ax.plot(xs, pw, color="#1f4e9c", lw=1.6)
    ax.axvline(design.effect(), color="#c8102e", lw=0.8, ls="--")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Effect Size")
    ax.set_ylabel("Power")
    sizes = design.sizes(n)
    label = ", ".join(f"{a} = {b}" for a, b in zip(design.size_labels(), sizes))
    ax.set_title(title or f"Power by Effect Size ({label})", fontsize=10)
    ax.grid(alpha=0.3)
    return png(fig)


def power_analysis(procedure: str, designs: list[Design], estimate: str = "n",
                   powers=(0.8,), ns=(), plots: dict | None = None,
                   precision: dict | None = None, syntax: str = "") -> Section:
    """Build the SPSS-style output. ``designs[0]`` is the main test; further designs are
    ANOVA contrasts / pairwise comparisons."""
    title = f"Power Analysis - {PROCEDURES.get(procedure, procedure)}"
    sec = Section(title)
    sec.log(syntax or f"POWER {procedure.upper()}")
    if estimate == "n":
        main_res = run_design(designs[0], "n", powers)
        results = [main_res] + [run_design(d, "n", powers) for d in designs[1:]]
    else:
        results = [run_design(d, "power", ns=ns) for d in designs]
    sec.table(power_table(designs[:1], results[:1], estimate))
    if len(designs) > 1:
        t = power_table(designs[1:], results[1:], estimate)
        t.title = "Power Analysis Table (Contrast Tests)"
        sec.table(t)
    if precision:
        sec.table(precision_table(**precision))
    plots = plots or {}
    d0 = designs[0]
    if plots.get("by_n"):
        marks = [(t, n) for t, n, _ in results[0].rows if t is not None]
        ref = max([n for _, n, _ in results[0].rows if n] or [d0.min_n + 20])
        lo, hi = plots.get("n_range") or (d0.min_n, max(int(ref * 2), d0.min_n + 20))
        sec.image("Power by Sample Size", power_curve_png(d0, lo, hi, marks))
    if plots.get("by_effect"):
        ref = [n for _, n, _ in results[0].rows if n]
        n = ref[0] if ref else (ns[0] if ns else 30)
        e = d0.effect()
        lo, hi = plots.get("effect_range") or _effect_range(d0, e)
        try:
            sec.image("Power by Effect Size", effect_curve_png(d0, n, lo, hi))
        except NotImplementedError:
            pass
    return sec


def _effect_range(d: Design, e: float) -> tuple[float, float]:
    if isinstance(d, Correlation):
        return (-0.95, 0.95) if d.sided == "two" else ((0.0, 0.95) if e >= 0 else (-0.95, 0))
    if isinstance(d, LinearRegression):
        return 0.0, max(0.5, 2.5 * e)
    if isinstance(d, OneWayAnova):
        return 0.0, max(0.6, 2.5 * e)
    span = max(1.0, 2.5 * abs(e))
    lo, hi = (-span, span) if d.sided == "two" else ((0.0, span) if e >= 0 else (-span, 0.0))
    if isinstance(d, (OneSampleBinomial, IndependentBinomial)):
        ref = d.p0 if isinstance(d, OneSampleBinomial) else d.p2
        a = 2 * math.asin(math.sqrt(ref))           # Cohen's h is bounded by p in (0, 1)
        lo, hi = max(lo, -a + 0.02), min(hi, math.pi - a - 0.02)
    return lo, hi


def precision_table(kind: str, sd: float, half_widths, level: float = 0.95,
                    ratio: float = 1.0, sd2: float | None = None) -> Table:
    labels = ["N"] if kind == "one" else ["N1", "N2"]
    rows = []
    for w in half_widths:
        sizes, actual = precision_n(kind, sd, float(w), level, ratio, sd2)
        rows.append(["Confidence Interval"] + [fmt_int(s) for s in sizes] +
                    [_f(actual, 3), _g(w), _g(level * 100) + "%", _g(sd)])
    cols = [""] + labels + ["Actual Half Width", ("Test Assumptions", "Half Width"),
                            ("Test Assumptions", "Confidence Level"),
                            ("Test Assumptions", "Std. Dev.")]
    return Table("Sample Size Based on Confidence Interval", cols, rows,
                 footnotes=["Half width of the t-based confidence interval computed with the "
                            "population standard deviation."])


def selftest() -> None:
    d = OneSampleT(mean=0.5, null=0.0, sd=1.0)
    assert solve_n(d, 0.8) == 34
    assert solve_n(IndependentT(mean1=0.5, mean2=0, sd1=1), 0.8) == 64
    a = 0.75 / 5 ** 0.5     # four equally spaced means with Cohen's f = .25
    assert solve_n(OneWayAnova(means=(-a, -a / 3, a / 3, a), sd=1.0), 0.8) == 45
    assert solve_n(Correlation(rho=0.3), 0.8) == 85
    sec = power_analysis("one_t", [d], "n", (0.8, 0.9), plots={"by_n": True,
                                                               "by_effect": True})
    assert sec.items[1].rows[0][1] == "34"
    power_analysis("one_binom", [OneSampleBinomial(p=0.65, p0=0.5)], "power", ns=(50, 80))
