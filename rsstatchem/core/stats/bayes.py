"""Analyze > Bayesian Statistics, mirroring SPSS Statistics 25+ (BAYES procedures).

Each procedure can characterize the posterior distribution, estimate a Bayes factor, or both
(SPSS's "Bayesian Analysis" choice):

* One Sample Normal / Related Samples Normal: posterior of the mean under a diffuse (Jeffreys)
  or conjugate normal / inverse-gamma prior, or with a known variance; JZS Bayes factor (Rouder
  et al., 2009) with an adjustable Cauchy scale.
* Independent Samples Normal: posterior of the mean difference (equal variances: Student t;
  unequal: Monte Carlo of the two posteriors); Bayes factor by Rouder's JZS method or Gonen et
  al.'s (2005) normal prior on the standardized effect.
* Pearson Correlation: Jeffreys' exact Bayes factor (Ly et al., 2016) with a stretched beta
  prior, posterior of rho on a grid.
* Linear Regression: ANOVA, model summary and Bayes factor against the intercept-only model by
  JZS / Zellner-Siow, Zellner's g or hyper-g priors (Liang et al., 2008); posterior of the
  coefficients and error variance under the reference prior (multivariate t / inverse gamma).
* One-way ANOVA: Bayes factor by Rouder et al.'s (2012) JZS fixed-effects prior or the g-prior
  family on the dummy-coded design; posterior of the group means and the error variance.
* One-way Repeated Measures ANOVA: BIC-approximated Bayes factor (Wagenmakers, 2007; Masson,
  2011) and pairwise JZS paired-samples Bayes factors.
* One Sample Binomial (beta prior) and One Sample Poisson (gamma prior): conjugate posteriors
  and Bayes factors against a point null.
* Log-Linear Models (two-way tables): Gunel-Dickey Bayes factor for independence under joint or
  independent multinomial sampling (Jamil et al., 2017) and posterior cell proportions.

Bayes factors are printed as BF01 (null versus alternative), as SPSS does, unless BF10 is
requested. Results agree with ``pingouin`` (bayesfactor_ttest / pearson / binom) and the
BayesFactor R package.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import integrate, special, stats

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import fmt, fmt_int, fmt_p, numeric, value_text
from rsstatchem.core.stats.wp4_util import figure, png, sort_key


def _f(x, d: int = 3) -> str:
    return fmt(x, d)


def _bf(x) -> str:
    if x is None or x != x:
        return ""
    if x >= 1e4 or x < 1e-3:
        return f"{x:.3E}"
    return fmt(x, 3)


@dataclass
class BayesOptions:
    analysis: str = "both"         # posterior | bf | both
    level: float = 0.95
    bf10: bool = False             # print BF10 instead of BF01
    plots: bool = False


def _bf_out(bf10: float, o: BayesOptions) -> float:
    return bf10 if o.bf10 else (1 / bf10 if bf10 else float("inf"))


def _bf_note(o: BayesOptions, what: str = "") -> str:
    base = "Bayes factor: " + ("Alternative versus null hypothesis (BF10)." if o.bf10 else
                               "Null versus alternative hypothesis (BF01).")
    return base + (" " + what if what else "")


def _want_post(o):
    return o.analysis in ("posterior", "both")


def _want_bf(o):
    return o.analysis in ("bf", "both")


# =========================================================================== Bayes factors
def jzs_ttest_bf10(t: float, n1: int, n2: int | None = None, r: float = 1.0) -> float:
    """JZS Bayes factor BF10 for a one-sample / paired (n2 None) or two-sample t statistic.

    IBM SPSS Algorithms (BAYES ONESAMPLE eq. 20, BAYES INDEPENDENT eq. 14) integrate against
    the Inverse-Gamma(1/2, 1/2) mixing density, i.e. delta ~ Cauchy(0, 1): the default scale is
    r = 1 (BayesFactor / JASP use r = sqrt(2)/2 = 0.707, available through ``r``)."""
    if n2 is None:
        n, df = n1, n1 - 1
    else:
        n, df = n1 * n2 / (n1 + n2), n1 + n2 - 2

    def fun(g):
        return ((1 + n * g * r ** 2) ** -0.5 *
                (1 + t ** 2 / ((1 + n * g * r ** 2) * df)) ** (-(df + 1) / 2) *
                (2 * math.pi) ** -0.5 * g ** -1.5 * math.exp(-1 / (2 * g)))

    num = integrate.quad(fun, 0, np.inf, limit=200)[0]
    return float(num / (1 + t ** 2 / df) ** (-(df + 1) / 2))


def gonen_bf10(t: float, n1: int, n2: int, mu_delta: float = 0.0, var_delta: float = 1.0
               ) -> float:
    """Gonen et al. (2005): delta ~ N(mu, var) on the standardized effect; exact via the
    noncentral (scaled) t density."""
    nd = n1 * n2 / (n1 + n2)
    df = n1 + n2 - 2
    scale = math.sqrt(1 + nd * var_delta)
    num = stats.nct.pdf(t / scale, df, math.sqrt(nd) * mu_delta / scale) / scale \
        if mu_delta else stats.t.pdf(t / scale, df) / scale
    return float(num / stats.t.pdf(t, df))


def pearson_bf10(r: float, n: int, kappa: float = 1.0) -> float:
    """Jeffreys' exact Bayes factor for rho = 0 (Ly, Verhagen & Wagenmakers, 2016)."""
    k = kappa
    log_hyper = np.log(special.hyp2f1((n - 1) / 2, (n - 1) / 2, (n + 2 / k) / 2, r ** 2))
    return float(np.exp((1 - 2 / k) * np.log(2) + 0.5 * np.log(np.pi) -
                        special.betaln(1 / k, 1 / k) + special.gammaln((n + 2 / k - 1) / 2) -
                        special.gammaln((n + 2 / k) / 2) + log_hyper))


def binom_bf10(k: int, n: int, p0: float = 0.5, a: float = 1.0, b: float = 1.0,
               a0: float | None = None, b0: float | None = None) -> float:
    """Binomial Bayes factor BF10 with Beta(a, b) under H1.

    With ``a0``/``b0`` the null proportion has a Beta(a0, b0) prior and the Bayes factor is the
    IBM SPSS Algorithms (BAYES ONESAMPLE binomial, eq. 1) beta-binomial ratio
    B(a1+y, b1+N-y) B(a0, b0) / (B(a0+y, b0+N-y) B(a1, b1)). Without them H0 is the point
    null p = p0 (the a0, b0 -> infinity limit of IBM's form)."""
    if a0 is not None and b0 is not None:
        return float(np.exp(special.betaln(k + a, n - k + b) + special.betaln(a0, b0)
                            - special.betaln(k + a0, n - k + b0) - special.betaln(a, b)))
    return float(np.exp(special.betaln(k + a, n - k + b) - special.betaln(a, b) -
                        k * np.log(p0) - (n - k) * np.log1p(-p0)))


def wetzels_pearson_bf10(r: float, n: int) -> float:
    """JZS Bayes factor BF10 for rho = 0 of Wetzels & Wagenmakers (2012), the method of IBM
    SPSS Statistics Algorithms, BAYES CORRELATION eq. (10):
    (W/2)^(1/2)/Gamma(1/2) int_0^inf (1+g)^((W-2)/2) [1+(1-r^2)g]^(-(W-1)/2)
    g^(-3/2) exp(-W/(2g)) dg. Not defined for |r| = 1 or W < 2."""
    if n < 2 or not np.isfinite(r) or abs(r) >= 1:
        return float("nan")
    W = float(n)

    def f(u):
        g = math.exp(u)
        return ((W - 2) / 2 * math.log1p(g) - (W - 1) / 2 * math.log1p((1 - r * r) * g)
                - 1.5 * u - W / (2 * g) + u)
    return float(math.exp(0.5 * math.log(W / 2) - special.gammaln(0.5) + _logint(f)))


def hyper_prior_ttest_bf10(t: float, n1: int, n2: int, a: float = -0.75) -> float:
    """Hyper-prior Bayes factor BF10 for the two-sample t test (Wang & Liu, 2016), closed form
    of IBM SPSS Algorithms, BAYES INDEPENDENT eq. (18); IBM's default a = -0.75."""
    nu = n1 + n2 - 2
    return float(math.exp(special.gammaln(nu / 2) + special.gammaln(a + 1.5)
                          - special.gammaln((nu + 1) / 2) - special.gammaln(a + 1)
                          + (nu - 2 * a - 2) / 2 * math.log1p(t * t / nu)))


def poisson_bf10(total: float, n: int, lam0: float, shape: float, rate: float) -> float:
    log_m1 = (shape * np.log(rate) - special.gammaln(shape) + special.gammaln(shape + total) -
              (shape + total) * np.log(rate + n))
    log_m0 = total * np.log(lam0) - n * lam0
    return float(np.exp(log_m1 - log_m0))


def _logint(fun_log, lo=-30.0, hi=30.0) -> float:
    """Integrate exp(fun_log(u)) du over the log scale robustly (shifted by the maximum)."""
    us = np.linspace(lo, hi, 2001)
    vals = np.array([fun_log(u) for u in us])
    m = np.nanmax(vals)
    res = integrate.quad(lambda u: math.exp(fun_log(u) - m), lo, hi, limit=400,
                         points=[us[int(np.nanargmax(vals))]])[0]
    return math.log(res) + m


def regression_bf10(r2: float, n: int, p: int, prior: str = "jzs", g: float | None = None,
                    r: float = 1.0, a: float = 3.0) -> float:
    """Bayes factor of a regression model with p predictors versus the intercept-only model.

    prior: "zellner" (fixed g, default g = n), "jzs" (Zellner-Siow Cauchy, g ~ IG(1/2, r^2 n/2)),
    "hyperg" (Liang et al.'s hyper-g with parameter a)."""
    def log_bf_g(gg):
        return (n - 1 - p) / 2 * math.log1p(gg) - (n - 1) / 2 * math.log1p(gg * (1 - r2))

    if prior == "zellner":
        return math.exp(log_bf_g(float(g if g else n)))
    if prior == "hyperg":
        def f(u):
            gg = math.exp(u)
            return log_bf_g(gg) + math.log((a - 2) / 2) - a / 2 * math.log1p(gg) + u
        return math.exp(_logint(f))
    s = r ** 2 * n / 2

    def f(u):
        gg = math.exp(u)
        # inverse gamma (1/2, s) density on g, times the Jacobian g
        return (log_bf_g(gg) + 0.5 * math.log(s) - special.gammaln(0.5) - 1.5 * u - s / gg + u)
    return math.exp(_logint(f))


def rouder_anova_bf10(y: np.ndarray, groups: np.ndarray, r: float = 0.5) -> float:
    """One-way fixed-effects JZS Bayes factor (Rouder et al., 2012; BayesFactor::anovaBF)."""
    levels = list(pd.unique(groups))
    k = len(levels)
    n = len(y)
    X = np.column_stack([(groups == lv).astype(float) for lv in levels])
    # orthonormal sum-to-zero contrasts
    c = np.eye(k) - 1 / k
    q, _ = np.linalg.qr(c)
    Q = q[:, :k - 1]
    Z = X @ Q
    Zc = Z - Z.mean(axis=0)
    yc = y - y.mean()
    lam, U = np.linalg.eigh(Zc.T @ Zc)
    cvec = U.T @ (Zc.T @ yc)
    syy = float(yc @ yc)

    def f(u):
        g = math.exp(u)
        det = float(np.log1p(g * lam).sum())
        quad = float((cvec ** 2 / (lam + 1 / g)).sum())
        resid = max(syy - quad, 1e-300)
        log_bf = -0.5 * det - (n - 1) / 2 * (math.log(resid) - math.log(syy))
        return log_bf + math.log(r) - 0.5 * math.log(2 * math.pi) - 1.5 * u - r ** 2 / (
            2 * g) + u

    return math.exp(_logint(f))


def gunel_dickey_bf10(table: np.ndarray, sampling: str = "joint", a: float = 1.0) -> float:
    """Bayes factor for association in a two-way table (Gunel & Dickey, 1974; Jamil et al.,
    2017). sampling: "joint" (joint multinomial), "rows" / "cols" (independent multinomial with
    fixed row / column totals)."""
    y = np.asarray(table, float)
    if sampling == "cols":
        y = y.T
        sampling = "rows"
    I, J = y.shape
    A = np.full_like(y, a)

    def lmb(v):
        v = np.asarray(v, float).ravel()
        return float(special.gammaln(v).sum() - special.gammaln(v.sum()))

    if sampling == "joint":
        xr = A.sum(axis=1) - J + 1
        xc = A.sum(axis=0) - I + 1
        log_bf = (lmb(y + A) - lmb(A) - (lmb(y.sum(axis=1) + xr) - lmb(xr) +
                                          lmb(y.sum(axis=0) + xc) - lmb(xc)))
        return math.exp(log_bf)
    xc = A.sum(axis=0) - I + 1
    log_bf = sum(lmb(y[i] + A[i]) - lmb(A[i]) for i in range(I)) - (
        lmb(y.sum(axis=0) + xc) - lmb(xc))
    return math.exp(log_bf)


# =========================================================================== posteriors
@dataclass
class Posterior:
    mode: float
    mean: float
    var: float
    lo: float
    hi: float
    dist: object = None    # frozen scipy distribution, when available

    def row(self, d: int = 3) -> list[str]:
        return [_f(self.mode, d), _f(self.mean, d), _f(self.var, d), _f(self.lo, d),
                _f(self.hi, d)]


def _from_dist(dist, mode, level) -> Posterior:
    lo, hi = dist.ppf([(1 - level) / 2, (1 + level) / 2])
    return Posterior(float(mode), float(dist.mean()), float(dist.var()), float(lo), float(hi),
                     dist)


def normal_mean_posterior(x: np.ndarray, level=0.95, prior: str = "diffuse",
                          known_var: float | None = None, mu0: float = 0.0,
                          kappa0: float = 1.0, a0: float = 1.0, b0: float = 1.0) -> Posterior:
    """Posterior of a normal mean.

    prior "diffuse": Jeffreys on (mu, sigma^2); "conjugate": mu | s2 ~ N(mu0, s2/kappa0),
    s2 ~ IG(a0, b0). With ``known_var``, mu ~ N(mu0, known_var/kappa0) ("conjugate") or flat."""
    n = len(x)
    xbar = float(np.mean(x))
    ss = float(((x - xbar) ** 2).sum())
    if known_var is not None:
        if prior == "conjugate":
            prec = kappa0 / known_var + n / known_var
            m = (kappa0 * mu0 + n * xbar) / (kappa0 + n)
        else:
            prec, m = n / known_var, xbar
        dist = stats.norm(m, math.sqrt(1 / prec))
        return _from_dist(dist, m, level)
    if prior == "conjugate":
        kn = kappa0 + n
        mn = (kappa0 * mu0 + n * xbar) / kn
        an = a0 + n / 2
        bn = b0 + ss / 2 + kappa0 * n * (xbar - mu0) ** 2 / (2 * kn)
    else:
        kn, mn, an, bn = n, xbar, (n - 1) / 2, ss / 2
    dist = stats.t(2 * an, loc=mn, scale=math.sqrt(bn / (an * kn)))
    return _from_dist(dist, mn, level)


def variance_posterior(shape: float, scale: float, level=0.95) -> Posterior:
    dist = stats.invgamma(shape, scale=scale)
    mode = scale / (shape + 1)
    return _from_dist(dist, mode, level)


def behrens_fisher_posterior(a: np.ndarray, b: np.ndarray, level=0.95) -> Posterior:
    """Posterior of mean(a) - mean(b) under diffuse priors with unequal variances, as in IBM
    SPSS Statistics Algorithms (BAYES INDEPENDENT eqs. 47-58): Behrens-Fisher distribution
    approximated by Patil's scaled t with nu = eta3 degrees of freedom."""
    n1, n2 = len(a), len(b)
    v1, v2 = a.var(ddof=1) / n1, b.var(ddof=1) / n2
    S = v1 + v2
    s1, c1 = v1 / S, v2 / S                       # sin^2 psi (group a), cos^2 psi (group b)
    eta1 = (n1 - 1) / (n1 - 3) * s1 + (n2 - 1) / (n2 - 3) * c1
    eta2 = ((n1 - 1) ** 2 / ((n1 - 3) ** 2 * (n1 - 5)) * s1 ** 2
            + (n2 - 1) ** 2 / ((n2 - 3) ** 2 * (n2 - 5)) * c1 ** 2)
    nu = 4 + eta1 ** 2 / eta2
    sigma2 = eta1 * (nu - 2) / nu * S
    mu = float(a.mean() - b.mean())
    dist = stats.t(nu, loc=mu, scale=math.sqrt(sigma2))
    post = _from_dist(dist, mu, level)
    post.var = float(nu / (nu - 2) * sigma2)
    return post


def correlation_posterior_fisher(r: float, n: int, level=0.95) -> Posterior:
    """Posterior of rho as in IBM SPSS Algorithms, BAYES CORRELATION eqs. (18)-(21): with a
    uniform prior xi = atanh(rho) ~ Normal(z - 5r/(2W), 1/(W - 3/2 + 5/2 (1 - r^2))), z =
    atanh(r). IBM summarises 10^4 simulated draws of tanh(xi); here the same distribution is
    summarised deterministically (exact quantiles, numerical mode / mean / variance)."""
    W = float(n)
    z = math.atanh(max(min(r, 0.999999999), -0.999999999))
    mu = z - 5 * r / (2 * W)
    sd = 1 / math.sqrt(W - 1.5 + 2.5 * (1 - r * r))
    xi = np.linspace(mu - 9 * sd, mu + 9 * sd, 20001)
    w = stats.norm.pdf(xi, mu, sd)
    rho = np.tanh(xi)
    mean = float(np.trapezoid(rho * w, xi))
    var = float(np.trapezoid((rho - mean) ** 2 * w, xi))
    dens_rho = w / (1 - rho ** 2)                     # density of rho = tanh(xi)
    mode = float(rho[np.argmax(dens_rho)])
    lo = float(np.tanh(mu + sd * stats.norm.ppf((1 - level) / 2)))
    hi = float(np.tanh(mu + sd * stats.norm.ppf((1 + level) / 2)))
    post = Posterior(mode, mean, var, lo, hi)
    post.dist = (rho, dens_rho)
    return post


def correlation_posterior(r: float, n: int, kappa: float = 1.0, level=0.95) -> Posterior:
    rho = np.linspace(-0.9999, 0.9999, 20001)
    loglik = (n - 1) / 2 * np.log1p(-rho ** 2) - (n - 1.5) * np.log1p(-rho * r)
    prior = (1 / kappa - 1) * np.log1p(-rho ** 2)
    lp = loglik + prior
    dens = np.exp(lp - lp.max())
    dens /= np.trapezoid(dens, rho)
    cdf = integrate.cumulative_trapezoid(dens, rho, initial=0)
    mean = float(np.trapezoid(rho * dens, rho))
    var = float(np.trapezoid((rho - mean) ** 2 * dens, rho))
    lo = float(np.interp((1 - level) / 2, cdf, rho))
    hi = float(np.interp((1 + level) / 2, cdf, rho))
    post = Posterior(float(rho[np.argmax(dens)]), mean, var, lo, hi)
    post.dist = (rho, dens)
    return post


# =========================================================================== plots
def density_png(title: str, posterior: Posterior, prior=None, xlabel: str = "") -> bytes:
    fig = figure(5.6, 3.8)
    ax = fig.add_subplot(111)
    if isinstance(posterior.dist, tuple):
        xs, ys = posterior.dist
        ax.plot(xs, ys, color="#1f4e9c", lw=1.6, label="Posterior")
    else:
        sd = math.sqrt(posterior.var) if posterior.var == posterior.var and \
            posterior.var > 0 else 1.0
        lo = posterior.dist.ppf(0.0005) if posterior.dist is not None else posterior.mean - 4 * sd
        hi = posterior.dist.ppf(0.9995) if posterior.dist is not None else posterior.mean + 4 * sd
        xs = np.linspace(lo, hi, 400)
        ax.plot(xs, posterior.dist.pdf(xs), color="#1f4e9c", lw=1.6, label="Posterior")
        if prior is not None:
            ax.plot(xs, prior.pdf(xs), color="#c8102e", lw=1.0, ls="--", label="Prior")
    ax.axvspan(posterior.lo, posterior.hi, color="#1f4e9c", alpha=0.08)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7)
    return png(fig)


# =========================================================================== procedures
def _post_cols(level):
    ci = f"{level * 100:g}% Credible Interval"
    return ["Posterior Mode", "Posterior Mean", "Posterior Variance", (ci, "Lower Bound"),
            (ci, "Upper Bound")]


def one_sample_normal(ds: Dataset, variables: list[str], null: float = 0.0,
                      o: BayesOptions | None = None, r: float = 1.0,
                      prior: str = "diffuse", known_var: float | None = None, mu0=0.0,
                      kappa0=1.0, a0=1.0, b0=1.0, paired: list | None = None) -> Section:
    """One Sample Normal (``variables``) or Related Samples Normal (``paired`` pairs)."""
    o = o or BayesOptions()
    related = paired is not None
    sec = Section("Bayesian Related Samples Inference" if related else
                  "Bayesian One Sample Inference")
    items = [(f"{ds.var(a).title} - {ds.var(b).title}",
              (numeric(ds, a) - numeric(ds, b)).dropna().to_numpy())
             for a, b in paired] if related else \
        [(ds.var(v).title, numeric(ds, v).dropna().to_numpy()) for v in variables]
    sec.log(("BAYES RELATED" if related else "BAYES ONESAMPLE") +
            f" /ANALYSIS={o.analysis.upper()} /CRITERIA CILEVEL={o.level * 100:g}" +
            (f" /INFERENCE TESTVAL={null:g}" if not related else "") +
            f" /ESTBF RSCALE={r:g}")
    if _want_bf(o):
        rows = []
        for name, x in items:
            n = len(x)
            m, s = float(np.mean(x)), float(np.std(x, ddof=1))
            se = s / math.sqrt(n)
            t = (m - null) / se
            bf10 = jzs_ttest_bf10(t, n, None, r)
            rows.append([name, fmt_int(n), _f(m), _f(s), _f(se), _bf(_bf_out(bf10, o)),
                         _f(t), fmt_int(n - 1), fmt_p(2 * stats.t.sf(abs(t), n - 1))])
        cols = ["", "N", "Mean", "Std. Deviation", "Std. Error Mean", "Bayes Factorᵃ", "t",
                "df", "Sig. (2-tailed)"]
        title = "Bayes Factor for Related Sample T-Test" if related else \
            "Bayes Factor for One-Sample T-Test"
        sec.table(Table(title, cols, rows,
                        footnotes=["a. " + _bf_note(o, f"JZS prior (Rouder et al., 2009), "
                                                       f"Cauchy scale r = {r:g}" +
                                   (f", test value = {null:g}." if not related else "."))]))
    if _want_post(o):
        rows = []
        posts = []
        for name, x in items:
            p = normal_mean_posterior(x, o.level, prior, known_var, mu0, kappa0, a0, b0)
            posts.append((name, p))
            rows.append([name, fmt_int(len(x))] + p.row())
        what = "Mean Difference" if related else "Mean"
        notes = ["Prior on mean: " + ("diffuse (Jeffreys)." if prior == "diffuse" else
                                      f"normal (location {mu0:g}, kappa {kappa0:g})."),
                 "Variance " + (f"known = {known_var:g}." if known_var is not None else
                                ("diffuse prior." if prior == "diffuse" else
                                 f"prior inverse gamma (shape {a0:g}, scale {b0:g})."))]
        kind = "Related-Sample" if related else "One-Sample"
        sec.table(Table(f"Posterior Distribution Characterization for {kind} {what}",
                        ["", "N"] + _post_cols(o.level), rows, footnotes=notes))
        if o.plots:
            for name, p in posts:
                sec.image(f"Posterior Distribution: {name}", density_png(name, p,
                                                                          xlabel=what))
    return sec


def independent_normal(ds: Dataset, variables: list[str], group: str, g1, g2,
                       o: BayesOptions | None = None, method: str = "rouder", r: float = 1.0,
                       mu_delta: float = 0.0, var_delta: float = 1.0, equal_var: bool = True,
                       seed: int = 2000000, hyper_a: float = -0.75) -> Section:
    """``method``: "rouder" (JZS, IBM default scale r = 1), "gonen" or "hyper" (Wang & Liu
    hyper-prior, IBM default a = -0.75)."""
    o = o or BayesOptions()
    sec = Section("Bayesian Independent Sample Inference")
    sec.log(f"BAYES INDEPENDENT {' '.join(variables)} BY {group}({g1} {g2})"
            f" /ANALYSIS={o.analysis.upper()} /ESTBF " +
            ("ROUDER" if method == "rouder" else f"HYPER({hyper_a:g})" if method == "hyper"
             else f"GONEN({mu_delta:g} {var_delta:g})"))
    gser = ds.analysis_series(group)
    gv = ds.var(group)
    stat_rows, bf_rows, post_rows, plots = [], [], [], []
    for v in variables:
        x = numeric(ds, v)
        a = x[gser == g1].dropna().to_numpy()
        b = x[gser == g2].dropna().to_numpy()
        name = ds.var(v).title
        for lv, arr in ((g1, a), (g2, b)):
            stat_rows.append([name, str(value_text(gv, lv)), fmt_int(len(arr)),
                              _f(arr.mean()) if len(arr) else "",
                              _f(arr.std(ddof=1)) if len(arr) > 1 else "",
                              _f(arr.std(ddof=1) / math.sqrt(len(arr))) if len(arr) > 1
                              else ""])
        n1, n2 = len(a), len(b)
        if n1 < 2 or n2 < 2:
            continue
        # IBM BAYES INDEPENDENT eq. (2): d = mean(group 2) - mean(group 1)
        diff = b.mean() - a.mean()
        sp2 = ((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2)
        se = math.sqrt(sp2 * (1 / n1 + 1 / n2))
        t = diff / se
        df = n1 + n2 - 2
        if method == "gonen":
            bf10 = gonen_bf10(t, n1, n2, mu_delta, var_delta)
        elif method == "hyper":
            bf10 = hyper_prior_ttest_bf10(t, n1, n2, hyper_a)
        else:
            bf10 = jzs_ttest_bf10(t, n1, n2, r)
        bf_rows.append([name, _f(diff), _f(se), _bf(_bf_out(bf10, o)), _f(t), fmt_int(df),
                        fmt_p(2 * stats.t.sf(abs(t), df))])
        if equal_var:
            dist = stats.t(df, loc=diff, scale=se)
            p = _from_dist(dist, diff, o.level)
        else:
            # IBM BAYES INDEPENDENT, diffuse priors with unequal variances: Behrens-Fisher
            # posterior with Patil's (1965) scaled-t approximation, eqs. (47)-(58)
            p = behrens_fisher_posterior(b, a, o.level)
        post_rows.append([name, fmt_int(n1 + n2)] + p.row())
        plots.append((name, p))
    sec.table(Table("Group Statistics", ["", gv.title, "N", "Mean", "Std. Deviation",
                                         "Std. Error Mean"], stat_rows, row_header_cols=2))
    if _want_bf(o):
        how = (f"Rouder's method (JZS), Cauchy scale r = {r:g}." if method == "rouder" else
               f"Hyper-prior method (Wang & Liu, 2016), a = {hyper_a:g}." if method == "hyper"
               else f"Gönen's method, effect size prior N({mu_delta:g}, {var_delta:g}).")
        sec.table(Table("Bayes Factor Independent Sample Test",
                        ["", "Mean Difference", "Pooled Std. Error Difference", "Bayes Factorᵃ",
                         "t", "df", "Sig. (2-tailed)"], bf_rows,
                        footnotes=["a. " + _bf_note(o, how)]))
    if _want_post(o):
        sec.table(Table("Posterior Distribution Characterization for Independent Sample Mean",
                        ["", "N"] + _post_cols(o.level), post_rows,
                        footnotes=["Diffuse (reference) priors. Variances " +
                                   ("assumed equal." if equal_var else
                                    "not assumed equal (Behrens-Fisher posterior, Patil's "
                                    "approximation)."),
                                   "Difference = group 2 minus group 1 (as in SPSS)."]))
        if o.plots:
            for name, p in plots:
                sec.image(f"Posterior Distribution: {name}",
                          density_png(name, p, xlabel="Mean Difference"))
    return sec


def pearson_correlation(ds: Dataset, variables: list[str], o: BayesOptions | None = None,
                        kappa: float = 1.0, method: str = "jzs") -> Section:
    """``method`` "jzs": IBM's Wetzels-Wagenmakers JZS Bayes factor and Fisher-z posterior
    (uniform prior); "stretched": Ly et al. (2016) stretched-beta prior with ``kappa`` (as in
    JASP; not an IBM SPSS method)."""
    o = o or BayesOptions()
    ibm = method != "stretched"
    sec = Section("Bayesian Pearson Correlation")
    sec.log(f"BAYES CORRELATION {' '.join(variables)} /ANALYSIS={o.analysis.upper()}"
            + (" /ESTBF COMPUTATION=JZS" if method != "stretched" else
               f" /ESTBF KAPPA={kappa:g}"))
    k = len(variables)
    titles = [ds.var(v).title for v in variables]
    rows, post_rows, plots = [], [], []
    for i in range(k):
        stat_lines = {"Pearson Correlation": [], "Bayes Factor": [], "N": []}
        for j in range(k):
            pair = pd.concat([numeric(ds, variables[i]), numeric(ds, variables[j])],
                             axis=1).dropna()
            n = len(pair)
            if i == j:
                stat_lines["Pearson Correlation"].append("1")
                stat_lines["Bayes Factor"].append("")
                stat_lines["N"].append(fmt_int(n))
                continue
            r = float(np.corrcoef(pair.iloc[:, 0], pair.iloc[:, 1])[0, 1]) if n > 2 else \
                float("nan")
            stat_lines["Pearson Correlation"].append(_f(r))
            bf = wetzels_pearson_bf10(r, n) if ibm else pearson_bf10(r, n, kappa)
            stat_lines["Bayes Factor"].append(_bf(_bf_out(bf, o))
                                              if n > 2 and np.isfinite(bf) else "")
            stat_lines["N"].append(fmt_int(n))
            if j > i and n > 3:
                p = (correlation_posterior_fisher(r, n, o.level) if ibm else
                     correlation_posterior(r, n, kappa, o.level))
                post_rows.append([f"{titles[i]} - {titles[j]}", fmt_int(n)] + p.row())
                plots.append((f"{titles[i]} - {titles[j]}", p))
        for name, vals in stat_lines.items():
            if name == "Bayes Factor" and not _want_bf(o):
                continue
            rows.append([titles[i], name + ("ᵃ" if name == "Bayes Factor" else "")] + vals)
    sec.table(Table("Bayes Factor Inference on Pairwise Correlations", ["", ""] + titles, rows,
                    row_header_cols=2,
                    footnotes=["a. " + _bf_note(o, "JZS prior (Wetzels & Wagenmakers, 2012)."
                                                if ibm else
                                                f"Stretched beta prior, kappa = {kappa:g} "
                                                "(Ly et al., 2016).")]))
    if _want_post(o) and post_rows:
        sec.table(Table("Posterior Distribution Characterization for Pairwise Correlations",
                        ["", "N"] + _post_cols(o.level), post_rows,
                        footnotes=["Uniform prior on rho; Fisher-z normal approximation "
                                   "(IBM BAYES CORRELATION eq. 20)." if ibm else
                                   f"Posterior under the stretched beta prior (kappa = "
                                   f"{kappa:g}), computed numerically."]))
        if o.plots:
            for name, p in plots:
                sec.image(f"Posterior Distribution: {name}", density_png(name, p,
                                                                          xlabel="rho"))
    return sec


def _design(ds: Dataset, factors: list[str], covariates: list[str], extra=()):
    cols = {c: numeric(ds, c) for c in covariates}
    for f in factors:
        cols[f] = ds.analysis_series(f)
    for e in extra:
        cols[e] = numeric(ds, e)
    frame = pd.DataFrame(cols).dropna()
    names, X = [], []
    for f in factors:
        levels = sorted(pd.unique(frame[f]), key=sort_key)
        for lv in levels[:-1]:       # last category is the reference, as SPSS GLM prints
            names.append(f"{ds.var(f).title} = {value_text(ds.var(f), lv)}")
            X.append((frame[f] == lv).to_numpy(float))
    for c in covariates:
        names.append(ds.var(c).title)
        X.append(frame[c].to_numpy(float))
    return frame, names, (np.column_stack(X) if X else np.zeros((len(frame), 0)))


def linear_regression(ds: Dataset, dependent: str, factors: list[str], covariates: list[str],
                      o: BayesOptions | None = None, prior: str = "jzs", r: float = 1.0,
                      g: float | None = None, a: float = 3.0) -> Section:
    o = o or BayesOptions()
    sec = Section("Bayesian Linear Regression")
    sec.log(f"BAYES REGRESSION {dependent} " + (f"BY {' '.join(factors)} " if factors else "")
            + (f"WITH {' '.join(covariates)}" if covariates else "") +
            f" /ANALYSIS={o.analysis.upper()} /ESTBF COMPUTATION={prior.upper()}")
    frame, names, X = _design(ds, factors, covariates, [dependent])
    y = frame[dependent].to_numpy(float)
    n, p = X.shape
    Xf = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(Xf, y, rcond=None)
    fitted = Xf @ beta
    sse = float(((y - fitted) ** 2).sum())
    sst = float(((y - y.mean()) ** 2).sum())
    ssr = sst - sse
    df_e = n - p - 1
    mse = sse / df_e
    r2 = ssr / sst if sst > 0 else 0.0
    F = (ssr / p) / mse if p else float("nan")
    bf10 = regression_bf10(r2, n, p, prior, g, r, a) if p else 1.0
    prior_text = {"jzs": f"JZS (Zellner-Siow) prior, scale r = {r:g}",
                  "zellner": f"Zellner's g-prior, g = {g if g else n:g}",
                  "hyperg": f"hyper-g prior, a = {a:g}"}[prior]
    sec.table(Table("ANOVAᵃ", ["Model", "Sum of Squares", "df", "Mean Square", "F", "Sig."],
                    [["Regression", _f(ssr), fmt_int(p), _f(ssr / p) if p else "", _f(F),
                      fmt_p(stats.f.sf(F, p, df_e)) if p else ""],
                     ["Residual", _f(sse), fmt_int(df_e), _f(mse), "", ""],
                     ["Total", _f(sst), fmt_int(n - 1), "", "", ""]],
                    footnotes=[f"a. Dependent Variable: {ds.var(dependent).title}",
                               "Model: (Intercept), " + ", ".join(names)]))
    if _want_bf(o):
        adj = 1 - (1 - r2) * (n - 1) / df_e
        sec.table(Table("Model Summary", ["Bayes Factorᵃ", "R", "R Square",
                                          "Adjusted R Square", "Std. Error of the Estimate"],
                        [[_bf(_bf_out(bf10, o)), _f(math.sqrt(max(r2, 0))), _f(r2), _f(adj),
                          _f(math.sqrt(mse))]], row_header_cols=0,
                        footnotes=["a. " + _bf_note(o, f"Model versus null model (Intercept "
                                                       f"only); {prior_text}.")]))
    if _want_post(o):
        cov = np.linalg.inv(Xf.T @ Xf) * mse
        rows = []
        for name, b, v in zip(["(Intercept)"] + names, beta, np.diag(cov)):
            dist = stats.t(df_e, loc=b, scale=math.sqrt(v))
            rows.append([name] + _from_dist(dist, b, o.level).row())
        sec.table(Table("Bayesian Estimates of Coefficientsᵃ ᵇ ᶜ", ["Parameter"] +
                        _post_cols(o.level), rows,
                        footnotes=[f"a. Dependent Variable: {ds.var(dependent).title}",
                                   "b. Model: (Intercept), " + ", ".join(names),
                                   "c. Assume standard reference priors."]))
        vp = variance_posterior(df_e / 2, sse / 2, o.level)
        sec.table(Table("Bayesian Estimates of Error Varianceᵃ", ["Parameter"] +
                        _post_cols(o.level), [["Error variance"] + vp.row()],
                        footnotes=["a. Assume standard reference priors."]))
    return sec


def oneway_anova(ds: Dataset, dependents: list[str], factor: str,
                 o: BayesOptions | None = None, method: str = "rouder", r: float = 1.0,
                 g: float | None = None, a: float = 3.0) -> Section:
    """``method``: "rouder" (IBM: g ~ Inverse-Gamma(1/2, N s^2/2) on the R^2 form, scale
    s = ``r``), "jzs" (Zellner-Siow), "zellner", "hyperg", or "rouder2012" (BayesFactor's
    anovaBF multivariate-g prior on the effects, not an IBM SPSS method; use r = 0.5)."""
    o = o or BayesOptions()
    sec = Section("Bayesian One-way ANOVA")
    sec.log(f"BAYES ANOVA {' '.join(dependents)} BY {factor} /ANALYSIS={o.analysis.upper()}"
            f" /ESTBF COMPUTATION={method.upper()}")
    fv = ds.var(factor)
    for dep in dependents:
        frame = pd.DataFrame({"y": numeric(ds, dep), "f": ds.analysis_series(factor)}).dropna()
        y = frame["y"].to_numpy(float)
        gvals = frame["f"].to_numpy(dtype=object)
        levels = sorted(pd.unique(gvals), key=sort_key)
        k, n = len(levels), len(y)
        if k < 2 or n <= k:
            sec.note("Warning", f"{ds.var(dep).title}: not enough groups or cases.", "warning")
            continue
        means = np.array([y[gvals == lv].mean() for lv in levels])
        ns = np.array([(gvals == lv).sum() for lv in levels])
        ssb = float((ns * (means - y.mean()) ** 2).sum())
        sst = float(((y - y.mean()) ** 2).sum())
        ssw = sst - ssb
        dfb, dfw = k - 1, n - k
        F = (ssb / dfb) / (ssw / dfw)
        if method == "rouder2012":
            bf10 = rouder_anova_bf10(y, gvals, r)
            how = (f"Rouder et al. (2012) multivariate JZS prior (BayesFactor anovaBF), "
                   f"scale r = {r:g}")
        elif method == "rouder":
            bf10 = regression_bf10(ssb / sst, n, k - 1, "jzs", None, r)
            how = f"Rouder's method, scale s = {r:g}"
        else:
            bf10 = regression_bf10(ssb / sst, n, k - 1, method, g, r if method == "jzs" else 1,
                                   a)
            how = {"jzs": f"Zellner-Siow (JZS g-prior), scale r = {r:g}",
                   "zellner": f"Zellner's g-prior, g = {g if g else n:g}",
                   "hyperg": f"hyper-g prior, a = {a:g}"}[method]
        cols = ["", "Sum of Squares", "df", "Mean Square", "F", "Sig."]
        bf_cell = []
        if _want_bf(o):
            cols.append("Bayes Factorᵃ")
            bf_cell = [_bf(_bf_out(bf10, o))]
        sec.table(Table(f"ANOVA: {ds.var(dep).title}", cols,
                        [["Between Groups", _f(ssb), fmt_int(dfb), _f(ssb / dfb), _f(F),
                          fmt_p(stats.f.sf(F, dfb, dfw))] + bf_cell,
                         ["Within Groups", _f(ssw), fmt_int(dfw), _f(ssw / dfw), "", ""] +
                         ([""] if bf_cell else []),
                         ["Total", _f(sst), fmt_int(n - 1), "", "", ""] + ([""] if bf_cell
                                                                          else [])],
                        footnotes=["a. " + _bf_note(o, how + ".")] if bf_cell else []))
        if _want_post(o):
            mse = ssw / dfw
            rows = []
            for lv, m, nn in zip(levels, means, ns):
                dist = stats.t(dfw, loc=m, scale=math.sqrt(mse / nn))
                rows.append([f"{fv.title} = {value_text(fv, lv)}"] +
                            _from_dist(dist, m, o.level).row())
            sec.table(Table("Bayesian Estimates of Coefficientsᵃ ᵇ", ["Parameter"] +
                            _post_cols(o.level), rows,
                            footnotes=[f"a. Dependent Variable: {ds.var(dep).title}",
                                       "b. Assume standard reference priors."]))
            vp = variance_posterior(dfw / 2, ssw / 2, o.level)
            sec.table(Table("Bayesian Estimates of Error Varianceᵃ", ["Parameter"] +
                            _post_cols(o.level), [["Error variance"] + vp.row()],
                            footnotes=["a. Assume standard reference priors."]))
    return sec


def repeated_measures_anova(ds: Dataset, variables: list[str], o: BayesOptions | None = None,
                            r: float = 1.0, pairwise: bool = True) -> Section:
    o = o or BayesOptions()
    sec = Section("Bayesian One-way Repeated Measures ANOVA")
    sec.log(f"BAYES REPANOVA {' '.join(variables)} /ANALYSIS={o.analysis.upper()}")
    data = pd.DataFrame({v: numeric(ds, v) for v in variables}).dropna().to_numpy(float)
    n, k = data.shape
    grand = data.mean()
    ss_sub = k * ((data.mean(axis=1) - grand) ** 2).sum()
    ss_cond = n * ((data.mean(axis=0) - grand) ** 2).sum()
    ss_tot = ((data - grand) ** 2).sum()
    ss_err = ss_tot - ss_sub - ss_cond
    df_c, df_e = k - 1, (k - 1) * (n - 1)
    F = (ss_cond / df_c) / (ss_err / df_e)
    # BIC approximation (Masson, 2011): Delta BIC = N ln(1 - eta_p^2) + df ln(N), N = n * k
    eta = ss_cond / (ss_cond + ss_err)
    nobs = n * (k - 1)
    dbic = nobs * math.log(1 - eta) + df_c * math.log(nobs)
    bf01 = math.exp(dbic / 2)
    bf10 = 1 / bf01
    titles = [ds.var(v).title for v in variables]
    sec.table(Table("Descriptive Statistics", ["", "N", "Mean", "Std. Deviation"],
                    [[t, fmt_int(n), _f(data[:, j].mean()), _f(data[:, j].std(ddof=1))]
                     for j, t in enumerate(titles)]))
    cols = ["Source", "Sum of Squares", "df", "Mean Square", "F", "Sig."]
    rows = [["Conditions", _f(ss_cond), fmt_int(df_c), _f(ss_cond / df_c), _f(F),
             fmt_p(stats.f.sf(F, df_c, df_e))],
            ["Subjects", _f(ss_sub), fmt_int(n - 1), _f(ss_sub / (n - 1)), "", ""],
            ["Error", _f(ss_err), fmt_int(df_e), _f(ss_err / df_e), "", ""]]
    if _want_bf(o):
        cols.append("Bayes Factorᵃ")
        rows[0].append(_bf(_bf_out(bf10, o)))
        rows[1].append("")
        rows[2].append("")
    sec.table(Table("Bayes Factor for Repeated Measures ANOVA", cols, rows,
                    footnotes=["a. " + _bf_note(o, "BIC approximation (Wagenmakers, 2007; "
                                                   "Masson, 2011).")] if _want_bf(o) else []))
    if pairwise and _want_bf(o) and k > 2:
        prow = []
        for i in range(k):
            for j in range(i + 1, k):
                d = data[:, i] - data[:, j]
                t = d.mean() / (d.std(ddof=1) / math.sqrt(n))
                prow.append([f"{titles[i]} - {titles[j]}", _f(d.mean()),
                             _bf(_bf_out(jzs_ttest_bf10(t, n, None, r), o)), _f(t),
                             fmt_int(n - 1), fmt_p(2 * stats.t.sf(abs(t), n - 1))])
        sec.table(Table("Pairwise Comparisons", ["", "Mean Difference", "Bayes Factorᵃ", "t",
                                                 "df", "Sig. (2-tailed)"], prow,
                        footnotes=["a. " + _bf_note(o, f"JZS prior, scale r = {r:g}.")]))
    if _want_post(o):
        mse = ss_err / df_e
        rows = []
        for j, t in enumerate(titles):
            m = data[:, j].mean()
            dist = stats.t(df_e, loc=m, scale=math.sqrt(mse / n))
            rows.append([t] + _from_dist(dist, m, o.level).row())
        sec.table(Table("Posterior Distribution Characterization for Condition Means",
                        ["Condition"] + _post_cols(o.level), rows,
                        footnotes=["Reference priors; error variance pooled from the "
                                   "condition-by-subject residual."]))
    return sec


def one_sample_binomial(ds: Dataset, variables: list[str], success, null: float = 0.5,
                        o: BayesOptions | None = None, a: float = 1.0, b: float = 1.0,
                        a_post: float | None = None, b_post: float | None = None,
                        null_a: float | None = None, null_b: float | None = None) -> Section:
    """``success``: a value (string or number) or ("le", cut) for "value <= cut" as SPSS's
    success categories (default: last category). ``null_a``/``null_b``: Beta(a0, b0) prior on
    the proportion under H0 (IBM BAYES ONESAMPLE binomial Bayes factor); when omitted H0 is
    the point null p = ``null``."""
    o = o or BayesOptions()
    sec = Section("Bayesian One Sample Binomial Inference")
    sec.log(f"BAYES BINOMIAL {' '.join(variables)} /SUCCESS={success} /TESTVAL={null:g}"
            f" /PRIOR BETA({a:g},{b:g})")
    ap = a if a_post is None else a_post
    bp = b if b_post is None else b_post
    bf_rows, post_rows, plots = [], [], []
    for v in variables:
        s = ds.analysis_series(v).dropna()
        if success is None:
            vals = sorted(pd.unique(s), key=sort_key)
            hit = s == vals[-1]
            stext = str(value_text(ds.var(v), vals[-1]))
        elif isinstance(success, tuple):
            hit = pd.to_numeric(s, errors="coerce") <= success[1]
            stext = f"<= {success[1]:g}"
        else:
            hit = s.astype(str) == str(success) if isinstance(success, str) else s == success
            stext = str(value_text(ds.var(v), success))
        n, k = len(s), int(hit.sum())
        name = ds.var(v).title
        bf10 = binom_bf10(k, n, null, a, b, null_a, null_b)
        bf_rows.append([name, stext, fmt_int(n), fmt_int(k), _f(k / n if n else float("nan")),
                        _bf(_bf_out(bf10, o))])
        dist = stats.beta(k + ap, n - k + bp)
        alpha_, beta_ = k + ap, n - k + bp
        mode = (alpha_ - 1) / (alpha_ + beta_ - 2) if alpha_ > 1 and beta_ > 1 else \
            float(alpha_ >= beta_)
        p = _from_dist(dist, mode, o.level)
        post_rows.append([name, fmt_int(n)] + p.row())
        plots.append((name, p, stats.beta(ap, bp)))
    if _want_bf(o):
        sec.table(Table("Binomial Test", ["", "Success Category", "N", "Successes",
                                          "Observed Success Rate", "Bayes Factorᵃ"], bf_rows,
                        footnotes=["a. " + _bf_note(o, f"Beta({a:g}, {b:g}) prior under the "
                                                       "alternative; " +
                                                       (f"Beta({null_a:g}, {null_b:g}) prior "
                                                        "under the null (IBM)."
                                                        if null_a is not None and
                                                        null_b is not None else
                                                        f"point null proportion {null:g}."))]))
    if _want_post(o):
        sec.table(Table("Posterior Distribution Characterization for Binomial Inference",
                        ["", "N"] + _post_cols(o.level), post_rows,
                        footnotes=[f"Prior: Beta({ap:g}, {bp:g})."]))
        if o.plots:
            for name, p, pr in plots:
                sec.image(f"Prior and Posterior: {name}", density_png(name, p, pr,
                                                                       "Proportion"))
    return sec


def one_sample_poisson(ds: Dataset, variables: list[str], null: float = 1.0,
                       o: BayesOptions | None = None, shape: float = 2.0,
                       rate: float = 1.0) -> Section:
    o = o or BayesOptions()
    sec = Section("Bayesian One Sample Poisson Inference")
    sec.log(f"BAYES POISSON {' '.join(variables)} /TESTVAL={null:g} /PRIOR GAMMA({shape:g},"
            f"{rate:g})")
    bf_rows, post_rows, plots = [], [], []
    for v in variables:
        x = numeric(ds, v).dropna().to_numpy()
        n, total = len(x), float(x.sum())
        name = ds.var(v).title
        bf10 = poisson_bf10(total, n, null, shape, rate)
        bf_rows.append([name, fmt_int(n), _f(total / n if n else float("nan")),
                        _bf(_bf_out(bf10, o))])
        a, b = shape + total, rate + n
        dist = stats.gamma(a, scale=1 / b)
        p = _from_dist(dist, (a - 1) / b if a >= 1 else 0.0, o.level)
        post_rows.append([name, fmt_int(n)] + p.row())
        plots.append((name, p, stats.gamma(shape, scale=1 / rate)))
    if _want_bf(o):
        sec.table(Table("Poisson Test", ["", "N", "Mean (Rate)", "Bayes Factorᵃ"], bf_rows,
                        footnotes=["a. " + _bf_note(o, f"Gamma(shape {shape:g}, rate "
                                                       f"{rate:g}) prior under the "
                                                       f"alternative; null rate {null:g}.")]))
    if _want_post(o):
        sec.table(Table("Posterior Distribution Characterization for Poisson Inference",
                        ["", "N"] + _post_cols(o.level), post_rows,
                        footnotes=[f"Prior: Gamma(shape {shape:g}, rate {rate:g})."]))
        if o.plots:
            for name, p, pr in plots:
                sec.image(f"Prior and Posterior: {name}", density_png(name, p, pr, "Rate"))
    return sec


def loglinear(ds: Dataset, row: str, col: str, o: BayesOptions | None = None,
              sampling: str = "joint", a: float = 1.0) -> Section:
    o = o or BayesOptions()
    sec = Section("Bayesian Log-Linear Models")
    sec.log(f"BAYES LOGLINEAR {row} BY {col} /MODEL={sampling.upper()} /PRIOR A={a:g}")
    frame = pd.DataFrame({"r": ds.analysis_series(row), "c": ds.analysis_series(col)}).dropna()
    rl = sorted(pd.unique(frame["r"]), key=sort_key)
    cl = sorted(pd.unique(frame["c"]), key=sort_key)
    tab = np.array([[((frame["r"] == i) & (frame["c"] == j)).sum() for j in cl] for i in rl],
                   float)
    rv, cv = ds.var(row), ds.var(col)
    clabels = [str(value_text(cv, j)) for j in cl]
    rows = [[str(value_text(rv, i))] + [fmt_int(x) for x in tab[a_]] + [fmt_int(tab[a_].sum())]
            for a_, i in enumerate(rl)]
    rows.append(["Total"] + [fmt_int(x) for x in tab.sum(axis=0)] + [fmt_int(tab.sum())])
    sec.table(Table(f"{rv.title} * {cv.title} Crosstabulation",
                    [""] + [(cv.title, c) for c in clabels] + ["Total"], rows))
    if _want_bf(o):
        bf10 = gunel_dickey_bf10(tab, sampling, a)
        exp_ = tab.sum(axis=1, keepdims=True) * tab.sum(axis=0, keepdims=True) / tab.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            chi = float(np.nansum((tab - exp_) ** 2 / exp_))
        df = (len(rl) - 1) * (len(cl) - 1)
        how = {"joint": "joint multinomial", "rows": "independent multinomial (row totals "
               "fixed)", "cols": "independent multinomial (column totals fixed)"}[sampling]
        sec.table(Table("Tests of Independence", ["", "Value", "df", "Asymp. Sig. (2-sided)",
                                                  "Bayes Factorᵃ"],
                        [["Pearson Chi-Square", _f(chi), fmt_int(df), fmt_p(stats.chi2.sf(chi, df)),
                          _bf(_bf_out(bf10, o))]],
                        footnotes=["a. " + _bf_note(o, f"Gunel-Dickey Bayes factor, {how} "
                                                       f"sampling, Dirichlet prior a = {a:g}. "
                                                       "The null hypothesis is "
                                                       "independence.")]))
    if _want_post(o):
        alpha = tab + a
        tot = alpha.sum()
        prow = []
        for i_, i in enumerate(rl):
            for j_, j in enumerate(cl):
                al = alpha[i_, j_]
                dist = stats.beta(al, tot - al)
                mode = (al - 1) / (tot - 2) if al > 1 else 0.0
                prow.append([str(value_text(rv, i)), str(value_text(cv, j))] +
                            _from_dist(dist, mode, o.level).row())
        sec.table(Table("Posterior Distribution Characterization for Cell Proportions",
                        [rv.title, cv.title] + _post_cols(o.level), prow, row_header_cols=2,
                        footnotes=[f"Dirichlet({a:g}) prior on the cell probabilities "
                                   "(joint multinomial)."]))
    return sec


def selftest() -> None:
    bf = jzs_ttest_bf10(2.0, 30)
    assert 0.5 < bf < 2.5
    assert abs(rouder_anova_bf10(np.r_[np.zeros(10), np.ones(10)] +
                                 np.tile(np.linspace(-1, 1, 10), 2),
                                 np.r_[["a"] * 10, ["b"] * 10]) -
               jzs_ttest_bf10(stats.ttest_ind(np.linspace(-1, 1, 10) + 1,
                                              np.linspace(-1, 1, 10)).statistic, 10, 10,
                              0.5 * math.sqrt(2))) < 1e-3
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"x": rng.normal(0.4, 1, 40), "y": rng.normal(size=40),
                       "g": np.repeat([1.0, 2.0], 20), "c": rng.poisson(3, 40).astype(float)})
    df["y"] += df["x"]
    ds = Dataset(df)
    o = BayesOptions(plots=True)
    one_sample_normal(ds, ["x"], 0, o)
    independent_normal(ds, ["x"], "g", 1.0, 2.0, o)
    pearson_correlation(ds, ["x", "y"], o)
    linear_regression(ds, "y", [], ["x"], o)
    oneway_anova(ds, ["y"], "g", o)
    one_sample_poisson(ds, ["c"], 3.0, o)
    one_sample_binomial(ds, ["g"], 2.0, 0.5, o)
    loglinear(ds, "g", "g", o)
