"""Analyze > Generalized Linear Models (SPSS GENLIN: generalized linear models and generalized
estimating equations), Analyze > Regression > Probit (SPSS PROBIT) and Analyze > Loglinear
(General = GENLOG Poisson/multinomial, Logit, Model Selection = HILOGLINEAR).

GENLIN mirrors SPSS's defaults: factors use the last category as redundant ("Set to zero
because this parameter is redundant"), the scale of normal / gamma / inverse Gaussian / Tweedie
models is the maximum-likelihood estimate, the covariance matrix comes from the observed
Hessian (Hybrid / Newton-Raphson) or the expected one (Fisher), optionally robust (Huber-White),
the log likelihood is the full one, Type III tests use sum-to-zero contrasts and binary
responses model the probability of the non-reference category (reference last by default, so
the lower category is modelled, as SPSS does). Fitting uses statsmodels GLM / GEE.

PROBIT fits probit or logit dose-response models to grouped data with an optional natural
response rate, Pearson goodness-of-fit with the heterogeneity factor (used when its p < .15),
Fieller fiducial limits for the dose at each probability and relative median potency.

The loglinear procedures are Poisson GLMs on the cell counts (delta = 0.5 added to every cell
of saturated models, as SPSS does).
"""
from __future__ import annotations

import itertools
import math
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

from rsstatchem.core.dataset import Dataset, format_code
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import fmt, fmt_int, fmt_p, numeric, pct, sup, value_text

DISTRIBUTIONS = {"normal": "Normal", "binomial": "Binomial", "gamma": "Gamma",
                 "igauss": "Inverse Gaussian", "poisson": "Poisson",
                 "negbin": "Negative binomial", "tweedie": "Tweedie"}
LINKS = {"identity": "Identity", "log": "Log", "logit": "Logit", "probit": "Probit",
         "cloglog": "Complementary log-log", "nloglog": "Negative log-log",
         "power": "Power", "cauchit": "Cauchit", "inverse": "Power(-1)"}
MODEL_TYPES = {  # SPSS "Type of Model" presets
    "linear": ("normal", "identity"), "gamma_log": ("gamma", "log"),
    "tweedie_log": ("tweedie", "log"), "tweedie_identity": ("tweedie", "identity"),
    "poisson": ("poisson", "log"), "negbin": ("negbin", "log"),
    "binary_logistic": ("binomial", "logit"), "binary_probit": ("binomial", "probit"),
    "interval_censored": ("binomial", "cloglog")}
CORR_TYPES = {"independent": "Independent", "exchangeable": "Exchangeable",
              "ar1": "AR(1)", "mdependent": "M-dependent", "unstructured": "Unstructured"}
ADJUSTMENTS = {"lsd": "Least significant difference", "bonferroni": "Bonferroni",
               "sidak": "Sidak", "seqbonferroni": "Sequential Bonferroni",
               "seqsidak": "Sequential Sidak"}


@dataclass
class GenlinResult:
    section: Section
    saved: dict = field(default_factory=dict)


def _sort_key(v):
    return (isinstance(v, str), v)


def _sig(p) -> str:
    return fmt_p(p) if p == p else ""


# ====================================================================== families / links
def _link(name: str, power: float = 1.0):
    from statsmodels.genmod.families import links as L

    return {"identity": L.Identity, "log": L.Log, "logit": L.Logit, "probit": L.Probit,
            "cloglog": L.CLogLog, "nloglog": L.LogLog, "cauchit": L.Cauchy,
            "inverse": L.InversePower}[name]() if name != "power" else L.Power(power)


def _family(dist: str, link: str, power: float = 1.0, nb_k: float = 1.0,
            tweedie_p: float = 1.5):
    from statsmodels.genmod import families as F

    lk = _link(link, power)
    if dist == "normal":
        return F.Gaussian(lk)
    if dist == "binomial":
        return F.Binomial(lk)
    if dist == "gamma":
        return F.Gamma(lk)
    if dist == "igauss":
        return F.InverseGaussian(lk)
    if dist == "poisson":
        return F.Poisson(lk)
    if dist == "negbin":
        return F.NegativeBinomial(lk, alpha=nb_k)
    if dist == "tweedie":
        return F.Tweedie(lk, var_power=tweedie_p, eql=False)
    raise ValueError(dist)


def _tweedie_loglike(y, mu, phi, p, w):
    """Full Tweedie log-likelihood by series summation (Dunn & Smyth 2005)."""
    out = np.zeros(len(y))
    phi_i = phi / w
    for i, (yi, mi, ph) in enumerate(zip(y, mu, phi_i)):
        if yi == 0:
            out[i] = -mi ** (2 - p) / (ph * (2 - p))
            continue
        alpha = (2 - p) / (1 - p)
        js = np.arange(1, 200)
        logz = (-alpha * np.log(yi) + alpha * np.log(p - 1) - (1 - alpha) * np.log(ph)
                - np.log(2 - p))
        logw = js * logz - special.gammaln(js + 1) - special.gammaln(-alpha * js)
        lw = special.logsumexp(logw) - np.log(yi)
        theta = mi ** (1 - p) / (1 - p)
        kappa = mi ** (2 - p) / (2 - p)
        out[i] = lw + (yi * theta - kappa) / ph
    return float(out.sum())


def _loglike(family, dist, y, mu, phi, w, n_trials=None, tweedie_p=1.5) -> float:
    if dist == "tweedie":
        return _tweedie_loglike(y, mu, phi, tweedie_p, w)
    if dist == "binomial":
        n = n_trials if n_trials is not None else np.ones(len(y))
        r = y * n
        with np.errstate(divide="ignore", invalid="ignore"):
            ll = (special.gammaln(n + 1) - special.gammaln(r + 1) - special.gammaln(n - r + 1)
                  + r * np.log(np.clip(mu, 1e-300, None))
                  + (n - r) * np.log(np.clip(1 - mu, 1e-300, None)))
        return float(np.sum(w * ll))
    return float(family.loglike(y, mu, var_weights=w, scale=phi))


def _ml_scale(family, dist, y, mu, w, tweedie_p=1.5) -> float:
    """Maximum-likelihood scale: closed form for normal / inverse Gaussian, 1-D search for
    gamma and Tweedie (the scale is orthogonal to the coefficients in a GLM)."""
    n = len(y)
    dev = float(np.sum(family.resid_dev(y, mu, var_weights=w) ** 2))
    if dist in ("normal", "igauss"):
        return dev / n
    f = (lambda lp: -_loglike(family, dist, y, mu, math.exp(lp), w, tweedie_p=tweedie_p))
    start = math.log(max(dev / n, 1e-8))
    res = optimize.minimize_scalar(f, bracket=(start - 1, start + 1))
    return math.exp(res.x)


# ====================================================================== design
def _build(frame: pd.DataFrame, factors: list[str], terms: list[str], flev: dict,
           coding: str = "dummy"):
    """Design columns per term. ``coding`` dummy = last level redundant (SPSS parameter
    estimates), deviation = sum-to-zero (Type III tests)."""
    cols, out = {}, []
    for t in terms:
        parts = [p for p in t.split("*") if p]
        comps = []
        for p in parts:
            if p in factors:
                lv = flev[p]
                k = len(lv)
                if coding == "dummy":
                    comps.append([(f"[{p}={format_code(v)}]",
                                   (frame[p] == v).astype(float).where(frame[p].notna()),
                                   i == k - 1) for i, v in enumerate(lv)])
                else:
                    last = (frame[p] == lv[-1]).astype(float)
                    comps.append([(f"[{p}={format_code(v)}]",
                                   ((frame[p] == v).astype(float) - last).where(
                                       frame[p].notna()), i == k - 1)
                                  for i, v in enumerate(lv)])
            else:
                comps.append([(p, frame[p].astype(float), False)])
        all_names, keep = [], []
        for combo in itertools.product(*comps):
            name = " * ".join(c[0] for c in combo)
            all_names.append(name)
            if not any(c[2] for c in combo):
                s = combo[0][1]
                for c in combo[1:]:
                    s = s * c[1]
                cols[name] = s
                keep.append(name)
        out.append((" * ".join(parts), keep, all_names))
    X = pd.DataFrame(cols, index=frame.index) if cols else pd.DataFrame(index=frame.index)
    return X, out


def main_terms(factors, covariates) -> list[str]:
    return list(factors) + list(covariates)


# ====================================================================== fitting core
@dataclass
class _GLMFit:
    params: np.ndarray
    cov: np.ndarray
    mu: np.ndarray
    eta: np.ndarray
    phi: float
    ll: float
    deviance: float
    pearson: float
    model: object
    res: object
    n: int
    converged: bool
    iterations: int
    nb_k: float = 1.0


def _fit_glm(y, X, family, dist, w, offset, scale_method, scale_value, cov_type, method,
             n_trials=None, tweedie_p=1.5, max_iter=100, nb_estimate=False, link="log",
             power=1.0) -> _GLMFit:
    import statsmodels.api as sm

    nb_k = getattr(family, "alpha", 1.0)
    if nb_estimate and dist == "negbin":
        def nll(lk):
            fam = _family("negbin", link, power, nb_k=math.exp(lk))
            r = sm.GLM(y, X, family=fam, var_weights=w, offset=offset).fit(maxiter=max_iter)
            return -_loglike(fam, dist, y, r.fittedvalues, 1.0, w)
        nb_k = math.exp(optimize.minimize_scalar(nll, bracket=(-2, 1)).x)
        family = _family("negbin", link, power, nb_k=nb_k)
    endog = y
    vw = w if n_trials is None else w * n_trials
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.GLM(endog, X, family=family, var_weights=vw, offset=offset)
        res = model.fit(maxiter=max_iter, tol=1e-10)
    mu = np.asarray(res.fittedvalues, float)
    eta = X @ np.asarray(res.params, float) + (0 if offset is None else offset)
    dev = float(res.deviance)
    pear = float(res.pearson_chi2)
    n, p = X.shape
    if dist in ("binomial", "poisson", "negbin") and scale_method == "ml":
        phi = 1.0
    elif scale_method == "ml":
        phi = _ml_scale(family, dist, y, mu, vw, tweedie_p)
    elif scale_method == "deviance":
        phi = dev / (n - p)
    elif scale_method == "pearson":
        phi = pear / (n - p)
    else:
        phi = float(scale_value)
    params = np.asarray(res.params, float)
    H = model.hessian(params, scale=phi, observed=(method != "fisher"))
    try:
        cov = np.linalg.inv(-H)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(-H)
    if cov_type == "robust":
        S = model.score_obs(params, scale=phi)
        cov = cov @ (S.T @ S) @ cov
    ll = _loglike(family, dist, y, mu, phi, w, n_trials=n_trials, tweedie_p=tweedie_p)
    it = int(getattr(res, "fit_history", {}).get("iteration", 0) or 0)
    return _GLMFit(params, cov, mu, eta, phi, ll, dev, pear, model, res, n,
                   bool(getattr(res, "converged", True)), it, nb_k)


def _wald(b, C) -> float:
    try:
        return float(b @ np.linalg.solve(C, b))
    except np.linalg.LinAlgError:
        return float(b @ np.linalg.pinv(C) @ b)


def _adjust(pvals: list[float], method: str) -> list[float]:
    m = len(pvals)
    p = np.asarray(pvals, float)
    if method == "bonferroni":
        return list(np.minimum(p * m, 1))
    if method == "sidak":
        return list(1 - (1 - p) ** m)
    if method in ("seqbonferroni", "seqsidak"):
        order = np.argsort(p)
        out = np.empty(m)
        prev = 0.0
        for rank, i in enumerate(order):
            k = m - rank
            adj = min(1.0, p[i] * k) if method == "seqbonferroni" else 1 - (1 - p[i]) ** k
            prev = max(prev, adj)
            out[i] = prev
        return list(out)
    return list(p)


# ====================================================================== GENLIN / GEE
def genlin(ds: Dataset, dependent: str, factors: list[str] = (), covariates: list[str] = (),
           distribution: str = "normal", link: str = "identity", *, terms=None,
           intercept: bool = True, offset: str | None = None, scale_weight: str | None = None,
           trials: str | float | None = None, reference: str = "last",
           factor_order: str = "ascending", power: float = 1.0, tweedie_p: float = 1.5,
           nb_k: float = 1.0, nb_estimate: bool = False, scale_method: str = "ml",
           scale_value: float = 1.0, cov_type: str = "model", method: str = "hybrid",
           max_iter: int = 100, level: float = 0.95, analysis_type: int = 3,
           chi_type: str = "wald", exp_params: bool = False, cov_matrix: bool = False,
           corr_matrix: bool = False, descriptives: bool = True, iteration_history: bool = False,
           emmeans: list[str] = (), emm_scale: str = "original",
           emm_adjust: str = "lsd", emm_pairwise: bool = True, save=(),
           repeated: dict | None = None) -> GenlinResult:
    """SPSS GENLIN. ``repeated`` turns on GEE: {"subject": [vars], "within": [vars],
    "corr": independent|exchangeable|ar1|mdependent|unstructured, "m": 1,
    "cov": robust|model}. ``trials`` is a variable name or a fixed number for events/trials
    binomial responses. ``save`` keys: mean, ci, xb, xbse, resid, pearson, deviance,
    stdpearson, stddeviance, likelihood, leverage, cooks."""
    factors, covariates = list(factors), list(covariates)
    terms = list(terms) if terms else main_terms(factors, covariates)
    gee = repeated is not None
    title = "Generalized Estimating Equations" if gee else "Generalized Linear Models"
    section = Section(title)
    result = GenlinResult(section)
    dvar = ds.var(dependent)
    section.log(_genlin_syntax(dependent, factors, covariates, distribution, link, terms,
                               intercept, offset, scale_weight, trials, reference,
                               scale_method, cov_type, method, level, analysis_type,
                               chi_type, emmeans, save, repeated, power, tweedie_p, nb_k,
                               nb_estimate))

    # ---------------------------------------------------------------- data
    need = [f for f in factors] + covariates + ([offset] if offset else []) + \
        ([scale_weight] if scale_weight else []) + \
        ([trials] if isinstance(trials, str) else [])
    if gee:
        need += list(repeated.get("subject", [])) + list(repeated.get("within", []))
    raw = pd.DataFrame({"__y": ds.analysis_series(dependent)}, index=ds.df.index)
    for v in dict.fromkeys(need):
        raw[v] = ds.analysis_series(v) if (v in factors or (gee and v in
                                                             repeated.get("subject", []) +
                                                             repeated.get("within", [])))\
            else numeric(ds, v)
    total = len(raw)
    frame = raw.dropna()
    if scale_weight:
        frame = frame[frame[scale_weight] > 0]
    n = len(frame)
    flev = {}
    for f in factors:
        lv = sorted(frame[f].unique(), key=_sort_key)
        flev[f] = lv if factor_order == "ascending" else lv[::-1]
    section.table(Table("Case Processing Summary", ["", "N", "Percent"], [
        ["Included", fmt_int(n), fmt(100 * n / total, 1) + "%"],
        ["Excluded", fmt_int(total - n), fmt(100 * (total - n) / total, 1) + "%"],
        ["Total", fmt_int(total), "100.0%"]]))
    if n < 2:
        section.note("Warning", "Not enough valid cases.", style="warning")
        return result

    # ---------------------------------------------------------------- response
    n_trials = None
    resp_note = ""
    if distribution == "binomial":
        if trials is not None:
            n_trials = (frame[trials].to_numpy(float) if isinstance(trials, str)
                        else np.full(n, float(trials)))
            yv = pd.to_numeric(frame["__y"], errors="coerce").to_numpy(float)
            if np.any(yv > n_trials) or np.any(yv < 0):
                section.note("Warning", "The number of events exceeds the number of trials "
                             "for some cases.", style="warning")
                return result
            y = yv / n_trials
        else:
            levels = sorted(frame["__y"].unique(), key=_sort_key)
            if len(levels) != 2:
                section.note("Warning", "A binary response must have exactly two values.",
                             style="warning")
                return result
            ref_v = levels[-1] if reference == "last" else levels[0]
            resp_v = levels[0] if reference == "last" else levels[-1]
            y = (frame["__y"] == resp_v).to_numpy(float)
            resp_note = (f"The procedure models {value_text(dvar, resp_v)} as the response, "
                         f"treating {value_text(dvar, ref_v)} as the reference category.")
    else:
        y = pd.to_numeric(frame["__y"], errors="coerce").to_numpy(float)
        if distribution in ("gamma", "igauss") and np.any(y <= 0):
            section.note("Warning", "The gamma and inverse Gaussian distributions need a "
                         "positive dependent variable.", style="warning")
            return result
        if distribution in ("poisson", "negbin", "tweedie") and np.any(y < 0):
            section.note("Warning", "The dependent variable must be non-negative.",
                         style="warning")
            return result
    w = frame[scale_weight].to_numpy(float) if scale_weight else np.ones(n)
    off = frame[offset].to_numpy(float) if offset else None

    Xd, tdummy = _build(frame, factors, terms, flev, "dummy")
    if intercept:
        Xd.insert(0, "(Intercept)", 1.0)
    X = Xd.to_numpy(float)
    names = list(Xd.columns)
    p = X.shape[1]
    family = _family(distribution, link, power, nb_k, tweedie_p)

    # ---------------------------------------------------------------- model information
    info = [["Dependent Variable", dvar.title + (sup("a") if resp_note else "")],
            ["Probability Distribution", DISTRIBUTIONS[distribution]],
            ["Link Function", LINKS[link] if link != "power" else f"Power({power:g})"]]
    if n_trials is not None:
        info = [["Events Variable", dvar.title],
                ["Trials Variable" if isinstance(trials, str) else "Number of Trials",
                 ds.var(trials).title if isinstance(trials, str) else f"{trials:g}"]] + \
            info[1:]
    if offset:
        info.append(["Offset Variable", ds.var(offset).title])
    if scale_weight:
        info.append(["Scale Weight Variable", ds.var(scale_weight).title])
    if gee:
        subj = repeated.get("subject", [])
        info.append(["Subject Effect", ", ".join(ds.var(s).title for s in subj)])
        if repeated.get("within"):
            info.append(["Within-Subject Effect", ", ".join(ds.var(s).title
                                                            for s in repeated["within"])])
        info.append(["Working Correlation Matrix Structure",
                     CORR_TYPES[repeated.get("corr", "independent")]
                     + (f" ({repeated.get('m', 1)})" if repeated.get("corr") == "mdependent"
                        else "")])
    section.table(Table("Model Information", ["", ""], info,
                        footnotes=[f"a. {resp_note}"] if resp_note else []))

    if descriptives:
        _variable_information(section, ds, frame, factors, covariates, dependent,
                              distribution, n_trials, flev)

    model_note = f"Dependent Variable: {dvar.title}"
    model_terms_txt = ", ".join((["(Intercept)"] if intercept else [])
                                + [t for t, _, _ in tdummy])
    model_note2 = f"Model: {model_terms_txt}"

    if gee:
        return _gee_part(section, result, ds, frame, y, X, names, tdummy, family, distribution,
                         link, w, off, repeated, level, exp_params, cov_matrix, corr_matrix,
                         model_note, model_note2, factors, terms, flev, intercept, save,
                         n_trials, emmeans, emm_scale, emm_adjust, emm_pairwise, covariates,
                         max_iter)

    fit = _fit_glm(y, X, family, distribution, w, off, scale_method, scale_value, cov_type,
                   method, n_trials, tweedie_p, max_iter, nb_estimate, link, power)
    if nb_estimate:
        family = _family("negbin", link, power, nb_k=fit.nb_k)
    if iteration_history:
        section.note("Iteration History", f"Iteratively reweighted least squares (Fisher "
                     f"scoring) converged: {'yes' if fit.converged else 'no'}; final "
                     f"log likelihood {fmt(fit.ll, 3)}.")
    if not fit.converged:
        section.note("Warning", "The maximum number of iterations was reached; the estimates "
                     "may not be valid.", style="warning")

    # ---------------------------------------------------------------- goodness of fit
    ml_scale = scale_method == "ml" and distribution not in ("binomial", "poisson", "negbin")
    k = p + (1 if ml_scale else 0) + (1 if nb_estimate and distribution == "negbin" else 0)
    df_res = n - p
    ll = fit.ll
    aic = -2 * ll + 2 * k
    aicc = aic + 2 * k * (k + 1) / (n - k - 1) if n - k - 1 > 0 else float("nan")
    bic = -2 * ll + k * math.log(n)
    caic = -2 * ll + k * (math.log(n) + 1)
    gof = [["Deviance", fmt(fit.deviance), fmt_int(df_res), fmt(fit.deviance / df_res)],
           ["Scaled Deviance", fmt(fit.deviance / fit.phi), fmt_int(df_res), ""],
           ["Pearson Chi-Square", fmt(fit.pearson), fmt_int(df_res), fmt(fit.pearson / df_res)],
           ["Scaled Pearson Chi-Square", fmt(fit.pearson / fit.phi), fmt_int(df_res), ""],
           ["Log Likelihoodᵇ", fmt(ll), "", ""],
           ["Akaike's Information Criterion (AIC)", fmt(aic), "", ""],
           ["Finite Sample Corrected AIC (AICC)", fmt(aicc), "", ""],
           ["Bayesian Information Criterion (BIC)", fmt(bic), "", ""],
           ["Consistent AIC (CAIC)", fmt(caic), "", ""]]
    gnotes = [model_note, model_note2,
              "a. Information criteria are in smaller-is-better form.",
              "b. The full log likelihood function is displayed and used in computing "
              "information criteria."]
    if scale_method in ("deviance", "pearson"):
        gnotes.append(f"c. The scale parameter is estimated by the "
                      f"{'deviance' if scale_method == 'deviance' else 'Pearson chi-square'} "
                      "divided by its degrees of freedom.")
    section.table(Table("Goodness of Fitᵃ", ["", "Value", "df", "Value/df"], gof,
                        footnotes=gnotes))

    # ---------------------------------------------------------------- omnibus
    def refit(cols_keep: list[int], design=None) -> _GLMFit:
        Xr = (X if design is None else design)[:, cols_keep] if cols_keep else \
            np.zeros((n, 0))
        if Xr.shape[1] == 0:
            Xr = np.zeros((n, 1))
            if off is None:
                mu0 = np.full(n, family.link.inverse(np.zeros(1))[0])
            else:
                mu0 = family.link.inverse(off)
            phi0 = fit.phi
            return _GLMFit(np.zeros(0), np.zeros((0, 0)), mu0, np.zeros(n), phi0,
                           _loglike(family, distribution, y, mu0, phi0, w, n_trials,
                                    tweedie_p), float(np.sum(family.resid_dev(
                                        y, mu0, var_weights=w if n_trials is None else
                                        w * n_trials) ** 2)), 0.0, None, None, n, True, 0)
        return _fit_glm(y, Xr, family, distribution, w, off,
                        scale_method if scale_method == "ml" else "fixed",
                        fit.phi, "model", method, n_trials, tweedie_p, max_iter)

    def lr_stat(red: _GLMFit) -> float:
        if ml_scale:
            return 2 * (fit.ll - red.ll)
        if distribution in ("binomial", "poisson", "negbin") and scale_method == "ml":
            return 2 * (fit.ll - red.ll)
        return (red.deviance - fit.deviance) / fit.phi

    if intercept:
        red0 = refit([0])
        chi = lr_stat(red0)
        dfo = p - 1
        section.table(Table("Omnibus Testᵃ", ["Likelihood Ratio Chi-Square", "df", "Sig."], [
            [fmt(chi), fmt_int(dfo), _sig(stats.chi2.sf(chi, dfo)) if dfo else ""]],
            footnotes=[model_note, model_note2,
                       "a. Compares the fitted model against the intercept-only model."]))

    # ---------------------------------------------------------------- tests of model effects
    if analysis_type in (1, 3, 13):
        _effects_tables(section, analysis_type, chi_type, fit, X, y, frame, factors, terms,
                        flev, intercept, family, distribution, w, off, scale_method,
                        cov_type, method, n_trials, tweedie_p, max_iter, refit, lr_stat,
                        model_note, model_note2)

    # ---------------------------------------------------------------- parameter estimates
    _parameter_table(section, fit, names, tdummy, intercept, level, exp_params,
                     model_note, model_note2, distribution, scale_method, ml_scale,
                     nb_estimate, y, w, family, tweedie_p)
    if cov_matrix or corr_matrix:
        lab = names
        if cov_matrix:
            section.table(Table("Covariances of Parameter Estimates", [""] + lab,
                                [[lab[i]] + [fmt(v) for v in fit.cov[i]]
                                 for i in range(p)], footnotes=[model_note, model_note2]))
        if corr_matrix:
            sd = np.sqrt(np.clip(np.diag(fit.cov), 1e-300, None))
            R = fit.cov / np.outer(sd, sd)
            section.table(Table("Correlations of Parameter Estimates", [""] + lab,
                                [[lab[i]] + [fmt(v) for v in R[i]] for i in range(p)],
                                footnotes=[model_note, model_note2]))

    # ---------------------------------------------------------------- EM means
    for term in emmeans:
        _emmeans(section, ds, term, frame, factors, covariates, terms, flev, intercept,
                 fit.params, fit.cov, family, offset and float(np.mean(off)), emm_scale,
                 emm_adjust, emm_pairwise, level, dvar.title, chi="Wald")

    # ---------------------------------------------------------------- save
    if save:
        _save(result, ds, frame, save, fit, X, y, w, family, level, n_trials)
    return result


def _variable_information(section, ds, frame, factors, covariates, dependent, distribution,
                          n_trials, flev) -> None:
    n = len(frame)
    rows = []
    cat_vars = list(factors)
    if distribution == "binomial" and n_trials is None:
        cat_vars = [dependent] + cat_vars
    for v in cat_vars:
        var = ds.var(v)
        col = frame["__y" if v == dependent else v]
        lv = flev.get(v) or sorted(col.unique(), key=_sort_key)
        for i, lev in enumerate(lv):
            c = int((col == lev).sum())
            rows.append(["Dependent Variable" if v == dependent and i == 0 else
                         ("Factor" if i == 0 and v == cat_vars[1 if v != dependent and
                                                             cat_vars[0] == dependent
                                                             else 0] else ""),
                         var.title if i == 0 else "", value_text(var, lev), fmt_int(c),
                         fmt(100 * c / n, 1) + "%"])
        rows.append(["", "", "Total", fmt_int(n), "100.0%"])
    if rows:
        section.table(Table("Categorical Variable Information", ["", "", "", "N", "Percent"],
                            rows, row_header_cols=3))
    crow = []
    cont = ([] if distribution == "binomial" and n_trials is None else [dependent]) + \
        list(covariates)
    for i, v in enumerate(cont):
        x = pd.to_numeric(frame["__y" if v == dependent else v], errors="coerce")
        crow.append(["Dependent Variable" if v == dependent else
                     ("Covariate" if v == covariates[0] else ""), ds.var(v).title,
                     fmt_int(len(x)), fmt(x.min()), fmt(x.max()), fmt(x.mean(), 4),
                     fmt(x.std(ddof=1), 5)])
    if crow:
        section.table(Table("Continuous Variable Information", ["", "", "N", "Minimum",
                                                                "Maximum", "Mean",
                                                                "Std. Deviation"], crow,
                            row_header_cols=2))


def _effects_tables(section, analysis_type, chi_type, fit, X, y, frame, factors, terms, flev,
                    intercept, family, distribution, w, off, scale_method, cov_type, method,
                    n_trials, tweedie_p, max_iter, refit, lr_stat, model_note,
                    model_note2) -> None:
    n = len(y)
    rows = []
    which = [3] if analysis_type == 3 else [1] if analysis_type == 1 else [1, 3]
    for typ in which:
        rows = []
        if typ == 3:
            Xe, tdev = _build(frame, factors, terms, flev, "deviation")
            if intercept:
                Xe.insert(0, "(Intercept)", 1.0)
            E = Xe.to_numpy(float)
            fe = _fit_glm(y, E, family, distribution, w, off, "fixed", fit.phi, cov_type,
                          method, n_trials, tweedie_p, max_iter)
            cols = list(Xe.columns)
            groups = ([("(Intercept)", [0])] if intercept else []) + \
                [(t, [cols.index(c) for c in keep]) for t, keep, _ in tdev]
            for name, idx in groups:
                if not idx:
                    continue
                if chi_type == "lr":
                    keep = [i for i in range(E.shape[1]) if i not in idx]
                    red = refit(keep, E)
                    chi = lr_stat(red)
                else:
                    chi = _wald(fe.params[idx], fe.cov[np.ix_(idx, idx)])
                rows.append([name, fmt(chi), fmt_int(len(idx)),
                             _sig(stats.chi2.sf(max(chi, 0), len(idx)))])
        else:
            Xd, tdum = _build(frame, factors, terms, flev, "dummy")
            if intercept:
                Xd.insert(0, "(Intercept)", 1.0)
            cols = list(Xd.columns)
            groups = ([("(Intercept)", [0])] if intercept else []) + \
                [(t, [cols.index(c) for c in keep]) for t, keep, _ in tdum]
            if chi_type == "lr":
                upto: list[int] = []
                prev = None
                for name, idx in groups:
                    upto = upto + idx
                    cur = refit(list(upto))
                    if prev is None:
                        prev = refit([])
                    chi = (2 * (cur.ll - prev.ll) if scale_method == "ml"
                           else (prev.deviance - cur.deviance) / fit.phi)
                    rows.append([name, fmt(chi), fmt_int(len(idx)),
                                 _sig(stats.chi2.sf(max(chi, 0), len(idx)))])
                    prev = cur
            else:
                # Type I Wald: L = rows of the Cholesky factor of X'WX at the solution
                C = fit.cov
                info = np.linalg.pinv(C)
                try:
                    R = np.linalg.cholesky(info).T
                except np.linalg.LinAlgError:
                    R = np.linalg.qr(np.linalg.cholesky(info + 1e-12 * np.eye(len(info))).T)[1]
                for name, idx in groups:
                    L = R[idx]
                    chi = _wald(L @ fit.params, L @ C @ L.T)
                    rows.append([name, fmt(chi), fmt_int(len(idx)),
                                 _sig(stats.chi2.sf(max(chi, 0), len(idx)))])
        head = "Likelihood Ratio Chi-Square" if chi_type == "lr" else "Wald Chi-Square"
        section.table(Table(f"Tests of Model Effects", [
            "Source", (f"Type {'III' if typ == 3 else 'I'}", head),
            (f"Type {'III' if typ == 3 else 'I'}", "df"),
            (f"Type {'III' if typ == 3 else 'I'}", "Sig.")], rows,
            footnotes=[model_note, model_note2]))


def _parameter_table(section, fit, names, tdummy, intercept, level, exp_params, model_note,
                     model_note2, distribution, scale_method, ml_scale, nb_estimate, y, w,
                     family, tweedie_p) -> None:
    se = np.sqrt(np.clip(np.diag(fit.cov), 0, None))
    z = stats.norm.ppf(0.5 + level / 2)
    g = f"{pct(level)} Wald Confidence Interval"
    cols = ["Parameter", "B", "Std. Error", (g, "Lower"), (g, "Upper"),
            ("Hypothesis Test", "Wald Chi-Square"), ("Hypothesis Test", "df"),
            ("Hypothesis Test", "Sig.")]
    if exp_params:
        ge = f"{pct(level)} Wald Confidence Interval for Exp(B)"
        cols += ["Exp(B)", (ge, "Lower"), (ge, "Upper")]
    rows = []
    redundant = False

    def add(name, i):
        b, s = fit.params[i], se[i]
        wv = (b / s) ** 2 if s > 0 else float("nan")
        row = [name, fmt(b), fmt(s), fmt(b - z * s), fmt(b + z * s), fmt(wv), "1",
               _sig(stats.chi2.sf(wv, 1))]
        if exp_params:
            row += [fmt(_exp(b)), fmt(_exp(b - z * s)), fmt(_exp(b + z * s))]
        rows.append(row)

    if intercept:
        add("(Intercept)", 0)
    for t, keep, all_names in tdummy:
        for nm in all_names:
            if nm in keep:
                add(nm, names.index(nm))
            else:
                redundant = True
                rows.append([nm, "0" + sup("a"), "", "", "", "", "", ""]
                            + (["", "", ""] if exp_params else []))
    notes = [model_note, model_note2]
    if redundant:
        notes.append("a. Set to zero because this parameter is redundant.")
    blank = ["", "", "", ""] + (["", "", ""] if exp_params else [])
    if distribution == "negbin":
        if nb_estimate:
            rows.append(["(Negative binomial)", fmt(fit.nb_k), "", "", ""] + blank)
        else:
            rows.append(["(Negative binomial)", fmt_int(fit.nb_k) + sup("b")
                         if float(fit.nb_k).is_integer() else fmt(fit.nb_k) + sup("b"),
                         "", "", ""] + blank)
    if distribution in ("binomial", "poisson", "negbin") and scale_method == "ml":
        rows.append(["(Scale)", "1" + sup("b"), "", "", ""] + blank)
        notes.append("b. Fixed at the displayed value.")
    elif ml_scale:
        phi = fit.phi
        f = (lambda lp: _loglike(family, distribution, y, fit.mu, math.exp(lp), w,
                                 tweedie_p=tweedie_p))
        lp = math.log(phi)
        h = 1e-4
        d2 = (f(lp + h) - 2 * f(lp) + f(lp - h)) / h ** 2
        se_lp = math.sqrt(-1 / d2) if d2 < 0 else float("nan")
        rows.append(["(Scale)", fmt(phi) + sup("c"), fmt(phi * se_lp),
                     fmt(phi * math.exp(-z * se_lp)), fmt(phi * math.exp(z * se_lp))] + blank)
        notes.append("c. Maximum likelihood estimate.")
    else:
        rows.append(["(Scale)", fmt(fit.phi) + sup("b"), "", "", ""] + blank)
        notes.append("b. Computed based on the "
                     + ({"deviance": "deviance", "pearson": "Pearson chi-square"}.get(
                         scale_method, "fixed value")) + ".")
    section.table(Table("Parameter Estimates", cols, rows, footnotes=notes))


def _exp(x):
    return math.exp(x) if x < 700 else float("inf")


def _emmeans(section, ds, term, frame, factors, covariates, terms, flev, intercept, params,
             cov, family, offset_mean, scale, adjust, pairwise, level, dep_title,
             chi="Wald") -> None:
    parts = [p for p in term.split("*") if p in factors]
    if not parts:
        return
    others = [f for f in factors if f not in parts]
    grid = list(itertools.product(*[flev[f] for f in factors]))
    G = pd.DataFrame(grid, columns=factors) if factors else pd.DataFrame(index=[0])
    for c in covariates:
        G[c] = float(pd.to_numeric(frame[c], errors="coerce").mean())
    Xg, _ = _build(G, factors, terms, flev, "dummy")
    if intercept:
        Xg.insert(0, "(Intercept)", 1.0)
    Xg = Xg.to_numpy(float)
    targets = list(itertools.product(*[flev[f] for f in parts]))
    Ls = []
    for tv in targets:
        mask = np.ones(len(G), bool)
        for f, v in zip(parts, tv):
            mask &= (G[f] == v).to_numpy()
        Ls.append(Xg[mask].mean(axis=0))
    L = np.array(Ls)
    off = offset_mean or 0.0
    eta = L @ params + off
    Ceta = L @ cov @ L.T
    if scale == "original":
        mu = family.link.inverse(eta)
        d = 1 / family.link.deriv(mu)
        est = mu
        C = Ceta * np.outer(d, d)
    else:
        est = eta
        C = Ceta
    se = np.sqrt(np.clip(np.diag(C), 0, None))
    z = stats.norm.ppf(0.5 + level / 2)
    eta_se = np.sqrt(np.clip(np.diag(Ceta), 0, None))
    lo_eta, hi_eta = eta - z * eta_se, eta + z * eta_se
    if scale == "original":
        a, b = family.link.inverse(lo_eta), family.link.inverse(hi_eta)
        lo, hi = np.minimum(a, b), np.maximum(a, b)
    else:
        lo, hi = lo_eta, hi_eta
    titles = [ds.var(f).title for f in parts]
    rows = []
    labels = []
    for i, tv in enumerate(targets):
        lab = [value_text(ds.var(f), v) for f, v in zip(parts, tv)]
        labels.append(", ".join(lab))
        rows.append(lab + [fmt(est[i]), fmt(se[i]), fmt(lo[i]), fmt(hi[i])])
    g = f"{pct(level)} Wald Confidence Interval"
    section.table(Table(f"Estimates — {' * '.join(titles)}", titles + [
        "Mean" if scale == "original" else "Linear Predictor", "Std. Error", (g, "Lower"),
        (g, "Upper")], rows, row_header_cols=len(parts),
        footnotes=["Covariates appearing in the model are fixed at the following values: "
                   + ", ".join(f"{c}={fmt(float(pd.to_numeric(frame[c]).mean()), 4)}"
                               for c in covariates)] if covariates else []))
    if pairwise and len(targets) > 1:
        pairs = [(i, j) for i in range(len(targets)) for j in range(len(targets)) if i != j]
        diffs, ses, ps = [], [], []
        for i, j in pairs:
            dv = est[i] - est[j]
            sv = math.sqrt(max(C[i, i] + C[j, j] - 2 * C[i, j], 0))
            diffs.append(dv)
            ses.append(sv)
            ps.append(float(2 * stats.norm.sf(abs(dv / sv))) if sv > 0 else float("nan"))
        uniq = sorted({tuple(sorted(p)) for p in pairs})
        m = len(uniq)
        adj_u = dict(zip(uniq, _adjust([ps[pairs.index(u)] for u in uniq], adjust)))
        alpha = 1 - level
        za = stats.norm.ppf(1 - (alpha / m if adjust == "bonferroni" else
                                 (1 - (1 - alpha) ** (1 / m)) if adjust == "sidak" else alpha)
                            / 2)
        prow = []
        for (i, j), dv, sv in zip(pairs, diffs, ses):
            pa = adj_u[tuple(sorted((i, j)))]
            mark = "*" if pa < alpha else ""
            prow.append([labels[i], labels[j], fmt(dv) + mark, fmt(sv), "1", _sig(pa),
                         fmt(dv - za * sv), fmt(dv + za * sv)])
        g2 = f"{pct(level)} Wald Confidence Interval for Difference"
        section.table(Table(f"Pairwise Comparisons — {' * '.join(titles)}", [
            f"(I) {' * '.join(titles)}", f"(J) {' * '.join(titles)}", "Mean Difference (I-J)",
            "Std. Error", "df", "Sig.", (g2, "Lower"), (g2, "Upper")], prow,
            row_header_cols=2, footnotes=[
                "Pairwise comparisons of estimated marginal means based on the "
                + ("original scale of dependent variable " if scale == "original" else
                   "linear predictor ") + dep_title,
                "*. The mean difference is significant at the "
                f"{fmt(alpha, 2)} level.", f"Adjustment for multiple comparisons: "
                                            f"{ADJUSTMENTS[adjust]}."]))
        k = len(targets)
        D = np.zeros((k - 1, k))
        for i in range(k - 1):
            D[i, i], D[i, i + 1] = 1, -1
        chi2 = _wald(D @ est, D @ C @ D.T)
        section.table(Table(f"Overall Test Results — {' * '.join(titles)}",
                            ["Wald Chi-Square", "df", "Sig."],
                            [[fmt(chi2), fmt_int(k - 1), _sig(stats.chi2.sf(chi2, k - 1))]],
                            footnotes=["The overall test is a Wald test of equal estimated "
                                       "marginal means."]))


GENLIN_SAVE = {"mean": ("MeanPredicted", "Predicted Value of Mean of Response"),
               "ci": ("CIMeanPredicted", "Confidence Interval for Mean"),
               "xb": ("XBPredicted", "Predicted Value of Linear Predictor"),
               "xbse": ("XBStandardError", "Estimated Standard Error of Predicted Value of "
                                           "Linear Predictor"),
               "resid": ("Residual", "Raw Residual"),
               "pearson": ("PearsonResidual", "Pearson Residual"),
               "deviance": ("DevianceResidual", "Deviance Residual"),
               "stdpearson": ("StdPearsonResidual", "Standardized Pearson Residual"),
               "stddeviance": ("StdDevianceResidual", "Standardized Deviance Residual"),
               "likelihood": ("LikelihoodResidual", "Likelihood Residual"),
               "leverage": ("Leverage", "Leverage Value"),
               "cooks": ("CooksDistance", "Cook's Distance")}


def glm_diagnostics(fit: _GLMFit, X, y, w, family, n_trials=None) -> dict:
    vw = w if n_trials is None else w * n_trials
    mu, phi = fit.mu, fit.phi
    var = family.variance(mu)
    deriv = family.link.deriv(mu)
    W = vw / (var * deriv ** 2)
    G = np.linalg.pinv((X * W[:, None]).T @ X)
    h = W * np.einsum("ij,jk,ik->i", X, G, X)
    one_h = np.clip(1 - h, 1e-12, None)
    e = y - mu
    rp = e * np.sqrt(vw / var)
    rd = np.asarray(family.resid_dev(y, mu, var_weights=vw), float)
    sp = rp / np.sqrt(phi * one_h)
    sd = rd / np.sqrt(phi * one_h)
    lik = np.sign(e) * np.sqrt(h * sp ** 2 + (1 - h) * sd ** 2)
    p = X.shape[1]
    return {"resid": e, "pearson": rp, "deviance": rd, "stdpearson": sp, "stddeviance": sd,
            "likelihood": lik, "leverage": h, "cooks": sp ** 2 * h / (p * one_h)}


def _save(result, ds, frame, save, fit, X, y, w, family, level, n_trials) -> None:
    idx = ds.df.index
    z = stats.norm.ppf(0.5 + level / 2)
    eta = fit.eta
    xbse = np.sqrt(np.clip(np.einsum("ij,jk,ik->i", X, fit.cov, X), 0, None))
    dg = glm_diagnostics(fit, X, y, w, family, n_trials)

    def put(key, values, label=None, root=None):
        s = pd.Series(np.nan, index=idx)
        s[frame.index] = values
        r, lab = GENLIN_SAVE[key]
        result.saved[root or r] = (label or lab, s)

    for key in GENLIN_SAVE:
        if key not in save:
            continue
        if key == "mean":
            put(key, fit.mu)
        elif key == "ci":
            a, b = family.link.inverse(eta - z * xbse), family.link.inverse(eta + z * xbse)
            put(key, np.minimum(a, b), f"Lower Bound of {pct(level)} Confidence Interval for "
                                       "Mean", "CIMeanPredictedLower")
            put(key, np.maximum(a, b), f"Upper Bound of {pct(level)} Confidence Interval for "
                                       "Mean", "CIMeanPredictedUpper")
        elif key == "xb":
            put(key, eta)
        elif key == "xbse":
            put(key, xbse)
        else:
            put(key, dg[key])


# ====================================================================== GEE
def _gee_part(section, result, ds, frame, y, X, names, tdummy, family, distribution, link, w,
              off, repeated, level, exp_params, cov_matrix, corr_matrix, model_note,
              model_note2, factors, terms, flev, intercept, save, n_trials, emmeans,
              emm_scale, emm_adjust, emm_pairwise, covariates, max_iter):
    import statsmodels.api as sm
    from statsmodels.genmod import cov_struct as CS

    subj = repeated.get("subject", [])
    within = repeated.get("within", [])
    corr = repeated.get("corr", "independent")
    cov_kind = repeated.get("cov", "robust")
    key = frame[subj].astype(str).agg("|".join, axis=1)
    order = np.lexsort(tuple([frame[v].map(_rank_map(frame[v])).to_numpy() for v in
                              reversed(within)]) + (key.map(_rank_map(key)).to_numpy(),)) \
        if within else np.argsort(key.map(_rank_map(key)).to_numpy(), kind="stable")
    groups = key.map(_rank_map(key)).to_numpy()[order]
    time = None
    if within:
        wkey = frame[within].astype(str).agg("|".join, axis=1)
        wmap = _rank_map(wkey)
        time = wkey.map(wmap).to_numpy()[order]
    Xo, yo, wo = X[order], y[order], w[order]
    oo = off[order] if off is not None else None
    if corr == "exchangeable":
        cs = CS.Exchangeable()
    elif corr == "ar1":
        cs = CS.Autoregressive(grid=True)
    elif corr == "mdependent":
        cs = CS.Stationary(max_lag=int(repeated.get("m", 1)), grid=True)
    elif corr == "unstructured":
        cs = CS.Unstructured()
    else:
        cs = CS.Independence()
    kwargs = {}
    if time is not None:
        kwargs["time"] = time
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.GEE(yo, Xo, groups=groups, family=family, cov_struct=cs, offset=oo,
                       weights=wo if np.any(wo != 1) else None, **kwargs)
        res = model.fit(maxiter=max_iter, cov_type="robust" if cov_kind == "robust"
                        else "naive")
    params = np.asarray(res.params)
    cov = np.asarray(res.cov_params())
    sizes = pd.Series(groups).value_counts()
    section.table(Table("Correlated Data Summary", ["", "", ""], [
        ["Number of Levels", "Subject Effect", fmt_int(len(sizes))],
        ["Number of Subjects", "", fmt_int(len(sizes))],
        ["Number of Measurements per Subject", "Minimum", fmt_int(sizes.min())],
        ["", "Maximum", fmt_int(sizes.max())],
        ["Correlation Matrix Dimension", "", fmt_int(sizes.max())]], row_header_cols=2))
    try:
        qic, qicu = res.qic(res.scale)
    except Exception:  # noqa: BLE001 - QIC unavailable for some structures
        qic, qicu = float("nan"), float("nan")
    section.table(Table("Goodness of Fitᵃ", ["", "Value"], [
        ["Quasi Likelihood under Independence Model Criterion (QIC)ᵇ", fmt(qic)],
        ["Corrected Quasi Likelihood under Independence Model Criterion (QICC)ᵇ", fmt(qicu)]],
        footnotes=[model_note, model_note2, "a. Information criteria are in smaller-is-better "
                   "form.", "b. Computed using the full log quasi-likelihood function."]))
    # Type III Wald tests with sum-to-zero coding (reparametrised fit)
    Xe, tdev = _build(frame, factors, terms, flev, "deviation")
    if intercept:
        Xe.insert(0, "(Intercept)", 1.0)
    E = Xe.to_numpy(float)[order]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cs2 = type(cs)(**({"grid": True} if corr in ("ar1",) else
                          {"max_lag": int(repeated.get("m", 1)), "grid": True}
                          if corr == "mdependent" else {}))
        re = sm.GEE(yo, E, groups=groups, family=family, cov_struct=cs2, offset=oo,
                    **kwargs).fit(maxiter=max_iter, cov_type="robust" if cov_kind == "robust"
                                  else "naive")
    cols = list(Xe.columns)
    rows = []
    for name, idx in ([("(Intercept)", [0])] if intercept else []) + \
            [(t, [cols.index(c) for c in keep]) for t, keep, _ in tdev]:
        if not idx:
            continue
        chi = _wald(np.asarray(re.params)[idx], np.asarray(re.cov_params())[np.ix_(idx, idx)])
        rows.append([name, fmt(chi), fmt_int(len(idx)), _sig(stats.chi2.sf(chi, len(idx)))])
    section.table(Table("Tests of Model Effects", ["Source", ("Type III", "Wald Chi-Square"),
                                                   ("Type III", "df"), ("Type III", "Sig.")],
                        rows, footnotes=[model_note, model_note2]))
    fake = _GLMFit(params, cov, np.asarray(res.fittedvalues), Xo @ params, float(res.scale),
                   float("nan"), float("nan"), float("nan"), model, res, len(yo), True, 0)
    _parameter_table(section, fake, names, tdummy, intercept, level, exp_params, model_note,
                     model_note2, distribution, "fixed", False, False, yo, wo, family, 1.5)
    section.items[-1].footnotes.append("Robust covariance estimator used." if
                                       cov_kind == "robust" else
                                       "Model-based covariance estimator used.")
    dim = int(sizes.max())
    wc = working_correlation(res.cov_struct, corr, dim)
    if wc is not None:
        section.table(Table("Working Correlation Matrix", ["Measurement"]
                            + [f"[{i + 1}]" for i in range(dim)],
                            [[f"[{i + 1}]"] + [fmt(v) for v in wc[i]] for i in range(dim)],
                            footnotes=[model_note, model_note2]))
    if cov_matrix:
        section.table(Table("Covariances of Parameter Estimates", [""] + names,
                            [[names[i]] + [fmt(v) for v in cov[i]] for i in range(len(names))]))
    if corr_matrix:
        sd = np.sqrt(np.clip(np.diag(cov), 1e-300, None))
        Rm = cov / np.outer(sd, sd)
        section.table(Table("Correlations of Parameter Estimates", [""] + names,
                            [[names[i]] + [fmt(v) for v in Rm[i]] for i in range(len(names))]))
    for term in emmeans:
        _emmeans(section, ds, term, frame, factors, covariates, terms, flev, intercept,
                 params, cov, family, None, emm_scale, emm_adjust, emm_pairwise, level,
                 model_note.split(": ", 1)[1])
    if save:
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        fit_back = _GLMFit(params, cov, fake.mu[inv], (Xo @ params)[inv], fake.phi,
                           float("nan"), 0, 0, None, None, len(yo), True, 0)
        _save(result, ds, frame, save, fit_back, X, y, w, family, level, n_trials)
    return result


def working_correlation(cs, corr: str, dim: int) -> np.ndarray | None:
    i, j = np.indices((dim, dim))
    lag = np.abs(i - j)
    dep = cs.dep_params
    if corr == "independent":
        return np.eye(dim)
    if corr == "exchangeable":
        return np.where(lag == 0, 1.0, float(dep))
    if corr == "ar1":
        return float(dep) ** lag
    if corr == "mdependent":
        d = np.asarray(dep, float).ravel()
        return np.where(lag < len(d), d[np.minimum(lag, len(d) - 1)], 0.0)
    if corr == "unstructured":
        M = np.asarray(dep, float)
        if M.shape == (dim, dim):
            s = np.sqrt(np.diag(M))
            return M / np.outer(s, s)
    return None


def _rank_map(series: pd.Series) -> dict:
    vals = sorted(series.unique(), key=_sort_key)
    return {v: i for i, v in enumerate(vals)}


def _genlin_syntax(dep, factors, covariates, dist, link, terms, intercept, offset, weight,
                   trials, reference, scale_method, cov_type, method, level, analysis_type,
                   chi_type, emmeans, save, repeated, power, tweedie_p, nb_k, nb_est) -> str:
    dist_kw = {"normal": "NORMAL", "binomial": "BINOMIAL", "gamma": "GAMMA",
               "igauss": "IGAUSS", "poisson": "POISSON",
               "negbin": f"NEGBIN({'MLE' if nb_est else f'{nb_k:g}'})",
               "tweedie": f"TWEEDIE({tweedie_p:g})"}[dist]
    link_kw = {"nloglog": "NLOGLOG", "cloglog": "CLOGLOG", "power": f"POWER({power:g})",
               "inverse": "POWER(-1)"}.get(link, link.upper())
    lines = [f"GENLIN {dep}" + (f" (REFERENCE={reference.upper()})" if dist == "binomial"
                                and trials is None else "")
             + (f" OF {trials}" if trials is not None else "")
             + (" BY " + " ".join(factors) + " (ORDER=ASCENDING)" if factors else "")
             + (" WITH " + " ".join(covariates) if covariates else ""),
             "  /MODEL " + " ".join(t.replace("*", "*") for t in terms)
             + f" INTERCEPT={'YES' if intercept else 'NO'}"
             + (f" OFFSET={offset}" if offset else "") + (f" SCALEWEIGHT={weight}"
                                                          if weight else "")
             + f"\n    DISTRIBUTION={dist_kw} LINK={link_kw}",
             f"  /CRITERIA METHOD={'FISHER' if method == 'fisher' else 'NEWTON'}"
             + f" SCALE={'MLE' if scale_method == 'ml' else scale_method.upper()}"
             + f" COVB={'ROBUST' if cov_type == 'robust' else 'MODEL'}"
             + f" ANALYSISTYPE={ {1: '1', 3: '3', 13: 'ALL'}[analysis_type]}"
             + f"({'LR' if chi_type == 'lr' else 'WALD'}) CILEVEL={level * 100:g}"
             + " LIKELIHOOD=FULL"]
    if repeated:
        lines.append(f"  /REPEATED SUBJECT={'*'.join(repeated.get('subject', []))}"
                     + (f" WITHINSUBJECT={'*'.join(repeated['within'])}"
                        if repeated.get("within") else "")
                     + f" CORRTYPE={repeated.get('corr', 'independent').upper()}"
                     + f" COVB={repeated.get('cov', 'robust').upper()}")
    for t in emmeans:
        lines.append(f"  /EMMEANS TABLES={t} SCALE=ORIGINAL COMPARE={t}")
    if save:
        lines.append("  /SAVE " + " ".join(GENLIN_SAVE[k][0].upper() for k in GENLIN_SAVE
                                           if k in save))
    lines.append("  /PRINT CPS DESCRIPTIVES MODELINFO FIT SUMMARY SOLUTION.")
    return "\n".join(lines)


# ====================================================================== PROBIT
PROBIT_PROBS = np.r_[np.arange(1, 11) / 100, np.arange(15, 90, 5) / 100,
                     np.arange(90, 100) / 100]


def _probit_ll(par, x, r, n, link, nat, fixed_c, groups, ng):
    a = par[:ng]
    b = par[ng]
    c = fixed_c if nat != "estimate" else special.expit(par[ng + 1])
    eta = a[groups] + b * x
    F = stats.norm.cdf(eta) if link == "probit" else special.expit(eta)
    P = c + (1 - c) * F
    P = np.clip(P, 1e-12, 1 - 1e-12)
    return float(np.sum(r * np.log(P) + (n - r) * np.log(1 - P)))


def probit(ds: Dataset, response: str, total: str, covariate: str, factor: str | None = None,
           model: str = "probit", transform: str = "log10", natural: str = "none",
           natural_value: float = 0.0, level: float = 0.95, frequencies: bool = True,
           relative_potency: bool = False, parallelism: bool = False,
           fiducial: bool = True, het_alpha: float = 0.15) -> Section:
    """SPSS PROBIT with one dose covariate. ``transform`` none / log10 / ln; ``natural``
    none / estimate / value (natural response rate)."""
    from statsmodels.tools.numdiff import approx_hess3

    section = Section("Probit Analysis")
    tkw = {"none": "NONE", "log10": "LG10", "ln": "LN"}[transform]
    section.log(f"PROBIT {response} OF {total}" + (f" BY {factor}" if factor else "")
                + f" WITH {covariate}\n  /LOG {tkw}\n  /MODEL {model.upper()}"
                + "\n  /PRINT " + " ".join(["FREQ"] * frequencies + ["CI"] * fiducial
                                          + ["RMP"] * relative_potency
                                          + ["PARALL"] * parallelism)
                + f"\n  /CRITERIA ITERATE(20) STEPLIMIT(.1)"
                + ("\n  /NATRES" + (f"={natural_value:g}" if natural == "value" else "")
                   if natural != "none" else "") + ".")
    frame = pd.DataFrame({"r": numeric(ds, response), "n": numeric(ds, total),
                          "x": numeric(ds, covariate)})
    if factor:
        frame["f"] = ds.analysis_series(factor)
    frame = frame.dropna()
    n_all = len(frame)
    bad_resp = frame["r"] > frame["n"]
    control = frame["x"] <= 0 if transform != "none" else pd.Series(False, index=frame.index)
    valid = frame[~bad_resp & ~control]
    rows = [["Cases", "Valid", fmt_int(len(valid))],
            ["", "Rejected", ""],
            ["", "Missing", fmt_int(ds.n_cases - n_all)],
            ["", "LOG Transform Cannot be Done", fmt_int(int(control.sum()))
             if natural == "none" else "0"],
            ["", "Number of Responses > Number of Subjects", fmt_int(int(bad_resp.sum()))],
            ["", "Control Group", fmt_int(int(control.sum())) if natural != "none" else "0"]]
    flv = []
    if factor:
        flv = sorted(valid["f"].unique(), key=_sort_key)
        for i, v in enumerate(flv):
            rows.append([ds.var(factor).title if i == 0 else "",
                         value_text(ds.var(factor), v), fmt_int(int((valid["f"] == v).sum()))])
    section.table(Table("Data Information", ["", "", "N of Cases"], rows, row_header_cols=2))
    if len(valid) < 2:
        section.note("Warning", "Not enough valid dose groups.", style="warning")
        return section
    tf = {"none": lambda v: v, "log10": np.log10, "ln": np.log}[transform]
    x = tf(valid["x"].to_numpy(float))
    r = valid["r"].to_numpy(float)
    nn = valid["n"].to_numpy(float)
    if factor:
        groups = valid["f"].map({v: i for i, v in enumerate(flv)}).to_numpy()
        ng = len(flv)
    else:
        groups = np.zeros(len(x), int)
        ng = 1
    # control group data contribute to the natural response rate
    if natural == "estimate" and control.any():
        cg = frame[control & ~bad_resp]
        xc = np.full(len(cg), -np.inf)
        x_all = np.r_[x, xc]
        r_all = np.r_[r, cg["r"].to_numpy(float)]
        n_all_ = np.r_[nn, cg["n"].to_numpy(float)]
        g_all = np.r_[groups, np.zeros(len(cg), int)]
    else:
        x_all, r_all, n_all_, g_all = x, r, nn, groups
    fixed_c = natural_value if natural == "value" else 0.0
    # starting values from an unweighted linear fit on empirical probits
    pe = np.clip((r + 0.5) / (nn + 1), 1e-3, 1 - 1e-3)
    zt = stats.norm.ppf(pe) if model == "probit" else special.logit(pe)
    b0 = np.polyfit(x, zt, 1)
    start = np.r_[np.full(ng, b0[1]), b0[0]] + 0.0
    if natural == "estimate":
        p0 = (r_all[np.isinf(x_all)].sum() / n_all_[np.isinf(x_all)].sum()) \
            if np.isinf(x_all).any() else 0.05
        start = np.r_[start, special.logit(np.clip(p0, 0.01, 0.5))]

    def nll(par):
        with np.errstate(all="ignore"):
            return -_probit_ll(par, np.where(np.isinf(x_all), -1e6, x_all), r_all, n_all_,
                               model, natural, fixed_c, g_all, ng)

    opt = optimize.minimize(nll, start, method="BFGS", options={"gtol": 1e-9,
                                                                "maxiter": 10000})
    opt = optimize.minimize(nll, opt.x, method="Nelder-Mead",
                            options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000})
    par = opt.x
    H = approx_hess3(par, nll)
    cov = np.linalg.pinv(H)
    c_hat = special.expit(par[ng + 1]) if natural == "estimate" else fixed_c
    if natural == "estimate":
        d = c_hat * (1 - c_hat)
        J = np.eye(len(par))
        J[-1, -1] = d
        cov = J @ cov @ J.T
    a, b = par[:ng], par[ng]
    eta = a[groups] + b * x
    F = stats.norm.cdf(eta) if model == "probit" else special.expit(eta)
    P = c_hat + (1 - c_hat) * F
    expct = nn * P
    chi = float(np.sum((r - expct) ** 2 / (nn * P * (1 - P))))
    df = len(x) - (ng + 1) - (1 if natural == "estimate" else 0)
    pchi = float(stats.chi2.sf(chi, df)) if df > 0 else float("nan")
    het = df > 0 and pchi < het_alpha
    hf = chi / df if het else 1.0
    cov_h = cov * hf
    se = np.sqrt(np.clip(np.diag(cov_h), 0, None))
    zq = stats.t.ppf(0.5 + level / 2, df) if het else stats.norm.ppf(0.5 + level / 2)
    label = "PROBIT" if model == "probit" else "LOGIT"
    tnote = {"none": "", "log10": " (Covariates X are transformed using the base 10.000 "
                                  "logarithm.)",
             "ln": " (Covariates X are transformed using the base 2.718 logarithm.)"}[transform]
    prow = [[f"{label}ᵃ", ds.var(covariate).title, fmt(b), fmt(se[ng]), fmt(b / se[ng]),
             _sig(2 * stats.norm.sf(abs(b / se[ng]))), fmt(b - zq * se[ng]),
             fmt(b + zq * se[ng])]]
    for i in range(ng):
        name = "Interceptᵇ" if not factor else value_text(ds.var(factor), flv[i])
        prow.append(["", name if factor else "Interceptᵇ", fmt(a[i]), fmt(se[i]),
                     fmt(a[i] / se[i]), _sig(2 * stats.norm.sf(abs(a[i] / se[i]))),
                     fmt(a[i] - zq * se[i]), fmt(a[i] + zq * se[i])])
    if natural == "estimate":
        prow.append(["Natural Response", "", fmt(c_hat), fmt(se[-1]), "", "", "", ""])
    g = f"{pct(level)} Confidence Interval"
    section.table(Table("Parameter Estimates", ["Parameter", "", "Estimate", "Std. Error", "Z",
                                                "Sig.", (g, "Lower Bound"),
                                                (g, "Upper Bound")], prow, row_header_cols=2,
                        footnotes=[f"a. {label} model: {label}(p) = Intercept + BX{tnote}",
                                   "b. Corresponds to the grouping variable " +
                                   ds.var(factor).title if factor else
                                   "b. Intercept of the probit line."]))
    crow = [["Pearson Goodness-of-Fit Test", fmt(chi), fmt_int(df), _sig(pchi)
             + (sup("b") if het else sup("a"))]]
    if parallelism and factor and ng > 1:
        def nll_sep(pp):
            tot = 0.0
            for gi in range(ng):
                m = groups == gi
                eta_g = pp[2 * gi] + pp[2 * gi + 1] * x[m]
                Fg = stats.norm.cdf(eta_g) if model == "probit" else special.expit(eta_g)
                Pg = np.clip(c_hat + (1 - c_hat) * Fg, 1e-12, 1 - 1e-12)
                tot += np.sum(r[m] * np.log(Pg) + (nn[m] - r[m]) * np.log(1 - Pg))
            return -tot
        sp = np.ravel([[a[i], b] for i in range(ng)])
        sep = optimize.minimize(nll_sep, sp, method="BFGS")
        pchi2 = 2 * (opt.fun - sep.fun)
        crow.append(["Parallelism Test", fmt(pchi2), fmt_int(ng - 1),
                     _sig(stats.chi2.sf(pchi2, ng - 1)) + sup("c")])
    section.table(Table("Chi-Square Tests", ["", "", "Chi-Square", "df", "Sig."],
                        [[label] + row for row in crow], row_header_cols=2, footnotes=[
                            "a. Since the significance level is greater than "
                            f"{fmt(het_alpha, 3)}, no heterogeneity factor is used in the "
                            "calculation of confidence limits." if not het else
                            "b. Since the significance level is less than "
                            f"{fmt(het_alpha, 3)}, a heterogeneity factor is used in the "
                            "calculation of confidence limits."]
                        + (["c. Statistics based on individual cases differ from statistics "
                            "based on aggregated cases (likelihood-ratio test of a common "
                            "slope)."] if parallelism and factor else [])))
    if frequencies:
        frows = []
        for i in range(len(x)):
            frows.append([str(i + 1)] + ([value_text(ds.var(factor), flv[groups[i]])]
                                         if factor else [])
                         + [fmt(x[i], 3), fmt_int(nn[i]), fmt_int(r[i]), fmt(expct[i], 3),
                            fmt(r[i] - expct[i], 3), fmt(P[i], 3)])
        section.table(Table("Cell Counts and Residuals", ["Number"] + (
            [ds.var(factor).title] if factor else []) + [
            ds.var(covariate).title + ("ᵃ" if transform != "none" else ""),
            "Number of Subjects", "Observed Responses", "Expected Responses", "Residual",
            "Probability"], frows, row_header_cols=0,
            footnotes=[f"a. Logarithm base = {'10' if transform == 'log10' else 'e'}."]
            if transform != "none" else []))
    if fiducial:
        inv_t = {"none": lambda v: v, "log10": lambda v: 10 ** v, "ln": np.exp}[transform]
        lrows = []
        for gi in range(ng):
            ia, ib = gi, ng
            V = cov_h[np.ix_([ia, ib], [ia, ib])]
            for k, pr in enumerate(PROBIT_PROBS):
                zp = stats.norm.ppf(pr) if model == "probit" else special.logit(pr)
                x0 = (zp - a[gi]) / b
                lo, hi = fieller(a[gi], b, V, zp, zq)
                head = (value_text(ds.var(factor), flv[gi]) if factor and k == 0 else "")
                row = ([head] if factor else []) + [fmt(pr, 3), fmt(inv_t(x0), 3),
                                                    fmt(inv_t(lo), 3), fmt(inv_t(hi), 3)]
                if transform != "none":
                    row += [fmt(x0, 3), fmt(lo, 3), fmt(hi, 3)]
                lrows.append(row)
        g1 = f"{pct(level)} Confidence Limits for {ds.var(covariate).title}"
        cols = ([ds.var(factor).title] if factor else []) + ["Probability", (g1, "Estimate"),
                                                             (g1, "Lower Bound"),
                                                             (g1, "Upper Bound")]
        if transform != "none":
            g2 = f"{pct(level)} Confidence Limits for log({ds.var(covariate).title})ᵇ"
            cols += [(g2, "Estimate"), (g2, "Lower Bound"), (g2, "Upper Bound")]
        notes = []
        if het:
            notes.append("a. A heterogeneity factor is used.")
        if transform != "none":
            notes.append(f"b. Logarithm base = {'10' if transform == 'log10' else 'e'}.")
        section.table(Table("Confidence Limits", cols, lrows,
                            row_header_cols=1 if factor else 0, footnotes=notes))
    if relative_potency and factor and ng > 1:
        rrows = []
        inv_t = {"none": lambda v: v, "log10": lambda v: 10 ** v, "ln": np.exp}[transform]
        for i in range(ng):
            for j in range(ng):
                if i == j:
                    continue
                # log relative potency = (a_i - a_j) / b  (dose_j / dose_i at equal response)
                V = cov_h[np.ix_([i, j, ng], [i, j, ng])]
                num_ = a[i] - a[j]
                v_num = V[0, 0] + V[1, 1] - 2 * V[0, 1]
                c_nb = V[0, 2] - V[1, 2]
                lo, hi = fieller(num_, b, np.array([[v_num, c_nb], [c_nb, V[2, 2]]]), 0.0, zq,
                                 ratio=True)
                est = num_ / b
                rrows.append([f"{i + 1}", f"{j + 1}", fmt(inv_t(est), 3), fmt(inv_t(lo), 3),
                              fmt(inv_t(hi), 3)])
        g3 = f"{pct(level)} Confidence Limits"
        section.table(Table("Relative Median Potency Estimates", [
            f"{ds.var(factor).title} (I)", f"{ds.var(factor).title} (J)", (g3, "Estimate"),
            (g3, "Lower Bound"), (g3, "Upper Bound")], rrows, row_header_cols=2,
            footnotes=["Levels: " + ", ".join(f"{k + 1} = {value_text(ds.var(factor), v)}"
                                              for k, v in enumerate(flv))]))
    return section


def fieller(a: float, b: float, V: np.ndarray, z: float, t: float,
            ratio: bool = False) -> tuple[float, float]:
    """Fieller limits for x = (z - a) / b (or (a - z) / b with ``ratio``), V = cov(a, b)."""
    va, vb, vab = V[0, 0], V[1, 1], V[0, 1]
    num_ = (a - z) if ratio else (z - a)
    sign_ab = vab if ratio else -vab
    A = b * b - t * t * vb
    B = -2 * (num_ * b - t * t * sign_ab)
    C = num_ * num_ - t * t * va
    disc = B * B - 4 * A * C
    if A <= 0 or disc < 0:
        return float("nan"), float("nan")
    r1 = (-B - math.sqrt(disc)) / (2 * A)
    r2 = (-B + math.sqrt(disc)) / (2 * A)
    return min(r1, r2), max(r1, r2)


# ====================================================================== loglinear
def _cells(ds: Dataset, factors: list[str], cell_cov: list[str] = (), weights: str | None = None):
    frame = pd.DataFrame({f: ds.analysis_series(f) for f in factors})
    for c in cell_cov:
        frame[c] = numeric(ds, c)
    frame = frame.dropna()
    flev = {f: sorted(frame[f].unique(), key=_sort_key) for f in factors}
    grid = pd.DataFrame(list(itertools.product(*[flev[f] for f in factors])), columns=factors)
    counts = frame.groupby(factors, sort=False).size()
    grid["count"] = [float(counts.get(tuple(r) if len(factors) > 1 else r[0], 0))
                     for r in grid[factors].itertuples(index=False)]
    for c in cell_cov:
        means_ = frame.groupby(factors)[c].mean()
        grid[c] = [float(means_.get(tuple(r) if len(factors) > 1 else r[0], np.nan))
                   for r in grid[factors].itertuples(index=False)]
    return grid, flev, len(frame)


def _poisson_fit(counts, X):
    import statsmodels.api as sm

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sm.GLM(counts, X, family=sm.families.Poisson()).fit(maxiter=200, tol=1e-12)
    return res


def _gof(obs, exp_) -> tuple[float, float]:
    with np.errstate(divide="ignore", invalid="ignore"):
        g2 = float(2 * np.nansum(np.where(obs > 0, obs * np.log(obs / exp_), 0)))
        x2 = float(np.nansum((obs - exp_) ** 2 / exp_))
    return g2, x2


def _hier_terms(gen: list[str]) -> list[str]:
    """All lower-order relatives of a generating class, ordered by size."""
    out = set()
    for t in gen:
        parts = t.split("*")
        for k in range(1, len(parts) + 1):
            for c in itertools.combinations(parts, k):
                out.add("*".join(c))
    return sorted(out, key=lambda s: (s.count("*"), s))


def loglinear_general(ds: Dataset, factors: list[str], terms: list[str] | None = None,
                      cell_covariates: list[str] = (), distribution: str = "poisson",
                      delta: float = 0.5, level: float = 0.95, frequencies: bool = True,
                      residuals: bool = True, estimates: bool = True) -> Section:
    """SPSS GENLOG: saturated model (``terms`` None) or custom terms."""
    section = Section("General Loglinear Analysis")
    saturated = terms is None
    terms = terms or ["*".join(c) for k in range(1, len(factors) + 1)
                      for c in itertools.combinations(factors, k)]
    section.log("GENLOG " + " ".join(factors)
                + (" WITH " + " ".join(cell_covariates) if cell_covariates else "")
                + f"\n  /MODEL={distribution.upper()}\n  /PRINT=FREQ RESID ESTIM"
                + f"\n  /CRITERIA=CIN({level * 100:g}) ITERATE(20) CONVERGE(0.001) "
                  f"DELTA({delta:g})"
                + "\n  /DESIGN " + " ".join(t for t in terms) + ".")
    grid, flev, ncases = _cells(ds, factors, cell_covariates)
    obs = grid["count"].to_numpy(float)
    used = obs + (delta if saturated else 0.0)
    X, tdum = _build(grid, factors, [t for t in terms], flev, "dummy")
    X.insert(0, "Constant", 1.0)
    res = _poisson_fit(used, X.to_numpy(float))
    exp_ = np.asarray(res.fittedvalues)
    section.table(Table("Data Information", ["", "", "N"], [
        ["Cases", "Valid", fmt_int(ncases)], ["", "Missing", fmt_int(ds.n_cases - ncases)],
        ["Cells", "Defined Cells", fmt_int(len(grid))],
        ["", "Structural Zeros", "0"], ["", "Sampling Zeros", fmt_int(int((obs == 0).sum()))]]
        + [["Categories", ds.var(f).title, fmt_int(len(flev[f]))] for f in factors],
        row_header_cols=2))
    g2, x2 = _gof(obs if not saturated else used, exp_)
    df = len(grid) - X.shape[1]
    section.table(Table("Goodness-of-Fit Testsᵃ ᵇ", ["", "Value", "df", "Sig."], [
        ["Likelihood Ratio", fmt(g2), fmt_int(df), _sig(stats.chi2.sf(g2, df)) if df > 0
         else ""],
        ["Pearson Chi-Square", fmt(x2), fmt_int(df), _sig(stats.chi2.sf(x2, df)) if df > 0
         else ""]], footnotes=[f"a. Model: {distribution.capitalize()}",
                               "b. Design: Constant + " + " + ".join(terms)]))
    if frequencies or residuals:
        _cell_residual_table(section, ds, grid, factors, obs, exp_, res, X.to_numpy(float),
                             residuals)
    if estimates:
        _param_rows(section, res, list(X.columns), tdum, level, "Constant",
                    [f"a. Model: {distribution.capitalize()}",
                     "b. Design: Constant + " + " + ".join(terms)])
    return section


def _cell_residual_table(section, ds, grid, factors, obs, exp_, res, X, residuals) -> None:
    N = obs.sum()
    W = exp_
    G = np.linalg.pinv((X * W[:, None]).T @ X)
    h = W * np.einsum("ij,jk,ik->i", X, G, X)
    rows = []
    for i in range(len(grid)):
        lab = [value_text(ds.var(f), grid.iloc[i][f]) for f in factors]
        row = lab + [fmt(obs[i], 3), fmt(100 * obs[i] / N, 1) + "%", fmt(exp_[i], 3),
                     fmt(100 * exp_[i] / N, 1) + "%"]
        if residuals:
            e = obs[i] - exp_[i]
            with np.errstate(divide="ignore", invalid="ignore"):
                dev = np.sign(e) * math.sqrt(max(2 * ((obs[i] * math.log(obs[i] / exp_[i])
                                                       if obs[i] > 0 else 0) - e), 0))
            row += [fmt(e, 3), fmt(e / math.sqrt(exp_[i]), 3),
                    fmt(e / math.sqrt(exp_[i] * max(1 - h[i], 1e-12)), 3), fmt(dev, 3)]
        rows.append(row)
    cols = [ds.var(f).title for f in factors] + [("Observed", "Count"), ("Observed", "%"),
                                                  ("Expected", "Count"), ("Expected", "%")]
    if residuals:
        cols += ["Residual", "Standardized Residual", "Adjusted Residual", "Deviance"]
    section.table(Table("Cell Counts and Residuals", cols, rows, row_header_cols=len(factors)))


def _param_rows(section, res, names, tdum, level, const_name, notes) -> None:
    params = np.asarray(res.params)
    se = np.asarray(res.bse)
    z = stats.norm.ppf(0.5 + level / 2)
    g = f"{pct(level)} Confidence Interval"
    rows = [[const_name, fmt(params[0]), fmt(se[0]), fmt(params[0] / se[0]),
             _sig(2 * stats.norm.sf(abs(params[0] / se[0]))), fmt(params[0] - z * se[0]),
             fmt(params[0] + z * se[0])]]
    redundant = False
    for t, keep, alln in tdum:
        for nm in alln:
            if nm in keep:
                i = names.index(nm)
                rows.append([nm, fmt(params[i]), fmt(se[i]), fmt(params[i] / se[i]),
                             _sig(2 * stats.norm.sf(abs(params[i] / se[i]))),
                             fmt(params[i] - z * se[i]), fmt(params[i] + z * se[i])])
            else:
                redundant = True
                rows.append([nm, "0" + sup("c"), "", "", "", "", ""])
    section.table(Table("Parameter Estimatesᵃ ᵇ", ["Parameter", "Estimate", "Std. Error", "Z",
                                                   "Sig.", (g, "Lower Bound"),
                                                   (g, "Upper Bound")], rows,
                        footnotes=notes + (["c. This parameter is set to zero because it is "
                                            "redundant."] if redundant else [])))


def loglinear_logit(ds: Dataset, dependents: list[str], factors: list[str],
                    terms: list[str] | None = None, delta: float = 0.5, level: float = 0.95,
                    frequencies: bool = True, residuals: bool = True,
                    estimates: bool = True) -> Section:
    """SPSS GENLOG logit model: dependent factor(s) explained by factor(s). The design always
    contains the full interaction of the independent factors (fixed margins)."""
    section = Section("Logit Loglinear Analysis")
    allf = list(dependents) + list(factors)
    saturated = terms is None
    if saturated:
        terms = []
        for d in dependents:
            terms.append(d)
            for k in range(1, len(factors) + 1):
                for c in itertools.combinations(factors, k):
                    terms.append("*".join((d,) + c))
    section.log("GENLOG " + " ".join(dependents) + " BY " + " ".join(factors)
                + f"\n  /MODEL=MULTINOMIAL\n  /PRINT=FREQ RESID ESTIM"
                + f"\n  /CRITERIA=CIN({level * 100:g}) ITERATE(20) CONVERGE(0.001) "
                  f"DELTA({delta:g})\n  /DESIGN " + " ".join(terms) + ".")
    grid, flev, ncases = _cells(ds, allf)
    obs = grid["count"].to_numpy(float)
    used = obs + (delta if saturated else 0.0)
    margin = ["*".join(c) for k in range(1, len(factors) + 1)
              for c in itertools.combinations(factors, k)]
    X, tdum = _build(grid, allf, margin + terms, flev, "dummy")
    X.insert(0, "Constant", 1.0)
    res = _poisson_fit(used, X.to_numpy(float))
    exp_ = np.asarray(res.fittedvalues)
    g2, x2 = _gof(used if saturated else obs, exp_)
    df = len(grid) - X.shape[1]
    design = "Constant + " + " + ".join(terms)
    section.table(Table("Goodness-of-Fit Testsᵃ ᵇ", ["", "Value", "df", "Sig."], [
        ["Likelihood Ratio", fmt(g2), fmt_int(df), _sig(stats.chi2.sf(g2, df)) if df > 0
         else ""],
        ["Pearson Chi-Square", fmt(x2), fmt_int(df), _sig(stats.chi2.sf(x2, df)) if df > 0
         else ""]], footnotes=["a. Model: Multinomial Logit", f"b. Design: {design}"]))
    # Analysis of dispersion (entropy and concentration), per SPSS GENLOG
    key = grid[factors].astype(str).agg("|".join, axis=1) if factors else \
        pd.Series("all", index=grid.index)
    n_i = pd.Series(obs).groupby(key.values).transform("sum").to_numpy()
    N = obs.sum()
    pj = pd.Series(obs).groupby(grid[dependents].astype(str).agg("|".join, axis=1).values
                                ).sum().to_numpy() / N
    with np.errstate(divide="ignore", invalid="ignore"):
        pi = exp_ / n_i
        ent_tot = float(-N * np.sum(pj * np.log(pj)))
        ent_res = float(-np.nansum(obs * np.log(pi)))
        con_tot = float(N / 2 * (1 - np.sum(pj ** 2)))
        con_res = float(0.5 * np.sum(obs * (1 - pi)))
    section.table(Table("Analysis of Dispersionᵃ ᵇ", ["Source of Dispersion", "Entropy",
                                                      "Concentration", "df"], [
        ["Model", fmt(ent_tot - ent_res), fmt(con_tot - con_res), fmt_int(X.shape[1]
                                                                         - 1 - len(margin))],
        ["Residual", fmt(ent_res), fmt(con_res), fmt_int(df)],
        ["Total", fmt(ent_tot), fmt(con_tot), ""]],
        footnotes=["a. Model: Multinomial Logit", f"b. Design: {design}"]))
    section.table(Table("Measures of Associationᵃ ᵇ", ["", ""], [
        ["Entropy", fmt((ent_tot - ent_res) / ent_tot) if ent_tot else ""],
        ["Concentration", fmt((con_tot - con_res) / con_tot) if con_tot else ""]],
        footnotes=["a. Model: Multinomial Logit", f"b. Design: {design}"]))
    if frequencies or residuals:
        _cell_residual_table(section, ds, grid, allf, obs, exp_, res, X.to_numpy(float),
                             residuals)
    if estimates:
        keep_terms = [t for t in tdum if any(d in t[0].split(" * ") for d in dependents)]
        names = list(X.columns)
        params, se = np.asarray(res.params), np.asarray(res.bse)
        z = stats.norm.ppf(0.5 + level / 2)
        g = f"{pct(level)} Confidence Interval"
        rows = []
        for t, keep, alln in keep_terms:
            for nm in alln:
                if nm in keep:
                    i = names.index(nm)
                    rows.append([nm, fmt(params[i]), fmt(se[i]), fmt(params[i] / se[i]),
                                 _sig(2 * stats.norm.sf(abs(params[i] / se[i]))),
                                 fmt(params[i] - z * se[i]), fmt(params[i] + z * se[i])])
                else:
                    rows.append([nm, "0" + sup("c"), "", "", "", "", ""])
        section.table(Table("Parameter Estimatesᵃ ᵇ", ["Parameter", "Estimate", "Std. Error",
                                                       "Z", "Sig.", (g, "Lower Bound"),
                                                       (g, "Upper Bound")], rows,
                            footnotes=["a. Model: Multinomial Logit", f"b. Design: {design}",
                                       "c. This parameter is set to zero because it is "
                                       "redundant.", "Constants for the independent factor "
                                                     "cells are not shown."]))
    return section


def loglinear_selection(ds: Dataset, factors: list[str], method: str = "backward",
                        generating: list[str] | None = None, p_remove: float = 0.05,
                        max_steps: int = 10, delta: float = 0.5, frequencies: bool = True,
                        residuals: bool = True, association: bool = True,
                        estimates: bool = True, level: float = 0.95) -> Section:
    """SPSS HILOGLINEAR: K-way tests, partial associations, backward elimination of a
    hierarchical model (or the given generating class entered in a single step)."""
    section = Section("Model Selection Loglinear Analysis")
    F = len(factors)
    gen = generating or ["*".join(factors)]
    section.log("HILOGLINEAR " + " ".join(f"{f}(1 {len(set(ds.analysis_series(f).dropna()))})"
                                         for f in factors)
                + f"\n  /METHOD={'BACKWARD' if method == 'backward' else 'ENTER'}"
                + f"\n  /CRITERIA MAXSTEPS({max_steps}) P({p_remove:g}) ITERATION(20) "
                  f"DELTA({delta:g})\n  /PRINT=FREQ RESID" + (" ASSOCIATION"
                                                               if association else "")
                + (" ESTIM" if estimates else "")
                + "\n  /DESIGN " + " ".join(gen) + ".")
    grid, flev, ncases = _cells(ds, factors)
    obs = grid["count"].to_numpy(float)
    section.table(Table("Data Information", ["", "", "N"], [
        ["Cases", "Valid", fmt_int(ncases)], ["", "Out of Range", "0"],
        ["", "Missing", fmt_int(ds.n_cases - ncases)],
        ["", "Weighted Valid", fmt_int(ncases)]]
        + [["Categories", ds.var(f).title, fmt_int(len(flev[f]))] for f in factors],
        row_header_cols=2))

    cache: dict = {}

    def fit_terms(hterms: list[str]):
        key = tuple(sorted(hterms))
        if key not in cache:
            X, _ = _build(grid, factors, list(hterms), flev, "dummy")
            X.insert(0, "Constant", 1.0)
            res = _poisson_fit(obs, X.to_numpy(float))
            e = np.asarray(res.fittedvalues)
            g2, x2 = _gof(obs, e)
            cache[key] = (g2, x2, len(grid) - X.shape[1], e, res, X)
        return cache[key]

    all_terms = _hier_terms(["*".join(factors)])
    krows, kwr = [], []
    stats_k = []
    for K in range(1, F + 1):
        lower = [t for t in all_terms if t.count("*") + 1 < K]
        g2, x2, df, *_ = fit_terms(lower)
        stats_k.append((g2, x2, df))
    stats_k.append((0.0, 0.0, 0))
    for K in range(1, F + 1):
        g2, x2, df = stats_k[K - 1]
        krows.append([str(K), fmt_int(df), fmt(g2), _sig(stats.chi2.sf(g2, df)) if df else "",
                      fmt(x2), _sig(stats.chi2.sf(x2, df)) if df else "", "0" if K == 1
                      else "2"])
    for K in range(1, F + 1):
        g2 = stats_k[K - 1][0] - stats_k[K][0]
        x2 = stats_k[K - 1][1] - stats_k[K][1]
        df = stats_k[K - 1][2] - stats_k[K][2]
        kwr.append([str(K), fmt_int(df), fmt(g2), _sig(stats.chi2.sf(g2, df)) if df else "",
                    fmt(x2), _sig(stats.chi2.sf(x2, df)) if df else "", "0"])
    section.table(Table("K-Way and Higher-Order Effects", [
        "", "K", "df", ("Likelihood Ratio", "Chi-Square"), ("Likelihood Ratio", "Sig."),
        ("Pearson", "Chi-Square"), ("Pearson", "Sig."), "Number of Iterations"],
        [["K-way and Higher Order Effectsᵃ" if i == 0 else ""] + r
         for i, r in enumerate(krows)]
        + [["K-way Effectsᵇ" if i == 0 else ""] + r for i, r in enumerate(kwr)],
        row_header_cols=2, footnotes=[
            "df used for these tests have NOT been adjusted for structural or sampling zeros. "
            "Tests using these df may be conservative.",
            "a. Tests that k-way and higher order effects are zero.",
            "b. Tests that k-way effects are zero."]))
    if association and F >= 3:
        prow = []
        for k in range(1, F):
            kterms = [t for t in all_terms if t.count("*") + 1 <= k]
            g_all = fit_terms(kterms)[0]
            for t in [t for t in all_terms if t.count("*") + 1 == k]:
                red = [x for x in kterms if x != t]
                g_red, _, df_red, *_ = fit_terms(red)
                df_full = fit_terms(kterms)[2]
                chi = g_red - g_all
                d = df_red - df_full
                prow.append([t.replace("*", "*"), fmt_int(d), fmt(chi),
                             _sig(stats.chi2.sf(chi, d)), "2"])
        section.table(Table("Partial Associations", ["Effect", "df", "Partial Chi-Square",
                                                     "Sig.", "Number of Iterations"], prow))
    if estimates:
        Xs, tdev = _build(grid, factors, all_terms, flev, "deviation")
        Xs.insert(0, "Constant", 1.0)
        res = _poisson_fit(obs + delta, Xs.to_numpy(float))
        params, se = np.asarray(res.params), np.asarray(res.bse)
        z = stats.norm.ppf(0.5 + level / 2)
        names = list(Xs.columns)
        rows = []
        for t, keep, _ in reversed(tdev):
            for j, nm in enumerate(keep):
                i = names.index(nm)
                rows.append([t if j == 0 else "", str(j + 1), fmt(params[i]), fmt(se[i]),
                             fmt(params[i] / se[i]), _sig(2 * stats.norm.sf(abs(params[i]
                                                                                / se[i]))),
                             fmt(params[i] - z * se[i]), fmt(params[i] + z * se[i])])
        g = f"{pct(level)} Confidence Interval"
        section.table(Table("Parameter Estimates", ["Effect", "Parameter", "Estimate",
                                                    "Std. Error", "Z", "Sig.",
                                                    (g, "Lower Bound"), (g, "Upper Bound")],
                            rows, row_header_cols=1,
                            footnotes=[f"Saturated model with delta = {delta:g} added to each "
                                       "cell; effect (deviation) coding, last category "
                                       "omitted."]))
    # backward elimination
    current = list(gen)
    srows = []

    def gc_text(g):
        return ", ".join(x for x in g) if g else "(none)"

    g2c, x2c, dfc, *_ = fit_terms(_hier_terms(current))
    step = 0
    srows.append([str(step), "Generating Classᵇ", gc_text(current), fmt(g2c), fmt_int(dfc),
                  _sig(stats.chi2.sf(g2c, dfc)) if dfc else "", ""])
    if method == "backward":
        while step < max_steps:
            cands = []
            for t in current:
                new = [x for x in current if x != t]
                parts = t.split("*")
                if len(parts) > 1:
                    for c in itertools.combinations(parts, len(parts) - 1):
                        cc = "*".join(c)
                        if not any(set(cc.split("*")) <= set(x.split("*")) for x in new):
                            new.append(cc)
                g2n, _, dfn, *_ = fit_terms(_hier_terms(new))
                chi = g2n - g2c
                d = dfn - dfc
                cands.append((float(stats.chi2.sf(chi, d)) if d else 1.0, t, new, chi, d))
            if not cands:
                break
            pbest = max(cands, key=lambda c: c[0])
            srows.append(["", "Deleted Effect", pbest[1], fmt(pbest[3]), fmt_int(pbest[4]),
                          _sig(pbest[0]), ""])
            if pbest[0] <= p_remove:
                srows[-1][1] = "Deleted Effect (not removed)"
                srows.pop()
                break
            step += 1
            current = pbest[2]
            g2c, x2c, dfc, *_ = fit_terms(_hier_terms(current))
            srows.append([str(step), "Generating Classᵇ", gc_text(current), fmt(g2c),
                          fmt_int(dfc), _sig(stats.chi2.sf(g2c, dfc)) if dfc else "", ""])
    section.table(Table("Step Summary", ["Stepᵃ", "", "Effects", "Chi-Squareᶜ", "df", "Sig.",
                                         "Number of Iterations"], srows, row_header_cols=2,
                        footnotes=["a. At each step, the effect with the largest significance "
                                   "level for the Likelihood Ratio Change is deleted, provided "
                                   f"the significance level is larger than {fmt(p_remove)}.",
                                   "b. Statistics are displayed for the best model at each "
                                   "step after step 0.",
                                   "c. For 'Deleted Effect', this is the change in the "
                                   "Chi-Square after the effect is deleted from the model."]))
    g2, x2, df, e, res, X = fit_terms(_hier_terms(current))
    section.note("Convergence Information", f"Generating Class: {gc_text(current)}")
    if frequencies or residuals:
        rows = []
        for i in range(len(grid)):
            lab = [value_text(ds.var(f), grid.iloc[i][f]) for f in factors]
            ee = obs[i] - e[i]
            rows.append(lab + [fmt(obs[i], 3), fmt(e[i], 3)]
                        + ([fmt(ee, 3), fmt(ee / math.sqrt(e[i]), 3)] if residuals else []))
        section.table(Table("Cell Counts and Residuals", [ds.var(f).title for f in factors]
                            + ["Observed Count", "Expected Count"]
                            + (["Residuals", "Std. Residuals"] if residuals else []), rows,
                            row_header_cols=F))
    section.table(Table("Goodness-of-Fit Tests", ["", "Chi-Square", "df", "Sig."], [
        ["Likelihood Ratio", fmt(g2), fmt_int(df), _sig(stats.chi2.sf(g2, df)) if df else ""],
        ["Pearson", fmt(x2), fmt_int(df), _sig(stats.chi2.sf(x2, df)) if df else ""]]))
    return section


def selftest() -> None:
    rng = np.random.default_rng(1)
    n = 200
    x = rng.normal(size=n)
    g = rng.integers(1, 4, n).astype(float)
    cnt = rng.poisson(np.exp(0.5 + 0.3 * x + 0.2 * (g == 1)))
    b = (rng.random(n) < special.expit(0.5 * x)).astype(float)
    subj = np.repeat(np.arange(50), 4).astype(float)
    t = np.tile(np.arange(1, 5), 50).astype(float)
    ds = Dataset(pd.DataFrame({"cnt": cnt.astype(float), "x": x, "g": g, "b": b,
                               "s": subj, "t": t, "y": 2 + x + rng.normal(size=n),
                               "pos": np.exp(0.2 * x + rng.normal(0, .3, n))}))
    genlin(ds, "cnt", ["g"], ["x"], "poisson", "log", emmeans=["g"], save=("mean", "cooks"))
    genlin(ds, "pos", ["g"], ["x"], "gamma", "log", analysis_type=13, chi_type="lr")
    genlin(ds, "b", [], ["x"], "binomial", "logit", exp_params=True)
    genlin(ds, "y", ["g"], ["x"], repeated={"subject": ["s"], "within": ["t"],
                                            "corr": "exchangeable"})
    dose = pd.DataFrame({"d": [10.2, 7.7, 5.1, 3.8, 2.6], "n": [50, 49, 46, 48, 50.],
                         "r": [44, 42, 24, 16, 6.]})
    probit(Dataset(dose), "r", "n", "d")
    loglinear_general(ds, ["g", "b"])
    loglinear_selection(ds, ["g", "b", "t"])
    loglinear_logit(ds, ["b"], ["g"])
