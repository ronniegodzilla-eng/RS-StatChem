"""Logistic (binomial, logit link) and Poisson (log link) regression for DOE responses, the
generalized linear models Design-Expert offers for pass/fail and count responses.

The model is fitted by iteratively reweighted least squares (McCullagh & Nelder 1989) on the
same model matrix as the least-squares analysis (coded factors, effect-coded categoric
factors and blocks, Scheffé terms for mixtures). :class:`GLMFit` follows the interface of
:class:`rsstatchem.core.doe.model.Fit`, so the Model tab (Auto Select by p-value, AICc or
BIC), the model graphs, optimization and point prediction work unchanged: predictions are
on the response scale (probability or expected count) and ``predict_coded`` returns the
linear predictor with its variance factor (dispersion fixed at 1).

Tests (instead of F tests):

* each term: likelihood-ratio chi-square of dropping it (Type III), with Wald z values in the
  coefficient table;
* the model: likelihood ratio against the intercept (and blocks) only;
* goodness of fit: deviance and Pearson chi-square on the residual df, and the Hosmer-Lemeshow
  test for binary responses.

statsmodels (BSD-3) serves as the oracle in the tests.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import special, stats

from rsstatchem.core.doe import model
from rsstatchem.core.doe.model import (Analysis, Fit, FitSummary, Observations, Space,
                                       Transform, build_matrix, drop_aliased, key_term,
                                       model_orders, model_terms, sort_terms, term_name)

FAMILY_TITLES = {"binomial": "Binomial (logit link)", "poisson": "Poisson (log link)"}
EPS = 1e-10


# ---------------------------------------------------------------------------- IRLS
def _mu(eta: np.ndarray, family: str) -> np.ndarray:
    if family == "binomial":
        return np.clip(special.expit(eta), EPS, 1 - EPS)
    return np.exp(np.clip(eta, -700, 700))


def _link(mu: np.ndarray, family: str) -> np.ndarray:
    return special.logit(mu) if family == "binomial" else np.log(mu)


def _variance(mu: np.ndarray, family: str) -> np.ndarray:
    return mu * (1 - mu) if family == "binomial" else mu


def _xlogy(x, y):
    return special.xlogy(x, y)


def unit_deviance(y: np.ndarray, mu: np.ndarray, m: np.ndarray, family: str) -> np.ndarray:
    if family == "binomial":
        return 2 * m * (_xlogy(y, y) - _xlogy(y, mu) + _xlogy(1 - y, 1 - y)
                        - _xlogy(1 - y, 1 - mu))
    return 2 * (_xlogy(y, y) - _xlogy(y, mu) - (y - mu))


def loglik(y: np.ndarray, mu: np.ndarray, m: np.ndarray, family: str) -> float:
    if family == "binomial":
        s = m * y
        const = special.gammaln(m + 1) - special.gammaln(s + 1) - special.gammaln(m - s + 1)
        return float(np.sum(const + _xlogy(s, mu) + _xlogy(m - s, 1 - mu)))
    return float(np.sum(_xlogy(y, mu) - mu - special.gammaln(y + 1)))


def irls(X: np.ndarray, y: np.ndarray, m: np.ndarray, family: str, max_iter: int = 100,
         tol: float = 1e-12) -> dict:
    """Maximum-likelihood fit of a canonical-link GLM. ``y`` is the proportion (binomial,
    with ``m`` trials) or the count (Poisson, m = 1)."""
    n, p = X.shape
    if family == "binomial":
        mu = (m * y + 0.5) / (m + 1)
    else:
        mu = y + 0.1 * max(float(np.mean(y)), 0.1)
    eta = _link(mu, family)
    dev = float(np.sum(unit_deviance(y, mu, m, family)))
    converged = False
    beta = np.zeros(p)
    it = 0
    for it in range(1, max_iter + 1):
        var = _variance(mu, family)
        w = m * var                      # canonical link: dmu/deta = V(mu)
        zw = eta + (y - mu) / var
        XtW = X.T * w
        try:
            beta_new = np.linalg.solve(XtW @ X, XtW @ zw)
        except np.linalg.LinAlgError:
            beta_new = np.linalg.lstsq(XtW @ X, XtW @ zw, rcond=None)[0]
        eta_new = X @ beta_new
        mu_new = _mu(eta_new, family)
        dev_new = float(np.sum(unit_deviance(y, mu_new, m, family)))
        step = 0
        while not np.isfinite(dev_new) or (dev_new > dev + 1e-9 * (1 + abs(dev)) and step < 30):
            beta_new = 0.5 * (beta_new + beta)   # step halving
            eta_new = X @ beta_new
            mu_new = _mu(eta_new, family)
            dev_new = float(np.sum(unit_deviance(y, mu_new, m, family)))
            step += 1
        beta, eta, mu = beta_new, eta_new, mu_new
        if abs(dev - dev_new) <= tol * (abs(dev_new) + 0.1):
            dev = dev_new
            converged = True
            break
        dev = dev_new
    var = _variance(mu, family)
    w = m * var
    info = (X.T * w) @ X
    try:
        cov = np.linalg.inv(info)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(info)
    return {"beta": beta, "eta": eta, "mu": mu, "deviance": dev, "cov": cov, "w": w,
            "converged": converged, "iterations": it}


# ---------------------------------------------------------------------------- the fit
class GLMFit(Fit):
    """Logistic or Poisson regression of one response for a given set of terms."""

    is_glm = True

    def __init__(self, space: Space, obs: Observations, terms: list, family: str = "binomial",
                 trials=None, response: str = "R1", response_name: str | None = None):
        if family not in FAMILY_TITLES:
            raise ValueError(f"Unknown distribution {family!r}.")
        self.space, self.obs = space, obs
        self.terms = sort_terms(terms)
        self.family = family
        self.response = response
        self.response_name = response_name or response
        self.response_units = ""
        self.curvature = False
        y = np.asarray(obs.y, float)
        n = len(y)
        m = np.ones(n) if trials is None else np.broadcast_to(np.asarray(trials, float),
                                                               (n,)).copy()
        if family == "binomial":
            if np.any(~np.isfinite(m)) or np.any(m <= 0):
                raise ValueError("The number of trials must be positive for every run.")
            if np.all((y >= 0) & (y <= 1)):   # pass/fail (0/1) or proportions
                yv = y
            else:                               # numbers of successes out of the trials
                if np.any(y < 0) or np.any(y > m + 1e-9):
                    raise ValueError("Binomial responses are proportions between 0 and 1, or "
                                     "numbers of successes between 0 and the trials.")
                yv = y / m
            self.transform = Transform("logit", lower=0.0, upper=1.0)
        else:
            if np.any(y < 0):
                raise ValueError("Poisson responses are counts (0 or more).")
            yv = y
            m = np.ones(n)
            self.transform = Transform("ln")
        self.m = m
        self.y = yv
        self.z = yv
        obs.z = yv
        self.mat = build_matrix(space, obs.coded, self.terms, obs.blocks)
        X = self.mat.X
        self.n, self.p = X.shape
        self.rank = int(np.linalg.matrix_rank(X)) if self.p else 0
        if self.rank < self.p:
            raise model.AliasedError("The model is aliased: some terms cannot be estimated "
                                     "from this design. Choose a smaller model.")
        if self.n <= self.p:
            raise ValueError(f"{self.n} observations are not enough for a model with "
                             f"{self.p} parameters.")
        res = irls(X, yv, m, family)
        self.converged = res["converged"]
        self.iterations = res["iterations"]
        self.beta = res["beta"]
        self.eta = res["eta"]
        self.fitted = res["mu"]
        self.mu = res["mu"]
        self.w = res["w"]
        self.cov = res["cov"]
        self.XtX_inv = self.cov          # variance of x'b = x' cov x (dispersion 1)
        self.ms_res = 1.0
        self.df_res = self.n - self.p
        self.deviance = res["deviance"]
        self.ss_res = self.deviance
        self.resid = yv - self.mu
        self.mean = float(np.average(yv, weights=m))
        self.loglik = loglik(yv, self.mu, m, family)
        self.null = self._null_fit()
        self.ss_cor = self.null["deviance"]
        sw = np.sqrt(self.w)
        Xw = X * sw[:, None]
        self.h = np.einsum("ij,ij->i", Xw @ self.cov, Xw)
        self.warnings: list[str] = []
        if not self.converged:
            self.warnings.append("The iterations did not converge: the data may be separated "
                                 "(a factor setting predicts the response perfectly); "
                                 "estimates and standard errors are unreliable.")
        if np.any(np.abs(self.beta) > 15) and family == "binomial":
            self.warnings.append("Very large coefficients: probabilities near 0 or 1 suggest "
                                 "(quasi-)complete separation.")
        self._anova = None

    # ------------------------------------------------------------------ helpers
    def _null_fit(self) -> dict:
        ones = np.ones((self.n, 1))
        mu0 = np.full(self.n, np.clip(np.average(self.y, weights=self.m), EPS,
                                      1 - EPS if self.family == "binomial" else np.inf))
        if self.family == "poisson":
            mu0 = np.maximum(mu0, EPS)
        return {"deviance": float(np.sum(unit_deviance(self.y, mu0, self.m, self.family))),
                "loglik": loglik(self.y, mu0, self.m, self.family), "X": ones}

    def _refit_deviance(self, keep: np.ndarray, add_intercept: bool = False) -> float:
        X = self.mat.X[:, keep]
        if add_intercept:
            X = np.hstack([np.ones((self.n, 1)), X])
        if X.shape[1] == 0:
            return self.null["deviance"]
        return irls(X, self.y, self.m, self.family)["deviance"]

    @property
    def family_title(self) -> str:
        return FAMILY_TITLES[self.family]

    # ------------------------------------------------------------------ tests
    def anova(self) -> dict:
        """Likelihood-ratio (deviance) tests: the model against intercept (+ blocks) only and
        each term against the model without it (Type III)."""
        if self._anova is not None:
            return self._anova
        p = self.p
        base = np.zeros(p, bool)
        if self.mat.intercept:
            base[0] = True
        if self.mat.block_slice is not None:
            base[self.mat.block_slice] = True
        dev0 = self._refit_deviance(base, add_intercept=not self.mat.intercept)
        df_model = int(p - base.sum() - (0 if self.mat.intercept else 1))
        chi_m = max(dev0 - self.deviance, 0.0)
        rows = []
        lin = [s for t, s in self.mat.groups if model.is_blend_linear(self.space, t)]
        if lin:
            keep = np.ones(p, bool)
            for s in lin:
                keep[s] = False
            chi = max(self._refit_deviance(keep, add_intercept=True) - self.deviance, 0.0)
            df = len(lin) - 1
            rows.append(("Linear Mixture", chi, df, float(stats.chi2.sf(chi, df)) if df else
                         float("nan"), ()))
        for t, s in self.mat.groups:
            if not t or model.is_blend_linear(self.space, t):
                continue
            keep = np.ones(p, bool)
            keep[s] = False
            chi = max(self._refit_deviance(keep) - self.deviance, 0.0)
            df = s.stop - s.start
            name = self.space.factors[t[0]].source if len(t) == 1 else term_name(t)
            rows.append((name, chi, df, float(stats.chi2.sf(chi, df)), t))
        self._anova = {"kind": "glm", "model": (chi_m, df_model,
                                                float(stats.chi2.sf(chi_m, df_model))
                                                if df_model > 0 else float("nan")),
                       "terms": rows, "deviance": (self.deviance, self.df_res),
                       "null": (dev0, self.n - (p - df_model)),
                       "pearson": (self.pearson(), self.df_res)}
        return self._anova

    def term_p_values(self) -> dict:
        return {t: (chi, p) for _n, chi, _df, p, t in self.anova()["terms"]}

    def model_test(self) -> tuple[float, float]:
        chi, df, p = self.anova()["model"]
        return chi, p

    def pearson(self) -> float:
        var = _variance(self.mu, self.family)
        return float(np.sum(self.m * (self.y - self.mu) ** 2 / var))

    def hosmer_lemeshow(self, groups: int = 10) -> dict | None:
        """Hosmer-Lemeshow test for binary (or binomial) responses: runs grouped by deciles
        of the fitted probability, chi-square on groups - 2 df."""
        if self.family != "binomial" or self.n < 6:
            return None
        g = min(groups, max(3, self.n // 2))
        order = np.argsort(self.mu, kind="stable")
        parts = np.array_split(order, g)
        chi = 0.0
        used = 0
        for idx in parts:
            if not len(idx):
                continue
            obs = float(np.sum(self.m[idx] * self.y[idx]))
            exp = float(np.sum(self.m[idx] * self.mu[idx]))
            tot = float(np.sum(self.m[idx]))
            pbar = exp / tot
            if 0 < pbar < 1:
                chi += (obs - exp) ** 2 / (tot * pbar * (1 - pbar))
                used += 1
        df = used - 2
        if df < 1:
            return None
        return {"chi2": chi, "df": df, "p": float(stats.chi2.sf(chi, df)), "groups": used}

    def goodness_of_fit(self) -> list[tuple[str, float, int, float]]:
        out = []
        df = self.df_res
        if df > 0:
            out.append(("Deviance", self.deviance, df, float(stats.chi2.sf(self.deviance, df))))
            pe = self.pearson()
            out.append(("Pearson Chi-Square", pe, df, float(stats.chi2.sf(pe, df))))
        hl = self.hosmer_lemeshow()
        if hl is not None:
            out.append(("Hosmer-Lemeshow", hl["chi2"], hl["df"], hl["p"]))
        return out

    # ------------------------------------------------------------------ statistics
    def press(self) -> float | None:
        return None

    def information(self) -> dict:
        n, k = self.n, self.p
        m2ll = -2 * self.loglik
        aicc = m2ll + 2 * k + (2 * k * (k + 1) / (n - k - 1) if n - k - 1 > 0 else math.inf)
        return {"m2ll": m2ll, "aic": m2ll + 2 * k, "aicc": aicc, "bic": m2ll + k * math.log(n),
                "k": k}

    def fit_statistics(self) -> dict:
        info = self.information()
        null = self.null
        pseudo = 1 - self.deviance / null["deviance"] if null["deviance"] > 0 else float("nan")
        mcf = 1 - self.loglik / null["loglik"] if null["loglik"] < 0 else float("nan")
        disp = self.pearson() / self.df_res if self.df_res > 0 else float("nan")
        return {"std_dev": float("nan"), "mean": self.mean, "cv": float("nan"),
                "r2": pseudo, "adj_r2": float("nan"), "pred_r2": float("nan"), "press": None,
                "adeq": float("nan"), "pseudo_r2": pseudo, "mcfadden_r2": mcf,
                "deviance": self.deviance, "null_deviance": null["deviance"],
                "pearson": self.pearson(), "dispersion": disp, "m2ll": info["m2ll"],
                "aicc": info["aicc"], "bic": info["bic"], "df_res": self.df_res}

    def coefficients(self, level: float = 0.95) -> list[dict]:
        se = np.sqrt(np.clip(np.diag(self.cov), 0, None))
        q = stats.norm.ppf(0.5 + level / 2)
        vif = self.vif()
        rows = []
        bs = self.mat.block_slice
        for j, lab in enumerate(self.mat.labels):
            b = float(self.beta[j])
            zval = b / se[j] if se[j] > 0 else float("nan")
            rows.append({"label": lab, "estimate": b, "se": float(se[j]),
                         "low": b - q * se[j], "high": b + q * se[j], "vif": vif.get(j),
                         "z": zval, "p": float(2 * stats.norm.sf(abs(zval)))
                         if math.isfinite(zval) else float("nan"),
                         "ratio": math.exp(b) if abs(b) < 700 else float("inf"),
                         "block": bs is not None and bs.start <= j < bs.stop,
                         "curvature": False})
        return rows

    # ------------------------------------------------------------------ diagnostics
    def diagnostics(self) -> dict:
        X = self.mat.X
        h = self.h
        hc = np.clip(h, 0, 1 - 1e-12)
        var = _variance(self.mu, self.family)
        rp = (self.y - self.mu) / np.sqrt(var / self.m)
        d = unit_deviance(self.y, self.mu, self.m, self.family)
        rd = np.sign(self.y - self.mu) * np.sqrt(np.clip(d, 0, None))
        with np.errstate(invalid="ignore", divide="ignore"):
            sp = rp / np.sqrt(1 - hc)
            sd = rd / np.sqrt(1 - hc)
            # likelihood residual (Williams 1987): externally studentized analogue
            lik = np.sign(self.y - self.mu) * np.sqrt(np.clip((1 - hc) * sd ** 2 + hc * sp ** 2,
                                                              0, None))
            cook = sp ** 2 * hc / (self.p * (1 - hc))
            dffits = lik * np.sqrt(hc / (1 - hc))
            delta = (self.cov @ X.T) * (self.m * (self.y - self.mu) / (1 - hc))  # one step
            dfbetas = (delta / np.sqrt(np.diag(self.cov))[:, None]).T
            covratio = 1 / (1 - hc)
        return {"leverage": h, "int_stud": sp, "ext_stud": lik, "cooks": cook,
                "dffits": dffits, "dfbetas": dfbetas, "covratio": covratio,
                "pearson": rp, "deviance_resid": rd, "run": self.obs.run, "std": self.obs.std,
                "actual": self.y, "predicted": self.mu, "residual": self.y - self.mu,
                "labels": list(self.mat.labels)}

    # ------------------------------------------------------------------ prediction
    def interval(self, actual: np.ndarray, level: float = 0.95) -> dict:
        """Mean (probability / expected count) with a Wald interval on the link scale; a
        GLM has no prediction interval for a single run, so ``pi`` repeats ``ci``."""
        eta, v = self.predict_coded(self.space.to_coded(actual))
        q = stats.norm.ppf(0.5 + level / 2)
        se = np.sqrt(np.clip(v, 0, None))
        tr = self.transform
        lo, hi = tr.inverse(eta - q * se), tr.inverse(eta + q * se)
        ci = (np.minimum(lo, hi), np.maximum(lo, hi))
        mean = tr.inverse(eta)
        return {"z": eta, "median": mean, "mean": mean, "se_mean": se, "se_pred": se,
                "ci": ci, "pi": ci, "sd": float("nan")}


# ---------------------------------------------------------------------------- fit summary
def glm_fit_summary(space: Space, obs: Observations, family: str, trials=None) -> FitSummary:
    """Sequential likelihood-ratio tests of the polynomial orders (Mean, Linear, 2FI, ...)
    with -2 log likelihood, AICc, BIC and the deviance pseudo-R² of each order."""
    rows_seq, rows_sum, extra, aliased = [], [], [], {}
    prev_dev, prev_rank, prev_name = None, None, "Mean"
    suggested, suggested_order = "Mean", "mean"
    try:
        f0 = GLMFit(space, obs, [], family, trials) if not space.scheffe else None
    except ValueError:
        f0 = None
    if f0 is not None:
        prev_dev, prev_rank = f0.deviance, f0.p
        info = f0.information()
        extra.append(["Mean", info["m2ll"], info["aicc"], info["bic"], 0.0, "", "mean"])
    terms_prev: list = []
    for name, order in model_orders(space):
        if order == "mean":
            continue
        terms = model_terms(space, order)
        if not terms or set(terms) == set(terms_prev):
            continue
        X = build_matrix(space, obs.coded, terms, obs.blocks).X
        rank = int(np.linalg.matrix_rank(X))
        is_aliased = rank < X.shape[1] or rank >= len(obs.y)
        aliased[order] = is_aliased
        if is_aliased:
            rows_seq.append([f"{name} vs {prev_name}", None, rank - (prev_rank or 0), None,
                             None, "Aliased", order])
            extra.append([name, None, None, None, None, "Aliased", order])
            rows_sum.append([name, float("nan"), float("nan"), float("nan"), float("nan"),
                             None, "Aliased", order])
            terms_prev = terms
            continue
        try:
            fit = GLMFit(space, obs, terms, family, trials)
        except ValueError:
            continue
        info = fit.information()
        pseudo = fit.fit_statistics()["pseudo_r2"]
        chi = p = None
        if prev_dev is not None and fit.p > prev_rank:
            chi = max(prev_dev - fit.deviance, 0.0)
            p = float(stats.chi2.sf(chi, fit.p - prev_rank))
        remark = "" if fit.converged else "Not converged"
        rows_seq.append([f"{name} vs {prev_name}", chi, fit.p - (prev_rank or 0), chi, p,
                         remark, order])
        extra.append([name, info["m2ll"], info["aicc"], info["bic"], pseudo, remark, order])
        rows_sum.append([name, float("nan"), pseudo, float("nan"), float("nan"), None, remark,
                         order])
        if not remark and p is not None and p < model.PREFERENCES.alpha:
            suggested, suggested_order = name, order
        prev_dev, prev_rank, prev_name, terms_prev = fit.deviance, fit.p, name, terms
    for rows in (rows_seq, rows_sum, extra):
        for r in rows:
            if r[-1] == suggested_order and r[-2] == "":
                r[-2] = "Suggested"
    return FitSummary([r[:6] for r in rows_seq], [], [r[:7] for r in rows_sum], suggested,
                      suggested_order, (0.0, 0), aliased, kind="glm",
                      extra=[r[:6] for r in extra])


# ---------------------------------------------------------------------------- top level
def trials_for(ds, spec, obs: Observations):
    """Trials per run of a binomial response: None (binary), a number, or a column."""
    t = spec.trials
    if t is None or t == "":
        return None
    if isinstance(t, str):
        from rsstatchem.core.stats.common import numeric

        if not ds.has_name(t):
            try:
                return float(t)
            except ValueError:
                raise ValueError(f"The trials column {t!r} does not exist.") from None
        return numeric(ds, ds.var(t).name).loc[obs.rows].to_numpy(float)
    return float(t)


def analyze_glm(ds, spec, space: Space, obs: Observations) -> Analysis:
    family = spec.family
    trials = trials_for(ds, spec, obs) if family == "binomial" else None
    summary = glm_fit_summary(space, obs, family, trials)
    if spec.model == "manual" and spec.terms is not None:
        terms = [key_term(k) for k in spec.terms]
    elif spec.model == "suggested":
        terms = model_terms(space, summary.suggested_order)
    else:
        terms = model_terms(space, spec.model)
    terms, dropped = drop_aliased(space, obs, terms)

    def make(ts):
        return GLMFit(space, obs, ts, family, trials, spec.response)

    steps: list = []
    if spec.reduction in ("backward", "forward", "stepwise"):
        fit, steps = model.auto_select(space, obs, terms, Transform(), spec.reduction,
                                       spec.criterion, spec.alpha_in, spec.alpha_out,
                                       spec.hierarchy, spec.response, fit_factory=make)
    else:
        fit = make(terms)
    fit.response_name = model.response_name(ds, spec.response)
    fit.response_units = model.response_units(ds, spec.response)
    return Analysis(fit, summary, spec, steps, dropped, space, obs)


def is_glm(fit) -> bool:
    return bool(getattr(fit, "is_glm", False))
