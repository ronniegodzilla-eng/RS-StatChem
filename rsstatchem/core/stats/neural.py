"""Analyze > Neural Networks: Multilayer Perceptron (MLP) and Radial Basis Function (RBF),
following IBM's "MLP Algorithms" and "RBF Algorithms" chapters. numpy only.

**MLP.** Inputs: factors one-of-c coded, covariates rescaled (standardised, normalised,
adjusted normalised or none) with training-sample statistics. One or two hidden layers with
hyperbolic tangent or sigmoid activation; output activation identity, softmax (categorical
targets with cross-entropy error), tanh or sigmoid; error sum-of-squares E = ½ΣΣ(y − ŷ)² or
cross-entropy E = −ΣΣ y log ŷ. Gradients by error back-propagation. Training: batch with
scaled conjugate gradient (Møller 1993; λ₁ = 5·10⁻⁷, σ₀ = 5·10⁻⁵) or gradient descent
(learning rate 0.4, lower bound 0.001, momentum 0.9), online or mini-batch gradient descent.
Initial weights: the best of four uniform draws in [−0.5, 0.5] (IBM's "alternated simulated
annealing" start, simplified), then perturbation restarts while the training error ratio
exceeds 0.1. Stopping rules in IBM's order: no decrease of the testing (or training) error
for n steps (weights of the minimum are reported), maximum steps, relative change in training
error < 10⁻⁴, training error ratio < 10⁻³. Automatic architecture: one tanh hidden layer
whose size is chosen on an internal 70/30 split of the training sample among 1…50 units
(limited so the weights do not exceed the training cases). Scale targets are rescaled like
the covariates (standardised for an identity output, adjusted normalised for tanh, normalised
for sigmoid) and reported in original units.

**RBF.** One hidden layer of Gaussian basis functions whose centres and widths come from
TwoStep clustering of the rescaled inputs (centre = cluster means / category proportions,
spherical width σ_j = √(h · mean_p s²_jp) with overlap factor h = 1 + 0.1P), normalised
(softmax) or ordinary activation; output weights by least squares (identity output, SSE).
The number of units is chosen in [K₁, K₂] (K₁ = min(K, R), K₂ = max(10, 2K, R), K from TwoStep
auto-clustering) by the testing error, or by BIC = N ln(MSE) + k ln N without testing data.

**Output statistics** (both): error, percent incorrect predictions or relative error
Σ(y − ŷ)²/Σ(y − ȳ)² per sample, classification table, ROC / AUC, cumulative gains and lift,
predicted-by-observed, residual-by-predicted, and sensitivity-based independent-variable
importance: each predictor is moved over {min, quartiles, max} (or all categories) for every
training/testing pattern; the average maximum output change is normalised to sum to 1.

Random initial weights and the partition draw use MT19937 with the given seed, so results
are reproducible but not identical to SPSS's.
"""
from __future__ import annotations

import io
import math
import time
from dataclasses import dataclass, field

import numpy as np

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import fmt, fmt_int
from rsstatchem.core.stats.ml_common import (RESCALE, Encoder, MLError, MLResult, auc,
                                             categorical, classification_rows, complete_rows,
                                             full_series, gains_lift_charts, importance_chart,
                                             importance_table, measure_of, partition,
                                             predicted_by_observed, processing_table,
                                             raw_column, rng, roc_chart,
                                             scatter_observed_predicted)
from rsstatchem.core.stats.roc import figure


# ============================================================================ network
def _act(z, kind):
    if kind == "tanh":
        return np.tanh(z)
    if kind == "sigmoid":
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    return z


def _dact(a, kind):
    if kind == "tanh":
        return 1 - a ** 2
    if kind == "sigmoid":
        return a * (1 - a)
    return np.ones_like(a)


@dataclass
class Net:
    sizes: list[int]                     # [inputs, hidden..., outputs]
    hidden_act: str = "tanh"
    out_act: str = "identity"
    error: str = "sse"
    groups: list = field(default_factory=list)   # softmax column groups of the output

    def shapes(self):
        return [(a + 1, b) for a, b in zip(self.sizes[:-1], self.sizes[1:])]

    def n_weights(self) -> int:
        return int(sum(a * b for a, b in self.shapes()))

    def unpack(self, w):
        out, k = [], 0
        for a, b in self.shapes():
            out.append(w[k:k + a * b].reshape(a, b))
            k += a * b
        return out

    def forward(self, w, X):
        Ws = self.unpack(w)
        acts = [X]
        a = X
        for li, W in enumerate(Ws):
            z = a @ W[1:] + W[0]
            if li < len(Ws) - 1:
                a = _act(z, self.hidden_act)
            else:
                a = self._output(z)
            acts.append(a)
        return acts

    def _output(self, z):
        if self.out_act == "softmax":
            out = z.copy()
            for g in self.groups:
                zz = z[:, g] - z[:, g].max(1, keepdims=True)
                e = np.exp(zz)
                out[:, g] = e / e.sum(1, keepdims=True)
            return out
        return _act(z, self.out_act)

    def predict(self, w, X):
        return self.forward(w, X)[-1]

    def error_value(self, Y, T) -> float:
        if self.error == "ce":
            return float(-(T * np.log(np.clip(Y, 1e-300, None))).sum())
        return float(0.5 * ((Y - T) ** 2).sum())

    def loss_grad(self, w, X, T):
        acts = self.forward(w, X)
        Ws = self.unpack(w)
        Y = acts[-1]
        E = self.error_value(Y, T)
        if self.error == "ce" or self.out_act == "identity":
            delta = Y - T
        else:
            delta = (Y - T) * _dact(Y, self.out_act)
        grads = []
        for li in range(len(Ws) - 1, -1, -1):
            a = acts[li]
            gW = np.vstack([delta.sum(0), a.T @ delta])
            grads.append(gW.ravel())
            if li > 0:
                delta = (delta @ Ws[li][1:].T) * _dact(a, self.hidden_act)
        return E, np.concatenate(grads[::-1])


# ============================================================================ training
@dataclass
class TrainSettings:
    algorithm: str = "scg"          # scg | gd
    mode: str = "batch"             # batch | online | minibatch
    batch_size: int | None = None
    lr: float = 0.4
    lr_low: float = 0.001
    momentum: float = 0.9
    lr_epochs: int = 10
    lam: float = 5e-7
    sigma: float = 5e-5
    interval_center: float = 0.0
    interval_offset: float = 0.5
    max_steps: int | None = None
    max_time: float = 15 * 60.0
    steps_no_decrease: int = 1
    min_rel_change: float = 1e-4
    min_ratio: float = 1e-3
    check_test: bool = True


def _scg(net: Net, w, X, T, s: TrainSettings, monitor):
    N = len(w)
    E, g = net.loss_grad(w, X, T)
    r = -g
    p = r.copy()
    lam, lam_bar = s.lam, 0.0
    success = True
    k = 0
    delta = 0.0
    while True:
        pp = float(p @ p)
        if pp < 1e-300:
            return w, "Error gradient is zero"
        if success:
            sig = s.sigma / math.sqrt(pp)
            _, g2 = net.loss_grad(w + sig * p, X, T)
            sv = (g2 - g) / sig
            delta = float(p @ sv)
        delta += (lam - lam_bar) * pp
        if delta <= 0:
            lam_bar = 2 * (lam - delta / pp)
            delta = -delta + lam * pp
            lam = lam_bar
        mu = float(p @ r)
        alpha = mu / delta
        w_new = w + alpha * p
        E_new, g_new = net.loss_grad(w_new, X, T)
        Delta = 2 * delta * (E - E_new) / (mu * mu) if mu else -1.0
        if Delta >= 0:
            w, E_old, E, g = w_new, E, E_new, g_new
            r_new = -g
            lam_bar = 0.0
            success = True
            k += 1
            if k % N == 0:
                p = r_new.copy()
            else:
                beta = float(r_new @ r_new - r_new @ r) / mu
                p = r_new + beta * p
            r = r_new
            if Delta >= 0.75:
                lam = lam / 4
            stop = monitor(w, E, E_old)
            if stop:
                return w, stop
        else:
            lam_bar = lam
            success = False
        if Delta < 0.25:
            lam = lam + delta * (1 - Delta) / pp
        lam = min(lam, 1e100)


def _gd_batch(net: Net, w, X, T, s: TrainSettings, monitor):
    E, g = net.loss_grad(w, X, T)
    lr = s.lr
    dw = np.zeros_like(w)
    while True:
        step = -lr * g + s.momentum * dw
        if float(step @ -g) <= 0:      # keep the steepest descent direction dominant
            step = -lr * g
        w_new = w + step
        E_new, g_new = net.loss_grad(w_new, X, T)
        if E_new < E:
            dw = step
            E_old, w, E, g = E, w_new, E_new, g_new
            lr = min(lr * 1.1, 10.0)
            stop = monitor(w, E, E_old)
            if stop:
                return w, stop
        else:
            lr = lr / 2
            dw = np.zeros_like(w)
            if lr < s.lr_low:
                lr = s.lr_low
                E_old, w, E, g = E, w_new, E_new, g_new
                stop = monitor(w, E, E_old)
                if stop:
                    return w, stop


def _gd_online(net: Net, w, X, T, s: TrainSettings, monitor, gen):
    n = len(X)
    bs = 1 if s.mode == "online" else int(s.batch_size or max(2, min(1000, n // 10)))
    dw = np.zeros_like(w)
    epoch = 0
    E_old = net.error_value(net.predict(w, X), T)
    while True:
        lr = max(s.lr_low, s.lr * (s.lr_low / s.lr) ** (epoch / max(s.lr_epochs, 1)))
        order = gen.permutation(n)
        for b in range(0, n, bs):
            sel = order[b:b + bs]
            _, g = net.loss_grad(w, X[sel], T[sel])
            g = g / len(sel)
            step = -lr * g + s.momentum * dw
            if float(step @ -g) <= 0:
                step = -lr * g
            w = w + step
            dw = step
        epoch += 1
        E = net.error_value(net.predict(w, X), T)
        stop = monitor(w, E, E_old)
        E_old = E
        if stop:
            return w, stop


def train(net: Net, X, T, Xte, Tte, s: TrainSettings, gen, w0=None):
    """Train with IBM's stopping rules; returns (weights, stopping rule text, steps)."""
    E_base = net.error_value(_baseline(net, T), T) or 1.0
    t0 = time.time()
    best = {"w": None, "E": math.inf, "k": 0, "steps": 0}
    max_steps = s.max_steps or (1000 if s.mode == "batch" else 200)
    has_test = Xte is not None and len(Xte) and s.check_test

    def monitor(w, E, E_old):
        best["steps"] += 1
        k = best["steps"]
        crit = net.error_value(net.predict(w, Xte), Tte) if has_test else E
        if not math.isfinite(best["E"]) or crit < best["E"] - 1e-12 * max(1.0, abs(best["E"])):
            best.update(w=w.copy(), E=crit, k=k)
        elif k - best["k"] >= s.steps_no_decrease and (has_test or s.mode != "batch"):
            return ("1 consecutive step(s) with no decrease in error"
                    if s.steps_no_decrease == 1 else
                    f"{s.steps_no_decrease} consecutive steps with no decrease in error") + \
                ("ᵃ" if has_test else "")
        if time.time() - t0 > s.max_time:
            return "Maximum training time exceeded"
        if k >= max_steps:
            return "Maximum number of epochs exceeded"
        if E_old and abs(E - E_old) / max(abs(E_old), 1e-300) < s.min_rel_change:
            return "Relative change in training error criterion achieved"
        if E / E_base < s.min_ratio:
            return "Training error ratio criterion achieved"
        return None

    w = w0 if w0 is not None else _init_weights(net, X, T, s, gen)
    if s.mode == "batch" and s.algorithm == "scg":
        w, rule = _scg(net, w, X, T, s, monitor)
    elif s.mode == "batch":
        w, rule = _gd_batch(net, w, X, T, s, monitor)
    else:
        w, rule = _gd_online(net, w, X, T, s, monitor, gen)
    if best["w"] is not None and (has_test or s.mode != "batch"):
        w = best["w"]
    return w, rule, best["steps"]


def _baseline(net: Net, T):
    return np.tile(T.mean(0), (len(T), 1))


def _init_weights(net: Net, X, T, s: TrainSettings, gen):
    a0, a = s.interval_center, s.interval_offset
    cands = [gen.uniform(a0 - a, a0 + a, net.n_weights()) for _ in range(4)]
    errs = [net.error_value(net.predict(w, X), T) for w in cands]
    return cands[int(np.argmin(errs))]


# ============================================================================ data
@dataclass
class TargetSpec:
    name: str
    kind: str                    # categorical | scale
    levels: list = field(default_factory=list)
    labels: list = field(default_factory=list)
    centre: float = 0.0
    scale: float = 1.0
    cols: list = field(default_factory=list)


def _targets(ds, deps, rows, train, out_act, rescale_dep):
    specs, cols, k = [], [], 0
    for d in deps:
        if measure_of(ds, d) != "scale":
            cat = categorical(ds, d, rows)
            lv = [cat.levels[c] for c in sorted(set(cat.codes[train].tolist())) if c >= 0]
            cat = categorical(ds, d, rows, lv)
            T = np.zeros((len(rows), len(lv)))
            ok = cat.codes >= 0
            T[np.where(ok)[0], cat.codes[ok]] = 1
            specs.append(TargetSpec(d, "categorical", lv, cat.labels, cols=list(
                range(k, k + len(lv)))))
            cols.append(T)
            k += len(lv)
        else:
            y = raw_column(ds, d).to_numpy(float)[rows]
            yt = y[train]
            how = rescale_dep
            if how == "auto":
                how = {"tanh": "adjusted", "sigmoid": "normalized"}.get(out_act, "standardized")
            if how == "standardized":
                c, s = yt.mean(), yt.std(ddof=1)
            elif how == "normalized":
                c, s = yt.min(), yt.max() - yt.min()
            elif how == "adjusted":
                c, s = (yt.max() + yt.min()) / 2, (yt.max() - yt.min()) / 2
            else:
                c, s = 0.0, 1.0
            s = s if s and np.isfinite(s) else 1.0
            specs.append(TargetSpec(d, "scale", centre=float(c), scale=float(s), cols=[k]))
            cols.append(((y - c) / s)[:, None])
            k += 1
    return specs, np.column_stack(cols)


def _prepare(ds, deps, factors, covariates, rescale, train_pct, test_pct, hold_pct,
             partition_var, seed):
    if not deps:
        raise MLError("Select a dependent variable.")
    if not factors and not covariates:
        raise MLError("Select at least one factor or covariate.")
    names = list(deps) + list(factors) + list(covariates) + ([partition_var]
                                                             if partition_var else [])
    rows = complete_rows(ds, names)
    if len(rows) < 5:
        raise MLError("Too few valid cases.")
    pv = raw_column(ds, partition_var).to_numpy(float)[rows] if partition_var else None
    sample = partition(len(rows), train_pct, test_pct, hold_pct, seed, pv)
    train = np.where(sample == 0)[0]
    if len(train) < 3:
        raise MLError("The training sample has too few cases.")
    enc = Encoder(list(factors), list(covariates), rescale).fit(ds, rows, train)
    X = enc.transform(ds, rows)
    return rows, sample, train, enc, X


def _outputs_to_predictions(specs, Y):
    preds = []
    for sp in specs:
        if sp.kind == "categorical":
            P = Y[:, sp.cols]
            preds.append(P.argmax(1))
        else:
            preds.append(Y[:, sp.cols[0]] * sp.scale + sp.centre)
    return preds


def _sample_errors(net_err, specs, Y, T, sample):
    rows = {}
    for k in range(3):
        sel = sample == k
        if not sel.any():
            continue
        rows[k] = {"error": net_err(Y[sel], T[sel])}
        per = []
        for sp in specs:
            if sp.kind == "categorical":
                pred = Y[sel][:, sp.cols].argmax(1)
                truth = T[sel][:, sp.cols].argmax(1)
                per.append(float(np.mean(pred != truth)))
            else:
                y, yh = T[sel][:, sp.cols[0]], Y[sel][:, sp.cols[0]]
                ss = float(((y - y.mean()) ** 2).sum())
                per.append(float(((y - yh) ** 2).sum()) / ss if ss > 0 else float("nan"))
        rows[k]["per"] = per
    return rows


# ============================================================================ importance
def sensitivity(predict, X: np.ndarray, groups: list[list[int]], kinds: list[str]) -> np.ndarray:
    """IBM sensitivity analysis: average maximum output change per predictor (normalised)."""
    out = np.zeros(len(groups))
    for p, (g, kind) in enumerate(zip(groups, kinds)):
        if kind == "factor":
            vals = []
            for c in range(len(g)):
                v = np.zeros(len(g))
                v[c] = 1.0
                vals.append(v)
        else:
            x = X[:, g[0]]
            vals = [np.array([q]) for q in np.quantile(x, [0, 0.25, 0.5, 0.75, 1.0])]
        preds = []
        for v in vals:
            Xm = X.copy()
            Xm[:, g] = v
            preds.append(predict(Xm))
        P = np.stack(preds)                 # (values, n, outputs)
        rng_ = P.max(0) - P.min(0)          # (n, outputs)
        out[p] = float(np.sqrt((rng_ ** 2).sum(1)).mean())
    return out / out.sum() if out.sum() > 0 else out


# ============================================================================ MLP
@dataclass
class MLPModel:
    net: Net
    w: np.ndarray
    specs: list
    encoder: Encoder
    rule: str = ""
    steps: int = 0
    seconds: float = 0.0


def mlp_fit(X, T, specs, hidden=(None,), hidden_act="tanh", out_act=None, settings=None,
            Xte=None, Tte=None, seed=None, h_max: int = 50, h_min: int = 1) -> tuple:
    s = settings or TrainSettings()
    gen = rng(seed)
    all_cat = all(sp.kind == "categorical" for sp in specs)
    if out_act is None:
        out_act = "softmax" if all_cat else "identity"
    error = "ce" if (out_act == "softmax" and all_cat) else "sse"
    if out_act == "softmax" and not all_cat:
        raise MLError("Softmax requires all dependent variables to be categorical.")
    groups = [sp.cols for sp in specs if sp.kind == "categorical"]
    P, R = X.shape[1], T.shape[1]
    hidden = list(hidden)
    if hidden == [None]:
        h = _choose_hidden(X, T, groups, hidden_act, out_act, error, s, gen, h_min, h_max)
        hidden = [h]
    net = Net([P] + [int(h) for h in hidden] + [R], hidden_act, out_act, error, groups)
    w, rule, steps = train(net, X, T, Xte, Tte, s, gen)
    E_base = net.error_value(_baseline(net, T), T) or 1.0
    ratio = net.error_value(net.predict(w, X), T) / E_base
    tries = 0
    while ratio > 0.1 and tries < 2 and not len(Xte if Xte is not None else []):
        tries += 1
        w2 = w + gen.uniform(-0.5 ** tries, 0.5 ** tries, len(w))
        w2, rule2, steps2 = train(net, X, T, Xte, Tte, s, gen, w2)
        r2 = net.error_value(net.predict(w2, X), T) / E_base
        if r2 < ratio:
            w, rule, steps, ratio = w2, rule2, steps2, r2
    return net, w, rule, steps


def _choose_hidden(X, T, groups, hidden_act, out_act, error, s, gen, h_min, h_max):
    n, P = X.shape
    R = T.shape[1]
    limit = max(1, int((n - R) / (P + R + 1)))
    hmax = max(h_min, min(h_max, limit))
    if n < 20 or hmax == h_min:
        return h_min
    perm = gen.permutation(n)
    cut = int(0.7 * n)
    tr, te = perm[:cut], perm[cut:]
    cands = sorted({h for h in (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 30, 40, 50)
                    if h_min <= h <= hmax} | {h_min, hmax})
    best = None
    s2 = TrainSettings(**{**s.__dict__, "max_steps": min(s.max_steps or 300, 300),
                          "steps_no_decrease": max(s.steps_no_decrease, 10)})
    worse = 0
    for h in cands:
        net = Net([P, h, R], hidden_act, out_act, error, groups)
        w, _, _ = train(net, X[tr], T[tr], X[te], T[te], s2, gen)
        e = net.error_value(net.predict(w, X[te]), T[te])
        if best is None or e < best[1] - 1e-9:
            best = (h, e)
            worse = 0
        else:
            worse += 1
            if worse >= 3:
                break
    return best[0]


def mlp_procedure(ds: Dataset, deps: list[str], factors: list[str], covariates: list[str],
                  rescale: str = "standardized", rescale_dep: str = "auto",
                  train_pct: float = 7, test_pct: float = 3, hold_pct: float = 0,
                  partition_var: str | None = None, auto_arch: bool = True,
                  h_min: int = 1, h_max: int = 50, layers: int = 1, units=(None, None),
                  hidden_act: str = "tanh", out_act: str | None = None,
                  settings: TrainSettings | None = None, seed: int | None = None,
                  output: dict | None = None, save: tuple = ()) -> MLResult:
    out = {"description": True, "diagram": True, "weights": False, "summary": True,
           "classification": True, "roc": True, "gains": True, "lift": True,
           "pred_obs": True, "resid": True, "importance": True, "processing": True,
           **(output or {})}
    s = settings or TrainSettings()
    rows, sample, train_idx, enc, X = _prepare(ds, deps, factors, covariates, rescale,
                                               train_pct, test_pct, hold_pct, partition_var,
                                               seed)
    if out_act is None:
        out_act = "softmax" if all(measure_of(ds, d) != "scale" for d in deps) else "identity"
    specs, T = _targets(ds, deps, rows, train_idx, out_act, rescale_dep)
    test_idx = np.where(sample == 1)[0]
    t0 = time.time()
    if auto_arch:
        hidden = (None,)
        hidden_act, out_act2 = "tanh", out_act
    else:
        hidden = tuple(units[:layers])
        out_act2 = out_act
    net, w, rule, steps = mlp_fit(X[train_idx], T[train_idx], specs, hidden, hidden_act,
                                  out_act2, s, X[test_idx] if len(test_idx) else None,
                                  T[test_idx] if len(test_idx) else None, seed, h_max, h_min)
    secs = time.time() - t0
    model = MLPModel(net, w, specs, enc, rule, steps, secs)
    Y = net.predict(w, X)
    section = Section("Multilayer Perceptron")
    section.log(_mlp_syntax(deps, factors, covariates, rescale, net, s, train_pct, test_pct,
                            hold_pct))
    _report(section, ds, "MLP", model, X, Y, T, sample, rows, specs, enc, factors, covariates,
            rescale, out, net.error, secs, rule)
    if out.get("diagram"):
        section.image("Network Diagram", network_diagram(net, enc, specs, w), width=600)
    if out.get("weights"):
        section.table(_parameter_table(net, w, enc, specs))
    return _finish(ds, section, "MLP", model, Y, specs, rows, sample, save, out,
                   lambda Xm: net.predict(w, Xm), X, enc, factors, covariates)


# ============================================================================ RBF
@dataclass
class RBFModel:
    centres: np.ndarray
    widths: np.ndarray
    W: np.ndarray                     # (J+1, R) output weights
    normalized: bool
    specs: list
    encoder: Encoder
    seconds: float = 0.0

    def hidden(self, X):
        d2 = ((X[:, None, :] - self.centres[None, :, :]) ** 2).sum(2)
        z = -d2 / (2 * self.widths[None, :] ** 2)
        if self.normalized:
            z = z - z.max(1, keepdims=True)
            e = np.exp(z)
            return e / e.sum(1, keepdims=True)
        return np.exp(z)

    def predict(self, X):
        H = self.hidden(X)
        return H @ self.W[1:] + self.W[0]


def rbf_fit_units(X, T, nf_cols: list[list[int]], n_cov: int, J: int, normalized: bool,
                  overlap: float | None, seed=None) -> RBFModel:
    """Centres/widths from TwoStep with J clusters, output weights by least squares."""
    from rsstatchem.core.stats import twostep as ts
    P = X.shape[1]
    Xcat = np.column_stack([X[:, g].argmax(1) for g in nf_cols]) if nf_cols else \
        np.zeros((len(X), 0), int)
    cov_cols = [c for c in range(P) if not any(c in g for g in nf_cols)]
    Xc = X[:, cov_cols] if cov_cols else np.zeros((len(X), 0))
    res = ts.twostep(Xc, Xcat, [len(g) for g in nf_cols], "ll", k=J)
    lab = res["labels"]
    ks = [k for k in range(1, res["k"] + 1) if (lab == k).any()]
    h = overlap if overlap else 1 + 0.1 * P
    C, S2 = [], []
    for k in ks:
        sel = lab == k
        c = X[sel].mean(0)
        v = X[sel].var(0) if sel.sum() > 1 else np.zeros(P)
        for g in nf_cols:
            v[g] = c[g] * (1 - c[g])
        C.append(c)
        S2.append(v)
    C, S2 = np.array(C), np.array(S2)
    sig = np.sqrt(h * S2.mean(1))
    pos = sig[sig > 0]
    fill = pos.mean() if len(pos) else 1.0
    sig = np.where(sig > 0, sig, fill)
    model = RBFModel(C, sig, np.zeros((len(C) + 1, T.shape[1])), normalized, [], None)
    H = model.hidden(X)
    A = np.column_stack([np.ones(len(X)), H])
    model.W = np.linalg.lstsq(A, T, rcond=None)[0]
    return model


def rbf_procedure(ds: Dataset, deps: list[str], factors: list[str], covariates: list[str],
                  rescale: str = "standardized", rescale_dep: str = "standardized",
                  train_pct: float = 7, test_pct: float = 3, hold_pct: float = 0,
                  partition_var: str | None = None, units: str = "auto",
                  u_min: int = 2, u_max: int = 10, u_fixed: int = 5,
                  activation: str = "normalized", overlap: float | None = None,
                  seed: int | None = None, output: dict | None = None,
                  save: tuple = ()) -> MLResult:
    out = {"description": True, "diagram": True, "weights": False, "summary": True,
           "classification": True, "roc": True, "gains": True, "lift": True,
           "pred_obs": True, "resid": True, "importance": True, "processing": True,
           **(output or {})}
    rows, sample, tr, enc, X = _prepare(ds, deps, factors, covariates, rescale, train_pct,
                                        test_pct, hold_pct, partition_var, seed)
    specs, T = _targets(ds, deps, rows, tr, "identity", rescale_dep)
    te = np.where(sample == 1)[0]
    nf_cols = enc.groups[:len(factors)]
    t0 = time.time()
    P, R = X.shape[1], T.shape[1]
    if units == "auto":
        from rsstatchem.core.stats import twostep as ts
        Xcat = np.column_stack([X[tr][:, g].argmax(1) for g in nf_cols]) if nf_cols else \
            np.zeros((len(tr), 0), int)
        cov_cols = [c for c in range(P) if not any(c in g for g in nf_cols)]
        K = ts.twostep(X[tr][:, cov_cols] if cov_cols else np.zeros((len(tr), 0)), Xcat,
                       [len(g) for g in nf_cols], "ll")["k"]
        lo = min(K, R) if activation == "ordinary" else max(2, min(K, R))
        hi = max(10, 2 * K, R)
        cand = list(range(lo, hi + 1))
    elif units == "range":
        cand = list(range(int(u_min), int(u_max) + 1))
    else:
        cand = [int(u_fixed)]
    best = None
    log = []
    for J in cand:
        if J > len(tr):
            break
        m = rbf_fit_units(X[tr], T[tr], nf_cols, len(covariates), J,
                          activation == "normalized", overlap, seed)
        if len(te):
            crit = float(((m.predict(X[te]) - T[te]) ** 2).sum()) / 2
        else:
            mse = float(((m.predict(X[tr]) - T[tr]) ** 2).mean())
            Jn = len(m.centres)
            npar = (P + 1 + R) * Jn + (R if activation == "ordinary" else 0)
            crit = len(tr) * math.log(max(mse, 1e-300)) + npar * math.log(len(tr))
        log.append((len(m.centres), crit))
        if best is None or crit < best[1] - 1e-12:
            best = (m, crit)
    model = best[0]
    model.specs, model.encoder = specs, enc
    model.seconds = time.time() - t0
    Y = model.predict(X)
    section = Section("Radial Basis Function")
    section.log(f"RBF {' '.join(deps)} BY {' '.join(factors)} WITH {' '.join(covariates)}\n"
                f"  /PARTITION TRAINING={train_pct:g} TESTING={test_pct:g} "
                f"HOLDOUT={hold_pct:g}\n  /ARCHITECTURE MINUNITS={min(cand)} "
                f"MAXUNITS={max(cand)} HIDDENFUNCTION="
                f"{'NRBF' if activation == 'normalized' else 'ORBF'}.")
    model_net_err = lambda Yp, Tt: float(0.5 * ((Yp - Tt) ** 2).sum())  # noqa: E731
    _report(section, ds, "RBF", model, X, Y, T, sample, rows, specs, enc, factors, covariates,
            rescale, out, "sse", model.seconds, "", rbf=True, err_fn=model_net_err)
    if out.get("diagram"):
        net = Net([P, len(model.centres), R], "softmax" if model.normalized else "exp",
                  "identity")
        section.image("Network Diagram", network_diagram(net, enc, specs, None, rbf=True),
                      width=600)
    if out.get("weights"):
        rows_w = []
        for i, c in enumerate(enc.columns):
            rows_w.append(["Input Layer", c] + [fmt(v, 3) for v in model.centres[:, i]] +
                          [""] * R)
        rows_w.append(["", "Hidden Unit Width"] + [fmt(v, 3) for v in model.widths] + [""] * R)
        rows_w.append(["Hidden Layer", "(Bias)"] + [""] * len(model.centres) +
                      [fmt(v, 3) for v in model.W[0]])
        for j in range(len(model.centres)):
            rows_w.append(["", f"H({j + 1})"] + [""] * len(model.centres) +
                          [fmt(v, 3) for v in model.W[j + 1]])
        section.table(Table("Parameter Estimates", ["Predictor", ""] +
                            [("Hidden Layer", f"H({j + 1})") for j in range(len(model.centres))]
                            + [("Output Layer", c) for c in _output_names(specs)], rows_w,
                            row_header_cols=2))
    if len(log) > 1:
        section.table(Table("Number of Hidden Units", ["Units",
                                                       "Testing Error" if len(te) else "BIC"],
                            [[str(a), fmt(b, 3)] for a, b in log]))
    return _finish(ds, section, "RBF", model, Y, specs, rows, sample, save, out,
                   model.predict, X, enc, factors, covariates)


# ============================================================================ reporting
def _output_names(specs) -> list[str]:
    out = []
    for sp in specs:
        if sp.kind == "categorical":
            out += [f"[{sp.name}={lab}]" for lab in sp.labels]
        else:
            out.append(sp.name)
    return out


def _mlp_syntax(deps, factors, covs, rescale, net, s, tr, te, ho) -> str:
    hid = ",".join(str(h) for h in net.sizes[1:-1])
    return (f"MULTILAYER PERCEPTRON {' '.join(deps)} BY {' '.join(factors)} WITH "
            f"{' '.join(covs)}\n  /RESCALE COVARIATE={rescale.upper()}\n"
            f"  /PARTITION TRAINING={tr:g} TESTING={te:g} HOLDOUT={ho:g}\n"
            f"  /ARCHITECTURE HIDDENLAYERS={len(net.sizes) - 2} NUMUNITS={hid} "
            f"HIDDENFUNCTION={net.hidden_act.upper()} OUTPUTFUNCTION={net.out_act.upper()}\n"
            f"  /CRITERIA TRAINING={s.mode.upper()} OPTIMIZATION="
            f"{'SCALEDCONJUGATE' if s.algorithm == 'scg' else 'GRADIENTDESCENT'}.")


def _report(section, ds, prefix, model, X, Y, T, sample, rows, specs, enc, factors, covs,
            rescale, out, error, secs, rule, rbf=False, err_fn=None):
    if out.get("processing"):
        processing_table(section, sample, ds.n_cases - len(rows))
    if out.get("description"):
        r = []
        for i, f in enumerate(factors):
            r.append(["Input Layer" if i == 0 else "", "Factors" if i == 0 else "", str(i + 1), f])
        for i, c in enumerate(covs):
            r.append(["" if (factors or i) else "Input Layer", "Covariates" if i == 0 else "",
                      str(i + 1), c])
        r.append(["", "Number of Unitsᵃ", "", fmt_int(len(enc.columns))])
        if covs:
            r.append(["", "Rescaling Method for Covariates", "", RESCALE.get(rescale, rescale)])
        if rbf:
            r.append(["Hidden Layer", "Number of Units", "", fmt_int(len(model.centres))])
            r.append(["", "Activation Function", "", "Softmax" if model.normalized
                      else "Exponential"])
        else:
            net = model.net
            r.append(["Hidden Layer(s)", "Number of Hidden Layers", "",
                      fmt_int(len(net.sizes) - 2)])
            for li, h in enumerate(net.sizes[1:-1], 1):
                r.append(["", f"Number of Units in Hidden Layer {li}ᵃ", "", fmt_int(h)])
            r.append(["", "Activation Function", "", {"tanh": "Hyperbolic tangent",
                                                       "sigmoid": "Sigmoid"}[net.hidden_act]])
        for i, sp in enumerate(specs):
            r.append(["Output Layer" if i == 0 else "", "Dependent Variables" if i == 0 else "",
                      str(i + 1), sp.name])
        r.append(["", "Number of Units", "", fmt_int(T.shape[1])])
        if any(sp.kind == "scale" for sp in specs):
            r.append(["", "Rescaling Method for Scale Dependents", "", "Standardized"])
        if rbf:
            r += [["", "Activation Function", "", "Identity"],
                  ["", "Error Function", "", "Sum of Squares"]]
        else:
            r += [["", "Activation Function", "", {"identity": "Identity", "softmax": "Softmax",
                                                   "tanh": "Hyperbolic tangent",
                                                   "sigmoid": "Sigmoid"}[model.net.out_act]],
                  ["", "Error Function", "", "Cross-entropy" if error == "ce"
                   else "Sum of Squares"]]
        section.table(Table("Network Information", ["", "", "", ""], r, row_header_cols=3,
                            footnotes=["a. Excluding the bias unit"]))
    err = err_fn or (lambda Yp, Tt: model.net.error_value(Yp, Tt))
    stats_ = _sample_errors(err, specs, Y, T, sample)
    if out.get("summary"):
        r = []
        ename = "Cross Entropy Error" if error == "ce" else "Sum of Squares Error"
        for k, lab in enumerate(("Training", "Testing", "Holdout")):
            if k not in stats_:
                continue
            first = True
            if k < 2:
                r.append([lab, ename, fmt(stats_[k]["error"], 3)])
                first = False
            for sp, v in zip(specs, stats_[k]["per"]):
                if sp.kind == "categorical":
                    r.append([lab if first else "", "Percent Incorrect Predictions" +
                              (f" ({sp.name})" if len(specs) > 1 else ""), f"{100 * v:.1f}%"])
                else:
                    r.append([lab if first else "", "Relative Error" +
                              (f" ({sp.name})" if len(specs) > 1 else ""), fmt(v, 3)])
                first = False
            if k == 0:
                if rule:
                    r.append(["", "Stopping Rule Used", rule])
                r.append(["", "Training Time", time.strftime("%H:%M:%S", time.gmtime(secs))
                          + f".{int((secs % 1) * 1000):03d}"])
        foot = [f"Dependent Variable: {', '.join(sp.name for sp in specs)}"]
        if "ᵃ" in rule:
            foot.append("a. Error computations are based on the testing sample.")
        section.table(Table("Model Summary", ["", "", ""], r, row_header_cols=2,
                            footnotes=foot))
    for sp in specs:
        if sp.kind != "categorical":
            continue
        P = Y[:, sp.cols]
        truth = T[:, sp.cols].argmax(1)
        pred = P.argmax(1)
        if out.get("classification"):
            rr = []
            for k, lab in enumerate(("Training", "Testing", "Holdout")):
                sel = sample == k
                if sel.any():
                    rr += classification_rows(truth[sel], pred[sel], sp.labels, lab)
            section.table(Table("Classification", ["Sample", "Observed"] +
                                [("Predicted", c) for c in sp.labels] +
                                [("Predicted", "Percent Correct")], rr, row_header_cols=2,
                                footnotes=[f"Dependent Variable: {sp.name}"]))
        Pc = np.clip(P, 1e-12, None)
        Pn = Pc / Pc.sum(1, keepdims=True)
        if out.get("roc"):
            section.image("ROC Curve", roc_chart(Pn, truth, sp.labels, sp.name), width=460)
            ar = [["", lab, fmt(auc(Pn[:, j], truth == j), 3)]
                  for j, lab in enumerate(sp.labels)]
            ar[0][0] = sp.name
            section.table(Table("Area Under the Curve", ["", "", "Area"], ar,
                                row_header_cols=2))
        if out.get("gains") or out.get("lift"):
            g, lft = gains_lift_charts(Pn, truth, sp.labels, sp.name)
            if out.get("gains"):
                section.image("Gain", g, width=460)
            if out.get("lift"):
                section.image("Lift", lft, width=460)
        if out.get("pred_obs"):
            section.image("Predicted-by-Observed", predicted_by_observed(Pn, truth, sp.labels,
                                                                        sp.name), width=480)
    for sp in specs:
        if sp.kind != "scale":
            continue
        y = T[:, sp.cols[0]] * sp.scale + sp.centre
        yh = Y[:, sp.cols[0]] * sp.scale + sp.centre
        if out.get("pred_obs"):
            section.image("Predicted-by-Observed", scatter_observed_predicted(y, yh, sp.name),
                          width=440)
        if out.get("resid"):
            section.image("Residual-by-Predicted", scatter_observed_predicted(
                y, yh, sp.name, residuals=True), width=440)


def _finish(ds, section, prefix, model, Y, specs, rows, sample, save, out, predict, X, enc,
            factors, covs):
    if out.get("importance"):
        sel = sample <= 1
        kinds = ["factor"] * len(factors) + ["cov"] * len(covs)
        imp = sensitivity(predict, X[sel], enc.groups, kinds)
        importance_table(section, enc.names, imp)
        section.image("Independent Variable Importance",
                      importance_chart(enc.names, imp, "Normalized Importance"))
    saved = {}
    preds = _outputs_to_predictions(specs, Y)
    for i, (sp, pr) in enumerate(zip(specs, preds)):
        suf = f"_{i + 1}" if len(specs) > 1 else ""
        if "predicted" in save:
            if sp.kind == "categorical":
                vals = np.array([sp.levels[p] if not isinstance(sp.levels[p], str) else np.nan
                                 for p in pr], float)
                saved[f"{prefix}_PredictedValue{suf}"] = (
                    f"Predicted Value for {sp.name}", full_series(ds, rows, vals))
            else:
                saved[f"{prefix}_PredictedValue{suf}"] = (
                    f"Predicted Value for {sp.name}", full_series(ds, rows, pr))
        if "probability" in save and sp.kind == "categorical":
            for j, lab in enumerate(sp.labels):
                saved[f"{prefix}_PseudoProbability{suf}_{j + 1}"] = (
                    f"Predicted Pseudo-probability for {sp.name}={lab}",
                    full_series(ds, rows, Y[:, sp.cols[j]]))
    if "partition" in save:
        saved["Partition"] = ("Partition (1 = training, 0 = testing, -1 = holdout)",
                              full_series(ds, rows, np.choose(sample, [1.0, 0.0, -1.0])))
    return MLResult(section, saved, model)


def _parameter_table(net: Net, w, enc: Encoder, specs) -> Table:
    Ws = net.unpack(w)
    layer_names = [f"Hidden Layer {i + 1}" for i in range(len(Ws) - 1)] + ["Output Layer"]
    unit_names = [[f"H({i + 1}:{j + 1})" for j in range(net.sizes[i + 1])]
                  for i in range(len(Ws) - 1)] + [_output_names(specs)]
    cols = ["Predictor", ""]
    for ln, un in zip(layer_names, unit_names):
        cols += [("Predicted", f"{ln}: {u}") for u in un]
    width = sum(len(u) for u in unit_names)
    rows = []
    in_names = [["(Bias)"] + enc.columns] + [["(Bias)"] + un for un in unit_names[:-1]]
    offset = 0
    for li, W in enumerate(Ws):
        lname = "Input Layer" if li == 0 else layer_names[li - 1]
        for r, nm in enumerate(in_names[li]):
            vals = [""] * width
            for c in range(W.shape[1]):
                vals[offset + c] = fmt(W[r, c], 3)
            rows.append([lname if r == 0 else "", nm] + vals)
        offset += W.shape[1]
    return Table("Parameter Estimates", cols, rows, row_header_cols=2)


def network_diagram(net: Net, enc: Encoder, specs, w=None, rbf: bool = False) -> bytes:
    """Layered network graph; for an MLP, line width shows |weight| and colour its sign."""
    layers = [["(Bias)"] + enc.columns]
    for i, h in enumerate(net.sizes[1:-1]):
        names = [f"H({j + 1})" if rbf else f"H({i + 1}:{j + 1})" for j in range(h)]
        layers.append(names if rbf else ["(Bias)"] + names)
    layers.append(_output_names(specs))
    big = max(len(lv) for lv in layers)
    fig = figure(7.5, max(3.5, 0.32 * big + 1.2))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.92])
    ax.axis("off")
    pos = []
    for li, names in enumerate(layers):
        ys = np.linspace(0.95, 0.05, len(names)) if len(names) > 1 else np.array([0.5])
        pos.append([(li, y) for y in ys])
    Ws = net.unpack(w) if w is not None else None
    for li in range(len(layers) - 1):
        src, dst = pos[li], pos[li + 1]
        dst_off = 0 if (li + 1 == len(layers) - 1 or rbf) else 1
        for a, (x0, y0) in enumerate(src):
            for b in range(len(dst) - dst_off):
                x1, y1 = dst[b + dst_off]
                if Ws is not None:
                    val = Ws[li][a, b]
                    lw = 0.3 + 2.2 * min(abs(val), 3) / 3
                    col = "#1f5fbf" if val > 0 else "#d9822b"
                else:
                    lw, col = 0.4, "#999"
                ax.plot([x0, x1], [y0, y1], color=col, lw=lw, alpha=0.6, zorder=1)
    fs = max(5, min(8, 120 / big))
    for li, names in enumerate(layers):
        for (x, y), nm in zip(pos[li], names):
            ax.scatter([x], [y], s=120, color="#4f81bd" if nm != "(Bias)" else "#999",
                       zorder=2)
            ha = "right" if li == 0 else ("left" if li == len(layers) - 1 else "center")
            dx = -0.06 if li == 0 else (0.06 if li == len(layers) - 1 else 0)
            dy = 0 if li in (0, len(layers) - 1) else 0.035
            ax.text(x + dx, y + dy, nm, ha=ha, va="center", fontsize=fs, zorder=3)
    ax.set_xlim(-0.9, len(layers) - 1 + 0.9)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Hidden layer activation function: " + (
        "Softmax" if rbf and net.hidden_act == "softmax" else "Exponential" if rbf else
        {"tanh": "Hyperbolic tangent", "sigmoid": "Sigmoid"}[net.hidden_act]) +
        "    Output layer activation function: " +
        {"identity": "Identity", "softmax": "Softmax", "tanh": "Hyperbolic tangent",
         "sigmoid": "Sigmoid"}[net.out_act], fontsize=8)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    return buf.getvalue()


def selftest() -> None:
    r = np.random.default_rng(0)
    X = r.normal(size=(120, 2))
    t = (X[:, 0] * X[:, 1] > 0).astype(int)
    T = np.eye(2)[t]
    spec = [TargetSpec("y", "categorical", [0, 1], ["0", "1"], cols=[0, 1])]
    net, w, rule, _ = mlp_fit(X, T, spec, (6,), seed=1)
    acc = np.mean(net.predict(w, X).argmax(1) == t)
    assert acc > 0.85, acc
    m = rbf_fit_units(X, T, [], 2, 6, True, None)
    assert np.mean(m.predict(X).argmax(1) == t) > 0.7
