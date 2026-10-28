"""Analyze > Classify > Nearest Neighbor (SPSS KNN), following IBM's "KNN Algorithms" chapter.

* Preprocessing: categorical features one-of-c coded; scale features optionally rescaled by
  adjusted normalisation x' = 2(x − min)/(max − min) − 1 with the training minimum/maximum.
* Distance: Euclidean or city block, optionally weighted by normalised feature importance.
* Number of neighbours: fixed k, or the k in [k_min, k_max] with the smallest V-fold
  cross-validated error rate (categorical target) or sum-of-squares error (scale target);
  ties choose the smallest k.
* Feature selection: forward wrapper (Cunningham & Delany 2007) starting from the forced
  features; each step adds the feature (a categorical feature's indicators as one set) that
  reduces the training error most; stops after a fixed number of features (automatic:
  min(20, P) − forced) or when the absolute change in the error ratio is below the minimum
  (0.01). The training error leaves the case itself out of its own neighbours. Combined
  k-and-feature selection runs forward selection for every k and keeps the best pair.
* Scoring: majority vote (ties → the category with most training cases, then the smallest
  value), Laplace-corrected probabilities (n_j + 1)/(k + J); mean or median for a scale
  target.
* Feature importance: the error ratio e(without p)/e of every feature in the model, normalised
  to sum to one.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import fmt, fmt_int
from rsstatchem.core.stats.ml_common import (Encoder, MLError, MLResult, categorical,
                                             classification_rows, complete_rows, full_series,
                                             importance_chart, measure_of, partition,
                                             processing_table, raw_column, rng)
from rsstatchem.core.stats.roc import figure, png, style


@dataclass
class KNNModel:
    k: int
    metric: str
    features: list[int]            # selected predictor indices (encoder groups)
    weights: np.ndarray | None
    Xtrain: np.ndarray
    ytrain: np.ndarray
    J: int                         # categories (0 = scale target)
    prior_counts: np.ndarray | None
    scale_fn: str = "mean"

    def columns(self, groups: list[list[int]]) -> list[int]:
        return [c for f in self.features for c in groups[f]]


def _dist2(A: np.ndarray, B: np.ndarray, metric: str, w: np.ndarray | None) -> np.ndarray:
    if A.shape[1] == 0:
        return np.zeros((len(A), len(B)))
    if metric == "cityblock":
        D = np.zeros((len(A), len(B)))
        for c in range(A.shape[1]):
            D += (1.0 if w is None else w[c]) * np.abs(A[:, None, c] - B[None, :, c])
        return D
    if w is not None:
        sw = np.sqrt(w)
        A, B = A * sw, B * sw
    sa, sb = (A ** 2).sum(1), (B ** 2).sum(1)
    return np.sqrt(np.maximum(sa[:, None] + sb[None, :] - 2 * A @ B.T, 0.0))


def neighbours(Xq: np.ndarray, Xt: np.ndarray, k: int, metric: str, w=None,
               exclude_self: bool = False, block: int = 512):
    """Indices (n_q × k) and distances of the k nearest training cases."""
    n = len(Xq)
    k = min(k, len(Xt) - (1 if exclude_self else 0))
    idx = np.zeros((n, k), int)
    dist = np.zeros((n, k))
    for s in range(0, n, block):
        D = _dist2(Xq[s:s + block], Xt, metric, w)
        if exclude_self:
            r = np.arange(s, min(s + block, n))
            D[r - s, r] = np.inf
        part = np.argpartition(D, k - 1, axis=1)[:, :k] if k < D.shape[1] else \
            np.tile(np.arange(D.shape[1]), (len(D), 1))
        dd = np.take_along_axis(D, part, 1)
        order = np.argsort(dd, axis=1, kind="mergesort")
        idx[s:s + block] = np.take_along_axis(part, order, 1)
        dist[s:s + block] = np.take_along_axis(dd, order, 1)
    return idx, dist


def predict_from(nb: np.ndarray, y: np.ndarray, J: int, counts: np.ndarray | None,
                 fn: str = "mean"):
    vals = y[nb]
    if J == 0:
        pred = np.median(vals, 1) if fn == "median" else vals.mean(1)
        return pred, None
    k = nb.shape[1]
    tally = np.zeros((len(nb), J))
    for j in range(J):
        tally[:, j] = (vals == j).sum(1)
    score = tally + (counts / (counts.sum() + 1.0))[None, :] * 1e-6 - \
        np.arange(J)[None, :] * 1e-12
    pred = score.argmax(1)
    prob = (tally + 1) / (k + J)
    return pred, prob


def _error(pred, y, J) -> float:
    if J == 0:
        return float(((y - pred) ** 2).sum())
    return float(np.mean(pred != y))


def train_error(X: np.ndarray, y: np.ndarray, k: int, metric: str, J: int, counts,
                w=None, fn="mean") -> float:
    nb, _ = neighbours(X, X, k, metric, w, exclude_self=True)
    pred, _ = predict_from(nb, y, J, counts, fn)
    return _error(pred, y, J)


def cv_error(X, y, k, metric, J, counts, folds: np.ndarray, w=None, fn="mean") -> float:
    errs = []
    for v in np.unique(folds):
        te, tr = folds == v, folds != v
        if tr.sum() < k:
            continue
        nb, _ = neighbours(X[te], X[tr], k, metric, w)
        pred, _ = predict_from(nb, y[tr], J, counts, fn)
        e = _error(pred, y[te], J)
        errs.append(e)
    return float(np.mean(errs)) if errs else float("nan")


def forward_select(X, y, groups, k, metric, J, counts, forced: list[int], max_add: int | None,
                   min_change: float = 0.01, fn="mean"):
    """IBM forward wrapper. Returns (selected groups, log rows [(n features, added, error)])."""
    chosen = list(forced)
    remaining = [g for g in range(len(groups)) if g not in chosen]
    cols = lambda S: [c for g in S for c in groups[g]]  # noqa: E731
    log = []
    cur = train_error(X[:, cols(chosen)], y, k, metric, J, counts, fn=fn) if chosen else \
        math.inf
    added = 0
    while remaining:
        if max_add is not None and added >= max_add:
            break
        errs = [train_error(X[:, cols(chosen + [g])], y, k, metric, J, counts, fn=fn)
                for g in remaining]
        b = int(np.argmin(errs))
        e = errs[b]
        if max_add is None and math.isfinite(cur):
            if e >= cur or cur == 0 or (cur - e) / cur < min_change:
                break
        chosen.append(remaining.pop(b))
        log.append((len(chosen), chosen[-1], e))
        cur = e
        added += 1
    return chosen, log


def feature_importance(X, y, groups, chosen, k, metric, J, counts, fn="mean") -> np.ndarray:
    cols = lambda S: [c for g in S for c in groups[g]]  # noqa: E731
    e = train_error(X[:, cols(chosen)], y, k, metric, J, counts, fn=fn)
    r = []
    for g in chosen:
        rest = [h for h in chosen if h != g]
        e_g = train_error(X[:, cols(rest)], y, k, metric, J, counts, fn=fn) if rest else \
            (1 - 1 / max(J, 1) if J else float(((y - y.mean()) ** 2).sum()))
        r.append(e_g / e if e > 0 else e_g + 1.0)
    r = np.array(r, float)
    return r / r.sum() if r.sum() > 0 else np.full(len(r), 1 / max(len(r), 1))


def knn_procedure(ds: Dataset, target: str, features: list[str], k: int = 3,
                  auto_k: bool = False, k_min: int = 3, k_max: int = 5, metric: str = "euclidean",
                  weight_importance: bool = False, normalize: bool = True,
                  scale_fn: str = "mean", select_features: bool = False,
                  forced: list[str] | None = None, max_features: int | None = None,
                  min_change: float = 0.01, train_pct: float = 70.0, holdout_pct: float = 30.0,
                  partition_var: str | None = None, folds: int = 10, fold_var: str | None = None,
                  focal_var: str | None = None, seed: int | None = None,
                  output: dict | None = None, save: tuple = ()) -> MLResult:
    out = {"charts": True, "classification": True, "neighbours": True, **(output or {})}
    if not features:
        raise MLError("Select at least one feature.")
    forced = [f for f in (forced or []) if f in features]
    names = [target] + list(features) + [v for v in (partition_var, fold_var) if v]
    rows = complete_rows(ds, names)
    if len(rows) < 3:
        raise MLError("Too few valid cases.")
    tkind = measure_of(ds, target)
    is_cat = tkind != "scale"
    if is_cat:
        cat = categorical(ds, target, rows)
        y_all = cat.codes.astype(int)
        ylabels = cat.labels
        J = len(ylabels)
    else:
        y_all = raw_column(ds, target).to_numpy(float)[rows]
        ylabels, J = [], 0
    pv = raw_column(ds, partition_var).to_numpy(float)[rows] if partition_var else None
    sample = partition(len(rows), train_pct, 0, holdout_pct, seed, pv)
    sample = np.where(sample == 1, 2, sample) if pv is None else sample
    train = np.where(sample == 0)[0]
    hold = np.where(sample == 2)[0]
    if len(train) < 3:
        raise MLError("The training sample has too few cases.")
    factors = [f for f in features if measure_of(ds, f) != "scale"]
    covs = [f for f in features if f not in factors]
    order = factors + covs
    enc = Encoder(factors, covs, "adjusted" if normalize else "none").fit(ds, rows, train)
    X = enc.transform(ds, rows)
    groups = enc.groups
    Xt, yt = X[train], y_all[train]
    counts = np.bincount(yt, minlength=J).astype(float) if is_cat else None
    if fold_var:
        fv = raw_column(ds, fold_var).to_numpy(float)[rows][train]
    else:
        fv = rng(seed).permutation(len(train)) % max(2, int(folds))
    allg = list(range(len(groups)))
    forced_g = [order.index(f) for f in forced]
    cols = lambda S: [c for g in S for c in groups[g]]  # noqa: E731
    ks = list(range(int(k_min), int(k_max) + 1)) if auto_k else [int(k)]
    k_log, fs_logs = [], {}
    best = None
    for kk in ks:
        if select_features:
            P = len(groups)
            madd = max_features if max_features is not None else None
            chosen, flog = forward_select(Xt, yt, groups, kk, metric, J, counts, forced_g, madd,
                                          min_change, scale_fn)
            if not chosen:
                chosen = allg
            fs_logs[kk] = flog
            err = train_error(Xt[:, cols(chosen)], yt, kk, metric, J, counts, fn=scale_fn)
        else:
            chosen = allg
            err = cv_error(Xt, yt, kk, metric, J, counts, fv, fn=scale_fn) if auto_k else \
                train_error(Xt, yt, kk, metric, J, counts, fn=scale_fn)
        k_log.append((kk, err))
        if best is None or err < best[1] - 1e-12:
            best = (kk, err, chosen)
    kbest, _, chosen = best
    fi = feature_importance(Xt, yt, groups, chosen, kbest, metric, J, counts, scale_fn)
    w = None
    if weight_importance:
        w = np.zeros(X.shape[1])
        for g, v in zip(chosen, fi):
            w[groups[g]] = v
        w = w[cols(chosen)]
    model = KNNModel(kbest, metric, chosen, w, Xt[:, cols(chosen)], yt, J, counts, scale_fn)
    Xs = X[:, cols(chosen)]
    nb_tr, d_tr = neighbours(Xs[train], model.Xtrain, kbest, metric, w, exclude_self=True)
    nb_all, d_all = neighbours(Xs, model.Xtrain, kbest, metric, w)
    nb_all[train], d_all[train] = nb_tr, d_tr
    pred, prob = predict_from(nb_all, yt, J, counts, scale_fn)

    section = Section("Nearest Neighbor Analysis")
    nbtxt = f"AUTO(KMIN={k_min}, KMAX={k_max})" if auto_k else f"FIXED(K={k})"
    section.log(f"KNN {target} {'(MLEVEL=N)' if is_cat else '(MLEVEL=S)'} BY {' '.join(factors)}"
                f" WITH {' '.join(covs)}\n  /MODEL METRIC={metric.upper()} "
                f"NEIGHBORS={nbtxt}"
                f" FEATURES={'FORWARDSELECTION' if select_features else 'ALL'}\n"
                f"  /PARTITION TRAINING={train_pct:g} HOLDOUT={holdout_pct:g}.")
    processing_table(section, np.where(sample == 2, 1, 0), ds.n_cases - len(rows),
                     ("Training", "Holdout"))
    sel_names = [order[g] for g in chosen]
    section.table(Table("Model Summary", ["", ""], [
        ["Number of Nearest Neighbors (k)", fmt_int(kbest)],
        ["Distance Computation", "Euclidean" if metric == "euclidean" else "City Block"],
        ["Feature Weighting", "Importance" if weight_importance else "Equal"],
        ["Rescaling of Scale Features", "Adjusted Normalized" if normalize else "None"],
        ["Features in Model", ", ".join(sel_names)],
        ["Prediction for Scale Target", scale_fn.title() if not is_cat else "Majority vote"]]))
    if is_cat and out.get("classification"):
        rows_c = []
        for lab, idx in (("Training", train), ("Holdout", hold)):
            if len(idx):
                rows_c += classification_rows(y_all[idx], pred[idx], ylabels, lab)
        section.table(Table("Classification Table", ["Partition", "Observed"] +
                            [("Predicted", c) for c in ylabels] +
                            [("Predicted", "Percent Correct")], rows_c, row_header_cols=2))
    rows_e = []
    for lab, idx in (("Training", train), ("Holdout", hold)):
        if len(idx):
            if is_cat:
                wrong = 100 * np.mean(pred[idx] != y_all[idx])
                rows_e.append([lab, "Percent Incorrect", f"{wrong:.1f}%"])
            else:
                sse = float(((pred[idx] - y_all[idx]) ** 2).sum())
                rows_e.append([lab, "Sum of Squares Error", fmt(sse, 3)])
    section.table(Table("Error Summary", ["Partition", "", "Value"], rows_e, row_header_cols=2))
    if auto_k:
        lab = "Error Rate" if is_cat else "Sum of Squares Error"
        section.table(Table("k Selection Error Log", ["k", lab],
                            [[str(a), fmt(b, 4)] for a, b in k_log],
                            footnotes=[f"{int(folds)}-fold cross-validation on the training "
                                       "sample" if not select_features else
                                       "Training error with forward feature selection"]))
        if out.get("charts"):
            fig = figure(4.8, 3.6)
            ax = fig.add_subplot(111)
            ax.plot([a for a, _ in k_log], [b for _, b in k_log], marker="o")
            style(ax, "Error Rate by k" if is_cat else "Error by k", "k", lab)
            section.image("k Selection Error Log", png(fig), width=420)
    if select_features and fs_logs.get(kbest):
        lab = "Error Rate" if is_cat else "Sum of Squares Error"
        section.table(Table("Feature Selection Error Log", ["Number of Features",
                                                            "Feature Added", lab],
                            [[str(n), order[g], fmt(e, 4)] for n, g, e in fs_logs[kbest]],
                            footnotes=[f"k = {kbest}; forced features: "
                                       f"{', '.join(forced) or 'none'}"]))
    if len(chosen) and out.get("charts"):
        section.table(Table("Feature Importance", ["Feature", "Importance"],
                            [[nm, fmt(v, 3)] for nm, v in zip(sel_names, fi)]))
        section.image("Predictor Importance", importance_chart(sel_names, fi,
                                                               "Predictor Importance",
                                                               "Importance"))
    if out.get("neighbours"):
        if focal_var:
            fvv = raw_column(ds, focal_var).to_numpy(float)[rows]
            focal = np.where(fvv == 1)[0]
        else:
            focal = hold[:10] if len(hold) else train[:10]
        if len(focal):
            cols_n = ["Focal Case"] + [("Nearest Neighbors", str(i + 1)) for i in range(kbest)] \
                + [("Nearest Distances", str(i + 1)) for i in range(kbest)]
            rows_n = []
            for f in focal[:50]:
                rows_n.append([str(rows[f] + 1)] + [str(rows[train[j]] + 1) for j in nb_all[f]]
                              + [fmt(d, 3) for d in d_all[f]])
            section.table(Table("k Nearest Neighbors and Distances", cols_n, rows_n,
                                footnotes=["Case numbers refer to the active dataset."]))
    saved = {}
    if "predicted" in save:
        if is_cat:
            lv = categorical(ds, target, rows).levels
            pvv = np.array([lv[p] if not isinstance(lv[p], str) else np.nan for p in pred],
                           float)
        else:
            pvv = pred
        saved["KNN_PredictedValue"] = ("Predicted Value", full_series(ds, rows, pvv))
    if "probability" in save and is_cat:
        for j in range(J):
            saved[f"KNN_PredictedProbability_{j + 1}"] = (
                f"Predicted Probability for {target} = {ylabels[j]}",
                full_series(ds, rows, prob[:, j]))
    if "partition" in save and not partition_var:
        saved["Partition"] = ("Partition (1 = training, -1 = holdout)",
                              full_series(ds, rows, np.where(sample == 0, 1.0, -1.0)))
    return MLResult(section, saved, model)


def selftest() -> None:
    r = np.random.default_rng(2)
    X = np.vstack([r.normal(0, 1, (50, 2)), r.normal(3, 1, (50, 2))])
    y = np.r_[np.zeros(50, int), np.ones(50, int)]
    nb, _ = neighbours(X, X, 3, "euclidean", exclude_self=True)
    pred, prob = predict_from(nb, y, 2, np.bincount(y).astype(float))
    assert np.mean(pred == y) > 0.9 and np.all((prob > 0) & (prob < 1))
