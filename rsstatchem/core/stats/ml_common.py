"""Shared helpers for the SPSS Decision Trees / Neural Networks / Nearest Neighbor / TwoStep
procedures (WP9): measurement levels, one-of-c coding, covariate rescaling, random partitions,
classification tables and the model-viewer charts (ROC, cumulative gains, lift, importance,
predicted-by-observed). numpy/scipy/matplotlib only (no scikit-learn in app code).

Random assignment uses numpy's MT19937 generator seeded like SPSS's Mersenne Twister (default
seed 2000000), so runs are reproducible, but the case-by-case draws differ from SPSS's.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from rsstatchem.core.dataset import NUMERIC, Dataset
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import fmt, fmt_int, value_text
from rsstatchem.core.stats.roc import figure, png, style

DEFAULT_SEED = 2000000
SAMPLES = ("Training", "Testing", "Holdout")


class MLError(ValueError):
    pass


@dataclass
class MLResult:
    section: Section
    saved: dict = field(default_factory=dict)
    model: object = None


def rng(seed: int | None) -> np.random.Generator:
    return np.random.Generator(np.random.MT19937(DEFAULT_SEED if seed is None else int(seed)))


# ----------------------------------------------------------------------------- variables
def measure_of(ds: Dataset, name: str) -> str:
    var = ds.var(name)
    if var.type != NUMERIC:
        return "nominal"
    return var.measure if var.measure in ("nominal", "ordinal", "scale") else "scale"


def raw_column(ds: Dataset, name: str) -> pd.Series:
    """User-missing aware column; numeric variables as float, strings as object."""
    s = ds.analysis_series(name)
    if ds.var(name).type == NUMERIC:
        return pd.to_numeric(s, errors="coerce")
    return s.where(s.astype(str).str.len() > 0)


def sort_levels(values) -> list:
    return sorted(set(values), key=lambda v: (isinstance(v, str), v))


def level_labels(ds: Dataset, name: str, levels: list) -> list[str]:
    var = ds.var(name)
    return [value_text(var, v) for v in levels]


@dataclass
class Categorical:
    name: str
    levels: list
    labels: list[str]
    codes: np.ndarray          # int codes, -1 = missing


def categorical(ds: Dataset, name: str, rows: np.ndarray | None = None,
                levels: list | None = None) -> Categorical:
    s = raw_column(ds, name)
    if rows is not None:
        s = s.iloc[rows]
    vals = s.to_numpy(object)
    present = [v for v in vals if not (v is None or (isinstance(v, float) and math.isnan(v)))]
    lv = sort_levels(present) if levels is None else list(levels)
    index = {v: i for i, v in enumerate(lv)}
    codes = np.array([index.get(v, -1) if not (v is None or (isinstance(v, float)
                                                               and math.isnan(v))) else -1
                      for v in vals], dtype=int)
    return Categorical(name, lv, level_labels(ds, name, lv), codes)


# ----------------------------------------------------------------------------- encoding
RESCALE = {"standardized": "Standardized", "normalized": "Normalized",
           "adjusted": "Adjusted Normalized", "none": "None"}


@dataclass
class Encoder:
    """Factors as one-of-c indicator columns, covariates rescaled with statistics from the
    training sample (as SPSS's MLP/RBF/KNN do)."""
    factors: list[str]
    covariates: list[str]
    rescale: str = "standardized"
    levels: dict = field(default_factory=dict)
    labels: dict = field(default_factory=dict)
    centre: np.ndarray | None = None
    scale: np.ndarray | None = None
    columns: list[str] = field(default_factory=list)
    groups: list[list[int]] = field(default_factory=list)  # column indices per predictor

    def fit(self, ds: Dataset, rows: np.ndarray, train: np.ndarray) -> "Encoder":
        for f in self.factors:
            cat = categorical(ds, f, rows)
            lv = sort_levels([cat.levels[c] for c in cat.codes[train] if c >= 0])
            self.levels[f] = lv
            self.labels[f] = level_labels(ds, f, lv)
        if self.covariates:
            X = np.column_stack([raw_column(ds, c).to_numpy(float)[rows] for c in
                                 self.covariates])[train]
            if self.rescale == "standardized":
                c, s = X.mean(0), X.std(0, ddof=1)
            elif self.rescale == "normalized":
                c, s = X.min(0), X.max(0) - X.min(0)
            elif self.rescale == "adjusted":
                c, s = (X.max(0) + X.min(0)) / 2, (X.max(0) - X.min(0)) / 2
            else:
                c, s = np.zeros(X.shape[1]), np.ones(X.shape[1])
            s = np.where(~np.isfinite(s) | (s == 0), 1.0, s)
            self.centre, self.scale = c, s
        self.columns, self.groups = [], []
        k = 0
        for f in self.factors:
            n = len(self.levels[f])
            self.columns += [f"[{f}={lab}]" for lab in self.labels[f]]
            self.groups.append(list(range(k, k + n)))
            k += n
        for c in self.covariates:
            self.columns.append(c)
            self.groups.append([k])
            k += 1
        return self

    @property
    def names(self) -> list[str]:
        return list(self.factors) + list(self.covariates)

    def transform(self, ds: Dataset, rows: np.ndarray) -> np.ndarray:
        parts = []
        for f in self.factors:
            cat = categorical(ds, f, rows, self.levels[f])
            Z = np.zeros((len(rows), len(self.levels[f])))
            ok = cat.codes >= 0
            Z[np.where(ok)[0], cat.codes[ok]] = 1.0
            parts.append(Z)
        if self.covariates:
            X = np.column_stack([raw_column(ds, c).to_numpy(float)[rows] for c in
                                 self.covariates])
            parts.append((X - self.centre) / self.scale)
        return np.column_stack(parts) if parts else np.zeros((len(rows), 0))


def complete_rows(ds: Dataset, names: list[str]) -> np.ndarray:
    ok = np.ones(ds.n_cases, bool)
    for n in names:
        s = raw_column(ds, n)
        ok &= s.notna().to_numpy()
    return np.where(ok)[0]


# ----------------------------------------------------------------------------- partitions
def partition(n: int, train: float = 70, test: float = 30, holdout: float = 0,
              seed: int | None = None, variable: np.ndarray | None = None) -> np.ndarray:
    """Sample code per case: 0 training, 1 testing, 2 holdout (SPSS Partitions tab).

    Relative numbers of cases are honoured in expectation (each case drawn independently with
    the given probabilities), as SPSS does. A partition variable assigns positive values to
    training, 0 to testing and negative values to holdout."""
    if variable is not None:
        v = np.asarray(variable, float)
        return np.where(v > 0, 0, np.where(v == 0, 1, 2)).astype(int)
    tot = float(train + test + holdout)
    if tot <= 0:
        raise MLError("The relative numbers of cases must sum to a positive number.")
    p = np.array([train, test, holdout], float) / tot
    u = rng(seed).random(n)
    return np.where(u < p[0], 0, np.where(u < p[0] + p[1], 1, 2)).astype(int)


def processing_table(section: Section, sample: np.ndarray, excluded: int,
                     names=SAMPLES) -> None:
    n = len(sample)
    rows = []
    for k, label in enumerate(names):
        c = int((sample == k).sum())
        if c or k == 0:
            rows.append(["Sample", label, fmt_int(c), _pct(c, n)])
    rows.append(["Valid", "", fmt_int(n), "100.0%"])
    rows.append(["Excluded", "", fmt_int(excluded), ""])
    rows.append(["Total", "", fmt_int(n + excluded), ""])
    section.table(Table("Case Processing Summary", ["", "", "N", "Percent"], rows,
                        row_header_cols=2))


def _pct(a: float, b: float, d: int = 1) -> str:
    return f"{100 * a / b:.{d}f}%" if b else ""


# ----------------------------------------------------------------------------- tables
def classification_rows(truth: np.ndarray, pred: np.ndarray, labels: list[str],
                        sample_label: str | None = None) -> list[list[str]]:
    """SPSS Classification table rows (Observed x Predicted, Percent Correct)."""
    J = len(labels)
    cm = np.zeros((J, J), int)
    for t, p in zip(truth, pred):
        if t >= 0 and p >= 0:
            cm[t, p] += 1
    rows = []
    first = True
    for i in range(J):
        lead = [sample_label if first else ""] if sample_label is not None else []
        first = False
        tot = cm[i].sum()
        rows.append(lead + [labels[i]] + [fmt_int(v) for v in cm[i]]
                    + [_pct(cm[i, i], tot) if tot else "0.0%"])
    n = cm.sum()
    lead = [""] if sample_label is not None else []
    rows.append(lead + ["Overall Percent"] + [_pct(cm[:, j].sum(), n) for j in range(J)]
                + [_pct(np.trace(cm), n)])
    return rows


def importance_table(section: Section, names: list[str], imp: np.ndarray,
                     title: str = "Independent Variable Importance") -> None:
    imp = np.asarray(imp, float)
    mx = imp.max() if len(imp) and imp.max() > 0 else 1.0
    rows = [[n, fmt(v, 3), f"{100 * v / mx:.1f}%"] for n, v in zip(names, imp)]
    section.table(Table(title, ["", "Importance", "Normalized Importance"], rows))


# ----------------------------------------------------------------------------- charts
def importance_chart(names: list[str], imp: np.ndarray, title: str = "Normalized Importance",
                     xlabel: str = "Normalized Importance") -> bytes:
    imp = np.asarray(imp, float)
    mx = imp.max() if len(imp) and imp.max() > 0 else 1.0
    order = np.argsort(imp)
    fig = figure(6.0, max(2.2, 0.38 * len(names) + 1.0))
    ax = fig.add_subplot(111)
    ax.barh([names[i] for i in order], 100 * imp[order] / mx, color="#4f81bd")
    ax.set_xlim(0, 100)
    style(ax, title, xlabel + " (%)", "")
    return png(fig)


def roc_points(score: np.ndarray, positive: np.ndarray):
    order = np.argsort(-score, kind="mergesort")
    s, y = score[order], positive[order].astype(float)
    distinct = np.r_[np.where(np.diff(s))[0], len(s) - 1]
    tps = np.cumsum(y)[distinct]
    fps = np.cumsum(1 - y)[distinct]
    P, N = max(y.sum(), 1e-300), max((1 - y).sum(), 1e-300)
    tpr = np.r_[0, tps / P]
    fpr = np.r_[0, fps / N]
    return fpr, tpr


def auc(score: np.ndarray, positive: np.ndarray) -> float:
    pos, neg = score[positive], score[~positive]
    if not len(pos) or not len(neg):
        return float("nan")
    from scipy.stats import rankdata
    r = rankdata(np.r_[pos, neg])
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def roc_chart(prob: np.ndarray, truth: np.ndarray, labels: list[str], dep: str) -> bytes:
    fig = figure(5.4, 4.6)
    ax = fig.add_subplot(111)
    for j, lab in enumerate(labels):
        pos = truth == j
        if pos.any() and (~pos).any():
            fpr, tpr = roc_points(prob[:, j], pos)
            ax.plot(fpr, tpr, label=lab)
    ax.plot([0, 1], [0, 1], color="#999", lw=0.8)
    ax.legend(title=dep, fontsize=8)
    style(ax, "ROC Curve", "1 - Specificity", "Sensitivity")
    return png(fig)


def gains_curves(prob: np.ndarray, truth: np.ndarray, j: int):
    order = np.argsort(-prob[:, j], kind="mergesort")
    y = (truth[order] == j).astype(float)
    n = len(y)
    pct = np.arange(1, n + 1) / n * 100
    gain = np.cumsum(y) / max(y.sum(), 1e-300) * 100
    lift = gain / pct
    return np.r_[0, pct], np.r_[0, gain], pct, lift


def gains_lift_charts(prob: np.ndarray, truth: np.ndarray, labels: list[str],
                      dep: str) -> tuple[bytes, bytes]:
    fig = figure(5.4, 4.6)
    ax = fig.add_subplot(111)
    fig2 = figure(5.4, 4.6)
    ax2 = fig2.add_subplot(111)
    for j, lab in enumerate(labels):
        if not (truth == j).any():
            continue
        x, g, x2, lift = gains_curves(prob, truth, j)
        ax.plot(x, g, label=lab)
        ax2.plot(x2, lift, label=lab)
    ax.plot([0, 100], [0, 100], color="#999", lw=0.8, label="Baseline")
    ax2.axhline(1, color="#999", lw=0.8)
    ax.legend(title=dep, fontsize=8)
    ax2.legend(title=dep, fontsize=8)
    style(ax, "Cumulative Gains", "Percentage", "Gain (%)")
    style(ax2, "Lift", "Percentage", "Lift")
    return png(fig), png(fig2)


def predicted_by_observed(prob: np.ndarray, truth: np.ndarray, labels: list[str],
                          dep: str) -> bytes:
    """Clustered box plot of predicted pseudo-probabilities by observed category."""
    J = len(labels)
    fig = figure(max(5.0, 1.2 * J + 2), 4.4)
    ax = fig.add_subplot(111)
    width = 0.8 / J
    colors = [f"C{i}" for i in range(J)]
    for i in range(J):
        for j in range(J):
            data = prob[truth == i, j]
            if len(data):
                bp = ax.boxplot(data, positions=[i + (j - (J - 1) / 2) * width],
                                widths=width * 0.9, patch_artist=True, showfliers=False)
                bp["boxes"][0].set_facecolor(colors[j])
    ax.set_xticks(range(J), labels)
    from matplotlib.patches import Patch
    ax.legend([Patch(color=c) for c in colors], labels, title="Predicted category",
              fontsize=8)
    style(ax, "Predicted-by-Observed", dep, "Predicted Pseudo-probability")
    return png(fig)


def scatter_observed_predicted(y: np.ndarray, yhat: np.ndarray, dep: str,
                               residuals: bool = False) -> bytes:
    fig = figure(5.0, 4.4)
    ax = fig.add_subplot(111)
    if residuals:
        ax.scatter(yhat, y - yhat, s=12, alpha=0.7)
        ax.axhline(0, color="#999", lw=0.8)
        style(ax, "Residual-by-Predicted", f"Predicted Value for {dep}", "Residual")
    else:
        ax.scatter(yhat, y, s=12, alpha=0.7)
        lo, hi = float(min(y.min(), yhat.min())), float(max(y.max(), yhat.max()))
        ax.plot([lo, hi], [lo, hi], color="#999", lw=0.8)
        style(ax, "Predicted-by-Observed", f"Predicted Value for {dep}", dep)
    return png(fig)


def full_series(ds: Dataset, rows: np.ndarray, values, dtype=float) -> pd.Series:
    out = pd.Series(np.nan, index=ds.df.index, dtype=float)
    out.iloc[rows] = np.asarray(values, dtype)
    return out
