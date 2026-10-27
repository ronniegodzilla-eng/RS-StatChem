"""Shared resampling machinery for the SPSS *Bootstrap…* and *Exact…* options and for the
automatic pooling of multiply imputed data (Phase 0 of the SPSS gap analysis).

Bootstrap (SPSS BOOTSTRAP command)
----------------------------------
Case resampling, *simple* (draw n cases with replacement) or *stratified* (draw within each
stratum, keeping the stratum sizes). Every statistic is re-estimated on each bootstrap sample
and SPSS's columns are reported:

* **Bias** = mean of the bootstrap estimates minus the original estimate,
* **Std. Error** = standard deviation (n-1) of the bootstrap estimates,
* **Sig. (2-tailed)** = ``min(1, 2·min(P*(θ* ≤ θ0), P*(θ* ≥ θ0)))`` — the significance level
  at which the percentile interval just excludes the null value θ0 (0 unless stated), so the
  p value and the percentile interval never contradict each other,
* **Confidence interval**: *percentile* or *bias-corrected and accelerated (BCa)* (Efron &
  Tibshirani 1993, §14.3), computed exactly as ``scipy.stats.bootstrap`` does (linear
  quantiles, z0 from the "mean" percentile of the original estimate, acceleration from the
  jackknife), so results with the same seed agree with SciPy to machine precision.

The random numbers come from NumPy's PCG64 generator seeded with the dialog's seed (SPSS uses
the Mersenne Twister, so individual bootstrap samples differ from SPSS, the method does not).
Statistics are evaluated on batches of index matrices so simple statistics are fully
vectorised; model refits (logistic, Cox, GENLIN…) go through ``ModelInfo.refit``.

Exact tests
-----------
:class:`ExactSpec` carries the *Exact Tests* dialog (Asymptotic only / Monte Carlo / Exact
with a time limit) and maps it onto the ``exact_*`` functions of
:mod:`rsstatchem.core.stats.nptests` and :func:`rsstatchem.core.stats.crosstabs.exact_tests`.

Multiple imputation
-------------------
:func:`rubin` (Rubin's rules with the Barnard-Rubin df) is re-exported from
:mod:`rsstatchem.core.stats.imputation`; :mod:`rsstatchem.core.stats.mi_pool` builds the
pooled SPSS tables on top of it.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy import special, stats

from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import fmt, fmt_int, fmt_p, sup
from rsstatchem.core.stats.imputation import Pooled, rubin  # noqa: F401 (re-export)

DEFAULT_SEED = 2000000
_LETTERS = "abcdefghijklmnopqrstuvwxyz"

# Optional progress hook installed by the UI: ``PROGRESS(done, total)``; it may raise
# :class:`Cancelled` to stop a long bootstrap.
PROGRESS: Callable[[int, int], None] | None = None


class Cancelled(Exception):
    """Raised by the progress hook when the user cancels a bootstrap."""


# ============================================================================ specifications
@dataclass
class BootSpec:
    samples: int = 1000
    seed: int | None = DEFAULT_SEED
    level: float = 0.95
    ci: str = "percentile"            # "percentile" | "bca"
    sampling: str = "simple"          # "simple" | "stratified"
    strata: tuple = ()

    @classmethod
    def from_options(cls, opts: dict | None) -> "BootSpec | None":
        """The dialog dictionary (``{"perform": True, ...}``) → spec, or None when off."""
        if not opts or not opts.get("perform"):
            return None
        seed = int(opts["seed"]) if opts.get("use_seed", True) and opts.get("seed") is not None \
            else None
        strata = tuple(opts.get("strata") or ()) if opts.get("sampling") == "stratified" else ()
        return cls(int(opts.get("samples", 1000)), seed, float(opts.get("level", 95)) / 100,
                   opts.get("ci", "percentile"),
                   "stratified" if strata else "simple", strata)

    @property
    def ci_name(self) -> str:
        return "Bias-corrected and accelerated (BCa)" if self.ci == "bca" else "Percentile"

    def syntax(self) -> str:
        text = "BOOTSTRAP\n  /SAMPLING METHOD=" + (
            f"STRATIFIED(STRATA={' '.join(self.strata)})" if self.strata else "SIMPLE")
        text += f"\n  /VARIABLES INPUT=ALL\n  /CRITERIA CILEVEL={self.level * 100:g} " \
                f"CITYPE={'BCA' if self.ci == 'bca' else 'PERCENTILE'} " \
                f"NSAMPLES={self.samples}\n  /MISSING USERMISSING=EXCLUDE."
        if self.seed is not None:
            text = f"SET SEED={self.seed}.\n" + text
        return text


@dataclass
class ExactSpec:
    method: str = "asymptotic"        # "asymptotic" | "mc" | "exact"
    mc_level: float = 0.99
    mc_samples: int = 10000
    time_limit: float | None = 5.0    # minutes per test (exact)
    seed: int = DEFAULT_SEED

    @classmethod
    def from_options(cls, opts: dict | None) -> "ExactSpec | None":
        if not opts or opts.get("method", "asymptotic") == "asymptotic":
            return None
        limit = float(opts.get("time_limit", 5.0)) if opts.get("use_limit", True) else None
        return cls(opts["method"], float(opts.get("mc_level", 99)) / 100,
                   int(opts.get("mc_samples", 10000)), limit,
                   int(opts.get("seed", DEFAULT_SEED)))

    def kwargs(self, kind: str = "samples") -> dict:
        """Keyword arguments for the ``nptests.exact_*`` functions. ``kind`` is the name of the
        Monte Carlo sample-size argument ("samples" or "samples_mc")."""
        from rsstatchem.core.stats import nptests

        scale = (self.time_limit or 60.0) / 5.0
        return {"method": "exact" if self.method == "exact" else "mc",
                kind: self.mc_samples, "level": self.mc_level, "seed": self.seed,
                "limit": int(nptests.EXACT_LIMIT * max(scale, 0.05))}

    def syntax(self) -> str:
        if self.method == "mc":
            return f"/METHOD=MC CIN({self.mc_level * 100:g}) SAMPLES({self.mc_samples})"
        return f"/METHOD=EXACT TIMER({self.time_limit or 0:g})"


# ============================================================================ resampling
def strata_codes(values) -> np.ndarray | None:
    """Integer stratum codes for a sequence of (possibly mixed) values; None if empty."""
    if values is None:
        return None
    arr = np.asarray(values, dtype=object)
    if arr.ndim == 2:
        arr = np.array(["\x1f".join(map(str, r)) for r in arr], dtype=object)
    _, codes = np.unique(arr.astype(str), return_inverse=True)
    return codes.astype(int)


def index_batches(n: int, samples: int, rng: np.random.Generator,
                  strata: np.ndarray | None = None, batch: int | None = None):
    """Yield (b, n) integer index matrices of bootstrap samples (rows = samples).

    Simple sampling consumes the generator exactly like ``scipy.stats.bootstrap`` with
    ``batch=None`` (``rng.integers(0, n, (B, n))``) when the batches are concatenated."""
    if batch is None:
        batch = max(1, min(samples, int(4e6 // max(n, 1))))
    groups = None
    if strata is not None:
        groups = [np.flatnonzero(strata == c) for c in np.unique(strata)]
    done = 0
    while done < samples:
        b = min(batch, samples - done)
        if groups is None:
            idx = rng.integers(0, n, (b, n))
        else:
            idx = np.empty((b, n), dtype=np.int64)
            pos = 0
            for g in groups:
                idx[:, pos:pos + len(g)] = g[rng.integers(0, len(g), (b, len(g)))]
                pos += len(g)
        done += b
        yield idx


def _jackknife_batches(n: int, batch: int | None = None):
    base = np.arange(n)
    if batch is None:
        batch = max(1, min(n, int(4e6 // max(n, 1))))
    for start in range(0, n, batch):
        rows = np.arange(start, min(n, start + batch))
        yield np.array([np.delete(base, i) for i in rows])


@dataclass
class BootResult:
    estimate: np.ndarray
    boot: np.ndarray                  # (B, k) bootstrap estimates (NaN = failed sample)
    spec: BootSpec
    bias: np.ndarray = None
    se: np.ndarray = None
    lower: np.ndarray = None
    upper: np.ndarray = None
    p: np.ndarray = None
    n_valid: np.ndarray = None
    accel: np.ndarray | None = None
    null: np.ndarray | float = 0.0
    seconds: float = 0.0

    def cells(self, i: int, decimals: int = 3, sig: bool = True) -> list[str]:
        """Bias, Std. Error, [Sig. (2-tailed)], Lower, Upper as formatted strings."""
        if not np.isfinite(self.estimate[i]) or self.n_valid[i] < 2:
            return [""] * (5 if sig else 4)
        out = [fmt(self.bias[i], decimals), fmt(self.se[i], decimals)]
        if sig:
            out.append(fmt_p(self.p[i]))
        return out + [fmt(self.lower[i], decimals), fmt(self.upper[i], decimals)]

    @property
    def min_valid(self) -> int:
        finite = self.n_valid[np.isfinite(self.estimate)]
        return int(finite.min()) if len(finite) else 0


def _summarise(res: BootResult, jack: np.ndarray | None) -> BootResult:
    boot, est, spec = res.boot, res.estimate, res.spec
    k = boot.shape[1]
    alpha = (1 - spec.level) / 2
    res.bias = np.full(k, np.nan)
    res.se = np.full(k, np.nan)
    res.lower = np.full(k, np.nan)
    res.upper = np.full(k, np.nan)
    res.p = np.full(k, np.nan)
    res.n_valid = np.zeros(k, int)
    res.accel = np.full(k, np.nan)
    null = np.broadcast_to(np.asarray(res.null, float), (k,))
    for j in range(k):
        col = boot[:, j]
        col = col[np.isfinite(col)]
        B = len(col)
        res.n_valid[j] = B
        if B < 2 or not np.isfinite(est[j]):
            continue
        res.bias[j] = col.mean() - est[j]
        res.se[j] = col.std(ddof=1)
        lo_q, hi_q = alpha, 1 - alpha
        if spec.ci == "bca" and jack is not None:
            jj = jack[:, j]
            jj = jj[np.isfinite(jj)]
            pct_score = (np.count_nonzero(col < est[j]) + np.count_nonzero(col <= est[j])) \
                / (2.0 * B)
            z0 = special.ndtri(pct_score)
            nj = float(len(jj))
            u = (nj - 1) * (jj.mean() - jj)
            den = np.sum(u ** 2) / nj ** 2
            a = (np.sum(u ** 3) / nj ** 3) / 6 / den ** 1.5 if den > 0 else 0.0
            res.accel[j] = a
            za = special.ndtri(alpha)
            n1, n2 = z0 + za, z0 - za
            lo_q = special.ndtr(z0 + n1 / (1 - a * n1))
            hi_q = special.ndtr(z0 + n2 / (1 - a * n2))
        if np.isfinite(lo_q) and np.isfinite(hi_q):
            res.lower[j], res.upper[j] = np.quantile(col, [lo_q, hi_q], method="linear")
        below = np.count_nonzero(col <= null[j]) / B
        above = np.count_nonzero(col >= null[j]) / B
        res.p[j] = min(1.0, 2 * min(below, above))
    return res


def bootstrap_indices(n: int, evaluate: Callable[[np.ndarray], np.ndarray],
                      estimate, spec: BootSpec, strata=None, null=0.0,
                      batch: int | None = None) -> BootResult:
    """Core engine. ``evaluate(idx)`` maps a (b, n) index matrix to (b, k) statistics (rows of
    NaN for samples where the statistic cannot be computed)."""
    t0 = time.perf_counter()
    estimate = np.atleast_1d(np.asarray(estimate, float))
    rng = np.random.default_rng(spec.seed)
    codes = strata_codes(strata) if strata is not None and len(spec.strata) else None
    out, done = [], 0
    for idx in index_batches(n, spec.samples, rng, codes, batch):
        vals = np.asarray(evaluate(idx), float).reshape(len(idx), -1)
        out.append(vals)
        done += len(idx)
        if PROGRESS is not None:
            PROGRESS(done, spec.samples)
    boot = np.vstack(out) if out else np.empty((0, len(estimate)))
    jack = None
    if spec.ci == "bca":
        jack = np.vstack([np.asarray(evaluate(j), float).reshape(len(j), -1)
                          for j in _jackknife_batches(n, batch)])
    res = BootResult(estimate, boot, spec, null=null)
    _summarise(res, jack)
    res.seconds = time.perf_counter() - t0
    return res


def bootstrap(data, statistic: Callable, spec: BootSpec, strata=None, vectorized: bool = False,
              null=0.0) -> BootResult:
    """Bootstrap a statistic of the rows of ``data`` (n x p array).

    ``statistic(x)`` gets an (n, p) array and returns k values; with ``vectorized=True`` it gets
    a (b, n, p) array and returns (b, k)."""
    x = np.asarray(data, float)
    if x.ndim == 1:
        x = x[:, None]
    n = len(x)
    est = np.atleast_1d(np.asarray(statistic(x[None])[0] if vectorized else statistic(x),
                                   float))

    def evaluate(idx):
        if vectorized:
            return statistic(x[idx])
        rows = []
        for r in idx:
            try:
                rows.append(np.atleast_1d(np.asarray(statistic(x[r]), float)))
            except (ValueError, ZeroDivisionError, np.linalg.LinAlgError, FloatingPointError):
                rows.append(np.full(len(est), np.nan))
        return np.vstack(rows)

    batch = max(1, int(1_500_000 // max(n * x.shape[1], 1))) if vectorized else 50
    with np.errstate(all="ignore"):
        return bootstrap_indices(n, evaluate, est, spec, strata, null, batch)


# ============================================================================ models
@dataclass
class ModelInfo:
    """Parameter estimates of a fitted model, for bootstrapping and for pooling.

    ``refit(idx)`` re-estimates the parameters on the analysis cases ``idx`` (positions
    0..n-1, with repeats) and returns k values (NaN if it fails); ``refit_batch(idx)`` is an
    optional vectorised version for a (b, n) index matrix."""
    names: list                       # row labels (str, or tuple of header cells)
    estimate: np.ndarray
    se: np.ndarray | None = None
    refit: Callable | None = None
    refit_batch: Callable | None = None
    n: int = 0
    rows: np.ndarray | None = None    # positions of the analysis cases in the dataset
    table: str = "Parameter Estimates"
    headers: list = field(default_factory=lambda: ["Parameter"])
    est_label: str = "B"
    group: str = ""                   # e.g. model number in Linear Regression
    df: float | None = None           # complete-data residual df (Barnard-Rubin)
    stat: str = "t"                   # pooled statistic: "t" or "wald"
    exp: bool = False                 # pooled table shows Exp(B)
    note: str = ""                    # why bootstrapping is not available (if set)
    bootable: bool = True
    batch: int | None = None          # bootstrap samples per refit_batch call
    stat_label: str = ""              # pooled statistic column title (default t / Wald)

    @property
    def p_values(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            z = self.estimate / self.se
        if self.stat == "t" and self.df:
            return 2 * stats.t.sf(np.abs(z), self.df)
        return stats.chi2.sf(z * z, 1)

    def label_cells(self) -> list[list[str]]:
        return [list(n) if isinstance(n, (tuple, list)) else [n] for n in self.names]


def bootstrap_model(model: ModelInfo, spec: BootSpec, strata=None) -> BootResult:
    if model.refit_batch is not None:
        evaluate = model.refit_batch
    else:
        def evaluate(idx):
            out = np.full((len(idx), len(model.estimate)), np.nan)
            for i, r in enumerate(idx):
                try:
                    out[i] = model.refit(r)
                except (ValueError, ZeroDivisionError, np.linalg.LinAlgError,
                        FloatingPointError, OverflowError):
                    pass
            return out
    batch = model.batch if model.refit_batch is not None else 25
    with np.errstate(all="ignore"):
        return bootstrap_indices(model.n, evaluate, model.estimate, spec, strata, batch=batch)


# ============================================================================ output helpers
def next_letter(table: Table) -> str:
    used = {f.split(".", 1)[0].strip() for f in table.footnotes if "." in f[:3]}
    for ch in _LETTERS:
        if ch not in used:
            return ch
    return "z"


def boot_footnote(spec: BootSpec, letter: str, res: BootResult | None = None) -> str:
    text = f"{letter}. Unless otherwise noted, bootstrap results are based on {spec.samples} " \
           f"{'stratified ' if spec.strata else ''}bootstrap samples"
    if res is not None and res.min_valid < spec.samples:
        text += f" (results based on {res.min_valid} valid samples where some samples could " \
                "not be used)"
    return text


def boot_headers(spec: BootSpec, letter: str, sig: bool = True) -> list:
    g = f"Bootstrap{sup(letter)}"
    ci = f"{spec.level * 100:g}% CI"
    out = [(g, "Bias"), (g, "Std. Error")]
    if sig:
        out.append((g, "Sig. (2-tailed)"))
    return out + [(g, f"{ci} Lower"), (g, f"{ci} Upper")]


def specifications_table(spec: BootSpec) -> Table:
    rows = [["Sampling Method", "Stratified" if spec.strata else "Simple"],
            ["Number of Samples", fmt_int(spec.samples)],
            ["Confidence Interval Level", f"{spec.level * 100:.1f}%"],
            ["Confidence Interval Type", spec.ci_name]]
    if spec.strata:
        rows.append(["Strata Variables", ", ".join(spec.strata)])
    if spec.seed is not None:
        rows.append(["Random Number Seed", str(spec.seed)])
    return Table("Bootstrap Specifications", ["", ""], rows)


def add_specifications(section: Section, spec: BootSpec) -> None:
    """Insert the SPSS "Bootstrap Specifications" table after the syntax log and put the
    BOOTSTRAP command in front of the procedure's syntax (as SPSS pastes it)."""
    if any(getattr(i, "title", "") == "Bootstrap Specifications" for i in section.items):
        return
    pos = 0
    while pos < len(section.items) and getattr(section.items[pos], "style", "") == "log":
        pos += 1
    section.items.insert(pos, specifications_table(spec))
    if pos and "BOOTSTRAP" not in section.items[0].text:
        section.items[0].text = spec.syntax() + "\n" + section.items[0].text


def add_subcommand(section: Section, text: str) -> None:
    """Append a subcommand (e.g. ``/METHOD=EXACT TIMER(5)``) to the section's syntax."""
    log = next((i for i in section.items if getattr(i, "style", "") == "log"), None)
    if log is None:
        section.log(text)
        return
    body = log.text.rstrip()
    log.text = (body[:-1] if body.endswith(".") else body) + f"\n  {text}."


def insert_boot_rows(table: Table, mapping: dict[int, dict[int, tuple[BootResult, int]]],
                     spec: BootSpec, decimals: int | dict = 3, stat_label: str = "Statistic",
                     sig: bool = False) -> Table:
    """Insert SPSS bootstrap sub-rows (Bias, Std. Error, [Sig.], CI Lower, CI Upper) after the
    data rows in ``mapping`` (row index → {column index → (BootResult, statistic index)}).

    One (or, for single-header tables, two) row-header columns are added, as in SPSS's
    Descriptives / Correlations / Group Statistics tables with bootstrapping."""
    letter = next_letter(table)
    h = table.row_header_cols
    add = 2 if h == 1 else 1
    labels = ["Bias", "Std. Error"] + (["Sig. (2-tailed)"] if sig else []) + \
        [f"{spec.level * 100:g}% CI Lower", f"{spec.level * 100:g}% CI Upper"]
    columns = list(table.columns[:h]) + [""] * add + list(table.columns[h:])
    rows = []
    min_valid = spec.samples
    for r, row in enumerate(table.rows):
        cells = list(row)
        if r in mapping:
            head = cells[:h] + ([stat_label, ""] if add == 2 else [""])
        else:
            head = cells[:h] + [""] * add
        rows.append(head + cells[h:])
        if r not in mapping:
            continue
        sub = [[""] * (h + add) + [""] * (len(cells) - h) for _ in labels]
        for k, lab in enumerate(labels):
            sub[k][h + add - 1] = lab
        if add == 2:
            sub[0][h] = f"Bootstrap{sup(letter)}"
        else:
            sub[0][h - 1] = f"Bootstrap{sup(letter)}"
        for col, (res, j) in mapping[r].items():
            d = decimals.get(col, 3) if isinstance(decimals, dict) else decimals
            vals = res.cells(j, d, sig)
            min_valid = min(min_valid, int(res.n_valid[j]) if np.isfinite(res.estimate[j])
                            else spec.samples)
            for k in range(len(labels)):
                sub[k][h + add + col - h] = vals[k]
        rows.extend(sub)
    out = Table(table.title, columns, rows, row_header_cols=h + add,
                footnotes=list(table.footnotes))
    note = boot_footnote(spec, letter)
    if min_valid < spec.samples:
        note += f" (some statistics are based on fewer valid samples, minimum {min_valid})"
    out.footnotes.append(note)
    return out


def append_boot_columns(table: Table, mapping: dict[int, tuple[BootResult, int]],
                        spec: BootSpec, decimals: int = 3, sig: bool = True) -> Table:
    """Append Bootstrap Bias / Std. Error / Sig. / CI columns to the rows in ``mapping``."""
    letter = next_letter(table)
    heads = boot_headers(spec, letter, sig)
    rows = []
    for r, row in enumerate(table.rows):
        extra = [""] * len(heads)
        if r in mapping:
            res, j = mapping[r]
            extra = res.cells(j, decimals, sig)
        rows.append(list(row) + extra)
    out = Table(table.title, list(table.columns) + heads, rows,
                row_header_cols=table.row_header_cols, footnotes=list(table.footnotes))
    out.footnotes.append(boot_footnote(spec, letter))
    return out


def boot_table(title: str, headers: list[str], labels: list[list[str]], estimates,
               res: BootResult, spec: BootSpec, est_label: str = "B", decimals: int = 3,
               sig: bool = True, footnotes: list[str] | None = None,
               index: list[int] | None = None) -> Table:
    """A separate SPSS "Bootstrap for …" table (row headers, estimate, bootstrap columns)."""
    letter = _LETTERS[len(footnotes or [])]
    heads = boot_headers(spec, letter, sig)
    rows = []
    index = list(range(len(labels))) if index is None else index
    for lab, j in zip(labels, index):
        est = estimates[j]
        if j < 0:
            rows.append(list(lab) + [""] * (1 + len(heads)))
            continue
        rows.append(list(lab) + [fmt(est, decimals)] + res.cells(j, decimals, sig))
    notes = list(footnotes or []) + [boot_footnote(spec, letter, res)]
    return Table(title, list(headers) + [est_label] + heads, rows,
                 row_header_cols=len(headers), footnotes=notes)


def model_boot_table(models: list[ModelInfo], results: list[BootResult], spec: BootSpec,
                     title: str | None = None, footnotes: list[str] | None = None) -> Table:
    """One "Bootstrap for <table>" table for one or several fitted models (e.g. blocks)."""
    m0 = models[0]
    grouped = any(m.group for m in models)
    letter = _LETTERS[len(footnotes or [])]
    heads = boot_headers(spec, letter)
    headers = (["Model"] if grouped else []) + list(m0.headers)
    rows = []
    for m, res in zip(models, results):
        labels = m.label_cells()
        for i, lab in enumerate(labels):
            lead = ([m.group if i == 0 else ""] if grouped else [])
            lab = lab + [""] * (len(m0.headers) - len(lab))
            rows.append(lead + lab + [fmt(m.estimate[i])] + res.cells(i))
    worst = min((r.min_valid for r in results), default=spec.samples)
    note = boot_footnote(spec, letter)
    if worst < spec.samples:
        note += f" (minimum {worst} valid samples; samples in which the model could not be " \
                "estimated are excluded)"
    return Table(title or f"Bootstrap for {m0.table}", headers + [m0.est_label] + heads, rows,
                 row_header_cols=len(headers), footnotes=list(footnotes or []) + [note])


def find_table(section: Section, title_prefix: str, start: int = 0) -> int | None:
    for i in range(start, len(section.items)):
        item = section.items[i]
        if getattr(item, "kind", "") == "table" and item.title.startswith(title_prefix):
            return i
    return None


def place_after(section: Section, title_prefix: str, item, last: bool = True) -> None:
    """Insert ``item`` after the (last) table whose title starts with ``title_prefix``."""
    pos = None
    for i, it in enumerate(section.items):
        if getattr(it, "kind", "") == "table" and it.title.startswith(title_prefix):
            pos = i
            if not last:
                break
    if pos is None:
        section.items.append(item)
    else:
        section.items.insert(pos + 1, item)


# ============================================================================ exact helpers
def exact_columns(spec: ExactSpec, one_sided: bool = False, point: bool = False) -> list:
    if spec.method == "mc":
        g2 = "Monte Carlo Sig. (2-sided)"
        ci = f"{spec.mc_level * 100:g}% Confidence Interval"
        out = [(g2, "Significance"), (g2, f"{ci} Lower Bound"), (g2, f"{ci} Upper Bound")]
        return out
    out = ["Exact Sig. (2-sided)"]
    if one_sided:
        out.append("Exact Sig. (1-sided)")
    if point:
        out.append("Point Probability")
    return out


def pvalue_text(p) -> str:
    return fmt_p(float(p)) if p is not None and np.isfinite(float(p)) else ""


def selftest() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(5, 2, 40)
    spec = BootSpec(samples=300, seed=7)
    res = bootstrap(x, lambda a: a.mean(axis=1), spec, vectorized=True)
    assert abs(res.se[0] - 2 / math.sqrt(40)) < 0.15 and res.lower[0] < 5.5 < res.upper[0] + 1
    bca = bootstrap(x, lambda a: a.mean(axis=1), BootSpec(200, 7, ci="bca"), vectorized=True)
    assert np.isfinite(bca.lower[0]) and np.isfinite(bca.accel[0])
    strat = bootstrap(x, lambda a: a.mean(axis=1), BootSpec(100, 7, strata=("g",)),
                      strata=np.repeat([0, 1], 20), vectorized=True)
    assert strat.n_valid[0] == 100
    pl = rubin(np.array([[1.0], [1.2], [0.9]]), np.array([[0.04], [0.05], [0.045]]))
    assert pl.m == 3 and 0 < pl.fmi[0] < 1
    t = insert_boot_rows(Table("T", ["", "Mean"], [["x", "5.0"]]), {0: {1: (res, 0)}}, spec)
    assert len(t.rows) == 5 and t.row_header_cols == 3
    assert stats.norm.ppf(0.5) == 0.0
