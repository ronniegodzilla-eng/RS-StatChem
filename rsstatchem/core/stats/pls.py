"""Analyze > Structural Equation Modeling > PLS-SEM (variance-based SEM, as in SmartPLS 3/4,
R seminr / plspm).

Runs the PLS path-modelling algorithm (path, factorial or centroid weighting; optional
consistent PLS), the measurement- and structural-model assessment of Hair et al. (*A Primer on
PLS-SEM*, 3rd ed., 2022), bootstrapping, blindfolding (Q²), PLSpredict and a path diagram.
Output tables use SmartPLS wording. The numerical work lives in :mod:`pls_engine`, the model
syntax in :mod:`pls_model` and the diagram in :mod:`pls_diagram`.
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats import pls_engine as E
from rsstatchem.core.stats.common import fmt, fmt_int, fmt_p, numeric
from rsstatchem.core.stats.pls_diagram import path_diagram
from rsstatchem.core.stats.pls_model import (  # noqa: F401
    MODE_TEXT,
    Construct,
    ModelError,
    PLSModel,
    mediation_template,
    parse_syntax,
)

DEFAULTS = {
    "scheme": "path", "max_iter": 300, "stop": 7, "missing": "listwise",
    "consistent": False, "bootstrap": True, "subsamples": 5000, "seed": 1,
    "ci": "percentile", "tails": 2, "alpha": 0.05, "blindfold": True, "omission": 7,
    "predict": False, "folds": 10, "reps": 10, "diagram": True, "diagram_value": "p",
    "crossloadings": True, "fit": True, "notes": True, "hypotheses": True,
    "save_scores": False,
}
CI_TEXT = {"percentile": "Percentile bootstrap", "bc": "Bias-corrected (BC) bootstrap",
           "bca": "Bias-corrected and accelerated (BCa) bootstrap"}


@dataclass
class PLSResult:
    section: Section
    saved: dict = field(default_factory=dict)
    structure: E.Structure | None = None
    estimates: dict = field(default_factory=dict)   # original-sample arrays (no batch axis)
    quality: dict = field(default_factory=dict)
    boot: E.BootResult | None = None
    n: int = 0


# ---------------------------------------------------------------------- data
def _data(ds: Dataset, names: list[str], missing: str) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.concat({n: numeric(ds, n) for n in dict.fromkeys(names)}, axis=1)
    if missing == "mean":
        frame = frame.fillna(frame.mean())
    rows = frame.notna().all(axis=1).to_numpy()
    return frame[rows].reset_index(drop=True), rows


def _f(x, d: int = 3) -> str:
    return fmt(x, d)


def _matrix_table(title: str, row_names: list[str], col_names: list[str], values,
                  show=lambda i, j: True, footnotes=None) -> Table:
    rows = []
    for i, rn in enumerate(row_names):
        rows.append([rn] + [_f(values[i, j]) if show(i, j) and np.isfinite(values[i, j])
                            else "" for j in range(len(col_names))])
    return Table(title, [""] + list(col_names), rows, footnotes=footnotes or [])


def _base_name(name: str) -> str:
    base = re.sub(r"\W", "_", name)
    return ("LV_" + base)[:56]


# ---------------------------------------------------------------------- main entry
def pls_sem(ds: Dataset, model: PLSModel | str, **options) -> PLSResult:
    o = dict(DEFAULTS)
    o.update(options)
    if isinstance(model, str):
        model = parse_syntax(model)
    model = copy.deepcopy(model)
    numeric_names = {v.name for v in ds.variables if v.type == "numeric"}
    warnings = model.validate(ds.names(), numeric_names)
    st1 = E.make_structure(model.main_model())
    frame, rows = _data(ds, st1.indicators, o["missing"])
    n = len(frame)
    if n < 3:
        raise ModelError("Too few valid cases for PLS-SEM.")
    sd = frame.std(ddof=0)
    if np.any(sd == 0):
        raise ModelError("Indicator(s) without variance: " + ", ".join(sd.index[sd == 0]))
    for j in range(st1.K):
        if st1.modes[j] == "B" and len(st1.blocks[j]) >= n:
            raise ModelError(f"Formative construct {st1.names[j]} has more indicators than "
                             "cases.")
    tol = 10.0 ** (-float(o["stop"]))
    scheme, max_iter, consistent = o["scheme"], int(o["max_iter"]), bool(o["consistent"])
    stage1 = None
    if model.interactions:
        st = E.make_structure(model.stage2_model())
        X1 = frame[st1.indicators].to_numpy(float)
        Z1, _, _ = E.standardize(X1)
        W1 = E.estimate((Z1.T @ Z1 / n)[None], st1, scheme, max_iter, tol)["W"][0]
        Y1 = Z1 @ W1
        for x, w, _t in model.interactions:
            name = model.interaction_name(x, w)
            frame[name] = Y1[:, st1.names.index(x)] * Y1[:, st1.names.index(w)]
        X = frame[st.indicators].to_numpy(float)
        terms = [(st.indicators.index(model.interaction_name(x, w)), st1.names.index(x),
                  st1.names.index(w)) for x, w, _t in model.interactions]
        stage1 = E.Stage1(st1, np.array([st.indicators.index(i) for i in st1.indicators]),
                          terms, scheme, max_iter, tol)
        if o["predict"]:
            o["predict"] = False
            warnings.append("PLSpredict is not available for models with interaction terms.")
        if consistent:
            warnings.append("PLSc with interaction terms: the interaction construct is a "
                            "single item built from stage-1 PLS scores.")
    else:
        st = st1
        X = frame[st.indicators].to_numpy(float)
    Z, _, _ = E.standardize(X)
    S = Z.T @ Z / n
    est = {k: v[0] for k, v in E.estimate(S[None], st, scheme, max_iter, tol,
                                          consistent).items()}
    labels = E.stat_labels(st)
    W, C, R, L, P, r2 = (est[k] for k in ("W", "C", "R", "loadings", "paths", "r2"))
    rel = E.reliability(S, L, est["rhoA"], st)
    f2 = E.f_squared(R[None], r2[None], st)[0]
    tot = E.total_effects(P[None], st)[0]
    ovif, ivif = E.outer_vif(S, st), E.inner_vif(R, st)
    htmt_pairs = [(i, j) for j, i, _ in labels["htmt"]]
    htmt_vals = E.htmt(S[None], st, htmt_pairs)[0] if htmt_pairs else np.zeros(0)
    quality = {"reliability": rel, "f2": f2, "total": tot, "outer_vif": ovif,
               "inner_vif": ivif, "htmt": dict(zip(htmt_pairs, htmt_vals)),
               "iterations": int(est["iterations"]), "converged": bool(est["converged"])}
    if o["fit"]:
        sat = E.implied_corr(S, L, R, st)
        phi = E.structural_implied(R, P, r2, st)
        estm = E.implied_corr(S, L, phi, st)
        quality["fit"] = {"saturated": E.fit_indices(S, sat, n),
                          "estimated": E.fit_indices(S, estm, n)}
    boot = None
    if o["bootstrap"]:
        boot = E.bootstrap(X, st, int(o["subsamples"]), _seed(o["seed"]), scheme, max_iter, tol,
                           consistent, o["ci"], int(o["tails"]), float(o["alpha"]),
                           stage1=stage1)
    if o["blindfold"]:
        quality["q2"] = E.blindfolding(Z, st, int(o["omission"]), scheme, max_iter, tol)
    if o["predict"]:
        quality["predict"] = E.pls_predict(X, st, int(o["folds"]), int(o["reps"]),
                                           _seed(o["seed"]), scheme, max_iter, tol)
    section = _report(ds, model, st, o, n, est, quality, boot, warnings)
    if model.interactions:
        _simple_slopes(section, model, st, est)
    saved = {}
    if o["save_scores"]:
        Y = Z @ W
        for j, name in enumerate(st.names):
            series = pd.Series(np.nan, index=ds.df.index)
            series.iloc[np.flatnonzero(rows)] = Y[:, j]
            saved[_base_name(name)] = (f"Latent variable score: {name} (PLS-SEM)", series, 5)
    return PLSResult(section, saved, st, est, quality, boot, n)


def _png_width(png: bytes) -> int:
    return int.from_bytes(png[16:20], "big") if png[:4] == b"\x89PNG" else 680


def _seed(value):
    try:
        v = int(value)
    except (TypeError, ValueError):
        return None
    return None if v <= 0 else v


# ---------------------------------------------------------------------- report
def _syntax_log(model: PLSModel, o: dict, n: int) -> str:
    lines = ["PLS-SEM"]
    lines += ["  /MODEL " + line for line in model.to_syntax().splitlines()]
    lines.append(f"  /ALGORITHM {'PLSC' if o['consistent'] else 'PLS'} "
                 f"SCHEME={o['scheme'].upper()} MAXITER={int(o['max_iter'])} "
                 f"STOP=1E-{int(o['stop'])} MISSING="
                 f"{'MEANREPLACE' if o['missing'] == 'mean' else 'LISTWISE'}")
    if o["bootstrap"]:
        test = "TWOTAILED" if int(o["tails"]) == 2 else "ONETAILED"
        lines.append(f"  /BOOTSTRAP SUBSAMPLES={int(o['subsamples'])} SEED={o['seed']} "
                     f"CI={o['ci'].upper()} TEST={test} ALPHA={o['alpha']:g}")
    if o["blindfold"]:
        lines.append(f"  /BLINDFOLDING OMISSION={int(o['omission'])}")
    if o["predict"]:
        lines.append(f"  /PLSPREDICT FOLDS={int(o['folds'])} REPETITIONS={int(o['reps'])}")
    return "\n".join(lines)


def _report(ds, model, st: E.Structure, o, n, est, q, boot, warnings) -> Section:
    names = st.names
    K = st.K
    W, C, R, L, P, r2 = (est[k] for k in ("W", "C", "R", "loadings", "paths", "r2"))
    sec = Section("PLS-SEM: Consistent PLS (PLSc)" if o["consistent"] else "PLS-SEM")
    sec.log(_syntax_log(model, o, n))
    for w in warnings:
        sec.note("Warning", w, style="warning")

    # ---- model and settings
    rows = []
    for j in range(K):
        kind = "Endogenous" if len(st.preds[j]) else "Exogenous"
        items = ", ".join(st.indicators[k] for k in st.blocks[j])
        if st.is_interaction(j):
            items = "product of the stage-1 construct scores"
        rows.append([names[j], MODE_TEXT[st.modes[j]], kind, str(len(st.blocks[j])), items])
    sec.table(Table("Constructs", ["Construct", "Measurement model", "Type", "Indicators",
                                   "Items"], rows))
    miss = "Mean replacement" if o["missing"] == "mean" else "Casewise deletion"
    settings = [["Number of observations used", fmt_int(n)],
                ["Missing value treatment", miss],
                ["Algorithm", "Consistent PLS (PLSc)" if o["consistent"] else "PLS algorithm"],
                ["Weighting scheme", E.SCHEME_TEXT[o["scheme"]]],
                ["Maximum iterations", fmt_int(o["max_iter"])],
                ["Stop criterion", f"1E-{int(o['stop'])}"],
                ["Initial weights", "1.0"],
                ["Number of iterations", fmt_int(q["iterations"])]]
    sec.table(Table("Setting", ["Setting", "Value"], settings))
    if q["converged"]:
        sec.note("", f"The PLS algorithm converged after {q['iterations']} iterations "
                 f"(stop criterion 1E-{int(o['stop'])}).")
    else:
        sec.note("Convergence", f"The PLS algorithm did not converge within "
                 f"{int(o['max_iter'])} iterations. Results should not be interpreted.",
                 style="warning")
    if o["consistent"]:
        bad = [names[j] for j in range(K) if st.reflective(j) and len(st.blocks[j]) > 1
               and not (0 < est["rhoA"][j] <= 1)]
        if bad or np.any(np.abs(R[~np.eye(K, dtype=bool)]) > 1):
            sec.note("PLSc", "Inadmissible PLSc solution (rho_a outside 0–1 or a corrected "
                     "construct correlation above 1" + (f": {', '.join(bad)}" if bad else "")
                     + "). Use the regular PLS algorithm or revise the measurement model.",
                     style="warning")
    min_n = max([len(st.preds[j]) for j in range(K)] +
                [len(st.blocks[j]) for j in range(K) if st.modes[j] == "B"])
    if n < 10 * min_n:
        sec.note("Sample size", f"N = {n} is below the 10-times rule ({10 * min_n}) for the "
                 "largest number of arrows pointing at a construct. Consider a power "
                 "analysis (inverse square root method).", style="warning")

    # ---- diagram
    inner_extra, outer_extra = {}, {}
    if boot is not None and o["diagram_value"] in ("p", "t"):
        key = "P" if o["diagram_value"] == "p" else "T"
        for (i, j, _), row in zip(boot.labels["paths"], boot.table["paths"]):
            if key in row:
                inner_extra[(i, j)] = fmt_p(row[key]) if key == "P" else fmt(row[key], 2)
    if o["diagram"]:
        png = path_diagram(st, W, L, P, r2, inner_extra, outer_extra)
        caption = "Path Diagram"
        if inner_extra:
            caption += " (path coefficients with " + \
                ("p values)" if o["diagram_value"] == "p" else "t values)")
        sec.image(caption, png, width=min(720, _png_width(png)))

    # ---- structural model point estimates
    pred_rows = [i for i in range(K) if len(st.succs[i])]
    endo = st.endogenous
    sec.table(_matrix_table("Path coefficients", [names[i] for i in pred_rows],
                            [names[j] for j in endo], P[np.ix_(pred_rows, endo)],
                            show=lambda a, b: st.D[pred_rows[a], endo[b]] == 1,
                            footnotes=["Rows: predictor constructs; columns: dependent "
                                       "constructs."]))
    rows = []
    for j in endo:
        k = len(st.preds[j])
        adj = 1 - (1 - r2[j]) * (n - 1) / (n - k - 1) if n - k - 1 > 0 else np.nan
        rows.append([names[j], _f(r2[j]), _f(adj)])
    sec.table(Table("R-square", ["", "R-square", "R-square adjusted"], rows))
    f2 = q["f2"]
    sec.table(_matrix_table("f-square", [names[i] for i in pred_rows],
                            [names[j] for j in endo], f2[np.ix_(pred_rows, endo)],
                            show=lambda a, b: st.D[pred_rows[a], endo[b]] == 1))

    if boot is None:  # point estimates of the effects (bootstrap tables show them otherwise)
        labels = E.stat_labels(st)
        vec = {g: v[0] for g, v in E.stat_vectors(
            np.eye(1), {k: np.asarray(v)[None] for k, v in est.items()}, st,
            {**labels, "htmt": []}).items() if g != "htmt"}
        for g, title in (("total_indirect", "Total indirect effects"),
                         ("specific_indirect", "Specific indirect effects"),
                         ("total", "Total effects")):
            if labels[g]:
                sec.table(Table(title, ["", "Value"],
                                [[lab[-1], _f(v)] for lab, v in zip(labels[g], vec[g])]))

    # ---- measurement model
    ind_labels = [st.indicators[k] for k in range(st.p)]

    def show_own(a, b):
        return st.owner[a] == b

    sec.table(_matrix_table("Outer loadings", ind_labels, names, L, show=show_own,
                            footnotes=(["Consistent (PLSc) loadings for reflective "
                                        "constructs."] if o["consistent"] else [])))
    sec.table(_matrix_table("Outer weights", ind_labels, names, W, show=show_own))
    rel = q["reliability"]
    rows = []
    for j in range(K):
        if not st.reflective(j):
            continue
        rows.append([names[j], _f(rel["alpha"][j]), _f(rel["rhoA"][j]), _f(rel["rhoC"][j]),
                     _f(rel["ave"][j])])
    if rows:
        sec.table(Table("Construct reliability and validity",
                        ["", "Cronbach's alpha", "Composite reliability (rho_a)",
                         "Composite reliability (rho_c)", "Average variance extracted (AVE)"],
                        rows, footnotes=["Reflective constructs only; internal consistency "
                                         "and AVE are not meaningful for formative "
                                         "constructs."]))
    # Fornell-Larcker
    fl = R.copy()
    for j in range(K):
        fl[j, j] = np.sqrt(rel["ave"][j]) if st.reflective(j) else np.nan
    sec.table(_matrix_table("Discriminant validity - Fornell-Larcker criterion", names, names,
                            fl, show=lambda a, b: b <= a,
                            footnotes=["Diagonal: square root of the AVE; off-diagonal: "
                                       "construct correlations."]))
    if q["htmt"]:
        refl = [j for j in range(K) if st.reflective(j)]
        H = np.full((K, K), np.nan)
        for (i, j), v in q["htmt"].items():
            H[j, i] = H[i, j] = v
        sec.table(_matrix_table("Discriminant validity - Heterotrait-monotrait ratio (HTMT) "
                                "- Matrix", [names[j] for j in refl], [names[j] for j in refl],
                                H[np.ix_(refl, refl)], show=lambda a, b: b < a,
                                footnotes=["Based on absolute indicator correlations; "
                                           "reflective constructs only."]))
    if o["crossloadings"]:
        sec.table(_matrix_table("Discriminant validity - Cross loadings", ind_labels, names, C))
    sec.table(Table("Collinearity statistics (VIF) - Outer model - List", ["", "VIF"],
                    [[st.indicators[k], _f(q["outer_vif"][k])] for k in range(st.p)]))
    iv = q["inner_vif"]
    sec.table(Table("Collinearity statistics (VIF) - Inner model - List", ["", "VIF"],
                    [[f"{names[i]} -> {names[j]}", _f(iv[i, j])] for i, j in st.edges()]))

    # ---- model fit
    if "fit" in q:
        sat, est_fit = q["fit"]["saturated"], q["fit"]["estimated"]
        rows = [[k, _f(sat[k]) if k != "Chi-square" else fmt(sat[k], 3),
                 _f(est_fit[k])] for k in ("SRMR", "d_ULS", "d_G", "Chi-square", "NFI")]
        sec.table(Table("Model fit", ["", "Saturated model", "Estimated model"], rows,
                        footnotes=["Saturated model: free construct correlations; estimated "
                                   "model: correlations implied by the structural model. "
                                   "SRMR < 0.08 indicates a good fit (Hu & Bentler, 1999); "
                                   "fit indices are of limited use in PLS-SEM."]))

    # ---- bootstrapping
    if boot is not None:
        _boot_tables(sec, boot, st, o)

    # ---- Q² and PLSpredict
    if "q2" in q:
        rows = []
        for j, (sso, sse) in q["q2"].items():
            rows.append([names[j], fmt(sso, 3), fmt(sse, 3), _f(1 - sse / sso)])
        notes = [f"Blindfolding with omission distance D = {int(o['omission'])}."]
        if n % int(o["omission"]) == 0:
            notes.append("The number of observations divided by D is an integer; choose "
                         "another omission distance (Hair et al., 2022).")
        sec.table(Table("Construct crossvalidated redundancy", ["", "SSO", "SSE",
                                                                "Q² (=1-SSE/SSO)"],
                        rows, footnotes=notes))
    if "predict" in q:
        rows = [[r["indicator"], _f(r["Q2predict"]), _f(r["PLS_RMSE"]), _f(r["PLS_MAE"]),
                 _f(r["LM_RMSE"]), _f(r["LM_MAE"])] for r in q["predict"]]
        sec.table(Table("PLSpredict - MV prediction summary",
                        ["", "Q²predict", "PLS-SEM_RMSE", "PLS-SEM_MAE", "LM_RMSE", "LM_MAE"],
                        rows, footnotes=[f"{int(o['folds'])} folds, {int(o['reps'])} "
                                         "repetitions; predictions from the exogenous "
                                         "constructs (earliest antecedents); LM benchmark "
                                         "uses all exogenous indicators."]))
    if o["notes"]:
        _interpretation(sec, st, o, n, est, q, boot)
    return sec


def _simple_slopes(sec: Section, model: PLSModel, st: E.Structure, est: dict) -> None:
    """Simple slope plot (moderator at -1 SD, mean, +1 SD), as in SmartPLS."""
    import io

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    P = est["paths"]
    for x, w, t in model.interactions:
        ix, iw, it = (st.names.index(k) for k in (x, w, t))
        ii = st.names.index(model.interaction_name(x, w))
        bx, bw, bi = P[ix, it], P[iw, it], P[ii, it]
        fig = Figure(figsize=(5.2, 3.6), dpi=120)
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        xs = np.array([-1.0, 1.0])
        for level, color, style in ((-1, "#c0392b", "--"), (0, "#333333", "-"),
                                    (1, "#2f6fae", "-.")):
            ax.plot(xs, (bx + bi * level) * xs + bw * level, color=color, ls=style, lw=1.8,
                    label=f"{w} at {'Mean' if level == 0 else f'{level:+d} SD'}")
        ax.set_xlabel(x)
        ax.set_ylabel(t)
        ax.set_xticks([-1, 0, 1])
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, frameon=False)
        ax.set_title(f"Simple slope analysis: {model.interaction_name(x, w)} -> {t}",
                     fontsize=9)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        sec.image(f"Simple slope analysis - {model.interaction_name(x, w)} -> {t}",
                  buf.getvalue(), width=520)
        sec.note("", f"Slope of {x} on {t}: {fmt(bx - bi)} at {w} = -1 SD, {fmt(bx)} at the "
                 f"mean and {fmt(bx + bi)} at +1 SD (standardized two-stage estimates).")


def _boot_tables(sec: Section, boot: E.BootResult, st: E.Structure, o: dict) -> None:
    lo_q, hi_q = E._ci_bounds(boot.alpha, boot.tails)
    lo_t, hi_t = f"{lo_q * 100:g}%", f"{hi_q * 100:g}%"
    corrected = boot.method in ("bc", "bca")
    ci_group = {"percentile": "Confidence interval", "bc": "Bias corrected CI",
                "bca": "BCa CI"}[boot.method]
    cols = ["", "Original sample (O)", "Sample mean (M)", "Standard deviation (STDEV)",
            "T statistics (|O/STDEV|)", "P values"]
    if corrected:
        cols.append("Bias")
    cols += [(ci_group, lo_t), (ci_group, hi_t)]
    tail = "two-tailed" if boot.tails == 2 else "one-tailed"
    foot = [f"{CI_TEXT[boot.method]}, {boot.B} subsamples ({boot.valid} valid), {tail} "
            f"test, significance level {boot.alpha:g}."]
    titles = (("paths", "Path coefficients"), ("total_indirect", "Total indirect effects"),
              ("specific_indirect", "Specific indirect effects"), ("total", "Total effects"),
              ("loadings", "Outer loadings"), ("weights", "Outer weights"),
              ("r2", "R-square"), ("f2", "f-square"),
              ("htmt", "Heterotrait-monotrait ratio (HTMT)"))
    for g, title in titles:
        rows_in = boot.table.get(g) or []
        if not rows_in:
            continue
        rows = []
        for r in rows_in:
            if "M" not in r:
                rows.append([r["label"], _f(r["O"])] + [""] * (len(cols) - 2))
                continue
            row = [r["label"], _f(r["O"]), _f(r["M"]), _f(r["STDEV"]),
                   _f(r["T"]) if g != "htmt" else "", fmt_p(r["P"]) if g != "htmt" else ""]
            if corrected:
                row.append(_f(r["bias"]))
            row += [_f(r["lo"]), _f(r["hi"])]
            rows.append(row)
        notes = list(foot)
        if g == "htmt":
            notes.append("T statistics and p values are not reported for HTMT; discriminant "
                         "validity is supported when the upper confidence bound is below 0.90 "
                         "(or 0.85); with a one-tailed test the bound is the 95% upper limit.")
        sec.table(Table(f"Bootstrapping - {title}", cols, rows, footnotes=notes))
    if o.get("hypotheses", True):
        rows = []
        h = 1
        for g in ("paths", "specific_indirect"):
            for r in boot.table.get(g) or []:
                if "P" not in r:
                    continue
                sig = np.isfinite(r["P"]) and r["P"] < boot.alpha
                rows.append([f"H{h}", r["label"], _f(r["O"]), _f(r["T"]), fmt_p(r["P"]),
                             f"[{_f(r['lo'])}, {_f(r['hi'])}]",
                             "Supported" if sig else "Not supported"])
                h += 1
        if rows:
            sec.table(Table("Hypothesis testing summary",
                            ["Hypothesis", "Path", "Coefficient", "T statistics", "P values",
                             f"CI {lo_t}-{hi_t}", "Decision"], rows,
                            footnotes=[f"Decision at α = {boot.alpha:g} ({tail}); specific "
                                       "indirect effects test mediation."]))


def _interpretation(sec, st: E.Structure, o, n, est, q, boot) -> None:
    names = st.names
    L, r2 = est["loadings"], est["r2"]
    rel = q["reliability"]
    lines = ["Measurement model (reflective):"]
    low = [f"{st.indicators[k]} ({_f(L[k, st.owner[k]])})" for k in range(st.p)
           if st.reflective(st.owner[k]) and L[k, st.owner[k]] < 0.708]
    lines.append("• Indicator reliability: loadings ≥ 0.708 are recommended. " +
                 ("All reflective loadings meet this." if not low else
                  "Below 0.708: " + ", ".join(low) + ". Items between 0.40 and 0.708 can be "
                  "kept if CR and AVE are adequate; consider deleting items < 0.40."))
    for j in range(st.K):
        if not st.reflective(j):
            continue
        cr, ave, rhoa = rel["rhoC"][j], rel["ave"][j], rel["rhoA"][j]
        issues = []
        if cr < 0.70:
            issues.append("rho_c < 0.70 (insufficient internal consistency)")
        elif cr > 0.95:
            issues.append("rho_c > 0.95 (redundant items)")
        if rhoa < 0.70:
            issues.append("rho_a < 0.70")
        if ave < 0.50:
            issues.append("AVE < 0.50 (convergent validity not established)")
        lines.append(f"• {names[j]}: rho_c = {_f(cr)}, AVE = {_f(ave)} — " +
                     ("; ".join(issues) if issues else "reliability and convergent "
                      "validity established."))
    if q["htmt"]:
        bad = [f"{names[j]}–{names[i]} ({_f(v)})" for (i, j), v in q["htmt"].items()
               if v >= 0.90]
        lines.append("• Discriminant validity (HTMT < 0.90; < 0.85 for conceptually distinct "
                     "constructs): " + ("not established for " + ", ".join(bad) + "." if bad
                                       else "established."))
    form = [j for j in range(st.K) if st.modes[j] == "B"]
    if form:
        lines.append("Measurement model (formative):")
        high = [f"{st.indicators[k]} ({_f(q['outer_vif'][k])})" for j in form
                for k in st.blocks[j] if q["outer_vif"][k] >= 3]
        lines.append("• Collinearity: indicator VIF should be < 3 (critical ≥ 5). " +
                     ("High: " + ", ".join(high) + "." if high else "No critical values."))
        lines.append("• Assess the significance and relevance of the outer weights "
                     "(bootstrapping); keep a non-significant indicator if its outer loading "
                     "≥ 0.50 or it is theoretically essential.")
    lines.append("Structural model:")
    iv = q["inner_vif"]
    high = [f"{names[i]} -> {names[j]} ({_f(iv[i, j])})" for i, j in st.edges()
            if iv[i, j] >= 3]
    lines.append("• Collinearity: inner VIF < 3 is ideal (≥ 5 critical). " +
                 ("Check: " + ", ".join(high) + "." if high else "No collinearity issues."))

    def level(v):
        return "substantial" if v >= 0.75 else "moderate" if v >= 0.50 else \
            "weak" if v >= 0.25 else "very weak"
    for j in st.endogenous:
        lines.append(f"• R² of {names[j]} = {_f(r2[j])} ({level(r2[j])}; 0.75 / 0.50 / 0.25 "
                     "= substantial / moderate / weak).")
    f2 = q["f2"]

    def size(v):
        return "large" if v >= 0.35 else "medium" if v >= 0.15 else \
            "small" if v >= 0.02 else "no effect"
    lines.append("• Effect sizes f²: " + "; ".join(
        f"{names[i]} -> {names[j]} = {_f(f2[i, j])} ({size(f2[i, j])})"
        for i, j in st.edges()) + " (0.02 / 0.15 / 0.35 = small / medium / large).")
    if "q2" in q:
        parts = []
        for j, (sso, sse) in q["q2"].items():
            v = 1 - sse / sso
            rel_txt = "large" if v > 0.50 else "medium" if v > 0.25 else \
                "small" if v > 0 else "no"
            parts.append(f"{names[j]} = {_f(v)} ({rel_txt})")
        lines.append("• Q² (blindfolding) > 0 indicates predictive relevance (0 / 0.25 / "
                     "0.50 = small / medium / large): " + "; ".join(parts) + ".")
    if "predict" in q:
        better = sum(r["PLS_RMSE"] < r["LM_RMSE"] for r in q["predict"])
        tot = len(q["predict"])
        pos = all(r["Q2predict"] > 0 for r in q["predict"])
        if not pos:
            power = "lacks predictive power (some Q²predict ≤ 0)"
        elif better == tot:
            power = "has high predictive power"
        elif better > tot / 2:
            power = "has medium predictive power"
        elif better > 0:
            power = "has low predictive power"
        else:
            power = "lacks predictive power"
        lines.append(f"• PLSpredict: PLS-SEM RMSE is lower than the LM benchmark for {better} "
                     f"of {tot} indicators; the model {power} (Shmueli et al., 2019).")
    if boot is not None:
        sig = [r["label"] for r in boot.table["paths"] if r.get("P", 1) < boot.alpha]
        ns = [r["label"] for r in boot.table["paths"] if not r.get("P", 1) < boot.alpha]
        lines.append(f"• Significant paths (p < {boot.alpha:g}): " +
                     (", ".join(sig) if sig else "none") + ". Not significant: " +
                     (", ".join(ns) if ns else "none") + ".")
        med = []
        for r in boot.table.get("specific_indirect") or []:
            if r.get("P", 1) < boot.alpha:
                chain = r["label"].split(" -> ")
                a, b = names.index(chain[0]), names.index(chain[-1])
                direct = next((x for x in boot.table["paths"]
                               if x["label"] == f"{chain[0]} -> {chain[-1]}"), None)
                if direct is None:
                    kind = "indirect-only (full) mediation; no direct path in the model"
                elif direct.get("P", 1) >= boot.alpha:
                    kind = "indirect-only (full) mediation"
                elif np.sign(direct["O"]) == np.sign(r["O"]):
                    kind = "complementary (partial) mediation"
                else:
                    kind = "competitive (partial) mediation"
                del a, b
                med.append(f"{r['label']}: {kind}")
        if med:
            lines.append("• Mediation (Zhao et al., 2010; Hair et al., 2022): " +
                         "; ".join(med) + ".")
        for r, (i, j, _) in zip(boot.table["paths"], boot.labels["paths"]):
            if not st.is_interaction(i):
                continue
            sig = r.get("P", 1) < boot.alpha
            direction = "strengthens" if r["O"] > 0 else "weakens"
            lines.append(f"• Moderation {r['label']}: interaction = {_f(r['O'])}, p = "
                         f"{fmt_p(r.get('P'))} — " +
                         (f"significant; a higher moderator {direction} the effect." if sig
                          else "not significant (no moderating effect).") +
                         f" f² = {_f(q['f2'][i, j])} (0.005 / 0.01 / 0.025 = small / medium "
                         "/ large for interactions; Kenny, 2018).")
    if "fit" in q:
        srmr = q["fit"]["saturated"]["SRMR"]
        lines.append(f"• SRMR (saturated) = {_f(srmr)} " +
                     ("< 0.08 (good fit)." if srmr < 0.08 else "≥ 0.08."))
    sec.note("Interpretation (thresholds from Hair et al., A Primer on PLS-SEM, 2022)",
             "\n".join(lines))


def selftest_data(n: int = 200, seed: int = 7) -> pd.DataFrame:
    """Synthetic questionnaire-like data for X -> M -> Y with the direct path X -> Y."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    m = 0.5 * x + rng.normal(size=n) * 0.85
    y = 0.3 * x + 0.45 * m + rng.normal(size=n) * 0.75
    data = {}
    for name, lv, lam in (("X", x, (0.85, 0.80, 0.75)), ("M", m, (0.85, 0.80, 0.70)),
                          ("Y", y, (0.80, 0.80, 0.75, 0.70))):
        lv = (lv - lv.mean()) / lv.std()
        for i, lam_i in enumerate(lam):
            data[f"{name}{i + 1}"] = lam_i * lv + np.sqrt(1 - lam_i ** 2) * rng.normal(size=n)
    return pd.DataFrame(data)


SELFTEST_SYNTAX = """X =~ X1 + X2 + X3
M =~ M1 + M2 + M3
Y =~ Y1 + Y2 + Y3 + Y4
M ~ X
Y ~ X + M"""


def selftest() -> None:
    ds = Dataset(selftest_data())
    res = pls_sem(ds, SELFTEST_SYNTAX, subsamples=200, seed=3, predict=True, folds=5, reps=1,
                  save_scores=True)
    est = res.estimates
    assert res.quality["converged"]
    Y = np.column_stack([s[1].to_numpy() for s in res.saved.values()])
    assert np.allclose(Y.mean(axis=0), 0, atol=1e-8) and np.allclose(Y.std(axis=0), 1)
    corr = np.corrcoef(Y.T)
    assert np.allclose(corr, est["R"], atol=1e-8)
    assert 0.3 < est["paths"][0, 1] < 0.7, est["paths"]
    for row in res.boot.table["paths"]:
        assert row["lo"] <= row["O"] <= row["hi"]
    assert all(t.rows for t in res.section.items if getattr(t, "kind", "") == "table")
