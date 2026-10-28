"""Data > Identify Unusual Cases (SPSS DETECTANOMALY, Data Preparation add-on), following IBM's
"DETECTANOMALY Algorithms" chapter.

* **Modeling**: cases with a value beyond 1e150 (and, unless missing values are included, any
  missing value) are dropped; constant variables are removed. With missing values included,
  continuous missings become the grand mean, categorical missings a valid "missing" category,
  and a continuous variable "percentage of missing variables" is added. Peer groups come from
  TwoStep clustering with noise handling (WP9 :mod:`rsstatchem.core.stats.twostep`); noise cases
  join their closest non-noise group.
* **Scoring**: the variable deviation index of variable k is its contribution to the
  log-likelihood distance between the case s and its peer group h, with the adjustment
  Δ_k = σ̂²_k / m (m = 6 by default) that avoids log 0:
  continuous ½[−N_h log(Δ_k + σ̂²_hk) − log Δ_k + (N_h + 1) log(Δ_k + σ̂²_<h,s>k)],
  categorical −N_h Ê_hk + (N_h + 1) Ê_<h,s>k (Ê = category entropy). GDI = Σ VDI; the anomaly
  index is GDI / mean GDI of the peer group (IBM's rules for 0/0 and x/0), and the variable
  contribution measure is VDI / GDI.
* **Reasoning**: cases sorted by anomaly index; the top pct (or n) cases above the cut point
  are anomalies; their top-k variables by VDI, values and norms (peer mean or mode) are the
  reasons.
"""
from __future__ import annotations

import math

import numpy as np

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import fmt, fmt_int, value_text
from rsstatchem.core.stats.ml_common import MLError, categorical, raw_column
from rsstatchem.core.stats.twostep import twostep

AnomalyError = MLError
MISSING_PCT = "Missing Proportion"


def _entropy(counts: np.ndarray) -> np.ndarray:
    """Row-wise category entropy −Σ p log p (natural log) of count rows."""
    c = np.atleast_2d(np.asarray(counts, float))
    n = c.sum(1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(n > 0, c / n, 0.0)
        return -np.where(p > 0, p * np.log(p), 0.0).sum(1)


def group_stats(Xc: np.ndarray, Xcat: np.ndarray, ncat: list[int], sel: np.ndarray) -> dict:
    """Sufficient statistics of one peer group (IBM's stored norms)."""
    Nh = float(sel.sum())
    kc = Xc.shape[1]
    Xh = Xc[sel]
    st = {"N": Nh, "S": Xh.sum(0), "Q": (Xh ** 2).sum(0),
          "mean": Xh.mean(0) if kc else np.zeros(0), "var": Xh.var(0) if kc else np.zeros(0),
          "sd": Xh.std(0, ddof=1) if Nh > 1 and kc else np.zeros(kc),
          "counts": [np.bincount(Xcat[sel, j], minlength=L).astype(float)
                     for j, L in enumerate(ncat)]}
    return st


def vdi_to_group(Xc: np.ndarray, Xcat: np.ndarray, st: dict, delta: np.ndarray) -> np.ndarray:
    """Variable deviation indices d_k(h, s) of every case s against peer group h."""
    n = len(Xc) if Xc.size else len(Xcat)
    Nh = st["N"]
    parts = []
    if Xc.shape[1]:
        S = st["S"][None, :] + Xc          # merged cluster <h, s>
        Q = st["Q"][None, :] + Xc ** 2
        vm = np.maximum(Q / (Nh + 1) - (S / (Nh + 1)) ** 2, 0.0)
        parts.append(0.5 * (-Nh * np.log(delta + st["var"])[None, :] - np.log(delta)[None, :]
                            + (Nh + 1) * np.log(delta[None, :] + vm)))
    for j, cnt in enumerate(st["counts"]):
        merged = np.tile(cnt, (n, 1))
        merged[np.arange(n), Xcat[:, j]] += 1
        parts.append((-Nh * _entropy(cnt)[0] + (Nh + 1) * _entropy(merged))[:, None])
    return np.maximum(np.column_stack(parts), 0.0)


def adjustment_delta(Xc: np.ndarray, m: float = 6.0) -> np.ndarray:
    """Δ_k = σ̂²_k / m with the sample variance over the whole training sample."""
    if not Xc.shape[1]:
        return np.zeros(0)
    v = Xc.var(0, ddof=1) if len(Xc) > 1 else np.ones(Xc.shape[1])
    return np.where(v > 0, v, 1.0) / m


def deviation_indices(Xc: np.ndarray, Xcat: np.ndarray, ncat: list[int], peer: np.ndarray,
                      m: float = 6.0) -> tuple[np.ndarray, dict]:
    """VDI matrix (cases × variables, continuous first) against each case's own peer group,
    and the peer-group statistics."""
    n = len(peer)
    delta = adjustment_delta(Xc, m)
    stats = {}
    vdi = np.zeros((n, Xc.shape[1] + len(ncat)))
    for h in sorted(set(peer.tolist())):
        sel = peer == h
        st = group_stats(Xc, Xcat, ncat, sel)
        stats[h] = st
        vdi[sel] = vdi_to_group(Xc[sel], Xcat[sel], st, delta)
    return vdi, stats


def anomaly_indices(gdi: np.ndarray, peer: np.ndarray, K: int) -> tuple[np.ndarray, np.ndarray]:
    """Anomaly index per case and a flag for the degenerate-cluster warning."""
    groups = sorted(set(peer.tolist()))
    sizes = {h: (peer == h).sum() for h in groups}
    avg_size = len(peer) / len(groups)
    mean_gdi = {h: gdi[peer == h].mean() for h in groups}
    out = np.zeros(len(gdi))
    warn = np.zeros(len(gdi), bool)
    for i, (g, h) in enumerate(zip(gdi, peer)):
        den = mean_gdi[h]
        if den > 0:
            out[i] = g / den
        elif g == 0:
            out[i] = avg_size / sizes[h]
        else:
            others = [h2 for h2 in groups if h2 != h]
            w = sum(sizes[h2] * mean_gdi[h2] for h2 in others)
            tot = sum(sizes[h2] for h2 in others)
            out[i] = g / (w / tot) if tot and w > 0 else avg_size / sizes[h]
            warn[i] = True
    return out, warn


def detect_anomaly(ds: Dataset, categorical_vars: list[str], continuous_vars: list[str],
                   id_var: str | None = None, include_missing: bool = False,
                   missing_pct_var: bool = True, adjustment: float = 6.0,
                   peer_min: int = 1, peer_max: int = 15, pct: float | None = 5.0,
                   number: int | None = None, cutpoint: float | None = 2.0, reasons: int = 1,
                   norms: bool = True, summaries: bool = True,
                   save: dict | None = None) -> Section:
    """``save`` keys: index, peers, reasons (bool each) and ``replace`` (overwrite old)."""
    names = list(categorical_vars) + list(continuous_vars)
    if not names:
        raise AnomalyError("Select at least one analysis variable.")
    n_all = ds.n_cases
    cols_c = {v: raw_column(ds, v).to_numpy(float) for v in continuous_vars}
    cats = {v: categorical(ds, v) for v in categorical_vars}
    ok = np.ones(n_all, bool)
    for v, x in cols_c.items():
        ok &= ~(np.abs(np.nan_to_num(x)) > 1e150)
    miss = np.zeros(n_all)
    for v, x in cols_c.items():
        miss += ~np.isfinite(x)
    for v, c in cats.items():
        miss += c.codes < 0
    if not include_missing:
        ok &= miss == 0
    rows = np.where(ok)[0]
    if len(rows) < 3:
        raise AnomalyError("Too few valid cases.")
    # processed variables
    cont_names, Xc = [], []
    for v, x in cols_c.items():
        xv = x[rows].copy()
        fin = np.isfinite(xv)
        if fin.sum() == 0 or np.nanmax(xv) == np.nanmin(xv):
            continue
        xv[~fin] = np.nanmean(xv)
        cont_names.append(v)
        Xc.append(xv)
    if include_missing and missing_pct_var:
        mp = miss[rows] / len(names)
        if mp.max() > mp.min():
            cont_names.append(MISSING_PCT)
            Xc.append(mp)
    cat_names, Xcat, ncat, cat_levels = [], [], [], []
    for v, c in cats.items():
        codes = c.codes[rows].copy()
        levels = list(c.labels)
        if (codes < 0).any():
            codes[codes < 0] = len(levels)
            levels.append("Missing")
        present = np.unique(codes)
        if len(present) < 2:
            continue
        remap = {old: k for k, old in enumerate(present)}
        Xcat.append(np.array([remap[cc] for cc in codes]))
        cat_levels.append([levels[p] for p in present])
        ncat.append(len(present))
        cat_names.append(v)
    if not cont_names and not cat_names:
        raise AnomalyError("All analysis variables are constant.")
    Xc = np.column_stack(Xc) if cont_names else np.zeros((len(rows), 0))
    Xcat = np.column_stack(Xcat).astype(int) if cat_names else np.zeros((len(rows), 0), int)
    sd = Xc.std(0, ddof=1) if cont_names else np.zeros(0)
    Z = (Xc - Xc.mean(0)) / np.where(sd > 0, sd, 1) if cont_names else Xc
    peer_max = max(1, int(peer_max))
    peer_min = max(1, min(int(peer_min), peer_max))
    fixed = peer_max if peer_min == peer_max else None
    res = twostep(Z, Xcat, ncat, "ll", fixed, peer_max, "bic", outliers=True)
    if fixed is None and res["k"] < peer_min:
        res = twostep(Z, Xcat, ncat, "ll", peer_min, peer_max, "bic", outliers=True)
    peer = res["labels"].copy()
    good = sorted(set(peer[peer > 0].tolist()))
    if not good:
        peer[:] = 1
    elif (peer == -1).any():   # noise -> closest non-noise peer group (log-likelihood)
        noise = np.where(peer == -1)[0]
        delta = adjustment_delta(Xc, adjustment)
        D = np.column_stack([
            vdi_to_group(Xc[noise], Xcat[noise], group_stats(Xc, Xcat, ncat, peer == h),
                         delta).sum(1) for h in good])
        peer[noise] = np.array(good)[np.argmin(D, axis=1)]
    remap = {h: k + 1 for k, h in enumerate(sorted(set(peer.tolist())))}
    peer = np.array([remap[h] for h in peer])
    vdi, pstats = deviation_indices(Xc, Xcat, ncat, peer, adjustment)
    gdi = vdi.sum(1)
    K = vdi.shape[1]
    aidx, warn = anomaly_indices(gdi, peer, K)
    with np.errstate(divide="ignore", invalid="ignore"):
        vcm = np.where(gdi[:, None] > 0, vdi / gdi[:, None], 1.0 / K)
    var_names = cont_names + cat_names
    # ---------------------------------------------------------------- anomaly list
    order = np.argsort(-aidx, kind="mergesort")
    if number:
        top = int(number)
    else:
        top = int(math.floor((pct or 5.0) / 100 * len(rows) + 1e-9))
        top = max(top, 1)
    anomalies = [i for i in order[:top] if cutpoint is None or aidx[i] > cutpoint]
    nreason = max(1, min(int(reasons), K))
    reason_idx = np.argsort(-vdi, axis=1, kind="mergesort")[:, :nreason]

    def value_of(i, k):
        if k < len(cont_names):
            nm = cont_names[k]
            if nm == MISSING_PCT:
                return fmt(Xc[i, k], 3)
            raw = cols_c[nm][rows[i]]
            return fmt(raw, 2) if np.isfinite(raw) else "Missing"
        j = k - len(cont_names)
        return str(cat_levels[j][Xcat[i, j]])

    def norm_of(h, k):
        st = pstats[h]
        if k < len(cont_names):
            return fmt(st["mean"][k], 2)
        j = k - len(cont_names)
        return str(cat_levels[j][int(np.argmax(st["counts"][j]))])

    section = Section("Identify Unusual Cases")
    section.log("DETECTANOMALY\n"
                + (f"  /VARIABLES CATEGORICAL={' '.join(categorical_vars)}"
                   if categorical_vars else "  /VARIABLES")
                + (f" SCALE={' '.join(continuous_vars)}" if continuous_vars else "")
                + (f" ID={id_var}" if id_var else "")
                + (f"\n  /CRITERIA PCTANOMALOUSCASES={pct or 5:g}" if number is None
                   else f"\n  /CRITERIA NUMANOMALOUSCASES={number}")
                + (f" ANOMALYCUTPOINT={cutpoint:g}" if cutpoint is not None else "")
                + f" MINNUMPEERS={peer_min} MAXNUMPEERS={peer_max} NUMREASONS={nreason}"
                + f"\n  /HANDLEMISSING APPLY={'YES' if include_missing else 'NO'}"
                + f" CREATEMISPROPVAR={'YES' if missing_pct_var else 'NO'}.")
    excl = n_all - len(rows)
    section.table(Table("Case Processing Summary", ["", "N", "% of Combined", "% of Total"], [
        ["Peer ID " + str(h), fmt_int(int((peer == h).sum())),
         f"{100 * (peer == h).sum() / len(rows):.1f}%", f"{100 * (peer == h).sum() / n_all:.1f}%"]
        for h in sorted(pstats)] + [
        ["Combined", fmt_int(len(rows)), "100.0%", f"{100 * len(rows) / n_all:.1f}%"],
        ["Excluded Cases", fmt_int(excl), "", f"{100 * excl / n_all:.1f}%"],
        ["Total", fmt_int(n_all), "", "100.0%"]]))
    idv = raw_column(ds, id_var) if id_var else None

    def case_cells(i):
        r = [str(rows[i] + 1)]
        if idv is not None:
            r.append(value_text(ds.var(id_var), idv.iloc[rows[i]]))
        return r

    head = ["Case"] + ([ds.var(id_var).title] if id_var else [])
    if anomalies:
        section.table(Table("Anomaly Case Index List", head + ["Anomaly Index"],
                            [case_cells(i) + [fmt(aidx[i], 3)] for i in anomalies]))
        section.table(Table("Anomaly Case Peer ID List", head + ["Peer ID", "Peer Size",
                                                                 "Peer Size Percent"],
                            [case_cells(i) + [str(peer[i]),
                                              fmt_int(int(pstats[peer[i]]["N"])),
                                              f"{100 * pstats[peer[i]]['N'] / len(rows):.1f}%"]
                             for i in anomalies]))
        for r in range(nreason):
            rows_r = []
            for i in anomalies:
                k = reason_idx[i, r]
                rows_r.append(case_cells(i) + [var_names[k], fmt(vcm[i, k], 3), value_of(i, k),
                                               norm_of(peer[i], k)])
            section.table(Table(f"Anomaly Case Reason List (Reason {r + 1})",
                                head + ["Reason Variable", "Variable Impact", "Variable Value",
                                        "Variable Norm"], rows_r))
    else:
        section.note("", "No case meets the anomaly criteria.")
    if norms:
        if cont_names:
            cols = [""] + [(f"Peer ID {h}", s) for h in sorted(pstats)
                           for s in ("Mean", "Std. Deviation")] + \
                [("Combined", "Mean"), ("Combined", "Std. Deviation")]
            rows_n = []
            for k, nm in enumerate(cont_names):
                r = [nm]
                for h in sorted(pstats):
                    r += [fmt(pstats[h]["mean"][k], 4), fmt(pstats[h]["sd"][k], 4)]
                r += [fmt(Xc[:, k].mean(), 4), fmt(Xc[:, k].std(ddof=1), 4)]
                rows_n.append(r)
            section.table(Table("Scale Variable Norms", cols, rows_n,
                                footnotes=["The mean and standard deviation are computed from "
                                           "the training sample (missing values replaced by "
                                           "the grand mean when they are included)."]))
        if cat_names:
            cols = [""] + [(f"Peer ID {h}", s) for h in sorted(pstats)
                           for s in ("Mode", "Frequency", "Frequency Percent")]
            rows_n = []
            for j, nm in enumerate(cat_names):
                r = [nm]
                for h in sorted(pstats):
                    cnt = pstats[h]["counts"][j]
                    mo = int(np.argmax(cnt))
                    r += [str(cat_levels[j][mo]), fmt_int(int(cnt[mo])),
                          f"{100 * cnt[mo] / cnt.sum():.1f}%"]
                rows_n.append(r)
            section.table(Table("Categorical Variable Norms", cols, rows_n))
    if summaries and anomalies:
        a = aidx[anomalies]
        section.table(Table("Anomaly Index Summary", ["", "N", "Minimum", "Maximum", "Mean",
                                                      "Std. Deviation"],
                            [["Anomaly Index", fmt_int(len(a)), fmt(a.min(), 3), fmt(a.max(), 3),
                              fmt(a.mean(), 3), fmt(a.std(ddof=1), 3) if len(a) > 1 else ""]]))
        for r in range(nreason):
            rows_s = []
            for k, nm in enumerate(var_names):
                hit = [i for i in anomalies if reason_idx[i, r] == k]
                imp = vcm[hit, k] if hit else np.zeros(0)
                rows_s.append([nm, fmt_int(len(hit)), f"{100 * len(hit) / len(anomalies):.1f}%",
                               fmt(imp.min(), 3) if hit else "", fmt(imp.max(), 3) if hit else "",
                               fmt(imp.mean(), 3) if hit else "",
                               fmt(imp.std(ddof=1), 3) if len(hit) > 1 else ""])
            section.table(Table(f"Reason Summary (Reason {r + 1})",
                                ["", ("Occurrence as Reason", "Frequency"),
                                 ("Occurrence as Reason", "Percent"),
                                 ("Variable Impact", "Minimum"), ("Variable Impact", "Maximum"),
                                 ("Variable Impact", "Mean"),
                                 ("Variable Impact", "Std. Deviation")], rows_s))
    if warn.any():
        section.note("", f"{int(warn.sum())} case(s) were assigned to a peer group made up of "
                         "identical cases.", style="warning")
    if save:
        _save(ds, save, rows, aidx, peer, pstats, reason_idx, vcm, var_names, value_of,
              norm_of, nreason, len(rows))
    section.result = {"index": aidx, "peer": peer, "rows": rows, "vdi": vdi,  # type: ignore
                      "anomalies": [int(rows[i]) for i in anomalies], "names": var_names}
    return section


def _save(ds, opts, rows, aidx, peer, pstats, reason_idx, vcm, var_names, value_of, norm_of,
          nreason, n):
    def put(name, values, label, string=False, decimals=3):
        if ds.has_name(name):
            if not opts.get("replace"):
                raise AnomalyError(f"A variable named {name} already exists (tick Replace "
                                   "existing variables).")
            ds.delete_variables([ds.index_of(name)])
        col = np.full(ds.n_cases, "" if string else np.nan, dtype=object if string else float)
        col[rows] = values
        var = ds.add_variable(name, vtype="string" if string else "numeric", values=col)
        var.label = label
        if not string:
            var.decimals = decimals
        return var

    if opts.get("index"):
        put("AnomalyIndex", aidx, "Anomaly Index")
    if opts.get("peers"):
        put("PeerId", peer.astype(float), "Peer Group ID", decimals=0)
        put("PeerSize", np.array([pstats[h]["N"] for h in peer]), "Peer Group Size", decimals=0)
        put("PeerPctSize", np.array([100 * pstats[h]["N"] / n for h in peer]),
            "Peer Group Size in Percentage", decimals=2)
    if opts.get("reasons"):
        for r in range(nreason):
            ks = reason_idx[:, r]
            put(f"ReasonVar_{r + 1}", np.array([var_names[k] for k in ks], dtype=object),
                f"Reason Variable {r + 1}", string=True)
            put(f"ReasonMeasure_{r + 1}", vcm[np.arange(len(ks)), ks],
                f"Variable Impact Measure {r + 1}")
            put(f"ReasonValue_{r + 1}", np.array([value_of(i, k) for i, k in enumerate(ks)],
                                                 dtype=object), f"Variable Value {r + 1}",
                string=True)
            put(f"ReasonNorm_{r + 1}", np.array([norm_of(peer[i], k) for i, k in enumerate(ks)],
                                                dtype=object), f"Variable Norm {r + 1}",
                string=True)


def selftest() -> None:
    r = np.random.default_rng(5)
    Xc = np.r_[r.normal(0, 1, (60, 2)), [[8.0, 8.0]]]
    peer = np.ones(61, int)
    vdi, _ = deviation_indices(Xc, np.zeros((61, 0), int), [], peer)
    a, _ = anomaly_indices(vdi.sum(1), peer, 2)
    assert int(np.argmax(a)) == 60 and a[60] > 2
