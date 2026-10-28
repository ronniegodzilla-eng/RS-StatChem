"""Analyze > Direct Marketing (SPSS Direct Marketing add-on): RFM Analysis, Cluster Analysis,
Prospect Profiles, Postal Code Response Rates, Propensity to Purchase and Control Package Test.

* **RFM Analysis** from transaction data (customer ID, date, amount; monetary = total, mean,
  median or maximum per customer) or customer data (recency as a date or days, frequency,
  monetary). Each of R, F and M is cut into k bins of (nearly) equal size by rank (tied values
  share a bin, or ties are broken at random with a seed); the most recent customers and the
  largest frequencies / amounts get the highest score. Nested binning cuts F within each R bin
  and M within each R×F cell; independent binning cuts each over all customers. RFM score =
  100 R + 10 F + M.
* **Cluster Analysis**: TwoStep (WP9) on the chosen fields, categorical by measurement level.
* **Prospect Profiles**: a CHAID tree (WP9) of the positive response on the profile fields;
  each terminal node is a profile with its size, responses and response rate, sorted by rate,
  with a minimum response-rate threshold.
* **Postal Code Response Rates**: response rate by postal code (optionally its first n
  characters), sorted, with deciles and an optional profit / ROI calculation (cost per mailing,
  revenue per response); the rates can be written to a new dataset.
* **Propensity to Purchase**: binary logistic regression of the positive response, validated
  on a random test partition (seed); AUC, classification table, cumulative gains; the predicted
  probability is saved as the propensity score for every case.
* **Control Package Test**: each test package against the control package — two-proportion
  z test of the response rates (or Welch t test of the spending amounts), Bonferroni adjusted
  for the number of comparisons — plus the overall chi-square test across packages.
"""
from __future__ import annotations

import datetime as _dt
import math

import numpy as np
import pandas as pd
from scipy import stats

from rsstatchem.core.dataset import NUMERIC, Dataset
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import fmt, fmt_int, fmt_p, value_text
from rsstatchem.core.stats.ml_common import (MLError, MLResult, measure_of, partition,
                                             raw_column, rng, sort_levels)
from rsstatchem.core.stats.roc import figure, png, style

MarketingError = MLError


# ============================================================================ RFM
def ntile(values: np.ndarray, k: int, ties: str = "same", seed: int | None = None,
          gen=None) -> np.ndarray:
    """Bins 1..k of equal size by rank (higher value -> higher bin). SPSS RANK /NTILES:
    bin = floor(rank · k / (n + 1)) + 1 with mean ranks for ties."""
    v = np.asarray(values, float)
    n = len(v)
    if n == 0:
        return np.zeros(0)
    if ties == "random":
        g = gen if gen is not None else rng(seed)
        order = np.lexsort((g.random(n), v))
        r = np.empty(n)
        r[order] = np.arange(1, n + 1)
    else:
        r = stats.rankdata(v, method="average")
    return np.floor(r * k / (n + 1)).astype(int) + 1.0


def rfm_scores(recency_days: np.ndarray, frequency: np.ndarray, monetary: np.ndarray,
               bins: tuple[int, int, int] = (5, 5, 5), nested: bool = True,
               ties: str = "same", seed: int | None = None) -> np.ndarray:
    """(n, 4) array of R, F, M and RFM scores."""
    g = rng(seed)
    kr, kf, km = bins
    R = ntile(-np.asarray(recency_days, float), kr, ties, gen=g)
    if nested:
        F = np.zeros(len(R))
        M = np.zeros(len(R))
        for r in np.unique(R):
            sel = R == r
            F[sel] = ntile(np.asarray(frequency, float)[sel], kf, ties, gen=g)
            for f in np.unique(F[sel]):
                s2 = sel & (F == f)
                M[s2] = ntile(np.asarray(monetary, float)[s2], km, ties, gen=g)
    else:
        F = ntile(frequency, kf, ties, gen=g)
        M = ntile(monetary, km, ties, gen=g)
    return np.column_stack([R, F, M, 100 * R + 10 * F + M])


def _days(values, reference: _dt.date) -> np.ndarray:
    s = pd.to_datetime(pd.Series(values), errors="coerce")
    ref = pd.Timestamp(reference)
    return (ref - s).dt.days.to_numpy(float)


def rfm_transactions(ds: Dataset, customer: str, date: str, amount: str,
                     summary: str = "total", reference: _dt.date | None = None,
                     **kw) -> tuple[Section, Dataset]:
    ref = reference or _dt.date.today()
    cid = raw_column(ds, customer)
    dt = pd.to_datetime(ds.df[date], errors="coerce") if ds.var(date).type != NUMERIC else \
        pd.to_datetime(pd.to_numeric(ds.df[date], errors="coerce"), unit="D",
                       origin="1899-12-30", errors="coerce")
    amt = pd.to_numeric(raw_column(ds, amount), errors="coerce")
    ok = cid.notna() & dt.notna() & amt.notna()
    df = pd.DataFrame({"id": cid[ok].to_numpy(object), "date": dt[ok].to_numpy(),
                       "amount": amt[ok].to_numpy(float)})
    agg = {"total": "sum", "mean": "mean", "median": "median", "maximum": "max"}[summary]
    grp = df.groupby("id", sort=True)
    out = pd.DataFrame({"last": grp["date"].max(), "count": grp["date"].size(),
                        "amount": grp["amount"].agg(agg)})
    out.index.name = None
    rec = _days(out["last"], ref)
    sc = rfm_scores(rec, out["count"].to_numpy(float), out["amount"].to_numpy(float), **kw)
    new = Dataset(pd.DataFrame({
        customer: list(out.index), "Date_Most_Recent": pd.to_datetime(out["last"]).to_numpy(),
        "Transaction_Count": out["count"].to_numpy(float),
        "Amount": out["amount"].to_numpy(float), "Recency_Score": sc[:, 0],
        "Frequency_Score": sc[:, 1], "Monetary_Score": sc[:, 2], "RFM_Score": sc[:, 3]}),
        name="RFM")
    for nm, lab in (("Date_Most_Recent", "Most recent transaction date"),
                    ("Transaction_Count", "Number of transactions"),
                    ("Amount", f"{summary.capitalize()} transaction amount"),
                    ("Recency_Score", "Recency score"), ("Frequency_Score", "Frequency score"),
                    ("Monetary_Score", "Monetary score"), ("RFM_Score", "RFM score")):
        if new.has_name(nm):
            new.var(nm).label = lab
            if nm.endswith("Score") or nm == "Transaction_Count":
                new.var(nm).decimals = 0
    section = _rfm_output(rec, out["count"].to_numpy(float), out["amount"].to_numpy(float), sc,
                          kw, f"RFM ANALYSIS /TRANSACTIONS ID={customer} DATE={date} "
                              f"AMOUNT={amount} SUMMARY={summary.upper()}",
                          len(out), ds.n_cases - len(df), transactions=len(df))
    return section, new


def rfm_customers(ds: Dataset, recency: str, frequency: str, monetary: str,
                  reference: _dt.date | None = None, save: bool = True,
                  **kw) -> Section:
    if ds.var(recency).type == NUMERIC and measure_of(ds, recency) == "scale" and \
            not _looks_like_dates(ds, recency):
        rec = pd.to_numeric(raw_column(ds, recency), errors="coerce").to_numpy(float)
    else:
        rec = _days(ds.df[recency], reference or _dt.date.today())
    fr = pd.to_numeric(raw_column(ds, frequency), errors="coerce").to_numpy(float)
    mo = pd.to_numeric(raw_column(ds, monetary), errors="coerce").to_numpy(float)
    ok = np.isfinite(rec) & np.isfinite(fr) & np.isfinite(mo)
    rows = np.where(ok)[0]
    sc = rfm_scores(rec[ok], fr[ok], mo[ok], **kw)
    section = _rfm_output(rec[ok], fr[ok], mo[ok], sc, kw,
                          f"RFM ANALYSIS /CUSTOMERS RECENCY={recency} FREQUENCY={frequency} "
                          f"MONETARY={monetary}", len(rows), ds.n_cases - len(rows))
    if save:
        names = []
        for j, nm in enumerate(("Recency_Score", "Frequency_Score", "Monetary_Score",
                                "RFM_Score")):
            base, k = nm, 1
            while ds.has_name(nm):
                k += 1
                nm = f"{base}_{k}"
            col = np.full(ds.n_cases, np.nan)
            col[rows] = sc[:, j]
            v = ds.add_variable(nm, values=col)
            v.label, v.decimals = base.replace("_", " ").replace("Score", "score"), 0
            names.append(nm)
        section.note("", "Variables created: " + ", ".join(names))
    section.result = sc  # type: ignore[attr-defined]
    return section


def _looks_like_dates(ds: Dataset, name: str) -> bool:
    return ds.var(name).type == "date"


def _rfm_output(rec, fr, mo, sc, kw, syntax, n, excluded, transactions=None) -> Section:
    bins = kw.get("bins", (5, 5, 5))
    section = Section("RFM Analysis")
    section.log(syntax + f"\n  /BINS RECENCY={bins[0]} FREQUENCY={bins[1]} MONETARY={bins[2]}"
                f" METHOD={'NESTED' if kw.get('nested', True) else 'INDEPENDENT'}"
                f" TIES={'SAME' if kw.get('ties', 'same') == 'same' else 'RANDOM'}.")
    rows_cs = ([["Valid transactions", fmt_int(transactions)]] if transactions is not None
               else []) + [["Customers scored", fmt_int(n)],
                           ["Excluded " + ("transactions" if transactions is not None
                                           else "cases") + " (missing values)",
                            fmt_int(excluded)]]
    section.table(Table("Case Summary", ["", "N"], rows_cs))
    rows = []
    for j, (name, vals) in enumerate((("Recency (days)", rec), ("Frequency", fr),
                                      ("Monetary", mo))):
        for b in range(1, bins[j] + 1):
            sel = sc[:, j] == b
            rows.append([name if b == 1 else "", str(b), fmt_int(int(sel.sum())),
                         fmt(vals[sel].min(), 2) if sel.any() else "",
                         fmt(vals[sel].max(), 2) if sel.any() else "",
                         fmt(vals[sel].mean(), 2) if sel.any() else ""])
    section.table(Table("RFM Bins", ["", "Score", "N", "Minimum", "Maximum", "Mean"], rows,
                        row_header_cols=2))
    kr, kf = bins[0], bins[1]
    H = np.full((kf, kr), np.nan)
    for r in range(1, kr + 1):
        for f in range(1, kf + 1):
            sel = (sc[:, 0] == r) & (sc[:, 1] == f)
            if sel.any():
                H[f - 1, r - 1] = mo[sel].mean()
    fig = figure(5.4, 4.2)
    ax = fig.add_subplot(111)
    im = ax.imshow(H, origin="lower", cmap="YlOrRd", aspect="auto",
                   extent=(0.5, kr + 0.5, 0.5, kf + 0.5))
    fig.colorbar(im, ax=ax, label="Mean monetary")
    style(ax, "Heat Map: Mean Monetary by Recency and Frequency", "Recency score",
          "Frequency score")
    section.image("Heat Map", png(fig), width=460)
    fig = figure(8.4, 2.8)
    for j, (name, vals) in enumerate((("Recency (days)", rec), ("Frequency", fr),
                                      ("Monetary", mo))):
        ax = fig.add_subplot(1, 3, j + 1)
        ax.hist(vals, bins="auto", color="#9ecae1", edgecolor="white")
        style(ax, name, "", "Count" if j == 0 else "")
    section.image("Histograms", png(fig), width=640)
    counts = pd.Series(sc[:, 3]).value_counts().sort_index()
    fig = figure(6.4, 3.2)
    ax = fig.add_subplot(111)
    ax.bar(range(len(counts)), counts.to_numpy(), color="#1f5fa8")
    step = max(1, len(counts) // 15)
    ax.set_xticks(range(0, len(counts), step),
                  [str(int(v)) for v in counts.index[::step]], rotation=90, fontsize=7)
    style(ax, "Bin Counts", "RFM score", "Customers")
    section.image("Bin Counts", png(fig), width=560)
    return section


# ============================================================================ cluster
def dm_cluster(ds: Dataset, fields: list[str], max_k: int = 15, fixed_k: int | None = None,
               save: bool = False) -> MLResult:
    from rsstatchem.core.stats.twostep import twostep_procedure

    cats = [f for f in fields if measure_of(ds, f) != "scale"]
    conts = [f for f in fields if f not in cats]
    res = twostep_procedure(ds, cats, conts, auto=fixed_k is None, max_k=max_k,
                            fixed_k=fixed_k or 3, save=save)
    res.section.title = "Direct Marketing: Cluster Analysis"
    if "TSC" in res.saved:
        res.saved = {"ClusterGroup": ("Cluster group (Direct Marketing)", res.saved["TSC"][1])}
    return res


# ============================================================================ profiles
def _positive(ds: Dataset, response: str, positive: list) -> pd.Series:
    s = raw_column(ds, response)
    return s.isin(positive).astype(float).where(s.notna())


def prospect_profiles(ds: Dataset, response: str, positive: list, fields: list[str],
                      min_group: int = 100, min_rate: float | None = None,
                      max_depth: int = 3) -> Section:
    from rsstatchem.core.stats import trees

    if not fields:
        raise MarketingError("Select at least one profile field.")
    y = _positive(ds, response, positive)
    ok = y.notna().to_numpy()
    rows = np.where(ok)[0]
    if len(rows) < 2 * min_group:
        min_group = max(2, len(rows) // 4)
    s = {"min_parent": int(2 * min_group), "min_child": int(min_group), "max_depth": max_depth}
    tmp = Dataset(ds.df.copy(), [v for v in ds.variables])
    tmp.add_variable("_resp_", values=y.to_numpy())
    tmp.var("_resp_").measure = "nominal"
    yv, ylabels, preds, yord = trees._prepare(tmp, "_resp_", fields, "chaid", s, rows,
                                              "nominal")
    model = trees.fit_tree(yv, "categorical", ylabels, preds, "chaid", s, dep="_resp_")
    if len(ylabels) < 2:
        raise MarketingError("The response field needs both positive and other responses.")
    pos_code = len(ylabels) - 1          # levels sorted: 0 (other), 1 (positive)
    overall = float((yv == pos_code).mean())
    term = model.terminals()
    prof = []
    for nd in term:
        n = len(nd.idx)
        r = int((yv[nd.idx] == pos_code).sum())
        path = []
        cur = nd
        while cur.parent is not None:
            par = model.nodes[cur.parent]
            path.append(f"{model.predictors[par.split.var].name} = {cur.branch}")
            cur = par
        prof.append((r / n if n else 0.0, n, r, " AND ".join(reversed(path)) or "All cases"))
    prof.sort(key=lambda t: -t[0])
    section = Section("Prospect Profiles")
    section.log(f"PROSPECT PROFILES RESPONSE={response}({', '.join(map(str, positive))}) "
                f"FIELDS={' '.join(fields)}\n  /CRITERIA MINGROUP={min_group}"
                + (f" MINRATE={min_rate:g}" if min_rate is not None else "") + ".")
    rows_t = []
    cum_n = cum_r = 0
    for rate, n, r, txt in prof:
        if min_rate is not None and rate * 100 < min_rate:
            continue
        cum_n += n
        cum_r += r
        rows_t.append([txt, fmt_int(n), fmt_int(r), f"{100 * rate:.1f}%",
                       f"{100 * cum_r / cum_n:.1f}%", fmt(rate / overall * 100 if overall
                                                          else np.nan, 0)])
    section.table(Table("Profiles", ["Profile", "Profile Size", "Responses", "Response Rate",
                                     "Cumulative Response Rate", "Index (%)"], rows_t,
                        footnotes=[f"Overall response rate {100 * overall:.1f}% (N = "
                                   f"{len(rows)}). Profiles come from a CHAID tree with at "
                                   f"least {min_group} cases per profile."
                                   + (f" Only profiles with a response rate of at least "
                                      f"{min_rate:g}% are shown." if min_rate is not None
                                      else "")]))
    if rows_t:
        fig = figure(6.0, 3.6)
        ax = fig.add_subplot(111)
        rates = [100 * p[0] for p in prof if min_rate is None or p[0] * 100 >= min_rate]
        ax.bar(range(1, len(rates) + 1), rates, color="#1f5fa8")
        ax.axhline(100 * overall, color="#c0392b", ls="--", label="Overall")
        ax.legend(fontsize=8)
        style(ax, "Response Rate by Profile", "Profile (sorted)", "Response rate (%)")
        section.image("Response Rate by Profile", png(fig), width=480)
    section.result = prof  # type: ignore[attr-defined]
    return section


# ============================================================================ postal codes
def postal_code_rates(ds: Dataset, response: str, positive: list, postal: str,
                      digits: int | None = None, min_rate: float | None = None,
                      cost: float | None = None, revenue: float | None = None,
                      ) -> tuple[Section, Dataset]:
    y = _positive(ds, response, positive)
    pc = raw_column(ds, postal)
    ok = y.notna() & pc.notna()
    codes = pc[ok].map(lambda v: (str(int(v)) if isinstance(v, float) and v.is_integer()
                                  else str(v)).strip())
    if digits:
        codes = codes.str[:int(digits)]
    df = pd.DataFrame({"code": codes.to_numpy(), "y": y[ok].to_numpy(float)})
    g = df.groupby("code")["y"].agg(["size", "sum"])
    g["rate"] = g["sum"] / g["size"]
    g = g.sort_values(["rate", "size"], ascending=[False, False])
    section = Section("Postal Code Response Rates")
    section.log(f"POSTAL CODE RESPONSE RATES RESPONSE={response} POSTALCODE={postal}"
                + (f" DIGITS={digits}" if digits else "") + ".")
    overall = df["y"].mean()
    rows = []
    cum_n = cum_r = 0.0
    for code, r in g.iterrows():
        if min_rate is not None and r["rate"] * 100 < min_rate:
            continue
        cum_n += r["size"]
        cum_r += r["sum"]
        row = [code, fmt_int(r["size"]), fmt_int(r["sum"]), f"{100 * r['rate']:.1f}%",
               f"{100 * cum_r / cum_n:.1f}%"]
        if cost is not None and revenue is not None:
            profit = r["sum"] * revenue - r["size"] * cost
            row += [fmt(profit, 2), f"{100 * profit / (r['size'] * cost):.1f}%"
                    if cost else ""]
        rows.append(row)
    cols = ["Postal Code", "Contacts", "Responses", "Response Rate", "Cumulative Rate"]
    if cost is not None and revenue is not None:
        cols += ["Profit", "ROI"]
    section.table(Table("Response Rates by Postal Code", cols, rows,
                        footnotes=[f"Overall response rate {100 * overall:.1f}% over "
                                   f"{len(df)} contacts and {len(g)} postal codes."]))
    # deciles of postal codes by response rate
    k = min(10, len(g))
    if k >= 2:
        dec = np.array_split(np.arange(len(g)), k)
        drows, rates = [], []
        for i, idx in enumerate(dec, 1):
            sub = g.iloc[idx]
            rt = sub["sum"].sum() / sub["size"].sum()
            rates.append(100 * rt)
            drows.append([str(i), fmt_int(len(sub)), fmt_int(sub["size"].sum()),
                          fmt_int(sub["sum"].sum()), f"{100 * rt:.1f}%"])
        section.table(Table("Response Rate by Decile", ["Decile", "Postal Codes",
                                                        "Contacts", "Responses",
                                                        "Response Rate"], drows))
        fig = figure(5.6, 3.4)
        ax = fig.add_subplot(111)
        ax.bar(range(1, k + 1), rates, color="#1f5fa8")
        ax.axhline(100 * overall, color="#c0392b", ls="--", label="Overall")
        ax.legend(fontsize=8)
        style(ax, "Response Rate by Decile", "Decile", "Response rate (%)")
        section.image("Response Rate by Decile", png(fig), width=460)
    new = Dataset(pd.DataFrame({"PostalCode": list(g.index),
                                "ResponseRate": g["rate"].to_numpy(float) * 100,
                                "Responses": g["sum"].to_numpy(float),
                                "Contacts": g["size"].to_numpy(float)}), name="PostalRates")
    new.var("ResponseRate").label = "Response rate (%)"
    return section, new


# ============================================================================ propensity
def _logit_fit(X: np.ndarray, y: np.ndarray, iters: int = 50) -> np.ndarray:
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = np.clip(X @ beta, -30, 30)
        p = 1 / (1 + np.exp(-eta))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + 1e-10 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p))
        beta += step
        if np.abs(step).max() < 1e-9:
            break
    return beta


def propensity_to_purchase(ds: Dataset, response: str, positive: list, predictors: list[str],
                           validate: bool = True, train_pct: float = 50.0,
                           seed: int | None = None, save: bool = True,
                           cut: float = 0.5) -> MLResult:
    if not predictors:
        raise MarketingError("Select at least one predictor.")
    y = _positive(ds, response, positive)
    cols, names = [], []
    for p in predictors:
        if measure_of(ds, p) == "scale" and ds.var(p).type == NUMERIC:
            cols.append(pd.to_numeric(raw_column(ds, p), errors="coerce").to_numpy(float))
            names.append(p)
        else:
            s = raw_column(ds, p)
            lv = sort_levels(s.dropna().tolist())
            for v in lv[:-1]:
                cols.append(np.where(s.isna(), np.nan, (s == v).astype(float)))
                names.append(f"{p}={value_text(ds.var(p), v)}")
    X = np.column_stack([np.ones(ds.n_cases)] + cols)
    ok = np.isfinite(X).all(1) & y.notna().to_numpy()
    rows = np.where(ok)[0]
    yy = y.to_numpy(float)[rows]
    if yy.min() == yy.max():
        raise MarketingError("The response field has only one outcome.")
    Xr = X[rows]
    sample = partition(len(rows), train_pct, 100 - train_pct, 0, seed) if validate \
        else np.zeros(len(rows), int)
    tr = sample == 0
    beta = _logit_fit(Xr[tr], yy[tr])
    prob = 1 / (1 + np.exp(-np.clip(Xr @ beta, -30, 30)))
    section = Section("Propensity to Purchase")
    section.log(f"PROPENSITY TO PURCHASE RESPONSE={response}({', '.join(map(str, positive))})"
                f" PREDICTORS={' '.join(predictors)}\n  /VALIDATE "
                + (f"TRAINING={train_pct:g} SEED={seed}" if validate else "NONE") + ".")
    te = sample == 1 if validate and (sample == 1).any() else tr
    from rsstatchem.core.stats.roc import auc_details

    auc = auc_details(prob[te][yy[te] == 1], prob[te][yy[te] == 0])["auc"]
    section.table(Table("Overall Model Quality", ["", "Value"], [
        ["Area under the ROC curve" + (" (testing sample)" if validate else ""), fmt(auc, 3)],
        ["Training cases", fmt_int(int(tr.sum()))],
        ["Testing cases", fmt_int(int((sample == 1).sum()))],
        ["Model quality", "Good" if auc >= 0.8 else "Fair" if auc >= 0.7 else "Poor"]]))
    section.table(Table("Variables in the Equation", ["", "B", "Exp(B)"],
                        [["Constant", fmt(beta[0], 3), fmt(math.exp(beta[0]), 3)]]
                        + [[nm, fmt(b, 3), fmt(math.exp(min(b, 700)), 3)]
                           for nm, b in zip(names, beta[1:])]))
    for label, sel in (("Training", tr), ("Testing", sample == 1)):
        if not sel.any():
            continue
        pred = prob[sel] >= cut
        obs = yy[sel] == 1
        tab = [[f"{label}", "No", fmt_int(int((~obs & ~pred).sum())),
                fmt_int(int((~obs & pred).sum())),
                f"{100 * (~obs & ~pred).sum() / max((~obs).sum(), 1):.1f}%"],
               ["", "Yes", fmt_int(int((obs & ~pred).sum())), fmt_int(int((obs & pred).sum())),
                f"{100 * (obs & pred).sum() / max(obs.sum(), 1):.1f}%"],
               ["", "Overall Percent", "", "",
                f"{100 * (pred == obs).mean():.1f}%"]]
        section.table(Table(f"Classification Table ({label})",
                            ["Sample", "Observed", ("Predicted", "No"), ("Predicted", "Yes"),
                             "Percent Correct"], tab, row_header_cols=2,
                            footnotes=[f"The cut value is {cut:g}."]))
    order = np.argsort(-prob[te], kind="mergesort")
    yt = yy[te][order]
    pct = np.arange(1, len(yt) + 1) / len(yt) * 100
    gain = np.cumsum(yt) / max(yt.sum(), 1) * 100
    fig = figure(5.2, 4.2)
    ax = fig.add_subplot(111)
    ax.plot(np.r_[0, pct], np.r_[0, gain], color="#1f5fa8", label="Model")
    ax.plot([0, 100], [0, 100], color="#999", lw=0.8, label="Baseline")
    ax.legend(fontsize=8)
    style(ax, "Cumulative Gains", "Percentage of contacts", "Percentage of responses")
    section.image("Cumulative Gains", png(fig), width=440)
    saved = {}
    if save:
        allp = np.full(ds.n_cases, np.nan)
        allp[rows] = prob
        saved["Propensity_Score"] = ("Propensity to purchase score",
                                     pd.Series(allp, index=ds.df.index), 4)
    return MLResult(section, saved, {"beta": beta, "auc": auc, "names": names})


# ============================================================================ package test
def control_package_test(ds: Dataset, campaign: str, control, effectiveness: str,
                         kind: str = "response", positive: list | None = None,
                         alpha: float = 0.05) -> Section:
    """kind = response (binary) or spending (amount)."""
    camp = raw_column(ds, campaign)
    if kind == "response":
        y = _positive(ds, effectiveness, positive or [1.0])
    else:
        y = pd.to_numeric(raw_column(ds, effectiveness), errors="coerce")
    ok = camp.notna() & y.notna()
    levels = sort_levels(camp[ok].tolist())
    if control not in levels:
        raise MarketingError("The control package value does not occur in the campaign field.")
    tests = [lv for lv in levels if lv != control]
    if not tests:
        raise MarketingError("At least one test package is needed.")
    m = len(tests)
    yc = y[ok & (camp == control)].to_numpy(float)
    section = Section("Control Package Test")
    section.log(f"CONTROL PACKAGE TEST CAMPAIGN={campaign} CONTROL={control} "
                f"EFFECTIVENESS={effectiveness} TYPE={kind.upper()}.")
    rows = []
    lab = lambda v: value_text(ds.var(campaign), v)  # noqa: E731
    stat_c = yc.mean()
    rows.append([lab(control) + " (control)", fmt_int(len(yc)),
                 f"{100 * stat_c:.2f}%" if kind == "response" else fmt(stat_c, 2), "", "", "",
                 ""])
    best = (stat_c, control)
    for lv in tests:
        yt = y[ok & (camp == lv)].to_numpy(float)
        st = yt.mean()
        if kind == "response":
            p_pool = (yt.sum() + yc.sum()) / (len(yt) + len(yc))
            se = math.sqrt(p_pool * (1 - p_pool) * (1 / len(yt) + 1 / len(yc)))
            z = (st - stat_c) / se if se > 0 else 0.0
            p = 2 * stats.norm.sf(abs(z))
            stat_txt = f"{100 * st:.2f}%"
            diff = f"{100 * (st - stat_c):+.2f}"
        else:
            t = stats.ttest_ind(yt, yc, equal_var=False)
            z, p = float(t.statistic), float(t.pvalue)
            stat_txt = fmt(st, 2)
            diff = fmt(st - stat_c, 2)
        padj = min(1.0, p * m)
        if padj < alpha and st > best[0]:
            best = (st, lv)
        rows.append([lab(lv), fmt_int(len(yt)), stat_txt, diff, fmt(z, 3), fmt_p(p),
                     fmt_p(padj)])
    eff = "Response Rate" if kind == "response" else "Mean Spending"
    section.table(Table("Comparison with the Control Package", [
        "Package", "N", eff, "Difference from Control" + (" (points)" if kind == "response"
                                                           else ""),
        "z" if kind == "response" else "t", "Sig.", "Adjusted Sig. (Bonferroni)"], rows,
        footnotes=[("Two-proportion z test (pooled)" if kind == "response" else
                    "Welch t test") + f" of each package against the control; Sig. adjusted "
                   f"for {m} comparison(s)."]))
    if kind == "response":
        tab = np.array([[((camp == lv) & ok & (y == 1)).sum(), ((camp == lv) & ok & (y == 0))
                         .sum()] for lv in levels], float)
        chi2, pc, dof, _ = stats.chi2_contingency(tab, correction=False)
        section.table(Table("Overall Test", ["", "Value", "df", "Sig."],
                            [["Pearson Chi-Square", fmt(chi2, 3), fmt_int(dof), fmt_p(pc)]]))
    else:
        f = stats.f_oneway(*[y[ok & (camp == lv)].to_numpy(float) for lv in levels])
        section.table(Table("Overall Test", ["", "Value", "df", "Sig."],
                            [["F (one-way ANOVA)", fmt(f.statistic, 3),
                              f"{len(levels) - 1}, {int(ok.sum()) - len(levels)}",
                              fmt_p(f.pvalue)]]))
    section.note("Recommendation", f"Best performing package: {lab(best[1])}"
                 + (" (no test package beats the control significantly)."
                    if best[1] == control else "."))
    return section


def selftest() -> None:
    b = ntile(np.arange(10.0), 5)
    assert list(b) == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    sc = rfm_scores(np.arange(20.0), np.arange(20.0), np.arange(20.0), (2, 2, 2))
    assert sc[0, 0] == 2 and sc[-1, 0] == 1
