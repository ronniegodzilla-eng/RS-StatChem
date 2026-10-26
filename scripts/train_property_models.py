"""Train the melting point / boiling point / logP models of the Chemical Properties window and
export them for numpy-only inference (``rsstatchem/core/chem_draw/ml_props.py``).

Needs scikit-learn (development dependency, ``requirements-dev.txt``; never bundled) and the
training table written by ``python scripts/build_property_db.py --training``::

    python scripts/train_property_models.py            # -> rsstatchem/resources/properties/
                                                        #    ml_models.npz
Procedure (fixed seeds, reproducible):

* compounds: the experimental consensus values of the property database; single-component,
  uncharged-overall organic molecules (C plus H, N, O, F, Cl, Br, I, S, P, Si, B, Se), stereo-
  isomers merged by InChIKey connectivity block (median), 2-150 heavy atoms;
* features: ``ml_props.features`` (RDKit descriptors, element counts, symmetry ratio, Morgan
  counts radius 2 / 1024 bins);
* model: ``HistGradientBoostingRegressor`` (squared error);
* held-out evaluation: 5-fold random cross-validation (every compound predicted once by a
  model that never saw it) and one Bemis-Murcko scaffold split (80/20, rare scaffolds in the
  test set, as in DeepChem's ScaffoldSplitter); RMSE, MAE and R^2 are stored in the model file.
  The cross-validated predictions are also binned by the compound's highest Tanimoto similarity
  to the training folds: that table is the *expected error* shown in the app;
* the exported model is refitted on all compounds with the same settings.

``--oof`` also writes the out-of-fold predictions (``<raw>/oof_<target>.tsv``), used by
``scripts/benchmark_properties.py`` to compare the model with the group-contribution methods
on the same held-out compounds.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RAW = ROOT / "build" / "propdb-raw"
OUT = ROOT / "rsstatchem" / "resources" / "properties" / "ml_models.npz"

ALLOWED = {"C", "H", "N", "O", "F", "Cl", "Br", "I", "S", "P", "Si", "B", "Se"}
PARAMS = dict(max_iter=700, learning_rate=0.07, max_leaf_nodes=48, min_samples_leaf=12,
              l2_regularization=1.0, max_bins=255, early_stopping=False, random_state=0)
AD_BINS = ((0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.0001))


def _prepare(smiles: str):
    """(features, packed fingerprint, scaffold) or None for molecules outside the scope."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    from rsstatchem.core.chem_draw import ml_props
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        return None
    symbols = {a.GetSymbol() for a in mol.GetAtoms()}
    if "C" not in symbols or not symbols <= ALLOWED or Chem.GetFormalCharge(mol) != 0:
        return None
    if not 2 <= mol.GetNumHeavyAtoms() <= 150 or \
            any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        return None
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:  # noqa: BLE001
        scaffold = ""
    return ml_props.features(mol), ml_props.fingerprint(mol), scaffold


def load(raw: Path, workers: int):
    rows = []
    with open(raw / "training.tsv", encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            rows.append(dict(zip(header, parts)))
    # merge stereoisomers / isotopologues by connectivity block (median per property)
    groups: dict = {}
    for r in rows:
        groups.setdefault(r["inchikey"][:14], []).append(r)
    merged = []
    for block, rs in groups.items():
        rs.sort(key=lambda r: (not r["inchikey"].endswith("UHFFFAOYSA-N"), r["inchikey"]))
        entry = {"block": block, "smiles": rs[0]["smiles"]}
        for t in ("mp", "bp", "logp"):
            vals = [float(r[t]) for r in rs if r[t]]
            entry[t] = float(np.median(vals)) if vals else math.nan
        merged.append(entry)
    print(f"{len(rows)} compounds, {len(merged)} after merging stereoisomers")
    t0 = time.time()
    with Pool(workers) as pool:
        prepared = pool.map(_prepare, [e["smiles"] for e in merged], chunksize=64)
    keep = [(e, p) for e, p in zip(merged, prepared) if p is not None]
    print(f"{len(keep)} in scope; features in {time.time() - t0:.0f} s")
    X = np.vstack([p[0] for _, p in keep])
    fps = np.vstack([p[1] for _, p in keep])
    scaffolds = [p[2] for _, p in keep]
    y = {t: np.array([e[t] for e, _ in keep]) for t in ("mp", "bp", "logp")}
    keys = [e["block"] for e, _ in keep]
    return X, fps, scaffolds, y, keys


def _metrics(y, p) -> dict:
    err = p - y
    ss = float(np.sum((y - y.mean()) ** 2))
    return {"n": int(len(y)), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mae": float(np.mean(np.abs(err))),
            "r2": float(1 - np.sum(err ** 2) / ss) if ss > 0 else math.nan}


def _max_sim(fps_test, fps_train) -> np.ndarray:
    from rsstatchem.core.chem_draw.ml_props import tanimoto_max
    return np.array([tanimoto_max(f, fps_train)[0] for f in fps_test])


def scaffold_split(scaffolds, idx, frac=0.2):
    """DeepChem-style: scaffold sets sorted by size, largest go to training first."""
    sets: dict = {}
    for i in idx:
        sets.setdefault(scaffolds[i], []).append(i)
    ordered = sorted(sets.values(), key=lambda s: (len(s), s[0]), reverse=True)
    n_train = int(round(len(idx) * (1 - frac)))
    train, test = [], []
    for s in ordered:
        (train if len(train) + len(s) <= n_train else test).extend(s)
    return np.array(train), np.array(test)


def export_trees(model) -> dict:
    """Flat arrays of a fitted HistGradientBoostingRegressor (single output)."""
    feats, thr, left, right, val, miss, offsets = [], [], [], [], [], [], []
    start = 0
    for (pred,) in model._predictors:
        nodes = pred.nodes
        offsets.append(start)
        leaf = nodes["is_leaf"].astype(bool)
        feats.append(np.where(leaf, 0, nodes["feature_idx"].astype(np.int64)).astype(np.int16))
        thr.append(nodes["num_threshold"].astype(np.float64))
        left.append(np.where(leaf, -1, nodes["left"].astype(np.int64)).astype(np.int16))
        right.append(np.where(leaf, -1, nodes["right"].astype(np.int64)).astype(np.int16))
        val.append(np.where(leaf, nodes["value"], 0.0).astype(np.float32))
        miss.append(nodes["missing_go_to_left"].astype(bool))
        start += len(nodes)
    return {"feature": np.concatenate(feats), "threshold": np.concatenate(thr),
            "left": np.concatenate(left), "right": np.concatenate(right),
            "value": np.concatenate(val), "missing_left": np.concatenate(miss),
            "offsets": np.array(offsets, dtype=np.int32),
            "baseline": float(np.ravel(model._baseline_prediction)[0])}


def train_target(target, X, fps, scaffolds, y, keys, raw: Path, oof: bool) -> tuple:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import KFold

    from rsstatchem.core.chem_draw.ml_props import GBTrees
    idx = np.where(np.isfinite(y))[0]
    yt = y[idx]
    print(f"\n== {target}: {len(idx)} compounds")
    pred = np.full(len(idx), np.nan)
    sims = np.zeros(len(idx))
    for k, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=0).split(idx)):
        m = HistGradientBoostingRegressor(**PARAMS).fit(X[idx[tr]], yt[tr])
        pred[te] = m.predict(X[idx[te]])
        sims[te] = _max_sim(fps[idx[te]], fps[idx[tr]])
        print(f"  fold {k + 1}: RMSE {np.sqrt(np.mean((pred[te] - yt[te]) ** 2)):.3f}")
    random_cv = _metrics(yt, pred)
    bins = []
    for lo, hi in AD_BINS:
        sel = (sims >= lo) & (sims < hi)
        if sel.sum() >= 20:
            bins.append([lo, min(hi, 1.0), float(np.sqrt(np.mean((pred[sel] - yt[sel]) ** 2))),
                         int(sel.sum())])
    in_ad = sims >= 0.3
    tr, te = scaffold_split(scaffolds, np.arange(len(idx)))
    m = HistGradientBoostingRegressor(**PARAMS).fit(X[idx[tr]], yt[tr])
    scaffold = _metrics(yt[te], m.predict(X[idx[te]]))
    print(f"  random 5-fold CV {random_cv}\n  scaffold split  {scaffold}\n  AD bins {bins}")
    final = HistGradientBoostingRegressor(**PARAMS).fit(X[idx], yt)
    arrays = export_trees(final)
    check = GBTrees(arrays["feature"], arrays["threshold"], arrays["left"], arrays["right"],
                    arrays["value"], arrays["missing_left"], arrays["offsets"],
                    arrays["baseline"])
    sample = idx[:: max(1, len(idx) // 300)]
    diff = np.max(np.abs(check.predict(X[sample]) - final.predict(X[sample])))
    print(f"  numpy export vs scikit-learn: max |difference| {diff:.2e}")
    if diff > 1e-3:
        raise SystemExit("exported trees do not reproduce scikit-learn")
    metrics = {"random_cv": random_cv, "scaffold": scaffold, "ad_bins": bins,
               "in_domain_fraction": float(in_ad.mean()),
               "in_domain_rmse": float(np.sqrt(np.mean((pred[in_ad] - yt[in_ad]) ** 2))),
               "out_domain_rmse": float(np.sqrt(np.mean((pred[~in_ad] - yt[~in_ad]) ** 2)))
               if (~in_ad).sum() else None,
               "n_train": int(len(idx)), "params": PARAMS}
    if oof:
        with open(raw / f"oof_{target}.tsv", "w", encoding="utf-8") as fh:
            fh.write("block\tobserved\tpredicted\tsimilarity\n")
            for i, p, s in zip(idx, pred, sims):
                fh.write(f"{keys[i]}\t{y[i]:.4f}\t{p:.4f}\t{s:.4f}\n")
    return arrays, metrics, idx


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--targets", default="mp,bp,logp")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--oof", action="store_true", help="write out-of-fold predictions")
    args = ap.parse_args(argv)
    import sklearn
    X, fps, scaffolds, y, keys = load(args.raw, args.workers)
    out: dict = {}
    used = np.zeros(len(fps), bool)
    for t in args.targets.split(","):
        arrays, metrics, idx = train_target(t, X, fps, scaffolds, y[t], keys, args.raw,
                                            args.oof)
        meta = {"baseline": arrays.pop("baseline"), "metrics": metrics,
                "sklearn": sklearn.__version__, "features": "ml_props.features v1"}
        for k, v in arrays.items():
            out[f"{t}_{k}"] = v
        out[f"{t}_meta"] = np.array(json.dumps(meta))
        out[f"{t}_fp_index"] = idx.astype(np.int32)
        used[idx] = True
    # keep only the fingerprints some model was trained on; remap the indices
    remap = np.cumsum(used) - 1
    for t in args.targets.split(","):
        out[f"{t}_fp_index"] = remap[out[f"{t}_fp_index"]].astype(np.int32)
    out["fps"] = fps[used]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f"\nwrote {args.out} ({args.out.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
