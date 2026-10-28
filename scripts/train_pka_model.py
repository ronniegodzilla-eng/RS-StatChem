"""Train the pKa and Henry's-law-constant models of the Chemical Properties window and export
them for numpy-only inference (``rsstatchem/core/chem_draw/pka.py``, ``henry.py``).

Needs scikit-learn (development dependency, never bundled) and two files of the US EPA OPERA
data (``OPERA_Data.zip``, https://github.com/NIEHS/OPERA, MIT licensed repository; the same
archive the property database uses)::

    python scripts/train_pka_model.py --download        # fetches OPERA_Data.zip once
    python scripts/train_pka_model.py                   # -> resources/properties/pka_model.npz
                                                        #    and henry_model.npz
Data
----
* ``pKa_QR.sdf``: the curated DataWarrior pKa set of K. Mansouri et al., "Open-source QSAR
  models for pKa prediction using multiple machine learning approaches", J. Cheminform. 11
  (2019) 60 (data under the BMC CC0 waiver; OPERA code MIT). 6 503 QSAR-ready structures with
  the strongest acidic pKa (``pKa_a``) and/or the strongest basic pKa (``pKa_b``, pKa of the
  conjugate acid). The file has no train/test flag, so the hold-out split is our own (seeded).
* ``HL_QR.sdf``: PHYSPROP Henry's law constants curated for OPERA (K. Mansouri et al.,
  J. Cheminform. 10 (2018) 10), log10 HLC in atm m3/mol at 25 °C, with OPERA's training/test
  flag (``Tr_1_Tst_0``).

Procedure (fixed seeds)
-----------------------
Same features and model family as the melting/boiling point models (``ml_props.features``;
HistGradientBoostingRegressor), stereoisomers merged by InChIKey connectivity (median):

* pKa (acidic, basic): 5-fold random CV, a Bemis-Murcko scaffold split (80/20) and a seeded
  random 75/25 hold-out (the proportion Mansouri et al. used); the expected-error table by
  similarity to the training folds as for the other ML models;
* HLC: 5-fold CV plus OPERA's own test set (model trained on OPERA's training set only);
  the Henry estimate from vapour pressure / water solubility (HLC = P_vap / S_w) is benchmarked
  on the same compounds by ``--bench-vpws``;
* the exported models are refitted on all compounds. The experimental values (InChIKey ->
  value) are stored too, so a compound in the data shows its measured value.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
RAW = ROOT / "build" / "propdb-raw" / "opera"
OUT = ROOT / "rsstatchem" / "resources" / "properties"
OPERA_URL = "https://github.com/NIEHS/OPERA/raw/master/OPERA_Data.zip"
PARAMS = dict(max_iter=500, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=10,
              l2_regularization=1.0, max_bins=255, early_stopping=False, random_state=0)


def download(raw: Path) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    if (raw / "pKa_QR.sdf").exists() and (raw / "HL_QR.sdf").exists():
        return
    print("OPERA data (~80 MB)")
    req = urllib.request.Request(OPERA_URL, headers={"User-Agent": "RS-StatChem build"})
    with urllib.request.urlopen(req, timeout=600) as r:
        blob = r.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for n in ("pKa_QR.sdf", "HL_QR.sdf"):
            (raw / n).write_bytes(z.read(n))


def read_sdf(path: Path) -> list[dict]:
    """[{field: value, 'molblock': ...}] without RDKit parsing (fields as text)."""
    out = []
    text = path.read_text("utf-8", errors="replace")
    for block in text.split("$$$$"):
        if "M  END" not in block:
            continue
        mb, _, rest = block.partition("M  END")
        rec = {"molblock": mb.lstrip("\r\n") + "M  END\n"}
        lines = rest.splitlines()
        for i, line in enumerate(lines):
            if line.startswith("> ") and "<" in line:
                key = line[line.index("<") + 1:line.rindex(">")]
                rec[key] = lines[i + 1].strip() if i + 1 < len(lines) else ""
        out.append(rec)
    return out


def _float(s) -> float:
    try:
        v = float(s)
    except (TypeError, ValueError):
        return math.nan
    return v if math.isfinite(v) else math.nan


def _prepare(smiles: str):
    import train_property_models as tpm
    return tpm._prepare(smiles)


def dataset(recs, fields, smiles_field="Canonical_QSARr", key_field="InChI Key_QSARr",
            extra=None, workers=6):
    """Merge by connectivity block; (X, fps, scaffolds, {field: y}, keys, inchikeys, extras)."""
    from multiprocessing import Pool
    groups: dict = {}
    for r in recs:
        key = r.get(key_field, "")
        if len(key) < 14 or not r.get(smiles_field):
            continue
        groups.setdefault(key[:14], []).append(r)
    merged = []
    for block, rs in groups.items():
        rs.sort(key=lambda r: (not r[key_field].endswith("UHFFFAOYSA-N"), r[key_field]))
        e = {"block": block, "smiles": rs[0][smiles_field], "keys": [r[key_field] for r in rs]}
        for f in fields:
            vals = [_float(r.get(f)) for r in rs]
            vals = [v for v in vals if not math.isnan(v)]
            e[f] = float(np.median(vals)) if vals else math.nan
        if extra:
            e["extra"] = extra(rs)
        merged.append(e)
    with Pool(workers) as pool:
        prepared = pool.map(_prepare, [e["smiles"] for e in merged], chunksize=32)
    keep = [(e, p) for e, p in zip(merged, prepared) if p is not None]
    print(f"{len(recs)} records, {len(merged)} compounds, {len(keep)} in scope")
    X = np.vstack([p[0] for _, p in keep])
    fps = np.vstack([p[1] for _, p in keep])
    scaffolds = [p[2] for _, p in keep]
    y = {f: np.array([e[f] for e, _ in keep]) for f in fields}
    return X, fps, scaffolds, y, [e for e, _ in keep]


def holdout(target, X, y, fps, test_mask) -> dict:
    """Train on ~test_mask, score on test_mask."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    import train_property_models as tpm
    idx = np.where(np.isfinite(y))[0]
    tr = idx[~test_mask[idx]]
    te = idx[test_mask[idx]]
    m = HistGradientBoostingRegressor(**PARAMS).fit(X[tr], y[tr])
    p = m.predict(X[te])
    res = tpm._metrics(y[te], p)
    sims = tpm._max_sim(fps[te], fps[tr])
    res["in_domain"] = tpm._metrics(y[te][sims >= 0.3], p[sims >= 0.3]) \
        if (sims >= 0.3).sum() >= 10 else None
    print(f"  {target} hold-out: {res}")
    return res, te, p


def train(name, X, fps, scaffolds, y, entries, raw, test_mask=None, test_label="holdout"):
    import train_property_models as tpm
    tpm.PARAMS = PARAMS
    arrays, metrics, idx = tpm.train_target(name, X, fps, scaffolds, y,
                                            [e["block"] for e in entries], raw, oof=True)
    if test_mask is not None:
        metrics[test_label], _te, _p = holdout(name, X, y, fps, test_mask)
    return arrays, metrics, idx


def export(path: Path, models: dict, fps, exp_tables: dict, info: dict) -> None:
    import sklearn
    out: dict = {}
    used = np.zeros(len(fps), bool)
    for t, (arrays, metrics, idx) in models.items():
        meta = {"baseline": arrays.pop("baseline"), "metrics": metrics,
                "sklearn": sklearn.__version__, "features": "ml_props.features v1",
                **info.get(t, {})}
        for k, v in arrays.items():
            out[f"{t}_{k}"] = v
        out[f"{t}_meta"] = np.array(json.dumps(meta))
        out[f"{t}_fp_index"] = idx.astype(np.int32)
        used[idx] = True
    remap = np.cumsum(used) - 1
    for t in models:
        out[f"{t}_fp_index"] = remap[out[f"{t}_fp_index"]].astype(np.int32)
    out["fps"] = fps[used]
    for t, table in exp_tables.items():
        keys = sorted(table)
        out[f"exp_{t}_keys"] = np.array(keys, dtype="U27")
        out[f"exp_{t}_values"] = np.array([table[k] for k in keys], dtype=np.float32)
    np.savez_compressed(path, **out)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB)")


def consistent(recs: list[dict]) -> list[dict]:
    """Blank pKa_a (pKa_b) of structures without an acidic (basic) group of the types in
    ``pka.ACIDIC_GROUPS`` / ``BASIC_GROUPS`` - the app never predicts those, and they are
    mostly values filed under the wrong kind (phenol listed with a basic pKa of 9.9)."""
    from rdkit import Chem, RDLogger

    from rsstatchem.core.chem_draw import pka
    RDLogger.DisableLog("rdApp.*")
    dropped = {"pKa_a": 0, "pKa_b": 0}
    for r in recs:
        mol = Chem.MolFromSmiles(r.get("Canonical_QSARr", ""))
        if mol is None:
            continue
        g = pka.ionisable_groups(mol)
        for field, kind in (("pKa_a", "acid"), ("pKa_b", "base")):
            if not math.isnan(_float(r.get(field))) and not g[kind]:
                r[field] = "NaN"
                dropped[field] += 1
    print(f"values without a matching group (dropped): {dropped}")
    return recs


def run_pka(raw: Path, out: Path, workers: int) -> dict:
    recs = read_sdf(raw / "pKa_QR.sdf")
    recs = consistent(recs)
    X, fps, scaffolds, y, entries = dataset(recs, ("pKa_a", "pKa_b"), workers=workers)
    rng = np.random.default_rng(0)
    test_mask = rng.random(len(entries)) < 0.25
    models = {}
    for field, t in (("pKa_a", "pka_a"), ("pKa_b", "pka_b")):
        models[t] = train(t, X, fps, scaffolds, y[field], entries, raw, test_mask,
                          "holdout_75_25")
    # experimental values by full InChIKey (every record of a merged compound)
    exp = {"pka_a": {}, "pka_b": {}}
    for r in recs:
        k = r.get("InChI Key_QSARr", "")
        for field, t in (("pKa_a", "pka_a"), ("pKa_b", "pka_b")):
            v = _float(r.get(field))
            if k and not math.isnan(v):
                exp[t][k] = v
    info = {t: {"data": "Mansouri et al. 2019 (OPERA pKa_QR.sdf), CC0"} for t in models}
    export(out / "pka_model.npz", models, fps, exp, info)
    return {t: m[1] for t, m in models.items()}


def run_henry(raw: Path, out: Path, workers: int) -> dict:
    recs = read_sdf(raw / "HL_QR.sdf")

    def split(rs):
        vals = [_float(r.get("Tr_1_Tst_0")) for r in rs]
        return 0 if any(v == 0 for v in vals) else 1

    X, fps, scaffolds, y, entries = dataset(recs, ("LogHL",), extra=split, workers=workers)
    test_mask = np.array([e["extra"] == 0 for e in entries])
    models = {"loghl": train("loghl", X, fps, scaffolds, y["LogHL"], entries, raw, test_mask,
                             "opera_test")}
    exp = {"loghl": {}}
    for r in recs:
        k = r.get("InChI Key_QSARr", "")
        v = _float(r.get("LogHL"))
        if k and not math.isnan(v):
            exp["loghl"][k] = v
    info = {"loghl": {"data": "PHYSPROP via OPERA (HL_QR.sdf), log10 atm m3/mol, 25 °C"}}
    export(out / "henry_model.npz", models, fps, exp, info)
    return {"loghl": models["loghl"][1]}


def bench_vpws(raw: Path) -> dict:
    """HLC = P_vap / S_w from the app's Vapor Pres. and LogS rows, against PHYSPROP HLC."""
    from rdkit import Chem, RDLogger

    from rsstatchem.core.chem_draw import henry
    RDLogger.DisableLog("rdApp.*")
    recs = read_sdf(raw / "HL_QR.sdf")
    err = {"all": [], "both_experimental": [], "estimated": []}
    for r in recs:
        mol = Chem.MolFromSmiles(r.get("Canonical_QSARr", ""))
        if mol is None:
            continue
        est = henry.vp_ws_estimate(mol)
        if est is None or est.value is None:
            continue
        obs = _float(r.get("LogHL")) + math.log10(101325.0)    # -> log10 Pa m3/mol
        e = math.log10(est.value) - obs
        err["all"].append(e)
        err["both_experimental" if est.experimental_inputs else "estimated"].append(e)
    out = {}
    for k, v in err.items():
        if v:
            a = np.array(v)
            out[k] = {"n": int(len(a)), "rmse": float(np.sqrt(np.mean(a ** 2))),
                      "mae": float(np.mean(np.abs(a))), "bias": float(np.mean(a))}
    print("VP/WS estimate:", json.dumps(out, indent=1))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--only", choices=("pka", "henry", "vpws"), default=None)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args(argv)
    if args.download:
        download(args.raw)
    t0 = time.time()
    summary = {}
    if args.only in (None, "pka"):
        summary["pka"] = run_pka(args.raw, args.out, args.workers)
    if args.only in (None, "henry"):
        summary["henry"] = run_henry(args.raw, args.out, args.workers)
    if args.only in (None, "vpws"):
        summary["vpws"] = bench_vpws(args.raw)
        path = args.out / "henry_vpws_benchmark.json"
        path.write_text(json.dumps(summary["vpws"], indent=1), "utf-8")
    print(f"done in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
