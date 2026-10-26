"""Benchmark every estimation method of the Chemical Properties window against the open
experimental data, and write ``rsstatchem/resources/properties/benchmarks.json`` (used by the
app to pick the default estimate per property and to show each method's error).

Inputs (see ``scripts/build_property_db.py --training`` and
``scripts/train_property_models.py --oof``): ``<raw>/training.tsv`` and the machine-learning
models' out-of-fold predictions ``<raw>/oof_<target>.tsv``. CDK logP needs the CDK jar and
helper (``scripts/fetch_cdk.py``) and Java::

    python scripts/benchmark_properties.py [--java PATH]

Compounds: as for the ML models (single-component, overall neutral organics, stereoisomers
merged). For each property and method: n (compounds the method covers), RMSE, MAE, bias, and
the same statistics on the *common* subset that every compared method covers (used for the
ranking). Caveat written into the file: the group-contribution and logP methods were fitted by
their authors on data that overlap these sets (an optimistic test for them), whereas the ML
numbers are 5-fold cross-validated (every compound predicted by a model that never saw it).
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
OUT = ROOT / "rsstatchem" / "resources" / "properties" / "benchmarks.json"
ALLOWED = {"C", "H", "N", "O", "F", "Cl", "Br", "I", "S", "P", "Si", "B", "Se"}
GC_PROPS = ("mp", "bp", "tc", "pc", "vc")
LOGP_METHODS = ("crippen", "jplogp", "xlogp", "alogp")


def load(raw: Path) -> list[dict]:
    rows = []
    with open(raw / "training.tsv", encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            rows.append(dict(zip(header, line.rstrip("\n").split("\t"))))
    groups: dict = {}
    for r in rows:
        groups.setdefault(r["inchikey"][:14], []).append(r)
    out = []
    for block, rs in groups.items():
        rs.sort(key=lambda r: (not r["inchikey"].endswith("UHFFFAOYSA-N"), r["inchikey"]))
        e = {"block": block, "smiles": rs[0]["smiles"]}
        for p in ("mp", "bp", "logp", "logs", "tc", "pc", "vc"):
            vals = [float(r[p]) for r in rs if r[p]]
            e[p] = float(np.median(vals)) if vals else math.nan
        out.append(e)
    return out


def _gc(smiles: str) -> dict | None:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Crippen, Descriptors

    from rsstatchem.core.chem_draw import properties as P
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        return None
    sym = {a.GetSymbol() for a in mol.GetAtoms()}
    if "C" not in sym or not sym <= ALLOWED or Chem.GetFormalCharge(mol) != 0 or \
            not 2 <= mol.GetNumHeavyAtoms() <= 150:
        return None
    out: dict = {}
    try:
        jb = P.joback(mol)
        if jb.ok:
            for p, v in (("mp", jb.tm), ("bp", jb.tb), ("tc", jb.tc), ("pc", jb.pc),
                         ("vc", jb.vc)):
                out[f"{p}:joback"] = v
        sb = P.stein_brown(mol)
        if sb.ok:
            out["bp:sb"] = sb.values.get("tb")
        for key, second in (("mg1", False), ("mg", True)):
            mg = P.marrero_gani(mol, second_order=second)
            if mg.ok:
                for p, k in (("mp", "tm"), ("bp", "tb"), ("tc", "tc"), ("pc", "pc"),
                             ("vc", "vc")):
                    out[f"{p}:{key}"] = mg.values.get(k)
        ag = P.abdulelah_gani(mol)
        if ag.ok:
            for p in ("tc", "pc", "vc"):
                out[f"{p}:ag"] = ag.values.get(p)
    except Exception:  # noqa: BLE001 - a failing estimate is simply missing
        pass
    logp = Crippen.MolLogP(mol)
    out["logp:crippen"] = logp
    mw = Descriptors.MolWt(mol)
    rb = Descriptors.NumRotatableBonds(mol)
    heavy = mol.GetNumHeavyAtoms() or 1
    ap = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic()) / heavy
    out["logs:esol"] = 0.16 - 0.63 * logp - 0.0062 * mw + 0.066 * rb - 0.74 * ap
    try:
        kek = Chem.Mol(mol)
        Chem.Kekulize(kek, clearAromaticFlags=True)
        out["_kekule"] = Chem.MolToSmiles(kek, kekuleSmiles=True)
    except Exception:  # noqa: BLE001
        out["_kekule"] = smiles
    return {k: v for k, v in out.items() if v is None or isinstance(v, str) or
            math.isfinite(v)}


def cdk_batch(smiles: list[str], java: str | None) -> list[dict]:
    from rsstatchem.core.chem_draw import cdk
    helper = cdk.CdkHelper(java=java)
    out = []
    t0 = time.time()
    try:
        for i, s in enumerate(smiles):
            try:
                out.append(helper.compute(s, timeout=60))
            except cdk.CdkError as exc:
                out.append({"error": str(exc)})
            if i % 2000 == 0:
                print(f"  CDK {i}/{len(smiles)} ({time.time() - t0:.0f} s)", flush=True)
    finally:
        helper.close()
    print(f"  CDK helper started {helper.starts} time(s), CDK {helper.version}")
    return out


def stats(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = p - y
    return {"n": int(len(y)), "rmse": round(float(np.sqrt(np.mean(e ** 2))), 3),
            "mae": round(float(np.mean(np.abs(e))), 3), "bias": round(float(np.mean(e)), 3)}


def evaluate(entries, prop, methods) -> dict:
    """Per-method statistics on covered compounds and on the common subset."""
    obs = [e for e in entries if math.isfinite(e[prop])]
    res: dict = {"methods": {}, "n_experimental": len(obs)}
    for m in methods:
        sel = [e for e in obs if e.get(f"{prop}:{m}") is not None]
        if len(sel) >= 10:
            s = stats([e[prop] for e in sel], [e[f"{prop}:{m}"] for e in sel])
            s["coverage"] = round(len(sel) / len(obs), 3)
            res["methods"][m] = s
    have = list(res["methods"])
    common = [e for e in obs if all(e.get(f"{prop}:{m}") is not None for m in have)]
    if len(common) >= 10:
        res["common"] = {"n": len(common), "methods": {
            m: stats([e[prop] for e in common], [e[f"{prop}:{m}"] for e in common])
            for m in have}}
    # compounds outside the ML applicability domain (similarity to the training folds < 0.3):
    # which method is best there? Compared on compounds that every method covers.
    if "ml" in have:
        ood = [e for e in obs if e.get(f"{prop}:sim", 1.0) < 0.3]
        ood_have = [m for m in have
                    if sum(1 for e in ood if e.get(f"{prop}:{m}") is not None) >= 20]
        ood_common = [e for e in ood if all(e.get(f"{prop}:{m}") is not None
                                             for m in ood_have)]
        if len(ood_common) >= 20:
            res["ood_common"] = {"n": len(ood_common), "methods": {
                m: stats([e[prop] for e in ood_common], [e[f"{prop}:{m}"] for e in ood_common])
                for m in ood_have}}
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--java", help="java executable for CDK (default: as the app finds it)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no-cdk", action="store_true")
    args = ap.parse_args(argv)
    entries = load(args.raw)
    t0 = time.time()
    with Pool(args.workers) as pool:
        gc = pool.map(_gc, [e["smiles"] for e in entries], chunksize=64)
    entries = [dict(e, **g) for e, g in zip(entries, gc) if g is not None]
    print(f"{len(entries)} compounds in scope; group contributions in {time.time() - t0:.0f} s")
    # machine-learning out-of-fold predictions
    for t in ("mp", "bp", "logp"):
        f = args.raw / f"oof_{t}.tsv"
        if not f.exists():
            continue
        oof = {}
        with open(f, encoding="utf-8") as fh:
            fh.readline()
            for line in fh:
                b, _o, p, s = line.rstrip("\n").split("\t")
                oof[b] = (float(p), float(s))
        for e in entries:
            if e["block"] in oof:
                e[f"{t}:ml"], e[f"{t}:sim"] = oof[e["block"]]
    if not args.no_cdk:
        logp_entries = [e for e in entries if math.isfinite(e["logp"])]
        print(f"CDK logP for {len(logp_entries)} compounds")
        res = cdk_batch([e["_kekule"] for e in logp_entries], args.java)
        failed = 0
        for e, r in zip(logp_entries, res):
            for m in ("jplogp", "xlogp", "alogp"):
                if r.get(m) is not None:
                    e[f"logp:{m}"] = r[m]
            failed += r.get("jplogp") is None
        print(f"  CDK gave no JPlogP for {failed} compounds")
    for e in entries:
        comps = [e.get(f"logp:{m}") for m in LOGP_METHODS]
        comps = [c for c in comps if c is not None]
        if len(comps) == len(LOGP_METHODS):
            e["logp:consensus4"] = float(np.mean(comps))
            if e.get("logp:ml") is not None:
                e["logp:consensus"] = float(np.mean(comps + [e["logp:ml"]]))
    out = {"_about": (
        "RS-StatChem benchmark of the property estimates against the open experimental data "
        "of exp_props.tsv.xz (scripts/benchmark_properties.py). Group-contribution and logP "
        "methods were fitted by their authors on data that overlap these sets (optimistic for "
        "them); 'ml' values are 5-fold cross-validated predictions (held out). Units: K for "
        "mp/bp/tc, bar for pc, cm3/mol for vc, log units for logp/logs."),
        "generated": time.strftime("%Y-%m-%d")}
    for prop in GC_PROPS:
        methods = ["joback", "sb", "mg1", "mg", "ag"] + (["ml"] if prop in ("mp", "bp") else [])
        out[prop] = evaluate(entries, prop, methods)
    out["logp"] = evaluate(entries, "logp", list(LOGP_METHODS) + ["ml", "consensus4",
                                                                   "consensus"])
    out["logs"] = evaluate(entries, "logs", ["esol"])
    args.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    for prop in GC_PROPS + ("logp", "logs"):
        print(f"\n== {prop} (n experimental = {out[prop]['n_experimental']})")
        for m, s in out[prop]["methods"].items():
            print(f"  {m:11s} n={s['n']:6d} cov={s['coverage']:.2f} RMSE={s['rmse']:8.3f} "
                  f"MAE={s['mae']:8.3f} bias={s['bias']:8.3f}")
        if "common" in out[prop]:
            c = out[prop]["common"]
            print(f"  common subset n={c['n']}: " + ", ".join(
                f"{m} {s['rmse']:.3f}" for m, s in c["methods"].items()))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
