"""Build the HOSE-code shift tables of the NMR predictor (``rsstatchem/core/chem_draw/nmr_predict.py``)
from the assigned 1H / 13C spectra of the BMRB metabolomics archive.

Data source and licence
-----------------------
BMRB (Biological Magnetic Resonance Data Bank) metabolomics entries ``bmse*``: experimental
1D/2D spectra of small molecules with assigned chemical shifts, NMR-STAR v3 files at
https://bmrb.io/ftp/pub/bmrb/metabolomics/entry_directories/. BMRB is a wwPDB core archive;
the wwPDB advisory notice (https://files.wwpdb.org/pub/pdb/advisory.txt) states that "the
archival data files in the wwPDB core archives (PDB, EMDB, and BMRB) are made freely available
to all users under CC0 license terms". We redistribute only derived numbers (HOSE-code shift
statistics), never modified BMRB files under their own names.

nmrshiftdb2 is *not* used: its database licence (ODbL-derived, section 4.5) requires NMR
prediction software that relies on the data to be distributed under an OSI-approved licence,
which RS-StatChem is not.

Usage::

    python scripts/build_nmr_db.py --download     # ~1 300 entries into build/nmr-raw/bmrb
    python scripts/build_nmr_db.py                # parse, benchmark, write resources/nmr/

Procedure
---------
* every entry with a covalent structure (``_Chem_comp_atom`` + ``_Chem_comp_bond``) and an
  assigned shift list is read; the structure is rebuilt from the NMR-STAR atom and bond loops
  (sanitised by RDKit), the shifts are attached to the atoms by their ``Atom_ID``;
* entries whose shifts look mis-assigned are dropped (an sp3 carbon above 100 ppm with no
  heteroatom neighbour, an aldehyde/carbonyl carbon below 150 ppm, an H shift outside -1..16);
* one molecule = one record (several entries of the same compound: the first, in BMRB order);
* held-out benchmark: compounds split 80/20 by a hash of their InChIKey connectivity block;
  the table built from the 80 % predicts the 20 % (MAE / RMSE per nucleus and per HOSE radius
  used); the shipped table is then built from all compounds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RAW = ROOT / "build" / "nmr-raw"
OUT = ROOT / "rsstatchem" / "resources" / "nmr"
API = "https://api.bmrb.io/v2/list_entries?database=metabolomics"
ENTRY = "https://bmrb.io/ftp/pub/bmrb/metabolomics/entry_directories/{0}/{0}.str"
UA = {"User-Agent": "RS-StatChem build_nmr_db (open-data HOSE table)"}


def _get(url: str, tries: int = 3) -> bytes:
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception:  # noqa: BLE001
            if k == tries - 1:
                raise
            time.sleep(2 + 3 * k)
    return b""


def download() -> None:
    d = RAW / "bmrb"
    d.mkdir(parents=True, exist_ok=True)
    ids = [i for i in json.loads(_get(API)) if i.startswith("bmse")]
    print(f"{len(ids)} metabolomics entries")

    def one(eid: str) -> str:
        path = d / f"{eid}.str"
        if path.exists() and path.stat().st_size > 0:
            return "cached"
        try:
            path.write_bytes(_get(ENTRY.format(eid)))
            return "ok"
        except Exception as exc:  # noqa: BLE001
            return f"failed {exc}"

    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(one, ids))
    print({k: res.count(k) for k in set(r.split()[0] for r in res)})


# --- minimal NMR-STAR v3 reader ---------------------------------------------------------------

_TOKEN = re.compile(r"""'(?:[^']|'(?!\s|$))*'(?=\s|$)|"(?:[^"]|"(?!\s|$))*"(?=\s|$)|\S+""")


def _tokens(text: str):
    """NMR-STAR tokens; semicolon-delimited text fields become one token."""
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith(";"):
            buf = [line[1:]]
            i += 1
            while i < len(lines) and not lines[i].startswith(";"):
                buf.append(lines[i])
                i += 1
            i += 1
            yield "\n".join(buf)
            continue
        if "#" in line and not line.lstrip().startswith(("'", '"')):
            # comments only at the start of a token
            pass
        for m in _TOKEN.finditer(line):
            t = m.group(0)
            if t.startswith("#"):
                break
            if len(t) >= 2 and t[0] == t[-1] and t[0] in "'\"":
                t = t[1:-1]
            yield t
        i += 1


def parse_star(text: str) -> dict:
    """{'tags': {tag: value}, 'loops': [ {columns: [...], rows: [[...]]} ]} (all save frames)."""
    tags: dict = {}
    loops: list = []
    it = iter(list(_tokens(text)))
    pending = None
    for tok in it:
        if tok == "loop_":
            cols, rows, row = [], [], []
            for t in it:
                if t.startswith("_") and not rows and not row:
                    cols.append(t)
                    continue
                if t == "stop_":
                    break
                row.append(t)
                if len(row) == len(cols):
                    rows.append(row)
                    row = []
            loops.append({"columns": cols, "rows": rows})
        elif tok.startswith("_"):
            pending = tok
        elif pending is not None:
            tags.setdefault(pending, tok)
            pending = None
    return {"tags": tags, "loops": loops}


def _loop(star: dict, first_col: str) -> list[dict]:
    out = []
    for lp in star["loops"]:
        if lp["columns"] and lp["columns"][0].startswith(first_col):
            cols = [c.split(".", 1)[1] for c in lp["columns"]]
            out += [dict(zip(cols, r)) for r in lp["rows"]]
    return out


SOLVENTS = {"D2O": "D2O", "H2O": "D2O", "CDCL3": "CDCl3", "CHLOROFORM": "CDCl3",
            "DMSO": "DMSO-d6", "METHANOL": "CD3OD", "CD3OD": "CD3OD", "ACETONE": "acetone-d6"}


def record_from_star(text: str, eid: str):
    """(record dict or None, reason)."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    star = parse_star(text)
    atoms = _loop(star, "_Chem_comp_atom.")
    bonds = _loop(star, "_Chem_comp_bond.")
    shifts = _loop(star, "_Atom_chem_shift.")
    if not atoms or not bonds or not shifts:
        return None, "no structure or shifts"
    comps = {a.get("Comp_ID") for a in atoms}
    if len(comps) != 1:
        return None, "several components"
    rw = Chem.RWMol()
    index = {}
    for a in atoms:
        sym = a.get("Type_symbol", "").capitalize()
        try:
            atom = Chem.Atom(sym)
        except Exception:  # noqa: BLE001
            return None, f"element {sym}"
        index[a["Atom_ID"]] = rw.AddAtom(atom)
    orders = {"SING": Chem.BondType.SINGLE, "DOUB": Chem.BondType.DOUBLE,
              "TRIP": Chem.BondType.TRIPLE, "AROM": Chem.BondType.AROMATIC}
    for b in bonds:
        if any(w in b.get("Type", "").lower() for w in ("hydrog", "metal", "coordin", "ionic")):
            return None, "non-covalent bond"     # 'covalent', 'carbonyl', 'amide', 'ester' are fine
        o = orders.get(b.get("Value_order", "").upper()[:4])
        i, j = index.get(b.get("Atom_ID_1")), index.get(b.get("Atom_ID_2"))
        if o is None or i is None or j is None or i == j:
            return None, "bad bond"
        if rw.GetBondBetweenAtoms(i, j) is None:
            rw.AddBond(i, j, o)
    mol = rw.GetMol()
    # charges are not in the atom loop: let RDKit assign them for N(4 bonds) / O(1 bond, no H)
    for at in mol.GetAtoms():
        at.SetNoImplicit(True)
        val = sum(b.GetBondTypeAsDouble() for b in at.GetBonds())
        s = at.GetSymbol()
        if s == "N" and val == 4:
            at.SetFormalCharge(1)
        elif s == "O" and val == 1 and not any(n.GetSymbol() == "H" for n in at.GetNeighbors()) \
                and all(b.GetBondTypeAsDouble() == 1 for b in at.GetBonds()):
            at.SetFormalCharge(-1)
    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:  # noqa: BLE001
        return None, f"sanitize {exc}"
    if len(Chem.GetMolFrags(mol)) != 1:
        return None, "several fragments"
    # shifts per explicit atom index
    vals: dict[int, list[float]] = {}
    for s in shifts:
        aid = s.get("Atom_ID")
        if aid not in index:
            continue
        try:
            v = float(s.get("Val"))
        except (TypeError, ValueError):
            continue
        vals.setdefault(index[aid], []).append(v)
    c13, h1 = {}, {}
    for i, vs in vals.items():
        sym = mol.GetAtomWithIdx(i).GetSymbol()
        v = sum(vs) / len(vs)
        if sym == "C":
            c13[i] = v
        elif sym == "H":
            h1[i] = v
    if not c13 and not h1:
        return None, "no C/H shifts"
    # sanity checks against mis-assignment
    for i, v in c13.items():
        a = mol.GetAtomWithIdx(i)
        het = any(n.GetSymbol() not in ("C", "H") for n in a.GetNeighbors())
        if not (-5 <= v <= 240):
            return None, "13C out of range"
        if a.GetHybridization() == Chem.HybridizationType.SP3 and not het and v > 100:
            return None, "sp3 C above 100 ppm"
        if any(b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(a).GetSymbol() == "O"
               for b in a.GetBonds()) and v < 150:
            return None, "C=O below 150 ppm"
    for i, v in h1.items():
        if not (-1 <= v <= 16):
            return None, "1H out of range"
    tags = star["tags"]
    solvent = ""
    for comp in _loop(star, "_Sample_component."):
        if comp.get("Type", "").lower() == "solvent":
            name = comp.get("Mol_common_name", "").upper()
            for k, s in SOLVENTS.items():
                if k in name:
                    solvent = s
                    break
            if solvent:
                break
    ph = ""
    for cond in _loop(star, "_Sample_condition_variable."):
        if cond.get("Type", "").lower() == "ph":
            ph = cond.get("Val", "")
    noh = Chem.RemoveHs(mol, sanitize=True)
    smiles = Chem.MolToSmiles(noh)
    return {"id": eid, "name": tags.get("_Chem_comp.Name", ""), "molblock":
            Chem.MolToMolBlock(mol), "smiles": smiles, "inchikey": Chem.MolToInchiKey(noh),
            "c13": {str(k): v for k, v in c13.items()}, "h1": {str(k): v for k, v in h1.items()},
            "solvent": solvent or "?", "ph": ph}, ""


def read_records() -> list[dict]:
    recs, why = [], {}
    seen = set()
    for path in sorted((RAW / "bmrb").glob("bmse*.str")):
        try:
            rec, reason = record_from_star(path.read_text("utf-8", errors="replace"), path.stem)
        except Exception as exc:  # noqa: BLE001
            rec, reason = None, f"error {type(exc).__name__}"
        if rec is None:
            key = reason.split(" ")[0] if reason.startswith(("sanitize", "error")) else reason
            why[key] = why.get(key, 0) + 1
            continue
        if rec["inchikey"] in seen:
            why["duplicate compound"] = why.get("duplicate compound", 0) + 1
            continue
        seen.add(rec["inchikey"])
        recs.append(rec)
    print(f"{len(recs)} records; dropped: {why}")
    return recs


def _is_test(inchikey: str) -> bool:
    h = hashlib.sha1(inchikey[:14].encode()).digest()[0]
    return h < 0.2 * 256


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    args = ap.parse_args()
    if args.download:
        download()
        return
    from rdkit import Chem

    from rsstatchem.core.chem_draw import nmr_predict as npd

    recs = read_records()
    for r in recs:
        r["mol"] = Chem.MolFromMolBlock(r["molblock"], removeHs=False)
    train = [r for r in recs if not _is_test(r["inchikey"])]
    test = [r for r in recs if _is_test(r["inchikey"])]
    print(f"train {len(train)}, test {len(test)}")
    t0 = time.time()
    table_tr = npd.build_table(train)
    bench = npd.benchmark(table_tr, test)
    print(json.dumps(bench, indent=1))
    table = npd.build_table(recs)
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"source": "BMRB metabolomics (CC0, wwPDB advisory)", "n_compounds": len(recs),
            "n_train": len(train), "n_test": len(test),
            "solvents": {s: sum(1 for r in recs if r["solvent"] == s)
                         for s in sorted({r["solvent"] for r in recs})},
            "benchmark": bench, "built": time.strftime("%Y-%m-%d"),
            "max_radius": npd.MAX_RADIUS}
    npd.save_table(table, OUT / "hose_shifts.npz", meta)
    (OUT / "benchmark.json").write_text(json.dumps(meta, indent=1), "utf-8")
    print(f"written in {time.time() - t0:.0f} s: {OUT / 'hose_shifts.npz'}")


if __name__ == "__main__":
    main()
