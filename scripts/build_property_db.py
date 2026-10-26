"""Build the offline database of *experimental* physical properties shown first in the Chemical
Drawing module's Chemical Properties window (``rsstatchem/resources/properties``).

Only openly licensed or public-domain data are used; every value keeps its source. See
``rsstatchem/resources/properties/README.md`` for the source list, licences and what was left
out (and why). Run from the project root::

    python scripts/build_property_db.py --download        # fetch raw data into build/propdb-raw
    python scripts/build_property_db.py                   # rebuild from the raw data

The raw downloads (~100 MB) stay in ``build/`` (git-ignored); only the compact processed file
``rsstatchem/resources/properties/exp_props.tsv.xz`` is committed. ``--training`` also writes
``build/propdb-raw/training.tsv`` (structure + consensus values), the input of
``scripts/train_property_models.py``.

Processing
----------
* Structures are read with RDKit; the key is the full standard InChIKey of the structure *as
  recorded by the source* (salts stay salts: a sodium salt's melting point is not attached to
  the free acid). Records whose structure cannot be read are dropped.
* Units are normalised: temperatures in K, logP (octanol/water) unitless, water solubility as
  log10(mol/L) near 25 degC, vapour pressure as log10(Pa) at 25 degC, critical pressure in bar.
* PubChem annotation texts are parsed strictly (a single value or range with a unit; ranges give
  their midpoint; boiling points only at ~1 atm; anything marked estimated / calculated, at
  another pressure, for another form or salt, a sublimation or decomposition *instead* of a
  value, is rejected).
* Several values for one compound: the median is used, the range (lowest - highest) and the
  number of values are kept, and every contributing source is listed.
"""
from __future__ import annotations

import argparse
import csv
import glob
import io
import json
import lzma
import math
import re
import statistics
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "build" / "propdb-raw"
OUT = ROOT / "rsstatchem" / "resources" / "properties" / "exp_props.tsv.xz"

BRADLEY_URL = "https://ndownloader.figshare.com/files/1503990"   # doi:10.6084/m9.figshare.1031637
OPERA_URL = "https://github.com/NIEHS/OPERA/raw/master/OPERA_Data.zip"
OPERA_FILES = ("MP_QR.sdf", "BP_QR.sdf", "LogP_QR.sdf", "WS_QR.sdf", "VP_QR.sdf")
AQSOLDB_URL = "https://dataverse.harvard.edu/api/access/datafile/3407241?format=original"
PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PUG_VIEW = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view"
HEADINGS = ("Melting Point", "Boiling Point", "LogP", "Vapor Pressure",
            "Critical Temperature & Pressure", "Other Experimental Properties")

# source code -> (short label shown in the app, licence)
SOURCES = {
    "B": ("Bradley Open Melting Point Dataset", "CC0 1.0"),
    "O": ("PHYSPROP via US EPA OPERA", "MIT (OPERA repository)"),
    "A": ("AqSolDB", "CC0 1.0"),
    "H": ("HSDB (NLM) via PubChem", "public domain (US NLM)"),
    "I": ("ILO-WHO ICSC via PubChem", "CC BY 4.0"),
    "N": ("NIOSH via PubChem", "public domain (US CDC/NIOSH)"),
    "D": ("DOE PAC database via PubChem", "public domain (US DOE)"),
}
# PubChem source name -> code; every other PubChem source is left out (licence restricts reuse
# or is not stated): HMDB, DrugBank, CAMEO Chemicals, JECFA, Haz-Map, OSHA (pages may contain
# third-party copyrighted content), SangsterLogP (no licence stated), EU Food Improvement
# Agents, individual contributors' annotations.
PUBCHEM_SOURCES = {
    "Hazardous Substances Data Bank (HSDB)": "H",
    "ILO-WHO International Chemical Safety Cards (ICSCs)": "I",
    "The National Institute for Occupational Safety and Health (NIOSH)": "N",
    "PAC Chemical Database, U.S. Department of Energy": "D",
}
PROPS = ("mp", "bp", "logp", "logs", "logvp", "tc", "pc", "vc")
# spread above which a compound's values are flagged as conflicting
CONFLICT = {"mp": 10.0, "bp": 10.0, "logp": 0.5, "logs": 0.5, "logvp": 0.5, "tc": 10.0,
            "pc": 3.0, "vc": 20.0}
# plausible ranges (anything outside is a parsing or unit error)
RANGE = {"mp": (10.0, 1200.0), "bp": (20.0, 1200.0), "logp": (-8.0, 12.0),
         "logs": (-14.0, 2.5), "logvp": (-12.0, 8.0), "tc": (100.0, 1500.0),
         "pc": (1.0, 200.0), "vc": (40.0, 3000.0)}


# --- download -----------------------------------------------------------------------------

def _ssl():
    import ssl
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _get(url: str, data: bytes | None = None, timeout: float = 300.0) -> bytes:
    req = urllib.request.Request(url, data=data,
                                 headers={"User-Agent": "RS-StatChem property-db build"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=_ssl()) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001 - network hiccups: retry
            if attempt == 5:
                raise
            print(f"  retry ({exc})")
            time.sleep(3 * (attempt + 1))
    raise RuntimeError("unreachable")


def download(raw: Path) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    f = raw / "BradleyMeltingPointDataset.xlsx"
    if not f.exists():
        print("Bradley Open Melting Point Dataset (figshare, CC0)")
        f.write_bytes(_get(BRADLEY_URL))
    if not all((raw / "opera" / n).exists() for n in OPERA_FILES):
        print("OPERA data (github.com/NIEHS/OPERA, ~80 MB)")
        blob = _get(OPERA_URL)
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            (raw / "opera").mkdir(exist_ok=True)
            for n in OPERA_FILES:
                (raw / "opera" / n).write_bytes(z.read(n))
    f = raw / "aqsoldb.csv"
    if not f.exists():
        print("AqSolDB (Harvard Dataverse doi:10.7910/DVN/OVHAW8, CC0)")
        f.write_bytes(_get(AQSOLDB_URL))
    pc = raw / "pubchem"
    pc.mkdir(exist_ok=True)
    for heading in HEADINGS:
        page = 1
        while True:
            fn = pc / f"{heading.replace(' ', '_')}_{page}.json"
            if not fn.exists():
                url = (f"{PUG_VIEW}/annotations/heading/JSON?heading="
                       f"{urllib.parse.quote(heading)}&heading_type=Compound&page={page}")
                fn.write_bytes(_get(url))
                time.sleep(0.4)
            total = json.loads(fn.read_text("utf-8"))["Annotations"]["TotalPages"]
            print(f"  PubChem '{heading}' page {page}/{total}", end="\r")
            if page >= total:
                break
            page += 1
        print()
    _download_cid_structures(raw)


def _download_cid_structures(raw: Path) -> None:
    """SMILES of every compound with a kept PubChem annotation (PUG-REST, 500 per request)."""
    out = raw / "pubchem_cids.tsv"
    have = {}
    if out.exists():
        for line in out.read_text("utf-8").splitlines():
            cid, _, smi = line.partition("\t")
            have[cid] = smi
    cids = sorted({str(c) for _src, cids, _h, _t in _pubchem_annotations(raw) for c in cids}
                  - set(have), key=int)
    with open(out, "a", encoding="utf-8") as fh:
        for i in range(0, len(cids), 500):
            chunk = cids[i:i + 500]
            data = ("cid=" + ",".join(chunk)).encode()
            text = _get(f"{PUG}/compound/cid/property/SMILES/TXT", data=data).decode()
            for cid, smi in zip(chunk, text.splitlines()):
                fh.write(f"{cid}\t{smi.strip()}\n")
            print(f"  PubChem structures {i + len(chunk)}/{len(cids)}", end="\r")
            time.sleep(0.3)
    print()


# --- reading the sources --------------------------------------------------------------------

_KEYS: dict[str, tuple[str, str] | None] = {}


def key_of(smiles: str):
    """(InChIKey, canonical SMILES) of a structure, or None when RDKit cannot read it."""
    if smiles in _KEYS:
        return _KEYS[smiles]
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    out = None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None and mol.GetNumAtoms():
            key = Chem.MolToInchiKey(mol)
            if key:
                out = (key, Chem.MolToSmiles(mol))
    except Exception:  # noqa: BLE001 - one bad record must not stop the build
        out = None
    _KEYS[smiles] = out
    return out


def _sdf_records(path: Path):
    """Data items of every record of an SD file (structure block ignored)."""
    rec: dict[str, str] = {}
    field = None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("$$$$"):
                yield rec
                rec, field = {}, None
            elif line.startswith("> "):
                m = re.match(r">\s*<([^>]+)>", line)
                field = m.group(1) if m else None
                if field:
                    rec[field] = ""
            elif field is not None:
                if line == "":
                    field = None
                else:
                    rec[field] = (rec[field] + "\n" + line).strip()


def read_bradley(raw: Path):
    import pandas as pd
    df = pd.read_excel(raw / "BradleyMeltingPointDataset.xlsx")
    for row in df.itertuples():
        if isinstance(row.donotuse, str) and row.donotuse.strip():
            continue                      # flagged by the curators
        try:
            t = float(row.mpC)
        except (TypeError, ValueError):
            continue
        if isinstance(row.smiles, str) and math.isfinite(t):
            yield row.smiles, "mp", t + 273.15, "B"


def read_opera(raw: Path):
    """Curated PHYSPROP training/test sets of OPERA (Mansouri et al. 2018). The structure as
    recorded (Original_SMILES) is used, not the desalted 'QSAR-ready' form."""
    spec = {"MP_QR.sdf": ("mp", "MP"), "BP_QR.sdf": ("bp", "BP"),
            "LogP_QR.sdf": ("logp", "LogP"), "WS_QR.sdf": ("logs", "LogMolar"),
            "VP_QR.sdf": ("logvp", "LogVP")}
    for fname, (prop, field) in spec.items():
        for rec in _sdf_records(raw / "opera" / fname):
            smi = rec.get("Original_SMILES", "")
            try:
                v = float(rec.get(field, ""))
            except ValueError:
                continue
            if prop in ("mp", "bp"):
                v += 273.15                         # degC -> K
            elif prop == "logvp":
                v += math.log10(133.322368)         # log10 mmHg -> log10 Pa
            if prop in ("logs", "logvp"):
                t = rec.get("WS Temperature" if prop == "logs" else "VP Temperature", "")
                try:
                    if t and not 20.0 <= float(t) <= 30.0:
                        continue
                except ValueError:
                    pass
            if smi:
                yield smi, prop, v, "O"


def read_aqsoldb(raw: Path):
    with open(raw / "aqsoldb.csv", encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
            try:
                v = float(row["Solubility"])
            except (KeyError, ValueError):
                continue
            if row.get("SMILES"):
                yield row["SMILES"], "logs", v, "A"


def _pubchem_annotations(raw: Path):
    """(source code, CIDs, heading, text) of every kept PubChem annotation value."""
    for fn in sorted(glob.glob(str(raw / "pubchem" / "*.json"))):
        heading = Path(fn).stem.rsplit("_", 1)[0].replace("_", " ")
        for ann in json.loads(Path(fn).read_text("utf-8"))["Annotations"]["Annotation"]:
            code = PUBCHEM_SOURCES.get(ann.get("SourceName", ""))
            cids = ann.get("LinkedRecords", {}).get("CID") or []
            if code is None or len(cids) != 1:
                continue                      # unlicensed source, or not one compound
            for item in ann.get("Data", []):
                val = item.get("Value", {})
                for s in val.get("StringWithMarkup", []):
                    text = s.get("String", "")
                    if text:
                        yield code, cids, heading, text
                if "Number" in val and len(val["Number"]) == 1:
                    yield code, cids, heading, f"{val['Number'][0]} {val.get('Unit', '')}"


_NUM = r"[-−–]?\d+(?:\.\d+)?"
_REJECT = re.compile(r"estimat|calculat|predict|qsar|model|extrapolat|approx|about|ca\.|"
                     r"isomer|form|salt|hydrate|chloride|bromide|sulfate|sodium|potassium|"
                     r"mixture|technical|grade|solution|commercial|sublim|explod|"
                     r" and |;|>|<|≥|≤|above|below|greater|less|between|range|/[a-z]", re.I)
_DECOMP = re.compile(r"\(\s*(?:with\s+|slight\s+)?(?:dec\.?|decomp(?:\.|oses|osition)?)\s*\)|"
                     r"\s+(?:with\s+)?decomp(?:oses|osition)?\.?$", re.I)


def _num(s: str) -> float:
    return float(s.replace("−", "-").replace("–", "-"))


def _temperature(text: str, allow_pressure: bool):
    """Kelvin from a strict 'value [unit]' text (range -> midpoint), else None."""
    t = text.strip().rstrip(".")
    t = re.sub(r"^(?:MP|BP|M\.?P\.?|B\.?P\.?|MELTING POINT|BOILING POINT)\s*[:=]?\s*", "", t,
               flags=re.I)
    t = re.sub(r"\(\s*(?:OECD|EU|EEC|ASTM)[^)]*\)", "", t)       # test guideline citations
    t = re.sub(r"\(\s*triple point\s*\)", "", t, flags=re.I)
    if allow_pressure and re.search(r"decomp|\bdec\b", t, flags=re.I):
        return None                         # decomposes: not a boiling point
    t = _DECOMP.sub("", t).strip()          # melting with decomposition is still a m.p.
    pressure_ok = True
    m = re.search(r"\s*(?:@|at)\s*(" + _NUM + r")\s*(\[?\s*mm\s*hg\s*\]?|torr|kpa|atm|mbar|"
                  r"hpa|bar)\s*$", t, flags=re.I)
    if m:
        p, unit = _num(m.group(1)), m.group(2).lower().replace("[", "").replace("]", "")
        unit = unit.replace(" ", "")
        mmhg = {"mmhg": p, "torr": p, "kpa": p * 7.50062, "atm": p * 760.0,
                "mbar": p * 0.750062, "hpa": p * 0.750062, "bar": p * 750.062}[unit]
        pressure_ok = 740.0 <= mmhg <= 780.0
        t = t[:m.start()].strip()
    if not pressure_ok or (m and not allow_pressure):
        return None
    if _REJECT.search(t):
        return None
    m = re.fullmatch(r"(" + _NUM + r")\s*(?:(?:-|–|to)\s*(" + _NUM + r"))?\s*"
                     r"(°\s*C|deg\s*C|°\s*F|deg\s*F|K)", t, flags=re.I)
    if not m:
        return None
    lo = _num(m.group(1))
    hi = _num(m.group(2)) if m.group(2) else lo
    if hi < lo or hi - lo > 15:
        return None
    v = (lo + hi) / 2
    unit = m.group(3).upper().replace(" ", "").replace("DEG", "°")
    if unit == "°C":
        return v + 273.15
    if unit == "°F":
        return (v - 32.0) * 5.0 / 9.0 + 273.15
    return v


def _logp(text: str):
    t = text.strip().rstrip(".")
    t = re.sub(r"^(?:log\s*(?:kow|pow|p|k\s*ow)|octanol[ /-]*water[^:=]*)\s*[:=]?\s*", "", t,
               flags=re.I)
    t = re.sub(r"\s*(?:@|at)\s*\d+(?:\.\d+)?\s*°\s*C\s*$", "", t)      # temperature: fine
    t = re.sub(r"\s*\(\s*(?:measured|experimental|exptl\.?)\s*\)\s*$", "", t, flags=re.I)
    if _REJECT.search(t) or re.search(r"\bpH\b", t, flags=re.I):
        return None
    m = re.fullmatch(r"(" + _NUM + r")(?:\s*(?:-|–|to)\s*(" + _NUM + r"))?", t)
    if not m:
        return None
    lo = _num(m.group(1))
    hi = _num(m.group(2)) if m.group(2) else lo
    if hi < lo or hi - lo > 0.6:
        return None
    return (lo + hi) / 2


_SCI = _NUM + r"(?:\s*[xX×]\s*10\s*[-−+]?\s*\d+)?"


def _sci(s: str) -> float:
    s = re.sub(r"\s", "", s).replace("−", "-").replace("×", "x").lower()
    if "x10" in s:
        base, exp = s.split("x10")
        return _num(base) * 10 ** int(exp.replace("+", ""))
    return _num(s)


def _critical(text: str):
    """[(prop, value)] for 'Critical temperature: ...; critical pressure: ...; critical
    volume: ...' texts (HSDB). Parts naming another form (/.../) are skipped."""
    out = []
    for part in text.split(";"):
        p = part.strip().rstrip(".")
        if not p or "/" in p.replace("cu cm/mol", "").replace("L/mol", ""):
            continue
        if re.search(r"estimat|calculat|predict", p, flags=re.I):
            continue
        tail = r"\s*(?:$|=|\()"
        m = re.match(r"cr\w*\s+temp\w*\s*[:=]?\s*(" + _NUM + r")(?:\s*-\s*(" + _NUM + r"))?\s*"
                     r"(°\s*C|°\s*F|deg\s*K|K)" + tail, p, flags=re.I)
        if m:
            lo = _num(m.group(1))
            hi = _num(m.group(2)) if m.group(2) else lo
            v, unit = (lo + hi) / 2, m.group(3).replace(" ", "").upper()
            if hi - lo <= 10:
                out.append(("tc", v + 273.15 if unit == "°C" else
                            (v - 32) * 5 / 9 + 273.15 if unit == "°F" else v))
            continue
        m = re.match(r"cr\w*\s+press\w*\s*[:=]?\s*(" + _SCI + r")\s*"
                     r"(atm|bar|kpa|mpa|pa|mm\s*hg|psia?)" + tail, p, flags=re.I)
        if m:
            unit = m.group(2).lower().replace(" ", "")
            bar = {"atm": 1.01325, "bar": 1.0, "kpa": 0.01, "mpa": 10.0, "pa": 1e-5,
                   "mmhg": 1.01325 / 760, "psi": 0.0689476, "psia": 0.0689476}[unit]
            out.append(("pc", bar * _sci(m.group(1))))
            continue
        m = re.match(r"cr\w*\s+(?:molar\s+)?vol\w*\s*[:=]?\s*(" + _NUM + r")\s*"
                     r"(cu\s*cm\s*/\s*mole?|cm3\s*/\s*mol|L\s*/\s*mol)" + tail, p, flags=re.I)
        if m:
            v = _num(m.group(1))
            out.append(("vc", v * 1000.0 if m.group(2).lower().startswith("l") else v))
    return out


def _vapour_pressure(text: str):
    """log10(Pa) at 25 degC from texts like '1.3 mm Hg at 25 °C' (only 25 degC values)."""
    t = text.strip().rstrip(".")
    units = r"(\[?\s*mm\s*Hg\s*\]?|torr|Pa|kPa|atm|mbar|hPa)"
    m = re.fullmatch(r"(" + _SCI + r")\s*" + units + r"\s*(?:@|at)\s*25(?:\.0)?\s*°\s*C", t,
                     flags=re.I)
    if m:
        v, unit = _sci(m.group(1)), m.group(2)
    else:                                   # ICSC: "Vapor pressure, Pa at 25 °C: 2.9"
        m = re.fullmatch(r"vapou?r pressure,\s*" + units + r"\s*at\s*25\s*°\s*C\s*:\s*(" +
                         _SCI + r")", t, flags=re.I)
        if not m:
            return None
        v, unit = _sci(m.group(2)), m.group(1)
    if v <= 0:
        return None
    unit = unit.lower().replace(" ", "").replace("[", "").replace("]", "")
    pa = {"mmhg": 133.322368, "torr": 133.322368, "pa": 1.0, "kpa": 1000.0,
          "atm": 101325.0, "mbar": 100.0, "hpa": 100.0}[unit]
    return math.log10(v * pa)


def read_pubchem(raw: Path):
    smiles = {}
    f = raw / "pubchem_cids.tsv"
    if f.exists():
        for line in f.read_text("utf-8").splitlines():
            cid, _, smi = line.partition("\t")
            smiles[int(cid)] = smi
    for code, cids, heading, text in _pubchem_annotations(raw):
        smi = smiles.get(int(cids[0]))
        if not smi:
            continue
        if heading == "Melting Point":
            v = _temperature(text, allow_pressure=False)
            if v is not None:
                yield smi, "mp", v, code
        elif heading == "Boiling Point":
            v = _temperature(text, allow_pressure=True)
            if v is not None:
                yield smi, "bp", v, code
        elif heading == "LogP":
            v = _logp(text)
            if v is not None:
                yield smi, "logp", v, code
        elif heading == "Vapor Pressure":
            v = _vapour_pressure(text)
            if v is not None:
                yield smi, "logvp", v, code
        elif heading in ("Other Experimental Properties", "Critical Temperature & Pressure"):
            for prop, v in _critical(text):
                yield smi, prop, v, code


# --- aggregation ------------------------------------------------------------------------------

def collect(raw: Path):
    """{inchikey: {prop: [(value, source)]}} and {inchikey: canonical SMILES}."""
    data: dict = defaultdict(lambda: defaultdict(list))
    smiles: dict = {}
    counts: dict = defaultdict(int)
    readers = [("Bradley", read_bradley), ("OPERA", read_opera), ("AqSolDB", read_aqsoldb),
               ("PubChem", read_pubchem)]
    for name, reader in readers:
        n = 0
        for smi, prop, v, code in reader(raw):
            lo, hi = RANGE[prop]
            if not (lo <= v <= hi):
                continue
            k = key_of(smi)
            if k is None:
                continue
            data[k[0]][prop].append((v, code))
            smiles.setdefault(k[0], k[1])
            counts[(code, prop)] += 1
            n += 1
        print(f"{name}: {n} values")
    return data, smiles, counts


def consensus(values: list[tuple[float, str]]):
    """(median, low, high, n, sources) of one compound's values for one property."""
    vs = sorted(v for v, _ in values)
    srcs = "".join(sorted({c for _, c in values}, key="BOAHIND".index))
    return statistics.median(vs), vs[0], vs[-1], len(vs), srcs


def _fmt(prop: str, v: float) -> str:
    return f"{v:.1f}" if prop in ("mp", "bp", "tc") else f"{v:.2f}"


def write_db(data: dict, out: Path, counts: dict) -> dict:
    stats = defaultdict(int)
    lines = []
    for key in sorted(data):
        cells = []
        for prop in PROPS:
            vals = data[key].get(prop)
            if not vals:
                cells.append("")
                continue
            med, lo, hi, n, srcs = consensus(vals)
            cell = f"{_fmt(prop, med)}|{srcs}|{n}"
            if n > 1 and hi - lo > 1e-9:
                cell += f"|{_fmt(prop, lo)}|{_fmt(prop, hi)}"
            cells.append(cell)
            stats[prop] += 1
            if hi - lo > CONFLICT[prop]:
                stats[prop + "_conflict"] += 1
        lines.append(key + "\t" + "\t".join(cells))
    header = ["# RS-StatChem experimental property database (scripts/build_property_db.py)",
              "# columns: inchikey, " + ", ".join(PROPS),
              "# units: mp/bp/tc K; logp; logs log10(mol/L) ~25 C; logvp log10(Pa) 25 C; "
              "pc bar; vc cm3/mol",
              "# cell: median|sources|n[|lowest|highest]"]
    for code, (label, lic) in SOURCES.items():
        header.append(f"# source {code}\t{label}\t{lic}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with lzma.open(out, "wt", encoding="utf-8", preset=9 | lzma.PRESET_EXTREME) as fh:
        fh.write("\n".join(header) + "\n")
        fh.write("\n".join(lines) + "\n")
    stats["compounds"] = len(lines)
    return stats


def write_training(data: dict, smiles: dict, out: Path) -> None:
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("inchikey\tsmiles\t" + "\t".join(PROPS) + "\t" +
                 "\t".join(p + "_n" for p in PROPS) + "\n")
        for key in sorted(data):
            vals, ns = [], []
            for prop in PROPS:
                v = data[key].get(prop)
                if v:
                    med, lo, hi, n, _ = consensus(v)
                    vals.append(f"{med:.4f}")
                    ns.append(f"{n}:{hi - lo:.3f}")
                else:
                    vals.append("")
                    ns.append("")
            fh.write(f"{key}\t{smiles[key]}\t" + "\t".join(vals) + "\t" + "\t".join(ns) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw", type=Path, default=RAW, help="raw data folder")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--download", action="store_true", help="download the raw data first")
    ap.add_argument("--training", action="store_true",
                    help="also write <raw>/training.tsv for scripts/train_property_models.py")
    args = ap.parse_args(argv)
    if args.download:
        download(args.raw)
    data, smiles, counts = collect(args.raw)
    stats = write_db(data, args.out, counts)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.2f} MB): "
          f"{stats['compounds']} compounds")
    for prop in PROPS:
        print(f"  {prop:6s} {stats[prop]:6d} compounds ({stats[prop + '_conflict']} with values "
              f"spread over more than {CONFLICT[prop]:g})")
    for (code, prop), n in sorted(counts.items()):
        print(f"  source {code} {prop:6s} {n:6d} values")
    if args.training:
        write_training(data, smiles, args.raw / "training.tsv")
        print(f"wrote {args.raw / 'training.tsv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
