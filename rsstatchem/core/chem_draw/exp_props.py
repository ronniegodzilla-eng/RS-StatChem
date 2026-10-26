"""Experimental physical properties of known compounds from the offline database
``resources/properties/exp_props.tsv.xz`` (built by ``scripts/build_property_db.py`` from openly
licensed / public-domain data; sources and licences in ``resources/properties/README.md``).

Lookup is by InChIKey: the full key first (same stereochemistry, isotopes, salt form); when that
fails, a record with the same connectivity and *unspecified* stereo is used and the result says
so (``ExpValue.stereo_note``). No Qt here.
"""
from __future__ import annotations

import lzma
import threading
from dataclasses import dataclass

__all__ = ["ExpValue", "lookup", "db_path", "db_size", "SOURCES", "PROPS", "source_text"]

PROPS = ("mp", "bp", "logp", "logs", "logvp", "tc", "pc", "vc")
SOURCES: dict[str, tuple[str, str]] = {}      # code -> (label, licence), from the file header


@dataclass
class ExpValue:
    prop: str
    value: float               # median of the recorded values (K, bar, log units, cm3/mol)
    sources: str               # source codes, e.g. "BOH"
    n: int = 1                 # number of recorded values
    low: float | None = None   # lowest / highest recorded value when they differ
    high: float | None = None
    stereo_note: str = ""      # set when matched through a stereo-unspecified record

    @property
    def spread(self) -> float:
        return (self.high - self.low) if self.low is not None and self.high is not None else 0.0

    def source_labels(self) -> list[str]:
        return [SOURCES.get(c, (c, ""))[0] for c in self.sources]


def source_text(codes: str, licences: bool = True) -> str:
    out = []
    for c in codes:
        label, lic = SOURCES.get(c, (c, ""))
        out.append(f"{label} ({lic})" if licences and lic else label)
    return "; ".join(out)


def db_path():
    from rsstatchem.paths import resource
    return resource("properties/exp_props.tsv.xz")


_DB: dict[str, str] | None = None
_BLOCKS: dict[str, str] | None = None
_LOCK = threading.Lock()


def _load() -> tuple[dict, dict]:
    """{inchikey: raw row} and {connectivity block: inchikey of the stereo-unspecified row}."""
    global _DB, _BLOCKS
    with _LOCK:
        if _DB is not None:
            return _DB, _BLOCKS
        db: dict[str, str] = {}
        blocks: dict[str, str] = {}
        try:
            with lzma.open(db_path(), "rt", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("#"):
                        if line.startswith("# source "):
                            parts = line[9:].rstrip("\n").split("\t")
                            if len(parts) == 3:
                                SOURCES[parts[0]] = (parts[1], parts[2])
                        continue
                    key, _, rest = line.rstrip("\n").partition("\t")
                    db[key] = rest
                    if key[15:25] == "UHFFFAOYSA":
                        blocks.setdefault(key[:14], key)
        except (OSError, lzma.LZMAError, EOFError):
            db, blocks = {}, {}
        _DB, _BLOCKS = db, blocks
        return db, blocks


def db_size() -> int:
    return len(_load()[0])


def _parse(row: str) -> dict[str, ExpValue]:
    out = {}
    for prop, cell in zip(PROPS, row.split("\t")):
        if not cell:
            continue
        parts = cell.split("|")
        try:
            v = ExpValue(prop, float(parts[0]), parts[1], int(parts[2]))
            if len(parts) >= 5:
                v.low, v.high = float(parts[3]), float(parts[4])
        except (ValueError, IndexError):
            continue
        out[prop] = v
    return out


def lookup(mol=None, inchikey: str | None = None) -> dict[str, ExpValue]:
    """{property: ExpValue} for a molecule (or InChIKey); empty when it is not in the database.
    """
    if inchikey is None:
        from rsstatchem.core.chem import molecule as mo
        try:
            inchikey = mo.to_inchikey(mol)
        except Exception:  # noqa: BLE001 - no key, no lookup
            return {}
    if not inchikey:
        return {}
    db, blocks = _load()
    row = db.get(inchikey)
    if row is not None:
        return _parse(row)
    alt = blocks.get(inchikey[:14])
    if alt is None or alt == inchikey:
        return {}
    out = _parse(db[alt])
    for v in out.values():
        v.stereo_note = "record without stereochemistry (stereoisomers may differ)"
    return out
