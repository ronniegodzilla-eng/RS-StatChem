"""Structure <-> name for ChemDraw's *Convert Structure to Name* / *Convert Name to Structure*.

RDKit has no IUPAC naming, so names come from two honest sources and are never generated:

1. an offline dictionary of common compounds (name, systematic name, SMILES) plus the
   template library; structures are matched by their full InChIKey (so stereo matters);
2. optionally PubChem PUG-REST (https://pubchem.ncbi.nlm.nih.gov/rest/pug): the IUPAC name
   and title of a compound by InChIKey, or a 2D structure (SDF) for a name.

The network functions block, so callers run them off the UI thread. No Qt here.
"""
from __future__ import annotations

import json
import ssl
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from functools import lru_cache

__all__ = ["COMMON_COMPOUNDS", "NameResult", "NameLookupError", "offline_name",
           "offline_structure", "pubchem_name", "pubchem_structure", "PUBCHEM"]

PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
TIMEOUT = 12.0

# (common name, systematic name or "", SMILES)
COMMON_COMPOUNDS: list[tuple[str, str, str]] = [
    ("water", "oxidane", "O"), ("ammonia", "azane", "N"),
    ("hydrogen peroxide", "", "OO"), ("carbon dioxide", "", "O=C=O"),
    ("methane", "methane", "C"), ("ethane", "ethane", "CC"), ("propane", "propane", "CCC"),
    ("butane", "butane", "CCCC"), ("pentane", "pentane", "CCCCC"),
    ("hexane", "hexane", "CCCCCC"), ("heptane", "heptane", "CCCCCCC"),
    ("octane", "octane", "CCCCCCCC"), ("isobutane", "2-methylpropane", "CC(C)C"),
    ("ethylene", "ethene", "C=C"), ("propylene", "prop-1-ene", "C=CC"),
    ("acetylene", "ethyne", "C#C"),
    ("cyclopropane", "cyclopropane", "C1CC1"), ("cyclobutane", "cyclobutane", "C1CCC1"),
    ("cyclopentane", "cyclopentane", "C1CCCC1"), ("cyclohexane", "cyclohexane", "C1CCCCC1"),
    ("cycloheptane", "cycloheptane", "C1CCCCCC1"),
    ("cyclooctane", "cyclooctane", "C1CCCCCCC1"), ("cyclohexene", "cyclohexene", "C1=CCCCC1"),
    ("benzene", "benzene", "c1ccccc1"), ("toluene", "methylbenzene", "Cc1ccccc1"),
    ("o-xylene", "1,2-dimethylbenzene", "Cc1ccccc1C"),
    ("m-xylene", "1,3-dimethylbenzene", "Cc1cccc(C)c1"),
    ("p-xylene", "1,4-dimethylbenzene", "Cc1ccc(C)cc1"),
    ("styrene", "ethenylbenzene", "C=Cc1ccccc1"),
    ("naphthalene", "naphthalene", "c1ccc2ccccc2c1"),
    ("anthracene", "anthracene", "c1ccc2cc3ccccc3cc2c1"),
    ("phenanthrene", "phenanthrene", "c1ccc2c(c1)ccc1ccccc12"),
    ("methanol", "methanol", "CO"), ("ethanol", "ethanol", "CCO"),
    ("1-propanol", "propan-1-ol", "CCCO"), ("isopropanol", "propan-2-ol", "CC(C)O"),
    ("1-butanol", "butan-1-ol", "CCCCO"), ("tert-butanol", "2-methylpropan-2-ol", "CC(C)(C)O"),
    ("ethylene glycol", "ethane-1,2-diol", "OCCO"),
    ("glycerol", "propane-1,2,3-triol", "OCC(O)CO"),
    ("phenol", "phenol", "Oc1ccccc1"), ("aniline", "aniline", "Nc1ccccc1"),
    ("anisole", "methoxybenzene", "COc1ccccc1"), ("benzaldehyde", "benzaldehyde", "O=Cc1ccccc1"),
    ("benzoic acid", "benzoic acid", "OC(=O)c1ccccc1"),
    ("acetophenone", "1-phenylethan-1-one", "CC(=O)c1ccccc1"),
    ("nitrobenzene", "nitrobenzene", "[O-][N+](=O)c1ccccc1"),
    ("chlorobenzene", "chlorobenzene", "Clc1ccccc1"),
    ("bromobenzene", "bromobenzene", "Brc1ccccc1"),
    ("benzyl alcohol", "phenylmethanol", "OCc1ccccc1"),
    ("benzonitrile", "benzonitrile", "N#Cc1ccccc1"),
    ("benzamide", "benzamide", "NC(=O)c1ccccc1"),
    ("hydroquinone", "benzene-1,4-diol", "Oc1ccc(O)cc1"),
    ("catechol", "benzene-1,2-diol", "Oc1ccccc1O"),
    ("resorcinol", "benzene-1,3-diol", "Oc1cccc(O)c1"),
    ("p-benzoquinone", "cyclohexa-2,5-diene-1,4-dione", "O=C1C=CC(=O)C=C1"),
    ("formaldehyde", "methanal", "C=O"), ("acetaldehyde", "ethanal", "CC=O"),
    ("acetone", "propan-2-one", "CC(C)=O"), ("butanone", "butan-2-one", "CCC(C)=O"),
    ("cyclohexanone", "cyclohexanone", "O=C1CCCCC1"),
    ("formic acid", "methanoic acid", "OC=O"), ("acetic acid", "ethanoic acid", "CC(=O)O"),
    ("propionic acid", "propanoic acid", "CCC(=O)O"),
    ("butyric acid", "butanoic acid", "CCCC(=O)O"),
    ("oxalic acid", "ethanedioic acid", "OC(=O)C(=O)O"),
    ("succinic acid", "butanedioic acid", "OC(=O)CCC(=O)O"),
    ("lactic acid", "2-hydroxypropanoic acid", "CC(O)C(=O)O"),
    ("citric acid", "2-hydroxypropane-1,2,3-tricarboxylic acid",
     "OC(=O)CC(O)(CC(=O)O)C(=O)O"),
    ("acetamide", "acetamide", "CC(N)=O"), ("acetic anhydride", "acetic anhydride",
                                              "CC(=O)OC(C)=O"),
    ("acetyl chloride", "acetyl chloride", "CC(Cl)=O"),
    ("methyl acetate", "methyl acetate", "COC(C)=O"),
    ("ethyl acetate", "ethyl acetate", "CCOC(C)=O"),
    ("ethyl benzoate", "ethyl benzoate", "CCOC(=O)c1ccccc1"),
    ("methyl salicylate", "methyl 2-hydroxybenzoate", "COC(=O)c1ccccc1O"),
    ("diethyl ether", "ethoxyethane", "CCOCC"), ("tetrahydrofuran", "oxolane", "C1CCOC1"),
    ("1,4-dioxane", "1,4-dioxane", "C1COCCO1"), ("acetonitrile", "acetonitrile", "CC#N"),
    ("dimethyl sulfoxide", "", "CS(C)=O"),
    ("N,N-dimethylformamide", "N,N-dimethylformamide", "CN(C)C=O"),
    ("dichloromethane", "dichloromethane", "ClCCl"),
    ("chloroform", "trichloromethane", "ClC(Cl)Cl"),
    ("carbon tetrachloride", "tetrachloromethane", "ClC(Cl)(Cl)Cl"),
    ("urea", "urea", "NC(N)=O"), ("methylamine", "methanamine", "CN"),
    ("triethylamine", "N,N-diethylethanamine", "CCN(CC)CC"),
    ("pyridine", "pyridine", "c1ccncc1"), ("pyrrole", "1H-pyrrole", "c1cc[nH]c1"),
    ("furan", "furan", "c1ccoc1"), ("thiophene", "thiophene", "c1ccsc1"),
    ("imidazole", "1H-imidazole", "c1c[nH]cn1"), ("indole", "1H-indole", "c1ccc2[nH]ccc2c1"),
    ("quinoline", "quinoline", "c1ccc2ncccc2c1"), ("pyrimidine", "pyrimidine", "c1cncnc1"),
    ("piperidine", "piperidine", "C1CCNCC1"), ("morpholine", "morpholine", "C1COCCN1"),
    ("piperazine", "piperazine", "C1CNCCN1"),
    ("caffeine", "1,3,7-trimethyl-3,7-dihydro-1H-purine-2,6-dione",
     "Cn1cnc2c1c(=O)n(C)c(=O)n2C"),
    ("theobromine", "3,7-dimethyl-3,7-dihydro-1H-purine-2,6-dione",
     "Cn1cnc2c1c(=O)[nH]c(=O)n2C"),
    ("theophylline", "1,3-dimethyl-3,7-dihydro-1H-purine-2,6-dione",
     "Cn1c(=O)c2[nH]cnc2n(C)c1=O"),
    ("aspirin", "2-(acetyloxy)benzoic acid", "CC(=O)Oc1ccccc1C(=O)O"),
    ("salicylic acid", "2-hydroxybenzoic acid", "OC(=O)c1ccccc1O"),
    ("paracetamol", "N-(4-hydroxyphenyl)acetamide", "CC(=O)Nc1ccc(O)cc1"),
    ("acetanilide", "N-phenylacetamide", "CC(=O)Nc1ccccc1"),
    ("ibuprofen", "2-[4-(2-methylpropyl)phenyl]propanoic acid",
     "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
    ("nicotinamide", "pyridine-3-carboxamide", "NC(=O)c1cccnc1"),
    ("vanillin", "4-hydroxy-3-methoxybenzaldehyde", "COc1cc(C=O)ccc1O"),
    ("cinnamaldehyde", "(2E)-3-phenylprop-2-enal", "O=C/C=C/c1ccccc1"),
    ("eugenol", "2-methoxy-4-(prop-2-en-1-yl)phenol", "C=CCc1ccc(O)c(OC)c1"),
    ("quercetin", "2-(3,4-dihydroxyphenyl)-3,5,7-trihydroxy-4H-chromen-4-one",
     "O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12"),
    ("curcumin", "", "COc1cc(/C=C/C(=O)CC(=O)/C=C/c2ccc(O)c(OC)c2)ccc1O"),
]
_SYNONYMS = {"acetylsalicylic acid": "aspirin", "acetaminophen": "paracetamol",
             "isopropyl alcohol": "isopropanol", "2-propanol": "isopropanol",
             "dmso": "dimethyl sulfoxide", "dmf": "N,N-dimethylformamide",
             "thf": "tetrahydrofuran", "dcm": "dichloromethane",
             "methylene chloride": "dichloromethane", "ether": "diethyl ether",
             "acetic acid ethyl ester": "ethyl acetate", "ethyl alcohol": "ethanol",
             "methyl alcohol": "methanol", "ethene": "ethylene", "ethyne": "acetylene"}


@dataclass
class NameResult:
    name: str                    # what to show (systematic name when known)
    common: str = ""
    source: str = ""             # "offline list", "template library" or "PubChem"
    molblock: str = ""           # structure for name -> structure
    smiles: str = ""


class NameLookupError(Exception):
    """Lookup failed; ``offline`` is True when PubChem could not be reached at all."""

    def __init__(self, message: str, offline: bool = False):
        super().__init__(message)
        self.offline = offline


def _entries() -> list[tuple[str, str, str, str]]:
    """(common, systematic, smiles, source) for the offline list and the templates."""
    from rsstatchem.core.chem_draw import templates

    out = [(c, s, smi, "offline list") for c, s, smi in COMMON_COMPOUNDS]
    for cat, items in templates.TEMPLATES.items():
        for name, smi in items:
            if "*" not in smi:
                out.append((name, "", smi, "template library"))
    return out


@lru_cache(maxsize=1)
def _by_key() -> dict[str, tuple[str, str, str, str]]:
    from rsstatchem.core.chem import molecule as mo

    table = {}
    for entry in _entries():
        try:
            key = mo.to_inchikey(mo.mol_from_text(entry[2]))
        except Exception:  # noqa: BLE001 - a bad entry must not break the lookup
            continue
        table.setdefault(key, entry)
    return table


def offline_name(mol) -> NameResult | None:
    """Name of a molecule from the offline dictionary (full InChIKey match) or None."""
    from rsstatchem.core.chem import molecule as mo

    try:
        key = mo.to_inchikey(mol)
    except Exception:  # noqa: BLE001
        return None
    hit = _by_key().get(key)
    if hit is None:
        return None
    common, systematic, smi, source = hit
    return NameResult(systematic or common, common, source, smiles=smi)


def offline_structure(name: str) -> NameResult | None:
    """SMILES for a common name (case-insensitive, a few synonyms) or None."""
    q = " ".join(name.strip().split()).lower()
    if not q:
        return None
    q = _SYNONYMS.get(q, q).lower()
    for common, systematic, smi, source in _entries():
        if q in (common.lower(), systematic.lower()):
            return NameResult(systematic or common, common, source, smiles=smi)
    return None


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:  # pragma: no cover
        return ssl.create_default_context()


def _get(url: str, timeout: float = TIMEOUT) -> bytes:
    from rsstatchem import APP_NAME, __version__

    req = urllib.request.Request(url, headers={"User-Agent": f"{APP_NAME}/{__version__}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise NameLookupError("PubChem has no compound for this query.") from None
        raise NameLookupError(f"PubChem answered HTTP {exc.code}.") from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        raise NameLookupError(f"Could not reach PubChem ({reason}). Check the internet "
                              "connection, or use a name from the offline list.",
                              offline=True) from None


def pubchem_name(inchikey: str, fetch=_get) -> NameResult:
    """IUPAC name and title of the PubChem compound with this InChIKey."""
    url = (f"{PUBCHEM}/compound/inchikey/{urllib.parse.quote(inchikey, safe='')}"
           "/property/IUPACName,Title/JSON")
    try:
        data = json.loads(fetch(url).decode("utf-8"))
        props = data["PropertyTable"]["Properties"][0]
    except NameLookupError:
        raise
    except (ValueError, KeyError, IndexError):
        raise NameLookupError("PubChem sent an unreadable answer.") from None
    iupac = str(props.get("IUPACName") or "")
    title = str(props.get("Title") or "")
    if not (iupac or title):
        raise NameLookupError("PubChem has no name for this structure.")
    return NameResult(iupac or title, title, "PubChem")


def pubchem_structure(name: str, fetch=_get) -> NameResult:
    """2D structure (MOL block) of the PubChem compound with this name."""
    url = (f"{PUBCHEM}/compound/name/{urllib.parse.quote(name.strip(), safe='')}"
           "/SDF?record_type=2d")
    text = fetch(url).decode("utf-8", errors="replace")
    block = text.split("$$$$")[0].rstrip() + "\n"
    if "V2000" not in block and "V3000" not in block:
        raise NameLookupError("PubChem sent an unreadable structure.")
    return NameResult(name.strip(), name.strip(), "PubChem", molblock=block)
