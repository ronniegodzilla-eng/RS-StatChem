"""pKa of the strongest acidic and the strongest basic group (Chemical Properties window).

Open replacement for ChemDraw's licensed MOSES pKa: gradient-boosted trees (numpy inference,
the same :class:`ml_props.GBTrees` format as the melting point / boiling point / logP models)
trained by ``scripts/train_pka_model.py`` on the curated DataWarrior pKa set of
K. Mansouri et al., J. Cheminform. 11 (2019) 60 (OPERA ``pKa_QR.sdf``; CC0 data, MIT code):

- ``pka_a``: the most acidic pKa of the molecule (the proton it loses first);
- ``pka_b``: the most basic site, given as the pKa of its conjugate acid (BH+).

A compound in the training data shows its measured value. Otherwise the model is used, but
only when the structure has a group of a kind that occurs in the data
(:data:`ACIDIC_GROUPS` / :data:`BASIC_GROUPS`, plain SMARTS written for this module): the
data contain only ionisable compounds, so the model alone cannot tell that, say, an alkane
has no pKa in the water range. Each prediction carries its held-out error (5-fold CV RMSE of
compounds with a similar Tanimoto similarity to the training folds) and an applicability-domain
flag, like the other ML estimates. The values are macroscopic, aqueous, 25 °C; no microstates.
No Qt here.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field

import numpy as np

from rsstatchem.core.chem_draw import ml_props

__all__ = ["ModelFile", "PkaValue", "PkaResult", "predict", "ionisable_groups",
           "ACIDIC_GROUPS", "BASIC_GROUPS", "DATA_SOURCE", "pka_models"]

DATA_SOURCE = "Mansouri et al. 2019 (DataWarrior pKa set, OPERA; CC0)"

# (name, SMARTS) - group types present in the training data; a molecule without any of them
# gets no pKa of that kind
ACIDIC_GROUPS = (
    ("carboxylic acid", "[CX3](=O)[OX2H1]"),
    ("phenol / enol", "[OX2H1][c,$(C=C)]"),
    ("thiol", "[SX2H1]"),
    ("sulfonic / sulfinic acid", "[SX4,SX3](=O)[OX2H1]"),
    ("phosphoric / phosphonic acid", "P(=O)[OX2H1]"),
    ("boronic acid", "B[OX2H1]"),
    ("imide / acyl-sulfonamide NH", "[NX3H1]([#6,#16]=O)[#6,#16]=O"),
    ("sulfonamide NH", "[NX3;H1,H2]S(=O)(=O)"),
    ("hydroxamic acid / oxime / N-hydroxy", "[OX2H1][NX3,NX2]"),
    ("tetrazole / triazole / imidazole NH", "[nH;r5]:[n,c]:[n]"),
    ("pyrazole / triazole NH", "[nH;r5]:[n]"),
    ("1,3-dicarbonyl C-H", "[CX4;H1,H2]([CX3]=O)[CX3,$(C#N),$([N+](=O)[O-])]=,#[O,N]"),
    ("nitroalkane C-H", "[CX4;H1,H2][N+](=O)[O-]"),
    ("activated alcohol", "[OX2H1][CX4]C(F)(F)"),
    ("amide / sulfonyl-activated NH", "[NX3H1](c)S(=O)=O"),
)
BASIC_GROUPS = (
    ("aliphatic amine", "[NX3;H2,H1,H0;+0;!$(N[a]);!$(N[#6,#16,#15]=[O,S,N]);!$(N[N,O,S]);"
                        "!$(NC#N);!$(N-C=C-[$(C=O),$(C#N)])]"),
    ("aniline / aryl amine", "[NX3;H2,H1,H0;+0;$(Na);!$(N[#6,#16]=[O,S]);!$(N[N,O])]"),
    ("pyridine-type N", "[nX2;r6]"),
    ("azole N (imidazole, ...)", "[nX2;r5]"),
    ("amidine / guanidine", "[CX3](=[NX2;!$(N[O,N])])[NX3]"),
    ("imine", "[CX3]=[NX2;!$(N[O,N]);!$(N=C[O,S])]"),
    ("hydrazine / hydroxylamine N", "[NX3;H2,H1;!$(NC=O)][NX3,OX2]"),
    ("N-oxide / amine oxide", "[NX4+][O-]"),
)

_PATTERNS: dict | None = None


def _patterns():
    global _PATTERNS
    if _PATTERNS is None:
        from rdkit import Chem
        _PATTERNS = {"acid": [(n, Chem.MolFromSmarts(s)) for n, s in ACIDIC_GROUPS],
                     "base": [(n, Chem.MolFromSmarts(s)) for n, s in BASIC_GROUPS]}
    return _PATTERNS


def ionisable_groups(mol) -> dict[str, list[str]]:
    """{'acid': [group names], 'base': [...]} found in the molecule."""
    out = {"acid": [], "base": []}
    for kind, pats in _patterns().items():
        for name, patt in pats:
            if patt is not None and mol.HasSubstructMatch(patt):
                out[kind].append(name)
    return out


class ModelFile:
    """Gradient-boosted tree models + training fingerprints + experimental values of one
    ``.npz`` file written by ``scripts/train_pka_model.py`` (loaded once, thread-safe)."""

    def __init__(self, name: str, targets: tuple[str, ...]):
        self.name = name
        self.targets = targets
        self._lock = threading.Lock()
        self._models: dict | None = None
        self._fps = None
        self._exp: dict = {}
        self._cache: dict = {}

    def path(self):
        from rsstatchem.paths import resource
        return resource(f"properties/{self.name}")

    def load(self) -> dict:
        with self._lock:
            if self._models is not None:
                return self._models
            out: dict = {}
            try:
                data = np.load(self.path(), allow_pickle=False)
            except (OSError, ValueError):
                self._models = {}
                return self._models
            self._fps = data["fps"] if "fps" in data else np.zeros((0, ml_props.FP_BITS // 8),
                                                                    np.uint8)
            for t in self.targets:
                if f"{t}_feature" not in data:
                    continue
                meta = json.loads(str(data[f"{t}_meta"]))
                trees = ml_props.GBTrees(data[f"{t}_feature"], data[f"{t}_threshold"],
                                         data[f"{t}_left"], data[f"{t}_right"],
                                         data[f"{t}_value"], data[f"{t}_missing_left"],
                                         data[f"{t}_offsets"], meta["baseline"])
                out[t] = ml_props.MLModel(t, trees, meta["metrics"], data[f"{t}_fp_index"])
                keys, vals = f"exp_{t}_keys", f"exp_{t}_values"
                if keys in data:
                    self._exp[t] = dict(zip((str(k) for k in data[keys]),
                                            (float(v) for v in data[vals])))
            self._models = out
            return out

    def experimental(self, target: str, inchikey: str) -> tuple[float | None, str]:
        """(value, note): exact InChIKey first, then the same connectivity (stereo ignored)."""
        self.load()
        table = self._exp.get(target, {})
        if inchikey in table:
            return table[inchikey], ""
        block = inchikey[:14]
        for k, v in table.items():
            if k.startswith(block):
                return v, "value of a stereoisomer / the unspecified form"
        return None, ""

    def predict(self, mol, target: str) -> ml_props.MLPrediction:
        ms = self.load()
        if target not in ms:
            return ml_props.MLPrediction(target, None, error="model file missing")
        why = ml_props.scope_problem(mol)
        if why:
            return ml_props.MLPrediction(target, None, error=why)
        from rdkit import Chem
        key = (target, Chem.MolToSmiles(mol))
        if key in self._cache:
            return self._cache[key]
        try:
            x = ml_props.features(mol)[None, :]
            m = ms[target]
            sims = ml_props.similarities(ml_props.fingerprint(mol), self._fps)
            value = float(m.trees.predict(x)[0])
            sim = float(sims[m.fp_index].max()) if m.fp_index.size else 0.0
            res = ml_props.MLPrediction(target, value, sim, sim >= ml_props.AD_THRESHOLD,
                                        m.expected_rmse(sim), m.metrics)
        except Exception as exc:  # noqa: BLE001 - an estimate must never break drawing
            res = ml_props.MLPrediction(target, None, error=str(exc))
        if len(self._cache) > 256:
            self._cache.clear()
        self._cache[key] = res
        return res

    def metrics(self, target: str) -> dict:
        m = self.load().get(target)
        return m.metrics if m is not None else {}


pka_models = ModelFile("pka_model.npz", ("pka_a", "pka_b"))


@dataclass
class PkaValue:
    kind: str                         # "acid" or "base"
    value: float | None
    source: str = ""                  # "experimental" | "ML" | ""
    note: str = ""                    # groups found / why missing / stereo note
    prediction: ml_props.MLPrediction | None = None
    groups: list = field(default_factory=list)

    @property
    def experimental(self) -> bool:
        return self.source == "experimental"


@dataclass
class PkaResult:
    acid: PkaValue
    base: PkaValue


def predict(mol) -> PkaResult:
    """Strongest acidic and basic pKa (aqueous, 25 °C) of one neutral molecule."""
    from rdkit import Chem
    out = {}
    try:
        key = Chem.MolToInchiKey(mol)
    except Exception:  # noqa: BLE001
        key = ""
    groups = ionisable_groups(mol) if mol is not None else {"acid": [], "base": []}
    for kind, target in (("acid", "pka_a"), ("base", "pka_b")):
        exp, note = pka_models.experimental(target, key) if key else (None, "")
        # the data set has a few values filed under the wrong kind (phenol as a "base"):
        # a measured value is shown only when the structure has a group of that kind
        if exp is not None and groups[kind]:
            out[kind] = PkaValue(kind, exp, "experimental", note or DATA_SOURCE,
                                 groups=groups[kind])
            continue
        if not groups[kind]:
            out[kind] = PkaValue(kind, None, "", f"no {'acidic' if kind == 'acid' else 'basic'}"
                                 " group of the types in the training data")
            continue
        p = pka_models.predict(mol, target)
        if p.value is None:
            out[kind] = PkaValue(kind, None, "", p.error, p, groups[kind])
        else:
            out[kind] = PkaValue(kind, p.value, "ML", ", ".join(groups[kind]), p, groups[kind])
    return PkaResult(out["acid"], out["base"])
