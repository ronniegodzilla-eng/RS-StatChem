"""Machine-learning estimates of melting point, boiling point and logP (numpy inference only).

The models are gradient-boosted regression trees trained by ``scripts/train_property_models.py``
(scikit-learn, a development-only dependency) on the open experimental data of
``resources/properties/exp_props.tsv.xz`` (see the README there), then exported to
``resources/properties/ml_models.npz``: every tree as flat arrays (feature, threshold, children,
leaf value). This module evaluates them with numpy alone - scikit-learn is never imported by
the app.

Each prediction comes with

- the model's held-out error (RMSE / MAE on a random 20 % test split and on a scaffold split,
  both stored in the model file), and the RMSE of held-out compounds with a similar nearest-
  neighbour similarity, used as the *expected error* of this prediction;
- an applicability-domain flag: the highest Tanimoto similarity (Morgan radius 2, 1024 bits) of
  the molecule to any training compound; below ``AD_THRESHOLD`` the molecule is outside the
  domain and the error can be much larger.

Features (identical for training and inference, :func:`features`): RDKit 2D descriptors (a
fixed list, :data:`DESCRIPTORS`), element counts, a symmetry ratio and Morgan count fingerprint
(radius 2, 1024 bins). No Qt here.
"""
from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass, field

import numpy as np

__all__ = ["DESCRIPTORS", "features", "fingerprint", "MLPrediction", "MLModel", "models",
           "predict", "model_path", "AD_THRESHOLD", "TARGETS", "tanimoto_max", "GBTrees"]

TARGETS = {"mp": ("melting point", "K"), "bp": ("boiling point", "K"), "logp": ("logP", "")}
AD_THRESHOLD = 0.30
FP_BITS = 1024
# RDKit descriptors used as features: everything in Descriptors._descList except partial-charge
# based, information-content (Ipc) and BCUT descriptors, QED/SPS and the Morgan densities
DESCRIPTORS = (
    "MaxAbsEStateIndex MaxEStateIndex MinAbsEStateIndex MinEStateIndex MolWt HeavyAtomMolWt "
    "NumValenceElectrons BalabanJ BertzCT Chi0 Chi0n Chi0v Chi1 Chi1n Chi1v Chi2n Chi2v Chi3n "
    "Chi3v Chi4n Chi4v HallKierAlpha Kappa1 Kappa2 Kappa3 LabuteASA PEOE_VSA1 PEOE_VSA10 "
    "PEOE_VSA11 PEOE_VSA12 PEOE_VSA13 PEOE_VSA14 PEOE_VSA2 PEOE_VSA3 PEOE_VSA4 PEOE_VSA5 "
    "PEOE_VSA6 PEOE_VSA7 PEOE_VSA8 PEOE_VSA9 SMR_VSA1 SMR_VSA10 SMR_VSA2 SMR_VSA3 SMR_VSA4 "
    "SMR_VSA5 SMR_VSA6 SMR_VSA7 SMR_VSA8 SMR_VSA9 SlogP_VSA1 SlogP_VSA10 SlogP_VSA11 "
    "SlogP_VSA12 SlogP_VSA2 SlogP_VSA3 SlogP_VSA4 SlogP_VSA5 SlogP_VSA6 SlogP_VSA7 SlogP_VSA8 "
    "SlogP_VSA9 TPSA EState_VSA1 EState_VSA10 EState_VSA11 EState_VSA2 EState_VSA3 "
    "EState_VSA4 EState_VSA5 EState_VSA6 EState_VSA7 EState_VSA8 EState_VSA9 VSA_EState1 "
    "VSA_EState10 VSA_EState2 VSA_EState3 VSA_EState4 VSA_EState5 VSA_EState6 VSA_EState7 "
    "VSA_EState8 VSA_EState9 FractionCSP3 HeavyAtomCount NHOHCount NOCount "
    "NumAliphaticCarbocycles NumAliphaticHeterocycles NumAliphaticRings NumAmideBonds "
    "NumAromaticCarbocycles NumAromaticHeterocycles NumAromaticRings NumBridgeheadAtoms "
    "NumHAcceptors NumHDonors NumHeteroatoms NumHeterocycles NumRotatableBonds "
    "NumSaturatedCarbocycles NumSaturatedHeterocycles NumSaturatedRings NumSpiroAtoms Phi "
    "RingCount MolLogP MolMR fr_Al_COO fr_Al_OH fr_ArN fr_Ar_COO fr_Ar_N fr_Ar_NH fr_Ar_OH "
    "fr_COO fr_C_O fr_C_S fr_HOCCN fr_Imine fr_NH0 fr_NH1 fr_NH2 fr_N_O fr_Nhpyrrole fr_SH "
    "fr_aldehyde fr_alkyl_carbamate fr_alkyl_halide fr_amide fr_amidine fr_aniline "
    "fr_aryl_methyl fr_azide fr_azo fr_barbitur fr_benzene fr_bicyclic fr_diazo fr_epoxide "
    "fr_ester fr_ether fr_furan fr_guanido fr_halogen fr_hdrzine fr_hdrzone fr_imidazole "
    "fr_imide fr_isocyan fr_isothiocyan fr_ketone fr_lactam fr_lactone fr_methoxy fr_nitrile "
    "fr_nitro fr_nitro_arom fr_nitroso fr_oxazole fr_oxime fr_para_hydroxylation fr_phenol "
    "fr_phos_acid fr_phos_ester fr_priamide fr_pyridine fr_quatN fr_sulfide fr_sulfonamd "
    "fr_sulfone fr_term_acetylene fr_tetrazole fr_thiazole fr_thiocyan fr_thiophene "
    "fr_unbrch_alkane fr_urea").split()
ELEMENTS = ("C", "N", "O", "F", "Cl", "Br", "I", "S", "P", "Si", "B", "Se")


def _descriptor_functions():
    from rdkit.Chem import Descriptors
    table = dict(Descriptors._descList)
    return [table[n] for n in DESCRIPTORS]


_FUNCS = None


def feature_names() -> list[str]:
    return (list(DESCRIPTORS) + [f"n_{e}" for e in ELEMENTS] + ["n_H", "symmetry", "charge"] +
            [f"morgan_{i}" for i in range(FP_BITS)])


def features(mol) -> np.ndarray:
    """Feature vector of one molecule (float64; NaN where a descriptor fails)."""
    global _FUNCS
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    if _FUNCS is None:
        _FUNCS = _descriptor_functions()
    desc = []
    for fn in _FUNCS:
        try:
            v = float(fn(mol))
            desc.append(v if math.isfinite(v) else math.nan)
        except Exception:  # noqa: BLE001 - one failing descriptor must not stop the rest
            desc.append(math.nan)
    counts = {e: 0 for e in ELEMENTS}
    n_h = 0
    for a in mol.GetAtoms():
        s = a.GetSymbol()
        if s in counts:
            counts[s] += 1
        n_h += a.GetTotalNumHs()
    heavy = mol.GetNumHeavyAtoms() or 1
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    symmetry = len(set(ranks)) / heavy                  # 1/heavy = fully symmetric
    charge = sum(abs(a.GetFormalCharge()) for a in mol.GetAtoms())
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_BITS)
    morgan = gen.GetCountFingerprintAsNumPy(mol).astype(np.float64)
    return np.concatenate([np.array(desc), np.array([counts[e] for e in ELEMENTS], float),
                           np.array([n_h, symmetry, charge], float), morgan])


def fingerprint(mol) -> np.ndarray:
    """Packed Morgan bit fingerprint (radius 2, 1024 bits) for the applicability domain."""
    from rdkit.Chem import rdFingerprintGenerator
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_BITS)
    bits = gen.GetFingerprintAsNumPy(mol).astype(np.uint8)
    return np.packbits(bits)


_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)


def _popcount_rows(a: np.ndarray) -> np.ndarray:
    """Set bits per row of a packed uint8 matrix."""
    if hasattr(np, "bitwise_count") and a.shape[1] % 8 == 0:
        return np.bitwise_count(np.ascontiguousarray(a).view(np.uint64)).sum(axis=1)
    return _POP[a].sum(axis=1)


def similarities(fp: np.ndarray, fps: np.ndarray) -> np.ndarray:
    """Tanimoto similarity of a packed fingerprint to every row of fps."""
    if fps.size == 0:
        return np.zeros(0)
    inter = _popcount_rows(np.bitwise_and(fps, fp))
    union = _popcount_rows(np.bitwise_or(fps, fp))
    return np.where(union > 0, inter / np.maximum(union, 1), 0.0)


def tanimoto_max(fp: np.ndarray, fps: np.ndarray) -> tuple[float, int]:
    """(highest Tanimoto similarity, index) of a packed fingerprint against rows of fps."""
    sim = similarities(fp, fps)
    if sim.size == 0:
        return 0.0, -1
    i = int(np.argmax(sim))
    return float(sim[i]), i


class GBTrees:
    """Gradient-boosted regression trees as flat arrays: prediction = baseline + sum of the
    leaf values. A sample goes left when ``x <= threshold`` (or when x is NaN and
    ``missing_left``), exactly as scikit-learn's HistGradientBoostingRegressor."""

    def __init__(self, feature, threshold, left, right, value, missing_left, offsets,
                 baseline: float):
        self.feature = feature.astype(np.int32)
        self.threshold = threshold.astype(np.float64)
        self.left = left.astype(np.int32)
        self.right = right.astype(np.int32)
        self.value = value.astype(np.float64)
        self.missing_left = missing_left.astype(bool)
        self.offsets = offsets.astype(np.int64)     # first node of each tree
        self.baseline = float(baseline)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        out = np.full(X.shape[0], self.baseline)
        roots = self.offsets
        for r in range(X.shape[0]):
            x = X[r]
            node = roots.copy()
            while True:
                leaf = self.left[node] < 0
                if leaf.all():
                    break
                f = self.feature[node]
                v = x[np.where(leaf, 0, f)]
                go_left = np.where(np.isnan(v), self.missing_left[node],
                                   v <= self.threshold[node])
                nxt = np.where(go_left, self.left[node], self.right[node]) + roots
                node = np.where(leaf, node, nxt)
            out[r] += self.value[node].sum()
        return out


@dataclass
class MLPrediction:
    target: str
    value: float | None
    similarity: float = 0.0              # highest Tanimoto similarity to the training set
    in_domain: bool = False
    expected_rmse: float | None = None   # held-out RMSE of compounds with similar similarity
    metrics: dict = field(default_factory=dict)
    error: str = ""


@dataclass
class MLModel:
    target: str
    trees: GBTrees
    metrics: dict          # {"random": {...}, "scaffold": {...}, "ad_bins": [...], "n": ...}
    fp_index: np.ndarray   # rows of the shared fingerprint matrix used to train this model

    def expected_rmse(self, sim: float) -> float | None:
        for lo, hi, rmse, _n in self.metrics.get("ad_bins", []):
            if lo <= sim < hi or (hi >= 1.0 and sim >= lo):
                return rmse
        return self.metrics.get("random_cv", {}).get("rmse")


def model_path():
    from rsstatchem.paths import resource
    return resource("properties/ml_models.npz")


_MODELS: dict | None = None
_FPS: np.ndarray | None = None
_LOCK = threading.Lock()


def models() -> dict:
    """{target: MLModel} from the model file (empty when it is missing); loaded once."""
    global _MODELS, _FPS
    with _LOCK:
        if _MODELS is not None:
            return _MODELS
        out: dict = {}
        try:
            data = np.load(model_path(), allow_pickle=False)
        except (OSError, ValueError):
            _MODELS = {}
            return _MODELS
        _FPS = data["fps"] if "fps" in data else np.zeros((0, FP_BITS // 8), np.uint8)
        for t in TARGETS:
            if f"{t}_feature" not in data:
                continue
            meta = json.loads(str(data[f"{t}_meta"]))
            trees = GBTrees(data[f"{t}_feature"], data[f"{t}_threshold"], data[f"{t}_left"],
                            data[f"{t}_right"], data[f"{t}_value"], data[f"{t}_missing_left"],
                            data[f"{t}_offsets"], meta["baseline"])
            out[t] = MLModel(t, trees, meta["metrics"], data[f"{t}_fp_index"])
        _MODELS = out
        return out


def scope_problem(mol) -> str:
    """Why a molecule is outside what the models were trained on ('' when it is inside)."""
    from rdkit import Chem
    if mol is None or mol.GetNumAtoms() == 0:
        return "nothing drawn"
    if len(Chem.GetMolFrags(mol)) != 1:
        return "select a single molecule"
    symbols = {a.GetSymbol() for a in mol.GetAtoms()}
    if "C" not in symbols:
        return "the models cover organic compounds only"
    bad = sorted(symbols - set(ELEMENTS) - {"H"})
    if bad:
        return f"no training data for {', '.join(bad)}"
    if Chem.GetFormalCharge(mol) != 0:
        return "charged species are not covered"
    return ""


_CACHE: dict = {}


def predict_all(mol) -> dict[str, MLPrediction]:
    """ML estimates of every target for one molecule (cached by canonical SMILES)."""
    from rdkit import Chem
    ms = models()
    if not ms:
        return {}
    why = scope_problem(mol)
    if why:
        return {t: MLPrediction(t, None, error=why) for t in ms}
    try:
        key = Chem.MolToSmiles(mol)
    except Exception:  # noqa: BLE001
        key = None
    if key is not None and key in _CACHE:
        return _CACHE[key]
    out = {}
    try:
        x = features(mol)[None, :]
        sims = similarities(fingerprint(mol), _FPS)
        for t, m in ms.items():
            value = float(m.trees.predict(x)[0])
            sim = float(sims[m.fp_index].max()) if m.fp_index.size else 0.0
            out[t] = MLPrediction(t, value, sim, sim >= AD_THRESHOLD, m.expected_rmse(sim),
                                  m.metrics)
    except Exception as exc:  # noqa: BLE001 - an estimate must never break drawing
        out = {t: MLPrediction(t, None, error=str(exc)) for t in ms}
    if key is not None:
        if len(_CACHE) > 256:
            _CACHE.clear()
        _CACHE[key] = out
    return out


def predict(mol, target: str) -> MLPrediction:
    """ML estimate of one property with its expected error and applicability domain."""
    if target not in models():
        return MLPrediction(target, None, error="model file missing")
    return predict_all(mol).get(target) or MLPrediction(target, None, error="no model")
