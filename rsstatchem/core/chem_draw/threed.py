"""3D model behind the *Structure Perspective* view (no Qt): an RDKit ETKDG conformer of a
drawn molecule (MMFF94-optimised when parameters exist, else UFF), rotations and the
projection back onto the drawing, keeping the stereochemistry (wedges are re-assigned for
the new 2D geometry).

ETKDG: S. Riniker, G. A. Landrum, J. Chem. Inf. Model. 55 (2015) 2562.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from rsstatchem.core.chem import molecule as mo
from rsstatchem.core.chem_draw.model import Document

__all__ = ["Model3D", "embed", "embed_doc", "rotation", "project", "apply_to_drawing",
           "CPK", "VDW"]

CPK = {"H": "#FFFFFF", "C": "#909090", "N": "#3050F8", "O": "#FF0D0D", "F": "#90E050",
       "Cl": "#1FF01F", "Br": "#A62929", "I": "#940094", "S": "#FFFF30", "P": "#FF8000",
       "B": "#FFB5B5", "Si": "#F0C8A0", "Na": "#AB5CF2", "K": "#8F40D4", "Mg": "#8AFF00",
       "Fe": "#E06633", "Cu": "#C88033", "Zn": "#7D80B0", "Se": "#FFA100"}
VDW = {"H": 1.10, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47, "Cl": 1.75, "Br": 1.85,
       "I": 1.98, "S": 1.80, "P": 1.80, "B": 1.92, "Si": 2.10}


@dataclass
class Model3D:
    symbols: list = field(default_factory=list)
    coords: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))   # Angstrom, centred
    bonds: list = field(default_factory=list)          # (i, j, order 1 / 1.5 / 2 / 3)
    doc_ids: list = field(default_factory=list)        # document atom id or None (added H)
    method: str = ""                                   # e.g. "ETKDG v3 + MMFF94"
    atoms: list = field(default_factory=list)          # document atom ids of the molecule

    def heavy(self) -> list[int]:
        return [i for i, s in enumerate(self.symbols) if s != "H"]


def embed(mol: Chem.Mol, seed: int = 0x5EED) -> Model3D:
    """3D conformer of ``mol`` (explicit hydrogens added). Raises ChemError."""
    m = Chem.AddHs(Chem.Mol(mol))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(m, params) != 0:
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(m, params) != 0:
            raise mo.ChemError("No 3D conformer could be generated for this structure.")
    method = "ETKDG v3"
    try:
        if AllChem.MMFFHasAllMoleculeParams(m):
            AllChem.MMFFOptimizeMolecule(m, maxIters=1000)
            method += " + MMFF94"
        elif AllChem.UFFHasAllMoleculeParams(m):
            AllChem.UFFOptimizeMolecule(m, maxIters=1000)
            method += " + UFF"
    except Exception:  # noqa: BLE001 - keep the unoptimised conformer
        pass
    pos = np.array(m.GetConformer().GetPositions(), dtype=float)
    pos -= pos.mean(axis=0)
    bonds = []
    for b in m.GetBonds():
        order = {Chem.BondType.DOUBLE: 2, Chem.BondType.TRIPLE: 3,
                 Chem.BondType.AROMATIC: 1.5}.get(b.GetBondType(), 1)
        bonds.append((b.GetBeginAtomIdx(), b.GetEndAtomIdx(), order))
    doc_ids = [a.GetIntProp("doc_id") if a.HasProp("doc_id") else None for a in m.GetAtoms()]
    return Model3D([a.GetSymbol() for a in m.GetAtoms()], pos, bonds, doc_ids, method)


def embed_doc(doc: Document, atom_ids) -> Model3D:
    """3D model of one drawn molecule (the fragment of ``atom_ids``)."""
    from rsstatchem.core.chem_draw import convert

    mol = convert.to_mol(doc, sorted(atom_ids))
    if len(Chem.GetMolFrags(mol)) > 1:
        raise mo.ChemError("Select a single molecule.")
    model = embed(mol)
    model.atoms = sorted(atom_ids)
    return model


def rotation(yaw: float, pitch: float, roll: float = 0.0) -> np.ndarray:
    """Rotation about the screen's vertical (yaw), horizontal (pitch) and viewing (roll)
    axes, in radians."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return rz @ rx @ ry


def project(model: Model3D, rot: np.ndarray) -> np.ndarray:
    """Rotated coordinates (x right, y up, z towards the viewer)."""
    return model.coords @ rot.T


def _median_heavy_bond(model: Model3D, pts: np.ndarray) -> float:
    lengths = sorted(float(np.linalg.norm(pts[i] - pts[j])) for i, j, _ in model.bonds
                     if model.symbols[i] != "H" and model.symbols[j] != "H")
    return lengths[len(lengths) // 2] if lengths else 1.5


def apply_to_drawing(doc: Document, model: Model3D, rot: np.ndarray) -> bool:
    """Replace the drawing of the model's molecule by its rotated 3D projection (centre kept,
    bond length scaled to the document's), then re-assign wedges for the new geometry."""
    from rsstatchem.core.chem_draw import convert

    comp = [a for a in model.atoms if a in doc.atoms]
    if len(comp) < 2:
        return False
    pts = project(model, rot)
    k = doc.style.bond_length / (_median_heavy_bond(model, model.coords) or 1.5)
    acc: dict[int, list] = {}
    for i, aid in enumerate(model.doc_ids):
        if aid is None or aid not in doc.atoms:
            continue
        if model.symbols[i] != "H" or doc.atoms[aid].symbol == "H":
            acc.setdefault(aid, []).append(pts[i])
    cx0 = sum(doc.atoms[a].x for a in comp) / len(comp)
    cy0 = sum(doc.atoms[a].y for a in comp) / len(comp)
    new = {aid: np.mean(v, axis=0) for aid, v in acc.items() if aid in doc.atoms}
    if not new:
        return False
    mx = float(np.mean([p[0] for p in new.values()]))
    my = float(np.mean([p[1] for p in new.values()]))
    for aid, p in new.items():
        doc.atoms[aid].x = cx0 + (float(p[0]) - mx) * k
        doc.atoms[aid].y = cy0 - (float(p[1]) - my) * k
    try:                                   # wedges that express the same stereo in 2D
        specs, bonds, origin = convert.to_specs(doc, comp, expand=False)
        mol = mo.mol_from_graph(specs, bonds, sanitize=True)
    except mo.ChemError:
        return True
    # the graph was built from the old wedges on the new geometry: take the stereo from the
    # 3D model instead (its chiral tags are the drawn ones)
    _restore_chirality(mol, origin, model)
    conf = mol.GetConformer()
    for i, aid in enumerate(origin):
        a = doc.atoms[aid]
        conf.SetAtomPosition(i, (a.x, -a.y, 0.0))
    convert._rewedge(doc, comp, mol, origin)
    return True


def _restore_chirality(mol: Chem.Mol, origin, model: Model3D) -> None:
    """Set the tetrahedral tags of ``mol`` (atoms = ``origin`` document ids) from the 3D
    model's geometry."""
    try:
        tmp = Chem.Mol(mol)
        conf = Chem.Conformer(tmp.GetNumAtoms())
        sums: dict[int, list] = {}
        for i, aid in enumerate(model.doc_ids):
            if aid is not None and model.symbols[i] != "H":
                sums.setdefault(aid, []).append(model.coords[i])
        for i, aid in enumerate(origin):
            p = np.mean(sums[aid], axis=0) if aid in sums else np.zeros(3)
            conf.SetAtomPosition(i, (float(p[0]), float(p[1]), float(p[2])))
        conf.Set3D(True)
        tmp.RemoveAllConformers()
        tmp.AddConformer(conf, assignId=True)
        Chem.AssignStereochemistryFrom3D(tmp)
        for a, b in zip(mol.GetAtoms(), tmp.GetAtoms()):
            a.SetChiralTag(b.GetChiralTag())
    except Exception:  # noqa: BLE001 - keep the tags from the drawing
        pass
