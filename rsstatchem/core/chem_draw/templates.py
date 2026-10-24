"""Template library of the editor (ChemDraw's template toolbars): amino acids,
nucleobases, sugars, heterocycles, functional groups and protecting groups.

Templates are SMILES; ``*`` marks an attachment point (drawn as a generic "R" label).
Stereocentres are given with the natural configuration (L-amino acids, D-sugars).
"""
from __future__ import annotations

__all__ = ["TEMPLATES", "categories", "find"]

TEMPLATES: dict[str, list[tuple[str, str]]] = {
    "Amino acids": [
        ("Glycine", "NCC(=O)O"), ("L-Alanine", "C[C@H](N)C(=O)O"),
        ("L-Valine", "CC(C)[C@H](N)C(=O)O"), ("L-Leucine", "CC(C)C[C@H](N)C(=O)O"),
        ("L-Isoleucine", "CC[C@H](C)[C@H](N)C(=O)O"), ("L-Proline", "OC(=O)[C@@H]1CCCN1"),
        ("L-Phenylalanine", "N[C@@H](Cc1ccccc1)C(=O)O"),
        ("L-Tyrosine", "N[C@@H](Cc1ccc(O)cc1)C(=O)O"),
        ("L-Tryptophan", "N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O"),
        ("L-Serine", "N[C@@H](CO)C(=O)O"), ("L-Threonine", "C[C@@H](O)[C@H](N)C(=O)O"),
        ("L-Cysteine", "N[C@@H](CS)C(=O)O"), ("L-Methionine", "CSCC[C@H](N)C(=O)O"),
        ("L-Asparagine", "NC(=O)C[C@H](N)C(=O)O"),
        ("L-Glutamine", "NC(=O)CC[C@H](N)C(=O)O"),
        ("L-Aspartic acid", "N[C@@H](CC(=O)O)C(=O)O"),
        ("L-Glutamic acid", "N[C@@H](CCC(=O)O)C(=O)O"),
        ("L-Lysine", "NCCCC[C@H](N)C(=O)O"),
        ("L-Arginine", "N[C@@H](CCCNC(=N)N)C(=O)O"),
        ("L-Histidine", "N[C@@H](Cc1c[nH]cn1)C(=O)O"),
    ],
    "Nucleobases": [
        ("Adenine", "Nc1ncnc2[nH]cnc12"), ("Guanine", "Nc1nc2[nH]cnc2c(=O)[nH]1"),
        ("Cytosine", "Nc1cc[nH]c(=O)n1"), ("Thymine", "Cc1c[nH]c(=O)[nH]c1=O"),
        ("Uracil", "O=c1cc[nH]c(=O)[nH]1"),
    ],
    "Sugars": [
        ("β-D-Glucopyranose", "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O"),
        ("α-D-Glucopyranose", "OC[C@H]1O[C@H](O)[C@H](O)[C@@H](O)[C@@H]1O"),
        ("β-D-Galactopyranose", "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@H]1O"),
        ("β-D-Ribofuranose", "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H]1O"),
        ("2-Deoxy-β-D-ribofuranose", "OC[C@H]1O[C@@H](O)C[C@@H]1O"),
        ("β-D-Fructofuranose", "OC[C@H]1O[C@@](O)(CO)[C@@H](O)[C@@H]1O"),
    ],
    "Rings & heterocycles": [
        ("Naphthalene", "c1ccc2ccccc2c1"), ("Anthracene", "c1ccc2cc3ccccc3cc2c1"),
        ("Pyridine", "c1ccncc1"), ("Pyrimidine", "c1cncnc1"), ("Pyrrole", "c1cc[nH]c1"),
        ("Furan", "c1ccoc1"), ("Thiophene", "c1ccsc1"), ("Imidazole", "c1c[nH]cn1"),
        ("Indole", "c1ccc2[nH]ccc2c1"), ("Quinoline", "c1ccc2ncccc2c1"),
        ("Piperidine", "C1CCNCC1"), ("Morpholine", "C1COCCN1"),
        ("Tetrahydrofuran", "C1CCOC1"), ("Cyclohexane", "C1CCCCC1"),
        ("Steroid core", "C1CCC2C(C1)CCC1C2CCC2CCCC21"),
    ],
    "Functional groups": [
        ("Carboxylic acid", "*C(=O)O"), ("Ester", "*C(=O)OC"), ("Amide", "*C(N)=O"),
        ("Aldehyde", "*C=O"), ("Ketone", "*C(C)=O"), ("Nitrile", "*C#N"),
        ("Nitro", "*[N+](=O)[O-]"), ("Sulfonamide", "*S(N)(=O)=O"),
        ("Sulfonic acid", "*S(=O)(=O)O"), ("Phenyl", "*c1ccccc1"), ("Phenol", "Oc1ccccc1"),
        ("Aniline", "Nc1ccccc1"), ("Acetyl chloride", "CC(Cl)=O"), ("Urea", "NC(N)=O"),
        ("Guanidine", "NC(N)=N"), ("Azide", "*N=[N+]=[N-]"),
    ],
    "Protecting groups": [
        ("Boc", "*C(=O)OC(C)(C)C"), ("Cbz", "*C(=O)OCc1ccccc1"),
        ("Fmoc", "*C(=O)OCC1c2ccccc2-c2ccccc12"), ("Acetyl (Ac)", "*C(C)=O"),
        ("Benzyl (Bn)", "*Cc1ccccc1"), ("PMB", "*Cc1ccc(OC)cc1"),
        ("TBS", "*[Si](C)(C)C(C)(C)C"), ("TMS", "*[Si](C)(C)C"),
        ("Tosyl (Ts)", "*S(=O)(=O)c1ccc(C)cc1"), ("Mesyl (Ms)", "*S(C)(=O)=O"),
        ("THP", "*C1CCCCO1"), ("MOM", "*COC"),
    ],
}


def categories() -> list[str]:
    return list(TEMPLATES)


def find(name: str) -> str | None:
    for items in TEMPLATES.values():
        for n, smi in items:
            if n.lower() == name.lower():
                return smi
    return None
