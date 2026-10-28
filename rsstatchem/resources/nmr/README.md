# NMR shift table (Chemical Drawing → Structure → Predict ¹H / ¹³C-NMR Shifts)

| File | What | Made by |
|---|---|---|
| `hose_shifts.npz` (0.37 MB) | HOSE-style atom-environment codes (radius 0–5, hashed to 63 bits) → mean / standard deviation / count of the observed ¹³C and ¹H shifts | `scripts/build_nmr_db.py` |
| `benchmark.json` | data summary and the held-out benchmark below | same |

Rebuild (raw NMR-STAR files, ~250 MB, go to the git-ignored `build/nmr-raw`):

    python scripts/build_nmr_db.py --download
    python scripts/build_nmr_db.py

## Source and licence

**BMRB metabolomics archive** (Biological Magnetic Resonance Data Bank, entries `bmse*`,
https://bmrb.io/ftp/pub/bmrb/metabolomics/entry_directories/): experimental spectra of small
molecules with assigned chemical shifts, structure and sample conditions in NMR-STAR v3.
BMRB is a wwPDB core archive; the wwPDB advisory notice
(https://files.wwpdb.org/pub/pdb/advisory.txt) reads: "The archival data files in the wwPDB
core archives (PDB, EMDB, and BMRB) are made freely available to all users under CC0 license
terms." Only derived statistics are shipped (no BMRB file, modified or not). Please cite BMRB:
J. C. Hoch et al., Nucleic Acids Res. 51 (2023) D368.

3 409 entries were downloaded; 1 288 distinct compounds were used (1 841 entries have no
assigned shift list or covalent structure, 198 duplicate a compound, 75 were dropped by the
sanity checks or RDKit). Solvents: D2O 496, CDCl3 509, acetone-d6 114, DMSO-d6 102, CD3OD 34,
unknown 33. D2O spectra were mostly recorded near pH 7.4 (acids as carboxylates, amines
protonated) - groups next to ionisable sites may differ from CDCl3 values.

**Not used:**

* **nmrshiftdb2** - its "nmrshiftdb2 Database License" (derived from the ODbL; text read from
  the archived copy of https://nmrshiftdb.nmr.uni-koeln.de/nmrshiftdbhtml/nmrshiftdb2datalicense.txt,
  the live server was unreachable) adds section 4.5: "If you build a software which relies on
  the database, a derivative database or a produced work for its functionality, this software
  must be licensed under an Open Source Initiative-approved license ... This includes ... NMR
  prediction software using the database for the prediction." RS-StatChem is not distributed
  under an OSI licence, so neither bundling nor downloading nmrshiftdb2 data for the predictor
  is allowed.
* Shift increment tables from books (Pretsch et al., ChemNMR) - copyrighted; none are used.
* HMDB / NP-MRD spectra (commercial reuse needs permission).

## Method

`nmr_predict.atom_codes`: radius r lists every heavy atom up to r bonds away as a canonical
rooted tree (element, aromaticity, charge, bond type from the parent, ring closures within
the sphere); the centre also carries its H count and heavy-atom degree. Concept after
W. Bremser, Anal. Chim. Acta 103 (1978) 355; the string format is our own. A prediction uses
the largest radius found. ¹H shifts are predicted per heavy atom (mean of its hydrogens).
Multiplets: first order with generic coupling constants (see the module docstring).

## Held-out benchmark

Compounds split 80/20 by a hash of the InChIKey connectivity block (stereoisomers stay on
one side); table built from 1 028 compounds, tested on 260:

| Nucleus | shifts | MAE (ppm) | RMSE (ppm) |
|---|---|---|---|
| ¹³C | 3 418 | 2.81 | 6.01 |
| ¹H | 2 203 | 0.21 | 0.41 |

By the radius that matched (¹³C MAE / n): 5: 1.51 / 1 888, 4: 2.43 / 358, 3: 3.12 / 476,
2: 5.13 / 503, 1: 8.37 / 183, 0: 28.2 / 10. (¹H: 0.14, 0.18, 0.24, 0.31, 0.57, 1.33.) The NMR
window flags radius ≤ 1 as low confidence. Ethyl acetate (CDCl3 literature 171.1, 60.5, 21.0,
14.2 / 4.12 q, 2.05 s, 1.26 t) is predicted as 170.1, 62.9, 20.6, 15.2 / 4.20 q, 2.03 s, 1.22 t.
Coverage is limited to environments similar to the 1 288 reference compounds (metabolites,
common solvents and simple organics); e.g. alkyl bromides fall back to radius 0–1.
