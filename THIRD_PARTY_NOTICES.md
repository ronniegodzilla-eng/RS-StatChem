# Third-party software and data

RS-StatChem's own code is licensed under the PolyForm Noncommercial License 1.0.0 (see
`LICENSE.md`). The packaged app also contains the components below, each under its own
licence; nothing here changes those terms. Licence texts ship with each package in the app
folder (Python packages keep theirs in their `*.dist-info` folders).

## Libraries bundled in the app

| Component | Licence |
|---|---|
| Qt for Python (PySide6) and Qt 6 | LGPL-3.0 (dynamically linked; you may replace them) |
| NumPy, SciPy, pandas, statsmodels, h5py | BSD-3-Clause |
| Matplotlib | Matplotlib licence (PSF-based) |
| RDKit | BSD-3-Clause |
| openpyxl, python-docx, pydicom, PyWavelets | MIT |
| pyreadstat (ReadStat) | Apache-2.0 (ReadStat: MIT) |
| tifffile, roifile | BSD-3-Clause |
| Pillow | MIT-CMU (HPND) |
| certifi | MPL-2.0 |
| EPAM Indigo (`epam.indigo`) | Apache-2.0 |
| OPSIN name-to-structure | MIT (`resources/opsin/LICENSE-OPSIN.txt`) |
| Chemistry Development Kit (CDK) 2.13, unmodified jars | LGPL-2.1-or-later; beam-core BSD-2-Clause; vecmath GPL-2.0 with Classpath Exception (`resources/java/LICENSE-CDK.txt`, sources on Maven Central) |
| Eclipse Temurin Java runtime (jlink subset) | GPL-2.0 with Classpath Exception |

## Algorithms ported or re-implemented

- ImageJ 1.x (NIH): public domain; algorithms in `rsstatchem/core/imaging/` are ported
  from its sources. Fiji plugins (GPL/LGPL) were not copied; their features were
  re-implemented from the published papers.
- Revvity/ChemDraw CDX format: written from Revvity's BSD-3-Clause `chemdraw` code used as a
  reference (no code copied).
- Statistical procedures follow published methods and IBM's public "SPSS Statistics
  Algorithms" documentation; no IBM code or documents are included.

## Data bundled in the app

| Data | Source and licence |
|---|---|
| Offline compound names (`resources/names`) | PubChem (NCBI), public domain |
| Experimental properties (`resources/properties`) | Bradley Open Melting Point Dataset (CC0), US EPA OPERA (MIT / open data), AqSolDB (CC0), HSDB, NIOSH, DOE PAC via PubChem (public domain), ILO-WHO International Chemical Safety Cards (CC BY 4.0, attribution: ILO/WHO ICSC) — details in `resources/properties/README.md` |
| pKa and Henry's law models | trained on OPERA data (Mansouri et al. 2019; CC0 data, MIT repository) |
| NMR shift statistics (`resources/nmr`) | derived from BMRB metabolomics entries (CC0) — `resources/nmr/README.md` |
| Group-contribution parameters | published Joback, Stein-Brown, Marrero-Gani, Abdulelah-Gani and Nannoolal values (numbers only), sources in `rsstatchem/core/chem_draw/gc_tables.py` |
| Glassware and template drawings | original work, see `resources/templates/ATTRIBUTION.md` |

## Test data (repository only, not shipped in the app)

NIST StRD datasets (public domain); ImageJ sample images `blobs.gif` and `Cell_Colony.jpg`
from the ImageJ repository (public domain) and golden outputs produced with ImageJ 1.54p;
small published example datasets from R packages and textbooks (AirPassengers, airquality,
gehan, lung, ovarian, cbpp, multicenter, side-effect), used only to check results.
Reference outputs from UCLA OARC are downloaded at test time and are not part of the
repository.

## Trademarks

SPSS is a trademark of IBM; Origin and OriginPro of OriginLab; Design-Expert of Stat-Ease;
ChemDraw of Revvity; ImageJ and Fiji are names of their respective projects. RS-StatChem is
not affiliated with or endorsed by any of them; the names only describe which workflow each
module follows.
