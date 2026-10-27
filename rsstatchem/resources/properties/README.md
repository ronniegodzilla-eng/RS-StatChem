# Experimental property data, ML models and benchmarks (Chemical Drawing → Chemical Properties)

| File | What | Made by |
|---|---|---|
| `exp_props.tsv.xz` (0.56 MB) | experimental values for 38 839 structures (InChIKey → value, sources) | `scripts/build_property_db.py` |
| `ml_models.npz` (2.3 MB) | gradient-boosted tree models (melting point, boiling point, logP) as numpy arrays, plus training fingerprints for the applicability domain | `scripts/train_property_models.py` |
| `benchmarks.json` (9 kB) | error of every estimation method against the experimental data | `scripts/benchmark_properties.py` |

Rebuild (raw downloads, ~100 MB, go to the git-ignored `build/propdb-raw`):

    python scripts/build_property_db.py --download --training
    python scripts/train_property_models.py --oof        # needs scikit-learn (dev only)
    python scripts/fetch_cdk.py                          # CDK jars + helper (Java, javac)
    python scripts/build_nannoolal_tables.py --download  # Nannoolal parameters (DDBST)
    python scripts/benchmark_properties.py

## Sources and licences

Only openly licensed or public-domain data are used. Every value in the app names its sources.

| Code | Source | Licence | Values used |
|---|---|---|---|
| B | Jean-Claude Bradley Open Melting Point Dataset, figshare doi:10.6084/m9.figshare.1031637 (rows flagged `donotuse` dropped) | CC0 1.0 | 27 976 melting points |
| O | US EPA OPERA curated PHYSPROP training/test sets (`*_QR.sdf` in `OPERA_Data.zip`, github.com/NIEHS/OPERA; K. Mansouri et al., J. Cheminform. 10 (2018) 10). The repository is MIT licensed and describes the data as open data; the underlying PHYSPROP database (Syracuse Research Corp.) was distributed free of charge through EPA. | MIT (OPERA repository) | 8 611 mp, 5 434 bp, 13 888 logP, 5 150 water solubility, 2 786 vapour pressure |
| A | AqSolDB, M. C. Sorkun et al., Sci. Data 6 (2019) 143, Harvard Dataverse doi:10.7910/DVN/OVHAW8 | CC0 1.0 | 9 980 water solubilities |
| H | Hazardous Substances Data Bank (US National Library of Medicine) via PubChem annotations | public domain (NLM web policy: government information) | 3 224 mp, 1 628 bp, 1 766 logP, 1 005 vapour pressure, 443 Tc, 365 Pc, 45 Vc |
| I | ILO-WHO International Chemical Safety Cards via PubChem | CC BY 4.0 (attribution: ILO/WHO ICSC) | 1 193 mp, 857 bp, 844 logP, 192 vapour pressure |
| N | NIOSH Pocket Guide (US CDC/NIOSH) via PubChem | public domain (US government work) | 487 mp, 412 bp |
| D | PAC database, US Department of Energy, via PubChem | public domain (energy.gov web policy) | 1 292 mp, 1 134 bp, 392 vapour pressure |

Left out because the licence restricts reuse or is not stated: PubChem annotations from HMDB
(commercial use needs permission), DrugBank (CC BY-NC), CAMEO Chemicals (part of the data is
third-party copyrighted), JECFA (CC BY-NC-SA), Haz-Map (all rights reserved), OSHA (pages may
contain third-party copyrighted content), SangsterLogP (no licence stated), EU Food Improvement
Agents and individual contributors' annotations (no licence stated); the OPERA 2.9 "update"
tables (they add values scraped from commercial web sites such as LookChem/ChemicalBook);
NIST WebBook / TRC (not redistributable); Wikipedia data pages (CC BY-SA, share-alike);
handbook tables bundled in other software (CRC, Yaws, DIPPR: copyrighted).

Compounds per property in the file: melting point 21 963, boiling point 6 021, logP 14 364,
water solubility 10 822, vapour pressure (25 °C) 3 122, Tc 431, Pc 360, Vc 45.

## Processing

* Key: full standard InChIKey of the structure as recorded by the source (salts stay salts).
  The app looks up the full key first, then a stereo-unspecified record with the same
  connectivity (and says so).
* Units: K (mp, bp, Tc), log10 Kow, log10(mol/L) near 25 °C (water solubility; OPERA records
  outside 20–30 °C dropped), log10(Pa) at 25 °C (vapour pressure), bar (Pc), cm³/mol (Vc).
* PubChem texts are parsed strictly: one value or a range (≤ 15 K, midpoint) with a unit; boiling
  points only at 740–780 mmHg and never "decomposes"; anything estimated / calculated, for
  another form or salt, a sublimation, or with several values is rejected. Check against the
  other sources (same compound): median absolute difference 0–0.25 K for HSDB/ICSC/NIOSH/DOE
  melting and boiling points (90th percentile 2–6 K), 0.00–0.01 for logP.
* Several values for one compound: the median is shown with the number of values; a spread
  above 10 K (mp, bp, Tc), 0.5 log units (logP, logS, log VP), 3 bar (Pc) or 20 cm³/mol (Vc) is
  shown as a range and flagged (636 mp, 87 bp, 65 logP, 215 logS compounds).

## Machine-learning models

HistGradientBoostingRegressor (700 trees, 48 leaves, learning rate 0.07) on RDKit 2D
descriptors, element counts, a symmetry ratio and Morgan count fingerprints (radius 2, 1024
bins); trained on single-component, overall neutral organic compounds (stereoisomers merged).
Exported trees reproduce scikit-learn to < 2e-6. The app evaluates them with numpy only.

| Model | n | 5-fold random CV RMSE / MAE | Scaffold split (20 % test) RMSE / MAE | R² (CV) |
|---|---|---|---|---|
| Melting point | 19 854 | 37.0 K / 26.9 K | 38.1 K / 27.4 K | 0.84 |
| Boiling point | 5 743 | 26.8 K / 13.5 K | 26.0 K / 12.7 K | 0.91 |
| logP | 14 049 | 0.47 / 0.31 | 0.48 / 0.31 | 0.94 |

Expected error shown next to a prediction = CV RMSE of compounds whose highest Tanimoto
similarity to the training folds falls in the same band:

| Similarity to training set | < 0.3 (outside domain) | 0.3–0.5 | 0.5–0.7 | ≥ 0.7 |
|---|---|---|---|---|
| Melting point RMSE | 52.6 K (n 225) | 46.8 K (3 156) | 35.9 K (10 409) | 32.2 K (6 064) |
| Boiling point RMSE | 63.4 K (78) | 36.5 K (1 037) | 23.7 K (3 136) | 21.1 K (1 492) |
| logP RMSE | 0.83 (159) | 0.69 (1 772) | 0.44 (6 534) | 0.38 (5 584) |

## Benchmark (benchmarks.json)

Same compound set as the models. The ML and consensus numbers are cross-validated (held out);
the published methods were fitted by their authors on data that overlap these sets, which
favours them. "n" = compounds the method covers.

| Property | Method | n | RMSE | MAE |
|---|---|---|---|---|
| Melting point (K) | ML | 19 854 | 37.0 | 26.9 |
| | Marrero-Gani 1st order | 16 994 | 54.2 | 40.0 |
| | Marrero-Gani 1st + 2nd + 3rd order | 16 802 | 64.3 | 43.8 |
| | Marrero-Gani 1st + 2nd order | 16 806 | 64.8 | 43.8 |
| | Joback | 14 922 | 152.9 | 96.6 |
| Boiling point (K) | ML | 5 743 | 26.8 | 13.5 |
| | Nannoolal (2004) | 5 397 | 28.3 | 13.9 |
| | Marrero-Gani 1st + 2nd + 3rd | 5 299 | 32.4 | 17.6 |
| | Marrero-Gani 1st + 2nd | 5 299 | 32.6 | 17.9 |
| | Marrero-Gani 1st | 5 299 | 33.3 | 18.8 |
| | Stein-Brown | 5 665 | 36.7 | 19.8 |
| | Joback | 5 299 | 55.6 | 31.2 |
| Tc (K) | Nannoolal (2007), experimental Tb | 331 | 42.7 | 14.8 |
| | Nannoolal (2007), Nannoolal Tb estimate | 331 | 47.2 | 21.2 |
| | Marrero-Gani 1st + 2nd | 342 | 53.5 | 26.1 |
| | Marrero-Gani 1st + 2nd + 3rd | 342 | 53.8 | 26.5 |
| | Marrero-Gani 1st | 342 | 54.8 | 27.5 |
| | Abdulelah-Gani | 354 | 63.5 | 30.7 |
| | Joback | 352 | 65.6 | 37.2 |
| Pc (bar) | Marrero-Gani 1st + 2nd + 3rd | 294 | 5.31 | 2.89 |
| | Marrero-Gani 1st + 2nd | 294 | 5.57 | 3.06 |
| | Joback | 303 | 6.14 | 3.65 |
| | Abdulelah-Gani | 303 | 7.68 | 4.28 |
| | Nannoolal (2007) | 287 | 8.20 | 4.17 |
| Vc (cm³/mol) | Marrero-Gani 1st + 2nd | 33 | 16.2 | 11.8 |
| | Marrero-Gani 1st + 2nd + 3rd | 33 | 16.4 | 12.1 |
| | Nannoolal (2007) | 32 | 17.5 | 9.8 |
| | Joback | 33 | 22.4 | 12.2 |
| | Abdulelah-Gani | 33 | 25.8 | 15.3 |
| log10 vapour pressure, 25 °C (Pa) | Nannoolal (2008), as shown by the app | 2 611 | 1.08 | 0.56 |
| | ... with experimental Tb, liquids | 1 405 | 0.43 | 0.18 |
| | ... with experimental Tb, solids | 438 | 1.09 | 0.64 |
| | ... with ML Tb, liquids | 124 | 1.09 | 0.68 |
| | ... with ML Tb, solids | 644 | 1.81 | 1.29 |
| logP | ML | 14 049 | 0.47 | 0.31 |
| | Consensus (Crippen, JPlogP, XLogP, ALogP, ML) | 14 043 | 0.67 | 0.48 |
| | Consensus of the 4 open atom/fragment methods | 14 050 | 0.77 | 0.54 |
| | ALogP (CDK) | 14 056 | 0.81 | 0.56 |
| | JPlogP (CDK) | 14 050 | 0.83 | 0.60 |
| | XLogP (CDK) | 14 056 | 0.88 | 0.62 |
| | Crippen (RDKit) | 14 056 | 0.92 | 0.65 |
| logS | ESOL | 8 964 | 1.34 | 1.00 |

Vapour pressure: the app uses the experimental Tb when there is one, else the default Tb
estimate (ML; the benchmark uses the shipped ML model for compounds without an experimental Tb,
which it never saw), and converts the subcooled-liquid value of a solid (melting point above
25 °C: experimental, else the default estimate) with Walden's rule, log10(Ps/PL) =
-56.5 J mol⁻¹ K⁻¹ (Tm - T) / (ln 10 R T) (S. H. Yalkowsky, Ind. Eng. Chem. Fundam. 18 (1979)
108). Without that conversion solids are overestimated (RMSE 2.04 instead of 1.56 log units,
bias +1.18); with the Nannoolal Tb estimate instead of ML the RMSE is 2.13. The expected error
shown next to a value is the RMSE of its subset (Tb source x liquid / solid).

The app shows, per property, the experimental value when there is one, otherwise the method
with the lowest RMSE on the compounds every compared method covers (`common` in the JSON; for
molecules outside the ML domain the `ood_common` subset, where ML still had the lowest error).
All other methods are listed in the tooltip and the Details panel with these errors. The CDK
methods are open-source alternatives, not BioByte CLogP/CMR; values differ.

## Group-contribution sources

* **Nannoolal / Rarey** (normal boiling point, Fluid Phase Equilib. 226 (2004) 45; critical
  properties, 252 (2007) 1; vapour pressure, 269 (2008) 117). Group definitions and priorities:
  Y. Nannoolal, PhD thesis, University of KwaZulu-Natal 2006, Appendix B (ResearchSpace,
  http://hdl.handle.net/10413/3766; the MSc thesis of 2004 for boiling-point details).
  Parameters: DDBST files "2012-01-18 Rarey-Nannoolal {Tb, Tcri, Pcri, vcri, Vap}.txt",
  https://chemthermo.ddbst.com/Parameters/ (published for the textbook Gmehling, Kolbe,
  Kleiber, Rarey, Chemical Thermodynamics for Process Simulation, Wiley-VCH 2012, to be
  implemented in software; no licence terms are stated - only numerical parameters are used).
  Their Tc, Pc, Vc and dB values are the PhD-thesis tables; the code reproduces the thesis'
  worked examples (Appendix D) and the MSc thesis' boiling-point examples (Appendix C, with the
  MSc parameters). The Tb equation uses the 2004 publication's constants (0.6583, 1.6868,
  84.3395): the DDBST "revised" constants with the same group values overestimate the open
  boiling points by 17 K (RMSE 35.1 K instead of 28.3 K).
* **Marrero-Gani third order**: values as tabulated in pychemqt (GPL; numerical data only, no
  code), Tb column checked against the MSc thesis Table A-8; group search written for
  RS-StatChem (``mg_third.py``), reproducing the paper's examples (N-phenyl-1,4-
  benzenediamine, pyrene, 4-aminobutanol, 1,9-nonanediol, dicoumarol, flutemazepam).

What became the default (lowest benchmark RMSE among the methods covering a molecule):
Nannoolal for Tc, Marrero-Gani 1st + 2nd + 3rd order for Pc, Nannoolal for the vapour pressure
(the only estimate). ML stays the default for Tb and Tm.

## CDK (resources/java)

``scripts/fetch_cdk.py`` downloads 13 unmodified jars from Maven Central into
``resources/java/cdk`` (each pinned by SHA-256; list, hashes and source locations in the
generated ``resources/java/LICENSE-CDK.txt`` and ``VERSION``): cdk-interfaces, cdk-core,
cdk-standard, cdk-silent, cdk-valencycheck, cdk-atomtype, cdk-isomorphism, cdk-smarts,
cdk-smiles, cdk-qsar, cdk-qsarmolecular 2.13 (LGPL-2.1-or-later; source
https://github.com/cdk/cdk/tree/cdk-2.13 and the ``-sources.jar`` files on Maven Central),
beam-core 1.3.10 (BSD 2-Clause) and vecmath 1.5.2 (GPL-2.0 with Classpath exception). 1.9 MB
instead of the 24.1 MB ``cdk-2.13.jar`` bundle; the helper gives byte-identical answers for all
52 895 benchmark and database SMILES with both (``fetch_cdk.py --compare-bundle FILE``).
