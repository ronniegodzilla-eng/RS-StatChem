# RS-StatChem Chemical Drawing vs ChemDraw Professional: gap research

Date: 2026-10-27. Scope: `rsstatchem/ui/chemdraw/`, `rsstatchem/ui/chem_editor/`, `rsstatchem/core/chem_draw/`,
`rsstatchem/core/chem/`, `rsstatchem/ui/chem/`, tests `tests/test_chem*.py`. I did not change any code.

## 0. Method and evidence

**ChemDraw sources used**
- **ChemDraw 17.1 User Guide** (PerkinElmer, 393 pp., official copy hosted by Uni Ulm:
  https://uni-ulm.de/fileadmin/website_uni_ulm/nawi.chemie/Downloads/ChemOffice/ChemDraw.pdf). I extracted the text
  and decoded it in the scratchpad (`cd171_flat.txt`). I also pulled every `Menu > Command` path the guide cites into
  `menu_paths_17_1.txt` (176 paths) in the same scratchpad folder. Page numbers below refer to this guide.
- **"Which ChemDraw is right for me" v23.0 chart** (Revvity, June 2025):
  https://assets.revvitysignals.com/m/6e4aae70c40d28af/original/Which-ChemDraw-is-right-for-me-v-23-0.pdf
  (text extracted to the scratchpad file `which_chemdraw.txt`). It lists the v22/23 Professional features: enhanced
  stereo perception, M/P labels for allenes and atropisomers, pKa/logP/logS predictions (Windows), ¹H/¹³C NMR
  prediction, HELM biopolymers with FASTA, 3D clean-up, Structure Perspective depth fading, 3MF export,
  atom/bond/ring-fill highlighting, Smart Paste, SciFinder/Reaxys integration and add-ins.
- **ChemDraw 23.0 release notes**, as summarised by Revvity and the UiO mirror
  (https://www.mn.uio.no/kjemi/tjenester/it/hjelp2/chemdraw/chemdraw-suite-v23.0-release-notes.pdf): dark-mode style
  sheet, atropisomer perception, Smart Paste, an H-bond option in 3D clean-up, and Signals licensing. The direct
  Revvity PDF returned HTTP 403.
- **Gaps in the sources.** I found no v22/v23 user guide published as a standalone official PDF. The v22 guide
  exists only as a third-party Scribd upload, which I did not use. The 17.1 guide covers the menus and tools that
  later versions still have. Items new since 17.1 come from the v23 chart.

**How I checked the app**
- I read the menus, palette, docks, model, ops, the hotkey handler and the core module docstrings.
- The existing suites all pass headlessly: `tests/test_chemdraw.py`, `tests/test_chemdraw_gaps.py` and
  `tests/test_chem_editor.py` give **103 passed in 7.7 s** (offscreen).
- A headless probe (`scratchpad/probe/probe.py`) confirmed three things:
  - Structure→Name returns nothing offline for a structure outside the name database (`CC(C)C(O)CCC(C)Cl`); the
    message tells the user to switch on PubChem.
  - A CDXML file written by RDKit loads in the app's reader.
  - RDKit reads the app's own CDXML (aspirin round-trips).
- In the venv (RDKit 2026.09.1) I confirmed:
  - `rdCIPLabeler` gives R/S and E/Z labels.
  - Atropisomer bond stereo (`STEREOATROPCW`) exists.
  - StereoGroups (AND/OR/ABS) are written to V3000 collections.
  - `MolFromHELM` / `MolToHELM` / `MolFromSequence` work for peptides, DNA and RNA.
  - `rdMolEnumerator` (Markush enumeration) is present.
  - `rdChemDraw.MolsFromChemDrawFile` (CDX/CDXML) is present and `Chem.HasChemDrawCDXSupport()` returns True.
    The wheel ships `libRDKitChemDraw`. The Python `MolToChemDrawBlock(..., CDX)` fails with a UTF-8 decode error,
    so binary CDX **writing** is not usable from Python.

**Legend:** ✅ present (works like ChemDraw) · 🟡 partial (what is missing is named) · ❌ missing · 🔒 licence-restricted
in ChemDraw (an open alternative is given).

---

## 1. Feature inventory with status

### 1.1 File menu
| ChemDraw | App | Notes |
|---|---|---|
| New Document, Open, Open Recent, Close (Ctrl+W), Save, Save As | ✅ | Opening a file in a non-empty document opens a new window, as ChemDraw does. |
| Open Style Sheets / Apply Document Settings from | 🟡 | The app has only 2 presets (`New Document`, `ACS Document 1996`, see `core/chem_draw/style.py:58`). ChemDraw ships about 12. The 17.1 guide (pp. 48–53) gives exact values for ACS 1996, Adv. Synth. Catal., J. Chin. Chem. Soc., J. Mol. Mod., New Document, New Slide, Phytomedicine, RSC, Science of Synthesis, Synthesis/Synlett, Helv. Chim. Acta and Wiley; Nature Research was added in 17.1 and a dark style in v23. These are numeric facts and can be added directly, e.g. RSC: 0.43 cm bond, 0.016 cm line, 0.056 cm bold, 20 % spacing, 9 pt labels. |
| Open Templates (Amino Acids, Aromatics, Bicyclics, Conformers, Clipware/glassware, Cp rings, Hexoses, Newman/Fischer, Polyhedra, Stereocenters, Supramolecules, BioDraw art …) | 🟡 | The app has 6 SMILES categories: amino acids, nucleobases, sugars, rings, functional groups, protecting groups. **No glassware/clipware, Newman, Fischer, Haworth, conformers (chairs exist only as two ring tools), polyhedra or bio-art.** |
| Open Samples, Document Annotations, List Nicknames, Page Setup, Revert | ❌ | Page Setup is limited to Letter/A4 in View. |
| Document Settings (dialog) | 🟡 | Covers bond length, widths, spacing and fonts. Missing: chain angle, multi-page drawing area, per-object settings ("Object Settings"). |
| Preferences (Building/Display, ChemNMR frequency/solvent, parentheses on stereo labels …) | ❌ | Partly covered by QSettings flags. |
| Print | ✅ | Prints to PDF (tested). |
| Import / Export (see §1.12) | 🟡 | |
| ChemScript / Run ChemScript 🔒 | ❌ | Licensed Python API. Open alternative: an embedded Python console with RDKit and the `Document` API (low priority). |
| Signals / ChemDraw Cloud 🔒 | ❌ (n/a) | Out of scope. |

### 1.2 Edit menu
| ChemDraw | App | Notes |
|---|---|---|
| Undo/Redo, Cut/Copy/Paste, Clear, Select All | ✅ | |
| Copy As: SMILES, InChI, InChIKey, CDXML Text, MOL (Alt+Shift+Ctrl+O), HELM, SLN | 🟡 | The app has SMILES, InChI, InChIKey, rxn-SMILES, CDXML, MOL, PNG and SVG. **HELM and SLN are missing.** ChemDraw's Copy as CDXML shortcut is Ctrl+D (p. 61). |
| Paste / Paste Special: SMILES, MOL, InChI, CDXML, HELM, FASTA DNA/PEPTIDE, Name | 🟡 | Text paste covers SMILES/MOL/InChI/RXN through Import. **HELM, FASTA and "Paste as Name" are missing**; RDKit can supply the first two (verified). |
| Smart Paste (v23: paste into free space) | 🟡 | `_free_spot` exists in `chem_editor/window.py:458`. Check that paste uses it. |
| Repeat last command (Ctrl+Y), Repeat Rotate | ❌ | |
| Insert Object / Get 3D Model (Chem3D OLE) 🔒 | ❌ | Not applicable. |

### 1.3 View menu
| ChemDraw | App | Notes |
|---|---|---|
| Show Main/General/Style toolbars, Templates | ✅ | |
| Show Analysis Window | ✅ | |
| Show Chemical Properties Window | ✅ | |
| Show Periodic Table Window | 🟡 | A modal picker dialog. It shows no element data. |
| Other Toolbars: Bracket, Chemical Symbols, Pen Tools, Query Tools, Drawing Elements, Chromatography (TLC/Gel), Orbitals, Fragmentation, BioDraw, Biopolymer, HELM Monomers | 🟡 | Bracket, pen, orbitals and fragmentation exist as fly-outs. **Query, Drawing Elements, Chromatography, BioDraw and Biopolymer/HELM are missing.** |
| Show Info Window, Rulers (F11), Crosshair, Character Map Window, Show Preview | ❌ | |
| Show Chemical Warnings (live wavy-box warnings, "Explain this warning") | 🟡 | Live valence marks and a status-bar count exist. No per-object toggle and no explanation tooltip. |
| Show Reaction Interpretation (reactants red, products magenta, conditions cyan/blue on hover; p. 170) | 🟡 | Reaction splitting works in `convert.py`. There is no visual interpretation overlay. |
| Magnify/Reduce/Actual/Fit (ChemDraw keys **F7/F8/F5/F6**, p. 61) | 🟡 | Present, but on Ctrl+]/[ and Ctrl+0/Ctrl+9. In the editor **F5 is Check Structure**, which conflicts with ChemDraw's F5 = Actual Size. |
| Show Grid, Page Size | ✅ / 🟡 | Page size offers Letter/A4 only, with no multi-page or poster layout. |
| Dark mode style (v23) | ❌ | |

### 1.4 Object menu
| ChemDraw | App | Notes |
|---|---|---|
| Fixed Lengths (Ctrl+L), Fixed Angles (Ctrl+E) | 🟡 | Present without these shortcuts. |
| Bring to Front (F2) / Send to Back (F3), Group/Ungroup, Align, Distribute, Center on Page | ✅ | The shortcuts differ: Ctrl+Shift+]/[ instead of F2/F3. |
| Join (Ctrl+J), Flatten (3D→2D), Add Frame, Attach Data, Annotate | ❌ | |
| Flip H/V, Rotate 180° H/V, Rotate… (Ctrl+R opens the dialog), Scale… (Ctrl+K) | 🟡 | Present, but Ctrl+R is "Rotate 90° CW" and Scale has no shortcut. There is no "Rotate atom labels" option. |
| Object Settings / Apply Object Settings from (per-object style) | 🟡 | Only colour and text style per object. |
| Show Stereochemistry (R/S, r/s, E/Z indicators, "Include parentheses", Hide/Position indicator; pp. 193–195) | ❌ | **Not in the drawing editor.** CIP labels exist only in stage-1 depiction (`core/chem/depict.py:34`). RDKit `rdCIPLabeler` gives correct labels (verified). |

### 1.5 Structure menu (cited paths from the 17.1 guide, plus the v23 chart)
| ChemDraw | App | Notes |
|---|---|---|
| Atom Properties, Bond Properties, Bracket Properties | 🟡 | Atom Properties ✅ and Arrow Properties ✅. **No Bond Properties dialog** (query/topology/reaction-centre flags) and **no Bracket Properties dialog** (polymer SRU type, repeat pattern, connectivity). |
| Expand Label ✅ / Contract Label ❌ / Define Nickname ❌ | 🟡 | 75 built-in nicknames (`abbrev.py`). Users cannot define their own. |
| Check Structure, Clean Up Structure (Shift+Ctrl+K) | ✅ | |
| Clean Up Reaction | ❌ | |
| Convert Name to Structure (Shift+Ctrl+N) | ✅ | Offline list, a 150k-name database, OPSIN (MIT, bundled with a JRE) and optional PubChem. |
| Convert Structure to Name (Alt+Ctrl+N) 🔒 | 🟡 | **Only known compounds**: exact InChIKey hits in the offline database, or PubChem when online. ChemDraw generates IUPAC names for any structure. See gap G2. The app's shortcut is Ctrl+Alt+Shift+N. |
| Add Structure to Dictionary | ❌ | |
| Predict ¹H-NMR / ¹³C-NMR Shifts (ChemNMR: shifts on the structure, line spectrum, protocol, 300 MHz default, CDCl₃/DMSO; pp. 217–224) 🔒 | ❌ | See gap G3. |
| Make / Clear Spectrum-Structure Assignment | ❌ | |
| Map Reaction Atoms / Clear Reaction Map | 🟡 | A manual map-number tool exists. **No automatic mapping and no clear command.** |
| Autonumber Reaction (bold compound numbers 1, 2, 3 …) | ❌ | |
| Analyze Stoichiometry (live grid under the reaction, Set Limiting; p. 173) | 🟡 | Present as a dialog (`StoichiometryDialog`), not as a live grid object in the document. |
| Enhanced Stereochemistry (ABS / AND / OR groups; saved to CDX, CDXML, MOL/RXN V3000) | ❌ | RDKit StereoGroups and V3000 collections were verified. |
| M/P labels for allenes and atropisomers (v23 chart) | ❌ | RDKit perceives atropisomer bond stereo (verified). Whether `rdCIPLabeler` produces M/P labels still needs checking. |
| R-Logic Query, Expand Generic Structure, Add Multi-Center Attachment, Add Variable Attachment, Add 3D Property | ❌ | Generic labels R/X/R1 become dummy atoms (🟡). RDKit `rdMolEnumerator` can expand variable attachments and link nodes. |
| Structure Perspective (rotate in 3D) / Mass Fragmentation tool | ✅ | ETKDG + MMFF; homolytic fragments with formula and mass. |
| Chem3D hotlink / Chem3D 🔒 | 🟡 | Structure Perspective is a viewer only. v23 3D clean-up (with H-bonds), depth fading and 3MF export are missing. |
| Biopolymer editor, HELM monomers, FASTA, disulfide/lactam bridges, Watson-Crick tools 🔒(parts) | ❌ | RDKit `MolFromSequence` / `MolFromHELM` cover sequence→structure and HELM for peptides and simple RNA (verified). |

### 1.6 Text, Curves, Colors, Window, Help
| ChemDraw | App | Notes |
|---|---|---|
| Text: Font, Size, Style (plain, bold, italic, underline, formula, sub/superscript), Justification, line spacing, Interpret Chemically | 🟡 | Font, size and style ✅. **No justification or line spacing.** The `TextBox.chemical` flag exists but has no "Interpret chemically" UI. |
| Curves: Plain, Dashed, Bold, Double, Closed, Filled, Faded, Shaded, arrowheads (full/half, start/end), Edit Curve tool | 🟡 | Everything except **Faded**. Editing works through the Pen tool's handle drag. |
| Colors: palette, Other…, By Element | ✅ | Atom/bond/ring-fill highlighting (v23) is missing. |
| Window: document list, Minimize | ✅ | |
| Help: Contents, Hotkeys cheat sheet | 🟡 | Only a hotkeys page, and it does not match ChemDraw's (see §1.11). |

### 1.7 Search / Online menu
| ChemDraw | App | Notes |
|---|---|---|
| Search SciFinder 🔒, Reaxys 🔒, ChemACX 🔒 (Find Structure from CAS-RN, Find Suppliers), Similarity search | ❌ | Open alternatives: PubChem PUG-REST structure, similarity and substructure search, CAS-RN→structure through PubChem synonyms, and the **PubChem LCSS / GHS safety** summary (open data; matches the v23 "PubChem Safety add-in"). The app already talks to PubChem for names (opt-in). |
| Google Scholar/Patents add-in | ❌ | A URL launcher would do (trivial). |

### 1.8 Main Toolbar tools and palettes
| ChemDraw tool / palette | App | Notes |
|---|---|---|
| Lasso, Marquee, Structure Perspective/rotate, Eraser, Text, Acyclic Chain | ✅ | |
| Bonds: solid, double (L/R/centred), triple, quadruple, dashed, hashed, hashed wedged, wedged, hollow wedged, bold, wavy, dative, "either", aromatic (dashed double), bold double, tautomeric | 🟡 | Present: single/double/triple/aromatic/dashed/hashed/hash-wedge/wedge/hollow/bold/wavy/dative, plus "either" in the model. **Missing: quadruple, bold-double (Shift+B hotkey), dashed-double/tautomeric variants.** |
| Rings: 3–8, benzene, cyclopentadiene, two chairs | ✅ | |
| Atom/element tools, periodic table | ✅ | |
| Arrows palette: plain, equilibrium, unbalanced equilibrium, retro, resonance, curved full/half (fishhook), no-go (crossed), dashed, bold, hollow/block (filled/faded/shaded), double-headed, arc arrows (90/180/270), dipole (context menu), connecting arrows, curved arrows anchored to bonds/atoms/lone pairs | 🟡 | 7 kinds (`ARROW_KINDS`): forward, equilibrium, retro, resonance, curved, fishhook, line. Curved arrows bend. **Missing: unbalanced equilibrium, no-go, dashed/bold, block arrows, double-headed, arc arrows, dipole, arrow connections, and anchoring of mechanism arrows to bonds or lone pairs.** |
| Chemical Symbols palette: ⊕/⊖ circled, + / −, radical •, radical cation/anion, **lone pair (:)**, δ+/δ−, ‡, partial charges; associated with the atom | 🟡 | Present: +/−, radical, map number. **Missing: lone pairs, δ+/δ−, circled charges, radical ions, ‡.** Mechanism teaching needs these. |
| Brackets: [ ], ( ), { } paired and single, polymer bracket properties | 🟡 | Paired only, with a free label (`n`). **No single brackets, SRU/copolymer semantics or MOL Sgroup export.** |
| Drawing elements: circle, oval, rectangle (rounded/shadowed), line (dashed/bold/arrow), arcs 90/120/180/270 | ❌ | Only Pen curves and the "line" arrow. |
| Orbitals: s, p, sigma, single lobe, hybrid, d(xy), d(z²) × plain/shaded/filled/phased | ✅ | The sigma orbital is missing (minor). |
| Chromatography: **TLC plate** (lanes, spots, Rf display/edit, crescents, smears, custom spots; pp. 212–216) and **Gel electrophoresis** (lanes, bands, scale, Excel paste) | ❌ | |
| Mass Fragmentation tools | ✅ | |
| Query tools (generic atoms A/Q/X/M, atom lists/NOT lists, H-count, substituents, ring/chain bond queries, reaction-centre marks, stereo flags, 3D properties; pp. 274–312) | ❌ | Low priority for teaching. |
| Tables tool, Pen tools | ❌ / ✅ | |
| Templates tool | 🟡 | Template fly-out plus docked panel; library is thin (see §1.1). |
| BioDraw palette (membranes, DNA, enzymes, receptors, cells, tRNA …) 🔒 art | ❌ | Open-licensed art: Servier Medical Art (CC BY 4.0), Bioicons (per-icon CC0/CC BY/MIT). |

### 1.9 Chemistry intelligence windows
| ChemDraw | App | Notes |
|---|---|---|
| Analysis window: formula, exact mass, MW, m/z, elemental analysis, Paste/caption | ✅ | Matches p. 193. Decimals control and a Copy button were added. |
| Chemical Properties: BP, MP, Tc, Pc, Vc, Gibbs energy, ΔHf, LogP, **CLogP/CMR 🔒 (BioByte)**, MR, tPSA, LogS, **pKa 🔒 (MOSES)**, **Henry's law constant** (p. 198) | 🟡 | Present with open methods and labelled sources: experimental database, ML models, Joback, Stein-Brown, Marrero-Gani (levels 1–3), Nannoolal, CDK JPlogP/XLogP/ALogP/AMR, ESOL, plus vapour pressure. **Missing: pKa and Henry's law constant.** CLogP is replaced by an open consensus; that is the right call. |
| ChemNMR ¹H/¹³C 🔒 | ❌ | Gap G3. |
| Isotope pattern / m/z | ✅ | `sections.isotope_pattern`. |
| Stereochemistry display | ❌ | Gap G1. |

### 1.10 Reactions
| ChemDraw | App |
|---|---|
| Reaction drawing, arrows, "+", conditions above/below | ✅ |
| Reaction interpretation | 🟡 logic only, no overlay |
| Ctrl+arrow keys: duplicate the selection across a new arrow (p. 60) | ❌ |
| Balance check | ✅ (app extra) |
| Stoichiometry grid | 🟡 dialog |
| Atom-atom mapping | 🟡 manual only |
| Clean Up Reaction | ❌ |
| Autonumber Reaction | ❌ |
| Reaction SMILES / RXN export | ✅ |
| RXN V3000 export | ❌ |

### 1.11 Hotkeys (ChemDraw 17.1+ "enhanced hotkeys", pp. 54–63) vs the app
- **Atom keys**
  - Matching element keys: c, n, o, s, p, f, i, h, k, l=Cl, b=Br, B, plus +, − and Enter.
  - Different: ChemDraw `.` adds an attachment point (the app uses it for a radical), `'` adds an atom number, and
    `/` or `?` opens Atom Properties (the app uses `/` too).
  - Different: ChemDraw `=` opens the nickname chooser (the app uses it to increase charge).
  - Missing labels: `q`=O, `w`=N, `m`=Me, `e`=Et, `Shift+p`=Ph, `Shift+a`=Ac, `Shift+e`=COOCH₃, `r`=R, `x`=X, `d`=D,
    `Shift+s`=Si, `Shift+n`=Na.
  - **ChemDraw number keys "sprout" groups:** 1 linear bond, 0 cyclic bond, 2 carbonyl/acetyl, 3 or a phenyl,
    4 wedge+Me, 5 hash+Me, 6 cyclohexyl, 7 cyclopentyl, 8 methylidene, 9 gem-dimethyl/iPr, z alkyne.
    **In the app, 3–8 attach a ring of that size instead.** Users with ChemDraw muscle memory will get wrong results.
- **Bond keys**
  - Matching: 1/2/3, d, y, l/r/c, b, w, h.
  - Missing: Shift+b (bold double), Shift+h (hashed), f (bring to front), a (fuse benzene), z (fuse cyclopentadiene),
    v (fuse cyclopropane), 9/0 (chairs).
  - Conflicts: the app's `v` makes a dative bond, the app's `a` makes an aromatic bond, and the app's `o` (hollow wedge)
    is not a ChemDraw key.
- **Tool keys** when the pointer is off any object: Space = Marquee, x = bond, Shift+x = chain, t = text,
  Shift+t = brackets, e = arrows, Shift+e = symbols, j = benzene, Shift+j = Cp, u = cyclohexane,
  Shift+u = cyclopentane, g = TLC/chromatography, Shift+g = orbitals, l = pen, Shift+l = drawing elements,
  k = fragmentation, Shift+k = query. **The app has none of these.**
- **Menu shortcuts:**
  - View: F5 actual, F6 fit, F7 magnify, F8 reduce, F11 rulers.
  - Object: Ctrl+L, Ctrl+E, Ctrl+J, F2/F3, Ctrl+R (rotate dialog), Ctrl+K (scale),
    Alt+Shift+Ctrl+L/C/R/T/M/B (align).
  - Edit: Alt+Ctrl+C (copy SMILES), Alt+Ctrl+P (paste SMILES), Ctrl+D (copy CDXML).
  - Structure: Alt+Ctrl+N (structure to name).
  - The app differs in most of these.

### 1.12 File formats
ChemDraw 17.1 import/export tables (pp. 320–322) compared with the app:

| Format | ChemDraw | App |
|---|---|---|
| CDX (binary) | read / write | ❌. RDKit can read the **chemistry** (no graphics) through the BSD Revvity code. |
| CDXML | read / write | 🟡 Subset read/write: fragments, text, arrows, brackets, curves, orbitals. Interop with RDKit verified both ways. |
| MOL / SDF V2000 | read / write | ✅ |
| MOL / SDF / RXN V3000 | read / write | 🟡 Read only through RDKit. **No V3000 export**, so enhanced stereo and Sgroups are lost. |
| RXN | read / write | ✅ |
| RGfile | read / write | ❌ |
| SMILES, InChI / InChIKey | read / write | ✅ (InChI is export / copy) |
| HELM, FASTA | read / write | ❌ |
| SLN, SKC/TGF, MSM, SMD | read / write | ❌ (obsolete, skip) |
| JCAMP-DX (.jdx, .dx), Galactic SPC | import (spectra) | ❌ |
| PNG | read / write | ✅ export |
| JPEG, TIFF, GIF, BMP | read / write | ❌ (Qt can write them trivially) |
| SVG | export | ✅ |
| PDF | read / write | ✅ export |
| EMF / WMF (Office editable vector) | read / write | ❌. Qt has no EMF writer. On Windows, Office accepts SVG and PNG. |
| EPS | read / write | ❌ |
| Templates CTP, style sheets CTS | read / write | ❌ |
| 3MF (3D print, v23) | write | ❌ |
| Clipboard | CDX + CDXML + EMF + MOL; OLE object in Office | 🟡 Picture + text. No embedded editable object. |

### 1.13 Integrations
| ChemDraw | App | Notes |
|---|---|---|
| ChemDraw for Excel / Office 🔒 (Windows) | 🟡 | Equivalent inside the app: "Insert SMILES into Data Editor", stage-1 descriptors from dataset columns, and the Output Viewer. |
| Add-in SDK 🔒 | ❌ | Not needed. |

---

## 2. Licence issues and open-source equivalents

The project's own data policy (`rsstatchem/resources/properties/README.md`) bundles **only openly licensed or
public-domain data**. It explicitly rejects CC BY-NC sources such as DrugBank and JECFA. The recommendations below
follow that rule.

| ChemDraw feature (licensed part) | Open equivalent | Licence | Size / platform notes | Verdict |
|---|---|---|---|---|
| CIP stereo labels, enhanced stereo, atropisomers | RDKit `rdCIPLabeler`, StereoGroups, V3000 | BSD-3 | Already a dependency | **Use** |
| CDX binary I/O | (a) RDKit `rdChemDraw.MolsFromChemDrawFile`, built on **Revvity's open-sourced ChemDraw format code** (github.com/Glysade/chemdraw, "Revvity Open Source ChemDraw File Format", **BSD 3-Clause**; RDKit fetches tag v1.0.14). Molecules and reactions only. (b) A pure-Python CDX tag reader/writer written against the same BSD code and the public CDX spec, for full documents. (c) Indigo CDX loader/saver. | BSD-3 / Apache-2.0 | (a) no new dependency. Confirm `HasChemDrawCDXSupport()` on the Windows wheel in CI. | **Use (a) now; (b) for full fidelity** |
| HELM / FASTA / sequences | RDKit `MolFromHELM`, `MolToHELM`, `MolFromSequence` (peptide, DNA, RNA flavours) | BSD-3 | Already a dependency. Peptides plus simple RNA; no full HELM monomer library. | **Use** for paste and copy |
| Markush / variable attachment expansion | RDKit `rdMolEnumerator` | BSD-3 | Already a dependency | Use (later) |
| Name→Structure | OPSIN 2.9.0 (already bundled with a jlinked JRE) | MIT | about 14 MB | ✅ done |
| Structure→Name 🔒 | No mature open IUPAC name generator exists. Options: (1) **in-house rule-based namer for teaching-scope classes**, accepted only if **OPSIN parses the name back to the same InChIKey**; (2) the existing 150k-name database and PubChem; (3) STOUT v2 (MIT): TensorFlow, large models, names not guaranteed correct. | own / MIT | (1) pure Python | **(1) plus round-trip check; reject STOUT** |
| ChemNMR ¹H/¹³C 🔒 (additivity rules, p. 218) | **HOSE-code predictor** (Bremser 1978; the method used by nmrshiftdb2 and nmrdb.org) trained on **nmrshiftdb2** assigned spectra. HOSE codes in Python/RDKit, or CDK `HOSECodeGenerator` through the existing CDK helper (LGPL, jars already bundled). Spectrum drawn with first-order multiplets in numpy; **nmrsim** (MIT) only as a reference, because it needs `sparse` and pulls numba. | Data: "nmrshiftdb2 Data License" (custom "open content" licence; the site says spectra are free to use; its predecessor used GNU FDL). Code: own. | A HOSE table of a few MB, xz-compressed. | **Use, but read `nmrshiftdb2datalicense.txt` first** (the server was unreachable during research). If share-alike or attribution applies, ship it as a separately downloaded resource with a notice, like CDK/OPSIN. Do **not** copy Pretsch/ChemNMR increment tables (copyrighted book data). |
| pKa 🔒 (MOSES) | **Gradient-boosted trees in the existing `ml_props` numpy pipeline**, trained on the **Mansouri et al. 2019 OPERA pKa dataset** (J. Cheminform. 11:60, about 7.9k compounds from DataWarrior; BMC data are under a CC0 waiver unless stated otherwise; OPERA code MIT). **Dimorphite-DL** 2.1.0 (Apache-2.0, 0.1 MB, needs only RDKit and loguru) can find ionisable sites and approximate site pKa for display. | CC0 / MIT / Apache-2.0 | No runtime ML dependency; sklearn is dev-only, as now. | **Use** |
| pKa (rejected) | IUPAC Digitized pKa Dataset (**CC BY-NC 4.0**, conflicts with the project policy); MolGpKa (MIT code, but trained on ACD/pKa *predictions*, needs torch); pkasolver and Uni-pKa (torch, torch-geometric, hundreds of MB) | | | **Avoid** |
| logP (CLogP 🔒 BioByte) | Already open: CDK JPlogP/XLogP/ALogP, ML model, consensus | LGPL / own | Bundled | ✅ |
| Henry's law constant | HLC ≈ P_vap·M / S_w from existing estimates; or the Meylan & Howard (1991) bond-contribution method (published, implement from the paper) | own | none | **Use** |
| Automatic atom-atom mapping | **Indigo** `automap` (epam.indigo 1.46.0, **Apache-2.0**). Indigo also reads and writes CDX/CDXML and has a structure checker. | Apache-2.0 | wheels: **5.8 MB macOS arm64, 9.3 MB win_amd64**, no Python dependencies. PyInstaller must collect its native library. | **Optional dependency, recommended** |
| Atom mapping (rejected) | RXNMapper (MIT, but torch + transformers, more than 700 MB); chython-rxnmap (**LGPL-3**, 78.5 MB ONNX weights + onnxruntime 21/15 MB); LocalMapper (GPL-3, torch/dgl) | | | **Avoid** |
| Structure toolkit (rejected) | Open Babel (openbabel-wheel, **GPL-2**, copyleft for the bundled app) | GPL | 11.6 / 5.1 MB | **Avoid** |
| SciFinder / Reaxys / ChemACX 🔒 | PubChem PUG-REST (search, CAS-RN synonyms), PubChem LCSS GHS safety | Public domain / open | Network, opt-in like the current name lookup | Use |
| Clipware / glassware / BioDraw art 🔒 (copyrighted art) | Draw in-house SVGs, or use **Servier Medical Art** (CC BY 4.0, attribution) and **Bioicons** (per-icon licence; keep only CC0/CC BY/MIT) | CC BY / CC0 | Small SVGs | Use, with an attribution file |
| Chem3D 🔒 | RDKit ETKDG + MMFF (present). The 3MF writer is a small zip+XML file. | BSD | none | Optional |
| ChemScript 🔒 | Python console with RDKit and the `Document` API | BSD | none | Low priority |

**Dependency summary**
- Almost every gap closes with **RDKit (already present) plus our own code and data**.
- The only recommended new runtime wheel is **epam.indigo** (optional, about 6–9 MB per platform, Apache-2.0). Both
  target platforms (macOS arm64, Windows x64) have wheels.
- New data: a pKa model npz (about 1–2 MB, built like `ml_models.npz`) and an NMR HOSE table (a few MB, pending the
  licence check).
- **No torch, TensorFlow or onnxruntime.** Release zips are about 165/207 MB today and should grow by roughly
  20 MB at most.

---

## 3. Prioritised gaps (what chemistry lecturers and students use most)

Ranked by teaching impact: organic, analytical and biochemistry courses, plus lab reports and theses.

1. **G1 – Show Stereochemistry (R/S, E/Z, r/s)** in the drawing, toggled per object or document, with the
   "parentheses" preference. Used constantly in organic chemistry. Low effort (RDKit). ❌
2. **G2 – Structure→Name for arbitrary teaching-level structures.** Today only known compounds get a name. Build a
   rule-based IUPAC namer covering alkanes and cycloalkanes, alkenes/alkynes, haloalkanes, alcohols/ethers,
   aldehydes/ketones, acids and derivatives (esters, amides, nitriles, acyl halides, anhydrides), amines,
   monocyclic aromatics/benzene derivatives, and simple stereo descriptors. Accept a name **only when OPSIN round-trips
   it to the same InChIKey**; otherwise fall back to the database or PubChem. High value; medium-high effort. 🟡
3. **G3 – NMR prediction (¹H/¹³C)**: shifts on atoms, a predicted spectrum window (frequency and solvent settings),
   and atom↔peak highlighting. Plus **JCAMP-DX import** to overlay measured spectra. Core of spectroscopy courses.
   Medium-high effort. ❌
4. **G4 – Mechanism drawing kit.** Lone pairs, δ+/δ−, circled charges, radical ions; curved arrows anchored to
   bonds and lone pairs; the full arrow palette (no-go, unbalanced equilibrium, dashed, block, double-headed, arc);
   Ctrl+arrow duplication. 🟡
5. **G5 – ChemDraw-compatible hotkeys and shortcuts**: sprout keys, generic tool keys and F-key zoom (§1.11). The
   F5 conflict matters. Students who learned ChemDraw expect these. 🟡
6. **G6 – Lab-report objects.** TLC plate (Rf), gel electrophoresis, glassware/clipware templates, drawing elements
   (boxes, circles, arcs), Autonumber Reaction, a live stoichiometry grid object. ❌/🟡
7. **G7 – Conformation and projection templates**: Newman, Fischer, Haworth, chair conformers, plus more template
   families and the journal style sheets (RSC, Wiley, New Slide …, values in §1.1). 🟡
8. **G8 – Chemical Properties: pKa and Henry's law** (open models, labelled like the existing rows). 🟡
9. **G9 – Interop**:
   - CDX import (RDKit) and a pure-Python CDX writer;
   - MOL/RXN V3000 export (enhanced stereo, polymer Sgroups);
   - HELM/FASTA paste and copy;
   - RGfile, TIFF/JPEG;
   - Copy As SMILES/MOL shortcuts. 🟡
10. **G10 – Reactions**:
    - automatic atom mapping (Indigo, optional) and Clear Map;
    - Show Reaction Interpretation overlay;
    - Clean Up Reaction. 🟡
11. **G11 – Biopolymers**: sequence paste (FASTA/HELM) giving a structure, and a peptide sequence tool.
    Full HELM monomer editing is later. ❌
12. **G12 – Less used**:
    - query tools / R-logic / Markush;
    - enhanced stereo groups UI;
    - per-object settings, multi-page documents, rulers/crosshair/Info window, Tables;
    - Character Map;
    - PubChem search/GHS safety;
    - 3D clean-up, depth fading, 3MF;
    - dark style;
    - Python scripting console.

---

## 4. Work packages (disjoint file ownership)

**Integration contract.** `ui/chemdraw/window.py` belongs to WP-C. WP-A and WP-B each own one "actions" module with
an `install(window)` function, which WP-C calls once from `ChemDrawWindow._build_menus`/`_build_docks`. Each module
adds its own menu items to the named menus, so nobody else edits `window.py`.

**Shared data contract.** WP-B adds the model fields: `Atom.show_stereo`, `Style.show_stereo` and
`Style.stereo_parens`. WP-A only provides pure functions:
- `stereo.labels(doc, atoms=None) -> {("atom", id): "R" | "S" | "r" | "s" | "M" | "P", ("bond", id): "E" | "Z"}`;
- `nmr_predict.predict(mol, nucleus) -> [(atom_idx, shift, multiplicity, J)]`.

### WP-A – Chemistry intelligence (no canvas or rendering code)
**Owns (new files):**
- `core/chem_draw/stereo.py` — CIP via `rdCIPLabeler`, StereoGroups, atropisomer M/P if available;
- `core/chem_draw/iupac_namer.py` — rule-based namer with OPSIN round-trip;
- `core/chem_draw/pka.py`, `scripts/train_pka_model.py`, `resources/properties/pka_model.npz`;
- `core/chem_draw/nmr_predict.py`, `scripts/build_nmr_db.py`, `resources/nmr/*`, `core/chem_draw/jcamp.py`;
- `ui/chemdraw/nmr_window.py` — spectrum view, atom↔peak highlighting, 300 MHz / CDCl₃ settings;
- `ui/chemdraw/intel_actions.py` — Structure: Predict ¹H/¹³C NMR, Show Stereochemistry toggle; File: Open Spectrum (JCAMP);
- `tests/test_chem_intel.py`.

**Owns (existing files):**
- `core/chem_draw/properties.py` — pKa and Henry rows;
- `core/chem_draw/names.py` — namer as tier 0, before the database;
- `ui/chemdraw/docks.py` — Chemical Properties pKa/HLC rows;
- `resources/properties/README.md` — licence rows.

**Licence gate:** read and record the nmrshiftdb2 data licence before bundling. pKa uses only CC0/MIT data.

**Acceptance:**
- (R)-alanine shows S; trans-2-butene shows E; cis-decalin shows r/s.
- At least 90 % of a curated teaching set (about 200 structures from intro organic textbooks, with names from
  PubChem) get a name that OPSIN round-trips. Other structures get no name, never a wrong one.
- ¹³C MAE ≤ 3 ppm and ¹H MAE ≤ 0.3 ppm on a held-out nmrshiftdb2 split.
- pKa RMSE reported against the Mansouri test split.

### WP-B – Drawing objects, tools and rendering
**Owns:**
- `core/chem_draw/model.py`, `geometry.py`, `ops.py`, `templates.py`, `style.py`, `abbrev.py`, `labels.py`,
  `cdxml.py` (round-trip of all new objects);
- `ui/chem_editor/canvas.py`, `render.py`, `icons.py`, `dialogs.py`, `panels.py`;
- `ui/chemdraw/palette.py`, `icons.py`, `canvas.py`, `ui/chemdraw/objects_actions.py` (new);
- `resources/templates/` (new SVG clipart plus `ATTRIBUTION.md`);
- `tests/test_chemdraw_objects.py`.

**Scope:**
- chemical-symbols palette: lone pair, δ±, circled charges, radical ions;
- full arrow palette, arrow anchoring and connection;
- drawing elements (circle, oval, rectangle, arcs) and single brackets;
- polymer bracket properties (SRU "n", copolymer, head-to-tail);
- **TLC plate and gel objects** (lanes, spots/bands, Rf display and edit);
- stereo indicator rendering (uses WP-A's `stereo.labels`);
- ChemDraw 17.1+ **hotkeys** (sprout, bond fuse, generic tool keys) in `canvas.hotkey`, with HOTKEYS_HTML text
  supplied to WP-C;
- Newman / Fischer / Haworth / conformer templates, glassware clipart;
- the 10+ journal **style presets** from §1.1;
- Faded curve style, quadruple and bold-double bonds;
- Contract Label, Define Nickname, user nickname file;
- Ctrl+arrow "duplicate across arrow";
- Autonumber Reaction captions.

**Acceptance:** every new object round-trips through `.rsc` and CDXML; RDKit `MolsFromCDXMLFile` still parses the
chemistry; the hotkey table reproduces 17.1 pp. 54–58 one to one (one test per key).

### WP-C – Shell, menus, file formats, interop
**Owns:**
- `ui/chemdraw/window.py`, `ui/chemdraw/lookup.py`;
- `ui/chem_editor/window.py`, `ui/chem_editor/export.py`;
- `core/chem_draw/convert.py`;
- new modules: `core/chem_draw/cdx.py` (pure-Python CDX reader/writer for document objects, with RDKit
  `rdChemDraw` as the chemistry fallback), `core/chem_draw/sequence.py` (HELM/FASTA via RDKit),
  `core/chem_draw/mapping.py` (optional Indigo automap, used only if installed);
- `scripts/` and packaging spec changes for Indigo;
- `requirements.txt` (optional extra);
- `tests/test_chemdraw_io.py`.

**Scope:**
- menu parity with the 17.1 path list: Edit > Paste Special (HELM, FASTA DNA/Peptide, Name), Copy As (MOL V3000,
  HELM), Repeat; Object > Join; Structure > Map Reaction Atoms / Clear Reaction Map / Clean Up Reaction;
  View > Show Reaction Interpretation (a request API to WP-B's renderer), Rulers, Crosshair; File > Open Style Sheets,
  Document Annotations;
- ChemDraw shortcut table (F5–F8, F2/F3, Ctrl+L/E/J/K/R, Alt+Ctrl+N, Alt+Ctrl+C/P), moving Check Structure off F5;
- exports: CDX, MOL/RXN V3000, RGfile, TIFF/JPEG, 3MF (optional);
- calling `intel_actions.install` and `objects_actions.install`;
- PubChem structure search and GHS safety (opt-in network, like the current lookup).

**Acceptance:**
- CDX files produced by ChemDraw open: test fixtures from RDKit's `External/ChemDraw/test_data` (BSD).
- V3000 keeps StereoGroups.
- `HasChemDrawCDXSupport()` is asserted in the CI smoke test on both platforms.
- With Indigo absent, the mapping menu greys out with an explanation.

**Order.** WP-A and WP-B run in parallel. WP-C starts with the shortcuts and file formats (independent of the others)
and lands the `install()` calls last. Following the release memory: push before spawning worktree agents, and use
the `qapp` fixture in every new test.

---

## 5. Verification sources to cite in the implementation

- **ChemDraw behaviour:** 17.1 User Guide pages as cited above, e.g. hotkeys pp. 54–63, style sheets pp. 48–53,
  stereo pp. 193–196, properties p. 198, TLC/Gel pp. 212–216, ChemNMR pp. 217–224, formats pp. 320–322. The v23
  "Which ChemDraw" chart covers newer items.
- **Stereo:** IUPAC 2013 CIP rules. The RDKit new CIP labeler follows Hanson et al., J. Chem. Inf. Model. 58 (2018)
  1755.
- **Names:** OPSIN round-trip (Lowe et al. 2011). Reference names from the PubChem IUPAC names already in
  `name_db.tsv.xz`.
- **NMR:** nmrshiftdb2 held-out split; HOSE codes per Bremser, Anal. Chim. Acta 103 (1978) 355.
- **pKa:** Mansouri et al., J. Cheminform. 11 (2019) 60, test split; Dimorphite-DL (Ropp et al., J. Cheminform. 11
  (2019) 14).
- **CDX:** github.com/Glysade/chemdraw (BSD-3) and its test data.
