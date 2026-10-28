# Template and clipware sources

Chemical Drawing templates and clipware in RS-StatChem.

| Item | Where | Source | Licence |
|---|---|---|---|
| Laboratory glassware (beaker, Erlenmeyer flask, round-bottom flask, test tube, graduated cylinder, funnel, separatory funnel, Liebig condenser, burette, thermometer, Petri dish, vial) | `rsstatchem/core/chem_draw/clipart.py` | Original vector outlines drawn for RS-StatChem from simple geometric primitives (lines, arcs, ellipses). No third-party artwork was copied or traced. | Same licence as RS-StatChem; the outlines may also be reused under CC0 1.0 |
| Newman, Fischer, Haworth, chair (axial / equatorial) and boat templates | `rsstatchem/core/chem_draw/templates.py` | Built from standard textbook geometry by RS-StatChem code | Same as above |
| SMILES templates (amino acids, sugars, aromatics, bicyclics and cages, functional and protecting groups) | `rsstatchem/core/chem_draw/templates.py` | Structures are facts. SMILES written for RS-StatChem | Same as above |

ChemDraw's own clipware and template files are copyrighted and are **not** included. The
style sheet values in `rsstatchem/core/chem_draw/style.py` are factual settings taken from the
ChemDraw 17.1 User Guide ("Document Settings", pp. 48-53).

To add third-party art later, accept only CC0, CC BY or MIT material (for example Servier
Medical Art, CC BY 4.0, or Bioicons icons with a CC0 / CC BY / MIT licence). Add a row to this
table with the author, the source URL and the licence, and credit CC BY works in the About box.
