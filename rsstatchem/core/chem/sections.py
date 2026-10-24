"""Output Viewer sections for the Chemistry menu (no Qt).

Each builder returns a :class:`~rsstatchem.core.output.Section` with an SPSS-style log line,
tables whose formulas are rendered with Unicode subscripts/superscripts, and structure
images. Numbers use ordinary chemistry notation (leading zero, scientific notation for very
small/large values) rather than SPSS's ".045" style.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pandas as pd

from rsstatchem.core.chem import formula as fm
from rsstatchem.core.chem import reaction as rx
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import sup

ATOMIC_WEIGHT_NOTE = "Standard atomic weights: IUPAC/CIAAW 2021, abridged values."
_SUPER = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def num(x, decimals: int = 4) -> str:
    """Fixed decimals; scientific (×10ⁿ) for very small or large magnitudes."""
    if x is None:
        return ""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(x) or math.isinf(x):
        return ""
    if x != 0 and (abs(x) < 10 ** -min(decimals, 3) or abs(x) >= 1e7):
        return sci(x)
    return f"{x:.{decimals}f}"


def sci(x: float, sig: int = 4) -> str:
    if x == 0:
        return "0"
    exp = int(math.floor(math.log10(abs(x))))
    mant = x / 10 ** exp
    if round(abs(mant), sig - 1) >= 10:
        mant /= 10
        exp += 1
    return f"{mant:.{sig - 1}f} × 10{str(exp).translate(_SUPER)}"


def sig(x: float, digits: int = 4) -> str:
    """``digits`` significant figures, switching to ×10ⁿ outside 1e-3 … 1e6."""
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return ""
    x = float(x)
    if x == 0:
        return "0"
    if abs(x) < 1e-3 or abs(x) >= 1e6:
        return sci(x, digits)
    decimals = max(0, digits - 1 - int(math.floor(math.log10(abs(x)))))
    return f"{x:.{decimals}f}"


def _charge_text(z: int) -> str:
    if not z:
        return "0"
    return ("+" if z > 0 else "−") + (str(abs(z)) if abs(z) > 1 else "1")


def count_str(n) -> str:
    return str(n.numerator) if n.denominator == 1 else f"{float(n):.4g}"


def uf(text: str) -> str:
    """Formula text with Unicode subscripts/superscripts."""
    return fm.format_formula(text, "unicode")


# --- formula & molar mass ------------------------------------------------------------------

def formula_section(texts: list[str], composition: bool = True, isotopes: bool = False
                    ) -> Section:
    section = Section("Formula and Molar Mass")
    section.log("CHEM FORMULA " + " ".join(f"'{t}'" for t in texts) +
                ("\n  /PRINT COMPOSITION" if composition else "") + ".")
    parsed, errors = [], []
    for t in texts:
        try:
            parsed.append(fm.parse_formula(t))
        except fm.FormulaError as exc:
            errors.append(f"{t}: {exc}")
    if errors:
        section.note("Warnings", "\n".join(errors), style="warning")
    if not parsed:
        return section
    charged = any(f.charge for f in parsed)
    cols = ["", "Hill formula", "Molar mass (g/mol)", "Monoisotopic mass (u)"]
    if charged:
        cols += ["Charge", "m/z"]
    cols += ["DBE"]
    rows, foot, estimated = [], [ATOMIC_WEIGHT_NOTE], set()
    for f in parsed:
        row = [uf(f.text), uf(f.hill()), num(f.molar_mass, 3), num(f.monoisotopic_mass, 5)]
        if charged:
            row += [_charge_text(f.charge), num(f.mz, 5) if f.mz else ""]
        d = f.dbe() if "C" in f.elements else None
        row.append("" if d is None else (f"{d:g}"))
        rows.append(row)
        estimated.update(f.estimated)
    foot.append("Monoisotopic mass: most abundant isotope of each element"
                + ("; ions corrected for the electron mass." if charged else "."))
    foot.append("DBE: double-bond equivalents (rings + π bonds), shown for organic formulas.")
    if estimated:
        foot.append("No stable isotope (mass number of the longest-lived isotope used): "
                    + ", ".join(sorted(estimated)) + ".")
    section.table(Table("Formula Summary", cols, rows, footnotes=foot))
    if composition:
        for f in parsed:
            if f.electron:
                continue
            comp = f.composition()
            rows = [[r.label if not r.label.startswith("[") else uf(r.label), count_str(r.count),
                     num(r.atomic_weight, 4), num(r.mass, 3), num(r.percent, 2)]
                    for r in comp]
            rows.append(["Total", count_str(f.atom_count), "", num(f.molar_mass, 3), "100.00"])
            section.table(Table(f"Elemental Composition: {uf(f.text)}",
                                ["Element", "Atoms", "Atomic weight", "Mass (g/mol)",
                                 "Mass %"], rows))
    if isotopes:
        for f in parsed:
            pattern = isotope_pattern(f)
            if pattern:
                section.table(Table(f"Isotope Pattern: {uf(f.text)}",
                                    ["Nominal mass", "Mass (u)", "Relative abundance (%)"],
                                    [[str(round(m)), num(m, 4), num(a, 2)] for m, a in pattern],
                                    footnotes=["Peaks ≥ 0.1 % of the most intense peak."]))
    return section


def isotope_pattern(f: fm.Formula, threshold: float = 0.1, max_atoms: int = 400) -> list:
    """Isotope distribution grouped by nominal mass (relative %, max = 100)."""
    from rsstatchem.core.chem import elements as el
    if not f.is_integral or f.electron or f.atom_count > max_atoms:
        return []
    peaks = [(0.0, 1.0)]
    for (sym, a), n in f.counts.items():
        if a is not None:
            iso = [(el.atomic_weight(sym, a), 1.0)]
        else:
            iso = [(i.mass, i.abundance / 100) for i in el.element(sym).natural_isotopes]
            if not iso:
                return []
        for _ in range(int(n)):
            new: dict = {}
            for m, p in peaks:
                for im, ip in iso:
                    key = round(m + im)
                    pp = p * ip
                    if key in new:
                        q, mm = new[key]
                        new[key] = (q + pp, mm + pp * (m + im))
                    else:
                        new[key] = (pp, pp * (m + im))
            top = max(v[0] for v in new.values())
            peaks = [(mm / q, q) for q, mm in new.values() if q > top * 1e-6]
    top = max(p for _, p in peaks)
    out = sorted((m - f.charge * el.ELECTRON_MASS, 100 * p / top) for m, p in peaks)
    return [(m, a) for m, a in out if a >= threshold]


def empirical_section(res: fm.EmpiricalResult, source: str = "") -> Section:
    section = Section("Empirical Formula")
    spec = " ".join(f"{r.symbol}={r.percent:.4g}" for r in res.rows)
    section.log(f"CHEM EMPIRICAL {spec}" + (f" /MOLARMASS={res.molar_mass:g}"
                                             if res.molar_mass else "") + ".")
    if source:
        section.note("Data", source)
    rows = []
    for r in res.rows:
        label = r.symbol + (sup("a") if r.symbol == res.by_difference else "")
        rows.append([label, num(r.percent, 3), num(r.atomic_weight, 4), num(r.moles, 4),
                     num(r.ratio, 3), num(r.scaled, 3), str(r.subscript)])
    foot = [f"Mole ratios multiplied by {res.multiplier} (largest deviation from a whole "
            f"number {res.max_deviation:.3f}).", ATOMIC_WEIGHT_NOTE]
    if res.by_difference:
        foot.insert(0, f"{sup('a')}Taken by difference from 100 %.")
    section.table(Table("Mole Ratio", ["Element", "Mass %", "Atomic weight",
                                       "mol per 100 g", "Ratio", f"Ratio × {res.multiplier}",
                                       "Subscript"], rows, footnotes=foot))
    out = [["Empirical formula", uf(res.empirical.hill()), num(res.empirical_mass, 3)]]
    if res.molecular is not None:
        out.append(["Molar mass given", "", num(res.molar_mass, 3)])
        out.append(["n = M / M(empirical)", f"{res.n:.3f}", ""])
        out.append(["Molecular formula", uf(res.molecular.hill()),
                    num(res.molecular.molar_mass, 3)])
    section.table(Table("Formula", ["", "Formula", "Molar mass (g/mol)"], out))
    check = res.molecular or res.empirical
    pct = check.mass_percent()
    rows = [[r.symbol, num(r.percent, 2), num(pct.get(r.symbol, 0), 2),
             num(pct.get(r.symbol, 0) - r.percent, 2)] for r in res.rows]
    section.table(Table(f"Composition Check: {uf(check.hill())}",
                        ["Element", "Given %", "Calculated %", "Difference"], rows))
    for w in res.warnings:
        section.note("", w, style="warning")
    return section


# --- molecules -----------------------------------------------------------------------------

_PROP_ROWS = [("Formula", "Molecular formula", None), ("MW", "Molecular weight (g/mol)", 3),
              ("ExactMass", "Exact mass (u)", 5), ("cLogP", "cLogP (Crippen)", 3),
              ("MR", "Molar refractivity", 3), ("TPSA", "TPSA (Å²)", 2),
              ("HBD", "H-bond donors", 0), ("HBA", "H-bond acceptors", 0),
              ("RotB", "Rotatable bonds", 0), ("Rings", "Rings", 0),
              ("AroRings", "Aromatic rings", 0), ("HeavyAtoms", "Heavy atoms", 0),
              ("Atoms", "Atoms (incl. H)", 0), ("Heteroatoms", "Heteroatoms", 0),
              ("FCsp3", "Fraction Csp3", 3), ("Charge", "Formal charge", 0),
              ("Stereo", "Stereocentres", 0), ("QED", "QED", 3)]


def _pval(key: str, v, dec) -> str:
    if key == "Formula":
        return uf(v)
    if dec == 0:
        return str(int(v)) if v == v else ""
    return num(v, dec)


def molecule_section(items: list, identifiers: bool = True, descriptors: bool = True,
                     rules: list[str] | None = None, image: bool = True,
                     size: tuple[int, int] = (320, 240), draw_options=None) -> Section:
    """``items``: list of (name, RDKit Mol)."""
    from rsstatchem.core.chem import depict, molecule as mo
    section = Section("Molecule Properties")
    section.log("CHEM MOLECULE " + " ".join(f"'{mo.to_smiles(m)}'" for _, m in items[:20]) +
                ("\n  /RULES " + " ".join(rules) if rules else "") + ".")
    if not items:
        section.note("", "No valid molecule.", style="warning")
        return section
    names = [n or f"Molecule {i + 1}" for i, (n, _) in enumerate(items)]
    mols = [m for _, m in items]
    opts = draw_options or depict.DrawOptions(width=size[0], height=size[1])
    if image:
        if len(mols) == 1:
            section.image(names[0], depict.depict_png(mols[0], options=opts), width=size[0])
        else:
            cols = min(4, len(mols))
            cell = depict.DrawOptions(**{**opts.__dict__, "width": 220, "height": 180})
            section.image("Structures", depict.depict_grid_png(mols[:48], names[:48], cols,
                                                              cell), width=220 * cols)
    props = [mo.properties(m) for m in mols]
    if identifiers:
        if len(mols) == 1:
            m = mols[0]
            rows = [["Name", names[0]], ["Canonical SMILES", mo.to_smiles(m)],
                    ["Molecular formula", uf(props[0]["Formula"])],
                    ["InChI", mo.to_inchi(m)], ["InChIKey", mo.to_inchikey(m)]]
            section.table(Table("Identifiers", ["", "Value"], rows))
        else:
            rows = [[n, mo.to_smiles(m), uf(p["Formula"]), mo.to_inchikey(m)]
                    for n, m, p in zip(names, mols, props)]
            section.table(Table("Identifiers", ["", "Canonical SMILES", "Formula", "InChIKey"],
                                rows))
    if descriptors:
        if len(mols) == 1:
            rows = [[label, _pval(k, props[0][k], d)] for k, label, d in _PROP_ROWS
                    if k in props[0]]
            section.table(Table("Molecular Properties", ["Property", "Value"], rows,
                                footnotes=["Computed with RDKit; cLogP and MR by the Crippen "
                                           "method (Wildman & Crippen 1999), TPSA by Ertl "
                                           "et al. (2000)."]))
        else:
            dec = {k: d for k, _, d in _PROP_ROWS}
            if len(mols) <= 6:
                rows = [[label] + [_pval(k, p[k], d) for p in props]
                        for k, label, d in _PROP_ROWS if k in props[0]]
                section.table(Table("Molecular Properties", ["Property"] + names, rows))
            else:
                keys = ["MW", "cLogP", "MR", "TPSA", "HBD", "HBA", "RotB", "Rings", "AroRings",
                        "HeavyAtoms", "FCsp3", "QED"]
                short = {"AroRings": "Arom. rings", "HeavyAtoms": "Heavy atoms", "FCsp3": "Fsp3"}
                rows = [[n, uf(p["Formula"])] + [_pval(k, p[k], dec[k]) for k in keys]
                        for n, p in zip(names, props)]
                section.table(Table("Molecular Properties", ["", "Formula"] +
                                    [short.get(k, k) for k in keys], rows))
    if rules:
        results = [mo.drug_likeness(m, rules) for m in mols]
        if len(mols) == 1:
            for r in results[0]:
                rows = [[c[0], num(c[1], 2) if isinstance(c[1], float) else str(c[1]), c[2],
                         "Pass" if c[3] else "Fail"] for c in r.criteria]
                rows.append(["Result", f"{r.violations} violation(s)",
                             f"≤ {r.allowed} allowed", "Pass" if r.passed else "Fail"])
                section.table(Table(f"Drug-likeness: {r.rule}", ["Criterion", "Value",
                                                                 "Limit", "Result"], rows,
                                    footnotes=[r.reference]))
        else:
            rows = [[n] + [f"{'Pass' if r.passed else 'Fail'} ({r.violations})" for r in res]
                    for n, res in zip(names, results)]
            section.table(Table("Drug-likeness", [""] + rules, rows,
                                footnotes=["Number of violated criteria in parentheses. "
                                           "Lipinski allows one violation."]))
    return section


def structure_section(text: str, options=None, title: str = "") -> Section:
    """Image-only section for SMILES / InChI / MOL text or reaction SMILES."""
    from rsstatchem.core.chem import depict, molecule as mo
    opts = options or depict.DrawOptions()
    section = Section("Structure")
    first = text.strip().splitlines()[0] if text.strip() else ""
    section.log(f"CHEM DRAW '{first if len(first) < 200 else first[:200] + '…'}'"
                f" /SIZE={opts.width}x{opts.height}.")
    if mo.detect_format(text) == "reaction":
        png = depict.depict_reaction_png(text, opts.width, opts.height, opts)
    else:
        png = depict.depict_png(mo.mol_from_text(text), options=opts)
    section.image(title or opts.legend or "Structure", png, width=opts.width)
    return section


# --- equations -----------------------------------------------------------------------------

def balance_section(res: rx.BalanceResult, stoich: rx.StoichResult | None = None,
                    text: str = "") -> Section:
    section = Section("Balance Equation")
    section.log(f"CHEM BALANCE '{text or res.equation('plain', [None] * len(res.species))}'"
                + ("\n  /STOICHIOMETRY" if stoich else "") + ".")
    style = "note" if res.status == "balanced" else "warning"
    if res.coefficients is not None:
        section.table(Table("Balanced Equation", ["", "Equation"],
                            [["Balanced", res.equation("unicode")],
                             ["Coefficients", " : ".join(str(c) for c in res.coefficients)]]))
    section.note("", res.message, style=style)
    if res.given_balanced is not None:
        section.note("", "The coefficients you entered are " +
                     ("balanced." if res.given_balanced else "NOT balanced."),
                     style="note" if res.given_balanced else "warning")
    if res.status == "not_unique" and res.basis:
        rows = [[f"Reaction {i + 1}", _basis_equation(res, v)] for i, v in enumerate(res.basis)]
        section.table(Table("Independent Reactions", ["", "Equation"], rows,
                            footnotes=["Any positive combination of these reactions is "
                                       "balanced."]))
    if res.coefficients is not None:
        rows = [[uf(lbl) if lbl != "Charge" else lbl, count_str(a), count_str(b),
                 "Yes" if a == b else "No"] for lbl, a, b in res.balance_table()]
        section.table(Table("Atom Balance", ["Element", "Reactants", "Products", "Balanced"],
                            rows))
    if stoich is not None:
        section.items.extend(stoich_section(res, stoich).items)
    return section


def _basis_equation(res: rx.BalanceResult, v: list[int]) -> str:
    left = [f"{abs(c) if abs(c) != 1 else ''}{' ' if abs(c) != 1 else ''}"
            f"{uf(s.text)}" for s, c in zip(res.species, v) if c * s.side > 0]
    right = [f"{abs(c) if abs(c) != 1 else ''}{' ' if abs(c) != 1 else ''}"
             f"{uf(s.text)}" for s, c in zip(res.species, v) if c * s.side < 0]
    return " + ".join(left) + " → " + " + ".join(right)


def stoich_section(res: rx.BalanceResult, st: rx.StoichResult) -> Section:
    section = Section("Stoichiometry")
    has_actual = any(r.actual_g is not None for r in st.rows)
    cols = ["Species", "Role", "Coefficient", "M (g/mol)", "Given (mol)",
            ("Reacting / formed", "mol"), ("Reacting / formed", "g"),
            ("Excess remaining", "mol"), ("Excess remaining", "g")]
    if has_actual:
        cols += ["Actual yield (g)", "Yield (%)"]
    rows = []
    for r in st.rows:
        name = uf(r.species) + (sup("a") if r.limiting else "")
        row = [name, r.role, str(r.coefficient), num(r.molar_mass, 3),
               sig(r.given_mol) if r.given_mol is not None else "",
               sig(r.reacting_mol), sig(r.reacting_g),
               sig(r.remaining_mol) if r.remaining_mol is not None else "",
               sig(r.remaining_g) if r.remaining_g is not None else ""]
        if has_actual:
            row += [sig(r.actual_g) if r.actual_g is not None else "",
                    num(r.percent_yield, 2) if r.percent_yield is not None else ""]
        rows.append(row)
    foot = []
    if st.limiting:
        foot.append(f"{sup('a')}Limiting reagent. Extent of reaction ξ = {sig(st.extent)} mol.")
    else:
        foot.append(f"Extent of reaction ξ = {sig(st.extent)} mol.")
    foot += st.notes
    section.table(Table("Stoichiometry", cols, rows, footnotes=foot))
    if st.limiting:
        names = ", ".join(uf(res.species[i].text) for i in st.limiting)
        prods = "; ".join(f"{uf(r.species)} {sig(r.reacting_g)} g ({sig(r.reacting_mol)} mol)"
                          for r in st.rows if r.role == "Product")
        section.note("", f"Limiting reagent: {names}. Theoretical yield: {prods}.")
    return section


# --- generic calculator --------------------------------------------------------------------

def calculator_section(title: str, log: str, inputs: list, results: list,
                       notes: list[str] | None = None, extra: list[Table] | None = None,
                       result_title: str = "Result") -> Section:
    """Inputs/results as two-column tables: lists of (label, value text[, unit])."""
    section = Section(title)
    section.log(log)

    def rows(pairs):
        return [[p[0], p[1], p[2] if len(p) > 2 else ""] for p in pairs]
    if inputs:
        section.table(Table("Input", ["", "Value", "Unit"], rows(inputs)))
    if results:
        section.table(Table(result_title, ["", "Value", "Unit"], rows(results)))
    for t in extra or []:
        section.table(t)
    for n in notes or []:
        section.note("", n)
    return section


# --- dataset integration -------------------------------------------------------------------

FORMULA_DESCRIPTORS = [("FW", "Molar mass from formula (g/mol)", 3),
                       ("MonoMass", "Monoisotopic mass from formula (u)", 5),
                       ("DBE", "Double-bond equivalents", 1)]


def descriptors_from_dataset(ds, smiles_var: str | None, formula_var: str | None,
                             keys: list[str], formula_keys: list[str] | None = None,
                             strip_salts: bool = False):
    """Compute descriptors for every case; returns ``.section`` and ``.saved``.

    Invalid or empty SMILES/formulas give system-missing values and are counted."""
    from rsstatchem.core.chem import molecule as mo
    section = Section("Molecular Descriptors")
    saved: dict = {}
    index = ds.df.index
    parts = []
    summary_rows = []
    warn = []
    if smiles_var:
        parts.append(f"SMILES={smiles_var}")
        col = ds.analysis_series(smiles_var)
        values = {k: np.full(len(index), np.nan) for k in keys}
        invalid, empty = [], 0
        cache: dict = {}
        for pos, raw in enumerate(col.to_numpy()):
            if raw is None or (isinstance(raw, float) and math.isnan(raw)) or not str(raw).strip():
                empty += 1
                continue
            text = str(raw).strip()
            if text not in cache:
                try:
                    m = mo.mol_from_text(text, "smiles")
                    if strip_salts:
                        m = mo.standardize(m)
                    cache[text] = mo.properties(m, keys)
                except Exception:  # noqa: BLE001
                    cache[text] = None
            props = cache[text]
            if props is None:
                invalid.append(pos + 1)
                continue
            for k in keys:
                values[k][pos] = props[k]
        for k in keys:
            d = mo.descriptor(k)
            saved[k] = (d.label, pd.Series(values[k], index=index), d.decimals)
            summary_rows.append(_summary_row(d.label, values[k]))
        if invalid:
            shown = ", ".join(str(i) for i in invalid[:15]) + (" …" if len(invalid) > 15 else "")
            warn.append(f"{len(invalid)} case(s) with invalid SMILES in {smiles_var} were set "
                        f"to system-missing (cases {shown}).")
        if empty:
            warn.append(f"{empty} case(s) with an empty {smiles_var} value are missing.")
    if formula_var and formula_keys:
        parts.append(f"FORMULA={formula_var}")
        col = ds.analysis_series(formula_var)
        values = {k: np.full(len(index), np.nan) for k in formula_keys}
        invalid = []
        for pos, raw in enumerate(col.to_numpy()):
            if raw is None or (isinstance(raw, float) and math.isnan(raw)) or not str(raw).strip():
                continue
            try:
                f = fm.parse_formula(str(raw))
            except fm.FormulaError:
                invalid.append(pos + 1)
                continue
            for k in formula_keys:
                if k == "FW":
                    values[k][pos] = f.molar_mass
                elif k == "MonoMass":
                    values[k][pos] = f.monoisotopic_mass
                elif k == "DBE":
                    d = f.dbe()
                    values[k][pos] = np.nan if d is None else d
        labels = {k: (lbl, dec) for k, lbl, dec in FORMULA_DESCRIPTORS}
        for k in formula_keys:
            saved[k] = (labels[k][0], pd.Series(values[k], index=index), labels[k][1])
            summary_rows.append(_summary_row(labels[k][0], values[k]))
        if invalid:
            shown = ", ".join(str(i) for i in invalid[:15]) + (" …" if len(invalid) > 15 else "")
            warn.append(f"{len(invalid)} case(s) with invalid formulas in {formula_var} were set "
                        f"to system-missing (cases {shown}).")
    section.log("CHEM DESCRIPTORS " + " ".join(parts) +
                f"\n  /SAVE {' '.join(keys + (formula_keys or []))}" +
                ("\n  /STRIPSALTS" if strip_salts else "") + ".")
    section.table(Table("Descriptive Statistics", ["", "N", "Missing", "Minimum", "Maximum",
                                                   "Mean", "Std. Deviation"], summary_rows))
    for w in warn:
        section.note("", w, style="warning")
    return SimpleNamespace(section=section, saved=saved)


def _summary_row(label: str, v: np.ndarray) -> list[str]:
    ok = v[~np.isnan(v)]
    if not len(ok):
        return [label, "0", str(len(v)), "", "", "", ""]
    sd = float(np.std(ok, ddof=1)) if len(ok) > 1 else math.nan
    return [label, str(len(ok)), str(len(v) - len(ok)), num(ok.min(), 3), num(ok.max(), 3),
            num(ok.mean(), 3), num(sd, 3)]
