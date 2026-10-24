"""Chemistry > Solution Calculator: tabbed lab calculators with live results.

Calculators: Prepare Solution, Stock Reagent, Dilution, Serial Dilution, Concentration Units, pH,
Buffer, Molality && Normality, Gas Law, Unit Conversion. OK sends the shown calculation
to the Output Viewer.
"""
from __future__ import annotations

import math

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
                               QListWidget, QScrollArea, QStackedWidget, QVBoxLayout,
                               QWidget)

from rsstatchem.core.chem import solutions as sol
from rsstatchem.core.chem.sections import calculator_section, sig
from rsstatchem.core.output import Table
from rsstatchem.ui.chem.widgets import (CalculatorDialog, FormulaField, combo, line,
                                        parse_number)

CONC_PREP_UNITS = {"M (mol/L)": ("M", 1.0), "mM": ("M", 1e-3), "µM": ("M", 1e-6),
                   "g/L": ("gL", 1.0), "mg/L (ppm)": ("gL", 1e-3), "µg/L (ppb)": ("gL", 1e-6),
                   "% (w/v)": ("gL", 10.0)}
VOL_UNITS = {"L": 1.0, "mL": 1e-3, "µL": 1e-6}
MASS_UNITS = {"g": 1.0, "mg": 1e-3, "kg": 1e3}


class Page(QWidget):
    """One calculator tab: fields in a form, a live result label, and ``calculate()``
    returning (inputs, results, notes, extra tables, log)."""

    title = ""

    def __init__(self, dialog: CalculatorDialog):
        super().__init__()
        self.dialog = dialog
        self.form = QFormLayout()
        self.form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.result = QLabel()
        self.result.setTextFormat(Qt.TextFormat.RichText)
        self.result.setWordWrap(True)
        self.result.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.result.setMinimumHeight(110)
        self.result.setStyleSheet("QLabel { padding: 8px; background: #fafafa; "
                                  "border: 1px solid #ddd; }")
        self._rows: dict = {}
        lay = QVBoxLayout(self)
        lay.addLayout(self.form)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(self.result)
        lay.addWidget(scroll, 1)
        self.help = QLabel()
        self.help.setWordWrap(True)
        self.help.setStyleSheet("color: #666;")
        lay.addWidget(self.help)

    def key(self, name: str) -> str:
        return f"{type(self).__name__}.{name}"

    def num(self, label: str, name: str, default: str = "", units: list | None = None,
            unit_default: str | None = None, placeholder: str = ""):
        edit = line(default, placeholder)
        edit.setMinimumWidth(140)
        self.dialog.remember(self.key(name), edit)
        edit.textChanged.connect(self.dialog.schedule)
        if units:
            box = combo(units, unit_default, width=110)
            self.dialog.remember(self.key(name + "_unit"), box)
            box.currentIndexChanged.connect(self.dialog.schedule)
            row = QHBoxLayout()
            row.addWidget(edit, 1)
            row.addWidget(box)
            self.form.addRow(label, row)
            self._rows[name] = row
            return edit, box
        self.form.addRow(label, edit)
        self._rows[name] = edit
        return edit

    def formula(self, label: str, name: str, default: str = "") -> FormulaField:
        f = FormulaField(default, "formula (e.g. NaCl) or molar mass in g/mol")
        self.dialog.remember(self.key(name), f.edit)
        f.changed.connect(self.dialog.schedule)
        self.form.addRow(label, f)
        self._rows[name] = f
        return f

    def choice(self, label: str, name: str, items: list, default: str | None = None):
        box = combo(items, default)
        self.dialog.remember(self.key(name), box)
        box.currentIndexChanged.connect(self.dialog.schedule)
        self.form.addRow(label, box)
        return box

    def check(self, text: str, name: str, on: bool = False) -> QCheckBox:
        cb = QCheckBox(text)
        cb.setChecked(on)
        cb.setProperty("default_value", on)
        self.dialog.remember(self.key(name), cb)
        cb.toggled.connect(self.dialog.schedule)
        self.form.addRow("", cb)
        return cb

    def show_rows(self, names: list[str], visible: bool) -> None:
        for n in names:
            self.form.setRowVisible(self._rows[n], visible)

    @staticmethod
    def v(edit: QLineEdit, required: str | None = None) -> float | None:
        x = parse_number(edit.text())
        if x is None and required:
            raise ValueError(f"Enter the {required}.")
        return x

    def calculate(self):
        raise NotImplementedError

    def refresh(self) -> None:
        try:
            inputs, results, notes, extra, _log = self.calculate()
        except (ValueError, KeyError, ZeroDivisionError, OverflowError) as exc:
            self.result.setText(f"<span style='color:#b00020'>{str(exc).strip(chr(39))}</span>")
            return
        rows = "".join(f"<tr><td>{r[0]}</td><td align='right'><b>{r[1]}</b></td>"
                       f"<td>&nbsp;{r[2] if len(r) > 2 else ''}</td></tr>" for r in results)
        html = f"<table cellspacing='4'>{rows}</table>"
        if extra:
            t = extra[0]
            head = "".join(f"<th align='left'>{c}&nbsp;</th>" for c in t.columns)
            body = "".join("<tr>" + "".join(f"<td>{c}&nbsp;&nbsp;</td>" for c in r) + "</tr>"
                           for r in t.rows[:20])
            html += (f"<b>{t.title}</b><table cellspacing='2'><tr>{head}</tr>{body}</table>")
        if notes:
            html += "<br>" + "<br>".join(f"<span style='color:#555'>{n}</span>" for n in notes)
        self.result.setText(html)

    def section(self):
        inputs, results, notes, extra, log = self.calculate()
        return calculator_section(self.title, log, inputs, results, notes, extra)


def _f(x, d=4):
    return sig(x, d)


class PreparePage(Page):
    title = "Prepare Solution"

    def __init__(self, dialog):
        super().__init__(dialog)
        self.mode = self.choice("Calculate:", "mode", ["Mass to weigh", "Concentration of a "
                                                       "solution"])
        self.solute = self.formula("Solute:", "solute", "NaCl")
        self.conc, self.conc_u = self.num("Concentration:", "conc", "0.1",
                                          list(CONC_PREP_UNITS), "M (mol/L)")
        self.mass, self.mass_u = self.num("Mass dissolved:", "mass", "", list(MASS_UNITS), "g")
        self.vol, self.vol_u = self.num("Final volume:", "vol", "250", list(VOL_UNITS), "mL")
        self.purity = self.num("Purity (%):", "purity", "100")
        self.mode.currentIndexChanged.connect(self._mode)
        self._mode()
        self.help.setText("Hydrates: enter the hydrate formula (e.g. CuSO4·5H2O) to weigh the "
                          "hydrate. Purity corrects the mass for the assay of the reagent.")

    def _mode(self) -> None:
        weigh = self.mode.currentIndex() == 0
        self.show_rows(["conc"], weigh)
        self.show_rows(["mass"], not weigh)

    def calculate(self):
        mm = self.solute.molar_mass()
        if mm is None:
            raise ValueError("Enter the solute formula or molar mass.")
        V = self.v(self.vol, "final volume") * VOL_UNITS[self.vol_u.currentText()]
        purity = self.v(self.purity) or 100.0
        inputs = [("Solute", self.solute.display()), ("Molar mass", _f(mm, 6), "g/mol"),
                  ("Final volume", _f(V * 1000), "mL"), ("Purity", _f(purity), "%")]
        if self.mode.currentIndex() == 0:
            c = self.v(self.conc, "concentration")
            kind, factor = CONC_PREP_UNITS[self.conc_u.currentText()]
            molar = c * factor if kind == "M" else c * factor / mm
            mass = sol.mass_for_solution(molar, V, mm, purity)
            inputs.insert(2, ("Target concentration", _f(c), self.conc_u.currentText()))
            results = [("Mass to weigh", _f(mass), "g"), ("", _f(mass * 1000), "mg"),
                       ("Amount of solute", _f(molar * V), "mol"),
                       ("Molarity", _f(molar), "mol/L"), ("Mass concentration", _f(molar * mm),
                                                         "g/L")]
            notes = [f"Dissolve {_f(mass)} g in less solvent, then make up to {_f(V * 1000)} mL "
                     "in a volumetric flask."]
            log = (f"CHEM SOLUTION PREPARE SOLUTE='{self.solute.text()}' CONC={c:g} "
                   f"VOLUME={V * 1000:g}mL PURITY={purity:g}.")
        else:
            m = self.v(self.mass, "mass dissolved") * MASS_UNITS[self.mass_u.currentText()]
            pure = m * purity / 100
            n = pure / mm
            inputs.insert(2, ("Mass dissolved", _f(m), "g"))
            results = [("Molarity", _f(n / V), "mol/L"), ("", _f(n / V * 1000), "mmol/L"),
                       ("Amount of solute", _f(n), "mol"),
                       ("Mass concentration", _f(pure / V), "g/L"),
                       ("", _f(pure / V * 1000), "mg/L (≈ ppm)"),
                       ("% (w/v)", _f(pure / V / 10), "%")]
            notes = []
            log = (f"CHEM SOLUTION CONCENTRATION SOLUTE='{self.solute.text()}' MASS={m:g}g "
                   f"VOLUME={V * 1000:g}mL.")
        return inputs, results, notes, None, log


class StockPage(Page):
    title = "Stock Reagent Dilution"

    def __init__(self, dialog):
        super().__init__(dialog)
        self.solute = self.formula("Reagent:", "solute", "HCl")
        self.pct = self.num("Assay (% w/w):", "pct", "37")
        self.rho = self.num("Density (g/mL):", "rho", "1.19")
        self.target = self.num("Target molarity (M):", "target", "1")
        self.vol, self.vol_u = self.num("Target volume:", "vol", "500", list(VOL_UNITS), "mL")
        self.help.setText("Concentrated reagents: HCl 37 % (1.19 g/mL), HNO₃ 65 % (1.39), "
                          "H₂SO₄ 98 % (1.84), CH₃COOH 100 % (1.05), NH₃ 25 % (0.91), "
                          "H₃PO₄ 85 % (1.69). Always add acid to water.")

    def calculate(self):
        mm = self.solute.molar_mass()
        if mm is None:
            raise ValueError("Enter the reagent formula or molar mass.")
        pct, rho = self.v(self.pct, "assay"), self.v(self.rho, "density")
        stock = sol.stock_molarity(pct, rho, mm)
        inputs = [("Reagent", self.solute.display()), ("Molar mass", _f(mm, 6), "g/mol"),
                  ("Assay", _f(pct), "% (w/w)"), ("Density", _f(rho), "g/mL")]
        results = [("Stock molarity", _f(stock), "mol/L")]
        notes = ["c(stock) = 10 · % · ρ / M"]
        target = self.v(self.target)
        vol = self.v(self.vol)
        if target and vol:
            V = vol * VOL_UNITS[self.vol_u.currentText()]
            vs = sol.stock_volume(target, V, pct, rho, mm)
            inputs += [("Target molarity", _f(target), "mol/L"), ("Target volume", _f(V * 1000),
                                                                  "mL")]
            results += [("Stock volume to take", _f(vs * 1000), "mL"),
                        ("Water to add (approx.)", _f((V - vs) * 1000), "mL")]
            notes.append(f"Take {_f(vs * 1000)} mL of the stock and dilute to {_f(V * 1000)} mL.")
        log = (f"CHEM SOLUTION STOCK REAGENT='{self.solute.text()}' ASSAY={pct:g} "
               f"DENSITY={rho:g}.")
        return inputs, results, notes, None, log


class DilutionPage(Page):
    title = "Dilution"

    def __init__(self, dialog):
        super().__init__(dialog)
        units = ["M", "mM", "µM", "mg/L", "µg/mL", "%", "ppm", "(any)"]
        self.unit = self.choice("Concentration unit:", "unit", units, "M")
        self.vunit = self.choice("Volume unit:", "vunit", ["mL", "L", "µL"], "mL")
        self.c1 = self.num("C₁ (stock):", "c1", "1")
        self.v1 = self.num("V₁ (stock volume):", "v1", "")
        self.c2 = self.num("C₂ (final):", "c2", "0.1")
        self.v2 = self.num("V₂ (final volume):", "v2", "100")
        self.help.setText("C₁V₁ = C₂V₂ — leave exactly one field empty.")

    def calculate(self):
        vals = sol.dilution(self.v(self.c1), self.v(self.v1), self.v(self.c2), self.v(self.v2))
        cu, vu = self.unit.currentText(), self.vunit.currentText()
        cu = "" if cu == "(any)" else cu
        inputs = [(k, _f(vals[n]), u) for k, n, u in (("C₁", "c1", cu), ("V₁", "v1", vu),
                                                       ("C₂", "c2", cu), ("V₂", "v2", vu))]
        results = [("V₁ (stock to take)", _f(vals["v1"]), vu), ("Solvent to add",
                                                                _f(vals["solvent"]), vu),
                   ("C₂", _f(vals["c2"]), cu), ("Dilution factor", _f(vals["factor"]), "×")]
        if self.v(self.c1) and self.v(self.c2) and self.v(self.v2) and self.v(self.v1) is None:
            inputs = [r for r in inputs if r[0] != "V₁"]
        log = "CHEM SOLUTION DILUTION " + " ".join(f"{k.upper()}={vals[k]:g}"
                                                   for k in ("c1", "v1", "c2", "v2")) + "."
        return inputs, results, ["C₁V₁ = C₂V₂"], None, log


class SerialPage(Page):
    title = "Serial Dilution"

    def __init__(self, dialog):
        super().__init__(dialog)
        self.c0 = self.num("Stock concentration:", "c0", "1000")
        self.unit = self.choice("Concentration unit:", "unit",
                                ["mg/L", "µg/mL", "M", "mM", "µM", "ppm", "%", "CFU/mL"], "mg/L")
        self.factor = self.num("Dilution factor per step:", "factor", "10")
        self.steps = self.num("Number of steps:", "steps", "5")
        self.vol = self.num("Final volume per tube (mL):", "vol", "10")

    def calculate(self):
        c0, f = self.v(self.c0, "stock concentration"), self.v(self.factor, "dilution factor")
        n = int(self.v(self.steps, "number of steps"))
        vol = self.v(self.vol, "final volume")
        steps = sol.serial_dilution(c0, f, n, vol)
        u = self.unit.currentText()
        rows = [[str(s.step), _f(s.concentration), _f(s.transfer_volume), _f(s.diluent_volume),
                 f"1:{s.cumulative_factor:g}"] for s in steps]
        table = Table("Serial Dilution Scheme", ["Tube", f"Concentration ({u})",
                                                 "Transfer from previous (mL)",
                                                 "Diluent (mL)", "Cumulative dilution"], rows,
                      footnotes=["Tube 1 receives the transfer from the stock solution. The "
                                 "transfer volume is removed from each tube for the next one, "
                                 "except from the last."])
        inputs = [("Stock concentration", _f(c0), u), ("Factor per step", _f(f), "×"),
                  ("Steps", str(n)), ("Final volume per tube", _f(vol), "mL")]
        results = [("Transfer volume", _f(vol / f), "mL"), ("Diluent per tube", _f(vol - vol / f),
                                                           "mL"),
                   ("Lowest concentration", _f(steps[-1].concentration), u)]
        log = f"CHEM SOLUTION SERIAL C0={c0:g} FACTOR={f:g} STEPS={n} VOLUME={vol:g}mL."
        return inputs, results, [], [table], log


class ConcUnitsPage(Page):
    title = "Concentration Units"

    def __init__(self, dialog):
        super().__init__(dialog)
        units = list(sol.CONC_UNITS)
        self.value = self.num("Value:", "value", "1")
        self.src = self.choice("From:", "src", units, "M (mol/L)")
        self.dst = self.choice("To:", "dst", units, "mg/L")
        self.solute = self.formula("Solute (for molar units):", "solute", "NaCl")
        self.rho = self.num("Solution density (g/mL):", "rho", "1.0")
        self.rho_solute = self.num("Solute density (g/mL, for % v/v):", "rho_solute", "")
        self.help.setText("For dilute aqueous solutions (density 1.0 g/mL) ppm (w/w) ≈ mg/L "
                          "and ppb ≈ µg/L.")

    def calculate(self):
        x = self.v(self.value, "value")
        try:
            mm = self.solute.molar_mass()
        except ValueError:
            mm = None
        rho = self.v(self.rho) or 1.0
        rs = self.v(self.rho_solute)
        src, dst = self.src.currentText(), self.dst.currentText()
        y = sol.convert_concentration(x, src, dst, mm, rho, rs)
        rows = []
        for u in sol.CONC_UNITS:
            try:
                rows.append([u, _f(sol.convert_concentration(x, src, u, mm, rho, rs))])
            except ValueError:
                continue
        table = Table("Equivalent Concentrations", ["Unit", "Value"], rows)
        inputs = [("Value", _f(x), src), ("Solute", self.solute.display() if mm else "–"),
                  ("Solution density", _f(rho), "g/mL")]
        if rs:
            inputs.append(("Solute density", _f(rs), "g/mL"))
        log = f"CHEM SOLUTION CONVERT {x:g} '{src}' TO '{dst}' DENSITY={rho:g}."
        return inputs, [("Converted value", _f(y, 5), dst)], [], [table], log


class PHPage(Page):
    title = "pH Calculation"

    def __init__(self, dialog):
        super().__init__(dialog)
        self.kind = self.choice("Solution:", "kind", ["Weak acid", "Weak base", "Strong acid",
                                                      "Strong base"], "Weak acid")
        self.conc = self.num("Concentration (M):", "conc", "0.1")
        self.k, self.k_type = self.num("Ka / Kb or pKa / pKb:", "k", "1.8e-5",
                                       ["Ka (Kb)", "pKa (pKb)"], "Ka (Kb)")
        self.n = self.num("H⁺ / OH⁻ per formula (strong):", "n", "1")
        self.kw = self.num("Kw:", "kw", "1.0e-14")
        self.water = self.check("Include water autoionisation (exact)", "water", False)
        self.help.setText("Weak acids/bases use the exact quadratic x = (−K + √(K² + 4KC))/2; "
                          "tick the box for very dilute solutions (full charge balance).")

    def calculate(self):
        kind = self.kind.currentText()
        c = self.v(self.conc, "concentration")
        kw = self.v(self.kw) or sol.KW_25C
        inputs = [("Solution", kind), ("Concentration", _f(c), "mol/L"), ("Kw", _f(kw))]
        if kind.startswith("Weak"):
            k = self.v(self.k, "dissociation constant")
            if self.k_type.currentIndex() == 1:
                k = 10 ** (-k)
            fn = sol.ph_weak_acid if kind == "Weak acid" else sol.ph_weak_base
            r = fn(c, k, kw, self.water.isChecked())
            label = "Ka" if kind == "Weak acid" else "Kb"
            inputs += [(label, _f(k)), (f"p{label}", f"{-math.log10(k):.3f}")]
        else:
            n = int(self.v(self.n) or 1)
            fn = sol.ph_strong_acid if kind == "Strong acid" else sol.ph_strong_base
            r = fn(c, n, kw, self.water.isChecked())
            inputs.append(("H⁺/OH⁻ per formula", str(n)))
        results = [("pH", f"{r.pH:.3f}"), ("pOH", f"{r.pOH:.3f}"),
                   ("[H₃O⁺]", _f(r.h), "mol/L"), ("[OH⁻]", _f(r.oh), "mol/L")]
        if r.ionization_pct is not None:
            results.append(("Degree of ionisation", f"{r.ionization_pct:.3f}", "%"))
        if r.approx_pH is not None and kind.startswith("Weak"):
            results.append(("pH (approximation √(KC))", f"{r.approx_pH:.3f}"))
        log = f"CHEM PH {kind.upper().replace(' ', '')} C={c:g}" + \
            (f" K={self.k.text()}" if kind.startswith("Weak") else "") + "."
        return inputs, results, [r.method] + r.notes, None, log


class BufferPage(Page):
    title = "Buffer (Henderson–Hasselbalch)"

    def __init__(self, dialog):
        super().__init__(dialog)
        self.mode = self.choice("Calculate:", "mode", ["pH of a buffer", "Buffer recipe for a "
                                                       "target pH"])
        self.pk, self.pk_type = self.num("pKa / Ka:", "pk", "4.76", ["pKa", "Ka"], "pKa")
        self.acid = self.num("Weak acid HA (mol or M):", "acid", "0.1")
        self.base = self.num("Conjugate base A⁻ (mol or M):", "base", "0.1")
        self.add_acid = self.num("Strong acid added (same units):", "add_acid", "0")
        self.add_base = self.num("Strong base added (same units):", "add_base", "0")
        self.target = self.num("Target pH (recipe):", "target", "5.0")
        self.total = self.num("Total buffer concentration (M):", "total", "0.1")
        self.vol = self.num("Volume (L):", "vol", "1")
        self.acid_f = self.formula("Acid (formula or M):", "acid_f", "CH3COOH")
        self.base_f = self.formula("Base salt (formula or M):", "base_f", "CH3COONa")
        self.mode.currentIndexChanged.connect(self._mode)
        self._mode()

    def _mode(self) -> None:
        ph = self.mode.currentIndex() == 0
        self.show_rows(["acid", "base", "add_acid", "add_base"], ph)
        self.show_rows(["target", "total", "vol", "acid_f", "base_f"], not ph)

    def _pka(self) -> float:
        k = self.v(self.pk, "pKa")
        return -math.log10(k) if self.pk_type.currentIndex() == 1 else k

    def calculate(self):
        pka = self._pka()
        if self.mode.currentIndex() == 0:
            ha, a = self.v(self.acid, "acid amount"), self.v(self.base, "base amount")
            r = sol.buffer_ph(pka, ha, a, self.v(self.add_acid) or 0.0,
                              self.v(self.add_base) or 0.0)
            inputs = [("pKa", f"{pka:.3f}"), ("HA", _f(ha)), ("A⁻", _f(a))]
            if self.v(self.add_acid):
                inputs.append(("Strong acid added", _f(self.v(self.add_acid))))
            if self.v(self.add_base):
                inputs.append(("Strong base added", _f(self.v(self.add_base))))
            results = [("pH", f"{r.pH:.3f}"), ("[A⁻]/[HA]", _f(r.ratio)), ("HA after addition",
                                                                           _f(r.acid)),
                       ("A⁻ after addition", _f(r.base))]
            notes = ["pH = pKa + log([A⁻]/[HA])"] + r.notes
            log = f"CHEM BUFFER PKA={pka:g} ACID={ha:g} BASE={a:g}."
        else:
            target, total = self.v(self.target, "target pH"), self.v(self.total, "total conc.")
            vol = self.v(self.vol, "volume")
            ma = self.acid_f.molar_mass()
            mb = self.base_f.molar_mass()
            r = sol.buffer_recipe(pka, target, total, vol, ma, mb)
            inputs = [("pKa", f"{pka:.3f}"), ("Target pH", f"{target:g}"),
                      ("Total concentration", _f(total), "mol/L"), ("Volume", _f(vol), "L")]
            results = [("[A⁻]/[HA]", _f(r["ratio"])), ("Acid", _f(r["acid_mol"]), "mol"),
                       ("Base", _f(r["base_mol"]), "mol"), ("[HA]", _f(r["acid_M"]), "mol/L"),
                       ("[A⁻]", _f(r["base_M"]), "mol/L")]
            if "acid_g" in r:
                results.append((f"Mass of {self.acid_f.display()}", _f(r["acid_g"]), "g"))
            if "base_g" in r:
                results.append((f"Mass of {self.base_f.display()}", _f(r["base_g"]), "g"))
            notes = ["Ratio from Henderson–Hasselbalch; check the final pH with a pH meter."]
            if abs(target - pka) > 1:
                notes.append("The target pH is more than 1 unit from the pKa: poor buffer "
                             "capacity.")
            log = f"CHEM BUFFER RECIPE PKA={pka:g} PH={target:g} TOTAL={total:g} VOLUME={vol:g}."
        return inputs, results, notes, None, log


class MolalityPage(Page):
    title = "Molality and Normality"

    def __init__(self, dialog):
        super().__init__(dialog)
        self.solute = self.formula("Solute:", "solute", "NaCl")
        self.mass = self.num("Mass of solute (g):", "mass", "5.85")
        self.solvent = self.num("Mass of solvent (g):", "solvent", "500")
        self.molar = self.num("Molarity for normality (M):", "molar", "0.1")
        self.eq = self.num("Equivalents per mole:", "eq", "2")
        self.help.setText("Equivalents: H⁺ or OH⁻ exchanged per formula unit (H₂SO₄ = 2, "
                          "Ca(OH)₂ = 2) or electrons transferred in redox (KMnO₄ in acid = 5).")

    def calculate(self):
        inputs, results = [], []
        mm = self.solute.molar_mass()
        m, solv = self.v(self.mass), self.v(self.solvent)
        log = "CHEM SOLUTION"
        if mm and m and solv:
            n = m / mm
            b = sol.molality(n, solv / 1000)
            inputs += [("Solute", self.solute.display()), ("Molar mass", _f(mm, 6), "g/mol"),
                       ("Mass of solute", _f(m), "g"), ("Mass of solvent", _f(solv), "g")]
            results += [("Amount of solute", _f(n), "mol"), ("Molality", _f(b), "mol/kg"),
                        ("Mass fraction", _f(100 * m / (m + solv)), "% (w/w)")]
            log += f" MOLALITY SOLUTE='{self.solute.text()}' MASS={m:g} SOLVENT={solv:g}"
        c, e = self.v(self.molar), self.v(self.eq)
        if c and e:
            inputs += [("Molarity", _f(c), "mol/L"), ("Equivalents per mole", _f(e))]
            results.append(("Normality", _f(sol.normality(c, e)), "eq/L (N)"))
            log += f" NORMALITY M={c:g} EQ={e:g}"
        if not results:
            raise ValueError("Enter solute, masses, or molarity and equivalents.")
        return inputs, results, [], None, log + "."


class GasPage(Page):
    title = "Ideal Gas Law"

    def __init__(self, dialog):
        super().__init__(dialog)
        self.p, self.pu = self.num("Pressure P:", "p", "1", ["atm", "kPa", "bar", "mmHg"], "atm")
        self.vv, self.vu = self.num("Volume V:", "v", "", ["L", "mL", "m³"], "L")
        self.n = self.num("Amount n (mol):", "n", "1")
        self.t, self.tu = self.num("Temperature T:", "t", "0", ["°C", "K"], "°C")
        self.help.setText("PV = nRT — leave exactly one field empty. "
                          "R = 0.082057 L·atm/(mol·K).")

    def calculate(self):
        pf = {"atm": 1.0, "kPa": 1 / 101.325, "bar": 1 / 1.01325, "mmHg": 1 / 760}
        vf = {"L": 1.0, "mL": 1e-3, "m³": 1e3}
        p, v, n, t = self.v(self.p), self.v(self.vv), self.v(self.n), self.v(self.t)
        P = p * pf[self.pu.currentText()] if p is not None else None
        V = v * vf[self.vu.currentText()] if v is not None else None
        T = (t + 273.15 if self.tu.currentText() == "°C" else t) if t is not None else None
        vals = sol.ideal_gas(P, V, n, T)
        inputs = [(k, _f(vals[k]), u) for k, u in (("P", "atm"), ("V", "L"), ("n", "mol"),
                                                   ("T", "K"))
                  if {"P": P, "V": V, "n": n, "T": T}[k] is not None]
        missing = next(k for k, x in {"P": P, "V": V, "n": n, "T": T}.items() if x is None)
        unit = {"P": "atm", "V": "L", "n": "mol", "T": "K"}[missing]
        results = [(missing, _f(vals[missing], 5), unit)]
        if missing == "P":
            results.append(("", _f(vals["P"] * 101.325, 5), "kPa"))
        if missing == "T":
            results.append(("", _f(vals["T"] - 273.15, 5), "°C"))
        log = "CHEM GAS " + " ".join(f"{k}={vals[k]:g}" for k in "PVnT") + "."
        return inputs, results, ["PV = nRT"], None, log


class UnitPage(Page):
    title = "Unit Conversion"

    def __init__(self, dialog):
        super().__init__(dialog)
        self.cat = self.choice("Quantity:", "cat", list(sol.UNITS), "Mass")
        self.value = self.num("Value:", "value", "1")
        self.src = combo([])
        self.dst = combo([])
        self.form.addRow("From:", self.src)
        self.form.addRow("To:", self.dst)
        self.cat.currentIndexChanged.connect(self._units)
        self.src.currentIndexChanged.connect(dialog.schedule)
        self.dst.currentIndexChanged.connect(dialog.schedule)
        self._units()

    def _units(self) -> None:
        units = list(sol.UNITS[self.cat.currentText()])
        for box, i in ((self.src, 0), (self.dst, min(1, len(units) - 1))):
            box.blockSignals(True)
            box.clear()
            box.addItems(units)
            box.setCurrentIndex(i)
            box.blockSignals(False)
        self.dialog.schedule()

    def calculate(self):
        x = self.v(self.value, "value")
        cat, a = self.cat.currentText(), self.src.currentText()
        rows = [[u, _f(sol.convert(x, a, u, cat), 6)] for u in sol.UNITS[cat]]
        y = sol.convert(x, a, self.dst.currentText(), cat)
        table = Table(f"{cat} Conversions", ["Unit", "Value"], rows)
        log = f"CHEM CONVERT {x:g} '{a}' TO '{self.dst.currentText()}'."
        return ([("Value", _f(x, 6), a)], [("Converted value", _f(y, 6), self.dst.currentText())],
                [], [table], log)


PAGES = [("Prepare Solution", PreparePage), ("Stock Reagent", StockPage),
         ("Dilution", DilutionPage), ("Serial Dilution", SerialPage),
         ("Concentration Units", ConcUnitsPage), ("pH", PHPage), ("Buffer", BufferPage),
         ("Molality && Normality", MolalityPage), ("Gas Law", GasPage),
         ("Unit Conversion", UnitPage)]


class SolutionCalculatorDialog(CalculatorDialog):
    """Lab calculators chosen from a list on the left; OK writes the shown one to the output."""

    def __init__(self, ds=None, parent=None, tab: int | None = None):
        super().__init__(ds, parent, "Solution Calculator")
        self.setMinimumSize(780, 520)
        self.nav = QListWidget()
        self.nav.setFixedWidth(170)
        self.nav.setSpacing(3)
        self.stack = QStackedWidget()
        self.pages: list[Page] = []
        for title, cls in PAGES:
            page = cls(self)
            self.pages.append(page)
            self.stack.addWidget(page)
            self.nav.addItem(title.replace("&&", "&"))
        self.remember("tab", self.stack)
        if tab is not None:
            self.stack.setCurrentIndex(tab)
        self.nav.setCurrentRow(self.stack.currentIndex())
        self.nav.currentRowChanged.connect(self._select)
        row = QHBoxLayout()
        row.addWidget(self.nav)
        row.addWidget(self.stack, 1)
        self.body.addLayout(row)
        self.preview()

    def _select(self, i: int) -> None:
        if i >= 0:
            self.stack.setCurrentIndex(i)
            self.schedule()

    def preview(self) -> None:
        self.pages[self.stack.currentIndex()].refresh()

    def _refresh(self) -> None:
        self.preview()

    def build(self):
        return self.pages[self.stack.currentIndex()].section()
