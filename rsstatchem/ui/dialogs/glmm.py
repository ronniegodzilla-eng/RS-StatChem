"""Analyze > Mixed Models > Generalized Linear (SPSS GENLINMIXED), laid out after the SPSS
dialog: Data Structure (subjects), Fields && Effects (target with distribution and link, fixed
effects, random effect blocks, weight and offset), Build Options, Estimated Means and Save.
Core code: :mod:`rsstatchem.core.stats.glmm`."""
from __future__ import annotations

import numpy as np
import pandas as pd
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                               QListWidget, QPushButton, QVBoxLayout)

from rsstatchem.core.dataset import NUMERIC, format_code
from rsstatchem.core.stats import glmm as gm
from rsstatchem.core.stats.glm_model import full_factorial, main_effects, parse_terms
from rsstatchem.ui.dialogs import glm_model_ui as gmu
from rsstatchem.ui.dialogs.base import SubDialog, VariablePickerDialog, is_numeric


def _buttons(dlg: QDialog, layout) -> None:
    box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                           | QDialogButtonBox.StandardButton.Cancel)
    box.button(QDialogButtonBox.StandardButton.Ok).setText("Continue")
    box.accepted.connect(dlg.accept)
    box.rejected.connect(dlg.reject)
    layout.addWidget(box)


class RandomBlockDialog(QDialog):
    """GENLINMIXED 'Random Effect Block': effects, intercept, subject combination,
    covariance type and the parameter predictions switch."""

    def __init__(self, effects: list[str], subjects: list[str], block: dict | None,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("Random Effect Block")
        block = block or {"effects": [], "intercept": True,
                          "subjects": subjects[:1], "covtype": "VC", "predictions": False}
        lay = QVBoxLayout(self)
        box = QGroupBox("Effects (random slopes for covariates, factor effects)")
        bl = QVBoxLayout(box)
        self.checks = {}
        for e in effects:
            cb = QCheckBox(e)
            cb.setChecked(e in block["effects"])
            bl.addWidget(cb)
            self.checks[e] = cb
        if not effects:
            bl.addWidget(QLabel("(Add factors or covariates in the main dialog to use them "
                                "as random effects.)"))
        lay.addWidget(box)
        self.intercept = QCheckBox("Include intercept")
        self.intercept.setChecked(block["intercept"])
        lay.addWidget(self.intercept)
        self.predictions = QCheckBox("Display parameter predictions for this block")
        self.predictions.setChecked(block.get("predictions", False))
        lay.addWidget(self.predictions)
        form = QFormLayout()
        self.subject = QComboBox()
        self.subject.addItem("None", [])
        for k in range(1, len(subjects) + 1):
            self.subject.addItem(" * ".join(subjects[:k]), subjects[:k])
        cur = block.get("subjects") or []
        idx = next((i for i in range(self.subject.count())
                    if self.subject.itemData(i) == cur), 1 if subjects else 0)
        self.subject.setCurrentIndex(idx)
        form.addRow("Subject combination:", self.subject)
        self.covtype = QComboBox()
        for k, text in gm.COVTYPES.items():
            self.covtype.addItem(text, k)
        self.covtype.setCurrentIndex(max(self.covtype.findData(block["covtype"]), 0))
        form.addRow("Random effect covariance type:", self.covtype)
        lay.addLayout(form)
        _buttons(self, lay)

    def result(self) -> dict:
        return {"effects": [e for e, cb in self.checks.items() if cb.isChecked()],
                "intercept": self.intercept.isChecked(),
                "subjects": self.subject.currentData(),
                "covtype": self.covtype.currentData(),
                "predictions": self.predictions.isChecked()}


def block_text(b: dict) -> str:
    parts = (["Intercept"] if b["intercept"] else []) + list(b["effects"])
    subj = " * ".join(b["subjects"]) or "None"
    return f"{' + '.join(parts) or '(no effects)'} | Subject: {subj} | {gm.COVTYPES[b['covtype']]}"


class RandomEffectsDialog(QDialog):
    """List of random effect blocks with Add Block / Edit Block / Delete."""

    def __init__(self, effects: list[str], subjects: list[str], blocks: list[dict],
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("Generalized Linear Mixed Models: Random Effects")
        self.effects, self.subjects = effects, subjects
        self.blocks = [dict(b) for b in blocks]
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Random Effect Blocks:"))
        self.list = QListWidget()
        lay.addWidget(self.list, 1)
        row = QHBoxLayout()
        for text, fn in (("Add Block...", self._add), ("Edit Block...", self._edit),
                         ("Delete", self._delete)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        row.addStretch()
        lay.addLayout(row)
        hint = QLabel("A random intercept for a subject is the usual multilevel model; add "
                      "covariates as random slopes. Nested subjects (school, school * class) "
                      "are separate blocks.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#666")
        lay.addWidget(hint)
        _buttons(self, lay)
        self._refresh()
        self.resize(560, 360)

    def _refresh(self):
        self.list.clear()
        for k, b in enumerate(self.blocks):
            self.list.addItem(f"Random Effect {k + 1}: {block_text(b)}")

    def _add(self):
        dlg = RandomBlockDialog(self.effects, self.subjects, None, self)
        if dlg.exec():
            self.blocks.append(dlg.result())
            self._refresh()

    def _edit(self):
        row = self.list.currentRow()
        if row < 0:
            return
        dlg = RandomBlockDialog(self.effects, self.subjects, self.blocks[row], self)
        if dlg.exec():
            self.blocks[row] = dlg.result()
            self._refresh()

    def _delete(self):
        row = self.list.currentRow()
        if row >= 0:
            del self.blocks[row]
            self._refresh()


class GeneralizedLinearMixedDialog(VariablePickerDialog):
    DEFAULTS = {"preset": "linear", "distribution": "normal", "link": "identity", "power": 1.0,
                "terms": None, "intercept": True, "blocks": None, "reference": "last",
                "trials_value": 0.0, "method": "pql", "max_iter": 100, "level": 95.0,
                "robust": False, "emm": [], "emm_scale": "original", "contrast": "none",
                "contrast_field": "", "adjust": "lsd", "save": [], "classification": True,
                "pseudo_r2": True}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Generalized Linear Mixed Models", parent)
        self.setMinimumSize(700, 620)
        self.add_target("subjects", "Subjects (Data Structure):", required=False)
        self.add_target("target", "Target:", single=True)
        self.add_target("trials", "Trials (events/trials target):", accepts=is_numeric,
                        single=True, required=False)
        self.add_target("factors", "Fixed Effects - Factors:", required=False)
        self.add_target("covariates", "Fixed Effects - Covariates:", accepts=is_numeric,
                        required=False)
        self.add_target("offset", "Offset:", accepts=is_numeric, single=True, required=False)
        self.add_target("weight", "Analysis Weight:", accepts=is_numeric, single=True,
                        required=False)
        self.add_option_button("Target Options...", self._target_options)
        self.add_option_button("Fixed Effects...", self._fixed)
        self.add_option_button("Random Effects...", self._random)
        self.add_option_button("Build Options...", self._build)
        self.add_option_button("Estimated Means...", self._emm)
        self.add_option_button("Save...", self._save)

        form = QFormLayout()
        self.preset = QComboBox()
        for k, (_, _, text) in gm.PRESETS.items():
            self.preset.addItem(text, k)
        self.preset.addItem("Custom", "custom")
        self.dist = QComboBox()
        for k, text in gm.DISTRIBUTIONS.items():
            self.dist.addItem(text, k)
        self.link = QComboBox()
        for k, text in gm.LINKS.items():
            self.link.addItem(text, k)
        self.power = QDoubleSpinBox()
        self.power.setRange(-10, 10)
        self.power.setDecimals(3)
        self.power.setValue(self.opts["power"])
        form.addRow("Target distribution and relationship (link):", self.preset)
        row = QHBoxLayout()
        row.addWidget(QLabel("Distribution:"))
        row.addWidget(self.dist)
        row.addWidget(QLabel("Link function:"))
        row.addWidget(self.link)
        row.addWidget(QLabel("Power:"))
        row.addWidget(self.power)
        form.addRow(row)
        self.options_layout.addLayout(form)
        self.preset.setCurrentIndex(max(self.preset.findData(self.opts["preset"]), 0))
        self.dist.setCurrentIndex(max(self.dist.findData(self.opts["distribution"]), 0))
        self.link.setCurrentIndex(max(self.link.findData(self.opts["link"]), 0))
        self.preset.currentIndexChanged.connect(self._preset_changed)
        self._preset_changed()
        self.finish()

    def _preset_changed(self, *_):
        key = self.preset.currentData()
        custom = key == "custom"
        self.dist.setEnabled(custom)
        self.link.setEnabled(custom)
        if not custom:
            d, l, _ = gm.PRESETS[key]
            self.dist.setCurrentIndex(self.dist.findData(d))
            self.link.setCurrentIndex(self.link.findData(l))
        self.power.setEnabled(custom)

    def _effects(self) -> list[str]:
        return self.selected("factors") + self.selected("covariates")

    def _target_options(self):
        dlg = SubDialog("Generalized Linear Mixed Models: Target", self, columns=1)
        box = dlg.group("Categorical target (binomial)")
        combo = QComboBox()
        combo.addItem("Last category (default)", "last")
        combo.addItem("First category", "first")
        t = self.selected_one("target")
        if t:
            vals = sorted(self.ds.analysis_series(t).dropna().unique(),
                          key=lambda v: (isinstance(v, str), v))[:50]
            for v in vals:
                combo.addItem(f"Value: {format_code(v) if not isinstance(v, str) else v}", v)
        combo.setCurrentIndex(max(combo.findData(self.opts["reference"]), 0))
        row = QFormLayout()
        row.addRow("Reference category:", combo)
        box.addLayout(row)
        dlg._getters["reference"] = combo.currentData
        note = QLabel("The probability of the non-reference category is modelled (SPSS: the "
                      "last, highest-valued category is the reference by default).")
        note.setWordWrap(True)
        note.setStyleSheet("color:#666")
        box.addWidget(note)
        box = dlg.group("Use number of trials as denominator")
        dlg.number(box, "trials_value", "Fixed value (0 = use the Trials variable):",
                   self.opts["trials_value"], 0, 1e9, 0)
        values = dlg.run()
        if values:
            self.opts.update(values)

    def _fixed(self):
        factors, covs = self.selected("factors"), self.selected("covariates")
        terms = self.opts.get("terms")
        if terms is None:
            terms = [t.syntax for t in main_effects(factors, covs)]
        dlg = gmu.ModelDialog("Generalized Linear Mixed Models", factors, covs, terms or None,
                              self.opts.get("intercept", True), 3, self)
        dlg.setWindowTitle("Generalized Linear Mixed Models: Fixed Effects")
        dlg.ss.setEnabled(False)
        dlg.ss.setToolTip("GENLINMIXED tests fixed effects with Type III tests.")
        if dlg.exec():
            v = dlg.result_values()
            self.opts["terms"] = v["terms"] if v["terms"] is not None else \
                [t.syntax for t in full_factorial(factors, covs)]
            self.opts["intercept"] = v["intercept"]

    def _default_blocks(self) -> list[dict]:
        subj = self.selected("subjects")
        if not subj:
            return []
        if len(subj) == 1:
            return [{"effects": [], "intercept": True, "subjects": subj[:1], "covtype": "VC",
                     "predictions": False}]
        return [{"effects": [], "intercept": True, "subjects": subj[:k], "covtype": "VC",
                 "predictions": False} for k in range(1, len(subj))]

    def _blocks(self) -> list[dict]:
        blocks = self.opts.get("blocks")
        if blocks is None:
            return self._default_blocks()
        subj = self.selected("subjects")
        eff = set(self._effects())
        out = []
        for b in blocks:
            if any(s not in subj for s in b["subjects"]):
                continue
            out.append(dict(b, effects=[e for e in b["effects"] if e in eff]))
        return out

    def _random(self):
        dlg = RandomEffectsDialog(self._effects(), self.selected("subjects"), self._blocks(),
                                  self)
        if dlg.exec():
            self.opts["blocks"] = dlg.blocks

    def _build(self):
        dlg = SubDialog("Generalized Linear Mixed Models: Build Options", self, columns=2)
        box = dlg.group("Estimation")
        dlg.radio(box, "method", list(gm.METHODS.items()), self.opts["method"])
        dlg.number(box, "max_iter", "Maximum iterations:", self.opts["max_iter"], 1, 1000, 0)
        box = dlg.group("Post-Estimation")
        dlg.number(box, "level", "Confidence level (%):", self.opts["level"], 50, 99.99, 2)
        box.addWidget(QLabel("Degrees of freedom: Residual method"))
        box.addWidget(QLabel("Tests of fixed effects and coefficients:"))
        dlg.radio(box, "robust", [(False, "Use model-based covariance"),
                                  (True, "Use robust covariance")], self.opts["robust"])
        box = dlg.group("Output")
        dlg.check(box, "classification", "Classification table (binary target)",
                  self.opts["classification"])
        dlg.check(box, "pseudo_r2", "Pseudo-R2 and intraclass correlation",
                  self.opts["pseudo_r2"])
        values = dlg.run()
        if values:
            values["max_iter"] = int(values["max_iter"])
            self.opts.update(values)

    def _emm(self):
        factors = self.selected("factors")
        terms = self.opts.get("terms") or [t.syntax for t in main_effects(factors, [])]
        effects = gmu.effects_of(factors, terms)
        dlg = SubDialog("Generalized Linear Mixed Models: Estimated Means", self, columns=2)
        box = dlg.group("Terms")
        for e in effects:
            dlg.check(box, f"emm::{e}", e.replace("*", " * "), e in self.opts["emm"])
        if not effects:
            box.addWidget(QLabel("(No fixed-effect terms made only of factors.)"))
        box = dlg.group("Contrasts")
        dlg.radio(box, "contrast", list(gm.CONTRASTS.items()), self.opts["contrast"])
        combo = QComboBox()
        combo.addItem("(first factor of the term)", "")
        for f in factors:
            combo.addItem(f, f)
        combo.setCurrentIndex(max(combo.findData(self.opts["contrast_field"]), 0))
        row = QFormLayout()
        row.addRow("Contrast field:", combo)
        box.addLayout(row)
        dlg._getters["contrast_field"] = combo.currentData
        box = dlg.group("Display estimated means in terms of")
        dlg.radio(box, "emm_scale", [("original", "Original target scale"),
                                     ("link", "Link function transformation")],
                  self.opts["emm_scale"])
        box = dlg.group("Adjust for multiple comparisons using")
        dlg.radio(box, "adjust", list(gm.ADJUST.items()), self.opts["adjust"])
        values = dlg.run()
        if values:
            self.opts["emm"] = [k.split("::", 1)[1] for k, v in values.items()
                                if k.startswith("emm::") and v]
            for key in ("contrast", "contrast_field", "emm_scale", "adjust"):
                self.opts[key] = values[key]

    def _save(self):
        dlg = SubDialog("Generalized Linear Mixed Models: Save", self, columns=1)
        box = dlg.group("Save Fields")
        for key, text in (("predicted", "Predicted values (PredictedValue)"),
                          ("probability", "Predicted probability for categorical targets "
                                          "(PredictedProbability)"),
                          ("ci", "Confidence intervals (CI_Lower, CI_Upper)"),
                          ("pearson", "Pearson residuals (PearsonResidual)")):
            dlg.check(box, key, text, key in self.opts["save"])
        values = dlg.run()
        if values is not None:
            self.opts["save"] = [k for k, v in values.items() if v]

    def validate(self):
        self.opts["preset"] = self.preset.currentData()
        self.opts["distribution"] = self.dist.currentData()
        self.opts["link"] = self.link.currentData()
        self.opts["power"] = self.power.value()
        target = self.selected_one("target")
        others = self._effects() + self.selected("subjects")
        if target in others:
            return "The target cannot also be a predictor or subject."
        if set(self.selected("subjects")) & set(self._effects()):
            return "A subject field cannot also be a fixed effect."
        dist, link = self.opts["distribution"], self.opts["link"]
        if link in ("logit", "probit", "cloglog", "nloglog", "logc", "cauchit") and \
                dist != "binomial":
            return f"The {gm.LINKS[link]} link is appropriate only with the binomial " \
                   "distribution."
        if dist != "binomial" and self.selected_one("trials"):
            return "A trials variable is used only with the binomial distribution."
        if dist != "binomial" and self.ds.var(target).type != NUMERIC:
            return "A string target can only have the binomial distribution."
        if self.opts.get("terms"):
            try:
                parse_terms(" ".join(self.opts["terms"]), self.selected("factors"),
                            self.selected("covariates"))
            except ValueError as exc:
                return f"The fixed-effects model refers to variables that are not selected " \
                       f"({exc})."
        return None

    def compute(self):
        o = self.opts
        factors, covs = self.selected("factors"), self.selected("covariates")
        terms = parse_terms(" ".join(o["terms"]), factors, covs) if o.get("terms") else None
        blocks = [gm.RandomBlock(effects=b["effects"], intercept=b["intercept"],
                                 subjects=b["subjects"], covtype=b["covtype"],
                                 predictions=b.get("predictions", False))
                  for b in self._blocks()]
        names = set(factors)
        emm = [e for e in o.get("emm", []) if all(p in names for p in e.split("*"))]
        ref = o["reference"]
        res = gm.glmm(self.ds, self.selected_one("target"), factors, covs,
                      self.selected("subjects"), blocks, o["distribution"], o["link"],
                      o["power"], self.selected_one("trials"),
                      o["trials_value"] if o["trials_value"] and
                      not self.selected_one("trials") and o["distribution"] == "binomial"
                      else None,
                      ref if ref in ("last", "first") else "last",
                      None if ref in ("last", "first") else ref,
                      self.selected_one("offset"), self.selected_one("weight"), terms,
                      o.get("intercept", True), o["method"], int(o["max_iter"]),
                      o["level"] / 100, o["robust"], emm, o["emm_scale"], o["contrast"],
                      o["contrast_field"] or None, o["adjust"], None, o["save"],
                      o["classification"], o["pseudo_r2"])
        return res if res.saved else res.section


MENUS = {"Analyze/Mixed Models": [("Generalized Linear...", GeneralizedLinearMixedDialog)]}


def selftest() -> None:
    gm.selftest()
    from rsstatchem.core.dataset import Dataset

    rng = np.random.default_rng(2)
    g = np.repeat(np.arange(10), 6)
    ds = Dataset(pd.DataFrame({"y": rng.normal(size=60) + rng.normal(size=10)[g],
                               "x": rng.normal(size=60), "g": g.astype(float)}))
    r = gm.glmm(ds, "y", covariates=["x"], subjects=["g"],
                random_blocks=[gm.RandomBlock(subjects=["g"])])
    assert r.fit is not None and r.fit.phi > 0
