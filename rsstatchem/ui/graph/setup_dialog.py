"""Graphs > Plot Setup dialog (OriginLab's "Plot Setup"), in the SPSS variable-picker style.

Choose a plot type, then move variables into X, Y (one or more), Z, X/Y Error, Labels and
Group-by. Only the roles the chosen plot type uses are shown. Data are read through the
Dataset API (``common.numeric``) so user-missing values are honoured.
"""
from __future__ import annotations

from PySide6.QtGui import QStandardItemModel
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QHBoxLayout, QLabel,
                               QLineEdit, QListWidget, QPushButton, QVBoxLayout, QWidget)

from rsstatchem.core.graph.build import ROLE_LABELS, ROLES, SUMMARIES, TEMPLATES, PlotSpec
from rsstatchem.core.graph.themes import DEFAULT_THEME, THEMES, list_user_templates
from rsstatchem.ui.dialogs.base import VariablePickerDialog, is_numeric

NUMERIC_ROLES = ("y", "z", "xerr", "yerr")


class PlotSetupDialog(VariablePickerDialog):
    DEFAULTS = {"kind": "line", "summary": "none", "theme": DEFAULT_THEME, "bins": "auto",
                "normal_curve": True, "title": ""}

    def __init__(self, ds, parent=None, kind: str | None = None, add_mode: bool = False):
        self._blocks: dict[str, tuple[QWidget, QLabel]] = {}
        self._pending_block = None
        super().__init__(ds, "Plot Setup" if not add_mode else "Add Plot to Layer", parent)
        self.setMinimumSize(720, 560)
        if kind:
            self.opts["kind"] = kind
        self.add_mode = add_mode

        top = QHBoxLayout()
        top.addWidget(QLabel("Plot type:"))
        self.kind_combo = QComboBox()
        self.kind_combo.setMaxVisibleItems(30)
        model: QStandardItemModel = self.kind_combo.model()
        category = None
        for t in TEMPLATES.values():
            if t.category != category:
                category = t.category
                self.kind_combo.addItem(f"— {category} —")
                model.item(self.kind_combo.count() - 1).setEnabled(False)
            self.kind_combo.addItem("    " + t.label, t.key)
        top.addWidget(self.kind_combo, 1)
        self.layout().insertLayout(0, top)

        for role in ROLES:
            accepts = is_numeric if role in NUMERIC_ROLES else None
            self.add_target(role, ROLE_LABELS[role], accepts=accepts, single=role != "y",
                            required=False)

        form = QFormLayout()
        self.summary_combo = QComboBox()
        for key, text in SUMMARIES.items():
            self.summary_combo.addItem(text, key)
        self.summary_combo.setToolTip("Summarise Y for each X category (or for each Y "
                                      "variable when X is empty), like Origin's statistics "
                                      "column/bar charts.")
        self.theme_combo = QComboBox()
        for name in THEMES:
            self.theme_combo.addItem(name, name)
        for name in list_user_templates():
            self.theme_combo.addItem(f"Template: {name}", f"user:{name}")
        self.title_edit = QLineEdit(self.opts.get("title", ""))
        self.title_edit.setPlaceholderText("optional — supports H$_2$O, cm$^{-1}$")
        self.bins_edit = QLineEdit(str(self.opts.get("bins", "auto")))
        self.bins_edit.setPlaceholderText("auto or a number")
        self.normal_check = QCheckBox("Normal curve")
        self.normal_check.setChecked(bool(self.opts.get("normal_curve", True)))
        hist_row = QHBoxLayout()
        hist_row.addWidget(self.bins_edit)
        hist_row.addWidget(self.normal_check)
        self.hist_widget = QWidget()
        self.hist_widget.setLayout(hist_row)
        hist_row.setContentsMargins(0, 0, 0, 0)
        form.addRow("Summary:", self.summary_combo)
        form.addRow("Bins:", self.hist_widget)
        form.addRow("Theme:", self.theme_combo)
        form.addRow("Graph title:", self.title_edit)
        self._form = form
        self.options_layout.addLayout(form)
        self.hint = QLabel()
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet("color:#555;")
        self.options_layout.addWidget(self.hint)
        if add_mode:
            self.theme_combo.setEnabled(False)
            self.title_edit.setEnabled(False)

        idx = self.kind_combo.findData(self.opts["kind"])
        self.kind_combo.setCurrentIndex(idx if idx >= 0 else 1)
        self.summary_combo.setCurrentIndex(max(0, self.summary_combo.findData(
            self.opts.get("summary", "none"))))
        self.theme_combo.setCurrentIndex(max(0, self.theme_combo.findData(
            self.opts.get("theme", DEFAULT_THEME))))
        self.kind_combo.currentIndexChanged.connect(self._kind_changed)
        self.finish()
        self._kind_changed()

    # ------------------------------------------------------------------ layout hooks
    def _target_block(self, label: str, widget: QListWidget, arrow: QPushButton,
                      stretch: int) -> None:
        container = QWidget()
        title = QLabel(label)
        arrow_column = QVBoxLayout()
        arrow_column.addSpacing(title.sizeHint().height() + 4)
        arrow_column.addWidget(arrow)
        arrow_column.addStretch()
        list_column = QVBoxLayout()
        list_column.addWidget(title)
        list_column.addWidget(widget, 1)
        block = QHBoxLayout(container)
        block.setContentsMargins(0, 0, 0, 0)
        block.addLayout(arrow_column)
        block.addLayout(list_column, 1)
        self._targets_column.addWidget(container, stretch)
        self._pending_block = (container, title)

    def add_target(self, key, label, accepts=None, single=False, required=True,
                   min_items=None):
        widget = super().add_target(key, label, accepts, single, required, min_items)
        self._blocks[key] = self._pending_block
        return widget

    def _move_to_first_accepting(self) -> None:
        from rsstatchem.ui.dialogs.base import NAME_ROLE
        for name in [i.data(NAME_ROLE) for i in self.source.selectedItems()]:
            for target in self._targets:
                if self._blocks[target.key][0].isHidden():
                    continue
                if (not target.single or not target.widget.count()) and \
                        self._take(name, target):
                    break
        self._update_arrows()

    # ------------------------------------------------------------------ state
    def kind(self) -> str:
        return self.kind_combo.currentData() or "line"

    def _kind_changed(self) -> None:
        if self.kind_combo.currentData() is None:   # a category header
            self.kind_combo.setCurrentIndex(self.kind_combo.currentIndex() + 1)
            return
        t = TEMPLATES[self.kind()]
        for role, (container, title) in self._blocks.items():
            container.setHidden(role not in t.roles)
            text = t.role_text.get(role, ROLE_LABELS[role]).replace("&&", "&")
            if role in t.required:
                text = text.rstrip(":") + " *:"
            title.setText(text)
        summary_ok = t.summary or t.plot in ("pie", "doughnut")
        self.summary_combo.setEnabled(summary_ok)
        self._form.labelForField(self.summary_combo).setEnabled(summary_ok)
        hist = t.plot == "histogram"
        self.hist_widget.setVisible(hist)
        self._form.labelForField(self.hist_widget).setVisible(hist)
        self.hint.setText(t.hint)
        self.hint.setVisible(bool(t.hint))
        self.setWindowTitle(("Add Plot to Layer" if getattr(self, "add_mode", False)
                             else "Plot Setup") + f": {t.label}")

    def spec(self) -> PlotSpec:
        t = TEMPLATES[self.kind()]

        def one(role):
            return self.selected_one(role) if role in t.roles else None
        options = {}
        if t.plot == "histogram":
            bins = self.bins_edit.text().strip() or "auto"
            options["bins"] = int(bins) if bins.isdigit() else bins
            options["normal_curve"] = self.normal_check.isChecked()
        theme = self.theme_combo.currentData() or DEFAULT_THEME
        return PlotSpec(kind=t.key, x=one("x"), ys=self.selected("y") if "y" in t.roles else [],
                        z=one("z"), xerr=one("xerr"), yerr=one("yerr"), labels=one("labels"),
                        group=one("group"),
                        summary=self.summary_combo.currentData()
                        if self.summary_combo.isEnabled() else "none",
                        theme=theme, title=self.title_edit.text().strip(), options=options)

    def validate(self) -> str | None:
        spec = self.spec()
        if spec.theme.startswith("user:"):
            spec.theme = DEFAULT_THEME
        error = spec.validate(self.ds)
        if error:
            return error
        self.opts.update(kind=spec.kind, summary=self.summary_combo.currentData(),
                         theme=self.theme_combo.currentData(), title=spec.title,
                         bins=self.bins_edit.text() or "auto",
                         normal_curve=self.normal_check.isChecked())
        return None


def blocks_visible(dlg: PlotSetupDialog) -> list[str]:
    """Roles currently offered (for tests)."""
    return [k for k, (w, _) in dlg._blocks.items() if not w.isHidden()]

