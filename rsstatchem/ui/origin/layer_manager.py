"""Layer Management dialog (Origin's Graph > Layer Management): reorder, add and delete
layers, arrange them on a rows × cols grid with spacing, link axis ranges to another layer
and place insets. Every change is applied to the graph window at once (undoable)."""
from __future__ import annotations

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout,
                               QListWidget, QPushButton, QSpinBox, QVBoxLayout)

from rsstatchem.core.graph import edit, layout


def _layer_text(graph, k, layer) -> str:
    kind = "" if layer.kind == "cartesian" else f" ({layer.kind})"
    inset = " (inset)" if layer.rect else ""
    names = ", ".join(p.name or p.type for p in layer.plots[:3])
    more = "…" if len(layer.plots) > 3 else ""
    return f"Layer {k + 1}{kind}{inset}: {names}{more}"


class LayerManagerDialog(QDialog):
    def __init__(self, gwin, parent=None):
        super().__init__(parent)
        self.gwin = gwin
        self.setWindowTitle(f"Layer Management — {gwin.graph.title}")
        self.resize(560, 520)
        self.list = QListWidget()
        self.list.currentRowChanged.connect(self._sync)
        up, down = QPushButton("Move Up"), QPushButton("Move Down")
        add, delete = QPushButton("Add Layer"), QPushButton("Delete")
        up.clicked.connect(lambda: self._move(-1))
        down.clicked.connect(lambda: self._move(1))
        add.clicked.connect(self._add)
        delete.clicked.connect(self._delete)
        row = QHBoxLayout()
        for b in (up, down, add, delete):
            row.addWidget(b)

        arr = QGroupBox("Arrange (non-inset layers, row by row)")
        af = QFormLayout(arr)
        n = max(1, len(layout.gridded(gwin.graph)))
        self.rows, self.cols = QSpinBox(), QSpinBox()
        self.rows.setRange(1, 12)
        self.cols.setRange(1, 12)
        self.rows.setValue(max(1, gwin.graph.rows))
        self.cols.setValue(max(1, gwin.graph.cols if gwin.graph.rows * gwin.graph.cols >= n
                               else n))
        self.hspace, self.wspace = QDoubleSpinBox(), QDoubleSpinBox()
        for sb, key in ((self.hspace, "hspace"), (self.wspace, "wspace")):
            sb.setRange(0.0, 1.0)
            sb.setSingleStep(0.02)
            sb.setValue(float(gwin.graph.meta.get(key, 0.04)))
        af.addRow("Rows", self.rows)
        af.addRow("Columns", self.cols)
        af.addRow("Vertical gap", self.hspace)
        af.addRow("Horizontal gap", self.wspace)
        apply_arr = QPushButton("Arrange")
        apply_arr.clicked.connect(self.arrange)
        af.addRow(apply_arr)

        link = QGroupBox("Link axes of the selected layer")
        lf = QFormLayout(link)
        self.master = QComboBox()
        self.link_x, self.link_y = QCheckBox("X range"), QCheckBox("Y range")
        lf.addRow("Follow layer", self.master)
        lf.addRow(self.link_x)
        lf.addRow(self.link_y)
        apply_link = QPushButton("Apply Link")
        apply_link.clicked.connect(self.apply_link)
        lf.addRow(apply_link)

        ins = QGroupBox("Inset position (fraction of the page)")
        inf = QFormLayout(ins)
        self.rect = []
        for name in ("Left", "Bottom", "Width", "Height"):
            sb = QDoubleSpinBox()
            sb.setRange(0.0, 1.0)
            sb.setDecimals(3)
            sb.setSingleStep(0.02)
            self.rect.append(sb)
            inf.addRow(name, sb)
        self.make_inset = QCheckBox("Free position (inset)")
        inf.addRow(self.make_inset)
        apply_rect = QPushButton("Apply Position")
        apply_rect.clicked.connect(self.apply_rect)
        inf.addRow(apply_rect)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.accept)
        lay = QVBoxLayout(self)
        lay.addWidget(self.list, 1)
        lay.addLayout(row)
        side = QHBoxLayout()
        side.addWidget(arr)
        side.addWidget(link)
        lay.addLayout(side)
        lay.addWidget(ins)
        lay.addWidget(buttons)
        self.refresh()

    # ------------------------------------------------------------------ state
    @property
    def graph(self):
        return self.gwin.graph

    def refresh(self, select: int | None = None) -> None:
        cur = self.list.currentRow() if select is None else select
        self.list.blockSignals(True)
        self.list.clear()
        for k, layer in enumerate(self.graph.layers):
            self.list.addItem(_layer_text(self.graph, k, layer))
        self.list.blockSignals(False)
        self.master.clear()
        self.master.addItem("(none)", None)
        for k in range(len(self.graph.layers)):
            self.master.addItem(f"Layer {k + 1}", k)
        self.list.setCurrentRow(min(max(cur, 0), len(self.graph.layers) - 1))
        self._sync(self.list.currentRow())

    def _sync(self, row: int) -> None:
        if not 0 <= row < len(self.graph.layers):
            return
        layer = self.graph.layers[row]
        master = layer.options.get("link_x", layer.options.get("link_y"))
        idx = self.master.findData(master)
        self.master.setCurrentIndex(max(0, idx))
        self.link_x.setChecked("link_x" in layer.options)
        self.link_y.setChecked("link_y" in layer.options)
        rect = layer.rect or [0.2, 0.55, 0.3, 0.3]
        for sb, v in zip(self.rect, rect):
            sb.setValue(float(v))
        self.make_inset.setChecked(bool(layer.rect))

    def current(self) -> int:
        return max(0, self.list.currentRow())

    def _do(self, fn, select=None) -> None:
        self.gwin.mutate(fn)
        self.refresh(select)

    # ------------------------------------------------------------------ actions
    def _move(self, step: int) -> None:
        k = self.current()
        target = k + step
        if 0 <= target < len(self.graph.layers):
            self._do(lambda: layout.move_layer(self.graph, k, step), target)

    def _add(self) -> None:
        like = self.graph.layers[self.current()] if self.graph.layers else None
        self._do(lambda: layout.add_layers(self.graph, 1, like), len(self.graph.layers))

    def _delete(self) -> None:
        k = self.current()
        self.gwin.selected = ("graph",)
        self._do(lambda: edit.remove_layer(self.graph, k), k - 1)

    def arrange(self) -> None:
        self._do(lambda: layout.arrange(self.graph, self.rows.value(), self.cols.value(),
                                        self.hspace.value(), self.wspace.value()))

    def apply_link(self) -> None:
        k = self.current()
        master = self.master.currentData()
        self._do(lambda: layout.link_layer(self.graph, k, master, self.link_x.isChecked(),
                                           self.link_y.isChecked()), k)

    def apply_rect(self) -> None:
        k = self.current()

        def fn():
            layer = self.graph.layers[k]
            layer.rect = [sb.value() for sb in self.rect] if self.make_inset.isChecked() \
                else None
            layout.relabel(self.graph)
        self._do(fn, k)
