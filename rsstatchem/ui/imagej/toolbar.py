"""The row of tool icons in the ImageJ bar (``ij/gui/Toolbar.java`` behaviour).

Contract: ``ToolBar.current`` (tool id), ``set_tool(id)``, signal ``tool_changed(int)``; tool
ids are ImageJ's ``ij.gui.Toolbar`` constants below. The straight-line button reports LINE,
POLYLINE or FREELINE depending on its variant, like ImageJ's ``getToolId()``.

Right-clicking (or long-pressing) a tool with variants shows them: Rectangle / Rounded
Rectangle / Rotated Rectangle; Oval / Elliptical / Selection Brush; Straight / Segmented /
Freehand Line / Arrow; Point / Multi-point. Double-clicking a tool opens its options
(``options_requested(tool)``; the bar maps it to Rounded Rect Tool..., Line Width...,
Arrow Tool..., Point Tool..., Wand Tool..., Fonts..., the Color Picker, ...). The colour
picker button shows the foreground/background colours.
"""
from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import QButtonGroup, QHBoxLayout, QMenu, QToolButton, QWidget

from rsstatchem.ui.imagej.icons import tool_icon

RECTANGLE, OVAL, POLYGON, FREEROI, LINE, POLYLINE, FREELINE, POINT, WAND, TEXT = range(10)
MAGNIFIER, HAND, DROPPER, ANGLE = 11, 12, 13, 14
# (id, tooltip) in ImageJ's left-to-right order
TOOLS = ((RECTANGLE, "Rectangle"), (OVAL, "Oval"), (POLYGON, "Polygon"),
         (FREEROI, "Freehand"), (LINE, "Straight Line"), (ANGLE, "Angle"),
         (POINT, "Point"), (WAND, "Wand (tracing)"), (TEXT, "Text"),
         (MAGNIFIER, "Magnifying glass"), (HAND, "Scrolling"), (DROPPER, "Color picker"))
VARIANTS = {
    RECTANGLE: ("Rectangle", "Rounded Rectangle", "Rotated Rectangle"),
    OVAL: ("Oval selections", "Elliptical selections", "Selection Brush Tool"),
    LINE: ("Straight Line", "Segmented Line", "Freehand Line", "Arrow tool"),
    POINT: ("Point Tool", "Multi-point Tool"),
}
TOOL_NAMES = {"rectangle": RECTANGLE, "roundrect": RECTANGLE, "rotrect": RECTANGLE,
              "oval": OVAL, "ellipse": OVAL, "brush": OVAL, "polygon": POLYGON,
              "freehand": FREEROI, "line": LINE, "polyline": POLYLINE, "freeline": FREELINE,
              "arrow": LINE, "angle": ANGLE, "point": POINT, "multipoint": POINT,
              "wand": WAND, "text": TEXT, "zoom": MAGNIFIER, "magnifier": MAGNIFIER,
              "hand": HAND, "dropper": DROPPER, "picker": DROPPER}
_NAME_VARIANT = {"roundrect": "Rounded Rectangle", "rotrect": "Rotated Rectangle",
                 "ellipse": "Elliptical selections", "brush": "Selection Brush Tool",
                 "polyline": "Segmented Line", "freeline": "Freehand Line",
                 "arrow": "Arrow tool", "multipoint": "Multi-point Tool",
                 "rectangle": "Rectangle", "oval": "Oval selections", "line": "Straight Line",
                 "point": "Point Tool"}


class _ToolButton(QToolButton):
    double_clicked = Signal()
    variant_menu = Signal()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        self.double_clicked.emit()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.RightButton:
            self.variant_menu.emit()
            return
        super().mousePressEvent(event)


class ToolBar(QWidget):
    tool_changed = Signal(int)
    options_requested = Signal(int)
    variant_changed = Signal(int, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(1)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.buttons: dict[int, QToolButton] = {}
        self.variant = {tool: names[0] for tool, names in VARIANTS.items()}
        self.foreground = "#ffffff"
        self.background = "#000000"
        for tool, name in TOOLS:
            b = _ToolButton(self)
            b.setCheckable(True)
            b.setIconSize(QSize(22, 22))
            b.setAutoRaise(True)
            b.setToolTip(name + ("  (right-click for variants)" if tool in VARIANTS else ""))
            b.double_clicked.connect(lambda t=tool: self.options_requested.emit(t))
            b.variant_menu.connect(lambda t=tool: self.show_variants(t))
            self.group.addButton(b, tool)
            self.buttons[tool] = b
            lay.addWidget(b)
        lay.addStretch(1)
        self._button_tool = RECTANGLE
        self.buttons[RECTANGLE].setChecked(True)
        self.group.idClicked.connect(self._clicked)
        self.update_icons()

    # ------------------------------------------------------------------ state
    @property
    def current(self) -> int:
        t = self._button_tool
        if t == LINE:
            v = self.variant[LINE]
            return POLYLINE if v == "Segmented Line" else FREELINE if v == "Freehand Line" \
                else LINE
        return t

    @property
    def multi_point(self) -> bool:
        return self.variant[POINT] == "Multi-point Tool"

    def current_variant(self) -> str:
        return self.variant.get(self._button_tool, "")

    def tool_name(self) -> str:
        names = {RECTANGLE: "rectangle", OVAL: "oval", POLYGON: "polygon",
                 FREEROI: "freehand", LINE: "line", POLYLINE: "polyline",
                 FREELINE: "freeline", ANGLE: "angle", POINT: "point", WAND: "wand",
                 TEXT: "text", MAGNIFIER: "zoom", HAND: "hand", DROPPER: "dropper"}
        v = self.current_variant()
        if v and v != VARIANTS[self._button_tool][0]:
            return next(k for k, n in _NAME_VARIANT.items() if n == v)
        return names.get(self.current, "rectangle")

    def _clicked(self, tool: int) -> None:
        self.set_tool(tool)

    def set_tool(self, tool) -> None:
        """Select by id or by ImageJ ``setTool`` name ("rectangle", "polyline",
        "multipoint", ...)."""
        if isinstance(tool, str):
            name = tool.lower()
            if name not in TOOL_NAMES:
                return
            if name in _NAME_VARIANT:
                self.variant[TOOL_NAMES[name] if TOOL_NAMES[name] in VARIANTS else LINE] = \
                    _NAME_VARIANT[name]
            tool = TOOL_NAMES[name]
        if tool in (POLYLINE, FREELINE):
            self.variant[LINE] = "Segmented Line" if tool == POLYLINE else "Freehand Line"
            tool = LINE
        if tool in self.buttons:
            self.buttons[tool].setChecked(True)
        self._button_tool = tool
        self.update_icons()
        self.tool_changed.emit(self.current)

    def set_variant(self, tool: int, variant: str) -> None:
        if variant not in VARIANTS.get(tool, ()):
            return
        self.variant[tool] = variant
        self.update_icons()
        self.variant_changed.emit(tool, variant)
        self.set_tool(tool)

    def show_variants(self, tool: int, exec_menu: bool = True):
        if tool not in VARIANTS:
            return None
        menu = QMenu(self)
        for name in VARIANTS[tool]:
            act = menu.addAction(tool_icon(tool, name), name)
            act.setCheckable(True)
            act.setChecked(self.variant[tool] == name)
            act.triggered.connect(lambda _=False, t=tool, n=name: self.set_variant(t, n))
        if exec_menu:
            b = self.buttons[tool]
            menu.exec(b.mapToGlobal(b.rect().bottomLeft()))
        return menu

    def set_colors(self, foreground: str, background: str) -> None:
        self.foreground, self.background = foreground, background
        self.update_icons()

    def update_icons(self) -> None:
        for tool, b in self.buttons.items():
            b.setIcon(tool_icon(tool, self.variant.get(tool, ""), 22, self.foreground,
                                self.background))
