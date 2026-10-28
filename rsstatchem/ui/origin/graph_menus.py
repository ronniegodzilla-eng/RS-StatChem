"""WP-4 menus for the Origin workspace: new Plot menu types, Graph menu (layers, axes,
reference lines, error bars, function plots, gadgets, formats, themes), File (layout page,
export all graphs, print) and Tools (digitizer).

``install(main)`` adds everything to ``main.menus[...]``; it never edits window.py. It only
calls public main-window methods (``plot``, ``new_graph``, ``new_book``,
``active_graph_window``, ``graph_windows``, ``message``, ``result``, ``_warn``, ``_act``).
"""
from __future__ import annotations

import copy

import numpy as np
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
                               QFormLayout, QInputDialog, QLineEdit, QMenu, QVBoxLayout)

from rsstatchem.core.graph import layout, themes
from rsstatchem.core.graph.model import Column, RefLine
from rsstatchem.core.graph.scales import LABELS as SCALE_LABELS
from rsstatchem.core.graph.scales import FormulaError, compile_formula
from rsstatchem.ui.origin import gadgets_ui
from rsstatchem.ui.origin.icons import icon
from rsstatchem.ui.origin.plotting import PLOT_ICONS, PLOT_TYPES, PlotError, plot_spec

# Plot menu additions: existing submenu title -> (icon, kinds); None = separator
PLOT_ADDITIONS = [
    ("Basic 2D", "g_line", ("step", "spline", "lollipop", None, "xyerror", "error_band",
                            "bubble_color")),
    ("Bar, Pie, Area", "g_column", (None, "bar_percent", "floating_column", None,
                                    "area_percent", "fill_area")),
    ("Multi-Panel/Axis", "g_double_y", (None, "panel_v2", "panel_h2", "panel_4",
                                        "panel_9")),
    ("Statistical", "g_histogram", (None, "violin", "box_half", "meanbar", "interval", None,
                                    "hist_kde", "hist_rug", "kde", "ecdf", None, "qq",
                                    "probability", None, "kde2d", "scatter_matrix",
                                    "marginal", "bland_altman", "pareto", None,
                                    "control_xbar", "control_imr")),
    ("3D", "g_3d", (None, "bar3d", "line3d", "waterfall3d")),
    ("Specialized", "g_polar", (None, "vector", "radar", "windrose")),
]
LINKED_PRESETS = [("Arrhenius: T (°C) on top of 1000/T (K⁻¹)", "1000/x - 273.15", "T (°C)"),
                  ("Arrhenius: 1000/T (K⁻¹) on top of T (°C)", "1000/(x + 273.15)",
                   "1000/T (K$^{-1}$)"),
                  ("Wavelength (nm) on top of wavenumber (cm⁻¹)", "1E7/x", "Wavelength (nm)"),
                  ("Wavenumber (cm⁻¹) on top of wavelength (nm)", "1E7/x",
                   "Wavenumber (cm$^{-1}$)"),
                  ("Energy (eV) on top of wavelength (nm)", "1239.84/x", "Energy (eV)"),
                  ("Same scale (mirror with labels)", "", "")]
GADGET_ORDER = ("statistics", "integrate", "fft", "differentiate", "rise_time", "intersect",
                "cursor")


def _clean(text: str) -> str:
    return text.replace("&", "").strip()


def find_submenu(menu: QMenu, title: str) -> QMenu | None:
    for act in menu.actions():
        sub = act.menu()
        if sub is not None and _clean(sub.title()) == _clean(title):
            return sub
    return None


def _gwin(main):
    gwin = main.active_graph_window()
    if gwin is None:
        main._warn("Activate a graph window first.", "Graph")
    return gwin


# ====================================================================== install
def install(main) -> None:
    """Add the WP-4 graph features to the Origin main window's menus (idempotent: a second
    call on the same window does nothing)."""
    if getattr(main, "_graph_menus_installed", False):
        return
    main._graph_menus_installed = True
    main.wp4_actions = {}
    _install_plot(main)
    _install_graph(main)
    _install_file_tools(main)
    main.menus["format"].aboutToShow.connect(lambda: _fill_format(main))
    main.graph_clipboard = None


def plot_action(main, kind: str):
    text = PLOT_TYPES[kind][0]
    act = main._act(text, lambda k=kind: main.plot(k), None, PLOT_ICONS.get(kind, "g_line"),
                    text)
    main.wp4_actions[f"plot:{kind}"] = act
    return act


def _install_plot(main) -> None:
    p = main.menus["plot"]
    for title, ic, kinds in PLOT_ADDITIONS:
        sub = find_submenu(p, title)
        if sub is None:
            sub = p.addMenu(icon(ic), title)
        for k in kinds:
            if k is None:
                sub.addSeparator()
            elif k in PLOT_TYPES:
                sub.addAction(plot_action(main, k))
    p.addSeparator()
    act = main._act("Plot &Setup...", lambda: plot_setup(main), None, "g_layers",
                    "Plot Setup: choose the plot type and the columns for each role")
    main.wp4_actions["plot_setup"] = act
    p.addAction(act)


def _add(main, menu: QMenu, key: str, text: str, slot, ic: str | None = None,
         tip: str | None = None):
    act = main._act(text, slot, None, ic, tip)
    main.wp4_actions[key] = act
    menu.addAction(act)
    return act


def _install_graph(main) -> None:
    g = main.menus["graph"]
    g.addSeparator()
    _add(main, g, "layer_manager", "Layer &Management...", lambda: layer_manager(main),
         "g_layers")
    panels = g.addMenu(icon("g_panel4"), "&Panel Layout")
    for key, text in (("v2", "2 Panels Vertical"), ("h2", "2 Panels Horizontal"),
                      ("4", "4 Panels (2 × 2)"), ("6", "6 Panels (2 × 3)"),
                      ("9", "9 Panels (3 × 3)")):
        _add(main, panels, f"panel:{key}", text, lambda k=key: panel_preset(main, k))
    _add(main, g, "merge", "Merge Graph &Windows...", lambda: _layout_view().
         merge_graph_windows(main))
    _add(main, g, "extract", "E&xtract Layers to Graphs", lambda: extract_layers(main))
    _add(main, g, "exchange_xy", "Exchange &X-Y Axes", lambda: exchange_xy(main))
    g.addSeparator()
    scale = g.addMenu("Axis &Scale")
    for which, name in (("x", "X"), ("y", "Y")):
        sub = scale.addMenu(f"{name} Axis")
        for s, label in SCALE_LABELS.items():
            _add(main, sub, f"scale:{which}:{s}", label,
                 lambda w=which, sc=s: set_scale(main, w, sc))
    _add(main, g, "breaks", "Axis &Breaks...", lambda: breaks_dialog(main), "g_break")
    top = g.addMenu("&Top Axis Linked by Formula")
    for k, (text, formula, title) in enumerate(LINKED_PRESETS):
        _add(main, top, f"top:{k}", text, lambda f=formula, t=title: set_top_axis(main, f, t))
    _add(main, top, "top:custom", "Custom Formula...", lambda: set_top_axis(main, None))
    _add(main, top, "top:off", "Remove Top Axis", lambda: set_top_axis(main, "", "", False))
    refs = g.addMenu(icon("g_ref"), "&Reference Lines")
    _add(main, refs, "ref:dialog", "Add Reference Line / Range...", lambda: ref_dialog(main))
    _add(main, refs, "ref:ymean", "Y Mean of Plot", lambda: add_ref(main, mode="mean"))
    _add(main, refs, "ref:ysd", "Y Mean ± 1 SD (shaded)",
         lambda: add_ref(main, mode="mean_sd", kind="band"))
    _add(main, refs, "ref:clear", "Remove All Reference Lines", lambda: clear_refs(main))
    g.addSeparator()
    _add(main, g, "error_bars", "Add &Error Bars...", lambda: error_bars_dialog(main))
    _add(main, g, "function", "Add &Function Plot...", lambda: function_dialog(main))
    g.addSeparator()
    _add(main, g, "copy_format", "&Copy Format", lambda: copy_format(main))
    _add(main, g, "paste_format", "&Paste Format", lambda: paste_format(main))
    g.addSeparator()
    gad = g.addMenu(icon("g_gadget"), "&Gadgets")
    for key in GADGET_ORDER:
        _add(main, gad, f"gadget:{key}", gadgets_ui.LABELS[key] + "...",
             lambda k=key: start_gadget(main, k))
    gad.addSeparator()
    _add(main, gad, "gadget:clear", "Remove ROI", lambda: _with_gwin(main,
                                                                    gadgets_ui.clear_roi))


def _install_file_tools(main) -> None:
    f = main.menus["file"]
    first = f.actions()[0] if f.actions() else None
    new = first.menu() if first is not None else None
    if new is not None:
        _add(main, new, "layout_page", "&Layout Page...", lambda: _layout_view().
             new_layout_page(main), "g_layout")
    exp = find_submenu(f, "Export")
    if exp is not None:
        _add(main, exp, "export_all", "All Graphs in &Project...",
             lambda: _layout_view().export_all_graphs(main))
    _add(main, f, "print", "&Print Graph...", lambda: _with_gwin(
        main, lambda gw: _layout_view().print_graph(main, gw)))
    t = main.menus["tools"]
    t.addSeparator()
    _add(main, t, "digitizer", "&Digitizer...", lambda: digitizer(main), "g_digitizer")


def _fill_format(main) -> None:
    gwin = main.active_graph_window()
    if gwin is None:
        return
    fm = main.menus["format"]
    fm.addSeparator()
    fm.addAction("Save Format as T&heme...", lambda: save_theme(main))
    user = themes.list_user_themes()
    if user:
        sub = fm.addMenu("Apply &User Theme")
        for name in user:
            sub.addAction(name, lambda n=name: apply_user_theme(main, n))


def _layout_view():
    from rsstatchem.ui.origin import layout_view
    return layout_view


def _with_gwin(main, fn):
    gwin = _gwin(main)
    if gwin is not None:
        return fn(gwin)
    return None


# ====================================================================== plot setup
def plot_setup(main):
    bw = main.active_book_window()
    ws = bw.worksheet() if bw is not None else None
    if ws is None:
        main._warn("Activate a workbook to choose columns in Plot Setup.", "Plot Setup")
        return None
    from rsstatchem.ui.graph.setup_dialog import PlotSetupDialog
    from rsstatchem.ui.origin.plotting import plot_dataset
    dlg = PlotSetupDialog(plot_dataset(ws), main)
    if not dlg.exec():
        return None
    try:
        graph = plot_spec(bw.book, ws, dlg.spec(), main.project.unique_name("Graph"))
    except PlotError as exc:
        main._warn(str(exc), "Plot Setup")
        return None
    return main.new_graph(graph, folder=main.project.root.find_window(bw.book.name))


# ====================================================================== layers
def layer_manager(main):
    gwin = _gwin(main)
    if gwin is None:
        return None
    from rsstatchem.ui.origin.layer_manager import LayerManagerDialog
    dlg = LayerManagerDialog(gwin, main)
    dlg.exec()
    return dlg


def panel_preset(main, key: str) -> None:
    gwin = _gwin(main)
    if gwin is None:
        return
    rows, cols = layout.PANEL_PRESETS[key]

    def fn():
        g = gwin.graph
        panels = layout.gridded(g)
        if len(panels) == 1 and len(panels[0].plots) > 1:
            layout.split_to_panels(g, rows, cols)
            return
        missing = rows * cols - len(panels)
        if missing > 0:
            for _ in range(missing):
                g.layers.append(layout.styled_empty_layer(panels[0] if panels else
                                                          g.layers[0]))
        layout.arrange(g, rows, cols)
    gwin.mutate(fn)


def extract_layers(main) -> list:
    gwin = _gwin(main)
    if gwin is None:
        return []
    out = []
    for g in layout.extract_layers(gwin.graph):
        g.title = main.project.unique_name(g.title.split("_L")[0] + "_L")
        g.meta["links"] = copy.deepcopy(gwin.graph.meta.get("links", {}))
        g.prune_data()
        out.append(main.new_graph(g, name=g.title))
    main.message(f"Extracted {len(out)} layers of {gwin.graph.title}")
    return out


def exchange_xy(main) -> None:
    gwin = _gwin(main)
    if gwin is None:
        return
    layer = gwin.active_layer()
    gwin.mutate(lambda: layout.exchange_xy(gwin.graph, layer))


# ====================================================================== axes
def _axis_target(gwin, which: str):
    return gwin.active_layer().axis(which)


def set_scale(main, which: str, scale: str) -> None:
    gwin = _gwin(main)
    if gwin is None:
        return
    axis = _axis_target(gwin, which)

    def fn():
        axis.scale = scale
        axis.min = axis.max = None
        axis.major_step = None
        if scale in ("probability", "probit", "logit", "weibull"):
            axis.grid_major = True
    gwin.mutate(fn, rebuild_tree=False)


class _FormDialog(QDialog):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.form = QFormLayout()
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addLayout(self.form)
        lay.addWidget(buttons)

    def combo(self, label, items, current=None):
        w = QComboBox()
        for value, text in items:
            w.addItem(text, value)
        if current is not None:
            w.setCurrentIndex(max(0, w.findData(current)))
        self.form.addRow(label, w)
        return w

    def number(self, label, value=0.0, lo=-1e12, hi=1e12, dec=6):
        w = QDoubleSpinBox()
        w.setRange(lo, hi)
        w.setDecimals(dec)
        w.setValue(value)
        self.form.addRow(label, w)
        return w

    def text(self, label, value=""):
        w = QLineEdit(value)
        self.form.addRow(label, w)
        return w


def breaks_dialog(main, which: str | None = None, text: str | None = None,
                  gap: float | None = None) -> bool:
    """Axis Breaks: "from-to@position%" items separated by ";" (empty removes them)."""
    from rsstatchem.ui.graph.properties import breaks_to_text, text_to_breaks
    gwin = _gwin(main)
    if gwin is None:
        return False
    layer = gwin.active_layer()
    if layer.kind != "cartesian":
        main._warn("Axis breaks need a 2D layer.", "Axis Breaks")
        return False
    if which is None:
        dlg = _FormDialog("Axis Breaks", main)
        w_axis = dlg.combo("Axis", [("x", "X (bottom)"), ("y", "Y (left)"),
                                    ("y2", "Right Y")], "x")
        w_text = dlg.text("Breaks (from-to@position%; …)", breaks_to_text(layer.x.breaks))
        w_axis.currentIndexChanged.connect(lambda: w_text.setText(
            breaks_to_text(layer.axis(w_axis.currentData()).breaks)))
        w_gap = dlg.number("Gap (% of axis)", layer.x.break_gap, 0.5, 20, 1)
        if not dlg.exec():
            return False
        which, text, gap = w_axis.currentData(), w_text.text(), w_gap.value()
    try:
        breaks = text_to_breaks(text or "")
    except ValueError as exc:
        main._warn(str(exc), "Axis Breaks")
        return False
    axis = layer.axis(which)

    def fn():
        axis.breaks = breaks
        if gap is not None:
            axis.break_gap = float(gap)
    gwin.mutate(fn, rebuild_tree=False)
    return True


def set_top_axis(main, formula: str | None, title: str | None = None,
                 show: bool = True) -> bool:
    gwin = _gwin(main)
    if gwin is None:
        return False
    layer = gwin.active_layer()
    if layer.kind != "cartesian":
        main._warn("A linked top axis needs a 2D layer.", "Top Axis")
        return False
    if formula is None:
        formula, ok = QInputDialog.getText(main, "Top Axis Linked by Formula",
                                           "Top X = f(x) of the bottom X, e.g. "
                                           "1000/(x+273.15), 1E7/x:", text=layer.x2_formula)
        if not ok:
            return False
        title = None
    if formula:
        try:
            compile_formula(formula)
        except FormulaError as exc:
            main._warn(str(exc), "Top Axis")
            return False

    def fn():
        layer.show_x2 = show
        layer.x2_formula = formula or ""
        if title is not None:
            layer.x2.title = title
        layer.x.mirror_ticks = False if show else layer.x.mirror_ticks
    gwin.mutate(fn)
    return True


# ====================================================================== reference lines
def add_ref(main, **kw) -> RefLine | None:
    gwin = _gwin(main)
    if gwin is None:
        return None
    layer = gwin.active_layer()
    if layer.kind != "cartesian":
        main._warn("Reference lines need a 2D layer.", "Reference Lines")
        return None
    sel = gwin.selected
    if "plot" not in kw and sel and sel[0] == "plot":
        kw["plot"] = sel[2]
    ref = RefLine(**kw)
    gwin.mutate(lambda: layer.refs.append(ref))
    return ref


def clear_refs(main) -> None:
    gwin = _gwin(main)
    if gwin is not None:
        layer = gwin.active_layer()
        gwin.mutate(lambda: setattr(layer, "refs", []))


def ref_dialog(main):
    gwin = _gwin(main)
    if gwin is None:
        return None
    dlg = _FormDialog("Add Reference Line", main)
    axis = dlg.combo("Axis", [("y", "Y (horizontal line)"), ("x", "X (vertical line)"),
                              ("y2", "Right Y")])
    kind = dlg.combo("Type", [("line", "Line"), ("band", "Shaded range")])
    mode = dlg.combo("At", [("value", "Value(s)"), ("mean", "Mean of plot"),
                            ("median", "Median of plot"), ("mean_sd", "Mean ± k·SD"),
                            ("mean_se", "Mean ± k·SE"), ("mean_ci", "Mean ± 95% CI")])
    v1 = dlg.number("Value / From")
    v2 = dlg.number("To (range)", 1.0)
    k = dlg.number("k", 1.0, 0, 10, 2)
    label = dlg.text("Label")
    if not dlg.exec():
        return None
    return add_ref(main, axis=axis.currentData(), kind=kind.currentData(),
                   mode=mode.currentData(), value=v1.value(), value2=v2.value(), k=k.value(),
                   label=label.text())


# ====================================================================== add data
def _active_plot(gwin):
    layer = gwin.active_layer()
    sel = gwin.selected
    if sel and sel[0] == "plot":
        p = gwin.resolve(sel)
        if p is not None:
            return p
    return next((p for p in layer.plots if p.y), None)


def add_error_bars(gwin, mode: str = "percent", value: float = 5.0, plot=None) -> str:
    """Origin's Add Error Bars: ± value % of Y, a constant, or the SD/SE of the Y data."""
    p = plot or _active_plot(gwin)
    if p is None:
        raise ValueError("The layer has no plot.")
    y = gwin.graph.values(p.y)
    if y is None or y.dtype == object:
        raise ValueError("The plot has no numeric Y data.")
    if mode == "percent":
        err = np.abs(y) * value / 100.0
    elif mode == "constant":
        err = np.full(len(y), abs(value))
    elif mode in ("sd", "se"):
        v = y[np.isfinite(y)]
        sd = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        err = np.full(len(y), sd if mode == "sd" else sd / np.sqrt(max(1, len(v))))
    else:
        raise ValueError(f"unknown error mode {mode!r}")
    key = gwin.graph.add_column(Column.from_array(f"{p.name} error", err,
                                                  f"Error ({mode})"), f"{p.y} error ({mode})")

    def fn():
        p.yerr = key
        if p.type not in ("line", "scatter", "line_symbol", "column", "bar"):
            p.type = "scatter" if p.type not in ("area",) else p.type
    gwin.mutate(fn)
    return key


def error_bars_dialog(main):
    gwin = _gwin(main)
    if gwin is None:
        return None
    dlg = _FormDialog("Add Error Bars", main)
    mode = dlg.combo("Error", [("percent", "Percent of Y"), ("constant", "Constant"),
                               ("sd", "SD of the Y data"), ("se", "SE of the Y data")])
    value = dlg.number("Value", 5.0, 0, 1e12, 4)
    if not dlg.exec():
        return None
    try:
        return add_error_bars(gwin, mode.currentData(), value.value())
    except ValueError as exc:
        main._warn(str(exc), "Add Error Bars")
        return None


def add_function_plot(gwin, formula: str, x0: float, x1: float, n: int = 400,
                      name: str | None = None):
    """Add y = f(x) evaluated on [x0, x1] to the active layer (Add Function Plot)."""
    from rsstatchem.ui.origin.plotting import add_xy_plot
    f = compile_formula(formula)
    xs = np.linspace(float(x0), float(x1), int(n))
    ys = f(xs)
    ys[~np.isfinite(ys)] = np.nan
    if not np.isfinite(ys).any():
        raise FormulaError(f"“{formula}” has no finite values on [{x0:g}, {x1:g}].")
    holder = {}

    def fn():
        holder["p"] = add_xy_plot(gwin.graph, gwin.active_index(), xs, ys,
                                  name or f"y = {formula}", "line")
        holder["p"].options["function"] = {"formula": formula, "x0": x0, "x1": x1, "n": n}
    gwin.mutate(fn)
    return holder.get("p")


def function_dialog(main):
    gwin = _gwin(main)
    if gwin is None:
        return None
    layer = gwin.active_layer()
    lo, hi = 0.0, 10.0
    info = gwin.rmap.axes.get(gwin.active_index()) if gwin.rmap else None
    if info is not None and layer.kind == "cartesian":
        lo, hi = sorted(info["ax"].get_xlim())
    dlg = _FormDialog("Add Function Plot", main)
    formula = dlg.text("y =", "sin(x)")
    a = dlg.number("X from", lo)
    b = dlg.number("X to", hi)
    if not dlg.exec():
        return None
    try:
        return add_function_plot(gwin, formula.text(), a.value(), b.value())
    except FormulaError as exc:
        main._warn(str(exc), "Add Function Plot")
        return None


# ====================================================================== formats / themes
def copy_format(main) -> dict | None:
    gwin = _gwin(main)
    if gwin is None:
        return None
    obj = gwin.resolve(gwin.selected) if gwin.selected[0] == "plot" else gwin.active_layer()
    main.graph_clipboard = layout.format_of(obj)
    main.show_status(f"Copied the format of the {main.graph_clipboard['kind']}.")
    return main.graph_clipboard


def paste_format(main) -> bool:
    gwin = _gwin(main)
    fmt = getattr(main, "graph_clipboard", None)
    if gwin is None or not fmt:
        return False
    obj = gwin.resolve(gwin.selected) if gwin.selected[0] == "plot" else gwin.active_layer()
    gwin.mutate(lambda: layout.apply_format(obj, fmt))
    return True


def save_theme(main, name: str | None = None):
    gwin = _gwin(main)
    if gwin is None:
        return None
    if name is None:
        name, ok = QInputDialog.getText(main, "Save Format as Theme", "Theme name:")
        if not ok or not name.strip():
            return None
    path = themes.save_user_theme(name.strip(), gwin.graph)
    main.message(f"Graph theme “{name.strip()}” saved ({path.name}).")
    return path


def apply_user_theme(main, name: str) -> None:
    gwin = _gwin(main)
    if gwin is not None:
        gwin.mutate(lambda: themes.apply_user_theme(gwin.graph, name), rebuild_tree=False)


# ====================================================================== gadgets / tools
def start_gadget(main, key: str):
    gwin = _gwin(main)
    if gwin is None:
        return None
    return gadgets_ui.start(main, gwin, key)


def digitizer(main):
    from rsstatchem.ui.origin.digitizer import DigitizerDialog
    dlg = DigitizerDialog(main)
    dlg.show()
    main.digitizer_dialog = dlg
    return dlg
