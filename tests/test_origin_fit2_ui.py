"""WP-1 in the Origin-style window: the fitting_menus hook (idempotent install, menu
entries), NLFit with constraints / global fit / replica through the dialog settings, the
NLFit and Peak Analyzer dialogs (page flow, anchors, peaks, fit, Finish), Peak Analyzer
reports with Recalculate after a data change, Change Parameters, masked points left out of
fits, the Fitting Function Organizer (per-user JSON), comparison and ranking reports,
errors-in-variables, implicit and surface fits, simulation, calibration and the spectroscopy
tools."""
from __future__ import annotations

import json

import numpy as np
import pytest

from rsstatchem.core.analysis import fitting as fit
from rsstatchem.core.analysis import functions as fl
from rsstatchem.ui.origin import fitting_menus as fmn
from rsstatchem.ui.origin.model import MatrixBook, ReportSheet, Workbook, Worksheet


@pytest.fixture
def ctx(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("RSSTATCHEM_HOME", str(tmp_path))
    from rsstatchem.app import AppContext

    context = AppContext(qapp)
    monkeypatch.setattr(context.app, "quit", lambda: None)
    return context


@pytest.fixture
def win(ctx, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    def boom(*args, **kwargs):
        raise AssertionError("unexpected modal dialog")
    for name in ("question", "warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name, boom)
    from rsstatchem.ui.origin import open_workspace
    w = open_workspace(ctx)
    w.quiet = True
    yield w
    w.modified = False
    w.close()


G = fl.LIBRARY["Gaussian"]
RNG = np.random.default_rng(3)


def _fill(win, columns: dict[str, tuple[str, list]]):
    """Put columns {long name: (designation, values)} into the active worksheet."""
    bw = win.active_book_window()
    bw.show_sheet(bw.book.sheets[0])
    ws = bw.worksheet()
    n = max(len(v) for _, v in columns.values())
    ws.set_rows(n)
    while len(ws.columns) < len(columns):
        ws.add_column("Y")
    for j, (name, (des, vals)) in enumerate(columns.items()):
        c = ws.columns[j]
        c.designation, c.long_name = des, name
        ws.set_column_values(j, list(map(float, vals)))
    bw.refresh()
    return bw, ws


def _spectrum_book(win):
    x = np.linspace(400, 700, 301)
    y = 0.2 + 0.0005 * (x - 400) + G(x, 0, 480, 20, 30) + G(x, 0, 520, 25, 20) + \
        G(x, 0, 610, 15, 12) + RNG.normal(0, 0.003, x.size)
    return _fill(win, {"Wavelength": ("X", x), "Absorbance": ("Y", y)})


def test_install_is_idempotent_and_adds_menus(win):
    acts = dict(win.wp1_actions)
    assert {"nlfit", "peak_analyzer", "compare_models", "rank_models", "surface", "york",
            "odr", "implicit", "tauc", "units", "kubelka_munk", "organizer",
            "calibration"} <= set(acts)
    n_before = len(win.menus["analysis"].actions())
    fmn.install(win)
    assert win.wp1_actions == acts and len(win.menus["analysis"].actions()) == n_before
    titles = [a.text().replace("&", "") for a in win.menus["analysis"].actions()]
    assert "Fitting" in titles and "Peaks and Baseline" in titles and "Spectroscopy" in titles
    fitting = next(a.menu() for a in win.menus["analysis"].actions()
                   if a.text().replace("&", "") == "Fitting")
    assert sum(a.objectName().startswith("wp1_") for a in fitting.actions()) >= 12


def test_nlfit_with_constraints_global_and_replica(win):
    x = np.linspace(0, 10, 40)
    bw, ws = _fill(win, {"t": ("X", x),
                         "A": ("Y", 1 + 3 * np.exp(-x / 2) + RNG.normal(0, 0.01, 40)),
                         "B": ("Y", 0.5 + 2 * np.exp(-x / 2) + RNG.normal(0, 0.01, 40))})
    choices, cur, _ = win.analysis_inputs()
    out = fmn.run_nlfit(win, {"inputs": [choices[0]], "model": "ExpDec1",
                              "multi_mode": "independent", "constraints": ["y0 >= 1.2"],
                              "residual_extra": True})
    assert out.result.params["y0"] == pytest.approx(1.2, abs=1e-8)
    titles = [i["title"] for i in out.report.items]
    assert {"Constraints", "Covariance", "Correlation", "Dependency", "Information Criteria",
            "Derived Parameters", "Normal Probability Plot of the Residual"} <= set(titles)
    assert out.report.op["settings"]["constraints"] == ["y0 >= 1.2"]
    # Recalculate keeps the constraint
    ws.set_column_values(1, list(1 + 3 * np.exp(-x / 2.2)))
    assert win.recalculate_report(bw.book, out.report)
    assert out.result.params["y0"] >= 1.2 - 1e-9
    g = fmn.run_nlfit(win, {"inputs": choices[:2], "model": "ExpDec1", "multi_mode": "global",
                            "shared": ["t1"]})
    assert g.report.name.startswith("FitNLGlobal")
    assert g.result.combined.params["t1"] == pytest.approx(2.1, abs=0.12)
    assert g.report.table("Parameters") is not None
    r = fmn.run_nlfit(win, {"inputs": [choices[0]], "model": "ExpDec1",
                            "multi_mode": "replica", "replicas": 2})
    assert r.report.name.startswith("FitNLReplica")


def test_masked_points_are_left_out_of_fits(win):
    x = np.linspace(0, 10, 30)
    y = 2 * x + 1 + RNG.normal(0, 0.05, 30)
    y[[4, 17]] += 30.0                          # outliers, masked below
    bw, ws = _fill(win, {"x": ("X", x), "y": ("Y", y)})
    ws.set_mask(1, [4, 17])
    choices, _, _ = win.analysis_inputs()
    out = fmn.run_nlfit(win, {"inputs": [choices[0]], "model": "Line",
                              "multi_mode": "independent"})
    keep = np.ones(30, bool)
    keep[[4, 17]] = False
    ref = fit.fit_curve(x[keep], y[keep], "Line")
    assert out.result.n == 28
    assert out.result.values == pytest.approx(ref.values, rel=1e-12)
    assert out.result.errors == pytest.approx(ref.errors, rel=1e-10)


def test_nlfit_dialog_values(win):
    _spectrum_book(win)
    choices, cur, _ = win.analysis_inputs()
    from rsstatchem.ui.origin.nlfit_dialog import NLFitDialog
    d = NLFitDialog(choices, win, cur)
    d.set_function("ExpDecay1")
    assert d.params.rowCount() == 4
    d.set_param("t1", 50.0, lower=1.0)
    d.set_param("y0", share=True)
    d.constraints.setPlainText("A1 >= 0\n")
    v = d.values()
    assert v["model"] == "ExpDecay1" and v["params0"]["t1"] == 50.0
    assert v["fixed"] == {"x0": None} and v["bounds"]["t1"] == (1.0, None)
    assert v["shared"] == ["y0"] and v["constraints"] == ["A1 >= 0"]
    d.set_function("y = a*x + b")
    assert d.model() == "y = a*x + b" and d.params.rowCount() == 2
    d.check_inputs([])
    with pytest.raises(ValueError):
        d.values()


def test_peak_analyzer_dialog_and_report_recalculate(win, tmp_path):
    bw, ws = _spectrum_book(win)
    choices, cur, _ = win.analysis_inputs()
    from rsstatchem.ui.origin.peak_analyzer_dialog import PeakAnalyzerDialog
    dlg = PeakAnalyzerDialog(choices, win, cur, win.resolve)
    dlg.bmode.setCurrentIndex(dlg.bmode.findData("anchors"))
    assert dlg.sequence() == ["goal", "baseline", "anchors", "find", "fit", "finish"]
    for _ in range(4):
        dlg.go(1)
    assert dlg.page == "fit" and dlg.analysis.fit is not None
    assert [round(r["center"]) for r in dlg.analysis.fit.peaks] == [480, 520, 610]
    assert [r["area"] for r in dlg.analysis.fit.peaks] == pytest.approx([30, 20, 12], rel=0.03)
    dlg.go(1)
    assert dlg.page == "finish" and "Peaks: 3" in dlg.summary.toPlainText()
    dlg.show()
    dlg.grab().save(str(tmp_path / "pa.png"))
    v = dlg.values()
    dlg.close()
    out = fmn.peak_analyzer(win, values=v)
    rep = out.report
    assert rep.op["fit_kind"] == "peaks" and rep.table("Peak Properties") is not None
    assert len(rep.table("Peak Properties")["rows"]) == 3
    data = bw.book.sheet(out.curve.name)
    assert [c.long_name for c in data.columns][:4] == ["Independent Variable", "Baseline",
                                                       "Subtracted Y", "Cumulative Fit Peak"]
    # data change -> Recalculate re-runs the stored Peak Analyzer settings
    y = ws.columns[1].numeric(ws.n_rows) * 2
    ws.set_column_values(1, list(y))
    assert win.recalculate_report(bw.book, rep)
    areas = [float(r[4]) for r in rep.table("Peak Properties")["rows"]]
    assert areas == pytest.approx([60, 40, 24], rel=0.05)
    # Change Parameters reopens the Peak Analyzer (wrapped by install); here the integrate goal
    called = {}
    orig = fmn.peak_analyzer

    def fake(main, goal=None, settings=None, report=None, values=None):
        called["settings"] = settings
        return orig(main, values={"input": choices[cur], "settings": dict(settings, goal="fit")},
                    report=report)
    fmn.peak_analyzer = fake
    try:
        win.change_report_parameters(bw.book, rep)
    finally:
        fmn.peak_analyzer = orig
    assert called["settings"]["baseline_mode"] == "anchors"


def test_quick_peaks_batch_and_manual_peaks(win):
    bw, ws = _spectrum_book(win)
    choices, cur, _ = win.analysis_inputs()
    out = fmn.quick_peaks(win)
    assert out.report.table("Integration Result") is not None
    settings = {"goal": "fit", "baseline_mode": "line", "baseline_opts": {"n_end": 10},
                "manual_peaks": [{"x": 480.0, "y": 1.0, "fwhm": 20.0},
                                 {"x": 520.0, "y": 0.6, "fwhm": 20.0}],
                "fit_opts": {"shape": "gauss"}}
    out2 = fmn.peak_analyzer(win, values={"input": choices[cur], "settings": settings})
    assert out2 is not None, win.messages.toPlainText()
    assert len(out2.report.table("Peak Properties")["rows"]) == 2
    bw.show_sheet(ws)
    rep = fmn.batch_peaks_dialog(win, values={"settings": {"goal": "integrate",
                                                           "baseline_mode": "line"}})
    assert rep.table("Summary")["rows"]


def test_function_organizer_persists_user_functions(win, tmp_path):
    from rsstatchem.ui.origin import fit_organizer as org
    f = org.upsert({"name": "MyDecay", "category": "Kinetics Lab", "type": "curve",
                    "equation": "y = A*exp(-k*x) + c", "initial": {"A": 2, "k": 0.5, "c": 0},
                    "description": "first order"})
    path = tmp_path / "fitting_functions.json"
    assert path.exists() and json.loads(path.read_text())["functions"][0]["name"] == "MyDecay"
    assert fl.get_function("MyDecay") is f and "Kinetics Lab" in fl.CATEGORIES
    x = np.linspace(0, 10, 30)
    r = fit.fit_curve(x, 2 * np.exp(-0.5 * x) + 1, "MyDecay")
    assert r.params == pytest.approx({"A": 2, "k": 0.5, "c": 1})
    dlg = org.FunctionOrganizerDialog(win)
    assert dlg.tree.topLevelItemCount() == 1
    dlg.name.setText("Bad")
    dlg.equation.setPlainText("y = a*")
    assert not dlg.check()
    dlg.name.setText("Gauss")
    dlg.equation.setPlainText("y = a*x")
    assert not dlg.save()
    with pytest.raises(ValueError):
        org.parse_initial("a = x")
    org.delete("MyDecay", "Kinetics Lab")
    assert fl.get_function("MyDecay") is None and org.load_entries() == []


def test_comparison_errors_in_variables_and_implicit(win):
    x = np.linspace(0, 5, 40)
    y = 1 + 0.5 * x + 0.3 * x ** 2 + RNG.normal(0, 0.05, 40)
    bw, ws = _fill(win, {"x": ("X", x), "y": ("Y", y), "xerr": ("xEr", np.full(40, 0.05)),
                         "yerr": ("yEr", np.full(40, 0.05))})
    choices, cur, _ = win.analysis_inputs()
    rep = fmn.compare_models_dialog(win, {"inputs": [choices[0]], "models": ["Line", "Poly2"],
                                          "criterion": "aicc"})
    assert rep.table("F-test") is not None and "Poly2" in rep.table(
        "Akaike Information Criterion (AIC) Test")["notes"][0]
    rk = fmn.rank_models_dialog(win, {"inputs": [choices[0]], "models": None,
                                      "category": "Polynomial", "criterion": "bic"})
    assert rk.table("Rank Models")["rows"][0][0] == "1"
    york = fmn.odr_dialog(win, "york", {"input": choices[0], "sx": 0.0, "sy": 0.0})
    assert york.table("Parameters") is not None
    o = fmn.odr_dialog(win, "odr", {"input": choices[0], "model": "Poly2"})
    assert len(o.table("Parameters")["rows"]) == 3
    cd = fmn.compare_datasets_dialog(win, {"inputs": [choices[0], choices[0]],
                                           "models": ["Poly2"]})
    assert cd.table("F-test") is not None
    t = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    _fill(win, {"cx": ("X", 1 + 2 * np.cos(t)), "cy": ("Y", -1 + 2 * np.sin(t))})
    choices, _, _ = win.analysis_inputs()
    imp = fmn.implicit_dialog(win, {"input": choices[0], "model": "Circle", "equation": ""})
    vals = {r[0]: float(r[1]) for r in imp.table("Parameters")["rows"]}
    assert vals == pytest.approx({"xc": 1, "yc": -1, "r": 2}, abs=1e-8)


def test_surface_simulate_calibration_mlr_spectroscopy(win):
    xy = RNG.uniform(-2, 2, (120, 2))
    z = 1 + 2 * xy[:, 0] - xy[:, 1]
    bw, ws = _fill(win, {"x": ("X", xy[:, 0]), "y": ("Y", xy[:, 1]), "z": ("Z", z)})
    rep = fmn.surface_dialog(win, {"x": "A", "y": "B", "z": "C", "model": "Plane",
                                   "equation": ""})
    vals = {r[1]: float(r[2]) for r in rep.table("Parameters")["rows"]}
    assert vals == pytest.approx({"z0": 1, "a": 2, "b": -1}, abs=1e-9)
    mb = fmn.simulate_surface_dialog(win, {"model": "Gauss2D", "params":
                                           "z0 = 0, A = 1, xc = 0, w1 = 1, yc = 0, w2 = 1",
                                           "x1": -3, "x2": 3, "y1": -3, "y2": 3, "n": 21})
    assert isinstance(mb, MatrixBook) and mb.sheets[0].data.shape == (21, 21)
    assert win.active_matrix_window() is not None
    mrep = fmn.surface_dialog(win, {"model": "Gauss2D", "equation": ""})
    mvals = {r[1]: float(r[2]) for r in mrep.table("Parameters")["rows"]}
    assert mvals["A"] == pytest.approx(1.0, abs=1e-8)
    book = fmn.simulate_curve_dialog(win, {"model": "Gauss", "params":
                                           "y0 = 0, xc = 5, w = 1, A = 1",
                                           "x1": 0, "x2": 10, "n": 101})
    assert isinstance(book, Workbook) and book.sheets[0].n_rows == 101
    # calibration with unknowns, MLR, spectroscopy on a fresh book
    win.new_book()
    conc = np.array([0, 1, 2, 4, 8, 16.0])
    resp = 0.02 + 0.1 * conc + np.array([0.003, -0.002, 0.001, -0.003, 0.002, -0.001])
    bw, ws = _fill(win, {"c": ("X", conc), "r": ("Y", resp)})
    choices, _, _ = win.analysis_inputs()
    cal = fmn.calibration_dialog(win, {"input": choices[0], "weighting": "none",
                                       "sigma": "residual", "blank_sd": 0.0,
                                       "unknowns": "0.5, 1.0", "replicates": 2})
    assert len(cal.table("Unknown Samples")["rows"]) == 2
    lod = float(dict((r[0], r[1]) for r in cal.table("Detection Limits")["rows"])
                ["LOD = 3.3·σ/b"])
    assert lod > 0
    nm = np.linspace(300, 500, 201)
    E = 1239.84198 / nm
    A = np.sqrt(np.clip(E - 3.0, 0, None)) * 5 / E + 0.01
    win.new_book()
    _fill(win, {"nm": ("X", nm), "A": ("Y", A), "R": ("Y", np.linspace(10, 90, 201))})
    choices, _, _ = win.analysis_inputs()
    tr = fmn.tauc_dialog(win, {"input": choices[0], "x_unit": "nm", "transition": "direct",
                               "e1": 3.1, "e2": 3.6})
    eg = float(dict((r[0], r[1]) for r in tr.table("Band Gap")["rows"])["Band Gap Eg (eV)"])
    assert eg == pytest.approx(3.0, abs=0.01)
    idx = fmn.unit_conversion_dialog(win, {"input": choices[0], "from_unit": "nm",
                                           "to_unit": "cm-1"})
    ws = win.active_book_window().book.sheets[0]
    assert ws.columns[idx].numeric(3)[0] == pytest.approx(1e7 / 300)
    km = fmn.kubelka_munk(win, {"input": choices[1]})
    assert ws.columns[km].numeric(1)[0] == pytest.approx((1 - 0.1) ** 2 / 0.2)
    ab = fmn.absorbance_dialog(win, {"input": choices[0], "to": "transmittance"})
    assert ab is not None
    win.active_book_window().select_columns([0, 1])
    mr = fmn.mlr_dialog(win, {"y": "B", "x": "A C", "intercept": True})
    assert isinstance(mr, ReportSheet) and mr.table("Parameters") is not None


def test_selftest():
    fmn.selftest()


def test_quick_fit_and_quick_peaks_gadgets(win):
    from rsstatchem.core.graph import gadgets as gd
    from rsstatchem.ui.origin import gadgets_ui
    assert "quick_fit" in gd.GADGETS and "quick_peaks" in gd.GADGETS
    bw, ws = _spectrum_book(win)
    bw.select_columns([1])
    gwin = win.plot("line")
    out = gadgets_ui.compute(gwin, 0, "quick_peaks", 440, 660, win)
    assert out["Peaks"] == 3 and out["Peak1 X"] == pytest.approx(480, abs=2)
    n_plots = len(gwin.graph.layers[0].plots)
    tool = fmn.quick_fit_gadget(win, "Gaussian")
    res = tool.run(0, 580, 640)
    assert res["Function"] == "Gaussian" and res["xc"] == pytest.approx(610, abs=0.5)
    assert len(gwin.graph.layers[0].plots) == n_plots + 1
    assert "Quick Fit gadget" in win.results.toPlainText()
    tool2 = fmn.quick_peaks_gadget(win)
    assert tool2.run(0, 440, 560)["Peaks"] == 2
    win.menus["graph"].aboutToShow.emit()
    gad = next((a.menu() for a in win.menus["graph"].actions() if a.menu() is not None and
                a.menu().title().replace("&", "") == "Gadgets"), None)
    if gad is not None:
        assert any(a.objectName() == "wp1_gadget_quick_fit" for a in gad.actions())
