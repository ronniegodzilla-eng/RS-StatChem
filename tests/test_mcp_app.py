"""The connector on the whole application: menus, dialogs, tables and every module, through
the same engine the MCP server and the live bridge call. Calls come from a worker thread while
the Qt event loop runs in the main one, as in the real program."""
from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from rsstatchem.core.dataset import Dataset
from rsstatchem.io import files

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def ctx(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("RSSTATCHEM_HOME", str(tmp_path / "home"))
    from rsstatchem.app import AppContext
    from rsstatchem.ui.dialogs.base import VariablePickerDialog

    # dialogs remember their last settings per class: do not leak them into other tests
    monkeypatch.setattr(VariablePickerDialog, "_states", {})
    monkeypatch.setattr(VariablePickerDialog, "_memory", {})
    c = AppContext(qapp)
    monkeypatch.setattr(c, "check_for_updates", lambda manual=False: None)
    monkeypatch.setattr(qapp, "quit", lambda: None)
    c.app = qapp
    yield c
    for w in list(c.windows):
        doc = getattr(w, "doc", None)
        if doc is not None and hasattr(doc, "modified"):
            doc.modified = False
        w.hide()


@pytest.fixture
def engine(ctx):
    from rsstatchem.mcp.engine import Engine

    e = Engine(ctx)
    yield e
    e.close()


@pytest.fixture
def run(qapp, engine):
    """run(script): ``script(call)`` runs on a worker thread with Qt's loop turning here."""
    def go(script, timeout=180):
        from PySide6.QtCore import QEventLoop, QTimer

        box: dict = {}
        loop = QEventLoop()
        done = QTimer()
        done.setInterval(20)

        def call(name, **args):
            return engine.call(name, args)

        def work():
            try:
                box["result"] = script(call)
            except BaseException as exc:  # noqa: BLE001 - raised in the test below
                box["error"] = exc
            finally:             # a dialog left open would keep its own event loop turning
                from rsstatchem.mcp import driver

                for _ in range(4):
                    engine.gui.call(lambda: [d.reject() for d in driver.open_dialogs()], 30)

        thread = threading.Thread(target=work, daemon=True)
        done.timeout.connect(lambda: thread.is_alive() or loop.quit())
        QTimer.singleShot(timeout * 1000, loop.quit)
        thread.start()
        done.start()
        loop.exec()
        done.stop()
        assert not thread.is_alive(), "the script did not finish"
        if "error" in box:
            raise box["error"]
        return box.get("result")
    return go


def data(parts) -> dict:
    return json.loads(parts[0])


def text(parts) -> str:
    return "\n".join(p for p in parts if isinstance(p, str))


@pytest.fixture
def study(tmp_path):
    rng = np.random.default_rng(5)
    n = 40
    group = np.repeat([1.0, 2.0], n // 2)
    x = rng.normal(50, 8, n).round(2)
    y = (x + 4 * (group - 1) + rng.normal(0, 5, n)).round(2)
    ds = Dataset(pd.DataFrame({"group": group, "x": x, "y": y}))
    ds.var("group").value_labels = {1.0: "A", 2.0: "B"}
    ds.var("group").measure = "nominal"
    path = tmp_path / "study.rsd"
    files.save_project(ds, path)
    return path, ds


# ---------------------------------------------------------------------------- statistics
def test_menu_command_through_its_dialog(run, study):
    path, ds = study

    def script(call):
        call("open_data", path=str(path))
        found = data(call("find_commands", module="stats", query="2 independent samples"))
        assert any("Legacy Dialogs > 2 Independent Samples" in c for c in found["commands"])
        first = data(call("run_command", command="Legacy Dialogs > 2 Independent Samples"))
        assert first["status"].startswith("waiting")
        boxes = {b["box"] for b in first["dialog"]["variables"]["boxes"]}
        assert boxes == {"vars", "group"}
        with pytest.raises(ValueError, match="no variable 'nope'"):
            call("fill_dialog", variables={"vars": ["nope"]})
        with pytest.raises(ValueError, match="is waiting"):
            call("run_command", command="Analyze > Correlate > Bivariate")
        return call("fill_dialog", variables={"vars": ["y"], "Grouping Variable": "group"},
                    press="OK")

    out = run(script)
    assert data(out)["status"] == "done"
    a, b = ds.df.y[:20], ds.df.y[20:]
    u = stats.mannwhitneyu(a, b).statistic
    assert f"{min(u, 400 - u):.3f}" in text(out)


def test_one_call_with_sub_dialog_and_output_viewer(run, ctx, study, tmp_path):
    path, _ = study

    def script(call):
        call("open_data", path=str(path))
        out = call("run_command", command="Analyze > Compare Means > One-Way ANOVA",
                   variables={"Dependent List": ["y"], "Factor": "group"},
                   steps=[{"open": "Options...", "values": {"Welch": True}}],
                   press="OK", charts=True)
        assert "Robust Tests of Equality of Means" in text(out)
        assert data(out)["output"] == ["Oneway"]
        again = call("get_output")
        assert "Oneway" in text(again)
        for ext in ("rso", "docx", "xlsx", "html", "pdf"):
            target = tmp_path / f"result.{ext}"
            call("save_output", path=str(target))
            assert target.stat().st_size > 500, ext
        with pytest.raises(ValueError, match="already exists"):
            call("save_output", path=str(tmp_path / "result.rso"))
        return out

    run(script)
    assert [s.title for s in ctx.output().doc.sections] == ["Oneway"]


def test_direct_tool_and_menus_share_datasets_and_output(run, ctx, study):
    path, ds = study

    def script(call):
        call("open_data", path=str(path))
        out = call("independent_samples_t_test", variables=["y"], group_variable="group")
        call("run_command", command="Transform > Compute Variable")
        state = data([call("read_window")[0]])
        call("fill_dialog", press="Cancel")
        return out, state

    out, state = run(script)
    t = stats.ttest_ind(ds.df.y[:20], ds.df.y[20:]).statistic
    assert f"{t:.3f}".replace("0.", ".", 1) in text(out) or f"{t:.3f}" in text(out)
    assert state["title"].startswith("Compute")
    assert len(ctx.output().doc.sections) == 1
    assert len(ctx.editors) == 1


def test_tables_can_be_read_and_written(run, ctx, study):
    path, _ = study

    def script(call):
        call("open_data", path=str(path))
        before = data(call("read_table", rows=3))
        call("write_table", cells=[[2, 99.5]], first_row=1, first_column=1)
        after = data(call("read_table", rows=1))
        return before, after

    before, after = run(script)
    assert before["rows"] >= 40 and len(before["cells"]) == 3
    assert ctx.editors[0].ds.df.x[0] == 99.5 and ctx.editors[0].ds.df.group[0] == 2
    assert after["cells"][0][1].startswith("99.5")


def test_message_box_and_file_dialog(run, study, tmp_path):
    path, _ = study
    target = tmp_path / "copy.rsd"

    def script(call):
        call("open_data", path=str(path))
        d = data(call("run_command", command="File > Save As", window="stats"))
        assert "file_path" in d["dialog"]["message"]
        d = data(call("fill_dialog", press="Cancel"))
        assert d["status"] == "done"
        call("run_command", command="File > Save As", window="stats")
        call("fill_dialog", file_path=str(target))
        return data(call("app_status"))

    assert run(script)["dialog"] is None
    assert target.exists()


def test_refusals(run, ctx, study):
    path, _ = study

    def script(call):
        call("open_data", path=str(path))
        for command, message in (("Claude Connector", "Only the user"),
                                 ("Analyze > No Such Thing", "No menu command")):
            with pytest.raises(ValueError, match=message):
                call("run_command", command=command, window="stats")

    run(script)
    from rsstatchem.mcp.driver import NATIVE_COMMANDS

    assert NATIVE_COMMANDS.match("Print") and NATIVE_COMMANDS.match("Page Setup")
    assert not NATIVE_COMMANDS.match("Print Preview")
    ctx.connector_calls = 1
    ctx.show_connector()           # returns at once: Claude cannot open its own settings


def test_process_result_reaches_the_output_viewer(run, ctx, study):
    """A PROCESS analysis returns a result object: the viewer must get its section."""
    path, _ = study

    def script(call):
        call("open_data", path=str(path))
        return call("run_command", command="PROCESS Model 1 (Moderation)",
                    variables={"y": ["y"], "x": ["x"], "w": ["group"]}, press="OK")

    out = run(script)
    assert "errors" not in data(out), data(out)
    assert all(hasattr(s, "id") for s in ctx.output().doc.sections)
    assert data(out)["output"]


def test_button_and_radio_of_the_same_name(run, study):
    path, _ = study

    def script(call):
        call("open_data", path=str(path))
        return call("run_command", command="Descriptive Statistics > Explore",
                    variables={"deps": ["y"], "factors": ["group"]},
                    steps=[{"open": "Plots", "values": {"Normality plots with tests": True}}],
                    press="OK")

    assert "Tests of Normality" in text(run(script))


def test_exception_in_a_slot_is_reported(run, ctx, study, monkeypatch):
    path, _ = study
    from rsstatchem.ui.data_editor import DataEditorWindow

    def broken(self):
        raise RuntimeError("boom in a slot")

    monkeypatch.setattr(DataEditorWindow, "choose_font", broken)

    def script(call):
        call("open_data", path=str(path))
        return data(call("run_command", command="View > Fonts"))

    assert "boom in a slot" in run(script)["errors"][0]


# ---------------------------------------------------------------------------- other modules
def test_rsm_from_design_to_optimum(run, tmp_path):
    def script(call):
        out = call("rsm_create_design", design="ccd", randomize=False,
                   factors=[{"name": "T", "low": 60, "high": 80},
                            {"name": "t", "low": 30, "high": 60}],
                   responses=[{"name": "Yield", "units": "%"}], options={"center_points": 5})
        assert "13 runs" in text(out)
        return out

    run(script)

    def script2(call):
        ys = []
        lines = [ln for ln in text(call("rsm_design")).splitlines() if ln.startswith("| ")][1:]
        for ln in lines:
            cells = [c.strip() for c in ln.strip("|").split("|")]
            nums = [float(c) for c in cells if c.replace(".", "").replace("-", "").isdigit()]
            T, t = nums[-2], nums[-1]
            a, b = (T - 70) / 10, (t - 45) / 15
            ys.append(80 + 2 * a - b - 3 * a * a - 2 * b * b)
        rng = np.random.default_rng(1)
        ys = list(np.round(np.array(ys) + rng.normal(0, .1, len(ys)), 3))
        call("rsm_set_response", values=ys)
        fit = call("rsm_analyze", model="quadratic", diagnostics=True)
        assert "Lack of Fit" in text(fit)
        reduced = call("rsm_analyze", terms=["A", "B", "A^2", "B^2"])
        assert "TERMS=A B A² B²" in text(reduced)
        graph = call("rsm_graph", graph="contour")
        assert graph[-1].png[:4] == b"\x89PNG"
        with pytest.raises(ValueError, match="not a model term"):
            call("rsm_analyze", terms=["Q"])
        best = call("rsm_optimize", goals=[{"name": "Yield", "goal": "maximize"}])
        call("rsm_save", path=str(tmp_path / "d.rsx"))
        return text(best)

    best = run(script2)
    assert "Solutions" in best and (tmp_path / "d.rsx").exists()


def test_graph_fit_matches_a_known_curve(run, tmp_path):
    x = np.arange(9.0)
    y = 10 * np.exp(-0.3 * x)

    def script(call):
        call("graph_data", columns={"t": list(x), "C": list(np.round(y, 6))})
        plot = call("graph_plot", kind="scatter", columns=["C"])
        assert plot[-1].png[:4] == b"\x89PNG"
        with pytest.raises(ValueError, match="scatter"):
            call("graph_plot", kind="nonsense", columns=["C"])
        out = call("graph_fit", model="y = a*exp(-k*x)", y_column="C",
                   start={"a": 8, "k": 0.2})
        call("graph_save", path=str(tmp_path / "p.rsop"))
        return text(out)

    out = run(script)
    assert "0.3" in out and (tmp_path / "p.rsop").exists()


def test_chemistry(run, tmp_path):
    def script(call):
        info = text(call("chem_compound", structure="caffeine"))
        assert "C8H10N4O2" in info.replace("₈", "8").replace("₁₀", "10").replace(
            "₄", "4").replace("₂", "2") or "1,3,7-trimethyl" in info
        drawn = call("chem_draw", structure="CC(=O)Oc1ccccc1C(=O)O")
        assert data(drawn)["formula"] == "C9H8O4"
        call("chem_save", path=str(tmp_path / "a.mol"))
        assert "4 Fe + 3 O₂ → 2 Fe₂O₃" in text(call("chem_balance", equation="Fe + O2 -> Fe2O3"))
        with pytest.raises(ValueError, match="neither"):
            call("chem_compound", structure="zzzznotacompound")

    run(script)
    assert "V2000" in (tmp_path / "a.mol").read_text()


def test_image_commands_and_macro(run, tmp_path):
    blobs = ROOT / "tests" / "data_imagej" / "inputs" / "blobs.gif"

    def script(call):
        opened = call("image_open", path=str(blobs))
        assert data(opened)["images"][0]["width"] == 256
        assert "sigma" in text(call("image_commands", query="gaussian blur"))
        call("image_command", command="Gaussian Blur...", options="sigma=1")
        out = call("image_macro", code='setAutoThreshold("Otsu"); run("Convert to Mask"); '
                                       'run("Analyze Particles...", "size=20-Infinity display");')
        with pytest.raises(ValueError, match="Macro"):
            call("image_macro", code="this is not a macro(")
        call("image_save", path=str(tmp_path / "b.png"))
        return text(out)

    assert "Area" in run(script)
    assert (tmp_path / "b.png").exists()


# ---------------------------------------------------------------------------- live mode, setup
def test_live_bridge_forwards_calls_and_checks_the_token(qapp, ctx, run, study):
    from rsstatchem.mcp.bridge import LiveClient, token_file

    path, _ = study
    assert LiveClient().call("app_status", {}) is None       # nobody listens
    assert ctx.start_connector()
    try:
        mode = token_file().stat().st_mode & 0o777
        assert mode == 0o600 or __import__("sys").platform == "win32"

        def script(call):
            client = LiveClient()
            parts = client.call("open_data", {"path": str(path)})
            status = data(client.call("app_status", {}))
            with pytest.raises(ValueError, match="no variable named"):
                client.call("descriptives", {"variables": ["zz"]})
            good = token_file().read_text()
            token_file().write_text("0" * 64)
            with pytest.raises(ValueError, match="closed the connection"):
                client.call("app_status", {})
            token_file().write_text(good)
            return parts, status

        parts, status = run(script)
        assert "study" in text(parts) and status["mode"].startswith("live")
    finally:
        ctx.stop_connector()
    assert LiveClient().call("app_status", {}) is None
    assert not token_file().exists()


def test_claude_configuration_entry(tmp_path):
    from rsstatchem.mcp import setup

    folder = tmp_path / "Claude"
    path = folder / "claude_desktop_config.json"
    with pytest.raises(ValueError, match="not found"):
        setup.add(path)
    folder.mkdir()
    path.write_text(json.dumps({"theme": "dark", "mcpServers": {"other": {"command": "x"}}}))
    assert setup.state(path) == "missing"
    backup = setup.add(path)
    saved = json.loads(path.read_text())
    assert setup.state(path) == "added" and backup.exists()
    assert saved["theme"] == "dark" and saved["mcpServers"]["other"] == {"command": "x"}
    assert saved["mcpServers"]["rs-statchem"]["args"][-1] == "--mcp"
    setup.remove(path)
    assert json.loads(path.read_text())["mcpServers"] == {"other": {"command": "x"}}
    path.write_text("{broken")
    assert setup.state(path) == "broken"
    with pytest.raises(ValueError):
        setup.add(path)
    assert path.read_text() == "{broken"


def test_dialog_cannot_be_operated_by_claude(qapp, ctx, monkeypatch):
    from rsstatchem.mcp import setup
    from rsstatchem.ui.dialogs.connector import ConnectorDialog

    monkeypatch.setattr(setup, "add", lambda: pytest.fail("changed the configuration"))
    dlg = ConnectorDialog(ctx)
    ctx.connector_calls = 1
    dlg.live.click()
    dlg.add_entry()
    assert not ctx.connector_listening() and not dlg.live.isChecked()


def test_every_tool_is_documented():
    from rsstatchem.mcp.engine import all_tools
    from rsstatchem.mcp.module_tools import FullWorkspace

    names = [t.name for t in all_tools()]
    assert len(names) == len(set(names)) >= 50
    for name in names:
        assert len(getattr(FullWorkspace, name).__doc__) > 60, name
