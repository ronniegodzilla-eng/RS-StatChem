"""Start screen and module routing."""
import pytest


@pytest.fixture
def ctx(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("RSSTATCHEM_HOME", str(tmp_path))
    from rsstatchem.app import AppContext

    context = AppContext(qapp)
    quits = []
    monkeypatch.setattr(context.app, "quit", lambda: quits.append(True))
    context.quits = quits
    return context


def test_four_modules_and_file_routing():
    from rsstatchem.ui.modules import MODULES, module_for_path

    assert [m.key for m in MODULES] == ["stats", "rsm", "graph", "chem"]
    assert module_for_path("data.SAV").key == "stats"
    assert module_for_path("design.rsx").key == "rsm"
    assert module_for_path("project.rsop").key == "graph"
    assert module_for_path("aspirin.rsc").key == "chem"
    assert module_for_path("output.rso").key == "stats"
    assert module_for_path("survey.dta").key == "stats"
    assert module_for_path("notes.docx") is None


def test_launcher_opens_module_and_hides(ctx):
    launcher = ctx.show_launcher()
    assert set(launcher.tiles) == {"stats", "rsm", "graph", "chem"}
    launcher.choose("stats")
    assert not launcher.isVisible()
    assert ctx.windows == ctx.editors and len(ctx.editors) == 1
    assert ctx.settings.value("last_module") == "stats"
    # the Statistics window is SPSS-only now; the other modules are reached from Window
    titles = [a.text() for a in ctx.editors[0].menuBar().actions()]
    assert "&RSM" not in titles and "&Chemistry" not in titles


def test_last_window_closing_quits(ctx):
    for key in ("stats", "rsm", "graph", "chem"):
        ctx.open_module(key)
    assert len(ctx.windows) == 4
    for window in list(ctx.windows):
        window.close()
    assert ctx.windows == [] and ctx.quits == [True]


def test_launcher_counts_as_a_window(ctx):
    launcher = ctx.show_launcher()
    editor = ctx.open_module("stats")
    editor.close()
    assert ctx.quits == []  # the launcher is still open
    launcher.close()
    assert ctx.quits == [True]


def test_open_file_goes_to_its_module(ctx, tmp_path, monkeypatch):
    from rsstatchem.ui import modules

    opened = []
    monkeypatch.setattr(modules, "open_module",
                        lambda c, key, path=None: opened.append((key, path)))
    ctx.open_file(str(tmp_path / "x.rsc"))
    ctx.open_file(str(tmp_path / "y.unknown"))
    assert opened == [("chem", str(tmp_path / "x.rsc")), ("stats", str(tmp_path / "y.unknown"))]
