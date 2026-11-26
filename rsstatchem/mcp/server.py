"""The MCP server of the RS-StatChem connector (``RS-StatChem --mcp``, stdio transport).

Built on the official ``mcp`` Python SDK (2.x, ``MCPServer``). The tools are the methods of
:class:`rsstatchem.mcp.module_tools.FullWorkspace`, each under its own name with its docstring
as the description; a problem a tool reports becomes a tool error the model can read and act on.
This is the only module that imports the SDK.

Threads: Qt owns the main thread (the app's windows exist off screen), the MCP server runs in a
second one. When the visible app is open and the user allowed it there, calls are forwarded to
it instead (:mod:`rsstatchem.mcp.bridge`).
"""
from __future__ import annotations

import functools
import inspect
import os
import sys
import threading

from mcp.server import MCPServer
from mcp.server.mcpserver import Image as McpImage
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from rsstatchem import APP_NAME, __version__
from rsstatchem.mcp.tools import TOOLS, Workspace

SERVER_NAME = "rs-statchem"

INSTRUCTIONS = """\
RS-StatChem is a desktop application for statistics and chemistry with five modules: stats \
(SPSS-style Data Editor, analyses, Output Viewer), rsm (design of experiments, response \
surfaces), graph (workbooks, graphs, curve fitting, signal and peak analysis), chem (structure \
drawing, names, properties) and image (ImageJ-style image analysis). These tools operate it. \
Numbers come from the app's own engine; tables use SPSS names and formats (".045" means 0.045, \
Sig. is the p-value, ".000" means p < .0005, report it as p < .001).

Statistics: open_data with the full path of the file (ask the user for it if you do not have \
it) and read the variable list it returns. Use the direct tools for common tests \
(descriptives, independent_samples_t_test, one_way_anova, correlations, linear_regression, \
...). Every other analysis, data transformation and chart is a menu command: find_commands \
(module "stats", words such as "mann whitney"), then run_command. The dialog it opens is \
returned with its variable boxes, fields and buttons; answer with fill_dialog(variables, \
values, steps for sub-dialogs, press="OK"), or give all of it in run_command at once. Results \
are returned and collected in the Output Viewer.

Other modules: rsm_*, graph_*, chem_* and image_* tools cover the usual work; everything else \
in those modules is in their menus too (open_module, find_commands, run_command, read_window, \
read_table, write_table, take_screenshot, mouse).

app_status tells whether you are in background mode (the windows are not visible to the user: \
save what they should keep with save_data, save_output, rsm_save, graph_save, chem_save, \
image_save, and offer show_in_app) or in live mode (the windows the user sees; do not close or \
overwrite their work without asking). Ask before replacing a file.

Check the assumptions of the test you choose and say which test you ran and why. Explain results \
in the language the user writes in; many users are Indonesian students and lecturers. Data \
stay on the computer, but whatever a tool returns becomes part of the conversation: do not \
list names, IDs or other identifying values unless the user asks for them.\
"""


def _reporting(fn):
    """Turn a failure into a tool error whose text the model sees (the SDK would otherwise
    replace an unexpected exception with a bare "Error executing tool")."""
    @functools.wraps(fn)
    def run(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
        except KeyError as exc:
            raise ToolError(f"Unknown name: {exc.args[0] if exc.args else exc}") from exc
        except OSError as exc:
            raise ToolError(f"File problem: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - any analysis failure is reported, not hidden
            raise ToolError(f"The analysis could not be completed ({type(exc).__name__}: "
                            f"{exc}).") from exc
    return run


def _content(parts: list):
    """Tool result -> MCP content: one text, or a list of texts and pictures."""
    out = [p if isinstance(p, str) else McpImage(data=p.png, format="png") for p in parts]
    return out[0] if len(out) == 1 and isinstance(out[0], str) else out


class Backend:
    """Where tool calls run: the visible app when it is listening, else our own windows."""

    def __init__(self, make_engine, live=None) -> None:
        self._make_engine = make_engine
        self._engine = None
        self._live = live
        self._was_live: bool | None = None
        self._lock = threading.Lock()       # one tool at a time: windows have one state

    def call(self, name: str, args: dict) -> list:
        with self._lock:
            parts = None
            if self._live is not None:
                parts = self._live.call(name, args)
            live = parts is not None
            if not live:
                if self._engine is None:
                    self._engine = self._make_engine()
                parts = self._engine.call(name, args)
            if self._was_live is not None and live != self._was_live:
                parts = [*parts, "Note: from now on the tools work in "
                         + ("the RS-StatChem window the user has open (live mode); it has its "
                            "own datasets." if live else
                            "background windows again (the RS-StatChem window was closed or no "
                            "longer allows it).")]
            self._was_live = live
            return parts


class Direct:
    """The statistics tools alone on a plain workspace (no Qt): self-test and unit tests."""

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    def call(self, name: str, args: dict) -> list:
        fn = getattr(self.workspace, name, None)
        if fn is None:
            raise ValueError(f"{name} needs the application.")
        return [fn(**args)]


def build_server(backend=None) -> MCPServer:
    """The server; ``backend`` is a :class:`Backend`, or anything with ``call(name, args)``."""
    from rsstatchem.mcp.engine import all_tools
    from rsstatchem.mcp.module_tools import FullWorkspace

    if backend is None:
        backend = Backend(_background_engine)
    elif isinstance(backend, Workspace):
        backend = Direct(backend)
    server = MCPServer(SERVER_NAME, title=f"{APP_NAME} connector", version=__version__,
                       instructions=INSTRUCTIONS, log_level="WARNING")
    for info in all_tools():
        server.add_tool(
            _reporting(_forward(getattr(FullWorkspace, info.name), info.name, backend)),
            name=info.name, title=info.title,
            description=inspect.cleandoc(getattr(FullWorkspace, info.name).__doc__),
            annotations=ToolAnnotations(title=info.title, read_only_hint=info.read_only,
                                        destructive_hint=not info.read_only,
                                        idempotent_hint=info.read_only, open_world_hint=False),
            structured_output=False)   # the text is the result; no duplicate structured copy
    return server


def _forward(method, name: str, backend):
    """A function with the tool's signature (the SDK builds the input schema from it)."""
    signature = inspect.signature(method)
    params = [p for p in signature.parameters.values() if p.name != "self"]
    hints = {k: v for k, v in __import__("typing").get_type_hints(method).items()
             if k != "return"}

    def run(**kwargs):
        return _content(backend.call(name, kwargs))

    run.__name__ = name
    run.__doc__ = method.__doc__
    run.__signature__ = signature.replace(
        parameters=[p.replace(kind=inspect.Parameter.KEYWORD_ONLY,
                              annotation=hints.get(p.name, p.annotation)) for p in params],
        return_annotation=inspect.Signature.empty)
    run.__annotations__ = hints
    return run


def _background_engine():
    """An engine on the QApplication of this process, created on the Qt thread."""
    from PySide6.QtCore import QMetaObject, QObject, QSettings, Qt, QThread, Slot
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        raise RuntimeError("the connector's Qt application is not running")

    def make():
        from rsstatchem.app import AppContext
        from rsstatchem.mcp.engine import Engine
        from rsstatchem.paths import user_data_dir

        ctx = AppContext(app)
        ctx.check_for_updates = lambda manual=False: None
        # the user's own settings (recent files, last module, live mode) stay theirs
        ctx.settings = QSettings(str(user_data_dir() / "connector.ini"),
                                 QSettings.Format.IniFormat)
        app._rs_context = ctx
        return Engine(ctx)

    if QThread.currentThread() is app.thread():
        return make()

    class Maker(QObject):
        result = error = None

        @Slot()
        def run(self):
            try:
                self.result = make()
            except BaseException as exc:  # noqa: BLE001 - raised on the calling thread
                self.error = exc

    maker = Maker()
    maker.moveToThread(app.thread())
    QMetaObject.invokeMethod(maker, "run", Qt.ConnectionType.BlockingQueuedConnection)
    if maker.error is not None:
        raise maker.error
    return maker.result


def _ensure_stdio() -> None:
    """The Windows build is a windowed program: Python leaves the standard streams as None
    unless the client's pipes are valid at start-up. Attach to the descriptors directly when
    they exist, and give stray diagnostics somewhere to go."""
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115
    if sys.stdin is None:
        sys.stdin = open(0, "r", encoding="utf-8", closefd=False)  # noqa: SIM115
    if sys.stdout is None:
        sys.stdout = open(1, "w", encoding="utf-8", closefd=False)  # noqa: SIM115


def serve() -> int:
    """Serve over stdin/stdout until the client closes the connection."""
    try:
        _ensure_stdio()
    except OSError:
        return 2    # started without a client attached (e.g. double-clicked with --mcp)
    app = _offscreen_app()
    from rsstatchem.mcp.bridge import LiveClient

    server = build_server(Backend(_background_engine, LiveClient()))
    code = {"value": 0}

    def run() -> None:
        try:
            server.run("stdio")
        except KeyboardInterrupt:
            pass
        except BaseException:  # noqa: BLE001 - shown in the client's log
            import traceback

            traceback.print_exc()
            code["value"] = 1
        finally:
            from PySide6.QtCore import QMetaObject, Qt

            QMetaObject.invokeMethod(app, "quit", Qt.ConnectionType.QueuedConnection)

    threading.Thread(target=run, name="mcp", daemon=True).start()
    app.exec()
    _close(app)
    return code["value"]


def _offscreen_platform() -> str:
    """The off-screen platform with a desktop-sized screen (the default is 800 x 800)."""
    import json

    from rsstatchem.paths import user_data_dir

    path = user_data_dir() / "offscreen.json"
    screen = {"name": "connector", "x": 0, "y": 0, "width": 1680, "height": 1050,
              "logicalDpi": 96, "logicalBaseDpi": 96, "dpr": 1}
    try:
        path.write_text(json.dumps({"screens": [screen]}), encoding="utf-8")
    except OSError:
        return "offscreen"
    text = str(path)
    # Qt splits platform options at ":", so a Windows drive letter would break the path
    return f"offscreen:configfile={text}" if ":" not in text else "offscreen"


def _offscreen_app():
    os.environ["QT_QPA_PLATFORM"] = _offscreen_platform()
    from PySide6.QtWidgets import QApplication

    from rsstatchem import APP_ORG
    from rsstatchem.app import use_window_menu_bars
    from rsstatchem.ui.theme import apply_theme

    use_window_menu_bars()
    app = QApplication.instance() or QApplication([APP_NAME])
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(APP_ORG)
    app.setQuitOnLastWindowClosed(False)
    apply_theme(app)
    return app


def _close(app) -> None:
    from rsstatchem.ui.qt_lifecycle import shutdown_application

    ctx = getattr(app, "_rs_context", None)
    if ctx is not None:
        ctx.editors.clear()
        ctx.windows.clear()
        ctx._launcher = ctx._output = ctx._updates = None
        del app._rs_context
    shutdown_application(app)


def selftest() -> None:
    """Headless check used by ``--selftest`` in the packaged app: an in-memory MCP client lists
    the tools, opens a data file and runs an analysis through the protocol layer."""
    import tempfile
    from pathlib import Path

    import anyio
    import numpy as np
    import pandas as pd
    from mcp import Client

    from rsstatchem.core.dataset import Dataset
    from rsstatchem.io import files

    async def check(path: str) -> None:
        async with Client(build_server(Workspace())) as client:
            names = {t.name for t in (await client.list_tools()).tools}
            assert {t.name for t in TOOLS} <= names, names
            opened = await client.call_tool("open_data", {"path": path})
            assert not opened.is_error and "| x |" in opened.content[0].text, opened
            result = await client.call_tool("one_sample_t_test", {"variables": ["x"]})
            assert not result.is_error and "One-Sample Test" in result.content[0].text, result
            bad = await client.call_tool("descriptives", {"variables": ["missing_name"]})
            assert bad.is_error and "no variable named" in bad.content[0].text, bad

    rng = np.random.default_rng(3)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "mcp.rsd"
        files.save_project(Dataset(pd.DataFrame({"x": rng.normal(size=30)})), path)
        anyio.run(check, str(path))
