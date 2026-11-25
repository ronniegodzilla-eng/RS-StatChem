"""The whole application as connector tools.

:class:`AppWorkspace` keeps the statistics tools of :class:`~rsstatchem.mcp.tools.Workspace` and
adds tools for every module: the app's own windows are opened (off screen in the background
connector, or the user's visible ones in live mode) and operated through
:mod:`rsstatchem.mcp.driver`. Datasets are the Data Editor's and results go to the Output Viewer,
so a tool and a menu command see the same state.

A tool returns text or a list of parts (``str`` and :class:`Picture`); ``server.py`` turns
them into MCP content. Tools listed with ``ui=True`` run on the caller's thread and reach the
Qt thread through ``self.gui``; all others run on the Qt thread as a whole.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from PySide6.QtWidgets import (QAbstractItemView, QApplication, QDialog, QFileDialog, QMenu,
                               QMessageBox, QWidget)

from rsstatchem.core import selection
from rsstatchem.core.output import Image, Section, Table
from rsstatchem.mcp import driver, tools
from rsstatchem.mcp.render import render, section_markdown
from rsstatchem.mcp.tools import ToolInfo, Workspace, _resolve, _section

Module = Literal["stats", "rsm", "graph", "chem", "image"]
MAX_PICTURES = 6
MAX_PARTS_TEXT = 60_000

APP_TOOLS: list[ToolInfo] = []
UI_TOOLS: set[str] = set()


@dataclass
class Picture:
    png: bytes
    title: str = ""


def app_tool(title: str, read_only: bool = False, ui: bool = False):
    def register(fn):
        APP_TOOLS.append(ToolInfo(fn.__name__, title, read_only))
        if ui:
            UI_TOOLS.add(fn.__name__)
        return fn
    return register


def all_tools() -> list[ToolInfo]:
    from rsstatchem.mcp import module_tools  # noqa: F401 - registers the module tools

    return [*tools.TOOLS, *APP_TOOLS]


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, default=str)


def _output_path(path: str, overwrite: bool, types: tuple[str, ...]) -> Path:
    target = _resolve(path)
    if target.suffix.lower() not in types:
        raise ValueError("The file must end with one of: " + ", ".join(types) + ".")
    if not target.parent.is_dir():
        raise ValueError(f"The folder does not exist: {target.parent}")
    if target.exists() and not overwrite:
        raise ValueError(f"{target} already exists. Confirm with the user, then repeat with "
                         "overwrite=true, or choose another name.")
    return target


class AppWorkspace(Workspace):
    """Connector session on a running application (``ctx``: the app's AppContext)."""

    def __init__(self, ctx, gui: driver.GuiThread, live: bool = False) -> None:
        self.ctx = ctx
        self.gui = gui
        self.live = live
        self.elements = driver.Elements()
        self.errors: list[str] = []
        self._seen: set[int] = set()         # output sections already reported
        self._known_windows: set[int] = set()
        super().__init__()
        self._mark_output()

    # ------------------------------------------------------------------ datasets = editors
    @staticmethod
    def _editor_name(editor) -> str:
        name = getattr(editor, "_connector_name", "")
        if not name:
            path = editor.ds.path
            name = tools.sanitize_name(Path(path).stem, fallback="data") if path \
                else editor.ds.name
        return name

    @property
    def datasets(self) -> dict:
        out: dict = {}
        for editor in self.ctx.editors:
            if editor.ds.n_vars == 0:
                continue
            name, k = self._editor_name(editor), 1
            while name in out:
                k += 1
                name = f"{self._editor_name(editor)}_{k}"
            out[name] = editor.ds
        return out

    @datasets.setter
    def datasets(self, value) -> None:
        pass

    @property
    def active(self) -> str | None:
        names = {id(ds): n for n, ds in self.datasets.items()}
        window = self.ctx.app.activeWindow()
        order = [window] if window in self.ctx.editors else []
        order += [e for e in [getattr(self, "_active_editor", None)] if e in self.ctx.editors]
        order += list(reversed(self.ctx.editors))
        return next((names[id(e.ds)] for e in order if id(e.ds) in names), None)

    @active.setter
    def active(self, value) -> None:
        pass

    def editor_of(self, name: str = ""):
        ds = self.dataset(name)
        return next(e for e in self.ctx.editors if e.ds is ds)

    def _register(self, ds, name: str, source: Path) -> str:
        base = tools.sanitize_name(name or source.stem, fallback="data")
        for editor in self.ctx.editors:      # the same file opened again: replace it
            if getattr(editor, "_connector_source", "") == str(source) and \
                    self._editor_name(editor).lower() == base.lower() and not editor.ds.modified:
                editor.load_dataset(ds)
                break
        else:
            blank = next((e for e in self.ctx.editors if e.is_pristine()), None)
            if blank is not None:
                blank.load_dataset(ds)
                editor = blank
            else:
                editor = self.ctx.new_editor(ds)
        taken = {self._editor_name(e).lower() for e in self.ctx.editors if e is not editor}
        key, k = base, 1
        while key.lower() in taken:
            k += 1
            key = f"{base}_{k}"
        editor._connector_name = key
        editor._connector_source = str(source)
        if source.suffix.lower() in (".rsd", ".sav"):
            ds.path = str(source)
        ds.source = str(source)
        ds.name = key
        self._active_editor = editor
        if hasattr(editor, "_update_title"):
            editor._update_title()
        return key

    def _run(self, ds, fn, output_format: str) -> str:   # noqa: D401 - see Workspace._run
        views = selection.analysis_views(ds)
        if len(views) == 1 and not selection.describe(ds):
            section = _section(fn(ds))
        else:
            section = selection.combine([(v.label, _section(fn(v.ds))) for v in views], ds)
        self.publish(section)
        return render(section, output_format)

    # ------------------------------------------------------------------ output
    def publish(self, section: Section) -> None:
        """Add a result to the Output Viewer (and bring it up in live mode)."""
        self.ctx.output().add_section(section)
        self._seen.add(section.id)
        if self.live:
            self.ctx.show_output_window()

    def _sections(self) -> list[Section]:
        return list(self.ctx._output.doc.sections) if self.ctx._output is not None else []

    def _mark_output(self) -> None:
        self._seen = {s.id for s in self._sections()}

    def _parts(self, sections: list[Section], charts: bool) -> list:
        parts: list = []
        for section in sections:
            text = section_markdown(section)
            if len(text) > MAX_PARTS_TEXT:
                text = text[:MAX_PARTS_TEXT] + "\n\n… (cut; read the rest with get_output)"
            if charts:
                text = text.replace("*[Chart not sent: ", "*[Chart: ")
            parts.append(text)
            if charts:
                pictures = [i for i in section.items if isinstance(i, Image)]
                parts += [Picture(i.png, i.title) for i in pictures[:MAX_PICTURES]]
        return parts

    # ------------------------------------------------------------------ windows
    def _window(self, name: str = "") -> QWidget:
        """A window by element id, title or module; default the one in front."""
        windows = driver.open_windows()
        text = (name or "").strip()
        if not text:
            dialogs = driver.open_dialogs()
            if dialogs:
                return dialogs[-1]
            active = self.ctx.app.activeWindow()
            if active in windows:
                return active
            last = getattr(self, "_last_window", None)
            if last in windows:
                return last
            if windows:
                return windows[-1]
            raise ValueError("No window is open. Use open_module first.")
        direct = self.elements.get(text)
        if direct is not None:
            return direct.window()
        pool = [*driver.open_dialogs(), *windows]
        hits = [w for w in pool if driver.title_of(w).lower() == text.lower()] or \
            [w for w in pool if driver.window_kind(w) == text.lower()
             and not isinstance(w, QDialog)] or \
            [w for w in pool if text.lower() in driver.title_of(w).lower()]
        if not hits:
            raise ValueError(f"No window '{name}'. Open windows: "
                             + ", ".join(driver.title_of(w) for w in pool) + ".")
        return hits[-1]

    def _menu_window(self, name: str = "") -> QWidget:
        """Like :meth:`_window`, but a window without a menu (an ImageJ image, a results
        table) stands for the menu window of its module."""
        target = self._window(name)
        if isinstance(target, QDialog):
            return target
        kind = driver.window_kind(target)
        if name.strip().lower() == kind == "image":      # its menus are in the ImageJ bar
            return max((w for w in driver.open_windows() if driver.window_kind(w) == kind),
                       key=lambda w: len(driver.commands(w)))
        if driver.commands(target):
            return target
        pool = [w for w in driver.open_windows()
                if driver.window_kind(w) == kind and driver.commands(w)]
        return pool[-1] if pool else target

    def _command(self, target: QWidget, command: str, fixed: bool):
        """The menu command in ``target``, else (no window named) in another open window."""
        try:
            return (target, *driver.find_command(target, command))
        except ValueError:
            if fixed:
                raise
            others = [w for w in driver.open_windows() if driver.commands(w) and w is not target]
            others.sort(key=lambda w: (w is getattr(self, "_last_window", None),
                                       w in self.ctx.editors))
            for other in reversed(others):
                try:
                    return (other, *driver.find_command(other, command))
                except ValueError:
                    continue
            raise

    def _windows_data(self) -> list[dict]:
        active = self.ctx.app.activeWindow()
        return [{"window": self.elements.id_of(w), "title": driver.title_of(w),
                 "module": driver.window_kind(w), **({"active": True} if w is active else {})}
                for w in driver.open_windows()]

    def _popup(self) -> QMenu | None:
        popup = self.ctx.app.activePopupWidget()
        return popup if isinstance(popup, QMenu) else None

    def _outcome(self, status: str, charts: bool = False) -> list:
        """What a command led to: dialog waiting, output, new windows, errors."""
        data: dict = {"status": {"done": "done", "dialog": "waiting for input in a dialog",
                                 "busy": "still computing: call wait"}[status]}
        if status == "busy":
            return [_json(data)]
        errors, self.errors = self.errors, []
        if errors:
            data["errors"] = errors[-3:]
        dialogs = driver.open_dialogs()
        if dialogs:
            data["dialog"] = driver.describe_window(dialogs[-1], self.elements)
        popup = self._popup()
        if popup is not None:
            data["menu"] = [driver.clean(a.text()) for a in popup.actions()
                            if a.isVisible() and a.isEnabled() and a.text()]
        windows = driver.open_windows()
        fresh = [w for w in windows if driver.key_of(w) not in self._known_windows]
        if fresh:
            data["new_windows"] = [
                {"window": self.elements.id_of(w), "title": driver.title_of(w),
                 "module": driver.window_kind(w)} for w in fresh]
            work = [w for w in fresh if driver.window_kind(w) != "output"]
            if work:
                self._last_window = work[-1]
        self._known_windows = {driver.key_of(w) for w in windows}
        new = [s for s in self._sections() if s.id not in self._seen]
        self._mark_output()
        if new:
            data["output"] = [s.title for s in new]
        return [_json(data), *self._parts(new, charts)]

    def _do(self, fn, charts: bool = False, timeout: float = 120) -> list:
        status = self.gui.interact(fn, timeout)
        return self._outcome(status) if status == "busy" else \
            self.gui.call(lambda: self._outcome(status, charts))

    # ------------------------------------------------------------------ app and windows
    @app_tool("Application status", read_only=True)
    def app_status(self) -> str:
        """Which windows are open, which dialog is waiting, which datasets are loaded and how
        many results are in the Output Viewer. Start here when you are unsure of the state."""
        dialogs = driver.open_dialogs()
        return _json({
            "mode": "live: the user's visible RS-StatChem windows" if self.live else
                    "background: windows are not visible to the user; save results to files",
            "windows": self._windows_data(),
            "dialog": driver.title_of(dialogs[-1]) if dialogs else None,
            "datasets": list(self.datasets), "active_dataset": self.active,
            "output_sections": len(self._sections())})

    @app_tool("Open a module or file", ui=True)
    def open_module(self, module: Module | Literal["auto"] = "auto", path: str = "") -> list:
        """Open a module window, optionally with a file: stats (Data Editor, .rsd .sav .xlsx
        .csv …), rsm (design of experiments, .rsx), graph (workbooks and graphs, .rsop .rsg),
        chem (chemical drawing, .rsc .mol .sdf .rxn .cdxml), image (image analysis, .tif .png
        .jpg .dcm …).

        Args:
            module: The module, or auto to choose by the file's type.
            path: Optional full path of a file to open.
        """
        def go():
            from rsstatchem.ui import modules

            text = str(_resolve(path)) if path.strip() else None
            if text and not Path(text).exists():
                raise ValueError(f"File not found: {text}")
            if module == "auto":
                if not text:
                    raise ValueError("Give a module or a file.")
                key = (modules.module_for_path(text) or modules.BY_KEY["stats"]).key
            else:
                key = module
            if key == "rsm" and not text:
                from rsstatchem.ui.dx import open_workspace

                window = open_workspace(self.ctx, wizard=False)
                self.ctx.register_window(window)
            else:
                window = modules.open_module(self.ctx, key, text)
            self._last_window = window

        return self._do(go)

    @app_tool("Find menu commands", read_only=True)
    def find_commands(self, query: str = "", window: str = "", module: str = "",
                      limit: int = 60) -> str:
        """Search the menu commands of a window ("every feature" of a module is in its menus).
        All words must occur in the menu path, in any order. An empty query lists the menus
        with their number of commands.

        Args:
            query: Words, for example "mann whitney", "smooth", "export pdf", "threshold".
            window: Window id, title or module key (stats, rsm, graph, chem, image, output).
            module: Open this module first when it has no window yet (stats, rsm, graph, …).
            limit: At most this many commands.
        """
        target = None
        if module:
            target = next((w for w in reversed(driver.open_windows())
                           if driver.window_kind(w) == module and driver.commands(w)), None)
            if target is None and module == "output":
                self.ctx.show_output_window()
                target = self.ctx.output()
            if target is None:
                raise ValueError(f"The module '{module}' is not open; call open_module first.")
        else:
            target = self._menu_window(window)
        entries = driver.commands(target)
        if not entries:
            raise ValueError(f"The window '{driver.title_of(target)}' has no menu. Windows: "
                             + ", ".join(driver.title_of(w) for w in driver.open_windows()))
        words = driver._norm(query).replace(">", " ").split()
        if not words:
            counts: dict[str, int] = {}
            for path, _ in entries:
                head = " > ".join(path.split(" > ")[:2]) if len(entries) < 400 else \
                    path.split(" > ")[0]
                counts[head] = counts.get(head, 0) + 1
            return _json({"window": driver.title_of(target), "commands": len(entries),
                          "menus": counts})
        hits = [(p, a) for p, a in entries
                if all(w in driver._norm(p) for w in words)]
        return _json({"window": driver.title_of(target), "matches": len(hits),
                      "commands": [p + ("" if a.isEnabled() else "   (disabled now)")
                                   for p, a in hits[:max(1, limit)]]})

    @app_tool("Run a menu command", ui=True)
    def run_command(self, command: str, window: str = "",
                    variables: dict[str, Any] | None = None,
                    values: dict[str, Any] | None = None,
                    steps: list[dict[str, Any]] | None = None, press: str = "",
                    charts: bool = False, timeout: int = 120) -> list:
        """Choose a menu command, for example "Analyze > Nonparametric Tests > Legacy Dialogs >
        2 Independent Samples". If it opens a dialog you get its description (variable boxes,
        fields, choices, buttons) back and continue with fill_dialog; or give everything in
        this one call with variables / values / steps and press="OK".

        Args:
            command: Full menu path, or its end when that is unique (see find_commands).
            window: Window id, title or module; default is the window in front.
            variables: Variable boxes of the dialog: {box key or label: name or [names]}.
            values: Fields of the dialog: {id or label: value}.
            steps: Sub-dialogs to fill before press, in order:
                [{"open": "Options...", "values": {...}, "variables": {...},
                  "press": "Continue"}].
            press: Button that ends the dialog, usually "OK". Empty = leave it open.
            charts: Also return the charts of the output as pictures.
            timeout: Seconds to wait for the computation.
        """
        def start():
            target = self._menu_window(window)
            if isinstance(target, QDialog):
                raise ValueError(f"The dialog '{driver.title_of(target)}' is waiting; finish it "
                                 "first (fill_dialog, or press 'Cancel').")
            target, path, action = self._command(target, command, bool(window.strip()))
            if path.split(" > ")[-1].lower() in driver.SKIP_COMMANDS:
                raise ValueError("The connector does not quit the application.")
            if path.endswith("Claude Connector"):
                raise ValueError("Only the user can change the connector's settings.")
            if driver.NATIVE_COMMANDS.match(path.split(" > ")[-1]):
                raise ValueError("Printing and page setup use the system's own panels, which the "
                                 "connector cannot operate. Export to PDF instead.")
            if not action.isEnabled():
                raise ValueError(f"'{path}' is not available at the moment (it is greyed out in "
                                 "the menu).")
            self._last_window = target
            action.trigger()

        status = self.gui.interact(start, timeout)
        if status == "dialog" and (variables or values or steps or press):
            return self._fill(variables, values, steps, press, "", "", charts, timeout)
        if status == "busy":
            return self._outcome(status)
        return self.gui.call(lambda: self._outcome(status, charts))

    def _fill(self, variables, values, steps, press, window, file_path, charts, timeout) -> list:
        def apply(dialog_name, variables, values):
            def go():
                dlg = self._window(dialog_name)
                if variables:
                    driver.fill_variables(dlg, variables)
                for name, value in (values or {}).items():
                    driver.set_value(driver.find_field(dlg, name, self.elements), value)
            return go

        status = "done"
        if variables or values:
            status = self.gui.interact(apply(window, variables, values), timeout)
        for step in steps or []:
            unknown = set(step) - {"open", "values", "variables", "press"}
            if unknown:
                raise ValueError(f"Unknown keys in a step: {', '.join(sorted(unknown))}.")
            opener = step.get("open", "")
            if opener:
                status = self.gui.interact(self._presser(opener, window), timeout)
            if step.get("variables") or step.get("values"):
                status = self.gui.interact(
                    apply("", step.get("variables"), step.get("values")), timeout)
            done = step.get("press", "Continue" if opener else "")
            if done:
                status = self.gui.interact(self._presser(done, ""), timeout)
        if file_path:
            def choose():
                dlg = self._window("")
                if not isinstance(dlg, QFileDialog):
                    raise ValueError("No file dialog is open.")
                driver.choose_file(dlg, str(_resolve(file_path)))
            status = self.gui.interact(choose, timeout)
        if press:
            status = self.gui.interact(self._presser(press, window), timeout)
        if status == "busy":
            return self._outcome(status)
        return self.gui.call(lambda: self._outcome(status, charts))

    def _presser(self, name: str, window: str):
        def go():
            popup = self._popup()
            if popup is not None:
                for act in popup.actions():
                    if driver.clean(act.text()).lower() == driver.clean(name).lower():
                        popup.close()
                        act.trigger()
                        return
            target = self._window(window)
            if isinstance(target, QFileDialog) and name.lower() == "cancel":
                target.reject()
                return
            button = driver.find_button(target, name, self.elements)
            if not button.isEnabled():
                raise ValueError(f"The button '{name}' is disabled: something required is "
                                 "missing in the dialog.")
            menu = button.menu() if hasattr(button, "menu") else None
            button.click()
            if menu is not None and not menu.isVisible():
                menu.popup(button.mapToGlobal(button.rect().bottomLeft()))
        return go

    @app_tool("Fill in a dialog", ui=True)
    def fill_dialog(self, variables: dict[str, Any] | None = None,
                    values: dict[str, Any] | None = None,
                    steps: list[dict[str, Any]] | None = None, press: str = "",
                    file_path: str = "", window: str = "", charts: bool = False,
                    timeout: int = 120) -> list:
        """Fill in the dialog (or the fields of a window) that is open and optionally press a
        button. Returns what is shown next: the next dialog, a message, or the output.

        Args:
            variables: Variable boxes: {box key or label: name or [names]} (pairs: [[a, b]]).
            values: Fields: {id or label: value}. Check boxes and radio buttons take true;
                choices take the text of the choice; lists take the item(s) to select.
            steps: Sub-dialogs, in order: [{"open": "Statistics...", "values": {...},
                "press": "Continue"}].
            press: Button to press at the end ("OK", "Cancel", "Yes", an id, or the item of
                an open pop-up menu).
            file_path: For a file dialog: full path to open or save.
            window: Another window's id or title; default is the dialog in front.
            charts: Also return the charts of the output as pictures.
            timeout: Seconds to wait for the computation.
        """
        return self._fill(variables, values, steps, press, window, file_path, charts, timeout)

    @app_tool("Wait for a computation", ui=True)
    def wait(self, timeout: int = 120, charts: bool = False) -> list:
        """Wait until a command that was "still computing" is finished and return its result.

        Args:
            timeout: Seconds to wait.
            charts: Also return the charts of the output as pictures.
        """
        status = self.gui.wait(timeout)
        if status == "busy":
            return self._outcome(status)
        return self.gui.call(lambda: self._outcome(status, charts))

    @app_tool("Read a window", read_only=True)
    def read_window(self, window: str = "") -> str:
        """Describe a window or dialog: fields with their values and choices, tables, buttons.
        The ids (e12) can be used in fill_dialog, read_table, write_table and take_screenshot.

        Args:
            window: Window id, title or module; default is the one in front.
        """
        target = self._window(window)
        return _json(driver.describe_window(target, self.elements))

    @app_tool("Close a window", ui=True)
    def close_window(self, window: str) -> list:
        """Close a window or dialog (like its close button). The app may ask whether to save
        changes; answer with fill_dialog.

        Args:
            window: Window id, title or module.
        """
        def go():
            target = self._window(window)
            if target is self.ctx._launcher and len(driver.open_windows()) == 1:
                raise ValueError("This is the last window.")
            target.close()
        return self._do(go)

    # ------------------------------------------------------------------ tables
    def _table(self, element: str, window: str) -> QAbstractItemView:
        widget = self.elements.get(element) if element else None
        if widget is None:
            target = self._window(element or window)
            views = [w for w in target.findChildren(QAbstractItemView)
                     if w.isVisible() and driver.table_size(w)[1] > 0
                     and not driver._inside_composite(w, target)]
            if element and self.elements.get(element) is None and not views:
                raise ValueError(f"There is no element '{element}'.")
            if not views:
                raise ValueError(f"'{driver.title_of(target)}' shows no table.")
            focus = target.focusWidget()
            widget = next((v for v in views if v is focus), None) or \
                max(views, key=lambda v: v.width() * v.height())
        if not isinstance(widget, QAbstractItemView):
            raise ValueError(f"The element '{element}' is not a table.")
        return widget

    @app_tool("Read a table", read_only=True)
    def read_table(self, element: str = "", window: str = "", first_row: int = 1,
                   rows: int = 30, first_column: int = 1, columns: int = 20) -> str:
        """Read cells of a table, list or worksheet shown in a window (data view, variable
        view, RSM design layout, graph worksheet, results table, tree).

        Args:
            element: Element id from read_window; default is the main table of the window.
            window: Window id, title or module when no element is given.
            first_row: First row to read (1 = first).
            rows: Number of rows (at most 500).
            first_column: First column to read (1 = first).
            columns: Number of columns (at most 100).
        """
        view = self._table(element, window)
        data = driver.read_cells(view, max(0, first_row - 1), min(max(rows, 1), 500),
                                 max(0, first_column - 1), min(max(columns, 1), 100))
        return _json({"element": self.elements.id_of(view), **data})

    @app_tool("Write into a table")
    def write_table(self, cells: list[list[Any]], element: str = "", window: str = "",
                    first_row: int = 1, first_column: int = 1) -> str:
        """Type values into the cells of an editable table (Data View, Variable View, RSM
        responses, graph worksheet, grids in dialogs), as a block of rows.

        Args:
            cells: Rows of values, for example [[1, 2.5], [2, 3.1]].
            element: Element id from read_window; default is the main table of the window.
            window: Window id, title or module when no element is given.
            first_row: Row of the first value (1 = first).
            first_column: Column of the first value (1 = first).
        """
        view = self._table(element, window)
        n = driver.write_cells(view, first_row - 1, first_column - 1, cells)
        return _json({"cells_set": n, "element": self.elements.id_of(view)})

    @app_tool("Select cells, rows or columns")
    def select_in_table(self, rows: list[int] | None = None, columns: list[int] | None = None,
                        element: str = "", window: str = "") -> str:
        """Select whole rows, whole columns, or a block of cells in a table. Commands of the
        graph module plot or analyse the selected columns.

        Args:
            rows: Row numbers (1 = first); omit to select whole columns.
            columns: Column numbers (1 = first); omit to select whole rows.
            element: Element id; default is the main table of the window.
            window: Window id, title or module when no element is given.
        """
        if rows is None and columns is None:
            raise ValueError("Give rows, columns or both.")
        view = self._table(element, window)
        driver.select_cells(view, None if rows is None else [r - 1 for r in rows],
                            None if columns is None else [c - 1 for c in columns])
        return _json({"selected": True, "element": self.elements.id_of(view)})

    # ------------------------------------------------------------------ pixels and input
    @app_tool("Picture of a window", read_only=True)
    def take_screenshot(self, window: str = "", element: str = "") -> list:
        """A picture of a window, dialog or element (graph, drawing canvas, image).

        Args:
            window: Window id, title or module; default is the one in front.
            element: Element id, to take only that part.
        """
        target = self.elements.get(element) if element else self._window(window)
        if target is None:
            raise ValueError(f"There is no element '{element}'.")
        size = f"{target.width()} x {target.height()}"
        return [f"{driver.title_of(target.window())} ({size} points)",
                Picture(driver.screenshot(target), driver.title_of(target.window()))]

    @app_tool("Mouse in a window", ui=True)
    def mouse(self, points: list[list[float]],
              action: Literal["click", "double_click", "move", "drag"] = "click",
              window: str = "", button: Literal["left", "right", "middle"] = "left",
              modifiers: list[Literal["shift", "ctrl", "alt"]] | None = None) -> list:
        """Use the mouse in a window, for drawing (chemical structures, ROIs), picking points
        on graphs, or anything that has no menu command. Coordinates are points from the
        top left corner of the window, in the size take_screenshot reports.

        Args:
            points: [[x, y]] for a click, [[x1, y1], [x2, y2], …] for a drag.
            action: click, double_click, move or drag.
            window: Window id, title or module; default is the one in front.
            button: Mouse button.
            modifiers: Keys held down.
        """
        spots = [(float(p[0]), float(p[1])) for p in points]
        return self._do(lambda: driver.mouse(self._window(window), spots, action, button,
                                             modifiers))

    @app_tool("Keyboard in a window", ui=True)
    def keyboard(self, text: str = "", keys: str = "", window: str = "") -> list:
        """Type text and/or press key combinations in the focused field or canvas.

        Args:
            text: Text to type.
            keys: Key combinations separated by spaces, for example "Ctrl+A Delete" or "Esc".
            window: Window id, title or module; default is the one in front.
        """
        return self._do(lambda: driver.type_keys(self._window(window), text, keys))

    # ------------------------------------------------------------------ output
    @app_tool("Read the output", read_only=True)
    def get_output(self, last: int = 1, charts: bool = False) -> list:
        """Read the latest results from the Output Viewer.

        Args:
            last: Number of results, counted from the end.
            charts: Also return their charts as pictures.
        """
        sections = self._sections()
        if not sections:
            return ["The Output Viewer is empty."]
        return self._parts(sections[-max(1, last):], charts)

    @app_tool("Save the output")
    def save_output(self, path: str, overwrite: bool = False) -> str:
        """Save everything in the Output Viewer to a file, by extension: .rso (opens in
        RS-StatChem), .docx (Word), .xlsx (Excel), .html or .pdf.

        Args:
            path: Full path of the file.
            overwrite: Must be true to replace an existing file. Ask the user first.
        """
        from rsstatchem.core import report
        from rsstatchem.core.output_store import save_output

        target = _output_path(path, overwrite, (".rso", ".docx", ".xlsx", ".html", ".pdf"))
        if not self._sections():
            raise ValueError("The Output Viewer is empty.")
        viewer = self.ctx.output()
        ext = target.suffix.lower()
        if ext == ".rso":
            save_output(viewer.doc, target, viewer.look)
        elif ext == ".docx":
            report.export_docx(viewer.doc, str(target), viewer.look)
        elif ext == ".xlsx":
            from rsstatchem.core.report_xlsx import export_xlsx

            export_xlsx(viewer.doc, str(target), "sheets", viewer.look)
        elif ext == ".html":
            target.write_text(report.document_html(viewer.doc), encoding="utf-8")
        else:
            self._output_pdf(viewer, target)
        return f"Saved {len(self._sections())} results to {target}"

    @staticmethod
    def _output_pdf(viewer, target: Path) -> None:
        from PySide6.QtCore import QMarginsF, QUrl
        from PySide6.QtGui import (QImage, QPageLayout, QPageSize, QPdfWriter, QTextDocument)

        from rsstatchem.core import report

        writer = QPdfWriter(str(target))
        writer.setPageSize(QPageSize(QPageSize.PageSizeId.A4))
        writer.setPageMargins(QMarginsF(15, 15, 15, 15), QPageLayout.Unit.Millimeter)
        doc = QTextDocument()
        for section in viewer.doc.sections:
            for item in section.items:
                if isinstance(item, Image):
                    doc.addResource(QTextDocument.ResourceType.ImageResource,
                                    QUrl(f"img://{item.id}"), QImage.fromData(item.png))
        doc.setHtml(report.document_html(viewer.doc, lambda img: f"img://{img.id}",
                                         standalone=False, look=viewer.look))
        doc.print_(writer)

    @app_tool("Open a file in the RS-StatChem window")
    def show_in_app(self, path: str) -> str:
        """Open a saved RS-StatChem file (.rsd data, .rso output, .rsx design, .rsop project,
        .rsg graph, .rsc drawing, image …) in the RS-StatChem program on the user's screen.

        Args:
            path: Full path of the file.
        """
        source = _resolve(path)
        if not source.exists():
            raise ValueError(f"File not found: {source}")
        if self.live:
            self.ctx.open_file(str(source))
            return f"Opened {source} in RS-StatChem."
        launch_app(source)
        return f"Asked the computer to open {source} in RS-StatChem."


def launch_app(path: Path) -> None:
    """Start the visible program with a file (a separate process from the connector)."""
    import subprocess

    env = {k: v for k, v in os.environ.items() if k != "QT_QPA_PLATFORM"}
    if getattr(sys, "frozen", False):
        command = [sys.executable, str(path)]
    else:
        command = [sys.executable, "-m", "rsstatchem", str(path)]
        root = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [root, env.get("PYTHONPATH")]))
    kwargs: dict = {"start_new_session": True} if sys.platform != "win32" else \
        {"creationflags": 0x00000008 | 0x00000200}   # detached, new process group
    subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, close_fds=True, **kwargs)


class Engine:
    """Runs tools by name: what the MCP server (background) and the live bridge call."""

    def __init__(self, ctx, live: bool = False) -> None:
        from rsstatchem.mcp import module_tools

        driver.release_static_dialogs()
        self.gui = driver.GuiThread()
        self.workspace = module_tools.FullWorkspace(ctx, self.gui, live)
        self._old_hook = sys.excepthook
        sys.excepthook = self._hook

    def _hook(self, kind, exc, tb) -> None:
        """An exception in a Qt slot never reaches the caller: keep it for the result."""
        text = "".join(traceback.format_exception(kind, exc, tb)[-3:]).strip()
        self.workspace.errors.append(text[-900:])
        self._old_hook(kind, exc, tb)

    def _direct(self, fn, args: dict):
        result = fn(**args)
        self.workspace._known_windows = {driver.key_of(w) for w in driver.open_windows()}
        return result

    def close(self) -> None:
        if sys.excepthook == self._hook:
            sys.excepthook = self._old_hook

    def call(self, name: str, args: dict) -> list:
        """Run a tool from any thread except Qt's; returns a list of str / Picture."""
        fn = getattr(self.workspace, name)
        ctx = self.workspace.ctx
        with file_dialogs_of_qt(self.gui):
            ctx.connector_calls += 1
            try:
                result = fn(**args) if name in UI_TOOLS else self.gui.call(
                    lambda: self._direct(fn, args))
            finally:
                ctx.connector_calls -= 1
        return result if isinstance(result, list) else [result]


class file_dialogs_of_qt:  # noqa: N801 - context manager
    """While a tool runs, dialogs are Qt's own (the system's file panel cannot be operated)."""

    depth = 0

    def __init__(self, gui: driver.GuiThread):
        self.gui = gui

    def _set(self, on: bool) -> None:
        from PySide6.QtCore import Qt

        QApplication.setAttribute(Qt.ApplicationAttribute.AA_DontUseNativeDialogs, on)

    def __enter__(self):
        cls = type(self)
        cls.depth += 1
        if cls.depth == 1:
            self.gui.call(lambda: self._set(True))
        return self

    def __exit__(self, *exc):
        cls = type(self)
        cls.depth -= 1
        if cls.depth == 0:
            try:
                self.gui.submit(lambda: self._set(False))
            except RuntimeError:
                pass
        return False


__all__ = ["APP_TOOLS", "AppWorkspace", "Engine", "Picture", "Table", "QMessageBox", "all_tools",
           "app_tool"]
