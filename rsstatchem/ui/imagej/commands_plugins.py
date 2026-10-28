"""Image Analysis commands: Plugins menu (Macros: Install, Run, Edit, Record; Shortcuts;
Utilities: Control Panel, Find Commands, Commands, Search, Capture, Benchmark, Reset; New:
Macro, Macro Tool, Text Window) and Process > Repeat Command. **Owned by WP-A.**

Macros run with :mod:`rsstatchem.core.imaging.macro` (ImageJ macro subset; ``.py`` files as
Python scripts with the ``ij`` facade).
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np

from rsstatchem.core.imaging import commands as C
from rsstatchem.core.imaging.commands import CommandError
from rsstatchem.core.imaging.image import ImageDoc
from rsstatchem.core.imaging.results import ResultsTable

SHORTCUT_KEYS = tuple(f"F{i}" for i in range(1, 13)) + tuple("0123456789") + \
    tuple("abcdefghijklmnopqrstuvwxyz")
MACRO_TEMPLATE = '// New macro\nprint("Hello, " + getTitle());\n'
TOOL_TEMPLATE = ('// Macro tool: runs when you click on the image\n'
                 'macro "Sample Tool - C00cT0f14S" {\n'
                 '    getCursorLoc(x, y, z, flags);\n'
                 '    print("Clicked at " + x + "," + y);\n}\n')


# ---------------------------------------------------------------------------------- repeat
def repeat_command(session, image):
    last = session.last_command
    if last is None:
        raise CommandError("There is no command to repeat")
    label, options = last
    return session.registry.run(session, label, options)


def _ui_repeat(bar, cmd):
    last = bar.session.last_command
    if last is None:
        bar.error("There is no command to repeat", "Repeat Command")
        return
    label, options = last
    bar.run_command(label, options)


# ---------------------------------------------------------------------------------- macros
def run_macro_cmd(session, image, macro="", argument=""):
    from rsstatchem.core.imaging.macro import MacroError, run_macro, run_python
    p = Path(macro)
    if not p.exists():
        raise CommandError(f"Macro or script not found:\n{macro}")
    code = p.read_text(errors="replace")
    try:
        if p.suffix.lower() == ".py":
            run_python(session, code, path=str(p))
        else:
            run_macro(session, code, argument, path=str(p))
    except MacroError as exc:
        raise CommandError(str(exc)) from None


def _ui_run(bar, cmd):
    path = bar.get_path("open", "Run Macro or Script...", bar.last_directory(),
                        "Macros and scripts (*.ijm *.txt *.py *.macro)")
    if path:
        bar.run_macro_file(path)


def _ui_install(bar, cmd):
    path = bar.get_path("open", "Install Macros", bar.last_directory(),
                        "Macros (*.ijm *.txt)")
    if path:
        bar.install_macros(Path(path).read_text(errors="replace"), path)


def install_cmd(session, image, macro=""):
    from rsstatchem.core.imaging.macro import macro_names
    names = macro_names(Path(macro).read_text(errors="replace"))
    session.options.extra["installed_macros"] = macro
    session.status(f"{len(names)} macros installed")


def _ui_edit(bar, cmd):
    path = bar.get_path("open", "Edit Macro...", bar.last_directory(),
                        "Macros and scripts (*.ijm *.txt *.py *.macro)")
    if path:
        p = Path(path)
        bar.session.emit("macro_opened", (p.name, p.read_text(errors="replace"), str(p)))


def _ui_record(bar, cmd):
    from rsstatchem.ui.imagej.frames.recorder import show_recorder
    return show_recorder(bar)


def _ui_new_macro(template: str, title: str):
    def action(bar, cmd):
        from rsstatchem.ui.imagej import windows as W
        from rsstatchem.ui.imagej.frames.text_window import TextWindow
        win = TextWindow(title, template, bar, macro=True)
        W.register(win)
        win.show()
        return win
    return action


def new_text_window(session, image, name="Untitled.txt", width=80, height=24, menu=True):
    session.emit("text_opened", (name, "", None))


# ---------------------------------------------------------------------------------- shortcuts
def _command_choices(session):
    return tuple(sorted(c.label for c in session.registry if c.available))


def add_shortcut(session, image, command, shortcut):
    sc = session.options.extra.setdefault("shortcuts", {})
    sc[shortcut] = command
    session.emit("shortcuts_changed", dict(sc))


def remove_shortcut(session, image, shortcut):
    sc = session.options.extra.setdefault("shortcuts", {})
    if shortcut not in sc:
        raise CommandError(f'No shortcut "{shortcut}" is installed')
    del sc[shortcut]
    session.emit("shortcuts_changed", dict(sc))


def list_shortcuts(session, image):
    from rsstatchem.core.imaging.menu_layout import iter_items, qt_shortcut
    rt = ResultsTable("Keyboard Shortcuts")
    for _, item in iter_items():
        if item.shortcut:
            rt.add_row({"Shortcut": qt_shortcut(item), "Command": item.label})
    from rsstatchem.core.imaging.menu_layout import PLAIN_KEYS
    for key, label in PLAIN_KEYS.items():
        rt.add_row({"Shortcut": key, "Command": label})
    for key, label in session.options.extra.get("shortcuts", {}).items():
        rt.add_row({"Shortcut": key, "Command": label + " (user)"})
    return rt


def list_commands(session, image):
    from rsstatchem.ui.imagej.frames.command_finder import command_rows
    rt = ResultsTable("Commands")
    for label, path, sc, ok in command_rows(session.registry):
        rt.add_row({"Command": label, "Menu Path": path, "Shortcut": sc,
                    "Available": "yes" if ok else "no"})
    return rt


# ---------------------------------------------------------------------------------- utilities
def _ui_find(bar, cmd):
    from rsstatchem.ui.imagej.frames.command_finder import show_command_finder
    return show_command_finder(bar)


def _ui_control_panel(bar, cmd):
    """Plugins > Utilities > Control Panel: the menu tree; double-click runs a command."""
    from PySide6.QtWidgets import QMainWindow, QTreeWidget, QTreeWidgetItem

    from rsstatchem.core.imaging.menu_layout import MENU_LAYOUT, Menu
    from rsstatchem.ui.imagej import windows as W

    win = QMainWindow()
    win.setWindowTitle("Control Panel")
    tree = QTreeWidget()
    tree.setHeaderLabels(["Command"])

    def fill(parent, items):
        for node in items:
            if node is None:
                continue
            if isinstance(node, Menu):
                it = QTreeWidgetItem(parent, [node.title])
                fill(it, node.items)
            else:
                it = QTreeWidgetItem(parent, [node.label])
                it.setData(0, 256, node.label)
    for top in MENU_LAYOUT:
        it = QTreeWidgetItem(tree, [top.title])
        fill(it, top.items)
    tree.itemDoubleClicked.connect(
        lambda item, col: item.data(0, 256) and bar.run_command(item.data(0, 256)))
    win.setCentralWidget(tree)
    win.resize(320, 500)
    W.register(win)
    win.show()
    win.closeEvent = lambda e: (W.unregister(win), e.accept())
    return win


def _ui_commands(bar, cmd):
    """Plugins > Utilities > Commands...: the recently used commands, click to rerun."""
    from PySide6.QtWidgets import QListWidget, QMainWindow

    from rsstatchem.ui.imagej import windows as W
    win = QMainWindow()
    win.setWindowTitle("Commands")
    lst = QListWidget()
    recent = list(dict.fromkeys(reversed(bar.session.__dict__.get("command_history", []))))
    lst.addItems(recent[:25] or ["Blobs", "Open...", "Find Commands..."])
    lst.itemDoubleClicked.connect(lambda item: bar.run_command(item.text()))
    win.setCentralWidget(lst)
    win.resize(240, 360)
    W.register(win)
    win.show()
    win.closeEvent = lambda e: (W.unregister(win), e.accept())
    return win


def search_cmd(session, image, search="", directory="", case=False):
    """Plugins > Utilities > Search...: lines containing the text in macro/script/text files
    of a folder, written to the Log."""
    if not search:
        raise CommandError("Search text required")
    root = Path(directory) if directory else Path.cwd()
    flags = 0 if case else re.IGNORECASE
    rx = re.compile(re.escape(search), flags)
    n = 0
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in (".ijm", ".txt", ".py", ".java", ".js", ".macro"):
            continue
        try:
            for i, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
                if rx.search(line):
                    session.log(f"{p}:{i}: {line.strip()}")
                    n += 1
        except OSError:
            continue
    session.log(f"{n} matches found")


def capture_image(session, image):
    """Plugins > Utilities > Capture Image: the image as displayed (LUT, overlay) as RGB."""
    from rsstatchem.core.imaging.lut import render
    rgb = render(image).copy()
    try:
        from rsstatchem.ui.imagej.commands_image import flatten
        rgb = flatten(image)
    except Exception:  # noqa: BLE001 - overlay rendering is optional here
        pass
    return ImageDoc.from_array(rgb, "YXS", "Capture of " + image.title.rsplit(".", 1)[0])


def capture_screen(session, image, delay=0.0):
    from PySide6.QtGui import QGuiApplication, QImage
    if delay:
        time.sleep(min(float(delay), 30.0))
    screen = QGuiApplication.primaryScreen()
    if screen is None:
        raise CommandError("No screen to capture")
    qimg = screen.grabWindow(0).toImage().convertToFormat(QImage.Format.Format_RGB888)
    w, h = qimg.width(), qimg.height()
    if w == 0 or h == 0:
        raise CommandError("The screen could not be captured")
    arr = np.frombuffer(qimg.constBits(), np.uint8, qimg.bytesPerLine() * h).reshape(h, -1)
    return ImageDoc.from_array(arr[:, :w * 3].reshape(h, w, 3).copy(), "YXS", "Screenshot")


def benchmark(session, image):
    """Plugins > Utilities > Benchmark: times a fixed set of operations (log)."""
    from scipy import ndimage
    rng = np.random.default_rng(1)
    a = rng.integers(0, 255, (1024, 1024)).astype(np.uint8)
    tests = {"invert": lambda: 255 - a, "smooth 3x3": lambda: ndimage.uniform_filter(a, 3),
             "median 3x3": lambda: ndimage.median_filter(a, 3),
             "gaussian 2": lambda: ndimage.gaussian_filter(a.astype(np.float32), 2),
             "rotate 15": lambda: ndimage.rotate(a, 15, reshape=False, order=1),
             "histogram": lambda: np.bincount(a.ravel(), minlength=256)}
    total = 0.0
    for name, fn in tests.items():
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        total += dt
        session.log(f"{name}: {dt * 1000:.1f} ms")
    session.log(f"Benchmark total: {total:.3f} seconds (1024x1024 8-bit)")


def reset_utilities(session, image, undo=True, clipboard=True):
    if undo:
        for im in session.images:
            im.forget_undo()
    if clipboard:
        session.clipboard = None
    session.status("Undo buffers and clipboard reset")


def register(registry) -> None:
    add = registry.add
    ui = {"needs_image": False, "undoable": False}
    add("Repeat Command", repeat_command, undoable=False, recordable=False, needs_image=False,
        ui_action=_ui_repeat)
    macro_field = [C.string("Macro:", "", key="macro", columns=30),
                   C.string("Argument:", "", key="argument")]
    add("Install...", install_cmd, macro_field[:1], recordable=False, ui_action=_ui_install,
        **ui)
    add("Run...", run_macro_cmd, macro_field, ui_action=_ui_run, **ui)
    add("Edit...", None, recordable=False, ui_action=_ui_edit, **ui)
    add("Record...", None, recordable=False, ui_action=_ui_record, **ui)
    add("Add Shortcut... ", add_shortcut, lambda s, im: [
        C.choice("Command:", _command_choices(s) or ("Blobs",)),
        C.choice("Shortcut:", SHORTCUT_KEYS, "F1")], **ui)
    add("Add Shortcut by Name... ", add_shortcut, [
        C.string("Command:", "", columns=24), C.choice("Shortcut:", SHORTCUT_KEYS, "F1")],
        **ui)
    add("Remove Shortcut...", remove_shortcut, lambda s, im: [
        C.choice("Shortcut:", tuple(s.options.extra.get("shortcuts", {})) or ("F1",))], **ui)
    add("List Shortcuts", list_shortcuts, recordable=False, **ui)
    add("List Commands", list_commands, recordable=False, **ui)
    add("Control Panel...", None, recordable=False, ui_action=_ui_control_panel, **ui)
    add("Find Commands...", None, recordable=False, ui_action=_ui_find, **ui)
    add("Commands...", None, recordable=False, ui_action=_ui_commands, **ui)
    add("Search...", search_cmd, [C.string("Search:", "", columns=24),
                                  C.Field("directory", "Folder:", "", key="directory"),
                                  C.checkbox("Case sensitive", False, key="case")], **ui)
    add("Capture Screen", capture_screen, **ui)
    add("Capture Delayed...", capture_screen, [C.number("Delay (seconds):", 3, 0,
                                                         key="delay")], **ui)
    add("Capture Image", capture_image, undoable=False)
    add("Benchmark", benchmark, recordable=False, **ui)
    add("Reset...", reset_utilities, [C.checkbox("Undo buffers", True, key="undo"),
                                      C.checkbox("Internal clipboard", True,
                                                 key="clipboard")], **ui)
    add("Macro", None, recordable=False, ui_action=_ui_new_macro(MACRO_TEMPLATE, "Macro.ijm"),
        **ui)
    add("Macro Tool", None, recordable=False,
        ui_action=_ui_new_macro(TOOL_TEMPLATE, "Macro_Tool.ijm"), **ui)
    add("Text Window...", new_text_window, [C.string("Name:", "Untitled.txt", columns=16),
                                            C.number("Width:", 80, 0, "characters",
                                                     integer=True),
                                            C.number("Height:", 24, 0, "lines", integer=True),
                                            C.checkbox("Menu bar", True, key="menu")], **ui)


def selftest() -> None:
    from rsstatchem.core.imaging.macro import run_macro
    from rsstatchem.core.imaging.session import Session
    s = Session()
    register(s.registry)
    run_macro(s, 'newImage("m", "8-bit ramp", 16, 8, 1); a = newArray(1, 2, 3);\n'
                 'for (i = 0; i < a.length; i++) setPixel(i, 0, a[i] * 10);\n'
                 'print("v=" + getPixel(2, 0) + " " + d2s(1/3, 3));')
    assert s.log_lines[-1] == "v=30 0.333", s.log_lines
