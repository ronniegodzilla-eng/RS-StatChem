"""Debugging aid: trigger every menu command of a module and report what breaks.

    QT_QPA_PLATFORM=offscreen python scripts/menu_crawler.py [stats rsm graph chem image]
                                                             [--json report.json]

Each module window gets demo content, then every enabled leaf action of its menu bar (and of
the extra menu states of the user guide) is triggered once. Modal dialogs are shown and
dismissed at once, file dialogs return nothing, message boxes are recorded, URLs are not
opened and the network is blocked, so a run needs no user and touches no files. Reported:

* exceptions raised by a command (with the traceback),
* error message boxes a command showed (often a missing guard),
* Qt warnings printed while it ran.

Nothing here is used by the app itself.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["RSSTATCHEM_HOME"] = tempfile.mkdtemp(prefix="rscrawl-")
os.environ["RSSTATCHEM_NO_SPLASH"] = "1"
os.environ["RSSTATCHEM_NO_UPDATE_CHECK"] = "1"

from PySide6.QtCore import QtMsgType, qInstallMessageHandler  # noqa: E402
from PySide6.QtGui import QDesktopServices  # noqa: E402
from PySide6.QtWidgets import (QApplication, QColorDialog, QDialog, QFileDialog,  # noqa: E402
                               QFontDialog, QInputDialog, QMenu, QMessageBox, QWizard)

SKIP = ("exit", "quit")          # would end the run
VERBOSE = False
events: list[dict] = []
current = {"module": "", "command": ""}


def note(kind: str, text: str) -> None:
    events.append({"module": current["module"], "command": current["command"], "kind": kind,
                   "text": text})


# ---------------------------------------------------------------------------- patches
def install_patches(app: QApplication) -> None:
    def fake_exec(self, *args, **kwargs):
        try:
            self.show()
            app.processEvents()
        finally:
            self.close()
        return 0

    for cls in (QDialog, QMessageBox, QFileDialog, QInputDialog, QColorDialog, QFontDialog,
                QWizard):
        cls.exec = fake_exec
    try:
        from PySide6.QtPrintSupport import QPageSetupDialog, QPrintDialog, QPrintPreviewDialog

        for cls in (QPrintDialog, QPageSetupDialog, QPrintPreviewDialog):
            cls.exec = fake_exec
    except ImportError:
        pass
    QMenu.exec = lambda self, *a, **k: None
    # dialogs that run their own local loop (Origin's roll-up dialogs) return at once
    from PySide6.QtCore import QEventLoop

    QEventLoop.exec = lambda self, *a, **k: 0

    def box(kind, answer):
        def call(parent=None, title="", text="", *args, **kwargs):
            if kind in ("warning", "critical"):
                note(kind, f"{title}: {text}")
            return answer
        return staticmethod(call)

    QMessageBox.information = box("information", QMessageBox.StandardButton.Ok)
    QMessageBox.warning = box("warning", QMessageBox.StandardButton.Ok)
    QMessageBox.critical = box("critical", QMessageBox.StandardButton.Ok)
    QMessageBox.question = box("question", QMessageBox.StandardButton.Cancel)
    QMessageBox.about = box("about", None)
    for name in ("getOpenFileName", "getSaveFileName"):
        setattr(QFileDialog, name, staticmethod(lambda *a, **k: ("", "")))
    QFileDialog.getOpenFileNames = staticmethod(lambda *a, **k: ([], ""))
    QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: "")
    for name in ("getText", "getMultiLineText", "getItem"):
        setattr(QInputDialog, name, staticmethod(lambda *a, **k: ("", False)))
    QInputDialog.getInt = staticmethod(lambda *a, **k: (0, False))
    QInputDialog.getDouble = staticmethod(lambda *a, **k: (0.0, False))
    QColorDialog.getColor = staticmethod(lambda *a, **k: __import__(
        "PySide6.QtGui").QtGui.QColor())
    QFontDialog.getFont = staticmethod(lambda *a, **k: (False, app.font()))
    QDesktopServices.openUrl = staticmethod(lambda url: True)

    import socket

    def no_network(*args, **kwargs):
        raise OSError("network disabled in the menu crawler")

    socket.create_connection = no_network

    def hook(exc_type, exc, tb):
        note("exception", "".join(traceback.format_exception(exc_type, exc, tb)))

    sys.excepthook = hook

    def qt_message(mode, context, message):
        if mode in (QtMsgType.QtWarningMsg, QtMsgType.QtCriticalMsg, QtMsgType.QtFatalMsg) \
                and "propagateSizeHints" not in message and "does not support" not in message:
            note("qt", message)

    qInstallMessageHandler(qt_message)


# ---------------------------------------------------------------------------- demo content
def demo(ctx, module: str):
    """A module window with something in it, so commands have data to act on."""
    import numpy as np
    import pandas as pd

    from rsstatchem.ui import modules

    rng = np.random.default_rng(7)
    if module == "stats":
        from rsstatchem.core.dataset import Dataset

        n = 60
        df = pd.DataFrame({
            "id": np.arange(1, n + 1.0), "grp": rng.integers(1, 4, n).astype(float),
            "sex": rng.integers(1, 3, n).astype(float), "x1": rng.normal(50, 10, n).round(2),
            "x2": rng.normal(20, 4, n).round(2), "x3": rng.normal(5, 1, n).round(2),
            "y": rng.normal(70, 8, n).round(2), "cnt": rng.poisson(3, n).astype(float),
            "flag": rng.integers(0, 2, n).astype(float),
            "txt": rng.choice(["a", "b", "c"], n)})
        df.loc[3, "x2"] = np.nan
        ds = Dataset(df)
        ds.var("grp").value_labels = {1.0: "A", 2.0: "B", 3.0: "C"}
        ds.var("sex").value_labels = {1.0: "L", 2.0: "P"}
        w = modules.open_module(ctx, "stats")
        w.load_dataset(ds)
        return w
    if module == "rsm":
        from rsstatchem.ui.dx.document import DesignDocument
        from rsstatchem.ui.dx.window import DXMainWindow

        doc = DesignDocument.from_spec({"design": "ccd", "randomize": False,
                                        "factors": [{"name": n} for n in "ABC"],
                                        "responses": [{"name": "Yield"}]})
        coded = doc.coded_matrix()
        y = 60 + 4 * coded[:, 0] - 3 * coded[:, 1] + 2 * coded[:, 0] * coded[:, 1] \
            - 2.5 * coded[:, 2] ** 2 + rng.normal(0, 0.6, len(coded))
        col = doc.responses[0]["column"]
        doc.ds.df[col] = np.round(y, 2)
        w = DXMainWindow(ctx, doc)
        ctx.register_window(w)
        w.show()
        return w
    if module == "graph":
        w = modules.open_module(ctx, "graph")
        book = next((b for b in getattr(w, "book_windows", lambda: [])()), None) \
            if callable(getattr(w, "book_windows", None)) else None
        try:
            bw = book or w.active_book_window()
            ws = bw.sheet()
            x = np.linspace(0, 10, 25)
            for r, (a, b) in enumerate(zip(x, 2.0 * x + 1 + rng.normal(0, 0.5, len(x)))):
                ws.set_cell(r, 0, float(a))
                ws.set_cell(r, 1, float(b))
            bw.refresh() if hasattr(bw, "refresh") else None
            bw.select_columns([1]) if hasattr(bw, "select_columns") else None
        except Exception:  # noqa: BLE001 - the crawl still runs on an empty workbook
            note("setup", traceback.format_exc())
        return w
    if module == "chem":
        w = modules.open_module(ctx, "chem")
        try:
            w.load_text("CC(=O)Oc1ccccc1C(=O)O")
            w.canvas.select_all() if hasattr(w.canvas, "select_all") else None
        except Exception:  # noqa: BLE001
            note("setup", traceback.format_exc())
        return w
    if module == "image":
        w = modules.open_module(ctx, "image")
        try:
            w.open_path(str(ROOT / "tests" / "data_imagej" / "inputs" / "blobs.gif"))
        except Exception:  # noqa: BLE001
            note("setup", traceback.format_exc())
        return w
    raise ValueError(module)


# ---------------------------------------------------------------------------- crawling
def leaf_actions(window):
    out = []

    def walk(menu, path):
        menu.aboutToShow.emit()
        for act in menu.actions():
            if act.isSeparator() or not act.isVisible():
                continue
            text = act.text().split("\t")[0].replace("&&", "\0").replace("&", "") \
                .replace("\0", "&").rstrip(".").strip()
            if not text:
                continue
            if act.menu() is not None:
                walk(act.menu(), path + [text])
            else:
                out.append((" > ".join(path + [text]), act))

    for top in window.menuBar().actions():
        if top.menu() is not None and top.isVisible():
            walk(top.menu(), [top.text().replace("&", "")])
    return out


def crawl(app, ctx, module: str) -> dict:
    current.update(module=module, command="(setup)")
    window = demo(ctx, module)
    app.processEvents()
    paths = [p for p, _ in leaf_actions(window)]
    stats = {"commands": len(paths), "triggered": 0, "disabled": 0, "skipped": 0}
    for path in paths:
        current["command"] = path
        if path.split(" > ")[-1].lower() in SKIP:
            stats["skipped"] += 1
            continue
        if window is None or not window.isVisible():
            # the previous command closed the window: start again with fresh content
            current["command"] = "(setup)"
            window = demo(ctx, module)
            app.processEvents()
            current["command"] = path
        action = dict(leaf_actions(window)).get(path)
        if action is None:
            continue                       # the menu changed (mode-dependent command)
        if not action.isEnabled():
            stats["disabled"] += 1
            continue
        before = set(map(id, app.topLevelWidgets()))
        if VERBOSE:
            print(f"  > {path}", file=sys.stderr, flush=True)
        try:
            action.trigger()
            for _ in range(3):
                app.processEvents()
        except Exception:  # noqa: BLE001
            note("exception", traceback.format_exc())
        stats["triggered"] += 1
        for w in app.topLevelWidgets():
            if id(w) not in before and w.isVisible() and w is not window \
                    and isinstance(w, (QDialog, QMenu)):
                w.close()
        app.processEvents()
    for w in list(app.topLevelWidgets()):
        try:
            if VERBOSE:
                print(f"  hide {type(w).__name__} {w.windowTitle()!r}", file=sys.stderr,
                      flush=True)
            w.hide()
        except RuntimeError:
            pass
    app.processEvents()
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("modules", nargs="*", default=["stats", "rsm", "graph", "chem", "image"])
    ap.add_argument("--json")
    ap.add_argument("-v", "--verbose", action="store_true", help="print each command first")
    args = ap.parse_args()
    global VERBOSE
    VERBOSE = args.verbose

    from rsstatchem.app import AppContext
    from rsstatchem.ui.theme import apply_theme

    app = QApplication(sys.argv[:1])
    apply_theme(app)
    ctx = AppContext(app)
    ctx.app.quit = lambda: None
    ctx.check_for_updates = lambda manual=False: None
    install_patches(app)
    summary = {}
    for module in args.modules:
        summary[module] = crawl(app, ctx, module)
        bad = [e for e in events if e["module"] == module and e["kind"] != "qt"]
        print(f"{module}: {summary[module]} -> {len(bad)} problem(s)", flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps({"summary": summary, "events": events},
                                              indent=1, ensure_ascii=False), encoding="utf-8")
    for e in events:
        if e["kind"] == "qt":
            continue
        head = e["text"].strip().splitlines()
        print(f"\n[{e['module']}] {e['command']} :: {e['kind']}\n  "
              + "\n  ".join(head[-6:] if e["kind"] == "exception" else head[:3]))
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)      # the windows of a crawl are in arbitrary states: skip Qt teardown
