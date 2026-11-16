"""Debugging aid: look for small display faults in every window and dialog of every module.

    python scripts/ui_lint.py [stats rsm graph chem image] [--shots DIR] [--json report.json]

Runs the menu crawler (scripts/menu_crawler.py: demo content, every menu command triggered
once, dialogs shown and dismissed) and checks each window and dialog the moment it is shown:

* clipped text: a label, button, check box, radio button or group box title narrower than its
  own text (the end of the text is cut off),
* a combo box whose list has an item without text, or whose field is too narrow for the
  chosen entry,
* a spin box too narrow for the numbers it accepts,
* a toolbar button without an icon among buttons with icons,
* a child widget that sticks out of its window,
* a window or dialog without a title, or taller/wider than a 1366 x 768 screen.

``--shots DIR`` also saves a picture of every dialog (``module__command.png``) for a visual
pass. Nothing here is used by the app itself.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# a 1366 x 768 laptop screen (the offscreen default is 800 x 600, which squeezes big windows)
_SCREEN_JSON = Path(__import__("tempfile").mkdtemp(prefix="uilint-")) / "screen.json"
_SCREEN_JSON.write_text('{"synchronousWindowSystemEvents": true, "windowFrameMargins": false, '
                        '"screens": [{"name": "lint", "x": 0, "y": 0, "width": 1366, '
                        '"height": 768, "logicalDpi": 96, "logicalBaseDpi": 96, "dpr": 1}]}',
                        encoding="utf-8")
os.environ.setdefault("QT_QPA_PLATFORM", f"offscreen:configfile={_SCREEN_JSON}")
import menu_crawler as crawler  # noqa: E402  (offscreen platform, temp home, patches)

from PySide6.QtWidgets import (QAbstractButton, QAbstractSpinBox, QApplication,  # noqa: E402
                               QCheckBox, QComboBox, QDialog, QGroupBox, QLabel, QLineEdit,
                               QMainWindow,
                               QPushButton, QRadioButton, QSpinBox, QStyle,
                               QStyleOptionButton, QToolBar, QToolButton, QWidget)

SCREEN = (1366, 768)
SLACK = 5          # px of overflow ignored (rounding, focus frames)
findings: dict[tuple, dict] = {}
SHOTS: Path | None = None
_seen_windows: set[int] = set()


def where(w: QWidget) -> str:
    names = []
    while w is not None and not QWidget.isWindow(w):
        names.append(w.objectName() or type(w).__name__)
        w = QWidget.parentWidget(w)
    return "/".join(reversed(names[:3]))


def report(window: QWidget, kind: str, widget: QWidget | None, detail: str) -> None:
    title = window.windowTitle().replace("[*]", "").strip() or type(window).__name__
    key = (crawler.current["module"], title, kind, detail)
    if key not in findings:
        findings[key] = {"module": crawler.current["module"], "command": crawler.current["command"],
                         "window": title, "kind": kind, "widget": where(widget) if widget else "",
                         "detail": detail}


def _text_of(w) -> str:
    return w.text().replace("&&", "\0").replace("&", "").replace("\0", "&")


def check(window: QWidget) -> None:
    if window is None or not window.isVisible():
        return
    QApplication.processEvents()
    title = window.windowTitle().replace("[*]", "").strip()
    if not title and isinstance(window, (QDialog, QMainWindow)):
        report(window, "no title", None, type(window).__name__)
    if window.width() > SCREEN[0] or window.height() > SCREEN[1] - 32:
        report(window, "too large for a 1366x768 screen", None,
               f"{window.width()} x {window.height()}")
    win_rect = window.rect()
    for w in window.findChildren(QWidget):
        if not w.isVisibleTo(window) or w.width() <= 1 or w.height() <= 1:
            continue
        if QWidget.window(w) is not window:
            continue
        fm = w.fontMetrics()
        if isinstance(w, QLabel) and not w.wordWrap() and w.textFormat() != 1 \
                and w.pixmap().isNull() and _text_of(w).strip() and "\n" not in w.text() \
                and not re.search(r"<[a-zA-Z/][^>]*>", w.text()):
            need = fm.horizontalAdvance(_text_of(w)) + 2 * w.margin() + w.indent() \
                if w.indent() > 0 else fm.horizontalAdvance(_text_of(w)) + 2 * w.margin()
            if need > w.contentsRect().width() + SLACK:
                report(window, "clipped label", w,
                       f"{_text_of(w)[:60]!r} needs {need}px, has {w.contentsRect().width()}px")
        elif isinstance(w, (QPushButton, QCheckBox, QRadioButton)) and _text_of(w).strip():
            hint = w.sizeHint().width()
            if isinstance(w, QPushButton) and w.icon().isNull() is False and not w.text():
                continue
            if hint > w.width() + SLACK:
                report(window, f"clipped {type(w).__name__}", w,
                       f"{_text_of(w)[:60]!r} needs {hint}px, has {w.width()}px")
        elif isinstance(w, QGroupBox) and w.title():
            need = fm.horizontalAdvance(w.title().replace("&", "")) + 24
            if need > w.width():
                report(window, "clipped group title", w, f"{w.title()!r}")
        if isinstance(w, QLineEdit) and not w.isReadOnly() and w.width() < 60 \
                and not isinstance(w.parentWidget(), (QAbstractSpinBox, QComboBox)):
            report(window, "text box too narrow to type in", w,
                   f"{w.width()}px wide (placeholder {w.placeholderText()[:30]!r})")
        if isinstance(w, QComboBox) and w.count():
            blanks = [i for i in range(w.count()) if not w.itemText(i).strip()
                      and w.itemIcon(i).isNull()]
            if blanks and not w.isEditable():
                report(window, "combo item without text", w,
                       f"items {blanks[:5]} of {[w.itemText(i) for i in range(min(w.count(), 6))]}")
            text = w.currentText()
            if text and not w.isEditable():
                avail = w.style().subControlRect(
                    QStyle.ComplexControl.CC_ComboBox, _combo_opt(w),
                    QStyle.SubControl.SC_ComboBoxEditField, w).width()
                need = fm.horizontalAdvance(text) + 4
                if avail > 0 and need > avail + SLACK:
                    report(window, "combo too narrow for its value", w,
                           f"{text[:50]!r} needs {need}px, has {avail}px")
        if isinstance(w, QSpinBox) or (isinstance(w, QAbstractSpinBox)
                                       and hasattr(w, "maximum") and hasattr(w, "decimals")):
            longest = max((w.textFromValue(w.minimum()), w.textFromValue(w.maximum())),
                          key=len) if hasattr(w, "textFromValue") else ""
            longest = w.prefix() + longest + w.suffix()
            edit = w.lineEdit().contentsRect().width() if w.lineEdit() else w.width()
            if len(longest) <= 12 and fm.horizontalAdvance(longest) > edit + SLACK:
                report(window, "spin box too narrow", w,
                       f"{longest!r} needs {fm.horizontalAdvance(longest)}px, has {edit}px")
        if isinstance(w, QToolBar):
            buttons = [b for b in w.findChildren(QToolButton) if b.isVisibleTo(window)
                       and b.defaultAction() is not None and b.parentWidget() is w]
            with_icon = [b for b in buttons if not b.icon().isNull()]
            without = [b for b in buttons if b.icon().isNull() and b.text().strip()]
            if len(with_icon) >= 3 and without:
                report(window, "toolbar button without icon", w,
                        ", ".join(sorted({_text_of(b) for b in without})))
        if w.parentWidget() is window or (w.parentWidget() is not None
                                          and w.parentWidget().layout() is not None):
            r = w.geometry()
            parent = w.parentWidget()
            pr = parent.rect() if parent is not None else win_rect
            if parent is not None and isinstance(parent, (QDialog, QGroupBox)) and \
                    (r.right() > pr.right() + 2 or r.bottom() > pr.bottom() + 2):
                report(window, "widget outside its window", w,
                       f"{type(w).__name__} {r.getRect()} in {pr.width()}x{pr.height()}")


def _combo_opt(combo):
    from PySide6.QtWidgets import QStyleOptionComboBox

    opt = QStyleOptionComboBox()
    combo.initStyleOption(opt)
    return opt


def shoot(window: QWidget) -> None:
    if SHOTS is None:
        return
    name = re.sub(r"[^A-Za-z0-9]+", "_", f"{crawler.current['module']}__"
                  f"{crawler.current['command']}__{window.windowTitle()}")[:150]
    window.grab().save(str(SHOTS / f"{name}.png"))


def install(app: QApplication) -> None:
    """Hook into the crawler: check every dialog when shown, every main window after each
    command."""
    original_show_event = QDialog.showEvent

    def show_event(self, event):
        original_show_event(self, event)
        try:
            self.adjustSize() if False else None
            QApplication.processEvents()
            check(self)
            shoot(self)
        except RuntimeError:
            pass

    QDialog.showEvent = show_event
    original_trigger_crawl = crawler.crawl

    def crawl(app_, ctx, module):
        stats = original_trigger_crawl(app_, ctx, module)
        return stats

    crawler.crawl = crawl


def main() -> int:
    global SHOTS
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("modules", nargs="*", default=["stats", "rsm", "graph", "chem", "image"])
    ap.add_argument("--shots")
    ap.add_argument("--json")
    ap.add_argument("-v", "--verbose", action="store_true", help="print each command first")
    args = ap.parse_args()
    crawler.VERBOSE = args.verbose
    if args.shots:
        SHOTS = Path(args.shots)
        SHOTS.mkdir(parents=True, exist_ok=True)

    from rsstatchem.app import AppContext
    from rsstatchem.ui.theme import apply_theme

    app = QApplication(sys.argv[:1])
    apply_theme(app)
    ctx = AppContext(app)
    ctx.app.quit = lambda: None
    ctx.check_for_updates = lambda manual=False: None
    crawler.install_patches(app)
    install(app)

    # the main windows themselves, after the demo content is in
    original_demo = crawler.demo

    def demo(ctx_, module):
        w = original_demo(ctx_, module)
        QApplication.processEvents()
        crawler.current["command"] = "(main window)"
        try:
            check(w)
        except Exception:  # noqa: BLE001
            crawler.note("lint", __import__("traceback").format_exc())
        if SHOTS is not None:
            shoot(w)
        return w

    crawler.demo = demo
    for module in args.modules:
        stats = crawler.crawl(app, ctx, module)
        n = sum(1 for f in findings.values() if f["module"] == module)
        print(f"{module}: {stats['triggered']} commands, {n} finding(s)", flush=True)

    rows = sorted(findings.values(), key=lambda f: (f["module"], f["kind"], f["window"]))
    for f in rows:
        print(f"[{f['module']}] {f['kind']}: {f['window']} | {f['widget']} | {f['detail']}"
              f"   (from {f['command']})")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1, ensure_ascii=False),
                                   encoding="utf-8")
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)
