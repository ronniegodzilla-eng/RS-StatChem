"""Stylesheet of the Graph & Analysis main window: a light-grey Windows desktop look like
OriginLab Origin (it overrides the app's red theme for this window and its dialogs)."""
from __future__ import annotations

from PySide6.QtGui import QColor, QPalette

ACCENT = "#0a64ad"
HOVER = "#e5f1fb"
HOVER_BORDER = "#7eb4ea"
CHECKED = "#cce4f7"
SELECT = "#b7d3f2"
CHROME = "#f0f0f0"
MDI_BACKGROUND = "#a0a0a0"

STYLESHEET = f"""
QMainWindow, QDialog {{ background: {CHROME}; }}
QMainWindow::separator {{ background: #d8d8d8; width: 4px; height: 4px; }}
QMenuBar {{ background: #ffffff; border-bottom: 1px solid #dadada; }}
QMenuBar::item {{ padding: 3px 8px; background: transparent; }}
QMenuBar::item:selected {{ background: {CHECKED}; color: #000; border: 1px solid {HOVER_BORDER}; }}
QMenu {{ background: #f9f9f9; border: 1px solid #a8a8a8; }}
QMenu::item {{ padding: 3px 26px 3px 26px; color: #000; }}
QMenu::item:selected {{ background: {CHECKED}; color: #000; }}
QMenu::item:disabled {{ color: #9c9c9c; }}
QMenu::separator {{ height: 1px; background: #d7d7d7; margin: 3px 4px 3px 26px; }}
QMenu::icon {{ padding-left: 4px; }}
QToolBar {{ background: {CHROME}; border: none; border-bottom: 1px solid #d9d9d9;
            spacing: 1px; padding: 1px 2px; }}
QToolBar::separator {{ background: #c8c8c8; width: 1px; height: 1px; margin: 3px 3px; }}
QToolBar::handle {{ background: transparent; width: 5px; height: 5px;
    image: none; border-left: 2px dotted #a8a8a8; margin: 3px 1px; }}
QToolBar QToolButton {{ border: 1px solid transparent; border-radius: 0px; padding: 2px;
    background: transparent; }}
QToolBar QToolButton:hover {{ background: {HOVER}; border-color: {HOVER_BORDER}; }}
QToolBar QToolButton:pressed, QToolBar QToolButton:checked {{ background: {CHECKED};
    border-color: {ACCENT}; }}
QToolBar QToolButton[popupMode="1"] {{ padding-right: 11px; }}
QToolBar QToolButton::menu-button {{ border: none; width: 10px; }}
QStatusBar {{ background: {CHROME}; border-top: 1px solid #cfcfcf; color: #000; }}
QStatusBar QLabel {{ padding: 0 6px; }}
QStatusBar QLabel#statusBox {{ border: 1px solid #bdbdbd; background: #fafafa; }}
QDockWidget {{ font-weight: normal; }}
QDockWidget::title {{ background: #dfe4ea; padding: 3px 6px; border: 1px solid #c3c9d1;
    text-align: left; }}
QMdiArea {{ background: {MDI_BACKGROUND}; }}
QHeaderView::section {{ background: #ececec; color: #000; border: none;
    border-right: 1px solid #c4c4c4; border-bottom: 1px solid #c4c4c4; padding: 2px 4px; }}
QHeaderView::section:checked {{ background: {SELECT}; color: #000; }}
QTableView {{ gridline-color: #d4d4d4; selection-background-color: {SELECT};
    selection-color: #000; border: none; background: #ffffff; }}
QTableView QTableCornerButton::section {{ background: #ececec; border: none;
    border-right: 1px solid #c4c4c4; border-bottom: 1px solid #c4c4c4; }}
QTabWidget::pane {{ border: none; }}
QTabWidget#sheetTabs QTabBar {{ background: #e6e6e6; }}
QTabWidget#sheetTabs QTabBar::tab {{ background: #e4e4e4; color: #000; padding: 2px 14px;
    border: 1px solid #a9a9a9; border-top: none; margin-right: -1px; min-width: 40px; }}
QTabWidget#sheetTabs QTabBar::tab:selected {{ background: #ffffff; font-weight: normal;
    border-bottom: 2px solid {ACCENT}; }}
QTabBar::tab {{ background: #e4e4e4; color: #000; padding: 3px 12px; border: 1px solid #b9b9b9;
    margin-right: -1px; }}
QTabBar::tab:selected {{ background: #ffffff; color: #000; font-weight: normal;
    border-bottom: 1px solid #ffffff; }}
QPushButton {{ background: #e1e1e1; border: 1px solid #adadad; border-radius: 0px;
    padding: 3px 12px; min-width: 64px; color: #000; }}
QPushButton:hover {{ background: {HOVER}; border-color: #0078d7; }}
QPushButton:default {{ border: 1px solid #0078d7; background: #e1e1e1; color: #000; }}
QPushButton:default:hover {{ background: {HOVER}; }}
QPushButton#arrow {{ min-width: 30px; max-width: 34px; padding: 3px 2px; }}
QPushButton:disabled {{ color: #a0a0a0; background: #f4f4f4; border-color: #d0d0d0; }}
QListWidget, QTreeWidget, QTextBrowser, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox,
QPlainTextEdit {{ border: 1px solid #b4b4b4; border-radius: 0px; background: #ffffff; }}
QListWidget::item:selected, QTreeWidget::item:selected {{ background: {SELECT}; color: #000; }}
QTreeWidget#peFolders, QTreeWidget#peWindows, QTreeWidget#objectTree {{ border: none; }}
QGroupBox {{ border: 1px solid #c8c8c8; border-radius: 0px; margin-top: 10px;
    padding-top: 6px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 3px; color: #000;
    font-weight: normal; }}
QPlainTextEdit#resultsLog, QPlainTextEdit#messagesLog {{ border: none;
    font-family: Consolas, Menlo, "Courier New", monospace; font-size: 9pt; }}
QSplitter::handle {{ background: #dcdcdc; }}
QTextBrowser#originReport {{ border: none; background: #ffffff; }}
"""


def palette(base: QPalette) -> QPalette:
    pal = QPalette(base)
    pal.setColor(QPalette.ColorRole.Window, QColor(CHROME))
    pal.setColor(QPalette.ColorRole.Highlight, QColor(SELECT))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor("#000000"))
    pal.setColor(QPalette.ColorRole.Link, QColor(ACCENT))
    return pal
