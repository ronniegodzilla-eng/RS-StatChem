"""Red-and-white RS theme on top of Qt's Fusion style (identical look on macOS and Windows)."""
from __future__ import annotations

from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication

RED = "#C62828"
RED_DARK = "#8E1B1B"
RED_LIGHT = "#FBE3E3"
SELECTION = "#F6C9C7"
GRID = "#E3E3E3"
HEADER = "#F4F4F4"
TEXT = "#1B1B1B"
MUTED = "#8A8A8A"

STYLESHEET = f"""
QMainWindow, QDialog {{ background: #FFFFFF; }}
QToolBar {{ background: #FFFFFF; border: none; border-bottom: 2px solid {RED};
            spacing: 2px; padding: 3px 4px; }}
QToolBar QToolButton {{ border: 1px solid transparent; border-radius: 4px; padding: 3px; }}
QToolBar QToolButton:hover {{ background: {RED_LIGHT}; border-color: #F0B5B3; }}
QToolBar QToolButton:pressed, QToolBar QToolButton:checked {{
    background: {SELECTION}; border-color: {RED}; }}
QMenuBar {{ background: #FFFFFF; }}
QMenuBar::item:selected {{ background: {RED_LIGHT}; color: {TEXT}; }}
QMenu {{ background: #FFFFFF; border: 1px solid #CFCFCF; }}
QMenu::item {{ padding: 4px 24px 4px 22px; }}
QMenu::item:selected {{ background: {RED}; color: #FFFFFF; }}
QMenu::item:disabled {{ color: #A8A8A8; }}
QMenu::separator {{ height: 1px; background: #E2E2E2; margin: 3px 8px; }}
QHeaderView::section {{ background: {HEADER}; border: none; border-right: 1px solid {GRID};
    border-bottom: 1px solid #CFCFCF; padding: 3px 6px; color: #333; }}
QHeaderView::section:checked {{ background: {SELECTION}; color: {RED_DARK}; }}
QTableView {{ gridline-color: {GRID}; selection-background-color: {SELECTION};
    selection-color: {TEXT}; border: none; }}
QTableView QTableCornerButton::section {{ background: {HEADER}; border: none;
    border-right: 1px solid {GRID}; border-bottom: 1px solid #CFCFCF; }}
QTabWidget::pane {{ border: none; border-top: 1px solid #D5D5D5; }}
QTabBar::tab {{ background: #F2F2F2; color: #444; padding: 5px 18px; border: 1px solid #D5D5D5;
    border-top: none; margin-right: -1px; }}
QTabBar::tab:selected {{ background: #FFFFFF; color: {RED_DARK}; font-weight: 600;
    border-bottom: 3px solid {RED}; }}
QStatusBar {{ background: #F7F7F7; border-top: 1px solid #DDDDDD; color: #444; }}
QPushButton {{ background: #FFFFFF; border: 1px solid #BDBDBD; border-radius: 4px;
    padding: 4px 14px; min-width: 64px; }}
QPushButton:hover {{ border-color: {RED}; background: {RED_LIGHT}; }}
QPushButton:default {{ background: {RED}; color: #FFFFFF; border-color: {RED_DARK}; }}
QPushButton:default:hover {{ background: {RED_DARK}; }}
QPushButton:disabled {{ color: #A0A0A0; background: #F4F4F4; border-color: #D8D8D8; }}
QPushButton#arrow {{ min-width: 28px; max-width: 28px; padding: 4px; font-weight: bold;
    color: {RED_DARK}; }}
QListWidget, QTreeWidget, QTextBrowser, QLineEdit, QComboBox, QSpinBox, QPlainTextEdit {{
    border: 1px solid #CACACA; border-radius: 3px; background: #FFFFFF; }}
QListWidget::item:selected, QTreeWidget::item:selected {{ background: {SELECTION};
    color: {TEXT}; }}
QGroupBox {{ border: 1px solid #D6D6D6; border-radius: 4px; margin-top: 10px; padding-top: 6px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; color: {RED_DARK};
    font-weight: 600; }}
QSplitter::handle {{ background: #E6E6E6; }}
"""


def apply_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#FFFFFF"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Base, QColor("#FFFFFF"))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#FAFAFA"))
    palette.setColor(QPalette.ColorRole.Text, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Button, QColor("#F5F5F5"))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(RED))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#FFFFFF"))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("#FFFFFF"))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Link, QColor(RED))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(MUTED))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor(MUTED))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(MUTED))
    app.setPalette(palette)
    font = QFont(app.font())
    if font.pointSizeF() < 10:
        font.setPointSizeF(10)
    app.setFont(font)
    app.setStyleSheet(STYLESHEET)
