"""Window stylesheet of the Chemical Drawing workspace: the light-gray, flat look of a
ChemDraw document window (overrides the app's red theme inside this window)."""
from __future__ import annotations

__all__ = ["STYLESHEET"]

ACCENT = "#0A64C8"
HOVER = "#E5F1FB"
CHECKED = "#CCE4F7"

STYLESHEET = f"""
QMainWindow {{ background: #E8E8E8; }}
QMainWindow::separator {{ background: #C9C9C9; width: 3px; height: 3px; }}
QMenuBar {{ background: #F3F3F3; border-bottom: 1px solid #CFCFCF; }}
QMenuBar::item {{ padding: 3px 9px; background: transparent; color: #1A1A1A; }}
QMenuBar::item:selected {{ background: {CHECKED}; color: #000000; }}
QMenu {{ background: #F8F8F8; border: 1px solid #A9A9A9; color: #1A1A1A; }}
QMenu::item {{ padding: 4px 30px 4px 28px; }}
QMenu::item:selected {{ background: {CHECKED}; color: #000000; }}
QMenu::item:disabled {{ color: #A0A0A0; }}
QMenu::separator {{ height: 1px; background: #D9D9D9; margin: 3px 6px; }}
QMenu::icon {{ padding-left: 4px; }}
QToolBar {{ background: #EFEFEF; border: none; border-bottom: 1px solid #CACACA;
    spacing: 1px; padding: 1px 2px; }}
QToolBar::separator {{ background: #C4C4C4; width: 1px; height: 1px; margin: 4px 3px; }}
QToolBar#chemdraw_main_toolbar {{ background: #E6E6E6; border: none;
    border-right: 1px solid #B4B4B4; padding: 0px; }}
QWidget#chemdraw_palette {{ background: #E6E6E6; }}
QToolBar QToolButton {{ border: 1px solid transparent; border-radius: 2px; padding: 2px;
    background: transparent; color: #1A1A1A; }}
QToolBar QToolButton:hover {{ background: {HOVER}; border-color: #9CC3EA; }}
QToolBar QToolButton:pressed, QToolBar QToolButton:checked {{ background: {CHECKED};
    border-color: #5C9BDB; }}
QToolButton::menu-indicator {{ image: none; width: 0px; }}
QMenu#chemdraw_flyout {{ background: #EDEDED; padding: 0px; }}
QWidget#chemdraw_flyout_strip {{ background: #EDEDED; }}
QWidget#chemdraw_flyout_strip QToolButton {{ border: 1px solid transparent; border-radius: 2px;
    background: transparent; }}
QWidget#chemdraw_flyout_strip QToolButton:hover {{ background: {HOVER};
    border-color: #9CC3EA; }}
QToolBar QComboBox, QToolBar QFontComboBox {{ min-height: 20px; }}
QStatusBar {{ background: #EDEDED; border-top: 1px solid #C6C6C6; color: #262626; }}
QStatusBar QLabel {{ color: #262626; }}
QDockWidget {{ color: #1A1A1A; titlebar-close-icon: none; }}
QDockWidget::title {{ background: #DCDCDC; padding: 4px 6px; border-bottom: 1px solid #BEBEBE;
    text-align: left; }}
QWidget#chemdraw_analysis, QWidget#chemdraw_properties {{ background: #FAFAFA; }}
QLabel#chemdraw_scope {{ color: #666666; }}
QLabel#chemdraw_note {{ color: #6A6A6A; font-size: 11px; }}
QLabel#chemdraw_value {{ color: #000000; }}
QPushButton {{ background: #F4F4F4; border: 1px solid #ADADAD; border-radius: 2px;
    padding: 3px 12px; color: #1A1A1A; }}
QPushButton:hover {{ background: {HOVER}; border-color: {ACCENT}; }}
QPushButton:default {{ background: #F4F4F4; color: #1A1A1A; border: 1px solid {ACCENT}; }}
QPushButton:default:hover {{ background: {HOVER}; }}
QPushButton:disabled {{ color: #A0A0A0; background: #F4F4F4; border-color: #D2D2D2; }}
QListWidget, QTreeWidget, QTextBrowser, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox,
QPlainTextEdit {{ background: #FFFFFF; border: 1px solid #ABABAB; border-radius: 1px;
    selection-background-color: {CHECKED}; selection-color: #000000; }}
QListWidget::item:selected, QTreeWidget::item:selected {{ background: {CHECKED};
    color: #000000; }}
QTableView {{ selection-background-color: {CHECKED}; selection-color: #000000; }}
QHeaderView::section {{ background: #F0F0F0; color: #1A1A1A; border: none;
    border-right: 1px solid #D5D5D5; border-bottom: 1px solid #D5D5D5; padding: 3px; }}
QTabBar::tab {{ background: #ECECEC; color: #333333; padding: 4px 14px;
    border: 1px solid #C8C8C8; border-bottom: none; }}
QTabBar::tab:selected {{ background: #FFFFFF; color: #000000; font-weight: normal;
    border-bottom: 2px solid {ACCENT}; }}
QGroupBox {{ border: 1px solid #CFCFCF; border-radius: 2px; margin-top: 10px;
    padding-top: 6px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; color: #1A1A1A; }}
QCheckBox {{ color: #1A1A1A; spacing: 5px; }}
QDialog {{ background: #F3F3F3; }}
"""
