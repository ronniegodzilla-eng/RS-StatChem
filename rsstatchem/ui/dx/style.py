"""Look of the RSM module: Design-Expert's light blue-grey workspace instead of the app's red
SPSS theme. The stylesheet is set on the module's own windows and dialogs, so it overrides the
application-wide theme there only."""
from __future__ import annotations

BLUE = "#1F4E8C"        # titles, section headers
BLUE_DARK = "#163A69"
ACCENT = "#2E7BD6"      # selection, focus
SELECT = "#CFE3FA"
NAV_BG = "#F4F7FB"
HEADER = "#E8EEF6"
GRID = "#D3DBE6"
TEXT = "#1A1A1A"
MUTED = "#6B7480"
FACTOR_BG = "#F3F6FA"   # factor cells of the design layout (not editable)
RESPONSE_BG = "#FFFFFF"  # response cells (editable)
IGNORED = "#B9B9B9"

# Design-Expert's p-value colours (coefficients table / ANOVA): very significant, significant,
# marginal, not significant.
P_COLORS = ((0.01, "#D0021B"), (0.05, "#1E8C2E"), (0.10, "#1F5FD0"))


def p_color(p: float | None) -> str:
    if p is None:
        return TEXT
    for limit, color in P_COLORS:
        if p < limit:
            return color
    return TEXT


STYLESHEET = f"""
QMainWindow, QDialog {{ background: #FFFFFF; }}
QWidget {{ color: {TEXT}; }}
QMenuBar {{ background: #FFFFFF; border-bottom: 1px solid #DCE3EC; }}
QMenuBar::item {{ padding: 4px 10px; background: transparent; }}
QMenuBar::item:selected {{ background: {SELECT}; color: {TEXT}; }}
QMenu {{ background: #FFFFFF; border: 1px solid #B9C6D6; }}
QMenu::item {{ padding: 4px 26px 4px 24px; }}
QMenu::item:selected {{ background: {ACCENT}; color: #FFFFFF; }}
QMenu::item:disabled {{ color: #A4ABB4; }}
QMenu::separator {{ height: 1px; background: #E1E6ED; margin: 3px 8px; }}
QToolBar {{ background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #FFFFFF,
            stop:1 #EEF2F7); border: none; border-bottom: 1px solid #C9D3DF;
            spacing: 2px; padding: 2px 4px; }}
QToolBar QToolButton {{ border: 1px solid transparent; border-radius: 3px; padding: 2px 4px; }}
QToolBar QToolButton:hover {{ background: #E3EEFB; border-color: #9DBFE8; }}
QToolBar QToolButton:pressed, QToolBar QToolButton:checked {{ background: {SELECT};
    border-color: {ACCENT}; }}
QToolBar::separator {{ width: 1px; background: #C9D3DF; margin: 4px 4px; }}
QStatusBar {{ background: #EEF2F7; border-top: 1px solid #C9D3DF; color: #3A4552; }}
QSplitter::handle {{ background: #D7DEE8; }}
QTreeWidget#navTree {{ background: {NAV_BG}; border: none; border-right: 1px solid #C9D3DF;
    font-size: 10pt; outline: 0; }}
QTreeWidget#navTree::item {{ padding: 3px 2px; }}
QTreeWidget#navTree::item:selected {{ background: {ACCENT}; color: #FFFFFF; }}
QTreeWidget#navTree::item:hover:!selected {{ background: #E1ECF9; }}
QLabel#navTitle {{ background: {BLUE}; color: #FFFFFF; font-weight: 600; padding: 5px 8px; }}
QLabel#pageTitle {{ color: {BLUE}; font-size: 13pt; font-weight: 600; padding: 2px 0; }}
QLabel#hint {{ background: #FFF8DC; border: 1px solid #EBD98F; border-radius: 3px;
    padding: 5px 8px; color: #5A4A00; }}
QLabel#error {{ color: #B00020; }}
QFrame#tabStrip {{ background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #FAFCFE,
    stop:1 #E7EDF5); border-bottom: 1px solid #C2CEDC; }}
QToolButton#tabButton {{ border: 1px solid transparent; border-radius: 3px; padding: 3px 8px;
    color: #22303F; }}
QToolButton#tabButton:hover {{ background: #E3EEFB; border-color: #9DBFE8; }}
QToolButton#tabButton:checked {{ background: #FFFFFF; border: 1px solid {ACCENT};
    border-bottom: 3px solid {ACCENT}; color: {BLUE_DARK}; font-weight: 600; }}
QFrame#toolPanel {{ background: {NAV_BG}; border-right: 1px solid #C9D3DF; }}
QGroupBox {{ border: 1px solid #C9D3DF; border-radius: 3px; margin-top: 11px;
    padding-top: 6px; background: #FFFFFF; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; color: {BLUE};
    font-weight: 600; }}
QHeaderView::section {{ background: {HEADER}; border: none; border-right: 1px solid {GRID};
    border-bottom: 1px solid #B9C6D6; padding: 3px 5px; color: #22303F; }}
QTableView, QTableWidget {{ gridline-color: {GRID}; selection-background-color: {SELECT};
    selection-color: {TEXT}; border: 1px solid #C9D3DF; background: #FFFFFF; }}
QTableView QTableCornerButton::section {{ background: {HEADER}; border: none; }}
QListWidget, QTreeWidget, QTextBrowser, QPlainTextEdit, QLineEdit, QComboBox, QSpinBox,
QDoubleSpinBox {{ border: 1px solid #B9C6D6; border-radius: 2px; background: #FFFFFF;
    selection-background-color: {ACCENT}; }}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
    color: #A0A8B2; background: #F2F4F7; }}
QListWidget::item {{ padding: 4px 4px; }}
QListWidget::item:selected, QTreeWidget::item:selected {{ background: {ACCENT};
    color: #FFFFFF; }}
QPushButton {{ background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #FFFFFF,
    stop:1 #E9EEF5); border: 1px solid #A9B7C8; border-radius: 3px; padding: 4px 14px;
    min-width: 64px; }}
QPushButton:hover {{ border-color: {ACCENT}; background: #E3EEFB; }}
QPushButton:default {{ background: {ACCENT}; color: #FFFFFF; border-color: {BLUE}; }}
QPushButton:default:hover {{ background: {BLUE}; }}
QPushButton:disabled {{ color: #A0A8B2; background: #F2F4F7; border-color: #D5DBE3; }}
QPushButton#bigButton {{ font-size: 11pt; padding: 10px 18px; text-align: left; }}
QTabWidget::pane {{ border: 1px solid #C9D3DF; top: -1px; }}
QTabBar::tab {{ background: #EEF2F7; color: #33414F; padding: 4px 14px;
    border: 1px solid #C9D3DF; margin-right: -1px; }}
QTabBar::tab:selected {{ background: #FFFFFF; color: {BLUE_DARK}; font-weight: 600;
    border-bottom-color: #FFFFFF; }}
QSlider::groove:horizontal {{ height: 4px; background: #C9D3DF; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {ACCENT}; width: 12px; margin: -5px 0;
    border-radius: 6px; }}
QSlider::sub-page:horizontal {{ background: #8DB7EA; border-radius: 2px; }}
"""

# HTML for report panes (ANOVA, fit summary, ...): Design-Expert's compact grids.
REPORT_CSS = f"""
body {{ font-family: 'Segoe UI', Arial, Helvetica, sans-serif; font-size: 9pt; color: {TEXT};
       background: #FFFFFF; }}
p.title {{ color: {BLUE}; font-size: 11pt; font-weight: bold; margin: 12px 0 4px 0; }}
p.sub {{ color: #33414F; font-size: 9pt; font-weight: bold; margin: 10px 0 2px 0; }}
table.dx {{ border-collapse: collapse; margin: 0 0 2px 0; }}
table.dx th {{ background: {HEADER}; color: #22303F; font-weight: bold; padding: 3px 8px;
              border: 1px solid #B9C6D6; }}
table.dx td {{ padding: 2px 8px; border: 1px solid {GRID}; }}
p.foot {{ font-size: 8pt; color: #4A5562; margin: 2px 0 0 0; }}
p.note {{ margin: 3px 0; }}
p.warning {{ margin: 4px 0; color: #7A4F00; background: #FFF6D8; padding: 4px 8px; }}
p.annot {{ margin: 4px 0; color: #24435F; background: #EEF5FD; padding: 5px 8px;
           border-left: 3px solid {ACCENT}; }}
"""
