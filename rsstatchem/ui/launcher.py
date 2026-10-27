"""Start screen: pick one of the four modules (or reopen a recent file)."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QByteArray, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QIcon, QKeySequence, QPainter, QPixmap, QShortcut
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel,
                               QListWidget, QListWidgetItem, QPushButton, QVBoxLayout, QWidget)

from rsstatchem import APP_NAME, COPYRIGHT, __version__
from rsstatchem.paths import resource
from rsstatchem.ui.modules import MODULES, module_for_path, open_module, open_path

R, D, L = "#C62828", "#3A3A3A", "#F3C5C3"

TILE_ICONS = {
    "stats": f'<rect x="4" y="6" width="40" height="36" rx="3" fill="#fff" stroke="{D}" '
             f'stroke-width="2"/><path d="M4 15h40M17 6v36" stroke="{D}" stroke-width="1.5"/>'
             f'<rect x="22" y="30" width="5" height="8" fill="{R}"/>'
             f'<rect x="29" y="23" width="5" height="15" fill="{R}"/>'
             f'<rect x="36" y="19" width="5" height="19" fill="{R}"/>',
    "rsm": f'<path d="M6 34l14-8 22 6-14 9z" fill="{L}" stroke="{D}" stroke-width="1.6"/>'
           f'<path d="M6 34C10 16 16 10 20 12s8 14 22 20" fill="none" stroke="{R}" '
           f'stroke-width="2.4"/><path d="M11 30c4-9 7-12 9-11s5 7 13 11" fill="none" '
           f'stroke="{R}" stroke-width="1.6" stroke-dasharray="3 2"/>'
           f'<circle cx="20" cy="12" r="2.6" fill="{D}"/>',
    "graph": f'<path d="M7 5v36h36" fill="none" stroke="{D}" stroke-width="2"/>'
             f'<path d="M9 36c6-2 9-20 15-20s8 14 18-8" fill="none" stroke="{R}" '
             f'stroke-width="2.4"/>' + "".join(
                 f'<circle cx="{x}" cy="{y}" r="2.4" fill="{D}"/>'
                 for x, y in ((11, 33), (17, 24), (23, 17), (29, 20), (35, 22), (40, 12))),
    "chem": f'<path d="M24 6l14 8v16l-14 8-14-8V14z" fill="none" stroke="{D}" '
            f'stroke-width="2.2"/><path d="M24 11l10 6M14 29l10 6M14 17v12" fill="none" '
            f'stroke="{R}" stroke-width="2.2"/><path d="M38 14l7-4M24 38v6" stroke="{D}" '
            f'stroke-width="2.2"/><text x="33" y="10" font-size="8" font-family="Arial" '
            f'font-weight="bold" fill="{R}">OH</text>',
    "image": f'<rect x="4" y="7" width="40" height="34" rx="3" fill="#fff" stroke="{D}" '
             f'stroke-width="2"/><circle cx="17" cy="19" r="5" fill="{L}" stroke="{R}" '
             f'stroke-width="1.8"/><circle cx="31" cy="27" r="7" fill="{L}" stroke="{R}" '
             f'stroke-width="1.8"/><circle cx="15" cy="32" r="3.5" fill="{L}" stroke="{R}" '
             f'stroke-width="1.6"/><path d="M36 10l6 0M39 7v6" stroke="{D}" stroke-width="1.6"/>',
}


def tile_icon(key: str, size: int = 48) -> QPixmap:
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48">{TILE_ICONS[key]}</svg>'
    renderer = QSvgRenderer(QByteArray(svg.encode()))
    pix = QPixmap(size * 2, size * 2)
    pix.setDevicePixelRatio(2.0)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    renderer.render(painter, QRectF(0, 0, size, size))
    painter.end()
    return pix


class ModuleTile(QFrame):
    """A large clickable card for one module."""

    clicked = Signal(str)

    def __init__(self, info, number: int, last_used: bool, parent=None):
        super().__init__(parent)
        self.key = info.key
        self.setObjectName("tile")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setProperty("last", last_used)
        self.setToolTip(f"Open {info.title} ({number})")
        self.setAccessibleName(info.title)
        icon = QLabel()
        icon.setPixmap(tile_icon(info.icon, 52))
        icon.setFixedSize(60, 60)
        title = QLabel(info.title)
        title.setObjectName("tileTitle")
        style = QLabel(f"{info.style}-style workspace")
        style.setObjectName("tileStyle")
        text = QLabel(info.subtitle)
        text.setObjectName("tileText")
        text.setWordWrap(True)
        key = QLabel(str(number))
        key.setObjectName("tileKey")
        key.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight)

        words = QVBoxLayout()
        words.setSpacing(2)
        words.addWidget(title)
        words.addWidget(style)
        words.addSpacing(4)
        words.addWidget(text)
        words.addStretch(1)
        row = QHBoxLayout(self)
        row.setContentsMargins(16, 16, 12, 14)
        row.setSpacing(14)
        row.addWidget(icon, 0, Qt.AlignmentFlag.AlignTop)
        row.addLayout(words, 1)
        row.addWidget(key, 0, Qt.AlignmentFlag.AlignTop)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.rect().contains(
                event.position().toPoint()):
            self.clicked.emit(self.key)
        super().mouseReleaseEvent(event)


class LauncherWindow(QWidget):
    STYLE = f"""
    LauncherWindow {{ background: #FAFAFA; }}
    QLabel#appName {{ font-size: 22pt; font-weight: bold; color: #1B1B1B; }}
    QLabel#appSub {{ color: #6B6B6B; font-size: 10.5pt; }}
    QLabel#section {{ color: #6B6B6B; font-weight: bold; font-size: 9pt;
                      letter-spacing: 1px; }}
    QFrame#tile {{ background: #FFFFFF; border: 1px solid #E0E0E0; border-radius: 10px; }}
    QFrame#tile:hover {{ border: 1.5px solid {R}; background: #FFF8F8; }}
    QFrame#tile[last="true"] {{ border: 1.5px solid #E7A6A3; }}
    QLabel#tileTitle {{ font-size: 15pt; font-weight: bold; color: #1B1B1B; }}
    QLabel#tileStyle {{ color: {R}; font-size: 9pt; font-weight: bold; }}
    QLabel#tileText {{ color: #555555; font-size: 9.5pt; }}
    QLabel#tileKey {{ color: #B0B0B0; font-size: 9pt; }}
    QLabel#copyright {{ color: #8A8A8A; font-size: 9pt; }}
    QListWidget {{ border: 1px solid #E0E0E0; border-radius: 6px; background: #FFFFFF; }}
    QListWidget::item {{ padding: 4px 6px; }}
    """

    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.setWindowTitle(f"{APP_NAME} — Choose a Module")
        self.setStyleSheet(self.STYLE)
        self.resize(1080, 640)

        logo = QLabel()
        pix = QPixmap(str(resource("icon.png")))
        logo.setPixmap(pix.scaled(64, 64, Qt.AspectRatioMode.KeepAspectRatio,
                                  Qt.TransformationMode.SmoothTransformation))
        name = QLabel(APP_NAME)
        name.setObjectName("appName")
        sub = QLabel(f"Version {__version__}  ·  Choose the module you want to work in")
        sub.setObjectName("appSub")
        head_text = QVBoxLayout()
        head_text.setSpacing(0)
        head_text.addWidget(name)
        head_text.addWidget(sub)
        head = QHBoxLayout()
        head.setSpacing(14)
        head.addWidget(logo)
        head.addLayout(head_text, 1)

        last = str(ctx.settings.value("last_module", ""))
        grid = QGridLayout()
        grid.setSpacing(14)
        self.tiles: dict[str, ModuleTile] = {}
        for i, info in enumerate(MODULES):
            tile = ModuleTile(info, i + 1, info.key == last)
            tile.setMinimumSize(QSize(320, 128))
            tile.clicked.connect(self.choose)
            grid.addWidget(tile, i // 3, i % 3)
            self.tiles[info.key] = tile
            QShortcut(QKeySequence(str(i + 1)), self,
                      activated=lambda k=info.key: self.choose(k))

        recent_label = QLabel("RECENT FILES")
        recent_label.setObjectName("section")
        self.recent = QListWidget()
        self.recent.setMaximumHeight(130)
        self.recent.itemActivated.connect(self._open_recent)
        open_btn = QPushButton("Open File...")
        open_btn.clicked.connect(self.open_dialog)
        bottom_head = QHBoxLayout()
        bottom_head.addWidget(recent_label)
        bottom_head.addStretch(1)
        bottom_head.addWidget(open_btn)

        modules_label = QLabel("MODULES")
        modules_label.setObjectName("section")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 22)
        layout.setSpacing(10)
        layout.addLayout(head)
        layout.addSpacing(10)
        layout.addWidget(modules_label)
        layout.addLayout(grid)
        layout.addSpacing(8)
        layout.addLayout(bottom_head)
        layout.addWidget(self.recent)
        copyright_label = QLabel(f"Copyright {COPYRIGHT}")
        copyright_label.setObjectName("copyright")
        copyright_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(copyright_label)
        self._fill_recent()

    def _fill_recent(self) -> None:
        self.recent.clear()
        for path in self.ctx.recent_files():
            info = module_for_path(path)
            item = QListWidgetItem(f"{Path(path).name}    —    {Path(path).parent}")
            if info is not None:
                item.setIcon(QIcon(tile_icon(info.icon, 16)))
            item.setData(Qt.ItemDataRole.UserRole, path)
            item.setToolTip(path)
            self.recent.addItem(item)
        if not self.recent.count():
            empty = QListWidgetItem("No recent files")
            empty.setFlags(Qt.ItemFlag.NoItemFlags)
            self.recent.addItem(empty)

    def showEvent(self, event) -> None:
        self._fill_recent()
        super().showEvent(event)

    def choose(self, key: str) -> None:
        window = open_module(self.ctx, key)
        if window is not None:
            self.hide()

    def _open_recent(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.ItemDataRole.UserRole)
        if path and open_path(self.ctx, path) is not None:
            self.hide()

    def open_dialog(self) -> None:
        patterns = " ".join(f"*{ext}" for info in MODULES for ext in info.extensions)
        start = str(self.ctx.settings.value("last_dir", str(Path.home())))
        path, _ = QFileDialog.getOpenFileName(self, "Open File", start,
                                              f"RS-StatChem files ({patterns});;All files (*)")
        if path and open_path(self.ctx, path) is not None:
            self.hide()

    def closeEvent(self, event) -> None:
        event.accept()
        self.ctx.launcher_closed()
