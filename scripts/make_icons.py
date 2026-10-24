"""Render the SVG logo/icon into the PNG, ICO and ICNS files used by the app and installers.

Run from the project root:  python scripts/make_icons.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

from PIL import Image
from PySide6.QtCore import QBuffer, QByteArray, QIODevice, Qt
from PySide6.QtGui import QGuiApplication, QImage, QPainter
from PySide6.QtSvg import QSvgRenderer

RES = Path(__file__).resolve().parents[1] / "rsstatchem" / "resources"


def render(svg: Path, width: int, height: int) -> Image.Image:
    renderer = QSvgRenderer(str(svg))
    image = QImage(width, height, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    renderer.render(painter)
    painter.end()
    data = QByteArray()
    buf = QBuffer(data)
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    image.save(buf, "PNG")
    return Image.open(io.BytesIO(bytes(data))).convert("RGBA")


def main() -> int:
    QGuiApplication(sys.argv)

    render(RES / "logo.svg", 1440, 720).save(RES / "logo.png")
    render(RES / "logo.svg", 720, 360).save(RES / "logo_small.png")

    icon = render(RES / "icon.svg", 1024, 1024)
    icon.resize((512, 512), Image.LANCZOS).save(RES / "icon.png")
    icon.save(RES / "icon.icns")
    icon.save(RES / "icon.ico", sizes=[(s, s) for s in (16, 24, 32, 48, 64, 128, 256)])
    print("Icons written to", RES)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
