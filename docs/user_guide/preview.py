"""Render PDF pages to PNG with Qt's PDF module: pdfpages.py file.pdf outdir first last [scale]"""
import sys
from pathlib import Path

from PySide6.QtCore import QSize
from PySide6.QtGui import QGuiApplication
from PySide6.QtPdf import QPdfDocument

app = QGuiApplication([])
doc = QPdfDocument()
doc.load(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
first, last = int(sys.argv[3]), int(sys.argv[4])
scale = float(sys.argv[5]) if len(sys.argv) > 5 else 1.4
print("pages:", doc.pageCount())
for i in range(first - 1, min(last, doc.pageCount())):
    size = doc.pagePointSize(i)
    img = doc.render(i, QSize(int(size.width() * scale), int(size.height() * scale)))
    img.save(str(out / f"p{i + 1:03d}.png"))
