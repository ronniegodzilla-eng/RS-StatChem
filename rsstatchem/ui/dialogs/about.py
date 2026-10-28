from __future__ import annotations

import platform

from PySide6 import __version__ as pyside_version
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QVBoxLayout

from rsstatchem import APP_NAME, COPYRIGHT, __version__
from rsstatchem.paths import resource, user_data_dir


class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"About {APP_NAME}")
        logo = QLabel()
        pm = QPixmap(str(resource("logo.png")))
        logo.setPixmap(pm.scaledToWidth(360, Qt.TransformationMode.SmoothTransformation))
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)

        import numpy
        import pandas
        import scipy
        import statsmodels

        info = QLabel(
            f"<h3 style='color:#B71C1C;margin-bottom:2px'>{APP_NAME} {__version__}</h3>"
            "Statistics · Response Surface Methodology · Scientific Graphing · Chemistry"
            "<br><br>"
            f"Python {platform.python_version()} ({platform.machine()}) · Qt/PySide6 "
            f"{pyside_version}<br>pandas {pandas.__version__} · NumPy {numpy.__version__} · "
            f"SciPy {scipy.__version__} · statsmodels {statsmodels.__version__}<br><br>"
            f"<span style='color:#666'>Settings folder: {user_data_dir()}</span><br><br>"
            f"<b>Copyright {COPYRIGHT}</b><br>"
            "<span style='color:#666'>Licensed under the PolyForm Noncommercial License "
            "1.0.0: free for personal, research and educational use.</span>")
        info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info.setWordWrap(True)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(self.accept)
        layout = QVBoxLayout(self)
        layout.addWidget(logo)
        layout.addWidget(info)
        layout.addWidget(buttons)
        self.setFixedWidth(480)
