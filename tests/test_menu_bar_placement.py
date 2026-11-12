"""On macOS the menu bars stay inside the windows (as on Windows and in the user guide)."""
import sys

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

ATTR = Qt.ApplicationAttribute.AA_DontUseNativeMenuBar


@pytest.fixture
def window_menu_bars(qapp):
    from rsstatchem.app import use_window_menu_bars

    before = QApplication.testAttribute(ATTR)
    use_window_menu_bars()
    yield
    QApplication.setAttribute(ATTR, before)


def test_menu_bars_are_not_native(window_menu_bars, qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("RSSTATCHEM_HOME", str(tmp_path))
    from rsstatchem.app import AppContext
    from rsstatchem.ui import modules

    assert QApplication.testAttribute(ATTR)
    ctx = AppContext(qapp)
    monkeypatch.setattr(ctx.app, "quit", lambda: None)
    for key in ("stats", "rsm", "graph", "chem", "image"):
        w = modules.open_module(ctx, key)
        assert not w.menuBar().isNativeMenuBar(), key
        assert len([a for a in w.menuBar().actions() if a.menu()]) >= 8, key


def test_image_windows_get_no_menu_strip(window_menu_bars, qapp, tmp_path, monkeypatch):
    """ImageJ's menus live on the main bar only; with the system menu bar (macOS default of
    Qt) an image window carried a copy so the global bar kept them."""
    monkeypatch.setenv("RSSTATCHEM_HOME", str(tmp_path))
    monkeypatch.setattr(sys, "platform", "darwin")
    from rsstatchem.app import AppContext
    from rsstatchem.core.imaging.image import ImageDoc
    from rsstatchem.ui import modules
    from rsstatchem.ui.imagej import windows as W

    ctx = AppContext(qapp)
    monkeypatch.setattr(ctx.app, "quit", lambda: None)
    bar = modules.open_module(ctx, "image")
    assert bar.image_window_menubar(bar) is None
    bar.session.add_image(ImageDoc.from_array(np.zeros((32, 32), np.uint8), "YX", "t"))
    qapp.processEvents()
    images = [w for w in W.WINDOWS if type(w).__name__ == "ImageWindow"]
    assert images and all(w.menuWidget() is None for w in images)
