"""Display details: drop-down highlight, the Values "..." button, windows that fit their contents
and the screen, decimal point in number boxes."""
import pandas as pd
import pytest
from PySide6.QtCore import QPoint, Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (QComboBox, QDialog, QDoubleSpinBox, QHBoxLayout, QPushButton,
                               QRadioButton, QVBoxLayout)

from rsstatchem.ui import theme


@pytest.fixture
def themed(qapp):
    theme.apply_theme(qapp)
    return qapp


def test_dropdown_shows_the_chosen_entry(themed):
    """The chosen entry of an open drop-down was white text on white (it looked empty)."""
    dlg = QDialog()
    combo = QComboBox(dlg)
    combo.addItems(["Mean ≠ test value", "Mean < test value", "Mean > test value"])
    QVBoxLayout(dlg).addWidget(combo)
    dlg.show()
    themed.processEvents()
    colors = {}

    def look():
        view = combo.view()
        popup = view.window()
        image = popup.grab().toImage()
        rect = view.visualRect(view.model().index(0, 0))
        y = view.mapTo(popup, rect.center()).y() * image.height() // max(popup.height(), 1)
        row = [image.pixelColor(x, y).name() for x in range(4, image.width() - 4, 3)]
        colors["background"] = max(set(row), key=row.count)
        colors["dark"] = sum(1 for c in row if sum(int(c[i:i + 2], 16) for i in (1, 3, 5)) < 300)
        combo.hidePopup()

    combo.showPopup()
    QTimer.singleShot(100, look)
    QTest.qWait(400)
    assert colors["background"] == theme.SELECTION.lower()
    assert colors["dark"] > 0          # the text is drawn in a dark colour on it


def test_analysis_dialog_grows_to_fit_a_long_option(themed):
    """The analysis dialogs set a minimum size (600 x 380), after which Qt stopped applying the
    layout's minimum: "Characterize Posterior Distribution" was cut off."""
    from rsstatchem.core.dataset import Dataset
    from rsstatchem.ui.dialogs.bayes import BayesOneSampleNormalDialog as Dialog

    dlg = Dialog(Dataset(pd.DataFrame({"x": [1.0, 2.0, 3.0]})))
    dlg.show()
    themed.processEvents()
    radios = [b for b in dlg.findChildren(QRadioButton) if b.isVisible()]
    assert any(b.text() == "Characterize Posterior Distribution" for b in radios)
    from rsstatchem.ui.theme import available_geometry

    need = dlg.layout().totalMinimumSize().width()
    avail = available_geometry(dlg).width()
    if need <= avail:            # everything fits on this screen: nothing is cut off
        for b in radios:
            assert b.width() >= b.sizeHint().width(), b.text()
    else:                        # a small test screen (800 px on Windows CI): as wide as it
        assert dlg.width() >= avail - 1


def test_buttons_keep_their_labels_in_a_crowded_row(themed):
    """The stylesheet's min-width let a row squeeze buttons below their labels
    ("pen JCAMP-DX." in NMR Prediction)."""
    dlg = QDialog()
    row = QHBoxLayout(dlg)
    buttons = [QPushButton(t) for t in ("Open JCAMP-DX...", "Clear Overlay",
                                        "Export JCAMP-DX...", "Copy Peak List")]
    for b in buttons:
        row.addWidget(b)
    dlg.resize(200, 60)
    dlg.show()
    themed.processEvents()
    for b in buttons:
        assert b.width() >= b.sizeHint().width(), b.text()


def test_window_taller_than_the_screen_is_reduced(themed):
    from rsstatchem.ui.theme import fit_to_screen

    dlg = QDialog()
    QVBoxLayout(dlg).addWidget(QPushButton("OK"))
    avail = dlg.screen().availableGeometry()
    dlg.resize(400, avail.height() + 300)
    fit_to_screen(dlg)
    assert dlg.height() <= avail.height()


def test_number_boxes_use_a_decimal_point(themed):
    box = QDoubleSpinBox()
    box.setDecimals(4)
    box.setValue(0.05)
    assert box.text() == "0.0500"


def test_values_cell_has_a_button_that_opens_value_labels(themed, tmp_path, monkeypatch):
    """Variable View > Values shows a "..." button (as in SPSS) on every variable row."""
    monkeypatch.setenv("RSSTATCHEM_HOME", str(tmp_path))
    import rsstatchem.ui.data_editor as de
    from rsstatchem.app import AppContext
    from rsstatchem.core.dataset import Dataset

    ctx = AppContext(themed)
    monkeypatch.setattr(ctx.app, "quit", lambda: None)
    editor = ctx.new_editor(Dataset(pd.DataFrame({"gender": [1.0, 2.0], "score": [7.5, 8.0]})))
    editor.tabs.setCurrentIndex(1)
    editor.resize(1100, 400)
    themed.processEvents()
    opened = []
    monkeypatch.setattr(de.ValueLabelsDialog, "exec", lambda self: opened.append(self) or 0)
    view = editor.var_view
    col = de.VariableTableModel.ATTR.index("value_labels")

    def click(row, near_right):
        rect = view.visualRect(view.model().index(row, col))
        x = rect.right() - 8 if near_right else rect.left() + 10
        QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton,
                         pos=QPoint(x, rect.center().y()))

    click(1, near_right=True)
    assert len(opened) == 1                       # the button opens Value Labels
    click(1, near_right=False)
    assert len(opened) == 1                       # a click on the text only selects the cell
    click(5, near_right=True)
    assert len(opened) == 1                       # empty rows have no button
    assert editor.values_delegate.has_button(0)
    assert not editor.values_delegate.has_button(2)


def test_bold_group_titles_fit_in_sub_dialogs(themed):
    """Explore > Plots cut its bold group titles ("Spread vs Level with Levene Tes"), and an
    explicit minimum on a group must not squeeze its contents ("Modified Breusch-Pagan")."""
    from PySide6.QtWidgets import QCheckBox, QGroupBox

    from rsstatchem.ui.dialogs.base import SubDialog
    from rsstatchem.ui.theme import group_title_width

    dlg = SubDialog("Explore: Plots", None, columns=2)
    dlg.radio(dlg.group("Spread vs Level with Levene Test"), "a", [("x", "None")], "x")
    dlg.radio(dlg.group("Power (Transformed)"), "b", [("y", "Natural log")], "y")
    dlg.check(dlg.group("Heteroskedasticity Tests"), "c", "Modified Breusch-Pagan test", False)
    dlg.check(dlg.group("Parameter estimates with robust standard errors"), "d", "HC3", False)
    dlg.resize(300, 200)
    dlg.show()
    themed.processEvents()
    for box in dlg.findChildren(QGroupBox):
        assert box.width() >= group_title_width(box), box.title()
    for cb in dlg.findChildren(QCheckBox):
        assert cb.width() >= cb.sizeHint().width(), cb.text()


def test_output_charts_are_sharp_on_retina():
    """Output Viewer charts are shown at Image.width logical px; the PNG must have at least
    twice as many pixels so a 2x screen does not blur them."""
    from PySide6.QtGui import QImage

    import numpy as np
    from rsstatchem.core.output import Image
    from rsstatchem.core.stats import charts

    png = charts.histogram(np.arange(20.0), "Histogram", "x", False)
    assert QImage.fromData(png).width() >= 2 * Image("t", png).width
