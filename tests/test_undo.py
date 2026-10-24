import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from rsstatchem.core.dataset import Dataset  # noqa: E402
from rsstatchem.core.transform import Rule, compute, recode  # noqa: E402
from rsstatchem.ui.document import DatasetDocument  # noqa: E402

app = QApplication.instance() or QApplication([])


@pytest.fixture
def doc():
    ds = Dataset(pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "z"]}))
    return DatasetDocument(ds)


def test_cell_edit_undo_redo(doc):
    doc.set_value(0, 0, "10")
    assert doc.ds.get_value(0, 0) == 10
    assert doc.undo_stack.undoText() == "Edit Cell"
    doc.undo_stack.undo()
    assert doc.ds.get_value(0, 0) == 1
    doc.undo_stack.redo()
    assert doc.ds.get_value(0, 0) == 10


def test_same_value_is_not_recorded(doc):
    doc.set_value(0, 0, "1")
    assert doc.undo_stack.count() == 0


def test_growing_edit_is_undone_completely(doc):
    doc.set_value(5, 3, "7")
    assert (doc.ds.n_cases, doc.ds.n_vars) == (6, 4)
    doc.undo_stack.undo()
    assert (doc.ds.n_cases, doc.ds.n_vars) == (3, 2)
    assert doc.ds.names() == ["a", "b"]


def test_paste_clear_and_attributes(doc):
    doc.paste_block(1, 0, [["20", "q"], ["30", "r"]])
    doc.clear_cells([(0, 0)])
    doc.set_variable_attr(0, "label", "Skor A")
    doc.set_variable_attr(0, "name", "skor")
    assert doc.ds.names() == ["skor", "b"]
    for _ in range(4):
        doc.undo_stack.undo()
    assert doc.ds.names() == ["a", "b"]
    assert doc.ds.var("a").label == ""
    assert doc.ds.df["a"].tolist() == [1, 2, 3]
    assert doc.ds.df["b"].tolist() == ["x", "y", "z"]


def test_compute_recode_sort_undo(doc):
    doc.mutate(lambda ds: compute(ds, "c", "a * 2"), "Compute Variable")
    doc.mutate(lambda ds: recode(ds, ["a"], [Rule(("value", 1.0), ("value", 9.0))]), "Recode")
    doc.mutate(lambda ds: ds.sort_cases(["a"], [False]), "Sort Cases")
    assert doc.undo_stack.undoText() == "Sort Cases"
    doc.undo_stack.undo()
    doc.undo_stack.undo()
    assert doc.ds.df["a"].tolist() == [1, 2, 3]
    doc.undo_stack.undo()
    assert not doc.ds.has_name("c")
    doc.undo_stack.redo()
    assert doc.ds.df["c"].tolist() == [2, 4, 6]


def test_failed_change_is_rolled_back(doc):
    def broken(ds):
        ds.add_variable("tmp")
        ds.df.iat[0, 0] = 99.0
        raise ValueError("boom")

    with pytest.raises(ValueError):
        doc.mutate(broken)
    assert doc.ds.names() == ["a", "b"]
    assert doc.ds.get_value(0, 0) == 1
    assert doc.undo_stack.count() == 0


def test_modified_follows_clean_state(doc):
    states = []
    doc.modified_changed.connect(states.append)
    doc.set_value(0, 0, "5")
    assert doc.ds.modified
    doc.mark_saved()
    assert not doc.ds.modified
    doc.undo_stack.undo()
    assert doc.ds.modified
    doc.undo_stack.redo()
    assert not doc.ds.modified


def test_new_dataset_clears_history(doc):
    doc.set_value(0, 0, "5")
    doc.set_dataset(Dataset(pd.DataFrame({"z": np.arange(3.0)})))
    assert doc.undo_stack.count() == 0
