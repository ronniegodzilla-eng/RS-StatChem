import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("RSSTATCHEM_NO_SPLASH", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from rsstatchem.core import selection  # noqa: E402
from rsstatchem.core.dataset import Dataset  # noqa: E402
from rsstatchem.core.stats import descriptive, regression  # noqa: E402


@pytest.fixture
def ds():
    df = pd.DataFrame({"g": [1, 1, 1, 2, 2, 2, 2, 1], "x": [1, 2, 3, 4, 5, 6, 7, 8.0],
                       "w": [1, 2, 1, 1, 0, 3, 1, 1.0]})
    data = Dataset(df)
    data.var("g").value_labels = {1.0: "A", 2.0: "B"}
    return data


def test_filter_if_creates_filter_variable(ds):
    sec, new = selection.select_cases(ds, "if", expression="x > 4")
    assert new is None and ds.has_name("filter_$")
    assert ds.df["filter_$"].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    views = selection.analysis_views(ds)
    assert len(views) == 1 and views[0].ds.n_cases == 4
    assert list(views[0].rows) == [4, 5, 6, 7]
    assert "FILTER BY filter_$" in sec.items[0].text
    selection.select_cases(ds, "all")
    assert selection.analysis_views(ds)[0].ds.n_cases == 8


def test_sample_range_variable_copy_delete(ds):
    selection.select_cases(ds, "sample", n_exact=3, from_first=8, seed=1)
    assert int(ds.df["filter_$"].sum()) == 3
    selection.select_cases(ds, "range", first=2, last=4)
    assert ds.df["filter_$"].tolist() == [0, 1, 1, 1, 0, 0, 0, 0]
    selection.select_cases(ds, "variable", variable="w")
    assert selection.analysis_views(ds)[0].ds.n_cases == 7  # w == 0 is not selected
    _, copy = selection.select_cases(ds, "if", expression="g = 2", output="copy")
    assert copy.n_cases == 4 and ds.n_cases == 8
    selection.select_cases(ds, "if", expression="g = 2", output="delete")
    assert ds.n_cases == 4 and "filter" not in ds.meta


def test_split_views_and_combine(ds):
    selection.split_file(ds, ["g"], "separate")
    assert ds.df["g"].tolist() == [1, 1, 1, 1, 2, 2, 2, 2]  # sorted by the split variable
    views = selection.analysis_views(ds)
    assert [v.label for v in views] == ["g = A", "g = B"]
    sec = selection.combine([(v.label, descriptive.descriptives(v.ds, ["x"])) for v in views],
                            ds)
    tables = [i for i in sec.items if i.kind == "table"]
    assert [t.title for t in tables] == ["Descriptive Statistics — g = A",
                                         "Descriptive Statistics — g = B"]
    assert sum(1 for i in sec.items if getattr(i, "style", "") == "log") == 1
    selection.split_file(ds, [], "off")
    assert len(selection.analysis_views(ds)) == 1


def test_weight_replicates_cases(ds):
    selection.weight_cases(ds, "w")
    view = selection.analysis_views(ds)[0]
    assert view.ds.n_cases == 10 and view.rows is None
    r = descriptive.summary(view.ds.df["x"])
    expected = np.repeat(ds.df["x"].to_numpy(), ds.df["w"].astype(int).to_numpy())
    assert r["mean"] == pytest.approx(expected.mean())
    selection.weight_cases(ds, None)
    assert "weight" not in ds.meta


def test_filter_and_split_together(ds):
    selection.select_cases(ds, "if", expression="x >= 3")
    selection.split_file(ds, ["g"])
    sizes = [v.ds.n_cases for v in selection.analysis_views(ds)]
    assert sizes == [2, 4]


def test_editor_runs_analyses_on_views_and_maps_saved_columns(ds, qapp):
    from rsstatchem.app import AppContext

    ctx = AppContext(qapp)
    editor = ctx.new_editor(ds)
    editor.doc.mutate(lambda d: selection.select_cases(d, "if", expression="x > 2"))
    assert "Filter On" in editor.state_label.text()
    assert editor.data_model.headerData(0, 2, 6) is not None  # struck-through font role

    res = editor._compute_on_views(lambda d: regression.linear(
        d, "x", ["w"], save=("resid",), plots=(), assumptions=()))
    full = res.saved["RES"][1]
    assert full.iloc[:2].isna().all() and full.iloc[2:].notna().all()
    sec = editor._compute_on_views(lambda d: descriptive.descriptives(d, ["x"]))
    assert any("Filter: filter_$" in getattr(i, "text", "") for i in sec.items)
    editor.doc.undo_stack.undo()
    assert "filter" not in editor.ds.meta
    editor.doc.set_modified(False)
    editor.close()
