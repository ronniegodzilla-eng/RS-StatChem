"""WP6: Chart Builder gallery charts (render every type, pyramid counts, dialog)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.graph import spss_charts as sc


@pytest.fixture()
def ds():
    rng = np.random.default_rng(9)
    n = 90
    d = Dataset(pd.DataFrame({
        "sex": rng.integers(1, 3, n).astype(float), "grp": rng.integers(1, 4, n).astype(float),
        "age": rng.integers(18, 70, n).astype(float), "x": rng.normal(size=n),
        "y": rng.normal(size=n), "z": rng.normal(size=n), "t": np.arange(n, dtype=float)}))
    for name in ("sex", "grp"):
        d.var(name).measure = "nominal"
    d.var("sex").value_labels = {1.0: "Male", 2.0: "Female"}
    d.add_variable("lo", values=(d.df["x"] - 2).to_numpy())
    d.add_variable("hi2", values=(d.df["x"] + 2).to_numpy())
    return d


ROLES = {
    "splom": {"vars": ["x", "y", "z"], "group": "sex"},
    "panel_scatter": {"x": "x", "y": "y", "panel": "grp", "group": "sex"},
    "dot": {"y": "x"}, "dropline": {"x": "grp", "y": ["x", "y"]},
    "summary_point": {"x": "grp", "y": "y", "group": "sex"},
    "violin": {"y": "y", "x": "grp"}, "box1d": {"vars": ["x", "y"]},
    "stacked_hist": {"y": "x", "group": "sex"}, "freq_polygon": {"y": "x", "group": "sex"},
    "pyramid": {"y": "age", "split": "sex"}, "panel_hist": {"y": "x", "panel": "grp"},
    "highlow": {"x": "t", "high": "hi2", "low": "lo", "close": "x"},
    "rangebar": {"x": "grp", "high": "hi2", "low": "lo"},
    "difference": {"x": "t", "y1": "x", "y2": "y"},
    "dualaxis": {"x": "grp", "y1": "x", "y2": "y"},
    "regplots": {"y": "y", "vars": ["x", "z"]},
}


@pytest.mark.parametrize("key", list(sc.CHARTS))
def test_every_chart_renders(ds, key):
    section = sc.chart_section(ds, key, ROLES[key])
    png = section.items[-1].png
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and len(png) > 2000
    assert "GGRAPH" in section.items[0].text


def test_pyramid_counts_match_crosstab(ds):
    ds.var("age").measure = "ordinal"
    rows, groups, counts = sc.pyramid_counts(ds, "grp", "sex")
    want = pd.crosstab(ds.df.grp, ds.df.sex)
    assert groups == ["Male", "Female"] and counts.tolist() == want.values.tolist()
    ds.var("age").measure = "scale"
    rows, _, counts = sc.pyramid_counts(ds, "age", "sex", bins=5)
    assert len(rows) == 5 and counts.sum() == ds.n_cases


def test_missing_role_is_reported(ds):
    with pytest.raises(sc.ChartError):
        sc.render(ds, "pyramid", {"y": "age"})
    with pytest.raises(sc.ChartError):
        sc.render(ds, "pyramid", {"y": "age", "split": "grp"})  # three groups


def test_chart_builder_dialog(qapp, ds):
    from rsstatchem.ui.graph.chart_builder import ChartBuilderDialog, preset

    dlg = preset("violin", "Violin")(ds)
    assert dlg.chart().key == "violin"
    assert not dlg.ok_button.isEnabled()
    dlg.set_roles({"y": "y", "x": "grp"})
    assert dlg.ok_button.isEnabled() and not dlg.preview.pixmap().isNull()
    section = dlg.compute()
    assert section.title == "GGraph: Violin Plot"
    other = ChartBuilderDialog(ds)
    other.select_chart("splom")
    other.set_roles({"vars": ["x", "y"]})
    assert other.roles()["vars"] == ["x", "y"]
