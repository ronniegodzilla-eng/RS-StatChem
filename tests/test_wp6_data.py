"""WP6: Stata/SAS import-export, Data/Transform/Utilities helpers, Reports, Data Editor menus."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyreadstat
import pytest

from rsstatchem.core import data_props as dp
from rsstatchem.core.dataset import DATE, Dataset
from rsstatchem.core.stats import reports as rp
from rsstatchem.io import files


@pytest.fixture()
def ds():
    df = pd.DataFrame({"id": [1.0, 2.0, 3.0, 4.0, 5.0], "jk": [1.0, 2.0, 2.0, 1.0, np.nan],
                       "nilai": [70.5, 80.25, 65.0, 90.0, 77.0],
                       "kota": ["Medan", "Padang", "Medan", "Bandung", "Padang"],
                       "tgl": ["17/08/2001", "01/02/2002", "15/03/2003", "31/12/2004", "x"]})
    d = Dataset(df)
    d.var("jk").label = "Jenis kelamin"
    d.var("jk").value_labels = {1.0: "Laki-laki", 2.0: "Perempuan"}
    d.var("jk").measure = "nominal"
    d.var("nilai").label = "Nilai ujian"
    d.var("nilai").decimals = 2
    return d


# ------------------------------------------------------------------ Stata / SAS
def test_stata_round_trip(tmp_path, ds):
    path = tmp_path / "data.dta"
    files.write_stata(ds, path)
    df, meta = pyreadstat.read_dta(str(path))
    assert list(df.columns) == ["id", "jk", "nilai", "kota", "tgl"]
    assert meta.column_names_to_labels["nilai"] == "Nilai ujian"
    back = files.read_stata(path)
    assert back.names() == ds.names()
    np.testing.assert_allclose(back.df["nilai"], ds.df["nilai"])
    assert back.var("jk").label == "Jenis kelamin"
    assert back.var("jk").value_labels == {1.0: "Laki-laki", 2.0: "Perempuan"}
    assert list(back.df["kota"]) == list(ds.df["kota"])
    assert back.var("nilai").decimals == 2 and back.var("id").decimals == 0
    assert files.kind_of(path) == "stata"


def test_sas_transport_round_trip(tmp_path, ds):
    path = tmp_path / "data.xpt"
    files.write_xport(ds, path)
    back = files.read_sas(path)
    assert [n.lower() for n in back.names()] == ds.names()
    np.testing.assert_allclose(back.df[back.names()[2]], ds.df["nilai"])
    assert back.variables[2].label == "Nilai ujian"
    assert files.kind_of(path) == "sas"


def test_sas7bdat_and_portable_import(tmp_path, ds):
    sav = tmp_path / "d.sav"
    files.write_sav(ds, sav)
    por = tmp_path / "d.por"
    df, meta = pyreadstat.read_sav(str(sav))
    pyreadstat.write_por(df, str(por), column_labels=meta.column_labels)
    back = files.read_por(por)
    np.testing.assert_allclose(back.df[back.names()[2]], ds.df["nilai"])
    assert files.kind_of(por) == "por"
    # no sas7bdat writer exists; reading goes through the same metadata path
    assert files.kind_of(tmp_path / "x.sas7bdat") == "sas"


def test_export_renames_invalid_names(tmp_path):
    d = Dataset(pd.DataFrame({"a.b": [1.0], "q.r.s": [2.0], "a_b": [3.0]}))
    rename = files.write_stata(d, tmp_path / "n.dta")
    assert rename == {"a.b": "a_b", "q.r.s": "q_r_s", "a_b": "a_b_1"}
    assert files.read_stata(tmp_path / "n.dta").names() == ["a_b", "q_r_s", "a_b_1"]


# ------------------------------------------------------------------ find / replace
def test_find_and_replace(ds):
    col = ds.index_of("kota")
    assert dp.find_cells(ds, "medan", [col]) == [(0, col), (2, col)]
    assert dp.find_cells(ds, "medan", [col], match_case=True) == []
    assert dp.find_cells(ds, "Pad", None, mode="begins") == [(1, col), (4, col)]
    assert dp.find_cells(ds, "Perempuan", [ds.index_of("jk")], use_labels=True) == \
        [(1, 1), (2, 1)]
    assert dp.find_cells(ds, "80.25", None) == [(1, 2)]
    assert dp.next_match([(0, 1), (2, 1)], (0, 1)) == (2, 1)
    assert dp.next_match([(0, 1), (2, 1)], (2, 1)) == (0, 1)  # wraps around
    assert dp.replace_all(ds, "Medan", "Binjai", [col]) == 2
    assert list(ds.df["kota"])[:3] == ["Binjai", "Padang", "Binjai"]
    assert dp.replace_all(ds, "65", "66", [2], mode="entire") == 1
    assert ds.df["nilai"].iloc[2] == 66.0
    assert dp.find_variables(ds, "ujian", "label") == [2]
    assert dp.find_variables(ds, "Laki", "value_labels") == [1]


# ------------------------------------------------------------------ properties / sort
def test_define_properties_scan_and_suggest(ds):
    values = dp.scan_values(ds, "kota")
    assert values == [("Bandung", 1, False), ("Medan", 2, False), ("Padang", 2, False)]
    assert dp.suggest_measure(ds, "kota")[0] == "nominal"
    assert dp.suggest_measure(ds, "nilai")[0] == "scale"
    assert dp.suggest_measure(ds, "jk")[0] == "nominal"
    dp.apply_properties(ds, "kota", label="Kota asal", value_labels={"Medan": "Kota Medan"},
                        missing=["Bandung"], measure="nominal")
    assert ds.var("kota").label == "Kota asal" and ds.var("kota").missing == ["Bandung"]
    assert ds.analysis_series("kota").isna().sum() == 1


def test_copy_properties_and_sort_variables(ds):
    other = Dataset(pd.DataFrame({"nilai": [1.0], "jk": [1.0], "baru": [3.0]}))
    sec = dp.copy_properties(ds, other, [("jk", "jk"), ("nilai", "nilai")],
                             ("label", "value_labels", "measure", "decimals"))
    assert other.var("jk").value_labels == ds.var("jk").value_labels
    assert other.var("nilai").label == "Nilai ujian" and other.var("nilai").decimals == 2
    assert sec.items[1].rows[0][:2] == ["jk", "jk"]
    dp.sort_variables(other, "name")
    assert other.names() == ["baru", "jk", "nilai"]
    dp.sort_variables(ds, "label", ascending=False, save_order_as="Asli")
    assert ds.names()[0] == "nilai" and ds.meta["attributes"]["Asli"][0] == "id"
    assert list(ds.df.columns) == ds.names()


def test_create_dummies(ds):
    dp.create_dummies(ds, ["kota"], omit="first")
    assert ds.has_name("kota_1") and ds.has_name("kota_2") and not ds.has_name("kota_3")
    assert list(ds.df["kota_1"]) == [1.0, 0.0, 1.0, 0.0, 0.0]   # Medan (Bandung omitted)
    assert ds.var("kota_1").label == "kota=Medan"
    dp.create_dummies(ds, ["jk"], roots=["d_jk"])
    assert ds.df["d_jk_1"].isna().iloc[4] and ds.var("d_jk_2").label == \
        "Jenis kelamin=Perempuan"


def test_date_time_wizard(ds):
    dp.wizard_parse(ds, "tgl", "tanggal", "%d/%m/%Y")
    assert ds.var("tanggal").type == DATE
    assert str(ds.df["tanggal"].iloc[0].date()) == "2001-08-17"
    assert ds.df["tanggal"].isna().iloc[4]
    dp.wizard_extract(ds, "tanggal", "bulan", "month")
    assert list(ds.df["bulan"][:4]) == [8.0, 2.0, 3.0, 12.0]
    dp.wizard_add(ds, "tanggal", "jatuh_tempo", 1, "months")
    assert str(ds.df["jatuh_tempo"].iloc[3].date()) == "2005-01-31"
    dp.wizard_difference(ds, "jatuh_tempo", "tanggal", "selisih", "days")
    assert ds.df["selisih"].iloc[0] == 31
    ds.add_variable("th", values=[2020.0, 2021, 2022, 2023, 2024])
    ds.add_variable("bl", values=[1.0, 2, 3, 4, 5])
    dp.wizard_from_parts(ds, "awal", "th", "bl")
    assert str(ds.df["awal"].iloc[1].date()) == "2021-02-01"
    with pytest.raises(dp.PropsError):
        dp.wizard_extract(ds, "tanggal", "bulan", "month")  # name taken


def test_compare_datasets_and_split(tmp_path, ds):
    other = Dataset(ds.df.copy(), [__import__("copy").deepcopy(v) for v in ds.variables])
    other.df.loc[2, "nilai"] = 99.0
    sec = dp.compare_datasets(ds, other, ["id"], flag="beda")
    assert sec.items[1].rows[1][1] == "1"
    assert list(ds.df["beda"]) == [0.0, 0.0, 1.0, 0.0, 0.0]
    paths, section = dp.split_into_files(ds, ["kota"], tmp_path / "split")
    assert len(paths) == 3 and all(p.exists() for p in paths)
    sizes = sorted(files.load_project(p).n_cases for p in paths)
    assert sizes == [1, 2, 2]


def test_variable_sets_and_comments(ds):
    dp.set_variable_sets(ds, {"Identitas": ["id", "jk"], "Nilai": ["nilai"]})
    assert dp.visible_variables(ds) is None
    dp.use_variable_sets(ds, ["Nilai"])
    assert dp.visible_variables(ds) == ["nilai"]
    dp.use_variable_sets(ds, None)
    assert dp.visible_variables(ds) is None
    dp.set_comments(ds, "Data penelitian 2024")
    assert dp.comments(ds) == "Data penelitian 2024"


# ------------------------------------------------------------------ reports
def test_reports(ds):
    cb = rp.codebook(ds, ["jk", "nilai"])
    jk = cb.items[1]
    assert any(r[1] == "1" and r[2] == "Laki-laki" and r[3].startswith("2") for r in jk.rows)
    nilai = cb.items[2]
    mean_row = next(r for r in nilai.rows if r[1] == "Mean")
    assert float(mean_row[3]) == pytest.approx(ds.df["nilai"].mean(), abs=1e-4)
    ol = rp.olap_cubes(ds, ["nilai"], ["jk"])
    rows = ol.items[-1].rows
    assert rows[0][0] == "Laki-laki" and float(rows[0][1]) == pytest.approx(160.5)
    assert rows[-1][0] == "Total" and rows[-1][2] == "4"
    assert rows[0][5] == "52.5%"  # % of total sum: 160.5 / 305.75 (cases valid on jk)
    cs = rp.case_summaries(ds, ["nilai", "kota"], ["jk"], limit=10)
    listing = cs.items[-1].rows
    assert listing[0][:3] == ["Laki-laki", "1", "70.50"]
    assert any(r[-3] == "Total N" or "Total N" in r for r in listing)
    rr = rp.report_rows(ds, ["nilai"], ["jk"], ("mean", "n"))
    assert rr.items[-1].rows[-2][0] == "Grand Total"
    rc = rp.report_columns(ds, [("nilai", "sum"), ("nilai", "mean")], ["jk"])
    assert rc.items[-1].columns[1] == ("Nilai ujian", "Sum")
    fi = rp.file_info(ds)
    assert fi.items[1].rows[1][:3] == ["jk", "2", "Jenis kelamin"]
    assert fi.items[2].rows[0] == ["jk", "1", "Laki-laki"]


# ------------------------------------------------------------------ menus / editor
def _labels(entries):
    return [e if isinstance(e, str) or e is None else e[0] for e in entries]


def _sub(entries, title):
    return next(e[1] for e in entries if isinstance(e, tuple) and isinstance(e[1], list)
                and e[0] == title)


def test_menu_order_follows_spss():
    from rsstatchem.ui.data_editor import menu_tree

    tree = menu_tree()
    top = [t for t in _labels(tree["Analyze"]) if t]
    assert top.index("Reports") < top.index("Descriptive Statistics") < top.index("Tables") \
        < top.index("Compare Means") < top.index("General Linear Model") \
        < top.index("Generalized Linear Models") < top.index("Mixed Models") \
        < top.index("Correlate") < top.index("Regression") < top.index("Loglinear") \
        < top.index("Classify") < top.index("Dimension Reduction") < top.index("Scale") \
        < top.index("Nonparametric Tests") < top.index("Forecasting") \
        < top.index("Survival") < top.index("Multiple Response")
    nonpar = [t for t in _labels(_sub(tree["Analyze"], "Nonparametric Tests")) if t]
    assert nonpar == ["One Sample...", "Independent Samples...", "Related Samples...",
                      "Legacy Dialogs"]
    desc = _labels(_sub(tree["Analyze"], "Descriptive Statistics"))
    assert desc[:4] == ["FREQ", "DESC", "Explore...", "Crosstabs..."]
    flat = []

    def walk(entries):
        for e in entries:
            if isinstance(e, str):
                flat.append(e)
            elif isinstance(e, tuple) and isinstance(e[1], list):
                walk(e[1])
    walk(tree["Analyze"])
    for old in ("Distances...", "Nonlinear...", "Ratio..."):
        assert old not in flat  # placeholders replaced by real dialogs
    data = [t for t in _labels(tree["Data"]) if t]
    assert data.index("Define Variable Properties...") < data.index("Sort Cases...") \
        < data.index("Sort Variables...") < data.index("Split File...") \
        < data.index("Weight Cases...")
    transform = [t for t in _labels(tree["Transform"]) if t]
    assert transform[0] == "Compute Variable..." and transform[-1] == "Run Pending Transforms"
    assert transform.index("Create Dummy Variables...") < transform.index("Visual Binning...")
    assert [t for t in _labels(tree["Utilities"]) if t][0] == "Variables..."
    reports = _labels(_sub(tree["Analyze"], "Reports"))
    assert reports == ["Codebook...", "OLAP Cubes...", "Case Summaries...",
                       "Report Summaries in Rows...", "Report Summaries in Columns..."]


def test_order_entries_puts_unknown_last():
    from rsstatchem.ui.menus import order_entries

    entries = [("B...", 1), None, ("New Thing...", 2), ("A...", 3)]
    assert order_entries(entries, ["A...", None, "B..."]) == \
        [("A...", 3), None, ("B...", 1), None, ("New Thing...", 2)]


def test_editor_find_goto_and_variable_sets(qapp, ds, tmp_path, monkeypatch):
    from PySide6.QtGui import QGuiApplication

    monkeypatch.setenv("RSSTATCHEM_HOME", str(tmp_path))
    from rsstatchem.app import AppContext

    ctx = AppContext(qapp)
    ed = ctx.new_editor(ds)
    titles = [a.text().replace("&", "") for a in ed.menuBar().actions()]
    spss = ["File", "Edit", "View", "Data", "Transform", "Analyze", "Graphs", "Utilities",
            "Window", "Help"]
    assert [t for t in titles if t in spss] == spss
    dlg = ed.find_dialog()
    dlg.find_edit.setText("Padang")
    dlg.all_columns.setChecked(True)
    assert dlg.find_next()
    assert (ed.data_view.currentIndex().row(), ed.data_view.currentIndex().column()) == (1, 3)
    dlg.find_next()
    assert ed.data_view.currentIndex().row() == 4
    dlg.do_replace.setChecked(True)
    dlg.replace_edit.setText("Bukittinggi")
    dlg.replace_everything()
    assert list(ed.ds.df["kota"]).count("Bukittinggi") == 2
    ed.doc.undo_stack.undo()
    assert list(ed.ds.df["kota"]).count("Padang") == 2
    ed.go_to_variable_index(2)
    assert ed.data_view.currentIndex().column() == 2
    ed.data_view.selectAll()
    ed.copy_with_header("label")
    text = QGuiApplication.clipboard().text().splitlines()
    assert text[0].split("\t")[:3] == ["id", "Jenis kelamin", "Nilai ujian"]
    dp.set_variable_sets(ed.ds, {"Nilai": ["nilai"]})
    dp.use_variable_sets(ed.ds, ["Nilai"])
    ed.apply_variable_sets()
    assert ed.data_view.isColumnHidden(0) and not ed.data_view.isColumnHidden(2)
    dp.use_variable_sets(ed.ds, None)
    ed.apply_variable_sets()
    assert not ed.data_view.isColumnHidden(0)
