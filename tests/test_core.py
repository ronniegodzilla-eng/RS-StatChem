import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from rsstatchem.core.dataset import NUMERIC, STRING, Dataset
from rsstatchem.core.report import document_html, export_docx
from rsstatchem.core.output import OutputDocument
from rsstatchem.core.stats import descriptive as d
from rsstatchem.io import files


@pytest.fixture
def ds():
    df = pd.DataFrame({
        "Nilai Ujian": [70, 85, 90, 65, 85, 99, 9],
        "gender": [1, 2, 2, 1, 1, 2, 1],
        "kota": ["Padang", "Medan", "", "Padang", "Bandung", "Medan", "Padang"],
    })
    data = Dataset(df)
    g = data.var("gender")
    g.value_labels = {1.0: "Laki-laki", 2.0: "Perempuan"}
    g.measure = "nominal"
    data.var("Nilai_Ujian").missing = [9.0]
    return data


def test_import_sanitizes_names_and_keeps_label(ds):
    assert ds.names() == ["Nilai_Ujian", "gender", "kota"]
    assert ds.var("Nilai_Ujian").label == "Nilai Ujian"
    assert ds.var("kota").type == STRING
    assert ds.var("gender").decimals == 0


def test_edit_grows_dataset_and_parses_values(ds):
    ds.set_value(9, 4, "3,5")
    assert ds.n_cases == 10 and ds.n_vars == 5
    assert [v.name for v in ds.variables[3:]] == ["VAR00001", "VAR00002"]
    assert ds.get_value(9, 4) == 3.5
    ds.set_value(0, 1, "perempuan")  # value label typed in
    assert ds.get_value(0, 1) == 2.0
    with pytest.raises(ValueError):
        ds.set_value(0, 0, "abc")


def test_change_type_roundtrip(ds):
    ds.change_type(1, STRING)
    assert ds.get_value(0, 1) == "1"
    assert "1" in ds.var("gender").value_labels
    ds.change_type(1, NUMERIC)
    assert ds.get_value(1, 1) == 2.0
    assert ds.var("gender").value_labels[2.0] == "Perempuan"


def test_user_missing_excluded(ds):
    s = ds.analysis_series("Nilai_Ujian")
    assert s.isna().sum() == 1


def test_summary_matches_scipy():
    x = pd.Series([2.0, 4, 4, 5, 7, 9, 10, 12, 15])
    r = d.summary(x)
    assert math.isclose(r["skew"], sps.skew(x, bias=False))
    assert math.isclose(r["kurt"], sps.kurtosis(x, bias=False))
    assert math.isclose(r["std"], np.std(x, ddof=1))
    # SPSS HAVERAGE percentiles for n=9: 25th at position 2.5
    assert math.isclose(r["p25"], 4.0)
    assert math.isclose(r["p75"], 11.0)
    assert math.isclose(r["se_skew"], 0.717, abs_tol=5e-4)
    assert math.isclose(r["se_kurt"], 1.400, abs_tol=5e-4)


def test_descriptives_and_frequencies_tables(ds):
    sec = d.descriptives(ds, ["Nilai_Ujian", "gender", "kota"], stats=("mean", "std", "skew"))
    table = sec.items[-1]
    assert table.rows[0][0] == "Nilai Ujian"
    assert table.rows[0][1] == "6"  # user-missing 9 excluded
    assert table.rows[-1][0] == "Valid N (listwise)"

    sec = d.frequencies(ds, ["gender", "kota", "Nilai_Ujian"], stats=("mean", "mode"),
                        chart="bar")
    freq_gender = sec.items[2]
    assert freq_gender.rows[0][1] == "Laki-laki"
    assert freq_gender.rows[0][2] == "4"
    freq_kota = sec.items[3]
    assert any(r[1] == "System" and r[2] == "1" for r in freq_kota.rows)
    freq_score = sec.items[4]
    assert any(r[1] == "9" and r[0] == "Missing" for r in freq_score.rows)
    assert any(i.kind == "image" for i in sec.items)


def test_standardize(ds):
    names = d.standardize(ds, ["Nilai_Ujian"])
    z = ds.df[names[0]].dropna()
    assert math.isclose(z.mean(), 0, abs_tol=1e-12)
    assert math.isclose(z.std(ddof=1), 1)


def test_project_roundtrip(tmp_path, ds):
    ds.set_value(0, 2, "Jakarta, Selatan")
    path = tmp_path / "data.rsd"
    files.save_project(ds, path)
    back = files.load_project(path)
    assert back.names() == ds.names()
    assert back.var("gender").value_labels == {1.0: "Laki-laki", 2.0: "Perempuan"}
    assert back.var("Nilai_Ujian").missing == [9.0]
    assert back.get_value(0, 2) == "Jakarta, Selatan"
    pd.testing.assert_frame_equal(back.df, ds.df)


def test_sav_roundtrip(tmp_path, ds):
    path = tmp_path / "data.sav"
    files.write_sav(ds, path)
    back = files.read_sav(path)
    assert back.names() == ds.names()
    assert back.var("gender").value_labels[1.0] == "Laki-laki"
    assert back.var("gender").measure == "nominal"
    assert back.var("Nilai_Ujian").missing == [9.0]
    assert back.var("Nilai_Ujian").label == "Nilai Ujian"
    assert back.get_value(5, 0) == 99.0


def test_csv_sniff_semicolon_comma(tmp_path):
    path = tmp_path / "id.csv"
    path.write_text("nama;tinggi;berat\nAni;160,5;50,2\nBudi;170,1;65,0\n", encoding="utf-8")
    enc = files.detect_encoding(path)
    delim, dec = files.sniff_text(path, enc)
    assert (delim, dec) == (";", ",")
    df = files.read_text(path, delim, dec, encoding=enc)
    assert df["tinggi"].tolist() == [160.5, 170.1]


def test_excel_roundtrip(tmp_path, ds):
    path = tmp_path / "data.xlsx"
    files.write_excel(ds, path)
    assert files.excel_sheets(path) == ["Data", "Variables"]
    df = files.read_excel(path, "Data")
    assert list(df.columns) == ds.names()


def test_report_exports(tmp_path, ds):
    doc = OutputDocument()
    doc.add(d.descriptives(ds, ["Nilai_Ujian"], stats=("mean", "std", "kurt")))
    doc.add(d.frequencies(ds, ["gender"], chart="pie"))
    html = document_html(doc)
    assert "Descriptive Statistics" in html and "data:image/png;base64" in html
    export_docx(doc, str(tmp_path / "out.docx"))
    assert (tmp_path / "out.docx").stat().st_size > 5000
