import math

import numpy as np
import pandas as pd
import pytest

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.expression import ExpressionError, check, evaluate
from rsstatchem.core.transform import Rule, compute, recode


@pytest.fixture
def ds():
    df = pd.DataFrame({
        "X1": [1, 2, 3, np.nan, 5],
        "X2": [2, 2, 9, 4, 5],
        "X3": [3, np.nan, 3, 4, 5],
        "nilai.akhir": [55, 70, 85, 40, 99],
        "nama": ["ani", "Budi", "", "dewi", "eka"],
        "kelas": ["A", "B", "A", "C", "B"],
    })
    data = Dataset(df)
    data.var("X2").missing = [9.0]
    return data


def vals(series):
    return [None if (isinstance(v, float) and math.isnan(v)) else v for v in series]


def test_arithmetic_and_missing_propagation(ds):
    assert vals(evaluate(ds, "X1 + X2 * 2")) == [5, 6, None, None, 15]  # 9 is user-missing
    assert vals(evaluate(ds, "(X1 + X3) / 2")) == [2, None, 3, None, 5]
    assert vals(evaluate(ds, "X1 / 0")) == [None] * 5
    assert vals(evaluate(ds, "-X1 ** 2"))[0] == -1


def test_statistical_functions_and_to(ds):
    assert vals(evaluate(ds, "SUM(X1 TO X3)")) == [6, 4, 6, 8, 15]
    assert vals(evaluate(ds, "MEAN(X1 TO X3)")) == [2, 2, 3, 4, 5]
    assert vals(evaluate(ds, "MEAN.3(X1, X2, X3)")) == [2, None, None, None, 5]
    assert vals(evaluate(ds, "NVALID(X1 TO X3)")) == [3, 2, 2, 2, 3]
    assert vals(evaluate(ds, "NMISS(X1 TO X3)")) == [0, 1, 1, 1, 0]
    assert vals(evaluate(ds, "VALUE(X2)"))[2] == 9
    assert vals(evaluate(ds, "MISSING(X2)")) == [0, 0, 1, 0, 0]
    assert vals(evaluate(ds, "SYSMIS(X2)")) == [0, 0, 0, 0, 0]
    assert round(evaluate(ds, "SD(X1, X2, X3)")[0], 6) == 1.0


def test_logic_three_valued(ds):
    assert vals(evaluate(ds, "X1 >= 3")) == [0, 0, 1, None, 1]
    assert vals(evaluate(ds, "X1 > 10 AND X3 > 0")) == [0, 0, 0, None, 0]  # missing AND true
    assert vals(evaluate(ds, "X1 > 10 AND X3 > 99")) == [0, 0, 0, 0, 0]  # missing AND false
    assert vals(evaluate(ds, "X1 > 1 & X3 > 0")) == [0, None, 1, None, 1]
    assert vals(evaluate(ds, "X1 = 1 OR X1 = 5")) == [1, 0, 0, None, 1]
    assert vals(evaluate(ds, "NOT (X1 < 3)")) == [0, 0, 1, None, 1]
    assert vals(evaluate(ds, "X1 ~= 2")) == [1, 0, 1, None, 1]
    assert vals(evaluate(ds, "RANGE(nilai.akhir, 60, 90)")) == [0, 1, 1, 0, 0]
    assert vals(evaluate(ds, "ANY(kelas, 'A', 'C')")) == [1, 0, 1, 1, 0]


def test_strings(ds):
    assert vals(evaluate(ds, "CONCAT(UPCASE(nama), '-', kelas)"))[:2] == ["ANI-A", "BUDI-B"]
    assert vals(evaluate(ds, "LENGTH(nama)")) == [3, 4, 0, 4, 3]
    assert vals(evaluate(ds, "NUMBER('3,5') + 1"))[0] == 4.5
    assert vals(evaluate(ds, "STRING(X1, 1)"))[0] == "1.0"
    assert vals(evaluate(ds, "kelas = 'A'")) == [1, 0, 1, 0, 0]


def test_rounding_and_math(ds):
    assert vals(evaluate(ds, "RND(2.5)"))[0] == 3
    assert vals(evaluate(ds, "RND(-2.5)"))[0] == -3
    assert vals(evaluate(ds, "RND(nilai.akhir, 10)")) == [60, 70, 90, 40, 100]
    assert vals(evaluate(ds, "MOD(-7, 3)"))[0] == -1
    assert vals(evaluate(ds, "LN(0)"))[0] is None
    assert vals(evaluate(ds, "SQRT(-1)"))[0] is None
    assert math.isclose(evaluate(ds, "CDF.NORMAL(1.96, 0, 1)")[0], 0.975, abs_tol=1e-3)


@pytest.mark.parametrize("bad, message", [
    ("X1 +", "incomplete"), ("FOO(X1)", "Unknown function"), ("X9 + 1", "Unknown variable"),
    ("nama + 1", "string"), ("kelas = 1", "Type mismatch"), ("__import__('os')", "Unknown"),
    ("X1.real", "Unknown variable"), ("X1 # 2", "Unexpected character"),
])
def test_errors_are_reported_not_executed(ds, bad, message):
    err = check(ds, bad)
    assert err is not None and message.lower() in err.lower(), err


def test_compute_new_existing_and_if(ds):
    sec = compute(ds, "total", "SUM(X1 TO X3)", label="Total skor")
    assert ds.var("total").label == "Total skor"
    assert vals(ds.df["total"]) == [6, 4, 6, 8, 15]
    assert "COMPUTE total=SUM(X1 TO X3)." in sec.items[0].text

    compute(ds, "total", "0", condition="kelas = 'A'")
    assert vals(ds.df["total"]) == [0, 4, 0, 8, 15]

    compute(ds, "lulus", "nilai.akhir >= 60")
    assert vals(ds.df["lulus"]) == [0, 1, 1, 0, 1]

    compute(ds, "inisial", "UPCASE(nama)")
    assert ds.var("inisial").type == "string"
    with pytest.raises(ExpressionError):
        compute(ds, "inisial", "X1 + 1")
    with pytest.raises(ExpressionError):
        compute(ds, "1bad", "X1")


def test_recode_into_different(ds):
    rules = [Rule(("lowest", 59.999), ("value", 1.0)), Rule(("range", 60.0, 79.999), ("value", 2.0)),
             Rule(("highest", 80.0), ("value", 3.0))]
    sec = recode(ds, ["nilai.akhir"], rules, targets=["kategori"], labels=["Kategori nilai"])
    assert vals(ds.df["kategori"]) == [1, 2, 3, 1, 3]
    assert ds.var("kategori").decimals == 0
    assert "INTO kategori" in sec.items[0].text

    recode(ds, ["X1"], [Rule(("value", 1.0), ("value", 10.0)), Rule(("sysmis",), ("value", 0.0))],
           targets=["X1r"])
    assert vals(ds.df["X1r"]) == [10, None, None, 0, None]  # unmatched -> sysmis

    recode(ds, ["X1"], [Rule(("value", 1.0), ("value", 10.0)), Rule(("else",), ("copy",))],
           targets=["X1c"])
    assert vals(ds.df["X1c"]) == [10, 2, 3, None, 5]

    recode(ds, ["kelas"], [Rule(("value", "A"), ("value", 1.0)), Rule(("value", "B"), ("value", 2.0)),
                           Rule(("else",), ("value", 9.0))], targets=["kelas_n"])
    assert vals(ds.df["kelas_n"]) == [1, 2, 1, 9, 2]


def test_recode_into_same_reverse_items_and_missing(ds):
    reverse = [Rule(("value", float(v)), ("value", float(6 - v))) for v in range(1, 6)]
    recode(ds, ["X1", "X3"], reverse)
    assert vals(ds.df["X1"]) == [5, 4, 3, None, 1]
    assert vals(ds.df["X3"]) == [3, None, 3, 2, 1]
    recode(ds, ["X2"], [Rule(("missing",), ("sysmis",))])
    assert vals(ds.df["X2"]) == [2, 2, None, 4, 5]
    with pytest.raises(ExpressionError):
        recode(ds, ["X1"], [Rule(("else",), ("copy",))])


def test_recode_with_condition_and_convert(ds):
    ds.add_variable("teks", "string", values=["1", "2", "x", "4", "5"])
    recode(ds, ["teks"], [Rule(("value", "x"), ("sysmis",))], targets=["angka"], convert=True)
    assert vals(ds.df["angka"]) == [1, 2, None, 4, 5]
    recode(ds, ["nilai.akhir"], [Rule(("else",), ("value", 0.0))], condition="kelas = 'B'")
    assert vals(ds.df["nilai.akhir"]) == [55, 0, 85, 40, 0]
