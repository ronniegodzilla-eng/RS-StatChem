"""Data / Transform procedures (core/data_ops.py) against reference implementations."""
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from rsstatchem.core import data_ops as do
from rsstatchem.core.dataset import Dataset


def vals(series):
    return [None if (isinstance(v, float) and math.isnan(v)) else v for v in series]


@pytest.fixture
def ds():
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6, 7, 8],
        "grp": [1, 1, 2, 2, 2, 1, 2, np.nan],
        "x": [10, 20, 20, 30, np.nan, 99, 40, 50],
        "y": [1, 3, 5, 3, 1, 9, 2, 4],
        "s": ["b", "a", "", "c", "a", "b", "B", "a"],
    })
    data = Dataset(df)
    data.var("x").missing = [99.0]
    data.var("s").value_labels = {"a": "Alpha"}
    return data


# ------------------------------------------------------------------ COUNT
def test_count_values_ranges_and_missing(ds):
    do.count_values(ds, "cnt", ["x", "y"], [("value", 20), ("range", 1, 3)])
    # x: 20 counts; y in 1..3 counts; 99 (user-missing) is not in any spec
    assert vals(ds.df["cnt"]) == [1, 2, 1, 1, 1, 0, 1, 0]
    do.count_values(ds, "nmis", ["x", "y"], [("missing",)])
    assert vals(ds.df["nmis"]) == [0, 0, 0, 0, 1, 1, 0, 0]
    do.count_values(ds, "nsys", ["x"], [("sysmis",)])
    assert vals(ds.df["nsys"]) == [0, 0, 0, 0, 1, 0, 0, 0]
    do.count_values(ds, "hi", ["x", "y"], [("highest", 40), ("lowest", 1)])
    assert vals(ds.df["hi"]) == [1, 0, 0, 0, 1, 1, 1, 1]  # 99 >= 40 counts (raw value)


def test_count_values_if_and_strings(ds):
    sec = do.count_values(ds, "c2", ["y"], [("value", 3)], condition="grp = 1")
    assert vals(ds.df["c2"]) == [0, 1, None, None, None, 0, None, None]
    assert "DO IF (grp = 1)" in sec.items[0].text and "COUNT c2=y(3)" in sec.items[0].text
    do.count_values(ds, "ca", ["s"], [("value", "a")])
    assert vals(ds.df["ca"]) == [0, 1, 0, 0, 1, 0, 0, 1]


# ------------------------------------------------------------------ AUTORECODE
def test_automatic_recode_string_order_and_labels(ds):
    do.automatic_recode(ds, ["s"], ["s_rec"])
    # ascending code-point order: '' < 'B' < 'a' < 'b' < 'c'
    assert vals(ds.df["s_rec"]) == [4, 3, 1, 5, 3, 4, 2, 3]
    v = ds.var("s_rec")
    assert v.value_labels == {1.0: "", 2.0: "B", 3.0: "Alpha", 4.0: "b", 5.0: "c"}
    do.automatic_recode(ds, ["s"], ["s_desc"], descending=True, blank_missing=True)
    assert vals(ds.df["s_desc"]) == [2, 3, 5, 1, 3, 2, 4, 3]
    assert ds.var("s_desc").missing == [5.0]


def test_automatic_recode_numeric_user_missing(ds):
    sec = do.automatic_recode(ds, ["x"], ["x_rec"])
    # valid 10 20 30 40 50 -> 1..5, user-missing 99 -> 6 (missing), sysmis stays sysmis
    assert vals(ds.df["x_rec"]) == [1, 2, 2, 3, None, 6, 4, 5]
    assert ds.var("x_rec").missing == [6.0]
    assert ds.var("x_rec").value_labels[6.0] == "99"
    assert "AUTORECODE VARIABLES=x" in sec.items[0].text


def test_automatic_recode_group_scheme():
    d = Dataset(pd.DataFrame({"a": [3, 1, 2], "b": [5, 3, 3]}))
    do.automatic_recode(d, ["a", "b"], ["ra", "rb"], group=True)
    assert vals(d.df["ra"]) == [3, 1, 2]
    assert vals(d.df["rb"]) == [4, 3, 3]


# ------------------------------------------------------------------ VISUAL BINNING
def test_visual_binning_included_excluded():
    d = Dataset(pd.DataFrame({"v": [5, 10, 15, 20, 25, np.nan, 99]}))
    d.var("v").decimals = 0
    d.var("v").missing = [99.0]
    labels = do.bin_labels([10, 20], 0, True)
    assert labels == ["<= 10", "11 - 20", "21+"]
    do.visual_binning(d, "v", "v_bin", [10, 20], labels)
    assert vals(d.df["v_bin"]) == [1, 1, 2, 2, 3, None, 99]
    assert d.var("v_bin").missing == [99.0]
    assert d.var("v_bin").value_labels[2.0] == "11 - 20"
    assert do.bin_labels([10, 20], 0, False) == ["< 10", "10 - 19", "20+"]
    do.visual_binning(d, "v", "v_ex", [10, 20], upper_included=False)
    assert vals(d.df["v_ex"]) == [1, 2, 2, 3, 3, None, 99]
    do.visual_binning(d, "v", "v_rev", [10, 20], reverse=True)
    assert vals(d.df["v_rev"]) == [3, 3, 2, 2, 1, None, 99]
    assert do.bin_counts(d, "v", [10, 20]) == [2, 2, 1]


def test_cutpoint_generators():
    assert do.equal_width_cutpoints(10, 3, 5) == [10, 15, 20]
    x = np.arange(1, 101, dtype=float)
    cuts = do.percentile_cutpoints(x, 3)
    assert cuts == pytest.approx(np.percentile(x, [25, 50, 75], method="weibull"))
    sd = do.sd_cutpoints(x, [1, 2])
    m, s = x.mean(), x.std(ddof=1)
    assert sd == pytest.approx([m - 2 * s, m - s, m, m + s, m + 2 * s])


# ------------------------------------------------------------------ RANK
@pytest.mark.parametrize("ties,method", [("MEAN", "average"), ("LOW", "min"),
                                         ("HIGH", "max"), ("CONDENSE", "dense")])
def test_rank_ties_match_scipy(ties, method):
    x = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5], float)
    assert do.rank_values(x, "RANK", ties) == pytest.approx(stats.rankdata(x, method))
    assert do.rank_values(x, "RANK", ties, ascending=False) == pytest.approx(
        stats.rankdata(-x, method))


def test_rank_derived_functions():
    x = np.array([3, 1, 4, 1, 5, 9, 2, 6], float)
    r, n = stats.rankdata(x), len(x)
    assert do.rank_values(x, "RFRACTION") == pytest.approx(r / n)
    assert do.rank_values(x, "PERCENT") == pytest.approx(100 * r / n)
    assert do.rank_values(x, "N") == pytest.approx([n] * n)
    blom = (r - 3 / 8) / (n + 1 / 4)
    assert do.rank_values(x, "PROPORTION", fraction="BLOM") == pytest.approx(blom)
    assert do.rank_values(x, "NORMAL", fraction="BLOM") == pytest.approx(stats.norm.ppf(blom))
    assert do.rank_values(x, "NORMAL", fraction="TUKEY") == pytest.approx(
        stats.norm.ppf((r - 1 / 3) / (n + 1 / 3)))
    assert do.rank_values(x, "NORMAL", fraction="RANKIT") == pytest.approx(
        stats.norm.ppf((r - 0.5) / n))
    assert do.rank_values(x, "NORMAL", fraction="VW") == pytest.approx(
        stats.norm.ppf(r / (n + 1)))
    # NTILES(4) of 20 untied cases: 5 per group
    y = np.arange(20, dtype=float)
    assert np.bincount(do.rank_values(y, "NTILES", ntiles=4).astype(int))[1:].tolist() == \
        [5, 5, 5, 5]


def test_savage_scores():
    x = np.array([1, 2, 3, 4], float)
    n = 4
    expected = [sum(1 / (n - j + 1) for j in range(1, i + 1)) - 1 for i in range(1, n + 1)]
    assert do.rank_values(x, "SAVAGE") == pytest.approx(expected)
    tied = do.rank_values(np.array([1, 2, 2, 4], float), "SAVAGE")
    assert tied[1] == pytest.approx((expected[1] + expected[2]) / 2) == tied[2]
    assert sum(expected) == pytest.approx(0)  # Savage scores sum to zero


def test_rank_cases_names_by_groups_and_missing(ds):
    sec = do.rank_cases(ds, ["x"], ["RANK", "RFRACTION", "NORMAL"], by=["grp"])
    assert {"Rx", "RFR001", "Nx"} <= set(ds.names())
    # x valid: group1 -> cases 0,1 (10,20); case 5 is user-missing; group2 -> 20,30,40
    assert vals(ds.df["Rx"]) == [1, 2, 1, 2, None, None, 3, None]
    assert ds.var("Rx").label == "Rank of x by grp"
    assert ds.var("Nx").label.endswith("using Blom's Formula by grp")
    table = sec.items[-1]
    assert table.title == "Created Variables" and len(table.rows) == 3


# ------------------------------------------------------------------ RMV
def test_replace_missing_methods():
    s = pd.Series([1.0, np.nan, 3.0, 4.0, np.nan, np.nan, 10.0, np.nan])
    assert vals(do.replace_series(s, "SMEAN")) == [1, 4.5, 3, 4, 4.5, 4.5, 10, 4.5]
    lint = do.replace_series(s, "LINT")
    assert vals(lint) == [1, 2, 3, 4, 6, 8, 10, None]
    med = do.replace_series(s, "MEDIAN", 1)
    assert vals(med)[:2] == [1, 2] and vals(med)[4] == 7 and vals(med)[7] is None
    mean2 = do.replace_series(s, "MEAN", 2)
    assert vals(mean2)[1] is None  # only one valid point above
    assert vals(mean2)[4] is None  # only one valid point below
    assert vals(do.replace_series(s, "MEAN", 1))[4] == 7
    t = np.arange(1, 9, dtype=float)
    ok = s.notna().to_numpy()
    b, a = np.polyfit(t[ok], s.to_numpy()[ok], 1)
    assert do.replace_series(s, "TREND")[7] == pytest.approx(a + b * 8)


def test_replace_missing_dataset(ds):
    sec = do.replace_missing(ds, [("x", "x_1", "SMEAN", None)])
    valid = [10, 20, 20, 30, 40, 50]
    assert ds.df["x_1"][4] == pytest.approx(np.mean(valid))
    assert ds.df["x_1"][5] == pytest.approx(np.mean(valid))  # user-missing replaced too
    assert ds.var("x_1").label == "SMEAN(x)"
    assert sec.items[-1].rows[0][1] == "2"


# ------------------------------------------------------------------ SHIFT
def test_shift_values(ds):
    do.shift_values(ds, [("y", "y_lag", True, 1), ("s", "s_lead", False, 2)])
    assert vals(ds.df["y_lag"]) == [None, 1, 3, 5, 3, 1, 9, 2]
    assert vals(ds.df["s_lead"])[:2] == ["", "c"] and vals(ds.df["s_lead"])[-1] is None
    assert ds.var("s_lead").value_labels == {"a": "Alpha"}


# ------------------------------------------------------------------ DUPLICATES
def test_identify_duplicates():
    d = Dataset(pd.DataFrame({"k": [2, 1, 2, 3, 2, 1], "v": [5, 6, 4, 1, 9, 0]}))
    do.identify_duplicates(d, ["k"], ["v"], primary="last", sequence_name="MatchSequence")
    assert vals(d.df["k"]) == [1, 1, 2, 2, 2, 3]
    assert vals(d.df["v"]) == [0, 6, 4, 5, 9, 1]
    assert vals(d.df["PrimaryLast"]) == [0, 1, 0, 0, 1, 1]
    assert vals(d.df["MatchSequence"]) == [1, 2, 1, 2, 3, 0]
    ref = d.df.duplicated("k", keep="last").map({True: 0.0, False: 1.0})
    assert vals(d.df["PrimaryLast"]) == vals(ref)
    e = Dataset(pd.DataFrame({"k": [3, 1, 1]}))
    do.identify_duplicates(e, ["k"], primary="first", move_to_top=True)
    assert vals(e.df["k"]) == [1, 1, 3] and vals(e.df["PrimaryFirst"]) == [1, 0, 1]


# ------------------------------------------------------------------ MERGE
def test_add_cases_pairing_and_source():
    a = Dataset(pd.DataFrame({"id": [1, 2], "x": [1.5, 2.5], "only_a": [7, 8]}))
    a.var("x").value_labels = {1.5: "low"}
    b = Dataset(pd.DataFrame({"ID": [3], "x": [3.5], "only_b": ["z"], "xx": [0]}))
    paired, un_a, un_b = do.match_variables(a, b)
    assert paired == [("id", "ID"), ("x", "x")]
    assert un_a == ["only_a"] and un_b == ["only_b", "xx"]
    do.add_cases(a, b, keep_active=["only_a"], keep_other=["only_b"], source_var="source01")
    assert a.names() == ["id", "x", "only_a", "only_b", "source01"]
    assert vals(a.df["id"]) == [1, 2, 3]
    assert vals(a.df["only_a"]) == [7, 8, None]
    assert vals(a.df["only_b"])[2] == "z"
    assert vals(a.df["source01"]) == [0, 0, 1]
    assert a.var("x").value_labels == {1.5: "low"}


def test_add_variables_by_key_outer_and_order():
    a = Dataset(pd.DataFrame({"k": [1, 2, 4], "x": [10, 20, 40]}))
    b = Dataset(pd.DataFrame({"k": [2, 3, 4], "y": [200, 300, 400], "x": [0, 0, 0]}))
    sec = do.add_variables(a, b, method="key", keys=["k"])
    assert a.names() == ["k", "x", "y"]  # duplicate x is excluded
    assert vals(a.df["k"]) == [1, 2, 3, 4]
    assert vals(a.df["x"]) == [10, 20, None, 40]
    assert vals(a.df["y"]) == [None, 200, 300, 400]
    ref = pd.merge(pd.DataFrame({"k": [1, 2, 4], "x": [10, 20, 40]}),
                   pd.DataFrame({"k": [2, 3, 4], "y": [200, 300, 400]}), on="k", how="outer")
    assert vals(a.df["y"]) == vals(ref["y"])
    assert any(r[0] == "Cases with a matching key" and r[1] == "2" for r in sec.items[-1].rows)
    c = Dataset(pd.DataFrame({"p": [1, 2, 3]}))
    d = Dataset(pd.DataFrame({"q": [9, 8]}))
    do.add_variables(c, d, method="order")
    assert vals(c.df["q"]) == [9, 8, None]


def test_add_variables_table_lookup_and_rename():
    cases = Dataset(pd.DataFrame({"school": [2, 1, 2, 3], "score": [5, 6, 7, 8]}))
    table = Dataset(pd.DataFrame({"school": [1, 2], "score": [100, 200], "town": ["A", "B"]}))
    do.add_variables(cases, table, method="table", keys=["school"],
                     include=["score", "town"], renames={"score": "sch_score"})
    assert vals(cases.df["school"]) == [1, 2, 2, 3]
    assert vals(cases.df["sch_score"]) == [100, 200, 200, None]
    assert vals(cases.df["town"]) == ["A", "B", "B", None]
    assert vals(cases.df["score"]) == [6, 5, 7, 8]
    # lookup table is the active dataset: cases come from the other file
    t2 = Dataset(pd.DataFrame({"school": [1, 2], "town": ["A", "B"]}))
    c2 = Dataset(pd.DataFrame({"school": [2, 2, 5], "score": [1, 2, 3]}))
    do.add_variables(t2, c2, method="table", keys=["school"], lookup="active")
    assert vals(t2.df["school"]) == [2, 2, 5]
    assert vals(t2.df["town"]) == ["B", "B", None]


# ------------------------------------------------------------------ AGGREGATE
def test_aggregate_against_pandas(ds):
    specs = [do.AggSpec("m", "MEAN", "y"), do.AggSpec("md", "MEDIAN", "y"),
             do.AggSpec("sm", "SUM", "x"), do.AggSpec("sd", "SD", "y"),
             do.AggSpec("mn", "MIN", "x"), do.AggSpec("mx", "MAX", "x"),
             do.AggSpec("f", "FIRST", "x"), do.AggSpec("l", "LAST", "x"),
             do.AggSpec("n", "N", "x"), do.AggSpec("nm", "NMISS", "x"),
             do.AggSpec("pg", "PGT", "y", (2,)), do.AggSpec("fl", "FLT", "y", (3,)),
             do.AggSpec("pin", "PIN", "y", (2, 4))]
    new, _ = do.aggregate(ds, ["grp"], specs, mode="new", n_break="N_BREAK")
    ref_df = pd.DataFrame({"grp": ds.df["grp"], "x": ds.analysis_series("x"),
                           "y": ds.df["y"]})
    g = ref_df.groupby("grp", dropna=False)
    assert vals(new.df["grp"]) == [None, 1, 2]
    row = {k: i for i, k in enumerate([np.nan, 1.0, 2.0])}
    for key, sub in g:
        i = row[1.0] if key == 1 else row[2.0] if key == 2 else row[np.nan]
        assert new.df["m"][i] == pytest.approx(sub["y"].mean())
        assert new.df["md"][i] == pytest.approx(sub["y"].median())
        assert new.df["sm"][i] == pytest.approx(sub["x"].sum())
        if len(sub) > 1:
            assert new.df["sd"][i] == pytest.approx(sub["y"].std())
        assert new.df["mn"][i] == pytest.approx(sub["x"].min())
        assert new.df["mx"][i] == pytest.approx(sub["x"].max())
        assert new.df["f"][i] == pytest.approx(sub["x"].dropna().iloc[0])
        assert new.df["l"][i] == pytest.approx(sub["x"].dropna().iloc[-1])
        assert new.df["n"][i] == sub["x"].count()
        assert new.df["nm"][i] == sub["x"].isna().sum()
        assert new.df["pg"][i] == pytest.approx(100 * (sub["y"] > 2).mean())
        assert new.df["fl"][i] == pytest.approx((sub["y"] < 3).mean())
        assert new.df["pin"][i] == pytest.approx(100 * sub["y"].between(2, 4).mean())
        assert new.df["N_BREAK"][i] == len(sub)


def test_aggregate_add_variables_and_strings(ds):
    do.aggregate(ds, ["grp"], [do.AggSpec("y_mean", "MEAN", "y"),
                               do.AggSpec("s_first", "FIRST", "s")], mode="add",
                 n_break="N_BREAK")
    assert ds.df["y_mean"][0] == pytest.approx(np.mean([1, 3, 9]))
    assert ds.df["y_mean"][7] == pytest.approx(4)  # missing break value is its own group
    assert vals(ds.df["s_first"])[:3] == ["b", "b", "c"]  # blank strings count as missing
    assert vals(ds.df["N_BREAK"]) == [3, 3, 4, 4, 4, 3, 4, 1]
    with pytest.raises(do.DataError):
        do.aggregate(ds, ["grp"], [do.AggSpec("bad", "MEAN", "s")])


def test_aggregate_columnwise():
    d = Dataset(pd.DataFrame({"g": [1, 1, 2], "v": [1, np.nan, 3]}))
    new, _ = do.aggregate(d, ["g"], [do.AggSpec("m", "MEAN", "v")], mode="new",
                          columnwise=True)
    assert vals(new.df["m"]) == [None, 3]


# ------------------------------------------------------------------ TRANSPOSE
def test_transpose_with_name_variable(ds):
    new, _ = do.transpose(ds, ["x", "y", "s"], name_var="id")
    assert new.names()[:3] == ["CASE_LBL", "V1", "V2"]
    assert vals(new.df["CASE_LBL"]) == ["x", "y"]  # string variable dropped
    assert vals(new.df["V6"]) == [99, 9]  # user-missing transposed as a value
    np.testing.assert_allclose(new.df.iloc[:, 1:].to_numpy(dtype=float),
                               ds.df[["x", "y"]].to_numpy(dtype=float).T)


# ------------------------------------------------------------------ RESTRUCTURE
def test_restructure_round_trip():
    wide = Dataset(pd.DataFrame({
        "id": [1, 2, 3], "sex": [1, 2, 1],
        "x.1": [5.0, 6.0, np.nan], "x.2": [7.0, 8.0, 9.0], "x.3": [1.0, 2.0, 3.0],
        "y.1": [0.1, 0.2, 0.3], "y.2": [1.1, 1.2, 1.3], "y.3": [2.1, 2.2, 2.3]}))
    long, _ = do.vars_to_cases(wide, [("x", ["x.1", "x.2", "x.3"]),
                                      ("y", ["y.1", "y.2", "y.3"])],
                               index_name="index1", id_var="id")
    assert long.names() == ["id", "sex", "index1", "x", "y"]
    assert long.n_cases == 9
    assert vals(long.df["index1"]) == [1, 2, 3] * 3
    assert vals(long.df["x"])[:3] == [5, 7, 1]
    back, _ = do.cases_to_vars(long, ["id"], ["index1"])
    assert back.names() == wide.names()
    pd.testing.assert_frame_equal(back.df, wide.df, check_dtype=False)
    by_index, _ = do.cases_to_vars(long, ["id"], ["index1"], groupby="index")
    assert by_index.names()[2:5] == ["x.1", "y.1", "x.2"]


def test_vars_to_cases_names_index_and_drop_null():
    d = Dataset(pd.DataFrame({"a": [1.0, np.nan], "b": [2.0, np.nan], "g": [7, 8]}))
    long, _ = do.vars_to_cases(d, [("v", ["a", "b"])], index_mode="names", drop_null=True)
    assert long.names() == ["id", "g", "index1", "v"]
    assert vals(long.df["index1"]) == ["a", "b"]
    assert vals(long.df["id"]) == [1, 1]
    no_index, _ = do.cases_to_vars(Dataset(pd.DataFrame({"id": [1, 1, 2], "v": [4, 5, 6]})),
                                   ["id"])
    assert no_index.names() == ["id", "v.1", "v.2"]
    assert vals(no_index.df["v.2"]) == [5, None]
    with pytest.raises(do.DataError):
        do.cases_to_vars(Dataset(pd.DataFrame({"id": [1, 1], "t": [1, 1], "v": [1, 2]})),
                         ["id"], ["t"])


def test_selftest_runs():
    from rsstatchem.ui.dialogs.data_ops import selftest
    selftest()
