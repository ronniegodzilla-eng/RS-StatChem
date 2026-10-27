"""WP6: Custom Tables vs pandas crosstab/groupby, multiple response vs hand counts, tests."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from rsstatchem.core import ctables as ct
from rsstatchem.core import mresponse as mr
from rsstatchem.core.dataset import Dataset


@pytest.fixture()
def survey():
    rng = np.random.default_rng(42)
    n = 150
    sex = rng.integers(1, 3, n).astype(float)
    region = rng.integers(1, 4, n).astype(float)
    edu = rng.integers(1, 4, n).astype(float)
    score = rng.normal(60, 8, n) + 6 * (region == 3)
    edu[[0, 5]] = np.nan
    score[7] = np.nan
    df = pd.DataFrame({"sex": sex, "region": region, "edu": edu, "score": score,
                       "income": rng.gamma(3, 2, n)})
    ds = Dataset(df)
    for name, labels in (("sex", {1.0: "Male", 2.0: "Female"}),
                         ("region", {1.0: "North", 2.0: "Centre", 3.0: "South"}),
                         ("edu", {1.0: "SMA", 2.0: "S1", 3.0: "S2"})):
        ds.var(name).measure = "nominal"
        ds.var(name).value_labels = labels
    return ds


def test_counts_and_percentages_match_pandas(survey):
    spec = ct.CTSpec(rows=["edu"], cols=["sex"],
                     cat_stats=("count", "col_pct", "row_pct", "table_pct"), totals=True)
    res = ct.compute(survey, spec)
    df = survey.df.dropna(subset=["edu"])
    counts = pd.crosstab(df.edu, df.sex)
    col_pct = pd.crosstab(df.edu, df.sex, normalize="columns") * 100
    row_pct = pd.crosstab(df.edu, df.sex, normalize="index") * 100
    tab_pct = pd.crosstab(df.edu, df.sex, normalize="all") * 100
    labels = {1.0: "SMA", 2.0: "S1", 3.0: "S2"}
    sexes = {1.0: "Male", 2.0: "Female"}
    for e, elab in labels.items():
        for s, slab in sexes.items():
            key = ((f"edu={elab}",), (f"sex={slab}",))
            assert res.value(*key, "count") == counts.loc[e, s]
            assert res.value(*key, "col_pct") == pytest.approx(col_pct.loc[e, s])
            assert res.value(*key, "row_pct") == pytest.approx(row_pct.loc[e, s])
            assert res.value(*key, "table_pct") == pytest.approx(tab_pct.loc[e, s])
    # totals: Total column = row counts, Total row = column counts, grand total
    assert res.value(("edu=S1",), ("sex=Total",), "count") == counts.loc[2.0].sum()
    assert res.value(("edu=Total",), ("sex=Female",), "count") == counts[2.0].sum()
    assert res.value(("edu=Total",), ("sex=Total",), "count") == counts.values.sum()
    assert res.value(("edu=Total",), ("sex=Male",), "col_pct") == pytest.approx(100.0)


def test_nested_rows_and_subtable_percentages(survey):
    spec = ct.CTSpec(rows=["region", "edu"], cols=["sex"], cat_stats=("count", "col_pct"))
    res = ct.compute(survey, spec)
    df = survey.df.dropna(subset=["edu"])
    for r in (1.0, 2.0, 3.0):
        sub = df[df.region == r]
        counts = pd.crosstab(sub.edu, sub.sex)
        pct = pd.crosstab(sub.edu, sub.sex, normalize="columns") * 100
        rlab = {1.0: "North", 2.0: "Centre", 3.0: "South"}[r]
        key = ((f"region={rlab}", "edu=S2"), ("sex=Male",))
        assert res.value(*key, "count") == counts.loc[3.0, 1.0]
        assert res.value(*key, "col_pct") == pytest.approx(pct.loc[3.0, 1.0])


def test_scale_summaries_match_groupby(survey):
    spec = ct.CTSpec(rows=["score", "income"], row_mode="stack", cols=["region"],
                     scale_stats=("mean", "median", "std", "n", "min", "max", "sum", "se"))
    res = ct.compute(survey, spec)
    g = survey.df.groupby("region")["score"]
    for code, lab in ((1.0, "North"), (3.0, "South")):
        assert res.value(("score",), (f"region={lab}",), "mean") == pytest.approx(
            g.mean()[code])
        assert res.value(("score",), (f"region={lab}",), "median") == pytest.approx(
            g.median()[code])
        assert res.value(("score",), (f"region={lab}",), "std") == pytest.approx(g.std()[code])
        assert res.value(("score",), (f"region={lab}",), "n") == g.count()[code]
        assert res.value(("score",), (f"region={lab}",), "se") == pytest.approx(
            g.sem()[code])
    gi = survey.df.groupby("region")["income"]
    assert res.value(("income",), ("region=Centre",), "sum") == pytest.approx(gi.sum()[2.0])
    assert res.value(("income",), ("region=Centre",), "max") == pytest.approx(gi.max()[2.0])


def test_chi_square_matches_scipy(survey):
    spec = ct.CTSpec(rows=["region"], cols=["sex"], chisq=True)
    res = ct.compute(survey, spec)
    chi2, p, dof, _ = stats.chi2_contingency(pd.crosstab(survey.df.region, survey.df.sex),
                                             correction=False)
    (_, _, got_chi2, got_df, got_p, _), = res.chisq
    assert got_chi2 == pytest.approx(chi2) and got_df == dof and got_p == pytest.approx(p)


def test_column_proportions_letters():
    # group A: 80% yes, B: 20% yes (n = 50 each) -> clearly different
    yes = [1.0] * 40 + [2.0] * 10 + [1.0] * 10 + [2.0] * 40
    grp = [1.0] * 50 + [2.0] * 50
    ds = Dataset(pd.DataFrame({"ans": yes, "grp": grp}))
    for name in ("ans", "grp"):
        ds.var(name).measure = "nominal"
    res = ct.compute(ds, ct.CTSpec(rows=["ans"], cols=["grp"], compare_props=True))
    assert res.col_letters == {0: "A", 1: "B"}
    assert res.letters[(0, 0)] == "B"   # ans=1: column A larger than B
    assert res.letters[(1, 1)] == "A"   # ans=2: column B larger than A
    # hand z-test: p1=.8, p2=.2, pooled .5, se = sqrt(.25 * 2/50)
    z = (0.8 - 0.2) / math.sqrt(0.25 * (2 / 50))
    assert 2 * stats.norm.sf(z) < 0.05


def test_column_means_letters_use_pooled_variance(survey):
    res = ct.compute(survey, ct.CTSpec(rows=["score"], cols=["region"], compare_means=True))
    groups = [survey.df.score[survey.df.region == r].dropna() for r in (1.0, 2.0, 3.0)]
    df_e = sum(len(g) for g in groups) - 3
    mse = sum(((g - g.mean()) ** 2).sum() for g in groups) / df_e
    t = (groups[2].mean() - groups[0].mean()) / math.sqrt(mse * (1 / len(groups[2])
                                                              + 1 / len(groups[0])))
    p = min(1, 2 * stats.t.sf(abs(t), df_e) * 3)
    assert ("A" in res.letters.get((0, 2), "")) == (p < 0.05)
    table = ct.to_table(survey, ct.CTSpec(rows=["score"], cols=["region"],
                                          compare_means=True), res)
    assert any("Bonferroni" in f for f in table.footnotes)
    assert table.columns[-1] == ("region", "South (C)")


def test_layers_make_one_table_per_category(survey):
    sec = ct.custom_tables(survey, ct.CTSpec(rows=["edu"], cols=["sex"], layers=["region"]))
    tables = [i for i in sec.items if hasattr(i, "rows")]
    assert len(tables) == 3
    assert tables[0].title == "Custom Table (region = North)"


def test_scale_must_be_innermost(survey):
    with pytest.raises(ct.CTableError):
        ct.compute(survey, ct.CTSpec(rows=["score", "edu"]))


@pytest.fixture()
def mrdata():
    # 6 respondents, 3 media options (dichotomies) and two "first/second choice" codes
    df = pd.DataFrame({"tv": [1, 1, 0, 0, 1, 0], "radio": [0, 1, 1, 0, 1, 0],
                       "web": [1, 1, 1, 0, 0, 0], "c1": [1, 2, 3, 1, np.nan, 2],
                       "c2": [2, 2, 1, np.nan, np.nan, 3], "sex": [1, 2, 1, 2, 1, 2]},
                      dtype=float)
    ds = Dataset(df)
    ds.var("tv").label = "Television"
    ds.var("c1").value_labels = {1.0: "Price", 2.0: "Quality", 3.0: "Brand"}
    ds.var("sex").value_labels = {1.0: "M", 2.0: "F"}
    ds.var("sex").measure = "nominal"
    mr.save_sets(ds, [mr.MRSet("$media", "Media used", "dichotomy", ["tv", "radio", "web"],
                               1.0),
                      mr.MRSet("$why", "Reasons", "category", ["c1", "c2"], 1.0, [1, 3])])
    return ds


def test_mr_dichotomy_frequencies_by_hand(mrdata):
    f = mr.frequency_counts(mrdata, "$media")
    # valid cases: rows 0,1,2,4 (row 3 and 5 chose nothing) -> 4 respondents
    assert list(f["n"]) == [3, 3, 3]
    assert list(f["label"]) == ["Television", "radio", "web"]
    assert f["pct_responses"].tolist() == pytest.approx([100 / 3] * 3)
    assert f["pct_cases"].tolist() == pytest.approx([75.0] * 3)
    sec = mr.frequencies(mrdata, ["$media"])
    summary, table = sec.items[1], sec.items[2]
    assert summary.rows[0][1] == "4" and summary.rows[0][3] == "2"
    assert table.rows[-1][2] == "9" and table.rows[-1][4] == "225.0%"


def test_mr_category_set_by_hand(mrdata):
    f = mr.frequency_counts(mrdata, "$why")
    # Price: rows 0,2,3 ; Quality: rows 0,1(twice -> once),5 ; Brand: rows 2,5
    assert list(f.index) == [1.0, 2.0, 3.0]
    assert list(f["n"]) == [3, 3, 2]
    assert list(f["label"]) == ["Price", "Quality", "Brand"]
    assert f["pct_cases"].tolist() == pytest.approx([60.0, 60.0, 40.0])  # 5 respondents


def test_mr_crosstabs_by_hand(mrdata):
    r, c, counts, valid = mr.crosstab_counts(mrdata, "$media", "sex")
    # media by sex: tv -> rows 0,1,4 (sex 1,2,1); radio -> 1,2,4 (2,1,1); web -> 0,1,2 (1,2,1)
    assert counts.tolist() == [[2, 1], [2, 1], [2, 1]]
    sec = mr.crosstabs(mrdata, ["$media"], ["sex"], col_pct=True)
    t = sec.items[-1]
    count_rows = [row for row in t.rows if row[2] == "Count"]
    assert count_rows[-1][3:] == ["3", "1", "4"]         # respondents per sex, total
    pct_rows = [row for row in t.rows if row[2].startswith("% within")]
    assert pct_rows[0][3] == "66.7%"                       # tv among male respondents (2/3)


def test_mr_set_in_custom_tables(mrdata):
    res = ct.compute(mrdata, ct.CTSpec(rows=["$media"], cols=["sex"],
                                       cat_stats=("count", "col_pct")))
    assert res.value(("Media used=Television",), ("sex=M",), "count") == 2
    # column % of respondents: male respondents of $media = rows 0,2,4 -> 3
    assert res.value(("Media used=Television",), ("sex=M",), "col_pct") == \
        pytest.approx(200 / 3)


def test_mr_validation(mrdata):
    assert mr.validate(mrdata, mr.MRSet("media", "", "dichotomy", ["tv", "web"])) is not None
    assert mr.validate(mrdata, mr.MRSet("$x", "", "dichotomy", ["tv"])) is not None
    assert mr.validate(mrdata, mr.MRSet("$media", "", "dichotomy", ["tv", "web"]),
                       mr.get_sets(mrdata)) is not None
    assert mr.validate(mrdata, mr.MRSet("$ok", "", "dichotomy", ["tv", "web"])) is None


def test_custom_tables_dialog_builds_spec(qapp, survey):
    from rsstatchem.ui.dialogs.ctables import CustomTablesDialog

    dlg = CustomTablesDialog(survey)
    dlg.set_selected("rows", ["edu"])
    dlg.set_selected("cols", ["sex"])
    dlg.opts["cat_stats"] = ["count", "col_pct"]
    dlg.update_preview()
    assert "Column N %" in dlg.preview.toPlainText()
    sec = dlg.compute()
    assert sec.items[1].title == "Custom Table"
