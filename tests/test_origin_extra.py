"""Graph & Analysis module, second round: row-range plots, sparklines, collapsible
recalculable reports, matrix books, Origin-style Statistics dialogs (numbers cross-checked
against scipy / statsmodels), Project Explorer drag and drop and the Format toolbar."""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from rsstatchem.core.analysis import matrix as mx
from rsstatchem.ui.origin import analysis as an
from rsstatchem.ui.origin import recalc
from rsstatchem.ui.origin import stat_tests as st
from rsstatchem.ui.origin.model import (MatrixBook, MatrixSheet, Project, ReportSheet, Workbook,
                                        load_project, parse_range, range_text, save_project)
from rsstatchem.ui.origin.plotting import (MATRIX_PLOTS, plot_columns, plot_matrix, plot_range,
                                           refresh_links)


def _book(n: int = 20, seed: int = 1) -> Workbook:
    rng = np.random.default_rng(seed)
    book = Workbook.new("Book1")
    ws = book.sheets[0]
    x = np.arange(1, n + 1, dtype=float)
    ws.set_column_values(0, list(x))
    ws.set_column_values(1, list(2.0 * x + 1.0 + rng.normal(0, 0.5, n)))
    ws.columns[1].long_name = "Signal"
    ws.add_column("Y")
    ws.set_column_values(2, list(rng.normal(10, 2, n)))
    ws.add_column("Y")
    ws.set_column_values(3, list(rng.normal(11, 2, n)))
    return book


@pytest.fixture
def ctx(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("RSSTATCHEM_HOME", str(tmp_path))
    from rsstatchem.app import AppContext

    context = AppContext(qapp)
    monkeypatch.setattr(context.app, "quit", lambda: None)
    return context


def _no_prompts(monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    def boom(*args, **kwargs):
        raise AssertionError("unexpected modal dialog")
    for name in ("question", "warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name, boom)
    monkeypatch.setattr(QMessageBox, "exec", boom)


# ====================================================================== ranges
def test_range_notation_round_trip():
    r = parse_range("[Book1]Sheet1!B[3:10]")
    assert (r.book, r.sheet, r.col, r.r1, r.r2) == ("Book1", "Sheet1", "B", 3, 10)
    assert r.text == "[Book1]Sheet1!B[3:10]" and r.rows == (3, 10)
    assert parse_range("C", "Book2", "Data").text == "[Book2]Data!C"
    assert parse_range("Sheet1!col(D)[4:*]", "Book1").rows == (4, 0)
    assert parse_range('"Signal"', "Book1", "Sheet1").col == '"Signal"'
    assert parse_range("B[10:3]").rows == (3, 10)
    assert range_text("B", "S", "A") == "[B]S!A"
    with pytest.raises(ValueError):
        parse_range("[Book1]Sheet1!")


def test_plot_and_fit_use_only_the_selected_rows():
    book = _book()
    ws = book.sheets[0]
    g = plot_columns(book, ws, [1], "scatter", "G", rows=(3, 10))
    p = g.layers[0].plots[0]
    assert p.y == "[Book1]Sheet1!B[3:10]" and p.x == "[Book1]Sheet1!A[3:10]"
    assert p.options["source"]["rows"] == [3, 10]
    assert plot_range(p) == "[Book1]Sheet1!B[3:10]"
    assert "[Book1]Sheet1!B[3:10]" in p.options["data_range"]
    assert list(g.values(p.x)) == list(np.arange(3, 11, dtype=float))
    # the graph stays linked to those rows only
    ws.set_cell(4, 1, 500.0)     # row 5: inside the range
    ws.set_cell(15, 1, -500.0)   # row 16: outside
    assert refresh_links(g, {"Book1": book})
    y = g.values(p.y)
    assert len(y) == 8 and y[2] == 500.0 and -500.0 not in y
    # a fit of the range uses rows 3..10
    data = an.xy_data(book, ws, 1, rows=(3, 10))
    assert data.range_text.endswith("[3:10]") and len(data.x) == 8
    out = an.run_fit(data, "Line", "linear")
    xs, ys = ws.columns[0].numeric()[2:10], ws.columns[1].numeric()[2:10]
    slope, icpt = np.polyfit(xs, ys, 1)
    assert out.result.params["Slope"] == pytest.approx(slope)
    assert out.report.table("Input Data")["rows"][1][1] == "[Book1]Sheet1!B[3:10]"
    assert out.report.op["input"]["rows"] == [3, 10]


# ====================================================================== matrices (core)
def test_matrix_operations():
    data = np.arange(12, dtype=float).reshape(3, 4)
    mapping = (0.0, 3.0, 10.0, 30.0)
    t, m = mx.transpose(data, mapping)
    assert t.shape == (4, 3) and m == (10.0, 30.0, 0.0, 3.0) and t[1, 2] == data[2, 1]
    f, _ = mx.flip(data, mapping, "horizontal")
    assert list(f[0]) == [3, 2, 1, 0]
    assert list(mx.flip(data, mapping, "vertical")[0][0]) == [8, 9, 10, 11]
    r, rm = mx.rotate(data, mapping, 90)
    assert np.array_equal(r, np.rot90(data)) and rm == (-8.5, 11.5, 18.5, 21.5)
    assert np.array_equal(mx.rotate(data, mapping, 180)[0], data[::-1, ::-1])
    e = mx.expand(data, 2, 2)
    assert e.shape == (5, 7) and e[0, 0] == 0 and e[0, 1] == pytest.approx(0.5) and \
        e[4, 6] == 11
    s = mx.shrink(np.arange(16, dtype=float).reshape(4, 4), 2, 2)
    assert s.tolist() == [[2.5, 4.5], [10.5, 12.5]]
    v = mx.evaluate("x^2 + 10*y + i*0 + j*0", (3, 4), mapping)
    assert v[0, 0] == pytest.approx(100.0) and v[2, 3] == pytest.approx(9 + 300)
    assert mx.evaluate("Mat(1) = i*j", (2, 2), mapping)[1, 1] == 4
    with pytest.raises(mx.MatrixError):
        mx.evaluate("__import__('os')", (2, 2), mapping)
    with pytest.raises(mx.MatrixError):
        mx.evaluate("foo(x)", (2, 2), mapping)
    x, y, z = mx.to_xyz(data, mapping)
    assert len(z) == 12 and x[1] == 1.0 and y[4] == 20.0 and z[5] == 5.0
    # XYZ gridding: regular data are pivoted exactly, scattered data interpolated
    gx, gy = np.meshgrid([1.0, 2.0, 3.0], [5.0, 6.0])
    gz = gx * 10 + gy
    d, mp, regular = mx.from_xyz(gx.ravel(), gy.ravel(), gz.ravel())
    assert regular and d.shape == (2, 3) and mp == (1.0, 3.0, 5.0, 6.0) and d[1, 2] == 36.0
    rng = np.random.default_rng(0)
    sx, sy = rng.uniform(0, 1, 200), rng.uniform(0, 1, 200)
    d2, _, regular2 = mx.from_xyz(sx, sy, sx + sy, 12, 9, "linear")
    assert not regular2 and d2.shape == (9, 12)
    ok = np.isfinite(d2)
    xx, yy = np.meshgrid(np.linspace(sx.min(), sx.max(), 12), np.linspace(sy.min(), sy.max(), 9))
    assert np.allclose(d2[ok], (xx + yy)[ok], atol=1e-6)
    dm, mm = mx.from_columns([np.array([1.0, 2.0]), np.array([3.0, np.nan, 5.0])],
                             np.array([10.0, 20.0, 30.0]), [0.5, 1.5])
    assert dm.shape == (3, 2) and mm == (0.5, 1.5, 10.0, 30.0) and np.isnan(dm[2, 0])


@pytest.mark.parametrize("kind", list(MATRIX_PLOTS))
def test_matrix_plots_render_and_stay_linked(kind):
    from rsstatchem.core.graph import export_bytes

    mb = MatrixBook.new("MBook1", 12, 15)
    ms = mb.sheets[0]
    ms.data = mx.evaluate("sin(x/3)*cos(y/4)", ms.data.shape, (ms.x1, ms.x2, ms.y1, ms.y2))
    ms.long_name, ms.x_name, ms.y_name = "Intensity", "Wavelength", "Time"
    g = plot_matrix(mb, ms, kind, "G")
    layer = g.layers[0]
    assert layer.kind == MATRIX_PLOTS[kind][2] and layer.plots[0].type == MATRIX_PLOTS[kind][1]
    assert layer.x.title == "Wavelength" and layer.y.title == "Time"
    assert export_bytes(g, "png", dpi=40)[:4] == b"\x89PNG"
    ms.data[0, 0] = 9.0
    assert refresh_links(g, {"MBook1": mb})
    assert np.nanmax(g.values(layer.plots[0].z)) == 9.0


def test_project_round_trip_with_matrix_ranges_and_report_state(tmp_path):
    book = _book()
    ws = book.sheets[0]
    ws.sparklines = True
    out = an.run_fit(an.xy_data(book, ws, 1, rows=(2, 12)), "Line", "linear", mode="auto")
    out.report.items[0]["collapsed"] = True
    p = Project()
    p.add(book)
    from rsstatchem.ui.origin.model import GraphDoc
    p.add(GraphDoc("Graph1", plot_columns(book, ws, [1], "line", "Graph1", rows=(4, 9))))
    mb = MatrixBook.new("MBook1", 5, 6)
    mb.sheets[0].data[2, 3] = 7.5
    mb.sheets[0].x1, mb.sheets[0].x2, mb.sheets[0].long_name = -1.0, 1.0, "Height"
    p.add(mb)
    recalc.stamp(p, out.report)
    path = tmp_path / "x.rsop"
    save_project(p, path)
    q, _ = load_project(path)
    b2 = q.windows["Book1"]
    assert b2.sheets[0].sparklines is True
    rep = b2.sheet(out.report.name)
    assert rep.op["mode"] == "auto" and rep.op["input"]["rows"] == [2, 12]
    assert rep.items[0]["collapsed"] is True and rep.op["hash"] == out.report.op["hash"]
    g2 = q.windows["Graph1"].graph
    assert g2.layers[0].plots[0].options["source"]["rows"] == [4, 9]
    assert "[Book1]Sheet1!B[4:9]" in g2.data
    m2 = q.windows["MBook1"]
    assert isinstance(m2, MatrixBook) and m2.sheets[0].data.shape == (5, 6)
    assert m2.sheets[0].data[2, 3] == 7.5 and np.isnan(m2.sheets[0].data[0, 0])
    assert (m2.sheets[0].x1, m2.sheets[0].long_name) == (-1.0, "Height")
    assert not recalc.stale_reports(q)


# ====================================================================== statistics vs scipy
def test_descriptive_statistics_quantities_match_scipy():
    book = _book()
    ws = book.sheets[0]
    v = ws.columns[2].numeric()[:20]
    d = st.describe(v)
    assert d["mean"] == pytest.approx(v.mean())
    assert d["sd"] == pytest.approx(v.std(ddof=1))
    assert d["skew"] == pytest.approx(stats.skew(v, bias=False))
    assert d["kurt"] == pytest.approx(stats.kurtosis(v, bias=False))
    assert d["median"] == pytest.approx(np.median(v))
    assert d["iqr"] == pytest.approx(stats.iqr(v))
    assert d["cv"] == pytest.approx(v.std(ddof=1) / v.mean())
    assert d["ucl"] - d["mean"] == pytest.approx(stats.t.ppf(0.975, 19) * stats.sem(v))
    assert st.describe(v, "n")["sd"] == pytest.approx(v.std(ddof=0))
    rep, text = st.run("desc", {"Book1": book},
                       {"ranges": ["[Book1]Sheet1!C", "[Book1]Sheet1!D[1:10]"],
                        "quantities": ["n", "mean", "sd", "gmean"], "histogram": True,
                        "box": True})
    table = rep.table("Descriptive Statistics")
    assert table["columns"] == ["", "N total", "Mean", "Standard Deviation", "Geometric Mean"]
    assert table["rows"][1][1] == "10" and table["rows"][1][0].endswith("[1:10]")
    assert float(table["rows"][0][4]) == pytest.approx(stats.gmean(v), rel=1e-5)
    assert any(i.get("group") == "Histograms" for i in rep.items)
    assert rep.op["kind"] == "stats" and "Statistics on Columns" in text


def test_hypothesis_tests_match_scipy():
    book = _book()
    books = {"Book1": book}
    ws = book.sheets[0]
    a, b = ws.columns[2].numeric()[:20], ws.columns[3].numeric()[:20]
    rep, _ = st.run("ttest1", books, {"ranges": ["[Book1]Sheet1!C"], "mean": 9.5})
    row = rep.table("t-Test Statistics")["rows"][0]
    res = stats.ttest_1samp(a, 9.5)
    assert float(row[1]) == pytest.approx(res.statistic, rel=1e-4)
    assert float(row[3]) == pytest.approx(res.pvalue, rel=1e-3)
    rep, _ = st.run("ttest2", books, {"ranges": ["[Book1]Sheet1!C", "[Book1]Sheet1!D"],
                                      "tail": "lower"})
    rows = rep.table("t-Test Statistics")["rows"]
    eq = stats.ttest_ind(a, b, alternative="less")
    welch = stats.ttest_ind(a, b, equal_var=False, alternative="less")
    assert float(rows[0][1]) == pytest.approx(eq.statistic, rel=1e-4)
    assert float(rows[0][3]) == pytest.approx(eq.pvalue, rel=1e-3)
    assert float(rows[1][3]) == pytest.approx(welch.pvalue, rel=1e-3)
    lev = rep.table("Equality of Variance Test")["rows"][0]
    assert float(lev[4]) == pytest.approx(stats.levene(a, b, center="mean").pvalue, rel=1e-3)
    rep, _ = st.run("ttest_pair", books, {"range1": "[Book1]Sheet1!C",
                                          "range2": "[Book1]Sheet1!D"})
    row = rep.table("t-Test Statistics")["rows"][0]
    assert float(row[3]) == pytest.approx(stats.ttest_rel(a, b).pvalue, rel=1e-3)
    # indexed two-sample data: values + group column
    w2 = book.add_sheet()
    w2.set_column_values(0, ["a"] * 6 + ["b"] * 6)
    vals = list(a[:6]) + list(b[:6])
    w2.set_column_values(1, vals)
    rep, _ = st.run("ttest2", books, {"form": "indexed", "data": f"[Book1]{w2.name}!B",
                                      "factor": f"[Book1]{w2.name}!A"})
    assert float(rep.table("t-Test Statistics")["rows"][0][3]) == pytest.approx(
        stats.ttest_ind(a[:6], b[:6]).pvalue, rel=1e-3)


def test_normality_and_correlation_match_references():
    from statsmodels.stats.diagnostic import lilliefors, normal_ad

    book = _book(30, seed=5)
    books = {"Book1": book}
    ws = book.sheets[0]
    v = ws.columns[2].numeric()[:30]
    rep, _ = st.run("normality", books, {"ranges": ["[Book1]Sheet1!C"],
                                         "tests": list(st.NORMALITY_TESTS)})
    assert float(rep.table("Shapiro-Wilk")["rows"][0][3]) == pytest.approx(
        stats.shapiro(v).pvalue, rel=1e-3)
    assert float(rep.table("Anderson-Darling")["rows"][0][2]) == pytest.approx(
        normal_ad(v)[0], rel=1e-4)
    assert float(rep.table("Lilliefors")["rows"][0][2]) == pytest.approx(
        lilliefors(v, pvalmethod="table")[0], rel=1e-4)
    assert float(rep.table("D'Agostino-K squared")["rows"][0][3]) == pytest.approx(
        stats.normaltest(v).pvalue, rel=1e-3)
    ks = stats.kstest((v - v.mean()) / v.std(ddof=1), "norm")
    assert float(rep.table("Kolmogorov-Smirnov")["rows"][0][2]) == pytest.approx(
        ks.statistic, rel=1e-4)
    assert any(i.get("group") == "Q-Q Plots" for i in rep.items)
    a, b = ws.columns[1].numeric()[:30], ws.columns[3].numeric()[:30]
    rep, _ = st.run("correlation", books, {"ranges": ["[Book1]Sheet1!B", "[Book1]Sheet1!D"],
                                           "methods": ["pearson", "spearman", "kendall"],
                                           "flag": False})
    for method, ref in (("Pearson", stats.pearsonr(a, b)), ("Spearman", stats.spearmanr(a, b)),
                        ("Kendall", stats.kendalltau(a, b))):
        r = float(rep.table(f"{method} Correlation Coefficients")["rows"][0][2])
        assert r == pytest.approx(ref[0], abs=1e-5)
        p = float(rep.table(f"{method} Correlation P-values")["rows"][0][2])
        assert p == pytest.approx(ref[1], rel=1e-3)


def test_anova_and_means_comparison_match_statsmodels():
    import pandas as pd
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    rng = np.random.default_rng(11)
    book = Workbook.new("Book1")
    ws = book.sheets[0]
    groups = [rng.normal(m, 1.0, 8) for m in (5.0, 6.5, 5.2)]
    for k, g in enumerate(groups):
        if k >= len(ws.columns):
            ws.add_column("Y")
        ws.set_column_values(k, list(g))
        ws.columns[k].designation = "Y"
        ws.columns[k].long_name = f"G{k + 1}"
    books = {"Book1": book}
    rep, text = st.run("anova1", books, {"ranges": ["[Book1]Sheet1!A", "[Book1]Sheet1!B",
                                                    "[Book1]Sheet1!C"],
                                         "means": ["tukey", "bonferroni", "scheffe"]})
    f, p = stats.f_oneway(*groups)
    model_row = rep.table("Overall ANOVA")["rows"][0]
    assert float(model_row[4]) == pytest.approx(f, rel=1e-4)
    assert float(model_row[5]) == pytest.approx(p, rel=1e-3)
    tuk = pairwise_tukeyhsd(np.concatenate(groups), np.repeat(["G1", "G2", "G3"], 8))
    ours = {r[0]: float(r[4]) for r in rep.table("Tukey Test")["rows"]}
    ref = {f"{b}  {a}": float(pv) for (a, b), pv in
           zip([(r[0], r[1]) for r in tuk._results_table.data[1:]], tuk.pvalues)}
    for key, pv in ref.items():
        assert ours[key] == pytest.approx(pv, abs=2e-3)
    assert rep.table("Bonferroni Test") is not None and "One-Way ANOVA" in text
    assert [i["group"] for i in rep.items if i["title"].endswith("Test")][:3] == \
        ["Means Comparisons"] * 3
    # two-way ANOVA (indexed, unbalanced) against statsmodels type III
    w2 = book.add_sheet()
    fa = ["lo", "hi"] * 12
    fb = (["x"] * 4 + ["y"] * 4 + ["z"] * 4) * 2
    y = [rng.normal(3 + (a_ == "hi") + 0.5 * (b_ == "z"), 0.4) for a_, b_ in zip(fa, fb)]
    y[0] = np.nan
    w2.set_column_values(0, fa)
    w2.set_column_values(1, fb)
    w2.add_column("Y")
    w2.set_column_values(2, y)
    rep2, _ = st.run("anova2", books, {"data": f"[Book1]{w2.name}!C",
                                       "factor_a": f"[Book1]{w2.name}!A",
                                       "factor_b": f"[Book1]{w2.name}!B",
                                       "interaction": True, "means": ["tukey"]})
    frame = pd.DataFrame({"A": fa, "B": fb, "y": y}).dropna()
    ref2 = anova_lm(smf.ols("y ~ C(A, Sum) * C(B, Sum)", data=frame).fit(), typ=3)
    rows = {r[0]: r for r in rep2.table("Overall ANOVA")["rows"]}
    assert float(rows["A"][4]) == pytest.approx(ref2.loc["C(A, Sum)", "F"], rel=1e-4)
    assert float(rows["A * B"][5]) == pytest.approx(ref2.loc["C(A, Sum):C(B, Sum)", "PR(>F)"],
                                                    rel=1e-3)
    assert rep2.table("Tukey Test (A)") is not None


def test_nonparametric_and_frequency_counts():
    book = _book(25, seed=9)
    books = {"Book1": book}
    ws = book.sheets[0]
    a, b, c = (ws.columns[k].numeric()[:25] for k in (1, 2, 3))
    rep, _ = st.run("nonparam", books, {"test": "mann_whitney",
                                        "ranges": ["[Book1]Sheet1!C", "[Book1]Sheet1!D"]})
    ref = stats.mannwhitneyu(b, c, method="asymptotic")
    row = rep.table("Test Statistics")["rows"][0]
    assert float(row[1]) == pytest.approx(ref.statistic) and \
        float(row[3]) == pytest.approx(ref.pvalue, rel=1e-3)
    rep, _ = st.run("nonparam", books, {"test": "kruskal", "ranges": [
        "[Book1]Sheet1!B", "[Book1]Sheet1!C", "[Book1]Sheet1!D"]})
    assert float(rep.table("Test Statistics")["rows"][0][3]) == pytest.approx(
        stats.kruskal(a, b, c).pvalue, rel=1e-3)
    rep, _ = st.run("nonparam", books, {"test": "wilcoxon",
                                        "ranges": ["[Book1]Sheet1!C", "[Book1]Sheet1!D"]})
    assert float(rep.table("Test Statistics")["rows"][0][3]) == pytest.approx(
        stats.wilcoxon(b, c).pvalue, rel=1e-3)
    rep, _ = st.run("nonparam", books, {"test": "friedman", "ranges": [
        "[Book1]Sheet1!B", "[Book1]Sheet1!C", "[Book1]Sheet1!D"]})
    assert float(rep.table("Test Statistics")["rows"][0][3]) == pytest.approx(
        stats.friedmanchisquare(a, b, c).pvalue, rel=1e-3)
    out, text = st.frequency_counts(books, {"range": "[Book1]Sheet1!C", "auto": False,
                                            "from": 4.0, "to": 16.0, "step": 2.0})
    counts, _ = np.histogram(b[(b >= 4) & (b <= 16)], bins=np.arange(4.0, 16.1, 2.0))
    assert out.columns[0].long_name == "Bin Center" and out.columns[3].long_name == "Count"
    assert list(out.columns[3].numeric()[:6]) == list(counts.astype(float))
    assert out.columns[4].numeric()[5] == counts.sum()
    ws.add_column("L")
    ws.set_column_values(4, ["red", "blue", "red", "green", "red"])
    out2, _ = st.frequency_counts(books, {"range": "[Book1]Sheet1!E"})
    assert out2.columns[0].texts(3) == ["blue", "green", "red"]
    assert list(out2.columns[1].numeric()[:3]) == [1.0, 1.0, 3.0]
    ws2, added, _ = st.stats_on_rows(books, {"ranges": ["[Book1]Sheet1!C[1:5]",
                                                        "[Book1]Sheet1!D[1:5]"],
                                             "quantities": ["mean", "max"]})
    assert ws2.columns[added[0]].numeric()[0] == pytest.approx((b[0] + c[0]) / 2)
    assert np.isnan(ws2.columns[added[1]].numeric()[6])


def test_recalculate_modes():
    book = _book()
    p = Project()
    p.add(book)
    books = p.data_windows()
    rep, _ = st.run("desc", books, {"ranges": ["[Book1]Sheet1!C"], "quantities": ["mean"]},
                    mode="manual")
    recalc.stamp(p, rep)
    rep.items[1]["collapsed"] = True
    ws = book.sheets[0]
    assert not recalc.stale_reports(p, "Book1")
    ws.set_cell(0, 2, 1000.0)
    assert recalc.stale_reports(p, "Book1") == [(book, rep)]
    assert not recalc.stale_reports(p, "Other")
    ws.set_cell(0, 3, 1.0)   # column D is not an input
    text = recalc.recalculate(p, book, rep)
    new_mean = float(rep.table("Descriptive Statistics")["rows"][0][1])
    assert new_mean == pytest.approx(np.mean(ws.columns[2].numeric()[:20]), rel=1e-5)
    assert rep.items[1]["collapsed"] is True and not rep.dirty and "Recalculate" in text
    rep.op["mode"] = "none"
    ws.set_cell(1, 2, 5.0)
    assert not recalc.stale_reports(p)
    # a fit report re-runs into the same report and curve sheets
    out = an.run_fit(an.xy_data(book, ws, 1), "Line", "linear")
    recalc.stamp(p, out.report)
    names = [s.name for s in book.sheets]
    ws.set_cell(3, 1, 99.0)
    assert (book, out.report) in recalc.stale_reports(p, "Book1")
    recalc.recalculate(p, book, out.report)
    assert [s.name for s in book.sheets] == names
    slope = np.polyfit(ws.columns[0].numeric()[:20], ws.columns[1].numeric()[:20], 1)[0]
    assert out.result.params["Slope"] != pytest.approx(slope)
    assert float(out.report.table("Parameters")["rows"][1][2]) == pytest.approx(slope, rel=1e-5)


# ====================================================================== Qt: worksheet & reports
def test_sparklines_row_and_row_range_selection(qapp):
    from rsstatchem.ui.origin.sheet_view import (HEADER_ROWS, SPARK_ROW, BookWindow,
                                                 sparkline_points)

    book = _book()
    bw = BookWindow(book)
    view = bw.sheet_view()
    assert view.isRowHidden(SPARK_ROW)
    assert view.model_.headerData(SPARK_ROW, __import__("PySide6.QtCore").QtCore.Qt
                                  .Orientation.Vertical) == "Sparklines"
    bw.set_sparklines(True)
    assert book.sheets[0].sparklines and not bw.sheet_view().isRowHidden(SPARK_ROW)
    x, y = sparkline_points(book.sheets[0], 1)
    assert len(x) == 20 and x[0] == 1.0
    assert not view.model_.setData(view.model_.index(SPARK_ROW, 1), "x")
    bw.undo_stack.undo()
    assert not book.sheets[0].sparklines
    bw.select_block(3, 1, 10, 2)
    assert bw.selected_row_range() == (3, 10) and bw.selected_columns() == [1, 2]
    bw.select_columns([1])
    assert bw.selected_row_range() is None
    bw.select_block(1, 1, 20, 1)   # every data row = the whole column
    assert bw.selected_row_range() is None
    bw.set_sparklines(True)
    bw.select_columns([0])
    assert bw.copy(with_headers=True).split("\n")[:5] == ["", "", "", "", "1"]
    assert view.model_.rowCount() == HEADER_ROWS + book.sheets[0].n_rows


def test_collapsible_report_view_copy_and_lock(qapp):
    from PySide6.QtWidgets import QApplication

    from rsstatchem.ui.origin.report_view import ReportView, copy_items

    book = _book()
    out = an.run_fit(an.xy_data(book, book.sheets[0], 1), "Line", "linear")
    view = ReportView(out.report)
    titles = [s.header.label.text() for s in view.sections]
    assert titles[:4] == ["Notes", "Input Data", "Parameters", "Statistics"]
    assert "Residual Plots" in titles   # a parent section holding the residual plots
    assert view.lock_icon() == "lock_manual"
    sec = view.sections[2]
    sec.set_expanded(False)
    assert out.report.items[2]["collapsed"] is True and not sec.wrap.isVisibleTo(view)
    group = next(s for s in view.sections if s.header.label.text() == "Residual Plots")
    group.set_expanded(False)
    first = next(i for i in out.report.items if i.get("group") == "Residual Plots")
    assert first["group_collapsed"] is True
    view.rebuild()
    assert not view.sections[2].expanded
    copy_items([out.report.table("Parameters")])
    mime = QApplication.clipboard().mimeData()
    assert "<table" in mime.html() and "Slope" in mime.text() and "\t" in mime.text()
    out.report.op["dirty"] = True
    view.refresh()
    assert view.lock_icon() == "lock_dirty"
    out.report.op["mode"] = "auto"
    out.report.op["dirty"] = False
    view.refresh()
    assert view.lock_icon() == "lock_auto"
    assert "Parameters" in view.toPlainText()
    plain = ReportView(ReportSheet("R", "plain", []))
    assert not plain.lock.isVisibleTo(plain)


# ====================================================================== Qt: main window
def test_main_window_range_plot_fit_and_recalculation(ctx, monkeypatch):
    _no_prompts(monkeypatch)
    from rsstatchem.ui.origin import open_workspace
    from rsstatchem.ui.origin.sheet_view import HEADER_ROWS

    win = open_workspace(ctx)
    bw = win.active_book_window()
    rows = [["Time", "Signal"]] + [[f"{v}", f"{2 * v + 1}"] for v in range(1, 16)]
    bw.paste_block(rows, HEADER_ROWS, 0)
    bw.select_block(3, 1, 10, 1)
    assert win.range_label.text() == "[Book1]Sheet1!Col(B)[3:10]"
    gwin = win.plot("scatter")
    assert list(gwin.graph.data) == ["[Book1]Sheet1!A[3:10]", "[Book1]Sheet1!B[3:10]"]
    assert "[Book1]Sheet1!B[3:10]" in gwin.tree.topLevelItem(0).child(0).child(2).child(0) \
        .text(0)
    choices, _, _ = win.analysis_inputs()
    assert choices[0].rows == (3, 10) and "[3:10]" in choices[0].label
    out = win.run_fit("linear", {"input": choices[0], "model": "Line", "to_graph": True,
                                 "mode": "manual"}, gwin)
    assert out.result.n == 8 and out.report.op["hash"]
    # Manual: editing an input cell flags the report; Recalculate updates it
    win.show_window("Book1")
    bw.tabs.setCurrentIndex(0)
    model = bw.sheet_view().model_
    model.setData(model.index(HEADER_ROWS + 4, 1), "40")
    assert out.report.dirty
    assert win.recalculate_report(bw.book, out.report) and not out.report.dirty
    assert out.report.table("Statistics") is not None
    # Auto: the next edit re-runs the fit at once
    win.set_report_mode(bw.book, out.report, "auto")
    bw.tabs.setCurrentIndex(0)
    model = bw.sheet_view().model_
    model.setData(model.index(HEADER_ROWS + 4, 1), "11")
    assert not out.report.dirty
    slope = float(out.report.table("Parameters")["rows"][1][2])
    assert slope == pytest.approx(2.0)
    # the report's fitted-curve plot opens as a graph window
    item = next(i for i in out.report.items if i.get("graph"))
    n = len(win.project.graphs())
    win.open_report_graph(bw.book, out.report, item)
    assert len(win.project.graphs()) == n + 1
    # undo of the recalculation restores the earlier report content
    win.modified = False
    win.close()


def test_main_window_matrix_book(ctx, monkeypatch, tmp_path):
    _no_prompts(monkeypatch)
    from rsstatchem.ui.origin import open_workspace
    from rsstatchem.ui.origin.explorer import window_size

    win = open_workspace(ctx)
    mw = win.new_matrix()
    assert win.active_matrix_window() is mw
    titles = [a.text().replace("&", "") for a in win.menuBar().actions()]
    assert "Matrix" in titles and "Worksheet" not in titles
    win.set_matrix_values(mw, "x + 100*y")
    ms = mw.sheet()
    assert ms.data[0, 0] == 101.0 and ms.data[31, 31] == 32 + 3200
    win.set_matrix_dimensions(mw, {"rows": 10, "cols": 20, "x1": 0.0, "x2": 1.9, "y1": 0.0,
                                   "y2": 0.9, "x_name": "Length", "x_units": "cm",
                                   "y_name": "Width", "y_units": "cm", "long_name": "Height",
                                   "units": "mm"})
    assert ms.data.shape == (10, 20) and ms.x_title == "Length (cm)"
    win.matrix_geometry("transpose")
    assert ms.data.shape == (20, 10) and ms.x_name == "Width"
    mw.undo_stack.undo()
    ms = mw.sheet()
    assert ms.data.shape == (10, 20)
    win.matrix_resample(mw, "expand", 2, 2)
    assert mw.sheet().data.shape == (19, 39)
    win.matrix_resample(mw, "shrink", 3, 3)
    assert mw.sheet().data.shape == (7, 13)
    win.act_image_mode.setChecked(True)
    win.toggle_image_mode()
    assert mw.view().currentIndex() == 1 and not mw.view().image.image().isNull()
    gwin = win.plot_matrix("contour")
    assert gwin is not None and gwin.graph.layers[0].plots[0].type == "contour"
    win.show_window(mw.book.name)
    g3 = win.plot("surface3d")   # the 2D Graphs toolbar plots the active matrix
    assert g3.graph.layers[0].kind == "3d"
    # editing the matrix refreshes the linked graphs
    win.show_window(mw.book.name)
    mw.edit_cell(0, 0, "12345")
    assert np.nanmax(gwin.graph.values(gwin.graph.layers[0].plots[0].z)) == 12345.0
    assert window_size(mw.book).startswith("7×13")
    win.explorer.fill_windows()
    items = [win.explorer.windows.topLevelItem(i) for i in
             range(win.explorer.windows.topLevelItemCount())]
    assert any(i.text(2) == "Matrix" for i in items)
    # Convert to Worksheet and back
    bw2 = win.matrix_to_worksheet()
    ws = bw2.worksheet()
    assert [c.designation for c in ws.columns] == ["X", "Y", "Z"]
    assert ws.data_rows() == 7 * 13
    bw2.select_columns([2])
    mw2 = win.convert_to_matrix_xyz({"method": "auto", "cols": 5, "rows": 5})
    assert mw2.sheet().data.shape == (7, 13)
    assert np.allclose(mw2.sheet().data, mw.sheet().data, equal_nan=True)
    win.show_window(bw2.book.name)
    bw2.select_columns([0, 1])
    mw3 = win.convert_to_matrix_direct({"first_col_y": False, "long_names_x": False})
    assert mw3.sheet().data.shape == (91, 2)
    # saved and reopened with its own window type
    path = tmp_path / "m.rsop"
    assert win.save_to(str(path))
    win2 = open_workspace(ctx, str(path))
    assert isinstance(win2.project.windows[mw.book.name], MatrixBook)
    assert win2.view_of(mw.book.name).sheet().data.shape == (7, 13)
    assert win2.subs[mw.book.name].windowIcon() is not None
    win2.modified = False
    win2.close()
    win.modified = False
    win.close()


def test_statistics_dialogs_collect_settings_and_run(ctx, monkeypatch):
    _no_prompts(monkeypatch)
    from rsstatchem.ui.origin import open_workspace
    from rsstatchem.ui.origin import stat_dialogs as sd
    from rsstatchem.ui.origin.sheet_view import HEADER_ROWS

    win = open_workspace(ctx)
    bw = win.active_book_window()
    rng = np.random.default_rng(2)
    rows = [["A", "B", "C"]] + [[f"{rng.normal(5, 1):.3f}", f"{rng.normal(6, 1):.3f}",
                                 f"{rng.normal(5.5, 1):.3f}"] for _ in range(12)]
    bw.paste_block(rows, HEADER_ROWS, 0)
    bw.select_columns([0, 1, 2])
    sel = win.selection_ranges()
    assert sel == ["[Book1]Sheet1!A", "[Book1]Sheet1!B", "[Book1]Sheet1!C"]
    for key in ("desc", "rows", "freq", "normality", "correlation", "ttest1", "ttest2",
                "ttest_pair", "anova1", "anova2", "nonparam"):
        dlg = sd.make_dialog(key, win, None, sel)
        values = dlg.values()
        assert "mode" in values
        dlg.deleteLater()
    dlg = sd.make_dialog("anova1", win, None, sel)
    dlg.widget("means").boxes["bonferroni"].setChecked(True)
    dlg.mode.setCurrentIndex(dlg.mode.findData("auto"))
    values = dlg.values()
    assert values["ranges"] == sel and values["means"] == ["tukey", "bonferroni"]
    rep = win.run_stat("anova1", values)
    assert rep is not None and rep.mode == "auto" and bw.book.sheet(rep.name) is rep
    assert rep.table("Bonferroni Test") is not None
    # indexed form hides the raw-data list
    dlg2 = sd.make_dialog("ttest2", win, {"form": "indexed"}, sel)
    assert not dlg2.widget("ranges").isVisibleTo(dlg2) and \
        dlg2.widget("data").isVisibleTo(dlg2)
    # Change Parameters reopens the dialog with the report settings
    monkeypatch.setattr(sd.StatDialog, "exec", lambda self: True)
    seen = {}
    real = sd.make_dialog

    def spy(key, main=None, settings=None, selection=None):
        seen["settings"] = dict(settings or {})
        d = real(key, main, settings, selection)
        d.widget("means").boxes["scheffe"].setChecked(True)
        return d
    monkeypatch.setattr(sd, "make_dialog", spy)
    win.change_report_parameters(bw.book, rep)
    assert seen["settings"]["means"] == ["tukey", "bonferroni"]
    assert rep.table("Scheffe Test") is not None
    # the old direct commands still work (toolbar / scripts)
    bw.tabs.setCurrentIndex(0)
    bw.select_columns([1])
    win.statistics_on_columns()
    assert bw.book.sheet("DescStatsOnCols1") is not None
    win.modified = False
    win.close()


def test_explorer_drag_and_drop(ctx, monkeypatch):
    _no_prompts(monkeypatch)
    from rsstatchem.ui.origin import open_workspace

    win = open_workspace(ctx)
    win.new_notes()
    a = win.new_folder("A")
    win.explorer.current = win.project.root
    win.explorer.rebuild()
    b = win.new_folder("B")
    win.explorer.current = win.project.root
    win.explorer.rebuild()
    mime = win.explorer.windows.mimeData([win.explorer.windows.topLevelItem(0)])
    assert bytes(mime.data("application/x-rsstatchem-origin-windows")).decode() == "Book1"
    win.explorer.drop_windows(["Book1", "Notes1"], a)
    assert a.windows == ["Book1", "Notes1"] and "Book1" not in win.project.root.windows
    win.explorer.drop_folder(b, a)
    assert b in a.folders and b not in win.project.root.folders
    win.explorer.drop_folder(a, b)   # a folder cannot go into its own subfolder
    assert b in a.folders
    win.modified = False
    win.close()


def test_format_toolbar_and_in_place_text(ctx, monkeypatch):
    _no_prompts(monkeypatch)
    from rsstatchem.core.graph.model import Annotation
    from rsstatchem.ui.origin import open_workspace
    from rsstatchem.ui.origin.format_bar import wrap_escape

    assert wrap_escape("cm-1", 2, 4, "+") == ("cm\\+(-1)", 5, 7)
    win = open_workspace(ctx)
    book = _book()
    win.new_book(book)
    bw = win.book_window(book)
    bw.select_columns([1])
    gwin = win.plot("line")
    bar = win.format_bar
    ann = Annotation(kind="text", text="Absorbance", x=0.3, y=0.8)
    gwin.graph.layers[0].annotations.append(ann)
    gwin.redraw(rebuild_tree=True)
    gwin.select(("annotation", 0, ann.id))
    assert bar.font_box.isEnabled() and bar.acts["sup"].isEnabled()
    bar.size_box.setCurrentText("16")
    bar._size_entered()
    assert ann.font.size == 16.0
    bar.acts["bold"].setChecked(True)
    bar.style_clicked("bold")
    assert ann.font.bold
    # editing in place: the toolbar wraps the selected characters in escape codes
    assert gwin.edit_text(("annotation", 0, ann.id))
    ed = gwin.text_editor
    ed.setText("H2O")
    ed.setSelection(1, 1)
    bar.escape("sub")
    assert ed.text() == "H\\-(2)O"
    gwin.commit_text()
    assert ann.text == "H\\-(2)O" and not ed.isVisible()
    # axis titles too; Esc cancels
    assert gwin.edit_text(("axis", 0, "x"))
    gwin.text_editor.setText("changed")
    gwin.cancel_text()
    assert gwin.graph.layers[0].x.title != "changed"
    gwin.select(("legend", 0))
    assert bar.font_box.isEnabled() and not bar.acts["sup"].isEnabled()
    gwin.select(("graph",))
    assert not bar.font_box.isEnabled()
    win.modified = False
    win.close()


def test_3d_rotation_is_kept(qapp):
    from rsstatchem.ui.graph.window import GraphWindow

    mb = MatrixBook.new("MBook1", 8, 8)
    ms = mb.sheets[0]
    ms.data = mx.evaluate("x*y", ms.data.shape, (ms.x1, ms.x2, ms.y1, ms.y2))
    g = plot_matrix(mb, ms, "surface3d_colormap", "G")
    win = GraphWindow(g)
    ax = win.rmap.axes[0]["ax"]
    ax.view_init(elev=50, azim=10)    # what a mouse drag does
    assert win._capture_3d_view()
    assert (g.layers[0].elev, g.layers[0].azim) == (50.0, 10.0)
    assert not win._capture_3d_view()
    win.undo()
    assert win.graph.layers[0].elev == 30.0
    win.rotate_3d(15, 5)
    assert win.graph.layers[0].azim == -40.0 and win.graph.layers[0].elev == 35.0
    # a real mouse drag over the 3D axes rotates it, and the release keeps the view
    from matplotlib.backend_bases import MouseEvent
    ax = win.rmap.axes[0]["ax"]
    cx, cy = ax.bbox.x0 + ax.bbox.width / 2, ax.bbox.y0 + ax.bbox.height / 2
    MouseEvent("button_press_event", win.canvas, cx, cy, button=1)._process()
    for k in range(1, 6):
        MouseEvent("motion_notify_event", win.canvas, cx + 12 * k, cy + 4 * k,
                   button=1)._process()
    MouseEvent("button_release_event", win.canvas, cx + 60, cy + 20, button=1)._process()
    layer = win.graph.layers[0]
    assert (layer.elev, layer.azim) != (35.0, -40.0)
    win.modified = False
    win.close()


def test_fit_change_parameters_and_matrix_dialogs(ctx, monkeypatch):
    _no_prompts(monkeypatch)
    from rsstatchem.ui.origin import open_workspace
    from rsstatchem.ui.origin.dialogs import (FitDialog, MatrixDimensionsDialog,
                                              MatrixValuesDialog)

    win = open_workspace(ctx)
    book = _book()
    win.new_book(book)   # "Book1" is taken: the new workbook becomes Book2 in the project
    assert book.name == "Book2" and win.project.windows["Book2"] is book
    bw = win.book_window(book)
    bw.select_columns([1])
    choices, _, _ = win.analysis_inputs()
    out = win.run_fit("poly", {"input": choices[0], "model": "Poly2", "mode": "manual"})
    assert out is not None, win.messages.toPlainText()
    assert out.report.op["settings"]["model"] == "Poly2"

    def fake_exec(self):
        assert self.order.value() == 2   # the dialog shows the report's settings
        self.order.setValue(3)
        return True
    monkeypatch.setattr(FitDialog, "exec", fake_exec)
    win.change_report_parameters(book, out.report)
    assert out.report.op["settings"]["model"] == "Poly3"
    assert any(r[1] == "B3" for r in out.report.table("Parameters")["rows"])
    mw = win.new_matrix()
    ms = mw.sheet()
    applied = []
    dlg = MatrixValuesDialog(ms, applied.append, win)
    dlg.edit.setPlainText("i+j")
    assert dlg.apply() and applied == ["i+j"]
    dims = MatrixDimensionsDialog(ms, win).values()
    assert (dims["rows"], dims["cols"], dims["x2"]) == (32, 32, 32.0)
    bad = MatrixValuesDialog(ms, lambda f: win.set_matrix_values(mw, f), win)
    bad.edit.setPlainText("sin(")
    assert not bad.apply() and "Syntax" in bad.error.text()
    win.modified = False
    win.close()


def test_matrix_sheet_helpers():
    ms = MatrixSheet.new("M", 3, 4)
    assert ms.xs().tolist() == [1, 2, 3, 4] and ms.ys().tolist() == [1, 2, 3]
    ms.data[0, 0] = 1.0
    ms.set_dimensions(2, 6)
    assert ms.data.shape == (2, 6) and ms.data[0, 0] == 1.0 and np.isnan(ms.data[1, 5])
    assert MatrixBook.new("MBook1").title == "MBook1"
