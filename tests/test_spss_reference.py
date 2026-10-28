"""WP7: RS-StatChem against published IBM SPSS output, one test per reference fixture.

Every expected value is the number SPSS printed in the cited source, compared at the printed
precision (``spss_ref.fetch.assert_printed``: our value rounded to the printed decimals must
equal the printed string).

Sources and licences
* UCLA OARC "SPSS Annotated Output", "Data Analysis Examples" and "What statistical analysis
  should I use?" pages. The UCLA data files carry no redistribution licence: they are
  downloaded at test time from the published URL into a cache (``tests/spss_ref/fetch.py``)
  and the tests are skipped offline. Nothing from UCLA is committed except the printed numbers.
* IBM SPSS Statistics 29 Case Studies and the IBM SPSS Exact Tests manual (v20): printed
  numbers only; small tables rebuilt from printed counts are embedded as numbers. IBM's
  sample files are not used.
* McCullagh & Nelder (1989) ship-damage table (as in MASS::ships), embedded as numbers.

The last block pins the IBM SPSS Statistics Algorithms formulas behind the WP7 changes that no
published output covers (Bayes conventions, omega-squared random, Type III/IV with empty cells,
Cox Kalbfleisch-Prentice baseline, R-E-G-W subsets, bootstrap order statistics).
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pytest

from rsstatchem.core.dataset import Dataset
from spss_ref.fetch import UCLA, UCLA_STAT, assert_printed, fetch, hsb2, read_sav  # noqa: F401


# ------------------------------------------------------------------ helpers
def _sav(name: str, base: str = UCLA):
    """The UCLA .sav file read by RS-StatChem's own SPSS reader (keeps print formats, labels),
    with lower-case variable names."""
    from rsstatchem.io.files import read_sav as rs_read_sav

    ds = rs_read_sav(fetch(base + name))
    for i, v in enumerate(list(ds.variables)):
        if v.name != v.name.lower():
            ds.rename_variable(i, v.name.lower())
    return ds


def _norm(s) -> str:
    s = "".join(ch for ch in str(s) if unicodedata.category(ch) != "Lm")
    return s.strip().rstrip("*").strip()


def _num(s):
    t = _norm(s).replace(",", "").replace("%", "")
    t = re.sub(r"[^0-9.eE+\-]+$", "", t)
    try:
        return float(t)
    except ValueError:
        return None


def _section(res):
    return getattr(res, "section", res)


def _tab(res, title: str, nth: int = 0):
    found = [t for t in _section(res).items
             if getattr(t, "kind", "") == "table" and _norm(t.title).startswith(title)]
    assert len(found) > nth, f"no table {title!r}; have " + repr(
        [getattr(t, "title", "") for t in _section(res).items])
    return found[nth]


def _col(t, spec) -> int:
    if isinstance(spec, int):
        return spec
    for i, c in enumerate(t.columns):
        if isinstance(spec, tuple):
            if isinstance(c, tuple) and tuple(_norm(x) for x in c) == spec:
                return i
        elif (_norm(c) if not isinstance(c, tuple) else None) == spec:
            return i
    for i, c in enumerate(t.columns):
        if isinstance(c, tuple) and not isinstance(spec, tuple) and spec in (_norm(x) for x in c):
            return i
    raise AssertionError(f"no column {spec!r} in {t.title!r}: {t.columns}")


def _row(t, labels):
    if isinstance(labels, str):
        labels = (labels,)
    ctx: list[str] = []
    for r in t.rows:
        first_num = next((i for i, c in enumerate(r) if _num(c) is not None and i > 0), len(r))
        if len(ctx) < len(r):
            ctx += [""] * (len(r) - len(ctx))
        for i in range(first_num):
            if _norm(r[i]):
                ctx[i] = _norm(r[i])
                for j in range(i + 1, first_num):
                    if not _norm(r[j]):
                        ctx[j] = ""
        heads = set(ctx[:first_num]) | {_norm(c) for c in r[:first_num]}
        if all(lb in heads for lb in labels):
            return r
    raise AssertionError(f"no row {labels!r} in {t.title!r}: {t.rows}")


def _cell(t, labels, col) -> str:
    return _row(t, labels)[_col(t, col)]


def chk(t, labels, col, printed: str):
    cell = _cell(t, labels, col)
    v = _num(cell)
    assert v is not None, f"{t.title} {labels} {col}: non-numeric cell {cell!r}"
    assert_printed(v, printed, f"{t.title} | {labels} | {col}")


def chk_row(t, labels, cols, printed):
    for c, p in zip(cols, printed):
        if p != "":
            chk(t, labels, c, p)


# ======================================================================
# UCLA "Descriptive statistics" annotated output (DESCRIPTIVES + EXAMINE of write)
# https://stats.oarc.ucla.edu/spss/output/descriptive-statistics/   data: hsb2.sav
def test_ucla_descriptives_examine():
    from rsstatchem.core.stats import descriptive as D
    from rsstatchem.core.stats import explore as E

    ds = _sav("hsb2.sav")
    sec = D.descriptives(ds, ["write"], ("mean", "std", "var", "min", "max", "sem", "kurt",
                                         "skew"))
    t = _tab(sec, "Descriptive Statistics")
    r = _row(t, "writing score")
    # N, Min, Max, Mean, SE mean, SD, Variance, Skew, SE, Kurt, SE (as printed by SPSS)
    want = ["200", "31.00", "67.00", "52.7750", ".67024", "9.47859", "89.844", "-.482", ".172",
            "-.750", ".342"]
    assert [_norm(c) for c in r[1:]] == want      # identical formatted strings

    sec = E.explore(ds, ["write"], percentiles=True, stem_leaf=True, normality=False,
                    boxplot=False, qq=False)
    t = _tab(sec, "Descriptives")
    for label, stat, se in [("Mean", "52.7750", ".67024"),
                            ("95% Confidence Interval for Mean: Lower Bound", "51.4533", ""),
                            ("95% Confidence Interval for Mean: Upper Bound", "54.0967", ""),
                            ("5% Trimmed Mean", "53.1389", ""), ("Median", "54.0000", ""),
                            ("Variance", "89.844", ""), ("Std. Deviation", "9.47859", ""),
                            ("Minimum", "31.00", ""), ("Maximum", "67.00", ""),
                            ("Range", "36.00", ""), ("Interquartile Range", "14.75", ""),
                            ("Skewness", "-.482", ".172"), ("Kurtosis", "-.750", ".342")]:
        r = _row(t, label)
        assert _norm(r[2]) == stat, (label, r)
        if se:
            assert _norm(r[3]) == se, (label, r)
    t = _tab(sec, "Percentiles")
    wa = _row(t, "Weighted Average(Definition 1)")
    assert [_norm(c) for c in wa[2:]] == ["35.0500", "39.0000", "45.2500", "54.0000", "60.0000",
                                          "65.0000", "65.0000"]
    th = _row(t, "Tukey's Hinges")
    assert [_norm(c) for c in th[4:7]] == ["45.5000", "54.0000", "60.0000"]
    # P99 (requested on the UCLA page with /PERCENTILES(5,10,25,50,75,90,95,99))
    x = ds.analysis_series("write").dropna().to_numpy(float)
    assert_printed(E.weighted_average_percentiles(x, [99])[0], "67.0000", "P99")
    # stem-and-leaf: SPSS printed 19 lines, 5 lines per stem (line width 2), stem width 10
    t = _tab(sec, "writing score Stem-and-Leaf Plot")
    spss_lines = [("4.00", "3", "1111"), ("4.00", "3", "3333"), ("2.00", "3", "55"),
                  ("5.00", "3", "66777"), ("6.00", "3", "899999"),
                  ("13.00", "4", "0001111111111"), ("3.00", "4", "223"),
                  ("13.00", "4", "4444444444445"), ("11.00", "4", "66666666677"),
                  ("11.00", "4", "99999999999"), ("2.00", "5", "00"),
                  ("16.00", "5", "2222222222222223"), ("20.00", "5", "44444444444444444555"),
                  ("12.00", "5", "777777777777"), ("25.00", "5", "9999999999999999999999999"),
                  ("8.00", "6", "00001111"), ("22.00", "6", "2222222222222222223333"),
                  ("16.00", "6", "5555555555555555"), ("7.00", "6", "7777777")]
    assert [tuple(r) for r in t.rows] == spss_lines
    assert any(f.startswith("Stem width: 10") for f in t.footnotes)


# ======================================================================
# UCLA annotated t-test output (one-sample, paired, independent)
# https://stats.oarc.ucla.edu/spss/output/t-test/   data: hsb2.sav
def test_ucla_ttest():
    from rsstatchem.core.stats import ttest as T

    ds = _sav("hsb2.sav")
    sec = T.one_sample(ds, ["write"], 50)
    chk_row(_tab(sec, "One-Sample Statistics"), "writing score",
            ["N", "Mean", "Std. Deviation", "Std. Error Mean"],
            ["200", "52.7750", "9.47859", ".67024"])
    t = _tab(sec, "One-Sample Test")
    chk_row(t, "writing score", [1, 2, 4, 5, 6, 7],
            ["4.140", "199", ".000", "2.77500", "1.4533", "4.0967"])

    sec = T.paired(ds, [("write", "read")])
    t = _tab(sec, "Paired Samples Statistics")
    chk_row(t, "writing score", ["Mean", "N", "Std. Deviation", "Std. Error Mean"],
            ["52.7750", "200", "9.47859", ".67024"])
    chk_row(t, "reading score", ["Mean", "N", "Std. Deviation", "Std. Error Mean"],
            ["52.2300", "200", "10.25294", ".72499"])
    t = _tab(sec, "Paired Samples Correlations")
    chk_row(t, "Pair 1", ["N", "Correlation", ("Significance", "Two-Sided p")],
            ["200", ".597", ".000"])
    t = _tab(sec, "Paired Samples Test")
    chk_row(t, "Pair 1", [2, 3, 4, 5, 6, "t", "df", ("Significance", "Two-Sided p")],
            [".54500", "8.88667", ".62838", "-.69414", "1.78414", ".867", "199", ".387"])

    sec = T.independent(ds, ["write"], "female", groups=(0, 1))
    t = _tab(sec, "Group Statistics")
    chk_row(t, "male", ["N", "Mean", "Std. Deviation", "Std. Error Mean"],
            ["91", "50.1209", "10.30516", "1.08027"])
    chk_row(t, "female", ["N", "Mean", "Std. Deviation", "Std. Error Mean"],
            ["109", "54.9908", "8.13372", ".77907"])
    t = _tab(sec, "Independent Samples Test")
    cols = [2, 3, 4, 5, 7, 8, 9, 10, 11]
    chk_row(t, "Equal variances assumed", cols,
            ["11.133", ".001", "-3.734", "198", ".000", "-4.86995", "1.30419", "-7.44183",
             "-2.29806"])
    # Welch df: this (older) SPSS page displays 169.70, a truncation of 169.7074; the newer
    # output on the UCLA "what statistical analysis" page prints 169.707 (tested there).
    chk_row(t, "Equal variances not assumed", [4, 7, 8, 9, 10, 11],
            ["-3.656", ".000", "-4.86995", "1.33189", "-7.49916", "-2.24073"])
    welch_df = float(_cell(t, "Equal variances not assumed", 5))
    assert f"{np.floor(welch_df * 100) / 100:.2f}" == "169.70"


# ======================================================================
# UCLA annotated correlation output (pairwise and listwise)
# https://stats.oarc.ucla.edu/spss/output/correlation/   data: hsb2.sav
def test_ucla_correlation():
    from rsstatchem.core.stats import correlate as C

    ds = _sav("hsb2.sav")
    names = ["read", "write", "math", "science", "female"]
    labels = ["reading score", "writing score", "math score", "science score", "female"]
    r_print = [["1", ".597", ".662", ".630", "-.053"],
               [".597", "1", ".617", ".570", ".256"],
               [".662", ".617", "1", ".631", "-.029"],
               [".630", ".570", ".631", "1", "-.128"],
               ["-.053", ".256", "-.029", "-.128", "1"]]
    p_print = [[".", ".000", ".000", ".000", ".455"],
               [".000", ".", ".000", ".000", ".000"],
               [".000", ".000", ".", ".000", ".680"],
               [".000", ".000", ".000", ".", ".071"],
               [".455", ".000", ".680", ".071", "."]]
    flagged = {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (1, 4), (2, 3)}
    for listwise in (False, True):
        t = _tab(C.bivariate(ds, names, listwise_deletion=listwise), "Correlations")
        for i, lab in enumerate(labels):
            for j, lab2 in enumerate(labels):
                c = _cell(t, (lab, "Pearson Correlation"), lab2)
                assert_printed(_num(c), r_print[i][j], f"r {lab}/{lab2}")
                assert c.endswith("**") == ((min(i, j), max(i, j)) in flagged), (lab, lab2, c)
                if i != j:
                    chk(t, (lab, "Sig. (2-tailed)"), lab2, p_print[i][j])
                if not listwise:
                    chk(t, (lab, "N"), lab2, "200")


# ======================================================================
# UCLA annotated regression output
# https://stats.oarc.ucla.edu/spss/output/regression-analysis/   data: hsb2.sav
def test_ucla_regression():
    from rsstatchem.core.stats import regression as R

    ds = _sav("hsb2.sav")
    res = R.linear(ds, "science", ["math", "female", "socst", "read"],
                   statistics=("estimates", "model_fit", "ci"), plots=(), assumptions=())
    chk_row(_tab(res, "Model Summary"), "1",
            ["R", "R Square", "Adjusted R Square", "Std. Error of the Estimate"],
            [".699", ".489", ".479", "7.14817"])
    t = _tab(res, "ANOVA")
    cols = ["Sum of Squares", "df", "Mean Square", "F", "Sig."]
    chk_row(t, "Regression", cols, ["9543.721", "4", "2385.930", "46.695", ".000"])
    chk_row(t, "Residual", cols[:3], ["9963.779", "195", "51.096"])
    chk_row(t, "Total", cols[:2], ["19507.500", "199"])
    t = _tab(res, "Coefficients")
    cols = [2, 3, 4, 5, 6, 7, 8]
    for lab, vals in [("(Constant)", ["12.325", "3.194", "", "3.859", ".000", "6.027", "18.624"]),
                      ("math score", [".389", ".074", ".368", "5.252", ".000", ".243", ".535"]),
                      ("female", ["-2.010", "1.023", "-.101", "-1.965", ".051", "-4.027", ".007"]),
                      ("social studies score",
                       [".050", ".062", ".054", ".801", ".424", "-.073", ".173"]),
                      ("reading score", [".335", ".073", ".347", "4.607", ".000", ".192", ".479"])]:
        chk_row(t, lab, cols, vals)


# ======================================================================
# UCLA annotated logistic regression output (+ CROSSTABS risk estimate)
# https://stats.oarc.ucla.edu/spss/output/logistic-regression/   data: hsb2.sav
def test_ucla_logistic_regression():
    from rsstatchem.core.stats import crosstabs as X
    from rsstatchem.core.stats import logistic as L

    ds = _sav("hsb2.sav")
    ds.add_variable("honcomp", values=(ds.df["write"] >= 60).astype(float))
    res = L.binary_logistic(ds, "honcomp", ["read", "science", "ses"],
                            categorical={"ses": ("indicator", "last")})
    t = _tab(res, "Categorical Variables Codings")
    # ses 1/2/3 (value labels low/middle/high); indicator coding, last category reference
    assert [[_norm(c) for c in r[2:]] for r in t.rows] == [
        ["47", "1.000", ".000"], ["95", ".000", "1.000"], ["58", ".000", ".000"]]
    t0 = _tab(res, "Classification Table")
    assert [_norm(c) for c in t0.rows[0][3:]] == ["147", "0", "100.0"]   # honcomp .00
    assert [_norm(c) for c in t0.rows[1][3:]] == ["53", "0", ".0"]       # honcomp 1.00
    chk(t0, "Overall Percentage", 5, "73.5")
    chk_row(_tab(res, "Variables in the Equation"), "Constant", [2, 3, 4, 5, 6, 7],
            ["-1.020", ".160", "40.540", "1", ".000", ".361"])
    t = _tab(res, "Variables not in the Equation")
    for lab, vals in [("read", ["47.906", "1", ".000"]), ("science", ["34.862", "1", ".000"]),
                      ("ses", ["14.783", "2", ".001"]), ("ses(1)", [".302", "1", ".582"]),
                      ("ses(2)", ["8.666", "1", ".003"]),
                      ("Overall Statistics", ["58.644", "4", ".000"])]:
        chk_row(t, lab, ["Score", "df", "Sig."], vals)
    t = _tab(res, "Omnibus Tests of Model Coefficients")
    for lab in ("Step", "Block", "Model"):
        chk_row(t, lab, ["Chi-square", "df", "Sig."], ["65.588", "4", ".000"])
    chk_row(_tab(res, "Model Summary"), "1",
            ["-2 Log likelihood", "Cox & Snell R Square", "Nagelkerke R Square"],
            ["165.701", ".280", ".408"])
    t1 = _tab(res, "Classification Table", 1)
    assert [_norm(c) for c in t1.rows[0][3:]] == ["132", "15", "89.8"]
    assert [_norm(c) for c in t1.rows[1][3:]] == ["26", "27", "50.9"]
    chk(t1, "Overall Percentage", 5, "79.5")
    t = _tab(res, "Variables in the Equation", 1)
    cols = ["B", "S.E.", "Wald", "df", "Sig.", "Exp(B)"]
    for lab, vals in [("read", [".098", ".025", "15.199", "1", ".000", "1.103"]),
                      ("science", [".066", ".027", "5.867", "1", ".015", "1.068"]),
                      ("ses", ["", "", "6.690", "2", ".035", ""]),
                      ("ses(1)", [".058", ".532", ".012", "1", ".913", "1.060"]),
                      ("ses(2)", ["-1.013", ".444", "5.212", "1", ".022", ".363"]),
                      ("Constant", ["-9.561", "1.662", "33.112", "1", ".000", ".000"])]:
        chk_row(t, lab, cols, vals)

    sec = X.crosstabs(ds, ["female"], ["honcomp"], statistics=("risk",), chisq=False, phi=False)
    t = _tab(sec, "female * honcomp Crosstabulation")
    assert [_norm(c) for c in t.rows[0][3:]] == ["73", "18", "91"]
    assert [_norm(c) for c in t.rows[1][3:]] == ["74", "35", "109"]
    t = _tab(sec, "Risk Estimate")
    # SPSS rows: "Odds Ratio for female (.00 / 1.00)", "For cohort honcomp = .00 / 1.00"
    # (our labels use the value labels male/female, and 0.00 with a leading zero)
    assert t.rows[0][0].startswith("Odds Ratio for female")
    for i, vals in enumerate([["1.918", ".997", "3.689"], ["1.182", "1.002", "1.393"],
                              [".616", ".375", "1.011"]]):
        for j, v in enumerate(vals):
            assert_printed(_num(t.rows[i][j + 1]), v, f"risk {i} {j}")
    chk(t, "N of Valid Cases", 1, "200")

    res = L.binary_logistic(ds, "honcomp", ["female"], ci=True)
    t = _tab(res, "Omnibus Tests of Model Coefficients")
    for lab in ("Step", "Block", "Model"):
        chk_row(t, lab, ["Chi-square", "df", "Sig."], ["3.935", "1", ".047"])
    t = _tab(res, "Variables in the Equation", 1)
    chk_row(t, "female", [2, 3, 4, 5, 6, 7, 8, 9],
            [".651", ".334", "3.811", "1", ".051", "1.918", ".997", "3.689"])
    chk_row(t, "Constant", [2, 3, 4, 5, 6, 7], ["-1.400", ".263", "28.305", "1", ".000", ".247"])


# ======================================================================
# UCLA "What statistical analysis should I use?" (SPSS): one-sample tests
# https://stats.oarc.ucla.edu/spss/whatstat/what-statistical-analysis-should-i-usestatistical-analyses-using-spss/
# data: hsb2-3.sav (identical to hsb2.sav)
def test_ucla_whatstat_onesample_tests():
    from rsstatchem.core.stats import nonparametric as N
    from rsstatchem.core.stats import nptests as NP
    from rsstatchem.core.stats import ttest as T

    ds = _sav("hsb2.sav")
    sec = T.one_sample(ds, ["write"], 50)
    chk_row(_tab(sec, "One-Sample Test"), "writing score", [1, 2, 4, 5, 6, 7],
            ["4.140", "199", ".000", "2.7750", "1.4533", "4.0967"])

    sec = NP.one_sample(ds, ["write"], tests=("wilcoxon",), wilcoxon_median=50, charts_on=False)
    t = _tab(sec, "Hypothesis Test Summary")
    r = _row(t, "1")
    assert "median of writing score equals 50.00" in r[1]
    assert _norm(r[3]) == ".000" and r[4].startswith("Reject")
    t = _tab(sec, "One-Sample Wilcoxon Signed Rank Test Summary")
    for lab, v in [("Total N", "200"), ("Test Statistic", "13,177.000"),
                   ("Standard Error", "806.235"), ("Standardized Test Statistic", "4.126")]:
        chk(t, lab, 1, v)
    assert _norm(_row(t, "Asymptotic Sig.(2-sided test)")[1]) == ".000"

    # Binomial: SPSS (this older output) labels the p ".229 Based on Z Approximation"; the
    # current Algorithms (NPAR TESTS, Binomial Test) document the exact two-sided p, which we
    # print (exact p = .229 as well). Observed proportion of group 2 is printed .54 by SPSS
    # (computed as 1 - .455); 109/200 = .545 rounds either way.
    t = _tab(N.binomial(ds, ["female"], 0.5), "Binomial Test")
    chk_row(t, "Group 1", ["N", "Observed Prop.", "Test Prop."], ["91", ".46", ".50"])
    chk(t, "Group 1", 6, ".229")
    chk_row(t, "Group 2", ["N", "Observed Prop."], ["109", ".54"])
    chk_row(t, "Total", ["N", "Observed Prop."], ["200", "1.00"])

    sec = N.chi_square(ds, ["race"], expected=[10, 10, 10, 70])
    t = _tab(sec, "race")
    for lab, vals in [("hispanic", ["24", "20.0", "4.0"]), ("asian", ["11", "20.0", "-9.0"]),
                      ("african-amer", ["20", "20.0", ".0"]), ("white", ["145", "140.0", "5.0"])]:
        assert [_norm(c) for c in _row(t, lab)[1:]] == vals
    t = _tab(sec, "Test Statistics")
    chk(t, "Chi-Square", 1, "5.029")
    chk(t, "df", 1, "3")
    chk(t, "Asymp. Sig.", 1, ".170")


# ======================================================================
# UCLA whatstat: independent t test and Mann-Whitney   data: hsb2-3.sav (= hsb2.sav)
def test_ucla_whatstat_independent_ttest_mannwhitney():
    from rsstatchem.core.stats import nonparametric as N
    from rsstatchem.core.stats import ttest as T

    ds = _sav("hsb2.sav")
    t = _tab(T.independent(ds, ["write"], "female", groups=(0, 1)), "Independent Samples Test")
    chk_row(t, "Equal variances assumed", [2, 3, 4, 5, 7, 8, 9, 10, 11],
            ["11.133", ".001", "-3.734", "198", ".000", "-4.8699", "1.30419", "-7.44183",
             "-2.29806"])
    chk_row(t, "Equal variances not assumed", [4, 5, 7, 8, 9, 10, 11],
            ["-3.656", "169.707", ".000", "-4.8699", "1.33189", "-7.49916", "-2.24073"])

    sec = N.mann_whitney(ds, ["write"], "female", (0, 1))
    t = _tab(sec, "Test Statistics")
    for lab, v in [("Mann-Whitney U", "3606.000"), ("Wilcoxon W", "7792.000"), ("Z", "-3.329"),
                   ("Asymp. Sig. (2-tailed)", ".001")]:
        chk(t, lab, 1, v)


# ======================================================================
# UCLA whatstat: paired t, Wilcoxon signed ranks, sign test, Friedman   data: hsb2.sav
def test_ucla_whatstat_paired_wilcoxon_sign_friedman():
    from rsstatchem.core.stats import nonparametric as N
    from rsstatchem.core.stats import ttest as T

    ds = _sav("hsb2.sav")
    sec = T.paired(ds, [("read", "write")])
    t = _tab(sec, "Paired Samples Test")
    chk_row(t, "Pair 1", [2, 3, 4, 5, 6, "t", "df", ("Significance", "Two-Sided p")],
            ["-.5450", "8.88667", ".62838", "-1.7841", ".6941", "-.867", "199", ".387"])

    sec = N.two_related(ds, [("write", "read")], wilcoxon=True)
    t = _tab(sec, "Wilcoxon Signed Ranks Test")
    assert _norm(t.rows[0][0]) == "reading score - writing score"     # SPSS labels it so too
    chk_row(t, "Negative Ranks", [2, 3, 4], ["97", "95.47", "9261.00"])
    chk_row(t, "Positive Ranks", [2, 3, 4], ["88", "90.27", "7944.00"])
    chk(t, "Ties", 2, "15")
    chk(t, "Total", 2, "200")
    t = _tab(sec, "Test Statistics")
    chk(t, "Z", 1, "-.903")
    chk(t, "Asymp. Sig. (2-tailed)", 1, ".366")

    sec = N.two_related(ds, [("read", "write")], wilcoxon=False, sign=True)
    t = _tab(sec, "Sign Test")
    chk(t, "Negative Differences", 2, "88")
    chk(t, "Positive Differences", 2, "97")
    chk(t, "Ties", 2, "15")
    t = _tab(sec, "Test Statistics")
    chk(t, "Z", 1, "-.588")
    chk(t, "Asymp. Sig. (2-tailed)", 1, ".556")

    sec = N.k_related(ds, ["read", "write", "math"])
    t = _tab(sec, "Ranks")
    for lab, v in [("reading score", "1.96"), ("writing score", "2.04"), ("math score", "2.01")]:
        chk(t, lab, 1, v)
    t = _tab(sec, "Test Statistics")
    for lab, v in [("N", "200"), ("Chi-Square", ".645"), ("df", "2"), ("Asymp. Sig.", ".724")]:
        chk(t, lab, 1, v)


# ======================================================================
# UCLA whatstat: crosstabs chi-square, Fisher, McNemar   data: hsb2.sav
def test_ucla_whatstat_crosstabs_chisq_mcnemar():
    from rsstatchem.core.stats import crosstabs as X

    ds = _sav("hsb2.sav")
    sec = X.crosstabs(ds, ["schtyp"], ["female"], statistics=("chisq",), phi=False)
    t = _tab(sec, "type of school * female Crosstabulation")
    assert [_norm(c) for c in _row(t, "public")[3:]] == ["77", "91", "168"]
    assert [_norm(c) for c in _row(t, "private")[3:]] == ["14", "18", "32"]
    t = _tab(sec, "Chi-Square Tests")
    chk_row(t, "Pearson Chi-Square", [1, 2, 3], [".047", "1", ".828"])
    chk_row(t, "Continuity Correction", [1, 2, 3], [".001", "1", ".981"])
    chk_row(t, "Likelihood Ratio", [1, 2, 3], [".047", "1", ".828"])
    chk_row(t, "Fisher's Exact Test", [4, 5], [".849", ".492"])
    chk_row(t, "Linear-by-Linear Association", [1, 2, 3], [".047", "1", ".829"])
    chk(t, "N of Valid Cases", 1, "200")
    assert any("minimum expected count is 14.56" in f for f in t.footnotes)

    sec = X.crosstabs(ds, ["female"], ["ses"], statistics=("chisq",), phi=False)
    t = _tab(sec, "female * ses Crosstabulation")
    assert [_norm(c) for c in _row(t, "male")[3:]] == ["15", "47", "29", "91"]
    assert [_norm(c) for c in t.rows[1][3:]] == ["32", "48", "29", "109"]
    t = _tab(sec, "Chi-Square Tests")
    chk_row(t, "Pearson Chi-Square", [1, 2, 3], ["4.577", "2", ".101"])
    chk_row(t, "Likelihood Ratio", [1, 2, 3], ["4.679", "2", ".096"])
    chk_row(t, "Linear-by-Linear Association", [1, 2, 3], ["3.110", "1", ".078"])
    assert any("minimum expected count is 21.39" in f for f in t.footnotes)

    ds.add_variable("himath", values=(ds.df["math"] > 60).astype(float))
    ds.add_variable("hiread", values=(ds.df["read"] > 60).astype(float))
    sec = X.crosstabs(ds, ["himath"], ["hiread"], statistics=("mcnemar",), chisq=False,
                      phi=False)
    t = _tab(sec, "himath * hiread Crosstabulation")
    assert [_norm(c) for c in t.rows[0][3:]] == ["135", "21", "156"]
    assert [_norm(c) for c in t.rows[1][3:]] == ["18", "26", "44"]
    t = _tab(sec, "Chi-Square Tests")
    chk(t, "McNemar Test", 4, ".749")
    assert any("Binomial distribution used" in f for f in t.footnotes)


# ======================================================================
# UCLA whatstat: one-way ANOVA, MEANS report, Kruskal-Wallis   data: hsb2.sav
def test_ucla_whatstat_oneway_anova_kruskal():
    from rsstatchem.core.stats import anova as A
    from rsstatchem.core.stats import means as M
    from rsstatchem.core.stats import nonparametric as N

    ds = _sav("hsb2.sav")
    t = _tab(A.oneway(ds, ["write"], "prog", descriptives=False, homogeneity=False,
                      effect_size=False), "ANOVA")
    cols = ["Sum of Squares", "df", "Mean Square", "F", "Sig."]
    chk_row(t, "Between Groups", cols, ["3175.698", "2", "1587.849", "21.275", ".000"])
    chk_row(t, "Within Groups", cols[:3], ["14703.177", "197", "74.635"])
    chk_row(t, "Total", cols[:2], ["17878.875", "199"])

    t = _tab(M.means(ds, ["write"], [["prog"]]), "Report")
    for lab, vals in [("general", ["51.3333", "45", "9.39778"]),
                      ("academic", ["56.2571", "105", "7.94334"]),
                      ("vocation", ["46.7600", "50", "9.31875"]),
                      ("Total", ["52.7750", "200", "9.47859"])]:
        assert [_norm(c) for c in _row(t, lab)[1:]] == vals

    sec = N.kruskal_wallis(ds, ["write"], "prog", 1, 3)
    t = _tab(sec, "Ranks")
    for lab, vals in [("general", ["45", "90.64"]), ("academic", ["105", "121.56"]),
                      ("vocation", ["50", "65.14"])]:
        chk_row(t, lab, ["N", "Mean Rank"], vals)
    t = _tab(sec, "Test Statistics")
    # older SPSS labels the statistic "Chi-Square", current SPSS "Kruskal-Wallis H"
    chk(t, "Kruskal-Wallis H", 1, "34.045")
    chk(t, "df", 1, "2")
    chk(t, "Asymp. Sig.", 1, ".000")


# ======================================================================
# UCLA whatstat: Spearman correlation   data: hsb2.sav
def test_ucla_whatstat_spearman():
    from rsstatchem.core.stats import correlate as C

    ds = _sav("hsb2.sav")
    t = _tab(C.bivariate(ds, ["read", "write"], pearson=False, spearman=True), "Correlations")
    chk(t, ("reading score", "Correlation Coefficient"), "reading score", "1.000")
    chk(t, ("reading score", "Correlation Coefficient"), "writing score", ".617")
    chk(t, ("writing score", "Correlation Coefficient"), "reading score", ".617")
    chk(t, ("reading score", "Sig. (2-tailed)"), "writing score", ".000")
    chk(t, ("writing score", "N"), "writing score", "200")


# ======================================================================
# UCLA whatstat: factor analysis (PCA, 2 factors, varimax)   data: hsb2.sav
def test_ucla_whatstat_factor_pca():
    from rsstatchem.core.stats import factor as F

    ds = _sav("hsb2.sav")
    res = F.factor_analysis(ds, ["read", "write", "math", "science", "socst"],
                            criterion="fixed", n_factors=2, rotation="varimax", scree=False,
                            kmo_bartlett=False)
    t = _tab(res, "Communalities")
    for lab, v in [("reading score", ".736"), ("writing score", ".704"), ("math score", ".750"),
                   ("science score", ".849"), ("social studies score", ".900")]:
        chk_row(t, lab, ["Initial", "Extraction"], ["1.000", v])
    t = _tab(res, "Total Variance Explained")
    spss = [["3.381", "67.616", "67.616", "3.381", "67.616", "67.616", "2.113", "42.267",
             "42.267"],
            [".557", "11.148", "78.764", ".557", "11.148", "78.764", "1.825", "36.497",
             "78.764"],
            [".407", "8.136", "86.900"], [".356", "7.123", "94.023"],
            [".299", "5.977", "100.000"]]
    for i, vals in enumerate(spss):
        chk_row(t, str(i + 1), list(range(1, len(vals) + 1)), vals)


# ======================================================================
# UCLA FAQ "What does Cronbach's alpha mean?" (RELIABILITY + FACTOR)
# https://stats.oarc.ucla.edu/spss/faq/what-does-cronbachs-alpha-mean/   data: alpha.sav
def test_ucla_reliability_alpha():
    from rsstatchem.core.stats import factor as F
    from rsstatchem.core.stats import reliability as RL

    ds = _sav("alpha.sav")
    sec = RL.reliability(ds, ["q1", "q2", "q3", "q4"])
    t = _tab(sec, "Case Processing Summary")
    chk_row(t, "Valid", ["N", "%"], ["60", "100.0"])
    chk_row(t, "Excluded", ["N", "%"], ["0", ".0"])
    t = _tab(sec, "Reliability Statistics")
    r = t.rows[0]
    assert_printed(_num(r[_col(t, "Cronbach's Alpha")]), ".839", "alpha")
    assert _norm(r[_col(t, "N of Items")]) == "4"

    res = F.factor_analysis(ds, ["q1", "q2", "q3", "q4"], sort=True, suppress=0.35,
                            scree=False, kmo_bartlett=False)
    t = _tab(res, "Communalities")
    for lab, v in [("q1", ".585"), ("q2", ".721"), ("q3", ".685"), ("q4", ".714")]:
        chk_row(t, lab, ["Initial", "Extraction"], ["1.000", v])
    t = _tab(res, "Total Variance Explained")
    for i, vals in enumerate([["2.706", "67.654", "67.654", "2.706", "67.654", "67.654"],
                              [".541", "13.531", "81.185"], [".400", "10.006", "91.191"],
                              [".352", "8.809", "100.000"]]):
        chk_row(t, str(i + 1), list(range(1, len(vals) + 1)), vals)
    t = _tab(res, "Component Matrix")
    assert [(_norm(r[0]), _norm(r[1])) for r in t.rows] == [
        ("q2", ".849"), ("q4", ".845"), ("q3", ".828"), ("q1", ".765")]


# ###################################################################################
# UCLA model pages
# ###################################################################################
SUPERSCRIPTS = "ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖʳˢᵗᵘᵛʷˣʸᶻ"


def num(text: str) -> float:
    return float(str(text).rstrip(SUPERSCRIPTS).replace(",", ""))


def items(sec):
    sec = getattr(sec, "section", sec)
    return sec.items


def table(sec, title: str):
    for it in items(sec):
        if getattr(it, "title", "").startswith(title):
            return it
    raise AssertionError(f"table {title!r} not found")


def rows_of(tab, first: int = 1):
    """Rows keyed by their label columns (blank leading labels filled from the row above)."""
    out, last = {}, [""] * first
    for r in tab.rows:
        lab = [r[i] if r[i] != "" else last[i] for i in range(first)]
        last = lab
        out[" | ".join(str(x).rstrip(SUPERSCRIPTS) for x in lab)] = r[first:]
    return out


def check(ours: list, printed: list, msg: str) -> None:
    """Compare a row of our formatted cells with SPSS's printed cells (blank = not printed)."""
    for i, (o, p) in enumerate(zip(ours, printed)):
        p = p.strip()
        if p in ("", ".", "0(a)", "0(b)"):
            continue
        assert_printed(num(o), p, f"{msg} col {i}")


# ===================================================================================
# Factorial ANOVA, GLM Type III (unbalanced 2x3)
# Source: https://stats.oarc.ucla.edu/spss/whatstat/what-statistical-analysis-should-i-usestatistical-analyses-using-spss/#factanova
# Syntax: glm write by female ses.   Data: hsb2.sav
# ===================================================================================
def test_ucla_factorial_anova_glm():
    from rsstatchem.core.stats import glm_output as go

    sec = go.univariate(Dataset(hsb2()), "write", ["female", "ses"], [], go.GLMOptions())
    t = rows_of(table(sec, "Tests of Between-Subjects Effects"))
    expected = {
        "Corrected Model": ["2278.244", "5", "455.649", "5.666", ".000"],
        "Intercept": ["473967.467", "1", "473967.467", "5893.972", ".000"],
        "female": ["1334.493", "1", "1334.493", "16.595", ".000"],
        "ses": ["1063.253", "2", "531.626", "6.611", ".002"],
        "female * ses": ["21.431", "2", "10.715", ".133", ".875"],
        "Error": ["15600.631", "194", "80.416", "", ""],
        "Total": ["574919.000", "200", "", "", ""],
        "Corrected Total": ["17878.875", "199", "", "", ""],
    }
    for k, v in expected.items():
        check(t[k], v, k)
    assert any("R Squared = .127 (Adjusted R Squared = .105)" in f
               for f in table(sec, "Tests of Between-Subjects Effects").footnotes)


# ===================================================================================
# ANCOVA, GLM Type III with a covariate
# Source: https://stats.oarc.ucla.edu/spss/whatstat/what-statistical-analysis-should-i-usestatistical-analyses-using-spss/#ancova
# Syntax: glm write with read by prog.   Data: hsb2.sav
# ===================================================================================
def test_ucla_ancova_glm():
    from rsstatchem.core.stats import glm_output as go

    sec = go.univariate(Dataset(hsb2()), "write", ["prog"], ["read"], go.GLMOptions())
    tab = table(sec, "Tests of Between-Subjects Effects")
    t = rows_of(tab)
    expected = {
        "Corrected Model": ["7017.681", "3", "2339.227", "42.213", ".000"],
        "Intercept": ["4867.964", "1", "4867.964", "87.847", ".000"],
        "read": ["3841.983", "1", "3841.983", "69.332", ".000"],
        "prog": ["650.260", "2", "325.130", "5.867", ".003"],
        "Error": ["10861.194", "196", "55.414", "", ""],
        "Total": ["574919.000", "200", "", "", ""],
        "Corrected Total": ["17878.875", "199", "", "", ""],
    }
    for k, v in expected.items():
        check(t[k], v, k)
    assert any("R Squared = .393 (Adjusted R Squared = .383)" in f for f in tab.footnotes)


# ===================================================================================
# One-way MANOVA, GLM multivariate
# Source: https://stats.oarc.ucla.edu/spss/whatstat/what-statistical-analysis-should-i-usestatistical-analyses-using-spss/#manova
# Syntax: glm read write math by prog.   Data: hsb2.sav
# ===================================================================================
def test_ucla_manova_glm():
    from rsstatchem.core.stats import manova as mv

    sec = mv.manova(Dataset(hsb2()), ["read", "write", "math"], ["prog"], descriptives=False,
                    homogeneity=False, effect_size=False)
    mt = table(sec, "Multivariate Tests")
    t = rows_of(mt, 2)
    expected = {
        "Intercept | Pillai's Trace": [".978", "2883.051", "3.000", "195.000", ".000"],
        "Intercept | Wilks' Lambda": [".022", "2883.051", "3.000", "195.000", ".000"],
        "Intercept | Hotelling's Trace": ["44.355", "2883.051", "3.000", "195.000", ".000"],
        "Intercept | Roy's Largest Root": ["44.355", "2883.051", "3.000", "195.000", ".000"],
        "prog | Pillai's Trace": [".267", "10.075", "6.000", "392.000", ".000"],
        "prog | Wilks' Lambda": [".734", "10.870", "6.000", "390.000", ".000"],
        "prog | Hotelling's Trace": [".361", "11.667", "6.000", "388.000", ".000"],
        "prog | Roy's Largest Root": [".356", "23.277", "3.000", "196.000", ".000"],
    }
    for k, v in expected.items():
        check(t[k], v, k)
    # exact-statistic and upper-bound footnotes, as SPSS marks them
    assert t["prog | Wilks' Lambda"][1].endswith("ᵃ")
    assert t["prog | Roy's Largest Root"][1].endswith("ᵇ")
    be = rows_of(table(sec, "Tests of Between-Subjects Effects"), 2)
    expected = {
        "Corrected Model | read": ["3716.861", "2", "1858.431", "21.282", ".000"],
        "Corrected Model | write": ["3175.698", "2", "1587.849", "21.275", ".000"],
        "Corrected Model | math": ["4002.104", "2", "2001.052", "29.279", ".000"],
        "Intercept | read": ["447178.672", "1", "447178.672", "5120.994", ".000"],
        "Intercept | write": ["460403.797", "1", "460403.797", "6168.704", ".000"],
        "Intercept | math": ["453421.258", "1", "453421.258", "6634.435", ".000"],
        "prog | read": ["3716.861", "2", "1858.431", "21.282", ".000"],
        "prog | write": ["3175.698", "2", "1587.849", "21.275", ".000"],
        "prog | math": ["4002.104", "2", "2001.052", "29.279", ".000"],
        "Error | read": ["17202.559", "197", "87.323"],
        "Error | write": ["14703.177", "197", "74.635"],
        "Error | math": ["13463.691", "197", "68.344"],
        "Total | read": ["566514.000", "200"],
        "Total | write": ["574919.000", "200"],
        "Total | math": ["571765.000", "200"],
        "Corrected Total | read": ["20919.420", "199"],
        "Corrected Total | write": ["17878.875", "199"],
        "Corrected Total | math": ["17465.795", "199"],
    }
    for k, v in expected.items():
        check(be[k], v, k)
    fn = " ".join(table(sec, "Tests of Between-Subjects Effects").footnotes)
    assert "R Squared = .178 (Adjusted R Squared = .169)" in fn
    assert "R Squared = .229 (Adjusted R Squared = .221)" in fn


# ===================================================================================
# One-way MANOVA, old MANOVA procedure (text output, 5 decimals)
# Source: https://stats.oarc.ucla.edu/spss/output/one-waymanova/
# Syntax: manova useful difficulty importance by group(1,3) /print=sig(eigen).
# Data: manova.sav (33 cases)
# ===================================================================================
def test_ucla_manova_oneway_text():
    from rsstatchem.core.stats import manova as mv

    df = read_sav(UCLA + "manova.sav")
    ds = Dataset(df)
    deps = ["useful", "difficulty", "importance"]
    model = mv.LinearModel(df[deps].to_numpy(float),
                           [mv.make_factor(ds, "group", df["group"])], {})
    term = next(t for t in model.terms if t.label == "group")
    H, E = model.hypothesis(term), model.error()
    tests = {t.name: t for t in mv.multivariate_tests(H, E, model.term_df(term), model.ve)}
    pil, wil, hot, roy = (tests["Pillai's Trace"], tests["Wilks' Lambda"],
                          tests["Hotelling's Trace"], tests["Roy's Largest Root"])
    for t, want in ((pil, [".47667", "3.02483", "6.00", "58.00", ".012"]),
                    (hot, [".89723", "4.03753", "6.00", "54.00", ".002"]),
                    (wil, [".52579", "3.53823", "6.00", "56.00", ".005"])):
        for got, w in zip((t.value, t.F, t.df1, t.df2, t.p), want):
            assert_printed(got, w, t.name)
    assert wil.exact                      # "F statistic for WILKS' Lambda is exact."
    # old MANOVA prints Roy's theta = lambda_1 / (1 + lambda_1)
    assert_printed(roy.value / (1 + roy.value), ".47146", "Roy theta")
    # Eigenvalues and canonical correlations of E^-1 H
    eig = np.sort(np.linalg.eigvals(np.linalg.solve(E, H)).real)[::-1][:2]
    pct = 100 * eig / eig.sum()
    for i, (ev, pc, cum, cc) in enumerate(((".892", "99.416", "99.416", ".687"),
                                           (".005", ".584", "100.000", ".072"))):
        assert_printed(eig[i], ev, "eigenvalue")
        assert_printed(pct[i], pc, "pct")
        assert_printed(np.cumsum(pct)[i], cum, "cum pct")
        assert_printed(np.sqrt(eig[i] / (1 + eig[i])), cc, "canon cor")
    # Univariate F-tests with (2,30) D.F.
    sec = mv.manova(ds, deps, ["group"], descriptives=False, homogeneity=False,
                    effect_size=False)
    be = rows_of(table(sec, "Tests of Between-Subjects Effects"), 2)
    for j, (dep, hss, ess, hms, ems, F, p) in enumerate((
            ("useful", "52.92424", "293.96544", "26.46212", "9.79885", "2.70053", ".083"),
            ("difficulty", "3.97515", "126.28728", "1.98758", "4.20958", ".47216", ".628"),
            ("importance", "81.82969", "426.37090", "40.91485", "14.21236", "2.87882",
             ".072"))):
        assert_printed(H[j, j], hss, dep)
        assert_printed(E[j, j], ess, dep)
        assert_printed(H[j, j] / 2, hms, dep)
        assert_printed(E[j, j] / 30, ems, dep)
        assert_printed((H[j, j] / 2) / (E[j, j] / 30), F, dep)
        assert_printed(num(be[f"group | {dep}"][4]), p, dep)
        assert be[f"group | {dep}"][1] == "2" and be[f"Error | {dep}"][1] == "30"


# ===================================================================================
# One-way repeated-measures ANOVA, GLM
# Source: https://stats.oarc.ucla.edu/spss/whatstat/what-statistical-analysis-should-i-usestatistical-analyses-using-spss/#1repmeas
# Syntax: glm y1 y2 y3 y4 /wsfactor a(4).   Data: rb4wide.sav (8 subjects)
# ===================================================================================
def test_ucla_repeated_measures_anova():
    from rsstatchem.core.stats import repeated

    sec = repeated.repeated_measures(Dataset(read_sav(UCLA + "rb4wide.sav")), [("a", 4)],
                                     ["y1", "y2", "y3", "y4"], effect_size=False,
                                     descriptives=False)
    mt = rows_of(table(sec, "Multivariate Tests"), 2)
    for name in ("Pillai's Trace", "Wilks' Lambda", "Hotelling's Trace", "Roy's Largest Root"):
        value = {"Pillai's Trace": ".754", "Wilks' Lambda": ".246"}.get(name, "3.068")
        check(mt[f"a | {name}"], [value, "5.114", "3.000", "5.000", ".055"], name)
    # Mauchly: Sig. includes IBM's second-order (omega_2) term -> .295 (plain chi2 gives .288);
    # Huynh-Feldt epsilon per IBM GLM Algorithms (n d eps - 2)/(d(n - r) - d^2 eps) = .834
    ma = table(sec, "Mauchly's Test of Sphericity").rows[0]
    check(ma[1:], [".339", "6.187", "5", ".295", ".620", ".834", ".333"], "Mauchly")
    w = rows_of(table(sec, "Tests of Within-Subjects Effects"), 2)
    expected = {
        "a | Sphericity Assumed": ["49.000", "3", "16.333", "11.627", ".000"],
        "a | Greenhouse-Geisser": ["49.000", "1.859", "26.365", "11.627", ".001"],
        "a | Huynh-Feldt": ["49.000", "2.503", "19.578", "11.627", ".000"],
        "a | Lower-bound": ["49.000", "1.000", "49.000", "11.627", ".011"],
        "Error(a) | Sphericity Assumed": ["29.500", "21", "1.405"],
        "Error(a) | Greenhouse-Geisser": ["29.500", "13.010", "2.268"],
        "Error(a) | Huynh-Feldt": ["29.500", "17.520", "1.684"],
        "Error(a) | Lower-bound": ["29.500", "7.000", "4.214"],
    }
    for k, v in expected.items():
        check(w[k], v, k)
    c = rows_of(table(sec, "Tests of Within-Subjects Contrasts"), 2)
    expected = {
        "a | Linear": ["44.100", "1", "44.100", "19.294", ".003"],
        "a | Quadratic": ["4.500", "1", "4.500", "3.150", ".119"],
        "a | Cubic": [".400", "1", ".400", ".800", ".401"],
        "Error(a) | Linear": ["16.000", "7", "2.286"],
        "Error(a) | Quadratic": ["10.000", "7", "1.429"],
        "Error(a) | Cubic": ["3.500", "7", ".500"],
    }
    for k, v in expected.items():
        check(c[k], v, k)
    b = rows_of(table(sec, "Tests of Between-Subjects Effects"))
    check(b["Intercept"], ["578.000", "1", "578.000", "128.444", ".000"], "Intercept")
    check(b["Error"], ["31.500", "7", "4.500"], "Error")


def _plum_checks(sec, fit, pseudo, params, parallel):
    mf = rows_of(table(sec, "Model Fitting Information"))
    check(mf["Intercept Only"], [fit[0]], "intercept only -2LL")
    check(mf["Final"], fit[1:], "final")
    ps = rows_of(table(sec, "Pseudo R-Square"))
    for k, v in pseudo.items():
        check(ps[k], [v], k)
    pe = rows_of(table(sec, "Parameter Estimates"), 2)
    for k, v in params.items():
        check(pe[k], v, k)
    if parallel:
        pl = rows_of(table(sec, "Test of Parallel Lines"))
        check(pl["Null Hypothesis"], [parallel[0]], "parallel null")
        check(pl["General"], parallel[1:], "parallel general")


# ===================================================================================
# Ordinal logistic regression, PLUM (UCLA "what statistical analysis")
# Source: https://stats.oarc.ucla.edu/spss/whatstat/what-statistical-analysis-should-i-usestatistical-analyses-using-spss/#ologit
# Syntax: write3 = 1/2/3 for write 30-48/49-57/58-70;
#         plum write3 with female read socst /link = logit /print = parameter summary tparallel.
# PLUM's -2LL is the kernel (covariate-pattern) likelihood; ours prints the same convention.
# Standard errors: SPSS PLUM uses the expected (Fisher) information.
# ===================================================================================
def test_ucla_whatstat_ordinal_plum():
    from rsstatchem.core.stats import logistic

    df = hsb2()
    w = df["write"]
    df["write3"] = np.where(w <= 48, 1.0, np.where(w <= 57, 2.0, 3.0))
    sec = logistic.ordinal(Dataset(df), "write3", [], ["female", "read", "socst"],
                           parallel=True)
    cps = rows_of(table(sec, "Case Processing Summary"), 2)
    for k, v in (("write3 | 1", ["61", "30.5%"]), ("write3 | 2", ["61", "30.5%"]),
                 ("write3 | 3", ["78", "39.0%"])):
        assert [c.rstrip(SUPERSCRIPTS) for c in cps[k]] == v
    _plum_checks(
        sec, ["376.226", "252.151", "124.075", "3", ".000"],
        {"Cox and Snell": ".462", "Nagelkerke": ".521", "McFadden": ".284"},
        {"Threshold | [write3 = 1]": ["9.704", "1.203", "65.109", "1", ".000", "7.347",
                                      "12.061"],
         "Threshold | [write3 = 2]": ["11.800", "1.312", "80.868", "1", ".000", "9.228",
                                      "14.372"],
         "Location | female": ["1.285", ".322", "15.887", "1", ".000", ".653", "1.918"],
         "Location | read": [".118", ".022", "29.867", "1", ".000", ".076", ".160"],
         "Location | socst": [".080", ".019", "17.781", "1", ".000", ".043", ".117"]},
        ["252.151", "250.104", "2.047", "3", ".563"])


# ===================================================================================
# Ordinal logistic regression, PLUM (UCLA DAE page, SPSS 20)
# Source: https://stats.oarc.ucla.edu/spss/dae/ordinal-logistic-regression/
# Syntax: plum apply with pared public gpa /link = logit /print = parameter summary [tparallel].
# Data: ologit.sav. Settles SE(pared) = .268 (Fisher information; observed Hessian gives .266).
# ===================================================================================
def test_ucla_ordinal_plum_logit():
    from rsstatchem.core.stats import logistic

    sec = logistic.ordinal(Dataset(read_sav(UCLA + "ologit.sav")), "apply", [],
                           ["pared", "public", "gpa"], parallel=True)
    cps = rows_of(table(sec, "Case Processing Summary"), 2)
    for k, v in (("apply | 0", ["220", "55.0%"]), ("apply | 1", ["140", "35.0%"]),
                 ("apply | 2", ["40", "10.0%"])):
        assert [c.rstrip(SUPERSCRIPTS) for c in cps[k]] == v
    _plum_checks(
        sec, ["557.272", "533.091", "24.180", "3", ".000"],
        {"Cox and Snell": ".059", "Nagelkerke": ".070", "McFadden": ".033"},
        {"Threshold | [apply = 0]": ["2.203", ".784", "7.890", "1", ".005", ".666", "3.741"],
         "Threshold | [apply = 1]": ["4.299", ".809", "28.224", "1", ".000", "2.713",
                                     "5.885"],
         "Location | pared": ["1.048", ".268", "15.231", "1", ".000", ".522", "1.574"],
         "Location | public": ["-.059", ".289", ".041", "1", ".839", "-.624", ".507"],
         "Location | gpa": [".616", ".263", "5.499", "1", ".019", ".101", "1.130"]},
        ["533.091", "529.077", "4.014", "3", ".260"])


# ===================================================================================
# Probit regression via PLUM /link=probit (UCLA DAE probit page, text output)
# Source: https://stats.oarc.ucla.edu/spss/dae/probit-regression/
# Syntax: plum admit BY rank WITH gre gpa /link=probit /print= parameter summary.
# Data: binary.sav. (The /TEST rank 3-df Wald = 21.361 is not covered: our PLUM has no /TEST.)
# ===================================================================================
def test_ucla_probit_plum_text():
    from rsstatchem.core.stats import crosstabs, descriptive, logistic

    df = read_sav(UCLA_STAT + "binary.sav")
    ds = Dataset(df)
    for var, (n, mn, mx, mean, sd) in (("gre", ("400", "220", "800", "587.70", "115.517")),
                                       ("gpa", ("400", "2.26", "4.00", "3.3899", ".38057"))):
        s = descriptive.summary(df[var])
        for got, want in zip((s["n"], s["min"], s["max"], s["mean"], s["std"]),
                             (n, mn, mx, mean, sd)):
            assert_printed(got, want, var)
    ct = crosstabs.crosstabs(ds, ["admit"], ["rank"], chisq=False)
    tab = next(i for i in items(ct) if "Crosstabulation" in getattr(i, "title", ""))
    counts = [list(r[-5:]) for r in tab.rows if r[-6] == "Count" or r[2] == "Count"]
    assert counts[:3] == [["28", "97", "93", "55", "273"], ["33", "54", "28", "12", "127"],
                          ["61", "151", "121", "67", "400"]]
    sec = logistic.ordinal(ds, "admit", ["rank"], ["gre", "gpa"], link="probit")
    _plum_checks(
        sec, ["493.620", "452.057", "41.563", "5", ".000"],
        {"Cox and Snell": ".099", "Nagelkerke": ".138", "McFadden": ".083"},
        {"Threshold | [admit = 0]": ["3.323", ".663", "25.090", "1", ".000", "2.023",
                                     "4.623"],
         "Location | gre": [".001", ".001", "4.478", "1", ".034", ".000", ".003"],
         "Location | gpa": [".478", ".197", "5.869", "1", ".015", ".091", ".864"],
         "Location | [rank=1]": [".936", ".245", "14.560", "1", ".000", ".455", "1.417"],
         "Location | [rank=2]": [".520", ".211", "6.091", "1", ".014", ".107", ".934"],
         "Location | [rank=3]": [".124", ".224", ".305", "1", ".581", "-.315", ".563"]},
        None)
    pe = rows_of(table(sec, "Parameter Estimates"), 2)
    assert pe["Location | [rank=4]"][0].startswith("0") and pe["Location | [rank=4]"][3] == "0"


# ===================================================================================
# Repeated-measures logistic regression, GENLIN GEE (exchangeable)
# Source: https://stats.oarc.ucla.edu/spss/whatstat/what-statistical-analysis-should-i-usestatistical-analyses-using-spss/#rmlogit
# Syntax: GENLIN highpulse (REFERENCE=LAST) BY diet (order = DESCENDING)
#         /MODEL diet DISTRIBUTION=BINOMIAL LINK=LOGIT /REPEATED SUBJECT=id CORRTYPE=EXCHANGEABLE.
# Data: exercise.sav. SPSS prints Std. Error with 4 decimals; ours prints 3 (compared at 3).
# ===================================================================================
def test_ucla_genlin_gee_logistic():
    from rsstatchem.core.stats import genlin as g

    sec = g.genlin(Dataset(read_sav(UCLA + "exercise.sav")), "highpulse", ["diet"], [],
                   "binomial", "logit", factor_order="descending",
                   repeated={"subject": ["id"], "corr": "exchangeable"})
    te = rows_of(table(sec, "Tests of Model Effects"))
    check(te["(Intercept)"], ["8.437", "1", ".004"], "intercept")
    check(te["diet"], ["1.562", "1", ".211"], "diet")
    pe = rows_of(table(sec, "Parameter Estimates"))
    for k, (b, se, lo, hi, wald, dfv, p) in {
            "(Intercept)": ("1.253", ".4328", ".404", "2.101", "8.377", "1", ".004"),
            "[diet=2]": ("-.754", ".6031", "-1.936", ".428", "1.562", "1", ".211")}.items():
        check([pe[k][i] for i in (0, 2, 3, 4, 5, 6)], [b, lo, hi, wald, dfv, p], k)
        assert_printed(float(se), pe[k][1], f"{k} SE (SPSS value at our 3 decimals)")
    assert pe["[diet=1]"][0].startswith("0")
    assert num(pe["(Scale)"][0]) == 1


# ===================================================================================
# Negative binomial regression, GENLIN negbin(MLE)
# Source: https://stats.oarc.ucla.edu/spss/dae/negative-binomial-regression/
# Syntax: genlin daysabs by prog (order = descending) with math
#         /model prog math distribution = negbin(MLE) link=log.   Data: nb_data.sav (SPSS 19)
# ===================================================================================
def _nb_fit():
    from rsstatchem.core.stats import genlin as g

    return g.genlin(Dataset(read_sav(UCLA + "nb_data.sav")), "daysabs", ["prog"], ["math"],
                    "negbin", "log", factor_order="descending", nb_estimate=True)


def test_ucla_negative_binomial_genlin():
    res = _nb_fit()
    gof = rows_of(table(res, "Goodness of Fit"))
    for k, v in (("Deviance", "358.519"), ("Scaled Deviance", "358.519"),
                 ("Pearson Chi-Square", "339.877"), ("Scaled Pearson Chi-Square", "339.877"),
                 ("Log Likelihood", "-865.629"),
                 ("Akaike's Information Criterion (AIC)", "1741.258"),
                 ("Finite Sample Corrected AIC (AICC)", "1741.453"),
                 ("Bayesian Information Criterion (BIC)", "1760.005"),
                 ("Consistent AIC (CAIC)", "1765.005")):
        check(gof[k], [v], k)
    te = rows_of(table(res, "Tests of Model Effects"))
    check(te["prog"], ["49.214", "2", ".000"], "prog")
    check(te["math"], ["5.714", "1", ".017"], "math")
    pe = rows_of(table(res, "Parameter Estimates"))
    for k, v in {"[prog=3]": ["-1.279", "", "-1.675", "-.883", "40.076", "1", ".000"],
                 "[prog=2]": ["-.441", "", "-.799", "-.083", "5.828", "1", ".016"],
                 "math": ["-.006", "", "-.011", "-.001", "5.714", "1", ".017"],
                 "(Intercept)": ["2.615", "", "2.230", "3.000", "", "1", ".000"]}.items():
        check(pe[k], v, k)
    assert pe["[prog=1]"][0].startswith("0")
    check(pe["(Negative binomial)"], [".968"], "negbin k")
    info = res.models[0]
    se = dict(zip(info.names, info.se))
    for k, v in (("(Intercept)", ".1964"), ("[prog=3]", ".2020"), ("[prog=2]", ".1826"),
                 ("math", ".0025")):
        assert_printed(se[k], v, f"SE {k}")


def test_ucla_negative_binomial_genlin_ancillary():
    res = _nb_fit()
    gof = rows_of(table(res, "Goodness of Fit"))
    check(gof["Deviance"], ["358.519", "309", "1.160"], "Deviance df")
    check(gof["Pearson Chi-Square"], ["339.877", "309", "1.100"], "Pearson df")
    check(table(res, "Omnibus Test").rows[0], ["61.687", "3", ".000"], "omnibus")
    te = rows_of(table(res, "Tests of Model Effects"))
    check(te["(Intercept)"], ["212.477", "1", ".000"], "Type III intercept")
    pe = rows_of(table(res, "Parameter Estimates"))
    check(pe["(Intercept)"], ["2.615", "", "2.230", "3.000", "177.403", "1", ".000"], "int")
    check(pe["(Negative binomial)"], [".968", ".0995", ".792", "1.184"], "negbin k")


# ===================================================================================
# Poisson regression with robust (sandwich, HC0) covariance, GENLIN
# Source: https://stats.oarc.ucla.edu/spss/dae/poisson-regression/
# Syntax: GENLIN num_awards BY prog WITH math /MODEL prog math INTERCEPT=YES
#         DISTRIBUTION=POISSON LINK=LOG /CRITERIA COVB=ROBUST /PRINT FIT SUMMARY SOLUTION.
# Data: poisson_sim.sav (SPSS 19)
# ===================================================================================
def test_ucla_poisson_genlin_robust():
    from rsstatchem.core.stats import genlin as g

    res = g.genlin(Dataset(read_sav(UCLA + "poisson_sim.sav")), "num_awards", ["prog"],
                   ["math"], "poisson", "log", cov_type="robust")
    gof = rows_of(table(res, "Goodness of Fit"))
    for k, v in (("Deviance", ["189.450", "196", ".967"]),
                 ("Scaled Deviance", ["189.450", "196"]),
                 ("Pearson Chi-Square", ["212.144", "196", "1.082"]),
                 ("Scaled Pearson Chi-Square", ["212.144", "196"]),
                 ("Log Likelihood", ["-182.752"]),
                 ("Akaike's Information Criterion (AIC)", ["373.505"]),
                 ("Finite Sample Corrected AIC (AICC)", ["373.710"]),
                 ("Bayesian Information Criterion (BIC)", ["386.698"]),
                 ("Consistent AIC (CAIC)", ["390.698"])):
        check(gof[k], v, k)
    check(table(res, "Omnibus Test").rows[0], ["98.223", "3", ".000"], "omnibus")
    te = rows_of(table(res, "Tests of Model Effects"))
    check(te["(Intercept)"], ["60.952", "1", ".000"], "intercept")
    check(te["prog"], ["14.838", "2", ".001"], "prog")
    check(te["math"], ["45.195", "1", ".000"], "math")
    pe = rows_of(table(res, "Parameter Estimates"))
    for k, v in {"(Intercept)": ["-4.877", "", "-6.112", "-3.643", "59.984", "1", ".000"],
                 "[prog=1]": ["-.370", "", "-1.155", ".415", ".853", "1", ".356"],
                 "[prog=2]": [".714", "", ".129", "1.299", "5.717", "1", ".017"],
                 "math": [".070", "", ".050", ".091", "45.195", "1", ".000"]}.items():
        check(pe[k], v, k)
    assert pe["[prog=3]"][0].startswith("0")
    se = dict(zip(res.models[0].names, res.models[0].se))
    for k, v in (("(Intercept)", ".6297"), ("[prog=1]", ".4004"), ("[prog=2]", ".2986"),
                 ("math", ".0104")):
        assert_printed(se[k], v, f"robust SE {k}")


# ===================================================================================
# Multinomial logistic regression, NOMREG
# Source: https://stats.oarc.ucla.edu/spss/dae/multinomial-logistic-regression/
# Syntax: nomreg prog (base = 2) by ses with write /print = lrt cps mfi parameter summary.
# Data: hsbdemo.sav (SPSS 20). NOMREG -2LL is covariate-pattern based; ours uses the same.
# ===================================================================================
def test_ucla_multinomial_nomreg():
    from rsstatchem.core.stats import descriptive, logistic

    df = read_sav(UCLA_STAT + "hsbdemo.sav")
    for code, (n, mean, sd) in ((1, ("45", "51.33", "9.398")), (2, ("105", "56.26", "7.943")),
                                (3, ("50", "46.76", "9.319"))):
        s = descriptive.summary(df.loc[df["prog"] == code, "write"])
        assert_printed(s["n"], n, "N")
        assert_printed(s["mean"], mean, "mean")
        assert_printed(s["std"], sd, "sd")
    sec = logistic.multinomial(Dataset(df), "prog", ["ses"], ["write"], reference="2")
    mf = rows_of(table(sec, "Model Fitting Information"))
    check(mf["Intercept Only"], ["254.986"], "-2LL intercept only")
    check(mf["Final"], ["206.756", "48.230", "6", ".000"], "final")
    ps = rows_of(table(sec, "Pseudo R-Square"))
    for k, v in (("Cox and Snell", ".214"), ("Nagelkerke", ".246"), ("McFadden", ".118")):
        check(ps[k], [v], k)
    lr = rows_of(table(sec, "Likelihood Ratio Tests"))
    check(lr["Intercept"], ["206.756", ".000", "0"], "intercept")
    check(lr["write"], ["238.203", "31.447", "2", ".000"], "write")
    check(lr["ses"], ["217.815", "11.058", "4", ".026"], "ses")
    pe = rows_of(table(sec, "Parameter Estimates"), 2)
    expected = {
        "1 | Intercept": ["1.689", "1.227", "1.896", "1", ".169"],
        "1 | write": ["-.058", ".021", "7.320", "1", ".007", ".944", ".905", ".984"],
        "1 | [ses=1]": ["1.163", ".514", "5.114", "1", ".024", "3.199", "1.168", "8.764"],
        "1 | [ses=2]": [".630", ".465", "1.833", "1", ".176", "1.877", ".754", "4.669"],
        "3 | Intercept": ["4.236", "1.205", "12.361", "1", ".000"],
        "3 | write": ["-.114", ".022", "26.139", "1", ".000", ".893", ".855", ".932"],
        "3 | [ses=1]": [".983", ".596", "2.722", "1", ".099", "2.672", ".831", "8.585"],
        "3 | [ses=2]": ["1.274", ".511", "6.214", "1", ".013", "3.575", "1.313", "9.736"],
    }
    for k, v in expected.items():
        check(pe[k], v, k)
    for grp in ("1", "3"):
        r = pe[f"{grp} | [ses=3]"]
        assert r[0].startswith("0") and r[3] == "0"


# ===================================================================================
# Canonical correlation (old MANOVA ... WITH ... /discrim all alpha(1), text output)
# Source: https://stats.oarc.ucla.edu/spss/output/canonical-correlation-analysis/
# Syntax: manova locus_of_control self_concept motivation with read write math science female
#         /discrim all alpha(1) /print=sig(eigen dim).   Data: mmr.sav (600 cases)
# Canonical-variate signs are arbitrary (SPSS's orientation rule is undocumented), so each
# function is compared up to one common sign flip of both sets.
# ===================================================================================
def test_ucla_canonical_correlation_text():
    from rsstatchem.core.stats import cancorr
    from rsstatchem.core.stats import manova as mv

    df = read_sav(UCLA + "mmr.sav")
    set1 = ["locus_of_control", "self_concept", "motivation"]
    set2 = ["read", "write", "math", "science", "female"]
    X, Y = df[set1].to_numpy(float), df[set2].to_numpy(float)
    n = len(df)
    # Multivariate tests of the regression effect (WITHIN CELLS Regression)
    model = mv.LinearModel(X, [], {v: df[v].to_numpy(float) for v in set2})
    E = model.E
    Xc = X - X.mean(axis=0)
    H = Xc.T @ Xc - E
    tests = {t.name: t for t in mv.multivariate_tests(H, E, len(set2), model.ve)}
    for name, want in (("Pillai's Trace", [".25425", "11.00057", "15.00", "1782.00", ".000"]),
                       ("Hotelling's Trace", [".31430", "12.37633", "15.00", "1772.00",
                                              ".000"]),
                       ("Wilks' Lambda", [".75436", "11.71573", "15.00", "1634.65", ".000"])):
        t = tests[name]
        for got, w in zip((t.value, t.F, t.df1, t.df2, t.p), want):
            assert_printed(got, w, name)
    roy = tests["Roy's Largest Root"].value
    assert_printed(roy / (1 + roy), ".21538", "Roy theta")
    # Eigenvalues and canonical correlations; dimension reduction analysis
    r, A, B, Rxx, Ryy, Rxy, Zx, Zy = cancorr.canonical(X, Y)
    eig = r ** 2 / (1 - r ** 2)
    pct = 100 * eig / eig.sum()
    for i, (ev, pc, cum, cc, sq) in enumerate((
            (".274", "87.336", "87.336", ".464", ".215"),
            (".029", "9.185", "96.522", ".168", ".028"),
            (".011", "3.478", "100.000", ".104", ".011"))):
        for got, want in zip((eig[i], pct[i], np.cumsum(pct)[i], r[i], r[i] ** 2),
                             (ev, pc, cum, cc, sq)):
            assert_printed(got, want, f"root {i + 1}")
    dims = cancorr.wilks_tests(r, n, len(set1), len(set2))
    for i, want in enumerate(([".75436", "11.71573", "15.00", "1634.65", ".000"],
                              [".96143", "2.94446", "8.00", "1186.00", ".003"],
                              [".98919", "2.16461", "3.00", "594.00", ".091"])):
        for got, w in zip(dims[i], want):
            assert_printed(got, w, f"dimension reduction {i + 1} TO 3")
    # coefficients and correlations (functions compared up to sign)
    sx, sy = X.std(axis=0, ddof=1), Y.std(axis=0, ddof=1)
    spss = {
        "raw_dep": [[1.254, -.621, .662], [-.351, -1.188, -.827], [1.262, 2.027, -2.000]],
        "std_dep": [[.840, -.417, .444], [-.248, -.838, -.583], [.433, .695, -.686]],
        "cor_dep": [[.904, -.390, .176], [.021, -.709, -.705], [.567, .351, -.745]],
        "raw_cov": [[.045, -.005, -.021], [.036, .042, -.091], [.023, .004, -.009],
                    [.005, -.085, .110], [.632, 1.085, 1.795]],
        "std_cov": [[.451, -.050, -.216], [.349, .409, -.888], [.220, .040, -.088],
                    [.049, -.827, 1.066], [.315, .541, .894]],
        "cor_cov": [[.840, -.359, -.135], [.877, .065, -.255], [.764, -.298, -.148],
                    [.658, -.677, .230], [.364, .755, .543]],
    }
    ours = {"raw_dep": A / sx[:, None], "std_dep": A, "cor_dep": Rxx @ A,
            "raw_cov": B / sy[:, None], "std_cov": B, "cor_cov": Ryy @ B}
    sign = np.sign(np.array(spss["std_dep"])[0] / ours["std_dep"][0])
    for key, mat in spss.items():
        mat = np.array(mat)
        for i in range(mat.shape[0]):
            for j in range(3):
                txt = f"{mat[i, j]:.3f}"
                assert_printed(ours[key][i, j] * sign[j], txt, f"{key}[{i},{j}]")
    # variance explained (percent): own set and redundancy
    Lx, Ly = Rxx @ A, Ryy @ B
    own_dep = 100 * np.mean(Lx ** 2, axis=0)
    own_cov = 100 * np.mean(Ly ** 2, axis=0)
    for j, (de, de_red, co, co_red) in enumerate((("37.980", "8.180", "52.488", "11.305"),
                                                  ("25.910", ".727", "24.994", ".701"),
                                                  ("36.111", ".391", "9.066", ".098"))):
        assert_printed(own_dep[j], de, "pct var dependents")
        assert_printed(own_dep[j] * r[j] ** 2, de_red, "dependents explained by covariates")
        assert_printed(own_cov[j], co, "pct var covariates")
        assert_printed(own_cov[j] * r[j] ** 2, co_red, "covariates explained by dependents")
    # the procedure's own output table carries the same correlations and Wilks tests
    sec = cancorr.canonical_correlation(Dataset(df), set1, set2)
    rows = table(sec, "Canonical Correlations").rows
    for i, (cc, wl) in enumerate(((".464", ".754"), (".168", ".961"), (".104", ".989"))):
        assert rows[i][1] == cc and rows[i][3] == wl


# ###################################################################################
# IBM-published SPSS output (IBM SPSS Statistics 29 Case Studies; IBM SPSS Exact Tests
# manual v20). Only printed numbers are used: small tables are rebuilt from printed counts
# and embedded below; IBM's sample files are never used or committed.
# ###################################################################################
def _ds(frame):
    import pandas as pd

    return Dataset(pd.DataFrame(frame).astype(float))


def near(value: float, printed: str, units: float, msg: str = "") -> None:
    """|value - printed| <= units * (last printed digit); for SPSS outputs whose iterative
    estimates stopped at a looser tolerance than ours (PROBIT uses NPSOL)."""
    p = printed.strip().replace(",", "")
    dec = len(p.split(".")[1]) if "." in p else 0
    assert abs(float(value) - float(p)) <= units * 10.0 ** (-dec) + 1e-12, \
        f"{msg}: {value!r} vs SPSS {printed!r}"


# ===================================================================================
# PROBIT with a factor and natural response rate
# Source: IBM SPSS Statistics 29 Case Studies, "Using Probit Analysis to Test Promotional
# Effects on Sales" (probit_offer_*.html). offer.sav is not public; the aggregated data
# below are the printed "Cell Counts and Residuals" table (site, offer value, subjects,
# responses). Syntax: PROBIT respondents OF subjects BY site(1 3) WITH value /LOG E
# /MODEL PROBIT /PRINT FREQ CI PARALL RMP /NATRES /CRITERIA ITERATE(100).
# ===================================================================================
PROBIT_OFFER = {
    "site": [1] * 7 + [2] * 7 + [3] * 7,
    "value": [15, 20, 25, 33, 50, 66, 75] * 3,
    "subjects": [36, 37, 39, 36, 33, 45, 48, 45, 37, 38, 40, 44, 33, 37,
                 43, 37, 48, 45, 33, 51, 38],
    "respondents": [2, 2, 7, 9, 19, 34, 41, 1, 3, 2, 10, 20, 16, 27,
                    2, 2, 5, 5, 10, 22, 23],
}


def test_ibm_probit_offer():
    from rsstatchem.core.stats import genlin as g

    sec = g.probit(_ds(PROBIT_OFFER), "respondents", "subjects", "value", factor="site",
                   transform="ln", natural="estimate", relative_potency=True,
                   parallelism=True)
    chi = rows_of(table(sec, "Chi-Square Tests"), 2)
    # Pearson goodness of fit and the IBM parallelism test (chi2_0 - sum chi2_j, groups
    # refitted separately with their own natural response rate)
    check(chi["PROBIT | Pearson Goodness-of-Fit Test"], ["8.934", "16", ".916"], "Pearson")
    check(chi["PROBIT | Parallelism Test"], ["2.060", "2", ".357"], "parallelism")
    pe = rows_of(table(sec, "Parameter Estimates"), 2)
    # slope: estimate, SE (unscaled I^-1: no heterogeneity factor in the parameter table),
    # Z, Sig., normal 95% CI
    check(pe["PROBIT | value"], ["1.880", ".216", "8.719", ".000", "1.457", "2.303"], "b")
    # intercepts: estimate, SE and Z. (SPSS prints Sig. 1.000 for these negative Z values and
    # an interval of +-1 SE; that display is not reproduced.)
    for lab, want in (("1", ["-7.219", ".861", "-8.384"]), ("2", ["-7.631", ".888", "-8.590"]),
                      ("3", ["-7.982", ".928", "-8.601"])):
        check(pe[f"PROBIT | {lab}"][:3], want, f"intercept {lab}")
    nr = table(sec, "Natural Response Rate").rows[0]
    check(nr[1:], [".041", ".019"], "natural response")
    cells = table(sec, "Cell Counts and Residuals").rows
    printed = [("2.048", "-.048", ".057"), ("3.509", "-1.509", ".095"),
               ("6.139", ".861", ".157"), ("10.425", "-1.425", ".290"),
               ("18.883", ".117", ".572"), ("33.978", ".022", ".755"),
               ("39.502", "1.498", ".823"), ("2.079", "-1.079", ".046"),
               ("2.323", ".677", ".063"), ("3.637", "-1.637", ".096"),
               ("7.206", "2.794", ".180"), ("18.312", "1.688", ".416"),
               ("20.250", "-4.250", ".614"), ("25.880", "1.120", ".699"),
               ("1.837", ".163", ".043"), ("1.846", ".154", ".050"),
               ("3.195", "1.805", ".067"), ("5.271", "-.271", ".117"),
               ("9.745", ".255", ".295"), ("24.493", "-2.493", ".480"),
               ("21.735", "1.265", ".572")]
    for r, (e, res, pr) in zip(cells, printed):
        # SPSS's NPSOL fit stops about 1e-3 short of the optimum on the expected counts
        near(num(r[5]), e, 2, "expected")
        near(num(r[6]), res, 2, "residual")
        assert_printed(num(r[7]), pr, "probability")
    cl = table(sec, "Confidence Limits").rows
    by = {}
    head = None
    for r in cl:
        head = r[0] or head
        by[(head, r[1])] = r[2:]
    fid = {("1", ".500"): ["46.518", "41.151", "51.899", "3.840", "3.718", "3.950"],
           ("1", ".250"): ["32.495", "26.985", "37.173", "3.481", "3.296", "3.616"],
           ("2", ".500"): ["57.905", "51.150", "65.365", "4.059", "3.935", "4.180"],
           ("2", ".750"): ["82.893", "72.946", "97.584", "4.418", "4.290", "4.581"],
           ("3", ".500"): ["69.800", "61.136", "79.518", "4.246", "4.114", "4.376"],
           ("3", ".300"): ["52.811", "44.769", "60.343", "3.967", "3.802", "4.100"]}
    for key, want in fid.items():
        for o, w in zip(by[key], want):
            near(num(o), w, 3, f"fiducial {key}")
    rmp = {(r[0], r[1]): r[2:] for r in table(sec, "Relative Median Potency").rows}
    # IBM's printed RMP formula M_jk = (a_k - a_j)/b with its Fieller limits
    want = {("1", "2"): [".803", ".660", ".942", "-.219", "-.415", "-.059"],
            ("1", "3"): [".666", ".508", ".811", "-.406", "-.678", "-.209"],
            ("2", "1"): ["1.245", "1.061", "1.514", ".219", ".059", ".415"],
            ("2", "3"): [".830", ".673", ".983", "-.187", "-.396", "-.017"],
            ("3", "2"): ["1.205", "1.017", "1.485", ".187", ".017", ".396"],
            ("3", "1"): ["1.500", "1.233", "1.969", ".406", ".209", ".678"]}
    for key, w in want.items():
        for o, p in zip(rmp[key], w):
            near(num(o), p, 1, f"RMP {key}")


# ===================================================================================
# Logit loglinear analysis: Analysis of Dispersion
# Source: IBM SPSS Statistics 29 Case Studies (loglog_cereal_disp.html). cereal.sav is not
# public; the Total row and every df depend only on N = 880, the dependent margins
# {231, 310, 339} (the unique integer solution reproducing the printed totals) and the
# design, so a table with those margins and the same design checks them exactly.
# Design: bfast + bfast*active + bfast*agecat, cells formed by active, gender, agecat.
# ===================================================================================
def test_ibm_logit_loglinear_dispersion_cereal():
    from rsstatchem.core.stats import genlin as g

    rng = np.random.default_rng(1)
    bf = np.repeat([1.0, 2.0, 3.0], [231, 310, 339])
    rng.shuffle(bf)
    frame = {"bfast": bf, "active": rng.integers(0, 2, 880), "gender": rng.integers(0, 2, 880),
             "agecat": rng.integers(1, 5, 880)}
    sec = g.loglinear_logit(_ds(frame), ["bfast"], ["active", "gender", "agecat"],
                            terms=["bfast", "bfast*active", "bfast*agecat"],
                            frequencies=False, estimates=False)
    disp = rows_of(table(sec, "Analysis of Dispersion"))
    # natural-log entropy and Gini concentration N(1 - sum p^2) with no 1/2 factor;
    # case-based df f(r-1), (N-f-1)(r-1), (N-1)(r-1)
    check(disp["Total"], ["955.781", "579.566", "1758"], "total")
    assert disp["Model"][2] == "8" and disp["Residual"][2] == "1750"
    gof = rows_of(table(sec, "Goodness-of-Fit Tests"))
    assert gof["Likelihood Ratio"][1] == "22"       # printed df 22
    # measures of association = Model / Total of each column
    assoc = rows_of(table(sec, "Measures of Association"))
    for k, col in (("Entropy", 0), ("Concentration", 1)):
        ratio = num(disp["Model"][col]) / num(disp["Total"][col])
        assert abs(num(assoc[k][0]) - ratio) < 1e-3


# ===================================================================================
# RELIABILITY /MODEL=PARALLEL
# Source: IBM SPSS Statistics 29 Case Studies (rely_tvsurvey_parallel.html), tv-survey.sav
# (906 cases, 7 items; not public). Printed: chi-square 1274.557, df 26, log determinants
# -13.362 (unconstrained) and -11.949 (constrained).
# ===================================================================================
def test_ibm_reliability_parallel_tvsurvey():
    from rsstatchem.core.stats import reliability as rel

    W, k = 906, 7
    dof = k * (k + 1) // 2 - 2
    corr = 1 - k * (k + 1) ** 2 * (2 * k - 3) / (12 * (k - 1) * dof * (W - 1))
    # IBM's goodness-of-fit chi-square from the printed log determinants, with the
    # correction multiplier: 1274.557 within the rounding of the two log determinants
    chi_printed = -(W - 1) * corr * (-13.362 - -11.949)
    assert dof == 26
    assert abs(chi_printed - 1274.557) <= (W - 1) * corr * 0.001
    # without the multiplier the value would be outside that band
    assert abs((W - 1) * 1.413 - 1274.557) > (W - 1) * 0.001
    # our parallel / strict-parallel model implements the same formulas
    x = (np.random.default_rng(5).random((W, k)) < np.linspace(.3, .6, k)).astype(float)
    for strict in (False, True):
        pm = rel.parallel_model(x, strict=strict)
        d = k * (k + 3) // 2 - 3 if strict else dof
        c = 1 - k * (k + 1) ** 2 * (2 * k - 3) / (12 * (k - 1) * d * (W - 1))
        assert pm["df"] == d
        assert pm["chi"] == pytest.approx(-(W - 1) * c * (pm["logdet_u"] - pm["logdet_c"]))
        assert pm["true"] + pm["error"] == pytest.approx(pm["common"])
        if strict:
            assert pm["unbiased"] == pytest.approx((3 + (W - 3) * pm["reliability"]) / W)
        else:
            assert pm["unbiased"] == pytest.approx((2 + (W - 3) * pm["reliability"]) / (W - 1))
            assert pm["reliability"] == pytest.approx(rel.cronbach_alpha(x))


# ===================================================================================
# CROSSTABS Goodman-Kruskal tau, uncertainty coefficient, phi / Cramer's V
# Source: IBM SPSS Exact Tests manual v20, ch. 15, figs 15.2-15.6 (counts printed inline):
# party preference x preferred cold-war ally (Bishop, Fienberg & Holland 1975) and survey
# disposition x financial organisation.
# ===================================================================================
PARTY_ALLY = np.array([[225, 3], [53, 1], [206, 12]], float)
SURVEY_ORG = np.array([[8, 8, 3, 11, 17, 2], [2, 5, 1, 2, 0, 13], [12, 8, 15, 19, 18, 0]],
                      float)    # columns AAA, AICPA, FAF, FASB, FEI, NAA


def test_ibm_exact_crosstabs_nominal_uncertainty_gktau():
    from rsstatchem.core.stats import crosstabs as ct

    m = ct.nominal_measures(PARTY_ALLY)
    # Goodman and Kruskal tau: value, ASE1 (IBM formula as SPSS computes it), chi-square Sig.
    for key, want in (("tau_R", [".010", ".006", ".008"]), ("tau_C", [".013", ".010", ".036"])):
        mm = m[key]
        for v, w in zip((mm.value, mm.ase, mm.p), want):
            assert_printed(v, w, key)
    for key, want in (("uc_S", [".012", ".009", "1.346", ".033"]),
                      ("uc_R", [".007", ".005", "1.346", ".033"]),
                      ("uc_C", [".048", ".034", "1.346", ".033"])):
        mm = m[key]
        for v, w in zip((mm.value, mm.ase, mm.t, mm.p), want):
            assert_printed(v, w, key)
    m3 = ct.nominal_measures(SURVEY_ORG[:, :3])
    for key, want in (("phi", [".359", ".092"]), ("v", [".254", ".092"])):
        assert_printed(m3[key].value, want[0], key)
        assert_printed(m3[key].p, want[1], key)
    m6 = ct.nominal_measures(SURVEY_ORG)
    assert_printed(m6["phi"].value, ".723", "phi")
    assert_printed(m6["v"].value, ".511", "V")
    assert_printed(m6["phi"].p, ".000", "phi Sig")


# ===================================================================================
# NPAR TESTS /MH (marginal homogeneity)
# Source: IBM SPSS Exact Tests manual v20, ch. 5, figs 5.7-5.10: endometrial cancer doses
# (Breslow & Day 1980) and Pap-smear ratings of two pathologists (Agresti 1990).
# ===================================================================================
MH_DOSE = [(0, 0, 6), (0, .2, 2), (0, .5125, 3), (0, .7, 1), (.2, 0, 9), (.2, .2, 4),
           (.2, .5125, 2), (.2, .7, 1), (.5125, 0, 9), (.5125, .2, 2), (.5125, .5125, 3),
           (.5125, .7, 1), (.7, 0, 12), (.7, .2, 1), (.7, .5125, 2), (.7, .7, 1)]
MH_PAP = [(1, 1, 22), (1, 2, 2), (1, 3, 2), (2, 1, 5), (2, 2, 7), (2, 3, 14), (3, 2, 2),
          (3, 3, 36), (4, 2, 1), (4, 3, 14), (4, 4, 7), (5, 3, 3), (5, 5, 3)]


def test_ibm_exact_marginal_homogeneity():
    from rsstatchem.core.stats import nptests as T

    for cells, want in ((MH_DOSE, ["4", "45", "6.687", "12.869", "1.655", "-3.735", ".000"]),
                        (MH_PAP, ["5", "43", "114.000", "118.500", "3.905", "-1.152", ".249"])):
        a = np.repeat([c[0] for c in cells], [c[2] for c in cells])
        b = np.repeat([c[1] for c in cells], [c[2] for c in cells])
        r = T.marginal_homogeneity_test(a, b)
        assert (str(r["distinct"]), str(r["off"])) == tuple(want[:2])
        for key, w in zip(("obs", "mean", "sd", "z", "p"), want[2:]):
            assert_printed(r[key], w, key)
    # (the exact significances .307 / .154 / .053 of fig. 5.10 are not implemented)


# ===================================================================================
# GENLIN Poisson log-linear with offset and Pearson scale
# Source: IBM SPSS Statistics 29 Case Studies, "Using Poisson Regression to Analyze Ship
# Damage Rates" (genlin_ships_*.html). Data: McCullagh & Nelder (1989) Table 6.2 (as in
# MASS::ships), the 34 rows with service > 0, embedded as numbers. Syntax: GENLIN incidents
# BY type construction operation (ORDER=DESCENDING) /MODEL ... DISTRIBUTION=POISSON LINK=LOG
# OFFSET=log_months_service /CRITERIA SCALE=PEARSON.
# ===================================================================================
SHIPS = [(1, 60, 60, 127, 0), (1, 60, 75, 63, 0), (1, 65, 60, 1095, 3), (1, 65, 75, 1095, 4),
         (1, 70, 60, 1512, 6), (1, 70, 75, 3353, 18), (1, 75, 75, 2244, 11),
         (2, 60, 60, 44882, 39), (2, 60, 75, 17176, 29), (2, 65, 60, 28609, 58),
         (2, 65, 75, 20370, 53), (2, 70, 60, 7064, 12), (2, 70, 75, 13099, 44),
         (2, 75, 75, 7117, 18), (3, 60, 60, 1179, 1), (3, 60, 75, 552, 1), (3, 65, 60, 781, 0),
         (3, 65, 75, 676, 1), (3, 70, 60, 783, 6), (3, 70, 75, 1948, 2), (3, 75, 75, 274, 1),
         (4, 60, 60, 251, 0), (4, 60, 75, 105, 0), (4, 65, 60, 288, 0), (4, 65, 75, 192, 0),
         (4, 70, 60, 349, 2), (4, 70, 75, 1208, 11), (4, 75, 75, 2051, 4), (5, 60, 60, 45, 0),
         (5, 65, 60, 789, 7), (5, 65, 75, 437, 7), (5, 70, 60, 1157, 5),
         (5, 70, 75, 2161, 12), (5, 75, 75, 542, 1)]


def test_ibm_genlin_poisson_ships():
    from rsstatchem.core.stats import genlin as g

    arr = np.array(SHIPS, float)
    frame = {"type": arr[:, 0], "construction": arr[:, 1], "operation": arr[:, 2],
             "log_months_service": np.log(arr[:, 3]), "incidents": arr[:, 4]}
    res = g.genlin(_ds(frame), "incidents", ["type", "construction", "operation"],
                   distribution="poisson", link="log", offset="log_months_service",
                   factor_order="descending", scale_method="pearson")
    gof = rows_of(table(res, "Goodness of Fit"))
    for k, v in (("Deviance", ["38.695", "25", "1.548"]), ("Scaled Deviance", ["22.883", "25"]),
                 ("Pearson Chi-Square", ["42.275", "25", "1.691"]),
                 ("Scaled Pearson Chi-Square", ["25.000", "25"]),
                 # IBM: with a deviance/Pearson scale the displayed full log likelihood and
                 # the information criteria use the assumed phi = 1
                 ("Log Likelihood", ["-68.281"]),
                 ("Akaike's Information Criterion (AIC)", ["154.562"]),
                 ("Finite Sample Corrected AIC (AICC)", ["162.062"]),
                 ("Bayesian Information Criterion (BIC)", ["168.299"]),
                 ("Consistent AIC (CAIC)", ["177.299"])):
        check(gof[k], v, k)
    pe = rows_of(table(res, "Parameter Estimates"))
    want = {"(Intercept)": ["-6.406", ".2828", "-6.960", "-5.852", "513.238", "1", ".000"],
            "[type=5]": [".326", ".3067", "-.276", ".927", "1.127", "1", ".288"],
            "[type=4]": ["-.076", ".3779", "-.817", ".665", ".040", "1", ".841"],
            "[type=3]": ["-.687", ".4279", "-1.526", ".151", "2.581", "1", ".108"],
            "[type=2]": ["-.543", ".2309", "-.996", "-.091", "5.536", "1", ".019"],
            "[construction=75]": [".453", ".3032", "-.141", "1.048", "2.236", "1", ".135"],
            "[construction=70]": [".818", ".2208", ".386", "1.251", "13.743", "1", ".000"],
            "[construction=65]": [".697", ".1946", ".316", "1.079", "12.835", "1", ".000"],
            "[operation=75]": [".384", ".1538", ".083", ".686", "6.249", "1", ".012"]}
    for k, v in want.items():
        check(pe[k], v, k)
    for k in ("[type=1]", "[construction=60]", "[operation=60]"):
        assert pe[k][0].startswith("0")
    assert_printed(num(pe["(Scale)"][0]), "1.691", "scale")


# ===================================================================================
# BAYES INDEPENDENT, posterior with diffuse priors and unequal variances
# Source: IBM SPSS Statistics 29 Case Studies (bayesian_independent_results.html), salary by
# gender in Employee data.sav (not public). The posterior depends only on the printed group
# statistics, so data with exactly those n / means / SDs reproduce it.
# ===================================================================================
def test_ibm_cs_bayes_independent_salary():
    from rsstatchem.core.stats import bayes as bs

    def exact(n, mean, sd, seed):
        z = np.random.default_rng(seed).normal(size=n)
        return mean + sd * (z - z.mean()) / z.std(ddof=1)

    female, male = exact(216, 26031.92, 7558.021, 1), exact(258, 41441.78, 19499.214, 2)
    frame = {"salary": np.r_[female, male], "gender": np.r_[np.zeros(216), np.ones(258)]}
    sec = bs.independent_normal(_ds(frame), ["salary"], "gender", 0.0, 1.0,
                                bs.BayesOptions(analysis="posterior"), equal_var=False)
    row = table(sec, "Posterior Distribution Characterization").rows[0]
    # group 2 - group 1 (male - female); Patil's Behrens-Fisher approximation
    check(row[2:], ["15409.86", "15409.86", "", "12813.84", "18005.88"], "posterior")
    # the variance (printed 1752221.660) inherits the rounding of the printed SDs (+-0.08)
    assert abs(num(row[4]) - 1752221.660) < 0.1


# ===================================================================================
# Kaplan-Meier (structural): IBM SPSS Statistics 29 Case Studies, pain_medication.sav (not
# public). SPSS prints median CI = estimate +- 1.96 SE and the three tests on 1 df; ours
# follows the same rules (checked on a public data set) and the printed rows are consistent.
# ===================================================================================
def test_ibm_cs_kaplan_meier_painmeds():
    from scipy import stats

    from rsstatchem.core.stats import survival as sv

    for est, se, lo, hi in (("3.700", ".292", "3.128", "4.272"),
                            ("4.867", ".360", "4.162", "5.572"),
                            ("5.185", ".350", "4.499", "5.871")):
        near(float(est) - 1.96 * float(se), lo, 2)
        near(float(est) + 1.96 * float(se), hi, 2)
    for chi, sig in (("0.379", ".538"), ("0.748", ".387"), ("0.705", ".401")):
        near(stats.chi2.sf(float(chi), 1), sig, 1)
    rng = np.random.default_rng(3)
    t = np.round(rng.exponential(5, 60), 1) + 0.1
    d = (rng.random(60) < 0.8).astype(float)
    c = sv.km_fit(t, d)
    med, se = sv.km_percentile(c, 0.5)
    sec = sv.kaplan_meier(_ds({"t": t, "d": d}), "t", "d", ("single", 1.0))
    mm = table(sec, "Means and Medians").rows[0]
    vals = [num(v) for v in mm if v not in ("",) and _isnum(v)]
    assert any(abs(v - (med - 1.959964 * se)) < 6e-4 for v in vals)


def _isnum(v) -> bool:
    try:
        num(v)
        return True
    except ValueError:
        return False


# ===================================================================================
# META-ANALYSIS (Psych 2022, 4(4), 49; CC BY 4.0): the SPSS output is published only as
# images and has not been transcribed yet, so there is nothing to compare against.
# ===================================================================================
@pytest.mark.skip(reason="mdpi2022 meta-analysis: SPSS output only as images, not transcribed")
def test_mdpi2022_meta_analysis():
    pass


# ###################################################################################
# IBM SPSS Statistics Algorithms: formula checks for the WP7 changes that no published
# SPSS output covers (kept next to the fixtures so the IBM conventions stay pinned).
# ###################################################################################
def test_ibm_algorithms_bayes_conventions():
    from scipy import integrate

    from rsstatchem.core.stats import bayes as bs

    # JZS t test: IG(1/2, 1/2) mixing (Cauchy scale 1) is the default
    t, n = 2.3, 25

    def ibm_bf01(t, n):
        nu = n - 1
        num_ = (1 + t * t / nu) ** (-(nu + 1) / 2)
        den = integrate.quad(lambda g: (1 + n * g) ** -0.5 * (1 + t * t / ((1 + n * g) * nu))
                             ** (-(nu + 1) / 2) * (2 * np.pi) ** -0.5 * g ** -1.5
                             * np.exp(-1 / (2 * g)), 0, np.inf)[0]
        return num_ / den
    assert bs.jzs_ttest_bf10(t, n) == pytest.approx(1 / ibm_bf01(t, n), rel=1e-6)
    # Wetzels-Wagenmakers correlation BF = Zellner-Siow regression BF with one predictor
    assert bs.wetzels_pearson_bf10(0.3, 50) == pytest.approx(
        bs.regression_bf10(0.09, 50, 1, "jzs", r=1.0), rel=1e-6)
    # binomial with a Beta(a0, b0) null: IBM eq. (1)
    from scipy.special import betaln
    k, nn = 7, 20
    bf01 = np.exp(betaln(3 + k, 4 + nn - k) + betaln(1, 1) - betaln(1 + k, 1 + nn - k)
                  - betaln(3, 4))
    assert bs.binom_bf10(k, nn, 0.5, 1, 1, 3, 4) == pytest.approx(1 / bf01)
    # hyper-prior two-sample BF, closed form eq. (18), a = -0.75
    from scipy.special import gammaln
    nu = 38
    ref = np.exp(gammaln(nu / 2) + gammaln(0.75) - gammaln((nu + 1) / 2) - gammaln(0.25)
                 + (nu + 1.5 - 2) / 2 * np.log1p(2.5 ** 2 / nu))
    assert bs.hyper_prior_ttest_bf10(2.5, 20, 20) == pytest.approx(ref)


def test_ibm_algorithms_oneway_omega_random():
    from rsstatchem.core.stats import anova
    from rsstatchem.core.stats.common import Group

    rng = np.random.default_rng(2)
    groups = [Group(i, str(i), rng.normal(i * 0.4, 1, n)) for i, n in enumerate((8, 11, 9))]
    F, dfb, dfw = 4.2, 2, 25
    cells = anova._omega_random(F, dfb, dfw, groups, 0.95)
    dft = dfb + dfw
    assert num(cells[0]) == pytest.approx((F - 1) / (F + dft), abs=6e-4)
    e_lo, e_hi = anova.eta_interval(F, dfb, dfw, 0.95)
    lim = [(dft * e - dfb) / ((dfw - dft * dfb) * e + dft * dfb) for e in (e_lo, e_hi)]
    assert num(cells[1]) == pytest.approx(lim[0], abs=6e-4)
    assert num(cells[2]) == pytest.approx(lim[1], abs=6e-4)


def test_ibm_algorithms_glm_type3_type4_empty_cells():
    from rsstatchem.core.stats import glm_model as gm

    rng = np.random.default_rng(1)
    cells = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2)]   # (1, 2) empty
    reps = [3, 5, 2, 4, 6, 2, 4, 5]
    a = np.repeat([c[0] for c in cells], reps)
    b = np.repeat([c[1] for c in cells], reps)
    y = rng.normal(size=len(a)) + 0.5 * a + 0.3 * b

    def fit(ss, aa=a, bb=b, yy=y):
        fa = gm.FactorInfo("A", sorted(set(aa)), [str(v) for v in sorted(set(aa))], aa)
        fb = gm.FactorInfo("B", sorted(set(bb)), [str(v) for v in sorted(set(bb))], bb)
        terms = [gm.TermSpec(("A",)), gm.TermSpec(("B",)), gm.TermSpec(("A", "B"))]
        return gm.GLMFit(yy, [fa, fb], {}, terms, True, ss)
    f3, f4 = fit(3), fit(4)
    assert f3.empty_cells
    t3 = {t.label: (float(f3.hypothesis(t)[0, 0]), f3.term_df(t)) for t in f3.terms}
    t4 = {t.label: (float(f4.hypothesis(t)[0, 0]), f4.term_df(t)) for t in f4.terms}
    assert t3["A * B"][1] == 3 and t3["A"][1] == 2
    assert t3["A * B"][0] == pytest.approx(t4["A * B"][0])       # highest-order term
    assert t3["A"][0] != pytest.approx(t4["A"][0])                # III and IV now differ
    # Type IV for A, by hand from IBM's rule: after elimination the A rows are A2 - A0 and
    # A1 - A0, spread over the B levels common to the compared A levels
    m = np.array([y[(a == i) & (b == j)].mean() for i, j in cells])
    nn = np.array(reps, float)
    L = np.zeros((2, 8))
    for c, (i, j) in enumerate(cells):
        L[0, c] = {2: 1 / 3, 0: -1 / 3}.get(i, 0.0)
        L[1, c] = 0.5 if i == 1 else (-0.5 if i == 0 and j < 2 else 0.0)
    Lm = L @ m
    ss4 = Lm @ np.linalg.solve(L @ np.diag(1 / nn) @ L.T, Lm)
    assert t4["A"][0] == pytest.approx(ss4)
    # without empty cells IBM's construction equals the effect-coded Type III
    full = [(i, j) for i in range(3) for j in range(3)]
    reps2 = [3, 5, 2, 4, 6, 3, 2, 4, 5]
    a2 = np.repeat([c[0] for c in full], reps2)
    b2 = np.repeat([c[1] for c in full], reps2)
    y2 = rng.normal(size=len(a2))
    g = fit(3, a2, b2, y2)
    eff = [float(g.hypothesis(t)[0, 0]) for t in g.terms]
    g.empty_cells = True
    assert [float(g.hypothesis(t)[0, 0]) for t in g.terms] == pytest.approx(eff)


def test_ibm_algorithms_cox_kalbfleisch_prentice_baseline():
    from rsstatchem.core.stats import survival as sv

    # tied events: alpha solves sum_D e/(1 - alpha^e) = sum_R e (IBM COXREG baseline)
    e = np.array([0.8, 1.3])
    risk = 5.0
    alpha = sv._kp_alpha(e, risk)
    assert np.sum(e / (1 - alpha ** e)) == pytest.approx(risk)
    # single event: closed form
    assert sv._kp_alpha(np.array([1.7]), 9.0) == pytest.approx((1 - 1.7 / 9.0) ** (1 / 1.7))


def test_ibm_algorithms_regw_uses_group_sizes():
    from rsstatchem.core.stats import posthoc_ext as ph
    from rsstatchem.core.stats.common import Group

    rng = np.random.default_rng(4)
    groups = [Group(i, str(i), rng.normal(mu, 1, n))
              for i, (mu, n) in enumerate(((0, 4), (0.6, 30), (1.1, 6), (1.3, 25)))]
    mse = float(np.mean([g.data.var(ddof=1) for g in groups]))
    dfe = sum(g.n for g in groups) - 4
    for method in ("regwf", "regwq"):
        order, subsets, sig = ph.homogeneous_subsets(groups, method, mse, dfe)
        assert set(sum(subsets, [])) == set(range(4)) and all(subsets)
        assert all(0 <= s <= 1 for s in sig if s == s)


def test_ibm_algorithms_bootstrap_rules():
    from rsstatchem.core.stats import resample as rs

    vals = np.sort(np.random.default_rng(1).normal(size=999))
    # (B+1) alpha integer -> the order statistic itself
    assert rs.ibm_order_statistic(vals, 0.025) == vals[24]
    vals2 = vals[:998]
    pos = 999 * 0.025
    k = int(pos)
    from scipy.stats import norm
    want = vals2[k - 1] + (norm.ppf(0.025) - norm.ppf(k / 999)) / \
        (norm.ppf((k + 1) / 999) - norm.ppf(k / 999)) * (vals2[k] - vals2[k - 1])
    assert rs.ibm_order_statistic(vals2, 0.025) == pytest.approx(want)
