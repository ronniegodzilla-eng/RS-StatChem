"""WP6: Compute function library (dates, distributions, strings) vs scipy/pandas/datetime."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from rsstatchem.core import expr_functions as X
from rsstatchem.core.dataset import Dataset
from rsstatchem.core.expression import FUNCTIONS, ExpressionError, evaluate


@pytest.fixture()
def ds():
    df = pd.DataFrame({"q": [0.2, 1.0, 2.5, 4.0], "p": [0.1, 0.25, 0.5, 0.9],
                       "k": [0.0, 1.0, 3.0, 5.0], "y": [2000.0, 2020.0, 1999.0, 2024.0],
                       "m": [1.0, 2.0, 12.0, 2.0], "d": [1.0, 29.0, 31.0, 29.0],
                       "s": ["Banda Aceh", "Medan", "  Padang", "Jakarta  "],
                       "g": [1.0, 2.0, 1.0, np.nan]})
    d = Dataset(df)
    d.var("g").value_labels = {1.0: "Laki-laki", 2.0: "Perempuan"}
    return d


def test_library_size():
    assert len(FUNCTIONS) >= 165  # 45 before WP6


@pytest.mark.parametrize("expr,ref", [
    ("CDF.BETA(p, 2, 3)", lambda d: stats.beta.cdf(d.p, 2, 3)),
    ("PDF.BETA(p, 2, 3)", lambda d: stats.beta.pdf(d.p, 2, 3)),
    ("IDF.BETA(p, 2, 3)", lambda d: stats.beta.ppf(d.p, 2, 3)),
    ("CDF.BINOM(k, 5, 0.3)", lambda d: stats.binom.cdf(d.k, 5, 0.3)),
    ("PDF.BINOM(k, 5, 0.3)", lambda d: stats.binom.pmf(d.k, 5, 0.3)),
    ("CDF.POISSON(k, 2.5)", lambda d: stats.poisson.cdf(d.k, 2.5)),
    ("PDF.POISSON(k, 2.5)", lambda d: stats.poisson.pmf(d.k, 2.5)),
    ("CDF.CHISQ(q, 3)", lambda d: stats.chi2.cdf(d.q, 3)),
    ("IDF.CHISQ(p, 3)", lambda d: stats.chi2.ppf(d.p, 3)),
    ("SIG.CHISQ(q, 3)", lambda d: stats.chi2.sf(d.q, 3)),
    ("SIG.F(q, 2, 10)", lambda d: stats.f.sf(d.q, 2, 10)),
    ("IDF.F(p, 2, 10)", lambda d: stats.f.ppf(d.p, 2, 10)),
    ("CDF.T(q, 7)", lambda d: stats.t.cdf(d.q, 7)),
    ("IDF.T(p, 7)", lambda d: stats.t.ppf(d.p, 7)),
    ("CDF.GAMMA(q, 2, 0.5)", lambda d: stats.gamma.cdf(d.q, 2, scale=2)),  # rate 0.5
    ("CDF.EXP(q, 2)", lambda d: stats.expon.cdf(d.q, scale=0.5)),          # rate 2
    ("IDF.EXP(p, 2)", lambda d: stats.expon.ppf(d.p, scale=0.5)),
    ("CDF.WEIBULL(q, 2, 1.5)", lambda d: 1 - np.exp(-(d.q / 2) ** 1.5)),
    ("CDF.LNORMAL(q, 2, 0.5)", lambda d: stats.norm.cdf((np.log(d.q) - np.log(2)) / 0.5)),
    ("CDF.LOGISTIC(q, 1, 2)", lambda d: stats.logistic.cdf(d.q, 1, 2)),
    ("CDF.LAPLACE(q, 1, 2)", lambda d: stats.laplace.cdf(d.q, 1, 2)),
    ("CDF.CAUCHY(q, 1, 2)", lambda d: stats.cauchy.cdf(d.q, 1, 2)),
    ("CDF.UNIFORM(q, 0, 5)", lambda d: stats.uniform.cdf(d.q, 0, 5)),
    ("IDF.UNIFORM(p, 2, 6)", lambda d: 2 + 4 * d.p),
    ("CDF.PARETO(q + 1, 1, 3)", lambda d: 1 - (1 / (d.q + 1)) ** 3),
    ("CDF.GEOM(k + 1, 0.3)", lambda d: 1 - 0.7 ** (d.k + 1)),
    ("CDF.NEGBIN(k + 2, 2, 0.4)", lambda d: stats.nbinom.cdf(d.k, 2, 0.4)),
    ("CDF.HYPER(k, 20, 6, 8)", lambda d: stats.hypergeom.cdf(d.k, 20, 8, 6)),
    ("CDF.BERNOULLI(k, 0.3)", lambda d: stats.bernoulli.cdf(d.k, 0.3)),
    ("CDF.HALFNRM(q, 0, 1)", lambda d: stats.halfnorm.cdf(d.q)),
    ("CDF.IGAUSS(q, 1, 2)", lambda d: stats.invgauss.cdf(d.q, 0.5, scale=2)),
    ("CDF.SRANGE(q + 2, 3, 12)", lambda d: stats.studentized_range.cdf(d.q + 2, 3, 12)),
    ("NCDF.CHISQ(q, 3, 1.5)", lambda d: stats.ncx2.cdf(d.q, 3, 1.5)),
    ("NCDF.F(q, 2, 10, 1)", lambda d: stats.ncf.cdf(d.q, 2, 10, 1)),
    ("NCDF.T(q, 7, 0.5)", lambda d: stats.nct.cdf(d.q, 7, 0.5)),
    ("NPDF.T(q, 7, 0.5)", lambda d: stats.nct.pdf(d.q, 7, 0.5)),
    ("CDF.NORMAL(q, 1, 2)", lambda d: stats.norm.cdf(d.q, 1, 2)),
    ("IDF.NORMAL(p, 1, 2)", lambda d: stats.norm.ppf(d.p, 1, 2)),
    ("LNGAMMA(q)", lambda d: np.log(np.abs([__import__("math").gamma(v) for v in d.q]))),
])
def test_distribution_functions_match_scipy(ds, expr, ref):
    got = evaluate(ds, expr).to_numpy(float)
    want = np.asarray(ref(ds.df), dtype=float)
    np.testing.assert_allclose(got, want, rtol=1e-8, atol=1e-12)


def test_noncentral_beta_reduces_to_beta():
    d = Dataset(pd.DataFrame({"x": [0.2, 0.5, 0.8]}))
    np.testing.assert_allclose(evaluate(d, "NCDF.BETA(x, 2, 3, 0)"),
                               stats.beta.cdf([0.2, 0.5, 0.8], 2, 3), rtol=1e-10)


def test_bivariate_normal():
    d = Dataset(pd.DataFrame({"a": [0.0, 1.0], "b": [0.0, -0.5]}))
    got = evaluate(d, "CDF.BVNOR(a, b, 0.4)")
    want = [stats.multivariate_normal([0, 0], [[1, .4], [.4, 1]]).cdf([a, b])
            for a, b in [(0, 0), (1, -.5)]]
    np.testing.assert_allclose(got, want, atol=1e-6)


def test_invalid_parameters_are_missing(ds):
    assert evaluate(ds, "IDF.NORMAL(p + 1, 0, 1)").isna().all()
    assert evaluate(ds, "CDF.NORMAL(q, 0, -1)").isna().all()


def test_random_numbers_are_reproducible_with_a_seed(ds):
    X.set_seed(2024)
    a = evaluate(ds, "RV.GAMMA(2, 0.5)")
    X.set_seed(2024)
    b = evaluate(ds, "RV.GAMMA(2, 0.5)")
    assert a.equals(b) and (a > 0).all()
    X.set_seed(1)
    big = Dataset(pd.DataFrame({"z": np.zeros(4000)}))
    assert abs(evaluate(big, "RV.EXP(2)").mean() - 0.5) < 0.05
    assert abs(evaluate(big, "RV.NEGBIN(3, 0.5)").mean() - 6) < 0.3  # trials, mean r/p
    assert set(evaluate(big, "RV.BERNOULLI(0.3)").unique()) <= {0.0, 1.0}
    X.set_seed(None)


def _spss(y, m, d):
    return (date(y, m, d) - date(1582, 10, 14)).days * 86400.0


def test_date_creation_and_extraction(ds):
    got = evaluate(ds, "DATE.DMY(d, m, y)")
    want = [_spss(2000, 1, 1), _spss(2020, 2, 29), _spss(1999, 12, 31), _spss(2024, 2, 29)]
    np.testing.assert_allclose(got, want)
    assert evaluate(ds, "DATE.DMY(30, 2, 2023)").isna().all()
    np.testing.assert_allclose(evaluate(ds, "XDATE.YEAR(DATE.DMY(d, m, y))"), ds.df.y)
    np.testing.assert_allclose(evaluate(ds, "XDATE.MONTH(DATE.MDY(m, d, y))"), ds.df.m)
    np.testing.assert_allclose(evaluate(ds, "XDATE.MDAY(DATE.DMY(d, m, y))"), ds.df.d)
    jday = [date(2000, 1, 1).timetuple().tm_yday, date(2020, 2, 29).timetuple().tm_yday,
            365, date(2024, 2, 29).timetuple().tm_yday]
    np.testing.assert_allclose(evaluate(ds, "XDATE.JDAY(DATE.DMY(d, m, y))"), jday)
    # SPSS weekday: 1 = Sunday
    wk = [(date(*v).isoweekday() % 7) + 1 for v in
          [(2000, 1, 1), (2020, 2, 29), (1999, 12, 31), (2024, 2, 29)]]
    np.testing.assert_allclose(evaluate(ds, "XDATE.WKDAY(DATE.DMY(d, m, y))"), wk)
    np.testing.assert_allclose(evaluate(ds, "XDATE.QUARTER(DATE.DMY(d, m, y))"),
                               [1, 1, 4, 1])
    t = evaluate(ds, "DATE.DMY(1, 1, 2000) + TIME.HMS(13, 45, 30)")
    assert evaluate(Dataset(pd.DataFrame({"t": t})), "XDATE.HOUR(t)").iloc[0] == 13
    assert evaluate(Dataset(pd.DataFrame({"t": t})), "XDATE.MINUTE(t)").iloc[0] == 45
    assert evaluate(Dataset(pd.DataFrame({"t": t})), "XDATE.SECOND(t)").iloc[0] == 30
    assert evaluate(ds, "YRMODA(1582, 10, 15)").iloc[0] == 1
    assert evaluate(ds, "DATE.QYR(3, 2021)").iloc[0] == _spss(2021, 7, 1)
    assert evaluate(ds, "DATE.MOYR(13, 2021)").iloc[0] == _spss(2022, 1, 1)
    assert evaluate(ds, "DATE.YRDAY(2021, 60)").iloc[0] == _spss(2021, 3, 1)


def test_date_arithmetic(ds):
    e = Dataset(pd.DataFrame({"a": [_spss(2000, 1, 31), _spss(2000, 3, 15)],
                              "b": [_spss(2001, 1, 30), _spss(1999, 3, 16)]}))
    np.testing.assert_allclose(evaluate(e, 'DATEDIFF(b, a, "days")'),
                               [(date(2001, 1, 30) - date(2000, 1, 31)).days,
                                (date(1999, 3, 16) - date(2000, 3, 15)).days])
    np.testing.assert_allclose(evaluate(e, 'DATEDIFF(b, a, "months")'), [11, -11])
    np.testing.assert_allclose(evaluate(e, 'DATEDIFF(b, a, "years")'), [0, 0])
    got = evaluate(e, 'DATESUM(a, 1, "months")')
    np.testing.assert_allclose(got, [_spss(2000, 2, 29), _spss(2000, 4, 15)])
    got = evaluate(e, 'DATESUM(a, 1, "months", "rollover")')
    assert got.iloc[0] == _spss(2000, 3, 2)
    np.testing.assert_allclose(evaluate(e, 'DATESUM(a, 10, "days")'),
                               [_spss(2000, 2, 10), _spss(2000, 3, 25)])
    assert evaluate(e, "CTIME.DAYS(b - a)").iloc[0] == 365
    with pytest.raises(ExpressionError):
        evaluate(e, 'DATEDIFF(b, a, "fortnights")')


def test_date_variables_use_spss_numbers():
    d = Dataset(pd.DataFrame({"t": pd.to_datetime(["2024-08-17", "1945-08-17"])}))
    np.testing.assert_allclose(evaluate(d, "XDATE.YEAR(t)"), [2024, 1945])
    np.testing.assert_allclose(evaluate(d, "t"), [_spss(2024, 8, 17), _spss(1945, 8, 17)])
    back = X.to_timestamps(evaluate(d, "t"))
    assert list(back.dt.strftime("%Y-%m-%d")) == ["2024-08-17", "1945-08-17"]


def test_string_functions(ds):
    assert list(evaluate(ds, "CHAR.INDEX(s, 'a')")) == [2, 4, 4, 2]
    assert list(evaluate(ds, "CHAR.RINDEX(s, 'a')")) == [5, 4, 6, 7]
    assert evaluate(ds, "CHAR.INDEX(s, 'x')").iloc[0] == 0
    assert list(evaluate(ds, "CHAR.SUBSTR(s, 1, 5)")) == ["Banda", "Medan", "  Pad", "Jakar"]
    assert evaluate(ds, "CHAR.SUBSTR(s, 7)").iloc[0] == "Aceh"
    assert evaluate(ds, "CHAR.LPAD(s, 8, '*')").iloc[1] == "***Medan"
    assert evaluate(ds, "CHAR.RPAD(s, 7, '.')").iloc[1] == "Medan.."
    assert evaluate(ds, "LTRIM(s)").iloc[2] == "Padang"
    assert evaluate(ds, "RTRIM(s)").iloc[3] == "Jakarta"
    assert evaluate(ds, "REPLACE(s, 'a', 'o')").iloc[1] == "Medon"
    assert evaluate(ds, "REPLACE(s, 'a', 'o', 1)").iloc[0] == "Bonda Aceh"
    assert evaluate(ds, "STRUNC(s, 5)").iloc[0] == "Banda"
    assert evaluate(ds, "UPCASE(CHAR.SUBSTR(s, 1, 3))").iloc[1] == "MED"
    assert list(evaluate(ds, "VALUELABEL(g)")) == ["Laki-laki", "Perempuan", "Laki-laki", ""]


def test_conversion_with_spss_formats(ds):
    assert evaluate(ds, "STRING(q, F8.2)").iloc[2] == "2.50"
    assert evaluate(ds, "STRING(k, N3)").iloc[3] == "005"
    assert evaluate(ds, "STRING(DATE.DMY(17, 8, 1945), DATE11)").iloc[0] == "17-AUG-1945"
    assert evaluate(ds, "STRING(DATE.DMY(17, 8, 1945), ADATE10)").iloc[0] == "08/17/1945"
    assert evaluate(ds, "NUMBER('1,234.5', COMMA9.1)").iloc[0] == 1234.5
    assert evaluate(ds, "NUMBER('17-AUG-1945', DATE11)").iloc[0] == _spss(1945, 8, 17)
    assert evaluate(ds, "NUMBER('12.5')").iloc[0] == 12.5


def test_statistical_and_misc_functions(ds):
    want = ds.df[["q", "k"]].std(axis=1, ddof=1) / ds.df[["q", "k"]].mean(axis=1)
    np.testing.assert_allclose(evaluate(ds, "CFVAR(q, k)"), want)
    assert list(evaluate(ds, "$CASENUM")) == [1, 2, 3, 4]
    assert evaluate(ds, "LAG(q)").isna().iloc[0]
    np.testing.assert_allclose(evaluate(ds, "LAG(q)").iloc[1:], ds.df.q.iloc[:-1])
    np.testing.assert_allclose(evaluate(ds, "LAG(q, 2)").iloc[2:], ds.df.q.iloc[:2])
    assert evaluate(ds, "TRUNC(q, 0.5)").iloc[2] == 2.5
    assert evaluate(ds, "$SYSMIS + 1").isna().all()
    assert evaluate(ds, "$JDATE").iloc[0] > 150000
    with pytest.raises(ExpressionError):
        evaluate(ds, "$NOPE")
