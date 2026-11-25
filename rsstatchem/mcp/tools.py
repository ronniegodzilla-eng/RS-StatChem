"""The connector's tools as plain Python: open data, look at variables, run analyses, save.

A :class:`Workspace` holds the datasets opened during one server session by name; every public
method listed in :data:`TOOLS` is one MCP tool (the docstring is what the model reads, so it
is written for the model). Nothing here imports Qt or the MCP SDK, so the tools are tested by
calling them directly. Analyses come from ``rsstatchem.core.stats`` and honour Select Cases,
Split File and Weight Cases stored in the dataset exactly as the Statistics module does.

Problems the user (or the model) can fix raise ``ValueError`` with a message that says how.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from rsstatchem.core import selection
from rsstatchem.core.dataset import NUMERIC, Dataset, format_code, sanitize_name
from rsstatchem.core.output import Section, Table
from rsstatchem.io import files
from rsstatchem.mcp.render import render

OutputFormat = Literal["markdown", "json"]
DescriptiveStat = Literal["mean", "sum", "std", "var", "range", "min", "max", "sem", "kurt",
                          "skew"]
FrequencyStat = Literal["quartiles", "mean", "median", "mode", "sum", "std", "var", "range",
                        "min", "max", "sem", "skew", "kurt"]
PostHoc = Literal["tukey", "bonferroni", "lsd", "scheffe", "sidak", "games_howell"]
CorrelationMethod = Literal["pearson", "spearman", "kendall"]
RegressionMethod = Literal["enter", "stepwise", "forward", "backward"]
Alternative = Literal["two-sided", "greater", "less"]

SAVE_TYPES = (".rsd", ".sav", ".csv", ".xlsx")
_MAX_LISTED = 60


@dataclass(frozen=True)
class ToolInfo:
    name: str
    title: str
    read_only: bool = True      # False: writes a file on the user's computer


TOOLS: list[ToolInfo] = []


def _tool(title: str, read_only: bool = True):
    def register(fn):
        TOOLS.append(ToolInfo(fn.__name__, title, read_only))
        return fn
    return register


def _resolve(path: str) -> Path:
    text = (path or "").strip().strip("\"'")
    if not text:
        raise ValueError("Give the full path of the file.")
    out = Path(text).expanduser()
    if not out.is_absolute():
        raise ValueError(
            f"'{text}' is not a full path. Give the complete location of the file, for example "
            "/Users/name/Documents/data.sav or C:\\Users\\name\\Documents\\data.sav.")
    return out


def _level(level: float) -> float:
    value = float(level)
    if 1 < value < 100:            # 95 meant as 95 %
        value /= 100
    if not 0.5 <= value < 1:
        raise ValueError("confidence_level must be between 0.5 and 1 (for example 0.95).")
    return value


def _section(result) -> Section:
    return result.section if hasattr(result, "section") else result


class Workspace:
    """The datasets of one connector session."""

    def __init__(self) -> None:
        self.datasets: dict[str, Dataset] = {}
        self.active: str | None = None
        # the SDK runs synchronous tools in worker threads; analyses are not re-entrant
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ helpers
    def dataset(self, name: str = "") -> Dataset:
        if not self.datasets:
            raise ValueError("No dataset is open. Open a data file with open_data first.")
        key = (name or "").strip() or self.active
        for known, ds in self.datasets.items():
            if known.lower() == str(key).lower():
                return ds
        raise ValueError(f"There is no open dataset named '{name}'. Open datasets: "
                         + ", ".join(self.datasets) + ".")

    def _register(self, ds: Dataset, name: str, source: Path) -> str:
        base = sanitize_name(name or source.stem, fallback="data")
        key, k = base, 1
        while True:
            taken = next((n for n in self.datasets if n.lower() == key.lower()), None)
            if taken is None:
                break
            if name or getattr(self.datasets[taken], "source", None) == str(source):
                del self.datasets[taken]     # asked for by name, or the same file reopened
                break
            k += 1
            key = f"{base}_{k}"
        ds.name = key
        ds.source = str(source)
        self.datasets[key] = ds
        self.active = key
        return key

    @staticmethod
    def _names(ds: Dataset, names, what: str = "variables", minimum: int = 1) -> list[str]:
        if isinstance(names, str):
            names = [names]
        out = []
        for raw in names or []:
            text = str(raw).strip()
            if not ds.has_name(text):
                shown = ds.names()[:_MAX_LISTED]
                more = "" if ds.n_vars <= _MAX_LISTED else f" … ({ds.n_vars} in total)"
                raise ValueError(f"There is no variable named '{text}' in dataset "
                                 f"'{ds.name}'. Variables: {', '.join(shown)}{more}.")
            canonical = ds.var(text).name
            if canonical not in out:
                out.append(canonical)
        if len(out) < minimum:
            raise ValueError(f"Give at least {minimum} variable name(s) in {what}.")
        return out

    @classmethod
    def _numeric(cls, ds: Dataset, names, what: str = "variables", minimum: int = 1
                 ) -> list[str]:
        out = cls._names(ds, names, what, minimum)
        for name in out:
            if ds.var(name).type != NUMERIC:
                raise ValueError(f"'{name}' is a {ds.var(name).type} variable; {what} must be "
                                 "numeric for this analysis.")
        return out

    @staticmethod
    def _run(ds: Dataset, fn, output_format: str) -> str:
        """Run ``fn(dataset)`` on the selected / weighted / split view(s), as the Statistics
        module does for its dialogs."""
        views = selection.analysis_views(ds)
        if len(views) == 1 and not selection.describe(ds):
            section = _section(fn(ds))
        else:
            section = selection.combine([(v.label, _section(fn(v.ds))) for v in views], ds)
        return render(section, output_format)

    @staticmethod
    def _group_value(ds: Dataset, group_variable: str, value):
        """A group given as a code (1, "1") or as its value label ("Kontrol")."""
        var = ds.var(group_variable)
        text = str(value).strip()
        for code, label in var.value_labels.items():
            if str(label).strip().lower() == text.lower():
                return code
        if var.type != NUMERIC:
            return text
        try:
            return float(text.replace(",", "."))
        except ValueError:
            labels = ", ".join(f"{format_code(c)} = {lb}" for c, lb in var.value_labels.items())
            raise ValueError(f"'{text}' is not a value of '{var.name}'"
                             + (f" ({labels})." if labels else "; give the numeric code.")
                             ) from None

    def _variables_section(self, ds: Dataset) -> Section:
        section = Section(f"Dataset {ds.name}")
        info = [f"{ds.n_cases} cases, {ds.n_vars} variables"]
        if getattr(ds, "source", None):
            info.append(f"file: {ds.source}")
        info += selection.describe(ds)
        section.note("", " · ".join(info))
        rows = []
        for var in ds.variables:
            labels = "; ".join(f"{format_code(c)} = {lb}" for c, lb in var.value_labels.items())
            missing = ", ".join(format_code(m) for m in var.missing)
            valid = int(ds.analysis_series(var.name).notna().sum())
            rows.append([var.name, var.type, var.measure, var.label, labels, missing,
                         str(valid)])
        section.table(Table("Variables", ["Name", "Type", "Measure", "Label", "Value labels",
                                          "User-missing values", "Valid N"], rows))
        return section

    # ------------------------------------------------------------------ data
    @_tool("Open a data file")
    def open_data(self, path: str, name: str = "", sheet: str = "",
                  output_format: OutputFormat = "markdown") -> str:
        """Open a data file from this computer into the session and list its variables.

        Reads RS-StatChem data (.rsd), SPSS (.sav, .por), Excel (.xlsx), text (.csv, .txt,
        .tsv; delimiter and decimal comma are detected), Stata (.dta) and SAS (.sas7bdat,
        .xpt). The file itself is not changed. The dataset becomes the active one: other tools
        use it when their `dataset` argument is left empty.

        Args:
            path: Full path of the file (a leading ~ is accepted). Ask the user if unknown.
            name: Optional short name for the dataset; default is the file name.
            sheet: Excel only: worksheet name; default is the first sheet.
        """
        with self._lock:
            source = _resolve(path)
            if not source.is_file():
                raise ValueError(f"File not found: {source}")
            kind = files.kind_of(source)
            extra = ""
            if kind == "project":
                ds = files.load_project(source)
            elif kind == "sav":
                ds = files.read_sav(source)
            elif kind == "por":
                ds = files.read_por(source)
            elif kind == "stata":
                ds = files.read_stata(source)
            elif kind == "sas":
                ds = files.read_sas(source)
            elif kind == "text":
                encoding = files.detect_encoding(source)
                delimiter, decimal = files.sniff_text(source, encoding)
                ds = Dataset(files.read_text(source, delimiter, decimal, encoding=encoding))
            elif kind == "excel":
                sheets = files.excel_sheets(source)
                chosen = sheet.strip() or sheets[0]
                match = next((s for s in sheets if s.lower() == chosen.lower()), None)
                if match is None:
                    raise ValueError(f"No worksheet named '{chosen}'. Worksheets: "
                                     + ", ".join(sheets) + ".")
                ds = Dataset(files.read_excel(source, match))
                if len(sheets) > 1:
                    extra = f"Worksheet '{match}' was read; the file has: {', '.join(sheets)}."
            else:
                raise ValueError(f"{source.suffix} files hold no data table; open a data "
                                 "file (.rsd, .sav, .xlsx, .csv).")
            self._register(ds, name.strip(), source)
            section = self._variables_section(ds)
            if extra:
                section.note("", extra)
            return render(section, output_format)

    @_tool("List open datasets")
    def list_datasets(self, output_format: OutputFormat = "markdown") -> str:
        """List the datasets opened in this session (name, size, file) and which is active."""
        with self._lock:
            section = Section("Open datasets")
            if not self.datasets:
                section.note("", "No dataset is open. Use open_data with the path of a file.")
            else:
                rows = [[name, "yes" if name == self.active else "", str(ds.n_cases),
                         str(ds.n_vars), getattr(ds, "source", "") or ""]
                        for name, ds in self.datasets.items()]
                section.table(Table("Datasets", ["Name", "Active", "Cases", "Variables",
                                                 "File"], rows))
            return render(section, output_format)

    @_tool("List variables")
    def list_variables(self, dataset: str = "",
                       output_format: OutputFormat = "markdown") -> str:
        """List the variables of a dataset: name, type, measurement level, label, value
        labels (code = meaning), user-missing codes and the number of valid cases. Call this
        before an analysis to pick exact variable names and group codes.

        Args:
            dataset: Dataset name; empty = the active (most recently opened) dataset.
        """
        with self._lock:
            return render(self._variables_section(self.dataset(dataset)), output_format)

    @_tool("Save a dataset", read_only=False)
    def save_data(self, path: str, dataset: str = "", overwrite: bool = False) -> str:
        """Save a dataset to a file on this computer. The type follows the extension: .rsd
        (RS-StatChem, keeps labels and missing values), .sav (SPSS), .xlsx or .csv.

        Args:
            path: Full path of the file to write; its folder must exist.
            dataset: Dataset name; empty = the active dataset.
            overwrite: Must be true to replace an existing file. Ask the user first.
        """
        with self._lock:
            ds = self.dataset(dataset)
            target = _resolve(path)
            ext = target.suffix.lower()
            if ext not in SAVE_TYPES:
                raise ValueError(f"Cannot save as '{ext or '(no extension)'}'. Use one of: "
                                 + ", ".join(SAVE_TYPES) + ".")
            if not target.parent.is_dir():
                raise ValueError(f"The folder does not exist: {target.parent}")
            if target.exists() and not overwrite:
                raise ValueError(f"{target} already exists. Confirm with the user, then call "
                                 "save_data again with overwrite=true, or choose another name.")
            if ext == ".rsd":
                files.save_project(ds, target)
            elif ext == ".sav":
                files.write_sav(ds, target)
            elif ext == ".xlsx":
                files.write_excel(ds, target)
            else:
                files.write_text(ds, target)
            return (f"Saved dataset '{ds.name}' ({ds.n_cases} cases, {ds.n_vars} variables) "
                    f"to {target}")

    # ------------------------------------------------------------------ descriptive
    @_tool("Descriptive statistics")
    def descriptives(self, variables: list[str], statistics: list[DescriptiveStat] | None = None,
                     dataset: str = "", output_format: OutputFormat = "markdown") -> str:
        """Descriptive Statistics table (SPSS DESCRIPTIVES) for numeric variables: N and the
        chosen statistics.

        Args:
            variables: Names of numeric variables.
            statistics: Any of mean, sum, std, var, range, min, max, sem (S.E. of mean),
                kurt, skew. Default: mean, std, min, max.
            dataset: Dataset name; empty = the active dataset.
        """
        from rsstatchem.core.stats import descriptive

        with self._lock:
            ds = self.dataset(dataset)
            names = self._numeric(ds, variables)
            stats = tuple(statistics) if statistics else descriptive.DEFAULT_DESCRIPTIVES
            return self._run(ds, lambda d: descriptive.descriptives(d, names, stats),
                             output_format)

    @_tool("Frequencies")
    def frequencies(self, variables: list[str], statistics: list[FrequencyStat] | None = None,
                    max_categories: int = 50, dataset: str = "",
                    output_format: OutputFormat = "markdown") -> str:
        """Frequency tables (SPSS FREQUENCIES): count, percent, valid percent and cumulative
        percent of every distinct value, for numeric or string variables. Meant for
        categorical variables: the table lists the values themselves, so a variable with more
        than `max_categories` distinct values (a score, a name, an ID) gets no table, only the
        requested statistics.

        Args:
            variables: Variable names.
            statistics: Optional summary statistics: quartiles, mean, median, mode, sum, std,
                var, range, min, max, sem, skew, kurt.
            max_categories: Largest number of distinct values a variable may have to get a
                frequency table (default 50).
            dataset: Dataset name; empty = the active dataset.
        """
        from rsstatchem.core.stats import descriptive

        with self._lock:
            ds = self.dataset(dataset)
            names = self._names(ds, variables)
            stats = tuple(statistics or ())
            limit = max(int(max_categories), 1)
            skipped = [n for n in names if ds.analysis_series(n).nunique() > limit]
            text = self._run(ds, lambda d: descriptive.frequencies(d, names, stats, limit=limit),
                             output_format)
            if skipped and output_format == "markdown":
                text += (f"\n\nNo frequency table for {', '.join(skipped)}: more than {limit} "
                         "distinct values (raise max_categories to list them).")
            return text

    # ------------------------------------------------------------------ means
    @_tool("One-sample t test")
    def one_sample_t_test(self, variables: list[str], test_value: float = 0.0,
                          confidence_level: float = 0.95, dataset: str = "",
                          output_format: OutputFormat = "markdown") -> str:
        """One-Sample T Test: is the mean of each variable different from `test_value`?
        Gives statistics, t, df, one- and two-sided p, the confidence interval of the
        difference and effect sizes (Cohen's d, Hedges' correction).

        Args:
            variables: Names of numeric test variables.
            test_value: The hypothesised population mean.
            confidence_level: For the interval, e.g. 0.95.
            dataset: Dataset name; empty = the active dataset.
        """
        from rsstatchem.core.stats import ttest

        with self._lock:
            ds = self.dataset(dataset)
            names = self._numeric(ds, variables)
            level = _level(confidence_level)
            return self._run(ds, lambda d: ttest.one_sample(d, names, float(test_value), level),
                             output_format)

    @_tool("One-sample Z test")
    def one_sample_z_test(self, variables: list[str], sigma: float, test_value: float = 0.0,
                          alternative: Alternative = "two-sided",
                          confidence_level: float = 0.95, dataset: str = "",
                          output_format: OutputFormat = "markdown") -> str:
        """One-Sample Z Test of a mean when the population standard deviation is known.

        Args:
            variables: Names of numeric test variables.
            sigma: Known population standard deviation (> 0).
            test_value: The hypothesised population mean.
            alternative: two-sided, greater (mean > test_value) or less.
            confidence_level: For the interval or bound, e.g. 0.95.
            dataset: Dataset name; empty = the active dataset.
        """
        from rsstatchem.core.stats import ztest

        with self._lock:
            ds = self.dataset(dataset)
            names = self._numeric(ds, variables)
            level = _level(confidence_level)
            return self._run(ds, lambda d: ztest.one_sample(
                d, names, float(sigma), float(test_value), level, alternative), output_format)

    @_tool("Independent-samples t test")
    def independent_samples_t_test(self, variables: list[str], group_variable: str,
                                   group1: str = "", group2: str = "",
                                   cut_point: float | None = None,
                                   confidence_level: float = 0.95, dataset: str = "",
                                   output_format: OutputFormat = "markdown") -> str:
        """Independent-Samples T Test: compare the means of two groups. Gives group
        statistics, Levene's test, t tests with and without equal variances assumed, the
        confidence interval of the difference and effect sizes.

        Args:
            variables: Names of numeric test variables.
            group_variable: The variable that defines the two groups.
            group1: Code or value label of the first group (e.g. "1" or "Kontrol").
            group2: Code or value label of the second group. Leave both empty when the
                grouping variable has exactly two values.
            cut_point: Alternative to group1/group2 for a numeric grouping variable: cases
                with a value >= cut_point form one group, the rest the other.
            confidence_level: For the interval, e.g. 0.95.
            dataset: Dataset name; empty = the active dataset.
        """
        from rsstatchem.core.stats import ttest

        with self._lock:
            ds = self.dataset(dataset)
            names = self._numeric(ds, variables)
            gname = self._names(ds, [group_variable], "group_variable")[0]
            level = _level(confidence_level)
            groups = None
            if cut_point is not None:
                if ds.var(gname).type != NUMERIC:
                    raise ValueError("cut_point needs a numeric grouping variable.")
            elif str(group1).strip() and str(group2).strip():
                groups = (self._group_value(ds, gname, group1),
                          self._group_value(ds, gname, group2))
            else:
                values = sorted(pd.unique(ds.analysis_series(gname).dropna()).tolist())
                if len(values) != 2:
                    shown = ", ".join(format_code(v) for v in values[:20])
                    raise ValueError(f"'{gname}' has {len(values)} distinct values ({shown}"
                                     f"{' …' if len(values) > 20 else ''}); give group1 and "
                                     "group2, or a cut_point.")
                groups = tuple(values)
            return self._run(ds, lambda d: ttest.independent(
                d, names, gname, groups=groups, cut=cut_point, level=level), output_format)

    @_tool("Paired-samples t test")
    def paired_samples_t_test(self, pairs: list[list[str]], confidence_level: float = 0.95,
                              dataset: str = "",
                              output_format: OutputFormat = "markdown") -> str:
        """Paired-Samples T Test: compare two measurements of the same cases (e.g. pre-test
        and post-test). Gives paired statistics, correlations, the t test of the mean
        difference with its confidence interval, and effect sizes.

        Args:
            pairs: List of [variable1, variable2] pairs, e.g. [["pretest", "posttest"]].
            confidence_level: For the interval, e.g. 0.95.
            dataset: Dataset name; empty = the active dataset.
        """
        from rsstatchem.core.stats import ttest

        with self._lock:
            ds = self.dataset(dataset)
            checked = []
            for pair in pairs or []:
                if isinstance(pair, str) or len(pair) != 2:
                    raise ValueError("Each pair must name exactly two variables, e.g. "
                                     '[["pretest", "posttest"]].')
                a, b = (self._numeric(ds, [p], "paired variables")[0] for p in pair)
                if a == b:
                    raise ValueError(f"A pair needs two different variables (got '{a}' twice).")
                checked.append((a, b))
            if not checked:
                raise ValueError("Give at least one pair of variables.")
            level = _level(confidence_level)
            return self._run(ds, lambda d: ttest.paired(d, checked, level), output_format)

    @_tool("One-way ANOVA")
    def one_way_anova(self, dependents: list[str], factor: str,
                      post_hoc: list[PostHoc] | None = None, welch: bool = False,
                      brown_forsythe: bool = False, confidence_level: float = 0.95,
                      dataset: str = "", output_format: OutputFormat = "markdown") -> str:
        """One-Way ANOVA: compare the means of three or more groups. Gives descriptives,
        Levene's test of homogeneity of variances, the ANOVA table and effect sizes
        (eta-squared, omega-squared); optionally robust tests and post hoc comparisons.

        Args:
            dependents: Names of numeric dependent variables (one ANOVA each).
            factor: The grouping variable.
            post_hoc: Multiple comparisons: tukey, bonferroni, lsd, scheffe, sidak (equal
                variances assumed) or games_howell (not assumed).
            welch: Add Welch's robust test of equality of means.
            brown_forsythe: Add the Brown-Forsythe robust test.
            confidence_level: For intervals and post hoc tests, e.g. 0.95.
            dataset: Dataset name; empty = the active dataset.
        """
        from rsstatchem.core.stats import anova

        with self._lock:
            ds = self.dataset(dataset)
            names = self._numeric(ds, dependents, "dependents")
            fname = self._names(ds, [factor], "factor")[0]
            level = _level(confidence_level)
            return self._run(ds, lambda d: anova.oneway(
                d, names, fname, welch=welch, brown_forsythe=brown_forsythe,
                posthoc_methods=tuple(post_hoc or ()), level=level), output_format)

    # ------------------------------------------------------------------ association
    @_tool("Correlations")
    def correlations(self, variables: list[str],
                     methods: list[CorrelationMethod] | None = None, two_tailed: bool = True,
                     dataset: str = "", output_format: OutputFormat = "markdown") -> str:
        """Bivariate correlation matrix with significance and N (pairwise deletion of
        missing values); * marks p < .05 and ** p < .01.

        Args:
            variables: Names of two or more numeric variables.
            methods: pearson (default), spearman and/or kendall (tau-b).
            two_tailed: False for one-tailed significance.
            dataset: Dataset name; empty = the active dataset.
        """
        from rsstatchem.core.stats import correlate

        with self._lock:
            ds = self.dataset(dataset)
            names = self._numeric(ds, variables, minimum=2)
            chosen = set(methods or ("pearson",))
            return self._run(ds, lambda d: correlate.bivariate(
                d, names, pearson="pearson" in chosen, kendall="kendall" in chosen,
                spearman="spearman" in chosen, two_tailed=two_tailed), output_format)

    @_tool("Linear regression")
    def linear_regression(self, dependent: str, independents: list[str],
                          method: RegressionMethod = "enter", assumption_tests: bool = True,
                          confidence_level: float = 0.95, dataset: str = "",
                          output_format: OutputFormat = "markdown") -> str:
        """Linear regression (simple or multiple, SPSS REGRESSION): Model Summary (R, R
        Square, adjusted, Durbin-Watson), ANOVA, and Coefficients (B, Std. Error, Beta, t,
        Sig., collinearity tolerance and VIF). Cases with a missing value on any variable
        are left out.

        Args:
            dependent: Name of the numeric dependent variable.
            independents: Names of the numeric independent variables (predictors). A
                categorical predictor must already be coded as numbers (0/1 dummies).
            method: enter (all predictors), stepwise, forward or backward selection.
            assumption_tests: Also report the classical assumption tests (normality of
                residuals, multicollinearity, heteroscedasticity, autocorrelation).
            confidence_level: For confidence intervals, e.g. 0.95.
            dataset: Dataset name; empty = the active dataset.
        """
        from rsstatchem.core.stats import regression

        with self._lock:
            ds = self.dataset(dataset)
            dep = self._numeric(ds, [dependent], "dependent")[0]
            indep = self._numeric(ds, independents, "independents")
            if dep in indep:
                raise ValueError(f"'{dep}' cannot be both the dependent and an independent "
                                 "variable.")
            level = _level(confidence_level)
            tests = regression.ASSUMPTIONS if assumption_tests else ()
            return self._run(ds, lambda d: regression.linear(
                d, dep, indep, method=method, plots=(), assumptions=tests, level=level),
                output_format)
