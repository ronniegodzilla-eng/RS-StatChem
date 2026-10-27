"""Merges the feature modules' ``MENUS`` into the Data Editor's menu tree.

Each extension module exports ``MENUS = {"Analyze/Regression": [(text, DialogClass |
callable), None, (submenu, [...])]}``. Entries replace a disabled placeholder with the same
text, or are appended. A module that fails to import is skipped (and reported on stderr) so
one broken feature never takes the whole app down.

After merging, :data:`MENU_ORDER` puts every listed menu in SPSS order (explicit lists, not
module load order). Items a module adds that are not listed yet are appended at the end of
their menu, so a new work package only needs its labels added to the list to land in place.
"""
from __future__ import annotations

import importlib
import sys
import traceback

EXTENSIONS = (
    "rsstatchem.ui.dialogs.data_ops",
    "rsstatchem.ui.dialogs.logistic",
    "rsstatchem.ui.dialogs.glm",
    "rsstatchem.ui.dialogs.genlin",
    "rsstatchem.ui.dialogs.regression_ext",
    "rsstatchem.ui.dialogs.fitting",
    "rsstatchem.ui.dialogs.doe",
    "rsstatchem.ui.graph",
    "rsstatchem.ui.dialogs.sem",
    "rsstatchem.ui.dialogs.pls",
    "rsstatchem.ui.chem_editor",  # before ui.chem so the editors head the Chemistry menu
    "rsstatchem.ui.chem",
    "rsstatchem.ui.dialogs.survival",
    "rsstatchem.ui.dialogs.missing",
    "rsstatchem.ui.dialogs.timeseries",
    "rsstatchem.ui.dialogs.agreement",
    "rsstatchem.ui.dialogs.nptests",
    "rsstatchem.ui.dialogs.distplots",
    "rsstatchem.ui.dialogs.data_props",
    "rsstatchem.ui.dialogs.mresponse",
    "rsstatchem.ui.dialogs.ctables",
    "rsstatchem.ui.dialogs.reports",
)
# Cross-links: an entry of one module that should also appear in another menu.
# (module, attribute, menu path, menu text)
ALIASES = (
)
# Top-level menus whose placeholder items are dropped once a module provides real entries.
REPLACEABLE = {"Graphs", "RSM", "Chemistry"}
OBSOLETE_PLACEHOLDERS = {"process-style model"}

# SPSS 30/31 menu order (labels compared case-insensitively without "..." and "&"; None is a
# separator). Labels of work packages that are not merged yet (Power Analysis, Meta Analysis,
# Bayesian Statistics, Proportions, Variance Components, Correspondence Analysis, MDS, ...) are
# already listed so their entries land in the SPSS position as soon as the module is added to
# EXTENSIONS. Unlisted entries are appended after the listed ones.
MENU_ORDER: dict[str, list] = {
    "Data": [
        "Define Variable Properties...", "Set Measurement Level for Unknown...",
        "Copy Data Properties...", "New Custom Attribute...", "Define Date and Time...",
        "Define Multiple Response Sets...", None,
        "Validation", "Identify Duplicate Cases...", "Identify Unusual Cases...",
        "Compare Datasets...", None,
        "Sort Cases...", "Sort Variables...", "Transpose...", "Adjust String Widths Across Files",
        "Merge Files", "Restructure...", "Rake Weights...", "Propensity Score Matching...",
        "Case Control Matching...", "Aggregate...", "Orthogonal Design", "Split into Files...",
        "Copy Dataset", None,
        "Split File...", "Select Cases...", "Weight Cases...",
    ],
    "Transform": [
        "Compute Variable...", "Programmability Transformation...",
        "Count Values within Cases...", "Shift Values...", None,
        "Recode into Same Variables...", "Recode into Different Variables...",
        "Automatic Recode...", "Create Dummy Variables...", "Visual Binning...",
        "Optimal Binning...", "Prepare Data for Modeling", None,
        "Rank Cases...", "Date and Time Wizard...", "Create Time Series...",
        "Replace Missing Values...", "Random Number Generators...", None,
        "Run Pending Transforms",
    ],
    "Analyze": [
        "Power Analysis", "Meta Analysis", "Reports", "Descriptive Statistics",
        "Bayesian Statistics", "Tables", "Compare Means and Proportions", "Compare Means",
        "General Linear Model", "Generalized Linear Models", "Mixed Models", "Correlate",
        "Regression", "Loglinear", "Mediation && Moderation", "Neural Networks", "Classify",
        "Dimension Reduction", "Scale", "Structural Equation Modeling", "Nonparametric Tests",
        "Forecasting", "Survival", "Multiple Response", None,
        "Missing Value Analysis...", "Multiple Imputation", "Complex Samples", "Simulation...",
        "Quality Control", "Spatial and Temporal Modeling", "Direct Marketing",
    ],
    "Analyze/Reports": [
        "Codebook...", "OLAP Cubes...", "Case Summaries...",
        "Report Summaries in Rows...", "Report Summaries in Columns...",
    ],
    "Analyze/Descriptive Statistics": [
        "FREQ", "DESC", "Explore...", "Crosstabs...", "TURF Analysis...", "Percentiles...",
        "Ratio...", "P-P Plots...", "Q-Q Plots...", "Normality Analysis...",
    ],
    "Analyze/Tables": ["Custom Tables...", "Multiple Response Sets..."],
    "Analyze/Compare Means": [
        "Means...", "One-Sample T Test...", "Independent-Samples T Test...",
        "Summary Independent-Samples T Test...", "Paired-Samples T Test...",
        "One-Way ANOVA...", None, "One-Sample Proportions...",
        "Independent-Samples Proportions...", "Paired-Samples Proportions...",
    ],
    "Analyze/General Linear Model": [
        "Univariate...", "Multivariate...", "Repeated Measures...", "Variance Components...",
    ],
    "Analyze/Generalized Linear Models": [
        "Generalized Linear Models...", "Generalized Estimating Equations...",
        "Multiple Adaptive Regression Splines...",
    ],
    "Analyze/Mixed Models": ["Linear...", "Generalized Linear..."],
    "Analyze/Correlate": ["Bivariate...", "Partial...", "Distances...",
                          "Canonical Correlation..."],
    "Analyze/Regression": [
        "Automatic Linear Modeling...", "Linear...", "Curve Estimation...",
        "Partial Least Squares...", None, "Binary Logistic...", "Multinomial Logistic...",
        "Ordinal...", "Probit...", None, "Nonlinear...", "Weight Estimation...",
        "2-Stage Least Squares...", "Quantile...", None, "Optimal Scaling (CATREG)...",
        "Linear OLS Alternatives",
    ],
    "Analyze/Loglinear": ["General...", "Logit...", "Model Selection..."],
    "Analyze/Classify": [
        "TwoStep Cluster...", "K-Means Cluster...", "Hierarchical Cluster...",
        "Cluster Silhouettes...", "Tree...", "Discriminant...", "Nearest Neighbor...", None,
        "ROC Curve...", "ROC Analysis...",
    ],
    "Analyze/Dimension Reduction": ["Factor...", "Correspondence Analysis...",
                                    "Optimal Scaling..."],
    "Analyze/Scale": [
        "Reliability Analysis...", "Weighted Kappa...", "Bland-Altman...",
        "Multidimensional Unfolding (PREFSCAL)...", "Multidimensional Scaling (PROXSCAL)...",
        "Multidimensional Scaling (ALSCAL)...",
    ],
    "Analyze/Nonparametric Tests": [
        "One Sample...", "Independent Samples...", "Related Samples...", None,
        "Legacy Dialogs",
    ],
    "Analyze/Survival": [
        "Life Tables...", "Kaplan-Meier...", "Cox Regression...", "Cox w/ Time-Dep Cov...",
        "Parametric Accelerated Failure Time (AFT)...", "Parametric Shared Frailty...",
    ],
    "Analyze/Multiple Response": ["Define Variable Sets...", "Frequencies...",
                                  "Crosstabs..."],
    "Graphs": ["Chart Builder...", "Graphboard Template Chooser...", "Weibull Plot...",
               "Compare Subgroups...", "Regression Variable Plots...", "Legacy Dialogs", None,
               "Graph Studio (Plot Setup)..."],
    "Utilities": [
        "Variables...", "OMS Control Panel...", "OMS Identifiers...", None,
        "Data File Comments...", "Define Variable Sets...", "Use Variable Sets...",
        "Show All Variables", None, "Run Script...",
    ],
}

_loaded: list = []
_failed: dict[str, str] = {}


def _key(text: str) -> str:
    return text.replace("&&", "&").replace("&", "").rstrip(".").strip().lower()


def extension_modules() -> list:
    if not _loaded and not _failed:
        for name in EXTENSIONS:
            try:
                _loaded.append(importlib.import_module(name))
            except Exception:  # noqa: BLE001 - keep the app usable if one module breaks
                # A missing module is a failure too: a packaged build once shipped without half
                # of these and the menus silently fell back to disabled placeholders.
                _failed[name] = traceback.format_exc()
        for name, tb in _failed.items():
            print(f"[RS-StatChem] extension {name} failed to load:\n{tb}", file=sys.stderr)
    return _loaded


def failed_extensions() -> dict[str, str]:
    return dict(_failed)


def _merge_list(target: list, entries: list) -> None:
    for entry in entries:
        if entry is None:
            target.append(None)
            continue
        text, payload = entry
        if isinstance(payload, list):
            sub = next((e[1] for e in target if isinstance(e, tuple) and isinstance(e[1], list)
                        and _key(e[0]) == _key(text)), None)
            if sub is None:
                sub = []
                target.append((text, sub))
            _merge_list(sub, payload)
            continue
        for i, existing in enumerate(target):
            label = existing if isinstance(existing, str) else (
                existing[0] if isinstance(existing, tuple) else None)
            if label is not None and label not in ("FREQ", "DESC") and \
                    _key(label) == _key(text):
                target[i] = entry
                break
        else:
            target.append(entry)
    target[:] = [e for e in target
                 if not (isinstance(e, str) and _key(e) in OBSOLETE_PLACEHOLDERS)]


def build_tree(base: dict[str, list]) -> dict[str, list]:
    tree = {k: _copy(v) for k, v in base.items()}
    replaced: set[str] = set()
    for module in extension_modules():
        for path, entries in getattr(module, "MENUS", {}).items():
            parts = path.split("/")
            top = parts[0]
            target = tree.setdefault(top, [])
            if top in REPLACEABLE and top not in replaced:
                target.clear()  # the built-in lists there are placeholders only
                replaced.add(top)
            for title in parts[1:]:
                sub = next((e[1] for e in target if isinstance(e, tuple)
                            and isinstance(e[1], list) and _key(e[0]) == _key(title)), None)
                if sub is None:
                    sub = []
                    target.append((title, sub))
                target = sub
            _merge_list(target, entries)
    loaded = {m.__name__: m for m in extension_modules()}
    for module, attr, path, text in ALIASES:
        target = getattr(loaded.get(module), attr, None)
        if target is None:
            continue
        node = tree.setdefault(path.split("/")[0], [])
        for title in path.split("/")[1:]:
            node = next((e[1] for e in node if isinstance(e, tuple) and isinstance(e[1], list)
                         and _key(e[0]) == _key(title)), node)
        _merge_list(node, [(text, target)])
    apply_order(tree)
    for entries in tree.values():
        _tidy(entries)
    return tree


def _find_menu(tree: dict, path: str) -> list | None:
    parts = path.split("/")
    node = tree.get(parts[0])
    for title in parts[1:]:
        if node is None:
            return None
        node = next((e[1] for e in node if isinstance(e, tuple) and isinstance(e[1], list)
                     and _key(e[0]) == _key(title)), None)
    return node


def _entry_key(entry) -> str | None:
    if entry is None:
        return None
    return _key(entry if isinstance(entry, str) else entry[0])


def order_entries(entries: list, order: list) -> list:
    """``entries`` rearranged to follow ``order``; unlisted entries go last, in their
    original order, after a separator."""
    by_key: dict[str, object] = {}
    for e in entries:
        k = _entry_key(e)
        if k is not None and k not in by_key:
            by_key[k] = e
    out: list = []
    used: set[str] = set()
    for label in order:
        if label is None:
            out.append(None)
            continue
        k = _key(label)
        if k in by_key and k not in used:
            out.append(by_key[k])
            used.add(k)
    rest = [e for e in entries if e is None or _entry_key(e) not in used]
    if any(e is not None for e in rest):
        out.append(None)
        out.extend(rest)  # their own separators are kept; _tidy collapses doubles
    return out


def apply_order(tree: dict) -> None:
    for path, order in MENU_ORDER.items():
        entries = _find_menu(tree, path)
        if entries:
            entries[:] = order_entries(entries, order)


def _tidy(entries: list) -> None:
    """Drop repeated and trailing separators left behind by replaced placeholders."""
    out = []
    for e in entries:
        if e is None and (not out or out[-1] is None):
            continue
        if isinstance(e, tuple) and isinstance(e[1], list):
            _tidy(e[1])
        out.append(e)
    while out and out[-1] is None:
        out.pop()
    entries[:] = out


def _copy(entries: list) -> list:
    out = []
    for e in entries:
        if isinstance(e, tuple) and isinstance(e[1], list):
            out.append((e[0], _copy(e[1])))
        else:
            out.append(e)
    return out


def selftests() -> list[tuple[str, str | None]]:
    """Run every extension's ``selftest()``; returns (module, error or None)."""
    results = []
    for module in extension_modules():
        fn = getattr(module, "selftest", None)
        if fn is None:
            continue
        try:
            fn()
            results.append((module.__name__, None))
        except Exception:  # noqa: BLE001
            results.append((module.__name__, traceback.format_exc()))
    results += [(name, tb) for name, tb in _failed.items()]
    return results
