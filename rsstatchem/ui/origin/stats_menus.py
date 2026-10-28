"""WP-3 Statistics menu entries for the Origin-style window.

:func:`install` adds Origin's Statistics submenus and entries (names as in Origin 2024–2026):

* Descriptive Statistics: Distribution Fit, Partial Correlation, Cross Tabulation, Grubbs
  Test, Dixon's Q Test
* Hypothesis Testing: One-Sample / Two-Sample Test for Variance, Tests for Equal Variance,
  One-Sample Test for Proportion, Two-Sample Test for Proportions
* ANOVA: Three-Way ANOVA, One-Way / Two-Way Repeated Measures ANOVA, One-Way MANOVA
* Nonparametric Tests: One-Sample Wilcoxon Signed Rank, Paired Sample Sign Test, Two-Sample
  Kolmogorov-Smirnov, Mood's Median Test
* Multivariate Analysis: Principal Component Analysis, K-Means Cluster Analysis,
  Hierarchical Cluster Analysis, Discriminant Analysis, Partial Least Squares
* Survival Analysis: Kaplan-Meier Estimator, Cox Proportional Hazards Model, Weibull Fit
* ROC Curve
* Power and Sample Size (one entry per procedure)

Existing submenus of ``main.menus["statistics"]`` are reused (matched by title), so the
entries join the built-in ones. Every entry opens the analysis dialog and runs it through
``main.run_stat`` (report sheet in the input's workbook, recalculation lock); analyses
without input ranges (proportion tests from summary data, power) write their report into
the active workbook (a new one when none is open).
"""
from __future__ import annotations

from PySide6.QtWidgets import QMenu

from rsstatchem.ui.origin import recalc
from rsstatchem.ui.origin import stat_dialogs as sd
from rsstatchem.ui.origin import stat_tests as st

MENU = [
    ("Descriptive Statistics", [("distfit", "Distribution &Fit..."),
                                ("partialcorr", "&Partial Correlation..."),
                                ("crosstab", "Cross &Tabulation (Chi-Square)..."),
                                None,
                                ("grubbs", "&Grubbs Test..."),
                                ("dixon", "&Dixon's Q Test...")]),
    ("Hypothesis Testing", [None,
                            ("var1", "One-Sample Test for &Variance..."),
                            ("var2", "Two-Sample Test for V&ariance..."),
                            ("vark", "Tests for &Equal Variance..."),
                            None,
                            ("prop1", "One-Sample Test for P&roportion..."),
                            ("prop2", "Two-Sample Test for Pr&oportions...")]),
    ("ANOVA", [None,
               ("anova3", "Th&ree-Way ANOVA..."),
               ("rmanova1", "One-Way Repeated Measures ANOVA..."),
               ("rmanova2", "Two-Way Repeated Measures ANOVA..."),
               ("manova", "One-Way &MANOVA...")]),
    ("Nonparametric Tests", [None,
                             ("wilcoxon1", "One-Sample Wilcoxon Signed Rank Test..."),
                             ("sign", "Paired Sample &Sign Test..."),
                             ("ks2", "Two-Sample &Kolmogorov-Smirnov Test..."),
                             ("median", "Mood's M&edian Test...")]),
    ("Multivariate Analysis", [("pca", "&Principal Component Analysis..."),
                               ("kmeans", "&K-Means Cluster Analysis..."),
                               ("hca", "&Hierarchical Cluster Analysis..."),
                               ("lda", "&Discriminant Analysis..."),
                               None,
                               ("pls", "Partial &Least Squares...")]),
    ("Survival Analysis", [("km", "&Kaplan-Meier Estimator..."),
                           ("cox", "&Cox Proportional Hazards Model..."),
                           ("weibull", "&Weibull Fit Model...")]),
    ("ROC Curve", [("roc", "&ROC Curve...")]),
]

POWER = [("t1", "One-Sample t-Test"), ("t2", "Two-Sample t-Test"),
         ("tpair", "Paired-Sample t-Test"), ("p1", "One Proportion Test"),
         ("p2", "Two Proportions Test"), ("var1", "One Variance Test"),
         ("var2", "Two Variances Test"), ("anova1", "One-Way ANOVA")]


def _plain(text: str) -> str:
    return text.replace("&", "").strip().lower()


def _submenu(parent: QMenu, title: str) -> QMenu:
    for act in parent.actions():
        m = act.menu()
        if m is not None and _plain(act.text()) in (_plain(title), _plain(title) + "s"):
            return m
        if m is not None and _plain(title) == "nonparametric tests" and \
                _plain(act.text()).startswith("nonparametric"):
            return m
    return parent.addMenu("&" + title)


def install(main) -> None:
    """Add the WP-3 Statistics entries to ``main.menus["statistics"]`` (once per window)."""
    if getattr(main, "wp3_actions", None):
        return
    st.load_extensions()
    stm = main.menus["statistics"]
    main.wp3_actions = {}
    for title, entries in MENU:
        sub = _submenu(stm, title)
        for entry in entries:
            if entry is None:
                if sub.actions():
                    sub.addSeparator()
                continue
            key, text = entry
            act = main._act(text, lambda k=key: open_dialog(main, k))
            act.setObjectName(f"wp3_{key}")
            sub.addAction(act)
            main.wp3_actions[key] = act
    ps = _submenu(stm, "Power and Sample Size")
    for proc, text in POWER:
        act = main._act(f"{text}...", lambda p=proc: open_dialog(main, "power",
                                                                 {"procedure": p}))
        act.setObjectName(f"wp3_power_{proc}")
        ps.addAction(act)
        main.wp3_actions[f"power_{proc}"] = act


def open_dialog(main, key: str, settings: dict | None = None):
    """Open the dialog of analysis ``key`` (with preset ``settings``) and run it on OK."""
    dlg = sd.make_dialog(key, main, settings, main.selection_ranges())
    if not dlg.exec():
        return None
    return run(main, key, dlg.values())


def run(main, key: str, settings: dict):
    """Run analysis ``key`` with dialog ``settings``: through ``main.run_stat`` when it has
    input ranges, else into the active (or a new) workbook. Returns the report or None."""
    values = {k: v for k, v in settings.items() if k != "mode"}
    if st.range_strings(values):
        return main.run_stat(key, settings)
    mode = settings.get("mode", "manual")
    bw = main.active_book_window()
    if bw is None:
        bw = main.new_book()
    book = bw.book
    values["book"] = book.name
    books = main.project.data_windows()
    try:
        report, text = main._run_in_book(book, st.DIALOG_SPECS[key]([])[0],
                                         lambda: st.run(key, books, values, mode=mode))
    except (st.StatError, ValueError) as exc:
        main._warn(str(exc), "Statistics")
        return None
    recalc.stamp(main.project, report)
    main.result(text)
    bw.refresh()
    bw.show_sheet(report)
    return report
