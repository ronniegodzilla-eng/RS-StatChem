"""`RS-StatChem --selftest`: exercise every lazily-imported dependency inside a packaged build.

Exits 0 when reading/writing all file formats, running the analyses and exporting reports
works; otherwise prints the traceback and exits 1. Used by CI after PyInstaller.
"""
from __future__ import annotations

import tempfile
import traceback
from pathlib import Path


def run() -> int:
    try:
        import numpy as np
        import pandas as pd
        import statsmodels.api  # noqa: F401 - must be bundled for later analyses

        from rsstatchem.core.dataset import Dataset
        from rsstatchem.core.output import OutputDocument
        from rsstatchem.core.report import document_html, export_docx
        from rsstatchem.core.stats import descriptive
        from rsstatchem.io import files

        rng = np.random.default_rng(1)
        ds = Dataset(pd.DataFrame({"x": rng.normal(size=40), "g": rng.integers(1, 3, 40),
                                   "s": rng.choice(["a", "b"], 40)}))
        doc = OutputDocument()
        doc.add(descriptive.descriptives(ds, ["x"], ("mean", "std", "skew")))
        doc.add(descriptive.frequencies(ds, ["g", "s"], ("mode",), chart="bar"))
        doc.add(descriptive.frequencies(ds, ["x"], (), show_tables=False, chart="histogram"))

        from rsstatchem.core.stats import (anova, correlate, crosstabs, explore, nonparametric,
                                           regression, reliability, ttest)

        ds.add_variable("y", values=ds.df["x"] * 2 + rng.normal(size=40))
        ds.add_variable("h", values=rng.integers(1, 4, 40))
        doc.add(explore.explore(ds, ["x"], ["g"], histogram=True))
        doc.add(ttest.independent(ds, ["x"], "g", groups=(1.0, 2.0)))
        doc.add(ttest.paired(ds, [("x", "y")]))
        doc.add(anova.oneway(ds, ["y"], "h", welch=True, brown_forsythe=True,
                             posthoc_methods=("tukey", "games_howell"), means_plot=True))
        doc.add(anova.univariate(ds, "y", ["g", "h"], homogeneity=True,
                                 posthoc_factors=["h"], posthoc_methods=("tukey",)))
        doc.add(correlate.bivariate(ds, ["x", "y"], spearman=True, kendall=True))
        doc.add(correlate.partial(ds, ["x", "y"], ["h"]))
        doc.add(regression.linear(ds, "y", ["x", "h"], method="stepwise").section)
        doc.add(crosstabs.crosstabs(ds, ["g"], ["s"]))
        doc.add(nonparametric.mann_whitney(ds, ["x"], "g", (1.0, 2.0)))
        doc.add(nonparametric.two_related(ds, [("x", "y")], sign=True))
        doc.add(nonparametric.k_related(ds, ["x", "y", "h"], kendall=True))
        doc.add(reliability.reliability(ds, ["x", "y", "h"], validity=True))

        from rsstatchem.core.transform import Rule, compute, recode

        doc.add(compute(ds, "m", "MEAN.2(x, y) + CDF.NORMAL(0, 0, 1)", condition="g = 1"))
        doc.add(recode(ds, ["h"], [Rule(("range", 1.0, 2.0), ("value", 0.0)),
                                   Rule(("else",), ("copy",))], targets=["h2"]))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            files.save_project(ds, out / "a.rsd")
            files.load_project(out / "a.rsd")
            files.write_sav(ds, out / "a.sav")
            files.read_sav(out / "a.sav")
            files.write_excel(ds, out / "a.xlsx")
            files.read_excel(out / "a.xlsx", "Data")
            files.write_text(ds, out / "a.csv")
            files.read_text(out / "a.csv")
            export_docx(doc, str(out / "o.docx"))
            (out / "o.html").write_text(document_html(doc), encoding="utf-8")
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 1

    from rsstatchem.ui.menus import selftests

    failures = 0
    for module, error in selftests():
        if error:
            failures += 1
            print(f"selftest FAILED: {module}\n{error}")
        else:
            print(f"selftest ok: {module}")
    # the module windows import lazily, so a packaged build could miss them unnoticed
    import importlib

    from rsstatchem.ui.modules import MODULES

    window_modules = ["rsstatchem.ui.launcher", "rsstatchem.ui.dx.window",
                      "rsstatchem.ui.origin.window", "rsstatchem.ui.chemdraw.window",
                      "rsstatchem.ui.imagej.bar", "rsstatchem.ui.imagej.image_window",
                      "rsstatchem.ui.imagej.dialogs.generic"]
    for name in [m.opener.split(":")[0] for m in MODULES] + window_modules:
        try:
            importlib.import_module(name)
            print(f"selftest ok: {name}")
        except Exception:  # noqa: BLE001
            failures += 1
            print(f"selftest FAILED: {name}\n{traceback.format_exc()}")
    try:  # Image Analysis: core contracts, bundled readers, the area modules' selftests
        from rsstatchem.ui.imagej import selftest as imagej_selftest

        imagej_selftest()
        print("selftest ok: rsstatchem.ui.imagej")
    except Exception:  # noqa: BLE001
        failures += 1
        print(f"selftest FAILED: rsstatchem.ui.imagej\n{traceback.format_exc()}")
    if failures:
        return 1
    print("RS-StatChem selftest OK")
    return 0
