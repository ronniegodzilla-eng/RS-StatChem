"""RSM menu: Design of Experiments / Response Surface Methodology (mirrors Design-Expert).

Create Design (factorial, fractional, general factorial, Plackett-Burman, Taguchi, central
composite, Box-Behnken, 3-level, D-optimal, simplex lattice / centroid, extreme vertices,
D-optimal mixture) -> enter responses -> Fit Summary & Analysis (model selection,
transformations, ANOVA, equations, effects, diagnostics) -> Model Graphs -> Optimization ->
Point Prediction; Taguchi S/N analysis. Core code lives in :mod:`rsstatchem.core.doe`.
"""
from __future__ import annotations

from rsstatchem.ui.doe.analysis import (DiagnosticsDialog, FitAnalysisDialog,
                                        ModelGraphsDialog, OptimizationDialog,
                                        PointPredictionDialog, TaguchiDialog,
                                        show_design_summary)
from rsstatchem.ui.doe.wizard import DesignWizard, open_wizard


def _wizard(category: str):
    def run(editor) -> None:
        open_wizard(editor, category)
    run.__name__ = f"create_{category}_design"
    return run


MENUS = {
    "RSM": [
        ("Create Design", [
            ("Full / Fractional Factorial...", _wizard("factorial")),
            ("General Factorial (Multilevel Categoric)...", _wizard("general")),
            ("Plackett-Burman...", _wizard("pb")),
            ("Taguchi Orthogonal Array...", _wizard("taguchi")),
            None,
            ("Central Composite (CCD)...", _wizard("ccd")),
            ("Box-Behnken...", _wizard("bbd")),
            ("3-Level Factorial...", _wizard("threelevel")),
            ("Optimal (D-optimal)...", _wizard("dopt")),
            None,
            ("Mixture Design...", _wizard("lattice")),
            ("Constrained Mixture (Extreme Vertices / D-optimal)...", _wizard("ev")),
        ]),
        ("Design Summary", show_design_summary),
        None,
        ("Fit Summary && Analysis...", FitAnalysisDialog),
        ("Diagnostics...", DiagnosticsDialog),
        ("Model Graphs...", ModelGraphsDialog),
        None,
        ("Optimization...", OptimizationDialog),
        ("Point Prediction...", PointPredictionDialog),
        None,
        ("Taguchi S/N Analysis...", TaguchiDialog),
    ],
}

__all__ = ["MENUS", "selftest", "DesignWizard", "FitAnalysisDialog", "DiagnosticsDialog",
           "ModelGraphsDialog", "OptimizationDialog", "PointPredictionDialog", "TaguchiDialog"]


def selftest() -> None:
    """Headless check of the DOE core: designs, Montgomery Ex. 11.2 fit, graphs, optimum."""
    import math

    import numpy as np

    from rsstatchem.core.dataset import Dataset
    from rsstatchem.core.doe import designs, model, optimize, report
    import pandas as pd

    res = designs.build_design({"design": "ccd", "seed": 7,
                                "factors": [{"name": "Time", "low": 80, "high": 90},
                                            {"name": "Temp", "low": 170, "high": 180}],
                                "responses": [{"name": "Yield"}]})
    assert res.meta["runs"] == 13 and abs(res.meta["alpha"] - math.sqrt(2)) < 1e-12
    x1 = [-1, -1, 1, 1, 0, 0, 0, 0, 0, 1.414, -1.414, 0, 0]
    x2 = [-1, 1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 1.414, -1.414]
    y = [76.5, 77.0, 78.0, 79.5, 79.9, 80.3, 80.0, 79.7, 79.8, 78.4, 75.6, 78.5, 77.0]
    ds = Dataset(pd.DataFrame({"x1": x1, "x2": x2, "y": y}))
    ds.meta["doe"] = {"kind": "process",
                      "factors": [{"name": n, "column": n, "low": -1, "high": 1}
                                  for n in ("x1", "x2")],
                      "responses": [{"name": "Yield", "column": "y"}]}
    a = model.analyze(ds, model.AnalysisSpec("y", ["x1", "x2"]))
    eq = dict((t, c) for c, t in a.fit.coded_equation())
    assert a.summary.suggested == "Quadratic"
    assert abs(eq[""] - 79.94) < 0.01 and abs(eq["A²"] + 1.376) < 0.002
    sections = [report.design_section(res.meta, res.notes, res.dataset),
                report.anova_section(a), report.diagnostics_section(a),
                report.graph_section(a, "contour", [0, 1], report.default_point(a.space))]
    prob = optimize.Problem(a.space, {"y": a.fit},
                            optimize.default_criteria({"y": a.fit}, a.space))
    sols = optimize.solve(prob, starts=5, seed=1)
    assert sols and sols[0]["D"] > 0.9
    sections.append(report.optimization_section(prob, sols, {"y": "Yield"}, [0, 1]))
    for kind in ("factorial", "bbd", "lattice", "taguchi"):
        designs.build_design({"design": kind, "factors": [{"name": n} for n in "PQR"],
                              "options": {"array": "L9"}, "seed": 1})
    assert all(s.items for s in sections)
    assert np.isfinite(a.fit.fit_statistics()["pred_r2"])
