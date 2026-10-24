"""Design of Experiments / Response Surface Methodology (mirrors Stat-Ease Design-Expert).

- :mod:`designs`  — design generation (factorial, fractional, Plackett-Burman, CCD, Box-Behnken,
  3-level, D-optimal, mixture, Taguchi) and the Dataset/meta builder.
- :mod:`model`    — coding, model terms, least-squares fit, Fit Summary, ANOVA, diagnostics,
  Box-Cox, backward reduction, coded/actual equations.
- :mod:`optimize` — Derringer-Suich desirability, numerical optimisation, point prediction,
  Taguchi signal-to-noise analysis.
- :mod:`plots`    — matplotlib figures (diagnostics, contour/3D/perturbation/interaction/cube,
  ternary, desirability, overlay, ramps, half-normal, Taguchi main effects).
- :mod:`report`   — high-level functions returning Output Viewer sections.

No Qt in this package.
"""
