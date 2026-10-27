# RS-StatChem RSM / DOE module vs. Stat-Ease Design-Expert (v13 – v25) and Stat-Ease 360

Research date: 2026-10-27. I did not change any code. I read the code in `rsstatchem/core/doe/*`, `rsstatchem/ui/dx/*` and `rsstatchem/ui/doe/*` and ran the six DOE/DX test files headlessly: **155 passed, 1 skipped (19 s)**.

## How this was checked

- **Stat-Ease sources.** The Stat-Ease documentation (v23.1 and v25.0), its changelog, "What's New" pages (v13, v25), the product page and the tutorial index. Several doc pages are thin or only show screenshots. Items marked **(exp.)** come from general knowledge of the product and should be confirmed against a running copy of Design-Expert if one is available.
- **Status labels.**
  - **P** = present and working like Design-Expert.
  - **Pa** = partial (what is missing is listed).
  - **M** = missing.
  - **360** = Stat-Ease 360 only (lower priority).

## 0. What the app has today (short)

- **Core** (`rsstatchem/core/doe/`):
  - `designs.py` (1876 lines): all the builders, foldover, split-plot builders, D-optimal point exchange, mixtures, minimum-run designs (MR5/MR4), Taguchi.
  - `model.py` (1791 lines): terms, OLS `Fit`, REML `SplitPlotFit` with Kenward-Roger, Fit Summary, Box-Cox, automatic selection, effects, Lenth.
  - `evaluation.py`: degrees of freedom, standard errors, VIF, Ri², power, leverage, FDS.
  - `optimize.py`: Derringer-Suich desirability, multi-start Nelder-Mead, Taguchi S/N.
  - `plots.py`: matplotlib graphs.
  - `report.py`: report sections.
- **UI** (`rsstatchem/ui/dx/`): a Design-Expert-like window with the tree Design / Analysis / Optimization / Post Analysis, the wizard, the layout grid, the analysis tabs (Transform, Fit Summary, Model, ANOVA, Diagnostics, Model Graphs; Effects for factorials; REML for split-plot), Numerical and Graphical optimization, Point Prediction, Confirmation and Coefficients Table.
  - Design files are saved as `.rsx` (JSON).
  - Import accepts CSV/XLSX responses; export gives CSV/XLSX layouts and PNG graphs.
  - Tutorial data: Montgomery 11.2, 6.2, 5.1 and 14.4.
- **Old SPSS-style front end** (`rsstatchem/ui/doe/`): dialogs on the same core. These can analyse arbitrary data sets, which is a partial form of "historical data".

## 1. Feature inventory with status

### 1.1 Design builders – Factorial

| Design-Expert feature | Status | Notes / what is missing |
|---|---|---|
| Regular two-level (full/fraction, generators, defining relation, resolution, alias list, "stoplight" colours, blocks by confounding, centre points, replicates) | Pa | The stoplight table that colours resolution by number of runs is **M** in the wizard. Blocking and aliases are P. |
| Irregular fractions (e.g. 12 runs for 4 factors, 3/4 fractions) | M | Screen tip "Irregular Fractions Factor Information". |
| Min-Run Characterize (MR5) and Min-Run Screen (MR4, +2 runs) | P | Run counts checked against Stat-Ease in the tests. |
| Plackett-Burman | P | 12, 20, 24 … runs from Hadamard matrices. |
| Taguchi orthogonal arrays (with S/N analysis) | P | |
| Multilevel categoric (general factorial; replicates, blocks) | P | |
| Optimal (custom) factorial | Pa | Only via the RSM "dopt" builder. No factorial-interaction model type, no I/A criteria, no coordinate exchange. |
| Split-plot: regular two-level, multilevel categoric, optimal | P/Pa | Very-hard-to-change factors (split-split-plot) are 360 (v25). |
| Up-front power on the response page (delta, sigma, signal/noise) | Pa | Signal/noise only on the Evaluation node, not in the wizard. |
| Power for binomial responses | M | Screen tip "Response Info (binomial on)". |
| Definitive screening (5–30 factors; 3-level numeric, 2-level categoric) | M | Listed under RSM → "Supersaturated" in the v25 docs. |

### 1.2 Design builders – Response surface

| Feature | Status | Notes |
|---|---|---|
| CCD: circumscribed / inscribed / face-centred; alpha rotatable / spherical / orthogonal quadratic / practical / custom; blocks; centre points by block; categoric factors crossed | P | |
| Box-Behnken (3–7 factors; blocks) | Pa | Blocking of BBD (e.g. 4 factors in 3 blocks, 5 factors in 2) is **M**. |
| One-factor RSM | M | |
| Miscellaneous: 3-level factorial (P), Hybrid, Pentagonal, Hexagonal | Pa | Hybrid, pentagonal and hexagonal are **M**. Stat-Ease itself does not recommend them. |
| Optimal (custom) RSM | Pa | See 1.5. |
| User-defined (candidate list of point types or a candidate file) | M | |
| Historical data (blank layout from limits; import existing data) | M in DX UI | The SPSS-style dialogs analyse arbitrary data. |
| Split-plot CCD | P | |
| Split-plot optimal RSM | P | Coordinate-type exchange inside `d_optimal_split_plot`. |
| RSM with random blocks (REML) | 360 / M | |

### 1.3 Design builders – Mixture

| Feature | Status | Notes |
|---|---|---|
| Simplex lattice {q,m} and simplex centroid, with augmentation (axial check blends and overall centroid) and replicates | P | |
| Screening designs: simplex (axial) and non-simplex (extreme vertices) | Pa | The "Screening (Extreme Vertices)" option exists. A simplex screening design (Snee-Marquardt axial design for linear blending) is **M**. |
| Optimal mixture: D only; LOF and replicate points; L-pseudo components | Pa | **M**: I/A criteria, multiple linear constraints, blocks, edit model, U-pseudo. |
| User-defined and historical mixture | M | |
| Split-plot mixture (HTC component) | M | |
| Mixture models: Scheffé L, Q, SC, C (P); special quartic and quartic, Slack models, inverse terms (M) | Pa | |
| Equations in pseudo / real / actual units | Pa | L-pseudo and actual are P. U-pseudo and real conversion are **M** (Stat-Ease advanced topics "Convert L/U pseudo to real"). |

### 1.4 Combined and custom designs

| Feature | Status |
|---|---|
| Combined mixture × process (optimal, user-defined, historical; crossed Scheffé × process models; Mix, Process, Mix×Process and Mix-Mix graphs) | **M**. This is a big gap: a whole new design kind. |
| Two independent mixtures | M |
| Split-plot combined | M (360 / advanced) |
| Simple sample (mean, SD and intervals of one column) | M |
| Space-filling: Optimal Distance (DX) and Latin hypercube (360) | M |

### 1.5 Optimal (custom) builder internals

| Feature | Status | Notes |
|---|---|---|
| Criteria: I (Stat-Ease's recommended criterion for RSM), D, A, Modified Distance, Distance | Pa | Only **D** is implemented. |
| Exchange methods: point exchange (P = Fedorov), coordinate exchange (M), Best = half of the starts each (M) | Pa | |
| Candidate set types: vertices, edge centres, thirds of edges, triple blends, constrained plane centroids, axial check blends, interior points, overall centroid; import a candidate file | Pa | The 3^k grid, a 4-level grid for cubic, and mixture vertex/edge/plane/centroid sets exist. Interior points, thirds of edges as an option, and candidate import are **M**. |
| Model points (≥ p), LOF points (distance-based, default 5), replicates (default 5), centroids | P | |
| Edit model (custom term list, higher-order terms) | M | Only the order can be chosen. |
| Blocks in optimal designs | M | |
| Force runs (include existing points) | M | Needed for optimal augment. |
| Multiple linear constraints (MLC, for example `0.2 ≤ A + 2B ≤ 1`) for process and mixture designs; Constraints node editable after the build (v13) | M | Only box limits for process factors; box limits plus the sum constraint for mixtures. |
| Complex (nonlinear) constraints | M | Low priority. |
| Discrete numeric factors | M | |

### 1.6 Design tools and editing

| Feature | Status | Notes |
|---|---|---|
| Sort by std/run, randomize, coded/actual display, show point type, design points on graphs | P | |
| Edit factor and response info; notes; ignore runs (Select column) | P | |
| Augment: foldover (full, single-factor, semifold) | P | |
| Augment: add centre points; add axial points (factorial → CCD); add blocks/replicates; optimal augment for a new model (doubles model points by default, adds a phase block) | **M** | Foldover is the only augmentation. |
| Modify Design Space wizard (move, grow, shrink; v13) | M | |
| Design-space augmentation | M | |
| Round factor values (decimals or significant digits; mixture keeps the sum) | M | |
| Insert or delete runs; add a response column after the build; rename blocks/groups (v25) | M | Not found in `document.py`. |
| Column Info Sheet (factor SDs for POE, discrete levels, response units and limits) | Pa | Name/units dialog only. |
| Import Data Set (paste data, set column properties, infer coding; v13); Excel import/export (v23) | Pa | Responses import from CSV/XLSX into an existing layout. Creating a design from imported data is **M**. |
| Covariates and extra explanatory variables | M | |
| Simulate response from an equation (Equation Entry) | M | Also useful for teaching and tests. |
| Graph Columns: scatter with colour-by (P); box-and-whisker (v13, **M**); correlation grid/heat map (**M**) | Pa | |

### 1.7 Design evaluation

| Feature | Status | Notes |
|---|---|---|
| Model choice by order | P | **M**: factorial "interactions" vs polynomial, Scheffé vs Slack, Add Term, error terms, ignore order. |
| Degrees of freedom (model, residual, LOF, pure error, blocks) | P | |
| Terms table: standard error / sigma, VIF, Ri², power | P | |
| Power options: delta and sigma separately, alpha, binomial | Pa | Signal/noise ratio only; alpha fixed at 0.05. |
| Aliases: regular-fraction alias list (P); alias matrix for any design and model, with partial-alias coefficients (PB, MR4, optimal) | Pa | |
| Defining contrasts | P | Shown in Summary. |
| Leverage table (with Space Type and Build Type) | P/Pa | Build Type for optimal designs is **M**. |
| Matrix measures: condition number, D/A/G/I efficiency, scaled D, average prediction variance, max/average leverage | M | |
| Correlation matrix of coefficients and Pearson r of model columns (heat maps) | M | |
| FDS graph | Pa | Mean-SE curve only. **M**: delta/sigma/alpha settings with the "FDS score", prediction vs mean, paired (difference) FDS. |
| Standard error of design (contour and 3D) | M | |
| Split-plot evaluation (GLS with variance ratio) | P | |

### 1.8 Analysis

| Feature | Status | Notes |
|---|---|---|
| Transforms: none, sqrt, ln, log10, inverse sqrt, inverse, power (with constant k) | P | |
| Transforms: logit (with bounds) and arcsine square root | M | |
| Box-Cox plot with recommended lambda, CI and "use recommendation" | P | |
| Logistic regression (binary; IRLS; success level; pseudo-R² and goodness-of-fit tests) | M | |
| Poisson regression (counts; v13) | M | |
| Gaussian process models (with diagnostics in v25) | 360 / M | |
| Multiple analyses per response (+ button, v13) | M | |
| Analysis Summary table (LOF and curvature p-values per analysis; v23) | M | |
| Model limits | M | |
| Effects (factorials): half-normal plot (P), Pareto chart with t and Bonferroni limits (P), Lenth PSE (P), click-to-select (P), split-plot separate whole-plot/subplot PSE (P) | P | **M**: full normal plot of effects, Shapiro-Wilk on the half-normal plot (exp.), automatic effect pick, multilevel-categoric selection list. |
| Fit Summary: sequential SS table, lack-of-fit tests, model summary statistics, suggested model, aliased flag; mixture variant | P | Combined designs **M**. |
| Model tab: orders mean to cubic; manual terms up to 6-factor interactions; hierarchy option; automatic selection forward/backward/stepwise with p-value, AICc or BIC criteria, alpha in/out, forced terms | P | **M**: quartic/5th/6th order process models, "all hierarchical" selection (exp.), error terms, Add Term. |
| ANOVA: SS, df, MS, F, p (Design-Expert colours), blocks, LOF, pure error; fit statistics (Std Dev, Mean, C.V. %, R², adj R², pred R², PRESS, Adeq Precision); coefficient table (SE, 95 % CI, VIF); coded and actual equations; REML ANOVA with Kenward-Roger, variance components with profile CIs | P | |
| Curvature test for factorials with centre points ("ANOVA with Center Points") | **M** | High priority, small. No "curvature" anywhere in the code. |
| ANOVA for one categoric factor (treatment means, pairwise LSD/Tukey) | M | |
| −2 log-likelihood / AICc / BIC on the ANOVA page | Pa | Used inside the selection steps; not reported in the fit statistics table (exp.). |
| Diagnostics plots: normal plot, residuals vs predicted/run/factor, predicted vs actual, Box-Cox, Cook's distance, leverage, DFFITS, DFBETAS; case statistics report with Bonferroni outlier limits | P | |
| Diagnostics: covariance ratio, Durbin-Watson (v23.1), residual type choice (internal/external/raw), colour-by, residuals vs covariate, random-effects Cook's distance | M | |
| Mean bias correction (predicted mean vs median under a transform) | P | In `Fit.interval`. |

### 1.9 Model graphs

| Graph / option | Status | Notes |
|---|---|---|
| Perturbation, one factor (with CI bands), interaction, cube, contour, rotatable 3D surface | P | |
| Mixture: ternary contour, 3D, Cox trace, one factor | P | Piepel direction trace **M**. |
| Flags on graphs, factors tool with sliders and axis choice, design points | P | v25 "flags anywhere" is P/Pa. |
| Standard Error plot (contour/3D of SE of prediction) | M | |
| All Factors plot; Categoric "contour"; 3D categoric; two-component mixture plot; Mix×Process graphs | M | |
| LSD bars on interaction / one-factor graphs; PI bands | M | |
| Set contour value / custom contour levels; 2D grid lines; colour maps (9 perceptually uniform in v25); graph preferences (fonts, axes, colours); multi-graph control (v13) | Pa/M | Fixed 10 levels and one colour map. |
| Copy / export graph as image | P | |

### 1.10 Optimization and Post Analysis

| Feature | Status | Notes |
|---|---|---|
| Numerical: goals none/max/min/target/in range for responses; max/min/target/in range for factors; limits, weights (0.1–10), importance (+ to +++++); D = weighted geometric mean | P | |
| Goal **Cpk** (LSL/USL; Cpk low 0, high 1.5) | M | |
| Factor goal **equal to** | M | |
| **Use interval** on a goal (CI, PI or TI; one-sided) | M | |
| Standard-error models (cap extrapolation) | M | |
| Discrete numeric factors in the search | M | |
| POE goal (minimize propagation of error) | M | |
| Options: number of starts, duplicate filter | Pa | Fixed 30 starts. |
| Solutions list, ramps, bar graph, desirability contour/3D, per-response desirability shown (v25) | P/Pa | Ramps and bars at any point in the design space (v25) are Pa. |
| Graphical overlay with the yellow window; ternary overlay | P | |
| Graphical: interval bands (CI/PI/TI, one-sided) on overlay limits | M | |
| Point Prediction: predicted mean and median, SD, SE mean, CI, PI | P | **M**: tolerance interval (default 99 % of the population at 95 % confidence), POE line. |
| Confirmation: PI for the mean of n runs, pass/fail | Pa | n and the mean are typed in. Design-Expert matches entered runs to the location and lists individual runs. |
| Coefficients table across responses with p-value colours | P | |
| Propagation of error (factor SDs → POE surface, POE in prediction and optimization) | M | |
| Python integration / scripting; Weibull lifetime regression | 360 / M | |

### 1.11 Files, preferences, reporting

| Feature | Status |
|---|---|
| Native file (`.rsx` JSON); open/save/recent | P |
| Read Design-Expert `.dxpx` files | M. The format is undocumented and proprietary; not recommended. Import via Excel/CSV export instead. |
| Export layout CSV/XLSX; graphs PNG; reports copyable as Word HTML and Excel TSV | P |
| Edit › Preferences: significance threshold, CI level, TI proportion, default model selection, hierarchy, graph defaults | M. Levels are hard-coded at 0.05 / 95 %. |
| Screen tips (F1); annotations toggle | P |
| Print / report builder | Pa. Copy and export exist; there is no assembled multi-node report (exp.). |

## 2. Prioritised gaps

**P1 – high value for chemists and pharma users, or small effort**

1. **Curvature test** in factorial ANOVA with centre points, plus an **Analysis Summary** node (LOF and curvature p-values). Small.
2. **Optimal (custom) engine.**
   - I- and A-optimal criteria (I is Design-Expert's default for RSM).
   - Coordinate exchange (Meyer & Nachtsheim 1995) and a "Best" mode.
   - Blocks and forced runs.
   - Custom model terms (Edit Model).
   - **Multiple linear constraints** for process and mixture designs (candidate vertices by Piepel's CONVRT / McLean-Anderson enumeration), editable afterwards.

   This is the foundation for items 3, 4 and 7.
3. **Augment design**: add centre points, add axial points (factorial → CCD), add replicates/blocks, **optimal augment** for a new model (with a phase block), then Modify Design Space.
4. **Combined mixture-process designs and models** (crossed Scheffé × process), with Mix×Process graphs.
5. **Optimization and post-analysis completeness.**
   - Tolerance interval in Point Prediction.
   - Goals: Cpk and equal-to.
   - Use Interval (CI/PI/TI, one-sided) in numerical and graphical optimization.
   - Standard-error models and discrete numeric factors.
   - **Propagation of error** (factor SDs, POE graph, POE goal, POE line in prediction).
6. **Historical data / user-defined / Import Data Set**: build a design document from arbitrary data with column properties, then analyse it in the DX UI.
7. **Transforms logit and arcsine square root**; **logistic and Poisson regression** (IRLS; statsmodels GLM can be the oracle or even the engine).

**P2**

8. Definitive screening designs (conference matrices) and the DSD analysis method (Jones & Nachtsheim 2017 two-step).
9. Evaluation: alias matrix for any design, matrix measures (D/A/G/I efficiency, condition number, leverage), correlation heat maps, Standard Error of Design graph, FDS options (delta/sigma/alpha, FDS score, prediction vs mean), power with delta and sigma, up-front power in the wizard.
10. Model and analysis extras: quartic to 6th order, Add Term, error terms, Slack models, special quartic / inverse mixture terms, U-pseudo and real equations, multiple analyses per response, model limits, one-categoric ANOVA with LSD.
11. Diagnostics extras: Durbin-Watson, covariance ratio, normal plot of effects, Shapiro-Wilk on the half-normal plot, residual type and colour-by, covariates.
12. Graph extras: Standard Error plot, All Factors, LSD bars, PI bands, categoric contour and 3D categoric, two-component plot, custom contour levels, colour maps, preferences, multi-graph control.
13. Smaller builders: one-factor RSM, irregular fractions, BBD blocks, simplex screening, simple sample, hybrid/pentagonal/hexagonal, Optimal-Distance and LHS space-filling designs.
14. Layout editing: round values, insert/delete runs, add a response, rename blocks, simulate response, box-whisker and correlation in Graph Columns, Preferences dialog.

**P3** (Stat-Ease 360 or niche): Gaussian process models, split-split-plot, random blocks, Weibull regression, Python scripting, binomial power, complex constraints.

## 3. Licences and open-source equivalents

**Design-Expert itself** is proprietary. The algorithms it uses are published methods, so re-implementing them from the literature is clean.

What to avoid:
- Copying Stat-Ease help text, screen-tip wording, icons or bundled tutorial data files verbatim.
- Reading or writing `.dxpx` (proprietary format).
- Presenting the app as "Design-Expert". Design-Expert® and Stat-Ease® are registered trademarks; "Design-Expert-style" wording in docs is the safe limit, and current in-app text already uses its own artwork.

Textbook data (Montgomery, Myers et al., Cornell) used as small test fixtures with a citation is normal practice.

| Need | Published method (implement ourselves) | Open-source reference | Licence | How to use |
|---|---|---|---|---|
| I/A/D criteria, coordinate exchange, point exchange | Fedorov 1972; Cook & Nachtsheim 1980; Meyer & Nachtsheim 1995 (coordinate exchange); Atkinson-Donev-Tobias 2007; Goos & Jones 2011 | **pyDOE3** 1.6.2 (Jan 2026): `optimal_design` with A/C/D/E/G/I/S/T/V criteria and Fedorov, modified Fedorov, DETMAX, Wynn-Mitchell | **BSD-3** | May be vendored or depended on with its notice. |
| | | **dexpy** (Stat-Ease's own, v0.12, 2017; builders, power, alias list, coordinate-exchange optimal) | **Apache-2.0** | Code may be ported with a NOTICE and attribution. Old, but written by Stat-Ease staff (Anderson, Bezener), so its power and alias conventions match Design-Expert. |
| | | R **skpr** 1.9.2 (D, I, A, Alias, E, T, G; power), **AlgDesign** 1.2.1.2 | GPL-3, GPL ≥ 2 | Oracle only: run R outside the product to produce expected values. Do not copy code into a closed-source build. |
| Fractions, aliases, PB, orthogonal arrays, conference matrices / DSD | Box-Hunter-Hunter; Plackett & Burman 1946; Jones & Nachtsheim 2011 (DSD); Xiao, Lin & Bai 2012 (conference-matrix DSD) | **OApackage** 2.7.21 (orthogonal arrays, conference designs) | **BSD-3** | Usable. |
| | | R **FrF2**, **DoE.base**, **daewr** (`DefScreen`, data sets) | GPL ≥ 2, GPL-2 | Oracle only. |
| RSM analysis, canonical analysis, steepest ascent | Box & Wilson 1951; Myers-Montgomery-Anderson-Cook | R **rsm** 2.10.6 | GPL-2/3 | Oracle only. |
| Mixtures, extreme vertices, constraints | Scheffé 1958; McLean & Anderson 1966; Snee & Marquardt 1974; Piepel 1983/1988 (CONVRT, CONAEV); Cornell 2002 | R **mixexp** (archived from CRAN 2026-04-10) | GPL | Oracle only, and fragile. Our own vertex enumeration by linear programming / combinatorics is simple. |
| Desirability | Derringer & Suich 1980; Derringer 1994 | R **desirability** | GPL | Oracle only. |
| Logistic / Poisson GLM; mixed models | McCullagh & Nelder; Kenward & Roger 1997 | **statsmodels** (already a dependency) | **BSD-3** | Can be the engine or the oracle. |
| Latin hypercube, Sobol, maximin distance | McKay 1979; Johnson-Moore-Ylvisaker 1990 | **scipy.stats.qmc** (`LatinHypercube` with optimization) | BSD-3 | Use directly. |
| Gaussian process (360) | Rasmussen & Williams 2006 | scikit-learn GPR (BSD-3; not currently a dependency) or numpy/scipy implementation | BSD | |
| Tolerance intervals | Wallis 1951; Howe 1969; Odeh & Owen 1980; Young 2013 | R **tolerance** (`regtol.int`, `K.factor`) | GPL | Oracle only. |
| POE / robust design | Box & Fung; Anderson & Whitcomb (Stat-Ease papers on POE); Myers-Montgomery-Anderson-Cook ch. 10 | none needed | | Taylor expansion of the fitted model. |
| Lenth, Box-Cox, Durbin-Watson, influence measures | Lenth 1989; Box & Cox 1964; Belsley-Kuh-Welsch 1980 | scipy / statsmodels | BSD | |
| MR5 / MR4 | Oehlert & Whitcomb 2002; Anderson & Whitcomb 2004 | | | Already implemented. |

Packaging rule of thumb: only BSD, MIT or Apache code (pyDOE3, dexpy, OApackage, scipy, statsmodels) may go into the PyInstaller bundle. Use GPL R packages solely to generate expected numbers for the tests.

## 4. Published examples for verification

Values marked (C) were recomputed by me with numpy/scipy from the stated data. (B) means quoted from a book or document. Flagged items need a check against the printed source.

**E1. Montgomery DAE Example 11.2, CCD chemical process** (data already in `ui/dx/examples.py` and tested). (B)
- Coded model: ŷ = 79.94 + 0.995 x1 + 0.515 x2 − 1.376 x1² − 1.001 x2² + 0.25 x1x2.
- Stationary point x = (0.389, 0.306), ŷ = 80.21.
- Montgomery Example 11.4 / Table 11.6 adds viscosity and molecular-weight responses for the desirability and overlay example: yield ≥ 78.5, 62 ≤ viscosity ≤ 68 (target 65), Mn ≤ 3400. **Transcribe y2 and y3 from Table 11.6**; I could not verify them online.

**E2. Chemical conversion CCD** (Myers & Montgomery; the same experiment as the Stat-Ease RSM tutorial).
- Factors: time 40–50 min, temperature 80–90 °C, catalyst 2–3 %; alpha 1.682; 6 centre points.
- Data (coded x1 x2 x3 → conversion, activity), runs 1–20:
  - Factorial points:
    - (−,−,−) 74, 53.2
    - (+,−,−) 51, 62.9
    - (−,+,−) 88, 53.4
    - (+,+,−) 70, 62.6
    - (−,−,+) 71, 57.3
    - (+,−,+) 90, 67.9
    - (−,+,+) 66, 59.8
    - (+,+,+) 97, 67.8
  - Axial points:
    - (−α,0,0) 76, 59.1
    - (+α,0,0) 79, 65.9
    - (0,−α,0) 85, 60.0
    - (0,+α,0) 97, 60.7
    - (0,0,−α) 55, 57.4
    - (0,0,+α) 81, 63.2
  - Centre points: 81, 75, 76, 83, 80, 91 for conversion; 59.2, 60.4, 59.1, 60.6, 60.8, 58.9 for activity.
  - The data come from a secondary web source (Vining-style blog). Conversion is verified by the coefficients below; **check the activity values against MMA-C**.
- Expected conversion, quadratic (C), unblocked:
  - b0 81.091, A 1.028, B 4.040, C 6.204, AB 2.125, AC 11.375, BC −3.875, A² −1.834, B² 2.938, C² −5.191.
  - R² 0.9199, adj R² 0.8478, pred R² 0.7566, PRESS 676.28, s 4.717.
  - The Stat-Ease tutorial quotes B = 4.04057 and AC = 11.375 (B). It runs the design in 2 blocks: block 1 = 8 factorial + 4 centre runs, block 2 = 6 axial + 2 centre runs.
- Activity, linear (C): 60.51 + 3.583 A + 0.255 B + 2.23 C. Stat-Ease says activity is linear.
- Optimization check (Stat-Ease tutorial pt 2/3):
  - Goals: conversion maximize, 80–100; activity target 63 (60–66).
  - The best solution has temperature near 90 °C, with a second local optimum near 80 °C.
  - Confirmation: 3 activity runs 62, 63, 64 with a 95 % PI.
  - Part 3 adds a POE (conversion) goal: minimize, limits 4–5.

**E3. Montgomery Example 6.2, 2⁴ filtration rate** (data in `examples.py`, standard order 45, 71, 48, 65, 68, 60, 80, 65, 43, 100, 45, 104, 75, 86, 70, 96).
- Effects (C):
  - Main effects: A 21.625, B 3.125, C 9.875, D 14.625.
  - Two-factor: AB 0.125, AC −18.125, AD 16.625, BC 2.375, BD −0.375, CD −1.125.
  - Higher order: ABC 1.875, ABD 4.125, ACD −1.625, BCD −2.625, ABCD 1.375.
- Lenth (C): s0 3.9375, PSE 2.625, d = 5.
  - ME = 6.748 (t 0.975,5 = 2.5706).
  - Lenth's SME = 13.699.
  - Bonferroni limit as the app and Design-Expert use it: t(1 − 0.05/30, 5) = 5.2474 → 13.77.
- Reduced model A, C, D, AC, AD (C = B):
  - 70.0625 + 10.8125A + 4.9375C + 7.3125D − 9.0625AC + 8.3125AD.
  - SS: A 1870.56, C 390.06, D 855.56, AC 1314.06, AD 1105.56; residual 195.125 on 10 df.

**E4. Curvature test (Montgomery §6.8, 2² + 5 centre points)** (C = B).
- Factorial yields (−,−) 39.3, (+,−) 40.9, (−,+) 40.0, (+,+) 41.5; centre 40.3, 40.5, 40.7, 40.2, 40.6.
- SS_A 2.4025, SS_B 0.4225, SS_AB 0.0025.
- SS_curvature = nF·nC(ȳF − ȳC)²/(nF + nC) = 0.00272.
- SS_pure error 0.1720 on 4 df; F 0.0633, p 0.8137.

**E5. Box-Cox drill advance (Montgomery Ex. 6.3)**: log transform recommended. Already in the tests.

**E6. Cornell yarn elongation (Montgomery Ex. 11.5)**, simplex lattice {3,2}, already in `tests/test_doe.py`. Expected 11.7 A + 9.4 B + 16.4 C + 19.0 AB + 11.4 AC − 9.6 BC.

**E7. Constrained mixture: McLean & Anderson (1966) flare** (Cornell 2002; Montgomery).
- Limits: magnesium 0.40–0.60, sodium nitrate 0.10–0.50, strontium nitrate 0.10–0.50, binder 0.03–0.08.
- Expected 8 extreme vertices (the upper limits of 0.5 are never active) (C):
  - (.4, .1, .47, .03), (.4, .1, .42, .08), (.6, .1, .27, .03), (.6, .1, .22, .08)
  - (.4, .47, .1, .03), (.4, .42, .1, .08), (.6, .27, .1, .03), (.6, .22, .1, .08)
- Overall centroid of the vertices: (0.5, 0.2225, 0.2225, 0.055).
- Six face centroids complete the classic 15-point design. Use this to test vertex enumeration and, by adding a constraint (for example A + D ≤ 0.65), to test multiple linear constraints. Illumination responses: see Cornell or McLean-Anderson.

**E8. Exact optimal designs, one factor, quadratic on [−1, 1], candidates {−1, 0, 1}** (C). I is the average variance over the uniform region, with moment matrix [[1,0,1/3],[0,1/3,0],[1/3,0,1/5]].

| n | D-optimal (det X'X) | I-optimal (average variance) | A-optimal (trace) |
|---|---|---|---|
| 4 | det 8; three ties: {−1,−1,0,1}, {−1,0,0,1}, {−1,0,1,1} | {−1,0,0,1}, I = 0.5333 (unique) | {−1,0,0,1}, 2.0 |
| 5 | det 16 (ties) | {−1,0,0,0,1}, I = 0.4444 | {−1,0,0,0,1}, 1.6667 |
| 6 | {−1,−1,0,0,1,1}, det 32 | I = 0.3778: {−1,−1,0,0,0,1} or mirror | 1.4167 |
| 7 | det 48 (ties) | {−1,−1,0,0,0,1,1}, I = 0.3111 | 1.1667 |

The continuous limits are D weights (1/3, 1/3, 1/3) and I weights (1/4, 1/2, 1/4). This shows that D and I differ, which is exactly why Design-Expert defaults to I for RSM.

**E9. Plackett-Burman 12**: generator row + + − + + + − − − + − with cyclic shifts plus a row of −1. X'X = 12 I (C). Two-factor interactions are partially aliased with main effects at ±1/3. Test the alias matrix with this.

**E10. DSD, m = 6** (Jones & Nachtsheim 2011), 13 runs = [C; −C; 0] with the conference matrix
C6 = [[0,1,1,1,1,1],[1,0,1,−1,−1,1],[1,1,0,1,−1,−1],[1,−1,1,0,1,−1],[1,−1,−1,1,0,1],[1,1,−1,−1,1,0]].
Checks (C): C'C = 5I; main effects orthogonal to every 2FI and every quadratic column (max |dot| = 0).

**E11. Tolerance-interval k-factor (Howe approximation)**, k = z_{(1+P)/2}·√((n−1)(1+1/n)/χ²_{1−γ, n−1}) (C):
- n = 10, P = 0.99, γ = 0.95: k = 4.4446 (exact tables ≈ 4.43).
- n = 20, P = 0.99, γ = 0.95: k = 3.6171.
- n = 10, P = 0.95, γ = 0.95: k = 3.3819.

For regression use the effective n* = 1/h(x) (Young 2013). Cross-check with R `tolerance::regtol.int` and, if possible, with Design-Expert output (its default is P = 99 %, 95 % confidence). Stat-Ease does not document its exact TI formula: **verify**.

**E12. Desirability by hand.**
- Maximize, L = 80, U = 100, y = 90: d = 0.5 with weight 1; 0.25 with weight 2; 0.7071 with weight 0.5.
- Target 63 on 60–66 at y = 64.5: d = 0.25.
- D with importances r: (Π d_i^{r_i})^{1/Σr}.
- Literature case: Derringer & Suich (1980) tire tread compound, a 20-run CCD with 4 responses.

**E13. POE.**
- Formula: POE(x) = √(Σ_i (∂ŷ/∂x_i)² σ_i² + σ̂²_resid), in actual units; the model must be hierarchical. Some texts add second-order terms ½Σ(∂²ŷ/∂x_i²)² σ_i⁴: **check which form Design-Expert uses**.
- Check against Monte-Carlo simulation of x ~ N(x0, σ) through the fitted model, and against the Stat-Ease tutorial pt 3 setup.

**E14. Split-plot**: Montgomery paper pulp (Table 14.16) and the Stat-Ease 9-factor split-plot tutorial (2^(9−1), 8 groups). Both are already in the tests.

**E15. GLM**: logistic and Poisson fits should agree with `statsmodels.GLM` (Binomial/logit, Poisson/log) to 1e-6. Data: the Stat-Ease tutorials "Logistic Regression Tools" and "Poisson Regression – Antiseptic" (data only inside the software), or Myers-Montgomery-Vining-Robinson *Generalized Linear Models* examples.

## 5. Work packages (disjoint file ownership)

Shared rules for all packages:
- Do not edit `rsstatchem/ui/doe/*` or the existing `tests/test_doe*.py` / `tests/test_dx*.py`. If a behaviour change breaks an old test, raise it with the coordinator.
- Each package adds its own test file.
- New tests use the `qapp` fixture (see the memory note on the Qt shutdown pitfall).

### WP-A: Design builders, constraints, augmentation, evaluation (P1 items 2, 3, 4-design, 6; P2 items 8, 9, 13, 14)

- **Owns:** `core/doe/designs.py`, new `core/doe/optimal.py`, new `core/doe/constraints.py`, `core/doe/evaluation.py`, new `core/doe/plots_design.py`, `ui/dx/wizard.py`, `ui/dx/document.py`, `ui/dx/info.py`, `ui/dx/layout.py`, `ui/dx/window.py` (menus and tree), `ui/dx/examples.py`, `ui/dx/start.py`, `ui/dx/icons.py`, new `tests/test_doe_build2.py`.
- **Scope:**
  - The optimal engine: I/A/D criteria, coordinate, point and best exchange, candidate types, blocks, forced runs, custom terms.
  - Multiple linear constraints (with a Constraints node).
  - Combined mixture-process builder and its term generation, in a new `core/doe/terms_combined.py` that WP-B imports.
  - DSD, one-factor RSM, BBD blocks, simplex screening, LHS / optimal distance.
  - Augment: centre points, axial points, replicates, optimal augment, modify design space.
  - Historical, user-defined and Import Data Set.
  - Evaluation extras: alias matrix, matrix measures, correlation heat maps, SE-of-design graph, FDS options, power with delta/sigma.
  - Layout editing: round, insert/delete runs, add response, rename blocks.
  - Hosting the menu and tree entries that WP-B and WP-C need. Those packages publish view classes; WP-A wires them into `window.py`, ideally in one integration commit at the end.
- **Verification:** E7, E8, E9, E10 and the Stat-Ease run counts.

### WP-B: Analysis engine and model graphs (P1 items 1, 7; P2 items 10, 11, 12)

- **Owns:** `core/doe/model.py`, `core/doe/report.py`, `core/doe/plots.py`, new `core/doe/glm.py`, `ui/dx/analysis.py`, `ui/dx/widgets.py`, `ui/dx/style.py`, new `tests/test_doe_analysis2.py`.
- **Scope:**
  - Curvature test and the Analysis Summary (provide the view class; WP-A adds the tree node).
  - logit and arcsine square root transforms.
  - Logistic and Poisson via IRLS (statsmodels as oracle).
  - Multiple analyses per response (stored in `meta["doe"]["analysis"]` through `model.save_spec`; no `.rsx` schema change in `document.py` is needed beyond the meta dict).
  - Higher orders, Add Term, error terms, Slack and special-quartic mixture terms, U-pseudo/real equations; combined-design fitting using WP-A's `terms_combined.py`.
  - One-categoric ANOVA with LSD.
  - Durbin-Watson, covariance ratio, normal plot of effects, Shapiro-Wilk, residual type and colour-by, covariates in the model matrix.
  - Graphs: Standard Error plot, All Factors, LSD and PI bands, categoric contour, 3D categoric, two-component and Mix×Process graphs, custom contour levels and colour maps, a preferences object (alpha, CI level, TI proportion) exposed from `model.py` for WP-C to read.
- **Verification:** E3, E4, E5, E6, E15, E2 (regression part).

### WP-C: Optimization, Post Analysis, POE, intervals (P1 item 5)

- **Owns:** `core/doe/optimize.py`, new `core/doe/intervals.py` (CI/PI/TI from the public `Fit` attributes `predict_coded`, `ms_res`, `df_res`, `transform`), new `core/doe/poe.py`, new `core/doe/report_post.py` (new prediction and optimization sections; the old functions in `report.py` stay untouched until WP-B removes them), new `core/doe/plots_opt.py`, `ui/dx/optimization.py`, new `tests/test_doe_opt2.py`.
- **Scope:**
  - Goals: Cpk and equal-to.
  - Use Interval (CI/PI/TI, one-sided) in numerical and graphical optimization.
  - Standard-error models and discrete numeric factors; options for starts and duplicate filtering.
  - Ramps and bars at any point.
  - TI in Point Prediction.
  - POE: factor SDs are stored in `doc.optimization["poe_sd"]`, which avoids `document.py`. Deliver the POE surface (a function plus a figure for WP-A to place under the analysis graphs), the POE goal and the POE line in prediction.
  - Confirmation with individual runs.
- **Verification:** E2 (optimization), E11, E12, E13, plus a Monte-Carlo POE check.

**Merge order:** WP-B and WP-C first (they do not touch the window), then WP-A, which wires their new views into `window.py`.

## 6. Notes and caveats

- The Stat-Ease doc pages for the optimal builder, model selection and diagnostics are short. Defaults stated here as (exp.) need confirmation in a real copy of Design-Expert: "all hierarchical" selection, Shapiro-Wilk on the half-normal plot, the exact TI and POE formulas, and alpha in/out defaults.
- Lenth: the app uses a Bonferroni limit t(1 − α/(2m), m/3), which matches Design-Expert's Pareto "Bonferroni limit". Lenth's own SME (γ = (1 + 0.95^{1/m})/2) is slightly lower (13.70 vs 13.77 for E3). Keep the Bonferroni limit.
- Stat-Ease sources used:
  - statease.com/docs: v23.1 screen-tips index, advanced-topics index, optimality criteria, optimal exchange methods, design augmentation, propagation of error, logistic regression, numerical and graphical optimization criteria, point prediction, confirmation, evaluation pages.
  - v25.0: designs, rsm, mixture, custom, optimal, space-filling, analysis configuration, changelog, what's new, tutorials.
  - v13 what's new; the product page.
- Licence sources: PyPI (dexpy, pyDOE3, OApackage), the dexpy GitHub README, and the CRAN pages for skpr, rsm, AlgDesign, FrF2, daewr and mixexp.
