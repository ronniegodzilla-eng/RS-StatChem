# RS-StatChem Statistics module vs IBM SPSS Statistics 29–31: gap analysis

Date: 2026-10-27. Code inspected read-only at `/Users/ronisaputra/Aplikasi statistik` (v0.8.2).

## 0. Method and evidence

**RS-StatChem side.** The merged menu tree was not inferred from the README. It was dumped by running
`rsstatchem.ui.data_editor.menu_tree()` headlessly (`QT_QPA_PLATFORM=offscreen`; all 10 extension
modules loaded, none failed). File/Edit/View/Window/Help are hard-coded in
`DataEditorWindow._build_menus()` (`rsstatchem/ui/data_editor.py` l.280–345). Every option listed as
present comes from the dialog source (`add_option_button`, `SubDialog.check/radio/number`) and from
the matching `core/stats/*.py` function. A keyword sweep over `core/stats`, `ui/dialogs`, `core/graph`,
`ui/graph`, `ui/output_viewer.py`, `core/report.py`, `io/` backs up each "missing" entry (for example
no McNemar, Kappa, Kaplan, ICC, Dunnett, Duncan, Tamhane, GEE, Poisson, Quantile, Ridge, Imputation
(MI), ARIMA, Custom Tables, Multiple Response, Find/Replace or Syntax editor anywhere in the
Statistics code).

**Legend.** **P** = present (SPSS-equivalent options) · **Pa** = partial (missing options listed) ·
**M** = missing (no menu entry) · **PH** = placeholder (entry shown but disabled) · **X** = present
but goes beyond SPSS · **IMP** = impractical / out of scope (reason given).

**SPSS reference (all fetched 2026-10-27).** The 30.0 module manuals (outlines extracted from the PDFs):
- Base: https://www.ibm.com/docs/en/SSLVMB_30.0.0/pdf/IBM_SPSS_Statistics_Base.pdf
- Advanced Statistics: https://www.ibm.com/docs/en/SSLVMB_30.0.0/pdf/IBM_SPSS_Advanced_Statistics.pdf
- Regression: https://www.ibm.com/docs/en/SSLVMB_30.0.0/pdf/IBM_SPSS_Regression.pdf
- Core System (Data/Transform/Utilities/Viewer/Syntax): https://www.ibm.com/docs/en/SSLVMB_30.0.0/pdf/IBM_SPSS_Statistics_Core_System_User_Guide.pdf
- Custom Tables: https://www.ibm.com/docs/en/SSLVMB_30.0.0/pdf/IBM_SPSS_Custom_Tables.pdf
- Categories: https://www.ibm.com/docs/en/SSLVMB_30.0.0/pdf/IBM_SPSS_Categories.pdf
- Forecasting: https://www.ibm.com/docs/en/SSLVMB_30.0.0/pdf/IBM_SPSS_Forecasting.pdf
- Missing Values: https://www.ibm.com/docs/en/SSLVMB_30.0.0/pdf/IBM_SPSS_Missing_Values.pdf
- Bootstrapping: https://www.ibm.com/docs/en/SSLVMB_30.0.0/pdf/IBM_SPSS_Bootstrapping.pdf
- Data Preparation: https://www.ibm.com/docs/en/SSLVMB_30.0.0/pdf/IBM_SPSS_Data_Preparation.pdf
- The other module manuals follow the same URL pattern but were not parsed: Exact Tests (the PDF
  fetched as an image-only cover), Decision Trees, Neural Network, Complex Samples, Conjoint,
  Direct Marketing.

Release notes:
- v27.0.1 (Proportions, McDonald's omega, correlation CIs, ANOVA contrast CIs): https://community.ibm.com/community/user/ai-datascience/blogs/brent-gardiner/2020/11/03/whats-new-in-spss-statistics-27010
- v27 (Power Analysis, weighted kappa, quantile regression, effect sizes): https://community.ibm.com/community/user/blogs/mathew-mathew1/2020/06/10/whats-new-in-spss-statistics-27 · https://www.spss-tutorials.com/spss-27-review-and-new-features
- v28 (Meta-analysis, power curves, Ratio PRB, *Compare simple main effects* in EM Means, one-sided t-test, Relationship Map): https://community.ibm.com/community/user/blogs/brent-gardiner/2021/05/24/whats-new-in-spss-statistics-28
- v29 (Linear Elastic Net/Lasso/Ridge, parametric AFT, pseudo-R² and ICC in MIXED/GENLINMIXED, violin plot): https://community.ibm.com/community/user/blogs/kennia-garcia/2022/08/08/whats-new-in-spss-statistics-29
- v29.0.1 (shared frailty, PRESS, Youden in ROC, EBLUP export, Percentiles dialog, Cox time-dependent dialog, Overview tab): https://community.ibm.com/community/user/blogs/indira-kumar2/2023/04/14/whats-new-in-spss-statistics-version-2901?hlmlt=QT
- v30 (Bland-Altman, Normality Analysis): https://community.ibm.com/community/user/blogs/indira-kumar2/2024/09/22/whats-new-in-ibm-spss-statistics-3000
- v31 (MARS under Generalized Linear Models, Chi-Square menu item, distance correlation, proximity mapping, t-test options): https://community.ibm.com/community/user/blogs/anu-j-r/2025/06/09/power-of-ibm-spss-statistics-version-31
- v32 (2026: Mediation Analysis, VAR, permutation tests, "Compare Means and Proportions" menu name): https://community.ibm.com/community/user/blogs/anu-j-r/2026/04/22/ibm-spss-statistics-3200-release

The reference tree below is SPSS 30/31 with every module installed. Extensions, plus v32-only
items, are shown only where they matter.

---

## 1. Menu tree and status

### File
| SPSS item | Status | Notes / evidence |
|---|---|---|
| New ▸ Data | P | `act_new` |
| New ▸ Syntax / Output / Workbook / Script | M | No syntax or workbook documents. The Output Viewer is a singleton window. |
| Open ▸ Data | P | .rsd, .sav, csv/txt/tsv/dat, xlsx/xlsm (`io/files.py OPEN_FILTER`). Also opens .rsg graphs and .rsc drawings. |
| Open ▸ Syntax / Output / Workbook / Internet data | M | Output cannot be saved or reopened (see Output Viewer). |
| Import Data ▸ Text, Excel | P | Text wizard `TextImportDialog`, Excel. |
| Import Data ▸ SAS / Stata / dBase / Lotus / SYLK / Database (ODBC) / Cognos / TM1 | M | `pyreadstat` is already bundled and can read .dta/.sas7bdat/.xpt/.por (quick win). ODBC, Cognos and TM1 are IMP (server connectors). |
| Close, Save, Save As | P | Save As offers .rsd and .sav. |
| Save All Data, Mark File Read Only, Revert to Saved, Rename Dataset | M | Low priority. |
| Export ▸ CSV/Text, Excel | P | `act_export_csv/excel` |
| Export ▸ SAS / Stata / fixed text / tab / database | M | Stata and SAS XPORT via pyreadstat are easy. Database export is IMP. |
| Display Data File Information (working / external) | M | Equivalent to a Codebook-lite. Easy. |
| Cache Data, Collect Variable Information, Stop Processor, Switch Server, Repository | IMP | Server/processor features. |
| Print Preview / Print (data) | M | Low priority. |
| Recently Used Data | P | `recent_menu` |
| Recently Used Files, Welcome dialog | M / X | A launcher (module picker) exists instead of the Welcome dialog. |
| Exit | P | |

### Edit
| SPSS item | Status | Notes |
|---|---|---|
| Undo / Redo | P | QUndoStack |
| Cut / Copy / Paste / Clear | P | |
| Copy with Variable Names / Labels, Paste Variables, Paste with Variable Names | M | |
| Insert Variable / Insert Cases | P | |
| Search Data (Find), Find Next, Replace | M | No Find/Replace anywhere in the Data Editor. High usability value. |
| Go to Case | P | Ctrl+G |
| Go to Variable, Go to Imputation | M | |
| Options… (global settings: output labels/values, decimal display, default p-value format, table look, etc.) | M | |

### View
| SPSS item | Status | Notes |
|---|---|---|
| Value Labels toggle | P | |
| Variables (toggle Data/Variable View) | P | Ctrl+T |
| Status Bar, Toolbars, Menu Editor, Fonts, Grid Lines, Customize Variable View, Mark Imputed Data | M | Mark Imputed Data depends on MI. The rest is cosmetic. |
| Overview tab (29.0.1) | M | Dataset summary with missing-value overview. |

### Data
| SPSS item | Status | Notes |
|---|---|---|
| Define Variable Properties… | M | Scans values and proposes labels and measurement level. |
| Set Measurement Level for Unknown… | M | |
| Copy Data Properties… | M | |
| New Custom Attribute… | M | |
| Define Date and Time… (Define Dates) | M | Needed by Forecasting. |
| Define Multiple Response Sets… | M | Also needed by Custom Tables. |
| Validation ▸ Load Predefined Rules / Define Rules / Validate Data | M | Data Preparation module. Low priority. |
| Identify Duplicate Cases… | P | |
| Identify Unusual Cases… | M | Data Preparation module (anomaly index). Low priority. |
| Compare Datasets… | M | |
| Sort Cases… | P | |
| Sort Variables… | M | Easy. |
| Transpose… | P | |
| Adjust String Widths Across Files | M | Low. |
| Merge Files ▸ Add Cases / Add Variables | P | |
| Restructure… | P | Wizard covering vars→cases and cases→vars. |
| Rake Weights… (extension) | M | Low. |
| Propensity Score Matching…, Case Control Matching… | M | PSM is common in health theses. Medium priority. |
| Aggregate… | P | |
| Orthogonal Design ▸ Generate / Display | M | Conjoint. Low. |
| Split into Files | M | Low. |
| Copy Dataset | M | Easy. |
| Split File… | P | Compare / separate modes. |
| Select Cases… | P | all / if / random sample / range / filter variable. Output filter / copy / delete. |
| Weight Cases… | P | Frequency weights, rounded to whole cases (`core/selection.py` l.61–99). SPSS uses non-integer weights in most procedures, so this is Pa. |

### Transform
| SPSS item | Status | Notes |
|---|---|---|
| Compute Variable… | Pa | Expression builder plus If. Only 45 functions (`core/expression.py FUNCTIONS`). Missing: all date/time functions (DATE.DMY, XDATE.*, DATEDIFF, CTIME…), most CDF/PDF/IDF/RV families (only normal, t, chi², F CDF; IDF normal/t; RV normal/uniform), string functions (CHAR.INDEX, CHAR.SUBSTR, REPLACE, LPAD/RPAD, CHAR.RINDEX…), LAG, $CASENUM, MEAN.n-style minimum-valid suffix, CFVAR, SUM.n, MISSING/VALUE are present. |
| Programmability Transformation | IMP | Needs the Python plug-in. |
| Count Values within Cases… | P | |
| Shift Values… | P | |
| Recode into Same / Different Variables… | P | |
| Automatic Recode… | P | |
| Create Dummy Variables (SPSS 26+) | M | Easy and useful for regression. |
| Visual Binning… | P | |
| Optimal Binning… | M | Data Preparation (MDLP). Low. |
| Prepare Data for Modeling ▸ Interactive / Automatic / Backtransform | M | Low. |
| Rank Cases… | P | All SPSS rank functions including Savage and normal scores. |
| Date and Time Wizard… | M | |
| Create Time Series… | M | Needed for forecasting (DIFF, SDIFF, MA, LAG, LEAD, CSUM, smoothing). |
| Replace Missing Values… | P | Series mean, mean/median of nearby points, linear interpolation, linear trend. |
| Random Number Generators… | M | Seed and generator setting. Compute has RV.NORMAL/RV.UNIFORM only. |
| Run Pending Transforms | IMP/NA | No deferred-transform model; transforms run immediately. |

### Analyze

#### Reports
| Item | Status | Notes |
|---|---|---|
| Codebook | M | |
| OLAP Cubes | M | |
| Case Summaries | M | Common for listing cases. Easy. |
| Report Summaries in Rows / Columns | M | Low. |

#### Descriptive Statistics
| Item | Status | Notes |
|---|---|---|
| Frequencies… | Pa | See §2. |
| Descriptives… | Pa | See §2. |
| Explore… | Pa | See §2. |
| Crosstabs… | Pa | See §2. Biggest gap in this group. |
| TURF Analysis | M | Low. |
| Ratio… | **PH** | Disabled placeholder (`data_editor.py ANALYZE`). |
| P-P Plots… / Q-Q Plots… | M | Exist only inside Explore and Regression. No stand-alone dialog (distribution choice, transforms). |
| Percentiles… (29.0.1) | M | Custom percentiles, 5 estimation methods, bootstrap CI. |
| Bland-Altman… (30) | M | Agreement analysis. Easy. |
| Normality Analysis (30, built-in extension) | M | Univariate and multivariate normality (Mardia, Henze-Zirkler, Royston). |

#### Bayesian Statistics (SPSS 25+): M (entire submenu)
One Sample Normal, One Sample Binomial, One Sample Poisson, Related Samples Normal, Independent
Samples Normal, Pearson Correlation, Linear Regression, One-way ANOVA, One-way Repeated Measures
ANOVA, Log-Linear Models.

#### Tables
| Item | Status | Notes |
|---|---|---|
| Custom Tables… | M | Table builder: stacking, nesting, layers, summary statistics, totals/subtotals, computed categories, column proportion/means tests. |
| Multiple Response Sets… (Tables) | M | |

#### Compare Means (and Proportions in 29+)
| Item | Status | Notes |
|---|---|---|
| Means… | Pa | 21 cell statistics and ANOVA/eta/linearity present. Only 2 layers (SPSS: unlimited Previous/Next layers). No Style or Bootstrap. |
| One-Sample / Independent-Samples / Paired-Samples Proportions (27.0.1) | M | Wald, Wilson, Clopper-Pearson, Agresti-Coull, Jeffreys CIs; z-tests. |
| One-Sample T Test… | Pa | Missing Bootstrap. Effect sizes (Cohen d, Hedges g) and one-sided p are present. |
| Independent-Samples T Test… | Pa | Levene, Welch row, effect sizes (d, g, Glass Δ) and one-sided p present. Missing Bootstrap and v31 options. |
| Summary Independent-Samples T Test (from means/SD/n) | M | Easy and useful. |
| Paired-Samples T Test… | Pa | Missing Bootstrap and missing-value options. |
| One-Way ANOVA… | Pa | See §2. |

#### General Linear Model
| Item | Status | Notes |
|---|---|---|
| Univariate… | Pa | Full-factorial model only (`core/stats/anova.py` l.252–257). See §2. |
| Multivariate… | Pa | See §2. |
| Repeated Measures… | Pa | See §2. |
| Variance Components… | M | ANOVA, MINQUE, ML, REML. |

#### Generalized Linear Models: M (entire submenu)
Generalized Linear Models (GENLIN), Generalized Estimating Equations (GEE), and Multiple Adaptive
Regression Splines (v31, extension).

#### Mixed Models
| Item | Status | Notes |
|---|---|---|
| Linear… | Pa | See §2. |
| Generalized Linear… (GLMM) | M | |

#### Correlate
| Item | Status | Notes |
|---|---|---|
| Bivariate… | Pa | Pearson, Kendall tau-b, Spearman, one- and two-tailed, flag, means/SD, pairwise/listwise. Missing: Confidence Interval sub-dialog (27.0.1; Fisher z, bias adjustment), cross-products and covariances, Bootstrap, distance correlation (v31). |
| Partial… | Pa | Missing means/SD, missing-value options and Bootstrap. |
| Distances… | **PH** | Disabled placeholder. |
| Canonical Correlation (27+) | M | |

#### Regression
| Item | Status | Notes |
|---|---|---|
| Automatic Linear Modeling… | M | Low priority. |
| Linear… | Pa | See §2. No hierarchical blocks (needed for MRA moderation). |
| Curve Estimation… | Pa | All 11 models, constant, plot, ANOVA, Time as X. Missing Save (predicted, residuals, prediction intervals, predict cases beyond the end). |
| Partial Least Squares… (PLS regression) | M | PLS-SEM exists (X), but PLS regression (PLS1/PLS2, VIP) does not. |
| Binary Logistic… | Pa | See §2. |
| Multinomial Logistic… | Pa | See §2. |
| Ordinal… | Pa | See §2. |
| Probit… | M | Dose-response probit/logit with LD50 and fiducial limits. Used in toxicology and agriculture theses. |
| Nonlinear… | Pa | Aliased to the Origin-style `NonlinearFitDialog` (`ui/menus.py ALIASES`). Missing SPSS NLR features: parameter dialog with start values, loss function, parameter constraints, bootstrap SE, Save (predicted, residuals, derivatives), conditional logic. |
| Weight Estimation… | M | WLS with power-weight search. |
| 2-Stage Least Squares… | M | |
| Quantile… (27/28) | M | |
| Optimal Scaling (CATREG)… | M | Categories module. |
| Linear OLS Alternatives ▸ Ridge / Lasso / Elastic Net (29) | M | |
| Kernel Ridge Regression (extension, in the 30 manuals) | M | Low. |

#### Loglinear: M
General, Logit, Model Selection (hierarchical loglinear / HILOGLINEAR).

#### Neural Networks: M
Multilayer Perceptron, Radial Basis Function.

#### Classify
| Item | Status | Notes |
|---|---|---|
| TwoStep Cluster… | M | |
| K-Means Cluster… | Pa | Iterate (max iterations, convergence, running means), Save (membership, distance), Options (initial centres, ANOVA, case info, distances) present. Missing: read/write initial and final centres file, Classify-only method, missing-value options. |
| Hierarchical Cluster… | Pa | 7 linkages, 6 interval measures, 5 standardisations, schedule, proximity matrix, dendrogram, membership save. Missing: binary and count measures, value transforms (absolute, reverse sign, rescale), icicle plot, cluster-by-variables option. |
| Cluster Silhouettes (extension) | M | |
| Tree… (CHAID, Exhaustive CHAID, CRT, QUEST) | M | |
| Discriminant… | Pa | See §2. |
| Nearest Neighbor… | M | |
| ROC Curve… | P/Pa | Larger/smaller positive, DeLong or Hanley-McNeil SE, CI level, coordinates, Youden. Present at Analyze▸ROC Curve (top level); SPSS 30 has it under Classify. |
| ROC Analysis… (26+) | M | Paired/independent AUC comparison, precision-recall curve, grouping variable. |

#### Dimension Reduction
| Item | Status | Notes |
|---|---|---|
| Factor… | Pa | See §2. |
| Correspondence Analysis… | M | |
| Optimal Scaling (CATPCA / MCA / OVERALS)… | M | |

#### Scale
| Item | Status | Notes |
|---|---|---|
| Reliability Analysis… | Pa | Cronbach α plus an Indonesian item-validity table (r item-total vs r-table, X). See §2. |
| Weighted Kappa… (27) | M | |
| Multidimensional Unfolding (PREFSCAL) | M | Low. |
| Multidimensional Scaling (PROXSCAL) / (ALSCAL) | M | |

#### Nonparametric Tests
| Item | Status | Notes |
|---|---|---|
| One Sample… (NPTESTS) | M | Binomial, chi-square, K-S, Wilcoxon signed-rank one-sample, runs, with Model Viewer-style hypothesis summary. |
| Independent Samples… (NPTESTS) | M | M-W, K-S, Moses, Wald-Wolfowitz, Hodges-Lehmann, K-W with Dunn pairwise and Bonferroni, median test, Jonckheere-Terpstra. |
| Related Samples… (NPTESTS) | M | Wilcoxon, sign, McNemar, marginal homogeneity, Friedman with pairwise, Kendall W, Cochran Q, Hodges-Lehmann. |
| Quade Nonparametric ANCOVA (extension) | M | Low. |
| Legacy Dialogs ▸ Chi-square / Binomial / Runs / 1-Sample K-S / 2 Independent / K Independent / 2 Related / K Related | Pa | All 8 entries exist. Missing options in §2. |

#### Forecasting: M (entire submenu)
Create Traditional Models (Expert Modeler, Exponential Smoothing, ARIMA), Apply Traditional Models,
Create/Apply Temporal Causal Models, Seasonal Decomposition, Spectral Analysis, Sequence Charts,
Autocorrelations, Cross-Correlations. VAR is v32.

#### Survival: M (entire submenu)
Life Tables, Kaplan-Meier, Cox Regression, Cox w/ Time-Dep Cov, Parametric Accelerated Failure Time
(29), Parametric Shared Frailty (29.0.1).

#### Multiple Response: M
Define Variable Sets, Frequencies, Crosstabs.

#### Missing Value Analysis…: M
Patterns, EM/regression estimates, Little's MCAR test.

#### Multiple Imputation: M
Analyze Patterns, Impute Missing Data Values (FCS/PMM/monotone), and pooled output across procedures.

#### Complex Samples: M
Plan, Select sample, Frequencies, Descriptives, Crosstabs, Ratios, GLM, Logistic, Ordinal, Cox.

#### Others in the SPSS Analyze menu
| Item | Status | Notes |
|---|---|---|
| Simulation… | M | Monte Carlo on models. Low. |
| Quality Control ▸ Control Charts / Pareto Charts | M | Medium for industrial/chem theses (X-bar/R/S, p, np, c, u; capability indices). |
| Spatial and Temporal Modeling | IMP | Needs map shapefiles and geospatial stack. |
| Direct Marketing ▸ Choose Technique (RFM, clustering, prospect profiles, postal-code response, propensity to purchase, control package test) | M | Low priority. RFM is easy. |
| Meta Analysis ▸ Continuous (raw / effect size), Binary (raw / effect size), Meta-Regression (28) | M | |
| Power Analysis ▸ Means (1-sample, paired, independent t, one-way ANOVA), Proportions (1-sample, related, independent binomial), Correlations (Pearson, Spearman, partial), Regression (univariate linear) (27/28) | M | Sample-size questions in proposals. High value. |
| Conjoint | M | Syntax-only in SPSS. Low. |

#### RS-StatChem items with no SPSS menu counterpart (X)
- Mediation & Moderation ▸ PROCESS Models 1/4/7/14. SPSS 32 adds a native Mediation Analysis.
- Structural Equation Modeling ▸ CB-SEM (AMOS-like), PLS-SEM (SmartPLS-like).
- Regression ▸ classical assumption tests (`Assumptions…` sub-dialog: K-S/Shapiro on residuals, VIF,
  Glejser/BP/White, DW/BG/runs).
- Item-validity r-table in Reliability.

### Direct Marketing (top menu in SPSS): M
Same items as above. Low priority.

### Graphs
The Statistics module's Graphs menu is Graph Studio, an OriginLab-style tool with 29 templates (see
`core/graph/build.py TEMPLATES`). The PlotSpec has optional group-by and summary (mean ± SD/SE/CI,
median, sum, count by category), so many Chart Builder charts can be built.

| SPSS item | Status | Notes |
|---|---|---|
| Chart Builder: Bar (simple/clustered/stacked) | P | column/bar and stacked variants with summaries |
| Bar 3-D | M | Low (SPSS discourages it) |
| Line (simple/multiple), Area (simple/stacked) | P | |
| Pie / Polar | P | |
| Scatter/Dot: simple, grouped, 3-D, simple 3-D grouped | P | |
| Scatter/Dot: scatterplot matrix, drop-line, summary point, simple dot plot | M | The scatter matrix is common for correlation chapters. |
| Histogram: simple (+normal curve) | P | |
| Histogram: stacked, frequency polygon, population pyramid | M | |
| High-Low: high-low-close, simple/clustered range bar, difference area | M | |
| Boxplot: simple, clustered | P | |
| Boxplot: 1-D | M | Trivial |
| Dual Axes | P | double_y |
| Panelling by row/column variables, error bars of means with CI by group | Pa | Error bars by summary are present. Row/column panel variables are missing. |
| Graphboard Template Chooser (incl. violin, 29) | M | Violin plot missing. |
| Relationship Map (28) | M | Low |
| Weibull Plot, Compare Subgroups, Regression Variable Plots | M | Regression variable plots: medium. |
| Legacy Dialogs (Bar, 3-D Bar, Line, Area, Pie, High-Low, Boxplot, Error Bar, Population Pyramid, Scatter/Dot, Histogram) | Pa | Covered by Graph Studio except the items marked M above. |
| Chart Editor (edit any chart in output) | Pa | Graph Studio windows are editable, but charts that analyses emit into the Output Viewer are static PNGs (`Section.image`). |

### Utilities
| SPSS item | Status | Notes |
|---|---|---|
| Variables… (variable information browser) | M | Easy |
| OMS Control Panel / OMS Identifiers | IMP | Output Management System is syntax-driven. A simple "export all tables of type X to Excel" could stand in. |
| Scoring Wizard / Merge Model XML | IMP | PMML model-scoring infrastructure. |
| Data File Comments… | M | Easy |
| Define Variable Sets / Use Variable Sets / Show All Variables | M | Useful with wide questionnaires. |
| Spelling… | IMP/low | |
| Run Script… / Production Facility / Map Conversion Utility | IMP | Need the Python/Basic script engine, batch jobs and geospatial maps. |
| Extensions ▸ Extension Hub / Custom Dialog Builder / Install bundle | IMP | Need the Python/R plug-ins. |

### Window
| Item | Status | Notes |
|---|---|---|
| List of open windows, Output Viewer, Modules | P/X | |
| Split (Data View panes), Minimize All Windows, Go to Designated Viewer/Syntax, Reset Dialog Sizes and Positions | M | Low |

### Help
Topics, Tutorial, Case Studies, Statistics Coach, Command Syntax Reference, Algorithms, and
context-sensitive dialog Help buttons are all **M**. Only Check for Updates and About exist. A
"Statistics Coach" (method chooser), written in Indonesian, would serve thesis students well
(low cost).

### Output Viewer (`ui/output_viewer.py`, `core/report.py`)
| SPSS feature | Status | Notes |
|---|---|---|
| Outline tree, select/delete items, clear all | P | |
| Copy, Copy Special (HTML / image / text) | P | |
| Export Word (.docx), HTML, PDF | P | |
| Export Excel, PowerPoint, plain text | M | Excel is the high-value one (openpyxl is bundled). |
| Save / open output document (.spv equivalent) | M | Output is lost on close. High priority for thesis work. |
| Pivot table editing: transpose | P | `act_transpose` |
| Pivoting trays (move rows/columns/layers), hide/show rows or columns, sort, cell editing, footnote editing, decimal changes | M | |
| TableLooks / APA style switch | M | APA-style table export is common in thesis requirements. |
| Hide/show items in outline, promote/demote, insert heading/title/text | M | |
| Syntax log | Pa | Every analysis writes an SPSS-like pseudo-syntax `Log` item (`Section.log`), but it cannot be run. |
| Syntax Editor / Paste button / run syntax | M → IMP (full) | A full SPSS command-language interpreter is impractical. A feasible substitute: record each dialog run as a JSON "RS job" that can be replayed and saved, with optional SPSS-syntax-like text for documentation. |
| Workbook mode (28), AI Output Assistant (31.0.1), Curated Help (31) | IMP / M | Workbook and AI assistant: out of scope. Curated help, i.e. rule-based Indonesian interpretation text under key tables, is feasible and valuable. |
| Charts editable in viewer (Chart Editor) | M | Output images are static PNGs. |

---

## 2. Missing sub-dialog options in existing procedures

Present options are listed briefly; the **missing** column is the actionable part. "Bootstrap"
refers to the SPSS Bootstrap sub-dialog (simple/stratified resampling, percentile/BCa, number of
samples, seed). It is absent from every Base procedure here; RS-StatChem bootstraps only inside
PROCESS, CB-SEM and PLS-SEM.

| Procedure (file) | Present | Missing vs SPSS |
|---|---|---|
| **Frequencies** (`ui/dialogs/descriptive.py`, `core/stats/descriptive.py`) | Stats: quartiles, mean, median, mode, sum, SD, variance, range, min, max, SE mean, skewness, kurtosis. Charts: none/bar/pie/histogram + normal curve. Display tables toggle. | Statistics: cut points for n equal groups, custom percentile list, "values are group midpoints". Charts: chart values (frequencies vs percentages). **Format** sub-dialog (order by ascending/descending values or counts; compare variables vs organise output by variable; suppress tables with more than n categories). Style. Bootstrap. |
| **Descriptives** | Mean, sum, SD, var, range, min, max, SE mean, kurtosis, skewness. Save z-scores. | Display order (variable list / alphabetic / ascending or descending means). Bootstrap. |
| **Explore** (`ui/dialogs/explore.py`, `core/stats/explore.py`) | Dependent/Factor lists, Display both/statistics/plots, descriptives with CI, boxplot (factor levels together), histogram, normality plots with tests (K-S Lilliefors, Shapiro-Wilk, Q-Q). | Label Cases by. Statistics: **M-estimators** (Huber, Tukey biweight, Hampel, Andrews), **Outliers** (5 extreme values), **Percentiles**. Plots: boxplot "dependents together" / none, **stem-and-leaf**, **spread vs level with Levene** (power estimation / transformed / untransformed), detrended Q-Q. Options: missing values (listwise / pairwise / report values). Bootstrap. |
| **Crosstabs** (`ui/dialogs/explore.py`, `core/stats/crosstabs.py`) | Rows × columns, chi-square (Pearson, LR, Fisher, linear-by-linear), phi/Cramér V, contingency coefficient; cells: observed, expected, row/column/total %. | **Layers** (and "display layer variables in table layers"), clustered bar charts, suppress tables. Statistics: correlations (Pearson r, Spearman), nominal **lambda, uncertainty coefficient**, ordinal **gamma, Somers' d, Kendall tau-b, tau-c**, nominal-by-interval **eta**, **kappa**, **risk** (odds ratio, relative risk with CI), **McNemar**, **Cochran's and Mantel-Haenszel** (Breslow-Day, Tarone, common OR test value). Cells: **residuals** (unstandardized, standardized, adjusted standardized), **z-test compare column proportions** (Bonferroni), hide small counts, noninteger-weight rounding options. Format: row order ascending/descending. **Exact** (exact / Monte Carlo). Bootstrap. |
| **Means** (`ui/dialogs/logistic.py MeansDialog`) | 21 cell statistics, ANOVA and eta, test for linearity, 2 layers. | Unlimited layers (Previous/Next). Style. Bootstrap. |
| **One-Sample / Independent / Paired t** (`ui/dialogs/compare.py`, `core/stats/ttest.py`) | CI level, listwise/analysis-by-analysis (one-sample, independent), Define Groups (values or cut point), effect sizes, one- and two-sided p. | Paired: missing-value options. All three: **Bootstrap**, effect-size standardizer choices (v28+). v31 Independent t options. |
| **One-Way ANOVA** (`compare.py`, `core/stats/anova.py`) | Post hoc: Tukey HSD, LSD, Bonferroni, Sidak, Scheffé, Games-Howell. Options: descriptive, homogeneity (Levene), Brown-Forsythe, Welch, means plot. Effect sizes (η², ε², ω²). Significance level. | **Contrasts** sub-dialog: polynomial (linear, quadratic, cubic…), custom coefficients, contrast CIs (27.0.1). Post hoc: **Duncan, S-N-K, Tukey's-b, R-E-G-W F and Q, Hochberg GT2, Gabriel, Waller-Duncan, Dunnett (control category, 2-sided/< />)**, and with unequal variances **Tamhane T2, Dunnett T3, Dunnett C**. Options: fixed and random effects, missing values. CIs for the effect sizes (SPSS 27+ reports point estimates plus 95% CI; RS-StatChem reports point estimates only). Bootstrap. Duncan, SNK and Dunnett are heavily used in Indonesian agri/food/pharmacy theses. |
| **GLM Univariate** (`compare.py UnivariateDialog`, `anova.univariate`) | DV, fixed factors, covariates, full factorial with Type III SS, post hoc (same 6 methods), options: descriptives, effect size (partial η²), Levene. | **Random factor(s)**, **WLS weight**. **Model** sub-dialog: custom terms (main effects only, interactions, nested), sum-of-squares Type I/II/III/IV, include intercept. This is critical: RAK/RCBD designs need a block main effect without interaction, and that cannot be done now. **Contrasts** (deviation, simple, difference, Helmert, repeated, polynomial). **Plots** (profile plots, error bars, line/bar). **EM Means** (with compare main effects, compare simple main effects (28), CI adjustment). There is no EM Means button on Univariate although Multivariate has one. **Save** (predicted, residuals, Cook's distance, leverage, standardized/studentized/deleted residuals). Options: parameter estimates (incl. HC0–HC4 robust SE), **observed power**, contrast coefficient matrix, general estimable function, lack-of-fit, spread-vs-level and residual plots, **heteroskedasticity tests** (modified BP, BP, F test, White). Post hoc: the 13 extra methods listed under One-Way ANOVA. Bootstrap. |
| **GLM Multivariate** (`ui/dialogs/glm.py`, `core/stats/manova.py`) | DVs, factors, covariates, multivariate tests (Pillai, Wilks, Hotelling, Roy), Box's M, Levene, between-subjects tests, post hoc, EM Means (overall, effects, compare main effects, LSD/Bonferroni/Sidak), options. | Model (custom / SS type), Contrasts, **Plots** (profile), **Save**, WLS weight. Options: parameter estimates, SSCP matrices, residual SSCP, Bartlett's sphericity, observed power, transformation matrix. Extra post hoc methods. Bootstrap. |
| **GLM Repeated Measures** (`glm.py`, `core/stats/repeated.py`) | Define factors (multiple within factors, measures), between factors, covariates, Mauchly, GG/HF/LB, multivariate tests, within-subject polynomial contrasts, Levene, profile plots, post hoc for between factors, EM Means with pairwise comparisons. | **Model** (custom within/between terms, SS type). **Contrasts** (change from polynomial to deviation/simple/difference/Helmert/repeated). **Save**. Options: parameter estimates, observed power, SSCP, transformation matrix, residual plots. Compare simple main effects (28). Error bars in profile plots. Extra post hoc. |
| **Linear Mixed Models** (`glm.py`, `core/stats/mixed.py`) | Subject, DV, factors, covariates. Fixed: full factorial or main effects. Random: intercept + covariate slopes, VC or UN. REML/ML. Descriptives, fixed estimates, covariance tests. | **Repeated** variable with residual covariance types (AR(1), ARH(1), CS, CSH, diagonal, Toeplitz, UN…), multiple Subjects/nesting, multiple random-effect blocks, factor random effects, custom fixed model builder, ~17 covariance structures (only VC/UN now). **EM Means**. **Save** (fixed predicted, predicted, residuals, EBLUPs export (29.0.1)). Estimation: iterations, convergence, Satterthwaite vs Kenward-Roger, scoring steps. Statistics: correlations of estimates, covariances of random effects/residuals, contrast coefficient matrix, pseudo-R²/ICC (29). WLS weight. |
| **Bivariate / Partial correlation** (`ui/dialogs/regression.py`, `core/stats/correlate.py`) | See §1. | **Confidence interval** sub-dialog (Fisher z, bias adjustment, Kendall/Spearman CI methods). Cross-product deviations and covariances. Partial: means/SD, missing options. Bootstrap. Point-biserial is just Pearson; a convenience note would do. |
| **Linear Regression** (`regression.py`, `core/stats/regression.py`) | Methods enter/stepwise/forward/backward, PIN/POUT. Statistics: estimates, CI, model fit, R² change, descriptives, collinearity (tolerance/VIF), DW. Plots: histogram, P-P, ZRESID vs ZPRED. Save: unstd/std predicted, unstd/std residuals. Classical assumption tests (X). | **Blocks (Previous/Next)**: hierarchical regression with R² change per block, the standard way to test moderation (MRA) in Indonesian theses. **Remove** method. **Selection Variable** + rule. **Case Labels**. **WLS Weight**. Statistics: covariance matrix, **part and partial correlations**, **casewise diagnostics** (outliers > n SD / all cases), collinearity eigenvalues/condition index/variance proportions. Plots: free Y/X builder (DEPENDNT, *ZPRED, *ZRESID, *DRESID, *ADJPRED, *SRESID, *SDRESID), **partial regression plots**, scatterplot per block. Save: adjusted predicted, SE of mean predictions, **studentized and deleted residuals**, distances (**Mahalanobis, Cook's, leverage**), prediction intervals (mean and individual), influence (DfBeta(s), standardized DfBeta(s), DfFit, covariance ratio), coefficient statistics to dataset. Options: use F value instead of probability, include constant, missing values (pairwise / replace with mean). PRESS (29.0.1). Style. Bootstrap. Robust (HC) SE are only in PROCESS. |
| **Binary Logistic** (`ui/dialogs/logistic.py`, `core/stats/logistic.py`) | Methods enter/forward/backward LR/Wald. Categorical: indicator/deviation, first/last reference. Save: probabilities, group. Options: Hosmer-Lemeshow, iteration history, CI for Exp(B), each step/last step, PIN/POUT, cut-off, max iterations, constant. | **Blocks** (Previous/Next) and **interaction terms (>a*b>)** in the covariate list. Forward/backward **Conditional**. Contrasts: **simple, difference, Helmert, repeated, polynomial**. Selection variable. Save: residuals (unstd, logit, studentized, standardized, deviance), influence (Cook's, leverage, DfBeta). Options: **classification plots**, **casewise listing of residuals**, correlations of estimates. Bootstrap. Note: SPSS 28 changed the default reference category to First. |
| **Multinomial Logistic** | Reference category (first/last/custom), factors/covariates, GoF, classification, LR tests, CI. Save. | **Model** (custom terms, stepwise forward/backward entry). Statistics: pseudo-R², step summary, model-fitting information, information criteria, cell probabilities, monotonicity measures, asymptotic correlations/covariances. **Criteria** (iterations, step-halving, convergence, singularity). **Options** (dispersion scale: deviance/Pearson; stepwise options). Save: estimated response probabilities, predicted category, predicted/actual category probability, export model. |
| **Ordinal (PLUM)** | Links logit/probit/cloglog/nloglog/cauchit, CI, GoF, test of parallel lines, classification, save est/pre/max probabilities. | **Location** (custom model terms), **Scale** model (heteroscedastic), Options (iterations, step-halving, convergence, singularity tolerance). Output: summary statistics (pseudo R²), parameter estimates on/off, asymptotic correlation/covariance, cell information. Save "actual category probability" and predicted category. Print iteration history. |
| **Curve Estimation** | 11 models, constant, plot, ANOVA, upper bound, time. | **Save** (predicted values, residuals, prediction intervals with level, predict through date/observation). |
| **K-Means / Hierarchical** | See §1. | See §1. |
| **Discriminant** (`ui/dialogs/classify.py`, `core/stats/classify.py`) | Grouping range, enter/stepwise (Wilks, F to enter/remove), means, univariate ANOVA, Box's M, Fisher and unstandardized coefficients, priors equal/size, LOO, save group/scores/probabilities. | Selection variable. Stepwise **methods** (unexplained variance, Mahalanobis distance, smallest F ratio, Rao's V) and "use probability of F", summary of steps, F for pairwise distances. Statistics matrices: within-groups correlation/covariance, separate-groups covariance, total covariance. Classify: use separate-groups covariance, **Plots** (combined groups, separate groups, territorial map), casewise results, summary table, replace missing with mean. Export model. Bootstrap. |
| **Factor** (`ui/dialogs/multivariate.py`, `core/stats/factor.py`) | Descriptives (univariate, initial, coefficients, significance, determinant, KMO/Bartlett, inverse, reproduced, anti-image). Extraction PC/PAF/ML, eigen or fixed, scree, unrotated. Rotation none/quartimax/varimax/equamax/oblimin(δ)/promax(κ), loading plot. Scores (regression/Bartlett/A-R, coefficient matrix). Sort, suppress. | **Extraction methods**: unweighted least squares, generalized least squares, alpha factoring, image factoring. **Analyze covariance matrix**. Selection variable. Options: missing values pairwise and replace with mean. Parallel analysis (not in SPSS, but often requested) is optional. |
| **Reliability** (`ui/dialogs/explore.py`, `core/stats/reliability.py`) | α, standardized α, item/scale/if-deleted descriptives, inter-item correlations, r-table validity (X). | **Model**: **Split-half** (Spearman-Brown, Guttman split-half), **Guttman λ1–λ6**, **Parallel**, **Strict parallel**, **McDonald's omega** (27.0.1). Statistics: inter-item covariances, **summaries** (means, variances, covariances, correlations), **ANOVA table** (F test / Friedman χ² / Cochran χ²), **Hotelling's T²**, **Tukey's test of additivity**, **intraclass correlation coefficient** (one-way random, two-way random, two-way mixed; consistency/absolute agreement; test value; CI). Scale label. |
| **Nonparametric legacy** (`ui/dialogs/nonparametric.py`, `core/stats/nonparametric.py`) | Chi-square (expected values list), binomial (test prop, cut point, exact sig.), runs (median/mean), 1-sample K-S (normal only, Lilliefors), Mann-Whitney (exact p when small), Kruskal-Wallis, Wilcoxon and sign, Friedman, Kendall W, Cochran Q. | Chi-square: **expected range** (get from data / use specified range). Runs: **mode** and **custom** cut point. **1-sample K-S**: **uniform, Poisson, exponential** distributions, plus the v28 simulation/Lilliefors option dialog. **2 Independent**: **K-S Z, Moses extreme reactions, Wald-Wolfowitz runs**. **K Independent**: **median test, Jonckheere-Terpstra**. **2 Related**: **McNemar, marginal homogeneity**. **Options** button on all: descriptive and quartiles, missing values (test-by-test / listwise). **Exact…** button (asymptotic / Monte Carlo with CI level and samples / exact with time limit). Effect size r for M-W/Wilcoxon is commonly requested by examiners. |
| **ROC Curve** | See §1. | Exclude/treat user-missing, and the separate **ROC Analysis** procedure (AUC comparison, precision-recall, group comparisons). |
| **Survival, GENLIN, GEE, GLMM, Probit, WLS, 2SLS, Quantile, Regularized** | — | Entire procedures missing (§1, §3). |

---

## 3. Implementation plan

### 3.0 Integration constraints (from `docs/DEVELOPMENT.md`)
- New features are new modules that export `MENUS` plus `selftest()`, and tests go in
  `tests/test_<area>.py`. New modules must be added to `EXTENSIONS` in `rsstatchem/ui/menus.py`,
  which is integrator-owned, so each work package (WP) reports its module path and the integrator
  adds one line.
- **Replacement mechanism:** `menus._merge_list` replaces any existing entry whose normalized label
  matches, not only placeholders. A WP can therefore ship a brand-new dialog (e.g.
  `ui/dialogs/crosstabs2.py` exporting `{"Analyze/Descriptive Statistics": [("Crosstabs...",
  NewCrosstabsDialog)]}`) and the old one is superseded without editing `data_editor.py`. Caveat:
  "FREQ"/"DESC" in `ANALYZE` are special-cased and not replaceable through MENUS (`_merge_list`
  skips them), so the Frequencies/Descriptives upgrades must edit `ui/dialogs/descriptive.py` in place.
- The contract says modules from earlier rounds "belong to their areas". The WPs below **explicitly
  take ownership** of the listed existing files so that no two WPs touch the same file.
- **Cross-cutting prerequisite (Phase 0, integrator, ~0.5 day):** add new
  `rsstatchem/core/stats/resample.py` with generic case bootstrap (simple/stratified, percentile and
  BCa, seed, vectorised), Monte Carlo/exact helpers, and Rubin's-rules pooling. Add new
  `rsstatchem/ui/dialogs/resample_ui.py` with a reusable **Bootstrap… / Exact…** SubDialog
  builder. WPs import these and never edit them.
- Runtime dependencies: numpy, scipy 1.18, pandas 3.1, statsmodels 0.15 (verified to include PHReg,
  SurvfuncRight/survdiff, GEE, GLM families, OrderedModel, QuantReg, IV2SLS (sandbox), WLS,
  `fit_regularized` elastic net, MICE, ARIMA/SARIMAX/ETS/Holt-Winters/seasonal_decompose/acf/ccf,
  `stats.power`, contingency_tables (mcnemar, StratifiedTable/CMH, SquareTable), inter_rater
  (cohens_kappa with weights, fleiss), meta_analysis (combine_effects), CanCorr, robust RLM,
  BinomialBayesMixedGLM, multitest), plus matplotlib. scipy has `bootstrap`, `permutation_test`,
  `somersd`, `dunnett`, `tukey_hsd`, `median_test`, `page_trend_test`, and noncentral t/F/χ².
  **No new runtime dependency is needed for WP1–WP5.**
- Test-only references: pingouin 0.7 (ICC, Bayes factors, Welch/Games-Howell, power,
  compute_effsize, mixed ANOVA, cronbach, Friedman, Cochran), scikit-learn 1.9 (Ridge/Lasso/
  ElasticNetCV, MLP, DecisionTree, KNN, MDS, PLSRegression), semopy, plspm. Optional test-only
  additions, all pure-Python wheels: **lifelines** (KM/Cox/AFT reference, macOS arm64 + Windows),
  **scikit-posthocs** (Dunn, Conover, Duncan/SNK-style references), **factor_analyzer** (ULS/GLS/
  alpha factoring reference). R reference values (survival, emmeans, metafor, pwr, BayesFactor,
  psych) can be committed as fixture numbers in tests.

### 3.1 Priorities: what Indonesian S1–S3 theses use most
1. **P1 (highest):** hierarchical regression blocks with R² change, interaction terms (MRA) and
   casewise/outlier diagnostics; GLM Univariate **custom model** (RAK/RCBD, factorial without some
   interactions) with EM Means, profile plots and **Duncan / SNK / Dunnett / Tamhane**; Crosstabs
   statistics (OR/RR, McNemar, kappa, gamma, residuals); Reliability models (split-half, omega, ICC);
   effect sizes and the missing legacy nonparametric tests; Output save/reopen and Excel/APA export.
2. **P2:** Survival (KM + log-rank, Cox); Generalized Linear Models (Poisson/NB/gamma/binary) and
   GEE; Probit (LD50); Power Analysis (sample size); Proportions; Exact tests; Bootstrap options;
   Multiple Imputation + Missing Value Analysis; Custom Tables + Multiple Response (questionnaires);
   P-P/Q-Q, Percentiles, Bland-Altman, Weighted Kappa.
3. **P3:** Meta-analysis; Bayesian; Quantile regression; Ridge/Lasso/Elastic Net; WLS; 2SLS; Mixed
   repeated covariance and GLMM; time series/forecasting; new NPTESTS dialogs; Discriminant/Factor
   extras; Distances; Ratio; Correspondence analysis; MDS.
4. **P4 (later or optional):** Decision Trees, Neural Networks (MLP/RBF), Nearest Neighbor, TwoStep,
   CATREG/CATPCA/MCA/OVERALS/PREFSCAL, Complex Samples, Direct Marketing, Conjoint, Simulation,
   Quality Control charts, TURF, Data Preparation (validation, unusual cases, optimal binning,
   ADP), Propensity Score Matching (move to P2 for health-science supervisors).

### 3.2 Work packages (disjoint files; can run in parallel after Phase 0)

#### WP1: Compare Means, GLM, Mixed (P1 core, then P2/P3)
- **Owns (edit):** `rsstatchem/ui/dialogs/compare.py`, `rsstatchem/ui/dialogs/glm.py`,
  `rsstatchem/core/stats/{ttest,anova,manova,repeated,mixed}.py`.
- **Creates:** `core/stats/glm_model.py` (term builder, SS types I–IV, contrasts, EMMs incl. simple
  main effects, Save), `core/stats/posthoc_ext.py`, `core/stats/proportions.py`,
  `core/stats/varcomp.py`, `ui/dialogs/proportions.py`, `ui/dialogs/glm_model_ui.py` (Model /
  Contrasts / Plots / Save sub-dialogs shared by Univariate/Multivariate/RM), tests
  `tests/test_glm_model.py`, `test_posthoc_ext.py`, `test_proportions.py`, `test_mixed_ext.py`.
- **Scope:**
  - Univariate: Model sub-dialog (custom terms, main-effects-only, nested; SS I–IV; intercept),
    random factors (expected mean squares, Satterthwaite error terms), WLS weight, Contrasts, Plots
    (profile with error bars), EM Means (compare main and simple main effects, LSD/Bonferroni/Sidak),
    Save (pred, resid, Cook, leverage), Options (parameter estimates with HC0–HC4, observed power,
    lack of fit, BP/White/F heteroskedasticity tests, residual and spread-vs-level plots).
  - Port Model/Contrasts/Save/power to Multivariate and Repeated Measures. RM contrasts (deviation,
    simple, difference, Helmert, repeated).
  - Post hoc: Duncan, S-N-K, Tukey-b, REGW-F/Q, Hochberg GT2, Gabriel, Waller-Duncan, Dunnett
    (2-sided/one-sided, control first/last), Tamhane T2, Dunnett T3, Dunnett C, with homogeneous-subsets
    tables, used by both One-Way ANOVA and GLM.
  - One-Way ANOVA: Contrasts (polynomial plus custom coefficients with CIs) and fixed/random effects.
  - t-tests: Bootstrap, paired missing options. Summary Independent-Samples t test.
  - Proportions (one-sample, independent, paired: Wald/Wilson/CP/Agresti-Coull/Jeffreys/Newcombe
    CIs and tests). (Means layers belong to WP2 because `MeansDialog` lives in `logistic.py`.)
  - Mixed: Repeated (AR1/CS/Toeplitz/UN/diag residual covariance via own REML on top of the existing
    likelihood code), EM Means, Save, pseudo-R²/ICC. Variance Components (ANOVA/ML/REML).
- **Libraries:** statsmodels OLS/WLS/anova_lm, `statsmodels.stats.multitest`, scipy
  studentized_range and `scipy.stats.dunnett`, multivariate t via `scipy.stats.multivariate_t` for
  Dunnett, statsmodels `proportion_confint`/`confint_proportions_2indep`, MixedLM. Repeated
  covariance structures need own GLS-REML.
- **Verification:** pingouin (anova, welch_anova, pairwise_tukey, pairwise_gameshowell, rm_anova,
  mixed_anova), scipy.stats.dunnett, statsmodels anova_lm typ=1/2/3, published SPSS outputs
  (Field, *Discovering Statistics using SPSS*, and the IBM case studies shipped with SPSS sample
  files), R emmeans/agricolae (Duncan, SNK, REGW) values as fixtures, Montgomery DOE textbook
  examples (RCBD).
- **Size:** XL (≈ 3,500–4,500 LOC incl. tests). If two agents are available, split into WP1a (GLM
  Model/Contrasts/EMM/Save + post hoc) and WP1b (Mixed + Variance Components + Proportions + t-test
  extras). They share no files if `posthoc_ext.py` belongs to WP1a.

#### WP2: Regression family and Generalized Linear Models (P1 for Linear/Logistic; P2/P3 for new procedures)
- **Owns (edit):** `ui/dialogs/regression.py`, `ui/dialogs/logistic.py` (which also holds the
  Means, Curve Estimation and ROC dialogs, so WP2 owns those too, including the Means layers work),
  `core/stats/{regression,correlate,logistic,curvefit_spss,roc,means}.py`. `ui/dialogs/fitting.py`
  is **not** edited: the new SPSS NLR dialog registers "Nonlinear..." under `Analyze/Regression`
  from a new module. Note that `ALIASES` in `ui/menus.py` is applied after MENUS, so the integrator
  must drop that alias line.
- **Creates:** `core/stats/genlin.py` (GENLIN + GEE + Probit + Loglinear-as-Poisson-GLM),
  `core/stats/regression_ext.py` (WLS power search, 2SLS, Quantile, Regularized
  ridge/lasso/elastic-net with own K-fold CV and trace plots, SPSS NLR with constraints/bootstrap,
  PLS regression), `core/stats/distances.py` (PROXIMITIES; replaces the **Distances** placeholder),
  `core/stats/cancorr.py`, `core/stats/roc_analysis.py`, `ui/dialogs/genlin.py`,
  `ui/dialogs/regression_ext.py`, tests `test_regression_ext.py`, `test_genlin.py`,
  `test_logistic_ext.py`, `test_correlate_ext.py`.
- **Scope:**
  - Linear: blocks (Previous/Next) with R² change per block, Remove method, selection variable,
    case labels, WLS weight, part/partial correlations, covariance matrix, casewise diagnostics,
    collinearity eigen/condition index, free plot builder plus partial plots, full Save list
    (studentized/deleted residuals, Mahalanobis/Cook/leverage, PI, DfBeta/DfFit/CovRatio), PRESS,
    Use-F criteria, Bootstrap.
  - Logistic: blocks and interaction terms, conditional methods, all contrasts, residual/influence
    saves, classification plot, casewise list.
  - Multinomial: Model/Criteria/Options/statistics extras. Ordinal: Location/Scale models plus
    output extras.
  - Curve Estimation Save.
  - Correlations: CI sub-dialog, covariances, Bootstrap. Partial extras. Canonical correlation.
  - New: GENLIN (normal/gamma/inverse-Gaussian/Poisson/NB/binomial/Tweedie; links; Type III LR/Wald
    tests; EMMs; Save), GEE (working correlation independent/exchangeable/AR1/M-dependent/unstructured,
    robust vs model SE, QIC), Probit (LD50/ED-x with fiducial limits, natural response rate), Weight
    Estimation, 2SLS, Quantile (multiple τ, rank-score/bootstrap SE), Linear OLS Alternatives
    (Ridge/Lasso/Elastic Net with CV and holdout), Loglinear General/Logit/Model Selection, ROC
    Analysis (paired DeLong comparison, precision-recall), Distances.
  - GLMM (binary/Poisson random intercept) is P3: statsmodels BinomialBayesMixedGLM is variational
    Bayes, so a Laplace-approximation implementation is needed for SPSS-like output. Mark it
    "approximate" or defer.
- **Libraries:** statsmodels OLS/WLS/GLM/GEE/QuantReg/OrderedModel/MNLogit/Logit/Probit,
  `statsmodels.sandbox.regression.gmm.IV2SLS` (or hand-rolled 2SLS, about 40 lines),
  `OLS.fit_regularized(L1_wt=…)` (elastic net; CV loop own), `OLSInfluence`/`GLMInfluence`
  (Cook, DfBeta, leverage), CanCorr, scipy.optimize (NLR constraints).
- **Verification:** statsmodels and R glm/geepack/quantreg/AER::ivreg fixtures, scikit-learn
  Ridge/Lasso/ElasticNet (test-only), pingouin linear_regression and partial_corr, SPSS published
  outputs (Field; IBM case studies such as *ceramics*, *drugdisc* for GENLIN), Finney (1971) probit
  data for LD50, Hosmer-Lemeshow textbook examples.
- **Size:** XL (≈ 4,500 LOC). Natural split: WP2a (existing Linear/Logistic/Ordinal/Multinomial/
  Correlate/Curve upgrades) and WP2b (new genlin.py, regression_ext.py, distances, cancorr,
  roc_analysis and their dialogs). All shared-file edits sit in WP2a.

#### WP3: Descriptives, Crosstabs, Scale/agreement, Nonparametric, Exact (P1/P2)
- **Owns (edit):** `ui/dialogs/descriptive.py`, `ui/dialogs/explore.py` (Explore, Crosstabs,
  Reliability), `ui/dialogs/nonparametric.py`,
  `core/stats/{descriptive,explore,crosstabs,reliability,nonparametric}.py`.
- **Creates:** `core/stats/agreement.py` (weighted kappa, Fleiss, ICC, Bland-Altman),
  `core/stats/nptests.py` (new One-Sample/Independent/Related NPTESTS with hypothesis-summary tables,
  pairwise Dunn-Bonferroni, Hodges-Lehmann, Jonckheere, Moses, Wald-Wolfowitz, marginal homogeneity),
  `core/stats/ratio.py` (replaces the **Ratio** placeholder), `core/stats/distplots.py` (stand-alone
  P-P/Q-Q with distributions and transforms, Percentiles dialog, Normality Analysis incl. Mardia),
  `ui/dialogs/{agreement,nptests,distplots}.py`, tests `test_crosstabs_ext.py`,
  `test_reliability_ext.py`, `test_nptests.py`, `test_agreement.py`.
- **Scope:** every "missing" cell in §2 for Frequencies, Descriptives, Explore, Crosstabs,
  Reliability and legacy nonparametrics. Exact/Monte-Carlo button on Crosstabs and NPAR tests.
  Effect size r. New NPTESTS dialogs. Weighted Kappa. Bland-Altman. Ratio. P-P/Q-Q.
  Percentiles. Normality Analysis. Bootstrap buttons on Frequencies, Descriptives, Explore,
  Crosstabs (via Phase 0 helper).
- **Libraries:** scipy.stats (fisher_exact, contingency, somersd, kendalltau variant c, median_test,
  mannwhitneyu exact, wilcoxon exact, ks_2samp, page_trend_test, permutation_test), statsmodels
  contingency_tables (mcnemar, StratifiedTable for CMH/Breslow-Day, SquareTable for marginal
  homogeneity / Bowker), inter_rater.cohens_kappa (weights), exact network algorithm only for small
  r×c (Monte Carlo otherwise).
- **Verification:** pingouin (cronbach_alpha, intraclass_corr, chi2_independence, mwu, wilcoxon,
  friedman, cochran, kruskal), scipy, R irr/psych (ICC, omega) and DescTools (gamma, Somers, lambda,
  uncertainty, Jonckheere) fixtures, Agresti *Categorical Data Analysis* examples, the existing
  `examples/contoh_kuesioner.rsd`.
- **Size:** L–XL (≈ 3,000–3,500 LOC).

#### WP4: Survival, Missing Data/Multiple Imputation, Time Series (P2/P3; new files only)
- **Owns:** nothing existing.
- **Creates:** `core/stats/survival.py`, `core/stats/missing.py` (MVA: patterns, univariate
  stats, separate-variance t-tests, EM means/covariances, Little's MCAR), `core/stats/imputation.py`
  (FCS with linear regression / PMM / logistic, monotone, constraints; m datasets stacked with an
  `Imputation_` variable like SPSS; pooled-estimate helpers for descriptives, t-test, correlation,
  linear and logistic regression via Rubin's rules), `core/stats/timeseries.py`,
  `ui/dialogs/{survival,missing,timeseries}.py`, tests `test_survival.py`, `test_missing.py`,
  `test_timeseries.py`.
- **Scope:**
  - Survival: Life Tables (intervals, Wilcoxon-Gehan comparison, survival/hazard/density plots);
    Kaplan-Meier (factor, strata, log-rank/Breslow/Tarone-Ware pooled, pairwise, linear trend;
    survival, 1−S, hazard and log-survival plots; quartiles and means; Save survival/SE/hazard/
    cumulative events); Cox (enter/forward/backward LR/Wald/conditional, categorical contrasts, strata,
    CI for exp(B), correlations, baseline function, survival/hazard/LML plots, Save, time-dependent
    covariates via episode splitting). Parametric AFT (Weibull, exponential, lognormal, loglogistic)
    is P3.
  - Missing Value Analysis.
  - Multiple Imputation (Analyze Patterns + Impute). Integrating "pooled output across every
    procedure" as SPSS does is large. Ship pooling for the most-used procedures, and note that the
    others analyse each imputation separately (the existing Split File machinery can split by
    `Imputation_`).
  - Forecasting: Sequence charts, Autocorrelations (ACF/PACF with Box-Ljung), Cross-correlations,
    Seasonal Decomposition (ratio/difference), Exponential smoothing (simple, Holt, damped,
    Winters), ARIMA/SARIMA custom with forecasts and fit table, a simple Expert Modeler (AIC search
    over ETS + ARIMA), Spectral analysis (periodogram via scipy.signal). Temporal causal models are
    IMP/P4.
  - Create Time Series (DIFF, SDIFF, lag, lead, MA, running median, cumulative sum): hand its core
    function to the integrator for the Transform menu, or register it under `Transform` from this
    module (new menu entry; no shared file edit).
- **Libraries:** statsmodels duration (SurvfuncRight, survdiff with Breslow/Tarone-Ware weights,
  PHReg with `ties="breslow"` to match SPSS), MICE pieces or own FCS, tsa (ARIMA, SARIMAX,
  ExponentialSmoothing, ETSModel, seasonal_decompose, acf/pacf/ccf, acorr_ljungbox), scipy.signal.
  **lifelines is not needed at runtime.** Add it as a test-only dependency for KM/Cox/AFT reference.
- **Verification:** lifelines (test-only), R survival fixtures (`lung`, `ovarian`), Kleinbaum &
  Klein textbook datasets, SPSS IBM case-study outputs (*telco* KM/Cox). For MI: R mice/naniar and
  Little's MCAR test values (R `naniar::mcar_test`) as fixtures. For time series: statsmodels
  results directly, Box-Jenkins airline series.
- **Size:** XL (survival ≈1,500, MVA+MI ≈1,800, time series ≈1,500 LOC). Splits cleanly into three
  agents.

#### WP5: Power Analysis, Meta-Analysis, Bayesian, Correspondence/MDS (P2/P3; new files only)
- **Owns:** nothing existing.
- **Creates:** `core/stats/power.py`, `core/stats/meta.py`, `core/stats/bayes.py`,
  `core/stats/correspondence.py`, `core/stats/mds.py`, `ui/dialogs/{power,meta,bayes,
  correspondence,mds}.py`, tests `test_power.py`, `test_meta.py`, `test_bayes.py`,
  `test_correspondence.py`.
- **Scope:**
  - Power Analysis: 11 SPSS procedures (one-sample/paired/independent t, one-way ANOVA with
    contrasts, one-sample/related/independent binomial, Pearson/Spearman/partial correlation,
    univariate linear regression), each "estimate sample size" or "estimate power", grid values and
    power curves, precision mode.
  - Meta-analysis: continuous and binary from raw data or effect sizes, fixed/random (DL, REML,
    Hartung-Knapp), heterogeneity (Q, I², τ²), subgroups, forest, cumulative forest, funnel,
    Galbraith, L'Abbé, bubble plots, Egger/Begg bias tests, trim-and-fill, meta-regression.
  - Bayesian Statistics menu: conjugate/analytic posteriors and Bayes factors for one-sample
    normal/binomial/Poisson, related and independent normal (JZS/Rouder BF), Pearson correlation
    (Ly/Jeffreys BF), linear regression (JZS/Zellner-Siow BF with numerical integration; posterior
    of coefficients under reference prior), one-way ANOVA (BF via g-prior integration),
    repeated-measures ANOVA, loglinear (Gunel-Dickey BF for contingency tables).
  - Correspondence Analysis (SVD, symmetric/row/column normalisation, contributions, biplot),
    Multidimensional Scaling (ALSCAL-like / SMACOF PROXSCAL-like with stress, Shepard plot).
- **Libraries:** scipy noncentral t/F/χ² and integrate, `statsmodels.stats.power` (TTestPower,
  FTestAnovaPower, NormalIndPower, GofChisquarePower), `statsmodels.stats.meta_analysis`
  (combine_effects, effectsize_smd, effectsize_2proportions), own REML τ², numpy SVD, own SMACOF.
- **Verification:** G*Power published values (Faul et al. 2007) and pingouin power_* functions;
  R metafor examples (BCG vaccine dataset `dat.bcg`) as fixtures; pingouin bayesfactor_ttest,
  bayesfactor_pearson, bayesfactor_binom; JASP/BayesFactor published values; sklearn MDS (test only);
  Greenacre *Correspondence Analysis in Practice* examples.
- **Size:** L–XL (power ≈900, meta ≈1,300, Bayes ≈1,300, CA/MDS ≈700 LOC).

#### WP6: Integrator track. Data Editor, Data/Transform/Utilities, Output Viewer, Custom Tables, Multiple Response, Reports, Graphs
These touch integrator-owned shared files, so run them **sequentially by one agent (the
integrator)** or give exclusive ownership to one agent for the whole sprint.
- **Owns:** `ui/data_editor.py`, `ui/output_viewer.py`, `core/report.py`, `core/output.py`
  (additive changes only), `core/expression.py`, `core/transform.py`, `ui/dialogs/transform.py`,
  `core/selection.py`, `io/files.py`, `ui/menus.py`, `core/graph/*`, `ui/graph/*`.
- **Creates:** `core/ctables.py` + `ui/dialogs/ctables.py` (Custom Tables builder),
  `core/mresponse.py` + `ui/dialogs/mresponse.py` (Define MR sets stored in `ds.meta["mrsets"]`, MR
  Frequencies/Crosstabs), `core/stats/reports.py` + `ui/dialogs/reports.py` (Codebook, Case
  Summaries, OLAP Cubes, Display Data File Information), `ui/dialogs/data_props.py` (Define Variable
  Properties, Copy Data Properties, Set Measurement Level, Sort Variables, Compare Datasets, Copy
  Dataset, Split into Files, Date & Time wizard, Random Number Generators, Create Dummy Variables,
  Propensity Score Matching / Case-Control Matching), `ui/output_store.py` (save/open `.rso` output
  documents as JSON+PNG zip).
- **Scope:**
  - Edit ▸ Find/Replace, Go to Variable, Copy with names/labels, Options (output label/value
    display, p-value format "<.001", default table style).
  - File ▸ import/export Stata/SAS via pyreadstat; Display Data File Information.
  - Compute: about 120 more SPSS functions (date/time, full CDF/PDF/IDF/RV families, string, LAG,
    $CASENUM).
  - Utilities ▸ Variables, Data File Comments, Variable Sets.
  - Output Viewer: save/reopen output, Excel/PPTX/text export, APA TableLook toggle (Word/HTML
    renderers), pivot trays (rows↔columns↔layers), hide rows/columns, cell edit, sort, outline
    hide/headings/text items, print. Curated-help style Indonesian interpretation notes (optional,
    rule-based).
  - Graphs: scatterplot matrix, dot/drop-line/summary point, stacked histogram/frequency
    polygon/population pyramid, high-low/range bars, 1-D boxplot, violin, panel row/column variables,
    Regression Variable Plots. Quality-control charts (X̄-R/S, p/np/c/u, Pareto) are P4.
  - Custom Tables and Multiple Response.
  - "RS job" recorder: replayable JSON of dialog runs. This replaces the full SPSS syntax editor,
    which is IMP.
- **Libraries:** existing stack (openpyxl, python-docx, pyreadstat, matplotlib). python-pptx would
  be a new pure-Python dependency (justify only if PowerPoint export is wanted).
- **Verification:** round-trip tests (save/open output, Stata/SAS import vs pyreadstat), Custom
  Tables vs pandas pivot_table and the IBM Custom Tables manual worked examples, expression
  functions vs SPSS Command Syntax Reference examples, screenshots.
- **Size:** XL (≈ 5,000+ LOC). Order inside the track: Output save/Excel/APA → Find/Replace + Compute
  functions → MR sets + Custom Tables → Reports → Data/Transform dialogs → Graph additions.

### 3.3 Phase-2 backlog (after WP1–WP6)
- Classify: Decision Trees (CRT and CHAID with own implementation; sklearn only as test
  reference), MLP/RBF neural networks (own numpy implementation; sklearn MLPClassifier as reference),
  Nearest Neighbor, TwoStep (BIRCH + log-likelihood distance; approximation acceptable).
- Categories: CATREG, CATPCA, MCA, OVERALS, PREFSCAL (alternating least squares; considerable work).
- Complex Samples (Taylor linearisation for means, proportions, regression with strata/PSU/weights;
  own code).
- Direct Marketing (RFM, control package test), Conjoint (orthogonal design + OLS part-worths),
  Simulation, TURF, Data Preparation module.
- GLMM with Laplace/AGQ; Parametric AFT and Shared Frailty; VAR (v32; statsmodels VAR available).

### 3.4 Impractical / out of scope (with reasons)
| Feature | Reason |
|---|---|
| Python/R programmability (BEGIN PROGRAM, Programmability Transformation, extension commands, Extension Hub, Custom Dialog Builder) | Embeds external interpreters and the IBM extension ecosystem. The app itself is Python but should not run user code. |
| Full SPSS command syntax language (Syntax Editor, INSERT, macros, MATRIX, OMS, Production Facility) | Re-implementing the SPSS language is a multi-year effort. Use the replayable "RS job" substitute. |
| Server features: Switch Server, Cache Data, Stop Processor, Repository (Collaboration and Deployment Services), ODBC/Cognos/TM1 import | Need an IBM server or enterprise connectors. |
| Scoring Wizard / Merge Model XML (PMML) | Needs PMML infrastructure and is rarely used in theses. |
| Geospatial: Spatial Modeling, Spatial Temporal Prediction, Spatial Association Rules, Map Conversion Utility | Needs shapefile/map stack and geodata. |
| Temporal Causal Models | Specialised and large. Low thesis use. |
| Workbook mode, AI Output Assistant, Dark mode details | IBM UI/AI features. The rule-based Indonesian "curated help" is the feasible analogue. |
| .spv output compatibility | Proprietary format. Use own `.rso`. |
| Exact network algorithm for large r×c tables | Combinatorially expensive. Use exact for 2×2 / small tables and Monte Carlo otherwise, which matches SPSS's own fallback. |

---

## 4. Top gaps at a glance (counts)
- **Placeholders:** 2 (Analyze ▸ Descriptive Statistics ▸ Ratio…, Analyze ▸ Correlate ▸ Distances…).
- **Whole SPSS Analyze submenus/entries missing (18):** Reports, Bayesian Statistics, Tables
  (Custom Tables), Generalized Linear Models, Loglinear, Neural Networks, Forecasting, Survival,
  Multiple Response, Missing Value Analysis, Multiple Imputation, Complex Samples, Power Analysis,
  Meta Analysis, Quality Control, Simulation, Direct Marketing, Spatial and Temporal Modeling. Also
  missing inside existing submenus: Variance Components, GLMM, Canonical Correlation, Probit, WLS,
  2SLS, Quantile, Ridge/Lasso/Elastic Net, PLS regression, Automatic Linear Modeling, TwoStep, Tree,
  Nearest Neighbor, ROC Analysis, Correspondence, Optimal Scaling, Weighted Kappa, MDS, NPTESTS (3),
  Proportions (3), Summary t test, P-P/Q-Q, Percentiles, Bland-Altman, Normality Analysis, TURF.
- **Existing procedures with notable option gaps:** 25 (§2). The most consequential for theses:
  Univariate full-factorial only (no RCBD), no regression blocks, Crosstabs limited to
  chi-square/phi/CC, Reliability α only, post hoc limited to 6 tests, no Bootstrap/Exact anywhere
  in Base procedures.
- **Infrastructure:** no output save/reopen, no Excel export of output, no pivot editing besides
  transpose, no Find/Replace, Compute has 45 of ~200 SPSS functions, no syntax editor.
