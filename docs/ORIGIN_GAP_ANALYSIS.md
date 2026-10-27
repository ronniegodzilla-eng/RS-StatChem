# RS-StatChem Graph & Analysis vs OriginLab Origin / OriginPro 2024b–2026: feature inventory, gaps, work packages

Date: 2026-10-27. Scope: the Origin-style module (`rsstatchem/ui/origin/`, `rsstatchem/ui/graph/`,
`rsstatchem/core/graph/`, `rsstatchem/core/analysis/`). No code was changed.

## 0. Method

**Origin side.** Read from docs.originlab.com: the Graph Types appendix and all 19 category pages
(each plot type), the Data Analysis user guide (which tools are Pro), the Statistics user guide
(every tool with its Pro mark), the Signal Processing chapter, the Mathematics chapter, the NLFit
chapter (theory, multi-curve/global fit, special operations, fit comparison), the Peak Analyzer
pages, the Curve Fitting Functions list (all categories and function names), the Import/Export
X-Function list, the Image X-Function categories, the Axis Scale tab, the 2025/2025b/2026 release
summaries, and the OriginLab blog post on validating against NIST StRD. The Origin vs OriginPro
page only names the Pro areas (peak fitting, surface fitting, statistics, signal processing,
image processing). The per-tool Pro marks below come from the user guide pages.

**App side.** Read all module sources and tests, then ran `OriginMainWindow` headless
(`QT_QPA_PLATFORM=offscreen`, scratch `RSSTATCHEM_HOME`) and dumped every menu (in all three menu
modes) and toolbar. The scripts and home folder are in the scratchpad; nothing was written to the
repo or the venv. I also checked which libraries the venv has, the packaging excludes, and the
reusable engines in `core/stats` (SPSS side).

Status codes: **Y** = present and works like Origin · **~** = partial (what is missing is noted) ·
**–** = missing · **n/a** = we should not copy it (proprietary or not useful on the desktop).
Pro = OriginPro only.

---

## 1. What the module has today (from the headless menu dump)

* Window model: Project Explorer with folders; Workbook (data sheets plus report sheets); Matrix
  book (image mode, X/Y mapping); Graph window (a Graph Studio canvas as an MDI child); Notes;
  Results Log and Messages Log; Object Manager. Menus switch with the active window type (book,
  graph, matrix). Projects save as `.rsop` (zip + JSON). Undo works in sheets, matrices and graphs.
* Worksheet: column header rows (Long Name, Units, Comments, F(x)= formula, Sparklines). Column
  designations X/Y/Z/xEr/yEr/L/N. Range notation `[Book1]Sheet1!B[3:10]`. Set Column Values uses a
  small Origin-like function set (sin…log, rnd, grnd). Fill with row numbers or uniform/normal
  random. Sort column/sheet. Transpose. Convert to Matrix (Direct, XYZ gridding). Add, duplicate,
  rename and delete sheets.
* Plot menu: 29 kinds. Line, Scatter, Line+Symbol, Y Error, Bubble, Color Mapped, Column (plain,
  stacked, 100 %), Bar (plain, stacked), Pie, Doughnut, Area (plain, stacked), Double-Y, Stack,
  Waterfall, Box, Histogram, Contour, Heatmap, 3D Scatter, 3D Surface, Polar, Ternary. Matrix plots
  add Contour (fill or B/W lines), Heatmap, Image, 3D colormap surface, 3D fill surface and Wire
  Frame.
* Graph: Add Plot to Layer, Rescale, New Layer (below/right/inset), Right Y, Remove Layer, Legend.
  Insert Text/Arrow/Line/Rect. Format offers Page/Layer/Axes/Plot/Legend properties, 7 themes,
  user templates and Rotate 3D. Axis scales are linear, log10, log2 and ln, with reverse. Plot
  property schemas cover line, symbol, fill/hatch, error bars and per-type options. In-place text
  editing with Origin escape codes. Export to PNG/JPG/TIF (DPI choice), SVG, PDF and EPS
  (Type 42 fonts). Copy Page.
* Analysis: Differentiate, Integrate and Normalize. Linear, Polynomial and NLFit fitting with 37
  built-in functions in 9 categories plus user-defined functions (safe AST parser). NLFit has
  initial values, fixed parameters and bounds; weighting none/instrumental/statistical/direct;
  confidence and prediction bands; residual plot; Find X/Y; lack-of-fit ANOVA. Smoothing
  (Savitzky-Golay, adjacent averaging, percentile, FFT low-pass, LOWESS). FFT power spectrum.
  Baseline (I-ModPoly, ALS). Find Peaks. Multiple Peak Fit (Gauss, Lorentz, Voigt). Reports
  recalculate in Auto or Manual mode.
* Statistics: Statistics on Columns and on Rows, Frequency Counts, Correlation (Pearson,
  Spearman, Kendall), Normality (Shapiro-Wilk, KS, Lilliefors, Anderson-Darling, D'Agostino,
  Chen-Shapiro). One-sample, two-sample and paired t-tests. One-way and two-way ANOVA (SS types
  I/II/III, Tukey, Bonferroni, Sidak, Fisher, Scheffé, Holm, Dunnett). Mann-Whitney, Wilcoxon
  signed rank, Kruskal-Wallis and Friedman.
* Import: one CSV/TXT or one Excel file. Export: ASCII/CSV, graphs.
* Tests: 78 test functions (test_origin 13, _extra 21, _round3 15, test_graph 25, test_wp6 4).

Engines already in the repo that the module does **not** use yet (core is code-complete; only the
Origin menu and report wrapper are missing):
* `core/analysis`: `mathematics.interpolate` (linear, spline, Akima, PCHIP) and
  `calibration.calibrate` (LOD/LOQ, inverse prediction).
* `core/stats`: survival (KM, life tables, Cox), roc/roc_analysis, power, factor (PCA with
  rotations), classify (k-means, hierarchical, discriminant), nptests (sign, KS 2-sample, Mood
  median, Wilcoxon one-sample, exact p values), proportions, repeated, manova, timeseries
  (ACF/PACF), distplots (P-P/Q-Q, 13 distributions), descriptive, distances, mds, varcomp.
  `core/stats/pls*` is PLS-SEM, **not** PLS regression.
* `core/graph/spss_charts.py`: scatter matrix, violin, dot, drop-line, high-low, population
  pyramid, stacked histogram, frequency polygon. These are static matplotlib images, not editable
  Graph models.

---

## 2. Origin feature inventory with status

### 2.1 File menu
| Origin feature | Status | Notes |
|---|---|---|
| New Project/Workbook/Matrix/Graph/Notes/Folder | Y | |
| New Function plot (Plot Function / 3D Function dialog) | – | Graph from `y=f(x)` or `z=f(x,y)`. The parser exists (`parse_user_function`, `matrix.evaluate`). |
| New Layout page (multi-graph publication page) | – | Origin's Layout window. |
| Open / Recent / Close / Save / Save As | Y | `.rsop` |
| Project templates / Save Window As (.ogwu/.otpu) | ~ | Graph templates only (JSON). No workbook or analysis templates. |
| Append project, Open Excel natively | – | |
| Import Single ASCII / CSV | ~ | No Import Wizard (header lines → Long Name/Units/Comments, skip lines, partial import, column delimiter preview, date formats, re-import, import filters). |
| Import Multiple ASCII (one book per file / sheets / columns) | – | Very common for spectra series. |
| Import Excel | ~ | One sheet only. No "all sheets", no named ranges. |
| Import JCAMP-DX (IR/Raman/NMR/MS) | – | Open IUPAC spec. Write our own parser (AFFN/ASDF compression). |
| Import Thermo/Galactic SPC | – | Spec is public. Write our own parser (avoid the GPL `spc` packages). |
| Import MATLAB .mat | – | `scipy.io.loadmat` (v4–7.2). v7.3 needs h5py. |
| Import HDF5 / NetCDF | – | h5py (optional). NetCDF3 via `scipy.io.netcdf_file`. NetCDF4 is HDF5, so h5py reads it. |
| Import NI TDM/TDMS, pCLAMP, Igor, Bruker OPUS, Famos, HEKA, Prism, SigmaPlot, KaleidaGraph, Minitab, SAS, SPSS, EDF, mzXML, WAV, SPE, binary 2D | – | Low priority except SPSS/SAS: `pyreadstat` already ships. WAV: `scipy.io.wavfile`. Binary 2D: numpy. Others are niche. |
| Import image to matrix (impImage, image stack) | – | Pillow is available. |
| Import Origin project .opj/.opju | n/a | Proprietary. See §3. |
| Data Connectors (live links, periodic refresh, web/SQL/JSON/HTML table) | – | CSV/Excel/JSON "connect and re-import" is feasible. Skip SQL and web for now. |
| Export ASCII / Excel / PDF worksheet / SQLite / XML / NetCDF matrix | ~ | ASCII only. Add Excel (openpyxl ships) and report export (HTML/Word/PDF: `core/report.py` and python-docx exist). |
| Export Graphs (batch, all graphs in project/folder, size/DPI) | ~ | Exports the active graph. No batch export of all graphs. No GIF/EMF/video. |
| Print / Print Preview | – | QPrinter. |
| Batch Processing (analysis template + many files → summary sheet) | – | Origin 2025 has a new batch dialog. Our recalc infrastructure makes this feasible. |

### 2.2 Edit menu
| Feature | Status | Notes |
|---|---|---|
| Undo/Redo, Cut/Copy/Paste, Copy with headers, Clear, Delete, Select All, Copy Page, Duplicate, Rename | Y | |
| Paste Transpose, Paste Link (cell links), Paste Special | – | |
| Find / Replace in sheet; Find in Project; Go To row | – | `ui/find_replace.py` exists on the SPSS side and can be reused. |
| Insert/Delete rows and columns | Y | |
| Copy/Paste Format (graph styles) | – | Origin's Copy Format / Paste Format of plots, layers and axes. |
| Button Edit mode, Merge/Unmerge cells | n/a / – | |

### 2.3 View menu
| Feature | Status | Notes |
|---|---|---|
| Toolbars, Project Explorer, Results/Messages Log, Object Manager, Full Screen, Zoom | Y | |
| Status bar quick stats on selection (mean/sum/count) | – | Cheap win. |
| Script Window / Command Window (LabTalk) | n/a | Offer a **Python console** instead (§2.14). |
| Code Builder, Smart Hint Log, Learning Center, Apps Gallery | n/a / – | |
| Data Slicer (2025), Project Browser (2025b), search box launching tools (2025b) | – | The search box is a cheap win (command palette over QActions). Slicer is medium effort. |

### 2.4 Plot menu (graph types; Origin's 19 categories)
"Have" means a menu entry that builds an editable Graph model.

**2D Line**
| Type | Status |
|---|---|
| Line | Y |
| Horizontal/Vertical Step, Spline | ~ (connect = step/spline exists as a plot option, no menu entries) |
| Stacked Lines by Y Offsets | ~ (Waterfall offsets) |
| Lollipop, Colormapped Line Series | – |

**2D Scatter**
| Type | Status |
|---|---|
| Scatter, Y Error, Bubble, Color Mapped | Y |
| Bubble + Color Map | ~ (template `bubble_color` exists, not in menu) |
| XY Error, Error Band (filled band) | – (xerr is in the model, renderer support partial) |
| Vertical Drop Line, Scatter + Rug, Column Scatter, Scatter Central, Density Dots, Color Dots | – |

**2D Line + Symbol**
| Type | Status |
|---|---|
| Line + Symbol | Y |
| Line Series, 2-point segment, 3-point segment | – |

**Column/Bar**
| Type | Status |
|---|---|
| Column, Bar, Stacked, 100 % Stacked Column | Y |
| 100 % Stacked Bar | ~ (template exists, not in menu) |
| Column + Label | ~ (show values option) |
| Floating Column/Bar, Ribbon, Circular Packing, Spiral Bar, Frequency Chart | – |

**Pie**
| Type | Status |
|---|---|
| 2D Pie, Doughnut | Y |
| 3D Pie, Pie/Bar of Pie, different-radius pie, Sunburst | – |

**Multi-Axis**
| Type | Status |
|---|---|
| Double Y | Y |
| Double-Y Column / Column-Line, 2Ys | ~ (per-plot right axis works, no presets) |
| Double X (independent / custom formula / Arrhenius 1/T top axis), Y-shared Double X | – (Arrhenius top axis matters to chemists) |
| 3Ys / 4Ys, Multiple Y Axes, Zoom Graph | – |

**Multi-Panel**
| Type | Status |
|---|---|
| Stack Panels | Y |
| Vertical/Horizontal 2 Panel, Inset | ~ (layers below/right/inset by hand) |
| 4 Panel, 9 Panel, Multiple Panels by group | – |
| Trellis, Trellis Double Y, Cluster Plot | – |

**Waterfall**
| Type | Status |
|---|---|
| 2D Waterfall | Y |
| Waterfall with Y/Z color mapping | – |
| 3D Waterfall (and color-mapped) | – |

**Statistical**
| Type | Status |
|---|---|
| Box Chart, Histogram (+ normal curve) | Y |
| Violin, Scatter Matrix | ~ (static images in `spss_charts`) |
| Probability Plot, Q-Q Plot | ~ (static images in `distplots`) |
| Interval Plot, Bar (mean ± SD/SE) | – |
| Histogram + Rug / Probabilities, Multi-Panel and Stacked Histograms, Marginal Box/Histogram | – |
| Population Pyramid | ~ (static image) |
| 2D Kernel Density, Beeswarm, Bland-Altman, Pareto (binned/raw), QC X-bar R, Run Chart, CDF Plot | – |
| Parallel Plot/Sets/Index, Dot/Stacked Dot, Ridgeline, Symmetry, Multi-Vari, Variability, Interaction/Main Effects | – |
| Sankey/Alluvial/Chord/Network/Edge bundling, Cause-and-Effect | – (low priority) |
| Pro output plots: survival curves, scree/biplot/loading, dendrogram, ROC | – in this module (engines produce static PNGs on the SPSS side) |

**Grouped** (scatter, columns, histogram, box, interval, mean bar, violin, dot, floating bar by an
indexed group column): – (Group-by exists only in the Graph Studio Plot Setup.)

**Area**
| Type | Status |
|---|---|
| Area, Stacked Area | Y |
| 100 % Stacked Area | ~ (template exists, not in menu) |
| Fill Area (between two curves), Fill Two Colors, Kite | – (fill between curves is common in spectroscopy) |

**Specialized**
| Type | Status |
|---|---|
| Polar (θ r), Ternary | Y |
| Vector XYAM / XYXY, Streamline, Contour + gradient vector | – |
| Polar vector, Compass, Radar/Spider, Radial bar, Wind Rose (binned/raw), Smith chart | – |
| Piper/Trilinear, Durov, Stiff (geochemistry), Right-triangle ternary, Ternary vector | – |
| Bullet chart, browser graph | – |

**Financial** (OHLC, candlestick, bridge/waterfall chart, high-low-close): ~ (static high-low image only).

**Map** (bar map, pie map, tile grid, treemap): – (low priority, needs shapefiles).

**3D XYY** (3D bars, ribbons, walls): –

**3D Surfaces**
| Type | Status |
|---|---|
| Color Map Surface, Color Fill Surface, Wire Frame (matrix) | Y |
| Wire Surface | ~ |
| Surface with projection | ~ (floor contour option) |
| X/Y Constant with Base, Multiple Surfaces, Surface with Error Bars, Parametric Surface, Stacked Surfaces/Heatmaps, 3D Ternary Surface | – |

**3D Symbol/Bar/Vector**
| Type | Status |
|---|---|
| 3D Scatter | Y |
| 3D Scatter + Z error, 3D Line/Trajectory, 3D Bars (incl. stacked/floating), 3D Vectors, 3D ternary symbol, Tetrahedral | – |

**Contour**
| Type | Status |
|---|---|
| Color Fill, B/W Lines with labels, Heatmap (XYZ and matrix) | Y |
| Gray-scale map | ~ (pick the colormap) |
| Heatmap with labels | ~ (show values) |
| Categorical Z, Split Heatmap, Polar Contour/Heatmap, Ternary Contour, Overlaid Contour | – |

**Image and Profiles**
| Type | Status |
|---|---|
| Image Plot (matrix) | Y |
| Image/Contour Profiles (line cuts with linked profiles), Image as background | – |

### 2.5 Column menu
| Feature | Status | Notes |
|---|---|---|
| Set As X/Y/Z/xEr/yEr/L/N | Y | |
| Set as Group / Subject designations | – | Needed for grouped plots and RM ANOVA. |
| Set Column Values, F(x)= auto-recalculate | ~ | Small function set. Origin adds: conditional `?:`/if, statistics functions (mean(col), sd, total, max…), date/time functions, string functions, `Normalize`, interpolation `table()`, before-formula scripts, `col(A)[i-1]` row offsets, references to other sheets. |
| Set Multiple Column Values | – | |
| Fill: row numbers, random numbers | Y | |
| Fill: arbitrary set / A–Z / month names | – | |
| Add/Insert/Delete/Move columns, Sort column | Y | |
| Swap columns, Sort by multiple keys, Sort by Long Name | – | |
| Column Properties | ~ | Name, label rows, designation. No Format (Numeric/Text/Date/Time/Month/Day-of-week/Text & Numeric), no Display (decimal places, scientific notation), no Data Type (double/int/complex), no width setting. |
| Mask/Unmask cells (excluded from plots/fits) | – | Important for fitting outliers. |
| Conditional formatting / Data Highlighter / heatmap cells | – | |
| Column Statistics quick stats, Frequency Count | Y | |
| Sparklines | Y | |

### 2.6 Worksheet menu and data manipulation
| Feature | Status | Notes |
|---|---|---|
| Clear, Set rows, Sort worksheet, Transpose, Convert to Matrix (Direct, XYZ Gridding) | Y | |
| Convert to Matrix: Expand, XYZ Log Gridding, gridding methods (Renka-Cline, Kriging, Shepard, thin-plate spline) | ~ | Linear/cubic/nearest only. `scipy.interpolate.RBFInterpolator` covers TPS and other RBFs. |
| Split / Stack / Unstack columns, Wide↔Long, Pivot Table | – | pandas. `core/pivot.py` exists on the SPSS side. |
| Reduce Rows / Reduce by group / Reduce Duplicate X / Reduce to evenly spaced X (reduce_ex, reducexy) | – | |
| Remove duplicates, Trim missing | – | |
| Worksheet Query / Data Filter (per-column filter, Origin's filter icons) | – | |
| Split Workbook, Append Worksheets, Merge Workbooks by key | – | |
| Compare sheets, Find/Replace (findreplace) | – | |
| Subtract Reference Data, Subtract Straight Line, Vertical/Horizontal translate | – | Common in spectroscopy. |
| Complex vectors (vc2ap, vri2c…) | – | Low priority. |
| Add/duplicate/rename/delete sheet | Y | |
| Copy Columns To another sheet | – | |

### 2.7 Matrix menu
| Feature | Status | Notes |
|---|---|---|
| Set Values (formula), Set Dimensions/Labels, Transpose, Flip, Rotate, Expand, Shrink, Convert to Worksheet (XYZ), Image Mode, Show X/Y, Add Sheet | Y | |
| Rotate by arbitrary angle, Crop/Extract, Inverse, Simple Math between matrices, Subtract | – | |
| Matrix Smoothing, Median/Custom/Special filter, 2D Interpolate/Extrapolate, 2D Integrate (volume, Pro), Surface area (Pro), Matrix statistics (mstackstats) | – | `scipy.ndimage` covers these. |
| Linear Fit of matrix stack | – | |
| Profiles (line, vertical, horizontal) | – | |

### 2.8 Graph menu, graph customisation, layers
| Feature | Status | Notes |
|---|---|---|
| Plot Details dialog: Line, Symbol, Fill/Pattern, Error Bars, per-type options | ~ | Missing: symbol color mapped by column, "increment by" lists, drop lines, data labels from a column with format and position, fill area under/between curves, vary-by-column (size, color, shape), transparency per element, skip points, gap/overlap for columns, error-bar caps per side, plot-level colormap with level editing. |
| Axis dialog: Scale (From/To/Type/Rescale/Increment/Minor) | ~ | Types now: linear, log10, log2, ln. Missing (Origin types): **Probability, Probit, Reciprocal, Offset Reciprocal, Logit, Double-log Reciprocal (Weibull), Discrete**, symmetrical log, custom formula on the top axis. Rescale modes (Normal, Auto, Fixed, Fixed-from) missing. |
| Axis Breaks (one or more per axis, break symbol, position %) | – | Frequently requested in spectroscopy and kinetics. |
| Tick labels: display formats (decimal, scientific, engineering, ×10^n, date/time), prefix/suffix, custom tick positions/labels from a column, tick label from a dataset | ~ | Decimals and rotation only. |
| Grids, ticks in/out, mirror | Y | |
| Reference lines (at a value, mean, ±SD, fit) and shaded ranges | – | |
| Top/Right axes linked to the bottom/left by formula | – | Needed for Arrhenius and wavenumber↔wavelength. |
| Layers: add (below/right/inset), remove | Y | |
| Layer Management dialog (arrange rows×cols, spacing, size in cm, link axes, swap) | – | |
| Linked layers (axis linkage, straight/custom), Merge Graph Windows, Extract layers to graphs, Arrange Layers | ~ | `edit.merge_plots` merges plots into one layer only. |
| Exchange X-Y axes | – | |
| Plot Setup / Layer Contents (add/remove/reorder/group plots, change data) | ~ | Exists in Graph Studio (`setup_dialog.py`), not wired into the Origin window. |
| Add Error Bars (from % or a column), Add Function Graph, Add Fit/Reference curve | – | |
| Legend: auto/update/reconstruct, custom entries, legend for colormap (Color Scale object), Bubble scale | ~ | Legend plus colorbar. No standalone Color Scale object, no bubble scale. |
| Themes / Theme Organizer / user templates / Graph Styles (2025) | ~ | 7 themes plus user templates. No theme for workbooks or dialogs. |
| Graph Maker (drag columns onto axes, 2022+) | – | |
| Mini-toolbars (context popups for quick formatting) | – | |
| Speed mode / large-dataset decimation | – | |
| Data menu (graph): Set Display Range, Mask points on graph, Remove bad data, Move data points, Reorder plots | – | |
| Data Reader, Zoom, Pan, Annotation tool | ~ | No annotation that labels a point with its X,Y value. |
| Region-of-interest gadgets (Quick Fit, Integrate, Statistics, FFT, Interpolate, Differentiate, Intersect, Rise Time, Cluster, Quick Peaks, Vertical Cursor) | – | Origin's "Gadgets" menu. High value. |
| Insert: equation (LaTeX), image, table, layer table, date/time, new legend, color scale | ~ | Text/Arrow/Line/Rect only. Matplotlib mathtext already renders `$...$`. |
| Rotate 3D, 3D view settings | Y | |
| 3D lighting, axes planes, OpenGL | ~ | matplotlib 3D (no OpenGL). |
| Copy/Paste Format between plots/layers | – | |
| Digitizer (pick data from a graph image) | – | Tools > Digitizer. Useful for literature data. |

### 2.9 Analysis: Mathematics
| Feature | Status | Notes |
|---|---|---|
| Differentiate (order, smoothing) | Y | |
| Integrate (area, baseline) | Y | |
| Normalize | Y | Methods: max, range, z-score… |
| Interpolate/Extrapolate, Interpolate Y from X | ~ | Engine exists. No menu/dialog. |
| Trace interpolation, XYZ trace, 2D/3D interpolation (3D is Pro) | – | |
| Simple Column/Curve Math (A op B with interpolation onto a common X) | – | |
| Average/Merge Multiple Curves (onto a common X) | – | Spectra averaging. |
| Polygon Area | – | |
| XYZ Surface Area (Pro), 2D Volume Integrate (Pro) | – | |
| Matrix: Inverse, Simple Math, Subtract, filters | – | See §2.7. |
| Integrate/Differentiate gadgets | – | |

### 2.10 Analysis: Fitting
| Feature | Status | Notes |
|---|---|---|
| Linear Fit (with confidence/prediction bands, ANOVA, residuals, Find X/Y) | Y | |
| Linear fit with fixed intercept / slope, apparent fit on log axes | ~ | Fixed parameters work in NLFit. Check that Linear Fit has the options. |
| Fit Linear with X Error (York / ODR; Pro) | – | |
| Polynomial (order ≤ 9) | Y | |
| Multiple Linear Regression | – | statsmodels OLS (SPSS-side `regression.py` exists). |
| NLFit: function library | ~ | 37 functions vs Origin's ~200 in 20 categories (list in Appendix A). Missing chemistry-relevant ones include PearsonVII, Asym2Sig, Bigaussian, GaussMod/EMG, ECS, Giddings, Logistpk, Extreme, InvsPoly; OneSiteBind/TwoSiteBind/Comp; enzyme inhibition (Comp/Noncomp/Uncomp/Mixed); BiDoseResp, Hill1, SLogistic, SWeibull; Rheology (Carreau, Cross, Herschel-Bulkley, Bingham, VFT); Waveform (Sine, SineDamp, Step); Piecewise PWL2/PWL3; statistical CDFs; LangmuirEXT, FreundlichEXT. |
| NLFit: parameters (initial, fixed, bounds) | Y | |
| NLFit: general linear constraints (`a+b<10`), parameter sharing | – | |
| NLFit: weighting | ~ | Have no / instrumental / statistical / direct. Missing arbitrary dataset and experimental. |
| NLFit: algorithms | ~ | L-M (and TRF with bounds) only. Missing Downhill Simplex, and ODR (Pro, X errors). |
| NLFit: multiple datasets fitted independently, Global fit with shared parameters, Replica (n peaks), Concatenate | – | Global fit is a key Pro-like feature for kinetics and binding. |
| NLFit: implicit fit (circle/ellipse; Pro) | – | ODR with f(x,y)=0. |
| NLFit: Surface fit on XYZ/matrix (Pro; Gauss2D, Plane, Poly2D, Lorentz2D, Voigt2D…) | – | |
| Fitting Function Builder / Organizer (save user functions, categories, initial-value code, derived parameters) | ~ | User functions can be typed but are not saved to a library. No derived parameters (FWHM, area, t½) apart from the hard-coded peak extras. |
| Fit report: Parameters, Statistics (R², Adj R², RSS, reduced χ², RMSE), ANOVA, Covariance/Correlation matrix, Dependency, CI of parameters, t-values, residual plots (histogram, normal probability, vs order, lag) | ~ | Covariance/correlation matrix, dependency, residual normal-probability plot and residual histogram are missing. |
| Fit with Themes, Fit with Analysis Templates, batch fit of many columns → summary sheet | – | |
| Simulate Curve / Simulate Surface (Pro) | – | |
| Fit Comparison: Compare Models (F-test, AIC, BIC), Compare Datasets, Rank Models (AIC/BIC; all Pro) | – | |
| Quick Fit gadget, Quick Sigmoidal fit gadget | – | |
| Calibration curve app (LOD/LOQ, unknowns) | ~ | Engine exists. No menu. |

### 2.11 Analysis: Signal Processing (Pro marked)
| Feature | Status | Notes |
|---|---|---|
| Smoothing | ~ | Have Savitzky-Golay, adjacent averaging, percentile, FFT, LOWESS. Missing LOESS, binomial, robust LOWESS/LOESS, boundary options (pad/reflect/periodic/extrapolate), SG derivative output, weighted averaging. |
| FFT / IFFT | ~ | Power spectrum only. Missing IFFT, phase/real/imag output, spectrum types (amplitude, PSD, MSA, dB), zero padding, shift. |
| FFT Filters | – | Low/High/Band-pass, Band-block, Threshold, Low-pass parabolic. |
| IIR Filter design (Butterworth, Chebyshev I/II, Elliptic, Bessel; Pro) | – | `scipy.signal`. |
| Convolution, Deconvolution | – | |
| Correlation (auto/cross), Coherence (Pro) | – | |
| Hilbert transform, Envelope, Decimation, STFT (all Pro) | – | |
| Wavelet: CWT, DWT, IDWT, multi-scale decomposition, wavelet smoothing, wavelet noise removal, 2D wavelets (Pro) | – | PyWavelets (scipy.signal.cwt was removed in SciPy 1.15). |
| 2D FFT / IFFT / 2D FFT filter / 2D correlation (Pro) | – | numpy.fft. |
| Rise Time gadget, signal statistics | – | |

### 2.12 Analysis: Peaks and Baseline / Spectroscopy
| Feature | Status | Notes |
|---|---|---|
| Quick Peaks gadget | – | |
| Peak Analyzer goals: Integrate Peaks, Create Baseline, Subtract Baseline, Find Peaks, Fit Peaks (Pro) | ~ | Separate dialogs exist. No wizard-style Peak Analyzer with a live preview. |
| Baseline modes | ~ | Have polynomial and ALS. Missing constant, user-defined anchor points (drag on graph; detected by second derivative / local minima / Origin's anchor methods), end-points weighted, straight line, use existing dataset, XPS (Shirley, Tougaard), arPLS/airPLS, SNIP, rolling ball, baseline fitting with a function. |
| Find peaks | ~ | Have scipy prominence/height. Missing Origin methods: local max, window search, 1st/2nd derivative, residual after 1st derivative, **hidden-peak finding**, peak filtering by height %/count, manual add/remove peaks on graph. |
| Integrate peaks | ~ | Area, centroid and FWHM in Find Peaks. Missing per-peak integration limits, % area, and a mode without baseline. |
| Fit peaks | ~ | Have Gauss/Lorentz/Voigt sum. Missing pseudo-Voigt, PearsonVII, asymmetric (Asym2Sig, Bigaussian, EMG/GaussMod), shared widths, constraints, baseline fitted together with peaks, per-peak report (area, %, FWHM, centre, height), cumulative peak curve on the graph. |
| Batch peak analysis using a theme | – | |
| Spectroscopy apps (Peak Deconvolution, Fourier Self-Deconvolution (Pro), Simple Spectroscopy, unit conversion nm↔cm⁻¹↔eV, Kubelka-Munk, Tauc plot) | – | Chemistry users rely on these. |

### 2.13 Statistics (Pro marks from the Statistics user guide)
| Feature | Status | Notes |
|---|---|---|
| Descriptive statistics on columns/rows, Frequency counts, Normality, Correlation | Y | |
| Statistics on Y by X bin, 2D Frequency/Binning, Discrete frequency | – | |
| Cross Tabulation + Chi-square (Pro) | – | `core/stats/crosstabs`. |
| Distribution Fit (Pro) | – | `scipy.stats` MLE plus `distplots`. |
| Partial Correlation (Pro) | – | `core/stats/correlate`. |
| Grubbs test, Dixon's Q test (outliers) | – | High priority for analytical chemistry. |
| One/Two/Paired t-tests | Y | |
| One- and two-sample variance tests (Pro) | – | Chi-square, F, Levene, Bartlett. |
| One/Two proportion tests (Pro) | – | `core/stats/proportions`. |
| t-tests on rows (Pro) | – | |
| One-way, Two-way ANOVA | Y | |
| Three-way ANOVA | – | |
| One/Two-way repeated-measures ANOVA (Pro) | – | `core/stats/repeated`. |
| Nonparametric: Mann-Whitney, paired Wilcoxon, Kruskal-Wallis, Friedman | Y | |
| Nonparametric: one-sample Wilcoxon, paired sign test, two-sample KS, Mood's median | – | `core/stats/nptests`. |
| Multivariate (Pro): PCA (scree, loadings, scores, biplot), K-means, Hierarchical cluster (dendrogram), Discriminant, Canonical discriminant | – | `core/stats/factor` and `classify` exist. |
| Partial Least Squares (regression, VIP, cross-validation; Pro) | – | Needs a new NIPALS/SIMPLS (sklearn is excluded from the build). Core chemometrics feature. |
| Machine Learning (Pro, 2023+: classification/regression apps via Python) | n/a / – | Out of scope for now. |
| Survival (Pro): Kaplan-Meier, compare curves (log-rank etc.), Cox PH, Weibull fit | – | `core/stats/survival`. Weibull fit needs a new MLE. |
| ROC curve (Pro) | – | `core/stats/roc`. |
| Power and sample size (Pro): 1/2 proportion, 1/2/paired t, 1/2 variance, one-way ANOVA | – | `core/stats/power`. |
| Statistics gadget, Cluster gadget, Stats Advisor app | – | |

### 2.14 Image, Tools, Window, Help, scripting, automation
| Feature | Status | Notes |
|---|---|---|
| Image: Adjustments | – | Auto level, brightness, contrast, gamma, histogram equalisation/contrast, hue, saturation, invert, balance, color replace, level (Pro), LUT (Pro). numpy/Pillow. |
| Image: Arithmetic | – | Simple math, subtract background, combine. |
| Image: Conversion | – | Binary, auto binary, threshold (Pro), gray/RGB split and merge, palette, dilate/erode (some Pro). |
| Image: Geometric | – | Rotate, flip, resize, crop, offset, shear, align. `scipy.ndimage`. |
| Image: Spatial filters | – | Average, Gaussian, median, sharpen, edge detect, Laplacian, noise; some Pro. `scipy.ndimage`. |
| Image: Analysis | – | Particle analysis/Image Objects (Pro): `ndimage.label` + regionprops-like measures. Profile tool. |
| Image import, image stacks, video export | – | Overlaps the planned ImageJ-style module (`ui/imagej` is a placeholder): share one `core` image library. |
| Tools: Options | ~ | Rows and folder-only setting. Missing number display, default font, autosave/backup, decimal separator, graph defaults. |
| Tools: Fitting Function Organizer/Builder, Template Library, Theme Organizer, Color Manager, Digitizer, Batch Processing, Package/App manager | – / ~ | |
| Window: Cascade, Tile, Duplicate, Refresh, window list | Y | |
| Help: Getting Started, About | Y | |
| Help: tutorials, context help per dialog | ~ | |
| Notes window | Y | Plain text. Origin's Notes are rich text/HTML with embedded graphs. |
| Layout page (arrange graphs, worksheets, text for publication) | – | |
| Analysis Templates (book with linked analyses re-run on new data) | ~ | Auto recalc exists. No "save as analysis template" or "import into template". |
| Batch processing (files × template → summary) | – | |
| LabTalk scripting | n/a | Proprietary language. Do not clone. |
| Origin C / X-Function builder / external DLLs | n/a | |
| Python: embedded Python with an `originpro`-style package (`op.find_sheet()`, `wks.from_df()`, `op.new_graph()`) | – | Practical: an in-app Python console (`code.InteractiveConsole` in a Qt widget) plus a small `rsop` API for sheets, graphs and analyses. Keep IPython/qtconsole out (excluded and heavy). |
| Apps (App Center) | n/a / – | Build the valuable ones in (Peak Deconvolution, Calibration, Stats Advisor, Tauc, Kubelka-Munk, Function plot, Digitizer). |
| Auto-update, Recent files | Y | |

---

## 3. Licence issues and open-source equivalents

1. **Origin project files (.opj/.opju) and templates/themes (.otpu, .oth, .ogwu).** These formats
   are proprietary and undocumented. `.opju` (2018+) has no public reader. **liborigin** reads
   older `.opj` but is **GPL-3** (C++). Linking or bundling it would force GPL on RS-StatChem, so
   do not use it. The `originpro`/`OriginExt` Python packages need a licensed Origin install and
   run on Windows only. Recommendation: no OPJ/OPJU import. Document "export from Origin as
   CSV/XLSX (or Origin's ASCII export with header lines) and import with the Import Wizard". Our
   own formats stay `.rsop` and JSON templates.
2. **NAG library.** Origin's statistics and many numerical routines are built on licensed NAG
   routines. Use these open equivalents (all already shipped):
   * Distributions and tests: `scipy.stats`, including `dunnett`, `tukey_hsd`, `levene`,
     `bartlett`, `fligner`, `mood`, `ansari`, `ks_2samp`, `ecdf`, `bootstrap`, `goodness_of_fit`.
   * ANOVA, GLM, multiple comparisons: statsmodels.
   * Survival, ROC, power, PCA, k-means, discriminant, nonparametric exact tests: the in-repo
     `core/stats` engines.
   * Optimisation: `scipy.optimize` (`least_squares` for L-M/TRF, `minimize` Nelder-Mead for
     Downhill Simplex, `linprog`/`minimize` with constraints for general linear constraints).
   * Interpolation and gridding: `scipy.interpolate` (Akima, PCHIP, RBF/thin-plate, griddata).
   * FFT and filters: `numpy.fft`, `scipy.signal` (`butter`/`cheby1`/`cheby2`/`ellip`/`bessel`,
     `sosfiltfilt`, `hilbert`, `envelope`, `decimate`, `coherence`, `ShortTimeFFT`, `correlate`,
     `fftconvolve`, `deconvolve`).
   * Image: `scipy.ndimage` plus Pillow.
3. **ODRPACK.** Origin's ODR uses ODRPACK95. `scipy.odr` is **deprecated since SciPy 1.17 and is
   removed in 1.19** (confirmed: a DeprecationWarning in the venv's SciPy 1.18.1). The suggested
   replacement `odrpack` (MIT, v0.7) is fine on licence, but its Windows wheel was about 17 MB
   (v0.3.4, bundled Fortran runtime). Recommendation: implement weighted ODR in-house. Use
   `scipy.optimize.least_squares` on the augmented vector (β, δ) with the Boggs–Rogers objective
   and a sparse Jacobian. Covariance comes from the J of β after eliminating δ. Validate in tests
   against `scipy.odr` while the test venv still has SciPy < 1.19, and against the NIST/ODRPACK
   examples. This also gives the implicit fit (f(x,y)=0) and York regression.
4. **Wavelets.** `scipy.signal.cwt`/`ricker` are gone (removed in 1.15). Use **PyWavelets** (MIT
   AND BSD-3; v1.10). It has wheels for macOS arm64 and win_amd64 (cp312/313), about 4 MB wheel
   and about 12–15 MB installed (approximate). Make it optional: hide the Wavelet submenu when the
   import fails.
5. **SPC / JCAMP / OPUS readers.** The common Python SPC readers (`spc`, `spc-spectra`) are GPL.
   Write our own SPC reader from the published Galactic SPC specification (old and new formats,
   multifile, XY), and our own JCAMP-DX 4.24/5/6 reader (IUPAC spec: AFFN, ASDF compression
   SQZ/DIF/DUP, XYDATA/XYPOINTS/PEAK TABLE, NTUPLES). Both are a few hundred lines with no
   licence risk. Bruker OPUS: skip, or read later from community documentation without copying
   GPL code.
6. **Optional new runtime dependencies** (all have macOS arm64 and win_amd64 wheels for
   cp312/cp313):
   | Package | Use | Licence | Size (approx.) | Recommendation |
   |---|---|---|---|---|
   | PyWavelets 1.10 | CWT/DWT/denoise, 2D wavelets | MIT/BSD-3 | wheel ~4 MB, installed ~12–15 MB | **Add** (optional import) |
   | h5py 3.16 | HDF5, NetCDF4 (as HDF5), MATLAB v7.3 | BSD-3 | wheel ~3 MB (bundles libhdf5), installed ~8–10 MB | **Add** (optional) in the I/O phase |
   | pybaselines 1.2.1 | arPLS, airPLS, SNIP, rolling-ball, morphological, spline baselines | BSD-3 | pure Python, 212 KB | Optional. These algorithms are short enough to implement in-house; keep pybaselines as a test reference. |
   | scikit-image 0.26 | image processing | BSD-3 | wheel ~13 MB plus networkx, imageio, tifffile, lazy_loader (~+15 MB) | **Do not add** for now. `scipy.ndimage` plus numpy cover Origin's Image menu. Reconsider for the ImageJ module. |
   | netCDF4 | NetCDF | MIT | bundles netcdf-c **and its own HDF5** (duplicate of h5py) | Do not add. Use h5py plus `scipy.io.netcdf_file`. |
   | lmfit | fitting | BSD-3 | pure Python plus asteval, uncertainties, dill | Do not add. Global fit and constraints are easy with `least_squares`. |
   | npTDMS | NI TDMS | LGPL-3 | pure Python | Optional later. LGPL is fine for an unmodified dynamic import. |
   | qtconsole/IPython/ipykernel | console | BSD | 30–40 MB, pyzmq | Do not add (IPython is in the packaging excludes). Write our own console widget. |
   | odrpack | ODR | MIT | Windows wheel ~17 MB | Do not add (see item 3). |

   Every new package must be added to `requirements.txt` and `packaging/rsstatchem.spec`
   (integrator-owned per `docs/DEVELOPMENT.md`). sklearn, pingouin and seaborn stay test-only.
7. **Look and naming.** Copying Origin's workflow and menu names is fine. Do not ship Origin's
   icons, sample data files, templates, help text or the "Origin"/"OriginLab" trademarks in
   product names. The app already says "Origin-style". The Origin sample datasets (`Samples\…`)
   belong to OriginLab, so do not bundle them. Use NIST StRD (public domain) and our own synthetic
   data in tests.

---

## 4. Prioritised gaps (chemists and experimental scientists first)

Ranking: how often scientists use the feature × the size of the gap. Effort: S/M/L.

1. **Peak Analyzer for spectra/chromatograms** (L). Anchor-point and ALS/arPLS/SNIP/Shirley
   baselines with a live preview; hidden-peak finding; per-peak integration with % area; peak fit
   with pseudo-Voigt/PearsonVII/asymmetric shapes, shared widths, cumulative curve and peak table.
2. **NLFit upgrades** (M–L). A function library of about 120 high-use Origin functions (pharma,
   enzyme, binding, peak, rheology, waveform, piecewise, statistics). Fit multiple datasets
   (independent, **global with shared parameters**, replica). Linear constraints. Simplex start.
   Covariance/correlation matrix. Derived parameters. Save user functions (Function Organizer).
3. **Fit comparison and model ranking** (S). F-test, AIC/AICc, BIC, Akaike weights, Rank Models
   over a category, Compare Datasets.
4. **Import Wizard plus multiple-file import, JCAMP-DX and SPC** (M). Instrument data enters the
   app this way.
5. **Axis features** (M). Axis breaks; probability, logit, reciprocal, offset-reciprocal (1/T)
   and Weibull scales; top axis linked by formula (Arrhenius, nm↔cm⁻¹); tick label formats
   (scientific, ×10ⁿ); reference lines and shaded bands.
6. **Masking points/cells**, excluded from fits and plots, with mask from the graph (M).
7. **Fill between curves / error band, XY error bars, drop lines, data labels from a column**
   (S–M).
8. **Outlier tests (Grubbs, Dixon Q) and variance tests (F, Levene, Bartlett)** plus the calibration
   app (LOD/LOQ) exposed in Statistics (S). Analytical-chemistry staples; the engines largely
   exist.
9. **Signal processing set** (M). FFT filters (low/high/band/band-block/threshold), IIR filter
   designer, IFFT with phase, convolution/deconvolution, auto/cross-correlation, Hilbert/envelope,
   decimation, STFT spectrogram, more smoothers (LOESS, binomial, robust).
10. **Mathematics set** (S–M). Interpolate (engine exists), Average Multiple Curves onto a common
    X, Simple Curve Math, Subtract Reference/Straight Line, Polygon Area, translate.
11. **Worksheet manipulation** (M). Stack/Unstack, Split, Pivot, Reduce rows/duplicates/evenly
    spaced X, Worksheet Query/filter, Find/Replace, Set Multiple Column Values, richer F(x)
    functions (conditional, statistics, date/time), column formats (date/time, display digits).
12. **Multivariate and chemometrics** (M–L). PCA with scores/loadings/biplot, HCA dendrogram,
    k-means, discriminant analysis (wrap `core/stats`), **PLS regression** (new NIPALS/SIMPLS with
    CV, VIP, scores).
13. **Graph management** (M). Plot Setup/Layer Contents in the Origin window, Layer Management
    (rows × cols, spacing, size), 4/9-panel presets, Merge Graphs, Extract layers, Exchange X-Y,
    Copy/Paste Format, add error bars or a function plot to a graph.
14. **Statistical plots as editable graphs** (M). Interval/mean-bar, violin, probability and Q-Q,
    scatter matrix, Bland-Altman, histogram + rug/probabilities, marginal plots, 2D kernel
    density, Pareto, grouped plots by a group column.
15. **Remaining Pro statistics** (M). Repeated-measures ANOVA, three-way ANOVA, proportion tests,
    one-sample Wilcoxon/sign/KS 2-sample/Mood median, cross-tab/chi-square, partial correlation,
    distribution fit, survival (KM/Cox/Weibull), ROC, power and sample size. All but Weibull and
    distribution fit are wrappers over `core/stats`.
16. **Gadgets / region-of-interest tools on graphs** (M). Quick Fit, Integrate, Statistics, FFT,
    Interpolate, Intersect, Quick Peaks, Vertical cursor.
17. **Surface fitting and Simulate Curve/Surface** (M). 2D Gauss/Lorentz/Plane/Poly2D on XYZ and
    matrices.
18. **ODR / fit with X error, implicit fit, York regression** (M).
19. **3D and specialized plots** (M–L). 3D bar/line/trajectory, multiple surfaces, surface error
    bars, vector XYAM/XYXY, radar, wind rose, ternary contour, polar contour, 3D waterfall, Smith,
    Piper.
20. **Export and automation** (M). Excel export, report export (HTML/Word/PDF), batch graph
    export, Layout page, Print, analysis templates and Batch Processing, Python console with a
    `rsop` API, Digitizer.
21. **Wavelets and 2D signal** (M). Wavelets via optional PyWavelets; 2D FFT/filter/correlation.
22. **Image menu** for matrix windows (M–L). Adjustments, conversion/threshold, geometric,
    spatial filters, particle analysis, line profiles. Coordinate with the ImageJ module.
23. **HDF5/NetCDF/MATLAB import** (S once h5py is added), image import to matrix (S).
24. **Polish** (S each). Status-bar quick stats; command search box; Options (number display,
    autosave); rich Notes.

---

## 5. Implementation work packages (disjoint file ownership)

Shared, integrator-owned files (no work package edits them):
* `rsstatchem/ui/origin/window.py` and `rsstatchem/ui/origin/__init__.py`.
* `requirements*.txt` and `packaging/*` (per DEVELOPMENT.md).

Each work package delivers a new module `rsstatchem/ui/origin/<area>_menus.py` with
`install(main: OriginMainWindow) -> None`. The function adds its submenus and actions to
`main.menus[...]` (keys `file`, `edit`, `view`, `plot`, `column`, `worksheet`, `analysis`,
`statistics`, `graph`, `format`, `insert`, `tools`, `mplot`, `matrix`) and its tool buttons. The
integrator adds four lines to `window.py._build_menus` (`fit_menus.install(self)` and so on) and
replaces the old menu entries those modules supersede. Work packages may *call* existing public
`window.py` methods (`new_book`, `new_graph`, `run_column_op`, `stat_dialog`, `_input_dialog`,
`message`, `result`, active-window getters) but never edit it.

Cross-package rule:
* **Graph model and renderer changes belong to WP-4.** WP-1, WP-2 and WP-3 build graphs only with
  existing plot types (line/scatter/column/area/contour/heatmap/box/histogram) or put matplotlib
  PNGs into report sheets.
* Once WP-4 lands new types (fill-between band, probability axis, dendrogram…), the other work
  packages switch over in a follow-up.
* `ui/origin/model.py` is owned by WP-2. Others import it read-only. WP-2 keeps the format
  backward compatible and bumps `VERSION` only with a migration.

### WP-1 Fitting, Peaks & Spectroscopy
**Scope:**
* Function library to about 120 Origin functions, with `linear` parameter sets and grid guesses as
  the current design uses, plus derived parameters. Appendix A lists priorities.
* NLFit: multiple datasets (independent / global with sharing / replica / concatenate), general
  linear constraints, Simplex pre-iteration, arbitrary-dataset and experimental weighting,
  covariance/correlation/dependency tables, residual diagnostics, batch fit of many columns to a
  summary sheet.
* Fitting Function Organizer (user functions saved as JSON in `RSSTATCHEM_HOME`).
* Fit comparison (F-test, AIC/AICc/BIC, Akaike weights), Rank Models.
* In-house ODR (fit with X error, York line, implicit fit).
* Surface fit (XYZ and matrix: Plane, Poly2D, Gauss2D, Lorentz2D, Voigt2D, user `z=f(x,y)`).
* Simulate Curve/Surface. Quick/Sigmoidal fit presets.
* **Peak Analyzer** wizard: baselines (constant, straight line, end-points weighted, user anchor
  points with auto detection, ALS, arPLS/airPLS, SNIP, Shirley, Tougaard, existing dataset,
  fitted baseline); find peaks (local max, window search, 1st/2nd derivative, residual after 1st
  derivative for hidden peaks); integrate (limits, % area, centroid, FWHM); fit (Gauss, Lorentz,
  Voigt, pseudo-Voigt, PearsonVII, Asym2Sig, Bigaussian, EMG/GaussMod; shared/fixed widths;
  baseline co-fit); batch over columns.
* Spectroscopy helpers: unit conversion nm/cm⁻¹/eV, Tauc plot, Kubelka-Munk, the Calibration
  curve app (wrap `core/analysis/calibration.py`).

**Files (owned):**
* Existing: `core/analysis/functions.py`, `core/analysis/fitting.py`,
  `core/analysis/calibration.py`, `ui/origin/analysis.py`, `ui/origin/dialogs.py` (holds
  FitDialog. WP-2 must not edit this file; it puts its new dialogs in new files).
* New core: `core/analysis/fit_multi.py` (global/replica/concatenate),
  `core/analysis/fit_compare.py`, `core/analysis/odr.py`, `core/analysis/surface_fit.py`,
  `core/analysis/baselines.py`, `core/analysis/peak_analyzer.py`,
  `core/analysis/spectroscopy.py`.
* New UI: `ui/origin/nlfit_dialog.py`, `ui/origin/peak_analyzer_dialog.py`,
  `ui/origin/fit_organizer.py`, `ui/origin/fit_menus.py`.
* Tests: `tests/test_origin_fit2.py`, `tests/test_origin_peaks.py`,
  `tests/data/nist_strd_nls/*.dat` (public domain).
* Note: `core/analysis/signal.py` keeps its current baseline and peak functions (owned by WP-2);
  WP-1 builds the richer versions in `baselines.py` and `peak_analyzer.py` and imports `signal`
  read-only.

**Libraries:** scipy (least_squares, minimize Nelder-Mead/SLSQP, `special.wofz` for Voigt),
statsmodels, numpy. No new runtime dependency. pybaselines and scipy.odr only as test references.

**Verification:**
* **NIST StRD Nonlinear Regression**, all 27 datasets (Misra1a–d, Chwirut1/2, Lanczos1–3, Gauss1–3,
  DanWood, MGH09/10/17, Kirby2, Hahn1, Nelson, Thurber, BoxBOD, Rat42/43, Eckerle4, ENSO,
  Bennett5, Roszman1). Certified parameters and SEs from both NIST start points; require ≥ 6
  significant digits on "lower/average" difficulty and report the others.
* Gauss1/2/3 (two Gaussians on an exponential baseline) are the multi-peak acceptance test.
  OriginLab's blog says Origin's own validation samples use NIST StRD.
* NIST StRD linear regression (Norris, Pontius, Filip poly-10, Wampler1–5, Longley, NoInt1/2) for
  Linear/Polynomial/MLR.
* ODR: compare to `scipy.odr` (still in the test venv, SciPy 1.18.1) on the Boggs pressure and
  ODRPACK examples, and York regression against York et al. (2004) Pearson's data with York
  weights.
* Global fit: synthetic multi-curve ExpDec1 with shared t1. Check against a manually stacked
  `least_squares` and against the structure of the OriginLab tutorial "Global Fitting with
  Parameter Sharing" (same parameter table layout).
* AIC/BIC: compare to statsmodels for linear models; F-test for nested models against hand
  calculation.
* Baselines: pybaselines (test-only) for ALS/arPLS/SNIP. Shirley against published XPS examples.
* Peak areas: analytic Gaussian/Lorentzian/Voigt areas.

### WP-2 Data, Import/Export, Mathematics & Signal Processing
**Scope:**
* Import Wizard: header lines → Long Name/Units/Comments, skip/partial import, delimiter and
  decimal preview, date parsing, multiple files (new book / new sheets / new columns), re-import,
  saved import settings.
* Excel all sheets. JCAMP-DX and SPC readers (own code). MATLAB (`scipy.io`). HDF5/NetCDF via
  optional h5py. Image to matrix (Pillow). WAV.
* Export Excel (openpyxl), export report sheets (HTML/Word via the existing `core/report.py` and
  python-docx), batch export of all graphs (calls the `core/graph/export` API read-only).
* Worksheet operations: Stack/Unstack, Split, Pivot, Reduce rows/duplicates/evenly spaced,
  Remove duplicates, Worksheet Query/filter, Find/Replace, Paste Transpose, multi-key sort,
  Set Multiple Column Values, column formats/display (date/time, digits, scientific), **cell and
  point masks** (model flag plus API `Worksheet.masked_rows(col)` that WP-1/3/4 honour),
  conditional formatting, status-bar quick stats.
* Richer F(x) language: conditional, statistics functions over columns, date/time, string, row
  offsets, other-sheet references.
* Mathematics: Interpolate/Extrapolate (expose the engine), Interpolate Y from X, Average/Merge
  curves, Simple Curve Math, Subtract Reference/Straight Line, translate, Polygon Area, 2D
  interpolate and Integrate on matrices, matrix arithmetic/inverse/filters/crop/arbitrary rotate.
* Signal: more smoothers (LOESS, binomial, robust), FFT/IFFT with all output types, FFT filters,
  IIR designer (Butterworth/Chebyshev I/II/Elliptic/Bessel with zero-phase option),
  convolution/deconvolution, correlation (auto/cross), coherence, Hilbert/envelope, decimation,
  STFT, 2D FFT/filter/correlation, wavelets (optional PyWavelets: CWT scalogram, DWT/IDWT,
  multilevel decomposition, denoise/smooth).

**Files (owned):**
* Existing: `core/analysis/signal.py`, `core/analysis/mathematics.py`,
  `core/analysis/matrix.py`, `ui/origin/model.py`, `ui/origin/formula.py`,
  `ui/origin/sheet_view.py`, `ui/origin/matrix_view.py`, `ui/origin/recalc.py`,
  `ui/origin/explorer.py`.
* New core: `core/analysis/filters.py`, `core/analysis/transforms.py` (FFT/IFFT, STFT, Hilbert,
  correlation, convolution, 2D FFT), `core/analysis/wavelets.py`, `core/analysis/datamanip.py`,
  `core/analysis/importers/{__init__,ascii_wizard,jcamp,spc,matlab,hdf}.py`,
  `core/analysis/exporters.py`.
* New UI: `ui/origin/import_wizard.py`, `ui/origin/worksheet_ops.py`,
  `ui/origin/signal_dialogs.py`, `ui/origin/math_dialogs.py`, `ui/origin/data_menus.py`.
* Tests: `tests/test_origin_data.py`, `tests/test_origin_signal.py`, `tests/test_origin_io.py`,
  `tests/data/io/` (self-made JCAMP/SPC samples).

**Libraries:** numpy, scipy (signal, fft, interpolate, ndimage, io), pandas, openpyxl, Pillow,
statsmodels (LOWESS). Optional: **PyWavelets**, **h5py** (the integrator adds them to
requirements and the spec).

**Verification:**
* JCAMP-DX: the IUPAC JCAMP-DX spec examples (McDonald & Wilks 1988 test files; the ASDF examples
  in the spec) and round-trip of our own writer. Cross-check with the `jcamp` package (MIT,
  test-only) if installed.
* SPC: the Galactic SPC spec sample layouts; build fixtures with a small writer in the test.
* FFT/IIR/Hilbert/STFT: identities and scipy references (impulse/step responses, Parseval,
  filtfilt zero phase). Origin doc formulas for the FFT normalisations (amplitude, MSA, PSD) on
  the FFT page.
* Savitzky-Golay: Savitzky & Golay (1964) coefficient tables (with Steinier et al. 1972
  corrections).
* LOESS: statsmodels/Cleveland (1979) example.
* Wavelets: PyWavelets perfect reconstruction; denoising against Donoho–Johnstone universal
  threshold examples.
* Interpolation: scipy references.
* Stack/Unstack/Pivot: pandas equivalence.
* Masks: a masked point must be excluded from fit results (cross-check with a WP-1 fit on the
  subset).

### WP-3 Statistics & Chemometrics (Origin Statistics menu, Pro level)
**Scope:**
* Descriptive additions: Y-by-X bin statistics, 2D frequency/binning, cross-tab + chi-square,
  partial correlation, distribution fit (normal, lognormal, gamma, Weibull, exponential, logistic,
  extreme; MLE with GOF tests), **Grubbs and Dixon Q tests**.
* Hypothesis: one- and two-sample variance tests (χ², F, Levene, Brown-Forsythe, Bartlett), one-
  and two-proportion tests, t-tests on rows.
* ANOVA: three-way, one- and two-way repeated measures (wrap `core/stats/repeated`, including
  Mauchly, GG/HF corrections).
* Nonparametric: one-sample Wilcoxon, paired sign, two-sample KS, Mood's median (wrap
  `core/stats/nptests`).
* Multivariate: PCA (correlation/covariance, scree, loadings, scores, biplot data; wrap
  `factor.extract_pc`), k-means and hierarchical cluster with dendrogram (wrap `classify`),
  discriminant / canonical discriminant (wrap `classify`), **PLS regression** (new NIPALS and
  SIMPLS, CV/PRESS, VIP, scores/loadings, predictions).
* Survival: Kaplan-Meier and comparison (wrap `survival`), Cox, Weibull fit model (new MLE with
  censoring).
* ROC curve (wrap `roc`). Power and sample size (8 Origin procedures; wrap `power`).
* All as Origin-style report sheets with recalc locks (reuse the `stat_tests` infrastructure).
  Report graphs use existing Graph types or PNGs until WP-4 adds native types.

**Files (owned):**
* Existing: `ui/origin/stat_tests.py`, `ui/origin/stat_dialogs.py`, `ui/origin/report_view.py`,
  `ui/origin/report_graphs.py`.
* New UI: `ui/origin/stat_extra.py` (descriptive/hypothesis/ANOVA/nonparametric additions),
  `ui/origin/stat_multivariate.py`, `ui/origin/stat_survival.py`, `ui/origin/stat_power_roc.py`,
  `ui/origin/stat_menus.py`.
* New core: `core/analysis/chemometrics.py` (PLS-R, PCA helpers, biplot scaling),
  `core/analysis/outliers.py`, `core/analysis/distfit.py`, `core/analysis/weibull_surv.py`.
* `core/stats/*` is imported read-only.
* Tests: `tests/test_origin_stats2.py`, `tests/test_origin_multivariate.py`.

**Libraries:** scipy.stats, statsmodels, numpy and the in-repo `core/stats`. sklearn, pingouin
and lifelines (optional) only in tests.

**Verification:**
* NIST StRD ANOVA (SiRstv, SmLs01–09, AtmWtAg) and univariate (PiDigits, Lottery, Lew, Mavro,
  Michelso, NumAcc1–4) certified values.
* Grubbs: NIST/SEMATECH e-Handbook §1.3.5.17 example. Dixon Q: Rorabacher (1991) *Anal. Chem.*
  63, 139 critical values.
* Variance and proportion tests: scipy/statsmodels.
* RM ANOVA: pingouin `rm_anova` (test-only) and existing `core/stats/repeated` tests.
* PCA/LDA/k-means: sklearn (test-only) on Fisher's Iris, up to sign.
* PLS: sklearn `PLSRegression` (NIPALS) to 1e-8 on a synthetic NIR-like set, plus SIMPLS
  against de Jong (1993) definitions.
* KM/Cox: existing survival tests and lifelines if present. Weibull with censoring: closed form
  for the exponential special case, lifelines `WeibullFitter` if present.
* ROC: DeLong against the existing roc tests. Power: G*Power/statsmodels `TTestIndPower` values.
* Origin docs' "Interpreting Results" pages give the table layouts (PCA, PLS, KM) to mirror.

### WP-4 Graphs (types, axes, layers, customisation)
**Scope:**
* Axis scale types (probability, probit, logit, reciprocal, offset reciprocal with offset,
  Weibull/double-log reciprocal, symmetric log), as matplotlib `FuncScale` with proper locators.
* **Axis breaks** (one or more, position %, break symbol), via split axes in the renderer.
* Linked top/right axes by formula. Tick label formats (decimal, scientific, engineering, ×10ⁿ,
  date/time, prefix/suffix, labels from a column). Rescale modes. Reference lines and shaded
  bands.
* Plot details: fill to zero/baseline/another curve, error band, XY error bars, drop lines, data
  labels from a column, vary symbol size/color/shape by column, skip points, column gap/overlap,
  transparency.
* New plot types:
  * 2D: step/spline/lollipop entries, floating column/bar, 100 % stacked bar/area entries,
    bubble + color map, stacked lines by offset, color-mapped waterfall.
  * Statistical: interval/mean bar, violin, probability and Q-Q, scatter matrix, Bland-Altman,
    histogram + rug/probabilities, marginal, 2D KDE, Pareto, grouped plots by a group column.
  * Multi-axis/panel: 3Ys/4Ys, 4/9-panel, trellis.
  * Specialized: vector XYAM/XYXY, radar, wind rose, polar contour, ternary contour, categorical
    contour.
  * 3D: bars, line/trajectory, multiple surfaces, surfaces with error bars, 3D waterfall.
  * Analysis-output types for WP-1/3: dendrogram, biplot, cumulative peak, survival step.
* Graph menu: Plot Setup/Layer Contents (wire `ui/graph/setup_dialog.py` into the Origin window),
  Layer Management dialog, Merge Graph Windows, Extract layers, Exchange X-Y, Add Error Bars,
  Add Function Plot / New Function graph, Copy/Paste Format, Color Scale and Bubble Scale
  objects, legend update/reconstruct, Insert Equation (mathtext/LaTeX subset), image, table.
* Gadgets framework (ROI rectangle on a graph with callbacks; ships Statistics, Integrate,
  Interpolate, Intersect and Vertical Cursor; WP-1 later plugs in Quick Fit and Quick Peaks).
* Digitizer. Layout page. Batch graph export API. Print.

**Files (owned):**
* All of `core/graph/*` (`model.py`, `render.py`, `build.py`, `edit.py`, `export.py`,
  `themes.py`, `text.py`, `ternary.py`, `gridding.py`, `spss_charts.py`) and new
  `core/graph/{scales.py, breaks.py, stat_plots.py, plots3d.py, gadgets.py, layout.py}`.
* All of `ui/graph/*`.
* `ui/origin/plotting.py`, `ui/origin/graph_view.py`, `ui/origin/format_bar.py`,
  `ui/origin/icons.py`, `ui/origin/style.py`.
* New UI: `ui/origin/graph_menus.py`, `ui/origin/layer_manager.py`,
  `ui/origin/gadgets_ui.py`, `ui/origin/digitizer.py`, `ui/origin/layout_view.py`.
* Tests: `tests/test_graph.py`, `tests/test_wp6_graphs.py`, new `tests/test_graph_axes.py`,
  `tests/test_graph_types2.py`.
* Graph JSON stays backward compatible: new fields get defaults; `VERSION` bumps only with a
  loader shim.

**Libraries:** matplotlib (FuncScale, `mpl_toolkits.mplot3d`, `axes_grid1`), numpy, scipy
(`gaussian_kde`, `stats.probplot`). No new dependency.

**Verification:**
* Axis transforms: check the Origin Scale-tab definitions numerically (probability limits
  0.0001–99.999 %, logit = ln(Y/(100−Y)), offset reciprocal 1/(X+273.15), Weibull
  ln(−ln(1−X))).
* Probability plot positions against `scipy.stats.probplot`. Q-Q and Bland-Altman limits against
  hand calculation (Bland & Altman 1986).
* KDE against `scipy.stats.gaussian_kde`.
* Golden-image smoke tests (render to Agg, check artists/ticks, no pixel compares).
* Export round-trip of `.rsg`/`.rsop` with the new fields; old files must still load.
* Headless screenshots of each new plot type for visual review.

### Integrator tasks (owner of window.py, requirements, packaging)
* Call the four `install()` hooks; remove superseded entries from `_build_menus`.
* Add optional PyWavelets and h5py to `requirements.txt` and `packaging/rsstatchem.spec`
  (hiddenimports for `pywt._extensions.*`, `h5py.defs`/`h5py.utils`/`h5py.h5ac`/`h5py._proxy`).
  Check the macOS arm64 and Windows CI builds and the bundle size (expect +20–25 MB).
* Remove the `scipy.odr` dependency risk: nothing in the app may import `scipy.odr`.

### Phase 2 (after WP-1…4 merge, since it touches several areas)
* Python console plus a `rsop` scripting API (`find_book`, `Worksheet.to_df/from_df`,
  `new_graph`, `plot`, `fit`, `run_stat`).
* Analysis Templates and Batch Processing (files × template → summary sheet).
* Image menu for matrices on `scipy.ndimage`, shared with the ImageJ-style module.
* Data connectors with periodic refresh.
* Command search box, Slicer, rich Notes, Options dialog extensions.

---

## 6. Appendix A: Origin built-in fitting functions (from the Curve Fitting Functions page)

Origin has about 200 functions. ✓ = already in `functions.py`. **Bold** = add in WP-1 (high use).

* **Origin Basic:** Allometric1✓, **Beta**, Boltzmann✓, **Dhyperbl**, ExpAssoc✓, ExpDec1–3✓,
  ExpGrow1–2✓, Gauss✓, GaussAmp✓, **Hyperbl**, Logistic✓, LogNormal✓, Lorentz✓, **Poisson**,
  **Pulse**, Rational0✓, **Sine**, Voigt✓.
* **Exponential:** Asymptotic1, BoxLucas1✓, BoxLucas1Mod, BoxLucas2, Chapman, Exp1P1…Exp3P2
  family, **ExpDecay1–3** (time-offset forms), ExpGro1–3, **ExpGrowDec**, ExpLinear, Langevin,
  MnMolecular, Shah, Stirling, YldFert.
* **Growth/Sigmoidal:** **BiDoseResp**, **BiHill**, **BoltzIV**, Boltzmann✓,
  **DoubleBoltzmann**, DoseResp✓, Hill✓, **Hill1**, Logistic✓, Logistic5✓, MichaelisMenten✓,
  **SGompertz**, **SLogistic1–3**, **SRichards1–2**, **SWeibull1–2**.
* **Hyperbola:** Dhyperbl, Hyperbl, HyperbolaGen, HyperbolaMod, RectHyperbola.
* **Logarithm:** Bradley, **Logarithm**, Log2P1/2, Log3P1.
* **Peak:** **Asym2Sig**, Beta, **Bigaussian**, **CCE**, **ECS**, **Extreme**, Gauss✓,
  Gaussian, GaussAmp✓, **GaussMod (EMG)**, **GCAS**, **Giddings**, **InvsPoly**, Laplace,
  LogNormal✓, **Logistpk**, Lorentz✓, **PearsonVII**, PsdVoigt1✓, **PsdVoigt2**, Voigt✓,
  Weibull3.
* **Piecewise:** **PWL2**, **PWL3**.
* **Polynomial:** Constant, Cubic, Line✓, LineMod, Parabola, Poly✓, Poly4/5✓.
* **Power:** Allometric1/2✓, Belehradek, BlNeld, FarazdaghiHarris, **FreundlichEXT**, Gunary,
  **LangmuirEXT1/2**, Pareto, Pow2P1–3, **Power**, Power0–2 (Power1✓).
* **Rational:** BET✓, **BETMod**, Holliday, Nelder, Rational0✓–5, **Reciprocal** family.
* **Waveform:** SawtoothWave, **Sine**, **SineDamp**, SineSqr, SquareWave, **Step**.
* **Chromatography:** CCE, ECS, GaussMod, GCAS, Giddings.
* **Electrophysiology:** BoltzIV, DoubleBoltzmann, **Goldman**.
* **Enzyme Kinetics:** **CompInhib**, **NoncompInhib**, **UncompInhib**, **MixedModelInhib**,
  **SubstrateInhib** (plus the extended "Enzyme Kinetics 2" set, about 60 functions: low
  priority).
* **Pharmacology:** BiDoseResp, **Biphasic**, **OneSiteBind**, **OneSiteComp**,
  **TwoSiteBind**, **TwoSiteComp**.
* **Rheology:** **Bingham**, **Carreau**, **Cross**, **Herschel**, MYEGA, **VFT**.
* **Statistics:** ExponentialCDF, GammaCDF, Gumbel, LognormalCDF, Pareto2, Rayleigh, PearsonIV,
  **Weibull**, **WeibullCDF**.
* **Multiple variables:** GaussianLorentz, Helix, HillBurk, Line3, LineExp.
* Already in the app but not in Origin's list: Arrhenius, Langmuir, Freundlich, Temkin
  (chemistry-specific; keep).
* **Surface (Pro):** Plane, Parabola2D, Poly2D, Gauss2D, Lorentz2D, Voigt2D, Extreme2D,
  Chebyshev2D, Logistic2D, Power2D, Exponential2D. **Implicit (Pro):** Circle, Ellipse, ParabolaI.

## 7. Sources
* Graph types: https://docs.originlab.com/origin-help/graphtypes/ and the category pages
  `/origin-help/{2dline-graph,2dsym-graph,2dlinesym-graph,2d-columnbar-graph,pie-charts,multi-y-graph,multi-panel-graph,waterfall,statistical-graph,grouped-lots,map-graph,area-graphs,specialized-graphs,stock-graph,3dxyy-graph,3d-surfaces,3dxyz-graph,contour-graphs,image-graph}/`.
* Analysis tools and Pro marks: https://docs.originlab.com/user-guide/data-analysis
* Statistics tools and Pro marks: https://docs.originlab.com/user-guide/statistics/ and
  https://docs.originlab.com/origin-help/statistics/
* Signal processing: https://docs.originlab.com/origin-help/signal-processing/
* Mathematics: https://docs.originlab.com/origin-help/mathematics/
* NLFit chapter: https://docs.originlab.com/origin-help/nonlinear-curve-fit/ (theory:
  `/origin-help/nlfit-theory/`, multi-curve: `/origin-help/multicurve-multipeak/`, special
  operations: `/origin-help/nlfit-specialoperation/`, comparison: `/origin-help/fit-comparison/`).
* Fitting functions: https://docs.originlab.com/origin-help/curve-fitting-function/
* Peak Analyzer: https://docs.originlab.com/origin-help/peakanalyzer-start/ and
  https://docs.originlab.com/quick-help/tools-for-peakanalysis/
* Import/Export X-Functions: https://docs.originlab.com/x-function/ref/import-and-export/
* Image X-Function categories: https://docs.originlab.com/x-function/ref/image/ and
  https://docs.originlab.com/x-function/ref/adjustments
* Axis scale types: https://docs.originlab.com/origin-help/axesdialog-scale-tab/
* Origin vs OriginPro: https://www.originlab.com/index.aspx?go=Products/Origin/OriginvsOriginPro
* 2025/2025b/2026 features:
  https://www.additive-net.de/en/software/produkte/originlab/originpro/funktionshistorie/neu-in-version-2025b
  and https://www.originlab.com/index.aspx?lm=57
* Validation against NIST StRD: https://blog.originlab.com/validating-origin-analysis-results
  and https://itl.nist.gov/div898/strd/general/dataarchive.html
* PyPI metadata: PyWavelets, h5py, odrpack, scikit-image, pybaselines (pypi.org JSON API).
