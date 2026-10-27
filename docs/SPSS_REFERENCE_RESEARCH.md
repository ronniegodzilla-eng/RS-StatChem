# RS-StatChem vs IBM SPSS: verification fixtures, formula clarifications, remaining gaps

Date: 2026-10-27. The repo was read-only (HEAD f8540dc, v0.9.1). Everything below lives in this scratchpad:
`spss_ref/data/` (datasets), `spss_ref/expected/*.json` (SPSS tables transcribed exactly as printed,
each with its source URL), `spss_ref/notes/algorithms_findings.md` (the full formula notes, with page
numbers and file:line), `menu_dump.txt` (the live menu tree).

**Status:** this report is partial. The three research sub-agents stopped during a network outage.
They were resumed but had not reported back when this file was written. What is on disk is complete
and usable. Algorithms topics 10–15 were not researched yet: Type III with empty cells, post hoc
formulas, bootstrap Sig., CA sign convention, KM/Cox and Monte Carlo defaults. Some fixture pages were
also not yet transcribed: UCLA discriminant, canonical, multinomial, Poisson/NB and probit DAE pages
(their data are downloaded), Kaplan-Meier/Cox, the meta-analysis prediction interval, ANOVA effect-size
CIs and Type III with empty cells.

## 1. Verification fixtures found

Licence summary:
- **UCLA OARC** pages and data files carry no explicit licence. They are (c) UC Regents and are
  "offered for free", which grants no right to redistribute. Policy: **do not commit**. The test
  downloads the file from the URL recorded in each JSON at test time and is skipped when offline.
  The files are cached in `spss_ref/data/ucla/` as .sav and .csv. There are several hsb2 variants;
  use `hsb2.sav` (the 2016/02 upload, byte-identical to /stat/data/hsb2.sav).
- **IBM case-study sample files** (offer.sav, tv-survey.sav, cereal.sav, ships.sav, salary data) ship
  only with an SPSS installation. There is no public IBM download and no redistribution grant. Only
  `public.dhe.ibm.com/.../statistics/samples/` holds MemoryTask.sav and meps1.sav.
  Workarounds:
  - Rebuild the data from printed tables. Probit was rebuilt from the printed "Cell Counts and
    Residuals" table, so it is fully checkable.
  - Or use the public original. Ships comes from McCullagh & Nelder 1989 via the MASS dataset (GPL),
    so the CSV can be committed.
- **IBM Exact Tests manual (v20 PDF)**: small contingency tables printed inline. The counts were
  transcribed to CSV. These are facts, so tiny tables are safe to commit as fixtures, with
  attribution to IBM.

| Procedure | Fixture JSON (`spss_ref/expected/`) | Source | Data | What can be checked |
|---|---|---|---|---|
| Descriptives / Explore | ucla_descriptives_examine.json | stats.oarc.ucla.edu/spss/output/descriptive-statistics/ | hsb2 | N, mean, SD, SE, skew/kurtosis + SEs, trimmed mean, percentiles (Weighted Average + Tukey's hinges), stem-and-leaf |
| One-sample, paired and independent t | ucla_ttest.json | …/spss/output/t-test/ | hsb2 | All tables: t, df, Sig., CI, Levene F=11.133. **Sanity-checked with scipy: every value matches** (t 4.140; paired .867; Welch df 169.70) |
| Correlation | ucla_correlation.json | …/spss/output/correlation/ | hsb2 | Pearson r, Sig., N, pairwise and listwise |
| Linear regression | ucla_regression.json | …/spss/output/regression-analysis/ | hsb2 | R, R², adj. R², SE est., ANOVA, B/SE/Beta/t/Sig. |
| Binary logistic | ucla_logistic_regression.json | …/spss/output/logistic-regression/ | hsb2 | Block 0/1 tables, codings, classification, score test, B/S.E./Wald/Exp(B) |
| One-sample t, NPTESTS Wilcoxon, binomial, χ² GOF | ucla_whatstat_onesample_tests.json | UCLA "What statistical analysis…" (SPSS) | hsb2 | Hypothesis summary, Wilcoxon one-sample, binomial, χ² GOF |
| Independent t + Mann-Whitney | ucla_whatstat_independent_ttest_mannwhitney.json | same | hsb2 | t, U, Z, Sig. |
| Paired t, Wilcoxon, sign, Friedman | ucla_whatstat_paired_wilcoxon_sign_friedman.json | same | hsb2 | Ranks, Z, Sig., sign test, Friedman χ² |
| Crosstabs χ², Fisher, McNemar | ucla_whatstat_crosstabs_chisq_mcnemar.json | same | hsb2 | Pearson/LR/linear-by-linear/continuity correction, Fisher, McNemar |
| One-way ANOVA + Kruskal-Wallis | ucla_whatstat_oneway_anova_kruskal.json | same | hsb2 | ANOVA table, means report, K-W ranks and χ² |
| Factorial ANOVA (GLM) | ucla_whatstat_factorial_anova_glm.json | same | hsb2 | Type III SS table |
| ANCOVA (GLM) | ucla_whatstat_ancova_glm.json | same | hsb2 | Type III SS with covariate |
| MANOVA (GLM) | ucla_whatstat_manova_glm.json | same | hsb2 | Pillai/Wilks/Hotelling/Roy, between-subjects effects |
| Repeated-measures ANOVA | ucla_whatstat_repeated_measures_anova.json | same | rb4wide | Multivariate tests, Mauchly, G-G/H-F/lower-bound, contrasts, between-subjects |
| Ordinal (PLUM) | ucla_whatstat_ordinal_plum.json | same | hsb2 | Model fit, pseudo-R², thresholds/location, test of parallel lines |
| GEE logistic (GENLIN) | ucla_whatstat_genlin_gee_logistic.json | same | exercise | Type III Wald χ², parameter estimates |
| Factor / PCA | ucla_whatstat_factor_pca.json | same | hsb2 | Communalities, total variance explained |
| Spearman | ucla_whatstat_spearman.json | same | hsb2 | rho, Sig. |
| **Probit** (priority) | ibm_cs_probit_offer.json | IBM 29 Case Studies "Using Probit Analysis to Test Promotional Effects" | data/ibm_casestudy/probit_offer_aggregated.csv (rebuilt from the printed cell table) | Pearson χ² GOF, parameter estimates **(settles the heterogeneity-SE question, §2.2)**, natural response rate, confidence limits, relative median potency |
| **Logit loglinear, Analysis of Dispersion** (priority) | ibm_cs_logit_loglinear_dispersion_cereal.json | IBM 29 Case Studies (loglog_cereal_*) | cereal.sav: not public, not redistributable | GOF, measures of association, **entropy/concentration table (settles the ½ factor and the df question, §2.3)**. Needs cereal.sav (from any SPSS install) or a table rebuilt from printed counts |
| **Reliability, parallel model** (priority) | ibm_cs_reliability_parallel_tvsurvey.json | IBM 29 Case Studies (rely_tvsurvey_parallel) | tv-survey.sav: not public | Test for model goodness of fit (χ², df, Sig., log determinants), common/true/error variance, reliability, unbiased reliability. Data not available, so this is structural only, unless the item covariance matrix can be rebuilt from printed item statistics (only 2 rows transcribed so far) |
| **Lambda / GK tau / uncertainty coefficient** (priority) | ibm_exact_crosstabs_nominal_uncertainty_gktau.json | IBM SPSS Exact Tests manual v20, ch.15 figs 15.1–15.6 | data/ibm_exact/party_ally.csv, survey_finance_org.csv | Value, **ASE**, Approx. T, approximate/exact/Monte Carlo Sig. for lambda, tau and UC. **Settles the GK-tau ASE1 and symmetric-lambda tie questions** |
| **Marginal homogeneity** (priority) | ibm_exact_marginal_homogeneity.json | IBM Exact Tests manual v20, ch.5 figs 5.7–5.10 | data/ibm_exact/mh_endometrial_dose.csv, mh_papsmear.csv | Observed MH statistic, mean, SD, std. MH statistic, asymptotic and exact Sig. |
| GENLIN Poisson (ship damage) | ibm_cs_genlin_poisson_ships.json | IBM 29 Case Studies genlin_ships_* | data/ships_mccullagh_nelder/ships.csv (MASS, GPL; commit OK) | Deviance/Pearson GOF, log-likelihood, AIC/BIC, parameter estimates with log(months) offset. Check that the IBM file's coding equals MASS::ships (the IBM case drops structural zeros) |
| Bayesian independent t (salary) | ibm_cs_bayes_independent_salary.json | IBM 29 Case Studies, Bayesian | data not public | Group statistics, posterior characterisation. Structural check only |

Also downloaded but not yet transcribed (`data/ucla/`; the matching DAE pages are saved in `html_ucla/*.txt`):
- probit.sav: UCLA probit regression DAE
- binary.sav: logit DAE
- mlogit.sav, hsbdemo.sav: multinomial
- ologit.sav: ordinal
- poisson_sim.sav: Poisson
- nb_data.sav: negative binomial
- discrim.sav: discriminant
- mmreg.sav: canonical correlation
- manova.sav: one-way MANOVA
- mvreg.sav: multivariate regression
- lahigh.sav, M255.sav, mmr.sav

These are the next fixtures to write, by the same method.

**Still unverified (no source with both data and SPSS output found yet):**
- Moses extreme reactions
- Bayesian Cauchy-scale outputs that come with data
- Meta-analysis prediction interval
- ANOVA effect-size CIs
- Type III with empty cells
- Kaplan-Meier / Cox

Candidate sources still to try:
- IBM Base/Advanced case studies: Kaplan-Meier uses pain_medication.sav; Cox uses customer_dbase / telco.sav. These are not public, so rebuild only if the case study prints the data.
- Andy Field's companion data (SAGE, licence needs checking).
- R reference values, recorded as "R-equivalent, not SPSS":
  - metafor
  - BayesFactor::ttestBF with rscale=1
  - MBESS/effectsize for ANOVA CIs
  - survival

## 2. Formula clarifications (IBM SPSS Statistics Algorithms)

Source: IBM_SPSS_Statistics_Algorithms.pdf, SPSS 29 edition
(https://www.ibm.com/docs/en/SSLVMB_30.0.0/pdf/IBM_SPSS_Statistics_Algorithms.pdf). The SPSS 31
edition was used for the ONEWAY effect sizes. Full formulas, page numbers and quotes are in
`spss_ref/notes/algorithms_findings.md`. Status means our code compared with IBM's documentation.

### 2.1 CROSSTABS
- Lambda ASE0/ASE1, UC ASE0/ASE1, kappa, gamma/d/tau-b/tau-c T: **match**.
- Symmetric lambda ASE1: **differs when row/column maxima are tied**. We use numerical delta;
  IBM's indicator formula takes one index.
- Symmetric Somers' d ASE1: **differs**. IBM gives 2√(DrDc)/(Dr+Dc)·ASE1(τb).
- GK tau ASE1: **unclear**, because IBM's printed formula has an index collision.
- Decide symmetric lambda and GK tau with `ibm_exact_crosstabs_*` before changing code.

### 2.2 PROBIT
Matches IBM:
- Pearson χ² and df
- Fieller fiducial limits with h and t_df
- heterogeneity threshold .15. This is consistent with SPSS output footnotes but not stated in the Algorithms.

Differs from IBM:
- We scale the parameter-table SEs by √h. IBM documents h only for the fiducial/RMP limits. **Check against ibm_cs_probit_offer.json.**
- Parallelism test: we use an LR difference; IBM uses Pearson χ²₀ − Σχ²ⱼ.
- ITERATE echo: IBM's default is max(50, 3(s+1)). Cosmetic only.

### 2.3 Logit loglinear, Analysis of Dispersion
- Entropy: **matches**.
- Concentration: **differs**.
  - IBM: S(B) = N(1−Σπ²) and S(B|A) = Σ m_ij(1−π_i|j), with no ½ factor.
  - Ours halves the values. R is unaffected.
- df: IBM uses case-based df f(r−1), (N−f−1)(r−1) and (N−1)(r−1). Ours differs (genlin.py:1532-1555).

### 2.4 RELIABILITY
Parallel model:
- CV/TV/EV, reliability and unbiased reliability (2+(W−3)A)/(W−1): **match**.
- GOF χ²: **missing IBM's Bartlett-type multiplier** 1 − k(k+1)²(2k−3)/(12(k−1)·df·(W−1)).

Strict parallel: **differs**.
- IBM: CV = var̄ + Σ(T_i−G)²/k; EV = MS within people; R̂ as in the notes.
- Unbiased reliability: (3+(W−3)Rel)/W.
- χ² uses IBM's L with df k(k+3)/2−3.

Hotelling T², Tukey nonadditivity and ICC (McGraw–Wong): **match**.

### 2.5 NPAR
- Moses: **matches**. Control group = first group; h = max(1, ⌊0.05·n_c⌋), capped so that 2h ≤ n_c−2; span rounded; exact formula as documented.
- Marginal homogeneity: **matches**.

### 2.6 GENLIN
- The default REFERENCE=LAST, so the **first** (lowest) category is modelled as the event: **matches**.
- Last factor level is redundant: **matches**.
- Scale default (MLE continuous, 1 discrete) and normal-z Wald CI: **match**.
- Caveat: deviance/Pearson scale should divide by N − (number of non-redundant columns).
- Subpopulation-based GOF for binary data is not implemented. Low priority.

### 2.7 Bayesian
JZS t tests:
- IBM's mixing density is IG(½,½), so **δ ~ Cauchy(0, 1)**. The scale is r = 1, not √2/2.
- Our default r = 0.707 **differs** (bayes.py:84, 329, 387). Fix: default r = 1.

Correlation:
- IBM uses the **Wetzels–Wagenmakers (2012)** JZS integral, or Kang's fractional BF.
- Ours uses the Ly et al. stretched-beta BF: **differs**.

Hyper-prior:
- Wang & Liu, a = −0.75 by default.
- Not implemented in our code.

ANOVA:
- IBM's "Rouder" is the scalar-g R² form with g ~ IG(½, N·s²/2).
- Our default is BayesFactor's multivariate-g with r = .5: **differs**.

Regression:
- Zellner, Zellner–Siow and hyper-g: **match**.

Binomial:
- IBM puts a Beta(a0, b0) prior under H0.
- Ours uses a point null: **differs**.

### 2.8 Meta-analysis
Match:
- τ² estimators HE/HS/DL/SJ/ML/REML/EB, with IBM's iteration settings (100 iterations, 5 step halvings, tol 1e-6, Hedges start)
- Knapp–Hartung with optional truncation, df k−p

Not documented by IBM:
- **The prediction interval is not documented** in either the v29 or the v31 Algorithms.
- Ours follows metafor (z, or t_{k−p} under KH). Higgins et al. 2009 use t_{k−2}.
- This needs real SPSS output. It remains open.

### 2.9 ONEWAY effect sizes (v31 Algorithms)
- CIs come from noncentral-F inversion (Steiger 2004). The η², ε² and ω²-fixed points and limits **match**.
- **ω² random differs.**
  - IBM: point = (F−1)/(F+df_t); limits = (df_t·η²−df_e)/((df_w−df_t·df_e)·η²+df_t·df_e), evaluated at the η² limits.
  - Ours uses n₀ and central-F limits (anova.py:414-426).

### 2.10–2.15 Not yet researched (agent stopped)
- Type III/IV with empty cells
- REGW / Dunnett C / Waller-Duncan
- Bootstrap Sig. and RNG defaults
- CA sign convention
- KM/Cox
- Exact Monte Carlo defaults

Pointers for follow-up: in the same PDF, see "GLM Algorithms" (Type III/IV hypothesis matrices, after Goodnight 1978), "Post Hoc Tests" in the ONEWAY/GLM chapters, "Bootstrapping Algorithms", "Random Number Generators" (MT19937; the default seed is set in Transform > Random Number Generators), "CORRESPONDENCE Algorithms", "KM Algorithms" and "COXREG Algorithms", and "Exact Tests" (Monte Carlo default of 10,000 samples at 99%).

## 3. Remaining feature gaps (SPSS 29–32, Base + all add-ons) vs the current menu tree

Evidence: `menu_tree()` dumped headlessly on 2026-10-27 (`QT_QPA_PLATFORM=offscreen`, script
`scratchpad/dump_menu.py`, output `scratchpad/menu_dump.txt`, 310 lines, no load errors) plus
`rsstatchem/ui/menus.py MENU_ORDER` (labels reserved for items without a module) and a keyword
sweep over `rsstatchem/` (AFT, frailty, GLMM, MARS, TwoStep, CHAID, MLP, KNN, CATREG, PREFSCAL,
control chart, propensity, TURF, RFM, conjoint, Complex Samples, optimal binning, unusual cases,
validation, simulation, kernel ridge, distance correlation, VAR, permutation). An item counts as
missing only when it has neither a menu entry nor a core function.

Constraint carried over from `docs/DEVELOPMENT.md` l.14: app code must not import scikit-learn,
pingouin or seaborn; runtime deps are numpy/scipy/pandas/statsmodels/matplotlib. So every ML
procedure below has to be own numpy code, with sklearn/R used only as test references.

Legend: **M** missing · **IMP** impractical (reason) · priority H/M/L for Indonesian S1–S3 theses.

### 3.1 Data / Transform
| SPSS item (module) | Status | Prio | Note / open-source alternative |
|---|---|---|---|
| Data ▸ Validation ▸ Load Predefined Rules / Define Rules / Validate Data (Data Preparation) | M | L | Rule engine over the expression evaluator (`core/expression.py`); single-variable rules (range, list, integer, missing) and cross-variable rules (expression); output: violations per case/variable, indicator variables. |
| Data ▸ Identify Unusual Cases (Data Preparation, DETECTANOMALY) | M | L | IBM algorithm = TwoStep peer-group clustering + anomaly index (deviation of variable from peer-group norm). Reuse the TwoStep implementation (§3.3). |
| Transform ▸ Optimal Binning (Data Preparation, OPTIMAL BINNING) | M | M | Supervised MDLP (Fayyad–Irani entropy, IBM documents MDLP + pre-binning + hybrid merge). Own numpy implementation; reference: R `discretization::mdlp`. |
| Transform ▸ Prepare Data for Modeling (ADP, Interactive/Automatic) | M | L | Large wizard; partial value only (outlier trimming, date features, merging sparse categories). |
| Data ▸ Orthogonal Design ▸ Generate / Display (Conjoint, ORTHOPLAN/PLANCARDS) + CONJOINT (syntax) | M | L | `core/doe/designs.py` already has Taguchi/Plackett–Burman arrays; a Conjoint dialog = orthogonal array + OLS part-worths + importance; holdout cards and Kendall tau/Pearson for fit. |
| Data ▸ Propensity Score Matching (SPSS 26+ built-in extension) | M | M | Logistic PS (reuse `logistic.py`), nearest-neighbour matching with caliper, with/without replacement, balance table (std. mean difference). Popular in health theses. |
| Data ▸ Case Control Matching (SPSS 26+) | M | L | Exact/fuzzy matching on keys, random draw with seed. |
| Data ▸ Rake Weights (SPSS 29 built-in extension SPSSINC RAKE) | M | L | Iterative proportional fitting of weights to marginal targets (statsmodels has no raking; ~80 lines numpy). |
| Data ▸ Set Measurement Level for Unknown, New Custom Attribute, Adjust String Widths Across Files | M | L | Data-editor conveniences. |
| Transform ▸ Programmability Transformation | IMP | – | Runs user Python; the app must not execute user code. |

### 3.2 Analyze: statistics procedures
| SPSS item (module) | Status | Prio | Note / alternative |
|---|---|---|---|
| Descriptive Statistics ▸ TURF Analysis (Base extension) | M | L | Reach/frequency over all combinations up to size k; exhaustive with heuristic cut-off. Easy. |
| Descriptive/Nonparametric ▸ Chi-Square (v31 menu item) | Pa | L | Legacy Chi-square and NPTESTS one-sample exist; v31 only adds a menu shortcut. |
| Correlate ▸ distance correlation, proximity mapping (v31) | M | L | dCor and its permutation/t-test (Székely–Rizzo); reference R `energy::dcor.test`. Proximity mapping is unclear in public docs. |
| Generalized Linear Models ▸ Multiple Adaptive Regression Splines (v31, MARS) | M | M | Own forward/backward MARS (Friedman 1991, GCV); py-earth is unmaintained; reference R `earth`. |
| Generalized Linear Models ▸ Bayesian Regression Variable Selection (v32 extension) | M | L | |
| Mixed Models ▸ Generalized Linear (GLMM, GENLINMIXED) | M | H | Binary/Poisson/ordinal with random intercepts/slopes. SPSS GENLINMIXED estimates by a PQL-type linearisation (Newsom, PSU handout https://web.pdx.edu/~newsomj/mlrclass/ho_binary%20estimation.pdf: SPSS offers only basic PQL since v19); whether v29–31 added Laplace/AGQ was not confirmed (check the GENLINMIXED Estimation dialog docs). statsmodels `BinomialBayesMixedGLM` is variational Bayes, not equivalent. Own PQL + Laplace on top of `mixed.py`; reference R `lme4::glmer` (Laplace, nAGQ). |
| Regression ▸ Automatic Linear Modeling (LINEAR) | M | L | Forward stepwise / best subsets with AICC/F/adj. R² criteria, outlier handling, bagging/boosting ensemble; data prep part overlaps ADP. |
| Regression ▸ Partial Least Squares (PLS extension, needs Python) | M | M | `pls.py` is PLS-SEM only. NIPALS PLS1/PLS2 with VIP, proportion of variance explained, scores/loadings: ~200 lines numpy. Reference sklearn `PLSRegression` (test only). |
| Regression ▸ Optimal Scaling (CATREG) (Categories) | M | M | Alternating least squares with nominal/ordinal/spline transformations. Reference R `Gifi`/`aspect`; IBM Categories algorithms chapter documents the ALS. |
| Regression ▸ Kernel Ridge (SPSS 28 extension, sklearn-based in IBM's own implementation) | M | L | Closed form `(K + αI)⁻¹y`; kernels linear/rbf/poly/sigmoid/laplacian; grid search CV. |
| Regression ▸ Permutation Tests (v32 extension) / Compare Means ▸ Independent Samples Permutation t (v32) | M | L | scipy `permutation_test` available. |
| Neural Networks ▸ Multilayer Perceptron / Radial Basis Function (Neural Networks module) | M | M | Own numpy MLP (1–2 hidden layers, tanh/sigmoid, softmax/identity output, scaled conjugate gradient or mini-batch GD, training/testing/holdout partitions, importance analysis); RBF = k-means centres + least-squares output layer. IBM's random initialisation cannot be reproduced exactly, so verification is structural (fit statistics within tolerance against sklearn `MLPClassifier`). |
| Classify ▸ TwoStep Cluster | M | M | BIRCH CF-tree pre-clustering + hierarchical merge with log-likelihood distance, BIC/AIC auto-k (IBM Algorithms documents it; Chiu et al. 2001). Exact equality with SPSS depends on case order. |
| Classify ▸ Cluster Silhouettes (extension) | M | L | Easy once TwoStep/K-means exist. |
| Classify ▸ Tree (CHAID, Exhaustive CHAID, CRT, QUEST) (Decision Trees) | M | H | Used in theses for segmentation. CHAID: Bonferroni-adjusted chi-square merge/split (Kass 1980) is fully documented by IBM; CRT: Gini/twoing with cost-complexity pruning; QUEST: Loh–Shih. Own implementation; references R `CHAID` (R-Forge), `rpart`, sklearn `DecisionTreeClassifier` (CRT only, test only). |
| Classify ▸ Nearest Neighbor (KNNE) | M | L | Own numpy kNN with feature selection forward wrapper and k selection by V-fold CV. |
| Classify ▸ Mixed Type Cluster with Variable Selection (v32 extension) | M | L | |
| Dimension Reduction ▸ Optimal Scaling: CATPCA, MCA (Multiple Correspondence), OVERALS (Nonlinear Canonical) (Categories) | M | M | MCA = homogeneity analysis (closed form via indicator-matrix SVD; matches SPSS with "variable principal" normalisation); CATPCA/OVERALS = ALS. Reference R `Gifi::princals`, `homals`, `overals`. |
| Scale ▸ Multidimensional Unfolding (PREFSCAL) | M | L | Penalised stress (Busing et al. 2005); reference R `smacof::unfolding`. |
| Nonparametric ▸ Quade nonparametric ANCOVA (extension) | M | L | Rank regression residuals + one-way ANOVA. |
| Forecasting ▸ Create Temporal Causal Models / Apply | IMP | – | Group-LASSO Granger graph over many series; large and rarely used in theses. |
| Forecasting ▸ Vector Autoregressive (VAR) (v32) | M | L | statsmodels `VAR` (runtime dep) gives estimates, IRF, FEVD, Granger; easy wrapper. |
| Survival ▸ Parametric Accelerated Failure Time (AFT) (v29) | M | M | Weibull, exponential, log-normal, log-logistic AFT by ML; reference `lifelines` (test only) / R `survival::survreg`. Labels already reserved in `menus.py` l.160. |
| Survival ▸ Parametric Shared Frailty (v29.0.1) | M | L | Gamma/inverse-Gaussian frailty with parametric baseline; reference R `frailtypack`/`parfm`. |
| Complex Samples (whole module: Plan, Prepare for Analysis, Frequencies, Descriptives, Crosstabs, Ratios, General Linear Model, Logistic, Ordinal, Cox) | M | M | Taylor linearisation with strata/PSU/weights/FPC, design effects; sampling wizard (SRS/PPS). Own code; references R `survey` (svymean, svyglm, svyolr, svycoxph). |
| Simulation (SIMPLAN/SIMRUN) | M | L | Monte Carlo of a fitted model with input distributions fitted to data, sensitivity/tornado charts. |
| Quality Control ▸ Control Charts (X-bar/R/S, individuals/moving range, p, np, c, u), Pareto Charts | M | M | Constants A2/D3/D4/B3/B4/c4/d2 are tabulated; capability Cp/Cpk/Pp/Ppk; Western Electric rules. Valuable for chemistry/industry theses; `core/doe/plots.py` already has Pareto for effects only. |
| Direct Marketing (RFM Analysis, Cluster, Prospect Profiles, Postal Code Response Rates, Propensity to Purchase, Control Package Test) | M | L | RFM binning (nested/independent) and Control Package Test (two-proportion z with Bonferroni-like comparisons) are easy; the others wrap existing procedures. |
| Spatial and Temporal Modeling (Spatial Association Rules, Spatial Temporal Prediction, Spatial Modeling) | IMP | – | Needs map/shapefile stack and geodata. |
| Mediation Analysis (v32 native MEDIATION) | Pa | L | PROCESS models 1/4/7/14 already cover thesis use; the SPSS v32 procedure itself is not documented in detail publicly yet. |
| Genomic Analysis (v32) | IMP | – | Domain specific, outside the app's audience. |

### 3.3 Graphs / Utilities / File
| SPSS item | Status | Prio | Note |
|---|---|---|---|
| Graphs ▸ Weibull Plot (v28) | M | L | Median ranks (Benard) on Weibull paper + ML/regression fit. Easy. |
| Graphs ▸ Compare Subgroups (v26), Relationship Map (v28) | M | L | |
| Graphs ▸ Graphboard Template Chooser | IMP | – | IBM visualization templates (.viztemplate). The Chart Builder gallery covers the use. |
| Utilities ▸ OMS Control Panel / OMS Identifiers, Run Script, Scoring Wizard, Merge Model XML, Production Facility, Extension Hub, Custom Dialog Builder | IMP | – | Need the SPSS syntax/OMS/PMML/extension infrastructure (see gap analysis §3.4). |
| File ▸ New/Open Syntax, Workbook | IMP | – | Full SPSS language not planned; the replayable RS job remains the substitute. |

## 4. Implementation work packages (disjoint file ownership; new files only unless noted)

Each new module exports `MENUS` and `selftest()` and is added to `EXTENSIONS` in `ui/menus.py`. That
one line belongs to the integrator. App code must not import sklearn, pingouin or lifelines; those
are test references only.

### WP7: SPSS-conformance fixes and the fixture test suite (highest value, about 2–3 days)
- **Scope:** the "differs" items in §2.
  - probit SE scaling and parallelism test
  - concentration ½ and dispersion df
  - reliability GOF multiplier and strict-parallel model
  - Bayes default r = 1, Wetzels–Wagenmakers correlation BF, IBM-Rouder ANOVA, Beta-null binomial
  - ω²-random
  - symmetric lambda ties and Somers' d ASE1
- Each change is guarded by the fixture JSONs.
- **Files:**
  - Edits to `core/stats/genlin.py`, `reliability.py`, `bayes.py`, `anova.py` and `crosstabs.py`. The owners of these files hand them to this WP.
  - New `tests/test_spss_reference.py`.
  - New `tests/spss_ref/` holding only the redistributable fixtures: ibm_exact CSVs, ships, the rebuilt probit CSV and the expected JSONs.
  - New `tests/spss_ref/fetch.py`, which downloads the UCLA data at test time and skips the test when offline.
- **Verification:** every fixture value must match at the printed decimals (compare formatted strings).

### WP8: Survival and mixed extensions (about 4–5 days)
- **Scope:**
  - Parametric AFT: Weibull, exponential, log-normal, log-logistic.
  - Parametric Shared Frailty: gamma.
  - GLMM (GENLINMIXED subset): binary, Poisson, ordinal; random intercepts and slopes; PQL plus an optional Laplace.
- **Files:**
  - new `core/stats/survival_param.py`, `core/stats/glmm.py`
  - new `ui/dialogs/survival_param.py`, `ui/dialogs/glmm.py`
- **Libraries:** scipy.optimize and numpy; reuses `mixed.py` and `genlin.py` read-only.
- **Verification:**
  - AFT against lifelines and R `survreg`
  - GLMM against R `lme4::glmer` / `MASS::glmmPQL`, stored as committed fixture numbers
  - R output is labelled as R-equivalent, not SPSS

### WP9: Classification and ML (the Decision Trees, Neural Networks and Classify add-ons, about 6–8 days)
- **Scope:**
  - Tree: CHAID and Exhaustive CHAID with Bonferroni, CRT with cost-complexity pruning, QUEST as an optional extra
  - TwoStep Cluster: CF-tree and log-likelihood distance with BIC auto-k
  - Cluster Silhouettes
  - Nearest Neighbor
  - MLP and RBF
  - Kernel Ridge
  - PLS regression with VIP
  - MARS under Generalized Linear Models
- **Files:**
  - new `core/stats/trees.py`, `twostep.py`, `neural.py`, `knn.py`, `pls_regression.py`, `mars.py`, `kernel_ridge.py`
  - new `ui/dialogs/classify_ml.py`, `ui/dialogs/regression_ml.py`
- **Libraries:** numpy and scipy only.
- **Verification:**
  - CRT, kNN, PLS, kernel ridge and MLP against sklearn in tests
  - CHAID against R `CHAID`
  - MARS against R `earth`
  - TwoStep is checked structurally against the IBM Algorithms TwoStep chapter, because SPSS results depend on case order

### WP10: Categories module and Complex Samples (about 6–8 days)
- **Scope:**
  - CATREG, CATPCA, MCA, OVERALS, PREFSCAL (ALS following the IBM Categories algorithms)
  - Complex Samples: Plan and Prepare wizards; CS Frequencies/Descriptives/Crosstabs/Ratios/GLM/Logistic/Ordinal/Cox with Taylor linearisation, strata, PSU, FPC and design effects
- **Files:**
  - new `core/stats/optscale.py`, `core/stats/complex_samples.py`
  - new `ui/dialogs/optscale.py`, `ui/dialogs/complex_samples.py`
- **Libraries:** numpy, scipy, statsmodels.
- **Verification:**
  - R `Gifi` (princals, homals, overals) and `smacof::unfolding`
  - R `survey` (svymean, svyglm, svyolr, svycoxph)
  - The MCA closed form is checked against correspondence analysis of the indicator matrix

### WP11: Data preparation, quality control, marketing and small extensions (about 4–5 days)
- **Scope:**
  - Validation rules / Validate Data
  - Identify Unusual Cases, which reuses WP9 TwoStep and must be scheduled after WP9
  - Optimal Binning (MDLP)
  - Propensity Score Matching and Case Control Matching
  - Rake Weights
  - Orthogonal Design and Conjoint
  - Quality Control: control charts and Pareto, Cp/Cpk
  - Direct Marketing: RFM and Control Package Test
  - TURF
  - VAR, through statsmodels VAR
  - Weibull Plot
  - distance correlation
  - permutation tests
  - Simulation, as a light version
- **Files:**
  - new `core/stats/dataprep.py`, `qc.py`, `marketing.py`, `conjoint.py`, `turf.py`, `var_model.py`, `simulation.py`
  - new `ui/dialogs/dataprep.py`, `ui/dialogs/qc.py`, `ui/dialogs/marketing.py`, `ui/dialogs/misc_ext.py`
- **Verification:**
  - Control-chart constants against the ASTM/Montgomery tables
  - MDLP against R `discretization`
  - PSM against R `MatchIt` (nearest, caliper)
  - VAR against statsmodels
  - dCor against R `energy`

Out of scope, with reasons in §3:
- Programmability Transformation
- OMS, scripting, Production Facility
- Scoring Wizard / PMML
- Graphboard
- Temporal Causal Models
- Spatial modelling
- Genomic Analysis
- syntax and workbook documents

## Addendum: late report from the non-UCLA fixtures agent

New fixtures in `spss_ref/expected/`:
- **ibm_cs_kaplan_meier_painmeds.json** (IBM v29 case study). It covers means and medians with their SEs and CIs; log-rank .379 (p .538); Breslow .748 (p .387); Tarone-Ware .705 (p .401). pain_medication.sav is not public, so this is a structural check only.
- **mdpi2022_meta_analysis_STUB.json.** The data are in `data/mdpi_psych2022_meta/`, licensed CC BY 4.0, so they can be committed. Expected values are not transcribed yet: the SPSS output appears only as images (Figs 7, 11, 18, 20). This is the best route to settling the prediction-interval question.

Confirmations from the agent's own spot checks:
- **Probit.** An independent ML fit reproduces IBM's b, a, C, ED50 and Pearson χ² (8.934, df 16) exactly from `probit_offer_aggregated.csv`. The parallelism value of 2.060 is printed, so WP7 can test it directly.
- **Analysis of Dispersion.** IBM's printed totals (entropy .197, concentration .189) are reproduced exactly from the marginals {231, 310, 339} using N(1−Σp²) with **no ½ factor**. This confirms that our concentration values are halved.
- **GENLIN ships.** statsmodels reproduces every B, SE, the deviance 38.695 and Pearson 42.275. The IBM log-likelihood (−68.281) uses a convention not yet explained (statsmodels gives −40.379 unscaled), so check how our genlin.py reports it.
- **Bayesian independent t.** The posterior variance matches the unequal-variance scaled-t formula.
- **Exact Tests tables.** U, GK tau, phi and V were recomputed and match. The manual prints **no lambda value**, so symmetric lambda with ties stays unverified.

Still unverified:
- Moses
- strict-parallel reliability
- GENLIN binary reference category
- ANOVA effect-size CIs
- Type III with empty cells
- Cox: the IBM case-study data (telco.sav, recidivism.sav) are not public

Easy next wins:
- More inline tables in the Exact Tests manual: Kendall's W, Cochran's Q, J-T, gamma/d/tau and kappa.
- The MDPI meta-analysis figures.
