# IBM SPSS Statistics Algorithms — formula findings for RS-StatChem

Source PDF: `IBM_SPSS_Statistics_Algorithms.pdf`, downloaded 2026-10-27 from
https://www.ibm.com/docs/en/SSLVMB_30.0.0/pdf/IBM_SPSS_Statistics_Algorithms.pdf
(the PDF's edition notice says "This edition applies to IBM SPSS Statistics 29", (c) 1989, 2022; 1402 pages).
Local copy: `spss_ref/algorithms_pdf/algo30.pdf`; text dump `spss_ref/algorithms_txt/algo30.txt`;
rendered pages `spss_ref/pages/pNNNN.png`. Page numbers below are **PDF page indices** (1-based) of
that file. Formulas are typeset as images in the PDF, so they were read from rendered pages;
quotes are from the running text.

Code paths are relative to `/Users/ronisaputra/Aplikasi statistik/rsstatchem/core/stats/`.

---

## 1. CROSSTABS — lambda, Goodman–Kruskal tau, uncertainty coefficient, ordinal measures, kappa

Citation: Algorithms (SPSS 29 ed.), "CROSSTABS Algorithms" > "Measures of Proportional Reduction in
Predictive Error" (pp. 225–228), "Cohen's Kappa" (p. 228–229), "Kendall's Tau-b and Tau-c", "Gamma",
"Somers' d" (pp. 229–232), "Eta" (p. 233).

### IBM formulas (notation: f_ij cells, r_i row totals, c_j column totals, W total;
f_im = max in row i, f_mj = max in column j, c_m = max column total, r_m = max row total)

- lambda_{Y|X} = (sum_i f_im - c_m) / (W - c_m)
  - ASE0 = sqrt( sum_ij f_ij (d_ij - d_j)^2 - (sum_i f_im - c_m)^2 / W ) / (W - c_m)
  - ASE1 = sqrt( sum_ij f_ij (d_ij - d_j + lambda d_j)^2 - W lambda^2 ) / (W - c_m)
    with d_ij = 1 if j is the column index of f_im, d_j = 1 if j is the index of c_m.
- symmetric lambda = (sum f_im + sum f_mj - c_m - r_m) / (2W - r_m - c_m)
  - ASE0 = sqrt( sum f_ij (d^r_ij + d^c_ij - d^r_i - d^c_j)^2 - (sum f_im + sum f_mj - c_m - r_m)^2/W ) / (2W - r_m - c_m)
  - ASE1 = sqrt( sum f_ij [d^r_ij + d^c_ij - d^r_i - d^c_j + lambda (d^r_i + d^c_j)]^2 - 4 W lambda^2 ) / (2W - r_m - c_m)
- Goodman–Kruskal tau_{Y|X} = (W sum f_ij^2/r_i - sum c_j^2) / (W^2 - sum c_j^2); only ASE1 printed:
  ASE1 = sqrt( 4/delta^4 sum_ij f_ij { (v - delta)(1/r_i sum_j f_ij c_j - c_j) - W delta (1/r_i^2 sum_j f_ij^2 - 1/r_i f_ij) }^2 ),
  delta = W^2 - sum c_j^2, v = W sum f_ij^2/r_i - sum c_j^2.
  Significance: (W-1)(C-1) tau_{Y|X} ~ chi2_{(R-1)(C-1)} (no ASE0 / Approx. T for tau).
- Uncertainty coefficient U_{Y|X} = (U(X)+U(Y)-U(XY))/U(Y) (natural logs)
  - ASE1 = 1/(W U(Y)^2) sqrt( sum f_ij { U(Y) ln(f_ij/r_i) + [U(X)-U(XY)] ln(c_j/W) }^2 )
  - ASE0 = sqrt( P - W[U(X)+U(Y)-U(XY)]^2 ) / (W U(Y)),  P = sum f_ij [ln(c_j r_i/(W f_ij))]^2
  - symmetric U = 2[U(X)+U(Y)-U(XY)]/(U(X)+U(Y)); ASE0 = 2/(W[U(X)+U(Y)]) sqrt(P - W[...]^2)
- Ordinal (P, Q = 2x usual concordant/discordant; D_r = W^2 - sum r_i^2, D_c = W^2 - sum c_j^2):
  - gamma ASE0 = 2/(P+Q) sqrt( sum f_ij (C_ij-D_ij)^2 - (P-Q)^2/W )
  - Somers' d_{Y|X} = (P-Q)/D_r, ASE0 = 2/D_r sqrt(same); symmetric d ASE0 = 4/(D_c+D_r) sqrt(same)
  - tau-b ASE0 = 2 sqrt( [sum f(C-D)^2 - (P-Q)^2/W] / (D_r D_c) )
  - tau-c ASE1 = ASE0 = 2q/((q-1)W^2) sqrt( sum f(C-D)^2 - (P-Q)^2/W ), q = min(R, C)
  - symmetric Somers' d ASE1 printed as  2 sigma^2_{tau_b} sqrt(D_r D_c)/(D_r + D_c)
    ("where sigma^2_{tau_b} is the variance of Kendall's tau_b" — dimensionally this must mean the
    ASE1 of tau-b, i.e. ASE1(d_sym) = 2 sqrt(D_r D_c)/(D_r+D_c) * ASE1(tau_b)).
- Kappa: var1 = Fleiss–Cohen–Everitt form (in counts);
  var0 = [W^2 sum r_i c_i + (sum r_i c_i)^2 - W sum r_i c_i (r_i + c_i)] / (W (W^2 - sum r_i c_i)^2).
- Which ASE feeds "Approx. T": the PDF does not state it in words, but it prints ASE0 ("or, under
  the hypothesis of independence") for lambda, UC, gamma, d, tau-b, tau-c, kappa. Note that
  value/ASE0 for gamma, d (all three), tau-b and tau-c all reduce algebraically to the same
  T = (P - Q) / (2 sqrt( sum f(C-D)^2 - (P-Q)^2/W )). Tau: chi-square based, no T. Eta: no ASE/T.

Quote (p. 227): "The significance level is based on the chi-square distribution, since"
Quote (p. 231): "or, under the hypothesis of independence,"

### Our code status
- Lambda ASE0 and ASE1 (asymmetric, closed Goodman–Kruskal form) — **matches**; verified numerically
  against the IBM delta formulas (crosstabs.py:174-217). Symmetric ASE1 via numerical delta
  (crosstabs.py:208) matches IBM except when the row/column maxima are **tied** (one random table
  gave 0.0509 vs 0.0582); IBM's indicator formula picks one index, central differences average.
- Lambda T uses ASE0 (crosstabs.py:212-214) — **matches**.
- GK tau: value and chi-square Sig. **match** (crosstabs.py:219-225). ASE1 is the exact multinomial
  delta method (crosstabs.py:69-82); the IBM printed ASE1 formula, read literally, gives different
  numbers (e.g. 0.0233 vs 0.0312) — **unclear**: the printed formula has an index collision (j used
  both inside and outside the sum) and is not the delta method. Validate against a real SPSS run
  before changing.
- UC: ASE0 (crosstabs.py:228-242) — **matches** IBM exactly (algebraically identical: var_info =
  (sum p g^2 - I^2)/W). ASE1 numerical delta = IBM ASE1 (both are the delta method). Sig. from
  likelihood-ratio chi-square — matches SPSS footnote behaviour.
- Gamma, d, tau-b, tau-c: single shared T = (P-Q)/(2 s0) (crosstabs.py:256-260) — **matches**
  (all IBM ASE0 ratios reduce to it). tau-c ASE1 (crosstabs.py:283) matches.
- Symmetric Somers' d ASE1 (crosstabs.py:288-291) uses a full delta-method formula — **differs**
  from IBM's printed 2 sqrt(DrDc)/(Dr+Dc) * ASE1(tau_b). Recommended: test both against SPSS output;
  if SPSS matches IBM's form, replace with `2*sqrt(dr*dc)/(dr+dc) * ase1_tau_b`.
- Kappa ASE1/ASE0/T (crosstabs.py:351-371) — **matches** (verified numerically, identical to 1e-6).

Recommended fixes: (a) symmetric-lambda ties: implement IBM indicator formula directly (pick the
first max index) instead of delta_ase; (b) check symmetric Somers' d ASE1 and GK tau ASE1 against
an SPSS reference table.

## 2. PROBIT — SEs, natural response rate, heterogeneity factor, fiducial limits, convergence

Citation: Algorithms (SPSS 29 ed.), "PROBIT Algorithms" (pp. 967–972): "Natural Response Rate",
"Initial Values", "Criteria", "Asymptotic Covariance Matrix", "Frequency Table and Goodness of
Fit", "Fiducial Limits for Effective dose x", "Relative Median Potency", "Parallelism Test".

### IBM
- Model: P_i = c + (1 - c) F(x_i' beta); ML by NPSOL (Gill, Murray, Saunders, Wright); only bound
  0 <= c < 1. Fixed NATRES: c fixed at the user value.
- Initial values: all beta = 0; c0 = control-group response ratio, else the minimum r_i/n_i.
- Criteria: ITER default = max(50, 3(s+1)); CONV = CNLR OPTOLERANCE (numeric default not stated in
  the PROBIT chapter).
- Covariance of (c, beta) = I^{-1} (negative Hessian of L; analytic second derivatives given).
  **No heterogeneity scaling of the parameter covariance is documented.**
- Pearson chi2 = sum (r_i - E_i)^2 / (E_i (1 - P_i)), E_i = n_i P_i; df = m - s if q = 1
  (printed "(q-1)m - s if q >= 2", which looks like a typo); s = p + q (+1 if c estimated).
- Fiducial limits (only with a single covariate), P = .01–.09, .10–.90 by .05, .91–.99, always 95%:
  x_j = (y - a_j)/g_hat; limits = x_j + g/(1-g) (x_j + cov(a_j,gamma)/v(gamma))
  +- t/(gamma(1-g)) sqrt{ [v(a_j) + 2 x_j cov + x_j^2 v(gamma) - g (v(a_j) - cov^2/v(gamma))] h* },
  g = t^2 v(gamma)/gamma^2 (x h when heterogeneity used), t = 1.96 or t_{0.025, df},
  h = chi2/df, h* = 1 or h. RMP: analogous Fieller with v11, v12, v22.
- "The heterogeneity factor is used if the Pearson chi-square statistic is significant." — the
  threshold (0.15) is **not stated in the Algorithms PDF** (it appears only in SPSS output
  footnotes "Since the significance level is less than .150 ..."; Command Syntax Reference).
- Parallelism test: chi2 = chi2_0 - sum_j chi2_j (Pearson chi-squares: common-slope model minus
  separate-group fits), df = q - 1.

Quote (p. 971): "The heterogeneity factor is used if the Pearson chi-square statistic is significant."
Quote (p. 969): "The default value is max(50, 3(s+1))."

### Our code (genlin.py:1109-1350)
- Pearson chi2, df = m - s, probability grid, Fieller limits with V*h and t_{df} — **match**
  (genlin.py:1094-1095, 1206-1212, 1336-1350; Fieller quadratic == IBM closed form).
- Parameter-table SEs and CIs are multiplied by sqrt(h) and use t_df when heterogeneity applies
  (genlin.py:1211-1213, cov_h) — **differs from the Algorithms text** (IBM only documents h in the
  fiducial/RMP limits). Verify against SPSS output; if SPSS prints unscaled SEs, use `cov` (not
  `cov_h`) for the Parameter Estimates table and z-based CIs.
- Parallelism test uses a likelihood-ratio difference 2*(LL_sep - LL_common) (genlin.py:1236-1250)
  — **differs**: IBM uses the Pearson chi-square difference chi2_0 - sum chi2_j. Fix: fit each group
  separately (c fixed at c_hat), compute each group's Pearson chi2_j, report chi2_0 - sum chi2_j.
- Syntax echo says ITERATE(20) (genlin.py:1125); Algorithms says max(50, 3(s+1)) — cosmetic.
- het threshold 0.15 (genlin.py:1113) — consistent with SPSS output, but not documented in Algorithms.
- Fiducial `level` is user-selectable; SPSS always prints 95% — fine.

## 3. GENLOG / LOGLINEAR logit — Analysis of Dispersion

Citation: Algorithms (SPSS 29 ed.), "GENLOG Multinomial Loglinear and Logit Models Algorithms" >
"Analysis of Dispersion (Logit Models Only)", "Degrees of Freedom" (pp. 560–561); also "LOGLINEAR
Algorithms" > "Analysis of Dispersion" (p. 659), following Haberman (1982, JASA 77:568–580).

### IBM (i = response category 1..r, j = covariate pattern 1..c, N_j pattern totals, N total)
- pi_i = sum_j m_ij / sum_j N_j ;  pi_{i|j} = m_ij / N_j   (fitted counts m_ij)
- Entropy: S(B) = -N sum_i pi_i log pi_i ;  S(B|A) = -sum_j N_j sum_i pi_{i|j} log pi_{i|j}
- Concentration: S(B) = N (1 - sum_i pi_i^2) ;  S(B|A) = sum_j N_j (1 - sum_i pi_{i|j}^2)
- S(A) = S(B) - S(B|A); R = S(A)/S(B)
- df: Model f(r-1); Residual (N - f - 1)(r-1); Total (N - 1)(r-1), f = p minus nonredundant
  columns for the main effects of the dependent factors.
Quote (p. 560): "The analysis of dispersion is based on two types of dispersion: entropy and concentration."

### Our code (genlin.py:1532-1555)
- Entropy total and residual — **match** (ours uses sum n_ij log pi_ij which equals sum m_ij log pi_ij
  at the ML fit of a model containing the pattern margin; not exactly equal when delta is added for
  saturated models).
- Concentration: ours = N/2 (1 - sum p^2) and 0.5 sum n_ij (1 - pi_ij) (genlin.py:1542-1543) —
  **differs**: IBM has no 1/2 factor and uses fitted counts: S(B|A) = N - sum_ij m_ij pi_ij.
  R is unaffected by the 1/2 but the table values are halved. Fix unless an SPSS output confirms
  the 1/2: `con_tot = N*(1-sum(pj**2))`, `con_res = sum(exp_*(1-pi))`.
- df column: ours Model = X.shape[1]-1-len(margin), Residual = cell-based GOF df, Total blank
  (genlin.py:1546-1549) — **differs** from IBM's case-based df (f(r-1), (N-f-1)(r-1), (N-1)(r-1)).

## 4. RELIABILITY — parallel / strict parallel, ICC, Hotelling's T², Tukey additivity

Citation: Algorithms (SPSS 29 ed.), "RELIABILITY Algorithms": "Tukey's Test for Nonadditivity"
(p. 1071), "Parallel Model (Kristof 1963)" (pp. 1075–1076), "Strict Parallel (Kristof 1963)"
(pp. 1076–1078), "Analysis of Variance Table" and "Hotelling's T-Squared (Winer, 1971)"
(pp. 1078–1079), "Intraclass Correlation Coefficients" (pp. 1080–1092).
Notation: k items, W cases, S_i^2 item variances, v_ij covariances, var-bar = mean S_i^2,
cov-bar = mean v_ij (i<j), T_i item means, G grand mean, V the item covariance matrix.

### IBM — Parallel
- CV = var-bar; TV = cov-bar; EV = var-bar - cov-bar; R-hat = cov-bar/var-bar;
  reliability A = k/(k-1) (1 - sum S_i^2 / S_p^2)  (= k R/(1+(k-1)R)).
- Unbiased: A-hat = (2 + (W-3) A) / (W - 1).
- GOF: chi2 = -(W-1) [1 - k(k+1)^2(2k-3) / (12(k-1)(k(k+1)/2 - 2)(W-1))] log L,
  L = |V| / ((var-bar - cov-bar)^{k-1} (var-bar + (k-1) cov-bar))  [printed "(k+1)" in L; log C
  uses (k-1)], df = k(k+1)/2 - 2. log UC = log|V|; log C = log((var-cov)^{k-1}(var+(k-1)cov)).
### IBM — Strict parallel
- CV = var-bar + (1/k) sum (T_i - G)^2; EV = MS_within people; TV = CV - EV
- R-hat = [cov-bar - sum(T_i-G)^2/((k-1)k)] / [var-bar + sum(T_i-G)^2/k]
- Rel = k R-hat / (1 + (k-1) R-hat); **Unbiased: Rel_u = (3 + (W-3) Rel) / W**
- chi2 = -(W-1)[1 - k(k+1)^2(2k-3) / (12(k-1)(k(k+3)/2 - 3)(W-1))] log L,
  L = |V| / ((var + (k-1)cov) (var - cov + (1/k) sum (T_i-G)^2)^{k-1}), df = k(k+3)/2 - 3.
  (log C on p.1078 prints 1/(k-1) instead of 1/k inside — internal inconsistency in the manual.)
Quote (p. 1076): "All mean squares are calculated as described in the analysis of variance table."
### IBM — others
- Hotelling: T^2 = W Y' B^{-1} Y, Y_i = T_i - T_k, B = C V C'; F = (W-k+1)/((W-1)(k-1)) T^2,
  df (k-1, W-k+1); not done if W < k or |B| < 1e-30.
- Tukey: D = [sum (T_i-G)^2][sum w_j (P_j-G)^2]; M = sum w_j (P_j-G) sum_i X_ji (T_i-G);
  SS_nonadd = M^2/D (df 1); F = MS_nonadd/MS_balance, df (1, (W-1)(k-1)-1); B = M/D; power
  p = 1 - B G.
- ICC (McGraw & Wong 1996): one-way ICC(1) = (MSBP-MSWP)/(MSBP+(k-1)MSWP), CI via F_{p/w};
  two-way type A single = (MSBP-MSRes)/(MSBP+(k-1)MSRes+k(MSBM-MSRes)/W) with Satterthwaite nu
  (a = k ICC/(W(1-ICC)), b = 1 + k ICC (W-1)/(W(1-ICC))); type A average = (MSBP-MSRes)/
  (MSBP+(MSBM-MSRes)/W); test F = MSBP/(a0 MSBM + b0 MSRes).

### Our code
- Parallel (reliability.py:95-122): CV/TV/EV/R/reliability/unbiased/logdets/df — **match**.
  GOF chi2 = (n-1)(log|S0| - log|S| + tr(S0^{-1}S) - k) = -(n-1) log L — **differs**: missing
  IBM's Bartlett-type multiplier [1 - k(k+1)^2(2k-3)/(12(k-1)(df)(W-1))]. Fix: multiply chi by
  `1 - k*(k+1)**2*(2*k-3) / (12*(k-1)*dof*(n-1))` (dof = k(k+1)/2-2 parallel, k(k+3)/2-3 strict).
- Strict parallel (reliability.py:102-119): ours inflates sum d^2 by n/(n-1) in CV/TV and uses an
  ML chi2 + n d' S0^{-1} d — **differs** from IBM (CV = var + sum d^2/k; EV = MS_within;
  R-hat formula above). Unbiased uses the parallel formula 1-(n-3)/(n-1)(1-rel) — **differs**:
  IBM strict = (3 + (W-3)Rel)/W. df matches (k(k+3)/2-3). Fix: implement the strict-parallel
  quantities literally as above; chi2 = -(W-1)*corr*log L with IBM's L.
- Hotelling T^2 (reliability.py:172-183) — **matches** (contrast choice is invariant).
- Tukey nonadditivity (reliability.py:186-201) — **matches** (unweighted).
- ICC (agreement.py:48-110) — **matches** IBM/McGraw–Wong for estimates, CIs and F tests
  (spot-checked one-way, consistency and type A forms).

## 5. NPAR TESTS — Moses extreme reactions; marginal homogeneity

Citation: Algorithms (SPSS 29 ed.), "NPAR TESTS Algorithms" > "Moses Test of Extreme Reaction"
(p. 850); "Nonparametric Tests Algorithms" > "Moses Test of Extreme Reactions" / "Censoring the
Range" (pp. 813–814) and "Marginal Homogeneity" (pp. 821–822).

### IBM
- Control group = first group (first value in ascending order). Joint ranks (mid-ranks).
  SPAN = Rank(largest control) - Rank(smallest control) + 1, rounded to nearest integer.
- Exact one-tailed: P(SPAN <= n_c - 2h + g) = sum_{i=0}^{g} C(i + n_c - 2h - 2, i) C(n_e + 2h + 1 - i, n_e - i) / C(n_c + n_e, n_c), h = 0 for the untrimmed span.
- Trimming: drop h lowest and h highest control ranks; default h = max(1, floor(0.05 n_c));
  if 2h > n_c - 2 use the largest h with 2h <= n_c - 2 (NPTESTS chapter); same formula with h.
- Marginal homogeneity (ordinal, scores w_u = category values): over off-diagonal cells
  T = sum_u sum_{v != u} w_u n_uv; mu_T = 1/2 sum sum (w_u + w_v) n_uv;
  sigma_T^2 = sum sum ((w_u - w_v)/2)^2 n_uv; T* = (T - mu_T)/sigma_T; p = 2(1 - Phi(|T*|)).
Quote (p. 850): "h is taken to be the integer part of 0.05n_c or 1, whichever is greater."
Quote (p. 822): "Any linear transformation of scores produces the same standardized test statistic"

### Our code
- nptests.py:311-341 (moses_span_pvalue, moses_test) — **matches** (span, rounding, g range,
  default h, cap (m-2)//2, trimmed p with h).
- nptests.py:648-662 (marginal_homogeneity_test) — **matches** (obs = sum x1 over off-diagonal
  pairs, mean, sd, z, two-sided p). Frequency weights must be applied by replicating cases.

## 6. GENLIN — binary reference, factor reference, scale parameter, Wald CI

Citation: Algorithms (SPSS 29 ed.), "GENLIN Algorithms" > "Notation"/"Model" (pp. 489–490),
"Scale parameter handling" (p. 492), "Parameter Estimate Covariance Matrix ... Confidence
Intervals" (p. 508), "Goodness-of-Fit Statistics" (pp. 513–515).

### IBM
- Binary response is recoded to 0/1 and Prob(y = 1) is modelled. "If the reference category is
  the last value (REFERENCE=LAST in the syntax), then the first category represents a success."
  The Algorithms text does **not** state which is the default; the Command Syntax Reference
  (GENLIN /MODEL ... REFERENCE) has LAST as the default — so by default SPSS models the **first
  (lowest)** category.
- Factors: full indicator coding; columns found dependent on previous columns by the sweep
  operator are redundant and fixed at 0 — i.e. the **last level** of each factor is redundant
  ("Set to zero because this parameter is redundant").
- Scale: continuous distributions — ML, fixed value, or deviance/Pearson divided by df
  (N - p_x, p_x = non-redundant columns); discrete distributions — fixed (1) or deviance/Pearson.
  Default method is not in the Algorithms (CSR: SCALE=MLE for normal/gamma/IG; 1 for discrete).
  With deviance/Pearson scale, covariance is multiplied by phi.
- Binary 0/1 with subpopulations specified: deviance/Pearson computed on events/trials
  aggregated subpopulations (pp. 514–515).
- Wald CI: beta_j +- z_{1-alpha/2} sigma_j (normal, not t); Exp(B) CI = exp of the limits;
  scale/negbin k CI: exp(tau +- z sigma_tau), tau = log(phi).
Quote (p. 489): "If the reference category is the last value (REFERENCE=LAST in the syntax)"
Quote (p. 508): "where z_p is the 100pth percentile of the standard normal distribution."

### Our code (genlin.py)
- reference default "last", modelling the first category (genlin.py:290, 366-367) — **matches**.
- Last factor level redundant (docstring genlin.py:5-6) — **matches**.
- Scale default "ml" → 1 for discrete, MLE for continuous (genlin.py:234-243) — **matches**.
  Deviance/Pearson divide by (n - p) with p = X.shape[1] (genlin.py:239-241) — matches only if X
  contains no redundant columns (IBM uses non-redundant count p_x); check.
- Wald CI uses norm.ppf (genlin.py:671); ML scale CI = phi*exp(+-z se(log phi)) (genlin.py:713-722)
  — **matches**.
- Subpopulation-based deviance/Pearson for 0/1 binomial is not implemented (case-based
  res.deviance) — **differs** only when a subpopulation list is requested; low priority.

## 7. Bayesian statistics (BAYES ONESAMPLE / RELATED / INDEPENDENT / ANOVA / REGRESSION / CORRELATION)

Citation: Algorithms (SPSS 29 ed.), "BAYES ANOVA Algorithms" (pp. 76–81), "BAYES CORRELATION
Algorithms" (pp. 82–85), "BAYES INDEPENDENT Algorithms" (pp. 86–92), "BAYES ONESAMPLE Algorithms"
(pp. 99–112), "BAYES REGRESSION Algorithms" (pp. 113–120).

### IBM
- One-sample / paired t, JZS (Rouder et al. 2009), eq. (20) p. 101:
  B01 = (1 + t^2/nu)^{-(nu+1)/2} / int_0^inf (1+Wg)^{-1/2} (1 + t^2/((1+Wg) nu))^{-(nu+1)/2}
        (2 pi)^{-1/2} g^{-3/2} e^{-1/(2g)} dg,  nu = W - 1.
  The mixing density (2pi)^{-1/2} g^{-3/2} e^{-1/(2g)} is Inverse-Gamma(1/2, 1/2), i.e.
  delta ~ **Cauchy(0, 1) — scale r = 1, not sqrt(2)/2**; no user scale is documented for this
  method. Known-variance and fixed-g normal variants also exist (eqs. 18–19).
- Independent t, Rouder (p. 87, eq. 14): same with N = W1 W2/(W1+W2), nu = W1+W2-2 (again r = 1,
  pooled t). Gönen (eq. 15–16): delta ~ N(lambda, sigma_delta^2), BF via noncentral t pdf.
  Hyper-prior (Wang & Liu 2016, eq. 18): closed form, default a = -0.75, reported as BF10.
- Correlation (p. 83, eq. 10): JZS BF of **Wetzels & Wagenmakers (2012)**:
  BF10 = (W/2)^{1/2}/Gamma(1/2) int (1+g)^{(W-2)/2} [1+(1-r^2)g]^{-(W-1)/2} g^{-3/2} e^{-W/(2g)} dg;
  alternative: fractional BF (Kang et al. 2001, fraction b). Posterior of rho: p(rho) ∝ (1-rho^2)^c
  (c = 0 or -3/2), Fisher-z normal approximation (eq. 20) sampled I = 10^4 draws by default.
- ANOVA and Regression (pp. 76–77, 113–114), all R^2-based with p = k-1 predictors:
  Zellner (fixed g): (1+g)^{(N-p-1)/2} [1+g(1-R^2)]^{-(N-1)/2};
  Zellner–Siow / JZS: g ~ IG(1/2, N/2);  hyper-g: p(g) = (a-2)/2 (1+g)^{-a/2};
  Rouder: g ~ IG(1/2, N s^2/2) with user s. Default method/values are not stated in the
  Algorithms (they are set in syntax; see Command Syntax Reference BAYES ANOVA/REGRESSION).
- Binomial (p. 105–106): BF01 = B(a0+y, b0+N-y) B(a1,b1) / (B(a1+y, b1+N-y) B(a0,b0)) with
  Beta(a0,b0) under H0 and Beta(a1,b1) under H1; posteriors Beta(a+y, b+N-y) (uniform a=b=1,
  Jeffreys a=b=0.5, Haldane a=b=0). Poisson: Gamma(a0,b0) under H0 analog. Loglinear: a_rs = 1.
Quote (p. 101): "we assume that δ∼ Cauchy, which is a t distribution with a single degree of freedom."
Quote (p. 88): "a is input by users (a = −0.75 is the setting by default)"

### Our code (bayes.py)
- jzs_ttest_bf10 default r = 0.707 (bayes.py:84; one_sample_normal r=0.707 bayes.py:329;
  independent_normal r=0.707 bayes.py:387) — **differs**: IBM uses r = 1. Fix: default r = 1.0
  (formula otherwise matches: n g r^2 with IG(1/2,1/2)).
- gonen_bf10 (bayes.py:100-109) — **matches** eq. 15/16.
- Hyper-prior method for independent t (Wang & Liu, a = -0.75) — **not implemented**.
- pearson_bf10 = Ly et al. (2016) stretched-beta exact BF (bayes.py:112-118) — **differs**: IBM
  uses Wetzels–Wagenmakers JZS integral above (or Kang fractional BF). correlation_posterior
  (grid, stretched beta) — **differs** from IBM's Fisher-z simulation (minor numerically).
- Regression JZS: g ~ IG(1/2, r^2 n/2) with default r = 1 (bayes.py:143-165, 531) — **matches**
  IBM Zellner–Siow; Zellner and hyper-g forms match.
- ANOVA default method "rouder" = Rouder et al. 2012 multivariate-g on orthonormal contrasts
  with r = 0.5 (BayesFactor::anovaBF; bayes.py:168-194, 587) — **differs**: IBM's "Rouder's
  method" is the scalar-g R^2 form with g ~ IG(1/2, N s^2/2). The non-rouder branch
  (regression_bf10 with p = k-1, bayes.py:616) matches IBM's Zellner / Zellner–Siow / hyper-g.
  Fix: make default "jzs" (Zellner–Siow, IG(1/2, N/2)) or implement IBM-Rouder as
  regression_bf10(..., prior="jzs", r=s).
- binom_bf10 uses a point null p0 (bayes.py:121-123) — **differs** from IBM's Beta(a0,b0) null
  (equal only in the limit a0,b0 -> inf at p0).

## 8. META-ANALYSIS (META CONTINUOUS / META BINARY / META REGRESSION)

Citation: Algorithms (SPSS 29 ed. and SPSS 31 ed. — the v31 PDF, `algorithms_pdf31/algo31.pdf`,
was also downloaded; same content here), "Meta Analysis: Continuous Outcomes" (pp. 688–692),
"Meta Analysis: Binary Outcomes" (pp. 678–687), "Meta Analysis: Meta Regression" (pp. 693–705).

### IBM
- Continuous/Binary chapters document only the inverse-variance common-effect estimate, Z test
  with normal CI, Q (df N-1), H^2 = Q/(N-1), I^2 = 100(Q-(N-1))/Q, H CI via se(log H)
  (Higgins & Thompson 2002), contrast Z test.
- tau^2 estimators (meta-regression chapter, pp. 698–700): HE (Hedges), HS (Hunter–Schmidt),
  DL, SJ (non-iterative, truncated at 0); ML, REML, EB iterative: tau_{i+1} = tau_i + Delta with
  step halving; **M = 100 iterations, J = 5 step halvings, tol 1e-6, start = Hedges estimate**.
  REML Delta = (theta'PPtheta - tr P)/tr(PP).
- Which estimator is the default is **not stated** in the Algorithms (the META dialogs/CSR default
  is REML).
- Knapp–Hartung (pp. 702–703): Sigma_KH = s^2_KH (X'W X)^-, s^2_KH = theta'P theta/(N-p*);
  optional truncation s^2_KH = max(1, ...); t tests / CIs with N - p* df; s^2 = 1 when N = p*.
  Without KH, the user may request t with N - p* df.
- **Prediction interval: not documented** in either edition (no "prediction interval" text in
  the meta chapters; v31 grep confirms). Best open-source references: metafor::predict.rma —
  z by default, t_{k-p} when test="knha"/"t"; Higgins, Thompson & Spiegelhalter (2009) and
  Riley et al. (2011) recommend t_{k-2}.
Quote (p. 703): "So, we allow user to truncate the value if it is less than 1"
Quote (p. 699): "the maximum number of iterations M(default is 100)"

### Our code (meta.py)
- Estimators and KH with optional truncation (meta.py:192-310) — **match** the documented forms
  (KH: q = sum w r^2/(k-p), truncation max(1,q), df k-p). Verify the REML start value (Hedges)
  and iteration limits (100 / 5 halvings / 1e-6) — only affects convergence edge cases.
- Prediction interval mu +- c*sqrt(tau^2 + Var(mu)) with c = z, or t_{k-p} under KH
  (meta.py:167-170) — **unclear** (IBM undocumented). Matches metafor's convention; differs from
  the Higgins t_{k-2} convention. Needs an SPSS output to decide.

## 9. ONEWAY effect sizes (eta^2, epsilon^2, omega^2 fixed/random) and CIs

Citation: Algorithms **SPSS 31 edition** (`algo31.pdf`), "ONEWAY Algorithms" > "Effect Size for
One-way ANOVA" (PDF pp. 878–881). (Absent from the v29-edition PDF.)

### IBM (df_e = k'-1, df_w = W-k', df_t = W-1; BSS, WSS, TSS, WSSM = WSS/df_w, BSSM, F)
- eta^2 = BSS/TSS; eps^2 = (BSS - df_e WSSM)/TSS;
  omega^2 fixed = (BSS - df_e WSSM)/(TSS + WSSM);
  **omega^2 random = (BSSM - WSSM)/(BSSM + df_t WSSM) = (F - 1)/(F + df_t)**.
- CIs: noncentral F inversion (Steiger 2004): lambda_L solves NCF.cdf(F; df_e, df_w, lambda) = 1 - alpha/2,
  lambda_U solves = alpha/2; eta^2_{L,U} = lambda/(lambda + df_e + df_w + 1);
  eps^2 limits = eta^2 - df_e/df_w (1 - eta^2) at each eta limit;
  omega^2 fixed limits = (df_t eta^2 - df_e)/(df_w + 1 - eta^2);
  omega^2 random limits = (df_t eta^2 - df_e)/((df_w - df_t df_e) eta^2 + df_t df_e) at each eta limit.
  Negative values are output as is. Default CILEVEL 0.95.
Quote (p. 880): "The interval estimation is based on the noncentral F-distribution [Steiger, 2004]."
Quote (p. 880): "In this case, we simply output the negative estimates."

### Our code (anova.py)
- eta_interval (anova.py:123-141), eta/eps/omega-fixed points and transformed limits
  (anova.py:295-309) — **match** (algebraically identical transforms).
- Omega-squared random (anova.py:414-426): point = (F-1)/(F + n0 - 1) with n0 = average group size,
  CI from central-F limits F/F_crit — **differs**. Fix: point = (F-1)/(F + (n-1)); CI = apply
  IBM eq. 0.0.24 to the noncentral-F eta^2 limits from eta_interval.
- Note: our lambda search floors at 0 (anova.py:132-133); IBM does not say; fine.

## 10. GLM Type III / Type IV sums of squares (empty cells), estimable functions

Citation: Algorithms (SPSS 29 ed.), Appendix "Sums of Squares" > "Type II ...", "Type III Sum of
Squares and Hypothesis Matrix", "Type IV Sum of Squares and Hypothesis Matrix" (pp. 1340–1342).

### IBM
- Partition X columns into X1 (effects not containing F), X2 (F), X3 (effects containing F);
  containment: same covariates, more factors, all factors of F present; intercept contained in
  all pure-factor effects.
- Type III L: start from H = (X'WX)^- X'WX (generating estimable function, g2 inverse; formula
  image), zero out columns of non-containing effects by pivoting/row ops, Gauss-eliminate on F's
  columns, drop zero rows; if containing effects exist, split rows into G0 (F columns zero) and
  G1, and **orthogonalise G1 rows to G0**. SS = (L b)'(L G L')^- (L b); df = row rank.
- Type IV: steps 1–3 of Type III, then distribute each F-level coefficient **equitably over the
  non-missing cells of containing effects** (divide by count of occurrences); undetermined
  entries set 0 with a "may not be unique" warning.
- The "Goodnight" references in the PDF are Goodnight (1979) SWEEP tutorial (used for Type I/II
  and GLM/MIXED sweeps) and Goodnight (1978) MIVQUE0 (VARCOMP) — not an estimable-function paper.
Quote (p. 1341): "adjusted for effects that do not contain it, and orthogonal to effects (if any) that contain it."
Quote (p. 1342): "distributed equitably across the columns of effects containing F."

### Our code (glm_model.py:1-20, 387-500)
- Type III via sum-to-zero effect coding — **matches only for designs without empty cells**
  (docstring glm_model.py:14-15 says so). With empty cells SPSS's H-matrix/orthogonalisation
  gives different hypotheses — **differs**. Type IV is treated as Type III — **differs** with
  empty cells. Fix: implement the H = (X'X)^-X'X construction on the overparameterised design
  (G2 inverse via sweep in column order), the pivot/eliminate steps, Gram–Schmidt of G1 against
  G0 (Type III) and the equitable distribution (Type IV). Open-source reference: SAS GLM
  "Estimable Functions" doc (Goodnight 1978 "Tests of hypotheses in fixed-effects linear
  models", SAS TR R-101); R `car::Anova(type=3)` does NOT reproduce this with empty cells.

## 11. Post hoc tests

Citation: Algorithms (SPSS 29 ed.), Appendix "Post Hoc Tests" (pp. 1330–1338): "Range Values",
"Dunnett Tests", "Ryan-Einot-Gabriel-Welsch (R-E-G-W) Multiple Stepdown Procedures",
"Unequal Sample Sizes and Unequal Variances", "Waller-Duncan t Test".

### IBM
- REGW: gamma_r = 1 - (1-eps)^{r/k} if r < k-1; eps if r >= k-1. QREGW uses S_{gamma_r, r, f};
  FREGW: [sum n_i xbar_i^2 - (sum n_i xbar_i)^2/sum n_i] / ((r-1) s_pp^2) >= F_{gamma_r, r-1, f}.
  For homogeneous subsets n_i, n_j are used (not harmonic mean) for REGW.
- Duncan alpha_r = 1-(1-eps)^{r-1}; Tukey-b (S_{r}+S_{k})/2; Scheffe sqrt(2(k-1)F); GT2
  R = sqrt2 M_{eps,k*,f}; Gabriel |d| >= s_pp (1/sqrt(2n_i) + 1/sqrt(2n_j)) M_{eps,k*,f},
  k* = k(k-1)/2.
- Unequal variances: v = (s_i^2/n_i + s_j^2/n_j)^2 / (s_i^4/(n_i^2 v_i) + s_j^4/(n_j^2 v_j));
  |d| >= Q*_{ij} R with Q* = sqrt(s_i^2/n_i + s_j^2/n_j):
  Games–Howell R = S_{eps,k,v}/sqrt2; Tamhane T2 R = t_{gamma,v}, gamma = 1-(1-eps)^{1/k*};
  Dunnett T3 R = M_{eps,k*,v}; Dunnett C R = (S_{eps,k,n_i-1} s_i^2/n_i + S_{eps,k,n_j-1} s_j^2/n_j) / (sqrt2 (s_i^2/n_i + s_j^2/n_j)).
- Waller–Duncan: Bayesian t(w, F, q, f); harmonic mean n for unequal sizes; only subsets.
  The k-ratio default (100) is in the CSR, not the Algorithms.
Quote (p. 1336): "For the Games-Howell, Tamhane's T2, Dunnett's T3, and Dunnett's C tests, only pairwise confidence intervals are given."

### Our code (posthoc_ext.py)
- REGW alpha (posthoc_ext.py:238-242), Duncan, T2/T3/GH/C (posthoc_ext.py:193-213), GT2/Gabriel
  (posthoc_ext.py:185-224), k_ratio = 100 (posthoc_ext.py:62) — **match**.
- Check: REGW homogeneous subsets should use n_i, n_j (IBM note), whereas posthoc_ext.py:12 says
  subsets use the harmonic mean for all range tests — **possible difference** for unbalanced REGW.

## 12. Bootstrap and random number generators

Citation: Algorithms (SPSS 29 ed.), "Bootstrapping Algorithms" > "Bootstrap Estimates" / "Percentile
confidence interval" / "BCa confidence interval" / "Percentile-t Pivotal Tests" (pp. 137–141);
Appendix D "Generation of Uniform Random Numbers" (pp. 1323–1326).

### IBM
- SE = sqrt( 1/(B-1) sum (T*_b - mean T*)^2 ); bias = mean T* - T.
- Percentile CI: theta_alpha = T*_{((B+1) alpha)}, theta_{1-alpha} = T*_{((B+1)(1-alpha))};
  non-integer (B+1)alpha → **interpolation on the normal-quantile scale** between T*_(k) and
  T*_(k+1): T*_(k) + [Phi^-1(alpha) - Phi^-1(k/(B+1))] / [Phi^-1((k+1)/(B+1)) - Phi^-1(k/(B+1))] (T*_(k+1) - T*_(k));
  fails if k = 0, B, B+1 (extreme value quoted).
- BCa: alpha~ = Phi(w + (w + z_alpha)/(1 - a(w + z_alpha))), w = Phi^-1(#{T*_b < t}/(B+1)),
  a = (1/6) sum f l^3 / (sum f l^2)^{3/2}, l = jackknife influence (N_s - 1)(T - T_{-k}) within strata.
- **Sig. (two-sided) = percentile-t pivotal test**: z0 = (T - T0)/SE, z*_b = (T*_b - T)/SE*_b,
  p = (#{z*_b^2 >= z0^2} + 1)/(B + 1) (SE from the procedure's own formula).
- Number of samples (1000), CI level, CI type default (percentile) and the seed are not in the
  Algorithms (dialog/CSR defaults).
- RNG: two generators, "Version 12 Compatible" (multiplicative congruential, a = 397204094,
  p = 2^31 - 1) and Mersenne Twister (MT19937). Which is the default and the default seed
  (2000000) are **not stated** in the Algorithms (see SET RNG / SET SEED in the CSR).
Quote (p. 141): "The bootstrap two-sided p-value is calculated as"
Quote (p. 1323): "Two different random number generators are available"

### Our code (resample.py)
- SE, bias — match. Defaults samples = 1000, seed 2000000, percentile (resample.py:51, 66-69) —
  consistent with SPSS dialogs.
- Percentile CI via np.quantile(method="linear") (resample.py:248) — **differs** in interpolation:
  IBM uses order statistic (B+1)alpha with normal-scale interpolation. Fix: implement IBM rule.
- Sig. (2-tailed) = 2 min(P(T* <= 0), P(T* >= 0)) (resample.py:12-14, 339-345) — **differs**:
  IBM uses the percentile-t pivotal test (needs per-sample SE*_b). Fix where the statistic has an
  analytic SE (means, regression coefficients): p = (#{z*^2 >= z0^2}+1)/(B+1).
- BCa (resample.py:15-18, SciPy-style) — w uses #{T* < t}/B vs IBM /(B+1); a from jackknife —
  minor **difference**.
- Generator is NumPy PCG64 (resample.py:20-21, 262) — **differs** from MT19937; sample-level
  reproduction of SPSS is impossible either way unless SPSS's MT seeding/draw-to-index mapping is
  replicated (not documented).

## 13. CORRESPONDENCE — normalization and sign convention

Citation: Algorithms (SPSS 29 ed.), "CORRESPONDENCE Algorithms" > "Singular value decomposition",
"Normalization of row and column scores" (pp. 200–201), "Singular Values, Maximum Rank and
Inertia" (p. 202).

### IBM
- SVD (Golub–Reinsch) of the standardized table; normalization q: 0 = symmetrical, 1 = row
  principal, -1 = column principal, any q in [-1, 1]; alpha = (1+q)/2, beta = (1-q)/2;
  "principal": alpha = beta = 1. Scores r_is = r~_is lambda_s^alpha, c_js = c~_js lambda_s^beta,
  with sum f_i+ r_is^2 = N lambda_s^{2 alpha} (and the same for columns with beta).
  Variances are adjusted for the normalization.
- **Sign/orientation of dimensions: not documented** (SVD signs are arbitrary). Best open-source
  reference: none matches SPSS by design; R `ca::ca` and FactoMineR leave LAPACK signs.
Quote (p. 201): "The normalization implies a compensatory rescaling of the coordinate axes"

### Our code (correspondence.py:44-76)
- q to (alpha, beta) mapping and principal case — **match**; symmetrical is the default
  (correspondence.py:6, 36) — matches the SPSS dialog default.
- Sign rule "largest absolute column coordinate positive" (correspondence.py:49-54) — **our own
  convention**; SPSS's rule is undocumented. To match a given SPSS output, compare signs per
  dimension empirically; flipping a dimension does not change any other statistic.

## 14. Kaplan–Meier and Cox regression

Citation: Algorithms (SPSS 29 ed.), "KM Algorithms" (pp. 617–621): "Estimation of Percentiles and
Standard Error", "Testing the Equality of the Survival Functions"; "COXREG Algorithms"
(pp. 205–215): "Estimation", "Estimation of the Baseline Function".

### IBM
- KM: product-limit S(t), Greenwood variance; percentile t_p = inf{t_i : S(t_i) <= p};
  var(t_p) = var(S(t_p)) / f(t_p)^2, f(t_p) = [S(u_{p+0.05}) - S(t_{p-0.05})] / (t_{p-0.05} - u_{p+0.05}),
  u_q = sup{t_i : S(t_i) >= q}. (CI for the median = t_p +- 1.96 SE, as printed by SPSS.)
- Tests: log rank (Mantel–Cox), Breslow (generalized Wilcoxon, weight n_i), Tarone–Ware
  (weight sqrt(n_i)); all optionally with trend.
- Cox: covariates centered at their overall (weighted) means; Newton–Raphson on the partial
  likelihood (Breslow handling of ties). **Baseline survival = Kalbfleisch–Prentice estimator**:
  S0(t) = prod_{t_i < t} alpha_i, alpha_i = [1 - w_i exp(x_i'b)/sum_{R_i} w_l exp(x_l'b)]^{exp(-x_i'b)}
  when |D_i| = 1; solved iteratively when |D_i| > 1, starting from Breslow's
  alpha_i = exp(-d_i / sum_{R_i} w_l exp(x_l'b)).
Quote (p. 209): "Breslow (1974) suggests using the equation ... however, we will use this as an initial estimate."
Quote (p. 207): "All the covariates are centered by their corresponding overall mean."

### Our code (survival.py)
- km_percentile (survival.py:153-174): t_p, u_{p+0.05}, t_{p-0.05}, Greenwood SE / f — **match**.
  Median CI = med +- z*SE (survival.py:398-403) — matches.
- Log rank / Breslow / Tarone–Ware weights (survival.py:205) — **match**.
- Cox partial likelihood with Breslow ties (survival.py:12-16) — **match**.
- Baseline (survival.py:1317-1330): Breslow estimator h0 = d / sum exp(...) — **differs** from
  IBM's Kalbfleisch–Prentice product estimator (differs a little even without ties, more with
  ties). Fix: compute alpha_i in closed form for single events, solve
  sum_{D_i} w exp(x'b)/(1 - alpha^{exp(x'b)}) = sum_{R_i} w exp(x'b) by Newton (start = Breslow)
  for tied events; S0 = cumulative product; report at covariate means as now.

## 15. Exact tests — Monte Carlo defaults

Citation: not in the Algorithms PDF (the Appendix "Significance Levels for Fisher's Exact Test",
pp. 1320–1322, covers only exact Fisher). The Monte Carlo defaults belong to the separate IBM
SPSS Exact Tests manual and the CSR (/METHOD=MC CIN(99) SAMPLES(10000)).

- Status: **not documented in the Algorithms**. SPSS dialog and CSR defaults are 10,000 samples
  and a 99% CI; the seed comes from SET SEED (2,000,000 in a fresh session, Version-12-compatible
  generator unless SET RNG=MT).
- Our code: crosstabs.py:596-598, 697-698 and resample.py:103-106 use 10000 samples, 99% and seed
  2000000 — consistent. The MC CI is the normal approximation p +- z sqrt(p(1-p)/B)
  (crosstabs.py:590-593), which is what the Exact Tests manual describes. Individual MC p values
  cannot match SPSS bit-for-bit because the generator differs (PCG64 vs SPSS MC/MT).

---
### Summary table
| # | Topic | Status |
|---|-------|--------|
| 1 | CROSSTABS ASEs / T | match, except sym. Somers' d ASE1 (differs), GK tau ASE1 (unclear), sym. lambda ties |
| 2 | PROBIT | grid/Fieller/chi2 match; SE scaling by h and LR parallelism test differ |
| 3 | GENLOG dispersion | entropy match; concentration has an extra 1/2 and uses observed counts; df differ |
| 4 | RELIABILITY | parallel: chi2 missing Bartlett factor; strict parallel quantities and unbiased differ; Hotelling/Tukey/ICC match |
| 5 | Moses, MH | match |
| 6 | GENLIN | match (subpopulation deviance not implemented) |
| 7 | BAYES | t-test r=0.707 vs IBM r=1; correlation BF method differs; ANOVA "Rouder" differs; regression matches |
| 8 | META | KH/estimators match; prediction interval undocumented |
| 9 | ONEWAY effect sizes | match except omega^2 random (point + CI) |
| 10 | GLM Type III/IV | differ with empty cells |
| 11 | Post hoc | match (check REGW subsets n_i vs harmonic mean) |
| 12 | Bootstrap | percentile interpolation and Sig. method differ; RNG differs |
| 13 | CORRESPONDENCE | normalization matches; SPSS sign rule undocumented |
| 14 | KM / Cox | KM matches; Cox baseline (KP vs Breslow) differs |
| 15 | MC exact | not in Algorithms; defaults consistent |
