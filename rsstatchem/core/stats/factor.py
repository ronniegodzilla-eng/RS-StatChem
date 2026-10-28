"""Analyze > Dimension Reduction > Factor, mirroring SPSS FACTOR output.

Extraction: Principal Components, Principal Axis Factoring (iterated, SMC start) and Maximum
Likelihood (uniquenesses optimised as in Joreskog's method, with the Bartlett-corrected
goodness-of-fit chi-square). Number of factors from eigenvalues over a cut-off or fixed.
Rotation with Kaiser normalisation: Varimax / Quartimax / Equamax (orthomax), Direct Oblimin
(delta, via statsmodels' GPA) and Promax (kappa). Tables: Descriptive Statistics, Correlation
Matrix (+ significance, determinant), Inverse, KMO and Bartlett's Test, Anti-image matrices
with MSA, Communalities, Total Variance Explained, Scree Plot, Component/Factor Matrix,
Reproduced Correlations, Rotated / Pattern / Structure matrices, factor correlation and
transformation matrices, score coefficients; factor scores (regression, Bartlett,
Anderson-Rubin) are returned in ``.saved``.
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize, stats

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import fmt, fmt_int, fmt_p, listwise, sup

EXTRACTION = {"pc": "Principal Component Analysis", "paf": "Principal Axis Factoring",
              "ml": "Maximum Likelihood"}
ROTATION = {"none": "None", "varimax": "Varimax", "quartimax": "Quartimax",
            "equamax": "Equamax", "oblimin": "Oblimin", "promax": "Promax"}
SCORES = {"regression": "REGR", "bartlett": "BART", "anderson_rubin": "A-R"}


@dataclass
class FactorResult:
    section: Section
    saved: dict[str, tuple[str, pd.Series]] = field(default_factory=dict)


# ------------------------------------------------------------------ statistics
def kmo(R: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Overall KMO, per-variable MSA and the anti-image correlation matrix (negative partial
    correlations off the diagonal, MSA on it, as SPSS prints it)."""
    Ri = np.linalg.inv(R)
    d = np.sqrt(np.diag(Ri))
    anti = Ri / np.outer(d, d)
    off = ~np.eye(len(R), dtype=bool)
    r2 = np.where(off, R ** 2, 0)
    q2 = np.where(off, anti ** 2, 0)
    msa = r2.sum(axis=0) / (r2.sum(axis=0) + q2.sum(axis=0))
    overall = r2.sum() / (r2.sum() + q2.sum())
    np.fill_diagonal(anti, msa)
    return float(overall), msa, anti


def bartlett_sphericity(R: np.ndarray, n: int) -> tuple[float, int, float]:
    p = len(R)
    chi = -(n - 1 - (2 * p + 5) / 6) * math.log(np.linalg.det(R))
    df = p * (p - 1) // 2
    return float(chi), df, float(stats.chi2.sf(chi, df))


def _sign(L: np.ndarray) -> np.ndarray:
    s = np.sign(L.sum(axis=0))
    s[s == 0] = 1
    return L * s


def extract_pc(R: np.ndarray, m: int) -> np.ndarray:
    w, V = np.linalg.eigh(R)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    return _sign(V[:, :m] * np.sqrt(np.clip(w[:m], 0, None)))


def smc(R: np.ndarray) -> np.ndarray:
    return 1 - 1 / np.diag(np.linalg.inv(R))


def extract_paf(R: np.ndarray, m: int, max_iter: int = 25, tol: float = 0.001):
    """Iterated principal axis factoring. Returns (loadings, iterations, converged)."""
    h = smc(R)
    L = None
    for it in range(1, max_iter + 1):
        Rr = R.copy()
        np.fill_diagonal(Rr, h)
        w, V = np.linalg.eigh(Rr)
        order = np.argsort(w)[::-1]
        w, V = w[order], V[:, order]
        L = V[:, :m] * np.sqrt(np.clip(w[:m], 0, None))
        h_new = (L ** 2).sum(axis=1)
        change = float(np.max(np.abs(h_new - h)))
        h = h_new
        if change < tol:
            return _sign(L), it, True
    return _sign(L), max_iter, False


def extract_ml(R: np.ndarray, m: int, max_iter: int = 25):
    """Maximum-likelihood factor analysis. Returns (loadings, iterations, F, converged)."""
    p = len(R)

    def parts(psi):
        sc = 1 / np.sqrt(psi)
        w, V = np.linalg.eigh(R * np.outer(sc, sc))
        order = np.argsort(w)[::-1]
        return w[order], V[:, order]

    def obj(psi):
        w, _ = parts(psi)
        e = w[m:]
        return float(np.sum(e - np.log(e)) - (p - m))

    def grad(psi):
        w, V = parts(psi)
        L = np.sqrt(psi)[:, None] * V[:, :m] * np.sqrt(np.clip(w[:m] - 1, 0, None))
        return (np.sum(L ** 2, axis=1) + psi - 1) / psi ** 2

    start = np.clip(1 - smc(R), 0.005, 1)
    res = optimize.minimize(obj, start, jac=grad, method="L-BFGS-B",
                            bounds=[(0.005, 1.0)] * p,
                            options={"maxiter": max(max_iter, 200) * 10, "ftol": 1e-12,
                                     "gtol": 1e-8})
    psi = res.x
    w, V = parts(psi)
    L = np.sqrt(psi)[:, None] * V[:, :m] * np.sqrt(np.clip(w[:m] - 1, 0, None))
    return _sign(L), int(res.nit), float(res.fun), bool(res.success)


def orthomax(A: np.ndarray, gamma: float, max_iter: int = 500, tol: float = 1e-8):
    """Orthomax rotation (1 = varimax, 0 = quartimax, k/2 = equamax). Returns (L, T, iters)."""
    p, k = A.shape
    T = np.eye(k)
    d = 0.0
    it = 0
    for it in range(1, max_iter + 1):
        L = A @ T
        B = A.T @ (L ** 3 - (gamma / p) * L @ np.diag(np.sum(L ** 2, axis=0)))
        u, s, vt = np.linalg.svd(B)
        T = u @ vt
        d_new = float(s.sum())
        if d_new < d * (1 + tol):
            break
        d = d_new
    return A @ T, T, it


def pairwise_orthomax(A: np.ndarray, method: str = "varimax", max_iter: int = 25,
                      eps: float = 1e-5):
    """IBM SPSS FACTOR orthogonal rotation (Algorithms, "Orthogonal Rotations (Harman,
    1976)"): cyclic planar rotations of every factor pair j < k by P = atan2(Y, X)/4 with
    Y = D - w·2AB/n, X = C - w(A^2 - B^2)/n (w = 1 varimax, m/2·... equamax, 0 quartimax),
    stopping when |SV(i) - SV(i-1)| <= 1e-5 or after ``max_iter`` iterations (SPSS default
    25). ``A`` is the (Kaiser-normalised) pattern. Returns (L, T, iterations, converged)."""
    L = np.array(A, float, copy=True)
    n, m = L.shape
    T = np.eye(m)

    def sv(M):
        s2 = (M ** 2).sum(axis=0)
        s4 = (M ** 4).sum(axis=0)
        if method == "varimax":
            return float(np.sum(n * s4 - s2 ** 2))
        if method == "equamax":
            return float(np.sum(n * s4 - m / 2 * s2 ** 2))
        return float(np.sum(s4))

    prev = None
    it = 0
    for it in range(1, max_iter + 1):
        cur = sv(L)
        if prev is not None and abs(cur - prev) <= eps:
            return L, T, it, True
        prev = cur
        for j in range(m - 1):
            for k in range(j + 1, m):
                x, y = L[:, j], L[:, k]
                u = x ** 2 - y ** 2
                v = 2 * x * y
                a, b = u.sum(), v.sum()
                c = float(np.sum(u ** 2 - v ** 2))
                d = float(np.sum(2 * u * v))
                if method == "varimax":
                    num, den = d - 2 * a * b / n, c - (a * a - b * b) / n
                elif method == "equamax":
                    num, den = d - m * a * b / n, c - m * (a * a - b * b) / n
                else:
                    num, den = d, c
                p = math.atan2(num, den) / 4
                if abs(math.sin(p)) <= 1e-15:
                    continue
                cs, sn = math.cos(p), math.sin(p)
                L[:, j], L[:, k] = x * cs + y * sn, -x * sn + y * cs
                tj, tk = T[:, j].copy(), T[:, k].copy()
                T[:, j], T[:, k] = tj * cs + tk * sn, -tj * sn + tk * cs
    cur = sv(L)
    return L, T, it, prev is not None and abs(cur - prev) <= eps


def _kaiser(A):
    h = np.sqrt((A ** 2).sum(axis=1))
    h[h == 0] = 1
    return A / h[:, None], h


def rotate(A: np.ndarray, method: str, delta: float = 0.0, kappa: float = 4.0,
           max_iter: int = 25):
    """Kaiser-normalised rotation. Returns (pattern, phi or None, T, iterations)."""
    k = A.shape[1]
    An, h = _kaiser(A)
    if method in ("varimax", "quartimax", "equamax"):
        # IBM's cyclic pairwise algorithm (SPSS default 25 iterations, |dSV| <= 1e-5); the
        # SVD orthomax iteration stopped short of the optimum (UCLA hsb2 PCA/varimax: SPSS
        # rotation sums 42.267 % / 36.497 %, SVD 42.265 / 36.499, pairwise 42.267 / 36.497).
        Ln, T, it, _conv = pairwise_orthomax(An, method, max_iter)
        return Ln * h[:, None], None, T, it
    if method == "oblimin":
        from statsmodels.multivariate.factor_rotation import rotate_factors

        Ln, T = rotate_factors(An, "oblimin", delta, "oblique", max_tries=max(max_iter, 1000))
        phi = T.T @ T
        return Ln * h[:, None], phi, T, None
    if method == "promax":
        Ln, T0, it, _conv = pairwise_orthomax(An, "varimax", max_iter)
        L = Ln * h[:, None]
        Lk, hk = _kaiser(L)
        P = Lk * np.abs(Lk) ** (kappa - 1)
        U = np.linalg.lstsq(Lk, P, rcond=None)[0]
        d = np.diag(np.linalg.inv(U.T @ U))
        U = U @ np.diag(np.sqrt(d))
        pattern = (Lk @ U) * hk[:, None]
        Ti = np.linalg.inv(U)
        phi = Ti @ Ti.T
        return pattern, phi, T0 @ U, it
    raise ValueError(method)


# ------------------------------------------------------------------ tables
def _mat_rows(names: list[str], M: np.ndarray, decimals: int = 3, blank_below=None,
              diag_mark: str = "") -> list[list[str]]:
    rows = []
    for i, nm in enumerate(names):
        cells = []
        for j in range(M.shape[1]):
            v = M[i, j]
            if blank_below is not None and abs(v) < blank_below:
                cells.append("")
            else:
                cells.append(fmt(v, decimals) + (diag_mark if diag_mark and i == j else ""))
        rows.append([nm] + cells)
    return rows


def _order(L: np.ndarray, sort: bool) -> list[int]:
    if not sort:
        return list(range(L.shape[0]))
    a = np.abs(L)
    best = a.argmax(axis=1)
    return sorted(range(L.shape[0]), key=lambda i: (best[i], -a[i, best[i]]))


def _png(fig) -> bytes:
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    return buf.getvalue()


def scree_plot(eigen: np.ndarray, label: str) -> bytes:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(6.0, 4.0), dpi=120)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    x = np.arange(1, len(eigen) + 1)
    ax.plot(x, eigen, marker="o", color="#C62828", linewidth=1.6)
    ax.set_xticks(x)
    ax.set_xlabel(f"{label} Number")
    ax.set_ylabel("Eigenvalue")
    ax.set_title("Scree Plot", fontsize=11, fontweight="bold")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return _png(fig)


def loading_plot(L: np.ndarray, names: list[str], label: str, rotated: bool) -> bytes:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(5.2, 5.0), dpi=120)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    ax.axhline(0, color="#999", linewidth=0.8)
    ax.axvline(0, color="#999", linewidth=0.8)
    ax.scatter(L[:, 0], L[:, 1], color="#C62828", s=18)
    for (x, y), nm in zip(L[:, :2], names):
        ax.annotate(nm, (x, y), fontsize=8, xytext=(3, 3), textcoords="offset points")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel(f"{label} 1")
    ax.set_ylabel(f"{label} 2")
    ax.set_title(f"{label} Plot" + (" in Rotated Space" if rotated else ""), fontsize=11,
                 fontweight="bold")
    return _png(fig)


# ------------------------------------------------------------------ main
def factor_analysis(ds: Dataset, variables: list[str], extraction: str = "pc",
                    criterion: str = "eigen", min_eigen: float = 1.0, n_factors: int = 2,
                    rotation: str = "none", delta: float = 0.0, kappa: float = 4.0,
                    max_iter: int = 25, rot_max_iter: int = 25, univariate: bool = False,
                    initial: bool = True, coefficients: bool = False,
                    significance: bool = False, determinant: bool = False,
                    kmo_bartlett: bool = True, inverse: bool = False, reproduced: bool = False,
                    anti_image: bool = False, unrotated: bool = True, scree: bool = True,
                    loading_plots: bool = False, sort: bool = False,
                    suppress: float | None = None, save: str | None = None,
                    score_coefficients: bool = False, level_note: bool = True) -> FactorResult:
    section = Section("Factor Analysis")
    names = [ds.var(v).name for v in variables]
    label = "Component" if extraction == "pc" else "Factor"
    ext_name = EXTRACTION[extraction]
    log = f"FACTOR\n  /VARIABLES {' '.join(names)}\n  /MISSING LISTWISE\n  /PRINT"
    pr = [k for k, on in (("UNIVARIATE", univariate), ("INITIAL", initial),
                          ("CORRELATION", coefficients), ("SIG", significance),
                          ("DET", determinant), ("KMO", kmo_bartlett), ("INV", inverse),
                          ("REPR", reproduced), ("AIC", anti_image),
                          ("EXTRACTION", unrotated), ("ROTATION", rotation != "none"),
                          ("FSCORE", score_coefficients)) if on]
    log += " " + " ".join(pr)
    if sort or suppress:
        log += "\n  /FORMAT" + (" SORT" if sort else "") + \
            (f" BLANK({suppress:g})" if suppress else "")
    if scree:
        log += "\n  /PLOT EIGEN" + (" ROTATION" if loading_plots else "")
    crit = f"MINEIGEN({min_eigen:g})" if criterion == "eigen" else f"FACTORS({n_factors})"
    log += (f"\n  /CRITERIA {crit} ITERATE({max_iter})\n  /EXTRACTION "
            f"{ {'pc': 'PC', 'paf': 'PAF', 'ml': 'ML'}[extraction]}")
    rot_kw = {"oblimin": "OBLIMIN", "promax": f"PROMAX({kappa:g})"}.get(rotation,
                                                                       rotation.upper())
    log += f"\n  /CRITERIA ITERATE({rot_max_iter})" + \
        (f" DELTA({delta:g})" if rotation == "oblimin" else "") + \
        f"\n  /ROTATION {'NOROTATE' if rotation == 'none' else rot_kw}"
    if save:
        log += f"\n  /SAVE {SCORES[save].replace('-', '')}(ALL)"
    section.log(log + "\n  /METHOD=CORRELATION.")

    data = listwise(ds, variables)
    n, p = data.shape
    result = FactorResult(section)
    if n < 3 or p < 2:
        section.note("Warning", "There are too few valid cases or variables for this analysis.",
                     style="warning")
        return result
    X = data.to_numpy(float)
    sd = X.std(axis=0, ddof=1)
    if np.any(sd == 0):
        section.note("Warning", "At least one variable has zero variance; the analysis is "
                     "not performed.", style="warning")
        return result
    R = np.corrcoef(X, rowvar=False)
    det = float(np.linalg.det(R))
    singular = det <= 1e-12
    ext_foot = f"Extraction Method: {ext_name}."
    rot_name = None if rotation == "none" else \
        f"{ROTATION[rotation]} with Kaiser Normalization."
    if univariate:
        section.table(Table("Descriptive Statistics", ["", "Mean", "Std. Deviation",
                                                       "Analysis N"],
                            [[ds.var(v).title, fmt(X[:, j].mean(), 4), fmt(sd[j], 5),
                              fmt_int(n)] for j, v in enumerate(variables)]))
    if coefficients or significance:
        rows = []
        if coefficients:
            for i, r in enumerate(_mat_rows(names, R)):
                rows.append(["Correlation" if i == 0 else ""] + r)
        if significance:
            for i in range(p):
                cells = []
                for j in range(p):
                    if i == j:
                        cells.append("")
                        continue
                    r = R[i, j]
                    t = r * math.sqrt((n - 2) / max(1 - r * r, 1e-300))
                    cells.append(fmt_p(stats.t.sf(abs(t), n - 2)))
                rows.append(["Sig. (1-tailed)" if i == 0 else "", names[i]] + cells)
        section.table(Table("Correlation Matrix" + (sup("a") if determinant else ""),
                            ["", ""] + names, rows, row_header_cols=2,
                            footnotes=[f"a. Determinant = {fmt(det)}"] if determinant else []))
    elif determinant:
        section.note("", f"Determinant of the correlation matrix = {fmt(det)}")
    if singular:
        section.note("Warning", "This matrix is not positive definite (determinant ≈ 0). "
                     "KMO, anti-image matrices, PAF and ML cannot be computed.", style="warning")
        if extraction != "pc":
            return result
    if inverse and not singular:
        section.table(Table("Inverse of Correlation Matrix", [""] + names,
                            _mat_rows(names, np.linalg.inv(R))))
    if kmo_bartlett and not singular:
        overall, msa, aic_corr = kmo(R)
        chi, df, pv = bartlett_sphericity(R, n)
        section.table(Table("KMO and Bartlett's Test", ["", "", ""],
                            [["Kaiser-Meyer-Olkin Measure of Sampling Adequacy.", "",
                              fmt(overall)],
                             ["Bartlett's Test of Sphericity", "Approx. Chi-Square", fmt(chi)],
                             ["", "df", fmt_int(df)], ["", "Sig.", fmt_p(pv)]],
                            row_header_cols=2))
    if anti_image and not singular:
        overall, msa, aic_corr = kmo(R)
        Ri = np.linalg.inv(R)
        aic_cov = Ri / np.outer(np.diag(Ri), np.diag(Ri))
        rows = []
        for i, r in enumerate(_mat_rows(names, aic_cov)):
            rows.append(["Anti-image Covariance" if i == 0 else ""] + r)
        for i, r in enumerate(_mat_rows(names, aic_corr, diag_mark=sup("a"))):
            rows.append(["Anti-image Correlation" if i == 0 else ""] + r)
        section.table(Table("Anti-image Matrices", ["", ""] + names, rows, row_header_cols=2,
                            footnotes=["a. Measures of Sampling Adequacy(MSA)"]))

    # extraction
    eigen = np.sort(np.linalg.eigvalsh(R))[::-1]
    m = int(np.sum(eigen > min_eigen)) if criterion == "eigen" else int(n_factors)
    m = max(1, min(m, p if extraction == "pc" else p - 1))
    iters, converged, ml_F = None, True, None
    if extraction == "pc":
        L = extract_pc(R, m)
        init_comm = np.ones(p)
    elif extraction == "paf":
        L, iters, converged = extract_paf(R, m, max_iter)
        init_comm = smc(R)
    else:
        if ((p - m) ** 2 - (p + m)) < 0:
            section.note("Warning", f"Too many factors ({m}) for maximum likelihood with "
                         f"{p} variables.", style="warning")
        L, iters, ml_F, converged = extract_ml(R, m, max_iter)
        init_comm = smc(R)
    comm = (L ** 2).sum(axis=1)
    ext_note = f"a. {m} {label.lower()}{'s' if m != 1 else ''} extracted."
    if iters is not None:
        ext_note = (f"a. {m} factor{'s' if m != 1 else ''} extracted. {iters} iterations "
                    "required." if converged else
                    f"a. Attempted to extract {m} factors. More than {max_iter} iterations "
                    "required. Extraction was terminated.")

    rows = [[ds.var(v).title] + ([fmt(init_comm[j])] if initial else []) + [fmt(comm[j])]
            for j, v in enumerate(variables)]
    section.table(Table("Communalities", [""] + (["Initial"] if initial else []) +
                        ["Extraction"], rows, footnotes=[ext_foot]))

    # rotation
    pattern = phi = T = None
    rot_iters = None
    if rotation != "none" and m > 1:
        pattern, phi, T, rot_iters = rotate(L, rotation, delta, kappa, rot_max_iter)
        if phi is None:
            pattern = _sign(pattern)
        else:
            s = np.sign(pattern.sum(axis=0))
            s[s == 0] = 1
            pattern = pattern * s
            phi = phi * np.outer(s, s)
    elif rotation != "none":
        section.note("", f"Only one {label.lower()} was extracted. The solution cannot be "
                     "rotated.")

    # Total Variance Explained
    ss_ext = (L ** 2).sum(axis=0)
    rows = []
    if pattern is not None:
        rot_ss = (pattern ** 2).sum(axis=0) if phi is None else ((pattern @ phi) ** 2).sum(axis=0)
    cum = 0.0
    cum_e = 0.0
    cum_r = 0.0
    for i in range(p):
        cum += eigen[i] / p * 100
        cells = [str(i + 1), fmt(eigen[i]), fmt(eigen[i] / p * 100), fmt(cum)]
        if i < m:
            cum_e += ss_ext[i] / p * 100
            cells += [fmt(ss_ext[i]), fmt(ss_ext[i] / p * 100), fmt(cum_e)]
        else:
            cells += ["", "", ""]
        if pattern is not None:
            if i < m and phi is None:
                cum_r += rot_ss[i] / p * 100
                cells += [fmt(rot_ss[i]), fmt(rot_ss[i] / p * 100), fmt(cum_r)]
            elif i < m:
                cells += [fmt(rot_ss[i])]
            else:
                cells += ["", "", ""] if phi is None else [""]
        rows.append(cells)
    cols = [label, ("Initial Eigenvalues", "Total"), ("Initial Eigenvalues", "% of Variance"),
            ("Initial Eigenvalues", "Cumulative %"),
            ("Extraction Sums of Squared Loadings", "Total"),
            ("Extraction Sums of Squared Loadings", "% of Variance"),
            ("Extraction Sums of Squared Loadings", "Cumulative %")]
    foot = [ext_foot]
    if pattern is not None:
        if phi is None:
            cols += [("Rotation Sums of Squared Loadings", "Total"),
                     ("Rotation Sums of Squared Loadings", "% of Variance"),
                     ("Rotation Sums of Squared Loadings", "Cumulative %")]
        else:
            cols += [("Rotation Sums of Squared Loadingsᵃ", "Total")]
            foot.append(f"a. When {label.lower()}s are correlated, sums of squared loadings "
                        "cannot be added to obtain a total variance.")
    section.table(Table("Total Variance Explained", cols, rows, footnotes=foot))
    if extraction == "ml" and ml_F is not None:
        dfm = ((p - m) ** 2 - (p + m)) / 2
        chi = (n - 1 - (2 * p + 5) / 6 - 2 * m / 3) * ml_F
        section.table(Table("Goodness-of-fit Test", ["Chi-Square", "df", "Sig."],
                            [[fmt(chi), fmt_int(dfm), fmt_p(stats.chi2.sf(chi, dfm))
                              if dfm > 0 else ""]], row_header_cols=0))
    if scree:
        section.image("Scree Plot", scree_plot(eigen, label), width=520)

    titles = [ds.var(v).title for v in variables]
    heads = [str(i + 1) for i in range(m)]
    if unrotated:
        order = _order(L, sort)
        section.table(Table(f"{label} Matrix" + sup("a"), [""] + [(label, h) for h in heads],
                            _mat_rows([titles[i] for i in order], L[order], 3, suppress),
                            footnotes=[ext_foot, ext_note]))
    if reproduced:
        Rh = L @ L.T
        resid = R - Rh
        big = int(np.sum(np.abs(resid[np.triu_indices(p, 1)]) > 0.05))
        tot = p * (p - 1) // 2
        rows = []
        for i, r in enumerate(_mat_rows(names, Rh, diag_mark=sup("b"))):
            rows.append(["Reproduced Correlation" if i == 0 else ""] + r)
        resid_display = resid.copy()
        np.fill_diagonal(resid_display, np.nan)
        for i, r in enumerate(_mat_rows(names, resid_display)):
            rows.append(["Residual" + sup("a") if i == 0 else ""] + r)
        section.table(Table("Reproduced Correlations", ["", ""] + names, rows,
                            row_header_cols=2,
                            footnotes=[ext_foot,
                                       f"a. Residuals are computed between observed and "
                                       f"reproduced correlations. There are {big} "
                                       f"({big / tot * 100:.0f}%) nonredundant residuals with "
                                       f"absolute values greater than 0.05.",
                                       "b. Reproduced communalities"]))
    rot_foot = []
    if pattern is not None:
        rot_foot = [ext_foot, f"Rotation Method: {rot_name}"]
        conv = f"a. Rotation converged in {rot_iters} iterations." if rot_iters else \
            "a. Rotation converged."
        order = _order(pattern, sort)
        if phi is None:
            section.table(Table(f"Rotated {label} Matrix" + sup("a"),
                                [""] + [(label, h) for h in heads],
                                _mat_rows([titles[i] for i in order], pattern[order], 3,
                                          suppress), footnotes=rot_foot + [conv]))
            section.table(Table(f"{label} Transformation Matrix", [label] + heads,
                                _mat_rows(heads, T), footnotes=rot_foot))
        else:
            structure = pattern @ phi
            section.table(Table("Pattern Matrix" + sup("a"), [""] + [(label, h) for h in heads],
                                _mat_rows([titles[i] for i in order], pattern[order], 3,
                                          suppress), footnotes=rot_foot + [conv]))
            order_s = _order(structure, sort)
            section.table(Table("Structure Matrix", [""] + [(label, h) for h in heads],
                                _mat_rows([titles[i] for i in order_s], structure[order_s], 3,
                                          suppress), footnotes=rot_foot))
            section.table(Table(f"{label} Correlation Matrix", [label] + heads,
                                _mat_rows(heads, phi), footnotes=rot_foot))
    if loading_plots and m >= 2:
        P = pattern if pattern is not None else L
        section.image(f"{label} Plot", loading_plot(P, names, label, pattern is not None),
                      width=460)

    # factor scores
    if save or score_coefficients:
        P = pattern if pattern is not None else L
        Phi = phi if phi is not None else np.eye(m)
        S = P @ Phi
        Ri = np.linalg.pinv(R)
        method = save or "regression"
        psi = np.clip(1 - comm, 1e-6, None)
        if extraction == "pc" or method == "regression":
            W = Ri @ S
        elif method == "bartlett":
            Pi = P / psi[:, None]
            W = Pi @ np.linalg.inv(P.T @ Pi)
        else:
            Pi = P / psi[:, None]
            Mx = Pi.T @ R @ Pi
            w, V = np.linalg.eigh(Mx)
            W = Pi @ (V @ np.diag(1 / np.sqrt(np.clip(w, 1e-12, None))) @ V.T)
        if score_coefficients:
            section.table(Table(f"{label} Score Coefficient Matrix",
                                [""] + [(label, h) for h in heads],
                                _mat_rows(titles, W),
                                footnotes=[ext_foot] + ([f"Rotation Method: {rot_name}"]
                                                        if rot_name and pattern is not None
                                                        else []) + [f"{label} Scores."]))
        if save:
            Z = (X - X.mean(axis=0)) / sd
            scores = Z @ W
            tag = SCORES[save]
            for j in range(m):
                s = pd.Series(np.nan, index=ds.df.index, dtype=float)
                s.loc[data.index] = scores[:, j]
                result.saved[f"FAC{j + 1}"] = (f"{tag} factor score {j + 1:>3} for analysis 1",
                                               s)
    return result
