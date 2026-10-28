"""Colocalisation analysis in the style of Fiji's Coloc 2, written from the publications
(no Fiji code; Coloc 2 is GPL):

* Pearson's correlation coefficient (all pixels, below and above the thresholds);
* Manders' overlap and split coefficients M1, M2 (Manders et al., J Microsc 1993) without
  threshold (signal = value > 0) and tM1, tM2 with the automatic thresholds;
* Costes' automatic threshold (Costes et al., Biophys J 2004): thresholds on the orthogonal
  regression line of channel 2 on channel 1, lowered (bisection) until Pearson's r of the
  pixels below either threshold drops to zero or below;
* Costes' significance test: Pearson's r of channel 1 scrambled in PSF-sized blocks versus
  channel 2, P = fraction of randomisations with r below the observed r;
* Li's intensity correlation quotient ICQ (Li et al., J Neurosci 2004);
* Spearman's rank and Kendall's tau-b correlations; the 2-D intensity histogram.

Optional mask (ROI) restricts all statistics.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return float("nan")
    am, bm = a.mean(), b.mean()
    da, db = a - am, b - bm
    den = math.sqrt(float((da * da).sum()) * float((db * db).sum()))
    return float((da * db).sum()) / den if den else float("nan")


def orthogonal_regression(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Slope and intercept of the orthogonal (total least squares) fit b = m a + c."""
    am, bm = a.mean(), b.mean()
    sxx = float(((a - am) ** 2).mean())
    syy = float(((b - bm) ** 2).mean())
    sxy = float(((a - am) * (b - bm)).mean())
    if sxy == 0:
        return 0.0, bm
    m = (syy - sxx + math.sqrt((syy - sxx) ** 2 + 4 * sxy * sxy)) / (2 * sxy)
    return m, bm - m * am


def costes_threshold(a: np.ndarray, b: np.ndarray, max_iter: int = 100):
    """(T1, T2, slope, intercept, r_below) of Costes' auto threshold (bisection on T1)."""
    m, c = orthogonal_regression(a, b)
    lo, hi = float(a.min()), float(a.max())
    integer = np.issubdtype(a.dtype, np.integer) or (np.all(a == np.round(a)) and
                                                      np.all(b == np.round(b)))
    best = None
    t_hi, t_lo = hi, lo
    for _ in range(max_iter):
        t1 = (t_hi + t_lo) / 2.0
        if integer:
            t1 = math.floor(t1)
        t2 = m * t1 + c
        below = (a < t1) | (b < t2)
        r = _pearson(a[below].astype(float), b[below].astype(float))
        if np.isnan(r) or r <= 0:
            best = (t1, t2, r)
            t_lo = t1
        else:
            t_hi = t1
        if (integer and t_hi - t_lo <= 1) or (not integer and t_hi - t_lo < 1e-6 * (hi - lo
                                                                                  or 1)):
            break
    if best is None:
        t1 = t_lo
        t2 = m * t1 + c
        below = (a < t1) | (b < t2)
        best = (t1, t2, _pearson(a[below].astype(float), b[below].astype(float)))
    return best[0], best[1], m, c, best[2]


def li_icq(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0:
        return float("nan")
    prod = (a - a.mean()) * (b - b.mean())
    pos = int((prod > 0).sum())
    neg = int((prod < 0).sum())
    return pos / float(pos + neg) - 0.5 if pos + neg else 0.0


def manders(a, b, t1=0.0, t2=0.0, thresholded=False):
    """(M1, M2): fraction of channel-1 intensity where channel 2 is above its threshold and
    vice versa (> 0 without thresholds)."""
    a = a.astype(float)
    b = b.astype(float)
    if thresholded:
        sa = a[a > t1].sum()
        sb = b[b > t2].sum()
        m1 = a[(a > t1) & (b > t2)].sum() / sa if sa else float("nan")
        m2 = b[(a > t1) & (b > t2)].sum() / sb if sb else float("nan")
    else:
        sa, sb = a.sum(), b.sum()
        m1 = a[b > 0].sum() / sa if sa else float("nan")
        m2 = b[a > 0].sum() / sb if sb else float("nan")
    return float(m1), float(m2)


def costes_randomization(c1: np.ndarray, c2: np.ndarray, mask: np.ndarray | None,
                         psf: int = 3, n: int = 10, seed: int | None = None):
    """(P-value, mean r, SD r, observed r) of the block-scrambling test (2-D planes)."""
    rng = np.random.default_rng(seed)
    h, w = c1.shape[-2:]
    psf = max(int(psf), 1)
    by, bx = h // psf, w // psf
    if by == 0 or bx == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    a = c1[..., :by * psf, :bx * psf].astype(float)
    b = c2[..., :by * psf, :bx * psf].astype(float)
    m = None if mask is None else mask[..., :by * psf, :bx * psf]
    sel = np.ones(b.shape, bool) if m is None else m
    observed = _pearson(a[sel], b[sel])
    blocks = a.reshape(*a.shape[:-2], by, psf, bx, psf)
    rs = []
    for _ in range(n):
        order = rng.permutation(by * bx)
        bb = blocks.swapaxes(-3, -2).reshape(*a.shape[:-2], by * bx, psf, psf)[..., order, :, :]
        shuffled = bb.reshape(*a.shape[:-2], by, bx, psf, psf).swapaxes(-3, -2) \
            .reshape(a.shape)
        rs.append(_pearson(shuffled[sel], b[sel]))
    rs = np.asarray(rs)
    p = float((rs < observed).sum()) / len(rs) if len(rs) else float("nan")
    return p, float(rs.mean()), float(rs.std(ddof=1)) if len(rs) > 1 else 0.0, observed


@dataclass
class ColocResult:
    values: dict = field(default_factory=dict)
    histogram2d: np.ndarray | None = None
    hist_range: tuple = (0, 0, 0, 0)

    def table(self, title: str = "Coloc 2"):
        from rsstatchem.core.imaging.results import ResultsTable
        rt = ResultsTable(title)
        for k, v in self.values.items():
            rt.add_row({"Value": v} if not isinstance(v, str) else {"Value": v}, label=k)
        return rt

    def text(self) -> str:
        lines = []
        for k, v in self.values.items():
            lines.append(f"{k}, {v:.4f}" if isinstance(v, float) else f"{k}, {v}")
        return "\n".join(lines)


def coloc(ch1: np.ndarray, ch2: np.ndarray, mask: np.ndarray | None = None, psf: int = 3,
          randomizations: int = 10, costes: bool = True, li: bool = True,
          spearman: bool = True, kendall: bool = False, manders_: bool = True,
          seed: int | None = None, bins: int = 256) -> ColocResult:
    """Coloc 2 statistics of two equally sized channels (2-D or 3-D)."""
    if ch1.shape != ch2.shape:
        raise ValueError("Both images must have the same size.")
    sel = np.ones(ch1.shape, bool) if mask is None else np.broadcast_to(mask, ch1.shape)
    with np.errstate(invalid="ignore"):
        finite = np.isfinite(ch1.astype(float)) & np.isfinite(ch2.astype(float))
    sel = sel & finite
    a = ch1[sel]
    b = ch2[sel]
    af, bf = a.astype(float), b.astype(float)
    res = ColocResult()
    v = res.values
    v["Pearson's R value (no threshold)"] = _pearson(af, bf)
    if costes:
        t1, t2, m, c, r_below = costes_threshold(a, b)
        v["Auto threshold slope"] = m
        v["Auto threshold intercept"] = c
        v["Ch1 Max Threshold"] = float(t1)
        v["Ch2 Max Threshold"] = float(t2)
        above = (af > t1) & (bf > t2)
        below = ~((af >= t1) & (bf >= t2))
        v["Pearson's R value (below threshold)"] = _pearson(af[below], bf[below])
        v["Pearson's R value (above threshold)"] = _pearson(af[above], bf[above])
    if li:
        v["Li's ICQ value"] = li_icq(af, bf)
    if spearman:
        from scipy.stats import spearmanr
        v["Spearman's rank correlation value"] = float(spearmanr(af, bf).statistic) \
            if af.size > 1 else float("nan")
        v["Spearman's rank correlation n"] = int(af.size)
    if kendall:
        from scipy.stats import kendalltau
        v["Kendall's Tau-b rank correlation value"] = float(kendalltau(af, bf).statistic) \
            if af.size > 1 else float("nan")
    if manders_:
        m1, m2 = manders(af, bf)
        v["Manders' M1 (no threshold)"] = m1
        v["Manders' M2 (no threshold)"] = m2
        if costes:
            tm1, tm2 = manders(af, bf, t1, t2, thresholded=True)
            v["Manders' tM1 (Above autothreshold)"] = tm1
            v["Manders' tM2 (Above autothreshold)"] = tm2
    if randomizations > 0 and ch1.ndim >= 2:
        p, mean_r, sd_r, obs = costes_randomization(np.where(sel, ch1, 0), np.where(sel, ch2, 0),
                                                    None if mask is None else sel, psf,
                                                    randomizations, seed)
        v["Costes P-Value"] = p
        v["Costes Shuffled Mean"] = mean_r
        v["Costes Shuffled Std.Dev."] = sd_r
        v["Costes number of shuffles"] = int(randomizations)
        v["PSF"] = int(psf)
    a_min, a_max = float(af.min()) if af.size else 0, float(af.max()) if af.size else 1
    b_min, b_max = float(bf.min()) if bf.size else 0, float(bf.max()) if bf.size else 1
    h2, _, _ = np.histogram2d(bf, af, bins=bins, range=[[b_min, b_max or 1], [a_min, a_max or 1]])
    res.histogram2d = h2[::-1]          # channel 2 upwards
    res.hist_range = (a_min, a_max, b_min, b_max)
    return res


def histogram_image(res: ColocResult) -> np.ndarray:
    """Log-scaled 2-D histogram as an 8-bit image (Coloc 2's scatter plot)."""
    h = res.histogram2d
    if h is None:
        return np.zeros((256, 256), np.uint8)
    lg = np.log1p(h)
    mx = lg.max() or 1.0
    return np.clip(lg / mx * 255.0 + 0.5, 0, 255).astype(np.uint8)
