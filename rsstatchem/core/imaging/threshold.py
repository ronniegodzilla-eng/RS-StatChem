"""Thresholding: auto-threshold methods, ImageJ's ``setAutoThreshold`` semantics, Make Binary /
Convert to Mask, and the colour thresholder's colour spaces and band filters.

Ported from ImageJ 1.54 ij/process/AutoThresholder.java, ij/process/ImageProcessor.java
(setAutoThreshold, scaleAndSetThreshold, getAutoThreshold), ij/IJ.java (setStackThreshold),
ij/process/StackStatistics.java, ij/plugin/Thresholder.java,
ij/plugin/frame/ColorThresholder.java, ij/process/ColorSpaceConverter.java and the Gaussian
kernel of ij/plugin/filter/GaussianBlur.java (public domain, NIH; W. Rasband, G. Landini,
M. Schmid, J. Schindelin et al.).

The methods **Intermodes, MinError, Minimum and Percentile** are *not* ported: in ImageJ they
derive from GPL-licensed Matlab code. They are written here from the original papers
(Prewitt & Mendelsohn 1966; Kittler & Illingworth 1986; Doyle 1962) and verified black-box
against ImageJ 1.54p's outputs (tests/data_imagej/golden/thresholds).

Java arithmetic is reproduced where it matters: ``int`` sums wrap at 32 bits, ``(int)``
truncates toward zero, ``Math.round`` is floor(x + 0.5), float32 where Java uses ``float``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np

# AutoThresholder.Method (ImageJ order; getList("threshold.methods"))
METHODS = ("Default", "Huang", "Intermodes", "IsoData", "IJ_IsoData", "Li", "MaxEntropy",
           "Mean", "MinError", "Minimum", "Moments", "Otsu", "Percentile", "RenyiEntropy",
           "Shanbhag", "Triangle", "Yen")
# Fiji's Auto Threshold plugin offers Huang2 and no IJ_IsoData
FIJI_METHODS = ("Default", "Huang", "Huang2", "Intermodes", "IsoData", "Li", "MaxEntropy",
                "Mean", "MinError", "Minimum", "Moments", "Otsu", "Percentile", "RenyiEntropy",
                "Shanbhag", "Triangle", "Yen")
DISPLAY_MODES = ("Red", "B&W", "Over/Under")          # Threshold window "display" choice
MODE_KEYS = {"Red": "red", "B&W": "bw", "Over/Under": "over_under"}
BACKGROUNDS = ("Default", "Dark", "Light")            # Thresholder (Convert Stack to Binary)
COLOR_SPACES = ("HSB", "RGB", "Lab", "YUV")
COLOR_MODES = ("Red", "White", "Black", "B&W")        # ColorThresholder "Threshold color"
CHANNEL_NAMES = {"HSB": ("Hue", "Saturation", "Brightness"), "RGB": ("Red", "Green", "Blue"),
                 "Lab": ("L*", "a*", "b*"), "YUV": ("Y", "U", "V")}
NO_THRESHOLD = -808080.0                              # ImageProcessor.NO_THRESHOLD
_DBL_EPS = 2.220446049250313E-16


def _i32(v: int) -> int:
    """Java int overflow (two's complement wrap)."""
    v &= 0xFFFFFFFF
    return v - 0x100000000 if v >= 0x80000000 else v


def _jint(v: float) -> int:
    """Java ``(int)`` cast of a double: truncation, NaN -> 0, saturating."""
    if v != v:
        return 0
    if v >= 2147483647:
        return 2147483647
    if v <= -2147483648:
        return -2147483648
    return int(v)


def _jround(v: float) -> int:
    """Java ``Math.round(double)`` (floor(v + 0.5)) as int."""
    if v != v:
        return 0
    return int(math.floor(v + 0.5))


# ===================================================================================
# AutoThresholder
# ===================================================================================
def bilevel(data: Sequence[int], subtract_one: bool = True) -> int:
    """AutoThresholder.bilevel: images with 1 or 2 grey levels."""
    first = second = -1
    n = 0
    for i, count in enumerate(data):
        if count > 0:
            n += 1
            if n > 2:
                return -1
            if first == -1:
                first = i
            else:
                second = i
    if n == 1:
        return first - (1 if subtract_one else 0)
    if n == 2:
        return second - (1 if subtract_one else 0)
    return -1


def auto_threshold(histogram: Sequence[int], method: str = "Default",
                   subtract_one: bool = True) -> int:
    """``AutoThresholder.getThreshold(method, histogram)``: the threshold bin index (pixels
    <= threshold are "below"). Histograms longer than 256 bins are trimmed to the occupied
    range first, as ImageJ does. ``method`` may carry options ("Otsu dark"); only the first
    word counts. Unknown methods raise ValueError."""
    method = method.split(" ")[0] if method else "Default"
    fn = _METHOD_FUNCS.get(method)
    if fn is None:
        raise ValueError(f'Invalid method ("{method}")')
    data = [int(v) for v in histogram]
    threshold = bilevel(data, subtract_one)
    if threshold >= 0:
        return threshold
    minbin, maxbin = 0, -1
    data2 = data
    if len(data) > 256:
        minbin = -1
        for i in range(len(data)):
            if data[i] > 0:
                maxbin = i
        for i in range(len(data) - 1, -1, -1):
            if data[i] > 0:
                minbin = i
        data2 = data[minbin:maxbin + 1]
    threshold = fn(list(data2))
    if threshold == -1:
        threshold = 0
    return threshold + minbin


def _default_isodata(data: list[int]) -> int:
    n = len(data)
    data2 = list(data)
    mode = max_count = 0
    for i in range(n):
        if data2[i] > max_count:
            max_count = data2[i]
            mode = i
    max_count2 = 0
    for i in range(n):
        if data2[i] > max_count2 and i != mode:
            max_count2 = data2[i]
    hmax = max_count
    if hmax > max_count2 * 2 and max_count2 != 0:
        hmax = int(max_count2 * 1.5)
        data2[mode] = hmax
    return _ij_isodata(data2)


def _ij_isodata(data: list[int]) -> int:
    """The original ImageJ IsoData (``IJIsoData`` / ``getAutoThreshold(int[])``)."""
    max_value = len(data) - 1
    count0, count_max = data[0], data[max_value]
    data = list(data)
    data[0] = 0
    data[max_value] = 0
    lo = 0
    while data[lo] == 0 and lo < max_value:
        lo += 1
    hi = max_value
    while data[hi] == 0 and hi > 0:
        hi -= 1
    if lo >= hi:
        return len(data) // 2
    # Java sums i*data[i] and data[i] in doubles; all terms are integers below 2^53, so
    # cumulative sums give exactly the same values
    d = np.asarray(data, dtype=np.float64)
    wi = np.arange(len(d), dtype=np.float64) * d
    c_w = np.cumsum(wi)
    c_n = np.cumsum(d)
    base_w = c_w[lo - 1] if lo > 0 else 0.0
    base_n = c_n[lo - 1] if lo > 0 else 0.0
    moving = lo
    while True:
        sum1 = float(c_w[moving] - base_w)
        sum2 = float(c_n[moving] - base_n)
        sum3 = float(c_w[hi] - c_w[moving])
        sum4 = float(c_n[hi] - c_n[moving])
        result = (_div(sum1, sum2) + _div(sum3, sum4)) / 2.0
        moving += 1
        if not ((moving + 1) <= result and moving < hi - 1):
            break
    return _jround(result)


def _div(a: float, b: float) -> float:
    """Java double division (x/0 -> +-Infinity or NaN)."""
    if b == 0:
        if a == 0 or a != a:
            return float("nan")
        return math.copysign(float("inf"), a)
    return a / b


def _huang(data: list[int]) -> int:
    n = len(data)
    first_bin = 0
    for ih in range(n):
        if data[ih] != 0:
            first_bin = ih
            break
    last_bin = n - 1
    for ih in range(n - 1, first_bin - 1, -1):
        if data[ih] != 0:
            last_bin = ih
            break
    term = _div(1.0, float(last_bin - first_bin))
    mu_0 = [0.0] * n
    sum_pix = num_pix = 0
    for ih in range(first_bin, n):
        sum_pix = _i32(sum_pix + ih * data[ih])
        num_pix = _i32(num_pix + data[ih])
        mu_0[ih] = _div(float(sum_pix), float(num_pix))
    mu_1 = [0.0] * n
    sum_pix = num_pix = 0
    for ih in range(last_bin, 0, -1):
        sum_pix = _i32(sum_pix + ih * data[ih])
        num_pix = _i32(num_pix + data[ih])
        mu_1[ih - 1] = _div(float(sum_pix), float(num_pix))
    if n > 4096:
        return _huang_numpy(data, term, mu_0, mu_1)
    threshold = -1
    min_ent = 1.7976931348623157e308
    log = math.log
    for it in range(n):
        ent = 0.0
        m0 = mu_0[it]
        for ih in range(it + 1):
            mu_x = 1.0 / (1.0 + term * abs(ih - m0))
            if not (mu_x < 1e-06 or mu_x > 0.999999):
                ent += data[ih] * (-mu_x * log(mu_x) - (1.0 - mu_x) * log(1.0 - mu_x))
        m1 = mu_1[it]
        for ih in range(it + 1, n):
            mu_x = 1.0 / (1.0 + term * abs(ih - m1))
            if not (mu_x < 1e-06 or mu_x > 0.999999):
                ent += data[ih] * (-mu_x * log(mu_x) - (1.0 - mu_x) * log(1.0 - mu_x))
        if ent < min_ent:
            min_ent = ent
            threshold = it
    return threshold


def _huang_numpy(data, term, mu_0, mu_1) -> int:
    """Huang for long (16-bit) histograms, vectorised per threshold (summation order may
    differ from Java in the last bits)."""
    d = np.asarray(data, float)
    idx = np.arange(len(d), dtype=float)
    best, threshold = np.inf, -1
    with np.errstate(divide="ignore", invalid="ignore"):
        for it in range(len(d)):
            mu = np.where(idx <= it, mu_0[it], mu_1[it])
            mx = 1.0 / (1.0 + term * np.abs(idx - mu))
            ok = ~((mx < 1e-06) | (mx > 0.999999))
            e = np.where(ok, d * (-mx * np.log(mx) - (1 - mx) * np.log(1 - mx)), 0.0)
            ent = float(e.sum())
            if ent < best:
                best, threshold = ent, it
    return threshold


def _huang2(data: list[int]) -> int:
    n = len(data)
    first = 0
    while first < n and data[first] == 0:
        first += 1
    last = n - 1
    while last > first and data[last] == 0:
        last -= 1
    if first == last:
        return 0
    S = [0.0] * (last + 1)
    W = [0.0] * (last + 1)
    S[0] = data[0]
    for i in range(max(1, first), last + 1):
        S[i] = S[i - 1] + data[i]
        W[i] = W[i - 1] + i * data[i]
    C = float(last - first)
    smu = [0.0] * (last + 1 - first)
    for i in range(1, len(smu)):
        mu = 1 / (1 + abs(i) / C)
        smu[i] = -mu * math.log(mu) - (1 - mu) * math.log(1 - mu)
    best_t, best_e = 0, 1.7976931348623157e308
    for t in range(first, last + 1):
        entropy = 0.0
        mu = _jround(_div(W[t], S[t]))
        for i in range(first, t + 1):
            entropy += smu[abs(i - mu)] * data[i]
        mu = _jround(_div(W[last] - W[t], S[last] - S[t]))
        for i in range(t + 1, last + 1):
            entropy += smu[abs(i - mu)] * data[i]
        if best_e > entropy:
            best_e, best_t = entropy, t
    return best_t


def _isodata(data: list[int]) -> int:
    n = len(data)
    g = 0
    for i in range(1, n):
        if data[i] > 0:
            g = i + 1
            break
    d = np.asarray(data, dtype=np.int64)
    c_n = np.concatenate(([0], np.cumsum(d)))            # c_n[k] = sum(data[:k])
    c_w = np.concatenate(([0], np.cumsum(d * np.arange(n, dtype=np.int64))))
    while True:
        # int sums wrap like Java (the wrap of a sum equals the sum of wrapped terms)
        totl = _i32(int(c_n[g]))
        lo = _i32(int(c_w[g]))
        toth = _i32(int(c_n[n] - c_n[min(g + 1, n)]))
        hi = _i32(int(c_w[n] - c_w[min(g + 1, n)]))
        if totl > 0 and toth > 0:
            lo = int(lo / totl)          # Java int division truncates toward zero
            hi = int(hi / toth)
            if g == _jround((lo + hi) / 2.0):
                break
        g += 1
        if g > n - 2:
            return -1
    return g


def _li(data: list[int]) -> int:
    n = len(data)
    tolerance = 0.5
    num_pixels = 0
    for v in data:
        num_pixels = _i32(num_pixels + v)
    mean = 0.0
    for ih in range(n):
        mean += ih * data[ih]
    mean /= num_pixels
    new_thresh = mean
    while True:
        old_thresh = new_thresh
        threshold = _jint(old_thresh + 0.5)
        sum_back = num_back = 0
        for ih in range(threshold + 1):
            sum_back = _i32(sum_back + ih * data[ih])
            num_back = _i32(num_back + data[ih])
        mean_back = 0.0 if num_back == 0 else sum_back / float(num_back)
        sum_obj = num_obj = 0
        for ih in range(threshold + 1, n):
            sum_obj = _i32(sum_obj + ih * data[ih])
            num_obj = _i32(num_obj + data[ih])
        mean_obj = 0.0 if num_obj == 0 else sum_obj / float(num_obj)
        temp = _div(mean_back - mean_obj, _jlog(mean_back) - _jlog(mean_obj))
        if temp < -2.220446049250313E-16:
            new_thresh = float(_jint(temp - 0.5))
        else:
            new_thresh = float(_jint(temp + 0.5))
        if not abs(new_thresh - old_thresh) > tolerance:
            break
    return threshold


def _jlog(x: float) -> float:
    if x != x or x < 0:
        return float("nan")
    if x == 0:
        return float("-inf")
    if x == float("inf"):
        return x
    return math.log(x)


def _norm_cumulative(data: list[int]):
    n = len(data)
    total = 0
    for v in data:
        total = _i32(total + v)
    norm = [v / total for v in data] if total else [float("nan")] * n
    p1 = [0.0] * n
    p2 = [0.0] * n
    p1[0] = norm[0]
    p2[0] = 1.0 - p1[0]
    for ih in range(1, n):
        p1[ih] = p1[ih - 1] + norm[ih]
        p2[ih] = 1.0 - p1[ih]
    first_bin = 0
    for ih in range(n):
        if not abs(p1[ih]) < _DBL_EPS:
            first_bin = ih
            break
    last_bin = n - 1
    for ih in range(n - 1, first_bin - 1, -1):
        if not abs(p2[ih]) < _DBL_EPS:
            last_bin = ih
            break
    return norm, p1, p2, first_bin, last_bin


def _entropy_terms_np(norm, p1, p2, first_bin, last_bin, nonzero):
    """Vectorised Shannon entropies of both classes for every threshold (long histograms)."""
    nrm = np.asarray(norm)
    with np.errstate(divide="ignore", invalid="ignore"):
        nlogn = np.where(nonzero, nrm * np.log(np.where(nonzero, nrm, 1.0)), 0.0)
        cs = np.cumsum(nlogn)
        cnt = np.cumsum(np.where(nonzero, nrm, 0.0))
        total_nl, total_cnt = cs[-1], cnt[-1]
        its = np.arange(first_bin, last_bin + 1)
        P1 = np.asarray(p1)[its]
        P2 = np.asarray(p2)[its]
        back = -(cs[its] / P1) + np.log(P1) * cnt[its] / P1
        obj = -((total_nl - cs[its]) / P2) + np.log(P2) * (total_cnt - cnt[its]) / P2
    return its, np.nan_to_num(back + obj, nan=-np.inf)


def _max_entropy(data: list[int]) -> int:
    n = len(data)
    norm, p1, p2, first_bin, last_bin = _norm_cumulative(data)
    if n > 4096:
        its, tot = _entropy_terms_np(norm, p1, p2, first_bin, last_bin, np.asarray(data) != 0)
        k = int(np.argmax(tot))
        return int(its[k]) if tot[k] > 4.9e-324 else -1
    threshold = -1
    max_ent = 4.9e-324                      # Double.MIN_VALUE
    log = math.log
    for it in range(first_bin, last_bin + 1):
        ent_back = 0.0
        a = p1[it]
        for ih in range(it + 1):
            if data[ih] != 0:
                q = norm[ih] / a
                ent_back -= q * log(q)
        ent_obj = 0.0
        b = p2[it]
        for ih in range(it + 1, n):
            if data[ih] != 0:
                q = _div(norm[ih], b)
                ent_obj -= q * _jlog(q)
        tot = ent_back + ent_obj
        if max_ent < tot:
            max_ent = tot
            threshold = it
    return threshold


def _mean(data: list[int]) -> int:
    tot = s = 0
    for i, v in enumerate(data):
        tot += v
        s += i * v
    return int(s // tot) if tot else 0


def _moments(data: list[int]) -> int:
    n = len(data)
    total = 0.0
    for v in data:
        total += v
    histo = [v / total for v in data]
    m0, m1, m2, m3 = 1.0, 0.0, 0.0, 0.0
    for i in range(n):
        m1 += i * histo[i]
        m2 += _i32(i * i) * histo[i]
        m3 += _i32(i * i * i) * histo[i]
    cd = m0 * m2 - m1 * m1
    c0 = (-m2 * m2 + m1 * m3) / cd
    c1 = (m0 * -m3 + m2 * m1) / cd
    disc = c1 * c1 - 4.0 * c0
    sq = math.sqrt(disc) if disc >= 0 else float("nan")
    z0 = 0.5 * (-c1 - sq)
    z1 = 0.5 * (-c1 + sq)
    p0 = _div(z1 - m1, z1 - z0)
    s = 0.0
    for i in range(n):
        s += histo[i]
        if s > p0:
            return i
    return -1


def _otsu(data: list[int]) -> int:
    n = len(data)
    num_pixels = 0
    for v in data:
        num_pixels = _i32(num_pixels + v)
    term = 1.0 / float(num_pixels)
    histo = [term * v for v in data]
    cnh = [0.0] * n
    mean = [0.0] * n
    cnh[0] = histo[0]
    for ih in range(1, n):
        cnh[ih] = cnh[ih - 1] + histo[ih]
    for ih in range(1, n):
        mean[ih] = mean[ih - 1] + ih * histo[ih]
    total_mean = mean[n - 1]
    threshold = -2147483648
    max_bcv = 0.0
    for ih in range(n):
        bcv = total_mean * cnh[ih] - mean[ih]
        bcv *= _div(bcv, cnh[ih] * (1.0 - cnh[ih]))
        if max_bcv < bcv:
            max_bcv = bcv
            threshold = ih
    return threshold


def _renyi_entropy(data: list[int]) -> int:
    n = len(data)
    norm, p1, p2, first_bin, last_bin = _norm_cumulative(data)
    if n > 4096:
        return _renyi_np(data, norm, p1, p2, first_bin, last_bin)
    log, sqrt = math.log, math.sqrt
    threshold = 0
    max_ent = 0.0
    for it in range(first_bin, last_bin + 1):
        ent_back = 0.0
        for ih in range(it + 1):
            if data[ih] != 0:
                q = norm[ih] / p1[it]
                ent_back -= q * log(q)
        ent_obj = 0.0
        for ih in range(it + 1, n):
            if data[ih] != 0:
                q = _div(norm[ih], p2[it])
                ent_obj -= q * _jlog(q)
        tot = ent_back + ent_obj
        if max_ent < tot:
            max_ent = tot
            threshold = it
    t_star2 = threshold

    threshold = 0
    max_ent = 0.0
    term = 1.0 / (1.0 - 0.5)
    for it in range(first_bin, last_bin + 1):
        ent_back = 0.0
        for ih in range(it + 1):
            ent_back += sqrt(norm[ih] / p1[it])
        ent_obj = 0.0
        for ih in range(it + 1, n):
            q = _div(norm[ih], p2[it])
            ent_obj += sqrt(q) if q >= 0 else float("nan")
        prod = ent_back * ent_obj
        tot = term * (log(prod) if prod > 0.0 else 0.0)
        if tot > max_ent:
            max_ent = tot
            threshold = it
    t_star1 = threshold

    threshold = 0
    max_ent = 0.0
    term = 1.0 / (1.0 - 2.0)
    for it in range(first_bin, last_bin + 1):
        ent_back = 0.0
        for ih in range(it + 1):
            ent_back += (norm[ih] * norm[ih]) / (p1[it] * p1[it])
        ent_obj = 0.0
        for ih in range(it + 1, n):
            ent_obj += _div(norm[ih] * norm[ih], p2[it] * p2[it])
        prod = ent_back * ent_obj
        tot = term * (log(prod) if prod > 0.0 else 0.0)
        if tot > max_ent:
            max_ent = tot
            threshold = it
    t_star3 = threshold

    return _renyi_combine(p1, p2, t_star1, t_star2, t_star3)


def _renyi_np(data, norm, p1, p2, first_bin, last_bin) -> int:
    """RenyiEntropy for long (16-bit) histograms, vectorised (summation order may differ
    from Java in the last bits)."""
    nrm = np.asarray(norm)
    its, tot = _entropy_terms_np(norm, p1, p2, first_bin, last_bin, np.asarray(data) != 0)
    t2 = int(its[int(np.argmax(tot))]) if tot.max() > 0 else 0
    P1 = np.asarray(p1)[its]
    P2 = np.asarray(p2)[its]
    with np.errstate(divide="ignore", invalid="ignore"):
        sq = np.cumsum(np.sqrt(nrm))
        back = sq[its] / np.sqrt(P1)
        obj = (sq[-1] - sq[its]) / np.sqrt(P2)
        prod = back * obj
        e1 = 2.0 * np.where(prod > 0, np.log(np.where(prod > 0, prod, 1)), 0.0)
        s2 = np.cumsum(nrm * nrm)
        back = s2[its] / (P1 * P1)
        obj = (s2[-1] - s2[its]) / (P2 * P2)
        prod = back * obj
        e3 = -1.0 * np.where(prod > 0, np.log(np.where(prod > 0, prod, 1)), 0.0)
    e1 = np.nan_to_num(e1, nan=-np.inf)
    e3 = np.nan_to_num(e3, nan=-np.inf)
    t1 = int(its[int(np.argmax(e1))]) if e1.max() > 0 else 0
    t3 = int(its[int(np.argmax(e3))]) if e3.max() > 0 else 0
    return _renyi_combine(p1, p2, t1, t2, t3)


def _renyi_combine(p1, p2, t_star1, t_star2, t_star3) -> int:
    if t_star2 < t_star1:
        t_star1, t_star2 = t_star2, t_star1
    if t_star3 < t_star2:
        t_star2, t_star3 = t_star3, t_star2
    if t_star2 < t_star1:
        t_star1, t_star2 = t_star2, t_star1
    if abs(t_star1 - t_star2) <= 5:
        if abs(t_star2 - t_star3) <= 5:
            beta1, beta2, beta3 = 1, 2, 1
        else:
            beta1, beta2, beta3 = 0, 1, 3
    else:
        if abs(t_star2 - t_star3) <= 5:
            beta1, beta2, beta3 = 3, 1, 0
        else:
            beta1, beta2, beta3 = 1, 2, 1
    omega = p1[t_star3] - p1[t_star1]
    return _jint(t_star1 * (p1[t_star1] + 0.25 * omega * beta1) + 0.25 * t_star2 * omega * beta2
                 + t_star3 * (p2[t_star3] + 0.25 * omega * beta3))


def _shanbhag(data: list[int]) -> int:
    n = len(data)
    norm, p1, p2, first_bin, last_bin = _norm_cumulative(data)
    threshold = -1
    min_ent = 1.7976931348623157e308
    if n > 4096:
        nrm, P1, P2 = np.asarray(norm), np.asarray(p1), np.asarray(p2)
        with np.errstate(divide="ignore", invalid="ignore"):
            for it in range(first_bin, last_bin + 1):
                term = 0.5 / P1[it]
                eb = -float(np.sum(nrm[1:it + 1] * np.log(1.0 - term * P1[:it]))) * term
                term = 0.5 / P2[it]
                eo = -float(np.sum(nrm[it + 1:] * np.log(1.0 - term * P2[it + 1:]))) * term
                tot = abs(eb - eo)
                if tot < min_ent:
                    min_ent, threshold = tot, it
        return threshold
    for it in range(first_bin, last_bin + 1):
        ent_back = 0.0
        term = _div(0.5, p1[it])
        for ih in range(1, it + 1):
            ent_back -= norm[ih] * _jlog(1.0 - term * p1[ih - 1])
        ent_back *= term
        ent_obj = 0.0
        term = _div(0.5, p2[it])
        for ih in range(it + 1, n):
            ent_obj -= norm[ih] * _jlog(1.0 - term * p2[ih])
        ent_obj *= term
        tot = abs(ent_back - ent_obj)
        if tot < min_ent:
            min_ent = tot
            threshold = it
    return threshold


def _triangle(data: list[int]) -> int:
    data = list(data)
    n = len(data)
    lo = dmax = mx = lo2 = 0
    for i in range(n):
        if data[i] > 0:
            lo = i
            break
    if lo > 0:
        lo -= 1
    for i in range(n - 1, 0, -1):
        if data[i] > 0:
            lo2 = i
            break
    if lo2 < n - 1:
        lo2 += 1
    for i in range(n):
        if data[i] > dmax:
            mx = i
            dmax = data[i]
    inverted = False
    if (mx - lo) < (lo2 - mx):
        inverted = True
        data.reverse()
        lo = n - 1 - lo2
        mx = n - 1 - mx
    if lo == mx:
        return lo
    nx = float(data[mx])
    ny = float(lo - mx)
    d = math.sqrt(nx * nx + ny * ny)
    nx /= d
    ny /= d
    d = nx * lo + ny * data[lo]
    split = lo
    split_distance = 0.0
    for i in range(lo + 1, mx + 1):
        new_distance = nx * i + ny * data[i] - d
        if new_distance > split_distance:
            split = i
            split_distance = new_distance
    split -= 1
    if inverted:
        return n - 1 - split
    return split


def _yen(data: list[int]) -> int:
    n = len(data)
    total = 0
    for v in data:
        total = _i32(total + v)
    norm = [v / total for v in data]
    p1 = [0.0] * n
    p1_sq = [0.0] * n
    p2_sq = [0.0] * n
    p1[0] = norm[0]
    for ih in range(1, n):
        p1[ih] = p1[ih - 1] + norm[ih]
    p1_sq[0] = norm[0] * norm[0]
    for ih in range(1, n):
        p1_sq[ih] = p1_sq[ih - 1] + norm[ih] * norm[ih]
    for ih in range(n - 2, -1, -1):
        p2_sq[ih] = p2_sq[ih + 1] + norm[ih + 1] * norm[ih + 1]
    threshold = -1
    max_crit = 4.9e-324
    log = math.log
    for it in range(n):
        a = p1_sq[it] * p2_sq[it]
        b = p1[it] * (1.0 - p1[it])
        crit = -1.0 * (log(a) if a > 0.0 else 0.0) + 2 * (log(b) if b > 0.0 else 0.0)
        if crit > max_crit:
            max_crit = crit
            threshold = it
    return threshold


# ---- clean-room methods (from the papers; not derived from ImageJ's GPL-origin code) ----
def _smooth3(y: np.ndarray) -> np.ndarray:
    """One pass of a 3-point running mean; at the two ends the missing neighbour counts
    as zero (window truncated at the histogram ends)."""
    n = len(y)
    if n < 3:
        return y.copy()
    out = np.empty_like(y)
    out[0] = (y[0] + y[1]) / 3.0
    out[1:-1] = (y[:-2] + y[1:-1] + y[2:]) / 3.0
    out[-1] = (y[-2] + y[-1]) / 3.0
    return out


def _peaks(y) -> list[int]:
    """Indices of strict local maxima (higher than both neighbours)."""
    y = np.asarray(y)
    k = np.nonzero((y[1:-1] > y[:-2]) & (y[1:-1] > y[2:]))[0] + 1
    return k.tolist()


def _smoothed_until_bimodal(data: list[int], max_iter: int = 10000) -> np.ndarray | None:
    """Prewitt & Mendelsohn (1966): smooth the histogram repeatedly with a 3-point mean
    until it has exactly two local maxima; None if that never happens."""
    y = np.asarray(data, dtype=np.float64)
    for _ in range(max_iter):
        if len(_peaks(y)) == 2:
            return y
        y = _smooth3(y)
    return None


def _intermodes(data: list[int]) -> int:
    """Threshold midway between the two modes of the smoothed bimodal histogram."""
    y = _smoothed_until_bimodal(data)
    if y is None:
        return -1
    p1, p2 = _peaks(y)
    return (p1 + p2) // 2


def _minimum(data: list[int]) -> int:
    """Threshold at the valley (lowest point) between the two modes of the smoothed
    bimodal histogram."""
    y = _smoothed_until_bimodal(data)
    if y is None:
        return -1
    p1, p2 = _peaks(y)
    for k in range(p1 + 1, p2):
        if y[k - 1] > y[k] <= y[k + 1]:
            return k
    return p1


def _percentile(data: list[int], fraction: float = 0.5) -> int:
    """Doyle (1962) p-tile method with p = 50%: the grey level at which the cumulative
    fraction of pixels is closest to ``fraction``."""
    total = float(sum(data))
    if total <= 0:
        return 0
    best_i, best = 0, float("inf")
    acc = 0.0
    for i, v in enumerate(data):
        acc += v
        d = abs(acc / total - fraction)
        if d < best:
            best, best_i = d, i
    return best_i


def _jfloor_int(x: float) -> int:
    """``(int)Math.floor(x)`` with Java semantics for NaN/infinity."""
    if x != x:
        return 0
    if x in (float("inf"), float("-inf")):
        return _jint(x)
    return _jint(math.floor(x))


def _minerror(data: list[int], log: Callable[[float], float] = math.log10) -> int:
    """Kittler & Illingworth (1986) minimum-error thresholding, iterative form.

    Each class is modelled as a normal distribution. From the current threshold T the
    priors P1, P2, means m1, m2 and variances v1, v2 of the two classes are estimated; the
    new threshold is where the weighted class densities are equal, i.e. the root of
    ``(1/v1 - 1/v2) T^2 - 2 (m1/v1 - m2/v2) T + (m1^2/v1 - m2^2/v2 + L) = 0`` with
    ``L = log(v1 P2^2 / (v2 P1^2))`` taken as ``(w1 + sqrt(w1^2 - w0 w2)) / w0`` and
    floored; iterate from the mean grey level until T no longer changes (stop when the
    discriminant is negative). Black-box comparison with ImageJ 1.54p shows that ImageJ
    evaluates ``L`` with a base-10 logarithm, hence the default ``log``; pass
    ``math.log`` for the textbook natural-log form. When the new threshold is undefined
    (NaN: an empty or zero-variance class) the iteration stops at the current threshold,
    as ImageJ does ("NaN, not converging"). Known difference: with the 65536-bin
    "16-bit" histogram option ImageJ's result differs (its GPL-derived implementation is
    not reproduced); 256-bin histograms match ImageJ."""
    n = len(data)
    t = _mean(data)
    for _ in range(10000):
        lo_end = min(max(t + 1, 0), n)
        c1 = float(sum(data[:lo_end]))
        c2 = float(sum(data[lo_end:]))
        s1 = float(sum(i * data[i] for i in range(lo_end)))
        s2 = float(sum(i * data[i] for i in range(lo_end, n)))
        m1, m2 = _div(s1, c1), _div(s2, c2)
        v1 = _div(sum((i - m1) ** 2 * data[i] for i in range(lo_end)), c1)
        v2 = _div(sum((i - m2) ** 2 * data[i] for i in range(lo_end, n)), c2)
        tot = c1 + c2
        p, q = _div(c1, tot), _div(c2, tot)
        w0 = _div(1.0, v1) - _div(1.0, v2)
        w1 = _div(m1, v1) - _div(m2, v2)
        ratio = _div(v1 * q * q, v2 * p * p)
        lg = float("nan") if ratio != ratio or ratio < 0 else (
            float("-inf") if ratio == 0 else (ratio if ratio == float("inf") else log(ratio)))
        w2 = _div(m1 * m1, v1) - _div(m2 * m2, v2) + lg
        disc = w1 * w1 - w0 * w2
        if disc < 0:
            break
        root = math.sqrt(disc) if disc == disc and disc != float("inf") else disc
        x = _div(w1 + root, w0)
        if x != x:            # undefined (an empty or zero-variance class): stop here
            break
        new = _jfloor_int(x)
        if new == t:
            break
        t = new
    return t


_METHOD_FUNCS: dict[str, Callable[[list[int]], int]] = {
    "Default": _default_isodata, "IJ_IsoData": _ij_isodata, "Huang": _huang,
    "Huang2": _huang2, "Intermodes": _intermodes, "IsoData": _isodata, "Li": _li,
    "MaxEntropy": _max_entropy, "Mean": _mean, "MinError": _minerror,
    "MinError(I)": _minerror, "Minimum": _minimum, "Moments": _moments, "Otsu": _otsu,
    "Percentile": _percentile, "RenyiEntropy": _renyi_entropy, "Shanbhag": _shanbhag,
    "Triangle": _triangle, "Yen": _yen,
}


def ij_auto_threshold(histogram: Sequence[int]) -> int:
    """``ImageProcessor.getAutoThreshold(int[])`` (legacy IsoData)."""
    return _ij_isodata([int(v) for v in histogram])


# ===================================================================================
# histograms of image planes (ImageJ conventions)
# ===================================================================================
def _bit_depth(plane: np.ndarray) -> int:
    return {np.dtype(np.uint8): 8, np.dtype(np.uint16): 16,
            np.dtype(np.float32): 32}.get(plane.dtype, 32)


def to_byte_scaled(plane: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """``ip.convertToByte(true)`` with display range vmin..vmax (TypeConverter /
    FloatProcessor.create8BitImage)."""
    a = np.asarray(plane)
    if a.dtype == np.uint8:
        return a.copy()
    if a.dtype == np.uint16:
        lo, hi = int(vmin), int(vmax)
        scale = 256.0 / (hi - lo + 1)
        v = np.maximum(a.astype(np.int64) - lo, 0)
        return np.minimum((v * scale + 0.5).astype(np.int64), 255).astype(np.uint8)
    f = a.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        scale = 255.0 / (vmax - vmin)
        v = f - vmin
        v = np.where(v < 0.0, 0.0, v)            # NaN stays NaN (comparison false)
        iv = v * scale + 0.5
    iv = np.where(np.isnan(iv), 0.0, iv)          # (int)NaN = 0
    iv = np.minimum(iv, 2147483647.0)
    return np.minimum(iv.astype(np.int64), 255).astype(np.uint8)


def rgb_to_byte(rgb: np.ndarray, weighted: bool = False) -> np.ndarray:
    """``TypeConverter.convertRGBToByte``: (byte)(r*rw + g*gw + b*bw + 0.5)."""
    w = (0.299, 0.587, 0.114) if weighted else (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    r, g, b = (rgb[..., k].astype(np.float64) for k in range(3))
    return (r * w[0] + g * w[1] + b * w[2] + 0.5).astype(np.int64).astype(np.uint8)


def float_min_max(plane: np.ndarray) -> tuple[float, float]:
    """``FloatProcessor.findMinAndMax`` (NaN skipped, infinities kept) / integer min-max."""
    a = np.asarray(plane)
    if a.dtype == np.float32:
        finite = a[~np.isnan(a)]
        if finite.size == 0:
            return float("nan"), float("nan")
        return float(finite.min()), float(finite.max())
    return float(a.min()), float(a.max())


def byte_histogram(plane8: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    data = plane8 if mask is None else plane8[mask]
    return np.bincount(data.ravel(), minlength=256)[:256].astype(np.int64)


def image_histogram(plane: np.ndarray, bit_depth: int | None = None,
                    display_range: tuple[float, float] | None = None,
                    mask: np.ndarray | None = None) -> np.ndarray:
    """The 256-bin histogram ImageJ thresholds: 8-bit counts, or for 16/32-bit the histogram
    of the plane converted to 8 bits with ``display_range`` (default: plane min/max)."""
    bit_depth = bit_depth or _bit_depth(plane)
    if bit_depth == 8:
        return byte_histogram(np.asarray(plane), mask)
    if display_range is None:
        display_range = float_min_max(plane)
    return byte_histogram(to_byte_scaled(plane, *display_range), mask)


def histogram16(plane: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    data = plane if mask is None else plane[mask]
    return np.bincount(np.asarray(data, np.int64).ravel(), minlength=65536)[:65536]


# ===================================================================================
# legacy setAutoThreshold(ISODATA2) - Gaussian smoothing of the histogram
# ===================================================================================
def _gaussian_kernel(sigma: float, accuracy: float, max_radius: int):
    """GaussianBlur.makeGaussianKernel (float32 kernel and running sums)."""
    f32 = np.float32
    k_radius = int(math.ceil(sigma * math.sqrt(-2 * math.log(accuracy)))) + 1
    if max_radius < 50:
        max_radius = 50
    if k_radius > max_radius:
        k_radius = max_radius
    k0 = [f32(math.exp(-0.5 * i * i / sigma / sigma)) for i in range(k_radius)]
    if k_radius < max_radius and k_radius > 3:
        sqrt_slope = 1.7976931348623157e308
        r = k_radius
        while r > k_radius // 2:
            r -= 1
            a = math.sqrt(float(k0[r])) / (k_radius - r)
            if a < sqrt_slope:
                sqrt_slope = a
            else:
                break
        for r1 in range(r + 2, k_radius):
            k0[r1] = f32((k_radius - r1) * (k_radius - r1) * sqrt_slope * sqrt_slope)
    if k_radius < max_radius:
        s = float(k0[0])
        for i in range(1, k_radius):
            s += 2 * float(k0[i])
    else:
        s = sigma * math.sqrt(2 * math.pi)
    rsum = 0.5 + 0.5 * float(k0[0]) / s
    kern, ksum = [], []
    for i in range(k_radius):
        v = float(k0[i]) / s
        kern.append(f32(v))
        rsum -= v
        ksum.append(f32(rsum))
    return kern, ksum


def _blur_line(values: Sequence[float], sigma: float = 2.0, accuracy: float = 0.01
               ) -> list:
    """GaussianBlur.blur1Direction of one line (no downscaling for small sigma),
    float32 arithmetic as convolveLine."""
    f32 = np.float32
    inp = [f32(v) for v in values]
    length = len(inp)
    kern, ksum = _gaussian_kernel(sigma, accuracy, length)
    k_radius = len(kern)
    first, last = inp[0], inp[length - 1]
    out = [f32(0)] * length
    k0 = kern[0]

    def edge_sum(i, result):
        for k in range(1, k_radius):
            v = f32(0)
            if i - k >= 0:
                v = f32(v + inp[i - k])
            if i + k < length:
                v = f32(v + inp[i + k])
            result = f32(result + f32(kern[k] * v))
        return result

    first_part = k_radius if k_radius < length else length
    i = 0
    while i < first_part:
        result = f32(inp[i] * k0)
        result = f32(result + f32(ksum[i] * first))
        if i + k_radius > length:
            result = f32(result + f32(ksum[length - i - 1] * last))
        out[i] = edge_sum(i, result)
        i += 1
    end_inside = length - k_radius if length - k_radius < length else length
    while i < end_inside:
        result = f32(inp[i] * k0)
        for k in range(1, k_radius):
            result = f32(result + f32(kern[k] * f32(inp[i - k] + inp[i + k])))
        out[i] = result
        i += 1
    while i < length:
        result = f32(inp[i] * k0)
        if i < k_radius:
            result = f32(result + f32(ksum[i] * first))
        if i + k_radius >= length:
            result = f32(result + f32(ksum[length - i - 1] * last))
        out[i] = edge_sum(i, result)
        i += 1
    return out


def legacy_threshold_8bit(hist: np.ndarray, inverted_lut: bool, stats_min: int,
                          stats_max: int) -> tuple[float, float]:
    """``ImageProcessor.setAutoThreshold(ISODATA2, ...)`` on a 256-bin byte histogram:
    (lower, upper) in 0..255. ``stats_min/max`` are the lowest/highest occupied bins."""
    histogram = [int(v) for v in hist]
    mode, max_count = 0, 0
    for i in range(256):
        if histogram[i] > max_count:
            max_count = histogram[i]
            mode = i
    original_mode_count = histogram[mode]
    max_count2 = 0
    for i in range(256):
        if histogram[i] > max_count2 and i != mode:
            max_count2 = histogram[i]
    hmax = max_count
    if hmax > max_count2 * 2 and max_count2 != 0:
        hmax = int(max_count2 * 1.5)
        histogram[mode] = hmax
    threshold = _ij_isodata(histogram)
    histogram[mode] = original_mode_count
    smoothed = _blur_line([float(v) for v in histogram], 2.0, 0.01)
    f32 = np.float32
    mx, total = f32(0), f32(0)
    smode = 0
    for i in range(256):
        count = smoothed[i]
        total = f32(total + count)
        if count > mx:
            mx = count
            smode = i
    avg = float(total) / 256.0
    if float(mx) / avg > 1.5:
        if (stats_max - smode) > (smode - stats_min):
            return float(threshold), 255.0
        return 0.0, float(threshold)
    if inverted_lut:
        return float(threshold), 255.0
    return 0.0, float(threshold)


# ===================================================================================
# setAutoThreshold for planes / images
# ===================================================================================
@dataclass
class AutoThresholdResult:
    lower: float
    upper: float
    display_min: float          # display range in effect (reset for 16/32-bit)
    display_max: float
    threshold_bin: int = -1     # raw AutoThresholder result (0..255 or 16-bit level)


def parse_method(spec: str | None) -> dict:
    """Split an ImageJ method string ("Huang dark b&w 16-bit stack no-reset")."""
    if spec is None:
        return {"method": None}
    words = spec.split()
    method = words[0] if words else "Default"
    lut = "red"
    if "b&w" in spec:
        lut = "bw"
    if "over" in spec:
        lut = "over_under"
    if "no-lut" in spec:
        lut = "none"
    return {"method": method, "dark": "dark" in spec, "no_reset": "no-reset" in spec,
            "sixteen_bit": "16" in spec, "stack": "stack" in spec, "mode": lut}


def _scale_and_set(lower: float, upper: float, bit_depth: int, vmin: float, vmax: float
                   ) -> tuple[float, float]:
    """``ImageProcessor.scaleAndSetThreshold`` (0..255 levels -> raw)."""
    if bit_depth != 8 and lower != NO_THRESHOLD:
        if vmax > vmin:
            if lower == 0.0:
                if bit_depth == 32:
                    lower = min(vmin, -1e30)
            else:
                lower = vmin + (lower / 255.0) * (vmax - vmin)
            if upper == 255.0:
                if bit_depth == 16:
                    upper = 65535.0
                elif bit_depth == 32:
                    upper = max(vmax, 1e30)
            else:
                upper = vmin + (upper / 255.0) * (vmax - vmin)
        else:
            lower = upper = vmin
    return float(lower), float(upper)


def java_threshold_levels(lower: float, upper: float, bit_depth: int,
                          vmin: float = 0.0, vmax: float = 0.0) -> tuple[float, float]:
    """What ``ip.setThreshold(lower, upper)`` stores: 16-bit levels are clamped to
    0..65535 and rounded (``ShortProcessor``); 8-bit and float keep the values."""
    if bit_depth == 16:
        lower = max(lower, 0.0)
        upper = min(upper, 65535.0)
        return float(_jround(lower)), float(_jround(upper))
    return float(lower), float(upper)


def _levels(threshold: int, dark: bool, inverted: bool, add_one: bool, tmax: float):
    one = 1 if add_one else 0
    if dark:
        return (0.0, float(threshold)) if inverted else (float(threshold + one), tmax)
    return (float(threshold + one), tmax) if inverted else (0.0, float(threshold))


def auto_threshold_plane(plane: np.ndarray, method: str | None = "Default", dark: bool = False,
                         inverted_lut: bool = False, display_range=None,
                         no_reset: bool = False, sixteen_bit: bool = False,
                         mask: np.ndarray | None = None, over_under: bool = False
                         ) -> AutoThresholdResult:
    """``ImageProcessor.setAutoThreshold(method, dark, lut)`` on one 8/16/32-bit plane.
    ``method=None`` is the legacy ``setAutoThreshold()`` (ISODATA2). The display range of
    16/32-bit planes is reset to min/max unless ``no_reset`` (as ImageJ, which changes the
    display); the returned display range is the one ImageJ ends with."""
    plane = np.asarray(plane)
    bit_depth = _bit_depth(plane)
    given = (float(display_range[0]), float(display_range[1])) if display_range is not None \
        else None
    hist16 = sixteen_bit and bit_depth == 16 and method is not None
    if bit_depth == 8:
        vmin, vmax = given or (0.0, 255.0)
    elif method is None or (not hist16 and (not no_reset or over_under)) or given is None:
        vmin, vmax = float_min_max(plane)
    else:
        vmin, vmax = given
    if method is None:                     # legacy ISODATA2
        p8 = plane if bit_depth == 8 else to_byte_scaled(plane, vmin, vmax)
        hist = byte_histogram(p8, mask)
        nz = np.nonzero(hist)[0]
        smin, smax = (int(nz[0]), int(nz[-1])) if nz.size else (0, 0)
        lower, upper = legacy_threshold_8bit(hist, inverted_lut, smin, smax)
        lower, upper = _scale_and_set(lower, upper, bit_depth, vmin, vmax)
        lower, upper = java_threshold_levels(lower, upper, bit_depth)
        return AutoThresholdResult(lower, upper, vmin, vmax)
    add_one = not sixteen_bit
    if hist16:
        hist = histogram16(plane, mask)
    else:
        p8 = plane if bit_depth == 8 else to_byte_scaled(plane, vmin, vmax)
        hist = byte_histogram(p8, mask)
    t = auto_threshold(hist, method, subtract_one=add_one)
    tmax = 65535.0 if len(hist) > 256 else 255.0
    lower, upper = _levels(t, dark, inverted_lut, add_one, tmax)
    if not hist16:
        if lower > 255:
            lower = 255.0
        lower, upper = _scale_and_set(lower, upper, bit_depth, vmin, vmax)
    lower, upper = java_threshold_levels(lower, upper, bit_depth)
    return AutoThresholdResult(float(lower), float(upper), vmin, vmax, t)


def stack_histogram(planes: Iterable[np.ndarray], bit_depth: int,
                    mask: np.ndarray | None = None, sixteen_bit: bool = False):
    """``StackStatistics(imp)``: (histogram, min, max) of all planes (256 bins between the
    stack min and max for 16/32-bit, the 65536-bin histogram with ``sixteen_bit``)."""
    planes = [np.asarray(p) for p in planes]
    if bit_depth == 8:
        hist = sum(byte_histogram(p, mask) for p in planes)
        return hist, 0.0, 255.0
    if bit_depth == 16:
        h16 = sum(histogram16(p, mask) for p in planes)
        nz = np.nonzero(h16)[0]
        lo, hi = (int(nz[0]), int(nz[-1])) if nz.size else (0, 0)
        if sixteen_bit:
            return h16, float(lo), float(hi)
        bin_size = (hi - lo + 1) / 256.0
        scale = 1.0 / bin_size
        idx = (scale * (np.arange(lo, hi + 1) - lo)).astype(np.int64)
        idx = np.minimum(idx, 255)
        hist = np.bincount(idx, weights=h16[lo:hi + 1], minlength=256)[:256].astype(np.int64)
        return hist, float(lo), float(hi)
    vals = np.concatenate([(p if mask is None else p[mask]).ravel().astype(np.float64)
                           for p in planes])
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return np.zeros(256, np.int64), 0.0, 0.0
    lo, hi = float(vals.min()), float(vals.max())
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = 256 / (hi - lo)
        idx = (scale * (vals - lo))
    idx = np.where(np.isfinite(idx), idx, 0).astype(np.int64)
    idx = np.minimum(idx, 255)
    return np.bincount(idx, minlength=256)[:256].astype(np.int64), lo, hi


def auto_threshold_stack(planes: Sequence[np.ndarray], method: str = "Default",
                         dark: bool = False, inverted_lut: bool = False,
                         sixteen_bit: bool = False, mask: np.ndarray | None = None
                         ) -> AutoThresholdResult:
    """``IJ.setStackThreshold``: threshold from the stack histogram."""
    bit_depth = _bit_depth(np.asarray(planes[0]))
    histo16 = bit_depth == 16 and sixteen_bit
    add_one = not sixteen_bit
    hist, smin, smax = stack_histogram(planes, bit_depth, mask, histo16)
    vmin, vmax = (0.0, 255.0) if bit_depth == 8 else (smin, smax)
    t = auto_threshold(hist, method, subtract_one=add_one)
    tmax = 65535.0 if len(hist) > 256 else 255.0
    lower, upper = _levels(t, dark, inverted_lut, add_one, tmax)
    if not histo16:
        if lower > 255:
            lower = 255.0
        if vmax > vmin:
            lower = vmin + (lower / 255.0) * (vmax - vmin)
            upper = vmin + (upper / 255.0) * (vmax - vmin)
        else:
            lower = upper = vmin
    lower, upper = java_threshold_levels(lower, upper, bit_depth)
    return AutoThresholdResult(float(lower), float(upper), vmin, vmax, t)


def set_auto_threshold(image, method: str | None = "Default", dark: bool = False,
                       stack_histogram: bool = False, no_reset: bool = False,
                       sixteen_bit: bool = False, use_roi: bool = True,
                       mode: str | None = None) -> tuple[float, float]:
    """``setAutoThreshold("method [dark] [16-bit] [stack] [no-reset]")``.

    ``image`` is an ImageDoc (current plane, its LUT, display range and area selection)
    or a 2-D array (normal LUT). ``method`` may be a full ImageJ option string. Returns the
    raw (lower, upper) levels as ``getThreshold`` reports them. For an ImageDoc and
    ``mode`` given ("red" | "bw" | "over_under" | "none") the threshold is also set on the
    image and the 16/32-bit display range updated like ImageJ."""
    if method is not None and " " in method.strip():
        opts = parse_method(method)
        method = opts["method"]
        dark = dark or opts["dark"]
        no_reset = no_reset or opts["no_reset"]
        sixteen_bit = sixteen_bit or opts["sixteen_bit"]
        stack_histogram = stack_histogram or opts["stack"]
        if mode is None and opts["mode"] != "red":
            mode = opts["mode"]
    if not hasattr(image, "pixels"):
        res = auto_threshold_plane(np.asarray(image), method, dark, False, None, no_reset,
                                   sixteen_bit)
        return res.lower, res.upper
    if image.is_rgb:
        raise ValueError("Non-RGB image required")
    mask = None
    if use_roi and image.roi is not None and image.roi.is_area():
        mask = image.roi.get_mask((image.height, image.width))
    inverted = image.is_inverted_lut()
    if stack_histogram and image.stack_size > 1 and method is not None:
        planes = [image.plane_at(i) for i in range(image.stack_size)]
        res = auto_threshold_stack(planes, method, dark, inverted, sixteen_bit, mask)
    else:
        res = auto_threshold_plane(image.plane(), method, dark, inverted, image.display_range,
                                   no_reset, sixteen_bit, mask,
                                   over_under=(mode == "over_under"))
    if mode is not None:
        from rsstatchem.core.imaging.image import Threshold
        if image.bit_depth != 8:
            image.set_display_range(res.display_min, res.display_max)
        image.threshold = Threshold(res.lower, res.upper, mode)
    return res.lower, res.upper


# ===================================================================================
# Thresholder: Make Binary / Convert to Mask
# ===================================================================================
def _apply_levels_8bit(plane8: np.ndarray, lower: float, upper: float) -> np.ndarray:
    i = np.arange(256)
    lut = np.where((i >= lower) & (i <= upper), 255, 0).astype(np.uint8)
    return lut[plane8]


def _gray_luts(inverted: bool):
    from rsstatchem.core.imaging.lut import LUT
    lut = LUT.grays()
    return lut.inverted() if inverted else lut


def convert_to_mask(image, black_background: bool = False, method: str = "Default",
                    background: str = "Default", calculate: bool = False,
                    only_current: bool = False, list_thresholds: bool = False,
                    create_stack: bool = False, log: Callable[[str], None] | None = None,
                    weighted_rgb: bool = False):
    """Process > Binary > Convert to Mask / Make Binary (``Thresholder``, macro mode).

    Works in place on ``image`` (an ImageDoc) and returns it, or returns a new ImageDoc
    ("MASK_<title>") with ``create_stack`` on a stack. Uses the image's threshold if set
    (ignoring the invisible binary threshold of an earlier mask), otherwise ImageJ's legacy
    auto-threshold; stacks follow the "Convert Stack to Binary" options."""
    from rsstatchem.core.imaging.image import COLOR_256, COLOR_RGB, GRAY8, Threshold

    if method not in METHODS:
        raise ValueError(f'Invalid method ("{method}")')
    if image.stack_size > 1:
        if only_current:
            calculate = list_thresholds = create_stack = False
            if image.bit_depth != 8:
                raise ValueError("8-bit stack required to process a single slice.")
        if create_stack:
            dup = image.duplicate("MASK_" + image.title)
            dup.roi = image.roi
            image = dup
        if calculate:
            _convert_stack_local(image, black_background, method, background,
                                 list_thresholds, log)
        else:
            _apply_threshold(image, black_background, only_current, weighted_rgb)
        return image
    _apply_threshold(image, black_background, False, weighted_rgb)
    return image


def _current_threshold(image):
    th = image.threshold
    if th is not None and th.lower == th.upper and th.mode == "none":
        return None          # resetBinaryThreshold: remove an invisible mask threshold
    return th


def _apply_threshold(image, black_background: bool, one_slice: bool, weighted_rgb: bool):
    from rsstatchem.core.imaging.image import COLOR_256, COLOR_RGB, GRAY8, GRAY16, GRAY32, \
        Threshold

    image.roi = None
    th = _current_threshold(image)
    if image.type in (GRAY16, GRAY32):
        _apply_short_or_float(image, th, black_background)
        return
    rgb_mask_levels = None
    if image.type == COLOR_RGB:
        planes8 = rgb_to_byte(image.pixels[..., :], weighted_rgb)
        image.replace_pixels(planes8, GRAY8)
        image.luts = [_gray_luts(False) for _ in range(image.n_channels)]
    elif image.type == COLOR_256:
        lut = image.luts[0].as_array()
        rgb = lut[image.pixels]
        planes8 = rgb_to_byte(rgb, weighted_rgb)
        image.replace_pixels(planes8, GRAY8)
        image.luts = [_gray_luts(False) for _ in range(image.n_channels)]
    if th is None:
        p8 = image.plane()
        hist = byte_histogram(p8)
        nz = np.nonzero(hist)[0]
        smin, smax = (int(nz[0]), int(nz[-1])) if nz.size else (0, 0)
        lower, upper = legacy_threshold_8bit(hist, image.is_inverted_lut(), smin, smax)
    else:
        lower, upper = th.lower, th.upper
    if rgb_mask_levels is not None:
        lower, upper = rgb_mask_levels
    # a colour LUT is replaced by the default grey LUT
    lut = image.luts[image.position[0]]
    if not (lut.is_grayscale()):
        image.luts[image.position[0]] = _gray_luts(False)
    if image.stack_size > 1 and not one_slice:
        image.pixels[...] = _apply_levels_8bit(image.pixels, lower, upper)
    else:
        image.set_plane(_apply_levels_8bit(image.plane(), lower, upper))
    if not one_slice:
        inverted = image.is_inverted_lut()
        if (inverted and black_background) or (not inverted and not black_background):
            c = image.position[0]
            image.luts[c] = image.luts[c].inverted()
    image.display_ranges = [(0.0, 255.0)] * image.n_channels
    image.threshold = Threshold(255.0, 255.0, "none")


def _apply_short_or_float(image, th, black_background: bool):
    from rsstatchem.core.imaging.image import GRAY8, Threshold

    if th is None:
        vmin, vmax = image.display_range
        p8 = to_byte_scaled(image.plane(), vmin, vmax)
        hist = byte_histogram(p8)
        nz = np.nonzero(hist)[0]
        smin, smax = (int(nz[0]), int(nz[-1])) if nz.size else (0, 0)
        lo, hi = legacy_threshold_8bit(hist, image.is_inverted_lut(), smin, smax)
        t1 = np.float32(vmin + (vmax - vmin) * (lo / 255.0))
        t2 = np.float32(vmin + (vmax - vmin) * (hi / 255.0))
    else:
        t1, t2 = np.float32(th.lower), np.float32(th.upper)
    values = image.pixels.astype(np.float32)
    mask = ((values >= t1) & (values <= t2)).astype(np.uint8) * 255
    image.replace_pixels(mask, GRAY8)
    image.luts = [_gray_luts(not black_background) for _ in range(image.n_channels)]
    if image.n_channels > 1:
        image.composite_mode = "grayscale"
    image.display_ranges = [(0.0, 255.0)] * image.n_channels
    image.threshold = Threshold(255.0, 255.0, "none")


def _convert_stack_local(image, black_background, method, background, list_thresholds, log):
    from rsstatchem.core.imaging.image import GRAY8, Threshold

    n = image.stack_size
    bit_depth = image.bit_depth
    inverted = image.is_inverted_lut()
    mins, maxs = [0.0] * n, [0.0] * n
    planes8 = []
    for i in range(n):
        p = image.plane_at(i)
        if bit_depth != 8:
            lo, hi = float_min_max(p)
            mins[i], maxs[i] = lo, hi
            planes8.append(to_byte_scaled(p, lo, hi))
        else:
            planes8.append(p.copy())
    if list_thresholds and log:
        log(f"Thresholding method: {method}")
    out = []
    for i, p8 in enumerate(planes8):
        hist = byte_histogram(p8)
        if method == "Default" and background == "Default":
            nz = np.nonzero(hist)[0]
            smin, smax = (int(nz[0]), int(nz[-1])) if nz.size else (0, 0)
            lower, upper = legacy_threshold_8bit(hist, inverted, smin, smax)
        else:
            # ip.setAutoThreshold(method, dark, NO_LUT_UPDATE) on a fresh processor: its
            # private 'addOne' flag is false (only the String form sets it), so no +1 and
            # no bilevel adjustment, exactly as ImageJ 1.54
            t = auto_threshold(hist, method, subtract_one=False)
            lower, upper = _levels(t, background != "Light", inverted, False, 255.0)
            if lower > 255:
                lower = 255.0
        if list_thresholds and log:
            t1, t2 = lower, upper
            if bit_depth != 8:
                t1 = mins[i] + (t1 / 255.0) * (maxs[i] - mins[i])
                t2 = mins[i] + (t2 / 255.0) * (maxs[i] - mins[i])
            digits = 2 if bit_depth == 32 else 0
            log(f"  {i + 1}: {t1:.{digits}f}-{t2:.{digits}f}")
        out.append(_apply_levels_8bit(p8, lower, upper))
    pixels = np.zeros(image.pixels.shape[:5], np.uint8)
    for i, m in enumerate(out):
        c, z, t = image.position_of(i)
        pixels[t, z, c] = m
    image.replace_pixels(pixels, GRAY8)
    image.luts = [_gray_luts(not black_background) for _ in range(image.n_channels)]
    if image.n_channels > 1:
        image.composite_mode = "grayscale"
    image.roi = None
    image.threshold = Threshold(255.0, 255.0, "none")


def make_binary(image, black_background: bool = False, **kw):
    """Process > Binary > Make Binary (macro mode: same result as Convert to Mask)."""
    return convert_to_mask(image, black_background, **kw)


def threshold_mask(plane: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Bool mask of lower <= v <= upper (float32 comparison as ImageJ's createMask)."""
    a = np.asarray(plane)
    if a.dtype == np.float32:
        return (a >= np.float32(lower)) & (a <= np.float32(upper))
    return (a >= lower) & (a <= upper)


# ===================================================================================
# colour thresholding (ColorThresholder)
# ===================================================================================
def rgb_to_hsb_bytes(rgb: np.ndarray):
    """``ColorProcessor.getHSB(byte[]...)``: java.awt.Color.RGBtoHSB in float32, then
    (byte)(int)(h * 255.0)."""
    f32 = np.float32
    r = rgb[..., 0].astype(np.int32)
    g = rgb[..., 1].astype(np.int32)
    b = rgb[..., 2].astype(np.int32)
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    bri = cmax.astype(f32) / f32(255.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        sat = np.where(cmax != 0, (cmax - cmin).astype(f32) / cmax.astype(f32), f32(0))
        span = (cmax - cmin).astype(f32)
        redc = (cmax - r).astype(f32) / span
        greenc = (cmax - g).astype(f32) / span
        bluec = (cmax - b).astype(f32) / span
        hue = np.where(r == cmax, bluec - greenc,
                       np.where(g == cmax, f32(2.0) + redc - bluec,
                                f32(4.0) + greenc - redc)).astype(f32)
        hue = (hue / f32(6.0)).astype(f32)
        hue = np.where(hue < 0, hue + f32(1.0), hue).astype(f32)
    hue = np.where(sat == 0, f32(0), hue).astype(f32)
    def to_b(v):
        return (v.astype(np.float64) * 255.0).astype(np.int64).astype(np.uint8)
    return to_b(hue), to_b(sat.astype(f32)), to_b(bri)


_LAB_LIN = None


def _srgb_linear_table() -> np.ndarray:
    global _LAB_LIN
    if _LAB_LIN is None:
        t = []
        for v in range(256):
            c = v / 255.0
            c = c / 12.92 if c <= 0.04045 else math.pow((c + 0.055) / 1.055, 2.4)
            t.append(c * 100.0)
        _LAB_LIN = np.array(t, dtype=np.float64)
    return _LAB_LIN


def rgb_to_lab_float(rgb: np.ndarray):
    """``ColorSpaceConverter.RGBtoLAB`` (D65, sRGB): L*, a*, b* as float64 arrays."""
    lin = _srgb_linear_table()
    r = lin[rgb[..., 0]]
    g = lin[rgb[..., 1]]
    b = lin[rgb[..., 2]]
    X = (r * 0.4124) + (g * 0.3576) + (b * 0.1805)
    Y = (r * 0.2126) + (g * 0.7152) + (b * 0.0722)
    Z = (r * 0.0193) + (g * 0.1192) + (b * 0.9505)

    def f(t):
        return np.where(t > 0.008856, np.power(t, 1.0 / 3.0), (7.787 * t) + (16.0 / 116.0))
    x = f(X / 95.0429)
    y = f(Y / 100.0)
    z = f(Z / 108.8900)
    return (116.0 * y) - 16.0, 500.0 * (x - y), 200.0 * (y - z)


def rgb_to_lab_bytes(rgb: np.ndarray):
    """``ColorThresholder.getLab``: L*2.55 truncated, a/b as floor(1.0625*v + 128 + 0.5),
    clamped to 0..255."""
    L, a, b = rgb_to_lab_float(rgb)
    L1 = np.trunc(L * 2.55)
    a1 = np.floor((1.0625 * a + 128) + 0.5)
    b1 = np.floor((1.0625 * b + 128) + 0.5)
    return tuple(np.clip(v, 0, 255).astype(np.uint8) for v in (L1, a1, b1))


def rgb_to_yuv_bytes(rgb: np.ndarray):
    """``ColorThresholder.getYUV`` (Y, U, V bytes; U/V wrap like Java's (byte) cast)."""
    r = rgb[..., 0].astype(np.float64)
    g = rgb[..., 1].astype(np.float64)
    b = rgb[..., 2].astype(np.float64)
    yf = 0.299 * r + 0.587 * g + 0.114 * b
    Y = np.floor(yf + 0.5).astype(np.int64)
    U = 128 + np.floor((0.493 * (b - yf)) + 0.5).astype(np.int64)
    V = 128 + np.floor((0.877 * (r - yf)) + 0.5).astype(np.int64)
    return tuple((v & 0xFF).astype(np.uint8) for v in (Y, U, V))


def color_channels(rgb: np.ndarray, space: str = "HSB"):
    """The three 8-bit channels the Color Threshold window filters."""
    rgb = np.asarray(rgb, np.uint8)
    if space == "HSB":
        return rgb_to_hsb_bytes(rgb)
    if space == "RGB":
        return rgb[..., 0].copy(), rgb[..., 1].copy(), rgb[..., 2].copy()
    if space == "Lab":
        return rgb_to_lab_bytes(rgb)
    if space == "YUV":
        return rgb_to_yuv_bytes(rgb)
    raise ValueError(f"unknown colour space {space!r}")


def color_threshold(rgb: np.ndarray | Sequence[np.ndarray], space: str = "HSB",
                    ranges: Sequence[tuple] = ((0, 255, True),) * 3) -> np.ndarray:
    """Bool mask of the Color Threshold filter (``ColorThresholder.apply``): channel k keeps
    pixels with min <= v <= max when its filter is "pass", outside when "stop"; the mask
    is the AND over the three channels. ``rgb`` is an (H, W, 3) RGB plane or the three
    precomputed channels; ``ranges`` = [(min, max, pass), ...] (pass may be "pass"/"stop")."""
    chans = color_channels(rgb, space) if not isinstance(rgb, (tuple, list)) else rgb
    mask = None
    for ch, (lo, hi, passing) in zip(chans, ranges):
        if isinstance(passing, str):
            passing = passing.lower() == "pass"
        inside = (ch >= lo) & (ch <= hi)
        m = inside if passing else ~inside
        mask = m if mask is None else (mask & m)
    return mask


def color_auto_ranges(chans: Sequence[np.ndarray], space: str, method: str = "Default",
                      dark: bool = False) -> list[tuple[int, int]]:
    """``ColorThresholder.autoSetThreshold``: auto-threshold the brightness (HSB), every
    channel (RGB) or the first channel (Lab, YUV); others keep 0..255."""
    ranges = [(0, 255), (0, 255), (0, 255)]
    which = {"HSB": (2,), "RGB": (0, 1, 2), "Lab": (0,), "YUV": (0,)}[space]
    for k in which:
        hist = np.bincount(np.asarray(chans[k]).ravel(), minlength=256)[:256]
        t = auto_threshold(hist, method)
        ranges[k] = (t + 1, 255) if dark else (0, t)
    return ranges


def sample_ranges(chans: Sequence[np.ndarray], mask: np.ndarray, space: str):
    """``ColorThresholder.sample``: ranges (and pass flags) from the pixels in ``mask``.
    For HSB the hue filter becomes "stop" over the largest gap when that is narrower."""
    vals = [np.asarray(c)[mask].astype(int) for c in chans]
    if vals[0].size == 0:
        return None
    out = [[int(v.min()), int(v.max()), True] for v in vals]
    if space == "HSB":
        bins = np.zeros(256, int)
        bins[np.unique(vals[0])] = 1
        gap = maxgap = 0
        maxgapst = maxgapen = -1
        gapst = 0
        if bins[0] == 0:
            gapst, gap = 0, 1
        for i in range(1, 256):
            if bins[i] == 0:
                if bins[i - 1] > 0:
                    gap, gapst = 1, i
                else:
                    gap += 1
                if gap > maxgap:
                    maxgap, maxgapst, maxgapen = gap, gapst, i
        lo = int(np.nonzero(bins)[0][0])
        hi = int(np.nonzero(bins)[0][-1])
        if (hi - lo) < maxgap:
            out[0] = [lo, hi, True]
        else:
            out[0] = [maxgapst, maxgapen, False]
    return [tuple(o) for o in out]


def apply_color_mode(rgb: np.ndarray, mask: np.ndarray, mode: str = "Red",
                     black_background: bool = False) -> np.ndarray:
    """The "Filtered" image: thresholded pixels painted Red/White/Black, or B&W (mask in
    the foreground colour on the background colour)."""
    out = np.array(rgb, np.uint8, copy=True)
    if mode == "B&W":
        fg = 255 if black_background else 0
        out[...] = 255 - fg
        out[mask] = fg
        return out
    color = {"Red": (255, 0, 0), "White": (255, 255, 255), "Black": (0, 0, 0)}[mode]
    out[mask] = color
    return out


def mask_to_roi(mask: np.ndarray):
    """An exact area ROI for a bool mask: a CompositeRoi of rectangles (row runs merged
    with identical runs on following rows), or a single RectRoi; None if empty."""
    from rsstatchem.core.imaging.roi.shapes import CompositeRoi, RectRoi

    mask = np.asarray(mask, bool)
    h, w = mask.shape
    open_runs: dict[tuple[int, int], int] = {}    # (x0, x1) -> start row
    rects: list[tuple[int, int, int, int]] = []
    for y in range(h + 1):
        runs = set()
        if y < h:
            row = mask[y]
            if row.any():
                d = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
                starts = np.nonzero(d == 1)[0]
                ends = np.nonzero(d == -1)[0]
                runs = set(zip(starts.tolist(), ends.tolist()))
        for run in list(open_runs):
            if run not in runs:
                y0 = open_runs.pop(run)
                rects.append((run[0], y0, run[1] - run[0], y - y0))
        for run in runs:
            if run not in open_runs:
                open_runs[run] = y
    if not rects:
        return None
    if len(rects) == 1:
        x, y, rw, rh = rects[0]
        return RectRoi(x, y, rw, rh)
    rects.sort(key=lambda r: (r[1], r[0]))
    return CompositeRoi(parts=[("or", RectRoi(x, y, rw, rh)) for x, y, rw, rh in rects])


# ===================================================================================
# Fiji-style Auto Threshold helpers
# ===================================================================================
def fiji_histogram(plane: np.ndarray, ignore_black: bool = False, ignore_white: bool = False,
                   sixteen_bit: bool = True, mask: np.ndarray | None = None):
    """Histogram used by the Auto Threshold command: full 8-bit, full 16-bit (offset to the
    data minimum) or 256 bins for 32-bit; extreme bins optionally cleared. Returns
    (histogram, offset, bin_width)."""
    a = np.asarray(plane)
    if a.dtype == np.uint8:
        hist = byte_histogram(a, mask)
        off, width = 0.0, 1.0
    elif a.dtype == np.uint16 and sixteen_bit:
        h = histogram16(a, mask)
        nz = np.nonzero(h)[0]
        lo, hi = (int(nz[0]), int(nz[-1])) if nz.size else (0, 0)
        hist = h[lo:hi + 1].copy()
        off, width = float(lo), 1.0
    else:
        lo, hi = float_min_max(a)
        hist = byte_histogram(to_byte_scaled(a, lo, hi), mask)
        off, width = lo, ((hi - lo) / 255.0 if hi > lo else 1.0)
    hist = np.array(hist, np.int64)
    integer = a.dtype == np.uint8 or (a.dtype == np.uint16 and sixteen_bit)
    if ignore_black and len(hist) and (not integer or off == 0):
        hist[0] = 0                                   # pixel value 0 (or lowest bin)
    white = 255 if a.dtype == np.uint8 else 65535
    if ignore_white and len(hist) and (not integer or off + len(hist) - 1 == white):
        hist[-1] = 0                                  # pixel value 255/65535 (highest bin)
    return hist, off, width


def fiji_threshold_level(hist: np.ndarray, method: str) -> int:
    """Threshold bin (pixels > level are objects with "white objects") using the
    AutoThresholder port; ImageJ's "Default" here is the IJ_IsoData variant used by
    Fiji's Auto Threshold."""
    m = "IJ_IsoData" if method == "Default" else method
    return auto_threshold(hist, m)
