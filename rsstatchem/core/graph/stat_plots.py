"""Statistical plot types computed from raw data at render time (Origin's Statistical graphs).

The graph stores only the raw columns; violins, Q-Q and probability plots, kernel densities,
ECDFs, Pareto charts, mean ± SD/SE/CI bars, Bland-Altman and Shewhart control charts are
computed when the graph is drawn, so they stay editable (Plot Details) and follow the
worksheet when its data change. The numeric helpers (plotting positions, Q-Q, KDE, control
limits, limits of agreement) are plain functions so tests can check them against scipy and
hand calculations. No Qt here.
"""
from __future__ import annotations

import math

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

# ====================================================================== numeric helpers
POSITIONS = {"filliben": "Filliben (scipy probplot)", "blom": "Blom (i−3/8)/(n+1/4)",
             "benard": "Benard median rank (i−0.3)/(n+0.4)", "hazen": "Hazen (i−1/2)/n",
             "vdw": "Van der Waerden i/(n+1)", "km": "Kaplan-Meier i/n"}


def plotting_positions(n: int, method: str = "filliben") -> np.ndarray:
    """Cumulative probabilities (0–1) of the i-th of n ordered values."""
    i = np.arange(1, n + 1, dtype=float)
    if n <= 0:
        return np.array([])
    if method == "filliben":      # scipy.stats.probplot's order-statistic medians
        m = (i - 0.3175) / (n + 0.365)
        m[-1] = 0.5 ** (1.0 / n)
        m[0] = 1.0 - m[-1]
        return m
    if method == "blom":
        return (i - 0.375) / (n + 0.25)
    if method == "benard":
        return (i - 0.3) / (n + 0.4)
    if method == "hazen":
        return (i - 0.5) / n
    if method == "vdw":
        return i / (n + 1.0)
    if method == "km":
        out = i / n
        out[-1] = (n - 0.5) / n   # keep the last point off 100 %
        return out
    raise ValueError(f"unknown plotting position {method!r}")


DISTS = {"norm": "Normal", "lognorm": "Lognormal", "expon": "Exponential",
         "weibull_min": "Weibull", "gamma": "Gamma", "logistic": "Logistic",
         "uniform": "Uniform"}


def _dist(name: str, values: np.ndarray):
    """Frozen scipy distribution for theoretical quantiles (shape fitted when needed)."""
    if name == "norm":
        return stats.norm()
    if name == "lognorm":
        s, _, _ = stats.lognorm.fit(values[values > 0], floc=0)
        return stats.lognorm(s)
    if name == "expon":
        return stats.expon()
    if name == "weibull_min":
        c, _, _ = stats.weibull_min.fit(values[values > 0], floc=0)
        return stats.weibull_min(c)
    if name == "gamma":
        a, _, _ = stats.gamma.fit(values[values > 0], floc=0)
        return stats.gamma(a)
    if name == "logistic":
        return stats.logistic()
    if name == "uniform":
        return stats.uniform()
    raise ValueError(f"unknown distribution {name!r}")


def qq_data(values, dist: str = "norm", method: str = "filliben"):
    """Q-Q plot: (theoretical quantiles, ordered data, (slope, intercept, r)). With the
    defaults this equals ``scipy.stats.probplot(values, fit=True)``."""
    v = np.sort(np.asarray(values, float)[np.isfinite(np.asarray(values, float))])
    if len(v) < 2:
        raise ValueError("a Q-Q plot needs at least 2 values")
    osm = _dist(dist, v).ppf(plotting_positions(len(v), method))
    slope, intercept, r, _, _ = stats.linregress(osm, v)
    return osm, v, (float(slope), float(intercept), float(r))


def probability_data(values, method: str = "filliben"):
    """Probability plot: (ordered data, cumulative percent)."""
    v = np.sort(np.asarray(values, float)[np.isfinite(np.asarray(values, float))])
    return v, plotting_positions(len(v), method) * 100.0


def kde_curve(values, bw="scott", n: int = 256, cut: float = 3.0, grid=None):
    """Gaussian kernel density (scipy.stats.gaussian_kde): (x, density)."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) < 2 or np.std(v) == 0:
        raise ValueError("a kernel density needs at least 2 different values")
    kde = stats.gaussian_kde(v, bw_method=_bw(bw))
    if grid is None:
        sd = kde.factor * v.std(ddof=1)
        grid = np.linspace(v.min() - cut * sd, v.max() + cut * sd, n)
    return np.asarray(grid, float), kde(grid)


def _bw(bw):
    if bw in (None, "", "scott"):
        return "scott"
    if bw == "silverman":
        return "silverman"
    try:
        return float(bw)
    except (TypeError, ValueError):
        return "scott"


def kde2d(x, y, bw="scott", n: int = 80):
    """2D kernel density on an n×n grid: (X, Y, Z)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3:
        raise ValueError("a 2D kernel density needs at least 3 points")
    kde = stats.gaussian_kde(np.vstack([x, y]), bw_method=_bw(bw))
    px, py = 0.15 * np.ptp(x) or 1.0, 0.15 * np.ptp(y) or 1.0
    X, Y = np.meshgrid(np.linspace(x.min() - px, x.max() + px, n),
                       np.linspace(y.min() - py, y.max() + py, n))
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    return X, Y, Z


def ecdf(values):
    v = np.sort(np.asarray(values, float)[np.isfinite(np.asarray(values, float))])
    return v, np.arange(1, len(v) + 1) / len(v)


def pareto_data(labels: list, values) -> tuple[list, np.ndarray, np.ndarray]:
    """Sort categories by value (descending): (labels, values, cumulative %)."""
    vals = np.asarray(values, float)
    order = np.argsort(-vals, kind="stable")
    vals = vals[order]
    total = vals.sum()
    cum = np.cumsum(vals) / total * 100 if total else np.zeros_like(vals)
    return [labels[k] for k in order], vals, cum


def mean_interval(values, stat: str = "sd") -> tuple[float, float, int]:
    """(mean, half-width, n) for mean ± SD, SE or 95 % CI."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    n = len(v)
    if not n:
        raise ValueError("no valid data")
    mean = float(v.mean())
    sd = float(v.std(ddof=1)) if n > 1 else 0.0
    if stat == "se":
        return mean, sd / math.sqrt(n), n
    if stat == "ci":
        return mean, (stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n) if n > 1 else 0.0), n
    return mean, sd, n


def bland_altman(a, b, k: float = 1.96) -> dict:
    """Bland & Altman (1986): means, differences (a − b), bias and limits of agreement
    bias ± k·SD of the differences."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 2:
        raise ValueError("Bland-Altman needs at least 2 pairs")
    mean, diff = (a + b) / 2, a - b
    bias, sd = float(diff.mean()), float(diff.std(ddof=1))
    return {"mean": mean, "diff": diff, "bias": bias, "sd": sd,
            "lower": bias - k * sd, "upper": bias + k * sd, "n": len(a)}


# Shewhart constants for subgroup size n (ASTM E2587 / Montgomery Appendix VI)
_A2 = {2: 1.880, 3: 1.023, 4: 0.729, 5: 0.577, 6: 0.483, 7: 0.419, 8: 0.373, 9: 0.337,
       10: 0.308, 11: 0.285, 12: 0.266, 13: 0.249, 14: 0.235, 15: 0.223, 16: 0.212,
       17: 0.203, 18: 0.194, 19: 0.187, 20: 0.180, 21: 0.173, 22: 0.167, 23: 0.162,
       24: 0.157, 25: 0.153}
_D3 = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0.076, 8: 0.136, 9: 0.184, 10: 0.223, 11: 0.256,
       12: 0.283, 13: 0.307, 14: 0.328, 15: 0.347, 16: 0.363, 17: 0.378, 18: 0.391,
       19: 0.403, 20: 0.415, 21: 0.425, 22: 0.434, 23: 0.443, 24: 0.451, 25: 0.459}
_D4 = {2: 3.267, 3: 2.574, 4: 2.282, 5: 2.114, 6: 2.004, 7: 1.924, 8: 1.864, 9: 1.816,
       10: 1.777, 11: 1.744, 12: 1.717, 13: 1.693, 14: 1.672, 15: 1.653, 16: 1.637,
       17: 1.622, 18: 1.608, 19: 1.597, 20: 1.585, 21: 1.575, 22: 1.566, 23: 1.557,
       24: 1.548, 25: 1.541}


def _c4(n: int) -> float:
    return math.sqrt(2.0 / (n - 1)) * math.exp(math.lgamma(n / 2) - math.lgamma((n - 1) / 2))


def control_limits(values, size: int = 5, chart: str = "xbar") -> dict:
    """Shewhart chart statistics: points, center line, UCL, LCL.

    chart: xbar / r / s (subgroups of ``size`` consecutive values), i (individuals) and
    mr (moving range of 2) — the X-bar R, X-bar S and I-MR charts of Origin's QC tool."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if chart in ("i", "mr"):
        if len(v) < 3:
            raise ValueError("an I-MR chart needs at least 3 values")
        mr = np.abs(np.diff(v))
        mrbar = float(mr.mean())
        if chart == "i":
            c = float(v.mean())
            return {"points": v, "cl": c, "ucl": c + 2.66 * mrbar, "lcl": c - 2.66 * mrbar,
                    "x": np.arange(1, len(v) + 1, dtype=float)}
        return {"points": mr, "cl": mrbar, "ucl": 3.267 * mrbar, "lcl": 0.0,
                "x": np.arange(2, len(v) + 1, dtype=float)}
    size = int(size)
    if not 2 <= size <= 25:
        raise ValueError("subgroup size must be 2–25")
    k = len(v) // size
    if k < 2:
        raise ValueError("a control chart needs at least 2 complete subgroups")
    groups = v[:k * size].reshape(k, size)
    xbar = groups.mean(axis=1)
    rng = np.ptp(groups, axis=1)
    sds = groups.std(axis=1, ddof=1)
    x = np.arange(1, k + 1, dtype=float)
    xbb, rbar, sbar = float(xbar.mean()), float(rng.mean()), float(sds.mean())
    if chart == "xbar":
        return {"points": xbar, "cl": xbb, "ucl": xbb + _A2[size] * rbar,
                "lcl": xbb - _A2[size] * rbar, "x": x}
    if chart == "r":
        return {"points": rng, "cl": rbar, "ucl": _D4[size] * rbar,
                "lcl": _D3[size] * rbar, "x": x}
    if chart == "s":
        c4 = _c4(size)
        b = 3 * math.sqrt(1 - c4 ** 2) / c4
        return {"points": sds, "cl": sbar, "ucl": sbar * (1 + b),
                "lcl": max(0.0, sbar * (1 - b)), "x": x}
    raise ValueError(f"unknown control chart {chart!r}")


# ====================================================================== drawing
def _R():
    import sys
    return sys.modules["rsstatchem.core.graph.render"]


def _values(ctx, key, p=None):
    v = ctx.parr(p, key) if p is not None else ctx.arr(key)
    if v is None or v.dtype == object:
        raise ValueError("no numeric data")
    return v[np.isfinite(v)]


def _box_position(p, state) -> float:
    boxes = state["boxes"]
    return float(p.options.get("position", boxes.index(p) + 1))


def _box_ticks(state):
    boxes = state["boxes"]
    if any("group_label" in q.options for q in boxes):
        seen = {}
        for q in boxes:
            if "group_label" in q.options:
                seen.setdefault(float(q.options["group_pos"]), q.options["group_label"])
        return list(seen), list(seen.values())
    labels = [(q.options.get("position", k + 1), q.name) for k, q in enumerate(boxes)]
    return [float(a) for a, _ in labels], [b for _, b in labels]


def draw_violin(ctx, li, layer, ax, p, i, state):
    v = _values(ctx, p.y, p)
    if len(v) < 2:
        raise ValueError("a violin needs at least 2 values")
    pos = _box_position(p, state)
    w = float(p.options.get("width", 0.7)) / 2
    face = ctx.fill_color(p, i) if p.fill.visible else "none"
    edge = p.fill.edge_color
    side = p.options.get("side", "both")
    if np.std(v) > 0:
        ys, dens = kde_curve(v, p.options.get("bw", "scott"), n=200, cut=0 if
                             p.options.get("trim", True) else 2)
    else:
        ys, dens = np.array([v[0], v[0]]), np.array([1.0, 1.0])
    half = dens / dens.max() * w
    left = pos - half if side in ("both", "left") else np.full_like(half, pos)
    right = pos + half if side in ("both", "right") else np.full_like(half, pos)
    arts = [ax.fill_betweenx(ys, left, right, facecolor=face, edgecolor=edge,
                             linewidth=p.fill.edge_width, alpha=p.fill.alpha,
                             hatch=ctx.hatch(p, i), zorder=2)]
    inner = p.options.get("inner", "box")
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    if inner in ("box", "quartile") and _dark(face) and _dark(edge):
        edge = "#ffffff"   # keep the inner box visible on a dark violin
    if inner == "box":
        iqr = q3 - q1
        lo = v[v >= q1 - 1.5 * iqr].min()
        hi = v[v <= q3 + 1.5 * iqr].max()
        arts.append(ax.vlines(pos, lo, hi, color=edge, lw=p.fill.edge_width, zorder=3))
        arts.append(ax.vlines(pos, q1, q3, color=edge, lw=max(3.0, w * 14), zorder=3))
        arts += ax.plot([pos], [med], "o", color="white", markersize=3.5, zorder=4)
    elif inner == "quartile":
        for q, ls in ((q1, "--"), (med, "-"), (q3, "--")):
            d = np.interp(q, ys, half)
            a, b = (pos - d if side != "right" else pos), (pos + d if side != "left" else pos)
            arts += ax.plot([a, b], [q, q], color=edge, ls=ls, lw=p.fill.edge_width, zorder=3)
    if inner == "points" or p.options.get("points"):
        rng = np.random.default_rng(12345)
        off = rng.uniform(-0.25, 0.25, len(v)) * w
        if side == "left":
            off = np.abs(off) * 2
        elif side == "right":
            off = -np.abs(off) * 2
        arts.append(ax.scatter(pos + off, v, s=p.symbol.size ** 2 * 0.5,
                               color=ctx.line_color(p, i), alpha=0.7, lw=0, zorder=3))
    if p.options.get("show_mean"):
        arts += ax.plot([pos], [v.mean()], "s", mfc="white", mec=edge, ms=4, zorder=4)
    _R()._register(ctx, li, p, ax, arts, np.full(len(v), pos), v)
    return Patch(facecolor=face, edgecolor=edge), _box_ticks(state)


def _dark(color) -> bool:
    from matplotlib.colors import to_rgb
    try:
        r, g, b = to_rgb(color)
    except ValueError:
        return False
    return 0.299 * r + 0.587 * g + 0.114 * b < 0.35


def draw_meanbar(ctx, li, layer, ax, p, i, state):
    v = _values(ctx, p.y, p)
    mean, err, _ = mean_interval(v, p.options.get("stat", "sd"))
    pos = _box_position(p, state)
    w = float(p.options.get("width", 0.6))
    ecol = p.error.color if p.error.color != "auto" else "#000000"
    style = p.options.get("style", "column")
    if style == "column":
        bars = ax.bar([pos], [mean], width=w, yerr=[err], capsize=p.error.cap,
                      color=ctx.fill_color(p, i) if p.fill.visible else "none",
                      edgecolor=p.fill.edge_color, linewidth=p.fill.edge_width,
                      hatch=ctx.hatch(p, i), alpha=p.fill.alpha,
                      error_kw={"elinewidth": p.error.width, "capthick": p.error.width,
                                "ecolor": ecol})
        arts = list(bars.patches) + (list(bars.errorbar.lines[2]) if bars.errorbar else [])
        handle = Patch(facecolor=ctx.fill_color(p, i), edgecolor=p.fill.edge_color)
    else:   # interval plot: mean point with error bar
        col = ctx.line_color(p, i)
        eb = ax.errorbar([pos], [mean], yerr=[err], fmt=ctx.marker(p, i), color=col,
                         ms=p.symbol.size, capsize=p.error.cap, elinewidth=p.error.width,
                         ecolor=ecol if p.error.color != "auto" else col)
        arts = [eb.lines[0]] + list(eb.lines[1]) + list(eb.lines[2])
        handle = eb.lines[0]
    if p.options.get("points"):
        rng = np.random.default_rng(12345)
        arts.append(ax.scatter(pos + rng.uniform(-0.2, 0.2, len(v)) * w, v, s=9,
                               color="#444444", alpha=0.6, lw=0, zorder=3))
    _R()._register(ctx, li, p, ax, arts, [pos], [mean])
    return handle, _box_ticks(state)


def _ref_color(ctx, layer, p, i) -> str:
    """Reference line colour: red for a single plot, the plot's colour when several."""
    if p.line.color != "auto":
        return p.line.color
    same = [q for q in layer.plots if q.type == p.type and q.visible]
    return ctx.line_color(p, i) if len(same) > 1 else "#FF0000"


def draw_qq(ctx, li, layer, ax, p, i, state):
    v = _values(ctx, p.y, p)
    osm, osr, (slope, icpt, r) = qq_data(v, p.options.get("dist", "norm"),
                                         p.options.get("positions", "filliben"))
    col = ctx.line_color(p, i)
    kw = ctx.symbol_kw(p, i)
    (h,) = ax.plot(osm, osr, ls="none", **kw)
    arts = [h]
    if p.options.get("ref_line", True):
        if p.options.get("ref", "fit") == "quartile":   # through the quartiles
            q = np.percentile(v, [25, 75])
            t = _dist(p.options.get("dist", "norm"), v).ppf([0.25, 0.75])
            slope = (q[1] - q[0]) / (t[1] - t[0])
            icpt = q[0] - slope * t[0]
        xs = np.array([osm.min(), osm.max()])
        lc = _ref_color(ctx, layer, p, i)
        arts += ax.plot(xs, icpt + slope * xs, color=lc, lw=p.line.width,
                        ls=ctx.dash(p, i) if p.line.style != "auto" else "-")
    _R()._register(ctx, li, p, ax, arts, osm, osr)
    ctx.map.results[p.id] = {"slope": slope, "intercept": icpt, "r": r}
    return h, None


def draw_probability(ctx, li, layer, ax, p, i, state):
    v = _values(ctx, p.y, p)
    x, pct = probability_data(v, p.options.get("positions", "filliben"))
    (h,) = ax.plot(x, pct, ls="none", **ctx.symbol_kw(p, i))
    arts = [h]
    if p.options.get("ref_line", True) and len(v) > 1:
        mu, sd = float(v.mean()), float(v.std(ddof=1))
        xs = np.linspace(x.min(), x.max(), 100)
        lc = _ref_color(ctx, layer, p, i)
        arts += ax.plot(xs, stats.norm.cdf(xs, mu, sd) * 100, color=lc, lw=p.line.width)
    _R()._register(ctx, li, p, ax, arts, x, pct)
    return h, None


def draw_kde(ctx, li, layer, ax, p, i, state):
    v = _values(ctx, p.y, p)
    xs, d = kde_curve(v, p.options.get("bw", "scott"))
    if p.options.get("scale") == "count":
        d = d * len(v)
    col = ctx.line_color(p, i)
    (h,) = ax.plot(xs, d, color=col, lw=p.line.width, ls=ctx.dash(p, i), label=p.name)
    arts = [h]
    if p.options.get("fill", True):
        arts.append(ax.fill_between(xs, d, color=ctx.fill_color(p, i)
                                    if p.fill.color != "auto" else col,
                                    alpha=p.fill.alpha if p.fill.alpha < 1 else 0.25, lw=0))
    if p.options.get("rug"):
        arts.append(ax.vlines(v, 0, 0.04, transform=ax.get_xaxis_transform(), colors=col,
                              lw=0.7))
    _R()._register(ctx, li, p, ax, arts, xs, d)
    return h, None


def draw_kde2d(ctx, li, layer, ax, p, i, state):
    x, y = ctx.parr(p, p.x), ctx.parr(p, p.y)
    if x is None or y is None:
        raise ValueError("2D kernel density needs X and Y")
    X, Y, Z = kde2d(x, y, p.options.get("bw", "scott"), int(p.options.get("grid_size", 80)))
    cmap = p.options.get("cmap", layer.cmap)
    levels = int(p.options.get("levels", 10))
    arts = []
    cf = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, alpha=p.fill.alpha)
    arts.append(cf)
    if p.line.visible:
        arts.append(ax.contour(X, Y, Z, levels=cf.levels, colors="#333333",
                               linewidths=0.4))
    if p.options.get("show_points", True):
        m = np.isfinite(x) & np.isfinite(y)
        arts.append(ax.scatter(x[m], y[m], s=6, c="k", marker=".", zorder=3))
    _R()._register(ctx, li, p, ax, arts, X.ravel(), Y.ravel())
    if layer.colorbar and state["cbar"] is None:
        state["cbar"] = _R()._colorbar(ctx, li, layer, ax, cf,
                                       p.options.get("colorbar_title") or "Density")
    return None, None


def draw_ecdf(ctx, li, layer, ax, p, i, state):
    v = _values(ctx, p.y, p)
    x, f = ecdf(v)
    scale = 100.0 if p.options.get("percent") else 1.0
    col = ctx.line_color(p, i)
    xs = np.r_[x[0], x]
    ys = np.r_[0.0, f] * scale
    (h,) = ax.step(xs, ys, where="post", color=col, lw=p.line.width, ls=ctx.dash(p, i))
    arts = [h]
    if p.options.get("normal_ref"):
        g = np.linspace(x.min(), x.max(), 200)
        arts += ax.plot(g, stats.norm.cdf(g, v.mean(), v.std(ddof=1)) * scale, color="#888",
                        lw=1, ls="--")
    _R()._register(ctx, li, p, ax, arts, x, f * scale)
    return h, None


def _category_values(ctx, p):
    """(labels, values) of a Pareto chart: Y summed per X category, or the counts of the
    Y (or X) values when there is only one column (Origin's "raw data" Pareto)."""
    xc = ctx.col(p.x)
    y = ctx.arr(p.y)
    R = _R()
    if xc is not None and y is not None and not p.options.get("raw"):
        x = xc.array()
        n = min(len(x), len(y))
        cats: dict = {}
        for xv, yv in zip(x[:n], y[:n]):
            if xc.kind != "text" and not np.isfinite(xv):
                continue
            if np.isfinite(yv):
                cats[xv] = cats.get(xv, 0.0) + yv
        labels = [str(k) if xc.kind == "text" else xc.value_labels.get(k, R._num(k))
                  for k in cats]
        return labels, np.array(list(cats.values()), float)
    col = ctx.col(p.y) or xc
    if col is None:
        raise ValueError("Pareto needs a column")
    vals = col.array()
    cats = {}
    for v in vals:
        if col.kind != "text" and not np.isfinite(v):
            continue
        cats[v] = cats.get(v, 0) + 1
    labels = [str(k) if col.kind == "text" else col.value_labels.get(k, R._num(k))
              for k in cats]
    return labels, np.array(list(cats.values()), float)


def draw_pareto(ctx, li, layer, ax, p, i, state):
    labels, vals = _category_values(ctx, p)
    if not len(vals):
        raise ValueError("no data for the Pareto chart")
    labels, vals, cum = pareto_data(labels, vals)
    pos = np.arange(len(vals), dtype=float)
    bars = ax.bar(pos, vals, width=1 - layer.gap,
                  color=ctx.fill_color(p, i) if p.fill.visible else "none",
                  edgecolor=p.fill.edge_color, linewidth=p.fill.edge_width,
                  hatch=ctx.hatch(p, i), alpha=p.fill.alpha)
    arts = list(bars.patches)
    ax2 = state.get("ax2") or ax
    lc = p.line.color if p.line.color != "auto" else "#FF0000"
    arts += ax2.plot(pos, cum, color=lc, lw=p.line.width, marker="o", ms=4, mfc=lc, mec=lc)
    if ax2 is not ax:
        state["post"].append(lambda: ax2.set_ylim(0, 105))
    _R()._register(ctx, li, p, ax, arts, pos, vals)
    return bars, (list(pos), labels)


def draw_bland_altman(ctx, li, layer, ax, p, i, state):
    a, b = ctx.parr(p, p.x), ctx.parr(p, p.y)
    if a is None or b is None:
        raise ValueError("Bland-Altman needs two method columns (X and Y)")
    k = float(p.options.get("k", 1.96))
    ba = bland_altman(a, b, k)
    (h,) = ax.plot(ba["mean"], ba["diff"], ls="none", **ctx.symbol_kw(p, i))
    arts = [h]
    font = ctx.font(_R().Font(size=layer.x.label_font.size * 0.8, color="#333333"))
    for val, text, ls in ((ba["bias"], f"Mean {ba['bias']:.3g}", "-"),
                          (ba["upper"], f"+{k:g} SD {ba['upper']:.3g}", "--"),
                          (ba["lower"], f"−{k:g} SD {ba['lower']:.3g}", "--")):
        arts.append(ax.axhline(val, color="#555555" if ls == "--" else "#1f4fa0", lw=1.0,
                               ls=ls, zorder=1.5))
        t = ax.text(0.99, val, text, transform=ax.get_yaxis_transform(), ha="right",
                    va="bottom", **font)
        t.set_in_layout(False)
        arts.append(t)
    _R()._register(ctx, li, p, ax, arts, ba["mean"], ba["diff"])
    ctx.map.results[p.id] = {"bias": ba["bias"], "sd": ba["sd"], "lower": ba["lower"],
                        "upper": ba["upper"], "n": ba["n"]}
    return h, None


def draw_control(ctx, li, layer, ax, p, i, state):
    v = ctx.parr(p, p.y)
    if v is None:
        raise ValueError("no Y column")
    c = control_limits(v, int(p.options.get("size", 5)), p.options.get("chart", "xbar"))
    col = ctx.line_color(p, i)
    (h,) = ax.plot(c["x"], c["points"], color=col, lw=p.line.width, marker="o", ms=4,
                   mfc=col, mec=col)
    arts = [h]
    out = (c["points"] > c["ucl"]) | (c["points"] < c["lcl"])
    if out.any():
        arts.append(ax.scatter(c["x"][out], c["points"][out], s=40, color="#FF0000",
                               zorder=4))
    font = ctx.font(_R().Font(size=layer.x.label_font.size * 0.8, color="#333333"))
    for val, name, color in ((c["ucl"], "UCL", "#FF0000"), (c["cl"], "CL", "#008000"),
                             (c["lcl"], "LCL", "#FF0000")):
        arts.append(ax.axhline(val, color=color, lw=1.0, ls="--" if name != "CL" else "-"))
        t = ax.text(1.005, val, f"{name}={val:.4g}", transform=ax.get_yaxis_transform(),
                    ha="left", va="center", clip_on=False, **font)
        t.set_in_layout(True)
        arts.append(t)
    _R()._register(ctx, li, p, ax, arts, c["x"], c["points"])
    ctx.map.results[p.id] = {"cl": c["cl"], "ucl": c["ucl"], "lcl": c["lcl"]}
    return h, None


def draw_floating(ctx, li, layer, ax, p, i, state):
    R = _R()
    x, lo = R._xy(ctx, p)
    hi = ctx.arr(p.z)
    if hi is None:
        raise ValueError("floating columns need a second Y column (top)")
    n = min(len(x), len(lo), len(hi))
    x, lo, hi = x[:n], lo[:n], hi[:n]
    m = np.isfinite(x) & np.isfinite(lo) & np.isfinite(hi)
    spacing = float(np.min(np.diff(np.unique(x[m])))) if m.sum() > 1 else 1.0
    bars = ax.bar(x[m], (hi - lo)[m], bottom=lo[m], width=spacing * (1 - layer.gap),
                  color=ctx.fill_color(p, i) if p.fill.visible else "none",
                  edgecolor=p.fill.edge_color, linewidth=p.fill.edge_width,
                  hatch=ctx.hatch(p, i), alpha=p.fill.alpha)
    R._register(ctx, li, p, ax, list(bars.patches), x[m], hi[m])
    return bars, None


def draw_lollipop(ctx, li, layer, ax, p, i, state):
    R = _R()
    x, y = R._xy(ctx, p)
    m = np.isfinite(x) & np.isfinite(y)
    col = ctx.line_color(p, i)
    base = float(p.options.get("baseline", 0.0))
    stems = ax.vlines(x[m], base, y[m], colors=col, lw=p.line.width)
    kw = ctx.symbol_kw(p, i)
    if p.symbol.shape == "auto":
        kw["marker"] = "o"
    (h,) = ax.plot(x[m], y[m], ls="none", **kw)
    R._register(ctx, li, p, ax, [stems, h], x[m], y[m])
    R._labels(ctx, li, ax, p, x[m], y[m], m, i)
    return h, None


def draw_vector(ctx, li, layer, ax, p, i, state):
    """Vector XYAM (angle in degrees, magnitude) or XYXY (start and end points)."""
    R = _R()
    x, y = R._xy(ctx, p)
    n = len(x)
    col = ctx.line_color(p, i)
    if p.options.get("mode", "xyam") == "xyxy":
        x2, y2 = ctx.arr(p.options.get("x2_col")), ctx.arr(p.options.get("y2_col"))
        if x2 is None or y2 is None:
            raise ValueError("vector XYXY needs end X and end Y columns")
        u, v = x2[:n] - x, y2[:n] - y
    else:
        ang = ctx.arr(p.z)
        mag = ctx.arr(p.options.get("mag_col"))
        if ang is None:
            raise ValueError("vector XYAM needs an angle column")
        mag = np.ones(n) if mag is None else mag[:n]
        rad = np.deg2rad(ang[:n])
        scale = float(p.options.get("arrow_scale", 1.0))
        u, v = mag * np.cos(rad) * scale, mag * np.sin(rad) * scale
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(u) & np.isfinite(v)
    q = ax.quiver(x[m], y[m], u[m], v[m], angles="xy", scale_units="xy", scale=1,
                  color=col, width=0.004 * p.line.width, headwidth=4, headlength=5)
    ax.update_datalim(np.c_[np.r_[x[m], x[m] + u[m]], np.r_[y[m], y[m] + v[m]]])
    R._register(ctx, li, p, ax, q, x[m], y[m])
    return Line2D([], [], color=col, lw=p.line.width, marker=">"), None


CART = {"violin": draw_violin, "meanbar": draw_meanbar, "qq": draw_qq,
        "probability": draw_probability, "kde": draw_kde, "kde2d": draw_kde2d,
        "ecdf": draw_ecdf, "pareto": draw_pareto, "bland_altman": draw_bland_altman,
        "control": draw_control, "floating_column": draw_floating,
        "lollipop": draw_lollipop, "vector": draw_vector}


# ====================================================================== histogram curves
def dist_curve(values, name: str, xs):
    """Fitted density of ``values`` on ``xs`` for Histogram + Distribution."""
    v = np.asarray(values, float)
    if name == "normal":
        return stats.norm.pdf(xs, v.mean(), v.std(ddof=1))
    if name == "kde":
        return kde_curve(v, grid=xs)[1]
    pos = v[v > 0]
    if len(pos) < 2:
        raise ValueError(f"{name} needs positive data")
    if name == "lognormal":
        s, loc, scale = stats.lognorm.fit(pos, floc=0)
        return stats.lognorm.pdf(xs, s, loc, scale)
    if name == "exponential":
        loc, scale = stats.expon.fit(pos, floc=0)
        return stats.expon.pdf(xs, loc, scale)
    if name == "gamma":
        a, loc, scale = stats.gamma.fit(pos, floc=0)
        return stats.gamma.pdf(xs, a, loc, scale)
    if name == "weibull":
        c, loc, scale = stats.weibull_min.fit(pos, floc=0)
        return stats.weibull_min.pdf(xs, c, loc, scale)
    raise ValueError(f"unknown distribution {name!r}")
