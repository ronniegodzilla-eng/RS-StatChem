"""Analyze > Calibrate... (density calibration) and Analyze > Tools > Curve Fitting...

Function list and formulas of ImageJ's ``CurveFitter`` / ``Calibrator``: None, the 25
CurveFitter functions, "Pixel Inverter" (255 - v, 65535 - v for 16-bit) and "Uncalibrated OD"
(log10(255 / v), 8-bit only). Polynomials and the "linear regression" variants are fitted by
linear least squares, the other functions by nonlinear least squares (scipy), started from
ImageJ-like initial parameters. The fitted function becomes the image's value calibration
(``Calibration.function/coefficients``; zero clip for non-negative standards on 8-bit images,
as ImageJ). Evaluators for the codes that Phase 0's calibration module does not know are
registered with ``register_value_function``.

Formulas from ImageJ 1.54 ij/measure/CurveFitter.java and ij/plugin/filter/Calibrator.java
(public domain, NIH).
"""
from __future__ import annotations

import math

import numpy as np

from rsstatchem.core.imaging import calibration as C

(STRAIGHT_LINE, POLY2, POLY3, POLY4, EXPONENTIAL, POWER, LOG, RODBARD, GAMMA_VARIATE, LOG2,
 RODBARD2, EXP_WITH_OFFSET, GAUSSIAN, EXP_RECOVERY, INV_RODBARD, EXP_REGRESSION,
 POWER_REGRESSION, POLY5, POLY6, POLY7, POLY8, GAUSSIAN_NOOFFSET, EXP_RECOVERY_NOOFFSET,
 CHAPMAN, ERF) = range(25)

FIT_LIST = ("Straight Line", "2nd Degree Polynomial", "3rd Degree Polynomial",
            "4th Degree Polynomial", "Exponential", "Power", "Log", "Rodbard", "Gamma Variate",
            "y = a+b*ln(x-c)", "Rodbard (NIH Image)", "Exponential with Offset", "Gaussian",
            "Exponential Recovery", "Inverse Rodbard", "Exponential (linear regression)",
            "Power (linear regression)", "5th Degree Polynomial", "6th Degree Polynomial",
            "7th Degree Polynomial", "8th Degree Polynomial", "Gaussian (no offset)",
            "Exponential Recovery (no offset)", "Chapman-Richards", "Error Function")
FORMULAS = ("y = a+bx", "y = a+bx+cx^2", "y = a+bx+cx^2+dx^3", "y = a+bx+cx^2+dx^3+ex^4",
            "y = a*exp(bx)", "y = a*x^b", "y = a*ln(bx)", "y = d+(a-d)/(1+(x/c)^b)",
            "y = b*(x-a)^c*exp(-(x-a)/d)", "y = a+b*ln(x-c)", "x = d+(a-d)/(1+(y/c)^b) [y = c*((x-a)/(d-x))^(1/b)]",
            "y = a*exp(-bx) + c", "y = a + (b-a)*exp(-(x-c)*(x-c)/(2*d*d))",
            "y = a*(1-exp(-b*x)) + c", "y = c*((x-a)/(d-x))^(1/b)", "y = a*exp(bx)",
            "y = a*x^b", "y = a+bx+cx^2+dx^3+ex^4+fx^5", "y = a+bx+cx^2+dx^3+ex^4+fx^5+gx^6",
            "y = a+bx+cx^2+dx^3+ex^4+fx^5+gx^6+hx^7",
            "y = a+bx+cx^2+dx^3+ex^4+fx^5+gx^6+hx^7+ix^8", "y = a*exp(-(x-b)*(x-b)/(2*c*c)))",
            "y = a*(1-exp(-b*x))", "y = a*(1-exp(-b*x))^c", "y = a+b*erf((x-c)/d)")
N_PARAMS = (2, 3, 4, 5, 2, 2, 2, 4, 4, 3, 4, 3, 4, 3, 4, 2, 2, 6, 7, 8, 9, 3, 2, 3, 4)
NONE_LABEL, INVERTER, UNCALIBRATED_OD = "None", "Pixel Inverter", "Uncalibrated OD"


def function_list() -> list[str]:
    """The Calibrate dialog's Function choices (None, fits, "-", inverter, OD)."""
    return [NONE_LABEL, *FIT_LIST, "-", INVERTER, UNCALIBRATED_OD]


def f(fit_type: int, p, x):
    """``CurveFitter.f`` vectorised over x."""
    x = np.asarray(x, dtype=float)
    p = [float(v) for v in p]
    with np.errstate(all="ignore"):
        if fit_type in (STRAIGHT_LINE, POLY2, POLY3, POLY4, POLY5, POLY6, POLY7, POLY8):
            out = np.zeros_like(x)
            for k in reversed(p):
                out = out * x + k
            return out
        if fit_type in (EXPONENTIAL, EXP_REGRESSION):
            return p[0] * np.exp(p[1] * x)
        if fit_type == EXP_WITH_OFFSET:
            return p[0] * np.exp(-p[1] * x) + p[2]
        if fit_type == EXP_RECOVERY:
            return p[0] * (1 - np.exp(-p[1] * x)) + p[2]
        if fit_type == EXP_RECOVERY_NOOFFSET:
            return p[0] * (1 - np.exp(-p[1] * x))
        if fit_type == CHAPMAN:
            return p[0] * np.power(1 - np.exp(-p[1] * x), p[2])
        if fit_type == GAUSSIAN:
            return p[0] + (p[1] - p[0]) * np.exp(-(x - p[2]) ** 2 / (2.0 * p[3] * p[3]))
        if fit_type == GAUSSIAN_NOOFFSET:
            return p[0] * np.exp(-(x - p[1]) ** 2 / (2.0 * p[2] * p[2]))
        if fit_type in (POWER, POWER_REGRESSION):
            return p[0] * np.power(x, p[1])
        if fit_type == LOG:
            return np.where(x == 0.0, -1000 * p[0], p[0] * np.log(p[1] * x))
        if fit_type == RODBARD:
            return p[3] + (p[0] - p[3]) / (1.0 + np.power(x / p[2], p[1]))
        if fit_type == GAMMA_VARIATE:
            if p[1] <= 0 or p[2] <= 0 or p[3] <= 0:
                return np.full_like(x, np.nan)
            return np.where(p[0] >= x, 0.0, p[1] * np.power(np.maximum(x - p[0], 0), p[2]) *
                            np.exp(-(x - p[0]) / p[3]))
        if fit_type == LOG2:
            tmp = x - p[2]
            return np.where(tmp <= 0, np.nan, p[0] + p[1] * np.log(np.where(tmp > 0, tmp, 1)))
        if fit_type in (INV_RODBARD, RODBARD2):
            bad = (p[3] - x < 2 * 5e-324) | (x < p[0])
            y = np.power((x - p[0]) / (p[3] - x), 1.0 / p[1]) * p[2]
            return np.where(bad, np.nan if fit_type == INV_RODBARD else 0.0, y)
        if fit_type == ERF:
            from scipy.special import erf
            return p[0] + p[1] * erf((x - p[2]) / p[3])
    return np.zeros_like(x)


def _evaluator(code):
    n = N_PARAMS[code]

    def fn(x, c):
        c = tuple(c)
        out = f(code, c[:n], x)
        if len(c) > n and c[n] == 1.0:      # ImageJ's zeroClip, stored as an extra flag
            out = np.maximum(out, 0.0)
        return out
    return fn


def _register() -> None:
    for code in range(25):
        if code == STRAIGHT_LINE:
            continue
        C.register_value_function(code, _evaluator(code))


_register()


def _initial(fit_type, x, y):
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    xmean, ymean = float(np.mean(x)), float(np.mean(y))
    slope = (y[-1] - y[0]) / (x[-1] - x[0]) if x[-1] != x[0] else 1.0
    yfirst, ylast = float(y[0]), float(y[-1])
    if fit_type in (EXPONENTIAL, EXP_REGRESSION):
        return [max(abs(yfirst), 1e-6) * np.sign(ymean or 1), 0.01 * np.sign(slope or 1)]
    if fit_type in (POWER, POWER_REGRESSION):
        return [max(ymean, 1e-6), 1.0]
    if fit_type == LOG:
        return [ymean, 1.0 / max(xmean, 1e-6)]
    if fit_type in (RODBARD, RODBARD2, INV_RODBARD):
        return [yfirst, 1.0, xmean, ylast]
    if fit_type == GAMMA_VARIATE:
        return [xmin - 0.1 * abs(xmax - xmin) - 1e-3, ymax, 2.0, max((xmax - xmin) / 4, 1e-3)]
    if fit_type == LOG2:
        return [ymean, (ymax - ymin) / max(math.log(max(xmax - xmin, 1.0) + 1), 1e-6),
                min(xmin - 0.1 * (xmax - xmin) - 1e-3, xmin - 1e-3)]
    if fit_type == EXP_WITH_OFFSET:
        return [ymax - ymin, 1.0 / max(xmax - xmin, 1e-6), ymin]
    if fit_type in (EXP_RECOVERY, EXP_RECOVERY_NOOFFSET, CHAPMAN):
        p = [ymax - ymin, 1.0 / max(xmax - xmin, 1e-6), ymin]
        if fit_type == EXP_RECOVERY_NOOFFSET:
            return p[:2]
        if fit_type == CHAPMAN:
            return [ymax, 1.0 / max(xmax - xmin, 1e-6), 1.5]
        return p
    if fit_type == GAUSSIAN:
        return [ymin, ymax, float(x[np.argmax(y)]), max((xmax - xmin) / 4, 1e-6)]
    if fit_type == GAUSSIAN_NOOFFSET:
        return [ymax, float(x[np.argmax(y)]), max((xmax - xmin) / 4, 1e-6)]
    if fit_type == ERF:
        return [ymean, (ymax - ymin) / 2, xmean, max((xmax - xmin) / 4, 1e-6)]
    return [1.0] * N_PARAMS[fit_type]


def fit(fit_type: int, x, y) -> tuple[np.ndarray, float]:
    """Fit ``fit_type`` to (x, y); returns (parameters, R^2)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y) or len(x) == 0:
        raise ValueError("To create a calibration curve, the left column must\n"
                         "contain a list of measured mean pixel values and the\n"
                         "right column must contain the same number of calibration\n"
                         "standard values.")
    poly_deg = {STRAIGHT_LINE: 1, POLY2: 2, POLY3: 3, POLY4: 4, POLY5: 5, POLY6: 6, POLY7: 7,
                POLY8: 8}
    if fit_type in poly_deg:
        deg = min(poly_deg[fit_type], len(x) - 1)
        c = np.polynomial.polynomial.polyfit(x, y, deg)
        p = np.zeros(poly_deg[fit_type] + 1)
        p[:len(c)] = c
    elif fit_type == EXP_REGRESSION:
        b, a = np.polyfit(x, np.log(y), 1)
        p = np.array([math.exp(a), b])
    elif fit_type == POWER_REGRESSION:
        b, a = np.polyfit(np.log(x), np.log(y), 1)
        p = np.array([math.exp(a), b])
    else:
        from scipy.optimize import least_squares

        p0 = np.asarray(_initial(fit_type, x, y), dtype=float)

        def resid(pp):
            r = f(fit_type, pp, x) - y
            return np.where(np.isfinite(r), r, 1e10)
        best = None
        for scale in (1.0, 0.5, 2.0, 0.1, 10.0):
            try:
                res = least_squares(resid, p0 * np.where(np.arange(len(p0)) == 1, scale, 1.0),
                                    method="lm" if len(x) > len(p0) else "trf",
                                    max_nfev=20000)
            except Exception:  # noqa: BLE001
                continue
            if best is None or res.cost < best.cost:
                best = res
        if best is None:
            raise ValueError("Curve fitting failed")
        p = best.x
    pred = f(fit_type, p, x)
    ss_res = float(np.nansum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return np.asarray(p, dtype=float), r2


def parse_column(text: str) -> np.ndarray:
    vals = []
    for tok in str(text).replace(",", " ").split():
        try:
            vals.append(float(tok))
        except ValueError:
            pass
    return np.asarray(vals, dtype=float)


def calibrate(image, function: str, unit: str = "Gray Value", x_text: str = "",
              y_text: str = "", global_cal: bool = False):
    """Apply Analyze > Calibrate to ``image``; returns (parameters, R^2) or None."""
    from rsstatchem.core.imaging.image import GRAY16, GRAY32

    cal = image.calibration
    if image.type == GRAY32:
        if function != NONE_LABEL:
            raise ValueError('Function must be "None" for 32-bit images,\n'
                             'but you can change the Unit.')
        cal.value_unit = unit
        return None
    is16 = image.type == GRAY16
    if function == NONE_LABEL:
        cal.function, cal.coefficients = C.NONE, ()
        cal.value_unit = unit or "Gray Value"
        cal.info.pop("zero_clip", None)
        return None
    if function == INVERTER:
        cal.function = C.STRAIGHT_LINE
        cal.coefficients = (65535.0 if is16 else 255.0, -1.0)
        cal.value_unit = "Inverted Gray Value"
        return None
    if function == UNCALIBRATED_OD:
        if is16:
            raise ValueError("Uncalibrated OD is not supported on 16-bit images.")
        cal.function, cal.coefficients = C.UNCALIBRATED_OD, ()
        cal.inverted_lut = image.is_inverted_lut()
        cal.value_unit = "Uncalibrated OD"
        return None
    if function not in FIT_LIST:
        raise ValueError(f'Unknown function "{function}"')
    code = FIT_LIST.index(function)
    x = parse_column(x_text)
    y = parse_column(y_text)
    if cal.is_signed16():
        x = x + 32768
    if len(x) == 0 or len(x) != len(y):
        raise ValueError("To create a calibration curve, the left column must\n"
                         "contain a list of measured mean pixel values and the\n"
                         "right column must contain the same number of calibration\n"
                         "standard values. Use Analyze>Measure to add mean pixel\n"
                         "value measurements to the Results table.")
    if len(x) < N_PARAMS[code] - (1 if code in (STRAIGHT_LINE,) else 0) and \
            code not in (STRAIGHT_LINE,):
        raise ValueError(f"{function} requires at least {N_PARAMS[code]} points")
    p, r2 = fit(code, x, y)
    zero_clip = (not is16 and code != STRAIGHT_LINE and bool(np.all(y >= 0)))
    coeffs = tuple(float(v) for v in p)
    if zero_clip:
        coeffs = coeffs + (1.0,)
    cal.function, cal.coefficients = code, coeffs
    cal.value_unit = unit
    return p, r2
