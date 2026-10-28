"""Readers for the NIST Statistical Reference Datasets (StRD) used by the WP-1 fitting tests.

The ``.dat`` files in ``tests/data/nist_strd_nls`` (nonlinear regression) and
``tests/data/nist_strd_lls`` (linear regression) are the unmodified NIST files from
https://www.itl.nist.gov/div898/strd/ (public domain, U.S. Government work).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent / "data"
NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[EeDd][-+]?\d+)?"

# model equations (in the syntax of the user-function parser) and NIST difficulty
NLS_MODELS = {
    "Misra1a": ("y = b1*(1-exp(-b2*x))", "lower"),
    "Chwirut2": ("y = exp(-b1*x)/(b2+b3*x)", "lower"),
    "Chwirut1": ("y = exp(-b1*x)/(b2+b3*x)", "lower"),
    "Lanczos3": ("y = b1*exp(-b2*x) + b3*exp(-b4*x) + b5*exp(-b6*x)", "lower"),
    "Gauss1": ("y = b1*exp(-b2*x) + b3*exp(-(x-b4)^2/b5^2) + b6*exp(-(x-b7)^2/b8^2)", "lower"),
    "Gauss2": ("y = b1*exp(-b2*x) + b3*exp(-(x-b4)^2/b5^2) + b6*exp(-(x-b7)^2/b8^2)", "lower"),
    "DanWood": ("y = b1*x^b2", "lower"),
    "Misra1b": ("y = b1*(1-(1+b2*x/2)^(-2))", "lower"),
    "Kirby2": ("y = (b1 + b2*x + b3*x^2)/(1 + b4*x + b5*x^2)", "average"),
    "Hahn1": ("y = (b1+b2*x+b3*x^2+b4*x^3)/(1+b5*x+b6*x^2+b7*x^3)", "average"),
    "Nelson": ("z = b1 - b2*x*exp(-b3*y)", "average"),
    "MGH17": ("y = b1 + b2*exp(-x*b4) + b3*exp(-x*b5)", "average"),
    "Lanczos1": ("y = b1*exp(-b2*x) + b3*exp(-b4*x) + b5*exp(-b6*x)", "average"),
    "Lanczos2": ("y = b1*exp(-b2*x) + b3*exp(-b4*x) + b5*exp(-b6*x)", "average"),
    "Gauss3": ("y = b1*exp(-b2*x) + b3*exp(-(x-b4)^2/b5^2) + b6*exp(-(x-b7)^2/b8^2)", "average"),
    "Misra1c": ("y = b1*(1-(1+2*b2*x)^(-0.5))", "average"),
    "Misra1d": ("y = b1*b2*x*((1+b2*x)^(-1))", "average"),
    "Roszman1": ("y = b1 - b2*x - atan(b3/(x-b4))/pi", "average"),
    "ENSO": ("y = b1 + b2*cos(2*pi*x/12) + b3*sin(2*pi*x/12) + b5*cos(2*pi*x/b4) "
             "+ b6*sin(2*pi*x/b4) + b8*cos(2*pi*x/b7) + b9*sin(2*pi*x/b7)", "average"),
    "MGH09": ("y = b1*(x^2+x*b2)/(x^2+x*b3+b4)", "higher"),
    "Thurber": ("y = (b1+b2*x+b3*x^2+b4*x^3)/(1+b5*x+b6*x^2+b7*x^3)", "higher"),
    "BoxBOD": ("y = b1*(1-exp(-b2*x))", "higher"),
    "Rat42": ("y = b1/(1+exp(b2-b3*x))", "higher"),
    "MGH10": ("y = b1*exp(b2/(x+b3))", "higher"),
    "Eckerle4": ("y = (b1/b2)*exp(-0.5*((x-b3)/b2)^2)", "higher"),
    "Rat43": ("y = b1/((1+exp(b2-b3*x))^(1/b4))", "higher"),
    "Bennett5": ("y = b1*(b2+x)^(-1/b3)", "higher"),
}


@dataclass
class NistSet:
    name: str
    start1: np.ndarray
    start2: np.ndarray | None
    certified: np.ndarray
    sd: np.ndarray
    rss: float
    resid_sd: float
    y: np.ndarray
    x: np.ndarray            # (n,) or (n, k)
    names: list[str]


def _floats(text: str) -> list[float]:
    return [float(t.replace("D", "E").replace("d", "e")) for t in re.findall(NUM, text)]


def _data_block(lines: list[str]) -> np.ndarray:
    starts = [i for i, ln in enumerate(lines) if ln.strip().startswith("Data:")]
    rows = []
    for ln in lines[starts[-1] + 1:]:
        vals = _floats(ln)
        if vals:
            rows.append(vals)
    return np.array(rows, float)


def load_nls(name: str) -> NistSet:
    lines = (DATA / "nist_strd_nls" / f"{name}.dat").read_text().splitlines()
    names, s1, s2, cert, sd = [], [], [], [], []
    for ln in lines:
        m = re.match(r"\s*(b\d+)\s*=\s*(.*)", ln)
        if m:
            vals = _floats(m.group(2))
            names.append(m.group(1))
            s1.append(vals[0])
            s2.append(vals[1])
            cert.append(vals[2])
            sd.append(vals[3])
    text = "\n".join(lines)
    rss = _floats(re.search(r"Residual Sum of Squares:\s*(\S+)", text).group(1))[0]
    rsd = _floats(re.search(r"Residual Standard Deviation:\s*(\S+)", text).group(1))[0]
    block = _data_block(lines)
    y, x = block[:, 0], block[:, 1:]
    if x.shape[1] == 1:
        x = x[:, 0]
    return NistSet(name, np.array(s1), np.array(s2), np.array(cert), np.array(sd), rss, rsd,
                   y, x, names)


def load_lls(name: str) -> NistSet:
    lines = (DATA / "nist_strd_lls" / f"{name}.dat").read_text().splitlines()
    names, cert, sd = [], [], []
    for ln in lines:
        m = re.match(r"\s*(B\d+)\s+(" + NUM + r")\s+(" + NUM + r")\s*$", ln)
        if m:
            names.append(m.group(1))
            cert.append(_floats(m.group(2))[0])
            sd.append(_floats(m.group(3))[0])
    text = "\n".join(lines)
    rsd = _floats(re.search(r"Residual\s+Standard Deviation\s+(\S+)", text).group(1))[0]
    block = _data_block(lines)
    y, x = block[:, 0], block[:, 1:]
    if x.shape[1] == 1:
        x = x[:, 0]
    return NistSet(name, np.zeros(len(cert)), None, np.array(cert), np.array(sd), math.nan,
                   rsd, y, x, names)


def lre(value: float, certified: float) -> float:
    """Log relative error (number of correct significant digits), as NIST uses it."""
    if value == certified:
        return 15.0
    if certified == 0:
        return -math.log10(abs(value)) if value else 15.0
    return min(15.0, -math.log10(abs(value - certified) / abs(certified)))
