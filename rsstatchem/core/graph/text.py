"""Rich text for graph labels and font resolution.

Labels accept matplotlib mathtext (``H$_2$O``, ``cm$^{-1}$``, ``$\\alpha$``) and Origin's
escape codes ``\\+(sup)``, ``\\-(sub)``, ``\\g(a)`` (Greek), ``\\b(bold)``, ``\\i(italic)``.
Invalid mathtext is shown literally instead of breaking the drawing.
"""
from __future__ import annotations

import functools
import re

from matplotlib import font_manager
from matplotlib.mathtext import MathTextParser

_GREEK = {"a": "alpha", "b": "beta", "g": "gamma", "d": "delta", "e": "epsilon",
          "z": "zeta", "h": "eta", "q": "theta", "i": "iota", "k": "kappa", "l": "lambda",
          "m": "mu", "n": "nu", "x": "xi", "p": "pi", "r": "rho", "s": "sigma", "t": "tau",
          "u": "upsilon", "f": "phi", "c": "chi", "y": "psi", "w": "omega",
          "D": "Delta", "G": "Gamma", "L": "Lambda", "P": "Pi", "S": "Sigma", "F": "Phi",
          "W": "Omega", "Q": "Theta", "X": "Xi", "Y": "Psi"}
_ESC = re.compile(r"\\([+\-gbi])\(([^()]*)\)")
_parser = MathTextParser("path")


def _origin_escapes(text: str) -> str:
    def repl(m: re.Match) -> str:
        code, body = m.group(1), m.group(2)
        if code == "+":
            return f"$^{{{_math_body(body)}}}$"
        if code == "-":
            return f"$_{{{_math_body(body)}}}$"
        if code == "g":
            return "".join(f"$\\{_GREEK[ch]}$" if ch in _GREEK else ch for ch in body)
        if code == "b":
            return f"$\\mathbf{{{_math_body(body)}}}$"
        return f"$\\mathit{{{_math_body(body)}}}$"
    return _ESC.sub(repl, text)


def _math_body(body: str) -> str:
    body = body.replace("$", "")
    return re.sub(r"([ %&#_{}])", lambda m: "\\ " if m.group(1) == " " else "\\" + m.group(1),
                  body)


@functools.lru_cache(maxsize=2048)
def rich(text: str) -> str:
    """Text ready for matplotlib: escapes converted, broken mathtext made literal."""
    if not text:
        return ""
    text = _origin_escapes(str(text))
    if text.count("$") - text.count(r"\$") == 0:
        return text
    parts = re.split(r"(?<!\\)\$", text)
    if len(parts) % 2 == 0:  # unbalanced $
        return text.replace("$", r"\$")
    for i in range(1, len(parts), 2):
        try:
            _parser.parse(f"${parts[i]}$")
        except Exception:  # noqa: BLE001 - any parse error -> show literally
            return text.replace("$", r"\$")
    return text


@functools.lru_cache(maxsize=1)
def _available() -> frozenset[str]:
    return frozenset(f.name for f in font_manager.fontManager.ttflist)


_FALLBACKS = {
    "sans": ["Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
    "serif": ["Times New Roman", "Times", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
    "mono": ["Courier New", "Liberation Mono", "DejaVu Sans Mono"],
}


@functools.lru_cache(maxsize=256)
def resolve_family(name: str) -> str:
    """An installed family close to ``name`` (avoids matplotlib's findfont warnings)."""
    avail = _available()
    if name in avail:
        return name
    low = (name or "").lower()
    group = "serif" if any(k in low for k in ("times", "serif", "roman", "georgia",
                                                "cambria")) else \
        "mono" if "mono" in low or "courier" in low else "sans"
    for cand in _FALLBACKS[group]:
        if cand in avail:
            return cand
    return "DejaVu Sans"


def installed_families() -> list[str]:
    return sorted(_available())
