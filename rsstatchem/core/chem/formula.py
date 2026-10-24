"""Chemical formula parsing, molar masses, composition and rich-text formatting.

Mirrors the formula tools of ChemDraw (Analysis window) and the molar-mass calculators
chemists use for theses: molar mass from IUPAC standard atomic weights, monoisotopic mass,
elemental composition (mass %), Hill and empirical formulas, empirical/molecular formula from
percent composition, degree of unsaturation, and formatting with subscripts/superscripts as
Unicode (``H₂SO₄``, ``SO₄²⁻``), HTML and matplotlib mathtext. No RDKit needed.

Accepted syntax: nested ``()``, ``[]``, ``{}`` with (decimal) multipliers; hydrates/adducts
``CuSO4·5H2O``, ``CuSO4.5H2O``, ``CuSO4*5H2O``, ``CaSO4·0.5H2O``; charges ``SO4^2-``,
``SO4^{2-}``, ``SO42-``, ``Fe3+``, ``NH4+``, ``[Fe(CN)6]3-``, ``Fe+++``, ``SO₄²⁻``;
isotopes ``[13C]``, ``¹³C``, ``D``, ``T``; phase labels ``(aq)``, ``(s)``, ``(l)``, ``(g)``;
the electron ``e-``. Without ``^`` a single digit right before the trailing sign is read as
the charge for monatomic ions (``Fe3+``), after ``]`` (``[Fe(CN)6]4-``) or when two digits
precede the sign (``SO42-``); otherwise it is a subscript (``NH4+``, ``NO3-``). Write ``^``
to be explicit.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from fractions import Fraction

from rsstatchem.core.chem import elements as el

__all__ = ["FormulaError", "Formula", "parse_formula", "molar_mass", "format_formula",
           "format_charge", "CompositionRow", "EmpiricalResult", "empirical_from_composition",
           "combustion_composition", "count_text"]


class FormulaError(ValueError):
    """Raised for formulas that cannot be parsed."""


_SUB = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
_SUP_DIGITS = "⁰¹²³⁴⁵⁶⁷⁸⁹"
_SUP_MAP = str.maketrans(_SUP_DIGITS + "⁺⁻", "0123456789+-")
_TO_SUB = str.maketrans("0123456789.", "₀₁₂₃₄₅₆₇₈₉.")
_TO_SUP = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")
_SEPARATORS = "·•∙⋅*"
_PHASE_RE = re.compile(r"\((aq|s|l|g|sol|cr|am)\)$", re.IGNORECASE)
_ISO_RE = re.compile(r"\[(\d+)([A-Z][a-z]{0,2})\]")
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")
_COEF_RE = re.compile(r"(\d+(?:\.\d+)?(?:/\d+)?|\d*/\d+)")
_OPEN = {"(": ")", "[": "]", "{": "}"}
_CLOSE = set(_OPEN.values())

Key = tuple  # (symbol, mass_number | None)


def _normalize(text: str) -> str:
    """ASCII digits, ``^`` charges and ``[A X]`` isotope labels from Unicode input."""
    text = text.strip().translate(_SUB)
    text = text.replace("−", "-").replace("–", "-").replace("→", "->")
    text = re.sub(r"\s+", "", text)
    text = re.sub(f"([{_SUP_DIGITS}]+)(?=[A-Z])",
                  lambda m: "[" + m.group(1).translate(_SUP_MAP) + "]", text)
    text = re.sub(r"\[(\d+)\]([A-Z][a-z]{0,2})", r"[\1\2]", text)
    m = re.search(f"[{_SUP_DIGITS}⁺⁻]+$", text)
    if m:
        text = text[:m.start()] + "^" + m.group(0).translate(_SUP_MAP)
    return text


def _split_phase(text: str) -> tuple[str, str]:
    m = _PHASE_RE.search(text)
    if m and m.start() > 0:
        return text[:m.start()], m.group(1).lower()
    return text, ""


def _split_charge(body: str) -> tuple[str, int]:
    """Separate a trailing charge from the formula body (see module docstring)."""
    m = re.search(r"\^\{?(\d*)([+-]+)\}?$", body) or re.search(r"\^\{?([+-])(\d+)\}?$", body)
    if m:
        a, b = m.groups()
        digits, signs = (a, b) if a == "" or a.isdigit() else (b, a)
        if digits and len(signs) > 1:
            raise FormulaError(f"Ambiguous charge '{m.group(0)}'")
        n = int(digits) if digits else len(signs)
        return body[:m.start()], n if signs[0] == "+" else -n
    m = re.search(r"([+-]+)$", body)
    if not m:
        return body, 0
    signs, head = m.group(1), body[:m.start()]
    sign = 1 if signs[0] == "+" else -1
    if any(s != signs[0] for s in signs):
        raise FormulaError(f"Invalid charge '{signs}'")
    d = re.search(r"(\d+)$", head)
    if not d or len(signs) > 1:
        return head, sign * len(signs)
    digits, prefix = d.group(1), head[:d.start()]
    if len(digits) >= 2 and digits[-1] != "0":
        return head[:-1], sign * int(digits[-1])
    if len(digits) == 1 and digits != "0":
        if prefix.endswith("]") and not _ISO_RE.fullmatch(prefix):
            return prefix, sign * int(digits)
        if re.fullmatch(r"(\[\d+)?[A-Z][a-z]{0,2}\]?|[DT]", prefix):
            return prefix, sign * int(digits)
    return head, sign


def _parse_counts(s: str, i: int, closer: str | None) -> tuple[dict, int]:
    counts: dict[Key, Fraction] = {}

    def add(key: Key, n: Fraction) -> None:
        counts[key] = counts.get(key, Fraction(0)) + n

    def number(j: int) -> tuple[Fraction, int]:
        m = _NUM_RE.match(s, j)
        return (Fraction(m.group(0)), m.end()) if m else (Fraction(1), j)

    while i < len(s):
        c = s[i]
        if c == "[" and _ISO_RE.match(s, i):
            m = _ISO_RE.match(s, i)
            sym, a = m.group(2), int(m.group(1))
            if sym in el.ISOTOPE_SYMBOLS or sym not in el.ELEMENTS:
                raise FormulaError(f"Unknown element '{sym}' in isotope label {m.group(0)}")
            try:
                el.element(sym).isotope(a)
            except KeyError as exc:
                raise FormulaError(str(exc).strip("'\"")) from None
            n, i = number(m.end())
            add((sym, a), n)
        elif c in _OPEN:
            sub, i = _parse_counts(s, i + 1, _OPEN[c])
            n, i = number(i)
            for key, v in sub.items():
                add(key, v * n)
        elif c in _CLOSE:
            if c != closer:
                raise FormulaError(f"Unmatched '{c}' at position {i + 1}")
            return counts, i + 1
        elif c.isupper():
            two = s[i:i + 2]
            if len(two) == 2 and two[1].islower():
                if not el.is_symbol(two):
                    raise FormulaError(f"Unknown element '{two}' at position {i + 1}")
                sym = two
            elif el.is_symbol(c):
                sym = c
            else:
                raise FormulaError(f"Unknown element '{c}' at position {i + 1}")
            n, i = number(i + len(sym))
            add(el.ISOTOPE_SYMBOLS.get(sym, (sym, None)), n)
        else:
            raise FormulaError(f"Unexpected '{c}' at position {i + 1}")
    if closer is not None:
        raise FormulaError(f"Missing '{closer}'")
    return counts, i


def _split_parts(body: str) -> list[str]:
    for sep in _SEPARATORS:
        body = body.replace(sep, "\0")
    pieces = []
    for raw in body.split("\0"):
        bits = raw.split(".")
        merged: list[str] = []
        for b in bits:
            # "CaSO4.0.5H2O": a bare integer followed by a part starting with a digit is
            # the integer half of a decimal coefficient; "Fe0.95O": a subscript "0" followed
            # by digits is a decimal subscript. Every other '.' is a hydrate dot.
            if merged and b[:1].isdigit() and (merged[-1].isdigit()
                                               or re.search(r"[A-Za-z)\]}]0$", merged[-1])):
                merged[-1] += "." + b
            else:
                merged.append(b)
        pieces.extend(merged)
    if any(p == "" for p in pieces):
        raise FormulaError("Empty part in formula (check '·' or '.' separators)")
    return pieces


@dataclass
class CompositionRow:
    label: str          # "C", "D", "[13C]"
    symbol: str
    count: Fraction
    atomic_weight: float
    mass: float          # count × atomic weight
    percent: float


@dataclass
class Formula:
    """A parsed formula: atom counts keyed by (symbol, mass number or None), plus charge."""

    counts: dict = field(default_factory=dict)
    charge: int = 0
    text: str = ""
    phase: str = ""
    electron: bool = False

    # --- construction -----------------------------------------------------------------
    @classmethod
    def parse(cls, text: str) -> "Formula":
        return parse_formula(text)

    @classmethod
    def from_elements(cls, counts: dict, charge: int = 0) -> "Formula":
        """From ``{"C": 6, "H": 12, ("C", 13): 1}``."""
        out = {}
        for key, n in counts.items():
            if isinstance(key, str):
                key = el.ISOTOPE_SYMBOLS.get(key, (key, None))
            el.element(key[0])
            out[key] = out.get(key, Fraction(0)) + Fraction(n).limit_denominator(10**6)
        f = cls(out, charge)
        f.text = f.hill()
        return f

    # --- basic properties -------------------------------------------------------------
    @property
    def elements(self) -> dict[str, Fraction]:
        """Counts per element (isotopes merged into their element)."""
        out: dict[str, Fraction] = {}
        for (sym, _a), n in self.counts.items():
            out[sym] = out.get(sym, Fraction(0)) + n
        return out

    @property
    def is_integral(self) -> bool:
        return all(n.denominator == 1 for n in self.counts.values())

    @property
    def molar_mass(self) -> float:
        """g/mol from IUPAC standard atomic weights (electron mass of ions neglected)."""
        if self.electron:
            return el.ELECTRON_MASS
        return sum(float(n) * el.atomic_weight(s, a) for (s, a), n in self.counts.items())

    @property
    def monoisotopic_mass(self) -> float:
        """Monoisotopic mass (u) of the most abundant isotopes, corrected for the charge."""
        if self.electron:
            return el.ELECTRON_MASS
        m = sum(float(n) * el.monoisotopic_mass(s, a) for (s, a), n in self.counts.items())
        return m - self.charge * el.ELECTRON_MASS

    @property
    def mz(self) -> float | None:
        return self.monoisotopic_mass / abs(self.charge) if self.charge else None

    @property
    def atom_count(self) -> Fraction:
        return sum(self.counts.values(), Fraction(0))

    @property
    def estimated(self) -> list[str]:
        """Elements whose weight is a mass number (no stable isotopes)."""
        return sorted({s for (s, a) in self.counts if a is None and el.element(s).estimated})

    def dbe(self) -> float | None:
        """Degree of unsaturation (rings + π bonds); None if the formula has other elements.

        DBE = 1 + C + Si − (H + X)/2 + (N + P)/2 for C, H, N, O, S, P, Si, halogens, Ge/Sn
        treated as tetravalent, B as trivalent."""
        e = self.elements
        tetra = {"C", "Si", "Ge", "Sn"}
        tri = {"N", "P", "As", "B", "Sb"}
        mono = {"H"} | el.HALOGENS | {"Li", "Na", "K"}
        di = {"O", "S", "Se", "Te"}
        if not e or any(s not in tetra | tri | mono | di for s in e):
            return None
        val = 1 + sum(float(e.get(s, 0)) for s in tetra) - \
            sum(float(e.get(s, 0)) for s in mono) / 2 + sum(float(e.get(s, 0)) for s in tri) / 2
        return val - self.charge / 2 if self.charge else val

    # --- derived formulas -------------------------------------------------------------
    def _ordered(self) -> list:
        keys = list(self.counts)
        syms = {s for s, _ in keys}
        if "C" in syms:
            first = ["C", "H"] if "H" in syms else ["C"]
            rest = sorted(syms - set(first))
            order = first + rest
        else:
            order = sorted(syms)
        return sorted(keys, key=lambda k: (order.index(k[0]), k[1] or 0))

    def hill(self, with_charge: bool = True) -> str:
        """Hill-order formula (C, H, then alphabetical; alphabetical without carbon)."""
        if self.electron:
            return "e^-"
        text = "".join(_label(k) + count_text(self.counts[k]) for k in self._ordered())
        if with_charge and self.charge:
            text += "^" + format_charge(self.charge)
        return text

    def empirical(self) -> "Formula":
        """Smallest whole-number ratio (empirical formula); charge is dropped."""
        if not self.counts:
            return Formula()
        lcm = 1
        for n in self.counts.values():
            lcm = lcm * n.denominator // math.gcd(lcm, n.denominator)
        ints = {k: int(n * lcm) for k, n in self.counts.items()}
        g = 0
        for v in ints.values():
            g = math.gcd(g, v)
        out = Formula({k: Fraction(v // g) for k, v in ints.items()}, 0)
        out.text = out.hill()
        return out

    def scaled(self, k) -> "Formula":
        f = Formula({key: n * Fraction(k) for key, n in self.counts.items()},
                    int(self.charge * k))
        f.text = f.hill()
        return f

    def __add__(self, other: "Formula") -> "Formula":
        counts = dict(self.counts)
        for k, n in other.counts.items():
            counts[k] = counts.get(k, Fraction(0)) + n
        f = Formula(counts, self.charge + other.charge)
        f.text = f.hill()
        return f

    def composition(self) -> list[CompositionRow]:
        """Mass % per element (and per labelled isotope), in Hill order."""
        total = self.molar_mass
        rows = []
        for key in self._ordered():
            n = self.counts[key]
            w = el.atomic_weight(*key)
            mass = float(n) * w
            rows.append(CompositionRow(_label(key), key[0], n, w, mass,
                                       100.0 * mass / total if total else math.nan))
        return rows

    def mass_percent(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for row in self.composition():
            out[row.symbol] = out.get(row.symbol, 0.0) + row.percent
        return out

    def same_as(self, other: "Formula") -> bool:
        a = {k: v for k, v in self.counts.items() if v}
        b = {k: v for k, v in other.counts.items() if v}
        return a == b and self.charge == other.charge

    # --- formatting -------------------------------------------------------------------
    def unicode(self, hill: bool = False) -> str:
        return format_formula(self.hill() if hill or not self.text else self.text, "unicode")

    def html(self, hill: bool = False) -> str:
        return format_formula(self.hill() if hill or not self.text else self.text, "html")

    def mathtext(self, hill: bool = False) -> str:
        return format_formula(self.hill() if hill or not self.text else self.text, "mathtext")

    def __str__(self) -> str:
        return self.text or self.hill()


def _label(key: Key) -> str:
    sym, a = key
    if a is None:
        return sym
    if sym == "H" and a in (2, 3):
        return "D" if a == 2 else "T"
    return f"[{a}{sym}]"


def count_text(n: Fraction) -> str:
    if n == 1:
        return ""
    if n.denominator == 1:
        return str(n.numerator)
    return f"{float(n):.6g}"


def format_charge(z: int) -> str:
    if z == 0:
        return ""
    sign = "+" if z > 0 else "-"
    return sign if abs(z) == 1 else f"{abs(z)}{sign}"


def parse_formula(text: str) -> Formula:
    """Parse a formula; raises :class:`FormulaError` with a helpful message."""
    if text is None or not str(text).strip():
        raise FormulaError("Empty formula")
    original = str(text).strip()
    s = _normalize(original)
    s, phase = _split_phase(s)
    if s in ("e", "e-", "e^-", "e^{-}"):
        return Formula({}, -1, original, phase, electron=True)
    body, charge = _split_charge(s)
    if not body:
        raise FormulaError("No atoms in formula")
    total: dict[Key, Fraction] = {}
    for part in _split_parts(body):
        m = _COEF_RE.match(part)
        coef = Fraction(1)
        if m and m.end() < len(part):
            coef = Fraction(m.group(0)) if not m.group(0).startswith("/") else Fraction(1)
            part = part[m.end():]
        elif m:
            raise FormulaError(f"'{part}' is a number, not a formula")
        counts, _ = _parse_counts(part, 0, None)
        for key, n in counts.items():
            total[key] = total.get(key, Fraction(0)) + n * coef
    return Formula(total, charge, original, phase)


def molar_mass(text: str) -> float:
    return parse_formula(text).molar_mass


# --- formatting ----------------------------------------------------------------------------

def _tokens(text: str) -> list[tuple[str, str]]:
    """Split a formula into (kind, text) with kind in normal/sub/sup/charge."""
    s = _normalize(text)
    s, phase = _split_phase(s)
    if s in ("e", "e-", "e^-"):
        return [("normal", "e"), ("charge", "-")] + ([("normal", f"({phase})")] if phase else [])
    try:
        body, charge = _split_charge(s)
    except FormulaError:
        body, charge = s, 0
    out: list[tuple[str, str]] = []
    i, start_of_part = 0, True
    while i < len(body):
        c = body[i]
        if c in _SEPARATORS or c == ".":
            out.append(("normal", "·"))
            start_of_part = True
            i += 1
            continue
        m = _ISO_RE.match(body, i)
        if m:
            out += [("sup", m.group(1)), ("normal", m.group(2))]
            i = m.end()
            start_of_part = False
            continue
        if c.isdigit():
            prev = body[i - 1] if i else ""
            kind = "normal" if start_of_part or not (prev.isalpha() or prev in _CLOSE) else "sub"
            pattern = r"\d+(?:\.\d+)?(?:/\d+)?" if kind == "normal" else r"0\.\d+|\d+"
            num = re.match(pattern, body[i:]).group(0)
            out.append((kind, num))
            i += len(num)
            continue
        out.append(("normal", c))
        start_of_part = False
        i += 1
    if charge:
        out.append(("charge", format_charge(charge)))
    if phase:
        out.append(("normal", f"({phase})"))
    return out


def format_formula(text: str, style: str = "unicode") -> str:
    """Render a formula with subscripts/superscripts.

    ``style``: ``unicode`` (H₂SO₄, SO₄²⁻), ``html`` (H<sub>2</sub>…), ``mathtext``
    (H$_2$SO$_4$, for matplotlib/Graph Studio labels) or ``plain``.
    Text that is not a formula is returned unchanged."""
    if style == "plain":
        return text
    try:
        parse_formula(text)
        toks = _tokens(text)
    except Exception:  # noqa: BLE001 - formatting must never fail
        return text
    if style == "unicode":
        return "".join(t.translate(_TO_SUB) if k == "sub" else
                       t.translate(_TO_SUP) if k in ("sup", "charge") else t for k, t in toks)
    if style == "html":
        out = []
        for k, t in toks:
            if k == "sub":
                out.append(f"<sub>{t}</sub>")
            elif k in ("sup", "charge"):
                out.append(f"<sup>{t.replace('-', '&minus;')}</sup>")
            else:
                out.append(t.replace("&", "&amp;").replace("<", "&lt;"))
        return "".join(out)
    if style == "mathtext":
        out, i = [], 0
        while i < len(toks):
            k, t = toks[i]
            if k == "sub" and i + 1 < len(toks) and toks[i + 1][0] == "charge":
                out.append(f"$_{{{t}}}^{{{toks[i + 1][1]}}}$")
                i += 2
                continue
            if k == "sub":
                out.append(f"$_{{{t}}}$")
            elif k in ("sup", "charge"):
                out.append(f"$^{{{t}}}$")
            else:
                out.append(t.replace("$", r"\$"))
            i += 1
        return "".join(out)
    raise ValueError(f"Unknown style '{style}'")


# --- composition → formula -------------------------------------------------------------

@dataclass
class EmpiricalRow:
    symbol: str
    percent: float
    atomic_weight: float
    moles: float        # mol per 100 g
    ratio: float        # moles / smallest
    scaled: float       # ratio × multiplier
    subscript: int


@dataclass
class EmpiricalResult:
    rows: list[EmpiricalRow]
    multiplier: int
    max_deviation: float
    empirical: Formula
    empirical_mass: float
    percent_sum: float
    by_difference: str | None = None
    molar_mass: float | None = None
    n: float | None = None           # molar mass / empirical mass
    molecular: Formula | None = None
    warnings: list[str] = field(default_factory=list)


def empirical_from_composition(percents: dict, molar_mass: float | None = None,
                               tolerance: float = 0.1, max_multiplier: int = 12
                               ) -> EmpiricalResult:
    """Empirical (and molecular) formula from mass percentages.

    ``percents`` maps element symbols to mass %; one value may be ``None`` to take it by
    difference from 100 % (usually oxygen). The mole ratios are multiplied by the smallest
    integer 1…``max_multiplier`` that brings every ratio within ``tolerance`` of a whole
    number (the textbook procedure). With ``molar_mass`` the molecular formula is the
    empirical formula × round(M / M_empirical)."""
    data = {}
    missing = [s for s, v in percents.items() if v is None or (isinstance(v, float)
                                                                and math.isnan(v))]
    if len(missing) > 1:
        raise ValueError("Only one element can be taken by difference.")
    for sym in percents:
        sym_ok = el.ISOTOPE_SYMBOLS.get(sym, (sym, None))[0]
        el.element(sym_ok)
    given = sum(float(v) for s, v in percents.items() if s not in missing)
    warnings = []
    for sym, v in percents.items():
        if sym in missing:
            rest = 100.0 - given
            if rest <= 0:
                raise ValueError("The given percentages already reach 100 %.")
            data[sym] = rest
        else:
            if float(v) < 0:
                raise ValueError(f"Negative percentage for {sym}.")
            if float(v) > 0:
                data[sym] = float(v)
    if not data:
        raise ValueError("Enter at least one element percentage.")
    total = sum(data.values())
    if abs(total - 100.0) > 1.0:
        warnings.append(f"The percentages sum to {total:.2f} %, not 100 %; ratios are still "
                        "valid if the remainder is an element you did not list.")
    moles = {s: p / el.atomic_weight(s) for s, p in data.items()}
    smallest = min(moles.values())
    ratios = {s: m / smallest for s, m in moles.items()}
    best_k, best_dev = 1, math.inf
    chosen = None
    for k in range(1, max_multiplier + 1):
        dev = max(abs(r * k - round(r * k)) for r in ratios.values())
        if dev < best_dev - 1e-12:
            best_k, best_dev = k, dev
        if dev <= tolerance:
            chosen = (k, dev)
            break
    if chosen is None:
        chosen = (best_k, best_dev)
        warnings.append(f"No multiplier up to {max_multiplier} gives whole numbers within "
                        f"{tolerance}; the closest (×{best_k}, deviation {best_dev:.3f}) "
                        "is used. Check the data.")
    k, dev = chosen
    subs = {s: max(1, round(r * k)) for s, r in ratios.items()}
    emp = Formula.from_elements(subs)
    rows = [EmpiricalRow(s, data[s], el.atomic_weight(s), moles[s], ratios[s], ratios[s] * k,
                         subs[s]) for s in _hill_symbols(list(data))]
    result = EmpiricalResult(rows, k, dev, emp, emp.molar_mass, total,
                             missing[0] if missing else None, warnings=warnings)
    if molar_mass:
        n = molar_mass / emp.molar_mass
        result.molar_mass = float(molar_mass)
        result.n = n
        nn = max(1, round(n))
        if abs(n - nn) > 0.1 * nn:
            warnings.append(f"M / M(empirical) = {n:.3f} is not close to a whole number.")
        result.molecular = emp.scaled(nn)
    return result


def _hill_symbols(symbols: list[str]) -> list[str]:
    syms = set(symbols)
    if "C" in syms:
        first = ["C", "H"] if "H" in syms else ["C"]
        return first + sorted(syms - set(first))
    return sorted(syms)


def combustion_composition(sample_mg: float, co2_mg: float, h2o_mg: float,
                           n2_mg: float = 0.0, other: dict | None = None) -> dict:
    """Mass % of C, H (and N) from a CHN combustion analysis; O is left by difference.

    Returns ``{"C": %, "H": %, "N": %, "O": None}`` ready for
    :func:`empirical_from_composition`."""
    if sample_mg <= 0:
        raise ValueError("Sample mass must be positive.")
    c = co2_mg * el.atomic_weight("C") / parse_formula("CO2").molar_mass
    h = h2o_mg * 2 * el.atomic_weight("H") / parse_formula("H2O").molar_mass
    out = {"C": 100 * c / sample_mg, "H": 100 * h / sample_mg}
    if n2_mg:
        out["N"] = 100 * n2_mg / sample_mg
    for sym, pct in (other or {}).items():
        out[sym] = pct
    if sum(out.values()) < 99.5:
        out["O"] = None
    return out
