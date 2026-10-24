"""Chemical equations: parsing, exact balancing and stoichiometry (no RDKit).

Balancing mirrors the equation balancers of ChemDraw / WebQC: the atom (and charge) balance
matrix is solved exactly over the rationals (``fractions.Fraction``) and the null-space
vector is scaled to the smallest whole numbers. Ionic and redox equations work because the
charge is one more conservation row (``e-`` is the electron). The result says when an equation
is impossible, cannot be balanced with positive coefficients, or has several independent
balancings (not unique).

The stoichiometry calculator converts given amounts (mass, moles, solution volume ×
molarity, gas volume via the ideal-gas law) to moles, finds the limiting reagent from the
reaction extent ξ = n/ν, and reports theoretical yields, the excess left over and percent
yields from actual yields.
"""
from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass, field
from fractions import Fraction

from rsstatchem.core.chem.formula import Formula, FormulaError, format_formula, parse_formula

__all__ = ["Species", "BalanceResult", "parse_equation", "balance", "Amount",
           "StoichRow", "StoichResult", "stoichiometry", "R_GAS", "MASS_UNITS",
           "AMOUNT_UNITS", "SOLUTION_UNITS", "GAS_UNITS"]

R_GAS = 0.082057366  # L·atm/(mol·K)
_ARROWS = r"<=+>|<-+>|⇌|⇄|⟷|↔|→|⟶|⟹|⇒|-+>|=+>|="
_ARROW_RE = re.compile(r"\s+(" + _ARROWS + r")\s+")
_ARROW_RE_TIGHT = re.compile(r"(" + _ARROWS + r")")
_COEF_RE = re.compile(r"^(\d+/\d+|\d+(?:\.\d+)?)\s*(?=[A-Z(\[{e¹²³⁴⁵⁶⁷⁸⁹⁰])")


@dataclass
class Species:
    text: str              # as typed, without coefficient
    formula: Formula
    side: int              # +1 reactant, -1 product
    given: Fraction | None = None

    @property
    def role(self) -> str:
        return "Reactant" if self.side > 0 else "Product"

    def display(self, style: str = "unicode") -> str:
        return format_formula(self.text, style)


def _split_species(side: str) -> list[str]:
    side = side.strip()
    if not side:
        return []
    if re.search(r"\s\+\s", side):
        return [p.strip() for p in re.split(r"\s+\+\s+", side) if p.strip()]
    s = re.sub(r"\s+", "", side)
    parts, start = [], 0
    for p, ch in enumerate(s):
        if ch != "+" or p == start:
            continue
        nxt = s[p + 1:p + 2]
        prev = s[p - 1]
        starts = nxt[:1].isdigit() or nxt[:1].isupper() or nxt in ("(", "[", "{") or \
            (nxt == "e" and s[p + 2:p + 3] in ("", "-", "^", "⁻", "+"))
        if starts and prev not in "^{":
            parts.append(s[start:p])
            start = p + 1
    parts.append(s[start:])
    return [p for p in parts if p]


def parse_equation(text: str) -> tuple[list[Species], str]:
    """Split ``"2 H2 + O2 -> 2 H2O"`` into species (coefficients kept as ``given``)."""
    if not text or not text.strip():
        raise FormulaError("Enter an equation such as  Fe + O2 -> Fe2O3")
    t = text.strip()
    m = _ARROW_RE.search(t) or _ARROW_RE_TIGHT.search(t)
    if not m:
        raise FormulaError("No reaction arrow found (use ->, →, = or <=>).")
    left, right, arrow = t[:m.start()], t[m.end():], m.group(1)
    if _ARROW_RE_TIGHT.search(right):
        raise FormulaError("More than one reaction arrow.")
    species = []
    for side_text, sign in ((left, 1), (right, -1)):
        items = _split_species(side_text)
        if not items:
            raise FormulaError("Both sides of the equation need at least one species.")
        for item in items:
            given = None
            cm = _COEF_RE.match(item)
            if cm:
                given = Fraction(cm.group(1))
                item = item[cm.end():].strip()
            try:
                f = parse_formula(item)
            except FormulaError as exc:
                raise FormulaError(f"{item}: {exc}") from None
            species.append(Species(item, f, sign, given))
    reversible = arrow in ("⇌", "⇄", "⟷", "↔") or arrow.startswith("<")
    return species, "⇌" if reversible else "→"


# --- exact linear algebra ------------------------------------------------------------------

def _nullspace(matrix: list[list[Fraction]], ncols: int) -> list[list[Fraction]]:
    a = [row[:] for row in matrix]
    pivots: list[int] = []
    r = 0
    for c in range(ncols):
        pivot = next((i for i in range(r, len(a)) if a[i][c] != 0), None)
        if pivot is None:
            continue
        a[r], a[pivot] = a[pivot], a[r]
        pv = a[r][c]
        a[r] = [x / pv for x in a[r]]
        for i in range(len(a)):
            if i != r and a[i][c] != 0:
                f = a[i][c]
                a[i] = [x - f * y for x, y in zip(a[i], a[r])]
        pivots.append(c)
        r += 1
        if r == len(a):
            break
    free = [c for c in range(ncols) if c not in pivots]
    basis = []
    for fc in free:
        v = [Fraction(0)] * ncols
        v[fc] = Fraction(1)
        for i, pc in enumerate(pivots):
            v[pc] = -a[i][fc]
        basis.append(v)
    return basis


def _integerize(v: list[Fraction]) -> list[int]:
    lcm = 1
    for x in v:
        lcm = lcm * x.denominator // math.gcd(lcm, x.denominator)
    ints = [int(x * lcm) for x in v]
    g = 0
    for x in ints:
        g = math.gcd(g, x)
    return [x // g for x in ints] if g else ints


@dataclass
class BalanceResult:
    species: list[Species]
    arrow: str
    status: str                      # balanced | not_unique | impossible | no_positive
    coefficients: list[int] | None
    basis: list[list[int]] = field(default_factory=list)
    message: str = ""
    elements: list = field(default_factory=list)   # (symbol, mass number) keys
    charged: bool = False
    given_balanced: bool | None = None

    @property
    def ok(self) -> bool:
        return self.coefficients is not None

    @property
    def reactants(self) -> list[int]:
        return [i for i, s in enumerate(self.species) if s.side > 0]

    @property
    def products(self) -> list[int]:
        return [i for i, s in enumerate(self.species) if s.side < 0]

    def equation(self, style: str = "unicode", coefficients: list | None = None) -> str:
        coefs = coefficients if coefficients is not None else self.coefficients
        sides = []
        for idx in (self.reactants, self.products):
            terms = []
            for i in idx:
                c = coefs[i] if coefs is not None else None
                lead = "" if c in (None, 1) else f"{c} "
                terms.append(lead + self.species[i].display(style))
            sides.append(" + ".join(terms))
        arrow = self.arrow if style != "mathtext" else (
            r"$\rightleftharpoons$" if self.arrow == "⇌" else r"$\rightarrow$")
        if style == "plain":
            arrow = "<=>" if self.arrow == "⇌" else "->"
        return f"{sides[0]} {arrow} {sides[1]}"

    def balance_table(self) -> list[tuple[str, Fraction, Fraction]]:
        """(element or 'Charge', left total, right total) for the balanced equation."""
        if self.coefficients is None:
            return []
        rows = []
        for key in self.elements:
            left = sum((self.coefficients[i] * s.formula.counts.get(key, 0)
                        for i, s in enumerate(self.species) if s.side > 0), Fraction(0))
            right = sum((self.coefficients[i] * s.formula.counts.get(key, 0)
                         for i, s in enumerate(self.species) if s.side < 0), Fraction(0))
            rows.append((_key_label(key), left, right))
        if self.charged:
            left = sum(self.coefficients[i] * s.formula.charge
                       for i, s in enumerate(self.species) if s.side > 0)
            right = sum(self.coefficients[i] * s.formula.charge
                        for i, s in enumerate(self.species) if s.side < 0)
            rows.append(("Charge", Fraction(left), Fraction(right)))
        return rows


def _key_label(key) -> str:
    sym, a = key
    return sym if a is None else ("D" if (sym, a) == ("H", 2) else f"[{a}{sym}]")


def balance(text_or_species, arrow: str = "→") -> BalanceResult:
    """Balance an equation given as text or as a list of :class:`Species`."""
    if isinstance(text_or_species, str):
        species, arrow = parse_equation(text_or_species)
    else:
        species = list(text_or_species)
    keys: list = []
    for s in species:
        for k in s.formula.counts:
            if k not in keys:
                keys.append(k)
    syms = {k[0] for k in keys}
    order = (["C", "H"] if "C" in syms else []) + sorted(syms - {"C", "H"}) + \
        (["H"] if "C" not in syms and "H" in syms else [])
    keys.sort(key=lambda k: (order.index(k[0]) if k[0] in order else 99, k[1] or 0))
    charged = any(s.formula.charge for s in species)
    rows = [[Fraction(s.side) * s.formula.counts.get(k, 0) for s in species] for k in keys]
    if charged:
        rows.append([Fraction(s.side * s.formula.charge) for s in species])
    n = len(species)
    res = BalanceResult(species, arrow, "impossible", None, elements=keys, charged=charged)
    if any(s.given is not None for s in species):  # omitted coefficients mean 1
        given = [s.given if s.given is not None else Fraction(1) for s in species]
        res.given_balanced = all(sum(r[i] * given[i] for i in range(n)) == 0 for r in rows)
    for k in keys:
        on_left = any(s.formula.counts.get(k) for s in species if s.side > 0)
        on_right = any(s.formula.counts.get(k) for s in species if s.side < 0)
        if on_left != on_right:
            res.message = (f"{_key_label(k)} appears only on the "
                           f"{'left' if on_left else 'right'} side; the equation cannot be "
                           "balanced. Check the formulas or add the missing species.")
            return res
    basis = _nullspace(rows, n)
    res.basis = [_integerize(v) for v in basis]
    if not basis:
        res.message = ("The equation cannot be balanced: the only solution of the atom"
                       + (" and charge" if charged else "") + " balance is all zeros.")
        return res
    if len(basis) == 1:
        v = _integerize(basis[0])
        if all(x < 0 for x in v):
            v = [-x for x in v]
        if all(x > 0 for x in v):
            res.status, res.coefficients = "balanced", v
            res.message = "The balanced coefficients are unique."
            return res
        res.status = "no_positive"
        zero = [species[i].text for i, x in enumerate(v) if x == 0]
        if zero:
            res.message = ("No balancing uses every species: " + ", ".join(zero) +
                           " would get a zero coefficient.")
        else:
            wrong = [species[i].text for i, x in enumerate(v) if x < 0]
            res.message = ("The equation balances only with negative coefficients for " +
                           ", ".join(wrong) + " — they probably belong on the other side.")
        res.basis = [v]
        return res
    res.status = "not_unique"
    best = _positive_combination(res.basis)
    res.message = (f"The coefficients are not unique: {len(basis)} independent reactions "
                   "are combined in this equation.")
    if best is not None:
        res.coefficients = best
        res.message += (" The smallest all-positive combination is shown; other ratios are "
                        "equally balanced.")
    else:
        res.message += " No all-positive combination with small multipliers was found."
    return res


def _positive_combination(basis: list[list[int]], limit: int = 8) -> list[int] | None:
    if len(basis) > 4:
        return None
    best, best_sum = None, None
    n = len(basis[0])
    for weights in itertools.product(range(-limit, limit + 1), repeat=len(basis)):
        if not any(weights):
            continue
        v = [sum(w * b[i] for w, b in zip(weights, basis)) for i in range(n)]
        if all(x < 0 for x in v):
            v = [-x for x in v]
        if all(x > 0 for x in v):
            g = 0
            for x in v:
                g = math.gcd(g, x)
            v = [x // g for x in v]
            if best is None or sum(v) < best_sum:
                best, best_sum = v, sum(v)
    return best


# --- stoichiometry -------------------------------------------------------------------------

MASS_UNITS = {"g": 1.0, "mg": 1e-3, "kg": 1e3, "µg": 1e-6}
AMOUNT_UNITS = {"mol": 1.0, "mmol": 1e-3, "µmol": 1e-6, "kmol": 1e3}
SOLUTION_UNITS = {"L (solution)": 1.0, "mL (solution)": 1e-3, "µL (solution)": 1e-6}
GAS_UNITS = {"L (gas)": 1.0, "mL (gas)": 1e-3, "m³ (gas)": 1e3}


@dataclass
class Amount:
    """A given quantity: mass, moles, solution volume (needs ``molarity``) or gas volume
    (ideal gas at ``temperature`` K and ``pressure`` atm, default 0 °C / 1 atm)."""

    value: float
    unit: str = "g"
    molarity: float | None = None
    temperature: float = 273.15
    pressure: float = 1.0

    def moles(self, molar_mass: float) -> float:
        u = self.unit
        if u in MASS_UNITS:
            return self.value * MASS_UNITS[u] / molar_mass
        if u in AMOUNT_UNITS:
            return self.value * AMOUNT_UNITS[u]
        if u in SOLUTION_UNITS:
            if not self.molarity:
                raise ValueError("A solution volume needs a molarity (mol/L).")
            return self.value * SOLUTION_UNITS[u] * self.molarity
        if u in GAS_UNITS:
            return self.pressure * self.value * GAS_UNITS[u] / (R_GAS * self.temperature)
        raise ValueError(f"Unknown unit '{u}'")


@dataclass
class StoichRow:
    index: int
    species: str
    role: str
    coefficient: int
    molar_mass: float
    given_mol: float | None
    reacting_mol: float        # consumed (reactants) or formed (products) at full extent
    reacting_g: float
    remaining_mol: float | None
    remaining_g: float | None
    actual_g: float | None = None
    percent_yield: float | None = None
    limiting: bool = False


@dataclass
class StoichResult:
    rows: list[StoichRow]
    extent: float              # mol of reaction ξ
    limiting: list[int]
    basis: str                 # "reactants" | "product"
    notes: list[str] = field(default_factory=list)


def stoichiometry(result: BalanceResult, amounts: dict, actual_g: dict | None = None,
                  molar_masses: dict | None = None) -> StoichResult:
    """Limiting reagent and theoretical yields for a balanced equation.

    ``amounts`` maps species index → :class:`Amount` (or a number in grams). Given reactants
    determine ξ = min(nᵢ/νᵢ); if only a product amount is given, the reactants needed to make
    it are computed instead. ``actual_g`` maps product index → isolated mass (g)."""
    if result.coefficients is None:
        raise ValueError("Balance the equation first.")
    coefs = result.coefficients
    masses = {i: (molar_masses or {}).get(i) or s.formula.molar_mass
              for i, s in enumerate(result.species)}
    given = {}
    for i, a in amounts.items():
        if a is None:
            continue
        a = a if isinstance(a, Amount) else Amount(float(a), "g")
        n = a.moles(masses[i])
        if not (n > 0):
            raise ValueError(f"The amount of {result.species[i].text} must be positive.")
        given[i] = n
    if not given:
        raise ValueError("Enter an amount for at least one species.")
    react = [i for i in given if result.species[i].side > 0]
    notes = []
    if react:
        ratios = {i: given[i] / coefs[i] for i in react}
        extent = min(ratios.values())
        limiting = [i for i in react if math.isclose(ratios[i], extent, rel_tol=1e-9)]
        basis = "reactants"
        missing = [result.species[i].text for i in result.reactants if i not in given]
        if missing:
            notes.append("No amount given for " + ", ".join(missing) +
                         "; assumed to be in excess.")
        ignored = [result.species[i].text for i in given if result.species[i].side < 0]
        if ignored:
            notes.append("Product amounts are ignored when reactant amounts are given "
                         "(use them as actual yields instead).")
    else:
        i = next(iter(given))
        extent = given[i] / coefs[i]
        limiting, basis = [], "product"
        notes.append(f"Amounts of reactants required to obtain the given amount of "
                     f"{result.species[i].text} (100 % yield).")
    rows = []
    for i, s in enumerate(result.species):
        nu = coefs[i]
        n_react = nu * extent
        rem = None
        if s.side > 0 and i in given:
            rem = max(given[i] - n_react, 0.0)
            if rem < 1e-12 * max(given[i], 1e-300):
                rem = 0.0
        row = StoichRow(i, s.text, s.role, nu, masses[i], given.get(i), n_react,
                        n_react * masses[i], rem, None if rem is None else rem * masses[i],
                        limiting=i in limiting)
        if s.side < 0 and actual_g and actual_g.get(i):
            row.actual_g = float(actual_g[i])
            row.percent_yield = 100.0 * row.actual_g / row.reacting_g
        rows.append(row)
    return StoichResult(rows, extent, limiting, basis, notes)
