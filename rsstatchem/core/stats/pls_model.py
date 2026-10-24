"""PLS-SEM model specification: constructs (measurement models), structural paths, a
lavaan/seminr-like text syntax and validation (as SmartPLS checks a model before running).

Syntax (one statement per line or separated by ``;``, ``#`` starts a comment)::

    X =~ X1 + X2 + X3        reflective construct (Mode A)
    F <~ F1 + F2             formative construct (Mode B)
    Y ~ X + M                structural paths X -> Y and M -> Y
    X -> M -> Y              path chains are accepted as well
    Y ~ X + W + X:W          moderation: interaction term X x W (two-stage approach)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

REFLECTIVE, FORMATIVE, INTERACTION = "A", "B", "I"
MODE_TEXT = {REFLECTIVE: "Reflective", FORMATIVE: "Formative",
             INTERACTION: "Interaction (two-stage)"}

_NAME = re.compile(r"^[A-Za-z_À-￿][\w.À-￿]*$")


class ModelError(ValueError):
    """Raised for syntax errors and invalid PLS path models."""


@dataclass
class Construct:
    name: str
    indicators: list[str] = field(default_factory=list)
    mode: str = REFLECTIVE

    @property
    def reflective(self) -> bool:
        return self.mode == REFLECTIVE


@dataclass
class PLSModel:
    constructs: list[Construct] = field(default_factory=list)
    paths: list[tuple[str, str]] = field(default_factory=list)  # (source, target)
    interactions: list[tuple[str, str, str]] = field(default_factory=list)  # (x, w, target)

    # ------------------------------------------------------------------ queries
    def names(self) -> list[str]:
        return [c.name for c in self.constructs]

    def construct(self, name: str) -> Construct:
        key = name.lower()
        for c in self.constructs:
            if c.name.lower() == key:
                return c
        raise KeyError(name)

    def has(self, name: str) -> bool:
        return any(c.name.lower() == name.lower() for c in self.constructs)

    def predecessors(self, name: str) -> list[str]:
        return [s for s, t in self.paths if t == name]

    def successors(self, name: str) -> list[str]:
        return [t for s, t in self.paths if s == name]

    def endogenous(self) -> list[str]:
        return [c.name for c in self.constructs if self.predecessors(c.name)]

    def exogenous(self) -> list[str]:
        return [c.name for c in self.constructs if not self.predecessors(c.name)]

    def topological_order(self) -> list[str]:
        """Constructs ordered so every path points forward; raises for a non-recursive model."""
        indeg = {c.name: 0 for c in self.constructs}
        for _, t in self.paths:
            indeg[t] += 1
        order, ready = [], [n for n in self.names() if indeg[n] == 0]
        while ready:
            n = ready.pop(0)
            order.append(n)
            for t in self.successors(n):
                indeg[t] -= 1
                if indeg[t] == 0:
                    ready.append(t)
        if len(order) != len(self.constructs):
            cyc = [n for n in self.names() if n not in order]
            raise ModelError("The structural model is not recursive (it contains a feedback "
                             "loop involving " + ", ".join(cyc) + "). PLS-SEM requires a "
                             "recursive model.")
        return order

    def layers(self) -> dict[str, int]:
        """Longest-path depth of each construct (0 = exogenous), used by the path diagram."""
        depth = {}
        for n in self.topological_order():
            preds = self.predecessors(n)
            depth[n] = 1 + max(depth[p] for p in preds) if preds else 0
        return depth

    def sorted(self) -> "PLSModel":
        """Copy with constructs in topological order (stable for ties)."""
        order = self.topological_order()
        return PLSModel([self.construct(n) for n in order], list(self.paths),
                        list(self.interactions))

    @staticmethod
    def interaction_name(x: str, w: str) -> str:
        return f"{x} x {w}"

    def main_model(self) -> "PLSModel":
        """The main-effects model (stage 1 of the two-stage moderation approach)."""
        return PLSModel(list(self.constructs), list(self.paths))

    def stage2_model(self) -> "PLSModel":
        """Main model plus a single-item construct for every interaction term."""
        cons, paths = list(self.constructs), list(self.paths)
        for x, w, t in self.interactions:
            name = self.interaction_name(x, w)
            if not any(c.name == name for c in cons):
                cons.append(Construct(name, [name], INTERACTION))
            paths.append((name, t))
        return PLSModel(cons, paths)

    # ------------------------------------------------------------------ syntax
    def to_syntax(self) -> str:
        lines = []
        for c in self.constructs:
            op = "=~" if c.reflective else "<~"
            lines.append(f"{c.name} {op} " + " + ".join(c.indicators))
        for c in self.constructs:
            preds = self.predecessors(c.name) + [f"{x}:{w}" for x, w, t in self.interactions
                                                 if t == c.name]
            if preds:
                lines.append(f"{c.name} ~ " + " + ".join(preds))
        return "\n".join(lines)

    def validate(self, available: list[str] | None = None,
                 numeric: set[str] | None = None) -> list[str]:
        """Check the model; raises ModelError. Returns a list of warnings.

        ``available`` are the dataset's variable names (matched case-insensitively and
        rewritten to their dataset spelling); ``numeric`` the numeric ones.
        """
        warnings: list[str] = []
        if not self.constructs:
            raise ModelError("Define at least two constructs (latent variables).")
        seen = set()
        for c in self.constructs:
            if not c.name or not _NAME.match(c.name):
                raise ModelError(f"'{c.name}' is not a valid construct name (use letters, "
                                 "digits and underscores, starting with a letter).")
            if c.name.lower() in seen:
                raise ModelError(f"Construct '{c.name}' is defined more than once.")
            seen.add(c.name.lower())
            if c.mode not in (REFLECTIVE, FORMATIVE):
                raise ModelError(f"Unknown measurement mode for '{c.name}'.")
            if not c.indicators:
                raise ModelError(f"Construct '{c.name}' has no indicators.")
            if len({i.lower() for i in c.indicators}) != len(c.indicators):
                raise ModelError(f"Construct '{c.name}' lists an indicator twice.")
        if available is not None:
            lookup = {n.lower(): n for n in available}
            for c in self.constructs:
                fixed = []
                for ind in c.indicators:
                    real = lookup.get(ind.lower())
                    if real is None:
                        raise ModelError(f"Indicator '{ind}' of construct '{c.name}' is not a "
                                         "variable in the active dataset.")
                    if numeric is not None and real not in numeric:
                        raise ModelError(f"Indicator '{real}' is not numeric.")
                    fixed.append(real)
                c.indicators = fixed
        canon = {c.name.lower(): c.name for c in self.constructs}
        paths = []
        for s, t in self.paths:
            for n in (s, t):
                if n.lower() not in canon:
                    raise ModelError(f"The structural path {s} -> {t} refers to '{n}', which "
                                     "is not a construct. Define it with =~ (reflective) or "
                                     "<~ (formative).")
            s, t = canon[s.lower()], canon[t.lower()]
            if s == t:
                raise ModelError(f"A construct cannot predict itself ({s} -> {t}).")
            if (s, t) not in paths:
                paths.append((s, t))
        self.paths = paths
        inter = []
        for x, w, t in self.interactions:
            for nm in (x, w, t):
                if nm.lower() not in canon:
                    raise ModelError(f"The interaction term {x}:{w} -> {t} refers to '{nm}', "
                                     "which is not a construct.")
            x, w, t = canon[x.lower()], canon[w.lower()], canon[t.lower()]
            if x == w or t in (x, w):
                raise ModelError(f"Invalid interaction term {x}:{w} for {t}.")
            for main in (x, w):
                if (main, t) not in self.paths:
                    raise ModelError(f"Moderation {x}:{w} -> {t} needs the main effect "
                                     f"{main} -> {t} in the model (Hair et al., 2022).")
            if (x, w, t) not in inter and (w, x, t) not in inter:
                inter.append((x, w, t))
        self.interactions = inter
        if not self.paths:
            raise ModelError("Add at least one structural path between constructs.")
        self.topological_order()
        for c in self.constructs:
            if not self.predecessors(c.name) and not self.successors(c.name):
                raise ModelError(f"Construct '{c.name}' is not connected to the structural "
                                 "model.")
        used: dict[str, list[str]] = {}
        for c in self.constructs:
            for ind in c.indicators:
                used.setdefault(ind, []).append(c.name)
        for ind, owners in used.items():
            if len(owners) > 1:
                warnings.append(f"Indicator {ind} is used by several constructs "
                                f"({', '.join(owners)}), e.g. a repeated-indicators "
                                "higher-order model.")
        return warnings


def _split_terms(text: str, where: str) -> list[str]:
    terms = [t.strip() for t in re.split(r"[+,]", text)]
    if any(not t for t in terms):
        raise ModelError(f"Empty term in '{where}'.")
    for t in terms:
        if not _NAME.match(t):
            raise ModelError(f"'{t}' is not a valid name (in '{where}').")
    return terms


def parse_syntax(text: str) -> PLSModel:
    """Parse the model syntax (see module docstring)."""
    model = PLSModel()
    order: dict[str, Construct] = {}
    for raw in re.split(r"[;\n]", text):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if "=~" in line or "<~" in line:
            op = "=~" if "=~" in line else "<~"
            lhs, rhs = (s.strip() for s in line.split(op, 1))
            if not _NAME.match(lhs):
                raise ModelError(f"'{lhs}' is not a valid construct name (in '{line}').")
            mode = REFLECTIVE if op == "=~" else FORMATIVE
            inds = _split_terms(rhs, line)
            key = lhs.lower()
            if key in order:
                if order[key].mode != mode:
                    raise ModelError(f"Construct '{lhs}' is defined both as reflective and "
                                     "formative.")
                order[key].indicators += [i for i in inds if i not in order[key].indicators]
            else:
                order[key] = Construct(lhs, inds, mode)
        elif "->" in line or "<-" in line:
            if "<-" in line:
                parts = [p.strip() for p in line.split("<-")][::-1]
            else:
                parts = [p.strip() for p in line.split("->")]
            groups = [_split_terms(p, line) for p in parts]
            for a, b in zip(groups, groups[1:]):
                for s in a:
                    for t in b:
                        model.paths.append((s, t))
        elif "~" in line:
            lhs, rhs = (s.strip() for s in line.split("~", 1))
            targets = _split_terms(lhs, line)
            terms = [t.strip() for t in re.split(r"[+,]", rhs)]
            plain = [t for t in terms if ":" not in t]
            inter = [t for t in terms if ":" in t]
            plain = _split_terms(" + ".join(plain), line) if plain else []
            for t in targets:
                for s in plain:
                    model.paths.append((s, t))
                for term in inter:
                    parts = _split_terms(term.replace(":", "+"), line)
                    if len(parts) != 2:
                        raise ModelError(f"Only two-way interactions are supported ('{term}').")
                    model.interactions.append((parts[0], parts[1], t))
        else:
            raise ModelError(f"Cannot read '{line}'. Use 'C =~ x1 + x2' (reflective), "
                             "'C <~ x1 + x2' (formative) or 'Y ~ X + M' (paths).")
    model.constructs = list(order.values())
    return model


def mediation_template(x: tuple[str, list[str]], m: tuple[str, list[str]],
                       y: tuple[str, list[str]], mode: str = REFLECTIVE) -> PLSModel:
    """Simple mediation X -> M -> Y with the direct path X -> Y."""
    (xn, xi), (mn, mi), (yn, yi) = x, m, y
    return PLSModel([Construct(xn, list(xi), mode), Construct(mn, list(mi), mode),
                     Construct(yn, list(yi), mode)],
                    [(xn, mn), (mn, yn), (xn, yn)])


EXAMPLE_SYNTAX = """# Reflective (=~) and formative (<~) constructs, then structural paths (~)
X =~ X1 + X2 + X3
M =~ X4 + X5 + X6
Y =~ Y1 + Y2 + Y3 + Y4 + Y5
M ~ X
Y ~ X + M
"""
