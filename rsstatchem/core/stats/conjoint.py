"""Data > Orthogonal Design > Generate / Display and Analyze > Conjoint Analysis (SPSS
ORTHOPLAN, PLANCARDS and CONJOINT, Conjoint add-on), following IBM's "ORTHOPLAN Algorithms"
and "CONJOINT Algorithms" chapters.

**ORTHOPLAN** picks the smallest orthogonal main-effects plan that fits the factors and has at
least the requested number of cards, from a library of prepared plans:

* Rao–Hamming arrays over GF(q), q ∈ {2, 3, 4, 5, 7, 8, 9}: q^r runs, (q^r − 1)/(q − 1)
  columns (these are IBM's PB 2.7/2.15/2.31, PB 3.4/3.13/3.40, PB 5.6, PB 7.8 and the
  Addelman 4^5, 4^21, 8^9 and 9^10 plans up to isomorphism);
* the cyclic Plackett–Burman 2-level plans with 12, 20, 24, 36, 44, 48 and 60 runs (IBM's
  generating columns) and Addelman's 18-run 3^7 plan;
* IBM's conversions: a 4-level column becomes three 2-level columns, an 8-level column seven
  2-level columns and a 9-level column four 3-level columns (GF(p)-coordinate combinations);
  conversely one 4/8/9-level column can replace 3/7/4 columns of a 2/3-level plan (IBM's
  "8 rows, 1 column of 4 levels plus 4 columns of 2 levels" etc.);
* fewer levels by the many-to-one map i mod n (orthogonality is kept), plan columns assigned
  to factors at random (seeded), holdout cards drawn at random from the full factorial.

Every library plan is orthogonal (all level pairs of two columns occur proportionally); the
test suite checks each one.

**CONJOINT** fits, per subject, ordinary least squares on the plan's design matrix: a constant,
deviation (effect) coding for discrete factors (last level −1), the centred value for linear
factors and centred value plus its square for ideal / anti-ideal factors. Ranks and sequences
become y = n + 1 − r. Part-worths: discrete u_jk = α̂_jk (last level −Σ), linear β̂ x_k,
quadratic γ̂₁ z + γ̂₂ z²; their standard errors follow IBM; importance = 100 · range / Σ ranges
per subject, averaged. Pearson's R and Kendall's tau between observed and predicted scores
(Kendall's tau for holdouts), reversals for MORE/LESS/IDEAL/ANTIIDEAL factors, and simulation
probabilities by maximum utility, BTL (r̂_i / Σ r̂) and logit (e^r̂ / Σ e^r̂), averaged over
subjects (BTL and logit only over subjects with all predicted scores positive).
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import fmt, fmt_int
from rsstatchem.core.stats.ml_common import MLError, rng
from rsstatchem.core.stats.roc import figure, png, style

ConjointError = MLError


# ============================================================================ Galois fields
_IRRED = {4: (2, 2, [1, 1]), 8: (2, 3, [1, 1, 0]), 9: (3, 2, [2, 0])}  # x^k = Σ c_i x^i


class GF:
    """Finite field GF(q) for q prime or 4, 8, 9 (elements 0..q−1, base-p digits)."""

    def __init__(self, q: int):
        self.q = q
        if q in _IRRED:
            p, k, red = _IRRED[q]
        else:
            p, k, red = q, 1, []
        self.p, self.k = p, k
        digits = lambda a: [(a // p ** i) % p for i in range(k)]  # noqa: E731
        num = lambda d: sum(int(c) * p ** i for i, c in enumerate(d))  # noqa: E731
        self.add = np.array([[num([(x + y) % p for x, y in zip(digits(a), digits(b))])
                              for b in range(q)] for a in range(q)], int)
        mul = np.zeros((q, q), int)
        for a in range(q):
            for b in range(q):
                prod = [0] * (2 * k - 1)
                for i, x in enumerate(digits(a)):
                    for j, y in enumerate(digits(b)):
                        prod[i + j] = (prod[i + j] + x * y) % p
                for deg in range(2 * k - 2, k - 1, -1):   # reduce x^deg
                    c = prod[deg]
                    if c:
                        prod[deg] = 0
                        for i, r in enumerate(red):
                            prod[deg - k + i] = (prod[deg - k + i] + c * r) % p
                mul[a, b] = num(prod[:k])
        self.mul = mul
        self.digits = digits


def rao_hamming(q: int, r: int) -> np.ndarray:
    """OA(q^r, (q^r − 1)/(q − 1), q, 2): rows all vectors of GF(q)^r, columns the normalised
    nonzero vectors, entries their inner products."""
    F = GF(q)
    rows = np.array(list(itertools.product(range(q), repeat=r)), int)
    cols = [v for v in itertools.product(range(q), repeat=r)
            if any(v) and next(x for x in v if x) == 1]
    out = np.zeros((len(rows), len(cols)), int)
    for j, v in enumerate(cols):
        acc = np.zeros(len(rows), int)
        for i in range(r):
            acc = F.add[acc, F.mul[rows[:, i], v[i]]]
        out[:, j] = acc
    return out


def split_column(col: np.ndarray, q: int) -> np.ndarray:
    """IBM's conversion of a q = p^k level column into (p^k − 1)/(p − 1) p-level columns
    (normalised GF(p) combinations of the base-p digits)."""
    F = GF(q)
    p, k = F.p, F.k
    D = np.array([F.digits(int(v)) for v in col], int)
    combos = [a for a in itertools.product(range(p), repeat=k)
              if any(a) and next(x for x in a if x) == 1]
    return np.column_stack([(D * np.array(a)).sum(1) % p for a in combos])


def combine_columns(arr: np.ndarray, p: int, r: int, d: int) -> tuple[np.ndarray, np.ndarray]:
    """For a Rao–Hamming GF(p)^r plan: one p^d-level column from the subspace spanned by the
    first d unit vectors, plus the columns not in that subspace."""
    cols = [v for v in itertools.product(range(p), repeat=r)
            if any(v) and next(x for x in v if x) == 1]
    rows = np.array(list(itertools.product(range(p), repeat=r)), int)
    big = sum(rows[:, i] * p ** i for i in range(d))
    keep = [j for j, v in enumerate(cols) if any(v[d:])]
    return big, arr[:, keep]


PB_GENERATORS = {   # IBM's generating columns (PB 2.23 corrected to the standard generator)
    12: "11011100010", 20: "1100111101010000110", 24: "11111010110011001010000",
    36: "01011100011111011100100001010110010",
    44: "1100101001110111110001011100000100011010110",
    48: "11111011110010101110010011011000101011000010000",
    60: "11011101010010011101111001111100000110000100011011010100010",
}
L18 = ["0000000", "0112111", "0221222", "1011120", "1120201", "1202012", "2022102",
       "2101210", "2210021", "0021011", "0100122", "0212200", "1002221", "1111002",
       "1220110", "2010212", "2122020", "2201101"]


def plackett_burman(n: int) -> np.ndarray:
    g = np.array([int(c) for c in PB_GENERATORS[n]])
    cols = np.column_stack([np.roll(g, j) for j in range(len(g))])
    return np.vstack([cols, np.zeros(len(g), int)])


def is_orthogonal(arr: np.ndarray) -> bool:
    """Every pair of columns has proportional joint frequencies (orthogonal main effects)."""
    N = len(arr)
    for a in range(arr.shape[1]):
        for b in range(a + 1, arr.shape[1]):
            x, y = arr[:, a], arr[:, b]
            for u in np.unique(x):
                for v in np.unique(y):
                    obs = ((x == u) & (y == v)).sum()
                    exp = (x == u).sum() * (y == v).sum() / N
                    if abs(obs - exp) > 1e-9:
                        return False
    return True


@dataclass
class Plan:
    name: str
    columns: list[np.ndarray]           # level codes 0..s-1
    levels: list[int]

    @property
    def runs(self) -> int:
        return len(self.columns[0]) if self.columns else 0


_LIBRARY: list[Plan] | None = None


def library() -> list[Plan]:
    """All prepared plans, including the converted / combined variants."""
    global _LIBRARY
    if _LIBRARY is not None:
        return _LIBRARY
    plans: list[Plan] = []
    for q, rs in ((2, (2, 3, 4, 5, 6)), (3, (2, 3, 4)), (4, (2, 3)), (5, (2,)), (7, (2,)),
                  (8, (2,)), (9, (2,))):
        for r in rs:
            arr = rao_hamming(q, r)
            plans.append(Plan(f"GF({q})^{r}", [arr[:, j] for j in range(arr.shape[1])],
                              [q] * arr.shape[1]))
            if q in (4, 8, 9):     # convert s of the columns into p-level columns
                F = GF(q)
                for s in range(1, arr.shape[1] + 1):
                    keep = [arr[:, j] for j in range(arr.shape[1] - s)]
                    split = [c for j in range(arr.shape[1] - s, arr.shape[1])
                             for c in split_column(arr[:, j], q).T]
                    plans.append(Plan(f"GF({q})^{r} with {s} converted", keep + split,
                                      [q] * len(keep) + [F.p] * len(split)))
            if q in (2, 3):        # one p^d-level column replacing a subspace
                for d in (2, 3):
                    if d < r and q ** d <= 9:
                        big, rest = combine_columns(arr, q, r, d)
                        plans.append(Plan(f"GF({q})^{r} with a {q ** d}-level column",
                                          [big] + [rest[:, j] for j in range(rest.shape[1])],
                                          [q ** d] + [q] * rest.shape[1]))
    for n in PB_GENERATORS:
        arr = plackett_burman(n)
        plans.append(Plan(f"PB 2.{n - 1}", [arr[:, j] for j in range(arr.shape[1])],
                          [2] * arr.shape[1]))
    arr = np.array([[int(c) for c in row] for row in L18])
    plans.append(Plan("Addelman 18 x 3^7", [arr[:, j] for j in range(7)], [3] * 7))
    _LIBRARY = plans
    return plans


def _assign(plan: Plan, levels: list[int]) -> list[int] | None:
    """Column index for each factor (factors with most levels first, smallest adequate
    column), or None."""
    free = list(range(len(plan.levels)))
    out = [None] * len(levels)
    for f in sorted(range(len(levels)), key=lambda i: -levels[i]):
        cand = [c for c in free if plan.levels[c] >= levels[f]]
        if not cand:
            return None
        c = min(cand, key=lambda c: (plan.levels[c], c))
        out[f] = c
        free.remove(c)
    return out


def choose_plan(levels: list[int], minimum: int = 0) -> tuple[Plan, list[int]]:
    feas = []
    for plan in library():
        a = _assign(plan, levels)
        if a is not None:
            waste = sum(plan.levels[c] - k for c, k in zip(a, levels))
            unbalanced = sum(plan.levels[c] % k != 0 for c, k in zip(a, levels))
            simple = 0 if "with" not in plan.name else 1 if "converted" in plan.name else 2
            feas.append((plan.runs, unbalanced, waste, simple, plan.name, plan, a))
    if not feas:
        raise ConjointError("There are too many factors (or levels) for the prepared plans.")
    ok = [f for f in feas if f[0] >= minimum]
    if ok:
        best = min(ok, key=lambda f: f[:5])
    else:       # IBM: the largest plan that can be adapted
        best = max(feas, key=lambda f: (f[0], -f[1], -f[2]))
    return best[5], best[6]


@dataclass
class Factor:
    name: str
    values: list
    labels: list[str] = field(default_factory=list)
    label: str = ""


def orthoplan(factors: list[Factor], minimum: int = 0, holdout: int = 0,
              seed: int | None = None, mix_holdout: bool = False) -> tuple[pd.DataFrame, str]:
    """The plan as a DataFrame (factor value columns, STATUS_ 0 design / 1 holdout, CARD_) and
    the name of the prepared plan used."""
    if not factors:
        raise ConjointError("Define at least one factor.")
    levels = [len(f.values) for f in factors]
    if min(levels) < 2:
        raise ConjointError("Every factor needs at least two levels.")
    plan, assign = choose_plan(levels, int(minimum))
    g = rng(seed)
    # random assignment among the plan columns with the same number of levels (IBM)
    by_level: dict[int, list[int]] = {}
    for f, c in enumerate(assign):
        by_level.setdefault(plan.levels[c], []).append(f)
    final = list(assign)
    for lv, fs in by_level.items():
        pool = [c for c in range(len(plan.levels)) if plan.levels[c] == lv]
        chosen = sorted(g.choice(pool, len(fs), replace=False).tolist())
        perm = g.permutation(len(fs))
        for f, c in zip(fs, [chosen[i] for i in perm]):
            final[f] = c
    codes = np.column_stack([plan.columns[c] % k for c, k in zip(final, levels)])
    data = {f.name: [f.values[i] for i in codes[:, j]] for j, f in enumerate(factors)}
    df = pd.DataFrame(data)
    df["STATUS_"] = 0.0
    if holdout:
        full = list(itertools.product(*[range(k) for k in levels]))
        used = {tuple(r) for r in codes.tolist()}
        rest = [r for r in full if r not in used]
        take = g.permutation(len(rest))[:int(holdout)]
        hold = pd.DataFrame({f.name: [f.values[rest[i][j]] for i in take]
                             for j, f in enumerate(factors)})
        hold["STATUS_"] = 1.0
        df = pd.concat([df, hold], ignore_index=True)
        if mix_holdout:
            df = df.iloc[g.permutation(len(df))].reset_index(drop=True)
    df["CARD_"] = np.arange(1, len(df) + 1, dtype=float)
    return df, plan.name


def plan_dataset(factors: list[Factor], df: pd.DataFrame) -> Dataset:
    ds = Dataset(df[[f.name for f in factors]].copy(), name="OrthoPlan")
    ds.add_variable("STATUS_", values=df["STATUS_"].to_numpy(float))   # SPSS names keep "_"
    ds.add_variable("CARD_", values=df["CARD_"].to_numpy(float))
    for f in factors:
        v = ds.var(f.name)
        v.label = f.label
        if f.labels and any(f.labels):
            v.value_labels = {val: lab for val, lab in zip(f.values, f.labels) if lab}
        v.measure = "nominal"
        v.decimals = 0 if all(float(x).is_integer() for x in f.values
                              if isinstance(x, (int, float))) else 2
    st = ds.var("STATUS_")
    st.value_labels = {0.0: "Design", 1.0: "Holdout", 2.0: "Simulation"}
    st.decimals = 0
    st.measure = "nominal"
    ds.var("CARD_").decimals = 0
    ds.var("CARD_").label = "Card"
    return ds


def generate_design(factors: list[Factor], minimum: int = 0, holdout: int = 0,
                    seed: int | None = None, mix_holdout: bool = False
                    ) -> tuple[Dataset, Section]:
    df, name = orthoplan(factors, minimum, holdout, seed, mix_holdout)
    ds = plan_dataset(factors, df)
    section = Section("Orthogonal Design")
    section.log("ORTHOPLAN\n  /FACTORS=" + " ".join(
        f"{f.name} " + ("'" + f.label + "' " if f.label else "") + "("
        + " ".join(f"{v}" + (f" '{lab}'" if lab else "") for v, lab in
                   itertools.zip_longest(f.values, f.labels or [], fillvalue=""))
        + ")" for f in factors)
        + (f"\n  /MINIMUM={minimum}" if minimum else "")
        + (f"\n  /HOLDOUT={holdout}" if holdout else "")
        + f"\n  /MIXHOLD={'YES' if mix_holdout else 'NO'}.")
    n_design = int((df["STATUS_"] == 0).sum())
    section.table(Table("Plan Summary", ["", "Value"], [
        ["Factors", fmt_int(len(factors))], ["Design cards", fmt_int(n_design)],
        ["Holdout cards", fmt_int(int((df["STATUS_"] == 1).sum()))],
        ["Full factorial", fmt_int(int(np.prod([len(f.values) for f in factors])))],
        ["Prepared plan", name]]))
    section.table(plan_listing(ds, [f.name for f in factors]))
    return ds, section


def plan_listing(ds: Dataset, factors: list[str]) -> Table:
    st, cd = _plan_var(ds, "STATUS"), _plan_var(ds, "CARD")
    status = ds.df[st] if st else pd.Series(0.0, index=ds.df.index)
    card = ds.df[cd] if cd else pd.Series(np.arange(1, ds.n_cases + 1))
    names = {0.0: "", 1.0: "Holdout", 2.0: "Simulation"}
    rows = []
    for i in range(ds.n_cases):
        rows.append([fmt_int(card.iloc[i])] + [ds.var(f).label_for(ds.df[f].iloc[i])
                                               for f in factors]
                    + [names.get(float(status.iloc[i]), "")])
    return Table("Plan Cards", ["Card ID"] + [ds.var(f).title for f in factors]
                 + ["Status"], rows)


def display_design(ds: Dataset, factors: list[str], listing: bool = True,
                   profiles: bool = False, title: str = "", footer: str = "") -> Section:
    """PLANCARDS: the experimenter's listing and/or one profile per card for subjects."""
    if not factors:
        raise ConjointError("Select the factors to display.")
    section = Section("Display Design")
    section.log(f"PLANCARDS\n  /FACTORS={' '.join(factors)}\n  /FORMAT="
                + ("BOTH" if listing and profiles else "CARD" if profiles else "LIST")
                + (f"\n  /TITLE='{title}'" if title else "")
                + (f"\n  /FOOTER='{footer}'" if footer else "") + ".")
    if listing:
        section.table(plan_listing(ds, factors))
    if profiles:
        cd = _plan_var(ds, "CARD")
        card = ds.df[cd] if cd else pd.Series(np.arange(1, ds.n_cases + 1))
        for i in range(ds.n_cases):
            text = "\n".join(f"{ds.var(f).title}: {ds.var(f).label_for(ds.df[f].iloc[i])}"
                             for f in factors)
            section.note(f"{title + '  ' if title else ''}Card {fmt_int(card.iloc[i])}",
                         text + (f"\n\n{footer}" if footer else ""))
    return section


# ============================================================================ CONJOINT
MODELS = {"discrete": "Discrete", "linear": "Linear", "ideal": "Ideal",
          "antiideal": "Anti-ideal"}


@dataclass
class FactorModel:
    name: str
    model: str = "discrete"       # discrete | linear | ideal | antiideal
    direction: str = ""           # "" | more | less (linear factors)


def design_matrix(plan: pd.DataFrame, models: list[FactorModel]):
    """IBM's design matrix and the bookkeeping needed for part-worths."""
    n = len(plan)
    cols = [np.ones(n)]
    info = []
    for fm in models:
        x = plan[fm.name].to_numpy()
        if fm.model == "discrete":
            levels = sorted(pd.unique(x).tolist(), key=lambda v: (isinstance(v, str), v))
            m = len(levels)
            start = len(cols)
            for j in range(m - 1):
                cols.append(np.where(x == levels[j], 1.0, np.where(x == levels[-1], -1.0, 0.0)))
            info.append(("discrete", fm, levels, list(range(start, start + m - 1)), None))
        else:
            xf = x.astype(float)
            mean = xf.mean()
            start = len(cols)
            cols.append(xf - mean)
            if fm.model in ("ideal", "antiideal"):
                cols.append((xf - mean) ** 2)
                info.append((fm.model, fm, sorted(set(xf.tolist())), [start, start + 1], mean))
            else:
                info.append(("linear", fm, sorted(set(xf.tolist())), [start], mean))
    return np.column_stack(cols), info


def part_worths(beta: np.ndarray, cov: np.ndarray, info) -> tuple[list, float, float, dict]:
    """Utility rows (factor, level, utility, se), the constant, its SE (IBM) and the linear /
    quadratic coefficients {factor: (b1, b2)} of the non-discrete factors."""
    rows = []
    coefs = {}
    b0, a_vec = beta[0], np.zeros(len(beta))
    a_vec[0] = 1.0
    for kind, fm, levels, idx, mean in info:
        if kind == "discrete":
            a = beta[idx]
            V = cov[np.ix_(idx, idx)]
            for k, lv in enumerate(levels):
                if k < len(levels) - 1:
                    rows.append((fm.name, lv, float(a[k]), float(math.sqrt(max(V[k, k], 0)))))
                else:
                    var = V.sum()
                    rows.append((fm.name, lv, float(-a.sum()), float(math.sqrt(max(var, 0)))))
        elif kind == "linear":
            b = beta[idx[0]]
            vb = cov[idx[0], idx[0]]
            b0 -= b * mean
            a_vec[idx[0]] = -mean
            coefs[fm.name] = (float(b), 0.0)
            for lv in levels:
                rows.append((fm.name, lv, float(b * lv), float(abs(lv) * math.sqrt(max(vb, 0)))))
        else:
            g1s, g2 = beta[idx[0]], beta[idx[1]]
            g1 = g1s - 2 * g2 * mean
            v11, v12, v22 = cov[idx[0], idx[0]], cov[idx[0], idx[1]], cov[idx[1], idx[1]]
            vg1 = v11 - 4 * mean * v12 + 4 * mean ** 2 * v22
            cg = v12 - 2 * mean * v22
            b0 -= g1s * mean - g2 * mean ** 2
            a_vec[idx[0]], a_vec[idx[1]] = -mean, mean ** 2
            coefs[fm.name] = (float(g1), float(g2))
            for z in levels:
                u = g1 * z + g2 * z ** 2
                var = z ** 2 * vg1 + 2 * z ** 3 * cg + z ** 4 * v22
                rows.append((fm.name, z, float(u), float(math.sqrt(max(var, 0)))))
    var0 = float(a_vec @ cov @ a_vec)
    return rows, float(b0), math.sqrt(max(var0, 0)), coefs


def predict(plan: pd.DataFrame, rows: list, const: float, models: list[FactorModel],
            coefs: dict) -> np.ndarray:
    """r̂ = β̂₀ + Σ_j u_j (discrete utilities looked up, linear / quadratic from coefs)."""
    util = {(r[0], r[1]): r[2] for r in rows}
    out = np.full(len(plan), const)
    for fm in models:
        x = plan[fm.name].to_numpy()
        if fm.model == "discrete":
            out += np.array([util.get((fm.name, v), np.nan) for v in x])
        else:
            b1, b2 = coefs[fm.name]
            xf = x.astype(float)
            out += b1 * xf + b2 * xf ** 2
    return out


def responses_to_scores(R: np.ndarray, method: str, n_cards: int) -> np.ndarray:
    """Subjects × cards scores y (higher = preferred) from SEQUENCE / RANK / SCORE data."""
    R = np.asarray(R, float)
    if method == "score":
        return R
    if method == "rank":
        return n_cards + 1 - R
    if method == "sequence":       # column k holds the card number in position k
        out = np.full((R.shape[0], n_cards), np.nan)
        for s in range(R.shape[0]):
            for k, card in enumerate(R[s]):
                if np.isfinite(card) and 1 <= card <= n_cards:
                    out[s, int(card) - 1] = n_cards + 1 - (k + 1)
        return out
    raise ConjointError(f"Unknown response type {method!r}.")


def conjoint(plan: pd.DataFrame, models: list[FactorModel], responses: np.ndarray,
             method: str = "score", subject_ids: list | None = None,
             individual: bool = False, plots: bool = True, utilities_table: bool = True,
             simulations: bool = True) -> Section:
    """``plan`` rows are cards in card order (STATUS_ 0 design, 1 holdout, 2 simulation);
    ``responses`` is subjects × cards (design and holdout cards, in plan order; simulation
    cards are never rated)."""
    if not models:
        raise ConjointError("Specify at least one factor.")
    status = plan["STATUS_"].to_numpy(float) if "STATUS_" in plan else np.zeros(len(plan))
    rated = status != 2
    plan_r = plan[rated].reset_index(drop=True)
    st_r = status[rated]
    n_cards = len(plan_r)
    Y = responses_to_scores(np.atleast_2d(responses), method, n_cards)
    if Y.shape[1] != n_cards:
        raise ConjointError(f"{Y.shape[1]} responses per subject but {n_cards} rated cards in "
                            "the plan.")
    design = st_r == 0
    X, info = design_matrix(plan_r[design], models)
    p = X.shape[1]
    if design.sum() <= p - 1:
        raise ConjointError("Too few design cards to estimate the model.")
    Xd = X
    keep = [s for s in range(Y.shape[0]) if np.isfinite(Y[s, design]).all()]
    if not keep:
        raise ConjointError("No subject has complete responses.")
    Y = Y[keep]
    ids = [subject_ids[s] for s in keep] if subject_ids is not None else \
        [str(s + 1) for s in keep]
    t = len(keep)
    XtX_inv = np.linalg.pinv(Xd.T @ Xd)
    B = (XtX_inv @ Xd.T @ Y[:, design].T).T             # subjects × p
    beta = B.mean(0)
    # IBM: σ̂² = ΣΣ (r_ij − r̂_ij)² / (nt − d − l − 2q − 1) with the pooled (summary) model
    resid = Y[:, design] - (Xd @ beta)[None, :]
    df = design.sum() * t - p
    sigma2 = float((resid ** 2).sum() / df) if df > 0 else float("nan")
    cov = sigma2 * XtX_inv / t
    rows, const, se0, coefs = part_worths(beta, cov, info)
    section = Section("Conjoint Analysis")
    sub = "SUBJECT=ID" if subject_ids is not None else ""
    section.log("CONJOINT PLAN=* DATA=*\n  /FACTORS=" + " ".join(
        fm.name + ("" if fm.model == "discrete" else
                   f" ({fm.model.upper()}{' ' + fm.direction.upper() if fm.direction else ''})")
        for fm in models) + f"\n  /{method.upper()}" + (f"\n  /{sub}" if sub else "")
        + "\n  /PRINT=ALL\n  /UTILITY=*.")
    section.table(Table("Model Description", ["", "N of Levels", "Relation to Ranks or Scores"],
                        [[fm.name, fmt_int(plan_r[fm.name].nunique()),
                          MODELS[fm.model] + (f" ({fm.direction})" if fm.direction else "")]
                         for fm in models],
                        footnotes=["All factors are orthogonal." if _orth(plan_r[design],
                                                                         models) else
                                   "The factors are not orthogonal in the design cards."]))

    def subject_stats(b):
        r_s, c_s, _, cf = part_worths(b, cov * t, info)
        ranges = {}
        for fm in models:
            u = [r[2] for r in r_s if r[0] == fm.name]
            ranges[fm.name] = max(u) - min(u)
        tot = sum(ranges.values())
        imp = {k: 100 * v / tot if tot > 0 else 0.0 for k, v in ranges.items()}
        return r_s, c_s, imp, cf

    per = [subject_stats(B[s]) for s in range(t)]
    imp_avg = {fm.name: float(np.mean([pp[2][fm.name] for pp in per])) for fm in models}
    if utilities_table:
        trows = []
        prev = None
        for f, lv, u, se in rows:
            trows.append([f if f != prev else "", _lvl(plan, f, lv), fmt(u, 3), fmt(se, 3)])
            prev = f
        trows.append(["(Constant)", "", fmt(const, 3), fmt(se0, 3)])
        section.table(Table("Utilities", ["", "", "Utility Estimate", "Std. Error"], trows,
                            row_header_cols=2))
    section.table(Table("Importance Values", ["", "Averaged Importance Score"],
                        [[fm.name, fmt(imp_avg[fm.name], 3)] for fm in models],
                        footnotes=["Averaged importance score over subjects."]))
    lin = [(fm, idx) for kind, fm, lv, idx, mean in info if kind != "discrete"]
    if lin:
        crow = []
        for fm, _ in lin:
            b1, b2 = coefs[fm.name]
            if fm.model == "linear":
                crow.append([fm.name, fmt(b1, 3)])
            else:
                crow += [[f"{fm.name} (linear)", fmt(b1, 3)],
                         [f"{fm.name} (quadratic)", fmt(b2, 3)]]
        section.table(Table("Coefficients", ["", "B Coefficient Estimate"], crow))
    # correlations on the averaged (summary) data
    ybar = np.nanmean(Y, 0)
    pred_all = predict(plan_r, rows, const, models, coefs)
    r_p, p_p = stats.pearsonr(ybar[design], pred_all[design]) if design.sum() > 2 \
        else (np.nan, np.nan)
    tau, p_tau = stats.kendalltau(ybar[design], pred_all[design], variant="b")
    corr = [["Pearson's R", fmt(r_p, 3), fmt(p_p / 2, 3)],
            ["Kendall's tau", fmt(tau, 3), fmt(p_tau / 2, 3)]]
    hold = st_r == 1
    if hold.sum() >= 2:
        th, ph = stats.kendalltau(ybar[hold], pred_all[hold], variant="b")
        corr.append(["Kendall's tau for Holdouts", fmt(th, 3), fmt(ph / 2, 3)])
    section.table(Table("Correlationsᵃ", ["", "Value", "Sig."], corr,
                        footnotes=["a. Correlations between observed and estimated "
                                   "preferences (one-tailed significance)."]))
    rev_rows = []
    for kind, fm, lv, idx, mean in info:
        if kind == "linear" and fm.direction:
            sign = B[:, idx[0]]
            bad = (sign < 0) if fm.direction == "more" else (sign > 0)
            rev_rows.append([fm.name, fmt_int(int(bad.sum()))])
        elif kind in ("ideal", "antiideal"):
            q = B[:, idx[1]]
            bad = (q > 0) if kind == "ideal" else (q < 0)
            rev_rows.append([fm.name, fmt_int(int(bad.sum()))])
    if rev_rows:
        section.table(Table("Number of Reversals", ["Factor", "Reversals"], rev_rows,
                            footnotes=["Subjects whose estimate has the opposite sign to the "
                                       "expected relation."]))
    sim_rows = []
    sims = plan[status == 2].reset_index(drop=True)
    if simulations and len(sims):
        P_max, P_btl, P_log = [], [], []
        for s in range(t):
            r_s, c_s, _, cf = per[s]
            pr = predict(sims, r_s, c_s, models, cf)
            mx = (pr == pr.max()).astype(float)
            P_max.append(mx / mx.sum())
            if (pr > 0).all():
                P_btl.append(pr / pr.sum())
                e = np.exp(pr - pr.max())
                P_log.append(e / e.sum())
        P_max = np.mean(P_max, 0)
        P_btl = np.mean(P_btl, 0) if P_btl else np.full(len(sims), np.nan)
        P_log = np.mean(P_log, 0) if P_log else np.full(len(sims), np.nan)
        card = sims["CARD_"].to_numpy() if "CARD_" in sims else np.arange(1, len(sims) + 1)
        for i in range(len(sims)):
            sim_rows.append([str(i + 1), fmt_int(card[i]), f"{100 * P_max[i]:.1f}%",
                             f"{100 * P_btl[i]:.1f}%" if np.isfinite(P_btl[i]) else "",
                             f"{100 * P_log[i]:.1f}%" if np.isfinite(P_log[i]) else ""])
        section.table(Table("Preference Probabilities of Simulationsᵃ",
                            ["Card Number", "ID", "Maximum Utilityᵇ", "Bradley-Terry-Luce",
                             "Logit"], sim_rows,
                            footnotes=["a. Probabilities averaged over subjects.",
                                       "b. Including tied simulations."]))
    if plots:
        fig = figure(5.4, 3.6)
        ax = fig.add_subplot(111)
        ax.bar([fm.name for fm in models], [imp_avg[fm.name] for fm in models],
               color="#1f5fa8")
        style(ax, "Averaged Importance", "", "Importance")
        section.image("Importance Summary", png(fig), width=440)
        for fm in models:
            fr = [r for r in rows if r[0] == fm.name]
            fig = figure(4.8, 3.2)
            ax = fig.add_subplot(111)
            ax.bar([_lvl(plan, fm.name, r[1]) for r in fr], [r[2] for r in fr],
                   color="#2e7d32")
            ax.axhline(0, color="#999", lw=0.8)
            style(ax, f"Summary Utilities: {fm.name}", fm.name, "Utility")
            section.image(f"Utilities: {fm.name}", png(fig), width=380)
    if individual and t > 1:
        cols = ["Subject"] + [f"{fm.name}" for fm in models] + ["Constant", "Pearson's R",
                                                                 "Kendall's tau"]
        irows = []
        for s in range(t):
            r_s, c_s, imp, cf = per[s]
            pr = predict(plan_r, r_s, c_s, models, cf)
            rr = stats.pearsonr(Y[s, design], pr[design])[0] if np.std(Y[s, design]) > 0 \
                else np.nan
            tt = stats.kendalltau(Y[s, design], pr[design])[0]
            irows.append([str(ids[s])] + [fmt(imp[fm.name], 2) for fm in models]
                         + [fmt(c_s, 3), fmt(rr, 3), fmt(tt, 3)])
        section.table(Table("Subject Importance Values", cols, irows))
    section.result = {"utilities": rows, "constant": const, "importance": imp_avg,  # type: ignore
                      "beta": beta, "sigma2": sigma2, "B": B, "sim": sim_rows,
                      "coefs": coefs, "predicted": pred_all, "observed": ybar}
    return section


def _lvl(plan: pd.DataFrame, factor: str, value) -> str:
    labels = getattr(plan, "attrs", {}).get("labels", {}).get(factor, {})
    if value in labels:
        return str(labels[value])
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _orth(plan: pd.DataFrame, models: list[FactorModel]) -> bool:
    try:
        codes = np.column_stack([pd.factorize(plan[fm.name])[0] for fm in models])
        return is_orthogonal(codes)
    except Exception:  # noqa: BLE001
        return False


def _plan_var(ds: Dataset, base: str) -> str | None:
    for nm in (base + "_", base):
        if ds.has_name(nm):
            return ds.var(nm).name
    return None


def plan_frame(ds: Dataset, factors: list[str]) -> pd.DataFrame:
    """Factor columns, STATUS_ and CARD_ of a plan dataset, with value labels in attrs."""
    df = pd.DataFrame({f: ds.analysis_series(f) for f in factors})
    st, cd = _plan_var(ds, "STATUS"), _plan_var(ds, "CARD")
    df["STATUS_"] = pd.to_numeric(ds.df[st], errors="coerce").fillna(0) if st else 0.0
    df["CARD_"] = pd.to_numeric(ds.df[cd], errors="coerce") if cd \
        else np.arange(1, ds.n_cases + 1, dtype=float)
    df.attrs["labels"] = {f: dict(ds.var(f).value_labels) for f in factors}
    return df


def selftest() -> None:
    for plan in library()[:6]:
        assert is_orthogonal(np.column_stack(plan.columns))
    df, name = orthoplan([Factor("a", [1, 2, 3, 4]), Factor("b", [1, 2]), Factor("c", [1, 2])])
    assert len(df) == 8, name
    models = [FactorModel("a"), FactorModel("b"), FactorModel("c")]
    y = 1 + (df["a"].to_numpy() == 1) * 2.0 + df["b"].to_numpy()
    sec = conjoint(df, models, y[None, :], "score", plots=False)
    assert abs(sec.result["importance"]["c"]) < 1e-6
