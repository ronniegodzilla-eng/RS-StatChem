"""Combined mixture-process designs: coding and crossed Scheffé x process model terms.

Factor order in a combined design: the q mixture components first (letters A, B, C, ...),
then the process factors (the next letters). Terms are the usual tuples of factor indices
(see :mod:`rsstatchem.core.doe.model`): a mixture term (e.g. ``(0, 1)`` = AB) crossed with a
process term (e.g. ``(3,)`` = D) is the concatenated tuple ``(0, 1, 3)`` = ABD, whose model
column is the product of the L-pseudo components and the coded (-1..+1) process factors.

The crossed model of Cornell (2002, ch. 7) / Design-Expert: every term of the chosen Scheffé
model (linear, quadratic, special cubic) times every term of the chosen process model (mean,
linear, 2FI, quadratic, cubic), including the empty process term; no intercept (the linear
blending terms absorb it). The full cubic Scheffé terms (difference terms) are not crossed.

:class:`CombinedSpace` is the :class:`rsstatchem.core.doe.model.Space` of a combined design
(``kind == "combined"``: components coded as L-pseudo components, process factors to -1..+1,
no intercept); :func:`combined_space` builds it from a design dataset. The term generator
delegates to :func:`rsstatchem.core.doe.model.combined_terms`; order keys are
"mixture*process" (e.g. "quadratic*linear").
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rsstatchem.core.doe import model as mdl

MIX_ORDERS = [("Linear", "linear"), ("Quadratic", "quadratic"),
              ("Special Cubic", "special_cubic")]
PROC_ORDERS = [("Mean", "mean"), ("Linear", "linear"), ("2FI", "2fi"),
               ("Quadratic", "quadratic"), ("Cubic", "cubic")]


def mixture_part(q: int, order: str) -> list[tuple]:
    if order == "cubic":
        raise ValueError("Full cubic mixture terms are not crossed with process factors; use "
                         "special cubic.")
    return [t for t in mdl.mixture_terms(q, order) if t and t[0] != -1]


def process_part(q: int, kp: int, order: str, categoric: list[bool] | None = None
                 ) -> list[tuple]:
    """Process terms (shifted to the factor indices after the q components), with the
    empty term first."""
    cat = categoric or [False] * kp
    terms = mdl.process_terms(kp, order, cat) if order != "mean" else []
    return [()] + [tuple(i + q for i in t) for t in terms]


def combined_terms(q: int, kp: int, mix_order: str = "quadratic", proc_order: str = "linear",
                   categoric: list[bool] | None = None,
                   cross_order: str | None = None) -> list[tuple]:
    """Crossed model terms (delegates to :func:`rsstatchem.core.doe.model.combined_terms`).
    ``cross_order`` (default ``mix_order``) limits which mixture terms are crossed with the
    process terms (e.g. a quadratic mixture model with only the linear blending terms times
    the process terms)."""
    if cross_order is None or cross_order == mix_order:
        mixture_part(q, mix_order)  # clear error for the full cubic
        return mdl.combined_terms(q, kp, mix_order, proc_order, categoric)
    mterms = mixture_part(q, mix_order)
    cterms = mixture_part(q, cross_order)
    out = list(mterms)
    for pt in process_part(q, kp, proc_order, categoric)[1:]:
        for mt in cterms:
            out.append(tuple(sorted(mt + pt)))
    return mdl.sort_terms(out)


def split_term(t: tuple, q: int) -> tuple[tuple, tuple]:
    """(mixture part, process part) of a combined term."""
    return tuple(i for i in t if i < q), tuple(i for i in t if i >= q)


def term_label(t: tuple, q: int) -> str:
    m, p = split_term(t, q)
    if not p:
        return mdl.term_name(m)
    return mdl.term_name(m) + "·" + mdl.term_name(p)


def combined_matrix(coded: np.ndarray, terms: list[tuple], categoric=None, nlevels=None
                    ) -> np.ndarray:
    """Model matrix (no intercept) of the combined terms at coded settings."""
    coded = np.atleast_2d(np.asarray(coded, float))
    return mdl.raw_matrix(coded, terms, intercept=False,
                          categoric=categoric if categoric and any(categoric) else None,
                          nlevels=nlevels)


@dataclass
class CombinedSpace(mdl.Space):
    """The space of a combined design (``kind == "combined"``); the coding (L-pseudo
    components, process factors -1..+1) and the intercept-free model matrices come from
    :class:`rsstatchem.core.doe.model.Space`. Adds the number of components."""

    @property
    def n_mix(self) -> int:
        return len(self.mix_idx)


def combined_space(ds) -> CombinedSpace:
    """The coding of a combined design dataset (``meta["doe"]["kind"] == "combined"``)."""
    import dataclasses

    base = mdl.space_from_dataset(ds)
    if not base.combined:
        raise ValueError("Not a combined mixture-process design.")
    return CombinedSpace(**{f.name: getattr(base, f.name) for f in dataclasses.fields(base)})


def split_order(key: str) -> tuple[str, str]:
    """"quadratic*linear" (or "quadratic|linear") -> ("quadratic", "linear")."""
    for sep in ("*", "|"):
        if sep in key:
            m, p = key.split(sep, 1)
            return m or "quadratic", p or "linear"
    return key or "quadratic", "mean"


def model_terms_for(space: CombinedSpace, mix_order: str, proc_order: str) -> list[tuple]:
    kp = space.k - space.n_mix
    return combined_terms(space.n_mix, kp, mix_order, proc_order,
                          space.categoric[space.n_mix:])


def candidate_terms(space: CombinedSpace) -> list[tuple]:
    """Every term offered for manual selection: special cubic x quadratic (+ cubic process
    terms crossed with the linear blending terms)."""
    kp = space.k - space.n_mix
    terms = combined_terms(space.n_mix, kp, "special_cubic", "quadratic",
                           space.categoric[space.n_mix:])
    extra = combined_terms(space.n_mix, kp, "linear", "cubic", space.categoric[space.n_mix:])
    return terms + [t for t in extra if t not in terms]


__all__ = ["MIX_ORDERS", "PROC_ORDERS", "combined_terms", "combined_matrix", "CombinedSpace",
           "combined_space", "split_term", "term_label", "model_terms_for", "candidate_terms",
           "mixture_part", "process_part", "split_order"]
