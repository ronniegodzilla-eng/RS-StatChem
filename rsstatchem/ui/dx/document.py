"""The design behind an RSM window: one Design-Expert-style design and its ``.rsx`` file.

No Qt here. The design layout is a :class:`Dataset` built by
:func:`rsstatchem.core.doe.designs.build_design` (columns Std, Run, Block, Space_Type, the
factors in actual units and the responses) with the design definition in ``ds.meta["doe"]``.
The rows keep their positions for the life of the document (sorting only changes the view),
so row indices are stable identifiers for undo and for ignored runs.

The ``.rsx`` file is JSON: format tag, version, the variables, the column data, the design
meta (factors, responses, analysis choices), notes, ignored runs, optimisation criteria and
display options.

Design Tools edits live here too: augmenting (foldover, centre points, axial points that turn
a factorial into a central composite design, replicates, optimal augmentation for a new model
in a new block), Modify Design Space (new factor ranges), the constraints of an optimal
design, rounding factor values, inserting and deleting runs, adding responses and naming
blocks. Structural edits are undone through :meth:`snapshot` / :meth:`restore`.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from rsstatchem.core.dataset import Dataset, Variable
from rsstatchem.core.doe import designs, model

RSX_FORMAT = "rs-statchem-rsx"
RSX_VERSION = 1


class DesignDocument:
    def __init__(self, ds: Dataset, build_notes: list[str] | None = None):
        self.ds = ds
        self.build_notes = list(build_notes or [])
        self.notes = ""
        self.path: str | None = None
        self.modified = False
        self.version = 0                     # bumped whenever run data change
        self.ignored: set[int] = set()       # row indices excluded from the analyses
        self.optimization: dict = {}         # remembered criteria (JSON-able)
        self.ui: dict = {}                   # display options saved with the design
        self.last_solutions: list = []       # numerical optimisation results (not saved)
        self._listeners: list = []
        self._cache: dict = {}

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_spec(cls, spec: dict) -> "DesignDocument":
        res = designs.build_design(spec)
        doc = cls(res.dataset, res.notes)
        doc.modified = True
        return doc

    @classmethod
    def from_data(cls, frame: pd.DataFrame, factor_cols: list[str], response_cols: list[str],
                  kind: str = "process", factor_info: dict | None = None,
                  total: float | None = None, name: str = "Historical data"
                  ) -> "DesignDocument":
        """Historical data / Import Data Set: a design from existing runs."""
        res = designs.design_from_data(frame, factor_cols, response_cols, kind, factor_info,
                                       total, name)
        doc = cls(res.dataset, res.notes)
        doc.modified = True
        return doc

    # ------------------------------------------------------------------ observers
    def subscribe(self, callback) -> None:
        """``callback(kind, detail)``: kind is "data", "meta", "analysis" or "notes"."""
        self._listeners.append(callback)

    def unsubscribe(self, callback) -> None:
        if callback in self._listeners:
            self._listeners.remove(callback)

    def notify(self, kind: str, detail=None) -> None:
        for cb in list(self._listeners):
            cb(kind, detail)

    def touch(self, kind: str = "meta", detail=None) -> None:
        self.modified = True
        if kind == "data":
            self.version += 1
            self._cache.clear()
        self.notify(kind, detail)

    # ------------------------------------------------------------------ structure
    @property
    def meta(self) -> dict:
        return self.ds.meta.setdefault("doe", {})

    @property
    def name(self) -> str:
        return Path(self.path).stem if self.path else (self.ds.name or "Untitled")

    @property
    def factors(self) -> list[dict]:
        return self.meta.get("factors", [])

    @property
    def responses(self) -> list[dict]:
        return self.meta.get("responses", [])

    @property
    def factor_cols(self) -> list[str]:
        return [f["column"] for f in self.factors]

    @property
    def response_cols(self) -> list[str]:
        return [r["column"] for r in self.responses]

    @property
    def n_runs(self) -> int:
        return self.ds.n_cases

    @property
    def mixture(self) -> bool:
        return self.meta.get("kind") == "mixture"

    @property
    def combined(self) -> bool:
        """Combined mixture-process design (components first, then process factors)."""
        return self.meta.get("kind") == "combined"

    @property
    def n_mix(self) -> int:
        return sum(1 for f in self.factors if f.get("type") == "mixture")

    @property
    def constraints(self) -> list[dict]:
        return list(self.meta.get("constraints") or [])

    @property
    def editable_factors(self) -> bool:
        """Historical / imported designs: the factor settings can be typed in."""
        return bool(self.meta.get("editable_factors"))

    @property
    def n_blocks(self) -> int:
        return int(self.meta.get("blocks", 1) or 1)

    @property
    def group_col(self) -> str | None:
        """The whole-plot (Group) column of a split-plot design, else None."""
        col = (self.meta.get("columns") or {}).get("group")
        return col if col and self.ds.has_name(col) else None

    @property
    def split_plot(self) -> bool:
        return self.group_col is not None and any(f.get("changes") == "hard"
                                                  for f in self.factors)

    def factor_title(self, j: int) -> str:
        return f"{designs.letter(j)}:{self.factors[j]['name']}"

    def response_title(self, i: int) -> str:
        return f"R{i + 1}:{self.responses[i]['name']}"

    def response_index(self, col: str) -> int:
        return self.response_cols.index(col)

    def column_title(self, col: str) -> str:
        if col in self.response_cols:
            return self.response_title(self.response_index(col))
        if col in self.factor_cols:
            return self.factor_title(self.factor_cols.index(col))
        return {"Space_Type": "Space Type", "Group": "Group"}.get(col, col)

    def space(self) -> model.Space:
        if self.combined:
            from rsstatchem.core.doe import terms_combined

            return terms_combined.combined_space(self.ds)
        return model.space_from_dataset(self.ds)

    # ------------------------------------------------------------------ run data
    def value(self, row: int, col: str) -> float:
        v = self.ds.df.at[row, col]
        return float(v) if v is not None and not pd.isna(v) else float("nan")

    def set_values(self, changes: list[tuple[int, str, float]]) -> None:
        """Apply (row, column, value) changes (NaN clears a cell) and notify once."""
        cols = set()
        for row, col, val in changes:
            self.ds.df.at[row, col] = float(val) if val is not None else float("nan")
            cols.add(col)
        if changes:
            self.touch("data", sorted(cols))

    def set_ignored(self, rows, ignored: bool) -> None:
        rows = set(rows)
        self.ignored = (self.ignored | rows) if ignored else (self.ignored - rows)
        self.touch("data", list(self.response_cols))

    def response_count(self, col: str) -> int:
        vals = self.ds.df[col].to_numpy(float)
        keep = np.ones(len(vals), bool)
        keep[list(self.ignored)] = False
        return int(np.sum(np.isfinite(vals) & keep))

    def coded_matrix(self) -> np.ndarray:
        sp = self.space()
        actual = self.ds.df[self.factor_cols].to_numpy(float)
        return sp.to_coded(actual) if len(actual) else np.zeros((0, sp.k))

    def row_order(self, by: str = "run") -> list[int]:
        col = {"std": "Std", "run": "Run"}[by]
        return [int(i) for i in np.argsort(self.ds.df[col].to_numpy(float), kind="stable")]

    def randomize(self, seed: int | None = None) -> None:
        """New run order: random within blocks; a split-plot keeps its restricted
        randomisation (groups in random order, runs random within each group)."""
        rng = np.random.default_rng(seed)
        blocks = self.ds.df["Block"].to_numpy(float)
        groups = self.ds.df[self.group_col].to_numpy(float) if self.group_col else None
        self.ds.df["Run"] = designs.randomized_run_order(blocks, groups, rng).astype(float)
        self.meta["randomized"] = True
        self.touch("data", [])

    def two_level(self) -> bool:
        """Every factor at two levels (numeric -1/+1, centre points allowed, or a two-level
        categoric factor): the design can be folded over."""
        if self.mixture or self.split_plot or not self.factors:
            return False
        coded = self.coded_matrix()
        numeric = []
        for j, f in enumerate(self.factors):
            if f.get("type") == "categoric":
                if len(f.get("levels", [])) != 2:
                    return False
            else:
                numeric.append(j)
        if not numeric:
            return True
        # every run is a factorial point (all numeric factors at -1/+1) or a centre point
        # (all at 0); a Box-Behnken or CCD run mixes 0 and +-1 (or +-alpha) and is not
        m = coded[:, numeric]
        factorial = np.all(np.isclose(np.abs(m), 1), axis=1)
        centre = np.all(np.isclose(m, 0), axis=1)
        return bool(np.all(factorial | centre) and factorial.any())

    def foldover(self, kind: str = "full", factor: int | None = None,
                 keep_factor: int | None = None, keep_level: float = 1.0,
                 seed: int | None = None) -> int:
        """Design Tools > Augment Design > Foldover: append the full, single-factor or semifold
        foldover of the factorial runs as a new block (empty responses). Returns the number
        of runs added."""
        if not self.two_level():
            raise ValueError("Foldover needs a two-level design (no mixture, split-plot or "
                             "multilevel factors).")
        sp = self.space()
        coded = self.coded_matrix().copy()
        for j, f in enumerate(sp.factors):
            if f.categoric:
                coded[:, j] = 2 * coded[:, j] - 1
        new = designs.foldover_runs(coded, kind, factor, keep_factor, keep_level)
        if not len(new):
            raise ValueError("The foldover adds no runs.")
        df = self.ds.df
        m = len(new)
        block = float(np.nanmax(df["Block"].to_numpy(float))) + 1
        rng = np.random.default_rng(seed)
        rows = {"Std": np.nanmax(df["Std"].to_numpy(float)) + 1 + np.arange(m),
                "Run": np.nanmax(df["Run"].to_numpy(float)) + 1 + rng.permutation(m),
                "Block": np.full(m, block)}
        if "Space_Type" in df:
            rows["Space_Type"] = np.ones(m)
        for j, f in enumerate(sp.factors):
            if f.categoric:
                rows[f.column] = np.asarray(f.levels, float)[(new[:, j] > 0).astype(int)]
            else:
                rows[f.column] = f.center + new[:, j] * f.half
        add = pd.DataFrame({c: rows.get(c, np.full(m, np.nan)) for c in df.columns})
        self.ds.df = pd.concat([df, add], ignore_index=True)
        var = self.ds.var("Block")
        var.value_labels = dict(var.value_labels or {})
        var.value_labels[block] = f"Block {block:g}"
        meta = self.meta
        meta["runs"] = self.ds.n_cases
        meta["blocks"] = int(block)
        label = {"full": "Full foldover", "single": "Single-factor foldover on "
                 + designs.letter(factor if factor is not None else 0),
                 "semifold": "Semifold on " + designs.letter(factor if factor is not None
                                                             else 0)}[kind]
        if kind == "semifold":
            label += f" keeping {designs.letter(keep_factor)}" \
                     f"{'+' if keep_level > 0 else '-'}"
        meta.setdefault("foldovers", []).append({"kind": kind, "factor": factor,
                                                 "keep_factor": keep_factor,
                                                 "keep_level": keep_level, "runs": m,
                                                 "block": int(block), "label": label})
        words = meta.get("defining_relation") or []
        if len(words) > 1:
            group = []
            for w in words[1:]:
                sign = -1 if w.startswith("-") else 1
                mask = 0
                for ch in w.lstrip("-"):
                    mask ^= 1 << designs.LETTERS.index(ch)
                group.append((mask, sign))
            new_group = designs.folded_group(group, kind, factor)
            if new_group is None:
                meta["aliases"] = []
                meta["resolution"] = None
                meta["alias_note"] = "Semifold: an irregular fraction; check the aliasing " \
                                     "with Design Evaluation."
            else:
                k = len(self.factors)
                meta["defining_relation"] = ["I"] + [("-" if s < 0 else "") +
                                                     designs.word_str(w) for w, s in new_group]
                meta["resolution"] = designs.resolution(new_group)
                meta["aliases"] = designs.alias_structure(k, new_group)
        elif meta.get("min_run") or meta.get("design") == "pb":
            meta["alias_note"] = f"{label} added; check the aliasing with Design Evaluation."
        self.version += 1
        self._cache.clear()
        self.touch("meta")
        self.touch("data", list(self.response_cols))
        return m

    # ------------------------------------------------------------------ snapshots (undo)
    def snapshot(self) -> dict:
        """Everything a structural edit can change (for undo)."""
        import copy

        return {"df": self.ds.df.copy(), "variables": [Variable.from_dict(v.to_dict())
                                                         for v in self.ds.variables],
                "meta": copy.deepcopy(self.ds.meta.get("doe", {})),
                "ignored": set(self.ignored)}

    def restore(self, snap: dict) -> None:
        import copy

        self.ds.df = snap["df"].copy()
        self.ds.variables = [Variable.from_dict(v.to_dict()) for v in snap["variables"]]
        for v in self.ds.variables:
            v.value_labels = {float(k): lab for k, lab in (v.value_labels or {}).items()}
        self.ds.meta["doe"] = copy.deepcopy(snap["meta"])
        self.ignored = set(snap["ignored"])
        self._structure_changed()

    def _structure_changed(self) -> None:
        self.meta["runs"] = self.ds.n_cases
        if "Block" in self.ds.df:
            b = self.ds.df["Block"].to_numpy(float)
            self.meta["blocks"] = int(len(np.unique(b[np.isfinite(b)]))) or 1
        self.version += 1
        self._cache.clear()
        self.touch("meta")
        self.touch("data", list(self.response_cols))

    # ------------------------------------------------------------------ models and regions
    def model_terms(self, order: str) -> list:
        """Terms of a model order; combined designs use "mixture|process" keys and
        "design" is the custom term list stored with an optimal design."""
        sp = self.space()
        if order == "design" and self.meta.get("terms"):
            return [model.key_term(k) for k in self.meta["terms"]]
        if self.combined:
            from rsstatchem.core.doe import terms_combined as tc

            mix, proc = tc.split_order(order or "quadratic*linear")
            return tc.model_terms_for(sp, mix, proc)
        return model.model_terms(sp, order)

    def region(self):
        from rsstatchem.core.doe import evaluation

        return evaluation.region_for(self.space(), self.constraints,
                                     (self.meta.get("mixture") or {}).get(
                                         "process_constraints"))

    # ------------------------------------------------------------------ appending runs
    def _append_runs(self, coded: np.ndarray, block, types, label: str, kind: str,
                     seed: int | None = None, new_block: bool = True, **info) -> int:
        """Append runs given in coded units (empty responses). ``block``: a label for all
        new runs (or an array). Returns the number added."""
        coded = np.atleast_2d(np.asarray(coded, float))
        m = len(coded)
        if not m:
            raise ValueError("No runs to add.")
        sp = self.space()
        actual = sp.to_actual(coded)
        df = self.ds.df
        rng = np.random.default_rng(seed)
        blocks = np.broadcast_to(np.asarray(block, float), (m,)).astype(float)
        run0 = np.nanmax(df["Run"].to_numpy(float)) if len(df) else 0
        std0 = np.nanmax(df["Std"].to_numpy(float)) if len(df) else 0
        rows = {"Std": std0 + 1 + np.arange(m), "Run": run0 + 1 + rng.permutation(m),
                "Block": blocks}
        if "Space_Type" in df:
            rows["Space_Type"] = np.broadcast_to(np.asarray(types, float), (m,))
        for j, f in enumerate(sp.factors):
            rows[f.column] = actual[:, j]
        if self.group_col:
            rows[self.group_col] = np.nanmax(df[self.group_col].to_numpy(float)) + 1 + \
                np.arange(m)
        add = pd.DataFrame({c: rows.get(c, np.full(m, np.nan)) for c in df.columns})
        self.ds.df = pd.concat([df, add], ignore_index=True)
        var = self.ds.var("Block")
        var.value_labels = dict(var.value_labels or {})
        for b in sorted(set(blocks.tolist())):
            var.value_labels.setdefault(b, f"Block {b:g}")
        if "Space_Type" in df:
            st = self.ds.var("Space_Type")
            st.value_labels = dict(st.value_labels or {})
            for t in set(np.atleast_1d(types).tolist()):
                st.value_labels[float(t)] = designs.SPACE_TYPES.get(int(t), "")
        self.meta.setdefault("augments", []).append(
            {"kind": kind, "runs": m, "label": label,
             "block": int(blocks[0]) if len(set(blocks.tolist())) == 1 else None, **info})
        self._structure_changed()
        return m

    def next_block(self) -> float:
        return float(np.nanmax(self.ds.df["Block"].to_numpy(float))) + 1

    def last_block(self) -> float:
        return float(np.nanmax(self.ds.df["Block"].to_numpy(float)))

    def _process_check(self, what: str) -> None:
        if self.mixture or self.combined:
            raise ValueError(f"{what} is for process (non-mixture) designs.")
        if self.split_plot:
            raise ValueError(f"{what} is not available for split-plot designs.")

    def add_center_points(self, n: int, new_block: bool = False,
                          seed: int | None = None) -> int:
        """Augment: ``n`` centre points (numeric factors at 0; at every categoric level
        combination; the overall centroid of a mixture)."""
        if n < 1:
            raise ValueError("Add at least one centre point.")
        if self.split_plot:
            raise ValueError("Centre points cannot be added to a split-plot design here.")
        sp = self.space()
        if self.mixture or self.combined:
            reg = self.region()
            cen = np.zeros(sp.k)
            cen[reg.cont] = reg.centroid()
            pts = np.tile(cen, (n, 1))
            t = 7
        else:
            cat = [j for j, f in enumerate(sp.factors) if f.categoric]
            combos = designs._level_combos([len(sp.factors[j].levels) for j in cat])
            pts = []
            for combo in combos:
                for _ in range(n):
                    row = np.zeros(sp.k)
                    row[cat] = combo
                    pts.append(row)
            pts, t = np.array(pts), 0
        block = self.next_block() if new_block else self.last_block()
        added = self._append_runs(pts, block, t, f"{n} centre point(s)", "center", seed)
        self.meta["center_points"] = int(self.meta.get("center_points", 0) or 0) + added
        return added

    def default_alpha(self) -> float:
        """Rotatable alpha for the factorial points present (nF^(1/4))."""
        coded = self.coded_matrix()
        sp = self.space()
        num = [j for j, f in enumerate(sp.factors) if not f.categoric]
        if not num:
            return 1.0
        fact = coded[np.all(np.isclose(np.abs(coded[:, num]), 1), axis=1)][:, num]
        nf = len(np.unique(np.round(fact, 8), axis=0)) or 2 ** len(num)
        return float(nf ** 0.25)

    def add_axial_points(self, alpha: float | None = None, center_points: int = 2,
                         new_block: bool = True, seed: int | None = None) -> int:
        """Augment a factorial to a central composite design: 2k axial points at +-alpha
        (crossed with the categoric levels) plus centre points, in a new block."""
        self._process_check("Adding axial points")
        sp = self.space()
        num = [j for j, f in enumerate(sp.factors) if not f.categoric]
        cat = [j for j in range(sp.k) if j not in num]
        if len(num) < 1:
            raise ValueError("Axial points need numeric factors.")
        alpha = float(alpha) if alpha else self.default_alpha()
        if alpha <= 0:
            raise ValueError("Alpha must be positive.")
        ax_num = designs.axial_runs(len(num), alpha)
        rows, types = [], []
        for combo in designs._level_combos([len(sp.factors[j].levels) for j in cat]):
            for a in ax_num:
                row = np.zeros(sp.k)
                row[num] = a
                row[cat] = combo
                rows.append(row)
                types.append(2)
            for _ in range(int(center_points)):
                row = np.zeros(sp.k)
                row[cat] = combo
                rows.append(row)
                types.append(0)
        block = self.next_block() if new_block else self.last_block()
        added = self._append_runs(np.array(rows), block, np.array(types),
                                  f"Axial points (alpha {alpha:.4g}) + {center_points} "
                                  "centre point(s)", "axial", seed, alpha=alpha)
        self.meta.update({"alpha": alpha, "alpha_type": "augmented", "ccd_type":
                          "augmented factorial", "model": "quadratic"})
        self.meta["center_points"] = int(self.meta.get("center_points", 0) or 0) + \
            int(center_points) * max(1, len(designs._level_combos(
                [len(sp.factors[j].levels) for j in cat])))
        return added

    def add_replicates(self, rows: list[int] | None = None, new_block: bool = True,
                       seed: int | None = None) -> int:
        """Augment: replicate the given runs (default every run) as new runs."""
        if self.split_plot:
            raise ValueError("Replicate a split-plot design by building it with replicates.")
        rows = list(range(self.n_runs)) if rows is None else sorted(set(rows))
        coded = self.coded_matrix()[rows]
        if not np.all(np.isfinite(coded)):
            raise ValueError("Some runs have no factor settings.")
        types = self.ds.df["Space_Type"].to_numpy(float)[rows] if "Space_Type" in self.ds.df \
            else 8
        if new_block:
            block = self.next_block()
        else:
            block = self.ds.df["Block"].to_numpy(float)[rows]
        added = self._append_runs(coded, block, types, f"Replicates of {len(rows)} run(s)",
                                  "replicates", seed)
        self.meta["replicates"] = int(self.meta.get("replicates", 1) or 1) + (
            1 if len(rows) == self.n_runs - added else 0)
        return added

    def augment_default_runs(self, terms) -> int:
        """Suggested number of new runs for optimal augmentation: enough to estimate the
        new model, plus 4 for error degrees of freedom."""
        sp = self.space()
        coded = self.coded_matrix()
        mat = model.build_matrix(sp, coded[np.all(np.isfinite(coded), axis=1)], list(terms))
        rank = np.linalg.matrix_rank(mat.X) if len(mat.X) else 0
        return int(max(mat.X.shape[1] - rank, 1) + 4)

    def optimal_augment(self, terms, n_add: int, criterion: str = "I",
                        exchange: str = "point", lof_points: int = 0,
                        new_block: bool = True, seed: int | None = None) -> int:
        """Optimal augmentation for a (bigger) model: ``n_add`` runs chosen by the optimal
        engine with every existing run forced into the design, in a new block (phase)."""
        from rsstatchem.core.doe import optimal

        if self.split_plot:
            raise ValueError("Optimal augmentation is not available for split-plot designs.")
        sp = self.space()
        terms = list(terms)
        coded = self.coded_matrix()
        ok = np.all(np.isfinite(coded), axis=1)
        forced = coded[ok]
        old_blocks = self.ds.df["Block"].to_numpy(float)[ok]
        region = self.region()

        def fmat(c):
            return model.build_matrix(sp, np.atleast_2d(c), terms).X

        rng = np.random.default_rng(seed)
        cand = None
        if exchange != "coordinate":
            cand, _ = region.candidates(("vertex", "edge", "thirds", "plane", "interior",
                                         "centroid"))
            if not region.constrained and not region.mix and len(region.cont) <= 6:
                import itertools

                grid = np.array(list(itertools.product(*[
                    (np.arange(region.nlevels[j], dtype=float) if region.nlevels[j] else
                     np.array([-1.0, 0.0, 1.0])) for j in range(region.k)])), float)
                cand = np.unique(np.vstack([cand, grid]).round(12), axis=0)
        nb_old = len(np.unique(old_blocks))
        res = optimal.optimal_design(
            fmat, int(n_add), region, cand, criterion, exchange, rng=rng, forced=forced,
            forced_blocks=old_blocks if (new_block or nb_old > 1) else None,
            block_sizes=[int(n_add)] if new_block else None, allow_repeats=True)
        new = res.coded[len(forced):]
        if lof_points:
            pool = cand if cand is not None else region.candidates()[0]
            lof = designs._distance_rows(pool, res.coded, int(lof_points))
            new = np.vstack([new, lof]) if len(lof) else new
        types = designs.classify_points(region, new)
        block = self.next_block() if new_block else self.last_block()
        added = self._append_runs(new, block, types, f"Optimal ({criterion}) augment: "
                                  f"{len(new)} run(s)", "optimal", seed,
                                  criterion=criterion,
                                  terms=[model.term_key(t) for t in terms])
        return added

    def modify_design_space(self, j: int, low: float, high: float) -> None:
        """Modify Design Space: new coded limits for factor j (move, grow or shrink the
        region). The actual settings of the runs stay; their coded values change."""
        f = self.factors[j]
        if f.get("type") == "categoric":
            raise ValueError("Categoric factors have no range.")
        if not high > low:
            raise ValueError("High must exceed Low.")
        f["low"], f["high"] = float(low), float(high)
        if f.get("type") == "mixture" and self.meta.get("mixture"):
            mix = self.meta["mixture"]
            i = [g for g in self.factors if g.get("type") == "mixture"].index(f)
            mix["lower"][i], mix["upper"][i] = float(low), float(high)
        self.meta.setdefault("design_space_changes", []).append(
            {"factor": j, "low": float(low), "high": float(high)})
        self._structure_changed()

    def set_constraints(self, constraints: list) -> None:
        """Replace the multiple linear constraints (texts or dicts, actual units)."""
        k = self.n_mix if self.combined else len(self.factors)
        parsed = designs._parse_constraints(constraints, k)
        if parsed:
            self.meta["constraints"] = parsed
        else:
            self.meta.pop("constraints", None)
        self._cache.clear()
        self.touch("meta")

    def infeasible_runs(self) -> list[int]:
        """Rows whose settings violate the bounds or constraints of the design region."""
        if not self.constraints:
            return []
        coded = self.coded_matrix()
        try:
            ok = self.region().feasible(coded)
        except ValueError:
            return []
        return [i for i in range(len(ok)) if not ok[i] and np.all(np.isfinite(coded[i]))]

    # ------------------------------------------------------------------ layout editing
    def round_factors(self, decimals: int | None = None, sig: int | None = None) -> int:
        """Round the numeric factor settings to ``decimals`` places or ``sig`` significant
        digits; mixture rows are fixed up so the components still add up to the total.
        Returns the number of cells changed."""
        df = self.ds.df
        changed = 0

        def rnd(v):
            if sig:
                with np.errstate(divide="ignore", invalid="ignore"):
                    mag = np.where(v == 0, 0, np.floor(np.log10(np.abs(v))))
                d = (sig - 1 - mag).astype(int)
                return np.array([round(float(x), int(dd)) if np.isfinite(x) else x
                                 for x, dd in zip(v, d)])
            return np.round(v, int(decimals or 0))

        mixcols = [f["column"] for f in self.factors if f.get("type") == "mixture"]
        for f in self.factors:
            if f.get("type") == "categoric":
                continue
            col = f["column"]
            old = df[col].to_numpy(float)
            new = rnd(old)
            changed += int(np.sum(~np.isclose(old, new, equal_nan=True)))
            df[col] = new
            dec = int(decimals) if decimals is not None else designs._decimals(new)
            self.ds.var(col).decimals = max(dec, 0)
        if mixcols:
            total = float((self.meta.get("mixture") or {}).get("total", 1.0))
            M = df[mixcols].to_numpy(float)
            for i in range(len(M)):
                if np.all(np.isfinite(M[i])):
                    diff = total - M[i].sum()
                    if abs(diff) > 1e-12:
                        jmax = int(np.argmax(M[i]))
                        M[i, jmax] = round(M[i, jmax] + diff, 12)
            df[mixcols] = M
        self.version += 1
        self._cache.clear()
        self.touch("data", list(self.response_cols) + self.factor_cols)
        return changed

    def insert_runs(self, actual_rows=None, n: int = 1, block: float | None = None) -> int:
        """Insert runs at the end of the layout: given factor settings (actual units) or
        ``n`` centre runs (blank rows in an editable historical design)."""
        sp = self.space()
        if actual_rows is None:
            if self.editable_factors:
                coded = np.full((n, sp.k), np.nan)
            else:
                coded = np.zeros((n, sp.k))
                if self.mixture or self.combined:
                    reg = self.region()
                    coded[:, reg.cont] = reg.centroid()
        else:
            coded = sp.to_coded(np.atleast_2d(np.asarray(actual_rows, float)))
        b = self.last_block() if block is None else float(block)
        df = self.ds.df
        m = len(coded)
        actual = sp.to_actual(np.where(np.isfinite(coded), coded, 0))
        actual[~np.isfinite(coded)] = np.nan
        rows = {"Std": np.nanmax(df["Std"].to_numpy(float)) + 1 + np.arange(m),
                "Run": np.nanmax(df["Run"].to_numpy(float)) + 1 + np.arange(m),
                "Block": np.full(m, b)}
        if "Space_Type" in df:
            rows["Space_Type"] = np.full(m, 10.0 if self.editable_factors else 8.0)
        for j, f in enumerate(sp.factors):
            rows[f.column] = actual[:, j]
        add = pd.DataFrame({c: rows.get(c, np.full(m, np.nan)) for c in df.columns})
        self.ds.df = pd.concat([df, add], ignore_index=True)
        st = self.ds.var("Space_Type") if self.ds.has_name("Space_Type") else None
        if st is not None:
            st.value_labels = dict(st.value_labels or {})
            t = 10.0 if self.editable_factors else 8.0
            st.value_labels[t] = designs.SPACE_TYPES[int(t)]
        self._structure_changed()
        return m

    def delete_runs(self, rows) -> int:
        """Delete runs; Std and Run are renumbered 1..n keeping their order."""
        rows = sorted({int(r) for r in rows})
        if not rows:
            return 0
        if len(rows) >= self.n_runs:
            raise ValueError("A design needs at least one run.")
        df = self.ds.df
        keep = [i for i in range(len(df)) if i not in set(rows)]
        old_index = {old: new for new, old in enumerate(keep)}
        df = df.iloc[keep].reset_index(drop=True)
        for col in ("Std", "Run"):
            order = np.argsort(df[col].to_numpy(float), kind="stable")
            vals = np.empty(len(df))
            vals[order] = np.arange(1, len(df) + 1)
            df[col] = vals
        self.ds.df = df
        self.ignored = {old_index[i] for i in self.ignored if i in old_index}
        self._structure_changed()
        return len(rows)

    def add_response(self, name: str, units: str = "") -> str:
        """Add an empty response column; returns its column name."""
        used = {c.lower() for c in self.ds.df.columns}
        i = len(self.responses)
        col = designs._unique_name(name, used, f"R{i + 1}")
        self.ds.df[col] = np.full(self.n_runs, np.nan)
        self.ds.variables.append(Variable(col, decimals=2, label=f"R{i + 1}: {name or col}"
                                          + (f" ({units})" if units else "")))
        self.meta.setdefault("responses", []).append({"name": name or col, "column": col,
                                                      "units": units})
        self._structure_changed()
        return col

    def rename_block(self, block: float, name: str) -> None:
        var = self.ds.var("Block")
        var.value_labels = dict(var.value_labels or {})
        var.value_labels[float(block)] = name.strip() or f"Block {block:g}"
        self.meta.setdefault("block_names", {})[f"{float(block):g}"] = var.value_labels[
            float(block)]
        self.touch("meta")

    def block_values(self) -> list[float]:
        b = self.ds.df["Block"].to_numpy(float)
        return sorted(set(b[np.isfinite(b)].tolist()))

    def block_name(self, block: float) -> str:
        return self.ds.var("Block").value_labels.get(float(block), f"Block {block:g}")

    def analysis_dataset(self) -> Dataset:
        """The design with ignored runs' responses blanked (they drop out of every fit)."""
        if not self.ignored:
            return self.ds
        ds = Dataset(self.ds.df.copy(), [Variable.from_dict(v.to_dict())
                                         for v in self.ds.variables], name=self.ds.name)
        ds.meta = self.ds.meta
        rows = sorted(self.ignored)
        for col in self.response_cols:
            ds.df.loc[rows, col] = np.nan
        return ds

    # ------------------------------------------------------------------ factor / response info
    def set_factor_info(self, j: int, name: str, units: str) -> None:
        f = self.factors[j]
        f["name"], f["units"] = name or f["name"], units
        var = self.ds.var(f["column"])
        var.label = f"{designs.letter(j)}: {f['name']}" + (f" ({units})" if units else "")
        self._cache.clear()
        self.touch("meta")

    def set_factor_variation(self, j: int, sd: float, levels: list[float] | None = None
                             ) -> None:
        """Column info of a numeric factor: the standard deviation of its setting (stored in
        ``optimization["poe_sd"]``, read by propagation of error) and its allowed discrete
        levels (``factors[j]["discrete_levels"]`` and ``optimization["discrete"]``)."""
        f = self.factors[j]
        col = f["column"]
        sds = self.optimization.setdefault("poe_sd", {})
        if sd and sd > 0:
            sds[col] = float(sd)
        else:
            sds.pop(col, None)
        disc = self.optimization.setdefault("discrete", {})
        levels = sorted({float(v) for v in levels or []})
        if levels:
            f["discrete_levels"] = levels
            disc[col] = levels
        else:
            f.pop("discrete_levels", None)
            disc.pop(col, None)
        self._cache.clear()
        self.touch("meta")

    def set_response_info(self, i: int, name: str, units: str) -> None:
        r = self.responses[i]
        r["name"], r["units"] = name or r["name"], units
        var = self.ds.var(r["column"])
        var.label = f"R{i + 1}: {r['name']}" + (f" ({units})" if units else "")
        self._cache.clear()
        self.touch("meta")

    # ------------------------------------------------------------------ analysis
    def spec(self, col: str) -> model.AnalysisSpec:
        return model.AnalysisSpec.from_meta(col, self.meta.get("analysis", {}).get(col))

    def set_spec(self, spec: model.AnalysisSpec) -> None:
        self.meta.setdefault("analysis", {})[spec.response] = spec.to_meta()
        self.touch("analysis", spec.response)

    def has_spec(self, col: str) -> bool:
        return col in self.meta.get("analysis", {})

    def analyze(self, col: str) -> model.Analysis:
        """The analysis of response ``col`` with its stored choices (cached until data or
        choices change). Raises ValueError when it cannot be fitted."""
        spec = self.spec(col)
        key = ("analysis", col, json.dumps(spec.to_meta(), sort_keys=True, default=str))
        if key not in self._cache:
            try:
                self._cache[key] = model.analyze(self.analysis_dataset(), spec)
            except (ValueError, np.linalg.LinAlgError) as exc:
                self._cache[key] = exc
        res = self._cache[key]
        if isinstance(res, Exception):
            raise res
        return res

    def fits(self) -> dict:
        """Fitted model of every response that can be analysed: {column: Fit}."""
        out = {}
        for col in self.response_cols:
            if self.response_count(col) < 2:
                continue
            try:
                out[col] = self.analyze(col).fit
            except (ValueError, np.linalg.LinAlgError):
                continue
        return out

    # ------------------------------------------------------------------ notes
    def set_notes(self, text: str) -> None:
        if text != self.notes:
            self.notes = text
            self.touch("notes")

    # ------------------------------------------------------------------ files
    def to_dict(self) -> dict:
        df = self.ds.df
        data = {}
        for v in self.ds.variables:
            col = df[v.name]
            data[v.name] = [None if (x is None or (isinstance(x, float) and math.isnan(x)))
                            else float(x) for x in col.tolist()]
        std = df["Std"].to_numpy(float)
        return {"format": RSX_FORMAT, "version": RSX_VERSION, "name": self.ds.name,
                "variables": [v.to_dict() for v in self.ds.variables], "data": data,
                "design": _jsonable(self.ds.meta.get("doe", {})), "notes": self.notes,
                "build_notes": self.build_notes,
                "ignored_std": sorted(int(std[i]) for i in self.ignored),
                "optimization": _jsonable(self.optimization), "ui": _jsonable(self.ui)}

    @classmethod
    def from_dict(cls, d: dict) -> "DesignDocument":
        if d.get("format") != RSX_FORMAT:
            raise ValueError("This is not an RS-StatChem design (.rsx) file.")
        if int(d.get("version", 1)) > RSX_VERSION:
            raise ValueError("This design was saved by a newer version of RS-StatChem.")
        variables = [Variable.from_dict(v) for v in d["variables"]]
        for v in variables:  # JSON turns the float keys of value labels into numbers already
            v.value_labels = {float(k): lab for k, lab in v.value_labels.items()}
        df = pd.DataFrame({v.name: [np.nan if x is None else x for x in d["data"][v.name]]
                           for v in variables})
        ds = Dataset(df, variables, name=d.get("name") or "Design")
        ds.meta["doe"] = d.get("design", {})
        doc = cls(ds, d.get("build_notes"))
        doc.notes = d.get("notes", "")
        doc.optimization = d.get("optimization", {}) or {}
        doc.ui = d.get("ui", {}) or {}
        std = ds.df["Std"].to_numpy(float)
        wanted = set(d.get("ignored_std", []))
        doc.ignored = {i for i, s in enumerate(std) if int(s) in wanted}
        return doc

    def save(self, path: str) -> None:
        text = json.dumps(self.to_dict(), indent=1, ensure_ascii=False)
        Path(path).write_text(text, encoding="utf-8")
        self.path = str(path)
        self.modified = False
        self.notify("saved", path)

    @classmethod
    def load(cls, path: str) -> "DesignDocument":
        try:
            d = json.loads(Path(path).read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"The file is damaged or not a design file ({exc}).") from exc
        doc = cls.from_dict(d)
        doc.path = str(path)
        return doc

    # ------------------------------------------------------------------ import
    def import_responses(self, frame: pd.DataFrame) -> list[str]:
        """Copy response columns from a table (matched by response name or column name, rows
        matched by a Std or Run column, else in run order). Returns the responses filled."""
        lower = {str(c).strip().lower(): c for c in frame.columns}

        def find(*names):
            for n in names:
                if n and n.strip().lower() in lower:
                    return lower[n.strip().lower()]
            return None

        key_src = find("Std", "Std Order", "Standard Order")
        key_dst = "Std"
        if key_src is None:
            key_src, key_dst = find("Run", "Run Order"), "Run"
        if key_src is not None:
            keys = pd.to_numeric(frame[key_src], errors="coerce").to_numpy(float)
            pos = {float(v): i for i, v in enumerate(self.ds.df[key_dst].to_numpy(float))}
            targets = [pos.get(float(k)) for k in keys]
        else:
            order = self.row_order("run")
            targets = [order[i] if i < len(order) else None for i in range(len(frame))]
        changes, filled = [], []
        for i, r in enumerate(self.responses):
            src = find(r["name"], r["column"], f"R{i + 1}", self.response_title(i),
                       f"R{i + 1}: {r['name']}")
            if src is None:
                continue
            values = pd.to_numeric(frame[src], errors="coerce").to_numpy(float)
            for t, v in zip(targets, values):
                if t is not None:
                    changes.append((t, r["column"], v))
            filled.append(self.response_title(i))
        self.set_values(changes)
        return filled


def _jsonable(obj):
    """Deep copy with numpy scalars/arrays and tuples turned into plain JSON types."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj
