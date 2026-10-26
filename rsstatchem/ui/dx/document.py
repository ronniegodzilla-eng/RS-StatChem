"""The design behind an RSM window: one Design-Expert-style design and its ``.rsx`` file.

No Qt here. The design layout is a :class:`Dataset` built by
:func:`rsstatchem.core.doe.designs.build_design` (columns Std, Run, Block, Space_Type, the
factors in actual units and the responses) with the design definition in ``ds.meta["doe"]``.
The rows keep their positions for the life of the document (sorting only changes the view),
so row indices are stable identifiers for undo and for ignored runs.

The ``.rsx`` file is JSON: format tag, version, the variables, the column data, the design
meta (factors, responses, analysis choices), notes, ignored runs, optimisation criteria and
display options.
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
    def n_blocks(self) -> int:
        return int(self.meta.get("blocks", 1) or 1)

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
        return {"Space_Type": "Space Type"}.get(col, col)

    def space(self) -> model.Space:
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
        rng = np.random.default_rng(seed)
        blocks = self.ds.df["Block"].to_numpy(float)
        run = np.empty(len(blocks))
        pos = 0
        for b in sorted(set(blocks.tolist())):
            members = np.flatnonzero(blocks == b)
            perm = rng.permutation(members)
            run[perm] = np.arange(pos + 1, pos + len(members) + 1)
            pos += len(members)
        self.ds.df["Run"] = run
        self.meta["randomized"] = True
        self.touch("data", [])

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
