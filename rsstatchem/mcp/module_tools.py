"""Connector tools for the RSM, Graph, Chemical Drawing and Image Analysis modules.

They cover the usual path through each module in a few calls. Everything else in those modules is
reached through its menus (``find_commands`` / ``run_command`` in :mod:`rsstatchem.mcp.engine`).
All of them work on the module's own window, so what they create can be saved, opened and
continued by hand.
"""
from __future__ import annotations

import base64
import math
from pathlib import Path
from typing import Any, Literal

import numpy as np

from rsstatchem.core.output import Section, Table
from rsstatchem.mcp import driver
from rsstatchem.mcp.engine import AppWorkspace, Picture, _json, _output_path, app_tool
from rsstatchem.mcp.render import section_markdown
from rsstatchem.mcp.tools import _resolve

Design = Literal["factorial", "general", "pb", "ccd", "bbd", "threelevel", "dopt", "lattice",
                 "centroid", "ev", "mixdopt", "taguchi", "mr5", "mr4", "dsd", "onefactor",
                 "simscreen", "combined", "lhs", "distance"]
ModelOrder = Literal["suggested", "mean", "linear", "2fi", "3fi", "quadratic", "cubic", "manual"]
TransformKind = Literal["none", "sqrt", "ln", "log10", "inv_sqrt", "inverse", "power", "logit",
                        "asin_sqrt"]
RsmGraph = Literal["contour", "surface", "one_factor", "interaction", "perturbation", "cube",
                   "ternary", "surface3", "se"]
PlotKind = str


def _finite(x):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else x


class FullWorkspace(AppWorkspace):
    # ================================================================== RSM / DOE
    def _rsm(self, window: str = ""):
        pool = [w for w in driver.open_windows() if driver.window_kind(w) == "rsm"
                and getattr(w, "doc", None) is not None]
        if window:
            target = self._window(window)
            if target in pool:
                return target
            raise ValueError(f"'{window}' is not an RSM window with a design.")
        if not pool:
            raise ValueError("No design is open. Use rsm_create_design, or open_module with a "
                             ".rsx file.")
        last = getattr(self, "_last_rsm", None)
        return last if last in pool else pool[-1]

    @staticmethod
    def _response_column(doc, name: str) -> str:
        text = (name or "").strip().lower()
        if not text and len(doc.response_cols) == 1:
            return doc.response_cols[0]
        for i, col in enumerate(doc.response_cols):
            r = doc.responses[i]
            if text in (col.lower(), str(r.get("name", "")).lower(),
                        doc.response_title(i).lower()):
                return col
        raise ValueError(f"No response '{name}'. Responses: "
                         + ", ".join(r.get("name") or r["column"] for r in doc.responses) + ".")

    def _layout(self, doc, limit: int = 200) -> Table:
        df = doc.ds.df
        rows = []
        order = doc.row_order("run")
        cols = [c for c in df.columns]
        titles = []
        for c in cols:
            try:
                titles.append(doc.column_title(c))
            except Exception:  # noqa: BLE001 - bookkeeping columns have no title
                titles.append(str(c))
        for r in order[:limit]:
            line = []
            for c in cols:
                v = df.iloc[r][c]
                label = doc.ds.var(c).value_labels.get(v) if not isinstance(v, str) else None
                if label is not None:
                    line.append(str(label))
                elif isinstance(v, (float, np.floating)):
                    line.append("" if not math.isfinite(v) else f"{v:.6g}")
                else:
                    line.append("" if v is None else str(v))
            rows.append(line)
        notes = [f"The first {limit} of {len(df)} runs."] if len(df) > limit else []
        return Table("Design (run order)", titles, rows, footnotes=notes)

    @app_tool("RSM: create a design")
    def rsm_create_design(self, design: Design, factors: list[dict[str, Any]],
                          responses: list[dict[str, Any]] | None = None,
                          options: dict[str, Any] | None = None, randomize: bool = True,
                          seed: int | None = None, name: str = "") -> str:
        """Create an experimental design in the RSM / DOE module and show its runs.

        Args:
            design: factorial (two-level, options {"generators": "E=ABCD", "replicates",
                "center_points", "blocks"}), general (multilevel categoric), pb
                (Plackett-Burman), ccd (options {"alpha_type": "rotatable" | "face" | …,
                "center_points"}), bbd (Box-Behnken), threelevel, dopt (optimal, options
                {"model": "quadratic", "runs", "criterion": "D" | "I" | "A"}), dsd (definitive
                screening), mr4, mr5 (min-run), taguchi, onefactor; mixtures: lattice,
                centroid, ev (extreme vertices), mixdopt, simscreen; combined
                (mixture-process); lhs, distance (space filling).
            factors: [{"name", "units", "low", "high"}]; a categoric factor has
                {"type": "categoric", "levels": ["A", "B"]}; "changes": "hard" makes a
                split-plot design.
            responses: [{"name", "units"}]; default one response.
            options: Options of the design.
            randomize: Randomise the run order.
            seed: Random seed, for a reproducible run order.
            name: Name of the design.
        """
        from rsstatchem.ui.dx.document import DesignDocument
        from rsstatchem.ui.dx.window import DXMainWindow

        spec: dict[str, Any] = {"design": design, "factors": factors,
                                "responses": responses or [], "options": options or {},
                                "randomize": randomize, "seed": seed}
        if name:
            spec["name"] = name
        doc = DesignDocument.from_spec(spec)
        window = DXMainWindow(self.ctx, doc)
        self.ctx.register_window(window)
        window.show()
        self._last_rsm = self._last_window = window
        section = Section(f"Design: {doc.meta.get('title', design)}")
        section.note("", f"{doc.n_runs} runs, {len(doc.factors)} factors. "
                     + " ".join(doc.build_notes))
        section.table(self._layout(doc))
        return f"window {self.elements.id_of(window)}\n\n" + section_markdown(section)

    @app_tool("RSM: show the design", read_only=True)
    def rsm_design(self, window: str = "") -> str:
        """The runs of the open design with factor settings and responses, in run order.

        Args:
            window: RSM window; default is the latest one.
        """
        w = self._rsm(window)
        section = Section(f"Design: {driver.title_of(w)}")
        section.table(self._layout(w.doc))
        return section_markdown(section)

    @app_tool("RSM: enter response values")
    def rsm_set_response(self, values: list[float | None], response: str = "",
                         order: Literal["run", "std"] = "run", window: str = "") -> str:
        """Enter measured values of a response, one per run.

        Args:
            values: The values (null = not measured), in the order given by `order`.
            response: Response name; may be empty when there is only one. An unknown name
                adds a new response.
            order: Values are in run order (as rsm_design lists them) or standard order.
            window: RSM window; default is the latest one.
        """
        w = self._rsm(window)
        doc = w.doc
        try:
            col = self._response_column(doc, response)
        except ValueError:
            if not response.strip():
                raise
            col = w.add_response(response.strip()) or self._response_column(doc, response)
        rows = doc.row_order(order)
        if len(values) != len(rows):
            raise ValueError(f"The design has {len(rows)} runs; {len(values)} values were given.")
        doc.set_values([(r, col, float("nan") if v is None else float(v))
                        for r, v in zip(rows, values)])
        return (f"{sum(v is not None for v in values)} values of "
                f"{doc.response_title(doc.response_index(col))} entered.")

    def _fit(self, doc, col: str, spec=None):
        from rsstatchem.core.doe import model

        return model.analyze(doc.ds, spec or model.stored_spec(doc.ds, col), doc.space())

    def _response_names(self, doc) -> dict:   # response column -> name
        return {c: (doc.responses[i].get("name") or c)
                for i, c in enumerate(doc.response_cols)}

    @app_tool("RSM: analyse a response")
    def rsm_analyze(self, response: str = "", model: ModelOrder = "suggested",
                    terms: list[str] | None = None,
                    reduction: Literal["none", "backward", "forward", "stepwise"] = "none",
                    transform: TransformKind = "none", transform_lambda: float = 1.0,
                    transform_constant: float = 0.0, diagnostics: bool = False,
                    charts: bool = False, window: str = "") -> list:
        """Fit a response: fit summary (sequential p, lack of fit, R²), ANOVA, fit statistics,
        coefficients and equations. The model is stored with the design and used by
        rsm_graph, rsm_predict and rsm_optimize.

        Args:
            response: Response name; may be empty when there is only one.
            model: suggested, or a polynomial order; manual uses `terms`.
            terms: For manual: terms such as ["A", "B", "AB", "A^2"].
            reduction: Remove or add terms automatically (p-value criterion).
            transform: Transformation of the response (see the Box-Cox result in diagnostics).
            transform_lambda: Power for transform="power".
            transform_constant: Constant added before the transformation.
            diagnostics: Also report residual diagnostics.
            charts: Return the diagnostic plots as pictures.
            window: RSM window; default is the latest one.
        """
        from rsstatchem.core.doe import model as m
        from rsstatchem.core.doe import report

        w = self._rsm(window)
        doc = w.doc
        col = self._response_column(doc, response)
        spec = m.stored_spec(doc.ds, col)
        spec.model = model
        spec.reduction = reduction
        spec.transform = m.Transform(kind=transform, k=transform_constant,
                                     lam=transform_lambda)
        spec.terms = None
        if terms:
            spec.model = "manual"
            spec.terms = self._term_keys(doc, terms)
        elif model == "manual":
            raise ValueError("model='manual' needs the list of terms.")
        analysis = m.analyze(doc.ds, spec, doc.space())
        doc.meta.setdefault("analysis", {})[doc.ds.var(col).name] = spec.to_meta()
        doc.touch("analysis", col)
        section = report.anova_section(analysis)
        sections = [section]
        if diagnostics:
            sections.append(report.diagnostics_section(analysis))
        for s in sections:
            self.publish(s)
        return self._parts(sections, charts)

    @staticmethod
    def _term_keys(doc, terms: list[str]) -> list[str]:
        """"AB", "A*B", "A^2", "A²" or factor names -> the model's term keys."""
        from rsstatchem.core.doe import model as m

        space = doc.space()
        out = []
        for raw in terms:
            text = str(raw).replace("²", "^2").replace("³", "^3").strip()
            try:
                out.append(m.term_key(m.parse_term(text, space)))
            except (ValueError, KeyError, IndexError) as exc:
                raise ValueError(f"'{raw}' is not a model term ({exc}); factors: "
                                 + ", ".join(f"{f.letter} = {f.name}" for f in space.factors)
                                 + ".") from None
        return out

    @app_tool("RSM: model graph", read_only=True)
    def rsm_graph(self, graph: RsmGraph = "contour", response: str = "",
                  factors: list[str] | None = None, at: dict[str, Any] | None = None,
                  window: str = "") -> list:
        """A model graph as a picture: contour, 3D surface, one factor, interaction,
        perturbation, cube, ternary (mixtures), surface3 (mixture 3D) or se (standard error of
        the design).

        Args:
            graph: Type of graph.
            response: Response name; may be empty when there is only one.
            factors: Factors on the axes (names or letters); default the first ones.
            at: Values of the other factors in actual units, {"name": value}.
            window: RSM window; default is the latest one.
        """
        from rsstatchem.core.doe import report

        w = self._rsm(window)
        doc = w.doc
        analysis = self._fit(doc, self._response_column(doc, response))
        space = analysis.space
        need = {"one_factor": 1, "ternary": 3, "surface3": 3, "cube": 3,
                "perturbation": space.k}.get(graph, 2)
        axes = [self._factor(space, f) for f in factors or []]
        for j in range(space.k):
            if len(axes) >= min(need, space.k):
                break
            if j not in axes and not (graph in ("contour", "surface", "se")
                                      and space.factors[j].categoric):
                axes.append(j)
        ref = self._point(space, at)
        kind = {"one_factor": "onefactor"}.get(graph, graph)
        try:
            section = report.graph_section(analysis, kind, axes, ref)
        except (KeyError, IndexError) as exc:
            raise ValueError(f"The graph '{graph}' cannot be drawn for this design: {exc}"
                             ) from None
        return self._parts([section], True)

    @staticmethod
    def _factor(space, name: str) -> int:
        text = str(name).strip().lower()
        for j, f in enumerate(space.factors):
            if text in (f.letter.lower(), f.name.lower(), f.column.lower(), f.title.lower()):
                return j
        raise ValueError(f"No factor '{name}'. Factors: "
                         + ", ".join(f"{f.letter} = {f.name}" for f in space.factors) + ".")

    def _point(self, space, at: dict | None) -> np.ndarray:
        from rsstatchem.core.doe import report

        x = report.default_point(space)
        for name, value in (at or {}).items():
            j = self._factor(space, name)
            f = space.factors[j]
            if f.categoric and isinstance(value, str):
                labels = [str(v).lower() for v in (f.labels or f.levels)]
                if value.lower() not in labels:
                    raise ValueError(f"'{value}' is not a level of {f.name}: "
                                     + ", ".join(map(str, f.labels or f.levels)) + ".")
                x[j] = f.levels[labels.index(value.lower())]
            else:
                x[j] = float(value)
        return x

    @app_tool("RSM: predict", read_only=True)
    def rsm_predict(self, at: dict[str, Any], confidence_level: float = 0.95,
                    window: str = "") -> str:
        """Predicted responses at a point, with confidence and prediction intervals.

        Args:
            at: Factor settings in actual units, {"name": value}; others are at the centre.
            confidence_level: For example 0.95.
            window: RSM window; default is the latest one.
        """
        from rsstatchem.core.doe import report

        w = self._rsm(window)
        doc = w.doc
        fits = {c: self._fit(doc, c).fit for c in doc.response_cols if doc.response_count(c) > 2}
        if not fits:
            raise ValueError("Enter response values first.")
        section = report.prediction_section(fits, doc.space(), self._point(doc.space(), at),
                                            self._response_names(doc), confidence_level)
        self.publish(section)
        return section_markdown(section)

    @app_tool("RSM: optimise")
    def rsm_optimize(self, goals: list[dict[str, Any]], charts: bool = False, starts: int = 30,
                     window: str = "") -> list:
        """Numerical optimisation by desirability over the analysed responses and factors.

        Args:
            goals: One entry per response or factor: {"name", "goal": "maximize" | "minimize" |
                "target" | "in_range" | "none", "lower", "upper", "target", "importance": 1-5,
                "w_lower", "w_upper"}. Limits default to the observed range. Factors not
                mentioned stay in range.
            charts: Return the ramps, bar and contour pictures.
            starts: Number of random starts.
            window: RSM window; default is the latest one.
        """
        from rsstatchem.core.doe import optimize, report
        from rsstatchem.ui.dx import optimization as page

        w = self._rsm(window)
        doc = w.doc
        space = doc.space()
        fits = {c: self._fit(doc, c).fit for c in doc.response_cols if doc.response_count(c) > 2}
        if not fits:
            raise ValueError("Enter response values and analyse them first.")
        criteria = optimize.default_criteria(fits, space)
        by_name = {c.name: c for c in criteria}
        for g in goals:
            name = str(g.get("name", ""))
            try:
                key = self._response_column(doc, name)
            except ValueError:
                key = space.factors[self._factor(space, name)].column
            c = by_name.get(key)
            if c is None:
                raise ValueError(f"'{name}' has no values to optimise.")
            goal = str(g.get("goal", "none"))
            goals_known = [k for k, _ in (optimize.FACTOR_GOALS if c.kind == "factor"
                                          else optimize.GOALS)]
            if goal not in goals_known:
                raise ValueError(f"The goal of {name} must be one of {', '.join(goals_known)}.")
            c.goal = goal
            for key2 in ("lower", "upper", "target", "w_lower", "w_upper"):
                if g.get(key2) is not None:
                    setattr(c, key2, float(g[key2]))
            if g.get("importance") is not None:
                c.importance = int(g["importance"])
            if c.goal == "target" and c.target is None:
                raise ValueError(f"The goal 'target' of {name} needs a target value.")
        problem = page.make_problem(doc, fits, criteria)
        solutions = optimize.solve(problem, starts=starts, seed=0,
                                   design_points=page.design_points(doc))
        if not solutions:
            raise ValueError("No solution satisfies the goals; widen the limits.")
        doc.last_solutions = solutions
        doc.optimization["criteria"] = [c.to_meta() for c in criteria]
        section = report.optimization_section(problem, solutions, self._response_names(doc), seed=0,
                                              starts=starts)
        self.publish(section)
        return self._parts([section], charts)

    @app_tool("RSM: save the design")
    def rsm_save(self, path: str, overwrite: bool = False, window: str = "") -> str:
        """Save the design with its responses and models as a .rsx file.

        Args:
            path: Full path ending in .rsx.
            overwrite: Must be true to replace an existing file. Ask the user first.
            window: RSM window; default is the latest one.
        """
        target = _output_path(path, overwrite, (".rsx",))
        w = self._rsm(window)
        w.doc.save(str(target))
        return f"Saved the design to {target}"

    # ================================================================== Graph & Analysis
    def _graph_window(self, window: str = ""):
        pool = [w for w in driver.open_windows() if driver.window_kind(w) == "graph"
                and hasattr(w, "project")]
        if window:
            target = self._window(window)
            if target in pool:
                return target
            raise ValueError(f"'{window}' is not a Graph & Analysis window.")
        if pool:
            last = getattr(self, "_last_graph", None)
            return last if last in pool else pool[-1]
        from rsstatchem.ui import modules

        self._last_graph = modules.open_module(self.ctx, "graph")
        return self._last_graph

    def _book(self, w, name: str = ""):
        if name:
            for book in w.project.books():
                if name.lower() in (book.name.lower(), str(book.long_name).lower()):
                    w.show_window(book.name)
                    return w.book_window(book)
            raise ValueError(f"No workbook '{name}'. Workbooks: "
                             + ", ".join(b.name for b in w.project.books()) + ".")
        bw = w.active_book_window()
        if bw is None:
            books = w.project.books()
            if not books:
                return w.new_book()
            w.show_window(books[-1].name)
            bw = w.book_window(books[-1])
        return bw

    @staticmethod
    def _columns(ws, names) -> list[int]:
        out = []
        for raw in names:
            text = str(raw).strip().lower()
            for i, c in enumerate(ws.columns):
                if text in (c.name.lower(), str(c.long_name).lower()) or text == str(i + 1):
                    out.append(i)
                    break
            else:
                raise ValueError(f"No column '{raw}'. Columns: " + ", ".join(
                    c.name + (f" ({c.long_name})" if c.long_name else "")
                    for c in ws.columns) + ".")
        return out

    @app_tool("Graph: put data in a workbook")
    def graph_data(self, columns: dict[str, list[Any]] | None = None, dataset: str = "",
                   variables: list[str] | None = None, path: str = "", name: str = "") -> str:
        """Put data in a workbook of the Graph & Analysis module: given columns, variables of
        an open dataset, or a CSV / Excel file. The first column becomes X, the others Y.

        Args:
            columns: {"column name": [values]}.
            dataset: Name of an open dataset to copy.
            variables: Which variables of the dataset; default all.
            path: Full path of a .csv, .txt or .xlsx file to import.
            name: Name of the workbook.
        """
        import pandas as pd

        w = self._graph_window()
        if path:
            bw = w.import_file(str(_resolve(path)))
            if bw is None:
                raise ValueError(f"Could not import {path}.")
        else:
            if columns:
                length = max(len(v) for v in columns.values())
                df = pd.DataFrame({k: list(v) + [None] * (length - len(v))
                                   for k, v in columns.items()})
            else:
                ds = self.dataset(dataset)
                names = [ds.var(v).name for v in variables] if variables else ds.names()
                df = ds.df[names].copy()
            bw = w.import_frame(df, name or dataset or "Data")
        self._last_graph = self._last_window = w
        ws = bw.worksheet()
        return _json({"window": self.elements.id_of(w), "workbook": bw.book.name,
                      "rows": ws.data_rows(),
                      "columns": [{"name": c.name, "long_name": c.long_name,
                                   "role": c.designation} for c in ws.columns]})

    @app_tool("Graph: draw a graph")
    def graph_plot(self, kind: PlotKind, columns: list[str], workbook: str = "",
                   window: str = "") -> list:
        """Draw a graph from worksheet columns and return its picture. X comes from the column
        marked X to the left of each Y column.

        Args:
            kind: line, scatter, line_symbol, yerror, column, column_stacked, bar, area, pie,
                histogram, hist_kde, box, violin, meanbar, interval, double_y, bubble, contour,
                heatmap, scatter3d, surface3d, polar, ternary, qq, probability, ecdf, pareto,
                control_xbar, control_imr, bland_altman, scatter_matrix, radar, … (an unknown
                kind lists them all).
            columns: Y columns (names, long names or numbers).
            workbook: Workbook name; default the active one.
            window: Graph window; default the latest one.
        """
        from rsstatchem.ui.origin.plotting import PLOT_TYPES, PlotError, plot_columns

        w = self._graph_window(window)
        bw = self._book(w, workbook)
        ws = bw.worksheet()
        if kind not in PLOT_TYPES:
            raise ValueError("Unknown kind. Kinds: " + ", ".join(
                f"{k} ({v[0]})" for k, v in PLOT_TYPES.items()))
        cols = self._columns(ws, columns)
        try:
            graph = plot_columns(bw.book, ws, cols, kind, w.project.unique_name("Graph"))
        except PlotError as exc:
            raise ValueError(str(exc)) from None
        gwin = w.new_graph(graph, folder=w.project.root.find_window(bw.book.name))
        self._last_graph = self._last_window = w
        return [f"{graph.title} drawn (window {self.elements.id_of(w)}).",
                *self._graph_picture(gwin.graph)]

    @staticmethod
    def _graph_picture(graph) -> list:
        from rsstatchem.core.graph import export

        return [Picture(export.png_bytes(graph, dpi=110), graph.title)]

    @app_tool("Graph: fit a curve")
    def graph_fit(self, model: str, y_column: str, kind: Literal["linear", "polynomial",
                                                                "nlfit"] = "nlfit",
                  start: dict[str, float] | None = None, fixed: dict[str, float] | None = None,
                  weighting: str = "none", workbook: str = "", window: str = "",
                  charts: bool = True) -> list:
        """Fit a curve to a worksheet column (X from the column marked X): parameters with
        standard errors, statistics and ANOVA, as the module's fit report.

        Args:
            model: A function of the library ("Line", "Boltzmann", "ExpDec1", "Gauss",
                "Michaelis-Menten", …; an unknown name lists them) or an equation such as
                "y = a*exp(-k*x) + c".
            y_column: Y column (name, long name or number).
            kind: Kind of fit.
            start: Start values {"a": 1.0}.
            fixed: Parameters held at a value.
            weighting: none, instrumental, statistical, …
            workbook: Workbook name; default the active one.
            window: Graph window; default the latest one.
            charts: Return the fit's graphs.
        """
        from rsstatchem.core.analysis import fitting
        from rsstatchem.ui.origin import analysis as an

        w = self._graph_window(window)
        bw = self._book(w, workbook)
        ws = bw.worksheet()
        col = self._columns(ws, [y_column])[0]
        x = ws.x_of(col)
        try:
            fitting.resolve_model(model)
        except (ValueError, KeyError) as exc:
            from rsstatchem.core.analysis.functions import LIBRARY

            raise ValueError(f"{exc} Library: {', '.join(sorted(LIBRARY))}") from None
        data = an.xy_data(bw.book, ws, col, x, with_error=True)
        params0 = dict(start or {})
        params0.update(fixed or {})
        try:
            outcome = w._run_in_book(bw.book, "Fit", lambda: an.run_fit(
                data, model, kind, weighting=weighting, params0=params0 or None,
                fixed=dict(fixed) if fixed else None))
        except (an.AnalysisError, ValueError, KeyError) as exc:
            raise ValueError(f"Fit failed: {exc}") from None
        bw.show_sheet(outcome.report)
        self._last_graph = self._last_window = w
        return self._report(outcome.report, charts)

    def _report(self, report, charts: bool) -> list:
        """A report sheet of the graph module as a section (also published)."""
        from rsstatchem.core.graph.model import graph_from_json_dict

        section = Section(report.title or report.name)
        pictures: list[Picture] = []
        for item in report.items:
            if item["kind"] == "table":
                section.table(Table(item["title"], [
                    tuple(c) if isinstance(c, list) else str(c) for c in item["columns"]],
                    item["rows"], footnotes=[str(n) for n in item.get("notes", ())]))
            elif item["kind"] == "text":
                section.note(item["title"], item["text"])
            elif charts and item["kind"] == "image":
                pictures.append(Picture(base64.b64decode(item["png"]), item["title"]))
            elif charts and item["kind"] == "graph" and len(pictures) < 4:
                pictures += self._graph_picture(graph_from_json_dict(item["graph"]))
        for p in pictures:
            section.image(p.title, p.png)
        self.publish(section)
        return self._parts([section], charts)

    @app_tool("Graph: save the project")
    def graph_save(self, path: str, overwrite: bool = False, window: str = "") -> str:
        """Save all workbooks and graphs as a .rsop project.

        Args:
            path: Full path ending in .rsop.
            overwrite: Must be true to replace an existing file. Ask the user first.
            window: Graph window; default the latest one.
        """
        from rsstatchem.ui.origin.model import save_project

        target = _output_path(path, overwrite, (".rsop",))
        w = self._graph_window(window)
        w.sync_models()
        save_project(w.project, target)
        return f"Saved the project to {target}"

    # ================================================================== Chemical Drawing
    def _chem_window(self, window: str = "", create: bool = True):
        pool = [w for w in driver.open_windows() if driver.window_kind(w) == "chem"
                and hasattr(w, "canvas")]
        if window:
            target = self._window(window)
            if target in pool:
                return target
            raise ValueError(f"'{window}' is not a Chemical Drawing window.")
        if pool:
            return pool[-1]
        if not create:
            raise ValueError("No drawing is open.")
        from rsstatchem.ui import modules

        return modules.open_module(self.ctx, "chem")

    @staticmethod
    def _smiles(structure: str) -> tuple[str, str]:
        """(SMILES, name) from a SMILES, InChI or compound name (offline sources only)."""
        from rdkit import Chem, RDLogger

        from rsstatchem.core.chem_draw import names

        text = structure.strip()
        if not text:
            raise ValueError("Give a structure (SMILES, InChI or name).")
        RDLogger.DisableLog("rdApp.*")
        try:
            mol = Chem.MolFromInchi(text) if text.startswith("InChI=") else \
                Chem.MolFromSmiles(text) if " " not in text else None
        finally:
            RDLogger.EnableLog("rdApp.*")
        if mol is not None:
            return Chem.MolToSmiles(mol), ""
        for find in (names.offline_structure, names.db_structure):
            hit = find(text)
            if hit is not None and hit.smiles:
                return hit.smiles, hit.name
        if names.opsin_available():
            try:
                hit = names.opsin_structure(text)
                if hit.smiles:
                    return hit.smiles, hit.name or text
            except names.NameLookupError:
                pass
        raise ValueError(f"'{structure}' is neither a valid SMILES / InChI nor a compound name "
                         "known offline. Use an English or IUPAC name, or a SMILES.")

    @app_tool("Chemistry: compound information", read_only=True)
    def chem_compound(self, structure: str, properties: bool = True) -> str:
        """Everything about a compound: SMILES, formula, molar mass, name, elemental analysis,
        identifiers, descriptors, and (properties=true) physical properties with their source
        (experimental or estimated), logP, solubility, pKa.

        Args:
            structure: SMILES, InChI, or an English / IUPAC name ("caffeine").
            properties: Include the physical properties.
        """
        from rdkit import Chem

        from rsstatchem.core.chem import sections
        from rsstatchem.core.chem_draw import names
        from rsstatchem.core.chem_draw import properties as props

        smiles, name = self._smiles(structure)
        mol = Chem.MolFromSmiles(smiles)
        section = sections.structure_section(smiles, title=structure)
        found = names.offline_name(mol) or names.db_name(mol)
        if found is None:
            found = names.generated_name(mol)
        if found is not None or name:
            section.note("Name", (found.name if found is not None else name)
                         + (f" ({found.common})" if found is not None and found.common else "")
                         + (f" — {found.source}" if found is not None else ""))
        if properties:
            rows = []
            for p in props.chemical_properties(mol, cdk_state="not computed here"):
                value = p.display or ("" if p.value is None else f"{p.value:.{p.decimals}f}")
                rows.append([p.name, value, p.unit, p.kind, p.method or p.short,
                             p.note or p.detail])
            section.table(Table("Properties", ["Property", "Value", "Unit", "Kind", "Method",
                                               "Note"], rows))
        self.publish(section)
        return section_markdown(section)

    @app_tool("Chemistry: draw a structure")
    def chem_draw(self, structure: str, window: str = "", new_window: bool = False) -> list:
        """Draw a structure or reaction in the Chemical Drawing module and return the picture.

        Args:
            structure: SMILES, reaction SMILES (A.B>>C), InChI, molfile text, or a name.
            window: Drawing window; default the latest one.
            new_window: Open a new drawing.
        """
        from rsstatchem.core.chem_draw import analysis
        from rsstatchem.ui import modules

        w = modules.open_module(self.ctx, "chem") if new_window else self._chem_window(window)
        text = structure.strip()
        if ">>" not in text and "\n" not in text:
            text = self._smiles(text)[0]
        keys = w.load_text(text)
        if not keys:
            raise ValueError("The structure could not be drawn.")
        self._last_window = w
        info = analysis.analyze(w.canvas.doc)
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            png = Path(tmp) / "drawing.png"
            w.export_as("png", str(png), 150, False)
            picture = png.read_bytes() if png.exists() else driver.screenshot(w.canvas)
        return [_json({"window": self.elements.id_of(w), "formula": info.formula,
                       "molar_mass": _finite(info.mw), "smiles": info.smiles}),
                Picture(picture, "drawing")]

    @app_tool("Chemistry: save or export the drawing")
    def chem_save(self, path: str, overwrite: bool = False, window: str = "") -> str:
        """Save the drawing as .rsc, or export it (.mol, .sdf, .cdxml, .png, .svg, .pdf …).

        Args:
            path: Full path; the type follows the extension.
            overwrite: Must be true to replace an existing file. Ask the user first.
            window: Drawing window; default the latest one.
        """
        target = _output_path(path, overwrite, (".rsc", ".mol", ".sdf", ".rxn", ".cdxml", ".png",
                                                ".svg", ".pdf", ".tif", ".emf"))
        w = self._chem_window(window, create=False)
        if target.suffix.lower() == ".rsc":
            ok = w.save_as(str(target))
        else:
            ok = w.export_as(target.suffix.lower().lstrip("."), str(target))
        if not target.exists():
            raise ValueError("The file was not written." if not ok else f"{target} is missing.")
        return f"Saved to {target}"

    @app_tool("Chemistry: balance an equation", read_only=True)
    def chem_balance(self, equation: str) -> str:
        """Balance a chemical equation, for example "KMnO4 + HCl -> KCl + MnCl2 + H2O + Cl2".

        Args:
            equation: The equation with formulas; -> or = between the sides.
        """
        from rsstatchem.core.chem import reaction, sections

        section = sections.balance_section(reaction.balance(equation))
        self.publish(section)
        return section_markdown(section)

    @app_tool("Chemistry: formula mass and composition", read_only=True)
    def chem_formula(self, formulas: list[str]) -> str:
        """Molar mass and elemental composition of chemical formulas.

        Args:
            formulas: For example ["C6H12O6", "CuSO4·5H2O"].
        """
        from rsstatchem.core.chem import sections

        section = sections.formula_section(formulas)
        self.publish(section)
        return section_markdown(section)

    # ================================================================== Image Analysis
    def _image_bar(self):
        from rsstatchem.ui import modules

        bar = modules.open_module(self.ctx, "image")
        return bar

    def _images_data(self, session) -> list[dict]:
        out = []
        for image in session.images:
            out.append({"title": image.title, "width": image.width, "height": image.height,
                        "slices": image.stack_size,
                        **({"active": True} if image is session.active else {})})
        return out

    def _image_parts(self, session, tables_before: dict, picture: bool) -> list:
        data: dict[str, Any] = {"images": self._images_data(session)}
        parts: list = [_json(data)]
        for title, table in session.tables.items():
            if self._size(table) != tables_before.get(title, 0):
                text = table.to_text("\t")
                lines = text.splitlines()
                shown = "\n".join(lines[:201])
                parts.append(f"Table {title} ({len(lines) - 1} rows)\n{shown}"
                             + ("\n…" if len(lines) > 201 else ""))
        if picture and session.active is not None:
            win = next((x for x in driver.open_windows()
                        if getattr(x, "doc", None) is session.active
                        or getattr(x, "image", None) is session.active), None)
            if win is not None:
                parts.append(Picture(driver.screenshot(win), session.active.title))
        return parts

    @staticmethod
    def _size(table) -> int:
        return len(table.to_rows())

    def _sizes(self, session) -> dict:
        return {t: self._size(x) for t, x in session.tables.items()}

    @app_tool("Image: open an image")
    def image_open(self, path: str, picture: bool = True) -> list:
        """Open an image, stack, ROI file or results table in the Image Analysis module.

        Args:
            path: Full path (.tif, .png, .jpg, .gif, .bmp, .dcm, .fits, .roi, .csv, .rsij …).
            picture: Return a picture of it.
        """
        source = _resolve(path)
        if not source.exists():
            raise ValueError(f"File not found: {source}")
        from rsstatchem.core.imaging.io import open_in_session

        bar = self._image_bar()
        before = self._sizes(bar.session)
        try:
            open_in_session(bar.session, str(source))
        except Exception as exc:  # noqa: BLE001 - any reader failure is a user message
            raise ValueError(f"Could not open {source.name}: {exc}") from None
        return self._image_parts(bar.session, before, picture)

    @app_tool("Image: list commands", read_only=True)
    def image_commands(self, query: str = "") -> str:
        """Commands of the Image Analysis module that image_command can run, with the keys of
        their options and default values.

        Args:
            query: Part of a command name, for example "threshold" or "blur". Empty = names of
                all commands.
        """
        session = self._image_bar().session
        registry = session.registry
        hits = registry.find(query) if query.strip() else list(registry)
        if not query.strip():
            return _json([c.label for c in hits if c.func is not None])
        out = []
        for c in hits[:25]:
            entry: dict[str, Any] = {"command": c.label, "batch": c.func is not None}
            try:
                entry["options"] = {
                    f.key: {"default": f.default, "label": f.label,
                            **({"choices": list(f.items)} if getattr(f, "items", None) else {})}
                    for f in c.dialog_fields(session, session.active) if f.key}
            except Exception:  # noqa: BLE001 - the dialog needs an image
                entry["options"] = "depend on the image"
            if c.help:
                entry["help"] = c.help
            out.append(entry)
        return _json(out)

    @app_tool("Image: run a command")
    def image_command(self, command: str, options: str = "", picture: bool = True) -> list:
        """Run a command of the Image Analysis module on the active image, as in an ImageJ
        macro: run("Gaussian Blur...", "sigma=2"). Returns the images, new results tables and
        a picture of the active image.

        Args:
            command: The command's name ("Gaussian Blur...", "Analyze Particles...").
            options: ImageJ option string ("sigma=2 stack"); see image_commands.
            picture: Return a picture of the active image.
        """
        from rsstatchem.core.imaging.commands import CommandError

        session = self._image_bar().session
        before = self._sizes(session)
        registry = session.registry
        label = command.strip()
        if label not in registry:
            exact = [c.label for c in registry
                     if driver.clean(c.label).lower() == driver.clean(label).lower()]
            if len(exact) != 1:
                near = [c.label for c in registry.find(driver.clean(label))][:12]
                raise ValueError(f"No command '{command}'."
                                 + (f" Similar: {', '.join(near)}." if near else ""))
            label = exact[0]
        try:
            session.run(label, options)
        except CommandError as exc:
            raise ValueError(str(exc)) from None
        return self._image_parts(session, before, picture)

    @app_tool("Image: run a macro")
    def image_macro(self, code: str, picture: bool = True) -> list:
        """Run an ImageJ macro (ImageJ macro language) in the Image Analysis module, for
        selections, loops, measurements and batch work.

        Args:
            code: The macro, for example
                'makeRectangle(10, 10, 100, 80); run("Measure");'.
            picture: Return a picture of the active image.
        """
        from rsstatchem.core.imaging.commands import CommandError
        from rsstatchem.core.imaging.macro import MacroError, run_macro

        bar = self._image_bar()
        session = bar.session
        before = self._sizes(session)
        logged: list[str] = []
        session.on("log", logged.append)
        try:
            result = run_macro(session, code, "", bar.macro_hooks(), None, None)
        except (MacroError, CommandError) as exc:
            raise ValueError(f"Macro: {exc}") from None
        finally:
            session.off("log", logged.append)
        parts = self._image_parts(session, before, picture)
        if logged:
            parts.insert(1, "Log:\n" + "\n".join(map(str, logged[-200:])))
        if result not in (None, ""):
            parts.insert(1, f"Returned: {result}")
        return parts

    @app_tool("Image: save the active image")
    def image_save(self, path: str, overwrite: bool = False) -> str:
        """Save the active image (.tif, .png, .jpg, .gif, .bmp).

        Args:
            path: Full path; the type follows the extension.
            overwrite: Must be true to replace an existing file. Ask the user first.
        """
        target = _output_path(path, overwrite, (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".gif",
                                                ".bmp"))
        bar = self._image_bar()
        doc = bar.session.active
        if doc is None:
            raise ValueError("There are no images open.")
        fmt = {".tif": "tiff", ".tiff": "tiff", ".jpg": "jpeg"}.get(
            target.suffix.lower(), target.suffix.lower().lstrip("."))
        bar.save_image(doc, str(target), fmt)
        if not Path(target).exists():
            raise ValueError("The image was not written.")
        return f"Saved {doc.title} to {target}"
