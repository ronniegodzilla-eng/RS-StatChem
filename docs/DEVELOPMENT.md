# RS-StatChem — Developer Guide & Module Contract

RS-StatChem is a PySide6 desktop app (Python 3.12) that mirrors SPSS (Data Editor, Analyze
menu, Output Viewer), OriginLab (graphing, curve fitting) and Design-Expert (DOE/RSM).
Read this whole file before writing code. It is the contract that lets several people (or
agents) build features in parallel without touching each other's files.

## Environment

- Python: `.venv/bin/python` (arm64, 3.12). Install a package only if truly needed:
  `.tools/bin/uv pip install --python .venv/bin/python <pkg>` and report it.
- Runtime libraries you may use: numpy, scipy, pandas (3.x: default string dtype is `str`),
  statsmodels, matplotlib, PySide6, openpyxl, pyreadstat, python-docx, rdkit (chemistry).
  **Do not import scikit-learn, pingouin or seaborn in app code** — they are excluded from the
  packaged build. They may be used in tests as reference implementations (pingouin is
  installed).
- Tests: `.venv/bin/python -m pytest -q tests/test_<area>.py -p no:warnings`
  Run only your own test files while others are working in parallel.
- Headless UI checks: `QT_QPA_PLATFORM=offscreen RSSTATCHEM_NO_SPLASH=1
  RSSTATCHEM_HOME=<scratch dir> PYTHONPATH=. .venv/bin/python script.py`; take screenshots
  with `widget.grab().save(path)` and look at them. Keep scratch files out of the repo.
- Example data: `examples/contoh_penelitian.rsd`, `examples/contoh_kuesioner.rsd`
  (load with `rsstatchem.io.files.load_project`).

## Layout

```
rsstatchem/
  core/            no Qt here
    dataset.py     Dataset + Variable metadata
    output.py      Section / Table / Text / Image (Output Viewer model)
    report.py      HTML + Word rendering of output
    expression.py  SPSS expression language (Compute / If)
    transform.py   Compute, Recode
    stats/         analyses: common.py helpers, descriptive, explore, ttest, anova,
                   correlate, regression, nonparametric, crosstabs, reliability, charts
  io/files.py      .rsd / .sav / csv / xlsx
  ui/              Qt
    data_editor.py main window + menus (integration point)
    dialogs/base.py VariablePickerDialog, SubDialog, DefineGroupsDialog
    dialogs/*.py   one module per menu area
tests/             pytest, one file per area
```

## File ownership (parallel work)

Create **new** files for your area. Do **not** edit these shared files — the integrator
wires them up from your report:

`ui/data_editor.py, app.py, ui/dialogs/base.py, ui/document.py, ui/models.py, ui/views.py,
ui/theme.py, ui/icons.py, ui/output_viewer.py, core/dataset.py, core/output.py,
core/report.py, core/stats/common.py, core/stats/charts.py, io/files.py, selftest.py,
requirements*.txt, packaging/*, README.md`

Also shared (integrator-owned): `ui/menus.py`, `core/selection.py`, `core/transform.py`,
`ui/dialogs/transform.py`, `ui/dialogs/selection.py`. Modules written in earlier rounds belong
to their areas — import from them freely, but do not edit them: `core/analysis/*`,
`core/doe/*`, `core/graph/*`, `core/data_ops.py`, `core/stats/*` (logistic, classify, roc,
curvefit_spss, means, repeated, manova, mixed, factor, process, …), `ui/graph/*`, `ui/doe/*`,
and the matching `ui/dialogs/*.py`.

If you need a helper from a shared file that does not exist, write it in your own module.
If something in a shared file blocks you, describe the exact change in your final report.

## Dataset API (`rsstatchem.core.dataset`)

- `Dataset(df=None, variables=None, name=...)`; `ds.df` (RangeIndex; numeric → float64 with
  NaN = system-missing, string → object, date → datetime64), `ds.variables: list[Variable]`,
  `ds.n_cases`, `ds.n_vars`, `ds.names()`, `ds.var(name)` (case-insensitive),
  `ds.index_of(name)`, `ds.has_name(name)`, `ds.meta: dict` (JSON-serialisable, saved in .rsd).
- `ds.analysis_series(name)` → column with user-missing codes and empty strings as NaN.
  **Use this (or `common.numeric` / `common.listwise`) in analyses**, never raw `ds.df`, so
  missing values are honoured.
- Edits: `ds.add_variable(name=None, vtype="numeric"|"string", index=None, values=None)` →
  Variable, `ds.set_value(row, col, text)`, `ds.delete_variables`, `ds.insert_cases`,
  `ds.sort_cases(names, ascending)`, `ds.rename_variable`, `ds.change_type`.
- `Variable` fields: name, type ("numeric"/"string"/"date"), width, decimals, label,
  value_labels {code: label}, missing [codes], columns, align, measure
  ("scale"/"ordinal"/"nominal"), role. `var.title` = label or name; `var.label_for(v)`.
- `NUMERIC, STRING, DATE` constants; `format_code(v)` prints 1.0 as "1".

## Output API (`rsstatchem.core.output`)

An analysis returns one `Section(title)`:
- `section.log(text)` — the SPSS-style syntax line(s), always add one.
- `section.table(Table(title, columns, rows, row_header_cols=1, footnotes=[...]))`
  - `columns`: list of str, or `(group, sub)` tuples for a two-level header (adjacent equal
    groups merge). Leading row-header columns are usually `""`.
  - `rows`: list of lists of **already formatted strings**.
- `section.note(title, text, style="note"|"warning")`
- `section.image(title, png_bytes, width=560)` — render matplotlib with the OO API:
  `fig = Figure(figsize=..., dpi=120); FigureCanvasAgg(fig); ...; fig.savefig(buf, format="png")`.
  Never use `pyplot` in core code.

Formatting helpers (`rsstatchem.core.stats.common`): `fmt(x, decimals=3)` (SPSS style
".045"), `fmt_p(p)`, `fmt_int`, `fmt_df`, `sup("a")` → "ᵃ" footnote marker, `stars(p)`,
`numeric(ds, name)`, `listwise(ds, names)`, `groups_of(ds, dep, factor)` → `[Group(value,
label, data)]`, `value_text(var, value)`, `t_interval`, `pct(level)`, `percentile` (SPSS
HAVERAGE), `levene_rows`, `lilliefors`, `shapiro`.
Output text (table titles, row labels, notes) is in **English, matching SPSS / Origin /
Design-Expert wording**, so users can compare with those programs.

## Dialog API (`rsstatchem.ui.dialogs.base`)

`VariablePickerDialog(ds, title, parent)` reproduces the SPSS dialog pattern:
- `add_target(key, label, accepts=is_numeric|None, single=False, required=True,
  min_items=None)`; `add_pair_target(key, label)`; `add_option_button(text, callback)`;
  `add_checkbox(opt_key, text)`; `self.options_layout` (QVBoxLayout under the lists);
  `self._option_column` (right-hand button column); call `self.finish()` last in `__init__`.
- Results: `selected(key)`, `selected_one(key)`, `pairs(key)`.
- Options: class attribute `DEFAULTS = {...}`; live copy in `self.opts` (remembered per class
  for the session). `ci_missing_options()` and `level()` for confidence-interval dialogs.
- Hooks: `validate() -> str | None` (error keeps dialog open), `targets_changed()`.
- `SubDialog(title, parent, columns=2)` builder for Options…/Statistics… windows:
  `box = dlg.group(title)`, `dlg.check(box, key, text, checked)`,
  `dlg.radio(box, key, [(value, text)], current)`, `dlg.number(box, key, text, value, min,
  max, decimals)`, `values = dlg.run()` → dict or None.
- `DefineGroupsDialog`, `DefineRangeDialog`; `ui/dialogs/compare.py: GroupingMixin` adds a
  grouping variable with Define Groups/Range.

### Analysis protocol (how the menu runs your dialog)

```python
dlg = YourDialog(ds, editor)        # editor = DataEditorWindow
if dlg.exec():
    result = dlg.compute()          # -> Section, or an object with .section and .saved
```
- Calculators that work without data (e.g. molar mass) set class attribute
  `needs_data = False`; otherwise the menu refuses to open on an empty dataset.
- If the dialog changes data, set class attribute `modifies_data = True`; `compute()` is then
  run inside an undoable `doc.mutate(...)` and the Data View refreshes.
- To save new variables without modifying data yourself, return an object with
  `.section` and `.saved = {"BASE": (label, pandas.Series aligned to ds.df.index[,
  decimals])}`; the
  editor creates `BASE_1`, `BASE_2`… (like SPSS PRE_1, RES_1).
- Long work: keep it synchronous but reasonably fast (datasets of a few thousand cases);
  bootstraps of a few thousand resamples should finish in seconds (vectorise).
- **Undo and Select Cases / Split File / Weight Cases are automatic**: a data-changing
  dialog is undoable through `doc.mutate`, and for other dialogs the editor temporarily sets
  `dlg.ds` to the filtered / weighted / split view and calls `compute()` once per split
  group. So always read data from `self.ds` inside `compute()`; never cache data in
  `__init__`.

### Plain actions (windows, wizards)

A menu entry can also be a callable `fn(editor)`. Useful editor/context members:
`editor.ds`, `editor.doc.mutate(fn)`, `editor.ctx.show_output(section)`,
`editor.ctx.new_editor(dataset)` (opens a new Data Editor window and returns it),
`editor.ctx.settings` (QSettings; keys like "last_dir"), `editor.statusBar()`.
Keep a strong reference to any top-level window you open (e.g. a module-level list).
Windows inherit the app's red/white theme; use `QMainWindow`/`QDialog` normally.

## Your module must export

1. `MENUS: dict[str, list]` — where your entries go, keyed by menu path, e.g.
   `{"Analyze/Regression": [("Binary Logistic...", BinaryLogisticDialog)], "Graphs":
   [("Graph Studio...", open_graph_studio), None]}`. An entry is `(text, DialogClass)`,
   `(text, callable)`, `None` (separator) or `(submenu title, [entries])`. Use `&&` for a
   literal ampersand in menu text.
2. `def selftest() -> None` — headless, no Qt widgets, exercises your core code on synthetic
   data and raises on failure (it runs inside the packaged app via `--selftest`).
3. Tests in `tests/test_<area>.py` comparing results with reference implementations
   (scipy / statsmodels / pingouin) or published textbook values.

## Style

- `from __future__ import annotations`, type hints, module docstring explaining scope and
  which program the feature mirrors; few comments (only for non-obvious "why").
- Lines ≤ 100 characters. Keep core (no Qt) separate from UI.
- Match the existing dialogs' look (see `ui/dialogs/compare.py`, `ui/dialogs/regression.py`).

## Final report (what to send back)

Files created, the `MENUS` you export (module path), dependencies added, shared-file
changes you need, test results (command + pass count), screenshots you checked, and an
honest list of what is not implemented or approximated.
