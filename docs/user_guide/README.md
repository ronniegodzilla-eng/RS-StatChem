# Panduan Pengguna (user guide) — how it is built

The PDF guide is generated from the running app on every release (`.github/workflows/build.yml`)
and attached to the release as `RS-StatChem-User-Guide.pdf`.

```
docs/user_guide/
  build.py          entry point
  preview.py        render PDF pages to PNG for checking: preview.py file.pdf outdir first last
  guide/            engine: studio.py (screenshots + annotations), markup.py, features.py, pdf.py
  content/NN_*.md   chapters, in file-name order (Indonesian)
  scenes/*.py       screenshot scenes used by the chapters
  features/<module>.tsv   Indonesian explanation of every menu command, per module
```

Build (about a minute; `--only` limits the chapters for a quick preview):

```bash
.venv/bin/python docs/user_guide/build.py --out /tmp/guide.pdf --shots /tmp/shots --only 10_
QT_QPA_PLATFORM=offscreen .venv/bin/python docs/user_guide/preview.py /tmp/guide.pdf /tmp/pages 1 40
```

`--strict` (used by CI and the tests) fails when a scene is missing or broken, or a menu command
has no explanation.

## Chapters (`content/*.md`)

See the docstring of `guide/markup.py`: `#` chapter, `##` section, `###` subsection, paragraphs,
`-` bullets, `1.` steps, `> [!TIP]` / `[!NOTE]` / `[!WARNING]` callouts, tables with `|`,
`![scene.key]` screenshots (optional `(caption)`), `!features <module>` for the generated menu
reference, inline `**bold**`, `*italic*`, `` `code` ``, `[[Menu > Item]]`, `++Ctrl+S++`.

## Scenes (`scenes/*.py`)

```python
from guide.studio import scene

@scene("stats.data_view", "Caption shown under the picture")
def data_view(st):                       # st: guide.studio.Studio
    w = st.module("stats", size=(1100, 640))   # a fresh module window
    ...                                  # put demo data in, open a dialog with .show(), ...
    shot = st.shot(w)                    # screenshot (title bar from the window title)
    shot.box(w.menuBar(), "Menu bar")    # numbered red box, text in the legend
    shot.box(st.toolbar_button(w.toolbar, "Open"), "Open data")
    shot.arrow(w.statusBar(), "Status bar", side="bottom")   # label + arrow on the picture
    shot.menu("Analyze > Compare Means", highlight="One-Way ANOVA")   # an opened menu
    return shot                          # or a list of shots: keys "x.y", "x.y.2", ...
```

Targets for `box` / `arrow` / `outline` / `crop`: a widget, a `QRect`/`QRectF` or `(x, y, w, h)`
in window coordinates, `st.tab(tabbar, "Name")`, `shot.menu_item_rect("File > Save")`.
`shot.overlay(dialog, at)` draws a dialog over the window. Rules: deterministic demo data (seeded
random numbers), no network, never `exec()` a dialog (use `show()`), keep a scene under ~10 s.
Windows are hidden after each scene by the builder.

## Feature texts (`features/<module>.tsv`)

One line per menu command, `Menu > Submenu > Item<TAB>Explanation` (labels exactly as the menu
shows them, without `&` and trailing `...`). `Menu > Submenu > *` explains a whole submenu in one
row (lists of samples, recent files, open windows, style sheets). The builder prints commands that
have no text and texts whose command no longer exists.

When a feature is added to the app, add its line here (the test suite fails otherwise) and, if
it changes a workflow shown in a chapter, update that chapter.
