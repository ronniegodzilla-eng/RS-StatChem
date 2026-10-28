# Image Analysis module: Phase 0 contracts

This document fixes the interfaces that the four parallel work packages (WP-A shell/I-O, WP-B
Process, WP-C Analyze/ROI, WP-D Image menu) build on. Background, licences and the feature list
are in `docs/IMAGEJ_RESEARCH.md`; read `docs/DEVELOPMENT.md` for the general module contract.

**Phase 0 files are read-only for the work packages.** If a contract blocks you, describe the
exact change in your report; the integrator applies it. Files marked "owned by WP-x after
Phase 0" below are stubs or minimal versions that the named package replaces or extends.

## 1. Ownership

| Files | Owner |
|---|---|
| `core/imaging/{__init__,image,calibration,lut,options,session,commands,menu_layout,results,selftest}.py`, `core/imaging/roi/{__init__,shapes}.py` | Phase 0 (read-only) |
| `ui/imagej/{menus,commands_shell}.py`, `ui/imagej/dialogs/{__init__,generic}.py` | Phase 0 (read-only) |
| `scripts/imagej_golden.py`, `scripts/java/BatchImageJ.java`, `tests/ij_golden.py`, `tests/data_imagej/cases/_lib.ijm`, `tests/test_imagej_contracts.py` | Phase 0 (add golden cases, see section 9) |
| `ui/imagej/{__init__,bar,toolbar,windows,image_window,canvas,roi_tools,icons,commands_file,commands_edit,commands_plugins}.py`, `ui/imagej/frames/{log_window,text_window,recorder,command_finder,color_picker}.py`, `core/imaging/io/*`, `core/imaging/{project,macro}.py`, `tests/test_imagej_{shell,io}.py` | **WP-A** |
| `core/imaging/{filters,threshold,binary,maxima,background,fft,math_ops,contrast}.py`, `core/imaging/fiji/{local_threshold,clahe,morpho}.py`, `ui/imagej/commands_process.py`, `ui/imagej/frames/{threshold_adjuster,color_threshold}.py`, `ui/imagej/dialogs/process_*.py`, `tests/test_imagej_process.py` | **WP-B** |
| `core/imaging/{measure,particles,histogram,profile,surface,gels,calibrate,scalebar,fractal}.py`, `core/imaging/roi/{mask,geometry,wand,roi_io,selection_ops}.py`, `core/imaging/fiji/{coloc,skeleton_analysis}.py`, `ui/imagej/commands_analyze.py`, `ui/imagej/frames/{results_window,roi_manager,histogram_window,plot_window,point_counter}.py`, `ui/imagej/dialogs/analyze_*.py`, `tests/test_imagej_{analyze,roi}.py` | **WP-C** |
| `core/imaging/{convert,adjust,stacks,project3d,transform,color,lut_tables}.py`, `core/imaging/fiji/{bleach,kymograph}.py`, `ui/imagej/commands_image.py`, `ui/imagej/frames/{bc_adjuster,channels_tool,ortho_views}.py`, `ui/imagej/dialogs/image_*.py`, `tests/test_imagej_image.py` | **WP-D** |

Shared app files touched in Phase 0 (integrator-owned as always): `requirements.txt`,
`packaging/rsstatchem.spec`, `rsstatchem/ui/qt_lifecycle.py`, `rsstatchem/selftest.py`.

Commands per area: `menu_layout.labels_of_area("file" | "edit" | "plugins" | "image" |
"process" | "analyze")` lists exactly which menu labels each package implements (file 81,
edit 28, plugins 22, image 114, process 87, analyze 53; shell 9 are done). Rules
(`menu_layout.OWNER_RULES`): Edit > Selection is WP-C; Image > Adjust > Threshold... and Color
Threshold... are WP-B; Process > Repeat Command is WP-A; Window and Help are the bar's.

**Licences**: port ImageJ 1.x code (public domain) with a header such as "Ported from ImageJ
1.54 ij/process/AutoThresholder.java (public domain, NIH; W. Rasband et al.)". Never port
GPL/LGPL code (Fiji plugins, and the 4 AutoThresholder methods Intermodes, MinErrorI, Minimum,
Percentile which came from GPL Matlab code); write those from the papers.

## 2. Image model: `core/imaging/image.py`

`ImageDoc` is ImageJ's ImagePlus + ImageStack + display state.

* **Pixels**: `doc.pixels` is `(T, Z, C, Y, X)` for grayscale / 8-bit colour, dtype `uint8`
  (GRAY8, COLOR_256), `uint16` (GRAY16) or `float32` (GRAY32); RGB is `(T, Z, C, Y, X, 3)`
  `uint8` (C normally 1). `doc.type` uses ImagePlus constants `GRAY8=0, GRAY16=1, GRAY32=2,
  COLOR_256=3, COLOR_RGB=4`; `bit_depth` is 8/16/32/24.
* **Construction**: `ImageDoc.from_array(array, axes="YX"|"ZYX"|"CYX"|"TZCYX"|"YXS"...,
  title=...)` (bool -> 0/255, int16 -> 16-bit with signed calibration, float64 -> float32);
  `ImageDoc.new(w, h, type, channels, slices, frames, fill)`; `doc.duplicate(title)`;
  `doc.replace_pixels(new_pixels, type)` after crop/resize/conversion/add slice.
* **Positions** are 0-based `(c, z, t)` in `doc.position`; `set_position(c, z, t)`.
  `stack_index(c, z, t)` is ImageJ's czt stack order (ImageJ index = it + 1), inverse
  `position_of(index)`. `plane(c, z, t)` returns a writable **view** (defaults: current
  position); `set_plane`, `plane_at(index)`, `iter_planes(all_planes, channel)`.
* **Display**: per channel `display_ranges[c] = (min, max)` (raw values) and `luts[c]`
  (`LUT`); `display_range`/`lut` for the current channel, `set_display_range`,
  `reset_display_range()` (0-255 for 8-bit/RGB, plane min/max for 16/32-bit),
  `is_inverted_lut()`. Composites: `composite_mode` ("composite" | "color" | "grayscale"),
  `active_channels`.
* **Threshold**: `doc.threshold = Threshold(lower, upper, mode) | None`, raw pixel values,
  mode "red" | "bw" | "over_under" | "none". WP-B sets it; WP-C reads it ("Limit to
  threshold", Analyze Particles).
* **Selection / overlay**: `doc.roi` (`Roi | None`), `doc.overlay` (`Overlay | None`).
* **Metadata**: `title`, `path`, `calibration`, `slice_labels` (list per stack index),
  `properties` (dict str->str), `info` (Show Info text), `hyperstack`, `changed`, `id`
  (negative, unique like ImageJ).
* **Undo** (single level, ImageJ semantics): `snapshot(label)` stores pixels, type,
  calibration, LUTs, ranges, labels, ROI, overlay, threshold; `undo()` swaps (undo twice =
  redo); `can_undo()`, `forget_undo()`. The registry snapshots automatically before an
  undoable command, so command functions normally do not call it.
* `raw_value(x, y)`, `value(x, y)` (calibrated), `plane_min_max()`, `subtitle()` (the info line
  "256x254 pixels; 8-bit (inverting LUT); 64K"), `size_bytes()`.

## 3. Calibration, LUTs, options

* `calibration.Calibration`: `pixel_width/height/depth`, `unit` (+ optional `y_unit`,
  `z_unit`), origins, `invert_y`, `frame_interval`, `time_unit`, value calibration
  `function` + `coefficients` (ImageJ CurveFitter codes; `NONE=20`, `UNCALIBRATED_OD=21`),
  `value_unit`. `scaled()`, `units`, `get_x/get_y/get_z`, `set_scale(pixels, known, unit,
  aspect)`, `cvalue(raw)` (scalar or array), `ctable(bit_depth)`, `raw_value(value, depth)`,
  `set_signed16()`. WP-C adds fitting functions with `register_value_function(code, fn)`.
* `lut.LUT(red, green, blue, name)` (uint8[256] each): `grays()`, `from_color(r, g, b)`,
  `from_array((256,3))`, `inverted()`, `is_inverted()`, `is_grayscale()`, `as_array()`.
  `NAMED_LUTS` / `register_lut(name, factory)` / `named_lut(name)`: Phase 0 has Grays and
  the primary colours; **WP-D** registers Fire, Ice, Spectrum, 3-3-2 RGB, ... from its own
  `core/imaging/lut_tables.py`. `to_8bit(plane, min, max)` is ImageJ's display mapping
  (ByteProcessor / ShortProcessor / FloatProcessor `create8BitImage`); `render(doc, c, z, t)`
  returns the RGB uint8 image the canvas shows. Type conversions with scaling (WP-D) must use
  ImageJ's own conversion rounding, not necessarily `to_8bit`.
* `options.Options`: global preferences (black background **False** like ImageJ, binary
  iterations/count/pad edges/EDM output, conversions, divide-by-zero value, I/O, colours,
  line width, wand, point tool, Set Measurements keys `measurements` + `redirect` +
  `decimal_places`, `threshold_method`, `extra` dict for anything else). One instance per
  session (`session.options`); the bar persists it as JSON under QSettings key
  `imagej/options`. Add fields only with defaults (integrator change request).

## 4. ROIs: `core/imaging/roi/`

* Coordinates: (0, 0) is the top-left **corner** of the top-left pixel; pixel (i, j) covers
  [i, i+1) x [j, j+1) and its centre is (i+0.5, j+0.5).
* Classes (`shapes.py`, dataclasses): `RectRoi(x, y, width, height, corner_diameter)`,
  `OvalRoi(x, y, width, height)`, `PolygonRoi(xs, ys, type=POLYGON|FREEROI|TRACED_ROI|
  POLYLINE|FREELINE|ANGLE)`, `LineRoi(x1, y1, x2, y2, width, arrow)`, `PointRoi(xs, ys,
  counters, positions)`, `RotatedRectRoi`, `EllipseRoi`, `TextRoi`, `CompositeRoi(parts=[(op,
  roi)])` with op "or"/"and"/"xor"/"not"; `Overlay(rois, show_labels, ...)`. Common keyword
  fields: `name`, `stroke_color`, `fill_color`, `stroke_width`, `position=(c, z, t)` (1-based,
  0 = all), `properties`. `type` is ImageJ's Roi type constant. Methods: `bounds()`,
  `polygon()` (float vertices), `is_area()`, `is_line()`, `get_mask(shape)`,
  `contains(x, y)`, `translate(dx, dy)`, `copy()`.
* **Mask contract**: `roi.get_mask((H, W)) -> bool[H, W]` calls
  `roi.mask.rasterize(roi, shape)`. Phase 0 already equals ImageJ for the golden rectangle and
  all 7 golden ovals; polygons are pixel-centre even-odd and differ from ImageJ's
  PolygonFiller by a few pixels (2 on `roi_08`; the self-intersecting `roi_11` differs more).
  **WP-C** owns `mask.py` and makes polygons exact (golden `masks/roi_*`). Line/point/text
  ROIs have an empty mask.

## 5. Results tables: `core/imaging/results.py`

`ResultsTable(title)`; `ResultsTable.results()` for a Results-style table (ImageJ standard
headings `DEFAULT_HEADINGS` first in fixed order, then custom columns; row numbers shown).
`increment_counter()`, `add_value(col, v)`, `add_label(s)`, `add_row({...}, label=...)`,
`set_value(col, row, v)`, `get_value`, `get_label`, `column(name) -> ndarray`, `row(i)`,
`headings`, `delete_row`, `delete_column`, `rename_column`, `reset()`, `sort`, `copy`,
`summarize()` (Mean/SD/Min/Max rows), `format_value` (AUTO_FORMAT: integers without
decimals, others `precision`=3 decimals, scientific for tiny/huge values; per-column
`decimal_places`), `to_text(delimiter)`, `save(path)` (ImageJ CSV/TSV format, " " row-number
column, quotes around commas), `ResultsTable.read(path)` (also reads the goldens),
`to_dataframe()`. New columns are 0 in earlier rows (NaN with `nan_empty_cells`).
The session's Results table is `session.results`; other tables `session.tables[title]`.

## 6. Commands, options strings, dialogs, recorder: `core/imaging/commands.py`

* Registry key = **exact ImageJ command name** from `menu_layout` ("Gaussian Blur...",
  "Properties... " with trailing space for Edit > Selection, "In [+]", "Close-").
* `Command(label, func, fields, needs_image=True, types=None, supports_stack=False,
  undoable=True, roi="optional"|"required"|"area"|"line"|"none", recordable=True,
  ui_action=None)`. Register in your `commands_<area>.register(registry)`:

  ```python
  from rsstatchem.core.imaging import commands as C
  from rsstatchem.core.imaging.image import COLOR_RGB, GRAY8, GRAY16, GRAY32

  def register(registry):
      registry.add("Gaussian Blur...", gaussian_blur,
                   [C.number("Sigma (Radius):", 2.0, 2), C.checkbox("Scaled Units", False)],
                   supports_stack=True, types=(GRAY8, GRAY16, GRAY32, COLOR_RGB))

  def gaussian_blur(session, image, sigma, scaled, stack=False):
      for c, z, t in image.iter_planes(stack):
          image.set_plane(filters.gaussian(image.plane(c, z, t), sigma), c, z, t)
  ```
* `func(session, image, **values)`. Return `None` (in-place change: the registry marks the
  image changed and emits "image_updated"), a new `ImageDoc` (opened as a new window), a
  `ResultsTable` (shown/refreshed), or a list of those. Raise `CommandError("message")` for
  user errors (shown like ImageJ's message box); call `session.progress(f)` in long loops
  (raises `Cancelled` when the user pressed Esc).
* `fields` is a list of `Field` (constructors `number`, `slider`, `checkbox`, `choice`,
  `radio`, `string`, `message`, `checkbox_group`; kinds also "file"/"directory") or a callable
  `(session, image) -> list` for image-dependent dialogs. `supports_stack=True` adds a
  "Process all N slices" checkbox (key `stack`) for stacks; accept `stack=False` in `func`.
* **Option strings** (ImageJ `Macro`): key = first word of the label, lower case, up to ':'
  (`macro_key`); `key=value`, `key=[value with spaces]`; a checkbox is True when its key
  appears, **False when absent**; missing numbers/choices/strings keep defaults. Override a
  key with `Field(key=...)` when two labels share a first word. `parse_options`,
  `values_from_options(fields, options)`, `format_options(fields, values)`.
* Running: `session.run("Gaussian Blur...", "sigma=2 stack")` (macro style, active image) or
  `registry.run(session, label, options, image=..., values=...)`; it checks image/type/ROI,
  snapshots, runs, notifies, sets `session.last_command` (Repeat Command) and records.
* `ui_action(bar, command)`: for commands that open a window instead of a dialog (Threshold...,
  B&C, ROI Manager..., Histogram windows). Such commands may also provide a headless `func`
  for macros (`setAutoThreshold`-like option strings).
* Recorder: `session.recorder = Recorder("Macro"|"Python")`; lines like
  `run("Gaussian Blur...", "sigma=2 stack");` or `ij.run('Gaussian Blur...', sigma=2,
  stack=True)`; `record_line()` for `open(...)` etc.; `listeners` for the Recorder window.
* Qt dialog: `ui/imagej/dialogs/generic.GenericDialog(title, fields, parent, preview=fn)`;
  `.get()` -> values dict or None, `.values()`, `.set_value(key, v)`. The bar uses it
  automatically for commands without `ui_action`.

## 7. Session and events: `core/imaging/session.py`

`Session` holds `options`, `registry`, `images`, `active`, `results`, `tables`,
`roi_manager` (list of Roi), `log_lines`, `recorder`, `last_command`, `clipboard`. Methods:
`add_image(doc)` (makes titles unique like ImageJ: "blobs-1.gif"), `close_image`,
`set_active`, `image_updated(doc)`, `find_image(title|id|index)`, `table(title)`,
`show_table(table)`, `log(text)`, `status(text)`, `progress(f)`, `run(label, options)`.
Events (`session.on(event, cb)`): "image_added", "image_closed", "image_updated",
"active_changed", "table_updated", "log", "status", "progress". The UI's `WindowManager`
(`ui/imagej/windows.py`) turns them into windows; core code never imports Qt.

## 8. Menu layout: `core/imaging/menu_layout.py`

`MENU_LAYOUT` is ImageJ 1.54p's tree (generated from `IJ_Props.txt` in ij-1.54p.jar plus the
skeleton of `ij/Menus.java`): `Menu(title, items)`, `MenuItem(label, shortcut, ij_class)`,
`None` = separator. 394 unique command labels; Java-only entries are left out (listed in the
module docstring); Help ends with "Check for Updates..." and "About RS-StatChem...". Shortcut
"n" = Ctrl+N, "N" = Ctrl+Shift+N (`qt_shortcut`); `PLAIN_KEYS` lists the unmodified keys
(+ - < > \ / ]). Queries: `iter_items()`, `iter_menus()`, `find(label)`, `path_of(label)`,
`owner_of(label)`, `labels_of_area(area)`, `AREAS` (area -> WP and module). `ui/imagej/
menus.py` builds the QMenus; entries without an available command are disabled, so each
package's entries light up when its `register` adds them (`bar.refresh_menus()`).

## 9. UI skeleton: `ui/imagej/`

* `open_workspace(ctx, path)` returns the single `ImageJBar` (kept in `windows.BARS`),
  creating it once per app context; returns immediately; open errors are deferred.
* `ImageJBar` (`bar.py`, WP-A): `ctx` contract (keeps `self.ctx`, `ctx.window_closed(self)`
  in `closeEvent`, Window > Modules via `add_module_menu`, Help > Check for Updates / About
  via `commands_shell`), `session`, `toolbar` (`ToolBar`, tool ids = ImageJ Toolbar
  constants), status line `show_status`, `show_progress`, `actions_by_label`, `menus`
  ("File>Import" -> QMenu), `run_command(label, options=None)`, `open_path(path)`,
  `refresh_menus()`, `error(text, title)` (tests set `bar.message_hook`), drag & drop open.
  At start-up it imports every `commands_<area>` module and calls `register` (a failing
  module is reported on stderr and skipped).
* `windows.py` (WP-A): `WINDOWS` (all image/aux windows; cleared by qt_lifecycle), `BARS`,
  `register/unregister(window)`, `window_for(doc)`, `WindowManager`;
  `TABLE_WINDOW_FACTORY(bar, table)` (WP-C sets it to the Results window class; the window
  needs `set_table(table)`), `LOG_WINDOW_FACTORY(bar, None)` (WP-A; needs `append(line)`).
* `ImageWindow(doc, bar)` (`image_window.py`, WP-A): `.doc`, `.canvas`, `.refresh()`,
  `.update_title()` ("blobs.gif (150%)"), info line, stack slider, status readout
  (`status_text(x, y)`); activation sets the session's active image; closing removes it.
* `ImageCanvas` (`canvas.py`, WP-A): `ZOOM_LEVELS` (ImageJ's 23 levels), `magnification`,
  `set_magnification(m, center)`, `zoom_in(at)`, `zoom_out(at)`, `refresh()`,
  `repaint_overlay()`, `image_to_screen`, `screen_to_image`, signal `mouse_moved(x, y)`,
  `paint_hooks` (callables `fn(painter, canvas)` for tools/overlays of other packages). It
  draws the ROI and overlay outlines; ctrl+wheel zooms.
* Platform note for WP-A: `bar.image_window_menubar(win)` returns None in Phase 0; on macOS
  image windows should get a menu bar that shares the bar's QActions.

## 10. Golden data from the real ImageJ

`python scripts/imagej_golden.py [--case NAME]` downloads `net.imagej:ij:1.54p` from Maven
Central into `build/imagej/` (SHA-256 `2e1a0996...41e20` pinned), fetches the public-domain
inputs from github.com/imagej/ImageJ `tests/data` at commit `a376e43f` (blobs.gif,
Cell_Colony.jpg; SHA-256 pinned; lena is never used), writes the seeded synthetic inputs
(`synth8/16/32.tif` 64x64, `particles8.tif` 160x128), then runs every
`tests/data_imagej/cases/<case>.ijm` (with `_lib.ijm` prepended) through
`scripts/java/BatchImageJ.java` and stores `tests/data_imagej/golden/<case>/` (CSV with 9
decimals; `images.npz` with each saved TIFF, array name = file stem) plus `manifest.json`.
`plain java -jar ij.jar -batch` does not work with JDK 26 (no java.applet) or headless (menu
bar / GenericDialog are AWT), hence the driver; it needs a desktop session and exits with the
dialog text if ImageJ shows any message. pytest never needs Java. Load goldens in tests with
`tests/ij_golden.py` (`golden_csv`, `golden_images`, `thresholds`, `input_path`, `blobs`).

Initial golden set (ImageJ 1.54p, black background off):

| Case | Content |
|---|---|
| `thresholds/thresholds.csv` | lower/upper of all 17 methods, with and without "dark", plus legacy `setAutoThreshold()`, on blobs.gif, Cell_Colony.jpg, synth16, synth32 (140 rows). blobs Default = 126-255 (dark: 0-125). |
| `particles/` | Analyze Particles on blobs.gif (Default threshold) and particles8 (128-255), variants default / exclude / include / size=100-Infinity / circularity=0.85-1.00, every measurement, plus the Summary row. **blobs.gif: 64 particles, total area 22243** (confirms the quoted figure); exclude edges 46, size>=100 52, circularity>=0.85 43. |
| `measure/` | Measure with every measurement on blobs.gif: whole image, rectangle, oval, polygon, line; area ROIs calibrated (4 px/um) and with "limit to threshold". |
| `filters/images.npz` | Gaussian Blur sigma 0.5/2/10 on synth8/16/32; Median/Mean/Minimum/Maximum/Variance r = 0.5/1/2/5 (8-bit) and r = 2 (16/32-bit); inputs as `input_synth*`. |
| `masks/` | Convert to Mask of blobs (Default, Otsu; black background off/on) and ROI masks of 12 rect/oval/polygon shapes (`roi_specs.csv`). |

To add a case: write `cases/<name>.ijm` using `init()`, `saveResults(name)`,
`saveTable(title, name)`, `saveImage(name)`, run `--case <name>`, commit the macro, inputs and
golden folder. Two macro-language pitfalls seen: write `func("" + arr[i] + "x")` (a call
argument starting with an array element fails to parse), and use `saveResults` instead of
`saveAs("Results")` (without ImageJ's main window it may save the Summary table).
Tolerances (research section 5): exact for counts, areas, thresholds, masks; +/-1 grey level
for 8-bit Gaussian/background/bicubic; rel 1e-5 for float statistics.

## 11. Dependencies, packaging, selftest

* `requirements.txt`: `tifffile==2026.10.15`, `roifile==2026.9.22`, `pydicom==3.0.2`
  (pure Python). Tests that need them use `pytest.importorskip`; the core imports them lazily.
* `packaging/rsstatchem.spec`: hidden imports for tifffile, roifile and all pydicom
  submodules except examples/data.
* `qt_lifecycle._WINDOW_REGISTRIES` clears `rsstatchem.ui.imagej.windows.WINDOWS` and `BARS`.
* `--selftest` imports `ui.imagej.bar`, `image_window`, `dialogs.generic` and runs
  `rsstatchem.ui.imagej.selftest()` = `core/imaging/selftest.run()`: contracts, the bundled
  readers, then `selftest()` of every `commands_<area>` module that defines one (add yours;
  no Qt widgets).
