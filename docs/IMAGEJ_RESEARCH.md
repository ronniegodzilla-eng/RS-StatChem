# RS-StatChem fifth module: "Image Analysis" (ImageJ 1.54 / Fiji style). Research report

Date: 2026-10-27. Scope: what ImageJ/Fiji does, how to build it in Python/PySide6 inside RS-StatChem, which licences
apply, and how to split the work between parallel agents.

Already in place (integrator): `rsstatchem/ui/modules.py` registers key **"image"** ("Image Analysis",
style "ImageJ", opener `rsstatchem.ui.imagej:open_workspace`, extensions `.tif .tiff .png .jpg .jpeg .bmp .gif
.dcm .fits .roi`, tile icon "image"). The launcher grid now has 3 columns (`i // 3, i % 3`) and
`tests/test_launcher.py` expects 5 modules. `rsstatchem/ui/imagej/__init__.py` is a placeholder
`QMainWindow`. `rsstatchem/core/imaging/` does not exist yet.

Main sources used:
* ImageJ 1.x source code, github.com/imagej/ImageJ (master = **1.54u**, `ij/ImageJ.java` VERSION). Menu definitions
  come from `IJ_Props.txt` ("Version 1.54") and `ij/Menus.java`. These are the authoritative 1.54 menu trees and
  list the class behind every command (quoted below).
* imagej.net/ij/docs/menus/{file,edit,image,process,analyze}.html, imagej.net/ij/docs/tools.html, and the ImageJ
  User Guide 1.46r (Ferreira & Rasband, imagej.net/ij/docs/guide/146.html).
* imagej.net/plugins/auto-threshold, the PyPI JSON API (wheel sizes and licences), and GitHub licence metadata of
  the Fiji plugin repositories.

---------------------------------------------------------------------------------------------------------------

## 0. Key findings (short)

1. **ImageJ 1.x is public domain.** `LICENSE.txt` in imagej/ImageJ is the NIH disclaimer: "Pursuant to Title 17,
   Section 105 ... not subject to copyright protection and is in the public domain". So its algorithms (rolling
   ball / sliding paraboloid, MaximumFinder, EDM watershed, ParticleAnalyzer, Wand, PolygonFiller, RankFilters,
   GaussianBlur, FFTFilter, skeletonize, ContrastEnhancer, EllipseFitter, Feret, RoiDecoder/Encoder) can be
   **ported line by line** into our proprietary app. This is the only practical way to match ImageJ numbers
   exactly. Keep an attribution note in each ported module.
   * One caveat: `ij/process/AutoThresholder.java` says it comes "from the Auto_Threshold plugin by G. Landini".
     Four of its methods (**Intermodes, MinErrorI, Minimum, Percentile**; lines 384, 678, 747 and 906) carry the
     comment "ported ... from Antti Niemisto's Matlab code (GPL)". Port the other 12 methods directly. Write these 4
     clean-room from the original papers, then black-box test them against ij.jar.
2. **Most Fiji plugins are GPL or LGPL.** RS-StatChem is proprietary ("Untuk penggunaan pribadi/non-komersial"), so
   **no GPL/LGPL code may be ported**. Reimplement from the papers, or use BSD libraries. Licences checked on GitHub:
   * GPL-3: Auto_Threshold, Auto_Local_Threshold, Cell_Counter, Trainable_Segmentation (Weka), Colocalisation_Analysis
     (Coloc 2), AnalyzeSkeleton, 3D_Viewer, TrackMate, CorrectBleach, register_virtual_stack_slices.
   * GPL-3+ (from pom.xml): Colour_Deconvolution, Skeletonize3D, Directionality.
   * GPL-2: Stitching, Bio-Formats, Fiji_Plugins.
   * LGPL-3: MorphoLibJ.
   * Permissive: BoneJ2 is **BSD-2** and stardist is **BSD-3** (it needs TensorFlow, though). KymographBuilder is
     **MIT**, imagej-ops is **BSD-2**, skan is BSD-3 (it needs numba).
3. **Few new dependencies are needed.** numpy 2.5, scipy 1.18, Pillow 13, matplotlib, pandas and PySide6 6.12 are
   already bundled.
   * Add **tifffile** (BSD-3, 0.28 MB pure-Python wheel). It reads ImageJ hyperstack metadata, overlays, OME-TIFF,
     LSM and STK.
   * Optionally add **roifile** (BSD-3, 20 KB) and **pydicom** (MIT, 2.4 MB pure-Python wheel).
   * Avoid opencv (44–48 MB wheels), astropy (6.7 MB wheel plus pyerfa, iers-data and PyYAML), fitsio (GPL) and
     imagecodecs (13–19 MB).
   * Treat scikit-image (BSD-3, 12 MB wheel, plus networkx, imageio and lazy_loader) as optional for phase 2. We need
     ported ImageJ code for parity anyway, and skimage results differ from ImageJ.
4. **Golden test values can be generated with the real ImageJ.** `net.imagej:ij:1.54p` is on Maven Central (2.6 MB
   jar, public domain). A JDK is on the dev Mac (`/Library/Java/JavaVirtualMachines/jdk-26.jdk`) and in CI
   (setup-java Temurin). The ImageJ repo itself ships `tests/data/` (blobs.gif, Cell_Colony.jpg, embryos.bmp, 16/32-bit
   TIFFs, 01.dcm, a FITS file, a PGM, raw files) in the public-domain repo.
   * Plan: run ImageJ headless (`java -Djava.awt.headless=true -jar ij.jar -batch case.ijm`) and commit small CSV/TIFF
     goldens. Our pytest then needs no Java.
   * I did not download or run ij.jar during this research. That is the first job of WP-0.
5. **Not verified: "blobs.gif gives 64 particles with the default threshold".** It is widely quoted, but I found no
   official page that states it (imagej.net docs and the 1.46r guide give no numbers). Treat it as a hypothesis and
   regenerate it with ij.jar.

---------------------------------------------------------------------------------------------------------------

## 1. Feature inventory (ImageJ 1.54 core + commonly used Fiji plugins)

Notation: `Label [key]` → implementing class in ImageJ (`ij.plugin.` / `ij.plugin.filter.` prefixes dropped).
P = priority. P1 = v1 must-have, P2 = second round, P3 = later/optional.

### 1.1 Main window ("ImageJ bar")
* A small floating window holds the menu bar, a row of tool icons and a status line. The status line shows
  `x=…, y=…, value=…`, calibrated units, the stack position, timings ("0.12 seconds") and a progress bar.
  * On macOS the menu bar is the global one and is also shown when an image window is active.
  * On Windows/Linux only the bar window has menus; image windows have none, but keyboard shortcuts work from them.
* Closing the bar quits ImageJ and asks to save changed images.
* The right-click popup menu on an image (from `IJ_Props popup01..11`) has: Show Info…, Properties…, Rename…,
  Measure, Histogram, Duplicate Image…, Original Scale, Record…, Find Commands…, Capture Screen.

### 1.2 Toolbar tools (`ij/gui/Toolbar.java`; IDs RECTANGLE=0 … ANGLE=14)

| Tool | Variants (right-click) | Modifiers / double-click | P |
|---|---|---|---|
| Rectangle | Rectangle, Rounded Rectangle (Rounded Rect Tool… corner diameter), Rotated Rectangle | shift=square, ctrl/cmd=resize about centre, alt=keep aspect, arrows move, alt+arrows resize; shift adds, alt subtracts (composite selections) | P1 (rotated: P2) |
| Oval | Oval, Elliptical selections (EllipseRoi), Selection Brush Tool (diameter) | shift=circle; double-click → options / brush size | P1 (brush: P2) |
| Polygon | – | click vertices, double-click or click start box to close; alt-click deletes vertex, shift-click adds | P1 |
| Freehand | – | drag | P1 |
| Straight line | Straight Line, Segmented Line, Freehand Line, Arrow tool | shift=horizontal/vertical; double-click → Line Width; line width >1 = wide line for profiles | P1 (arrow: P2) |
| Angle | – | three points; "m" measures the angle | P1 |
| Point | Point, Multi-point (counters: Cell-Counter-like types 0..n, Point Tool… options: type, colour, size, label, Auto-measure, Add to ROI Manager, Auto-next slice) | shift-click adds, alt-click deletes | P1 |
| Wand | Wand Tool… modes Legacy / 4-connected / 8-connected, tolerance, smooth | click inside an object | P1 |
| Text | Fonts… | creates a TextRoi; draw or overlay it | P2 |
| Magnifier | – | click = zoom in, right/alt-click = out, double-click = original scale | P1 |
| Scrolling (hand) | – | space bar gives a temporary hand | P1 |
| Colour picker | – | sets foreground colour, alt = background; double-click → Color Picker | P1 |
| Built-in extra tools (`builtInTools`): Arrow, Brush, Command Finder, Developer Menu, Flood Filler, LUT Menu, Overlay Brush, Pencil, Pixel Inspector, Spray Can, Stacks Menu, … via ">>" | | | P2/P3 |

### 1.3 File menu (Menus.java + IJ_Props)
* New ▸ Image… [n] (name, type 8/16/32/RGB, fill White/Black/Ramp/Noise/Random, w, h, slices), Hyperstack…, Text
  Window [N], Internal Clipboard, System Clipboard [V]. **P1** (Text Window P3)
* Open… [o]. Opens TIFF (1/8/16/32-bit, RGB, multi-page = stacks, ImageJ hyperstacks with calibration, overlays and
  ROIs), GIF (LUT kept; animated GIF = stack), JPEG, PNG (8/16-bit, RGB), BMP, PGM/PBM/PPM, DICOM (uncompressed),
  FITS, AVI, .lut, .roi, .zip (TIFF-in-zip or ROI set), .txt/.csv (results/text image). **P1**
* Open Next [O], Open Samples ▸ (30 samples downloaded from imagej.net: AuPbSn 40, Bat Cochlea, Blobs [B], Boats,
  Cardio (RGB DICOM), Cell Colony, Clown, Confocal Series, CT (16-bit DICOM), Dot Blot, Embryos, Fluorescent Cells,
  Fly Brain, Gel, HeLa Cells (48-bit RGB), Image with Overlay, Leaf, Line Graph, Mitosis (5D stack), MRI Stack, M51
  Galaxy (16-bit), Neuron (5 channels), Nile Bend, Organ of Corti (4D), Particles, T1 Head (16-bit), T1 Head
  Renderings, TEM Filter, Tree Rings) plus Cache Sample Images. **P2**: download on demand into a cache, nothing
  bundled. Open Recent ▸ **P1**.
* Import ▸: Image Sequence… (FolderOpener: count, start, step, filter, scale, sort numerically, virtual), Raw… (type,
  w, h, offset, n images, gap, white is zero, little-endian, open all, virtual), LUT…, Text Image…, Text File…,
  Results…, Table…, URL…, Stack From List…, TIFF Virtual Stack…, AVI…, XY Coordinates…. **P1**: sequence, raw,
  LUT, text image, results/table. **P2**: URL, virtual stacks, AVI, XY coords.
* Show Folder ▸ (P3), Close [w], Close All [W], Save [s] (TIFF), Revert [R], Page Setup…, Print… [p] **P2**, Quit.
* Save As ▸: Tiff…, Gif…, Jpeg… (quality from Input/Output options), Text Image…, ZIP…, Raw Data…, Image Sequence…,
  AVI…, BMP…, PNG…, PGM…, FITS…, LUT…, Selection… (.roi), XY Coordinates…, Results…, Text…. **P1** except AVI/FITS
  (P2).
* Batch ▸ (in Process): Measure…, Convert…, Macro…, Virtual Stack…. **P2**.

### 1.4 Edit menu
* Undo [z] (single level), Cut [x], Copy [c], Copy to System, Paste [v] (floating, draggable), Paste Control… (paste
  modes = Image Calculator ops + Blend + Transparent-white/zero), Clear, Clear Outside, Fill [f], Draw [d], Invert
  [I]. **P1**
* Selection ▸: Select All [a], Select None [A], Restore Selection [E], Fit Spline, Fit Circle, Fit Ellipse, Fit
  Rectangle, Interpolate, Convex Hull, Make Inverse, Create Selection (from threshold/mask), Create Mask,
  Properties… [y] (name, stroke colour/width, fill, position), Scale…, Rotate…, Translate…, Enlarge… (EDM-based,
  RoiEnlarger), Make Band…, Specify… (rect/oval, centred, scaled units), Straighten…, To Bounding Box, Line to Area,
  Area to Line, Image to Selection…, Add to Manager [t]. **P1**; Straighten and Image to Selection are P2.
* Options ▸: Appearance…, Arrow Tool…, Colors… (foreground/background/selection), Compiler…, Conversions… (scale
  when converting, weighted RGB, full-range 16-bit inversion), DICOM…, Fonts…, Input/Output… (JPEG quality, table
  extension, save column headers/row numbers, Intel byte order), Line Width…, Memory & Threads…, Misc… (divide-by-
  zero value, …), Plots…, Point Tool…, Proxy Settings…, Roi Defaults…, Rounded Rect Tool…, Startup…, Wand Tool…,
  Fresh Start, Reset…. **P1**: Colors, Conversions, Fonts, I/O, Line Width, Misc, Plots, Point, Roi Defaults, Wand,
  Reset. **Do not build** Compiler, Memory and Proxy.

### 1.5 Image menu
* Type ▸: 8-bit, 16-bit, 32-bit, 8-bit Color (median cut, n colours), RGB Color, RGB Stack, HSB Stack, HSB (32-bit),
  Lab Stack. **P1**
* Adjust ▸ (frame classes): Brightness/Contrast… [C] (min/max/brightness/contrast sliders, Auto (AUTO_THRESHOLD=5000
  algorithm), Reset, Set, Apply), Window/Level…, Color Balance…, Threshold… [T] (two sliders, 17 methods, Red / B&W
  / Over/Under display, Dark background, Stack histogram, Don't reset range, Raw values, 16-bit histogram, Auto, Apply,
  Reset, Set), Color Threshold… (HSB/RGB/Lab/YUV, pass/stop per channel, method, Select/Sample/Filtered/Macro), Size…
  (w, h, depth, constrain, average when downsizing, interpolation None/Bilinear/Bicubic), Canvas Size… (position,
  zero fill), Line Width…, Coordinates…. **P1** (Window/Level, Color Balance, Coordinates: P2)
* Show Info… [i] (incl. DICOM/FITS header and TIFF tags), Properties… [P] (channels/slices/frames, unit, pixel
  width/height, voxel depth, frame interval, origin, "global"). **P1**
* Color ▸: Split Channels, Merge Channels… (up to 7, composite, keep source), Arrange Channels…, Invert LUTs, Channels
  Tool… [Z] (Composite/Color/Grayscale, channel checkboxes), Stack to RGB, Make Composite, Show LUT, Display LUTs,
  Edit LUT…, Color Picker… [K]. **P1** (Edit LUT, Display LUTs: P2)
* Stacks ▸: Add Slice, Delete Slice, Next Slice [>], Previous Slice [<], Set Slice…, Images to Stack, Stack to Images,
  Make Montage…, Reslice [/]…, Orthogonal Views [H], Z Project… (Average, Max, Min, Sum, SD, Median; start/stop; all
  time frames), 3D Project… (Projector: nearest/brightest/mean, axis, angle, interval, opacity, depth cueing,
  interpolate), Plot Z-axis Profile, Measure Stack…, Label…, Statistics. **P1** (3D Project, Ortho Views: P2)
  * Animation ▸: Start [\], Stop, Animation Options… (fps, first/last, loop back and forth). **P1**
  * Tools ▸: Combine…, Concatenate…, Grouped Z Project…, Insert…, Magic Montage Tools, Make Substack…, Montage to
    Stack…, Plot XY Profile, Reduce…, Remove Slice Labels, Reverse, Set Label…. **P1/P2**
* Hyperstacks ▸: New Hyperstack…, Stack to Hyperstack… (order xyczt etc., display mode), Hyperstack to Stack, Reduce
  Dimensionality…, Make Subset…. **P1**
* Crop [X], Duplicate… [D] (title, range, channels/slices/frames, ignore selection), Rename…, Scale… [E] (x/y/z
  factors or size, interpolation, average when downsizing, process entire stack, create new window). **P1**
* Transform ▸: Flip Horizontally, Flip Vertically, Flip Z, Rotate 90 Degrees Right/Left, Rotate… (angle, grid lines,
  interpolation, fill with background, enlarge), Translate…, Bin… (x/y/z, average/median/max/min/sum), Image to
  Results, Results to Image. **P1**
* Zoom ▸: In [+], Out [-], Original Scale [4], View 100% [5], To Selection, Scale to Fit, Set…, Maximize. Zoom levels
  (`ImageCanvas.zoomLevels`): 1/72, 1/48, 1/32, 1/24, 1/16, 1/12, 1/8, 1/6, 1/4, 1/3, 1/2, 0.75, 1, 1.5, 2, 3, 4, 6,
  8, 12, 16, 24, 32. **P1**
* Overlay ▸: Add Selection… [b], Add Image…, Hide Overlay, Show Overlay, From ROI Manager, To ROI Manager, Remove
  Overlay, Toggle Overlay, List Elements, Flatten [F], Labels…, Measure Overlay, Overlay Options… [Y]. **P1** (Add
  Image, Labels: P2)
* Lookup Tables ▸: Invert LUT, Apply LUT, Fire, Grays, Ice, Spectrum, 3-3-2 RGB, Red, Green, Blue, Cyan, Magenta,
  Yellow, Red/Green. These are generated by formulas in LutLoader.java (public domain). Fiji adds many more (16
  colors, glasbey, mpl-viridis/magma/…, phase, Thermal …). Viridis-family LUTs can be produced from matplotlib
  colormaps; Glasbey is a published palette. **P1**

### 1.6 Process menu
* Smooth [S] (3×3 mean), Sharpen (kernel −1…12…−1, /4), Find Edges (Sobel √(gx²+gy²)), Find Maxima… (prominence,
  strict, exclude edge maxima, light background, above lower threshold; output Single Points / Maxima Within
  Tolerance / Segmented Particles / Point Selection / List / Count), Enhance Contrast… (saturated %, normalize,
  equalize (sqrt-modified; alt = classic), process all slices, use stack histogram). **P1**
* Noise ▸: Add Noise (σ=25), Add Specified Noise…, Salt and Pepper (2.5%+2.5%, 8-bit), Despeckle (3×3 median, r=1),
  Remove Outliers… (radius, threshold, bright/dark), Remove NaNs… (median, radius). **P1**
* Shadows ▸: North … Northwest (8 3×3 kernels), Shadows Demo. **P1** (trivial)
* Binary ▸: Make Binary, Convert to Mask, Erode, Dilate, Open, Close-, Median, Outline, Fill Holes, Skeletonize,
  Distance Map, Ultimate Points, Watershed, Voronoi, Dither…, Options… (iterations, count, black background, pad
  edges when eroding, EDM output Overwrite/8/16/32-bit, "do" command for preview). **P1**
* Math ▸: Add, Subtract, Multiply, Divide, AND, OR, XOR, Min, Max, Gamma, Set, Log, Exp, Square, Square Root,
  Reciprocal, NaN Background, Abs, Macro… (expression in v, x, y, z, w, h, d, a; functions sin, cos, sqrt, …). **P1**
* FFT ▸: FFT (FHT, power spectrum 8-bit log-scaled, padding to a power of 2), Inverse FFT (masked by black/white
  painting on the spectrum), Redisplay Power Spectrum, FFT Options… (raw power spectrum, FHT, complex FT, do forward
  on inverse), Bandpass Filter… (large/small structure sizes, suppress stripes None/Horizontal/Vertical, tolerance of
  direction, autoscale, saturate, display filter, process stack), Custom Filter…, FD Math… (correlate, convolve,
  deconvolve, do inverse), Swap Quadrants, Make Circular Selection…. **P1** (FD Math, Custom: P2)
* Filters ▸: Convolve… (text kernel, normalize, preview), Gaussian Blur… (sigma, scaled units), Median…, Mean…,
  Minimum…, Maximum…, Unsharp Mask… (σ, weight), Variance…, Top Hat… (radius, light background, don't subtract),
  Gaussian Blur 3D…, Median/Mean/Minimum/Maximum/Variance 3D…, Show Circular Masks…. **P1** (3D filters: P2)
* Image Calculator… (Add, Subtract, Multiply, Divide, AND, OR, XOR, Min, Max, Average, Difference, Copy, Transparent-
  zero; create new window; 32-bit result; stacks). Subtract Background… (rolling ball radius, light background,
  separate colours, create background, sliding paraboloid, disable smoothing, preview). Repeat Command [R]. **P1**

### 1.7 Analyze menu
* Measure [m]. Area ROIs give the selected measurements. Lines give Length, Angle, and with profile Mean/Min/Max.
  Points give X, Y, value. Angle ROIs give Angle.
* Analyze Particles… (size range in pixel or calibrated units, "pixel units", circularity range; Show: Nothing,
  Outlines, Bare Outlines, Ellipses, Masks, Count Masks, Overlay, Overlay Masks; Display results, Clear results,
  Summarize, Add to Manager, Exclude on edges, Include holes, Record starts, In situ show, Overlay, Composite ROIs).
* Summarize (Mean/SD/Min/Max rows), Distribution… (histogram of a results column), Label, Clear Results.
* Set Measurements…: Area, Mean gray value, Standard deviation, Modal gray value, Min & max gray value, Centroid,
  Center of mass, Perimeter, Bounding rectangle, Fit ellipse, Shape descriptors, Feret's diameter, Integrated
  density, Median, Skewness, Kurtosis, Area fraction, Stack position, Limit to threshold, Display label, Invert Y
  coordinates, Scientific notation, Add to overlay, NaN empty cells; Redirect to; Decimal places (0–9).
* Set Scale… (distance in pixels, known distance, pixel aspect ratio, unit, global; prefilled from a line ROI),
  Calibrate… (function: None, Straight Line, 2nd–4th Degree Poly, Exponential, Power, Log, Rodbard, Gamma, Uncalibrated
  OD…; unit; known values text; global; show plot).
* Histogram [h] (bins, x min/max, y max, stack histogram; window with List / Copy / Log / Live, statistics shown).
* Plot Profile [k] (line, wide line = average, rectangle = column average (alt = row average), polyline; Plot window
  with List / Data>> / More>> / Live, X/Y range set, log axes).
* Surface Plot… (IJ1 SurfacePlotter: polygon multiplier, draw wireframe/axis, shade, black fill, smooth).
* Gels ▸: Select First Lane [1], Select Next Lane [2], Plot Lanes [3], Re-plot Lanes, Reset, Label Peaks, Gel Analyzer
  Options… (vertical/horizontal lanes, uncalibrated OD, label with percentages, invert peaks). Peak areas are
  measured with the wand inside line-closed peaks.
* Tools ▸: Save XY Coordinates…, Fractal Box Count…, Analyze Line Graph, Curve Fitting… (CurveFitter functions),
  ROI Manager…, Scale Bar…, Calibration Bar…, Synchronize Windows, Grid…, Compare Images….
* Priorities: **P1** Measure, Particles, Summarize, Clear, Set Measurements, Set Scale, Histogram, Profile, ROI
  Manager, Scale Bar, Calibration Bar. **P2** Calibrate, Surface Plot, Gels, Grid, Sync Windows, Curve Fitting,
  Distribution, Label. **P3** Fractal, Line Graph, Compare Images.
* ROI Manager: Add [t], Update, Delete, Rename…, Measure, Deselect, Properties…, Flatten [F], More » (Open…, Save…,
  Fill, Draw, AND, OR (Combine), XOR, Split, Add Particles, Multi Measure, Multi Plot, Sort, Specify, Remove Positions,
  Labels…, List, Interpolate ROIs, Translate, Save/Load ROIs as .roi/.zip, Options…), Show All, Labels. **P1**
* Results table window: File (Save As csv/tsv/xls, Rename, Duplicate), Edit (Copy, Clear, Select All, Delete rows,
  Sort, Plot, Summarize, Distribution, Set Decimal places, Options), Font, Results (Clear, Summarize, Distribution,
  Set Measurements, Rename, Duplicate, Apply macro). Default column order (`ResultsTable.defaultHeadings`): Area, Mean,
  StdDev, Mode, Min, Max, X, Y, XM, YM, Perim., BX, BY, Width, Height, Major, Minor, Angle, Circ., Feret, IntDen,
  Median, Skew, Kurt, %Area, RawIntDen, Ch, Slice, Frame, FeretX, FeretY, FeretAngle, MinFeret, AR, Round, Solidity,
  MinThr, MaxThr. **P1**

### 1.8 Plugins, Window, Help
* Plugins ▸ Macros ▸: Install…, Run…, Edit…, Startup Macros…, Interactive Interpreter… [j], Record… [u]. Shortcuts
  ▸ (Add/Remove/List Shortcuts, List Commands). Utilities ▸: Control Panel… [U], Find Commands… [l], Commands… [L],
  Search…, Monitor Memory…, Capture Screen [G] / Delayed / Image, ImageJ Properties, Threads, Benchmark, Reset…,
  DICOM Metadata…. New ▸: Macro, Macro Tool, JavaScript, Plugin…, Text Window….
  * **P1**: Record (recorder), Find Commands. **P2**: Run (macro subset), Install, Edit, Startup Macros, Capture
    Image, Benchmark. Everything Java is N/A.
* Window: Show All []], Main Window, Put Behind [tab], Cascade, Tile, then a list of open windows (images, Results,
  Log, ROI Manager, plots). In RS-StatChem also the shared `add_module_menu`. **P1**
* Help: ImageJ Website…, Dev. Resources…, Macro Functions…, Examples ▸, Update ImageJ…, Release Notes…, Refresh Menus,
  About Plugins ▸, About ImageJ…. In RS-StatChem: About/Docs only. **P1-min**

### 1.9 Commonly used Fiji plugins (Fiji menu paths approximate)

| Plugin | Licence | Approach for us | P |
|---|---|---|---|
| Bio-Formats importer (File ▸ Import ▸ Bio-Formats) | GPL-2 | tifffile covers OME-TIFF, LSM, STK, FluoView, NDPI, SVS, ScanImage, ImageJ hyperstacks. Optional readers if installed: `nd2` (BSD-3), `czifile` (BSD-3, but needs imagecodecs). Plus "Series" chooser. | P1 (tifffile) / P3 |
| Auto Threshold (Image ▸ Adjust) incl. "Try all" montage, Huang2, ignore black/white, stack histogram, 16-bit full-range | GPL-3 | IJ1 public-domain AutoThresholder (12 methods) + clean-room for 4; full 16-bit histogram variant ourselves | P1 |
| Auto Local Threshold: Bernsen, Contrast, Mean, Median, MidGrey, Niblack, Otsu, Phansalkar, Sauvola | GPL-3 | Reimplement from papers with circular kernels (reuse RankFilters line radii) | P1 |
| Enhance Local Contrast (CLAHE) | GPL | Reimplement from Zuiderveld (1994). skimage `equalize_adapthist` differs | P2 |
| Colour Deconvolution (H&E, H DAB, FastRed/FastBlue/DAB, …, user vectors) | GPL-3+ | Ruifrok & Johnston (2001) matrix maths. Stain vectors are published numbers | P2 |
| Cell Counter | GPL-3 | Already covered by IJ1 multi-point counters (public domain); add a "Counters" panel with table | P1 |
| Coloc 2 (Pearson, Manders M1/M2 with Costes threshold, Li ICQ, Spearman, Kendall tau-b, Costes randomisation P, 2D histogram) | GPL-3 | Reimplement from Manders 1993, Costes 2004, Li 2004 | P2 |
| MorphoLibJ (morphological filters with shapes, reconstruction, fill holes, kill borders, regional/extended min-max, marker-controlled and classic watershed, label analysis, chamfer distance maps) | LGPL-3 | scipy.ndimage + own reconstruction (Vincent 1993 queue). Optional skimage `morphology.reconstruction` / `segmentation.watershed` (BSD) | P2 |
| Analyze Skeleton (2D/3D) branch/junction table, tagged skeleton | GPL-3 | Own: neighbour-count pixel classes, branch tracing with √2 diagonals; output columns as Fiji | P2 |
| Skeletonize (2D/3D) (Lee 1994) | GPL-3 | 2D = IJ1 table skeleton; 3D = Lee algorithm reimplemented (skimage `skeletonize(method='lee')` BSD as reference) | P2/P3 |
| BoneJ (thickness, anisotropy, connectivity, fractal, …) | BSD-2 (BoneJ2) | Portable with attribution; Local Thickness (Hildebrand–Rüegsegger) from scipy EDT | P3 |
| Trainable Weka Segmentation | GPL-3 | Own: feature stack (Gaussian, DoG, Hessian, Sobel, membrane, variance, min/max/median) via scipy + random forest. sklearn is excluded from the build, so write a small numpy RF or lift the exclusion (31 MB) | P3 |
| StarDist | BSD-3 (needs TensorFlow) | Optional: run the exported ONNX model with onnxruntime (≈21 MB wheel) if installed. Not bundled | P3 |
| TrackMate | GPL-3 | Own: LoG/DoG detector (scipy) + sub-pixel fit + LAP linking (scipy `linear_sum_assignment`, Jaqaman 2008) | P3 |
| Stitching (pairwise/grid) | GPL-2 | Phase correlation (numpy FFT, Preibisch idea re-derived) + linear blending | P3 |
| Kymograph (Multi Kymograph / KymographBuilder) | KymographBuilder MIT | IJ1 Reslice/Straighten along polyline with width averaging (public domain) | P2 |
| Time Series Analyzer | n/a | ROI Manager Multi Measure over frames + plot | P2 |
| Bleach Correction (simple ratio, exponential fit, histogram matching) | GPL-3 | Reimplement (Miura 2014); use existing `core/analysis/fitting.py` | P2 |
| Directionality | GPL-3+ | Reimplement (Fourier / local gradient orientation histogram) | P3 |
| 3D Viewer | GPL-3 (Java3D) | v1: IJ1 3D Project + Orthogonal Views + Surface Plot. Later a QOpenGLWidget ray-caster. PySide6 here has QtOpenGL, QtQuick3D, QtDataVisualization, QtGraphs | P3 |
| Fourier tools / Bandpass | – | Core IJ1 FFT, already covered | P1 |

### 1.10 Behaviours that make it feel like ImageJ (all P1)
* **Image window**: the title shows the name and zoom ("blobs.gif (150%)"). The info line above the image shows
  "256x254 pixels; 8-bit (inverting LUT); 64K", or for stacks "1/27 (label); 2.6x2.6 µm; …".
* **Stack sliders**: hyperstacks get c/z/t sliders with a play button for t/z. Mouse wheel scrolls slices; ctrl+wheel
  zooms.
* **Zoom and scroll**: zoom about the cursor with the IJ zoom levels. Windows grow until they fill the screen, then
  scroll.
* **Status bar**: shows calibrated x, y, and the value(s) under the cursor (RGB triplet; raw and calibrated value).
  While dragging a ROI it shows w/h/angle/length.
* **ROIs**: handles, move/resize, composite ROIs (shift/alt), arrow-key nudging, "Restore Selection", and a ROI
  transferred to another image by keyboard [shift-E].
* **Session state**: multiple images open with an "active image" (last activated window), and single-level undo.
  Display range and LUT are per image (per channel for composites). The ROI overlay is drawn non-destructively.
* **Text windows**: Results, Log, histogram and plot windows are separate top-level windows, each with List/Copy/Live.
* **Shortcuts**: IJ single-key shortcuts without ctrl (e.g. `m`, `t`, `f`, `h`, `k`, `+`, `-`, `<`, `>`, `\`).
  ImageJ 1.54 also has a "Require control key" option.
* **Calibration**: the spatial unit, pixel size and value calibration are respected by every measurement.

---------------------------------------------------------------------------------------------------------------

## 2. Mapping features to implementations

Rule of thumb:
* Wherever an ImageJ number is user-visible (measurements, thresholds, counts, filter output) → **port the IJ1 Java
  code** into `core/imaging` with numpy/scipy, vectorised where possible.
* Where IJ just does standard maths → numpy/scipy.
* Where IJ uses AWT/Java I/O → Pillow/tifffile.

| Area | Implementation | Notes / parity traps |
|---|---|---|
| TIFF read/write (stacks, hyperstacks, calibration, LUTs, ranges, overlays/ROIs, slice labels, Info) | **tifffile** (`imagej=True`, `imagej_metadata`, `imagej_description`) + roifile for the Overlay/ROI byte blobs | Without imagecodecs tifffile decodes deflate/zlib, lzma, packbits, predictors and float24 (its `_imagecodecs.py` fallbacks), **not LZW/JPEG**. Fall back to Pillow page by page (Pillow decodes LZW), or port IJ's public-domain LZW decoder (`ij/io/ImageReader`). Write uncompressed big-endian TIFF like IJ (option for little-endian). |
| PNG/JPEG/GIF/BMP/PGM | **Pillow** (bundled) | GIF/indexed PNG: keep the palette as the LUT (blobs.gif has an **inverting LUT**). 16-bit PNG = `I;16`. JPEG quality from I/O options. |
| DICOM | **pydicom** (MIT, pure Python) or a port of `ij/plugin/DICOM.java` | Show Info must dump tags like IJ ("0028,0030 Pixel Spacing: …"). Rescale slope/intercept → value calibration (IJ calibrates 16-bit signed as `-32768 + …`). Uncompressed only, as IJ. |
| FITS | **Own ~200-line reader/writer** (2880-byte blocks, BITPIX 8/16/32/−32/−64, BZERO/BSCALE, NAXIS3 = stack) | IJ flips FITS vertically on open; keep that. Avoid astropy (size) and fitsio (GPL). |
| AVI | Port IJ1 `AVI_Reader`/`AVI_Writer` (uncompressed, MJPEG/PNG frames via Pillow) | No ffmpeg. |
| Raw, Text Image, Image Sequence, LUT (.lut binary 768/800 bytes and text) | Own (numpy `frombuffer`, sorting "numerically" as IJ's `StringSorter`) | |
| ROI .roi / .zip | **roifile** (BSD-3, 20 KB), or port `RoiDecoder`/`RoiEncoder` (public domain, ~600+400 lines) | The format is documented in RoiDecoder.java. Support rect, oval, polygon, freehand, traced, polyline, freeline, angle, point (+counters), line (+arrow), composite (ShapeRoi), text, rotated rect, ellipse, stroke/fill colour, position c/z/t, name. |
| Type conversion | Port `TypeConverter`/`ImageConverter` semantics | 16/32→8 scales with the **display range** when "Scale when converting" is on. Rounding is `(int)(v+0.5)`. RGB→8-bit uses unweighted (r+g+b)/3 unless the Weighted option is set. 8-bit Color = `MedianCut.java` port. HSB/Lab exactly as `ColorSpaceConverter`. |
| B&C, Enhance Contrast, Window/Level | Port `ContrastAdjuster.autoAdjust` (AUTO_THRESHOLD = 5000, threshold = pixelCount/autoThreshold, increments) and `ContrastEnhancer` (saturated % split over both tails; equalize uses sqrt of histogram counts) | |
| Threshold auto methods | Port `AutoThresholder` + `ImageProcessor.setAutoThreshold` | "Default" = IJ_IsoData variant ignoring the extreme bins. 16/32-bit images use **256 bins between min and max**, then map back. "Dark background" flips which side is selected. The Fiji plugin's 16-bit full-histogram mode is a separate option. |
| Make Binary / Convert to Mask / Create Selection | Port `Thresholder` + `ThresholdToSelection` | Watch "Black background" (IJ default **false**, `Prefs.blackBackground`). Fiji installs with true. Make it a visible setting; goldens set it explicitly. Inverting LUT on masks when black background is false. |
| Smooth/Sharpen/Find Edges/Shadows/Convolve | `scipy.ndimage.correlate` with `mode='nearest'` (IJ duplicates edge pixels) | 8-bit: round + clamp; RGB per channel. Convolver normalises by kernel sum when ≠0. |
| Gaussian Blur | Port `GaussianBlur.makeGaussianKernel` (accuracy 0.002 for 8-bit/RGB, 0.0002 for 16/32; kRadius = ceil(σ√(−2 ln acc))+1; polynomial tail edge correction) + `scipy.ndimage.correlate1d(mode='nearest')` | IJ downsamples for large σ (UPSCALE_K_RADIUS = 2); skipping it changes values by < ~0.5 gray. Port it if goldens demand. |
| Median/Mean/Min/Max/Variance/Top Hat/Remove Outliers/Despeckle/Remove NaNs | Port `RankFilters.makeLineRadii` (r2 = (int)(r²)+1) for the exact circular kernel | Mean/variance: row-segment sums from a cumulative sum (vectorised, O(r)). Min/max: union of `minimum_filter1d` per kernel row (exact, fast). Median: `scipy.ndimage.median_filter(footprint=…)`; for 8/16-bit a histogram-based sliding median if speed requires. 3D variants with an ellipsoid footprint. |
| Unsharp Mask | (I − w·G)/(1 − w) with the IJ Gaussian | float then convert back. |
| Subtract Background | **Port `BackgroundSubtracter`** (rolling ball with shrink factor and ball approximation; sliding paraboloid; 3×3 max + mean pre-smoothing; light background; separate colours; create background) | skimage `rolling_ball` gives different numbers. |
| Find Maxima, Segmented Particles, Watershed, Ultimate Points, Voronoi | **Port `MaximumFinder` + `EDM`** | Watershed = EDM (float) → MaximumFinder segmentation, tolerance 0.5. Distance map: scipy `distance_transform_edt` is exact like IJ, then the IJ output types (BYTE_OVERWRITE rounds and clips at 255). skimage watershed splits differently. |
| Erode/Dilate/Open/Close/Outline/Median/Fill Holes/Skeletonize | Port `Binary.java` + `BinaryProcessor` | iterations + count semantics (a pixel is removed if ≥ count background neighbours), pad edges option. **Skeletonize uses IJ's lookup table**, not skimage. Fill Holes = 4-connected background flood fill from the edges (≈ `binary_fill_holes`; verify). |
| Math, Image Calculator, Paste modes | numpy with IJ clipping/rounding; Java float32 arithmetic → compute in float32 | Divide by zero: 8/16-bit leave as is (IJ ignores 0 divisors except 32-bit); 32-bit uses the "divide-by-zero value" option (default Infinity). Math ▸ Macro: safe AST evaluator over v, x, y, z, w, h, d, a. |
| FFT family | numpy.fft for speed, but reproduce IJ **FHT conventions**: pad to a power of 2 (mean-padded), power spectrum = log-scaled 8-bit like `FHT.getPowerSpectrum`, quadrant swap, inverse with 0/255 mask, FD Math | Port `FFTFilter` exactly (mirror padding to 1.5× then power of 2, Gaussian band-pass in the frequency domain, stripes, autoscale/saturate). |
| Measurements | **Port `ImageStatistics`/`ByteStatistics`/`ShortStatistics`/`FloatStatistics`, `Analyzer`, `EllipseFitter`, `Roi.getFeretValues`, `PolygonRoi.getTracedPerimeter`** | StdDev uses n−1. Skew/kurtosis use the population variance (kurtosis is excess). Mode of 16/32-bit images uses binned histograms. Centroid = mean of pixel centres (x+0.5). Traced perimeter = `sumdx*w + sumdy*h − nCorners*((w+h) − √(w²+h²))`. Circularity is capped at 1. Solidity = area / convex hull area. Feret = rotating calipers on the convex hull. |
| ROI masks | Port `PolygonFiller` and `OvalRoi.getPolygon/getMask` (IJ's own "cached" integer oval) | Using PIL/QPainter rasterisation would change areas by a few pixels. Pixel-corner coordinate convention. |
| Analyze Particles | Port `ParticleAnalyzer`, accelerated: `scipy.ndimage.label` (8-connectivity; raster-order labels = IJ scan order) gives pixel sets equal to IJ's flood-fill (no "include holes"). With include holes, fill holes per component and drop components nested inside. Outline polygon from a ported Wand (crack following on the component's bounding box only). | Exclude on edges = bounding box touches image/selection edge (IJ uses the traced polygon bounds). FOUR_CONNECTED forces include holes. Outputs: outlines (numbered, IJ font), bare outlines, masks, count masks (16-bit), ellipses, overlays, summary table (Slice, Count, Total Area, Average Size, %Area, Mean, plus selected measurements). |
| Histogram, Plot Profile, Z-profile, Surface Plot | numpy + **matplotlib** (already used by the app) inside IJ-styled windows; profile interpolation like `ProfilePlot`/`ImageProcessor.getLine` (bilinear, step 1 px); wide lines via the Straightener port | Histogram window statistics text like IJ (N, Mean, StdDev, Min, Max, Mode, bins, bin width). |
| Calibrate / Curve Fitting | Reuse `rsstatchem/core/analysis/fitting.py` (`fit_curve`) for fitting; IJ function list and formulas from `ij/measure/CurveFitter` | Uncalibrated OD = log10(255/(255−v)) style per Calibrator. |
| Reslice, Ortho Views, Montage, Z Project, 3D Project, Bin, Scale/Size/Rotate | numpy/scipy (`ndimage.map_coordinates/zoom/affine_transform`), but follow IJ conventions | Rotation about `r.x + (r.width−1)/2`. IJ bicubic = cubic convolution a = 0.5. "Average when downsizing" = IJ's area averaging. Z Project SD uses n−1. 3D Project = port of `Projector`. |
| Overlays, Flatten, Scale Bar, Calibration Bar, Labels | Core keeps overlay data; rendering in UI (QPainter); Flatten renders the overlay to an RGB numpy array via `QImage` (UI helper) | |
| Macro recorder / scripting | Own command registry (see §3.5) | |

**Licence-safe ports**: everything from `ij/**` except the 4 AutoThresholder methods named above. Write a header in each
ported module such as: "Ported from ImageJ 1.54 ij/…/X.java (public domain, NIH; W. Rasband, M. Schmid et al.)".

---------------------------------------------------------------------------------------------------------------

## 3. Dependency recommendation

PyPI metadata as of 2026-10-27. Wheel = compressed download; installed sizes are estimates.

| Package | Licence | Wheel (mac arm64 / win64) | Decision |
|---|---|---|---|
| numpy 2.5, scipy 1.18, Pillow 13, matplotlib 3.11, pandas, PySide6 6.12 | BSD/HPND/LGPL | bundled already | use |
| **tifffile** 2026.10.15 | BSD-3 | 0.28 MB pure Python (requires Python ≥3.12, numpy ≥2.1, both OK) | **add (required)** |
| **roifile** 2026.9.22 | BSD-3 | 0.02 MB pure Python | add (or port RoiDecoder/Encoder; either is fine) |
| **pydicom** 3.0.2 | MIT | 2.38 MB pure Python, no deps | add (P1 DICOM). PyInstaller: add `pydicom` data/hidden imports for pixel handlers; selftest must open a DICOM. |
| scikit-image 0.26.0 | BSD-3 | 12.1 / 11.9 MB (+ networkx 2.1 MB, imageio 0.33 MB, lazy-loader; tifffile/Pillow/scipy shared); est. 40–55 MB installed in total | **optional, phase 2** (MorphoLibJ-like reconstruction/watershed, Lee 3D skeleton, colour conversions). Can also be used **in tests only** as an independent reference (the way pingouin is used today). |
| imagecodecs 2026.10.10 | BSD-3 | 13.4 / 19–20 MB | no (Pillow/own LZW fallback instead) |
| opencv-python-headless 5.0 | Apache-2.0 | 48.3 / 43.8 MB | no |
| astropy 8.0 | BSD-3 | 6.7 MB + pyerfa + astropy-iers-data + PyYAML | no (own FITS) |
| fitsio 1.4 | **GPL-2+** | 0.7 MB | no (licence) |
| PyWavelets 1.10 | MIT AND BSD-3 | 4.3 MB | not needed for ImageJ parity |
| mahotas 1.4.19 | MIT | 1.8 MB | redundant |
| napari | BSD-3 | heavy (vispy, dask, …) | reference only |
| pyimagej 1.8 | Apache-2.0 | small, but downloads ImageJ2/Maven at runtime via jgo/JPype | no |
| skan 0.13 | BSD-3 | needs numba/llvmlite (~30+ MB) | no; write the skeleton analysis ourselves |
| nd2 0.12 / czifile | BSD-3 | czifile needs imagecodecs | optional readers loaded only if installed (P3) |
| onnxruntime 1.31 | MIT | ~21 MB | optional StarDist backend only (P3) |
| ij.jar 1.54p (Maven Central `net.imagej:ij`) | public domain | 2.6 MB | **dev/CI only**, for golden values. Bundling as an "exact engine" is possible (the app already ships a jlinked JRE for OPSIN/CDK), but ImageJ needs `java.desktop` added to the jlink module list (currently `java.base,java.xml`). Not recommended for v1. |

Net bundle growth for v1: about 3–4 MB (tifffile + roifile + pydicom). Adding scikit-image later costs about 45 MB.
**sklearn stays excluded** (packaging excludes it). A Weka-like classifier needs its own small random forest or a
deliberate decision to lift the exclusion.

---------------------------------------------------------------------------------------------------------------

## 4. Module architecture

### 4.1 Integration with the existing app (already registered)
* `ModuleInfo("image", "Image Analysis", …, "ImageJ", "rsstatchem.ui.imagej:open_workspace", (.tif .tiff .png .jpg
  .jpeg .bmp .gif .dcm .fits .roi), "image")`. Suggested additions: `.rsij` (own project, below), and optionally
  `.pgm .avi .lut .fit .fts`. Do **not** claim `.zip/.csv/.txt` (they belong to other modules or are ambiguous); open
  them through the module's own File ▸ Open.
* `open_workspace(ctx, path)` behaviour:
  * Return the **ImageJ bar window** (one per app, like ImageJ's single instance). If it already exists, raise it and
    open `path` there.
  * The bar is the window registered with `ctx`. It keeps `self.ctx = ctx`, has `add_module_menu` under Window, and
    calls `ctx.window_closed(self)` in `closeEvent` once accepted.
  * Closing the bar first closes all image and auxiliary windows, asking "Save changes to …?" for modified images.
  * Image, Results, Log and ROI Manager windows are **not** registered with ctx. They live in a module-level registry
    `rsstatchem.ui.imagej.windows.WINDOWS`. Add that registry to `ui/qt_lifecycle._WINDOW_REGISTRIES`
    (integrator-owned) and clear icon caches there too.
  * Never run a modal loop inside `open_workspace`. Report file errors deferred, as chemdraw does.
* Cross-module bonus (P2): Results ▸ "Send to Statistics" uses `ctx.new_editor(Dataset(df))`, and "Send to Graph &
  Analysis" opens an Origin workbook. Plot windows → existing graph export.

### 4.2 Core package `rsstatchem/core/imaging/` (no Qt; numpy/scipy/Pillow/tifffile only)
```
core/imaging/
  __init__.py
  image.py        ImageDoc: pixels ndarray in TZCYX order (RGB: uint8 TZCYX3 or "S" axis), kind 8/16/32/RGB/
                  8-bit-color, title, path, dims (c, z, t), current position, Calibration, per-channel
                  DisplayRange+LUT, composite mode, slice labels, info/properties dict, overlay, roi, changed flag,
                  snapshot()/undo() single level, plane(c,z,t) accessors, ImageStack helpers
  calibration.py  Calibration (pixel w/h/d, unit(s), frame interval, origin, invertY, value function + coeffs, unit
                  of value), scaled()/raw() helpers
  lut.py          built-in LUTs (LutLoader formulas), invert, apply, .lut read/write, 8-bit rendering helper
                  (raw → 8-bit index via display range, vectorised)
  options.py      global ImageJ preferences dataclass (black background, conversions, measurements, binary
                  options, colours, line width, decimals, …) persisted by the UI via QSettings
  roi/            shapes.py (Rect/RoundRect/RotatedRect/Oval/Ellipse/Polygon(types)/Line/Angle/Point(+counters)/
                  Text/Composite), mask.py (PolygonFiller + OvalRoi ports), geometry.py (convex hull, Feret, spline
                  fit, interpolate, enlarge/band via EDM, fit circle/ellipse/rect, scale/rotate/translate),
                  wand.py (Wand port), roi_io.py (.roi/.zip), selection_ops.py (Edit ▸ Selection operations)
  io/             tiff.py, pillow_io.py, dicom.py, fits.py, raw.py, avi.py, text_image.py, sequence.py, lut_io.py,
                  results_io.py, opener.py (dispatch by extension/magic bytes), writer.py
  convert.py      Image ▸ Type conversions (+ MedianCut)
  adjust.py       auto B&C, apply LUT, size/canvas, color balance
  stacks.py       add/delete slice, images↔stack, montage, reslice, z project, grouped z project, substack,
                  reduce, reverse, concatenate, combine, insert, hyperstack convert/reduce, label, statistics
  project3d.py    Projector port; ortho slices helper
  transform.py    flip/rotate/translate/scale/bin/image↔results
  color.py        split/merge/arrange channels, stack↔RGB, composite, HSB/Lab, colour deconvolution
  filters.py      3×3 family, Convolver, Gaussian (2D/3D), RankFilters (2D/3D), Unsharp, Top Hat, noise, outliers
  threshold.py    AutoThresholder (+clean-room 4), setAutoThreshold semantics, Make Binary/Convert to Mask,
                  colour threshold, local thresholds (Fiji)
  binary.py       Binary/BinaryProcessor ports, EDM, watershed/UEP/voronoi
  maxima.py       MaximumFinder port
  background.py   BackgroundSubtracter port
  fft.py          FFT/inverse/power spectrum/bandpass/custom/FD math/swap
  math_ops.py     Process ▸ Math, Image Calculator, paste modes, expression evaluator
  contrast.py     Enhance Contrast
  measure.py      ImageStatistics + Analyzer (area/line/point/angle), measurement flags, redirect, limit
  particles.py    ParticleAnalyzer
  results.py      ResultsTable (ordered columns, row labels, decimals, NaN-empty, summarize, distribution, CSV)
  histogram.py, profile.py, surface.py, gels.py, calibrate.py, scalebar.py (geometry only), fractal.py
  commands.py     Command registry + macro option-string parse/format + Recorder (no Qt)
  macro.py        IJ-macro subset interpreter (P2) + Python scripting facade
  project.py      .rsij save/load
  fiji/           coloc.py, local_threshold.py, clahe.py, skeleton_analysis.py, morpho.py, bleach.py,
                  kymograph.py, tracking.py (P3), stitching.py (P3), weka.py (P3)
```
Every core operation is a pure function or method:
* Input: `ImageDoc`/ndarray + options. Output: a new or modified `ImageDoc`, ResultsTable, ROI list, etc.
* An optional `progress(fraction)` and `cancelled()` callback lets the UI show IJ's progress bar and support Esc.

### 4.3 UI package `rsstatchem/ui/imagej/`
```
ui/imagej/
  __init__.py        open_workspace (singleton bar)
  bar.py             ImageJBar(QMainWindow): menubar, ToolBar widget, status label, progress bar, drag&drop
  toolbar.py         tools with IJ-like icons (QPainter), right-click variants, double-click options, tool state
  menus.py           static MENU_LAYOUT (labels/order/shortcuts exactly as IJ_Props + Menus.java) → QMenus from
                     the command registry; missing commands appear disabled (lets WPs land independently)
  windows.py         WindowManager: active image, image list, WINDOWS registry, Window menu list, tile/cascade
  image_window.py    ImageWindow: title, info line, Canvas, c/z/t sliders + play, per-platform menubar/shortcuts
  canvas.py          ImageCanvas(QWidget): zoom levels, scrolling, 8-bit/RGB QImage rendering (cached), overlay +
                     ROI painting, cursor readout signal
  roi_tools.py       mouse interaction per tool (create/move/resize/handles, composite add/subtract, wand, point
                     counters, magnifier, hand, picker, text)
  dialogs/generic.py GenericDialog equivalent: addNumericField/Slider/Checkbox/Choice/String/Message/Preview;
                     keys derived like IJ macros; records; runs in batch from an option string
  dialogs/*.py       per-area dialogs not expressible as GenericDialog
  frames/            results_window.py, log_window.py, text_window.py, roi_manager.py, histogram_window.py,
                     plot_window.py (matplotlib), bc_adjuster.py, threshold_adjuster.py, color_threshold.py,
                     channels_tool.py, color_picker.py, recorder.py, command_finder.py, ortho_views.py,
                     point_counter.py
  commands_file.py, commands_edit.py, commands_image.py, commands_process.py, commands_analyze.py,
  commands_plugins.py   each exposes register(registry) binding command labels to core functions + dialogs
  icons.py
```
Platform behaviour:
* **macOS**: give every ImageWindow its own QMenuBar built from the *same QActions*, so the global menu shows the IJ
  menus whenever an image is active. Do not use a parentless global QMenuBar, because the launcher would show it too.
* **Windows/Linux**: only the bar has a menubar. Image windows `addActions(all_shortcut_actions)` so the single-key
  shortcuts work from them.

### 4.4 Own file formats
* **`.rsij` project** (zip):
  * `manifest.json`: version, windows, active image, settings snapshot.
  * `images/NN.tif`: ImageJ-compatible hyperstack TIFF with calibration, LUTs, ranges and overlay, so each image can
    be opened in real ImageJ.
  * `rois/RoiSet.zip`: ROI Manager in the ImageJ format.
  * `results/*.csv`, and `log.txt`.
* **Interchange with ImageJ itself**: .roi and RoiSet .zip (public-domain format, via roifile or the port), TIFF with
  IJ metadata, .lut, Results .csv (IJ ≥1.52 default extension), and `.ijm` macros (recorded output is IJ-compatible
  for the `run(...)` subset).

### 4.5 Commands, recorder and macros
* **Command registry** (`core/imaging/commands.py`). Each command has:
  * `label` (exact IJ menu text, e.g. `"Gaussian Blur..."`), `menu_path`, `func(session, image, **opts)`;
  * an `options` schema: list of fields with IJ label, type, default and range;
  * flags: `needs_image`, `supports_stack` ("Process all N slices?" prompt), `undoable`, `roi_aware`.
* **Macro option keys**: the key is the first word of the dialog label, lower-cased, with trailing ':' removed (IJ
  `Macro.trimKey`). Booleans count as true when the key is present. Values with spaces go in `[ ]`.
  * This lets real ImageJ strings run unchanged, e.g. `run("Analyze Particles...", "size=0-Infinity display clear
    summarize")` or `run("Gaussian Blur...", "sigma=2")`.
* **Recorder window**: language choice "Macro" (IJ syntax) or "Python" (`ij.run("Gaussian Blur...", sigma=2)`), with
  Create/Run buttons.
* **Macro runner (P2)**: a subset interpreter.
  * Calls: `run`, `open`, `close`, `selectWindow`/`selectImage`, `rename`, `getTitle`, `nImages`, `nResults`,
    `getResult`, `setResult`, `setAutoThreshold`, `setThreshold`, `setOption`, `make*` ROI functions, `roiManager()`,
    `saveAs`, `print`, `setSlice`, `Stack.*`.
  * Language: variables, arithmetic, strings, arrays, if/else, for/while, functions.
  * Python scripts (`Plugins ▸ New ▸ Script`) run with an `ij` facade object. That is the "Fiji script editor"
    equivalent.
* **Find Commands [l]**: a search box over the registry. Very useful, cheap to build.

---------------------------------------------------------------------------------------------------------------

## 5. Verification strategy

1. **Golden values from real ImageJ (WP-0)**.
   * `scripts/imagej_golden.py`: fetch `ij-1.54p.jar` from Maven Central with a pinned SHA-256 (same pattern as
     `scripts/fetch_cdk.py`). Then, for each case in `tests/data_imagej/cases/*.ijm`, run
     `java -Djava.awt.headless=true -jar ij.jar -batch case.ijm <in> <out>` with the dev JDK.
   * Each case writes Results CSVs and output TIFFs. Commit the outputs to `tests/data_imagej/golden/`. Keep them
     small: 256×256 inputs, plus checksum or stats for larger outputs.
   * pytest compares against the goldens and needs no Java. Re-run the script only when goldens change. It is not
     part of CI, which saves macOS minutes.
   * Headless batch mode supports filters, thresholds, Measure, Analyze Particles, EDM and FFT. GUI-only commands
     (ortho views, B&C window) are tested by behaviour instead.
2. **Inputs**.
   * (a) Synthetic images generated by `tests/data_imagej/make_synthetic.py` with a fixed seed: discs/ellipses of
     known area, touching blobs for watershed, ramps, noise, 16/32-bit ranges, NaNs, RGB, 3D stacks, hyperstacks.
   * (b) Files from the public-domain ImageJ repo `tests/data/`: blobs.gif, Cell_Colony.jpg, embryos.bmp,
     head8bit.tif, head.tif, gray8/16/24/32 TIFF fixtures, gray16signed TIFFs, 01.dcm, bat-cochlea-renderings.fits,
     Tree_Rings.pgm, clown.raw.
     * **Exclude `lena-std.png`** (third-party copyright).
     * imagej.net/ij/images/ carries **no licence statement**: use those samples for manual checks or download them
       on demand, but don't commit them (Open Samples downloads them at runtime, as ImageJ does).
3. **Golden cases to cover first**:
   * all 17 auto-threshold methods on blobs, Cell_Colony, a 16-bit and a 32-bit image (with/without dark background);
   * Measure with every flag on rect/oval/polygon/traced ROIs, calibrated and uncalibrated;
   * Analyze Particles on blobs (Default threshold; with/without exclude edges and include holes; summary values; the
     particle count expected to be 64 is to be **confirmed**) and on Cell_Colony;
   * Gaussian σ = 0.5/2/10 on 8/16/32-bit; median/mean/min/max/variance r = 0.5/1/2/5;
   * Subtract Background r = 50 (ball and paraboloid, light background);
   * Find Maxima prominence 10/50, all outputs; Watershed / UEP / Voronoi / Skeletonize / Outline / Fill Holes /
     erode/dilate with count/iterations;
   * Bandpass defaults (40/3); FFT power spectrum;
   * Enhance Contrast 0.35% with normalize and equalize; B&C Auto min/max;
   * type conversions with and without scaling; Z Project types; Scale/Rotate bilinear and bicubic;
   * ROI round-trip .roi/.zip (read files written by ij.jar and vice versa).
4. **Tolerances**:
   * exact for integer outputs, counts, areas, thresholds and traced perimeters;
   * ±1 gray level for 8-bit Gaussian, background and bicubic;
   * rel 1e-5 for float statistics (Windows float order: see the release memory note, use `rel=1e-5`).
5. **Independent cross-checks in tests only**: scipy EDT vs our EDM, skimage Otsu/Li/Yen/triangle (expect small
   bin-convention differences; document them), numpy statistics.
6. **UI tests**: offscreen Qt with the session `qapp` fixture (never create a QApplication).
   * Open the module from the launcher; open/close image windows; tool interactions via synthesized mouse events;
     menus complete vs `MENU_LAYOUT`; recorder output strings; the bar's closeEvent quits only when it is the last
     window.
   * Screenshots via `widget.grab()` for manual review.
7. **Packaged `--selftest`**: import `core.imaging`, open a tiny TIFF/PNG/DICOM/FITS from resources, run threshold +
   particles on a 32×32 synthetic image.

---------------------------------------------------------------------------------------------------------------

## 6. Implementation work packages (disjoint file ownership)

**Phase 0 — integrator contract (short, before the parallel work)**:
* `core/imaging/image.py`, `calibration.py`, `options.py`, `roi/shapes.py` (data classes only), `results.py`
  (ResultsTable minimal API), `commands.py` (registry, option strings, recorder).
* `ui/imagej/dialogs/generic.py` (API + basic implementation), the `ui/imagej/menus.py` MENU_LAYOUT table.
* `tests/data_imagej/` layout + `scripts/imagej_golden.py` (WP-0).
* Plus the shared edits: `requirements.txt` (tifffile, roifile, pydicom), packaging spec hidden imports,
  `qt_lifecycle` registry, `selftest.py`, extensions in `modules.py`.
* After Phase 0 these files are read-only for the WPs. Requested changes go in their reports.

Each WP registers its commands in its own `commands_<area>.py` with `register(registry)`. `menus.py` already lists
every label, so menus fill in as WPs land.

### WP-A — Shell, display, tools, File/Edit/Window/Plugins, I/O
* Owns: `ui/imagej/{__init__,bar,toolbar,windows,image_window,canvas,roi_tools,icons,commands_file,commands_edit,
  commands_plugins}.py`, `ui/imagej/frames/{log_window,text_window,recorder,command_finder,color_picker}.py`,
  `core/imaging/io/*`, `core/imaging/lut.py`, `core/imaging/project.py`, `core/imaging/macro.py` (P2),
  `tests/test_imagej_shell.py`, `tests/test_imagej_io.py`.
* Delivers:
  * the ImageJ bar, all toolbar tools and interactions, zoom/scroll/status readout, stack sliders and animation;
  * multi-window management, undo plumbing, clipboard/paste control, Fill/Clear/Draw/Invert;
  * Edit ▸ Options dialogs, Window and Help menus, Open/Save/Import/Export of all formats;
  * `.rsij` projects, Open Samples (download cache), Recorder, Find Commands, macro subset (P2).
* Ports: LutLoader, Wand (shared with WP-C through `core/imaging/roi/wand.py`, owned by WP-C; WP-A only calls it),
  AVI reader/writer, FITS, LZW fallback.

### WP-B — Process menu + segmentation
* Owns: `core/imaging/{filters,threshold,binary,maxima,background,fft,math_ops,contrast}.py`,
  `core/imaging/fiji/{local_threshold,clahe,morpho}.py`,
  `ui/imagej/{commands_process}.py`,
  `ui/imagej/frames/{threshold_adjuster,color_threshold}.py`, `ui/imagej/dialogs/process_*.py`,
  `tests/test_imagej_process.py`.
* Delivers:
  * everything in §1.6, plus Image ▸ Adjust ▸ Threshold/Color Threshold (segmentation lives with the threshold maths);
  * Fiji Auto Threshold ("Try all" montage), Auto Local Threshold, CLAHE (P2), MorphoLibJ-lite (P2);
  * batch processing commands (Process ▸ Batch).
* Ports: AutoThresholder (12 methods) + clean-room 4, Thresholder, GaussianBlur, RankFilters, Convolver, Binary,
  BinaryProcessor, EDM, MaximumFinder, BackgroundSubtracter, FFT/FHT/FFTFilter/FFTMath, ContrastEnhancer, ImageMath.

### WP-C — Analyze menu, measurements, ROIs and ROI Manager
* Owns: `core/imaging/{measure,particles,histogram,profile,surface,gels,calibrate,scalebar,fractal}.py`,
  `core/imaging/roi/{mask,geometry,wand,roi_io,selection_ops}.py`,
  `core/imaging/fiji/{coloc,skeleton_analysis}.py`,
  `ui/imagej/commands_analyze.py`,
  `ui/imagej/frames/{results_window,roi_manager,histogram_window,plot_window,point_counter}.py`,
  `ui/imagej/dialogs/analyze_*.py`, `tests/test_imagej_analyze.py`, `tests/test_imagej_roi.py`.
* Delivers:
  * everything in §1.7, plus Edit ▸ Selection operations (geometry);
  * ROI .roi/.zip, Results window and export (CSV/xlsx, send to Statistics/Graph);
  * histogram, profile and Z-profile plots, ROI Manager with Multi Measure;
  * scale bar, calibration bar, grid; Cell Counter panel; Coloc 2 (P2); Analyze Skeleton (P2); Time Series Analyzer
    (P2).
* Ports: ImageStatistics family, Analyzer, ParticleAnalyzer, PolygonFiller, OvalRoi mask, EllipseFitter, Feret,
  traced perimeter, RoiDecoder/Encoder (or roifile), RoiEnlarger, ProfilePlot, SurfacePlotter, GelAnalyzer,
  Calibrator (fitting via `core/analysis/fitting.py`), ScaleBar, CalibrationBar.

### WP-D — Image menu: types, display, colour, stacks, transforms, overlays
* Owns: `core/imaging/{convert,adjust,stacks,project3d,transform,color}.py`,
  `core/imaging/fiji/{bleach,kymograph}.py`,
  `ui/imagej/commands_image.py`,
  `ui/imagej/frames/{bc_adjuster,channels_tool,ortho_views}.py`, `ui/imagej/dialogs/image_*.py`,
  `tests/test_imagej_image.py`.
* Delivers:
  * everything in §1.5 except Threshold/Color Threshold (WP-B): Type, Adjust (B&C, W/L, Color Balance, Size, Canvas,
    Line Width, Coordinates), Show Info, Properties, Color, Stacks (incl. Tools, Animation options), Hyperstacks;
  * Crop/Duplicate/Rename/Scale, Transform, Zoom commands (calling the WP-A canvas API), Overlay commands, Lookup
    Tables;
  * 3D Project, Orthogonal Views, Montage, Reslice, Kymograph (P2), Bleach Correction (P2), Colour Deconvolution (P2;
    put it in `color.py`).
* Ports: TypeConverter/MedianCut, ContrastAdjuster auto, Resizer/CanvasResizer, Slicer, ZProjector, MontageMaker,
  Projector, Rotator/Scaler/Binner, RGBStackMerge/ChannelSplitter, HyperStackConverter/Reducer, SubstackMaker.

### Later round — WP-E (P3, optional)
Trainable segmentation (own RF), StarDist via optional onnxruntime, TrackMate-lite, Stitching, BoneJ (BSD-2 port),
3D volume viewer (QOpenGLWidget ray-caster), CZI/ND2 readers.

### Order and risk
* The cross-WP touch points are the canvas API (WP-A ↔ D for zoom/overlay), ROI masks (C ↔ all measure-aware
  commands), and the threshold state on `ImageDoc` (B ↔ C for "Limit to threshold" and particle analysis).
  Phase 0 must fix these interfaces:
  * `ImageDoc.threshold = (lower, upper, mode) | None`;
  * `ImageDoc.roi`, `ImageDoc.overlay`;
  * `roi.get_mask(shape) -> bool ndarray` (stub; WP-C replaces the implementation behind the same function);
  * `canvas.set_magnification`, `canvas.repaint_overlay`.
* Largest effort: WP-A (UI-heavy) and WP-C (many measurement details). Consider giving WP-A's I/O (`core/imaging/io/*`)
  to WP-D if WP-A is overloaded; the ownership boundary stays clean.

---------------------------------------------------------------------------------------------------------------

## 7. Open checks for implementers
* Confirm the blobs.gif particle count and summary with ij.jar (the "64 particles" figure is unverified).
* Confirm Fill Holes connectivity and the legacy Wand 8-connectivity on diagonal-touching pixels against ij.jar.
* Decide the default of "Black background": ImageJ false (`Prefs.blackBackground`), Fiji true.
* Check whether tifffile accepts a registered LZW decoder, or use the Pillow fallback.
* Confirm the exact Fiji menu paths before labelling Fiji commands (Fiji is not installed locally).
