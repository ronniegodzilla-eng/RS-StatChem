"""The ImageJ 1.54 menu tree and who implements each command (Phase 0 contract).

Generated from ImageJ 1.54p's ``IJ_Props.txt`` (inside ij-1.54p.jar) and the skeleton built in
``ij/Menus.java``, so labels, order, separators and shortcuts are ImageJ's. Each leaf is a
:class:`MenuItem` whose ``label`` is the exact ImageJ command name and registry key: a
letter/digit shortcut in brackets is stripped ("Image...[n]" -> "Image..." with shortcut "n";
an upper-case letter means shift), while other bracketed keys stay part of the name ("In [+]",
"Next Slice [>]") and trailing spaces are significant ("Properties... " is Edit > Selection,
"Properties..." is Image). ``ij_class`` names the ImageJ class to port.

Not included (Java-specific, per docs/IMAGEJ_RESEARCH.md): Compiler..., Memory & Threads...,
Proxy Settings..., Compile and Run..., Plugins > Install..., the Java plugin templates, ImageJ
Properties, Threads, Monitor Events/Memory, Startup Macros, Interactive Interpreter, Help >
Examples / Update ImageJ / Release Notes / Refresh Menus / About Plugins. RS-StatChem's Help
menu ends with Check for Updates and About. "Open Recent" is filled at run time, and the Window
menu lists open windows after its fixed items.

Ownership: :data:`OWNER_RULES` maps menu paths to an area key in :data:`AREAS`; each area's
``rsstatchem.ui.imagej.commands_<area>`` module registers its commands with
``register(registry)``. Menu entries without a registered command are shown disabled.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator


@dataclass(frozen=True)
class MenuItem:
    label: str
    shortcut: str = ""        # ImageJ key: "n" = Ctrl+N, "N" = Ctrl+Shift+N, "" = none
    ij_class: str = ""        # implementing class in ImageJ (ij.plugin. prefix dropped)


@dataclass(frozen=True)
class Menu:
    title: str
    items: tuple = field(default_factory=tuple)   # MenuItem | Menu | None (separator)


@dataclass(frozen=True)
class Area:
    key: str
    work_package: str
    module: str               # Python module with register(registry)
    description: str


SEP = None
I = MenuItem  # noqa: E741
M = Menu

AREAS: dict[str, Area] = {a.key: a for a in (
    Area("file", "WP-A", "rsstatchem.ui.imagej.commands_file",
         "File menu: new/open/import/save, samples, batch I/O"),
    Area("edit", "WP-A", "rsstatchem.ui.imagej.commands_edit",
         "Edit menu (undo, clipboard, fill/clear/draw/invert) and Edit > Options"),
    Area("plugins", "WP-A", "rsstatchem.ui.imagej.commands_plugins",
         "Plugins menu (macros, recorder, shortcuts, utilities) and Repeat Command"),
    Area("shell", "Phase 0 / WP-A", "rsstatchem.ui.imagej.commands_shell",
         "Window and Help menus (built with the ImageJ bar)"),
    Area("image", "WP-D", "rsstatchem.ui.imagej.commands_image",
         "Image menu except thresholds: types, adjust, color, stacks, transform, zoom, "
         "overlay, lookup tables"),
    Area("process", "WP-B", "rsstatchem.ui.imagej.commands_process",
         "Process menu and Image > Adjust > Threshold / Color Threshold"),
    Area("analyze", "WP-C", "rsstatchem.ui.imagej.commands_analyze",
         "Analyze menu and Edit > Selection"),
)}

# (menu path prefix, area); the longest matching prefix wins. Paths join titles with ">".
OWNER_RULES: tuple[tuple[str, str], ...] = (
    ("File", "file"),
    ("Edit", "edit"),
    ("Edit>Selection", "analyze"),
    ("Image", "image"),
    ("Image>Adjust>Threshold...", "process"),
    ("Image>Adjust>Color Threshold...", "process"),
    ("Process", "process"),
    ("Process>Repeat Command", "plugins"),
    ("Analyze", "analyze"),
    ("Plugins", "plugins"),
    ("Window", "shell"),
    ("Help", "shell"),
)

# Keys that ImageJ handles without a modifier when "Require control key" is off: every
# letter/digit menu shortcut above plus these plain keys (ij.IJ / ImageJ.keyPressed).
PLAIN_KEYS = {"+": "In [+]", "=": "In [+]", "-": "Out [-]", "<": "Previous Slice [<]",
              ",": "Previous Slice [<]", ">": "Next Slice [>]", ".": "Next Slice [>]",
              "\\": "Start Animation [\\]", "/": "Reslice [/]...", "]": "Show All"}

MENU_LAYOUT: tuple[Menu, ...] = (
    M('File', (
        M('New', (
            I('Image...', 'n', 'Commands("new")'),
            I('Hyperstack...', '', 'HyperStackMaker'),
            I('Text Window', 'N', 'NewPlugin("text")'),
            I('Internal Clipboard', '', 'Clipboard("show")'),
            I('System Clipboard', 'V', 'Clipboard("showsys")'),
        )),
        I('Open...', 'o', 'Commands("open")'),
        I('Open Next', 'O', 'NextImageOpener'),
        M('Open Samples', (
            I('AuPbSn 40', '', 'URLOpener("AuPbSn40.jpg")'),
            I('Bat Cochlea Volume', '', 'URLOpener("bat-cochlea-volume.zip")'),
            I('Bat Cochlea Renderings', '', 'URLOpener("bat-cochlea-renderings.zip")'),
            I('Blobs', 'B', 'URLOpener("blobs.gif")'),
            I('Boats', '', 'URLOpener("boats.gif")'),
            I('Cardio (RGB DICOM)', '', 'URLOpener("cardio.dcm.zip")'),
            I('Cell Colony', '', 'URLOpener("Cell_Colony.jpg")'),
            I('Clown', '', 'URLOpener("clown.jpg")'),
            I('Confocal Series', '', 'URLOpener("confocal-series.zip")'),
            I('CT (16-bit DICOM)', '', 'URLOpener("ct.dcm.zip")'),
            I('Dot Blot', '', 'URLOpener("Dot_Blot.jpg")'),
            I('Embryos', '', 'URLOpener("embryos.jpg")'),
            I('Fluorescent Cells', '', 'URLOpener("FluorescentCells.zip")'),
            I('Fly Brain', '', 'URLOpener("flybrain.zip")'),
            I('Gel', '', 'URLOpener("gel.gif")'),
            I('HeLa Cells (48-bit RGB)', '', 'URLOpener("hela-cells.zip")'),
            I('Image with Overlay', '', 'URLOpener("ImageWithOverlay.zip")'),
            I('Leaf', '', 'URLOpener("leaf.jpg")'),
            I('Line Graph', '', 'URLOpener("LineGraph.jpg")'),
            I('Mitosis (5D stack)', '', 'URLOpener("Spindly-GFP.zip")'),
            I('MRI Stack', '', 'URLOpener("mri-stack.zip")'),
            I('M51 Galaxy (16-bits)', '', 'URLOpener("m51.zip")'),
            I('Neuron (5 channels)', '', 'URLOpener("Rat_Hippocampal_Neuron.zip")'),
            I('Nile Bend', '', 'URLOpener("NileBend.jpg")'),
            I('Organ of Corti (4D stack)', '', 'URLOpener("organ-of-corti.zip")'),
            I('Particles', '', 'URLOpener("particles.gif")'),
            I('T1 Head (16-bits)', '', 'URLOpener("t1-head.zip")'),
            I('T1 Head Renderings', '', 'URLOpener("t1-rendering.zip")'),
            I('TEM Filter', '', 'URLOpener("TEM_filter_sample.jpg")'),
            I('Tree Rings', '', 'URLOpener("Tree_Rings.jpg")'),
            SEP,
            I('Cache Sample Images ', '', 'URLOpener("cache")'),
        )),
        M('Open Recent', (
        )),
        M('Import', (
            I('Image Sequence...', '', 'FolderOpener'),
            I('Raw...', '', 'Raw'),
            I('LUT... ', '', 'LutLoader'),
            I('Text Image... ', '', 'TextReader'),
            I('Text File... ', '', 'TextFileReader'),
            I('Results... ', '', 'SimpleCommands("import")'),
            I('Table... ', '', 'SimpleCommands("table")'),
            I('URL...', '', 'URLOpener'),
            I('Stack From List...', '', 'ListVirtualStack'),
            I('TIFF Virtual Stack...', '', 'FileInfoVirtualStack'),
            I('AVI...', '', 'AVI_Reader'),
            I('XY Coordinates... ', '', 'XY_Reader'),
        )),
        M('Show Folder', (
            I('Image'),
            I('Plugins'),
            I('Macros'),
            I('LUTs'),
            I('ImageJ'),
            I('temp'),
            I('Home'),
        )),
        SEP,
        I('Close', 'w', 'Commands("close")'),
        I('Close All', 'W', 'Commands("close-all")'),
        I('Save', 's', 'Commands("save")'),
        M('Save As', (
            I('Tiff...', '', 'filter.Writer("tiff")'),
            I('Gif...', '', 'filter.Writer("gif")'),
            I('Jpeg...', '', 'filter.Writer("jpeg")'),
            I('Text Image...', '', 'filter.Writer("text")'),
            I('ZIP...', '', 'filter.Writer("zip")'),
            I('Raw Data...', '', 'filter.Writer("raw")'),
            I('Image Sequence... ', '', 'StackWriter'),
            I('AVI... ', '', 'filter.AVI_Writer'),
            I('BMP...', '', 'filter.Writer("bmp")'),
            I('PNG...', '', 'filter.Writer("png")'),
            I('PGM...', '', 'filter.Writer("pgm")'),
            I('FITS...', '', 'filter.Writer("fits")'),
            I('LUT...', '', 'filter.Writer("lut")'),
            I('Selection...', '', 'filter.RoiWriter'),
            I('XY Coordinates...', '', 'filter.XYWriter'),
            I('Results...', '', 'MeasurementsWriter'),
            I('Text...', '', 'TextWriter'),
        )),
        I('Revert', 'R', 'Commands("revert")'),
        SEP,
        I('Page Setup...', '', 'filter.Printer("setup")'),
        I('Print...', 'p', 'filter.Printer("print")'),
        SEP,
        I('Quit', '', 'Commands("quit")'),
    )),
    M('Edit', (
        I('Undo', 'z', 'Commands("undo")'),
        SEP,
        I('Cut', 'x', 'Clipboard("cut")'),
        I('Copy', 'c', 'Clipboard("copy")'),
        I('Copy to System', '', 'Clipboard("scopy")'),
        I('Paste', 'v', 'Clipboard("paste")'),
        I('Paste Control...', '', 'frame.PasteController'),
        SEP,
        I('Clear', '', 'filter.Filler("clear")'),
        I('Clear Outside', '', 'filter.Filler("outside")'),
        I('Fill', 'f', 'filter.Filler("fill")'),
        I('Draw', 'd', 'filter.Filler("draw")'),
        I('Invert', 'I', 'filter.Filters("invert")'),
        SEP,
        M('Selection', (
            I('Select All', 'a', 'Selection("all")'),
            I('Select None', 'A', 'Selection("none")'),
            I('Restore Selection', 'E', 'Selection("restore")'),
            I('Fit Spline', '', 'Selection("spline")'),
            I('Fit Circle', '', 'Selection("circle")'),
            I('Fit Ellipse', '', 'Selection("ellipse")'),
            I('Fit Rectangle', '', 'Selection("rect")'),
            I('Interpolate', '', 'Selection("interpolate")'),
            I('Convex Hull', '', 'Selection("hull")'),
            I('Make Inverse', '', 'Selection("inverse")'),
            I('Create Selection', '', 'Selection("from")'),
            I('Create Mask', '', 'Selection("mask")'),
            SEP,
            I('Properties... ', 'y', 'Selection("properties")'),
            I('Scale... ', '', 'RoiScaler'),
            I('Rotate...', '', 'Selection("rotate")'),
            I('Translate... ', '', 'Selection("translate")'),
            I('Enlarge...', '', 'RoiEnlarger'),
            I('Make Band...', '', 'Selection("band")'),
            I('Specify...', '', 'SpecifyROI'),
            I('Straighten...', '', 'Straightener'),
            I('To Bounding Box', '', 'Selection("tobox")'),
            I('Line to Area', '', 'Selection("toarea")'),
            I('Area to Line', '', 'Selection("toline")'),
            I('Image to Selection...', '', 'OverlayCommands("image-roi")'),
            I('Add to Manager', 't', 'Selection("add")'),
        )),
        M('Options', (
            I('Appearance...', '', 'AppearanceOptions'),
            I('Arrow Tool...', '', 'ArrowToolOptions'),
            I('Colors...', '', 'Colors'),
            I('Conversions...', '', 'Options("conv")'),
            I('DICOM...', '', 'Options("dicom")'),
            I('Fonts...', '', 'SimpleCommands("fonts")'),
            I('Input/Output...', '', 'Options("io")'),
            I('Line Width...', '', 'Options("line")'),
            I('Misc...', '', 'Options("misc")'),
            I('Plots...', '', 'Profiler("set")'),
            I('Point Tool...', '', 'PointToolOptions'),
            I('Roi Defaults...', '', 'ij.gui.RoiDefaultsDialog'),
            I('Rounded Rect Tool...', '', 'RectToolOptions'),
            I('Startup...', '', 'Startup'),
            I('Wand Tool...', '', 'WandToolOptions'),
            SEP,
            I('Fresh Start', '', 'Options("fresh-start")'),
            I('Reset... ', '', 'Options("reset")'),
        )),
    )),
    M('Image', (
        M('Type', (
            I('8-bit', '', 'Converter("8-bit")'),
            I('16-bit', '', 'Converter("16-bit")'),
            I('32-bit', '', 'Converter("32-bit")'),
            I('8-bit Color', '', 'Converter("8-bit Color")'),
            I('RGB Color', '', 'Converter("RGB Color")'),
            SEP,
            I('RGB Stack', '', 'Converter("RGB Stack")'),
            I('HSB Stack', '', 'Converter("HSB Stack")'),
            I('HSB (32-bit)', '', 'Converter("HSB (32-bit)")'),
            I('Lab Stack', '', 'Converter("Lab Stack")'),
        )),
        SEP,
        M('Adjust', (
            I('Brightness/Contrast...', 'C', 'frame.ContrastAdjuster'),
            I('Window/Level...', '', 'frame.ContrastAdjuster("wl")'),
            I('Color Balance...', '', 'frame.ContrastAdjuster("balance")'),
            I('Threshold...', 'T', 'frame.ThresholdAdjuster'),
            I('Color Threshold...', '', 'frame.ColorThresholder'),
            I('Size...', '', 'Resizer'),
            I('Canvas Size...', '', 'CanvasResizer'),
            I('Line Width... ', '', 'frame.LineWidthAdjuster'),
            I('Coordinates...', '', 'Coordinates'),
        )),
        I('Show Info...', 'i', 'ImageInfo'),
        I('Properties...', 'P', 'filter.ImageProperties'),
        M('Color', (
            I('Split Channels', '', 'ChannelSplitter'),
            I('Merge Channels...', '', 'RGBStackMerge'),
            I('Arrange Channels...', '', 'ChannelArranger'),
            I('Invert LUTs', '', 'SimpleCommands("invert")'),
            I('Channels Tool...', 'Z', 'frame.Channels'),
            SEP,
            I('Stack to RGB', '', 'RGBStackConverter'),
            I('Make Composite', '', 'CompositeConverter'),
            I('Show LUT', '', 'filter.LutViewer'),
            I('Display LUTs', '', 'SimpleCommands("display")'),
            I('Edit LUT...', '', 'LUT_Editor'),
            I('Color Picker...', 'K', 'frame.ColorPicker'),
        )),
        M('Stacks', (
            I('Add Slice', '', 'StackEditor("add")'),
            I('Delete Slice', '', 'StackEditor("delete")'),
            I('Next Slice [>]', '', 'Animator("next")'),
            I('Previous Slice [<]', '', 'Animator("previous")'),
            I('Set Slice...', '', 'Animator("set")'),
            SEP,
            I('Images to Stack', '', 'ImagesToStack'),
            I('Stack to Images', '', 'StackEditor("toimages")'),
            I('Make Montage...', '', 'MontageMaker'),
            I('Reslice [/]...', '', 'Slicer'),
            I('Orthogonal Views', 'H', 'Orthogonal_Views'),
            I('Z Project...', '', 'ZProjector'),
            I('3D Project...', '', 'Projector'),
            I('Plot Z-axis Profile', '', 'ZAxisProfiler'),
            I('Measure Stack...', '', 'SimpleCommands("measure")'),
            I('Label...', '', 'filter.StackLabeler'),
            I('Statistics', '', 'Stack_Statistics'),
            M('Animation', (
                I('Start Animation [\\]', '', 'Animator("start")'),
                I('Stop Animation', '', 'Animator("stop")'),
                I('Animation Options...', '', 'Animator("options")'),
            )),
            M('Tools', (
                I('Combine...', '', 'StackCombiner'),
                I('Concatenate...', '', 'Concatenator'),
                I('Grouped Z Project...', '', 'GroupedZProjector'),
                I('Insert...', '', 'StackInserter'),
                I('Magic Montage Tools', '', 'SimpleCommands("magic")'),
                I('Make Substack...', '', 'SubstackMaker'),
                I('Montage to Stack...', '', 'StackMaker'),
                I('Plot XY Profile', '', 'StackPlotter'),
                I('Reduce...', '', 'StackReducer'),
                I('Remove Slice Labels', '', 'SimpleCommands("remove")'),
                I('Reverse', '', 'StackReverser'),
                I('Set Label...', '', 'SimpleCommands("set")'),
            )),
        )),
        M('Hyperstacks', (
            I('New Hyperstack...', '', 'HyperStackMaker'),
            I('Stack to Hyperstack...', '', 'HyperStackConverter("stacktohs")'),
            I('Hyperstack to Stack', '', 'HyperStackConverter("hstostack")'),
            I('Reduce Dimensionality...', '', 'HyperStackReducer'),
            I('Make Subset...', '', 'SubstackMaker'),
        )),
        SEP,
        I('Crop', 'X', 'Resizer("crop")'),
        I('Duplicate...', 'D', 'Duplicator'),
        I('Rename...', '', 'SimpleCommands("rename")'),
        I('Scale...', 'e', 'Scaler'),
        M('Transform', (
            I('Flip Horizontally', '', 'filter.Transformer("fliph")'),
            I('Flip Vertically', '', 'filter.Transformer("flipv")'),
            I('Flip Z', '', 'StackReverser'),
            I('Rotate 90 Degrees Right', '', 'filter.Transformer("right")'),
            I('Rotate 90 Degrees Left', '', 'filter.Transformer("left")'),
            I('Rotate... ', '', 'filter.Rotator'),
            I('Translate...', '', 'filter.Translator'),
            I('Bin...', '', 'Binner'),
            SEP,
            I('Image to Results', '', 'SimpleCommands("itor")'),
            I('Results to Image', '', 'SimpleCommands("rtoi")'),
        )),
        M('Zoom', (
            I('In [+]', '', 'Zoom("in")'),
            I('Out [-]', '', 'Zoom("out")'),
            I('Original Scale', '4', 'Zoom("orig")'),
            I('View 100%', '5', 'Zoom("100%")'),
            I('To Selection', '', 'Zoom("to")'),
            I('Scale to Fit', '', 'Zoom("scale")'),
            I('Set... ', '', 'Zoom("set")'),
            I('Maximize', '', 'Zoom("max")'),
        )),
        M('Overlay', (
            I('Add Selection...', 'b', 'OverlayCommands("add")'),
            I('Add Image...', '', 'OverlayCommands("image")'),
            I('Hide Overlay', '', 'OverlayCommands("hide")'),
            I('Show Overlay', '', 'OverlayCommands("show")'),
            I('From ROI Manager', '', 'OverlayCommands("from")'),
            I('To ROI Manager', '', 'OverlayCommands("to")'),
            I('Remove Overlay', '', 'OverlayCommands("remove")'),
            I('Toggle Overlay', '', 'OverlayCommands("toggle")'),
            I('List Elements', '', 'OverlayCommands("list")'),
            I('Flatten', 'F', 'OverlayCommands("flatten")'),
            I('Labels...', '', 'OverlayLabels'),
            I('Measure Overlay', '', 'OverlayCommands("measure")'),
            I('Overlay Options...', 'Y', 'OverlayCommands("options")'),
        )),
        SEP,
        M('Lookup Tables', (
            I('Invert LUT', '', 'LutLoader("invert")'),
            I('Apply LUT', '', 'filter.LutApplier'),
            SEP,
            I('Fire', '', 'LutLoader("fire")'),
            I('Grays', '', 'LutLoader("grays")'),
            I('Ice', '', 'LutLoader("ice")'),
            I('Spectrum', '', 'LutLoader("spectrum")'),
            I('3-3-2 RGB', '', 'LutLoader("3-3-2 RGB")'),
            I('Red', '', 'LutLoader("red")'),
            I('Green', '', 'LutLoader("green")'),
            I('Blue', '', 'LutLoader("blue")'),
            I('Cyan', '', 'LutLoader("cyan")'),
            I('Magenta', '', 'LutLoader("magenta")'),
            I('Yellow', '', 'LutLoader("yellow")'),
            I('Red/Green', '', 'LutLoader("redgreen")'),
        )),
    )),
    M('Process', (
        I('Smooth', 'S', 'filter.Filters("smooth")'),
        I('Sharpen', '', 'filter.Filters("sharpen")'),
        I('Find Edges', '', 'filter.Filters("edge")'),
        I('Find Maxima...', '', 'filter.MaximumFinder'),
        I('Enhance Contrast...', '', 'ContrastEnhancer'),
        M('Noise', (
            I('Add Noise', '', 'filter.Filters("add")'),
            I('Add Specified Noise...', '', 'filter.Filters("noise")'),
            I('Salt and Pepper', '', 'filter.SaltAndPepper'),
            SEP,
            I('Despeckle', '', 'filter.RankFilters("despeckle")'),
            I('Remove Outliers...', '', 'filter.RankFilters("outliers")'),
            I('Remove NaNs...', '', 'filter.RankFilters("nan")'),
        )),
        M('Shadows', (
            I('North', '', 'filter.Shadows("north")'),
            I('Northeast', '', 'filter.Shadows("northeast")'),
            I('East', '', 'filter.Shadows("east")'),
            I('Southeast', '', 'filter.Shadows("southeast")'),
            I('South', '', 'filter.Shadows("south")'),
            I('Southwest', '', 'filter.Shadows("southwest")'),
            I('West', '', 'filter.Shadows("west")'),
            I('Northwest', '', 'filter.Shadows("northwest")'),
            SEP,
            I('Shadows Demo', '', 'filter.Shadows("demo")'),
        )),
        M('Binary', (
            I('Make Binary', '', 'Thresholder'),
            I('Convert to Mask', '', 'Thresholder("mask")'),
            SEP,
            I('Erode', '', 'filter.Binary("erode")'),
            I('Dilate', '', 'filter.Binary("dilate")'),
            I('Open', '', 'filter.Binary("open")'),
            I('Close-', '', 'filter.Binary("close")'),
            I('Median', '', 'filter.Binary("median")'),
            SEP,
            I('Outline', '', 'filter.Binary("outline")'),
            I('Fill Holes', '', 'filter.Binary("fill")'),
            I('Skeletonize', '', 'filter.Binary("skel")'),
            SEP,
            I('Distance Map', '', 'filter.EDM("edm")'),
            I('Ultimate Points', '', 'filter.EDM("points")'),
            I('Watershed', '', 'filter.EDM("watershed")'),
            I('Voronoi', '', 'filter.EDM("voronoi")'),
            SEP,
            I('Options...', '', 'filter.Binary("options")'),
        )),
        M('Math', (
            I('Add...', '', 'filter.ImageMath("add")'),
            I('Subtract...', '', 'filter.ImageMath("sub")'),
            I('Multiply...', '', 'filter.ImageMath("mul")'),
            I('Divide...', '', 'filter.ImageMath("div")'),
            I('AND...', '', 'filter.ImageMath("and")'),
            I('OR...', '', 'filter.ImageMath("or")'),
            I('XOR...', '', 'filter.ImageMath("xor")'),
            I('Min...', '', 'filter.ImageMath("min")'),
            I('Max...', '', 'filter.ImageMath("max")'),
            I('Gamma...', '', 'filter.ImageMath("gamma")'),
            I('Set...', '', 'filter.ImageMath("set")'),
            I('Log', '', 'filter.ImageMath("log")'),
            I('Exp', '', 'filter.ImageMath("exp")'),
            I('Square', '', 'filter.ImageMath("sqr")'),
            I('Square Root', '', 'filter.ImageMath("sqrt")'),
            I('Reciprocal', '', 'filter.ImageMath("reciprocal")'),
            I('NaN Background', '', 'filter.ImageMath("nan")'),
            I('Abs', '', 'filter.ImageMath("abs")'),
            I('Macro...', '', 'filter.ImageMath("macro")'),
        )),
        M('FFT', (
            I('FFT', '', 'FFT("fft")'),
            I('Inverse FFT', '', 'FFT("inverse")'),
            I('Redisplay Power Spectrum', '', 'FFT("redisplay")'),
            I('FFT Options...', '', 'FFT("options")'),
            SEP,
            I('Bandpass Filter...', '', 'filter.FFTFilter'),
            I('Custom Filter...', '', 'filter.FFTCustomFilter'),
            I('FD Math...', '', 'FFTMath'),
            I('Swap Quadrants', '', 'FFT("swap")'),
            I('Make Circular Selection...', '', 'CircularRoiMaker'),
        )),
        M('Filters', (
            I('Convolve...', '', 'filter.Convolver'),
            I('Gaussian Blur...', '', 'filter.GaussianBlur'),
            I('Median...', '', 'filter.RankFilters("median")'),
            I('Mean...', '', 'filter.RankFilters("mean")'),
            I('Minimum...', '', 'filter.RankFilters("min")'),
            I('Maximum...', '', 'filter.RankFilters("max")'),
            I('Unsharp Mask...', '', 'filter.UnsharpMask'),
            I('Variance...', '', 'filter.RankFilters("variance")'),
            I('Top Hat...', '', 'filter.RankFilters("tophat")'),
            SEP,
            I('Gaussian Blur 3D...', '', 'GaussianBlur3D'),
            I('Median 3D...', '', 'Filters3D("median")'),
            I('Mean 3D...', '', 'Filters3D("mean")'),
            I('Minimum 3D...', '', 'Filters3D("min")'),
            I('Maximum 3D...', '', 'Filters3D("max")'),
            I('Variance 3D...', '', 'Filters3D("var")'),
            SEP,
            I('Show Circular Masks...', '', 'filter.RankFilters("masks")'),
        )),
        SEP,
        M('Batch', (
            I('Measure...', '', 'BatchMeasure'),
            I('Convert...', '', 'BatchConverter'),
            I('Macro... ', '', 'BatchProcessor'),
            I('Virtual Stack...', '', 'BatchProcessor("stack")'),
        )),
        I('Image Calculator...', '', 'ImageCalculator'),
        I('Subtract Background...', '', 'filter.BackgroundSubtracter'),
        I('Repeat Command', 'r'),
    )),
    M('Analyze', (
        I('Measure', 'm', 'filter.Analyzer'),
        I('Analyze Particles...', '', 'filter.ParticleAnalyzer'),
        I('Summarize', '', 'filter.Analyzer("sum")'),
        I('Distribution...', '', 'Distribution'),
        I('Label', '', 'filter.Filler("label")'),
        I('Clear Results', '', 'filter.Analyzer("clear")'),
        I('Set Measurements...', '', 'filter.Analyzer("set")'),
        SEP,
        I('Set Scale...', '', 'filter.ScaleDialog'),
        I('Calibrate...', '', 'filter.Calibrator'),
        I('Histogram', 'h', 'Histogram'),
        I('Plot Profile', 'k', 'Profiler("plot")'),
        I('Surface Plot...', '', 'SurfacePlotter'),
        M('Gels', (
            I('Select First Lane', '1', 'GelAnalyzer("first")'),
            I('Select Next Lane', '2', 'GelAnalyzer("next")'),
            I('Plot Lanes', '3', 'GelAnalyzer("plot")'),
            I('Re-plot Lanes', '', 'GelAnalyzer("replot")'),
            I('Reset', '', 'GelAnalyzer("reset")'),
            I('Label Peaks', '', 'GelAnalyzer("label")'),
            I('Gel Analyzer Options...', '', 'GelAnalyzer("options")'),
        )),
        M('Tools', (
            I('Save XY Coordinates...', '', 'XYCoordinates'),
            I('Fractal Box Count...', '', 'filter.FractalBoxCounter'),
            I('Analyze Line Graph', '', 'filter.LineGraphAnalyzer'),
            I('Curve Fitting...', '', 'frame.Fitter'),
            I('ROI Manager...', '', 'frame.RoiManager'),
            I('Scale Bar...', '', 'ScaleBar'),
            I('Calibration Bar...', '', 'CalibrationBar'),
            I('Synchronize Windows', '', 'frame.SyncWindows'),
            I('Grid...', '', 'Grid'),
        )),
    )),
    M('Plugins', (
        M('Macros', (
            I('Install...', '', 'MacroInstaller'),
            I('Run...', '', 'Macro_Runner'),
            I('Edit...', '', 'Compiler("edit")'),
            I('Record...', 'u', 'frame.Recorder'),
        )),
        M('Shortcuts', (
            I('Add Shortcut... ', '', 'Hotkeys("install")'),
            I('Add Shortcut by Name... ', '', 'Hotkeys("install2")'),
            I('Remove Shortcut...', '', 'Hotkeys("remove")'),
            I('List Shortcuts', '', 'CommandLister("shortcuts")'),
            I('List Commands', '', 'Hotkeys("list")'),
        )),
        M('Utilities', (
            I('Control Panel...', 'U', 'ControlPanel'),
            I('Find Commands...', 'l', 'CommandFinder'),
            I('Commands...', 'L', 'frame.Commands'),
            I('Search...', '', 'SimpleCommands("search")'),
            SEP,
            I('Capture Screen', 'G', 'ScreenGrabber'),
            I('Capture Delayed...', '', 'ScreenGrabber("delay")'),
            I('Capture Image', '', 'ScreenGrabber("image")'),
            SEP,
            I('Benchmark', '', 'Benchmark'),
            I('Reset...', '', 'SimpleCommands("reset")'),
        )),
        M('New', (
            I('Macro', '', 'NewPlugin("macro")'),
            I('Macro Tool', '', 'NewPlugin("macro-tool")'),
            SEP,
            I('Text Window...', '', 'NewPlugin("text+dialog")'),
        )),
    )),
    M('Window', (
        I('Show All', '', 'WindowOrganizer("show")'),
        I('Main Window [return]', '', 'WindowOrganizer("imagej")'),
        I('Put Behind [tab]', '', 'Commands("tab")'),
        I('Cascade', '', 'WindowOrganizer("cascade")'),
        I('Tile', '', 'WindowOrganizer("tile")'),
    )),
    M('Help', (
        I('ImageJ Website...', '', 'BrowserLauncher'),
        I('Macro Functions...', '', 'BrowserLauncher'),
        SEP,
        I('Check for Updates...', '', 'RS-StatChem'),
        I('About RS-StatChem...', '', 'RS-StatChem'),
    )),
)

# ---------------------------------------------------------------------------------- queries


def iter_items(menus=MENU_LAYOUT, path: tuple = ()) -> Iterator[tuple[tuple, MenuItem]]:
    """Every leaf as ``(path, item)``; ``path`` holds the menu titles from the top."""
    for node in menus:
        if node is None:
            continue
        if isinstance(node, Menu):
            yield from iter_items(node.items, path + (node.title,))
        else:
            yield path, node


def iter_menus(menus=MENU_LAYOUT, path: tuple = ()) -> Iterator[tuple[tuple, Menu]]:
    for node in menus:
        if isinstance(node, Menu):
            yield path + (node.title,), node
            yield from iter_menus(node.items, path + (node.title,))


def _owner_for_path(path: str) -> str | None:
    best, area = -1, None
    for prefix, key in OWNER_RULES:
        if (path == prefix or path.startswith(prefix + ">")) and len(prefix) > best:
            best, area = len(prefix), key
    return area


_INDEX: dict[str, tuple[tuple, MenuItem]] = {}


def _index() -> dict[str, tuple[tuple, MenuItem]]:
    if not _INDEX:
        for path, item in iter_items():
            _INDEX.setdefault(item.label, (path, item))
    return _INDEX


def find(label: str) -> MenuItem | None:
    hit = _index().get(label)
    return hit[1] if hit else None


def path_of(label: str) -> tuple | None:
    """Menu titles leading to ``label`` (e.g. ("Process", "Filters"))."""
    hit = _index().get(label)
    return hit[0] if hit else None


def owner_of(label: str) -> str | None:
    """Area key owning a command label (None when the label is not in the menus)."""
    path = path_of(label)
    if path is None:
        return None
    return _owner_for_path(">".join(path + (label,)))


def labels_of_area(area: str) -> list[str]:
    return [item.label for path, item in iter_items()
            if _owner_for_path(">".join(path + (item.label,))) == area]


def qt_shortcut(item: MenuItem) -> str:
    """Qt key sequence for a menu shortcut ("Ctrl" is Cmd on macOS); "" when none."""
    k = item.shortcut
    if not k:
        return ""
    if k.isalpha() and k.isupper():
        return f"Ctrl+Shift+{k}"
    return f"Ctrl+{k.upper()}"
