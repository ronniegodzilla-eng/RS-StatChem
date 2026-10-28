// WP-B thresholding: extra auto-threshold levels (synth8, bilevel particles8, ROI-limited,
// no-reset, 16-bit full histogram, stack histograms), Convert to Mask / Make Binary on 8/16/32-bit
// and RGB images (legacy auto threshold and preset thresholds, black background off/on), stack
// conversions ("Convert Stack to Binary" options), and the colour spaces of the Color Threshold
// window (HSB Stack bytes, ColorThresholder.RGBtoLab / RGBtoYUV bytes) of a synthetic RGB image.
init();
methods = getList("threshold.methods");
f = File.open(OUT + "extra_thresholds.csv");
print(f, "case,options,lower,upper");

function thr(fh, name, opts) {
    setAutoThreshold(opts);
    getThreshold(lower, upper);
    print(fh, name + "," + opts + "," + fmt(lower) + "," + fmt(upper));
}

// synth8: all methods, both backgrounds
open(IN + "synth8.tif");
for (m = 0; m < methods.length; m++) {
    mm = methods[m];
    thr(f, "synth8", mm);
    thr(f, "synth8", mm + " dark");
}
close();
// bilevel image
open(IN + "particles8.tif");
thr(f, "particles8", "Default");
thr(f, "particles8", "Otsu dark");
thr(f, "particles8", "Huang");
close();
// rectangle selection on blobs
open(IN + "blobs.gif");
makeRectangle(20, 30, 100, 80);
thr(f, "blobs_rect", "Otsu");
thr(f, "blobs_rect", "Default dark");
thr(f, "blobs_rect", "Li");
close();
// no-reset on 32-bit with a narrowed display range
open(IN + "synth32.tif");
setMinAndMax(-1, 3);
thr(f, "synth32_noreset", "Otsu no-reset");
setMinAndMax(-1, 3);
thr(f, "synth32_noreset", "Default dark no-reset");
close();
// 16-bit full histogram
open(IN + "synth16.tif");
m16 = newArray("Default", "IJ_IsoData", "IsoData", "Li", "Mean", "Moments", "Otsu", "Percentile",
               "Triangle", "Yen", "Intermodes", "Minimum", "MinError", "MaxEntropy", "RenyiEntropy");
for (m = 0; m < m16.length; m++) {
    mm = m16[m];
    thr(f, "synth16_16bit", mm + " 16-bit");
    thr(f, "synth16_16bit", mm + " dark 16-bit");
}
close();

// stacks
function makeStack(file, name, bits) {
    open(IN + file);
    rename("s1");
    open(IN + file);
    if (bits == 8) run("Subtract...", "value=30");
    else run("Multiply...", "value=0.6");
    rename("s2");
    open(IN + file);
    if (bits == 8) run("Add...", "value=40");
    if (bits == 16) run("Add...", "value=3000");
    if (bits == 32) run("Add...", "value=1.5");
    rename("s3");
    run("Images to Stack", "name=" + name + " title=[] use");
}
makeStack("synth16.tif", "stack16", 16);
thr(f, "stack16", "Otsu dark stack");
thr(f, "stack16", "Default stack");
thr(f, "stack16", "Li dark 16-bit stack");
setSlice(2);
thr(f, "stack16_slice2", "Otsu dark");
saveImage("stack16_input");
close();
makeStack("synth32.tif", "stack32", 32);
thr(f, "stack32", "Li stack");
thr(f, "stack32", "Triangle dark stack");
saveImage("stack32_input");
close();
makeStack("synth8.tif", "stack8", 8);
thr(f, "stack8", "Huang dark stack");
thr(f, "stack8", "Default stack");
saveImage("stack8_input");
close();
File.close(f);

// ---------------------------------------------------------------------- masks
g = File.open(OUT + "mask_luts.csv");
print(g, "name,inverted,lower,upper");
function saveMask(fh, name) {
    inv = 0;
    if (is("Inverting LUT")) inv = 1;
    getThreshold(lower, upper);
    print(fh, name + "," + inv + "," + fmt(lower) + "," + fmt(upper));
    saveImage(name);
}

images = newArray("blobs.gif", "Cell_Colony.jpg", "synth8.tif", "synth16.tif", "synth32.tif");
stems = newArray("blobs", "colony", "synth8", "synth16", "synth32");
for (i = 0; i < images.length; i++) {
    for (bb = 0; bb <= 1; bb++) {
        setOption("BlackBackground", bb == 1);
        open(IN + images[i]);
        run("Convert to Mask");
        saveMask(g, "auto_" + stems[i] + "_bb" + bb);
        close();
    }
}
preset = newArray("Huang dark", "Otsu", "Triangle dark", "MinError", "Percentile dark");
presetNames = newArray("huangdark", "otsu", "triangledark", "minerror", "percentiledark");
pimages = newArray("Cell_Colony.jpg", "synth16.tif", "synth32.tif");
pstems = newArray("colony", "synth16", "synth32");
for (i = 0; i < pimages.length; i++) {
    for (p = 0; p < preset.length; p++) {
        for (bb = 0; bb <= 1; bb++) {
            setOption("BlackBackground", bb == 1);
            open(IN + pimages[i]);
            setAutoThreshold(preset[p]);
            run("Convert to Mask");
            saveMask(g, "preset_" + pstems[i] + "_" + presetNames[p] + "_bb" + bb);
            close();
        }
    }
}
setOption("BlackBackground", true);
open(IN + "Cell_Colony.jpg");
setAutoThreshold("Li dark");
run("Make Binary");
saveMask(g, "makebinary_colony_li_bb1");
close();
setOption("BlackBackground", false);
open(IN + "blobs.gif");
run("Make Binary");
saveMask(g, "makebinary_blobs_bb0");
close();
open(IN + "synth32.tif");
setThreshold(-0.5, 2.0);
run("Convert to Mask");
saveMask(g, "setthr_synth32_bb0");
close();
open(IN + "synth16.tif");
setThreshold(15000, 30000);
run("Convert to Mask");
saveMask(g, "setthr_synth16_bb0");
close();

// RGB input built from synth8 (red), its inverse (green) and a blurred copy (blue)
open(IN + "synth8.tif");
rename("r");
run("Duplicate...", "title=g");
run("Invert");
selectImage("r");
run("Duplicate...", "title=b");
run("Gaussian Blur...", "sigma=3");
run("Merge Channels...", "c1=r c2=g c3=b");
rename("rgb");
saveImage("rgb_input");
run("Duplicate...", "title=hsb");
run("HSB Stack");
saveImage("rgb_hsb");
close();
selectImage("rgb_input.tif");
run("Duplicate...", "title=lab");
call("ij.plugin.frame.ColorThresholder.RGBtoLab");
saveImage("rgb_lab");
close();
selectImage("rgb_input.tif");
run("Duplicate...", "title=yuv");
call("ij.plugin.frame.ColorThresholder.RGBtoYUV");
saveImage("rgb_yuv");
close();
selectImage("rgb_input.tif");
setOption("BlackBackground", false);
run("Duplicate...", "title=rgbmask");
run("Convert to Mask");
saveMask(g, "auto_rgb_bb0");
close();
selectImage("rgb_input.tif");
close();

// stack conversions ("Convert Stack to Binary"; the "black" key sets the background)
makeStack("synth16.tif", "stack16", 16);
run("Convert to Mask", "method=Otsu background=Dark calculate black");
saveMask(g, "stack16_otsu_dark_calc_black");
close();
makeStack("synth16.tif", "stack16", 16);
run("Convert to Mask", "method=Default background=Default");
saveMask(g, "stack16_legacy");
close();
makeStack("synth16.tif", "stack16", 16);
setAutoThreshold("Huang dark");
run("Convert to Mask", "method=Default background=Default black");
saveMask(g, "stack16_huangdark_preset_black");
close();
makeStack("synth8.tif", "stack8", 8);
run("Convert to Mask", "method=Default background=Default calculate");
saveMask(g, "stack8_default_calc");
close();
makeStack("synth8.tif", "stack8", 8);
run("Convert to Mask", "method=Li background=Light calculate black");
saveMask(g, "stack8_li_light_calc_black");
close();
makeStack("synth8.tif", "stack8", 8);
setSlice(2);
setAutoThreshold("Otsu");
run("Convert to Mask", "method=Otsu background=Light only");
saveMask(g, "stack8_only_slice2");
close();
makeStack("synth32.tif", "stack32", 32);
run("Convert to Mask", "method=Triangle background=Dark calculate");
saveMask(g, "stack32_triangle_dark_calc");
close();
makeStack("synth32.tif", "stack32", 32);
setAutoThreshold("Otsu dark stack");
run("Convert to Mask", "method=Default background=Default black");
saveMask(g, "stack32_otsustack_preset_black");
close();
File.close(g);
setOption("BlackBackground", false);
