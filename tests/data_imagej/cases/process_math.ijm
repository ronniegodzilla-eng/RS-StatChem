// Process > Math (every operation) on synth8/16/32 and a synthetic RGB image, Math > Macro,
// Image Calculator (every operator; 8/16/32-bit, RGB, mixed types, 32-bit result, different
// sizes, divide-by-zero value, stacks) and Enhance Contrast (saturated stretch -> display
// range in contrast_ranges.csv, normalize, equalize incl. classic, selections, stacks).
init();

function makeRGB(title, flip) {
    newImage(title, "RGB black", 64, 64, 1);
    for (y = 0; y < 64; y++) {
        for (x = 0; x < 64; x++) {
            xx = x;
            if (flip) xx = 63 - x;
            r = (xx * 4 + y) % 256;
            g = (y * 3 + 40) % 256;
            b = (xx * y) % 256;
            if ((xx + y) % 9 == 0) { r = 0; g = 0; b = 0; }
            setPixel(x, y, (r << 16) | (g << 8) | b);
        }
    }
}

function mathOp(img, cmd, opts, name) {
    if (img == "rgb") makeRGB("rgb", false);
    else open(IN + img + ".tif");
    if (opts == "") run(cmd);
    else run(cmd, opts);
    saveImage("" + img + "_" + name);
    close();
}

// ---------------------------------------------------------------- inputs
makeRGB("rgb", false);
saveImage("input_rgb");
close();
makeRGB("rgb2", true);
saveImage("input_rgb2");
close();

// ---------------------------------------------------------------- Math
imgs = newArray("synth8", "synth16", "synth32", "rgb");
cmds = newArray("Add...", "Subtract...", "Multiply...", "Divide...", "Divide...", "AND...",
    "OR...", "XOR...", "Min...", "Max...", "Gamma...", "Gamma...", "Set...", "Log", "Exp",
    "Square", "Square Root", "Reciprocal", "Abs");
opts = newArray("value=25", "value=25", "value=1.5", "value=3", "value=0", "value=01010101",
    "value=01010101", "value=01010101", "value=100", "value=100", "value=0.5", "value=2",
    "value=77", "", "", "", "", "", "");
names = newArray("add", "sub", "mul", "div", "div0", "and", "or", "xor", "min", "max",
    "gamma05", "gamma2", "set", "log", "exp", "sqr", "sqrt", "recip", "abs");
for (i = 0; i < imgs.length; i++) {
    for (k = 0; k < cmds.length; k++) {
        mathOp(imgs[i], cmds[k], opts[k], names[k]);
    }
}
macros = newArray("v=v*2+x", "v=sin(d/4)*a*20+v", "if (d>20) v=0;", "v=getPixel(x+3,y+2)",
    "v=v-y*3", "a=round(a/0.5); if (a%2==0) v=v/2; else v=v+10;");
for (i = 0; i < imgs.length; i++) {
    for (k = 0; k < macros.length; k++) {
        mathOp(imgs[i], "Macro...", "code=[" + macros[k] + "]", "macro" + k);
    }
}
// NaN Background on a thresholded 32-bit image
open(IN + "synth32.tif");
setThreshold(-1, 2);
run("NaN Background");
saveImage("synth32_nan");
close();
// 16-bit gamma/log/exp use the display range
for (k = 0; k < 3; k++) {
    open(IN + "synth16.tif");
    setMinAndMax(2000, 30000);
    if (k == 0) run("Gamma...", "value=0.5");
    if (k == 1) run("Log");
    if (k == 2) run("Exp");
    saveImage("synth16_range" + k);
    close();
}
// selections: rectangle (8-bit add), oval (16-bit multiply), oval macro (32-bit)
open(IN + "synth8.tif");
makeRectangle(10, 12, 30, 20);
run("Add...", "value=60");
saveImage("synth8_roi_add");
close();
open(IN + "synth16.tif");
makeOval(8, 6, 40, 30);
run("Multiply...", "value=1.7");
saveImage("synth16_roi_mul");
close();
open(IN + "synth32.tif");
makeOval(8, 6, 40, 30);
run("Macro...", "code=[v=v*10+x]");
saveImage("synth32_roi_macro");
close();
open(IN + "synth8.tif");
makeRectangle(10, 12, 30, 20);
run("Macro...", "code=[v=getPixel(x+1,y)]");
saveImage("synth8_roi_getpixel");
close();
// stack math
newImage("stk", "8-bit ramp", 64, 64, 3);
setSlice(2);
run("Add...", "value=40 slice");
setSlice(3);
run("Multiply...", "value=0.5 slice");
saveImage("input_stack");
run("Add...", "value=10 stack");
run("Macro...", "code=[v=v+z*20] stack");
saveImage("stack_math");
close();

// ---------------------------------------------------------------- Image Calculator
ops = newArray("Add", "Subtract", "Multiply", "Divide", "AND", "OR", "XOR", "Min", "Max",
    "Average", "Difference", "Copy", "Transparent-zero");
lops = newArray("add", "sub", "mul", "div", "and", "or", "xor", "min", "max", "ave", "diff",
    "copy", "zero");
open(IN + "blobs.gif");
makeRectangle(0, 0, 64, 64);
run("Crop");
rename("b8");
saveImage("input_b8");
rename("b8");
open(IN + "synth8.tif");
open(IN + "synth16.tif");
run("Duplicate...", "title=b16");
run("Flip Horizontally");
open(IN + "synth32.tif");
run("Duplicate...", "title=b32");
run("Flip Horizontally");
run("Duplicate...", "title=b32z");
run("Min...", "value=0");
makeRGB("rgb", false);
makeRGB("rgb2", true);
for (k = 0; k < ops.length; k++) {
    imageCalculator(ops[k] + " create", "synth8.tif", "b8");
    saveImage("calc8_" + lops[k]);
    close();
    imageCalculator(ops[k] + " create", "synth16.tif", "b16");
    saveImage("calc16_" + lops[k]);
    close();
    imageCalculator(ops[k] + " create", "synth32.tif", "b32");
    saveImage("calc32_" + lops[k]);
    close();
    imageCalculator(ops[k] + " create", "rgb", "rgb2");
    saveImage("calcrgb_" + lops[k]);
    close();
    imageCalculator(ops[k] + " create 32-bit", "synth8.tif", "b8");
    saveImage("calc8f_" + lops[k]);
    close();
}
imageCalculator("Divide create", "synth32.tif", "b32z");
saveImage("calc32_divzero");
close();
run("Misc...", "divide=0");
imageCalculator("Divide create", "synth32.tif", "b32z");
saveImage("calc32_divzero_value");
close();
run("Misc...", "divide=Infinity");
// mixed types
imageCalculator("Add create", "synth8.tif", "synth16.tif");
saveImage("calcmix_8_16");
close();
imageCalculator("Subtract create", "synth16.tif", "synth8.tif");
saveImage("calcmix_16_8");
close();
imageCalculator("Multiply create", "synth8.tif", "synth32.tif");
saveImage("calcmix_8_32mul");
close();
imageCalculator("Add create", "synth8.tif", "synth32.tif");
saveImage("calcmix_8_32add");
close();
imageCalculator("Add create", "synth32.tif", "synth16.tif");
saveImage("calcmix_32_16");
close();
imageCalculator("Add create", "rgb", "synth8.tif");
saveImage("calcmix_rgb_8");
close();
imageCalculator("Add create 32-bit", "rgb", "rgb2");
saveImage("calcrgb_add_float");
close();
// in place (image1 changed)
imageCalculator("Difference", "synth16.tif", "b16");
selectImage("synth16.tif");
saveImage("calc16_inplace_diff");
rename("synth16.tif");
// different sizes
open(IN + "particles8.tif");
imageCalculator("Add create", "particles8.tif", "synth8.tif");
saveImage("calcsize_create");
close();
imageCalculator("Max", "particles8.tif", "synth8.tif");
selectImage("particles8.tif");
saveImage("calcsize_inplace");
close();
// stacks
newImage("stk", "8-bit ramp", 64, 64, 3);
setSlice(2);
run("Add...", "value=40 slice");
setSlice(3);
run("Multiply...", "value=0.5 slice");
imageCalculator("Add create stack", "stk", "synth8.tif");
saveImage("calcstack_add");
close();
run("Duplicate...", "title=stk2 duplicate");
run("Flip Vertically", "stack");
imageCalculator("Subtract create 32-bit stack", "stk", "stk2");
saveImage("calcstack_sub_float");
close();
imageCalculator("Average stack", "stk", "stk2");
selectImage("stk");
saveImage("calcstack_inplace_ave");
run("Close All");

// ---------------------------------------------------------------- Enhance Contrast
f = File.open(OUT + "contrast_ranges.csv");
print(f, "name,min,max");
srcs = newArray("synth8.tif", "blobs.gif", "synth16.tif", "synth32.tif", "Cell_Colony.jpg");
sats = newArray("0.35", "5", "0");
for (i = 0; i < srcs.length; i++) {
    for (s = 0; s < sats.length; s++) {
        open(IN + srcs[i]);
        run("Enhance Contrast...", "saturated=" + sats[s]);
        getMinAndMax(min, max);
        print(f, srcs[i] + "_" + sats[s] + "," + fmt(min) + "," + fmt(max));
        close();
    }
}
// selection: stats from the selection only
open(IN + "synth16.tif");
makeOval(10, 10, 30, 25);
run("Enhance Contrast...", "saturated=1");
getMinAndMax(min, max);
print(f, "synth16_oval_1," + fmt(min) + "," + fmt(max));
close();
open(IN + "synth32.tif");
makeRectangle(5, 5, 30, 40);
run("Enhance Contrast...", "saturated=2");
getMinAndMax(min, max);
print(f, "synth32_rect_2," + fmt(min) + "," + fmt(max));
close();
File.close(f);
names = newArray("synth8", "synth16", "synth32", "blobs");
files = newArray("synth8.tif", "synth16.tif", "synth32.tif", "blobs.gif");
for (i = 0; i < files.length; i++) {
    open(IN + files[i]);
    run("Enhance Contrast...", "saturated=0.35 normalize");
    saveImage("" + names[i] + "_normalize");
    close();
    open(IN + files[i]);
    run("Enhance Contrast...", "saturated=3 normalize");
    saveImage("" + names[i] + "_normalize3");
    close();
    if (i != 2) {
        open(IN + files[i]);
        run("Enhance Contrast...", "saturated=0.35 equalize");
        saveImage("" + names[i] + "_equalize");
        close();
        open(IN + files[i]);
        setKeyDown("alt");
        run("Enhance Contrast...", "saturated=0.35 equalize");
        setKeyDown("none");
        saveImage("" + names[i] + "_equalize_classic");
        close();
    }
}
// RGB stretch changes the pixels
makeRGB("rgb", false);
run("Enhance Contrast...", "saturated=1");
saveImage("rgb_stretch1");
close();
makeRGB("rgb", false);
run("Enhance Contrast...", "saturated=0.35 equalize");
saveImage("rgb_equalize");
close();
// normalize inside a selection (default: update selection only) and "update all"
open(IN + "synth8.tif");
makeOval(10, 8, 34, 30);
run("Enhance Contrast...", "saturated=0.35 normalize");
saveImage("synth8_normalize_oval");
close();
open(IN + "synth8.tif");
makeOval(10, 8, 34, 30);
run("Enhance Contrast...", "saturated=0.35 normalize update");
saveImage("synth8_normalize_oval_all");
close();
// stacks
for (k = 0; k < 4; k++) {
    newImage("stk", "8-bit ramp", 64, 64, 3);
    setSlice(2);
    run("Add...", "value=40 slice");
    setSlice(3);
    run("Multiply...", "value=0.5 slice");
    setSlice(1);
    if (k == 0) run("Enhance Contrast...", "saturated=0.35 process_all");
    if (k == 1) run("Enhance Contrast...", "saturated=0.35 process_all use");
    if (k == 2) run("Enhance Contrast...", "saturated=0.35 equalize process_all");
    if (k == 3) run("Enhance Contrast...", "saturated=0.35 equalize process_all use");
    saveImage("stack_contrast" + k);
    close();
}
