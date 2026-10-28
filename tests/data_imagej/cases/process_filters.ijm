// WP-B Process > Smooth/Sharpen/Find Edges, Shadows, Noise, Filters (2-D and 3-D) on the
// synthetic 8/16/32-bit images, an RGB image and stacks built here (saved as input_*), blobs.gif
// (inverting LUT) and Cell_Colony.jpg. Each output is <input>_<operation>[_<parameters>].tif.
init();

function openIn(name) {
    if (startsWith(name, "input_")) open(OUT + name + ".tif");
    else if (name == "blobs") open(IN + "blobs.gif");
    else if (name == "cell") open(IN + "Cell_Colony.jpg");
    else open(IN + name + ".tif");
}

function apply(img, cmd, opt, name) {
    openIn(img);
    if (opt == "") run(cmd);
    else run(cmd, opt);
    saveImage(img + "_" + name);
    close();
}

function applyRoi(img, cmd, opt, name, roi) {
    openIn(img);
    if (roi == "rect") makeRectangle(10, 12, 30, 25);
    else makeOval(8, 10, 40, 30);
    if (opt == "") run(cmd);
    else run(cmd, opt);
    run("Select None");
    saveImage(img + "_" + name + "_" + roi);
    close();
}

// ---- inputs built in the macro (and the decoded blobs / Cell_Colony pixels)
open(IN + "Cell_Colony.jpg");
saveImage("input_cell");
close();
open(IN + "blobs.gif");
saveImage("input_blobs");
close();
open(IN + "synth8.tif");
w = getWidth(); h = getHeight();
v = newArray(w * h);
for (y = 0; y < h; y++) for (x = 0; x < w; x++) v[x + y * w] = getPixel(x, y);
close();
newImage("rgb", "RGB black", w, h, 1);
for (y = 0; y < h; y++) for (x = 0; x < w; x++) {
    r = v[x + y * w]; g = v[(w - 1 - x) + y * w]; b = 255 - v[y + x * w];
    setPixel(x, y, (r << 16) + (g << 8) + b);
}
saveImage("input_rgb");
close();
newImage("stk8", "8-bit black", w, h, 5);
for (s = 1; s <= 5; s++) {
    setSlice(s);
    for (y = 0; y < h; y++) for (x = 0; x < w; x++) setPixel(x, y, v[((x + 7 * s) % w) + y * w]);
}
saveImage("input_stk8");
run("16-bit");
run("Multiply...", "value=157 stack");
saveImage("input_stk16");
run("32-bit");
run("Divide...", "value=4133 stack");
saveImage("input_stk32");
close();
newImage("rgbstk", "RGB black", 32, 32, 4);
for (s = 1; s <= 4; s++) {
    setSlice(s);
    for (y = 0; y < 32; y++) for (x = 0; x < 32; x++) {
        r = v[(x + 5 * s) + y * w]; g = v[x + (y + 3 * s) * w]; b = 255 - v[y + x * w];
        setPixel(x, y, (r << 16) + (g << 8) + b);
    }
}
saveImage("input_rgbstk");
close();
open(IN + "synth32.tif");
for (i = 0; i < 60; i++) setPixel((i * 37) % 64, (i * 13 + i * i) % 64, NaN);
for (y = 30; y < 34; y++) for (x = 20; x < 25; x++) setPixel(x, y, NaN);
saveImage("input_nan32");
close();

imgs = newArray("synth8", "synth16", "synth32", "input_rgb");
// ---- 3x3 filters and shadows
dirs = newArray("North", "Northeast", "East", "Southeast", "South", "Southwest", "West", "Northwest");
for (i = 0; i < imgs.length; i++) {
    apply("" + imgs[i], "Smooth", "", "smooth");
    apply("" + imgs[i], "Sharpen", "", "sharpen");
    apply("" + imgs[i], "Find Edges", "", "edges");
    for (d = 0; d < dirs.length; d++) apply("" + imgs[i], dirs[d], "", "shadow_" + toLowerCase(dirs[d]));
    apply("" + imgs[i], "Despeckle", "", "despeckle");
}
// ---- Gaussian, unsharp mask
sig = newArray("0.5", "2", "10", "25");
for (i = 0; i < imgs.length; i++) {
    for (s = 0; s < sig.length; s++) apply("" + imgs[i], "Gaussian Blur...", "sigma=" + sig[s], "gauss_" + sig[s]);
    apply("" + imgs[i], "Unsharp Mask...", "radius=1 mask=0.6", "unsharp_1_0.6");
    apply("" + imgs[i], "Unsharp Mask...", "radius=3 mask=0.9", "unsharp_3_0.9");
}
apply("cell", "Gaussian Blur...", "sigma=25", "gauss_25");
apply("cell", "Gaussian Blur...", "sigma=3.3", "gauss_3.3");
// ---- rank filters
rank = newArray("Median...", "Mean...", "Minimum...", "Maximum...", "Variance...");
rn = newArray("median", "mean", "min", "max", "variance");
radii = newArray("0.5", "1", "1.5", "2", "2.5", "5", "10");
for (k = 0; k < rank.length; k++) {
    for (r = 0; r < radii.length; r++) apply("synth8", rank[k], "radius=" + radii[r], "" + rn[k] + "_" + radii[r]);
    for (i = 1; i < imgs.length; i++) {
        apply("" + imgs[i], rank[k], "radius=2", "" + rn[k] + "_2");
        apply("" + imgs[i], rank[k], "radius=5", "" + rn[k] + "_5");
    }
}
apply("cell", "Median...", "radius=10", "median_10");
// ---- top hat, outliers
for (i = 0; i < imgs.length; i++) {
    apply("" + imgs[i], "Top Hat...", "radius=5", "tophat_5");
    apply("" + imgs[i], "Top Hat...", "radius=5 light", "tophat_5_light");
    apply("" + imgs[i], "Top Hat...", "radius=5 don't", "tophat_5_dont");
    apply("" + imgs[i], "Top Hat...", "radius=5 light don't", "tophat_5_light_dont");
}
apply("blobs", "Top Hat...", "radius=8", "tophat_8");
apply("blobs", "Top Hat...", "radius=8 light", "tophat_8_light");
apply("synth8", "Remove Outliers...", "radius=2 threshold=10 which=Bright", "outliers_2_10_bright");
apply("synth8", "Remove Outliers...", "radius=2 threshold=50 which=Bright", "outliers_2_50_bright");
apply("synth8", "Remove Outliers...", "radius=2 threshold=10 which=Dark", "outliers_2_10_dark");
apply("synth8", "Remove Outliers...", "radius=4 threshold=50 which=Dark", "outliers_4_50_dark");
apply("synth16", "Remove Outliers...", "radius=2 threshold=500 which=Bright", "outliers_2_500_bright");
apply("synth16", "Remove Outliers...", "radius=3 threshold=1500 which=Dark", "outliers_3_1500_dark");
apply("synth32", "Remove Outliers...", "radius=2 threshold=0.3 which=Bright", "outliers_2_0.3_bright");
apply("synth32", "Remove Outliers...", "radius=2 threshold=0.3 which=Dark", "outliers_2_0.3_dark");
apply("input_rgb", "Remove Outliers...", "radius=2 threshold=20 which=Bright", "outliers_2_20_bright");
apply("blobs", "Remove Outliers...", "radius=3 threshold=20 which=Bright", "outliers_3_20_bright");
// ---- NaN handling (32-bit)
apply("input_nan32", "Remove NaNs...", "radius=2", "removenan_2");
apply("input_nan32", "Remove NaNs...", "radius=1", "removenan_1");
apply("input_nan32", "Median...", "radius=2", "median_2");
apply("input_nan32", "Mean...", "radius=2", "mean_2");
apply("input_nan32", "Maximum...", "radius=2", "max_2");
apply("input_nan32", "Minimum...", "radius=1", "min_1");
// ---- convolve
k1 = "text1=[1 2 3 2 1\n2 4 6 4 2\n3 6 9 6 3\n2 4 6 4 2\n1 2 3 2 1\n]";
k2 = "text1=[-1 -1 -1 -1 -1\n-1 -1 -1 -1 -1\n-1 -1 24 -1 -1\n-1 -1 -1 -1 -1\n-1 -1 -1 -1 -1\n]";
k3 = "text1=[0 1 0 -1 0\n1 2.5 0 -2 -1\n0 1 0 -1 0\n]";
for (i = 0; i < imgs.length; i++) {
    apply("" + imgs[i], "Convolve...", k1 + " normalize", "convolve_k1_norm");
    apply("" + imgs[i], "Convolve...", k1, "convolve_k1");
    apply("" + imgs[i], "Convolve...", k2 + " normalize", "convolve_k2_norm");
    apply("" + imgs[i], "Convolve...", k3 + " normalize", "convolve_k3_norm");
}
// ---- selections
applyRoi("synth8", "Gaussian Blur...", "sigma=2", "gauss_2", "rect");
applyRoi("synth8", "Gaussian Blur...", "sigma=2", "gauss_2", "oval");
applyRoi("cell", "Gaussian Blur...", "sigma=12", "gauss_12", "rect");
applyRoi("synth16", "Median...", "radius=2", "median_2", "oval");
applyRoi("synth8", "Mean...", "radius=3", "mean_3", "rect");
applyRoi("synth32", "Unsharp Mask...", "radius=2 mask=0.6", "unsharp_2_0.6", "rect");
applyRoi("synth8", "Find Edges", "", "edges", "oval");
applyRoi("synth8", "Top Hat...", "radius=5", "tophat_5", "rect");
applyRoi("input_rgb", "Smooth", "", "smooth", "oval");
applyRoi("input_rgb", "Sharpen", "", "sharpen", "rect");
applyRoi("synth8", "Convolve...", k1 + " normalize", "convolve_k1_norm", "oval");
// ---- stacks: 2-D filter on all slices, 3-D filters
stks = newArray("input_stk8", "input_stk16", "input_stk32");
f3 = newArray("Median 3D...", "Mean 3D...", "Minimum 3D...", "Maximum 3D...", "Variance 3D...");
fn3 = newArray("median3d", "mean3d", "min3d", "max3d", "variance3d");
for (i = 0; i < stks.length; i++) {
    apply("" + stks[i], "Gaussian Blur...", "sigma=2 stack", "gauss_2_stack");
    apply("" + stks[i], "Gaussian Blur 3D...", "x=2 y=2 z=2", "gauss3d_2_2_2");
    apply("" + stks[i], "Gaussian Blur 3D...", "x=1 y=1.5 z=0.7", "gauss3d_1_1.5_0.7");
    for (k = 0; k < f3.length; k++) {
        apply("" + stks[i], f3[k], "x=2 y=2 z=2", "" + fn3[k] + "_2_2_2");
        apply("" + stks[i], f3[k], "x=1 y=1.5 z=1", "" + fn3[k] + "_1_1.5_1");
    }
}
apply("input_rgbstk", "Gaussian Blur 3D...", "x=1 y=1 z=1.5", "gauss3d_1_1_1.5");
apply("input_rgbstk", "Median 3D...", "x=1 y=1 z=1", "median3d_1_1_1");
apply("input_rgbstk", "Mean 3D...", "x=1 y=1 z=1", "mean3d_1_1_1");
// ---- seeded noise (ImageProcessor.setRandomSeed via random("seed", n))
for (i = 0; i < imgs.length; i++) {
    openIn("" + imgs[i]);
    random("seed", 7);
    run("Add Noise");
    saveImage("" + imgs[i] + "_addnoise_seed7");
    close();
    openIn("" + imgs[i]);
    random("seed", 11);
    run("Add Specified Noise...", "standard=10");
    saveImage("" + imgs[i] + "_noise10_seed11");
    close();
}
openIn("synth8");
makeRectangle(10, 12, 30, 25);
random("seed", 3);
run("Add Specified Noise...", "standard=40");
run("Select None");
saveImage("synth8_noise40_seed3_rect");
close();
// ---- Show Circular Masks
run("Show Circular Masks...");
saveImage("circular_masks");
close();
