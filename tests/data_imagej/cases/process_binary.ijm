// Process > Binary (Erode, Dilate, Open, Close-, Median, Outline, Fill Holes, Skeletonize,
// Distance Map, Ultimate Points, Watershed, Voronoi with the Binary Options iterations / count /
// black background / pad edges / EDM output) and Process > Find Maxima (every output type,
// prominence, strict, exclude edges, light background, above lower threshold).
// Inputs saved as in_<name>: blobsF = Convert to Mask of blobs.gif (Default, black background
// off -> inverting LUT), blobsT (black background on), partF / partT = particles8.tif with black
// background off / on (normal LUT), big = synthetic large shapes (EDM approximation, watershed).
init();

function opts(iter, count, bb, pad, edm) {
    s = "iterations=" + iter + " count=" + count;
    if (bb) s = s + " black";
    if (pad) s = s + " pad";
    s = s + " edm=" + edm + " do=Nothing";
    run("Options...", s);
}

function prepare(name) {
    if (name == "blobsF" || name == "blobsT") {
        setOption("BlackBackground", name == "blobsT");
        open(IN + "blobs.gif");
        setAutoThreshold("Default");
        run("Convert to Mask");
    } else if (name == "big") {
        newImage("big", "8-bit black", 200, 160, 1);
        makeOval(10, 10, 120, 100);
        run("Set...", "value=255");
        makeOval(95, 40, 90, 90);
        run("Set...", "value=255");
        makeRectangle(20, 125, 160, 22);
        run("Set...", "value=255");
        makeOval(40, 40, 30, 20);
        run("Set...", "value=0");
        run("Select None");
    } else {
        open(IN + "particles8.tif");
    }
    rename(name);
}

function isBB(name) {
    if (name == "blobsT") return true;
    if (name == "partT") return true;
    if (name == "big") return true;
    return false;
}

masks = newArray("blobsF", "blobsT", "partF", "partT", "big");
ops = newArray("Erode", "Dilate", "Open", "Close-");
opNames = newArray("erode", "dilate", "open", "close");
iters = newArray(1, 2, 3, 1);
counts = newArray(1, 1, 3, 4);
for (m = 0; m < masks.length; m++) {
    name = masks[m];
    bb = isBB(name);
    opts(1, 1, bb, false, "Overwrite");
    prepare(name);
    saveImage("in_" + name);
    close();
    for (o = 0; o < ops.length; o++) {
        for (k = 0; k < iters.length; k++) {
            opts(iters[k], counts[k], bb, false, "Overwrite");
            prepare(name);
            run(ops[o]);
            saveImage("" + name + "_" + opNames[o] + "_i" + iters[k] + "c" + counts[k]);
            close();
        }
    }
    // pad edges when eroding
    for (k = 0; k < 3; k += 2) {
        opts(iters[k], counts[k], bb, true, "Overwrite");
        prepare(name);
        run("Erode");
        saveImage("" + name + "_erodepad_i" + iters[k] + "c" + counts[k]);
        close();
        opts(iters[k], counts[k], bb, true, "Overwrite");
        prepare(name);
        run("Open");
        saveImage("" + name + "_openpad_i" + iters[k] + "c" + counts[k]);
        close();
    }
    opts(1, 1, bb, false, "Overwrite");
    simple = newArray("Outline", "Fill Holes", "Skeletonize", "Watershed");
    simpleNames = newArray("outline", "fill", "skel", "watershed");
    for (s = 0; s < simple.length; s++) {
        prepare(name);
        run(simple[s]);
        saveImage("" + name + "_" + simpleNames[s]);
        close();
    }
    prepare(name);
    run("Median", "radius=3");
    saveImage("" + name + "_median_r3");
    close();
    prepare(name);
    run("Median", "radius=1.5");
    saveImage("" + name + "_median_r1.5");
    close();
    edms = newArray("Overwrite", "8-bit", "16-bit", "32-bit");
    for (e = 0; e < edms.length; e++) {
        opts(1, 1, bb, false, edms[e]);
        cmds = newArray("Distance Map", "Ultimate Points", "Voronoi");
        cmdNames = newArray("edm", "uep", "voronoi");
        for (c = 0; c < cmds.length; c++) {
            prepare(name);
            run(cmds[c]);
            saveImage("" + name + "_" + cmdNames[c] + "_" + edms[e]);
            close();
            if (edms[e] != "Overwrite") {
                selectImage(name);
                close();
            }
        }
    }
    opts(1, 1, false, false, "Overwrite");
}

// ---------------------------------------------------------------- Find Maxima
setOption("BlackBackground", false);
run("Options...", "iterations=1 count=1 edm=Overwrite do=Nothing");
images = newArray("synth8.tif", "synth16.tif", "synth32.tif", "blobs.gif");
names = newArray("synth8", "synth16", "synth32", "blobs");
lowP = newArray("10", "1500", "0.3", "10");
highP = newArray("50", "8000", "1.5", "50");
variants = newArray("", "strict", "exclude", "light", "strict exclude");
variantNames = newArray("plain", "strict", "exclude", "light", "strictexclude");
pts = "";
pts = "case,x,y\n";
for (i = 0; i < images.length; i++) {
    for (p = 0; p < 2; p++) {
        prom = lowP[i];
        if (p == 1) prom = highP[i];
        for (v = 0; v < variants.length; v++) {
            base = "" + names[i] + "_p" + p + "_" + variantNames[v];
            open(IN + images[i]);
            run("Find Maxima...", "prominence=" + prom + " " + variants[v] + " output=[Point Selection]");
            if (selectionType() == 10) {
                getSelectionCoordinates(xs, ys);
                for (k = 0; k < xs.length; k++) pts = pts + base + "," + xs[k] + "," + ys[k] + "\n";
            } else {
                pts = pts + base + ",-1,-1\n";
            }
            run("Select None");
            run("Find Maxima...", "prominence=" + prom + " " + variants[v] + " output=List");
            saveResults("maxima_" + base + "_list.csv");
            run("Clear Results");
            outs = newArray("Single Points", "Maxima Within Tolerance", "Segmented Particles");
            outNames = newArray("single", "tol", "seg");
            for (o = 0; o < outs.length; o++) {
                selectImage(1);
                run("Find Maxima...", "prominence=" + prom + " " + variants[v] + " output=[" + outs[o] + "]");
                saveImage("maxima_" + base + "_" + outNames[o]);
                close();
            }
            close("*");
        }
    }
}
// above lower threshold (dark background, normal LUT)
open(IN + "synth8.tif");
setThreshold(100, 255);
run("Find Maxima...", "prominence=10 above output=List");
saveResults("maxima_synth8_above_list.csv");
run("Clear Results");
selectImage(1);
run("Find Maxima...", "prominence=10 above output=[Segmented Particles]");
saveImage("maxima_synth8_above_seg");
close("*");
// Find Maxima in a rectangle and an oval selection
open(IN + "synth8.tif");
makeRectangle(10, 8, 40, 30);
run("Find Maxima...", "prominence=10 output=List");
saveResults("maxima_synth8_rect_list.csv");
run("Clear Results");
makeRectangle(10, 8, 40, 30);
run("Find Maxima...", "prominence=10 output=[Maxima Within Tolerance]");
saveImage("maxima_synth8_rect_tol");
close();
selectImage(1);
makeOval(5, 5, 50, 45);
run("Find Maxima...", "prominence=10 output=List");
saveResults("maxima_synth8_oval_list.csv");
run("Clear Results");
close("*");
File.saveString(pts, OUT + "maxima_points.csv");
