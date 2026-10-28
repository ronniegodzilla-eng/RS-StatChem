// Process > Subtract Background: rolling ball (radius 50 = shrink 4, 20 = shrink 2, 8 = no
// shrink) and sliding paraboloid, light background, create background, disable smoothing, on
// blobs.gif (inverting LUT), Cell_Colony.jpg, synth8/16/32 and an RGB image (brightness and
// separate colours). Saved as <image>_bg_<variant>.
init();
images = newArray("blobs.gif", "Cell_Colony.jpg", "synth8.tif", "synth16.tif", "synth32.tif");
names = newArray("blobs", "colony", "synth8", "synth16", "synth32");
opts = newArray("rolling=50", "rolling=20 light", "rolling=8", "rolling=50 create",
                "rolling=20 disable", "rolling=50 sliding", "rolling=10 light sliding",
                "rolling=30 sliding disable", "rolling=15 create sliding",
                "rolling=120");
vn = newArray("r50", "r20_light", "r8", "r50_create", "r20_disable", "p50", "p10_light",
              "p30_disable", "p15_create", "r120");
for (i = 0; i < images.length; i++) {
    for (k = 0; k < opts.length; k++) {
        // the large Cell_Colony image only for a few variants (keeps the goldens small)
        if (names[i] == "colony" && vn[k] != "r50" && vn[k] != "r20_light" && vn[k] != "p50" && vn[k] != "r120")
            continue;
        open(IN + images[i]);
        run("Subtract Background...", opts[k]);
        saveImage("" + names[i] + "_bg_" + vn[k]);
        close();
    }
}
// RGB
open(IN + "synth8.tif");
rename("r");
open(IN + "synth8.tif");
run("Flip Horizontally");
rename("g");
open(IN + "synth8.tif");
run("Invert");
rename("b");
run("Merge Channels...", "c1=r c2=g c3=b");
rename("rgb");
saveImage("rgb_input");
close();
rgbopts = newArray("rolling=20", "rolling=20 separate", "rolling=10 light sliding",
                   "rolling=10 separate light sliding", "rolling=20 separate create");
rgbn = newArray("r20", "r20_sep", "p10_light", "p10_sep_light", "r20_sep_create");
for (k = 0; k < rgbopts.length; k++) {
    open(OUT + "rgb_input.tif");
    run("Subtract Background...", rgbopts[k]);
    saveImage("rgb_bg_" + rgbn[k]);
    close();
}
