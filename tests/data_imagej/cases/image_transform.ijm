// Image > Scale, Adjust > Size / Canvas Size, Transform > Rotate / Translate / Bin on the
// synthetic 8/16/32-bit images and an RGB image (input_rgb, as in image_types), for the
// ImageProcessor resize/scale/rotate/translate interpolation ports and Binner.
init();
open(IN + "synth8.tif"); rename("r");
run("Duplicate...", "title=g"); run("Invert");
selectImage("r"); run("Duplicate...", "title=b"); run("Rotate 90 Degrees Right");
run("Merge Channels...", "c1=r c2=g c3=b");
saveImage("input_rgb"); close("*");

images = newArray("synth8", "synth16", "synth32", "rgb");
function openIn(name) {
    if (name == "rgb") open(OUT + "input_rgb.tif");
    else open(IN + name + ".tif");
}
methods = newArray("None", "Bilinear", "Bicubic");
for (i = 0; i < images.length; i++) {
    name = images[i];
    for (m = 0; m < methods.length; m++) {
        meth = methods[m];
        // Scale (new window): up, down without and with averaging
        openIn(name);
        run("Scale...", "x=1.7 y=1.3 interpolation=" + meth + " create title=up");
        saveImage("" + name + "_scale_up_" + meth); close("*");
        openIn(name);
        run("Scale...", "x=0.6 y=0.45 interpolation=" + meth + " create title=dn");
        saveImage("" + name + "_scale_down_" + meth); close("*");
        openIn(name);
        run("Scale...", "x=0.6 y=0.45 interpolation=" + meth + " average create title=dna");
        saveImage("" + name + "_scale_downavg_" + meth); close("*");
        openIn(name);
        run("Scale...", "x=0.5 y=1.5 interpolation=" + meth + " average create title=mix");
        saveImage("" + name + "_scale_mix_" + meth); close("*");
        // Scale in place (no new window)
        openIn(name);
        run("Scale...", "x=0.7 y=0.8 interpolation=" + meth);
        saveImage("" + name + "_scale_inplace_" + meth); close("*");
        // Adjust > Size
        openIn(name);
        run("Size...", "width=50 height=37 depth=1 average interpolation=" + meth);
        saveImage("" + name + "_size_" + meth); close("*");
        // Rotate
        openIn(name);
        run("Rotate... ", "angle=15 grid=1 interpolation=" + meth);
        saveImage("" + name + "_rot15_" + meth); close("*");
        openIn(name);
        run("Rotate... ", "angle=-33.3 grid=1 interpolation=" + meth + " enlarge");
        saveImage("" + name + "_rotm33_enlarge_" + meth); close("*");
        // Translate
        openIn(name);
        run("Translate...", "x=3.5 y=-2.25 interpolation=" + meth);
        saveImage("" + name + "_translate_" + meth); close("*");
    }
    openIn(name);
    run("Translate...", "x=5 y=-3 interpolation=None");
    saveImage("" + name + "_translate_int"); close("*");
    bins = newArray("Average", "Median", "Min", "Max", "Sum");
    for (b = 0; b < bins.length; b++) {
        openIn(name);
        run("Bin...", "x=3 y=2 bin=" + bins[b]);
        saveImage("" + name + "_bin_" + bins[b]); close("*");
    }
    openIn(name);
    run("Canvas Size...", "width=80 height=50 position=Bottom-Right zero");
    saveImage("" + name + "_canvas"); close("*");
}
