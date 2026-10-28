// Process > FFT family on synth8/16/32 (64x64, power of 2) and blobs.gif (256x254: padded to
// 256x256): FFT power spectrum, Inverse FFT (unmodified and with black "filter" / white "pass"
// masks painted on the spectrum), Bandpass Filter (defaults, stripes, no autoscale, no
// saturate, in a rectangle), Swap Quadrants, FD Math (correlate, convolve, deconvolve; with and
// without inverse), Custom Filter (8-bit filter of the padded size and a smaller one that is
// resized) and an RGB image (brightness route).
// saveImage() renames the saved window, so images are tracked by ID.
init();
images = newArray("synth8.tif", "synth16.tif", "synth32.tif", "blobs.gif");
names = newArray("synth8", "synth16", "synth32", "blobs");

function saveCopy(name) {
    id0 = getImageID();
    run("Duplicate...", "title=copy");
    saveImage(name);
    close();
    selectImage(id0);
}

for (i = 0; i < images.length; i++) {
    open(IN + images[i]);
    id = getImageID();
    run("FFT");
    fid = getImageID();
    saveCopy("" + names[i] + "_fft");
    run("Inverse FFT");
    saveImage("" + names[i] + "_ifft");
    close();
    // filter mode: black spots
    selectImage(fid);
    w = getWidth();
    setColor(0);
    fillOval(w/2 + 6, w/2 - 3, 8, 8);
    fillOval(w/2 - 30, w/2 + 10, 12, 5);
    run("Select None");
    saveCopy("" + names[i] + "_fft_black");
    run("Inverse FFT");
    saveImage("" + names[i] + "_ifft_black");
    close();
    selectImage(fid);
    close();
    // pass mode: white disc in the centre
    selectImage(id);
    run("FFT");
    fid = getImageID();
    w = getWidth();
    setColor(255);
    fillOval(w/2 - 10, w/2 - 10, 20, 20);
    run("Select None");
    saveCopy("" + names[i] + "_fft_white");
    run("Inverse FFT");
    saveImage("" + names[i] + "_ifft_white");
    close();
    selectImage(fid);
    close();
    selectImage(id);
    close();
}

// Bandpass Filter
bp = newArray("filter_large=40 filter_small=3 suppress=None tolerance=5 autoscale saturate",
              "filter_large=20 filter_small=2 suppress=Horizontal tolerance=5 autoscale saturate",
              "filter_large=30 filter_small=1 suppress=Vertical tolerance=20 autoscale",
              "filter_large=40 filter_small=3 suppress=None tolerance=5");
bpn = newArray("default", "horizontal", "vertical", "noscale");
for (i = 0; i < images.length; i++) {
    for (k = 0; k < bp.length; k++) {
        open(IN + images[i]);
        run("Bandpass Filter...", bp[k]);
        saveImage("" + names[i] + "_bandpass_" + bpn[k]);
        close();
    }
}
// Bandpass inside a rectangle
open(IN + "blobs.gif");
makeRectangle(30, 40, 100, 70);
run("Bandpass Filter...", "filter_large=40 filter_small=3 suppress=None tolerance=5 autoscale saturate");
run("Select None");
saveImage("blobs_bandpass_rect");
close();

// Swap Quadrants
open(IN + "synth8.tif");
run("Swap Quadrants");
saveImage("synth8_swap");
close();
open(IN + "synth32.tif");
run("Swap Quadrants");
saveImage("synth32_swap");
close();

// FD Math
ops = newArray("Correlate", "Convolve", "Deconvolve");
open(IN + "synth8.tif");
rename("a");
ida = getImageID();
open(IN + "synth32.tif");
rename("b");
idb = getImageID();
for (k = 0; k < ops.length; k++) {
    run("FD Math...", "image1=a operation=" + ops[k] + " image2=b result=Result do");
    saveImage("fdmath_" + ops[k]);
    close();
}
run("FD Math...", "image1=a operation=Correlate image2=a result=Result do");
saveImage("fdmath_autocorrelate");
close();
run("FD Math...", "image1=a operation=Convolve image2=b result=Result");
pid = getImageID();
saveCopy("fdmath_convolve_ps");
run("Inverse FFT");
saveImage("fdmath_convolve_ps_inverse");
close();
selectImage(pid);
close();
selectImage(ida);
close();
selectImage(idb);
close();

// Custom Filter (64x64 -> padded 128x128)
newImage("filt128", "8-bit black", 128, 128, 1);
setColor(255);
fillOval(34, 34, 60, 60);
setColor(128);
fillOval(54, 54, 20, 20);
id128 = getImageID();
saveCopy("custom_filter128");
newImage("filt64", "8-bit ramp", 64, 64, 1);
id64 = getImageID();
saveCopy("custom_filter64");
images2 = newArray("synth8.tif", "synth16.tif", "synth32.tif");
for (i = 0; i < images2.length; i++) {
    open(IN + images2[i]);
    run("Custom Filter...", "filter=filt128");
    saveImage("" + names[i] + "_custom128");
    close();
    open(IN + images2[i]);
    run("Custom Filter...", "filter=filt64");
    saveImage("" + names[i] + "_custom64");
    close();
}
selectImage(id128);
close();
selectImage(id64);
close();

// RGB: built from synth8 (r), flipped synth8 (g), inverted synth8 (b)
open(IN + "synth8.tif");
rename("r");
open(IN + "synth8.tif");
run("Flip Horizontally");
rename("g");
open(IN + "synth8.tif");
run("Invert");
rename("b");
run("Merge Channels...", "c1=r c2=g c3=b");
rid = getImageID();
saveCopy("rgb_input");
run("FFT");
fid = getImageID();
saveCopy("rgb_fft");
run("Inverse FFT");
saveImage("rgb_ifft");
close();
selectImage(fid);
close();
selectImage(rid);
run("Bandpass Filter...", "filter_large=40 filter_small=3 suppress=None tolerance=5 autoscale saturate");
saveImage("rgb_bandpass");
close();
