// Auto-threshold levels (Image > Adjust > Threshold, setAutoThreshold) for every method, with
// and without "dark", on 8-bit (blobs.gif with its inverting LUT, Cell_Colony.jpg) and on the
// synthetic 16-bit and 32-bit images. "legacy" = setAutoThreshold() without a method.
init();
methods = getList("threshold.methods");
images = newArray("blobs.gif", "Cell_Colony.jpg", "synth16.tif", "synth32.tif");
f = File.open(OUT + "thresholds.csv");
print(f, "image,method,dark,lower,upper");
for (i = 0; i < images.length; i++) {
    open(IN + images[i]);
    setAutoThreshold();
    getThreshold(lower, upper);
    print(f, images[i] + ",legacy,0," + fmt(lower) + "," + fmt(upper));
    for (m = 0; m < methods.length; m++) {
        for (dark = 0; dark <= 1; dark++) {
            arg = methods[m];
            if (dark == 1) arg = arg + " dark";
            setAutoThreshold(arg);
            getThreshold(lower, upper);
            print(f, images[i] + "," + methods[m] + "," + dark + "," + fmt(lower) + "," + fmt(upper));
        }
    }
    close();
}
File.close(f);
