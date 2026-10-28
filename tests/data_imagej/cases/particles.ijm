// Analyze Particles on blobs.gif (Default threshold, black background off) and on the
// synthetic particle image, with the main option combinations. Results with every
// measurement; Summary tables as ImageJ shows them.
init();
run("Set Measurements...", "area mean standard modal min centroid center perimeter bounding fit shape feret's integrated median skewness kurtosis area_fraction stack display redirect=None decimal=9");
variants = newArray("default", "exclude", "include", "size", "circ");
opts = newArray("display clear summarize", "display exclude clear summarize",
    "display clear include summarize", "size=100-Infinity display clear summarize",
    "size=0-Infinity circularity=0.85-1.00 display clear summarize");
images = newArray("blobs.gif", "particles8.tif");
stems = newArray("blobs", "particles8");
for (i = 0; i < images.length; i++) {
    open(IN + images[i]);
    setAutoThreshold("Default");
    if (i == 1) setThreshold(128, 255);
    for (v = 0; v < variants.length; v++) {
        run("Analyze Particles...", opts[v]);
        base = "" + stems[i] + "_" + variants[v];
        saveResults(base + ".csv");
        saveTable("Summary", base + "_summary.csv");
        Table.reset("Summary");
        selectImage(images[i]);
    }
    close();
}
