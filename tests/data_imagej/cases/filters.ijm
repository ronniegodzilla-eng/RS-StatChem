// Process > Filters on the synthetic 8/16/32-bit images: Gaussian Blur (sigma 0.5, 2, 10) and
// the RankFilters (median, mean, minimum, maximum, variance) with radius 0.5, 1, 2, 5 on 8-bit,
// radius 2 on 16/32-bit. Each output is saved as <input>_<filter>_<parameter>.tif.
init();
images = newArray("synth8", "synth16", "synth32");
sigmas = newArray("0.5", "2", "10");
rank = newArray("Median...", "Mean...", "Minimum...", "Maximum...", "Variance...");
rankNames = newArray("median", "mean", "min", "max", "variance");
for (i = 0; i < images.length; i++) {
    for (s = 0; s < sigmas.length; s++) {
        open(IN + images[i] + ".tif");
        run("Gaussian Blur...", "sigma=" + sigmas[s]);
        saveImage("" + images[i] + "_gauss_" + sigmas[s]);
        close();
    }
    radii = newArray("2");
    if (i == 0) radii = newArray("0.5", "1", "2", "5");
    for (k = 0; k < rank.length; k++) {
        for (r = 0; r < radii.length; r++) {
            open(IN + images[i] + ".tif");
            cmd = rank[k];
            run(cmd, "radius=" + radii[r]);
            saveImage("" + images[i] + "_" + rankNames[k] + "_" + radii[r]);
            close();
        }
    }
}
