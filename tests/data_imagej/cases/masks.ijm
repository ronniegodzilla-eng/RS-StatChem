// Process > Binary > Convert to Mask of blobs.gif (Default and Otsu thresholds; black
// background off and on), and ROI masks (Edit > Selection > Create Mask) of rectangles, ovals
// and polygons on a 64x64 image, for the PolygonFiller / OvalRoi mask ports.
init();
methods = newArray("Default", "Otsu");
for (m = 0; m < methods.length; m++) {
    for (bb = 0; bb <= 1; bb++) {
        setOption("BlackBackground", bb == 1);
        open(IN + "blobs.gif");
        setAutoThreshold(methods[m]);
        run("Convert to Mask");
        saveImage("blobs_mask_" + methods[m] + "_bb" + bb);
        close();
    }
}
setOption("BlackBackground", false);
// shape specs: kind,x,y,w,h for rect/oval; kind then x/y pairs for polygons
specs = newArray(
    "rect,5,7,20,13", "oval,0,0,10,10", "oval,3,2,7,12", "oval,10,7,31,20", "oval,1,1,2,3",
    "oval,20,20,1,1", "oval,5,5,40,41", "oval,30,10,15,33",
    "poly,10,10,50,14,40,55,12,40", "poly,5,5,30,5,30,30,18,12,5,30",
    "poly,2.5,3.5,40.25,10.75,20.5,50.5", "poly,10,10,40,10,10,40,40,40");
f = File.open(OUT + "roi_specs.csv");
print(f, "name,spec");
for (i = 0; i < specs.length; i++) {
    newImage("canvas", "8-bit black", 64, 64, 1);
    p = split(specs[i], ",");
    if (p[0] == "rect") makeRectangle(parseFloat(p[1]), parseFloat(p[2]), parseFloat(p[3]), parseFloat(p[4]));
    if (p[0] == "oval") makeOval(parseFloat(p[1]), parseFloat(p[2]), parseFloat(p[3]), parseFloat(p[4]));
    if (p[0] == "poly") {
        n = (p.length - 1) / 2;
        xs = newArray(n); ys = newArray(n);
        for (k = 0; k < n; k++) { xs[k] = parseFloat(p[1 + 2*k]); ys[k] = parseFloat(p[2 + 2*k]); }
        makeSelection("polygon", xs, ys);
    }
    run("Create Mask");
    name = "roi_" + IJ.pad(i, 2);
    selectImage("Mask");
    saveImage(name);
    close();
    selectImage("canvas");
    close();
    print(f, name + ",\"" + specs[i] + "\"");
}
File.close(f);
