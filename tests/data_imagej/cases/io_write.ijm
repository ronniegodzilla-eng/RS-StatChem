// File I/O written by ImageJ (WP-A reads these back): TIFF (calibration, display range, value
// calibration, slice labels, Info, properties, hyperstacks with per-channel LUTs/ranges,
// overlays), ZIP, GIF, PNG, BMP, PGM, JPEG, raw, text image, FITS, AVI, LUT, XY coordinates,
// .roi files of every ROI type and a RoiSet.zip. meta.csv / rois.csv / luts.csv record what
// ImageJ had in memory. Files that must stay byte-exact use extensions other than ".tif"
// (".tif" files are absorbed into images.npz by scripts/imagej_golden.py).
init();
meta = File.open(OUT + "meta.csv");
print(meta, "file,width,height,channels,slices,frames,bitDepth,pw,ph,pd,unit,dmin,dmax,mean,finterval,cal0");

function writeMeta(f, name) {
    getDimensions(w, h, c, z, t);
    getVoxelSize(pw, ph, pd, unit);
    getMinAndMax(dmin, dmax);
    m = getValue("Mean raw");
    fi = Stack.getFrameInterval();
    print(f, name + "," + w + "," + h + "," + c + "," + z + "," + t + "," + bitDepth() + "," + fmt(pw) + "," + fmt(ph) + "," + fmt(pd) + "," + unit + "," + fmt(dmin) + "," + fmt(dmax) + "," + fmt(m) + "," + fmt(fi) + "," + fmt(calibrate(0)));
}

// ---- blobs: 8-bit with inverting LUT, spatial calibration, Info and a property
open(IN + "blobs.gif");
run("Set Scale...", "distance=4 known=1 unit=um");
setMetadata("Info", "blobs info line 1\nline 2");
Property.set("Sample", "blobs");
writeMeta(meta, "blobs_cal.tiff");
saveAs("Tiff", OUT + "blobs_cal.tiff");
saveAs("ZIP", OUT + "blobs_cal.zip");
saveAs("Gif", OUT + "blobs_out.gif");
saveAs("PNG", OUT + "blobs_out.png");
saveAs("BMP", OUT + "blobs_out.bmp");
saveAs("PGM", OUT + "blobs_out.pgm");
saveAs("Raw Data", OUT + "blobs_out.raw");
saveAs("Text Image", OUT + "blobs_out.txt");
saveAs("FITS", OUT + "blobs_out.fits");
run("Input/Output...", "jpeg=85");
saveAs("Jpeg", OUT + "blobs_out.jpg");
close();

// ---- 16-bit: display range, anisotropic calibration, slice label
open(IN + "synth16.tif");
run("Properties...", "channels=1 slices=1 frames=1 pixel_width=0.25 pixel_height=0.3 voxel_depth=1.5");
run("Set Scale...", "distance=4 known=1 pixel=1.2 unit=mm");
setMinAndMax(2000, 30000);
setMetadata("Label", "plane one");
writeMeta(meta, "synth16_cal.tiff");
saveAs("Tiff", OUT + "synth16_cal.tiff");
saveAs("PNG", OUT + "synth16_out.png");
saveAs("Text Image", OUT + "synth16_out.txt");
saveAs("FITS", OUT + "synth16_out.fits");
saveAs("Raw Data", OUT + "synth16_out.raw");
close();

// ---- 32-bit
open(IN + "synth32.tif");
setMinAndMax(-2, 3);
writeMeta(meta, "synth32_out.tiff");
saveAs("Tiff", OUT + "synth32_out.tiff");
saveAs("Text Image", OUT + "synth32_out.txt");
saveAs("FITS", OUT + "synth32_out.fits");
saveAs("Raw Data", OUT + "synth32_out.raw");
close();

// ---- hyperstack 2c x 3z x 4t, composite with LUTs, ranges, labels, frame interval
newImage("hs", "16-bit ramp", 40, 30, 2, 3, 4);
for (t = 1; t <= 4; t++) {
    for (z = 1; z <= 3; z++) {
        for (c = 1; c <= 2; c++) {
            Stack.setPosition(c, z, t);
            run("Add...", "value=" + (c * 1000 + z * 100 + t * 10) + " slice");
            setMetadata("Label", "c" + c + "z" + z + "t" + t);
        }
    }
}
run("Properties...", "channels=2 slices=3 frames=4 unit=um pixel_width=0.5 pixel_height=0.5 voxel_depth=2 frame=[1.5 sec]");
Stack.setDisplayMode("composite");
Stack.setPosition(1, 1, 1);
run("Magenta");
setMinAndMax(100, 50000);
Stack.setPosition(2, 1, 1);
run("Cyan");
setMinAndMax(2000, 60000);
Stack.setPosition(1, 2, 3);
writeMeta(meta, "hyperstack.tiff");
saveAs("Tiff", OUT + "hyperstack.tiff");
close();

// ---- RGB
newImage("rgb", "RGB black", 64, 48, 1);
setColor(255, 0, 0);
fillRect(4, 4, 20, 12);
setColor(10, 200, 30);
fillRect(30, 10, 25, 20);
setColor(20, 40, 250);
fillOval(8, 24, 30, 20);
writeMeta(meta, "rgb.tiff");
saveAs("Tiff", OUT + "rgb.tiff");
saveAs("PNG", OUT + "rgb_out.png");
saveAs("BMP", OUT + "rgb_out.bmp");
saveAs("Raw Data", OUT + "rgb_out.raw");
saveAs("ZIP", OUT + "rgb.zip");
close();

// ---- 8-bit stack -> AVI (uncompressed and PNG) and a multi-page TIFF
newImage("stk", "8-bit ramp", 48, 32, 5);
for (i = 1; i <= nSlices; i++) {
    setSlice(i);
    run("Add...", "value=" + (i * 20) + " slice");
}
setSlice(1);
writeMeta(meta, "stack8.tiff");
saveAs("Tiff", OUT + "stack8.tiff");
run("AVI... ", "compression=Uncompressed frame=7 save=[" + OUT + "stack8_raw.avi]");
run("AVI... ", "compression=PNG frame=7 save=[" + OUT + "stack8_png.avi]");
close();
File.close(meta);

// ---- LUT files
open(IN + "blobs.gif");
run("Fire");
saveAs("LUT", OUT + "fire.lut");
getLut(r, g, b);
f = File.open(OUT + "luts.csv");
print(f, "index,r,g,b");
for (i = 0; i < 256; i++) print(f, "" + i + "," + r[i] + "," + g[i] + "," + b[i]);
File.close(f);
close();

// ---- ROIs: one .roi per type, a RoiSet.zip, an overlay, XY coordinates
newImage("roi", "8-bit black", 100, 80, 1);
rf = File.open(OUT + "rois.csv");
print(rf, "name,type,x,y,w,h,xs,ys,stroke,width");

function saveRoi(f, name) {
    Roi.setName(name);
    getSelectionBounds(x, y, w, h);
    getSelectionCoordinates(xs, ys);
    sx = ""; sy = "";
    for (i = 0; i < xs.length; i++) { sx = sx + " " + fmt(xs[i]); sy = sy + " " + fmt(ys[i]); }
    print(f, name + "," + selectionType() + "," + x + "," + y + "," + w + "," + h + "," + String.trim(sx) + "," + String.trim(sy) + "," + Roi.getStrokeColor + "," + fmt(Roi.getStrokeWidth));
    saveAs("Selection", OUT + name + ".roi");
    roiManager("Add");
}

makeRectangle(10, 12, 30, 20);
Roi.setStrokeColor("red");
saveRoi(rf, "rect");
makeRectangle(12, 14, 40, 30, 10);
saveRoi(rf, "roundrect");
makeOval(20, 15, 40, 25);
Roi.setStrokeWidth(2);
saveRoi(rf, "oval");
makePolygon(10, 10, 50, 14, 40, 55, 12, 40);
saveRoi(rf, "polygon");
makeSelection("polygon", newArray(2.5, 40.25, 20.5), newArray(3.5, 10.75, 50.5));
saveRoi(rf, "subpixel");
makeSelection("freehand", newArray(10, 12, 15, 20, 18, 11), newArray(10, 8, 9, 15, 22, 18));
saveRoi(rf, "freehand");
makeLine(5, 5, 60, 40);
Roi.setStrokeWidth(3);
saveRoi(rf, "line");
makeLine(5, 5, 20, 40, 50, 10);
saveRoi(rf, "polyline");
makeSelection("freeline", newArray(5, 8, 12, 18, 25), newArray(30, 28, 31, 35, 33));
saveRoi(rf, "freeline");
makeSelection("angle", newArray(10, 30, 50), newArray(40, 10, 40));
saveRoi(rf, "angle");
makePoint(12, 13);
saveRoi(rf, "point");
makeSelection("point", newArray(5, 25, 45, 65), newArray(6, 26, 46, 66));
saveRoi(rf, "multipoint");
makeArrow(10, 10, 50, 50, "filled");
saveRoi(rf, "arrow");
makeRotatedRectangle(10, 20, 60, 50, 15);
saveRoi(rf, "rotrect");
makeEllipse(20, 30, 70, 50, 0.5);
saveRoi(rf, "ellipse");
makeRectangle(10, 10, 40, 40);
setKeyDown("shift");
makeOval(30, 30, 40, 40);
setKeyDown("none");
saveRoi(rf, "composite");
makeText("Hello", 10, 10);
saveRoi(rf, "text");
File.close(rf);
roiManager("Save", OUT + "RoiSet.zip");
makePolygon(10, 10, 50, 14, 40, 55, 12, 40);
saveAs("XY Coordinates", OUT + "polygon_xy.txt");
run("Select None");
makeRectangle(10, 12, 30, 20);
Overlay.addSelection("green");
makeOval(20, 15, 40, 25);
Overlay.addSelection("#ff00ff", 2);
makeLine(5, 5, 60, 40);
Overlay.addSelection("yellow");
run("Select None");
saveAs("Tiff", OUT + "overlay.tiff");
close();
