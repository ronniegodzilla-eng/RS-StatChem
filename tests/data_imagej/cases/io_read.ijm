// Files written by RS-StatChem (tests/data_imagej/inputs/io, made by make_io_inputs.py) opened
// with ImageJ: what ImageJ sees is recorded in read_meta.csv / read_rois.csv / read_set.csv /
// read_lut.csv, and the pixels as ImageJ read them are saved as images.
init();
D = IN + "io" + File.separator;
meta = File.open(OUT + "read_meta.csv");
print(meta, "file,title,width,height,channels,slices,frames,bitDepth,pw,ph,pd,unit,dmin,dmax,mean,cal100,inverting,overlay,seltype,label,infolen,prop,finterval,mode,dicom_spacing");

function writeMeta(f, name) {
    getDimensions(w, h, c, z, t);
    getVoxelSize(pw, ph, pd, unit);
    getMinAndMax(dmin, dmax);
    m = getValue("Mean raw");
    inv = 0;
    if (is("Inverting LUT")) inv = 1;
    mode = "";
    if (c > 1 && bitDepth() != 24) Stack.getDisplayMode(mode);
    lab = getMetadata("Label");
    lab = replace(lab, ",", ";");
    lab = replace(lab, "\n", " | ");
    info = getMetadata("Info");
    prop = Property.get("Sample");
    sp = "";
    if (endsWith(name, ".dcm")) sp = getInfo("0028,0030");
    print(f, name + "," + getTitle() + "," + w + "," + h + "," + c + "," + z + "," + t + "," + bitDepth() + "," + fmt(pw) + "," + fmt(ph) + "," + fmt(pd) + "," + unit + "," + fmt(dmin) + "," + fmt(dmax) + "," + fmt(m) + "," + fmt(calibrate(100)) + "," + inv + "," + Overlay.size + "," + selectionType() + "," + lab + "," + lengthOf(info) + "," + prop + "," + fmt(Stack.getFrameInterval()) + "," + mode + "," + sp);
}

files = newArray("rs_blobs.tif", "rs_blobs_le.tif", "rs_s16cal.tif", "rs_signed.tif", "rs_32.tif",
    "rs_hs.tif", "rs_rgb.tif", "rs_rgbstack.tif", "rs_stack.tif", "rs_blobs.fits", "rs_s16.fits",
    "rs_32.fits", "rs_s16.png", "rs_blobs.png", "rs_blobs.gif", "rs_stack.gif", "rs_blobs.bmp",
    "rs_blobs.pgm", "rs_rgb.png", "rs_blobs.zip", "rs_s16.dcm", "rs_stack_raw.avi", "rs_rgb_png.avi");
for (i = 0; i < files.length; i++) {
    open(D + files[i]);
    writeMeta(meta, files[i]);
    stem = replace(files[i], ".", "_");
    run("Select None");
    Overlay.remove;
    saveImage("read_" + stem);
    close("*");
}
run("Text Image... ", "open=[" + D + "rs_32.txt]");
writeMeta(meta, "rs_32.txt");
saveImage("read_rs_32_txt");
close("*");
run("Raw...", "open=[" + D + "rs_s16_be.raw] image=[16-bit Unsigned] width=64 height=64 number=1");
writeMeta(meta, "rs_s16_be.raw");
saveImage("read_rs_s16_be_raw");
close("*");
File.close(meta);

open(IN + "blobs.gif");
run("LUT... ", "open=[" + D + "rs.lut]");
getLut(r, g, b);
f = File.open(OUT + "read_lut.csv");
print(f, "index,r,g,b");
for (i = 0; i < 256; i++) print(f, "" + i + "," + r[i] + "," + g[i] + "," + b[i]);
File.close(f);
close("*");

newImage("roicanvas", "8-bit black", 100, 80, 1);
names = newArray("rect", "roundrect", "oval", "subpixel_oval", "polygon", "subpixel", "freehand",
    "traced", "line", "arrow", "polyline", "freeline", "angle", "point", "multipoint", "rotrect",
    "ellipse", "composite", "text", "positioned");
rf = File.open(OUT + "read_rois.csv");
print(rf, "name,type,x,y,w,h,n,xs,ys,roiname,stroke,fill,width,area,c,z,t");
for (i = 0; i < names.length; i++) {
    open(D + "rs_" + names[i] + ".roi");
    getSelectionBounds(x, y, w, h);
    getSelectionCoordinates(xs, ys);
    sx = ""; sy = "";
    for (k = 0; k < xs.length; k++) { sx = sx + " " + fmt(xs[k]); sy = sy + " " + fmt(ys[k]); }
    area = -1;
    st = selectionType();
    if (st >= 0 && st <= 4 || st == 9) area = getValue("Area");
    Roi.getPosition(pc, pz, pt);
    print(rf, names[i] + "," + st + "," + x + "," + y + "," + w + "," + h + "," + xs.length + "," + String.trim(sx) + "," + String.trim(sy) + "," + Roi.getName + "," + Roi.getStrokeColor + "," + Roi.getFillColor + "," + fmt(Roi.getStrokeWidth) + "," + fmt(area) + "," + pc + "," + pz + "," + pt);
    run("Select None");
}
File.close(rf);
roiManager("Open", D + "rs_RoiSet.zip");
sf = File.open(OUT + "read_set.csv");
print(sf, "index,name,type");
n = roiManager("count");
for (i = 0; i < n; i++) {
    roiManager("select", i);
    print(sf, "" + i + "," + Roi.getName + "," + selectionType());
}
File.close(sf);
close("*");
