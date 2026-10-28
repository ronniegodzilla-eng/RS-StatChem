// Edit menu pixel operations (WP-A): Fill / Clear / Clear Outside / Draw (line widths 1-3,
// rectangle, polygon, line) / Invert (8, 16, 32-bit, with a selection, display ranges) and
// paste transfer modes.
init();
setForegroundColor(255, 255, 255);
setBackgroundColor(0, 0, 0);

open(IN + "synth8.tif");
makeOval(10, 10, 30, 20);
setForegroundColor(200, 200, 200);
run("Fill", "slice");
saveImage("fill8_oval");
close();

open(IN + "synth8.tif");
makeRectangle(5, 5, 20, 15);
setBackgroundColor(30, 30, 30);
run("Clear", "slice");
saveImage("clear8");
run("Clear Outside", "slice");
saveImage("outside8");
close();
setBackgroundColor(0, 0, 0);

setForegroundColor(255, 255, 255);
widths = newArray(1, 2, 3);
for (i = 0; i < widths.length; i++) {
    run("Line Width...", "line=" + widths[i]);
    open(IN + "synth8.tif");
    makeRectangle(10, 12, 30, 20);
    run("Draw", "slice");
    saveImage("draw_rect_w" + widths[i]);
    close();
    open(IN + "synth8.tif");
    makePolygon(10, 10, 50, 14, 40, 55, 12, 40);
    run("Draw", "slice");
    saveImage("draw_poly_w" + widths[i]);
    close();
    open(IN + "synth8.tif");
    makeLine(5, 5, 60, 40);
    run("Draw", "slice");
    saveImage("draw_line_w" + widths[i]);
    close();
}
run("Line Width...", "line=1");

open(IN + "synth8.tif");
run("Invert");
saveImage("invert8");
close();
open(IN + "synth8.tif");
makeOval(10, 10, 30, 20);
run("Invert", "slice");
saveImage("invert8_oval");
close();
open(IN + "synth16.tif");
run("Invert");
saveImage("invert16");
close();
open(IN + "synth16.tif");
setMinAndMax(2000, 30000);
makeRectangle(5, 5, 30, 30);
run("Invert", "slice");
saveImage("invert16_rect");
close();
open(IN + "synth32.tif");
setMinAndMax(-1, 2);
run("Invert");
saveImage("invert32");
close();

open(IN + "synth16.tif");
setMinAndMax(2000, 30000);
setForegroundColor(128, 128, 128);
makeRectangle(5, 5, 20, 10);
run("Fill", "slice");
saveImage("fill16");
close();
open(IN + "synth32.tif");
setMinAndMax(-1, 2);
makeRectangle(5, 5, 20, 10);
run("Fill", "slice");
saveImage("fill32");
close();
open(IN + "blobs.gif");
setForegroundColor(255, 255, 255);
makeRectangle(5, 5, 20, 10);
run("Fill", "slice");
saveImage("fill_blobs_white");
close();

modes = newArray("Copy", "Add", "Difference", "Transparent-zero", "Blend", "Subtract", "Max");
for (i = 0; i < modes.length; i++) {
    open(IN + "synth8.tif");
    makeRectangle(0, 0, 20, 20);
    run("Copy");
    makeRectangle(30, 30, 20, 20);
    setPasteMode(modes[i]);
    run("Paste");
    run("Select None");
    saveImage("paste_" + replace(modes[i], "-", "_"));
    close();
}
setPasteMode("Copy");
