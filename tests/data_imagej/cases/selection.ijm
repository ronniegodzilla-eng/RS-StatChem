// Edit > Selection operations and Plot Profile on blobs.gif (WP-C): masks of the resulting
// selections (Create Mask) and the profile values of a line, a wide line and a rectangle.
init();
open(IN + "blobs.gif");
id = getImageID();
function maskOf(name) {
    run("Create Mask");
    selectImage("Mask");
    saveImage(name);
    close();
    selectImage(id);
}
makePolygon(30, 30, 120, 40, 100, 140, 40, 110);
run("Enlarge...", "enlarge=7 pixel");
maskOf("enlarge_polygon");
makePolygon(30, 30, 120, 40, 100, 140, 40, 110);
run("Enlarge...", "enlarge=-6 pixel");
maskOf("shrink_polygon");
makeOval(60, 50, 80, 60);
run("Make Band...", "band=8");
maskOf("band_oval");
makePolygon(30, 30, 120, 40, 75, 60, 100, 140, 40, 110);
run("Convex Hull");
maskOf("convex_hull");
makePolygon(30, 30, 120, 40, 100, 140, 40, 110);
run("Fit Ellipse");
maskOf("fit_ellipse");
makeRectangle(40, 40, 60, 30);
run("Make Inverse");
maskOf("inverse_rect");
makeLine(10, 20, 200, 180);
p = getProfile();
f = File.open(OUT + "profile_line.csv");
print(f, "Y");
for (i = 0; i < p.length; i++) print(f, fmt(p[i]));
File.close(f);
makeRectangle(20, 30, 80, 60);
p = getProfile();
f = File.open(OUT + "profile_rect.csv");
print(f, "Y");
for (i = 0; i < p.length; i++) print(f, fmt(p[i]));
File.close(f);
close();
