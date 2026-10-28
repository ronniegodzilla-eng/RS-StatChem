// Analyze > Measure with every measurement on blobs.gif: whole image, rectangle, oval,
// polygon, line; the area ROIs again with a spatial scale (Set Scale, 4 pixels/um) and with
// "limit to threshold" (Default threshold). One CSV per measurement: <set>_<roi>.csv.
// (Without ImageJ's main window each Measure starts a new Results table.)
init();
all = "area mean standard modal min centroid center perimeter bounding fit shape feret's integrated median skewness kurtosis area_fraction stack display redirect=None decimal=9";
run("Set Measurements...", all);
open(IN + "blobs.gif");
function measureAs(name) {
    run("Measure");
    saveResults(name + ".csv");
    run("Clear Results");
}
for (pass = 0; pass < 3; pass++) {
    set = "uncalibrated";
    if (pass == 1) {
        set = "calibrated";
        run("Set Scale...", "distance=10 known=2.5 unit=um");
    }
    if (pass == 2) {
        set = "limit";
        run("Set Scale...", "distance=0 known=0 unit=pixel");
        run("Set Measurements...", "area mean standard min centroid integrated area_fraction limit display redirect=None decimal=9");
        setAutoThreshold("Default");
    }
    run("Select None");
    measureAs(set + "_whole");
    makeRectangle(20, 30, 80, 60);
    measureAs(set + "_rect");
    makeOval(100, 50, 71, 90);
    measureAs(set + "_oval");
    makePolygon(10, 10, 120, 30, 90, 140, 20, 100);
    measureAs(set + "_polygon");
    if (pass == 0) {
        makeLine(10, 20, 200, 180);
        measureAs(set + "_line");
    }
}
