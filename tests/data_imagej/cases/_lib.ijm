// Shared helpers, prepended to every case by scripts/imagej_golden.py.
// The argument is "<inputs dir>|<output dir>". Results are written with 9 decimals by
// saveResults()/saveTable() (not saveAs("Results"), whose target table is ambiguous without the
// ImageJ main window); images with saveImage() as TIFF.
var IN = "";
var OUT = "";

function init() {
    args = split(getArgument(), "|");
    IN = args[0] + File.separator;
    OUT = args[1] + File.separator;
    setBatchMode(true);
    setOption("BlackBackground", false);
    run("Options...", "iterations=1 count=1");
    run("Clear Results");
}

function fmt(v) {
    if (isNaN(v)) return "NaN";
    return d2s(v, 9);
}

function csvCell(s) {
    if (indexOf(s, ",") >= 0) return "\"" + s + "\"";
    return s;
}

// Results table -> CSV (columns of String.getResultsHeadings, full precision, no row numbers)
function saveResults(name) {
    heads = split(String.getResultsHeadings, "\t");
    cols = newArray(0);
    for (i = 0; i < heads.length; i++) {
        h = String.trim(heads[i]);
        if (h != "") cols = Array.concat(cols, h);
    }
    f = File.open(OUT + name);
    print(f, String.join(cols, ","));
    for (r = 0; r < nResults; r++) {
        line = "";
        for (i = 0; i < cols.length; i++) {
            if (cols[i] == "Label") cell = csvCell(getResultLabel(r));
            else cell = fmt(getResult(cols[i], r));
            if (i > 0) line = line + ",";
            line = line + cell;
        }
        print(f, line);
    }
    File.close(f);
}

// Any named table (e.g. "Summary") -> CSV; string cells are kept as text
function saveTable(title, name) {
    heads = split(Table.headings(title), "\t");
    f = File.open(OUT + name);
    print(f, String.join(heads, ","));
    n = Table.size(title);
    for (r = 0; r < n; r++) {
        line = "";
        for (i = 0; i < heads.length; i++) {
            text = Table.getString(heads[i], r, title);
            if (isNaN(parseFloat(text))) cell = csvCell(text);
            else cell = fmt(Table.get(heads[i], r, title));
            if (i > 0) line = line + ",";
            line = line + cell;
        }
        print(f, line);
    }
    File.close(f);
}

function saveImage(name) {
    saveAs("Tiff", OUT + name + ".tif");
}
