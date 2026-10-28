// Image > Type conversions (TypeConverter / ImageConverter / StackConverter / MedianCut /
// ColorSpaceConverter) and Image > Lookup Tables (LutLoader) with ImageJ's scaling rules.
// input_rgb is made here from synth8 (R), inverted synth8 (G) and synth8 rotated 90 (B).
init();
open(IN + "synth8.tif"); rename("r");
run("Duplicate...", "title=g"); run("Invert");
selectImage("r"); run("Duplicate...", "title=b"); run("Rotate 90 Degrees Right");
run("Merge Channels...", "c1=r c2=g c3=b");
saveImage("input_rgb"); close("*");

function rgbIn() { open(OUT + "input_rgb.tif"); }

// 16-bit -> 8-bit / 32-bit / RGB
open(IN + "synth16.tif"); run("8-bit"); saveImage("s16_8_auto"); close();
open(IN + "synth16.tif"); setMinAndMax(3000, 30000); run("8-bit"); saveImage("s16_8_range"); close();
setOption("ScaleConversions", false);
open(IN + "synth16.tif"); run("8-bit"); saveImage("s16_8_noscale"); close();
setOption("ScaleConversions", true);
open(IN + "synth16.tif"); run("32-bit"); saveImage("s16_32"); close();
open(IN + "synth16.tif"); setMinAndMax(3000, 30000); run("RGB Color"); saveImage("s16_rgb"); close();
// 32-bit -> 8 / 16 / RGB
open(IN + "synth32.tif"); run("8-bit"); saveImage("s32_8_auto"); close();
open(IN + "synth32.tif"); setMinAndMax(-1, 4); run("8-bit"); saveImage("s32_8_range"); close();
open(IN + "synth32.tif"); run("16-bit"); saveImage("s32_16_auto"); close();
open(IN + "synth32.tif"); setMinAndMax(-1, 4); run("16-bit"); saveImage("s32_16_range"); close();
setOption("ScaleConversions", false);
open(IN + "synth32.tif"); run("8-bit"); saveImage("s32_8_noscale"); close();
open(IN + "synth32.tif"); run("16-bit"); saveImage("s32_16_noscale"); close();
setOption("ScaleConversions", true);
open(IN + "synth32.tif"); setMinAndMax(-1, 4); run("RGB Color"); saveImage("s32_rgb"); close();
// 8-bit with LUTs -> RGB
open(IN + "blobs.gif"); run("RGB Color"); saveImage("blobs_rgb"); close();
open(IN + "synth8.tif"); run("Fire"); setMinAndMax(40, 210); run("RGB Color"); saveImage("s8_fire_rgb"); close();
open(IN + "synth8.tif"); setMinAndMax(40, 210); run("8-bit"); saveImage("s8_8_range"); close();
// RGB -> ...
rgbIn(); run("8-bit"); saveImage("rgb_8"); close();
run("Conversions...", "scale weighted");
rgbIn(); run("8-bit"); saveImage("rgb_8_weighted"); close();
rgbIn(); run("32-bit"); saveImage("rgb_32_weighted"); close();
run("Conversions...", "scale");
rgbIn(); run("16-bit"); saveImage("rgb_16"); close();
rgbIn(); run("32-bit"); saveImage("rgb_32"); close();
rgbIn(); run("RGB Stack"); saveImage("rgb_rgbstack"); run("RGB Color"); saveImage("rgb_rgbstack_back"); close();
rgbIn(); run("HSB Stack"); saveImage("rgb_hsb"); run("RGB Color"); saveImage("rgb_hsb_back"); close();
rgbIn(); run("HSB (32-bit)"); saveImage("rgb_hsb32"); run("RGB Color"); saveImage("rgb_hsb32_back"); close();
rgbIn(); run("Lab Stack"); saveImage("rgb_lab"); run("RGB Color"); saveImage("rgb_lab_back"); close("*");
rgbIn(); run("8-bit Color", "number=256"); saveImage("rgb_mc256"); run("RGB Color"); saveImage("rgb_mc256_rgb"); close();
rgbIn(); run("8-bit Color", "number=16"); saveImage("rgb_mc16"); run("RGB Color"); saveImage("rgb_mc16_rgb"); close();
open(IN + "Cell_Colony.jpg"); run("RGB Color"); run("8-bit Color", "number=7"); run("RGB Color"); saveImage("colony_mc7_rgb"); close();
// composite -> RGB
open(IN + "synth8.tif"); rename("c1"); setMinAndMax(20, 180);
run("Duplicate...", "title=c2"); run("Invert"); setMinAndMax(50, 255);
run("Merge Channels...", "c1=c1 c2=c2 create");
run("RGB Color"); saveImage("composite_rgb"); close("*");
// Lookup tables on a 256x1 ramp, as RGB
luts = newArray("Fire", "Grays", "Ice", "Spectrum", "3-3-2 RGB", "Red", "Green", "Blue", "Cyan", "Magenta", "Yellow", "Red/Green");
names = newArray("fire", "grays", "ice", "spectrum", "rgb332", "red", "green", "blue", "cyan", "magenta", "yellow", "redgreen");
for (i = 0; i < luts.length; i++) {
    newImage("ramp", "8-bit ramp", 256, 1, 1);
    run(luts[i]);
    run("RGB Color");
    saveImage("lut_" + names[i]);
    close();
}
newImage("ramp", "8-bit ramp", 256, 1, 1); run("Fire"); run("Invert LUT"); run("RGB Color"); saveImage("lut_fire_inverted"); close();
// Apply LUT
open(IN + "synth8.tif"); setMinAndMax(60, 200); run("Apply LUT"); saveImage("apply8"); close();
open(IN + "synth16.tif"); setMinAndMax(3000, 30000); run("Apply LUT"); saveImage("apply16"); close();
