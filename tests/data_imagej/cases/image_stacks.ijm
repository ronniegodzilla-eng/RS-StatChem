// Image > Stacks: Z Project (all methods), Grouped Z Project, Reslice, Make Montage, Montage to
// Stack, 3D Project (Projector), Reduce, Reverse, Substack, Combine, Concatenate, stack type
// conversions and Stack to Hyperstack. The input stacks (6 crops of the synthetic images, saved
// as input_stack8/16/32/rgb) are built here with Crop + Images to Stack.
init();
open(IN + "synth8.tif"); rename("r");
run("Duplicate...", "title=g"); run("Invert");
selectImage("r"); run("Duplicate...", "title=b"); run("Rotate 90 Degrees Right");
run("Merge Channels...", "c1=r c2=g c3=b");
saveImage("input_rgb"); close("*");

kinds = newArray("8", "16", "32", "rgb");
for (k = 0; k < kinds.length; k++) {
    for (s = 0; s < 6; s++) {
        if (kinds[k] == "rgb") open(OUT + "input_rgb.tif");
        else open(IN + "synth" + kinds[k] + ".tif");
        makeRectangle(2 * s, s, 48, 40);
        run("Crop");
        rename("slice" + s);
    }
    run("Images to Stack", "name=stack" + kinds[k] + " title=slice use");
    saveImage("input_stack" + kinds[k]);
    close("*");
}
function stackIn(kind) { open(OUT + "input_stack" + kind + ".tif"); }

methods = newArray("Average Intensity", "Max Intensity", "Min Intensity", "Sum Slices", "Standard Deviation", "Median");
short = newArray("avg", "max", "min", "sum", "sd", "median");
for (k = 0; k < kinds.length; k++) {
    for (m = 0; m < methods.length; m++) {
        stackIn(kinds[k]);
        run("Z Project...", "start=1 stop=6 projection=[" + methods[m] + "]");
        saveImage("zp_" + kinds[k] + "_" + short[m]); close("*");
    }
    stackIn(kinds[k]);
    run("Z Project...", "start=2 stop=5 projection=[Max Intensity]");
    saveImage("zp_" + kinds[k] + "_max_2_5"); close("*");
    stackIn(kinds[k]);
    run("Grouped Z Project...", "projection=[Average Intensity] group=2");
    saveImage("gzp_" + kinds[k] + "_avg2"); close("*");
    stackIn(kinds[k]);
    run("Reslice [/]...", "output=1.000 start=Top avoid");
    saveImage("reslice_" + kinds[k] + "_top"); close("*");
    stackIn(kinds[k]);
    run("Reslice [/]...", "output=1.000 start=Left avoid");
    saveImage("reslice_" + kinds[k] + "_left"); close("*");
    stackIn(kinds[k]);
    run("Reslice [/]...", "output=1.000 start=Bottom flip rotate avoid");
    saveImage("reslice_" + kinds[k] + "_bottom_flip_rotate"); close("*");
    stackIn(kinds[k]);
    run("Properties...", "channels=1 slices=6 frames=1 pixel_width=1 pixel_height=1 voxel_depth=2.5");
    run("Reslice [/]...", "output=1.000 start=Right");
    saveImage("reslice_" + kinds[k] + "_right_z2.5"); close("*");
    stackIn(kinds[k]);
    run("Make Montage...", "columns=3 rows=2 scale=1");
    saveImage("montage_" + kinds[k]); close("*");
    stackIn(kinds[k]);
    run("Make Montage...", "columns=2 rows=2 scale=0.5 first=2 last=6 increment=1 border=2");
    saveImage("montage_" + kinds[k] + "_half_border"); close("*");
    stackIn(kinds[k]);
    run("Make Montage...", "columns=3 rows=2 scale=1");
    run("Montage to Stack...", "columns=3 rows=2 border=0");
    saveImage("m2s_" + kinds[k]); close("*");
    stackIn(kinds[k]);
    run("Reduce...", "reduction=2");
    saveImage("reduce_" + kinds[k]); close("*");
    stackIn(kinds[k]);
    run("Bin...", "x=2 y=2 z=3 bin=Average");
    saveImage("binz_" + kinds[k]); close("*");
}
// 3D Project (8-bit, 16-bit and RGB stacks)
stackIn("8"); run("3D Project...", "projection=[Brightest Point] axis=Y-Axis slice=1 initial=0 total=90 rotation=30 lower=1 upper=255 opacity=0 surface=100 interior=100");
saveImage("p3d_8_bright_y"); close("*");
stackIn("8"); run("3D Project...", "projection=[Brightest Point] axis=X-Axis slice=1 initial=10 total=60 rotation=25 lower=1 upper=255 opacity=0 surface=100 interior=50");
saveImage("p3d_8_bright_x_cue"); close("*");
stackIn("8"); run("3D Project...", "projection=[Nearest Point] axis=X-Axis slice=1 initial=0 total=90 rotation=45 lower=100 upper=255 opacity=0 surface=50 interior=100");
saveImage("p3d_8_nearest_x_cue"); close("*");
stackIn("8"); run("3D Project...", "projection=[Mean Value] axis=Z-Axis slice=1 initial=0 total=90 rotation=30 lower=1 upper=255 opacity=0 surface=100 interior=100");
saveImage("p3d_8_mean_z"); close("*");
stackIn("8"); run("3D Project...", "projection=[Brightest Point] axis=Z-Axis slice=1 initial=0 total=40 rotation=20 lower=50 upper=200 opacity=40 surface=70 interior=60");
saveImage("p3d_8_bright_z_opacity"); close("*");
stackIn("8"); run("3D Project...", "projection=[Brightest Point] axis=Y-Axis slice=3 initial=0 total=60 rotation=30 lower=1 upper=255 opacity=0 surface=100 interior=100 interpolate");
saveImage("p3d_8_bright_y_interp"); close("*");
stackIn("8"); run("3D Project...", "projection=[Nearest Point] axis=Y-Axis slice=2 initial=0 total=60 rotation=30 lower=90 upper=255 opacity=0 surface=100 interior=100");
saveImage("p3d_8_nearest_y_z2"); close("*");
stackIn("16"); run("3D Project...", "projection=[Brightest Point] axis=Y-Axis slice=1 initial=0 total=30 rotation=30 lower=1 upper=255 opacity=0 surface=100 interior=100");
saveImage("p3d_16_bright_y"); close("*");
stackIn("rgb"); run("3D Project...", "projection=[Brightest Point] axis=Y-Axis slice=1 initial=0 total=30 rotation=30 lower=1 upper=255 opacity=0 surface=100 interior=100");
saveImage("p3d_rgb_bright_y"); close("*");
// stack tools
stackIn("8"); run("Reverse"); saveImage("reverse_8"); close("*");
stackIn("8"); run("Make Substack...", "slices=2-5"); saveImage("substack_8_2_5"); close("*");
stackIn("8"); run("Make Substack...", "slices=1,3,6"); saveImage("substack_8_list"); close("*");
stackIn("8"); rename("a"); stackIn("8"); rename("b"); run("Reverse");
run("Combine...", "stack1=a stack2=b"); saveImage("combine_8"); close("*");
stackIn("8"); rename("a"); stackIn("8"); rename("b"); run("Reverse");
run("Combine...", "stack1=a stack2=b combine"); saveImage("combine_8_vertical"); close("*");
stackIn("8"); rename("a"); stackIn("8"); rename("b"); run("Reverse");
run("Concatenate...", "  title=cat image1=a image2=b"); saveImage("concat_8"); close("*");
// stack conversions
stackIn("16"); setMinAndMax(3000, 30000); run("8-bit"); saveImage("stack16_8_range"); close("*");
stackIn("32"); run("16-bit"); saveImage("stack32_16"); close("*");
stackIn("32"); run("8-bit"); saveImage("stack32_8"); close("*");
stackIn("8"); run("Stack to Hyperstack...", "order=xyzct channels=2 slices=3 frames=1 display=Color");
saveImage("hyper_xyzct"); close("*");
stackIn("8"); run("Stack to Hyperstack...", "order=xytzc channels=3 slices=1 frames=2 display=Color");
saveImage("hyper_xytzc"); close("*");
stackIn("8"); run("Statistics"); saveResults("stats_8.csv"); close("*");
stackIn("16"); run("Statistics"); saveResults("stats_16.csv"); close("*");
