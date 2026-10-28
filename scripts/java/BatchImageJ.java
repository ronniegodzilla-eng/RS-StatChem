import ij.IJ;
import ij.Menus;
import ij.macro.Interpreter;
import java.lang.invoke.MethodHandles;
import java.util.Hashtable;
import java.util.Properties;

/**
 * Runs an ImageJ 1.x macro in batch mode without ImageJ's main window, for
 * scripts/imagej_golden.py (golden test values from the real ImageJ 1.54p).
 *
 * Plain "java -jar ij.jar -batch" cannot be used with current JDKs: ImageJ's main class needs
 * java.applet (removed in JDK 26), and the first open() or run() builds the AWT menu bar. This
 * driver fills ImageJ's command table (Menus.pluginsTable: command name -> implementing class)
 * from IJ_Props.txt inside ij.jar plus the commands ij/Menus.java adds in code, then runs the
 * macro with Interpreter.batchMode set. A display is still needed (GenericDialog is an AWT
 * Dialog, even when its values come from the macro options), so run it in a desktop session.
 *
 * Usage: java -Dapple.awt.UIElement=true -cp ij.jar BatchImageJ.java macro.ijm [argument]
 */
public class BatchImageJ {
    static final String[][] FIXED = {
        {"Open...", "ij.plugin.Commands(\"open\")"},
        {"Open Next", "ij.plugin.NextImageOpener"},
        {"Cache Sample Images ", "ij.plugin.URLOpener(\"cache\")"},
        {"Image", "ij.plugin.SimpleCommands(\"showdirImage\")"},
        {"Plugins", "ij.plugin.SimpleCommands(\"showdirPlugins\")"},
        {"Macros", "ij.plugin.SimpleCommands(\"showdirMacros\")"},
        {"LUTs", "ij.plugin.SimpleCommands(\"showdirLuts\")"},
        {"ImageJ", "ij.plugin.SimpleCommands(\"showdirImageJ\")"},
        {"temp", "ij.plugin.SimpleCommands(\"showdirTemp\")"},
        {"Home", "ij.plugin.SimpleCommands(\"showdirHome\")"},
        {"Close", "ij.plugin.Commands(\"close\")"},
        {"Close All", "ij.plugin.Commands(\"close-all\")"},
        {"Save", "ij.plugin.Commands(\"save\")"},
        {"Revert", "ij.plugin.Commands(\"revert\")"},
        {"Page Setup...", "ij.plugin.filter.Printer(\"setup\")"},
        {"Print...", "ij.plugin.filter.Printer(\"print\")"},
        {"Undo", "ij.plugin.Commands(\"undo\")"},
        {"Cut", "ij.plugin.Clipboard(\"cut\")"},
        {"Copy", "ij.plugin.Clipboard(\"copy\")"},
        {"Copy to System", "ij.plugin.Clipboard(\"scopy\")"},
        {"Paste", "ij.plugin.Clipboard(\"paste\")"},
        {"Paste Control...", "ij.plugin.frame.PasteController"},
        {"Clear", "ij.plugin.filter.Filler(\"clear\")"},
        {"Clear Outside", "ij.plugin.filter.Filler(\"outside\")"},
        {"Fill", "ij.plugin.filter.Filler(\"fill\")"},
        {"Draw", "ij.plugin.filter.Filler(\"draw\")"},
        {"Invert", "ij.plugin.filter.Filters(\"invert\")"},
        {"8-bit", "ij.plugin.Converter(\"8-bit\")"},
        {"16-bit", "ij.plugin.Converter(\"16-bit\")"},
        {"32-bit", "ij.plugin.Converter(\"32-bit\")"},
        {"8-bit Color", "ij.plugin.Converter(\"8-bit Color\")"},
        {"RGB Color", "ij.plugin.Converter(\"RGB Color\")"},
        {"RGB Stack", "ij.plugin.Converter(\"RGB Stack\")"},
        {"HSB Stack", "ij.plugin.Converter(\"HSB Stack\")"},
        {"HSB (32-bit)", "ij.plugin.Converter(\"HSB (32-bit)\")"},
        {"Lab Stack", "ij.plugin.Converter(\"Lab Stack\")"},
        {"Show Info...", "ij.plugin.ImageInfo"},
        {"Properties...", "ij.plugin.filter.ImageProperties"},
        {"Crop", "ij.plugin.Resizer(\"crop\")"},
        {"Duplicate...", "ij.plugin.Duplicator"},
        {"Rename...", "ij.plugin.SimpleCommands(\"rename\")"},
        {"Scale...", "ij.plugin.Scaler"},
        {"Smooth", "ij.plugin.filter.Filters(\"smooth\")"},
        {"Sharpen", "ij.plugin.filter.Filters(\"sharpen\")"},
        {"Find Edges", "ij.plugin.filter.Filters(\"edge\")"},
        {"Find Maxima...", "ij.plugin.filter.MaximumFinder"},
        {"Enhance Contrast...", "ij.plugin.ContrastEnhancer"},
        {"Image Calculator...", "ij.plugin.ImageCalculator"},
        {"Subtract Background...", "ij.plugin.filter.BackgroundSubtracter"},
        {"Measure", "ij.plugin.filter.Analyzer"},
        {"Analyze Particles...", "ij.plugin.filter.ParticleAnalyzer"},
        {"Summarize", "ij.plugin.filter.Analyzer(\"sum\")"},
        {"Distribution...", "ij.plugin.Distribution"},
        {"Label", "ij.plugin.filter.Filler(\"label\")"},
        {"Clear Results", "ij.plugin.filter.Analyzer(\"clear\")"},
        {"Set Measurements...", "ij.plugin.filter.Analyzer(\"set\")"},
        {"Set Scale...", "ij.plugin.filter.ScaleDialog"},
        {"Calibrate...", "ij.plugin.filter.Calibrator"},
        {"Histogram", "ij.plugin.Histogram"},
        {"Histogram", "ij.plugin.Histogram"},
        {"Plot Profile", "ij.plugin.Profiler(\"plot\")"},
        {"Surface Plot...", "ij.plugin.SurfacePlotter"},
        {"Show All", "ij.plugin.WindowOrganizer(\"show\")"},
        {"Put Behind [tab]", "ij.plugin.Commands(\"tab\")"},
        {"Cascade", "ij.plugin.WindowOrganizer(\"cascade\")"},
        {"Tile", "ij.plugin.WindowOrganizer(\"tile\")"},
        {"ImageJ Website...", "ij.plugin.BrowserLauncher"},
        {"Macro Functions...", "ij.plugin.BrowserLauncher(\"https://wsr.imagej.net/developer/macro/functions.html\")"},
        {"Open as Panel", "ij.plugin.SimpleCommands(\"opencp\")"},
        {"Update ImageJ...", "ij.plugin.ImageJ_Updater"},
        {"Release Notes...", "ij.plugin.BrowserLauncher(\"https://wsr.imagej.net/notes.html\")"},
        {"Refresh Menus", "ij.plugin.ImageJ_Updater(\"menus\")"},
        {"About ImageJ...", "ij.plugin.AboutBox"},
        {"Quit", "ij.plugin.Commands(\"quit\")"},
    };

    /** A macro error (or any message) opens a modal dialog that would block forever: print
     * its text and exit with status 2 instead. */
    static void startDialogWatchdog() {
        Thread t = new Thread(() -> {
            while (true) {
                try { Thread.sleep(500); } catch (InterruptedException e) { return; }
                for (java.awt.Window w : java.awt.Window.getWindows()) {
                    if (w.isShowing() && w instanceof java.awt.Dialog) {
                        StringBuilder sb = new StringBuilder();
                        collectText(w, sb);
                        System.err.println("ImageJ dialog \"" + ((java.awt.Dialog) w).getTitle()
                            + "\":" + sb);
                        System.exit(2);
                    }
                }
            }
        });
        t.setDaemon(true);
        t.start();
    }

    static void collectText(java.awt.Component c, StringBuilder sb) {
        for (String getter : new String[] {"getText", "getLabel"}) {
            try {
                Object text = c.getClass().getMethod(getter).invoke(c);
                if (text != null && !text.toString().isEmpty())
                    sb.append("\n  ").append(text);
                break;
            } catch (ReflectiveOperationException ignored) {
            }
        }
        if (c instanceof ij.gui.MultiLineLabel) {
            try {
                java.lang.reflect.Field f = ij.gui.MultiLineLabel.class.getDeclaredField("lines");
                f.setAccessible(true);
                for (String line : (String[]) f.get(c))
                    sb.append("\n  ").append(line);
            } catch (ReflectiveOperationException ignored) {
            }
        }
        if (c instanceof java.awt.Container)
            for (java.awt.Component child : ((java.awt.Container) c).getComponents())
                collectText(child, sb);
    }

    public static void main(String[] args) throws Throwable {
        Hashtable<String, String> table = new Hashtable<>();
        Properties props = new Properties();
        props.load(IJ.class.getResourceAsStream("/IJ_Props.txt"));
        for (String key : props.stringPropertyNames()) {
            String v = props.getProperty(key);
            int comma = v.lastIndexOf(',');
            if (!v.startsWith("\"") || comma < 2)
                continue;
            String command = v.substring(1, comma - 1);
            if (command.endsWith("]")) {   // "Image...[n]" -> "Image..." (as Menus.addPluginItem)
                int open = command.lastIndexOf('[');
                String sc = open > 0 ? command.substring(open + 1, command.length() - 1) : "";
                if (sc.length() == 1 && Character.isLetterOrDigit(sc.charAt(0)))
                    command = command.substring(0, open);
            }
            table.put(command, v.substring(comma + 1).trim());
        }
        for (String[] pair : FIXED)
            table.put(pair[0], pair[1]);
        // A method handle resolves just this field; Class.getDeclaredField would load the
        // types of all of Menus' fields, including java.applet.Applet.
        MethodHandles.privateLookupIn(Menus.class, MethodHandles.lookup())
            .findStaticSetter(Menus.class, "pluginsTable", Hashtable.class).invoke(table);
        startDialogWatchdog();
        Interpreter.batchMode = true;
        String result = IJ.runMacroFile(args[0], args.length > 1 ? args[1] : null);
        System.exit("[aborted]".equals(result) ? 1 : 0);
    }
}
