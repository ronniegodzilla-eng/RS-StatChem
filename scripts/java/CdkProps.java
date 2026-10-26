/*
 * RS-StatChem helper: logP and molar refractivity from the Chemistry Development Kit (CDK,
 * https://cdk.github.io, LGPL-2.1). It runs as one long-lived process next to the app
 * (rsstatchem/core/chem_draw/cdk.py) so the JVM starts only once.
 *
 * Protocol (UTF-8, one line per request / answer):
 *   start-up  -> {"ready":true,"cdk":"2.13"}
 *   "<id>\t<SMILES>"  -> {"id":"<id>","jplogp":..,"xlogp":..,"alogp":..,"amr":..,"error":".."}
 * Values that cannot be computed are null; "error" is present only when something failed.
 *
 * Methods (as implemented by CDK):
 *   JPlogP  - J. Plante, S. Werner, J. Cheminform. 10 (2018) 61
 *   XLogP   - R. Wang, Y. Fu, L. Lai, J. Chem. Inf. Comput. Sci. 37 (1997) 615 (v2.0)
 *   ALogP / AMR - A. K. Ghose, V. N. Viswanadhan, J. J. Wendoloski, J. Phys. Chem. A 102
 *             (1998) 3762 (atom-type contributions to logP and molar refractivity)
 *
 * Build (scripts/fetch_cdk.py does this):
 *   javac --release 11 -cp cdk.jar -d build scripts/java/CdkProps.java
 *   jar --create --file rsstatchem-cdk.jar -C build .
 */

import java.io.BufferedReader;
import java.io.FileDescriptor;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.Locale;

import org.openscience.cdk.CDK;
import org.openscience.cdk.aromaticity.Aromaticity;
import org.openscience.cdk.interfaces.IAtomContainer;
import org.openscience.cdk.interfaces.IChemObjectBuilder;
import org.openscience.cdk.qsar.DescriptorValue;
import org.openscience.cdk.qsar.IMolecularDescriptor;
import org.openscience.cdk.qsar.result.DoubleArrayResult;
import org.openscience.cdk.qsar.result.DoubleResult;
import org.openscience.cdk.qsar.result.IDescriptorResult;
import org.openscience.cdk.qsar.descriptors.molecular.ALOGPDescriptor;
import org.openscience.cdk.qsar.descriptors.molecular.JPlogPDescriptor;
import org.openscience.cdk.qsar.descriptors.molecular.XLogPDescriptor;
import org.openscience.cdk.silent.SilentChemObjectBuilder;
import org.openscience.cdk.smiles.SmilesParser;
import org.openscience.cdk.tools.CDKHydrogenAdder;
import org.openscience.cdk.tools.manipulator.AtomContainerManipulator;

public final class CdkProps {

    private final IChemObjectBuilder builder = SilentChemObjectBuilder.getInstance();
    private final SmilesParser parser = new SmilesParser(builder);
    private final JPlogPDescriptor jplogp = new JPlogPDescriptor();
    private final XLogPDescriptor xlogp = new XLogPDescriptor();
    private final ALOGPDescriptor alogp;

    private CdkProps() throws Exception {
        alogp = new ALOGPDescriptor();
        for (IMolecularDescriptor d : new IMolecularDescriptor[]{jplogp, xlogp, alogp}) {
            d.initialise(builder);
        }
        // XLogP: perceive aromaticity, apply the salicylic-acid correction (Wang 1997)
        xlogp.setParameters(new Object[]{Boolean.TRUE, Boolean.TRUE});
    }

    private static String num(double v) {
        if (Double.isNaN(v) || Double.isInfinite(v)) {
            return "null";
        }
        return String.format(Locale.ROOT, "%.4f", v);
    }

    private static String esc(String s) {
        StringBuilder b = new StringBuilder();
        for (char c : s.toCharArray()) {
            if (c == '"' || c == '\\') {
                b.append('\\').append(c);
            } else if (c < 0x20) {
                b.append(' ');
            } else {
                b.append(c);
            }
        }
        return b.toString();
    }

    private static double scalar(DescriptorValue v) {
        if (v == null || v.getException() != null) {
            return Double.NaN;
        }
        IDescriptorResult r = v.getValue();
        return r instanceof DoubleResult ? ((DoubleResult) r).doubleValue() : Double.NaN;
    }

    /** ALogP needs explicit hydrogens and perceived aromaticity (CDK documentation). */
    private static IAtomContainer prepared(IAtomContainer mol) throws Exception {
        IAtomContainer m = mol.clone();
        AtomContainerManipulator.percieveAtomTypesAndConfigureAtoms(m);
        CDKHydrogenAdder.getInstance(m.getBuilder()).addImplicitHydrogens(m);
        AtomContainerManipulator.convertImplicitToExplicitHydrogens(m);
        Aromaticity.cdkLegacy().apply(m);
        return m;
    }

    String answer(String id, String smiles) {
        StringBuilder out = new StringBuilder("{\"id\":\"").append(esc(id)).append('"');
        StringBuilder err = new StringBuilder();
        IAtomContainer mol;
        try {
            mol = parser.parseSmiles(smiles);
        } catch (Exception e) {
            return out.append(",\"jplogp\":null,\"xlogp\":null,\"alogp\":null,\"amr\":null,")
                      .append("\"error\":\"SMILES: ").append(esc(String.valueOf(e.getMessage())))
                      .append("\"}").toString();
        }
        double jp = Double.NaN, xl = Double.NaN, al = Double.NaN, mr = Double.NaN;
        try {
            jp = scalar(jplogp.calculate(mol));
        } catch (Exception | StackOverflowError e) {
            err.append("JPlogP: ").append(e.getMessage()).append("; ");
        }
        try {
            xl = scalar(xlogp.calculate(mol));
        } catch (Exception | StackOverflowError e) {
            err.append("XLogP: ").append(e.getMessage()).append("; ");
        }
        try {
            DescriptorValue v = alogp.calculate(prepared(mol));
            if (v.getException() == null && v.getValue() instanceof DoubleArrayResult) {
                DoubleArrayResult r = (DoubleArrayResult) v.getValue();
                al = r.get(0);
                mr = r.get(2);
            } else if (v.getException() != null) {
                err.append("ALogP: ").append(v.getException().getMessage()).append("; ");
            }
        } catch (Exception | StackOverflowError e) {
            err.append("ALogP: ").append(e.getMessage()).append("; ");
        }
        out.append(",\"jplogp\":").append(num(jp)).append(",\"xlogp\":").append(num(xl))
           .append(",\"alogp\":").append(num(al)).append(",\"amr\":").append(num(mr));
        if (err.length() > 0) {
            out.append(",\"error\":\"").append(esc(err.toString().trim())).append('"');
        }
        return out.append('}').toString();
    }

    public static void main(String[] args) throws Exception {
        PrintStream out = new PrintStream(new FileOutputStream(FileDescriptor.out), true, "UTF-8");
        System.setOut(System.err);           // library messages must not mix with the answers
        CdkProps props = new CdkProps();
        out.println("{\"ready\":true,\"cdk\":\"" + esc(String.valueOf(CDK.getVersion())) + "\"}");
        BufferedReader in = new BufferedReader(new InputStreamReader(System.in,
                                                                     StandardCharsets.UTF_8));
        String line;
        while ((line = in.readLine()) != null) {
            line = line.trim();
            if (line.isEmpty()) {
                continue;
            }
            if (line.equals("quit")) {
                break;
            }
            int tab = line.indexOf('\t');
            String id = tab < 0 ? "" : line.substring(0, tab);
            String smiles = tab < 0 ? line : line.substring(tab + 1);
            String reply;
            try {
                reply = props.answer(id, smiles);
            } catch (Throwable t) {          // never let one molecule end the process
                reply = "{\"id\":\"" + esc(id) + "\",\"jplogp\":null,\"xlogp\":null,"
                        + "\"alogp\":null,\"amr\":null,\"error\":\"" + esc(String.valueOf(t))
                        + "\"}";
            }
            out.println(reply);
        }
    }
}
