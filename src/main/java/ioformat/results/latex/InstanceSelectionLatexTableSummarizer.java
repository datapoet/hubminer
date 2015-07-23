/**
* Hub Miner: a hubness-aware machine learning experimentation library.
* Copyright (C) 2014  Nenad Tomasev. Email: nenad.tomasev at gmail.com
* 
* This program is free software: you can redistribute it and/or modify it under
* the terms of the GNU General Public License as published by the Free Software
* Foundation, either version 3 of the License, or (at your option) any later
* version.
* 
* This program is distributed in the hope that it will be useful, but WITHOUT
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
* FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with
* this program. If not, see <http://www.gnu.org/licenses/>.
*/
package ioformat.results.latex;

import data.representation.util.DataMineConstants;
import ioformat.FileUtil;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import util.CommandLineParser;

/**
 * This class implements a LaTeX table generator for instance selection
 * experiments. There is a biased and non-biased option contain within, which
 * has to do with whether biased or non-biased hubness-aware learning was done
 * on top of instance selections. For more details, see the appropriate papers.
 * This script (as well as other LaTeX generating scripts) were written with
 * very specific result visualizations in mind and can/should be extended and
 * modified to cover other cases that other people might encounted while using
 * this library.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class InstanceSelectionLatexTableSummarizer {

    // Various files that will be examined.
    private File parentDir, selectionMethodsFile, datasetListFile,
            classifierFile, outputFile;
    // Neighborhood size.
    private int k;
    // Arrays of names of instance selection methods, datasets and classifiers.
    private String[] selectionMethods;
    private String[] datasetList;
    private String[] classifiers;
    // Accuracies and standard deviations, for the biased and non-biased case.
    private float[][][] accTableBiased = null;
    private float[][][] accTableUnbiased = null;
    private float[][][] stDevTableBiased = null;
    private float[][][] stDevTableUnbiased = null;
    private float[][] avgAccBiased = null;
    private float[][] avgAccUnbiased = null;
    private boolean[][][] boldUnbiased = null;

    /**
     * Initialization
     *
     * @param parentDir Directory that contains all the experiment
     * sub-directories.
     * @param selectionMethodsFile File that contains a list of the employed
     * instance selection methods.
     * @param datasetListFile File that contains a list of dataset names.
     * @param classifierFile File that contains a list of classifier names.
     * @param k Integer that is the neighborhood size used in the tests.
     * @param outputFile File that will be used to print the generated LaTeX
     * table to.
     */
    public InstanceSelectionLatexTableSummarizer(File parentDir,
            File selectionMethodsFile,
            File datasetListFile, File classifierFile, int k, File outputFile) {
        this.parentDir = parentDir;
        this.selectionMethodsFile = selectionMethodsFile;
        this.datasetListFile = datasetListFile;
        this.classifierFile = classifierFile;
        this.k = k;
        this.outputFile = outputFile;
    }

    /**
     * Just a handle to quickly obtain the reader.
     *
     * @param inFile File that is to be read.
     * @return Reader for the specified file.
     * @throws Exception
     */
    private static BufferedReader getReader(File inFile) throws Exception {
        BufferedReader br = new BufferedReader(new InputStreamReader(
                new FileInputStream(inFile)));
        return br;
    }

    /**
     * Parse all the configuration and specification files, in order to extract
     * selection method, dataset and classifier names.
     *
     * @throws Exception
     */
    private void parseSpecifications() throws Exception {
        BufferedReader br = null;
        try {
            br = getReader(selectionMethodsFile);
            selectionMethods = br.readLine().split(",");
            for (int sIndex = 0; sIndex < selectionMethods.length; sIndex++) {
                selectionMethods[sIndex] = selectionMethods[sIndex].trim();
            }
            System.out.println(selectionMethods.length
                    + " selection methods: ");
            for (String s : selectionMethods) {
                System.out.print(s + " ");
            }
            System.out.println();
            br.close();
            br = getReader(datasetListFile);
            datasetList = br.readLine().split(",");
            for (int dIndex = 0; dIndex < datasetList.length; dIndex++) {
                datasetList[dIndex] = datasetList[dIndex].trim();
            }
            System.out.println(datasetList.length + " datasets: ");
            for (String s : datasetList) {
                System.out.print(s + " ");
            }
            System.out.println();
            br.close();
            br = getReader(classifierFile);
            classifiers = br.readLine().split(",");
            for (int cIndex = 0; cIndex < classifiers.length; cIndex++) {
                classifiers[cIndex] = classifiers[cIndex].trim();
            }
            System.out.println(classifiers.length + " classifiers: ");
            for (String s : classifiers) {
                System.out.print(s + " ");
            }
            System.out.println();
            br.close();
            accTableBiased = new float[classifiers.length][datasetList.length][
                    selectionMethods.length];
            accTableUnbiased = new float[classifiers.length][
                    datasetList.length][selectionMethods.length];
            boldUnbiased = new boolean[classifiers.length][datasetList.length][
                    selectionMethods.length];
            stDevTableBiased = new float[classifiers.length][
                    datasetList.length][selectionMethods.length];
            stDevTableUnbiased = new float[classifiers.length][
                    datasetList.length][selectionMethods.length];
            avgAccBiased = new float[classifiers.length][
                    selectionMethods.length];
            avgAccUnbiased = new float[classifiers.length][
                    selectionMethods.length];
        } catch (Exception e) {
            throw e;
        } finally {
            if (br != null) {
                br.close();
            }
        }
    }

    /**
     * Fetches the experiment values for the biased and non-biased case.
     *
     * @throws Exception
     */
    private void getValues() throws Exception {
        File[] selMetDirs = parentDir.listFiles();
        for (int i = 0; i < selectionMethods.length; i++) {
            String method = selectionMethods[i];
            // Directories for the biased and non-biased case end in different
            // ways and this is used here. This is idiosyncratic to how the
            // experiments were performed and would not hold in the general 
            // case.
            for (File f : selMetDirs) {
                if (f.getName().toLowerCase().startsWith(method.toLowerCase())
                        && f.getName().toLowerCase().endsWith("red")) {
                    getValues(i, f, accTableBiased, stDevTableBiased,
                            avgAccBiased);
                }
                if (f.getName().toLowerCase().startsWith(method.toLowerCase())
                        && f.getName().toLowerCase().endsWith("all")) {
                    getValues(i, f, accTableUnbiased, stDevTableUnbiased,
                            avgAccUnbiased);
                }
            }
        }
    }

    /**
     * Fetches the values from a directory of results into the specified result
     * tables for a particular selection method.
     *
     * @param selMetIndex Index of the selection method that is currently being
     * traversed.
     * @param resultDirectory Directory of the results.
     * @param accTable float[][][] that is the accuracy table to use for storing
     * the parsed values.
     * @param stDevTable float[][][] that is the target table of accuracy
     * standard deviations.
     * @param avgAcc float[][] that is the target table of average accuracies.
     * @throws Exception
     */
    private void getValues(int selMetIndex, File resultDirectory,
            float[][][] accTable, float[][][] stDevTable, float[][] avgAcc)
            throws Exception {
        if (!resultDirectory.getName().toLowerCase().contains("test")
                && resultDirectory.listFiles() != null
                && resultDirectory.listFiles().length > 0) {
            File[] tsDirs = resultDirectory.listFiles();
            for (File f : tsDirs) {
                if (f.getName().toLowerCase().
                        contains("test")) {
                    getValues(selMetIndex, f, accTable, stDevTable, avgAcc);
                    break;
                }
            }
        } else {
            // Now we are in the test result directory for a given selection
            // method.
            for (int dataIndex = 0; dataIndex < datasetList.length;
                    dataIndex++) {
                String dName = datasetList[dataIndex];
                for (int methodIndex = 0; methodIndex < classifiers.length;
                        methodIndex++) {
                    String cName = classifiers[methodIndex];
                    File resFile = new File(resultDirectory, dName
                            + File.separator + "k" + k + File.separator
                            + "ml0.0" + File.separator + "noise0.0"
                            + File.separator + cName + File.separator
                            + "avg.txt");
                    if (resFile.exists()) {
                        BufferedReader br = getReader(resFile);
                        String line;
                        String[] lineItems;
                        try {
                            line = br.readLine(); // The header.
                            line = br.readLine(); // The values.
                            lineItems = line.split(",");
                            accTable[methodIndex][dataIndex][selMetIndex] =
                                    Float.parseFloat(lineItems[0]);
                            line = br.readLine();
                            while (line != null && !line.startsWith(
                                    "accuracyStDev")) {
                                line = br.readLine();
                            }
                            if (line != null) {
                                line = br.readLine();
                            }
                            // This is not null due to the way the result files
                            // are set up.
                            if (line == null) {
                                throw new Exception("Bad result file format.");
                            }
                            lineItems = line.split(",");
                            stDevTable[methodIndex][dataIndex][selMetIndex] =
                                    Float.parseFloat(lineItems[0]);
                        } catch (Exception e) {
                            System.err.println(e.getMessage());
                            throw e;
                        } finally {
                            if (br != null) {
                                br.close();
                            }
                        }
                    }
                    avgAcc[methodIndex][selMetIndex] +=
                            (accTable[methodIndex][dataIndex][selMetIndex]
                            / (float) (datasetList.length));
                }
            }
        }
    }

    /**
     * This method compares the biased and non-biased accuracy tables and
     * determines what is to be shown in bold in the LaTeX table.
     */
    private void compareTables() {
        for (int i = 0; i < classifiers.length; i++) {
            for (int j = 0; j < datasetList.length; j++) {
                for (int p = 0; p < selectionMethods.length; p++) {
                    if (accTableBiased[i][j][p] < accTableUnbiased[i][j][p]) {
                        boldUnbiased[i][j][p] = true;
                    }
                }
            }
        }
    }

    /**
     * This method prints the tables themselves.
     *
     * @throws Exception
     */
    private void outputSummaries() throws Exception {
        FileUtil.createFile(outputFile);
        PrintWriter pw = new PrintWriter(new FileWriter(outputFile));
        try {
            for (int cIndex = 0; cIndex < classifiers.length; cIndex++) {
                pw.println();
                pw.println("---------------------------------------");
                pw.println(classifiers[cIndex]);
                pw.println("BIASED");
                pw.println("---------------------------------------");
                pw.println();
                pw.print("Data set & \\multicolumn{3}{c}{None}");
                for (int sIndex = 0; sIndex < selectionMethods.length;
                        sIndex++) {
                    pw.print(" & \\multicolumn{4}{c}{");
                    pw.print(selectionMethods[sIndex]);
                    pw.println("}");
                }
                for (int dIndex = 0; dIndex < datasetList.length; dIndex++) {
                    // The first place is for the none selection entry which is
                    // manually inserted later.
                    pw.print(datasetList[dIndex] + " & & $\\pm$ & & ");
                    for (int sIndex = 0; sIndex < selectionMethods.length;
                            sIndex++) {
                        if (DataMineConstants.isNonZero(accTableBiased[
                                cIndex][dIndex][sIndex])) {
                            int num = (int) (accTableBiased[cIndex][dIndex][
                                    sIndex] * 1000);
                            int decim = num % 10;
                            num = num / 10;
                            int ones = num % 10;
                            num = num / 10;
                            int tens = num % 10;
                            if (tens > 0) {
                                pw.print(Integer.toString(tens)
                                        + Integer.toString(ones) + "."
                                        + Integer.toString(decim));
                            } else {
                                pw.print(Integer.toString(ones) + "."
                                        + Integer.toString(decim));
                            }
                            pw.print("& $\\pm$ &");
                            num = (int) (stDevTableBiased[cIndex][dIndex][
                                    sIndex] * 1000);
                            decim = num % 10;
                            num = num / 10;
                            ones = num % 10;
                            num = num / 10;
                            tens = num % 10;
                            if (tens > 0) {
                                pw.print(Integer.toString(tens)
                                        + Integer.toString(ones) + "."
                                        + Integer.toString(decim));
                            } else {
                                pw.print(Integer.toString(ones) + "."
                                        + Integer.toString(decim));
                            }
                            if (sIndex < selectionMethods.length - 1) {
                                // The space for $\bullet$, if needed.
                                pw.print(" & & "); //
                            } else {
                                pw.print(" & ");
                            }
                        } else {
                            if (sIndex < selectionMethods.length - 1) {
                                pw.print(" & $\\pm$ &  & & ");
                            } else {
                                pw.print(" & $\\pm$ &  & ");
                            }
                        }
                    }
                    pw.println("\\\\");
                }
                pw.print("AVG & & & &");
                for (int sIndex = 0; sIndex < selectionMethods.length;
                        sIndex++) {
                    int num = (int) (avgAccBiased[cIndex][sIndex] * 1000);
                    int decim = num % 10;
                    num = num / 10;
                    int ones = num % 10;
                    num = num / 10;
                    int tens = num % 10;
                    if (tens > 0) {
                        pw.print(Integer.toString(tens)
                                + Integer.toString(ones) + "."
                                + Integer.toString(decim));
                    } else {
                        pw.print(Integer.toString(ones) + "."
                                + Integer.toString(decim));
                    }
                    if (sIndex < selectionMethods.length - 1) {
                        pw.print("& & & &");
                    } else {
                        pw.print("& & &");
                    }
                }
                pw.println("\\\\");
                pw.println();
                pw.println("---------------------------------------");
                pw.println(classifiers[cIndex]);
                pw.println("UNBIASED");
                pw.println("---------------------------------------");
                pw.println();
                pw.print("Data set & \\multicolumn{3}{c}{None}");
                for (int sIndex = 0; sIndex < selectionMethods.length;
                        sIndex++) {
                    pw.print(" & \\multicolumn{4}{c}{");
                    pw.print(
                            selectionMethods[sIndex]);
                    pw.println("}");
                }
                for (int dIndex = 0; dIndex < datasetList.length; dIndex++) {
                    // The first place is for the none selection entry which is
                    // manually inserted later.
                    pw.print(datasetList[dIndex] + " & & $\\pm$ & & ");
                    for (int sIndex = 0; sIndex < selectionMethods.length;
                            sIndex++) {
                        if (DataMineConstants.isNonZero(accTableUnbiased[
                                cIndex][dIndex][sIndex])) {
                            int num = (int) (accTableUnbiased[cIndex][dIndex][
                                    sIndex] * 1000);
                            int decim = num % 10;
                            num = num / 10;
                            int ones = num % 10;
                            num = num / 10;
                            int tens = num % 10;
                            if (boldUnbiased[cIndex][dIndex][sIndex]) {
                                pw.print("\\textbf{");
                            }
                            if (tens > 0) {
                                pw.print(Integer.toString(tens)
                                        + Integer.toString(ones) + "."
                                        + Integer.toString(decim));
                            } else {
                                pw.print(Integer.toString(ones) + "."
                                        + Integer.toString(decim));
                            }
                            if (boldUnbiased[cIndex][dIndex][sIndex]) {
                                pw.print("}");
                            }
                            pw.print("& $\\pm$ &");
                            num = (int) (stDevTableUnbiased[cIndex][dIndex][
                                    sIndex] * 1000);
                            decim = num % 10;
                            num = num / 10;
                            ones = num % 10;
                            num = num / 10;
                            tens = num % 10;
                            if (boldUnbiased[cIndex][dIndex][sIndex]) {
                                pw.print("\\textbf{");
                            }
                            if (tens > 0) {
                                pw.print(Integer.toString(tens)
                                        + Integer.toString(ones) + "."
                                        + Integer.toString(decim));
                            } else {
                                pw.print(Integer.toString(ones) + "."
                                        + Integer.toString(decim));
                            }
                            if (boldUnbiased[cIndex][dIndex][sIndex]) {
                                pw.print("}");
                            }
                            if (sIndex < selectionMethods.length - 1) {
                                // The space for $\bullet$, if needed.
                                pw.print(" & & ");
                            } else {
                                pw.print(" & ");
                            }
                        } else {
                            if (sIndex < selectionMethods.length - 1) {
                                pw.print(" & $\\pm$ &  & & ");
                            } else {
                                pw.print(" & $\\pm$ &  & ");
                            }
                        }
                    }
                    pw.println("\\\\");
                }
                pw.print("AVG & & & &");
                for (int sIndex = 0; sIndex < selectionMethods.length;
                        sIndex++) {
                    if (avgAccUnbiased[cIndex][sIndex] > avgAccBiased[cIndex][
                            sIndex]) {
                        pw.print("\\textbf{");
                    }

                    int num = (int) (avgAccUnbiased[cIndex][sIndex] * 1000);
                    int decim = num % 10;
                    num = num / 10;
                    int ones = num % 10;
                    num = num / 10;
                    int tens = num % 10;
                    if (tens > 0) {
                        pw.print(Integer.toString(tens)
                                + Integer.toString(ones) + "."
                                + Integer.toString(decim));
                    } else {
                        pw.print(Integer.toString(ones) + "."
                                + Integer.toString(decim));
                    }
                    if (avgAccUnbiased[cIndex][sIndex] > avgAccBiased[cIndex][
                            sIndex]) {
                        pw.print("}");
                    }
                    if (sIndex < selectionMethods.length - 1) {
                        pw.print("& & & &");
                    } else {
                        pw.print("& & &");
                    }
                }
                pw.println("\\\\");
            }
        } catch (Exception e) {
            throw e;
        } finally {
            pw.close();
        }
    }

    /**
     * This script creates a LaTeX table based on instance selection experiment
     * results.
     *
     * @param args Command line parameters, as specified.
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        CommandLineParser clp = new CommandLineParser(true);
        clp.addParam("-parentDir", "Parent directory for the experiments.",
                CommandLineParser.STRING, true, false);
        clp.addParam("-selMethods", "File containing a list of comma-separated"
                + "instance selection methods in one line, that will define"
                + "the order in which they will appear in the table.",
                CommandLineParser.STRING, true, false);
        clp.addParam("-datasetList", "File containing a list of comma-separated"
                + "datasets in one line, that will define the order in which "
                + "they will appear in the table.",
                CommandLineParser.STRING, true, false);
        clp.addParam("-classifiers", "File containing a list of comma-separated"
                + " classification methods in one line, that will define the "
                + "order in which they will appear.",
                CommandLineParser.STRING, true, false);
        clp.addParam("-outFile", "Output file.",
                CommandLineParser.STRING, true, false);
        clp.addParam("-k", "Neighborhood size.",
                CommandLineParser.INTEGER, true, false);
        clp.parseLine(args);
        File parentDir = new File((String) clp.getParamValues("-parentDir").
                get(0));
        File selMethodsFile = new File((String) clp.getParamValues(
                "-selMethods").get(0));
        File classificationMethodsFile = new File((String) clp.getParamValues(
                "-classifiers").get(0));
        File datasetsFile = new File((String) clp.getParamValues(
                "-datasetList").get(0));
        File outFile = new File((String) clp.getParamValues("-outFile").get(0));
        int k = (Integer) clp.getParamValues("-k").get(0);
        InstanceSelectionLatexTableSummarizer summarizer =
                new InstanceSelectionLatexTableSummarizer(parentDir,
                selMethodsFile,
                datasetsFile, classificationMethodsFile, k, outFile);
        summarizer.parseSpecifications();
        summarizer.getValues();
        summarizer.compareTables();
        summarizer.outputSummaries();
    }
}