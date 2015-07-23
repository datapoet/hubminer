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
import java.util.ArrayList;
import java.util.HashMap;
import util.CommandLineParser;

/**
 * This class implements the methods that generate a LaTeX table based on
 * classification experimental results. Mislabeling is also supported in this
 * script. Noise levels are currently not, this should be included later.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class LatexTableClassificationSummarizer {

    // File that are examined by the script.
    private File parentDir, datasetListFile, classifierFile, outputFile;
    private int k;
    private String mlRate;
    // Dataset and classifier names for inclusion in the table.
    private String[] datasetList;
    private String[] classifiers;
    // Accuracies extracted from the results.
    float[][] accTable = null;
    float[][] stDevTable = null;
    float[] avgAcc = null;

    /**
     * Initialization.
     *
     * @param parentDir Directory that contains all the experiment
     * sub-directories.
     * @param datasetListFile File that contains a list of dataset names.
     * @param classifierFile File that contains a list of classifier names.
     * @param k Integer that is the neighborhood size used in the tests.
     * @param mlRate Mislabeling rate that was used.
     * @param outputFile File that will be used to print the generated LaTeX
     * table to.
     */
    public LatexTableClassificationSummarizer(File parentDir,
            File datasetListFile, File classifierFile, int k, String mlRate,
            File outputFile) {
        this.parentDir = parentDir;
        this.datasetListFile = datasetListFile;
        this.classifierFile = classifierFile;
        this.k = k;
        this.mlRate = mlRate;
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
     * the dataset and classifier names.
     *
     * @throws Exception
     */
    private void parseSpecifications() throws Exception {
        BufferedReader br = null;
        try {
            // First the dataset list.
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
            // Then the classifier names.
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
            accTable = new float[classifiers.length][datasetList.length];
            stDevTable = new float[classifiers.length][datasetList.length];
            avgAcc = new float[classifiers.length];
        } catch (Exception e) {
            throw e;
        } finally {
            if (br != null) {
                br.close();
            }
        }
    }

    /**
     * Fetches the experiment values for all the tested classifiers.
     *
     * @throws Exception
     */
    private void getValues() throws Exception {
        for (int dataIndex = 0; dataIndex < datasetList.length; dataIndex++) {
            String dName = datasetList[dataIndex];
            for (int methodIndex = 0; methodIndex < classifiers.length;
                    methodIndex++) {
                String cName = classifiers[methodIndex];
                File resFile = new File(parentDir, dName + File.separator
                        + "k" + k + File.separator + "ml" + mlRate
                        + File.separator + "noise0.0" + File.separator + cName
                        + File.separator + "avg.txt");
                if (resFile.exists()) {
                    BufferedReader br = getReader(resFile);
                    String line;
                    String[] lineParse;
                    try {
                        line = br.readLine(); // The header.
                        line = br.readLine(); // The values.
                        lineParse = line.split(",");
                        accTable[methodIndex][dataIndex] =
                                Float.parseFloat(lineParse[0]);
                        line = br.readLine();
                        while (line != null && !line.startsWith(
                                "accuracyStDev")) {
                            line = br.readLine();
                        }
                        if (line != null) {
                            line = br.readLine();
                        }
                        // This is not null due to the way the result files are
                        // set up.
                        if (line == null) {
                            throw new Exception("Bad result file format.");
                        }
                        lineParse = line.split(",");
                        stDevTable[methodIndex][dataIndex] =
                                Float.parseFloat(lineParse[0]);
                    } catch (Exception e) {
                        System.err.println(e.getMessage());
                        throw e;
                    } finally {
                        if (br != null) {
                            br.close();
                        }
                    }
                }
                avgAcc[methodIndex] += (accTable[methodIndex][dataIndex]
                        / (float) (datasetList.length));
            }
        }
    }

    /**
     * This method prints the LaTeX table.
     *
     * @throws Exception
     */
    private void outputSummaries() throws Exception {
        FileUtil.createFile(outputFile);
        PrintWriter pw = new PrintWriter(new FileWriter(outputFile));
        try {
            pw.println();
            pw.println("---------------------------------------");
            pw.println("classifiers");
            pw.println("---------------------------------------");
            pw.println();
            pw.print("Data set");
            pw.print(" & \\multicolumn{3}{c}{");
            pw.print(classifiers[0]);
            pw.print("}");
            for (int cIndex = 1; cIndex < classifiers.length; cIndex++) {
                pw.print(" & \\multicolumn{4}{c}{");
                pw.print(classifiers[cIndex]);
                pw.print("}");
            }
            pw.println("\\\\");
            for (int dIndex = 0; dIndex < datasetList.length; dIndex++) {
                // Find the best entries.
                ArrayList<Integer> maxEntries =
                        new ArrayList<>(classifiers.length);
                float maxAcc = 0;
                for (int cIndex = 0; cIndex < classifiers.length; cIndex++) {
                    if ((1000 * accTable[cIndex][dIndex]) % 1000
                            > (1000 * maxAcc) % 1000) {
                        maxAcc = accTable[cIndex][dIndex];
                        maxEntries.clear();
                        maxEntries.add(cIndex);
                    } else if ((1000 * accTable[cIndex][dIndex]) % 1000
                            == (1000 * maxAcc) % 1000) {
                        maxEntries.add(cIndex);
                    }
                }
                HashMap<Integer, Float> maxEntryHash =
                        new HashMap<>(classifiers.length);
                for (int maxIndex : maxEntries) {
                    maxEntryHash.put(maxIndex, maxAcc);
                }
                pw.print(datasetList[dIndex] + " & ");
                if (DataMineConstants.isNonZero(accTable[0][dIndex])) {
                    int num = (int) (accTable[0][dIndex] * 1000);
                    int decim = num % 10;
                    num = num / 10;
                    int ones = num % 10;
                    num = num / 10;
                    int tens = num % 10;
                    num = num / 10;
                    int hundreds = num % 10;
                    if (maxEntryHash.containsKey(0)) {
                        pw.print("\\textbf{");
                    }
                    if (hundreds > 0) {
                        pw.print(Integer.toString(hundreds)
                                + Integer.toString(tens)
                                + Integer.toString(ones) + "."
                                + Integer.toString(decim));
                    } else if (tens > 0) {
                        pw.print(Integer.toString(tens)
                                + Integer.toString(ones) + "."
                                + Integer.toString(decim));
                    } else {
                        pw.print(Integer.toString(ones) + "."
                                + Integer.toString(decim));
                    }
                    if (maxEntryHash.containsKey(0)) {
                        pw.print("}");
                    }
                    pw.print("& $\\pm$ &");
                    if (maxEntryHash.containsKey(0)) {
                        pw.print("\\textbf{");
                    }
                    num = (int) (stDevTable[0][dIndex] * 1000);
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
                    if (maxEntryHash.containsKey(0)) {
                        pw.print("}");
                    }
                    pw.print(" & ");

                }
                for (int cIndex = 1; cIndex < classifiers.length;
                        cIndex++) {
                    if (DataMineConstants.isNonZero(accTable[cIndex][
                                dIndex])) {
                        int num = (int) (accTable[cIndex][dIndex] * 1000);
                        int decim = num % 10;
                        num = num / 10;
                        int ones = num % 10;
                        num = num / 10;
                        int tens = num % 10;
                        num = num / 10;
                        int hundreds = num % 10;
                        if (maxEntryHash.containsKey(cIndex)) {
                            pw.print("\\textbf{");
                        }
                        if (hundreds > 0) {
                            pw.print(Integer.toString(hundreds)
                                    + Integer.toString(tens)
                                    + Integer.toString(ones) + "."
                                    + Integer.toString(decim));
                        } else if (tens > 0) {
                            pw.print(Integer.toString(tens)
                                    + Integer.toString(ones) + "."
                                    + Integer.toString(decim));
                        } else {
                            pw.print(Integer.toString(ones) + "."
                                    + Integer.toString(decim));
                        }
                        if (maxEntryHash.containsKey(cIndex)) {
                            pw.print("}");
                        }
                        pw.print("& $\\pm$ &");
                        num = (int) (stDevTable[cIndex][dIndex] * 1000);
                        decim = num % 10;
                        num = num / 10;
                        ones = num % 10;
                        num = num / 10;
                        tens = num % 10;
                        if (maxEntryHash.containsKey(cIndex)) {
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
                        if (maxEntryHash.containsKey(cIndex)) {
                            pw.print("}");
                        }
                        if (cIndex < classifiers.length - 1) {
                            // The space for $\bullet$, if needed.
                            pw.print(" & & ");
                        } else {
                            pw.print(" & ");
                        }
                    }
                }
                pw.println("\\\\");
            }
            pw.print("AVG & ");
            int num = (int) (avgAcc[0] * 1000);
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
            pw.print("& & &");
            for (int cIndex = 1; cIndex < classifiers.length; cIndex++) {
                num = (int) (avgAcc[cIndex] * 1000);
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
                if (cIndex < classifiers.length - 1) {
                    pw.print("& & & &");
                } else {
                    pw.print("& & &");
                }
            }
            pw.println("\\\\");
        } catch (Exception e) {
            throw e;
        } finally {
            pw.close();
        }
    }

    /**
     * This script creates a LaTeX table based on mislabeling and classification
     * experiment results.
     *
     * @param args Command line parameters, as specified.
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        CommandLineParser clp = new CommandLineParser(true);
        clp.addParam("-parentDir", "Parent directory for the experiments.",
                CommandLineParser.STRING, true, false);
        clp.addParam("-datasetList", "File containing a list of comma-separated"
                + "datasets in one line, that will define the order in which "
                + "they will appear in the table.",
                CommandLineParser.STRING, true, false);
        clp.addParam("-classifiers", "File containing a list of comma-separated"
                + "classification methods in one line, that will define the "
                + "order in which they will appear in the table.",
                CommandLineParser.STRING, true, false);
        clp.addParam("-outFile", "Output file.",
                CommandLineParser.STRING, true, false);
        clp.addParam("-mlRate", "Mislabeling rate, as a string - in the way it "
                + "appears in the generated results.",
                CommandLineParser.STRING, true, false);
        clp.addParam("-k", "Neighborhood size.",
                CommandLineParser.INTEGER, true, false);
        clp.parseLine(args);
        File parentDir = new File((String) clp.getParamValues("-parentDir").
                get(0));
        File classificationMethodsFile = new File((String) clp.getParamValues(
                "-classifiers").get(0));
        File datasetsFile = new File((String) clp.getParamValues(
                "-datasetList").get(0));
        File outFile = new File((String) clp.getParamValues("-outFile").get(0));
        int k = (Integer) clp.getParamValues("-k").get(0);
        String mlString = (String) clp.getParamValues("-mlRate").get(0);
        LatexTableClassificationSummarizer summarizer;
        summarizer = new LatexTableClassificationSummarizer(parentDir,
                datasetsFile, classificationMethodsFile, k, mlString, outFile);
        summarizer.parseSpecifications();
        summarizer.getValues();
        summarizer.outputSummaries();
    }
}