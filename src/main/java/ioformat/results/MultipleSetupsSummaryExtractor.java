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
package ioformat.results;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.ArrayList;
import util.BasicMathUtil;
import util.CommandLineParser;

/**
 * If there were multiple experimental setups in different directories, this
 * utility script extracts the best score for each algorithm and each dataset
 * across multiple setups and also outputs the best k value and the setup for
 * which it was reached.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class MultipleSetupsSummaryExtractor {

    // The internal data structures.
    private HashMap<String, Integer> algHash = null;
    private HashMap<String, Integer> datasetHash = null;
    private ArrayList<String> algNames = null;
    private ArrayList<String> dsNames = null;
    private ArrayList<String> experimentalSetupPaths = null;
    private float[][][] setupAlgDSAccuracy;
    private int[][][] setupAlgDSBestK;

    /**
     * This method writes the summaries to a specified output file.
     *
     * @param outFile File to write the summaries to.
     * @throws Exception
     */
    public void writeSummaries(File outFile) throws Exception {
        PrintWriter pw = new PrintWriter(new FileWriter(outFile));
        try {
            for (int dataIndex = 0; dataIndex < dsNames.size(); dataIndex++) {
                pw.println(dsNames.get(dataIndex));
                for (int algIndex = 0; algIndex < algNames.size(); algIndex++) {
                    pw.print(algNames.get(algIndex) + " acc:");
                    float maxAcc = 0;
                    int maxT = 0;
                    for (int t = 0; t < experimentalSetupPaths.size(); t++) {
                        if (setupAlgDSAccuracy[t][algIndex][dataIndex]
                                > maxAcc) {
                            maxAcc = setupAlgDSAccuracy[t][algIndex][dataIndex];
                            maxT = t;
                        }
                    }
                    pw.print(BasicMathUtil.makeADecimalCutOff(maxAcc, 3));
                    pw.print(" maxK:");
                    pw.print(setupAlgDSBestK[maxT][algIndex][dataIndex]);
                    pw.print(" maxT:");
                    pw.print(maxT);
                    pw.println();
                }
            }
        } catch (Exception e) {
            System.err.println(e.getMessage());
        } finally {
            pw.close();
        }
    }

    /**
     * Traverse the test results and read in all the accuracies.
     *
     * @throws Exception
     */
    public void getAllAccuracies() throws Exception {
        setupAlgDSAccuracy = new float[experimentalSetupPaths.size()][
                algNames.size()][dsNames.size()];
        setupAlgDSBestK = new int[experimentalSetupPaths.size()][
                algNames.size()][dsNames.size()];
        BufferedReader br;
        for (int setupIndex = 0; setupIndex < experimentalSetupPaths.size();
                setupIndex++) {
            for (int dataIndex = 0; dataIndex < dsNames.size(); dataIndex++) {
                br = null;
                try {
                    br = new BufferedReader(new InputStreamReader(
                            new FileInputStream(new File(
                            experimentalSetupPaths.get(setupIndex),
                            dsNames.get(dataIndex) + File.separator
                            + "totalSummary.txt"))));
                    String line = br.readLine();
                    String[] pair;
                    String[] lineItems;
                    while (line != null) {
                        pair = line.split(":");
                        pair[0] = pair[0].trim();
                        pair[1] = pair[1].trim();
                        int algIndex = algHash.get(pair[0]);
                        lineItems = pair[1].split(",");
                        float acc = Float.parseFloat(lineItems[0]);
                        float tempAcc;
                        setupAlgDSAccuracy[setupIndex][algIndex][dataIndex] =
                                acc;
                        for (int k = 1; k < lineItems.length; k++) {
                            tempAcc = Float.parseFloat(lineItems[k]);
                            if (tempAcc == acc) {
                                setupAlgDSBestK[setupIndex][algIndex][
                                        dataIndex] = k;
                                break;
                            }
                        }
                        line = br.readLine();
                    }
                } catch (IOException | NumberFormatException e) {
                    System.err.println(e.getMessage());
                    System.err.println("Could not summarize from: "
                            + new File(experimentalSetupPaths.get(setupIndex),
                            dsNames.get(dataIndex) + File.separator
                            + "totalSummary.txt").getPath());
                } finally {
                    if (br != null) {
                        br.close();
                    }
                }
            }
        }
    }

    /**
     * This method reads the experiment configuration from a prepared
     * configuration file that should be provided with the following bullets for
     * all the algorithms, datasets and setups: "
     *
     * @algorithm alg_name",
     * "
     * @ds_name ds_name", "
     * @setup setup_path", where the last bullet contains the path to a folder
     * representing a setup which contains all the data subdirectories with the
     * results. Multiple setups can be defined and this is the point of this
     * script.
     * @param configFile File that contains the experimental configuration.
     */
    public void readConfig(File configFile) throws Exception {
        BufferedReader br = new BufferedReader(new InputStreamReader(
                new FileInputStream(configFile)));
        String line = br.readLine();
        String[] pair;
        algHash = new HashMap<>(200, 20);
        datasetHash = new HashMap<>(200, 200);
        algNames = new ArrayList<>(50);
        dsNames = new ArrayList<>(100);
        experimentalSetupPaths = new ArrayList<>(50);
        try {
            while (line != null) {
                line = line.trim();
                if (line.startsWith("@algorithm")) {
                    pair = line.split(" ");
                    String algName = pair[1].trim();
                    algHash.put(algName, algNames.size());
                    algNames.add(algName);
                } else if (line.startsWith("@ds_name")) {
                    pair = line.split(" ");
                    String dsName = pair[1].trim();
                    datasetHash.put(dsName, dsNames.size());
                    dsNames.add(dsName);
                } else if (line.startsWith("@setup_path")) {
                    pair = line.split(" ");
                    String settingPath = pair[1].trim();
                    experimentalSetupPaths.add(settingPath);
                }
                line = br.readLine();
            }
        } catch (Exception e) {
            System.err.println(e.getMessage());
        } finally {
            br.close();
        }
    }

    /**
     * This script extracts the result summaries from multiple experimental
     * setups.
     *
     * @param args Command line parameters, as specified.
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        CommandLineParser clp = new CommandLineParser(true);
        clp.addParam("-config", "Path to the configuration file.",
                CommandLineParser.STRING, true, false);
        clp.addParam("-outDir", "Path to the output directory.",
                CommandLineParser.STRING, true, false);
        clp.parseLine(args);
        File configFile = new File((String) clp.getParamValues("-config").
                get(0));
        File outDir = new File((String) clp.getParamValues("-outDir").get(0));
        MultipleSetupsSummaryExtractor extractor =
                new MultipleSetupsSummaryExtractor();
        extractor.readConfig(configFile);
        extractor.getAllAccuracies();
        extractor.writeSummaries(outDir);
    }
}