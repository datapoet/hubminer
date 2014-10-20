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
package configuration;

import com.google.gson.Gson;
import data.neighbors.hubness.BatchHubnessAnalyzer.Normalization;
import distances.primary.CombinedMetric;
import distances.primary.DistanceMeasure;
import distances.sparse.SparseCombinedMetric;
import distances.sparse.SparseMetric;
import ioformat.FileUtil;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import learning.supervised.evaluation.cv.BatchClassifierTester.SecondaryDistance;
import util.ReaderToStringUtil;

/**
 * This class is a configuration class for batch hubness stats calculations,
 * which allows the batch tester to be invoked from other parts of the code, as
 * well as allowing customizable file format for saving the configuration. In
 * this case, it supports JSON I/O, which makes it easy to automatically
 * generate the hubness evaluation requests from external code.
 * 
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class BatchHubnessAnalysisConfig {
    
    public SecondaryDistance secondaryDistanceType;
    // Neighborhood size to use for secondary distances.
    public int secondaryDistanceK = 50;
    // Normalization types.
    // The normalization type to actually use in the experiments.
    public Normalization normType = Normalization.STANDARDIZE;
    // The upper limit on the neighborhood sizes to examine.
    public int kMax = 50;
    // Noise and mislabeling levels to vary, with default values.
    public float noiseMin = 0, noiseMax = 0, noiseStep = 1, mlMin = 0,
            mlMax = 0, mlStep = 1;
    // Input and output files and directories.
    public File inDir, outDir, mlWeightsDir;
    // Paths to the datasets that are being processed.
    public ArrayList<String> dsPaths = new ArrayList<>(100);
    // A list of metrics corresponding to the datasets.
    public ArrayList<CombinedMetric> dsMetric = new ArrayList<>(100);
    // Directory containing the distances.
    public File distancesDir;
    // The number of threads used for distance matrix and kNN set calculations.
    public int numCommonThreads = 8;
    
    /**
     * This method prints the hubness analysis configuration to a Json string.
     * 
     * @return String that is the Json representation of this  configuration.
     */
    public String toJsonString() {
        Gson gson = new Gson();
        String jsonString = gson.toJson(this, BatchHubnessAnalysisConfig.class);
        return jsonString;
    }
    
    /**
     * This method loads the hubness analysis configuration from a Json string.
     * 
     * @param jsonString String that is the Json representation of the
     * configuration.
     * @return BatchHubnessAnalysisConfig corresponding to the Json string.
     */
    public static BatchHubnessAnalysisConfig fromJsonString(String jsonString) {
        Gson gson = new Gson();
        BatchHubnessAnalysisConfig configObj = gson.fromJson(jsonString,
                BatchHubnessAnalysisConfig.class);
        return configObj;
    }
    
    /**
     * This method prints this hubness analysis configuration to a Json file.
     * 
     * @param outFile File to print the Json configuration to.
     * @throws IOException 
     */
    public void toJsonFile(File outFile) throws IOException {
        if (!outFile.exists() || !outFile.isFile()) {
            throw new IOException("Bad file path.");
        } else {
            FileUtil.createFile(outFile);
            try (PrintWriter pw = new PrintWriter(new FileWriter(outFile));) {
                pw.write(toJsonString());
            } catch (IOException e) {
                throw e;
            }
        }
    }
    
    /**
     * This method loads this hubness analysis configuration from a Json file.
     * 
     * @param inFile File containing the Json configuration.
     * @return BatchHubnessAnalysisConfig corresponding to the Json
     * specification.
     * @throws Exception 
     */
    public static BatchHubnessAnalysisConfig fromJsonFile(File inFile)
            throws Exception {
        if (!inFile.exists() || !inFile.isFile()) {
            throw new IOException("Bad file path.");
        } else {
            try (BufferedReader br = new BufferedReader(new InputStreamReader(
                    new FileInputStream(inFile)))) {
                String jsonString = ReaderToStringUtil.readAsSingleString(br);
                return fromJsonString(jsonString);
            } catch (IOException e) {
                throw e;
            }
        }
    }
    
    /**
     * This method loads the parameters from the configuration file.
     *
     * @param inConfigFile File to load the configuration from.
     * @throws Exception
     */
    public void loadParameters(File inConfigFile) throws Exception {
        try (BufferedReader br = new BufferedReader(
                new InputStreamReader(new FileInputStream(inConfigFile)))) {
            String s = br.readLine();
            String[] lineParse;
            // Integer and float metrics.
            Class currIntMet;
            Class currFloatMet;
            secondaryDistanceType = SecondaryDistance.NONE;
            // Read the file line by line.
            while (s != null) {
                s = s.trim();
                if (s.startsWith("@in_directory")) {
                    // Input directory.
                    lineParse = s.split("\\s+");
                    inDir = new File(lineParse[1]);
                } else if (s.startsWith("@out_directory")) {
                    // Output directory.
                    lineParse = s.split("\\s+");
                    outDir = new File(lineParse[1]);
                } else if (s.startsWith("@k_max")) {
                    // Maximal k-value to which to iterate.
                    lineParse = s.split("\\s+");
                    kMax = Integer.parseInt(lineParse[1]);
                } else if (s.startsWith("@noise_range")) {
                    // Noise range: min, max, increment.
                    lineParse = s.split("\\s+");
                    noiseMin = Float.parseFloat(lineParse[1]);
                    noiseMax = Float.parseFloat(lineParse[2]);
                    noiseStep = Float.parseFloat(lineParse[3]);
                } else if (s.startsWith("@mislabeled_range")) {
                    // Mislabeling range: min, max, increment.
                    lineParse = s.split("\\s+");
                    mlMin = Float.parseFloat(lineParse[1]);
                    mlMax = Float.parseFloat(lineParse[2]);
                    mlStep = Float.parseFloat(lineParse[3]);
                } else if (s.startsWith("@mislabeling_weights_dir")) {
                    // Directory with the mislabeling instance weights, if
                    // the user specifies the instance-weight-proportional
                    // mislabeling scheme, such as hubness-proportional label
                    // noise.
                    lineParse = s.split("\\s+");
                    mlWeightsDir = new File(lineParse[1]);
                } else if (s.startsWith("@common_threads")) {
                    // The number of threads to use in distance matrix and kNN
                    // calculations.
                    lineParse = s.split("\\s+");
                    numCommonThreads = Integer.parseInt(lineParse[1]);
                } else if (s.startsWith("@normalization")) {
                    // Normalization specification.
                    lineParse = s.split("\\s+");
                    if (lineParse[1].toLowerCase().compareTo("no") == 0) {
                        normType = Normalization.NONE;
                    } else if (lineParse[1].toLowerCase().compareTo(
                            "normalizeTo01".toLowerCase()) == 0) {
                        normType = Normalization.NORM_01;
                    } else if (lineParse[1].toLowerCase().compareTo(
                            "TFIDF".toLowerCase()) == 0) {
                        normType = Normalization.TFIDF;
                    } else {
                        normType = Normalization.STANDARDIZE;
                    }
                } else if (s.startsWith("@secondary_distance")) {
                    // Secondary distance specification.
                    lineParse = s.split("\\s+");
                    switch (lineParse[1].toLowerCase()) {
                        case "simcos": {
                            secondaryDistanceType = SecondaryDistance.SIMCOS;
                            break;
                        }
                        case "simhub": {
                            secondaryDistanceType = SecondaryDistance.SIMHUB;
                            break;
                        }
                        case "mp": {
                            secondaryDistanceType = SecondaryDistance.MP;
                            break;
                        }
                        case "ls": {
                            secondaryDistanceType = SecondaryDistance.LS;
                            break;
                        }
                        case "nicdm": {
                            secondaryDistanceType = SecondaryDistance.NICDM;
                            break;
                        }
                        default: {
                            secondaryDistanceType = SecondaryDistance.SIMCOS;
                            break;
                        }
                    }
                    if (lineParse.length >= 3) {
                        secondaryDistanceK = Integer.parseInt(lineParse[2]);
                    } else {
                        secondaryDistanceK = 50;
                    }
                } else if (s.startsWith("@distances_directory")) {
                    // Directory for loading and/or persisting the distance
                    // matrices.
                    lineParse = s.split("\\s+");
                    distancesDir = new File(lineParse[1]);
                } else if (s.startsWith("@dataset")) {
                    // Dataset specification.
                    lineParse = s.split("\\s+");
                    // The data path relative to the input directory.
                    dsPaths.add(lineParse[1]);
                    // What follows is the metric to use on the dataset.
                    if (lineParse[1].startsWith("sparse:")) {
                        // If the path is preceded by "sparse:", we read in a
                        // sparse metric.
                        SparseCombinedMetric cmetSparse =
                                new SparseCombinedMetric(null, null,
                                (SparseMetric) (Class.forName(
                                lineParse[2]).newInstance()),
                                CombinedMetric.DEFAULT);
                        dsMetric.add(cmetSparse);
                    } else {
                        // Load the specified metric.
                        CombinedMetric cmetLoaded = new CombinedMetric();
                        if (!lineParse[2].equals("null")) {
                            currIntMet = Class.forName(lineParse[2]);
                            cmetLoaded.setIntegerMetric(
                                    (DistanceMeasure) (
                                    currIntMet.newInstance()));
                        }
                        if (!lineParse[3].equals("null")) {
                            currFloatMet = Class.forName(lineParse[3]);
                            cmetLoaded.setFloatMetric(
                                    (DistanceMeasure) (
                                    currFloatMet.newInstance()));
                        }
                        cmetLoaded.setCombinationMethod(CombinedMetric.DEFAULT);
                        dsMetric.add(cmetLoaded);
                    }
                }
                s = br.readLine();
            }
            // Convert relative to absolute paths, by pre-pending the input
            // directory path.
            for (int i = 0; i < dsPaths.size(); i++) {
                if (!dsPaths.get(i).startsWith("sparse:")) {
                    dsPaths.set(i, (new File(inDir, dsPaths.get(i))).getPath());
                } else {
                    dsPaths.set(i, "sparse:" + (new File(inDir,
                            dsPaths.get(i).substring(
                            dsPaths.get(i).indexOf(":") + 1,
                            dsPaths.get(i).length()))).getPath());
                }
            }
        } catch (Exception e) {
            throw e;
        }
    }
    
}
