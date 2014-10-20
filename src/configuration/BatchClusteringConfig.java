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
import distances.kernel.Kernel;
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
import java.lang.reflect.Field;
import java.util.ArrayList;
import learning.supervised.evaluation.cv.BatchClassifierTester;
import learning.unsupervised.evaluation.BatchClusteringTester;
import util.ReaderToStringUtil;

/**
 * This class is a configuration class for batch clustering testing, which
 * allows the batch tester to be invoked from other parts of the code, as well
 * as allowing customizable file format for saving the configuration. In this
 * case, it supports JSON I/O, which makes it easy to automatically generate
 * clustering evaluation requests from external code.
 * 
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class BatchClusteringConfig {
    
    // Whether to calculate model perplexity as a quality measure, as it can be
    // very slow for larger and high-dimensional datasets, prohibitively so.
    public boolean calculatePerplexity = false;
    // Whether to use split testing (training/test data).
    public boolean splitTesting = false;
    public float splitPerc = 1;
    // Whether a specific number of desired clusters was specified.
    public boolean nClustSpecified = true;
    // The normalization type to actually use in the experiments.
    public BatchClusteringTester.Normalization normType =
            BatchClusteringTester.Normalization.STANDARDIZE;
    public boolean clustersAutoSet = false;
    // The number of times a clustering is performed on every single dataset.
    public int timesOnDataSet = 30;
    // The preferred number of iterations, where applicable.
    public int minIter;
    public ArrayList<String> clustererNames = new ArrayList<>(10);
    // The experimental neighborhood range, with default values.
    public int kMin = 5, kMax = 5, kStep = 1;
    // Noise and mislabeling experimental ranges, with default values.
    public float noiseMin = 0, noiseMax = 0, noiseStep = 1, mlMin = 0,
            mlMax = 0, mlStep = 1;
    // Files pointing to the input and output.
    public File inDir, outDir, distancesDir, mlWeightsDir;
    // Paths to the tested datasets.
    public ArrayList<String> dsPaths = new ArrayList<>(100);
    // Paths to the corresponding metrics for the datasets.
    public ArrayList<CombinedMetric> dsMetric = new ArrayList<>(100);
    public Kernel ker = null;
    // Secondary distance specification.
    public BatchClassifierTester.SecondaryDistance secondaryDistanceType =
            BatchClassifierTester.SecondaryDistance.NONE;
    public int secondaryK = 50;
    // Clustering range.
    public int cluNumMin;
    public int cluNumMax;
    public int cluNumStep;
    // Aproximate kNN calculations specification.
    public float approximateNeighborsAlpha = 1f;
    public boolean approximateNeighbors = false;
    // The number of threads used for distance matrix and kNN set calculations.
    public int numCommonThreads = 8;
    
    /**
     * This method prints the clustering configuration to a Json string.
     * 
     * @return String that is the Json representation of this clustering
     * configuration.
     */
    public String toJsonString() {
        Gson gson = new Gson();
        String jsonString = gson.toJson(this, BatchClusteringConfig.class);
        return jsonString;
    }
    
    /**
     * This method loads the clustering configuration from a Json string.
     * 
     * @param jsonString String that is the Json representation of the
     * clustering configuration.
     * @return BatchClusteringConfig corresponding to the Json string.
     */
    public static BatchClusteringConfig fromJsonString(String jsonString) {
        Gson gson = new Gson();
        BatchClusteringConfig configObj = gson.fromJson(jsonString,
                BatchClusteringConfig.class);
        return configObj;
    }
    
    /**
     * This method prints this clustering configuration to a Json file.
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
     * This method loads this clustering configuration from a Json file.
     * 
     * @param inFile File containing the Json clustering configuration.
     * @return BatchClusteringConfig corresponding to the Json specification.
     * @throws Exception 
     */
    public static BatchClusteringConfig fromJsonFile(File inFile)
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
     * This method loads all the parameters from the provided clustering
     * configuration file.
     * 
     * @param inConfigFile File to load the configuration from.
     * @throws Exception
     */
    public void loadParameters(File inConfigFile) throws Exception {
        try (BufferedReader br = new BufferedReader(new InputStreamReader(
                new FileInputStream(inConfigFile)));) {
            String s = br.readLine();
            String[] lineParse;
            Class currIntMet;
            Class currFloatMet;

            while (s != null) {
                s = s.trim();
                if (s.startsWith("@algorithm")) {
                    // Clustering algorithm name. Can appear multiple times,
                    // defining multiple algorithms for comparisons.
                    lineParse = s.split(" ");
                    clustererNames.add(lineParse[1].toLowerCase());
                    System.out.println("Gonna test " + lineParse[1]);
                } else if (s.startsWith("@in_directory")) {
                    // Directory containing the datasets to be clustered.
                    lineParse = s.split("\\s+");
                    inDir = new File(lineParse[1]);
                } else if (s.startsWith("@distances_directory")) {
                    // Directory for the distance matrices. This is where they
                    // are persisted and/or loaded from.
                    lineParse = s.split("\\s+");
                    distancesDir = new File(lineParse[1]);
                } else if (s.startsWith("@alpha") ||
                        s.startsWith("@approximateNN")) {
                    // Approximate nearest neighbors with parameter alpha.
                    lineParse = s.split("\\s+");
                    approximateNeighborsAlpha = Float.parseFloat(lineParse[1]);
                    if (approximateNeighborsAlpha < 1f) {
                        approximateNeighbors = true;
                        System.out.println("alpha set to: "
                                + approximateNeighborsAlpha);
                    }
                } else if (s.startsWith("@normalization")) {
                    // The normalization procedure to apply.
                    lineParse = s.split("\\s+");
                    if (lineParse[1].toLowerCase().compareTo("no") == 0) {
                        normType = BatchClusteringTester.Normalization.NONE;
                    } else if (lineParse[1].toLowerCase().compareTo(
                            "normalizeTo01".toLowerCase()) == 0) {
                        normType = BatchClusteringTester.Normalization.NORM_01;
                    } else if (lineParse[1].toLowerCase().compareTo(
                            "TFIDF".toLowerCase()) == 0) {
                        normType = BatchClusteringTester.Normalization.TFIDF;
                    } else if (lineParse[1].toLowerCase().compareTo(
                            "standardize".toLowerCase()) == 0) {
                        normType = BatchClusteringTester.Normalization.STANDARDIZE;
                    } else {
                        normType = BatchClusteringTester.Normalization.STANDARDIZE;
                    }
                } else if (s.startsWith("@secondary_distance")) {
                    // Secondary distance specification.
                    lineParse = s.split("\\s+");
                    switch (lineParse[1].toLowerCase()) {
                        case "simcos": {
                            secondaryDistanceType =
                                    BatchClassifierTester.
                                    SecondaryDistance.SIMCOS;
                            break;
                        }
                        case "simhub": {
                            secondaryDistanceType =
                                    BatchClassifierTester.
                                    SecondaryDistance.SIMHUB;
                            break;
                        }
                        case "mp": {
                            secondaryDistanceType =
                                    BatchClassifierTester.SecondaryDistance.MP;
                            break;
                        }
                        case "ls": {
                            secondaryDistanceType =
                                    BatchClassifierTester.SecondaryDistance.LS;
                            break;
                        }
                        case "nicdm": {
                            secondaryDistanceType =
                                    BatchClassifierTester.
                                    SecondaryDistance.NICDM;
                            break;
                        }
                        default: {
                            secondaryDistanceType =
                                    BatchClassifierTester.
                                    SecondaryDistance.SIMCOS;
                            break;
                        }
                    }
                    if (lineParse.length >= 3) {
                        secondaryK = Integer.parseInt(lineParse[2]);
                    } else {
                        secondaryK = 50;
                    }
                } else if (s.startsWith("@out_directory")) {
                    // Directory where to output all the results.
                    lineParse = s.split("\\s+");
                    outDir = new File(lineParse[1]);
                } else if (s.startsWith("@split_training")) {
                    // Whether to use training/test splits or not.
                    lineParse = s.split(" ");
                    splitTesting = true;
                    splitPerc = Float.parseFloat(lineParse[1]);
                } else if (s.startsWith("@num_clusters")) {
                    // Range of cluster numbers to be tested for, or a
                    // specification of natural clustering where the number of
                    // classes is to be used as a default value.
                    clustersAutoSet = false;
                    lineParse = s.split("\\s+");
                    if (lineParse[1].toLowerCase().equals("natural")) {
                        clustersAutoSet = true;
                        nClustSpecified = true;
                    } else {
                        cluNumMin = Integer.parseInt(lineParse[1]);
                        cluNumMax = Integer.parseInt(lineParse[2]);
                        cluNumStep = Integer.parseInt(lineParse[3]);
                        clustersAutoSet = false;
                        nClustSpecified = true;
                    }
                } else if (s.startsWith("@k_range")) {
                    // Neighborhood size range.
                    lineParse = s.split("\\s+");
                    kMin = Integer.parseInt(lineParse[1]);
                    kMax = Integer.parseInt(lineParse[2]);
                    kStep = Integer.parseInt(lineParse[3]);
                } else if (s.startsWith("@noise_range")) {
                    // Uniform feature noiseRate range.
                    lineParse = s.split("\\s+");
                    noiseMin = Float.parseFloat(lineParse[1]);
                    noiseMax = Float.parseFloat(lineParse[2]);
                    noiseStep = Float.parseFloat(lineParse[3]);
                } else if (s.startsWith("@mislabeled_range")) {
                    // Introduced mislabeling rate.
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
                } else if (s.startsWith("@times")) {
                    // Number of times a clustering is repeated on a single
                    // dataset.
                    lineParse = s.split("\\s+");
                    timesOnDataSet = Integer.parseInt(lineParse[1]);
                } else if (s.startsWith("@iter")) {
                    // Minimum number of iterations to perform by the algorithms
                    lineParse = s.split("\\s+");
                    minIter = Integer.parseInt(lineParse[1]);
                } else if (s.startsWith("@kernel")) {
                    // Kernel specification.
                    lineParse = s.split("\\s+");
                    String kerName = null;
                    ArrayList<String> paramNames = new ArrayList<>(4);
                    ArrayList<Float> paramValues = new ArrayList<>(4);
                    for (int i = 1; i < lineParse.length; i++) {
                        String[] pair = lineParse[i].split(":");
                        if (pair[0].equals("name")) {
                            kerName = pair[1];
                        } else {
                            paramNames.add(pair[0]);
                            paramValues.add(Float.parseFloat(pair[1]));
                        }
                    }
                    if (kerName != null) {
                        Class clazz = Class.forName(kerName);
                        ker = (Kernel) (clazz.newInstance());
                        System.out.println("Using kernel: " + clazz.getName());
                        for (int i = 0; i < paramNames.size(); i++) {
                            Field field = clazz.getField(paramNames.get(i));
                            field.set(ker, paramValues.get(i));
                        }
                    }
                } else if (s.startsWith("@dataset")) {
                    // Dataset specification: data path + metric to use.
                    lineParse = s.split("\\s+");
                    dsPaths.add(lineParse[1]);
                    if (lineParse[1].startsWith("sparse:")) {
                        SparseCombinedMetric scmet = new SparseCombinedMetric(
                                null, null, (SparseMetric) (Class.forName(
                                lineParse[1]).newInstance()),
                                CombinedMetric.DEFAULT);
                        dsMetric.add(scmet);
                    } else {
                        CombinedMetric cmet = new CombinedMetric();
                        if (!lineParse[2].equals("null")) {
                            currIntMet = Class.forName(lineParse[2]);
                            cmet.setIntegerMetric((DistanceMeasure) (
                                    currIntMet.newInstance()));
                        }
                        if (!lineParse[3].equals("null")) {
                            currFloatMet = Class.forName(lineParse[3]);
                            cmet.setFloatMetric((DistanceMeasure) (
                                    currFloatMet.newInstance()));
                        }
                        cmet.setCombinationMethod(CombinedMetric.DEFAULT);
                        dsMetric.add(cmet);
                    }
                } else if (s.startsWith("@")) {
                    // This means that there is probably a typo in the
                    // configuration file or an option is being set that is not
                    // supported.
                    System.err.println("The following option line was "
                            + "ignored: " + s);
                }
                s = br.readLine();
            }
            // Now correct the paths by adding inDir in front.
            for (int i = 0; i < dsPaths.size(); i++) {
                if (!dsPaths.get(i).startsWith("sparse:")) {
                    dsPaths.set(i, (new File(inDir, dsPaths.get(i))).getPath());
                } else {
                    dsPaths.set(i, "sparse:"
                            + (new File(inDir, dsPaths.get(i).substring(
                            dsPaths.get(i).indexOf(":") + 1,
                            dsPaths.get(i).length()))).getPath());
                }
            }
        } catch (Exception e) {
            throw e;
        }
    }
    
}
