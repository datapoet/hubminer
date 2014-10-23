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
import data.representation.DataSet;
import data.representation.sparse.BOWDataSet;
import distances.primary.CombinedMetric;
import distances.primary.DistanceMeasure;
import distances.sparse.SparseCombinedMetric;
import distances.sparse.SparseMetric;
import ioformat.FileUtil;
import ioformat.IOARFF;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import learning.supervised.evaluation.cv.BatchClassifierTester;
import learning.supervised.evaluation.cv.CVFoldsIO;
import learning.supervised.evaluation.cv.MultiCrossValidation;
import networked_experiments.DataFromOpenML;
import networked_experiments.HMOpenMLConnector;
import preprocessing.instance_selection.InstanceSelector;
import preprocessing.instance_selection.ReducersFactory;
import util.ReaderToStringUtil;

/**
 * This class is a configuration class for batch classification testing, which
 * allows the batch tester to be invoked from other parts of the code, as well
 * as allowing customizable file format for saving the configuration. In this
 * case, it supports JSON I/O, which makes it easy to automatically generate
 * classification evaluation requests from external code.
 * 
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class BatchClassifierConfig {
    
    /**
     * The default constructor.
     */
    public BatchClassifierConfig() {
    }
    
    // OpenML authentication. Required if OpenML data sources are pulled for
    // experiments, not necessary otherwise.
    private String openmlUsername;
    private String openmlPassword;
    // List of the specified OpenML task IDs.
    public ArrayList<Integer> openMLTaskIDList = new ArrayList<>();
    // List of dataset names that the task IDs correspond to.
    public ArrayList<String> openMLTaskDataSetNameList = new ArrayList<>();
    public ArrayList<Integer> openMLTaskDataIndex = new ArrayList<>();
    public HashMap<Integer, Integer> dataIndexToOpenMLCounterMap =
            new HashMap<>();
    // This is used for registering versioned source code files with OpenML.
    public File hubMinerSourceDir;
    public BatchClassifierTester.SecondaryDistance secondaryDistanceType =
            BatchClassifierTester.SecondaryDistance.NONE;
    public int secondaryDistanceK = 50;
    public BatchClassifierTester.Normalization normType =
            BatchClassifierTester.Normalization.NONE;
    public int numTimes = 10;
    public int numFolds = 10;
    // A range of neighborhood sizes to test for, with default values.
    public int kMin = 5, kMax = 5, kStep = 1;
    // A range of noise and mislabeling rates to test for, with default values.
    public float noiseMin = 0, noiseMax = 0, noiseStep = 1, mlMin = 0,
            mlMax = 0, mlStep = 1;
    public File inDir, outDir, summaryDir, inLabelFile, distancesDir,
            mlWeightsDir, foldsDir;
    public ArrayList<String> classifierNames = new ArrayList<>(10);
    // List of paths to the datasets that the experiment is to be executed on.
    // In the OpenML mode, the data is saved to these paths prior to using it.
    public ArrayList<String> dsPaths = new ArrayList<>(100);
    // List of CombinedMetric objects for distance calculations that correspond
    // to different datasets. Different datasets might require different
    // metrics to be used, so this is why it is necessary to explicitly specify
    // a metric for each dataset.
    public ArrayList<CombinedMetric> dsMetric = new ArrayList<>(100);
    public ArrayList<Integer>[][][] allDataSetFolds;
    // An alternative specification to the fold specification above, used for
    // OpenML compatibility.
    public ArrayList<Integer>[][][][] trainTestIndexes;
    // Whether we are in the multi-label experimental mode, where we are testing
    // different representations of the same underlying dataset that has
    // multiple label distributions / classification problems defined on top.
    public boolean multiLabelMode = false;
    // The number of classification problems on the dataset.
    public int numDifferentLabelings = 1;
    // Separator in the label file.
    public String lsep;
    public float alphaAppKNNs = 1f;
    public boolean approximateKNNs = false;
    // Current instance selector.
    public InstanceSelector selector = null;
    // Current selection rate.
    public float selectorRate = 0.1f;
    // Hubness estimation mode for the prototypes in instance selection. It can
    // be estimated from all the training data (retained and rejected), which
    // is an unbiased approach to hubness estimation in instance selection - or
    // it can be simply estimated from the selected prototype set, which is
    // simpler but introduces a bias in the estimates.
    public int protoHubnessMode = MultiCrossValidation.PROTO_UNBIASED;
    // The number of threads used for distance matrix and kNN set calculations.
    public int numCommonThreads = 8;
    
    /**
     * Check whether a dataset that is listed is an openML task or an ordinary
     * local dataset. The index that is given as a parameter is the index of
     * the dataset in the list of datasets parsed from the configuration file.
     * 
     * @param datasetIndex Integer that is the index of the dataset in the list
     * of datasets parsed from the configuration file.
     * @return True if the corresponding dataset is of OpenML variety or false
     * if it is a local dataset.
     */
    public boolean dataIndexIsForOpenML(int datasetIndex) {
        if (dataIndexToOpenMLCounterMap != null) {
            return dataIndexToOpenMLCounterMap.containsKey(datasetIndex);
        } else {
            return false;
        }
    }
    
    /**
     * @return OpenMLConnector object corresponding to the provided
     * authentication specification.
     */
    public HMOpenMLConnector getOpenMLConnector() {
        if (openmlUsername != null && openmlPassword != null) {
            return new HMOpenMLConnector(openmlUsername, openmlPassword);
        } else {
            return null;
        }
    }
    
    /**
     * This method prints the classification configuration to a Json string.
     * 
     * @return String that is the Json representation of this classification
     * configuration.
     */
    public String toJsonString() {
        Gson gson = new Gson();
        String jsonString = gson.toJson(this, BatchClassifierConfig.class);
        return jsonString;
    }
    
    /**
     * This method loads the classification configuration from a Json string.
     * 
     * @param jsonString String that is the Json representation of the
     * classification configuration.
     * @return BatchClassifierConfig corresponding to the Json string.
     */
    public static BatchClassifierConfig fromJsonString(String jsonString) {
        Gson gson = new Gson();
        BatchClassifierConfig configObj = gson.fromJson(jsonString,
                BatchClassifierConfig.class);
        return configObj;
    }
    
    /**
     * This method prints this classification configuration to a Json file.
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
     * This method loads this classification configuration from a Json file.
     * 
     * @param inFile File containing the Json classification configuration.
     * @return BatchClassifierConfig corresponding to the Json specification.
     * @throws Exception 
     */
    public static BatchClassifierConfig fromJsonFile(File inFile)
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
     * This method loads the experimental configuration from the configuration
     * file.
     * 
     * @param inConfigFile File to load the configuration from.
     * @throws Exception 
     */
    public void loadParameters(File inConfigFile) throws Exception {
        try (BufferedReader br = new BufferedReader(new InputStreamReader(
                new FileInputStream(inConfigFile)));) {
            String s = br.readLine();
            String[] lineItems;
            Class currIntMetric;
            Class currFloatMetric;
            // Go through the configuration file.
            int dataIndex = -1;
            while (s != null) {
                s = s.trim();
                if (s.startsWith("@algorithm")) {
                    // Add an algorithm to the list of tested algorithms.
                    lineItems = s.split("\\s+");
                    classifierNames.add(lineItems[1]);
                    System.out.println("Preparing to test " + lineItems[1]);
                } else if (s.startsWith("@hubminer_source_directory")) {
                    lineItems = s.split("\\s+");
                    hubMinerSourceDir = new File(lineItems[1]);
                } else if (s.startsWith("@openml_authentication")) {
                    // Specification of OpenML authentication.
                    lineItems = s.split("\\s+");
                    openmlUsername = lineItems[1];
                    openmlPassword = lineItems[2];
                } else if (s.startsWith("@openml_task")) {
                    dataIndex++;
                    // Specification of an OpenML task. The first parameters is
                    // the path to where the dataset is to be persisted, that is
                    // fetched via the OpenML connection. The second and the
                    // third parameter correspond to integer and float metrics,
                    // as usual. The fourth and the last parameter is the actual
                    // task ID from OpenML, used to identify the data and fetch
                    // the data and the folds.
                    lineItems = s.split("\\s+");
                    dsPaths.add(lineItems[1]);
                    if (lineItems[1].startsWith("sparse:")) {
                        // Sparse datasets in the sparse format should be
                        // precluded with "sparse:".
                        // The second item is the SparseMetric specification.
                        SparseCombinedMetric smc = new SparseCombinedMetric(
                                null, null, (SparseMetric) (
                                Class.forName(lineItems[2]).newInstance()),
                                CombinedMetric.DEFAULT);
                        dsMetric.add(smc);
                    } else {
                        // The second item is the CombinedMetric specification.
                        CombinedMetric dsCmet = new CombinedMetric();
                        if (!lineItems[2].equals("null")) {
                            currIntMetric = Class.forName(lineItems[3]);
                            dsCmet.setIntegerMetric((DistanceMeasure)
                                    (currIntMetric.newInstance()));
                        }
                        if (!lineItems[3].equals("null")) {
                            currFloatMetric = Class.forName(lineItems[3]);
                            dsCmet.setFloatMetric((DistanceMeasure)
                                    (currFloatMetric.newInstance()));
                        }
                        dsCmet.setCombinationMethod(CombinedMetric.DEFAULT);
                        dsMetric.add(dsCmet);
                    }
                    int taskId = Integer.parseInt(lineItems[4]);
                    openMLTaskIDList.add(taskId);
                    openMLTaskDataIndex.add(dataIndex);
                    openMLTaskDataSetNameList.add(getOpenMLConnector().
                            getDataSetNameForTaskID(taskId));
                    dataIndexToOpenMLCounterMap.put(dataIndex,
                            openMLTaskIDList.size() - 1);
                } else if (s.startsWith("@cross_validation")) {
                    // Specify the number of times and folds.
                    lineItems = s.split("\\s+");
                    numTimes = Integer.parseInt(lineItems[1]);
                    numFolds = Integer.parseInt(lineItems[2]);
                } else if (s.startsWith("@folds_directory")) {
                    // Specify the directory where the folds are to be stored
                    // and/or loaded from.
                    lineItems = s.split("\\s+");
                    foldsDir = new File(lineItems[1]);
                } else if (s.startsWith("@in_directory")) {
                    // Specify an input directory for the data files.
                    lineItems = s.split("\\s+");
                    inDir = new File(lineItems[1]);
                } else if (s.startsWith("@distances_directory")) {
                    // Specify the load/save directory for the distance
                    // matrices.
                    lineItems = s.split("\\s+");
                    distancesDir = new File(lineItems[1]);
                } else if (s.startsWith("@alpha") ||
                        s.startsWith("@approximateNN")) {
                    // The approximate kNN calculation mode is on.
                    lineItems = s.split("\\s+");
                    alphaAppKNNs = Float.parseFloat(lineItems[1]);
                    if (alphaAppKNNs < 1f) {
                        approximateKNNs = true;
                        System.out.println("alpha set to: " + alphaAppKNNs);
                    }
                } else if (s.startsWith("@instance_selection")) {
                    // Specification of instance selection.
                    lineItems = s.split("\\s+");
                    // The first value is the selector name.
                    selector = ReducersFactory.getReducerForName(lineItems[1]);
                    System.out.println("Instance selection with: " +
                            lineItems[1]);
                    if (lineItems.length > 2) {
                        // The second value is the selection rate.
                        System.out.println("Retained sample rate: " +
                                lineItems[2] + " , if applicable");
                        selectorRate = Float.parseFloat(lineItems[2]);
                    } else {
                        // Autimatic selection rate, if and when applicable.
                        if (selector instanceof preprocessing.
                                instance_selection.CNN || selector instanceof 
                                preprocessing.instance_selection.GCNN) {
                            selectorRate = 0;
                            System.out.println("Automatic sampling rate");
                        }
                    }
                } else if (s.startsWith("@protohubness") ||
                        s.startsWith("@proto_hubness")) {
                    // Whether to use the unbiased or biased estimates for
                    // hubness in instance selection.
                    lineItems = s.split("\\s+");
                    if (lineItems[1].equalsIgnoreCase("unbiased")) {
                        protoHubnessMode = MultiCrossValidation.PROTO_UNBIASED;
                    } else {
                        protoHubnessMode = MultiCrossValidation.PROTO_BIASED;
                    }
                } else if (s.startsWith("@normalization")) {
                    // Different normalization modes.
                    lineItems = s.split("\\s+");
                    if (lineItems[1].toLowerCase().compareTo("no") == 0) {
                        normType = BatchClassifierTester.Normalization.NONE;
                    } else if (lineItems[1].toLowerCase().compareTo(
                            "normalizeTo01".toLowerCase()) == 0) {
                        normType = BatchClassifierTester.Normalization.NORM_01;
                    } else if (lineItems[1].toLowerCase().compareTo(
                            "TFIDF".toLowerCase()) == 0) {
                        normType = BatchClassifierTester.Normalization.TFIDF;
                    } else if (lineItems[1].equalsIgnoreCase("standardize")) {
                        normType = BatchClassifierTester.
                                Normalization.STANDARDIZE;
                    } else {
                        normType = BatchClassifierTester.
                                Normalization.STANDARDIZE;
                    }
                } else if (s.startsWith("@secondary_distance")) {
                    // Secondary distance specification.
                    lineItems = s.split("\\s+");
                    switch (lineItems[1].toLowerCase()) {
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
                    if (lineItems.length >= 3) {
                        secondaryDistanceK = Integer.parseInt(lineItems[2]);
                    } else {
                        secondaryDistanceK = 50;
                    }
                } else if (s.startsWith("@out_directory")) {
                    // Output directory for the results.
                    lineItems = s.split("\\s+");
                    outDir = new File(lineItems[1]);
                } else if (s.startsWith("@label_file")) {
                    // In case of multi-labeled classification (not default),
                    // this file specified multiple label assignments for the
                    // tested representations.
                    lineItems = s.split("\\s+");
                    inLabelFile = new File(lineItems[1]);
                    multiLabelMode = true;
                    if (lineItems.length >= 3) {
                        String tempSep = lineItems[2];
                        for (int hrm = 3; hrm < lineItems.length; hrm++) {
                            tempSep += " ";
                            tempSep += lineItems[hrm];
                        }
                        lsep = tempSep.substring(1, tempSep.length() - 1);
                        System.out.println("label separator \"" + lsep + "\"");
                    }
                } else if (s.startsWith("@summary_directory")) {
                    // Output directory for the classification summaries.
                    lineItems = s.split("\\s+");
                    summaryDir = new File(lineItems[1]);
                } else if (s.startsWith("@k_range")) {
                    // The range of tested neighborhood sizes.
                    lineItems = s.split("\\s+");
                    kMin = Integer.parseInt(lineItems[1]);
                    kMax = Integer.parseInt(lineItems[2]);
                    kStep = Integer.parseInt(lineItems[3]);
                } else if (s.startsWith("@noise_range")) {
                    // The range of tested feature noise rates.
                    lineItems = s.split("\\s+");
                    noiseMin = Float.parseFloat(lineItems[1]);
                    noiseMax = Float.parseFloat(lineItems[2]);
                    noiseStep = Float.parseFloat(lineItems[3]);
                } else if (s.startsWith("@mislabeled_range")) {
                    // The range of tested label noise rates.
                    lineItems = s.split("\\s+");
                    mlMin = Float.parseFloat(lineItems[1]);
                    mlMax = Float.parseFloat(lineItems[2]);
                    mlStep = Float.parseFloat(lineItems[3]);
                } else if (s.startsWith("@mislabeling_weights_dir")) {
                    // Directory with the mislabeling instance weights, if
                    // the user specifies the instance-weight-proportional
                    // mislabeling scheme, such as hubness-proportional label
                    // noise.
                    lineItems = s.split("\\s+");
                    mlWeightsDir = new File(lineItems[1]);
                } else if (s.startsWith("@common_threads")) {
                    // The number of threads to use in distance matrix and kNN
                    // calculations.
                    lineItems = s.split("\\s+");
                    numCommonThreads = Integer.parseInt(lineItems[1]);
                } else if (s.startsWith("@dataset")) {
                    dataIndex++;
                    // Dataset specification.
                    lineItems = s.split("\\s+");
                    dsPaths.add(lineItems[1]);
                    if (lineItems[1].startsWith("sparse:")) {
                        // Sparse datasets in the sparse format should be
                        // precluded with "sparse:".
                        // The second item is the SparseMetric specification.
                        SparseCombinedMetric smc = new SparseCombinedMetric(
                                null, null, (SparseMetric) (
                                Class.forName(lineItems[2]).newInstance()),
                                CombinedMetric.DEFAULT);
                        dsMetric.add(smc);
                    } else {
                        // The second item is the CombinedMetric specification.
                        CombinedMetric dsCmet = new CombinedMetric();
                        if (!lineItems[2].equals("null")) {
                            currIntMetric = Class.forName(lineItems[3]);
                            dsCmet.setIntegerMetric((DistanceMeasure)
                                    (currIntMetric.newInstance()));
                        }
                        if (!lineItems[3].equals("null")) {
                            currFloatMetric = Class.forName(lineItems[3]);
                            dsCmet.setFloatMetric((DistanceMeasure)
                                    (currFloatMetric.newInstance()));
                        }
                        dsCmet.setCombinationMethod(CombinedMetric.DEFAULT);
                        dsMetric.add(dsCmet);
                    }
                } else if (s.startsWith("@")) {
                    // This means that there is probably a typo in the
                    // configuration file or an option is being set that is not
                    // supported.
                    System.err.println("WARNING: The following option line was "
                            + "ignored: " + s);
                }
                s = br.readLine();
            }
            // Prepend the input directory to the relative data paths.
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
            // If the data source is an OpenML data source, we fetch the data
            // here and save it to the specified path as ARFF, so that it can
            // be loaded later into the cross-validation batch experimentational
            // framework.
            trainTestIndexes = new ArrayList[dsPaths.size()][][][];
            if (dataIndexToOpenMLCounterMap.containsKey(dataIndex)) {
                HMOpenMLConnector openMLProxy = getOpenMLConnector();
                int openMLArrIndex = dataIndexToOpenMLCounterMap.get(dataIndex);
                int taskID = openMLTaskIDList.get(openMLArrIndex);
                DataFromOpenML openMLData = openMLProxy.fetchExperimentData(
                        taskID);
                DataSet dset = openMLData.filteredDSet;
                trainTestIndexes[dataIndex] = openMLData.trainTestIndexes;
                String dataSavePath =
                        dsPaths.get(dataIndex).startsWith("sparse:") ?
                        dsPaths.get(dataIndex).substring(7) :
                        dsPaths.get(dataIndex);
                IOARFF saver = new IOARFF();
                if (dsPaths.get(dataIndex).startsWith("sparse:")) {
                    saver.saveSparseLabeled((BOWDataSet) dset, dataSavePath);
                    System.out.println("Data downloaded to: " + dataSavePath);
                } else {
                    saver.saveLabeled(dset, dataSavePath);
                    System.out.println("Data downloaded to: " + dataSavePath);
                }
            }
            allDataSetFolds = new ArrayList[dsPaths.size()][][];
            if (foldsDir != null) {
                int datasetIndex = -1;
                for (String dsPath : dsPaths) {
                    datasetIndex++;
                    if (!dataIndexIsForOpenML(datasetIndex)) {
                        File dsFile = new File(dsPath);
                        File foldsFile = new File(foldsDir,
                                dsFile.getName().substring(0, dsFile.getName().
                                lastIndexOf(".")) + "_cv_" + numTimes + "_" +
                                numFolds + ".json");
                        if (foldsFile.exists()) {
                            System.out.println(
                                    "Loading the existing folds from: " +
                                    foldsFile.getPath());
                            allDataSetFolds[datasetIndex] =
                                    CVFoldsIO.loadAllFolds(foldsFile);
                        }
                    }
                }
            }
        } catch (Exception e) {
            throw e;
        }
    }
}
