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
package learning.supervised.evaluation.cv;

import com.google.gson.Gson;
import configuration.BatchClassifierConfig;
import data.neighbors.NSFUserInterface;
import data.representation.DataSet;
import data.representation.discrete.DiscretizedDataSet;
import data.representation.discrete.tranform.EntropyMDLDiscretizer;
import data.representation.sparse.BOWDataSet;
import data.representation.util.DataMineConstants;
import distances.primary.CombinedMetric;
import distances.sparse.SparseCombinedMetric;
import filters.TFIDF;
import ioformat.DistanceMatrixIO;
import ioformat.FileUtil;
import ioformat.IOARFF;
import ioformat.IOCSV;
import ioformat.SupervisedLoader;
import ioformat.results.BatchStatSummarizer;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;
import learning.supervised.ClassifierFactory;
import learning.supervised.DiscreteClassifier;
import learning.supervised.evaluation.ClassificationEstimator;
import learning.supervised.evaluation.ClassifierParametrization;
import learning.supervised.evaluation.ValidateableInterface;
import learning.supervised.interfaces.DistMatrixUserInterface;
import learning.supervised.interfaces.NeighborPointsQueryUserInterface;
import networked_experiments.ClassificationResultHandler;
import networked_experiments.HMOpenMLConnector;
import preprocessing.instance_selection.InstanceSelector;
import preprocessing.instance_selection.ReducersFactory;

/**
 * This class implements the functionality for batch testing a series of
 * classification algorithms on a series of datasets, with possible instance
 * selection and / or metric learning, over a series of different neighborhood
 * sizes (for kNN methods), Gaussian feature noise levels and mislabeling levels
 * that can be specified via the configuration file.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class BatchClassifierTester {

    // Connector for API calls to OpenML in case when networked experiments are
    // performed and data and training/test splits are obtained from OpenML
    // over the network.
    private HMOpenMLConnector openmlConnector;
    // This directory specification is necessary for registering versioned
    // algorithm source files with OpenML. It is not necessary otherwise, for
    // local experiments.
    private File hubMinerSourceDir;
    // Types of applicable secondary distances.
    public enum SecondaryDistance {

        NONE, SIMCOS, SIMHUB, MP, LS, NICDM
    }
    // Secondary distances are not used by default, unless specified by the
    // user.
    private SecondaryDistance secondaryDistanceType = SecondaryDistance.NONE;
    // Neighborhood size to use for secondary distance calculations. The default
    // is 50, but it can also be directly specified in the experimental
    // configuration file.
    private int secondaryDistanceK = 50;
    // Applicable types of feature normalization.

    public enum Normalization {

        NONE, STANDARDIZE, NORM_01, TFIDF;
    }
    // The normalization type to actually use in the experiments.
    private Normalization normType = Normalization.NONE;
    
    // Paramters for the number of times and folds in cross-validation.
    private int numTimes = 10;
    private int numFolds = 10;
    
    // A range of neighborhood sizes to test for, with default values.
    private int kMin = 5, kMax = 5, kStep = 1;
    // A range of noise and mislabeling rates to test for, with default values.
    private float noiseMin = 0, noiseMax = 0, noiseStep = 1, mlMin = 0,
            mlMax = 0, mlStep = 1;
    // Various files and directories used in the experiment.
    private File inConfigFile, inDir, outDir, currOutDSDir, summaryDir,
            inLabelFile, distancesDir, mlWeightsDir, foldsDir;
    // A list of classifier names to use in the experiment.
    private ArrayList<String> classifierNames = new ArrayList<>(10);
    // Possible parameter value maps to use with certain algorithms.
    HashMap<String, HashMap<String, Object>> algorithmParametrizationMap;
    // Classifier prototype arrays.
    ValidateableInterface[] nonDiscreteArray, discreteArray;
    // List of paths to the datasets that the experiment is to be executed on.
    private ArrayList<String> dsPaths = new ArrayList<>(100);
    // List of CombinedMetric objects for distance calculations that correspond
    // to different datasets. Different datasets might require different
    // metrics to be used, so this is why it is necessary to explicitly specify
    // a metric for each dataset.
    private ArrayList<CombinedMetric> dsMetric = new ArrayList<>(100);
    private ArrayList<Integer>[][][] allDataSetFolds;
    // An alternative specification to the fold specification above, used for
    // OpenML compatibility.
    public ArrayList<Integer>[][][][] trainTestIndexes;
    // Folds to use for testing on the dataset.
    private ArrayList<Integer>[][] dsFolds;
    // Class names are kept track of only for OpenML data sources, as the labels
    // need to be specified in their original form (not just the indexes) when
    // the predictions are uploaded. Class name arrays for non-OpenML data
    // sources will be null in this structure.
    String[][] classNames;
    // DataSet objects.
    private DataSet originalDSet, currDSet;
    // The discretized data object.
    private DiscretizedDataSet currDiscDSet;
    // Current CombinedMetric object.
    private CombinedMetric cmet;
    // The current distance matrix, given as an upper triangular distance matrix
    // so that each row i contains only the distances to j > i, encoded so that
    // distMat[i][j] is the distance between i and i + j + 1. This is the
    // standard matrix format throughout this library.
    private float[][] distMat;
    // A boolean flag array indicating which algorithms require discretized
    // data objects.
    private boolean[] isDiscreteAlgorithm;
    // Whether there is a need for discretizing the data.
    private boolean discreteExists = false;
    // A label array prior to any mislabeling. It is used for evaluation, as
    // classifiers that were trained on mislabeled data need to be evaluated on
    // the exact labels.
    private int[] originalLabels;
    // Integer that is the number of categories in the data.
    private int numCategories;
    // Whether we are in the multi-label experimental mode, where we are testing
    // different representations of the same underlying dataset that has
    // multiple label distributions / classification problems defined on top.
    private boolean multiLabelMode = false;
    // The number of classification problems on the dataset.
    private int numDifferentLabelings = 1;
    // Separator in the label file.
    private String lsep;
    // Whether there are distance users in the continous or discretized case,
    // indicating whether there is a need for calculating the distance matrix
    // or not.
    private boolean distUserPresentDisc = false;
    private boolean distUserPresentNonDisc = false;
    // Whether there is a need to calculate the kNN sets.
    private boolean kNNUserPresent = false;
    // Whether to use the approximate kNN sets and which quality parameter to
    // use.
    private float alphaAppKNNs = 1f;
    private boolean approximateKNNs = false;
    // Current instance selector.
    private InstanceSelector selector = null;
    // Current selection rate.
    private float selectorRate = 0.1f;
    // Hubness estimation mode for the prototypes in instance selection. It can
    // be estimated from all the training data (retained and rejected), which
    // is an unbiased approach to hubness estimation in instance selection - or
    // it can be simply estimated from the selected prototype set, which is
    // simpler but introduces a bias in the estimates.
    private int protoHubnessMode = MultiCrossValidation.PROTO_UNBIASED;
    // The number of threads used for distance matrix and kNN set calculations.
    private int numCommonThreads = 8;
    // OpenML taskID-s and a map that checks whether a particular dataset is a
    // OpenML data source.
    public ArrayList<Integer> openMLTaskIDList;
    public HashMap<Integer, Integer> dataIndexToOpenMLCounterMap =
            new HashMap<>();
    private ExternalExperimentalContext contextObjects; 

    /**
     * Get the human-readable name of a secondary distance type.
     *
     * @param dType SecondaryDistance to get the name of.
     * @return String that is the secondary distance name.
     */
    private String getSecondaryDistanceName(SecondaryDistance dType) {
        switch (dType) {
            case SIMCOS: {
                return "simcos";
            }
            case SIMHUB: {
                return "simhub";
            }
            case MP: {
                return "mutualproximity";
            }
            case LS: {
                return "localscaling";
            }
            case NICDM: {
                return "nicdm";
            }
            default: {
                return "none";
            }
        }
    }

    /**
     * Initialization.
     *
     * @param inConfigFile File that contains the experimental configuration.
     */
    public BatchClassifierTester(File inConfigFile) {
        this.inConfigFile = inConfigFile;
    }

    /**
     * This method runs all the experiments that were specified in the
     * configuration.
     */
    public void runAllTests() throws Exception {
        int datasetIndex = 0;
        DataSet labelCol = null;
        if (multiLabelMode) {
            // Each label array is a column in the label file.
            if (inLabelFile.getPath().endsWith(".arff")) {
                IOARFF aPers = new IOARFF();
                labelCol = aPers.load(inLabelFile.getPath());
            } else if (inLabelFile.getPath().endsWith(".csv")) {
                IOCSV reader = new IOCSV(false, lsep,
                        DataMineConstants.INTEGER);
                labelCol = reader.readData(inLabelFile);
            } else {
                System.out.println("Wrong label format");
                throw new Exception();
            }
            // The number of different classification problems defined on top
            // of the data.
            numDifferentLabelings = labelCol.getNumIntAttr();
        }
        // Iterate over all data representations / datasets.
        for (String dsPath : dsPaths) {
            cmet = dsMetric.get(datasetIndex);
            File dsFile = new File(dsPath);
            originalDSet = SupervisedLoader.loadData(dsFile, multiLabelMode);
            System.out.println("Testing on: " + dsPath);
            // Make all category indexes be in the range [0 .. numCategores - 1]
            originalDSet.standardizeCategories();
            // Perform feature normalization, if specified.
            if (normType != Normalization.NONE) {
                System.out.print("Normalizing features-");
                if (normType == Normalization.NORM_01) {
                    // Normalization to the 0-1 range.
                    originalDSet.normalizeFloats();
                } else if (normType == Normalization.STANDARDIZE) {
                    // Feature standardization.
                    originalDSet.standardizeAllFloats();
                } else if (normType == Normalization.TFIDF) {
                    // TFIDF normalization.
                    boolean[] fBool;
                    if (originalDSet instanceof BOWDataSet) {
                        fBool = new boolean[((BOWDataSet) originalDSet).
                                getNumDifferentWords()];
                    } else {
                        fBool = new boolean[originalDSet.getNumFloatAttr()];
                    }
                    Arrays.fill(fBool, true);
                    TFIDF filterTFIDF = new TFIDF(fBool,
                            DataMineConstants.FLOAT);
                    if (originalDSet instanceof BOWDataSet) {
                        filterTFIDF.setSparse(true);
                    }
                    filterTFIDF.filter(originalDSet);
                }
                System.out.println("-Normalization complete.");
            } else {
                System.out.println("Skipping feature normalization.");
            }
            // Get the original label array.
            originalLabels = originalDSet.obtainLabelArray();
            // Get the number of classes in the data.
            numCategories = originalDSet.countCategories();

            // Initialize the discrete classifier flag array.
            for (int cIndex = 0; cIndex < classifierNames.size(); cIndex++) {
                String cName = classifierNames.get(cIndex);
                if (isDiscrete(cName)) {
                    discreteExists = true;
                    break;
                }
            }
            dsFolds = null;
            if (allDataSetFolds == null && foldsDir != null) {
                File foldsFile = new File(foldsDir,
                        dsFile.getName().substring(0, dsFile.getName().
                        lastIndexOf(".")) + "_cv_" + numTimes + "_" + numFolds +
                        ".json");
                if (foldsFile.exists()) {
                    System.out.println("Loading the existing folds from: " +
                            foldsFile.getPath());
                    dsFolds = CVFoldsIO.loadAllFolds(foldsFile);
                }
            } else {
                dsFolds = allDataSetFolds[datasetIndex];
            }
            int memCleanCount = 0;
            // Iterate over all the noise and mislabeling rates, for all the
            // label assignments (if in the multi-label mode).
            for (float noise = noiseMin; noise <= noiseMax;
                    noise += noiseStep) {
                for (int lIndex = 0; lIndex < numDifferentLabelings; lIndex++) {
                    for (float ml = mlMin; ml <= mlMax; ml += mlStep) {
                        if (++memCleanCount % 5 == 0) {
                            // Try initiating some clean-up periodically.
                            System.gc();
                        }
                        if (ml > 0 || noise > 0) {
                            // If some noise or mislabeling is to be applied,
                            // first make a copy of the original data.
                            currDSet = originalDSet.copy();
                        } else {
                            currDSet = originalDSet;
                        }
                        if (multiLabelMode && labelCol != null) {
                            // If in the multi-label mode, assign the
                            // appropriate labels to the data points.
                            for (int dInd = 0; dInd < currDSet.size(); dInd++) {
                                currDSet.data.get(dInd).setCategory(
                                        labelCol.getInstance(
                                        dInd).iAttr[lIndex]);
                            }
                            numCategories = currDSet.countCategories();
                            originalLabels = currDSet.obtainLabelArray();
                        }
                        if (ml > 0) {
                            // First check if any mislabeling instance weights
                            // were provided, that make certain mislabelings
                            // more probable than others.
                            String weightsPath = null;
                            if (mlWeightsDir != null) {
                                if (!(cmet instanceof SparseCombinedMetric)) {
                                    String metricDir = cmet.getFloatMetric()
                                            != null ?
                                            cmet.getFloatMetric().getClass().
                                            getName() : cmet.getIntegerMetric().
                                            getClass().getName();
                                    switch (normType) {
                                        case NONE:
                                            weightsPath = "NO" + File.separator
                                                    + metricDir + File.separator
                                                    + "ml_weights.txt";
                                            break;
                                        case NORM_01:
                                            weightsPath = "NORM01" +
                                                    File.separator + metricDir +
                                                    File.separator +
                                                    "ml_weights.txt";
                                            break;
                                        case STANDARDIZE:
                                            weightsPath = "STANDARDIZED" +
                                                    File.separator + metricDir +
                                                    File.separator +
                                                    "ml_weights.txt";
                                            break;
                                        case TFIDF:
                                            weightsPath = "TFIDF" +
                                                    File.separator + metricDir +
                                                    File.separator +
                                                    "ml_weights.txt";
                                            break;
                                    }
                                } else {
                                    switch (normType) {
                                        case NONE:
                                            weightsPath = "NO" + File.separator
                                                    + ((SparseCombinedMetric)
                                                    cmet).getSparseMetric().
                                                    getClass().getName() +
                                                    File.separator +
                                                    "ml_weights.txt";
                                            break;
                                        case NORM_01:
                                            weightsPath = "NORM01" +
                                                    File.separator +
                                                    ((SparseCombinedMetric)
                                                    cmet).getSparseMetric().
                                                    getClass().getName() +
                                                    File.separator +
                                                    "ml_weights.txt";
                                            break;
                                        case STANDARDIZE:
                                            weightsPath = "STANDARDIZED" +
                                                    File.separator +
                                                    ((SparseCombinedMetric)
                                                    cmet).getSparseMetric().
                                                    getClass().getName() +
                                                    File.separator +
                                                    "ml_weights.txt";
                                            break;
                                        case TFIDF:
                                            weightsPath = "TFIDF" +
                                                    File.separator +
                                                    ((SparseCombinedMetric)
                                                    cmet).getSparseMetric().
                                                    getClass().getName() +
                                                    File.separator +
                                                    "ml_weights.txt";
                                            break;
                                    }
                                }
                                File inWeightFile = new File(mlWeightsDir,
                                        weightsPath);
                                 try (BufferedReader br = new BufferedReader(
                                         new InputStreamReader(
                                         new FileInputStream(inWeightFile)));) {
                                     String[] weightStrs = br.readLine().split(
                                             " ");
                                     float[] mlWeights = new float[
                                             weightStrs.length];
                                     for (int i = 0; i < weightStrs.length;
                                             i++) {
                                         mlWeights[i] = Float.parseFloat(
                                                 weightStrs[i]);
                                     }
                                     currDSet.induceWeightProportionalMislabeling(
                                             ml, numCategories, mlWeights);
                                 }
                            } else {
                                // Induce the specified mislabeling rate.
                                currDSet.induceMislabeling(ml, numCategories);
                            }
                        }
                        if (noise > 0) {
                            // Induce Gaussian featue noise.
                            currDSet.addGaussianNoiseToNormalizedCollection(
                                    noise, 0.1f);
                        }
                        if (discreteExists) {
                            // Make a discretized version of the original data.
                            currDiscDSet = new DiscretizedDataSet(currDSet);
                            EntropyMDLDiscretizer discretizer =
                                    new EntropyMDLDiscretizer(
                                    currDSet, currDiscDSet, numCategories);
                            discretizer.discretizeAll();
                            currDiscDSet.discretizeDataSet(currDSet);
                        }
                        for (int k = kMax; k >= kMin; k -= kStep) {
                            // Iterate over different neighborhood sizes.
                            if (multiLabelMode) {
                                currOutDSDir = new File(outDir,
                                        dsFile.getName().substring(0,
                                        dsFile.getName().lastIndexOf(".")) +
                                        "L" + lIndex + File.separator + "k" +
                                        k + File.separator + "ml" + ml +
                                        File.separator + "noise" + noise);
                            } else {
                                currOutDSDir = new File(outDir,
                                        dsFile.getName().substring(0,
                                        dsFile.getName().lastIndexOf(".")) +
                                        File.separator + "k" + k +
                                        File.separator + "ml" + ml +
                                        File.separator + "noise" + noise);
                            }
                            FileUtil.createDirectory(currOutDSDir);
                            isDiscreteAlgorithm =
                                    new boolean[classifierNames.size()];
                            // Initialize algorithm lists.
                            ArrayList<ValidateableInterface> nonDiscreteAlgs =
                                    new ArrayList<>(20);
                            ArrayList<ValidateableInterface> discreteAlgs =
                                    new ArrayList<>(20);

                            for (int cIndex = 0; cIndex <
                                    classifierNames.size(); cIndex++) {
                                // Place the algorithm in the appropriate list.
                                String cName = classifierNames.get(cIndex);
                                ValidateableInterface cInstance;
                                if (algorithmParametrizationMap.containsKey(
                                        cName)) {
                                    cInstance = getClassifierForName(cName,
                                            cIndex, numCategories, cmet, k,
                                            algorithmParametrizationMap.get(
                                            cName));
                                } else {
                                    cInstance = getClassifierForName(cName,
                                            cIndex, numCategories, cmet, k,
                                            null);
                                }
                                if (cInstance instanceof DiscreteClassifier) {
                                    isDiscreteAlgorithm[cIndex] = true;
                                }
                                if (isDiscreteAlgorithm[cIndex]) {
                                    discreteAlgs.add(cInstance);
                                } else {
                                    nonDiscreteAlgs.add(cInstance);
                                }
                            }
                            discreteArray =
                                    new ValidateableInterface[
                                            discreteAlgs.size()];
                            if (discreteArray.length > 0) {
                                discreteArray = discreteAlgs.toArray(
                                        discreteArray);
                            }
                            nonDiscreteArray =
                                    new ValidateableInterface[
                                            nonDiscreteAlgs.size()];
                            if (nonDiscreteArray.length > 0) {
                                nonDiscreteArray = nonDiscreteAlgs.toArray(
                                        nonDiscreteArray);
                            }
                            // Check for distance matrix users.
                            for (int cIndex = 0; cIndex < discreteArray.length;
                                    cIndex++) {
                                if (discreteArray[cIndex] instanceof
                                        DistMatrixUserInterface) {
                                    distUserPresentDisc = true;
                                    break;
                                }
                            }
                            for (int cIndex = 0; cIndex <
                                    nonDiscreteArray.length; cIndex++) {
                                if (nonDiscreteArray[cIndex] instanceof
                                        DistMatrixUserInterface) {
                                    distUserPresentNonDisc = true;
                                    break;
                                }
                                if (nonDiscreteArray[cIndex] instanceof
                                        NeighborPointsQueryUserInterface ||
                                        nonDiscreteArray[cIndex] instanceof
                                        NSFUserInterface ) {
                                    kNNUserPresent = true;
                                    break;
                                }
                            }
                            MultiCrossValidation discreteCV = null;
                            MultiCrossValidation nonDiscreteCV;
                            if (distUserPresentDisc || distUserPresentNonDisc ||
                                    kNNUserPresent) {
                                // Load or calculate the distance matrix.
                                String dMatPath = null;
                                // Calculate the appropriate distance matrix
                                // path.
                                if (distancesDir != null) {
                                    if (!(cmet instanceof SparseCombinedMetric)) {
                                        String metricDir =
                                                cmet.getFloatMetric() != null ?
                                                cmet.getFloatMetric().
                                                getClass().getName() :
                                                cmet.getIntegerMetric().
                                                getClass().getName();
                                        switch (normType) {
                                            case NONE:
                                                dMatPath = "NO" + File.separator
                                                        + metricDir +
                                                        File.separator +
                                                        "dMat.txt";
                                                break;
                                            case NORM_01:
                                                dMatPath = "NORM01" +
                                                        File.separator +
                                                        metricDir +
                                                        File.separator +
                                                        "dMat.txt";
                                                break;
                                            case STANDARDIZE:
                                                dMatPath = "STANDARDIZED" +
                                                        File.separator +
                                                        metricDir +
                                                        File.separator +
                                                        "dMat.txt";
                                                break;
                                            case TFIDF:
                                                dMatPath = "TFIDF" +
                                                        File.separator +
                                                        metricDir +
                                                        File.separator +
                                                        "dMat.txt";
                                                break;
                                        }
                                    } else {
                                        switch (normType) {
                                            case NONE:
                                                dMatPath = "NO" +
                                                        File.separator +
                                                        ((SparseCombinedMetric)
                                                        cmet).getSparseMetric().
                                                        getClass().getName() +
                                                        File.separator +
                                                        "dMat.txt";
                                                break;
                                            case NORM_01:
                                                dMatPath = "NORM01" +
                                                        File.separator +
                                                        ((SparseCombinedMetric)
                                                        cmet).getSparseMetric().
                                                        getClass().getName() +
                                                        File.separator +
                                                        "dMat.txt";
                                                break;
                                            case STANDARDIZE:
                                                dMatPath = "STANDARDIZED" +
                                                        File.separator +
                                                        ((SparseCombinedMetric)
                                                        cmet).getSparseMetric().
                                                        getClass().getName() +
                                                        File.separator +
                                                        "dMat.txt";
                                                break;
                                            case TFIDF:
                                                dMatPath = "TFIDF" +
                                                        File.separator +
                                                        ((SparseCombinedMetric)
                                                        cmet).getSparseMetric().
                                                        getClass().getName() +
                                                        File.separator +
                                                        "dMat.txt";
                                                break;
                                        }
                                    }
                                }
                                File dMatFile = null;
                                Class cmetClass = originalDSet.getClass();
                                if (dMatPath != null && noise == 0) {
                                    dMatFile = new File(distancesDir,
                                            dsFile.getName().substring(0,
                                            dsFile.getName().lastIndexOf(".")) +
                                            File.separator + dMatPath);
                                    cmetClass = Class.forName(dMatFile.
                                            getParentFile().getName());
                                }
                                if (contextObjects == null) {
                                    contextObjects =
                                            new ExternalExperimentalContext();
                                }
                                if (distMat == null) {
                                    if (dMatFile == null || !dMatFile.exists()
                                            || !(cmetClass.isInstance(
                                            cmet.getFloatMetric()))) {
                                        System.out.print(
                                                "Calculating distances-");
                                        distMat = currDSet.
                                                calculateDistMatrixMultThr(
                                                cmet, 8);
                                        System.out.println(
                                                "-distance calculated.");
                                        if (dMatFile != null) {
                                            DistanceMatrixIO.printDMatToFile(
                                                    distMat, dMatFile);
                                        }
                                    } else {
                                        System.out.print("Loading distances-");
                                        distMat =
                                                DistanceMatrixIO.
                                                loadDMatFromFile(dMatFile);
                                        System.out.println(
                                                "-distance loaded from file: " +
                                                dMatFile.getPath());
                                    }
                                }
                                contextObjects.setDistances(distMat);
                                if (discreteExists) {
                                    discreteCV =
                                            new MultiCrossValidation(
                                                    numTimes,
                                                    numFolds,
                                                    numCategories,
                                                    currDiscDSet,
                                                    currDiscDSet.data,
                                                    discreteArray,
                                                    distMat);
                                    if (trainTestIndexes == null ||
                                            trainTestIndexes[datasetIndex] ==
                                            null) {
                                        discreteCV.setAllFolds(dsFolds);
                                    } else {
                                        discreteCV.setTrainTestIndexes(
                                                trainTestIndexes[datasetIndex]);
                                    }
                                    discreteCV.setExternalContext(
                                            contextObjects);
                                }
                                nonDiscreteCV =
                                        new MultiCrossValidation(
                                                numTimes,
                                                numFolds,
                                                numCategories,
                                                currDSet,
                                                currDSet.data,
                                                nonDiscreteArray, distMat);
                                if (trainTestIndexes == null ||
                                            trainTestIndexes[datasetIndex] ==
                                            null) {
                                    nonDiscreteCV.setAllFolds(dsFolds);
                                } else {
                                    nonDiscreteCV.setTrainTestIndexes(
                                            trainTestIndexes[datasetIndex]);
                                }
                                nonDiscreteCV.setExternalContext(
                                        contextObjects);
                            } else {
                                if (discreteExists) {
                                    discreteCV =
                                            new MultiCrossValidation(
                                                    numTimes,
                                                    numFolds,
                                                    numCategories,
                                                    currDiscDSet,
                                                    currDiscDSet.data,
                                                    discreteArray);
                                    if (trainTestIndexes == null ||
                                            trainTestIndexes[datasetIndex] ==
                                            null) {
                                        discreteCV.setAllFolds(dsFolds);
                                    } else {
                                        discreteCV.setTrainTestIndexes(
                                                trainTestIndexes[datasetIndex]);
                                    }
                                }
                                nonDiscreteCV =
                                        new MultiCrossValidation(
                                                numTimes,
                                                numFolds,
                                                numCategories,
                                                currDSet,
                                                currDSet.data,
                                                nonDiscreteArray);
                                if (trainTestIndexes == null ||
                                            trainTestIndexes[datasetIndex] ==
                                            null) {
                                    nonDiscreteCV.setAllFolds(dsFolds);
                                } else {
                                    nonDiscreteCV.setTrainTestIndexes(
                                            trainTestIndexes[datasetIndex]);
                                }
                            }
                            // First handle the discretized case.
                            if (discreteExists && discreteCV != null) {
                                discreteCV.validateOnSeparateLabelArray(
                                        originalLabels);
                                discreteCV.useMultipleCommonThreads(
                                        numCommonThreads);
                            }
                            nonDiscreteCV.validateOnSeparateLabelArray(
                                    originalLabels);
                            nonDiscreteCV.useMultipleCommonThreads(
                                        numCommonThreads);
                            if (approximateKNNs) {
                                // In case of approximate kNN set calculations.
                                if (discreteExists && discreteCV != null) {
                                    discreteCV.setApproximateKNNParams(
                                            approximateKNNs, alphaAppKNNs);
                                }
                                nonDiscreteCV.setApproximateKNNParams(
                                        approximateKNNs, alphaAppKNNs);
                            }
                            // Performance estimators.
                            ClassificationEstimator[] avgDiscrete = null;
                            ClassificationEstimator[][] allDiscreteEstimates =
                                    null;
                            if (discreteExists && discreteCV != null) {
                                // Initialize the cross-validation object.
                                avgDiscrete = null;
                                allDiscreteEstimates = null;
                                if (discreteArray.length > 0) {
                                    discreteCV.setKValue(k);
                                    discreteCV.setCombinedMetric(cmet);
                                    discreteCV.setProtoHubnessMode(
                                            protoHubnessMode);
                                    if (selector != null) {
                                        discreteCV.setDataReducer(
                                                selector, selectorRate);
                                    }
                                    // Perform the tests.
                                    discreteCV.performAllTests();
                                    System.out.println();
                                    if (foldsDir != null) {
                                        File foldsFile = new File(foldsDir,
                                                dsFile.getName().substring(0,
                                                dsFile.getName().lastIndexOf(
                                                ".")) + "_cv_" + numTimes +
                                                "_" + numFolds + ".json");
                                        dsFolds = discreteCV.getAllFolds();
                                        // We over-write in any case, as the
                                        // folds might have been loaded
                                        // externally via OpenML and we might
                                        // like to save them for future off-line
                                        // testing.
                                        System.out.println("Saving the folds "
                                                + "to:" + foldsFile.getPath());
                                        CVFoldsIO.saveAllFolds(dsFolds,
                                                numTimes, numFolds, foldsFile);
                                    }
                                    avgDiscrete =
                                            discreteCV.getAverageResults();
                                    allDiscreteEstimates =
                                            discreteCV.getEstimators();
                                }
                            }
                            // Now the continuous case.
                            ClassificationEstimator[] avgNonDiscrete = null;
                            ClassificationEstimator[][] allNonDiscreteEstimates
                                    = null;
                            if (nonDiscreteArray.length > 0) {
                                // Initialize the cross-validation object.
                                nonDiscreteCV.setKValue(k);
                                nonDiscreteCV.setCombinedMetric(cmet);
                                nonDiscreteCV.setProtoHubnessMode(
                                        protoHubnessMode);
                                if (selector != null) {
                                    nonDiscreteCV.setDataReducer(
                                            selector, selectorRate);
                                }
                                if (secondaryDistanceType !=
                                        SecondaryDistance.NONE) {
                                    nonDiscreteCV.useSecondaryDistances(
                                            secondaryDistanceType,
                                            secondaryDistanceK);
                                }
                                // Perform the tests.
                                nonDiscreteCV.performAllTests();
                                System.out.print("tEstTraining: " + (long) (
                                        nonDiscreteCV.execTimeAllOneRun));
                                System.out.println();
                                if (foldsDir != null) {
                                    File foldsFile = new File(foldsDir,
                                            dsFile.getName().substring(0,
                                            dsFile.getName().lastIndexOf(".")) +
                                            "_cv_" + numTimes + "_" + numFolds +
                                            ".json");
                                    dsFolds = nonDiscreteCV.getAllFolds();
                                    // We over-write in any case, as the folds
                                    // might have been loaded externally via
                                    // OpenML and we might like to save them
                                    // for future off-line testing.
                                    System.out.println("Saving the folds "
                                                + "to:" + foldsFile.getPath());
                                    CVFoldsIO.saveAllFolds(dsFolds, numTimes,
                                            numFolds, foldsFile);
                                }
                                avgNonDiscrete =
                                        nonDiscreteCV.getAverageResults();
                                allNonDiscreteEstimates =
                                        nonDiscreteCV.getEstimators();
                            }
                            int discreteCounter = -1;
                            int nonDiscreteCounter = -1;
                            float[][][][] allFuzzyPredictionsNonDisc = null;
                            if (nonDiscreteCV != null) {
                                allFuzzyPredictionsNonDisc =
                                    nonDiscreteCV.getAllFuzzyLabelAssignments();
                                if (dataIndexToOpenMLCounterMap.containsKey(
                                        datasetIndex)) {
                                    for (int algIndex = 0; algIndex < 
                                            nonDiscreteArray.length; algIndex++) {
                                        ClassificationResultHandler handler =
                                                new ClassificationResultHandler
                                                (openmlConnector.getConnector(),
                                                hubMinerSourceDir, currDSet,
                                                classNames[datasetIndex]);
                                        handler.uploadClassificationResults(
                                                openmlConnector.getTaskForTaskID(
                                                    openMLTaskIDList.get(
                                                    dataIndexToOpenMLCounterMap.get(
                                                    datasetIndex))),
                                                    nonDiscreteArray[algIndex],
                                                algIndex,
                                                ClassifierParametrization.
                                                    getClassifierParameterStringValues(
                                                    nonDiscreteArray[algIndex]),
                                                numTimes,
                                                numFolds,
                                                trainTestIndexes[datasetIndex],
                                                allFuzzyPredictionsNonDisc);
                                    }
                                }
                            }
                            float[][][][] allFuzzyPredictionsDisc = null;
                            if (discreteCV != null) {
                                allFuzzyPredictionsDisc =
                                    discreteCV.getAllFuzzyLabelAssignments();
                                if (dataIndexToOpenMLCounterMap.containsKey(
                                        datasetIndex)) {
                                    for (int algIndex = 0; algIndex < 
                                            discreteArray.length; algIndex++) {
                                        ClassificationResultHandler handler =
                                                new ClassificationResultHandler
                                                (openmlConnector.getConnector(),
                                                hubMinerSourceDir, currDSet,
                                                classNames[datasetIndex]);
                                        handler.uploadClassificationResults(
                                                openmlConnector.getTaskForTaskID(
                                                    openMLTaskIDList.get(
                                                    dataIndexToOpenMLCounterMap.get(
                                                    datasetIndex))),
                                                    discreteArray[algIndex],
                                                algIndex,
                                                ClassifierParametrization.
                                                    getClassifierParameterStringValues(
                                                    discreteArray[algIndex]),
                                                numTimes,
                                                numFolds,
                                                trainTestIndexes[datasetIndex],
                                                allFuzzyPredictionsDisc);
                                    }
                                }
                            }
                            float[][][] averageFuzzyPredictions =
                                    new float[classifierNames.size()][
                                            currDSet.size()][numCategories];
                            // Persist the average classifier performance
                            // indicators.
                            for (int cIndex = 0; cIndex <
                                    classifierNames.size(); cIndex++) {
                                File outAlgDir;
                                if (secondaryDistanceType ==
                                        SecondaryDistance.NONE) {
                                    outAlgDir = new File(currOutDSDir,
                                            classifierNames.get(cIndex));
                                } else {
                                    outAlgDir = new File(currOutDSDir,
                                            classifierNames.get(cIndex) +
                                            getSecondaryDistanceName(
                                            secondaryDistanceType));
                                }
                                if (isDiscreteAlgorithm[cIndex]) {
                                    discreteCounter++;
                                    for (int i = 0; i < currDSet.size(); i++) {
                                        for (int c = 0; c < numCategories;
                                                c++) {
                                            for (int rep = 0; rep < numTimes;
                                                    rep++) {
                                                averageFuzzyPredictions[cIndex][
                                                        i][c] +=
                                                        allFuzzyPredictionsDisc[
                                                        discreteCounter][
                                                        rep][i][c];
                                            }
                                        }
                                    }
                                    for (int i = 0; i < currDSet.size(); i++) {
                                        for (int c = 0; c < numCategories;
                                                c++) {
                                            averageFuzzyPredictions[cIndex][i][
                                                    c] /= numTimes;
                                        }
                                    }
                                    avgDiscrete[discreteCounter].
                                            printEstimatorToFile(new File(
                                            outAlgDir, "avg.txt"));
                                    ClassificationEstimator.
                                            printMainPointsOfEstimatorsToFile(
                                            allDiscreteEstimates[
                                            discreteCounter], new File(
                                            outAlgDir, "allSummed.txt"));
                                    ClassificationEstimator.
                                            appendMainPointsStDevsToFile(
                                            allDiscreteEstimates[
                                            discreteCounter], new File(
                                            outAlgDir, "avg.txt"));
                                    for (int j = 0; j < allDiscreteEstimates[
                                            discreteCounter].length; j++) {
                                        allDiscreteEstimates[
                                                discreteCounter][j].
                                                printEstimatorToFile(
                                                new File(new File(
                                                outAlgDir, "all"), "run" + j +
                                                ".txt"));
                                    }
                                } else {
                                    nonDiscreteCounter++;
                                    for (int i = 0; i < currDSet.size(); i++) {
                                        for (int c = 0; c < numCategories;
                                                c++) {
                                            for (int rep = 0; rep < numTimes;
                                                    rep++) {
                                                averageFuzzyPredictions[cIndex][
                                                        i][c] +=
                                                        allFuzzyPredictionsNonDisc[
                                                        nonDiscreteCounter][
                                                        rep][i][c];
                                            }
                                        }
                                    }
                                    for (int i = 0; i < currDSet.size(); i++) {
                                        for (int c = 0; c < numCategories;
                                                c++) {
                                            averageFuzzyPredictions[cIndex][i][
                                                    c] /= numTimes;
                                        }
                                    }
                                    avgNonDiscrete[nonDiscreteCounter].
                                            printEstimatorToFile(
                                            new File(outAlgDir, "avg.txt"));
                                    ClassificationEstimator.
                                            printMainPointsOfEstimatorsToFile(
                                            allNonDiscreteEstimates[
                                            nonDiscreteCounter], new File(
                                            outAlgDir, "allSummed.txt"));
                                    ClassificationEstimator.
                                            appendMainPointsStDevsToFile(
                                            allNonDiscreteEstimates[
                                            nonDiscreteCounter], new File(
                                            outAlgDir, "avg.txt"));
                                    for (int j = 0; j < allNonDiscreteEstimates[
                                            nonDiscreteCounter].length; j++) {
                                        allNonDiscreteEstimates[
                                                nonDiscreteCounter][j].
                                                printEstimatorToFile(new File(
                                                new File(outAlgDir, "all"),
                                                "run" + j + ".txt"));
                                    }
                                }
                                // Persist all fuzzy classifications as json.
                                Gson gson = new Gson();
                                String jsonAssignmentsString = gson.toJson(
                                        averageFuzzyPredictions[cIndex],
                                        float[][].class);
                                File outFuzzyLabelAssignmentsFile = new File(
                                        outAlgDir,
                                        "avgProbClassAssignments.json");
                                FileUtil.createFile(
                                        outFuzzyLabelAssignmentsFile);
                                try (PrintWriter pw = new PrintWriter(
                                        new FileWriter(
                                        outFuzzyLabelAssignmentsFile))) {
                                    pw.write(jsonAssignmentsString);
                                } catch (Exception e) {
                                    throw e;
                                }   
                            }
                        }
                    }
                }
                distMat = null;
                contextObjects = null;
            }
            datasetIndex++;
            System.gc();
        }
        if (summaryDir != null) {
            BatchStatSummarizer summarizer =
                    new BatchStatSummarizer(outDir, summaryDir, numTimes,
                    numFolds);
            summarizer.summarize();
        }
    }

    /**
     * This method checks if an algorithm name corresponds to a discrete
     * algorithm.
     * @param algName String that is the algorithm name.
     * @return True if it is a discrete classifier, false otherwise.
     */
    public boolean isDiscrete(String algName) {
        ClassifierFactory cfact = new ClassifierFactory();
        return cfact.getClassifierForName(algName, 2,
                CombinedMetric.EUCLIDEAN, 5) instanceof DiscreteClassifier;
    }

    
    /**
     * This method sets the appropriate instance selection method for the
     * specified name.
     * @param name String that is the name of the instance selector to use.
     */
    public void setInstanceSelector(String name) {
        selector = ReducersFactory.getReducerForName(name);
        System.out.println("instance selector set to:" + name);
    }

    /**
     * This method instantiates an algorithm instance for the specified name
     * with the provided parameters.
     * 
     * @param algName String that is the algorithm name.
     * @param index Integer that is the internal algorithm index.
     * @param numClasses Integer that is the number of classes in the data.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Integer representing the neighborhood size.
     * @param algorithmParametrization HashMap<String, Object> representing the
     * parameter values to set to the algorithm.
     * @return 
     */
    private ValidateableInterface getClassifierForName(String algName,
            int index, int numClasses, CombinedMetric cmet, int k,
            HashMap<String, Object> algorithmParametrization) {
        System.out.print("testing " + algName);
        ClassifierFactory cfact = new ClassifierFactory();
        ValidateableInterface classAlg = cfact.getClassifierForName(algName,
                numClasses, cmet, k);
        try {
            ClassifierParametrization.setParameterValuesToClassifier(classAlg,
                    algorithmParametrization);
            if (algorithmParametrization != null) {
                Set<String> paramNames = algorithmParametrization.keySet();
                System.out.print(" with ");
                for (String paramName: paramNames) {
                    System.out.print(paramName + "=" +
                            algorithmParametrization.get(paramName) + " ");
                }
            }
        } catch (NoSuchFieldException | IllegalArgumentException |
                IllegalAccessException ex) {
            Logger.getLogger(
                    BatchClassifierTester.class.getName()).log(
                    Level.SEVERE, null, ex);
        }
        System.out.println();
        isDiscreteAlgorithm[index] = classAlg instanceof DiscreteClassifier;
        return classAlg;
    }
    
    /**
     * This method loads the configuration from the configuration object.
     * 
     * @param conf BatchClassifierConfig that is the configuration object.
     */
    public void loadFromConfigurationObject(BatchClassifierConfig conf) {
        if (conf == null) {
            return;
        }
        secondaryDistanceType = conf.secondaryDistanceType;
        secondaryDistanceK = conf.secondaryDistanceK;
        normType = conf.normType;
        numTimes = conf.numTimes;
        numFolds = conf.numFolds;
        kMin = conf.kMin;
        kMax = conf.kMax;
        kStep = conf.kStep;
        noiseMin = conf.noiseMin;
        noiseMax = conf.noiseMax;
        noiseStep = conf.noiseStep;
        mlMin = conf.mlMin;
        mlMax = conf.mlMax;
        mlStep = conf.mlStep;
        inDir = conf.inDir;
        outDir = conf.outDir;
        summaryDir = conf.summaryDir;
        inLabelFile = conf.inLabelFile;
        distancesDir = conf.distancesDir;
        mlWeightsDir = conf.mlWeightsDir;
        foldsDir = conf.foldsDir;
        classifierNames = conf.classifierNames;
        dsPaths = conf.dsPaths;
        dsMetric = conf.dsMetric;
        allDataSetFolds = conf.allDataSetFolds;
        multiLabelMode = conf.multiLabelMode;
        numDifferentLabelings = conf.numDifferentLabelings;
        lsep = conf.lsep;
        alphaAppKNNs = conf.alphaAppKNNs;
        approximateKNNs = conf.approximateKNNs;
        selector = conf.selector;
        selectorRate = conf.selectorRate;
        protoHubnessMode = conf.protoHubnessMode;
        numCommonThreads = conf.numCommonThreads;
        openmlConnector = conf.getOpenMLConnector();
        trainTestIndexes = conf.trainTestIndexes;
        hubMinerSourceDir = conf.hubMinerSourceDir;
        algorithmParametrizationMap = conf.algorithmParametrizationMap;
        openMLTaskIDList = conf.openMLTaskIDList;
        dataIndexToOpenMLCounterMap = conf.dataIndexToOpenMLCounterMap;
        classNames = conf.classNames;
    }

    /**
     * This method loads the experimental configuration from the configuration
     * file.
     * @throws Exception 
     */
    public void loadParameters() throws Exception {
        BatchClassifierConfig config = new BatchClassifierConfig();
        config.loadParameters(inConfigFile);
        loadFromConfigurationObject(config);
    }

    /**
     * The main method that runs the entire experiment.
     * @param args Command line arguments - a single parameter that is the
     * path of the experimental configuration file.
     * @throws Exception 
     */
    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            System.out.println("1 parameter - file with test configuration");
            return;
        }
        File inConfigFile = new File(args[0]);
        BatchClassifierTester tester = new BatchClassifierTester(inConfigFile);
        tester.loadParameters();
        tester.runAllTests();
    }
}
