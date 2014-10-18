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
package learning.unsupervised.evaluation;

import configuration.BatchClusteringConfig;
import data.neighbors.NeighborSetFinder;
import data.neighbors.approximate.AppKNNGraphLanczosBisection;
import data.representation.DataInstance;
import data.representation.DataSet;
import data.representation.discrete.DiscretizedDataSet;
import data.representation.discrete.tranform.EntropyMDLDiscretizer;
import data.representation.sparse.BOWDataSet;
import data.representation.util.DataMineConstants;
import distances.kernel.Kernel;
import distances.kernel.KernelMatrixUserInterface;
import distances.primary.CombinedMetric;
import distances.sparse.SparseCombinedMetric;
import feature.evaluation.Info;
import filters.TFIDF;
import ioformat.FileUtil;
import ioformat.IOARFF;
import ioformat.IOCSV;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.ArrayList;
import learning.supervised.interfaces.DistMatrixUserInterface;
import data.neighbors.NSFUserInterface;
import data.neighbors.SharedNeighborFinder;
import distances.secondary.LocalScalingCalculator;
import distances.secondary.MutualProximityCalculator;
import distances.secondary.NICDMCalculator;
import distances.secondary.snd.SharedNeighborCalculator;
import ioformat.DistanceMatrixIO;
import learning.supervised.evaluation.cv.BatchClassifierTester.SecondaryDistance;
import learning.unsupervised.Cluster;
import learning.unsupervised.ClustererFactory;
import learning.unsupervised.ClusteringAlg;
import learning.unsupervised.evaluation.quality.QIndexDunn;
import learning.unsupervised.evaluation.quality.QIndexIsolation;
import learning.unsupervised.evaluation.quality.QIndexRand;
import learning.unsupervised.evaluation.quality.QIndexSilhouette;
import learning.unsupervised.methods.DBScan;

import probability.GaussianMixtureModel;
import probability.Perplexity;
import sampling.UniformSampler;

/**
 * This class implements the functionality for cross-algorithm clustering
 * comparisons on multiple datasets with multiple metrics, under possible
 * inclusion of feature noiseRate or on the exact feature vectors. Acceptable
 * data formats include ARFF and CSV. There is an option of splitting the data
 * to training and test sets automatically for evaluation as well - though it is
 * not strictly speaking necessary for avoiding over-fitting in clustering.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class BatchClusteringTester {

    // Whether to calculate model perplexity as a quality measure, as it can be
    // very slow for larger and high-dimensional datasets, prohibitively so.
    private boolean calculatePerplexity = false;
    // Whether to use split testing (training/test data).
    private boolean splitTesting = false;
    private float splitPerc = 1;
    // Whether a specific number of desired clusters was specified.
    private boolean nClustSpecified = true;
    private volatile int globalIterationCounter;
    // Applicable types of feature normalization.

    public enum Normalization {

        NONE, STANDARDIZE, NORM_01, TFIDF;
    }
    // The normalization type to actually use in the experiments.
    private Normalization normType = Normalization.STANDARDIZE;
    private boolean clustersAutoSet = false;
    private double execTimeAllOneRun; // In miliseconds.
    // The number of times a clustering is performed on every single dataset.
    private int timesOnDataSet = 30;
    // The preferred number of iterations, where applicable.
    private int minIter;
    private ArrayList<String> clustererNames = new ArrayList<>(10);
    // The experimental neighborhood range, with default values.
    private int kMin = 5, kMax = 5, kStep = 1;
    private int kGenMin = 1;
    // Noise and mislabeling experimental ranges, with default values.
    private float noiseMin = 0, noiseMax = 0, noiseStep = 1, mlMin = 0,
            mlMax = 0, mlStep = 1;
    // Files pointing to the input and output.
    private File inConfigFile, inDir, outDir, distancesDir, mlWeightsDir;
    // Paths to the tested datasets.
    private ArrayList<String> dsPaths = new ArrayList<>(100);
    // Paths to the corresponding metrics for the datasets.
    private ArrayList<CombinedMetric> dsMetric = new ArrayList<>(100);
    private DataSet originalDSet, currDSet;
    private DiscretizedDataSet currDiscDSet;
    private CombinedMetric currCmet;
    private int[] originalLabels;
    private int[] trainingLabels;
    private int[] testLabels;
    private int numCategories;
    private boolean[] discrete;
    private float[][] distMat;
    private float[][] kMat;
    private float[][] kernelDMat;
    private Kernel ker = null;
    // Interface user flags.
    private boolean distUserPresentNonDisc = false;
    private boolean distUserPresentDisc = false;
    private boolean kernelUserPresent = false;
    private boolean discreteExists;
    private boolean nsfUserPresent;
    private boolean kernelNSFUserPresent;
    // Secondary distance specification.
    private SecondaryDistance secondaryDistanceType;
    private int secondaryK = 50;
    // Clustering range.
    private int cluNumMin;
    private int cluNumMax;
    private int cluNumStep;
    // Aproximate kNN calculations specification.
    private float approximateNeighborsAlpha = 1f;
    private boolean approximateNeighbors = false;
    // The number of threads used for distance matrix and kNN set calculations.
    private int numCommonThreads = 8;

    /**
     * Reads the parameters from the configuration file.
     *
     * @param inConfigFile Configuration file containing all the parameters. The
     * exact format of the configuration file can be discerned from the
     * loadParameters() method in this class.
     */
    public BatchClusteringTester(File inConfigFile) {
        this.inConfigFile = inConfigFile;
    }

    public void runAllTests() throws Exception {
        // Index of the currently examined dataset.
        int dsCounter = 0;
        // We iterate over dataset paths.
        for (String dsPath : dsPaths) {
            // This loads the appropriate metrics object for this particular
            // dataset.
            currCmet = dsMetric.get(dsCounter);
            File dsFile = new File(dsPath);
            // It is possible to indicate that the data in the file is given in
            // sparse format by precluding the dataset name with "sparse:". This
            // code chops off the prefix and extracts the true data name and
            // loads the data in.
            if (dsPath.startsWith("sparse:")) {
                String trueDSPath = dsPath.substring(
                        dsPath.indexOf(':') + 1, dsPath.length());
                IOARFF pers = new IOARFF();
                originalDSet = pers.loadSparse(trueDSPath);
            } else {
                // This is for usual, dense data representations.
                if (dsPath.endsWith(".csv")) {
                    try {
                        // First we try reading it as if the data were
                        // comma-separated.
                        IOCSV reader = new IOCSV(true, ",");
                        originalDSet = reader.readData(dsFile);
                    } catch (Exception e) {
                        try {
                            // If not comma, then empty spaces.
                            IOCSV reader = new IOCSV(true, " +");
                            originalDSet = reader.readData(dsFile);
                        } catch (Exception e1) {
                            // Prior attempts have all assumed there was a
                            // category label as the last attribute. This
                            // one doesn't - the data is loaded without the
                            // labels.
                            try {
                                IOCSV reader = new IOCSV(false, ",");
                                originalDSet = reader.readData(dsFile);
                            } catch (Exception e2) {
                                IOCSV reader = new IOCSV(false, " +");
                                originalDSet = reader.readData(dsFile);
                            }
                        }
                    }
                } else if (dsPath.endsWith(".tsv")) {
                    // Similar as above, but now for .tsv files instead of .csv
                    try {
                        IOCSV reader = new IOCSV(true, " +");
                        originalDSet = reader.readData(dsFile);
                    } catch (Exception e) {
                        try {
                            IOCSV reader = new IOCSV(true, "\t");
                            originalDSet = reader.readData(dsFile);
                        } catch (Exception e1) {
                            try {
                                IOCSV reader = new IOCSV(false, " +");
                                originalDSet =
                                        reader.readData(dsFile);
                            } catch (Exception e2) {
                                IOCSV reader = new IOCSV(false, "\t");
                                originalDSet =
                                        reader.readData(dsFile);
                            }
                        }
                    }
                } else if (dsPath.endsWith(".arff")) {
                    // Similar as above, though now for .arff files.
                    IOARFF persister = new IOARFF();
                    originalDSet = persister.load(dsPath);
                } else {
                    // If everything fails, report an error.
                    System.out.println("Error, could not read: " + dsPath);
                    continue;
                }
            }
            System.out.println(" Testing on: " + dsPath);
            // Category standardization ensures the class labels are subsequent
            // integers, i.e. there are no 'holes' like 1,2,4,6,7. This would
            // have been standardized to 1,2,3,4,5.
            originalDSet.standardizeCategories();
            if (!nClustSpecified) {
                // This testing mode is for algorithms that can determine the
                // optimal cluster number on their own.
                clustersAutoSet = true;
            }
            if (clustersAutoSet) {
                // Those algorithms that do not determine the optimal number of
                // clusters on their own have the predefined cluster number set
                // to the number of categories in the data. The number of
                // clusters is set to 2 for those datasets that have no labels.
                int numCat = originalDSet.countCategories();
                cluNumMin = Math.max(numCat, 2);
                cluNumMax = Math.max(numCat, 2);
                cluNumStep = 1;
            }
            System.out.print(" Normalizing features-");
            // Apply the chosen feature normalization.
            if (normType == Normalization.NORM_01) {
                originalDSet.normalizeFloats();
            } else if (normType == Normalization.STANDARDIZE) {
                originalDSet.standardizeAllFloats();
            } else if (normType == Normalization.TFIDF) {
                boolean[] fBool;
                if (originalDSet instanceof BOWDataSet) {
                    // The sparse case.
                    fBool = new boolean[((BOWDataSet) originalDSet).
                            getNumDifferentWords()];
                } else {
                    // The dense case.
                    fBool = new boolean[originalDSet.getNumFloatAttr()];
                }
                Arrays.fill(fBool, true);
                TFIDF filterTFIDF = new TFIDF(fBool, DataMineConstants.FLOAT);
                if (originalDSet instanceof BOWDataSet) {
                    filterTFIDF.setSparse(true);
                }
                filterTFIDF.filter(originalDSet);
            }
            System.out.println("-normalization complete");
            // The original labels are stored in a separate arrau.
            originalLabels = originalDSet.obtainLabelArray();
            trainingLabels = originalLabels;
            numCategories = originalDSet.countCategories();
            for (int i = 0; i < clustererNames.size(); i++) {
                String cName = clustererNames.get(i);
                if (cName.equals("dbscan")) {
                    // DBScan requires a certain neighborhood size - so even
                    // if smaller max neighborhood sizes are requrested
                    // explicitly, implicitly a larger one might actually
                    // be calculated in the embedding NeighborSetFinder
                    // object - if DBScan is among the tested approaches.
                    nsfUserPresent = true;
                    kGenMin = 20;
                }
                if (isDiscrete(cName)) {
                    // Works on discretized representations.
                    discreteExists = true;
                }
                if (requiresNSF(cName)) {
                    // Requires neighbor sets to be provided.
                    nsfUserPresent = true;
                }
                if (requiresKernelNSF(cName)) {
                    // Requires neighbor sets calculated in the kernel space.
                    kernelNSFUserPresent = true;
                }
            }
            // The embedding NeighborSetFinder objects that will be used to
            // spawn the neighbor set holding objects that are to be passed
            // along to the clusterers at the proper time.
            NeighborSetFinder bigNSF = null;
            NeighborSetFinder bigNSFTest = null;
            NeighborSetFinder bigKernelNSF = null;
            NeighborSetFinder bigKernelNSFTest = null;
            // Counter for garbage collection invocations.
            int memCleanCount = 0;
            // Introducing feature noiseRate, if specified.
            for (float noise = noiseMin; noise <= noiseMax;
                    noise += noiseStep) {
                // Introducing mislabeling, if specified.
                for (float ml = mlMin; ml <= mlMax; ml += mlStep) {
                    if (++memCleanCount % 5 == 0) {
                        System.gc();
                    }
                    // If noiseRate and/or mislabeling are introduced to the
                    // data, we must first make a copy of the original data for
                    // later comparisons, as noiseRate is not introduced to test
                    // data and test labels.
                    if (ml > 0 || noise > 0) {
                        currDSet = originalDSet.copy();
                    } else {
                        currDSet = originalDSet;
                    }
                    if (ml > 0) {
                        // First check if any mislabeling instance weights
                        // were provided, that make certain mislabelings
                        // more probable than others.
                        String weightsPath = null;
                        if (mlWeightsDir != null) {
                            if (!(currCmet instanceof SparseCombinedMetric)) {
                                String metricDir = currCmet.getFloatMetric()
                                        != null
                                        ? currCmet.getFloatMetric().getClass().
                                        getName() : currCmet.getIntegerMetric().
                                        getClass().getName();
                                switch (normType) {
                                    case NONE:
                                        weightsPath = "NO" + File.separator
                                                + metricDir + File.separator
                                                + "ml_weights.txt";
                                        break;
                                    case NORM_01:
                                        weightsPath = "NORM01"
                                                + File.separator + metricDir
                                                + File.separator
                                                + "ml_weights.txt";
                                        break;
                                    case STANDARDIZE:
                                        weightsPath = "STANDARDIZED"
                                                + File.separator + metricDir
                                                + File.separator
                                                + "ml_weights.txt";
                                        break;
                                    case TFIDF:
                                        weightsPath = "TFIDF"
                                                + File.separator + metricDir
                                                + File.separator
                                                + "ml_weights.txt";
                                        break;
                                }
                            } else {
                                switch (normType) {
                                    case NONE:
                                        weightsPath = "NO" + File.separator
                                                + ((SparseCombinedMetric)
                                                currCmet).getSparseMetric().
                                                getClass().getName()
                                                + File.separator
                                                + "ml_weights.txt";
                                        break;
                                    case NORM_01:
                                        weightsPath = "NORM01"
                                                + File.separator
                                                + ((SparseCombinedMetric)
                                                currCmet).getSparseMetric().
                                                getClass().getName()
                                                + File.separator
                                                + "ml_weights.txt";
                                        break;
                                    case STANDARDIZE:
                                        weightsPath = "STANDARDIZED"
                                                + File.separator
                                                + ((SparseCombinedMetric)
                                                currCmet).getSparseMetric().
                                                getClass().getName()
                                                + File.separator
                                                + "ml_weights.txt";
                                        break;
                                    case TFIDF:
                                        weightsPath = "TFIDF"
                                                + File.separator
                                                + ((SparseCombinedMetric)
                                                currCmet).getSparseMetric().
                                                getClass().getName()
                                                + File.separator
                                                + "ml_weights.txt";
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
                                float[] mlWeights =
                                        new float[weightStrs.length];
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
                        currDSet.addGaussianNoiseToNormalizedCollection(
                                noise, 0.1f);
                    }
                    // We generate a discretized data set, if necessary.
                    if (discreteExists) {
                        currDiscDSet = new DiscretizedDataSet(currDSet);
                        EntropyMDLDiscretizer discretizer =
                                new EntropyMDLDiscretizer(
                                currDSet, currDiscDSet, numCategories);
                        discretizer.discretizeAll();
                        // Or possibly: discretizer.discretizeAllBinary();
                        currDiscDSet.discretizeDataSet(currDSet);
                    }
                    // We keep track of which algorithm is discrete and which
                    // works on continuous data.
                    discrete = new boolean[clustererNames.size()];
                    ArrayList<ClusteringAlg> nonDiscreteAlgs =
                            new ArrayList<>(20);
                    ArrayList<ClusteringAlg> discreteAlgs =
                            new ArrayList<>(20);
                    for (int i = 0; i < clustererNames.size(); i++) {
                        String cName = clustererNames.get(i);
                        ClusteringAlg cInstance = getClustererForName(
                                cName, currDSet, i, kMin, distMat,
                                null, null, null, numCategories);
                        if (discrete[i]) {
                            discreteAlgs.add(cInstance);
                        } else {
                            nonDiscreteAlgs.add(cInstance);
                        }
                    }
                    ClusteringAlg[] discreteArray =
                            new ClusteringAlg[discreteAlgs.size()];
                    if (discreteArray.length > 0) {
                        discreteArray = discreteAlgs.toArray(discreteArray);
                    }
                    ClusteringAlg[] nonDiscreteArray =
                            new ClusteringAlg[nonDiscreteAlgs.size()];
                    if (nonDiscreteArray.length > 0) {
                        nonDiscreteArray =
                                nonDiscreteAlgs.toArray(nonDiscreteArray);
                    }
                    // Here we check for which algorithms require distance
                    // matrices and/or kernel matrices to be calculated. If
                    // none require them, they won't be generated, thereby
                    // saving time.
                    for (int i = 0; i < discreteArray.length; i++) {
                        if (discreteArray[i] instanceof
                                DistMatrixUserInterface) {
                            distUserPresentDisc = true;
                            break;
                        }
                    }
                    for (int i = 0; i < nonDiscreteArray.length; i++) {
                        if (nonDiscreteArray[i] instanceof
                                DistMatrixUserInterface) {
                            distUserPresentNonDisc = true;
                            break;
                        }
                    }
                    for (int i = 0; i < discreteArray.length; i++) {
                        if (discreteArray[i] instanceof
                                KernelMatrixUserInterface) {
                            kernelUserPresent = true;
                            break;
                        }
                    }
                    for (int i = 0; i < nonDiscreteArray.length; i++) {
                        if (nonDiscreteArray[i] instanceof
                                KernelMatrixUserInterface) {
                            kernelUserPresent = true;
                            break;
                        }
                    }
                    if (distUserPresentDisc || distUserPresentNonDisc) {
                        // Here a distance matrix path is generated based on the
                        // distance name and the normalization name.
                        String dMatPath = null;
                        if (distancesDir != null) {
                            if (!(currCmet instanceof SparseCombinedMetric)) {
                                switch (normType) {
                                    case NONE:
                                        dMatPath = "NO"
                                                + File.separator
                                                + currCmet.getFloatMetric().
                                                getClass().getName()
                                                + File.separator + "dMat.txt";
                                        break;
                                    case NORM_01:
                                        dMatPath = "NORM01"
                                                + File.separator
                                                + currCmet.getFloatMetric().
                                                getClass().getName()
                                                + File.separator + "dMat.txt";
                                        break;
                                    case STANDARDIZE:
                                        dMatPath =
                                                "STANDARDIZED" + File.separator
                                                + currCmet.getFloatMetric().
                                                getClass().getName()
                                                + File.separator + "dMat.txt";
                                        break;
                                    case TFIDF:
                                        dMatPath = "TFIDF"
                                                + File.separator
                                                + currCmet.getFloatMetric().
                                                getClass().getName()
                                                + File.separator + "dMat.txt";
                                        break;
                                }
                            } else {
                                switch (normType) {
                                    case NONE:
                                        dMatPath = "NO"
                                                + File.separator
                                                + ((SparseCombinedMetric)
                                                currCmet).getSparseMetric().
                                                getClass().getName()
                                                + File.separator + "dMat.txt";
                                        break;
                                    case NORM_01:
                                        dMatPath = "NORM01"
                                                + File.separator
                                                + ((SparseCombinedMetric)
                                                currCmet).getSparseMetric().
                                                getClass().getName()
                                                + File.separator + "dMat.txt";
                                        break;
                                    case STANDARDIZE:
                                        dMatPath =
                                                "STANDARDIZED" + File.separator
                                                + ((SparseCombinedMetric)
                                                currCmet).getSparseMetric().
                                                getClass().getName()
                                                + File.separator + "dMat.txt";
                                        break;
                                    case TFIDF:
                                        dMatPath = "TFIDF"
                                                + File.separator
                                                + ((SparseCombinedMetric)
                                                currCmet).getSparseMetric().
                                                getClass().getName()
                                                + File.separator + "dMat.txt";
                                        break;
                                }
                            }
                        }
                        File dMatFile = null;
                        Class cmetClass = originalDSet.getClass();
                        // Just a dummy, so that no nullPointerExceptions get
                        // evoked later on if there is no path to the distance
                        // file.  
                        if (dMatPath != null) {
                            dMatFile = new File(distancesDir,
                                    dsFile.getName().substring(
                                    0, dsFile.getName().lastIndexOf("."))
                                    + File.separator + dMatPath);
                            cmetClass = Class.forName(
                                    dMatFile.getParentFile().getName());
                        }
                        if (distMat == null) {
                            if (dMatFile == null || !dMatFile.exists()
                                    || !(cmetClass.isInstance(
                                    currCmet.getFloatMetric()))) {
                                // Distances are not available and need to be
                                // calculated here.
                                System.out.print("Calculating distances-");
                                distMat = currDSet.calculateDistMatrixMultThr(
                                        currCmet, 8);
                                System.out.println("-distance calculated.");
                                if (dMatFile != null) {
                                    DistanceMatrixIO.printDMatToFile(
                                            distMat, dMatFile);
                                }
                            } else {
                                // They have previously been calculated and we
                                // load them here.
                                System.out.print("Loading distances-");
                                distMat = DistanceMatrixIO.loadDMatFromFile(
                                        dMatFile);
                                System.out.println(
                                        "-distance loaded from file: "
                                        + dMatFile.getPath());
                            }
                        }
                    }
                    if (kernelUserPresent) {
                        if (ker == null) {
                            // If none was specified, but some is required.
                            ker = new distances.kernel.MinKernel();
                        }
                        // Here we determine the path to the kernel matrix file.
                        String kerMatPath = "KERNEL" + File.separator
                                + ker.getClass().getName() + File.separator
                                + "kerMat.txt";
                        switch (normType) {
                            case NONE:
                                kerMatPath =
                                        "NO" + File.separator + "KERNEL"
                                        + File.separator
                                        + ker.getClass().getName()
                                        + File.separator + "kerMat.txt";
                                break;
                            case NORM_01:
                                kerMatPath = "NORM01" + File.separator
                                        + "KERNEL" + File.separator
                                        + ker.getClass().getName()
                                        + File.separator
                                        + "kerMat.txt";
                                break;
                            case STANDARDIZE:
                                kerMatPath = "STANDARDIZED" + File.separator
                                        + "KERNEL" + File.separator
                                        + ker.getClass().getName()
                                        + File.separator + "kerMat.txt";
                                break;
                            case TFIDF:
                                kerMatPath = "TFIDF" + File.separator
                                        + "KERNEL" + File.separator
                                        + ker.getClass().getName()
                                        + File.separator
                                        + "kerMat.txt";
                                break;
                        }
                        File kerMatFile = new File(
                                distancesDir, dsFile.getName().substring(0,
                                dsFile.getName().lastIndexOf("."))
                                + "\\" + kerMatPath);
                        Class kerClass = Class.forName(
                                kerMatFile.getParentFile().getName());
                        if (kMat == null) {
                            if (!kerMatFile.exists()
                                    || !(kerClass.isInstance(ker))) {
                                // We need to calculate the kernel matrix.
                                System.out.print("Calculating kernel-");
                                kMat = currDSet.calculateKernelMatrixMultThr(
                                        ker, 8);
                                System.out.println("-kernel calculated");
                                DistanceMatrixIO.printDMatToFile(
                                        kMat, kerMatFile);
                            } else {
                                // We can load the kernel matrix.
                                System.out.print("Loading kernel-");
                                kMat = DistanceMatrixIO.loadDMatFromFile(
                                        kerMatFile);
                                System.out.println("-kernel loaded from file: "
                                        + kerMatFile.getPath());
                            }
                        }
                        if (kernelNSFUserPresent) {
                            // NeighborSetFinder objects require distances and
                            // kernels are similarities of sorts - this is why
                            // a kernel distance matrix needs to be calculated
                            // from the kernel matrix.
                            kernelDMat = new float[kMat.length][];
                            for (int i = 0; i < kMat.length; i++) {
                                kernelDMat[i] = new float[kMat[i].length - 1];
                                for (int j = 1; j < kMat[i].length; j++) {
                                    kernelDMat[i][j - 1] = kMat[i][0]
                                            + kMat[j][0] - 2 * kMat[i][j];
                                }
                            }
                        }
                    }
                    // Auxiliary distance and kernel objects for training and
                    // testing.
                    float[][] trainingDist = null;
                    float[][] testDist = null;
                    float[][] trainingKernelMat = null;
                    float[][] testToTrainingKernel = null;
                    float[][] trainingKernelDist = null;
                    float[][] testKernelDist = null;
                    float[] selfKernelsTest = null;
                    DataSet trainingSet, testSet;
                    testSet = null;
                    // If we need a separate training and test set.
                    if (splitTesting) {
                        int testSize = (int) ((1 - splitPerc)
                                * currDSet.size());
                        // Sample the data.
                        int[] trainingIndexes, testIndexes;
                        do {
                            testIndexes =
                                    UniformSampler.getSample(
                                    currDSet.size(), testSize);
                            trainingIndexes = new int[currDSet.size()
                                    - testSize];
                            trainingSet = currDSet.cloneDefinition();
                            trainingSet.data = new ArrayList<>(
                                    currDSet.size() - testSize);
                            testSet = currDSet.cloneDefinition();
                            testSet.data = new ArrayList<>(testSize);
                            boolean[] testSelected =
                                    new boolean[currDSet.size()];
                            trainingLabels =
                                    new int[currDSet.size() - testSize];
                            testLabels = new int[testSize];
                            for (int i = 0; i < testIndexes.length; i++) {
                                testSet.addDataInstance(
                                        currDSet.getInstance(testIndexes[i]));
                                testLabels[i] = originalLabels[testIndexes[i]];
                                testSelected[testIndexes[i]] = true;
                            }
                            for (int i = 0; i < testSelected.length; i++) {
                                if (!testSelected[i]) {
                                    trainingLabels[trainingSet.data.size()] =
                                            originalLabels[i];
                                    trainingIndexes[
                                            trainingSet.data.size()] =i;
                                    trainingSet.addDataInstance(
                                            currDSet.getInstance(i));
                                }
                            }
                        } while (trainingSet.countCategories() !=
                                currDSet.countCategories() ||
                                testSet.countCategories() !=
                                currDSet.countCategories());
                        if (distUserPresentNonDisc || distUserPresentDisc
                                || nsfUserPresent) {
                            trainingDist = new float[trainingSet.size()][];
                            for (int i = 0; i < trainingDist.length; i++) {
                                trainingDist[i] =
                                        new float[trainingDist.length - i - 1];
                                for (int j = i + 1; j < trainingIndexes.length;
                                        j++) {
                                    trainingDist[i][j - i - 1] =
                                            distMat[Math.min(
                                            trainingIndexes[i],
                                            trainingIndexes[j])][
                                            Math.max(trainingIndexes[i],
                                            trainingIndexes[j]) - Math.min(
                                            trainingIndexes[i],
                                            trainingIndexes[j]) - 1];
                                }
                            }
                            testDist = new float[testSet.size()][];
                            for (int i = 0; i < testDist.length; i++) {
                                testDist[i] = new float[testDist.length - i
                                        - 1];
                                for (int j = i + 1; j < testIndexes.length;
                                        j++) {
                                    testDist[i][j - i - 1] = distMat[
                                            Math.min(testIndexes[i],
                                            testIndexes[j])][Math.max(
                                            testIndexes[i],
                                            testIndexes[j]) - Math.min(
                                            testIndexes[i], testIndexes[j])
                                            - 1];
                                }
                            }
                            if (secondaryDistanceType
                                    != SecondaryDistance.NONE) {
                                NeighborSetFinder secondaryNSF;
                                if (!approximateNeighbors
                                        || approximateNeighborsAlpha == 1) {
                                    secondaryNSF = new NeighborSetFinder(
                                            trainingSet, trainingDist,
                                            currCmet);
                                    secondaryNSF.calculateNeighborSetsMultiThr(
                                            Math.max(secondaryK, kGenMin),
                                            numCommonThreads);
                                } else {
                                    AppKNNGraphLanczosBisection appNSF =
                                            new AppKNNGraphLanczosBisection(
                                            trainingSet, trainingDist,
                                            Math.max(secondaryK, kGenMin),
                                            approximateNeighborsAlpha);
                                    appNSF.calculateApproximateNeighborSets();
                                    secondaryNSF =
                                            NeighborSetFinder.
                                            constructFromAppFinder(appNSF,
                                            false);
                                }
                                NeighborSetFinder secondaryNSFTest;
                                if (!approximateNeighbors
                                        || approximateNeighborsAlpha == 1) {
                                    secondaryNSFTest =
                                            new NeighborSetFinder(testSet,
                                            testDist, currCmet);
                                    secondaryNSFTest.
                                            calculateNeighborSetsMultiThr(
                                            Math.max(secondaryK, kGenMin),
                                            numCommonThreads);
                                } else {
                                    AppKNNGraphLanczosBisection appNSFTest =
                                            new AppKNNGraphLanczosBisection(
                                            testSet, testDist,
                                            Math.max(secondaryK, kGenMin),
                                            approximateNeighborsAlpha);
                                    appNSFTest.
                                            calculateApproximateNeighborSets();
                                    secondaryNSFTest = NeighborSetFinder.
                                            constructFromAppFinder(
                                            appNSFTest, false);
                                }
                                if (secondaryDistanceType
                                        == SecondaryDistance.SIMCOS) {
                                    SharedNeighborFinder snf =
                                            new SharedNeighborFinder(
                                            secondaryNSF,
                                            Math.min(10, secondaryK));
                                    snf.setNumClasses(numCategories);
                                    snf.countSharedNeighborsMultiThread(
                                            numCommonThreads);
                                    SharedNeighborCalculator calc =
                                            new SharedNeighborCalculator(
                                            snf, SharedNeighborCalculator.
                                            WeightingType.NONE);
                                    // First fetch the similarities.
                                    float[][] secondaryDistMatrix =
                                            snf.getSharedNeighborCounts();
                                    // Then transform them into distances.
                                    for (int indexFirst = 0; indexFirst
                                            < secondaryDistMatrix.length;
                                            indexFirst++) {
                                        for (int indexSecond = 0; indexSecond
                                                < secondaryDistMatrix[
                                                indexFirst].length;
                                                indexSecond++) {
                                            secondaryDistMatrix[indexFirst][
                                                    indexSecond] = secondaryK
                                                    - secondaryDistMatrix[
                                                    indexFirst][indexSecond];
                                        }
                                    }
                                    // Now for the test distance matrix.
                                    float[][] secondaryTestMatrix =
                                            new float[testSet.size()][];
                                    for (int i = 0; i < testDist.length;
                                            i++) {
                                        secondaryTestMatrix[i] = new float[
                                                secondaryTestMatrix.length
                                                - i - 1];
                                        for (int j = i + 1;
                                                j < testIndexes.length; j++) {
                                            DataInstance firstInstance =
                                                    currDSet.getInstance(
                                                    testIndexes[i]);
                                            DataInstance secondInstance =
                                                    currDSet.getInstance(
                                                    testIndexes[j]);
                                            float[] distsFirst = new float[
                                                    trainingIndexes.length];
                                            float[] distsSecond = new float[
                                                    trainingIndexes.length];
                                            int minIndex, maxIndex;
                                            for (int tI = 0; tI
                                                    < trainingIndexes.length;
                                                    tI++) {
                                                minIndex = Math.min(
                                                        testIndexes[i],
                                                        trainingIndexes[tI]);
                                                maxIndex = Math.max(
                                                        testIndexes[i],
                                                        trainingIndexes[tI]);
                                                distsFirst[tI] = distMat[
                                                            minIndex][maxIndex
                                                        - minIndex - 1];
                                                minIndex = Math.min(
                                                        testIndexes[j],
                                                        trainingIndexes[tI]);
                                                maxIndex = Math.max(
                                                        testIndexes[j],
                                                        trainingIndexes[tI]);
                                                distsSecond[tI] = distMat[
                                                            minIndex][maxIndex
                                                        - minIndex - 1];
                                            }
                                            secondaryTestMatrix[i][j - i
                                                    - 1] = calc.dist(
                                                    firstInstance,
                                                    secondInstance,
                                                    distsFirst,
                                                    distsSecond);
                                        }
                                    }
                                    testDist = secondaryTestMatrix;
                                    currCmet = calc;
                                    trainingDist = secondaryDistMatrix;
                                } else if (secondaryDistanceType
                                        == SecondaryDistance.SIMHUB) {
                                    SharedNeighborFinder snf =
                                            new SharedNeighborFinder(
                                            secondaryNSF,
                                            Math.min(10, secondaryK));
                                    snf.setNumClasses(numCategories);
                                    snf.obtainWeightsFromHubnessInformation(0);
                                    snf.countSharedNeighborsMultiThread(
                                            numCommonThreads);
                                    SharedNeighborCalculator calc =
                                            new SharedNeighborCalculator(
                                            snf, SharedNeighborCalculator.
                                            WeightingType.HUBNESS_INFORMATION);
                                    // First fetch the similarities.
                                    float[][] secondaryDistMatrix =
                                            snf.getSharedNeighborCounts();
                                    // Then transform them into distances.
                                    for (int indexFirst = 0; indexFirst
                                            < secondaryDistMatrix.length;
                                            indexFirst++) {
                                        for (int indexSecond = 0; indexSecond
                                                < secondaryDistMatrix[
                                                indexFirst].length;
                                                indexSecond++) {
                                            secondaryDistMatrix[indexFirst][
                                                    indexSecond] = secondaryK
                                                    - secondaryDistMatrix[
                                                    indexFirst][indexSecond];
                                        }
                                    }
                                    // Now for the test distance matrix.
                                    float[][] secondaryTestMatrix =
                                            new float[testSet.size()][];
                                    for (int i = 0; i < testDist.length;
                                            i++) {
                                        secondaryTestMatrix[i] = new float[
                                                secondaryTestMatrix.length
                                                - i - 1];
                                        for (int j = i + 1; j
                                                < testIndexes.length; j++) {
                                            DataInstance firstInstance =
                                                    currDSet.getInstance(
                                                    testIndexes[i]);
                                            DataInstance secondInstance =
                                                    currDSet.getInstance(
                                                    testIndexes[j]);
                                            float[] distsFirst = new float[
                                                    trainingIndexes.length];
                                            float[] distsSecond = new float[
                                                    trainingIndexes.length];
                                            int minIndex, maxIndex;
                                            for (int tI = 0; tI
                                                    < trainingIndexes.length;
                                                    tI++) {
                                                minIndex = Math.min(
                                                        testIndexes[i],
                                                        trainingIndexes[tI]);
                                                maxIndex = Math.max(
                                                        testIndexes[i],
                                                        trainingIndexes[tI]);
                                                distsFirst[tI] = distMat[
                                                            minIndex][maxIndex
                                                        - minIndex - 1];
                                                minIndex = Math.min(
                                                        testIndexes[j],
                                                        trainingIndexes[tI]);
                                                maxIndex = Math.max(
                                                        testIndexes[j],
                                                        trainingIndexes[tI]);
                                                distsSecond[tI] = distMat[
                                                            minIndex][maxIndex
                                                        - minIndex - 1];
                                            }
                                            secondaryTestMatrix[i][j - i
                                                    - 1] = calc.dist(
                                                    firstInstance,
                                                    secondInstance,
                                                    distsFirst,
                                                    distsSecond);
                                        }
                                    }
                                    testDist = secondaryTestMatrix;
                                    currCmet = calc;
                                    trainingDist = secondaryDistMatrix;
                                } else if (secondaryDistanceType
                                        == SecondaryDistance.MP) {
                                    MutualProximityCalculator calc =
                                            new MutualProximityCalculator(
                                            secondaryNSF.getDistances(),
                                            secondaryNSF.getDataSet(),
                                            secondaryNSF.getCombinedMetric());
                                    float[][] secondaryDistMatrix =
                                            calc.calculateSecondaryDistMatrixMultThr(
                                            secondaryNSF, 8);
                                    // Now for the test distance matrix.
                                    float[][] secondaryTestMatrix =
                                            new float[testSet.size()][];
                                    for (int i = 0; i < testDist.length;
                                            i++) {
                                        secondaryTestMatrix[i] = new float[
                                                secondaryTestMatrix.length
                                                - i - 1];
                                        for (int j = i + 1;
                                                j < testIndexes.length; j++) {
                                            DataInstance firstInstance =
                                                    currDSet.getInstance(
                                                    testIndexes[i]);
                                            DataInstance secondInstance =
                                                    currDSet.getInstance(
                                                    testIndexes[j]);
                                            float[] kDistsFirst =
                                                    secondaryNSFTest.
                                                    getKDistances()[i];
                                            float[] kDistsSecond =
                                                    secondaryNSFTest.
                                                    getKDistances()[j];
                                            secondaryTestMatrix[i][j - i
                                                    - 1] = calc.dist(
                                                    firstInstance,
                                                    secondInstance,
                                                    kDistsFirst,
                                                    kDistsSecond);
                                        }
                                    }
                                    testDist = secondaryTestMatrix;
                                    currCmet = calc;
                                    trainingDist = secondaryDistMatrix;
                                } else if (secondaryDistanceType
                                        == SecondaryDistance.LS) {
                                    LocalScalingCalculator calc =
                                            new LocalScalingCalculator(
                                            secondaryNSF);
                                    float[][] secondaryDistMatrix =
                                            calc.getTransformedDMatFromNSFPrimaryDMat();
                                    // Now for the test distance matrix.
                                    float[][] secondaryTestMatrix =
                                            new float[testSet.size()][];
                                    for (int i = 0; i < testDist.length;
                                            i++) {
                                        secondaryTestMatrix[i] = new float[
                                                secondaryTestMatrix.length
                                                - i - 1];
                                        for (int j = i + 1;
                                                j < testIndexes.length; j++) {
                                            DataInstance firstInstance =
                                                    currDSet.getInstance(
                                                    testIndexes[i]);
                                            DataInstance secondInstance =
                                                    currDSet.getInstance(
                                                    testIndexes[j]);
                                            float[] kDistsFirst =
                                                    secondaryNSFTest.
                                                    getKDistances()[i];
                                            float[] kDistsSecond =
                                                    secondaryNSFTest.
                                                    getKDistances()[j];
                                            secondaryTestMatrix[i][j - i
                                                    - 1] = calc.distFromKDists(
                                                    firstInstance,
                                                    secondInstance,
                                                    kDistsFirst,
                                                    kDistsSecond);
                                        }
                                    }
                                    testDist = secondaryTestMatrix;
                                    currCmet = calc;
                                    trainingDist = secondaryDistMatrix;
                                } else if (secondaryDistanceType
                                        == SecondaryDistance.NICDM) {
                                    NICDMCalculator calc =
                                            new NICDMCalculator(secondaryNSF);
                                    float[][] secondaryDistMatrix =
                                            calc.
                                            getTransformedDMatFromNSFPrimaryDMat();
                                    // Now for the test distance matrix.
                                    float[][] secondaryTestMatrix =
                                            new float[testSet.size()][];
                                    for (int i = 0; i < testDist.length;
                                            i++) {
                                        secondaryTestMatrix[i] = new float[
                                                secondaryTestMatrix.length
                                                - i - 1];
                                        for (int j = i + 1; j
                                                < testIndexes.length; j++) {
                                            DataInstance firstInstance =
                                                    currDSet.getInstance(
                                                    testIndexes[i]);
                                            DataInstance secondInstance =
                                                    currDSet.getInstance(
                                                    testIndexes[j]);
                                            float[] kDistsFirst =
                                                    secondaryNSFTest.
                                                    getKDistances()[i];
                                            float[] kDistsSecond =
                                                    secondaryNSFTest.
                                                    getKDistances()[j];
                                            secondaryTestMatrix[i][j - i
                                                    - 1] = calc.distFromKDists(
                                                    firstInstance,
                                                    secondInstance,
                                                    kDistsFirst,
                                                    kDistsSecond);
                                        }
                                    }
                                    testDist = secondaryTestMatrix;
                                    currCmet = calc;
                                    trainingDist = secondaryDistMatrix;
                                }
                            }
                        }
                        // Now the kernel objects.
                        if (kernelUserPresent) {
                            trainingKernelMat = new float[trainingSet.size()][];
                            testToTrainingKernel = new float[testSet.size()][
                                    trainingSet.size()];
                            selfKernelsTest = new float[testSet.size()];
                            for (int i = 0; i < trainingIndexes.length; i++) {
                                trainingKernelMat[i] =
                                        new float[trainingIndexes.length - i];
                                for (int j = i; j < trainingIndexes.length;
                                        j++) {
                                    trainingKernelMat[i][j - i] = kMat[
                                            Math.min(trainingIndexes[i],
                                            trainingIndexes[j])][Math.max(
                                            trainingIndexes[i],
                                            trainingIndexes[j]) - Math.min(
                                            trainingIndexes[i],
                                            trainingIndexes[j])];
                                }
                                for (int j = 0; j < testIndexes.length; j++) {
                                    testToTrainingKernel[j][i] = kMat[
                                            Math.min(trainingIndexes[i],
                                            testIndexes[j])][Math.max(
                                            trainingIndexes[i], testIndexes[j])
                                            - Math.min(trainingIndexes[i],
                                            testIndexes[j])];
                                }
                            }
                            for (int j = 0; j < testIndexes.length; j++) {
                                selfKernelsTest[j] = kMat[testIndexes[j]][0];
                            }
                            trainingKernelDist =
                                    new float[trainingSet.size()][];
                            testKernelDist = new float[testSet.size()][];
                            for (int i = 0; i < trainingIndexes.length; i++) {
                                trainingKernelDist[i] = new float[
                                        trainingKernelDist.length - i - 1];
                                for (int j = i + 1; j < trainingIndexes.length;
                                        j++) {
                                    trainingKernelDist[i][j - i - 1] =
                                            kernelDMat[
                                            Math.min(trainingIndexes[i],
                                            trainingIndexes[j])][Math.max(
                                            trainingIndexes[i],
                                            trainingIndexes[j]) - Math.min(
                                            trainingIndexes[i],
                                            trainingIndexes[j]) - 1];
                                }
                            }
                            for (int i = 0; i < testKernelDist.length; i++) {
                                testKernelDist[i] =
                                        new float[testKernelDist.length - i
                                        - 1];
                                for (int j = i + 1; j < testIndexes.length;
                                        j++) {
                                    testKernelDist[i][j - i - 1] = kernelDMat[
                                            Math.min(testIndexes[i],
                                            testIndexes[j])][Math.max(
                                            testIndexes[i],
                                            testIndexes[j]) - Math.min(
                                            testIndexes[i], testIndexes[j])
                                            - 1];
                                }
                            }
                        }
                        // We print out the size of the training/test splits.
                        System.out.println("Training on "
                                + trainingIndexes.length + " , testing on "
                                + testIndexes.length);
                    } else {
                        // In case no splits are required, we train and evaluate
                        // on the whole data.
                        trainingSet = currDSet;
                        trainingDist = distMat;
                        trainingKernelDist = kernelDMat;
                        trainingKernelMat = kMat;
                        selfKernelsTest = new float[trainingSet.size()];
                        if (kMat != null) {
                            for (int j = 0; j < trainingSet.size(); j++) {
                                selfKernelsTest[j] = kMat[j][0];
                            }
                        }
                        if (secondaryDistanceType != SecondaryDistance.NONE) {
                            NeighborSetFinder secondaryNSF;
                            if (!approximateNeighbors
                                    || approximateNeighborsAlpha == 1) {
                                secondaryNSF = new NeighborSetFinder(
                                        trainingSet, trainingDist,
                                        currCmet);
                                secondaryNSF.calculateNeighborSetsMultiThr(
                                        Math.max(secondaryK, kGenMin),
                                        numCommonThreads);
                            } else {
                                AppKNNGraphLanczosBisection appNSF =
                                        new AppKNNGraphLanczosBisection(
                                        trainingSet, trainingDist,
                                        Math.max(secondaryK, kGenMin),
                                        approximateNeighborsAlpha);
                                appNSF.calculateApproximateNeighborSets();
                                secondaryNSF =
                                        NeighborSetFinder.
                                        constructFromAppFinder(appNSF,
                                        false);
                            }
                            if (secondaryDistanceType
                                    == SecondaryDistance.SIMCOS) {
                                // Simcos shared-neighbor similarity.
                                SharedNeighborFinder snf =
                                        new SharedNeighborFinder(
                                        secondaryNSF, Math.min(10, secondaryK));
                                snf.setNumClasses(numCategories);
                                snf.countSharedNeighborsMultiThread(
                                        numCommonThreads);
                                SharedNeighborCalculator calc =
                                        new SharedNeighborCalculator(
                                        snf, SharedNeighborCalculator.
                                        WeightingType.NONE);
                                // First fetch the similarities.
                                float[][] secondaryDistMatrix =
                                        snf.getSharedNeighborCounts();
                                // Then transform them into distances.
                                for (int indexFirst = 0; indexFirst
                                        < secondaryDistMatrix.length;
                                        indexFirst++) {
                                    for (int indexSecond = 0; indexSecond
                                            < secondaryDistMatrix[
                                            indexFirst].length; indexSecond++) {
                                        secondaryDistMatrix[indexFirst][
                                                indexSecond] = secondaryK
                                                - secondaryDistMatrix[
                                                indexFirst][indexSecond];
                                    }
                                }
                                currCmet = calc;
                                trainingDist = secondaryDistMatrix;
                            } else if (secondaryDistanceType
                                    == SecondaryDistance.SIMHUB) {
                                // Hubness-aware simhub shared-neighbor
                                // secondary similarity.
                                SharedNeighborFinder snf =
                                        new SharedNeighborFinder(
                                        secondaryNSF, Math.min(10, secondaryK));
                                snf.setNumClasses(numCategories);
                                snf.obtainWeightsFromHubnessInformation();
                                snf.countSharedNeighborsMultiThread(
                                        numCommonThreads);
                                SharedNeighborCalculator calc =
                                        new SharedNeighborCalculator(
                                        snf, SharedNeighborCalculator.
                                        WeightingType.HUBNESS_INFORMATION);
                                // First fetch the similarities.
                                float[][] secondaryDistMatrix =
                                        snf.getSharedNeighborCounts();
                                // Then transform them into distances.
                                for (int indexFirst = 0; indexFirst
                                        < secondaryDistMatrix.length;
                                        indexFirst++) {
                                    for (int indexSecond = 0; indexSecond
                                            < secondaryDistMatrix[
                                            indexFirst].length; indexSecond++) {
                                        secondaryDistMatrix[indexFirst][
                                                indexSecond] = secondaryK
                                                - secondaryDistMatrix[
                                                indexFirst][indexSecond];
                                    }
                                }
                                currCmet = calc;
                                trainingDist = secondaryDistMatrix;
                            } else if (secondaryDistanceType
                                    == SecondaryDistance.MP) {
                                // Mutual proximity as a secondary distance
                                // measure.
                                MutualProximityCalculator calc =
                                        new MutualProximityCalculator(
                                        secondaryNSF.getDistances(),
                                        secondaryNSF.getDataSet(),
                                        secondaryNSF.getCombinedMetric());
                                float[][] secondaryDistMatrix =
                                        calc.calculateSecondaryDistMatrixMultThr(
                                        secondaryNSF, 8);
                                currCmet = calc;
                                trainingDist = secondaryDistMatrix;
                            } else if (secondaryDistanceType
                                    == SecondaryDistance.LS) {
                                // Local scaling as a secondary distance
                                // measure.
                                LocalScalingCalculator calc =
                                        new LocalScalingCalculator(
                                        secondaryNSF);
                                float[][] secondaryDistMatrix =
                                        calc.getTransformedDMatFromNSFPrimaryDMat();
                                currCmet = calc;
                                trainingDist = secondaryDistMatrix;
                            } else if (secondaryDistanceType
                                    == SecondaryDistance.NICDM) {
                                // The NICDM secondary distance measure.
                                NICDMCalculator calc =
                                        new NICDMCalculator(secondaryNSF);
                                float[][] secondaryDistMatrix =
                                        calc.getTransformedDMatFromNSFPrimaryDMat();
                                currCmet = calc;
                                trainingDist = secondaryDistMatrix;
                            }
                        }
                    }
                    // In case neighbor sets need to be calculated.
                    if (nsfUserPresent) {
                        System.out.println("Calculating kNN sets...");
                        // No metric learning in form of shared neighbor
                        // distances.
                        if (!approximateNeighbors
                                || approximateNeighborsAlpha == 1) {
                            bigNSF = new NeighborSetFinder(trainingSet,
                                    trainingDist, currCmet);
                            bigNSF.calculateNeighborSetsMultiThr(
                                    Math.max(2 * kMax, kGenMin),
                                    numCommonThreads);
                        } else {
                            // Approximate kNN sets.
                            AppKNNGraphLanczosBisection appNSF =
                                    new AppKNNGraphLanczosBisection(
                                    trainingSet, trainingDist,
                                    Math.max(2 * kMax, kGenMin),
                                    approximateNeighborsAlpha);
                            appNSF.calculateApproximateNeighborSets();
                            bigNSF =
                                    NeighborSetFinder.constructFromAppFinder(
                                    appNSF, false);
                        }
                        // If there is a training and a test split, we need to
                        // generate test neighbor sets as well.
                        if (splitTesting) {
                            if (!approximateNeighbors
                                    || approximateNeighborsAlpha == 1) {
                                bigNSFTest = new NeighborSetFinder(
                                        testSet, testDist, currCmet);
                                bigNSFTest.calculateNeighborSetsMultiThr(
                                        Math.max(2 * kMax, kGenMin),
                                        numCommonThreads);
                            } else {
                                AppKNNGraphLanczosBisection appNSFTest =
                                        new AppKNNGraphLanczosBisection(
                                        testSet, testDist,
                                        Math.max(2 * kMax, kGenMin),
                                        approximateNeighborsAlpha);
                                appNSFTest.
                                        calculateApproximateNeighborSets();
                                bigNSFTest =
                                        NeighborSetFinder.
                                        constructFromAppFinder(
                                        appNSFTest, false);
                            }
                        }
                        System.out.println("kNN sets calculated.");
                    }
                    // Now for neighbor sets in the kernel space.
                    if (kernelNSFUserPresent) {
                        System.out.println("calculating kernel kNN sets");
                        if (!approximateNeighbors
                                || approximateNeighborsAlpha == 1) {
                            bigKernelNSF = new NeighborSetFinder(
                                    trainingSet, trainingKernelDist, currCmet);
                            bigKernelNSF.calculateNeighborSetsMultiThr(
                                    Math.max(2 * kMax, kGenMin),
                                    numCommonThreads);
                        } else {
                            AppKNNGraphLanczosBisection appKernelNSF =
                                    new AppKNNGraphLanczosBisection(
                                    trainingSet, trainingKernelDist,
                                    Math.max(2 * kMax, kGenMin),
                                    approximateNeighborsAlpha);
                            appKernelNSF.calculateApproximateNeighborSets();
                            bigKernelNSF =
                                    NeighborSetFinder.constructFromAppFinder(
                                    appKernelNSF, false);
                        }
                        if (splitTesting) {
                            if (!approximateNeighbors
                                    || approximateNeighborsAlpha == 1) {
                                bigKernelNSFTest = new NeighborSetFinder(
                                        testSet, testKernelDist, currCmet);
                                bigKernelNSFTest.calculateNeighborSetsMultiThr(
                                        Math.max(2 * kMax, kGenMin),
                                        numCommonThreads);
                            } else {
                                AppKNNGraphLanczosBisection appKernelNSFTest =
                                        new AppKNNGraphLanczosBisection(testSet,
                                        testKernelDist,
                                        Math.max(2 * kMax, kGenMin),
                                        approximateNeighborsAlpha);
                                appKernelNSFTest.
                                        calculateApproximateNeighborSets();
                                bigKernelNSFTest =
                                        NeighborSetFinder.
                                        constructFromAppFinder(
                                        appKernelNSFTest, false);
                            }
                        }
                        System.out.println("kernel kNN sets calculated");
                    }
                    // Now we iterated over the specified range of neighborhood
                    // sizes for those algorithms that depend onthem.
                    for (int k = kMin; k <= kMax; k += kStep) {
                        System.out.println("Using k: " + k);
                        NeighborSetFinder nsf = null;
                        NeighborSetFinder nsfTest = null;
                        NeighborSetFinder nsfKernel = null;
                        NeighborSetFinder nsfKernelTest = null;
                        if (nsfUserPresent && bigNSF != null) {
                            // Create a NeighborSetFinder object for the current
                            // neighborhood size.
                            nsf = bigNSF.getSubNSF(Math.max(k, kGenMin));
                            if (splitTesting && bigNSFTest != null) {
                                nsfTest = bigNSFTest.getSubNSF(Math.max(k,
                                        kGenMin));
                            }
                        }
                        if (kernelNSFUserPresent && bigKernelNSF != null) {
                            nsfKernel = bigKernelNSF.getSubNSF(Math.max(k,
                                    kGenMin));
                            if (splitTesting && bigKernelNSFTest != null) {
                                nsfKernelTest = bigKernelNSFTest.getSubNSF(
                                        Math.max(k, kGenMin));
                            }
                        }
                        NeighborSetFinder nsfSmaller = null;
                        NeighborSetFinder nsfTestSmaller = null;
                        NeighborSetFinder nsfKernelSmaller = null;
                        NeighborSetFinder nsfKernelTestSmaller = null;
                        if (k < kGenMin) {
                            if (nsfUserPresent) {
                                if (nsf != null) {
                                    nsfSmaller = nsf.getSubNSF(k);
                                }
                                if (nsfTest != null) {
                                    nsfTestSmaller = nsfTest.getSubNSF(k);
                                }
                            }
                            if (kernelNSFUserPresent) {
                                if (nsfKernel != null) {
                                    nsfKernelSmaller = nsfKernel.getSubNSF(k);
                                }
                                if (nsfKernelTest != null) {
                                    nsfKernelTestSmaller =
                                            nsfKernelTest.getSubNSF(k);
                                }
                            }
                        }
                        for (int numClusters = cluNumMin;
                                numClusters <= cluNumMax;
                                numClusters += cluNumStep) {
                            System.out.println("Clustering for " + numClusters
                                    + " clusters...");
                            Thread[] algThreads =
                                    new Thread[clustererNames.size()];
                            for (int i = 0; i < clustererNames.size(); i++) {
                                String cName = clustererNames.get(i);
                                ClusteringAlg cTemp = getClustererForName(
                                        cName, trainingSet, i, k, trainingDist,
                                        nsf, trainingKernelMat, nsfKernel,
                                        numCategories);
                                globalIterationCounter = 0;
                                System.out.println();
                                if ((cTemp instanceof NSFUserInterface)
                                        || (k == kMin)) {
                                    if (!cName.equals("dbscan")
                                            && k < kGenMin) {
                                        algThreads[i] = new Thread(
                                                new AlgorithmTesterThread(
                                                dsFile, ml, noise, cName, i,
                                                trainingSet, k, trainingDist,
                                                testDist, nsfSmaller,
                                                nsfTestSmaller, testSet,
                                                currCmet, numClusters,
                                                trainingKernelMat,
                                                testToTrainingKernel,
                                                selfKernelsTest,
                                                nsfKernelSmaller));
                                    } else {
                                        algThreads[i] = new Thread(
                                                new AlgorithmTesterThread(
                                                dsFile, ml, noise, cName, i,
                                                trainingSet, k, trainingDist,
                                                testDist, nsf, nsfTest, testSet,
                                                currCmet, numClusters,
                                                trainingKernelMat,
                                                testToTrainingKernel,
                                                selfKernelsTest, nsfKernel));
                                    }
                                    algThreads[i].start();
                                } else {
                                    // If the algorithm does not depend on k and
                                    // we are iterating over k, then it doesn't
                                    // make sense to re-run it every time, so we
                                    // just copy the results from the previous
                                    // run, when available.
                                    if (!(cTemp instanceof NSFUserInterface)
                                            && k > kMin) {
                                        File prevOutDSDir = new File(
                                                outDir, dsFile.getName().
                                                substring(0, dsFile.getName().
                                                lastIndexOf("."))
                                                + File.separator + "clust"
                                                + numClusters + File.separator
                                                + "k" + kMin + File.separator
                                                + "ml" + ml + File.separator
                                                + "noise" + noise);
                                        File prevOutFile = new File(
                                                prevOutDSDir, cName +
                                                "Report.csv");
                                        File targetOutDSDir = new File(outDir,
                                                dsFile.getName().substring(0,
                                                dsFile.getName().
                                                lastIndexOf("."))
                                                + File.separator + "clust"
                                                + numClusters + File.separator
                                                + "k" + k + File.separator
                                                + "ml" + ml + File.separator
                                                + "noise" + noise);
                                        File targetOutFile = new File(
                                                targetOutDSDir, cName
                                                + "Report.csv");
                                        FileUtil.copyFile(
                                                prevOutFile, targetOutFile);
                                    }
                                }
                            }
                            for (int algIndex = 0;
                                    algIndex < clustererNames.size();
                                    algIndex++) {
                                if (algThreads[algIndex] != null) {
                                    try {
                                        algThreads[algIndex].join();
                                    } catch (Throwable t) {
                                    }
                                }
                            }
                            System.gc();
                        }
                    }
                }
                distMat = null;
                kMat = null;
            }
            dsCounter++;
            System.gc();
        }
    }

    /**
     * @param algName String that is the algorithm name.
     * @return True if the algorithm requires discretized data, otherwise false.
     */
    public boolean isDiscrete(String algName) {
        // Currently there is no implementation on discretized data.
        return false;
    }

    /**
     * A kind of a hack for now - signals if a certain clusterer requires
     * k-nearest neighbor sets based on its name. This check is done prior to
     * generating the clusterer instances. TODO: Change this mechanism.
     *
     * @param algName String that is the algorithm name.
     * @return True if the algorithm requires kNN sets, false otherwise.
     */
    public boolean requiresNSF(String algName) {
        switch (algName.toLowerCase()) {
            case "ghpc":
                return true;
            case "ghpkm":
                return true;
            case "gkh":
                return true;
            case "dbscan":
                return true;
            default:
                return false;
        }
    }

    /**
     * Signals if an algorithm requires kernel k-nearest neighbor sets.
     *
     * @param algName String that is the algorithm name.
     * @return True if the algorithm needs kernel kNN sets, false otherwise.
     */
    public boolean requiresKernelNSF(String algName) {
        if (algName.toLowerCase().equals("kernel-ghpkm")) {
            return true;
        } else {
            return false;
        }
    }
    
    /**
     * This is how initial clusterer instances are currently generated, as they
     * require initial parametrizations and doing all of that via reflection in
     * every single case might be somewhat non-trivial in the general case -
     * though it is certainly preferable to having the algorithm names hardcoded
     * inside. TODO: change this clusterer initialization mechanism.
     *
     * @param cName Clusterer name.
     * @param dset DataSet object.
     * @param index Index of the clusterer within the algorithm array.
     * @param k Neighborhood size.
     * @param distMat Distance matrix.
     * @param nsf NeighborSetFinder object for kNN sets.
     * @param trainingKernelMat Kernel matrix.
     * @param nsfKernel Kernel kNN object.
     * @param numClusters Number of clusters.
     * @return
     */
    public ClusteringAlg getClustererForName(
            String cName,
            DataSet dset,
            int index,
            int k,
            float[][] distMat,
            NeighborSetFinder nsf,
            float[][] trainingKernelMat,
            NeighborSetFinder nsfKernel,
            int nClust) {
        discrete[index] = false;
        return ClustererFactory.getClustererForName(cName, dset, k, distMat,
                nsf, trainingKernelMat, nsfKernel, nClust, currCmet, ker);
    }
    
    /**
     * This method loads the configuration from the configuration object.
     * 
     * @param conf BatchClusteringConfig that is the configuration object.
     */
    public void loadFromConfigurationObject(BatchClusteringConfig conf) {
        calculatePerplexity = conf.calculatePerplexity;
        splitTesting = conf.splitTesting;
        splitPerc = conf.splitPerc;
        nClustSpecified = conf.nClustSpecified;
        normType = conf.normType;
        clustersAutoSet = conf.clustersAutoSet;
        timesOnDataSet = conf.timesOnDataSet;
        minIter = conf.minIter;
        clustererNames = conf.clustererNames;
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
        distancesDir = conf.distancesDir;
        mlWeightsDir = conf.mlWeightsDir;
        dsPaths = conf.dsPaths;
        dsMetric = conf.dsMetric;
        ker = conf.ker;
        secondaryDistanceType = conf.secondaryDistanceType;
        secondaryK = conf.secondaryK;
        cluNumMin = conf.cluNumMin;
        cluNumMax = conf.cluNumMax;
        cluNumStep = conf.cluNumStep;
        approximateNeighborsAlpha = conf.approximateNeighborsAlpha;
        approximateNeighbors = conf.approximateNeighbors;
        numCommonThreads = conf.numCommonThreads;
    }

    /**
     * This method loads all the parameters from the provided clustering
     * configuration file.
     *
     * @throws Exception
     */
    public void loadParameters() throws Exception {
        BatchClusteringConfig config = new BatchClusteringConfig();
        config.loadParameters(inConfigFile);
        loadFromConfigurationObject(config);
    }

    /**
     * A class responsible for running clustering algorithm testing.
     */
    class AlgorithmTesterThread implements Runnable {

        float[][] trainingDist = null;
        float[][] testDist = null;
        NeighborSetFinder nsf;
        NeighborSetFinder nsfKernel;
        NeighborSetFinder nsfTest;
        int k;
        DataSet dsetTraining = null;
        DataSet dsetTest = null;
        String cName;
        CombinedMetric cmet = null;
        int numClusters;
        File dsFile;
        float mlRate;
        float noiseRate;
        int clustererIndex;
        float[][] trainingKernelMat;
        float[][] testToTrainingKernel;
        float[] selfKernelsTest;

        /**
         * Set up all the testing parameters.
         *
         * @param dsFile DataSet file.
         * @param ml Mislabeling rate.
         * @param noise Noise rate.
         * @param cName Clusterer name.
         * @param clustererIndex Index within the algorithm array.
         * @param dsetTraining DataSet object for the training data.
         * @param k Neighborhood size.
         * @param trainingDist Training distances.
         * @param testDist Test distances.
         * @param nsf NeighborSetFinder object for the kNN sets on training data
         * @param nsfTest NeighborSetFinder object for the test kNN sets.
         * @param dsetTest DataSet object for the test data.
         * @param cmet CombinedMetric object for the distances.
         * @param numClust Number of clusters.
         * @param trainingKernelMat Kernel matrix on the training data.
         * @param testToTrainingKernelMat Kernel matrix between training and
         * test data.
         * @param selfKernelsTest Diagonal kernel elements from the test kernel
         * matrix.
         * @param nsfKernel NeighborSetFinder object for the kNN sets in the
         * kernel space.
         */
        public AlgorithmTesterThread(
                File dsFile,
                float ml,
                float noise,
                String cName,
                int clustererIndex,
                DataSet dsetTraining,
                int k,
                float[][] trainingDist,
                float[][] testDist,
                NeighborSetFinder nsf,
                NeighborSetFinder nsfTest,
                DataSet dsetTest,
                CombinedMetric cmet,
                int numClust,
                float[][] trainingKernelMat,
                float[][] testToTrainingKernelMat,
                float[] selfKernelsTest,
                NeighborSetFinder nsfKernel) {
            this.trainingDist = trainingDist;
            this.cName = cName;
            this.k = k;
            this.nsf = nsf;
            if (dsetTraining != null) {
                try {
                    this.dsetTraining = dsetTraining.copy();
                } catch (Exception e) {
                }
            }
            if (dsetTest != null) {
                try {
                    this.dsetTest = dsetTest.copy();
                } catch (Exception e) {
                }
            }
            this.cmet = cmet;
            this.numClusters = numClust;
            this.dsFile = dsFile;
            this.mlRate = ml;
            this.noiseRate = noise;
            this.clustererIndex = clustererIndex;
            this.testDist = testDist;
            this.nsfTest = nsfTest;
            this.trainingKernelMat = trainingKernelMat;
            this.testToTrainingKernel = testToTrainingKernelMat;
            this.selfKernelsTest = selfKernelsTest;
            this.nsfKernel = nsfKernel;
        }

        @Override
        public void run() {
            try {
                long startTime = System.nanoTime();

                // Silhouette index scores, general.
                float[] silScores;
                // Silhouette index scores, decomposed per point types and
                // averaged.
                //-------------------------------------------------------------
                float[] HASCORES;
                float avgHA;
                float[] HBSCORES;
                float avgHB;
                float[] AHASCORES;
                float avgAHA;
                float[] AHBSCORES;
                float avgAHB;
                float[] REGASCORES;
                float avgREGA;
                float[] REGBSCORES;
                float avgREGB;
                //-------------------------------------------------------------
                // Squared error.
                float[] avgError;
                // Cluster non-homogeneity measured by entropy.
                float[] avgClusterEntropy;
                // Elapsed clustering times.
                double[] elapsedTimes;
                // Average values for the above quantities.
                float avgSil;
                float avgErr;
                float avgTime = 0;
                float avgEntropy;

                // Same as above, only now on the test set.
                float[] silScoresTest;
                //-------------------------------------------------------------
                float[] HASCORESTest;
                float avgHATest;
                float[] HBSCORESTest;
                float avgHBTest;
                float[] AHASCORESTest;
                float avgAHATest;
                float[] AHBSCORESTest;
                float avgAHBTest;
                float[] REGASCORESTest;
                float avgREGATest;
                float[] REGBSCORESTest;
                float avgREGBTest;
                //-------------------------------------------------------------
                float[] avgErrorTest;
                float[] avgClusterEntropyTest;
                float avgSilTest;
                float avgErrTest;
                float avgEntropyTest;

                // Dunn clustering quality index values.
                float[] dunnValues;
                float[] dunnValuesTest;

                // First cluster associations for stability tests.
                int[] firstAsoc = null;
                int[] firstAsocTest = null;

                File currOutDSDir = new File(
                        outDir, dsFile.getName().substring(0,
                        dsFile.getName().lastIndexOf(".")) + File.separator
                        + "clust" + numClusters + File.separator + "k" + k
                        + File.separator + "ml" + mlRate + File.separator
                        + "noise" + noiseRate);
                FileUtil.createDirectory(currOutDSDir);
                // Initialize all the measures defined above.
                avgHA = 0;
                avgHB = 0;
                avgAHA = 0;
                avgAHB = 0;
                avgREGA = 0;
                avgREGB = 0;
                ClusteringAlg clusterer;
                // Hub points.
                HASCORES = new float[timesOnDataSet];
                HBSCORES = new float[timesOnDataSet];
                // Anti-hub points.
                AHASCORES = new float[timesOnDataSet];
                AHBSCORES = new float[timesOnDataSet];
                // Regular points.
                REGASCORES = new float[timesOnDataSet];
                REGBSCORES = new float[timesOnDataSet];
                silScores = new float[timesOnDataSet];
                avgError = new float[timesOnDataSet];
                elapsedTimes = new double[timesOnDataSet];
                avgClusterEntropy = new float[timesOnDataSet];
                avgSil = 0;
                avgErr = 0;
                avgEntropy = 0;
                avgTime = 0;

                dunnValues = new float[timesOnDataSet];
                dunnValuesTest = new float[timesOnDataSet];

                double avgDunn = 0;
                double avgDunnTest = 0;

                // Rand clustering quality index.
                float[] randValues = new float[timesOnDataSet];
                float[] randValuesTest = new float[timesOnDataSet];

                double avgRand = 0;
                double avgRandTest = 0;

                // Rand stability index.
                float[] randStabilityTest = new float[timesOnDataSet];
                double avgRandStabilityTest = 0;

                float[] randStability = new float[timesOnDataSet];
                double avgRandStability = 0;

                // Isolation index values that measures the percentage of
                // neighbor points in kNN sets that belong to the same cluster
                // as their reverse neighbors.

                float[] isolationValues = new float[timesOnDataSet];
                float[] isolationValuesTest = new float[timesOnDataSet];

                double avgIsolation = 0;
                double avgIsolationTest = 0;

                avgHATest = 0;
                avgHBTest = 0;
                avgAHATest = 0;
                avgAHBTest = 0;
                avgREGATest = 0;
                avgREGBTest = 0;
                // Hub points.
                HASCORESTest = new float[timesOnDataSet];
                HBSCORESTest = new float[timesOnDataSet];
                // Anti-hub points.
                AHASCORESTest = new float[timesOnDataSet];
                AHBSCORESTest = new float[timesOnDataSet];
                // Regular points.
                REGASCORESTest = new float[timesOnDataSet];
                REGBSCORESTest = new float[timesOnDataSet];
                silScoresTest = new float[timesOnDataSet];
                avgErrorTest = new float[timesOnDataSet];
                avgClusterEntropyTest = new float[timesOnDataSet];
                avgSilTest = 0;
                avgErrTest = 0;
                avgEntropyTest = 0;

                // Perplexity - a time consuming quality index that is not
                // calculated by default in this implementation.
                double[] perp = new double[timesOnDataSet];
                double perpAvg = 0;

                // Error tracking.
                int numErrors;
                int numOffSil = 0;
                int numOffAssigns = 0;

                int numNonOnePerplexity = 0;

                int t = -1;
                while (++t < timesOnDataSet) {
                    numErrors = 0;
                    boolean doneCorrectly = false;
                    // Sometimes some exceptions might be raised and this code
                    // allows the clustering to be run a couple of times before
                    // it breaks - 10 consecutive misses by default.
                    do {
                        // Initialize the clusterer object.
                        clusterer = getClustererForName(
                                cName,
                                dsetTraining,
                                clustererIndex,
                                k,
                                trainingDist,
                                nsf,
                                trainingKernelMat,
                                nsfKernel,
                                numClusters);
                        clusterer.setMinIterations(minIter);
                        long allTestStartTime = System.nanoTime();
                        try {
                            allTestStartTime = System.nanoTime();
                            clusterer.cluster();
                            doneCorrectly = true;
                        } catch (Exception e) {
                            if (!(e instanceof EmptyClusterException)) {
                                System.err.println(cName + " error: "
                                        + e.getMessage());
                            }
                            numErrors++;
                            if (numErrors > 10) {
                                doneCorrectly = true;
                            }
                        } finally {
                            execTimeAllOneRun = (System.nanoTime()
                                    - allTestStartTime) / 1000;
                        }
                    } while (!doneCorrectly);
                    avgTime += execTimeAllOneRun / 1000; // In seconds.
                    elapsedTimes[t] = execTimeAllOneRun / 1000;
                    Cluster[] testConfig = clusterer.getMinimizingClusters();
                    if (t == 0) {
                        firstAsoc = clusterer.getClusterAssociations();
                    }
                    int nClustConf = testConfig.length;
                    QIndexSilhouette silIndex = new QIndexSilhouette(
                            nClustConf, clusterer.getClusterAssociations(),
                            dsetTraining);
                    silIndex.setDistanceMatrix(trainingDist);
                    if (nsf != null) {
                        silIndex.hubnessArray =
                                nsf.getNeighborOccFrequencies(k);
                    }
                    silScores[t] = silIndex.validity();
                    // Hub points.
                    HASCORES[t] = (float) silIndex.HATOTAL;
                    avgHA += HASCORES[t];
                    HBSCORES[t] = (float) silIndex.HBTOTAL;
                    avgHB += HBSCORES[t];
                    // Anti-hub points.
                    AHASCORES[t] = (float) silIndex.AHATOTAL;
                    avgAHA += AHASCORES[t];
                    AHBSCORES[t] = (float) silIndex.AHBTOTAL;
                    avgAHB += AHBSCORES[t];
                    // Regular points.
                    REGASCORES[t] = (float) silIndex.REGATOTAL;
                    avgREGA += REGASCORES[t];
                    REGBSCORES[t] = (float) silIndex.REGBTOTAL;
                    avgREGB += REGBSCORES[t];
                    avgSil += silScores[t];
                    DataInstance[] centroids =
                            new DataInstance[testConfig.length];
                    int numNonEmpty = 0;
                    for (int cIndex = 0; cIndex < centroids.length;
                            cIndex++) {
                        if (testConfig[cIndex] != null
                                && testConfig[cIndex].size() > 0) {
                            centroids[cIndex] =
                                    testConfig[cIndex].getCentroid();
                            numNonEmpty++;
                            for (int p = 0; p < testConfig[cIndex].size();
                                    p++) {
                                avgError[t] += currCmet.dist(
                                        centroids[cIndex],
                                        testConfig[cIndex].getInstance(p));
                            }
                        }
                    }
                    avgError[t] /= dsetTraining.size();
                    avgErr += avgError[t];
                    int currIndex = -1;
                    QIndexDunn di = new QIndexDunn(testConfig, currCmet);
                    dunnValues[t] = di.validity();
                    avgDunn += dunnValues[t];

                    QIndexRand rand = new QIndexRand(
                            dsetTraining, clusterer.getClusterAssociations());
                    randValues[t] = rand.validity();
                    avgRand += randValues[t];
                    QIndexIsolation isolationIndex = new QIndexIsolation(
                            nsf, clusterer.getClusterAssociations());
                    isolationValues[t] = isolationIndex.validity();
                    avgIsolation += isolationValues[t];
                    randStability[t] = rand.compareToConfiguration(firstAsoc);
                    avgRandStability += randValues[t];

                    // Now check for the configuration entropy on original
                    // labels.
                    ArrayList<Integer>[] split = new ArrayList[numNonEmpty];
                    for (int cIndex = 0; cIndex < split.length; cIndex++) {
                        split[cIndex] = new ArrayList(1500);
                    }
                    for (int cIndex = 0; cIndex < testConfig.length;
                            cIndex++) {
                        if (testConfig[cIndex] != null
                                && testConfig[cIndex].size() > 0) {
                            ++currIndex;
                            for (int kIndex = 0; kIndex
                                    < testConfig[cIndex].indexes.size();
                                    kIndex++) {
                                if (trainingLabels[testConfig[cIndex].
                                        indexes.get(kIndex)] != -1) {
                                    split[currIndex].add(trainingLabels[
                                             testConfig[cIndex].indexes.get(
                                            kIndex)]);
                                }
                            }
                        }
                    }
                    avgClusterEntropy[t] = Info.evaluateInfoOfCategorySplit(
                            split, numCategories);
                    avgEntropy += avgClusterEntropy[t];
                    // Now for the evaluation on the test split, if needed.
                    if (splitTesting) {
                        int[] testClusterAssociations = null;
                        try {
                            if (clusterer instanceof
                                    KernelMatrixUserInterface) {
                                testClusterAssociations =
                                        clusterer.assignPointsToModelClusters(
                                        dsetTest, nsfTest, testToTrainingKernel,
                                        selfKernelsTest);
                            } else {
                                testClusterAssociations =
                                        clusterer.assignPointsToModelClusters(
                                        dsetTest, nsfTest);
                            }
                            numOffAssigns = 0;
                        } catch (Exception e) {
                            if (numOffAssigns < 8) {
                                numOffAssigns++;
                                avgEntropy -= avgClusterEntropy[t];
                                avgErr -= avgError[t];
                                avgSil -= silScores[t];
                                avgTime -= elapsedTimes[t];
                                avgDunn -= dunnValues[t];
                                avgRand -= randValues[t];
                                avgIsolation -= isolationValues[t];
                                avgRandStability -= randStability[t];
                                avgHA -= HASCORES[t];
                                avgHB -= HBSCORES[t];
                                avgAHA -= AHASCORES[t];
                                avgAHB -= AHBSCORES[t];
                                avgREGA -= REGASCORES[t];
                                avgREGB -= REGBSCORES[t];
                                t--;
                            }
                            continue;
                        } finally {
                        }
                        // Cluster configuration on the test data.
                        testConfig = Cluster.getConfigurationFromAssociations(
                                testClusterAssociations, dsetTest);
                        nClustConf = testConfig.length;

                        if (t == 0) {
                            firstAsocTest = testClusterAssociations;
                        }

                        silIndex = new QIndexSilhouette(nClustConf,
                                testClusterAssociations, dsetTest);
                        silIndex.setDistanceMatrix(testDist);
                        if (nsfTest != null) {
                            silIndex.hubnessArray =
                                    nsfTest.getNeighborOccFrequencies(k);
                        }
                        silScoresTest[t] = silIndex.validity();
                        if ((silScoresTest[t] == 1 && numOffSil < 10
                                && !(clusterer instanceof DBScan)
                                || silScores[t] == 1)) {
                            // Error, wrong number of clusters or something.
                            numOffSil++;
                            avgEntropy -= avgClusterEntropy[t];
                            avgErr -= avgError[t];
                            avgSil -= silScores[t];
                            avgTime -= elapsedTimes[t];
                            avgDunn -= dunnValues[t];
                            avgRand -= randValues[t];
                            avgIsolation -= isolationValues[t];
                            avgRandStability -= randStability[t];
                            avgHA -= HASCORES[t];
                            avgHB -= HBSCORES[t];
                            avgAHA -= AHASCORES[t];
                            avgAHB -= AHBSCORES[t];
                            avgREGA -= REGASCORES[t];
                            avgREGB -= REGBSCORES[t];
                            t--;
                            continue;
                        } else {
                            numOffSil = 0;
                        }
                        di = new QIndexDunn(testConfig, currCmet);
                        dunnValuesTest[t] = di.validity();
                        avgDunnTest += dunnValuesTest[t];
                        rand = new QIndexRand(dsetTest,
                                testClusterAssociations);
                        randValuesTest[t] = rand.validity();
                        avgRandTest += randValuesTest[t];
                        randStabilityTest[t] =
                                rand.compareToConfiguration(firstAsocTest);
                        avgRandStabilityTest += randValuesTest[t];
                        isolationIndex = new QIndexIsolation(
                                nsfTest, testClusterAssociations);
                        isolationValuesTest[t] = isolationIndex.validity();
                        avgIsolationTest += isolationValuesTest[t];

                        HASCORESTest[t] = (float) silIndex.HATOTAL;
                        avgHATest += HASCORESTest[t];
                        HBSCORESTest[t] = (float) silIndex.HBTOTAL;
                        avgHBTest += HBSCORESTest[t];
                        AHASCORESTest[t] = (float) silIndex.AHATOTAL;
                        avgAHATest += AHASCORESTest[t];
                        AHBSCORESTest[t] = (float) silIndex.AHBTOTAL;
                        avgAHBTest += AHBSCORESTest[t];
                        REGASCORESTest[t] = (float) silIndex.REGATOTAL;
                        avgREGATest += REGASCORESTest[t];
                        REGBSCORESTest[t] = (float) silIndex.REGBTOTAL;
                        avgREGBTest += REGBSCORESTest[t];
                        avgSilTest += silScoresTest[t];
                        centroids = new DataInstance[testConfig.length];
                        numNonEmpty = 0;
                        for (int cIndex = 0; cIndex < centroids.length;
                                cIndex++) {
                            if (testConfig[cIndex] != null
                                    && testConfig[cIndex].size() > 0) {
                                centroids[cIndex] =
                                        testConfig[cIndex].getCentroid();
                                numNonEmpty++;
                                for (int p = 0; p < testConfig[cIndex].size();
                                        p++) {
                                    avgErrorTest[t] +=
                                            currCmet.dist(centroids[cIndex],
                                            testConfig[cIndex].getInstance(p));
                                }
                            }
                        }
                        avgErrorTest[t] /= dsetTest.size();
                        avgErrTest += avgErrorTest[t];
                        currIndex = -1;
                        // Check for configuration entropy on original labels.
                        split = new ArrayList[numNonEmpty];
                        for (int cIndex = 0; cIndex < split.length; cIndex++) {
                            split[cIndex] = new ArrayList(1500);
                        }
                        for (int cIndex = 0; cIndex < testConfig.length;
                                cIndex++) {
                            if (testConfig[cIndex] != null
                                    && testConfig[cIndex].size() > 0) {
                                ++currIndex;
                                for (int kIndex = 0; kIndex
                                        < testConfig[cIndex].indexes.size();
                                        kIndex++) {
                                    if (testLabels[testConfig[cIndex].indexes.
                                            get(kIndex)] != -1) {
                                        split[currIndex].add(
                                                testLabels[testConfig[cIndex].
                                                indexes.get(kIndex)]);
                                    }
                                }
                            }
                        }
                        avgClusterEntropyTest[t] =
                                Info.evaluateInfoOfCategorySplit(
                                split, numCategories);
                        avgEntropyTest += avgClusterEntropyTest[t];
                        if (calculatePerplexity) {
                            GaussianMixtureModel gmm =
                                    new GaussianMixtureModel(testConfig);
                            perp[t] = Perplexity.getModelPerplexity(
                                    gmm, dsetTest.getDataAsArray());
                            if (perp[t] > 1) {
                                numNonOnePerplexity++;
                                perpAvg += perp[t];
                            }
                        }
                    }
                    synchronized (this) {
                        // This generates a stream in the command line that
                        // resembles a progress bar of the fastest among the
                        // tested approaches.
                        if (t + 1 > globalIterationCounter) {
                            globalIterationCounter = t + 1;
                            System.out.print("|");
                            if (globalIterationCounter % 5 == 0) {
                                System.out.print(" ");
                            }
                        }
                    }
                    System.gc();
                }
                avgSil /= timesOnDataSet;
                avgErr /= timesOnDataSet;
                avgTime /= timesOnDataSet;
                avgEntropy /= timesOnDataSet;
                avgDunn /= timesOnDataSet;
                avgRand /= timesOnDataSet;
                avgIsolation /= timesOnDataSet;
                avgHA /= timesOnDataSet;
                avgHB /= timesOnDataSet;
                avgAHA /= timesOnDataSet;
                avgAHB /= timesOnDataSet;
                avgREGA /= timesOnDataSet;
                avgREGB /= timesOnDataSet;
                avgRandStability /= timesOnDataSet;
                if (numNonOnePerplexity > 0) {
                    perpAvg /= numNonOnePerplexity;
                } else {
                    perpAvg = 1;
                }
                avgRandStabilityTest /= timesOnDataSet;
                avgSilTest /= timesOnDataSet;
                avgErrTest /= timesOnDataSet;
                avgDunnTest /= timesOnDataSet;
                avgRandTest /= timesOnDataSet;
                avgIsolationTest /= timesOnDataSet;
                avgEntropyTest /= timesOnDataSet;
                avgHATest /= timesOnDataSet;
                avgHBTest /= timesOnDataSet;
                avgAHATest /= timesOnDataSet;
                avgAHBTest /= timesOnDataSet;
                avgREGATest /= timesOnDataSet;
                avgREGBTest /= timesOnDataSet;
                long endTime = System.nanoTime();
                File outFile = new File(currOutDSDir, cName + "Report.csv");
                try (PrintWriter pw = new PrintWriter(
                        new FileWriter(outFile));) {
                    pw.println("RAND_STABILITY" + ", "
                            + "RAND_QUALITY" + ", "
                            + "ISOLATION" + ", "
                            + "DUNN" + ", "
                            + "SILHOUETTE" + ", "
                            + "AVG_ERROR" + ", "
                            + "AVG_CLUSTER_ENTROPY" + ", "
                            + "HUBS_A" + ", "
                            + "HUBS_B" + ", "
                            + "ANTIHUBS_A" + ", "
                            + "ANTIHUBS_B" + ", "
                            + "REGULARS_A" + ", "
                            + "REGULARS_B,"
                            + "time ");
                    for (t = 0; t < timesOnDataSet; t++) {
                        pw.println(randStability[t] + ", " + randValues[t]
                                + ", " + isolationValues[t] + ", "
                                + dunnValues[t] + ", " + silScores[t] + ", "
                                + avgError[t] + ", " + avgClusterEntropy[t]
                                + ", " + HASCORES[t] + ", " + HBSCORES[t] + ", "
                                + AHASCORES[t] + ", " + AHBSCORES[t] + ", "
                                + REGASCORES[t] + ", " + REGBSCORES[t] + ","
                                + elapsedTimes[t]);
                    }
                    pw.println(avgRandStability + ", " + avgRand + ", "
                            + avgIsolation + ", " + avgDunn + ", " + avgSil
                            + ", " + avgErr + ", " + avgEntropy + ", " + avgHA
                            + ", " + avgHB + ", " + avgAHA + ", " + avgAHB
                            + ", " + avgREGA + ", " + avgREGB + ", " + avgTime);
                } catch (Exception e) {
                    throw e;
                }
                File outFileTest = new File(currOutDSDir, cName
                        + "ReportTestSplit.csv");
                try (PrintWriter pwTest = new PrintWriter(
                        new FileWriter(outFileTest))) {
                    pwTest.println("RAND_STABILITY" + ", " + "RAND_QUALITY"
                            + ", " + "ISOLATION" + ", " + "DUNN" + ", "
                            + "SILHOUETTE" + ", " + "AVG_ERROR" + ", "
                            + "AVG_CLUSTER_ENTROPY" + ", " + "HUBS_A" + ", "
                            + "HUBS_B" + ", " + "ANTIHUBS_A" + ", "
                            + "ANTIHUBS_B" + ", " + "REGULARS_A" + ", "
                            + "REGULARS_B," + "PERPLEXITY");
                    for (t = 0; t < timesOnDataSet; t++) {
                        pwTest.println(randStabilityTest[t] + ", "
                                + randValuesTest[t] + ", "
                                + isolationValuesTest[t] + ", "
                                + dunnValuesTest[t] + ", " + silScoresTest[t]
                                + ", " + avgErrorTest[t] + ", "
                                + avgClusterEntropyTest[t] + ", "
                                + HASCORESTest[t] + ", " + HBSCORESTest[t]
                                + ", " + AHASCORESTest[t] + ", "
                                + AHBSCORESTest[t] + ", " + REGASCORESTest[t]
                                + ", " + REGBSCORESTest[t] + ", " + perp[t]);
                    }
                    pwTest.println(avgRandStabilityTest + ", " + avgRandTest
                            + ", " + avgIsolationTest + ", " + avgDunnTest
                            + ", " + avgSilTest + ", " + avgErrTest + ", "
                            + avgEntropyTest + ", " + avgHATest + ", "
                            + avgHBTest + ", " + avgAHATest + ", " + avgAHBTest
                            + ", " + avgREGATest + ", " + avgREGBTest + ", "
                            + perpAvg);
                } catch (Exception e) {
                    throw e;
                }
            } catch (Exception e) {
                System.err.println("Error while testing " + cName);
                System.err.println(e.getMessage());
            }
        }
    }

    /**
     * An optional utility method for normalizing a distance matrix. Not used by
     * default.
     *
     * @param dMat
     */
    public static void normalizeDMat(float[][] dMat) {
        float maxValue = DataMineConstants.EPSILON;
        float minValue = Float.MAX_VALUE;
        for (int i = 0; i < dMat.length; i++) {
            for (int j = 0; j < dMat[i].length; j++) {
                if (DataMineConstants.isAcceptableFloat(dMat[i][j])) {
                    maxValue = Math.max(maxValue, dMat[i][j]);
                    minValue = Math.min(minValue, dMat[i][j]);
                }
            }
        }
        // So that the distances are not actually zero in any case, which is
        // useful to avoid some pathological normalization issues.
        minValue = minValue - 0.5f;
        if (maxValue > DataMineConstants.EPSILON
                && DataMineConstants.isAcceptableFloat(maxValue)
                && minValue < Float.MAX_VALUE
                && DataMineConstants.isAcceptableFloat(minValue)) {
            for (int i = 0; i < dMat.length; i++) {
                for (int j = 0; j < dMat[i].length; j++) {
                    dMat[i][j] = (dMat[i][j] - minValue)
                            / (maxValue - minValue);
                }
            }
        }
    }

    /**
     * Runs the batch clustering testing, as specified in the configuration
     * file.
     *
     * @param args One argument that is the path to the testing configuration
     * file.
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            System.out.println("1 parameter - file with test configuration");
            return;
        }
        File inConfigFile = new File(args[0]);
        BatchClusteringTester tester = new BatchClusteringTester(inConfigFile);
        tester.loadParameters();
        tester.runAllTests();
    }
}
