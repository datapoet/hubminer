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
package data.neighbors.hubness;

import configuration.BatchHubnessAnalysisConfig;
import data.neighbors.NeighborSetFinder;
import data.neighbors.SharedNeighborFinder;
import data.representation.DataSet;
import data.representation.sparse.BOWDataSet;
import data.representation.util.DataMineConstants;
import distances.primary.CombinedMetric;
import distances.secondary.LocalScalingCalculator;
import distances.secondary.MutualProximityCalculator;
import distances.secondary.NICDMCalculator;
import distances.secondary.snd.SharedNeighborCalculator;
import distances.sparse.SparseCombinedMetric;
import filters.TFIDF;
import ioformat.DistanceMatrixIO;
import ioformat.FileUtil;
import ioformat.SupervisedLoader;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.ArrayList;
import learning.supervised.evaluation.cv.BatchClassifierTester.SecondaryDistance;
import util.BasicMathUtil;
import util.NonHomogenityCalculator;

/**
 * This class acts as a script for batch analysis of hubness stats on a series
 * of datasets.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class BatchHubnessAnalyzer {

    private SecondaryDistance secondaryDistanceType;
    // Neighborhood size to use for secondary distances.
    private int secondaryDistanceK = 50;
    // Normalization types.

    public enum Normalization {

        NONE, STANDARDIZE, NORM_01, TFIDF;
    }
    // The normalization type to actually use in the experiments.
    private Normalization normType = Normalization.STANDARDIZE;
    // The upper limit on the neighborhood sizes to examine.
    private int kMax = 50;
    // Noise and mislabeling levels to vary, with default values.
    private float noiseMin = 0, noiseMax = 0, noiseStep = 1, mlMin = 0,
            mlMax = 0, mlStep = 1;
    // Input and output files and directories.
    private File inConfigFile, inDir, outDir, currOutDSDir, mlWeightsDir;
    // Paths to the datasets that are being processed.
    private ArrayList<String> dsPaths = new ArrayList<>(100);
    // A list of metrics corresponding to the datasets.
    private ArrayList<CombinedMetric> dsMetric = new ArrayList<>(100);
    // Data holders.
    private DataSet originalDSet, currDSet;
    // Object for distance calculations.
    private CombinedMetric cmet;
    // Number of categories in the data.
    private int numCategories;
    // Directory containing the distances.
    private File distancesDir;
    // The number of threads used for distance matrix and kNN set calculations.
    private int numCommonThreads = 8;

    /**
     * Initialization.
     *
     * @param inConfigFile File containing the experiment configuration
     * parameters.
     */
    public BatchHubnessAnalyzer(File inConfigFile) {
        this.inConfigFile = inConfigFile;
    }
    
    /**
     * This method loads the configuration from the configuration object.
     * 
     * @param conf BatchHubnessAnalysisConfig that is the configuration object.
     */
    public void loadFromConfigurationObject(BatchHubnessAnalysisConfig conf) {
        secondaryDistanceType = conf.secondaryDistanceType;
        secondaryDistanceK = conf.secondaryDistanceK;
        normType = conf.normType;
        kMax = conf.kMax;
        noiseMin = conf.noiseMin;
        noiseMax = conf.noiseMax;
        noiseStep = conf.noiseStep;
        mlMin = conf.mlMin;
        mlMax = conf.mlMax;
        mlStep = conf.mlStep;
        inDir = conf.inDir;
        outDir = conf.outDir;
        mlWeightsDir = conf.mlWeightsDir;
        dsPaths = conf.dsPaths;
        dsMetric = conf.dsMetric;
        distancesDir = conf.distancesDir;
        numCommonThreads = conf.numCommonThreads;
    }

    /**
     * This method runs the script and performs batch analysis of the stats
     * relevant for interpreting the hubness of the data on a series on
     * datasets.
     *
     * @throws Exception
     */
    public void runAllTests() throws Exception {
        int dsIndex = 0;
        for (String dsPath : dsPaths) {
            File dsFile = new File(dsPath);
            originalDSet = SupervisedLoader.loadData(dsFile, false);
            System.out.println("Testing on: " + dsPath);
            // Standardize the categories into [0..numCat-1] range.
            originalDSet.standardizeCategories();
            if (normType != Normalization.NONE) {
                System.out.print("Normalizing features-");
                if (normType == Normalization.NORM_01) {
                    // Normalize all float features to the [0, 1] range.
                    originalDSet.normalizeFloats();
                } else if (normType == Normalization.STANDARDIZE) {
                    // Standardize all float values.
                    originalDSet.standardizeAllFloats();
                } else if (normType == Normalization.TFIDF) {
                    // Perform TFIDF weighting.
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
            // Get the number of classes in the data.
            numCategories = originalDSet.countCategories();
            // Initialize the counter that fires occasional garbage collection.
            int memCleanCount = 0;
            // Iterate over the desired range of noise and mislabeling rates.
            for (float noise = noiseMin; noise <= noiseMax;
                    noise += noiseStep) {
                for (float ml = mlMin; ml <= mlMax; ml += mlStep) {
                    if (++memCleanCount % 5 == 0) {
                        // Try some clean-up, if possible.
                        System.gc();
                    }
                    // Make a copy of the original data.
                    currDSet = originalDSet.copy();
                    if (ml > 0) {
                        // First check if any mislabeling instance weights
                        // were provided, that make certain mislabelings
                        // more probable than others.
                        String weightsPath = null;
                        if (mlWeightsDir != null) {
                            if (!(cmet instanceof SparseCombinedMetric)) {
                                String metricDir = cmet.getFloatMetric()
                                        != null
                                        ? cmet.getFloatMetric().getClass().
                                        getName() : cmet.getIntegerMetric().
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
                                                + ((SparseCombinedMetric) cmet).
                                                getSparseMetric().
                                                getClass().getName()
                                                + File.separator
                                                + "ml_weights.txt";
                                        break;
                                    case NORM_01:
                                        weightsPath = "NORM01"
                                                + File.separator
                                                + ((SparseCombinedMetric) cmet).
                                                getSparseMetric().
                                                getClass().getName()
                                                + File.separator
                                                + "ml_weights.txt";
                                        break;
                                    case STANDARDIZE:
                                        weightsPath = "STANDARDIZED"
                                                + File.separator
                                                + ((SparseCombinedMetric) cmet).
                                                getSparseMetric().
                                                getClass().getName()
                                                + File.separator
                                                + "ml_weights.txt";
                                        break;
                                    case TFIDF:
                                        weightsPath = "TFIDF"
                                                + File.separator
                                                + ((SparseCombinedMetric) cmet).
                                                getSparseMetric().
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
                        // Induce Gaussian float feature noise.
                        currDSet.addGaussianNoiseToNormalizedCollection(
                                noise, 0.1f);
                    }
                    // Generate the appropriate output directory.
                    currOutDSDir = new File(outDir,
                            dsFile.getName().substring(0, dsFile.getName().
                            lastIndexOf(".")) + File.separator + "k" + kMax
                            + File.separator + "ml" + ml + File.separator
                            + "noise" + noise);
                    FileUtil.createDirectory(currOutDSDir);
                    cmet = dsMetric.get(dsIndex);
                    // Count the number of zero vectors, empty data
                    // representations in the data instances.
                    int zeroVectorsNum = currDSet.countZeroFloatVectors();
                    float[] classPriors = currDSet.getClassPriors();
                    // Calculate or load the distance matrix.
                    NeighborSetFinder nsf = new NeighborSetFinder(currDSet,
                            cmet);
                    // Determine the correct distance matrix path.
                    String dMatPath = null;
                    if (distancesDir != null) {
                        if (!(cmet instanceof SparseCombinedMetric)) {
                            switch (normType) {
                                case NONE:
                                    dMatPath = "NO" + File.separator
                                            + cmet.getFloatMetric().getClass().
                                            getName() + File.separator
                                            + "dMat.txt";
                                    break;
                                case NORM_01:
                                    dMatPath = "NORM01" + File.separator
                                            + cmet.getFloatMetric().getClass().
                                            getName() + File.separator
                                            + "dMat.txt";
                                    break;
                                case STANDARDIZE:
                                    dMatPath = "STANDARDIZED" + File.separator
                                            + cmet.getFloatMetric().getClass().
                                            getName() + File.separator
                                            + "dMat.txt";
                                    break;
                                case TFIDF:
                                    dMatPath = "TFIDF" + File.separator
                                            + cmet.getFloatMetric().getClass().
                                            getName() + File.separator
                                            + "dMat.txt";
                                    break;
                            }
                        } else {
                            switch (normType) {
                                case NONE:
                                    dMatPath = "NO" + File.separator
                                            + ((SparseCombinedMetric) cmet).
                                            getSparseMetric().getClass().
                                            getName() + File.separator
                                            + "dMat.txt";
                                    break;
                                case NORM_01:
                                    dMatPath = "NORM01" + File.separator
                                            + ((SparseCombinedMetric) cmet).
                                            getSparseMetric().getClass().
                                            getName() + File.separator
                                            + "dMat.txt";
                                    break;
                                case STANDARDIZE:
                                    dMatPath = "STANDARDIZED" + File.separator
                                            + ((SparseCombinedMetric) cmet).
                                            getSparseMetric().getClass().
                                            getName() + File.separator
                                            + "dMat.txt";
                                    break;
                                case TFIDF:
                                    dMatPath = "TFIDF" + File.separator
                                            + ((SparseCombinedMetric) cmet).
                                            getSparseMetric().getClass().
                                            getName() + File.separator
                                            + "dMat.txt";
                                    break;
                            }
                        }
                    }
                    File dMatFile = null;
                    float[][] distMat = null;
                    // First initialize with a dummy class object, to avoid
                    // some warnings and exceptions in the pathological cases.
                    Class cmetClass = originalDSet.getClass();
                    // Determine the proper metric class and the distance matrix
                    // file.
                    if (dMatPath != null && noise == 0) {
                        dMatFile = new File(
                                distancesDir, dsFile.getName().substring(0,
                                dsFile.getName().lastIndexOf("."))
                                + File.separator + dMatPath);
                        cmetClass = Class.forName(
                                dMatFile.getParentFile().getName());
                    }
                    if (distMat == null) {
                        if (dMatFile == null
                                || !dMatFile.exists()
                                || !(cmetClass.isInstance(
                                cmet.getFloatMetric()))) {
                            // If the file does not exist or the loaded name is
                            // not an appropriate float metric, then calculate
                            // the distances with the specified metric.
                            System.out.print("Calculating distances-");
                            distMat = currDSet.calculateDistMatrixMultThr(
                                    cmet, numCommonThreads);
                            System.out.println("-distances calculated.");
                            if (dMatFile != null) {
                                // If the file path is good, persist the newly
                                // calculated distance matrix.
                                DistanceMatrixIO.printDMatToFile(
                                        distMat, dMatFile);
                            }
                        } else {
                            // Load the distances from an existing source.
                            System.out.print("Loading distances-");
                            distMat = DistanceMatrixIO.loadDMatFromFile(
                                    dMatFile);
                            System.out.println("-distance loaded from file: "
                                    + dMatFile.getPath());
                        }
                    }
                    if (secondaryDistanceType == SecondaryDistance.NONE) {
                        // Use the primary distance matrix for kNN calculations.
                        nsf.setDistances(distMat);
                    } else {
                        // Use the secondary shared-neighbor distances.
                        if (secondaryDistanceType == SecondaryDistance.SIMCOS) {
                            // The simcos secondary distance.
                            NeighborSetFinder nsfSND =
                                    new NeighborSetFinder(currDSet, distMat,
                                    cmet);
                            nsfSND.calculateNeighborSetsMultiThr(
                                    secondaryDistanceK, numCommonThreads);
                            SharedNeighborFinder snf =
                                    new SharedNeighborFinder(nsfSND);
                            snf.countSharedNeighbors();
                            SharedNeighborCalculator snc =
                                    new SharedNeighborCalculator(snf,
                                    SharedNeighborCalculator.
                                    WeightingType.NONE);
                            float[][] simcosSimMat =
                                    snf.getSharedNeighborCounts();
                            float[][] simcosDMat =
                                    new float[simcosSimMat.length][];
                            // Transform similarities into distances.
                            for (int i = 0; i < simcosDMat.length; i++) {
                                simcosDMat[i] =
                                        new float[simcosSimMat[i].length];
                                for (int j = 0;
                                        j < simcosDMat[i].length; j++) {
                                    simcosDMat[i][j] = secondaryDistanceK
                                            - simcosSimMat[i][j];
                                }
                            }
                            // Normalize the scores.
                            float max = 0;
                            float min = Float.MAX_VALUE;
                            for (int i = 0; i < simcosDMat.length; i++) {
                                for (int j = 0; j < simcosDMat[i].length; j++) {
                                    max = Math.max(max, simcosDMat[i][j]);
                                    min = Math.min(min, simcosDMat[i][j]);
                                }
                            }
                            for (int i = 0; i < simcosDMat.length; i++) {
                                for (int j = 0; j < simcosDMat[i].length; j++) {
                                    simcosDMat[i][j] = (simcosDMat[i][j] - min)
                                            / (max - min);
                                }
                            }
                            // Use the simcos distance matrix for kNN set
                            // calculations.
                            nsf = new NeighborSetFinder(originalDSet,
                                    simcosDMat, snc);
                        } else if (secondaryDistanceType
                                == SecondaryDistance.SIMHUB) {
                            // The hubness-aware simhub secondary distance
                            // measure based on shared-neighbor methodology.
                            NeighborSetFinder nsfSND = new NeighborSetFinder(
                                    currDSet, distMat, cmet);
                            nsfSND.calculateNeighborSetsMultiThr(
                                    secondaryDistanceK, numCommonThreads);
                            SharedNeighborFinder snf =
                                    new SharedNeighborFinder(nsfSND, 5);
                            snf.obtainWeightsFromHubnessInformation(0);
                            snf.countSharedNeighbors();
                            SharedNeighborCalculator snc =
                                    new SharedNeighborCalculator(snf,
                                    SharedNeighborCalculator.
                                    WeightingType.HUBNESS_INFORMATION);
                            float[][] simhubSimMat =
                                    snf.getSharedNeighborCounts();
                            // Transform similarities into distances.
                            float[][] simhubDMat =
                                    new float[simhubSimMat.length][];
                            for (int i = 0; i < simhubDMat.length; i++) {
                                simhubDMat[i] =
                                        new float[simhubSimMat[i].length];
                                for (int j = 0; j < simhubDMat[i].length; j++) {
                                    simhubDMat[i][j] = secondaryDistanceK
                                            - simhubSimMat[i][j];
                                }
                            }
                            // Normalize the scores.
                            float max = 0;
                            float min = Float.MAX_VALUE;
                            for (int i = 0; i < simhubDMat.length; i++) {
                                for (int j = 0; j < simhubDMat[i].length; j++) {
                                    max = Math.max(max, simhubDMat[i][j]);
                                    min = Math.min(min, simhubDMat[i][j]);
                                }
                            }
                            for (int i = 0; i < simhubDMat.length; i++) {
                                for (int j = 0; j < simhubDMat[i].length; j++) {
                                    simhubDMat[i][j] = (simhubDMat[i][j] - min)
                                            / (max - min);
                                }
                            }
                            nsf = new NeighborSetFinder(
                                    originalDSet, simhubDMat, snc);
                        } else if (secondaryDistanceType
                                == SecondaryDistance.MP) {
                            // Mutual proximity secondary similarity measure.
                            NeighborSetFinder nsfSecondary =
                                    new NeighborSetFinder(currDSet, distMat,
                                    cmet);
                            nsfSecondary.calculateNeighborSetsMultiThr(
                                    secondaryDistanceK, numCommonThreads);
                            MutualProximityCalculator calc =
                                    new MutualProximityCalculator(
                                    nsfSecondary.getDistances(),
                                    nsfSecondary.getDataSet(),
                                    nsfSecondary.getCombinedMetric());
                            float[][] mpDistMat =
                                    calc.calculateSecondaryDistMatrixMultThr(
                                    nsfSecondary, numCommonThreads);
                            // Normalize the scores.
                            float max = 0;
                            float min = Float.MAX_VALUE;
                            for (int i = 0; i < mpDistMat.length; i++) {
                                for (int j = 0; j < mpDistMat[i].length; j++) {
                                    max = Math.max(max, mpDistMat[i][j]);
                                    min = Math.min(min, mpDistMat[i][j]);
                                }
                            }
                            for (int i = 0; i < mpDistMat.length; i++) {
                                for (int j = 0; j < mpDistMat[i].length; j++) {
                                    mpDistMat[i][j] = (mpDistMat[i][j] - min)
                                            / (max - min);
                                }
                            }
                            nsf = new NeighborSetFinder(
                                    originalDSet, mpDistMat, calc);
                        } else if (secondaryDistanceType
                                == SecondaryDistance.LS) {
                            // Local scaling secondary distance measure.
                            NeighborSetFinder nsfSecondary =
                                    new NeighborSetFinder(currDSet, distMat,
                                    cmet);
                            nsfSecondary.calculateNeighborSetsMultiThr(
                                    secondaryDistanceK, numCommonThreads);
                            LocalScalingCalculator lsc =
                                    new LocalScalingCalculator(nsfSecondary);
                            float[][] lsDistMat = lsc.
                                    getTransformedDMatFromNSFPrimaryDMat();
                            // Normalize the scores.
                            float max = 0;
                            float min = Float.MAX_VALUE;
                            for (int i = 0; i < lsDistMat.length; i++) {
                                for (int j = 0; j < lsDistMat[i].length; j++) {
                                    max = Math.max(max, lsDistMat[i][j]);
                                    min = Math.min(min, lsDistMat[i][j]);
                                }
                            }
                            for (int i = 0; i < lsDistMat.length; i++) {
                                for (int j = 0; j < lsDistMat[i].length; j++) {
                                    lsDistMat[i][j] = (lsDistMat[i][j] - min)
                                            / (max - min);
                                }
                            }
                            nsf = new NeighborSetFinder(
                                    originalDSet, lsDistMat, lsc);
                        } else if (secondaryDistanceType
                                == SecondaryDistance.NICDM) {
                            // NICDM secondary distance measure.
                            NeighborSetFinder nsfSecondary =
                                    new NeighborSetFinder(currDSet, distMat,
                                    cmet);
                            nsfSecondary.calculateNeighborSetsMultiThr(
                                    secondaryDistanceK, numCommonThreads);
                            NICDMCalculator nicdmCalc =
                                    new NICDMCalculator(nsfSecondary);
                            float[][] lsDistMat = nicdmCalc.
                                    getTransformedDMatFromNSFPrimaryDMat();
                            // Normalize the scores.
                            float max = 0;
                            float min = Float.MAX_VALUE;
                            for (int i = 0; i < lsDistMat.length; i++) {
                                for (int j = 0; j < lsDistMat[i].length; j++) {
                                    max = Math.max(max, lsDistMat[i][j]);
                                    min = Math.min(min, lsDistMat[i][j]);
                                }
                            }
                            for (int i = 0; i < lsDistMat.length; i++) {
                                for (int j = 0; j < lsDistMat[i].length; j++) {
                                    lsDistMat[i][j] = (lsDistMat[i][j] - min)
                                            / (max - min);
                                }
                            }
                            nsf = new NeighborSetFinder(
                                    originalDSet, lsDistMat, nicdmCalc);
                        }
                    }
                    nsf.calculateNeighborSets(kMax);
                    // Initialize the hubness stats calculators.
                    HubnessAboveThresholdExplorer hte =
                            new HubnessAboveThresholdExplorer(1, true, nsf);
                    HubnessSkewAndKurtosisExplorer hske =
                            new HubnessSkewAndKurtosisExplorer(nsf);
                    HubnessExtremesGrabber heg =
                            new HubnessExtremesGrabber(true, nsf);
                    HubnessVarianceExplorer hve =
                            new HubnessVarianceExplorer(nsf);
                    TopHubsClusterUtil thcu =
                            new TopHubsClusterUtil(nsf);
                    KNeighborEntropyExplorer knee =
                            new KNeighborEntropyExplorer(nsf, numCategories);
                    // Use the hubness stats calculators to obtain the stats
                    // for interpreting the hubness of the data.
                    // The percentages of points that occur at least once.
                    float[] aboveZeroArray = hte.getThresholdPercentageArray();
                    hske.calcSkewAndKurtosisArrays();
                    // Skewness of the neighbor occurrence frequency
                    // distribution.
                    float[] skewArray = hske.getOccFreqsSkewnessArray();
                    // Kurtosis of the neighbor occurrence frequency
                    // distribution.
                    float[] kurtosisArray = hske.getOccFreqsKurtosisArray();
                    // Highest neighbor occurrence frequencies.
                    float[][] highestOccFreqs =
                            heg.getHubnessExtremesForKValues(15);
                    float[] stDevArray = hve.getStDevForKRange();
                    thcu.calcTopHubnessDiamAndAvgDist(10);
                    float[] topHubClustDiamsArr =
                            thcu.getTopHubClusterDiameters();
                    float[] topHubClustAvgDistArr =
                            thcu.getTopHubClusterAvgDists();
                    knee.calculateAllKNNEntropyStats();
                    // Direct and reverse kNN entropy distributions.
                    float[] kEntropiesMeans = knee.getDirectEntropyMeans();
                    float[] kRNNEntropiesMeans = knee.getReverseEntropyMeans();
                    float[] kEntropiesStDevs =
                            knee.getDirectEntropyStDevs();
                    float[] kRNNEntropiesStDevs =
                            knee.getReverseEntropyStDevs();
                    float[] kEntropiesSkews =
                            knee.getDirectEntropySkews();
                    float[] kRNNEntropiesSkews =
                            knee.getReverseEntropySkews();
                    float[] kEntropiesKurtosis =
                            knee.getDirectEntropyKurtosisVals();
                    float[] kRNNEntropiesKurtosis =
                            knee.getReverseEntropyKurtosisVals();
                    float[] entDiffs = knee.
                            getAverageDirectAndReverseEntropyDifs();
                    // Bad neighbor occurrence frequencies.
                    float[] bhArray = nsf.getLabelMismatchPercsAllK(kMax);
                    float[][][] gClasstoClassHubness = new float[kMax][][];
                    for (int kVal = 1; kVal <= kMax; kVal++) {
                        gClasstoClassHubness[kVal - 1] =
                                nsf.getGlobalClassToClassForKforFuzzy(
                                kVal, numCategories, 0.01f, true);
                    }

                    File currOutFile = new File(
                            currOutDSDir, "hubnessOverview.txt");
                    // Print out the results.
                    try (PrintWriter pw = new PrintWriter(
                            new FileWriter(currOutFile))) {
                        pw.println("dataset: " + dsFile.getName());
                        pw.println("k_max: " + kMax);
                        pw.println("noise: " + noise);
                        pw.println("ml: " + ml);
                        pw.println("instances: " + originalDSet.size());
                        pw.println("numCat: " + numCategories);
                        pw.println("nZeroVects: " + zeroVectorsNum);
                        pw.print("class priors: ");
                        for (int cIndex = 0; cIndex < numCategories; cIndex++) {
                            pw.print(BasicMathUtil.makeADecimalCutOff(
                                    classPriors[cIndex], 3));
                            pw.print(" ");
                        }
                        pw.println();
                        pw.println("RelativeImbalance "
                                + NonHomogenityCalculator.
                                calculateNonHomogeneity(classPriors));
                        pw.println("dim: " + originalDSet.fAttrNames.length);
                        pw.println("-------------------------------------");
                        pw.println("stDevArray: ");
                        pw.print(BasicMathUtil.makeADecimalCutOff(
                                stDevArray[0], 3));
                        for (int i = 1; i < stDevArray.length; i++) {
                            pw.print("," + BasicMathUtil.makeADecimalCutOff(
                                    stDevArray[i], 3));
                        }
                        pw.println();
                        pw.println("-------------------------------------");
                        pw.println("skewArray: ");
                        pw.print(BasicMathUtil.makeADecimalCutOff(
                                skewArray[0], 3));
                        for (int i = 1; i < skewArray.length; i++) {
                            pw.print("," + BasicMathUtil.makeADecimalCutOff(
                                    skewArray[i], 3));
                        }
                        pw.println();
                        pw.println("-------------------------------------");
                        pw.println("kurtosisArray: ");
                        pw.print(BasicMathUtil.makeADecimalCutOff(
                                kurtosisArray[0], 3));
                        for (int i = 1; i < kurtosisArray.length; i++) {
                            pw.print("," + BasicMathUtil.makeADecimalCutOff(
                                    kurtosisArray[i], 3));
                        }
                        pw.println();
                        pw.println("-------------------------------------");
                        pw.println("bad hubness: ");
                        pw.print(BasicMathUtil.makeADecimalCutOff(
                                bhArray[0], 3));
                        for (int i = 1; i < bhArray.length; i++) {
                            pw.print("," + BasicMathUtil.makeADecimalCutOff(
                                    bhArray[i], 3));
                        }
                        pw.println();
                        pw.println("-------------------------------------");
                        pw.println("kEntropyMeans: ");
                        pw.print(BasicMathUtil.makeADecimalCutOff(
                                kEntropiesMeans[0], 3));
                        for (int i = 1; i < kEntropiesMeans.length; i++) {
                            pw.print(","
                                    + BasicMathUtil.makeADecimalCutOff(
                                    kEntropiesMeans[i], 3));
                        }
                        pw.println();
                        pw.println("-------------------------------------");
                        pw.println("kEntropyStDevs: ");
                        pw.print(BasicMathUtil.makeADecimalCutOff(
                                kEntropiesStDevs[0], 3));
                        for (int i = 1; i < kEntropiesStDevs.length; i++) {
                            pw.print(","
                                    + BasicMathUtil.makeADecimalCutOff(
                                    kEntropiesStDevs[i], 3));
                        }
                        pw.println();
                        pw.println("-------------------------------------");
                        pw.println("kEntropySkews: ");
                        pw.print(BasicMathUtil.makeADecimalCutOff(
                                kEntropiesSkews[0], 3));
                        for (int i = 1; i < kEntropiesSkews.length; i++) {
                            pw.print(","
                                    + BasicMathUtil.makeADecimalCutOff(
                                    kEntropiesSkews[i], 3));
                        }
                        pw.println();
                        pw.println("-------------------------------------");
                        pw.println("kEntropyKurtosis: ");
                        pw.print(BasicMathUtil.makeADecimalCutOff(
                                kEntropiesKurtosis[0], 3));
                        for (int i = 1; i < kEntropiesKurtosis.length; i++) {
                            pw.print(","
                                    + BasicMathUtil.makeADecimalCutOff(
                                    kEntropiesKurtosis[i], 3));
                        }
                        pw.println();
                        pw.println("-------------------------------------");
                        pw.println("kRNNEntropyMeans: ");
                        pw.print(BasicMathUtil.makeADecimalCutOff(
                                kRNNEntropiesMeans[0], 3));
                        for (int i = 1; i < kRNNEntropiesMeans.length; i++) {
                            pw.print(","
                                    + BasicMathUtil.makeADecimalCutOff(
                                    kRNNEntropiesMeans[i], 3));
                        }
                        pw.println();
                        pw.println("-------------------------------------");
                        pw.println("kRNNEntropyStDevs: ");
                        pw.print(BasicMathUtil.makeADecimalCutOff(
                                kRNNEntropiesStDevs[0], 3));
                        for (int i = 1; i < kRNNEntropiesStDevs.length; i++) {
                            pw.print(","
                                    + BasicMathUtil.makeADecimalCutOff(
                                    kRNNEntropiesStDevs[i], 3));
                        }
                        pw.println();
                        pw.println("-------------------------------------");
                        pw.println("kRNNEntropySkews: ");
                        pw.print(BasicMathUtil.makeADecimalCutOff(
                                kRNNEntropiesSkews[0], 3));
                        for (int i = 1; i < kRNNEntropiesSkews.length; i++) {
                            pw.print(","
                                    + BasicMathUtil.makeADecimalCutOff(
                                    kRNNEntropiesSkews[i], 3));
                        }
                        pw.println();
                        pw.println("-------------------------------------");
                        pw.println("kRNNEntropyKurtosis: ");
                        pw.print(BasicMathUtil.makeADecimalCutOff(
                                kRNNEntropiesKurtosis[0], 3));
                        for (int i = 1; i < kRNNEntropiesKurtosis.length;
                                i++) {
                            pw.print(","
                                    + BasicMathUtil.makeADecimalCutOff(
                                    kRNNEntropiesKurtosis[i], 3));
                        }
                        pw.println();
                        pw.println("-------------------------------------");
                        pw.println("kEnt - khEnt avgs: ");
                        pw.print(BasicMathUtil.makeADecimalCutOff(
                                entDiffs[0], 3));
                        for (int i = 1; i < entDiffs.length; i++) {
                            pw.print("," + BasicMathUtil.makeADecimalCutOff(
                                    entDiffs[i], 3));
                        }
                        pw.println();
                        pw.println("-------------------------------------");
                        pw.println("Hubness above zero percentage Array: ");
                        pw.print(BasicMathUtil.makeADecimalCutOff(
                                aboveZeroArray[0], 2));
                        for (int i = 1; i < aboveZeroArray.length; i++) {
                            pw.print("," + BasicMathUtil.makeADecimalCutOff(
                                    aboveZeroArray[i], 2));
                        }
                        pw.println();
                        hte = new HubnessAboveThresholdExplorer(2, true, nsf);
                        aboveZeroArray = hte.getThresholdPercentageArray();
                        pw.println("-------------------------------------");
                        pw.println("Hubness above one percentage Array: ");
                        pw.print(BasicMathUtil.makeADecimalCutOff(
                                aboveZeroArray[0], 2));
                        for (int i = 1; i < aboveZeroArray.length; i++) {
                            pw.print("," + BasicMathUtil.makeADecimalCutOff(
                                    aboveZeroArray[i], 2));
                        }
                        pw.println();
                        hte = new HubnessAboveThresholdExplorer(3, true, nsf);
                        aboveZeroArray = hte.getThresholdPercentageArray();
                        pw.println("-------------------------------------");
                        pw.println("Hubness above two percentage Array: ");
                        pw.print(BasicMathUtil.makeADecimalCutOff(
                                aboveZeroArray[0], 2));
                        for (int i = 1; i < aboveZeroArray.length; i++) {
                            pw.print("," + BasicMathUtil.makeADecimalCutOff(
                                    aboveZeroArray[i], 2));
                        }
                        pw.println();
                        hte = new HubnessAboveThresholdExplorer(4, true, nsf);
                        aboveZeroArray = hte.getThresholdPercentageArray();
                        pw.println("-------------------------------------");
                        pw.println("Hubness above three percentage Array: ");
                        pw.print(BasicMathUtil.makeADecimalCutOff(
                                aboveZeroArray[0], 2));
                        for (int i = 1; i < aboveZeroArray.length; i++) {
                            pw.print("," + BasicMathUtil.makeADecimalCutOff(
                                    aboveZeroArray[i], 2));
                        }
                        pw.println();
                        hte = new HubnessAboveThresholdExplorer(5, true, nsf);
                        aboveZeroArray = hte.getThresholdPercentageArray();
                        pw.println("-------------------------------------");
                        pw.println("Hubness above four percentage Array: ");
                        pw.print(BasicMathUtil.makeADecimalCutOff(
                                aboveZeroArray[0], 2));
                        for (int i = 1; i < aboveZeroArray.length; i++) {
                            pw.print("," + BasicMathUtil.makeADecimalCutOff(
                                    aboveZeroArray[i], 2));
                        }
                        pw.println();
                        pw.println("-------------------------------------");
                        pw.println("Top ten hubs diam: ");
                        pw.print(BasicMathUtil.makeADecimalCutOff(
                                topHubClustDiamsArr[0], 2));
                        for (int i = 1; i < topHubClustDiamsArr.length; i++) {
                            pw.print("," + BasicMathUtil.makeADecimalCutOff(
                                    topHubClustDiamsArr[i], 2));
                        }
                        pw.println();
                        pw.println("-------------------------------------");
                        pw.println("Top ten hubs avg within cluster dist: ");
                        pw.print(BasicMathUtil.makeADecimalCutOff(
                                topHubClustAvgDistArr[0], 2));
                        for (int i = 1; i < topHubClustAvgDistArr.length; i++) {
                            pw.print("," + BasicMathUtil.makeADecimalCutOff(
                                    topHubClustAvgDistArr[i], 2));
                        }
                        pw.println();
                        pw.println("-------------------------------------");
                        thcu.calcTopHubnessDiamAndAvgDist(5);
                        topHubClustDiamsArr = thcu.getTopHubClusterDiameters();
                        topHubClustAvgDistArr = thcu.getTopHubClusterAvgDists();
                        pw.println("Top five hubs diam: ");
                        pw.print(BasicMathUtil.makeADecimalCutOff(
                                topHubClustDiamsArr[0], 2));
                        for (int i = 1; i < topHubClustDiamsArr.length; i++) {
                            pw.print("," + BasicMathUtil.makeADecimalCutOff(
                                    topHubClustDiamsArr[i], 2));
                        }
                        pw.println();
                        pw.println("-------------------------------------");
                        pw.println("Top five hubs avg within cluster dist: ");
                        pw.print(BasicMathUtil.makeADecimalCutOff(
                                topHubClustAvgDistArr[0], 2));
                        for (int i = 1; i < topHubClustAvgDistArr.length; i++) {
                            pw.print("," + BasicMathUtil.makeADecimalCutOff(
                                    topHubClustAvgDistArr[i], 2));
                        }
                        pw.println();
                        pw.println("-------------------------------------");
                        pw.println("Highest occurrence frequencies (each line "
                                + "is for one k value, lines go from zero to"
                                + " k_max): ");
                        for (int kVal = 0; kVal < kMax; kVal++) {
                            pw.print("k: " + (kVal + 1) + ":: ");
                            pw.print(BasicMathUtil.makeADecimalCutOff(
                                    highestOccFreqs[kVal][0], 3));
                            for (int i = 1; i < 15; i++) {
                                pw.print("," + BasicMathUtil.
                                        makeADecimalCutOff(
                                        highestOccFreqs[kVal][i], 3));
                            }
                            pw.println();
                        }
                        pw.println("-------------------------------------");
                        pw.println("Global class to class hubness matrices for "
                                + "all K-s: ");
                        for (int kVal = 1; kVal <= kMax; kVal++) {
                            pw.println("k = " + kVal);
                            for (int c1 = 0; c1 < numCategories; c1++) {
                                for (int c2 = 0; c2 < numCategories; c2++) {
                                    pw.print(BasicMathUtil.makeADecimalCutOff(
                                            gClasstoClassHubness[
                                            kVal - 1][c1][c2], 3));
                                    pw.print(" ");
                                }
                                pw.println();
                            }
                            pw.println();
                        }
                    } catch (Exception e) {
                        System.err.println(e.getMessage());
                    }
                }

            }
            dsIndex++;
        }
    }

    /**
     * This method loads the parameters from the configuration file.
     *
     * @throws Exception
     */
    public void loadParameters() throws Exception {
        BatchHubnessAnalysisConfig config = new BatchHubnessAnalysisConfig();
        config.loadParameters(inConfigFile);
        loadFromConfigurationObject(config);
    }

    /**
     * This method runs the batch hubness analysis script.
     *
     * @param args Command line parameters, as specified.
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            System.out.println("1 parameter - file with test configuration");
            System.out.println("Consult HubMiner documentation for details on"
                    + "the configuration file format.");
            return;
        }
        File inConfigFile = new File(args[0]);
        BatchHubnessAnalyzer tester = new BatchHubnessAnalyzer(inConfigFile);
        tester.loadParameters();
        tester.runAllTests();
    }
}
