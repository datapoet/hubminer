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

import data.neighbors.NeighborSetFinder;
import data.neighbors.SharedNeighborFinder;
import data.representation.DataSet;
import data.representation.sparse.BOWDataSet;
import data.representation.util.DataMineConstants;
import distances.primary.CombinedMetric;
import distances.primary.DistanceMeasure;
import distances.secondary.LocalScalingCalculator;
import distances.secondary.MutualProximityCalculator;
import distances.secondary.NICDMCalculator;
import distances.secondary.snd.SharedNeighborCalculator;
import distances.sparse.SparseCombinedMetric;
import distances.sparse.SparseMetric;
import filters.TFIDF;
import ioformat.DistanceMatrixIO;
import ioformat.FileUtil;
import ioformat.IOARFF;
import ioformat.IOCSV;
import ioformat.SupervisedLoader;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import learning.supervised.evaluation.cv.BatchClassifierTester;
import util.BasicMathUtil;

/**
 * This class acts as a script for batch analysis of hubness stats on a series
 * of datasets. It is meant for the multi-label case, when several different
 * label arrays are provided for the datasets. Different datasets are different
 * feature representations of the same underlying data. This is different from
 * the BatchHubnessAnalyzer class, which handles the general batch processing
 * case. However, this class is more efficient for multi-label analysis, as the
 * distance matrices and kNN sets are only calculated once for each data
 * representation.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class MultiLabelBatchHubnessAnalyzer {

    // Normalization types.
    private static final int NORM_STANDARDIZE = 0;
    private static final int NORM_NO = 1;
    private static final int NORM_01 = 2;
    private static final int N_TFIDF = 3;
    // Normalization to use on the features.
    private int normType = NORM_STANDARDIZE;
    private BatchClassifierTester.SecondaryDistance secondaryDistanceType;
    // Neighborhood size to use for secondary distances.
    private int secondaryDistanceK = 50;
    // The upper bound on the neighborhood size to test. All smaller 
    // neighborhood sizes will be examined.
    private int kMax = 50;
    // Noise and mislabeling range definitions, with default values.
    private float noiseMin = 0, noiseMax = 0, noiseStep = 1, mlMin = 0,
            mlMax = 0, mlStep = 1;
    // Input and output files and directories.
    private File inConfigFile, inDir, outDir, currOutDSDir, inLabelFile;
    // Dataset paths.
    private ArrayList<String> dsPaths = new ArrayList<>(100);
    // A list of corresponding dataset metrics.
    private ArrayList<CombinedMetric> dsMetric = new ArrayList<>(100);
    // Data holders.
    private DataSet originalDSet, currDSet;
    // The current metric object.
    private CombinedMetric cmet;
    // Number of categories in the data.
    private int numCategories;
    // Label separator in the label file.
    private String labelSeparator;
    // Directory containing the distances.
    File distancesDir;

    /**
     * Initialization.
     *
     * @param inConfigFile File containing the experiment configuration.
     */
    public MultiLabelBatchHubnessAnalyzer(File inConfigFile) {
        this.inConfigFile = inConfigFile;
    }

    /**
     * This method runs the script and performs batch analysis of the stats
     * relevant for interpreting the hubness of the data on a series on
     * datasets.
     *
     * @throws Exception
     */
    public void runAllTests() throws Exception {
        // First load different data label arrays.
        DataSet labelDataset = null;
        int labelArrayLength;
        if (inLabelFile.getPath().endsWith(".arff")) {
            // Each label array is a column, i.e. corresponds to a feature in
            // the dataset.
            IOARFF aPers = new IOARFF();
            labelDataset = aPers.load(inLabelFile.getPath());
        } else if (inLabelFile.getPath().endsWith(".csv")) {
            IOCSV reader = new IOCSV(false, labelSeparator,
                    DataMineConstants.INTEGER);
            labelDataset = reader.readData(inLabelFile);
        } else {
            System.out.println("Wrong label format.");
            throw new Exception();
        }
        labelArrayLength = labelDataset.getNumIntAttr();
        int dsIndex = 0;
        for (String dsPath : dsPaths) {
            File dsFile = new File(dsPath);
            // Load in the multi-label mode.
            originalDSet = SupervisedLoader.loadData(dsFile, true);
            System.out.println("Testing on: " + dsPath);
            originalDSet.standardizeCategories();
            // Count the categories in the data.
            numCategories = originalDSet.countCategories();
            if (normType != NORM_NO) {
                System.out.print("Normalizing features-");
                if (normType == NORM_01) {
                    // Normalize all float features to the [0, 1] range.
                    originalDSet.normalizeFloats();
                } else if (normType == NORM_STANDARDIZE) {
                    // Standardize all float values.
                    originalDSet.standardizeAllFloats();
                } else if (normType == N_TFIDF) {
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
            // First iterate over different noise levels.
            for (float noise = noiseMin; noise <= noiseMax; noise +=
                    noiseStep) {
                System.gc();
                currDSet = originalDSet.copy();
                // Add noise if a positive noise level was indicated.
                if (noise > 0) {
                    currDSet.addGaussianNoiseToNormalizedCollection(
                            noise, 0.1f);
                }
                for (int lIndex = 0; lIndex < labelArrayLength; lIndex++) {
                    // Assign labels to the data.
                    // Iterate over the mislabeling levels.
                    for (float ml = mlMin; ml <= mlMax; ml += mlStep) {
                        for (int dInd = 0; dInd < currDSet.size(); dInd++) {
                            currDSet.data.get(dInd).setCategory(
                                    labelDataset.data.get(dInd).iAttr[lIndex]);
                        }
                        // Induce mislabeling, if specified.
                        if (ml > 0) {
                            currDSet.induceMislabeling(ml, numCategories);
                        }
                        // Calculate the out directory.
                        currOutDSDir = new File(outDir,
                                dsFile.getName().substring(0,
                                dsFile.getName().lastIndexOf(".")) + "l"
                                + lIndex + File.separator + "k" + kMax
                                + File.separator + "ml" + ml + File.separator
                                + "noise" + noise);
                        FileUtil.createDirectory(currOutDSDir);
                        // Get the appropriate metric.
                        cmet = dsMetric.get(dsIndex);
                        // Calculate class priors.
                        float[] classPriors = currDSet.getClassPriors();
                        // Calculate the k-nearest neighbor sets.
                        NeighborSetFinder nsf = new NeighborSetFinder(
                                currDSet, cmet);
                        // Determine the correct distance matrix path.
                        String dMatPath = null;
                        if (distancesDir != null) {
                            if (!(cmet instanceof SparseCombinedMetric)) {
                                switch (normType) {
                                    case NORM_NO:
                                        dMatPath = "NO" + File.separator
                                                + cmet.getFloatMetric().
                                                getClass().getName() +
                                                File.separator + "dMat.txt";
                                        break;
                                    case NORM_01:
                                        dMatPath = "NORM01" + File.separator
                                                + cmet.getFloatMetric().
                                                getClass().getName() +
                                                File.separator + "dMat.txt";
                                        break;
                                    case NORM_STANDARDIZE:
                                        dMatPath = "STANDARDIZED" +
                                                File.separator + cmet.
                                                getFloatMetric().getClass().
                                                getName() + File.separator
                                                + "dMat.txt";
                                        break;
                                    case N_TFIDF:
                                        dMatPath = "TFIDF" + File.separator
                                                + cmet.getFloatMetric().
                                                getClass().getName() +
                                                File.separator + "dMat.txt";
                                        break;
                                }
                            } else {
                                switch (normType) {
                                    case NORM_NO:
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
                                    case NORM_STANDARDIZE:
                                        dMatPath = "STANDARDIZED" +
                                                File.separator +
                                                ((SparseCombinedMetric) cmet).
                                                getSparseMetric().getClass().
                                                getName() + File.separator
                                                + "dMat.txt";
                                        break;
                                    case N_TFIDF:
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
                        // some warnings and exceptions in the pathological
                        // cases.
                        Class cmetClass = originalDSet.getClass();
                        // Determine the proper metric class and the distance
                        // matrix file.
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
                                // If the file does not exist or the loaded name
                                // is not an appropriate float metric, then
                                // calculate the distances with the specified
                                // metric.
                                System.out.print("Calculating distances-");
                                distMat = currDSet.calculateDistMatrixMultThr(
                                        cmet, 4);
                                System.out.println("-distances calculated.");
                                if (dMatFile != null) {
                                    // If the file path is good, persist the
                                    // newly calculated distance matrix.
                                    DistanceMatrixIO.printDMatToFile(
                                            distMat, dMatFile);
                                }
                            } else {
                                // Load the distances from an existing source.
                                System.out.print("Loading distances-");
                                distMat = DistanceMatrixIO.loadDMatFromFile(
                                        dMatFile);
                                System.out.println("-distance loaded from "
                                        + "file: " + dMatFile.getPath());
                            }
                        }
                        if (secondaryDistanceType == BatchClassifierTester.
                                SecondaryDistance.NONE) {
                            // Use the primary distance matrix for kNN
                            // calculations.
                            nsf.setDistances(distMat);
                        } else {
                            // Use the secondary shared-neighbor distances.
                            if (secondaryDistanceType == BatchClassifierTester.
                                    SecondaryDistance.SIMCOS) {
                                // The simcos secondary distance.
                                NeighborSetFinder nsfSND =
                                        new NeighborSetFinder(currDSet, distMat,
                                        cmet);
                                nsfSND.calculateNeighborSetsMultiThr(
                                        secondaryDistanceK, 8);
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
                                    for (int j = 0; j < simcosDMat[i].length;
                                            j++) {
                                        max = Math.max(max, simcosDMat[i][j]);
                                        min = Math.min(min, simcosDMat[i][j]);
                                    }
                                }
                                for (int i = 0; i < simcosDMat.length; i++) {
                                    for (int j = 0; j < simcosDMat[i].length;
                                            j++) {
                                        simcosDMat[i][j] =
                                                (simcosDMat[i][j] - min)
                                                / (max - min);
                                    }
                                }
                                // Use the simcos distance matrix for kNN set
                                // calculations.
                                nsf = new NeighborSetFinder(originalDSet,
                                        simcosDMat, snc);
                            } else if (secondaryDistanceType
                                    == BatchClassifierTester.
                                    SecondaryDistance.SIMHUB) {
                                // The hubness-aware simhub secondary distance
                                // measure based on shared-neighbor methodology.
                                NeighborSetFinder nsfSND =
                                        new NeighborSetFinder(
                                        currDSet, distMat, cmet);
                                nsfSND.calculateNeighborSetsMultiThr(
                                        secondaryDistanceK, 8);
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
                                    for (int j = 0; j < simhubDMat[i].length;
                                            j++) {
                                        simhubDMat[i][j] = secondaryDistanceK
                                                - simhubSimMat[i][j];
                                    }
                                }
                                // Normalize the scores.
                                float max = 0;
                                float min = Float.MAX_VALUE;
                                for (int i = 0; i < simhubDMat.length; i++) {
                                    for (int j = 0; j < simhubDMat[i].length;
                                            j++) {
                                        max = Math.max(max, simhubDMat[i][j]);
                                        min = Math.min(min, simhubDMat[i][j]);
                                    }
                                }
                                for (int i = 0; i < simhubDMat.length; i++) {
                                    for (int j = 0; j < simhubDMat[i].length;
                                            j++) {
                                        simhubDMat[i][j] =
                                                (simhubDMat[i][j] - min)
                                                / (max - min);
                                    }
                                }
                                nsf = new NeighborSetFinder(
                                        originalDSet, simhubDMat, snc);
                            } else if (secondaryDistanceType
                                    == BatchClassifierTester.
                                    SecondaryDistance.MP) {
                                // Mutual proximity secondary similarity
                                // measure.
                                NeighborSetFinder nsfSecondary =
                                        new NeighborSetFinder(currDSet, distMat,
                                        cmet);
                                nsfSecondary.calculateNeighborSetsMultiThr(
                                        secondaryDistanceK, 8);
                                MutualProximityCalculator calc =
                                        new MutualProximityCalculator(
                                        nsfSecondary.getDistances(),
                                        nsfSecondary.getDataSet(),
                                        nsfSecondary.getCombinedMetric());
                                float[][] mpDistMat =
                                        calc.calculateSecondaryDistMatrixMultThr(
                                        nsfSecondary, 8);
                                // Normalize the scores.
                                float max = 0;
                                float min = Float.MAX_VALUE;
                                for (int i = 0; i < mpDistMat.length; i++) {
                                    for (int j = 0; j < mpDistMat[i].length;
                                            j++) {
                                        max = Math.max(max, mpDistMat[i][j]);
                                        min = Math.min(min, mpDistMat[i][j]);
                                    }
                                }
                                for (int i = 0; i < mpDistMat.length; i++) {
                                    for (int j = 0; j < mpDistMat[i].length;
                                            j++) {
                                        mpDistMat[i][j] =
                                                (mpDistMat[i][j] - min)
                                                / (max - min);
                                    }
                                }
                                nsf = new NeighborSetFinder(
                                        originalDSet, mpDistMat, calc);
                            } else if (secondaryDistanceType
                                    == BatchClassifierTester.
                                    SecondaryDistance.LS) {
                                // Local scaling secondary distance measure.
                                NeighborSetFinder nsfSecondary =
                                        new NeighborSetFinder(currDSet, distMat,
                                        cmet);
                                nsfSecondary.calculateNeighborSetsMultiThr(
                                        secondaryDistanceK, 8);
                                LocalScalingCalculator lsc =
                                        new LocalScalingCalculator(
                                        nsfSecondary);
                                float[][] lsDistMat = lsc.
                                        getTransformedDMatFromNSFPrimaryDMat();
                                // Normalize the scores.
                                float max = 0;
                                float min = Float.MAX_VALUE;
                                for (int i = 0; i < lsDistMat.length; i++) {
                                    for (int j = 0; j < lsDistMat[i].length;
                                            j++) {
                                        max = Math.max(max, lsDistMat[i][j]);
                                        min = Math.min(min, lsDistMat[i][j]);
                                    }
                                }
                                for (int i = 0; i < lsDistMat.length; i++) {
                                    for (int j = 0; j < lsDistMat[i].length;
                                            j++) {
                                        lsDistMat[i][j] =
                                                (lsDistMat[i][j] - min)
                                                / (max - min);
                                    }
                                }
                                nsf = new NeighborSetFinder(
                                        originalDSet, lsDistMat, lsc);
                            } else if (secondaryDistanceType
                                    == BatchClassifierTester.
                                    SecondaryDistance.NICDM) {
                                // NICDM secondary distance measure.
                                NeighborSetFinder nsfSecondary =
                                        new NeighborSetFinder(currDSet, distMat,
                                        cmet);
                                nsfSecondary.calculateNeighborSetsMultiThr(
                                        secondaryDistanceK, 8);
                                NICDMCalculator nicdmCalc =
                                        new NICDMCalculator(nsfSecondary);
                                float[][] lsDistMat = nicdmCalc.
                                        getTransformedDMatFromNSFPrimaryDMat();
                                // Normalize the scores.
                                float max = 0;
                                float min = Float.MAX_VALUE;
                                for (int i = 0; i < lsDistMat.length; i++) {
                                    for (int j = 0; j < lsDistMat[i].length;
                                            j++) {
                                        max = Math.max(max, lsDistMat[i][j]);
                                        min = Math.min(min, lsDistMat[i][j]);
                                    }
                                }
                                for (int i = 0; i < lsDistMat.length; i++) {
                                    for (int j = 0; j < lsDistMat[i].length;
                                            j++) {
                                        lsDistMat[i][j] =
                                                (lsDistMat[i][j] - min)
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
                        TopHubsClusterUtil thcu = new TopHubsClusterUtil(nsf);
                        KNeighborEntropyExplorer knee =
                                new KNeighborEntropyExplorer(nsf,
                                numCategories);

                        // Use the hubness stats calculators to obtain the stats
                        // for interpreting the hubness of the data.
                        // The percentages of points that occur at least once.
                        float[] aboveZeroArray =
                                hte.getThresholdPercentageArray();
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
                        float[] kRNNEntropiesMeans =
                                knee.getReverseEntropyMeans();
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
                        float[] bhArray = nsf.getLabelMismatchPercsAllK();
                        float[][][] gClasstoClassHubness = new float[kMax][][];
                        for (int kVal = 1; kVal <= kMax; kVal++) {
                            gClasstoClassHubness[kVal - 1] = nsf.
                                    getGlobalClassToClassForKforFuzzy(
                                    kVal, numCategories, 0.01f, true);
                        }

                        File currOutFile = new File(currOutDSDir,
                                "hubnessOverview.txt");
                        // Print out the results.
                        try (PrintWriter pw =
                                new PrintWriter(new FileWriter(currOutFile));) {
                            pw.println("dataset: " + dsFile.getName());
                            pw.println("k_max: " + kMax);
                            pw.println("noise: " + noise);
                            pw.println("ml: " + ml);
                            pw.println("instances: " + originalDSet.size());
                            pw.println("numCat: " + numCategories);
                            pw.print("class priors: ");
                            for (int cIndex = 0; cIndex < numCategories;
                                    cIndex++) {
                                pw.print(BasicMathUtil.makeADecimalCutOff(
                                        classPriors[cIndex], 3));
                                pw.print(" ");
                            }
                            pw.println();
                            try {
                                pw.println("dim: "
                                        + originalDSet.fAttrNames.length);
                            } catch (Exception e) {
                                System.err.println(e.getMessage());
                            }
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
                            for (int i = 1; i < stDevArray.length; i++) {
                                pw.print(","
                                        + BasicMathUtil.makeADecimalCutOff(
                                        kEntropiesMeans[i], 3));
                            }
                            pw.println();
                            pw.println("-------------------------------------");
                            pw.println("kEntropyStDevs: ");
                            pw.print(BasicMathUtil.makeADecimalCutOff(
                                    kEntropiesStDevs[0], 3));
                            for (int i = 1; i < stDevArray.length; i++) {
                                pw.print(","
                                        + BasicMathUtil.makeADecimalCutOff(
                                        kEntropiesStDevs[i], 3));
                            }
                            pw.println();
                            pw.println("-------------------------------------");
                            pw.println("kEntropySkews: ");
                            pw.print(BasicMathUtil.makeADecimalCutOff(
                                    kEntropiesSkews[0], 3));
                            for (int i = 1; i < stDevArray.length; i++) {
                                pw.print(","
                                        + BasicMathUtil.makeADecimalCutOff(
                                        kEntropiesSkews[i], 3));
                            }
                            pw.println();
                            pw.println("-------------------------------------");
                            pw.println("kEntropyKurtosis: ");
                            pw.print(BasicMathUtil.makeADecimalCutOff(
                                    kEntropiesKurtosis[0], 3));
                            for (int i = 1; i < stDevArray.length; i++) {
                                pw.print(","
                                        + BasicMathUtil.makeADecimalCutOff(
                                        kEntropiesKurtosis[i], 3));
                            }
                            pw.println();
                            pw.println("-------------------------------------");
                            pw.println("kHubnessEntropyMeans: ");
                            pw.print(BasicMathUtil.makeADecimalCutOff(
                                    kRNNEntropiesMeans[0], 3));
                            for (int i = 1; i < stDevArray.length; i++) {
                                pw.print(","
                                        + BasicMathUtil.makeADecimalCutOff(
                                        kRNNEntropiesMeans[i], 3));
                            }
                            pw.println();
                            pw.println("-------------------------------------");
                            pw.println("kHubnessEntropyStDevs: ");
                            pw.print(BasicMathUtil.makeADecimalCutOff(
                                    kRNNEntropiesStDevs[0], 3));
                            for (int i = 1; i < stDevArray.length; i++) {
                                pw.print(","
                                        + BasicMathUtil.makeADecimalCutOff(
                                        kRNNEntropiesStDevs[i], 3));
                            }
                            pw.println();
                            pw.println("-------------------------------------");
                            pw.println("kHubnessEntropySkews: ");
                            pw.print(BasicMathUtil.makeADecimalCutOff(
                                    kRNNEntropiesSkews[0], 3));
                            for (int i = 1; i < stDevArray.length; i++) {
                                pw.print(","
                                        + BasicMathUtil.makeADecimalCutOff(
                                        kRNNEntropiesSkews[i], 3));
                            }
                            pw.println();
                            pw.println("-------------------------------------");
                            pw.println("kHubnessEntropyKurtosis: ");
                            pw.print(BasicMathUtil.makeADecimalCutOff(
                                    kRNNEntropiesKurtosis[0], 3));
                            for (int i = 1; i < stDevArray.length; i++) {
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
                            hte = new HubnessAboveThresholdExplorer(2, true,
                                    nsf);
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
                            hte = new HubnessAboveThresholdExplorer(3, true,
                                    nsf);
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
                            hte = new HubnessAboveThresholdExplorer(4, true,
                                    nsf);
                            aboveZeroArray = hte.getThresholdPercentageArray();
                            pw.println("-------------------------------------");
                            pw.println("Hubness above three percentage "
                                    + "Array: ");
                            pw.print(BasicMathUtil.makeADecimalCutOff(
                                    aboveZeroArray[0], 2));
                            for (int i = 1; i < aboveZeroArray.length; i++) {
                                pw.print("," + BasicMathUtil.makeADecimalCutOff(
                                        aboveZeroArray[i], 2));
                            }
                            pw.println();
                            hte = new HubnessAboveThresholdExplorer(5, true,
                                    nsf);
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
                            for (int i = 1; i < topHubClustDiamsArr.length;
                                    i++) {
                                pw.print("," + BasicMathUtil.makeADecimalCutOff(
                                        topHubClustDiamsArr[i], 2));
                            }
                            pw.println();
                            pw.println("-------------------------------------");
                            pw.println("Top ten hubs avg within cluster "
                                    + "dist: ");
                            pw.print(BasicMathUtil.makeADecimalCutOff(
                                    topHubClustAvgDistArr[0], 2));
                            for (int i = 1; i < topHubClustAvgDistArr.length;
                                    i++) {
                                pw.print("," + BasicMathUtil.makeADecimalCutOff(
                                        topHubClustAvgDistArr[i], 2));
                            }
                            pw.println();
                            pw.println("-------------------------------------");
                            thcu.calcTopHubnessDiamAndAvgDist(5);
                            topHubClustDiamsArr =
                                    thcu.getTopHubClusterDiameters();
                            topHubClustAvgDistArr =
                                    thcu.getTopHubClusterAvgDists();
                            pw.println("Top five hubs diam: ");
                            pw.print(BasicMathUtil.makeADecimalCutOff(
                                    topHubClustDiamsArr[0], 2));
                            for (int i = 1; i < topHubClustDiamsArr.length;
                                    i++) {
                                pw.print("," + BasicMathUtil.makeADecimalCutOff(
                                        topHubClustDiamsArr[i], 2));
                            }
                            pw.println();
                            pw.println("-------------------------------------");
                            pw.println("Top five hubs avg within cluster "
                                    + "dist: ");
                            pw.print(BasicMathUtil.makeADecimalCutOff(
                                    topHubClustAvgDistArr[0], 2));
                            for (int i = 1; i < topHubClustAvgDistArr.length;
                                    i++) {
                                pw.print("," + BasicMathUtil.makeADecimalCutOff(
                                        topHubClustAvgDistArr[i], 2));
                            }
                            pw.println();
                            pw.println("-------------------------------------");
                            pw.println("highest occurrence frequencies (each"
                                    + " line is for one k value, lines go from"
                                    + " zero to k_max): ");
                            for (int k = 0; k < kMax; k++) {
                                pw.print("k: " + (k + 1) + ":: ");
                                pw.print(BasicMathUtil.makeADecimalCutOff(
                                        highestOccFreqs[k][0], 3));
                                for (int i = 1; i < 20; i++) {
                                    pw.print(","
                                            + BasicMathUtil.makeADecimalCutOff(
                                            highestOccFreqs[k][i], 3));
                                }
                                pw.println();
                            }
                            pw.println("-------------------------------------");
                            pw.println("global class to class hubness matrices"
                                    + " for all K-s: ");
                            for (int k = 1; k <= kMax; k++) {
                                pw.println("k = " + k);
                                for (int c1 = 0; c1 < numCategories; c1++) {
                                    for (int c2 = 0; c2 < numCategories; c2++) {
                                        pw.print(BasicMathUtil.
                                                makeADecimalCutOff(
                                                gClasstoClassHubness[k - 1][c1][
                                                c2], 3));
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
        try (BufferedReader br = new BufferedReader(new InputStreamReader(
                new FileInputStream(inConfigFile)));) {
            String s = br.readLine();
            String[] lineParse;
            // Integer and float metrics.
            Class currIntMet;
            Class currFloatMet;
            // Read the file line by line.
            while (s != null) {
                s = s.trim();
                if (s.startsWith("@in_directory")) {
                    // Input directory.
                    lineParse = s.split(" ");
                    inDir = new File(lineParse[1]);
                } else if (s.startsWith("@out_directory")) {
                    // Output directory.
                    lineParse = s.split(" ");
                    outDir = new File(lineParse[1]);
                } else if (s.startsWith("@k_max")) {
                    // Maximal k-value to which to iterate.
                    lineParse = s.split(" ");
                    kMax = Integer.parseInt(lineParse[1]);
                } else if (s.startsWith("@noise_range")) {
                    // Noise range: min, max, increment.
                    lineParse = s.split(" ");
                    noiseMin = Float.parseFloat(lineParse[1]);
                    noiseMax = Float.parseFloat(lineParse[2]);
                    noiseStep = Float.parseFloat(lineParse[3]);
                } else if (s.startsWith("@mislabeled_range")) {
                    // Mislabeling range: min, max, increment.
                    lineParse = s.split(" ");
                    mlMin = Float.parseFloat(lineParse[1]);
                    mlMax = Float.parseFloat(lineParse[2]);
                    mlStep = Float.parseFloat(lineParse[3]);
                } else if (s.startsWith("@normalization")) {
                    // Normalization specification.
                    lineParse = s.split("\\s+");
                    if (lineParse[1].toLowerCase().compareTo("no") == 0) {
                        normType = NORM_NO;
                    } else if (lineParse[1].toLowerCase().compareTo(
                            "normalizeTo01".toLowerCase()) == 0) {
                        normType = NORM_01;
                    } else if (lineParse[1].toLowerCase().compareTo(
                            "TFIDF".toLowerCase()) == 0) {
                        normType = N_TFIDF;
                    } else {
                        normType = NORM_STANDARDIZE;
                    }
                } else if (s.startsWith("@label_file")) {
                    // Path to the file containing the labels.
                    lineParse = s.split(" ");
                    inLabelFile = new File(lineParse[1]);
                    if (lineParse.length >= 3) {
                        String separatorBuffer = lineParse[2];
                        for (int hrm = 3; hrm < lineParse.length; hrm++) {
                            separatorBuffer += " ";
                            separatorBuffer += lineParse[hrm];
                        }
                        labelSeparator = separatorBuffer.substring(1,
                                separatorBuffer.length() - 1);
                        System.out.println("label separator \""
                                + labelSeparator + "\"");
                    }
                } else if (s.startsWith("@dataset")) {
                    // Dataset specification.
                    lineParse = s.split(" ");
                    dsPaths.add(lineParse[1]);
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
                            currIntMet = Class.forName(lineParse[3]);
                            cmetLoaded.setIntegerMetric((DistanceMeasure)
                                    (currIntMet.newInstance()));
                        }
                        if (!lineParse[3].equals("null")) {
                            currFloatMet = Class.forName(lineParse[4]);
                            cmetLoaded.setFloatMetric((DistanceMeasure)
                                    (currFloatMet.newInstance()));
                        }
                        cmetLoaded.setCombinationMethod(CombinedMetric.DEFAULT);
                        dsMetric.add(cmetLoaded);
                    }
                } else if (s.startsWith("@distances_directory")) {
                    // Directory for loading and/or persisting the distance
                    // matrices.
                    lineParse = s.split("\\s+");
                    distancesDir = new File(lineParse[1]);
                } else if (s.startsWith("@secondary_distance")) {
                    // Secondary distance specification.
                    lineParse = s.split("\\s+");
                    switch (lineParse[1].toLowerCase()) {
                        case "simcos": {
                            secondaryDistanceType =BatchClassifierTester.
                                    SecondaryDistance.SIMCOS;
                            break;
                        }
                        case "simhub": {
                            secondaryDistanceType = BatchClassifierTester.
                                    SecondaryDistance.SIMHUB;
                            break;
                        }
                        case "mp": {
                            secondaryDistanceType = BatchClassifierTester.
                                    SecondaryDistance.MP;
                            break;
                        }
                        case "ls": {
                            secondaryDistanceType = BatchClassifierTester.
                                    SecondaryDistance.LS;
                            break;
                        }
                        case "nicdm": {
                            secondaryDistanceType = BatchClassifierTester.
                                    SecondaryDistance.NICDM;
                            break;
                        }
                        default: {
                            secondaryDistanceType = BatchClassifierTester.
                                    SecondaryDistance.SIMCOS;
                            break;
                        }
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
                    dsPaths.set(i, "sparse:" + (new File(inDir, dsPaths.get(i).
                            substring(dsPaths.get(i).indexOf(":") + 1,
                            dsPaths.get(i).length()))).getPath());
                }
            }
        } catch (Exception e) {
            throw e;
        }
    }

    /**
     * This method runs the batch multi-label hubness analysis script.
     *
     * @param args Command line parameters, as specified.
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            System.out.println("1 parameter - file with test configuration");
            return;
        }
        File inConfigFile = new File(args[0]);
        MultiLabelBatchHubnessAnalyzer tester =
                new MultiLabelBatchHubnessAnalyzer(inConfigFile);
        tester.loadParameters();
        tester.runAllTests();
    }
}
