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

import combinatorial.Permutation;
import data.neighbors.NSFUserInterface;
import data.neighbors.NeighborSetFinder;
import data.neighbors.SharedNeighborFinder;
import data.neighbors.approximate.AppKNNGraphLanczosBisection;
import data.representation.DataInstance;
import data.representation.DataSet;
import data.representation.discrete.DiscretizedDataInstance;
import data.representation.discrete.DiscretizedDataSet;
import data.representation.sparse.BOWDataSet;
import data.representation.sparse.BOWInstance;
import data.representation.util.DataMineConstants;
import distances.primary.CombinedMetric;
import distances.secondary.LocalScalingCalculator;
import distances.secondary.MutualProximityCalculator;
import distances.secondary.NICDMCalculator;
import distances.secondary.snd.SharedNeighborCalculator;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import learning.supervised.Category;
import learning.supervised.DiscreteCategory;
import learning.supervised.evaluation.ClassificationEstimator;
import learning.supervised.evaluation.ValidateableInterface;
import learning.supervised.evaluation.cv.BatchClassifierTester.SecondaryDistance;
import learning.supervised.interfaces.AutomaticKFinderInterface;
import learning.supervised.interfaces.DiscreteDistToPointsQueryUserInterface;
import learning.supervised.interfaces.DiscreteNeighborPointsQueryUserInterface;
import learning.supervised.interfaces.DistMatrixUserInterface;
import learning.supervised.interfaces.DistToPointsQueryUserInterface;
import learning.supervised.interfaces.NeighborPointsQueryUserInterface;
import preprocessing.instance_selection.InstanceSelector;

/**
 * This class implements the functionality necessary to conduct a
 * cross-validation procedure for evaluating a set of classifiers on a dataset.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class MultiCrossValidation {

    // Instance selection specification, in case the instance selection is to be
    // applied during the folds.
    private InstanceSelector dreducer = null;
    private InstanceSelector foldReducer;
    private float selectionRate = 0;
    public final static int PROTO_UNBIASED = 0;
    public final static int PROTO_BIASED = 1;
    private int protoHubnessMode = PROTO_UNBIASED;
    // The number of times and folds to use.
    private int times = 10;
    private int numFolds = 10;
    // The data context.
    private Object dataType;
    // The data instance list.
    private ArrayList data;
    // Number of classes in the data.
    private int numClasses = 2;
    // Execution times.
    public double[] execTimeTotal;
    public double execTimeAllOneRun;
    // Classifier prototypes.
    private ValidateableInterface[] classifiers;
    // Currently active learners (under training or testing).
    private ValidateableInterface[] currClassifierInstances = null;
    // Evaluation structures.
    private boolean keepAllEvaluations = true;
    private ClassificationEstimator[][] estimators = null;
    private ClassificationEstimator[] currEstimator = null;
    private ClassificationEstimator[] averageEstimator = null;
    private float[][] correctPointClassificationArray = null;
    // The neighborhood sizes, in case of automatic k calculation within the
    // algorithms.
    private int kMin = 1;
    private int kMax = 20;
    // The neighborhood size, in case a pre-defined value is to be used.
    private int kValue = 5;
    // Whether to use the interval best k search or the fixed k-value mode.
    private int kMode = SINGLE;
    public static final int SINGLE = 0;
    public static final int INTERVAL = 1;
    // Secondary distance type and the secondary neighborhood size.
    private SecondaryDistance secondaryDistanceType = SecondaryDistance.NONE;
    private int secondaryK = 50;
    // The object that does all the distance calculations.
    private CombinedMetric cmet;
    // In case some fold evaluations return errors and are discarded, these
    // numbers are used for averaging the estimators for each algorithm.
    private int[] numFullFolds;
    // These structures are used for the training/test splits and folds in the
    // cross-validation procedure.
    private ArrayList<Integer>[][] allFolds = null;
    private boolean foldsLoaded = false;
    private ArrayList[] dataFolds = null;
    private ArrayList currentTraining;
    private ArrayList<Integer>[] foldIndexes = null;
    private ArrayList<Integer> currentTrainingIndexes;
    private ArrayList<Integer> currentTestIndexes;
    // In case of instance selection.
    private ArrayList<Integer> currentPrototypeIndexes;
    // A separate label array is kept to be used in case of mislabeling data
    // experiments.
    private boolean validateOnExternalLabels = false;
    private int[] testLabelArray = null;
    // The total distance matrix, as an upper triangular matrix.
    public float[][] totalDistMat = null;
    // Flags indicating whether there are users of the distance matrix or the
    // total kNN sets on the training data.
    private boolean distUserPresent = false;
    private boolean nsfUserPresent = false;
    // The kNN structures.
    private NeighborSetFinder bigNSF;
    private NeighborSetFinder nsfCurrent = null;
    private int[][] testPointNeighbors = null;
    private boolean approximateNNs = false;
    private float alphaAppKNN = 1f;
    // The number of threads used for distance matrix and kNN set calculations.
    private int numCommonThreads = 8;

    /**
     * The default constructor.
     */
    public MultiCrossValidation() {
    }

    /**
     * Initialization.
     *
     * @param times Integer that is the number of times to repeat the fold split
     * and evaluation process.
     * @param folds Integer that is the number of folds to generate in each
     * repetition.
     * @param numClasses Integer that is the number of classes in the data.
     */
    public MultiCrossValidation(int times, int folds, int numClasses) {
        this.times = times;
        this.numFolds = folds;
        this.numClasses = numClasses;
    }

    /**
     * Initialization.
     *
     * @param times Integer that is the number of times to repeat the fold split
     * and evaluation process.
     * @param folds Integer that is the number of folds to generate in each
     * repetition.
     * @param numClasses Integer that is the number of classes in the data.
     * @param totalDistMat float[][] that is the upper triangular distance
     * matrix on the training data.
     */
    public MultiCrossValidation(int times, int folds, int numClasses,
            float[][] totalDistMat) {
        this.times = times;
        this.numFolds = folds;
        this.numClasses = numClasses;
        this.totalDistMat = totalDistMat;
    }

    /**
     * Initialization with a single classifier.
     *
     * @param times Integer that is the number of times to repeat the fold split
     * and evaluation process.
     * @param folds Integer that is the number of folds to generate in each
     * repetition.
     * @param numClasses Integer that is the number of classes in the data.
     * @param dataType Object that is the data context.
     * @param data ArrayList of data instances.
     * @param classifier ValidateableInterface signifying the classifier to
     * evaluate.
     */
    public MultiCrossValidation(int times, int folds, int numClasses,
            Object dataType, ArrayList data, ValidateableInterface classifier) {
        this.times = times;
        this.numFolds = folds;
        this.dataType = dataType;
        this.data = data;
        this.classifiers = new ValidateableInterface[1];
        classifiers[0] = classifier;
        currClassifierInstances = new ValidateableInterface[1];
        execTimeTotal = new double[1];
        this.numClasses = numClasses;
    }

    /**
     * Initialization with a single classifier..
     *
     * @param times Integer that is the number of times to repeat the fold split
     * and evaluation process.
     * @param folds Integer that is the number of folds to generate in each
     * repetition.
     * @param numClasses Integer that is the number of classes in the data.
     * @param dataType Object that is the data context.
     * @param data ArrayList of data instances.
     * @param classifier ValidateableInterface signifying the classifier to
     * evaluate.
     * @param totalDistMat
     */
    public MultiCrossValidation(int times, int folds, int numClasses,
            Object dataType, ArrayList data, ValidateableInterface classifier,
            float[][] totalDistMat) {
        this.times = times;
        this.numFolds = folds;
        this.dataType = dataType;
        this.data = data;
        this.classifiers = new ValidateableInterface[1];
        classifiers[0] = classifier;
        currClassifierInstances = new ValidateableInterface[1];
        execTimeTotal = new double[1];
        this.numClasses = numClasses;
        this.totalDistMat = totalDistMat;
    }

    /**
     * Initialization.
     *
     * @param times Integer that is the number of times to repeat the fold split
     * and evaluation process.
     * @param folds Integer that is the number of folds to generate in each
     * repetition.
     * @param numClasses Integer that is the number of classes in the data.
     * @param dataType Object that is the data context.
     * @param data ArrayList of data instances.
     * @param classifiers ValidateableInterface[] classifiers signifying an
     * array of classifiers to evaluate.
     */
    public MultiCrossValidation(int times, int folds, int numClasses,
            Object dataType, ArrayList data,
            ValidateableInterface[] classifiers) {
        this.times = times;
        this.numFolds = folds;
        this.dataType = dataType;
        this.data = data;
        this.classifiers = classifiers;
        currClassifierInstances = new ValidateableInterface[classifiers.length];
        execTimeTotal = new double[classifiers.length];
        this.numClasses = numClasses;
    }

    /**
     * Initialization.
     *
     * @param times Integer that is the number of times to repeat the fold split
     * and evaluation process.
     * @param folds Integer that is the number of folds to generate in each
     * repetition.
     * @param numClasses Integer that is the number of classes in the data.
     * @param dataType Object that is the data context.
     * @param data ArrayList of data instances.
     * @param classifiers ValidateableInterface[] classifiers signifying an
     * array of classifiers to evaluate.
     * @param totalDistMat float[][] that is the upper triangular distance
     * matrix on the training data.
     */
    public MultiCrossValidation(int times, int folds, int numClasses,
            Object dataType, ArrayList data,
            ValidateableInterface[] classifiers, float[][] totalDistMat) {
        this.times = times;
        this.numFolds = folds;
        this.dataType = dataType;
        this.data = data;
        this.classifiers = classifiers;
        currClassifierInstances = new ValidateableInterface[classifiers.length];
        execTimeTotal = new double[classifiers.length];
        this.numClasses = numClasses;
        this.totalDistMat = totalDistMat;
    }
    
    /**
     * Sets the pre-computed folds to the cross-validation procedure. An
     * exception is thrown if the dimensions of the array of lists of indexes do
     * not correspond to the number of times and folds that have been set in
     * this cross-validation context.
     * 
     * @param allFolds ArrayList<Integer>[][] representing all folds in all
     * repetitions of the cross-validation procedure, by giving indexes of the
     * points within the dataset in each split.
     * @throws Exception if the fold array dimensionality does not correspond
     * to the times x numFolds that are set within this cross-validation
     * context.
     */
    public void setAllFolds(ArrayList<Integer>[][] allFolds) throws Exception {
        this.allFolds = allFolds;
        if (allFolds != null) {
            if (allFolds.length != times) {
                throw new Exception("Bad fold format, not matching the "
                        + "specified number of repetitions.");
            } else {
                for (int repIndex = 0; repIndex < times; repIndex++) {
                    if (allFolds[repIndex] == null ||
                            allFolds[repIndex].length != numFolds) {
                        throw new Exception("Bad fold format. The number of"
                                + "folds in some repetitions does not match"
                                + "the specified number of folds in this"
                                + "cross-validation context.");
                    }
                }
            }
            foldsLoaded = true;
        }
    }
    
    /**
     * @return ArrayList<Integer>[][] representing all folds in all repetitions 
     * of the cross-validation procedure, by giving indexes of the points within
     * the dataset in each split.
     */
    public ArrayList<Integer>[][] getAllFolds() {
        return allFolds;
    }
    
    /**
     * This sets the number of threads to use for distance matrix and kNN set
     * calculations.
     * 
     * @param numCommonThreads Integer that is the number of threads to use for
     * distance matrix and kNN set calculations.
     */
    public void useMultipleCommonThreads(int numCommonThreads) {
        this.numCommonThreads = numCommonThreads;
    }

    /**
     * @param protoHubnessMode Integer code that indicates which prototype
     * hubness estimation mode to use, whether to use the biased simple approach
     * or to estimate the prototype occurrence frequencies on the rejected
     * points as well.
     */
    public void setProtoHubnessMode(int protoHubnessMode) {
        this.protoHubnessMode = protoHubnessMode;
    }

    /**
     * Sets the instance selection structures.
     * @param dreducer InstanceSelector that is to be used for data reduction.
     * @param selectionRate Float that is the instance selection rate.
     */
    public void setDataReducer(InstanceSelector dreducer, float selectionRate) {
        this.dreducer = dreducer;
        this.selectionRate = selectionRate;
    }

    /**
     * @return InstanceSelector that is being used for data reduction.
     */
    public InstanceSelector getDataReducer() {
        return dreducer;
    }

    /**
     * @return float[][] representing the point classification precision for
     * each tested algorithm.
     */
    public float[][] getPerPointClassificationPrecision() {
        return correctPointClassificationArray;
    }

    /**
     * @param secondaryDistanceType SecondaryDistance that is to be used.
     * @param secondaryK Integer that is the neighborhood size to use when
     * calculating the secondary distances.
     */
    public void useSecondaryDistances(SecondaryDistance secondaryDistanceType,
            int secondaryK) {
        this.secondaryDistanceType = secondaryDistanceType;
        this.secondaryK = secondaryK;
    }

    /**
     * @param cmet CombinedMetric object for distance calculations.
     */
    public void setCombinedMetric(CombinedMetric cmet) {
        this.cmet = cmet;
    }

    /**
     * @param k Integer that is the neighborhood size to test for. 
     */
    public void setKValue(int k) {
        this.kValue = k;
        this.kMin = k;
        this.kMax = k;
        kMode = SINGLE;
    }

    /**
     * In case the algorithms are expected to automatically determine the
     * optimal k-value.
     * @param kMin Integer that is the lower range value for the neighborhood
     * size.
     * @param kMax Integer that is the upper range value for the neighborhood
     * size.
     */
    public void setAutoKRange(int kMin, int kMax) {
        this.kMin = kMin;
        this.kMax = kMax;
        kMode = INTERVAL;
    }

    /**
     * @param approximate Boolean flag indicating whether to use the approximate
     * kNN sets.
     * @param alpha Float that is the quality parameter for the Lanczos
     * bisection approximate kNN method.
     */
    public void setApproximateKNNParams(boolean approximate, float alpha) {
        this.approximateNNs = approximate;
        this.alphaAppKNN = alpha;
    }

    /**
     * @param testLabels int[] representing the labels of the training points
     * to evaluate the classification performance against.
     */
    public void validateOnSeparateLabelArray(int[] testLabels) {
        validateOnExternalLabels = true;
        testLabelArray = testLabels;
    }

    /**
     * This method obtains a NeighborSetFinder object for the current fold split
     * from the big NSF object, based on subsampling of the larger neighbor sets
     * according to the provided data indexes and/or some additional
     * calculations.
     * 
     * @param trainFoldIndexes ArrayList<Integer> representing the training data
     * indexes.
     * @param trainDSet DataSet that is the training data context.
     * @param foldDistMat float[][] representing the fold distance matrix.
     * @param pointDistances float[][] representing the distances between test
     * and training points.
     * @param k Integer that is the current neighborhood size.
     * @return NeighborSetFinder object for the current training/test split.
     */
    private NeighborSetFinder obtainFoldNSF(ArrayList<Integer> trainFoldIndexes,
            DataSet trainDSet, float[][] foldDistMat, int k) {
        NeighborSetFinder nsf = new NeighborSetFinder(
                trainDSet, foldDistMat, cmet);
        // Initialize a HashMap indicating which indexes belong to the training
        // data.
        HashMap<Integer, Integer> trainHash = new HashMap<>(
                4 * trainFoldIndexes.size(), 1000);
        for (int i = 0; i < trainFoldIndexes.size(); i++) {
            trainHash.put(trainFoldIndexes.get(i), i);
        }
        // Initialize the neighbor sets and the k-distance arrays.
        int[][] kneighbors = new int[trainFoldIndexes.size()][k];
        float[][] kDistances = new float[trainFoldIndexes.size()][k];
        // Obtain the kNN sets and the k-distance arrays of the big NSF object.
        int[][] knBig = bigNSF.getKNeighbors();
        float[][] kDistancesBig = bigNSF.getKDistances();
        // Auxiliary variables.
        int[] currNSet;
        int kIndex;
        int index;
        int lowerIntervalBound, upperIntervalBound, minIndex, maxIndex, l;
        int datasize = trainDSet.size();
        ArrayList<Integer> intervals;
        for (int i = 0; i < trainFoldIndexes.size(); i++) {
            currNSet = knBig[trainFoldIndexes.get(i)];
            kIndex = 0;
            index = 0;
            // First re-use as many neighbors as are present in the training
            // split, from the big NSF object.
            while (kIndex < k && index < knBig[0].length) {
                if (trainHash.containsKey(currNSet[index])) {
                    kneighbors[i][kIndex] = trainHash.get(currNSet[index]);
                    kDistances[i][kIndex] =
                            kDistancesBig[trainFoldIndexes.get(i)][index];
                    kIndex++;
                }
                index++;
            }
            // Perform additional calculations.
            if (kIndex < k) {
                intervals = new ArrayList<>(kIndex + 2);
                intervals.add(-1);
                for (int j = 0; j < kIndex; j++) {
                    intervals.add(kneighbors[i][j]);
                }
                intervals.add(datasize + 1);
                Collections.sort(intervals);
                int iSizeRed = intervals.size() - 1;
                for (int ind = 0; ind < iSizeRed; ind++) {
                    lowerIntervalBound = intervals.get(ind);
                    upperIntervalBound = intervals.get(ind + 1);
                    for (int j = lowerIntervalBound + 1;
                            j < upperIntervalBound - 1; j++) {
                        if (i != j) {
                            minIndex = Math.min(i, j);
                            maxIndex = Math.max(i, j);
                            if (kIndex > 0) {
                                if (kIndex == k) {
                                    if (foldDistMat[minIndex][
                                            maxIndex - minIndex - 1] <
                                            kDistances[i][kIndex - 1]) {
                                        // Search and insert.
                                        l = k - 1;
                                        while ((l >= 1) && foldDistMat[
                                                minIndex][maxIndex - minIndex -
                                                1] < kDistances[i][l - 1]) {
                                            kDistances[i][l] = kDistances[i][
                                                    l - 1];
                                            kneighbors[i][l] = kneighbors[i][
                                                    l - 1];
                                            l--;
                                        }
                                        kDistances[i][l] = foldDistMat[
                                                minIndex][maxIndex -
                                                minIndex - 1];
                                        kneighbors[i][l] = j;
                                    }
                                } else {
                                    if (foldDistMat[minIndex][
                                            maxIndex - minIndex - 1] <
                                            kDistances[i][kIndex - 1]) {
                                        // Search and insert.
                                        l = kIndex - 1;
                                        kDistances[i][kIndex] = kDistances[i][
                                                kIndex - 1];
                                        kneighbors[i][kIndex] = kneighbors[i][
                                                kIndex - 1];
                                        while ((l >= 1) && foldDistMat[
                                                minIndex][maxIndex - minIndex -
                                                1] < kDistances[i][l - 1]) {
                                            kDistances[i][l] = kDistances[i][
                                                    l - 1];
                                            kneighbors[i][l] = kneighbors[i][
                                                    l - 1];
                                            l--;
                                        }
                                        kDistances[i][l] = foldDistMat[
                                                minIndex][maxIndex - minIndex
                                                - 1];
                                        kneighbors[i][l] = j;
                                        kIndex++;
                                    } else {
                                        kDistances[i][kIndex] = foldDistMat[
                                                minIndex][maxIndex - minIndex
                                                - 1];
                                        kneighbors[i][kIndex] = j;
                                        kIndex++;
                                    }
                                }
                            } else {
                                kDistances[i][0] = foldDistMat[minIndex][
                                        maxIndex - minIndex - 1];
                                kneighbors[i][0] = j;
                                kIndex = 1;
                            }
                        }
                    }
                }
            }
        }
        int[] kCurrLen = new int[trainFoldIndexes.size()];
        Arrays.fill(kCurrLen, k);
        nsf.setKNeighbors(kneighbors, kDistances, kCurrLen);
        return nsf;
    }

    /**
     * This method obtains the kNN sets of test points containing the training
     * neighbor points.
     * 
     * @param trainFoldIndexes ArrayList<Integer> representing the training data
     * indexes.
     * @param testFoldIndexes ArrayList<Integer> representing the test data
     * indexes.
     * @param trainDSet DataSet that is the training data context.
     * @param pointDistances float[][] representing the distances between test
     * and training points.
     * @param k Integer that is the current neighborhood size.
     * @return int[][] that are the desired kNN sets.
     */
    public int[][] obtainFoldTestNeighbors(ArrayList<Integer> trainFoldIndexes,
            ArrayList<Integer> testFoldIndexes, DataSet trainDSet,
            float[][] pointDistances, int k) {
        // Initialize a HashMap indicating which indexes belong to the training
        // data.
        HashMap<Integer, Integer> trainHash =
                new HashMap<>(4 * trainFoldIndexes.size(), 1000);
        for (int i = 0; i < trainFoldIndexes.size(); i++) {
            trainHash.put(trainFoldIndexes.get(i), i);
        }
        // Initialize the neighbor sets and the k-distance arrays.
        int[][] testNeighbors = new int[testFoldIndexes.size()][k];
        float[][] kDistances = new float[testFoldIndexes.size()][k];
        // Obtain the kNN sets and the k-distance arrays of the big NSF object.
        int[][] kNeighborsAllBigK = bigNSF.getKNeighbors();
        float[][] kDistancesBig = bigNSF.getKDistances();

        int[] currLargerNSet;
        int kIndex;
        int index;
        int lowerIntervalBound, upperIntervalBound, l;
        int datasize = trainDSet.size();
        for (int i = 0; i < testFoldIndexes.size(); i++) {
            currLargerNSet = kNeighborsAllBigK[testFoldIndexes.get(i)];
            kIndex = 0;
            index = 0;
            // First re-use the existing neighbor information.
            while (kIndex < k && index < kNeighborsAllBigK[0].length) {
                if (trainHash.containsKey(currLargerNSet[index])) {
                    testNeighbors[i][kIndex] = trainHash.get(
                            currLargerNSet[index]);
                    kDistances[i][kIndex] = kDistancesBig[
                            testFoldIndexes.get(i)][index];
                    kIndex++;
                }
                index++;
            }
            // Additional calculations, if necessary (very rarely, in this
            // implementation).
            if (kIndex < k) {
                ArrayList<Integer> intervals = new ArrayList<>(kIndex + 2);
                intervals.add(-1);
                for (int j = 0; j < kIndex; j++) {
                    intervals.add(testNeighbors[i][j]);
                }
                intervals.add(datasize + 1);
                Collections.sort(intervals);
                int iSizeRed = intervals.size() - 1;
                for (int ind = 0; ind < iSizeRed; ind++) {
                    lowerIntervalBound = intervals.get(ind);
                    upperIntervalBound = intervals.get(ind + 1);
                    for (int j = lowerIntervalBound + 1;
                            j < upperIntervalBound - 1; j++) {
                        if (i != j) {
                            if (kIndex > 0) {
                                if (kIndex == k) {
                                    if (pointDistances[i][j] <
                                            kDistances[i][kIndex - 1]) {
                                        // Search and insert.
                                        l = k - 1;
                                        while ((l >= 1) && pointDistances[i][j]
                                                < kDistances[i][l - 1]) {
                                            kDistances[i][l] = kDistances[i][
                                                    l - 1];
                                            testNeighbors[i][l] =
                                                    testNeighbors[i][l - 1];
                                            l--;
                                        }
                                        kDistances[i][l] = pointDistances[i][j];
                                        testNeighbors[i][l] = j;
                                    }
                                } else {
                                    if (pointDistances[i][j] <
                                            kDistances[i][kIndex - 1]) {
                                        // Search and insert.
                                        l = kIndex - 1;
                                        kDistances[i][kIndex] = kDistances[i][
                                                kIndex - 1];
                                        testNeighbors[i][kIndex] =
                                                testNeighbors[i][kIndex - 1];
                                        while ((l >= 1) && pointDistances[i][j]
                                                < kDistances[i][l - 1]) {
                                            kDistances[i][l] = kDistances[i][
                                                    l - 1];
                                            testNeighbors[i][l] =
                                                    testNeighbors[i][l - 1];
                                            l--;
                                        }
                                        kDistances[i][l] = pointDistances[i][j];
                                        testNeighbors[i][l] = j;
                                        kIndex++;
                                    } else {
                                        kDistances[i][kIndex] =
                                                pointDistances[i][j];
                                        testNeighbors[i][kIndex] = j;
                                        kIndex++;
                                    }
                                }
                            } else {
                                kDistances[i][0] = pointDistances[i][j];
                                testNeighbors[i][0] = j;
                                kIndex = 1;
                            }
                        }
                    }
                }
            }
        }
        return testNeighbors;
    }

    /**
     * This method runs the experimental protocol for multi-threaded
     * cross-validation and compares multiple algorithms on a single dataset.
     * @throws Exception 
     */
    public void performAllTests() throws Exception {
        long allTestStartTime = System.nanoTime();
        int numAlgs = classifiers.length;
        if (approximateNNs && alphaAppKNN < 1f) {
            System.out.println("App kNN sets with alpha " + alphaAppKNN);
        }
        for (int algIndex = 0; algIndex < numAlgs; algIndex++) {
            if (classifiers[algIndex] instanceof DistMatrixUserInterface ||
                    classifiers[algIndex] instanceof
                    learning.supervised.interfaces.
                    DistToPointsQueryUserInterface) {
                distUserPresent = true;
            }
            if (classifiers[algIndex] instanceof NSFUserInterface ||
                    classifiers[algIndex] instanceof
                    learning.supervised.interfaces.
                    NeighborPointsQueryUserInterface) {
                nsfUserPresent = true;
            }
        }
        if (nsfUserPresent) {
            DataSet dataContextForNSF;
            if (dataType instanceof DataSet) {
                dataContextForNSF = (DataSet) dataType;
            } else {
                dataContextForNSF =
                        ((DiscretizedDataSet) dataType).getOriginalData();
            }
            if (secondaryDistanceType == SecondaryDistance.NONE) {
                // In case the classification is done with the primary
                // distances.
                if (!approximateNNs || alphaAppKNN == 1) {
                    // Exact kNN set calculations. A larger NSF object is first
                    // created, so that few recalculations need ever be done
                    // later on.
                    bigNSF = new NeighborSetFinder(dataContextForNSF,
                            totalDistMat, cmet);
                    bigNSF.calculateNeighborSetsMultiThr(2 * kMax + 10,
                            numCommonThreads);
                } else {
                    // Approximate kNN set calculations. A larger NSF object is
                    // first created, so that few recalculations need ever be
                    // done later on.
                    AppKNNGraphLanczosBisection appNSF =
                            new AppKNNGraphLanczosBisection(
                            dataContextForNSF, totalDistMat, 2 * kMax,
                            alphaAppKNN);
                    appNSF.calculateApproximateNeighborSets();
                    bigNSF = NeighborSetFinder.constructFromAppFinder(appNSF,
                            false);
                }
            } else {
                // In case the secondary distances need to be calculated as
                // well.
                if (!approximateNNs || alphaAppKNN == 1) {
                    // Exact kNN calculations.
                    bigNSF = new NeighborSetFinder(dataContextForNSF,
                            totalDistMat, cmet);
                    bigNSF.calculateNeighborSetsMultiThr(secondaryK + kMax + 10,
                            numCommonThreads);
                } else {
                    // Approximate kNN calculations.
                    AppKNNGraphLanczosBisection appNSF =
                            new AppKNNGraphLanczosBisection(dataContextForNSF,
                            totalDistMat, secondaryK + kMax, alphaAppKNN);
                    appNSF.calculateApproximateNeighborSets();
                    bigNSF = NeighborSetFinder.constructFromAppFinder(
                            appNSF, false);
                }
            }
        }
        // Classification estimators..
        currEstimator = new ClassificationEstimator[classifiers.length];
        numFullFolds = new int[numAlgs];
        averageEstimator = new ClassificationEstimator[numAlgs];
        correctPointClassificationArray =
                new float[classifiers.length][data.size()];
        // Initialize the average estimator.
        for (int algIndex = 0; algIndex < numAlgs; algIndex++) {
            averageEstimator[algIndex] = new ClassificationEstimator(
                    new float[numClasses][numClasses]);
            averageEstimator[algIndex].setPrecision(new float[numClasses]);
            averageEstimator[algIndex].setRecall(new float[numClasses]);
        }
        int totalTests = times * numFolds;
        estimators = new ClassificationEstimator[numAlgs][totalTests];
        if (times == 0 && numFolds == 1) {
            // A pathological case.
            execTimeAllOneRun =
                    (System.nanoTime() - allTestStartTime) / 1000;
        }
        if (!foldsLoaded) {
            allFolds = new ArrayList[times][];
        }
        // Now perform the cross-validation.
        for (int i = 0; i < times; i++) {
            // Generate a split into folds.
            int[] classCounts;
            if (dataType instanceof DataSet) {
                classCounts = ((DataSet) dataType).getClassFrequencies();
            } else {
                classCounts = ((DiscretizedDataSet) dataType).
                        getClassFrequencies();
            }
            if (!foldsLoaded) {
                dataFolds = new ArrayList[numFolds];
                foldIndexes = new ArrayList[numFolds];
                for (int foldIndex = 0; foldIndex < numFolds; foldIndex++) {
                    dataFolds[foldIndex] = new ArrayList(400);
                    foldIndexes[foldIndex] = new ArrayList<>(400);
                }
                int[][] classPermutationsIndexes = new int[numClasses][];
                for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                    classPermutationsIndexes[cIndex] =
                            Permutation.obtainRandomPermutation(
                            classCounts[cIndex]);
                }
                // Current class index counters.
                int[] classIndexes = new int[numClasses];
                int targetFold;
                for (int j = 0; j < data.size(); j++) {
                    int label;
                    if (dataType instanceof DataSet) {
                        label = ((DataSet) dataType).getLabelOf(j);
                    } else {
                        label = ((DiscretizedDataSet) dataType).getLabelOf(j);
                    }
                    targetFold = (classPermutationsIndexes[label][
                            classIndexes[label]] + label) % numFolds;
                    dataFolds[targetFold].add(data.get(j));
                    foldIndexes[targetFold].add(j);
                    classIndexes[label]++;
                }
                allFolds[i] = foldIndexes;
            } else {
                dataFolds = new ArrayList[numFolds];
                foldIndexes = new ArrayList[numFolds];
                for (int foldIndex = 0; foldIndex < numFolds; foldIndex++) {
                    dataFolds[foldIndex] = new ArrayList(400);
                    foldIndexes[foldIndex] = new ArrayList<>(400);
                }
                for (int j = 0; j < numFolds; j++) {
                    for (int index = 0; index < allFolds[i][j].size();
                            index++) {
                        dataFolds[j].add(data.get(allFolds[i][j].get(index)));
                        foldIndexes[j].add(allFolds[i][j].get(index));
                    }
                }
            }
            // Now go through all the folds.
            for (int j = 0; j < numFolds; j++) { 
                // Create the training and test data structures.
                currentTestIndexes = foldIndexes[j];
                currentTraining = new ArrayList();
                currentTrainingIndexes = new ArrayList();
                for (int k = 0; k < j; k++) {
                    currentTraining.addAll(dataFolds[k]);
                    currentTrainingIndexes.addAll(foldIndexes[k]);
                }
                for (int k = j + 1; k < numFolds; k++) {
                    currentTraining.addAll(dataFolds[k]);
                    currentTrainingIndexes.addAll(foldIndexes[k]);
                }
                float[][] foldDistMatrix = null;
                float[][] pointDistances = null;
                float[][] foldDistMatrixPrimaryMetric;
                float[][] pointDistancesMPrimaryMetric;
                // The following two are there for the case of instance
                // selection.
                float[][] foldDistMatrixReduced = null;
                float[][] testToTrainingDistancesReduced = null;
                // The index permutation that corresponds to the order in which
                // they are set to the classifiers.
                ArrayList<Integer> trainingIndexesReArr = getDataIndexes(
                        currentTrainingIndexes, dataType);
                if (distUserPresent || nsfUserPresent) {
                    // There exist users of distances and neighbor sets.
                    pointDistancesMPrimaryMetric =
                            new float[currentTestIndexes.size()][
                                    trainingIndexesReArr.size()];
                    int minIndex, maxIndex;
                    for (int indexFirst = 0; indexFirst <
                            currentTestIndexes.size(); indexFirst++) {
                        for (int indexSecond = 0; indexSecond <
                                trainingIndexesReArr.size(); indexSecond++) {
                            minIndex = Math.min(currentTestIndexes.get(
                                    indexFirst), trainingIndexesReArr.get(
                                    indexSecond));
                            maxIndex = Math.max(currentTestIndexes.get(
                                    indexFirst), trainingIndexesReArr.get(
                                    indexSecond));
                            pointDistancesMPrimaryMetric[indexFirst][
                                    indexSecond] = totalDistMat[minIndex][
                                    maxIndex - minIndex - 1];
                        }
                    }
                    foldDistMatrixPrimaryMetric = new float[
                            trainingIndexesReArr.size()][];
                    // Take the fold primary matrix as a sub-matrix of the total
                    // distance matrix.
                    for (int indexFirst = 0; indexFirst <
                            foldDistMatrixPrimaryMetric.length; indexFirst++) {
                        foldDistMatrixPrimaryMetric[indexFirst] =
                                new float[foldDistMatrixPrimaryMetric.length -
                                indexFirst - 1];
                        for (int indexSecond = indexFirst + 1; indexSecond <
                                foldDistMatrixPrimaryMetric.length;
                                indexSecond++) {
                            minIndex = Math.min(trainingIndexesReArr.get(
                                    indexFirst), trainingIndexesReArr.get(
                                    indexSecond));
                            maxIndex = Math.max(trainingIndexesReArr.get(
                                    indexFirst), trainingIndexesReArr.get(
                                    indexSecond));
                            foldDistMatrixPrimaryMetric[indexFirst][
                                    indexSecond - indexFirst - 1] =
                                    totalDistMat[minIndex][maxIndex -
                                    minIndex - 1];
                        }
                    }
                    if (secondaryDistanceType == SecondaryDistance.NONE) {
                        // In this case the primary matrix is used as the fold
                        // matrix.
                        foldDistMatrix = foldDistMatrixPrimaryMetric;
                        pointDistances = pointDistancesMPrimaryMetric;
                    } else {
                        // Initialize the data context for the kNN finder.
                        DataSet dataContextForNSF = new DataSet();
                        int numNom = ((DataInstance) data.get(
                                trainingIndexesReArr.get(0))).getNumNAtt();
                        int numInt = ((DataInstance) data.get(
                                trainingIndexesReArr.get(0))).getNumIAtt();
                        int numFloat = ((DataInstance) data.get(
                                trainingIndexesReArr.get(0))).getNumFAtt();
                        // Generate the generic feature names.
                        if (numFloat > 0) {
                            dataContextForNSF.fAttrNames = new String[numFloat];
                            for (int aInd = 0; aInd < numFloat; aInd++) {
                                dataContextForNSF.fAttrNames[aInd] =
                                        "fAt " + aInd;
                            }
                        }
                        if (numInt > 0) {
                            dataContextForNSF.iAttrNames = new String[numInt];
                            for (int aInd = 0; aInd < numInt; aInd++) {
                                dataContextForNSF.iAttrNames[aInd] =
                                        "iAt " + aInd;
                            }
                        }
                        if (numNom > 0) {
                            dataContextForNSF.sAttrNames = new String[numNom];
                            for (int aInd = 0; aInd < numNom; aInd++) {
                                dataContextForNSF.sAttrNames[aInd] =
                                        "nAt " + aInd;
                            }
                        }
                        // Fill the dataset.
                        dataContextForNSF.data =
                                new ArrayList<>(trainingIndexesReArr.size());
                        for (int dIndex = 0; dIndex < trainingIndexesReArr.size();
                                dIndex++) {
                            dataContextForNSF.data.add(
                                    (DataInstance) data.get(
                                    trainingIndexesReArr.get(dIndex)));
                        }
                        // The kNN finder object with the secondary neighborhood
                        // size, taken as subset of the big-k NSF.
                        NeighborSetFinder nsfSecK;
                        nsfSecK = this.obtainFoldNSF(trainingIndexesReArr,
                                dataContextForNSF, foldDistMatrixPrimaryMetric,
                                secondaryK);

                        if (secondaryDistanceType == SecondaryDistance.SIMCOS) {
                            // The simcos shared neighbor secondary distance
                            // measure.
                            SharedNeighborFinder snf =
                                    new SharedNeighborFinder(nsfSecK, kValue);
                            snf.setNumClasses(numClasses);
                            snf.countSharedNeighborsMultiThread(
                                    numCommonThreads);
                            // First fetch the similarities.
                            foldDistMatrix = snf.getSharedNeighborCounts();
                            // Then transform them into distances.
                            for (int indexFirst = 0; indexFirst <
                                    foldDistMatrix.length; indexFirst++) {
                                for (int indexSecond = 0; indexSecond <
                                        foldDistMatrix[indexFirst].length;
                                        indexSecond++) {
                                    foldDistMatrix[indexFirst][indexSecond] =
                                            secondaryK -
                                            foldDistMatrix[indexFirst][
                                            indexSecond];
                                }
                            }
                            // Calculate the test-to-training point distances.
                            SharedNeighborCalculator snc =
                                    new SharedNeighborCalculator(
                                    snf,SharedNeighborCalculator.
                                    WeightingType.NONE);
                            DataInstance firstInstance, secondInstance;
                            pointDistances = new float[
                                    currentTestIndexes.size()][
                                    trainingIndexesReArr.size()];
                            int[][] pointNeighborsSec = new int[
                                    currentTestIndexes.size()][secondaryK];
                            for (int index = 0; index <
                                    currentTestIndexes.size(); index++) {
                                firstInstance = (DataInstance) (data.get(
                                        currentTestIndexes.get(index)));
                                pointNeighborsSec[index] =
                                        NeighborSetFinder.getIndexesOfNeighbors(
                                        dataContextForNSF, firstInstance,
                                        secondaryK,
                                        pointDistancesMPrimaryMetric[index]);
                            }
                            for (int firstIndex = 0; firstIndex <
                                    currentTestIndexes.size(); firstIndex++) {
                                for (int secondIndex = 0; secondIndex <
                                        trainingIndexesReArr.size();
                                        secondIndex++) {
                                    firstInstance = (DataInstance) (data.get(
                                            currentTestIndexes.get(
                                            firstIndex)));
                                    secondInstance = (DataInstance) (
                                            data.get(trainingIndexesReArr.get(
                                            secondIndex)));
                                    pointDistances[firstIndex][secondIndex] =
                                            snc.dist(firstInstance,
                                            secondInstance, pointNeighborsSec[
                                            firstIndex], 
                                            nsfSecK.getKNeighbors()[
                                            secondIndex]);
                                }
                            }
                        } else if (secondaryDistanceType ==
                                SecondaryDistance.SIMHUB) {
                            // The simhub hubness-aware secondary
                            // shared-neighbor distances.
                            SharedNeighborFinder snf =
                                    new SharedNeighborFinder(nsfSecK, kValue);
                            snf.setNumClasses(numClasses);
                            snf.obtainWeightsFromHubnessInformation(0);
                            snf.countSharedNeighborsMultiThread(
                                    numCommonThreads);
                            foldDistMatrix = snf.getSharedNeighborCounts();
                            for (int indexFirst = 0; indexFirst <
                                    foldDistMatrix.length; indexFirst++) {
                                for (int indexSecond = 0; indexSecond <
                                        foldDistMatrix[indexFirst].length;
                                        indexSecond++) {
                                    foldDistMatrix[indexFirst][indexSecond] =
                                            secondaryK - foldDistMatrix[
                                            indexFirst][indexSecond];
                                }
                            }
                            // Calculate the test-to-training point distances.
                            SharedNeighborCalculator snc =
                                    new SharedNeighborCalculator(snf,
                                    SharedNeighborCalculator.WeightingType.
                                    HUBNESS_INFORMATION);
                            DataInstance firstInstance, secondInstance;
                            pointDistances = new float[
                                    currentTestIndexes.size()][
                                    trainingIndexesReArr.size()];
                            int[][] pointNeighborsSec = new int[
                                    currentTestIndexes.size()][secondaryK];
                            for (int index = 0; index <
                                    currentTestIndexes.size(); index++) {
                                firstInstance = (DataInstance) (data.get(
                                        currentTestIndexes.get(index)));
                                pointNeighborsSec[index] =
                                        NeighborSetFinder.getIndexesOfNeighbors(
                                        dataContextForNSF, firstInstance,
                                        secondaryK,
                                        pointDistancesMPrimaryMetric[index]);
                            }
                            for (int indexFirst = 0; indexFirst <
                                    currentTestIndexes.size(); indexFirst++) {
                                for (int indexSecond = 0; indexSecond <
                                        trainingIndexesReArr.size(); indexSecond++) {
                                    firstInstance = (DataInstance) (data.get(
                                            currentTestIndexes.get(indexFirst)));
                                    secondInstance = (DataInstance) (data.get(
                                            trainingIndexesReArr.get(indexSecond)));
                                    pointDistances[indexFirst][indexSecond] =
                                            snc.dist(firstInstance,
                                            secondInstance, pointNeighborsSec[
                                            indexFirst],
                                            nsfSecK.getKNeighbors()[
                                            indexSecond]);
                                }
                            }
                        } else if (secondaryDistanceType ==
                                SecondaryDistance.MP) {
                            // Use mutual proximity as the secondary distance
                            // measure.
                            MutualProximityCalculator calc =
                                    new MutualProximityCalculator(
                                    nsfSecK.getDistances(),
                                    nsfSecK.getDataSet(),
                                    nsfSecK.getCombinedMetric());
                            foldDistMatrix =
                                    calc.calculateSecondaryDistMatrixMultThr(
                                    nsfSecK, 8);
                            // Calculate the test-to-training point distances.
                            DataInstance firstInstance, secondInstance;
                            pointDistances = new float[
                                    currentTestIndexes.size()][
                                    trainingIndexesReArr.size()];
                            int[][] pointNeighborsSec = new int[
                                    currentTestIndexes.size()][secondaryK];
                            for (int index = 0; index <
                                    currentTestIndexes.size(); index++) {
                                firstInstance = (DataInstance) (
                                        data.get(
                                        currentTestIndexes.get(index)));
                                pointNeighborsSec[index] =
                                        NeighborSetFinder.getIndexesOfNeighbors(
                                        dataContextForNSF, firstInstance,
                                        secondaryK,
                                        pointDistancesMPrimaryMetric[index]);
                            }
                            for (int indexFirst = 0; indexFirst <
                                    currentTestIndexes.size(); indexFirst++) {
                                for (int indexSecond = 0; indexSecond <
                                        trainingIndexesReArr.size(); indexSecond++) {
                                    firstInstance = (DataInstance) (data.get(
                                            currentTestIndexes.get(
                                            indexFirst)));
                                    secondInstance = (DataInstance) (data.get(
                                            trainingIndexesReArr.get(indexSecond)));
                                    int[] firstNeighbors = pointNeighborsSec[
                                            indexFirst];
                                    float[] kDistsFirst = new float[secondaryK];
                                    float[] kDistsSecond =
                                            nsfSecK.getKDistances()[
                                            indexSecond];
                                    for (int kInd = 0; kInd < secondaryK;
                                            kInd++) {
                                        kDistsFirst[kInd] = 
                                                pointDistancesMPrimaryMetric[
                                                indexFirst][firstNeighbors[
                                                kInd]];
                                    }
                                    pointDistances[indexFirst][indexSecond] =
                                            calc.dist(firstInstance,
                                            secondInstance, kDistsFirst,
                                            kDistsSecond);
                                }
                            }
                        } else if (secondaryDistanceType ==
                                SecondaryDistance.LS) {
                            // Local scaling as the secondary distance measure.
                            LocalScalingCalculator lsc =
                                    new LocalScalingCalculator(nsfSecK);
                            foldDistMatrix =
                                    lsc.getTransformedDMatFromNSFPrimaryDMat();
                            // Calculate the test-to-training point distances.
                            DataInstance firstInstance, secondInstance;
                            pointDistances = new float[
                                    currentTestIndexes.size()][
                                    trainingIndexesReArr.size()];
                            int[][] pointNeighborsSec = new int[
                                    currentTestIndexes.size()][secondaryK];
                            for (int index = 0; index <
                                    currentTestIndexes.size(); index++) {
                                firstInstance = (DataInstance) (
                                        data.get(currentTestIndexes.get(
                                        index)));
                                pointNeighborsSec[index] =
                                        NeighborSetFinder.getIndexesOfNeighbors(
                                        dataContextForNSF, firstInstance,
                                        secondaryK,
                                        pointDistancesMPrimaryMetric[index]);
                            }
                            for (int indexFirst = 0; indexFirst <
                                    currentTestIndexes.size(); indexFirst++) {
                                for (int indexSecond = 0; indexSecond <
                                        trainingIndexesReArr.size();
                                        indexSecond++) {
                                    firstInstance = (DataInstance) (data.get(
                                            currentTestIndexes.get(
                                            indexFirst)));
                                    secondInstance = (DataInstance) (data.get(
                                            trainingIndexesReArr.get(
                                            indexSecond)));
                                    int[] firstNeighbors = pointNeighborsSec[
                                            indexFirst];
                                    float[] kDistsFirst = new float[secondaryK];
                                    float[] kDistsSecond =
                                            nsfSecK.getKDistances()[
                                            indexSecond];
                                    for (int kInd = 0; kInd < secondaryK;
                                            kInd++) {
                                        kDistsFirst[kInd] =
                                                pointDistancesMPrimaryMetric[
                                                indexFirst][firstNeighbors[
                                                kInd]];
                                    }
                                    pointDistances[indexFirst][indexSecond] =
                                            lsc.distFromKDists(firstInstance,
                                            secondInstance, kDistsFirst,
                                            kDistsSecond);
                                }
                            }
                        } else if (secondaryDistanceType ==
                                SecondaryDistance.NICDM) {
                            // NICDM secondary distance measure.
                            NICDMCalculator nsc = new NICDMCalculator(nsfSecK);
                            foldDistMatrix =
                                    nsc.getTransformedDMatFromNSFPrimaryDMat();
                            // Calculate the test-to-training point distances.
                            DataInstance firstInstance, secondInstance;
                            pointDistances = new float[
                                    currentTestIndexes.size()][
                                    trainingIndexesReArr.size()];
                            int[][] pointNeighborsSec = new int[
                                    currentTestIndexes.size()][secondaryK];
                            for (int index = 0; index <
                                    currentTestIndexes.size(); index++) {
                                firstInstance = (DataInstance) (
                                        data.get(currentTestIndexes.get(
                                        index)));
                                pointNeighborsSec[index] =
                                        NeighborSetFinder.getIndexesOfNeighbors(
                                        dataContextForNSF, firstInstance,
                                        secondaryK,
                                        pointDistancesMPrimaryMetric[index]);
                            }
                            for (int indexFirst = 0; indexFirst <
                                    currentTestIndexes.size(); indexFirst++) {
                                for (int indexSecond = 0; indexSecond <
                                        trainingIndexesReArr.size();
                                        indexSecond++) {
                                    firstInstance = (DataInstance) (
                                            data.get(currentTestIndexes.get(
                                            indexFirst)));
                                    secondInstance = (DataInstance) (
                                            data.get(trainingIndexesReArr.get(
                                            indexSecond)));
                                    int[] firstNeighbors =
                                            pointNeighborsSec[indexFirst];
                                    float[] kDistsFirst = new float[secondaryK];
                                    float[] kDistsSecond =
                                            nsfSecK.getKDistances()[
                                            indexSecond];
                                    for (int kInd = 0; kInd < secondaryK;
                                            kInd++) {
                                        kDistsFirst[kInd] =
                                                pointDistancesMPrimaryMetric[
                                                indexFirst][firstNeighbors[
                                                kInd]];
                                    }
                                    pointDistances[indexFirst][indexSecond] =
                                            nsc.distFromKDists(firstInstance,
                                            secondInstance, kDistsFirst,
                                            kDistsSecond);
                                }
                            }
                        }

                    }
                }
                if (nsfUserPresent && kMode == SINGLE) {
                    // Again, generate an appropriate data context for the fold
                    // NeighborSetFinder objects.
                    DataSet dataContextForNSF = new DataSet();
                    int numNom = 0;
                    int numInt = 0;
                    int numFloat = 0;
                    if (data.get(trainingIndexesReArr.get(0)) instanceof
                            DataInstance) {
                        numNom = ((DataInstance) data.get(
                                trainingIndexesReArr.get(0))).getNumNAtt();
                        numInt = ((DataInstance) data.get(
                                trainingIndexesReArr.get(0))).getNumIAtt();
                        numFloat = ((DataInstance) data.get(
                                trainingIndexesReArr.get(0))).getNumFAtt();
                    } else if (data.get(trainingIndexesReArr.get(0)) instanceof
                            DiscretizedDataInstance) {
                        numNom = ((DiscretizedDataInstance) data.get(
                                trainingIndexesReArr.get(0))).
                                getOriginalInstance().getNumNAtt();
                        numInt = ((DiscretizedDataInstance) data.get(
                                trainingIndexesReArr.get(0))).
                                getOriginalInstance().getNumIAtt();
                        numFloat = ((DiscretizedDataInstance) data.get(
                                trainingIndexesReArr.get(0))).
                                getOriginalInstance().getNumFAtt();
                    }
                    // Generate the generic feature names.
                    if (numFloat > 0) {
                        dataContextForNSF.fAttrNames = new String[numFloat];
                        for (int aInd = 0; aInd < numFloat; aInd++) {
                            dataContextForNSF.fAttrNames[aInd] = "fAt " + aInd;
                        }
                    }
                    if (numInt > 0) {
                        dataContextForNSF.iAttrNames = new String[numInt];
                        for (int aInd = 0; aInd < numInt; aInd++) {
                            dataContextForNSF.iAttrNames[aInd] = "iAt " + aInd;
                        }
                    }
                    if (numNom > 0) {
                        dataContextForNSF.sAttrNames = new String[numNom];
                        for (int aInd = 0; aInd < numNom; aInd++) {
                            dataContextForNSF.sAttrNames[aInd] = "nAt " + aInd;
                        }
                    }

                    dataContextForNSF.data = new ArrayList<>(
                            trainingIndexesReArr.size());
                    if (data.get(trainingIndexesReArr.get(0)) instanceof
                            DataInstance) {
                        // The continuous case.
                        for (int dIndex = 0; dIndex < trainingIndexesReArr.size();
                                dIndex++) {
                            dataContextForNSF.data.add((DataInstance) data.get(
                                    trainingIndexesReArr.get(dIndex)));
                        }
                    } else if (data.get(trainingIndexesReArr.get(0)) instanceof
                            DiscretizedDataInstance) {
                        // The discretized case.
                        for (int dIndex = 0; dIndex < trainingIndexesReArr.size();
                                dIndex++) {
                            dataContextForNSF.data.add(((
                                    DiscretizedDataInstance) data.get(
                                    trainingIndexesReArr.get(dIndex))).
                                    getOriginalInstance());
                        }
                    }
                    for (int cInd = 0; cInd < numAlgs; cInd++) {
                        // Explicitly state that the classifiers may not modify
                        // the NeighborSetFinder object on the fly without
                        // making a copy.
                        if (classifiers[cInd] instanceof NSFUserInterface) {
                            ((NSFUserInterface) (classifiers[cInd])).
                                    noRecalcs();
                        }
                    }
                    if (secondaryDistanceType == SecondaryDistance.NONE) {
                        // Obtain the fold NeighborSetFinder object from the
                        // primary kNN sets.
                        nsfCurrent = obtainFoldNSF(trainingIndexesReArr,
                                dataContextForNSF, foldDistMatrix, kValue);
                        if (dreducer == null) {
                            testPointNeighbors = obtainFoldTestNeighbors(
                                    trainingIndexesReArr, currentTestIndexes,
                                    dataContextForNSF, pointDistances, kValue);
                        }
                    } else {
                        // In this case, we need to calculate the kNN sets from
                        // the secondary distance matrices.
                        if (!approximateNNs || alphaAppKNN == 1f) {
                            nsfCurrent = new NeighborSetFinder(
                                    dataContextForNSF, foldDistMatrix, cmet);
                            nsfCurrent.calculateNeighborSetsMultiThr(kValue,
                                    numCommonThreads);
                        } else {
                            AppKNNGraphLanczosBisection appNSF =
                                    new AppKNNGraphLanczosBisection(
                                    dataContextForNSF, foldDistMatrix,
                                    kValue, alphaAppKNN);
                            appNSF.calculateApproximateNeighborSets();
                            nsfCurrent = NeighborSetFinder.
                                    constructFromAppFinder(appNSF, false);
                        }
                        if (dreducer == null) {
                            testPointNeighbors =
                                    new int[currentTestIndexes.size()][];
                            for (int ni = 0; ni <
                                    testPointNeighbors.length; ni++) {
                                DataInstance inst = (DataInstance) (
                                        data.get(currentTestIndexes.get(ni)));
                                testPointNeighbors[ni] = NeighborSetFinder.
                                        getIndexesOfNeighbors(
                                        dataContextForNSF, inst, kValue,
                                        pointDistances[ni]);
                            }
                        }
                    }
                    // Perform instance selection (optional).
                    if (dreducer != null) {
                        foldReducer = dreducer.copy();
                        foldReducer.setOriginalDataSet(dataContextForNSF);
                        if (foldReducer instanceof NSFUserInterface) {
                            // If the selector needs the kNN sets, we provide
                            // them.
                            ((NSFUserInterface) foldReducer).setNSF(
                                    nsfCurrent.copy());
                        }
                        if (DataMineConstants.isZero(selectionRate)) {
                            // Find the appropriate selection rate
                            // automatically, if possible.
                            foldReducer.reduceDataSet();
                        } else {
                            // Perform the reduction with the specified
                            // selection rate.
                            foldReducer.reduceDataSet(selectionRate);
                        }
                        foldReducer.sortSelectedIndexes();
                        if (nsfUserPresent) {
                            // Calculate the unbiased hubness estimates.
                            foldReducer.calculatePrototypeHubness(kValue);
                        }
                        // The data context for kNN calculations on the
                        // selected prototype set.
                        DataSet dataContextForNSFReduced =
                                dataContextForNSF.cloneDefinition();
                        dataContextForNSFReduced.data = null;
                        ArrayList<Integer> protoIndexes =
                                foldReducer.getPrototypeIndexes();
                        currentPrototypeIndexes =
                                new ArrayList<>(protoIndexes.size());
                        for (int dIndex = 0; dIndex < protoIndexes.size();
                                dIndex++) {
                            currentPrototypeIndexes.add(trainingIndexesReArr.get(
                                    protoIndexes.get(dIndex)));
                        }
                        dataContextForNSFReduced.data =
                                new ArrayList<>(protoIndexes.size());
                        for (int dIndex = 0; dIndex < protoIndexes.size();
                                dIndex++) {
                            dataContextForNSFReduced.data.add((DataInstance)
                                    data.get(currentPrototypeIndexes.get(
                                    dIndex)));
                        }

                        foldDistMatrixReduced =
                                new float[currentPrototypeIndexes.size()][];
                        int minIndex, maxIndex;
                        for (int indexFirst = 0; indexFirst <
                                foldDistMatrixReduced.length; indexFirst++) {
                            foldDistMatrixReduced[indexFirst] =
                                    new float[foldDistMatrixReduced.length -
                                    indexFirst - 1];
                            for (int indexSecond = indexFirst + 1; indexSecond <
                                    foldDistMatrixReduced.length;
                                    indexSecond++) {
                                minIndex = Math.min(
                                        currentPrototypeIndexes.get(
                                        indexFirst), currentPrototypeIndexes.
                                        get(indexSecond));
                                maxIndex = Math.max(currentPrototypeIndexes.
                                        get(indexFirst),
                                        currentPrototypeIndexes.get(
                                        indexSecond));
                                foldDistMatrixReduced[indexFirst][indexSecond -
                                        indexFirst - 1] = totalDistMat[
                                        minIndex][maxIndex - minIndex - 1];
                            }
                        }
                        testToTrainingDistancesReduced = new float[
                                currentTestIndexes.size()][
                                currentPrototypeIndexes.size()];
                        for (int indexFirst = 0; indexFirst <
                                currentTestIndexes.size();indexFirst++) {
                            for (int indexSecond = 0; indexSecond <
                                    currentPrototypeIndexes.size();
                                    indexSecond++) {
                                minIndex = Math.min(currentTestIndexes.get(
                                        indexFirst), currentPrototypeIndexes.
                                        get(indexSecond));
                                maxIndex = Math.max(currentTestIndexes.get(
                                        indexFirst), currentPrototypeIndexes.
                                        get(indexSecond));
                                testToTrainingDistancesReduced[indexFirst][
                                        indexSecond] = totalDistMat[minIndex][
                                        maxIndex - minIndex - 1];
                            }
                        }
                        if (secondaryDistanceType == SecondaryDistance.NONE) {
                            nsfCurrent = obtainFoldNSF(trainingIndexesReArr,
                                    dataContextForNSF, foldDistMatrix, kValue);
                            NeighborSetFinder nsfProto = nsfCurrent.getSubNSF(
                                    kValue, protoIndexes, foldDistMatrixReduced,
                                    dataContextForNSFReduced);
                            nsfCurrent = nsfProto;
                        } else {
                            if (!approximateNNs || alphaAppKNN == 1f) {
                                nsfCurrent = new NeighborSetFinder(
                                        dataContextForNSFReduced,
                                        foldDistMatrixReduced, cmet);
                                nsfCurrent.calculateNeighborSetsMultiThr(
                                        kValue, numCommonThreads);
                            } else {
                                AppKNNGraphLanczosBisection appNSF =
                                        new AppKNNGraphLanczosBisection(
                                        dataContextForNSFReduced,
                                        foldDistMatrixReduced, kValue,
                                        alphaAppKNN);
                                appNSF.calculateApproximateNeighborSets();
                                nsfCurrent =
                                        NeighborSetFinder.
                                        constructFromAppFinder(appNSF, false);
                            }
                        }
                        testPointNeighbors =
                                new int[currentTestIndexes.size()][];
                        for (int index = 0; index < testPointNeighbors.length;
                                index++) {
                            DataInstance inst = (DataInstance) (data.get(
                                    currentTestIndexes.get(index)));
                            testPointNeighbors[index] = NeighborSetFinder.
                                    getIndexesOfNeighbors(
                                    dataContextForNSFReduced, inst, kValue,
                                    testToTrainingDistancesReduced[index]);
                        }
                    }
                }
                // Train the models and test the algorithms.
                Thread[] algThreads = new Thread[numAlgs];
                for (int algIndex = 0; algIndex < numAlgs; algIndex++) {
                    algThreads[algIndex] = null;
                    if (dreducer == null) {
                        algThreads[algIndex] = new Thread(
                                new AlgorithmTesterThread(
                                algIndex, i, j, foldDistMatrix, pointDistances,
                                testPointNeighbors));
                    } else {
                        algThreads[algIndex] = new Thread(
                                new AlgorithmTesterThread(
                                algIndex, i, j, foldDistMatrixReduced,
                                testToTrainingDistancesReduced,
                                testPointNeighbors));
                    }
                    algThreads[algIndex].start();
                }
                for (int algIndex = 0; algIndex < numAlgs; algIndex++) {
                    if (algThreads[algIndex] != null) {
                        try {
                            algThreads[algIndex].join();
                        } catch (Throwable t) {
                        }
                    }
                }
                System.out.print("|");
                if ((i * numFolds + j - 4) % 5 == 0) {
                    System.out.print(" ");
                }
            }
            System.gc();
        }
        System.out.println();
        // Turn the sums in the average estimator into averages by normalizing
        // them.
        for (int algIndex = 0; algIndex < numAlgs; algIndex++) {
            averageEstimator[algIndex].setAccuracy(averageEstimator[algIndex].
                    getAccuracy() / (float) totalTests);
            averageEstimator[algIndex].setAvgPrecision(
                    averageEstimator[algIndex].getAvgPrecision() /
                    (float) totalTests);
            averageEstimator[algIndex].setAvgRecall(averageEstimator[algIndex].
                    getAvgRecall() / (float) totalTests);
            averageEstimator[algIndex].setMicroFMeasure(averageEstimator[
                    algIndex].getMicroFMeasure() / (float) totalTests);
            averageEstimator[algIndex].setMacroFMeasure(averageEstimator[
                    algIndex].getMacroFMeasure() / (float) totalTests);
            averageEstimator[algIndex].setMatthewsCorrCoef(averageEstimator[
                    algIndex].getMatthewsCorrCoef() / (float) totalTests);
            if (numFullFolds[algIndex] > 0) {
                float[][] avgConfMat = averageEstimator[algIndex].
                        getConfusionMatrix();
                for (int cFirst = 0; cFirst < numClasses; cFirst++) {
                    averageEstimator[algIndex].
                        getPrecision()[cFirst] /= numFullFolds[algIndex];
                    averageEstimator[algIndex].getRecall()[cFirst] /=
                            numFullFolds[algIndex];
                    for (int cSecond = 0; cSecond < numClasses; cSecond++) {
                        avgConfMat[cFirst][cSecond] /= times;
                    }
                }
            }
        }
        for (int i = 0; i < execTimeTotal.length; i++) {
            execTimeTotal[i] /= 1000;
        }
    }

    /**
     * This class implements the thread-based algorithm testing and evaluation,
     * so that one thread evaluates one classifier.
     */
    class AlgorithmTesterThread implements Runnable {

        private int algIndex;
        private int repetitionIndex;
        private int foldIndex;
        private float[][] foldDistMatrix = null;
        private float[][] testToTrainingDistances = null;
        private int[][] testToTrainingNeighbors = null;

        /**
         * Initialization.
         * @param algIndex Integer that is the index of the classifier to
         * evaluate.
         * @param repetitionIndex Integer that is the index of the current
         * repetition in the CV framework.
         * @param foldIndex Integer that is the fold index in the current CV
         * framework.
         */
        public AlgorithmTesterThread(int algIndex, int repetitionIndex,
                int foldIndex) {
            this.algIndex = algIndex;
            this.repetitionIndex = repetitionIndex;
            this.foldIndex = foldIndex;
        }

        /**
         * Initialization.
         * @param algIndex Integer that is the index of the classifier to
         * evaluate.
         * @param repetitionIndex Integer that is the index of the current
         * repetition in the CV framework.
         * @param foldIndex Integer that is the fold index in the current CV
         * framework.
         * @param foldDistMatrix float[][] representing the upper triangular
         * distance matrix on the training data.
         * @param testToTrainingDistances float[][] representing the distances
         * between the test and the training points.
         * @param testToTrainingNeighbors int[][] representing the neighbors
         * of the test points among the training points.
         */
        public AlgorithmTesterThread(int algIndex,
                int repetitionIndex,
                int foldIndex,
                float[][] foldDistMatrix,
                float[][] testToTrainingDistances,
                int[][] testToTrainingNeighbors) {
            this.algIndex = algIndex;
            this.repetitionIndex = repetitionIndex;
            this.foldIndex = foldIndex;
            this.foldDistMatrix = foldDistMatrix;
            this.testToTrainingDistances = testToTrainingDistances;
            this.testToTrainingNeighbors = testToTrainingNeighbors;
        }

        @Override
        public void run() {
            try {
                long startTime = System.nanoTime();
                // The following method essentially does all the testing.
                testAndEvaluateClassifier(algIndex, repetitionIndex, foldIndex,
                        foldDistMatrix, testToTrainingDistances,
                        testToTrainingNeighbors);
                long endTime = System.nanoTime();
                execTimeTotal[algIndex] += (startTime - endTime);
            } catch (Exception e) {
                // In case some error occurs.
                System.err.println("Algorithm index: " + algIndex);
                System.err.println("Error while testing " +
                        currClassifierInstances[algIndex].getClass().getName());
                System.err.println(e.getMessage());
            }
        }
    }

    /**
     * This method does all the classifier testing and evaluation.
     * 
     * @param algIndex Integer that is the index of the classifier to evaluate.
     * @param repetitionIndex Integer that is the index of the current
     * repetition in the CV framework.
     * @param foldIndex Integer that is the fold index in the current CV
     * framework.
     * @param foldDistMatrix float[][] representing the upper triangular
     * distance matrix on the training data.
     * @param testToTrainingDistances float[][] representing the distances
     * between the test and the training points.
     * @param testToTrainingNeighbors int[][] representing the neighbors of the
     * test points among the training points.
     * @throws Exception 
     */
    public void testAndEvaluateClassifier(int algIndex, int repetitionIndex,
            int foldIndex, float[][] foldDistMatrix,
            float[][] testToTrainingDistances,
            int[][] testToTrainingNeighbors) throws Exception {
        // Use the prototype to spawn a new, empty copy of the initial
        // classifier configuration.
        currClassifierInstances[algIndex] =
                classifiers[algIndex].copyConfiguration();
        if (dreducer == null) {
            // No instance selection.
            currClassifierInstances[algIndex].setDataIndexes(
                    currentTrainingIndexes, dataType);
        } else {
            // Instance selection.
            currClassifierInstances[algIndex].setDataIndexes(
                    currentPrototypeIndexes, dataType);
        }
        if (currClassifierInstances[algIndex] instanceof
                DistMatrixUserInterface) {
            // Provide the fold distance matrix, if needed.
            ((DistMatrixUserInterface) currClassifierInstances[algIndex]).
                    setDistMatrix(foldDistMatrix);
        }
        if (currClassifierInstances[algIndex] instanceof
                AutomaticKFinderInterface && kMode == INTERVAL) {
            // Find the optimal k value, if so specified.
            ((AutomaticKFinderInterface) currClassifierInstances[algIndex]).
                    findK(kMin, kMax);
        }
        if (currClassifierInstances[algIndex] instanceof NSFUserInterface &&
                kMode == SINGLE) {
            // Set the training kNN graph, if required.
            ((NSFUserInterface) currClassifierInstances[algIndex]).setNSF(
                    nsfCurrent);
        }
        if (dreducer == null) {
            // Train the classifier.
            currClassifierInstances[algIndex].train();
        } else {
            if (protoHubnessMode != PROTO_UNBIASED) {
                // Just train on the reduced data with the same method as with
                // no instance selection.
                currClassifierInstances[algIndex].train();
            } else {
                // Train while compensating for the instance selection bias.
                currClassifierInstances[algIndex].trainOnReducedData(
                        foldReducer);
            }
        }
        // Test the classifier. Different methods are invoked based on the
        // interfaces that the classifier implements.
        if (currClassifierInstances[algIndex] instanceof
                DistToPointsQueryUserInterface ||
                currClassifierInstances[algIndex] instanceof
                NeighborPointsQueryUserInterface) {
            if (validateOnExternalLabels) {
                currEstimator[algIndex] = currClassifierInstances[algIndex].
                        test(correctPointClassificationArray[algIndex],
                        currentTestIndexes, dataType, testLabelArray,
                        numClasses, testToTrainingDistances,
                        testToTrainingNeighbors);
            } else {
                currEstimator[algIndex] = currClassifierInstances[algIndex].
                        test(correctPointClassificationArray[algIndex],
                        currentTestIndexes, dataType, numClasses,
                        testToTrainingDistances, testToTrainingNeighbors);
            }
        } else if (currClassifierInstances[algIndex] instanceof
                DiscreteDistToPointsQueryUserInterface ||
                currClassifierInstances[algIndex] instanceof
                DiscreteNeighborPointsQueryUserInterface) {
            if (validateOnExternalLabels) {
                currEstimator[algIndex] = currClassifierInstances[algIndex].
                        test(correctPointClassificationArray[algIndex],
                        currentTestIndexes, dataType, testLabelArray,
                        numClasses, testToTrainingDistances,
                        testToTrainingNeighbors);
            } else {
                currEstimator[algIndex] = currClassifierInstances[algIndex].
                        test(correctPointClassificationArray[algIndex],
                        currentTestIndexes, dataType, numClasses,
                        testToTrainingDistances, testToTrainingNeighbors);
            }
        } else {
            if (validateOnExternalLabels) {
                currEstimator[algIndex] = currClassifierInstances[algIndex].
                        test(correctPointClassificationArray[algIndex],
                        currentTestIndexes, dataType, testLabelArray,
                        numClasses);
            } else {
                currEstimator[algIndex] = currClassifierInstances[algIndex].
                        test(correctPointClassificationArray[algIndex],
                        currentTestIndexes, dataType, numClasses);
            }
        }
        // Save the evaluation object.
        if (keepAllEvaluations) {
            estimators[algIndex][repetitionIndex * numFolds + foldIndex] =
                    currEstimator[algIndex];
        }
        // Sum the evaluations up in the cumulative classification estimator.
        // They will be normalized into proper averages later on.
        if (currEstimator[algIndex].getConfusionMatrix().length == numClasses) {
            numFullFolds[algIndex]++;
            float[][] avgConfMat =
                    averageEstimator[algIndex].getConfusionMatrix();
            float[][] currConfMat = currEstimator[algIndex].getConfusionMatrix();
            float[] currPrecision = currEstimator[algIndex].getPrecision();
            float[] currRecall = currEstimator[algIndex].getRecall();
            for (int cFirst = 0; cFirst < numClasses; cFirst++) {
                averageEstimator[algIndex].getPrecision()[cFirst] +=
                        currPrecision[cFirst];
                averageEstimator[algIndex].getRecall()[cFirst] +=
                        currRecall[cFirst];
                for (int cSecond = 0; cSecond < numClasses; cSecond++) {
                    avgConfMat[cFirst][cSecond] += currConfMat[cFirst][cSecond];
                }
            }
        }
        averageEstimator[algIndex].setAccuracy(averageEstimator[algIndex].
                getAccuracy() + currEstimator[algIndex].getAccuracy());
        averageEstimator[algIndex].setAvgPrecision(averageEstimator[algIndex].
                getAvgPrecision() + currEstimator[algIndex].getAvgPrecision());
        averageEstimator[algIndex].setAvgRecall(averageEstimator[algIndex].
                getAvgRecall() + currEstimator[algIndex].getAvgRecall());
        averageEstimator[algIndex].setMicroFMeasure(averageEstimator[algIndex].
                getMicroFMeasure() +
                currEstimator[algIndex].getMicroFMeasure());
        averageEstimator[algIndex].setMacroFMeasure(averageEstimator[algIndex].
                getMacroFMeasure() +
                currEstimator[algIndex].getMacroFMeasure());
        averageEstimator[algIndex].setMatthewsCorrCoef(
                averageEstimator[algIndex].getMatthewsCorrCoef() +
                currEstimator[algIndex].getMatthewsCorrCoef());
    }

    /**
     * @return ClassificationEstimator[][] representing an array of arrays of
     * classification estimators for each algorithm and round in the
     * cross-validation framework.
     */
    public ClassificationEstimator[][] getEstimators() {
        return estimators;
    }

    /**
     * @return ClassificationEstimator[] representing the average classification
     * results for each algorithm.
     */
    public ClassificationEstimator[] getAverageResults() {
        return averageEstimator;
    }

    /**
     * @return Integer that is the number of classes in the data.
     */
    public int getNumClasses() {
        return numClasses;
    }

    /**
     * @param numClasses Integer that is the number of classes in the data.
     */
    public void setNumClasses(int numClasses) {
        this.numClasses = numClasses;
    }

    /**
     * @param classifiers ValidateableInterface[] representing the classifiers
     * to be evaluated.
     */
    public void setClassifiers(ValidateableInterface[] classifiers) {
        this.classifiers = classifiers;
    }

    /**
     * @param classifier ValidateableInterface representing a single classifier
     * to be evaluated.
     */
    public void setClassifier(ValidateableInterface classifier) {
        this.classifiers = new ValidateableInterface[1];
        this.classifiers[0] = classifier;
    }

    /**
     * @return ValidateableInterface[] representing the classifiers to be
     * evaluated.
     */
    public ValidateableInterface[] getClassifiers() {
        return classifiers;
    }

    /**
     * @param times Integer that is the number of repetitions in the
     * cross-validation framework.
     */
    public void setTimes(int times) {
        this.times = times;
    }

    /**
     * @param folds Integer that is the number of folds in the cross-validation
     * framework.
     */
    public void setFolds(int folds) {
        this.numFolds = folds;
    }

    /**
     * @return Integer that is the number of repetitions in the cross-validation
     * framework.
     */
    public int getTimes() {
        return times;
    }

    /**
     * @return Integer that is the number of folds in the cross-validation
     * framework.
     */
    public int getFolds() {
        return numFolds;
    }

    /**
     * @param dataType Object that is the evaluation data context.
     */
    public void setDataType(Object dataType) {
        this.dataType = dataType;
    }

    /**
     * @return Object that is the evaluation data context.
     */
    public Object getDataType() {
        return dataType;
    }

    /**
     * @param data ArrayList of data instances.
     */
    public void setData(ArrayList data) {
        this.data = data;
    }

    /**
     * @return ArrayList of data instances.
     */
    public ArrayList getData() {
        return data;
    }

    /**
     * @param keepAll Boolean flag indicating whether to keep track of all
     * classifier evaluations or just the average ones after all the runs. The
     * default is true.
     */
    public void setKeepAll(boolean keepAll) {
        this.keepAllEvaluations = keepAll;
    }

    /**
     * @return Boolean flag indicating whether to keep track of all classifier
     * evaluations or just the average ones after all the runs. The default is
     * true.
     */
    public boolean getKeepAll() {
        return keepAllEvaluations;
    }

    /**
     * Gets the re-ordered indexes, the way that they will be set to the
     * classifiers.
     * 
     * @param currentIndexes ArrayList<Integer> that are the current indexes.
     * @param objectType Object that is the data context.
     * @return ArrayList<Integer> of re-arranged indexes according to the
     * order in which they will be set to the classifiers.
     */
    public static ArrayList<Integer> getDataIndexes(
            ArrayList<Integer> currentIndexes, Object objectType) {
        if (currentIndexes != null && currentIndexes.size() > 0) {
            if (objectType instanceof BOWDataSet) {
                BOWDataSet bowDSet = (BOWDataSet) objectType;
                ArrayList<BOWInstance> trueDataVect = new ArrayList<>(
                        currentIndexes.size());
                for (int i = 0; i < currentIndexes.size(); i++) {
                    trueDataVect.add((BOWInstance) (bowDSet.data.get(
                            currentIndexes.get(i))));
                }
                return getDataWithIndexes(trueDataVect, objectType,
                        currentIndexes);
            } else if (objectType instanceof DataSet) {
                DataSet dset = (DataSet) objectType;
                ArrayList<DataInstance> trueDataVect =
                        new ArrayList<>(currentIndexes.size());
                for (int i = 0; i < currentIndexes.size(); i++) {
                    trueDataVect.add(dset.data.get(currentIndexes.get(i)));
                }
                return getDataWithIndexes(trueDataVect, objectType,
                        currentIndexes);
            } else if (objectType instanceof DiscretizedDataSet) {
                DiscretizedDataSet dset = (DiscretizedDataSet) objectType;
                ArrayList<DiscretizedDataInstance> trueDataVect =
                        new ArrayList<>(currentIndexes.size());
                for (int i = 0; i < currentIndexes.size(); i++) {
                    trueDataVect.add(dset.data.get(currentIndexes.get(i)));
                }
                return getDataWithIndexes(trueDataVect, objectType,
                        currentIndexes);
            } else {
                return null;
            }
        } else {
            return null;
        }
    }

    /**
     * Gets the re-ordered indexes, the way that they will be set to the
     * classifiers.
     * 
     * @param data ArrayList of data instances.
     * @param dataType Object that is the data context.
     * @param currentIndexes ArrayList<Integer> that are the current indexes.
     * @return ArrayList<Integer> of re-arranged indexes according to the
     * order in which they will be set to the classifiers.
     */
    private static ArrayList<Integer> getDataWithIndexes(ArrayList data,
            Object dataType, ArrayList<Integer> currentIndexes) {
        ArrayList<Integer> permutatedIndexes = null;
        if (data != null && !data.isEmpty()) {
            permutatedIndexes = new ArrayList<>(currentIndexes.size());
            Category[] classes = null;
            int numClasses = 0;
            int currClass;
            if (data.get(0) instanceof BOWInstance) {
                BOWInstance instance;
                BOWDataSet bowDSet = (BOWDataSet) dataType;
                BOWDataSet bowDSetCopy = bowDSet.cloneDefinition();
                bowDSetCopy.data = data;
                for (int i = 0; i < data.size(); i++) {
                    if (data.get(i) != null) {
                        instance = (BOWInstance) (data.get(i));
                        currClass = instance.getCategory();
                        if (currClass > numClasses) {
                            numClasses = currClass;
                        }
                    }
                }
                numClasses = numClasses + 1;
                classes = new Category[numClasses];
                for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                    classes[cIndex] = new Category("number" + cIndex, 200,
                            bowDSet);
                    classes[cIndex].setDefinitionDataset(bowDSetCopy);
                }
                for (int i = 0; i < data.size(); i++) {
                    if (data.get(i) != null) {
                        instance = (BOWInstance) (data.get(i));
                        currClass = instance.getCategory();
                        classes[currClass].addInstance(currentIndexes.get(i));
                    }
                }
            } else if (data.get(0) instanceof DataInstance) {
                DataInstance instance;
                DataSet dset = (DataSet) dataType;
                DataSet dsetCopy = dset.cloneDefinition();
                dsetCopy.data = data;
                for (int i = 0; i < data.size(); i++) {
                    if (data.get(i) != null) {
                        instance = (DataInstance) (data.get(i));
                        currClass = instance.getCategory();
                        if (currClass > numClasses) {
                            numClasses = currClass;
                        }
                    }
                }
                numClasses = numClasses + 1;
                classes = new Category[numClasses];
                for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                    classes[cIndex] = new Category("number" + cIndex, 200,
                            dset);
                    classes[cIndex].setDefinitionDataset(dsetCopy);
                }
                for (int i = 0; i < data.size(); i++) {
                    if (data.get(i) != null) {
                        instance = (DataInstance) (data.get(i));
                        currClass = instance.getCategory();
                        classes[currClass].addInstance(currentIndexes.get(i));
                    }
                }
            } else if (data.get(0) instanceof DiscretizedDataInstance) {
                DiscretizedDataInstance instance;
                DiscretizedDataSet discDSet = (DiscretizedDataSet) dataType;
                DiscretizedDataSet discDSetCopy = discDSet.cloneDefinition();
                discDSetCopy.data = data;
                for (int i = 0; i < data.size(); i++) {
                    if (data.get(i) != null) {
                        instance = (DiscretizedDataInstance) (data.get(i));
                        currClass = instance.getCategory();
                        if (currClass > numClasses) {
                            numClasses = currClass;
                        }
                    }
                }
                numClasses = numClasses + 1;
                classes = new Category[numClasses];
                for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                    classes[cIndex] = new DiscreteCategory("number" + cIndex,
                            discDSetCopy, 200);
                }
                for (int i = 0; i < data.size(); i++) {
                    if (data.get(i) != null) {
                        instance = (DiscretizedDataInstance) (data.get(i));
                        currClass = instance.getCategory();
                        ((DiscreteCategory) classes[currClass]).
                                addInstanceIndex(currentIndexes.get(i));
                    }
                }
            }
            if (classes != null) {
                for (int cIndex = 0; cIndex < classes.length; cIndex++) {
                    permutatedIndexes.addAll(classes[cIndex].indexes);
                }
            }
        }
        return permutatedIndexes;
    }

    /**
     * Unlike the getDataIndexes, this doesn't output the actual pointer indexes
     * to places in the newly set DataSet it rather gives the permutation of the
     * original index array, so it is the 'intermediate' information, used to
     * re-sort some of the arrays in the original index ordering.
     *
     * @param currentIndexes ArrayList<Integer> that are the current indexes to
     * be set as the training data.
     * @param objectType Object that is the data context.
     * @return ArrayList<Integer> pointing to the locations of the original
     * indexes, when viewed from within the set training data in the
     * classifiers.
     */
    public static ArrayList<Integer> getIndexPermutation(
            ArrayList<Integer> currentIndexes, Object objectType) {
        if (currentIndexes != null && currentIndexes.size() > 0) {
            if (objectType instanceof BOWDataSet) {
                BOWDataSet bowDSet = (BOWDataSet) objectType;
                ArrayList<BOWInstance> trueDataVect =
                        new ArrayList<>(currentIndexes.size());
                for (int i = 0; i < currentIndexes.size(); i++) {
                    trueDataVect.add((BOWInstance) (bowDSet.data.get(
                            currentIndexes.get(i))));
                }
                return getIndexPermutationInner(trueDataVect, objectType);
            } else if (objectType instanceof DataSet) {
                DataSet dset = (DataSet) objectType;
                ArrayList<DataInstance> trueDataVect = new ArrayList<>(
                        currentIndexes.size());
                for (int i = 0; i < currentIndexes.size(); i++) {
                    trueDataVect.add(dset.data.get(currentIndexes.get(i)));
                }
                return getIndexPermutationInner(trueDataVect, objectType);
            } else if (objectType instanceof DiscretizedDataSet) {
                DiscretizedDataSet dset = (DiscretizedDataSet) objectType;
                ArrayList<DiscretizedDataInstance> trueDataVect =
                        new ArrayList<>(currentIndexes.size());
                for (int i = 0; i < currentIndexes.size(); i++) {
                    trueDataVect.add(dset.data.get(currentIndexes.get(i)));
                }
                return getIndexPermutationInner(trueDataVect, objectType);
            } else {
                return null;
            }
        } else {
            return null;
        }
    }

    /**
     * Unlike the getDataIndexes, this doesn't output the actual pointer indexes
     * to places in the newly set DataSet it rather gives the permutation of the
     * original index array, so it is the 'intermediate' information, used to
     * re-sort some of the arrays in the original index ordering.
     * 
     * @param data ArrayList containing the data instances.
     * @param dataType Object that is the data context.
     * @return ArrayList<Integer> pointing to the locations of the original
     * indexes, when viewed from within the set training data in the
     * classifiers.
     */
    private static ArrayList<Integer> getIndexPermutationInner(ArrayList data,
            Object dataType) {
        ArrayList<Integer> permutatedIndexes = null;
        if (data != null && !data.isEmpty()) {
            permutatedIndexes = new ArrayList<>(data.size());
            Category[] classes = null;
            int numClasses = 0;
            int currClass;
            if (data.get(0) instanceof BOWInstance) {
                BOWInstance instance;
                BOWDataSet bowDSet = (BOWDataSet) dataType;
                BOWDataSet bowDSetCopy = bowDSet.cloneDefinition();
                bowDSetCopy.data = data;
                for (int i = 0; i < data.size(); i++) {
                    if (data.get(i) != null) {
                        instance = (BOWInstance) (data.get(i));
                        currClass = instance.getCategory();
                        if (currClass > numClasses) {
                            numClasses = currClass;
                        }
                    }
                }
                numClasses = numClasses + 1;
                classes = new Category[numClasses];
                for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                    classes[cIndex] = new Category("number" + cIndex, 200,
                            bowDSet);
                    classes[cIndex].setDefinitionDataset(bowDSetCopy);
                }
                for (int i = 0; i < data.size(); i++) {
                    if (data.get(i) != null) {
                        instance = (BOWInstance) (data.get(i));
                        currClass = instance.getCategory();
                        classes[currClass].addInstance(i);
                    }
                }
            } else if (data.get(0) instanceof DataInstance) {
                DataInstance instance;
                DataSet dset = (DataSet) dataType;
                DataSet dsetCopy = dset.cloneDefinition();
                dsetCopy.data = data;
                for (int i = 0; i < data.size(); i++) {
                    if (data.get(i) != null) {
                        instance = (DataInstance) (data.get(i));
                        currClass = instance.getCategory();
                        if (currClass > numClasses) {
                            numClasses = currClass;
                        }
                    }
                }
                numClasses = numClasses + 1;
                classes = new Category[numClasses];
                for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                    classes[cIndex] = new Category("number" + cIndex, 200,
                            dset);
                    classes[cIndex].setDefinitionDataset(dsetCopy);
                }
                for (int i = 0; i < data.size(); i++) {
                    if (data.get(i) != null) {
                        instance = (DataInstance) (data.get(i));
                        currClass = instance.getCategory();
                        classes[currClass].addInstance(i);
                    }
                }
            } else if (data.get(0) instanceof DiscretizedDataInstance) {
                DiscretizedDataInstance instance;
                DiscretizedDataSet discDSet = (DiscretizedDataSet) dataType;
                DiscretizedDataSet discDSetCopy = discDSet.cloneDefinition();
                discDSetCopy.data = data;
                for (int i = 0; i < data.size(); i++) {
                    if (data.get(i) != null) {
                        instance = (DiscretizedDataInstance) (data.get(i));
                        currClass = instance.getCategory();
                        if (currClass > numClasses) {
                            numClasses = currClass;
                        }
                    }
                }
                numClasses = numClasses + 1;
                classes = new Category[numClasses];
                for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                    classes[cIndex] = new DiscreteCategory("number" + cIndex,
                            discDSetCopy, 200);
                }
                for (int i = 0; i < data.size(); i++) {
                    if (data.get(i) != null) {
                        instance = (DiscretizedDataInstance) (data.get(i));
                        currClass = instance.getCategory();
                        ((DiscreteCategory) classes[currClass]).
                                addInstance(i);
                    }
                }
            }
            if (classes != null) {
                for (int cIndex = 0; cIndex < classes.length; cIndex++) {
                    permutatedIndexes.addAll(classes[cIndex].indexes);
                }
            }
        }
        return permutatedIndexes;
    }
}
