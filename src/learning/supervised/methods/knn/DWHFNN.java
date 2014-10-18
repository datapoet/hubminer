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
package learning.supervised.methods.knn;

import learning.supervised.interfaces.AutomaticKFinderInterface;
import data.neighbors.NSFUserInterface;
import data.neighbors.NeighborSetFinder;
import data.representation.DataInstance;
import data.representation.DataSet;
import distances.primary.CombinedMetric;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import learning.supervised.Category;
import learning.supervised.Classifier;
import learning.supervised.evaluation.ValidateableInterface;
import learning.supervised.evaluation.cv.MultiCrossValidation;
import learning.supervised.interfaces.DistMatrixUserInterface;
import learning.supervised.interfaces.DistToPointsQueryUserInterface;
import learning.supervised.interfaces.NeighborPointsQueryUserInterface;
import preprocessing.instance_selection.InstanceSelector;

/**
 * This paper implements the dwh-FNN algorithm that was proposed in the
 * following paper: "Hubness-based Fuzzy Measures for High-dimensional
 * K-Nearest-Neighbor classification" at MLDM 2011. It is reverse-neighbor-based
 * extension of the classical FNN classifier.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class DWHFNN extends Classifier implements AutomaticKFinderInterface,
        DistMatrixUserInterface, NSFUserInterface,
        DistToPointsQueryUserInterface, NeighborPointsQueryUserInterface,
        Serializable {
    
    private static final long serialVersionUID = 1L;

    // The parameter that defines when the global rule should not be fired due
    // to little data, so a special routine for anti-hub handling is invoked.
    private int thetaCutoff = 0;
    // Distance weighting parameter.
    float mValue = 2;
    // Neighborhood size.
    private int k = 5;
    // Object that stores and calculates the kNN sets.
    private NeighborSetFinder nsf = null;
    // Training data to learn from.
    private DataSet trainingData = null;
    // Number of classes in the data.
    private int numClasses = 0;
    // Class-conditional neighbor occurrence frequencies.
    private float[][] classDataKNeighborRelation = null;
    // Prior class counts or probabilities.
    private float[] classPriors = null;
    // Laplace estimator for probability distribution smoothing.
    private float laplaceEstimator = 0.001f;
    // Neighbor occurrence frequencies of training data points.
    private int[] neighborOccurrenceFreqs = null;
    // Anti-hub handling special-case data structures.
    // Local estimate for low hubness without leading label value.
    private float[][] localClassDistribution = null;
    // Local estimate for low hubness with leading label value.
    private float[][] localFClassDistribution = null;
    // Local estimate for low hubness with just the label and Laplace estimates.
    private float[][] localCrispClassDistribution = null;
    // Global estimate for low hubness.
    private float[][] classToClassPriors = null;
    private float[][] distMat = null;
    // The local anti-hub vote estimation method.
    int localEstimateMethod = LABEL;
    public static final int GLOBAL = 0;
    public static final int LOCAL = 1;
    public static final int LOCALF = 2;
    public static final int LABEL = 3;

    /**
     * @param methodIndex Integer that indicates which estimation method to use.
     * Possible values include: DWHFNN.GLOBAL, DWHFNN.LOCAL, DWHFNN.LOCALF and
     * DWHFNN.LABEL.
     */
    public void setAntiHubVoteEstimateMethod(int methodIndex) {
        localEstimateMethod = methodIndex % 4;
    }

    @Override
    public void setDistMatrix(float[][] distMatrix) {
        this.distMat = distMatrix;
    }

    @Override
    public float[][] getDistMatrix() {
        return distMat;
    }
    boolean noRecalc = false;

    @Override
    public void noRecalcs() {
        noRecalc = true;
    }

    @Override
    public String getName() {
        return "dwh-FNN";
    }

    @Override
    public void findK(int kMin, int kMax) throws Exception {
        // TODO: improve the implementation to how it's done in HIKNN.
        DataSet dset = trainingData;
        // Get the k-nearest neighbor sets.
        NeighborSetFinder nsfLOU;
        if (distMat == null) {
            nsfLOU = new NeighborSetFinder(dset, getCombinedMetric());
            nsfLOU.calculateDistances();
        } else {
            nsfLOU = new NeighborSetFinder(dset, distMat, getCombinedMetric());
        }
        nsfLOU.calculateNeighborSets(kMax);
        // Here we will log the accuracy for each k value.
        float[] accuracyArray = new float[kMax - kMin + 1];
        float currMaxAcc = -1f;
        // The optimal neighborhood size.
        int currMaxK = 0;
        // The optimal cut-off point.
        int currMaxTheta = 0;
        // This variable will hold the index of the best anti-hub handling
        // method.
        int currMaxScheme = GLOBAL;
        int numElements = dset.size();
        float currMaxVote;
        int[][] kneighbors = nsfLOU.getKNeighbors();
        // Current accuracy.
        float currAcc;
        // Distances to the k-nearest neighbors.
        float[][] kDistances = nsfLOU.getKDistances();
        // Calculate the distance weights.
        float[][] distance_weightsAll = new float[numElements][kMax];
        float[] dwSumAll = new float[numElements];
        for (int index = 0; index < numElements; index++) {
            for (int i = 0; i < kMax; i++) {
                if (kDistances[index][i] != 0) {
                    distance_weightsAll[index][i] = 1f
                            / ((float) Math.pow(kDistances[index][i],
                            (2f / (mValue - 1f))));
                } else {
                    distance_weightsAll[index][i] = 10000f;
                }
                dwSumAll[index] += distance_weightsAll[index][i];
            }
        }
        classPriors = dset.getClassPriors();
        int kEstimate = Math.min(kMax, 10);
        localClassDistribution = new float[trainingData.size()][];
        localFClassDistribution = new float[trainingData.size()][];
        localCrispClassDistribution = new float[trainingData.size()][];
        int currClass = 0;
        float laplaceTotal = numClasses * laplaceEstimator;
        // Calculate all the anti-hub handling structures for all supported
        // vote estimation methods.
        for (int i = 0; i < trainingData.size(); i++) {
            localClassDistribution[i] = new float[numClasses];
            localFClassDistribution[i] = new float[numClasses];
            for (int j = 0; j < kEstimate; j++) {
                localClassDistribution[i][trainingData.data.get(
                        kneighbors[i][j]).getCategory()]++;
            }
            localClassDistribution[i][trainingData.data.get(i).getCategory()]++;
            for (int j = 0; j < numClasses; j++) {
                localClassDistribution[i][j] += laplaceEstimator;
                localClassDistribution[i][j] /= (kEstimate + 1 + laplaceTotal);
                if (trainingData.data.get(i).getCategory() == j) {
                    localFClassDistribution[i][j] = 0.51f + 0.49f
                            * localClassDistribution[i][j];
                } else {
                    localFClassDistribution[i][j] = 0.49f
                            * localClassDistribution[i][j];
                }
            }
            localCrispClassDistribution[i] = new float[numClasses];
            for (int j = 0; j < numClasses; j++) {
                if (trainingData.data.get(i).getCategory() == j) {
                    localCrispClassDistribution[i][j] = (1 + laplaceEstimator)
                            / (1 + (numClasses * laplaceEstimator));
                } else {
                    localCrispClassDistribution[i][j] = (laplaceEstimator)
                            / (1 + (numClasses * laplaceEstimator));
                }
            }
        }
        // This will hold the class-conditional neighbor occurrence frequencies
        // for all the k-values.
        float[][][] classDataKNeighborRelationAllK =
                new float[accuracyArray.length][numClasses][
                        trainingData.size()];
        // Neighbor occurrence frequencies for all the k-values.
        int[][] neighborOccFreqsAllK = new int[kMax][];
        // Class-to-class occurrences for all the k-values.
        float[][][] classToClassPriorsAllK =
                new float[accuracyArray.length][numClasses][numClasses];
        // Normalization factors for the class to class priors.
        float[][] classHubnessSumsAllK =
                new float[accuracyArray.length][numClasses];
        // Iterate over the neighborhood range.
        for (int kCurr = kMin; kCurr <= kMax; kCurr++) {
            nsfLOU.recalculateStatsForSmallerK(kCurr);
            neighborOccFreqsAllK[kCurr - 1] = Arrays.copyOf(
                    nsfLOU.getNeighborFrequencies(),
                    nsfLOU.getNeighborFrequencies().length);
            // Calculate the class-conditional and class-to-class occurrence
            // frequencies.
            for (int i = 0; i < trainingData.size(); i++) {
                currClass = trainingData.data.get(i).getCategory();
                classDataKNeighborRelationAllK[kCurr - kMin][currClass][i]++;
                for (int j = 0; j < kCurr; j++) {
                    classDataKNeighborRelationAllK[kCurr - kMin][currClass][
                            kneighbors[i][j]]++;
                    classToClassPriorsAllK[kCurr - kMin][
                            trainingData.data.get(
                            kneighbors[i][j]).getCategory()][currClass]++;
                    classHubnessSumsAllK[kCurr - kMin][
                            trainingData.data.get(
                            kneighbors[i][j]).getCategory()]++;
                }
            }
            // Perform smoothing with the Laplace estimator.
            laplaceTotal = numClasses * laplaceEstimator;
            for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                for (int j = 0; j < trainingData.size(); j++) {
                    classDataKNeighborRelationAllK[kCurr - kMin][cIndex][j] +=
                            laplaceEstimator;
                    classDataKNeighborRelationAllK[kCurr - kMin][cIndex][j] /=
                            (neighborOccFreqsAllK[kCurr - 1][j] + 1
                            + laplaceTotal);
                }
            }
            laplaceTotal = numClasses * laplaceEstimator;
            for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                for (int j = 0; j < numClasses; j++) {
                    classToClassPriorsAllK[kCurr - kMin][cIndex][j] +=
                            laplaceEstimator;
                    classToClassPriorsAllK[kCurr - kMin][cIndex][j] /=
                            (classHubnessSumsAllK[kCurr - kMin][cIndex]
                            + laplaceTotal);
                }
            }
        }
        // Iterate over all possible cut-offs for the anti-hub case.
        for (thetaCutoff = 0; thetaCutoff < 10; thetaCutoff++) {
            // Iterate over different neighborhood sizes in the range.
            for (int kInc = 0; kInc < accuracyArray.length; kInc++) {
                // The structures that keep track of the total votes in case
                // of different anti-hub handling strategies.
                // Votes in the global estimation case.
                float[][] currVoteCountsG = new float[numElements][numClasses];
                int[] currVoteLabelsG = new int[numElements];
                // Votes in the local estimation case.
                float[][] currVoteCountsL = new float[numElements][numClasses];
                int[] currVoteLabelsL = new int[numElements];
                // Votes in the FNN local estimation case.
                float[][] currVoteCountsLF = new float[numElements][numClasses];
                int[] currVoteLabelsLF = new int[numElements];
                // Votes in the crisp estimation case.
                float[][] currVoteCountsC = new float[numElements][numClasses];
                int[] currVoteLabelsC = new int[numElements];
                currAcc = 0;
                // Iterate over all the data points.
                // First check for the global anti-hub handling strategy.
                for (int index = 0; index < numElements; index++) {
                    currMaxVote = 0;
                    for (int classIndex = 0; classIndex < numClasses;
                            classIndex++) {
                        // Go through the current kNN set.
                        for (int kCurr = 0; kCurr <= kMin + kInc - 1; kCurr++) {
                            // Simulate voting.
                            if (neighborOccFreqsAllK[kMin + kInc - 1][
                                    kneighbors[index][kCurr]] > thetaCutoff) {
                                // The normal case.
                                currVoteCountsG[index][classIndex] +=
                                        classDataKNeighborRelationAllK[kInc][
                                        classIndex][kneighbors[index][kCurr]]
                                        * distance_weightsAll[index][
                                        kMin + kInc - 1] / dwSumAll[index];
                            } else {
                                // The special case.
                                currVoteCountsG[index][classIndex] +=
                                        classToClassPriorsAllK[kInc][
                                        classIndex][trainingData.data.get(
                                        kneighbors[index][kCurr]).
                                        getCategory()]
                                        * distance_weightsAll[index][
                                        kMin + kInc - 1] / dwSumAll[index];
                            }
                        }
                        if (currVoteCountsG[index][classIndex] > currMaxVote) {
                            currMaxVote = currVoteCountsG[index][classIndex];
                            currVoteLabelsG[index] = classIndex;
                        }
                    }
                    if (currVoteLabelsG[index]
                            == dset.data.get(index).getCategory()) {
                        // Increase the accuracy if correct.
                        currAcc++;
                    }
                }
                currAcc /= (float) numElements;
                accuracyArray[kInc] = currAcc;
                if (currMaxAcc < currAcc) {
                    currMaxAcc = currAcc;
                    currMaxK = kMin + kInc;
                    currMaxTheta = thetaCutoff;
                    currMaxScheme = GLOBAL;
                }
                // Now check the accuracy of the crisp anti-hub handling method.
                currAcc = 0;
                for (int index = 0; index < numElements; index++) {
                    currMaxVote = 0;
                    for (int classIndex = 0; classIndex < numClasses;
                            classIndex++) {
                        // Go through the current kNN set.
                        for (int kCurr = 0; kCurr <= kMin + kInc - 1; kCurr++) {
                            if (neighborOccFreqsAllK[kMin + kInc - 1][
                                    kneighbors[index][kCurr]] > thetaCutoff) {
                                // The normal case.
                                currVoteCountsC[index][classIndex] +=
                                        classDataKNeighborRelationAllK[kInc][
                                        classIndex][kneighbors[index][kCurr]]
                                        * distance_weightsAll[index][
                                        kMin + kInc - 1] / dwSumAll[index];
                            } else {
                                // The special anti-hub case.
                                currVoteCountsC[index][classIndex] +=
                                        localCrispClassDistribution[
                                        kneighbors[index][kCurr]][classIndex]
                                        * distance_weightsAll[index][
                                        kMin + kInc - 1] / dwSumAll[index];
                            }
                        }
                        if (currVoteCountsC[index][classIndex] > currMaxVote) {
                            currMaxVote = currVoteCountsC[index][classIndex];
                            currVoteLabelsC[index] = classIndex;
                        }
                    }
                    if (currVoteLabelsC[index]
                            == dset.data.get(index).getCategory()) {
                        // Increase the accuracy if correct.
                        currAcc++;
                    }
                }
                currAcc /= (float) numElements;
                accuracyArray[kInc] = currAcc;
                if (currMaxAcc < currAcc) {
                    currMaxAcc = currAcc;
                    currMaxK = kMin + kInc;
                    currMaxTheta = thetaCutoff;
                    currMaxScheme = LABEL;
                }
                // Now check the accuracy of the local anti-hub handling method.
                currAcc = 0;
                for (int index = 0; index < numElements; index++) {
                    currMaxVote = 0;
                    for (int classIndex = 0; classIndex < numClasses;
                            classIndex++) {
                        // Go through the current kNN set.
                        for (int kCurr = 0; kCurr <= kMin + kInc - 1; kCurr++) {
                            if (neighborOccFreqsAllK[kMin + kInc - 1][
                                    kneighbors[index][kCurr]] > thetaCutoff) {
                                // The normal case.
                                currVoteCountsLF[index][classIndex] +=
                                        classDataKNeighborRelationAllK[kInc][
                                        classIndex][kneighbors[index][kCurr]]
                                        * distance_weightsAll[index][
                                        kMin + kInc - 1] / dwSumAll[index];
                            } else {
                                // The special anti-hub case.
                                currVoteCountsLF[index][classIndex] +=
                                        localFClassDistribution[
                                        kneighbors[index][kCurr]][classIndex]
                                        * distance_weightsAll[index][
                                        kMin + kInc - 1] / dwSumAll[index];
                            }
                        }
                        if (currVoteCountsLF[index][classIndex] > currMaxVote) {
                            currMaxVote = currVoteCountsLF[index][classIndex];
                            currVoteLabelsLF[index] = classIndex;
                        }
                    }
                    if (currVoteLabelsLF[index]
                            == dset.data.get(index).getCategory()) {
                        // Increase the accuracy if correct.
                        currAcc++;
                    }
                }
                currAcc /= (float) numElements;
                accuracyArray[kInc] = currAcc;
                if (currMaxAcc < currAcc) {
                    currMaxAcc = currAcc;
                    currMaxK = kMin + kInc;
                    currMaxTheta = thetaCutoff;
                    currMaxScheme = LOCALF;
                }
                // Now test the local FNN-based anti-hub handling method.
                currAcc = 0;
                for (int index = 0; index < numElements; index++) {
                    currMaxVote = 0;
                    for (int classIndex = 0; classIndex < numClasses;
                            classIndex++) {
                        // Go through the current kNN set.
                        for (int kCurr = 0; kCurr <= kMin + kInc - 1; kCurr++) {
                            if (neighborOccFreqsAllK[kMin + kInc - 1][
                                    kneighbors[index][kCurr]] > thetaCutoff) {
                                // The normal case.
                                currVoteCountsL[index][classIndex] +=
                                        classDataKNeighborRelationAllK[kInc][
                                        classIndex][kneighbors[index][kCurr]]
                                        * distance_weightsAll[index][
                                        kMin + kInc - 1] / dwSumAll[index];
                            } else {
                                // The special anti-hub handling case.
                                currVoteCountsL[index][classIndex] +=
                                        localClassDistribution[
                                        kneighbors[index][kCurr]][classIndex]
                                        * distance_weightsAll[index][
                                        kMin + kInc - 1] / dwSumAll[index];
                            }
                        }
                        if (currVoteCountsL[index][classIndex] > currMaxVote) {
                            currMaxVote = currVoteCountsL[index][classIndex];
                            currVoteLabelsL[index] = classIndex;
                        }
                    }
                    if (currVoteLabelsL[index]
                            == dset.data.get(index).getCategory()) {
                        // Increase the accuracy if correct.
                        currAcc++;
                    }
                }
                currAcc /= (float) numElements;
                accuracyArray[kInc] = currAcc;
                if (currMaxAcc < currAcc) {
                    currMaxAcc = currAcc;
                    currMaxK = kMin + kInc;
                    currMaxTheta = thetaCutoff;
                    currMaxScheme = LOCAL;
                }
            }
        }
        // Set the best found parameters.
        thetaCutoff = currMaxTheta;
        k = currMaxK;
        localEstimateMethod = currMaxScheme;
        localClassDistribution = null;
        localFClassDistribution = null;
    }

    /**
     * The default constructor.
     */
    public DWHFNN() {
    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     */
    public DWHFNN(int k) {
        this.k = k;
    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     * @param cmet CombinedMetric object for distance calculations.
     */
    public DWHFNN(int k, CombinedMetric cmet) {
        this.k = k;
        setCombinedMetric(cmet);
    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     * @param cmet CombinedMetric object for distance calculations.
     * @param numClasses Integer that is the number of classes in the data.
     */
    public DWHFNN(int k, CombinedMetric cmet, int numClasses) {
        this.k = k;
        setCombinedMetric(cmet);
        this.numClasses = numClasses;

    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     * @param laplaceEstimator for probability distribution smoothing.
     */
    public DWHFNN(int k, float laplaceEstimator) {
        this.k = k;
        this.laplaceEstimator = laplaceEstimator;
    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     * @param laplaceEstimator for probability distribution smoothing.
     * @param cmet CombinedMetric object for distance calculations.
     */
    public DWHFNN(int k, float laplaceEstimator, CombinedMetric cmet) {
        this.k = k;
        this.laplaceEstimator = laplaceEstimator;
        setCombinedMetric(cmet);
    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     * @param laplaceEstimator for probability distribution smoothing.
     * @param cmet CombinedMetric object for distance calculations.
     * @param numClasses Integer that is the number of classes in the data.
     */
    public DWHFNN(int k, float laplaceEstimator, CombinedMetric cmet,
            int numClasses) {
        this.k = k;
        this.laplaceEstimator = laplaceEstimator;
        setCombinedMetric(cmet);
        this.numClasses = numClasses;
    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     * @param laplaceEstimator for probability distribution smoothing.
     * @param cmet CombinedMetric object for distance calculations.
     * @param numClasses Integer that is the number of classes in the data.
     * @param mValue Float that is the distance weighting parameter.
     */
    public DWHFNN(int k, float laplaceEstimator, CombinedMetric cmet,
            int numClasses, float mValue) {
        this.k = k;
        this.laplaceEstimator = laplaceEstimator;
        setCombinedMetric(cmet);
        this.numClasses = numClasses;
        this.mValue = mValue;
    }

    /**
     * Initialization.
     *
     * @param dset DataSet to train the model on.
     * @param numClasses Integer that is the number of classes in the data.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Integer that is the neighborhood size.
     */
    public DWHFNN(DataSet dset, int numClasses, CombinedMetric cmet, int k) {
        trainingData = dset;
        this.numClasses = numClasses;
        setCombinedMetric(cmet);
        this.k = k;
    }

    /**
     * Initialization.
     *
     * @param dset DataSet to train the model on.
     * @param numClasses Integer that is the number of classes in the data.
     * @param nsf NeighborSetFinder object for kNN set holding and calculations.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Integer that is the neighborhood size.
     */
    public DWHFNN(DataSet dset, int numClasses, NeighborSetFinder nsf,
            CombinedMetric cmet, int k) {
        trainingData = dset;
        this.numClasses = numClasses;
        this.nsf = nsf;
        setCombinedMetric(cmet);
        this.k = k;
    }

    /**
     * Initialization.
     *
     * @param categories Category[] of classes to train on.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Integer that is the neighborhood size.
     */
    public DWHFNN(Category[] categories, CombinedMetric cmet, int k) {
        int totalSize = 0;
        int indexFirstNonEmptyClass = -1;
        for (int cIndex = 0; cIndex < categories.length; cIndex++) {
            totalSize += categories[cIndex].size();
            if (indexFirstNonEmptyClass == -1
                    && categories[cIndex].size() > 0) {
                indexFirstNonEmptyClass = cIndex;
            }
        }
        // This is an internal data context, so the instances are not embedded.
        trainingData = new DataSet();
        trainingData.fAttrNames = categories[indexFirstNonEmptyClass].
                getInstance(0).getEmbeddingDataset().fAttrNames;
        trainingData.iAttrNames = categories[indexFirstNonEmptyClass].
                getInstance(0).getEmbeddingDataset().iAttrNames;
        trainingData.sAttrNames = categories[indexFirstNonEmptyClass].
                getInstance(0).getEmbeddingDataset().sAttrNames;
        trainingData.data = new ArrayList<>(totalSize);
        for (int cIndex = 0; cIndex < categories.length; cIndex++) {
            for (int j = 0; j < categories[cIndex].size(); j++) {
                categories[cIndex].getInstance(j).setCategory(cIndex);
                trainingData.addDataInstance(categories[cIndex].getInstance(j));
            }
        }
        setCombinedMetric(cmet);
        this.k = k;
        numClasses = trainingData.countCategories();
    }

    /**
     * @param laplaceEstimator Float value used for probability distribution
     * smoothing.
     */
    public void setLaplaceEstimator(float laplaceEstimator) {
        this.laplaceEstimator = laplaceEstimator;
    }

    @Override
    public void setClasses(Category[] categories) {
        int totalSize = 0;
        int indexFirstNonEmptyClass = -1;
        for (int cIndex = 0; cIndex < categories.length; cIndex++) {
            totalSize += categories[cIndex].size();
            if (indexFirstNonEmptyClass == -1
                    && categories[cIndex].size() > 0) {
                indexFirstNonEmptyClass = cIndex;
            }
        }
        // This is an internal data context, so the instances are not embedded.
        trainingData = new DataSet();
        trainingData.fAttrNames = categories[indexFirstNonEmptyClass].
                getInstance(0).getEmbeddingDataset().fAttrNames;
        trainingData.iAttrNames = categories[indexFirstNonEmptyClass].
                getInstance(0).getEmbeddingDataset().iAttrNames;
        trainingData.sAttrNames = categories[indexFirstNonEmptyClass].
                getInstance(0).getEmbeddingDataset().sAttrNames;
        trainingData.data = new ArrayList<>(totalSize);
        for (int cIndex = 0; cIndex < categories.length; cIndex++) {
            for (int j = 0; j < categories[cIndex].size(); j++) {
                categories[cIndex].getInstance(j).setCategory(cIndex);
                trainingData.addDataInstance(categories[cIndex].getInstance(j));
            }
        }
        numClasses = trainingData.countCategories();
    }

    /**
     * @param trainingData DataSet object to train the models on.
     */
    public void setTrainingSet(DataSet trainingData) {
        this.trainingData = trainingData;
    }

    /**
     * @param numClasses Integer that is the number of classes in the data.
     */
    public void setNumClasses(int numClasses) {
        this.numClasses = numClasses;
    }

    /**
     * @return Integer that is the number of classes in the data.
     */
    public int getNumClasses() {
        return numClasses;
    }

    /**
     * @return Integer that is the neighborhood size.
     */
    public int getK() {
        return k;
    }

    /**
     * @param k Integer that is the neighborhood size.
     */
    public void setK(int k) {
        this.k = k;
    }

    @Override
    public void setNSF(NeighborSetFinder nsf) {
        this.nsf = nsf;
    }

    @Override
    public NeighborSetFinder getNSF() {
        return nsf;
    }

    /**
     * Calculate the neighbor sets, if not already calculated.
     *
     * @throws Exception
     */
    public void calculateNeighborSets() throws Exception {
        if (distMat == null) {
            nsf = new NeighborSetFinder(trainingData, getCombinedMetric());
            nsf.calculateDistances();
        } else {
            nsf = new NeighborSetFinder(trainingData, distMat,
                    getCombinedMetric());
        }
        nsf.calculateNeighborSets(k);
    }

    @Override
    public ValidateableInterface copyConfiguration() {
        DWHFNN result = new DWHFNN(k, laplaceEstimator, getCombinedMetric(),
                numClasses, mValue);
        result.noRecalc = noRecalc;
        result.localEstimateMethod = localEstimateMethod;
        result.thetaCutoff = thetaCutoff;
        return result;
    }

    @Override
    public void train() throws Exception {
        if (k <= 0) {
            // Setting an invalid k-value is a signal for automatically
            // searching for the optimal value in the low-value range.
            findK(1, 20);
        }
        if (nsf == null) {
            calculateNeighborSets();
        }
        // Calculate class priors.
        classPriors = trainingData.getClassPriors();
        // These structures are used in the special anti-hub vote estimation
        // case.
        localClassDistribution = new float[trainingData.size()][];
        localFClassDistribution = new float[trainingData.size()][];
        localCrispClassDistribution = new float[trainingData.size()][];
        neighborOccurrenceFreqs = nsf.getNeighborFrequencies();
        // Get the kNN sets.
        int[][] kneighbors = nsf.getKNeighbors();
        classDataKNeighborRelation = new float[numClasses][trainingData.size()];
        int currClass;
        // The smoothing total.
        float laplaceTotal;
        // Get the distance matrix on the training data.
        float[][] distMatrix = nsf.getDistances();
        classToClassPriors = new float[numClasses][numClasses];
        // Normalization factors.
        float[] classHubnessSums = new float[numClasses];
        // Iterate over the training data.
        for (int i = 0; i < trainingData.size(); i++) {
            currClass = trainingData.data.get(i).getCategory();
            classDataKNeighborRelation[currClass][i]++;
            // Calculate the class-conditional neighbor occurrence frequencies.
            for (int kIndex = 0; kIndex < k; kIndex++) {
                classDataKNeighborRelation[currClass][kneighbors[i][kIndex]]++;
                classToClassPriors[trainingData.data.get(kneighbors[i][kIndex]).
                        getCategory()][currClass]++;
                classHubnessSums[trainingData.data.get(kneighbors[i][kIndex]).
                        getCategory()]++;
            }
            if (neighborOccurrenceFreqs[i] <= thetaCutoff) {
                // The special anti-hub handling case.
                laplaceTotal = 10 * laplaceEstimator;
                localCrispClassDistribution[i] = new float[numClasses];
                for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                    if (trainingData.data.get(i).getCategory() == cIndex) {
                        localCrispClassDistribution[i][cIndex] =
                                (1 + laplaceEstimator) / (1
                                + (numClasses * laplaceEstimator));
                    } else {
                        localCrispClassDistribution[i][cIndex] =
                                (laplaceEstimator) / (1
                                + (numClasses * laplaceEstimator));
                    }
                }
                // Calculate the estimators for different operating modes.
                if (localEstimateMethod != LABEL) {
                    if (k >= 10) {
                        localClassDistribution[i] = new float[numClasses];
                        localFClassDistribution[i] = new float[numClasses];
                        localCrispClassDistribution[i] = new float[numClasses];
                        for (int kIndex = 0; kIndex < 10; kIndex++) {
                            localClassDistribution[i][trainingData.data.get(
                                    kneighbors[i][kIndex]).getCategory()]++;
                        }
                        localClassDistribution[i][trainingData.data.get(i).
                                getCategory()]++;
                        for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                            localClassDistribution[i][cIndex] +=
                                    laplaceEstimator;
                            localClassDistribution[i][cIndex] /=
                                    (10 + 1 + laplaceTotal);
                            if (trainingData.data.get(i).getCategory()
                                    == cIndex) {
                                localFClassDistribution[i][cIndex] =
                                        0.51f + 0.49f
                                        * localClassDistribution[i][cIndex];
                            } else {
                                localFClassDistribution[i][cIndex] = 0.49f
                                        * localClassDistribution[i][cIndex];
                            }
                        }
                    } else {
                        // In this case, we need to extend the provided kNN sets
                        // with additional elements to obtain proper estimates.
                        int[] lneighbors = new int[10];
                        float[] kDistances = new float[10];
                        for (int kIndex = 0; kIndex < k; kIndex++) {
                            lneighbors[kIndex] = kneighbors[i][kIndex];
                            kDistances[kIndex] = nsf.getKDistances()[i][kIndex];
                        }
                        int kcurrLen = k;
                        int l;
                        float currDist;
                        boolean insertable;
                        for (int kIndex = 0; kIndex < trainingData.size();
                                kIndex++) {
                            if (kIndex != i) {
                                currDist = getDistanceForElements(
                                        distMatrix, i, kIndex);
                                if (kcurrLen == 10) {
                                    if (currDist < kDistances[kcurrLen - 1]) {
                                        // Search to see where to insert.
                                        insertable = true;
                                        for (int index = 0; index < kcurrLen;
                                                index++) {
                                            if (kIndex == lneighbors[index]) {
                                                insertable = false;
                                                break;
                                            }
                                        }
                                        if (insertable) {
                                            l = kcurrLen - 1;
                                            while ((l >= 1) && currDist
                                                    < kDistances[l - 1]) {
                                                kDistances[l] =
                                                        kDistances[l - 1];
                                                lneighbors[l] =
                                                        lneighbors[l - 1];
                                                l--;
                                            }
                                            kDistances[l] = currDist;
                                            lneighbors[l] = kIndex;
                                        }
                                    }
                                } else {
                                    if (currDist < kDistances[kcurrLen - 1]) {
                                        // Search to see where to insert.
                                        insertable = true;
                                        for (int index = 0; index < kcurrLen;
                                                index++) {
                                            if (kIndex == lneighbors[index]) {
                                                insertable = false;
                                                break;
                                            }
                                        }
                                        if (insertable) {
                                            l = kcurrLen - 1;
                                            kDistances[kcurrLen] =
                                                    kDistances[kcurrLen - 1];
                                            lneighbors[kcurrLen] =
                                                    lneighbors[kcurrLen - 1];
                                            while ((l >= 1) && currDist
                                                    < kDistances[l - 1]) {
                                                kDistances[l] =
                                                        kDistances[l - 1];
                                                lneighbors[l] =
                                                        lneighbors[l - 1];
                                                l--;
                                            }
                                            kDistances[l] = currDist;
                                            lneighbors[l] = kIndex;
                                            kcurrLen++;
                                        }
                                    } else {
                                        kDistances[kcurrLen] = currDist;
                                        lneighbors[kcurrLen] = kIndex;
                                        kcurrLen++;
                                    }
                                }
                            }
                        }
                        localClassDistribution[i] = new float[numClasses];
                        localFClassDistribution[i] = new float[numClasses];
                        for (int kIndex = 0; kIndex < 10; kIndex++) {
                            localClassDistribution[i][trainingData.data.get(
                                    lneighbors[kIndex]).getCategory()]++;
                        }
                        localClassDistribution[i][trainingData.data.get(i).
                                getCategory()]++;
                        for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                            localClassDistribution[i][cIndex] +=
                                    laplaceEstimator;
                            localClassDistribution[i][cIndex] /= (10 + 1
                                    + laplaceTotal);
                            if (trainingData.data.get(i).getCategory()
                                    == cIndex) {
                                localFClassDistribution[i][cIndex] = 0.51f
                                        + 0.49f
                                        * localClassDistribution[i][cIndex];
                            } else {
                                localFClassDistribution[i][cIndex] = 0.49f
                                        * localClassDistribution[i][cIndex];
                            }
                        }
                    }
                }
            }
        }
        // Smoothing and normalization..
        laplaceTotal = numClasses * laplaceEstimator;
        for (int cIndex = 0; cIndex < numClasses; cIndex++) {
            for (int j = 0; j < trainingData.size(); j++) {
                classDataKNeighborRelation[cIndex][j] += laplaceEstimator;
                classDataKNeighborRelation[cIndex][j] /=
                        (neighborOccurrenceFreqs[j] + 1 + laplaceTotal);
            }
        }
        laplaceTotal = numClasses * laplaceEstimator;
        for (int cFirst = 0; cFirst < numClasses; cFirst++) {
            for (int cSecond = 0; cSecond < numClasses; cSecond++) {
                classToClassPriors[cFirst][cSecond] += laplaceEstimator;
                classToClassPriors[cFirst][cSecond] /=
                        (classHubnessSums[cFirst] + laplaceTotal);
            }
        }
    }

    @Override
    public void trainOnReducedData(InstanceSelector reducer) throws Exception {
        // Get the index permutation in the experimental framework.
        ArrayList<Integer> indexPermutation = MultiCrossValidation.
                getIndexPermutation(reducer.getPrototypeIndexes(),
                reducer.getOriginalDataSet());
        int kDr = reducer.getNeighborhoodSize();
        int[] neighbOccFreqsProto = reducer.getPrototypeHubness();
        // The prototypes are set as the training dataset during
        // cross-validation.
        classPriors = trainingData.getClassPriors();
        localClassDistribution = new float[trainingData.size()][];
        localFClassDistribution = new float[trainingData.size()][];
        localCrispClassDistribution = new float[trainingData.size()][];
        neighborOccurrenceFreqs = new int[neighbOccFreqsProto.length];
        for (int i = 0; i < neighborOccurrenceFreqs.length; i++) {
            neighborOccurrenceFreqs[i] =
                    neighbOccFreqsProto[indexPermutation.get(i)];
        }
        float[] classHubnessSums = new float[numClasses];
        // The indexes in the kNN sets are permuted to the ones in DataSet that
        // was provided to the class constructor, so the index permutation
        // must be used.
        int[][] kneighbors = reducer.getProtoNeighborSets();
        classDataKNeighborRelation = new float[numClasses][trainingData.size()];
        float[][] classDataKNeighborRelationTemp =
                reducer.getClassDataNeighborRelationforFuzzy(numClasses,
                laplaceEstimator);
        // Get the class-conditional prototype occurrence frequencies.
        for (int c = 0; c < numClasses; c++) {
            for (int i = 0; i < neighborOccurrenceFreqs.length; i++) {
                classDataKNeighborRelation[c][i] =
                        classDataKNeighborRelationTemp[c][
                        indexPermutation.get(i)];
            }
        }
        int currClass;
        classToClassPriors = new float[numClasses][numClasses];
        float laplaceTotal;
        int permutedIndex;
        // Calculate the anti-hub vote estimation structures.
        for (int i = 0; i < trainingData.size(); i++) {
            permutedIndex = indexPermutation.get(i);
            laplaceTotal = kDr * laplaceEstimator;
            currClass = trainingData.getLabelOf(permutedIndex);
            for (int kIndex = 0; kIndex < kDr; kIndex++) {
                classToClassPriors[reducer.getPrototypeLabel(
                        kneighbors[permutedIndex][kIndex])][currClass]++;
                classHubnessSums[
                        reducer.getPrototypeLabel(kneighbors[
                        permutedIndex][kIndex])]++;
            }
            if (neighborOccurrenceFreqs[i] <= thetaCutoff) {
                localCrispClassDistribution[i] = new float[numClasses];
                for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                    if (trainingData.data.get(i).getCategory() == cIndex) {
                        localCrispClassDistribution[i][cIndex] =
                                (1 + laplaceEstimator) / (1
                                + (numClasses * laplaceEstimator));
                    } else {
                        localCrispClassDistribution[i][cIndex] =
                                (laplaceEstimator) / (1 + (numClasses
                                * laplaceEstimator));
                    }
                }
                localClassDistribution[i] = new float[numClasses];
                localFClassDistribution[i] = new float[numClasses];
                for (int kIndex = 0; kIndex < kDr; kIndex++) {
                    localClassDistribution[i][reducer.getPrototypeLabel(
                            kneighbors[permutedIndex][kIndex])]++;
                }
                localClassDistribution[i][currClass]++;
                for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                    localClassDistribution[i][cIndex] += laplaceEstimator;
                    localClassDistribution[i][cIndex] /= (kDr + 1
                            + laplaceTotal);
                    if (currClass == cIndex) {
                        localFClassDistribution[i][cIndex] = 0.51f + 0.49f
                                * localClassDistribution[i][cIndex];
                    } else {
                        localFClassDistribution[i][cIndex] = 0.49f
                                * localClassDistribution[i][cIndex];
                    }
                }
            }
        }
        // Normalization and smoothing.
        laplaceTotal = numClasses * laplaceEstimator;
        for (int cFirst = 0; cFirst < numClasses; cFirst++) {
            for (int cSecond = 0; cSecond < numClasses; cSecond++) {
                classToClassPriors[cFirst][cSecond] += laplaceEstimator;
                classToClassPriors[cFirst][cSecond] /=
                        (classHubnessSums[cFirst] + laplaceTotal);
            }
        }
    }

    /**
     * Gets the proper distance from the distance matrix.
     *
     * @param distMatrix float[][] that is the upper triangular distance matrix.
     * @param i Integer that is the index of the first instance.
     * @param j Integer that is the index of the second instance.
     * @return Float that is the distance between the two instances.
     */
    private static float getDistanceForElements(float[][] distMatrix,
            int i, int j) {
        if (j > i) {
            return distMatrix[i][j - i - 1];
        } else if (i > j) {
            return distMatrix[j][i - j - 1];
        } else {
            return 0;
        }
    }

    @Override
    public int classify(DataInstance instance) throws Exception {
        float[] classProbs = classifyProbabilistically(instance);
        float maxProb = 0;
        int result = 0;
        for (int cIndex = 0; cIndex < numClasses; cIndex++) {
            if (classProbs[cIndex] > maxProb) {
                maxProb = classProbs[cIndex];
                result = cIndex;
            }
        }
        return result;
    }

    @Override
    public float[] classifyProbabilistically(DataInstance instance)
            throws Exception {
        CombinedMetric cmet = getCombinedMetric();
        //first calculate nearest neighbors from training set
        float[] kDistances = new float[k];
        for (int kIndex = 0; kIndex < k; kIndex++) {
            kDistances[kIndex] = Float.MAX_VALUE;
        }
        int[] kNeighbors = new int[k];
        float currDistance;
        int index;
        // Calculate the kNN set.
        for (int i = 0; i < trainingData.size(); i++) {
            currDistance = cmet.dist(trainingData.data.get(i), instance);
            if (currDistance < kDistances[k - 1]) {
                // Insertion.
                index = k - 1;
                while (index > 0 && kDistances[index - 1] > currDistance) {
                    kDistances[index] = kDistances[index - 1];
                    kNeighbors[index] = kNeighbors[index - 1];
                    index--;
                }
                kDistances[index] = currDistance;
                kNeighbors[index] = i;
            }
        }
        // Calculate the distance weights.
        float[] distanceWeights = new float[k];
        float dwSum = 0;
        for (int kIndex = 0; kIndex < k; kIndex++) {
            if (kDistances[kIndex] != 0) {
                distanceWeights[kIndex] = 1f /
                        ((float) Math.pow(kDistances[kIndex],
                        (2f / (mValue - 1f))));
            } else {
                distanceWeights[kIndex] = 10000f;
            }
            dwSum += distanceWeights[kIndex];
        }
        float[] classProbEstimates = new float[numClasses];
        // Perform the voting.
        for (int kIndex = 0; kIndex < k; kIndex++) {
            if (neighborOccurrenceFreqs[kNeighbors[kIndex]] > thetaCutoff) {
                // The normal case.
                for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                    classProbEstimates[cIndex] +=
                            classDataKNeighborRelation[cIndex][
                            kNeighbors[kIndex]]
                            * distanceWeights[kIndex] / dwSum;
                }
            } else {
                // The special anti-hub handling case.
                for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                    switch (localEstimateMethod) {
                        case LOCAL:
                            classProbEstimates[cIndex] +=
                                    localClassDistribution[kNeighbors[kIndex]][
                                    cIndex] * distanceWeights[kIndex] / dwSum;
                            break;
                        case LOCALF:
                            classProbEstimates[cIndex] +=
                                    localFClassDistribution[
                                    kNeighbors[kIndex]][cIndex]
                                    * distanceWeights[kIndex] / dwSum;
                            break;
                        case GLOBAL:
                            classProbEstimates[cIndex] +=
                                    classToClassPriors[cIndex][
                                    trainingData.data.get(kNeighbors[kIndex]).
                                    getCategory()] * distanceWeights[kIndex]
                                    / dwSum;
                            break;
                        case LABEL:
                            classProbEstimates[cIndex] +=
                                    localCrispClassDistribution[
                                    kNeighbors[kIndex]][cIndex]
                                    * distanceWeights[kIndex] / dwSum;
                            break;
                        default:
                            classProbEstimates[cIndex] +=
                                    localFClassDistribution[kNeighbors[kIndex]][
                                    cIndex] * distanceWeights[kIndex] / dwSum;
                    }
                }
            }
        }
        // Normalization.
        float probTotal = 0;
        for (int cIndex = 0; cIndex < numClasses; cIndex++) {
            probTotal += classProbEstimates[cIndex];
        }
        if (probTotal > 0) {
            for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                classProbEstimates[cIndex] /= probTotal;
            }
        } else {
            classProbEstimates = Arrays.copyOf(classPriors, numClasses);
        }
        return classProbEstimates;
    }

    @Override
    public float[] classifyProbabilistically(DataInstance instance,
            float[] distToTraining) throws Exception {
        // Calculate the kNN set.
        float[] kDistances = new float[k];
        for (int kIndex = 0; kIndex < k; kIndex++) {
            kDistances[kIndex] = Float.MAX_VALUE;
        }
        int[] kNeighbors = new int[k];
        float currDistance;
        int index;
        for (int i = 0; i < trainingData.size(); i++) {
            currDistance = distToTraining[i];
            if (currDistance < kDistances[k - 1]) {
                // Insertion.
                index = k - 1;
                while (index > 0 && kDistances[index - 1] > currDistance) {
                    kDistances[index] = kDistances[index - 1];
                    kNeighbors[index] = kNeighbors[index - 1];
                    index--;
                }
                kDistances[index] = currDistance;
                kNeighbors[index] = i;
            }
        }
        // Calculate the distance weights.
        float[] distanceWeights = new float[k];
        float dwSum = 0;
        for (int kIndex = 0; kIndex < k; kIndex++) {
            if (kDistances[kIndex] != 0) {
                distanceWeights[kIndex] = 1f
                        / ((float) Math.pow(kDistances[kIndex],
                        (2f / (mValue - 1f))));
            } else {
                distanceWeights[kIndex] = 10000f;
            }
            dwSum += distanceWeights[kIndex];
        }
        float[] classProbEstimates = new float[numClasses];
        for (int kIndex = 0; kIndex < k; kIndex++) {
            if (neighborOccurrenceFreqs[kNeighbors[kIndex]] > thetaCutoff) {
                // The normal case.
                for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                    classProbEstimates[cIndex] += classDataKNeighborRelation[
                            cIndex][kNeighbors[kIndex]]
                            * distanceWeights[kIndex] / dwSum;
                }
            } else {
                // The special anti-hub handling case.
                for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                    switch (localEstimateMethod) {
                        case LOCAL:
                            classProbEstimates[cIndex] +=
                                    localClassDistribution[
                                    kNeighbors[kIndex]][cIndex]
                                    * distanceWeights[kIndex] / dwSum;
                            break;
                        case LOCALF:
                            classProbEstimates[cIndex] +=
                                    localFClassDistribution[
                                    kNeighbors[kIndex]][cIndex]
                                    * distanceWeights[kIndex] / dwSum;
                            break;
                        case GLOBAL:
                            classProbEstimates[cIndex] +=
                                    classToClassPriors[cIndex][
                                    trainingData.data.get(kNeighbors[kIndex]).
                                    getCategory()] * distanceWeights[kIndex]
                                    / dwSum;
                            break;
                        case LABEL:
                            classProbEstimates[cIndex] +=
                                    localCrispClassDistribution[
                                    kNeighbors[kIndex]][cIndex]
                                    * distanceWeights[kIndex] / dwSum;
                            break;
                        default:
                            classProbEstimates[cIndex] +=
                                    localFClassDistribution[
                                    kNeighbors[kIndex]][cIndex]
                                    * distanceWeights[kIndex] / dwSum;
                    }
                }
            }
        }
        // Normalization.
        float probTotal = 0;
        for (int cIndex = 0; cIndex < numClasses; cIndex++) {
            probTotal += classProbEstimates[cIndex];
        }
        if (probTotal > 0) {
            for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                classProbEstimates[cIndex] /= probTotal;
            }
        } else {
            classProbEstimates = Arrays.copyOf(classPriors, numClasses);
        }
        return classProbEstimates;
    }

    @Override
    public int classify(DataInstance instance, float[] distToTraining)
            throws Exception {
        float[] classProbs = classifyProbabilistically(instance,
                distToTraining);
        float maxProb = 0;
        int result = 0;
        for (int cIndex = 0; cIndex < numClasses; cIndex++) {
            if (classProbs[cIndex] > maxProb) {
                maxProb = classProbs[cIndex];
                result = cIndex;
            }
        }
        return result;
    }

    @Override
    public int classify(DataInstance instance, float[] distToTraining,
            int[] trNeighbors) throws Exception {
        float[] classProbs = classifyProbabilistically(instance, distToTraining,
                trNeighbors);
        float maxProb = 0;
        int result = 0;
        for (int cIndex = 0; cIndex < numClasses; cIndex++) {
            if (classProbs[cIndex] > maxProb) {
                maxProb = classProbs[cIndex];
                result = cIndex;
            }
        }
        return result;
    }

    @Override
    public float[] classifyProbabilistically(DataInstance instance,
            float[] distToTraining, int[] trNeighbors) throws Exception {
        float[] classProbEstimates = new float[numClasses];
        float[] distanceWeights = new float[k];
        float dwSum = 0;
        for (int kIndex = 0; kIndex < k; kIndex++) {
            if (distToTraining[trNeighbors[kIndex]] != 0) {
                distanceWeights[kIndex] = 1f /
                        ((float) Math.pow(distToTraining[trNeighbors[kIndex]],
                        (2f / (mValue - 1f))));
            } else {
                distanceWeights[kIndex] = 10000f;
            }
            dwSum += distanceWeights[kIndex];
        }
        // Perform the voting.
        for (int kIndex = 0; kIndex < k; kIndex++) {
            if (neighborOccurrenceFreqs[trNeighbors[kIndex]] > thetaCutoff) {
                // The normal case.
                for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                    classProbEstimates[cIndex] +=
                            classDataKNeighborRelation[cIndex][
                            trNeighbors[kIndex]] * distanceWeights[kIndex]
                            / dwSum;
                }
            } else {
                // The special anti-hub handling case.
                for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                    switch (localEstimateMethod) {
                        case LOCAL:
                            classProbEstimates[cIndex] +=
                                    localClassDistribution[
                                    trNeighbors[kIndex]][cIndex]
                                    * distanceWeights[kIndex] / dwSum;
                            break;
                        case LOCALF:
                            classProbEstimates[cIndex] +=
                                    localFClassDistribution[
                                    trNeighbors[kIndex]][cIndex]
                                    * distanceWeights[kIndex] / dwSum;
                            break;
                        case GLOBAL:
                            classProbEstimates[cIndex] +=
                                    classToClassPriors[cIndex][
                                    trainingData.data.get(
                                    trNeighbors[kIndex]).getCategory()]
                                    * distanceWeights[kIndex] / dwSum;
                            break;
                        case LABEL:
                            classProbEstimates[cIndex] +=
                                    localCrispClassDistribution[
                                    trNeighbors[kIndex]][cIndex]
                                    * distanceWeights[kIndex] / dwSum;
                            break;
                        default:
                            classProbEstimates[cIndex] +=
                                    localFClassDistribution[
                                    trNeighbors[kIndex]][cIndex]
                                    * distanceWeights[kIndex] / dwSum;
                    }
                }
            }
        }
        // Normalization.
        float probTotal = 0;
        for (int cIndex = 0; cIndex < numClasses; cIndex++) {
            probTotal += classProbEstimates[cIndex];
        }
        if (probTotal > 0) {
            for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                classProbEstimates[cIndex] /= probTotal;
            }
        } else {
            classProbEstimates = Arrays.copyOf(classPriors, numClasses);
        }
        return classProbEstimates;
    }
    
    @Override
    public int getNeighborhoodSize() {
        return k;
    }
}
