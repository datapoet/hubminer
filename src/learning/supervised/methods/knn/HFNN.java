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
import java.util.Random;
import learning.supervised.Category;
import learning.supervised.Classifier;
import learning.supervised.evaluation.ValidateableInterface;
import learning.supervised.evaluation.cv.MultiCrossValidation;
import learning.supervised.interfaces.DistMatrixUserInterface;
import learning.supervised.interfaces.DistToPointsQueryUserInterface;
import learning.supervised.interfaces.NeighborPointsQueryUserInterface;
import preprocessing.instance_selection.InstanceSelector;

/**
 * This paper implements the h-FNN algorithm that was proposed in the following
 * paper: "Hubness-based Fuzzy Measures for High-dimensional K-Nearest-Neighbor
 * classification" at MLDM 2011. It is reverse-neighbor-based extension of the
 * classical FNN classifier.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class HFNN extends Classifier implements AutomaticKFinderInterface,
        DistMatrixUserInterface, NSFUserInterface,
        DistToPointsQueryUserInterface, NeighborPointsQueryUserInterface,
        Serializable {
    
    private static final long serialVersionUID = 1L;

    // The cut-off parameter for the special anti-hub handling routine.
    private int thetaCutoff = 0;
    // Neighborhood size.
    private int k = 5;
    // kNN holding object.
    private NeighborSetFinder nsf = null;
    // Data used for model training.
    private DataSet trainingData = null;
    // Number of classes.
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
    private int localEstimateMethod = GLOBAL;
    public static final int GLOBAL = 0;
    public static final int LOCAL = 1;
    public static final int LOCALF = 2;
    public static final int LABEL = 3;
    private boolean noRecalc = false;

    @Override
    public void noRecalcs() {
        noRecalc = true;
    }

    @Override
    public void setDistMatrix(float[][] distMatrix) {
        this.distMat = distMatrix;
    }

    @Override
    public float[][] getDistMatrix() {
        return distMat;
    }

    @Override
    public String getName() {
        return "h-FNN";
    }

    @Override
    public void findK(int kMin, int kMax) throws Exception {
        DataSet dset = trainingData;
        NeighborSetFinder nsfLOU = null;
        if (distMat == null) {
            // If no distances were provided, calculate them.
            nsfLOU = new NeighborSetFinder(dset, getCombinedMetric());
            nsfLOU.calculateDistances();
        } else {
            nsfLOU = new NeighborSetFinder(dset, distMat, getCombinedMetric());
        }
        // Calculate the kNN sets.
        nsfLOU.calculateNeighborSets(kMax);
        // This array stores the accuracies for different k-values.
        float[] accuracyArray = new float[kMax - kMin + 1];
        float currMaxAcc = -1f;
        int currMaxK = 0;
        int currMaxTheta = 0;
        int currMaxAntiHubScheme = GLOBAL;
        int numElements = dset.size();
        float currMaxVote;
        int[][] kneighbors = nsfLOU.getKNeighbors();
        float currAccuracy;
        // Calculate class priors.
        classPriors = trainingData.getClassPriors();
        // Neighborhood size used for the fuzzy estimate.
        int kEstimate = Math.min(kMax, 10);
        // Distributions for different anti-hub handling schemes.
        localClassDistribution = new float[trainingData.size()][];
        localFClassDistribution = new float[trainingData.size()][];
        localCrispClassDistribution = new float[trainingData.size()][];
        float laplaceTotal = numClasses * laplaceEstimator;
        for (int i = 0; i < trainingData.size(); i++) {
            localClassDistribution[i] = new float[numClasses];
            localFClassDistribution[i] = new float[numClasses];
            // Get the local class distributions in the kNN sets.
            for (int kIndex = 0; kIndex < kEstimate; kIndex++) {
                localClassDistribution[i][trainingData.data.get(
                        kneighbors[i][kIndex]).getCategory()]++;
            }
            localClassDistribution[i][
                    trainingData.data.get(i).getCategory()]++;
            // Include the label information and perform some smoothing.
            for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                localClassDistribution[i][cIndex] += laplaceEstimator;
                localClassDistribution[i][cIndex] /= (kEstimate + 1
                        + laplaceTotal);
                if (trainingData.data.get(i).getCategory() == cIndex) {
                    localFClassDistribution[i][cIndex] = 0.51f
                            + 0.49f * localClassDistribution[i][cIndex];
                } else {
                    localFClassDistribution[i][cIndex] = 0.49f
                            * localClassDistribution[i][cIndex];
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
        // Initialize the class-conditional neighbor occurrence frequencies.
        float[][][] classDataKNeighborRelationAllK =
                new float[accuracyArray.length][numClasses][
                        trainingData.size()];
        int[][] neighbOccFreqsAllK = new int[kMax][];
        float[][][] classToClassPriorsAllK =
                new float[accuracyArray.length][numClasses][numClasses];
        float[][] classHubnessSumsAllK =
                new float[accuracyArray.length][numClasses];
        // Iterate over the k-range.
        for (int kCurrent = kMin; kCurrent <= kMax; kCurrent++) {
            nsfLOU.recalculateStatsForSmallerK(kCurrent);
            neighbOccFreqsAllK[kCurrent - 1] = Arrays.copyOf(
                    nsfLOU.getNeighborFrequencies(),
                    nsfLOU.getNeighborFrequencies().length);
            // Calculate the class-conditional neighbor occurrence frequencies.
            for (int i = 0; i < trainingData.size(); i++) {
                int currClass = trainingData.data.get(i).getCategory();
                classDataKNeighborRelationAllK[kCurrent - kMin][currClass][i]++;
                for (int kIndex = 0; kIndex < kCurrent; kIndex++) {
                    classDataKNeighborRelationAllK[kCurrent - kMin][currClass][
                            kneighbors[i][kIndex]]++;
                    classToClassPriorsAllK[kCurrent - kMin][
                            trainingData.data.get(kneighbors[i][kIndex]).
                            getCategory()][currClass]++;
                    classHubnessSumsAllK[kCurrent - kMin][
                            trainingData.data.get(kneighbors[i][kIndex]).
                            getCategory()]++;
                }
            }
            // Normalize and perform smoothing.
            laplaceTotal = numClasses * laplaceEstimator;
            for (int i = 0; i < numClasses; i++) {
                for (int j = 0; j < trainingData.size(); j++) {
                    classDataKNeighborRelationAllK[kCurrent - kMin][i][j] +=
                            laplaceEstimator;
                    classDataKNeighborRelationAllK[kCurrent - kMin][i][j] /=
                            (neighbOccFreqsAllK[kCurrent - 1][j] + 1
                            + laplaceTotal);
                }
            }
            laplaceTotal = numClasses * laplaceEstimator;
            for (int i = 0; i < numClasses; i++) {
                for (int j = 0; j < numClasses; j++) {
                    classToClassPriorsAllK[kCurrent - kMin][i][j] +=
                            laplaceEstimator;
                    classToClassPriorsAllK[kCurrent - kMin][i][j] /=
                            (classHubnessSumsAllK[kCurrent - kMin][i]
                            + laplaceTotal);
                }
            }
        }
        // A coin is flipped if the votes are equal.
        Random randa = new Random();
        // Iterate over different thresholds for anti-hub routine invocation.
        for (thetaCutoff = 0; thetaCutoff < 10; thetaCutoff++) {
            for (int kInc = 0; kInc < accuracyArray.length; kInc++) {
                // Incrementally count the votes and remember the labels from
                // past iterations.
                float[][] currVoteCountsG = new float[numElements][numClasses];
                int[] currVoteLabelsG = new int[numElements];
                float[][] currVoteCountsL = new float[numElements][numClasses];
                int[] currVoteLabelsL = new int[numElements];
                float[][] currVoteCountsLF = new float[numElements][numClasses];
                int[] currVoteLabelsLF = new int[numElements];
                float[][] currVoteCountsC = new float[numElements][numClasses];
                int[] currVoteLabelsC = new int[numElements];
                // Find the accuracy of the global estimation method.
                currAccuracy = 0;
                for (int index = 0; index < numElements; index++) {
                    currMaxVote = 0;
                    for (int classIndex = 0; classIndex < numClasses;
                            classIndex++) {
                        for (int kCurr = 0; kCurr <= kMin + kInc - 1; kCurr++) {
                            if (neighbOccFreqsAllK[kMin + kInc - 1][
                                    kneighbors[index][kCurr]] > thetaCutoff) {
                                // The normal case.
                                currVoteCountsG[index][classIndex] +=
                                        classDataKNeighborRelationAllK[kInc][
                                        classIndex][kneighbors[index][kCurr]];
                            } else {
                                // The special anti-hub case.
                                currVoteCountsG[index][classIndex] +=
                                        classToClassPriorsAllK[kInc][
                                        classIndex][trainingData.data.get(
                                        kneighbors[index][kCurr]).
                                        getCategory()];
                            }
                        }
                        if (currVoteCountsG[index][classIndex] > currMaxVote) {
                            currMaxVote = currVoteCountsG[index][classIndex];
                            currVoteLabelsG[index] = classIndex;
                        } else if (currVoteCountsG[index][classIndex]
                                == currMaxVote && randa.nextFloat() < 0.5f) {
                            currMaxVote = currVoteCountsG[index][classIndex];
                            currVoteLabelsG[index] = classIndex;
                        }
                    }
                    if (currVoteLabelsG[index] == dset.data.get(index).
                            getCategory()) {
                        // If correct, increase the accuracy.
                        currAccuracy++;
                    }
                }
                currAccuracy /= (float) numElements;
                accuracyArray[kInc] = currAccuracy;
                // Update the current best parameter configuration.
                if (currMaxAcc < currAccuracy) {
                    currMaxAcc = currAccuracy;
                    currMaxK = kMin + kInc;
                    currMaxTheta = thetaCutoff;
                    currMaxAntiHubScheme = GLOBAL;
                }

                // Estimate the accuracy of the crisp anti-hub handling scheme.
                currAccuracy = 0;
                for (int index = 0; index < numElements; index++) {
                    currMaxVote = 0;
                    for (int classIndex = 0; classIndex < numClasses;
                            classIndex++) {
                        for (int kCurr = 0; kCurr <= kMin + kInc - 1; kCurr++) {
                            if (neighbOccFreqsAllK[kMin + kInc - 1][
                                    kneighbors[index][kCurr]] > thetaCutoff) {
                                // The normal case.
                                currVoteCountsC[index][classIndex] +=
                                        classDataKNeighborRelationAllK[kInc][
                                        classIndex][kneighbors[index][kCurr]];
                            } else {
                                // The special anti-hub handling case.
                                currVoteCountsC[index][classIndex] +=
                                        localCrispClassDistribution[
                                        kneighbors[index][kCurr]][classIndex];
                            }
                        }
                        // Update the vote.
                        if (currVoteCountsC[index][classIndex] > currMaxVote) {
                            currMaxVote = currVoteCountsC[index][classIndex];
                            currVoteLabelsC[index] = classIndex;
                        } else if (currVoteCountsG[index][classIndex]
                                == currMaxVote && randa.nextFloat() < 0.5f) {
                            currMaxVote = currVoteCountsG[index][classIndex];
                            currVoteLabelsG[index] = classIndex;
                        }
                    }
                    if (currVoteLabelsC[index]
                            == dset.data.get(index).getCategory()) {
                        // If correct, increase the accuracy.
                        currAccuracy++;
                    }
                }
                currAccuracy /= (float) numElements;
                accuracyArray[kInc] = currAccuracy;
                // Update the current optimal parameter configuration.
                if (currMaxAcc < currAccuracy) {
                    currMaxAcc = currAccuracy;
                    currMaxK = kMin + kInc;
                    currMaxTheta = thetaCutoff;
                    currMaxAntiHubScheme = LABEL;
                }
                // Estimate the accuracy of the method from FNN for estimating
                // the fuzziness of anti-hub points.
                currAccuracy = 0;
                for (int index = 0; index < numElements; index++) {
                    currMaxVote = 0;
                    for (int classIndex = 0; classIndex < numClasses;
                            classIndex++) {
                        for (int kCurr = 0; kCurr <= kMin + kInc - 1; kCurr++) {
                            if (neighbOccFreqsAllK[kMin + kInc - 1][
                                    kneighbors[index][kCurr]] > thetaCutoff) {
                                // The normal case.
                                currVoteCountsLF[index][classIndex] +=
                                        classDataKNeighborRelationAllK[kInc][
                                        classIndex][kneighbors[index][kCurr]];
                            } else {
                                // The special anti-hub handling case.
                                currVoteCountsLF[index][classIndex] +=
                                        localFClassDistribution[
                                        kneighbors[index][kCurr]][classIndex];
                            }
                        }
                        // Update the vote.
                        if (currVoteCountsLF[index][classIndex] > currMaxVote) {
                            currMaxVote = currVoteCountsLF[index][classIndex];
                            currVoteLabelsLF[index] = classIndex;
                        } else if (currVoteCountsG[index][classIndex]
                                == currMaxVote && randa.nextFloat() < 0.5f) {
                            currMaxVote = currVoteCountsG[index][classIndex];
                            currVoteLabelsG[index] = classIndex;
                        }
                    }
                    if (currVoteLabelsLF[index]
                            == dset.data.get(index).getCategory()) {
                        // If correct, increase the accuracy.
                        currAccuracy++;
                    }
                }
                currAccuracy /= (float) numElements;
                accuracyArray[kInc] = currAccuracy;
                // Update the optimal parameter configuration.
                if (currMaxAcc < currAccuracy) {
                    currMaxAcc = currAccuracy;
                    currMaxK = kMin + kInc;
                    currMaxTheta = thetaCutoff;
                    currMaxAntiHubScheme = LOCALF;
                }
                // Estimate the accuracy of the local estimate for handling
                // anti-hubs.
                currAccuracy = 0;
                for (int index = 0; index < numElements; index++) {
                    currMaxVote = 0;
                    for (int classIndex = 0; classIndex < numClasses;
                            classIndex++) {
                        for (int kCurr = 0; kCurr <= kMin + kInc - 1; kCurr++) {
                            if (neighbOccFreqsAllK[kMin + kInc - 1][
                                    kneighbors[index][kCurr]] > thetaCutoff) {
                                // The normal case.
                                currVoteCountsL[index][classIndex] +=
                                        classDataKNeighborRelationAllK[kInc][
                                        classIndex][kneighbors[index][kCurr]];
                            } else {
                                // The special anti-hub handling case.
                                currVoteCountsL[index][classIndex] +=
                                        localClassDistribution[
                                        kneighbors[index][kCurr]][classIndex];
                            }
                        }
                        // Update the vote.
                        if (currVoteCountsL[index][classIndex] > currMaxVote) {
                            currMaxVote = currVoteCountsL[index][classIndex];
                            currVoteLabelsL[index] = classIndex;
                        } else if (currVoteCountsG[index][classIndex]
                                == currMaxVote && randa.nextFloat() < 0.5f) {
                            currMaxVote = currVoteCountsG[index][classIndex];
                            currVoteLabelsG[index] = classIndex;
                        }
                    }
                    if (currVoteLabelsL[index]
                            == dset.data.get(index).getCategory()) {
                        // If correct, increase the accuracy.
                        currAccuracy++;
                    }
                }
                // Update the optimal parameter configuration.
                currAccuracy /= (float) numElements;
                accuracyArray[kInc] = currAccuracy;
                if (currMaxAcc < currAccuracy) {
                    currMaxAcc = currAccuracy;
                    currMaxK = kMin + kInc;
                    currMaxTheta = thetaCutoff;
                    currMaxAntiHubScheme = LOCAL;
                }
            }
        }
        // Set the optimal configuration.
        thetaCutoff = currMaxTheta;
        k = currMaxK;
        localEstimateMethod = currMaxAntiHubScheme;
        // Resert the estimation arrays.
        localClassDistribution = null;
        localFClassDistribution = null;
    }

    /**
     * The default constructor.
     */
    public HFNN() {
    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     */
    public HFNN(int k) {
        this.k = k;
    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     * @param cmet CombinedMetric object for distance calculations.
     */
    public HFNN(int k, CombinedMetric cmet) {
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
    public HFNN(int k, CombinedMetric cmet, int numClasses) {
        this.k = k;
        setCombinedMetric(cmet);
        this.numClasses = numClasses;

    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     * @param laplaceEstimator Float that is the Laplace estimator for
     * smoothing.
     */
    public HFNN(int k, float laplaceEstimator) {
        this.k = k;
        this.laplaceEstimator = laplaceEstimator;
    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     * @param laplaceEstimator Float that is the Laplace estimator for
     * smoothing.
     * @param cmet CombinedMetric object for distance calculations.
     */
    public HFNN(int k, float laplaceEstimator, CombinedMetric cmet) {
        this.k = k;
        this.laplaceEstimator = laplaceEstimator;
        setCombinedMetric(cmet);
    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     * @param laplaceEstimator Float that is the Laplace estimator for
     * smoothing.
     * @param cmet CombinedMetric object for distance calculations.
     * @param numClasses Integer that is the number of classes in the data.
     */
    public HFNN(int k, float laplaceEstimator, CombinedMetric cmet,
            int numClasses) {
        this.k = k;
        this.laplaceEstimator = laplaceEstimator;
        setCombinedMetric(cmet);
        this.numClasses = numClasses;
    }

    /**
     * Initialization.
     *
     * @param dset DataSet object that is the training data.
     * @param numClasses Integer that is the number of classes in the data.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Integer that is the neighborhood size.
     */
    public HFNN(DataSet dset, int numClasses, CombinedMetric cmet, int k) {
        trainingData = dset;
        this.numClasses = numClasses;
        setCombinedMetric(cmet);
        this.k = k;
    }

    /**
     * Initialization.
     *
     * @param dset DataSet object that is the training data.
     * @param numClasses Integer that is the number of classes in the data.
     * @param nsf NeighborSetFinder object for kNN sets.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Integer that is the neighborhood size.
     */
    public HFNN(DataSet dset, int numClasses, NeighborSetFinder nsf,
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
     * @param categories Category[] representing the training data.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Integer that is the neighborhood size.
     */
    public HFNN(Category[] categories, CombinedMetric cmet, int k) {
        int totalSize = 0;
        int indexFirstNonEmptyClass = -1;
        for (int i = 0; i < categories.length; i++) {
            totalSize += categories[i].size();
            if (indexFirstNonEmptyClass == -1 && categories[i].size() > 0) {
                indexFirstNonEmptyClass = i;
            }
        }
        // Instances won't be embedded in the internal data context.
        trainingData = new DataSet();
        trainingData.fAttrNames = categories[indexFirstNonEmptyClass].
                getInstance(0).getEmbeddingDataset().fAttrNames;
        trainingData.iAttrNames = categories[indexFirstNonEmptyClass].
                getInstance(0).getEmbeddingDataset().iAttrNames;
        trainingData.sAttrNames = categories[indexFirstNonEmptyClass].
                getInstance(0).getEmbeddingDataset().sAttrNames;
        trainingData.data = new ArrayList<>(totalSize);
        for (int cIndex = 0; cIndex < categories.length; cIndex++) {
            for (int i = 0; i < categories[cIndex].size();
                    i++) {
                categories[cIndex].getInstance(i).setCategory(cIndex);
                trainingData.addDataInstance(categories[cIndex].getInstance(
                        i));
            }
        }
        setCombinedMetric(cmet);
        this.k = k;
        numClasses = trainingData.countCategories();
    }

    /**
     * @param laplaceEstimator Float that is the Laplace estimator for
     * smoothing.
     */
    public void setLaplaceEstimator(float laplaceEstimator) {
        this.laplaceEstimator = laplaceEstimator;
    }

    @Override
    public void setClasses(Category[] categories) {
        int totalSize = 0;
        int indexFirstNonEmptyClass = -1;
        for (int i = 0; i < categories.length; i++) {
            totalSize += categories[i].size();
            if (indexFirstNonEmptyClass == -1 && categories[i].size() > 0) {
                indexFirstNonEmptyClass = i;
            }
        }
        // Instances won't be embedded in the internal data context.
        trainingData = new DataSet();
        trainingData.fAttrNames = categories[indexFirstNonEmptyClass].
                getInstance(0).getEmbeddingDataset().fAttrNames;
        trainingData.iAttrNames = categories[indexFirstNonEmptyClass].
                getInstance(0).getEmbeddingDataset().iAttrNames;
        trainingData.sAttrNames = categories[indexFirstNonEmptyClass].
                getInstance(0).getEmbeddingDataset().sAttrNames;
        trainingData.data = new ArrayList<>(totalSize);
        for (int cIndex = 0; cIndex < categories.length; cIndex++) {
            for (int i = 0; i < categories[cIndex].size();
                    i++) {
                categories[cIndex].getInstance(i).setCategory(cIndex);
                trainingData.addDataInstance(categories[cIndex].getInstance(
                        i));
            }
        }
        numClasses = trainingData.countCategories();
    }

    /**
     * @param trainingData DataSet object that is the training data.
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
     * @return Integer that is the neighborhood size to use for kNN
     * calculations.
     */
    public int getK() {
        return k;
    }

    /**
     * @param k Integer that is the neighborhood size to use for kNN
     * calculations.
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
     * Calculate the neighbor sets if not already provided.
     *
     * @throws Exception
     */
    private void calculateNeighborSets() throws Exception {
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
        HFNN classifierCopy = new HFNN(k, laplaceEstimator,
                getCombinedMetric(), numClasses);
        classifierCopy.noRecalc = noRecalc;
        classifierCopy.localEstimateMethod = localEstimateMethod;
        classifierCopy.thetaCutoff = thetaCutoff;
        return classifierCopy;
    }

    @Override
    public void train() throws Exception {
        if (k <= 0) {
            // If invalid neighborhood size was provided, try default automatic
            // search for the best k-value in the low-value range.
            findK(1, 20);
        }
        if (nsf == null) {
            // If the kNN sets were not provided, calculate them.
            calculateNeighborSets();
        }
        // Calculate the class distribution.
        classPriors = trainingData.getClassPriors();
        // Initialize the anti-hub vote estimation arrays.
        localClassDistribution = new float[trainingData.size()][];
        localFClassDistribution = new float[trainingData.size()][];
        localCrispClassDistribution = new float[trainingData.size()][];
        // Get the neighbor occurrence frequencies.
        neighborOccurrenceFreqs = nsf.getNeighborFrequencies();
        // Fetch the kNN sets.
        int[][] kneighbors = nsf.getKNeighbors();
        // Initialize the class-conditional occurrence counts.
        classDataKNeighborRelation = new float[numClasses][trainingData.size()];
        int currClass;
        // For smoothing.
        float laplaceTotal;
        // Fetch the distance matrix.
        float[][] distMatrix = nsf.getDistances();
        // Initialize class to class priors and hubness.
        classToClassPriors = new float[numClasses][numClasses];
        float[] classHubnessSums = new float[numClasses];
        // Iterate over all data points.
        for (int i = 0; i < trainingData.size(); i++) {
            currClass = trainingData.data.get(i).getCategory();
            // Each element is placed in its own kNN set as 0-th neighbor by
            // default.
            classDataKNeighborRelation[currClass][i]++;
            // Increment the occurrence counts.
            for (int kIndex = 0; kIndex < k; kIndex++) {
                classDataKNeighborRelation[currClass][kneighbors[i][kIndex]]++;
                classToClassPriors[trainingData.data.get(kneighbors[i][kIndex]).
                        getCategory()][currClass]++;
                classHubnessSums[trainingData.data.get(kneighbors[i][kIndex]).
                        getCategory()]++;
            }
            if (neighborOccurrenceFreqs[i] < thetaCutoff) {
                // The anti-hub handling case.
                laplaceTotal = 10 * laplaceEstimator;
                localCrispClassDistribution[i] = new float[numClasses];
                for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                    if (trainingData.data.get(i).getCategory() == cIndex) {
                        localCrispClassDistribution[i][cIndex] =
                                (1 + laplaceEstimator)
                                / (1 + (numClasses * laplaceEstimator));
                    } else {
                        localCrispClassDistribution[i][cIndex] =
                                (laplaceEstimator)
                                / (1 + (numClasses * laplaceEstimator));
                    }
                }
                if (k >= 10) {
                    // The neighborhood size is already big enough for all the
                    // estimates to take into account.
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
                        if (trainingData.data.get(i).getCategory() == cIndex) {
                            localFClassDistribution[i][cIndex] = 0.51f
                                    + 0.49f * localClassDistribution[i][cIndex];
                        } else {
                            localFClassDistribution[i][cIndex] = 0.49f
                                    * localClassDistribution[i][cIndex];
                        }
                    }
                } else {
                    // In this case, some additional calculations are in order.
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
                    // Calculate the additional k-neighbors all the way to k=10.
                    for (int j = 0; j < trainingData.size(); j++) {
                        if (j != i) {
                            currDist = getDistanceForElements(distMatrix, i, j);
                            if (kcurrLen == 10) {
                                if (currDist < kDistances[kcurrLen - 1]) {
                                    // Search to see where to insert.
                                    insertable = true;
                                    for (int index = 0; index < kcurrLen;
                                            index++) {
                                        if (j == lneighbors[index]) {
                                            insertable = false;
                                            break;
                                        }
                                    }
                                    if (insertable) {
                                        l = kcurrLen - 1;
                                        while ((l >= 1) && currDist
                                                < kDistances[l - 1]) {
                                            kDistances[l] = kDistances[l - 1];
                                            lneighbors[l] = lneighbors[l - 1];
                                            l--;
                                        }
                                        kDistances[l] = currDist;
                                        lneighbors[l] = j;
                                    }
                                }
                            } else {
                                if (currDist < kDistances[kcurrLen - 1]) {
                                    // Search to see where to insert.
                                    insertable = true;
                                    for (int index = 0; index < kcurrLen;
                                            index++) {
                                        if (j == lneighbors[index]) {
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
                                            kDistances[l] = kDistances[l - 1];
                                            lneighbors[l] = lneighbors[l - 1];
                                            l--;
                                        }
                                        kDistances[l] = currDist;
                                        lneighbors[l] = j;
                                        kcurrLen++;
                                    }
                                } else {
                                    kDistances[kcurrLen] = currDist;
                                    lneighbors[kcurrLen] = j;
                                    kcurrLen++;
                                }
                            }
                        }
                    }
                    // Now calculate all the anti-hub vote estimates.
                    localClassDistribution[i] = new float[numClasses];
                    localFClassDistribution[i] = new float[numClasses];
                    for (int kIndex = 0; kIndex < 10; kIndex++) {
                        localClassDistribution[i][trainingData.data.get(
                                lneighbors[kIndex]).getCategory()]++;
                    }
                    localClassDistribution[i][trainingData.data.get(i).
                            getCategory()]++;
                    for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                        localClassDistribution[i][cIndex] += laplaceEstimator;
                        localClassDistribution[i][cIndex] /=
                                (10 + 1 + laplaceTotal);
                        if (trainingData.data.get(i).getCategory() == cIndex) {
                            localFClassDistribution[i][cIndex] = 0.51f
                                    + 0.49f * localClassDistribution[i][cIndex];
                        } else {
                            localFClassDistribution[i][cIndex] = 0.49f
                                    * localClassDistribution[i][cIndex];
                        }
                    }
                }
            }
        }
        // Normalize and perform some smoothing.
        laplaceTotal = numClasses * laplaceEstimator;
        for (int i = 0; i < numClasses; i++) {
            for (int j = 0; j < trainingData.size(); j++) {
                classDataKNeighborRelation[i][j] += laplaceEstimator;
                classDataKNeighborRelation[i][j] /=
                        (neighborOccurrenceFreqs[j] + 1 + laplaceTotal);
            }
        }
        laplaceTotal = numClasses * laplaceEstimator;
        for (int i = 0; i < numClasses; i++) {
            for (int j = 0; j < numClasses; j++) {
                classToClassPriors[i][j] += laplaceEstimator;
                classToClassPriors[i][j] /=
                        (classHubnessSums[i] + laplaceTotal);
            }
        }
    }

    @Override
    public void trainOnReducedData(InstanceSelector reducer) throws Exception {
        // Get the index permutation for the indexes of the reduced dataset.
        ArrayList<Integer> indexPermutation = MultiCrossValidation.
                getIndexPermutation(reducer.getPrototypeIndexes(),
                reducer.getOriginalDataSet());
        int[] protoOccFreqs = reducer.getPrototypeHubness();
        int kDr = reducer.getNeighborhoodSize();
        // The prototypes are set as the training set.
        classPriors = trainingData.getClassPriors();
        // Initialize the anti-hub vote estimation structures.
        localClassDistribution = new float[trainingData.size()][];
        localFClassDistribution = new float[trainingData.size()][];
        localCrispClassDistribution = new float[trainingData.size()][];
        // Set the prototype occurrence frequencies, while taking the index
        // permutation into account.
        neighborOccurrenceFreqs = new int[protoOccFreqs.length];
        for (int i = 0; i < neighborOccurrenceFreqs.length; i++) {
            neighborOccurrenceFreqs[i] = protoOccFreqs[indexPermutation.get(i)];
        }
        // Get the kNN sets with the selected prototypes.
        int[][] kneighbors = reducer.getProtoNeighborSets();
        classDataKNeighborRelation = new float[numClasses][trainingData.size()];
        // Extract the class-conditional prototype occurrence frequencies for
        // training, based on the index permutation.
        float[][] classProtoKNeighborRelation = reducer.
                getClassDataNeighborRelationforFuzzy(numClasses,
                laplaceEstimator);
        for (int cIndex = 0; cIndex < numClasses; cIndex++) {
            for (int i = 0; i < neighborOccurrenceFreqs.length; i++) {
                classDataKNeighborRelation[cIndex][i] =
                        classProtoKNeighborRelation[cIndex][
                        indexPermutation.get(i)];
            }
        }
        int currClass;
        classToClassPriors = reducer.calculateClassToClassPriorsFuzzy();
        float laplaceTotal;
        int permutedIndex;
        // Iterate over all examples.
        for (int i = 0; i < trainingData.size(); i++) {
            permutedIndex = indexPermutation.get(i);
            laplaceTotal = kDr * laplaceEstimator;
            currClass = trainingData.getLabelOf(permutedIndex);
            if (neighborOccurrenceFreqs[i] <= thetaCutoff) {
                // The anti-hub case.
                localCrispClassDistribution[i] = new float[numClasses];
                for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                    if (trainingData.data.get(i).getCategory() == cIndex) {
                        localCrispClassDistribution[i][cIndex] =
                                (1 + laplaceEstimator)
                                / (1 + (numClasses * laplaceEstimator));
                    } else {
                        localCrispClassDistribution[i][cIndex] =
                                (laplaceEstimator)
                                / (1 + (numClasses * laplaceEstimator));
                    }
                }
                localClassDistribution[i] = new float[numClasses];
                localFClassDistribution[i] = new float[numClasses];
                for (int kIndex = 0; kIndex < kDr; kIndex++) {
                    localClassDistribution[i][
                            reducer.getPrototypeLabel(
                            kneighbors[permutedIndex][kIndex])]++;
                }
                localClassDistribution[i][currClass]++;
                for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                    localClassDistribution[i][cIndex] += laplaceEstimator;
                    localClassDistribution[i][cIndex] /=
                            (kDr + 1 + laplaceTotal);
                    if (currClass == cIndex) {
                        localFClassDistribution[i][cIndex] = 0.51f
                                + 0.49f * localClassDistribution[i][cIndex];
                    } else {
                        localFClassDistribution[i][cIndex] = 0.49f
                                * localClassDistribution[i][cIndex];
                    }
                }
            }
        }
    }

    /**
     * @param distMatrix
     * @param i Integer that is the index of the first example.
     * @param j Integer that is the index of the second example.
     * @return Float that is the distance between the specified indexes in the
     * upper triangular distance matrix.
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
        // Find the k-nearest neighbors.
        float[] kDistances = new float[k];
        for (int kIndex = 0; kIndex < k; kIndex++) {
            kDistances[kIndex] = Float.MAX_VALUE;
        }
        int[] kNeighbors = new int[k];
        float currDistance;
        int index;
        // Iterate over the data.
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
        // Perform the voting.
        float[] classProbEstimates = new float[numClasses];
        for (int kIndex = 0; kIndex < k; kIndex++) {
            if (neighborOccurrenceFreqs[kNeighbors[kIndex]] > thetaCutoff) {
                // The normal case.
                for (int j = 0; j < numClasses; j++) {
                    classProbEstimates[j] +=
                            classDataKNeighborRelation[j][kNeighbors[kIndex]];
                }
            } else {
                // The special anti-hub handling case.
                for (int j = 0; j < numClasses; j++) {
                    switch (localEstimateMethod) {
                        case LOCAL:
                            classProbEstimates[j] +=
                                    localClassDistribution[
                                    kNeighbors[kIndex]][j];
                            break;
                        case LOCALF:
                            classProbEstimates[j] +=
                                    localFClassDistribution[
                                    kNeighbors[kIndex]][j];
                            break;
                        case GLOBAL:
                            classProbEstimates[j] +=
                                    classToClassPriors[j][
                                    trainingData.data.get(
                                    kNeighbors[kIndex]).getCategory()];
                            break;
                        case LABEL:
                            classProbEstimates[j] +=
                                    localCrispClassDistribution[
                                    kNeighbors[kIndex]][j];
                            break;
                        default:
                            classProbEstimates[j] +=
                                    localFClassDistribution[
                                    kNeighbors[kIndex]][j];
                    }
                }
            }
        }
        // Normalize.
        float probTotal = 0;
        for (int i = 0; i < numClasses; i++) {
            probTotal += classProbEstimates[i];
        }
        if (probTotal > 0) {
            for (int i = 0; i < numClasses; i++) {
                classProbEstimates[i] /= probTotal;
            }
        } else {
            classProbEstimates = Arrays.copyOf(classPriors, numClasses);
        }
        return classProbEstimates;
    }

    @Override
    public float[] classifyProbabilistically(DataInstance instance,
            float[] distToTraining) throws Exception {
        // Find the k-nearest neighbors.
        float[] kDistances = new float[k];
        for (int i = 0; i < k; i++) {
            kDistances[i] = Float.MAX_VALUE;
        }
        int[] kNeighbors = new int[k];
        float currDistance;
        int index;
        // Iterate overa all examples.
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
        // Perform the voting.
        float[] classProbEstimates = new float[numClasses];
        for (int kIndex = 0; kIndex < k; kIndex++) {
            if (neighborOccurrenceFreqs[kNeighbors[kIndex]] > thetaCutoff) {
                // The normal case.
                for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                    classProbEstimates[cIndex] +=
                            classDataKNeighborRelation[cIndex][
                            kNeighbors[kIndex]];
                }
            } else {
                // The special anti-hub handling case.
                for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                    switch (localEstimateMethod) {
                        case LOCAL:
                            classProbEstimates[cIndex] +=
                                    localClassDistribution[
                                    kNeighbors[kIndex]][cIndex];
                            break;
                        case LOCALF:
                            classProbEstimates[cIndex] +=
                                    localFClassDistribution[
                                    kNeighbors[kIndex]][cIndex];
                            break;
                        case GLOBAL:
                            classProbEstimates[cIndex] +=
                                    classToClassPriors[cIndex][
                                    trainingData.data.get(
                                    kNeighbors[kIndex]).getCategory()];
                            break;
                        case LABEL:
                            classProbEstimates[cIndex] +=
                                    localCrispClassDistribution[
                                    kNeighbors[kIndex]][cIndex];
                            break;
                        default:
                            classProbEstimates[cIndex] +=
                                    localFClassDistribution[
                                    kNeighbors[kIndex]][cIndex];
                    }
                }
            }
        }
        // Normalize.
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
        float[] classProbs = classifyProbabilistically(
                instance, distToTraining);
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
        // Perform the voting.
        float[] classProbEstimates = new float[numClasses];
        for (int kIndex = 0; kIndex < k; kIndex++) {
            if (neighborOccurrenceFreqs[trNeighbors[kIndex]] > thetaCutoff) {
                // The normal case.
                for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                    classProbEstimates[cIndex] +=
                            classDataKNeighborRelation[cIndex][
                            trNeighbors[kIndex]];
                }
            } else {
                // The special anti-hub handling case.
                for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                    switch (localEstimateMethod) {
                        case LOCAL:
                            classProbEstimates[cIndex] +=
                                    localClassDistribution[
                                    trNeighbors[kIndex]][cIndex];
                            break;
                        case LOCALF:
                            classProbEstimates[cIndex] +=
                                    localFClassDistribution[
                                    trNeighbors[kIndex]][cIndex];
                            break;
                        case GLOBAL:
                            classProbEstimates[cIndex] +=
                                    classToClassPriors[cIndex][
                                    trainingData.data.get(
                                    trNeighbors[kIndex]).getCategory()];
                            break;
                        case LABEL:
                            classProbEstimates[cIndex] +=
                                    localCrispClassDistribution[
                                    trNeighbors[kIndex]][cIndex];
                            break;
                        default:
                            classProbEstimates[cIndex] +=
                                    localFClassDistribution[
                                    trNeighbors[kIndex]][cIndex];
                    }
                }
            }
        }
        // Normalize.
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
