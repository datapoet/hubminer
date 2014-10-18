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
import learning.supervised.Category;
import learning.supervised.Classifier;
import learning.supervised.evaluation.ValidateableInterface;
import learning.supervised.interfaces.DistToPointsQueryUserInterface;
import learning.supervised.interfaces.NeighborPointsQueryUserInterface;

/**
 * This paper implements the classical fuzzy k-nearest neighbor algorithm.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class FNN extends Classifier implements AutomaticKFinderInterface,
        NSFUserInterface, DistToPointsQueryUserInterface,
        NeighborPointsQueryUserInterface, Serializable {
    
    private static final long serialVersionUID = 1L;

    // Distance weighting parameter.
    private float mFactor = 2;
    private int k = 5;
    private NeighborSetFinder nsf = null;
    private DataSet trainingData = null;
    private int numClasses = 0;
    private float laplaceEstimator = 0.001f;
    private float[][] localClassDistribution = null;

    @Override
    public void noRecalcs() {
    }

    @Override
    public String getName() {
        return "FNN";
    }

    @Override
    public void findK(int kMin, int kMax) throws Exception {
        DataSet dset = trainingData;
        NeighborSetFinder nsfLOU = new NeighborSetFinder(dset,
                getCombinedMetric());
        nsfLOU.calculateDistances();
        nsfLOU.calculateNeighborSets(kMax);
        // Array that holds the accuracies for the tested k-range.
        float[] accuracyArray = new float[kMax - kMin + 1];
        float currMaxAcc = -1f;
        int currMaxK = 0;
        int numElements = dset.size();
        // Used incrementally to determine the votes.
        float[][] currVoteCounts = new float[numElements][numClasses];
        float currMaxVote;
        // Current label prediction, according to the current vote.
        int[] currVoteLabels = new int[numElements];
        // Get the kNN set.
        int[][] kneighbors = nsfLOU.getKNeighbors();
        float currAccuracy;
        // Neighborhood size for fuzziness estimation. Preferrably 10.
        int kEstimate = Math.min(kMax, 10);
        localClassDistribution = new float[trainingData.size()][];
        float laplaceTotal = numClasses * laplaceEstimator;
        for (int i = 0; i < trainingData.size(); i++) {
            // Calculate the class distribution in the kNN set.
            localClassDistribution[i] = new float[numClasses];
            for (int kInd = 0; kInd < kEstimate; kInd++) {
                localClassDistribution[i][trainingData.data.get(
                        kneighbors[i][kInd]).getCategory()]++;
            }
            // Perform some smoothing and account for the influence of the
            // query point's label.
            localClassDistribution[i][trainingData.data.get(i).getCategory()]++;
            for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                localClassDistribution[i][cIndex] += laplaceEstimator;
                localClassDistribution[i][cIndex] /=
                        (kEstimate + 1 + laplaceTotal);
                if (trainingData.data.get(i).getCategory() == cIndex) {
                    localClassDistribution[i][cIndex] = 0.51f + 0.49f
                            * localClassDistribution[i][cIndex];
                } else {
                    localClassDistribution[i][cIndex] = 0.49f
                            * localClassDistribution[i][cIndex];
                }
            }
        }
        // Go through all the k-increments.
        for (int kInc = 0; kInc < accuracyArray.length; kInc++) {
            currAccuracy = 0;
            // Update the current votes and calculate the accuracy.
            for (int index = 0; index < numElements; index++) {
                currMaxVote = 0;
                // Place additional votes to account for new neighbors
                // incrementally.
                for (int classIndex = 0; classIndex < numClasses;
                        classIndex++) {
                    currVoteCounts[index][classIndex] +=
                            localClassDistribution[kneighbors[index][
                            kMin + kInc - 1]][classIndex];
                    if (currVoteCounts[index][classIndex] > currMaxVote) {
                        currMaxVote = currVoteCounts[index][classIndex];
                        currVoteLabels[index] = classIndex;
                    }
                }
                // If the vote was correct, increase the current accuracy count.
                if (currVoteLabels[index]
                        == dset.data.get(index).getCategory()) {
                    currAccuracy++;
                }
            }
            // Normalize.
            currAccuracy /= (float) numElements;
            // Keep track of the best accuracy and update the optimal k choice.
            accuracyArray[kInc] = currAccuracy;
            if (currMaxAcc < currAccuracy) {
                currMaxAcc = currAccuracy;
                currMaxK = kMin + kInc;
            }
        }
        k = currMaxK;
        localClassDistribution = null;
    }

    /**
     * Default constructor.
     */
    public FNN() {
    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     */
    public FNN(int k) {
        this.k = k;
    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     * @param cmet CombinedMetric object for distance calculations.
     */
    public FNN(int k, CombinedMetric cmet) {
        this.k = k;
        setCombinedMetric(cmet);
    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     * @param cmet CombinedMetric object for distance calculations.
     * @param numClasses Integer that is the number of classes.
     */
    public FNN(int k, CombinedMetric cmet, int numClasses) {
        this.k = k;
        setCombinedMetric(cmet);
        this.numClasses = numClasses;

    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     * @param cmet CombinedMetric object for distance calculations.
     * @param numClasses Integer that is the number of classes.
     * @param mValue Float that is the distance weighting parameter.
     */
    public FNN(int k, CombinedMetric cmet, int numClasses, float mValue) {
        this.k = k;
        setCombinedMetric(cmet);
        this.numClasses = numClasses;
        this.mFactor = mValue;
    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     * @param laplaceEstimator Float that is the Laplace estimator for
     * smoothing.
     */
    public FNN(int k, float laplaceEstimator) {
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
    public FNN(int k, float laplaceEstimator, CombinedMetric cmet) {
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
     * @param numClasses Integer that is the number of classes.
     */
    public FNN(int k, float laplaceEstimator, CombinedMetric cmet,
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
     * @param laplaceEstimator Float that is the Laplace estimator for
     * smoothing.
     * @param cmet CombinedMetric object for distance calculations.
     * @param numClasses Integer that is the number of classes.
     * @param mValue Float that is the distance weighting parameter.
     */
    public FNN(int k, float laplaceEstimator, CombinedMetric cmet,
            int numClasses, float mValue) {
        this.k = k;
        this.laplaceEstimator = laplaceEstimator;
        setCombinedMetric(cmet);
        this.numClasses = numClasses;
    }

    /**
     * Initialization.
     *
     * @param dset DataSet that is the training data.
     * @param numClasses Integer that is the number of classes.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Integer that is the neighborhood size.
     */
    public FNN(DataSet dset, int numClasses, CombinedMetric cmet, int k) {
        trainingData = dset;
        this.numClasses = numClasses;
        setCombinedMetric(cmet);
        this.k = k;
    }

    /**
     * Initialization.
     *
     * @param dset DataSet that is the training data.
     * @param numClasses Integer that is the number of classes.
     * @param nsf NeighborSetFinder object that holds kNN sets.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Integer that is the neighborhood size.
     */
    public FNN(DataSet dset, int numClasses, NeighborSetFinder nsf,
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
     * @param categories Category[] array of categories that represent the
     * training data.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Integer that is the neighborhood size.
     */
    public FNN(Category[] categories, CombinedMetric cmet, int k) {
        int totalSize = 0;
        int indexFirstNonEmptyClass = -1;
        for (int cIndex = 0; cIndex < categories.length; cIndex++) {
            totalSize += categories[cIndex].size();
            if (indexFirstNonEmptyClass == -1
                    && categories[cIndex].size() > 0) {
                indexFirstNonEmptyClass = cIndex;
            }
        }
        // An internal DataSet, instances won't be embedded.
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
                trainingData.addDataInstance(categories[cIndex].
                        getInstance(i));
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
        for (int cIndex = 0; cIndex < categories.length; cIndex++) {
            totalSize += categories[cIndex].size();
            if (indexFirstNonEmptyClass == -1 &&
                    categories[cIndex].size() > 0) {
                indexFirstNonEmptyClass = cIndex;
            }
        }
        // An internal DataSet, instances won't be embedded.
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
                trainingData.addDataInstance(categories[cIndex].
                        getInstance(i));
            }
        }
        numClasses = trainingData.countCategories();
    }

    /**
     * @param trainingData DataSet to train the model on.
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
     * Calculate the kNN sets according to the current specification.
     *
     * @throws Exception
     */
    public void calculateNeighborSets() throws Exception {
        nsf = new NeighborSetFinder(trainingData, getCombinedMetric());
        nsf.calculateDistances();
        nsf.calculateNeighborSets(Math.max(10, k));
    }

    @Override
    public ValidateableInterface copyConfiguration() {
        return new FNN(k, laplaceEstimator,
                getCombinedMetric(), numClasses, mFactor);
    }

    @Override
    public void train() throws Exception {
        if (k <= 0) {
            // Default k-value search if no value was provided.
            findK(1, 20);
        }
        if (nsf == null || nsf.getCurrK() < 10) {
            // If the kNN sets have not been passed, calculate them.
            calculateNeighborSets();
        }
        // Learn the fuzzy votes.
        localClassDistribution = new float[trainingData.size()][];
        int[][] kneighbors = nsf.getKNeighbors();
        float laplaceTotal = numClasses * laplaceEstimator;
        for (int i = 0; i < trainingData.size(); i++) {
            localClassDistribution[i] = new float[numClasses];
            // Calculate the class distribution among the kNN sets.
            for (int kInd = 0; kInd < 10; kInd++) {
                localClassDistribution[i][trainingData.data.get(
                        kneighbors[i][kInd]).getCategory()]++;
            }
            // Smooth the fuzzy votes.
            localClassDistribution[i][trainingData.data.get(i).getCategory()]++;
            for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                localClassDistribution[i][cIndex] += laplaceEstimator;
                localClassDistribution[i][cIndex] /= (10 + 1 + laplaceTotal);
                if (trainingData.data.get(i).getCategory() == cIndex) {
                    localClassDistribution[i][cIndex] =
                            0.51f + 0.49f * localClassDistribution[i][cIndex];
                } else {
                    localClassDistribution[i][cIndex] =
                            0.49f * localClassDistribution[i][cIndex];
                }
            }
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
        // Calculate the k-nearest neighbors.
        float[] kDistances = new float[k];
        for (int kInd = 0; kInd < k; kInd++) {
            kDistances[kInd] = Float.MAX_VALUE;
        }
        int[] kNeighbors = new int[k];
        float currDistance;
        int index;
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
        for (int kInd = 0; kInd < k; kInd++) {
            if (kDistances[kInd] != 0) {
                distanceWeights[kInd] = 1f
                        / ((float) Math.pow(kDistances[kInd],
                        (2f / (mFactor - 1f))));
            } else {
                distanceWeights[kInd] = 10000f;
            }
            dwSum += distanceWeights[kInd];
        }

        float[] classProbEstimates = new float[numClasses];
        // Perform the voting.
        for (int kInd = 0; kInd < k; kInd++) {
            for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                classProbEstimates[cIndex] +=
                        localClassDistribution[kNeighbors[kInd]][cIndex]
                        * distanceWeights[kInd] / dwSum;
            }
        }
        // Normalize.
        float probTotal = 0;
        for (int cIndex = 0; cIndex < numClasses; cIndex++) {
            probTotal += classProbEstimates[cIndex];
        }
        for (int cIndex = 0; cIndex < numClasses; cIndex++) {
            classProbEstimates[cIndex] /= probTotal;
        }
        return classProbEstimates;
    }

    @Override
    public float[] classifyProbabilistically(DataInstance instance,
            float[] distToTraining) throws Exception {
        // Find the kNN set.
        float[] kDistances = new float[k];
        for (int kInd = 0; kInd < k; kInd++) {
            kDistances[kInd] = Float.MAX_VALUE;
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
        for (int kInd = 0; kInd < k; kInd++) {
            if (kDistances[kInd] != 0) {
                distanceWeights[kInd] = 1f
                        / ((float) Math.pow(kDistances[kInd],
                        (2f / (mFactor - 1f))));
            } else {
                distanceWeights[kInd] = 10000f;
            }
            dwSum += distanceWeights[kInd];
        }
        float[] classProbEstimates = new float[numClasses];
        // Perform the voting.
        for (int kInd = 0; kInd < k; kInd++) {
            for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                classProbEstimates[cIndex] +=
                        localClassDistribution[kNeighbors[kInd]][cIndex]
                        * distanceWeights[kInd] / dwSum;
            }
        }
        // Normalize.
        float probTotal = 0;
        for (int cIndex = 0; cIndex < numClasses; cIndex++) {
            probTotal += classProbEstimates[cIndex];
        }
        for (int cIndex = 0; cIndex < numClasses; cIndex++) {
            classProbEstimates[cIndex] /= probTotal;
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
        float[] classProbs = classifyProbabilistically(instance,
                distToTraining, trNeighbors);
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
        // Calculate the distance weights.
        float[] distanceWeights = new float[k];
        float dwSum = 0;
        for (int kInd = 0; kInd < k; kInd++) {
            if (distToTraining[trNeighbors[kInd]] != 0) {
                distanceWeights[kInd] = 1f / ((float) Math.pow(
                        distToTraining[trNeighbors[kInd]],
                        (2f / (mFactor - 1f))));
            } else {
                distanceWeights[kInd] = 10000f;
            }
            dwSum += distanceWeights[kInd];
        }
        // Perform the voting.
        float[] classProbEstimates = new float[numClasses];
        for (int kInd = 0; kInd < k; kInd++) {
            for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                classProbEstimates[cIndex] += localClassDistribution[
                        trNeighbors[kInd]][cIndex]
                        * distanceWeights[kInd] / dwSum;
            }
        }
        float probTotal = 0;
        for (int cIndex = 0; cIndex < numClasses; cIndex++) {
            probTotal += classProbEstimates[cIndex];
        }
        for (int cIndex = 0; cIndex < numClasses; cIndex++) {
            classProbEstimates[cIndex] /= probTotal;
        }
        return classProbEstimates;
    }
    
    @Override
    public int getNeighborhoodSize() {
        return k;
    }
}
