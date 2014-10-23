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

import learning.supervised.evaluation.ValidateableInterface;
import distances.primary.CombinedMetric;
import data.representation.DataInstance;
import data.representation.DataSet;
import data.representation.util.DataMineConstants;
import java.io.Serializable;
import learning.supervised.Category;
import learning.supervised.Classifier;
import learning.supervised.interfaces.DistToPointsQueryUserInterface;
import learning.supervised.interfaces.NeighborPointsQueryUserInterface;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

/**
 * This class implements the distance-weighted extension of the basic k-nearest
 * neighbor classifier.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class DWKNN extends Classifier implements DistToPointsQueryUserInterface,
        NeighborPointsQueryUserInterface, Serializable {
    
    private static final long serialVersionUID = 1L;

    // The training dataset.
    private DataSet trainingData = null;
    // The number of classes in the data.
    private int numClasses = 0;
    // The neighborhood size.
    private int k = 1;
    // The prior class distribution.
    private float[] classPriors;
    // The distance weighting parameter.
    private float mValue = 2;
    
    @Override
    public HashMap<String, String> getParameterNamesAndDescriptions() {
        HashMap<String, String> paramMap = new HashMap<>();
        paramMap.put("k", "Neighborhood size.");
        paramMap.put("mValue", "Exponent for distance weighting. Defaults"
                + " to 2.");

        return paramMap;
    }

    @Override
    public String getName() {
        return "dwKNN";
    }
    
    @Override
    public long getVersion() {
        return serialVersionUID;
    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     * @param cmet CombinedMetric object for distance calculations.
     */
    public DWKNN(int k, CombinedMetric cmet) {
        setCombinedMetric(cmet);
        this.k = k;
    }

    /**
     * Initialization.
     *
     * @param dset DataSet object used for model training.
     * @param numClasses Integer that is the number of classes in the data.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Integer that is the neighborhood size.
     */
    public DWKNN(DataSet dset, int numClasses, CombinedMetric cmet, int k) {
        trainingData = dset;
        this.numClasses = numClasses;
        setCombinedMetric(cmet);
        this.k = k;
    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     * @param cmet CombinedMetric object for distance calculations.
     * @param mValue Float that is the distance weighting parameter.
     */
    public DWKNN(int k, CombinedMetric cmet, float mValue) {
        setCombinedMetric(cmet);
        this.k = k;
        this.mValue = mValue;
    }

    /**
     * Initialization.
     *
     * @param dset DataSet object used for model training.
     * @param numClasses Integer that is the number of classes in the data.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Integer that is the neighborhood size.
     * @param mValue Float that is the distance weighting parameter.
     */
    public DWKNN(DataSet dset, int numClasses, CombinedMetric cmet, int k,
            float mValue) {
        trainingData = dset;
        this.numClasses = numClasses;
        setCombinedMetric(cmet);
        this.k = k;
        this.mValue = mValue;
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
     * @return Integer that is the neighborhood size used in calculations.
     */
    public int getK() {
        return k;
    }

    /**
     * @param k Integer that is the neighborhood size used in calculations.
     */
    public void setK(int k) {
        this.k = k;
    }

    /**
     * Initialization.
     *
     * @param categories Category[] representing the training data.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Integer that is the neighborhood size.
     */
    public DWKNN(Category[] categories, CombinedMetric cmet, int k) {
        int totalSize = 0;
        int indexFirstNonEmptyClass = -1;
        for (int cIndex = 0; cIndex < categories.length; cIndex++) {
            totalSize += categories[cIndex].size();
            if (indexFirstNonEmptyClass == -1
                    && categories[cIndex].size() > 0) {
                indexFirstNonEmptyClass = cIndex;
            }
        }
        // Instances are not embedded in the internal data context.
        trainingData = new DataSet();
        trainingData.fAttrNames = categories[indexFirstNonEmptyClass].
                getInstance(0).getEmbeddingDataset().fAttrNames;
        trainingData.iAttrNames = categories[indexFirstNonEmptyClass].
                getInstance(0).getEmbeddingDataset().iAttrNames;
        trainingData.sAttrNames = categories[indexFirstNonEmptyClass].
                getInstance(0).getEmbeddingDataset().sAttrNames;
        trainingData.data = new ArrayList<>(totalSize);
        for (int cIndex = 0; cIndex < categories.length; cIndex++) {
            for (int i = 0; i < categories[cIndex].size(); i++) {
                categories[cIndex].getInstance(i).setCategory(cIndex);
                trainingData.addDataInstance(categories[cIndex].getInstance(i));
            }
        }
        setCombinedMetric(cmet);
        this.k = k;
        numClasses = trainingData.countCategories();
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
        // Instances are not embedded in the internal data context.
        trainingData = new DataSet();
        trainingData.fAttrNames = categories[indexFirstNonEmptyClass].
                getInstance(0).getEmbeddingDataset().fAttrNames;
        trainingData.iAttrNames = categories[indexFirstNonEmptyClass].
                getInstance(0).getEmbeddingDataset().iAttrNames;
        trainingData.sAttrNames = categories[indexFirstNonEmptyClass].
                getInstance(0).getEmbeddingDataset().sAttrNames;
        trainingData.data = new ArrayList<>(totalSize);
        for (int cIndex = 0; cIndex < categories.length; cIndex++) {
            for (int i = 0; i < categories[cIndex].size(); i++) {
                categories[cIndex].getInstance(i).setCategory(cIndex);
                trainingData.addDataInstance(categories[cIndex].getInstance(i));
            }
        }
        numClasses = trainingData.countCategories();
    }

    @Override
    public ValidateableInterface copyConfiguration() {
        return new DWKNN(trainingData, numClasses, getCombinedMetric(), k,
                mValue);
    }

    @Override
    public void train() throws Exception {
        if (trainingData != null) {
            numClasses = trainingData.countCategories();
            classPriors = trainingData.getClassPriors();
        }
    }

    @Override
    public int classify(DataInstance instance) throws Exception {
        float[] classProbs = classifyProbabilistically(instance);
        float maxProb = 0;
        int maxClass = 0;
        for (int cIndex = 0; cIndex < numClasses; cIndex++) {
            if (classProbs[cIndex] > maxProb) {
                maxProb = classProbs[cIndex];
                maxClass = cIndex;
            }
        }
        return maxClass;
    }

    @Override
    public float[] classifyProbabilistically(DataInstance instance)
            throws Exception {
        CombinedMetric cmet = getCombinedMetric();
        // Calculate the kNN set.
        float[] kDistances = new float[k];
        Arrays.fill(kDistances, Float.MAX_VALUE);
        int[] kNeighbors = new int[k];
        float currDistance;
        int index;
        // Initialize the distance weights.
        float[] distanceWeights = new float[k];
        float dwSum = 0;
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
        for (int kIndex = 0; kIndex < k; kIndex++) {
            if (DataMineConstants.isAcceptableFloat(
                    kDistances[kIndex])
                    && DataMineConstants.isPositive(
                    kDistances[kIndex])) {
                distanceWeights[kIndex] = 1f
                        / ((float) Math.pow(kDistances[kIndex],
                        (2f / (mValue - 1f))));
            } else if (DataMineConstants.isZero(kDistances[kIndex])) {
                distanceWeights[kIndex] = 1000f;
            } else {
                distanceWeights[kIndex] = Float.MIN_VALUE;
            }
            dwSum += distanceWeights[kIndex];
        }
        // Perform the voting.
        float[] classProbEstimates = new float[numClasses];
        for (int kIndex = 0; kIndex < k; kIndex++) {
            if (dwSum > 0) {
                classProbEstimates[trainingData.getLabelOf(
                        kNeighbors[kIndex])] +=
                        distanceWeights[kIndex] / dwSum;
            } else {
                classProbEstimates[trainingData.getLabelOf(
                        kNeighbors[kIndex])]++;
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
        float[] classProbs = classifyProbabilistically(instance,
                distToTraining);
        float maxProb = 0;
        int maxClassIndex = 0;
        for (int cIndex = 0; cIndex < numClasses; cIndex++) {
            if (classProbs[cIndex] > maxProb) {
                maxProb = classProbs[cIndex];
                maxClassIndex = cIndex;
            }
        }
        return maxClassIndex;
    }

    @Override
    public float[] classifyProbabilistically(DataInstance instance,
            float[] distToTraining) throws Exception {
        // Calculate the kNN set.
        float[] kDistances = new float[k];
        Arrays.fill(kDistances, Float.MAX_VALUE);
        int[] kNeighbors = new int[k];
        int index;
        for (int i = 0; i < trainingData.size(); i++) {
            if (distToTraining[i] < kDistances[k - 1]) {
                // Insertion.
                index = k - 1;
                while (index > 0 && kDistances[index - 1] > distToTraining[i]) {
                    kDistances[index] = kDistances[index - 1];
                    kNeighbors[index] = kNeighbors[index - 1];
                    index--;
                }
                kDistances[index] = distToTraining[i];
                kNeighbors[index] = i;
            }
        }
        // Calculate the distance weights.
        float[] distanceWeights = new float[k];
        float dwSum = 0;
        for (int kIndex = 0; kIndex < k; kIndex++) {
            if (DataMineConstants.isAcceptableFloat(
                    kDistances[kIndex])
                    && DataMineConstants.isPositive(
                    kDistances[kIndex])) {
                distanceWeights[kIndex] = 1f
                        / ((float) Math.pow(kDistances[kIndex],
                        (2f / (mValue - 1f))));
            } else if (DataMineConstants.isZero(kDistances[kIndex])) {
                distanceWeights[kIndex] = 1000f;
            } else {
                distanceWeights[kIndex] = Float.MIN_VALUE;
            }
            dwSum += distanceWeights[kIndex];
        }
        float[] classProbEstimates = new float[numClasses];
        // Perform the voting.
        for (int kIndex = 0; kIndex < k; kIndex++) {
            if (dwSum > 0) {
                classProbEstimates[trainingData.getLabelOf(
                        kNeighbors[kIndex])] +=
                        distanceWeights[kIndex] / dwSum;
            } else {
                classProbEstimates[trainingData.getLabelOf(
                        kNeighbors[kIndex])]++;
            }
        }
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

    /**
     * Classify the point of interest based on the kNN set and the distances to
     * the neighbor points.
     *
     * @param instance DataInstance object that is to be classified.
     * @param kDistances float[] representing the distances to the k-nearest
     * neighbors.
     * @param trNeighbors int[] representing the indexes of the kNN set.
     * @return Integer that is the predicted class affiliation in the point of
     * interest.
     * @throws Exception
     */
    public int classifyWithKDistAndNeighbors(DataInstance instance,
            float[] kDistances, int[] trNeighbors) throws Exception {
        // Calculate the distance weights.
        float[] distanceWeights = new float[k];
        float dwSum = 0;
        for (int kIndex = 0; kIndex < k; kIndex++) {
            if (DataMineConstants.isAcceptableFloat(
                    kDistances[kIndex])
                    && DataMineConstants.isPositive(
                    kDistances[kIndex])) {
                distanceWeights[kIndex] = 1f
                        / ((float) Math.pow(kDistances[kIndex],
                        (2f / (mValue - 1f))));
            } else if (DataMineConstants.isZero(kDistances[kIndex])) {
                distanceWeights[kIndex] = 1000f;
            } else {
                distanceWeights[kIndex] = Float.MIN_VALUE;
            }
            dwSum += distanceWeights[kIndex];
        }
        float[] classProbEstimates = new float[numClasses];
        // Perform the voting.
        for (int kIndex = 0; kIndex < k; kIndex++) {
            if (dwSum > 0) {
                classProbEstimates[trainingData.getLabelOf(
                        trNeighbors[kIndex])] +=
                        distanceWeights[kIndex] / dwSum;
            } else {
                classProbEstimates[trainingData.getLabelOf(
                        trNeighbors[kIndex])]++;
            }
        }
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
        float maxProb = 0;
        int maxClassIndex = 0;
        for (int cIndex = 0; cIndex < numClasses; cIndex++) {
            if (classProbEstimates[cIndex] > maxProb) {
                maxProb = classProbEstimates[cIndex];
                maxClassIndex = cIndex;
            }
        }
        return maxClassIndex;
    }

    /**
     * Classify the point of interest based on the kNN set and the distances to
     * the neighbor points.
     *
     * @param instance DataInstance object that is to be classified.
     * @param kDistances float[] representing the distances to the k-nearest
     * neighbors.
     * @param trNeighbors int[] representing the indexes of the kNN set.
     * @return float[] that is the predicted class distribution in the point of
     * interest.
     * @throws Exception
     */
    public float[] classifyProbabilisticallyWithKDistAndNeighbors(
            DataInstance instance, float[] kDistances, int[] trNeighbors)
            throws Exception {
        // Calculate the distance weights.
        float[] distanceWeights = new float[k];
        float dwSum = 0;
        for (int kIndex = 0; kIndex < k; kIndex++) {
            if (DataMineConstants.isAcceptableFloat(
                    kDistances[kIndex])
                    && DataMineConstants.isPositive(
                    kDistances[kIndex])) {
                distanceWeights[kIndex] = 1f
                        / ((float) Math.pow(kDistances[kIndex],
                        (2f / (mValue - 1f))));
            } else if (DataMineConstants.isZero(kDistances[kIndex])) {
                distanceWeights[kIndex] = 1000f;
            } else {
                distanceWeights[kIndex] = Float.MIN_VALUE;
            }
            dwSum += distanceWeights[kIndex];
        }
        float[] classProbEstimates = new float[numClasses];
        // Perform the voting.
        for (int kIndex = 0; kIndex < k; kIndex++) {
            if (dwSum > 0) {
                classProbEstimates[trainingData.getLabelOf(
                        trNeighbors[kIndex])] +=
                        distanceWeights[kIndex] / dwSum;
            } else {
                classProbEstimates[trainingData.getLabelOf(
                        trNeighbors[kIndex])]++;
            }
        }
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
    public int classify(DataInstance instance, float[] distToTraining,
            int[] trNeighbors) throws Exception {
        float[] classProbs = classifyProbabilistically(instance, distToTraining,
                trNeighbors);
        float maxProb = 0;
        int maxClassIndex = 0;
        for (int cIndex = 0; cIndex < numClasses; cIndex++) {
            if (classProbs[cIndex] > maxProb) {
                maxProb = classProbs[cIndex];
                maxClassIndex = cIndex;
            }
        }
        return maxClassIndex;
    }

    @Override
    public float[] classifyProbabilistically(DataInstance instance,
            float[] distToTraining, int[] trNeighbors) throws Exception {
        // Calculate the distance weights.
        float[] distanceWeights = new float[k];
        float dwSum = 0;
        for (int kIndex = 0; kIndex < k; kIndex++) {
            if (DataMineConstants.isAcceptableFloat(
                    distToTraining[trNeighbors[kIndex]])
                    && DataMineConstants.isPositive(
                    distToTraining[trNeighbors[kIndex]])) {
                distanceWeights[kIndex] = 1f
                        / ((float) Math.pow(distToTraining[trNeighbors[kIndex]],
                        (2f / (mValue - 1f))));
            } else if (DataMineConstants.isZero(
                    distToTraining[trNeighbors[kIndex]])) {
                distanceWeights[kIndex] = 1000f;
            } else {
                distanceWeights[kIndex] = Float.MIN_VALUE;
            }
            dwSum += distanceWeights[kIndex];
        }
        float[] classProbEstimates = new float[numClasses];
        // Perform the voting.
        for (int kIndex = 0; kIndex < k; kIndex++) {
            if (dwSum > 0) {
                classProbEstimates[trainingData.getLabelOf(
                        trNeighbors[kIndex])] +=
                        distanceWeights[kIndex] / dwSum;
            } else {
                classProbEstimates[trainingData.getLabelOf(
                        trNeighbors[kIndex])]++;
            }
        }
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
}
