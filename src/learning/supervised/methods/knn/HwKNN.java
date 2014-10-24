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
import java.util.HashMap;
import learning.supervised.Category;
import learning.supervised.Classifier;
import learning.supervised.evaluation.ValidateableInterface;
import learning.supervised.evaluation.cv.MultiCrossValidation;
import learning.supervised.interfaces.DistMatrixUserInterface;
import learning.supervised.interfaces.DistToPointsQueryUserInterface;
import learning.supervised.interfaces.NeighborPointsQueryUserInterface;
import preprocessing.instance_selection.InstanceSelector;

/**
 * This class implements the hw-kNN method for hubness-based instance weighting
 * in kNN classification. This approach was proposed in the following paper:
 * "Nearest Neighbors in High-Dimensional Data: The Emergence and Influence of
 * Hubs" at ICML 2009.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class HwKNN extends Classifier implements AutomaticKFinderInterface,
        DistMatrixUserInterface, NSFUserInterface,
        DistToPointsQueryUserInterface, NeighborPointsQueryUserInterface,
        Serializable {
    
    private static final long serialVersionUID = 1L;

    private DataSet trainingData = null;
    private int numClasses = 0;
    private float[] instanceWeights = null;
    private float[][] distMat = null;
    private NeighborSetFinder nsf = null;
    private int k = 5;
    
    /**
     * Default constructor.
     */
    public HwKNN() {
    }
    
    @Override
    public HashMap<String, String> getParameterNamesAndDescriptions() {
        HashMap<String, String> paramMap = new HashMap<>();
        paramMap.put("k", "Neighborhood size.");
        return paramMap;
    }
    
    @Override
    public long getVersion() {
        return serialVersionUID;
    }

    @Override
    public void noRecalcs() {
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
        return "hw-KNN";
    }

    @Override
    public void findK(int kMin, int kMax) throws Exception {
        // This is an approximate fast k-search procedure. It incrementally
        // updates the votes as k is increased. However, the votes for lower
        // k-s are not updated. TODO: change into an exact method instead.
        DataSet dset = trainingData;
        // Calculate the kNN sets if they haven't already been provided.
        NeighborSetFinder nsfLOU;
        if (distMat == null) {
            nsfLOU = new NeighborSetFinder(dset, getCombinedMetric());
            nsfLOU.calculateDistances();
        } else {
            nsfLOU = new NeighborSetFinder(dset, distMat, getCombinedMetric());
        }
        nsfLOU.calculateNeighborSets(kMax);
        // This array will store the accuracies for all the k-values.
        float[] accuracyArray = new float[kMax - kMin + 1];
        float currMaxAcc = -1f;
        int currMaxK = 0;
        int numElements = dset.size();
        float[][] currVoteCounts = new float[numElements][numClasses];
        int[] currVoteLabels = new int[numElements];
        int currNeighborLabel;
        int[][] kneighbors = nsfLOU.getKNeighbors();
        float currAcc;
        // For the entire k-range.
        for (int kInc = 0; kInc < accuracyArray.length; kInc++) {
            currAcc = 0;
            nsfLOU.recalculateStatsForSmallerK(kMin + kInc);
            instanceWeights = nsfLOU.getHWKNNWeightingScheme();
            // Find the accuracy of the method on training data for given k
            // value. This is done incrementally, as with each new k a new vote
            // is added.
            for (int index = 0; index < numElements; index++) {
                currNeighborLabel = dset.data.get(kneighbors[index][
                        kMin + kInc - 1]).getCategory();
                currVoteCounts[index][currNeighborLabel] +=
                        instanceWeights[kneighbors[index][kMin + kInc - 1]];
                if (currVoteLabels[index] != currNeighborLabel) {
                    if (currVoteCounts[index][currNeighborLabel]
                            > currVoteCounts[index][currVoteLabels[index]]) {
                        currVoteLabels[index] = currNeighborLabel;
                    }
                }
                if (currVoteLabels[index]
                        == dset.data.get(index).getCategory()) {
                    currAcc++;
                }
            }
            currAcc /= (float) numElements;
            accuracyArray[kInc] = currAcc;
            if (currMaxAcc < currAcc) {
                currMaxAcc = currAcc;
                currMaxK = kMin + kInc;
            }
        }
        k = currMaxK;
        instanceWeights = null;
    }

    /**
     * Initialization.
     *
     * @param numClasses Integer that is the number of classes.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Integer that is the neighborhood size.
     */
    public HwKNN(int numClasses, CombinedMetric cmet, int k) {
        this.numClasses = numClasses;
        if (trainingData != null) {
            instanceWeights = new float[trainingData.size()];
        }
        setCombinedMetric(cmet);
        this.k = k;
    }

    /**
     * Initialization.
     *
     * @param dset DataSet object representing the training data.
     * @param numClasses Integer that is the number of classes.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Integer that is the neighborhood size.
     */
    public HwKNN(DataSet dset, int numClasses, CombinedMetric cmet, int k) {
        trainingData = dset;
        this.numClasses = numClasses;
        if (trainingData != null) {
            instanceWeights = new float[trainingData.size()];
        }
        setCombinedMetric(cmet);
        this.k = k;
    }

    /**
     * Initialization
     *
     * @param dset DataSet object representing the training data.
     * @param numClasses Integer that is the number of classes.
     * @param weights float[] containing the instance weights.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Integer that is the neighborhood size.
     */
    public HwKNN(DataSet dset, int numClasses, float[] weights,
            CombinedMetric cmet, int k) {
        trainingData = dset;
        this.numClasses = numClasses;
        this.instanceWeights = weights;
        setCombinedMetric(cmet);
        this.k = k;
    }

    /**
     * Initialization
     *
     * @param categories Category[] of categories composing the training data.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Integer that is the neighborhood size.
     */
    public HwKNN(Category[] categories, CombinedMetric cmet, int k) {
        setClasses(categories);
        instanceWeights = new float[trainingData.size()];
        setCombinedMetric(cmet);
        this.k = k;
    }

    @Override
    public ValidateableInterface copyConfiguration() {
        return new HwKNN(trainingData, numClasses, instanceWeights,
                getCombinedMetric(), k);
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
                break;
            }
        }
        // An internal data context, so the instances are not embedded.
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
        instanceWeights = new float[trainingData.size()];
        numClasses = trainingData.countCategories();
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
     * @param instanceWeights float[] of instance weights if they are
     * pre-computed. Hw-kNN actually calculates them internally.
     */
    public void setWeights(float[] instanceWeights) {
        this.instanceWeights = instanceWeights;
    }

    /**
     * @return float[] of instance weights.
     */
    public float[] getWeights() {
        return instanceWeights;
    }

    /**
     * @return DataSet object that is the training data.
     */
    public DataSet getTrainingSet() {
        return trainingData;
    }

    /**
     * @param trainingData DataSet object that is the training data.
     */
    public void setTrainingSet(DataSet trainingData) {
        this.trainingData = trainingData;
    }

    /**
     * @param numClasses Integer that is the number of classes.
     */
    public void setNumClasses(int numClasses) {
        this.numClasses = numClasses;
    }

    /**
     * @return Integer that is the number of classes.
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

    /**
     * Calculate the hubness-based weighting scheme.
     *
     * @throws Exception
     */
    public void getWeightsFromHubness() throws Exception {
        if (nsf == null) {
            if (distMat == null) {
                nsf = new NeighborSetFinder(trainingData, getCombinedMetric());
                nsf.calculateDistances();
            } else {
                nsf = new NeighborSetFinder(trainingData, distMat,
                        getCombinedMetric());
            }
            nsf.calculateNeighborSets(k);
        }
        instanceWeights = nsf.getHWKNNWeightingScheme();
    }

    @Override
    public void train() throws Exception {
        if (k <= 0) {
            findK(1, 20);
        }
        getWeightsFromHubness();
    }

    @Override
    public void trainOnReducedData(InstanceSelector reducer) throws Exception {
        ArrayList<Integer> indexPermutation = MultiCrossValidation.
                getIndexPermutation(reducer.getPrototypeIndexes(),
                reducer.getOriginalDataSet());
        ArrayList<Integer> protoIndexes = reducer.getPrototypeIndexes();
        instanceWeights = new float[protoIndexes.size()];
        float[] unpermutedWeights = reducer.getKNNHubnessWeightingScheme();
        for (int i = 0; i < instanceWeights.length; i++) {
            instanceWeights[i] = unpermutedWeights[indexPermutation.get(i)];
        }
    }

    @Override
    public int classify(DataInstance instance) throws Exception {
        CombinedMetric cmet = getCombinedMetric();
        if (trainingData == null || instanceWeights == null
                || trainingData.size() != instanceWeights.length) {
            throw new Exception("Bad classifier initialization.");
        }
        if (instance == null) {
            return -1;
        }
        // Find the kNN set.
        int[] nearestInstances = new int[k];
        float[] nearestDistances = new float[k];
        for (int kIndex = 0; kIndex < k; kIndex++) {
            nearestDistances[kIndex] = Float.MAX_VALUE;
            nearestInstances[kIndex] = -1;
        }
        float currDist;
        int index;
        for (int i = 0; i < trainingData.size(); i++) {
            currDist = cmet.dist(instance, trainingData.data.get(i));
            index = k - 1;
            while (index >= 0 && nearestDistances[index] > currDist) {
                index--;
            }
            if (index < k - 1) {
                for (int j = k - 1; j > index + 1; j--) {
                    nearestDistances[j] = nearestDistances[j - 1];
                    nearestInstances[j] = nearestInstances[j - 1];
                }
                nearestInstances[index + 1] = i;
                nearestDistances[index + 1] = currDist;
            }
        }
        // Perform the weighted vote.
        float[] classProbabilities = new float[numClasses];
        float probTotal = 0;
        for (int kIndex = 0; kIndex < k; kIndex++) {
            classProbabilities[trainingData.data.get(nearestInstances[kIndex]).
                    getCategory()] += instanceWeights[nearestInstances[kIndex]];
            probTotal += instanceWeights[nearestInstances[kIndex]];
        }
        // Normalize.
        int maxClassIndex = -1;
        float maxProb = 0;
        if (probTotal == 0) {
            return 0;
        } else {
            for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                classProbabilities[cIndex] /= probTotal;
                if (classProbabilities[cIndex] >= maxProb) {
                    maxClassIndex = cIndex;
                    maxProb = classProbabilities[cIndex];
                }
            }
        }
        return maxClassIndex;
    }

    @Override
    public float[] classifyProbabilistically(DataInstance instance)
            throws Exception {
        CombinedMetric cmet = getCombinedMetric();
        if (trainingData == null || instanceWeights == null
                || trainingData.size() != instanceWeights.length) {
            throw new Exception("Bad classifier initialization");
        }
        if (instance == null) {
            return null;
        }
        // Find the kNN set.
        int[] nearestInstances = new int[k];
        float[] nearestDistances = new float[k];
        for (int kIndex = 0; kIndex < k; kIndex++) {
            nearestDistances[kIndex] = Float.MAX_VALUE;
            nearestInstances[kIndex] = -1;
        }
        float currDist;
        int index;
        for (int i = 0; i < trainingData.size(); i++) {
            currDist = cmet.dist(instance, trainingData.data.get(i));
            index = k - 1;
            while (index >= 0 && nearestDistances[index] > currDist) {
                index--;
            }
            if (index < k - 1) {
                for (int j = k - 1; j > index + 1; j--) {
                    nearestDistances[j] = nearestDistances[j - 1];
                    nearestInstances[j] = nearestInstances[j - 1];
                }
                nearestInstances[index + 1] = i;
                nearestDistances[index + 1] = currDist;
            }
        }
        // Perform the weighted voting.
        float[] classProbabilities = new float[numClasses];
        float probTotal = 0;
        for (int kIndex = 0; kIndex < k; kIndex++) {
            classProbabilities[trainingData.data.get(
                    nearestInstances[kIndex]).getCategory()] +=
                    instanceWeights[nearestInstances[kIndex]];
            probTotal += instanceWeights[nearestInstances[kIndex]];
        }
        // Normalize.
        if (probTotal == 0) {
            return new float[numClasses];
        } else {
            for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                classProbabilities[cIndex] /= probTotal;
            }
        }
        return classProbabilities;
    }

    @Override
    public float[] classifyProbabilistically(DataInstance instance,
            float[] distToTraining) throws Exception {
        if (trainingData == null || instanceWeights == null
                || trainingData.size() != instanceWeights.length) {
            throw new Exception("Bad classifier initialization");
        }
        if (instance == null) {
            return null;
        }
        // Find the kNN set.
        int[] nearestInstances = new int[k];
        float[] nearestDistances = new float[k];
        for (int kIndex = 0; kIndex < k; kIndex++) {
            nearestDistances[kIndex] = Float.MAX_VALUE;
            nearestInstances[kIndex] = -1;
        }
        float currDist;
        int index;
        for (int i = 0; i < trainingData.size(); i++) {
            currDist = distToTraining[i];
            index = k - 1;
            while (index >= 0 && nearestDistances[index] > currDist) {
                index--;
            }
            if (index < k - 1) {
                for (int j = k - 1; j > index + 1; j--) {
                    nearestDistances[j] = nearestDistances[j - 1];
                    nearestInstances[j] = nearestInstances[j - 1];
                }
                nearestInstances[index + 1] = i;
                nearestDistances[index + 1] = currDist;
            }
        }
        // Perform the weighted vote.
        float[] classProbabilities = new float[numClasses];
        float probTotal = 0;
        for (int kIndex = 0; kIndex < k; kIndex++) {
            classProbabilities[trainingData.data.get(nearestInstances[kIndex]).
                    getCategory()] += instanceWeights[nearestInstances[kIndex]];
            probTotal += instanceWeights[nearestInstances[kIndex]];
        }
        // Normalize.
        if (probTotal == 0) {
            return new float[numClasses];
        } else {
            for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                classProbabilities[cIndex] /= probTotal;
            }
        }
        return classProbabilities;
    }

    @Override
    public int classify(DataInstance instance, float[] distToTraining)
            throws Exception {
        if (trainingData == null || instanceWeights == null
                || trainingData.size() != instanceWeights.length) {
            throw new Exception("Bad classifier initialization");
        }
        if (instance == null) {
            return -1;
        }
        // Find the kNN set.
        int[] nearestInstances = new int[k];
        float[] nearestDistances = new float[k];
        for (int kIndex = 0; kIndex < k; kIndex++) {
            nearestDistances[kIndex] = Float.MAX_VALUE;
            nearestInstances[kIndex] = -1;
        }
        float currDist;
        int index;
        for (int i = 0; i < trainingData.size(); i++) {
            currDist = distToTraining[i];
            index = k - 1;
            while (index >= 0 && nearestDistances[index] > currDist) {
                index--;
            }
            if (index < k - 1) {
                for (int j = k - 1; j > index + 1; j--) {
                    nearestDistances[j] = nearestDistances[j - 1];
                    nearestInstances[j] = nearestInstances[j - 1];
                }
                nearestInstances[index + 1] = i;
                nearestDistances[index + 1] = currDist;
            }
        }
        // Perform the weighted vote.
        float[] classProbabilities = new float[numClasses];
        float probTotal = 0;
        for (int kIndex = 0; kIndex < k; kIndex++) {
            classProbabilities[trainingData.data.get(nearestInstances[kIndex]).
                    getCategory()] += instanceWeights[nearestInstances[kIndex]];
            probTotal += instanceWeights[nearestInstances[kIndex]];
        }
        // Normalize.
        int maxClassIndex = -1;
        float maxProb = 0;
        if (probTotal == 0) {
            return 0;
        } else {
            for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                classProbabilities[cIndex] /= probTotal;
                if (classProbabilities[cIndex] >= maxProb) {
                    maxClassIndex = cIndex;
                    maxProb = classProbabilities[cIndex];
                }
            }
        }
        if (maxClassIndex < 0) {
            maxClassIndex = 0;
        }
        return maxClassIndex;
    }

    @Override
    public int classify(DataInstance instance, float[] distToTraining,
            int[] trNeighbors) throws Exception {
        float[] classProbs = classifyProbabilistically(instance,
                distToTraining, trNeighbors);
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
        if (trainingData == null || instanceWeights == null
                || trainingData.size() != instanceWeights.length) {
            throw new Exception("Bad classifier initialization");
        }
        if (instance == null) {
            return null;
        }
        float[] classProbabilities = new float[numClasses];
        float probTotal = 0;
        // Perform the weighted vote.
        for (int kIndex = 0; kIndex < k; kIndex++) {
            classProbabilities[trainingData.data.get(trNeighbors[kIndex]).
                    getCategory()] += instanceWeights[trNeighbors[kIndex]];
            probTotal += instanceWeights[trNeighbors[kIndex]];
        }
        // Normalize.
        if (probTotal == 0) {
            return new float[numClasses];
        } else {
            for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                classProbabilities[cIndex] /= probTotal;
            }
        }
        return classProbabilities;
    }
    
    @Override
    public int getNeighborhoodSize() {
        return k;
    }
}
