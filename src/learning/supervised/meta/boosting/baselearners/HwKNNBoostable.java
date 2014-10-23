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
package learning.supervised.meta.boosting.baselearners;

import data.neighbors.NSFUserInterface;
import data.neighbors.NeighborSetFinder;
import data.representation.DataInstance;
import data.representation.DataSet;
import distances.primary.CombinedMetric;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import learning.supervised.Category;
import learning.supervised.evaluation.ValidateableInterface;
import learning.supervised.interfaces.DistMatrixUserInterface;
import learning.supervised.interfaces.DistToPointsQueryUserInterface;
import learning.supervised.interfaces.NeighborPointsQueryUserInterface;
import learning.supervised.meta.boosting.BoostableClassifier;
import util.ArrayUtil;

/**
 * This class implements the hw-kNN method for hubness-based instance weighting
 * in kNN classification. This approach was proposed in the following paper:
 * "Nearest Neighbors in High-Dimensional Data: The Emergence and Influence of
 * Hubs" at ICML 2009.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class HwKNNBoostable extends BoostableClassifier implements
        DistMatrixUserInterface, NSFUserInterface,
        DistToPointsQueryUserInterface, NeighborPointsQueryUserInterface,
        Serializable {

    private static final long serialVersionUID = 1L;
    private DataSet trainingData = null;
    private int numClasses = 0;
    private float[] instanceHubnessWeights = null;
    private float[][] distMat = null;
    private NeighborSetFinder nsf = null;
    private int k = 5;
    // Boosting weights.
    private double[] instanceWeights;
    private double[][] instanceLabelWeights;
    // Boosting mode.
    public static final int B1 = 0;
    public static final int B2 = 1;
    private int boostingMode = B1;
    
    @Override
    public HashMap<String, String> getParameterNamesAndDescriptions() {
        HashMap<String, String> paramMap = new HashMap<>();
        paramMap.put("k", "Neighborhood size.");
        paramMap.put("boostingMode", "Type of re-weighting procedure.");
        return paramMap;
    }
    
    @Override
    public long getVersion() {
        return serialVersionUID;
    }

    /**
     * @param boostingMode Integer that is the current boosting mode: B1 or B2.
     */
    public void setBoostingMode(int boostingMode) {
        this.boostingMode = boostingMode;
    }

    @Override
    public void setTotalInstanceWeights(double[] instanceWeights) {
        this.instanceWeights = instanceWeights;
    }

    @Override
    public void setMisclassificationCostDistribution(
            double[][] instanceLabelWeights) {
        this.instanceLabelWeights = instanceLabelWeights;
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

    /**
     * Initialization.
     *
     * @param numClasses Integer that is the number of classes.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Integer that is the neighborhood size.
     */
    public HwKNNBoostable(int numClasses, CombinedMetric cmet, int k) {
        this.numClasses = numClasses;
        if (trainingData != null) {
            instanceHubnessWeights = new float[trainingData.size()];
        }
        setCombinedMetric(cmet);
        this.k = k;
    }

    /**
     * Initialization.
     *
     * @param numClasses Integer that is the number of classes.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Integer that is the neighborhood size.
     * @param boostingMode Integer that corresponds to B1 or B2 boosting.
     */
    public HwKNNBoostable(int numClasses, CombinedMetric cmet, int k,
            int boostingMode) {
        this.numClasses = numClasses;
        if (trainingData != null) {
            instanceHubnessWeights = new float[trainingData.size()];
        }
        setCombinedMetric(cmet);
        this.k = k;
        this.boostingMode = boostingMode;
    }

    /**
     * Initialization.
     *
     * @param dset DataSet object representing the training data.
     * @param numClasses Integer that is the number of classes.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Integer that is the neighborhood size.
     */
    public HwKNNBoostable(DataSet dset, int numClasses, CombinedMetric cmet,
            int k) {
        trainingData = dset;
        this.numClasses = numClasses;
        if (trainingData != null) {
            instanceHubnessWeights = new float[trainingData.size()];
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
    public HwKNNBoostable(DataSet dset, int numClasses, float[] weights,
            CombinedMetric cmet, int k) {
        trainingData = dset;
        this.numClasses = numClasses;
        this.instanceHubnessWeights = weights;
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
    public HwKNNBoostable(Category[] categories, CombinedMetric cmet, int k) {
        setClasses(categories);
        instanceHubnessWeights = new float[trainingData.size()];
        setCombinedMetric(cmet);
        this.k = k;
    }

    @Override
    public ValidateableInterface copyConfiguration() {
        return new HwKNNBoostable(numClasses, getCombinedMetric(), k,
                boostingMode);
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
        for (int cFirst = 0; cFirst < categories.length; cFirst++) {
            for (int cSecond = 0; cSecond < categories[cFirst].size();
                    cSecond++) {
                categories[cFirst].getInstance(cSecond).setCategory(cFirst);
                trainingData.addDataInstance(categories[cFirst].
                        getInstance(cSecond));
            }
        }
        instanceHubnessWeights = new float[trainingData.size()];
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
    public void setHubnessWeights(float[] instanceHubnessWeights) {
        this.instanceHubnessWeights = instanceHubnessWeights;
    }

    /**
     * @return float[] of instance weights.
     */
    public float[] getHubnessWeights() {
        return instanceHubnessWeights;
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
        int[][] kneighbors = nsf.getKNeighbors();
        int numInstances = kneighbors.length;
        double[] weightedBadHubness = new double[numInstances];
        instanceHubnessWeights = new float[numInstances];
        // Only used in B2 boosting.
        double[][] classDataKNeighborRelation =
                new double[numClasses][numInstances];
        for (int i = 0; i < numInstances; i++) {
            int label = trainingData.getLabelOf(i);
            if (boostingMode == B2) {
                classDataKNeighborRelation[label][i] += instanceWeights[i];
                for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                    if (cIndex != label) {
                        classDataKNeighborRelation[cIndex][i] -=
                                instanceLabelWeights[i][cIndex]
                                * instanceWeights[i];
                    }
                }
            }
            for (int kIndex = 0; kIndex < k; kIndex++) {
                int neighborLabel = trainingData.getLabelOf(
                        kneighbors[i][kIndex]);
                if (boostingMode == B1) {
                    if (label != neighborLabel) {
                        weightedBadHubness[kneighbors[i][kIndex]] +=
                                instanceWeights[i];
                    }
                } else {
                    // In case of B2 boosting.
                    classDataKNeighborRelation[
                            label][kneighbors[i][kIndex]] +=
                            instanceWeights[i];
                    for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                        if (cIndex != label) {
                            classDataKNeighborRelation[cIndex][
                                    kneighbors[i][kIndex]] -=
                                    instanceLabelWeights[i][cIndex]
                                    * instanceWeights[i];
                        }
                    }
                }
            }
        }
        if (boostingMode == B2) {
            weightedBadHubness = new double[numInstances];
            for (int j = 0; j < trainingData.size(); j++) {
                int label = trainingData.getLabelOf(j);
                double minValue = Double.MAX_VALUE;
                for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                    minValue = Math.min(minValue,
                            classDataKNeighborRelation[cIndex][j]);
                }
                double denominator = 0;
                if (minValue < 0) {
                    for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                        classDataKNeighborRelation[cIndex][j] +=
                                Math.abs(minValue);
                        denominator += classDataKNeighborRelation[cIndex][j];
                    }
                }
                for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                    classDataKNeighborRelation[cIndex][j] /= denominator;
                    if (cIndex != label) {
                        weightedBadHubness[j] +=
                                classDataKNeighborRelation[cIndex][j];
                    }
                }
            }
        }
        double[] standardizedWeightedBHubness = weightedBadHubness.clone();
        ArrayUtil.zStandardize(standardizedWeightedBHubness);
        for (int i = 0; i < numInstances; i++) {
            instanceHubnessWeights[i] =
                    (float) Math.exp(-standardizedWeightedBHubness[i]);
        }
    }

    @Override
    public void train() throws Exception {
        getWeightsFromHubness();
    }

    @Override
    public int classify(DataInstance instance) throws Exception {
        CombinedMetric cmet = getCombinedMetric();
        if (trainingData == null || instanceHubnessWeights == null
                || trainingData.size() != instanceHubnessWeights.length) {
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
                    getCategory()] +=
                    instanceHubnessWeights[nearestInstances[kIndex]];
            probTotal += instanceHubnessWeights[nearestInstances[kIndex]];
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
        if (trainingData == null || instanceHubnessWeights == null
                || trainingData.size() != instanceHubnessWeights.length) {
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
                    instanceHubnessWeights[nearestInstances[kIndex]];
            probTotal += instanceHubnessWeights[nearestInstances[kIndex]];
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
        if (trainingData == null || instanceHubnessWeights == null
                || trainingData.size() != instanceHubnessWeights.length) {
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
                    getCategory()] +=
                    instanceHubnessWeights[nearestInstances[kIndex]];
            probTotal += instanceHubnessWeights[nearestInstances[kIndex]];
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
        if (trainingData == null || instanceHubnessWeights == null
                || trainingData.size() != instanceHubnessWeights.length) {
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
                    getCategory()] +=
                    instanceHubnessWeights[nearestInstances[kIndex]];
            probTotal += instanceHubnessWeights[nearestInstances[kIndex]];
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
        if (trainingData == null || instanceHubnessWeights == null
                || trainingData.size() != instanceHubnessWeights.length) {
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
                    getCategory()] +=
                    instanceHubnessWeights[trNeighbors[kIndex]];
            probTotal += instanceHubnessWeights[trNeighbors[kIndex]];
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
