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
import data.representation.util.DataMineConstants;
import distances.primary.CombinedMetric;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import learning.supervised.Category;
import learning.supervised.evaluation.ValidateableInterface;
import learning.supervised.interfaces.DistMatrixUserInterface;
import learning.supervised.interfaces.DistToPointsQueryUserInterface;
import learning.supervised.interfaces.NeighborPointsQueryUserInterface;
import learning.supervised.meta.boosting.BoostableClassifier;
import util.ArrayUtil;

/**
 * This paper implements the h-FNN algorithm that was proposed in the following
 * paper: "Hubness-based Fuzzy Measures for High-dimensional K-Nearest-Neighbor
 * classification" at MLDM 2011. It is reverse-neighbor-based extension of the
 * classical FNN classifier. This is an extension that supports instance weights
 * for boosting.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class HFNNBoostable extends BoostableClassifier implements
        DistMatrixUserInterface, NSFUserInterface,
        DistToPointsQueryUserInterface, NeighborPointsQueryUserInterface,
        Serializable {
    
    private static final long serialVersionUID = 1L;

    // The parameter that defines when the global rule should not be fired due
    // to little data, so a special routine for anti-hub handling is invoked.
    private int thetaCutoff = 0;
    // Neighborhood size.
    private int k = 5;
    // Object that stores and calculates the kNN sets.
    private NeighborSetFinder nsf = null;
    // Training data to learn from.
    private DataSet trainingData = null;
    // Number of classes in the data.
    private int numClasses = 0;
    // Class-conditional neighbor occurrence frequencies.
    private double[][] classDataKNeighborRelation = null;
    // Prior class counts or probabilities.
    private float[] classPriors = null;
    // Laplace estimator for probability distribution smoothing.
    private float laplaceEstimator = 0.0000000000000000000000000000000001f;
    // Neighbor occurrence frequencies of training data points.
    private int[] neighborOccurrenceFreqs = null;
    private double[] neighborOccurrenceFreqsWeighted = null;
    // Local estimate for low hubness.
    private double[][] localCrispClassDistribution = null;
    private float[][] distMat = null;
    private boolean noRecalc = true;
    ;

    
    // Boosting weights.
    private double[] instanceWeights;
    private double[][] instanceLabelWeights;
    // Boosting mode.
    public static final int B1 = 0;
    public static final int B2 = 1;
    private int boostingMode = B1;

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
    public void setDistMatrix(float[][] distMatrix) {
        this.distMat = distMatrix;
    }

    @Override
    public float[][] getDistMatrix() {
        return distMat;
    }

    @Override
    public void noRecalcs() {
        noRecalc = true;
    }

    @Override
    public String getName() {
        return "dwh-FNN";
    }

    /**
     * The default constructor.
     */
    public HFNNBoostable() {
    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     */
    public HFNNBoostable(int k) {
        this.k = k;
    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     * @param cmet CombinedMetric object for distance calculations.
     */
    public HFNNBoostable(int k, CombinedMetric cmet) {
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
    public HFNNBoostable(int k, CombinedMetric cmet, int numClasses) {
        this.k = k;
        setCombinedMetric(cmet);
        this.numClasses = numClasses;
    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     * @param cmet CombinedMetric object for distance calculations.
     * @param numClasses Integer that is the number of classes in the data.
     * @param boostingMode Integer that is the current boosting mode.
     */
    public HFNNBoostable(int k, CombinedMetric cmet, int numClasses,
            int boostingMode) {
        this.k = k;
        setCombinedMetric(cmet);
        this.numClasses = numClasses;
        this.boostingMode = boostingMode;
    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     * @param laplaceEstimator for probability distribution smoothing.
     */
    public HFNNBoostable(int k, float laplaceEstimator) {
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
    public HFNNBoostable(int k, float laplaceEstimator, CombinedMetric cmet) {
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
    public HFNNBoostable(int k, float laplaceEstimator, CombinedMetric cmet,
            int numClasses) {
        this.k = k;
        this.laplaceEstimator = laplaceEstimator;
        setCombinedMetric(cmet);
        this.numClasses = numClasses;
    }

    /**
     * Initialization.
     *
     * @param dset DataSet to train the model on.
     * @param numClasses Integer that is the number of classes in the data.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Integer that is the neighborhood size.
     */
    public HFNNBoostable(DataSet dset, int numClasses, CombinedMetric cmet,
            int k) {
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
    public HFNNBoostable(DataSet dset, int numClasses, NeighborSetFinder nsf,
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
    public HFNNBoostable(Category[] categories, CombinedMetric cmet, int k) {
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
        HFNNBoostable classifierCopy = new HFNNBoostable(k,
                laplaceEstimator, getCombinedMetric(), numClasses);
        classifierCopy.noRecalc = noRecalc;
        classifierCopy.thetaCutoff = thetaCutoff;
        classifierCopy.boostingMode = boostingMode;
        return classifierCopy;
    }

    @Override
    public void train() throws Exception {
        if (nsf == null) {
            calculateNeighborSets();
        }
        // Calculate class priors.
        classPriors = trainingData.getClassPriors();
        // Set default values for instance weights if none have been provided.
        if (instanceWeights == null) {
            instanceWeights = new double[trainingData.size()];
            Arrays.fill(instanceWeights, 1d);
        }
        if (instanceLabelWeights == null) {
            instanceLabelWeights = new double[trainingData.size()][numClasses];
        }
        localCrispClassDistribution =
                new double[trainingData.size()][numClasses];
        neighborOccurrenceFreqs = nsf.getNeighborFrequencies();
        // Weighted occurrences are used for vote normalization.
        neighborOccurrenceFreqsWeighted = new double[trainingData.size()];
        // Get the kNN sets.
        int[][] kneighbors = nsf.getKNeighbors();
        classDataKNeighborRelation =
                new double[numClasses][trainingData.size()];
        int currClass;
        // Iterate over the training data.
        for (int i = 0; i < trainingData.size(); i++) {
            // Calculate the class-conditional neighbor occurrence frequencies.
            currClass = trainingData.data.get(i).getCategory();
            if (boostingMode == B1) {
                neighborOccurrenceFreqsWeighted[i] += instanceWeights[i];
                classDataKNeighborRelation[currClass][i] += instanceWeights[i];
                for (int kIndex = 0; kIndex < k; kIndex++) {
                    neighborOccurrenceFreqsWeighted[kneighbors[i][kIndex]] +=
                            instanceWeights[i];
                    classDataKNeighborRelation[
                            currClass][kneighbors[i][kIndex]] +=
                            instanceWeights[i];
                }
            } else {
                // B2 boosting.
                classDataKNeighborRelation[currClass][i] += instanceWeights[i];
                for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                    if (cIndex != currClass) {
                        classDataKNeighborRelation[cIndex][i] -=
                                instanceLabelWeights[i][cIndex]
                                * instanceWeights[i];
                    }
                }
                for (int kIndex = 0; kIndex < k; kIndex++) {
                    classDataKNeighborRelation[
                            currClass][kneighbors[i][kIndex]] +=
                            instanceWeights[i];
                    for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                        if (cIndex != currClass) {
                            classDataKNeighborRelation[cIndex][
                                    kneighbors[i][kIndex]] -=
                                    instanceLabelWeights[i][cIndex]
                                    * instanceWeights[i];
                        }
                    }
                }
            }
            if (neighborOccurrenceFreqs[i] <= thetaCutoff) {
                // The special anti-hub handling case.
                if (boostingMode == B1) {
                    for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                        if (trainingData.data.get(i).getCategory() == cIndex) {
                            localCrispClassDistribution[i][cIndex] =
                                    (instanceWeights[i] + laplaceEstimator)
                                    / (instanceWeights[i]
                                    + (numClasses * laplaceEstimator));
                        } else {
                            localCrispClassDistribution[i][cIndex] =
                                    (laplaceEstimator) / (instanceWeights[i]
                                    + (numClasses * laplaceEstimator));
                        }
                    }
                } else {
                    localCrispClassDistribution[i] = new double[numClasses];
                    localCrispClassDistribution[i][trainingData.data.get(i).
                            getCategory()] = instanceWeights[i];
                    for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                        if (trainingData.data.get(i).getCategory() != cIndex) {
                            localCrispClassDistribution[i][cIndex] -=
                                    instanceLabelWeights[i][cIndex]
                                    * instanceWeights[i];
                        }
                    }
                    double minValue = Double.MAX_VALUE;
                    for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                        minValue = Math.min(minValue,
                                localCrispClassDistribution[i][cIndex]);
                    }
                    double denominator = 0;
                    if (minValue < 0) {
                        for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                            localCrispClassDistribution[i][cIndex] +=
                                    Math.abs(minValue);
                            denominator +=
                                    localCrispClassDistribution[i][cIndex];
                        }
                    }
                    for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                        localCrispClassDistribution[i][cIndex] /= denominator;
                    }
                }
            }
        }
        // Normalization.
        if (boostingMode == B1) {
            for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                for (int j = 0; j < trainingData.size(); j++) {
                    if (neighborOccurrenceFreqsWeighted[j] > 0) {
                        double backupValue =
                                classDataKNeighborRelation[cIndex][j];
                        classDataKNeighborRelation[cIndex][j] /=
                                (neighborOccurrenceFreqsWeighted[j]);
                        if (!DataMineConstants.isAcceptableDouble(
                                classDataKNeighborRelation[cIndex][j])) {
                            classDataKNeighborRelation[cIndex][j] = backupValue;
                        }
                    }
                }
            }
        } else {
            for (int j = 0; j < trainingData.size(); j++) {
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
        float[] classProbEstimates = new float[numClasses];
        // Perform the voting.
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
                    classProbEstimates[cIndex] +=
                            localCrispClassDistribution[
                            kNeighbors[kIndex]][cIndex];
                }
            }
        }
        // Normalization.
        float minVal = ArrayUtil.min(classProbEstimates);
        for (int cIndex = 0; cIndex < numClasses; cIndex++) {
            if (minVal < 0) {
                classProbEstimates[cIndex] -= minVal;
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
        float[] classProbEstimates = new float[numClasses];
        for (int kIndex = 0; kIndex < k; kIndex++) {
            if (neighborOccurrenceFreqs[kNeighbors[kIndex]] > thetaCutoff) {
                // The normal case.
                for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                    classProbEstimates[cIndex] += classDataKNeighborRelation[
                            cIndex][kNeighbors[kIndex]];
                }
            } else {
                // The special anti-hub handling case.
                for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                    classProbEstimates[cIndex] +=
                            localCrispClassDistribution[
                            kNeighbors[kIndex]][cIndex];
                }
            }
        }
        // Normalization.
        float minVal = ArrayUtil.min(classProbEstimates);
        for (int cIndex = 0; cIndex < numClasses; cIndex++) {
            if (minVal < 0) {
                classProbEstimates[cIndex] -= minVal;
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
        // Perform the voting.
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
                    classProbEstimates[cIndex] +=
                            localCrispClassDistribution[
                            trNeighbors[kIndex]][cIndex];
                }
            }
        }
        // Normalization.
        float minVal = ArrayUtil.min(classProbEstimates);
        for (int cIndex = 0; cIndex < numClasses; cIndex++) {
            if (minVal < 0) {
                classProbEstimates[cIndex] -= minVal;
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
    public int getNeighborhoodSize() {
        return k;
    }
}
