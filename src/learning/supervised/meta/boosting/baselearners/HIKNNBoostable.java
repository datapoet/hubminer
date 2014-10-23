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
import java.util.HashMap;
import learning.supervised.Category;
import learning.supervised.evaluation.ValidateableInterface;
import learning.supervised.interfaces.DistMatrixUserInterface;
import learning.supervised.interfaces.DistToPointsQueryUserInterface;
import learning.supervised.interfaces.NeighborPointsQueryUserInterface;
import learning.supervised.meta.boosting.BoostableClassifier;
import util.ArrayUtil;
import util.BasicMathUtil;

/**
 * This class implements the HIKNN algorithm that was proposed in the paper
 * titled: "Nearest Neighbor Voting in High Dimensional Data: Learning from Past
 * Occurrences" published in Computer Science and Information Systems in 2011.
 * The algorithm is an extension of h-FNN that gives preference to rare neighbor
 * points and uses some label information. This is an extension that supports
 * instance weights for boosting.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class HIKNNBoostable extends BoostableClassifier implements
        DistMatrixUserInterface, NSFUserInterface,
        DistToPointsQueryUserInterface, NeighborPointsQueryUserInterface,
        Serializable {
    
    private static final long serialVersionUID = 1L;

    private int k = 5;
    // NeighborSetFinder object for kNN calculations.
    private NeighborSetFinder nsf = null;
    private DataSet trainingData = null;
    private int numClasses = 0;
    // The distance weighting parameter.
    private float mValue = 2;
    private double[][] classDataKNeighborRelation = null;
    // Information contained in the neighbors' labels.
    private float[] labelInformationFactor = null;
    // The prior class distribution.
    private float[] classPriors = null;
    private float laplaceEstimator = 0.001f;
    private int[] neighborOccurrenceFreqs = null;
    private double[] neighborOccurrenceFreqsWeighted = null;
    // The distance matrix.
    private float[][] distMat;
    private boolean noRecalc = true;
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
        paramMap.put("mValue", "Exponent for distance weighting. Defaults"
                + " to 2.");
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
    public String getName() {
        return "HIKNN";
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

    /**
     * The default constructor.
     */
    public HIKNNBoostable() {
    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     */
    public HIKNNBoostable(int k) {
        this.k = k;
    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     * @param cmet CombinedMetric object for distance calculations.
     */
    public HIKNNBoostable(int k, CombinedMetric cmet) {
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
    public HIKNNBoostable(int k, CombinedMetric cmet, int numClasses) {
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
     * @param boostingMode Integer that is the current boosting mode.
     */
    public HIKNNBoostable(int k, CombinedMetric cmet, int numClasses,
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
     * @param laplaceEstimator Float that is the Laplace estimator for smoothing
     * the probability distributions.
     */
    public HIKNNBoostable(int k, float laplaceEstimator) {
        this.k = k;
        this.laplaceEstimator = laplaceEstimator;
    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     * @param laplaceEstimator Float that is the Laplace estimator for smoothing
     * the probability distributions.
     * @param cmet CombinedMetric object for distance calculations.
     */
    public HIKNNBoostable(int k, float laplaceEstimator, CombinedMetric cmet) {
        this.k = k;
        this.laplaceEstimator = laplaceEstimator;
        setCombinedMetric(cmet);
    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     * @param laplaceEstimator Float that is the Laplace estimator for smoothing
     * the probability distributions.
     * @param cmet CombinedMetric object for distance calculations.
     * @param numClasses Integer that is the number of classes.
     */
    public HIKNNBoostable(int k, float laplaceEstimator, CombinedMetric cmet,
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
     * @param numClasses Integer that is the number of classes.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Integer that is the neighborhood size.
     */
    public HIKNNBoostable(DataSet dset, int numClasses, CombinedMetric cmet,
            int k) {
        trainingData = dset;
        this.numClasses = numClasses;
        setCombinedMetric(cmet);
        this.k = k;
    }

    /**
     * Initialization.
     *
     * @param dset DataSet object that is the training data.
     * @param numClasses Integer that is the number of classes.
     * @param nsf NeighborSetFinder object for kNN calculations.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Integer that is the neighborhood size.
     */
    public HIKNNBoostable(DataSet dset, int numClasses, NeighborSetFinder nsf,
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
    public HIKNNBoostable(Category[] categories, CombinedMetric cmet, int k) {
        int totalSize = 0;
        int indexFirstNonEmptyClass = -1;
        for (int i = 0; i < categories.length; i++) {
            totalSize += categories[i].size();
            if (indexFirstNonEmptyClass == -1 && categories[i].size() > 0) {
                indexFirstNonEmptyClass = i;
            }
        }
        // As this is an internal DataSet object, data instances will not have
        // it set as their data context.
        trainingData = new DataSet();
        trainingData.fAttrNames = categories[indexFirstNonEmptyClass].
                getInstance(0).getEmbeddingDataset().fAttrNames;
        trainingData.iAttrNames = categories[indexFirstNonEmptyClass].
                getInstance(0).getEmbeddingDataset().iAttrNames;
        trainingData.sAttrNames = categories[indexFirstNonEmptyClass].
                getInstance(0).getEmbeddingDataset().sAttrNames;
        trainingData.data = new ArrayList<>(totalSize);
        for (int i = 0; i < categories.length; i++) {
            for (int j = 0; j < categories[i].size(); j++) {
                categories[i].getInstance(j).setCategory(i);
                trainingData.addDataInstance(categories[i].getInstance(j));
            }
        }
        setCombinedMetric(cmet);
        this.k = k;
    }

    /**
     * @param laplaceEstimator Float that is the Laplace estimator for smoothing
     * the probability distributions.
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
        // An internal training data representation.
        trainingData = new DataSet();
        trainingData.fAttrNames = categories[
                indexFirstNonEmptyClass].
                getInstance(0).getEmbeddingDataset().fAttrNames;
        trainingData.iAttrNames =
                categories[indexFirstNonEmptyClass]
                .getInstance(0).getEmbeddingDataset().iAttrNames;
        trainingData.sAttrNames =
                categories[indexFirstNonEmptyClass].getInstance(0).
                getEmbeddingDataset().sAttrNames;
        trainingData.data = new ArrayList<>(totalSize);
        for (int cIndex = 0; cIndex < categories.length; cIndex++) {
            for (int j = 0; j < categories[cIndex].size(); j++) {
                categories[cIndex].getInstance(j).setCategory(cIndex);
                trainingData.addDataInstance(categories[cIndex].getInstance(j));
            }
        }
    }

    /**
     * @param trainingData DataSet object to train the model on.
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
        HIKNNBoostable classifierCopy = new HIKNNBoostable(k, laplaceEstimator,
                getCombinedMetric(), numClasses);
        classifierCopy.noRecalc = noRecalc;
        classifierCopy.boostingMode = boostingMode;
        return classifierCopy;
    }

    @Override
    public void train() throws Exception {
        if (nsf == null) {
            // If the kNN sets have not been provided, calculate them.
            calculateNeighborSets();
        }
        // Find the class priors.
        classPriors = trainingData.getClassPriors();
        if (!noRecalc) {
            nsf.recalculateStatsForSmallerK(k);
        }
        // Set default values for instance weights if none have been provided.
        if (instanceWeights == null) {
            instanceWeights = new double[trainingData.size()];
            Arrays.fill(instanceWeights, 1d);
        }
        if (instanceLabelWeights == null) {
            instanceLabelWeights = new double[trainingData.size()][numClasses];
        }
        // Get the neighbor occurrence frequencies. Despite the instance
        // weights, non-weighted total frequencies are used for information
        // content estimation of different occurrences.
        neighborOccurrenceFreqs = nsf.getNeighborFrequencies();
        // Weighted occurrences are used for vote normalization.
        neighborOccurrenceFreqsWeighted = new double[trainingData.size()];
        // Find the class-conditional neighbor occurrence frequencies.
        int[][] kneighbors = nsf.getKNeighbors();
        classDataKNeighborRelation =
                new double[numClasses][trainingData.size()];
        labelInformationFactor = new float[trainingData.size()];
        int currClass;
        float maxHubness = 0;
        float minHubness = Float.MAX_VALUE;
        for (int i = 0; i < trainingData.size(); i++) {
            if (neighborOccurrenceFreqs[i] > maxHubness) {
                maxHubness = neighborOccurrenceFreqs[i];
            }
            if (neighborOccurrenceFreqs[i] < minHubness) {
                minHubness = neighborOccurrenceFreqs[i];
            }
        }
        // Calculate the neighbor occurrence informativeness and the
        // class-conditional neighbor occurrence frequencies.
        float eventInfo;
        float minEventInfo;
        float maxEventInfo;
        minEventInfo = (float) BasicMathUtil.log2(
                ((float) trainingData.size()) / (maxHubness + 1f));
        maxEventInfo = (float) BasicMathUtil.log2(
                ((float) trainingData.size()) / (0 + 1f));
        for (int i = 0; i < trainingData.size(); i++) {
            eventInfo = (float) BasicMathUtil.log2(
                    ((float) trainingData.size())
                    / ((float) neighborOccurrenceFreqs[i] + 1f));
            labelInformationFactor[i] = (eventInfo - minEventInfo)
                    / (maxEventInfo - minEventInfo + 0.0001f);
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
        for (int i = 0; i < numClasses; i++) {
            if (classProbs[i] > maxProb) {
                maxProb = classProbs[i];
                result = i;
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
        for (int i = 0; i < k; i++) {
            kDistances[i] = Float.MAX_VALUE;
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
        float[] distance_weights = new float[k];
        float dwSum = 0;
        for (int i = 0; i < k; i++) {
            if (kDistances[i] != 0) {
                distance_weights[i] = 1f / ((float) Math.pow(kDistances[i],
                        (2f / (mValue - 1f))));
            } else {
                distance_weights[i] = 10000f;
            }
            dwSum += distance_weights[i];
        }
        float[] classProbEstimates = new float[numClasses];
        // Perform the voting.
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < numClasses; j++) {
                if (trainingData.data.get(kNeighbors[i]).getCategory() == j) {
                    classProbEstimates[j] +=
                            ((labelInformationFactor[kNeighbors[i]]
                            + ((1 - labelInformationFactor[kNeighbors[i]])
                            * classDataKNeighborRelation[j][kNeighbors[i]]))
                            * (float) BasicMathUtil.log2(
                            ((float) trainingData.size())
                            / (1f + neighborOccurrenceFreqs[kNeighbors[i]])))
                            * distance_weights[i] / dwSum;
                } else {
                    classProbEstimates[j] +=
                            ((((1 - labelInformationFactor[kNeighbors[i]]))
                            * classDataKNeighborRelation[j][kNeighbors[i]])
                            * (float) BasicMathUtil.log2(
                            ((float) trainingData.size())
                            / (1f + neighborOccurrenceFreqs[kNeighbors[i]])))
                            * distance_weights[i] / dwSum;
                }
            }
        }
        // Normalize.
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
        // Find the k-nearest neighbors.
        float[] kDistances = new float[k];
        for (int i = 0; i < k; i++) {
            kDistances[i] = Float.MAX_VALUE;
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
        float[] distance_weights = new float[k];
        float dwSum = 0;
        for (int i = 0; i < k; i++) {
            if (kDistances[i] != 0) {
                distance_weights[i] = 1f / ((float) Math.pow(kDistances[i],
                        (2f / (mValue - 1f))));
            } else {
                distance_weights[i] = 10000f;
            }
            dwSum += distance_weights[i];
        }
        float[] classProbEstimates = new float[numClasses];
        // Perform the voting.
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < numClasses; j++) {
                if (trainingData.data.get(kNeighbors[i]).getCategory() == j) {
                    classProbEstimates[j] +=
                            ((labelInformationFactor[kNeighbors[i]]
                            + ((1 - labelInformationFactor[kNeighbors[i]])
                            * classDataKNeighborRelation[j][kNeighbors[i]]))
                            * (float) BasicMathUtil.log2(
                            ((float) trainingData.size())
                            / (1f + neighborOccurrenceFreqs[kNeighbors[i]])))
                            * distance_weights[i] / dwSum;
                } else {
                    classProbEstimates[j] += ((((1
                            - labelInformationFactor[kNeighbors[i]]))
                            * classDataKNeighborRelation[j][kNeighbors[i]])
                            * (float) BasicMathUtil.log2(
                            ((float) trainingData.size())
                            / (1f + neighborOccurrenceFreqs[kNeighbors[i]])))
                            * distance_weights[i] / dwSum;
                }
            }

        }
        // Normalize the probabilities.
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
        float[] distance_weights = new float[k];
        float dwSum = 0;
        for (int i = 0; i < k; i++) {
            if (distToTraining[trNeighbors[i]] != 0) {
                distance_weights[i] = 1f / ((float) Math.pow(
                        distToTraining[trNeighbors[i]], (2f / (mValue - 1f))));
            } else {
                distance_weights[i] = 10000f;
            }
            dwSum += distance_weights[i];
        }
        float[] classProbEstimates = new float[numClasses];
        for (int i = 0; i < numClasses; i++) {
            classProbEstimates[i] = 0;
        }
        // Perform the voting.
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < numClasses; j++) {
                if (trainingData.data.get(trNeighbors[i]).getCategory() == j) {
                    classProbEstimates[j] +=
                            ((labelInformationFactor[trNeighbors[i]]
                            + ((1 - labelInformationFactor[trNeighbors[i]])
                            * classDataKNeighborRelation[j][trNeighbors[i]]))
                            * (float) BasicMathUtil.log2(
                            ((float) trainingData.size())
                            / (1f + neighborOccurrenceFreqs[trNeighbors[i]])))
                            * distance_weights[i] / dwSum;
                } else {
                    classProbEstimates[j] +=
                            ((((1 - labelInformationFactor[trNeighbors[i]]))
                            * classDataKNeighborRelation[j][trNeighbors[i]])
                            * (float) BasicMathUtil.log2(
                            ((float) trainingData.size())
                            / (1f + neighborOccurrenceFreqs[trNeighbors[i]])))
                            * distance_weights[i] / dwSum;
                }
            }
        }
        // Normalize.
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

    /**
     * Classify a new instance.
     *
     * @param instance DataInstance object to classify.
     * @param kDists float[] of distances to the k-nearest neighbors.
     * @param trNeighbors int[] holding the indexes of the k-nearest neighbors.
     * @return float[] of class affiliation probabilities.
     * @throws Exception
     */
    public float[] classifyProbabilisticallyWithKDistAndNeighbors(
            DataInstance instance, float[] kDists, int[] trNeighbors)
            throws Exception {
        // Calculate the distance weights.
        float[] distance_weights = new float[k];
        float dwSum = 0;
        for (int i = 0; i < k; i++) {
            if (kDists[i] != 0) {
                distance_weights[i] = 1f
                        / ((float) Math.pow(kDists[i], (2f / (mValue - 1f))));
            } else {
                distance_weights[i] = 10000f;
            }
            dwSum += distance_weights[i];
        }
        float[] classProbEstimates = new float[numClasses];
        for (int i = 0; i < numClasses; i++) {
            classProbEstimates[i] = 0;
        }
        // Perform the voting.
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < numClasses; j++) {
                if (trainingData.data.get(trNeighbors[i]).getCategory() == j) {
                    classProbEstimates[j] +=
                            ((labelInformationFactor[trNeighbors[i]]
                            + ((1 - labelInformationFactor[trNeighbors[i]])
                            * classDataKNeighborRelation[j][trNeighbors[i]]))
                            * (float) BasicMathUtil.log2(
                            ((float) trainingData.size())
                            / (1f + neighborOccurrenceFreqs[trNeighbors[i]])))
                            * distance_weights[i] / dwSum;
                } else {
                    classProbEstimates[j] +=
                            ((((1 - labelInformationFactor[trNeighbors[i]]))
                            * classDataKNeighborRelation[j][trNeighbors[i]])
                            * (float) BasicMathUtil.log2(
                            ((float) trainingData.size())
                            / (1f + neighborOccurrenceFreqs[trNeighbors[i]])))
                            * distance_weights[i] / dwSum;
                }
            }
        }
        // Normalize the probabilities.
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
