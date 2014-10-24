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
package learning.supervised.methods;

import combinatorial.Permutation;
import data.representation.DataInstance;
import data.representation.DataSet;
import data.representation.util.DataMineConstants;
import distances.primary.CombinedMetric;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import learning.supervised.Category;
import learning.supervised.Classifier;
import learning.supervised.evaluation.ValidateableInterface;
import sampling.UniformSampler;
import statistics.HigherMoments;

/**
 * This class implements the robust stochastic learning vector quantization
 * classification method.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class RSLVQ extends Classifier implements Serializable {
    
    private static final long serialVersionUID = 1L;

    private DataSet trainingData = null;
    private int numClasses = 0;
    // Each class is represented by a number of prototypes.
    private DataInstance[][] classPrototypes;
    private float[][] protoDispersions;
    // Number of prototypes per class.
    private int numProtoPerClass = 5;
    //the learning rates
    float alphaProto = 0.5f;
    float alphaVariance = 0.3f;
    
    @Override
    public HashMap<String, String> getParameterNamesAndDescriptions() {
        HashMap<String, String> paramMap = new HashMap<>();
        paramMap.put("alphaProto", "Learning rate.");
        paramMap.put("alphaVariance", "Learning rate.");
        return paramMap;
    }
    
    @Override
    public long getVersion() {
        return serialVersionUID;
    }

    @Override
    public String getName() {
        return "Robust Stochastic Learning Vector Quantization";
    }
    
    /**
     * Default constructor.
     */
    public RSLVQ() {
    }

    /**
     * Initialization.
     *
     * @param dataClasses Category[] representing the training data.
     */
    public RSLVQ(Category[] dataClasses) {
        setClasses(dataClasses);
        numClasses = dataClasses.length;
    }

    /**
     * Initialization.
     *
     * @param dset DataSet object representing the training data.
     * @param numClasses Integer that is the number of classes in the data.
     */
    public RSLVQ(DataSet dset, int numClasses) {
        this.trainingData = dset;
        this.numClasses = numClasses;
        setData(dset.data, dset);
    }

    /**
     * Initialization.
     *
     * @param dataClasses Category[] representing the training data.
     * @param cmet CombinedMetric object for distance calculations.
     */
    public RSLVQ(Category[] dataClasses, CombinedMetric cmet) {
        setClasses(dataClasses);
        numClasses = dataClasses.length;
        setCombinedMetric(cmet);
    }

    /**
     * Initialization.
     *
     * @param dset DataSet object representing the training data.
     * @param numClasses Integer that is the number of classes in the data.
     * @param cmet CombinedMetric object for distance calculations.
     */
    public RSLVQ(DataSet dset, int numClasses, CombinedMetric cmet) {
        this.trainingData = dset;
        this.numClasses = numClasses;
        setData(dset.data, dset);
        setCombinedMetric(cmet);
    }

    /**
     * Initialization.
     *
     * @param dset DataSet object representing the training data.
     * @param numClasses Integer that is the number of classes in the data.
     * @param cmet CombinedMetric object for distance calculations.
     * @param numProtoPerClass Integer that is the number of prototypes per
     * class to use.
     */
    public RSLVQ(DataSet dset, int numClasses, CombinedMetric cmet,
            int numProtoPerClass) {
        this.trainingData = dset;
        this.numClasses = numClasses;
        setData(dset.data, dset);
        setCombinedMetric(cmet);
        this.numProtoPerClass = numProtoPerClass;
    }

    /**
     * Initialization.
     *
     * @param numClasses Integer that is the number of classes in the data.
     * @param cmet CombinedMetric object for distance calculations.
     */
    public RSLVQ(int numClasses, CombinedMetric cmet) {
        this.numClasses = numClasses;
        setCombinedMetric(cmet);
    }

    /**
     * Initialization.
     *
     * @param numClasses Integer that is the number of classes in the data.
     * @param cmet CombinedMetric object for distance calculations.
     * @param numProtoPerClass Integer that is the number of prototypes per
     * class to use.
     */
    public RSLVQ(int numClasses, CombinedMetric cmet, int numProtoPerClass) {
        this.numClasses = numClasses;
        setCombinedMetric(cmet);
        this.numProtoPerClass = numProtoPerClass;
    }

    @Override
    public ValidateableInterface copyConfiguration() {
        RSLVQ classifierCopy = new RSLVQ(numClasses, getCombinedMetric());
        return classifierCopy;
    }

    @Override
    public void train() throws Exception {
        CombinedMetric cmet = getCombinedMetric();
        Category[] dataClasses = getClasses();
        int dim = trainingData.getNumFloatAttr();
        // Initialize the prototypes by random point selection.
        numClasses = dataClasses.length;
        numProtoPerClass = Math.min(50, Math.max(5,
                (trainingData.size() / (numClasses * 15))));
        classPrototypes = new DataInstance[numClasses][];
        protoDispersions = new float[numClasses][];
        // Distances to prototypes.
        float[][] protoDists = new float[numClasses][];
        // Exponential distances to prototypes.
        float[][] protoDistExps = new float[numClasses][];
        int tempInt;
        for (int c = 0; c < numClasses; c++) {
            // A class could in principle have fewer members than the desired
            // number of prototypes, so we have to be careful.
            int prLen = Math.max(numProtoPerClass, dataClasses[c].size());
            classPrototypes[c] = new DataInstance[prLen];
            protoDispersions[c] = new float[prLen];
            protoDists[c] = new float[prLen];
            protoDistExps[c] = new float[prLen];
            int[] indexes = UniformSampler.getSample(
                    dataClasses[c].size(), prLen);
            int[] varianceSample = UniformSampler.getSample(
                    dataClasses[c].size(), 20);
            float[] varDists = new float[(varianceSample.length *
                    (varianceSample.length - 1)) / 2];
            tempInt = -1;
            // Calculate the variance from the random sample.
            for (int i = 0; i < varianceSample.length; i++) {
                for (int j = i + 1; j < varianceSample.length; j++) {
                    varDists[++tempInt] = cmet.dist(
                            dataClasses[c].getInstance(varianceSample[i]),
                            dataClasses[c].getInstance(varianceSample[j]));
                }
            }
            float meanVal = HigherMoments.calculateArrayMean(varDists);
            float stDev = HigherMoments.calculateArrayStDev(meanVal,
                    varDists);
            float varianceEstimate = stDev * stDev;
            for (int i = 0; i < prLen; i++) {
                classPrototypes[c][i] = dataClasses[c].getInstance(
                        indexes[i]).copy();
                protoDispersions[c][i] = varianceEstimate;
            }
        }
        // Iterate through the data according to a random permutation.
        int[] indexPermutation = Permutation.obtainRandomPermutation(
                trainingData.size());
        // Current class and current distance.
        int currClass;
        float currDist;
        // Best selection parameters.
        float minDist;
        int minProtoIndex;
        int minClassIndex;

        float sumAll;
        float[] classSums = new float[numClasses];

        float choosingProbTotal;
        float choosingProbInClass;

        float deltaProtoFact;
        float deltaVariance;

        for (int i : indexPermutation) {
            // Find the closest prototype.
            currClass = trainingData.getLabelOf(i);
            DataInstance instance = trainingData.getInstance(i);
            minProtoIndex = 0;
            minDist = Float.MAX_VALUE;
            sumAll = 0;
            minClassIndex = 0;
            for (int c = 0; c < numClasses; c++) {
                classSums[c] = 0;
                for (int j = 0; j < classPrototypes[c].length; j++) {
                    currDist = cmet.dist(instance, classPrototypes[c][j]);
                    protoDists[c][j] = currDist;
                    protoDistExps[c][j] = (float) Math.exp(-currDist * currDist
                            / (2 * protoDispersions[c][j]));
                    sumAll += protoDistExps[c][j];
                    classSums[c] += protoDistExps[c][j];
                    if (currDist < minDist) {
                        minDist = currDist;
                        minProtoIndex = j;
                        minClassIndex = c;
                    }
                }
            }
            if (DataMineConstants.isZero(sumAll)) {
                sumAll = DataMineConstants.EPSILON;
            }
            if (DataMineConstants.isZero(classSums[minClassIndex])) {
                classSums[minClassIndex] = DataMineConstants.EPSILON;
            }
            choosingProbTotal = protoDistExps[minClassIndex][minProtoIndex]
                    / sumAll;
            choosingProbInClass = protoDistExps[minClassIndex][minProtoIndex]
                    / classSums[minClassIndex];

            if (minClassIndex == currClass) {
                // Reward.
                deltaProtoFact = (alphaProto / Math.max(
                        protoDispersions[minClassIndex][minProtoIndex],
                        DataMineConstants.EPSILON)) * (choosingProbInClass
                        - choosingProbTotal);
                deltaVariance = (alphaVariance / Math.max(
                        protoDispersions[minClassIndex][minProtoIndex],
                        DataMineConstants.EPSILON)) * (choosingProbInClass
                        - choosingProbTotal) * (-dim + (minDist * minDist
                        / Math.max(
                        protoDispersions[minClassIndex][minProtoIndex],
                        DataMineConstants.EPSILON)));
            } else {
                // Penalty.
                deltaProtoFact = (alphaProto / Math.max(protoDispersions[
                        minClassIndex][minProtoIndex],
                        DataMineConstants.EPSILON)) * (-choosingProbTotal);
                deltaVariance = (alphaVariance / Math.max(protoDispersions[
                        minClassIndex][minProtoIndex],
                        DataMineConstants.EPSILON)) * (-choosingProbTotal)
                        * (-dim + (minDist * minDist /
                        Math.max(protoDispersions[minClassIndex][minProtoIndex],
                        DataMineConstants.EPSILON)));
            }
            protoDispersions[minClassIndex][minProtoIndex] += deltaVariance;
            for (int d = 0; d < dim; d++) {
                classPrototypes[minClassIndex][minProtoIndex].fAttr[d] +=
                        deltaProtoFact * (instance.fAttr[d] - classPrototypes[
                        minClassIndex][minProtoIndex].fAttr[d]);
            }
        }

    }

    @Override
    public int classify(DataInstance instance) throws Exception {
        float[] classProbs = classifyProbabilistically(instance);
        float maxProb = 0;
        int maxClassIndex = 0;
        for (int i = 0; i < numClasses; i++) {
            if (classProbs[i] > maxProb) {
                maxProb = classProbs[i];
                maxClassIndex = i;
            }
        }
        return maxClassIndex;
    }

    @Override
    public float[] classifyProbabilistically(DataInstance instance)
            throws Exception {
        float[] classProbs = new float[numClasses];
        float classTotal = 0;
        float[] classMinDists = new float[numClasses];
        CombinedMetric cmet = getCombinedMetric();
        float dist;
        float maxDist = 0;
        for (int c = 0; c < numClasses; c++) {
            classMinDists[c] = Float.MAX_VALUE;
            for (int i = 0; i < classPrototypes[c].length; i++) {
                dist = cmet.dist(classPrototypes[c][i], instance);
                if (dist < classMinDists[c]) {
                    classMinDists[c] = dist;
                }
                if (dist > maxDist) {
                    maxDist = dist;
                }
            }
        }
        maxDist = Math.max(maxDist, 1);
        for (int c = 0; c < numClasses; c++) {
            classProbs[c] = maxDist - classMinDists[c];
            classTotal += classProbs[c];
        }
        if (classTotal != 0) {
            for (int c = 0; c < numClasses; c++) {
                classProbs[c] /= classTotal;
            }
        }
        return classProbs;
    }

    @Override
    public void setClasses(Category[] categories) {
        super.setClasses(categories);
        int totalSize = 0;
        int indexFirstNonEmptyClass = -1;
        for (int cIndex = 0; cIndex < categories.length; cIndex++) {
            totalSize += categories[cIndex].size();
            if (indexFirstNonEmptyClass == -1
                    && categories[cIndex].size() > 0) {
                indexFirstNonEmptyClass = cIndex;
            }
        }
        // Data instances are not embedded in the internal data context.
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
}
