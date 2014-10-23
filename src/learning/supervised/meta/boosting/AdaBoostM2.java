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
package learning.supervised.meta.boosting;

import data.neighbors.NSFUserInterface;
import data.neighbors.NeighborSetFinder;
import data.representation.DataInstance;
import data.representation.DataSet;
import data.representation.util.DataMineConstants;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import learning.supervised.Category;
import learning.supervised.Classifier;
import learning.supervised.evaluation.ValidateableInterface;
import learning.supervised.interfaces.DistMatrixUserInterface;
import learning.supervised.interfaces.DistToPointsQueryUserInterface;
import learning.supervised.interfaces.NeighborPointsQueryUserInterface;
import util.ArrayUtil;
import util.BasicMathUtil;

/**
 * This class implements a classical multi-class boosting method AdaBoost.M2. It
 * re-weights the instances so as to emphasize difficult examples. It also
 * provides a label weighting function for each instance. The final
 * classification is reached by a weighted ensemble vote over all generated
 * models.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class AdaBoostM2 extends Classifier implements
        DistMatrixUserInterface, NSFUserInterface,
        DistToPointsQueryUserInterface, NeighborPointsQueryUserInterface,
        Serializable {
    
    private static final long serialVersionUID = 1L;

    // The base classifier to use in boosting.
    private BoostableClassifier weakLearner;
    // All of the trained models.
    private ArrayList<BoostableClassifier> trainedModels;
    private int numIterationsTrain = 100;
    private int numIterationsTest;
    private int numClasses = 1;
    float[] classPriors;
    private DataSet trainingData;
    private Category[] categories;
    // Distributions over the training data.
    private ArrayList<double[]> distributions;
    // Class-conditional difficulty weights.
    private ArrayList<double[][]> weightsPerLabel;
    // Label weighting function.
    private ArrayList<double[][]> labelWeightingFunction;
    // Total instance difficulty weights.
    private ArrayList<double[]> totalWeights;
    // Pseudo-loss over the iterations.
    private ArrayList<Double> pseudoLoss;
    // Loss ratios over the iterations, used to determine learner weights.
    private ArrayList<Double> lossRatios;
    // Learner predictions over the iterations.
    private ArrayList<double[][]> predictions;
    // For corrections when the weights approach the minimal double value too
    // much.
    private static final double SAFE_FACTOR = Math.pow(2, 50);
    
    @Override
    public HashMap<String, String> getParameterNamesAndDescriptions() {
        HashMap<String, String> paramMap = new HashMap<>();
        paramMap.put("numIterationsTrain", "Number of iterations to use for"
                + "boosting.");
        HashMap<String, String> baseParamMap =
                weakLearner.getParameterNamesAndDescriptions();
        HashMap<String, String> resultingMap = new HashMap<>();
        resultingMap.putAll(paramMap);
        baseParamMap.putAll(baseParamMap);
        return resultingMap;
    }
    
    @Override
    public long getVersion() {
        return serialVersionUID;
    }

    /**
     * Initialization.
     *
     * @param weakLearner BoostableClassifier that is to be boosted.
     */
    public AdaBoostM2(BoostableClassifier weakLearner) {
        this.weakLearner = weakLearner;
    }

    /**
     * Initialization.
     *
     * @param weakLearner BoostableClassifier that is to be boosted.
     * @param numIterations Integer that is the number of boosting iterations.
     */
    public AdaBoostM2(BoostableClassifier weakLearner, int numIterations) {
        this.weakLearner = weakLearner;
        this.numIterationsTrain = numIterations;
    }

    /**
     * Initialization.
     *
     * @param weakLearner BoostableClassifier that is to be boosted.
     * @param numIterations Integer that is the number of boosting iterations.
     * @param trainingData DataSet object to train the model on.
     */
    public AdaBoostM2(BoostableClassifier weakLearner, int numIterations,
            DataSet trainingData) {
        this.weakLearner = weakLearner;
        this.numIterationsTrain = numIterations;
        this.trainingData = trainingData;
    }

    @Override
    public String getName() {
        if (weakLearner != null) {
            return "AdaBoostM2(" + weakLearner.getName() + ")";
        } else {
            return "AdaBoostM2";
        }
    }

    @Override
    public void train() throws Exception {
        NeighborSetFinder nsf = getNSF();
        float[][] dMat = getDistMatrix();
        if (weakLearner instanceof NSFUserInterface && nsf == null) {
            // Then we should calculate the NSF object, for latter speed-ups.
            nsf = new NeighborSetFinder(trainingData, getCombinedMetric());
            if (dMat != null) {
                nsf.setDistances(dMat);
            } else {
                nsf.calculateDistances();
            }
            nsf.calculateNeighborSets(((NSFUserInterface)weakLearner).
                    getNeighborhoodSize());
        }
        int numInstances = trainingData.size();
        classPriors = trainingData.getClassPriors();
        // Because of the interfaces it pays off to duplicate the matrix in a
        // different format.
        float[][] distToTrain = new float[numInstances][numInstances];
        if (dMat != null) {
            for (int i = 0; i < numInstances; i++) {
                for (int j = i + 1; j < numInstances; j++) {
                    distToTrain[i][j] = dMat[i][j - i - 1];
                    distToTrain[j][i] = dMat[i][j - i - 1];
                }
            }
        }
        numClasses = trainingData.countCategories();
        distributions = new ArrayList<>(numIterationsTrain);
        weightsPerLabel = new ArrayList<>(numIterationsTrain);
        labelWeightingFunction = new ArrayList<>(numIterationsTrain);
        totalWeights = new ArrayList<>(numIterationsTrain);
        pseudoLoss = new ArrayList<>(numIterationsTrain);
        lossRatios = new ArrayList<>(numIterationsTrain);
        // Initialize the distribution.
        double[] distribution = new double[numInstances];
        Arrays.fill(distribution, 1f / numInstances);
        double[][] weightsPerLabelArray = new double[numInstances][numClasses];
        for (int i = 0; i < numInstances; i++) {
            int label = trainingData.getLabelOf(i);
            for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                if (cIndex != label) {
                    weightsPerLabelArray[i][cIndex] = distribution[i]
                            / (numClasses - 1);
                }
            }
        }
        weightsPerLabel.add(weightsPerLabelArray);
        trainedModels = new ArrayList<>(numIterationsTrain);
        predictions = new ArrayList<>(numIterationsTrain);
        for (int iterationIndex = 0; iterationIndex < numIterationsTrain;
                iterationIndex++) {
            double iterationPseudoLoss = 0;
            double[] totalWeightsArray = new double[numInstances];
            double[][] labelWeightingFunctionArray =
                    new double[numInstances][numClasses];
            double[][] predictionArray = new double[numInstances][];
            double totalWeightSum = 0;
            distribution = new double[numInstances];
            for (int i = 0; i < numInstances; i++) {
                int label = trainingData.getLabelOf(i);
                for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                    if (cIndex != label) {
                        totalWeightsArray[i] +=
                                weightsPerLabelArray[i][cIndex];
                    }
                }
                totalWeightSum += totalWeightsArray[i];
                for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                    if (cIndex != label) {
                        if (totalWeightsArray[i] == 0) {
                            break;
                        }
                        labelWeightingFunctionArray[i][cIndex] =
                                weightsPerLabelArray[i][cIndex]
                                / totalWeightsArray[i];
                    }
                }
            }
            for (int i = 0; i < numInstances; i++) {
                if (totalWeightSum > 0) {
                    distribution[i] = totalWeightsArray[i] / totalWeightSum;
                }
            }
            distributions.add(distribution);
            weightsPerLabel.add(weightsPerLabelArray);
            labelWeightingFunction.add(labelWeightingFunctionArray);
            totalWeights.add(totalWeightsArray);
            // Here we update the model by adding another weak learner.
            BoostableClassifier iterationLearner =
                    (BoostableClassifier) (weakLearner.copyConfiguration());
            iterationLearner.setClasses(categories);
            // Assign the required objects.
            if (weakLearner instanceof DistMatrixUserInterface) {
                ((DistMatrixUserInterface) iterationLearner).
                        setDistMatrix(dMat);
            }
            if (weakLearner instanceof NSFUserInterface) {
                ((NSFUserInterface) iterationLearner).
                        setNSF(nsf);
            }
            iterationLearner.setTotalInstanceWeights(totalWeightsArray);
            iterationLearner.setMisclassificationCostDistribution(
                    labelWeightingFunctionArray);
            // Train the learner.
            iterationLearner.train();
            trainedModels.add(iterationLearner);
            for (int i = 0; i < numInstances; i++) {
                int label = trainingData.getLabelOf(i);
                float[] classProbs;
                if (weakLearner instanceof DistToPointsQueryUserInterface
                        && weakLearner instanceof
                        NeighborPointsQueryUserInterface && nsf != null) {
                    classProbs = ((NeighborPointsQueryUserInterface)
                            iterationLearner).classifyProbabilistically(
                            trainingData.getInstance(i),
                            distToTrain[i], nsf.getKNeighbors()[i]);
                } else if (weakLearner instanceof
                        DistToPointsQueryUserInterface) {
                    classProbs = ((DistToPointsQueryUserInterface)
                            iterationLearner).classifyProbabilistically(
                            trainingData.getInstance(i),
                            distToTrain[i]);
                } else {
                    classProbs = iterationLearner.classifyProbabilistically(
                            trainingData.getInstance(i));
                }
                predictionArray[i] = new double[numClasses];
                double correctProb = classProbs[label];
                double penaltySum = 0;
                for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                    predictionArray[i][cIndex] = classProbs[cIndex];
                    if (cIndex != label) {
                        penaltySum += classProbs[cIndex]
                                * labelWeightingFunctionArray[i][cIndex];
                    }
                }
                if (!DataMineConstants.isAcceptableDouble(penaltySum)
                        || !DataMineConstants.isAcceptableDouble(
                        distribution[i])
                        || !DataMineConstants.isAcceptableDouble(correctProb)) {
                    continue;
                }
                iterationPseudoLoss += (distribution[i] / 2) * (1 - correctProb
                        + penaltySum);
            }
            predictions.add(predictionArray);
            pseudoLoss.add(iterationPseudoLoss);
            double lossRatio = iterationPseudoLoss
                    / (1 - iterationPseudoLoss);
            // In case of long iterations, the weights can become very small,
            // so here we add some correction logic.
            double maxWeight = 0;
            for (int i = 0; i < numInstances; i++) {
                maxWeight = Math.max(maxWeight, ArrayUtil.max(
                        weightsPerLabelArray[i]));
            }
            if (maxWeight < Double.MIN_VALUE * SAFE_FACTOR) {
                for (int i = 0; i < numInstances; i++) {
                    int label = trainingData.getLabelOf(i);
                    for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                        if (cIndex != label) {
                            weightsPerLabelArray[i][cIndex] *= SAFE_FACTOR;
                        }
                    }
                }
            }
            weightsPerLabelArray = new double[numInstances][numClasses];
            // Now update the weights based on the pseudo-loss.
            lossRatios.add(lossRatio);
            for (int i = 0; i < numInstances; i++) {
                int label = trainingData.getLabelOf(i);
                for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                    double updateExponent = 0.5 * (1
                            + predictionArray[i][label]
                            - predictionArray[i][cIndex]);
                    double updateFactor = Math.pow(lossRatio,
                            updateExponent);
                    if (cIndex != label) {
                        if (DataMineConstants.isAcceptableDouble(
                                updateFactor)) {
                            weightsPerLabelArray[i][cIndex] =
                                    weightsPerLabel.get(iterationIndex)[i][
                                    cIndex] * updateFactor;
                        } else {
                            // If something went wrong, keep the previous
                            // weight.
                            weightsPerLabelArray[i][cIndex] =
                                    weightsPerLabel.get(iterationIndex)[i][
                                    cIndex];
                        }
                    }
                }
            }
        }
        numIterationsTest = numIterationsTrain;
        float[] totalAccuracyCount = new float[numIterationsTrain];
        float[][] currPrediction = new float[numInstances][numClasses];
        float bestAccuracyCount = 0;
        for (int iterIndex = 0; iterIndex < numIterationsTrain; iterIndex++) {
            BoostableClassifier iterationLearner = trainedModels.get(
                    iterIndex);
            double lossRatio = lossRatios.get(iterIndex);
            float factor = (float) BasicMathUtil.log2(1. / lossRatio);
            for (int i = 0; i < numInstances; i++) {
                DataInstance instance = trainingData.getInstance(i);
                float[] classProbsIteration;
                if (weakLearner instanceof DistToPointsQueryUserInterface
                        && weakLearner instanceof
                        NeighborPointsQueryUserInterface && nsf != null) {
                    classProbsIteration = ((NeighborPointsQueryUserInterface)
                            iterationLearner).classifyProbabilistically(
                            instance, distToTrain[i], nsf.getKNeighbors()[i]);
                } else if (weakLearner instanceof
                        DistToPointsQueryUserInterface) {
                    classProbsIteration = ((DistToPointsQueryUserInterface)
                            iterationLearner).classifyProbabilistically(
                            instance, distToTrain[i]);
                } else {
                    classProbsIteration =
                            iterationLearner.
                            classifyProbabilistically(instance);
                }
                float maxClassIndex = 0;
                float maxClassVote = -Float.MAX_VALUE;
                for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                    currPrediction[i][cIndex] += factor
                            * classProbsIteration[cIndex];
                    if (currPrediction[i][cIndex] > maxClassVote) {
                        maxClassVote = currPrediction[i][cIndex];
                        maxClassIndex = cIndex;
                    }
                }
                if (maxClassIndex == instance.getCategory()) {
                    totalAccuracyCount[iterIndex]++;
                }
            }
            if (totalAccuracyCount[iterIndex] > bestAccuracyCount) {
                bestAccuracyCount = totalAccuracyCount[iterIndex];
                numIterationsTest = iterIndex + 1;
            }
        }
    }

    @Override
    public float[] classifyProbabilistically(DataInstance instance,
            float[] distToTraining, int[] trNeighbors) throws Exception {
        float[] classProbEstimates = new float[numClasses];
        for (int iterationIndex = 0; iterationIndex < numIterationsTest;
                iterationIndex++) {
            BoostableClassifier iterationLearner = trainedModels.get(
                    iterationIndex);
            double lossRatio = lossRatios.get(iterationIndex);
            float[] classProbsIteration;
            if (weakLearner instanceof DistToPointsQueryUserInterface
                    && weakLearner instanceof
                    NeighborPointsQueryUserInterface) {
                classProbsIteration = ((NeighborPointsQueryUserInterface)
                        iterationLearner).classifyProbabilistically(
                        instance, distToTraining, trNeighbors);
            } else if (weakLearner instanceof DistToPointsQueryUserInterface) {
                classProbsIteration = ((DistToPointsQueryUserInterface)
                        iterationLearner).classifyProbabilistically(
                        instance, distToTraining);
            } else {
                classProbsIteration =
                        iterationLearner.classifyProbabilistically(instance);
            }
            float factor = (float) BasicMathUtil.log2(1. / lossRatio);
            if (!DataMineConstants.isAcceptableFloat(factor)) {
                continue;
            }
            for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                if (DataMineConstants.isAcceptableFloat(
                        classProbsIteration[cIndex])) {
                    classProbEstimates[cIndex] += factor
                            * classProbsIteration[cIndex];
                }
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
        float[] classProbEstimates = new float[numClasses];
        for (int iterationIndex = 0; iterationIndex < numIterationsTest;
                iterationIndex++) {
            BoostableClassifier iterationLearner = trainedModels.get(
                    iterationIndex);
            double lossRatio = lossRatios.get(iterationIndex);
            float[] classProbsIteration;
            if (weakLearner instanceof DistToPointsQueryUserInterface) {
                classProbsIteration = ((DistToPointsQueryUserInterface)
                        iterationLearner).classifyProbabilistically(
                        instance, distToTraining);
            } else {
                classProbsIteration =
                        iterationLearner.classifyProbabilistically(instance);
            }
            float factor = (float) BasicMathUtil.log2(1. / lossRatio);
            if (!DataMineConstants.isAcceptableFloat(factor)) {
                continue;
            }
            for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                if (DataMineConstants.isAcceptableFloat(
                        classProbsIteration[cIndex])) {
                    classProbEstimates[cIndex] += factor
                            * classProbsIteration[cIndex];
                }
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
    public float[] classifyProbabilistically(DataInstance instance)
            throws Exception {
        float[] classProbEstimates = new float[numClasses];
        for (int iterationIndex = 0; iterationIndex < numIterationsTest;
                iterationIndex++) {
            BoostableClassifier iterationLearner = trainedModels.get(
                    iterationIndex);
            double lossRatio = lossRatios.get(iterationIndex);
            float[] classProbsIteration =
                    iterationLearner.classifyProbabilistically(instance);
            float factor = (float) BasicMathUtil.log2(1. / lossRatio);
            if (!DataMineConstants.isAcceptableFloat(factor)) {
                continue;
            }
            for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                if (DataMineConstants.isAcceptableFloat(
                        classProbsIteration[cIndex])) {
                    classProbEstimates[cIndex] += factor
                            * classProbsIteration[cIndex];
                }
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
    public int classify(DataInstance instance, float[] distToTraining)
            throws Exception {
        float[] classProbs = classifyProbabilistically(instance,
                distToTraining);
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
    public int classify(DataInstance instance, float[] distToTraining,
            int[] trNeighbors) throws Exception {
        float[] classProbs = classifyProbabilistically(instance,
                distToTraining, trNeighbors);
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
        this.categories = categories;
    }

    @Override
    public ValidateableInterface copyConfiguration() {
        AdaBoostM2 classifierCopy = new AdaBoostM2(weakLearner,
                numIterationsTrain);
        return classifierCopy;
    }

    @Override
    public void setNSF(NeighborSetFinder nsf) {
        if (weakLearner != null && weakLearner instanceof NSFUserInterface) {
            ((NSFUserInterface) weakLearner).setNSF(nsf);
        }
    }

    @Override
    public NeighborSetFinder getNSF() {
        if (weakLearner != null && weakLearner instanceof NSFUserInterface) {
            NeighborSetFinder nsf = ((NSFUserInterface) weakLearner).getNSF();
            return nsf;
        } else {
            return null;
        }
    }

    @Override
    public void setDistMatrix(float[][] distMatrix) {
        if (weakLearner != null && weakLearner instanceof
                DistMatrixUserInterface) {
            ((DistMatrixUserInterface) weakLearner).setDistMatrix(distMatrix);
        }
    }

    @Override
    public float[][] getDistMatrix() {
        if (weakLearner != null && weakLearner instanceof
                DistMatrixUserInterface) {
            return ((DistMatrixUserInterface) weakLearner).getDistMatrix();
        } else {
            return null;
        }
    }

    @Override
    public void noRecalcs() {
    }
    
    @Override
    public int getNeighborhoodSize() {
        if (weakLearner != null && weakLearner instanceof NSFUserInterface) {
            return ((NSFUserInterface)weakLearner).getNeighborhoodSize();
        } else {
            return 0;
        }
    }
}
