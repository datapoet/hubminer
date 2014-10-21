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
package learning.supervised;

import data.representation.discrete.DiscretizedDataInstance;
import data.representation.discrete.DiscretizedDataSet;
import data.representation.DataSet;
import learning.supervised.evaluation.ClassificationEstimator;
import learning.supervised.evaluation.ValidateableInterface;
import java.util.ArrayList;
import preprocessing.instance_selection.InstanceSelector;
import distances.primary.CombinedMetric;
import ioformat.FileUtil;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import learning.supervised.interfaces.DiscreteDistToPointsQueryUserInterface;
import learning.supervised.interfaces.DiscreteNeighborPointsQueryUserInterface;
import learning.supervised.interfaces.DistToPointsQueryUserInterface;

/**
 * This class implements the methods used for classifier training and testing on
 * discretized data instances.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public abstract class DiscreteClassifier implements ValidateableInterface,
        Serializable {
    
    private static final long serialVersionUID = 1L;

    private DiscreteCategory[] tainingClasses = null;
    private DiscretizedDataSet dataType = null;
    // In case of hybrid learning from both discretized and non-discretized
    // data sources.
    private CombinedMetric cmet;

    @Override
    public void trainOnReducedData(InstanceSelector reducer) throws Exception {
        setDataIndexes(reducer.getPrototypeIndexes(),
                reducer.getReducedDataSet(true));
        train();
    }

    @Override
    public void setData(ArrayList data, Object dataType) {
        if (data != null && data.size() > 0) {
            ArrayList<DiscretizedDataInstance> dataVect =
                    new ArrayList<>(data.size());
            for (int i = 0; i < data.size(); i++) {
                dataVect.add((DiscretizedDataInstance) (data.get(i)));
            }
            DiscretizedDataSet definition = (DiscretizedDataSet) dataType;
            this.dataType = definition.cloneDefinition();
            this.dataType.data = dataVect;
            this.dataType.setOriginalData(definition.getOriginalData());
            generateClassesFromDataType();
        }
    }

    @Override
    public void setDataIndexes(ArrayList<Integer> currentIndexes,
            Object objectType) {
        if (currentIndexes != null && currentIndexes.size() > 0) {
            DiscretizedDataSet definition = (DiscretizedDataSet) objectType;
            DataSet originalDefinition = definition.getOriginalData();
            DataSet originalDataSubset = originalDefinition.cloneDefinition();
            dataType = definition.cloneDefinition();
            dataType.setOriginalData(originalDataSubset);
            dataType.data = new ArrayList<>(currentIndexes.size());
            originalDataSubset.data = new ArrayList<>(currentIndexes.size());
            for (int i = 0; i < currentIndexes.size(); i++) {
                dataType.data.add(definition.data.get(currentIndexes.get(i)));
                originalDataSubset.data.add(originalDefinition.data.get(
                        currentIndexes.get(i)));
            }
            generateClassesFromDataType();
        }
    }

    /**
     * Generate the discrete category array as a training data representation,
     * instead of the discretized data sets.
     */
    public void generateClassesFromDataType() {
        int numClasses = 0;
        int currClass;
        DiscretizedDataInstance instance;
        DiscretizedDataSet discDSet = (DiscretizedDataSet) dataType;
        for (int i = 0; i < dataType.data.size(); i++) {
            if (dataType.data.get(i) != null) {
                instance = (DiscretizedDataInstance) (dataType.data.get(i));
                currClass = instance.getCategory();
                if (currClass > numClasses) {
                    numClasses = currClass;
                }
            }
        }
        numClasses = numClasses + 1;
        tainingClasses = new DiscreteCategory[numClasses];
        for (int cIndex = 0; cIndex < numClasses; cIndex++) {
            tainingClasses[cIndex] =
                    new DiscreteCategory("number" + cIndex, discDSet, 200);
        }
        for (int i = 0; i < dataType.data.size(); i++) {
            if (dataType.data.get(i) != null) {
                instance = (DiscretizedDataInstance) (dataType.data.get(i));
                currClass = instance.getCategory();
                tainingClasses[currClass].indexes.add(new Integer(i));
            }
        }
    }

    @Override
    public void setClasses(Object[] dataClasses) {
        if (dataClasses == null || dataClasses.length == 0) {
            return;
        }
        DiscreteCategory[] dClasses = new DiscreteCategory[dataClasses.length];
        for (int cIndex = 0; cIndex < dataClasses.length; cIndex++) {
            dClasses[cIndex] = (DiscreteCategory) (dataClasses[cIndex]);
        }
        setClasses(dClasses);
    }

    /**
     * @param dataClasses DiscreteCategory[] representing the discretized
     * training data.
     */
    public void setClasses(DiscreteCategory[] dataClasses) {
        this.tainingClasses = dataClasses;
    }

    /**
     * @return DiscreteCategory[] representing the discretized training data.
     */
    public DiscreteCategory[] getClasses() {
        return tainingClasses;
    }

    /**
     * @param dataType DiscretizedDataSet object representing the training data.
     */
    public void setDataType(DiscretizedDataSet dataType) {
        this.dataType = dataType;
    }

    /**
     * @return DiscretizedDataSet object representing the training data.
     */
    public DiscretizedDataSet getDataType() {
        return dataType;
    }

    /**
     * @return DataSet object that the training data was discretized from.
     */
    public DataSet getOriginalDataType() {
        if (dataType != null) {
            return dataType.getOriginalData();
        } else {
            return null;
        }
    }

    @Override
    public abstract ValidateableInterface copyConfiguration();

    @Override
    public abstract void train() throws Exception;

    public abstract int classify(DiscretizedDataInstance instance)
            throws Exception;

    public abstract float[] classifyProbabilistically(
            DiscretizedDataInstance instance) throws Exception;

    /**
     * This method performs batch classification of DiscretizedDataInstance
     * objects.
     *
     * @param instances DiscretizedDataInstance[] that is the discretized data
     * to classify.
     * @return int[] corresponding the the classification results.
     * @throws Exception
     */
    public int[] classify(DiscretizedDataInstance[] instances)
            throws Exception {
        int[] classificationResults = null;
        if ((instances == null) || (instances.length == 0)) {
            return null;
        } else {
            classificationResults = new int[instances.length];
            for (int i = 0; i < instances.length; i++) {
                classificationResults[i] = classify(instances[i]);

            }
            return classificationResults;
        }
    }

    /**
     * This method performs batch classification of DiscretizedDataInstance
     * objects.
     *
     * @param instances DiscretizedDataInstance[] that is the discretized data
     * to classify.
     * @return float[][] corresponding the the classification results.
     * @throws Exception
     */
    public float[][] classifyProbabilistically(
            DiscretizedDataInstance[] instances) throws Exception {
        float[][] classificationResults;
        if ((instances == null) || (instances.length == 0)) {
            return null;
        } else {
            classificationResults = new float[instances.length][];
            for (int i = 0; i < instances.length; i++) {
                classificationResults[i] =
                        classifyProbabilistically(instances[i]);
            }
            return classificationResults;
        }
    }

    /**
     * This method performs batch classification of DiscretizedDataInstance
     * objects.
     *
     * @param instances ArrayList<DiscretizedDataInstance> that is the
     * discretized data to classify.
     * @return int[] corresponding the the classification results.
     * @throws Exception
     */
    public int[] classify(ArrayList<DiscretizedDataInstance> instances)
            throws Exception {
        int[] classificationResults;
        if ((instances == null) || (instances.isEmpty())) {
            return null;
        } else {
            classificationResults = new int[instances.size()];
            for (int i = 0; i < instances.size(); i++) {
                classificationResults[i] = classify(instances.get(i));

            }
            return classificationResults;
        }
    }

    /**
     * This method performs batch classification of DiscretizedDataInstance
     * objects.
     *
     * @param instances ArrayList<DiscretizedDataInstance> that is the
     * discretized data to classify.
     * @return float[][] corresponding the the classification results.
     * @throws Exception
     */
    public float[][] classifyProbabilistically(
            ArrayList<DiscretizedDataInstance> instances) throws Exception {
        float[][] classificationResults;
        if ((instances == null) || (instances.isEmpty())) {
            return null;
        } else {
            classificationResults = new float[instances.size()][];
            for (int i = 0; i < instances.size(); i++) {
                classificationResults[i] =
                        classifyProbabilistically(instances.get(i));
            }
            return classificationResults;
        }
    }

    @Override
    public ClassificationEstimator test(Object[] testClasses) throws Exception {
        return test((DiscreteCategory[]) testClasses);
    }

    public ClassificationEstimator test(DiscretizedDataSet testDiscDSet,
            int numClasses) throws Exception {
        DiscretizedDataInstance instance;
        int classificationResult;
        float[][] confusionMatrix = new float[numClasses][numClasses];
        if (testDiscDSet.size() > 0) {
            for (int i = 0; i < testDiscDSet.size(); i++) {
                instance = testDiscDSet.data.get(i);
                classificationResult = classify(instance);
                confusionMatrix[instance.getCategory()][classificationResult]++;
            }
            ClassificationEstimator estimator =
                    new ClassificationEstimator(confusionMatrix);
            estimator.calculateEstimates();
            return estimator;
        } else {
            return null;
        }
    }

    /**
     * This method evaluates the trained model on the test data.
     *
     * @param testClasses DiscreteCategory[] representing the test data.
     * @return ClassificationEstimator holding the classifier evaluation quality
     * metrics.
     * @throws Exception
     */
    public ClassificationEstimator test(DiscreteCategory[] testClasses)
            throws Exception {
        if ((testClasses == null) || (testClasses.length == 0)) {
            return null;
        } else {
            int[] classificationResult;
            float[][] confusionMatrix = new float[testClasses.length][
                    testClasses.length];
            for (int cIndex = 0; cIndex < testClasses.length; cIndex++) {
                classificationResult = classify(testClasses[cIndex].getData());
                if (classificationResult != null) {
                    for (int i = 0; i < classificationResult.length; i++) {
                        confusionMatrix[cIndex][classificationResult[i]]++;
                    }
                }
            }
            ClassificationEstimator estimator =
                    new ClassificationEstimator(confusionMatrix);
            estimator.calculateEstimates();
            return estimator;
        }
    }

    /**
     * This method evaluates the trained model on the test data.
     *
     * @param correctPointClassificationArray float[] that is updated with the
     * total point-wise classification precision.
     * @param testClasses DiscreteCategory[] representing the test data.
     * @return ClassificationEstimator holding the classifier evaluation quality
     * metrics.
     * @throws Exception
     */
    public ClassificationEstimator test(float[] correctPointClassificationArray,
            DiscreteCategory[] testClasses) throws Exception {
        if ((testClasses == null) || (testClasses.length == 0)) {
            return null;
        } else {
            int[] classificationResult;
            float[][] confusionMatrix = new float[testClasses.length][
                    testClasses.length];
            for (int cIndex = 0; cIndex < testClasses.length; cIndex++) {
                classificationResult = classify(testClasses[cIndex].getData());
                if (classificationResult != null) {
                    for (int i = 0; i < classificationResult.length; i++) {
                        confusionMatrix[cIndex][classificationResult[i]]++;
                        if (classificationResult[i] == cIndex) {
                            correctPointClassificationArray[
                                    testClasses[cIndex].indexes.get(i)]++;
                        }
                    }
                }
            }
            ClassificationEstimator estimator =
                    new ClassificationEstimator(confusionMatrix);
            estimator.calculateEstimates();
            return estimator;
        }
    }

    @Override
    public ClassificationEstimator test(ArrayList<Integer> indexes,
            Object dataType, int numClasses) throws Exception {
        if (indexes != null && indexes.isEmpty()) {
            DiscreteCategory[] testClasses;
            DiscretizedDataInstance instance;
            DiscretizedDataSet discDSet = (DiscretizedDataSet) dataType;
            testClasses = new DiscreteCategory[numClasses];
            for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                testClasses[cIndex] =
                        new DiscreteCategory("number" + cIndex, discDSet, 200);
            }
            for (int i = 0; i < indexes.size(); i++) {
                instance = discDSet.data.get(indexes.get(i));
                testClasses[instance.getCategory()].indexes.add(indexes.get(i));
            }
            return test(testClasses);
        } else {
            return null;
        }
    }

    @Override
    public ClassificationEstimator test(ArrayList<Integer> indexes,
            Object dataType, int[] testLabelArray, int numClasses)
            throws Exception {
        if (indexes != null && !indexes.isEmpty()) {
            DiscreteCategory[] testClasses;
            DiscretizedDataInstance instance;
            DiscretizedDataSet discDSet = (DiscretizedDataSet) dataType;
            testClasses = new DiscreteCategory[numClasses];
            for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                testClasses[cIndex] =
                        new DiscreteCategory("number" + cIndex, discDSet, 200);
            }
            for (int i = 0; i < indexes.size(); i++) {
                instance = discDSet.data.get(indexes.get(i));
                testClasses[testLabelArray[indexes.get(i)]].indexes.add(
                        indexes.get(i));
                testClasses[instance.getCategory()].indexes.add(indexes.get(i));
            }
            return test(testClasses);
        } else {
            return null;
        }
    }

    @Override
    public ClassificationEstimator test(float[] correctPointClassificationArray,
            ArrayList<Integer> indexes, Object dataType, int numClasses)
            throws Exception {
        if (indexes != null && !indexes.isEmpty()) {
            DiscreteCategory[] testClasses;
            DiscretizedDataInstance instance;
            DiscretizedDataSet discDSet = (DiscretizedDataSet) dataType;
            testClasses = new DiscreteCategory[numClasses];
            for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                testClasses[cIndex] =
                        new DiscreteCategory("number" + cIndex, discDSet, 200);
            }
            for (int i = 0; i < indexes.size(); i++) {
                instance = discDSet.data.get(indexes.get(i));
                testClasses[instance.getCategory()].indexes.add(indexes.get(i));
            }
            return test(correctPointClassificationArray, testClasses);
        } else {
            return null;
        }
    }

    @Override
    public ClassificationEstimator test(float[] correctPointClassificationArray,
            ArrayList<Integer> indexes, Object dataType, int[] testLabelArray,
            int numClasses) throws Exception {
        if (indexes != null && !indexes.isEmpty()) {
            DiscreteCategory[] testClasses;
            DiscretizedDataInstance instance;
            DiscretizedDataSet discDSet = (DiscretizedDataSet) dataType;
            testClasses = new DiscreteCategory[numClasses];
            for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                testClasses[cIndex] =
                        new DiscreteCategory("number" + cIndex, discDSet, 200);
            }
            for (int i = 0; i < indexes.size(); i++) {
                instance = discDSet.data.get(indexes.get(i));
                testClasses[testLabelArray[indexes.get(i)]].indexes.add(
                        indexes.get(i));
                testClasses[instance.getCategory()].indexes.add(indexes.get(i));
            }
            return test(correctPointClassificationArray, testClasses);
        } else {
            return null;
        }
    }

    /**
     * @param cmet CombinedMetric object for distance calculations. It may be
     * used in hybrid methods that combine the information from the discretized
     * and non-discretized data sources.
     */
    public void setCombinedMetric(CombinedMetric cmet) {
        this.cmet = cmet;
    }

    /**
     * @return CombinedMetric object for distance calculations. It may be used
     * in hybrid methods that combine the information from the discretized and
     * non-discretized data sources.
     */
    public CombinedMetric getCombinedMetric() {
        return cmet;
    }

    @Override
    public ClassificationEstimator test(ArrayList<Integer> indexes,
            Object dataType, int numClasses, float[][] pointDistances)
            throws Exception {
        return test(indexes, dataType, numClasses);
    }

    @Override
    public ClassificationEstimator test(ArrayList<Integer> indexes,
            Object dataType, int[] testLabelArray, int numClasses,
            float[][] pointDistances) throws Exception {
        int[] classificationResult = new int[indexes.size()];
        float[][] confusionMatrix = new float[numClasses][numClasses];
        for (int i = 0; i < indexes.size(); i++) {
            if (this instanceof DiscreteDistToPointsQueryUserInterface) {
                classificationResult[i] =
                        ((DiscreteDistToPointsQueryUserInterface) this).
                        classify(((DiscretizedDataSet) dataType).getInstance(
                        indexes.get(i)), pointDistances[i]);
            } else {
                classificationResult[i] =
                        classify(((DiscretizedDataSet) dataType).getInstance(
                        indexes.get(i)));
            }
            confusionMatrix[classificationResult[i]][testLabelArray[
                    indexes.get(i)]]++;
        }
        ClassificationEstimator estimator =
                new ClassificationEstimator(confusionMatrix);
        estimator.calculateEstimates();
        return estimator;
    }

    @Override
    public ClassificationEstimator test(float[] correctPointClassificationArray,
            ArrayList<Integer> indexes, Object dataType, int numClasses,
            float[][] pointDistances) throws Exception {
        int[] classificationResult = null;
        float[][] confusionMatrix = new float[numClasses][numClasses];
        for (int i = 0; i < indexes.size(); i++) {
            if (this instanceof DiscreteDistToPointsQueryUserInterface) {
                classificationResult[i] =
                        ((DiscreteDistToPointsQueryUserInterface) this).
                        classify(((DiscretizedDataSet) dataType).getInstance(
                        indexes.get(i)), pointDistances[i]);
            } else {
                classificationResult[i] =
                        classify(((DiscretizedDataSet) dataType).getInstance(
                        indexes.get(i)));
            }
            confusionMatrix[classificationResult[i]][
                    ((DiscretizedDataSet) dataType).getLabelOf(
                    indexes.get(i))]++;
            if (classificationResult[i]
                    == ((DiscretizedDataSet) dataType).getLabelOf(
                    indexes.get(i))) {
                correctPointClassificationArray[indexes.get(i)]++;
            }
        }
        ClassificationEstimator estimator =
                new ClassificationEstimator(confusionMatrix);
        estimator.calculateEstimates();
        return estimator;
    }

    @Override
    public ClassificationEstimator test(float[] correctPointClassificationArray,
            ArrayList<Integer> indexes, Object dataType, int[] testLabelArray,
            int numClasses, float[][] pointDistances) throws Exception {
        int[] classificationResult = new int[indexes.size()];
        float[][] confusionMatrix = new float[numClasses][numClasses];
        for (int i = 0; i < indexes.size(); i++) {
            if (this instanceof DiscreteDistToPointsQueryUserInterface) {
                classificationResult[i] =
                        ((DiscreteDistToPointsQueryUserInterface) this).
                        classify(((DiscretizedDataSet) dataType).getInstance(
                        indexes.get(i)), pointDistances[i]);
            } else {
                classificationResult[i] = classify(
                        ((DiscretizedDataSet) dataType).getInstance(
                        indexes.get(i)));
            }
            confusionMatrix[classificationResult[i]][testLabelArray[
                    indexes.get(i)]]++;
            if (classificationResult[i] == testLabelArray[indexes.get(i)]) {
                correctPointClassificationArray[indexes.get(i)]++;
            }
        }
        ClassificationEstimator estimator =
                new ClassificationEstimator(confusionMatrix);
        estimator.calculateEstimates();
        return estimator;
    }

    @Override
    public ClassificationEstimator test(float[] correctPointClassificationArray,
            ArrayList<Integer> indexes, Object dataType, int numClasses,
            float[][] pointDistances, int[][] pointNeighbors) throws Exception {
        int[] classificationResult = null;
        float[][] confusionMatrix = new float[numClasses][numClasses];
        for (int i = 0; i < indexes.size(); i++) {
            if (this instanceof DiscreteNeighborPointsQueryUserInterface) {
                classificationResult[i] =
                        ((DiscreteNeighborPointsQueryUserInterface) this).
                        classify(((DiscretizedDataSet) dataType).getInstance(
                        indexes.get(i)), pointDistances[i], pointNeighbors[i]);
            } else if (this instanceof DiscreteDistToPointsQueryUserInterface) {
                classificationResult[i] =
                        ((DiscreteDistToPointsQueryUserInterface) this).
                        classify(((DiscretizedDataSet) dataType).getInstance(
                        indexes.get(i)), pointDistances[i]);
            } else {
                classificationResult[i] = classify(
                        ((DiscretizedDataSet) dataType).getInstance(
                        indexes.get(i)));
            }
            confusionMatrix[classificationResult[i]][
                    ((DiscretizedDataSet) dataType).getLabelOf(
                    indexes.get(i))]++;
            if (classificationResult[i]
                    == ((DiscretizedDataSet) dataType).getLabelOf(
                    indexes.get(i))) {
                correctPointClassificationArray[indexes.get(i)]++;
            }
        }
        ClassificationEstimator estimator =
                new ClassificationEstimator(confusionMatrix);
        estimator.calculateEstimates();
        return estimator;
    }

    @Override
    public ClassificationEstimator test(float[] correctPointClassificationArray,
            ArrayList<Integer> indexes, Object dataType, int[] testLabelArray,
            int numClasses, float[][] pointDistances, int[][] pointNeighbors)
            throws Exception {
        int[] classificationResult = new int[indexes.size()];
        float[][] confusionMatrix = new float[numClasses][numClasses];
        for (int i = 0; i < indexes.size(); i++) {
            if (this instanceof DiscreteNeighborPointsQueryUserInterface) {
                classificationResult[i] =
                        ((DiscreteNeighborPointsQueryUserInterface) this).
                        classify(((DiscretizedDataSet) dataType).getInstance(
                        indexes.get(i)), pointDistances[i], pointNeighbors[i]);
            } else if (this instanceof DiscreteDistToPointsQueryUserInterface) {
                classificationResult[i] =
                        ((DiscreteDistToPointsQueryUserInterface) this).
                        classify(((DiscretizedDataSet) dataType).getInstance(
                        indexes.get(i)), pointDistances[i]);
            } else {
                classificationResult[i] = classify(
                        ((DiscretizedDataSet) dataType).getInstance(
                        indexes.get(i)));
            }
            confusionMatrix[classificationResult[i]][
                    testLabelArray[indexes.get(i)]]++;
            if (classificationResult[i] == testLabelArray[indexes.get(i)]) {
                correctPointClassificationArray[indexes.get(i)]++;
            }
        }
        ClassificationEstimator estimator =
                new ClassificationEstimator(confusionMatrix);
        estimator.calculateEstimates();
        return estimator;
    }

    @Override
    public ClassificationEstimator test(ArrayList<Integer> indexes,
            Object dataType, int[] testLabelArray, int numClasses,
            float[][] pointDistances, int[][] pointNeighbors) throws Exception {
        int[] classificationResult = new int[indexes.size()];
        float[][] confusionMatrix = new float[numClasses][numClasses];
        for (int i = 0; i < indexes.size(); i++) {
            if (this instanceof DiscreteNeighborPointsQueryUserInterface) {
                classificationResult[i] =
                        ((DiscreteNeighborPointsQueryUserInterface) this).
                        classify(((DiscretizedDataSet) dataType).getInstance(
                        indexes.get(i)), pointDistances[i], pointNeighbors[i]);
            } else if (this instanceof DiscreteDistToPointsQueryUserInterface) {
                classificationResult[i] = (
                        (DistToPointsQueryUserInterface) this).classify(
                        ((DataSet) dataType).getInstance(indexes.get(i)),
                        pointDistances[i]);
            } else {
                classificationResult[i] = classify(
                        ((DiscretizedDataSet) dataType).getInstance(
                        indexes.get(i)));
            }
            confusionMatrix[classificationResult[i]][testLabelArray[
                    indexes.get(i)]]++;
        }
        ClassificationEstimator estimator =
                new ClassificationEstimator(confusionMatrix);
        estimator.calculateEstimates();
        return estimator;
    }
    
    /**
     * This method evaluates the trained model on the test data.
     *
     * @param predictedLabelsAllData int[] representing the current predicted
     * labels for all data points (not only the test points in the current
     * iteration, but rather all points from the original data.)
     * @param correctPointClassificationArray float[] that is updated with the
     * total point-wise classification precision.
     * @param testClasses DiscreteCategory[] representing the test data.
     * @return ClassificationEstimator holding the classifier evaluation quality
     * metrics.
     * @throws Exception
     */
    public ClassificationEstimator test(int[] predictedLabelsAllData,
            float[] correctPointClassificationArray,
            DiscreteCategory[] testClasses) throws Exception {
        if ((testClasses == null) || (testClasses.length == 0)) {
            return null;
        } else {
            int[] classificationResult;
            float[][] confusionMatrix = new float[testClasses.length][
                    testClasses.length];
            for (int cIndex = 0; cIndex < testClasses.length; cIndex++) {
                classificationResult = classify(testClasses[cIndex].getData());
                if (classificationResult != null) {
                    for (int i = 0; i < classificationResult.length; i++) {
                        confusionMatrix[cIndex][classificationResult[i]]++;
                        if (classificationResult[i] == cIndex) {
                            correctPointClassificationArray[
                                    testClasses[cIndex].indexes.get(i)]++;
                        }
                        predictedLabelsAllData[
                                testClasses[cIndex].indexes.get(i)] =
                                classificationResult[i];
                    }
                }
            }
            ClassificationEstimator estimator =
                    new ClassificationEstimator(confusionMatrix);
            estimator.calculateEstimates();
            return estimator;
        }
    }
    
    @Override
    public ClassificationEstimator test(int[] predictedLabelsAllData,
            float[] correctPointClassificationArray,
            ArrayList<Integer> indexes, Object dataType, int numClasses)
            throws Exception {
        if (indexes != null && !indexes.isEmpty()) {
            DiscreteCategory[] testClasses;
            DiscretizedDataInstance instance;
            DiscretizedDataSet discDSet = (DiscretizedDataSet) dataType;
            testClasses = new DiscreteCategory[numClasses];
            for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                testClasses[cIndex] =
                        new DiscreteCategory("number" + cIndex, discDSet, 200);
            }
            for (int i = 0; i < indexes.size(); i++) {
                instance = discDSet.data.get(indexes.get(i));
                testClasses[instance.getCategory()].indexes.add(indexes.get(i));
            }
            return test(predictedLabelsAllData, correctPointClassificationArray,
                    testClasses);
        } else {
            return null;
        }
    }

    @Override
    public ClassificationEstimator test(int[] predictedLabelsAllData,
            float[] correctPointClassificationArray,
            ArrayList<Integer> indexes, Object dataType, int[] testLabelArray,
            int numClasses) throws Exception {
        if (indexes != null && !indexes.isEmpty()) {
            DiscreteCategory[] testClasses;
            DiscretizedDataInstance instance;
            DiscretizedDataSet discDSet = (DiscretizedDataSet) dataType;
            testClasses = new DiscreteCategory[numClasses];
            for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                testClasses[cIndex] =
                        new DiscreteCategory("number" + cIndex, discDSet, 200);
            }
            for (int i = 0; i < indexes.size(); i++) {
                instance = discDSet.data.get(indexes.get(i));
                testClasses[testLabelArray[indexes.get(i)]].indexes.add(
                        indexes.get(i));
                testClasses[instance.getCategory()].indexes.add(indexes.get(i));
            }
            return test(predictedLabelsAllData, correctPointClassificationArray,
                    testClasses);
        } else {
            return null;
        }
    }
    
    @Override
    public ClassificationEstimator test(int[] predictedLabelsAllData,
            float[] correctPointClassificationArray,
            ArrayList<Integer> indexes, Object dataType, int numClasses,
            float[][] pointDistances) throws Exception {
        int[] classificationResult = null;
        float[][] confusionMatrix = new float[numClasses][numClasses];
        for (int i = 0; i < indexes.size(); i++) {
            if (this instanceof DiscreteDistToPointsQueryUserInterface) {
                classificationResult[i] =
                        ((DiscreteDistToPointsQueryUserInterface) this).
                        classify(((DiscretizedDataSet) dataType).getInstance(
                        indexes.get(i)), pointDistances[i]);
            } else {
                classificationResult[i] =
                        classify(((DiscretizedDataSet) dataType).getInstance(
                        indexes.get(i)));
            }
            confusionMatrix[classificationResult[i]][
                    ((DiscretizedDataSet) dataType).getLabelOf(
                    indexes.get(i))]++;
            if (classificationResult[i]
                    == ((DiscretizedDataSet) dataType).getLabelOf(
                    indexes.get(i))) {
                correctPointClassificationArray[indexes.get(i)]++;
            }
            predictedLabelsAllData[indexes.get(i)] = classificationResult[i];
        }
        ClassificationEstimator estimator =
                new ClassificationEstimator(confusionMatrix);
        estimator.calculateEstimates();
        return estimator;
    }

    @Override
    public ClassificationEstimator test(int[] predictedLabelsAllData,
            float[] correctPointClassificationArray,
            ArrayList<Integer> indexes, Object dataType, int[] testLabelArray,
            int numClasses, float[][] pointDistances) throws Exception {
        int[] classificationResult = new int[indexes.size()];
        float[][] confusionMatrix = new float[numClasses][numClasses];
        for (int i = 0; i < indexes.size(); i++) {
            if (this instanceof DiscreteDistToPointsQueryUserInterface) {
                classificationResult[i] =
                        ((DiscreteDistToPointsQueryUserInterface) this).
                        classify(((DiscretizedDataSet) dataType).getInstance(
                        indexes.get(i)), pointDistances[i]);
            } else {
                classificationResult[i] = classify(
                        ((DiscretizedDataSet) dataType).getInstance(
                        indexes.get(i)));
            }
            confusionMatrix[classificationResult[i]][testLabelArray[
                    indexes.get(i)]]++;
            if (classificationResult[i] == testLabelArray[indexes.get(i)]) {
                correctPointClassificationArray[indexes.get(i)]++;
            }
            predictedLabelsAllData[indexes.get(i)] = classificationResult[i];
        }
        ClassificationEstimator estimator =
                new ClassificationEstimator(confusionMatrix);
        estimator.calculateEstimates();
        return estimator;
    }

    @Override
    public ClassificationEstimator test(int[] predictedLabelsAllData,
            float[] correctPointClassificationArray,
            ArrayList<Integer> indexes, Object dataType, int numClasses,
            float[][] pointDistances, int[][] pointNeighbors) throws Exception {
        int[] classificationResult = null;
        float[][] confusionMatrix = new float[numClasses][numClasses];
        for (int i = 0; i < indexes.size(); i++) {
            if (this instanceof DiscreteNeighborPointsQueryUserInterface) {
                classificationResult[i] =
                        ((DiscreteNeighborPointsQueryUserInterface) this).
                        classify(((DiscretizedDataSet) dataType).getInstance(
                        indexes.get(i)), pointDistances[i], pointNeighbors[i]);
            } else if (this instanceof DiscreteDistToPointsQueryUserInterface) {
                classificationResult[i] =
                        ((DiscreteDistToPointsQueryUserInterface) this).
                        classify(((DiscretizedDataSet) dataType).getInstance(
                        indexes.get(i)), pointDistances[i]);
            } else {
                classificationResult[i] = classify(
                        ((DiscretizedDataSet) dataType).getInstance(
                        indexes.get(i)));
            }
            confusionMatrix[classificationResult[i]][
                    ((DiscretizedDataSet) dataType).getLabelOf(
                    indexes.get(i))]++;
            if (classificationResult[i]
                    == ((DiscretizedDataSet) dataType).getLabelOf(
                    indexes.get(i))) {
                correctPointClassificationArray[indexes.get(i)]++;
            }
            predictedLabelsAllData[indexes.get(i)] = classificationResult[i];
        }
        ClassificationEstimator estimator =
                new ClassificationEstimator(confusionMatrix);
        estimator.calculateEstimates();
        return estimator;
    }

    @Override
    public ClassificationEstimator test(int[] predictedLabelsAllData,
            float[] correctPointClassificationArray,
            ArrayList<Integer> indexes, Object dataType, int[] testLabelArray,
            int numClasses, float[][] pointDistances, int[][] pointNeighbors)
            throws Exception {
        int[] classificationResult = new int[indexes.size()];
        float[][] confusionMatrix = new float[numClasses][numClasses];
        for (int i = 0; i < indexes.size(); i++) {
            if (this instanceof DiscreteNeighborPointsQueryUserInterface) {
                classificationResult[i] =
                        ((DiscreteNeighborPointsQueryUserInterface) this).
                        classify(((DiscretizedDataSet) dataType).getInstance(
                        indexes.get(i)), pointDistances[i], pointNeighbors[i]);
            } else if (this instanceof DiscreteDistToPointsQueryUserInterface) {
                classificationResult[i] =
                        ((DiscreteDistToPointsQueryUserInterface) this).
                        classify(((DiscretizedDataSet) dataType).getInstance(
                        indexes.get(i)), pointDistances[i]);
            } else {
                classificationResult[i] = classify(
                        ((DiscretizedDataSet) dataType).getInstance(
                        indexes.get(i)));
            }
            confusionMatrix[classificationResult[i]][
                    testLabelArray[indexes.get(i)]]++;
            if (classificationResult[i] == testLabelArray[indexes.get(i)]) {
                correctPointClassificationArray[indexes.get(i)]++;
            }
            predictedLabelsAllData[indexes.get(i)] = classificationResult[i];
        }
        ClassificationEstimator estimator =
                new ClassificationEstimator(confusionMatrix);
        estimator.calculateEstimates();
        return estimator;
    }
    
    /**
     * This method saves the classifier model.
     * 
     * @param ous OutputStream to write the model to.
     * @throws Exception 
     */
    public void save(ObjectOutputStream ous) throws Exception {
        ous.writeObject(this);
    }

    /**
     * This method loads the classifier.
     * 
     * @param ins InputStream to read the model from.
     * @return DiscreteClassifier that is the loaded model.
     * @throws Exception 
     */
    public static DiscreteClassifier load(ObjectInputStream ins)
            throws Exception {
        DiscreteClassifier loadedModel = (DiscreteClassifier)ins.readObject();
        return loadedModel;
    }
    
    /**
     * This method saves the classifier model.
     * 
     * @param outFile File to write the model to.
     * @throws Exception 
     */
    public void save(File outFile) throws Exception {
        FileUtil.createFile(outFile);
        try (ObjectOutputStream ous =
                new ObjectOutputStream(new FileOutputStream(outFile))) {
            save(ous);
        }
    }

    /**
     * This method loads the classifier.
     * 
     * @param inFile File to load the model from.
     * @return DiscreteClassifier that is the loaded model.
     * @throws Exception 
     */
    public static DiscreteClassifier load(File inFile)
            throws Exception {
        DiscreteClassifier loadedModel;
        try (ObjectInputStream reader = new ObjectInputStream(
                     new FileInputStream(inFile))) {
            loadedModel = load(reader);
        }
        return loadedModel;
    }
}
