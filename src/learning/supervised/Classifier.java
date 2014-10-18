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

import data.representation.DataInstance;
import data.representation.DataSet;
import data.representation.sparse.BOWDataSet;
import data.representation.sparse.BOWInstance;
import distances.primary.CombinedMetric;
import ioformat.FileUtil;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import learning.supervised.evaluation.ClassificationEstimator;
import learning.supervised.evaluation.ValidateableInterface;
import learning.supervised.interfaces.DistToPointsQueryUserInterface;
import learning.supervised.interfaces.NeighborPointsQueryUserInterface;
import preprocessing.instance_selection.InstanceSelector;

/**
 * This class implements the methods used for classifier training and testing.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public abstract class Classifier implements ValidateableInterface,
        Serializable {

    private static final long serialVersionUID = 1L;
    private Category[] trainingClasses = null;
    private CombinedMetric cmet = null;

    @Override
    public abstract ValidateableInterface copyConfiguration();

    @Override
    public void setDataIndexes(ArrayList<Integer> currentIndexes,
            Object dataType) {
        if (currentIndexes != null && currentIndexes.size() > 0) {
            if (dataType instanceof BOWDataSet) {
                BOWDataSet bowDSet = (BOWDataSet) dataType;
                ArrayList<BOWInstance> trueDataVect =
                        new ArrayList<>(currentIndexes.size());
                for (int i = 0; i < currentIndexes.size(); i++) {
                    trueDataVect.add((BOWInstance) (bowDSet.data.get(
                            currentIndexes.get(i))));
                }
                setData(trueDataVect, dataType);
            } else if (dataType instanceof DataSet) {
                DataSet dset = (DataSet) dataType;
                ArrayList<DataInstance> trueDataVect =
                        new ArrayList<>(currentIndexes.size());
                for (int i = 0; i < currentIndexes.size(); i++) {
                    trueDataVect.add(dset.data.get(currentIndexes.get(i)));
                }
                setData(trueDataVect, dataType);
            }
        }
    }

    @Override
    public void setData(ArrayList data, Object dataType) {
        if (data != null && !data.isEmpty()) {
            Category[] catArray = null;
            int numClasses = 0;
            int currClass;
            if (data.get(0) instanceof BOWInstance) {
                BOWInstance instance;
                BOWDataSet bowDSet = (BOWDataSet) dataType;
                BOWDataSet bowDSetCopy = bowDSet.cloneDefinition();
                bowDSetCopy.data = data;
                for (int i = 0; i < data.size(); i++) {
                    if (data.get(i) != null) {
                        instance = (BOWInstance) (data.get(i));
                        currClass = instance.getCategory();
                        if (currClass > numClasses) {
                            numClasses = currClass;
                        }
                    }
                }
                numClasses = numClasses + 1;
                catArray = new Category[numClasses];
                for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                    catArray[cIndex] = new Category("number" + cIndex, 200,
                            bowDSet);
                    catArray[cIndex].setDefinitionDataset(bowDSetCopy);
                }
                for (int i = 0; i < data.size(); i++) {
                    if (data.get(i) != null) {
                        instance = (BOWInstance) (data.get(i));
                        currClass = instance.getCategory();
                        catArray[currClass].addInstance(i);
                    }
                }
            } else if (data.get(0) instanceof DataInstance) {
                DataInstance instance;
                DataSet dset = (DataSet) dataType;
                DataSet dsetCopy = dset.cloneDefinition();
                dsetCopy.data = data;
                for (int i = 0; i < data.size(); i++) {
                    if (data.get(i) != null) {
                        instance = (DataInstance) (data.get(i));
                        currClass = instance.getCategory();
                        if (currClass > numClasses) {
                            numClasses = currClass;
                        }
                    }
                }
                numClasses = numClasses + 1;
                catArray = new Category[numClasses];
                for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                    catArray[cIndex] = new Category("number" + cIndex, 200,
                            dset);
                    catArray[cIndex].setDefinitionDataset(dsetCopy);
                }
                for (int i = 0; i < data.size(); i++) {
                    if (data.get(i) != null) {
                        instance = (DataInstance) (data.get(i));
                        currClass = instance.getCategory();
                        catArray[currClass].addInstance(i);
                    }
                }
            }
            setClasses(catArray);
        }
    }

    @Override
    public void setClasses(Object[] dataClasses) {
        if (dataClasses == null || dataClasses.length == 0) {
            return;
        }
        Category[] catArray = new Category[dataClasses.length];
        for (int cIndex = 0; cIndex < dataClasses.length; cIndex++) {
            catArray[cIndex] = (Category) (dataClasses[cIndex]);
        }
        setClasses(catArray);
    }

    /**
     * @param dataClasses Category[] representing the training data.
     */
    public void setClasses(Category[] dataClasses) {
        this.trainingClasses = dataClasses;
    }

    /**
     * @return Category[] representing the training data.
     */
    public Category[] getClasses() {
        return trainingClasses;
    }

    /**
     * @param cmet CombinedMetric object for distance calculations.
     */
    public void setCombinedMetric(CombinedMetric cmet) {
        this.cmet = cmet;
    }

    /**
     * @return CombinedMetric object for distance calculations.
     */
    public CombinedMetric getCombinedMetric() {
        return cmet;
    }

    /**
     * This method runs the classifier training.
     */
    public void run() {
        try {
            train();
        } catch (Exception e) {
            System.err.println(e.getMessage());
        }
    }

    @Override
    public abstract void train() throws Exception;

    @Override
    public void trainOnReducedData(InstanceSelector reducer) throws Exception {
        train();
    }

    public abstract int classify(DataInstance instance) throws Exception;

    public abstract float[] classifyProbabilistically(DataInstance instance)
            throws Exception;

    /**
     * This method performs batch classification of an array of DataInstance
     * objects.
     *
     * @param instances DataInstance[] array to classify.
     * @return int[] that are the resulting predicted class affiliations.
     * @throws Exception
     */
    public int[] classify(DataInstance[] instances) throws Exception {
        int[] classificationResults;
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
     * This method performs batch classification of an array of DataInstance
     * objects.
     *
     * @param instances DataInstance[] array to classify.
     * @return float[][] that are the resulting predicted probabilistic class
     * assignments.
     * @throws Exception
     */
    public float[][] classifyProbabilistically(DataInstance[] instances)
            throws Exception {
        float[][] classificationResults;
        if ((instances == null) || (instances.length == 0)) {
            return null;
        } else {
            classificationResults = new float[instances.length][];
            for (int i = 0; i < instances.length; i++) {
                classificationResults[i] = classifyProbabilistically(
                        instances[i]);
            }
            return classificationResults;
        }
    }

    /**
     * This method performs batch classification of a list of DataInstance
     * objects.
     *
     * @param instances ArrayList<DataInstance> to classify.
     * @return int[] that are the resulting predicted class affiliations.
     * @throws Exception
     */
    public int[] classify(ArrayList<DataInstance> instances) throws Exception {
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
     * This method performs batch classification of a list of DataInstance
     * objects.
     *
     * @param instances ArrayList<DataInstance> to classify.
     * @return float[][] that are the resulting predicted probabilistic class
     * assignments.
     * @throws Exception
     */
    public float[][] classifyProbabilistically(
            ArrayList<DataInstance> instances) throws Exception {
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
        return test((Category[]) testClasses);
    }

    /**
     * This method tests and evaluates the classifier.
     *
     * @param dataClasses Array of data categories representing the test data.
     * @return ClassificationEstimator containing the resulting classification
     * quality measures.
     * @throws Exception
     */
    public ClassificationEstimator test(Category[] dataClasses) throws Exception {
        if ((dataClasses == null) || (dataClasses.length == 0)) {
            return null;
        } else {
            int[] classificationResult;
            float[][] confusionMatrix = new float[dataClasses.length][
                    dataClasses.length];
            for (int cIndex = 0; cIndex < dataClasses.length; cIndex++) {
                classificationResult = classify(dataClasses[cIndex].
                        getAllInstances());
                if (classificationResult != null) {
                    for (int i = 0; i < classificationResult.length; i++) {
                        confusionMatrix[classificationResult[i]][cIndex]++;
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
     * This method tests and evaluates the classifier.
     *
     * @param correctPointClassificationArray float[] that is updated with the
     * total point-wise classification precision.
     * @param dataClasses Array of data categories representing the test data.
     * @return ClassificationEstimator containing the resulting classification
     * quality measures.
     * @throws Exception
     */
    public ClassificationEstimator test(float[] correctPointClassificationArray,
            Category[] dataClasses) throws Exception {
        if ((dataClasses == null) || (dataClasses.length == 0)) {
            return null;
        } else {
            int[] classificationResult;
            float[][] confusionMatrix = new float[dataClasses.length][
                    dataClasses.length];
            for (int Cindex = 0; Cindex < dataClasses.length; Cindex++) {
                classificationResult = classify(dataClasses[Cindex].
                        getAllInstances());
                if (classificationResult != null) {
                    for (int i = 0; i < classificationResult.length; i++) {
                        confusionMatrix[classificationResult[i]][Cindex]++;
                        if (classificationResult[i] == Cindex) {
                            correctPointClassificationArray[
                                    dataClasses[Cindex].indexes.get(i)]++;
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
            Object dataType, int numClasses, float[][] pointDistances)
            throws Exception {
        int[] classificationResult = null;
        float[][] confusionMatrix = new float[numClasses][numClasses];
        for (int i = 0; i < indexes.size(); i++) {
            if (this instanceof DistToPointsQueryUserInterface) {
                classificationResult[i] =
                        ((DistToPointsQueryUserInterface) this).classify(
                        ((DataSet) dataType).getInstance(indexes.get(i)),
                        pointDistances[i]);
            } else {
                classificationResult[i] = classify(((DataSet) dataType).
                        getInstance(indexes.get(i)));
            }
            confusionMatrix[classificationResult[i]][((DataSet) dataType).
                    getLabelOf(indexes.get(i))]++;
        }
        ClassificationEstimator estimator =
                new ClassificationEstimator(confusionMatrix);
        estimator.calculateEstimates();
        return estimator;
    }

    @Override
    public ClassificationEstimator test(ArrayList<Integer> indexes,
            Object dataType, int[] testLabelArray, int numClasses,
            float[][] pointDistances) throws Exception {
        int[] classificationResult = new int[indexes.size()];
        float[][] confusionMatrix = new float[numClasses][numClasses];
        for (int i = 0; i < indexes.size(); i++) {
            if (this instanceof DistToPointsQueryUserInterface) {
                classificationResult[i] =
                        ((DistToPointsQueryUserInterface) this).classify(
                        ((DataSet) dataType).getInstance(indexes.get(i)),
                        pointDistances[i]);
            } else {
                classificationResult[i] = classify(((DataSet) dataType).
                        getInstance(indexes.get(i)));
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
    public ClassificationEstimator test(ArrayList<Integer> indexes,
            Object dataType, int[] testLabelArray, int numClasses,
            float[][] pointDistances, int[][] pointNeighbors) throws Exception {
        int[] classificationResult = new int[indexes.size()];
        float[][] confusionMatrix = new float[numClasses][numClasses];
        for (int i = 0; i < indexes.size(); i++) {
            if (this instanceof NeighborPointsQueryUserInterface) {
                classificationResult[i] =
                        ((NeighborPointsQueryUserInterface) this).classify(
                        ((DataSet) dataType).getInstance(indexes.get(i)),
                        pointDistances[i], pointNeighbors[i]);
            } else if (this instanceof DistToPointsQueryUserInterface) {
                classificationResult[i] =
                        ((DistToPointsQueryUserInterface) this).classify(
                        ((DataSet) dataType).getInstance(indexes.get(i)),
                        pointDistances[i]);
            } else {
                classificationResult[i] =
                        classify(((DataSet) dataType).getInstance(
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

    public ClassificationEstimator test(DataSet dset) throws Exception {
        ArrayList<Integer> indexes = new ArrayList<>(dset.size());
        for (int i = 0; i < dset.size(); i++) {
            indexes.add(i);
        }
        return test(indexes, dset, dset.countCategories());
    }

    @Override
    public ClassificationEstimator test(ArrayList<Integer> indexes,
            Object dataType, int numClasses) throws Exception {
        if (indexes != null && !indexes.isEmpty()) {
            Category[] testClasses = null;
            if (dataType instanceof BOWDataSet) {
                BOWInstance instance;
                BOWDataSet bowDset = (BOWDataSet) dataType;
                testClasses = new Category[numClasses];
                for (int i = 0; i < numClasses; i++) {
                    testClasses[i] = new Category("number" + i, 200, bowDset);
                }
                for (int i = 0; i < indexes.size(); i++) {
                    instance = (BOWInstance) bowDset.data.get(indexes.get(i));
                    testClasses[instance.getCategory()].addInstance(
                            indexes.get(i));
                }
            } else if (dataType instanceof DataSet) {
                DataInstance instance;
                DataSet dset = (DataSet) dataType;
                testClasses = new Category[numClasses];
                for (int i = 0; i < numClasses; i++) {
                    testClasses[i] = new Category("number" + i, 200, dset);
                }
                for (int i = 0; i < indexes.size(); i++) {
                    instance = dset.data.get(indexes.get(i));
                    testClasses[instance.getCategory()].addInstance(
                            indexes.get(i));
                }
            }
            return test(testClasses);
        } else {
            return null;
        }
    }

    @Override
    public ClassificationEstimator test(ArrayList<Integer> indexes,
            Object dataType, int[] testLabelArray, int numClasses) throws Exception {
        if (indexes != null && !indexes.isEmpty()) {
            Category[] testClasses = null;
            if (dataType instanceof BOWDataSet) {
                BOWDataSet bowDset = (BOWDataSet) dataType;
                testClasses = new Category[numClasses];
                for (int i = 0; i < numClasses; i++) {
                    testClasses[i] = new Category("number" + i, 200, bowDset);
                }
                for (int i = 0; i < indexes.size(); i++) {
                    testClasses[testLabelArray[indexes.get(i)]].addInstance(
                            indexes.get(i));
                }
            } else if (dataType instanceof DataSet) {
                DataSet dset = (DataSet) dataType;
                testClasses = new Category[numClasses];
                for (int i = 0; i < numClasses; i++) {
                    testClasses[i] = new Category("number" + i, 200, dset);
                }
                for (int i = 0; i < indexes.size(); i++) {
                    testClasses[testLabelArray[indexes.get(i)]].addInstance(
                            indexes.get(i));
                }
            }
            return test(testClasses);
        } else {
            return null;
        }
    }

    @Override
    public ClassificationEstimator test(float[] correctPointClassificationArray,
            ArrayList<Integer> indexes, Object dataType, int numClasses,
            float[][] pointDistances) throws Exception {
        int[] classificationResult = new int[indexes.size()];
        float[][] confusionMatrix = new float[numClasses][numClasses];
        for (int i = 0; i < indexes.size(); i++) {
            if (this instanceof DistToPointsQueryUserInterface) {
                classificationResult[i] =
                        ((DistToPointsQueryUserInterface) this).classify(
                        ((DataSet) dataType).getInstance(indexes.get(i)),
                        pointDistances[i]);
            } else {
                classificationResult[i] =
                        classify(((DataSet) dataType).getInstance(
                        indexes.get(i)));
            }
            confusionMatrix[classificationResult[i]][((DataSet) dataType).
                    getLabelOf(indexes.get(i))]++;
            if (classificationResult[i] == ((DataSet) dataType).getLabelOf(
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
            if (this instanceof DistToPointsQueryUserInterface) {
                classificationResult[i] =
                        ((DistToPointsQueryUserInterface) this).classify(
                        ((DataSet) dataType).getInstance(indexes.get(i)),
                        pointDistances[i]);
            } else {
                classificationResult[i] =
                        classify(((DataSet) dataType).getInstance(
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
    public ClassificationEstimator test(
            float[] correctPointClassificationArray, ArrayList<Integer> indexes,
            Object dataType, int numClasses, float[][] pointDistances,
            int[][] pointNeighbors) throws Exception {
        int[] classificationResult = new int[indexes.size()];
        float[][] confusionMatrix = new float[numClasses][numClasses];
        for (int i = 0; i < indexes.size(); i++) {
            if (this instanceof NeighborPointsQueryUserInterface) {
                classificationResult[i] =
                        ((NeighborPointsQueryUserInterface) this).classify(
                        ((DataSet) dataType).getInstance(indexes.get(i)),
                        pointDistances[i], pointNeighbors[i]);
            } else if (this instanceof DistToPointsQueryUserInterface) {
                classificationResult[i] =
                        ((DistToPointsQueryUserInterface) this).classify(
                        ((DataSet) dataType).getInstance(indexes.get(i)),
                        pointDistances[i]);
            } else {
                classificationResult[i] =
                        classify(((DataSet) dataType).getInstance(
                        indexes.get(i)));
            }
            confusionMatrix[classificationResult[i]][((DataSet) dataType).
                    getLabelOf(indexes.get(i))]++;
            if (classificationResult[i] == ((DataSet) dataType).getLabelOf(
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
    public ClassificationEstimator test(
            float[] correctPointClassificationArray,
            ArrayList<Integer> indexes, Object dataType,
            int[] testLabelArray, int numClasses, float[][] pointDistances,
            int[][] pointNeighbors) throws Exception {
        int[] classificationResult = new int[indexes.size()];
        float[][] confusionMatrix = new float[numClasses][numClasses];
        for (int i = 0; i < indexes.size(); i++) {
            if (this instanceof NeighborPointsQueryUserInterface) {
                classificationResult[i] =
                        ((NeighborPointsQueryUserInterface) this).classify(
                        ((DataSet) dataType).getInstance(indexes.get(i)),
                        pointDistances[i], pointNeighbors[i]);
            } else if (this instanceof DistToPointsQueryUserInterface) {
                classificationResult[i] =
                        ((DistToPointsQueryUserInterface) this).classify(
                        ((DataSet) dataType).getInstance(indexes.get(i)),
                        pointDistances[i]);
            } else {
                classificationResult[i] =
                        classify(((DataSet) dataType).getInstance(
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
            ArrayList<Integer> indexes, Object dataType, int numClasses)
            throws Exception {
        if (indexes != null && !indexes.isEmpty()) {
            Category[] testClasses = null;
            if (dataType instanceof BOWDataSet) {
                BOWInstance instance;
                BOWDataSet bowDSet = (BOWDataSet) dataType;
                testClasses = new Category[numClasses];
                for (int i = 0; i < numClasses; i++) {
                    testClasses[i] = new Category("number" + i, 200, bowDSet);
                }
                for (int i = 0; i < indexes.size(); i++) {
                    instance = (BOWInstance) bowDSet.data.get(indexes.get(i));
                    testClasses[instance.getCategory()].addInstance(
                            indexes.get(i));
                }
            } else if (dataType instanceof DataSet) {
                DataInstance instance;
                DataSet dset = (DataSet) dataType;
                testClasses = new Category[numClasses];
                for (int i = 0; i < numClasses; i++) {
                    testClasses[i] = new Category("number" + i, 200, dset);
                }
                for (int i = 0; i < indexes.size(); i++) {
                    instance = dset.data.get(indexes.get(i));
                    testClasses[instance.getCategory()].addInstance(
                            indexes.get(i));
                }
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
            Category[] testClasses = null;
            if (dataType instanceof BOWDataSet) {
                BOWDataSet bowDSet = (BOWDataSet) dataType;
                testClasses = new Category[numClasses];
                for (int i = 0; i < numClasses; i++) {
                    testClasses[i] = new Category("number" + i, 200, bowDSet);
                }
                for (int i = 0; i < indexes.size(); i++) {
                    testClasses[testLabelArray[indexes.get(i)]].addInstance(
                            indexes.get(i));
                }
            } else if (dataType instanceof DataSet) {
                DataSet dset = (DataSet) dataType;
                testClasses = new Category[numClasses];
                for (int i = 0; i < numClasses; i++) {
                    testClasses[i] = new Category("number" + i, 200, dset);
                }
                for (int i = 0; i < indexes.size(); i++) {
                    testClasses[testLabelArray[indexes.get(i)]].addInstance(
                            indexes.get(i));
                }
            }
            return test(correctPointClassificationArray, testClasses);
        } else {
            return null;
        }
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
     * @return Classifier that is the loaded model.
     * @throws Exception 
     */
    public static Classifier load(ObjectInputStream ins)
            throws Exception {
        Classifier loadedModel = (Classifier)ins.readObject();
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
     * @return Classifier that is the loaded model.
     * @throws Exception 
     */
    public static Classifier load(File inFile)
            throws Exception {
        Classifier loadedModel;
        try (ObjectInputStream reader = new ObjectInputStream(
                     new FileInputStream(inFile))) {
            loadedModel = load(reader);
        }
        return loadedModel;
    }
}