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
package learning.supervised.evaluation;

import java.util.ArrayList;
import preprocessing.instance_selection.InstanceSelector;

/**
 * This interface declares the methods needed for successful classifier
 * evaluation, for both continuous and discretized data classifiers.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public interface ValidateableInterface {

    /**
     * @return String that is the classifier name.
     */
    public String getName();

    /**
     * @param dataClasses Array of categories to set as the training data. It
     * can correspond to continuous or discretized categories.
     */
    public void setClasses(Object[] dataClasses);

    /**
     * This method sets the training data array to the classifier.
     *
     * @param data ArrayList of data points.
     * @param dataType Object that is the data context. It can correspond to
     * dense and sparse DataSet objects, as well as a DiscretizedDataSet data
     * context.
     */
    public void setData(ArrayList data, Object dataType);

    /**
     * This method sets the training data by indexes pointing to the data
     * context.
     *
     * @param currentIndexes ArrayList<Integer> of training data indexes.
     * @param dataType Object that is the data context. It can correspond to
     * dense and sparse DataSet objects, as well as a DiscretizedDataSet data
     * context.
     */
    public void setDataIndexes(ArrayList<Integer> currentIndexes,
            Object dataType);

    /**
     * This method tests and evaluates the classifier.
     *
     * @param indexes ArrayList<Integer> of test data indexes.
     * @param dataType Object that is the test data context. It can correspond
     * to dense and sparse DataSet objects, as well as a DiscretizedDataSet data
     * context.
     * @param numClasses Integer that is the number of classes in the data.
     * @return ClassificationEstimator containing the resulting classification
     * quality measures.
     * @throws Exception
     */
    public ClassificationEstimator test(ArrayList<Integer> indexes,
            Object dataType, int numClasses) throws Exception;

    /**
     * This method tests and evaluates the classifier.
     *
     * @param indexes ArrayList<Integer> of test data indexes.
     * @param dataType Object that is the test data context. It can correspond
     * to dense and sparse DataSet objects, as well as a DiscretizedDataSet data
     * context.
     * @param testLabelArray int[] representing the separate test data label
     * array.
     * @param numClasses Integer that is the number of classes in the data.
     * @return ClassificationEstimator containing the resulting classification
     * quality measures.
     * @throws Exception
     */
    public ClassificationEstimator test(ArrayList<Integer> indexes,
            Object dataType, int[] testLabelArray, int numClasses)
            throws Exception;

    /**
     * This method tests and evaluates the classifier.
     *
     * @param indexes ArrayList<Integer> of test data indexes.
     * @param dataType Object that is the test data context. It can correspond
     * to dense and sparse DataSet objects, as well as a DiscretizedDataSet data
     * context.
     * @param numClasses Integer that is the number of classes in the data.
     * @param trainingToTestDistances float[][] representing a matrix of
     * distances from the training points to the test points.
     * @return ClassificationEstimator containing the resulting classification
     * quality measures.
     * @throws Exception
     */
    public ClassificationEstimator test(ArrayList<Integer> indexes,
            Object dataType, int numClasses,
            float[][] trainingToTestDistances) throws Exception;

    /**
     * This method tests and evaluates the classifier.
     *
     * @param indexes ArrayList<Integer> of test data indexes.
     * @param dataType Object that is the test data context. It can correspond
     * to dense and sparse DataSet objects, as well as a DiscretizedDataSet data
     * context.
     * @param testLabelArray int[] representing the separate test data label
     * array.
     * @param numClasses Integer that is the number of classes in the data.
     * @param trainingToTestDistances float[][] representing a matrix of
     * distances from the training points to the test points.
     * @return ClassificationEstimator containing the resulting classification
     * quality measures.
     * @throws Exception
     */
    public ClassificationEstimator test(ArrayList<Integer> indexes,
            Object dataType, int[] testLabelArray, int numClasses,
            float[][] trainingToTestDistances) throws Exception;

    /**
     * This method tests and evaluates the classifier, while keeping track of
     * correct classifications for each class.
     *
     * @param correctPointClassificationCounter float[] representing the counter
     * of correct class-specific classifications, for latter analysis.
     * @param indexes ArrayList<Integer> of test data indexes.
     * @param dataType Object that is the test data context. It can correspond
     * to dense and sparse DataSet objects, as well as a DiscretizedDataSet data
     * context.
     * @param numClasses Integer that is the number of classes in the data.
     * @return ClassificationEstimator containing the resulting classification
     * quality measures.
     * @throws Exception
     */
    public ClassificationEstimator test(
            float[] correctPointClassificationCounter,
            ArrayList<Integer> indexes, Object dataType, int numClasses)
            throws Exception;

    /**
     * This method tests and evaluates the classifier, while keeping track of
     * correct classifications for each class.
     *
     * @param correctPointClassificationCounter float[] representing the counter
     * of correct class-specific classifications, for latter analysis.
     * @param indexes ArrayList<Integer> of test data indexes.
     * @param dataType Object that is the test data context. It can correspond
     * to dense and sparse DataSet objects, as well as a DiscretizedDataSet data
     * context.
     * @param testLabelArray int[] representing the separate test data label
     * array.
     * @param numClasses Integer that is the number of classes in the data.
     * @return ClassificationEstimator containing the resulting classification
     * quality measures.
     * @throws Exception
     */
    public ClassificationEstimator test(
            float[] correctPointClassificationCounter,
            ArrayList<Integer> indexes, Object dataType,
            int[] testLabelArray, int numClasses) throws Exception;

    /**
     * This method tests and evaluates the classifier, while keeping track of
     * correct classifications for each class.
     *
     * @param correctPointClassificationCounter float[] representing the counter
     * of correct class-specific classifications, for latter analysis.
     * @param indexes ArrayList<Integer> of test data indexes.
     * @param dataType Object that is the test data context. It can correspond
     * to dense and sparse DataSet objects, as well as a DiscretizedDataSet data
     * context.
     * @param numClasses Integer that is the number of classes in the data.
     * @param trainingToTestDistances float[][] representing a matrix of
     * distances from the training points to the test points.
     * @return ClassificationEstimator containing the resulting classification
     * quality measures.
     * @throws Exception
     */
    public ClassificationEstimator test(
            float[] correctPointClassificationCounter,
            ArrayList<Integer> indexes, Object dataType,
            int numClasses, float[][] trainingToTestDistances) throws Exception;

    /**
     * This method tests and evaluates the classifier, while keeping track of
     * correct classifications for each class.
     *
     * @param correctPointClassificationCounter float[] representing the counter
     * of correct class-specific classifications, for latter analysis.
     * @param indexes ArrayList<Integer> of test data indexes.
     * @param dataType Object that is the test data context. It can correspond
     * to dense and sparse DataSet objects, as well as a DiscretizedDataSet data
     * context.
     * @param testLabelArray int[] representing the separate test data label
     * array.
     * @param numClasses Integer that is the number of classes in the data.
     * @param trainingToTestDistances float[][] representing a matrix of
     * distances from the training points to the test points.
     * @return ClassificationEstimator containing the resulting classification
     * quality measures.
     * @throws Exception
     */
    public ClassificationEstimator test(
            float[] correctPointClassificationCounter,
            ArrayList<Integer> indexes, Object dataType,
            int[] testLabelArray, int numClasses,
            float[][] trainingToTestDistances) throws Exception;

    /**
     * This method tests and evaluates the classifier.
     *
     * @param indexes ArrayList<Integer> of test data indexes.
     * @param dataType Object that is the test data context. It can correspond
     * to dense and sparse DataSet objects, as well as a DiscretizedDataSet data
     * context.
     * @param testLabelArray int[] representing the separate test data label
     * array.
     * @param numClasses Integer that is the number of classes in the data.
     * @param trainingToTestDistances float[][] representing a matrix of
     * distances from the training points to the test points.
     * @param pointNeighbors int[][] representing and array of arrays of
     * k-nearest neighbor indexes for the test data points among the training
     * data, for kNN-based classification methods.
     * @return ClassificationEstimator containing the resulting classification
     * quality measures.
     * @throws Exception
     */
    public ClassificationEstimator test(
            ArrayList<Integer> indexes, Object dataType,
            int[] testLabelArray, int numClasses,
            float[][] trainingToTestDistances,
            int[][] pointNeighbors) throws Exception;

    /**
     * This method tests and evaluates the classifier, while keeping track of
     * correct classifications for each class.
     *
     * @param correctPointClassificationCounter float[] representing the counter
     * of correct class-specific classifications, for latter analysis.
     * @param indexes ArrayList<Integer> of test data indexes.
     * @param dataType Object that is the test data context. It can correspond
     * to dense and sparse DataSet objects, as well as a DiscretizedDataSet data
     * context.
     * @param numClasses Integer that is the number of classes in the data.
     * @param trainingToTestDistances float[][] representing a matrix of
     * distances from the training points to the test points.
     * @param pointNeighbors int[][] representing and array of arrays of
     * k-nearest neighbor indexes for the test data points among the training
     * data, for kNN-based classification methods.
     * @return ClassificationEstimator containing the resulting classification
     * quality measures.
     * @throws Exception
     */
    public ClassificationEstimator test(
            float[] correctPointClassificationCounter,
            ArrayList<Integer> indexes, Object dataType, int numClasses,
            float[][] trainingToTestDistances, int[][] pointNeighbors)
            throws Exception;

    /**
     * This method tests and evaluates the classifier, while keeping track of
     * correct classifications for each class.
     *
     * @param correctPointClassificationCounter float[] representing the counter
     * of correct class-specific classifications, for latter analysis.
     * @param indexes ArrayList<Integer> of test data indexes.
     * @param dataType Object that is the test data context. It can correspond
     * to dense and sparse DataSet objects, as well as a DiscretizedDataSet data
     * context.
     * @param testLabelArray int[] representing the separate test data label
     * array.
     * @param numClasses Integer that is the number of classes in the data.
     * @param trainingToTestDistances float[][] representing a matrix of
     * distances from the training points to the test points.
     * @param pointNeighbors int[][] representing and array of arrays of
     * k-nearest neighbor indexes for the test data points among the training
     * data, for kNN-based classification methods.
     * @return ClassificationEstimator containing the resulting classification
     * quality measures.
     * @throws Exception
     */
    public ClassificationEstimator test(
            float[] correctPointClassificationCounter,
            ArrayList<Integer> indexes, Object dataType,
            int[] testLabelArray, int numClasses,
            float[][] trainingToTestDistances,
            int[][] pointNeighbors) throws Exception;

    /**
     * This method tests and evaluates the classifier.
     *
     * @param dataClasses Array of data categories representing the test data.
     * @return ClassificationEstimator containing the resulting classification
     * quality measures.
     * @throws Exception
     */
    public ClassificationEstimator test(Object[] dataClasses) throws Exception;

    /**
     * This method copies the initial classifier configuration, without copying
     * the entire trained model. It essentially copies the parametrized
     * classifier type.
     *
     * @return ValidateableInterface that is the copy of the parametrized
     * initial classifier configuration.
     */
    public ValidateableInterface copyConfiguration();

    /**
     * This method trains the classification model, based on the provided
     * training data and parameters.
     *
     * @throws Exception
     */
    public void train() throws Exception;

    /**
     * This method trains the classification model on the reduced dataset, based
     * on the reduced training data and the additional information contained in
     * the InstanceSelector object.
     *
     * @param reducer InstanceSelector object containing the additional data
     * reduction - related information, including the unbiased neighbor
     * occurrence frequency estimates.
     * @throws Exception
     */
    public void trainOnReducedData(InstanceSelector reducer) throws Exception;
    
    /**
     * This method tests and evaluates the classifier, while keeping track of
     * correct classifications for each class.
     *
     * @param predictedProbLabelsAllData float[][] representing the current
     * predicted fuzzy labels for all data points (not only the test points in
     * the current iteration, but rather all points from the original data.)
     * @param correctPointClassificationCounter float[] representing the counter
     * of correct class-specific classifications, for latter analysis.
     * @param indexes ArrayList<Integer> of test data indexes.
     * @param dataType Object that is the test data context. It can correspond
     * to dense and sparse DataSet objects, as well as a DiscretizedDataSet data
     * context.
     * @param numClasses Integer that is the number of classes in the data.
     * @return ClassificationEstimator containing the resulting classification
     * quality measures.
     * @throws Exception
     */
    public ClassificationEstimator test(
            float[][] predictedProbLabelsAllData,
            float[] correctPointClassificationCounter,
            ArrayList<Integer> indexes, Object dataType, int numClasses)
            throws Exception;

    /**
     * This method tests and evaluates the classifier, while keeping track of
     * correct classifications for each class.
     *
     * @param predictedProbLabelsAllData float[][] representing the current
     * predicted fuzzy labels for all data points (not only the test points in
     * the current iteration, but rather all points from the original data.)
     * @param correctPointClassificationCounter float[] representing the counter
     * of correct class-specific classifications, for latter analysis.
     * @param indexes ArrayList<Integer> of test data indexes.
     * @param dataType Object that is the test data context. It can correspond
     * to dense and sparse DataSet objects, as well as a DiscretizedDataSet data
     * context.
     * @param testLabelArray int[] representing the separate test data label
     * array.
     * @param numClasses Integer that is the number of classes in the data.
     * @return ClassificationEstimator containing the resulting classification
     * quality measures.
     * @throws Exception
     */
    public ClassificationEstimator test(
            float[][] predictedProbLabelsAllData,
            float[] correctPointClassificationCounter,
            ArrayList<Integer> indexes, Object dataType,
            int[] testLabelArray, int numClasses) throws Exception;

    /**
     * This method tests and evaluates the classifier, while keeping track of
     * correct classifications for each class.
     *
     * @param predictedProbLabelsAllData float[][] representing the current
     * predicted fuzzy labels for all data points (not only the test points in
     * the current iteration, but rather all points from the original data.)
     * @param correctPointClassificationCounter float[] representing the counter
     * of correct class-specific classifications, for latter analysis.
     * @param indexes ArrayList<Integer> of test data indexes.
     * @param dataType Object that is the test data context. It can correspond
     * to dense and sparse DataSet objects, as well as a DiscretizedDataSet data
     * context.
     * @param numClasses Integer that is the number of classes in the data.
     * @param trainingToTestDistances float[][] representing a matrix of
     * distances from the training points to the test points.
     * @return ClassificationEstimator containing the resulting classification
     * quality measures.
     * @throws Exception
     */
    public ClassificationEstimator test(
            float[][] predictedProbLabelsAllData,
            float[] correctPointClassificationCounter,
            ArrayList<Integer> indexes, Object dataType,
            int numClasses, float[][] trainingToTestDistances) throws Exception;

    /**
     * This method tests and evaluates the classifier, while keeping track of
     * correct classifications for each class.
     *
     * @param predictedProbLabelsAllData float[][] representing the current
     * predicted fuzzy labels for all data points (not only the test points in
     * the current iteration, but rather all points from the original data.)
     * @param correctPointClassificationCounter float[] representing the counter
     * of correct class-specific classifications, for latter analysis.
     * @param indexes ArrayList<Integer> of test data indexes.
     * @param dataType Object that is the test data context. It can correspond
     * to dense and sparse DataSet objects, as well as a DiscretizedDataSet data
     * context.
     * @param testLabelArray int[] representing the separate test data label
     * array.
     * @param numClasses Integer that is the number of classes in the data.
     * @param trainingToTestDistances float[][] representing a matrix of
     * distances from the training points to the test points.
     * @return ClassificationEstimator containing the resulting classification
     * quality measures.
     * @throws Exception
     */
    public ClassificationEstimator test(
            float[][] predictedProbLabelsAllData,
            float[] correctPointClassificationCounter,
            ArrayList<Integer> indexes, Object dataType,
            int[] testLabelArray, int numClasses,
            float[][] trainingToTestDistances) throws Exception;
    
    /**
     * This method tests and evaluates the classifier, while keeping track of
     * correct classifications for each class.
     *
     * @param predictedProbLabelsAllData float[][] representing the current
     * predicted fuzzy labels for all data points (not only the test points in
     * the current iteration, but rather all points from the original data.)
     * @param correctPointClassificationCounter float[] representing the counter
     * of correct class-specific classifications, for latter analysis.
     * @param indexes ArrayList<Integer> of test data indexes.
     * @param dataType Object that is the test data context. It can correspond
     * to dense and sparse DataSet objects, as well as a DiscretizedDataSet data
     * context.
     * @param numClasses Integer that is the number of classes in the data.
     * @param trainingToTestDistances float[][] representing a matrix of
     * distances from the training points to the test points.
     * @param pointNeighbors int[][] representing and array of arrays of
     * k-nearest neighbor indexes for the test data points among the training
     * data, for kNN-based classification methods.
     * @return ClassificationEstimator containing the resulting classification
     * quality measures.
     * @throws Exception
     */
    public ClassificationEstimator test(
            float[][] predictedProbLabelsAllData,
            float[] correctPointClassificationCounter,
            ArrayList<Integer> indexes, Object dataType, int numClasses,
            float[][] trainingToTestDistances, int[][] pointNeighbors)
            throws Exception;

    /**
     * This method tests and evaluates the classifier, while keeping track of
     * correct classifications for each class.
     *
     * @param predictedProbLabelsAllData float[][] representing the current
     * predicted fuzzy labels for all data points (not only the test points in
     * the current iteration, but rather all points from the original data.)
     * @param correctPointClassificationCounter float[] representing the counter
     * of correct class-specific classifications, for latter analysis.
     * @param indexes ArrayList<Integer> of test data indexes.
     * @param dataType Object that is the test data context. It can correspond
     * to dense and sparse DataSet objects, as well as a DiscretizedDataSet data
     * context.
     * @param testLabelArray int[] representing the separate test data label
     * array.
     * @param numClasses Integer that is the number of classes in the data.
     * @param trainingToTestDistances float[][] representing a matrix of
     * distances from the training points to the test points.
     * @param pointNeighbors int[][] representing and array of arrays of
     * k-nearest neighbor indexes for the test data points among the training
     * data, for kNN-based classification methods.
     * @return ClassificationEstimator containing the resulting classification
     * quality measures.
     * @throws Exception
     */
    public ClassificationEstimator test(
            float[][] predictedProbLabelsAllData,
            float[] correctPointClassificationCounter,
            ArrayList<Integer> indexes, Object dataType,
            int[] testLabelArray, int numClasses,
            float[][] trainingToTestDistances,
            int[][] pointNeighbors) throws Exception;
}
