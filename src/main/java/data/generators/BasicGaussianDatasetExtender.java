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
package data.generators;

import data.representation.DataInstance;
import data.representation.DataSet;
import data.representation.util.DataMineConstants;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import util.ArrayUtil;

/**
 * This class finds the mean and standard deviation for each float feature in
 * each class and makes a standardized Gaussian generator which is used to
 * generate synthetic DataInstance objects. In this particular implementation, a
 * separate one-dimensional generator is built for each feature. For more
 * general Gaussian models, one should calculate the covariance matrix.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class BasicGaussianDatasetExtender {

    // DataSet object that is to be extended by synthetic DataInstances from a
    // Gaussian model.
    private DataSet dset;
    // Means and standard deviations for float features for each class.
    private double[][] classAttMeans;
    private double[][] classAttStDevs;
    private int numClasses = 1;
    private Random randa;

    public BasicGaussianDatasetExtender(DataSet dset) {
        this.dset = dset;
        if (dset != null) {
            numClasses = dset.countCategories();
        } else {
            numClasses = 0;
        }
        randa = new Random();
    }

    /**
     * This method learns a Gaussian model from the data. This model is later
     * used for generating new synthetic DataInstance objects.
     */
    public void generateGaussianModel() {
        if (dset == null) {
            return;
        }
        // Initialization.
        classAttMeans = new double[numClasses][dset.getNumFloatAttr()];
        classAttStDevs = new double[numClasses][dset.getNumFloatAttr()];
        // To properly handle missing values.
        int[][] classFreqsForValidFeatures =
                new int[dset.getNumFloatAttr()][numClasses];
        for (int attIndex = 0; attIndex < dset.getNumFloatAttr();
                attIndex++) {
            Arrays.fill(classFreqsForValidFeatures[attIndex], 0);
        }
        int currClass;
        // Calculate the mean values for each feature within each class.
        for (int i = 0; i < dset.size(); i++) {
            DataInstance instance = dset.data.get(i);
            currClass = instance.getCategory();
            for (int attIndex = 0; attIndex < dset.getNumFloatAttr();
                    attIndex++) {
                if (DataMineConstants.isAcceptableFloat(
                        instance.fAttr[attIndex])) {
                    classAttMeans[currClass][attIndex] +=
                            instance.fAttr[attIndex];
                    classFreqsForValidFeatures[attIndex][currClass]++;
                }
            }
        }
        for (int classIndex = 0; classIndex < numClasses; classIndex++) {
            for (int attIndex = 0; attIndex < dset.getNumFloatAttr();
                    attIndex++) {
                if (classFreqsForValidFeatures[attIndex][classIndex] > 0) {
                    classAttMeans[classIndex][attIndex] /=
                            (double) classFreqsForValidFeatures[attIndex][
                            classIndex];
                }
            }
        }
        // Calculate the standard deviation for each feature within each class.
        for (int i = 0; i < dset.size(); i++) {
            DataInstance instance = dset.data.get(i);
            currClass = instance.getCategory();
            for (int attIndex = 0; attIndex < dset.getNumFloatAttr();
                    attIndex++) {
                if (DataMineConstants.isAcceptableFloat(
                        instance.fAttr[attIndex])) {
                    classAttStDevs[currClass][attIndex] +=
                            (instance.fAttr[attIndex]
                            - classAttMeans[currClass][attIndex])
                            * (instance.fAttr[attIndex]
                            - classAttMeans[currClass][attIndex]);
                }
            }
        }
        for (int classIndex = 0; classIndex < numClasses; classIndex++) {
            for (int attIndex = 0; attIndex < dset.getNumFloatAttr();
                    attIndex++) {
                if (classFreqsForValidFeatures[attIndex][classIndex] > 0) {
                    classAttStDevs[classIndex][attIndex] /=
                            (double) classFreqsForValidFeatures[attIndex][
                            classIndex];
                    classAttStDevs[classIndex][attIndex] =
                            (float) Math.sqrt(classAttStDevs[classIndex][
                            attIndex]);
                }
            }
        }
    }

    /**
     * Generates a new instance according to the model.
     *
     * @param classIndex Index of the class to generate the instance for.
     * @return A new synthetic instance of the specified class, according to the
     * underlying model.
     */
    public DataInstance generateInstanceForClass(int classIndex) {
        DataInstance instance = new DataInstance(dset);
        float genF;
        for (int i = 0; i < instance.fAttr.length; i++) {
            genF = (float) randa.nextGaussian();
            instance.fAttr[i] = (float) classAttMeans[classIndex][i]
                    + (genF * (float) classAttStDevs[classIndex][i]);
        }
        instance.setCategory(classIndex);
        return instance;
    }

    /**
     * Generate an ArrayList of instances of the specified class according to
     * the underlying model.
     *
     * @param numInstances Number of instances to generate.
     * @param classIndex Index of the class to generate the instances for.
     * @return ArrayList<DataInstance> of generated instances.
     */
    public ArrayList<DataInstance> generateSyntheticInstances(
            int numInstances, int classIndex) {
        ArrayList result = new ArrayList(numInstances);
        for (int i = 0; i < numInstances; i++) {
            result.add(generateInstanceForClass(classIndex));
        }
        return result;
    }

    /**
     * Generates an ArrayList of instances according to the underlying model,
     * while keeping the proportion of class frequencies.
     *
     * @return ArrayList<DataInstance> of generated instances.
     */
    public ArrayList<DataInstance> generateSyntheticInstances(
            int numInstances) {
        ArrayList result = new ArrayList(numInstances);
        float[] classPriors = dset.getClassPriors();
        float choice;
        float[] cumulativeProbs = new float[classPriors.length + 1];
        cumulativeProbs[0] = 0;
        for (int i = 1; i < cumulativeProbs.length; i++) {
            cumulativeProbs[i] = cumulativeProbs[i - 1] + classPriors[i - 1];
        }
        for (int i = 0; i < numInstances; i++) {
            choice = randa.nextFloat();
            int classIndex = ArrayUtil.findIndex(choice, cumulativeProbs);
            classIndex = Math.min(classIndex, classPriors.length - 1);
            result.add(generateInstanceForClass(classIndex));
        }
        return result;
    }
}
