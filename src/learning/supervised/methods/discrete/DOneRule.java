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
package learning.supervised.methods.discrete;

import data.representation.discrete.DiscretizedDataInstance;
import data.representation.discrete.DiscretizedDataSet;
import data.representation.util.DataMineConstants;
import feature.evaluation.DiscreteAttributeValueSplitter;
import feature.evaluation.Info;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import learning.supervised.DiscreteCategory;
import learning.supervised.DiscreteClassifier;
import learning.supervised.evaluation.ValidateableInterface;

/**
 * This class implements the trivial One-rule classifier that learns a one-level
 * decision tree.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class DOneRule extends DiscreteClassifier implements Serializable {
    
    private static final long serialVersionUID = 1L;

    private int[] majorityClassesForSplits = null;
    private float[][] classDistributionsForValues = null;
    private int attType = 0;
    private int index = 0;
    private int[] splitValues;
    private HashMap<Integer, float[]> valueCDistMap;
    private HashMap<Integer, Integer> valueClassMap;
    private float[] classPriors;
    private int majorityClassIndex;

    @Override
    public String getName() {
        return "One-Rule";
    }

    /**
     * The default constructor.
     */
    public DOneRule() {
    }

    /**
     * Initialization.
     *
     * @param discDSet DiscretizedDataSet that is the training data.
     */
    public DOneRule(DiscretizedDataSet discDSet) {
        setDataType(discDSet);
        generateClassesFromDataType();
    }

    /**
     * Initialization.
     *
     * @param discDSet DiscretizedDataSet that is the training data context.
     * @param dataClasses DiscretizedDataSet that is the training data.
     */
    public DOneRule(DiscretizedDataSet discDSet,
            DiscreteCategory[] dataClasses) {
        setDataType(discDSet);
        setClasses(dataClasses);
    }

    @Override
    public ValidateableInterface copyConfiguration() {
        return new DOneRule();
    }

    
    @Override
    public void train() throws Exception {
        DiscreteCategory[] dataClasses = getClasses();
        DiscretizedDataSet discDSet = getDataType();
        // Obtain the class priors.
        classPriors = discDSet.getClassPriors();
        // Calculate the majority class.
        majorityClassIndex = 0;
        float majorityClassProb = 0;
        for (int cIndex = 0; cIndex < classPriors.length; cIndex++) {
            if (classPriors[cIndex] > majorityClassProb) {
                majorityClassProb = classPriors[cIndex];
                majorityClassIndex = cIndex;
            }
        }
        // Find the best attribute to split on.
        Info infoCalculator = new Info(
                new DiscreteAttributeValueSplitter(discDSet),
                dataClasses.length);
        // Lowest information value corresponds to the highest information gain.
        int[] typeAndIndex = infoCalculator.
                getTypeAndIndexOfLowestEvaluatedFeature();
        attType = typeAndIndex[0];
        index = typeAndIndex[1];
        // Generate a split on the attribute.
        DiscreteAttributeValueSplitter davs =
                new DiscreteAttributeValueSplitter(discDSet);
        ArrayList<Integer>[] split = davs.generateIndexedSplitOnAttribute(
                typeAndIndex[0], typeAndIndex[1]);
        // Initialize the maps.
        valueCDistMap = new HashMap<>(1000);
        valueClassMap = new HashMap<>(1000);
        // Obtain the split values.
        splitValues = new int[split.length];
        //fill in the values for following checks for classification
        for (int i = 0; i < split.length; i++) {
            if (attType == DataMineConstants.FLOAT) {
                splitValues[i] = discDSet.data.get(split[i].get(0)).
                        floatIndexes[index];
            } else if (attType == DataMineConstants.INTEGER) {
                splitValues[i] = discDSet.data.get(split[i].get(0)).
                        integerIndexes[index];
            } else {
                splitValues[i] = discDSet.data.get(split[i].get(0)).
                        nominalIndexes[index];
            }
        }
        // Calculate the class distribution for each value.
        classDistributionsForValues = new float[split.length][
                dataClasses.length];
        majorityClassesForSplits = new int[split.length];
        float maxClassVal;
        for (int i = 0; i < split.length; i++) {
            for (int j = 0; j < split[i].size(); j++) {
                classDistributionsForValues[i][
                        discDSet.data.get(j).getCategory()]++;
            }
            maxClassVal = 0;
            for (int cIndex = 0; cIndex < dataClasses.length; cIndex++) {
                if (classDistributionsForValues[i][cIndex] > maxClassVal) {
                    maxClassVal = classDistributionsForValues[i][cIndex];
                    majorityClassesForSplits[i] = cIndex;
                }
                classDistributionsForValues[i][cIndex] /=
                        (float) split[i].size();
            }
            valueCDistMap.put(splitValues[i], classDistributionsForValues[i]);
            valueClassMap.put(splitValues[i], majorityClassesForSplits[i]);
        }
    }

    
    @Override
    public int classify(DiscretizedDataInstance instance) throws Exception {
        int targetValue;
        if (attType == DataMineConstants.FLOAT) {
            targetValue = instance.floatIndexes[index];
        } else if (attType == DataMineConstants.INTEGER) {
            targetValue = instance.integerIndexes[index];
        } else {
            targetValue = instance.nominalIndexes[index];;
        }
        if (valueClassMap.containsKey(targetValue)) {
            return valueClassMap.get(targetValue);
        } else {
            return majorityClassIndex;
        }
    }

    
    @Override
    public float[] classifyProbabilistically(DiscretizedDataInstance instance)
            throws Exception {
        int targetValue;
        if (attType == DataMineConstants.FLOAT) {
            targetValue = instance.floatIndexes[index];
        } else if (attType == DataMineConstants.INTEGER) {
            targetValue = instance.integerIndexes[index];
        } else {
            targetValue = instance.nominalIndexes[index];
        }
        if (valueCDistMap.containsKey(targetValue)) {
            return valueCDistMap.get(targetValue);
        } else {
            return classPriors;
        }
    }
}
