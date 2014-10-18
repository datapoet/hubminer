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
package feature.evaluation;

import data.representation.DataInstance;
import data.representation.discrete.DiscretizedDataSet;
import java.util.ArrayList;
import util.BasicMathUtil;

/**
 * This class implements methods for calculating the information content of
 * value splits on different features in the data.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class Info extends DiscreteAttributeEvaluator {

    private DiscreteAttributeValueSplitter valueSplitter = null;
    // Defaults to the binary case, can be specified in the constructor.
    private int numClasses = 2;

    /**
     * @param valueSplitter DiscreteAttributeValueSplitter object for value
     * splitting.
     * @param numClasses Number of classes in the data.
     */
    public Info(DiscreteAttributeValueSplitter valueSplitter,
            int numClasses) {
        this.valueSplitter = valueSplitter;
        setDiscretizedDataSet(valueSplitter.getDataContext());
        this.numClasses = numClasses;
    }

    /**
     *
     * @param valueSplitter creteAttributeValueSplitter object for value
     * splitting.
     */
    public void setAttValueSplitter(
            DiscreteAttributeValueSplitter valueSplitter) {
        this.valueSplitter = valueSplitter;
        setDiscretizedDataSet(valueSplitter.getDataContext());
    }

    /**
     * @return creteAttributeValueSplitter object for value splitting.
     */
    public DiscreteAttributeValueSplitter getAttValueSplitter() {
        return valueSplitter;
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

    @Override
    public float evaluate(int attType, int index) {
        ArrayList<Integer>[] split = valueSplitter.
                generateIndexedSplitOnAttribute(attType, index);
        return evaluateSplit(split);
    }

    @Override
    public float evaluateOnSubset(ArrayList<Integer> subset, int attType,
            int index) {
        ArrayList<Integer>[] split = valueSplitter.
                generateIndexedSplitFromGivenDiscretization(subset, attType,
                index);
        return evaluateSplitOnSubset(subset, split);
    }

    /**
     * Same as other methods, but with DataInstance objects listed in the split
     * instead of the indexes.
     *
     * @param split A split of the data given as an array of lists of instances.
     * @return Float value that is the information content of the split.
     */
    public float evaluateSplitofDataInstances(ArrayList<DataInstance>[] split) {
        if (split == null || split.length == 0) {
            return 0;
        }
        int[] tempClassFrequencies = new int[numClasses];
        float info = 0;
        float totalSize = getDiscretizedDataSet().size();
        float currSize;
        float currEntropy;
        float factor;
        for (int i = 0; i < split.length; i++) {
            if (split[i] == null) {
                continue;
            }
            for (int j = 0; j < numClasses; j++) {
                tempClassFrequencies[j] = 0;
            }
            currSize = 0;
            currEntropy = 0;
            for (int j = 0; j < split[i].size(); j++) {
                // Noise is sometimes marked by -1 labels.
                if (split[i].get(j).getCategory() >= 0) {
                    tempClassFrequencies[split[i].get(j).getCategory()]++;
                    currSize++;
                }
            }
            currSize = Math.max(currSize, 1);
            for (int j = 0; j < numClasses; j++) {
                factor = (float) tempClassFrequencies[j] / (float) currSize;
                if (factor > 0) {
                    currEntropy -= factor * BasicMathUtil.log2(factor);
                }
            }
            info += ((float) split[i].size() / (float) totalSize) * currEntropy;
        }
        return info;
    }

    @Override
    public float evaluateSplit(ArrayList<Integer>[] split) {
        if (split == null || split.length == 0) {
            return 0;
        }
        int[] tempClassFrequencies = new int[numClasses];
        float info = 0;
        DiscretizedDataSet discDSet = valueSplitter.getDataContext();
        setDiscretizedDataSet(discDSet);
        float totalSize = discDSet.size();
        float currSize;
        float currEntropy;
        float factor;
        for (int i = 0; i < split.length; i++) {
            for (int j = 0; j < numClasses; j++) {
                tempClassFrequencies[j] = 0;
            }
            currSize = 0;
            currEntropy = 0;
            for (int j = 0; j < split[i].size(); j++) {
                if (discDSet.data.get(split[i].get(j)).getCategory() >= 0) {
                    tempClassFrequencies[discDSet.data.get(split[i].get(j)).
                            getCategory()]++;
                    currSize++;
                }
            }
            currSize = Math.max(currSize, 1);
            for (int j = 0; j < numClasses; j++) {
                factor = (float) tempClassFrequencies[j] / (float) currSize;
                if (factor > 0) {
                    currEntropy -= factor * BasicMathUtil.log2(factor);
                }
            }
            info += ((float) split[i].size() / (float) totalSize) * currEntropy;
        }
        return info;
    }

    @Override
    public float evaluateSplitOnSubset(ArrayList<Integer> subset,
            ArrayList<Integer>[] split) {
        if (split == null || split.length == 0 || subset == null) {
            return 0;
        }
        int[] tempClassFrequencies = new int[numClasses];
        float info = 0;
        float totalSize = subset.size();
        DiscretizedDataSet discDSet = valueSplitter.getDataContext();
        setDiscretizedDataSet(discDSet);
        float currSize;
        float currEntropy;
        float factor;
        for (int i = 0; i < split.length; i++) {
            if (split[i] != null && split[i].size() > 0) {
                for (int j = 0; j < numClasses; j++) {
                    tempClassFrequencies[j] = 0;
                }
                currSize = 0;
                currEntropy = 0;
                for (int j = 0; j < split[i].size(); j++) {
                    if (discDSet.data.get(split[i].get(j)).getCategory() >= 0) {
                        tempClassFrequencies[discDSet.data.get(split[i].get(j)).
                                getCategory()]++;
                        currSize++;
                    }
                }
                currSize = Math.max(currSize, 1);
                for (int j = 0; j < numClasses; j++) {
                    factor = (float) tempClassFrequencies[j] / (float) currSize;
                    if (factor > 0) {
                        currEntropy -= factor * BasicMathUtil.log2(factor);
                    }
                }
                info += ((float) split[i].size() / (float) totalSize)
                        * currEntropy;
            }
        }
        return info;
    }

    /**
     * This method handles the case when the category values are given directly
     * instead of indexes to DataInstance objects.
     *
     * @param split An array of integer valued lists that denote the categories
     * of the instances in the split.
     * @param numClasses Integer that is the number of classes in the data.
     * @return Float value that is the information content of the split.
     */
    public static float evaluateInfoOfCategorySplit(ArrayList<Integer>[] split,
            int numClasses) {
        if (split == null || split.length == 0) {
            return 0;
        }
        int[] tempClassFrequencies = new int[numClasses];
        float info = 0;
        float totalSize = 0;
        float[] sizes = new float[split.length];
        for (int i = 0; i < split.length; i++) {
            if (split[i] != null && split[i].size() > 0) {
                for (int j = 0; j < split[i].size(); j++) {
                    if (split[i].get(j) >= 0) {
                        totalSize++;
                        sizes[i]++;
                    }
                }
            }
        }
        float currEntropy;
        float factor;
        for (int i = 0; i < split.length; i++) {
            if (split[i] != null && split[i].size() > 0) {
                for (int j = 0; j < numClasses; j++) {
                    tempClassFrequencies[j] = 0;
                }
                currEntropy = 0;
                int category;
                for (int j = 0; j < split[i].size(); j++) {
                    category = split[i].get(j);
                    // Only the non-noisy labels.
                    if (category >= 0) {
                        tempClassFrequencies[category]++;
                    }
                }
                for (int j = 0; j < numClasses; j++) {
                    factor = (float) tempClassFrequencies[j] / sizes[i];
                    if (factor > 0) {
                        currEntropy -= factor * BasicMathUtil.log2(factor);
                    }
                }
                info += (sizes[i] / totalSize) * currEntropy;
            }
        }
        return info;
    }
}
