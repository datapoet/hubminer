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

import data.representation.discrete.DiscretizedDataInstance;
import data.representation.discrete.DiscretizedDataSet;
import java.util.ArrayList;
import util.BasicMathUtil;

/**
 * This class implements the calculations of the information gain on feature
 * value splits in the data.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class InfoGain extends SplitAssesment {

    private int numClasses = 2;
    // The entropy of the entire data prior to the split.
    private float priorEntropy = -1;
    private ArrayList<Integer> currSubset = null;
    // The entropy of the current subset prior to the split.
    private float currSubsetEntropy = -1;

    /**
     *
     * @param discDSet Discretized data set.
     * @param numClasses Integer that is the number of classes.
     */
    public InfoGain(DiscretizedDataSet discDSet, int numClasses) {
        super(discDSet);
        this.numClasses = numClasses;
    }

    /**
     * @param priorEntropy Float value that is the prior entropy of the whole
     * data.
     */
    public void setWholeEntropy(float priorEntropy) {
        this.priorEntropy = priorEntropy;
    }

    /**
     * @param currSubsetEntropy Float value that is the entropy of the data
     * subset that is currently being analyzed.
     */
    public void setCurrSubsetEntropy(float currSubsetEntropy) {
        this.currSubsetEntropy = currSubsetEntropy;
    }

    @Override
    public float assesSplitOnWhole(ArrayList<Integer>[] split) {
        if (split == null || split.length == 0) {
            return 0;
        }
        DiscretizedDataSet discDSet = getDataContext();
        Info infoChecker = new Info(
                new DiscreteAttributeValueSplitter(discDSet), numClasses);
        float infoValue = infoChecker.evaluateSplit(split);
        // Calculate prior entropy if it wasn't provided.
        if (priorEntropy == -1) {
            DiscretizedDataInstance instance;
            float[] classFrequencies = new float[numClasses];
            float denom = 0;
            for (int i = 0; i < discDSet.size(); i++) {
                instance = discDSet.data.get(i);
                if (instance.getCategory() >= 0) {
                    classFrequencies[instance.getCategory()]++;
                    denom++;
                }
            }
            priorEntropy = 0;
            float factor;
            for (int i = 0; i < numClasses; i++) {
                factor = classFrequencies[i] / denom;
                if (classFrequencies[i] > 0) {
                    priorEntropy -= factor * BasicMathUtil.log2(factor);
                }
            }
        }
        // The information gain is the difference in entropies.
        return priorEntropy - infoValue;
    }

    @Override
    public float assesSplitOnSubset(ArrayList<Integer> subset,
            ArrayList<Integer>[] split) {
        if (split == null || split.length == 0 || subset == null) {
            return 0;
        }
        DiscretizedDataSet discDSet = getDataContext();
        Info infoChecker = new Info(
                new DiscreteAttributeValueSplitter(discDSet), numClasses);
        float infoValue = infoChecker.evaluateSplitOnSubset(subset, split);
        // Calculate the prior subset entropy if it wasn't provided.
        if (currSubsetEntropy == -1 || currSubset != subset) {
            DiscretizedDataInstance instance;
            float[] classFrequencies = new float[numClasses];
            float denom = 0;
            for (int i = 0; i < subset.size(); i++) {
                instance = discDSet.data.get(subset.get(i));
                if (instance.getCategory() >= 0) {
                    classFrequencies[instance.getCategory()]++;
                    denom++;
                }
            }
            currSubsetEntropy = 0;
            float factor;
            for (int i = 0; i < numClasses; i++) {
                factor = classFrequencies[i] / denom;
                if (classFrequencies[i] > 0) {
                    currSubsetEntropy -= factor * BasicMathUtil.log2(factor);
                }
            }
            currSubset = subset;
        }
        return currSubsetEntropy - infoValue;
    }
}
