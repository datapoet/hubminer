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

import data.representation.discrete.DiscretizedDataSet;
import java.util.ArrayList;
import util.BasicMathUtil;

/**
 * This class implements the gain ratio feature evaluation measure.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class GainRatio extends SplitAssesment {

    private InfoGain infoGainCalculator = null;

    /**
     * @param infoGainCalculator InformationGainCalculator object that
     * calculates the information gain evaluation measure.
     */
    public void setInfoGainCalculator(InfoGain infoGainCalculator) {
        this.infoGainCalculator = infoGainCalculator;
    }

    /**
     *
     * @param discDSet Discretized dataset.
     * @param numClasses Integer that is the number of classes in the data.
     */
    public GainRatio(DiscretizedDataSet discDSet, int numClasses) {
        super(discDSet);
        infoGainCalculator = new InfoGain(discDSet, numClasses);
    }

    /**
     * Calculates the entropy of the raw split, by looking at the split group
     * sizes.
     *
     * @param split A data split given as an array of lists of indexes.
     * @return Float value that is the raw entropy of the split.
     */
    public static float calculateRawSplitEntropy(ArrayList<Integer>[] split) {
        if (split == null || split.length == 0) {
            return 0;
        }
        float splitEntropy = 0;
        float totalSize = 0;
        for (int i = 0; i < split.length; i++) {
            if (split[i] != null) {
                totalSize += split[i].size();
            }
        }
        float factor;
        for (int i = 0; i < split.length; i++) {
            if (split[i] != null && split[i].size() > 0) {
                factor = (float) split[i].size() / totalSize;
                splitEntropy -= factor * BasicMathUtil.log2(factor);
            }
        }
        return splitEntropy;
    }

    @Override
    public float assesSplitOnWhole(ArrayList<Integer>[] split) {
        float infoGainValue = infoGainCalculator.assesSplitOnWhole(split);
        // Calculate the raw entropy and normalize.
        float splitEntropy = calculateRawSplitEntropy(split);
        return infoGainValue / splitEntropy;
    }

    @Override
    public float assesSplitOnSubset(ArrayList<Integer> subset,
            ArrayList<Integer>[] split) {
        float infoGainValue = infoGainCalculator.assesSplitOnSubset(
                subset, split);
        // Calculate the raw entropy and normalize.
        float splitEntropy = calculateRawSplitEntropy(split);
        return infoGainValue / splitEntropy;
    }
}
