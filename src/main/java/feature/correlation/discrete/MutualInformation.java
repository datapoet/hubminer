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
package feature.correlation.discrete;

import data.representation.discrete.DiscretizedDataSet;
import data.representation.util.DataMineConstants;
import util.BasicMathUtil;

/**
 * This class implements the calculation of mutual information between two
 * discrete attributes.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class MutualInformation extends DiscreteCorrelationCoefficient {

    /**
     *
     * @param discDSet DiscretizedDataSet data context.
     * @param firstType Type of the first feature.
     * @param firstIndex Index of the first feature.
     * @param secondType Type of the second feature.
     * @param secondIndex Index of the second feature.
     */
    public MutualInformation(DiscretizedDataSet discDSet, int firstType,
            int firstIndex, int secondType, int secondIndex) {
        super(discDSet, firstType, firstIndex, secondType, secondIndex);
    }

    /**
     * Calculate a value distribution of the value array.
     *
     * @param values Integer value array.
     * @param type Feature type.
     * @param index Feature index.
     * @return Float array that is the feature value distribution.
     */
    private float[] createDistributionForAttribute(int[] values, int type,
            int index) {
        if (values == null) {
            return null;
        }
        float[] valueDistribution;
        float counter = 0;
        if (type == DataMineConstants.FLOAT) {
            // We skip the last entry in the interval divisions, because the
            // last entry in that array is dummy holding Float.MAX_VALUE.
            valueDistribution = new float[discDSet.getFloatIntervalDivisions()[
                    index].length - 1];
            for (int i = 0; i < values.length; i++) {
                if (values[i] < valueDistribution.length) {
                    valueDistribution[values[i]]++;
                    counter++;
                }
            }
            if (counter > 0) {
                for (int i = 0; i < values.length; i++) {
                    valueDistribution[values[i]] /= counter;
                }
            }
            return valueDistribution;
        } else if (type == DataMineConstants.INTEGER) {
            // We skip the last entry in the interval divisions, because the
            // last entry in that array is dummy holding Float.MAX_VALUE.
            valueDistribution = new float[discDSet.getIntIntervalDivisions()[
                    index].length - 1];
            for (int i = 0; i < values.length; i++) {
                if (values[i] < valueDistribution.length) {
                    valueDistribution[values[i]]++;
                    counter++;
                }
            }
            if (counter > 0) {
                for (int i = 0; i < values.length; i++) {
                    valueDistribution[values[i]] /= counter;
                }
            }
            return valueDistribution;
        } else {
            valueDistribution = new float[discDSet.getNominalVocabularies()[
                    index].size()];
            for (int i = 0; i < values.length; i++) {
                valueDistribution[values[i]]++;
                counter++;
            }
            if (counter > 0) {
                for (int i = 0; i < values.length; i++) {
                    valueDistribution[values[i]] /= counter;
                }
            }
            return valueDistribution;
        }
    }

    private float[][] createMutualDistribution(int[] first, int[] second) {
        if (first == null || second == null || first.length != second.length) {
            return null;
        }
        int lenFirst, lenSecond;
        if (firstType == DataMineConstants.FLOAT) {
            lenFirst = discDSet.getFloatIntervalDivisions()[firstIndex].length
                    - 1;
        } else if (firstType == DataMineConstants.INTEGER) {
            lenFirst = discDSet.getIntIntervalDivisions()[firstIndex].length
                    - 1;
        } else {
            lenFirst = discDSet.getNominalVocabularies()[firstIndex].size();
        }
        if (secondType == DataMineConstants.FLOAT) {
            lenSecond = discDSet.getFloatIntervalDivisions()[secondIndex].length
                    - 1;
        } else if (secondType == DataMineConstants.INTEGER) {
            lenSecond = discDSet.getIntIntervalDivisions()[secondIndex].length
                    - 1;
        } else {
            lenSecond = discDSet.getNominalVocabularies()[secondIndex].size();
        }
        // This could also be implemented via a HashMap in those cases when
        // the number of features is too high for building full matrices.
        float[][] mutualDistr = new float[lenFirst][lenSecond];
        float count = 0;
        for (int i = 0; i < first.length; i++) {
            if (first[i] < lenFirst && second[i] < lenSecond) {
                mutualDistr[first[i]][second[i]]++;
                count++;
            }
        }
        // Normalize.
        if (count > 0) {
            for (int i = 0; i < lenFirst; i++) {
                for (int j = 0; j < lenSecond; j++) {
                    mutualDistr[i][j] /= count;
                }
            }
        }
        return mutualDistr;
    }

    @Override
    public float correlation(int[] first, int[] second) throws Exception {
        if (first == null || second == null || first.length != second.length
                || first.length <= 1) {
            return 0;
        } else {
            // Create feature value distributions.
            float[] firstDistr = createDistributionForAttribute(first,
                    firstType, firstIndex);
            float[] secondDistr = createDistributionForAttribute(second,
                    secondType, secondIndex);
            float[][] mutualDistr = createMutualDistribution(first, second);
            int lenFirst, lenSecond;
            lenFirst = firstDistr.length;
            lenSecond = secondDistr.length;
            float corr = 0;
            for (int i = 0; i < lenFirst; i++) {
                for (int j = 0; j < lenSecond; j++) {
                    if (mutualDistr[i][j] > 0) {
                        // The individual distribution values must also be
                        // positive here, so no need to check explicitly.
                        corr += mutualDistr[i][j]
                                * BasicMathUtil.log2(mutualDistr[i][j]
                                / (firstDistr[i] * secondDistr[j]));
                    }
                }
            }
            return corr;
        }
    }
}
