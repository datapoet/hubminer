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

import data.representation.discrete.DiscretizedDataInstance;
import data.representation.discrete.DiscretizedDataSet;
import data.representation.util.DataMineConstants;

/**
 * This class implements the correlation coefficient for discretized data
 * instances.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class DiscreteCorrelationCoefficient {

    public DiscretizedDataSet discDSet;
    public int firstType, secondType, firstIndex, secondIndex;

    /**
     *
     * @param discDSet DiscretizedDataSet data context.
     * @param firstType Type of the first feature.
     * @param firstIndex Index of the first feature.
     * @param secondType Type of the second feature.
     * @param secondIndex Index of the second feature.
     */
    public DiscreteCorrelationCoefficient(DiscretizedDataSet discDSet,
            int firstType, int firstIndex, int secondType, int secondIndex) {
        this.discDSet = discDSet;
        this.firstType = firstType;
        this.secondType = secondType;
        this.firstIndex = firstIndex;
        this.secondIndex = secondIndex;
    }

    /**
     * Calculates the correlation between two value arrays.
     *
     * @param first Integer value array.
     * @param second Integer value array.
     * @return Correlation between the two value arrays.
     * @throws Exception
     */
    public float correlation(int[] first, int[] second) throws Exception {
        // Meant to be overriden in inheriting classes.
        return 0;
    }

    /**
     * Get a value array for a given feature type and index, from the specified
     * data context.
     *
     * @param discDSet DiscretizedDataSet data context.
     * @param type Feature type.
     * @param index Feature index.
     * @return Value array of the given discrete feature within the data
     * context.
     * @throws Exception
     */
    private static int[] fillArray(DiscretizedDataSet discDSet, int type,
            int index) throws Exception {
        if (type == DataMineConstants.FLOAT) {
            if (index >= discDSet.getNumFloatAttr()) {
                throw new Exception("Index out of range: " + index);
            } else {
                int[] a = new int[discDSet.size()];
                DiscretizedDataInstance instance;
                for (int i = 0; i < discDSet.size(); i++) {
                    instance = discDSet.data.get(i);
                    a[i] = instance.floatIndexes[index];
                }
                return a;
            }
        } else if (type == DataMineConstants.INTEGER) {
            if (index >= discDSet.getNumIntAttr()) {
                throw new Exception("Index out of range: " + index);
            } else {
                int[] a = new int[discDSet.size()];
                DiscretizedDataInstance instance;
                for (int i = 0; i < discDSet.size(); i++) {
                    instance = discDSet.data.get(i);
                    a[i] = instance.integerIndexes[index];
                }
                return a;
            }
        } else {
            if (index >= discDSet.getNumNominalAttr()) {
                throw new Exception("Index out of range: " + index);
            } else {
                int[] a = new int[discDSet.size()];
                DiscretizedDataInstance instance;
                for (int i = 0; i < discDSet.size(); i++) {
                    instance = discDSet.data.get(i);
                    a[i] = instance.nominalIndexes[index];
                }
                return a;
            }
        }
    }

    /**
     * Calculates the correlation between two specified discretized features
     * within the specified data context.
     *
     * @return Float value that is the correlation between the specified
     * discretized features within the data context.
     * @throws Exception
     */
    public float getCorrelation() throws Exception {
        if (discDSet == null || discDSet.isEmpty()) {
            return 0;
        }
        int[] first = fillArray(discDSet, firstType, firstIndex);
        int[] second = fillArray(discDSet, secondType, secondIndex);
        return correlation(first, second);
    }
}
