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
package distances.sparse;

import data.representation.util.DataMineConstants;
import java.util.HashMap;
import java.util.Set;

/**
 * This class implements Minkowski distances for sparse representations
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class SparseMinkowskiMetric extends SparseMetric {

    // Exponent.
    private int M = 2;

    public SparseMinkowskiMetric() {
    }

    /**
     * @param M Integer exponent. M=2 is for Euclidean.
     */
    SparseMinkowskiMetric(int M) {
        this.M = 2;
    }

    @Override
    public float dist(HashMap<Integer, Float> firstMap,
            HashMap<Integer, Float> secondMap)
            throws Exception {
        if ((firstMap == null || firstMap.isEmpty())
                && (secondMap == null || secondMap.isEmpty())) {
            return 0;
        } else if ((firstMap == null || firstMap.isEmpty())
                && (secondMap != null && !secondMap.isEmpty())) {
            return 1;
        } else if ((secondMap == null || secondMap.isEmpty())
                && (firstMap != null && !firstMap.isEmpty())) {
            return 1;
        } else {
            float result = 0;
            // Iterating over one of the maps and looking into the other is
            // enough to find all the common keys.
            Set<Integer> keys = firstMap.keySet();
            for (int index : keys) {
                if (secondMap.containsKey(index)) {
                    if (DataMineConstants.isAcceptableFloat(firstMap.get(index))
                            && DataMineConstants.isAcceptableFloat(
                            secondMap.get(index))) {
                        result += Math.pow(Math.abs(
                                firstMap.get(index) - secondMap.get(index)), M);
                    }
                } else {
                    if (DataMineConstants.isAcceptableFloat(
                            firstMap.get(index))) {
                        result += Math.pow(Math.abs(firstMap.get(index)), M);
                    }
                }
            }
            keys = secondMap.keySet();
            for (int index : keys) {
                if (!firstMap.containsKey(index)
                        && DataMineConstants.isAcceptableFloat(
                        secondMap.get(index))) {
                    result += Math.pow(Math.abs(secondMap.get(index)), M);
                }
            }
            result = (float) Math.pow(result, 1f / M);
            return result;
        }
    }
}
