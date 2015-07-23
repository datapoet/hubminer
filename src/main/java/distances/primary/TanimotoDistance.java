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
package distances.primary;

import data.representation.DataInstance;
import data.representation.util.DataMineConstants;
import java.io.Serializable;

/**
 * This class implements Tanimoto distance, which is calculated as a number of
 * different elements in the two sets scaled by the total in the union.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 * @extends DistanceMeasure
 */
public class TanimotoDistance extends DistanceMeasure implements Serializable {
    
    private static final long serialVersionUID = 1L;

    @Override
    public float dist(float[] arrFirst, float[] arrSecond)
            throws MetricException {
        DistanceMeasure.assertArrays(arrFirst, arrSecond);
        float scalarProduct = 0;
        float normFirst = 0;
        float normSecond = 0;
        for (int i = 0; i < arrFirst.length; i++) {
            normFirst += arrFirst[i] * arrFirst[i];
            normSecond += arrSecond[i] * arrSecond[i];
            scalarProduct += arrFirst[i] * arrSecond[i];
        }
        return (scalarProduct / (normFirst + normSecond - scalarProduct));
    }

    @Override
    public float dist(int[] arrFirst, int[] arrSecond) throws MetricException {
        DistanceMeasure.assertArrays(arrFirst, arrSecond);
        int countFirst = 0;
        int countSecond = 0;
        int countUnion = 0;
        for (int i = 0; i < arrFirst.length; i++) {
            if (DataMineConstants.isAcceptableInt(arrFirst[i])) {
                countFirst += arrFirst[i];
            }
            if (DataMineConstants.isAcceptableInt(arrSecond[i])) {
                countSecond += arrSecond[i];
            }
            if (DataMineConstants.isAcceptableInt(arrFirst[i])
                    && DataMineConstants.isAcceptableInt(arrSecond[i])) {
                countUnion += Math.min(arrFirst[i], arrSecond[i]);
            }
        }
        float tanimotoDistance = 0;
        int denominator = countFirst + countSecond - countUnion;
        if (denominator != 0) {
            tanimotoDistance = (denominator - countUnion) / denominator;
        }
        return tanimotoDistance;
    }

    /**
     * @param first Data instance.
     * @param second Data instance.
     * @return Distance.
     * @throws MetricException
     */
    public float dist(DataInstance first, DataInstance second)
            throws MetricException {
        DistanceMeasure.assertInstances(first, second);
        return dist(first.iAttr, second.iAttr);
    }
}