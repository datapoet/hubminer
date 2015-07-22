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
 * Maximal distance between some of the objects' features.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class MaxMetric extends DistanceMeasure implements Serializable {
    
    private static final long serialVersionUID = 1L;

    @Override
    public float dist(float[] arrFirst, float[] arrSecond)
            throws MetricException {
        DistanceMeasure.assertArrays(arrFirst, arrSecond);
        float max = 0;
        for (int i = 0; i < arrFirst.length; i++) {
            if (!DataMineConstants.isAcceptableFloat(arrFirst[i])
                    || !DataMineConstants.isAcceptableFloat(arrSecond[i])) {
                continue;
            }
            max = (float) Math.max(max, Math.abs(arrFirst[i] - arrSecond[i]));
        }
        return max;
    }

    @Override
    public float dist(int[] arrFirst, int[] arrSecond)
            throws MetricException {
        DistanceMeasure.assertArrays(arrFirst, arrSecond);
        float max = 0;
        for (int i = 0; i < arrFirst.length; i++) {
            if (!DataMineConstants.isAcceptableFloat(arrFirst[i])
                    || !DataMineConstants.isAcceptableFloat(arrSecond[i])) {
                continue;
            }
            max = (float) Math.max(max, Math.abs(arrFirst[i] - arrSecond[i]));
        }
        return max;
    }

    public float dist(DataInstance first, DataInstance second)
            throws MetricException {
        DistanceMeasure.assertInstances(first, second);
        float max = 0;
        if (first.hasFloatAtt() && second.hasFloatAtt()) {
            max = (float) Math.max(max, dist(first.fAttr, second.fAttr));
        }
        if (first.hasIntAtt() && second.hasIntAtt()) {
            max = (float) Math.max(max, dist(first.iAttr, second.iAttr));
        }
        return max;
    }
}