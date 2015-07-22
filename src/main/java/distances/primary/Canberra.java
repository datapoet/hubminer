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
 * This class implements the Canberra distance.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class Canberra extends DistanceMeasure implements Serializable {
    
    private static final long serialVersionUID = 1L;

    @Override
    public float dist(float[] arrFirst, float[] arrSecond)
            throws MetricException {
        DistanceMeasure.assertArrays(arrFirst, arrSecond);
        float sum = 0;
        for (int i = 0; i < arrFirst.length; i++) {
            if (!DataMineConstants.isAcceptableFloat(arrFirst[i])
                    || !DataMineConstants.isAcceptableFloat(arrSecond[i])) {
                continue;
            }
            if (arrFirst[i] != 0 || arrSecond[i] != 0) {
                sum += (Math.abs(arrFirst[i] - arrSecond[i])
                        / (Math.abs(arrFirst[i]) + Math.abs(arrSecond[i])));
            }
        }
        return sum;
    }

    @Override
    public float dist(int[] arrFirst, int[] arrSecond) throws MetricException {
        DistanceMeasure.assertArrays(arrFirst, arrSecond);
        float sum = 0;
        for (int i = 0; i < arrFirst.length; i++) {
            if (!DataMineConstants.isAcceptableFloat(arrFirst[i])
                    || !DataMineConstants.isAcceptableFloat(arrSecond[i])) {
                continue;
            }
            if (arrFirst[i] != 0 || arrSecond[i] != 0) {
                sum += (Math.abs(arrFirst[i] - arrSecond[i])
                        / (Math.abs(arrFirst[i]) + Math.abs(arrSecond[i])));
            }
        }
        return sum;
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
        float sum = 0;
        for (int i = 0; i < first.getNumFAtt(); i++) {
            if (!DataMineConstants.isAcceptableFloat(first.fAttr[i])
                    || !DataMineConstants.isAcceptableFloat(second.fAttr[i])) {
                continue;
            }
            if (first.fAttr[i] != 0 || second.fAttr[i] != 0) {
                sum += (Math.abs((float) (first.fAttr[i] - second.fAttr[i]))
                        / (Math.abs(first.fAttr[i]) + Math.abs(
                        second.fAttr[i])));
            }
        }
        for (int i = 0; i < first.getNumIAtt(); i++) {
            if (!DataMineConstants.isAcceptableFloat(first.iAttr[i])
                    || !DataMineConstants.isAcceptableFloat(second.iAttr[i])) {
                continue;
            }
            if (first.iAttr[i] != 0 || second.iAttr[i] != 0) {
                sum += (Math.abs((float) (first.iAttr[i] - second.iAttr[i]))
                        / (Math.abs(first.iAttr[i]) + Math.abs(
                        second.iAttr[i])));
            }
        }
        return sum;
    }
}
