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
 * Meant for measuring distances between probability distributions. This
 * implementation does not assume that the histograms are normalized, so it
 * includes divisions by the feature sums.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class BhattacharyyaDistance extends DistanceMeasure
implements Serializable {

    private static final long serialVersionUID = 1L;
    /**
     * @param arrFirst Feature array.
     * @param arrSecond Feature array.
     * @return Distance.
     * @throws MetricException
     */
    @Override
    public float dist(float[] arrFirst, float[] arrSecond)
            throws MetricException {
        DistanceMeasure.assertArrays(arrFirst, arrSecond);
        float sumFirst = 0f;
        float sumSecond = 0f;
        for (int i = 0; i < arrFirst.length; i++) {
            if (!DataMineConstants.isAcceptableFloat(arrFirst[i])
                    || !DataMineConstants.isAcceptableFloat(arrSecond[i])) {
                continue;
            }
            sumFirst += Math.abs(arrFirst[i]);
            sumSecond += Math.abs(arrSecond[i]);
        }
        float sumBC = 0;
        sumFirst = Math.max(sumFirst, DataMineConstants.EPSILON);
        sumSecond = Math.max(sumSecond, DataMineConstants.EPSILON);
        for (int i = 0; i < arrFirst.length; i++) {
            if (!DataMineConstants.isAcceptableFloat(arrFirst[i])
                    || !DataMineConstants.isAcceptableFloat(arrSecond[i])) {
                continue;
            }
            sumBC += Math.sqrt(
                    (arrFirst[i] / sumFirst) * (arrSecond[i] / sumSecond));
        }
        return (float) (-Math.log(sumBC));
    }

    /**
     * @param arrFirst Feature array.
     * @param arrSecond Feature array.
     * @return Distance.
     * @throws MetricException
     */
    @Override
    public float dist(int[] arrFirst, int[] arrSecond) throws MetricException {
        DistanceMeasure.assertArrays(arrFirst, arrSecond);
        float sumFirst = 0f;
        float sumSecond = 0f;
        for (int i = 0; i < arrFirst.length; i++) {
            if (!DataMineConstants.isAcceptableFloat(arrFirst[i])
                    || !DataMineConstants.isAcceptableFloat(arrSecond[i])) {
                continue;
            }
            sumFirst += Math.abs(arrFirst[i]);
            sumSecond += Math.abs(arrSecond[i]);
        }
        float sumBC = 0;
        sumFirst = Math.max(sumFirst, DataMineConstants.EPSILON);
        sumSecond = Math.max(sumSecond, DataMineConstants.EPSILON);
        for (int i = 0; i < arrFirst.length; i++) {
            if (!DataMineConstants.isAcceptableFloat(arrFirst[i])
                    || !DataMineConstants.isAcceptableFloat(arrSecond[i])) {
                continue;
            }
            sumBC += Math.sqrt((arrFirst[i] / sumFirst) *
                    (arrSecond[i] / sumSecond));
        }
        return (float) (-Math.log(sumBC));
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
        float sum = 0f;
        if (first.getNumFAtt() == second.getNumFAtt()) {
            sum += this.dist(first.fAttr, second.fAttr);
        }
        if (first.getNumIAtt() == second.getNumIAtt()) {
            sum += this.dist(first.iAttr, second.iAttr);
        }
        return sum;
    }
}
