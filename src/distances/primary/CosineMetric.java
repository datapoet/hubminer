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

import data.representation.util.DataMineConstants;
import java.io.Serializable;

/**
 * This class implements the standard cosine distance.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 * @extends DistanceMeasure
 */
public class CosineMetric extends DistanceMeasure implements Serializable {
    
    private static final long serialVersionUID = 1L;

    /**
     * Generates a new CosineMetric instance.
     *
     * @return CosineMetric instance.
     */
    public CosineMetric newInstance() {
        return new CosineMetric();
    }

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
            sum += arrFirst[i] * arrSecond[i];
        }
        MinkowskiMetric mm = new MinkowskiMetric(2);
        float normFirst = mm.norm(arrFirst);
        float normSecond = mm.norm(arrSecond);
        if (DataMineConstants.isNonZero(normFirst)
                && DataMineConstants.isNonZero(normSecond)) {
            sum = sum / (normFirst * normSecond);
        } else if (DataMineConstants.isZero(normFirst)
                && DataMineConstants.isZero(normSecond)) {
            sum = 1;
        } else {
            sum = -1;
        }
        return (1f - sum) * 0.5f;
    }

    @Override
    public float dist(int[] arrFirst, int[] arrSecond) throws MetricException {
        DistanceMeasure.assertArrays(arrFirst, arrSecond);
        float sum = 0;
        for (int i = 0; i < arrFirst.length; i++) {
            if (!DataMineConstants.isAcceptableInt(arrFirst[i])
                    || !DataMineConstants.isAcceptableInt(arrSecond[i])) {
                continue;
            }
            sum += arrFirst[i] * arrSecond[i];
        }
        MinkowskiMetric mm = new MinkowskiMetric(2);
        float normFirst = mm.norm(arrFirst);
        float normSecond = mm.norm(arrSecond);
        if (DataMineConstants.isNonZero(normFirst)
                && DataMineConstants.isNonZero(normSecond)) {
            sum = sum / (normFirst * normSecond);
        } else if (DataMineConstants.isZero(normFirst)
                && DataMineConstants.isZero(normSecond)) {
            sum = 1;
        } else {
            sum = -1;
        }
        return (1f - sum) * 0.5f;
    }
}