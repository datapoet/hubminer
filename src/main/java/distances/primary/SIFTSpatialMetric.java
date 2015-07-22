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
import java.io.Serializable;

/**
 * This class expects SIFTVector-s or their float arrays. It assumes the first
 * two values are (x,y) coordinates in an image and calculates the squared (x,y)
 * Euclidean distance.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class SIFTSpatialMetric extends DistanceMeasure implements Serializable {
    
    private static final long serialVersionUID = 1L;

    @Override
    public float dist(float[] arrFirst, float[] arrSecond)
            throws MetricException {
        float result = 0;
        float factor = arrFirst[0] - arrSecond[0];
        result += factor * factor;
        factor = arrFirst[1] - arrSecond[1];
        result += factor * factor;
        return result;
    }

    /**
     * @param first Data instance.
     * @param second Data instance.
     * @return Distance.
     * @throws MetricException
     */
    public float dist(DataInstance first, DataInstance second)
            throws MetricException {
        return dist(first.fAttr, second.fAttr);
    }
}
