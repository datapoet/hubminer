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
 * Distance between SIFTVector-s.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class SIFTMetric extends DistanceMeasure implements Serializable {
    
    private static final long serialVersionUID = 1L;

    /**
     * This method calculates the distance between two data instances that
     * represent SIFT features, so that the first 4 values are ignores, since
     * they represent location, angle and scale.
     *
     * @param first DataInstance that is the first instance.
     * @param second DataInstance that is the second instance.
     * @return float value that is the distance.
     * @throws MetricException
     */
    public float dist(DataInstance first, DataInstance second)
            throws MetricException {
        float sum = 0;
        float factor;
        for (int i = 4; i < first.fAttr.length; i++) {
            factor = (first.fAttr[i] - second.fAttr[i]);
            sum += factor * factor;
        }
        sum = (float) Math.sqrt(sum);
        return sum;
    }

    @Override
    public float dist(float[] siftArrayFirst, float[] siftArraySecond)
            throws MetricException {
        // The first four elements in the array are the x/y values, scale and
        // angle. This distance takes into account only the descriptor itself
        // hence it iterates from i=4.
        float sum = 0;
        float factor;
        for (int i = 4; i < siftArrayFirst.length; i++) {
            factor = (siftArrayFirst[i] - siftArraySecond[i]);
            sum += factor * factor;
        }
        sum = (float) Math.sqrt(sum);
        return sum;
    }
}
