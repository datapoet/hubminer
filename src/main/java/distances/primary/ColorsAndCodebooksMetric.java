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
 * There are two types of attributes in the array, so they have to be weighted
 * differently.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class ColorsAndCodebooksMetric extends DistanceMeasure
implements Serializable {
    
    private static final long serialVersionUID = 1L;

    private int numColHist;
    private float fact = 0.35f;

    public ColorsAndCodebooksMetric() {
        numColHist = 16;
    }

    /**
     * @param numColHist Number of color histograms.
     */
    public ColorsAndCodebooksMetric(int numColHist) {
        this.numColHist = numColHist;
    }

    /**
     * @param numColHist Number of color histograms.
     * @param fact Weighting factor.
     */
    public ColorsAndCodebooksMetric(int numColHist, float fact) {
        this.numColHist = numColHist;
        this.fact = fact;
    }

    @Override
    public float dist(float[] arrFirst, float[] arrSecond)
            throws MetricException {
        float sum = 0;
        for (int i = 0; i < numColHist; i++) {
            if (!DataMineConstants.isAcceptableFloat(arrFirst[i])
                    || !DataMineConstants.isAcceptableFloat(arrSecond[i])) {
                continue;
            }
            sum += Math.abs(arrFirst[i] - arrSecond[i]);
        }
        sum *= fact;
        for (int i = numColHist; i < arrFirst.length; i++) {
            if (!DataMineConstants.isAcceptableFloat(arrFirst[i])
                    || !DataMineConstants.isAcceptableFloat(arrSecond[i])) {
                continue;
            }
            sum += Math.abs(arrFirst[i] - arrSecond[i]);
        }
        return sum;
    }

    public float dist(DataInstance first, DataInstance second)
            throws MetricException {
        return dist(first.fAttr, second.fAttr);
    }
}
