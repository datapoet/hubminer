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
 * Essentially, Manhattan metric, but the distance between the data instances is
 * calculated as the distances between their integer histogram arrays.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class SIFTImageHistogramMetric extends Manhattan
implements Serializable {
    
    private static final long serialVersionUID = 1L;

    /**
     * The default constructor.
     */
    public SIFTImageHistogramMetric() {
    }

    @Override
    public float dist(DataInstance first, DataInstance second)
            throws MetricException {
        return dist(first.iAttr, second.iAttr);
    }
}
