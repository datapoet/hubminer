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
 * This class assigns distances to pairs of data instances based on a
 * precomputed data matrix. This makes sense when distances are computed in a
 * complicated fashion, geodesic or by clustering, etc. They are not calculated
 * directly from the features.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 * @extends CombinedMetric
 */
public class ImageMetricsFromClustering extends CombinedMetric
implements Serializable {
    
    private static final long serialVersionUID = 1L;

    private float[][] imageDistances = null;
    private int indexOfMatrixIndex = 0;

    /**
     * Initialization.
     * 
     * @param imageDistances The pre-computed distance matrix.
     */
    public ImageMetricsFromClustering(float[][] imageDistances) {
        this.imageDistances = imageDistances;
    }

    /**
     * Initialization.
     * 
     * @param imageDistances The pre-computed distance matrix.
     * @param indexOfMatrixIndex Integer that is the index of the integer
     * feature in the data instance identifiers that points towards the
     * corresponding matrix row/column index.
     */
    public ImageMetricsFromClustering(float[][] imageDistances,
            int indexOfMatrixIndex) {
        this.imageDistances = imageDistances;
        this.indexOfMatrixIndex = indexOfMatrixIndex;
    }

    @Override
    public float dist(DataInstance first, DataInstance second)
            throws Exception {
        return imageDistances[first.getIdentifier().iAttr[indexOfMatrixIndex]][
                second.getIdentifier().iAttr[indexOfMatrixIndex]];
    }
}