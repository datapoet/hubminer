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
package data.neighbors.hubness;

import data.neighbors.NeighborSetFinder;
import statistics.HigherMoments;
import util.ArrayUtil;

/**
 * This class implements the method for batch-calculating the standard deviation
 * of the neighbor occurrence frequency for a range of k-values.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class HubnessVarianceExplorer {

    // Object that holds the kNN sets.
    private NeighborSetFinder nsf = null;

    /**
     * Initialization.
     *
     * @param nsf NeighborSetFinder object that holds the kNN sets that have
     * been calculated beforehand.
     */
    public HubnessVarianceExplorer(NeighborSetFinder nsf) {
        this.nsf = nsf;
    }

    /**
     * This method calculates the standard deviation of the neighbor occurrence
     * frequency for a range of k-values supported by the provided
     * NeighborSetFinder object.
     *
     * @return
     */
    public float[] getStDevForKRange() {
        if (nsf == null) {
            return null;
        }
        // Get the upper limit on the k-value.
        int maxNeighborhoodSize = nsf.getKNeighbors()[0].length;
        float[] occFreqStDevs = new float[maxNeighborhoodSize];
        // Go through the k-range.
        for (int kIndex = 0; kIndex < maxNeighborhoodSize; kIndex++) {
            // Re-calculate the neighbor occurrence frequency.
            nsf.recalculateStatsForSmallerK(kIndex + 1);
            // Get the neighbor occurrence frequencies.
            float[] neighbOccFreqs = nsf.getFloatOccFreqs();
            float hMean = ArrayUtil.findMean(neighbOccFreqs);
            occFreqStDevs[kIndex] = HigherMoments.calculateArrayStDev(hMean,
                    neighbOccFreqs);
        }
        return occFreqStDevs;
    }
}
