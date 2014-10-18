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

/**
 * This utility class helps with identifying how many points lie above or below
 * a certain neighbor occurrence frequency threshold for various k values.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class HubnessAboveThresholdExplorer {

    // The occurrence frequency threshold.
    private int occurrenceThreshold = 0;
    // Whether to count the points above or below the threshold.
    private boolean selectAboveThreshold = true;
    // Object that holds the kNN sets that have been calculated beforehand.
    private NeighborSetFinder nsf = null;

    /**
     * Initialization.
     *
     * @param occurrenceThreshold Integer that is the desired occurrence
     * frequency threshold.
     * @param selectAboveThreshold Boolean flag indicating whether to count the
     * points above or below the threshold.
     * @param nsf NeighborSetFinder object that holds the kNN sets that have
     * been calculated beforehand.
     */
    public HubnessAboveThresholdExplorer(int occurrenceThreshold,
            boolean selectAboveThreshold, NeighborSetFinder nsf) {
        this.occurrenceThreshold = occurrenceThreshold;
        this.selectAboveThreshold = selectAboveThreshold;
        this.nsf = nsf;
    }

    /**
     * This method calculates the percentage of points that lie above or below
     * the specified occurrence threshold for each k value up until the maximum
     * neighborhood size supported by the current calculations in the
     * NeighborSetFinder object that was provided at initialization time.
     *
     * @return float[] representing a series of percentages of points that lie
     * either above or below the occurrence threshold (as specified in the
     * constructor) for each k value from the supported k-range.
     */
    public float[] getThresholdPercentageArray() {
        if (nsf == null) {
            return null;
        }
        int maxNeighborhoodSize = nsf.getKNeighbors()[0].length;
        float[] pointPercArray = new float[maxNeighborhoodSize];
        // There are two operating modes.
        if (selectAboveThreshold) {
            // Select above the threshold.
            pointPercArray[maxNeighborhoodSize - 1] =
                    nsf.getPercFrequentAtLeast(occurrenceThreshold);
        } else {
            // Select below the threshold.
            pointPercArray[maxNeighborhoodSize - 1] =
                    nsf.getPercFrequentLessOrEqualThan(occurrenceThreshold);
        }
        // Go over the rest of the range.
        for (int kIndex = 1; kIndex < maxNeighborhoodSize; kIndex++) {
            // Re-calculate the occurrence frequencies.
            nsf.recalculateStatsForSmallerK(kIndex);
            if (selectAboveThreshold) {
                // Select above the threshold.
                pointPercArray[kIndex - 1] =
                        nsf.getPercFrequentAtLeast(occurrenceThreshold);
            } else {
                // Select below the threshold.
                pointPercArray[kIndex - 1] =
                        nsf.getPercFrequentLessOrEqualThan(occurrenceThreshold);
            }
        }
        // Re-calculate the occurrence frequencies.
        nsf.recalculateStatsForSmallerK(maxNeighborhoodSize);
        return pointPercArray;
    }
}
