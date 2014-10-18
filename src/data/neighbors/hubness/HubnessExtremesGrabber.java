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
import java.util.Arrays;
import util.AuxSort;

/**
 * This class calculates the extremal points w.r.t. hubness for a range of
 * neighborhood sizes, implicitly defined by the provided NeighborSetFinder
 * object. It can calculate either the highest hubness points or the lowest
 * hubness points, depending on the operating mode.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class HubnessExtremesGrabber {

    // A flag indicating whether to fetch the upper or the lower extremes.
    private boolean fetchHigher = true;
    // Object that holds the kNN sets.
    private NeighborSetFinder nsf = null;
    // The neighbor occurrence frequencies of the extremes for all the k-values.
    private float[][] hubnessExtremalScores;
    // The indexes of the extremal points w.r.t. hubness for all the k-values.
    private int[][] hubnessExtremalIndexes;

    /**
     * Initialization.
     *
     * @param fetchHigher Boolean flag indicating whether to fetch the upper or
     * the lower extremes.
     * @param nsf NeighborSetFinder object for kNN calculations. To be provided
     * with pre-computed kNN sets.
     */
    public HubnessExtremesGrabber(boolean fetchHigher, NeighborSetFinder nsf) {
        this.fetchHigher = fetchHigher;
        this.nsf = nsf;
    }

    /**
     * This method calculates the extremal points w.r.t. to the neighbor
     * occurrence frequency for a range of neighborhood sizes.
     *
     * @param numElementsToFetch Integer that is the number of extremes to
     * fetch.
     * @return float[][] representing the extremal neighbor occurrence
     * frequencies for a range of neighborhood sizes.
     */
    public float[][] getHubnessExtremesForKValues(int numElementsToFetch) {
        try {
            // Get the maximum supported neighborhood size.
            int maxNeighborhoodSize = nsf.getKNeighbors()[0].length;
            // Initialize the result arrays.
            float[][] extremalScores =
                    new float[maxNeighborhoodSize][numElementsToFetch];
            int[][] extremalIndexes =
                    new int[maxNeighborhoodSize][numElementsToFetch];
            for (int kIndex = 0; kIndex < maxNeighborhoodSize; kIndex++) {
                // Re-calculate the occurrence counts for the current
                // neighborhood size.
                nsf.recalculateStatsForSmallerK(kIndex + 1);
                // Fetch the neighbor occurrence frequencies as floats.
                float[] neighbOccFreqs = Arrays.copyOf(nsf.getFloatOccFreqs(),
                        nsf.getFloatOccFreqs().length);
                // Ascending sort.
                int[] indexPermutation = AuxSort.sortIndexedValue(
                        neighbOccFreqs, false);
                for (int exIndex = 0; exIndex < numElementsToFetch; exIndex++) {
                    if (fetchHigher) {
                        // Copy from the end.
                        extremalScores[kIndex][exIndex] =
                                neighbOccFreqs[neighbOccFreqs.length
                                - (numElementsToFetch - exIndex)];
                        extremalIndexes[kIndex][exIndex] =
                                indexPermutation[neighbOccFreqs.length
                                - (numElementsToFetch - exIndex)];
                    } else {
                        // Copy from the beginning.
                        extremalScores[kIndex][exIndex] =
                                neighbOccFreqs[exIndex];
                        extremalIndexes[kIndex][exIndex] =
                                indexPermutation[exIndex];
                    }
                }
            }
            this.hubnessExtremalScores = extremalScores;
            this.hubnessExtremalIndexes = extremalIndexes;
            return extremalScores;
        } catch (Exception e) {
            System.err.println(e.getMessage());
            return null;
        }
    }

    /**
     * @return int[][] representing the indexes of the extremal points w.r.t.
     * hubness for a range of neighborhood sizes.
     */
    public int[][] getExtremeIndexes() {
        return hubnessExtremalIndexes;
    }

    /**
     * @return float[][] representing the neighbor occurrence frequencies of the
     * extremal points w.r.t. hubness for a range of neighborhood sizes.
     */
    public float[][] getExtremeScores() {
        return hubnessExtremalScores;
    }
}
