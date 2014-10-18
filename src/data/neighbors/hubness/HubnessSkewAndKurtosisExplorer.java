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

/**
 * This utility class implements the methods that fetch the skewness and
 * kurtosis of the neighbor occurrence frequency distribution for a range of
 * neighborhood sizes.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class HubnessSkewAndKurtosisExplorer {

    // Object that holds the kNN sets.
    private NeighborSetFinder nsf = null;
    // Skewness of the neighbor occurrence frequency distributions over a range
    // of neighborhood sizes.
    private float[] occFreqsSkewness = null;
    // Kurtosis of the neighbor occurrence frequency distributions over a range
    // of neighborhood sizes.
    private float[] occFreqsKurtosis = null;

    /**
     * Initialization.
     *
     * @param nsf NeighborSetFinder object with pre-computed kNN sets.
     */
    public HubnessSkewAndKurtosisExplorer(NeighborSetFinder nsf) {
        this.nsf = nsf;
    }

    /**
     * @return float[] representing the skewness of the neighbor occurrence
     * frequency distributions over a range of neighborhood sizes.
     */
    public float[] getOccFreqsSkewnessArray() {
        return occFreqsSkewness;
    }

    /**
     * @return float[] representing the kurtosis of the neighbor occurrence
     * frequency distributions over a range of neighborhood sizes.
     */
    public float[] getOccFreqsKurtosisArray() {
        return occFreqsKurtosis;
    }

    /**
     * This method calculates the neighbor occurrence frequency skewness and
     * kurtosis for all the neighborhood sizes in the range available in the
     * provided NeighborSetFinder object.
     */
    public void calcSkewAndKurtosisArrays() {
        if (nsf == null) {
            return;
        }
        int kMax = nsf.getKNeighbors()[0].length;
        occFreqsSkewness = new float[kMax];
        occFreqsKurtosis = new float[kMax];
        // Fetch the neighbor occurrence frequencies.
        float[] kneighborFrequencies = nsf.getFloatOccFreqs();
        occFreqsSkewness[kMax - 1] =
                HigherMoments.calculateSkewForSampleArray(kneighborFrequencies);
        occFreqsKurtosis[kMax - 1] =
                HigherMoments.calculateKurtosisForSampleArray(
                kneighborFrequencies);
        for (int kIndex = 0; kIndex < kMax - 1; kIndex++) {
            // Re-calculatethe kNN stats.
            nsf.recalculateStatsForSmallerK(kIndex + 1);
            // Fetch the neighbor occurrence frequencies.
            kneighborFrequencies = nsf.getFloatOccFreqs();
            occFreqsSkewness[kIndex] =
                    HigherMoments.calculateSkewForSampleArray(
                    kneighborFrequencies);
            occFreqsKurtosis[kIndex] =
                    HigherMoments.calculateKurtosisForSampleArray(
                    kneighborFrequencies);
        }
    }
}
