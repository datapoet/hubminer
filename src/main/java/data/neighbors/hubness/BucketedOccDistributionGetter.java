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
import util.ArrayUtil;

/**
 * This utility class helps with visualizing the neighbor occurrence frequency
 * distributions over a range of specified neighborhood sizes. For each
 * neighborhood size, the neighbor occurrence frequency array is obtained and
 * then turned into a histogram with a certain bucket width.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class BucketedOccDistributionGetter {

    private int bucketWidth;
    private NeighborSetFinder nsf;
    private int kMax;

    /**
     * Initialization.
     *
     * @param nsf NeighborSetFinder object for kNN calculations.
     * @param kMax Integer that is the maximal neighborhood size to consider.
     * @param bucketWidth Integer that is the discrete bucket width for the
     * histograms that will be generated.
     */
    public BucketedOccDistributionGetter(NeighborSetFinder nsf, int kMax,
            int bucketWidth) {
        this.bucketWidth = bucketWidth;
        this.nsf = nsf;
        this.kMax = kMax;
    }

    /**
     * Calculates and outputs the bucketed neighbor occurrence frequency
     * distributions for the range of k values.
     *
     * @return int[][] representing an array of neighbor occurrence frequency
     * histograms for each evaluated neighborhood size.
     */
    public int[][] getBucketedDistributions() {
        int[][] histogramArray = new int[kMax][];
        for (int kIndex = 0; kIndex < kMax; kIndex++) {
            nsf.recalculateStatsForSmallerK(kIndex + 1);
            int[] neighbOccFreqs = nsf.getNeighborFrequencies();
            histogramArray[kIndex] =
                    getBucketedDistributionForNumberArray(neighbOccFreqs);
        }
        return histogramArray;
    }

    /**
     * This method transforms a value array into a histogram with the previously
     * specified bucket width.
     *
     * @param valueArray int[] of values to turn into a histogram.
     * @return int[] that is the histogram generated from the specified values.
     */
    private int[] getBucketedDistributionForNumberArray(int[] valueArray) {
        int maxValue = ArrayUtil.max(valueArray);
        int numBuckets = (maxValue / bucketWidth) + 1;
        int[] valueHistogram = new int[numBuckets];
        int buckIndex;
        for (int i = 0; i < valueArray.length; i++) {
            buckIndex = valueArray[i] / bucketWidth;
            valueHistogram[buckIndex]++;
        }
        return valueHistogram;
    }
}
