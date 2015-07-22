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
import statistics.HigherMoments;

/**
 * This class implements the methods for calculating the entropy of kNN sets and
 * reverse kNN sets. This kNN set (non)-homogeneity can be used to quantify the
 * semantic inconsistencies under the assumption of hubness in intrinsically
 * high-dimensional data.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class KNeighborEntropyExplorer {

    // Object that holds the kNN sets.
    private NeighborSetFinder nsf = null;
    // Number of classes in the data.
    private int numClasses = 2;
    // The mean kNN entropy values for a range of neighborhood sizes.
    private float[] currMeanEntArr;
    // The mean reverse neighbor entropy values for a neighborhood size range.
    private float[] currMeanRNNEntArr;
    // The stdev kNN entropy values for a range of neighborhood sizes.
    private float[] currStDevEntArr;
    // The stdev reverse neighbor entropy values for a neighborhood size range.
    private float[] currStDevRNNEntArr;
    // The skew kNN entropy values for a range of neighborhood sizes.
    private float[] currSkewEntArr;
    // The skew reverse neighbor entropy values for a neighborhood size range.
    private float[] currSkewRNNEntArr;
    // The kurtosis kNN entropy values for a range of neighborhood sizes.
    private float[] currKurtosisEntArr;
    // The kurtosis reverse neighbor entropy values for a neighborhood size
    // range.
    private float[] currKurtosisRNNEntArr;

    /**
     * @return float[] that is the array of kNN set entropy means for a range of
     * neighborhood sizes.
     */
    public float[] getDirectEntropyMeans() {
        return currMeanEntArr;
    }

    /**
     * @return float[] that is the array of reverse kNN set entropy means for a
     * range of neighborhood sizes.
     */
    public float[] getReverseEntropyMeans() {
        return currMeanRNNEntArr;
    }

    /**
     * @return float[] that is the array of kNN set entropy standard deviations
     * for a range of neighborhood sizes.
     */
    public float[] getDirectEntropyStDevs() {
        return currStDevEntArr;
    }

    /**
     * @return float[] that is the array of reverse kNN set entropy standard
     * deviations for a range of neighborhood sizes.
     */
    public float[] getReverseEntropyStDevs() {
        return currStDevRNNEntArr;
    }

    /**
     * @return float[] that is the array of kNN set entropy skews for a range of
     * neighborhood sizes.
     */
    public float[] getDirectEntropySkews() {
        return currSkewEntArr;
    }

    /**
     * @return float[] that is the array of reverse kNN set entropy skews for a
     * range of neighborhood sizes.
     */
    public float[] getReverseEntropySkews() {
        return currSkewRNNEntArr;
    }

    /**
     * @return float[] that is the array of kNN set entropy kurtosis for a range
     * of neighborhood sizes.
     */
    public float[] getDirectEntropyKurtosisVals() {
        return currKurtosisEntArr;
    }

    /**
     * @return float[] that is the array of reverse kNN set entropy kurtosis for
     * a range of neighborhood sizes.
     */
    public float[] getReverseEntropyKurtosisVals() {
        return currKurtosisRNNEntArr;
    }

    /**
     * Initialization.
     *
     * @param nsf NeighborSetFinder object that holds the pre-computed kNN sets.
     * @param numClasses Integer that is the number of classes in the data.
     */
    public KNeighborEntropyExplorer(NeighborSetFinder nsf, int numClasses) {
        this.nsf = nsf;
        this.numClasses = numClasses;
    }

    /**
     * This method computes the average difference between the direct and
     * reverse neighbor set entropies over a range of neighborhood sizes.
     *
     * @return float[] that is the array of differences between the direct and
     * reverse kNN entropies, over a range of neighborhood sizes.
     */
    public float[] getAverageDirectAndReverseEntropyDifs() {
        float[] entDifs = new float[currMeanEntArr.length];
        for (int i = 0; i < entDifs.length; i++) {
            entDifs[i] = currMeanEntArr[i] - currMeanRNNEntArr[i];
        }
        return entDifs;
    }

    /**
     * This method calculates the stats of the kNN entropy distribution for a
     * range of k-values, as implicitly defined by the provided
     * NeighborSetFinder object.
     */
    public void calculateDirectEntropyStats() {
        if (nsf == null) {
            return;
        }
        int kMax = nsf.getKNeighbors()[0].length;
        currMeanEntArr = new float[kMax];
        currStDevEntArr = new float[kMax];
        currSkewEntArr = new float[kMax];
        currKurtosisEntArr = new float[kMax];
        for (int kIndex = 0; kIndex < kMax; kIndex++) {
            // Re-calculate the kNN stats.
            nsf.recalculateStatsForSmallerK(kIndex + 1);
            nsf.calculateKEntropies(numClasses, kIndex + 1);
            float[] kEnts = Arrays.copyOf(nsf.getKEntropies(),
                    nsf.getKEntropies().length);
            currMeanEntArr[kIndex] = HigherMoments.calculateArrayMean(kEnts);
            currStDevEntArr[kIndex] = HigherMoments.calculateArrayStDev(
                    currMeanEntArr[kIndex], kEnts);
            currSkewEntArr[kIndex] = HigherMoments.calculateSkewForSampleArray(
                    kEnts);
            currKurtosisEntArr[kIndex] =
                    HigherMoments.calculateKurtosisForSampleArray(kEnts);
        }
    }

    /**
     * This method calculates the stats of the reverse kNN entropy distribution
     * for a range of k-values, as implicitly defined by the provided
     * NeighborSetFinder object.
     */
    public void calculateRNNEntropyStats() {
        int kMax = nsf.getKNeighbors()[0].length;
        currMeanRNNEntArr = new float[kMax];
        currStDevRNNEntArr = new float[kMax];
        currSkewRNNEntArr = new float[kMax];
        currKurtosisRNNEntArr = new float[kMax];
        for (int kIndex = 0; kIndex < kMax; kIndex++) {
            // Re-calculate the kNN stats.
            nsf.recalculateStatsForSmallerK(kIndex + 1);
            nsf.calculateReverseNeighborEntropies(numClasses);
            float[] kEnts = Arrays.copyOf(nsf.getReverseNeighborEntropies(),
                    nsf.getReverseNeighborEntropies().length);
            currMeanRNNEntArr[kIndex] = HigherMoments.calculateArrayMean(kEnts);
            currStDevRNNEntArr[kIndex] = HigherMoments.calculateArrayStDev(
                    currMeanEntArr[kIndex], kEnts);
            currSkewRNNEntArr[kIndex] =
                    HigherMoments.calculateSkewForSampleArray(kEnts);
            currKurtosisRNNEntArr[kIndex] =
                    HigherMoments.calculateKurtosisForSampleArray(kEnts);
        }
    }

    /**
     * This method calculates the stats of both direct and reverse kNN entropy
     * distributions for a range of k-values, as implicitly defined by the
     * provided NeighborSetFinder object.
     */
    public void calculateAllKNNEntropyStats() {
        calculateDirectEntropyStats();
        calculateRNNEntropyStats();
    }
}
