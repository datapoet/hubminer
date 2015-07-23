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
import data.representation.DataSet;
import java.util.Arrays;
import learning.unsupervised.Cluster;
import util.AuxSort;

/**
 * This utility class implements the methods for batch-calculating the diameters
 * and average intra-cluster distances of top hub clusters over a range of
 * neighborhood sizes.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class TopHubsClusterUtil {

    // Object that holds the kNN sets.
    private NeighborSetFinder nsf = null;
    // Diameters of clusters formed by a specified number of top hub-points in
    // the data, over a range of neighborhood sizes.
    private float[] topHubClusterDiameters;
    // Average intra-cluster distances of clusters formed by a specified number
    // of top hub-points in the data, over a range of neighborhood sizes.
    private float[] topHubClusterAvgDists;

    /**
     * Initialization.
     *
     * @param nsf NeighborSetFinder object with the pre-computed kNN sets.
     */
    public TopHubsClusterUtil(NeighborSetFinder nsf) {
        this.nsf = nsf;
    }

    /**
     * @return Diameters of clusters formed by a specified number of top
     * hub-points in the data, over a range of neighborhood sizes.
     */
    public float[] getTopHubClusterDiameters() {
        return topHubClusterDiameters;
    }

    /**
     * @return Average intra-cluster distances of clusters formed by a specified
     * number of top hub-points in the data, over a range of neighborhood sizes.
     */
    public float[] getTopHubClusterAvgDists() {
        return topHubClusterAvgDists;
    }

    /**
     * This method calculates the diameters of the top-hub clusters and their
     * average intra-cluster distances, over a range of neighborhood sizes that
     * are supported by the provided NeighborSetFinder object.
     *
     * @param numTopHubs Integer that is the number of top hubs to form the
     * clusters from.
     * @throws Exception
     */
    public void calcTopHubnessDiamAndAvgDist(int numTopHubs) throws Exception {
        if (nsf == null) {
            return;
        }
        int kMax = nsf.getKNeighbors()[0].length;
        // Initialize the result arrays.
        topHubClusterDiameters = new float[kMax];
        topHubClusterAvgDists = new float[kMax];
        for (int kIndex = 0; kIndex < kMax; kIndex++) {
            // Re-calculate the kNN stats.
            nsf.recalculateStatsForSmallerK(kIndex + 1);
            float[] neighbOccFreqs = Arrays.copyOf(nsf.getFloatOccFreqs(),
                    nsf.getFloatOccFreqs().length);
            // Descending sort.
            int[] indexPermutation =
                    AuxSort.sortIndexedValue(neighbOccFreqs, true);
            DataSet dset = nsf.getDataSet();
            // Form a cluster from the top hubs.
            Cluster dataTopHubsCluster = new Cluster(dset, numTopHubs);
            // Insert the points into the cluster.
            for (int j = 0; j < Math.min(numTopHubs, dset.size()); j++) {
                dataTopHubsCluster.addInstance(indexPermutation[j]);
            }
            // Calculate the cluster diameter.
            topHubClusterDiameters[kIndex] =
                    dataTopHubsCluster.calculateDiameter(
                    nsf.getCombinedMetric());
            // Calculate the average intra-cluster distance.
            topHubClusterAvgDists[kIndex] =
                    dataTopHubsCluster.averageIntraDistance(
                    nsf.getCombinedMetric());
        }
    }
}
