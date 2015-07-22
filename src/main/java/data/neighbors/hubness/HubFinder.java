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
import data.representation.DataInstance;
import data.representation.DataSet;
import distances.primary.CombinedMetric;
import java.util.ArrayList;
import statistics.HigherMoments;

/**
 * This utility class helps with quickly detecting hubs in the data. It
 * implements the methods for automatically calculating the proper neighbor
 * occurrence frequency thresholds and filtering the points based on the hub
 * criterion.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class HubFinder {

    // The data to find the hubs in.
    private DataSet dset = null;
    // The metric to find the hubs for.
    private CombinedMetric cmet = null;
    // Internal object for kNN calculations.
    private NeighborSetFinder nsf = null;
    // The currently employed neighborhood size.
    private int currK = 10;

    /**
     * Initialization.
     *
     * @param dset DataSet object that is the target data.
     * @param cmet CombinedMetric object for distance calculations.
     */
    public HubFinder(DataSet dset, CombinedMetric cmet) {
        this.dset = dset;
        this.cmet = cmet;
        nsf = new NeighborSetFinder(dset, this.cmet);
    }

    /**
     * Initialization.
     *
     * @param dset DataSet object that is the target data.
     * @param distMatrix float[][] that is the upper triangular distance matrix,
     * as used throughout the library.
     * @param cmet CombinedMetric object for distance calculations.
     */
    public HubFinder(DataSet dset, float[][] distMatrix, CombinedMetric cmet) {
        this.dset = dset;
        this.cmet = cmet;
        nsf = new NeighborSetFinder(dset, distMatrix, cmet);
    }

    /**
     * Find the hub points for the specified neighborhood size.
     *
     * @param k Integer that is the neighborhood size to detect the hubs for.
     * @return A list of hub point indexes.
     * @throws Exception
     */
    public ArrayList<Integer> findHubsForK(int k) throws Exception {
        currK = k;
        // This method can be invoked multiple times and the distances are only
        // calculated once.
        if (!nsf.distancesCalculated()) {
            nsf.calculateDistances();
        }
        // Calculate the kNN sets for the current k-value.
        nsf.calculateNeighborSets(currK);
        // The standard deviation of neighbor occurrence frequencies.
        double occFreqStd = Math.sqrt(HigherMoments.calculateArrayStDev(
                k, nsf.getNeighborFrequencies()));
        // The occurrence threshold for hubs.
        int threshold = (int) (k + 2 * occFreqStd) + 1;
        // Fetch the list of hub indexes and return.
        return nsf.getFrequentAtLeast(threshold);
    }

    /**
     * Find the hub points for the specified neighborhood size.
     *
     * @param k Integer that is the neighborhood size to detect the hubs for.
     * @return DataInstance[] of hub data points.
     * @throws Exception
     */
    public DataInstance[] findHubArrayForK(int k) throws Exception {
        ArrayList<Integer> hubIndexes = findHubsForK(k);
        DataInstance[] hubPointArray = new DataInstance[hubIndexes.size()];
        for (int i = 0; i < hubIndexes.size(); i++) {
            hubPointArray[i] = dset.data.get((hubIndexes.get(i)).intValue());
        }
        return hubPointArray;
    }
}
