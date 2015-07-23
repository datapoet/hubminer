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
package learning.unsupervised.outliers;

import java.util.ArrayList;

import data.neighbors.NeighborSetFinder;
import data.representation.DataSet;
import data.representation.util.DataMineConstants;
import distances.primary.CombinedMetric;

/**
 * This class implements the algorithm first described in 2000 in a paper by
 * Markus M. Breunig, Hans-Peter Kriegel, Raymond T. Ng and Jörg Sander titled
 * 'LOF: Identifying Density-Based Local Outliers' that was presented at the ACM
 * SIGMOD conference. A local outlier factor is defined as a ratio between the
 * local reachability density in the point of interest and the average
 * reachability density of its neighbor points.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class LocalOutlierFactor extends OutlierDetector {

    // Relative difference in densities that defines outliers.
    public static final float DEFAULT_CUTOFF_THRESHOLD = 1.4f;
    private float cutoffThreshold = DEFAULT_CUTOFF_THRESHOLD;
    // Neighborhood size to consider.
    private int k = 5;
    private NeighborSetFinder nsf;
    private CombinedMetric cmet = CombinedMetric.FLOAT_EUCLIDEAN;
    private float[] localReachabilityDensity;
    private float[] localOutlierFactor;

    /**
     * @param dset Dataset to be analyzed.
     * @param k Neighborhood size.
     * @param cutoffThreshold Cutoff ratio for flagging outliers.
     */
    public LocalOutlierFactor(DataSet dset, int k, float cutoffThreshold) {
        setDataSet(dset);
        this.k = k;
        this.cutoffThreshold = cutoffThreshold;
    }

    /**
     * @param dset Dataset to be analyzed.
     * @param nsf NeighborSetFinder object.
     * @param cutoffThreshold Cutoff ratio for flagging outliers.
     */
    public LocalOutlierFactor(DataSet dset, NeighborSetFinder nsf,
            float cutoffThreshold) {
        setDataSet(dset);
        this.nsf = nsf;
        this.cutoffThreshold = cutoffThreshold;
    }

    /**
     * @param dset Dataset to be analyzed.
     * @param cmet Metric.
     * @param k Neighborhood size.
     * @param cutoffThreshold Cutoff ratio for flagging outliers.
     */
    public LocalOutlierFactor(DataSet dset, CombinedMetric cmet, int k,
            float cutoffThreshold) {
        setDataSet(dset);
        this.cmet = cmet;
        this.k = k;
        this.cutoffThreshold = cutoffThreshold;
    }

    /**
     * @param dataset Dataset to be analyzed.
     * @param cmet Metric.
     * @param nsf Neighborhood size.
     * @param cutoffThreshold Cutoff ratio for flagging outliers.
     */
    public LocalOutlierFactor(DataSet dataset, CombinedMetric cmet,
            NeighborSetFinder nsf, float cutoffThreshold) {
        setDataSet(dataset);
        this.cmet = cmet;
        this.nsf = nsf;
        this.cutoffThreshold = cutoffThreshold;
    }

    /**
     * Calculates the reachability distance of i from j.
     *
     * @param i Index of the first point.
     * @param j Index of the second point.
     * @return Reachability distance.
     */
    private float getReachabilityDistance(int i, int j) {
        int first = Math.min(i, j);
        int second = Math.max(i, j);
        return Math.max(nsf.getDistances()[first][second - first - 1],
                nsf.getKDistances()[second][k - 1]);
    }

    @Override
    public void detectOutliers() throws Exception {
        DataSet dset = getDataSet();
        if (dset == null || dset.isEmpty()) {
            throw new OutlierDetectionException("Empty DataSet provided.");
        }
        ArrayList<Float> outlierScores = new ArrayList<>(dset.size());
        ArrayList<Integer> outlierIndexes =
                new ArrayList<>(dset.size());
        localReachabilityDensity = new float[dset.size()];
        localOutlierFactor = new float[dset.size()];
        if (nsf == null || nsf.getKNeighbors() == null) {
            nsf = new NeighborSetFinder(dset, cmet);
            nsf.calculateDistances();
            nsf.calculateNeighborSetsMultiThr(k, 6);
        }
        int[][] kneighbors = nsf.getKNeighbors();
        float sum;
        for (int i = 0; i < dset.size(); i++) {
            sum = 0;
            for (int j = 0; j < k; j++) {
                sum += getReachabilityDistance(i, kneighbors[i][j]);
            }
            localReachabilityDensity[i] = k / sum;
        }
        float maxOutlierScore = 0;
        int count;
        for (int i = 0; i < dset.size(); i++) {
            sum = 0;
            count = 0;
            for (int j = 0; j < k; j++) {
                if (DataMineConstants.isAcceptableFloat(
                        localReachabilityDensity[kneighbors[i][j]])) {
                    count++;
                    sum += localReachabilityDensity[kneighbors[i][j]];
                }
            }
            if (DataMineConstants.isAcceptableFloat(sum)
                    && DataMineConstants.isAcceptableFloat(
                    localReachabilityDensity[i])
                    && localReachabilityDensity[i] != 0) {
                localOutlierFactor[i] =
                        sum / (localReachabilityDensity[i] * count);
                // Here we test if the point is an outlier.
                if (localOutlierFactor[i] > cutoffThreshold) {
                    outlierIndexes.add(i);
                    outlierScores.add(localOutlierFactor[i]);
                    if (localOutlierFactor[i] > maxOutlierScore) {
                        maxOutlierScore = localOutlierFactor[i];
                    }
                }
            }
        }
        if (maxOutlierScore > 0) {
            for (int j = 0; j < outlierScores.size(); j++) {
                outlierScores.set(j, outlierScores.get(j) / maxOutlierScore);
            }
        }
        setOutlierIndexes(outlierIndexes, outlierScores);
    }
}
