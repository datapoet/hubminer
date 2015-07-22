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

import data.representation.DataSet;
import learning.unsupervised.Cluster;
import learning.unsupervised.ClusteringAlg;
import util.AuxSort;

/**
 * This method searches for outliers as objects that are often associated with
 * small clusters. Of course, the results depend on many things, like the choice
 * of clustering method, the number of clusters and, consequently , what is
 * considered 'small', given the size of the data and the number of clusters.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class IterativeClusteringOutlierDetection extends OutlierDetector {

    private static final int DEFAULT_NUM_CLUSTERS = 50;
    private static final int DEFAULT_ALLOWED_FAILOVERS = 10;
    private int numRuns = 10;
    private int numClusters;
    private int clusterSizeThreshold = 10;
    private ClusteringAlg clusterer;
    private float outlierPercentage = 0.02f;

    /**
     * @param clusterer The clustering method to be applied to the data.
     */
    public IterativeClusteringOutlierDetection(ClusteringAlg clusterer) {
        this.clusterer = clusterer;
    }

    /**
     * @param clusterer The clustering method to be applied to the data.
     * @param numRuns The number of clustering runs.
     * @param clusterSizeThreshold
     */
    public IterativeClusteringOutlierDetection(
            ClusteringAlg clusterer, int numRuns, int clusterSizeThreshold) {
        this.clusterer = clusterer;
        this.numRuns = numRuns;
        this.clusterSizeThreshold = clusterSizeThreshold;
    }

    /**
     * @param perc The percentage of points with the highest outlier scores that
     * are to be considered outliers, according to this method.
     */
    public void setDesiredOutlierPercentage(float perc) {
        outlierPercentage = perc;
    }

    @Override
    public void detectOutliers() throws Exception {
        DataSet dataset = getDataSet();
        if (dataset == null || dataset.isEmpty()) {
            throw new OutlierDetectionException("Empty DataSet provided.");
        }
        float[] outlierScores = new float[dataset.size()];
        numClusters = numClusters == 0 ? DEFAULT_NUM_CLUSTERS : numClusters;
        Cluster[] clustering;
        // Now the clustering is performed the specified number of times.
        for (int runIndex = 0; runIndex < numRuns; runIndex++) {
            boolean clusteringCompleted = false;
            int numAttempts = 0;
            while (!clusteringCompleted) {
                numAttempts++;
                clusteringCompleted = true;
                try {
                    clusterer.cluster();
                } catch (Exception e) {
                    if (numAttempts < DEFAULT_ALLOWED_FAILOVERS) {
                        clusteringCompleted = false;
                    } else {
                        throw new OutlierDetectionException("Clustering fail:"
                                + e.getMessage());
                    }
                }
            }
            clustering = clusterer.getClusters();
            ArrayList<Cluster> smallClusters = new ArrayList<>(
                    clustering.length);
            for (Cluster c : clustering) {
                if (c != null && c.size() < clusterSizeThreshold) {
                    smallClusters.add(c);
                }
            }
            for (Cluster c : smallClusters) {
                for (int index : c.indexes) {
                    outlierScores[index]++;
                }
            }
        }
        int numOutliers = (int) (dataset.size() * outlierPercentage);
        // This can be improved by automatically detecting the outlier number.
        for (int i = 0; i < outlierScores.length; i++) {
            outlierScores[i] /= (float) numRuns;
        }
        int[] reArrIndexes = AuxSort.sortIndexedValue(outlierScores, true);
        ArrayList<Integer> outliers = new ArrayList<>(numOutliers);
        ArrayList<Float> outScoresFinal = new ArrayList<>(numOutliers);
        for (int i = 0; i < numOutliers; i++) {
            outliers.add(reArrIndexes[i]);
            outScoresFinal.add(outlierScores[reArrIndexes[i]]);
        }
        setOutlierIndexes(outliers, outScoresFinal);
    }
}