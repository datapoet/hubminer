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
package learning.unsupervised.evaluation.quality;

import data.representation.DataSet;
import distances.primary.CombinedMetric;
import learning.unsupervised.Cluster;
import statistics.HigherMoments;
import util.AuxSort;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * Assigns an index value close to 1 to the good configurations and a value
 * close to -1 to the bad ones. This particular implementation also keeps track
 * of the A and B values in hubs, anti-hubs and regular points. A values are the
 * average dissimilarities to points from within the same cluster and B values
 * are the lowest average dissimilarities to another cluster.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class QIndexSilhouette extends ClusteringQualityIndex {

    // Object that does distance calculations.
    private CombinedMetric cmet = null;
    // Number of clusters in the data.
    private int numClusters;
    // An array of cluster associations for all data points.
    private int[] clusterAssociations;
    // 'A' values for all data points.
    private float[] instanceAarray;
    // 'B' values for all data points.
    private float[] instanceBarray;
    // Silhouette index values for all data points.
    private float[] instanceSilhouetteArray;
    // Average Silhouette values for all clusters.
    private float[] clusterSilhouetteArray;
    // Distance matrix.
    private float[][] distances;
    // Whether distances were set explicitly or they need to be calculated.
    private boolean dGiven = false;
    // Neighbor occurrence frequencies for all data points.
    public int[] hubnessArray = null;
    // Total 'A' and 'B' Silhouette values, corresponding to average
    // within cluster distances and lowest average inter-cluster distances.
    public double ATOTAL = 0;
    public double BTOTAL = 0;
    // 'A' and 'B' values for hub points within the data.
    public double HATOTAL = 0;
    public double HBTOTAL = 0;
    // 'A' and 'B' values for anti-hubs.
    public double AHATOTAL = 0;
    public double AHBTOTAL = 0;
    // 'A' and 'B' values of regular points that are neither hubs nor anti-hubs.
    public double REGATOTAL = 0;
    public double REGBTOTAL = 0;

    /**
     * Sets the distance matrix.
     *
     * @param distances Distance matrix, given in the typical HubMiner way -
     * each row contains only d(i, j) where j > i, so that distances[i, j] = d(i
     * + j + 1)
     */
    public void setDistanceMatrix(float[][] distances) {
        this.distances = distances;
        if (distances != null) {
            this.dGiven = true;
        }
    }

    /**
     * @param numClusters Number of clusters in the configuration.
     * @param clusterAssociations Cluster association array for the points.
     * @param dset DataSet object.
     */
    public QIndexSilhouette(int numClusters, int[] clusterAssociations,
            DataSet dset) {
        this.clusterAssociations = clusterAssociations;
        this.numClusters = numClusters;
        setDataSet(dset);
        cmet = CombinedMetric.EUCLIDEAN;
    }

    /**
     * @param numClusters Number of clusters in the configuration.
     * @param clusterAssociations Cluster association array for the points.
     * @param dset DataSet object.
     * @param cmet Metric to use for estimating the quality.
     */
    public QIndexSilhouette(int numClusters, int[] clusterAssociations,
            DataSet dset, CombinedMetric cmet) {
        this.clusterAssociations = clusterAssociations;
        this.numClusters = numClusters;
        setDataSet(dset);
        this.cmet = cmet;
    }

    /**
     *
     * @param clusteringConfiguration Cluster configuration.
     * @param dset DataSet object.
     * @param cmet Metric to use for estimating the quality.
     */
    public QIndexSilhouette(Cluster[] clusteringConfiguration, DataSet dset,
            CombinedMetric cmet) {
        setClusters(clusteringConfiguration);
        setDataSet(dset);
        clusterAssociations = Cluster.getAssociationsForClustering(
                clusteringConfiguration, dset);
        numClusters = clusteringConfiguration == null
                ? 0 : clusteringConfiguration.length;
        this.cmet = cmet;
    }

    /**
     * @return Silhouette values for all instances.
     */
    public float[] getInstanceSilhouetteArray() {
        return instanceSilhouetteArray;
    }

    /**
     * @return Silhouette values for all clusters.
     */
    public float[] getClusterSilhouetteArray() {
        return clusterSilhouetteArray;
    }

    /**
     * @param cIndex Cluster index.
     * @return Silhouette value for the specified cluster.
     */
    public float getSilhouetteForCluster(int cIndex) {
        return clusterSilhouetteArray[cIndex];
    }

    @Override
    public float validity() throws Exception {
        float resultingIndex = 0f;
        if (clusterAssociations == null) {
            return 0;
        }
        int dataSize = clusterAssociations.length;
        int trueDataSize = 0;
        ATOTAL = 0;
        BTOTAL = 0;
        HATOTAL = 0;
        HBTOTAL = 0;
        AHATOTAL = 0;
        AHBTOTAL = 0;
        REGATOTAL = 0;
        REGBTOTAL = 0;
        DataSet instances = getDataSet();
        float[] elementsPerCluster = new float[numClusters];
        float[][] avgElClustDist = new float[dataSize][numClusters];
        int offset;
        if (!dGiven) {
            distances = new float[dataSize][];
            for (int i = 0; i < dataSize; i++) {
                distances[i] = new float[dataSize - i - 1];
            }
        }
        for (int i = 0; i < dataSize; i++) {
            if (instances.data.get(i).getCategory() != -1
                    && clusterAssociations[i] >= 0) {
                trueDataSize++;
                elementsPerCluster[clusterAssociations[i]]++;
                for (int j = i + 1; j < dataSize; j++) {
                    if (instances.data.get(j).getCategory() != -1
                            && clusterAssociations[j] >= 0) {
                        offset = j - i - 1;
                        if (!dGiven) {
                            distances[i][offset] = cmet.dist(
                                    instances.data.get(i),
                                    instances.data.get(j));
                        }
                        avgElClustDist[i][clusterAssociations[j]] +=
                                distances[i][offset];
                        avgElClustDist[j][clusterAssociations[i]] +=
                                distances[i][offset];
                    }
                }
            }
        }
        boolean[] nonEmpty = new boolean[numClusters];
        // Detect if there are any empty clusters, since they will be ignored.
        for (int i = 0; i < numClusters; i++) {
            if (elementsPerCluster[i] == 0) {
                nonEmpty[i] = false;
            } else {
                nonEmpty[i] = true;
            }
        }
        // Now turn the totals into averages.
        for (int i = 0; i < dataSize; i++) {
            for (int j = 0; j < numClusters; j++) {
                if (nonEmpty[j]) {
                    avgElClustDist[i][j] /= elementsPerCluster[j];
                }
            }
        }
        // Now find the actual index values for all the data instances and track
        // the totals for the clusters.
        instanceSilhouetteArray = new float[dataSize];
        instanceAarray = new float[dataSize];
        instanceBarray = new float[dataSize];
        clusterSilhouetteArray = new float[numClusters];
        float a; // Avg dist to own cluster.
        float b; // Avg dist to closest other cluster.
        int ownCluster;
        for (int i = 0; i < dataSize; i++) {
            b = Float.MAX_VALUE;
            ownCluster = clusterAssociations[i];
            if (instances.data.get(i).getCategory() != -1
                    && clusterAssociations[i] >= 0) {
                a = avgElClustDist[i][ownCluster];
                for (int j = 0; j < ownCluster; j++) {
                    if (nonEmpty[j]) {
                        b = Math.min(b, avgElClustDist[i][j]);
                    }
                }
                for (int j = ownCluster + 1; j < numClusters; j++) {
                    if (nonEmpty[j]) {
                        b = Math.min(b, avgElClustDist[i][j]);
                    }
                }
                instanceSilhouetteArray[i] = (b - a) / Math.max(Math.abs(a),
                        Math.abs(b));
                instanceAarray[i] = a;
                instanceBarray[i] = b;
                ATOTAL += a;
                BTOTAL += b;
                clusterSilhouetteArray[ownCluster] +=
                        instanceSilhouetteArray[i];
            }
        }
        // Now find the actual average index for each cluster.
        for (int i = 0; i < numClusters; i++) {
            if (nonEmpty[i]) {
                clusterSilhouetteArray[i] /= elementsPerCluster[i];
            }
        }
        // Now find the actual SilhouetteIndex.
        for (int i = 0; i < numClusters; i++) {
            resultingIndex += elementsPerCluster[i] * clusterSilhouetteArray[i];
        }
        resultingIndex /= (float) trueDataSize;
        ATOTAL /= (float) trueDataSize;
        BTOTAL /= (float) trueDataSize;
        if (hubnessArray != null) {
            // Now here we've got to do some stuff with hubness correlations...
            // first of all, we need to divide points into: hubs, anti-hubs and
            // regular. Hubs will be those that are 2 stdevs away from the AVG
            // hubness. We will take the same number of points as anti-hubs.
            // Afterwards, we calculate the averages of Silhouette index
            // components for these point types.
            float med = HigherMoments.calculateArrayMean(hubnessArray);
            float stDev = HigherMoments.calculateArrayStDev(med, hubnessArray);
            float up = med + 2 * stDev;
            ArrayList<Integer> hubs = new ArrayList<>(500);
            ArrayList<Integer> antiHubs = new ArrayList<>(2000);
            ArrayList<Integer> regulars = new ArrayList<>(2000);
            for (int i = 0; i < hubnessArray.length; i++) {
                if (hubnessArray[i] > up) {
                    hubs.add(i);
                }
            }
            int n = hubs.size();
            // Now pick n antihubs.
            int[] hArrCopy = Arrays.copyOf(hubnessArray, hubnessArray.length);
            // Ascending sort.
            int[] reSortIndexes = AuxSort.sortIndexedValue(hArrCopy, false);
            for (int i = 0; i < n; i++) {
                antiHubs.add(reSortIndexes[i]);
            }
            for (int i = n; i < hubnessArray.length - n; i++) {
                regulars.add(reSortIndexes[i]);
            }
            // Ok, the points are divided, now calculate the relevant quantities
            for (int i = 0; i < hubs.size(); i++) {
                HATOTAL += instanceAarray[hubs.get(i)];
                HBTOTAL += instanceBarray[hubs.get(i)];
            }
            HATOTAL /= hubs.size();
            HBTOTAL /= hubs.size();
            for (int i = 0; i < antiHubs.size(); i++) {
                AHATOTAL += instanceAarray[antiHubs.get(i)];
                AHBTOTAL += instanceBarray[antiHubs.get(i)];
            }
            AHATOTAL /= antiHubs.size();
            AHBTOTAL /= antiHubs.size();
            for (int i = 0; i < regulars.size(); i++) {
                REGATOTAL += instanceAarray[regulars.get(i)];
                REGBTOTAL += instanceBarray[regulars.get(i)];
            }
            REGATOTAL /= regulars.size();
            REGBTOTAL /= regulars.size();
            hubnessArray = null;
        }
        return resultingIndex;
    }
}
