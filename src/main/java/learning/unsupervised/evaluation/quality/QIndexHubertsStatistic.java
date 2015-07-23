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

import data.representation.DataInstance;
import data.representation.DataSet;
import distances.primary.CombinedMetric;
import learning.unsupervised.Cluster;

/**
 * This class implements the Hubert's G statistic for internal clustering 
 * evaluation. The index is applied to the proximity matrices. Normalization 
 * option is also available, to reach the normalized Hubert's statistic.
 * 
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class QIndexHubertsStatistic extends ClusteringQualityIndex {
    
    private CombinedMetric cmet = null;
    private int[] clusterAssociations;
    private float[][] distances;
    private boolean dGiven = false;
    private boolean normalizedMeasure = true;
    
    /**
     * Initialization.
     * 
     * @param clusterAssociations Cluster association array for the points.
     * @param dset DataSet object.
     */
    public QIndexHubertsStatistic(int[] clusterAssociations,
            DataSet dset) {
        this.clusterAssociations = clusterAssociations;
        setDataSet(dset);
        cmet = CombinedMetric.EUCLIDEAN;
    }

    /**
     * Initialization.
     * 
     * @param clusterAssociations Cluster association array for the points.
     * @param dset DataSet object.
     * @param cmet CombinedMetric object for distance calculations.
     */
    public QIndexHubertsStatistic(int[] clusterAssociations,
            DataSet dset, CombinedMetric cmet) {
        this.clusterAssociations = clusterAssociations;
        setDataSet(dset);
        this.cmet = cmet;
    }
    
    /**
     * Initialization.
     * 
     * @param clusterAssociations Cluster association array for the points.
     * @param dset DataSet object.
     * @param normalizedMeasure Boolean flag indicating whether to use the
     * normalized Hubert's statistic or the non-normalized one.
     */
    public QIndexHubertsStatistic(int[] clusterAssociations,
            DataSet dset, boolean normalizedMeasure) {
        this.clusterAssociations = clusterAssociations;
        setDataSet(dset);
        cmet = CombinedMetric.EUCLIDEAN;
        this.normalizedMeasure = normalizedMeasure;
    }

    /**
     * Initialization.
     * 
     * @param clusterAssociations Cluster association array for the points.
     * @param dset DataSet object.
     * @param cmet CombinedMetric object for distance calculations.
     * @param normalizedMeasure Boolean flag indicating whether to use the
     * normalized Hubert's statistic or the non-normalized one.
     */
    public QIndexHubertsStatistic(int[] clusterAssociations,
            DataSet dset, CombinedMetric cmet, boolean normalizedMeasure) {
        this.clusterAssociations = clusterAssociations;
        setDataSet(dset);
        this.cmet = cmet;
        this.normalizedMeasure = normalizedMeasure;
    }

    /**
     * @param distances float[][] representing the upper triangular distance 
     * matrix of the data.
     */
    public void setDistanceMatrix(float[][] distances) {
        this.distances = distances;
        this.dGiven = true;
    }
    
    @Override
    public float validity() throws Exception {
        DataSet instances = getDataSet();
        if (!dGiven) {
            distances = instances.calculateDistMatrix(cmet);
        }
        Cluster[] clusterConfiguration =
                Cluster.getConfigurationFromAssociations(clusterAssociations,
                instances);
        int numClusters = clusterConfiguration.length;
        DataInstance[] centroids = new DataInstance[numClusters];
        for (int cIndex = 0; cIndex < numClusters; cIndex++) {
            centroids[cIndex] = clusterConfiguration[cIndex].getCentroid();
        }
        // First calculate the centroid-to-centroid distances.
        float[][] centroidDists = new float[numClusters][];
        for (int cIndex = 0; cIndex < numClusters; cIndex++) {
            centroidDists[cIndex] = new float[numClusters - cIndex - 1];
            for (int cInc = 0; cInc < centroidDists[cIndex].length; cInc++) {
                centroidDists[cIndex][cInc] = cmet.dist(centroids[cIndex],
                        centroids[cIndex + cInc + 1]);
            }
        }
        // Generate a new proximity matrix based on the centroid-to-centroid
        // distances.
        int dataSize = instances.size();
        float[][] distancesAsCDists = new float[dataSize][];
        int minClusterIndex, maxClusterIndex;
        for (int i = 0; i < dataSize; i++) {
            distancesAsCDists[i] = new float[dataSize - i - 1];
            if (clusterAssociations[i] <= 0 ||
                    instances.getInstance(i).isNoise()) {
                continue;
            }
            for (int j = 0; j < distancesAsCDists[i].length; j++) {
                if (clusterAssociations[i + j + 1] <= 0 ||
                        instances.getInstance(i + j + 1).isNoise()) {
                    continue;
                }
                minClusterIndex = Math.min(clusterAssociations[i],
                        clusterAssociations[i + j + 1]);
                maxClusterIndex = Math.max(clusterAssociations[i],
                        clusterAssociations[i + j + 1]);
                if (minClusterIndex != maxClusterIndex) {
                    distancesAsCDists[i][j] = centroidDists[minClusterIndex][
                            maxClusterIndex - minClusterIndex - 1];
                }
            }
        }
        // Now, if normalized measure is to be used, calculate the mean values
        // and the variances.
        double avgOriginalDist = 0;
        double avgCentroidDist = 0;
        double distStDevOriginal = 0;
        double distStDevCentroid = 0;
        if (normalizedMeasure) {
            long dCounter = 0;
            for (int i = 0; i < dataSize; i++) {
                if (clusterAssociations[i] <= 0 ||
                        instances.getInstance(i).isNoise()) {
                    continue;
                }
                for (int j = 0; j < distances[i].length; j++) {
                    if (clusterAssociations[i + j + 1] <= 0 ||
                            instances.getInstance(i + j + 1).isNoise()) {
                        continue;
                    }
                    ++dCounter;
                    avgOriginalDist = (avgOriginalDist / dCounter) *
                            (dCounter - 1) + distances[i][j] / dCounter;
                    avgCentroidDist = (avgCentroidDist / dCounter) *
                            (dCounter - 1) + distancesAsCDists[i][j] / dCounter;
                }
            }
            double distVarOriginal = 0;
            double distVarCentroid = 0;
            dCounter = 0;
            for (int i = 0; i < dataSize; i++) {
                if (clusterAssociations[i] <= 0 ||
                        instances.getInstance(i).isNoise()) {
                    continue;
                }
                for (int j = 0; j < distances[i].length; j++) {
                    if (clusterAssociations[i + j + 1] <= 0 ||
                            instances.getInstance(i + j + 1).isNoise()) {
                        continue;
                    }
                    ++dCounter;
                    distVarOriginal = (distVarOriginal / dCounter) *
                            (dCounter - 1) + ((1d / dCounter) *
                            (distances[i][j] - avgOriginalDist) *
                            (distances[i][j] - avgOriginalDist));
                    distVarCentroid = (distVarCentroid / dCounter) *
                            (dCounter - 1) + ((1d / dCounter) *
                            (distancesAsCDists[i][j] - avgCentroidDist) *
                            (distancesAsCDists[i][j] - avgCentroidDist));
                }
            }
            distStDevOriginal = Math.sqrt(distVarOriginal);
            distStDevCentroid = Math.sqrt(distVarCentroid);
        }
        // Now calculate the Hubert's statistic. First do the summations.
        double hubertsIndex = 0;
        long numComparisons = 0;
        for (int i = 0; i < dataSize; i++) {
            if (clusterAssociations[i] <= 0 ||
                    instances.getInstance(i).isNoise()) {
                continue;
            }
            for (int j = 0; j < distances[i].length; j++) {
                if (clusterAssociations[i + j + 1] <= 0 ||
                        instances.getInstance(i + j + 1).isNoise()) {
                    continue;
                }
                numComparisons++;
                if (normalizedMeasure) {
                    hubertsIndex += (distances[i][j] - avgOriginalDist) *
                            (distancesAsCDists[i][j] - avgCentroidDist);
                } else {
                    hubertsIndex += (distances[i][j] * distancesAsCDists[i][j]);
                }
            }
        }
        hubertsIndex /= numComparisons;
        if (normalizedMeasure) {
            hubertsIndex /= (distStDevOriginal * distStDevCentroid);
        }
        return (float) hubertsIndex;
    }
    
}
