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
import data.representation.util.DataMineConstants;
import distances.primary.CombinedMetric;
import learning.unsupervised.Cluster;
import statistics.FeatureVariances;

/**
 * An implementation of the SD clustering clusteringConfiguration quality index.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class QIndexSD extends ClusteringQualityIndex {

    private float alpha = Float.MAX_VALUE;

    public QIndexSD(Cluster[] clusteringConfiguration, DataSet wholeDataSet) {
        setClusters(clusteringConfiguration);
        setDataSet(wholeDataSet);
    }

    /**
     * @return alpha parameter in alpha*scatter + dist
     */
    public float getAlpha() {
        return alpha;
    }

    /**
     * Sets the quality index parameter.
     *
     * @param alpha parameter in alpha*scatter + dist
     */
    public void setAlpha(float alpha) {
        this.alpha = alpha;
    }

    @Override
    public float validity() throws Exception {
        if (!DataMineConstants.isAcceptableFloat(alpha)) {
            throw new Exception("Parameter alpha in alpha*scatter + dist"
                    + " needs to be set.");
        }
        float dist = calcCentroidDistanceRatio();
        float scatter = calcScatter();
        float sd = (alpha * scatter) + dist;
        // Original index: better clustering corresponds to lower values. This
        // is transformed to be in accordance with the overall strategy of
        // higher=better that is used throughout other quality indices.
        return 1 / sd;
    }

    /**
     * Calculates the factor that measures between-cluster separation based on
     * distances between the centroids.
     *
     * @return The separation factor for SD index formula.
     * @throws Exception
     */
    private float calcCentroidDistanceRatio() throws Exception {
        Cluster[] clusteringConfiguration = getClusters();
        if (clusteringConfiguration == null
                || clusteringConfiguration.length == 0) {
            throw new Exception("No clustering configuration provided.");
        }
        CombinedMetric cmet = CombinedMetric.EUCLIDEAN;
        int numClusters = clusteringConfiguration.length;
        DataInstance[] centroids = new DataInstance[numClusters];
        for (int i = 0; i < centroids.length; i++) {
            centroids[i] = clusteringConfiguration[i].getCentroid();
        }
        float maxCentroidDistance = 0;
        float minCentroidDistance = Float.MAX_VALUE;
        float centroidDistance;
        float[] denomSums = new float[numClusters];
        for (int i = 0; i < numClusters; i++) {
            if (centroids[i] == null) {
                continue;
            }
            for (int j = i + 1; j < numClusters; j++) {
                if (centroids[j] == null) {
                    continue;
                }
                centroidDistance = cmet.dist(centroids[i], centroids[j]);
                maxCentroidDistance = Math.max(maxCentroidDistance,
                        centroidDistance);
                minCentroidDistance = Math.min(minCentroidDistance,
                        centroidDistance);
                denomSums[i] += centroidDistance;
                denomSums[j] += centroidDistance;
            }
        }
        float distFactor = 0;
        for (int i = 0; i < numClusters; i++) {
            if (DataMineConstants.isAcceptableFloat(denomSums[i])) {
                distFactor += 1f / denomSums[i];
            }
        }
        if (DataMineConstants.isPositive(minCentroidDistance)
                && DataMineConstants.isAcceptableFloat(maxCentroidDistance)) {
            distFactor *= (maxCentroidDistance / minCentroidDistance);
            return distFactor;
        } else {
            return Float.MAX_VALUE;
        }
    }

    /**
     * Calculates the relative variance of the clusters compared to all of data.
     *
     * @return Float value that is the scatter component of the SD index.
     * @throws Exception
     */
    private float calcScatter() throws Exception {
        DataSet dataset = getDataSet();
        Cluster[] clusteringConfiguration = getClusters();
        if (dataset == null || dataset.isEmpty()
                || clusteringConfiguration == null) {
            throw new Exception("No data provided.");
        }
        FeatureVariances dataVariance = new FeatureVariances(dataset);
        dataVariance.calculateAllVariances();
        float varianceNorm = dataVariance.varianceNorm();
        int numClusters = clusteringConfiguration.length;
        float scatter = 0;
        for (int i = 0; i < numClusters; i++) {
            FeatureVariances clusterVariance = new FeatureVariances(dataset);
            clusterVariance.calculateAllVariances(clusteringConfiguration[i]);
            scatter += clusterVariance.varianceNorm();
        }
        scatter = scatter / (varianceNorm * numClusters);
        return scatter;
    }
}