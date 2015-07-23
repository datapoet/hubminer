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

/**
 * Davies-Bouldin index is a sum of maximized inter-cluster ratios of cumulative
 * dispersion over inter-cluster distance.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class QIndexDaviesBouldin extends ClusteringQualityIndex {

    CombinedMetric cmet = null;

    /**
     * @param clusteringConfiguration An array of clusters.
     * @param dataset DataSet object.
     */
    public QIndexDaviesBouldin(Cluster[] clusteringConfiguration,
            DataSet dataset) {
        setClusters(clusteringConfiguration);
        setDataSet(dataset);
        cmet = CombinedMetric.EUCLIDEAN;
    }

    /**
     * @param clusteringConfiguration An array of clusters.
     * @param dataset DataSet object.
     * @param cmet CombinedMetric object.
     */
    public QIndexDaviesBouldin(Cluster[] clusteringConfiguration,
            DataSet dataset, CombinedMetric cmet) {
        setClusters(clusteringConfiguration);
        setDataSet(dataset);
        this.cmet = cmet;
    }

    @Override
    public float validity() throws Exception {
        float dbIndex = 0;
        Cluster[] clusters = getClusters();
        if (clusters == null || clusters.length == 0) {
            return 0;
        }
        int numClusters = clusters.length;
        DataInstance[] centroids = new DataInstance[numClusters];
        for (int i = 0; i < numClusters; i++) {
            centroids[i] = clusters[i].getCentroid();
        }
        // R-factors for each cluster, that maximize the between-cluster
        // dispersion over distance ratios.
        float[] R = new float[numClusters];
        // An array of dispersion measures.
        float[] dispersions = new float[numClusters];
        for (int i = 0; i < R.length; i++) {
            dispersions[i] = clusters[i].averageIntraDistance(centroids[i],
                    cmet);
        }
        float centroidDistance;
        for (int i = 0; i < numClusters; i++) {
            for (int j = 0; j < i; j++) {
                centroidDistance = cmet.dist(centroids[i], centroids[j]);
                if (DataMineConstants.isZero(centroidDistance)) {
                    return 0;
                }
                R[i] = Math.max(R[i],
                        ((dispersions[i] + dispersions[j]) / centroidDistance));

            }
            for (int j = i + 1; j < numClusters; j++) {
                centroidDistance = cmet.dist(centroids[i], centroids[j]);
                if (DataMineConstants.isZero(centroidDistance)) {
                    return 0;
                }
                R[i] = Math.max(R[i],
                        ((dispersions[i] + dispersions[j]) / centroidDistance));

            }
            dbIndex += R[i];
        }
        dbIndex = dbIndex / numClusters;
        return 1f / dbIndex;
    }
}