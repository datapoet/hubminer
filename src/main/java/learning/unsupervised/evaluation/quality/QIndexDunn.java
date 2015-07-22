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
 * Dunn index is the ratio of the smallest inter-cluster distance and the
 * maximum cluster diameter.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class QIndexDunn extends ClusteringQualityIndex {

    private CombinedMetric cmet = null;

    /**
     * @param clusteringConfiguration An array of clusters.
     * @param dataset Dataset object.
     * @param cmet CombinedMetric object.
     */
    public QIndexDunn(Cluster[] clusteringConfiguration, DataSet dataset,
            CombinedMetric cmet) {
        setClusters(clusteringConfiguration);
        setDataSet(dataset);
        this.cmet = cmet;
    }

    /**
     * @param clusteringConfiguration An array indicating data to cluster
     * associations.
     * @param cmet CombinedMetric object.
     */
    public QIndexDunn(Cluster[] clusteringConfiguration, CombinedMetric cmet) {
        setClusters(clusteringConfiguration);
        this.cmet = cmet;
    }

    /**
     * @param clusteringConfiguration An array indicating data to cluster
     * associations.
     * @param dataset DataSet object.
     */
    public QIndexDunn(Cluster[] clusteringConfiguration, DataSet dataset) {
        setClusters(clusteringConfiguration);
        setDataSet(dataset);
        cmet = CombinedMetric.EUCLIDEAN;
    }

    @Override
    public float validity() throws Exception {
        Cluster[] clusters = getClusters();
        if (clusters == null || clusters.length == 0) {
            return 0;
        }
        int numClusters = clusters.length;
        float maxClusterDiameter = 0;
        DataInstance[] centroids = new DataInstance[numClusters];
        // Here we calculate the maximal cluster diameter.
        for (int i = 0; i < numClusters; i++) {
            if (clusters[i] == null || clusters[i].isEmpty()) {
                continue;
            }
            centroids[i] = clusters[i].getCentroid();
            maxClusterDiameter = Math.max(maxClusterDiameter,
                    clusters[i].calculateDiameter(centroids[i], cmet));
        }
        // Now we calculate the minimal between-cluster distance.
        float minBetweenClusterDistance = Float.MAX_VALUE;
        for (int i = 0; i < numClusters; i++) {
            for (int j = i + 1; j < numClusters; j++) {
                if (clusters[i] == null || clusters[j] == null
                        || clusters[i].isEmpty() || clusters[j].isEmpty()) {
                    continue;
                }
                minBetweenClusterDistance = Math.min(minBetweenClusterDistance,
                        cmet.dist(centroids[i], centroids[j]));
            }
        }
        if (DataMineConstants.isZero(maxClusterDiameter)) {
            return Float.MAX_VALUE;
        } else {
            return (minBetweenClusterDistance / maxClusterDiameter);
        }
    }
}