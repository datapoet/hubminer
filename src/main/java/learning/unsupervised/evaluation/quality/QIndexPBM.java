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
import java.util.ArrayList;
import learning.unsupervised.Cluster;

/**
 * This class implements the commonly used PBM clustering quality index, which 
 * was first proposed in the paper titled: 'Validity Index for Crisp and Fuzzy 
 * Clusters', which was published in Pattern Recognition in 2004 and authored by
 * Malay K. Pakhira, Sanghamitra Bandyopadhyay and Ujjwal Maulik. It is the 
 * normalized ratio between all centered data distances and the within-cluster 
 * point-to-centroid data distances.
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class QIndexPBM extends ClusteringQualityIndex {
    
    // Number of clusters in the data.
    private int numClusters;
    // An array of cluster associations for all data points.
    private int[] clusterAssociations;
    // Object that does distance calculations.
    private CombinedMetric cmet = CombinedMetric.EUCLIDEAN;
    
    /**
     * Initialization.
     *
     * @param numClusters Number of clusters in the configuration.
     * @param clusterAssociations Cluster association array for the points.
     * @param dset DataSet object.
     */
    public QIndexPBM(int numClusters, int[] clusterAssociations,
            DataSet dset) {
        this.clusterAssociations = clusterAssociations;
        this.numClusters = numClusters;
        setDataSet(dset);
    }
    
    /**
     * Initialization.
     *
     * @param clusteringConfiguration Cluster configuration.
     * @param dset DataSet object.
     */
    public QIndexPBM(Cluster[] clusteringConfiguration,
            DataSet dset) {
        setClusters(clusteringConfiguration);
        setDataSet(dset);
        clusterAssociations = Cluster.getAssociationsForClustering(
                clusteringConfiguration, dset);
        numClusters = clusteringConfiguration == null
                ? 0 : clusteringConfiguration.length;
    }
    
    /**
     * Initialization.
     *
     * @param numClusters Number of clusters in the configuration.
     * @param clusterAssociations Cluster association array for the points.
     * @param dset DataSet object.
     * @param cmet CombinedMetric object for distance calculations.
     */
    public QIndexPBM(int numClusters, int[] clusterAssociations,
            DataSet dset, CombinedMetric cmet) {
        this.clusterAssociations = clusterAssociations;
        this.numClusters = numClusters;
        setDataSet(dset);
        this.cmet = cmet;
    }
    
    /**
     * Initialization.
     *
     * @param clusteringConfiguration Cluster configuration.
     * @param dset DataSet object.
     * @param cmet CombinedMetric object for distance calculations.
     */
    public QIndexPBM(Cluster[] clusteringConfiguration,
            DataSet dset, CombinedMetric cmet) {
        setClusters(clusteringConfiguration);
        setDataSet(dset);
        clusterAssociations = Cluster.getAssociationsForClustering(
                clusteringConfiguration, dset);
        numClusters = clusteringConfiguration == null
                ? 0 : clusteringConfiguration.length;
        this.cmet = cmet;
    }
    
    @Override
    public float validity() throws Exception {
        DataSet dset = getDataSet();
        if (clusterAssociations == null) {
            throw new Exception("Null cluster associations array. "
                    + "No configuration to evaluate.");
        }
        // Initialize and populate the non-noisy index lists.
        ArrayList<Integer> allNonNoisyIndexes =
                new ArrayList<>(clusterAssociations.length);
        ArrayList<Integer>[] clusterIndexes =
                new ArrayList[numClusters];
        for (int cIndex = 0; cIndex < numClusters; cIndex++) {
            clusterIndexes[cIndex] = new ArrayList<>(
                    Math.max(20, clusterAssociations.length / numClusters));
        }
        for (int i = 0; i < dset.size(); i++) {
            if (clusterAssociations[i] >= 0) {
                allNonNoisyIndexes.add(i);
                clusterIndexes[clusterAssociations[i]].add(i);
            }
        }
        // Get the global and local cluster centroids.
        // The global centroid is calculated from a restriction on non-noisy
        // data objects.
        Cluster globalCluster =
                dset.getSubsample(allNonNoisyIndexes).makeClusterObject();
        DataInstance globalCentroid = globalCluster.getCentroid();
        DataInstance[] clusterCentroids = new DataInstance[numClusters];
        Cluster[] clusters = new Cluster[numClusters];
        for (int cIndex = 0; cIndex < numClusters; cIndex++) {
            clusters[cIndex] = new Cluster(dset);
            for (int index: clusterIndexes[cIndex]) {
                clusters[cIndex].addInstance(index);
            }
            clusterCentroids[cIndex] = clusters[cIndex].getCentroid();
        }
        double globalCentroidDistSum = 0;
        double withinClusterCentroidDistSum = 0;
        float distGlobal, distLocal, distCentroids;
        float maxCentroidDist = 0;
        for (int cIndex = 0; cIndex < numClusters; cIndex++) {
            for (int index: clusterIndexes[cIndex]) {
                DataInstance instance = dset.getInstance(index);
                distGlobal = cmet.dist(instance, globalCentroid);
                distLocal = cmet.dist(instance, clusterCentroids[cIndex]);
                globalCentroidDistSum += distGlobal;
                withinClusterCentroidDistSum += distLocal;
            }
            for (int cSecond = cIndex + 1; cSecond < numClusters; cSecond++) {
                distCentroids = cmet.dist(clusterCentroids[cIndex],
                        clusterCentroids[cSecond]);
                maxCentroidDist = Math.max(maxCentroidDist, distCentroids);
            }
        }
        double indexValue = Math.pow(
                (globalCentroidDistSum / withinClusterCentroidDistSum) *
                (maxCentroidDist / numClusters), 2);
        return (float) indexValue;
    }
    
}
