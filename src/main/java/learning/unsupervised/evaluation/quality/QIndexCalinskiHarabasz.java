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
import java.util.ArrayList;
import learning.unsupervised.Cluster;

/**
 * This class implements the Calinski-Harabasz clustering quality index, that is
 * based on a variance ratio between inter-cluster and intra-cluster variances. 
 * It is computed as a ratio of traces of data scatter matrices. The index 
 * values are not limited to the [0, 1] range and it can in fact obtain quite 
 * high values. However, the normalization factor assures that the index does 
 * not monotonously increase with the number of clusters in the data, so that it
 * can be used as an external optimization criterion for determining the optimal
 * cluster configuration and the optimal number of clusters in the data.
 * 
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class QIndexCalinskiHarabasz extends ClusteringQualityIndex {
    
    // Number of clusters in the data.
    private int numClusters;
    // An array of cluster associations for all data points.
    private int[] clusterAssociations;
    
    /**
     * Initialization.
     *
     * @param numClusters Number of clusters in the configuration.
     * @param clusterAssociations Cluster association array for the points.
     * @param dset DataSet object.
     */
    public QIndexCalinskiHarabasz(int numClusters, int[] clusterAssociations,
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
    public QIndexCalinskiHarabasz(Cluster[] clusteringConfiguration,
            DataSet dset) {
        setClusters(clusteringConfiguration);
        setDataSet(dset);
        clusterAssociations = Cluster.getAssociationsForClustering(
                clusteringConfiguration, dset);
        numClusters = clusteringConfiguration == null
                ? 0 : clusteringConfiguration.length;
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
            if (clusterAssociations[i] >= 0 && !dset.getInstance(i).isNoise()) {
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
        // Trace of the intra-cluster data scatter matrix.
        double intraTrace = 0;
        // Trace of the total data scatter matrix.
        double totalTrace = 0;
        int cIndex;
        for (int index: allNonNoisyIndexes) {
            DataInstance instance = dset.getInstance(index);
            cIndex = clusterAssociations[index];
            for (int fIndex = 0; fIndex < dset.getNumFloatAttr(); fIndex++) {
                if (DataMineConstants.isAcceptableFloat(instance.fAttr[fIndex])
                        && DataMineConstants.isAcceptableFloat(
                        globalCentroid.fAttr[fIndex])) {
                    totalTrace += (instance.fAttr[fIndex] -
                            globalCentroid.fAttr[fIndex]) * (
                            instance.fAttr[fIndex] -
                            globalCentroid.fAttr[fIndex]);
                }
                if (DataMineConstants.isAcceptableFloat(instance.fAttr[fIndex])
                        && DataMineConstants.isAcceptableFloat(
                        clusterCentroids[cIndex].fAttr[fIndex])) {
                    intraTrace += (instance.fAttr[fIndex] -
                            clusterCentroids[cIndex].fAttr[fIndex]) * (
                            instance.fAttr[fIndex] -
                            clusterCentroids[cIndex].fAttr[fIndex]);
                }
            }
            for (int iIndex = 0; iIndex < dset.getNumIntAttr(); iIndex++) {
                if (DataMineConstants.isAcceptableInt(instance.iAttr[iIndex])
                        && DataMineConstants.isAcceptableInt(
                        globalCentroid.iAttr[iIndex])) {
                    totalTrace += (instance.iAttr[iIndex] -
                            globalCentroid.iAttr[iIndex]) * (
                            instance.iAttr[iIndex] -
                            globalCentroid.iAttr[iIndex]);
                }
                if (DataMineConstants.isAcceptableInt(instance.iAttr[iIndex])
                        && DataMineConstants.isAcceptableInt(
                        clusterCentroids[cIndex].iAttr[iIndex])) {
                    intraTrace += (instance.iAttr[iIndex] -
                            clusterCentroids[cIndex].iAttr[iIndex]) * (
                            instance.iAttr[iIndex] -
                            clusterCentroids[cIndex].iAttr[iIndex]);
                }
            }
        }
        double interTrace = totalTrace - intraTrace;
        float numNonNoisy = allNonNoisyIndexes.size();
        // The final normalization term prevents the index for increasing 
        // monotonously with the number of clusters, so that it can be used to 
        // distinguish between different cluster configurations.
        double indexValue = (interTrace / intraTrace) *
                ((numNonNoisy - numClusters) / (numClusters - 1));
        return (float) indexValue;
    }
    
}
