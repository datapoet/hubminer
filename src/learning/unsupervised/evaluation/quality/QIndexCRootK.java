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
 * This class implements the C root k index, that measures the root ratio
 * between  average inter-cluster and total feature variances. It is normalized
 * by the root of the number of clusters, to account for varying cluster
 * numbers. The higher values correspond to better clustering configurations.
 * 
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class QIndexCRootK extends ClusteringQualityIndex {
    
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
    public QIndexCRootK(int numClusters, int[] clusterAssociations,
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
    public QIndexCRootK(Cluster[] clusteringConfiguration,
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
        int numFloatFeatures = dset.getNumFloatAttr();
        int numIntFeatures = dset.getNumIntAttr();
        double[] sqSumTotalFloats = new double[numFloatFeatures];
        double[] sqSumWithinFloats = new double[numFloatFeatures];
        double[] sqSumTotalInts = new double[numIntFeatures];
        double[] sqSumWithinInts = new double[numIntFeatures];
        int clusterIndex;
        // Calculate all the square difference sums for integer and float
        // attributes, both to the local and to the global centroids.
        for (int i = 0; i < dset.size(); i++) {
            DataInstance instance = dset.getInstance(i);
            if (clusterAssociations[i] < 0 || instance.isNoise()) {
                continue;
            }
            clusterIndex = clusterAssociations[i];
            // Float features.
            for (int fIndex = 0; fIndex < numFloatFeatures; fIndex++) {
                // Global squared differences.
                if (DataMineConstants.isAcceptableFloat(instance.fAttr[fIndex])
                        && DataMineConstants.isAcceptableFloat(
                        globalCentroid.fAttr[fIndex])) {
                    sqSumTotalFloats[fIndex] += (instance.fAttr[fIndex] -
                            globalCentroid.fAttr[fIndex]) *
                            (instance.fAttr[fIndex] -
                            globalCentroid.fAttr[fIndex]);
                }
                // Local squared differences.
                if (DataMineConstants.isAcceptableFloat(instance.fAttr[fIndex])
                        && DataMineConstants.isAcceptableFloat(
                        clusterCentroids[clusterIndex].fAttr[fIndex])) {
                    sqSumWithinFloats[fIndex] += (instance.fAttr[fIndex] -
                            clusterCentroids[clusterIndex].fAttr[fIndex]) *
                            (instance.fAttr[fIndex] -
                            clusterCentroids[clusterIndex].fAttr[fIndex]);
                }
            }
            // Integer features.
            for (int iIndex = 0; iIndex < numIntFeatures; iIndex++) {
                // Global squared differences.
                if (DataMineConstants.isAcceptableInt(instance.iAttr[iIndex])
                        && DataMineConstants.isAcceptableInt(
                        globalCentroid.iAttr[iIndex])) {
                    sqSumTotalInts[iIndex] += (instance.iAttr[iIndex] -
                            globalCentroid.iAttr[iIndex]) *
                            (instance.iAttr[iIndex] -
                            globalCentroid.iAttr[iIndex]);
                }
                // Local squared differences.
                if (DataMineConstants.isAcceptableInt(instance.iAttr[iIndex])
                        && DataMineConstants.isAcceptableInt(
                        clusterCentroids[clusterIndex].iAttr[iIndex])) {
                    sqSumWithinInts[iIndex] += (instance.iAttr[iIndex] -
                            clusterCentroids[clusterIndex].iAttr[iIndex]) *
                            (instance.iAttr[iIndex] -
                            clusterCentroids[clusterIndex].iAttr[iIndex]);
                }
            }
        }
        double rootRatioSum = 0;
        for (int fIndex = 0; fIndex < numFloatFeatures; fIndex++) {
            if (DataMineConstants.isPositive(sqSumTotalFloats[fIndex])) {
                // The root ratios of inter-cluster sums and the totals.
                rootRatioSum += Math.sqrt(
                        (sqSumTotalFloats[fIndex] -
                        sqSumWithinFloats[fIndex]) / sqSumTotalFloats[fIndex]);
            }
        }
        for (int iIndex = 0; iIndex < numIntFeatures; iIndex++) {
            if (DataMineConstants.isPositive(sqSumTotalInts[iIndex])) {
                rootRatioSum += Math.sqrt(
                        (sqSumTotalInts[iIndex] -
                        sqSumWithinInts[iIndex]) / sqSumTotalInts[iIndex]);
            }
        }
        int numFeatures = numIntFeatures + numFloatFeatures;
        double cRootKIndex = rootRatioSum * (1 / Math.sqrt(numClusters)) *
                (1d / numFeatures);
        return (float) cRootKIndex;
    }
    
}
