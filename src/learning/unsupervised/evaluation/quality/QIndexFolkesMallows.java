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
import java.util.HashMap;
import learning.unsupervised.Cluster;

/**
 * This class implements the Folkes-Mallows index for estimating clustering
 * quality. This index requires some form of ground-truth partitioning to exist
 * or if not - then it relies on the existing data labels / categories instead.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class QIndexFolkesMallows extends ClusteringQualityIndex {

    // Number of clusters in the data.
    private int numClusters;
    // An array of cluster associations for all data points.
    private int[] clusterAssociations;
    // Ground truth to compare the associations to.
    private int numClasses;
    private int[] classLabels;

    /**
     * Initialization.
     *
     * @param numClusters Number of clusters in the configuration.
     * @param clusterAssociations Cluster association array for the points.
     * @param dset DataSet object.
     */
    public QIndexFolkesMallows(int numClusters, int[] clusterAssociations,
            DataSet dset) {
        this.clusterAssociations = clusterAssociations;
        this.numClusters = numClusters;
        setDataSet(dset);
        if (dset != null) {
            classLabels = dset.obtainLabelArray();
        }
    }

    /**
     * Initialization.
     *
     * @param numClusters Number of clusters in the configuration.
     * @param clusterAssociations Cluster association array for the points.
     * @param dset DataSet object.
     * @param classLabels int[] that is the ground truth to compare to.
     */
    public QIndexFolkesMallows(int numClusters, int[] clusterAssociations,
            DataSet dset, int[] classLabels) {
        this.clusterAssociations = clusterAssociations;
        this.numClusters = numClusters;
        setDataSet(dset);
        this.classLabels = classLabels;
    }

    /**
     * Initialization. In this case no DataSet object is provided. When the
     * ground truth is given separately, it is not necessary.
     *
     * @param numClusters Number of clusters in the configuration.
     * @param clusterAssociations Cluster association array for the points.
     * @param classLabels int[] that is the ground truth to compare to.
     */
    public QIndexFolkesMallows(int numClusters, int[] clusterAssociations,
            int[] classLabels) {
        this.clusterAssociations = clusterAssociations;
        this.numClusters = numClusters;
        this.classLabels = classLabels;
    }

    /**
     * Initialization.
     *
     * @param clusteringConfiguration Cluster configuration.
     * @param dset DataSet object.
     */
    public QIndexFolkesMallows(Cluster[] clusteringConfiguration,
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
        if (classLabels == null) {
            if (dset != null) {
                classLabels = dset.obtainLabelArray();
                if (classLabels == null) {
                    throw new Exception("Ground truth not available.");
                }
            }
        }
        if (clusterAssociations == null) {
            throw new Exception("Null cluster associations array. "
                    + "No configuration to evaluate.");
        }
        if (classLabels.length != clusterAssociations.length) {
            throw new Exception("Ground truth length not equal to the provided "
                    + "cluster associations array. Abort.");
        }
        // Now standardize the classes and count them, to ensure that there are
        // no holes in the range.
        numClasses = 0;
        HashMap<Integer, Integer> indexToClassMap = new HashMap<>(
                Math.max(2 * numClusters + 1, 20));
        int dataSize = classLabels.length;
        for (int i = 0; i < dataSize; i++) {
            if (!indexToClassMap.containsKey(classLabels[i])) {
                indexToClassMap.put(classLabels[i], numClasses);
                numClasses++;
            }
            classLabels[i] = indexToClassMap.get(classLabels[i]);
        }
        float[][] clusterClassDistributions =
                new float[numClusters][numClasses];
        for (int i = 0; i < dataSize; i++) {
            clusterClassDistributions[clusterAssociations[i]][classLabels[i]]++;
        }
        // Same cluster, same class.
        float aPairs = 0;
        // Same cluster, different class.
        float bPairs = 0;
        // Different cluster, same class.
        float cPairs = 0;
        for (int clustIndex = 0; clustIndex < numClusters; clustIndex++) {
            // First all the pairs within the same clusters.
            for (int classIndex = 0; classIndex < numClasses; classIndex++) {
                if (clusterClassDistributions[clustIndex][classIndex] > 1) {
                    aPairs += (clusterClassDistributions[clustIndex][classIndex]
                            * (clusterClassDistributions[clustIndex][
                            classIndex] - 1)) / 2;
                }
                for (int cSecond = classIndex + 1; cSecond < numClasses;
                        cSecond++) {
                    bPairs += clusterClassDistributions[clustIndex][classIndex]
                            * clusterClassDistributions[clustIndex][cSecond];
                }
            }
            for (int clustSecond = clustIndex + 1; clustSecond < numClusters;
                    clustSecond++) {
                for (int classIndex = 0; classIndex < numClasses;
                        classIndex++) {
                    cPairs += clusterClassDistributions[clustIndex][classIndex]
                            * clusterClassDistributions[clustSecond][
                            classIndex];
                }
            }
        }
        double qIndex = aPairs / Math.sqrt((aPairs + bPairs)
                * (aPairs + cPairs));
        return (float) qIndex;
    }
}
