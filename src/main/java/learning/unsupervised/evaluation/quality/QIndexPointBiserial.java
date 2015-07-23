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
 * This class implements the point-biserial clustering quality index.
 * 
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class QIndexPointBiserial extends ClusteringQualityIndex {
    
    private CombinedMetric cmet = null;
    private int[] clusterAssociations;
    private float[][] distances;
    private boolean dGiven = false;
    
    /**
     * Initialization.
     * 
     * @param clusterAssociations Cluster association array for the points.
     * @param dset DataSet object.
     */
    public QIndexPointBiserial(int[] clusterAssociations,
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
    public QIndexPointBiserial(int[] clusterAssociations,
            DataSet dset, CombinedMetric cmet) {
        this.clusterAssociations = clusterAssociations;
        setDataSet(dset);
        this.cmet = cmet;
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
        // The number of intra- and inter-cluster distances.
        long numIntraDists = 0;
        long numInterDists = 0;
        int numClusters = clusterConfiguration.length;
        if (numClusters < 2) {
            return 0;
        }
        int minIndex, maxIndex;
        DataInstance instanceFirst, instanceSecond;
        // Average intra- and inter-cluster distances.
        double avgIntraDist = 0;
        double avgInterDist = 0;
        for (int c1 = 0; c1 < numClusters; c1++) {
            // Skip empty clusters.
            if (clusterConfiguration[c1].isEmpty()) {
                continue;
            }
            for (int i = 0; i < clusterConfiguration[c1].size(); i++) {
                for (int j = i + 1; j < clusterConfiguration[c1].size(); j++) {
                    minIndex = Math.min(clusterConfiguration[c1].indexes.get(i),
                            clusterConfiguration[c1].indexes.get(j));
                    maxIndex = Math.max(clusterConfiguration[c1].indexes.get(i),
                            clusterConfiguration[c1].indexes.get(j));
                    instanceFirst = instances.getInstance(minIndex);
                    instanceSecond = instances.getInstance(maxIndex);
                    if (!instanceFirst.isNoise() && !instanceSecond.isNoise() &&
                            clusterAssociations[minIndex] >= 0 &&
                            clusterAssociations[maxIndex] >= 0) {
                        // Update the average.
                        ++numIntraDists;
                        avgIntraDist = (avgIntraDist / numIntraDists) *
                                (numIntraDists - 1) + distances[minIndex][
                                maxIndex - minIndex - 1] / numIntraDists;
                    }
                }
            }
            for (int c2 = c1 + 1; c2 < numClusters; c2++) {
                // Skip empty clusters.
                if (clusterConfiguration[c2].isEmpty()) {
                    continue;
                }
                for (int i = 0; i < clusterConfiguration[c1].size(); i++) {
                    for (int j = 0; j < clusterConfiguration[c2].size(); j++) {
                        minIndex = Math.min(
                                clusterConfiguration[c1].indexes.get(i),
                                clusterConfiguration[c2].indexes.get(j));
                        maxIndex = Math.max(
                                clusterConfiguration[c1].indexes.get(i),
                                clusterConfiguration[c2].indexes.get(j));
                        instanceFirst = instances.getInstance(minIndex);
                        instanceSecond = instances.getInstance(maxIndex);
                        if (!instanceFirst.isNoise() &&
                                !instanceSecond.isNoise() &&
                                clusterAssociations[minIndex] >= 0 &&
                                clusterAssociations[maxIndex] >= 0) {
                            // Update the average.
                            ++numInterDists;
                            avgInterDist = (avgInterDist / numInterDists) *
                                    (numInterDists - 1) + distances[minIndex][
                                    maxIndex - minIndex - 1] / numInterDists;
                        }
                    }
                }
            }
        }
        long numTotalDists = numInterDists + numIntraDists;
        double distMean = ((avgInterDist * numInterDists) +
                (avgIntraDist * numIntraDists)) / numTotalDists;
        double distVariance = 0;
        long pairCounter = 0;
        float currDist;
        // Now do a pass to determine the overall distance variance.
        for (int i = 0; i < instances.size(); i++) {
            for (int j = i + 1; j < instances.size(); j++) {
                instanceFirst = instances.getInstance(i);
                instanceSecond = instances.getInstance(j);
                if (!instanceFirst.isNoise() && !instanceSecond.isNoise() &&
                        clusterAssociations[i] >= 0 &&
                        clusterAssociations[j] >= 0) {
                    currDist = distances[i][j - i - 1];
                    ++pairCounter;
                    distVariance = (distVariance / pairCounter) *
                            (pairCounter - 1) + ((1d / pairCounter) *
                            (currDist - distMean) *
                            (currDist - distMean));
                }
            }
        }
        double distStDev = Math.sqrt(distVariance);
        double totalDists = numIntraDists + numInterDists;
        double pbValue = ((avgInterDist - avgIntraDist) *
                Math.sqrt((numInterDists * numIntraDists) /
                (totalDists * totalDists))) / distStDev;
        return (float) pbValue;
    }
    
}
