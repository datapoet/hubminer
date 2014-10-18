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
import data.neighbors.NeighborSetFinder;

/**
 * Isolation index was defined by Pauwels and Frederix in 1999 as the average
 * proportion of elements in k-nearest neighbor sets that are in the same
 * cluster as the point itself. It is a measure of clustering homogeneity.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class QIndexIsolation extends ClusteringQualityIndex {

    private NeighborSetFinder nsf = null;
    private int[] clusterAssociations = null;
    // If left as null, it will be calculated from getCategory() calls.

    /**
     * @param nsf NeighborSetFinder object.
     */
    public QIndexIsolation(NeighborSetFinder nsf) {
        this.nsf = nsf;
        setDataSet(nsf.getDataSet());
    }

    /**
     * @param nsf NeighborSetFinder object.
     * @param clusterAssociations An array indicating data to cluster
     * associations.
     */
    public QIndexIsolation(NeighborSetFinder nsf, int[] clusterAssociations) {
        this.nsf = nsf;
        if (nsf != null) {
            setDataSet(nsf.getDataSet());
        }
        this.clusterAssociations = clusterAssociations;
    }

    @Override
    public float validity() throws Exception {
        DataSet instances = getDataSet();
        if (nsf == null || instances == null) {
            return Float.NaN;
        }
        int[][] kneighbors = nsf.getKNeighbors();
        int k = nsf.getCurrK(); //k = kneighbors[0].length;
        float localNotNoisy;
        float localSameClust;
        float nonNoisyNum = 0;
        float sameClusterRate = 0;
        for (int i = 0; i < instances.size(); i++) {
            if (instances.data.get(i).notNoise()) {
                localNotNoisy = 0;
                localSameClust = 0;
                for (int j = 0; j < k; j++) {
                    if (instances.getInstance(kneighbors[i][j]).notNoise()) {
                        localNotNoisy++;
                        if (clusterAssociations != null) {
                            if (clusterAssociations[i]
                                    == clusterAssociations[j]) {
                                localSameClust++;
                            }
                        } else {
                            if (instances.getInstance(
                                    kneighbors[i][j]).getCategory()
                                    == instances.getInstance(i).getCategory()) {
                                localSameClust++;
                            }
                        }
                    }
                }
                if (localNotNoisy != 0) {
                    nonNoisyNum++;
                    sameClusterRate += (localSameClust / localNotNoisy);
                }
            }
        }
        sameClusterRate /= nonNoisyNum;
        return sameClusterRate;
    }
}
