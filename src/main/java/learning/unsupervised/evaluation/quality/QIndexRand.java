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

/**
 * a + d / a + b + c + d
 * a = in same cluster with same labels b = in same cluster with different
 * labels c = in different clusters with same labels d = in different cluster
 * with different labels labels are assumed to be given in instance category
 * fields, clustering in clusterAssociations array can also be used to compare
 * the similarity between two cluster configurations, used for stability
 * testing.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class QIndexRand extends ClusteringQualityIndex {

    private int[] clusterAssociations;
    DataSet dset = null;

    /**
     *
     * @param dset Dataset.
     * @param clusterAssociations Integer array marking cluster associations.
     */
    public QIndexRand(DataSet dset, int[] clusterAssociations) {
        this.dset = dset;
        this.clusterAssociations = clusterAssociations;
    }

    @Override
    public float validity() throws Exception {
        float resultingIndex;
        if (dset == null || clusterAssociations == null) {
            return 0;
        }
        float a = 0;
        float d = 0;
        float totalPairs = (dset.size() * (dset.size() - 1)) / 2;
        for (int i = 0; i < dset.size(); i++) {
            for (int j = i + 1; j < dset.size(); j++) {
                if (dset.getLabelOf(i) == dset.getLabelOf(j)
                        && clusterAssociations[i] != -1
                        && clusterAssociations[j] != -1) {
                    if (clusterAssociations[i] == clusterAssociations[j]) {
                        a++;
                    }
                } else {
                    if (clusterAssociations[i] != clusterAssociations[j]) {
                        d++;
                    }
                }
            }
        }
        resultingIndex = (a + d) / totalPairs;
        return resultingIndex;
    }

    /**
     * Compare the current cluster configuration with the second one.
     *
     * @param secondAssociations Integer array defining the second configuration
     * @return
     */
    public float compareToConfiguration(int[] secondAssociations) {
        float resultingIndex;
        float a = 0;
        float d = 0;
        if (secondAssociations == null || clusterAssociations == null) {
            return 0;
        }
        float totalPairs = (dset.size() * (dset.size() - 1)) / 2;
        for (int i = 0; i < dset.size(); i++) {
            for (int j = i + 1; j < dset.size(); j++) {
                if (secondAssociations[i] == secondAssociations[j]) {
                    if (clusterAssociations[i] == clusterAssociations[j]) {
                        a++;
                    }
                } else {
                    if (clusterAssociations[i] != clusterAssociations[j]) {
                        d++;
                    }
                }
            }
        }
        resultingIndex = (a + d) / totalPairs;
        return resultingIndex;
    }
}
