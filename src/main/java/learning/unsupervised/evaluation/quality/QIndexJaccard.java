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
 * Jaccard index is defined as: A / A + B + C where: A = Number of pairs of
 * points in same cluster with same labels; B = Number of pairs of points in
 * same cluster with different labels; C = Number of pairs of points in
 * different clusters with same labels;
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class QIndexJaccard extends ClusteringQualityIndex {

    private int[] clusterAssociations;
    private DataSet dataContext = null;

    /**
     * @param dataset DataSet object.
     * @param clusterAssociations An array of clusters.
     */
    public QIndexJaccard(DataSet dataset, int[] clusterAssociations) {
        this.dataContext = dataset;
        this.clusterAssociations = clusterAssociations;
    }

    @Override
    public float validity() throws Exception {
        float jaccardIndex;
        float a = 0;
        float b = 0;
        float c = 0;
        int size = dataContext.size();
        for (int i = 0; i < size; i++) {
            for (int j = i + 1; j < size; j++) {
                if (dataContext.getInstance(i).isNoise()
                        || dataContext.getInstance(j).isNoise()) {
                    continue;
                }
                if (dataContext.getLabelOf(i) == dataContext.getLabelOf(j)) {
                    if (clusterAssociations[i] == clusterAssociations[j]) {
                        a++;
                    } else {
                        c++;
                    }
                } else {
                    if (clusterAssociations[i] == clusterAssociations[j]) {
                        b++;
                    }
                }
            }
        }
        jaccardIndex = a / (a + b + c);
        return jaccardIndex;
    }
}
