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
import data.representation.DataInstance;
import learning.unsupervised.Cluster;

/**
 * An implementation of the R squared clustering configuration quality index.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class QIndexRS extends ClusteringQualityIndex {

    /**
     * @param clusteringConfiguration An array of clusters.
     * @param dataset Dataset.
     */
    public QIndexRS(Cluster[] clusteringConfiguration, DataSet dataset) {
        setClusters(clusteringConfiguration);
        setDataSet(dataset);
    }

    @Override
    public float validity() throws Exception {
        Cluster[] clusters = getClusters();
        int numClusters = clusters.length;
        DataInstance[] centroids = new DataInstance[numClusters];
        DataSet dataset = getDataSet();
        for (int i = 0; i < numClusters; i++) {
            centroids[i] = clusters[i].getCentroid();
        }
        DataInstance globalDataCentroid = dataset.getCentroid();
        double sst = 0, ssw = 0;
        for (int i = 0; i < dataset.getNumIntAttr(); i++) {
            for (DataInstance instance : dataset.data) {
                sst += Math.pow(instance.iAttr[i]
                        - globalDataCentroid.iAttr[i], 2);
            }
            for (int cIndex = 0; cIndex < numClusters; cIndex++) {
                for (DataInstance instance :
                        clusters[cIndex].getAllInstances()) {
                    ssw += Math.pow(
                            instance.iAttr[i] - centroids[cIndex].iAttr[i], 2);
                }
            }
        }
        for (int i = 0; i < dataset.getNumFloatAttr(); i++) {
            for (DataInstance instance : dataset.data) {
                sst += Math.pow(instance.fAttr[i]
                        - globalDataCentroid.fAttr[i], 2);
            }
            for (int cIndex = 0; cIndex < numClusters; cIndex++) {
                for (DataInstance instance :
                        clusters[cIndex].getAllInstances()) {
                    ssw += Math.pow(
                            instance.fAttr[i] - centroids[cIndex].fAttr[i], 2);
                }
            }
        }
        return (float) (1 - ((sst - ssw) / sst));
    }
}