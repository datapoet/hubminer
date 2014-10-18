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
import learning.unsupervised.Cluster;

/**
 * This class defines the methods for clustering quality assessment.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class ClusteringQualityIndex {

    private Cluster[] configuration;
    private DataSet dataset;

    /**
     * @return Clustering validity.
     */
    public float validity() throws Exception {
        return 0f;
    }

    /**
     * @return Clustering configuration.
     */
    public Cluster[] getClusters() {
        return configuration;
    }

    /**
     * @param configuration Clusters.
     */
    public void setClusters(Cluster[] configuration) {
        this.configuration = configuration;
    }

    /**
     * @return The dataset.
     */
    public DataSet getDataSet() {
        return dataset;
    }

    /**
     * @param dataset The dataset to analyze.
     */
    public void setDataSet(DataSet dataset) {
        this.dataset = dataset;
    }
}