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
package statistics;

import data.representation.DataInstance;
import data.representation.DataSet;
import distances.primary.CombinedMetric;
import learning.unsupervised.Cluster;

/**
 * The correlation ratio is a measure of the relationship between the
 * statistical dispersion within individual categories and the dispersion across
 * the whole population or sample.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class CorrelationRatio {

    DataSet dset;
    int numCategories;
    CombinedMetric cmet;

    /**
     * @param dset DataSet object.
     * @param numCategories Number of categories in the data.
     * @param cmet CombinedMetric object for distance calculations.
     */
    public CorrelationRatio(
            DataSet dset,
            int numCategories,
            CombinedMetric cmet) {
        this.dset = dset;
        this.numCategories = numCategories;
        this.cmet = cmet;
    }

    /**
     * @param dset DataSet object.
     * @param numCategories Number of categories in the data.
     */
    public CorrelationRatio(DataSet dset, int numCategories) {
        this.dset = dset;
        this.numCategories = numCategories;
        this.cmet = CombinedMetric.MANHATTAN;
    }

    /**
     * Calculates the correlation ratio, based on the classes in the provided
     * data set.
     *
     * @return The correlation ratio, as a float value.
     * @throws Exception
     */
    public float getCorrelationRatio() throws Exception {
        DataInstance globalCentroid;
        DataInstance[] clusterCentroids = new DataInstance[numCategories];
        Cluster[] clusters = new Cluster[numCategories];
        for (int i = 0; i < numCategories; i++) {
            clusters[i] = new Cluster(dset, 500);
        }
        DataInstance instance;
        float correlation;
        // Fill in clusters with instances from the same class.
        for (int i = 0; i < dset.size(); i++) {
            instance = dset.data.get(i);
            clusters[instance.getCategory()].addInstance(i);
        }
        // Get the global data centroid.
        globalCentroid = dset.getCentroid();
        float categoriesVar = 0;
        float tmpDist;
        // Calculate class-conditional variance.
        for (int i = 0; i < numCategories; i++) {
            if (!clusters[i].isEmpty()) {
                clusterCentroids[i] = clusters[i].getCentroid();
                tmpDist = cmet.dist(clusterCentroids[i], globalCentroid);
                categoriesVar += clusters[i].size() * tmpDist * tmpDist;
            }
        }
        // Calculate total variance.
        float totalVar = 0;
        for (int i = 0; i < dset.size(); i++) {
            instance = dset.data.get(i);
            tmpDist = cmet.dist(instance, globalCentroid);
            totalVar += tmpDist * tmpDist;
        }
        if (totalVar > 0) {
            // Correlation ratio is defined as a ratio between the dispersions.
            correlation = categoriesVar / totalVar;
            return correlation;
        } else {
            // In this case all examples are the same.
            return 1;
        }
    }
}
