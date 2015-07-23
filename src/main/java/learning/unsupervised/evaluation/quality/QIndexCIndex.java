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
import data.representation.util.DataMineConstants;
import distances.primary.CombinedMetric;
import java.util.ArrayList;
import java.util.Collections;

/**
 * This class implements a complement of the C index that was introduced by
 * Hubert and Schultz in 1976 - so that we have "the higher the better", as in
 * other quality indices. C index is calculated by comparing the sum of
 * intra-cluster distances to max and min distance sums that have the same
 * number of distances as there are intra-cluster pairs.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class QIndexCIndex extends ClusteringQualityIndex {

    private CombinedMetric cmet = null;
    private int[] clusterAssociations;
    private float[][] distMat;

    /**
     * @param clusterAssociations Cluster association array for the points.
     * @param dataset DataSet object.
     */
    public QIndexCIndex(int[] clusterAssociations, DataSet dataset) {
        this.clusterAssociations = clusterAssociations;
        setDataSet(dataset);
        cmet = CombinedMetric.EUCLIDEAN;
    }

    /**
     * @param clusterAssociations Cluster association array for the points.
     * @param dataset DataSet object.
     * @param cmet CombinedMetricObject.
     */
    public QIndexCIndex(int[] clusterAssociations, DataSet dataset,
            CombinedMetric cmet) {
        setDataSet(dataset);
        this.clusterAssociations = clusterAssociations;
        this.cmet = cmet;
    }

    /**
     * @param distances Float matrix of distances between data points.
     */
    public void setDistanceMatrix(float[][] distances) {
        this.distMat = distances;
    }

    @Override
    public float validity() throws Exception {
        DataSet instances = getDataSet();
        if (distMat == null) {
            distMat = instances.calculateDistMatrix(cmet);
        }
        int size = instances.size();
        float CIndexComplement;
        int numIntraDistances = 0;
        float sumIntraDistances = 0;
        int intraPairCount = 0;

        ArrayList<Float> admissableDistances =
                new ArrayList<>(size * (size - 1) / 2);
        for (int i = 0; i < size; i++) {
            for (int j = i + 1; j < size; j++) {
                if (instances.getInstance(i).notNoise()
                        && instances.getInstance(j).notNoise()) {
                    intraPairCount++;
                    admissableDistances.add(distMat[i][j - i - 1]);
                    if (clusterAssociations[i] == clusterAssociations[j]) {
                        numIntraDistances++;
                        sumIntraDistances += distMat[i][j - i - 1];
                    }
                }
            }
        }
        if (numIntraDistances == 0 || !DataMineConstants.isAcceptableFloat(
                sumIntraDistances)) {
            return 0;
        }
        Collections.sort(admissableDistances); // Ascending order.
        float minSums = 0;
        float maxSums = 0;
        for (int i = 0; i < numIntraDistances; i++) {
            minSums += admissableDistances.get(i);
            maxSums += admissableDistances.get(intraPairCount - 1 - i);
        }
        if (DataMineConstants.isZero(maxSums - minSums)) {
            return 0;
        } else {
            CIndexComplement = (sumIntraDistances - minSums)
                    / (maxSums - minSums);
            return 1f - CIndexComplement;
        }
    }
}
