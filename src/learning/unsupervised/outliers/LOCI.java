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
package learning.unsupervised.outliers;

import java.util.ArrayList;

import data.neighbors.NeighborSetFinder;
import data.representation.DataSet;
import distances.primary.CombinedMetric;
import statistics.HigherMoments;

/**
 * This class implements the Local Correlation Integral approach to outlier
 * detection and removal, as described in the following paper: 'LOCI: Fast
 * Outlier Detection Using the Local Correlation Integral' by Spiros
 * Papadimitriou, Hiroyuki Kitagawa, Phillip B. Gibbons and Christos Faloutsos
 * that was presented at IEEE 19th International Conference on Data Engineering
 * (ICDE'03) in Bangalore, India.
 *
 * @author
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class LOCI extends OutlierDetector {

    // The part of the neighborhood used for calculating MDEFs.
    private float alpha = 0.5f;
    // The multiple of standard deviations that defines outliers.
    private float ksigma = 3f;
    // The minimum number of neighbors in an r-neighborhood.
    private int minNeighbors = 20;
    private CombinedMetric cmet = CombinedMetric.FLOAT_EUCLIDEAN;

    /**
     * @param dataset Dataset to be analyzed.
     * @param cmet The metric object.
     */
    public LOCI(DataSet dataset, CombinedMetric cmet) {
        setDataSet(dataset);
        this.cmet = cmet;
    }

    /**
     * @param dataset Dataset to be analyzed.
     * @param cmet The metric object.
     * @param minNeighbors Minimal number of neighbors for a neighborhood.
     * @param alpha The part of the neighborhood to count.
     * @param ksigma The number of deviations that define outliers.
     */
    public LOCI(DataSet dataset, CombinedMetric cmet, int minNeighbors,
            float alpha, float ksigma) {
        setDataSet(dataset);
        this.cmet = cmet;
        this.minNeighbors = minNeighbors;
        this.alpha = alpha;
        this.ksigma = ksigma;
    }

    @Override
    public void detectOutliers() throws Exception {
        DataSet dataset = getDataSet();
        if (dataset == null || dataset.isEmpty()) {
            throw new OutlierDetectionException("Empty DataSet provided.");
        }
        ArrayList<Float> outlierScores = new ArrayList<>(dataset.size());
        ArrayList<Integer> outlierIndexes =
                new ArrayList<>(dataset.size());
        NeighborSetFinder nsf = new NeighborSetFinder(dataset, cmet);
        nsf.calculateDistances();
        nsf.calculateNeighborSetsMultiThr(minNeighbors, 6);
        float[] alphaRadius = new float[dataset.size()];
        float[] numNeighborsInAlphaRadius = new float[dataset.size()];
        float[] avgNeighborsAlphaNeighbors = new float[dataset.size()];
        float[] mdef = new float[dataset.size()];
        float[] stDevmdef = new float[dataset.size()];
        float[][] kDistances = nsf.getKDistances();
        int[][] kneighbors = nsf.getKNeighbors();
        for (int i = 0; i < dataset.size(); i++) {
            alphaRadius[i] = kDistances[i][minNeighbors - 1] * alpha;
            int k = minNeighbors - 2;
            while (k >= 0 && kDistances[i][k] > alphaRadius[i]) {
                k--;
            }
            numNeighborsInAlphaRadius[i] = k + 1;
        }
        float sum;
        float maxOutlierScore = 0;
        for (int i = 0; i < dataset.size(); i++) {
            sum = 0;
            float[] localAlphaCounts = new float[minNeighbors + 1];
            for (int j = 0; j < minNeighbors; j++) {
                sum += (numNeighborsInAlphaRadius[kneighbors[i][j]] + 1);
                localAlphaCounts[j] =
                        numNeighborsInAlphaRadius[kneighbors[i][j]] + 1;
            }
            localAlphaCounts[minNeighbors] = numNeighborsInAlphaRadius[i] + 1;
            sum += (numNeighborsInAlphaRadius[i] + 1);
            avgNeighborsAlphaNeighbors[i] = sum
                    / (numNeighborsInAlphaRadius[i] + 1);
            mdef[i] = 1 - (numNeighborsInAlphaRadius[i]
                    / avgNeighborsAlphaNeighbors[i]);
            stDevmdef[i] = HigherMoments.calculateArrayStDev(
                    avgNeighborsAlphaNeighbors[i], localAlphaCounts)
                    / avgNeighborsAlphaNeighbors[i];
            if (mdef[i] > ksigma * stDevmdef[i]) {
                // Mark this point as outlier.
                outlierIndexes.add(i);
                outlierScores.add(mdef[i]);
                if (mdef[i] > maxOutlierScore) {
                    maxOutlierScore = mdef[i];
                }
            }
        }
        if (maxOutlierScore > 0) {
            for (int j = 0; j < outlierScores.size(); j++) {
                outlierScores.set(j, outlierScores.get(j) / maxOutlierScore);
            }
        }
        setOutlierIndexes(outlierIndexes, outlierScores);
    }
}
