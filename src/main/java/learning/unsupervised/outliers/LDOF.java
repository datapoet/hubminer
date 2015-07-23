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

import data.neighbors.NSFUserInterface;
import java.util.ArrayList;

import data.neighbors.NeighborSetFinder;
import data.representation.DataSet;
import distances.primary.CombinedMetric;
import java.util.Arrays;
import util.AuxSort;

/**
 * This class implements the local distance-based outlier factor that was 
 * proposed in the paper titled 'A New Local Distance-Based Outlier Detection
 * Approach for Scattered Real-World Data' by Ke Zhang and Marcus Hutter and 
 * Huidong Jin at PAKDD 2009. The LDOF factor is the ratio between the average
 * distance to the k-NNs and the average distance among the k-NNs.
 * 
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class LDOF extends OutlierDetector implements NSFUserInterface {
    // Relative difference in densities that defines outliers.
    public static final float DEFAULT_OUTLIER_RATIO = 0.05f;
    private float outlierRatio = DEFAULT_OUTLIER_RATIO;
    // Neighborhood size to consider.
    private int k = 5;
    private NeighborSetFinder nsf;
    private CombinedMetric cmet = CombinedMetric.FLOAT_EUCLIDEAN;
    private float[] ldofValues;

    /**
     * @param dset Dataset to be analyzed.
     * @param k Neighborhood size.
     * @param outlierRatio Float value that is the proportion of points to treat
     * as outliers.
     */
    public LDOF(DataSet dset, int k, float outlierRatio) {
        setDataSet(dset);
        this.k = k;
        this.outlierRatio = outlierRatio;
    }

    /**
     * @param dset Dataset to be analyzed.
     * @param nsf NeighborSetFinder object.
     * @param outlierRatio Float value that is the proportion of points to treat
     * as outliers.
     */
    public LDOF(DataSet dset, NeighborSetFinder nsf,
            float outlierRatio) {
        setDataSet(dset);
        this.nsf = nsf;
        this.outlierRatio = outlierRatio;
        if (nsf != null) {
            this.cmet = nsf.getCombinedMetric();
            this.nsf = nsf;
            this.k = nsf.getCurrK();
        }
    }

    /**
     * @param dset Dataset to be analyzed.
     * @param cmet Metric.
     * @param k Neighborhood size.
     * @param outlierRatio Float value that is the proportion of points to treat
     * as outliers.
     */
    public LDOF(DataSet dset, CombinedMetric cmet, int k,
            float outlierRatio) {
        setDataSet(dset);
        this.cmet = cmet;
        this.k = k;
        this.outlierRatio = outlierRatio;
    }

    /**
     * @param dataset Dataset to be analyzed.
     * @param cmet Metric.
     * @param nsf Neighborhood size.
     * @param outlierRatio Float value that is the proportion of points to treat
     * as outliers.
     */
    public LDOF(DataSet dataset, CombinedMetric cmet,
            NeighborSetFinder nsf, float outlierRatio) {
        setDataSet(dataset);
        this.cmet = cmet;
        this.nsf = nsf;
        this.outlierRatio = outlierRatio;
        if (nsf != null) {
            this.k = nsf.getCurrK();
        }
    }
    
    @Override
    public void setNSF(NeighborSetFinder nsf) {
        this.nsf = nsf;
        if (nsf != null) {
            this.k = nsf.getCurrK();
        }
    }

    @Override
    public NeighborSetFinder getNSF() {
        return nsf;
    }

    @Override
    public void noRecalcs() {
    }
    
    @Override
    public int getNeighborhoodSize() {
        return k;
    }

    @Override
    public void detectOutliers() throws Exception {
        DataSet dset = getDataSet();
        if (dset == null || dset.isEmpty()) {
            throw new OutlierDetectionException("Empty DataSet provided.");
        }
        if (outlierRatio > 1 || outlierRatio < 0) {
            throw new Exception("Invalid outlier ratio: " + outlierRatio);
        }
        int size = dset.size();
        ArrayList<Float> outlierScores = new ArrayList<>(dset.size());
        ArrayList<Integer> outlierIndexes =
                new ArrayList<>(size);
        ldofValues = new float[size];
        // If not calculated, calculate the kNN sets.
        if (nsf == null || nsf.getKNeighbors() == null) {
            nsf = new NeighborSetFinder(dset, cmet);
            nsf.calculateDistances();
            nsf.calculateNeighborSetsMultiThr(k, 6);
        }
        int[][] kNeighbors = nsf.getKNeighbors();
        // Distances to neighbor points.
        float[][] kDistances = nsf.getKDistances();
        // The whole distance matrix on the training data.
        float[][] dMat = nsf.getDistances();
        double kDistancesAvg;
        double interNeighborDistAvg;
        int first, second;
        for (int i = 0; i < size; i++) {
            kDistancesAvg = 0;
            interNeighborDistAvg = 0;
            for (int kInd = 0; kInd < k; kInd++) {
                kDistancesAvg += kDistances[i][kInd];
                for (int kIndSecond = kInd + 1; kIndSecond < k; kIndSecond++) {
                    first = Math.min(kNeighbors[i][kInd],
                            kNeighbors[i][kIndSecond]);
                    second = Math.max(kNeighbors[i][kInd],
                            kNeighbors[i][kIndSecond]);
                    interNeighborDistAvg += dMat[first][second - first - 1];
                }
            }
            if (interNeighborDistAvg > 0) {
                ldofValues[i] = ((float) (kDistancesAvg / k)) /
                        ((float) interNeighborDistAvg / (k * (k - 1)));
            }
        }
        float maxOutlierScore = 0;
        int numOutliers = Math.min( (int) Math.ceil(outlierRatio * size), size);
        // Descending sort.
        float[] ldofValuesSorted = Arrays.copyOf(ldofValues, size);
        int[] reArr = AuxSort.sortIndexedValue(ldofValuesSorted, true);
        for (int i = 0; i < numOutliers; i++) {
            // All points with LDOF <= 0.5 are not considered outliers, 
            // regardless of the specified outlier ratio.
            if (ldofValuesSorted[i] > 0.5f) {
                maxOutlierScore = Math.max(maxOutlierScore,
                        ldofValuesSorted[i]);
                outlierScores.add(ldofValuesSorted[i]);
                outlierIndexes.add(reArr[i]);
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
