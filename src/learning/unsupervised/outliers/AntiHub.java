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
import data.neighbors.NeighborSetFinder;
import data.representation.DataSet;
import distances.primary.CombinedMetric;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

/**
 * This class implements the AntiHub outlier detection method proposed in the 
 * paper titled: "Reverse Nearest Neighbors in Unsupervised Distance-Based 
 * Outlier Detection" by Milos Radovanovic et al., that was published in IEEE 
 * Transactions on Knowledge and Data Engineering (TKDE) in 2014.
 * 
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class AntiHub extends OutlierDetector implements NSFUserInterface {
    
    // The parameter used for summing up the neighbor occurrence frequencies of
    // neighbor points for anti-hub estimation. It is automatically determined
    // within the method, so it does not need to be set manually by the users.
    private float alpha;
    public static final float DEFAULT_ALPHA_STEP = 0.05f;
    public static final float DEFAULT_OUTLIER_RATIO = 0.05f;
    // Step size while searching for the optimal alpha value.
    private float alphaStep = DEFAULT_ALPHA_STEP;
    // The object used for metric calculations, if necessary.
    private CombinedMetric cmet = CombinedMetric.FLOAT_EUCLIDEAN;
    // Distance matrix, if available.
    private float[][] dMat;
    // The object to calculate and/or hold the kNN sets.
    private NeighborSetFinder nsf;
    // Neighborhood size.
    private int k;
    private float outlierRatio = DEFAULT_OUTLIER_RATIO;
    
    /**
     * Default empty constructor.
     */
    public AntiHub() {
    }
    
    /**
     * Initialization.
     * 
     * @param dset DataSet object to find outliers for.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Integer that is the neighborhood size.
     */
    public AntiHub(DataSet dset, CombinedMetric cmet, int k) {
        setDataSet(dset);
        this.cmet = cmet;
        this.k = k;
    }
    
    /**
     * Initialization.
     * 
     * @param dset DataSet object to find outliers for.
     * @param dMat float[][] that is the upper triangular distance matrix.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Integer that is the neighborhood size.
     */
    public AntiHub(DataSet dset, float[][] dMat, CombinedMetric cmet,
            int k) {
        setDataSet(dset);
        this.cmet = cmet;
        this.k = k;
        this.dMat = dMat;
    }
    
    /**
     * Initialization.
     * 
     * @param dset DataSet object to find outliers for.
     * @param nsf NeighborSetFinder object that holds the calculated kNN sets.
     * @param k Integer that is the neighborhood size.
     */
    public AntiHub(DataSet dset, NeighborSetFinder nsf, int k) {
        setDataSet(dset);
        this.k = k;
        if (nsf != null) {
            this.cmet = nsf.getCombinedMetric();
            this.nsf = nsf;
        }
    }
    
    /**
     * Initialization.
     * 
     * @param dset DataSet object to find outliers for.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Integer that is the neighborhood size.
     * @param outlierRatio Float value that it the outlier ratio to use, the
     * proportion of points to select as outliers.
     */
    public AntiHub(DataSet dset, CombinedMetric cmet, int k,
            float outlierRatio) {
        setDataSet(dset);
        this.cmet = cmet;
        this.k = k;
        this.outlierRatio = outlierRatio;
    }
    
    /**
     * Initialization.
     * 
     * @param dset DataSet object to find outliers for.
     * @param dMat float[][] that is the upper triangular distance matrix.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Integer that is the neighborhood size.
     * @param outlierRatio Float value that it the outlier ratio to use, the
     * proportion of points to select as outliers.
     */
    public AntiHub(DataSet dset, float[][] dMat, CombinedMetric cmet,
            int k, float outlierRatio) {
        setDataSet(dset);
        this.cmet = cmet;
        this.k = k;
        this.dMat = dMat;
        this.outlierRatio = outlierRatio;
    }
    
    /**
     * Initialization.
     * 
     * @param dset DataSet object to find outliers for.
     * @param nsf NeighborSetFinder object that holds the calculated kNN sets.
     * @param k Integer that is the neighborhood size.
     * @param outlierRatio Float value that it the outlier ratio to use, the
     * proportion of points to select as outliers.
     */
    public AntiHub(DataSet dset, NeighborSetFinder nsf, int k,
            float outlierRatio) {
        setDataSet(dset);
        this.k = k;
        if (nsf != null) {
            this.cmet = nsf.getCombinedMetric();
            this.nsf = nsf;
        }
        this.outlierRatio = outlierRatio;
    }
    
    /**
     * @param outlierRatio Float value that it the outlier ratio to use, the
     * proportion of points to select as outliers.
     */
    public void setOutlierRatio(float outlierRatio) {
        this.outlierRatio = outlierRatio;
    }
    
    /**
     * @param alphaStep Float value that is the step to use in parameter search. 
     */
    public void setAlphaStep(float alphaStep) {
        this.alphaStep = alphaStep;
    }
    
    /**
     * @return Float value that is the employed alpha parameter for combining
     * neighbor occurrence frequency with the neighbors' neighbor occurrence
     * frequencies when reaching the final AntiHub outlier score.
     */
    public float getAlpha() {
        return alpha;
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
            return;
        }
        if (nsf == null) {
            // If the kNN sets have not been provided, calculate them here.
            if (dMat == null) {
                dMat = dset.calculateDistMatrix(cmet);
            }
            nsf = new NeighborSetFinder(dset, dMat, cmet);
            nsf.calculateNeighborSets(k);
        }
        if (nsf.getCurrK() < k) {
            // Re-calculate the kNN sets, since they are incompatible with the
            // specified neighborhood size.
            if (dMat == null) {
                dMat = nsf.getDistances();
            }
            if (cmet == null) {
                cmet = nsf.getCombinedMetric();
            }
            NeighborSetFinder nsfLargerK = new NeighborSetFinder(dset, dMat,
                    cmet);
            nsfLargerK.calculateNeighborSets(k);
            nsf = nsfLargerK;
        } else if (nsf.getCurrK() > k) {
            // Sub-sample the kNN sets, since they are incompatible with the
            // specified neighborhood size.
            nsf = nsf.getSubNSF(k);
        }
        int size = dset.size();
        // Get the kNN sets and the neighbor occurrence frequencies.
        float[] occFreqs = nsf.getFloatOccFreqs();
        int[][] kNeighbors = nsf.getKNeighbors();
        // Calculate the neighbor occurrence frequency sums across all 
        // neighborhoods.
        float[] neighborhoodOccSums = new float[size];
        for (int i = 0; i < size; i++) {
            for (int kInd = 0; kInd < k; kInd++) {
                neighborhoodOccSums[i] += occFreqs[kNeighbors[i][kInd]];
            }
        }
        float[] currAlphaAHScores;
        float[] bestAlphaAHScores = null;
        float bestAlpha = 0;
        int bestNumDistinct = 0;
        int numDistinct;
        float outlierThreshold;
        float bestOutlierThreshold = 0;
        HashMap<Float, Integer> scoreMap;
        // Search through various possible alpha values for making a convex
        // combination of the point-wise neighbor occurrence frequencies and
        // the total neighborhood occurrence frequency sums.
        for (alpha = 0f; alpha <= 1f; alpha += alphaStep) {
            // The hash map is used for tracking distinct values, as this is a
            // quality criterion.
            scoreMap = new HashMap<>(size);
            currAlphaAHScores = Arrays.copyOf(occFreqs, size);
            // Calculate the scores according to the current alpha value.
            for (int i = 0; i < size; i++) {
                currAlphaAHScores[i] = (1 - alpha) * currAlphaAHScores[i] +
                        alpha * neighborhoodOccSums[i];
            }
            // Find the limit value based on the current outlierRatio.
            float[] scoresCopy = Arrays.copyOf(currAlphaAHScores, size);
            Arrays.sort(scoresCopy);
            int outlierLimitIndex = (int)(Math.min(
                    Math.ceil(outlierRatio * size), size)) - 1;
            outlierThreshold = scoresCopy[outlierLimitIndex];
            // Calculate the number of distinct values among the outliers.
            for (int i = 0; i < size; i++) {
                if (currAlphaAHScores[i] <= outlierThreshold) {
                    if (!scoreMap.containsKey(currAlphaAHScores[i])) {
                        scoreMap.put(currAlphaAHScores[i], 1);
                    } else {
                        scoreMap.put(currAlphaAHScores[i],
                                scoreMap.get(currAlphaAHScores[i]) + 1);
                    }
                }
            }
            numDistinct = scoreMap.keySet().size();
            // If it exceeds the current best, update the best parameter values.
            if (numDistinct > bestNumDistinct) {
                bestNumDistinct = numDistinct;
                bestAlpha = alpha;
                bestAlphaAHScores = currAlphaAHScores;
                bestOutlierThreshold = outlierThreshold;
            }
        }
        alpha = bestAlpha;
        ArrayList<Float> outlierScores = new ArrayList<>(size);
        ArrayList<Integer> outlierIndexes = new ArrayList<>(size);
        float maxOutlierScore = 0;
        float minOutlierScore = Float.MAX_VALUE;
        // Form an outlier list and a list of the associated outlier scores.
        for (int i = 0; i < size; i++) {
            if (bestAlphaAHScores[i] <= bestOutlierThreshold) {
                outlierIndexes.add(i);
                outlierScores.add(bestAlphaAHScores[i]);
                if (bestAlphaAHScores[i] > maxOutlierScore) {
                    maxOutlierScore = bestAlphaAHScores[i];
                }
                if (bestAlphaAHScores[i] < minOutlierScore) {
                    minOutlierScore = bestAlphaAHScores[i];
                }
            }
        }
        // Normalize. Also, transform the scores so that now the higher scores
        // correspond to more likely outliers.
        if (maxOutlierScore > 0) {
            for (int j = 0; j < outlierScores.size(); j++) {
                outlierScores.set(j, 1 - ((outlierScores.get(j) -
                        minOutlierScore) / (maxOutlierScore -
                        minOutlierScore)));
            }
        }
        setOutlierIndexes(outlierIndexes, outlierScores);
    } 
}
