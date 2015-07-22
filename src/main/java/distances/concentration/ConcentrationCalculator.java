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
package distances.concentration;

import data.representation.DataSet;
import distances.primary.CombinedMetric;

/**
 * Examine the concentration of distances in high-dimensional feature spaces.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class ConcentrationCalculator {

    private DataSet dataset;
    private float[][] distMatrix;
    private CombinedMetric cmet;
    private boolean distancesCalculated = false;
    private double variance = 0;
    private double mean = 0;
    private double min = Float.MAX_VALUE;
    private double max = -Float.MAX_VALUE;
    private double relativeContrast = 0;
    private double contrast = 0;
    private double relativeVariance;
    // According to Chavez, Navaro, "Searching in Metric Spaces", 2001
    private double intrinsicDim;

    /**
     * @param dataset Dataset.
     * @param distMatrix Distance matrix.
     */
    public ConcentrationCalculator(DataSet dataset, float[][] distMatrix) {
        this.distMatrix = distMatrix;
        this.dataset = dataset;
        distancesCalculated = true;
    }

    /**
     * @param dataset Dataset.
     * @param cmet Metrics object.
     */
    public ConcentrationCalculator(DataSet dataset, CombinedMetric cmet) {
        this.dataset = dataset;
        distancesCalculated = false;
        this.cmet = cmet;
    }

    /**
     * Performs all the calculations.
     *
     * @throws Exception
     */
    public void calculateMeasures() throws Exception {
        if (distancesCalculated) {
            calculateMeanAndVariance();
        } else {
            calculateDistances();
        }
    }

    /**
     * @return Relative contrast between the distances.
     */
    public double getRelativeContrast() {
        return relativeContrast;
    }

    /**
     * @return Absolute contrast between the distances.
     */
    public double getContrast() {
        return contrast;
    }

    /**
     * @return The smallest distance.
     */
    public double getMinDist() {
        return min;
    }

    /**
     * @return The largest distance.
     */
    public double getMaxDist() {
        return max;
    }

    /**
     * @return Relative variance.
     */
    public double getRelativeVariance() {
        return relativeVariance;
    }

    /**
     * @return Average distance.
     */
    public double getMeanDist() {
        return mean;
    }

    /**
     * @return Variance of the distances.
     */
    public double getDistVariance() {
        return variance;
    }

    /**
     * @return An estimate of the intrinsic dimensionality of the data.
     */
    public double getIntrinsicDimensionality() {
        return intrinsicDim;
    }

    /**
     * Calculates the statistics of the distance distribution.
     */
    public void calculateMeanAndVariance() {
        for (int i = 0; i < distMatrix.length; i++) {
            for (int j = 0; j < distMatrix[i].length; j++) {
                if (distMatrix[i][j] > max) {
                    max = distMatrix[i][j];
                }
                if (distMatrix[i][j] < min) {
                    min = distMatrix[i][j];
                }
                mean += distMatrix[i][j];
            }
        }
        // Here we calculate the sample mean.
        mean = 2 * mean / (dataset.size() * (dataset.size() - 1));
        variance = 0; // Now iterate over the matrix and calculate it.
        for (int i = 0; i < distMatrix.length; i++) {
            for (int j = 0; j < distMatrix[i].length; j++) {
                variance += (mean - distMatrix[i][j])
                        * (mean - distMatrix[i][j]);
            }
        }
        // Now we calculate the sample variance.
        variance = 2 * variance / (dataset.size() * (dataset.size() - 1));
        relativeContrast = (max - min) / min;
        contrast = max - min;
        relativeVariance = Math.sqrt(variance) / mean;
        if (relativeVariance > 0) {
            intrinsicDim = 1f / Math.pow(relativeVariance, 2);
        }
    }

    /**
     * Calculate the distance matrix.
     *
     * @throws Exception
     */
    public void calculateDistances() throws Exception {
        if (dataset == null || dataset.isEmpty()) {
            return;
        }
        // The distance matrix is symmetric, so only the upper half is
        // calculated, without the diagonal. In order to save space, each row
        // in the implementation has a different length, the empty fields are
        // ommitted. This is the way these matrices are usually handled within
        // HubMiner.
        distMatrix = new float[dataset.size()][];
        mean = 0;
        for (int i = 0; i < dataset.size(); i++) {
            distMatrix[i] = new float[distMatrix.length - i - 1];
            for (int j = i + 1; j < distMatrix.length; j++) {
                distMatrix[i][j - i - 1] =
                        cmet.dist(dataset.data.get(i), dataset.data.get(j));
                mean += distMatrix[i][j - i - 1];
            }
        }
        mean = 2 * mean / (dataset.size() * (dataset.size() - 1)); //sample mean
        variance = 0; // Now iterate over the matrix and calculate it.
        for (int i = 0; i < distMatrix.length; i++) {
            for (int j = i + 1; j < distMatrix[i].length; j++) {
                if (distMatrix[i][j] > max) {
                    max = distMatrix[i][j];
                }
                if (distMatrix[i][j] < min) {
                    min = distMatrix[i][j];
                }
                variance += (mean - distMatrix[i][j - i - 1])
                        * (mean - distMatrix[i][j - i - 1]);
            }
        }
        // Sample variance.
        variance = 2 * variance / (dataset.size() * (dataset.size() - 1));
        distancesCalculated = true;
        relativeContrast = (max - min) / min;
        contrast = max - min;
        relativeVariance = Math.sqrt(variance) / mean;
        if (relativeVariance > 0) {
            intrinsicDim = 1f / Math.pow(relativeVariance, 2);
        }
    }

    /**
     * Calculates the statistics of the distance distribution, over a part of
     * the distance matrix.
     *
     * @param restriction Part of the distance matrix to analyze.
     * @throws Exception
     */
    public void calculateMeasures(int restriction) throws Exception {
        calculateMeanAndVariance(restriction);
    }

    /**
     * Calculates the statistics of the distance distribution, over a part of
     * the distance matrix.
     *
     * @param restriction Part of the distance matrix to analyze.
     */
    public void calculateMeanAndVariance(int restriction) {
        for (int i = 0; i < restriction; i++) {
            for (int j = 0; j < restriction - i - 1; j++) {
                if (distMatrix[i][j] > max) {
                    max = distMatrix[i][j];
                }
                if (distMatrix[i][j] < min) {
                    min = distMatrix[i][j];
                }
                mean += distMatrix[i][j];
            }
        }
        mean = 2 * mean / (restriction * (restriction - 1)); //sample mean
        variance = 0; //now iterate over the matrix and calculate it
        for (int i = 0; i < distMatrix.length; i++) {
            for (int j = 0; j < restriction - i - 1; j++) {
                variance += (mean - distMatrix[i][j])
                        * (mean - distMatrix[i][j]);
            }
        }
        //sample variance
        variance = 2 * variance / (restriction * (restriction - 1));
        relativeContrast = (max - min) / min;
        contrast = max - min;
        relativeVariance = Math.sqrt(variance) / mean;
        if (relativeVariance > 0) {
            intrinsicDim = 1f / Math.pow(relativeVariance, 2);
        }
    }
}
