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
package probability;

import Jama.LUDecomposition;
import Jama.Matrix;
import data.representation.DataSet;
import learning.unsupervised.Cluster;
import linear.LinearOperator;
import linear.matrix.SVD;
import statistics.CovarianceFinder;

/**
 * This class implements a simple Gaussian model of a data distribution.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class GaussianModel {

    int dim = 0;
    float[] mean;
    float[][] covariance;
    double[][] covInverse;
    double det;
    double[][] pseudoInverse;
    double pseudoDet;
    LinearOperator exOp = null;
    double normalizer = 1;
    int rank;
    Cluster sample;
    // If it is calculated from a sample.

    /**
     * @param mean The mean of the distribution.
     * @param covariance Covariance matrix.
     */
    public GaussianModel(float[] mean, float[][] covariance) {
        this.mean = mean;
        this.covariance = covariance;
        if (mean != null) {
            dim = mean.length;
        }
    }

    /**
     * @param sample Cluster to calculate the model from.
     */
    public GaussianModel(Cluster sample) {
        this.sample = sample;
        if (sample.size() > 0) {
            CovarianceFinder cf = new CovarianceFinder(sample.intoDataSet());
            covariance = cf.calculateFloatCovariance();
            mean = cf.getFloatMeans();
            if (mean != null) {
                dim = mean.length;
            }
        }
    }

    /**
     * @param sample DataSet to calculate the model from.
     */
    public GaussianModel(DataSet sample) {
        this(sample.makeClusterObject());
    }

    public void calculatePredictiveModel() {
        if (sample == null || sample.isEmpty()) {
            return;
        }
        // This is a bit awkward, but we need to pass a double matrix instead of
        // a float matrix.
        double[][] covDoub = new double[covariance.length][covariance.length];
        for (int i = 0; i < covariance.length; i++) {
            for (int j = 0; j < covariance.length; j++) {
                covDoub[i][j] = covariance[i][j];
            }
        }
        Matrix covMatrix = new Matrix(covDoub);
        LUDecomposition lud = new LUDecomposition(covMatrix);
        rank = covMatrix.rank();
        if (lud.isNonsingular()) {
            // It is possible to calculate the inverse of the covariance matrix.
            Matrix covInvMat = covMatrix.inverse();
            covInverse = covInvMat.getArray();
            det = covInvMat.det();
            exOp = new LinearOperator(covInverse);
            normalizer = 1 / (Math.sqrt(Math.abs(det))
                    * Math.pow((Math.PI * 2), dim / 2));
        } else {
            // We have to use the pseudoinverse covariance matrix.
            covInverse = null;
            SVD svd = new SVD(covariance);
            svd.decomposeMatrix();
            pseudoInverse = svd.getPseudoInverse();
            svd.decomposeMatrix();
            pseudoDet = svd.getPseudoDet();
            exOp = new LinearOperator(pseudoInverse);
            normalizer = 1 / Math.sqrt(Math.abs(
                    Math.pow(Math.PI * 2, rank) * pseudoDet));
        }
    }

    /**
     * Calculate a probability of x if it originated from the model.
     *
     * @param x A float array representing a feature vector.
     * @return The probability of the observed feature vector.
     */
    public double getProbability(float[] x) {
        if (sample == null || sample.isEmpty()) {
            return 0;
        }
        if (x.length == dim) {
            float[] xminusMean = new float[x.length];
            for (int i = 0; i < x.length; i++) {
                xminusMean[i] = x[i] - mean[i];
            }
            float[] invXFromLeftMultiplied =
                    exOp.leftMultiplyVector(xminusMean);
            float expScalar = 0;
            for (int i = 0; i < invXFromLeftMultiplied.length; i++) {
                expScalar += xminusMean[i] * invXFromLeftMultiplied[i];
            }
            expScalar = -0.5f * expScalar;
            return normalizer * Math.exp(expScalar);
        } else {
            return 0;
        }
    }
}
