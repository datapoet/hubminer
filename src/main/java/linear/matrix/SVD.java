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
package linear.matrix;

import Jama.Matrix;
import Jama.SingularValueDecomposition;
import data.representation.util.DataMineConstants;
import linear.LinearOperator;

/**
 * This class uses Jama for performing SVD and also implements the matrix
 * pseudo-inverse and pseudo-determinant operations. M = U Sigma Vtranspose U
 * and V rotational matrices and Sigma the non-negative diagonal scaling matrix
 * .U has dimensions m x m, V has dimensions n x n, Sigma has dimensions m x n.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class SVD {

    double[][] mat;
    double[][] U;
    double[][] V;
    double[] Sigma;

    /**
     * @param mat A two-dimensional double array that is the data matrix.
     */
    public SVD(double[][] mat) {
        this.mat = mat;
    }

    /**
     * @param mat A two-dimensional float array that is the data matrix.
     */
    public SVD(float[][] mat) {
        if (mat != null) {
            this.mat = new double[mat.length][mat[0].length];
            for (int i = 0; i < mat.length; i++) {
                for (int j = 0; j < mat[i].length; j++) {
                    this.mat[i][j] = mat[i][j];
                }
            }
        }
    }

    /**
     * Calculates the matrix pseudo-inverse.
     *
     * @return The matrix pseudo-inverse.
     */
    public double[][] getPseudoInverse() {
        double[] sigmaPseudo = new double[Sigma.length];
        int m = U.length;
        int n = V.length;
        for (int i = 0; i < sigmaPseudo.length; i++) {
            if (DataMineConstants.isNonZero(Sigma[i])) {
                sigmaPseudo[i] = 1 / Sigma[i];
            }
        }
        // Sigma pseudo is transposed - has dimensions n x m.
        double[][] sigmaPseudoTimesUtransposed = new double[n][m];
        for (int i = 0; i < Math.min(m, n); i++) {
            for (int j = 0; j < m; j++) {
                sigmaPseudoTimesUtransposed[i][j] = sigmaPseudo[i] * U[j][i];
            }
        }
        LinearOperator Vop = new LinearOperator(V);
        LinearOperator temp = new LinearOperator(sigmaPseudoTimesUtransposed);
        LinearOperator result = LinearOperator.multiply(Vop, temp);
        return result.getOperatorMatrix();
    }

    /**
     * Calculates the matrix pseudo-determinant.
     *
     * @return Matrix pseudo-determinant.
     */
    public double getPseudoDet() {
        double pseudoDet = 1;
        for (int i = 0; i < Sigma.length; i++) {
            if (DataMineConstants.isNonZero(Sigma[i])) {
                pseudoDet *= 1 / Sigma[i];
            }
        }
        return pseudoDet;
    }

    /**
     * Perform SVD on the provided matrix.
     */
    public void decomposeMatrix() {
        Matrix A = new Matrix(mat);
        SingularValueDecomposition svd = A.svd();
        U = svd.getU().getArray();
        V = svd.getV().getArray();
        Sigma = svd.getSingularValues();

    }

    /**
     * @return U matrix in M = U Sigma Vtranspose.
     */
    public double[][] getU() {
        return U;
    }

    /**
     * @return V matrix in M = U Sigma Vtranspose.
     */
    public double[][] getV() {
        return V;
    }

    /**
     * @return The diagonal sigma elements.
     */
    public double[] getSigmaDiagonal() {
        return Sigma;
    }
}
