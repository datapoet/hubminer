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
package linear;

/**
 * Implements a basic linear operator.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class LinearOperator {

    double[][] mat;

    /**
     * @param mat Matrix defining the operator.
     */
    public LinearOperator(double[][] mat) {
        this.mat = mat;
    }

    /**
     * @param mat Matrix defining the operator.
     */
    public LinearOperator(float[][] mat) {
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
     * @return Matrix defining the operator.
     */
    public double[][] getOperatorMatrix() {
        return mat;
    }

    /**
     * Construct a transpose of the current operator.
     *
     * @return Transposed operator.
     */
    public LinearOperator constructTranspose() {
        if (mat == null) {
            return null;
        }
        double[][] res = new double[mat.length][mat[0].length];
        for (int i = 0; i < res.length; i++) {
            for (int j = 0; j < res.length; j++) {
                res[i][j] = mat[j][i];
            }
        }
        return new LinearOperator(res);
    }

    /**
     * Left multiply s vector.
     *
     * @param vect Vector given as a float array.
     * @return Multiplication result, a vector given as a float array.
     */
    public float[] leftMultiplyVector(float[] vect) {
        if (mat != null && mat[0] != null && mat[0].length == vect.length) {
            float[] res = new float[mat.length];
            for (int i = 0; i < mat.length; i++) {
                for (int j = 0; j < vect.length; j++) {
                    res[i] += mat[i][j] * vect[j];
                }
            }
            return res;
        } else {
            return null;
        }
    }

    /**
     * Right multiply a vector.
     *
     * @param vect Vector given as a float array.
     * @return Multiplication result, a vector given as a float array.
     */
    public float[] rightMultiplyVectorTranspose(float[] vect) {
        if (mat != null && mat[0] != null && mat.length == vect.length) {
            float[] res = new float[mat[0].length];
            for (int i = 0; i < mat[0].length; i++) {
                for (int j = 0; j < vect.length; j++) {
                    res[i] += mat[j][i] * vect[j];
                }
            }
            return res;
        } else {
            return null;
        }
    }

    /**
     * Multiply two linear operators.
     *
     * @param firstOperator Linear operator.
     * @param secondOperator Linear operator.
     * @return Linear operator that is the multiplication result.
     */
    public static LinearOperator multiply(LinearOperator firstOperator,
            LinearOperator secondOperator) {
        if (firstOperator != null && secondOperator != null &&
                firstOperator.mat != null && secondOperator.mat != null
                & firstOperator.mat[0].length == secondOperator.mat.length) {
            double[][] res = new double[firstOperator.mat.length][
                    secondOperator.mat[0].length];
            for (int i = 0; i < firstOperator.mat.length; i++) {
                for (int j = 0; j < secondOperator.mat[0].length; j++) {
                    for (int k = 0; k < firstOperator.mat[0].length; k++) {
                        res[i][j] += firstOperator.mat[i][k] *
                                secondOperator.mat[k][j];
                    }
                }
            }
            LinearOperator resultingOperator = new LinearOperator(res);
            return resultingOperator;
        } else {
            return null;
        }
    }
}
