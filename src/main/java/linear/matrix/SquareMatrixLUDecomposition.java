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

import combinatorial.Permutation;
import data.representation.util.DataMineConstants;
import java.util.Arrays;

/**
 * LU decomposition of square matrices.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class SquareMatrixLUDecomposition {

    private float[][] matrix;
    // Both L and U matrices are encoded here in LUmat, for convenience.
    private float[][] LUmat;
    private int[] perm;
    // Perm is a row permutation 'matrix' - it is encoded in this vector perm 
    // that permutes the rows of matrix during the LU decomposition. This means
    // that PA = LU. Therefore, if invA is sought, invLU must be permuted by P
    // from the right (the columns).
    private boolean decompositionFinished = false;
    private int rank;

    /**
     * The constructor.
     *
     * @param matrix Matrix to decompose.
     */
    public SquareMatrixLUDecomposition(float[][] matrix) {
        this.matrix = matrix;
        if (matrix != null && matrix.length > 0) {
            LUmat = new float[matrix.length][matrix.length];
            // Initializes to a copy of the original.
            for (int i = 0; i < matrix.length; i++) {
                for (int j = 0; j < matrix.length; j++) {
                    LUmat[i][j] = matrix[i][j];
                }
            }
            perm = new int[matrix.length];
            for (int i = 0; i < matrix.length; i++) {
                perm[i] = i;
            }
        }
    }

    /**
     * @return Matrix rank after the LU composition has been performed to
     * determine it.
     */
    public int getRank() {
        return rank;
    }

    /**
     * @return Matrix inverse.
     */
    public float[][] getMatrixInverse() {
        if (decompositionFinished) {
            double det = calculateDeterminant();
            if (det == 0) {
                // Maybe do something else here.
                return null;
            }
            int len = matrix.length;
            float[] z = new float[len];
            float[] x = new float[len];
            float sum;
            float[][] inverse = new float[len][len];
            for (int i = 0; i < len; i++) {
                // Calculate the i-th column of the inverse in two steps.
                Arrays.fill(z, 0);
                Arrays.fill(x, 0);
                for (int j = 0; j < len; j++) {
                    // Lz = e Ux = z
                    sum = 0;
                    for (int k = 0; k < j; k++) {
                        sum += LUmat[j][k] * z[k];
                    }
                    z[j] = (j == i) ? 1 - sum : 0 - sum;
                }
                for (int j = 0; j < len; j++) {// Lz = e Ux = z
                    sum = 0;
                    for (int k = j + 1; k < len; k++) {
                        sum += LUmat[j][k] * x[k];
                    }
                    x[j] = (z[j] - sum) / LUmat[j][j];
                }
                for (int j = 0; j < len; j++) {
                    inverse[j][i] = x[j];
                }
            }
            // Now permute if necessary.
            if (!Permutation.isIdentity(perm)) {
                inverse = Permutation.permuteSquareMatrixColumns(inverse, perm);
            }
            return inverse;
        } else {
            performLUdecomposition();
            return getMatrixInverse();
        }
    }

    /**
     * @return Value of the matrix determinant.
     */
    public double calculateDeterminant() {
        if (decompositionFinished) {
            double result = 1;
            for (int i = 0; i < matrix.length; i++) {
                result *= LUmat[i][i];
            }
            if (Permutation.isOddPermutation(perm)) {
                result *= -1;
            }
            return result;
        } else {
            performLUdecomposition();
            return calculateDeterminant();
        }
    }

    /**
     * @return Permutation.
     */
    public int[] getPerm() {
        return perm;
    }

    /**
     * @return The original matrix.
     */
    public float[][] getOriginalMatrix() {
        return matrix;
    }

    /**
     * @return The calculated LU matrix.
     */
    public float[][] getLUMatrix() {
        return LUmat;
    }

    /**
     * @return The calculated L matrix.
     */
    public float[][] getFullLMatrix() {
        float[][] lMat = new float[matrix.length][matrix.length];
        for (int i = 0; i < matrix.length; i++) {
            lMat[i][i] = 1;
            for (int j = i + 1; j < matrix.length; j++) {
                lMat[j][i] = LUmat[j][i];
            }
        }
        return lMat;
    }

    /**
     * @return The calculated U matrix.
     */
    public float[][] getFullUMatrix() {
        float[][] uMat = new float[matrix.length][matrix.length];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = i; j < matrix.length; j++) {
                uMat[i][j] = LUmat[i][j];
            }
        }
        return uMat;
    }

    /**
     * Returns the index of the first non-zero row, by looking below the
     * diagonal, for a given column index.
     *
     * @param column Column index.
     * @return
     */
    private int getFirstNonZeroRowIndex(int column) {
        int row = column;
        while (row < LUmat.length
                && DataMineConstants.isZero(LUmat[row][column])) {
            row++;
        }
        if (row == LUmat.length) {
            return -1;
        } else {
            return row;
        }
    }

    /**
     * Perform LU decomposition.
     */
    public void performLUdecomposition() {
        int len = matrix.length;
        float quotient;
        float[] tempRow;
        int nextNonz;
        for (int i = 0; i < len - 1; i++) {
            // len-1, since there is nothing below the main diagonal in the
            // last column. Zero out everything below the main diagonal in the
            // i-th column.
            nextNonz = getFirstNonZeroRowIndex(i);
            if (nextNonz == -1) {
                // Nothing to do, zero diagonal element.
                continue;
            }
            if (nextNonz != i) {
                // Switch rows.
                tempRow = LUmat[i];
                LUmat[i] = LUmat[nextNonz];
                LUmat[nextNonz] = tempRow;
                perm[i] = nextNonz;
                perm[nextNonz] = i;
            }
            for (int j = i + 1; j < len; j++) {
                // j is the row index.
                quotient = -LUmat[j][i] / LUmat[i][i];
                for (int k = i + 1; k < len; k++) {
                    LUmat[j][k] += quotient * LUmat[i][k];
                }
                // Now just fill the zeroed element with its L-matrix value,
                // which turns out to be: -quotient
                LUmat[j][i] = -quotient;
            }
        }
        int numNonZeroRows = 0;
        for (int i = 0; i < len; i++) {
            for (int j = i; j < len; j++) {
                if (DataMineConstants.isNonZero(LUmat[i][j])) {
                    numNonZeroRows++;
                    break;
                }
            }
        }
        rank = numNonZeroRows;
        decompositionFinished = true;
    }
}
