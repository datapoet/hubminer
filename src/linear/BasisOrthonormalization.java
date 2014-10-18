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
 * Gram-Schmidt vector space basis orthonormalization.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class BasisOrthonormalization {

    /**
     * Takes the vector space basis as a two-dimensional float array (matrix)
     * and produces an orthogonal, normalized basis.
     *
     * @param basis Original vector space basis.
     * @return Two-dimensional float array representing the orthonormalized
     * vector space basis.
     */
    public static float[][] orthonormalize(float[][] basis) {
        float[][] newBasis = new float[basis.length][basis[0].length];
        newBasis[0] = basis[0];
        float fact;
        float[] hNorms = new float[basis.length];
        for (int k = 0; k < basis[0].length; k++) {
            hNorms[0] += newBasis[0][k] * newBasis[0][k];
        }
        for (int k = 0; k < basis[0].length; k++) {
            newBasis[0][k] /= hNorms[0];
        }
        for (int i = 1; i < basis.length; i++) {
            newBasis[i] = basis[i];
            for (int j = 0; j < i; j++) {
                fact = 0f;
                for (int k = 0; k < basis[i].length; k++) {
                    fact += basis[i][k] * newBasis[j][k];
                }
                for (int k = 0; k < basis[i].length; k++) {
                    newBasis[i][k] -= fact * newBasis[j][k];
                }
            }
            for (int k = 0; k < basis[i].length; k++) {
                hNorms[i] += newBasis[i][k] * newBasis[i][k];
            }
            for (int k = 0; k < basis[0].length; k++) {
                newBasis[i][k] /= hNorms[i];
            }
        }
        return newBasis;
    }

    /**
     * Takes the vector space basis as a two-dimensional float array (matrix)
     * and produces an orthogonal, normalized basis.
     *
     * @param basis Original vector space basis.
     * @param currBasisSize Size of the current basis.
     * @return Two-dimensional float array representing the orthonormalized
     * vector space basis.
     */
    public static float[][] orthonormalize(float[][] basis,
            int currBasisSize) {
        float[][] newBasis = new float[currBasisSize][basis[0].length];
        newBasis[0] = basis[0];
        float fact;
        float[] hNorms = new float[currBasisSize];
        for (int k = 0; k < basis[0].length; k++) {
            hNorms[0] += newBasis[0][k] * newBasis[0][k];
        }
        for (int k = 0; k < basis[0].length; k++) {
            newBasis[0][k] /= hNorms[0];
        }
        for (int i = 1; i < currBasisSize; i++) {
            newBasis[i] = basis[i];
            for (int j = 0; j < i; j++) {
                fact = 0f;
                for (int k = 0; k < basis[i].length; k++) {
                    fact += basis[i][k] * newBasis[j][k];
                }
                for (int k = 0; k < basis[i].length; k++) {
                    newBasis[i][k] -= fact * newBasis[j][k];
                }
            }
            for (int k = 0; k < basis[i].length; k++) {
                hNorms[i] += newBasis[i][k] * newBasis[i][k];
            }
            for (int k = 0; k < basis[0].length; k++) {
                newBasis[i][k] /= hNorms[i];
            }
        }
        return newBasis;
    }

    /**
     * Takes the vector space basis as a two-dimensional double array (matrix)
     * and produces an orthogonal, normalized basis.
     *
     * @param basis Original vector space basis.
     * @return Two-dimensional double array representing the orthonormalized
     * vector space basis.
     */
    public static double[][] orthonormalize(double[][] basis) {
        double[][] newBasis = new double[basis.length][basis[0].length];
        newBasis[0] = basis[0];
        double fact;
        double[] hNorms = new double[basis.length];
        for (int k = 0; k < basis[0].length; k++) {
            hNorms[0] += newBasis[0][k] * newBasis[0][k];
        }
        for (int k = 0; k < basis[0].length; k++) {
            newBasis[0][k] /= hNorms[0];
        }
        for (int i = 1; i < basis.length; i++) {
            newBasis[i] = basis[i];
            for (int j = 0; j < i; j++) {
                fact = 0f;
                for (int k = 0; k < basis[i].length; k++) {
                    fact += basis[i][k] * newBasis[j][k];
                }
                for (int k = 0; k < basis[i].length; k++) {
                    newBasis[i][k] -= fact * newBasis[j][k];
                }
            }
            for (int k = 0; k < basis[i].length; k++) {
                hNorms[i] += newBasis[i][k] * newBasis[i][k];
            }
            for (int k = 0; k < basis[0].length; k++) {
                newBasis[i][k] /= hNorms[i];
            }
        }
        return newBasis;
    }

    /**
     * Takes the vector space basis as a two-dimensional double array (matrix)
     * and produces an orthogonal, normalized basis.
     *
     * @param basis Original vector space basis.
     * @param currBasisSize Size of the current basis.
     * @return Two-dimensional double array representing the orthonormalized
     * vector space basis.
     */
    public static double[][] orthonormalize(double[][] basis,
            int currBasisSize) {
        double[][] newBasis = new double[currBasisSize][basis[0].length];
        newBasis[0] = basis[0];
        double fact;
        double[] hNorms = new double[currBasisSize];
        for (int k = 0; k < basis[0].length; k++) {
            hNorms[0] += newBasis[0][k] * newBasis[0][k];
        }
        for (int k = 0; k < basis[0].length; k++) {
            newBasis[0][k] /= hNorms[0];
        }
        for (int i = 1; i < currBasisSize; i++) {
            newBasis[i] = basis[i];
            for (int j = 0; j < i; j++) {
                fact = 0f;
                for (int k = 0; k < basis[i].length; k++) {
                    fact += basis[i][k] * newBasis[j][k];
                }
                for (int k = 0; k < basis[i].length; k++) {
                    newBasis[i][k] -= fact * newBasis[j][k];
                }
            }
            for (int k = 0; k < basis[i].length; k++) {
                hNorms[i] += newBasis[i][k] * newBasis[i][k];
            }
            for (int k = 0; k < basis[0].length; k++) {
                newBasis[i][k] /= hNorms[i];
            }
        }
        return newBasis;
    }
}
