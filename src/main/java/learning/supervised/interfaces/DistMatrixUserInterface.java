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
package learning.supervised.interfaces;

/**
 * This interface declares the getters and setters for the upper triangular
 * float distance matrix. It can be used by classifiers and clusterers to
 * request a distance matrix from the evaluation environment, which calculates
 * it once for all the algorithms, if distance matrix users are present.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public interface DistMatrixUserInterface {

    /**
     * Distance matrix setter.
     *
     * @param distMatrix float[][] representing the upper triangular distance
     * matrix. Each row i contains only entries for j > i, due to distances
     * being symmetric. For compression, the zeroed lower triangular entries are
     * not stored, so each row i contains (size - i - 1) elements. The entry in
     * distMatrix[i][j] corresponds to the distance between data points i and i
     * + j + 1.
     */
    public void setDistMatrix(float[][] distMatrix);

    /**
     * Distance matrix getter.
     *
     * @return float[][] representing the upper triangular distance matrix. Each
     * row i contains only entries for j > i, due to distances being symmetric.
     * For compression, the zeroed lower triangular entries are not stored, so
     * each row i contains (size - i - 1) elements. The entry in
     * distMatrix[i][j] corresponds to the distance between data points i and i
     * + j + 1.
     */
    public float[][] getDistMatrix();
}
