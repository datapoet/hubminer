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

/**
 * This interface declares the methods used on the data matrix object.
 * 
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public interface DataMatrixInterface {


    /**
     * @return The number of rows in the matrix. 
     */
    public int numberOfRows();


    /**
     * @return The number of columns in the matrix. 
     */
    public int numberOfColumns();


    /**
     * Sets the element in the matrix to a specified value.
     * @param row Row index.
     * @param column Column index.
     * @param value Value to set the element to.
     */
    public void setElementAt(int row, int column, float value);


    /**
     * @param row Row index.
     * @param column Column index.
     * @return Value at the specified position.
     */
    public float getElementAt(int row, int column);


    /**
     * Sets the entire row to a specified vector of values.
     * @param row Row index.
     * @param values A float array of values to set the matrix row to.
     * @throws Exception 
     */
    public void setRow(int row, float[] values) throws Exception;


    /**
     * Sets the entire column to a specified vector of values.
     * @param row Column index.
     * @param values A float array of values to set the matrix column to.
     * @throws Exception 
     */
    public void setColumn(int column, float[] values) throws Exception;


    /**
     * @param row Row index.
     * @return Row values in a float array.
     */
    public float[] getRow(int row);


    /**
     * @param column Column index.
     * @return Column values in a float array.
     */
    public float[] getColumn(int column);


    /**
     * @return True if the number of rows equals the number of columns, false
     * otherwise.
     */
    public boolean isSquare();


    /**
     * This method does not in fact check for symmetry in the specific matrix
     * but rather returns an indicator of whether the specific implementation
     * enforces that matrix be symmetric or not. Some of the implementations
     * that do also save space by only holding one copy of m(i,j) and m(j,i).
     * @return True, if for each i,j it holds that m(i,j) = m(j,i).
     */
    public boolean isSymmetricMatrixImplementation();


    /**
     * Matrix addition.
     * @param matrix Matrix to add to the current matrix.
     * @return Addition result.
     * @throws Exception 
     */
    public DataMatrixInterface plus(DataMatrixInterface matrix)
            throws Exception;


    /**
     * Matrix subtraction.
     * @param matrix Matrix to subtract from the current matrix.
     * @return Subtraction result.
     * @throws Exception 
     */
    public DataMatrixInterface minus(DataMatrixInterface matrix)
            throws Exception;


    /**
     * Matrix multiplication.
     * @param matrix Matrix to right-multiply with the current matrix.
     * @return Multiplication result.
     * @throws Exception 
     */
    public DataMatrixInterface multiply(DataMatrixInterface matrix)
            throws Exception;


    /**
     * Scalar multiplication.
     * @param scalar Value to multiply all the cells in the matrix with.
     * @return Scalar multiplication result.
     */
    public DataMatrixInterface multiply(float scalar);


    /**
     * Matrix inversion.
     * @return Matrix inverse, if it exists.
     * @throws Exception 
     */
    public DataMatrixInterface calculateInverse() throws Exception;


    /**
     * @return Matrix determinant.
     */
    public float calculateDeterminant();
}
