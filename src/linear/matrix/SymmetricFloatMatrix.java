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

import ioformat.FileUtil;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;

/**
 * In this variant diagonal elements do exist (they are not equal to zero).
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class SymmetricFloatMatrix implements DataMatrixInterface {

    private float[][] data = null;

    /**
     * @return Two-dimensional float array that is the row representation of the
     * symmetric float matrix.
     */
    public float[][] getMatrix2DArray() {
        return data;
    }

    public SymmetricFloatMatrix() {
    }

    /**
     * @param dim Number of rows and columns.
     */
    public SymmetricFloatMatrix(int dim) {
        data = new float[dim][];
        for (int i = 0; i < dim; i++) {
            data[i] = new float[dim - i];
        }
    }

    /**
     * @param symmetricMatrix Two-dimensional float array that is the row
     * representation of the symmetric float matrix.
     */
    public SymmetricFloatMatrix(float[][] symmetricMatrix) {
        this.data = symmetricMatrix;
    }

    @Override
    public boolean isSymmetricMatrixImplementation() {
        return true;
    }

    @Override
    public boolean isSquare() {
        return true;
    }

    @Override
    public float getElementAt(int row, int col) {
        if (row <= col) {
            return data[row][col - row];
        } else {
            return data[col][row - col];
        }
    }

    @Override
    public void setElementAt(int row, int col, float newValue) {
        if (row <= col) {
            data[row][col - row] = newValue;
        } else {
            data[col][row - col] = newValue;
        }
    }

    @Override
    public int numberOfRows() {
        return data.length;
    }

    @Override
    public int numberOfColumns() {
        return data.length;
    }

    @Override
    public float[] getRow(int row) {
        // Creates an entire row as if the matrix wasn't only half-filled.
        float[] result = new float[data.length];
        for (int i = 0; i < row; i++) {
            result[i] = data[i][row - i];
        }
        for (int i = row; i < data.length; i++) {
            result[i] = data[row][i - row];
        }
        return result;
    }

    @Override
    public float[] getColumn(int col) {
        // Creates an entire column as if the matrix wasn't only half-filled.
        float[] result = new float[data.length];
        for (int i = 0; i < col; i++) {
            result[i] = data[i][col - i];
        }
        for (int i = col; i < data.length; i++) {
            result[i] = data[col][i - col];
        }
        return result;
    }

    @Override
    public void setRow(int row, float[] newValues) throws Exception {
        for (int i = 0; i < row; i++) {
            data[i][row - i] = newValues[i];
        }
        for (int i = row; i < data.length; i++) {
            data[row][i - row] = newValues[i];
        }
    }

    @Override
    public void setColumn(int col, float[] newValues) throws Exception {
        for (int i = 0; i < col; i++) {
            data[i][col - i] = newValues[i];
        }
        for (int i = col; i < data.length; i++) {
            data[col][i - col] = newValues[i];
        }
    }

    @Override
    public DataMatrixInterface calculateInverse() throws Exception {
        DataMatrixInterface result = new DataMatrix(data.length, data.length);
        if (data.length == 1) {
            if (data[0][0] != 0) {
                result.setElementAt(0, 0, 1 / data[0][0]);
            } else {
                return null;
            }
        }
        int[] indexes;
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data.length; j++) {
                indexes = new int[data.length - 1];
                for (int k = 0; k < j; k++) {
                    indexes[k] = k;
                }
                for (int k = j + 1; k < data.length; k++) {
                    indexes[k - 1] = k;
                }
                if (i == 0) {
                    result.setElementAt(i, j, calculateCofactor(indexes, i, 1));
                } else {
                    result.setElementAt(i, j, calculateCofactor(indexes, i, 0));
                }
            }
        }
        // Now calculate determinant from first row minors.
        float detValue = 0;
        for (int i = 0; i < data.length; i++) {
            if (i % 2 == 0) {
                detValue += data[0][i] * result.getElementAt(0, i);
            } else {
                detValue -= data[0][i] * result.getElementAt(0, i);
            }
        }
        if (detValue == 0) {
            return null;
        }
        // Now turn all minors into cofactors and divide by determinant value to
        // get the final result.
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data.length; j++) {
                if (i + j % 2 == 0) {
                    result.setElementAt(i, j,
                            result.getElementAt(i, j) / detValue);
                } else {
                    result.setElementAt(i, j,
                            (-result.getElementAt(i, j)) / detValue);
                }
            }
        }
        return result;
    }

    /**
     * Calculate cofactor within the matrix, recursively.
     *
     * @param indexes A sorted array of indexes.
     * @param indexOfFirstCofactRow
     * @param expansionIndex Index of the intersection element.
     * @return
     * @throws Exception
     */
    private float calculateCofactor(int[] indexes,
            int indexOfFirstCofactRow, int expansionIndex) throws Exception {
        if (indexes.length == 1) {
            if (!(indexOfFirstCofactRow == data.length - 1)) {
                return getElementAt(data.length - 1, indexes[0]);
            } else {
                return getElementAt(data.length - 2, indexes[0]);
            }
        } else if (indexes.length == 2) {
            if (!(indexOfFirstCofactRow == data.length - 1
                    || indexOfFirstCofactRow == data.length - 2)) {
                return (getElementAt(data.length - 2, indexes[0])
                        * getElementAt(data.length - 1, indexes[1])
                        - getElementAt(data.length - 2, indexes[1])
                        * getElementAt(data.length - 1, indexes[0]));
            } else if (indexOfFirstCofactRow == data.length - 1) {
                return (getElementAt(data.length - 3, indexes[0])
                        * getElementAt(data.length - 2, indexes[1])
                        - getElementAt(data.length - 3, indexes[1])
                        * getElementAt(data.length - 2, indexes[0]));
            } else {
                return (getElementAt(data.length - 3, indexes[0])
                        * getElementAt(data.length - 1, indexes[1])
                        - getElementAt(data.length - 3, indexes[1])
                        * getElementAt(data.length - 1, indexes[0]));
            }
        } else {
            float sum = 0;
            int[] tempIndexes;
            for (int i = 0; i < indexes.length; i++) {
                // Make cofactors and expand.
                tempIndexes = new int[indexes.length - 1];
                for (int j = 0; j < i; j++) {
                    tempIndexes[j] = indexes[j];
                }
                for (int j = i + 1; j < indexes.length; j++) {
                    tempIndexes[j - 1] = indexes[j];
                }
                if (i % 2 == 0) {
                    if (getElementAt(expansionIndex, indexes[i]) != 0) {
                        if (expansionIndex + 1 == indexOfFirstCofactRow) {
                            sum += getElementAt(expansionIndex, indexes[i])
                                    * calculateCofactor(tempIndexes,
                                    indexOfFirstCofactRow,
                                    expansionIndex + 2);
                        } else {
                            sum += getElementAt(expansionIndex, indexes[i])
                                    * calculateCofactor(tempIndexes,
                                    indexOfFirstCofactRow,
                                    expansionIndex + 1);
                        }
                    }
                } else {
                    if (getElementAt(expansionIndex, indexes[i]) != 0) {
                        if (expansionIndex + 1 == indexOfFirstCofactRow) {
                            sum -= getElementAt(expansionIndex, indexes[i])
                                    * calculateCofactor(tempIndexes,
                                    indexOfFirstCofactRow,
                                    expansionIndex + 2);
                        } else {
                            sum -= getElementAt(expansionIndex, indexes[i])
                                    * calculateCofactor(tempIndexes,
                                    indexOfFirstCofactRow,
                                    expansionIndex + 1);
                        }
                    }
                }
            }
            return sum;
        }
    }

    @Override
    public DataMatrixInterface multiply(DataMatrixInterface second)
            throws Exception {
        if (!(second.numberOfRows() == data.length)) {
            throw new Exception("Cannot multiply matrices.");
        }
        DataMatrixInterface result;
        if (second.isSymmetricMatrixImplementation()) {
            //The product is not necessarily symmetric.
            result = new DataMatrix(data.length, data.length);
            for (int i = 0; i < data.length; i++) {
                for (int j = 0; j < data.length; j++) {
                    for (int k = 0; k < data.length; k++) {
                        result.setElementAt(i, j, result.getElementAt(i, j)
                                + data[i][k] * second.getElementAt(k, j));
                    }
                }
            }

        } else {
            result = new DataMatrix(data.length, second.numberOfColumns());
            for (int i = 0; i < data.length; i++) {
                for (int j = 0; j < second.numberOfColumns(); j++) {
                    for (int k = 0; k < data.length; k++) {
                        result.setElementAt(i, j, result.getElementAt(i, j)
                                + data[i][k] * second.getElementAt(k, j));
                    }
                }
            }
        }
        return result;
    }

    @Override
    public DataMatrixInterface plus(DataMatrixInterface second) throws Exception {
        if (!(second.numberOfColumns() == data.length
                && second.numberOfRows() == data.length)) {
            throw new Exception("Cannot add matrices of different sizes.");
        }
        DataMatrixInterface result;
        if (second.isSymmetricMatrixImplementation()) {
            result = new SymmetricFloatMatrix(data.length);
            for (int i = 0; i < data.length; i++) {
                for (int j = i; j < data.length; j++) {
                    result.setElementAt(i, j, data[i][j]
                            + second.getElementAt(i, j));
                }
            }
        } else {
            result = new DataMatrix(data.length, data.length);
            for (int i = 0; i < data.length; i++) {
                for (int j = i; j < data.length; j++) {
                    result.setElementAt(i, j, data[i][j]
                            + second.getElementAt(i, j));
                    result.setElementAt(j, i, data[i][j]
                            + second.getElementAt(j, i));
                }
            }
        }
        return result;
    }

    @Override
    public DataMatrixInterface multiply(float scalar) {
        DataMatrixInterface result;
        result = new SymmetricFloatMatrix(data.length);
        for (int i = 0; i < data.length; i++) {
            for (int j = i; j < data.length; j++) {
                result.setElementAt(i, j, data[i][j] * scalar);
            }
        }
        return result;
    }

    @Override
    public DataMatrixInterface minus(DataMatrixInterface second)
            throws Exception {
        if (!(second.numberOfColumns() == data.length
                && second.numberOfRows() == data.length)) {
            throw new Exception("Cannot add matrices of different sizes.");
        }
        DataMatrixInterface result;
        if (second.isSymmetricMatrixImplementation()) {
            result = new SymmetricFloatMatrix(data.length);
            for (int i = 0; i < data.length; i++) {
                for (int j = i; j < data.length; j++) {
                    result.setElementAt(i, j, data[i][j]
                            - second.getElementAt(i, j));
                }
            }
        } else {
            result = new DataMatrix(data.length, data.length);
            for (int i = 0; i < data.length; i++) {
                for (int j = i; j < data.length; j++) {
                    result.setElementAt(i, j, data[i][j]
                            - second.getElementAt(i, j));
                    result.setElementAt(j, i, data[i][j]
                            - second.getElementAt(j, i));
                }
            }
        }
        return result;
    }

    @Override
    public float calculateDeterminant() {
        int[] indexes = new int[data.length];
        for (int i = 0; i < data.length; i++) {
            indexes[i] = i;
        }
        return calculateDet(indexes, 0);
    }

    /**
     * Calculate determinant by cofactor expansion.
     *
     * @param indexes Index array, ordered ASC.
     * @param expansionIndex Index of the intersection element.
     * @return
     */
    private float calculateDet(int[] indexes, int expansionIndex) {
        // This implementation takes advantage of the fact that first row is
        // always expanded upon. This means that when the size of the matrix is
        // 2, it will be reaching the elements in the last two rows.
        if (indexes.length == 1) {
            return getElementAt(data.length - 1, indexes[0]);
        } else if (indexes.length == 2) {
            return (getElementAt(data.length - 2, indexes[0])
                    * getElementAt(data.length - 1, indexes[1])
                    - getElementAt(data.length - 2, indexes[1])
                    * getElementAt(data.length - 1, indexes[0]));
        } else {
            float sum = 0;
            int[] tempInt;
            for (int i = 0; i < indexes.length; i++) {
                //Take the first row, make cofactors and expand.
                tempInt = new int[indexes.length - 1];
                for (int j = 0; j < i; j++) {
                    tempInt[j] = indexes[j];
                }
                for (int j = i + 1; j < indexes.length; j++) {
                    tempInt[j - 1] = indexes[j];
                }
                if (i % 2 == 0) {
                    if (getElementAt(expansionIndex, indexes[i]) != 0) {
                        sum += getElementAt(expansionIndex, indexes[i])
                                * calculateDet(tempInt, expansionIndex + 1);
                    }
                } else {
                    if (getElementAt(expansionIndex, indexes[i]) != 0) {
                        sum -= getElementAt(expansionIndex, indexes[i])
                                * calculateDet(tempInt, expansionIndex + 1);
                    }
                }
            }
            return sum;
        }
    }

    /**
     * Prints the matrix to a file.
     *
     * @param dMatFile File to print the matrix to.
     * @throws Exception
     */
    public void printDMatToFile(File dMatFile) throws Exception {
        FileUtil.createFile(dMatFile);
        try (PrintWriter pw = new PrintWriter(new FileWriter(dMatFile));) {
            pw.println(data.length);
            for (int i = 0; i < data.length - 1; i++) {
                pw.print(data[i][0]);
                for (int j = 1; j < data[i].length; j++) {
                    pw.print("," + data[i][j]);
                }
                pw.println();
            }
        } catch (Exception e) {
            throw e;
        }
    }

    /**
     * Load the matrix from a file.
     *
     * @param dMatFile File to load the matrix from.
     * @throws Exception
     */
    public void loadDMatFromFile(File dMatFile) throws Exception {
        BufferedReader br = new BufferedReader(
                new InputStreamReader(new FileInputStream(dMatFile)));
        float[][] dMatLoaded = null;
        String s;
        String[] lineParse;
        try {
            int size = Integer.parseInt(br.readLine());
            dMatLoaded = new float[size][];
            for (int i = 0; i < size - 1; i++) {
                dMatLoaded[i] = new float[size - i - 1];
                s = br.readLine();
                lineParse = s.split(",");
                for (int j = 0; j < lineParse.length; j++) {
                    dMatLoaded[i][j] = Float.parseFloat(lineParse[j]);
                }
            }
            dMatLoaded[size - 1] = new float[0];
        } catch (IOException | NumberFormatException e) {
            throw e;
        } finally {
            br.close();
        }
        data = dMatLoaded;
    }

    /**
     * Print the matrix to a file without its diagonal elements.
     *
     * @param dMatFile File to print the matrix to.
     * @throws Exception
     */
    public void printDMatToFileNoDiag(File dMatFile) throws Exception {
        FileUtil.createFile(dMatFile);
        try (PrintWriter pw = new PrintWriter(new FileWriter(dMatFile));) {
            pw.println(data.length);
            for (int i = 0; i < data.length - 1; i++) {
                pw.print(data[i][1]);
                for (int j = 2; j < data[i].length; j++) {
                    pw.print("," + data[i][j]);
                }
                pw.println();
            }
        } catch (Exception e) {
            throw e;
        }
    }

    /**
     * Load the matrix from a file where its diagonal elements are not given
     * (they are all equal to zero).
     *
     * @param dMatFile
     * @throws Exception
     * @return Two-dimensional array that is a symmetric float matrix without
     * the diagonal, each row i contains only elements m(i,j) for j > i.
     */
    public float[][] loadDMatFromFileNoDiag(File dMatFile) throws Exception {
        BufferedReader br = new BufferedReader(
                new InputStreamReader(new FileInputStream(dMatFile)));
        float[][] dMatLoaded = null;
        String s;
        String[] lineParse;
        try {
            int size = Integer.parseInt(br.readLine());
            dMatLoaded = new float[size - 1][];
            for (int i = 0; i < size - 1; i++) {
                dMatLoaded[i] = new float[size - i - 1];
                s = br.readLine();
                lineParse = s.split(",");
                for (int j = 0; j
                        < Math.min(lineParse.length,
                        dMatLoaded[i].length); j++) {
                    dMatLoaded[i][j] = Float.parseFloat(lineParse[j]);
                }
            }
        } catch (IOException | NumberFormatException e) {
            throw e;
        } finally {
            br.close();
        }
        return dMatLoaded;
    }
}
