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
 * The basic float matrix implementation.
 *
 * @author Nenad Tomasev
 */
public class DataMatrix implements DataMatrixInterface {

    private float[][] data;

    /**
     * @param rows Number of rows.
     * @param columns Number of columns.
     */
    public DataMatrix(int rows, int columns) {
        data = new float[rows][columns];
    }

    /**
     * @param data Two-dimensional float array that represents the matrix.
     */
    public DataMatrix(float[][] data) {
        this.data = data;
    }

    /**
     * @return Two-dimensional float array that represents the matrix.
     */
    public float[][] getFloatMatrix() {
        return data;
    }

    @Override
    public int numberOfRows() {
        if (data != null) {
            return data.length;
        } else {
            return 0;
        }
    }

    @Override
    public int numberOfColumns() {
        if (numberOfRows() > 0) {
            if (data[0] != null) {
                return data[0].length;
            } else {
                return 0;
            }
        } else {
            return 0;
        }
    }

    @Override
    public void setElementAt(int row, int column, float value) {
        if (row < numberOfRows() && column < numberOfColumns() && row >= 0
                && column >= 0) {
            data[row][column] = value;
        }
    }

    @Override
    public float getElementAt(int row, int column) {
        if (row < numberOfRows() && column < numberOfColumns() && row >= 0
                && column >= 0) {
            return data[row][column];
        } else {
            return Float.NaN;
        }
    }

    @Override
    public void setRow(int row, float[] values) throws Exception {
        if (row < numberOfRows() && row >= 0) {
            data[row] = values;
        }
    }

    @Override
    public void setColumn(int column, float[] values) throws Exception {
        if (column < numberOfColumns() && column >= 0) {
            for (int i = 0; i < numberOfRows(); i++) {
                data[i][column] = values[i];
            }
        }
    }

    @Override
    public float[] getRow(int row) {
        if (row < numberOfRows()) {
            return data[row];
        } else {
            return null;
        }
    }

    @Override
    public float[] getColumn(int column) {
        float[] values = null;
        if (column < numberOfColumns() && column >= 0) {
            values = new float[numberOfRows()];
            for (int i = 0; i < numberOfRows(); i++) {
                values[i] = data[i][column];
            }
            return values;
        } else {
            return values;
        }
    }

    @Override
    public boolean isSquare() {
        return numberOfRows() == numberOfColumns();
    }

    @Override
    public boolean isSymmetricMatrixImplementation() {
        // This method is not really concerned with whether the current matrix
        // is in fact symmetric or not but rather with the implementation itself
        // , whether it enforces symmetricity or not.
        return false;
    }

    @Override
    public DataMatrixInterface plus(DataMatrixInterface matrix)
            throws Exception {
        if (numberOfRows() == matrix.numberOfRows()
                && numberOfColumns() == matrix.numberOfColumns()) {
            DataMatrixInterface result =
                    new DataMatrix(numberOfRows(), numberOfColumns());
            for (int i = 0; i < numberOfRows(); i++) {
                for (int j = 0; j < numberOfColumns(); j++) {
                    result.setElementAt(i, j,
                            data[i][j] + matrix.getElementAt(i, j));
                }
            }
            return result;
        } else {
            return null;
        }
    }

    @Override
    public DataMatrixInterface minus(DataMatrixInterface matrix)
            throws Exception {
        if (numberOfRows() == matrix.numberOfRows()
                && numberOfColumns() == matrix.numberOfColumns()) {
            DataMatrixInterface result =
                    new DataMatrix(numberOfRows(), numberOfColumns());
            for (int i = 0; i < numberOfRows(); i++) {
                for (int j = 0; j < numberOfColumns(); j++) {
                    result.setElementAt(i, j,
                            data[i][j] - matrix.getElementAt(i, j));
                }
            }
            return result;
        } else {
            return null;
        }
    }

    @Override
    public DataMatrixInterface multiply(DataMatrixInterface matrix)
            throws Exception {
        if (numberOfColumns() == matrix.numberOfRows()) {
            float[][] resultMat =
                    new float[numberOfRows()][matrix.numberOfColumns()];
            for (int i = 0; i < numberOfRows(); i++) {
                for (int j = 0; j < matrix.numberOfColumns(); j++) {
                    for (int k = 0; k < numberOfColumns(); k++) {
                        resultMat[i][j] +=
                                data[i][k] * matrix.getElementAt(k, j);
                    }
                }
            }
            DataMatrixInterface result = new DataMatrix(resultMat);
            return result;
        } else {
            return null;
        }
    }

    @Override
    public DataMatrixInterface multiply(float scalar) {
        DataMatrixInterface result =
                new DataMatrix(numberOfRows(), numberOfColumns());
        for (int i = 0; i < numberOfRows(); i++) {
            for (int j = 0; j < numberOfColumns(); j++) {
                result.setElementAt(i, j, scalar * data[i][j]);
            }
        }
        return result;
    }

    @Override
    public float calculateDeterminant() {
        SquareMatrixLUDecomposition helper =
                new SquareMatrixLUDecomposition(data);
        helper.performLUdecomposition();
        return (float) helper.calculateDeterminant();
    }

    @Override
    public DataMatrixInterface calculateInverse() throws Exception {
        SquareMatrixLUDecomposition helper =
                new SquareMatrixLUDecomposition(data);
        float[][] inverseData = helper.getMatrixInverse();
        DataMatrix result = new DataMatrix(inverseData);
        return result;
    }
}
