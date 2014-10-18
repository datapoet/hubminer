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
package combinatorial;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Random;
import util.AuxSort;

/**
 * This class implements some basic permutation-related operations.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class Permutation {

    /**
     * Tests to see if the permutation is an identity map.
     *
     * @param perm Integer array representing the permutation.
     * @return True if for all i, perm[i]==i. False otherwise.
     */
    public static boolean isIdentity(int[] perm) {
        if (perm == null || perm.length == 1) {
            return true;
        }
        for (int i = 0; i < perm.length; i++) {
            if (i != perm[i]) {
                return false;
            }
        }
        return true;
    }

    /**
     * Generates a random permutation.
     *
     * @param len Length of the desired permutation.
     * @return Integer array that is a random permutation of numbers 0..len-1
     */
    public static int[] obtainRandomPermutation(int len) throws Exception {
        if (len < 1) {
            return null;
        }
        Random randa = new Random();
        int[] valArray = new int[len];
        for (int i = 0; i < len; i++) {
            valArray[i] = randa.nextInt(len);
        }
        if (len == 1) {
            return valArray;
        }
        return AuxSort.sortIndexedValue(valArray, true);
    }

    /**
     * Permutes the rows of a square matrix.
     *
     * @param matrix Float 2D array that is the square matrix.
     * @param perm Permutation to apply.
     */
    public static void permuteSquareMatrixRows(float[][] matrix, int[] perm) {
        if (matrix == null || perm == null || matrix.length != perm.length) {
            return;
        }
        HashMap permMap = new HashMap(perm.length * 3);
        // For a quick lookup into where a specific element has been permuted.
        for (int i = 0; i < perm.length; i++) {
            permMap.put(perm[i], i);
        }
        int curr;
        int length;
        float[] tmpRow1, tmpRow2;
        boolean[] isProcessed = new boolean[perm.length];
        Arrays.fill(isProcessed, false);
        ArrayList<Integer> cycle;
        for (int i = 0; i < perm.length; i++) {
            if (!isProcessed[i]) {
                curr = i;
                length = 0;
                cycle = new ArrayList<>(10);
                do {
                    curr = perm[curr];
                    isProcessed[curr] = true;
                    length++;
                    cycle.add(i);
                } while (curr != i);
                length--;
                if (length > 1) {
                    tmpRow2 = matrix[cycle.get(0)];
                    for (int j = 0; j < cycle.size() - 1; j++) {
                        tmpRow1 = matrix[cycle.get(j + 1)];
                        matrix[cycle.get(j + 1)] = tmpRow2;
                        tmpRow2 = tmpRow1;
                    }
                    matrix[cycle.get(0)] = tmpRow2;
                } else if (length == 1) {
                    tmpRow1 = matrix[cycle.get(0)];
                    matrix[cycle.get(0)] = matrix[cycle.get(1)];
                    matrix[cycle.get(1)] = tmpRow1;
                }
            }
        }
    }

    /**
     * Copies and transposes a square matrix.
     *
     * @param matrix Float 2D array that is the square matrix.
     * @return A transposed copy of the original matrix.
     */
    private static float[][] copyAndTransposeSquareMat(float[][] matrix) {
        if (matrix == null) {
            return null;
        }
        float[][] result = new float[matrix.length][matrix.length];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix.length; j++) {
                result[i][j] = matrix[j][i];
            }
        }
        return result;
    }

    /**
     * Permutes the columns of a square matrix.
     *
     * @param matrix Float 2D array that is the square matrix.
     * @param perm Permutation to apply.
     */
    public static float[][] permuteSquareMatrixColumns(
            float[][] matrix, int[] perm) {
        if (matrix == null || perm == null || matrix.length != perm.length) {
            return null;
        }
        float[][] matTransposedCopy = copyAndTransposeSquareMat(matrix);
        permuteSquareMatrixRows(matTransposedCopy, perm);
        float[][] matPermutedCopy =
                copyAndTransposeSquareMat(matTransposedCopy);
        return matPermutedCopy;
    }

    /**
     * Permutes the rows of a square matrix.
     *
     * @param matrix Double 2D array that is the square matrix.
     * @param perm Permutation to apply.
     */
    public static void permuteSquareMatrixRows(double[][] matrix, int[] perm) {
        if (matrix == null || perm == null || matrix.length != perm.length) {
            return;
        }
        HashMap permMap = new HashMap(perm.length * 3);
        // For a quick lookup into where a specific element has been permuted.
        for (int i = 0; i < perm.length; i++) {
            permMap.put(perm[i], i);
        }
        int curr;
        int length;
        double[] tmpRow1, tmpRow2;
        boolean[] isProcessed = new boolean[perm.length];
        Arrays.fill(isProcessed, false);
        ArrayList<Integer> cycle;
        for (int i = 0; i < perm.length; i++) {
            if (!isProcessed[i]) {
                curr = i;
                length = 0;
                cycle = new ArrayList<>(10);
                do {
                    curr = perm[curr];
                    isProcessed[curr] = true;
                    length++;
                    cycle.add(i);
                } while (curr != i);
                length--;
                if (length > 1) {
                    tmpRow2 = matrix[cycle.get(0)];
                    for (int j = 0; j < cycle.size() - 1; j++) {
                        tmpRow1 = matrix[cycle.get(j + 1)];
                        matrix[cycle.get(j + 1)] = tmpRow2;
                        tmpRow2 = tmpRow1;
                    }
                    matrix[cycle.get(0)] = tmpRow2;
                } else if (length == 1) {
                    tmpRow1 = matrix[cycle.get(0)];
                    matrix[cycle.get(0)] = matrix[cycle.get(1)];
                    matrix[cycle.get(1)] = tmpRow1;
                }
            }
        }
    }

    /**
     * Copies and transposes a square matrix.
     *
     * @param matrix Double 2D array that is the square matrix.
     * @return A transposed copy of the original matrix.
     */
    private static double[][] copyAndTransposeSquareMat(double[][] matrix) {
        if (matrix == null) {
            return null;
        }
        double[][] result = new double[matrix.length][matrix.length];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix.length; j++) {
                result[i][j] = matrix[j][i];
            }
        }
        return result;
    }

    /**
     * Permutes the columns of a square matrix.
     *
     * @param matrix Double 2D array that is the square matrix.
     * @param perm Permutation to apply.
     */
    public static double[][] permuteSquareMatrixColumns(
            double[][] matrix, int[] perm) {
        if (matrix == null || perm == null || matrix.length != perm.length) {
            return null;
        }
        double[][] matTransposedCopy = copyAndTransposeSquareMat(matrix);
        permuteSquareMatrixRows(matTransposedCopy, perm);
        double[][] matPermutedCopy =
                copyAndTransposeSquareMat(matTransposedCopy);
        return matPermutedCopy;
    }

    /**
     * Inverts the permutation.
     *
     * @param perm Integer array representing the permutation.
     * @return Integer array representing the inverse permutation.
     */
    public static int[] invertPermutation(int[] perm) {
        if (perm == null) {
            return null;
        }
        HashMap permMap = new HashMap(perm.length * 3);
        for (int i = 0; i < perm.length; i++) {
            permMap.put(perm[i], i);
        }
        int curr;
        int length;
        int[] inverse = Arrays.copyOf(perm, perm.length);
        boolean[] isProcessed = new boolean[perm.length];
        Arrays.fill(isProcessed, false);
        ArrayList<Integer> cycle;
        for (int i = 0; i < perm.length; i++) {
            if (!isProcessed[i]) {
                curr = i;
                length = 0;
                cycle = new ArrayList<>(10);
                do {
                    curr = perm[curr];
                    isProcessed[curr] = true;
                    length++;
                    cycle.add(i);
                } while (curr != i);
                length--;
                if (length > 1) {
                    for (int j = cycle.size() - 1; j > 0; j--) {
                        inverse[cycle.get(j)] = cycle.get(j - 1);
                    }
                    inverse[cycle.get(0)] = cycle.get(cycle.size() - 1);
                }
            }
        }
        return inverse;
    }

    /**
     * Counts the number of transpositions in the given permutation.
     *
     * @param perm Integer array representing the permutation.
     * @return Integer representing the number of transpositions in the given
     * permutation.
     */
    public static int countTranspositions(int[] perm) {
        if (perm == null) {
            return 0;
        }
        HashMap permMap = new HashMap(perm.length * 3);
        for (int i = 0; i < perm.length; i++) {
            permMap.put(perm[i], i);
        }
        int curr;
        int length;
        int transpCount = 0;
        boolean[] isProcessed = new boolean[perm.length];
        Arrays.fill(isProcessed, false);
        for (int i = 0; i < perm.length; i++) {
            if (!isProcessed[i]) {
                curr = i;
                length = 0;
                do {
                    curr = perm[curr];
                    isProcessed[curr] = true;
                    length++;
                } while (curr != i);
                length--;
                transpCount += length;
            }
        }
        return transpCount;
    }

    /**
     * Checks if a permutation has an even number of transpositions.
     *
     * @param perm Integer array representing the permutation.
     * @return True if the number of transpositions is even, false otherwise.
     */
    public static boolean isEvenPermutation(int[] perm) {
        return countTranspositions(perm) % 2 == 0;
    }

    /**
     * Checks if a permutation has an odd number of transpositions.
     *
     * @param perm Integer array representing the permutation.
     * @return True if the number of transpositions is odd, false otherwise.
     */
    public static boolean isOddPermutation(int[] perm) {
        return countTranspositions(perm) % 2 != 0;
    }
}
