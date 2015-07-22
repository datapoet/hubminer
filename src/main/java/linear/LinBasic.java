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

import data.representation.util.DataMineConstants;

/**
 * Basic linear operations on vectors.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class LinBasic {

    /**
     * Modus vector operation, the norm.
     *
     * @param vect Vector in a form of a float array.
     * @return Float value representing the vector modus.
     */
    public static float modus(float[] vect) {
        if (vect == null || vect.length == 0) {
            return 0;
        }
        float result = 0;
        for (int i = 0; i < vect.length; i++) {
            result += vect[i] * vect[i];
        }
        result = (float) Math.sqrt(result);
        return result;
    }

    /**
     * Vector subtraction.
     *
     * @param vect1 Vector to subtract from, in a form of a float array.
     * @param vect2 Vector to subtract, in a form of a float array.
     * @return The difference between the vectors.
     */
    public static float[] decr(float[] vect1, float[] vect2) {
        float[] result = new float[vect1.length];
        for (int i = 0; i < result.length; i++) {
            result[i] = vect1[i] - vect2[i];
        }
        return result;
    }

    /**
     * Vector addition.
     *
     * @param vect1 First vector to add, in a form of a float array.
     * @param vect2 Second vector to add, in a form of a float array.
     * @return Vector sum of two given vectors, in a form of a float array.
     */
    public static float[] add(float[] vect1, float[] vect2) {
        float[] result = new float[vect1.length];
        for (int i = 0; i < result.length; i++) {
            result[i] = vect1[i] + vect2[i];
        }
        return result;
    }

    /**
     * Vector addition where the result is stored in the first vector.
     *
     * @param vect1 First vector to add and modify, in a form of a float array.
     * @param vect2 Second vector to add, in a form of a float array.
     */
    public static void addSecondToFirst(float[] vect1, float[] vect2) {
        for (int i = 0; i < vect1.length; i++) {
            vect1[i] = vect1[i] + vect2[i];
        }
    }

    /**
     * Vector subtraction, where the result is stored in the first vector.
     *
     * @param vect1 Vector to subtract from, in a form of a float array.
     * @param vect2 Vector to subtract, in a form of a float array.
     */
    public static void decrSecondFromFirst(float[] vect1, float[] vect2) {
        for (int i = 0; i < vect1.length; i++) {
            vect1[i] = vect1[i] - vect2[i];
        }
    }

    /**
     * Performs a dot product between two vectors.
     *
     * @param vect1 First vector, in a form of a float array.
     * @param vect2 Second vector, in a form of a float array.
     * @return Dot product of the two vectors, in a form of a float array.
     */
    public static float dotProduct(float[] vect1, float[] vect2) {
        float result = 0;
        for (int i = 0; i < vect1.length; i++) {
            result += vect1[i] * vect2[i];
        }
        return result;
    }

    /**
     * Cosine of the angle between two vectors.
     *
     * @param vect1 First vector, in a form of a float array.
     * @param vect2 Second vector, in a form of a float array.
     * @return Cosine of the angle between two vectors, as float.
     */
    public static float angleCosine(float[] vect1, float[] vect2) {
        float dot = dotProduct(vect1, vect2);
        float m1 = modus(vect1);
        float m2 = modus(vect2);
        if (DataMineConstants.isZero(m1) || DataMineConstants.isZero(m2)) {
            return 0f;
        } else {
            return (dot / (m1 * m2));
        }
    }

    /**
     * Multiply a vector by a scalar and output a new vector as a result.
     *
     * @param vect Vector, in a form of a float array.
     * @param scalar Scalar value.
     * @return Vector that is the result of scalar multiplication.
     */
    public static float[] scalarMultiply(float[] vect, float scalar) {
        float[] result = new float[vect.length];
        for (int i = 0; i < vect.length; i++) {
            result[i] = vect[i] * scalar;
        }
        return result;
    }

    /**
     * Modify a vector by multiplying it by a scalar value.
     *
     * @param vect Vector, in a form of a float array.
     * @param scalar Scalar value.
     */
    public static void scalarMultiplyThisVector(float[] vect, float scalar) {
        for (int i = 0; i < vect.length; i++) {
            vect[i] = vect[i] * scalar;
        }
    }

    /**
     * Generate a unit matrix of a given dimensionality.
     *
     * @param dim Number of rows/columns.
     * @return Unit matrix of the specified size.
     */
    public static float[][] getUnitMatrix(int dim) {
        float[][] unit = new float[dim][dim];
        for (int i = 0; i < dim; i++) {
            unit[i][i] = 1;
        }
        return unit;
    }

    /**
     * Modus vector operation, the norm.
     *
     * @param vect Vector in a form of a double array.
     * @return Double value representing the vector modus.
     */
    public static double modus(double[] vect) {
        if (vect == null || vect.length == 0) {
            return 0;
        }
        double result = 0;
        for (int i = 0; i < vect.length; i++) {
            result += vect[i] * vect[i];
        }
        result = Math.sqrt(result);
        return result;
    }

    /**
     * Vector subtraction.
     *
     * @param vect1 Vector to subtract from, in a form of a double array.
     * @param vect2 Vector to subtract, in a form of a double array.
     * @return The difference between the vectors.
     */
    public static double[] decr(double[] vect1, double[] vect2) {
        double[] result = new double[vect1.length];
        for (int i = 0; i < result.length; i++) {
            result[i] = vect1[i] - vect2[i];
        }
        return result;
    }

    /**
     * Vector addition.
     *
     * @param vect1 First vector to add, in a form of a double array.
     * @param vect2 Second vector to add, in a form of a double array.
     * @return Vector sum of two given vectors, in a form of a double array.
     */
    public static double[] add(double[] vect1, double[] vect2) {
        double[] result = new double[vect1.length];
        for (int i = 0; i < result.length; i++) {
            result[i] = vect1[i] + vect2[i];
        }
        return result;
    }

    /**
     * Vector addition where the result is stored in the first vector.
     *
     * @param vect1 First vector to add and modify, in a form of a double array.
     * @param vect2 Second vector to add, in a form of a double array.
     */
    public static void addSecondToFirst(double[] vect1, double[] vect2) {
        for (int i = 0; i < vect1.length; i++) {
            vect1[i] = vect1[i] + vect2[i];
        }
    }

    /**
     * Vector subtraction, where the result is stored in the first vector.
     *
     * @param vect1 Vector to subtract from, in a form of a double array.
     * @param vect2 Vector to subtract, in a form of a double array.
     */
    public static void decrSecondFromFirst(double[] vect1, double[] vect2) {
        for (int i = 0; i < vect1.length; i++) {
            vect1[i] = vect1[i] - vect2[i];
        }
    }

    /**
     * Performs a dot product between two vectors.
     *
     * @param vect1 First vector, in a form of a double array.
     * @param vect2 Second vector, in a form of a double array.
     * @return Dot product of the two vectors, in a form of a double array.
     */
    public static double dotProduct(double[] vect1, double[] vect2) {
        double result = 0;
        for (int i = 0; i < vect1.length; i++) {
            result += vect1[i] * vect2[i];
        }
        return result;
    }

    /**
     * Cosine of the angle between two vectors.
     *
     * @param vect1 First vector, in a form of a double array.
     * @param vect2 Second vector, in a form of a double array.
     * @return Cosine of the angle between two vectors, as double.
     */
    public static double angleCosine(double[] vect1, double[] vect2) {
        double dot = dotProduct(vect1, vect2);
        double m1 = modus(vect1);
        double m2 = modus(vect2);
        if (DataMineConstants.isZero(m1) || DataMineConstants.isZero(m2)) {
            return 0;
        } else {
            return (dot / (m1 * m2));
        }
    }

    /**
     * Multiply a vector by a scalar and output a new vector as a result.
     *
     * @param vect Vector, in a form of a double array.
     * @param scalar Scalar value.
     * @return Vector that is the result of scalar multiplication.
     */
    public static double[] scalarMultiply(double[] vect, double scalar) {
        double[] result = new double[vect.length];
        for (int i = 0; i < vect.length; i++) {
            result[i] = vect[i] * scalar;
        }
        return result;
    }

    /**
     * Modify a vector by multiplying it by a scalar value.
     *
     * @param vect Vector, in a form of a double array.
     * @param scalar Scalar value.
     */
    public static void scalarMultiplyThisVector(double[] vect, double scalar) {
        for (int i = 0; i < vect.length; i++) {
            vect[i] = vect[i] * scalar;
        }
    }

    /**
     * Generate a unit matrix of a given dimensionality.
     *
     * @param dim Number of rows/columns.
     * @return Unit matrix of the specified size.
     */
    public static double[][] getDoubleUnitMatrix(int dim) {
        double[][] unit = new double[dim][dim];
        for (int i = 0; i < dim; i++) {
            unit[i][i] = 1;
        }
        return unit;
    }
}
