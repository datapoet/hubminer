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
package data.representation.util;

/**
 * This class holds some general constants and utility methods.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class DataMineConstants {

    public static final int INTEGER = 0;
    public static final int FLOAT = 1;
    public static final int NOMINAL = 2;
    public static final float EPSILON =
            (float) 0.00000000000000000000000000000000000001;
    public static final double EPSILON_CONVERGENCE = 0.000001;

    /**
     * Checks if a number is not zero.
     *
     * @param val Float value.
     * @return False if zero, true otherwise.
     */
    public static boolean isNonZero(float val) {
        return isAcceptableFloat(val) && Math.abs(val) >= EPSILON;
    }

    /**
     * Checks if a number is zero.
     *
     * @param val Float value.
     * @return True if zero, false otherwise.
     */
    public static boolean isZero(float val) {
        return isAcceptableFloat(val) && Math.abs(val) < EPSILON;
    }

    /**
     * Checks if a number is negative.
     *
     * @param val Float value.
     * @return True if negative, false otherwise.
     */
    public static boolean isNegative(float val) {
        return isAcceptableFloat(val) && val <= -EPSILON;
    }

    /**
     * Checks if a number is positive.
     *
     * @param val Float value.
     * @return True if positive, false otherwise.
     */
    public static boolean isPositive(float val) {
        return isAcceptableFloat(val) && val >= EPSILON;
    }

    /**
     * Checks if a number is not zero.
     *
     * @param val Double value.
     * @return False if zero, true otherwise.
     */
    public static boolean isNonZero(double val) {
        return isAcceptableDouble(val) && Math.abs(val) >= EPSILON;
    }

    /**
     * Checks if a number is positive.
     *
     * @param val Double value.
     * @return True if positive, false otherwise.
     */
    public static boolean isPositive(double val) {
        return isAcceptableDouble(val) && val >= EPSILON;
    }

    /**
     * Checks if a number is negative.
     *
     * @param val Double value.
     * @return True if negative, false otherwise.
     */
    public static boolean isNegative(double val) {
        return isAcceptableDouble(val) && val <= -EPSILON;
    }

    /**
     * Checks if a number is zero.
     *
     * @param val Double value.
     * @return True if zero, false otherwise.
     */
    public static boolean isZero(double val) {
        return isAcceptableDouble(val) && Math.abs(val) < EPSILON;
    }

    /**
     * Checks if a number is acceptable, in a sense that it is not one of the
     * special limit or undefined values.
     *
     * @param val Double value.
     * @return True if not a max value, NaN or infinity.
     */
    public static boolean isAcceptableDouble(double val) {
        if (val == Double.NaN || val == Double.MAX_VALUE
                || val == Double.POSITIVE_INFINITY
                || val == Double.NEGATIVE_INFINITY) {
            return false;
        } else {
            return true;
        }
    }

    /**
     * Checks if a number is acceptable, in a sense that it is not one of the
     * special limit or undefined values.
     *
     * @param val Float value.
     * @return True if not a max value, NaN or infinity.
     */
    public static boolean isAcceptableFloat(float val) {
        if (val == Float.NaN || val == Float.MAX_VALUE
                || val == Float.POSITIVE_INFINITY
                || val == Float.NEGATIVE_INFINITY) {
            return false;
        } else {
            return true;
        }
    }

    /**
     * Checks if a number is acceptable, in a sense that it is not one of the
     * special limit values.
     *
     * @param val Integer value.
     * @return True if not a Integer.MAX_VALUE or Integer.MIN_VALUE.
     */
    public static boolean isAcceptableInt(int val) {
        if (val == Integer.MAX_VALUE || val == Integer.MIN_VALUE) {
            return false;
        } else {
            return true;
        }
    }
}
