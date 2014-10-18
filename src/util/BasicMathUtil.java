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
package util;

/**
 * Some basic math utility methods.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class BasicMathUtil {

    public static final double L = Math.log10(2);

    /**
     * The binary logarithm.
     *
     * @param f Value to take the logarithm of.
     * @return Base 2 logarithm of the passed value.
     */
    public static double log2(double f) {
        return Math.log10(f) / L;
    }

    /**
     * The binary logarithm.
     *
     * @param f Value to take the logarithm of.
     * @return Base 2 logarithm of the passed value.
     */
    public static double log2(float f) {
        return Math.log10(f) / L;
    }

    /**
     * Cut off some decimal places that are not significant.
     *
     * @param x A float value.
     * @param numDecimalPlaces Number of decimal places to keep.
     * @return The float value with certain decimals cut off.
     */
    public static float makeADecimalCutOff(float x, int numDecimalPlaces) {
        float tenPower = (float) Math.pow(10, numDecimalPlaces);
        float result = tenPower * x;
        int resultInt = (int) result;
        if (result - resultInt > 0.5) {
            resultInt++;
        } else if (result - resultInt == 0.5) {
            if (resultInt % 2 == 1) {
                // Odd numbers are rounded up on 0.5 case.
                resultInt++;
            }
        }
        result = (float) resultInt / tenPower;
        return result;
    }

    /**
     * Cut off some decimal places that are not significant.
     *
     * @param x A double value.
     * @param numDecimalPlaces Number of decimal places to keep.
     * @return The float value with certain decimals cut off.
     */
    public static double makeADecimalCutOff(double x, int numDecimalPlaces) {
        double tenPower = Math.pow(10, numDecimalPlaces);
        double result = tenPower * x;
        int resultInt = (int) result;
        if (result - resultInt > 0.5) {
            resultInt++;
        } else if (result - resultInt == 0.5) {
            // Odd numbers are rounded up on 0.5 case.
            if (resultInt % 2 == 1) {
                resultInt++;
            }
        }
        result = resultInt / tenPower;
        return result;
    }
}
