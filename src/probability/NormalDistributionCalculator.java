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
package probability;

/**
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class NormalDistributionCalculator {

    // Parameters for the cumulative estimate Zelen & Severo (1964)
    static double b0 = 0.2316419;
    static double b1 = 0.319381530;
    static double b2 = -0.356563782;
    static double b3 = 1.781477937;
    static double b4 = -1.821255978;
    static double b5 = 1.330274429;
    // 1/2*pi factor.
    public static final double normFact = 1 / Math.sqrt(2 * Math.PI);

    /**
     * @param x Double value.
     * @return The probability density function of a normal x, assuming zero
     * mean and unit variance.
     */
    public static double phi(double x) {
        return normFact * Math.exp(-x * x / 2);
    }

    /**
     *
     * @param x Double value
     * @param mean Mean value.
     * @param sigma Standard deviation.
     * @return Gaussian probability density of the passed value for the given
     * mean and standard deviation.
     */
    public static double phi(double x, double mean, double sigma) {
        return (1 / sigma) * phi((x - mean) / sigma);
    }

    /**
     * Zelen & Severo (1964) - method for approximating the cumulative
     * distribution function
     *
     * @param x Double value.
     * @return Cumulative distribution function for the normal distribution,
     * upper bounded by the passed value x.
     */
    public static double PhiCumulative(double x) {
        double t = 1 / (1 + b0 * x);
        double tDeg = t;
        double result = b1 * tDeg;
        tDeg *= t;
        result += b2 * tDeg;
        tDeg *= t;
        result += b3 * tDeg;
        tDeg *= t;
        result += b4 * tDeg;
        tDeg *= t;
        result += b5 * tDeg;
        result = 1 - phi(x) * result;
        return result;
    }

    /**
     * Zelen & Severo (1964) - method for approximating the cumulative
     * distribution function
     *
     * @param x Double value.
     * @param mean Mean value.
     * @param sigma Standard deviation.
     * @return Cumulative distribution function for the normal distribution,
     * upper bounded by the passed value x.
     */
    public static double PhiCumulative(double x, double mean, double stDev) {
        return PhiCumulative((x - mean) / stDev);
    }
}
