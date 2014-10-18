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
package feature.correlation;

import data.representation.DataSet;
import data.representation.util.DataMineConstants;

/**
 * Pearson correlation produces values from -1 to 1, 0 if independent. The
 * reverse does not necessarily hold. If the correlation is found to be 0, it
 * does not entail independence! Pearson coefficient mostly checks for linear
 * dependencies. Different approaches are required for nonlinear correlations.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class PearsonCorrelation extends CorrelationCoefficient {

    /**
     * Calculates the Pearson product moment as a correlation between the two
     * value arrays.
     *
     * @param first Integer value array.
     * @param second Integer value array.
     * @return Float that is the Pearson correlation between the two value
     * arrays. It ranges from -1 to +1.
     * @throws Exception
     */
    public static float correlation(int[] first, int[] second)
            throws Exception {
        if (first == null || second == null || first.length != second.length) {
            return 0;
        }
        double p = 0;
        double stDevFirst = 0;
        double stDevSecond = 0;
        double meanFirst = 0;
        double meanSecond = 0;
        int length = first.length;
        for (int i = 0; i < length; i++) {
            if (DataMineConstants.isAcceptableInt(first[i])) {
                meanFirst += first[i];
            }
            if (DataMineConstants.isAcceptableInt(second[i])) {
                meanSecond += second[i];
            }
        }
        meanFirst /= (double) (length);
        meanSecond /= (double) (length);
        double difFirst, difSecond;
        for (int i = 0; i < length; i++) {
            difFirst = (first[i] - meanFirst);
            difSecond = (second[i] - meanSecond);
            if (DataMineConstants.isAcceptableDouble(difFirst)
                    && DataMineConstants.isAcceptableDouble(difSecond)) {
                stDevFirst += difFirst * difFirst;
                stDevSecond += difSecond * difSecond;
                p += difFirst * difSecond;
            }
        }
        stDevFirst = Math.sqrt(stDevFirst);
        stDevSecond = Math.sqrt(stDevSecond);
        if (stDevFirst == 0 || stDevSecond == 0) {
            return 0;
        }
        // Normalize by the product of standard deviations.
        p /= (stDevFirst * stDevSecond);
        return (float) p;
    }

    /**
     * Calculates the Pearson product moment as a correlation between the two
     * value arrays.
     *
     * @param first Float value array.
     * @param second Float value array.
     * @return Float that is the Pearson correlation between the two value
     * arrays. It ranges from -1 to +1.
     * @throws Exception
     */
    public static float correlation(float[] first, float[] second)
            throws Exception {
        if (first == null || second == null || first.length != second.length) {
            return 0;
        }
        double p = 0;
        double stDevFirst = 0;
        double stDevSecond = 0;
        double meanFirst = 0;
        double meanSecond = 0;
        int length = first.length;
        for (int i = 0; i < length; i++) {
            if (DataMineConstants.isAcceptableFloat(first[i])) {
                meanFirst += first[i];
            }
            if (DataMineConstants.isAcceptableFloat(second[i])) {
                meanSecond += second[i];
            }
        }
        meanFirst /= (double) (length);
        meanSecond /= (double) (length);
        double difFirst, difSecond;
        for (int i = 0; i < length; i++) {
            difFirst = (first[i] - meanFirst);
            difSecond = (second[i] - meanSecond);
            if (DataMineConstants.isAcceptableDouble(difFirst)
                    && DataMineConstants.isAcceptableDouble(difSecond)) {
                stDevFirst += difFirst * difFirst;
                stDevSecond += difSecond * difSecond;
                p += difFirst * difSecond;
            }
        }
        stDevFirst = Math.sqrt(stDevFirst);
        stDevSecond = Math.sqrt(stDevSecond);
        if (stDevFirst == 0 || stDevSecond == 0) {
            return 0;
        }
        // Normalize by the product of standard deviations.
        p /= (stDevFirst * stDevSecond);
        return (float) p;
    }

    /**
     * Calculates the Pearson product moment as a correlation between the two
     * value arrays.
     *
     * @param first Double value array.
     * @param second Double value array.
     * @return Float that is the Pearson correlation between the two value
     * arrays. It ranges from -1 to +1.
     * @throws Exception
     */
    public static float correlation(double[] first, double[] second)
            throws Exception {
        if (first == null || second == null || first.length != second.length) {
            return 0;
        }
        double p = 0;
        double stDevFirst = 0;
        double stDevSecond = 0;
        double meanFirst = 0;
        double meanSecond = 0;
        int length = first.length;
        for (int i = 0; i < length; i++) {
            if (DataMineConstants.isAcceptableDouble(first[i])) {
                meanFirst += first[i];
            }
            if (DataMineConstants.isAcceptableDouble(second[i])) {
                meanSecond += second[i];
            }
        }
        meanFirst /= (double) (length);
        meanSecond /= (double) (length);
        double difFirst, difSecond;
        for (int i = 0; i < length; i++) {
            difFirst = (first[i] - meanFirst);
            difSecond = (second[i] - meanSecond);
            if (DataMineConstants.isAcceptableDouble(difFirst)
                    && DataMineConstants.isAcceptableDouble(difSecond)) {
                stDevFirst += difFirst * difFirst;
                stDevSecond += difSecond * difSecond;
                p += difFirst * difSecond;
            }
        }
        stDevFirst = Math.sqrt(stDevFirst);
        stDevSecond = Math.sqrt(stDevSecond);
        if (stDevFirst == 0 || stDevSecond == 0) {
            return 0;
        }
        // Normalize by the product of standard deviations.
        p /= (stDevFirst * stDevSecond);
        return (float) p;
    }

    /**
     * Calculates the Pearson product moment as a correlation between the two
     * value arrays.
     *
     * @param first Float value array.
     * @param second Double value array.
     * @return Float that is the Pearson correlation between the two value
     * arrays. It ranges from -1 to +1.
     * @throws Exception
     */
    public static float correlation(float[] first, double[] second)
            throws Exception {
        if (first == null || second == null || first.length != second.length) {
            return 0;
        }
        double p = 0;
        double stDevFirst = 0;
        double stDevSecond = 0;
        double meanFirst = 0;
        double meanSecond = 0;
        int length = first.length;
        for (int i = 0; i < length; i++) {
            if (DataMineConstants.isAcceptableFloat(first[i])) {
                meanFirst += first[i];
            }
            if (DataMineConstants.isAcceptableDouble(second[i])) {
                meanSecond += second[i];
            }
        }
        meanFirst /= (double) (length);
        meanSecond /= (double) (length);
        double difFirst, difSecond;
        for (int i = 0; i < length; i++) {
            difFirst = (first[i] - meanFirst);
            difSecond = (second[i] - meanSecond);
            if (DataMineConstants.isAcceptableDouble(difFirst)
                    && DataMineConstants.isAcceptableDouble(difSecond)) {
                stDevFirst += difFirst * difFirst;
                stDevSecond += difSecond * difSecond;
                p += difFirst * difSecond;
            }
        }
        stDevFirst = Math.sqrt(stDevFirst);
        stDevSecond = Math.sqrt(stDevSecond);
        if (stDevFirst == 0 || stDevSecond == 0) {
            return 0;
        }
        // Normalize by the product of standard deviations.
        p /= (stDevFirst * stDevSecond);
        return (float) p;
    }

    public static float correlation(DataSet dset, float[] first, float[] second)
            throws Exception {
        return correlation(first, second);
    }
}
