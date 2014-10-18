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
import linear.matrix.SymmetricFloatMatrix;

/**
 * Distance correlation is a correlation measure that is able to detect more
 * than Pearson coefficient due to the fact that it is non-linear. It returns
 * values from 0 to 1 and it equals 0 if and only if there is no statistical
 * dependence between the variable arrays.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class DistanceCorrelation extends CorrelationCoefficient {

    /**
     * Calculates the distance correlation.
     *
     * @param first Double array of values.
     * @param second Double array of values.
     * @return Distance correlation between the two arrays.
     * @throws Exception
     */
    public static double correlation(double[] first, double[] second)
            throws Exception {
        if (first == null || second == null || first.length != second.length) {
            return 0;
        }
        int length = first.length;
        SymmetricFloatMatrix dMatFirst = new SymmetricFloatMatrix(length);
        SymmetricFloatMatrix dMatSecond = new SymmetricFloatMatrix(length);
        double[] avgFirst = new double[length];
        double[] avgSecond = new double[length];
        double firstTotalAvg = 0;
        double secondTotalAvg = 0;
        double distanceStdevFirst = 0;
        double distanceStdevSecond = 0;
        double distanceCorr = 0;
        double currDist;
        for (int i = 0; i < length; i++) {
            for (int j = i + 1; j < length; j++) {
                currDist = Math.abs(first[i] - first[j]);
                dMatFirst.setElementAt(i, j, (float) currDist);
                if (DataMineConstants.isAcceptableDouble(currDist)) {
                    avgFirst[i] += currDist;
                    avgFirst[j] += currDist;
                    firstTotalAvg += currDist;
                }
                currDist = Math.abs(second[i] - second[j]);
                dMatSecond.setElementAt(i, j, (float) currDist);
                if (DataMineConstants.isAcceptableDouble(currDist)) {
                    avgSecond[i] += currDist;
                    avgSecond[j] += currDist;
                    secondTotalAvg += currDist;
                }
            }
        }
        // Correct the averages.
        float divisor = length * length;
        firstTotalAvg *= 2;
        secondTotalAvg *= 2;
        firstTotalAvg /= divisor;
        secondTotalAvg /= divisor;
        for (int i = 0; i < length; i++) {
            avgFirst[i] /= (double) (length);
            avgSecond[i] /= (double) (length);
        }
        double factFirst, factSecond;
        for (int i = 0; i < length; i++) {
            for (int j = i + 1; j < length; j++) {
                factFirst = dMatFirst.getElementAt(i, j) - avgFirst[i]
                        - avgFirst[j] + firstTotalAvg;
                factSecond = dMatSecond.getElementAt(i, j) - avgSecond[i]
                        - avgSecond[j] + secondTotalAvg;
                distanceCorr += factFirst * factSecond;
                if (DataMineConstants.isAcceptableDouble(factFirst)) {
                    distanceStdevFirst += factFirst * factFirst;
                }
                if (DataMineConstants.isAcceptableDouble(factSecond)) {
                    distanceStdevSecond += factSecond * factSecond;
                }
            }
        }
        distanceCorr *= 2;
        distanceStdevFirst *= 2;
        distanceStdevSecond *= 2;
        for (int i = 0; i < length; i++) {
            factFirst = dMatFirst.getElementAt(i, i) - avgFirst[i]
                    - avgFirst[i] + firstTotalAvg;
            factSecond = dMatSecond.getElementAt(i, i) - avgSecond[i]
                    - avgSecond[i] + secondTotalAvg;
            distanceCorr += factFirst * factSecond;
            if (DataMineConstants.isAcceptableDouble(factFirst)) {
                distanceStdevFirst += factFirst * factFirst;
            }
            if (DataMineConstants.isAcceptableDouble(factSecond)) {
                distanceStdevSecond += factSecond * factSecond;
            }
        }
        distanceCorr /= divisor;
        distanceCorr = Math.sqrt(distanceCorr);
        distanceStdevFirst /= divisor;
        distanceStdevSecond /= divisor;
        distanceStdevFirst = Math.sqrt(distanceStdevFirst);
        distanceStdevSecond = Math.sqrt(distanceStdevSecond);
        distanceCorr /= Math.sqrt(distanceStdevFirst * distanceStdevSecond);
        return (float) distanceCorr;
    }

    /**
     * Calculates the distance correlation.
     *
     * @param first Float array of values.
     * @param second Float array of values.
     * @return Distance correlation between the two arrays.
     * @throws Exception
     */
    public static float correlation(float[] first, float[] second)
            throws Exception {
        if (first == null || second == null || first.length != second.length) {
            return 0;
        }
        int length = first.length;
        SymmetricFloatMatrix dMatFirst = new SymmetricFloatMatrix(length);
        SymmetricFloatMatrix dMatSecond = new SymmetricFloatMatrix(length);
        double[] avgFirst = new double[length];
        double[] avgSecond = new double[length];
        double firstTotalAvg = 0;
        double secondTotalAvg = 0;
        double distanceStdevFirst = 0;
        double distanceStdevSecond = 0;
        double distanceCorr = 0;
        double currDist;
        for (int i = 0; i < length; i++) {
            for (int j = i + 1; j < length; j++) {
                currDist = Math.abs(first[i] - first[j]);
                dMatFirst.setElementAt(i, j, (float) currDist);
                if (DataMineConstants.isAcceptableDouble(currDist)) {
                    avgFirst[i] += currDist;
                    avgFirst[j] += currDist;
                    firstTotalAvg += currDist;
                }
                currDist = Math.abs(second[i] - second[j]);
                dMatSecond.setElementAt(i, j, (float) currDist);
                if (DataMineConstants.isAcceptableDouble(currDist)) {
                    avgSecond[i] += currDist;
                    avgSecond[j] += currDist;
                    secondTotalAvg += currDist;
                }
            }
        }
        // Correct the averages.
        float divisor = length * length;
        firstTotalAvg *= 2;
        secondTotalAvg *= 2;
        firstTotalAvg /= divisor;
        secondTotalAvg /= divisor;
        for (int i = 0; i < length; i++) {
            avgFirst[i] /= (float) (length);
            avgSecond[i] /= (float) (length);
        }
        double factFirst, factSecond;
        for (int i = 0; i < length; i++) {
            for (int j = i + 1; j < length; j++) {
                factFirst = dMatFirst.getElementAt(i, j) - avgFirst[i]
                        - avgFirst[j] + firstTotalAvg;
                factSecond = dMatSecond.getElementAt(i, j) - avgSecond[i]
                        - avgSecond[j] + secondTotalAvg;
                distanceCorr += factFirst * factSecond;
                if (DataMineConstants.isAcceptableDouble(factFirst)) {
                    distanceStdevFirst += factFirst * factFirst;
                }
                if (DataMineConstants.isAcceptableDouble(factSecond)) {
                    distanceStdevSecond += factSecond * factSecond;
                }
            }
        }
        distanceCorr *= 2;
        distanceStdevFirst *= 2;
        distanceStdevSecond *= 2;
        for (int i = 0; i < length; i++) {
            factFirst = dMatFirst.getElementAt(i, i) - avgFirst[i]
                    - avgFirst[i] + firstTotalAvg;
            factSecond = dMatSecond.getElementAt(i, i) - avgSecond[i]
                    - avgSecond[i] + secondTotalAvg;
            distanceCorr += factFirst * factSecond;
            if (DataMineConstants.isAcceptableDouble(factFirst)) {
                distanceStdevFirst += factFirst * factFirst;
            }
            if (DataMineConstants.isAcceptableDouble(factSecond)) {
                distanceStdevSecond += factSecond * factSecond;
            }
        }
        distanceCorr /= divisor;
        distanceStdevFirst /= divisor;
        distanceStdevSecond /= divisor;
        distanceStdevFirst = Math.sqrt(distanceStdevFirst);
        distanceStdevSecond = Math.sqrt(distanceStdevSecond);
        distanceCorr = Math.sqrt(distanceCorr);
        distanceCorr /= Math.sqrt(distanceStdevFirst * distanceStdevSecond);
        return (float) distanceCorr;
    }

    /**
     * Calculates the distance correlation.
     *
     * @param first Float array of values.
     * @param second Double array of values.
     * @return Distance correlation between the two arrays.
     * @throws Exception
     */
    public static float correlation(float[] first, double[] second)
            throws Exception {
        if (first == null || second == null || first.length != second.length) {
            return 0;
        }
        int length = first.length;
        SymmetricFloatMatrix dMatFirst = new SymmetricFloatMatrix(length);
        SymmetricFloatMatrix dMatSecond = new SymmetricFloatMatrix(length);
        double[] avgFirst = new double[length];
        double[] avgSecond = new double[length];
        double firstTotalAvg = 0;
        double secondTotalAvg = 0;
        double distanceStdevFirst = 0;
        double distanceStdevSecond = 0;
        double distanceCorr = 0;
        double currDist;
        for (int i = 0; i < length; i++) {
            for (int j = i + 1; j < length; j++) {
                currDist = Math.abs(first[i] - first[j]);
                dMatFirst.setElementAt(i, j, (float) currDist);
                if (DataMineConstants.isAcceptableDouble(currDist)) {
                    avgFirst[i] += currDist;
                    avgFirst[j] += currDist;
                    firstTotalAvg += currDist;
                }
                currDist = Math.abs(second[i] - second[j]);
                dMatSecond.setElementAt(i, j, (float) currDist);
                if (DataMineConstants.isAcceptableDouble(currDist)) {
                    avgSecond[i] += currDist;
                    avgSecond[j] += currDist;
                    secondTotalAvg += currDist;
                }
            }
        }
        // Correct the averages.
        float divisor = length * length;
        firstTotalAvg *= 2;
        secondTotalAvg *= 2;
        firstTotalAvg /= divisor;
        secondTotalAvg /= divisor;
        for (int i = 0; i < length; i++) {
            avgFirst[i] /= (float) (length);
            avgSecond[i] /= (float) (length);
        }
        double factFirst, factSecond;
        for (int i = 0; i < length; i++) {
            for (int j = i + 1; j < length; j++) {
                factFirst = dMatFirst.getElementAt(i, j) - avgFirst[i]
                        - avgFirst[j] + firstTotalAvg;
                factSecond = dMatSecond.getElementAt(i, j) - avgSecond[i]
                        - avgSecond[j] + secondTotalAvg;
                distanceCorr += factFirst * factSecond;
                if (DataMineConstants.isAcceptableDouble(factFirst)) {
                    distanceStdevFirst += factFirst * factFirst;
                }
                if (DataMineConstants.isAcceptableDouble(factSecond)) {
                    distanceStdevSecond += factSecond * factSecond;
                }
            }
        }
        distanceCorr *= 2;
        distanceStdevFirst *= 2;
        distanceStdevSecond *= 2;
        for (int i = 0; i < length; i++) {
            factFirst = dMatFirst.getElementAt(i, i) - avgFirst[i]
                    - avgFirst[i] + firstTotalAvg;
            factSecond = dMatSecond.getElementAt(i, i) - avgSecond[i]
                    - avgSecond[i] + secondTotalAvg;
            distanceCorr += factFirst * factSecond;
            if (DataMineConstants.isAcceptableDouble(factFirst)) {
                distanceStdevFirst += factFirst * factFirst;
            }
            if (DataMineConstants.isAcceptableDouble(factSecond)) {
                distanceStdevSecond += factSecond * factSecond;
            }
        }
        distanceCorr /= divisor;
        distanceCorr = Math.sqrt(distanceCorr);
        distanceStdevFirst /= divisor;
        distanceStdevSecond /= divisor;
        distanceStdevFirst = Math.sqrt(distanceStdevFirst);
        distanceStdevSecond = Math.sqrt(distanceStdevSecond);
        distanceCorr /= Math.sqrt(distanceStdevFirst * distanceStdevSecond);
        return (float) distanceCorr;
    }

    public static float correlation(DataSet dset, float[] first, float[] second)
            throws Exception {
        return correlation(first, second);
    }
}
