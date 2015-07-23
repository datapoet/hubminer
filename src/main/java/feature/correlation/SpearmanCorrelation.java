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

import java.util.Arrays;
import util.AuxSort;

/**
 * This class implements the Spearman correlation coefficient that is
 * essentially rank correlation and is capable of capturing some forms of
 * non-linear correlation.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class SpearmanCorrelation extends CorrelationCoefficient {

    /**
     * Calculates the Spearman correlation between value arrays.
     *
     * @param first Double value array.
     * @param second Double value array.
     * @return Double that is the Spearman correlation between the two arrays.
     * @throws Exception
     */
    public static double correlation(double[] first, double[] second)
            throws Exception {
        if (first == null || second == null || first.length != second.length) {
            return 0;
        }
        double[] firstCopy = Arrays.copyOf(first, first.length);
        double[] secondCopy = Arrays.copyOf(second, first.length);
        int[] firstSortedIndexes = AuxSort.sortIndexedValue(firstCopy, true);
        int[] secondSortedIndexes = AuxSort.sortIndexedValue(secondCopy, true);
        // Check for ties. Important.
        float[] actualRanksFirst = new float[first.length];
        float[] actualRanksSecond = new float[first.length];
        int index = 0;
        float sum;
        // The beginning of a tied group.
        int startingPoint;
        int numEqual;
        while (index < first.length) {
            sum = 0;
            numEqual = 1;
            startingPoint = index;
            while (index + 1 < first.length
                    && (firstCopy[index + 1] == firstCopy[index])) {
                sum += (index + 1);
                index++;
                numEqual++;
            }
            sum += (index + 1);
            sum /= (float) numEqual;
            for (int i = startingPoint; i < startingPoint + numEqual; i++) {
                actualRanksFirst[firstSortedIndexes[i]] = sum;
            }
            index++;
        }
        index = 0;
        while (index < second.length) {
            sum = 0;
            numEqual = 1;
            startingPoint = index;
            while (index + 1 < second.length
                    && (secondCopy[index + 1] == secondCopy[index])) {
                sum += (index + 1);
                index++;
                numEqual++;
            }
            sum += (index + 1);
            sum /= (float) numEqual;
            for (int i = startingPoint; i < startingPoint + numEqual; i++) {
                actualRanksSecond[secondSortedIndexes[i]] = sum;
            }
            index++;
        }
        // Now that the ranks have been processed, we calculate the Pearson
        // correlation between the rank arrays.
        return PearsonCorrelation.correlation(actualRanksFirst,
                actualRanksSecond);
    }

    /**
     * Calculates the Spearman correlation between value arrays.
     *
     * @param first Float value array.
     * @param second Float value array.
     * @return Double that is the Spearman correlation between the two arrays.
     * @throws Exception
     */
    public static float correlation(float[] first, float[] second)
            throws Exception {
        if (first == null || second == null || first.length != second.length) {
            return 0;
        }
        float[] firstCopy = Arrays.copyOf(first, first.length);
        float[] secondCopy = Arrays.copyOf(second, first.length);
        int[] firstSortedIndexes = AuxSort.sortIndexedValue(firstCopy, true);
        int[] secondSortedIndexes = AuxSort.sortIndexedValue(secondCopy, true);
        // Check for ties. Important.
        float[] actualRanksFirst = new float[first.length];
        float[] actualRanksSecond = new float[first.length];
        int index = 0;
        float sum;
        // The beginning of a tied group.
        int startingPoint;
        int numEqual;
        while (index < first.length) {
            sum = 0;
            numEqual = 1;
            startingPoint = index;
            while (index + 1 < first.length
                    && (firstCopy[index + 1] == firstCopy[index])) {
                sum += (index + 1);
                index++;
                numEqual++;
            }
            sum += (index + 1);
            sum /= (float) numEqual;
            for (int i = startingPoint; i < startingPoint + numEqual; i++) {
                actualRanksFirst[firstSortedIndexes[i]] = sum;
            }
            index++;
        }
        index = 0;
        while (index < second.length) {
            sum = 0;
            numEqual = 1;
            startingPoint = index;
            while (index + 1 < second.length
                    && (secondCopy[index + 1] == secondCopy[index])) {
                sum += (index + 1);
                index++;
                numEqual++;
            }
            sum += (index + 1);
            sum /= (float) numEqual;
            for (int i = startingPoint; i < startingPoint + numEqual; i++) {
                actualRanksSecond[secondSortedIndexes[i]] = sum;
            }
            index++;
        }
        // Now that the ranks have been processed, we calculate the Pearson
        // correlation between the rank arrays.
        return PearsonCorrelation.correlation(actualRanksFirst,
                actualRanksSecond);
    }

    /**
     * Calculates the Spearman correlation between value arrays.
     *
     * @param first Float value array.
     * @param second Double value array.
     * @return Double that is the Spearman correlation between the two arrays.
     * @throws Exception
     */
    public static float correlation(float[] first, double[] second)
            throws Exception {
        if (first == null || second == null || first.length != second.length) {
            return 0;
        }
        float[] firstCopy = Arrays.copyOf(first, first.length);
        double[] secondCopy = Arrays.copyOf(second, first.length);
        int[] firstSortedIndexes = AuxSort.sortIndexedValue(firstCopy, true);
        int[] secondSortedIndexes = AuxSort.sortIndexedValue(secondCopy, true);
        // Check for ties. Important.
        float[] actualRanksFirst = new float[first.length];
        float[] actualRanksSecond = new float[first.length];
        int index = 0;
        float sum;
        // The beginning of a tied group.
        int startingPoint;
        int numEqual;
        while (index < first.length) {
            sum = 0;
            numEqual = 1;
            startingPoint = index;
            while (index + 1 < first.length
                    && (firstCopy[index + 1] == firstCopy[index])) {
                sum += (index + 1);
                index++;
                numEqual++;
            }
            sum += (index + 1);
            sum /= (float) numEqual;
            for (int i = startingPoint; i < startingPoint + numEqual; i++) {
                actualRanksFirst[firstSortedIndexes[i]] = sum;
            }
            index++;
        }
        index = 0;
        while (index < second.length) {
            sum = 0;
            numEqual = 1;
            startingPoint = index;
            while (index + 1 < second.length
                    && (secondCopy[index + 1] == secondCopy[index])) {
                sum += (index + 1);
                index++;
                numEqual++;
            }
            sum += (index + 1);
            sum /= (float) numEqual;
            for (int i = startingPoint; i < startingPoint + numEqual; i++) {
                actualRanksSecond[secondSortedIndexes[i]] = sum;
            }
            index++;
        }
        // Now that the ranks have been processed, we calculate the Pearson
        // correlation between the rank arrays.
        return PearsonCorrelation.correlation(actualRanksFirst,
                actualRanksSecond);
    }
}
