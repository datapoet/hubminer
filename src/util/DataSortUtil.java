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

import data.representation.DataInstance;
import data.representation.util.DataMineConstants;
import java.util.ArrayList;
import java.util.Comparator;

/**
 * A utility class for sorting DataInstance objects.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 * @extends DistanceMeasure
 */
public class DataSortUtil {

    /**
     * A quicksort implementation for ArrayList-s of DataInstance-s.
     *
     * @param data ArrayList of DataInstances.
     * @param dataComparator Data comparator.
     * @param lower Lower range index.
     * @param upper Upper range index.
     * @throws Exception
     */
    static void quicksort(
            ArrayList<DataInstance> data,
            Comparator dataComparator,
            int lower,
            int upper) throws Exception {
        int i = lower;
        int j = upper;
        if (data == null || dataComparator == null) {
            return;
        }
        DataInstance temp;
        DataInstance half = data.get((lower + upper) / 2);
        do {
            while (dataComparator.compare(data.get(i), half)
                    < (-DataMineConstants.EPSILON)) {
                i++;
            }
            while (dataComparator.compare(data.get(j), half) > 0) {
                j--;
            }
            if (i <= j) {
                temp = data.get(i);
                data.set(i, data.get(j));
                data.set(j, temp);
                i++;
                j--;
            }
        } while (i <= j);
        if (lower < j) {
            quicksort(data, dataComparator, lower, j);
        }
        if (i < upper) {
            quicksort(data, dataComparator, i, upper);
        }
    }

    /**
     * A quicksort implementation for ArrayList-s of DataInstance-s.
     *
     * @param data ArrayList of DataInstance-s.
     * @param dataComparator Comparator to be used for sorting.
     * @throws Exception
     */
    public static void sort(ArrayList data, Comparator dataComparator)
            throws Exception {
        if ((data != null) && (dataComparator != null)) {
            quicksort(data, dataComparator, 0, data.size() - 1);
        }
    }
}
