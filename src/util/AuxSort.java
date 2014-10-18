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

import java.util.ArrayList;
import java.util.Comparator;

/**
 * This class implements utility sorting methods. Some of them might not be
 * needed anymore as they can be replaced by Collections.sort() in case of
 * ArrayList objects - but the useful ones are those that return the permutation
 * as an integer array, so that one can keep track which instance the sorted
 * property originated from.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class AuxSort {

    public static final boolean ASCENDING = false;
    public static final boolean DESCENDING = true;

    /**
     * Sorts the ArrayList.
     *
     * @param <K> Class of the objects in the list.
     * @param values Values to be sorted, given as ArrayList.
     * @param comp Comparator object.
     * @param descending True if descending, false if ascending.
     * @return The permutation of the sort, as an integer array.
     * @throws Exception
     */
    public static <K> int[] sortGenericArrayList(
            ArrayList<K> values,
            Comparator<K> comp,
            boolean descending) throws Exception {
        double factor = 1.3;
        int j;
        boolean exchange;
        int step = values.size();
        // This array will hold the permutation.
        int[] returnIndexes = new int[values.size()];
        // It is initialized to identity transformation.
        for (int i = 0; i < values.size(); i++) {
            returnIndexes[i] = i;
        }
        K temp;
        int tempInt;
        do {
            // Calculate the current sorting step.
            step = (int) ((double) step / factor);
            if (step < 1) {
                step = 1;
            }
            if ((step == 9) || (step == 10)) {
                step = 11;
            }
            exchange = false;
            for (int i = 0; i < values.size() - step; i++) {
                j = i + step;
                if (descending) {
                    if (comp.compare(values.get(i), values.get(j)) < 0) {
                        exchange = true;
                        temp = values.get(j);
                        values.set(j, values.get(i));
                        values.set(i, temp);
                        tempInt = returnIndexes[j];
                        returnIndexes[j] = returnIndexes[i];
                        returnIndexes[i] = tempInt;
                    }
                } else {
                    if (comp.compare(values.get(i), values.get(j)) > 0) {
                        exchange = true;
                        temp = values.get(j);
                        values.set(j, values.get(i));
                        values.set(i, temp);
                        tempInt = returnIndexes[j];
                        returnIndexes[j] = returnIndexes[i];
                        returnIndexes[i] = tempInt;
                    }
                }
            }
        } while (exchange || (step != 1));
        return returnIndexes;
    }

    /**
     * Sorts an ArrayList of integers.
     *
     * @param arr Values to be sorted, given as an integer ArrayList.
     * @param descending True if descending, false if ascending.
     * @throws Exception
     */
    public static void sortIntArrayList(
            ArrayList<Integer> arr, boolean descending) throws Exception {
        double factor = 1.3;
        int i;
        int j;
        Integer temp;
        boolean exchange;
        int step = arr.size();
        do {
            step = (int) ((double) step / factor);
            if (step < 1) {
                step = 1;
            }
            if ((step == 9) || (step == 10)) {
                step = 11;
            }
            exchange = false;
            for (i = 0; i < arr.size() - step; i++) {
                j = i + step;
                if (descending) {
                    if (arr.get(i) < arr.get(j)) {
                        temp = arr.get(i);
                        arr.set(i, arr.get(j));
                        arr.set(j, temp);
                        exchange = true;
                    }
                } else {
                    if (arr.get(i) > arr.get(j)) {
                        temp = arr.get(i);
                        arr.set(i, arr.get(j));
                        arr.set(j, temp);
                        exchange = true;
                    }
                }
            }
        } while (exchange || !(step == 1));
    }

    /**
     * Sorts an ArrayList of floats.
     *
     * @param vector Values to be sorted, given as a float ArrayList.
     * @param descending True if descending, false if ascending.
     * @throws Exception
     */
    public static void sortFloatArrayList(
            ArrayList<Float> arr, boolean descending) throws Exception {
        double factor = 1.3;
        int i;
        int j;
        Float temp;
        boolean exchange;
        int step = arr.size();
        do {
            step = (int) ((double) step / factor);
            if (step < 1) {
                step = 1;
            }
            if ((step == 9) || (step == 10)) {
                step = 11;
            }
            exchange = false;
            for (i = 0; i < arr.size() - step; i++) {
                j = i + step;
                if (descending) {
                    if (arr.get(i) < arr.get(j)) {
                        temp = arr.get(i);
                        arr.set(i, arr.get(j));
                        arr.set(j, temp);
                        exchange = true;
                    }
                } else {
                    if (arr.get(i) > arr.get(j)) {
                        temp = arr.get(i);
                        arr.set(i, arr.get(j));
                        arr.set(j, temp);
                        exchange = true;
                    }
                }
            }
        } while (exchange || !(step == 1));
    }

    /**
     * Sorts the array.
     *
     * @param values Values given as a double array.
     * @param descending True if descending, false if ascending.
     * @return An integer array representing the final permutation.
     * @throws Exception
     */
    public static int[] sortIndexedValue(
            double[] values, boolean descending) throws Exception {
        double factor = 1.3;
        int j;
        boolean exchange;
        int step = values.length;
        // This array will hold the permutation.
        int[] returnIndexes = new int[values.length];
        // It is initialized to identity transformation.
        for (int i = 0; i < values.length; i++) {
            returnIndexes[i] = i;
        }
        double temp;
        int tempInt;
        do {
            step = (int) ((double) step / factor);
            if (step < 1) {
                step = 1;
            }
            if ((step == 9) || (step == 10)) {
                step = 11;
            }
            exchange = false;
            for (int i = 0; i < values.length - step; i++) {
                j = i + step;
                if (descending) {
                    if (values[j] > values[i]) {
                        exchange = true;
                        temp = values[j];
                        values[j] = values[i];
                        values[i] = temp;
                        tempInt = returnIndexes[j];
                        returnIndexes[j] = returnIndexes[i];
                        returnIndexes[i] = tempInt;
                    }
                } else {
                    if (values[j] < values[i]) {
                        exchange = true;
                        temp = values[j];
                        values[j] = values[i];
                        values[i] = temp;
                        tempInt = returnIndexes[j];
                        returnIndexes[j] = returnIndexes[i];
                        returnIndexes[i] = tempInt;
                    }
                }
            }
        } while (exchange || (step != 1));
        return returnIndexes;
    }

    /**
     * Sorts the array.
     *
     * @param values Values given as a float array.
     * @param descending True if descending, false if ascending.
     * @return An integer array representing the final permutation.
     * @throws Exception
     */
    public static int[] sortIndexedValue(
            float[] values, boolean descending) throws Exception {
        double factor = 1.3;
        int j;
        boolean exchange;
        int step = values.length;
        // This array will hold the permutation.
        int[] returnIndexes = new int[values.length];
        // It is initialized to identity transformation.
        for (int i = 0; i < values.length; i++) {
            returnIndexes[i] = i;
        }
        float temp;
        int tempInt;
        do {
            step = (int) ((double) step / factor);
            if (step < 1) {
                step = 1;
            }
            if ((step == 9) || (step == 10)) {
                step = 11;
            }
            exchange = false;
            for (int i = 0; i < values.length - step; i++) {
                j = i + step;
                if (descending) {
                    if (values[j] > values[i]) {
                        exchange = true;
                        temp = values[j];
                        values[j] = values[i];
                        values[i] = temp;
                        tempInt = returnIndexes[j];
                        returnIndexes[j] = returnIndexes[i];
                        returnIndexes[i] = tempInt;
                    }
                } else {
                    if (values[j] < values[i]) {
                        exchange = true;
                        temp = values[j];
                        values[j] = values[i];
                        values[i] = temp;
                        tempInt = returnIndexes[j];
                        returnIndexes[j] = returnIndexes[i];
                        returnIndexes[i] = tempInt;
                    }
                }
            }
        } while (exchange || (step != 1));
        return returnIndexes;
    }

    /**
     * Sorts the array.
     *
     * @param values Values given as an integer array.
     * @param descending True if descending, false if ascending.
     * @return An integer array representing the final permutation.
     * @throws Exception
     */
    public static int[] sortIndexedValue(
            int[] values, boolean descending) throws Exception {
        double factor = 1.3;
        int j;
        boolean exchange;
        int step = values.length;
        // This array will hold the permutation.
        int[] returnIndexes = new int[values.length];
        // It is initialized to identity transformation.
        for (int i = 0; i < values.length; i++) {
            returnIndexes[i] = i;
        }
        int temp;
        int tempIndex;
        do {
            step = (int) ((double) step / factor);
            if (step < 1) {
                step = 1;
            }
            if ((step == 9) || (step == 10)) {
                step = 11;
            }
            exchange = false;
            for (int i = 0; i < values.length - step; i++) {
                j = i + step;
                if (descending) {
                    if (values[j] > values[i]) {
                        exchange = true;
                        temp = values[j];
                        values[j] = values[i];
                        values[i] = temp;
                        tempIndex = returnIndexes[j];
                        returnIndexes[j] = returnIndexes[i];
                        returnIndexes[i] = tempIndex;
                    }
                } else {
                    if (values[j] < values[i]) {
                        exchange = true;
                        temp = values[j];
                        values[j] = values[i];
                        values[i] = temp;
                        tempIndex = returnIndexes[j];
                        returnIndexes[j] = returnIndexes[i];
                        returnIndexes[i] = tempIndex;
                    }
                }
            }
        } while (exchange || (step != 1));
        return returnIndexes;
    }

    /**
     * Sorts the array.
     *
     * @param values Values given as an integer ArrayList.
     * @param descending True if descending, false if ascending.
     * @return An integer array representing the final permutation.
     * @throws Exception
     */
    public static int[] sortIIndexedValue(
            ArrayList<Integer> values, boolean descending) throws Exception {
        double factor = 1.3;
        int j;
        boolean exchange;
        int step = values.size();
        // This array will hold the permutation.
        int[] returnIndexes = new int[values.size()];
        // It is initialized to identity transformation.
        for (int i = 0; i < values.size(); i++) {
            returnIndexes[i] = i;
        }
        int temp;
        int tempIndex;
        do {
            step = (int) ((double) step / factor);
            if (step < 1) {
                step = 1;
            }
            if ((step == 9) || (step == 10)) {
                step = 11;
            }
            exchange = false;
            for (int i = 0; i < values.size() - step; i++) {
                j = i + step;
                if (descending) {
                    if (values.get(j) > values.get(i)) {
                        exchange = true;
                        temp = values.get(j);
                        values.set(j, values.get(i));
                        values.set(i, temp);
                        tempIndex = returnIndexes[j];
                        returnIndexes[j] = returnIndexes[i];
                        returnIndexes[i] = tempIndex;
                    }
                } else {
                    if (values.get(j) < values.get(i)) {
                        exchange = true;
                        temp = values.get(j);
                        values.set(j, values.get(i));
                        values.set(i, temp);
                        tempIndex = returnIndexes[j];
                        returnIndexes[j] = returnIndexes[i];
                        returnIndexes[i] = tempIndex;
                    }
                }
            }
        } while (exchange || (step != 1));
        return returnIndexes;
    }

    /**
     * Sorts the array.
     *
     * @param values Values given as a double ArrayList.
     * @param descending True if descending, false if ascending.
     * @return An integer array representing the final permutation.
     * @throws Exception
     */
    public static int[] sortDIndexedValue(
            ArrayList<Double> values, boolean descending) throws Exception {
        double factor = 1.3;
        int j;
        boolean exchange;
        int step = values.size();
        // This array will hold the permutation.
        int[] returnIndexes = new int[values.size()];
        // It is initialized to identity transformation.
        for (int i = 0; i < values.size(); i++) {
            returnIndexes[i] = i;
        }
        double temp;
        int tempIndex;
        do {
            step = (int) ((double) step / factor);
            if (step < 1) {
                step = 1;
            }
            if ((step == 9) || (step == 10)) {
                step = 11;
            }
            exchange = false;
            for (int i = 0; i < values.size() - step; i++) {
                j = i + step;
                if (descending) {
                    if (values.get(j) > values.get(i)) {
                        exchange = true;
                        temp = values.get(j);
                        values.set(j, values.get(i));
                        values.set(i, temp);
                        tempIndex = returnIndexes[j];
                        returnIndexes[j] = returnIndexes[i];
                        returnIndexes[i] = tempIndex;
                    }
                } else {
                    if (values.get(j) < values.get(i)) {
                        exchange = true;
                        temp = values.get(j);
                        values.set(j, values.get(i));
                        values.set(i, temp);
                        tempIndex = returnIndexes[j];
                        returnIndexes[j] = returnIndexes[i];
                        returnIndexes[i] = tempIndex;
                    }
                }
            }
        } while (exchange || (step != 1));
        return returnIndexes;
    }

    /**
     * Sorts the array.
     *
     * @param values Values given as a float ArrayList.
     * @param descending True if descending, false if ascending.
     * @return An integer array representing the final permutation.
     * @throws Exception
     */
    public static int[] sortFIndexedValue(
            ArrayList<Float> values, boolean descending) throws Exception {
        double factor = 1.3;
        int j;
        boolean exchange;
        int step = values.size();
        // This array will hold the permutation.
        int[] returnIndexes = new int[values.size()];
        // It is initialized to identity transformation.
        for (int i = 0; i < values.size(); i++) {
            returnIndexes[i] = i;
        }
        float temp;
        int tempIndex;
        do {
            step = (int) ((double) step / factor);
            if (step < 1) {
                step = 1;
            }
            if ((step == 9) || (step == 10)) {
                step = 11;
            }
            exchange = false;
            for (int i = 0; i < values.size() - step; i++) {
                j = i + step;
                if (descending) {
                    if (values.get(j) > values.get(i)) {
                        exchange = true;
                        temp = values.get(j);
                        values.set(j, values.get(i));
                        values.set(i, temp);
                        tempIndex = returnIndexes[j];
                        returnIndexes[j] = returnIndexes[i];
                        returnIndexes[i] = tempIndex;
                    }
                } else {
                    if (values.get(j) < values.get(i)) {
                        exchange = true;
                        temp = values.get(j);
                        values.set(j, values.get(i));
                        values.set(i, temp);
                        tempIndex = returnIndexes[j];
                        returnIndexes[j] = returnIndexes[i];
                        returnIndexes[i] = tempIndex;
                    }
                }
            }
        } while (exchange || (step != 1));
        return returnIndexes;
    }

    /**
     * For sorting in text-mode (deprecated).
     *
     * @param a A matrix of strings. Objects in rows, quantities in columns.
     * @param index Index of the quantity that is to be sorted.
     * @param descending True if descending, false if ascending.
     * @throws Exception
     */
    public static void sortTxt(
            String[][] a, int index, boolean descending) throws Exception {
        double factor = 1.3;
        int j;
        boolean exchange;
        int step = a.length;
        String[] temp = new String[a[0].length];
        do {
            step = (int) ((double) step / factor);
            if (step < 1) {
                step = 1;
            }
            if ((step == 9) || (step == 10)) {
                step = 11;
            }
            exchange = false;
            for (int i = 0; i < a.length - step; i++) {
                j = i + step;
                if (descending) {
                    if ((new Integer(a[j][index])).
                            compareTo(new Integer(a[i][index])) > 0) {
                        exchange = true;
                        for (int k = 0; k < temp.length; k++) {
                            temp[k] = a[j][k];
                            a[j][k] = a[i][k];
                            a[i][k] = temp[k];
                        }
                    }
                } else {
                    if ((new Integer(a[j][index])).
                            compareTo(new Integer(a[i][index])) < 0) {
                        exchange = true;
                        for (int k = 0; k < temp.length; k++) {
                            temp[k] = a[j][k];
                            a[j][k] = a[i][k];
                            a[i][k] = temp[k];
                        }
                    }
                }
            }
        } while (exchange || (step != 1));
    }
}