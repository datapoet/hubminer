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

import data.representation.util.DataMineConstants;
import java.util.ArrayList;

/**
 * A utility class for working with arrays of measurements.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class ArrayUtil {

    /**
     * Performs z-standardization of a float array.
     *
     * @param arr An array of float values.
     */
    public static void zStandardize(float[] arr) {
        if (arr == null || arr.length == 0) {
            return;
        }
        float mean = findMean(arr);
        float stDev = findStdev(arr, mean);
        for (int i = 0; i < arr.length; i++) {
            arr[i] = (arr[i] - mean) / stDev;
        }
    }

    /**
     * Performs z-standardization of a double array.
     *
     * @param arr An array of double values.
     */
    public static void zStandardize(double[] arr) {
        if (arr == null || arr.length == 0) {
            return;
        }
        double mean = findMean(arr);
        double stDev = findStdev(arr, mean);
        for (int i = 0; i < arr.length; i++) {
            arr[i] = (arr[i] - mean) / stDev;
        }
    }

    /**
     * Finds a mean of a float array.
     *
     * @param arr An array of float values.
     * @return Mean value.
     */
    public static float findMean(float[] arr) {
        if (arr == null || arr.length == 0) {
            return 0;
        }
        float mean;
        double totalSum = 0;
        for (int i = 0; i < arr.length; i++) {
            totalSum += arr[i];
        }
        mean = (float) (totalSum / (float) arr.length);
        return mean;
    }

    /**
     * Finds a mean of a double array.
     *
     * @param arr An array of double values.
     * @return Mean value.
     */
    public static double findMean(double[] arr) {
        if (arr == null || arr.length == 0) {
            return 0;
        }
        double mean;
        double totalSum = 0;
        for (int i = 0; i < arr.length; i++) {
            totalSum += arr[i];
        }
        mean = (double) (totalSum / (double) arr.length);
        return mean;
    }

    /**
     * Finds a mean of a float ArrayList.
     *
     * @param arr An array of float values.
     * @return Mean value.
     */
    public static float findMean(ArrayList<Float> arr) {
        if (arr == null || arr.isEmpty()) {
            return 0;
        }
        float mean;
        double totalSum = 0;
        for (int i = 0; i < arr.size(); i++) {
            totalSum += arr.get(i);
        }
        mean = (float) (totalSum / (float) arr.size());
        return mean;
    }

    /**
     * Finds a standard deviation of a float array.
     *
     * @param arr An array of float values.
     * @param mean Standard deviation.
     * @return
     */
    public static float findStdev(float[] arr, float mean) {
        if (arr == null || arr.length == 0) {
            return 0;
        }
        double totalSum = 0;
        for (int i = 0; i < arr.length; i++) {
            totalSum += (arr[i] - mean) * (arr[i] - mean);
        }
        totalSum /= (double) arr.length;
        return ((float) (Math.sqrt(totalSum)));
    }

    /**
     * Finds a standard deviation of a double array.
     *
     * @param arr An array of double values.
     * @param mean Standard deviation.
     * @return
     */
    public static double findStdev(double[] arr, double mean) {
        if (arr == null || arr.length == 0) {
            return 0;
        }
        double totalSum = 0;
        for (int i = 0; i < arr.length; i++) {
            totalSum += (arr[i] - mean) * (arr[i] - mean);
        }
        totalSum /= (double) arr.length;
        return Math.sqrt(totalSum);
    }

    /**
     * Standardizes an array of measurements into a new array instance.
     *
     * @param arr array of float values.
     * @return A standardized copy of the original array.
     */
    public static float[] standardizeAndCopyArray(float[] arr) {
        if (arr == null || arr.length == 0) {
            return arr;
        }
        float mean = findMean(arr);
        float stDev = findStdev(arr, mean);
        float[] res = new float[arr.length];
        if (stDev != 0) {
            for (int i = 0; i < arr.length; i++) {
                res[i] = (arr[i] - mean) / stDev;
            }
        }
        return res;
    }

    /**
     * Standardizes an array of measurements into a new array instance.
     *
     * @param arr array of double values.
     * @return A standardized copy of the original array.
     */
    public static double[] standardizeAndCopyArray(double[] arr) {
        if (arr == null || arr.length == 0) {
            return arr;
        }
        double mean = findMean(arr);
        double stDev = findStdev(arr, mean);
        double[] res = new double[arr.length];
        if (stDev != 0) {
            for (int i = 0; i < arr.length; i++) {
                res[i] = (arr[i] - mean) / stDev;
            }
        }
        return res;
    }

    /**
     * Checks if an array is already sorted.
     *
     * @param fArr An array of float values.
     * @param ascending A boolean value set to true if the desired order is
     * ascending, false if checking for a descending order.
     * @return
     */
    public static boolean checkIfSorted(float[] fArr, boolean ascending) {
        if (fArr == null || fArr.length == 0) {
            return true;
        }
        if (fArr.length == 1) {
            return true;
        }
        for (int i = 0; i < fArr.length - 1; i++) {
            if (ascending) {
                if (fArr[i] > fArr[i + 1]) {
                    return false;
                }
            } else {
                if (fArr[i] < fArr[i + 1]) {
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * Checks if an array is already sorted.
     *
     * @param fArr An array of double values.
     * @param ascending A boolean value set to true if the desired order is
     * ascending, false if checking for a descending order.
     * @return
     */
    public static boolean checkIfSorted(double[] dArr, boolean ascending) {
        if (dArr == null || dArr.length == 0) {
            return true;
        }
        if (dArr.length == 1) {
            return true;
        }
        for (int i = 0; i < dArr.length - 1; i++) {
            if (ascending) {
                if (dArr[i] > dArr[i + 1]) {
                    return false;
                }
            } else {
                if (dArr[i] < dArr[i + 1]) {
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * Binary search.
     *
     * @param searchValue Integer value.
     * @param probArray Array to perform the search in.
     * @return Index where the value is found or -1 otherwise.
     */
    public static int findIndex(int searchValue, ArrayList<Integer> probArray) {
        if (probArray == null || probArray.isEmpty()) {
            return -1;
        }
        return findIndex(searchValue, probArray, 0, probArray.size() - 1);
    }

    /**
     * Binary search.
     *
     * @param searchValue Integer value.
     * @param probArray Array to perform the search in.
     * @param first Lower bound.
     * @param second Upper bound.
     * @return Index where the value is found or -1 otherwise.
     */
    private static int findIndex(
            int searchValue,
            ArrayList<Integer> probArray,
            int first,
            int second) {
        if (second - first <= 1) {
            if (probArray.get(second) == searchValue) {
                return second;
            } else {
                return -1;
            }
        }
        int middle = (first + second) / 2;
        if (probArray.get(middle) < searchValue) {
            return findIndex(searchValue, probArray, middle, second);
        } else {
            return findIndex(searchValue, probArray, first, middle);
        }
    }

    /**
     * Binary search.
     *
     * @param searchValue Integer value.
     * @param probArray Array to perform the search in.
     * @return Index where the value is found or of the bucket where it would
     * fall into.
     */
    public static int findIndex(int searchValue, int[] probArray) {
        if (probArray == null || probArray.length == 0) {
            return -1;
        }
        return findIndex(searchValue, probArray, 0, probArray.length - 1);
    }

    /**
     * Binary search.
     *
     * @param searchValue Integer value.
     * @param probArray Array to perform the search in.
     * @param first Lower bound.
     * @param second Upper bound.
     * @return Index where the value is found or of the bucket where it would
     * fall into.
     */
    private static int findIndex(
            int searchValue,
            int[] probArray,
            int first,
            int second) {
        if (second - first <= 1) {
            // First isn't, so it must be the second.
            return second;
        }
        int middle = (first + second) / 2;
        if (probArray[middle] < searchValue) {
            return findIndex(searchValue, probArray, middle, second);
        } else {
            return findIndex(searchValue, probArray, first, middle);
        }
    }

    /**
     * Binary search.
     *
     * @param searchValue Float value.
     * @param probArray Array to perform the search in.
     * @return Index where the value is found or of the bucket where it would
     * fall into.
     */
    public static int findIndex(float searchValue, float[] probArray) {
        if (probArray == null || probArray.length == 0) {
            return -1;
        }
        return findIndex(searchValue, probArray, 0, probArray.length - 1);
    }

    /**
     * Binary search.
     *
     * @param searchValue Float value.
     * @param probArray Array to perform the search in.
     * @param first Lower bound.
     * @param second Upper bound.
     * @return Index where the value is found or of the bucket where it would
     * fall into.
     */
    private static int findIndex(
            float searchValue,
            float[] probArray,
            int first,
            int second) {
        if (second - first <= 1) {
            // First isn't, so it must be the second.
            return second;
        }
        int middle = (first + second) / 2;
        if (probArray[middle] < searchValue) {
            return findIndex(searchValue, probArray, middle, second);
        } else {
            return findIndex(searchValue, probArray, first, middle);
        }
    }

    /**
     * Binary search.
     *
     * @param searchValue Double value.
     * @param probArray Array to perform the search in.
     * @return Index where the value is found or of the bucket where it would
     * fall into.
     */
    public static int findIndex(double searchValue, double[] probArray) {
        if (probArray == null || probArray.length == 0) {
            return -1;
        }
        return findIndex(searchValue, probArray, 0, probArray.length - 1);
    }

    /**
     * Binary search.
     *
     * @param searchValue Double value.
     * @param probArray Array to perform the search in.
     * @param first Lower bound.
     * @param second Upper bound.
     * @return Index where the value is found or of the bucket where it would
     * fall into.
     */
    private static int findIndex(
            double searchValue,
            double[] probArray,
            int first,
            int second) {
        if (second - first <= 1) {
            // First isn't, so it must be the second.
            return second;
        }
        int middle = (first + second) / 2;
        if (probArray[middle] < searchValue) {
            return findIndex(searchValue, probArray, middle, second);
        } else {
            return findIndex(searchValue, probArray, first, middle);
        }
    }

    /**
     * @param arr Integer array.
     * @return Minimum value.
     */
    public static int min(int[] arr) {
        if (arr == null || arr.length == 0) {
            return -1;
        } else {
            int min = Integer.MAX_VALUE;
            for (int el : arr) {
                if (el < min) {
                    min = el;
                }
            }
            return min;
        }
    }

    /**
     * @param arr Integer array.
     * @return Sum of the elements in the array.
     */
    public static double sum(int[] arr) {
        if (arr == null || arr.length == 0) {
            return 0;
        } else {
            double sum = 0;
            for (int el : arr) {
                if (DataMineConstants.isAcceptableInt(el)) {
                    sum += el;
                }
            }
            return sum;
        }
    }

    /**
     * @param arr Float array.
     * @return Sum of the elements in the array.
     */
    public static double sum(float[] arr) {
        if (arr == null || arr.length == 0) {
            return 0;
        } else {
            double sum = 0;
            for (float el : arr) {
                if (DataMineConstants.isAcceptableFloat(el)) {
                    sum += el;
                }
            }
            return sum;
        }
    }

    /**
     * @param arr Double array.
     * @return Sum of the elements in the array.
     */
    public static double sum(double[] arr) {
        if (arr == null || arr.length == 0) {
            return 0;
        } else {
            double sum = 0;
            for (double el : arr) {
                if (DataMineConstants.isAcceptableDouble(el)) {
                    sum += el;
                }
            }
            return sum;
        }
    }

    /**
     * First return value is the value itself, second is its index.
     *
     * @param arr Integer array.
     * @return A two element array, where the first return value is the minimum
     * of the array that was passed to the method and the second is its index.
     */
    public static int[] minWithIndex(int[] arr) {
        int[] res = new int[2];
        if (arr == null || arr.length == 0) {
            res[0] = -1;
            res[1] = -1;
            return res;
        } else {
            res[0] = Integer.MAX_VALUE;
            for (int i = 0; i < arr.length; i++) {
                if (!DataMineConstants.isAcceptableInt(arr[i])) {
                    continue;
                }
                if (arr[i] < res[0]) {
                    res[0] = arr[i];
                    res[1] = i;
                }
            }
            return res;
        }
    }

    /**
     *
     * @param arr Integer ArrayList.
     * @return Minimum value.
     */
    public static int min(ArrayList<Integer> arr) {
        if (arr == null || arr.isEmpty()) {
            return -1;
        } else {
            int min = Integer.MAX_VALUE;
            for (int el : arr) {
                if (!DataMineConstants.isAcceptableInt(el)) {
                    continue;
                }
                if (el < min) {
                    min = el;
                }
            }
            return min;
        }
    }

    /**
     *
     * @param arr Integer ArrayList.
     * @return Maximum value.
     */
    public static int max(ArrayList<Integer> arr) {
        if (arr == null || arr.isEmpty()) {
            return -1;
        } else {
            int max = Integer.MIN_VALUE;
            for (int el : arr) {
                if (!DataMineConstants.isAcceptableInt(el)) {
                    continue;
                }
                if (el > max) {
                    max = el;
                }
            }
            return max;
        }
    }
    
    /**
     *
     * @param arr Float ArrayList.
     * @return Maximum value.
     */
    public static float maxOfFloatList(ArrayList<Float> arr) {
        if (arr == null || arr.isEmpty()) {
            return -1;
        } else {
            float max = -Float.MAX_VALUE;
            for (float el : arr) {
                if (!DataMineConstants.isAcceptableFloat(el)) {
                    continue;
                }
                if (el > max) {
                    max = el;
                }
            }
            return max;
        }
    }
    
    /**
     *
     * @param arr Float ArrayList.
     * @return Minimum value.
     */
    public static float minOfFloatList(ArrayList<Float> arr) {
        if (arr == null || arr.isEmpty()) {
            return -1;
        } else {
            float min = Float.MAX_VALUE;
            for (float el : arr) {
                if (!DataMineConstants.isAcceptableFloat(el)) {
                    continue;
                }
                if (el < min) {
                    min = el;
                }
            }
            return min;
        }
    }

    /**
     *
     * @param arr Integer array.
     * @return Maximum value.
     */
    public static int max(int[] arr) {
        if (arr == null || arr.length == 0) {
            return -1;
        } else {
            int max = Integer.MIN_VALUE;
            for (int el : arr) {
                if (!DataMineConstants.isAcceptableInt(el)) {
                    continue;
                }
                if (el > max) {
                    max = el;
                }
            }
            return max;
        }
    }

    /**
     * First return value is the value itself, second is its index.
     *
     * @param arr Integer array.
     * @return A two element array, where the first return value is the maximum
     * of the array that was passed to the method and the second is its index.
     */
    public static int[] maxWithIndex(int[] arr) {
        int[] res = new int[2];
        if (arr == null || arr.length == 0) {
            res[0] = -1;
            res[1] = -1;
            return res;
        } else {
            res[0] = Integer.MIN_VALUE;
            for (int i = 0; i < arr.length; i++) {
                if (!DataMineConstants.isAcceptableInt(arr[i])) {
                    continue;
                }
                if (arr[i] > res[0]) {
                    res[0] = arr[i];
                    res[1] = i;
                }
            }
            return res;
        }
    }

    /**
     * @param arr An array of float values.
     * @return Minimum of the array.
     */
    public static float min(float[] arr) {
        if (arr == null || arr.length == 0) {
            return -1;
        } else {
            float min = Float.MAX_VALUE;
            for (float el : arr) {
                if (!DataMineConstants.isAcceptableFloat(el)) {
                    continue;
                }
                if (el < min) {
                    min = el;
                }
            }
            return min;
        }
    }

    /**
     * First return value is the value itself, second is its index.
     *
     * @param arr Float array.
     * @return A two element array, where the first return value is the minimum
     * of the array that was passed to the method and the second is its index.
     */
    public static float[] minWithIndex(float[] arr) {
        float[] res = new float[2];
        if (arr == null || arr.length == 0) {
            res[0] = -1;
            res[1] = -1;
            return res;
        } else {
            res[0] = Float.MAX_VALUE;
            for (int i = 0; i < arr.length; i++) {
                if (!DataMineConstants.isAcceptableFloat(arr[i])) {
                    continue;
                }
                if (arr[i] < res[0]) {
                    res[0] = arr[i];
                    res[1] = i;
                }
            }
            return res;
        }
    }

    /**
     * @param arr Float array.
     * @return Maximum value.
     */
    public static float max(float[] arr) {
        if (arr == null || arr.length == 0) {
            return -1;
        } else {
            float max = -Float.MAX_VALUE;
            for (float el : arr) {
                if (!DataMineConstants.isAcceptableFloat(el)) {
                    continue;
                }
                if (el != Float.MAX_VALUE && el > max) {
                    max = el;
                }
            }
            return max;
        }
    }

    /**
     * First return value is the value itself, second is its index.
     *
     * @param arr Float array.
     * @return A two element array, where the first return value is the maximum
     * of the array that was passed to the method and the second is its index.
     */
    public static float[] maxWithIndex(float[] arr) {
        float[] res = new float[2];
        if (arr == null || arr.length == 0) {
            res[0] = -1;
            res[1] = -1;
            return res;
        } else {
            res[0] = -Float.MAX_VALUE;
            for (int i = 0; i < arr.length; i++) {
                if (!DataMineConstants.isAcceptableFloat(arr[i])) {
                    continue;
                }
                if (arr[i] != Float.MAX_VALUE && arr[i] > res[0]) {
                    res[0] = arr[i];
                    res[1] = i;
                }
            }
            return res;
        }
    }

    /**
     * @param arr An array of double values.
     * @return Minimum value.
     */
    public static double min(double[] arr) {
        if (arr == null || arr.length == 0) {
            return -1;
        } else {
            double min = Double.MAX_VALUE;
            for (double el : arr) {
                if (!DataMineConstants.isAcceptableDouble(el)) {
                    continue;
                }
                if (el < min) {
                    min = el;
                }
            }
            return min;
        }
    }

    /**
     * First return value is the value itself, second is its index.
     *
     * @param arr Double array.
     * @return A two element array, where the first return value is the minimum
     * of the array that was passed to the method and the second is its index.
     */
    public static double[] minWithIndex(double[] arr) {
        double[] res = new double[2];
        if (arr == null || arr.length == 0) {
            res[0] = -1;
            res[1] = -1;
            return res;
        } else {
            res[0] = Double.MAX_VALUE;
            for (int i = 0; i < arr.length; i++) {
                if (!DataMineConstants.isAcceptableDouble(arr[i])) {
                    continue;
                }
                if (arr[i] < res[0]) {
                    res[0] = arr[i];
                    res[1] = i;
                }
            }
            return res;
        }
    }

    public static double max(double[] arr) {
        if (arr == null || arr.length == 0) {
            return -1;
        } else {
            double max = -Double.MAX_VALUE;
            for (double el : arr) {
                if (el != Float.MAX_VALUE && el > max) {
                    max = el;
                }
            }
            return max;
        }
    }

    /**
     * First return value is the value itself, second is its index.
     *
     * @param arr Double array.
     * @return A two element array, where the first return value is the maximum
     * of the array that was passed to the method and the second is its index.
     */
    public static double[] maxWithIndex(double[] arr) {
        double[] res = new double[2];
        if (arr == null || arr.length == 0) {
            res[0] = -1;
            res[1] = -1;
            return res;
        } else {
            res[0] = -Double.MAX_VALUE;
            for (int i = 0; i < arr.length; i++) {
                if (!DataMineConstants.isAcceptableDouble(arr[i])) {
                    continue;
                }
                if (arr[i] > res[0]) {
                    res[0] = arr[i];
                    res[1] = i;
                }
            }
            return res;
        }
    }
    
    /**
     * This method returns the index of the maximum value..
     *
     * @param arr Double array.
     * @return Integer that is the index of the maximum value.
     */
    public static int indexOfMax(double[] arr) {
        if (arr == null || arr.length == 0) {
            return -1;
        } else {
            double maxVal = -Double.MAX_VALUE;
            int maxIndex = -1;
            for (int i = 0; i < arr.length; i++) {
                if (!DataMineConstants.isAcceptableDouble(arr[i])) {
                    continue;
                }
                if (arr[i] > maxVal) {
                    maxVal = arr[i];
                    maxIndex = i;
                }
            }
            return maxIndex;
        }
    }
    
    /**
     * This method returns the index of the maximum value..
     *
     * @param arr Float array.
     * @return Integer that is the index of the maximum value.
     */
    public static int indexOfMax(float[] arr) {
        if (arr == null || arr.length == 0) {
            return -1;
        } else {
            float maxVal = -Float.MAX_VALUE;
            int maxIndex = -1;
            for (int i = 0; i < arr.length; i++) {
                if (!DataMineConstants.isAcceptableFloat(arr[i])) {
                    continue;
                }
                if (arr[i] > maxVal) {
                    maxVal = arr[i];
                    maxIndex = i;
                }
            }
            return maxIndex;
        }
    }
    
    /**
     * This method returns the index of the maximum value..
     *
     * @param arr Integer array.
     * @return Integer that is the index of the maximum value.
     */
    public static int indexOfMax(int[] arr) {
        if (arr == null || arr.length == 0) {
            return -1;
        } else {
            int maxVal = -Integer.MAX_VALUE;
            int maxIndex = -1;
            for (int i = 0; i < arr.length; i++) {
                if (!DataMineConstants.isAcceptableInt(arr[i])) {
                    continue;
                }
                if (arr[i] > maxVal) {
                    maxVal = arr[i];
                    maxIndex = i;
                }
            }
            return maxIndex;
        }
    }

    /**
     * First return value is the value itself, second is its index.
     *
     * @param arr Double ArrayList.
     * @return A two element array, where the first return value is the minimum
     * of the array that was passed to the method and the second is its index.
     */
    public static double[] minWithIndexDouble(ArrayList<Double> arr) {
        double[] res = new double[2];
        if (arr == null || arr.isEmpty()) {
            res[0] = -1;
            res[1] = -1;
            return res;
        } else {
            res[0] = Double.MAX_VALUE;
            for (int i = 0; i < arr.size(); i++) {
                if (!DataMineConstants.isAcceptableDouble(arr.get(i))) {
                    continue;
                }
                if (arr.get(i) < res[0]) {
                    res[0] = arr.get(i);
                    res[1] = i;
                }
            }
            return res;
        }
    }

    /**
     * First return value is the value itself, second is its index.
     *
     * @param arr Double ArrayList.
     * @return A two element array, where the first return value is the maximum
     * of the array that was passed to the method and the second is its index.
     */
    public static double[] maxWithIndexDouble(ArrayList<Double> arr) {
        double[] res = new double[2];
        if (arr == null || arr.isEmpty()) {
            res[0] = -1;
            res[1] = -1;
            return res;
        } else {
            res[0] = -Double.MAX_VALUE;
            for (int i = 0; i < arr.size(); i++) {
                if (!DataMineConstants.isAcceptableDouble(arr.get(i))) {
                    continue;
                }
                if (arr.get(i) > res[0]) {
                    res[0] = arr.get(i);
                    res[1] = i;
                }
            }
            return res;
        }
    }

    /**
     * First return value is the value itself, second is its index.
     *
     * @param arr Float ArrayList.
     * @return A two element array, where the first return value is the minimum
     * of the array that was passed to the method and the second is its index.
     */
    public static float[] minWithIndexFloat(ArrayList<Float> arr) {
        float[] res = new float[2];
        if (arr == null || arr.isEmpty()) {
            res[0] = -1;
            res[1] = -1;
            return res;
        } else {
            res[0] = Float.MAX_VALUE;
            for (int i = 0; i < arr.size(); i++) {
                if (!DataMineConstants.isAcceptableFloat(arr.get(i))) {
                    continue;
                }
                if (arr.get(i) < res[0]) {
                    res[0] = arr.get(i);
                    res[1] = i;
                }
            }
            return res;
        }
    }

    /**
     * First return value is the value itself, second is its index.
     *
     * @param arr Float ArrayList.
     * @return A two element array, where the first return value is the maximum
     * of the array that was passed to the method and the second is its index.
     */
    public static float[] maxWithIndexFloat(ArrayList<Float> arr) {
        float[] res = new float[2];
        if (arr == null || arr.isEmpty()) {
            res[0] = -1;
            res[1] = -1;
            return res;
        } else {
            res[0] = -Float.MAX_VALUE;
            for (int i = 0; i < arr.size(); i++) {
                if (!DataMineConstants.isAcceptableFloat(arr.get(i))) {
                    continue;
                }
                if (arr.get(i) > res[0]) {
                    res[0] = arr.get(i);
                    res[1] = i;
                }
            }
            return res;
        }
    }

    /**
     * First return value is the value itself, second is its index.
     *
     * @param arr Integer ArrayList.
     * @return A two element array, where the first return value is the minimum
     * of the array that was passed to the method and the second is its index.
     */
    public static int[] minWithIndexInt(ArrayList<Integer> arr) {
        int[] res = new int[2];
        if (arr == null || arr.isEmpty()) {
            res[0] = -1;
            res[1] = -1;
            return res;
        } else {
            res[0] = Integer.MAX_VALUE;
            for (int i = 0; i < arr.size(); i++) {
                if (!DataMineConstants.isAcceptableInt(arr.get(i))) {
                    continue;
                }
                if (arr.get(i) < res[0]) {
                    res[0] = arr.get(i);
                    res[1] = i;
                }
            }
            return res;
        }
    }

    /**
     * First return value is the value itself, second is its index.
     *
     * @param arr Integer ArrayList.
     * @return A two element array, where the first return value is the maximum
     * of the array that was passed to the method and the second is its index.
     */
    public static int[] maxWithIndexInt(ArrayList<Integer> arr) {
        int[] res = new int[2];
        if (arr == null || arr.isEmpty()) {
            res[0] = -1;
            res[1] = -1;
            return res;
        } else {
            res[0] = -Integer.MAX_VALUE;
            for (int i = 0; i < arr.size(); i++) {
                if (!DataMineConstants.isAcceptableInt(arr.get(i))) {
                    continue;
                }
                if (arr.get(i) > res[0]) {
                    res[0] = arr.get(i);
                    res[1] = i;
                }
            }
            return res;
        }
    }
}
