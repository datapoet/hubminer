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

import java.io.PrintWriter;
import java.util.ArrayList;

/**
 * A utility class that contains methods for printing our data arrays.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class SOPLUtil {

    /**
     * Prints out the array to standard output.
     *
     * @param arr An array of float values.
     */
    public static void printArray(float[] arr) {
        if (arr == null) {
            System.out.println("Null array.");
        }
        for (int i = 0; i < arr.length - 1; i++) {
            System.out.print(arr[i] + " ");
        }
        System.out.print(arr[arr.length - 1]);
        System.out.println();
    }

    /**
     * Prints out the array to standard output.
     *
     * @param arr An array of integer values.
     */
    public static void printArray(int[] arr) {
        if (arr == null) {
            System.out.println("Null array.");
        }
        for (int i = 0; i < arr.length - 1; i++) {
            System.out.print(arr[i] + " ");
        }
        System.out.print(arr[arr.length - 1]);
        System.out.println();
    }

    /**
     * Prints out the array to standard output.
     *
     * @param arr An array of double values.
     */
    public static void printArray(double[] arr) {
        if (arr == null) {
            System.out.println("Null array.");
        }
        for (int i = 0; i < arr.length - 1; i++) {
            System.out.print(arr[i] + " ");
        }
        System.out.print(arr[arr.length - 1]);
        System.out.println();
    }

    /**
     * Prints out the array to standard output.
     *
     * @param arr An array of string values.
     */
    public static void printArray(String[] arr) {
        if (arr == null) {
            System.out.println("Null array.");
        }
        for (int i = 0; i < arr.length - 1; i++) {
            System.out.print(arr[i] + " ");
        }
        System.out.print(arr[arr.length - 1]);
        System.out.println();
    }

    /**
     * Prints out the ArrayList to standard output.
     *
     * @param arr An ArrayList of double values.
     */
    public static void printArrayList(ArrayList arr) {
        if (arr == null) {
            System.out.println("Null ArrayList.");
            return;
        }
        if (arr.isEmpty()) {
            System.out.println("Empty ArrayList");
            return;
        }
        for (int i = 0; i < arr.size() - 1; i++) {
            System.out.print(arr.get(i) + " ");
        }
        System.out.print(arr.get(arr.size() - 1));
        System.out.println();
    }

    /**
     * Prints out the array to the specified stream output.
     *
     * @param arr An array of double values.
     * @param pw PrintWriter object for output.
     */
    public static void printArrayToStream(double[] arr, PrintWriter pw) {
        if (arr == null) {
            pw.println("Null array.");
        }
        for (int i = 0; i < arr.length - 1; i++) {
            pw.print(arr[i] + " ");
        }
        pw.print(arr[arr.length - 1]);
        pw.println();
    }

    /**
     * Prints out the array to the specified stream output.
     *
     * @param arr An array of integer values.
     * @param pw PrintWriter object for output.
     */
    public static void printArrayToStream(int[] arr, PrintWriter pw) {
        if (arr == null) {
            pw.println("Null array.");
        }
        for (int i = 0; i < arr.length - 1; i++) {
            pw.print(arr[i] + " ");
        }
        pw.print(arr[arr.length - 1]);
        pw.println();
    }

    /**
     * Prints out the array to the specified stream output.
     *
     * @param arr An array of float values.
     * @param pw PrintWriter object for output.
     */
    public static void printArrayToStream(float[] arr, PrintWriter pw) {
        if (arr == null) {
            pw.println("Null array.");
        }
        for (int i = 0; i < arr.length - 1; i++) {
            pw.print(arr[i] + " ");
        }
        pw.print(arr[arr.length - 1]);
        pw.println();
    }

    /**
     * Prints out the array to the specified stream output.
     *
     * @param arr An array of string values.
     * @param pw PrintWriter object for output.
     */
    public static void printArrayToStream(String[] arr, PrintWriter pw) {
        if (arr == null) {
            pw.println("Null array.");
        }
        for (int i = 0; i < arr.length - 1; i++) {
            pw.print(arr[i] + " ");
        }
        pw.print(arr[arr.length - 1]);
        pw.println();
    }

    /**
     * Prints out the ArrayList to the specified stream output.
     *
     * @param arr An ArrayList of double values.
     * @param pw PrintWriter object for output.
     */
    public static void printArrayListToStream(ArrayList arr, PrintWriter pw) {
        if (arr == null) {
            pw.println("Null ArrayList.");
            return;
        }
        if (arr.isEmpty()) {
            pw.println("Empty ArrayList");
            return;
        }
        for (int i = 0; i < arr.size() - 1; i++) {
            pw.print(arr.get(i) + " ");
        }
        pw.print(arr.get(arr.size() - 1));
        pw.println();
    }

    /**
     * Prints out the array to the specified stream output.
     *
     * @param arr An array of double values.
     * @param pw PrintWriter object for output.
     * @param sep String separator.
     */
    public static void printArrayToStream(double[] arr, PrintWriter pw,
            String sep) {
        if (arr == null) {
            pw.println("Null array.");
        }
        for (int i = 0; i < arr.length - 1; i++) {
            pw.print(arr[i] + sep);
        }
        pw.print(arr[arr.length - 1]);
        pw.println();
    }

    /**
     * Prints out the array to the specified stream output.
     *
     * @param arr An array of integer values.
     * @param pw PrintWriter object for output.
     * @param sep String separator.
     */
    public static void printArrayToStream(int[] arr, PrintWriter pw,
            String sep) {
        if (arr == null) {
            pw.println("Null array.");
        }
        for (int i = 0; i < arr.length - 1; i++) {
            pw.print(arr[i] + sep);
        }
        pw.print(arr[arr.length - 1]);
        pw.println();
    }

    /**
     * Prints out the array to the specified stream output.
     *
     * @param arr An array of float values.
     * @param pw PrintWriter object for output.
     * @param sep String separator.
     */
    public static void printArrayToStream(float[] arr, PrintWriter pw,
            String sep) {
        if (arr == null) {
            pw.println("Null array.");
        }
        for (int i = 0; i < arr.length - 1; i++) {
            pw.print(arr[i] + sep);
        }
        pw.print(arr[arr.length - 1]);
        pw.println();
    }

    /**
     * Prints out the array to the specified stream output.
     *
     * @param arr An array of string values.
     * @param pw PrintWriter object for output.
     * @param sep String separator.
     */
    public static void printArrayToStream(String[] arr, PrintWriter pw,
            String sep) {
        if (arr == null) {
            pw.println("Null array.");
        }
        for (int i = 0; i < arr.length - 1; i++) {
            pw.print(arr[i] + sep);
        }
        pw.print(arr[arr.length - 1]);
        pw.println();
    }

    /**
     * Prints out the ArrayList to the specified stream output.
     *
     * @param arr An ArrayList of double values.
     * @param pw PrintWriter object for output.
     * @param sep String separator.
     */
    public static void printArrayListToStream(ArrayList arr, PrintWriter pw,
            String sep) {
        if (arr == null) {
            pw.println("Null ArrayList.");
            return;
        }
        if (arr.isEmpty()) {
            pw.println("Empty ArrayList");
            return;
        }
        for (int i = 0; i < arr.size() - 1; i++) {
            pw.print(arr.get(i) + sep);
        }
        pw.print(arr.get(arr.size() - 1));
        pw.println();
    }
}
