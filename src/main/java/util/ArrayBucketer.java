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

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;

/**
 * A utility class that counts the number of occurrences of a quantity in each
 * bucket, where the size of a bucket is given as an input parameter. Output
 * goes to the command line. Redirect the output to a file for writing it down.
 * In the input file, quantities are separated by a comma and the position of
 * the desired quantity is also given as input parameter.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class ArrayBucketer {

    public static void main(String args[]) throws Exception {
        if (args.length != 3) {
            System.out.println("arg0: Input file");
            System.out.println("arg1: Bucket width (int)");
            System.out.println("arg2: Position of the desired quantity in the"
                    + "comma-separated line.");
            return;
        }
        int bucketWidth = Integer.parseInt(args[1]);
        ArrayList<Integer> numbers = new ArrayList<>(20000);
        ArrayList<Integer> counts = new ArrayList<>(5000);
        BufferedReader br = new BufferedReader(
                new InputStreamReader(new FileInputStream(new File(args[0]))));
        int num;
        int maxNum = 0;
        String[] line;
        int quantityPosition = Integer.parseInt(args[2]);
        try {
            // Skip header. The first line is ignored.
            String s = br.readLine();
            s = br.readLine();
            while (s != null) {
                s = s.trim();
                line = s.split(",");
                num = Integer.parseInt(line[quantityPosition]);
                if (num > maxNum) {
                    maxNum = num;
                }
                numbers.add(num);
                s = br.readLine();
            }
        } catch (IOException | NumberFormatException e) {
            throw e;
        } finally {
            br.close();
        }
        int numBuckets = (maxNum / bucketWidth) + 1;
        for (int i = 0; i < numBuckets; i++) {
            counts.add(new Integer(0));
        }
        int buckIndex;
        for (int i = 0; i < numbers.size(); i++) {
            buckIndex = numbers.get(i) / bucketWidth;
            counts.set(buckIndex, counts.get(buckIndex) + 1);
        }
        for (int i = 0; i < counts.size(); i++) {
            System.out.println(counts.get(i));
        }
    }
}
