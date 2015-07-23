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

/**
 * A utility script that parses a file where each row contains neighbor
 * occurrence frequencies of data points, first the total, then the bad, then
 * the good. It calculates the average label mismatch percentage.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class ArrayBHPercGetter {

    /**
     * @param args One command line argument: file name.
     * @throws Exception
     */
    public static void main(String args[]) throws Exception {
        if (args.length != 1) {
            System.out.println("0: file");
            return;
        }
        double total = 0;
        double badTotal = 0;
        BufferedReader br = new BufferedReader(
                new InputStreamReader(new FileInputStream(new File(args[0]))));
        String[] line;
        try {
            // Skip header.
            String s = br.readLine();
            s = br.readLine();
            while (s != null) {
                s = s.trim();
                line = s.split(",");
                total += Integer.parseInt(line[0]);
                badTotal += Integer.parseInt(line[1]);
                s = br.readLine();
            }
        } catch (IOException | NumberFormatException e) {
            throw e;
        } finally {
            br.close();
        }
        System.out.println(badTotal / total);
    }
}
