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

import ioformat.FileUtil;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.io.PrintWriter;

/**
 * A utility class for having a peek into a big file.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class SneakPeek {

    /**
     * Takes a sneak peek into a larger file.
     *
     * @param args Command line arguments. Sourse file, number of lines to fetch
     * and the target file.
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        if (args.length != 3) {
            System.out.println("arg0: Input file path.");
            System.out.println("arg1: Number of lines to fetch.");
            System.out.println("arg2: Output file path.");
            return;
        }
        File inFile = new File(args[0]);
        int numLines = Integer.parseInt(args[1]);
        BufferedReader br = new BufferedReader(
                new InputStreamReader(new FileInputStream(inFile)));
        File outFile = new File(args[2]);
        if (!outFile.exists()) {
            FileUtil.createFileFromPath(outFile.getPath());
        }
        PrintWriter pw = new PrintWriter(new FileWriter(outFile.getPath()));
        String s;
        try {
            for (int i = 0; i < numLines; i++) {
                s = br.readLine();
                pw.println(s);
            }
        } catch (Exception e) {
            throw e;
        } finally {
            br.close();
            pw.close();
        }
    }
}
