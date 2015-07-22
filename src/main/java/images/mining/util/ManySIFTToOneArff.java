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
package images.mining.util;

import ioformat.images.SiftUtil;

/**
 * This is a command line mini-script for invoking the conversion of data from a
 * directory of SIFT key files into a single ARFF file.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class ManySIFTToOneArff {

    /**
     * Information about the command line usage.
     */
    public static void info() {
        System.out.println("arg0: Path to the input directory of SIFT key files"
                + " that are output by Lowe's binary.");
        System.out.println("arg1: File to the output ARFF file.");
    }

    /**
     * Main method performing the batch file conversion and merger.
     *
     * @param args Command line arguments, as specified in the info() method.
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            info();
        } else {
            SiftUtil.siftFolderToOneArffFile(args[0], args[1]);
        }
    }
}