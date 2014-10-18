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

import data.representation.images.sift.SIFTRepresentation;
import ioformat.images.SiftUtil;
import java.io.File;

/**
 * A utility script for quickly calculating the minimal, maximal and average
 * number of SIFT features per image from a directory of SIFT keyfiles.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class SIFTCounter {

    /**
     * This executes the script.
     * 
     * @param args Command line parameters, as specified.
     * @throws Exception 
     */
    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            System.out.println("A script that calculates the min max and avg "
                    + "number of features in SIFT files in a given directory");
            System.out.println("arg0: Path to the target directory.");
            return;
        }
        File inDir = new File(args[0]);
        File[] children = inDir.listFiles();
        float fCounter = 0;
        int minNum = Integer.MAX_VALUE;
        int maxNum = Integer.MIN_VALUE;
        int tempSIFT;
        float avg = 0;
        for (int i = 0; i < children.length; i++) {
            if (children[i].isFile()) {
                SIFTRepresentation tempRep =
                        SiftUtil.importFeaturesFromSift(children[i]);
                tempSIFT = tempRep.size();
                if (tempSIFT > maxNum) {
                    maxNum = tempSIFT;
                }
                if (tempSIFT < minNum) {
                    minNum = tempSIFT;
                }
                avg += tempSIFT;
                fCounter++;
            }
        }
        avg = avg / fCounter;
        System.out.println("min " + minNum + " max " + maxNum + " avg " + avg);
    }
}
