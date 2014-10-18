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

import ioformat.images.SegmentationIO;
import java.io.File;

/**
 * This utility script calculates the minimum, maximum and average number of
 * segments in processed segmented image files in a directory.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class SegmentCounter {

    /**
     * This utility script calculates the minimum, maximum and average number of
     * segments in processed segmented image files in a directory.
     *
     * @param args One argument: a path to the target directory.
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            System.out.println("arg0: Path to a directory of segmentation "
                    + "files.");
            return;
        }
        File inDir = new File(args[0]);
        File[] children = inDir.listFiles();
        float fCounter = 0;
        int minNum = Integer.MAX_VALUE;
        int maxNum = Integer.MIN_VALUE;
        int tempSeg;
        float avg = 0;
        for (int i = 0; i < children.length; i++) {
            if (children[i].isFile()) {
                SegmentationIO segIO = new SegmentationIO();
                segIO.read(children[i]);
                tempSeg = segIO.getNumSegments();
                if (tempSeg > maxNum) {
                    maxNum = tempSeg;
                }
                if (tempSeg < minNum) {
                    minNum = tempSeg;
                }
                avg += tempSeg;
                fCounter++;
            }
        }
        avg = avg / fCounter;
        System.out.println("min " + minNum + " max " + maxNum + " avg " + avg);
    }
}
