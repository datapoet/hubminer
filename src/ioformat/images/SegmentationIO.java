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
package ioformat.images;

import images.mining.segmentation.SegmentationInterface;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.io.PrintWriter;

/**
 * This class implements the persistence of image segmentations.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class SegmentationIO {

    /**
     * The default constructor.
     */
    public SegmentationIO() {
    }
    private int numberOfSegments;
    private int width, height;
    private int[][] segmentation;

    /**
     * Write the segmentation to a file.
     *
     * @param segmenter SegmentationInterface object.
     * @param outFile Output file.
     * @throws Exception
     */
    public void write(SegmentationInterface segmenter, File outFile)
            throws Exception {
        numberOfSegments = segmenter.getNumberOfSegments();
        segmentation = segmenter.getSegmentation();
        PrintWriter pw = new PrintWriter(new FileWriter(outFile));
        width = segmentation.length;
        height = segmentation[0].length;
        try {
            pw.println(numberOfSegments + " " + width + " " + height);
            for (int i = 0; i < width; i++) {
                for (int j = 0; j < height - 1; j++) {
                    pw.print(segmentation[i][j] + " ");
                }
                pw.println(segmentation[i][height - 1]);
            }
        } catch (Exception e) {
            System.err.println(e.getMessage());
            throw e;
        } finally {
            pw.close();
        }
    }

    /**
     * Read the segmentation from a file.
     *
     * @param inFile File that contains the segmentation.
     * @throws Exception
     */
    public void read(File inFile) throws Exception {
        BufferedReader br = new BufferedReader(new InputStreamReader(
                new FileInputStream(inFile)));
        try {
            String line = br.readLine();
            String[] lineItems = line.split(" ");
            numberOfSegments = Integer.parseInt(lineItems[0]);
            width = Integer.parseInt(lineItems[1]);
            height = Integer.parseInt(lineItems[2]);
            segmentation = new int[width][height];
            for (int i = 0; i < width; i++) {
                line = br.readLine();
                lineItems = line.split(" ");
                for (int j = 0; j < lineItems.length; j++) {
                    segmentation[i][j] = Integer.parseInt(lineItems[j]);
                }
            }
        } catch (Exception e) {
            System.err.println(e.getMessage());
            throw e;
        } finally {
            br.close();
        }
    }

    /**
     * Get the width and height in an array, where the width comes first and the
     * height follows.
     *
     * @return int[] of length 2, containing the width at index 0 and the height
     * at index 1.
     */
    public int[] getWidthAndHeight() {
        int[] res = new int[2];
        res[0] = width;
        res[1] = height;
        return res;
    }

    /**
     * @return Integer that is the number of segments.
     */
    public int getNumSegments() {
        return numberOfSegments;
    }

    /**
     * @return The represented image segmentation.
     */
    public int[][] getSegmentation() {
        return segmentation;
    }
}