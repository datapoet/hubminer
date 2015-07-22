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
package images.mining.segmentation;

import java.awt.Color;
import java.awt.Image;
import java.util.ArrayList;

/**
 * This interface is to be implemented by classes that represent image
 * segmentation methods.
 * 
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public interface SegmentationInterface {

    /**
     * @return The number of segments produced by the segmentation.
     */
    public int getNumberOfSegments();

    /**
     * Obtain the current image segmentation.
     *
     * @return Integer 2D array, where the (x,y) holds the index of the segment
     * to which the corresponding image pixel belongs.
     */
    public int[][] getSegmentation();

    /**
     * Calculate the average color in a segment.
     *
     * @param segmentIndex Index of the specified segment.
     * @return Color object holding the average color in the segment.
     */
    public Color getAverageColorForSegment(int segmentIndex);

    /**
     * Calculate a list of average colors for all image segments.
     *
     * @return ArrayList<Color> of average segment colors.
     */
    public ArrayList<Color> getAverageSegmentColors();

    /**
     * Generate an image that is the segmented image.
     *
     * @return Image that is the segmented image according to the current
     * segmentation.
     */
    public Image getSegmentedImage();

    /**
     * Perform image segmentation.
     *
     * @throws Exception
     */
    public void segment() throws Exception;
}
