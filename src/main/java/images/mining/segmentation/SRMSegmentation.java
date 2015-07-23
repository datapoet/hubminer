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

import ioformat.images.ImageFromRaster;
import java.awt.Color;
import java.awt.Image;
import java.awt.image.PixelGrabber;
import java.util.ArrayList;
import java.util.HashMap;

/**
 * This class implements the statistical region merging segmentation algorithm
 * and the code is heavily based on an applet created by Frank Nielsen and
 * Richard Nock.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class SRMSegmentation implements SegmentationInterface {

    private Image originalImage = null;
    private Image segmentedImage;
    private PixelGrabber pg;
    // Each segment has a root and this maps the roots to segment indexes.
    private HashMap<Integer, Integer> rootIndexesToSegments =
            new HashMap<>(1500);
    // A list of segment root indexes.
    private ArrayList<Integer> segmentsToRootIndexesList = new ArrayList<>(500);
    // Average within-segment colors.
    private ArrayList<Color> avgSegmentColors = new ArrayList<>(500);
    private int numberOfSegments = 0;
    private UnionFind uFind;
    // Pixel raster arrays.
    private int[] imageRaster;
    private int[] segmentedImageRaster;
    // Segment associations for each image pixel, as a 2D array.
    private int[][] segmentAssociations;
    private int width, height, numPixels;
    private double aspectRatio;
    // Number of levels in a color channel.
    private static final double CHANNEL_LEVELS = 256.0;
    private double logdelta;
    private static final double QPARAM = 32;
    // Auxilliary buffers for union-find operations
    private int[] regionSize;
    private double[] redAvg;
    private double[] greenAvg;
    private double[] blueAvg;
    // A number of pixels that defines a collapsable region.
    private int smallRegionSize;
    private int borderThickness;
    private int[] classIndex;

    /**
     * Initialization.
     *
     * @param image Image to segment.
     */
    public SRMSegmentation(Image image) {
        this.originalImage = image;
    }

    /**
     * @return Double that is the aspect ratio of the image.
     */
    public double getAspectRatio() {
        return aspectRatio;
    }

    @Override
    public int getNumberOfSegments() {
        return numberOfSegments;
    }

    @Override
    public int[][] getSegmentation() {
        return segmentAssociations;
    }

    @Override
    public ArrayList<Color> getAverageSegmentColors() {
        return avgSegmentColors;
    }

    @Override
    public Color getAverageColorForSegment(int i) {
        return avgSegmentColors.get(i);
    }

    @Override
    public Image getSegmentedImage() {
        return segmentedImage;
    }

    @Override
    public void segment() throws Exception {
        initialize();
        performSegmentation();
    }

    /**
     * Initialize all the structures and parameters.
     *
     * @throws Exception
     */
    public void initialize() throws Exception {
        if (originalImage == null) {
            throw new Exception("Cannot segment a NULL image.");
        }
        // Region border thickness.
        borderThickness = 0;
        pg = new PixelGrabber(originalImage, 0, 0, -1, -1, true);
        try {
            pg.grabPixels();
        } catch (InterruptedException e) {
            System.err.println(e.getMessage());
        }
        width = pg.getWidth();
        height = pg.getHeight();
        imageRaster = (int[]) pg.getPixels();
        aspectRatio = (double) height / (double) width;
        numPixels = width * height;
        // Algorithm-specific thresholds.
        logdelta = 2.0 * Math.log(6.0 * numPixels);
        // Small regions are those that contain less than 0.1% of image pixels.
        smallRegionSize = (int) (0.001 * numPixels);
    }

    /**
     * Performs image segmentation according to statistical region merging.
     */
    private void performSegmentation() {
        uFind = new UnionFind(numPixels);
        redAvg = new double[numPixels];
        greenAvg = new double[numPixels];
        blueAvg = new double[numPixels];
        regionSize = new int[numPixels];
        classIndex = new int[numPixels];
        segmentedImageRaster = new int[numPixels];
        performInitialSegmentation();
        performFullSegmentation();
        // Create the actual segmented image for visualization.
        this.segmentedImage = ImageFromRaster.createImage(
                segmentedImageRaster, pg.getWidth(), pg.getHeight());
    }

    /**
     * Performs the initial image segmentation, where each pixel is assigned to
     * its own region.
     */
    private void performInitialSegmentation() {
        int red, green, blue, rasterIndex;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                rasterIndex = y * width + x;
                red = (imageRaster[y * width + x] & 0xFF);
                green = ((imageRaster[y * width + x] & 0xFF00) >> 8);
                blue = ((imageRaster[y * width + x] & 0xFF0000) >> 16);
                regionSize[rasterIndex] = 1;
                classIndex[rasterIndex] = rasterIndex;
                redAvg[rasterIndex] = red;
                greenAvg[rasterIndex] = green;
                blueAvg[rasterIndex] = blue;
            }
        }
    }

    /**
     * Performs the actual image segmentation, after the initialization.
     */
    private void performFullSegmentation() {
        segmentInternal();
        mergeSmallRegions();
        prepareOutput();
        applyBorders();
    }

    /**
     * Calculates the maximum of three different double numbers.
     *
     * @param x Double value.
     * @param y Double value.
     * @param z Double value.
     * @return Double value that is the maximum of the three double values.
     */
    private double maxOfThree(double x, double y, double z) {
        return Math.max(x, Math.max(y, z));
    }

    /**
     * Decides whether to merge the two regions.
     *
     * @param firstRegionIndex Integer that is the index of the first region.
     * @param secondRegionIndex Integer that is the index of the second region.
     * @return Boolean value indicating whether it is possible or not..
     */
    boolean canMergeRegions(int firstRegionIndex, int secondRegionIndex) {
        double deltaRed, deltaGreen, deltaBlue;
        double logFirst, logSecond;
        double deltaLimitFirst, deltaLimitSecond, deltaLimit;
        deltaRed = (redAvg[firstRegionIndex] - redAvg[secondRegionIndex]);
        deltaRed *= deltaRed;
        deltaGreen = (greenAvg[firstRegionIndex] - greenAvg[secondRegionIndex]);
        deltaGreen *= deltaGreen;
        deltaBlue = (blueAvg[firstRegionIndex] - blueAvg[secondRegionIndex]);
        deltaBlue *= deltaBlue;
        logFirst = Math.min(CHANNEL_LEVELS, regionSize[firstRegionIndex])
                * Math.log(1.0 + regionSize[firstRegionIndex]);
        logSecond = Math.min(CHANNEL_LEVELS, regionSize[secondRegionIndex])
                * Math.log(1.0 + regionSize[secondRegionIndex]);
        deltaLimitFirst = ((CHANNEL_LEVELS * CHANNEL_LEVELS) / (2.0 * QPARAM
                * regionSize[firstRegionIndex])) * (logFirst + logdelta);
        deltaLimitSecond = ((CHANNEL_LEVELS * CHANNEL_LEVELS) / (2.0 * QPARAM
                * regionSize[secondRegionIndex])) * (logSecond + logdelta);
        deltaLimit = deltaLimitFirst + deltaLimitSecond;
        return ((deltaRed < deltaLimit) && (deltaGreen < deltaLimit)
                && (deltaBlue < deltaLimit));
    }

    /**
     * Bucket sorting.
     *
     * @param pairsDiffs RegionPairDiff[] array.
     * @return RegionPairDiff[] array that is sorted by the desired criteria.
     */
    private RegionPairDiff[] bucketSort(RegionPairDiff[] pairsDiffs) {
        int numPairs = pairsDiffs != null ? pairsDiffs.length : 0;
        int[] histogram = new int[256];
        int[] cumulativeHistogram = new int[256];
        RegionPairDiff[] pairsDiffsSorted = new RegionPairDiff[numPairs];
        // Classify all elements according to their family.
        for (int i = 0; i < numPairs; i++) {
            histogram[pairsDiffs[i].delta]++;
        }
        // Calculate the cumulative histogram.
        cumulativeHistogram[0] = 0;
        for (int c = 1; c < 256; c++) {
            // The index of the first element of category c.
            cumulativeHistogram[c] = cumulativeHistogram[c - 1]
                    + histogram[c - 1];
        }
        // Allocation.
        for (int i = 0; i < numPairs; i++) {
            pairsDiffsSorted[cumulativeHistogram[pairsDiffs[i].delta]++] =
                    pairsDiffs[i];
        }
        return pairsDiffsSorted;
    }

    /**
     * Merges the two regions.
     *
     * @param firstRegionIndex Integer that is the index of the first region.
     * @param secondRegionIndex Integer that is the index of the second region.
     */
    private void mergeRegions(int firstRegionIndex, int secondRegionIndex) {
        int rootIndex = uFind.mergeRoots(firstRegionIndex, secondRegionIndex);
        int regionSizeMerged = regionSize[firstRegionIndex]
                + regionSize[secondRegionIndex];
        double redAverage = (regionSize[firstRegionIndex]
                * redAvg[firstRegionIndex] + regionSize[secondRegionIndex]
                * redAvg[secondRegionIndex]) / regionSizeMerged;
        double greenAverage = (regionSize[firstRegionIndex]
                * greenAvg[firstRegionIndex] + regionSize[secondRegionIndex]
                * greenAvg[secondRegionIndex]) / regionSizeMerged;
        double blueAverage = (regionSize[firstRegionIndex]
                * blueAvg[firstRegionIndex] + regionSize[secondRegionIndex]
                * blueAvg[secondRegionIndex]) / regionSizeMerged;
        // Update the structures.
        regionSize[rootIndex] = regionSizeMerged;
        redAvg[rootIndex] = redAverage;
        greenAvg[rootIndex] = greenAverage;
        blueAvg[rootIndex] = blueAverage;
    }

    private void segmentInternal() {
        int rasterIndex, firstRegionIndex, secondRegionIndex;
        int pairCount = 0;
        int redFirst, greenFirst, blueFirst, redSecond, greenSecond, blueSecond;
        int numRegionPairs = 2 * (width - 1) * (height - 1) + (height - 1)
                + (width - 1);
        RegionPairDiff[] pairsDiffs = new RegionPairDiff[numRegionPairs];
        // Building the initial image RAG.
        for (int y = 0; y < height - 1; y++) {
            for (int x = 0; x < width - 1; x++) {
                rasterIndex = y * width + x;
                // C4 connectivity left.
                pairsDiffs[pairCount] = new RegionPairDiff();
                pairsDiffs[pairCount].first = rasterIndex;
                pairsDiffs[pairCount].second = rasterIndex + 1;
                redFirst = imageRaster[rasterIndex] & 0xFF;
                greenFirst = ((imageRaster[rasterIndex] & 0xFF00) >> 8);
                blueFirst = ((imageRaster[rasterIndex] & 0xFF0000) >> 16);
                redSecond = imageRaster[rasterIndex + 1] & 0xFF;
                greenSecond = ((imageRaster[rasterIndex + 1] & 0xFF00) >> 8);
                blueSecond = ((imageRaster[rasterIndex + 1] & 0xFF0000) >> 16);
                pairsDiffs[pairCount].delta = (int) maxOfThree(
                        Math.abs(redSecond - redFirst),
                        Math.abs(greenSecond - greenFirst),
                        Math.abs(blueSecond - blueFirst));
                pairCount++;
                // C4 connectivity below.
                pairsDiffs[pairCount] = new RegionPairDiff();
                pairsDiffs[pairCount].first = rasterIndex;
                pairsDiffs[pairCount].second = rasterIndex + width;
                redSecond = imageRaster[rasterIndex + width] & 0xFF;
                greenSecond =
                        ((imageRaster[rasterIndex + width] & 0xFF00) >> 8);
                blueSecond =
                        ((imageRaster[rasterIndex + width] & 0xFF0000) >> 16);
                pairsDiffs[pairCount].delta = (int) maxOfThree(
                        Math.abs(redSecond - redFirst),
                        Math.abs(greenSecond - greenFirst),
                        Math.abs(blueSecond - blueFirst));
                pairCount++;
            }
        }
        // The two borders.
        for (int y = 0; y < height - 1; y++) {
            rasterIndex = y * width + width - 1;
            pairsDiffs[pairCount] = new RegionPairDiff();
            pairsDiffs[pairCount].first = rasterIndex;
            pairsDiffs[pairCount].second = rasterIndex + width;
            redFirst = imageRaster[rasterIndex] & 0xFF;
            greenFirst = ((imageRaster[rasterIndex] & 0xFF00) >> 8);
            blueFirst = ((imageRaster[rasterIndex] & 0xFF0000) >> 16);
            redSecond = imageRaster[rasterIndex + width] & 0xFF;
            greenSecond = ((imageRaster[rasterIndex + width] & 0xFF00) >> 8);
            blueSecond = ((imageRaster[rasterIndex + width] & 0xFF0000) >> 16);
            pairsDiffs[pairCount].delta = (int) maxOfThree(
                    Math.abs(redSecond - redFirst),
                    Math.abs(greenSecond - greenFirst),
                    Math.abs(blueSecond - blueFirst));
            pairCount++;
        }
        for (int x = 0; x < width - 1; x++) {
            rasterIndex = (height - 1) * width + x;
            pairsDiffs[pairCount] = new RegionPairDiff();
            pairsDiffs[pairCount].first = rasterIndex;
            pairsDiffs[pairCount].second = rasterIndex + 1;
            redFirst = imageRaster[rasterIndex] & 0xFF;
            greenFirst = ((imageRaster[rasterIndex] & 0xFF00) >> 8);
            blueFirst = ((imageRaster[rasterIndex] & 0xFF0000) >> 16);
            redSecond = imageRaster[rasterIndex + 1] & 0xFF;
            greenSecond = ((imageRaster[rasterIndex + 1] & 0xFF00) >> 8);
            blueSecond = ((imageRaster[rasterIndex + 1] & 0xFF0000) >> 16);
            pairsDiffs[pairCount].delta = (int) maxOfThree(
                    Math.abs(redSecond - redFirst),
                    Math.abs(greenSecond - greenFirst),
                    Math.abs(blueSecond - blueFirst));
            pairCount++;
        }
        // Sorting all edges via bucket sort, by the maximum color channel
        // difference.
        pairsDiffs = bucketSort(pairsDiffs);
        // The main SRM part.
        for (int i = 0; i < numRegionPairs; i++) {
            firstRegionIndex = pairsDiffs[i].first;
            int firstRoot = uFind.findRoot(firstRegionIndex);
            secondRegionIndex = pairsDiffs[i].second;
            int secondRoot = uFind.findRoot(secondRegionIndex);
            if ((firstRoot != secondRoot)
                    && (canMergeRegions(firstRoot, secondRoot))) {
                mergeRegions(firstRoot, secondRoot);
            }
        }
    }

    /**
     * Prepare the output segmentation data structures.
     */
    void prepareOutput() {
        segmentAssociations = new int[width][height];
        int rasterIndex, rootIndex;
        int red, green, blue, rgb;
        int segmentIndex = -1;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                rasterIndex = y * width + x;
                rootIndex = uFind.findRoot(rasterIndex);
                if (!rootIndexesToSegments.containsKey(rootIndex)) {
                    numberOfSegments++;
                    ++segmentIndex;
                    segmentsToRootIndexesList.add(rootIndex);
                    rootIndexesToSegments.put(rootIndex, segmentIndex);
                    segmentAssociations[x][y] = segmentIndex;
                } else {
                    segmentAssociations[x][y] =
                            rootIndexesToSegments.get(rootIndex);
                }
                // Average color for the segment.
                red = (int) redAvg[rootIndex];
                green = (int) greenAvg[rootIndex];
                blue = (int) blueAvg[rootIndex];
                rgb = (0xff000000 | blue << 16 | green << 8 | red);
                segmentedImageRaster[rasterIndex] = rgb;
                avgSegmentColors.add(new Color(red, green, blue));
            }
        }
    }

    /**
     * Merge the remaining small regions.
     */
    void mergeSmallRegions() {
        int rasterIndex, firstRegionRootIndex, secondRegionRootIndex;
        for (int y = 0; y < height; y++) {
            for (int x = 1; x < width; x++) {
                rasterIndex = y * width + x;
                firstRegionRootIndex = uFind.findRoot(rasterIndex);
                secondRegionRootIndex = uFind.findRoot(rasterIndex - 1);
                if (secondRegionRootIndex != firstRegionRootIndex) {
                    if ((regionSize[secondRegionRootIndex] < smallRegionSize)
                            || (regionSize[firstRegionRootIndex]
                            < smallRegionSize)) {
                        mergeRegions(firstRegionRootIndex,
                                secondRegionRootIndex);
                    }
                }
            }
        }
    }

    /**
     * Draw the border lines.
     */
    void applyBorders() {
        int firstRegionRootIndex, secondRegionRootIndex, rasterIndex;
        for (int y = 1; y < height; y++) {
            for (int x = 1; x < width; x++) {
                rasterIndex = y * width + x;
                firstRegionRootIndex = uFind.findRoot(rasterIndex);
                secondRegionRootIndex = uFind.findRoot(rasterIndex - 1 - width);
                if (secondRegionRootIndex != firstRegionRootIndex) {
                    for (int k = -borderThickness; k <= borderThickness; k++) {
                        for (int l = -borderThickness; l <= borderThickness;
                                l++) {
                            rasterIndex = (y + k) * width + (x + l);
                            if ((rasterIndex >= 0)
                                    && (rasterIndex < width * height)) {
                                segmentedImageRaster[rasterIndex] =
                                        (0xff000000
                                        | 255 << 16 | 255 << 8 | 255);
                            }
                        }
                    }
                }
            }
        }
    }

    /**
     * A data structure of Tarjan for disjoint sets.
     */
    class UnionFind {

        int[] rank;
        int[] parent;

        /**
         * Initialization.
         *
         * @param size Integer that is the number of elements.
         */
        UnionFind(int size) {
            parent = new int[size];
            rank = new int[size];
            for (int i = 0; i < size; i++) {
                parent[i] = i;
                rank[i] = 0;
            }
        }

        /**
         * Search.
         *
         * @param index Integer that is the index to search for.
         * @return Integer that is the root.
         */
        int findRoot(int index) {
            int i = index;
            while (parent[i] != i) {
                i = parent[i];
            }
            return i;
        }

        /**
         * Merges two roots.
         *
         * @param firstRoot Integer that is the first root.
         * @param secondRoot Integer that is the second root.
         * @return Integer that is the resulting root, -1 if failure.
         */
        int mergeRoots(int firstRoot, int secondRoot) {
            if (firstRoot == secondRoot) {
                return -1;
            }
            if (rank[firstRoot] > rank[secondRoot]) {
                parent[secondRoot] = firstRoot;
                return firstRoot;
            } else {
                parent[firstRoot] = secondRoot;
                if (rank[firstRoot] == rank[secondRoot]) {
                    rank[secondRoot]++;
                }
                return secondRoot;
            }
        }
    }

    /**
     * Two indexes and a difference between the regions.
     */
    class RegionPairDiff {

        int first, second;
        int delta;

        RegionPairDiff() {
            first = 0;
            second = 0;
            delta = 0;
        }
    }
}
