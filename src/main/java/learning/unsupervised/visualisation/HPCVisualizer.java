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
package learning.unsupervised.visualisation;

import data.generators.NoisyGaussianMix;
import data.neighbors.NeighborSetFinder;
import data.representation.DataSet;
import distances.primary.CombinedMetric;
import draw.basic.RotatedEllipse;
import ioformat.FileUtil;
import ioformat.IOARFF;
import ioformat.IOCSV;
import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
import java.util.Random;
import javax.imageio.ImageIO;
import learning.unsupervised.methods.GHPC;
import mdsj.MDSJ;

/**
 * The class that is meant to visualize the execution of Hubness-proportional
 * clustering on MDS projections of the data.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class HPCVisualizer {

    boolean blackBGD = false;
    public static final int[] kVals = {1, 2, 5, 10, 30, 100};
    public boolean drawOnGrid = false;
    int numClusters = 5;
    int numDimensions = 50;
    int numOriginalInstances = 1000;
    int numNoisyInstances = 0;
    int numTotalInstances = 0;
    Color[] colors;
    DataSet dset = null;
    boolean genericNaming = false;
    public String origDName = null;
    float[][] distances = null;
    float[][] fullDistForMDS = null;
    float[][] pointXY = null;
    int probIters = 20;
    final static int IMAGE_X = 1500;
    final static int IMAGE_Y = 1250;

    /**
     * @param numClusters Number of clusters to be used.
     * @param numDimensions Number of dimensions - features.
     * @param numOriginalInstances Number of instances to cluster.
     * @param numNoisyInstances Number of noisy instances to insert.
     */
    public HPCVisualizer(int numClusters, int numDimensions,
            int numOriginalInstances, int numNoisyInstances) {
        this.numClusters = numClusters;
        this.numDimensions = numDimensions;
        this.numOriginalInstances = numOriginalInstances;
        System.out.println("numOrigInstances set to: " + numOriginalInstances);
        this.numNoisyInstances = numNoisyInstances;
        System.out.println("numNoisyInstances set to: " + numNoisyInstances);
        genericNaming = true;
    }

    /**
     *
     * @param dset DataSet object.
     */
    public HPCVisualizer(DataSet dset) {
        this.dset = dset;
    }

    /**
     * @param dir Target directory for generated images.
     * @param useDiffColors Boolean variable indicating whether to use different
     * colors for different classes/clusters.
     * @param drawOnGrid Boolean variable indicating whether to draw the results
     * on a grid or as graph nodes.
     * @throws Exception
     */
    public void generateAndPrintOutTo(File dir, boolean useDiffColors,
            boolean drawOnGrid) throws Exception {
        if (dset == null) {
            NoisyGaussianMix genMix =
                    new NoisyGaussianMix(
                    numClusters,
                    numDimensions,
                    numOriginalInstances,
                    false,
                    numNoisyInstances);
            dset = genMix.generateRandomDataSet();
        } else {
            numClusters = dset.findMaxLabel() + 1;
        }
        numTotalInstances = dset.size();
        if (numNoisyInstances == 0) {
            numOriginalInstances = numTotalInstances;
        }
        colors = new Color[numClusters];
        Random randa = new Random();
        for (int i = 0; i < numClusters; i++) {
            colors[i] = new Color(
                    randa.nextFloat(),
                    randa.nextFloat(),
                    randa.nextFloat(),
                    1f);
        }
        distances = dset.calculateDistMatrix(CombinedMetric.FLOAT_MANHATTAN);
        System.out.println("Dataset prepared");
        System.out.println("Performing MDS (projecting onto 2D)...");
        float[] lBounds = new float[2];
        lBounds[0] = 1;
        lBounds[1] = 1;
        float[] uBounds = new float[2];
        uBounds[0] = IMAGE_X;
        uBounds[1] = IMAGE_Y;
        double[][] fullDist = new double[dset.size()][dset.size()];
        for (int i = 0; i < dset.size(); i++) {
            fullDist[i][i] = 0;
            for (int j = i + 1; j < dset.size(); j++) {
                fullDist[i][j] = distances[i][j - i - 1];
                fullDist[j][i] = distances[i][j - i - 1];
            }
        }
        double[][] resultsReversed = MDSJ.classicalScaling(fullDist);
        pointXY = new float[dset.size()][2];
        float maxX = -Float.MAX_VALUE;
        float maxY = -Float.MAX_VALUE;
        float minX = Float.MAX_VALUE;
        float minY = Float.MAX_VALUE;
        for (int i = 0; i < dset.size(); i++) {
            pointXY[i][0] = (float) resultsReversed[0][i];
            pointXY[i][1] = (float) resultsReversed[1][i];
            if (pointXY[i][0] > maxX) {
                maxX = pointXY[i][0];
            }
            if (pointXY[i][0] < minX) {
                minX = pointXY[i][0];
            }
            if (pointXY[i][1] > maxY) {
                maxY = pointXY[i][1];
            }
            if (pointXY[i][1] < minY) {
                minY = pointXY[i][1];
            }
        }
        // Now rescale the result to fit the window.
        for (int i = 0; i < dset.size(); i++) {
            pointXY[i][0] =
                    ((pointXY[i][0] - minX) / (maxX - minX)) * IMAGE_X;
            pointXY[i][1] =
                    ((pointXY[i][1] - minY) / (maxY - minY)) * IMAGE_Y;
        }
        System.out.println("Multi-dimensional scaling has been performed.");
        for (int i = 0; i < kVals.length; i++) {
            internalPrintOut(dir, useDiffColors, drawOnGrid, kVals[i]);
        }
    }

    /**
     * @param br BufferedImage to draw hub selection history to.
     * @param hubHistory Array list of integer arrays representing the hub
     * selection history during clustering that is to be drawn on the image.
     */
    private void drawHubHistoryFromIndexes(BufferedImage br,
            ArrayList<int[]> hubHistory) {
        Graphics2D g = br.createGraphics();
        if (blackBGD || drawOnGrid) {
            g.setColor(Color.WHITE);
        } else {
            g.setColor(Color.BLACK);
        }
        int[] hubArray;
        int[] prevArray;
        hubArray = hubHistory.get(0);
        for (int i = 1; i < hubHistory.size(); i++) {
            prevArray = hubArray;
            hubArray = hubHistory.get(i);
            for (int j = 0; j < hubArray.length; j++) {
                g.drawLine(
                        (int) (pointXY[prevArray[j]][0]),
                        (int) (pointXY[prevArray[j]][1]),
                        (int) (pointXY[hubArray[j]][0]),
                        (int) (pointXY[hubArray[j]][1]));
            }
        }
    }

    /**
     * @param dir Target directory.
     * @param useDiffColors Boolean variable indicating whether to use different
     * colors for different classes/clusters.
     * @param drawOnGrid Boolean variable indicating whether to draw the results
     * on a grid or as graph nodes.
     * @param k Integer that represents the neighborhood size.
     * @throws Exception
     */
    public void internalPrintOut(File dir,
            boolean useDiffColors, boolean drawOnGrid, int k) throws Exception {
        NeighborSetFinder nsf = new NeighborSetFinder(
                dset, distances, CombinedMetric.FLOAT_MANHATTAN);
        nsf.calculateNeighborSets(k);
        int[] hubnessArray = nsf.getNeighborFrequencies();
        int[] goodHubnessArray = nsf.getGoodFrequencies();
        int[] badHubnessArray = nsf.getBadFrequencies();
        GHPC clusterer = new GHPC(dset, numClusters, k);
        clusterer.setHubness(hubnessArray);
        clusterer.keepHistory(true);
        clusterer.probabilisticIterations = probIters;
        clusterer.cluster();
        ArrayList<int[]> hubsOverIterations = clusterer.getHubHistory();
        BufferedImage br = null;
        File target;
        if (drawOnGrid) {
            if (genericNaming) {
                br = drawAverageGridHubness(
                        pointXY, hubnessArray, dset, useDiffColors);
                drawHubHistoryFromIndexes(br, hubsOverIterations);
                target = new File(
                        dir,
                        "ds_"
                        + numOriginalInstances
                        + "nClust" + numClusters
                        + "nDim" + numDimensions + "k" + k + "GridGHPC.jpg");
                FileUtil.createFile(target);
                ImageIO.write(br, "jpg", target);
            } else {
                br = drawAverageGridHubness(
                        pointXY, hubnessArray, dset, useDiffColors);
                drawHubHistoryFromIndexes(br, hubsOverIterations);
                target = new File(dir, origDName + "k" + k + "GridGHPC.jpg");
                FileUtil.createFile(target);
                ImageIO.write(br, "jpg", target);
            }
        } else {
            if (genericNaming) {
                br = drawDataNodes(pointXY, hubnessArray, dset, useDiffColors);
                drawHubHistoryFromIndexes(br, hubsOverIterations);
                target = new File(
                        dir,
                        "ds_" + numOriginalInstances + "nClust"
                        + numClusters + "nDim" + numDimensions + "k" + k
                        + "GHPC.jpg");
                FileUtil.createFile(target);
                ImageIO.write(br, "jpg", target);
                br = drawAndPersistGBH(
                        pointXY, goodHubnessArray, badHubnessArray, dset);
                drawHubHistoryFromIndexes(br, hubsOverIterations);
                target = new File(
                        dir,
                        "ds_" + numOriginalInstances + "nClust" + numClusters
                        + "nDim" + numDimensions + "GBHk" + k + "GHPC.jpg");
                FileUtil.createFile(target);
                ImageIO.write(br, "jpg", target);
            } else {
                br = drawDataNodes(pointXY, hubnessArray, dset, useDiffColors);
                drawHubHistoryFromIndexes(br, hubsOverIterations);
                target = new File(dir, origDName + "k" + k + "GHPC.jpg");
                FileUtil.createFile(target);
                ImageIO.write(br, "jpg", target);
                br = drawAndPersistGBH(
                        pointXY, goodHubnessArray, badHubnessArray, dset);
                drawHubHistoryFromIndexes(br, hubsOverIterations);
                target = new File(dir, origDName + "GBHk" + k + "GHPC.jpg");
                FileUtil.createFile(target);
                ImageIO.write(br, "jpg", target);
            }
        }
    }

    /**
     * @param pointXY The 2D array representing projected data points.
     * @param hubnessArray Array of neighbor occurrence frequencies - pointwise
     * hubness.
     * @param dset DataSet object.
     * @param useDiffColors Boolean variable indicating whether to use different
     * colors for different classes/clusters.
     * @return BufferedImage that represents hubness density of the projected
     * data.
     * @throws Exception
     */
    public BufferedImage drawAverageGridHubness(float[][] pointXY,
            int[] hubnessArray, DataSet dset, boolean useDiffColors)
            throws Exception {
        float maxHubness = 0;
        BufferedImage br =
                new BufferedImage(IMAGE_X, IMAGE_Y, BufferedImage.TYPE_INT_RGB);
        Graphics2D graphics = br.createGraphics();
        if (!useDiffColors) {
            int[] data = new int[IMAGE_X * IMAGE_Y];
            // First determine the average hubness values in grid cells,then
            // normalize and then calculate and set colors and color the image.
            // Image size must be divisible by 10 in this implementation.
            float[][] grid = new float[IMAGE_X / 10][IMAGE_Y / 10];
            int[][] gridCounts = new int[IMAGE_X / 10][IMAGE_Y / 10];
            int[][] gridColors = new int[IMAGE_X / 10][IMAGE_Y / 10];
            for (int i = 0; i < dset.size(); i++) {
                int gridX = Math.min((int) pointXY[i][0] / 10, grid.length - 1);
                int gridY = Math.min((int) pointXY[i][1] / 10,
                        grid[0].length - 1);
                gridCounts[gridX][gridY]++;
                grid[gridX][gridY] += hubnessArray[i];
            }
            for (int i = 0; i < grid.length; i++) {
                for (int j = 0; j < grid[i].length; j++) {
                    if (gridCounts[i][j] > 0) {
                        grid[i][j] /= (float) gridCounts[i][j];
                        if (grid[i][j] > maxHubness) {
                            maxHubness = grid[i][j];
                        }
                    }
                }
            }
            int r, g, b;
            r = 0;
            g = 0;
            b = 0;
            for (int i = 0; i < grid.length; i++) {
                for (int j = 0; j < grid[i].length; j++) {
                    if (gridCounts[i][j] > 0) {
                        grid[i][j] /= maxHubness;
                    }
                    if (grid[i][j] < 0.7) {
                        r = 0;
                        g = 0;
                        b = (int) (255 * ((grid[i][j])));
                    } else {
                        // Painting them in clear yellow makes high hubness
                        // elements easy to notice on the map.
                        r = 255;
                        g = 255;
                        b = 0;
                    }
                    gridColors[i][j] = (r << 16) | (g << 8) | b;
                }
            }
            int GIx, GIy;
            for (int i = 0; i < data.length; i++) {
                GIx = (i % IMAGE_X) / 10;
                GIy = (i / IMAGE_X) / 10;
                data[i] = gridColors[GIx][GIy];
            }
            br.setRGB(0, 0, IMAGE_X, IMAGE_Y, data, 0, IMAGE_X);
        } else {
        }
        return br;
    }

    /**
     * @param pointXY The 2D array representing projected data points.
     * @param hubnessArray Array of neighbor occurrence frequencies - pointwise
     * hubness.
     * @param dset DataSet object.
     * @param useDiffColors Boolean variable indicating whether to use different
     * colors for different classes/clusters.
     * @return BufferedImage that visualizes data nodes in different sizes based
     * on their hubness.
     * @throws Exception
     */
    public BufferedImage drawDataNodes(float[][] pointXY, int[] hubnessArray,
            DataSet dset, boolean useDiffColors) throws Exception {
        // Calculate size for every point, maximum 50, minimum 2.
        float MAX = 50;
        float MIN = 2;
        // Find the maximum hubness, normalize and then multiply by MAX.
        float maxHubness = 0;
        float[] size = new float[pointXY.length];
        for (int i = 0; i < pointXY.length; i++) {
            if (hubnessArray[i] > maxHubness) {
                maxHubness = hubnessArray[i];
            }
        }
        for (int i = 0; i < pointXY.length; i++) {
            size[i] = Math.max(
                    MIN, ((((float) hubnessArray[i]) / maxHubness) * MAX));
        }
        System.out.println("Finished normalizing, starting to draw...");
        BufferedImage br = new BufferedImage(
                IMAGE_X, IMAGE_Y, BufferedImage.TYPE_INT_RGB);
        Graphics2D graphics = br.createGraphics();
        graphics.setStroke(new BasicStroke(3f));
        if (!blackBGD) {
            graphics.setColor(Color.WHITE);
            graphics.fillRect(0, 0, br.getWidth(), br.getHeight());
        }
        RotatedEllipse[] ellipses = new RotatedEllipse[pointXY.length];
        for (int i = numOriginalInstances; i < numTotalInstances; i++) {
            ellipses[i] = new RotatedEllipse(
                    pointXY[i][0], pointXY[i][1], size[i], size[i], 0);
            ellipses[i].setColor(Color.gray);
        }
        for (int i = 0; i < numOriginalInstances; i++) {
            ellipses[i] = new RotatedEllipse(
                    pointXY[i][0], pointXY[i][1], size[i], size[i], 0);
            if (useDiffColors) {
                if (dset.data.get(i).getCategory() != -1) {
                    ellipses[i].setColor(
                            colors[dset.data.get(i).getCategory()]);
                }
            } else {
                ellipses[i].setColor(Color.red);
            }
        }
        for (int i = 0; i < pointXY.length; i++) {
            ellipses[i].drawOnGraphics(graphics);
        }
        return br;
    }

    /**
     * @param pointXY The 2D array representing projected data points.
     * @param goodHubnessArray Array of good neighbor occurrence frequencies -
     * pointwise good hubness.
     * @param badHubnessArray Array of bad neighbor occurrence frequencies -
     * pointwise bad hubness.
     * @param dset DataSet object.
     * @return BufferedImage that visualizes data nodes in different sizes based
     * on their good and bad hubness.
     * @throws Exception
     */
    public BufferedImage drawAndPersistGBH(
            float[][] pointXY, int[] goodHubnessArray, int[] badHubnessArray,
            DataSet dset) throws Exception {
        // Calculate size for every point, maximum 50, minimum 2.
        float MAX = 50;
        float MIN = 2;
        // Find the maximum hubness, normalize and then multiply by MAX.
        float maxHubness = 0;
        float maxGoodHubness = 0;
        float maxBadHubness = 0;
        float[] size = new float[pointXY.length];
        for (int i = 0; i < pointXY.length; i++) {
            if ((goodHubnessArray[i] + badHubnessArray[i]) > maxHubness) {
                maxHubness = goodHubnessArray[i] + badHubnessArray[i];
            }
            if (goodHubnessArray[i] > maxGoodHubness) {
                maxGoodHubness = goodHubnessArray[i];
            }
            if (badHubnessArray[i] > maxBadHubness) {
                maxBadHubness = badHubnessArray[i];
            }
        }
        for (int i = 0; i < pointXY.length; i++) {
            size[i] = Math.max(MIN,
                    ((((float) (goodHubnessArray[i]
                    + badHubnessArray[i])) / maxHubness) * MAX));
        }
        System.out.println("Finished normalizing, starting to draw...");
        BufferedImage br = new BufferedImage(
                IMAGE_X, IMAGE_Y, BufferedImage.TYPE_INT_RGB);
        Graphics2D graphics = br.createGraphics();
        if (!blackBGD) {
            graphics.setColor(Color.WHITE);
            graphics.fillRect(0, 0, br.getWidth(), br.getHeight());
        }
        RotatedEllipse[] ellipses = new RotatedEllipse[pointXY.length];
        for (int i = numOriginalInstances; i < numTotalInstances; i++) {
            ellipses[i] = new RotatedEllipse(
                    pointXY[i][0], pointXY[i][1], size[i], size[i], 0);
            ellipses[i].setColor(Color.gray);
        }
        int r = 0;
        int g = 0;
        int b = 0;
        Color currColor;
        for (int i = 0; i < numOriginalInstances; i++) {
            ellipses[i] = new RotatedEllipse(
                    pointXY[i][0], pointXY[i][1], size[i], size[i], 0);
            if (badHubnessArray[i] + goodHubnessArray[i] > 0) {
                r = (int) (255 * badHubnessArray[i]
                        / (badHubnessArray[i] + goodHubnessArray[i]));
                g = (int) (255 * goodHubnessArray[i]
                        / (badHubnessArray[i] + goodHubnessArray[i]));
                currColor = new Color(((r << 16) | (g << 8) | b));
                ellipses[i].setColor(currColor);
            } else {
                ellipses[i].setColor(Color.BLUE);
            }
        }
        for (int i = 0; i < pointXY.length; i++) {
            ellipses[i].drawOnGraphics(graphics);
        }
        return br;
    }

    /**
     * Information about how to input the parameters.
     */
    public static void info() {
        System.out.println("arg0: numClusters");
        System.out.println("arg1: numDimensions");
        System.out.println("arg2: numOriginalInstances");
        System.out.println("arg3: numNoisyInstances");
        System.out.println("arg4: outputDirectory");
        System.out.println("arg5: useDifferentColorsForClusters");
        System.out.println("arg6: drawOnGrid");
        System.out.println("arg7: num prob iters");
        System.out.println("---------------------------------------");
        System.out.println("OR");
        System.out.println("---------------------------------------");
        System.out.println("arg0: -CSV::filePath or -ARFF:filePath");
        System.out.println("arg1: with category");
        System.out.println("arg2: outputDirectory");
        System.out.println("arg3: useDifferentColorsForClusters");
        System.out.println("arg4: drawOnGrid");
        System.out.println("arg5: num prob iters");
    }

    /**
     * Main method - input the desired parameters and generate clustering
     * visualizations.
     *
     * @param args A string array of input parameters.
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        if (args.length != 8 && args.length != 6) {
            info();
            return;
        }
        if (args.length == 8) {
            HPCVisualizer visualizer = new HPCVisualizer(
                    Integer.parseInt(args[0]),
                    Integer.parseInt(args[1]),
                    Integer.parseInt(args[2]),
                    Integer.parseInt(args[3]));
            visualizer.probIters = Integer.parseInt(args[7]);
            visualizer.drawOnGrid = Boolean.parseBoolean(args[6]);
            visualizer.generateAndPrintOutTo(
                    new File(args[4]),
                    Boolean.parseBoolean(args[5]),
                    Boolean.parseBoolean(args[6]));
        } else {
            DataSet dc;
            String[] pair = args[0].split("::");
            System.out.println("loading file: " + pair[1]);
            if (pair[0].equals("-CSV")) {
                IOCSV reader = new IOCSV(Boolean.parseBoolean(args[1]), ",");
                dc = reader.readData(new File(pair[1]));
            } else {
                IOARFF arff = new IOARFF();
                dc = arff.load(pair[1]);
            }
            HPCVisualizer visualizer = new HPCVisualizer(dc);
            visualizer.probIters = Integer.parseInt(args[5]);
            String tempName = (new File(pair[1])).getName();
            int index = tempName.lastIndexOf('.');
            visualizer.origDName = tempName.substring(0, index);
            System.out.println("naming name set to: " + visualizer.origDName);
            visualizer.generateAndPrintOutTo(
                    new File(args[2]),
                    Boolean.parseBoolean(args[3]),
                    Boolean.parseBoolean(args[4]));
        }
    }
}
