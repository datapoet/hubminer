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
package data.neighbors.hubness.visualization;

import data.generators.NoisyGaussianMix;
import data.neighbors.NeighborSetFinder;
import data.representation.DataSet;
import distances.primary.CombinedMetric;
import distances.primary.Manhattan;
import draw.basic.RotatedEllipse;
import ioformat.FileUtil;
import ioformat.SupervisedLoader;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.Random;
import javax.imageio.ImageIO;
import mdsj.MDSJ;
import util.ArrayUtil;
import util.CommandLineParser;

/**
 * This script generates data with the designated level of noise, performs MDS
 * and visualizes some properties of data hubness OR loads the data and
 * visualizes the hubness properties of some real-world datasets.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class BasicDataHubnessVisualizer {

    // Number of clusters in the data.
    private int numClusters = 5;
    // Number of dimensions in the data.
    private int numDimensions = 50;
    // Number of data instances that belong to Gaussian clusters.
    private int numOriginalInstances = 1000;
    // Number of noise instances drawn from a uniform distribution.
    private int numNoisyInstances = 0;
    // Total number of instances.
    private int numTotalInstances = 0;
    // Class colors.
    private Color[] colors;
    // DataSet object that contains all of data.
    private DataSet dset = null;
    // Whether to perform generic naming.
    private boolean genericNaming = false;
    private String origDName = null;
    private final static int IMAGE_X = 2000;
    private final static int IMAGE_Y = 1250;
    private static final int[] kValues = {1, 3, 5, 10, 20, 50, 100};
    private static final int MAX_NODE_SIZE = 30;
    private static final int MIN_NODE_SIZE = 2;

    /**
     * Initialization.
     *
     * @param numClusters Integer that is the number of clusters.
     * @param numDimensions Integer that is the number of dimensions.
     * @param numOriginalInstances Integer that is the number of original
     * non-noisy data instances to generate.
     * @param numNoisyInstances Integer that is the number of noisy instances to
     * generate, by drawing them from a uniform distribution.
     */
    public BasicDataHubnessVisualizer(int numClusters, int numDimensions,
            int numOriginalInstances, int numNoisyInstances) {
        this.numClusters = numClusters;
        this.numDimensions = numDimensions;
        this.numOriginalInstances = numOriginalInstances;
        this.numNoisyInstances = numNoisyInstances;
        genericNaming = true;
    }

    /**
     * Initialization.
     *
     * @param dset DataSet object to visualize.
     */
    public BasicDataHubnessVisualizer(DataSet dset) {
        this.dset = dset;
    }

    /**
     * Generate the visualizations and save them to a file.
     *
     * @param outDir Output directory.
     * @param useDiffColors Boolean flag indicating whether to use a
     * multi-colored visualization scheme.
     * @param drawOnGrid Boolean flag indicating whether to draw on grid and
     * make a density plot or to draw nodes instead.
     * @throws Exception
     */
    public void generateAndPrintOutTo(File outDir, boolean useDiffColors,
            boolean drawOnGrid) throws Exception {
        if (dset == null) {
            // Synthetic case, generate data.
            NoisyGaussianMix genMix = new NoisyGaussianMix(numClusters,
                    numDimensions, numOriginalInstances, false,
                    numNoisyInstances);
            dset = genMix.generateRandomDataSet();
        } else {
            // Data already provided.
            numClusters = dset.findMaxLabel() + 1;
        }
        numTotalInstances = dset.size();
        if (numNoisyInstances == 0) {
            numOriginalInstances = numTotalInstances;
        }
        colors = new Color[numClusters];
        Random randa = new Random();
        for (int cIndex = 0; cIndex < numClusters; cIndex++) {
            // Generate random colors.
            colors[cIndex] = new Color(randa.nextFloat(), randa.nextFloat(),
                    randa.nextFloat(), 1f);
        }
        // Initialize the metric object.
        CombinedMetric cmet = new CombinedMetric(null, new Manhattan(),
                CombinedMetric.DEFAULT);
        // Initialize the kNN finder.
        NeighborSetFinder nsf = new NeighborSetFinder(dset, cmet);
        nsf.calculateDistances();
        System.out.println("Calculated distances.");
        float[][] dMatUpperTriangular = nsf.getDistances();
        double[][] dMatSquare = new double[dset.size()][dset.size()];
        // Form a square distance matrix for the MDSJ component.
        for (int i = 0; i < dset.size(); i++) {
            dMatSquare[i][i] = 0;
            for (int j = i + 1; j < dset.size(); j++) {
                dMatSquare[i][j] = dMatUpperTriangular[i][j - i - 1];
                dMatSquare[j][i] = dMatUpperTriangular[i][j - i - 1];
            }
        }
        // Perform multi-dimensional scaling.
        double[][] scalingResults = MDSJ.classicalScaling(dMatSquare);
        System.out.println("MDS completed.");
        float[][] pointXY = new float[dset.size()][2];
        float maxX = -Float.MAX_VALUE;
        float maxY = -Float.MAX_VALUE;
        float minX = Float.MAX_VALUE;
        float minY = Float.MAX_VALUE;
        // Find the bounding rectangle of the scaled results.
        for (int i = 0; i < dset.size(); i++) {
            pointXY[i][0] = (float) scalingResults[0][i];
            pointXY[i][1] = (float) scalingResults[1][i];
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
        // Re-scale the coordinates to fit the display.
        for (int i = 0; i < dset.size(); i++) {
            pointXY[i][0] = ((pointXY[i][0] - minX) / (maxX - minX)) * IMAGE_X;
            pointXY[i][1] = ((pointXY[i][1] - minY) / (maxY - minY)) * IMAGE_Y;
        }
        System.out.println("Calculating kNN sets.");
        int kMax = ArrayUtil.max(kValues);
        nsf.calculateNeighborSets(kMax);
        System.out.println("Calculated kNN sets.");
        // Initialize the hubness arrays.
        int[] hubnessArray;
        int[] goodHubnessArray;
        int[] badHubnessArray;
        for (int k : kValues) {
            // For every specified neighborhood size.
            System.out.println("Analyzing data for k = " + k);
            NeighborSetFinder nsfIter = nsf.copy();
            // Re-calculate the kNN sets.
            nsfIter.recalculateStatsForSmallerK(k);
            // Get the hubness arrays.
            hubnessArray = nsfIter.getNeighborFrequencies();
            goodHubnessArray = nsfIter.getGoodFrequencies();
            badHubnessArray = nsfIter.getBadFrequencies();
            // Select a proper drawing scheme.
            if (drawOnGrid) {
                if (genericNaming) {
                    drawAverageGridHubness(pointXY, hubnessArray, new File(
                            outDir, "ds_" + numOriginalInstances + "nClust"
                            + numClusters + "nDim" + numDimensions + "k" + k
                            + "Grid.jpg"), dset);
                } else {
                    drawAverageGridHubness(pointXY, hubnessArray, new File(
                            outDir, origDName + "k" + k + "Grid.jpg"), dset);
                }
            } else {
                if (genericNaming) {
                    drawAndPersist(pointXY, hubnessArray, new File(outDir,
                            "ds_" + numOriginalInstances + "nClust"
                            + numClusters + "nDim" + numDimensions + "k" + k
                            + ".jpg"), dset, useDiffColors);
                    drawAndPersistGBH(pointXY, goodHubnessArray,
                            badHubnessArray, new File(outDir, "ds_"
                            + numOriginalInstances + "nClust" + numClusters
                            + "nDim" + numDimensions + "GBHk" + k + ".jpg"),
                            dset);
                } else {
                    drawAndPersist(pointXY, hubnessArray, new File(outDir,
                            origDName + "k" + k + ".jpg"), dset, useDiffColors);
                    drawAndPersistGBH(pointXY, goodHubnessArray,
                            badHubnessArray, new File(outDir, origDName
                            + "GBHk" + k + ".jpg"), dset);
                }
            }
        }
    }

    /**
     * Draw the average data hubness on a grid, a density-plot.
     *
     * @param pointXY float[][] point coordinates determined by MDS.
     * @param hubnessArray int[] array of neighbor occurrence frequencies.
     * @param outFile File to save the results to.
     * @param dset DataSet object that the results refer to.
     * @throws Exception
     */
    public void drawAverageGridHubness(float[][] pointXY, int[] hubnessArray,
            File outFile, DataSet dset)
            throws Exception {
        FileUtil.createFile(outFile);
        float maxHubness = 0;
        // Create a new BufferedImage object to draw on.
        BufferedImage visualization = new BufferedImage(IMAGE_X, IMAGE_Y,
                BufferedImage.TYPE_INT_RGB);
        int[] outputRaster = new int[IMAGE_X * IMAGE_Y];
        // First determine the average hubness values in grid cells, then
        // normalize and calculate the color values.
        // In this implementation, image size must be divisible by 10.
        float[][] grid = new float[IMAGE_X / 10][IMAGE_Y / 10];
        int[][] gridCounts = new int[IMAGE_X / 10][IMAGE_Y / 10];
        int[][] gridColors = new int[IMAGE_X / 10][IMAGE_Y / 10];
        for (int i = 0; i < dset.size(); i++) {
            int gridX = Math.min((int) pointXY[i][0] / 10, grid.length - 1);
            int gridY = Math.min((int) pointXY[i][1] / 10, grid[0].length - 1);
            gridCounts[gridX][gridY]++;
            grid[gridX][gridY] += hubnessArray[i];
        }
        // Average out and get the scaling factor.
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
        // Red, green and blue pixel values.
        int r, g, b;
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[i].length; j++) {
                // Normalize to [0, 1].
                if (gridCounts[i][j] > 0) {
                    grid[i][j] /= maxHubness;
                }
                if (grid[i][j] < 0.7) {
                    r = 0;
                    g = 0;
                    b = (int) (255 * ((grid[i][j])));
                } else {
                    // Painting them in clear yellow makes high hubness elements
                    // easy to notice on the map.
                    r = 255;
                    g = 255;
                    b = 0;
                }
                gridColors[i][j] = (r << 16) | (g << 8) | b;
            }
        }
        // Update the raster.
        int xGrid, yGrid;
        for (int i = 0; i < outputRaster.length; i++) {
            xGrid = (i % IMAGE_X) / 10;
            yGrid = (i / IMAGE_X) / 10;
            outputRaster[i] = gridColors[xGrid][yGrid];
        }
        visualization.setRGB(0, 0, IMAGE_X, IMAGE_Y, outputRaster, 0, IMAGE_X);
        ImageIO.write(visualization, "jpg", outFile);
    }

    /**
     * Performs data visualization. In the multi-color mode, the points from
     * different classes will be drawn in different colors.
     *
     * @param pointXY float[][] point coordinates determined by MDS.
     * @param hubnessArray int[] array of neighbor occurrence frequencies.
     * @param outFile File to save the results to.
     * @param dset DataSet object that the results refer to.
     * @param useDiffColors Boolean flag indicating whether to use a
     * multi-colored visualization scheme.
     * @throws Exception
     */
    public void drawAndPersist(float[][] pointXY, int[] hubnessArray,
            File outFile, DataSet dset, boolean useDiffColors)
            throws Exception {
        FileUtil.createFile(outFile);
        // Calculate the size for every data node.
        float maxHubness = ArrayUtil.max(hubnessArray);
        float[] nodeSize = new float[pointXY.length];
        for (int i = 0; i < pointXY.length; i++) {
            nodeSize[i] = Math.max(MIN_NODE_SIZE,
                    ((((float) hubnessArray[i]) / maxHubness) * MAX_NODE_SIZE));
        }
        System.out.println("Finished normalizing, starting to draw...");
        // Generate an image to draw on.
        BufferedImage visualization = new BufferedImage(IMAGE_X, IMAGE_Y,
                BufferedImage.TYPE_INT_RGB);
        Graphics2D graphics = visualization.createGraphics();
        RotatedEllipse[] ellipses = new RotatedEllipse[pointXY.length];
        // Draw all the noisy nodes.
        for (int i = numOriginalInstances; i < numTotalInstances; i++) {
            ellipses[i] = new RotatedEllipse(pointXY[i][0], pointXY[i][1],
                    nodeSize[i], nodeSize[i], 0);
            ellipses[i].setColor(Color.gray);
        }
        // Draw all the non-noisy nodes.
        for (int i = 0; i < numOriginalInstances; i++) {
            ellipses[i] = new RotatedEllipse(pointXY[i][0], pointXY[i][1],
                    nodeSize[i], nodeSize[i], 0);
            if (useDiffColors) {
                if (dset.data.get(i).getCategory() != -1) {
                    ellipses[i].setColor(colors[dset.data.get(i).
                            getCategory()]);
                }
            } else {
                ellipses[i].setColor(Color.red);
            }
        }
        for (int i = 0; i < pointXY.length; i++) {
            ellipses[i].drawOnGraphics(graphics);
        }
        ImageIO.write(visualization, "jpg", outFile);
    }

    /**
     * Performs data visualization. Good and bad hubness is taken into account
     * to produce images where nodes are green if they are good hubs and red if
     * they are bad hubs.
     *
     * @param pointXY float[][] point coordinates determined by MDS.
     * @param hubnessArray int[] array of neighbor occurrence frequencies.
     * @param outFile File to save the results to.
     * @param dset DataSet object that the results refer to.
     * @param useDiffColors Boolean flag indicating whether to use a
     * multi-colored visualization scheme.
     * @throws Exception
     */
    public void drawAndPersistGBH(float[][] pointXY, int[] goodHubnessArray,
            int[] badHubnessArray, File outFile, DataSet dset)
            throws Exception {
        FileUtil.createFile(outFile);
        // Calculate the size for every data node.
        float maxHubness = 0;
        float maxGoodHubness = 0;
        float maxBadHubness = 0;
        float[] nodeSize = new float[pointXY.length];
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
            nodeSize[i] = Math.max(MIN_NODE_SIZE, ((((float)
                    (goodHubnessArray[i] + badHubnessArray[i]))
                    / maxHubness) * MAX_NODE_SIZE));
        }
        System.out.println("Finished normalizing, starting to draw...");
        // Generate an image to draw on.
        BufferedImage visualization = new BufferedImage(IMAGE_X, IMAGE_Y,
                BufferedImage.TYPE_INT_RGB);
        Graphics2D graphics = visualization.createGraphics();
        RotatedEllipse[] ellipses = new RotatedEllipse[pointXY.length];
        // Draw all the noisy instances.
        for (int i = numOriginalInstances; i < numTotalInstances; i++) {
            ellipses[i] = new RotatedEllipse(pointXY[i][0], pointXY[i][1],
                    nodeSize[i], nodeSize[i], 0);
            ellipses[i].setColor(Color.gray);
        }
        int r;
        int g;
        int b = 0;
        Color currColor;
        // Draw all the non-noisy instances.
        for (int i = 0; i < numOriginalInstances; i++) {
            ellipses[i] = new RotatedEllipse(pointXY[i][0], pointXY[i][1],
                    nodeSize[i], nodeSize[i], 0);
            if (badHubnessArray[i] + goodHubnessArray[i] > 0) {
                r = (int) (255 * badHubnessArray[i] / (badHubnessArray[i]
                        + goodHubnessArray[i]));
                g = (int) (255 * goodHubnessArray[i] / (badHubnessArray[i]
                        + goodHubnessArray[i]));
                currColor = new Color(((r << 16) | (g << 8) | b));
                ellipses[i].setColor(currColor);
            } else {
                ellipses[i].setColor(Color.BLUE);
            }
        }
        for (int i = 0; i < pointXY.length; i++) {
            ellipses[i].drawOnGraphics(graphics);
        }
        ImageIO.write(visualization, "jpg", outFile);
    }

    /**
     * This script performs some basic hubness visualization for synthetic
     * Gaussian or loaded real-world data.
     *
     * @param args Command line parameters, as specified.
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        CommandLineParser clp = new CommandLineParser(true);
        clp.addParam("-mode", "'synthetic' or 'real-world'.",
                CommandLineParser.STRING, true, false);
        clp.setIgnoreExtraParams(true);
        clp.parseLine(args);
        String mode = (String) clp.getParamValues("-mode").get(0);
        if (mode.equalsIgnoreCase("synthetic")) {
            clp = new CommandLineParser(true);
            clp.addParam("-numClust", "Number of clusters.",
                    CommandLineParser.INTEGER, true, false);
            clp.addParam("-dim", "Number of dimensions.",
                    CommandLineParser.INTEGER, true, false);
            clp.addParam("-numNonNoisy", "Number of non-noisy Gaussian points.",
                    CommandLineParser.INTEGER, true, false);
            clp.addParam("-numNoisy", "Number of noisy uniform data points.",
                    CommandLineParser.INTEGER, true, false);
            clp.addParam("-outDir", "Output directory.",
                    CommandLineParser.STRING, true, false);
            clp.addParam("-multicolor", "True / false.",
                    CommandLineParser.BOOLEAN, true, false);
            clp.addParam("-densityMap", "True / false.",
                    CommandLineParser.BOOLEAN, true, false);
            clp.setIgnoreExtraParams(true);
            clp.parseLine(args);
            int numClust = (Integer) clp.getParamValues("-numClust").get(0);
            int numDim = (Integer) clp.getParamValues("-dim").get(0);
            int numNonNoisy = (Integer) clp.getParamValues("-numNonNoisy").
                    get(0);
            int numNoisy = (Integer) clp.getParamValues("-numNoisy").get(0);
            File outDir = new File((String) clp.getParamValues("-outDir").
                    get(0));
            boolean multiColor = (Boolean) clp.getParamValues("-multicolor").
                    get(0);
            boolean densityMap = (Boolean) clp.getParamValues("-densityMap").
                    get(0);
            BasicDataHubnessVisualizer visualizer =
                    new BasicDataHubnessVisualizer(numClust, numDim,
                    numNonNoisy, numNoisy);
            visualizer.generateAndPrintOutTo(outDir, multiColor, densityMap);
        } else if (mode.equalsIgnoreCase("real-world")) {
            clp = new CommandLineParser(true);
            clp.addParam("-inFile", "Input file.",
                    CommandLineParser.STRING, true, false);
            clp.addParam("-outDir", "Output directory.",
                    CommandLineParser.STRING, true, false);
            clp.addParam("-multicolor", "True / false.",
                    CommandLineParser.BOOLEAN, true, false);
            clp.addParam("-densityMap", "True / false.",
                    CommandLineParser.BOOLEAN, true, false);
            clp.setIgnoreExtraParams(true);
            clp.parseLine(args);
            File inFile = new File((String) clp.getParamValues("-inFile").
                    get(0));
            File outDir = new File((String) clp.getParamValues("-outDir").
                    get(0));
            boolean multiColor = (Boolean) clp.getParamValues("-multicolor").
                    get(0);
            boolean densityMap = (Boolean) clp.getParamValues("-densityMap").
                    get(0);
            DataSet dset = SupervisedLoader.loadData(inFile, false);
            BasicDataHubnessVisualizer visualizer =
                    new BasicDataHubnessVisualizer(dset);
            String fileName = inFile.getName();
            int index = fileName.lastIndexOf('.');
            visualizer.origDName = fileName.substring(0, index);
            visualizer.generateAndPrintOutTo(outDir, multiColor, densityMap);
        } else {
            System.err.println("Incorrect mode specified.");
        }
    }
}
