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

import data.neighbors.NeighborSetFinder;
import data.representation.DataInstance;
import data.representation.DataSet;
import distances.primary.CombinedMetric;
import distances.primary.Minkowski3D;
import ioformat.FileUtil;
import ioformat.SupervisedLoader;
import java.awt.Image;
import java.awt.Toolkit;
import java.awt.image.BufferedImage;
import java.awt.image.MemoryImageSource;
import java.io.File;
import java.util.ArrayList;
import javax.imageio.ImageIO;
import learning.supervised.Classifier;
import learning.supervised.interfaces.DistMatrixUserInterface;
import learning.supervised.methods.knn.HFNN;
import learning.supervised.methods.knn.HwKNN;
import learning.supervised.methods.knn.HIKNN;
import learning.supervised.methods.knn.HIKNNNonDW;
import learning.supervised.methods.knn.KNN;
import learning.supervised.methods.knn.NHBNN;
import data.neighbors.NSFUserInterface;

/**
 * This class helps with visualizing the kNN structure and the hubness of the
 * data by visualizing it in 3D instead of 2D, which can help since any
 * additional dimension makes for a better approximation of the skewness in the
 * original high-dimensional space. Of course, this is merely for show, as it is
 * also more difficult to interpret 3D visualization. The script takes as input
 * a 3D dataset (if not, it'll just use the first three dimensions) and then
 * takes 500 2D slices in each direction, amounting to 1500 slices for each
 * algorithm. It calculates the kNN sets for k = 1, 5, 10 and the corresponding
 * probability predictions for each sampled voxel. The script uses alpha
 * compositing to build a single 3D->2D projection for each direction and each
 * evaluated kNN classification algorithm and each k value. The output is then
 * persisted to automatically generated subdirectories of the specified target
 * output directory. As this class was a proof-of-concept of sorts, it is still
 * to be optimized, as it doesn't perform fast 3D Voronoi tesselation, but has a
 * naive voxel classification implementation instead. Still, it was used to
 * generate some nice visualizations.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class Neighbor3DColorVisualizer {

    private DataSet dset;
    private File outDir;
    private int width = 500;
    private int height = 500;
    private int length = 500;
    private float[][] dMat;
    private NeighborSetFinder nsf;

    /**
     * Initialization.
     *
     * @param dset DataSet object to visualize.
     * @param outDir Output directory.
     */
    public Neighbor3DColorVisualizer(DataSet dset, File outDir) {
        this.dset = dset;
        this.outDir = outDir;
    }

    /**
     * Initialization.
     *
     * @param dset DataSet object to visualize.
     * @param outDir Output directory.
     * @param dMat float[][] that is the upper triangular distance matrix.
     * @param nsf NeighborSetFinder object for kNN calculations.
     */
    public Neighbor3DColorVisualizer(DataSet dset, File outDir, float[][] dMat,
            NeighborSetFinder nsf) {
        this.dset = dset;
        this.outDir = outDir;
        this.dMat = dMat;
        this.nsf = nsf;
    }

    /**
     * @param nsf NeighborSetFinder object for kNN calculations.
     */
    public void setNSF(NeighborSetFinder nsf) {
        this.nsf = nsf;
    }

    /**
     * @param dMat float[][] that is the upper triangular distance matrix.
     */
    public void setDistances(float[][] dMat) {
        this.dMat = dMat;
    }

    /**
     * This script-like method generates the 3D visualizations for a single
     * specified algorithm.
     *
     * @param algName String that is the algorithm name.
     * @param k Integer that is the neighborhood size.
     * @throws Exception
     */
    public void generateImagesForAlgorithm(String algName,
            int k) throws Exception {
        Classifier classifier = null;
        // Get the CombinedMetric object for distance calculations.
        CombinedMetric cmet = new CombinedMetric(null, new Minkowski3D(),
                CombinedMetric.DEFAULT);
        // Get the number of classes.
        int numClasses = dset.countCategories();
        // Get the appropriate classifier object.
        if (algName.equals("KNN")) {
            classifier = new KNN(k, cmet);
        } else if (algName.equals("NHBNN")) {
            classifier = new NHBNN(k, cmet, numClasses);
        } else if (algName.equals("HIKNN")) {
            classifier = new HIKNN(k, cmet, numClasses);
        } else if (algName.equals("HIKNNnonWeighted")) {
            classifier = new HIKNNNonDW(k, cmet, numClasses);
        } else if (algName.equals("hwKNN")) {
            classifier = new HwKNN(numClasses, cmet, k);
        } else if (algName.equals("hFNN")) {
            classifier = new HFNN(k, cmet, numClasses);
        } else {
            System.err.println("Bad classifier name.");
            return;
        }
        // Generate the appropriate output sub-directory.
        File outSubDir = new File(outDir, "k" + k + File.separator + algName);
        if (!outSubDir.exists()) {
            FileUtil.createDirectory(outSubDir);
        }
        // Set the data to the classifier for training.
        ArrayList<Integer> dIndexes = new ArrayList(dset.size());
        for (int j = 0; j < dset.size(); j++) {
            dIndexes.add(j);
        }
        classifier.setDataIndexes(dIndexes, dset);
        // Re-use the distance matrix and the kNN sets, if possible.
        if (classifier instanceof DistMatrixUserInterface) {
            ((DistMatrixUserInterface) classifier).setDistMatrix(dMat);
        }
        if (classifier instanceof NSFUserInterface) {
            ((NSFUserInterface) classifier).setNSF(nsf);
        }
        // Learn the model.
        System.out.println("Training the classifier " + algName);
        classifier.train();
        System.out.println("Classifier trained");

        // Class affiliation probabilities of a single classification query.
        float[] classProbabilities;
        // Pixel raster for a single slice of visualization.
        int[] slicePixelRaster;
        // Intensity of individual pixels, between 0 and 255. It is float to
        // avoid losing precision during calculations.
        float[] pixelIntensity;
        // Illumination in different points.
        float[] lightIntensity;
        // Auxiliary variables for the RGB calculations.
        int red, green, blue;
        int rgba;

        // Current output file.
        File currentOutFile;
        // Visualization slice images.
        BufferedImage visBufferedImage;
        Image visImage;

        // All slices.
        float[][][] slices = new float[500][numClasses][width * height];
        // Current illumination value.
        float illumination;
        // Normals in each voxel.
        float[][][][][] pointNormals = new float[numClasses][width][height][
                length][3];
        // Auxiliary variables for calculating point normals.
        float sum, sumX, sumY, sumZ, neighborProbability, currProbability;
        DataInstance instance;
        // Calculate all the point class probabilities.
        for (int z = 0; z < length; z++) {
            for (int x = 0; x < width; x++) {
                for (int y = 0; y < height; y++) {
                    instance = new DataInstance(dset);
                    instance.fAttr[0] = (float) x / (float) width;
                    instance.fAttr[1] = (float) y / (float) height;
                    instance.fAttr[2] = (float) z / (float) length;
                    classProbabilities = classifier.classifyProbabilistically(
                            instance);
                    for (int c = 0; c < numClasses; c++) {
                        slices[z][c][y * width + x] = classProbabilities[c];
                    }
                }
            }
        }
        // Calculate all the normals.
        for (int c = 0; c < numClasses; c++) {
            for (int x = 0; x < width; x++) {
                for (int y = 0; y < height; y++) {
                    for (int z = 0; z < length; z++) {
                        // sumX, sumY and sumZ correspond to the components of
                        // the normal vector.
                        sumX = 0;
                        sumY = 0;
                        sumZ = 0;
                        sum = 0;
                        currProbability = slices[z][c][y * width + x];
                        // We compare the probability density in the current
                        // point with the probability density in the neighboring
                        // points.
                        if (x + 1 < width) {
                            neighborProbability =
                                    slices[z][c][y * width + x + 1];
                            sumX += (neighborProbability - currProbability);
                        }
                        if (x + 1 < width && y + 1 < height) {
                            neighborProbability =
                                    slices[z][c][(y + 1) * width + x + 1];
                            sumX += (neighborProbability - currProbability);
                            sumY += (neighborProbability - currProbability);
                        }
                        if (x + 1 < width && y - 1 >= 0) {
                            neighborProbability =
                                    slices[z][c][(y - 1) * width + x + 1];
                            sumX += (neighborProbability - currProbability);
                            sumY -= (neighborProbability - currProbability);
                        }
                        if (x + 1 < width && z + 1 < length) {
                            neighborProbability =
                                    slices[z + 1][c][y * width + x + 1];
                            sumX += (neighborProbability - currProbability);
                            sumZ += (neighborProbability - currProbability);
                        }
                        if (x + 1 < width && y + 1 < height && z + 1 < length) {
                            neighborProbability =
                                    slices[z + 1][c][(y + 1) * width + x + 1];
                            sumX += (neighborProbability - currProbability);
                            sumY += (neighborProbability - currProbability);
                            sumZ += (neighborProbability - currProbability);
                        }
                        if (x + 1 < width && y - 1 >= 0 && z + 1 < length) {
                            neighborProbability =
                                    slices[z + 1][c][(y - 1) * width + x + 1];
                            sumX += (neighborProbability - currProbability);
                            sumY -= (neighborProbability - currProbability);
                            sumZ += (neighborProbability - currProbability);
                        }
                        if (x + 1 < width && z - 1 >= 0) {
                            neighborProbability =
                                    slices[z - 1][c][y * width + x + 1];
                            sumX += (neighborProbability - currProbability);
                            sumZ -= (neighborProbability - currProbability);
                        }
                        if (x + 1 < width && y + 1 < height && z - 1 >= 0) {
                            neighborProbability =
                                    slices[z - 1][c][(y + 1) * width + x + 1];
                            sumX += (neighborProbability - currProbability);
                            sumY += (neighborProbability - currProbability);
                            sumZ -= (neighborProbability - currProbability);
                        }
                        if (x + 1 < width && y - 1 >= 0 && z - 1 >= 0) {
                            neighborProbability =
                                    slices[z - 1][c][(y - 1) * width + x + 1];
                            sumX += (neighborProbability - currProbability);
                            sumY -= (neighborProbability - currProbability);
                            sumZ -= (neighborProbability - currProbability);
                        }
                        if (y + 1 < height) {
                            neighborProbability =
                                    slices[z][c][(y + 1) * width + x];
                            sumY += (neighborProbability - currProbability);
                        }
                        if (y - 1 >= 0) {
                            neighborProbability =
                                    slices[z][c][(y - 1) * width + x];
                            sumY -= (neighborProbability - currProbability);
                        }
                        if (z + 1 < length) {
                            neighborProbability =
                                    slices[z + 1][c][y * width + x];
                            sumZ += (neighborProbability - currProbability);
                        }
                        if (y + 1 < height && z + 1 < length) {
                            neighborProbability =
                                    slices[z + 1][c][(y + 1) * width + x];
                            sumY += (neighborProbability - currProbability);
                            sumZ += (neighborProbability - currProbability);
                        }
                        if (y - 1 >= 0 && z + 1 < length) {
                            neighborProbability =
                                    slices[z + 1][c][(y - 1) * width + x];
                            sumY -= (neighborProbability - currProbability);
                            sumZ += (neighborProbability - currProbability);
                        }
                        if (z - 1 >= 0) {
                            neighborProbability =
                                    slices[z - 1][c][y * width + x];
                            sumZ -= (neighborProbability - currProbability);
                        }
                        if (y + 1 < height && z - 1 >= 0) {
                            neighborProbability =
                                    slices[z - 1][c][(y + 1) * width + x];
                            sumY += (neighborProbability - currProbability);
                            sumZ -= (neighborProbability - currProbability);
                        }
                        if (y - 1 >= 0 && z - 1 >= 0) {
                            neighborProbability =
                                    slices[z - 1][c][(y - 1) * width + x];
                            sumY -= (neighborProbability - currProbability);
                            sumZ -= (neighborProbability - currProbability);
                        }
                        if (x - 1 >= 0) {
                            neighborProbability =
                                    slices[z][c][y * width + x - 1];
                            sumX -= (neighborProbability - currProbability);
                        }
                        if (x - 1 >= 0 && y + 1 < height) {
                            neighborProbability =
                                    slices[z][c][(y + 1) * width + x - 1];
                            sumX -= (neighborProbability - currProbability);
                            sumY += (neighborProbability - currProbability);
                        }
                        if (x - 1 >= 0 && y - 1 >= 0) {
                            neighborProbability =
                                    slices[z][c][(y - 1) * width + x - 1];
                            sumX -= (neighborProbability - currProbability);
                            sumY -= (neighborProbability - currProbability);
                        }
                        if (x - 1 >= 0 && z + 1 < length) {
                            neighborProbability =
                                    slices[z + 1][c][y * width + x - 1];
                            sumX -= (neighborProbability - currProbability);
                            sumZ += (neighborProbability - currProbability);
                        }
                        if (x - 1 >= 0 && y + 1 < height && z + 1 < length) {
                            neighborProbability =
                                    slices[z + 1][c][(y + 1) * width + x - 1];
                            sumX -= (neighborProbability - currProbability);
                            sumY += (neighborProbability - currProbability);
                            sumZ += (neighborProbability - currProbability);
                        }
                        if (x - 1 >= 0 && y - 1 >= 0 && z + 1 < length) {
                            neighborProbability =
                                    slices[z + 1][c][(y - 1) * width + x - 1];
                            sumX -= (neighborProbability - currProbability);
                            sumY -= (neighborProbability - currProbability);
                            sumZ += (neighborProbability - currProbability);
                        }
                        if (x - 1 >= 0 && z - 1 >= 0) {
                            neighborProbability =
                                    slices[z - 1][c][y * width + x - 1];
                            sumX -= (neighborProbability - currProbability);
                            sumZ -= (neighborProbability - currProbability);
                        }
                        if (x - 1 >= 0 && y + 1 < height && z - 1 >= 0) {
                            neighborProbability =
                                    slices[z - 1][c][(y + 1) * width + x - 1];
                            sumX -= (neighborProbability - currProbability);
                            sumY += (neighborProbability - currProbability);
                            sumZ -= (neighborProbability - currProbability);
                        }
                        if (x - 1 >= 0 && y - 1 >= 0 && z - 1 >= 0) {
                            neighborProbability =
                                    slices[z - 1][c][(y - 1) * width + x - 1];
                            sumX -= (neighborProbability - currProbability);
                            sumY -= (neighborProbability - currProbability);
                            sumZ -= (neighborProbability - currProbability);
                        }

                        sum += Math.sqrt(Math.pow(sumX, 2)
                                + Math.pow(sumY, 2) + Math.pow(sumZ, 2));
                        if (sum > 0) {
                            sumX /= sum;
                            sumY /= sum;
                            sumZ /= sum;
                        }
                        pointNormals[c][x][y][z][0] = sumX;
                        pointNormals[c][x][y][z][1] = sumY;
                        pointNormals[c][x][y][z][2] = sumZ;
                    }
                }
            }
        }
        // We begin by taking slices by z and combine the slices in both
        // directions so as to get two views.
        for (int c = 0; c < numClasses; c++) {
            slicePixelRaster = new int[width * height];
            pixelIntensity = new float[width * height];
            lightIntensity = new float[width * height];
            for (int i = 0; i < slicePixelRaster.length; i++) {
                pixelIntensity[i] = 255;
                lightIntensity[i] = 255;
            }
            // Alpha-compositing.
            for (int z = 0; z < 500; z++) {
                for (int x = 0; x < width; x++) {
                    for (int y = 0; y < height; y++) {
                        pixelIntensity[y * width + x] =
                                pixelIntensity[y * width + x]
                                * (1 - slices[z][c][y * width + x])
                                + ((slices[z][c][y * width + x])
                                * (1 - slices[z][c][y * width + x]) * 255);
                        illumination = Math.max(0, -1
                                * pointNormals[c][x][y][z][2]);
                        lightIntensity[y * width + x] =
                                lightIntensity[y * width + x]
                                * (1 - slices[z][c][y * width + x])
                                + (slices[z][c][y * width + x])
                                * illumination * 255;
                    }
                }
            }
            // Pixel coloring.
            for (int x = 0; x < width; x++) {
                for (int y = 0; y < height; y++) {
                    red = (int) (pixelIntensity[y * width + x] * 0.67f + 0.33f
                            * lightIntensity[y * width + x]);
                    green = (int) (pixelIntensity[y * width + x] * 0.67f);
                    blue = (int) (pixelIntensity[y * width + x] * 0.67f);
                    rgba = (0xff000000 | red << 16 | green << 8 | blue);
                    slicePixelRaster[y * width + x] = rgba;
                }
            }
            currentOutFile = new File(outSubDir, "xyPosDir" + "class" + c
                    + ".jpg");
            visBufferedImage = new BufferedImage(width, height,
                    BufferedImage.TYPE_INT_RGB);
            visImage = Toolkit.getDefaultToolkit().createImage(
                    new MemoryImageSource(width, height, slicePixelRaster, 0,
                    width));
            visBufferedImage.getGraphics().drawImage(visImage, 0, 0, null);
            try {
                ImageIO.write(visBufferedImage, "JPG", currentOutFile);
            } catch (Exception e) {
                System.err.println(e.getMessage());
            }
            // The other direction.
            slicePixelRaster = new int[width * height];
            pixelIntensity = new float[width * height];
            lightIntensity = new float[width * height];
            for (int i = 0; i < slicePixelRaster.length; i++) {
                pixelIntensity[i] = 255;
                lightIntensity[i] = 255;
            }
            // Alpha-compositing.
            for (int z = 0; z < 500; z++) {
                for (int x = 0; x < width; x++) {
                    for (int y = 0; y < height; y++) {
                        pixelIntensity[y * width + x] =
                                pixelIntensity[y * width + x]
                                * (1 - slices[499 - z][c][y * width + x])
                                + ((slices[499 - z][c][y * width + x])
                                * (1 - slices[499 - z][c][y * width + x]) *
                                255);
                        illumination = Math.max(0, 1
                                * pointNormals[c][x][y][499 - z][2]);
                        lightIntensity[y * width + x] =
                                lightIntensity[y * width + x]
                                * (1 - slices[499 - z][c][y * width + x])
                                + (slices[499 - z][c][y * width + x])
                                * illumination * 255;
                    }
                }
            }
            // Pixel coloring.
            for (int x = 0; x < width; x++) {
                for (int y = 0; y < height; y++) {
                    red = (int) (pixelIntensity[y * width + x] * 0.67f
                            + 0.33f * lightIntensity[y * width + x]);
                    green = (int) (pixelIntensity[y * width + x] * 0.67f);
                    blue = (int) (pixelIntensity[y * width + x] * 0.67f);
                    rgba = (0xff000000 | red << 16 | green << 8 | blue);
                    slicePixelRaster[y * width + x] = rgba;
                }
            }
            currentOutFile = new File(outSubDir, "xyNegDir" + "class" + c
                    + ".jpg");
            visBufferedImage = new BufferedImage(width, height,
                    BufferedImage.TYPE_INT_RGB);
            visImage = Toolkit.getDefaultToolkit().createImage(
                    new MemoryImageSource(width, height, slicePixelRaster, 0,
                    width));
            visBufferedImage.getGraphics().drawImage(visImage, 0, 0, null);
            try {
                ImageIO.write(visBufferedImage, "JPG", currentOutFile);
            } catch (Exception e) {
                System.err.println(e.getMessage());
            }
            System.out.println("One direction finished");
        }
        // Here we take slices by x.
        for (int c = 0; c < numClasses; c++) {
            slicePixelRaster = new int[length * height];
            lightIntensity = new float[width * height];
            pixelIntensity = new float[length * height];
            for (int i = 0; i < slicePixelRaster.length; i++) {
                pixelIntensity[i] = 255;
                lightIntensity[i] = 255;
            }
            // Alpha compositing.
            for (int x = 0; x < width; x++) {
                for (int z = 0; z < length; z++) {
                    for (int y = 0; y < height; y++) {
                        pixelIntensity[y * length + z] =
                                pixelIntensity[y * length + z]
                                * (1 - slices[z][c][y * width + x])
                                + (slices[z][c][y * width + x]
                                * (1 - slices[z][c][y * width + x]) * 255);
                        illumination = Math.max(0, -1
                                * pointNormals[c][x][y][z][0]);
                        lightIntensity[y * width + z] =
                                lightIntensity[y * width + z]
                                * (1 - slices[z][c][y * width + x])
                                + (slices[z][c][y * width + x])
                                * illumination * 255;
                    }
                }
            }
            // Pixel coloring.
            for (int z = 0; z < length; z++) {
                for (int y = 0; y < height; y++) {
                    red = (int) (pixelIntensity[y * width + z] * 0.67f
                            + 0.33f * lightIntensity[y * width + z]);
                    green = (int) (pixelIntensity[y * width + z] * 0.67f);
                    blue = (int) (pixelIntensity[y * width + z] * 0.67f);
                    rgba = (0xff000000 | red << 16 | green << 8 | blue);
                    slicePixelRaster[y * length + z] = rgba;
                }
            }
            currentOutFile = new File(outSubDir, "yzPosDir" + "class" + c
                    + ".jpg");
            visBufferedImage = new BufferedImage(length, height,
                    BufferedImage.TYPE_INT_RGB);
            visImage = Toolkit.getDefaultToolkit().createImage(
                    new MemoryImageSource(width, height, slicePixelRaster, 0,
                    width));
            visBufferedImage.getGraphics().drawImage(visImage, 0, 0, null);
            try {
                ImageIO.write(visBufferedImage, "JPG", currentOutFile);
            } catch (Exception e) {
                System.err.println(e.getMessage());
            }
            // The other direction.
            slicePixelRaster = new int[length * height];
            pixelIntensity = new float[length * height];
            lightIntensity = new float[width * height];
            for (int i = 0; i < slicePixelRaster.length; i++) {
                pixelIntensity[i] = 255;
                lightIntensity[i] = 255;
            }
            // Alpha compositing.
            for (int x = 0; x < width; x++) {
                for (int z = 0; z < length; z++) {
                    for (int y = 0; y < height; y++) {
                        pixelIntensity[y * length + z] =
                                pixelIntensity[y * length + z]
                                * (1 - slices[z][c][y * width + (499 - x)])
                                + (slices[z][c][y * width + (499 - x)]
                                * (1 - slices[z][c][y * width + (499 - x)])
                                * 255);
                        illumination = Math.max(0, 1
                                * pointNormals[c][(499 - x)][y][z][0]);
                        lightIntensity[y * width + z] =
                                lightIntensity[y * width + z]
                                * (1 - slices[z][c][y * width + (499 - x)])
                                + (slices[z][c][y * width + (499 - x)])
                                * illumination * 255;
                    }
                }
            }
            // Pixel coloring.
            for (int z = 0; z < length; z++) {
                for (int y = 0; y < height; y++) {
                    red = (int) (pixelIntensity[y * width + z] * 0.67f
                            + 0.33f * lightIntensity[y * width + z]);
                    green = (int) (pixelIntensity[y * width + z] * 0.67f);
                    blue = (int) (pixelIntensity[y * width + z] * 0.67f);
                    rgba = (0xff000000 | red << 16 | green << 8 | blue);
                    slicePixelRaster[y * length + z] = rgba;
                }
            }
            currentOutFile = new File(outSubDir, "yzNegDir" + "class" + c
                    + ".jpg");
            visBufferedImage = new BufferedImage(length, height,
                    BufferedImage.TYPE_INT_RGB);
            visImage = Toolkit.getDefaultToolkit().createImage(
                    new MemoryImageSource(width, height, slicePixelRaster, 0,
                    width));
            visBufferedImage.getGraphics().drawImage(visImage, 0, 0, null);
            try {
                ImageIO.write(visBufferedImage, "JPG", currentOutFile);
            } catch (Exception e) {
                System.err.println(e.getMessage());
            }
            System.out.println("One direction finished");
        }
        // Here we take slices by y.
        for (int c = 0; c < numClasses; c++) {
            slicePixelRaster = new int[width * length];
            pixelIntensity = new float[width * length];
            lightIntensity = new float[width * height];
            for (int i = 0; i < slicePixelRaster.length; i++) {
                pixelIntensity[i] = 255;
                lightIntensity[i] = 255;
            }
            // Alpha compositing.
            for (int y = 0; y < 500; y++) {
                for (int x = 0; x < width; x++) {
                    for (int z = 0; z < length; z++) {
                        pixelIntensity[z * width + x] =
                                pixelIntensity[z * width + x]
                                * (1 - slices[z][c][y * width + x])
                                + (slices[z][c][y * width + x]
                                * (1 - slices[z][c][y * width + x]) * 255);
                        illumination = Math.max(0, -1
                                * pointNormals[c][x][y][z][1]);
                        lightIntensity[z * width + x] =
                                lightIntensity[z * width + x]
                                * (1 - slices[z][c][y * width + x])
                                + (slices[z][c][y * width + x]) * illumination
                                * 255;
                    }
                }
            }
            // Pixel coloring.
            for (int x = 0; x < width; x++) {
                for (int z = 0; z < length; z++) {
                    red = (int) (pixelIntensity[z * width + x] * 0.67f
                            + 0.33f * lightIntensity[z * width + x]);
                    green = (int) (pixelIntensity[z * width + x] * 0.67f);
                    blue = (int) (pixelIntensity[z * width + x] * 0.67f);
                    rgba = (0xff000000 | red << 16 | green << 8 | blue);
                    slicePixelRaster[z * width + x] = rgba;
                }
            }
            currentOutFile = new File(outSubDir, "zxPosDir" + "class" + c
                    + ".jpg");
            visBufferedImage = new BufferedImage(width, length,
                    BufferedImage.TYPE_INT_RGB);
            visImage = Toolkit.getDefaultToolkit().createImage(
                    new MemoryImageSource(width, height, slicePixelRaster, 0,
                    width));
            visBufferedImage.getGraphics().drawImage(visImage, 0, 0, null);
            try {
                ImageIO.write(visBufferedImage, "JPG", currentOutFile);
            } catch (Exception e) {
                System.err.println(e.getMessage());
            }
            // The other direction.
            slicePixelRaster = new int[width * length];
            pixelIntensity = new float[width * length];
            lightIntensity = new float[width * height];
            for (int i = 0; i < slicePixelRaster.length; i++) {
                pixelIntensity[i] = 255;
                lightIntensity[i] = 255;
            }
            // Alpha compositing.
            for (int y = 0; y < 500; y++) {
                for (int x = 0; x < width; x++) {
                    for (int z = 0; z < height; z++) {
                        pixelIntensity[z * width + x] =
                                pixelIntensity[z * width + x]
                                * (1 - slices[z][c][(499 - y) * width + x])
                                + (slices[z][c][(499 - y) * width + x]
                                * (1 - slices[z][c][(499 - y) * width + x])
                                * 255);
                        illumination = Math.max(0, 1
                                * pointNormals[c][x][(499 - y)][z][1]);
                        lightIntensity[z * width + x] =
                                lightIntensity[z * width + x]
                                * (1 - slices[z][c][(499 - y) * width + x])
                                + (slices[z][c][(499 - y) * width + x])
                                * illumination * 255;
                    }
                }
            }
            for (int x = 0; x < width; x++) {
                for (int z = 0; z < height; z++) {
                    red = (int) (pixelIntensity[z * width + x] * 0.67f
                            + 0.33f * lightIntensity[z * width + x]);
                    green = (int) (pixelIntensity[z * width + x] * 0.67f);
                    blue = (int) (pixelIntensity[z * width + x] * 0.67f);
                    rgba = (0xff000000 | red << 16 | green << 8 | blue);
                    slicePixelRaster[z * width + x] = rgba;
                }
            }
            currentOutFile = new File(outSubDir, "zxNegDir" + "class" + c
                    + ".jpg");
            visBufferedImage = new BufferedImage(width, length,
                    BufferedImage.TYPE_INT_RGB);
            visImage = Toolkit.getDefaultToolkit().createImage(
                    new MemoryImageSource(width, height, slicePixelRaster, 0,
                    width));
            visBufferedImage.getGraphics().drawImage(visImage, 0, 0, null);
            try {
                ImageIO.write(visBufferedImage, "JPG", currentOutFile);
            } catch (Exception e) {
                System.err.println(e.getMessage());
            }
            System.out.println("One direction finished");
        }
        System.out.println("Finished visualizing for " + algName + " ,k = "
                + k);
    }

    /**
     * This runnable class is used to spawn workers to perform the visualization
     * in multiple threads, as it is computationally intensive.
     */
    class VisualizerThread implements Runnable {

        private String algName;
        private int k;
        private Neighbor3DColorVisualizer n3D;

        public VisualizerThread(Neighbor3DColorVisualizer n3D, String algName,
                int k) {
            this.algName = algName;
            this.k = k;
            this.n3D = n3D;
        }

        @Override
        public void run() {
            try {
                System.out.println("starting work on algorithm " + algName
                        + " for k = " + k);
                n3D.generateImagesForAlgorithm(algName, k);
            } catch (Exception e) {
                System.err.println(e.getMessage());
            }
        }
    }

    /**
     * Invoke the 3D probability density landscape visualization for all the
     * tested algorithms in several 3-thread batches.
     *
     * @throws Exception
     */
    public void allVisualizationsMultiThread() throws Exception {
        Thread workerFirst = new Thread(new VisualizerThread(
                new Neighbor3DColorVisualizer(dset, outDir), "KNN", 1));
        Thread workerSecond = new Thread(new VisualizerThread(
                new Neighbor3DColorVisualizer(dset, outDir), "HIKNN", 1));
        Thread workerThird = new Thread(new VisualizerThread(
                new Neighbor3DColorVisualizer(dset, outDir),
                "HIKNNnonWeighted", 1));
        workerFirst.start();
        workerSecond.start();
        workerThird.start();
        try {
            workerFirst.join();
            workerSecond.join();
            workerThird.join();
        } catch (Throwable t) {
        }
        workerFirst = new Thread(new VisualizerThread(
                new Neighbor3DColorVisualizer(dset, outDir), "hFNN", 1));
        workerSecond = new Thread(new VisualizerThread(
                new Neighbor3DColorVisualizer(dset, outDir), "hwKNN", 1));
        workerThird = new Thread(new VisualizerThread(
                new Neighbor3DColorVisualizer(dset, outDir), "NHBNN", 1));
        workerFirst.start();
        workerSecond.start();
        workerThird.start();
        try {
            workerFirst.join();
            workerSecond.join();
            workerThird.join();
        } catch (Throwable t) {
        }
        workerFirst = new Thread(new VisualizerThread(
                new Neighbor3DColorVisualizer(dset, outDir), "KNN", 5));
        workerSecond = new Thread(new VisualizerThread(
                new Neighbor3DColorVisualizer(dset, outDir), "HIKNN", 5));
        workerThird = new Thread(new VisualizerThread(
                new Neighbor3DColorVisualizer(dset, outDir),
                "HIKNNnonWeighted", 1));
        workerFirst.start();
        workerSecond.start();
        workerThird.start();
        try {
            workerFirst.join();
            workerSecond.join();
            workerThird.join();
        } catch (Throwable t) {
        }
        workerFirst = new Thread(new VisualizerThread(
                new Neighbor3DColorVisualizer(dset, outDir), "hFNN", 5));
        workerSecond = new Thread(new VisualizerThread(
                new Neighbor3DColorVisualizer(dset, outDir), "hwKNN", 5));
        workerThird = new Thread(new VisualizerThread(
                new Neighbor3DColorVisualizer(dset, outDir), "NHBNN", 5));
        workerFirst.start();
        workerSecond.start();
        workerThird.start();
        try {
            workerFirst.join();
            workerSecond.join();
            workerThird.join();
        } catch (Throwable t) {
        }
        workerFirst = new Thread(new VisualizerThread(
                new Neighbor3DColorVisualizer(dset, outDir), "KNN", 10));
        workerSecond = new Thread(new VisualizerThread(
                new Neighbor3DColorVisualizer(dset, outDir), "HIKNN", 10));
        workerThird = new Thread(new VisualizerThread(
                new Neighbor3DColorVisualizer(dset, outDir),
                "HIKNNnonWeighted", 10));
        workerFirst.start();
        workerSecond.start();
        workerThird.start();
        try {
            workerFirst.join();
            workerSecond.join();
            workerThird.join();
        } catch (Throwable t) {
        }
        workerFirst = new Thread(new VisualizerThread(
                new Neighbor3DColorVisualizer(dset, outDir), "hFNN", 10));
        workerSecond = new Thread(new VisualizerThread(
                new Neighbor3DColorVisualizer(dset, outDir), "hwKNN", 10));
        workerThird = new Thread(new VisualizerThread(
                new Neighbor3DColorVisualizer(dset, outDir), "NHBNN", 10));
        workerFirst.start();
        workerSecond.start();
        workerThird.start();
        try {
            workerFirst.join();
            workerSecond.join();
            workerThird.join();
        } catch (Throwable t) {
        }
    }

    /**
     * Command line parameter specification.
     */
    public static void info() {
        System.out.println("arg0: Input data file (arff or csv).");
        System.out.println("arg1: Output directory.");
    }

    /**
     * A script for performing 3D classification landscape visualization for
     * several kNN methods on real-world data (that was scaled down to 3D
     * previously by some dimensionality reduction method like MDS).
     *
     * @param args Command line parameters, as specified.
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            info();
            return;
        }
        DataSet dset = SupervisedLoader.loadData(args[0], false);
        File outDir = new File(args[1]);
        if (!outDir.exists()) {
            FileUtil.createDirectory(outDir);
        }
        dset.normalizeFloats();
        float[][] dMat = dset.calculateDistMatrixMultThr(
                CombinedMetric.FLOAT_EUCLIDEAN, 8);
        NeighborSetFinder nsf = new NeighborSetFinder(dset, dMat,
                CombinedMetric.FLOAT_EUCLIDEAN);
        nsf.calculateNeighborSetsMultiThr(10, 8);

        Neighbor3DColorVisualizer n3D = new Neighbor3DColorVisualizer(dset,
                outDir);
        // Can also call n3D.doAllTheWork() here, but only for small datasets,
        // as it eats up lots of memory in the current implementation.
        n3D.setDistances(dMat);
        n3D.setNSF(nsf);
        n3D.generateImagesForAlgorithm("KNN", 10);
        System.gc();
        n3D.generateImagesForAlgorithm("HIKNN", 10);
        System.gc();
        n3D.generateImagesForAlgorithm("HIKNNnonWeighted", 10);
        System.gc();
        n3D.generateImagesForAlgorithm("hFNN", 10);
        System.gc();
        n3D.generateImagesForAlgorithm("hwKNN", 10);
        System.gc();
        n3D.generateImagesForAlgorithm("NHBNN", 10);
        System.gc();
        NeighborSetFinder nsfTemp = nsf.getSubNSF(5);
        n3D.setNSF(nsfTemp);
        n3D.generateImagesForAlgorithm("KNN", 5);
        System.gc();
        n3D.generateImagesForAlgorithm("HIKNN", 5);
        System.gc();
        n3D.generateImagesForAlgorithm("HIKNNnonWeighted", 5);
        System.gc();
        n3D.generateImagesForAlgorithm("hFNN", 5);
        System.gc();
        n3D.generateImagesForAlgorithm("hwKNN", 5);
        System.gc();
        n3D.generateImagesForAlgorithm("NHBNN", 5);
        System.gc();
        nsfTemp = nsf.getSubNSF(1);
        n3D.setNSF(nsfTemp);
        n3D.generateImagesForAlgorithm("KNN", 1);
        System.gc();
        n3D.generateImagesForAlgorithm("HIKNN", 1);
        System.gc();
        n3D.generateImagesForAlgorithm("HIKNNnonWeighted", 1);
        System.gc();
        n3D.generateImagesForAlgorithm("hFNN", 1);
        System.gc();
        n3D.generateImagesForAlgorithm("hwKNN", 1);
        System.gc();
        n3D.generateImagesForAlgorithm("NHBNN", 1);
        System.gc();
    }
}
