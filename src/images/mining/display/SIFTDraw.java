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
package images.mining.display;

import data.representation.DataInstance;
import data.representation.images.sift.SIFTRepresentation;
import data.representation.images.sift.SIFTVector;
import data.representation.images.sift.util.ClusteredSIFTRepresentation;
import draw.basic.BoxBlur;
import draw.basic.RotatedEllipse;
import ioformat.IOARFF;
import ioformat.images.SiftUtil;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
import java.util.Random;
import javax.imageio.ImageIO;
import learning.unsupervised.Cluster;
import statistics.Variance2D;
import util.ImageUtil;

/**
 * This class enables a visualization of SIFT feature distributions on top of an
 * image.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class SIFTDraw {

    // Clusters of visual words correspond to object in an image.
    private Cluster[] visualObjectClusters = null;
    // The basis image.
    private BufferedImage image = null;

    /**
     *
     * @param visualObjectClusters Clusters of features in the image.
     * @param imagePath String that is the path to load the image from.
     */
    public SIFTDraw(Cluster[] visualObjectClusters, String imagePath) {
        this.visualObjectClusters = visualObjectClusters;
        try {
            image = ImageIO.read(new File(imagePath));
        } catch (Exception e) {
            System.err.println(e.getMessage());
        }
    }

    /**
     *
     * @param visualObjectClusters Clusters of features in the image.
     * @param image BufferedImage object that is the basis image.
     */
    public SIFTDraw(Cluster[] visualObjectClusters, BufferedImage image) {
        this.visualObjectClusters = visualObjectClusters;
        this.image = image;
    }

    /**
     * This method draws the ellipses that correspond to SIFT clusters on top of
     * an image.
     *
     * @param ellipses Array of RotatedEllipse objects corresponding to SIFT
     * clusters.
     * @param oldImage Image that the ellipses will be drawn on top of.
     * @param useGradientDraw Whether to use gradients when drawing the
     * ellipses.
     * @return BufferedImage object that is the image with ellipses drawn on
     * top.
     * @throws Exception
     */
    public static BufferedImage drawClusterEllipsesOnImage(
            RotatedEllipse[] ellipses, BufferedImage oldImage,
            boolean useGradientDraw) throws Exception {
        if (oldImage == null) {
            return null;
        }
        if (ellipses == null || ellipses.length == 0) {
            // For consistency, even if no ellipses are to be drawn, we create
            // a new image object.
            return ImageUtil.copyImage(oldImage);
        }
        BufferedImage newImage = ImageUtil.copyImage(oldImage);
        Color[] ellipseColors = new Color[ellipses.length];
        Graphics2D graphics = newImage.createGraphics();
        Random randa = new Random();
        // Assign random colors to the ellipses.
        for (int i = 0; i < ellipseColors.length; i++) {
            if (useGradientDraw) {
                ellipseColors[i] = new Color(randa.nextFloat(),
                        randa.nextFloat(), randa.nextFloat(), 0.75f);
            } else {
                ellipseColors[i] = new Color(randa.nextFloat(),
                        randa.nextFloat(), randa.nextFloat(), 0.5f);
            }
        }
        for (int i = 0; i < ellipses.length; i++) {
            ellipses[i].setColor(ellipseColors[i]);
            if (!useGradientDraw) {
                ellipses[i].drawOnGraphics(graphics);
            } else {
                ellipses[i].drawWithGradient(graphics);
            }
        }
        return newImage;
    }

    /**
     * This method draws the ellipses that correspond to SIFT clusters on top of
     * an image.
     *
     * @param features ClusteredSIFTRepresentation object representing clusters
     * of SIFT features.
     * @param image Image that the ellipses will be drawn on top of.
     * @param outImagePath String that is the path where the new image will be
     * persisted.
     * @param useGradientDraw Whether to use gradients when drawing the
     * ellipses.
     * @throws Exception
     */
    public static void drawClustersOnImageAsEllipses(
            ClusteredSIFTRepresentation features, BufferedImage image,
            String outImagePath, boolean useGradientDraw) throws Exception {
        if (image == null) {
            return;
        }
        if (features == null || features.isEmpty()) {
            return;
        }
        // Get the cluster configuration.
        Cluster[] clusters = features.representAsClusters();
        Color[] clusterColors = new Color[clusters.length];
        Graphics2D graphics = image.createGraphics();
        // Assign random colors to the clusters for display.
        Random randa = new Random();
        for (int i = 0; i < clusterColors.length; i++) {
            if (useGradientDraw) {
                clusterColors[i] = new Color(randa.nextFloat(),
                        randa.nextFloat(), randa.nextFloat(), 0.75f);
            } else {
                clusterColors[i] = new Color(randa.nextFloat(),
                        randa.nextFloat(), randa.nextFloat(), 0.5f);
            }
        }
        // Find the variance vectors.
        Variance2D var = new Variance2D();
        RotatedEllipse[] ellipses = var.
                findVarianceEllipseForSIFTCLusterConfiguration(clusters);
        for (int i = 0; i < clusters.length; i++) {
            ellipses[i].setColor(clusterColors[i]);
            if (!useGradientDraw) {
                ellipses[i].drawOnGraphics(graphics);
            } else {
                ellipses[i].drawWithGradient(graphics);
            }
        }
        // Persist the resulting image.
        File outImageFile = new File(outImagePath);
        ImageIO.write(image, "jpg", outImageFile);
    }

    /**
     * This method draws the ellipses that correspond to SIFT clusters on top of
     * an image.
     *
     * @param arffPath String that is the path to the .arff file containing the
     * ClusteredSIFTRepresentation that describes SIFT clusters on an image.
     * @param image Image that the ellipses will be drawn on top of.
     * @param outImagePath String that is the path where the new image will be
     * persisted.
     * @param useGradientDraw Whether to use gradients when drawing the
     * ellipses.
     * @throws Exception
     */
    public static void drawClustersOnImageAsEllipses(
            String arffPath, BufferedImage image,
            String outImagePath, boolean useGradientDraw) throws Exception {
        IOARFF arff = new IOARFF();
        ClusteredSIFTRepresentation features = new ClusteredSIFTRepresentation(
                new SIFTRepresentation(arff.load(arffPath)));
        drawClustersOnImageAsEllipses(features, image, outImagePath,
                useGradientDraw);
    }

    /**
     * This method draws the ellipses that correspond to SIFT clusters on top of
     * an image.
     *
     * @param arffPath String that is the path to the .arff file containing the
     * ClusteredSIFTRepresentation that describes SIFT clusters on an image.
     * @param inImagePath String that is the path where the original image is
     * stored.
     * @param outImagePath String that is the path where the new image will be
     * persisted.
     * @param useGradientDraw Whether to use gradients when drawing the
     * ellipses.
     * @throws Exception
     */
    public static void drawClustersOnImageAsEllipses(String arffPath,
            String inImagePath,
            String outImagePath, boolean useGradientDraw) throws Exception {
        BufferedImage image = ImageIO.read(new File(inImagePath));
        drawClustersOnImageAsEllipses(arffPath, image, outImagePath,
                useGradientDraw);
    }

    /**
     * Draws arrows corresponding to SIFT features on an image and colors them
     * according to their clusters. The arrow length corresponds to the scale at
     * which the particular SIFT feature was found.
     *
     * @param arffPath String that is the path to the .arff file containing the
     * ClusteredSIFTRepresentation that describes SIFT clusters on an image.
     * @param inImagePath String that is the path where the original image is
     * stored.
     * @param outImagePath String that is the path where the new image will be
     * persisted.
     * @throws Exception
     */
    public static void drawClusteredSIFTImage(String arffPath,
            String inImagePath, String outImagePath) throws Exception {
        IOARFF persister = new IOARFF();
        ClusteredSIFTRepresentation features = new ClusteredSIFTRepresentation(
                new SIFTRepresentation(persister.load(arffPath)));
        File outImageFile = new File(outImagePath);
        BufferedImage image = null;
        try {
            image = ImageIO.read(new File(inImagePath));
        } catch (Exception e) {
            throw new Exception("Error in reading image file " + inImagePath
                    + " " + e.getMessage());
        }
        Graphics2D graphics = image.createGraphics();
        int maxClusterIndex = -1;
        for (int i = 0; i < features.size(); i++) {
            int tmpClusterIndex = (features.data.get(i)).iAttr[0];
            if (tmpClusterIndex > maxClusterIndex) {
                maxClusterIndex = tmpClusterIndex;
            }
        }
        Color[] clusterColors = new Color[maxClusterIndex + 1];
        // Assign random colors.
        Random randa = new Random();
        for (int i = 0; i < clusterColors.length; i++) {
            clusterColors[i] = new Color(randa.nextFloat(), randa.nextFloat(),
                    randa.nextFloat());
        }
        for (int i = 0; i < features.data.size(); i++) {
            SIFTVector v = new SIFTVector(features.data.get(i));
            graphics.setColor(clusterColors[(features.data.get(i)).iAttr[0]]);
            transformAndDrawLine(graphics, v, 0.f, 0.f, 1.f, 0.f);
            transformAndDrawLine(graphics, v, 0.85f, 0.1f, 1.f, 0.f);
            transformAndDrawLine(graphics, v, 0.85f, -0.1f, 1.f, 0.f);
        }
        ImageIO.write(image, "jpg", outImageFile);
    }

    /**
     * Draws arrows corresponding to SIFT features on an image and colors them
     * according to their clusters. The arrow length corresponds to the scale at
     * which the particular SIFT feature was found.
     *
     * @param siftClusters
     * @param oldImage Image that the arrows will be drawn on top of.
     * @return BufferedImage object that is the original image with the arrows
     * corresponding to SIFT feature drawn on top, colored according to their
     * respective clusters.
     * @throws Exception
     */
    public static BufferedImage drawClusteredSIFTImage(Cluster[] siftClusters,
            BufferedImage oldImage) throws Exception {
        if (siftClusters == null || siftClusters.length == 0) {
            if (oldImage == null) {
                return null;
            } else {
                return ImageUtil.copyImage(oldImage);
            }
        }
        BufferedImage newImage = ImageUtil.copyImage(oldImage);
        Graphics2D graphics = newImage.createGraphics();
        // Assign random colors.
        Color[] clusterColors = new Color[siftClusters.length];
        Random randa = new Random();
        for (int i = 0; i < clusterColors.length; i++) {
            clusterColors[i] = new Color(randa.nextFloat(), randa.nextFloat(),
                    randa.nextFloat());
        }
        for (int i = 0; i < siftClusters.length; i++) {
            if (siftClusters[i] != null) {
                for (int j = 0; j < siftClusters[i].size(); j++) {
                    SIFTVector v = new SIFTVector(
                            siftClusters[i].getInstance(j));
                    graphics.setColor(clusterColors[i]);
                    transformAndDrawLine(graphics, v, 0.f, 0.f, 1.f, 0.f);
                    transformAndDrawLine(graphics, v, 0.85f, 0.1f, 1.f, 0.f);
                    transformAndDrawLine(graphics, v, 0.85f, -0.1f, 1.f, 0.f);
                }
            }
        }
        return newImage;
    }

    /**
     * Draws arrows corresponding to SIFT features on an image and colors them
     * according to their clusters. The arrow length corresponds to the scale at
     * which the particular SIFT feature was found.
     *
     * @param arffPath String that is the path to the .arff file containing the
     * ClusteredSIFTRepresentation that describes SIFT clusters on an image.
     * @param image Image that the arrows will be drawn on top of.
     * @param outImagePath String that is the path where the new image will be
     * persisted.
     * @throws Exception
     */
    public static void drawClusteredSIFTImage(
            String arffPath, BufferedImage image, String outImagePath)
            throws Exception {
        IOARFF persister = new IOARFF();
        ClusteredSIFTRepresentation features = new ClusteredSIFTRepresentation(
                new SIFTRepresentation(persister.load(arffPath)));
        File outImageFile = new File(outImagePath);
        Graphics2D graphics = image.createGraphics();
        int maxClusterIndex = -1;
        for (int i = 0; i < features.size(); i++) {
            int tmpClusterIndex = (features.data.get(i)).iAttr[0];
            if (tmpClusterIndex > maxClusterIndex) {
                maxClusterIndex = tmpClusterIndex;
            }
        }
        // Assign random colors.
        Color[] clusterColors = new Color[maxClusterIndex + 1];
        Random randa = new Random();
        for (int i = 0; i < clusterColors.length; i++) {
            clusterColors[i] = new Color(randa.nextFloat(), randa.nextFloat(),
                    randa.nextFloat());
        }
        for (int i = 0; i < features.size(); i++) {
            SIFTVector v = new SIFTVector(features.data.get(i));
            graphics.setColor(clusterColors[(features.data.get(i)).iAttr[0]]);
            transformAndDrawLine(graphics, v, 0.f, 0.f, 1.f, 0.f);
            transformAndDrawLine(graphics, v, 0.85f, 0.1f, 1.f, 0.f);
            transformAndDrawLine(graphics, v, 0.85f, -0.1f, 1.f, 0.f);
        }
        ImageIO.write(image, "jpg", outImageFile);
    }

    /**
     * Draws arrows corresponding to SIFT features on an image. The arrow length
     * corresponds to the scale at which the particular SIFT feature was found.
     *
     * @param arffPath String that is the path to the .arff file containing the
     * corresponding SIFTRepresentation.
     * @param inImagePath String that is the path to the original image file.
     * @param outImagePath String that is the path to where the new image is to
     * be saved.
     * @throws Exception
     */
    public static void drawSIFTImage(String arffPath, String inImagePath,
            String outImagePath) throws Exception {
        SIFTRepresentation features = SiftUtil.importFeaturesFromArff(arffPath);
        File outImageFile = new File(outImagePath);
        BufferedImage image = null;
        try {
            image = ImageIO.read(new File(inImagePath));
        } catch (Exception e) {
            throw new Exception("Error in reading image file " + inImagePath
                    + " " + e.getMessage());
        }
        Graphics2D graphics = image.createGraphics();
        for (int i = 0; i < features.size(); i++) {
            SIFTVector v = new SIFTVector((DataInstance) (
                    features.data.get(i)));
            transformAndDrawLine(graphics, v, 0.f, 0.f, 1.f, 0.f);
            transformAndDrawLine(graphics, v, 0.85f, 0.1f, 1.f, 0.f);
            transformAndDrawLine(graphics, v, 0.85f, -0.1f, 1.f, 0.f);
        }
        ImageIO.write(image, "jpg", outImageFile);
    }

    /**
     * Draws arrows corresponding to SIFT features on an image. The arrow length
     * corresponds to the scale at which the particular SIFT feature was found.
     *
     * @param features SIFTRepresentation object holding the features to be
     * drawn.
     * @param inImage BufferedImage as the basis for the drawing. This object is
     * not modified by the method, as the copy is made prior to the drawing.
     * @return BufferedImage that the features have been drawn on.
     * @throws Exception
     */
    public static BufferedImage drawSIFTImage(SIFTRepresentation features,
            BufferedImage inImage) throws Exception {
        if (inImage == null) {
            return null;
        }
        BufferedImage outImage = ImageUtil.copyImage(inImage);
        Graphics2D graphics = outImage.createGraphics();
        for (int i = 0; i < features.data.size(); i++) {
            SIFTVector v = new SIFTVector(features.data.get(i));
            transformAndDrawLine(graphics, v, 0.f, 0.f, 1.f, 0.f);
            transformAndDrawLine(graphics, v, 0.85f, 0.1f, 1.f, 0.f);
            transformAndDrawLine(graphics, v, 0.85f, -0.1f, 1.f, 0.f);
        }
        return outImage;
    }

    /**
     * Draw a red-green overlay on top of an image where the landscape
     * corresponds to the utility of visual words occurring in certain image
     * regions.
     *
     * @param features SIFTRepresentation that holds the features of the image.
     * @param featureGoodness Array of float values between zero and one.
     * @param inImage BufferedImage holding the image that is to be processed.
     * @return BufferedImage object that is the new image.
     * @throws Exception
     */
    public static BufferedImage drawSIFTGoodnessOnImage(
            SIFTRepresentation features, float[] featureGoodness,
            BufferedImage inImage) throws Exception {
        BufferedImage outImage = ImageUtil.copyImage(inImage);
        Graphics2D graphics = outImage.createGraphics();
        // The image is divided into buckets and the goodness of each bucket
        // is determined.
        int val;
        int width = inImage.getWidth();
        int height = inImage.getHeight();
        // Step is the bucket width and height.
        int step = 40;
        int steppedWidth = width / step;
        if (width % step != 0) {
            steppedWidth++;
        }
        int steppedHeight = height / step;
        if (height % step != 0) {
            steppedHeight++;
        }
        int bucketX, bucketY;
        // Buckets hold a list of indexes of the image features.
        ArrayList<Integer>[][] bucketedData =
                new ArrayList[steppedWidth][steppedHeight];
        // Bucket initialization.
        for (int i = 0; i < steppedWidth; i++) {
            for (int j = 0; j < steppedHeight; j++) {
                bucketedData[i][j] = new ArrayList<>(5);
            }
        }
        // Fill the buckets.
        for (int i = 0; i < features.size(); i++) {
            bucketX = (int) (((SIFTVector) (
                    features.data.get(i))).getX() / step);
            bucketY = (int) (((SIFTVector) (
                    features.data.get(i))).getY() / step);
            bucketedData[bucketX][bucketY].add(i);
        }
        int[] goodnessByteArr = new int[width * height];
        double weight;
        double sigma = 0.05;
        // X and Y coordinates of individual pixels.
        int pX;
        int pY;
        int fIndex;
        double gh;
        double bh;
        for (int i = 0; i < goodnessByteArr.length; i++) {
            pX = i % width;
            pY = i / width;
            bucketX = (pX) / step;
            bucketY = (pY) / step;
            gh = 0;
            bh = 0;
            for (int j = 0; j < bucketedData[bucketX][bucketY].size(); j++) {
                // Index of the feature.
                fIndex = bucketedData[bucketX][bucketY].get(j);
                // Weight that measures the influence of that feature from the
                // bucket on the currently considered pixel.
                weight = Math.min(Math.exp(-sigma
                        * ((((SIFTVector) (
                        features.data.get(fIndex))).getX() - pX)
                        * (((SIFTVector) (features.data.get(fIndex))).getX()
                        - pX) + (((SIFTVector) (
                        features.data.get(fIndex))).getY()
                        - pY) * (((SIFTVector) (features.data.get(fIndex)))
                        .getY() - pY))), 1);
                // Update the good and bad totals.
                gh += weight * featureGoodness[fIndex];
                bh += weight * (1 - featureGoodness[fIndex]);
            }
            if (gh > 0 || bh > 0) {
                val = (int) (255 * ((gh) / (gh + bh)));
                goodnessByteArr[i] = val << 8 | (255 - val) << 16 | 180 << 24;
            } else {
                goodnessByteArr[i] = 0x50809080;
            }
        }
        BufferedImage overlay = new BufferedImage(width, height,
                BufferedImage.TYPE_INT_ARGB);
        // Now perform smoothing by applying several passes of box blur.
        BoxBlur bb = new BoxBlur(8);
        bb.blurPixelsWithAlpha(goodnessByteArr, new int[goodnessByteArr.length],
                new Dimension(width, height));
        bb = new BoxBlur(9);
        bb.blurPixelsWithAlpha(goodnessByteArr,
                new int[goodnessByteArr.length], new Dimension(width, height));
        bb = new BoxBlur(7);
        bb.blurPixelsWithAlpha(goodnessByteArr, new int[goodnessByteArr.length],
                new Dimension(width, height));
        overlay.setRGB(0, 0, width, height, goodnessByteArr, 0, width);
        // Now draw the feature utility landscape over the original image.
        graphics.drawImage(overlay, 0, 0, null);
        // In the end, also draw individual features by applying the same color
        // scheme as for the regions.
        for (int i = 0; i < features.data.size(); i++) {
            SIFTVector v = new SIFTVector(features.data.get(i));
            val = (int) (255 * (Math.min(1, featureGoodness[i])));
            Color col = new Color(val << 8 | (255 - val) << 16);
            graphics.setColor(col);
            transformAndDrawLine(graphics, v, 0.f, 0.f, 1.f, 0.f);
            transformAndDrawLine(graphics, v, 0.85f, 0.1f, 1.f, 0.f);
            transformAndDrawLine(graphics, v, 0.85f, -0.1f, 1.f, 0.f);
        }
        return outImage;
    }

    /**
     * Draw clustered SIFT features on an image and save the result.
     *
     * @param outPath String that is the file path denoting where to save the
     * resulting image.
     * @throws Exception
     */
    public void drawClusters(String outPath) throws Exception {
        File outImageFile = new File(outPath);
        BufferedImage imageCopy = ImageUtil.copyImage(image);
        Graphics2D graphics = imageCopy.createGraphics();
        Random randa = new Random();
        Color col;
        for (int i = 0; i < visualObjectClusters.length; i++) {
            col = new Color(randa.nextFloat(), randa.nextFloat(),
                    randa.nextFloat());
            graphics.setColor(col);
            for (int j = 0; j < visualObjectClusters[i].size(); j++) {
                SIFTVector v = new SIFTVector(
                        visualObjectClusters[i].getInstance(j));
                transformAndDrawLine(graphics, v, 0.f, 0.f, 1.f, 0.f);
                transformAndDrawLine(graphics, v, 0.85f, 0.1f, 1.f, 0.f);
                transformAndDrawLine(graphics, v, 0.85f, -0.1f, 1.f, 0.f);
            }
        }
        ImageIO.write(imageCopy, "jpg", outImageFile);
    }

    /**
     * Draws a line at a location specified by the X,Y coordinates in the
     * feature vector.
     *
     * @param graphics Graphics2D object to draw the line on.
     * @param v SIFTVector feature object.
     * @param xFirst X coordinate of the first point.
     * @param yFirst Y coordinate of the first point.
     * @param xSecond X coordinate of the second point.
     * @param ySecond Y coordinate of the second point.
     */
    private static void transformAndDrawLine(Graphics2D graphics, SIFTVector v,
            float xFirst, float yFirst, float xSecond, float ySecond) {
        float x = v.getX();
        float y = v.getY();
        float scale = v.getScale();
        float angle = v.getAngle();
        float len = 6f * scale;
        float s = (float) Math.sin(angle);
        float c = (float) Math.cos(angle);
        int r1 = (int) (x + len * (c * xFirst - s * yFirst));
        int c1 = (int) (y + len * (s * xFirst + c * yFirst));
        int r2 = (int) (x + len * (c * xSecond - s * ySecond));
        int c2 = (int) (y + len * (s * xSecond + c * ySecond));
        graphics.drawLine(r1, c1, r2, c2);
    }

    /**
     * Prints out the usage information.
     */
    public static void info() {
        System.out.println("arg0: Input arff of clustered SIFT features.");
        System.out.println("arg1: Input image.");
        System.out.println("arg2: Output image.");
    }

    /**
     * Reads in an arff file of clustered SIFT features and draws the
     * corresponding cluster ellipses on an image and saves the resulting image.
     *
     * @param args Command line arguments, as specified in the info() method.
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        if (args.length != 3) {
            info();
            return;
        }
        drawClustersOnImageAsEllipses(args[0], args[1], args[2], true);
    }
}