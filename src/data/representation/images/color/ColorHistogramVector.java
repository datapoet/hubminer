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
package data.representation.images.color;

import data.representation.DataInstance;
import java.awt.image.BufferedImage;
import java.awt.image.PixelGrabber;
import java.util.Arrays;

/**
 * This class sets up the definitions for a simple color histogram for image
 * analysis.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class ColorHistogramVector extends DataInstance {

    // The number of bins of values in different color channels.
    public static final int DEFAULT_NUM_BINS = 4;
    private int numBins = DEFAULT_NUM_BINS;

    /**
     * Initialize the structure.
     */
    public ColorHistogramVector() {
        fAttr = new float[numBins * numBins];
        sAttr = new String[1];
    }

    /**
     * Initialize the structure.
     *
     * @param numBins The number of bins of values in different color channels.
     */
    public ColorHistogramVector(int numBins) {
        this.numBins = numBins;
        fAttr = new float[numBins * numBins];
        sAttr = new String[1];
    }

    public void populateFromImage(BufferedImage inputImage, String imageName) {
        sAttr[0] = imageName;
        int[][] imageBins = new int[numBins][numBins];
        double intervalWidth = 1. / (double) numBins;
        int imageWidth = inputImage.getWidth();
        int imageHeight = inputImage.getHeight();
        // Obtain the image raster.
        int[] imagePixels = new int[imageWidth * imageHeight];
        PixelGrabber firstGrabber = new PixelGrabber(inputImage, 0, 0,
                imageWidth, imageHeight, imagePixels, 0, imageWidth);
        try {
            firstGrabber.grabPixels();
        } catch (Exception e) {
            System.err.println(e.getMessage());
            System.err.println("PixelGrabber exception for image: "
                    + imageName);
        }
        int red;
        int green;
        int blue;
        double d_red;
        double d_green;
        double denominator;
        // Fill the bins for the image.
        for (int i = 0; i < imagePixels.length; i++) {
            red = (imagePixels[i] & 0x00ff0000) >> 16;
            green = (imagePixels[i] & 0x0000ff00) >> 8;
            blue = imagePixels[i] & 0x000000ff;
            denominator = red + green + blue;
            d_red = red / denominator;
            d_green = green / denominator;
            // Increase the value in the corresponding bin.
            imageBins[Math.min((int) (d_red / intervalWidth), 3)][
                    Math.min((int) (d_green / intervalWidth), 3)]++;
        }
        // Normalize.
        for (int i = 0; i < numBins; i++) {
            for (int j = 0; j < numBins; j++) {
                fAttr[i * numBins + j] =
                        ((float) imageBins[i][j]) /
                        ((float) imagePixels.length);
            }
        }
    }

    @Override
    public ColorHistogramVector copyContent() throws Exception {
        ColorHistogramVector instanceCopy = new ColorHistogramVector();
        instanceCopy.embedInDataset(getEmbeddingDataset());
        if (hasIntAtt()) {
            instanceCopy.iAttr = Arrays.copyOf(iAttr, iAttr.length);
        }
        if (hasFloatAtt()) {
            instanceCopy.fAttr = Arrays.copyOf(fAttr, fAttr.length);
        }
        if (hasNomAtt()) {
            instanceCopy.sAttr = Arrays.copyOf(sAttr, sAttr.length);
        }
        instanceCopy.numBins = numBins;
        return instanceCopy;
    }

    @Override
    public ColorHistogramVector copy() throws Exception {
        ColorHistogramVector instanceCopy;
        try {
            instanceCopy = this.copyContent();
        } catch (Exception e) {
            instanceCopy = null;
            throw e;
        }
        return instanceCopy;
    }
}
