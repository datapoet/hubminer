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
package draw.basic;

import java.awt.Graphics;
import java.awt.image.BufferedImage;

/**
 * A utility class for converting color images into grayscale prior to feature
 * extraction.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class GrayScaleConverter {

    /**
     * @param r Integer value of the red channel.
     * @param g Integer value of the green channel.
     * @param b Integer value of the blue channel.
     * @return Luma for the given RGB specification.
     */
    public static int getLuma(int r, int g, int b) {
        return (int) (0.21 * r + 0.72 * g + 0.7 * b);
    }

    /**
     * @param rgb Integer array of three elements corresponding to the RGB
     * channels.
     * @return Luma for the given RGB specification.
     */
    public static int getLuma(int[] rgb) {
        return (int) (0.21 * rgb[0] + 0.72 * rgb[1] + 0.7 * rgb[2]);
    }

    /**
     * @param r Double value of the red channel.
     * @param g Double value of the green channel.
     * @param b Double value of the blue channel.
     * @return Luma for the given RGB specification.
     */
    public static double getLuma(double r, double g, double b) {
        return (0.21 * r + 0.72 * g + 0.7 * b);
    }

    /**
     * @param rgb Double array of three elements corresponding to the RGB
     * channels.
     * @return Luma for the given RGB specification.
     */
    public static double getLuma(double[] rgb) {
        return (0.21 * rgb[0] + 0.72 * rgb[1] + 0.7 * rgb[2]);
    }

    /**
     * Converts an image object to grayscale.
     *
     * @param colorImage BufferedImage object.
     * @return BufferedImage object that is the grayscale version of the one
     * that was passed in.
     */
    public static BufferedImage convertToGrayScale(BufferedImage colorImage) {
        if (colorImage == null) {
            return null;
        }
        BufferedImage image = new BufferedImage(colorImage.getWidth(),
                colorImage.getHeight(), BufferedImage.TYPE_BYTE_GRAY);
        Graphics g = image.getGraphics();
        g.drawImage(colorImage, 0, 0, null);
        g.dispose();
        return image;
    }
}
