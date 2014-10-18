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
package images.mining.calc;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.awt.image.PixelGrabber;

/**
 * This class implements the functionality for getting an average color in an
 * neighborhood of a point in an array of pixels.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class AverageColorGrabber {

    // Must be an odd number.
    public static final int WINDOW_WIDTH = 7;
    BufferedImage image;

    /**
     * @param image BufferedImage that is to be analyzed.
     */
    public AverageColorGrabber(BufferedImage image) {
        this.image = image;
    }

    /**
     * Get the average color in a neighborhood of a point.
     *
     * @param x Float that is the x coordinate.
     * @param y Float that is the y coordinate.
     * @return Integer array of red, green and blue color values of the
     * calculated average color.
     */
    public int[] getAverageColorInArray(float x, float y) {
        // Lower and upper bounds.
        int x_l = Math.max((int) x - WINDOW_WIDTH / 2, 0);
        int y_l = Math.max((int) y - WINDOW_WIDTH / 2, 0);
        int x_h = Math.min(image.getWidth() - 1, (int) x + WINDOW_WIDTH / 2);
        int y_h = Math.min(image.getHeight() - 1, (int) y + WINDOW_WIDTH / 2);
        // Width and height.
        int w_actual = x_h - x_l + 1;
        int h_actual = y_h - y_l + 1;
        int[] pixels = new int[h_actual * w_actual];
        PixelGrabber grabber = new PixelGrabber(image, x_l, y_l, w_actual,
                h_actual, pixels, 0, image.getWidth());
        try {
            grabber.grabPixels();
        } catch (Exception e) {
            System.err.println(e.getMessage());
        }
        int red;
        int green;
        int blue;
        float red_avg = 0;
        float blue_avg = 0;
        float green_avg = 0;
        // Go pixel by pixel, get the colors and update the averages.
        for (int i = 0; i < pixels.length; i++) {
            red = (pixels[i] & 0x00ff0000) >> 16;
            green = (pixels[i] & 0x0000ff00) >> 8;
            blue = pixels[i] & 0x000000ff;
            red_avg += red;
            green_avg += green;
            blue_avg += blue;
        }
        // Normalize the averages.
        red_avg /= pixels.length;
        blue_avg /= pixels.length;
        green_avg /= pixels.length;
        // Generate the resulting color array.
        int[] rgbArray = new int[3];
        rgbArray[0] = (int) red_avg;
        rgbArray[1] = (int) green_avg;
        rgbArray[2] = (int) blue_avg;
        return rgbArray;
    }

    /**
     * Get the average color in a neighborhood of a point.
     *
     * @param x Float that is the x coordinate.
     * @param y Float that is the y coordinate.
     * @return Color that is the average color in the neighborhood of the
     * specified point.
     */
    public Color getAverageColor(float x, float y) {
        int[] rgb = getAverageColorInArray(x, y);
        return new Color(rgb[0], rgb[1], rgb[2], 255);
    }
}
