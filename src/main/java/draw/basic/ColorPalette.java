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

import java.awt.Color;
import java.util.Random;

/**
 * This class encodes some basic colors that are used in UIs. The alpha values
 * are given as a parameter in the constructor.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class ColorPalette {

    public int alpha = 255;
    public Color DARK_GREEN;
    public Color NEON_BLUE;
    public Color OLD_GOLD;
    public Color SLATE_BLUE;
    public Color STEEL_BLUE;
    public Color YELLOW_GREEN;
    public Color SPRING_GREEN;
    public Color LIME_GREEN;
    public Color HUNTER_GREEN;
    public Color DIM_GREY;
    public Color DARK_OLIVE_GREEN;
    public Color FIREBRICK_RED;
    public Color ORANGE;
    public Color ORANGE_RED;
    public Color PLUM;
    public Color PEACH_PUFF;
    public Color SALMON;
    public Color BRONZE;
    public Color COPPER;
    public Color MAROON;
    public Color KHAKI;
    public Color TAN;
    public Color VIOLET;
    public Color GREY;
    public Color DARK_BROWN;
    public Color DARK_WOOD;
    public Color SKY_BLUE;
    public Color SPICY_PINK;
    public Color MEDIUM_SPRING_GREEN;
    public Color MEDIUM_AQUAMARINE;
    // This array is used as a predefined sequence of class colors in hub
    // explorer and related UIs.
    public Color[] classColorArray;

    /**
     *
     */
    public ColorPalette() {
        initColors();
    }

    /**
     * Initialization of the color values.
     *
     * @param alpha Integer alpha channel value.
     */
    public ColorPalette(int alpha) {
        this.alpha = alpha;
        initColors();
    }

    /**
     * Initialization of the color values.
     *
     * @param alpha Double alpha channel value.
     */
    public ColorPalette(double alpha) {
        this.alpha = (int) (alpha * 255);
        initColors();
    }

    /**
     * Initializes the colors in the palette.
     */
    public final void initColors() {
        DARK_GREEN = new Color(47, 79, 47, alpha);
        NEON_BLUE = new Color(77, 77, 255, alpha);
        OLD_GOLD = new Color(207, 181, 59, alpha);
        SLATE_BLUE = new Color(0, 127, 255, alpha);
        STEEL_BLUE = new Color(35, 107, 142, alpha);
        YELLOW_GREEN = new Color(153, 204, 50, alpha);
        SPRING_GREEN = new Color(0, 255, 127, alpha);
        LIME_GREEN = new Color(50, 205, 50, alpha);
        HUNTER_GREEN = new Color(33, 94, 33, alpha);
        DIM_GREY = new Color(84, 84, 84, alpha);
        DARK_OLIVE_GREEN = new Color(79, 79, 27, alpha);
        FIREBRICK_RED = new Color(128, 35, 35, alpha);
        ORANGE = new Color(255, 127, 0, alpha);
        ORANGE_RED = new Color(255, 36, 0, alpha);
        PLUM = new Color(234, 173, 234, alpha);
        PEACH_PUFF = new Color(255, 224, 137, alpha);
        SALMON = new Color(111, 66, 66, alpha);
        BRONZE = new Color(140, 120, 83, alpha);
        COPPER = new Color(184, 115, 51, alpha);
        MAROON = new Color(142, 35, 107, alpha);
        KHAKI = new Color(159, 159, 95, alpha);
        TAN = new Color(219, 147, 112, alpha);
        VIOLET = new Color(79, 47, 79, alpha);
        GREY = new Color(192, 192, 192, alpha);
        DARK_BROWN = new Color(92, 64, 51, alpha);
        DARK_WOOD = new Color(133, 94, 66, alpha);
        SKY_BLUE = new Color(50, 153, 204, alpha);
        SPICY_PINK = new Color(255, 28, 174, alpha);
        MEDIUM_SPRING_GREEN = new Color(127, 255, 0, alpha);
        MEDIUM_AQUAMARINE = new Color(50, 205, 153, alpha);
        // Initialize the array of subsequent class colors for visualization.
        classColorArray = new Color[20];
        classColorArray[0] = NEON_BLUE;
        classColorArray[1] = FIREBRICK_RED;
        classColorArray[2] = OLD_GOLD;
        classColorArray[3] = SPRING_GREEN;
        classColorArray[4] = DARK_BROWN;
        classColorArray[5] = ORANGE;
        classColorArray[6] = LIME_GREEN;
        classColorArray[7] = PLUM;
        classColorArray[8] = MEDIUM_AQUAMARINE;
        classColorArray[9] = GREY;
        classColorArray[10] = MAROON;
        classColorArray[11] = TAN;
        classColorArray[12] = BRONZE;
        classColorArray[13] = DARK_OLIVE_GREEN;
        classColorArray[14] = ORANGE_RED;
        classColorArray[15] = YELLOW_GREEN;
        classColorArray[16] = VIOLET;
        classColorArray[17] = PEACH_PUFF;
        classColorArray[18] = SKY_BLUE;
        classColorArray[19] = DIM_GREY;
    }

    /**
     *
     * @param length Integer that is the length of the array to fetch.
     * @return An array of class colors for visualization.
     */
    public Color[] getClassColorArray(int length) {
        Color[] result = new Color[length];
        for (int i = 0; i < length; i++) {
            if (i < 20) {
                result[i] = classColorArray[i];
            } else {
                // If the original array is not enough, generate some random
                // colors.
                switch (i % 6) {
                    case 0:
                        result[i] = getRandomBlue();
                        break;
                    case 1:
                        result[i] = getRandomRed();
                        break;
                    case 2:
                        result[i] = getRandomYellow();
                        break;
                    case 3:
                        result[i] = getRandomRed();
                        break;
                    case 4:
                        result[i] = getRandomCyan();
                        break;
                    case 5:
                        result[i] = getRandomMagenta();
                        break;
                    default:
                        result[i] = getRandomGreen();
                }
            }
        }
        return result;
    }

    /**
     * Transforms HSL into RGB.
     *
     * @param inArray An array of three double values that encodes the HSL color
     * representation.
     * @return Double array that encodes the RGB color representation.
     */
    public double[] getRGBDoubleArrayFromHSL(double[] inArray) {
        return getRGBDoubleArrayFromHSL(inArray[0], inArray[1], inArray[2]);
    }

    /**
     * Transforms HSL into RGB.
     *
     * @param h Double representing hue.
     * @param s Double representing saturation.
     * @param l Double representing luminosity.
     * @return Integer array representing the corresponding RGB values.
     */
    private int[] getRGBIntArrayFromHSL(double h, double s, double l) {
        double[] dArray = getRGBDoubleArrayFromHSL(h, s, l);
        int[] outArray = new int[3];
        outArray[0] = (int) (dArray[0] * 255);
        outArray[1] = (int) (dArray[1] * 255);
        outArray[2] = (int) (dArray[2] * 255);
        Random randa = new Random();
        if (outArray[0] > 255 || outArray[0] < 0) {
            outArray[0] = randa.nextInt(255);
        }
        if (outArray[1] > 255 || outArray[1] < 0) {
            outArray[1] = randa.nextInt(255);
        }
        if (outArray[2] > 255 || outArray[2] < 0) {
            outArray[2] = randa.nextInt(255);
        }
        return outArray;
    }

    /**
     * Transforms HSL into RGB.
     *
     * @param inArray An array of three double values that encodes the HSL color
     * representation.
     * @return Integer array that encodes the RGB color representation.
     */
    public int[] getRGBIntArrayFromHSL(double[] inArray) {
        double[] dArray = getRGBDoubleArrayFromHSL(
                inArray[0], inArray[1], inArray[2]);
        int[] outArray = new int[3];
        outArray[0] = (int) (dArray[0] * 255);
        outArray[1] = (int) (dArray[1] * 255);
        outArray[2] = (int) (dArray[2] * 255);
        return outArray;
    }

    /**
     * Transforms HSL into RGB.
     *
     * @param h Double representing hue.
     * @param s Double representing saturation.
     * @param l Double representing luminosity.
     * @return Double array representing the corresponding RGB values.
     */
    private double[] getRGBDoubleArrayFromHSL(double h, double s, double l) {
        // h is in [0, 2 * pi] interval
        // s is in [0, 1] interval
        // l is in [0, 1] interval
        double r_d = 0.;
        double g_d = 0.;
        double b_d = 0.;
        double[] output = new double[3];
        double chroma;
        if (l > 0.5) {
            chroma = (2 - 2 * l) * s;
        } else {
            chroma = 2 * l * s;
        }
        double h_1 = h / (Math.PI / 3);
        int h_1i = (int) h_1;
        double x = chroma * (1. - Math.abs(((h_1i % 2) + h_1 - h_1i) - 1.));
        switch (h_1i) {
            case 0:
                r_d = chroma;
                g_d = x;
                break;
            case 1:
                r_d = x;
                g_d = chroma;
                break;
            case 2:
                g_d = chroma;
                b_d = x;
                break;
            case 3:
                g_d = x;
                b_d = chroma;
                break;
            case 4:
                r_d = x;
                b_d = chroma;
                break;
            case 5:
                r_d = chroma;
                b_d = x;
                break;
        }
        double m = l - 0.5 * chroma;
        r_d += m;
        g_d += m;
        b_d += m;
        if (r_d > 1) {
            r_d = 1;
        }
        if (g_d > 1) {
            g_d = 1;
        }
        if (b_d > 1) {
            b_d = 1;
        }
        output[0] = r_d;
        output[1] = g_d;
        output[2] = b_d;
        return output;
    }

    /**
     * @return A random green color.
     */
    public Color getRandomGreen() {
        // Hue 120 deg.
        return getRandomGreen(255);
    }

    /**
     * Generates a random green color.
     *
     * @param alpha Integer value of the alpha channel.
     * @return A random green color.
     */
    public Color getRandomGreen(int alpha) {
        // Hue 120 deg.
        double center = Math.PI * 2 / 3;
        Random randa = new Random();
        double hue = Math.abs(center
                + Math.min(Math.max(randa.nextGaussian(), -3), 3) * 6.);
        double saturation = 0.05 + randa.nextDouble() * 0.9;
        double lightness = 0.05 + randa.nextDouble() * 0.9;
        int[] rgb = getRGBIntArrayFromHSL(hue, saturation, lightness);
        return new Color(rgb[0], rgb[1], rgb[2], alpha);
    }

    /**
     * @return A random red color.
     */
    public Color getRandomRed() {
        // Hue 0 deg.
        return getRandomRed(255);
    }

    /**
     * Generates a random red color.
     *
     * @param alpha Integer value of the alpha channel.
     * @return A random red color.
     */
    public Color getRandomRed(int alpha) {
        // Hue 0 deg.
        double center = 0;
        Random randa = new Random();
        double hue = Math.abs(center + Math.min(
                Math.max(randa.nextGaussian(), -3), 3) * 6.);
        double saturation = 0.05 + randa.nextDouble() * 0.9;
        double lightness = 0.05 + randa.nextDouble() * 0.9;
        int[] rgb = getRGBIntArrayFromHSL(hue, saturation, lightness);
        return new Color(rgb[0], rgb[1], rgb[2], alpha);
    }

    /**
     * @return A random yellow color.
     */
    public Color getRandomYellow() {
        // Hue 60 deg.
        return getRandomYellow(255);
    }

    /**
     * Generates a random yellow color.
     *
     * @param alpha Integer value of the alpha channel.
     * @return A random yellow color.
     */
    public Color getRandomYellow(int alpha) {
        // Hue 60 deg.
        double center = Math.PI * 1 / 3;
        Random randa = new Random();
        double hue = Math.abs(center + Math.min(
                Math.max(randa.nextGaussian(), -3), 3) * 6.);
        double saturation = 0.05 + randa.nextDouble() * 0.9;
        double lightness = 0.05 + randa.nextDouble() * 0.9;
        int[] rgb = getRGBIntArrayFromHSL(hue, saturation, lightness);
        return new Color(rgb[0], rgb[1], rgb[2], alpha);
    }

    /**
     * @return A random cyan color.
     */
    public Color getRandomCyan() {
        // Hue 180 deg.
        return getRandomCyan(255);
    }

    /**
     * Generates a random cyan color.
     *
     * @param alpha Integer value of the alpha channel.
     * @return A random cyan color.
     */
    public Color getRandomCyan(int alpha) {
        // Hue 180 deg.
        double center = Math.PI;
        Random randa = new Random();
        double hue = Math.abs(center + Math.min(
                Math.max(randa.nextGaussian(), -3), 3) * 6.);
        double saturation = 0.05 + randa.nextDouble() * 0.9;
        double lightness = 0.05 + randa.nextDouble() * 0.9;
        int[] rgb = getRGBIntArrayFromHSL(hue, saturation, lightness);
        return new Color(rgb[0], rgb[1], rgb[2], alpha);
    }

    /**
     * @return A random blue color.
     */
    public Color getRandomBlue() {
        // Hue 240 deg.
        return getRandomBlue(255);
    }

    /**
     * Generates a random blue color.
     *
     * @param alpha Integer value of the alpha channel.
     * @return A random blue color.
     */
    public Color getRandomBlue(int alpha) {
        // Hue 240 deg.
        double center = Math.PI * 4 / 3;
        Random randa = new Random();
        double hue = Math.abs(center + Math.min(
                Math.max(randa.nextGaussian(), -3), 3) * 6.);
        double saturation = 0.05 + randa.nextDouble() * 0.9;
        double lightness = 0.05 + randa.nextDouble() * 0.9;
        int[] rgb = getRGBIntArrayFromHSL(hue, saturation, lightness);
        return new Color(rgb[0], rgb[1], rgb[2], alpha);
    }

    /**
     * @return A random magenta color.
     */
    public Color getRandomMagenta() {
        // Hue 300 deg.
        return getRandomMagenta(255);
    }

    /**
     * Generates a random magenta color.
     *
     * @param alpha Integer value of the alpha channel.
     * @return A random magenta color.
     */
    public Color getRandomMagenta(int alpha) {
        // Hue 300 deg.
        double center = Math.PI * 5 / 3;
        Random randa = new Random();
        double hue = Math.abs(center + Math.min(
                Math.max(randa.nextGaussian(), -3), 3) * 6.);
        double saturation = 0.05 + randa.nextDouble() * 0.9;
        double lightness = 0.05 + randa.nextDouble() * 0.9;
        int[] rgb = getRGBIntArrayFromHSL(hue, saturation, lightness);
        return new Color(rgb[0], rgb[1], rgb[2], alpha);
    }
}
