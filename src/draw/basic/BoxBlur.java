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

import java.awt.Dimension;

/**
 * Blur filter sets the value of each pixel to the average value of its
 * neighboring pixels. The filter kernel used here has a cross shape instead of
 * square, the size of the kernel depends on radius.
 *
 * @author phu004 (http://www.java-gaming.org/index.php?;topic=24808.0),
 * modified and extended by Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class BoxBlur {

    private int radius;

    /**
     * @param radius Radius of the transformation range.
     */
    public BoxBlur(int radius) {
        this.radius = radius;
    }

    /**
     * Perform the box blur.
     *
     * @param img Image raster.
     * @param offsSreenBuffer Buffer for off-screen processing.
     * @param dim Dimension object.
     */
    public void blurPixels(int[] img, int[] offSreenBuffer, Dimension dim) {

        // First blur the image in the horizontal direction then in the vertical
        // direction.
        horizontalBlur(img, offSreenBuffer, dim);
        verticalBlur(offSreenBuffer, img, dim);

        // Repeat once again for better quality.
        horizontalBlur(img, offSreenBuffer, dim);
        verticalBlur(offSreenBuffer, img, dim);

    }

    /**
     * Perform the horizontal blur.
     *
     * @param originalPixels Original image raster.
     * @param blurredPixels Blurred image raster.
     * @param dim Dimension object.
     */
    public void horizontalBlur(int[] originalPixels,
            int[] blurredPixels, Dimension dim) {
        int width = dim.width;
        int height = dim.height;
        int sourcePosition, destPosition, rgb1, rgb2, tr, tg, tb, pixelCount;
        for (int i = 0; i < height; i++) {
            sourcePosition = i * width;
            destPosition = i + height * (width - 1);
            tr = 0;
            tg = 0;
            tb = 0;
            for (int j = 0; j <= radius; j++) {
                rgb1 = originalPixels[sourcePosition + j];
                tr += ((rgb1 & 0xff0000) >> 16);
                tg += ((rgb1 & 0x00ff00) >> 8);
                tb += (rgb1 & 0xff);
            }
            pixelCount = radius + 1;
            blurredPixels[destPosition] = ((tr / pixelCount) << 16)
                    | ((tg / pixelCount) << 8) | (tb / pixelCount);
            sourcePosition++;
            destPosition -= height;
            pixelCount++;
            for (int j = 1; j <= radius; j++, sourcePosition++,
                    destPosition -= height, pixelCount++) {
                rgb1 = originalPixels[sourcePosition + radius];
                tr += ((rgb1 & 0xff0000) >> 16);
                tg += ((rgb1 & 0x00ff00) >> 8);
                tb += (rgb1 & 0xff);
                blurredPixels[destPosition] = ((tr / pixelCount) << 16)
                        | ((tg / pixelCount) << 8) | (tb / pixelCount);
            }
            pixelCount--;
            for (int j = radius + 1; j < width - radius; j++, sourcePosition++,
                    destPosition -= height) {
                rgb1 = originalPixels[sourcePosition + radius];
                rgb2 = originalPixels[sourcePosition - radius - 1];
                tr += (((rgb1 & 0xff0000) - (rgb2 & 0xff0000)) >> 16);
                tg += (((rgb1 & 0x00ff00) - (rgb2 & 0x00ff00)) >> 8);
                tb += ((rgb1 & 0xff) - (rgb2 & 0xff));
                blurredPixels[destPosition] = ((tr / pixelCount) << 16)
                        | ((tg / pixelCount) << 8) | (tb / pixelCount);
            }
            pixelCount--;
            for (int j = width - radius; j < width; j++, sourcePosition++,
                    destPosition -= height, pixelCount--) {
                rgb2 = originalPixels[sourcePosition - radius - 1];
                tr -= ((rgb2 & 0xff0000) >> 16);
                tg -= ((rgb2 & 0x00ff00) >> 8);
                tb -= (rgb2 & 0xff);
                blurredPixels[destPosition] = ((tr / pixelCount) << 16)
                        | ((tg / pixelCount) << 8) | (tb / pixelCount);
            }
        }
    }

    /**
     * Perform the vertical blur.
     *
     * @param originalPixels Original image raster.
     * @param blurredPixels Blurred image raster.
     * @param dim Dimension object.
     */
    public void verticalBlur(int[] originalPixels, int[] blurredPixels,
            Dimension dim) {
        int width = dim.width;
        int height = dim.height;
        int sourcePosition, destPosition, rgb1, rgb2, tr, tg, tb, pixelCount;
        for (int i = 0; i < width; i++) {
            sourcePosition = i * height;
            destPosition = (width - 1) - i;
            tr = 0;
            tg = 0;
            tb = 0;
            for (int j = 0; j <= radius; j++) {
                rgb1 = originalPixels[sourcePosition + j];
                tr += ((rgb1 & 0xff0000) >> 16);
                tg += ((rgb1 & 0x00ff00) >> 8);
                tb += (rgb1 & 0xff);
            }
            pixelCount = radius + 1;
            blurredPixels[destPosition] = ((tr / pixelCount) << 16)
                    | ((tg / pixelCount) << 8) | (tb / pixelCount);
            sourcePosition++;
            destPosition += width;
            pixelCount++;
            for (int j = 1; j <= radius; j++, sourcePosition++,
                    destPosition += width, pixelCount++) {
                rgb1 = originalPixels[sourcePosition + radius];
                tr += ((rgb1 & 0xff0000) >> 16);
                tg += ((rgb1 & 0x00ff00) >> 8);
                tb += (rgb1 & 0xff);
                blurredPixels[destPosition] = ((tr / pixelCount) << 16)
                        | ((tg / pixelCount) << 8) | (tb / pixelCount);
            }
            pixelCount--;
            for (int j = radius + 1; j < height - radius; j++, sourcePosition++,
                    destPosition += width) {
                rgb1 = originalPixels[sourcePosition + radius];
                rgb2 = originalPixels[sourcePosition - radius - 1];
                tr += (((rgb1 & 0xff0000) - (rgb2 & 0xff0000)) >> 16);
                tg += (((rgb1 & 0x00ff00) - (rgb2 & 0x00ff00)) >> 8);
                tb += ((rgb1 & 0xff) - (rgb2 & 0xff));
                blurredPixels[destPosition] = ((tr / pixelCount) << 16)
                        | ((tg / pixelCount) << 8) | (tb / pixelCount);
            }
            pixelCount--;
            for (int j = height - radius; j < height; j++, sourcePosition++,
                    destPosition += width, pixelCount--) {
                rgb2 = originalPixels[sourcePosition - radius - 1];
                tr -= ((rgb2 & 0xff0000) >> 16);
                tg -= ((rgb2 & 0x00ff00) >> 8);
                tb -= (rgb2 & 0xff);
                blurredPixels[destPosition] = ((tr / pixelCount) << 16)
                        | ((tg / pixelCount) << 8) | (tb / pixelCount);
            }
        }
    }

    /**
     * Perform the box blur.
     *
     * @param img Image raster.
     * @param offsSreenBuffer Buffer for off-screen processing.
     * @param dim Dimension object.
     */
    public void blurPixelsWithAlpha(int[] img, int[] offSreenBuffer,
            Dimension dim) {

        //blur image in horizontal direction then vertical direction
        horizontalBlurTransp(img, offSreenBuffer, dim);
        verticalBlurTransp(offSreenBuffer, img, dim);

        //repeat for another time for better quality
        horizontalBlurTransp(img, offSreenBuffer, dim);
        verticalBlurTransp(offSreenBuffer, img, dim);

    }

    /**
     * Perform the horizontal blur while handling transparency and the alpha
     * channel.
     *
     * @param originalPixels Original image raster.
     * @param blurredPixels Blurred image raster.
     * @param dim Dimension object.
     */
    public void horizontalBlurTransp(int[] originalPixels, int[] blurredPixels,
            Dimension dim) {
        int width = dim.width;
        int height = dim.height;
        int sourcePosition, destPosition, rgb1, rgb2, tr, tg, tb, pixelCount,
                alpha;
        for (int i = 0; i < height; i++) {
            sourcePosition = i * width;
            destPosition = i + height * (width - 1);
            tr = 0;
            tg = 0;
            tb = 0;
            alpha = 0;
            for (int j = 0; j <= radius; j++) {
                rgb1 = originalPixels[sourcePosition + j];
                tr += ((rgb1 & 0xff0000) >> 16);
                tg += ((rgb1 & 0x00ff00) >> 8);
                tb += (rgb1 & 0xff);
                alpha += ((rgb1 & 0xff000000) >> 24);
            }
            pixelCount = radius + 1;
            blurredPixels[destPosition] = ((alpha / pixelCount) << 24)
                    | ((tr / pixelCount) << 16) | ((tg / pixelCount) << 8)
                    | (tb / pixelCount);
            sourcePosition++;
            destPosition -= height;
            pixelCount++;
            for (int j = 1; j <= radius; j++, sourcePosition++,
                    destPosition -= height, pixelCount++) {
                rgb1 = originalPixels[sourcePosition + radius];
                tr += ((rgb1 & 0xff0000) >> 16);
                tg += ((rgb1 & 0x00ff00) >> 8);
                tb += (rgb1 & 0xff);
                alpha += ((rgb1 & 0xff000000) >> 24);
                blurredPixels[destPosition] = ((alpha / pixelCount) << 24)
                        | ((tr / pixelCount) << 16) | ((tg / pixelCount) << 8)
                        | (tb / pixelCount);
            }
            pixelCount--;
            for (int j = radius + 1; j < width - radius; j++, sourcePosition++,
                    destPosition -= height) {
                rgb1 = originalPixels[sourcePosition + radius];
                rgb2 = originalPixels[sourcePosition - radius - 1];
                tr += (((rgb1 & 0xff0000) - (rgb2 & 0xff0000)) >> 16);
                tg += (((rgb1 & 0x00ff00) - (rgb2 & 0x00ff00)) >> 8);
                tb += ((rgb1 & 0xff) - (rgb2 & 0xff));
                alpha += ((rgb1 & 0xff000000) >> 24)
                        - ((rgb2 & 0xff000000) >> 24);
                blurredPixels[destPosition] = ((alpha / pixelCount) << 24)
                        | ((tr / pixelCount) << 16) | ((tg / pixelCount) << 8)
                        | (tb / pixelCount);
            }
            pixelCount--;
            for (int j = width - radius; j < width; j++, sourcePosition++,
                    destPosition -= height, pixelCount--) {
                rgb2 = originalPixels[sourcePosition - radius - 1];
                tr -= ((rgb2 & 0xff0000) >> 16);
                tg -= ((rgb2 & 0x00ff00) >> 8);
                tb -= (rgb2 & 0xff);
                alpha -= ((rgb2 & 0xff000000) >> 24);
                blurredPixels[destPosition] = ((alpha / pixelCount) << 24)
                        | ((tr / pixelCount) << 16) | ((tg / pixelCount) << 8)
                        | (tb / pixelCount);
            }
        }
    }

    /**
     * Perform the vertical blur while handling transparency and the alpha
     * channel.
     *
     * @param originalPixels Original image raster.
     * @param blurredPixels Blurred image raster.
     * @param dim Dimension object.
     */
    public void verticalBlurTransp(int[] originalPixels, int[] blurredPixels,
            Dimension dim) {
        int width = dim.width;
        int height = dim.height;
        int sourcePosition, destPosition, rgb1, rgb2, tr, tg, tb, pixelCount,
                alpha;
        for (int i = 0; i < width; i++) {
            sourcePosition = i * height;
            destPosition = (width - 1) - i;
            tr = 0;
            tg = 0;
            tb = 0;
            alpha = 0;
            for (int j = 0; j <= radius; j++) {
                rgb1 = originalPixels[sourcePosition + j];
                tr += ((rgb1 & 0xff0000) >> 16);
                tg += ((rgb1 & 0x00ff00) >> 8);
                tb += (rgb1 & 0xff);
                alpha += ((rgb1 & 0xff000000) >> 24);
            }
            pixelCount = radius + 1;
            blurredPixels[destPosition] = ((alpha / pixelCount) << 24)
                    | ((tr / pixelCount) << 16) | ((tg / pixelCount) << 8)
                    | (tb / pixelCount);
            sourcePosition++;
            destPosition += width;
            pixelCount++;
            for (int j = 1; j <= radius; j++, sourcePosition++,
                    destPosition += width, pixelCount++) {
                rgb1 = originalPixels[sourcePosition + radius];
                tr += ((rgb1 & 0xff0000) >> 16);
                tg += ((rgb1 & 0x00ff00) >> 8);
                tb += (rgb1 & 0xff);
                alpha += ((rgb1 & 0xff000000) >> 24);
                blurredPixels[destPosition] = ((alpha / pixelCount) << 24)
                        | ((tr / pixelCount) << 16) | ((tg / pixelCount) << 8)
                        | (tb / pixelCount);
            }
            pixelCount--;
            for (int j = radius + 1; j < height - radius; j++, sourcePosition++,
                    destPosition += width) {
                rgb1 = originalPixels[sourcePosition + radius];
                rgb2 = originalPixels[sourcePosition - radius - 1];
                tr += (((rgb1 & 0xff0000) - (rgb2 & 0xff0000)) >> 16);
                tg += (((rgb1 & 0x00ff00) - (rgb2 & 0x00ff00)) >> 8);
                tb += ((rgb1 & 0xff) - (rgb2 & 0xff));
                alpha += ((rgb1 & 0xff000000) >> 24)
                        - ((rgb2 & 0xff000000) >> 24);
                blurredPixels[destPosition] = ((alpha / pixelCount) << 24)
                        | ((tr / pixelCount) << 16) | ((tg / pixelCount) << 8)
                        | (tb / pixelCount);
            }
            pixelCount--;
            for (int j = height - radius; j < height; j++, sourcePosition++,
                    destPosition += width, pixelCount--) {
                rgb2 = originalPixels[sourcePosition - radius - 1];
                tr -= ((rgb2 & 0xff0000) >> 16);
                tg -= ((rgb2 & 0x00ff00) >> 8);
                tb -= (rgb2 & 0xff);
                alpha -= ((rgb2 & 0xff000000) >> 24);
                blurredPixels[destPosition] = ((alpha / pixelCount) << 24)
                        | ((tr / pixelCount) << 16) | ((tg / pixelCount) << 8)
                        | (tb / pixelCount);
            }
        }
    }
}
