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
package ioformat.images;

import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;

/**
 * This class implements the methods for generating image thumbnails of the
 * specified size.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class ThumbnailMaker {

    private int imageArea = 0;
    public static final int DEFAULT_AREA = 5000;

    /**
     * Initialization.
     */
    public ThumbnailMaker() {
        imageArea = DEFAULT_AREA;
    }

    /**
     * Initialization.
     *
     * @param imageArea Integer that is the desired image area of the thumbnail.
     */
    public ThumbnailMaker(int imageArea) {
        this.imageArea = imageArea;
    }

    /**
     * Create a thumbnail.
     *
     * @param bi BufferedImage that is the original image.
     * @return BufferedImage that is the thumbnail of the original image.
     */
    public BufferedImage createThumbnail(BufferedImage bi) {
        if (bi == null) {
            return null;
        }
        int imageWidth = bi.getWidth(null);
        int imageHeight = bi.getHeight(null);
        double imageRatio = (double) imageWidth / (double) imageHeight;
        int thumbHeight = (int) (Math.sqrt(((float) imageArea) / imageRatio));
        int thumbWidth = (int) ((float) thumbHeight * imageRatio);
        BufferedImage thumbImage = new BufferedImage(thumbWidth, thumbHeight,
                BufferedImage.TYPE_INT_RGB);
        Graphics2D graphics2D = thumbImage.createGraphics();
        graphics2D.setRenderingHint(RenderingHints.KEY_INTERPOLATION,
                RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        graphics2D.drawImage(bi, 0, 0, thumbWidth, thumbHeight, null);
        return thumbImage;
    }
}
