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

import java.awt.Image;
import java.awt.Toolkit;
import java.awt.image.MemoryImageSource;

/**
 * This class implements a method for creating an Image object from a raster
 * with height and width specified.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class ImageFromRaster {

    /**
     * Generate an Image object for a raster with the specified width and
     * height.
     *
     * @param raster int[] that is the rater to be turned into an Image object.
     * @param width Integer that is the image width.
     * @param height Integer that is the image height.
     * @return
     */
    public static Image createImage(int[] raster, int width, int height) {
        MemoryImageSource mis = new MemoryImageSource(width, height, raster, 0,
                width);
        Toolkit tk = Toolkit.getDefaultToolkit();
        return tk.createImage(mis);
    }
}
