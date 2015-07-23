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
package util;

import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
import java.io.File;
import java.util.ArrayList;

/**
 * A utility class for image files.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class ImageUtil {

    /**
     * Copies an image into a new BufferedImage object.
     *
     * @param bi BufferedImage to copy.
     * @return Copy of the original image.
     */
    public static BufferedImage copyImage(BufferedImage bi) {
        if (bi == null) {
            return null;
        }
        WritableRaster wr = bi.copyData(null);
        BufferedImage output = new BufferedImage(
                bi.getColorModel(),
                wr,
                bi.isAlphaPremultiplied(),
                null);
        return output;
    }

    /**
     * Get the list of image names of a given extension in a directory.
     * Non-recursive.
     *
     * @param dir Target directory.
     * @param extension Extension to check for.
     * @return
     */
    public static ArrayList<String> getImageNameList(
            File dir,
            String extension) {
        if (dir == null || !dir.isDirectory()) {
            return null;
        } else {
            File[] children = dir.listFiles();
            if (children == null || children.length == 0) {
                return null;
            } else {
                ArrayList<String> result = new ArrayList<>(500);
                int index;
                for (int i = 0; i < children.length; i++) {
                    if (children[i].getName().endsWith(extension)) {
                        index = children[i].getName().lastIndexOf(".");
                        result.add(children[i].getName().substring(0, index));
                    }
                }
                return result;
            }
        }
    }

    /**
     * Get an array of image names of a given extension in a directory.
     * Non-recursive.
     *
     * @param dir Target directory.
     * @param extension Extension to check for.
     * @return
     */
    public static String[] getImageNamesArray(File dir, String extension) {
        if (dir == null || !dir.isDirectory()) {
            return null;
        } else {
            File[] children = dir.listFiles();
            // First just count.
            if (children == null || children.length == 0) {
                return null;
            } else {
                int count = 0;
                int index;
                for (int i = 0; i < children.length; i++) {
                    if (children[i].getName().endsWith(extension)) {
                        count++;
                    }
                }
                String[] result = new String[count];
                count = 0;
                for (int i = 0; i < children.length; i++) {
                    if (children[i].getName().endsWith(extension)) {
                        index = children[i].getName().lastIndexOf(".");
                        result[count++] =
                                children[i].getName().substring(0, index);
                    }
                }
                return result;
            }
        }
    }
}
