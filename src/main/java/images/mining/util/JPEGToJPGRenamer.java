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
package images.mining.util;

import java.io.File;

/**
 * A utility script for renaming all jpeg files to jpg. Nothing else changes,
 * but some image processing packages have rather rigid extension regex checks,
 * so it is sometimes useful to have uniform naming throughout the data.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class JPEGToJPGRenamer {

    /**
     * Apply jpeg to jpg renaming recursively.
     *
     * @param dir File that is the directory where to apply the recursive jpeg
     * to jpg renaming.
     */
    public static void convertDir(File dir) {
        File[] children = dir.listFiles();
        String newName;
        int index;
        if (children != null && children.length > 0) {
            for (int i = 0; i < children.length; i++) {
                if (children[i].isDirectory()) {
                    convertDir(children[i]);
                } else {
                    if (children[i].getPath().endsWith(".JPEG")
                            || children[i].getPath().endsWith(".jpeg")) {
                        index = children[i].getPath().lastIndexOf('.');
                        newName = children[i].getPath().substring(0, index)
                                + ".jpg";
                        children[i].renameTo(new File(newName));
                    }
                }
            }
        }
    }

    /**
     * Perform the jpeg to jpg renaming.
     *
     * @param args One command line argument - the target directory path.
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            System.out.println("directory where jpeg files will be renamed to "
                    + "jpg files recursively");
        } else {
            convertDir(new File(args[0]));
        }
    }
}
