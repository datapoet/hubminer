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

import java.io.File;

/**
 * A utility class that makes sure that one directory has analogous files to a
 * reference directory. This is useful, for instance, in feature extraction when
 * a user extracts different feature types in different files in same file
 * structures in different directories. In such cases, some extractors might
 * fail and fail to produce files. For later comparisons, it is important to
 * take only those feature representations that have corresponding reference
 * representations - in other words, to take the intersection of all produced
 * corresponding files.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class DirectoryAnalogizer {

    /**
     * The default constructor.
     */
    public DirectoryAnalogizer() {
    }

    /**
     *
     * @param referenceDir Directory used for validating the file structure.
     * @param dir Directory to be validated.
     * @param extension File extension to consider.
     */
    public void analogizeDirectory(
            File referenceDir,
            File dir,
            String extension) {
        String[] extensions = new String[1];
        extensions[0] = extension;
        analogizeDirectory(referenceDir, dir, extensions);
    }

    /**
     * Deletes files (representations) that do not have a corresponding
     * representation in the reference directory.
     *
     * @param referenceDir Validation directory.
     * @param dir Directory that is being purged of extra files.
     * @param extensions String[] of all the extensions to consider.
     */
    public static void analogizeDirectory(
            File referenceDir,
            File dir,
            String[] extensions) {
        if (referenceDir == null || dir == null || extensions == null
                || !(referenceDir.exists() && referenceDir.isDirectory())
                || !(dir.exists() && dir.isDirectory())) {
            return;
        }
        File[] dirChildren = dir.listFiles();
        File checkFile;
        String s;
        String sName;
        for (int i = 0; i < dirChildren.length; i++) {
            s = dirChildren[i].getName();
            if (dirChildren[i].isDirectory()) {
                checkFile = new File(referenceDir, s);
                if (!checkFile.exists()) {
                    dirChildren[i].delete();
                } else {
                    analogizeDirectory(checkFile, dirChildren[i], extensions);
                }
            } else {
                sName = getNameWithExtensionRemoved(s);
                boolean exists = false;
                for (int j = 0; j < extensions.length; j++) {
                    checkFile = new File(referenceDir,
                            sName + "." + extensions[j]);
                    if (checkFile.exists()) {
                        exists = true;
                        break;
                    }
                    checkFile = new File(referenceDir,
                            sName + "." + extensions[j].toLowerCase());
                    if (checkFile.exists()) {
                        exists = true;
                        break;
                    }
                    checkFile = new File(referenceDir,
                            sName + "." + extensions[j].toUpperCase());
                    if (checkFile.exists()) {
                        exists = true;
                        break;
                    }
                }
                if (!exists) {
                    dirChildren[i].delete();
                }
            }
        }
    }

    /**
     * Gets a filename with the extension removed.
     *
     * @param fileName String that is the file name.
     * @return String that is the file name with extension removed.
     */
    public static String getNameWithExtensionRemoved(String fileName) {
        int loc = fileName.lastIndexOf(".");
        if (loc < 0) {
            return fileName;
        } else {
            return fileName.substring(0, loc);
        }
    }

    public static void info() {
        System.out.println("Deletes extra files in a directory that do not "
                + "match anything in the reference directory");
        System.out.println("Three parameters:");
        System.out.println("arg0: Reference directory");
        System.out.println("arg1: Directory that is going to get purged of "
                + "extra files");
        System.out.println("arg2: Extension of the files in the reference "
                + "directory");
    }

    /**
     * Performs deletions in the target directory of those files that have no
     * corresponding mates in the reference directory.
     *
     * @param args
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        if (args.length != 3) {
            info();
            return;
        }
        System.out.println("analogizing...");
        DirectoryAnalogizer da = new DirectoryAnalogizer();
        da.analogizeDirectory(new File(args[0]), new File(args[1]), args[2]);
        System.out.println("analogized...");
    }
}
