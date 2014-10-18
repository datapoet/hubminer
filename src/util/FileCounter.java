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
import java.util.ArrayList;

/**
 * A utility class for traversing the directory structure, counting files and
 * fetching all relative paths of files in directories and subdirectories. It
 * focused on the specified file extension only.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class FileCounter {

    private File dir;
    private String fileExtension;

    /**
     *
     * @param dir Target directory.
     * @param fileExtension File extension to consider.
     */
    public FileCounter(File dir, String fileExtension) {
        this.dir = dir;
        this.fileExtension = fileExtension;
    }

    /**
     * Counts all the files of the extension specified in the constructor
     * previously.
     *
     * @return The number of files in the directory and all its subdirectories
     * that have a certain extension.
     */
    public int countFiles() {
        return countFilesInternal(dir);
    }

    /**
     * Counts all the files of the extension specified in the constructor
     * previously.
     *
     * @param dir File that is the directory to search.
     * @return The number of files in the directory and all its subdirectories
     * that have a certain extension.
     */
    private int countFilesInternal(File dir) {
        int res = 0;
        File[] children = dir.listFiles();
        for (int i = 0; i < children.length; i++) {
            if (children[i].isDirectory()) {
                res += countFilesInternal(children[i]);
            } else if (children[i].getPath().endsWith(fileExtension)) {
                res++;
            } else {
            }
        }
        return res;
    }

    /**
     * Finds all the relative paths to a directory that was specified in the
     * constructor.
     *
     * @return An ArrayList of relative paths of all files of the specified
     * extension that are located in this directory and its subdirectories. The
     * paths are relative to the current directory.
     */
    public ArrayList<String> findAllRelativePaths() {
        return findAllRelativePathsInternal(dir, "");
    }

    /**
     * Finds all the relative paths of files of a given extension to a directory
     * that contains them, recursively.
     *
     * @param dir File that is the current directory to search.
     * @param tempPath String that is the current temporary relative path from
     * the upper-level directory to this one.
     * @return ArrayList of relative paths to the upper-level directory that are
     * contained in this directory, filtered by the specified file extension.
     */
    private ArrayList<String> findAllRelativePathsInternal(
            File dir,
            String tempPath) {
        ArrayList<String> res = new ArrayList<>(30);
        ArrayList<String> relNameList;
        File[] children = dir.listFiles();
        for (int i = 0; i < children.length; i++) {
            if (children[i].isDirectory()) {
                relNameList = findAllRelativePathsInternal(
                        children[i], tempPath + File.separator
                        + children[i].getName());
                for (int j = 0; j < relNameList.size(); j++) {
                    res.add(relNameList.get(j));
                }
            } else if (children[i].getPath().endsWith(fileExtension)) {
                res.add(tempPath + File.separator + children[i].getName());
            } else {
            }
        }
        return res;
    }

    /**
     * Finds the names of all the files with a specified extension in the top
     * directory, recursively.
     *
     * @return An ArrayList of file names of a given extension contained in the
     * directory specified in FileCounter constructor.
     */
    public ArrayList<String> findAllRelativeNames() {
        return findAllRelativeNamesInternal(dir);
    }

    /**
     * Finds the names of all the files with a specified extension in the top
     * directory, recursively.
     *
     * @param dir File that is the current directory to search.
     * @return An ArrayList of file names of a given extension contained in the
     * currently searched directory.
     */
    private ArrayList<String> findAllRelativeNamesInternal(File dir) {
        ArrayList<String> res = new ArrayList<>(30);
        ArrayList<String> relNameList;
        File[] children = dir.listFiles();
        for (int i = 0; i < children.length; i++) {
            if (children[i].isDirectory()) {
                relNameList = findAllRelativeNamesInternal(children[i]);
                for (int j = 0; j < relNameList.size(); j++) {
                    res.add(relNameList.get(j));
                }
            } else if (children[i].getPath().endsWith(fileExtension)) {
                res.add(children[i].getName());
            } else {
            }
        }
        return res;
    }
}
