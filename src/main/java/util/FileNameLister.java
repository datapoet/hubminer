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
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;

/**
 * A utility class that fetches all relative paths of a given extension to a
 * given top directory, recursively, as well as a list of file names.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class FileNameLister {

    private File dir;
    private String ext;
    private ArrayList<String> names = null;

    /**
     *
     * @param dir Directory to examine.
     * @param ext String that is the file extension.
     */
    public FileNameLister(File dir, String ext) {
        this.dir = dir;
        this.ext = ext;
    }

    /**
     * Gets all the relative file paths of a given extension.
     *
     * @return ArrayList of Strings that are the relative paths of files of a
     * given extension contained in the specified top directory.
     */
    public ArrayList<String> getAllPaths() {
        if (dir == null || !(dir.exists() && dir.isDirectory())
                || ext == null) {
            return null;
        }
        FileCounter fc = new FileCounter(dir, ext);
        return fc.findAllRelativePaths();
    }

    /**
     * Gets all the relative file paths of a given extension.
     *
     * @return ArrayList of Strings that are the names of files of a given
     * extension contained in the specified top directory.
     */
    public ArrayList<String> getAllNames() {
        if (dir == null || !(dir.exists() && dir.isDirectory())
                || ext == null) {
            return null;
        }
        FileCounter fc = new FileCounter(dir, ext);
        names = fc.findAllRelativeNames();
        return names;
    }

    public void writeToFile(File outFile) throws Exception {
        if (names == null) {
            getAllNames();
        }
        if (names != null && names.size() > 0) {
            try (PrintWriter pw = new PrintWriter(new FileWriter(outFile))) {
                pw.println(names.size());
                for (int i = 0; i < names.size(); i++) {
                    pw.println(names.get(i));
                }
            } catch (Exception e) {
                throw e;
            }
        }
    }

    /**
     * Command line arguments information.
     */
    public static void info() {
        System.out.println("3 args");
        System.out.println("arg0: directory");
        System.out.println("arg1: extension");
        System.out.println("arg2: output file");
    }

    /**
     *
     * @param args Command line arguments: directory to search, file extension
     * to consider, output file to write the results to.
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        if (args.length != 3) {
            info();
        } else {
            FileNameLister fnl = new FileNameLister(new File(args[0]), args[1]);
            fnl.writeToFile(new File(args[2]));
        }
    }
}
