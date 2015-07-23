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

import data.representation.DataSet;
import ioformat.IOARFF;
import java.io.File;
import util.fileFilters.ARFFFileNameFilter;
import util.fileFilters.DirectoryFilter;

/**
 * Joins a directory of ARFF files into a single ARFF file.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class ARFFJoiner {

    /**
     * Command line args info.
     */
    public static void info() {
        System.out.println("Joins a folder of arff files into one"
                + "big arff file");
        System.out.println("arg0: inputDirectory");
        System.out.println("arg1: outFile");
    }

    /**
     * @param args Command line arguments.
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            info();
            return;
        }
        ARFFJoiner joiner = new ARFFJoiner(new File(args[0]), true);
        joiner.join(new File(args[1]));
    }
    File dir;
    boolean recursive;

    /**
     * @param dir Directory of ARFF files to join.
     * @param recursive Boolean flag indicating whether to join sub-dirs or not.
     */
    public ARFFJoiner(File dir, boolean recursive) {
        this.dir = dir;
        this.recursive = recursive;
    }

    public DataSet grabDSets() throws Exception {
        File[] arffFiles = dir.listFiles(new ARFFFileNameFilter());
        File[] subdirs = dir.listFiles(new DirectoryFilter());
        DataSet newDSet;
        DataSet[] dsets = new DataSet[arffFiles.length];
        IOARFF persister;
        for (int i = 0; i < arffFiles.length; i++) {
            persister = new IOARFF();
            dsets[i] = persister.load(arffFiles[i].getPath());
        }
        // Now collect the data from the subdirectories.
        DataSet subDSets = null;
        if (recursive) {
            ARFFJoiner tempJoiner;
            DataSet[] subCollections = new DataSet[subdirs.length];
            for (int i = 0; i < subdirs.length; i++) {
                tempJoiner = new ARFFJoiner(subdirs[i], recursive);
                subCollections[i] = tempJoiner.grabDSets();
            }
            try {
                subDSets = DataSetJoiner.join(subCollections);
            } catch (Exception e) {
                System.err.println(e.getMessage());
                subDSets = null;
            }
        }
        newDSet = DataSetJoiner.join(dsets);
        // If there were subdirectories and the recursive flag was on, join the
        // two data sources.
        if (subDSets != null) {
            DataSet[] both = new DataSet[2];
            both[0] = newDSet;
            both[1] = subDSets;
            newDSet = DataSetJoiner.join(both);
        }
        return newDSet;
    }

    /**
     * Join all the ARFF files from the specified directory into a single file
     * and persist it.
     *
     * @param targetFile
     * @throws Exception
     */
    public void join(File targetFile) throws Exception {
        DataSet allData = grabDSets();
        IOARFF persister = new IOARFF();
        if ((allData.getIdentifiers() == null)
                || allData.getIdentifiers().isEmpty()) {
            persister.save(allData, targetFile.getPath(), null);
        } else {
            File parentDir = targetFile.getParentFile();
            persister.save(
                    allData,
                    targetFile.getPath(),
                    parentDir.getPath() + File.separator
                    + targetFile.getName().substring(
                    0, targetFile.getName().lastIndexOf("."))
                    + "_PK.arff");
        }
    }
}
