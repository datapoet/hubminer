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
package ioformat;

import data.representation.DataSet;
import java.io.File;
import java.io.IOException;

/**
 * A class that implements methods for labeled data load from possible multiple
 * data formats. It tries several options in a certain order and settles on the
 * one that finishes successfully. This is a good option to use in more complex
 * components where the input format can vary and cannot be fixed a priori.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class SupervisedLoader {

    /**
     * Loads labeled data into a DataSet object.
     *
     * @param inFile File to load the data from.
     * @param multiLabelMode Boolean flag indicating whether to load the labels
     * from the data file if in single labeled mode, where the class label is
     * the last one on each line or to go into the multi-labeled mode, where
     * labels are loaded later from a separate file, outside of this method or
     * this class.
     * @return DataSet object with loaded data.
     * @throws Exception
     */
    public static DataSet loadData(File inFile, boolean multiLabelMode)
            throws Exception {
        if (inFile == null || !inFile.exists() || !inFile.isFile()) {
            throw new IOException("Non-existing data path provided.");
        }
        DataSet originalDSet = null;
        String inPath = inFile.getPath();
        if (inPath.startsWith("sparse:")) {
            String trueDSPath = inPath.substring(inPath.indexOf(':') + 1,
                    inPath.length());
            IOARFF pers = new IOARFF();
            originalDSet = pers.loadSparse(trueDSPath);
        } else {
            if (inPath.endsWith(".csv")) {
                if (multiLabelMode) {
                    try {
                        IOCSV reader = new IOCSV(false, ",");
                        originalDSet = reader.readData(inFile);
                    } catch (Exception e) {
                        IOCSV reader = new IOCSV(false, " +");
                        originalDSet = reader.readData(inFile);
                    }
                } else {
                    try {
                        IOCSV reader = new IOCSV(true, ",");
                        originalDSet = reader.readData(inFile);
                    } catch (Exception e) {
                        IOCSV reader = new IOCSV(true, " +");
                        originalDSet = reader.readData(inFile);
                    }
                }
            } else if (inPath.endsWith(".tsv")) {
                if (multiLabelMode) {
                    try {
                        IOCSV reader = new IOCSV(false, " +");
                        originalDSet = reader.readData(inFile);
                    } catch (Exception e) {
                        IOCSV reader = new IOCSV(false, "\t");
                        originalDSet = reader.readData(inFile);
                    }
                } else {
                    try {
                        IOCSV reader = new IOCSV(true, " +");
                        originalDSet = reader.readData(inFile);
                    } catch (Exception e) {
                        IOCSV reader = new IOCSV(true, "\t");
                        originalDSet = reader.readData(inFile);
                    }
                }
            } else if (inPath.endsWith(".arff")) {
                IOARFF persister = new IOARFF();
                originalDSet = persister.load(inPath);
            }
        }
        return originalDSet;
    }

    /**
     * Loads labeled data into a DataSet object.
     *
     * @param inPath String that is the path to load the data from.
     * @param multiLabelMode Boolean flag indicating whether to load the labels
     * from the data file if in single labeled mode, where the class label is
     * the last one on each line or to go into the multi-labeled mode, where
     * labels are loaded later from a separate file, outside of this method or
     * this class.
     * @return DataSet object with loaded data.
     * @throws Exception
     */
    public static DataSet loadData(String inPath, boolean multiLabelMode)
            throws Exception {
        File inFile = new File(inPath);
        return loadData(inFile, multiLabelMode);
    }
}
