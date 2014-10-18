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

import data.representation.DataInstance;
import data.representation.DataSet;
import data.representation.util.DataMineConstants;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;

/**
 * This class handles loading and persisting data in CSV format.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class IOCSV {

    // If the data is loaded or saved with the category, then the category is
    // read from or appended to the end of the line as a last item. If not, the
    // data is loaded or saved without the labels.
    boolean withCategory = true;
    String separator;
    int type;

    /**
     * Initialization.
     *
     * @param withCategory Boolean flag indicating whether to operate in a
     * supervised or unsupervised mode. If set to true, the labels will be read
     * from or appended to the end of the line.
     * @param separator String that is the separator to use (can be a regex).
     * @param type Integer that is the feature type to load, as per
     * DataMineConstants.
     */
    public IOCSV(boolean withCategory, String separator, int type) {
        this.withCategory = withCategory;
        this.separator = separator;
        this.type = type;
    }

    /**
     * Initialization. Sets the default data type to float.
     *
     * @param withCategory Boolean flag indicating whether to operate in a
     * supervised or unsupervised mode. If set to true, the labels will be read
     * from or appended to the end of the line.
     * @param separator String that is the separator to use (can be a regex).
     */
    public IOCSV(boolean withCategory, String separator) {
        this.withCategory = withCategory;
        this.separator = separator;
        this.type = DataMineConstants.FLOAT;
    }

    /**
     * Write the data to an output csv file.
     *
     * @param dset DataSet object to persist.
     * @param outFile File that is the target output.
     * @throws Exception
     */
    public void writeData(DataSet dset, File outFile) throws Exception {
        if (dset == null || dset.isEmpty()) {
            return;
        }
        // Count the number of attributes.
        int numAtt = 0;
        switch (type) {
            case DataMineConstants.FLOAT: {
                numAtt = dset.getNumFloatAttr();
                break;
            }
            case DataMineConstants.INTEGER: {
                numAtt = dset.getNumIntAttr();
                break;
            }
            case DataMineConstants.NOMINAL: {
                numAtt = dset.getNumNominalAttr();
                break;
            }
        }
        // Initialize the output stream.
        PrintWriter pw = new PrintWriter(new FileWriter(outFile));
        try {
            if (withCategory) {
                // If labels are also to be persisted.
                for (DataInstance instance : dset.data) {
                    for (int attIndex = 0; attIndex < numAtt; attIndex++) {
                        switch (type) {
                            case DataMineConstants.FLOAT: {
                                pw.print(instance.fAttr[attIndex] + separator);
                                break;
                            }
                            case DataMineConstants.INTEGER: {
                                pw.print(instance.iAttr[attIndex] + separator);
                                break;
                            }
                            case DataMineConstants.NOMINAL: {
                                pw.print(instance.sAttr[attIndex] + separator);
                                break;
                            }
                        }
                    }
                    pw.print(instance.getCategory());
                    pw.println();
                }
            } else {
                for (DataInstance instance : dset.data) {
                    switch (type) {
                        case DataMineConstants.FLOAT: {
                            pw.print(instance.fAttr[0]);
                            break;
                        }
                        case DataMineConstants.INTEGER: {
                            pw.print(instance.iAttr[0]);
                            break;
                        }
                        case DataMineConstants.NOMINAL: {
                            pw.print(instance.sAttr[0]);
                            break;
                        }
                    }
                    for (int attIndex = 1; attIndex < numAtt; attIndex++) {
                        switch (type) {
                            case DataMineConstants.FLOAT: {
                                pw.print(separator + instance.fAttr[attIndex]);
                                break;
                            }
                            case DataMineConstants.INTEGER: {
                                pw.print(separator + instance.iAttr[attIndex]);
                                break;
                            }
                            case DataMineConstants.NOMINAL: {
                                pw.print(separator + instance.sAttr[attIndex]);
                                break;
                            }
                        }
                    }
                    pw.println();
                }
            }
        } catch (Exception e) {
            System.err.println(e.getMessage());
            throw e;
        } finally {
            pw.close();
        }
    }

    /**
     * Reads CSV data into a DataSet object.
     *
     * @param inFile File to load the data from.
     * @return DataSet object that is loaded from the target input file.
     * @throws Exception
     */
    public DataSet readData(File inFile) throws Exception {
        try (BufferedReader br = new BufferedReader(new InputStreamReader(
                     new FileInputStream(inFile)))) {
            HashMap<String, Integer> classNameToIndexMap = new HashMap<>(100);
            int maxClassIndex = -1;
            String[] lineItems;
            String line = br.readLine();
            line = line.trim();
            // Read the first instance to see how many features there are.
            lineItems = line.split(separator);
            int numAtt;
            if (withCategory) {
                // In this case, class is the last entry in the line.
                numAtt = lineItems.length - 1;
            } else {
                numAtt = lineItems.length;
            }
            DataSet dset = new DataSet();
            // Generate feature names.
            if (type == DataMineConstants.FLOAT) {
                dset.fAttrNames = new String[numAtt];
                for (int i = 0; i < numAtt; i++) {
                    dset.fAttrNames[i] = "fAtt" + i;
                }
            } else if (type == DataMineConstants.INTEGER) {
                dset.iAttrNames = new String[numAtt];
                for (int i = 0; i < numAtt; i++) {
                    dset.iAttrNames[i] = "iAtt" + i;
                }
            } else {
                dset.sAttrNames = new String[numAtt];
                for (int i = 0; i < numAtt; i++) {
                    dset.sAttrNames[i] = "nAtt" + i;
                }
            }
            int initialCapacity = 2000;
            dset.data = new ArrayList<>(initialCapacity);
            DataInstance instance = new DataInstance(dset);
            instance.embedInDataset(dset);
            // Load the feature values.
            for (int attIndex = 0; attIndex < numAtt; attIndex++) {
                if (type == DataMineConstants.FLOAT) {
                    instance.fAttr[attIndex] = Float.parseFloat(
                            lineItems[attIndex].trim());
                } else if (type == DataMineConstants.INTEGER) {
                    instance.iAttr[attIndex] = (int) Float.parseFloat(
                            lineItems[attIndex].trim());
                } else {
                    instance.sAttr[attIndex] = lineItems[attIndex].trim();
                }
            }
            // Load the category, if working in the supervised mode.
            if (withCategory) {
                String classNameString = lineItems[lineItems.length - 1].trim();
                if (!classNameToIndexMap.containsKey(classNameString)) {
                    ++maxClassIndex;
                    classNameToIndexMap.put(classNameString,
                            maxClassIndex);
                }
                instance.setCategory(classNameToIndexMap.get(
                        classNameString));
            }
            dset.addDataInstance(instance);
            line = br.readLine();
            // Now handle the remaining lines in the file.
            try {
                while (line != null) {
                    line = line.trim();
                    lineItems = line.split(separator);
                    instance = new DataInstance(dset);
                    instance.embedInDataset(dset);
                    // Load the feature values.
                    for (int attIndex = 0; attIndex < numAtt; attIndex++) {
                        if (type == DataMineConstants.FLOAT) {
                            instance.fAttr[attIndex] = Float.parseFloat(
                                    lineItems[attIndex].trim());
                        } else if (type == DataMineConstants.INTEGER) {
                            instance.iAttr[attIndex] = (int) Float.parseFloat(
                                    lineItems[attIndex].trim());
                        } else {
                            instance.sAttr[attIndex] = lineItems[attIndex].
                                    trim();
                        }
                    }
                    if (withCategory) {
                        // Load the labels.
                        String classNameString =
                                lineItems[lineItems.length - 1].trim();
                        if (!classNameToIndexMap.containsKey(classNameString)) {
                            ++maxClassIndex;
                            classNameToIndexMap.put(classNameString,
                                    maxClassIndex);
                        }
                        instance.setCategory(classNameToIndexMap.get(
                                classNameString));
                    }
                    dset.addDataInstance(instance);
                    line = br.readLine();
                }
            } catch (NumberFormatException | IOException eInside) {
                System.err.println(eInside.getMessage());
                throw new Exception("Bad parse of line: " + line);
            }
            return dset;
        } catch (Exception e) {
            System.err.println(e.getMessage());
            throw e;
        }
    }
}