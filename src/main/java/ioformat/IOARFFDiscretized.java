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
import data.representation.discrete.DiscretizedDataInstance;
import data.representation.discrete.DiscretizedDataSet;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;

/**
 * A class that implements methods for data load and persistence to and from a
 * modified version of the ARFF data format. The details of the basic format can
 * be accessed at the following link:
 * http://www.cs.waikato.ac.nz/~ml/weka/arff.html This class deals with the
 * persistence of discretized data. This is discrete data that has been obtained
 * by discretizing real-valued data. Therefore, both types of information should
 * be persisted separately.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class IOARFFDiscretized {

    /**
     * The default constructor.
     */
    public IOARFFDiscretized() {
    }

    /**
     * Writes a DiscretizedDataSet object to a file. This method ignores data
     * labels.
     *
     * @param discDSet DiscretizedDataSet object to persist.
     * @param outPath String that is the path to the output file.
     * @param nonDiscreteDataPath String that is the path to where the original
     * non-discrete data is or will be persisted.
     * @throws Exception
     */
    public void saveUnlabeled(DiscretizedDataSet discDSet, String outPath,
            String nonDiscreteDataPath) throws Exception {
        saveUnlabeled(discDSet, new File(outPath), nonDiscreteDataPath);
    }

    /**
     * Writes a DiscretizedDataSet object to a file. This method ignores data
     * labels.
     *
     * @param discDSet DiscretizedDataSet object to persist.
     * @param outFile File where the data will be persisted.
     * @param nonDiscreteDataPath String that is the path to where the original
     * non-discrete data is or will be persisted.
     * @throws Exception
     */
    public void saveUnlabeled(DiscretizedDataSet discDSet, File outFile,
            String nonDiscreteDataPath) throws Exception {
        if (discDSet == null || discDSet.isEmpty()) {
            return;
        }
        PrintWriter pw = new PrintWriter(new FileWriter(outFile));
        // Get the definitions of the discrete intervals.
        int[][] intIntervalDivisions = discDSet.getIntIntervalDivisions();
        float[][] floatIntervalDivisions =
                discDSet.getFloatIntervalDivisions();
        ArrayList<String>[] nominalVocabularies =
                discDSet.getNominalVocabularies();
        try {
            // First print the discretization definition. The description is put
            // under arff comments in order not to violate the format
            // constraints, so that the data can still be loaded into Weka.
            pw.println("%@COLLECTION " + nonDiscreteDataPath);
            if (intIntervalDivisions != null) {
                pw.println("%@INT_ATTRIBUTE_NUMBER "
                        + intIntervalDivisions.length);
                for (int i = 0; i < intIntervalDivisions.length; i++) {
                    if (intIntervalDivisions[i] != null) {
                        // The number of split points.
                        pw.println("%@INT_SP_NUMBER "
                                + intIntervalDivisions[i].length);
                    } else {
                        pw.println("%@INT_SP_NUMBER 0");
                    }
                    // Print the split points.
                    pw.print("%@INT_SP_ARRAY");
                    if (intIntervalDivisions[i] != null) {
                        for (int j = 0; j < intIntervalDivisions[i].length;
                                j++) {
                            pw.print(" " + intIntervalDivisions[i][j]);
                        }
                    }
                    pw.println();
                }
            }
            if (floatIntervalDivisions != null) {
                pw.println("%@FLOAT_ATTRIBUTE_NUMBER "
                        + floatIntervalDivisions.length);
                for (int i = 0; i < floatIntervalDivisions.length; i++) {
                    if (floatIntervalDivisions[i] != null) {
                        // The number of split points.
                        pw.println("%@FLOAT_SP_NUMBER "
                                + floatIntervalDivisions[i].length);
                    } else {
                        pw.println("%@FLOAT_SP_NUMBER 0");
                    }
                    // Print the split points.
                    pw.print("%@FLOAT_SP_ARRAY");
                    if (floatIntervalDivisions[i] != null) {
                        for (int j = 0; j < floatIntervalDivisions[i].length;
                                j++) {
                            pw.print(" " + floatIntervalDivisions[i][j]);
                        }
                    }
                    pw.println();
                }
            }
            // Print the nominal vocabularies.
            if (nominalVocabularies != null) {
                nominalVocabularies = discDSet.getNominalVocabularies();
                pw.println("%@NOMINAL_ATTRIBUTE_NUMBER "
                        + nominalVocabularies.length);
                for (int i = 0; i < nominalVocabularies.length; i++) {
                    if (nominalVocabularies[i] != null) {
                        pw.println("%@VOCABULARY_SIZE "
                                + nominalVocabularies[i].size());
                    } else {
                        pw.println("%@VOCABULARY_SIZE 0");
                    }
                    pw.print("%@VOCABULARY");
                    for (int j = 0; j < nominalVocabularies[i].size(); j++) {
                        pw.print(" " + nominalVocabularies[i].get(j));
                    }
                    pw.println();
                }
            }
            // Insert ARFF attribute definition headers. All discretized values
            // are inserted as integers.
            pw.println("@RELATION Dataset");
            if (intIntervalDivisions != null) {
                for (int i = 0; i < intIntervalDivisions.length; i++) {
                    pw.println("@ATTRIBUTE Int" + i + " integer");
                }
            }
            if (floatIntervalDivisions != null) {
                for (int i = 0; i < floatIntervalDivisions.length; i++) {
                    pw.println("@ATTRIBUTE Float" + i + " integer");
                }
            }
            if (nominalVocabularies != null) {
                for (int i = 0; i < nominalVocabularies.length; i++) {
                    pw.println("@ATTRIBUTE Nominal" + i + " integer");
                }
            }
            // Write the discretized data itself.
            pw.println("@DATA");
            for (int i = 0; i < discDSet.data.size(); i++) {
                boolean first = true;
                DiscretizedDataInstance instance = discDSet.data.get(i);
                if (instance.integerIndexes != null) {
                    for (int j = 0; j < instance.integerIndexes.length; j++) {
                        if (first) {
                            pw.print(instance.integerIndexes[j]);
                            first = false;
                        } else {
                            pw.print("," + instance.integerIndexes[j]);
                        }
                    }
                }
                if (instance.floatIndexes != null) {
                    for (int j = 0; j < instance.floatIndexes.length; j++) {
                        if (first) {
                            pw.print(instance.floatIndexes[j]);
                            first = false;
                        } else {
                            pw.print("," + instance.floatIndexes[j]);
                        }
                    }
                }
                if (instance.nominalIndexes != null) {
                    for (int j = 0; j < instance.nominalIndexes.length; j++) {
                        if (first) {
                            pw.print(instance.nominalIndexes[j]);
                            first = false;
                        } else {
                            pw.print("," + instance.nominalIndexes[j]);
                        }
                    }
                }
                pw.println();
            }
        } catch (Exception e) {
            throw e;
        } finally {
            pw.close();
        }
    }

    /**
     *
     * @param discDSet
     * @param outFile
     * @param nonDiscreteDataPath
     * @throws Exception
     */
    public void saveLabeled(DiscretizedDataSet discDSet, File outFile,
            String nonDiscreteDataPath) throws Exception {
        if (discDSet == null || discDSet.isEmpty()) {
            return;
        }
        PrintWriter pw = new PrintWriter(new FileWriter(outFile));
        int[][] intIntervalDivisions = discDSet.getIntIntervalDivisions();
        float[][] floatIntervalDivisions =
                discDSet.getFloatIntervalDivisions();
        ArrayList<String>[] nominalVocabularies =
                discDSet.getNominalVocabularies();
        try {
            // First print the discretization definition. The description is put
            // under arff comments in order not to violate the format
            // constraints, so that the data can still be loaded into Weka.
            pw.println("%@COLLECTION " + nonDiscreteDataPath);
            if (intIntervalDivisions != null) {
                pw.println("%@INT_ATTRIBUTE_NUMBER "
                        + intIntervalDivisions.length);
                for (int i = 0; i < intIntervalDivisions.length; i++) {
                    if (intIntervalDivisions[i] != null) {
                        // The number of split points.
                        pw.println("%@INT_SP_NUMBER "
                                + intIntervalDivisions[i].length);
                    } else {
                        pw.println("%@INT_SP_NUMBER 0");
                    }
                    // Print the split points.
                    pw.print("%@INT_SP_ARRAY");
                    if (intIntervalDivisions[i] != null) {
                        for (int j = 0; j < intIntervalDivisions[i].length;
                                j++) {
                            pw.print(" " + intIntervalDivisions[i][j]);
                        }
                    }
                    pw.println();
                }
            }
            if (floatIntervalDivisions != null) {
                pw.println("%@FLOAT_ATTRIBUTE_NUMBER "
                        + floatIntervalDivisions.length);
                for (int i = 0; i < floatIntervalDivisions.length; i++) {
                    if (floatIntervalDivisions[i] != null) {
                        // The number of split points.
                        pw.println("%@FLOAT_SP_NUMBER "
                                + floatIntervalDivisions[i].length);
                    } else {
                        pw.println("%@FLOAT_SP_NUMBER 0");
                    }
                    // Print the split points.
                    pw.print("%@FLOAT_SP_ARRAY");
                    if (floatIntervalDivisions[i] != null) {
                        for (int j = 0; j < floatIntervalDivisions[i].length;
                                j++) {
                            pw.print(" " + floatIntervalDivisions[i][j]);
                        }
                    }
                    pw.println();
                }
            }
            // Print the nominal vocabularies.
            if (nominalVocabularies != null) {
                nominalVocabularies = discDSet.getNominalVocabularies();
                pw.println("%@NOMINAL_ATTRIBUTE_NUMBER "
                        + nominalVocabularies.length);
                for (int i = 0; i < nominalVocabularies.length; i++) {
                    if (nominalVocabularies[i] != null) {
                        pw.println("%@VOCABULARY_SIZE "
                                + nominalVocabularies[i].size());
                    } else {
                        pw.println("%@VOCABULARY_SIZE 0");
                    }
                    pw.print("%@VOCABULARY");
                    for (int j = 0; j < nominalVocabularies[i].size(); j++) {
                        pw.print(" " + nominalVocabularies[i].get(j));
                    }
                    pw.println();
                }
            }
            // First insert the ARFF feature headers.
            pw.println("@RELATION Dataset");
            if (intIntervalDivisions != null) {
                for (int i = 0; i < intIntervalDivisions.length; i++) {
                    pw.println("@ATTRIBUTE Int" + i + " integer");
                }
            }
            if (floatIntervalDivisions != null) {
                for (int i = 0; i < floatIntervalDivisions.length; i++) {
                    pw.println("@ATTRIBUTE Float" + i + " integer");
                }
            }
            if (nominalVocabularies != null) {
                for (int i = 0; i < nominalVocabularies.length; i++) {
                    pw.println("@ATTRIBUTE Nominal" + i + " integer");
                }
            }
            // The class feature, here as integer.
            pw.println("@ATTRIBUTE category integer");
            // Write the discretized feature values for all instances.
            pw.println("@DATA");
            for (int i = 0; i < discDSet.size(); i++) {
                boolean first = true;
                DiscretizedDataInstance instance = discDSet.data.get(i);
                if (instance.integerIndexes != null) {
                    for (int j = 0; j < instance.integerIndexes.length; j++) {
                        if (first) {
                            pw.print(instance.integerIndexes[j]);
                            first = false;
                        } else {
                            pw.print("," + instance.integerIndexes[j]);
                        }
                    }
                }
                if (instance.floatIndexes != null) {
                    for (int j = 0; j < instance.floatIndexes.length; j++) {
                        if (first) {
                            pw.print(instance.floatIndexes[j]);
                            first = false;
                        } else {
                            pw.print("," + instance.floatIndexes[j]);
                        }
                    }
                }
                if (instance.nominalIndexes != null) {
                    for (int j = 0; j < instance.nominalIndexes.length; j++) {
                        if (first) {
                            pw.print(instance.nominalIndexes[j]);
                            first = false;
                        } else {
                            pw.print("," + instance.nominalIndexes[j]);
                        }
                    }
                }
                if (first) {
                    pw.print(instance.getCategory());
                } else {
                    pw.print("," + instance.getCategory());
                }
                pw.println();
            }
        } catch (Exception e) {
            throw e;
        } finally {
            pw.close();
        }
    }

    /**
     * Loads the discretized dataset from a file. This method ignores the
     * labels.
     *
     * @param inPath String that is the path to the input file.
     * @param nonDiscreteDataPath String that is the path to the file containing
     * the non-discrete data that this discretized dataset is based upon.
     * @return DiscretizedDataSet object that is loaded from the file.
     * @throws Exception
     */
    public DiscretizedDataSet loadUnlabeled(String inPath,
            String nonDiscreteDataPath) throws Exception {
        return loadUnlabeled(new File(inPath), new File(nonDiscreteDataPath));
    }

    /**
     * Loads the discretized dataset from a file. This method ignores the
     * labels.
     *
     * @param inPath String that is the path to the input file.
     * @param nonDiscreteDataFile File that contains the non-discrete data that
     * this discretized dataset is based upon.
     * @return DiscretizedDataSet object that is loaded from the file.
     * @throws Exception
     */
    public DiscretizedDataSet loadUnlabeled(String inPath,
            File nonDiscreteDataFile) throws Exception {
        return loadUnlabeled(new File(inPath), nonDiscreteDataFile);
    }

    /**
     * Loads the discretized dataset from a file. This method ignores the
     * labels.
     *
     * @param inFile File that contains the target data.
     * @param nonDiscreteDataPath String that is the path to the file containing
     * the non-discrete data that this discretized dataset is based upon.
     * @return DiscretizedDataSet object that is loaded from the file.
     * @throws Exception
     */
    public DiscretizedDataSet loadUnlabeled(File inFile,
            String nonDiscreteDataPath) throws Exception {
        return loadUnlabeled(inFile, new File(nonDiscreteDataPath));
    }

    /**
     * Loads the discretized dataset from a file. This method ignores the
     * labels.
     *
     * @param inFile File that contains the target data.
     * @param nonDiscreteDataFile File that contains the non-discrete data that
     * this discretized dataset is based upon.
     * @return DiscretizedDataSet object that is loaded from the file.
     * @throws Exception
     */
    public DiscretizedDataSet loadUnlabeled(File inFile,
            File nonDiscreteDataFile) throws Exception {
        IOARFF persister = new IOARFF();
        DataSet dset = persister.load(nonDiscreteDataFile.getPath());
        return loadUnlabeled(inFile, dset);
    }

    /**
     * Loads the discretized dataset from a file. This method loads the labels.
     *
     * @param inPath String that is the path to the input file.
     * @param nonDiscreteDataPath String that is the path to the file containing
     * the non-discrete data that this discretized dataset is based upon.
     * @return DiscretizedDataSet object that is loaded from the file.
     * @throws Exception
     */
    public DiscretizedDataSet loadLabeled(String inPath,
            String nonDiscreteDataPath) throws Exception {
        return loadLabeled(new File(inPath), new File(nonDiscreteDataPath));
    }

    /**
     * Loads the discretized dataset from a file. This method loads the labels.
     *
     * @param inPath String that is the path to the input file.
     * @param nonDiscreteDataFile File that contains the non-discrete data that
     * this discretized dataset is based upon.
     * @return DiscretizedDataSet object that is loaded from the file.
     * @throws Exception
     */
    public DiscretizedDataSet loadLabeled(String inPath,
            File nonDiscreteDataFile) throws Exception {
        return loadLabeled(new File(inPath), nonDiscreteDataFile);
    }

    /**
     * Loads the discretized dataset from a file. This method loads the labels.
     *
     * @param inFile File that contains the target data.
     * @param nonDiscreteDataPath String that is the path to the file containing
     * the non-discrete data that this discretized dataset is based upon.
     * @return DiscretizedDataSet object that is loaded from the file.
     * @throws Exception
     */
    public DiscretizedDataSet loadLabeled(File inFile,
            String nonDiscreteDataPath) throws Exception {
        return loadLabeled(inFile, new File(nonDiscreteDataPath));
    }

    /**
     * Loads the discretized dataset from a file. This method loads the labels.
     *
     * @param inFile File that contains the target data.
     * @param nonDiscreteDataFile File that contains the non-discrete data that
     * this discretized dataset is based upon.
     * @return DiscretizedDataSet object that is loaded from the file.
     * @throws Exception
     */
    public DiscretizedDataSet loadLabeled(File inFile,
            File nonDiscreteDataFile) throws Exception {
        IOARFF persister = new IOARFF();
        DataSet dset = persister.load(nonDiscreteDataFile.getPath());
        return loadLabeled(inFile, dset);
    }

    /**
     * Loads the discretized dataset from a file. This method ignores the
     * labels.
     *
     * @param inFile File that contains the target data.
     * @param nonDiscreteData DataSet the contains the non-discrete data that
     * this data is based upon.
     * @return DiscretizedDataSet object that is loaded from the file.
     * @throws Exception
     */
    public DiscretizedDataSet loadUnlabeled(File inFile,
            DataSet nonDiscreteData) throws Exception {
        DiscretizedDataSet result = new DiscretizedDataSet(nonDiscreteData);
        try (BufferedReader br = new BufferedReader(new InputStreamReader(
                new FileInputStream(inFile)));) {
            // This flag is set when the read enters the part of the file where
            // the values are.
            boolean dataMode = false;
            // Variables that hold the number of discretized int, float and
            // nominal features.
            int intSize = 0;
            int floatSize = 0;
            int nominalSize;
            int intIndex = -1;
            int floatIndex = -1;
            int nominalIndex = -1;
            String[] lineItems;
            String line = newLine(br);
            String token;
            DiscretizedDataInstance instance;
            int dataIndex = -1;
            int initialVocabularySize;

            HashMap[] nominalHashes = null;
            ArrayList<String>[] nominalVocabularies = null;
            // By convention, they define [) intervals and are supposed to be
            // ordered.
            int[][] intIntervalDivisions = null;
            float[][] floatIntervalDivisions = null;

            while (line != null) {
                if (dataMode) {
                    if (!line.startsWith("%")) {
                        ++dataIndex;
                        instance = new DiscretizedDataInstance(result, true);
                        lineItems = line.split(",");
                        for (int i = 0; i < lineItems.length; i++) {
                            token = lineItems[i].trim();
                            if (i < intSize) {
                                instance.integerIndexes[i] =
                                        Integer.parseInt(token);
                            } else if (i < intSize + floatSize) {
                                // It is a float discretizer.
                                instance.floatIndexes[i - intSize] =
                                        Integer.parseInt(token);
                            } else {
                                // It is a nominal discretizer.
                                instance.nominalIndexes[i - intSize -
                                        floatSize] = Integer.parseInt(token);
                            }
                        }
                        instance.setOriginalInstance(result.
                                getOriginalData().data.get(dataIndex));
                        result.data.add(instance);
                    }
                } else {
                    // Because of how newLine(BufferedReader br) works, this is
                    // not a comment line - either data definition or basically
                    // the beginning of the data part.
                    if (line.startsWith("%")) {
                        if (line.startsWith("%@COLLECTION")) {
                            line = line.substring(12, line.length());
                            line = line.trim();
                            // If no corresponding non-discrete data was
                            // provided, try to loaded it from this path.
                            if (nonDiscreteData == null) {
                                try {
                                    IOARFF persister = new IOARFF();
                                    result.setOriginalData(
                                            persister.load(line));
                                    result.data = new ArrayList<>(
                                            result.getOriginalData().size());
                                } catch (Exception e) {
                                    System.err.println(e.getMessage());
                                }
                            }
                        } else if (line.startsWith("%@INT_ATTRIBUTE_NUMBER ")) {
                            line = line.substring(23, line.length());
                            line = line.trim();
                            intSize = Integer.parseInt(line);
                            if (intSize > 0) {
                                intIntervalDivisions = new int[intSize][];
                            }
                        } else if (line.startsWith(
                                "%@FLOAT_ATTRIBUTE_NUMBER ")) {
                            line = line.substring(25, line.length());
                            line = line.trim();
                            floatSize = Integer.parseInt(line);
                            if (floatSize > 0) {
                                floatIntervalDivisions = new float[floatSize][];
                            }
                        } else if (line.startsWith(
                                "%@NOMINAL_ATTRIBUTE_NUMBER ")) {
                            line = line.substring(27, line.length());
                            line = line.trim();
                            nominalSize = Integer.parseInt(line);
                            if (nominalSize > 0) {
                                nominalHashes = new HashMap[nominalSize];
                            }
                        } else if (line.startsWith("%@INT_SP_NUMBER ")) {
                            line = line.substring(16, line.length());
                            line = line.trim();
                            ++intIndex;
                            intIntervalDivisions[intIndex] =
                                    new int[Integer.parseInt(line)];
                        } else if (line.startsWith("%@FLOAT_SP_NUMBER ")) {
                            line = line.substring(18, line.length());
                            line = line.trim();
                            ++floatIndex;
                            floatIntervalDivisions[floatIndex] =
                                    new float[Integer.parseInt(line)];
                        } else if (line.startsWith("%@VOCABULARY_SIZE ")) {
                            line = line.substring(18, line.length());
                            line = line.trim();
                            ++nominalIndex;
                            initialVocabularySize = Integer.parseInt(line);
                            nominalHashes[nominalIndex] = new HashMap<>(
                                    2 * initialVocabularySize);
                            nominalVocabularies[nominalIndex] =
                                    new ArrayList<>(initialVocabularySize);
                        } else if (line.startsWith("%@INT_SP_ARRAY ")) {
                            line = line.substring(15, line.length());
                            line = line.trim();
                            lineItems = line.split(" ");
                            for (int i = 0; i < lineItems.length; i++) {
                                intIntervalDivisions[intIndex][i] =
                                        Integer.parseInt(lineItems[i]);
                            }
                        } else if (line.startsWith("%@FLOAT_SP_ARRAY ")) {
                            line = line.substring(17, line.length());
                            line = line.trim();
                            lineItems = line.split(" ");
                            for (int i = 0; i < lineItems.length; i++) {
                                floatIntervalDivisions[floatIndex][i] =
                                        Float.parseFloat(lineItems[i]);
                            }
                        } else if (line.startsWith("%@VOCABULARY ")) {
                            line = line.substring(13, line.length());
                            line = line.trim();
                            lineItems = line.split(" ");
                            for (int i = 0; i < lineItems.length; i++) {
                                nominalVocabularies[nominalIndex].add(
                                        lineItems[i]);
                                nominalHashes[nominalIndex].put(lineItems[i],
                                        new Integer(i));
                            }
                        } else {
                        }
                    } else {
                        // The assumption is that first all the integer
                        // discretizations are given, then the float
                        // discretizations and then the nominal discretizations.
                        if (line.startsWith("@RELATION")) {
                            // Do nothing (might be changed to read in a
                            // separate relation name).
                        } else if (line.startsWith("@ATTRIBUTE")) {
                            // All the definition data is loaded from the
                            // comments above, so there is no need to go though
                            // this formal definition here again.
                        } else if (line.startsWith("@DATA")) {
                            dataMode = true;
                            if (result.data == null) {
                                result.data = new ArrayList<>(1000);
                            }
                        }
                    }
                }
                line = newLine(br);
            }
            result.setIntIntervalDivisions(intIntervalDivisions);
            result.setFloatIntervalDivisions(floatIntervalDivisions);
            result.setNominalHashes(nominalHashes);
            result.setNominalVocabularies(nominalVocabularies);
        } catch (Exception e) {
            System.err.println(e.getMessage());
            throw e;
        }
        return result;
    }

    /**
     * Reads a new line from the stream while ignoring ARFF comments.
     *
     * @param br BufferedReader input stream.
     * @return String that is the next line in the file, ignoring ARFF comments.
     * @throws Exception
     */
    private String newLine(BufferedReader br) throws Exception {
        String line = br.readLine();
        while (line != null && line.startsWith("%") && !line.startsWith("%@")) {
            line = br.readLine();
        }
        return line;
    }

    /**
     * Loads the discretized dataset from a file. This method ignores the
     * labels.
     *
     * @param inFile File that contains the target data.
     * @param nonDiscreteData DataSet the contains the non-discrete data that
     * this data is based upon.
     * @return DiscretizedDataSet object that is loaded from the file.
     * @throws Exception
     */
    public DiscretizedDataSet loadLabeled(File inFile,
            DataSet nonDiscreteData) throws Exception {
        DiscretizedDataSet result = new DiscretizedDataSet(nonDiscreteData);
        try (BufferedReader br = new BufferedReader(new InputStreamReader(
                     new FileInputStream(inFile)))) {
            // This flag is set when the read enters the part of the file where
            // the values are.
            boolean dataMode = false;
            // Variables that hold the number of discretized int, float and
            // nominal features.
            int intSize = 0;
            int floatSize = 0;
            int nominalSize = 0;
            int intIndex = -1;
            int floatIndex = -1;
            int nominalIndex = -1;
            String[] lineItems;
            String line = newLine(br);
            String token;
            int dataIndex = -1;
            int initialVocabularySize;
            HashMap<String, Integer> classNameToIndexMap = new HashMap<>(100);
            int maxClassIndex = -1;

            HashMap[] nominalHashes = null;
            ArrayList<String>[] nominalVocabularies = null;
            // By convention, they define [) intervals and are supposed to be
            // ordered.
            int[][] intIntervalDivisions = null;
            float[][] floatIntervalDivisions = null;

            while (line != null) {
                if (dataMode) {
                    if (!line.startsWith("%")) {
                        ++dataIndex;
                        DiscretizedDataInstance instance =
                                new DiscretizedDataInstance(result, true);
                        lineItems = line.split(",");
                        for (int i = 0; i < lineItems.length; i++) {
                            token = lineItems[i].trim();
                            if (i < intSize) {
                                instance.integerIndexes[i] =
                                        Integer.parseInt(token);
                            } else if (i < intSize + floatSize) {
                                // It is a float discretizer.
                                instance.floatIndexes[i - intSize] =
                                        Integer.parseInt(token);
                            } else if (i < intSize + floatSize + nominalSize) {
                                // It is a nominal discretizer.
                                instance.nominalIndexes[i - intSize -
                                        floatSize] = Integer.parseInt(token);
                            } else {
                                // Class information is at the end.
                                if (!classNameToIndexMap.containsKey(token)) {
                                    ++maxClassIndex;
                                    classNameToIndexMap.put(token,
                                            maxClassIndex);
                                }
                                instance.setCategory(classNameToIndexMap.get(
                                        token));
                            }
                        }
                        instance.setOriginalInstance(result.
                                getOriginalData().data.get(dataIndex));
                        result.data.add(instance);
                    }
                } else {
                    // Because of how newLine(BufferedReader br) works, this is
                    // not a comment line - either data definition or basically
                    // the beginning of the data part.
                    if (line.startsWith("%")) {
                        if (line.startsWith("%@COLLECTION")) {
                            line = line.substring(12, line.length());
                            line = line.trim();
                            // If no corresponding non-discrete data was
                            // provided, try to loaded it from this path.
                            if (nonDiscreteData == null) {
                                try {
                                    IOARFF persister = new IOARFF();
                                    result.setOriginalData(
                                            persister.load(line));
                                    result.data = new ArrayList<>(
                                            result.getOriginalData().size());
                                } catch (Exception e) {
                                }
                            }
                        } else if (line.startsWith("%@INT_ATTRIBUTE_NUMBER ")) {
                            line = line.substring(23, line.length());
                            line = line.trim();
                            intSize = Integer.parseInt(line);
                            if (intSize > 0) {
                                intIntervalDivisions = new int[intSize][];
                            }
                            result.setIntIntervalDivisions(
                                    intIntervalDivisions);
                        } else if (line.startsWith(
                                "%@FLOAT_ATTRIBUTE_NUMBER ")) {
                            line = line.substring(25, line.length());
                            line = line.trim();
                            floatSize = Integer.parseInt(line);
                            if (floatSize > 0) {
                                floatIntervalDivisions = new float[floatSize][];
                            }
                            result.setFloatIntervalDivisions(
                                    floatIntervalDivisions);
                        } else if (line.startsWith(
                                "%@NOMINAL_ATTRIBUTE_NUMBER ")) {
                            line = line.substring(27, line.length());
                            line = line.trim();
                            nominalSize = Integer.parseInt(line);
                            if (nominalSize > 0) {
                                nominalHashes = new HashMap[nominalSize];
                                nominalVocabularies =
                                        new ArrayList[nominalSize];
                            }
                            result.setNominalHashes(nominalHashes);
                            result.setNominalVocabularies(nominalVocabularies);
                        } else if (line.startsWith("%@INT_SP_NUMBER ")) {
                            line = line.substring(16, line.length());
                            line = line.trim();
                            ++intIndex;
                            intIntervalDivisions[intIndex] =
                                    new int[Integer.parseInt(line)];
                        } else if (line.startsWith("%@FLOAT_SP_NUMBER ")) {
                            line = line.substring(18, line.length());
                            line = line.trim();
                            ++floatIndex;
                            floatIntervalDivisions[floatIndex] =
                                    new float[Integer.parseInt(line)];
                        } else if (line.startsWith("%@VOCABULARY_SIZE ")) {
                            line = line.substring(18, line.length());
                            line = line.trim();
                            ++nominalIndex;
                            initialVocabularySize = Integer.parseInt(line);
                            nominalHashes[nominalIndex] =
                                    new HashMap<>(2 * initialVocabularySize);
                            nominalVocabularies[nominalIndex] =
                                    new ArrayList<>(initialVocabularySize);
                        } else if (line.startsWith("%@INT_SP_ARRAY ")) {
                            line = line.substring(15, line.length());
                            line = line.trim();
                            lineItems = line.split(" ");
                            for (int i = 0; i < lineItems.length; i++) {
                                intIntervalDivisions[intIndex][i] =
                                        Integer.parseInt(lineItems[i]);
                            }
                        } else if (line.startsWith("%@FLOAT_SP_ARRAY ")) {
                            line = line.substring(17, line.length());
                            line = line.trim();
                            lineItems = line.split(" ");
                            for (int i = 0; i < lineItems.length; i++) {
                                floatIntervalDivisions[floatIndex][i] =
                                        Float.parseFloat(lineItems[i]);
                            }
                        } else if (line.startsWith("%@VOCABULARY ")) {
                            line = line.substring(13, line.length());
                            line = line.trim();
                            lineItems = line.split(" ");
                            for (int i = 0; i < lineItems.length; i++) {
                                nominalVocabularies[nominalIndex].add(
                                        lineItems[i]);
                                nominalHashes[nominalIndex].put(lineItems[i],
                                        new Integer(i));
                            }
                        } else {
                        }
                    } else {
                        // The assumption is that first all the integer
                        // discretizations are given, then the float
                        // discretizations and then the nominal discretizations.
                        if (line.startsWith("@RELATION")) {
                            // Do nothing (might be changed to read in a
                            // separate relation name).
                        } else if (line.startsWith("@ATTRIBUTE")) {
                            // All the definition data is loaded from the
                            // comments above, so there is no need to go though
                            // this formal definition here again.
                        } else if (line.startsWith("@DATA")) {
                            dataMode = true;
                            if (result.data == null) {
                                result.data = new ArrayList<>(1000);
                            }
                        }
                    }
                }
                line = newLine(br);
            }
        } catch (Exception e) {
            throw e;
        }
        return result;
    }
}