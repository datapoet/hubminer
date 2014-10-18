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

import data.representation.DataInstance;
import data.representation.DataSet;
import data.representation.util.DataMineConstants;
import ioformat.IOARFF;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import learning.supervised.Category;
import util.CommandLineParser;

/**
 * This class implements a data load for data which has filepaths in the
 * representation that are to be transformed into a class label. The class
 * assumes a one level deep directory structure and each directory corresponds
 * to a class and the directory name is the class name. This class can also
 * perform selection on the fly and only select certain categories.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class DirectoryCategorizedDataImporter {

    private File parentDirectory = null;
    private boolean relative = false;
    private DataSet loadedData;
    private ArrayList<String> categoryNames;
    private HashMap<String, Integer> categoryHash;
    private ArrayList<Category> categories;
    private boolean selectionMode = false;

    /**
     * The default constructor.
     */
    public DirectoryCategorizedDataImporter() {
    }

    /**
     * Initialization.
     *
     * @param parentDirectory File that is the parent directory to which the
     * paths in the loaded files are relative.
     */
    public DirectoryCategorizedDataImporter(File parentDirectory) {
        if (parentDirectory != null) {
            this.parentDirectory = parentDirectory;
            relative = true;
        }
    }

    /**
     * Select certain categories among the data loaded by this object.
     *
     * @param selectedCatNames ArrayList<String> of permissable category names.
     * @param selectedNamesHash HashMap<String,Integer> that maps the
     * permissable category names to their integer indexes.
     * @param copyWhenSelecting Boolean flag controlling whether to copy the
     * instances when selecting or to only copy the reference.
     * @return
     * @throws Exception
     */
    public DirectoryCategorizedDataImporter chooseCategories(
            ArrayList<String> selectedCatNames,
            HashMap<String, Integer> selectedNamesHash,
            boolean copyWhenSelecting) throws Exception {
        // This new object will hold the selected result.
        DirectoryCategorizedDataImporter result =
                new DirectoryCategorizedDataImporter();
        result.categoryNames = selectedCatNames;
        result.categoryHash = selectedNamesHash;
        result.relative = relative;
        result.loadedData = new DataSet();
        result.loadedData.fAttrNames = loadedData.fAttrNames;
        result.loadedData.iAttrNames = loadedData.iAttrNames;
        result.loadedData.sAttrNames = loadedData.sAttrNames;
        int size = loadedData.size();
        int basicSize = Math.max(50, size / 6);
        result.loadedData.data = new ArrayList<>(basicSize);
        if (loadedData.getIdentifiers() != null) {
            result.loadedData.setIdentifiers(new DataSet());
            DataSet allIdentifiers = loadedData.getIdentifiers();
            DataSet selectedIdentifiers = result.loadedData.getIdentifiers();
            selectedIdentifiers.fAttrNames = allIdentifiers.fAttrNames;
            selectedIdentifiers.iAttrNames = allIdentifiers.iAttrNames;
            selectedIdentifiers.sAttrNames = allIdentifiers.sAttrNames;
            selectedIdentifiers.data = new ArrayList<>(basicSize);
        }
        for (int i = 0; i < loadedData.size(); i++) {
            DataInstance instance = loadedData.data.get(i);
            if (selectedNamesHash.containsKey(
                    categoryNames.get(instance.getCategory()))) {
                if (copyWhenSelecting) {
                    instance = instance.copy();
                }
                result.loadedData.addDataInstance(instance);
                instance.embedInDataset(loadedData);
                instance.setCategory(selectedNamesHash.get(
                        categoryNames.get(instance.getCategory())));
                if (instance.getIdentifier() != null) {
                    instance = instance.getIdentifier();
                    result.loadedData.getIdentifiers().
                            addDataInstance(instance);
                    instance.embedInDataset(result.loadedData.getIdentifiers());
                }
            }
        }
        return result;
    }

    /**
     * This method performs the data import when the class of the instances is
     * determined by the paths in one of their attributes.
     *
     * @return
     */
    public void importRepresentation(File inFile) throws Exception {
        importRepresentation(inFile, "relative_path");
    }

    /**
     * This method performs the data import when the class of the instances is
     * determined by the paths in one of their attributes.
     *
     * @param inFile File that is the ARFF input.
     * @param pathAttName String that is the name of the feature that holds the
     * relative path data that will be used to derive class affiliation.
     * @throws Exception
     */
    public void importRepresentation(File inFile, String pathAttName)
            throws Exception {
        IOARFF persister = new IOARFF();
        System.out.println("Importing data from: " + inFile.getPath());
        loadedData = persister.load(inFile.getPath());
        int[] typeAndIndex = loadedData.getTypeAndIndexForAttrName(pathAttName);
        if (typeAndIndex[0] == -1 || typeAndIndex[1] == -1) {
            throw new Exception("Attribute with the specified name does not"
                    + "exist.");
        }
        if (typeAndIndex[0] != DataMineConstants.NOMINAL) {
            throw new Exception("Attribute with such a bane is not a "
                    + "string-holding attribute");
        }
        int index = typeAndIndex[1];
        File fileForClassChecks;
        String extractedClassName;
        categoryNames = new ArrayList<>(100);
        categories = new ArrayList<>(100);
        categoryHash = new HashMap<>(500, 500);
        int numCategories = 0;
        int catIndex;
        if (loadedData.size() > 0) {
            DataInstance instance;
            for (int i = 0; i < loadedData.size(); i++) {
                instance = loadedData.data.get(i);
                if (relative) {
                    fileForClassChecks = new File(parentDirectory,
                            instance.sAttr[index]);
                } else {
                    fileForClassChecks = new File(instance.sAttr[index]);
                }
                extractedClassName = fileForClassChecks.getParentFile().
                        getName();
                if (!categoryHash.containsKey(extractedClassName)) {
                    categoryHash.put(extractedClassName, numCategories++);
                    categoryNames.add(extractedClassName);
                    Category newCat = new Category(extractedClassName,
                            loadedData);
                    newCat.indexes = new ArrayList<>(2000);
                    categories.add(newCat);
                }
                catIndex = categoryHash.get(extractedClassName);
                instance.setCategory(catIndex);
                categories.get(catIndex).addInstance(i);
            }
        }
    }

    /**
     * @return DataSet that has been loaded.
     */
    public DataSet getLoadedData() {
        return loadedData;
    }

    /**
     * @return Integer that is the number of categories in the data.
     */
    public int getNumCategories() {
        if (categoryNames == null) {
            return 0;
        } else {
            return categoryNames.size();
        }
    }

    /**
     * @return ArrayList<String> of category names.
     */
    public ArrayList<String> getCategoryNames() {
        return categoryNames;
    }

    /**
     * @return ArrayList<Category> of categories.
     */
    public ArrayList<Category> getCategories() {
        return categories;
    }

    /**
     * The main body of the script.
     *
     * @param args Command line parameters, as specified.
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        CommandLineParser clp = new CommandLineParser(true);
        clp.addParam("-inDataFile", "Path to the input file.",
                CommandLineParser.STRING, true, false);
        clp.addParam("-parentDir", "Path to the parent directory for the "
                + "relative paths.", CommandLineParser.STRING, true, false);
        clp.addParam("-outFile", "Path to where the output is to be saved.",
                CommandLineParser.STRING, true, false);
        clp.addParam("-outClassList", "Path to the out class list file.",
                CommandLineParser.STRING, true, false);
        clp.addParam("-selectionMode", "Set whether used in selection mode.",
                CommandLineParser.BOOLEAN, true, false);
        clp.parseLine(args);
        File inFile = new File((String) clp.getParamValues("-inDataFile").
                get(0));
        File parentDir = new File((String) clp.getParamValues("-parentDir").
                get(0));
        File outFile = new File((String) clp.getParamValues("-outFile").get(0));
        File outClassListFile = new File((String) clp.getParamValues(
                "-outClassList").get(0));
        boolean selectionMode = (Boolean) clp.getParamValues(
                "-selectionMode").get(0);
        DirectoryCategorizedDataImporter importer =
                new DirectoryCategorizedDataImporter(parentDir);
        importer.importRepresentation(inFile);
        importer.selectionMode = selectionMode;
        DirectoryCategorizedDataImporter resultingImport = importer;
        if (importer.selectionMode) {
            BufferedReader br = new BufferedReader(new InputStreamReader(
                    new FileInputStream(outClassListFile)));
            ArrayList<String> catNames;
            HashMap<String, Integer> namesHash;
            try {
                String line = br.readLine();
                int numCats = Integer.parseInt(line);
                catNames = new ArrayList(numCats);
                namesHash = new HashMap(200);
                for (int i = 0; i < numCats; i++) {
                    line = br.readLine();
                    line = line.trim();
                    catNames.add(line);
                    namesHash.put(line, i);
                }
                resultingImport = importer.chooseCategories(catNames,
                        namesHash, false);
            } catch (Exception e) {
                System.err.println(e.getMessage());
            } finally {
                br.close();
            }
        }
        DataSet dset = resultingImport.getLoadedData();
        IOARFF persister = new IOARFF();
        persister.saveLabeledWithIdentifiers(dset, outFile, null);
        if (!importer.selectionMode) {
            PrintWriter pw = new PrintWriter(new FileWriter(outClassListFile));
            ArrayList<String> catNames = importer.getCategoryNames();
            try {
                pw.println(catNames.size());
                for (String s : catNames) {
                    pw.println(s);
                }
            } catch (Exception e) {
                throw e;
            } finally {
                pw.close();
            }
        }
    }
}
