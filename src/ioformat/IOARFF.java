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
import data.representation.sparse.BOWDataSet;
import data.representation.sparse.BOWInstance;
import data.representation.util.DataMineConstants;
import ioformat.parsing.DataFeature;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Set;
import java.util.StringTokenizer;
import learning.unsupervised.Cluster;
import learning.unsupervised.ClusterConfigurationCleaner;

/**
 * A class that implements methods for data load and persistence to and from the
 * ARFF data format, as used in Weka (for cross-compatibility). The details of
 * the format can be accessed at the following link:
 * http://www.cs.waikato.ac.nz/~ml/weka/arff.html It should be noted, though,
 * that this library uses a different sparse ARFF data format, so sparse and
 * combined dense+sparse data would have to be transformed prior to external
 * usage.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class IOARFF {

    // Hashing for the nominal features to reduce the memory footprint.
    private boolean useNominalHashing = true;
    // Contains a hash for each nominal feature.
    private ArrayList<HashMap> nominalHashes = new ArrayList<>(10);
    // Contains a vocabulary for each nominal feature.
    private ArrayList<String>[] nominalVocabularies = null;

    /**
     * A utility method for quickly persisting the medoids of a cluster
     * configuration to a file in an ARFF format.
     *
     * @param clusterConfiguration Cluster[] that is the cluster configuration.
     * @param dataContext DataSet that is the data context.
     * @param outPath String that is the output path.
     * @throws Exception
     */
    public void writeMedoidsToFile(Cluster[] clusterConfiguration,
            DataSet dataContext, String outPath) throws Exception {
        if (clusterConfiguration == null || dataContext == null
                || clusterConfiguration.length == 0) {
            return;
        }
        clusterConfiguration = ClusterConfigurationCleaner.
                removeEmptyClusters(clusterConfiguration);
        DataSet medoidCollection = dataContext.cloneDefinition();
        medoidCollection.data = new ArrayList<>(clusterConfiguration.length);
        for (int i = 0; i < clusterConfiguration.length; i++) {
            medoidCollection.data.add(clusterConfiguration[i].getMedoid().
                    copyContent());
            medoidCollection.data.get(i).setCategory(i);
        }
        saveLabeledWithIdentifiers(medoidCollection, outPath, null);
    }

    /**
     * Saves labeled data that has an associated identifier DataSet.
     *
     * @param dset DataSet that is the data to persist.
     * @param outPath String that is the output path for the representational
     * data.
     * @param idPath String that is the output path for the ID data.
     * @throws IOException
     */
    public void saveLabeledWithIdentifiers(DataSet dset, String outPath,
            String idPath) throws IOException {
        if (outPath != null) {
            saveLabeled(dset, outPath);
        }
        if (idPath != null) {
            saveUnlabeled(dset.getIdentifiers(), idPath);
        }
    }

    /**
     * Saves labeled data that has an associated identifier DataSet.
     *
     * @param dset DataSet that is the data to persist.
     * @param outFile File where the representational data will be persisted.
     * @param idFile File where the ID data will be persisted.
     * @throws IOException
     */
    public void saveLabeledWithIdentifiers(DataSet dset, File outFile,
            File idFile) throws IOException {
        if (outFile != null) {
            saveLabeled(dset, outFile.getPath());
        }
        if (idFile != null) {
            saveUnlabeled(dset.getIdentifiers(), idFile.getPath());
        }
    }

    /**
     * Saves the data into an ARFF format.
     *
     * @param dset DataSet object to persist.
     * @param outPath String that is the path to the target file for storing the
     * representational data.
     * @param idPath String that is the path to the target file for storing the
     * ID data.
     * @throws IOException
     */
    public void save(DataSet dset, String outPath, String idPath)
            throws IOException {
        if (outPath != null) {
            saveUnlabeled(dset, outPath);
        }
        if (idPath != null) {
            saveUnlabeled(dset.getIdentifiers(), idPath);
        }
    }

    /**
     * Loads the data that has identifiers stored externally in a separate ARFF
     * file.
     *
     * @param inPath String that is the input path to the representational part
     * of the data.
     * @param idPath String that is the input path to the ID part of the data.
     * @return DataSet object that is the loaded data.
     * @throws IOException
     */
    public DataSet loadWithIdentifiers(String inPath, String idPath)
            throws IOException {
        if (inPath == null) {
            return null;
        }
        DataSet dset = load(inPath);
        if (idPath != null) {
            DataSet idDSet = load(idPath);
            dset.setIdentifiers(idDSet);
            for (int i = 0; i < idDSet.size(); i++) {
                dset.data.get(i).setIdentifier(idDSet.data.get(i));
            }
        }
        return dset;
    }

    /**
     * @param useNominalHashing Boolean flag indicating whether to use nominal
     * feature hashing for reduced memory footprint of handling nominal
     * features.
     */
    public void setNominalHashing(boolean useNominalHashing) {
        this.useNominalHashing = useNominalHashing;
    }

    /**
     * Loads a sparse data representation. This is one of the two supported
     * variants of the same format, where the last attribute is interpreted as
     * class attribute.
     *
     * @param inPath String that is the path to the input file.
     * @return BOWDataSet that is the sparse dataset object.
     * @throws Exception
     */
    public BOWDataSet loadSparseCategoryLast(String inPath) throws Exception {
        HashMap<String, Integer> classNameToIndexMap = new HashMap<>(100);
        int maxClassIndex = -1;
        BOWDataSet bowDSet = new BOWDataSet();
        bowDSet.data = new ArrayList<>(10000);
        // dataMode gets set to true when the header has been processed and the
        // main data load begins.
        boolean dataMode = false;
        ArrayList<String> featureNameList = new ArrayList<>(10000);
        try (BufferedReader br = new BufferedReader((new InputStreamReader(
                     new FileInputStream(new File(inPath)), "UTF-8")))) {
            String[] pair;
            String[] lineItems;
            String line = br.readLine();
            // First line is the @RELATION line with the relation name.
            try {
                // Sometimes it might not be provided.
                pair = line.split(" ");
                bowDSet.setName(pair[1]);
            } catch (Exception e) {
            }
            line = br.readLine();
            int classIndex = -1;
            int featureIndex;
            float featureValue;
            while (line != null) {
                line = line.trim();
                if (!dataMode) {
                    if (line.startsWith("@ATTRIBUTE")
                            || line.startsWith("@attribute")
                            || line.startsWith("@Attribute")) {
                        lineItems = line.split(" ");
                        ++classIndex;
                        featureNameList.add(lineItems[1]);
                    } else if (line.startsWith("@DATA")
                            || line.startsWith("@data")
                            || line.startsWith("@Data")) {
                        // Class was the last feature.
                        featureNameList.remove(featureNameList.size() - 1);
                        // Entering the data mode.
                        dataMode = true;
                        // Initialize the sparse structures.
                        ArrayList<String> vocabulary =
                                new ArrayList<>(classIndex + 1);
                        ArrayList<Float> wordFrequencies =
                                new ArrayList<>(classIndex + 1);
                        HashMap<String, Integer> vocabularyHash =
                                new HashMap<>((classIndex + 1) * 3, 500);
                        for (int index = 0; index < featureNameList.size();
                                index++) {
                            vocabulary.add(featureNameList.get(index));
                            wordFrequencies.add(0f);
                            vocabularyHash.put(featureNameList.get(index),
                                    index);
                        }
                        bowDSet.setVocabularyData(vocabulary, vocabularyHash,
                                wordFrequencies);
                    }
                } else {
                    line = line.trim();
                    line = line.substring(1, line.length() - 1);
                    if (line.equals("")) {
                        // Handle an empty line.
                        line = br.readLine();
                        BOWInstance instance = new BOWInstance(bowDSet);
                        bowDSet.data.add(instance);
                        continue;
                    }
                    lineItems = line.split(",");
                    BOWInstance instance = new BOWInstance(bowDSet);
                    for (int i = 0; i < lineItems.length - 1; i++) {
                        pair = lineItems[i].split(" ");
                        featureIndex = Integer.parseInt(pair[0]);
                        featureValue = Float.parseFloat(pair[1]);
                        instance.addWord(featureIndex, featureValue);
                    }
                    pair = lineItems[lineItems.length - 1].split(" ");
                    featureIndex = Integer.parseInt(pair[0]);
                    if (featureIndex == classIndex) {
                        // Set the data label.
                        String classNameString = pair[1];
                        if (!classNameToIndexMap.containsKey(classNameString)) {
                            ++maxClassIndex;
                            classNameToIndexMap.put(classNameString,
                                    maxClassIndex);
                        }
                        instance.setCategory(classNameToIndexMap.get(
                                classNameString));
                    } else {
                        // Missing class label.
                        featureValue = Float.parseFloat(pair[1]);
                        instance.addWord(featureIndex, featureValue);
                        instance.setCategory(0);
                    }
                    bowDSet.data.add(instance);
                }
                line = br.readLine();
            }
            for (int i = 0; i < bowDSet.data.size(); i++) {
                BOWInstance instance = (BOWInstance) (bowDSet.data.get(i));
                HashMap<Integer, Float> indexMap =
                        instance.getWordIndexesHash();
                Set<Integer> keys = indexMap.keySet();
                for (int index : keys) {
                    bowDSet.increaseFrequency(index, indexMap.get(index));
                }
            }
        } catch (IOException e) {
            throw e;
        }
        return bowDSet;
    }

    /**
     * Loads a sparse data representation. This is one of the two supported
     * variants of the same format, where the class information is given at the
     * end of each line along with the number of features. Also, this variant
     * ignores feature names and just numerates them.
     *
     * @param inPath String that is the path to the input file.
     * @return BOWDataSet that is the sparse dataset object.
     * @throws Exception
     */
    public BOWDataSet loadSparse(String inPath) throws Exception {
        HashMap<String, Integer> classNameToIndexMap = new HashMap<>(100);
        int maxClassIndex = -1;
        BOWDataSet bowDSet = new BOWDataSet();
        bowDSet.data = new ArrayList<>(10000);
        boolean dataMode = false;
        // Whether there is class information will be determined.
        boolean hasCategory = false;
        // We specify the encoding explicitly, to avoid encoding issues.
        try (BufferedReader br = new BufferedReader((new InputStreamReader(
                new FileInputStream(new File(inPath)), "UTF-8")));) {
            String[] lineItems;
            String[] pair;
            String line = br.readLine();
            // First line is the @RELATION line with the relation name.
            try {
                // Sometimes it might not be provided.
                pair = line.split(" ");
                bowDSet.setName(pair[1]);
            } catch (Exception e) {
            }
            line = br.readLine();
            int maxFeatureIndex = -1;
            int featureIndex;
            float featureValue;
            while (line != null) {
                line = line.trim();
                if (!dataMode) {
                    if (line.startsWith("@ATTRIBUTE")
                            || line.startsWith("@attribute")
                            || line.startsWith("@Attribute")) {
                        lineItems = line.split(" ");
                        if (lineItems[1].toLowerCase().equals("class")) {
                            // Labels exist in the representation.
                            hasCategory = true;
                        } else {
                            maxFeatureIndex++;
                        }
                    } else if (line.startsWith("@DATA")
                            || line.startsWith("@data")
                            || line.startsWith("@Data")) {
                        // Entering the data mode.
                        dataMode = true;
                        ArrayList<String> vocabulary =
                                new ArrayList<>(maxFeatureIndex + 1);
                        ArrayList<Float> wordFrequencies =
                                new ArrayList<>(maxFeatureIndex + 1);
                        HashMap<String, Integer> vocabularyHash =
                                new HashMap<>((maxFeatureIndex + 1) * 3, 500);
                        for (int i = 0; i < maxFeatureIndex + 1; i++) {
                            // Features are just numerated.
                            String strInt = (new Integer(i)).toString();
                            vocabulary.add(strInt);
                            wordFrequencies.add(0f);
                            vocabularyHash.put(strInt, i);
                        }
                        bowDSet.setVocabularyData(vocabulary, vocabularyHash,
                                wordFrequencies);
                    }
                } else {
                    line = line.trim();
                    line = line.substring(1, line.length() - 1);
                    if (line.equals("")) {
                        line = br.readLine();
                        BOWInstance instance = new BOWInstance(bowDSet);
                        bowDSet.data.add(instance);
                        continue;
                    }
                    lineItems = line.split(",");
                    BOWInstance instance = new BOWInstance(bowDSet);
                    if (!hasCategory) {
                        // Class information is not present.
                        try {
                            for (int i = 0; i < lineItems.length; i++) {
                                pair = lineItems[i].split(" ");
                                featureIndex = Integer.parseInt(pair[0]);
                                featureValue = Float.parseFloat(pair[1]);
                                instance.addWord(featureIndex, featureValue);
                            }
                        } catch (Exception e) {
                            System.err.println(e.getMessage());
                            throw e;
                        }
                    } else {
                        for (int i = 0; i < lineItems.length - 1; i++) {
                            pair = lineItems[i].split(" ");
                            featureIndex = Integer.parseInt(pair[0]);
                            featureValue = Float.parseFloat(pair[1]);
                            instance.addWord(featureIndex, featureValue);
                        }
                        pair = lineItems[lineItems.length - 1].split(" ");
                        String classNameString = pair[1];
                        if (!classNameToIndexMap.containsKey(classNameString)) {
                            ++maxClassIndex;
                            classNameToIndexMap.put(classNameString,
                                    maxClassIndex);
                        }
                        instance.setCategory(classNameToIndexMap.get(
                                classNameString));
                    }
                    bowDSet.data.add(instance);
                }
                line = br.readLine();
            }
            for (int i = 0; i < bowDSet.data.size(); i++) {
                BOWInstance instance = (BOWInstance) (bowDSet.data.get(i));
                HashMap<Integer, Float> indexMap =
                        instance.getWordIndexesHash();
                Set<Integer> keys = indexMap.keySet();
                for (int index : keys) {
                    bowDSet.increaseFrequency(index, indexMap.get(index));
                }
            }
        } catch (IOException e) {
            throw e;
        }
        return bowDSet;
    }

    /**
     * This method performs the data load from the specified ARFF target.
     *
     * @param inPath String that is the path to the data file in ARFF format.
     * @return DataSet object that is the loaded data.
     * @throws IOException
     */
    public DataSet load(String inPath) throws IOException {
        // We specify the encoding explicitly, to avoid encoding issues.
        BufferedReader br = new BufferedReader((new InputStreamReader(
                new FileInputStream(new File(inPath)), "UTF-8")));
        DataSet dset = new DataSet();
        try {
            ArrayList<DataFeature> features = loadFeatures(br);
            setDefinition(dset, features);
            loadRepresentation(dset, br, features);
            // Initialize feature name hashes.
            dset.makeFeatureMappings();
        } catch (IOException e) {
            throw e;
        } finally {
            try {
                br.close();
            } catch (Exception e) {
            }
            return dset;
        }
    }

    /**
     * This method loads the representation from the BufferedReader for the
     * specified list of features.
     * 
     * @param dset DataSet object to load into.
     * @param br BufferedReader used as the source.
     * @param features List of DataFeature feature specifications.
     * @throws IOException 
     */
    private void loadRepresentation(DataSet dset, BufferedReader br,
            ArrayList<DataFeature> features) throws IOException {
        String line = br.readLine();
        HashMap<String, Integer> classNameToIndexMap = new HashMap<>(100);
        int maxClassIndex = -1;
        // Initialize the vocabularies for nominal feature hashing.
        if (useNominalHashing) {
            nominalVocabularies = new ArrayList[nominalHashes.size()];
            for (int i = 0; i < nominalVocabularies.length; i++) {
                nominalVocabularies[i] = new ArrayList<>(500);
            }
        }
        while (line != null) {
            DataInstance instance = new DataInstance(dset);
            StringTokenizer st = new StringTokenizer(line, ",");
            for (int i = 0; i < features.size(); i++) {
                DataFeature feature = features.get(i);
                switch (feature.getFeatureType()) {
                    case DataMineConstants.INTEGER: {
                        if (st.hasMoreTokens()) {
                            // Fill in an integer feature.
                            instance.iAttr[feature.getFeatureIndex()] =
                                    Integer.parseInt(st.nextToken());
                        }
                        break;
                    }
                    case DataMineConstants.FLOAT: {
                        if (st.hasMoreTokens()) {
                            instance.fAttr[feature.getFeatureIndex()] =
                                    Float.parseFloat(st.nextToken());
                        }
                        break;
                    }
                    case DataMineConstants.NOMINAL: {
                        if (st.hasMoreTokens()) {
                            String nominalValue = st.nextToken();
                            // Trim the string.
                            nominalValue = nominalValue.trim();
                            // Handle quotations.
                            if (nominalValue.startsWith("'")) {
                                nominalValue = nominalValue.substring(1);
                            }
                            if (nominalValue.endsWith("'")) {
                                nominalValue = nominalValue.substring(0,
                                        nominalValue.length() - 1);
                            }
                            // Class is also written down as a nominal
                            // attribute.
                            if (!feature.getFeatureName().equals("class")) {
                                if (useNominalHashing) {
                                    int currIndex;
                                    if (!nominalHashes.get(
                                            feature.getFeatureIndex()).
                                            containsKey(nominalValue)) {
                                        nominalHashes.get(
                                                feature.getFeatureIndex()).
                                                put(nominalValue,
                                                new Integer(nominalVocabularies[
                                                feature.getFeatureIndex()].
                                                size()));
                                        nominalVocabularies[feature.
                                                getFeatureIndex()].add(
                                                nominalValue);
                                        instance.sAttr[feature.
                                                getFeatureIndex()] =
                                                nominalValue;
                                    } else {
                                        currIndex = ((Integer) (
                                                nominalHashes.get(feature.
                                                getFeatureIndex()).get(
                                                nominalValue))).intValue();
                                        instance.sAttr[feature.
                                                getFeatureIndex()] =
                                                nominalVocabularies[feature.
                                                getFeatureIndex()].get(
                                                currIndex);
                                    }
                                } else {
                                    instance.sAttr[feature.getFeatureIndex()] =
                                            nominalValue;
                                }
                            } else {
                                if (!classNameToIndexMap.containsKey(
                                        nominalValue)) {
                                    ++maxClassIndex;
                                    classNameToIndexMap.put(nominalValue,
                                            maxClassIndex);
                                }
                                instance.setCategory(classNameToIndexMap.get(
                                            nominalValue));
                            }
                        }
                        break;
                    }
                }
            }
            dset.addDataInstance(instance);
            line = br.readLine();
        }
        System.gc();
    }

    /**
     * This method sets the feature definitions that have been parsed to the
     * DataSet object.
     *
     * @param dset DataSet object to set the definitions to.
     * @param features ArrayList<DataFeature> that holds the data definition.
     */
    private void setDefinition(DataSet dset, ArrayList<DataFeature> features) {
        int numFloatFeatures = 0;
        int numIntFeatures = 0;
        int numNominalFeatures = 0;
        for (DataFeature feature : features) {
            switch (feature.getFeatureType()) {
                case DataMineConstants.INTEGER: {
                    numIntFeatures++;
                    break;
                }
                case DataMineConstants.FLOAT: {
                    numFloatFeatures++;
                    break;
                }
                case DataMineConstants.NOMINAL: {
                    if (!feature.getFeatureName().equals("class")) {
                        numNominalFeatures++;
                    }
                    break;
                }
            }
        }
        dset.fAttrNames = new String[numFloatFeatures];
        dset.iAttrNames = new String[numIntFeatures];
        dset.sAttrNames = new String[numNominalFeatures];
        for (DataFeature feature : features) {
            switch (feature.getFeatureType()) {
                case DataMineConstants.INTEGER: {
                    dset.iAttrNames[feature.getFeatureIndex()] =
                            feature.getFeatureName();
                    break;
                }
                case DataMineConstants.FLOAT: {
                    dset.fAttrNames[feature.getFeatureIndex()] =
                            feature.getFeatureName();
                    break;
                }
                case DataMineConstants.NOMINAL: {
                    if (!feature.getFeatureName().equals("class")) {
                        dset.sAttrNames[feature.getFeatureIndex()] =
                                feature.getFeatureName();
                    }
                    break;
                }
            }
        }
        if (dset.sAttrNames.length == 0) {
            dset.sAttrNames = null;
        }
        if (dset.fAttrNames.length == 0) {
            dset.fAttrNames = null;
        }
        if (dset.iAttrNames.length == 0) {
            dset.iAttrNames = null;
        }
    }

    /**
     * This method loads the feature definition from the stream.
     *
     * @param br BufferedReader that is the current input stream.
     * @return ArrayList<DataFeature> that is the data definition.
     * @throws IOException
     */
    private ArrayList<DataFeature> loadFeatures(BufferedReader br)
            throws IOException {
        int numFloatFeatures = 0;
        int numIntFeatures = 0;
        int numNominalFeatures = 0;
        ArrayList<DataFeature> features = new ArrayList<>();
        String line = br.readLine();
        // Read while not in data mode.
        while ((line != null) && (!line.trim().equalsIgnoreCase("@DATA"))) {
            line = line.trim();
            if (line.length() >= "@ATTRIBUTE".length()) {
                if (line.substring(0, "@ATTRIBUTE".length()).equalsIgnoreCase(
                        "@ATTRIBUTE")) {
                    String featureSpecString = line.substring(
                            "@ATTRIBUTE".length()).trim();
                    String featureType = featureSpecString.substring(
                            featureSpecString.lastIndexOf(' ') + 1);
                    String featureName = featureSpecString.substring(0,
                            featureSpecString.lastIndexOf(' ')).trim();
                    DataFeature feature = new DataFeature();
                    feature.setFeatureName(featureName);
                    if (featureType.equalsIgnoreCase("string")
                            || featureType.equalsIgnoreCase("nominal")) {
                        feature.setFeatureType(DataMineConstants.NOMINAL);
                        feature.setFeatureIndex(numNominalFeatures++);
                        if (useNominalHashing) {
                            nominalHashes.add(new HashMap(2000));
                        }
                    }
                    if (featureType.equalsIgnoreCase("integer")) {
                        feature.setFeatureType(DataMineConstants.INTEGER);
                        feature.setFeatureIndex(numIntFeatures++);
                    }
                    if (featureType.equalsIgnoreCase("real")
                            || featureType.equalsIgnoreCase("numeric")
                            || featureType.equalsIgnoreCase("float")) {
                        feature.setFeatureType(DataMineConstants.FLOAT);
                        feature.setFeatureIndex(numFloatFeatures++);
                    }
                    features.add(feature);
                }
            }
            line = br.readLine();
        }
        return features;
    }

    /**
     * This method saves data in an unlabeled way.
     *
     * @param dset DataSet object to save.
     * @param outPath String that is the output path to save the object to.
     * @throws IOException
     */
    public void saveUnlabeled(DataSet dset, String outPath)
            throws IOException {
        if (dset == null) {
            throw new IOException("Null data provided.");
        }
        FileUtil.createFileFromPath(outPath);
        PrintWriter pw = new PrintWriter(new OutputStreamWriter(
                new FileOutputStream(new File(outPath)), "UTF-8"));
        try {
            pw.println("@RELATION " + dset.getName());
            if (dset.hasIntAttr()) {
                for (String featureName : dset.iAttrNames) {
                    pw.println("@ATTRIBUTE " + featureName + " integer");
                }
            }
            if (dset.hasFloatAttr()) {
                for (String featureName : dset.fAttrNames) {
                    pw.println("@ATTRIBUTE " + featureName + " real");
                }
            }
            if (dset.hasNominalAttr()) {
                for (String featureName : dset.sAttrNames) {
                    pw.println("@ATTRIBUTE " + featureName + " string");
                }
            }
            pw.println("@DATA");
            for (DataInstance instance : dset.data) {
                // It will be set to true as soon as anything is printed on this
                // line.
                boolean printSeparator = false;
                if (dset.hasIntAttr()) {
                    for (int iVal : instance.iAttr) {
                        if (printSeparator) {
                            pw.print(",");
                        }
                        pw.print(iVal);
                        printSeparator = true;
                    }
                }
                if (dset.hasFloatAttr()) {
                    for (float fVal : instance.fAttr) {
                        if (printSeparator) {
                            pw.print(",");
                        }
                        pw.print(fVal);
                        printSeparator = true;
                    }
                }
                if (dset.hasNominalAttr()) {
                    for (String sVal : instance.sAttr) {
                        if (printSeparator) {
                            pw.print(",");
                        }
                        pw.print("'" + sVal + "'");
                        printSeparator = true;
                    }
                }
                pw.println();
            }
        } catch (Exception e) {
            System.err.println(e.getMessage());
            throw e;
        } finally {
            pw.flush();
            pw.close();
        }
    }

    /**
     * This method saves a sparse data representation into a file. Labels are
     * not saved by this method.
     *
     * @param bowDSet BOWDataSet object holding the sparse data representation.
     * @param outPath String that is the target output path.
     * @throws IOException
     */
    public void saveSparseUnlabeled(BOWDataSet bowDSet, String outPath)
            throws IOException {
        if (bowDSet == null) {
            throw new IOException("Null data provided.");
        }
        FileUtil.createFileFromPath(outPath);
        PrintWriter pw = new PrintWriter(new OutputStreamWriter(
                new FileOutputStream(new File(outPath)), "UTF-8"));
        try {
            ArrayList<String> vocabulary = bowDSet.getVocabulary();
            if (vocabulary == null || vocabulary.isEmpty()) {
                return;
            }
            pw.println("@RELATION " + bowDSet.getName());
            for (String word : vocabulary) {
                pw.println("@ATTRIBUTE " + word + " real");
            }
            pw.println("@DATA");
            for (int i = 0; i < bowDSet.size(); i++) {
                boolean printSeparator = false;
                pw.print("{");
                BOWInstance instance = (BOWInstance) (bowDSet.data.get(i));
                HashMap<Integer, Float> indexMap =
                        instance.getWordIndexesHash();
                Set<Integer> keys = indexMap.keySet();
                for (int index : keys) {
                    if (printSeparator) {
                        pw.print(",");
                    } else {
                        printSeparator = true;
                    }
                    pw.print(index);
                    pw.print(" ");
                    pw.print(indexMap.get(index));
                }
                pw.print("}");
                pw.println();
            }
        } catch (Exception e) {
            System.err.println(e.getMessage());
            throw e;
        } finally {
            pw.flush();
            pw.close();
        }
    }

    /**
     * This method saves a sparse data representation into a file. Labels are
     * also persisted by this method.
     *
     * @param bowDSet BOWDataSet object holding the sparse data representation.
     * @param outPath String that is the target output path.
     * @throws IOException
     */
    public void saveSparseLabeled(BOWDataSet bowDSet, String outPath)
            throws IOException {
        if (bowDSet == null) {
            throw new IOException("Null data provided.");
        }
        FileUtil.createFileFromPath(outPath);
        PrintWriter pw = new PrintWriter(new OutputStreamWriter(
                new FileOutputStream(new File(outPath)), "UTF-8"));
        try {
            ArrayList<String> vocabulary = bowDSet.getVocabulary();
            if (vocabulary == null || vocabulary.isEmpty()) {
                return;
            }
            pw.println("@RELATION " + bowDSet.getName());
            for (String word : vocabulary) {
                pw.println("@ATTRIBUTE " + word + " real");
            }
            pw.println("@ATTRIBUTE class string");
            pw.println("@DATA");
            for (int i = 0; i < bowDSet.size(); i++) {
                boolean printSeparator = false;
                pw.print("{");
                BOWInstance instance = (BOWInstance) (bowDSet.data.get(i));
                HashMap<Integer, Float> indexMap =
                        instance.getWordIndexesHash();
                Set<Integer> keys = indexMap.keySet();
                for (int index : keys) {
                    if (printSeparator) {
                        pw.print(",");
                    } else {
                        printSeparator = true;
                    }
                    pw.print(index);
                    pw.print(" ");
                    pw.print(indexMap.get(index));
                }
                if (printSeparator) {
                    pw.print(",");
                }
                pw.print(keys.size());
                pw.print(" ");
                pw.print(instance.getCategory());
                pw.print("}");
                pw.println();
            }
        } catch (Exception e) {
            System.err.println(e.getMessage());
            throw e;
        } finally {
            pw.flush();
            pw.close();
        }
    }

    /**
     * This method saves the data with the labels.
     *
     * @param dset DataSet object that is to be saved.
     * @param outPath String that is the path to save the data to.
     * @throws IOException
     */
    public void saveLabeled(DataSet dset, String outPath) throws IOException {
        if (dset == null) {
            throw new IOException("Null data provided.");
        }
        FileUtil.createFileFromPath(outPath);
        PrintWriter pw = new PrintWriter(new OutputStreamWriter(
                new FileOutputStream(new File(outPath)), "UTF-8"));
        try {
            pw.println("@RELATION " + dset.getName());
            if (dset.hasIntAttr()) {
                for (String featureName : dset.iAttrNames) {
                    pw.println("@ATTRIBUTE " + featureName + " integer");
                }
            }
            if (dset.hasFloatAttr()) {
                for (String featureName : dset.fAttrNames) {
                    pw.println("@ATTRIBUTE " + featureName + " real");
                }
            }
            if (dset.hasNominalAttr()) {
                for (String featureName : dset.sAttrNames) {
                    pw.println("@ATTRIBUTE " + featureName + " string");
                }
            }
            // Class is the last attribute.
            pw.println("@ATTRIBUTE class string");
            pw.println("@DATA");
            for (DataInstance instance : dset.data) {
                boolean printSeparator = false;
                if (dset.hasIntAttr()) {
                    for (int iVal : instance.iAttr) {
                        if (printSeparator) {
                            pw.print(",");
                        }
                        pw.print(iVal);
                        printSeparator = true;
                    }
                }
                if (dset.hasFloatAttr()) {
                    for (float fVal : instance.fAttr) {
                        if (printSeparator) {
                            pw.print(",");
                        }
                        pw.print(fVal);
                        printSeparator = true;
                    }
                }
                if (dset.hasNominalAttr()) {
                    for (String sVal : instance.sAttr) {
                        if (printSeparator) {
                            pw.print(",");
                        }
                        pw.print("'" + sVal + "'");
                        printSeparator = true;
                    }
                }
                if (printSeparator) {
                    pw.print(",");
                }
                pw.print("'" + instance.getCategory() + "'");
                pw.println();
            }
        } catch (Exception e) {
            System.err.println(e.getMessage());
            throw e;
        } finally {
            pw.flush();
            pw.close();
        }
    }
}