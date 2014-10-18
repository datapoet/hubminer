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
package data.representation.sparse;

import data.representation.DataSet;
import data.representation.DataInstance;
import java.util.HashMap;
import java.io.PrintWriter;
import java.io.FileWriter;
import java.io.File;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.FileInputStream;
import java.util.ArrayList;
import ioformat.FileUtil;
import java.util.Random;
import java.util.Set;

/**
 * This class implements a document corpus data holder. DataInstance objects are
 * all sparse bag-of-words representations.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class BOWDataSet extends DataSet {

    public static final int DEFAULT_INIT_WORDS = 20000;
    // The number of words in the vocabulary.
    private int numWords = -1;
    // Map of words to their indexes in the vocabulary.
    private HashMap<String, Integer> vocabularyHash;
    // Vocabulary of all the represented words.
    private ArrayList<String> vocabulary = null;
    // Word frequencies.
    private ArrayList<Float> wordFrequencies = null;

    @Override
    public DataSet copy() throws Exception {
        BOWDataSet dsetCopy = cloneDefinition();
        dsetCopy.data = new ArrayList<>(data.size());
        for (int i = 0; i < size(); i++) {
            dsetCopy.data.add(data.get(i).copyContent());
        }
        return dsetCopy;
    }

    /**
     * Normalize the data so as to make the weights of words in each instance
     * sum up to 1.
     */
    public void normalizeAsDistributions() {
        for (DataInstance instance : data) {
            float totalWeight = ((BOWInstance) instance).getDocumentLength();
            if (totalWeight > 0) {
                ((BOWInstance) instance).multiplyByScalar(1f / totalWeight);
            }
        }
    }

    @Override
    public BOWDataSet getSubsample(int[] indexes) {
        if (indexes == null) {
            return new BOWDataSet();
        }
        BOWDataSet result = cloneDefinition();
        result.data = new ArrayList<>(indexes.length);
        if (identifiers != null) {
            result.identifiers = identifiers.cloneDefinition();
            result.identifiers.data = new ArrayList<>(indexes.length);
        }
        for (int i = 0; i < indexes.length; i++) {
            result.addDataInstance(data.get(indexes[i]));
            if (identifiers != null) {
                result.identifiers.addDataInstance(
                        identifiers.data.get(indexes[i]));
            }
        }
        return result;
    }

    @Override
    public BOWDataSet getSubsample(ArrayList<Integer> indexes) {
        if (indexes == null) {
            return new BOWDataSet();
        }
        BOWDataSet result = cloneDefinition();
        result.data = new ArrayList<>(indexes.size());
        if (identifiers != null) {
            result.identifiers = identifiers.cloneDefinition();
            result.identifiers.data = new ArrayList<>(indexes.size());
        }
        for (int i = 0; i < indexes.size(); i++) {
            result.addDataInstance(data.get(indexes.get(i)));
            if (identifiers != null) {
                result.identifiers.addDataInstance(
                        identifiers.data.get(indexes.get(i)));
            }
        }
        return result;
    }

    /**
     * @return The total number of different words.
     */
    public int getNumDifferentWords() {
        if (vocabulary == null) {
            return 0;
        } else {
            return vocabulary.size();
        }
    }

    /**
     * This method eliminates the empty documents from the corpus.
     */
    public void eliminateEmptyInstances() {
        ArrayList<DataInstance> nonEmptyBOWInstances = new ArrayList<>(size());
        for (int i = 0; i < size(); i++) {
            BOWInstance instance = (BOWInstance) (data.get(i));
            if (instance != null && !instance.isEmpty()) {
                nonEmptyBOWInstances.add(instance);
            }
        }
        data = nonEmptyBOWInstances;
    }

    @Override
    public BOWDataSet cloneDefinition() {
        BOWDataSet clone = new BOWDataSet();
        if (vocabulary != null) {
            clone.vocabulary = (ArrayList<String>) vocabulary.clone();
        }
        if (vocabularyHash != null) {
            clone.vocabularyHash = (HashMap) vocabularyHash.clone();
        }
        if (wordFrequencies != null) {
            clone.wordFrequencies = (ArrayList<Float>) wordFrequencies.clone();
        }
        if (fAttrNames != null) {
            clone.fAttrNames = (String[]) fAttrNames.clone();
        }
        if (iAttrNames != null) {
            clone.iAttrNames = (String[]) fAttrNames.clone();
        }
        if (sAttrNames != null) {
            clone.sAttrNames = (String[]) fAttrNames.clone();
        }
        return clone;
    }

    /**
     *
     */
    public BOWDataSet() {
        super();
        vocabularyHash = new HashMap<>(DEFAULT_INIT_WORDS);
        wordFrequencies = new ArrayList<>(DEFAULT_INIT_WORDS);
        vocabulary = new ArrayList<>(DEFAULT_INIT_WORDS);
    }

    /**
     * @param hashSize Integer that is the initial hash size.
     */
    public BOWDataSet(int hashSize) {
        super();
        vocabularyHash = new HashMap<>(hashSize);
        wordFrequencies = new ArrayList<>(hashSize);
        vocabulary = new ArrayList<>(hashSize);
    }

    /**
     * @param integerAttrNames String[] representing integer feature names.
     * @param floatAttrNames String[] representing float feature names.
     * @param nominalAttrNames String[] representing nominal feature names.
     * @param hashSize Initial hash size.
     */
    public BOWDataSet(String[] integerAttrNames,
            String[] floatAttrNames, String[] nominalAttrNames, int hashSize) {
        super(integerAttrNames, floatAttrNames, nominalAttrNames);
        vocabularyHash = new HashMap<>(hashSize);
        wordFrequencies = new ArrayList<>(DEFAULT_INIT_WORDS);
        vocabulary = new ArrayList<>(DEFAULT_INIT_WORDS);
    }

    @Override
    public void addGaussianNoiseToNormalizedCollection(float pMutate,
            float stDev) {
        BOWInstance instance;
        Random randa = new Random();
        float choice;
        float value;
        for (int i = 0; i < size(); i++) {
            instance = (BOWInstance) data.get(i);
            HashMap<Integer, Float> indexMap =
                    instance.getWordIndexesHash();
            Set<Integer> keys = indexMap.keySet();
            for (int index : keys) {
                choice = randa.nextFloat();
                if (choice < pMutate) {
                    value = indexMap.get(index);
                    value = (float) Math.max(
                            value + stDev * randa.nextGaussian(), 0);
                    indexMap.put(index, value);
                }
            }
        }
    }

    /**
     * @return ArrayList<String> that is the current corpus word vocabulary.
     */
    public ArrayList<String> getVocabulary() {
        return vocabulary;
    }

    /**
     * @param vocabulary ArrayList<String> that is the current corpus word
     * vocabulary.
     * @param vocabularyHash HashMap<String, Integer> that maps words to index
     * values.
     * @param wordFrequencies ArrayList<Float> mapping the word frequencies.
     */
    public void setVocabularyData(
            ArrayList<String> vocabulary,
            HashMap<String, Integer> vocabularyHash,
            ArrayList<Float> wordFrequencies) {
        this.vocabulary = vocabulary;
        this.vocabularyHash = vocabularyHash;
        this.wordFrequencies = wordFrequencies;
    }

    /**
     * Sets all internal word tracking data structures to null.
     */
    public void releaseVocabulary() {
        vocabularyHash = null;
        vocabulary = null;
        wordFrequencies = null;
        numWords = 0;
    }

    /**
     * Initialize the vocabulary and related data structures.
     *
     * @param hashSize Initial hash size.
     */
    public void initVocabulary(int hashSize) {
        vocabularyHash = new HashMap(hashSize);
        vocabulary = new ArrayList<>(DEFAULT_INIT_WORDS);
        wordFrequencies = new ArrayList<>(DEFAULT_INIT_WORDS);
        numWords = 0;
    }

    /**
     * @param word String that is the word to get the index for.
     * @return Index of the word in the vocabulary or -1 if it doesn't exist.
     */
    public int getIndexForWord(String word) {
        if (!vocabularyHash.containsKey(word)) {
            return -1;
        } else {
            return vocabularyHash.get(word);
        }
    }

    /**
     * Inserts the word if it doesn't exist in the vocabulary and returns its
     * index. If it already exists, just returns the existing index.
     *
     * @param word String that is the word to insert.
     * @return The index of the word in the vocabulary.
     */
    public int insertWord(String word) {
        if (!vocabularyHash.containsKey(word)) {
            vocabularyHash.put(word, new Integer(++numWords));
            wordFrequencies.add(new Float(1));
            vocabulary.add(word);
            return numWords;
        } else {
            int index = ((Integer) vocabularyHash.get(word)).intValue();
            wordFrequencies.set(index,
                    new Float(wordFrequencies.get(index).floatValue() + 1));
            return index;
        }
    }

    /**
     * Inserts the word if it doesn't exist in the vocabulary and returns its
     * index. If it already exists, just returns the existing index. The
     * frequency is updated based on the freq parameter.
     *
     * @param word String that is the word to insert.
     * @param freq Float value that is the word's frequency to insert.
     * @return The index of the word in the vocabulary.
     */
    public int insertWord(String word, float freq) {
        if (!vocabularyHash.containsKey(word)) {
            vocabularyHash.put(word, new Integer(++numWords));
            wordFrequencies.add(freq);
            vocabulary.add(word);
            return numWords;
        } else {
            int index = ((Integer) vocabularyHash.get(word)).intValue();
            wordFrequencies.set(index,
                    new Float(wordFrequencies.get(index).floatValue() + freq));
            return index;
        }
    }

    /**
     * Increases the frequency of a word in the corpus.
     *
     * @param index Index of the word to increase the frequency of.
     * @param freq Float value that is the frequency increment.
     */
    public void increaseFrequency(int index, float freq) {
        wordFrequencies.set(index,
                new Float(wordFrequencies.get(index).floatValue() + freq));
    }

    /**
     * Prints the sparse DataSet to a sparse matrix file that has in each line
     * the index of the instance, the index of the word and the frequency,
     * separated by whitespace.
     *
     * @param path String that is the file path.
     * @throws Exception
     */
    public void printToSparseMatrixFile(String path) throws Exception {
        File outFile = new File(path);
        FileUtil.createFileFromPath(path);
        PrintWriter pw = new PrintWriter(new FileWriter(outFile));
        BOWInstance instance;
        try {
            for (int i = 0; i < size(); i++) {
                instance = (BOWInstance) (data.get(i));
                if (instance == null || instance.isEmpty()) {
                    continue;
                }
                HashMap<Integer, Float> indexMap =
                        instance.getWordIndexesHash();
                Set<Integer> keys = indexMap.keySet();
                if (keys == null || keys.isEmpty()) {
                    continue;
                }
                for (int wordIndex : keys) {
                    pw.println(i + " " + wordIndex + " "
                            + indexMap.get(wordIndex));
                }
            }
        } catch (Exception e) {
            System.err.println(e.getMessage());
            System.err.println("For file: " + path);
        } finally {
            pw.close();
        }
    }

    /**
     * Loads the sparse corpus from the sparse matrix file that has in each line
     * the index of the instance, the index of the word and the frequency,
     * separated by whitespace. This load overwrites any data previously in this
     * corpus. Additionally, no words are available in this load, so they are
     * generated from the integer indexes.
     *
     * @param path String that is the file path.
     * @param offset Integer that is the instance index offset, the difference
     * in indexing protocols.
     * @param featOffset Integer that is the feature index offset, the
     * difference in indexing protocols.
     * @throws Exception
     */
    public void loadFromSparseMatrixFile(String path, int offset,
            int featOffset) throws Exception {
        File inFile = new File(path);
        BufferedReader br = new BufferedReader(
                new InputStreamReader(new FileInputStream(inFile)));
        // Initialization.
        data = new ArrayList<>(5000);
        int index;
        int featureIndex;
        int maxFeatureIndex = 0;
        String[] lineItems;
        String s = br.readLine();
        while (s != null) {
            s = s.trim();
            lineItems = s.split(" +");
            index = (int) Float.parseFloat(lineItems[0]);
            if (index - offset >= data.size()) {
                int tempMaxIndex = data.size() - 1;
                while (tempMaxIndex++ < index - offset) {
                    BOWInstance newInstance = new BOWInstance(this);
                    data.add(newInstance);
                }
            }
            featureIndex = (int) Float.parseFloat(lineItems[1]);
            try {
                BOWInstance instance = (BOWInstance) (data.get(index - offset));
                instance.addWord(featureIndex - featOffset,
                        Float.parseFloat(lineItems[2]));
            } catch (Exception e) {
                System.err.println(e.getMessage());
                System.err.println("InstanceIndex: " + index);
                System.err.println("Data size: " + data.size());
                System.err.println("FeatureIndex " + featureIndex);
                throw e;
            }
            s = br.readLine();
            if (featureIndex > maxFeatureIndex) {
                maxFeatureIndex = featureIndex;
            }
        }
        vocabulary = new ArrayList<>(maxFeatureIndex - offset + 1);
        wordFrequencies = new ArrayList<>(maxFeatureIndex - offset + 1);
        vocabularyHash = new HashMap((maxFeatureIndex - offset + 1) * 2, 500);
        for (int i = 0; i < maxFeatureIndex - offset + 1; i++) {
            String strInt = (new Integer(i)).toString();
            vocabulary.add(strInt);
            wordFrequencies.add(0f);
            vocabularyHash.put(strInt, i);
        }
        for (int i = 0; i < data.size(); i++) {
            BOWInstance instance = (BOWInstance) (data.get(i));
            if (instance == null || instance.isEmpty()) {
                continue;
            }
            HashMap<Integer, Float> indexMap =
                    instance.getWordIndexesHash();
            Set<Integer> keys = indexMap.keySet();
            if (keys == null || keys.isEmpty()) {
                continue;
            }
            for (int wordIndex : keys) {
                increaseFrequency(wordIndex, indexMap.get(wordIndex));
            }
        }
        numWords = vocabulary.size();
    }

    /**
     * Prints out the vocabulary of words in this corpus.
     *
     * @param path String that is the output file path.
     * @throws Exception
     */
    public void printVocabulary(String path) throws Exception {
        File outFile = new File(path);
        FileUtil.createFileFromPath(path);
        PrintWriter pw = new PrintWriter(new FileWriter(outFile));
        try {
            pw.println("vsize: " + numWords);
            for (int i = 0; i < numWords; i++) {
                pw.println(vocabulary.get(i) + " "
                        + wordFrequencies.get(i).intValue());
            }
        } catch (Exception e) {
            throw e;
        } finally {
            pw.close();
        }
    }

    /**
     * Loads the word vocabulary from the file. Each word/frequency pair is on a
     * separate line, separated by whitespace. The first line contains the
     * information on the total number of words.
     *
     * @param path String that is the input file path.
     * @throws Exception
     */
    public void loadVocabulary(String path) throws Exception {
        initVocabulary(DEFAULT_INIT_WORDS);
        File outFile = new File(path);
        if (!outFile.exists()) {
            throw new Exception("File " + path + " does not exist");
        }
        BufferedReader br = new BufferedReader(new InputStreamReader(
                new FileInputStream(outFile)));
        try {
            String[] wordAndFrequencyPair;
            String s = br.readLine();
            s = s.trim();
            wordAndFrequencyPair = s.split(" ");
            numWords = Integer.parseInt(wordAndFrequencyPair[1]);
            for (int i = 0; i < numWords; i++) {
                s = br.readLine();
                s = s.trim();
                wordAndFrequencyPair = s.split(" ");
                vocabulary.add(wordAndFrequencyPair[0]);
                vocabularyHash.put(wordAndFrequencyPair[0], new Integer(i));
                wordFrequencies.add(
                        Float.parseFloat(wordAndFrequencyPair[1]));
            }
        } catch (Exception e) {
            System.err.println(e.getMessage());
            throw e;
        } finally {
            br.close();
        }
    }

    /**
     * Removes a word from all sparse representations in the corpus. It remains
     * in the vocabulary.
     *
     * @param word String that is the word to remove from sparse
     * representations.
     */
    public void removeWord(String word) {
        int index = getIndexForWord(word);
        if (index != -1) {
            removeWord(index);
        }
    }

    /**
     * Removes a word from all sparse representations in the corpus. It remains
     * in the vocabulary.
     *
     * @param index Integer that is the index of the word to remove.
     */
    public void removeWord(int index) {
        for (int i = 0; i < size(); i++) {
            ((BOWInstance) data.get(i)).removeWord(index);
        }
    }

    /**
     * Removes words from all sparse representations in the corpus. They remain
     * in the vocabulary.
     *
     * @param indexes Integer array of indexes of the words to remove.
     */
    public void removeWords(int[] indexes) {
        for (int i = 0; i < indexes.length; i++) {
            for (int j = 0; j < size(); j++) {
                ((BOWInstance) data.get(j)).removeWord(indexes[i]);
            }
        }
    }

    /**
     * Removes words from all sparse representations in the corpus. They remain
     * in the vocabulary.
     *
     * @param words String[] representing the words to remove.
     */
    public void removeWords(String[] words) {
        int index;
        for (int i = 0; i < words.length; i++) {
            index = getIndexForWord(words[i]);
            if (index != -1) {
                for (int j = 0; j < size(); j++) {
                    ((BOWInstance) data.get(j)).removeWord(index);
                }
            }
        }
    }

    /**
     * Removes from the sparse representations all words that occur below the
     * specified frequency threshold.
     *
     * @param minFreq Minimal frequency of words that will remain in the sparse
     * representations after the removal.
     */
    public void removeLessFrequentWords(int minFreq) {
        int termFrequency;
        for (int i = 0; i < wordFrequencies.size(); i++) {
            termFrequency = wordFrequencies.get(i).intValue();
            if (termFrequency < minFreq) {
                removeWord(i);
            }
        }
    }
}
