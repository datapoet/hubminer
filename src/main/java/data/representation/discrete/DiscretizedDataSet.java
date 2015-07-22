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
package data.representation.discrete;

import data.representation.DataInstance;
import data.representation.DataSet;
import data.representation.sparse.BOWDataSet;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;

/**
 * Discretization of continuous attributes or compressed integer values yields
 * DiscretizedDataSet instances that can be easily handled by learning
 * algorithms such as decision trees or Bayesian methods.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class DiscretizedDataSet implements Serializable {
    
    private static final long serialVersionUID = 1L;

    // Original DataSet from which this one was created.
    private DataSet originalData;
    // Data points.
    public ArrayList<DiscretizedDataInstance> data = null;
    // Discretization structures.
    private HashMap<String, Integer>[] nominalHashes = null;
    private int[] hashSizes = null;
    private ArrayList<String>[] nominalVocabularies = null;
    // By convention, they define [) intervals and are supposed to be ordered.
    // IMPORTANT: Float/Integer -MAX_VALUE / MAX_VALUE are supposed to be at the
    // front and the back of the interval definitions. This is implicitly
    // assumed in the methods defined below.
    private int[][] intIntervalDivisions = null;
    private float[][] floatIntervalDivisions = null;

    public DiscretizedDataSet() {
    }

    /**
     * @param originalData DataSet holder of the original data that produced
     * this discretized data set.
     */
    public DiscretizedDataSet(DataSet originalData) {
        this.originalData = originalData;
        this.data = new ArrayList<>(originalData.size());
    }

    /**
     * @param index Integer index of the instance to fetch.
     * @return DiscretizedDataInstance object at the specified index.
     */
    public DiscretizedDataInstance getInstance(int index) {
        if (data != null && data.size() > index) {
            return data.get(index);
        } else {
            return null;
        }
    }

    /**
     * @return int[] of class frequencies in the original data.
     */
    public int[] getClassFrequencies() {
        if (originalData != null) {
            return originalData.getClassFrequencies();
        } else {
            return null;
        }
    }

    /**
     * @return float[] of class priors in the original data.
     */
    public float[] getClassPriors() {
        if (originalData != null) {
            return originalData.getClassPriors();
        } else {
            return null;
        }
    }

    /**
     * @param index Integer index value.
     * @return Category label at the specified index.
     */
    public int getLabelOf(int index) {
        return getInstance(index).getCategory();
    }

    /**
     * @param indexes Integer array of index values.
     * @return DiscretizedDataSet object containing a subsample of current data.
     */
    public DiscretizedDataSet getSubsample(int[] indexes) {
        if (indexes == null) {
            return cloneDefinition();
        }
        DiscretizedDataSet result = cloneDefinition();
        result.data = new ArrayList<>(indexes.length);
        for (int i = 0; i < indexes.length; i++) {
            result.data.add(data.get(indexes[i]));
        }
        result.originalData = originalData.getSubsample(indexes);
        return result;
    }

    /**
     * @return Integer array of category labels of current data.
     */
    public int[] obtainLabelArray() {
        if (size() == 0) {
            return null;
        } else {
            int[] results = new int[size()];
            for (int i = 0; i < size(); i++) {
                results[i] = data.get(i).getCategory();
            }
            return results;
        }
    }

    /**
     * Rewrites the category labels of the contained data.
     *
     * @param labelArray Integer array of new labels to write to the data.
     * @throws Exception
     */
    public void rewriteLabels(int[] labelArray) throws Exception {
        if (labelArray == null) {
            throw new Exception("null label array given");
        }
        if (labelArray.length != size()) {
            throw new Exception("sizes mismatch");
        }
        for (int i = 0; i < size(); i++) {
            if (data.get(i) != null) {
                data.get(i).setCategory(labelArray[i]);
            }
        }
    }

    /**
     * @return Integer that is the number of integer features.
     */
    public int getNumIntAttr() {
        if (intIntervalDivisions == null || intIntervalDivisions.length == 0) {
            return 0;
        } else {
            return intIntervalDivisions.length;
        }
    }

    /**
     * @return Integer that is the number of float features.
     */
    public int getNumFloatAttr() {
        if (floatIntervalDivisions == null
                || floatIntervalDivisions.length == 0) {
            return 0;
        } else {
            return floatIntervalDivisions.length;
        }
    }

    /**
     * @return Integer that is the number of nominal features.
     */
    public int getNumNominalAttr() {
        if (nominalVocabularies == null || nominalVocabularies.length == 0) {
            return 0;
        } else {
            return nominalVocabularies.length;
        }
    }

    /**
     * Clone the data definition.
     *
     * @return DiscretizedDataSet void of data, but containing all the data
     * definitions from this DiscretizedDataSet.
     */
    public DiscretizedDataSet cloneDefinition() {
        DiscretizedDataSet result = new DiscretizedDataSet();
        result.setFloatIntervalDivisions(floatIntervalDivisions);
        result.setIntIntervalDivisions(intIntervalDivisions);
        result.setNominalHashes(nominalHashes);
        result.setNominalVocabularies(nominalVocabularies);
        return result;
    }

    /**
     * @return Integer that is the data size.
     */
    public int size() {
        if (data == null) {
            return 0;
        } else {
            return data.size();
        }
    }

    /**
     * Checks if the data is empty.
     *
     * @return True if the object contains no data, false otherwise.
     */
    public boolean isEmpty() {
        return size() == 0;
    }

    /**
     * Initialize nominal hashes.
     *
     * @param sizeIncrements Size increments for the hashes.
     */
    public void initHashes(int[][] sizeIncrements) {
        if (sizeIncrements != null && sizeIncrements.length > 0) {
            nominalHashes = new HashMap[sizeIncrements.length];
            nominalVocabularies = new ArrayList[sizeIncrements.length];
            hashSizes = new int[nominalHashes.length];
            for (int i = 0; i < sizeIncrements.length; i++) {
                nominalHashes[i] = new HashMap<>(sizeIncrements[i][0],
                        sizeIncrements[i][1]);
                nominalVocabularies[i] = new ArrayList(sizeIncrements[i][0]);
            }
        }
    }

    /**
     * @param word String that is the word to include.
     * @param index The index of the discretized structure to aggregate this
     * word with.
     * @return
     */
    public int insertWord(String word, int index) {
        if (!nominalHashes[index].containsKey(word)) {
            nominalHashes[index].put(word, hashSizes[index]++);
            nominalVocabularies[index].add(word);
            return (hashSizes[index] - 1);
        } else {
            int wordIndex = nominalHashes[index].get(word);
            return wordIndex;
        }
    }

    /**
     * Binary search for the appropriate interval.
     *
     * @param searchValue Integer search value.
     * @param attIndex Index of the integer attribute.
     * @param first Lower bound.
     * @param second Upper bound.
     * @return Index of the bucket the value belongs to.
     */
    public int findIntIndex(int searchValue, int attIndex, int first,
            int second) {
        // Looking for the last element in the split which is smaller or equal
        // to the search element.
        if (second - first <= 1) {
            if (intIntervalDivisions[attIndex][second] <= searchValue) {
                return second;
            } else {
                return first;
            }
        }
        int middle = (first + second) / 2;
        if (intIntervalDivisions[attIndex][middle] <= searchValue) {
            return findIntIndex(searchValue, attIndex, middle, second);
        } else {
            return findIntIndex(searchValue, attIndex, first, middle);
        }
    }

    /**
     * Binary search for the appropriate interval.
     *
     * @param searchValue Float search value.
     * @param attIndex Index of the float attribute.
     * @param first Lower bound.
     * @param second Upper bound.
     * @return Index of the bucket the value belongs to.
     */
    public int findFloatIndex(float searchValue, int attIndex, int first,
            int second) {
        // Looking for the last element in the split which is smaller or equal
        // to the search element.
        if (second - first <= 1) {
            if (floatIntervalDivisions[attIndex][second] <= searchValue) {
                return second;
            } else {
                return first;
            }
        }
        int middle = (first + second) / 2;
        if (floatIntervalDivisions[attIndex][middle] <= searchValue) {
            return findFloatIndex(searchValue, attIndex, middle, second);
        } else {
            return findFloatIndex(searchValue, attIndex, first, middle);
        }
    }

    /**
     * Performs the discretization of the non-discrete DataSet, based on the
     * available interval buckets.
     *
     * @param dset DataSet to discretize.
     */
    public void discretizeDataSet(DataSet dset) {
        if (dset == null) {
            return;
        }
        this.originalData = dset;
        if (originalData instanceof BOWDataSet) {
            return;
        }
        if (!dset.isEmpty()) {
            data = new ArrayList<>(dset.size());
            for (int i = 0; i < dset.size(); i++) {
                data.add(discretizeInstance(dset.data.get(i)));
            }
        }
    }

    /**
     * Discretize an individual instance based on the available interval bucket
     * definition.
     *
     * @param instance DataInstance to discretize.
     * @return DiscretizedDataInstance that is the result of discretization.
     */
    public DiscretizedDataInstance discretizeInstance(DataInstance instance) {
        if (instance == null) {
            return null;
        }
        DiscretizedDataInstance dInstance =
                new DiscretizedDataInstance(this, true);
        if (instance.fAttr != null) {
            for (int i = 0; i < instance.fAttr.length; i++) {
                dInstance.floatIndexes[i] =
                        findFloatIndex(instance.fAttr[i], i, 0,
                        floatIntervalDivisions[i].length - 1);
            }
        }
        if (instance.iAttr != null) {
            for (int i = 0; i < instance.iAttr.length; i++) {
                dInstance.integerIndexes[i] =
                        findIntIndex(instance.iAttr[i], i, 0,
                        intIntervalDivisions[i].length - 1);
            }
        }
        if (instance.sAttr != null) {
            for (int i = 0; i < instance.sAttr.length; i++) {
                dInstance.nominalIndexes[i] = insertWord(instance.sAttr[i], i);
            }
        }
        dInstance.setOriginalInstance(instance);
        dInstance.setDataContext(this);
        dInstance.setCategory(instance.getCategory());
        return dInstance;
    }

    /**
     * Discretize and insert a non-discrete DataInstance object.
     *
     * @param instance DataInstance object.
     */
    public void insertDataInstance(DataInstance instance) {
        DiscretizedDataInstance dInstance = discretizeInstance(instance);
        data.add(dInstance);
        dInstance.setDataContext(this);
    }

    /**
     * @return DataSet object that this DiscretizedDataSet was derived from.
     */
    public DataSet getOriginalData() {
        return originalData;
    }

    /**
     * @param dset DataSet object that this DiscretizedDataSet was derived from.
     */
    public void setOriginalData(DataSet dset) {
        originalData = dset;
    }

    /**
     * @return HashMap[] of nominal value hashes.
     */
    public HashMap[] getNominalHashes() {
        return nominalHashes;
    }

    /**
     * @param hashes HashMap[] of nominal value hashes.
     */
    public void setNominalHashes(HashMap[] hashes) {
        nominalHashes = hashes;
    }

    /**
     * @return ArrayList<String>[] of nominal vocabularies.
     */
    public ArrayList<String>[] getNominalVocabularies() {
        return nominalVocabularies;
    }

    /**
     * @param nominalVocabularies ArrayList<String>[] of nominal vocabularies.
     */
    public void setNominalVocabularies(ArrayList<String>[]
            nominalVocabularies) {
        this.nominalVocabularies = nominalVocabularies;
    }

    /**
     * @return A 2D integer array that is the definition of integer buckets.
     */
    public int[][] getIntIntervalDivisions() {
        return intIntervalDivisions;
    }

    /**
     * @param intIntervalDivisions A 2D integer array that is the definition of
     * integer buckets.
     */
    public void setIntIntervalDivisions(int[][] intIntervalDivisions) {
        this.intIntervalDivisions = intIntervalDivisions;
    }

    /**
     * @return A 2D float array that is the definition of float buckets.
     */
    public float[][] getFloatIntervalDivisions() {
        return floatIntervalDivisions;
    }

    /**
     * @param floatIntervalDivisions A 2D float array that is the definition of
     * float buckets.
     */
    public void setFloatIntervalDivisions(float[][] floatIntervalDivisions) {
        this.floatIntervalDivisions = floatIntervalDivisions;
    }
}
