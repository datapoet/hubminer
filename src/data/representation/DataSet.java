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
package data.representation;

import data.representation.util.DataMineConstants;
import distances.kernel.Kernel;
import distances.primary.CombinedMetric;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Random;
import java.util.Set;
import learning.supervised.Category;
import learning.supervised.evaluation.cv.MultiCrossValidation;
import learning.unsupervised.Cluster;
import util.ArrayUtil;

/**
 * A class that defines a dataset. This particular implementation supports dense
 * features and it is extended by several other classes that include additional
 * representational capabilities like bag of words or discrete features.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class DataSet implements Serializable {

    private static final long serialVersionUID = 1L;
    private String name;
    // The corresponding DataSet object holding the identifiers of these
    // feature representations, represented as DataInstances. This way of
    // representing the data is optional, but allows for more complex keys and
    // splitting the features that machine learning is to be based upon from
    // the auxiliary data features that can still be contained within the
    // identifier context.
    public DataSet identifiers = null;
    // Feature names.
    public String[] iAttrNames = null;
    public String[] fAttrNames = null;
    public String[] sAttrNames = null;
    // A map between feature names and their indexes.
    private HashMap<String, Integer> attNameMappings;
    // A list of data instances.
    public ArrayList<DataInstance> data;
    private static final int DEFAULT_INIT_CAPACITY = 1000;
    private int initCapacity;

    /**
     * Makes a cluster that contains all data in this DataSet.
     *
     * @return
     */
    public Cluster makeClusterObject() {
        Cluster res = new Cluster(this, size());
        for (int i = 0; i < size(); i++) {
            res.addInstance(i);
        }
        return res;
    }
    
    /**
     * This method re-orders the data instances so that they are grouped by
     * their labels in increasing order.
     */
    public void orderInstancesByClasses() {
        if (!isEmpty()) {
            ArrayList<Integer> indexes = new ArrayList<>(size());
            for (int i = 0; i < size(); i++) {
                indexes.add(i);
            }
            ArrayList<Integer> reOrderedIndexes =
                    MultiCrossValidation.getDataIndexes(indexes, this);
            ArrayList<DataInstance> reOrderedData = new ArrayList<>(size());
            for (int i = 0; i < size(); i++) {
                reOrderedData.add(getInstance(reOrderedIndexes.get(i)));
            }
            this.data = reOrderedData;
        }
    }

    /**
     * Get the data in form of an instance array.
     *
     * @return DataInstance[] of data in this DataSet.
     */
    public DataInstance[] getDataAsArray() {
        if (!isEmpty()) {
            return (DataInstance[]) (data.toArray());
        } else {
            return new DataInstance[0];
        }
    }

    /**
     * Sometimes zero vectors, empty representations, might occur due to I/O
     * errors this method looks it up in the dataset and lets you know if there
     * are any in the data.
     *
     * @return Integer that is the number of empty representations in the data.
     */
    public int countZeroFloatVectors() {
        if (isEmpty()) {
            return 0;
        }
        int sum = 0;
        for (DataInstance instance : data) {
            if (instance.isZeroFloatVector()) {
                sum++;
            }
        }
        return sum;
    }

    /**
     * Sometimes zero vectors, empty representations, might occur due to I/O
     * errors this method looks it up in the dataset and lets you know if there
     * are any in the data.
     *
     * @return ArrayList of integer indexes of empty instances in the data.
     */
    public ArrayList<Integer> getEmptyInstanceIndexes() {
        if (isEmpty()) {
            return new ArrayList<>();
        }
        ArrayList<Integer> emptyInstanceIndexes = new ArrayList<>();
        for (int i = 0; i < size(); i++) {
            DataInstance instance = data.get(i);
            if (instance == null) {
                continue;
            }
            if (instance.isZeroFloatVector()) {
                emptyInstanceIndexes.add(i);
            }
        }
        return emptyInstanceIndexes;
    }

    /**
     * Sometimes zero vectors, empty representations, might occur due to I/O
     * errors this method looks it up in the data and removes them.
     */
    public void removeZeroVectorInstances() {
        ArrayList<Integer> emptyInstanceIndexes = getEmptyInstanceIndexes();
        for (int i = 0; i < emptyInstanceIndexes.size(); i++) {
            // Since the list shifts to the left with each remove.
            data.remove(emptyInstanceIndexes.get(i) - i);
        }
    }

    /**
     * This method merges two representations of the same data, two disjoint
     * feature sets. If there are any common features between the two DataSet
     * objects, they should first be removed so that only one copy remains.
     *
     * @param dsetA First DataSet object to merge.
     * @param dsetB Second DataSet object to merge.
     * @return Merged DataSet object, the same size but containing both feature
     * representations.
     * @throws Exception
     */
    public static DataSet mergeDisjointRepresentations(
            DataSet dsetA, DataSet dsetB) throws Exception {
        DataSet result;
        if (dsetA == null && dsetB == null) {
            return null;
        } else if (dsetA == null) {
            return dsetB;
        } else if (dsetB == null) {
            return dsetA;
        }
        int sizeA, sizeB;
        sizeA = dsetA.size();
        sizeB = dsetB.size();
        if (sizeA != sizeB) {
            throw new Exception("DataSet objects for merger contain a different"
                    + " number of elements. The first one has size " + sizeA
                    + " and the second one size " + sizeB);
        }
        result = new DataSet();
        int iLength, fLength, nLength;
        iLength = 0;
        fLength = 0;
        nLength = 0;
        iLength += dsetA.getNumIntAttr();
        iLength += dsetB.getNumIntAttr();
        fLength += dsetA.getNumFloatAttr();
        fLength += dsetB.getNumFloatAttr();
        nLength += dsetA.getNumFloatAttr();
        nLength += dsetB.getNumFloatAttr();
        // First handle the specification of representation, the model.
        // Integer feature names.
        if (iLength > 0) {
            result.iAttrNames = new String[iLength];
            int index = 0;
            if (dsetA.hasIntAttr()) {
                for (int i = 0; i < dsetA.iAttrNames.length; i++) {
                    result.iAttrNames[index++] = dsetA.iAttrNames[i];
                }
            }
            if (dsetB.hasIntAttr()) {
                for (int i = 0; i < dsetB.iAttrNames.length; i++) {
                    result.iAttrNames[index++] = dsetB.iAttrNames[i];
                }
            }
        }
        // Float feature names.
        if (fLength > 0) {
            result.fAttrNames = new String[fLength];
            int index = 0;
            if (dsetA.hasFloatAttr()) {
                for (int i = 0; i < dsetA.fAttrNames.length; i++) {
                    result.fAttrNames[index++] = dsetA.fAttrNames[i];
                }
            }
            if (dsetB.hasFloatAttr()) {
                for (int i = 0; i < dsetB.fAttrNames.length; i++) {
                    result.fAttrNames[index++] = dsetB.fAttrNames[i];
                }
            }
        }
        // Nominal feature names.
        if (nLength > 0) {
            result.sAttrNames = new String[nLength];
            int index = 0;
            if (dsetA.hasNominalAttr()) {
                for (int i = 0; i < dsetA.sAttrNames.length; i++) {
                    result.sAttrNames[index++] = dsetA.sAttrNames[i];
                }
            }
            if (dsetB.hasNominalAttr()) {
                for (int i = 0; i < dsetB.sAttrNames.length; i++) {
                    result.sAttrNames[index++] = dsetB.sAttrNames[i];
                }
            }
        }
        result.makeFeatureMappings();
        DataInstance instanceA, instanceB;
        for (int i = 0; i < sizeA; i++) {
            instanceA = dsetA.getInstance(i);
            instanceB = dsetB.getInstance(i);
            DataInstance instance = new DataInstance(result);
            instance.embedInDataset(result);
            // Include integer features of both representations.
            if (iLength > 0) {
                int index = 0;
                if (dsetA.hasIntAttr()) {
                    for (int j = 0; j < dsetA.iAttrNames.length; j++) {
                        instance.iAttr[index++] = instanceA.iAttr[j];
                    }
                }
                if (dsetB.hasIntAttr()) {
                    for (int j = 0; j < dsetB.iAttrNames.length; j++) {
                        instance.iAttr[index++] = instanceB.iAttr[j];
                    }

                }
            }
            // Include float features of both representations.
            if (fLength > 0) {
                int index = 0;
                if (dsetA.hasFloatAttr()) {
                    for (int j = 0; j < dsetA.fAttrNames.length; j++) {
                        instance.fAttr[index++] = instanceA.fAttr[j];
                    }
                }
                if (dsetB.hasFloatAttr()) {
                    for (int j = 0; j < dsetB.fAttrNames.length; j++) {
                        instance.fAttr[index++] = instanceB.fAttr[j];
                    }
                }
            }
            // Include nominal features of both representations.
            if (nLength > 0) {
                int index = 0;
                if (dsetA.hasNominalAttr()) {
                    for (int j = 0; j < dsetA.iAttrNames.length; j++) {
                        instance.sAttr[index++] = instanceA.sAttr[j];
                    }
                }
                if (dsetB.hasNominalAttr()) {
                    for (int j = 0; j < dsetB.iAttrNames.length; j++) {
                        instance.sAttr[index++] = instanceB.sAttr[j];
                    }
                }
            }
            instance.setIdentifier(instanceA.getIdentifier());
            result.addDataInstance(instance);
        }
        result.initCapacity = dsetA.initCapacity;
        result.setName(dsetA.name);
        return result;
    }

    /**
     * Compares the feature definition of this object to a feature definition of
     * another DataSet that is passed as a parameter.
     *
     * @param dset DataSet to compare to.
     * @return True if they have equal feature definitions, false otherwise.
     */
    public boolean equalsInFeatureDefinition(DataSet dset) {
        if (this.getNumFloatAttr() == dset.getNumFloatAttr()
                && this.getNumIntAttr() == dset.getNumIntAttr()
                && this.getNumNominalAttr() == dset.getNumNominalAttr()) {
            if (attNameMappings != null && dset.attNameMappings != null) {
                Set<String> firstKeys = attNameMappings.keySet();
                Set<String> secondKeys = dset.attNameMappings.keySet();
                return firstKeys.equals(secondKeys);
            }
            return true;
        } else {
            return false;
        }
    }

    /**
     * Insert a new nominal attribute into the DataSet.
     *
     * @param attName String that is the nominal attribute name to insert.
     */
    public void addNewNominalAtt(String attName) {
        DataInstance instance;
        if (sAttrNames != null) {
            String[] newAttNames = new String[sAttrNames.length + 1];
            for (int i = 0; i < sAttrNames.length; i++) {
                newAttNames[i] = sAttrNames[i];
            }
            newAttNames[sAttrNames.length] = attName;
            if (!isEmpty()) {
                for (int i = 0; i < size(); i++) {
                    instance = data.get(i);
                    String[] newAttVals = new String[sAttrNames.length + 1];
                    for (int j = 0; j < sAttrNames.length; j++) {
                        newAttVals[j] = instance.sAttr[j];
                    }
                    instance.sAttr = newAttVals;
                    instance.sAttr[sAttrNames.length] = "";
                }
            }
            sAttrNames = newAttNames;
        } else {
            sAttrNames = new String[1];
            sAttrNames[0] = attName;
            if (!isEmpty()) {
                for (int i = 0; i < size(); i++) {
                    instance = data.get(i);
                    instance.sAttr = new String[1];
                    instance.sAttr[0] = "";
                }
            }
        }
    }

    /**
     * @return True if there is a link to a DataSet of identifiers for this
     * data, false otherwise.
     */
    public boolean hasIdentifiers() {
        if (identifiers == null || identifiers.isEmpty()) {
            return false;
        } else {
            return true;
        }
    }

    /**
     * Gets the label of the DataInstance at the specified index.
     *
     * @param index Index of the instance.
     * @return Integer that is the label of the DataInstance at the specified
     * index.
     */
    public int getLabelOf(int index) {
        if (index > size() || data.get(index) == null) {
            return -1;
        }
        return data.get(index).getCategory();
    }

    /**
     * Sets the label of the DataInstance at the specified index.
     *
     * @param index Index of the instance.
     * @param label Integer that is the label to set the instance label to.
     */
    public void setLabelOf(int index, int label) {
        if (index < size() && data.get(index) != null) {
            data.get(index).setCategory(label);
        }
    }

    /**
     * @param index Index of the instance.
     * @return The DataInstance at the specified index.
     */
    public DataInstance getInstance(int index) {
        if (index > size() || data.get(index) == null) {
            return null;
        }
        return data.get(index);
    }

    /**
     * An auxiliary method that returns the largest absolute float value among
     * all the attributes and all the instances.
     *
     * @return Float value that is maximal across all features and instances.
     */
    public float getMaxFloatFeatureValue() {
        if (size() == 0 || !hasFloatAttr()) {
            return -Float.MAX_VALUE;
        } else {
            float max = -Float.MAX_VALUE;
            for (DataInstance instance : data) {
                for (int i = 0; i < instance.fAttr.length; i++) {
                    if (!DataMineConstants.isAcceptableFloat(
                            instance.fAttr[i])) {
                        continue;
                    }
                    if (Math.abs(instance.fAttr[i]) > max) {
                        max = Math.abs(instance.fAttr[i]);
                    }
                }
            }
            return max;
        }
    }

    /**
     * @return Float array representing prior class probabilities
     */
    public float[] getClassPriors() {
        HashMap<Integer, Integer> categoryHash = new HashMap<>(100, 10);
        int numClasses = 0;
        int tempCat;
        for (int i = 0; i < size(); i++) {
            if (data.get(i) != null) {
                tempCat = data.get(i).getCategory();
                if (!categoryHash.containsKey(new Integer(tempCat))) {
                    categoryHash.put(
                            new Integer(tempCat), new Integer(numClasses++));
                }
            }
        }
        float[] distr = new float[numClasses];
        for (DataInstance instance : data) {
            distr[categoryHash.get(instance.getCategory())]++;
        }
        for (int i = 0; i < distr.length; i++) {
            distr[i] /= size();
        }
        return distr;
    }

    /**
     * @return Integer array representing class frequencies.
     */
    public int[] getClassFrequencies() {
        int numClasses = countCategories();
        int[] freqs = new int[numClasses];
        for (int i = 0; i < size(); i++) {
            freqs[data.get(i).getCategory()]++;
        }
        return freqs;
    }

    /**
     * @return Integer array representing class frequencies.
     */
    public float[] getClassFrequenciesAsFloatArray() {
        int numClasses = countCategories();
        float[] freqs = new float[numClasses];
        for (int i = 0; i < size(); i++) {
            freqs[data.get(i).getCategory()]++;
        }
        return freqs;
    }

    /**
     * Standardize all float values by subtracting their mean and dividing by
     * standard deviation.
     */
    public void standardizeAllFloats() {
        if (fAttrNames == null) {
            return;
        }
        for (int i = 0; i < fAttrNames.length; i++) {
            standardizeFloatAtt(i);
        }
    }

    /**
     * Standardizes the attribute values by subtracting its mean and dividing by
     * standard deviation.
     *
     * @param attIndex index of the attribute within the float array.
     */
    public void standardizeFloatAtt(int attIndex) {
        double mean = 0;
        double variance = 0;
        int numAcceptableFAttValues = 0;
        for (DataInstance instance : data) {
            if (!DataMineConstants.isAcceptableFloat(
                    instance.fAttr[attIndex])) {
                continue;
            }
            mean += instance.fAttr[attIndex];
            numAcceptableFAttValues++;
        }
        if (numAcceptableFAttValues == 0) {
            return;
        }
        mean /= numAcceptableFAttValues;
        for (DataInstance instance : data) {
            variance += (mean - instance.fAttr[attIndex])
                    * (mean - instance.fAttr[attIndex]);
        }
        if (variance > 0) {
            variance /= numAcceptableFAttValues;
            double stDev = Math.sqrt(variance);
            for (DataInstance instance : data) {
                if (!DataMineConstants.isAcceptableFloat(
                        instance.fAttr[attIndex])) {
                    continue;
                }
                instance.fAttr[attIndex] =
                        (float) (instance.fAttr[attIndex] - mean) /
                        (float) stDev;
            }
        } else {
            for (DataInstance instance : data) {
                instance.fAttr[attIndex] = 0;
            }
        }
    }

    /**
     * This method makes a copy where only the references are genuinely copied.
     * The arrays that store the feature values and attribute names are not
     * copied, so there is no overhead if those values are not going to be
     * modified (but rather the data labels or something like that).
     *
     * @return An array referenced copy of the DataSet.
     */
    public DataSet makeArrayReferencedCopy() {
        DataSet result = new DataSet();
        result.iAttrNames = iAttrNames;
        result.fAttrNames = fAttrNames;
        result.sAttrNames = sAttrNames;
        result.attNameMappings = attNameMappings;
        if (identifiers != null) {
            result.identifiers = identifiers.makeArrayReferencedCopy();
        }
        result.data = new ArrayList(size());
        for (int i = 0; i < size(); i++) {
            DataInstance instance = data.get(i).makeArrayReferencedCopy();
            instance.embedInDataset(result);
            result.addDataInstance(instance);
        }
        return result;
    }

    /**
     * Generates an array of Category objects that correspond to this data.
     *
     * @param numCategories Integer that is the number of categories in the
     * data.
     * @return Category[] containing the data in this DataSet.
     */
    public Category[] getClassesArray(int numCategories) {
        Category[] result = new Category[numCategories];
        for (int i = 0; i < numCategories; i++) {
            result[i] = new Category("category " + i, 1000, this);
        }
        for (int i = 0; i < size(); i++) {
            // Noise labels are set to -1 usually, so here we ignore them.
            if (data.get(i).getCategory() >= 0) {
                result[data.get(i).getCategory()].addInstance(i);
            }
        }
        return result;
    }

    /**
     * Takes a subsample from the data.
     *
     * @param indexes An array of indexes to sample.
     * @return DataSet that is the subsample of the original data.
     */
    public DataSet getSubsample(int[] indexes) {
        if (indexes == null) {
            return new DataSet();
        }
        DataSet result = cloneDefinition();
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

    /**
     * Takes a subsample from the data.
     *
     * @param indexes An ArrayList of integer indexes to sample.
     * @return DataSet that is the subsample of the original data.
     */
    public DataSet getSubsample(ArrayList<Integer> indexes) {
        if (indexes == null) {
            return new DataSet();
        }
        DataSet result = cloneDefinition();
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
     * Get all data labels as an array.
     *
     * @return An array of labels of instances in the DataSet.
     */
    public int[] obtainLabelArray() {
        if (size() == 0) {
            return new int[0];
        } else {
            int[] results = new int[size()];
            for (int i = 0; i < size(); i++) {
                results[i] = data.get(i).getCategory();
            }
            return results;
        }
    }

    /**
     * Get a subset of labels as an array.
     *
     * @param beginIndex Integer that is the start index.
     * @param endIndex Integer that is the end index.
     * @return An array of integer labels starting a instance indexed beginIndex
     * and ending at instance labeled endIndex.
     * @throws Exception
     */
    public int[] obtainLabelArray(int beginIndex, int endIndex)
            throws Exception {
        if (endIndex >= size()) {
            throw new ArrayIndexOutOfBoundsException("End index out of range "
                    + endIndex + " compared to " + size());
        }
        if (beginIndex < 0) {
            throw new ArrayIndexOutOfBoundsException("Begin index out of range "
                    + beginIndex);
        }
        if (size() == 0) {
            return new int[0];
        } else {
            int length = endIndex - beginIndex + 1;
            int[] results = new int[length];
            for (int i = beginIndex; i <= endIndex; i++) {
                results[i - beginIndex] = data.get(i).getCategory();
            }
            return results;
        }
    }

    /**
     * Rewrite the existing data labels with the provided ones.
     *
     * @param labelArray Label array to rewrite the original labels with.
     * @throws Exception
     */
    public void rewriteLabels(int[] labelArray) throws Exception {
        if (labelArray == null) {
            throw new NullPointerException("Null label array given");
        }
        if (labelArray.length != size()) {
            throw new Exception("Size mismatch");
        }
        for (int i = 0; i < size(); i++) {
            if (data.get(i) != null) {
                data.get(i).setCategory(labelArray[i]);
            }
        }
    }

    /**
     * Rewrite the existing data labels with the provided ones.
     *
     * @param labelList Label ArrayList to rewrite the original labels with.
     * @throws Exception
     */
    public void rewriteLabels(ArrayList<Integer> labelList) throws Exception {
        if (labelList == null) {
            throw new NullPointerException("Null label array given");
        }
        if (labelList.size() != size()) {
            throw new Exception("Ssizes mismatch");
        }
        for (int i = 0; i < size(); i++) {
            if (data.get(i) != null) {
                data.get(i).setCategory(labelList.get(i));
            }
        }
    }

    /**
     * Rewrite a subset of the labels with the provided label array.
     *
     * @param labelArray Label array to rewrite the original labels with.
     * @param beginIndex Integer that is the start index.
     * @param endIndex Integer that is the end index.
     * @throws Exception
     */
    public void rewriteLabels(int[] labelArray, int beginIndex, int endIndex)
            throws Exception {
        if (labelArray == null) {
            throw new NullPointerException("Null label array given");
        }
        if (endIndex >= size()) {
            throw new ArrayIndexOutOfBoundsException("End index out of bounds "
                    + endIndex + " compared to " + size());
        }
        if (beginIndex < 0) {
            throw new ArrayIndexOutOfBoundsException("Begin index out of "
                    + "bounds " + beginIndex);
        }
        if (labelArray.length != (endIndex - beginIndex + 1)) {
            throw new Exception("Sizes mismatch");
        }
        for (int i = beginIndex; i <= endIndex; i++) {
            if (data.get(i) != null) {
                data.get(i).setCategory(labelArray[i]);
            }
        }
    }

    /**
     * Rewrite a subset of the labels with the provided label ArrayList.
     *
     * @param labelArray Label ArrayList to rewrite the original labels with.
     * @param beginIndex Integer that is the start index.
     * @param endIndex Integer that is the end index.
     * @throws Exception
     */
    public void rewriteLabels(ArrayList<Integer> labelArray, int beginIndex,
            int endIndex) throws Exception {
        if (labelArray == null) {
            throw new NullPointerException("Null label array given");
        }
        if (endIndex >= size()) {
            throw new ArrayIndexOutOfBoundsException("End index out of bounds "
                    + endIndex + " compared to " + size());
        }
        if (beginIndex < 0) {
            throw new ArrayIndexOutOfBoundsException("Begin index out of "
                    + "bounds " + beginIndex);
        }
        if (labelArray.size() != (endIndex - beginIndex + 1)) {
            throw new Exception("Sizes mismatch");
        }
        for (int i = beginIndex; i <= endIndex; i++) {
            if (data.get(i) != null) {
                data.get(i).setCategory(labelArray.get(i));
            }
        }
    }

    /**
     * @param probability Float that is the probability of an item being
     * mislabeled.
     */
    public void induceWeightProportionalMislabeling(float probability,
            float[] weights) {
        induceWeightProportionalMislabeling(probability, findMaxLabel() + 1,
                weights);
    }

    /**
     * @param probability Float that is the probability of an item being
     * mislabeled.
     * @param numCategories Number of categories in the data.
     */
    public void induceWeightProportionalMislabeling(float probability,
            int numCategories, float[] weights) {
        // First construct a cumulative weight array.
        float[] cumulWeights = new float[weights.length];
        cumulWeights[0] = weights[0];
        for (int i = 1; i < weights.length; i++) {
            cumulWeights[i] = cumulWeights[i - 1] + weights[i];
        }
        Random randa = new Random();
        int newCat;
        float choice;
        int chosenIndex;
        // Initializes to false.
        boolean[] selectedInstances = new boolean[data.size()];
        int numMislabeled = (int) (data.size() * probability);
        if (probability > 0) {
            numMislabeled = Math.max(1, numMislabeled);
        }
        for (int i = 0; i < numMislabeled; i++) {
            // Select an instance.
            do {
                choice = randa.nextFloat();
                choice *= cumulWeights[cumulWeights.length - 1];
                chosenIndex = ArrayUtil.findIndex(choice, cumulWeights);
            } while (selectedInstances[chosenIndex]);
            selectedInstances[chosenIndex] = true;
            DataInstance instance = data.get(chosenIndex);
            newCat = randa.nextInt(numCategories);
            while (newCat == instance.getCategory()) {
                newCat = randa.nextInt(numCategories);
            }
            instance.setCategory(newCat);
        }
    }

    /**
     * @param probability Float that is the probability of an item being
     * mislabeled.
     */
    public void induceMislabeling(float probability) {
        induceMislabeling(probability, findMaxLabel() + 1);
    }

    /**
     * @param probability Float that is the probability of an item being
     * mislabeled.
     * @param numCategories Number of categories in the data.
     */
    public void induceMislabeling(float probability, int numCategories) {
        Random randa = new Random();
        int newCat;
        float choice;
        for (DataInstance instance : data) {
            choice = randa.nextFloat();
            if (choice < probability) {
                newCat = randa.nextInt(numCategories);
                while (newCat == instance.getCategory()) {
                    newCat = randa.nextInt(numCategories);
                }
                instance.setCategory(newCat);
            }
        }
    }

    /**
     *
     * @return The number of categories in this data.
     */
    public int countCategories() {
        HashMap<Integer, Integer> categoryHash = new HashMap<>(100, 10);
        int numClasses = 0;
        int currCat;
        for (int i = 0; i < size(); i++) {
            if (data.get(i) != null) {
                currCat = data.get(i).getCategory();
                if (!categoryHash.containsKey(new Integer(currCat))) {
                    categoryHash.put(new Integer(currCat),
                            new Integer(numClasses++));
                }
            }
        }
        return numClasses;
    }

    /**
     * This method standardizes category labels, so that they span from zero to
     * numCat-1. This makes later processing easier.
     */
    public void standardizeCategories() {
        HashMap<Integer, Integer> categoryHash = new HashMap<>(100, 10);
        int numClasses = 0;
        int currCat, newCat;
        for (int i = 0; i < size(); i++) {
            if (data.get(i) != null) {
                currCat = data.get(i).getCategory();
                if (!categoryHash.containsKey(new Integer(currCat))) {
                    categoryHash.put(new Integer(currCat),
                            new Integer(numClasses++));
                }
            }
        }
        for (int i = 0; i < size(); i++) {
            DataInstance temp = (DataInstance) (data.get(i));
            currCat = temp.getCategory();
            newCat = categoryHash.get(new Integer(currCat));
            temp.setCategory(newCat);
        }
    }

    /**
     * Calculates the distance matrix.
     *
     * @param cmet CombinedMetric object for distance calculations.
     * @return The distance matrix in the specified metric.
     * @throws Exception
     */
    public float[][] calculateDistMatrix(CombinedMetric cmet) throws Exception {
        if (size() == 0) {
            return null;
        } else {
            float[][] distances = new float[size()][];
            for (int i = 0; i < size(); i++) {
                distances[i] = new float[size() - i - 1];
                for (int j = i + 1; j < size(); j++) {
                    distances[i][j - i - 1] = cmet.dist(data.get(i),
                            data.get(j));
                }
            }
            return distances;
        }
    }

    /**
     * Calculates the kernel matrix in a multi-threaded way.
     *
     * @param ker Kernel to use.
     * @param numThreads Number of threads to use.
     * @return Kernel matrix.
     * @throws Exception
     */
    public float[][] calculateKernelMatrixMultThr(Kernel ker, int numThreads)
            throws Exception {
        if (size() == 0) {
            return null;
        } else {
            float[][] distances = new float[size()][];
            int size = size();
            int chunkSize = size / numThreads;
            Thread[] threads = new Thread[numThreads];
            for (int i = 0; i < numThreads - 1; i++) {
                threads[i] = new Thread(new KerMatCalculator(
                        i * chunkSize, (i + 1) * chunkSize - 1,
                        distances, ker));
                threads[i].start();
            }
            threads[numThreads - 1] = new Thread(new KerMatCalculator(
                    (numThreads - 1) * chunkSize, size - 1, distances, ker));
            threads[numThreads - 1].start();
            for (int i = 0; i < numThreads; i++) {
                if (threads[i] != null) {
                    try {
                        threads[i].join();
                    } catch (Throwable t) {
                    }
                }
            }
            return distances;
        }
    }

    /**
     * A class that is used for multi-threaded kernel matrix calculations.
     */
    class KerMatCalculator implements Runnable {

        int startRow;
        int endRow;
        float[][] kmat;
        Kernel ker;

        /**
         * Calculates matrix values for a block of the matrix.
         *
         * @param startRow Start row index.
         * @param endRow End row index.
         * @param kmat Kernel matrix to write to.
         * @param ker Kernel to use.
         */
        public KerMatCalculator(
                int startRow, int endRow, float[][] kmat, Kernel ker) {
            this.startRow = startRow;
            this.endRow = endRow;
            this.kmat = kmat;
            this.ker = ker;
        }

        @Override
        public void run() {
            try {
                for (int i = startRow; i <= endRow; i++) {
                    kmat[i] = new float[size() - i];
                    for (int j = i; j < size(); j++) {
                        kmat[i][j - i] = ker.dot(data.get(i), data.get(j));
                    }
                }
            } catch (Exception e) {
            }
        }
    }

    /**
     * Calculate the distance matrix in a multi-threaded way.
     *
     * @param cmet CombinedMetric object for distance calculations.
     * @param numThreads Number of threads to use.
     * @return Distance matrix.
     * @throws Exception
     */
    public float[][] calculateDistMatrixMultThr(CombinedMetric cmet,
            int numThreads) throws Exception {
        if (size() == 0) {
            return null;
        } else {
            float[][] distances = new float[size()][];
            int size = size();
            int chunkSize = size / numThreads;
            Thread[] threads = new Thread[numThreads];
            for (int i = 0; i < numThreads - 1; i++) {
                threads[i] = new Thread(
                        new DmCalculator(i * chunkSize, (i + 1) * chunkSize - 1,
                        distances, cmet));
                threads[i].start();
            }
            threads[numThreads - 1] = new Thread(
                    new DmCalculator((numThreads - 1) * chunkSize, size - 1,
                    distances, cmet));
            threads[numThreads - 1].start();
            for (int i = 0; i < numThreads; i++) {
                if (threads[i] != null) {
                    try {
                        threads[i].join();
                    } catch (Throwable t) {
                    }
                }
            }
            return distances;
        }
    }

    class DmCalculator implements Runnable {

        int startRow;
        int endRow;
        float[][] distances;
        CombinedMetric cmet;

        /**
         * Calculates the distances in a block of the matrix.
         *
         * @param startRow Start row index.
         * @param endRow End row index.
         * @param distances Distance matrix to write to.
         * @param cmet CombinedMetric object for distance calculations.
         */
        public DmCalculator(int startRow, int endRow, float[][] distances,
                CombinedMetric cmet) {
            this.startRow = startRow;
            this.endRow = endRow;
            this.distances = distances;
            this.cmet = cmet;
        }

        @Override
        public void run() {
            try {
                for (int i = startRow; i <= endRow; i++) {
                    distances[i] = new float[size() - i - 1];
                    for (int j = i + 1; j < size(); j++) {
                        distances[i][j - i - 1] =
                                cmet.dist(data.get(i), data.get(j));
                    }
                }
            } catch (Exception e) {
                System.err.println(e.getMessage());
            }
        }
    }

    /**
     * @return The maximum value of any label in the data, the highest class
     * index. This may or may not correspond to the number of categories,
     * depending on the labeling.
     */
    public int findMaxLabel() {
        if (!isEmpty()) {
            int maxLabel = -1;
            for (int i = 0; i < size(); i++) {
                if (data.get(i).getCategory() > maxLabel) {
                    maxLabel = data.get(i).getCategory();
                }
            }
            return maxLabel;
        } else {
            return -1;
        }
    }

    /**
     * Introduces some noise to the features of the instances.
     *
     * @param pMutate Probability of mutation.
     * @param stDev Standard deviation of mutations.
     */
    public void addGaussianNoiseToNormalizedCollection(
            float pMutate, float stDev) {
        DataInstance instance;
        Random randa = new Random();
        float choice;
        for (int i = 0; i < size(); i++) {
            instance = data.get(i);
            for (int j = 0; j < getNumFloatAttr(); j++) {
                if (!DataMineConstants.isAcceptableFloat(instance.fAttr[j])) {
                    continue;
                }
                choice = randa.nextFloat();
                if (choice < pMutate) {
                    instance.fAttr[j] += stDev * randa.nextGaussian();
                    instance.fAttr[j] = Math.max(instance.fAttr[j], 0);
                    instance.fAttr[j] = Math.min(instance.fAttr[j], 1);
                }
            }
        }
    }

    /**
     * Gets the radius of the volume containing the instances.
     *
     * @param indexes Instance indexes.
     * @param cmet CombinedMetric object for distance calculations.
     * @return Radius of the volume containing the instances.
     * @throws Exception
     */
    public float getRadiusOfVolumeForInstances(int[] indexes,
            CombinedMetric cmet) throws Exception {
        Cluster clust = new Cluster(this, indexes.length);
        for (int i = 0; i < indexes.length; i++) {
            clust.addInstance(indexes[i]);
        }
        DataInstance centroid = clust.getCentroid();
        // Find maximum distance to the centroid, the hypersphere radius.
        float maxDist = -Float.MAX_VALUE;
        float currDist;
        for (int i = 0; i < indexes.length; i++) {
            currDist = cmet.dist(centroid, data.get(indexes[i]));
            if (currDist > maxDist) {
                maxDist = currDist;
            }
        }
        return maxDist;
    }

    /**
     * Gets the radius of the volume containing the instances.
     *
     * @param indexes Instance indexes.
     * @param cmet CombinedMetric object for distance calculations.
     * @param upperBound The index up until which to take data from indexes.
     * @return Radius of the volume containing the instances.
     * @throws Exception
     */
    public float getRadiusOfVolumeForInstances(int[] indexes,
            CombinedMetric cmet, int upperBound) throws Exception {
        Cluster clust = new Cluster(this, indexes.length);
        for (int i = 0; i < upperBound; i++) {
            clust.addInstance(indexes[i]);
        }
        DataInstance centroid = clust.getCentroid();
        // Find maximum distance to the centroid, the hypersphere radius.
        float maxDist = -Float.MAX_VALUE;
        float currDist;
        for (int i = 0; i < indexes.length; i++) {
            currDist = cmet.dist(centroid, data.get(indexes[i]));
            if (currDist > maxDist) {
                maxDist = currDist;
            }
        }
        return maxDist;
    }

    /**
     * The volume isn't multiplied by a constant factor which depends on
     * dimensionality. The reason for this is that it takes time to calculate
     * this factor and it's irrelevant for checking relative densities, since
     * there will get normalized anyway. There is still a problem if the
     * dimensionality is really high, the volume will be enormous.
     *
     * @param indexes Instance indexes.
     * @param cmet CombinedMetric object for distance calculations.
     * @return Radial volume for the given instances, up to a multiplicative
     * constant.
     * @throws Exception
     */
    public double getNonExactRadialVolumeForInstances(int[] indexes,
            CombinedMetric cmet) throws Exception {
        Cluster clust = new Cluster(this, indexes.length);
        for (int i = 0; i < indexes.length; i++) {
            clust.addInstance(indexes[i]);
        }
        DataInstance centroid = clust.getCentroid();
        // Find maximum distance to the centroid, the hypersphere radius.
        float maxDist = -Float.MAX_VALUE;
        float currDist;
        for (int i = 0; i < indexes.length; i++) {
            currDist = cmet.dist(centroid, data.get(indexes[i]));
            if (currDist > maxDist) {
                maxDist = currDist;
            }
        }
        return Math.pow(maxDist, getNumFloatAttr());
    }

    /**
     * Calculate the volume of the cube containing the data.
     *
     * @param indexes Instance indexes.
     * @return The volume of the cube containing the data.
     */
    public float getCubeVolumeForInstances(int[] indexes) {
        if (!hasFloatAttr()) {
            return 1f;
        } else {
            int nF = getNumFloatAttr();
            float[] lBounds = new float[nF];
            float[] uBounds = new float[nF];
            for (int j = 0; j < nF; j++) {
                lBounds[j] = Float.MAX_VALUE;
                uBounds[j] = -Float.MAX_VALUE;
            }
            DataInstance instance;
            for (int i = 0; i < indexes.length; i++) {
                instance = data.get(indexes[i]);
                for (int j = 0; j < nF; j++) {
                    if (!DataMineConstants.isAcceptableFloat(
                            instance.fAttr[j])) {
                        continue;
                    }
                    if (instance.fAttr[j] > uBounds[j]) {
                        uBounds[j] = instance.fAttr[j];
                    }
                    if (instance.fAttr[j] < lBounds[j]) {
                        lBounds[j] = instance.fAttr[j];
                    }
                }
            }
            float volume = 1f;
            for (int j = 0; j < nF; j++) {
                volume *= (uBounds[j] - lBounds[j]);
            }
            return volume;
        }
    }

    /**
     * Clones the data definition.
     *
     * @return DataSet with a same feature definition as this DataSet.
     */
    public DataSet cloneDefinition() {
        DataSet result = new DataSet();
        if (attNameMappings != null) {
            result.attNameMappings = (HashMap) (attNameMappings.clone());
        }
        if (iAttrNames != null) {
            result.iAttrNames = Arrays.copyOf(iAttrNames, iAttrNames.length);
        }
        if (sAttrNames != null) {
            result.sAttrNames = Arrays.copyOf(sAttrNames, sAttrNames.length);
        }
        if (fAttrNames != null) {
            result.fAttrNames = Arrays.copyOf(fAttrNames, fAttrNames.length);
        }
        result.initCapacity = initCapacity;
        result.setName(name);
        return result;
    }

    /**
     * @param name String that is the dataset name.
     */
    public void setName(String name) {
        this.name = name;
    }

    /**
     * @return String that is the dataset name.
     */
    public String getName() {
        if (name != null) {
            return name;
        } else {
            return "DataSet";
        }
    }

    /**
     * Normalize all float values to the [0, 1] range.
     */
    public void normalizeFloats() {
        if (hasFloatAttr()) {
            float[] maxVal = new float[getNumFloatAttr()];
            for (int i = 0; i < maxVal.length; i++) {
                maxVal[i] = -Float.MAX_VALUE;
            }
            DataInstance instance;
            for (int i = 0; i < size(); i++) {
                instance = data.get(i);
                for (int j = 0; j < getNumFloatAttr(); j++) {
                    if (DataMineConstants.isAcceptableFloat(
                            instance.fAttr[j])) {
                        if (Math.abs(instance.fAttr[j]) > maxVal[j]) {
                            maxVal[j] = instance.fAttr[j];
                        }
                    }
                }
            }
            for (int i = 0; i < size(); i++) {
                instance = data.get(i);
                for (int j = 0; j < getNumFloatAttr(); j++) {
                    if (DataMineConstants.isAcceptableFloat(
                            instance.fAttr[j])) {
                        if (maxVal[j] != 0) {
                            instance.fAttr[j] /= maxVal[j];
                        }
                    }
                }
            }
        }
    }

    /**
     * @return The total number of features.
     */
    public int getNumAttr() {
        return getNumIntAttr() + getNumFloatAttr() + getNumNominalAttr();
    }

    /**
     * @return The number of integer features.
     */
    public int getNumIntAttr() {
        if (iAttrNames == null) {
            return 0;
        } else {
            return iAttrNames.length;
        }
    }

    /**
     * @return The number of float features.
     */
    public int getNumFloatAttr() {
        if (fAttrNames == null) {
            return 0;
        } else {
            return fAttrNames.length;
        }
    }

    /**
     * @return The number of nominal features.
     */
    public int getNumNominalAttr() {
        if (sAttrNames == null) {
            return 0;
        } else {
            return sAttrNames.length;
        }
    }

    /**
     * Fill in the feature name / index hash.
     */
    public void makeFeatureMappings() {
        int attNum = getNumAttr();
        attNameMappings = new HashMap<>(attNum * 5 + 2000);
        int index;
        if (iAttrNames != null) {
            for (int i = 0; i < iAttrNames.length; i++) {
                index = i * 10 + DataMineConstants.INTEGER;
                attNameMappings.put(iAttrNames[i], index);
            }
        }
        if (fAttrNames != null) {
            for (int i = 0; i < fAttrNames.length; i++) {
                index = i * 10 + DataMineConstants.FLOAT;
                attNameMappings.put(fAttrNames[i], index);
            }
        }
        if (sAttrNames != null) {
            for (int i = 0; i < sAttrNames.length; i++) {
                index = i * 10 + DataMineConstants.NOMINAL;
                attNameMappings.put(sAttrNames[i], index);
            }
        }
    }

    /**
     * @param name Attribute name string.
     * @return Index in the corresponding feature array.
     */
    public int getIndexForAttributeName(String name) {
        if (attNameMappings != null) {
            if (attNameMappings.containsKey(name)) {
                return attNameMappings.get(name) / 10;
            } else {
                return -1;
            }
        } else {
            return -1;
        }
    }

    /**
     * @param name Attribute name string.
     * @return Feature type.
     */
    public int getTypeForAttributeName(String name) {
        if (attNameMappings != null) {
            if (attNameMappings.containsKey(name)) {
                return attNameMappings.get(name) % 10;
            } else {
                return -1;
            }
        } else {
            return -1;
        }
    }

    /**
     * Changes the attribute name, if such an attribute exists/
     *
     * @param name Old name
     * @param newName New name
     */
    public void renameAttribute(String name, String newName) {
        if (attNameMappings != null && attNameMappings.containsKey(name)) {
            int val = attNameMappings.get(name);
            int type = val % 10;
            int index = val / 10;
            attNameMappings.remove(name);
            attNameMappings.put(newName, val);
            switch (type) {
                case DataMineConstants.FLOAT: {
                    fAttrNames[index] = newName;
                    break;
                }
                case DataMineConstants.INTEGER: {
                    iAttrNames[index] = newName;
                    break;
                }
                case DataMineConstants.NOMINAL: {
                    sAttrNames[index] = newName;
                    break;
                }
            }
        }
    }

    /**
     * If it doesn't exist, both will be set to -1.
     *
     * @return A length 2 array, where the first field encodes the type and the
     * second the feature index.
     */
    public int[] getTypeAndIndexForAttrName(String name) {
        int[] result = new int[2];
        if (!attNameMappings.containsKey(name)) {
            result[0] = -1;
            result[1] = -1;
        } else {
            int codedIndex = (Integer) (attNameMappings.get(name));
            result[0] = codedIndex % 10;
            result[1] = codedIndex / 10;
        }
        return result;
    }

    /**
     * @param name Attribute name.
     * @return True if it exists, false otherwise.
     */
    public boolean hasAttName(String name) {
        return attNameMappings.containsKey(name);
    }

    /**
     * @return True if there are integer features, false otherwise.
     */
    public boolean hasIntAttr() {
        if (iAttrNames == null || iAttrNames.length == 0) {
            return false;
        } else {
            return true;
        }
    }

    /**
     * @return True if there are float features, false otherwise.
     */
    public boolean hasFloatAttr() {
        if (fAttrNames == null || fAttrNames.length == 0) {
            return false;
        } else {
            return true;
        }
    }

    /**
     * @return True if there are nominal features, false otherwise.
     */
    public boolean hasNominalAttr() {
        if (sAttrNames == null || sAttrNames.length == 0) {
            return false;
        } else {
            return true;
        }
    }

    /**
     * @return DataSet holding the identifier instances of the analyzed data.
     */
    public DataSet getIdentifiers() {
        return identifiers;
    }

    /**
     * @param identifiers DataSet containing the primary keys of the analyzed
     * data.
     */
    public void setIdentifiers(DataSet identifiers) {
        this.identifiers = identifiers;
    }

    public DataSet() {
        data = new ArrayList(DEFAULT_INIT_CAPACITY);
    }

    /**
     * @param iAttrNames Names of the integer attributes.
     * @param fAttrNames Names of the float attributes.
     * @param sAttrNames Names of the nominal attributes.
     */
    public DataSet(String[] integerAttrNames, String[] floatAttrNames,
            String[] sAttrNames) {
        this.iAttrNames = integerAttrNames;
        this.fAttrNames = floatAttrNames;
        this.sAttrNames = sAttrNames;
        data = new ArrayList<>(DEFAULT_INIT_CAPACITY);
    }

    /**
     * @param numberInstances Initial data list size.
     */
    public DataSet(int numInstances) {
        data = new ArrayList<>(numInstances);
        initCapacity = numInstances;
    }

    /**
     * @param iAttrNames Names of the integer attributes.
     * @param fAttrNames Names of the float attributes.
     * @param sAttrNames Names of the nominal attributes.
     * @param numberInstances Initial data list size.
     */
    public DataSet(String[] integerAttrNames, String[] floatAttrNames,
            String[] nominalAttrNames, int numInstances) {
        this.iAttrNames = integerAttrNames;
        this.fAttrNames = floatAttrNames;
        this.sAttrNames = nominalAttrNames;
        data = new ArrayList<>(numInstances);
        initCapacity = numInstances;
    }

    /**
     * @param iAttrNames Names of the integer attributes.
     * @param fAttrNames Names of the float attributes.
     * @param sAttrNames Names of the nominal attributes.
     * @param data vector
     */
    public DataSet(String[] integerAttrNames, String[] floatAttrNames,
            String[] nominalAttrNames, ArrayList<DataInstance> data) {
        this.iAttrNames = integerAttrNames;
        this.fAttrNames = floatAttrNames;
        this.sAttrNames = nominalAttrNames;
        this.data = data;
    }

    /**
     * @return True if empty, false otherwise.
     */
    public boolean isEmpty() {
        if ((data == null) || (data.isEmpty())) {
            return true;
        } else {
            return false;
        }
    }

    /**
     *
     * @return DataSet deep copy of the current DataSet object.
     * @throws Exception
     */
    public DataSet copy() throws Exception {
        String[] copyIntegerAttrNames = null;
        if (hasIntAttr()) {
            // Since strings are immutable in Java.
            copyIntegerAttrNames = Arrays.copyOf(iAttrNames, iAttrNames.length,
                    String[].class);
        }
        String[] copyFloatAttrNames = null;
        if (hasFloatAttr()) {
            // Since strings are immutable in Java.
            copyFloatAttrNames = Arrays.copyOf(fAttrNames, fAttrNames.length,
                    String[].class);
        }
        String[] copyNominalAttrNames = null;
        if (hasNominalAttr()) {
            // Since strings are immutable in Java.
            copyNominalAttrNames = Arrays.copyOf(sAttrNames, sAttrNames.length,
                    String[].class);
        }
        DataSet newDSet;
        if (initCapacity > 0) {
            newDSet = new DataSet(copyIntegerAttrNames, copyFloatAttrNames,
                    copyNominalAttrNames, initCapacity);
        } else {
            newDSet = new DataSet(copyIntegerAttrNames, copyFloatAttrNames,
                    copyNominalAttrNames, DEFAULT_INIT_CAPACITY);
        }
        if (data == null) {
            newDSet.data = null;
        } else {
            newDSet.data = new ArrayList<>(size());
            for (int i = 0; i < data.size(); i++) {
                newDSet.data.add((data.get(i)).copy());
            }
        }
        if (identifiers != null) {
            newDSet.setIdentifiers(identifiers.copy());
        }
        return newDSet;
    }

    /**
     * @param instance DataInstance for insertion.
     */
    public void addDataInstance(DataInstance instance) {
        if (data == null) {
            data = new ArrayList<>(DEFAULT_INIT_CAPACITY);
        }
        data.add(instance);
    }

    /**
     * @return Size of the dataset.
     */
    public int size() {
        if (data != null) {
            return data.size();
        } else {
            return 0;
        }
    }

    /**
     * @param cmet CombinedMetric object for distance calculations.
     * @throws Exception
     * @return DataInstance that is the medoid for the given data..
     */
    public DataInstance getMedoid(CombinedMetric cmet) throws Exception {
        if (data != null) {
            DataInstance centroid = getCentroid();
            DataInstance medoid = null;
            float minDistance;
            float currDistance;
            // Initialization
            medoid = data.get(0);
            try {
                minDistance = cmet.dist(centroid, medoid);
            } catch (Exception e) {
                minDistance = Float.MAX_VALUE;
            }
            for (int i = 1; i < size(); i++) {
                try {
                    currDistance = cmet.dist(centroid, data.get(i));
                } catch (Exception e) {
                    currDistance = Float.MAX_VALUE;
                }
                if (currDistance < minDistance) {
                    minDistance = currDistance;
                    medoid = data.get(i);
                }
            }
            return medoid;
        } else {
            return null;
        }
    }

    /**
     * @param centroid DataInstance Centroid instance.
     * @param cmet CombinedMetric used for the distance calculations.
     * @param res Upper index up until which to check.
     * @throws Exception
     * @return DataInstance that is the medoid for the given data.
     */
    public DataInstance getMedoidUpUntilIdex(DataInstance centroid,
            CombinedMetric cmet, int res) throws Exception {
        if (data != null) {
            DataInstance medoid = null;
            float minDistance;
            float currDistance;
            // Initialization
            medoid = data.get(0);
            try {
                minDistance = cmet.dist(centroid, medoid);
            } catch (Exception e) {
                minDistance = Float.MAX_VALUE;
            }
            for (int i = 1; i < Math.min(res, size()); i++) {
                try {
                    currDistance = cmet.dist(centroid, data.get(i));
                } catch (Exception e) {
                    currDistance = Float.MAX_VALUE;
                }
                if (currDistance < minDistance) {
                    minDistance = currDistance;
                    medoid = data.get(i);
                }
            }
            return medoid;
        } else {
            return null;
        }
    }

    /**
     * @param cmet CombinedMetric used for distance calculations.
     * @throws Exception
     * @return DataInstance that is the medoid for the given data.
     */
    public DataInstance getMedoidTo(DataInstance centroid, CombinedMetric cmet)
            throws Exception {
        if (data != null) {
            DataInstance medoid = null;
            float minDistance;
            float currDistance;
            // Initialization
            medoid = data.get(0);
            try {
                minDistance = cmet.dist(centroid, medoid);
            } catch (Exception e) {
                minDistance = Float.MAX_VALUE;
            }
            for (int i = 1; i < size(); i++) {
                try {
                    currDistance = cmet.dist(centroid, data.get(i));
                } catch (Exception e) {
                    currDistance = Float.MAX_VALUE;
                }
                if (currDistance < minDistance) {
                    minDistance = currDistance;
                    medoid = data.get(i);
                }
            }
            return medoid;
        } else {
            return null;
        }
    }

    /**
     * @return DataInstance that is the centroid for this data.
     * @throws Exception
     */
    public DataInstance getCentroid() throws Exception {
        Cluster dataCluster = Cluster.fromEntireDataset(this);
        return dataCluster.getCentroid();
    }

    /**
     * @return True if there are missing values in the data collection, false
     * otherwise.
     * @throws Exception
     */
    public boolean hasMissingValues() throws Exception {
        if (data == null) {
            return false;
        } else {
            boolean hasMissing = false;
            DataInstance instance;
            int i = 0;
            while ((!hasMissing) && (i < data.size())) {
                instance = data.get(i);
                if (instance != null) {
                    hasMissing = instance.hasMissingValues();
                }
                i++;
            }
            return hasMissing;
        }
    }
}