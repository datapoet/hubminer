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
package data.representation.discrete.tranform;

import data.representation.DataSet;
import data.representation.discrete.DiscretizedDataSet;
import data.representation.sparse.BOWDataSet;
import java.util.ArrayList;
import util.AuxSort;

/**
 * A supervised data discretization approach based on entropy.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class EntropyMDLDiscretizer {

    public static final int INT = 0;
    public static final int FLOAT = 1;
    private DataSet dset;
    private DiscretizedDataSet dsetDisc;
    private int numCategories = 2;

    public EntropyMDLDiscretizer() {
    }

    /**
     * @param dset DataSet to discretize.
     * @param dsetDisc DiscretizedDataSet that is the target.
     * @param numCategories Number of categories in the data.
     */
    public EntropyMDLDiscretizer(DataSet dset, DiscretizedDataSet dsetDisc,
            int numCategories) {
        this.dset = dset;
        this.dsetDisc = dsetDisc;
        this.numCategories = numCategories;
    }

    /**
     * @return DiscretizedDataSet object.
     */
    public DiscretizedDataSet getDiscretizedDataSet() {
        return dsetDisc;
    }

    /**
     * @return DataSet object that is being discretized.
     */
    public DataSet getOriginalDataSet() {
        return dset;
    }

    /**
     * @param dsetDisc DiscretizedDataSet object that is the target for
     * discretization.
     */
    public void setDiscretizedDataSet(DiscretizedDataSet dsetDisc) {
        this.dsetDisc = dsetDisc;
    }

    /**
     * @param dset DataSet object that is being discretized.
     */
    public void setDataSet(DataSet dset) {
        this.dset = dset;
    }

    /**
     * @param numCategories Integer that is the number of categories in the
     * data.
     */
    public void setNumCategories(int numCategories) {
        this.numCategories = numCategories;
    }

    /**
     * Performs a discretization of the specified feature into an automatically
     * determined number of buckets.
     *
     * @param floatOrInt Integer that is 0 if the target feature is integer and
     * 1 if the target feature is float.
     * @param index Index of the feature in the respective feature array.
     * @throws Exception
     */
    public void discretize(int floatOrInt, int index) throws Exception {
        if (floatOrInt == INT) {
            discretizeInt(index);
        } else {
            discretizeFloat(index);
        }
    }

    /**
     * Performs a discretization of the specified feature into two buckets.
     *
     * @param floatOrInt Integer that is 0 if the target feature is integer and
     * 1 if the target feature is float.
     * @param index Index of the feature in the respective feature array.
     * @throws Exception
     */
    public void discretizeBinary(int floatOrInt, int index) throws Exception {
        if (floatOrInt == INT) {
            discretizeIntBinary(index);
        } else {
            discretizeFloatBinary(index);
        }
    }

    /**
     * Discretize an integer feature.
     *
     * @param index Index of the integer feature to discretize.
     * @throws Exception
     */
    private void discretizeInt(int index) throws Exception {
        // In trivial cases no discretization is needed.
        if (dset.size() > 1) {
            int[] intValues = new int[dset.size()];
            int[] categories = new int[dset.size()];
            for (int i = 0; i < dset.size(); i++) {
                intValues[i] = dset.data.get(i).iAttr[index];
            }
            // We sort by value and re-arrange the categories of the points
            // accordingly.
            int[] rearrange = AuxSort.sortIndexedValue(intValues, false);
            for (int i = 0; i < dset.size(); i++) {
                categories[i] = dset.data.get(rearrange[i]).getCategory();
            }
            ArrayList<Integer> splitPoints = new ArrayList<>(10);
            // Perform a recursive discretization call. The result is returned
            // in the splitPoints list.
            discretizeInternal(categories, 0, dset.size() - 1, splitPoints);
            AuxSort.sortIntArrayList(splitPoints, false);
            int[][] intIntervalDivisions = dsetDisc.getIntIntervalDivisions();
            intIntervalDivisions[index] = new int[splitPoints.size() + 2];
            intIntervalDivisions[index][0] = Integer.MIN_VALUE;
            intIntervalDivisions[index][
                    intIntervalDivisions[index].length - 1] = Integer.MAX_VALUE;
            for (int i = 1; i < intIntervalDivisions[index].length - 1; i++) {
                intIntervalDivisions[index][i] =
                        intValues[splitPoints.get(i - 1)];
            }
            dsetDisc.setIntIntervalDivisions(intIntervalDivisions);
        }
    }

    /**
     * Discretize an integer feature into two buckets.
     *
     * @param index Index of the integer feature to discretize.
     * @throws Exception
     */
    private void discretizeIntBinary(int index) throws Exception {
        if (dset.size() > 1) {
            int[] intValues = new int[dset.size()];
            int[] categories = new int[dset.size()];
            for (int i = 0; i < dset.size(); i++) {
                intValues[i] = dset.data.get(i).iAttr[index];
            }
            // We sort by value and re-arrange the categories of the points
            // accordingly.
            int[] rearrange = AuxSort.sortIndexedValue(intValues, false);
            for (int i = 0; i < dset.size(); i++) {
                categories[i] = dset.data.get(rearrange[i]).getCategory();
            }
            ArrayList<Integer> splitPoints = new ArrayList<>(10);
            // Perform a discretization call.
            discretizeInternal(categories, 0, dset.size() - 1, splitPoints);
            AuxSort.sortIntArrayList(splitPoints, false);
            int[][] intIntervalDivisions = dsetDisc.getIntIntervalDivisions();
            intIntervalDivisions[index] = new int[splitPoints.size() + 2];
            intIntervalDivisions[index][0] = Integer.MIN_VALUE;
            intIntervalDivisions[index][
                    intIntervalDivisions[index].length - 1] = Integer.MAX_VALUE;
            for (int i = 1; i < intIntervalDivisions[index].length - 1; i++) {
                intIntervalDivisions[index][i] =
                        intValues[splitPoints.get(i - 1)];
            }
            dsetDisc.setIntIntervalDivisions(intIntervalDivisions);
        }
    }

    /**
     * Discretize a float feature.
     *
     * @param index Index of the float feature to discretize.
     * @throws Exception
     */
    private void discretizeFloat(int index) throws Exception {
        if (dset.size() > 1) {
            float[] floatValues = new float[dset.size()];
            int[] categories = new int[dset.size()];
            for (int i = 0; i < dset.size(); i++) {
                floatValues[i] = dset.data.get(i).fAttr[index];
            }
            // We sort by value and re-arrange the categories of the points
            // accordingly.
            int[] rearrange = AuxSort.sortIndexedValue(floatValues, false);
            for (int i = 0; i < dset.size(); i++) {
                categories[i] = dset.data.get(rearrange[i]).getCategory();
            }
            // Perform a recursive discretization call. The result is returned
            // in the splitPoints list.
            ArrayList<Integer> splitPoints = new ArrayList<>(10);
            discretizeInternal(categories, 0, dset.size() - 1, splitPoints);
            AuxSort.sortIntArrayList(splitPoints, false);
            float[][] floatIntervalDivisions =
                    dsetDisc.getFloatIntervalDivisions();
            floatIntervalDivisions[index] = new float[splitPoints.size() + 2];
            floatIntervalDivisions[index][0] = -Float.MAX_VALUE;
            floatIntervalDivisions[index][
                    floatIntervalDivisions[index].length - 1] = Float.MAX_VALUE;
            for (int i = 1; i < floatIntervalDivisions[index].length - 1; i++) {
                floatIntervalDivisions[index][i] =
                        floatValues[splitPoints.get(i - 1)];
            }
            dsetDisc.setFloatIntervalDivisions(floatIntervalDivisions);
        }
    }

    /**
     * Discretize a float feature into two buckets.
     *
     * @param index Index of the float feature to discretize.
     * @throws Exception
     */
    private void discretizeFloatBinary(int index) throws Exception {
        if (dset.size() > 1) {
            float[] floatValues = new float[dset.size()];
            int[] categories = new int[dset.size()];
            for (int i = 0; i < dset.size(); i++) {
                floatValues[i] = dset.data.get(i).fAttr[index];
            }
            // We sort by value and re-arrange the categories of the points
            // accordingly.
            int[] rearrange = AuxSort.sortIndexedValue(floatValues, false);
            for (int i = 0; i < dset.size(); i++) {
                categories[i] = dset.data.get(rearrange[i]).getCategory();
            }
            // Perform a discretization call.
            ArrayList<Integer> splitPoints = new ArrayList<>(10);
            discretizeInternalBinary(categories, 0, dset.size() - 1,
                    splitPoints);
            AuxSort.sortIntArrayList(splitPoints, false);
            float[][] floatIntervalDivisions =
                    dsetDisc.getFloatIntervalDivisions();
            floatIntervalDivisions[index] = new float[splitPoints.size() + 2];
            floatIntervalDivisions[index][0] = -Float.MAX_VALUE;
            floatIntervalDivisions[index][
                    floatIntervalDivisions[index].length - 1] = Float.MAX_VALUE;
            for (int i = 1; i < floatIntervalDivisions[index].length - 1; i++) {
                floatIntervalDivisions[index][i] =
                        floatValues[splitPoints.get(i - 1)];
            }
            dsetDisc.setFloatIntervalDivisions(floatIntervalDivisions);
        }
    }

    /**
     * A recursive call for performing discretization that keeps track of all
     * previously found split points in the list that is passed on.
     *
     * @param categories Category array.
     * @param beginIndex Lower index.
     * @param endIndex Upper index.
     * @param splitPoints ArrayList of integer indexes of current split points.
     */
    private void discretizeInternal(int[] categories, int beginIndex,
            int endIndex, ArrayList<Integer> splitPoints) {
        if (endIndex - beginIndex < 1) {
            return;
        }
        // Find the potential best splitpoint.
        float fullEntropy = 0;
        int fullClassNum = 0;
        int firstClassNum = 0;
        int secondClassNum = 0;
        // Current entropy values over iterations.
        float currEntropyFirst;
        float currEntropySecond;
        float currInfo;
        int currSpIndex;
        // Not the minimal entropy, but the entropy of the first segment when
        // the mininal information is used to specify class.
        float minEntropyFirst;
        // The same, but for the second segment.
        float minEntropySecond = 0;
        // Minimum overall information.
        float minInfo;
        // The split index that achieves the best split.
        int minSpIndex = beginIndex + 1;
        float ratio;
        float rangeSize;
        //first step with some initializations
        int[] categoryFreqsFirst = new int[numCategories];
        int[] categoryFreqsSecond = new int[numCategories];
        categoryFreqsFirst[categories[beginIndex]]++;
        // Calculate the full entropy.
        for (int i = beginIndex; i <= endIndex; i++) {
            // This array is used here instead of defining a separate one. It
            // will be adjusted later to represent the frequencies in the second
            // part of the split.
            categoryFreqsSecond[categories[i]]++;
        }
        rangeSize = endIndex - beginIndex + 1;
        for (int i = 0; i < numCategories; i++) {
            if (categoryFreqsSecond[i] > 0) {
                fullClassNum++;
                ratio = ((float) categoryFreqsSecond[i]) / rangeSize;
                if (ratio > 0) {
                    fullEntropy -= ratio * Math.log(ratio);
                }
            }
        }
        // Now give back the extra class occurrence from first index.
        categoryFreqsSecond[categories[beginIndex]]--;
        // Since the first part of the split currently has just one element, its
        // entropy is zero.
        minEntropyFirst = 0;
        rangeSize = endIndex - beginIndex;
        for (int i = 0; i < numCategories; i++) {
            if (categoryFreqsSecond[i] > 0) {
                ratio = ((float) categoryFreqsSecond[i]) / rangeSize;
                if (ratio > 0) {
                    minEntropySecond -= ratio * Math.log(ratio);
                }
            }
        }
        // No need to divide by the total number of elements, as it does not
        // affect the comparisons.
        minInfo = rangeSize * minEntropySecond;
        // Iterating through the array from left to right.
        if (endIndex - beginIndex >= 2) {
            for (int i = beginIndex + 2; i <= endIndex; i++) {
                // First update the frequencies.
                categoryFreqsFirst[categories[i]]++;
                categoryFreqsSecond[categories[i]]--;
                // Reinitialize variables.
                currEntropyFirst = 0;
                currEntropySecond = 0;
                currInfo = 0;
                currSpIndex = i;
                // Calculate the entropies.
                // First interval.
                rangeSize = i - beginIndex;
                for (int j = 0; j < numCategories; j++) {
                    if (categoryFreqsFirst[j] > 0) {
                        ratio = ((float) categoryFreqsFirst[j]) / rangeSize;
                        if (ratio > 0) {
                            currEntropyFirst -= ratio * Math.log(ratio);
                        }
                    }
                }
                // Second interval.
                rangeSize = endIndex - i + 1;
                for (int j = 0; j < numCategories; j++) {
                    if (categoryFreqsSecond[j] > 0) {
                        ratio = ((float) categoryFreqsSecond[j]) / rangeSize;
                        if (ratio > 0) {
                            currEntropySecond -= ratio * Math.log(ratio);
                        }
                    }
                }
                // Calculate the information value of the split. Here it is not
                // normalized as there is no need for comparisons.
                currInfo = (i - beginIndex) * currEntropyFirst
                        + (endIndex - i + 1) * currEntropySecond;
                if (currInfo < minInfo) {
                    minInfo = currInfo;
                    minEntropyFirst = currEntropyFirst;
                    minEntropySecond = currEntropySecond;
                    minSpIndex = currSpIndex;
                }
            }
        }
        // Calculate if the split is feasible by checking the number of
        // categories in each subinterval.
        for (int i = 0; i < numCategories; i++) {
            categoryFreqsFirst[i] = 0;
            categoryFreqsSecond[i] = 0;
        }
        for (int i = beginIndex; i < minSpIndex; i++) {
            categoryFreqsFirst[categories[i]]++;
        }
        for (int i = minSpIndex; i <= endIndex; i++) {
            categoryFreqsSecond[categories[i]]++;
        }
        for (int i = 0; i < numCategories; i++) {
            if (categoryFreqsFirst[i] > 0) {
                firstClassNum++;
            }
            if (categoryFreqsSecond[i] > 0) {
                secondClassNum++;
            }
        }
        // Feasibility criterion is based on the MDL principle.
        boolean feasibleSplit = ((endIndex - beginIndex + 1) * fullEntropy
                - (minSpIndex - beginIndex) * minEntropyFirst
                - (endIndex - minSpIndex + 1) * minEntropySecond
                - Math.log(endIndex - beginIndex)
                + fullClassNum * fullEntropy
                - firstClassNum * minEntropyFirst
                - secondClassNum * minEntropySecond
                - Math.log(Math.pow(3, fullClassNum) - 2)) > 0;
        // If the split is feasible, make a recursive call to further splits.
        if (feasibleSplit) {
            discretizeInternal(categories, beginIndex, minSpIndex - 1,
                    splitPoints);
            splitPoints.add(minSpIndex);
            discretizeInternal(categories, minSpIndex, endIndex, splitPoints);
        }
    }

    /**
     * A recursive call for performing discretization that keeps track of all
     * previously found split points in the list that is passed on.
     *
     * @param categories Category array.
     * @param beginIndex Lower index.
     * @param endIndex Upper index.
     * @param splitPoints ArrayList of integer indexes of current split points.
     */
    public void discretizeInternalBinary(int[] categories, int beginIndex,
            int endIndex, ArrayList<Integer> splitPoints) {
        if (endIndex - beginIndex < 1) {
            return;
        }
        float currEntropyFirst;
        float currEntropySecond;
        float currInfo;
        int currSpIndex;
        float minEntropySecond = 0;
        float minInfo;
        int minSpIndex = beginIndex + 1;
        float ratio;
        float rangeSize;
        // First step with initializations.
        int[] categoryFreqsFirst = new int[numCategories];
        int[] categoryFreqsSecond = new int[numCategories];
        categoryFreqsFirst[categories[beginIndex]]++;
        for (int i = beginIndex; i <= endIndex; i++) {
            // This array is used here instead of defining a separate one. It
            // will be adjusted later to represent the frequencies in the second
            // part of the split.
            categoryFreqsSecond[categories[i]]++;
        }
        rangeSize = endIndex - beginIndex;
        for (int i = 0; i < numCategories; i++) {
            if (categoryFreqsSecond[i] > 0) {
                ratio = ((float) categoryFreqsSecond[i]) / rangeSize;
                if (ratio > 0) {
                    minEntropySecond -= ratio * Math.log(ratio);
                }
            }
        }
        // Now give back the extra class occurrence from first index.
        categoryFreqsSecond[categories[beginIndex]]--;
        // No need to divide by the total number of elements, as it does not
        // affect the comparisons.
        minInfo = rangeSize * minEntropySecond;
        if (endIndex - beginIndex >= 2) {
            for (int i = beginIndex + 2; i <= endIndex; i++) {
                // Update the frequencies.
                categoryFreqsFirst[categories[i]]++;
                categoryFreqsSecond[categories[i]]--;
                // Reinitialize variables.
                currEntropyFirst = 0;
                currEntropySecond = 0;
                currSpIndex = i;
                // Calculate the entropies of the current split.
                // First interval.
                rangeSize = i - beginIndex;
                for (int j = 0; j < numCategories; j++) {
                    if (categoryFreqsFirst[j] > 0) {
                        ratio = ((float) categoryFreqsFirst[j]) / rangeSize;
                        if (ratio > 0) {
                            currEntropyFirst -= ratio * Math.log(ratio);
                        }
                    }
                }
                // Second interval.
                rangeSize = endIndex - i + 1;
                for (int j = 0; j < numCategories; j++) {
                    if (categoryFreqsSecond[j] > 0) {
                        ratio = ((float) categoryFreqsSecond[j]) / rangeSize;
                        if (ratio > 0) {
                            currEntropySecond -= ratio * Math.log(ratio);
                        }
                    }
                }
                // Calculate the information value of the split. Here it is not
                // normalized as there is no need for comparisons.
                currInfo = (i - beginIndex) * currEntropyFirst
                        + (endIndex - i + 1) * currEntropySecond;
                if (currInfo < minInfo) {
                    minInfo = currInfo;
                    minSpIndex = currSpIndex;
                }
            }
        }
        splitPoints.add(minSpIndex);
    }

    /**
     * Initialize the arrays that will hold the bucket definitions of float
     * interval values.
     */
    public void initializeFloatIntervalSplits() {
        if (dset.fAttrNames != null && dset.fAttrNames.length > 0) {
            float[][] floatIntervalDivisions =
                    new float[dset.fAttrNames.length][];
            dsetDisc.setFloatIntervalDivisions(floatIntervalDivisions);
        }
    }

    /**
     * Initialize the arrays that will hold the bucket definitions of integer
     * interval values.
     */
    public void initializeIntIntervalSplits() {
        if (dset.iAttrNames != null && dset.iAttrNames.length > 0) {
            int[][] intIntervalDivisions = new int[dset.iAttrNames.length][];
            dsetDisc.setIntIntervalDivisions(intIntervalDivisions);
        }
    }

    /**
     * Initialize hashes for nominal features.
     */
    public void makeEmptyNominalHashes() {
        if (dset.sAttrNames != null && dset.sAttrNames.length != 0) {
            int[][] sizeIncrements = new int[dset.sAttrNames.length][2];
            for (int i = 0; i < sizeIncrements.length; i++) {
                sizeIncrements[i][0] = 3000;
                sizeIncrements[i][1] = 500;
            }
            dsetDisc.initHashes(sizeIncrements);
        }
    }

    /**
     * Discretize all integer features.
     *
     * @throws Exception
     */
    public void discretizeAllInt() throws Exception {
        if (dset.iAttrNames != null && dset.iAttrNames.length != 0) {
            for (int i = 0; i < dset.iAttrNames.length; i++) {
                discretize(EntropyMDLDiscretizer.INT, i);
            }
        }
    }

    /**
     * Discretize all binary features.
     *
     * @throws Exception
     */
    public void discretizeAllFloat() throws Exception {
        if (dset.fAttrNames != null && dset.fAttrNames.length != 0) {
            for (int i = 0; i < dset.fAttrNames.length; i++) {
                discretize(EntropyMDLDiscretizer.FLOAT, i);
            }
        }
    }

    /**
     * Perform binary discretization for all integer features.
     *
     * @throws Exception
     */
    public void discretizeAllIntBinary() throws Exception {
        if (dset.iAttrNames != null && dset.iAttrNames.length != 0) {
            for (int i = 0; i < dset.iAttrNames.length; i++) {
                discretizeBinary(EntropyMDLDiscretizer.INT, i);
            }
        }
    }

    /**
     * Perform binary discretization for all float features.
     *
     * @throws Exception
     */
    public void discretizeAllFloatBinary() throws Exception {
        if (dset.fAttrNames != null && dset.fAttrNames.length != 0) {
            for (int i = 0; i < dset.fAttrNames.length; i++) {
                discretizeBinary(EntropyMDLDiscretizer.FLOAT, i);
            }
        }
    }

    /**
     * Discretize all features. Automatically determine the number of buckets.
     *
     * @throws Exception
     */
    public void discretizeAll() throws Exception {
        initializeFloatIntervalSplits();
        initializeIntIntervalSplits();
        makeEmptyNominalHashes();
        discretizeAllInt();
        discretizeAllFloat();
    }

    /**
     * Perform binary discretization of all features.
     *
     * @throws Exception
     */
    public void discretizeAllBinary() throws Exception {
        if (dset instanceof BOWDataSet) {
            return;
        }
        initializeFloatIntervalSplits();
        initializeIntIntervalSplits();
        makeEmptyNominalHashes();
        discretizeAllIntBinary();
        discretizeAllFloatBinary();
    }
}
