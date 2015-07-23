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
package feature.evaluation;

import data.representation.discrete.DiscretizedDataInstance;
import data.representation.discrete.DiscretizedDataSet;
import data.representation.util.DataMineConstants;
import java.util.ArrayList;

/**
 * This class implements the split logic for discrete feature values.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class DiscreteAttributeValueSplitter {

    // The discretized dataset that is being analyzed.
    private DiscretizedDataSet discDSet;
    // A boolean flag indicating whether to tolerate empty split branches in
    // internal evaluations. False by default.
    private boolean emptySplitBranchesTolerated = false;
    private static final int DEFAULT_GROUP_SIZE = 400;

    /**
     * @param discDSet The discretized dataset that is being analyzed.
     */
    public DiscreteAttributeValueSplitter(DiscretizedDataSet discDSet) {
        this.discDSet = discDSet;
    }

    /**
     * @return The discretized dataset that is being analyzed.
     */
    public DiscretizedDataSet getDataContext() {
        return discDSet;
    }

    /**
     * @param emptySplitBranchesTolerated A boolean flag indicating whether to
     * tolerate empty split branches in internal evaluations.
     */
    public void setEmptyToleration(boolean emptySplitBranchesTolerated) {
        this.emptySplitBranchesTolerated = emptySplitBranchesTolerated;
    }

    /**
     * Generates a split as an array of lists from the existing bucketing in the
     * discretized dataset.
     *
     * @param indexedSubset A list of integers representing a subset of the
     * data.
     * @param attType Integer representing the feature type, as defined in
     * DataMineConstants.
     * @param attIndex Index of the feature of the specified type in its feature
     * group.
     * @return Array of ArrayList objects of data indexes, representing the
     * split.
     */
    public ArrayList<Integer>[] generateIndexedSplitFromGivenDiscretization(
            ArrayList<Integer> indexedSubset, int attType, int attIndex) {
        if (discDSet == null || discDSet.isEmpty()) {
            return null;
        }
        if (indexedSubset == null || indexedSubset.isEmpty()) {
            return null;
        }
        if (attType == DataMineConstants.INTEGER) {
            // An integer feature.
            DiscretizedDataInstance instance;
            int[][] intIntervalDivisions = discDSet.getIntIntervalDivisions();
            if (intIntervalDivisions != null
                    && intIntervalDivisions.length > attIndex) {
                // The last in the bucket delimiters is the Integer.MAX_VALUE, 
                // so we can ignore it.
                ArrayList<Integer>[] split =
                        new ArrayList[
                                intIntervalDivisions[attIndex].length - 1];
                for (int i = 0; i < split.length; i++) {
                    split[i] = new ArrayList<>(DEFAULT_GROUP_SIZE);
                }
                for (int i = 0; i < indexedSubset.size(); i++) {
                    instance = discDSet.data.get(indexedSubset.get(i));
                    split[instance.integerIndexes[attIndex]].add(
                            new Integer(i));
                }
                if (emptySplitBranchesTolerated) {
                    return split;
                }
                // Check for empty splits and clear them out.
                int numNonEmptySplits = 0;
                for (int i = 0; i < split.length; i++) {
                    if (split[i].size() > 0) {
                        numNonEmptySplits++;
                    }
                }
                if (numNonEmptySplits != split.length) {
                    ArrayList<Integer>[] tempSplit =
                            new ArrayList[numNonEmptySplits];
                    int nonEmptyIndex = -1;
                    for (int i = 0; i < split.length; i++) {
                        if (split[i].size() > 0) {
                            tempSplit[++nonEmptyIndex] = split[i];
                        }
                    }
                    split = tempSplit;
                }
                return split;
            } else {
                return null;
            }
        } else if (attType == DataMineConstants.FLOAT) {
            DiscretizedDataInstance instance;
            float[][] floatIntervalDivisions =
                    discDSet.getFloatIntervalDivisions();
            if (floatIntervalDivisions != null
                    && floatIntervalDivisions.length > attIndex) {
                // The last in the bucket delimiters is the Float.MAX_VALUE, 
                // so we can ignore it.
                ArrayList<Integer>[] split =
                        new ArrayList[
                                floatIntervalDivisions[attIndex].length - 1];
                for (int i = 0; i < split.length; i++) {
                    split[i] = new ArrayList<>(DEFAULT_GROUP_SIZE);
                }
                for (int i = 0; i < indexedSubset.size(); i++) {
                    instance = discDSet.data.get(indexedSubset.get(i));
                    split[instance.floatIndexes[attIndex]].add(new Integer(i));
                }
                if (emptySplitBranchesTolerated) {
                    return split;
                }
                // Check for empty splits and clear them out.
                int numNonEmptySplits = 0;
                for (int i = 0; i < split.length; i++) {
                    if (split[i].size() > 0) {
                        numNonEmptySplits++;
                    }
                }
                if (numNonEmptySplits != split.length) {
                    ArrayList<Integer>[] tempSplit =
                            new ArrayList[numNonEmptySplits];
                    int nonEmptyIndex = -1;
                    for (int i = 0; i < split.length; i++) {
                        if (split[i].size() > 0) {
                            tempSplit[++nonEmptyIndex] = split[i];
                        }
                    }
                    split = tempSplit;
                }
                return split;
            } else {
                return null;
            }
        } else {
            DiscretizedDataInstance instance;
            ArrayList<String>[] vocabularies =
                    discDSet.getNominalVocabularies();
            if (vocabularies != null && vocabularies.length > attIndex) {
                ArrayList<Integer>[] split =
                        new ArrayList[vocabularies[attIndex].size()];
                for (int i = 0; i < split.length; i++) {
                    split[i] = new ArrayList<>(DEFAULT_GROUP_SIZE);
                }
                for (int i = 0; i < indexedSubset.size(); i++) {
                    instance = discDSet.data.get(indexedSubset.get(i));
                    split[instance.nominalIndexes[attIndex]].add(
                            new Integer(i));
                }
                if (emptySplitBranchesTolerated) {
                    return split;
                }
                // Check for empty splits and clear them out.
                int numNonEmptySplits = 0;
                for (int i = 0; i < split.length; i++) {
                    if (split[i].size() > 0) {
                        numNonEmptySplits++;
                    }
                }
                if (numNonEmptySplits != split.length) {
                    ArrayList<Integer>[] tempSplit =
                            new ArrayList[numNonEmptySplits];
                    int nonEmptyIndex = -1;
                    for (int i = 0; i < split.length; i++) {
                        if (split[i].size() > 0) {
                            tempSplit[++nonEmptyIndex] = split[i];
                        }
                    }
                    split = tempSplit;
                }
                return split;
            } else {
                return null;
            }
        }
    }

    /**
     * Generates a split as an array of lists from the existing bucketing in the
     * discretized dataset.
     *
     * @param attType Integer representing the feature type, as defined in
     * DataMineConstants.
     * @param attIndex Index of the feature of the specified type in its feature
     * group.
     * @return Array of ArrayList objects of data indexes, representing the
     * split.
     */
    public ArrayList<Integer>[] generateIndexedSplitOnAttribute(int attType,
            int attIndex) {
        if (discDSet == null || discDSet.isEmpty()) {
            return null;
        }
        if (attType == DataMineConstants.INTEGER) {
            DiscretizedDataInstance instance;
            int[][] intIntervalDivisions = discDSet.getIntIntervalDivisions();
            if (intIntervalDivisions != null
                    && intIntervalDivisions.length > attIndex) {
                // The last in the bucket delimiters is the Integer.MAX_VALUE, 
                // so we can ignore it.
                ArrayList<Integer>[] split =
                        new ArrayList[
                                intIntervalDivisions[attIndex].length - 1];
                for (int i = 0; i < split.length; i++) {
                    split[i] = new ArrayList<>(DEFAULT_GROUP_SIZE);
                }
                for (int i = 0; i < discDSet.size(); i++) {
                    instance = discDSet.data.get(i);
                    split[instance.integerIndexes[attIndex]].add(
                            new Integer(i));
                }
                if (emptySplitBranchesTolerated) {
                    return split;
                }
                // Check for empty splits and clear them out.
                int numNonEmptySplits = 0;
                for (int i = 0; i < split.length; i++) {
                    if (split[i].size() > 0) {
                        numNonEmptySplits++;
                    }
                }
                if (numNonEmptySplits != split.length) {
                    ArrayList<Integer>[] tempSplit =
                            new ArrayList[numNonEmptySplits];
                    int nonEmptyIndex = -1;
                    for (int i = 0; i < split.length; i++) {
                        if (split[i].size() > 0) {
                            tempSplit[++nonEmptyIndex] = split[i];
                        }
                    }
                    split = tempSplit;
                }
                return split;
            } else {
                return null;
            }
        } else if (attType == DataMineConstants.FLOAT) {
            DiscretizedDataInstance instance;
            float[][] floatIntervalDivisions =
                    discDSet.getFloatIntervalDivisions();
            if (floatIntervalDivisions != null
                    && floatIntervalDivisions.length > attIndex) {
                // The last in the bucket delimiters is the Float.MAX_VALUE, 
                // so we can ignore it.
                ArrayList<Integer>[] split =
                        new ArrayList[
                                floatIntervalDivisions[attIndex].length - 1];
                for (int i = 0; i < split.length; i++) {
                    split[i] = new ArrayList<>(DEFAULT_GROUP_SIZE);
                }
                for (int i = 0; i < discDSet.size(); i++) {
                    instance = discDSet.data.get(i);
                    split[instance.floatIndexes[attIndex]].add(
                            new Integer(i));
                }
                if (emptySplitBranchesTolerated) {
                    return split;
                }
                // Check for empty splits and clear them out.
                int numNonEmptySplits = 0;
                for (int i = 0; i < split.length; i++) {
                    if (split[i].size() > 0) {
                        numNonEmptySplits++;
                    }
                }
                if (numNonEmptySplits != split.length) {
                    ArrayList<Integer>[] tempSplit =
                            new ArrayList[numNonEmptySplits];
                    int nonEmptyIndex = -1;
                    for (int i = 0; i < split.length; i++) {
                        if (split[i].size() > 0) {
                            tempSplit[++nonEmptyIndex] = split[i];
                        }
                    }
                    split = tempSplit;
                }
                return split;
            } else {
                return null;
            }
        } else {
            DiscretizedDataInstance instance;
            ArrayList<String>[] vocabularies =
                    discDSet.getNominalVocabularies();
            if (vocabularies != null && vocabularies.length > attIndex) {
                ArrayList<Integer>[] split =
                        new ArrayList[vocabularies[attIndex].size()];
                for (int i = 0; i < split.length; i++) {
                    split[i] = new ArrayList<>(DEFAULT_GROUP_SIZE);
                }
                for (int i = 0; i < discDSet.size(); i++) {
                    instance = discDSet.data.get(i);
                    split[instance.integerIndexes[attIndex]].add(
                            new Integer(i));
                }
                if (emptySplitBranchesTolerated) {
                    return split;
                }
                // Check for empty splits and clear them out.
                int numNonEmptySplits = 0;
                for (int i = 0; i < split.length; i++) {
                    if (split[i].size() > 0) {
                        numNonEmptySplits++;
                    }
                }
                if (numNonEmptySplits != split.length) {
                    ArrayList<Integer>[] tempSplit =
                            new ArrayList[numNonEmptySplits];
                    int nonEmptyIndex = -1;
                    for (int i = 0; i < split.length; i++) {
                        if (split[i].size() > 0) {
                            tempSplit[++nonEmptyIndex] = split[i];
                        }
                    }
                    split = tempSplit;
                }
                return split;
            } else {
                return null;
            }
        }
    }

    /**
     * Generates a split as an array of discretized datasets.
     *
     * @param attType Integer representing the feature type, as defined in
     * DataMineConstants.
     * @param attIndex Index of the feature of the specified type in its feature
     * group.
     * @return Array of ArrayList objects of data indexes, representing the
     * split.
     */
    public DiscretizedDataSet[] generateSplitOnAttribute(int attType,
            int attIndex) {
        if (discDSet == null || discDSet.isEmpty()) {
            return null;
        }
        if (attType == DataMineConstants.INTEGER) {
            DiscretizedDataInstance instance;
            int[][] intIntervalDivisions = discDSet.getIntIntervalDivisions();
            if (intIntervalDivisions != null
                    && intIntervalDivisions.length > attIndex) {
                // The last in the bucket delimiters is the Integer.MAX_VALUE, 
                // so we can ignore it.
                DiscretizedDataSet[] split =
                        new DiscretizedDataSet[
                                intIntervalDivisions[attIndex].length - 1];
                for (int i = 0; i < split.length; i++) {
                    split[i] = discDSet.cloneDefinition();
                    split[i].data = new ArrayList<>(DEFAULT_GROUP_SIZE);
                }
                for (int i = 0; i < discDSet.size(); i++) {
                    instance = discDSet.data.get(i);
                    split[instance.integerIndexes[attIndex]].data.add(instance);
                }
                if (emptySplitBranchesTolerated) {
                    return split;
                }
                // Check for empty splits and clear them out.
                int numNonEmptySplits = 0;
                for (int i = 0; i < split.length; i++) {
                    if (split[i].size() > 0) {
                        numNonEmptySplits++;
                    }
                }
                if (numNonEmptySplits != split.length) {
                    DiscretizedDataSet[] tempSplit =
                            new DiscretizedDataSet[numNonEmptySplits];
                    int nonEmptyIndex = -1;
                    for (int i = 0; i < split.length; i++) {
                        if (split[i].size() > 0) {
                            tempSplit[++nonEmptyIndex] = split[i];
                        }
                    }
                    split = tempSplit;
                }
                return split;
            } else {
                return null;
            }
        } else if (attType == DataMineConstants.FLOAT) {
            DiscretizedDataInstance instance;
            float[][] floatIntervalDivisions =
                    discDSet.getFloatIntervalDivisions();
            if (floatIntervalDivisions != null
                    && floatIntervalDivisions.length > attIndex) {
                // The last in the bucket delimiters is the Float.MAX_VALUE, 
                // so we can ignore it.
                DiscretizedDataSet[] split =
                        new DiscretizedDataSet[
                                floatIntervalDivisions[attIndex].length - 1];
                for (int i = 0; i < split.length; i++) {
                    split[i] = discDSet.cloneDefinition();
                    split[i].data = new ArrayList<>(DEFAULT_GROUP_SIZE);
                }
                for (int i = 0; i < discDSet.size(); i++) {
                    instance = discDSet.data.get(i);
                    split[instance.floatIndexes[attIndex]].data.add(instance);
                }
                if (emptySplitBranchesTolerated) {
                    return split;
                }
                // Check for empty splits and clear them out.
                int numNonEmptySplits = 0;
                for (int i = 0; i < split.length; i++) {
                    if (split[i].size() > 0) {
                        numNonEmptySplits++;
                    }
                }
                if (numNonEmptySplits != split.length) {
                    DiscretizedDataSet[] tempSplit =
                            new DiscretizedDataSet[numNonEmptySplits];
                    int nonEmptyIndex = -1;
                    for (int i = 0; i < split.length; i++) {
                        if (split[i].size() > 0) {
                            tempSplit[++nonEmptyIndex] = split[i];
                        }
                    }
                    split = tempSplit;
                }
                return split;
            } else {
                return null;
            }
        } else {
            DiscretizedDataInstance instance;
            ArrayList<String>[] vocabularies =
                    discDSet.getNominalVocabularies();
            if (vocabularies != null && vocabularies.length > attIndex) {
                DiscretizedDataSet[] split =
                        new DiscretizedDataSet[vocabularies[attIndex].size()];
                for (int i = 0; i < split.length; i++) {
                    split[i] = discDSet.cloneDefinition();
                    split[i].data = new ArrayList<>(DEFAULT_GROUP_SIZE);
                }
                for (int i = 0; i < discDSet.size(); i++) {
                    instance = discDSet.data.get(i);
                    split[instance.nominalIndexes[attIndex]].data.add(instance);
                }
                if (emptySplitBranchesTolerated) {
                    return split;
                }
                // Check for empty splits and clear them out.
                int numNonEmptySplits = 0;
                for (int i = 0; i < split.length; i++) {
                    if (split[i].size() > 0) {
                        numNonEmptySplits++;
                    }
                }
                if (numNonEmptySplits != split.length) {
                    DiscretizedDataSet[] tempSplit =
                            new DiscretizedDataSet[numNonEmptySplits];
                    int nonEmptyIndex = -1;
                    for (int i = 0; i < split.length; i++) {
                        if (split[i].size() > 0) {
                            tempSplit[++nonEmptyIndex] = split[i];
                        }
                    }
                    split = tempSplit;
                }
                return split;
            } else {
                return null;
            }
        }
    }
}
