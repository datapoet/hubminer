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
package learning.supervised.methods.discrete.trees;

import data.representation.discrete.DiscretizedDataInstance;
import data.representation.discrete.DiscretizedDataSet;
import data.representation.util.DataMineConstants;
import feature.evaluation.DiscreteAttributeValueSplitter;
import feature.evaluation.Info;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import learning.supervised.DiscreteCategory;
import learning.supervised.DiscreteClassifier;
import learning.supervised.evaluation.ValidateableInterface;
import util.BasicMathUtil;

/**
 * This class implements the standard ID3 decision tree classification method.
 * It is a basic tree method.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class DCT_ID3 extends DiscreteClassifier implements Serializable {
    
    private static final long serialVersionUID = 1L;

    // The root node.
    private DecisionTreeNode root = null;
    // Boolean arrays controlling which features to consider for the splits.
    private boolean[] acceptableFloat;
    private boolean[] acceptableInt;
    private boolean[] acceptableNominal;
    // Number of classes in the data.
    private int numClasses;
    // The overall majority class in the data.
    private int generalMajorityClass;
    // The overall 
    private float[] classPriors;
    private int totalNumAtt;
    private int currDepth = 0;

    @Override
    public String getName() {
        return "ID3";
    }

    
    /**
     * The default constructor.
     */
    public DCT_ID3() {
    }

    
    /**
     * Initialization.
     *
     * @param discDSet DiscretizedDataSet that is the training data.
     */
    public DCT_ID3(DiscretizedDataSet discDSet) {
        setDataType(discDSet);
        generateClassesFromDataType();
    }

    
    /**
     * Initialization.
     *
     * @param discDSet DiscretizedDataSet that is the training data context.
     * @param dataClasses DiscreteCategory[] that is the training data.
     */
    public DCT_ID3(DiscretizedDataSet discDSet, DiscreteCategory[] dataClasses) {
        setClasses(dataClasses);
        setDataType(discDSet);
    }

    
    @Override
    public ValidateableInterface copyConfiguration() {
        return new DCT_ID3();
    }

    
    /**
     * This method makes a subtree below the current node.
     *
     * @param node DecisionTreeNode to expand.
     */
    public void makeTree(DecisionTreeNode node) {
        currDepth++;
        node.depth = currDepth;
        node.currInfoValue = 0;
        if (node.indexes != null && node.indexes.size() >= 1) {
            // Calculate the class distribution within the node.
            node.classPriorsLocal = new float[numClasses];
            for (int i = 0; i < node.indexes.size(); i++) {
                node.classPriorsLocal[(node.discDSet.data.get(
                        node.indexes.get(i))).getCategory()]++;
            }
            float localLargestFreq = 0;
            int numNonZeroClasses = 0;
            for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                if (node.classPriorsLocal[cIndex] > 0) {
                    numNonZeroClasses++;
                    // Normalize the prior.
                    node.classPriorsLocal[cIndex] /=
                            (float) node.indexes.size();
                    // Keep track of the locally largest frequency.
                    if (node.classPriorsLocal[cIndex] > localLargestFreq) {
                        node.majorityClass = cIndex;
                        localLargestFreq = node.classPriorsLocal[cIndex];
                    }
                    node.currInfoValue -= node.classPriorsLocal[cIndex] *
                            BasicMathUtil.log2(node.classPriorsLocal[cIndex]);
                }
            }
            if (numNonZeroClasses == 1) {
                // All instances in the node belong to the same class, so there
                // is no need to expand further. The work is done here.
                currDepth--;
                return;
            }
            if (currDepth <= totalNumAtt) {
                // There are more attributes to try.
                Info infoCalculator = new Info(
                        new DiscreteAttributeValueSplitter(node.discDSet),
                        numClasses);
                int[] typeAndIndex = infoCalculator.
                        getTypeAndIndexOfLowestEvaluatedFeatureOnSubset(
                        node.indexes, acceptableFloat, acceptableInt,
                        acceptableNominal);
                DiscreteAttributeValueSplitter davs =
                        new DiscreteAttributeValueSplitter(node.discDSet);
                // There might be nodes with no instances, but this way there
                // will be a bijective correspondence between the children
                // indexes and split values, which makes cross-referencing
                // easier.
                davs.setEmptyToleration(true);
                // Generate the split.
                ArrayList<Integer>[] split = davs.
                        generateIndexedSplitFromGivenDiscretization(
                        node.indexes, typeAndIndex[0], typeAndIndex[1]);
                node.attType = typeAndIndex[0];
                node.attIndex = typeAndIndex[1];
                // Calculate the information value of the split to see if there
                // is any improvement.
                infoCalculator = new Info(new DiscreteAttributeValueSplitter(
                        node.discDSet), numClasses);
                float splitInfo = infoCalculator.evaluateSplitOnSubset(
                        node.indexes, split);
                if (splitInfo < node.currInfoValue) {
                    // The attribute is flagged as in use.
                    if (node.attType == DataMineConstants.FLOAT) {
                        acceptableFloat[node.attIndex] = false;
                    } else if (node.attType == DataMineConstants.INTEGER) {
                        acceptableInt[node.attIndex] = false;
                    } else {
                        acceptableNominal[node.attIndex] = false;
                    }
                    // Since there is improvement, split on the attribute.
                    node.children = new ArrayList<>(split.length);
                    DecisionTreeNode childNode;
                    for (int sIndex = 0; sIndex < split.length; sIndex++) {
                        childNode = new DecisionTreeNode();
                        childNode.discDSet = node.discDSet;
                        childNode.indexes = split[sIndex];
                        childNode.branchValue = sIndex;
                        node.children.add(childNode);
                        makeTree(childNode);
                    }
                    // Since we are leaving this branch, flag the attribute as
                    // available once again.
                    if (node.attType == DataMineConstants.FLOAT) {
                        acceptableFloat[node.attIndex] = true;
                    } else if (node.attType == DataMineConstants.INTEGER) {
                        acceptableInt[node.attIndex] = true;
                    } else {
                        acceptableNominal[node.attIndex] = true;
                    }
                } else {
                    // No improvement, so we stop.
                    currDepth--;
                    return;
                }
            }
        } else {
            node.majorityClass = generalMajorityClass;
            node.classPriorsLocal = classPriors;
        }
        currDepth--;
    }

    
    @Override
    public void train() throws Exception {
        // Fetch the data.
        DiscretizedDataSet discDSet = getDataType();
        // Count the attributes.
        int floatSize = discDSet.getNumFloatAttr();
        int intSize = discDSet.getNumIntAttr();
        int nominalSize = discDSet.getNumNominalAttr();
        // Initialize the availability arrays.
        acceptableFloat = new boolean[floatSize];
        acceptableInt = new boolean[intSize];
        acceptableNominal = new boolean[nominalSize];
        totalNumAtt = floatSize + intSize + nominalSize;
        Arrays.fill(acceptableFloat, true);
        Arrays.fill(acceptableInt, true);
        Arrays.fill(acceptableNominal, true);
        DiscreteCategory[] classes = getClasses();
        float globalLargestFrequency = 0;
        classPriors = new float[classes.length];
        for (int cIndex = 0; cIndex < classes.length; cIndex++) {
            if (classes[cIndex] != null && classes[cIndex].indexes.size() > 0) {
                classPriors[cIndex] = classes[cIndex].indexes.size() /
                        (float) discDSet.size();
                if (classPriors[cIndex] > globalLargestFrequency) {
                    globalLargestFrequency = classPriors[cIndex];
                    generalMajorityClass = cIndex;
                }
            }
        }
        numClasses = classes.length;
        // Initialize the root node.
        root = new DecisionTreeNode();
        root.discDSet = discDSet;
        root.indexes = new ArrayList(discDSet.size());
        for (int i = 0; i < discDSet.size(); i++) {
            root.indexes.add(i);
        }
        root.branchValue = -1;
        root.currInfoValue = Float.MAX_VALUE;
        makeTree(root);
    }

    
    @Override
    public int classify(DiscretizedDataInstance instance) throws Exception {
        return root.classify(instance);
    }

    
    @Override
    public float[] classifyProbabilistically(DiscretizedDataInstance instance)
            throws Exception {
        return root.classifyProbabilistically(instance);
    }
}
