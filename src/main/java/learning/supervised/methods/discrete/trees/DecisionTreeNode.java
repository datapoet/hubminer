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
import java.io.Serializable;
import java.util.ArrayList;

/**
 * This class implements a node for decision trees.
 * 
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class DecisionTreeNode implements Serializable {
    
    private static final long serialVersionUID = 1L;

    // As this is used internally in the decision tree algorithms, the fields
    // have been made public for efficiency and easy of use.

    public ArrayList<Integer> indexes = null;
    public DiscretizedDataSet discDSet = null;
    // Value from the previous split that defines this node.
    public int branchValue;
    // Attribute type of the split attribute for this node.
    public int attType;
    // Attribute index of the split attribute for this node.
    public int attIndex;
    // Child nodes. Sorted. The i-th child should have a value of i.
    public ArrayList<DecisionTreeNode> children;
    // Class priors in the node.
    public float[] classPriorsLocal = null;
    // Majority class in the node.
    public int majorityClass;
    // Information value of the node.
    public float currInfoValue = 0;
    // Depth in the tree.
    public int depth = 1;

    
    /**
     * Performs classification by delegating it to some child node or directly
     * if this is a leaf node.
     * 
     * @param instance DiscretizedDataInstance instance that is to be
     * classified.
     * 
     * @return Integer that is the predicted class assignment.
     * @throws Exception 
     */
    public int classify(DiscretizedDataInstance instance) throws Exception {
        if (children != null && children.size() > 0) {
            int splitValue;
            if (attType == DataMineConstants.FLOAT) {
                splitValue = instance.floatIndexes[attIndex];
            } else if (attType == DataMineConstants.INTEGER) {
                splitValue = instance.integerIndexes[attIndex];
            } else {
                splitValue = instance.nominalIndexes[attIndex];
            }
            // There is a child for every value, so this is safe. No NullPointer
            // exceptions here.
            return children.get(splitValue).classify(instance);
        } else {
            // The case for the leaf nodes.
            return majorityClass;
        }
    }
    
    
    /**
     * Performs classification by delegating it to some child node or directly
     * if this is a leaf node.
     * 
     * @param instance DiscretizedDataInstance instance that is to be
     * classified.
     * 
     * @return float[] that is the predicted probabilistic class assignment.
     * @throws Exception 
     */
    public float[] classifyProbabilistically(DiscretizedDataInstance instance)
            throws Exception {
        if (children != null && children.size() > 0) {
            int splitValue;
            if (attType == DataMineConstants.FLOAT) {
                splitValue = instance.floatIndexes[attIndex];
            } else if (attType == DataMineConstants.INTEGER) {
                splitValue = instance.integerIndexes[attIndex];
            } else {
                splitValue = instance.nominalIndexes[attIndex];
            }
            // There is a child for every value, so this is safe. No NullPointer
            // exceptions here.
            return children.get(splitValue).classifyProbabilistically(instance);
        } else {
            // The case for the leaf nodes.
            return classPriorsLocal;
        }
    }
}
