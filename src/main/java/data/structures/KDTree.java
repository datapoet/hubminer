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
package data.structures;

import data.representation.DataInstance;
import data.representation.DataSet;
import data.representation.util.DataMineConstants;
import learning.unsupervised.Cluster;

/**
 * This class implements the KD-tree data structure.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class KDTree {

    static final int MIN_NODE_SIZE = 15;
    // Root node of the tree.
    KDDataNode root = null;
    // Data that is represented by the tree.
    DataSet dset = null;

    /**
     * @return KDDataNode that is the root of the tree.
     */
    public KDDataNode getRoot() {
        return root;
    }

    /**
     * Creates and fills a new data tree from the provided data.
     *
     * @param dset DataSet object.
     * @throws Exception
     */
    public void createDataTree(DataSet dset) throws Exception {
        if (dset == null || dset.isEmpty()) {
            return;
        }
        this.dset = dset;
        Cluster clust = Cluster.fromEntireDataset(dset);
        root = new KDDataNode(dset);
        for (int i = 0; i < dset.size(); i++) {
            root.addInstance(dset.getInstance(i), i);
        }
        root.calculateVariables();
        fillSubTree(clust, 0, root);
        calculateVariables(root, -1, null);
    }

    /**
     * @param clust Cluster.
     * @return True if it is possible to split into smaller cluster nodes, false
     * otherwise.
     */
    private boolean canSplitCluster(Cluster clust) {
        if (clust == null || clust.isEmpty()) {
            return false;
        } else {
            return clust.size() >= (2 * MIN_NODE_SIZE + 1);
        }
    }

    /**
     * @param clust Cluster
     * @param dim Dimension to split by.
     * @param node Node to branch out to.
     * @throws Exception
     */
    private void fillSubTree(Cluster clust, int dim, KDDataNode node)
            throws Exception {
        if (node != root) {
            for (int i = 0; i < clust.size(); i++) {
                node.addInstance(clust.getInstance(i),
                        clust.getWithinDataSetIndexOf(i));
            }
        }
        if (canSplitCluster(clust)) {
            int totalDim = dset.getNumIntAttr() + dset.getNumFloatAttr();
            if (totalDim <= dim) {
                // Can not split any more.
                node.left = null;
                node.right = null;
                return;
            }
            float medianValue;
            node.left = new KDDataNode(dset);
            Cluster clustLeft = new Cluster(dset, clust.size() / 2 + 1);
            node.right = new KDDataNode(dset);
            Cluster clustRight = new Cluster(dset, clust.size() / 2 + 1);
            DataInstance instance;
            if (!dset.hasIntAttr()) {
                if (dset.hasFloatAttr()) {
                    medianValue = clust.getMedianForDimension(
                            dim, DataMineConstants.FLOAT);
                    for (int i = 0; i < clust.size(); i++) {
                        instance = clust.getInstance(i);
                        if (instance.fAttr[dim] < medianValue) {
                            clustLeft.addInstance(
                                    clust.getWithinDataSetIndexOf(i));
                        } else {
                            clustRight.addInstance(
                                    clust.getWithinDataSetIndexOf(i));
                        }
                    }
                } else {
                    // This case should never occur.
                    throw new Exception("No suitable features available.");
                }
            } else if (dim < dset.getNumIntAttr()) {
                medianValue = clust.getMedianForDimension(dim,
                        DataMineConstants.INTEGER);
                for (int i = 0; i < clust.size(); i++) {
                    instance = clust.getInstance(i);
                    if (instance.iAttr[dim] < medianValue) {
                        clustLeft.addInstance(clust.getWithinDataSetIndexOf(i));
                    } else {
                        clustRight.addInstance(
                                clust.getWithinDataSetIndexOf(i));
                    }
                }
            } else {
                if (dset.hasFloatAttr()) {
                    medianValue = clust.getMedianForDimension(
                            dim - dset.getNumIntAttr(),
                            DataMineConstants.FLOAT);
                    for (int i = 0; i < clust.size(); i++) {
                        instance = clust.getInstance(i);
                        if (instance.fAttr[dim - dset.getNumIntAttr()]
                                < medianValue) {
                            clustLeft.addInstance(
                                    clust.getWithinDataSetIndexOf(i));
                        } else {
                            clustRight.addInstance(
                                    clust.getWithinDataSetIndexOf(i));
                        }
                    }
                } else {
                    // This case should never occur.
                    throw new Exception("No suitable features available.");
                }
            }
            if (!clustLeft.isEmpty()) {
                fillSubTree(clustLeft, dim + 1, node.left);
            } else {
                node.left = null;
            }
            if (!clustRight.isEmpty()) {
                fillSubTree(clustRight, dim + 1, node.right);
            } else {
                node.right = null;
            }
        } else {
            node.left = null;
            node.right = null;
        }
    }

    /**
     * Calculates upper and lower bounds for the node feature values. Recursive.
     *
     * @param node Current node.
     * @param dim Dimension on which it was split when producing this node.
     * @param parentNode Parent node.
     */
    private void calculateVariables(KDDataNode node, int dim,
            KDDataNode parentNode) {
        // It will first recurse down to the leaf nodes and then calculate
        // their variables and propagate and merge the values up the tree to
        // the root.
        if ((node.left == null) || (node.right == null)) {
            node.calculateVariables();
        } else {
            calculateVariables(node.left, dim + 1, node);
            calculateVariables(node.right, dim + 1, node);
        }
        if (parentNode != null) {
            parentNode.squareSum += node.squareSum;
            if (dset.hasIntAttr()) {
                for (int i = 0; i < dset.getNumIntAttr(); i++) {
                    parentNode.linearISum[i] += node.linearISum[i];
                }
                if (dim >= dset.getNumIntAttr()) {
                    parentNode.upperFBounds[dim - dset.getNumIntAttr()] =
                            Math.max(parentNode.upperFBounds[
                                    dim - dset.getNumIntAttr()],
                            node.upperFBounds[
                                    dim - dset.getNumIntAttr()]);
                    parentNode.lowerFBounds[dim - dset.getNumIntAttr()] =
                            Math.min(parentNode.lowerFBounds[
                                    dim - dset.getNumIntAttr()],
                            node.lowerFBounds[
                                    dim - dset.getNumIntAttr()]);
                } else {
                    parentNode.upperIBounds[dim] =
                            Math.max(parentNode.upperIBounds[dim],
                            node.upperIBounds[dim]);
                    parentNode.lowerIBounds[dim] =
                            Math.min(parentNode.lowerIBounds[dim],
                            node.lowerIBounds[dim]);
                }
            } else {
                parentNode.upperFBounds[dim] =
                        Math.max(parentNode.upperFBounds[dim],
                        node.upperFBounds[dim]);
                parentNode.lowerFBounds[dim] =
                        Math.min(parentNode.lowerFBounds[dim],
                        node.lowerFBounds[dim]);
            }
            if (dset.hasFloatAttr()) {
                for (int i = 0; i < dset.getNumFloatAttr(); i++) {
                    parentNode.linearFSum[i] += node.linearFSum[i];
                }
            }
        }
    }
}
