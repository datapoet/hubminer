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
import distances.primary.CombinedMetric;
import java.util.ArrayList;
import java.util.Arrays;

/**
 * Node type for the KDTree
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class KDDataNode {

    DataSet defSet = null;
    public ArrayList<DataInstance> instances = null;
    public ArrayList<Integer> instanceIndexes = null;
    public KDDataNode left = null;
    public KDDataNode right = null;
    public float[] linearISum = null;
    public float[] linearFSum = null;
    public float squareSum = 0f;
    public int[] upperIBounds = null;
    public int[] lowerIBounds = null;
    public float[] upperFBounds = null;
    public float[] lowerFBounds = null;

    /**
     * @return Number of instances in the node.
     */
    public int size() {
        if (instances == null) {
            return 0;
        } else {
            return instances.size();
        }
    }

    /**
     * @return True if the node is a leaf, false otherwise.
     */
    public boolean isLeaf() {
        if ((left == null) && (right == null)) {
            return true;
        } else {
            return false;
        }
    }

    /**
     * @param defSet Definition DataSet.
     */
    public KDDataNode(DataSet defSet) {
        this.defSet = defSet;
        if (defSet == null) {
            return;
        }
        if (defSet.hasIntAttr()) {
            int iNum = defSet.getNumIntAttr();
            linearISum = new float[iNum];
            upperIBounds = new int[iNum];
            lowerIBounds = new int[iNum];
            Arrays.fill(upperIBounds, Integer.MIN_VALUE);
            Arrays.fill(lowerIBounds, Integer.MAX_VALUE);
        }
        if (defSet.hasFloatAttr()) {
            int fNum = defSet.getNumFloatAttr();
            linearFSum = new float[fNum];
            upperFBounds = new float[fNum];
            lowerFBounds = new float[fNum];
            Arrays.fill(lowerFBounds, Float.MAX_VALUE);
            Arrays.fill(upperFBounds, ((-Float.MAX_VALUE) + 1));
        }
    }

    /**
     * Add a data instance to the node.
     *
     * @param instance DataInstance object.
     * @param index Index of the instance within the definition set.
     */
    public void addInstance(DataInstance instance, int index) {
        if (instance == null) {
            return;
        }
        if (instances == null) {
            instances = new ArrayList<>(KDTree.MIN_NODE_SIZE);
            instanceIndexes = new ArrayList<>(KDTree.MIN_NODE_SIZE);
        }
        instances.add(instance);
        instanceIndexes.add(index);
    }

    /**
     * Calculates the variables within the node.
     */
    public void calculateVariables() {
        for (DataInstance instance : instances) {
            for (int i = 0; i < defSet.getNumIntAttr(); i++) {
                if (!DataMineConstants.isAcceptableInt(instance.iAttr[i])) {
                    continue;
                }
                squareSum += (float) Math.pow(instance.iAttr[i], 2);
                linearISum[i] += instance.iAttr[i];
                upperIBounds[i] = Math.max(upperIBounds[i], instance.iAttr[i]);
                lowerIBounds[i] = Math.min(lowerIBounds[i], instance.iAttr[i]);
            }
            for (int i = 0; i < defSet.getNumFloatAttr(); i++) {
                if (!DataMineConstants.isAcceptableFloat(instance.fAttr[i])) {
                    continue;
                }
                squareSum += (float) Math.pow(instance.fAttr[i], 2);
                linearFSum[i] += instance.fAttr[i];
                upperFBounds[i] =
                        (float) Math.max(upperFBounds[i], instance.fAttr[i]);
                lowerFBounds[i] =
                        (float) Math.min(lowerFBounds[i], instance.fAttr[i]);
            }
        }
    }

    /**
     * Calculates the maximum distances from this node to a data instance.
     *
     * @param instance DataInstance object.
     * @param cmet CombinedMetric object for calculating distances.
     * @return Max distance from the node to the instance.
     * @throws Exception
     */
    float maxDistToDataPoint(DataInstance instance, CombinedMetric cmet)
            throws Exception {
        if (instance == null) {
            throw new Exception("Can not calculate distance to an empty "
                    + "instance");
        }
        DataInstance furthestCorner = new DataInstance(defSet);
        for (int i = 0; i < defSet.getNumIntAttr(); i++) {
            if (!DataMineConstants.isAcceptableInt(instance.iAttr[i])) {
                continue;
            }
            if (Math.abs(upperIBounds[i] - instance.iAttr[i])
                    > Math.abs(lowerIBounds[i] - instance.iAttr[i])) {
                furthestCorner.iAttr[i] = upperIBounds[i];
            } else {
                furthestCorner.iAttr[i] = lowerIBounds[i];
            }
        }
        for (int i = 0; i < defSet.getNumFloatAttr(); i++) {
            if (!DataMineConstants.isAcceptableFloat(instance.fAttr[i])) {
                continue;
            }
            if (Math.abs(upperFBounds[i] - instance.fAttr[i])
                    > Math.abs(lowerFBounds[i] - instance.fAttr[i])) {
                furthestCorner.fAttr[i] = upperFBounds[i];
            } else {
                furthestCorner.fAttr[i] = lowerFBounds[i];
            }
        }
        return cmet.dist(instance, furthestCorner);
    }

    /**
     * Calculates the minimum distances from this node to a data instance.
     *
     * @param instance DataInstance object.
     * @param cmet CombinedMetric object for calculating distances.
     * @return Min distance from the node to the instance.
     * @throws Exception
     */
    float minDistToDataPoint(DataInstance instance, CombinedMetric cmet)
            throws Exception {
        if (instance == null) {
            throw new Exception("Can not calculate distance to an empty "
                    + "instance");
        }
        DataInstance closestCorner = new DataInstance(defSet);
        for (int i = 0; i < defSet.getNumIntAttr(); i++) {
            if (!DataMineConstants.isAcceptableInt(instance.iAttr[i])) {
                continue;
            }
            if (Math.abs(upperIBounds[i] - instance.iAttr[i])
                    < Math.abs(lowerIBounds[i] - instance.iAttr[i])) {
                closestCorner.iAttr[i] = upperIBounds[i];
            } else {
                closestCorner.iAttr[i] = lowerIBounds[i];
            }
        }
        for (int i = 0; i < defSet.getNumFloatAttr(); i++) {
            if (!DataMineConstants.isAcceptableFloat(instance.fAttr[i])) {
                continue;
            }
            if (Math.abs(upperFBounds[i] - instance.fAttr[i])
                    < Math.abs(lowerFBounds[i] - instance.fAttr[i])) {
                closestCorner.fAttr[i] = upperFBounds[i];
            } else {
                closestCorner.fAttr[i] = lowerFBounds[i];
            }
        }
        return cmet.dist(instance, closestCorner);
    }

    /**
     * Prune the list of centroids with respect to this node.
     *
     * @param centroids Centroid candidates.
     * @param cmet CombinedMetric object for calculating distances.
     * @return Pruned centroid candidates.
     * @throws Exception
     */
    public ArrayList<DataInstance> prune(ArrayList<DataInstance> centroids,
            CombinedMetric cmet) throws Exception {
        if (centroids == null || centroids.isEmpty()) {
            throw new Exception("No centroids passed for pruning.");
        }
        ArrayList<DataInstance> candidateCentroids = new ArrayList<>(10);
        float minMaxDist = Float.MAX_VALUE;
        for (DataInstance centroid : centroids) {
            minMaxDist = Math.min(
                    minMaxDist, maxDistToDataPoint(centroid, cmet));
        }
        for (DataInstance centroid : centroids) {
            if (minDistToDataPoint(centroid, cmet) <= minMaxDist) {
                candidateCentroids.add(centroid);
            }
        }
        if (!candidateCentroids.isEmpty()) {
            return candidateCentroids;
        } else {
            return centroids;
        }
    }
}
