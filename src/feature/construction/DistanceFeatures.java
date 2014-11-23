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
package feature.construction;

import data.neighbors.NeighborSetFinder;
import data.representation.DataInstance;
import data.representation.DataSet;
import distances.primary.CombinedMetric;
import java.util.ArrayList;
import sampling.UniformSampler;
import sampling.WeightProportionalSampler;
import util.AuxSort;

/**
 * This class implements an approach that takes distances to certain training 
 * points as features. The subset of points used for this transformation can 
 * be either random or selected in some other way. For instance, it is possible 
 * to consider taking distances to most influential or least influential 
 * neighbor points instead. This hubness-aware feature construction approach 
 * was considered in the following book chapter: Hubness-aware Classication,
 *Instance Selection and Feature Construction: Survey and Extensions to
 * Time-Series, published in: Feature Selection for Data and Pattern 
 * Recognition. The chapter was authored by Nenad Tomasev, Krisztian Buza, 
 * Kristof Marussy and Piroska B. Kis. However, this class implements a slightly
 * modified version of the approach, where there is one training set and the 
 * target points for calculating distance features belong to the same set as the
 * points that are to be represented in that way. In the chapter, these two sets
 * were disjoint. In practice, there is little difference between the two. For 
 * completeness, this class also offers a method that calculates the distances 
 * to external points, so that the same analysis as in the chapter can be 
 * performed.
 * 
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class DistanceFeatures {
    
    // DataSet object to generate the new representation for.
    private DataSet dset;
    // Metric used for calculating the distance features.
    private CombinedMetric cmet = CombinedMetric.EUCLIDEAN;
    // Upper triangular distance matrix.
    private float[][] dMat;
    
    /**
     * Initialization.
     * @param dset DataSet to calculate the distance features for.
     */
    public DistanceFeatures(DataSet dset) {
        this.dset = dset;
    }
    
    /**
     * Initialization.
     * @param dset DataSet to calculate the distance features for.
     * @param cmet CombinedMetric object for distance calculations.
     */
    public DistanceFeatures(DataSet dset, CombinedMetric cmet) {
        this.dset = dset;
        this.cmet = cmet;
    }
    
    /**
     * Initialization.
     * @param dset DataSet to calculate the distance features for.
     * @param cmet CombinedMetric object for distance calculations.
     * @param dMat float[][] representing the upper triangular distance matrix.
     */
    public DistanceFeatures(DataSet dset, CombinedMetric cmet, float[][] dMat) {
        this.dset = dset;
        this.cmet = cmet;
        this.dMat = dMat;
    }
    
    /**
     * This method generates a new data representation where features are the 
     * distances to the provided target data instances from within the original 
     * collection.
     * 
     * @param targetIndexes ArrayList<Integer> that are the indexes of instances
     * to take distance to as features.
     * 
     * @return DataSet with the new data representation.
     * @throws Exception 
     */
    public DataSet getDistanceFeaturesForIndexes(
            ArrayList<Integer> targetIndexes) throws Exception {
        if (targetIndexes == null || targetIndexes.isEmpty()) {
            throw new Exception("Target indexes not specified.");
        }
        if (dset == null || dset.isEmpty()) {
            throw new Exception("Target data not specified.");
        }
        int numTargets = targetIndexes.size();
        DataSet distFeatDSet = new DataSet();
        String[] fAttNames = new String[numTargets];
        for (int d = 0; d < numTargets; d++) {
            fAttNames[d] = "DistTo" + targetIndexes.get(d);
        }
        distFeatDSet.fAttrNames = fAttNames;
        distFeatDSet.data = new ArrayList<>(dset.size());
        int first, second;
        for (int i = 0; i < dset.size(); i++) {
            DataInstance instance = dset.getInstance(i);
            DataInstance distFeatInstance = new DataInstance(distFeatDSet);
            distFeatInstance.embedInDataset(distFeatDSet);
            distFeatDSet.addDataInstance(distFeatInstance);
            // Now add distances as features.
            for (int d = 0; d < numTargets; d++) {
                if (dMat != null) {
                    first = Math.min(i, targetIndexes.get(d));
                    second = Math.max(i, targetIndexes.get(d));
                    distFeatInstance.fAttr[d] = dMat[first][second - first - 1];
                } else {
                    distFeatInstance.fAttr[d] = cmet.dist(instance,
                            dset.getInstance(targetIndexes.get(d)));
                }
            }
            // Set the class label.
            distFeatInstance.setCategory(instance.getCategory());
        }
        return distFeatDSet;
    }
    
    /**
     * This method generates a new data representation where features are the 
     * distances to the provided target data instances from within the original 
     * collection.
     * 
     * @param targetInstances ArrayList<DataInstance> Instances used as distance
     * targets for calculating the features.
     * 
     * @return DataSet with the new data representation.
     * @throws Exception 
     */
    public DataSet getDistanceFeaturesForTargetInstances(
            ArrayList<DataInstance> targetInstances) throws Exception {
        if (targetInstances == null || targetInstances.isEmpty()) {
            throw new Exception("Target instances not specified.");
        }
        if (dset == null || dset.isEmpty()) {
            throw new Exception("Target data not specified.");
        }
        int numTargets = targetInstances.size();
        DataSet distFeatDSet = new DataSet();
        String[] fAttNames = new String[numTargets];
        for (int d = 0; d < numTargets; d++) {
            fAttNames[d] = "DistTo" + d;
        }
        distFeatDSet.fAttrNames = fAttNames;
        distFeatDSet.data = new ArrayList<>(dset.size());
        for (int i = 0; i < dset.size(); i++) {
            DataInstance instance = dset.getInstance(i);
            DataInstance distFeatInstance = new DataInstance(distFeatDSet);
            distFeatInstance.embedInDataset(distFeatDSet);
            distFeatDSet.addDataInstance(distFeatInstance);
            // Now add distances as features.
            for (int d = 0; d < numTargets; d++) {
                distFeatInstance.fAttr[d] = cmet.dist(instance,
                        targetInstances.get(d));
            }
            distFeatInstance.setCategory(instance.getCategory());
        }
        return distFeatDSet;
    }
    
    /**
     * This method calculates the distance feature representation based on the
     * distances to hub points.
     * @param nsf NeighborSetFinder object holding the kNN sets.
     * @param numTargets Integer that is the number of hub points to take for 
     * distance calculations.
     * @return DataSet with the new data representation.
     * @throws Exception 
     */
    public DataSet getDistanceFeaturesFromHubs(
            NeighborSetFinder nsf, int numTargets) throws Exception {
        if (nsf == null) {
            throw new Exception("No kNN sets were provided.");
        }
        if (dset == null || dset.isEmpty()) {
            throw new Exception("Target data not specified.");
        }
        if (numTargets < 0 || numTargets > dset.size()) {
            throw new Exception("Invalid number of distance tagets: " +
                    numTargets);
        }
        int[] occFreqs = nsf.getNeighborFrequencies();
        // Descending sort.
        int[] reArr = AuxSort.sortIndexedValue(occFreqs, true);
        ArrayList<Integer> targetIndexes = new ArrayList<>(numTargets);
        for (int i = 0; i < numTargets; i++) {
            targetIndexes.add(reArr[i]);
        }
        return getDistanceFeaturesForIndexes(targetIndexes);
    }
    
    /**
     * This method calculates the distance feature representation based on the
     * distances to good hub points.
     * @param nsf NeighborSetFinder object holding the kNN sets.
     * @param numTargets Integer that is the number of good hub points to take
     * for distance calculations.
     * @return DataSet with the new data representation.
     * @throws Exception 
     */
    public DataSet getDistanceFeaturesFromGoodHubs(
            NeighborSetFinder nsf, int numTargets) throws Exception {
        if (nsf == null) {
            throw new Exception("No kNN sets were provided.");
        }
        if (dset == null || dset.isEmpty()) {
            throw new Exception("Target data not specified.");
        }
        if (numTargets < 0 || numTargets > dset.size()) {
            throw new Exception("Invalid number of distance tagets: " +
                    numTargets);
        }
        int[] occFreqs = nsf.getGoodFrequencies();
        // Descending sort.
        int[] reArr = AuxSort.sortIndexedValue(occFreqs, true);
        ArrayList<Integer> targetIndexes = new ArrayList<>(numTargets);
        for (int i = 0; i < numTargets; i++) {
            targetIndexes.add(reArr[i]);
        }
        return getDistanceFeaturesForIndexes(targetIndexes);
    }
    
    /**
     * This method calculates the distance feature representation based on the
     * distances to bad hub points.
     * @param nsf NeighborSetFinder object holding the kNN sets.
     * @param numTargets Integer that is the number of bad hub points to take
     * for  distance calculations.
     * @return DataSet with the new data representation.
     * @throws Exception 
     */
    public DataSet getDistanceFeaturesFromBadHubs(
            NeighborSetFinder nsf, int numTargets) throws Exception {
        if (nsf == null) {
            throw new Exception("No kNN sets were provided.");
        }
        if (dset == null || dset.isEmpty()) {
            throw new Exception("Target data not specified.");
        }
        if (numTargets < 0 || numTargets > dset.size()) {
            throw new Exception("Invalid number of distance tagets: " +
                    numTargets);
        }
        int[] occFreqs = nsf.getBadFrequencies();
        // Descending sort.
        int[] reArr = AuxSort.sortIndexedValue(occFreqs, true);
        ArrayList<Integer> targetIndexes = new ArrayList<>(numTargets);
        for (int i = 0; i < numTargets; i++) {
            targetIndexes.add(reArr[i]);
        }
        return getDistanceFeaturesForIndexes(targetIndexes);
    }
    
    /**
     * This method calculates the distance feature representation based on the
     * distances to anti-hub points.
     * @param nsf NeighborSetFinder object holding the kNN sets.
     * @param numTargets Integer that is the number of anti-hub points to take
     * for distance calculations.
     * @return DataSet with the new data representation.
     * @throws Exception 
     */
    public DataSet getDistanceFeaturesFromAntiHubs(
            NeighborSetFinder nsf, int numTargets) throws Exception {
        if (nsf == null) {
            throw new Exception("No kNN sets were provided.");
        }
        if (dset == null || dset.isEmpty()) {
            throw new Exception("Target data not specified.");
        }
        if (numTargets < 0 || numTargets > dset.size()) {
            throw new Exception("Invalid number of distance tagets: " +
                    numTargets);
        }
        int[] occFreqs = nsf.getNeighborFrequencies();
        // Ascending sort.
        int[] reArr = AuxSort.sortIndexedValue(occFreqs, false);
        ArrayList<Integer> targetIndexes = new ArrayList<>(numTargets);
        for (int i = 0; i < numTargets; i++) {
            targetIndexes.add(reArr[i]);
        }
        return getDistanceFeaturesForIndexes(targetIndexes);
    }
    
    /**
     * This method calculates the distance feature representation based on the
     * distances to randomly selected training points.
     * @param numTargets Integer that is the number of random points to take for
     * distance calculations.
     * @return DataSet with the new data representation.
     * @throws Exception 
     */
    public DataSet getDistanceFeaturesFromRandom(int numTargets)
            throws Exception {
        if (dset == null || dset.isEmpty()) {
            throw new Exception("Target data not specified.");
        }
        if (numTargets < 0 || numTargets > dset.size()) {
            throw new Exception("Invalid number of distance tagets: " +
                    numTargets);
        }
        int[] targetIndexArr =
                UniformSampler.getSample(dset.size(), numTargets);
        ArrayList<Integer> targetIndexes = new ArrayList<>(numTargets);
        for (int i = 0; i < numTargets; i++) {
            targetIndexes.add(targetIndexArr[i]);
        }
        return getDistanceFeaturesForIndexes(targetIndexes);
    }
    
    /**
     * This method calculates the distance feature representation based on the
     * distances to stochastically selected points, in a hubness-proportional 
     * way.
     * @param nsf NeighborSetFinder object holding the kNN sets.
     * @param numTargets Integer that is the number of stochastically selected 
     * points to take for distance calculations.
     * @return DataSet with the new data representation.
     * @throws Exception 
     */
    public DataSet getDistanceFeaturesHubnessProportional(NeighborSetFinder nsf,
            int numTargets)
            throws Exception {
        if (dset == null || dset.isEmpty()) {
            throw new Exception("Target data not specified.");
        }
        if (numTargets < 0 || numTargets > dset.size()) {
            throw new Exception("Invalid number of distance tagets: " +
                    numTargets);
        }
        float[] selectionWeights = nsf.getFloatOccFreqs();
        int[] targetIndexArr =
                WeightProportionalSampler.getSampleNoReps(numTargets,
                selectionWeights);
        ArrayList<Integer> targetIndexes = new ArrayList<>(numTargets);
        for (int i = 0; i < numTargets; i++) {
            targetIndexes.add(targetIndexArr[i]);
        }
        return getDistanceFeaturesForIndexes(targetIndexes);
    }
}
