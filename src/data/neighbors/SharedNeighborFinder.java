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
package data.neighbors;

import data.representation.DataInstance;
import data.representation.DataSet;
import distances.primary.CombinedMetric;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Set;
import util.ArrayUtil;
import util.BasicMathUtil;

/**
 * This class calculates the kNN set intersections between different points in
 * order to later calculate shared-neighbor similarity/dissimilarity measures.
 * These secondary measures are often better suited for high-dimensional data
 * than the primary metrics.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class SharedNeighborFinder implements Serializable {
    
    private static final long serialVersionUID = 1L;

    private NeighborSetFinder nsf = null;
    private DataSet dset = null;
    private CombinedMetric cmet = null;
    // This neighborhood size is used for finding shared neighbors.
    private int k = 50;
    // This neigborhood size is used for calculating instance weights in
    // certain weighting schemes.
    private int kClassification = 5;
    // Hash the neighbors of each point for faster calculations of the
    // intersections.
    private HashMap<Integer, Integer>[] neighborHash;
    private int numClasses;
    // Instance weights to use in distance calculations.
    float[] instanceWeights;
    // Shared neighbor lists.
    private ArrayList<Integer>[][] sharedNeighbors = null;
    // Shared neighbor counts.
    private float[][] sharedNeighborCount = null;
    public static final int DEFAULT_NUM_THREADS = 8;

    /**
     * Initializes the neighbor hashes.
     */
    private void initializeHashes() {
        if (dset == null || nsf == null) {
            return;
        }
        neighborHash = new HashMap[dset.size()];
        int[][] kneighbors = nsf.getKNeighbors();
        for (int i = 0; i < dset.size(); i++) {
            neighborHash[i] = new HashMap<>(2 * k);
            for (int j = 0; j < k; j++) {
                neighborHash[i].put(kneighbors[i][j], j);
            }
        }
    }

    /**
     * Checks if neighbor hashes have been initialized.
     *
     * @return True if the hashes are initialized, false otherwise.
     */
    private boolean hashInitialized() {
        if (neighborHash == null || neighborHash.length == 0) {
            return false;
        } else {
            return true;
        }
    }

    /**
     * @param nsf NeighborSetFinder object.
     */
    public void setNSF(NeighborSetFinder nsf) {
        this.nsf = nsf;
    }

    /**
     * @return NeighborSetFinder object.
     */
    public NeighborSetFinder getNSF() {
        return nsf;
    }

    /**
     * Calculates the instance weights for the class-imbalanced case.
     */
    public void obtainWeightsForClassImbalancedData() {
        int size = dset.size();
        float[] weightArray = new float[size];
        int numCat = dset.countCategories();
        int[] classCounts = dset.getClassFrequencies();
        float maxEntropy = (float) BasicMathUtil.log2((float) numCat);
        float[][] chubness = nsf.getClassDataNeighborRelationNonNormalized(
                k, numCat, false);
        float[][] classToClass = nsf.getGlobalClassToClassNonNormalized(
                kClassification, numCat);
        float[] classRelevance = new float[numCat];
        for (int c = 0; c < numCat; c++) {
            classRelevance[c] = (float) (1 - ((classToClass[c][c] + 0.00001)
                    / (kClassification * classCounts[c] + 0.00001)));
        }
        if (nsf.getReverseNeighborEntropies() == null) {
            nsf.calculateReverseNeighborEntropiesWeighted(numCat,
                    classRelevance);
        }
        int[] neighbOccFreqs = nsf.getNeighborFrequencies();
        float maxWeight = 1;
        for (int i = 0; i < size; i++) {
            weightArray[i] = (float) BasicMathUtil.log2(((float) size)
                    / ((float) nsf.getNeighborFrequencies()[i] + 1));
            maxWeight = Math.max(Math.abs(weightArray[i]), maxWeight);
        }
        for (int i = 0; i < size; i++) {
            weightArray[i] /= maxWeight;
        }
        // This was the I_n / max I_n part of the weight.              
        maxWeight = 0;
        float min = Float.MAX_VALUE;
        // Difference between good and bad hubness.
        float indicator;
        // Good and bad hubness sums.
        double ghsum;
        double bhsum;
        float[] weightArraySecond = new float[size];
        float[] ghsumAll = new float[size];
        float[] bhsumAll = new float[size];

        for (int i = 0; i < size; i++) {
            if (neighbOccFreqs[i] < 1) {
                weightArraySecond[i] = 1;
                continue;
            }
            ghsum = 0;
            bhsum = 0;
            for (int c = 0; c < numCat; c++) {
                ghsum += (chubness[c][i] * (chubness[c][i] - 1) / 2)
                        * (1 - ((classToClass[c][c] + 0.00001)
                        / (kClassification * classCounts[c] + 0.00001)));
                for (int c1 = 0; c1 < numCat; c1++) {
                    if (c != c1) {
                        bhsum += (chubness[c][i] * chubness[c1][i] * 0.5
                                * (((classToClass[c1][c] + 0.00001)
                                / (kClassification * classCounts[c] + 0.00001))
                                + ((classToClass[c][c1] + 0.00001)
                                / (kClassification * classCounts[c1]
                                + 0.00001))));
                    }
                }
            }
            ghsumAll[i] = (float) ghsum;
            bhsumAll[i] = (float) bhsum;
            indicator = (float) (ghsum - bhsum);
            weightArraySecond[i] = indicator;
            maxWeight = Math.max(weightArraySecond[i], maxWeight);
            min = Math.min(weightArraySecond[i], min);
        }
        ArrayUtil.zStandardize(weightArraySecond);
        ArrayUtil.zStandardize(ghsumAll);
        ArrayUtil.zStandardize(bhsumAll);
        for (int i = 0; i < size; i++) {
            weightArraySecond[i] = 1f
                    / (1f + (float) Math.exp(-weightArraySecond[i]));
        }
        for (int i = 0; i < size; i++) {
            ghsumAll[i] = 1f / (1f + (float) Math.exp(-ghsumAll[i]));
        }
        for (int i = 0; i < size; i++) {
            bhsumAll[i] = 1f / (1f + (float) Math.exp(bhsumAll[i]));
        }

        maxWeight = 1;
        float[] weightArrayProfile = new float[size];
        for (int i = 0; i < size; i++) {
            weightArrayProfile[i] = (maxEntropy
                    - nsf.getReverseNeighborEntropies()[i]);
            maxWeight = Math.max(Math.abs(weightArray[i]), maxWeight);
        }
        for (int i = 0; i < size; i++) {
            weightArrayProfile[i] /= maxWeight;
        }

        for (int i = 0; i < size; i++) {
            weightArray[i] = weightArray[i] * weightArrayProfile[i]
                    * weightArraySecond[i];
        }
        instanceWeights = weightArray;
    }

    /**
     * @param cmet CombinedMetric object that is the primary distance measure.
     */
    public void setPrimaryMetricsCalculator(CombinedMetric cmet) {
        this.cmet = cmet;
    }

    /**
     * @param numClasses Integer that is the number of classes.
     */
    public void setNumClasses(int numClasses) {
        this.numClasses = numClasses;
    }

    /**
     *
     */
    public SharedNeighborFinder() {
    }

    /**
     *
     * @param dset DataSet that is being analyzed.
     * @param k Neighborhood size for shared-neighbor calculation.
     * @param cmet CombinedMetric object for primary distances.
     * @param kClassification Neighborhood size used for classification.
     */
    public SharedNeighborFinder(DataSet dset,
            int k, CombinedMetric cmet, int kClassification) {
        this.dset = dset;
        this.k = k;
        this.cmet = cmet;
        this.kClassification = kClassification;
    }

    /**
     *
     * @param dset DataSet that is being analyzed.
     * @param k Neighborhood size for shared-neighbor calculation.
     * @param cmet CombinedMetric object for primary distances.
     */
    public SharedNeighborFinder(DataSet dset, int k, CombinedMetric cmet) {
        this.dset = dset;
        this.k = k;
        this.cmet = cmet;
    }

    /**
     *
     * @param nsf NeighborSetFinder containing the kNN information.
     */
    public SharedNeighborFinder(NeighborSetFinder nsf) {
        this.nsf = nsf;
        this.k = nsf.getCurrK();
        this.dset = nsf.getDataSet();
        this.cmet = nsf.getCombinedMetric();
        initializeHashes();
    }

    /**
     *
     * @param nsf NeighborSetFinder containing the kNN information.
     * @param kClassification Neighborhood size that will be later used in
     * classification. It is sometimes used for finding instance weights. The k
     * that will be used for shared-neighbor calculations is extracted from the
     * NeighborSetFinder object instead.
     */
    public SharedNeighborFinder(NeighborSetFinder nsf, int kClassification) {
        this.nsf = nsf;
        this.k = nsf.getCurrK();
        this.dset = nsf.getDataSet();
        this.cmet = nsf.getCombinedMetric();
        this.kClassification = kClassification;
        initializeHashes();
    }

    /**
     * Sets the neighborhood size that will be used in shared-neighbor
     * calculations.
     *
     * @param k Integer that is the neighborhood size that will be used in
     * shared-neighbor calculations.
     */
    public void setSNK(int k) {
        this.k = k;
    }

    /**
     * @param kClassification Neighborhood size that will be later used in
     * classification. It is sometimes used for finding instance weights.
     */
    public void setKClassification(int kClassification) {
        this.kClassification = kClassification;
    }

    /**
     * @return A float matrix that has stores the counts of shared neighbors
     * between pairs of points. It is symmetric, so we use the upper diagonal
     * form, where each row stores only d(i, j) for i > j.
     */
    public float[][] getSharedNeighborCounts() {
        return sharedNeighborCount;
    }

    /**
     * @return CombinedMetric object for primary distance calculations.
     */
    public CombinedMetric getCombinedMetric() {
        return cmet;
    }

    /**
     * @return DataSet that is being analyzed.
     */
    public DataSet getData() {
        return dset;
    }

    /**
     * @return Integer that is the neighborhood size that will be used in
     * shared-neighbor calculations.
     */
    public int getSNK() {
        return k;
    }

    /**
     * Elements that are hubs contribute less to similarity between other points
     * when found as shared neighbors.
     */
    public void obtainWeightsFromGeneralHubness() {
        instanceWeights = nsf.getPenalizeHubnessWeightingScheme();
    }

    /**
     * Elements that occur in neighbor sets between several categories are going
     * to have their weights reduced down, thereby increasing intraclass
     * similarity and decreasing interclass similarity.
     */
    public void obtainWeightsFromBadHubness() {
        instanceWeights = nsf.getHWKNNWeightingScheme();
    }

    /**
     * Get the overall hubness information instance weighting scheme.
     *
     * @param theta The tradeoff parameter.
     */
    public void obtainWeightsFromHubnessInformation(float theta) {
        if (numClasses > 0) {
            instanceWeights = nsf.getSimhubWeightingScheme(
                    numClasses, theta);
        } else {
            instanceWeights = nsf.getSimhubWeightingScheme(theta);
        }
    }

    /**
     * Get the overall hubness information instance weighting scheme while using
     * default tradeoff.
     */
    public void obtainWeightsFromHubnessInformation() {
        if (numClasses > 0) {
            instanceWeights = nsf.getSimhubWeightingScheme(
                    numClasses, 0);
        } else {
            instanceWeights = nsf.getSimhubWeightingScheme(0);
        }
    }

    /**
     * Sets the instance weights.
     *
     * @param instanceWeights Float array of instance weights.
     */
    public void setWeights(float[] instanceWeights) {
        this.instanceWeights = instanceWeights;
    }

    /**
     * @return Float array of instance weights.
     */
    public float[] getInstanceWeights() {
        return instanceWeights;
    }

    /**
     * Remove the currently used instance weights, reset them.
     */
    public void removeWeights() {
        instanceWeights = null;
    }

    /**
     * Counts the number of shared k-nearest neighbors between the two points.
     *
     * @param instanceFirst First DataInstance.
     * @param instanceSecond Second DataInstance.
     * @param kNeighborsFirst Integer array as the kNN set of the first data
     * instance.
     * @param kNeighborsSecond Integer array as the kNN set of the second data
     * instance.
     * @return Float that is the weighted shared-neighbor count between the two
     * instances.
     * @throws Exception
     */
    public float countSharedNeighborsWithRespectToDataset(
            DataInstance instanceFirst, DataInstance instanceSecond,
            int[] kNeighborsFirst, int[] kNeighborsSecond) throws Exception {
        // We just go through the neighbor sets and compare. Using HashMaps is
        // another options, though that makes more sense when all pairwise
        // distances are needed or when the neighborhood size is exceedingly
        // large.
        if (kNeighborsFirst == null || kNeighborsSecond == null) {
            return 0;
        }
        float result = 0;
        for (int k1 = 0; k1 < k; k1++) {
            for (int k2 = 0; k2 < k; k2++) {
                if (kNeighborsFirst[k1] == kNeighborsSecond[k2]) {
                    if (instanceWeights == null) {
                        result++;
                    } else {
                        result += instanceWeights[kNeighborsFirst[k1]];
                    }
                    // As the shared neighbor was found, we can skip the
                    // rest of the inner loop.
                    break;
                }
            }
        }
        return result;
    }

    /**
     * Counts the number of shared k-nearest neighbors between the two points.
     *
     * @param instanceFirst First DataInstance.
     * @param instanceSecond Second DataInstance.
     * @param neighborHashFirst HashMap that has k-neighbors of the first
     * instance as keys.
     * @param neighborHashSecond HashMap that has k-neighbors of the second
     * instance as keys.
     * @return Float that is the weighted shared-neighbor count between the two
     * instances.
     * @throws Exception
     */
    public float countSharedNeighborsWithRespectToDataset(
            DataInstance instanceFirst, DataInstance instanceSecond,
            HashMap<Integer, Object> neighborHashFirst,
            HashMap<Integer, Object> neighborHashSecond) throws Exception {
        if (neighborHashFirst == null || neighborHashSecond == null) {
            return 0;
        }
        float result = 0;
        Set<Integer> keysFirst = neighborHashFirst.keySet();
        for (int index : keysFirst) {
            if (neighborHashSecond.containsKey(index)) {
                if (instanceWeights == null) {
                    result++;
                } else {
                    result += instanceWeights[index];
                }
            }
        }
        return result;
    }

    /**
     * Counts the number of shared k-nearest neighbors between the two points.
     *
     * @param instanceFirst First DataInstance.
     * @param instanceSecond Second DataInstance.
     * @param distToDataFromFirst Float array representing the distances from
     * the first data instance to the training data points.
     * @param kNeighborsSecond Integer array as the kNN set of the second data
     * instance.
     * @return Float that is the weighted shared-neighbor count between the two
     * instances.
     * @throws Exception
     */
    public float countSharedNeighborsWithRespectToDataset(
            DataInstance instanceFirst, DataInstance instanceSecond,
            float[] distToDataFromFirst, int[] kNeighborsSecond)
            throws Exception {
        // First we find the kNN set of the first point, to compare it with
        // the kNN set of the second point.
        if (distToDataFromFirst == null || kNeighborsSecond == null) {
            return 0;
        }
        float[] kDistancesFirst = new float[k];
        for (int i = 0; i < k; i++) {
            kDistancesFirst[i] = Float.MAX_VALUE;
        }
        int[] kNeighborsFirst = new int[k];
        float currDistance;
        int index;
        for (int i = 0; i < dset.size(); i++) {
            currDistance = distToDataFromFirst[i];
            if (currDistance < kDistancesFirst[k - 1]) {
                // Insertion.
                index = k - 1;
                while (index > 0 && kDistancesFirst[index - 1] > currDistance) {
                    kDistancesFirst[index] = kDistancesFirst[index - 1];
                    kNeighborsFirst[index] = kNeighborsFirst[index - 1];
                    index--;
                }
                kDistancesFirst[index] = currDistance;
                kNeighborsFirst[index] = i;
            }
        }
        // Now for the main comparisons.
        float result = 0;
        for (int k1 = 0; k1 < k; k1++) {
            for (int k2 = 0; k2 < k; k2++) {
                if (kNeighborsFirst[k1] == kNeighborsSecond[k2]) {
                    if (instanceWeights == null) {
                        result++;
                    } else {
                        result += instanceWeights[kNeighborsFirst[k1]];
                    }
                    break;
                }
            }
        }
        return result;
    }

    /**
     * Counts the number of shared k-nearest neighbors between the two points.
     *
     * @param instanceFirst First DataInstance.
     * @param instanceSecond Second DataInstance.
     * @param distToDataFromFirst Float array representing the distances from
     * the first data instance to the training data points.
     * @param distToDataFromSecond Float array representing the distances from
     * the second data instance to the training data points.
     * @return Float that is the weighted shared-neighbor count between the two
     * instances.
     * @throws Exception
     */
    public float countSharedNeighborsWithRespectToDataset(
            DataInstance instanceFirst, DataInstance instanceSecond,
            float[] distToDataFromFirst, float[] distToDataFromSecond)
            throws Exception {
        // Find the neighbors of the first point.
        float[] kDistancesFirst = new float[k];
        for (int i = 0; i < k; i++) {
            kDistancesFirst[i] = Float.MAX_VALUE;
        }
        int[] kNeighborsFirst = new int[k];
        float currDistance;
        int index;
        for (int i = 0; i < dset.size(); i++) {
            currDistance = distToDataFromFirst[i];
            if (currDistance < kDistancesFirst[k - 1]) {
                // Insertion.
                index = k - 1;
                while (index > 0 && kDistancesFirst[index - 1] > currDistance) {
                    kDistancesFirst[index] = kDistancesFirst[index - 1];
                    kNeighborsFirst[index] = kNeighborsFirst[index - 1];
                    index--;
                }
                kDistancesFirst[index] = currDistance;
                kNeighborsFirst[index] = i;
            }
        }
        // Find the neighbors of the second point.
        float[] kDistancesSecond = new float[k];
        for (int i = 0; i < k; i++) {
            kDistancesSecond[i] = Float.MAX_VALUE;
        }
        int[] kNeighborsSecond = new int[k];
        for (int i = 0; i < dset.size(); i++) {
            currDistance = distToDataFromSecond[i];
            if (currDistance < kDistancesSecond[k - 1]) {
                // Insertion.
                index = k - 1;
                while (index > 0
                        && kDistancesSecond[index - 1] > currDistance) {
                    kDistancesSecond[index] = kDistancesSecond[index - 1];
                    kNeighborsSecond[index] = kNeighborsSecond[index - 1];
                    index--;
                }
                kDistancesSecond[index] = currDistance;
                kNeighborsSecond[index] = i;
            }
        }
        // Look for the shared neighbors.
        float result = 0;
        for (int k1 = 0; k1 < k; k1++) {
            for (int k2 = 0; k2 < k; k2++) {
                if (kNeighborsFirst[k1] == kNeighborsSecond[k2]) {
                    if (instanceWeights == null) {
                        result++;
                    } else {
                        result += instanceWeights[kNeighborsFirst[k1]];
                    }
                    break;
                }
            }
        }
        return result;
    }

    /**
     * Counts the number of shared k-nearest neighbors between the two points.
     *
     * @param instanceFirst First DataInstance.
     * @param instanceSecond Second DataInstance.
     * @return Float that is the weighted shared-neighbor count between the two
     * instances.
     * @throws Exception
     */
    public float countSharedNeighborsWithRespectToDataset(
            DataInstance instanceFirst, DataInstance instanceSecond)
            throws Exception {
        // Find the neighbors of the two instances and then compare the kNN
        // sets to find the shared neighbor points.
        float[] kDistancesFirst = new float[k];
        for (int i = 0; i < k; i++) {
            kDistancesFirst[i] = Float.MAX_VALUE;
        }
        int[] kNeighborsFirst = new int[k];
        float currDistance;
        int index;
        for (int i = 0; i < dset.size(); i++) {
            currDistance = cmet.dist(dset.data.get(i), instanceFirst);
            if (currDistance < kDistancesFirst[k - 1]) {
                // Insertion.
                index = k - 1;
                while (index > 0 && kDistancesFirst[index - 1] > currDistance) {
                    kDistancesFirst[index] = kDistancesFirst[index - 1];
                    kNeighborsFirst[index] = kNeighborsFirst[index - 1];
                    index--;
                }
                kDistancesFirst[index] = currDistance;
                kNeighborsFirst[index] = i;
            }
        }
        // Now for the second instance.
        float[] kDistancesSecond = new float[k];
        for (int i = 0; i < k; i++) {
            kDistancesSecond[i] = Float.MAX_VALUE;
        }
        int[] kNeighborsSecond = new int[k];
        for (int i = 0; i < dset.size(); i++) {
            currDistance = cmet.dist(dset.data.get(i), instanceSecond);
            if (currDistance < kDistancesSecond[k - 1]) {
                // Insertion.
                index = k - 1;
                while (index > 0 && kDistancesSecond[index - 1] > currDistance) {
                    kDistancesSecond[index] = kDistancesSecond[index - 1];
                    kNeighborsSecond[index] = kNeighborsSecond[index - 1];
                    index--;
                }
                kDistancesSecond[index] = currDistance;
                kNeighborsSecond[index] = i;
            }
        }
        float result = 0;
        for (int k1 = 0; k1 < k; k1++) {
            for (int k2 = 0; k2 < k; k2++) {
                if (kNeighborsFirst[k1] == kNeighborsSecond[k2]) {
                    if (instanceWeights == null) {
                        result++;
                    } else {
                        result += instanceWeights[kNeighborsFirst[k1]];
                    }
                    break;
                }
            }
        }

        return result;
    }

    /**
     * Finds the shared k-nearest neighbors between the two points.
     *
     * @param instanceFirst First DataInstance.
     * @param instanceSecond Second DataInstance.
     * @return ArrayList of integer indexes of the neighbor points that are
     * shared between the two instances.
     * @throws Exception
     */
    public ArrayList<Integer> findSharedNeighborsWithRespectToDataset(
            DataInstance instanceFirst, DataInstance instanceSecond)
            throws Exception {
        // First find the neighbors.
        float[] kDistancesFirst = new float[k];
        for (int i = 0; i < k; i++) {
            kDistancesFirst[i] = Float.MAX_VALUE;
        }
        int[] kNeighborsFirst = new int[k];
        float currDistance;
        int index;
        for (int i = 0; i < dset.size(); i++) {
            currDistance = cmet.dist(dset.data.get(i), instanceFirst);
            if (currDistance < kDistancesFirst[k - 1]) {
                // Insertion.
                index = k - 1;
                while (index > 0 && kDistancesFirst[index - 1] > currDistance) {
                    kDistancesFirst[index] = kDistancesFirst[index - 1];
                    kNeighborsFirst[index] = kNeighborsFirst[index - 1];
                    index--;
                }
                kDistancesFirst[index] = currDistance;
                kNeighborsFirst[index] = i;
            }
        }
        // Now for the second instance.
        float[] kDistancesSecond = new float[k];
        for (int i = 0; i < k; i++) {
            kDistancesSecond[i] = Float.MAX_VALUE;
        }
        int[] kNeighborsSecond = new int[k];
        for (int i = 0; i < dset.size(); i++) {
            currDistance = cmet.dist(dset.data.get(i), instanceSecond);
            if (currDistance < kDistancesSecond[k - 1]) {
                // Insertion.
                index = k - 1;
                while (index > 0 && kDistancesSecond[index - 1] >
                        currDistance) {
                    kDistancesSecond[index] = kDistancesSecond[index - 1];
                    kNeighborsSecond[index] = kNeighborsSecond[index - 1];
                    index--;
                }
                kDistancesSecond[index] = currDistance;
                kNeighborsSecond[index] = i;
            }
        }
        // Now look for the shared neighbor points.
        ArrayList<Integer> result = new ArrayList<>(k);
        for (int k1 = 0; k1 < k; k1++) {
            for (int k2 = 0; k2 < k; k2++) {
                if (kNeighborsFirst[k1] == kNeighborsSecond[k2]) {
                    result.add(kNeighborsSecond[k2]);
                    break;
                }
            }
        }

        return result;
    }

    /**
     * Counts how many neighbors this point shares with the training data.
     *
     * @param instance DataInstance that is being analyzed.
     * @return The count of shared neighbors between the instance and the
     * training data points.
     * @throws Exception
     */
    public float[] coundSharedNeighborsWithDataset(DataInstance instance)
            throws Exception {
        // First find the neighbors of the query instance.
        float[] kDistances = new float[k];
        for (int i = 0; i < k; i++) {
            kDistances[i] = Float.MAX_VALUE;
        }
        int[] kNeighbors = new int[k];
        float currDistance;
        int index;
        for (int i = 0; i < dset.size(); i++) {
            currDistance = cmet.dist(dset.data.get(i), instance);
            if (currDistance < kDistances[k - 1]) {
                // Insertion.
                index = k - 1;
                while (index > 0 && kDistances[index - 1] > currDistance) {
                    kDistances[index] = kDistances[index - 1];
                    kNeighbors[index] = kNeighbors[index - 1];
                    index--;
                }
                kDistances[index] = currDistance;
                kNeighbors[index] = i;
            }
        }
        if (!hashInitialized()) {
            initializeHashes();
        }
        float[] result = new float[dset.size()];
        for (int i = 0; i < dset.size(); i++) {
            for (int k1 = 0; k1 < k; k1++) {
                if (neighborHash[i].containsKey(kNeighbors[k1])) {
                    if (instanceWeights == null) {
                        result[i]++;
                    } else {
                        result[i] += instanceWeights[kNeighbors[k1]];
                    }
                }
            }
        }
        return result;
    }

    /**
     * Analyzes how many neighbors this point shares with the training data.
     *
     * @param instance DataInstance that is being analyzed.
     * @return The lists of shared neighbors between the instance and the
     * training data points.
     * @throws Exception
     */
    public ArrayList<Integer>[] findSharedNeighborsWithDataset(
            DataInstance instance) throws Exception {
        // First find the neighbors of the query instance.
        float[] kDistances = new float[k];
        for (int i = 0; i < k; i++) {
            kDistances[i] = Float.MAX_VALUE;
        }
        int[] kNeighbors = new int[k];
        float currDistance;
        int index;
        for (int i = 0; i < dset.size(); i++) {
            currDistance = cmet.dist(dset.data.get(i), instance);
            if (currDistance < kDistances[k - 1]) {
                // Insertion.
                index = k - 1;
                while (index > 0 && kDistances[index - 1] > currDistance) {
                    kDistances[index] = kDistances[index - 1];
                    kNeighbors[index] = kNeighbors[index - 1];
                    index--;
                }
                kDistances[index] = currDistance;
                kNeighbors[index] = i;
            }
        }
        if (!hashInitialized()) {
            initializeHashes();
        }
        ArrayList<Integer>[] result = new ArrayList[dset.size()];
        for (int i = 0; i < dset.size(); i++) {
            result[i] = new ArrayList<>(k);
            for (int k1 = 0; k1 < k; k1++) {
                if (neighborHash[i].containsKey(kNeighbors[k1])) {
                    result[i].add(kNeighbors[k1]);
                }
            }
        }
        return result;
    }

    /**
     * Gets the count of shared neighbors for an index pair, according to
     * previous calculations.
     *
     * @param firstIndex Index of the first data instance.
     * @param secondIndex Index of the second data instance.
     * @return The count of shared neighbors between the two.
     */
    public float getCountOfSharedNeighborsFor(int firstIndex, int secondIndex) {
        if (sharedNeighborCount == null || sharedNeighborCount.length
                <= firstIndex || sharedNeighborCount.length <= secondIndex) {
            return 0;
        }
        if (firstIndex == secondIndex) {
            return nsf.getKNeighbors()[firstIndex].length;
        }
        int min = Math.min(firstIndex, secondIndex);
        int max = Math.max(firstIndex, secondIndex);
        return sharedNeighborCount[min][max - min - 1];
    }

    /**
     * Gets the list of shared neighbors for an index pair, according to
     * previous calculations.
     *
     * @param firstIndex Index of the first data instance.
     * @param secondIndex Index of the second data instance.
     * @return The list of indexes of shared neighbors between the two.
     */
    public ArrayList<Integer> getSharedNeighborsFor(int firstIndex,
            int secondIndex) {
        if (sharedNeighborCount == null || sharedNeighborCount.length
                <= firstIndex || sharedNeighborCount.length <= secondIndex) {
            return null;
        }
        if (firstIndex == secondIndex) {
            ArrayList<Integer> result = new ArrayList<>(
                    nsf.getKNeighbors()[firstIndex].length);
            for (int index : nsf.getKNeighbors()[firstIndex]) {
                result.add(index);
            }
            return result;
        }
        int min = Math.min(firstIndex, secondIndex);
        int max = Math.max(firstIndex, secondIndex);
        return sharedNeighbors[min][max - min - 1];
    }

    /**
     * This method finds all the shared neighbors between pairs of points on the
     * training data and memorizes the lists and counts.
     */
    public void findSharedNeighbors() throws Exception {
        if (nsf == null && dset == null) {
            return;
        }
        if (nsf == null) {
            nsf = new NeighborSetFinder(dset, cmet);
        }
        if (dset == null) {
            dset = nsf.getDataSet();
        }
        if (nsf.getKNeighbors() == null || nsf.getKNeighbors().length == 0) {
            if (!nsf.distancesCalculated()) {
                nsf.calculateDistances();
            }
            nsf.calculateNeighborSets(k);
        }
        if (!hashInitialized()) {
            initializeHashes();
        }
        sharedNeighbors = new ArrayList[dset.size()][];
        sharedNeighborCount = new float[dset.size()][];
        for (int i = 0; i < dset.size(); i++) {
            sharedNeighbors[i] = new ArrayList[dset.size() - i - 1];
            sharedNeighborCount[i] = new float[dset.size() - i - 1];
            Set<Integer> keysFirst = neighborHash[i].keySet();
            for (int j = i + 1; j < dset.size(); j++) {
                sharedNeighbors[i][j - i - 1] = new ArrayList<>(k);
                for (int index : keysFirst) {
                    if (neighborHash[j].containsKey(index)) {
                        sharedNeighbors[i][j - i - 1].add(index);
                        if (instanceWeights == null) {
                            sharedNeighborCount[i][j - i - 1]++;
                        } else {
                            sharedNeighborCount[i][j - i - 1] +=
                                    instanceWeights[index];
                        }
                    }
                }
            }
        }
    }

    /**
     * This class is a worker class for multi-threaded shared-neighbor count
     * calculations.
     */
    class SharedNeighborCounterThread implements Runnable {

        float[][] sharedNeighborCount;
        int startIndex;
        int limitIndex;

        public SharedNeighborCounterThread(
                float[][] sharedNeighborCount, int startIndex, int limitIndex) {
            this.sharedNeighborCount = sharedNeighborCount;
            this.startIndex = startIndex;
            this.limitIndex = limitIndex;
        }

        @Override
        public void run() {
            try {
                for (int i = startIndex; i < limitIndex; i++) {
                    sharedNeighborCount[i] = new float[dset.size() - i - 1];
                    Set<Integer> keysFirst = neighborHash[i].keySet();
                    for (int j = i + 1; j < dset.size(); j++) {
                        for (int index : keysFirst) {
                            if (neighborHash[j].containsKey(index)) {
                                if (instanceWeights == null) {
                                    sharedNeighborCount[i][j - i - 1]++;
                                } else {
                                    sharedNeighborCount[i][j - i - 1] +=
                                            instanceWeights[index];
                                }
                            }
                        }
                    }
                }
            } catch (Exception e) {
                System.err.println("SNN multithread error.");
                System.err.println(e.getMessage());
                System.err.println("Limit indexes: " + startIndex + " "
                        + limitIndex);
            }
        }
    }

    /**
     * Count the shared neighbors in a multi-threaded way.
     *
     * @throws Exception
     */
    public void countSharedNeighborsMultiThread() throws Exception {
        countSharedNeighborsMultiThread(DEFAULT_NUM_THREADS);
    }

    /**
     * Count the shared neighbors in a multi-threaded way.
     *
     * @param numThreads Integer that is the number of threads to use.
     * @throws Exception
     */
    public void countSharedNeighborsMultiThread(int numThreads)
            throws Exception {
        if (nsf == null && dset == null) {
            return;
        }
        if (nsf == null) {
            nsf = new NeighborSetFinder(dset, cmet);
        }
        if (dset == null) {
            dset = nsf.getDataSet();
        }
        if (nsf.getKNeighbors() == null || nsf.getKNeighbors().length == 0) {
            if (!nsf.distancesCalculated()) {
                try {
                    nsf.setDistances(dset.
                            calculateDistMatrixMultThr(cmet, numThreads));
                } catch (Exception e) {
                    System.err.println(e.getMessage());
                }
            }
            nsf.calculateNeighborSets(k);
        }
        if (!hashInitialized()) {
            initializeHashes();
        }
        sharedNeighborCount = new float[dset.size()][];
        int[][] kneighbors = nsf.getKNeighbors();
        int size = kneighbors.length;
        int segLength = size / numThreads;
        Thread[] countThreads = new Thread[numThreads];
        for (int i = 0; i < numThreads - 1; i++) {
            countThreads[i] = new Thread(
                    new SharedNeighborCounterThread(
                    sharedNeighborCount,
                    i * segLength,
                    (i + 1) * segLength));
            countThreads[i].start();
        }
        countThreads[numThreads - 1] =
                new Thread(new SharedNeighborCounterThread(
                sharedNeighborCount, (numThreads - 1) * segLength, size));
        countThreads[numThreads - 1].start();
        for (int i = 0; i < numThreads; i++) {
            if (countThreads[i] != null) {
                try {
                    countThreads[i].join();
                } catch (Throwable t) {
                }
            }
        }
    }

    /**
     * This method counts all the shared neighbors between pairs of points on
     * the training data and memorizes the lists and counts.
     */
    public void countSharedNeighbors() throws Exception {
        if (nsf == null && dset == null) {
            return;
        }
        if (nsf == null) {
            nsf = new NeighborSetFinder(dset, cmet);
        }
        if (dset == null) {
            dset = nsf.getDataSet();
        }
        if (nsf.getKNeighbors() == null || nsf.getKNeighbors().length == 0) {
            if (!nsf.distancesCalculated()) {
                nsf.calculateDistances();
            }
            nsf.calculateNeighborSets(k);
        }
        if (!hashInitialized()) {
            initializeHashes();
        }
        sharedNeighborCount = new float[dset.size()][];
        for (int i = 0; i < dset.size(); i++) {
            sharedNeighborCount[i] = new float[dset.size() - i - 1];
            Set<Integer> keysFirst = neighborHash[i].keySet();
            for (int j = i + 1; j < dset.size(); j++) {
                for (int index : keysFirst) {
                    if (neighborHash[j].containsKey(index)) {
                        if (instanceWeights == null) {
                            sharedNeighborCount[i][j - i - 1]++;
                        } else {
                            sharedNeighborCount[i][j - i - 1] +=
                                    instanceWeights[index];
                        }
                    }
                }
            }
        }
    }
}
