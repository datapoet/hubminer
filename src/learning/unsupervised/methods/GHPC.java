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
package learning.unsupervised.methods;

import data.neighbors.NeighborSetFinder;
import data.representation.DataInstance;
import data.representation.DataSet;
import data.representation.util.DataMineConstants;
import distances.primary.CombinedMetric;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import learning.unsupervised.Cluster;
import learning.unsupervised.ClusteringAlg;
import learning.unsupervised.initialization.PlusPlusSeeder;
import util.ArrayUtil;

/**
 * Global Hubness-proportional Clustering that was analyzed in the paper titled
 * "The Role of Hubness in Clustering High-dimensional Data", which was
 * presented at PAKDD in 2011. Hubness is calculated locally here, within the
 * clusters.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class GHPC extends ClusteringAlg implements
        learning.supervised.interfaces.DistMatrixUserInterface,
        data.neighbors.NSFUserInterface {

    public static final int UNSUPERVISED = 0;
    public static final int SUPERVISED = 1;
    private static final double ERROR_THRESHOLD = 0.001;
    private static final int MAX_ITER = 100;
    boolean unsupervisedHubness = true;
    private float[][] distances = null;
    private double[] cumulativeProbabilities = null;
    private float smallestError = Float.MAX_VALUE;
    private int[] bestAssociations = null;
    int[] hubnessArray = null;
    private int k = 10;
    private NeighborSetFinder nsf;
    public int probabilisticIterations = 20;
    boolean history = false;
    ArrayList<int[]> historyArrayList;
    DataInstance[] endCentroids = null;

    public GHPC() {
    }

    /**
     * @param dset DataSet object.
     * @param cmet CombinedMetric object for distance calculations.
     * @param numClusters A pre-defined number of clusters.
     * @param k Neighborhood size.
     */
    public GHPC(DataSet dset, CombinedMetric cmet, int numClusters, int k) {
        setNumClusters(numClusters);
        setCombinedMetric(cmet);
        setDataSet(dset);
        this.k = k;
    }

    /**
     * @param dset DataSet object.
     * @param numClusters A pre-defined number of clusters.
     * @param k Neighborhood size.
     */
    public GHPC(DataSet dset, int numClusters, int k) {
        setNumClusters(numClusters);
        setCombinedMetric(CombinedMetric.EUCLIDEAN);
        setDataSet(dset);
        this.k = k;
    }

    @Override
    public void cluster() throws Exception {
        performBasicChecks();
        flagAsActive();
        DataSet dset = getDataSet();
        CombinedMetric cmet = getCombinedMetric();
        int numClusters = getNumClusters();
        cmet = cmet != null ? cmet : CombinedMetric.EUCLIDEAN;
        boolean trivial = checkIfTrivial();
        if (trivial) {
            return;
        } // Nothing needs to be done in this case.
        int[] clusterAssociations = new int[dset.size()];
        Arrays.fill(clusterAssociations, 0, dset.size(), -1);
        setClusterAssociations(clusterAssociations);
        PlusPlusSeeder seeder =
                new PlusPlusSeeder(numClusters, dset.data, cmet);
        int[] clusterHubIndexes = seeder.getCentroidIndexes();
        DataInstance[] clusterHubs = new DataInstance[numClusters];
        for (int cIndex = 0; cIndex < numClusters; cIndex++) {
            clusterAssociations[clusterHubIndexes[cIndex]] = cIndex;
            clusterHubs[cIndex] = (dset.data.get(clusterHubIndexes[cIndex]));
        }
        Cluster[] clusters;
        if (hubnessArray == null) {
            calculateHubness(k, cmet);
        }
        if (history) {
            historyArrayList = new ArrayList<>(2 * MAX_ITER);
        }
        try {
            if (distances == null) {
                distances = getNSFDistances();
            }
        } catch (Exception e) {
        }
        if (distances == null) {
            distances = dset.calculateDistMatrixMultThr(cmet, 4);
        }
        double errorPrevious;
        double errorCurrent = Double.MAX_VALUE;
        setIterationIndex(0);
        boolean noReassignments;
        boolean errorDifferenceSignificant = true;
        int fi, se;
        int closestHubIndex;
        float smallestDistance;
        float currentDistance;
        int[] iterHubInd;
        // It's best if the first assignment is done before and if the
        // assignments are done at the end of the do-while loop, therefore
        // allowing for better calculateIterationError estimates.
        for (int i = 0; i < dset.size(); i++) {
            if (history) {
                iterHubInd = new int[clusterHubIndexes.length];
                System.arraycopy(clusterHubIndexes, 0, iterHubInd, 0,
                        clusterHubIndexes.length);
                historyArrayList.add(iterHubInd);
            }
            closestHubIndex = -1;
            smallestDistance = Float.MAX_VALUE;
            for (int cIndex = 0; cIndex < numClusters; cIndex++) {
                if (clusterHubIndexes[cIndex] >= 0) {
                    if (clusterHubIndexes[cIndex] != i) {
                        fi = Math.min(i, clusterHubIndexes[cIndex]);
                        se = Math.max(i, clusterHubIndexes[cIndex]);
                        currentDistance = distances[fi][se - fi - 1];
                    } else {
                        closestHubIndex = cIndex;
                        break;
                    }
                } else {
                    currentDistance = cmet.dist(dset.getInstance(i),
                            clusterHubs[cIndex]);
                }
                if (currentDistance < smallestDistance) {
                    smallestDistance = currentDistance;
                    closestHubIndex = cIndex;
                }
            }
            clusterAssociations[i] = closestHubIndex;
        }
        do {
            nextIteration();
            noReassignments = true;
            clusters = getClusters();
            int first, second;
            int maxFrequency;
            int maxIndex;
            int maxActualIndex = 0;
            int currSize;
            float probDet = getProbFromSchedule();
            for (int cIndex = 0; cIndex < clusters.length; cIndex++) {
                currSize = clusters[cIndex].indexes.size();
                if (currSize == 1) {
                    clusterHubs[cIndex] = clusters[cIndex].getInstance(0);
                    clusterHubIndexes[cIndex] = clusters[cIndex].indexes.get(0);
                    continue;
                }
                Random randa = new Random();
                double decision = randa.nextFloat();
                if (decision > probDet) {
                    // A squared hubness proportional stochastic selection.
                    cumulativeProbabilities = new double[currSize];
                    if (currSize == 0) {
                        continue;
                    }
                    cumulativeProbabilities[0] = 0;
                    for (int j = 1; j < currSize; j++) {
                        cumulativeProbabilities[j] =
                                cumulativeProbabilities[j - 1]
                                + hubnessArray[clusters[cIndex].indexes.get(j)]
                                * hubnessArray[clusters[cIndex].indexes.get(j)];
                    }
                    decision = randa.nextFloat()
                            * cumulativeProbabilities[currSize - 1];
                    int foundIndex = findIndex(decision, 0, currSize - 1);
                    clusterHubIndexes[cIndex] =
                            clusters[cIndex].indexes.get(foundIndex);
                    clusterHubs[cIndex] = dset.getInstance(
                            clusters[cIndex].indexes.get(foundIndex));
                } else {
                    // Deterministic approach.
                    maxFrequency = 0;
                    maxIndex = 0;
                    for (int j = 0; j < currSize; j++) {
                        if (hubnessArray[clusters[cIndex].indexes.get(j)]
                                > maxFrequency) {
                            maxFrequency = hubnessArray[
                                    clusters[cIndex].indexes.get(j)];
                            maxIndex = j;
                            maxActualIndex = clusters[cIndex].indexes.get(j);
                        }
                    }
                    clusterHubs[cIndex] =
                            clusters[cIndex].getInstance(maxIndex);
                    clusterHubIndexes[cIndex] = maxActualIndex;
                }
            }
            if (history) {
                iterHubInd = new int[clusterHubIndexes.length];
                System.arraycopy(clusterHubIndexes, 0, iterHubInd, 0,
                        clusterHubIndexes.length);
                historyArrayList.add(iterHubInd);
            }
            for (int i = 0; i < dset.size(); i++) {
                closestHubIndex = -1;
                smallestDistance = Float.MAX_VALUE;
                for (int cIndex = 0; cIndex < numClusters; cIndex++) {
                    if (clusterHubIndexes[cIndex] > 0) {
                        if (clusterHubIndexes[cIndex] != i) {
                            first = Math.min(i, clusterHubIndexes[cIndex]);
                            second = Math.max(i, clusterHubIndexes[cIndex]);
                            currentDistance =
                                    distances[first][second - first - 1];
                        } else {
                            closestHubIndex = cIndex;
                            break;
                        }
                    } else {
                        currentDistance = cmet.dist(dset.getInstance(i),
                                clusterHubs[cIndex]);
                    }
                    if (currentDistance < smallestDistance) {
                        smallestDistance = currentDistance;
                        closestHubIndex = cIndex;
                    }
                }
                if (closestHubIndex != clusterAssociations[i]) {
                    noReassignments = false;
                }
                clusterAssociations[i] = closestHubIndex;
            }
            errorPrevious = errorCurrent;
            errorCurrent = calculateIterationError(clusterHubs,
                    clusterHubIndexes);
            if (errorCurrent < smallestError) {
                bestAssociations = clusterAssociations;
            }
            if (getIterationIndex() >= probabilisticIterations) {
                if (DataMineConstants.isAcceptableDouble(errorPrevious)
                        && DataMineConstants.isAcceptableDouble(errorCurrent)
                        && (Math.abs(errorCurrent / errorPrevious) - 1f)
                        < ERROR_THRESHOLD) {
                    errorDifferenceSignificant = false;
                } else {
                    errorDifferenceSignificant = true;
                }
            }
        } while (errorDifferenceSignificant && !noReassignments
                && getIterationIndex() < MAX_ITER);
        endCentroids = clusterHubs;
        setClusterAssociations(clusterAssociations);
        flagAsInactive();
    }

    /**
     * @param clusterHubs An array of cluster hubs.
     * @param clusterHubIndexes An array of indexes of cluster hubs.
     * @return The current iteration squared error.
     * @throws Exception
     */
    private double calculateIterationError(DataInstance[] clusterHubs,
            int[] clusterHubIndexes) throws Exception {
        int[] clusterAssociations = getClusterAssociations();
        double error = 0;
        int first;
        int second;
        CombinedMetric cmet = getCombinedMetric();
        DataSet dset = getDataSet();
        for (int i = 0; i < clusterAssociations.length; i++) {
            if (clusterHubIndexes[clusterAssociations[i]] != -1) {
                if (clusterHubIndexes[clusterAssociations[i]] != i) {
                    first = Math.min(i,
                            clusterHubIndexes[clusterAssociations[i]]);
                    second = Math.max(i,
                            clusterHubIndexes[clusterAssociations[i]]);
                    error += distances[first][second - first - 1]
                            * distances[first][second - first - 1];
                }
            } else {
                error += Math.pow(cmet.dist(
                        clusterHubs[clusterAssociations[i]],
                        dset.data.get(i)), 2);
            }
        }
        return error;
    }

    /**
     * @return Probability of a deterministic iteration.
     */
    private float getProbFromSchedule() {
        int iteration = getIterationIndex();
        if (iteration < probabilisticIterations) {
            return (float) ((float) iteration /
                    (float) probabilisticIterations);
        } else {
            return 1f;
        }
    }

    /**
     * Binary search over cumulative squared neighbor occurrence frequencies.
     *
     * @param searchValue Query value.
     * @param first First index.
     * @param second Second index.
     * @return Match index.
     */
    public int findIndex(double searchValue, int first, int second) {
        if (second - first <= 1) {
            return second;
        }
        int middle = (first + second) / 2;
        if (cumulativeProbabilities[middle] < searchValue) {
            return findIndex(searchValue, middle, second);
        } else {
            return findIndex(searchValue, first, middle);
        }
    }

    @Override
    public int[] assignPointsToModelClusters(DataSet dsetTest,
            NeighborSetFinder nsfTest) {
        if (dsetTest == null || dsetTest.isEmpty()) {
            return null;
        } else {
            int[] clusterAssociations = new int[dsetTest.size()];
            if (endCentroids == null) {
                return clusterAssociations;
            }
            float minDist;
            float dist;
            CombinedMetric cmet = getCombinedMetric();
            cmet = cmet != null ? cmet : CombinedMetric.EUCLIDEAN;
            for (int i = 0; i < dsetTest.size(); i++) {
                minDist = Float.MAX_VALUE;
                for (int cIndex = 0; cIndex < endCentroids.length; cIndex++) {
                    dist = Float.MAX_VALUE;
                    try {
                        dist = cmet.dist(
                                endCentroids[cIndex], dsetTest.getInstance(i));
                    } catch (Exception e) {
                    }
                    if (dist < minDist) {
                        clusterAssociations[i] = cIndex;
                        minDist = dist;
                    }
                }
            }
            return clusterAssociations;
        }
    }

    /**
     * @param hubnessArray Integer array of neighbor occurrence frequencies.
     */
    public void setHubness(int[] hubnessArray) {
        this.hubnessArray = hubnessArray;
    }

    /**
     * @param k Neighborhood size.
     * @param cmet CombinedMetric object.
     * @throws Exception
     */
    public void calculateHubness(int k, CombinedMetric cmet) throws Exception {
        if (nsf == null) {
            if (cmet == null) {
                cmet = CombinedMetric.EUCLIDEAN;
            }
            nsf = new NeighborSetFinder(getDataSet(), cmet);
            nsf.calculateDistances();
            nsf.calculateNeighborSets(k);
            if (unsupervisedHubness) {
                hubnessArray = nsf.getNeighborFrequencies();
            } else {
                float[] weights =
                        nsf.getSimhubAlternateWeightsGoodnessProportional(k);
                int[] ghArray = nsf.getGoodFrequencies();
                int[] bhArray = nsf.getBadFrequencies();
                float[] sArray = new float[ghArray.length];
                for (int i = 0; i < sArray.length; i++) {
                    sArray[i] = (ghArray[i] + bhArray[i]) * weights[i];
                }
                // If the data has much more bad than good hubness, one has to
                // be careful - since hubnessArray is an int array, things would
                // get truncated to zero easily. Hence, some lightweight
                // re-scaling to fix the issue if it exists. Re-scaling doesn't
                // affect the algorithm anyway, but it has to be done since
                // hubnessArray is not float.
                float mean = ArrayUtil.findMean(sArray);
                float max = ArrayUtil.max(sArray);
                float fact = 1;
                if (mean <= 1 || max < 50) {
                    fact = Math.max(100, 1 / mean);
                }
                hubnessArray = new int[sArray.length];
                for (int i = 0; i < sArray.length; i++) {
                    hubnessArray[i] = (int) (sArray[i] * fact);
                }
            }
        } else {
            // If the nsf is provided, the hubness scores only need to be
            // recalculated for a smaller k - in case a larger one is given
            // in the nsf.
            if (nsf.getCurrK() > k) {
                nsf = nsf.getSubNSF(k);
            }
            if (unsupervisedHubness) {
                hubnessArray = nsf.getNeighborFrequencies();
            } else {
                float[] weights =
                        nsf.getSimhubAlternateWeightsGoodnessProportional(k);
                int[] ghArray = nsf.getGoodFrequencies();
                int[] bhArray = nsf.getBadFrequencies();
                float[] sArray = new float[ghArray.length];
                for (int i = 0; i < sArray.length; i++) {
                    sArray[i] = (ghArray[i] + bhArray[i]) * weights[i];
                }
                // If the data has much more bad than good hubness, one has to
                // be careful - since hubnessArray is an int array, things would
                // get truncated to zero easily. Hence, some lightweight
                // re-scaling to fix the issue if it exists. Re-scaling doesn't
                // affect the algorithm anyway, but it has to be done since
                // hubnessArray is not float.
                float mean = ArrayUtil.findMean(sArray);
                float max = ArrayUtil.max(sArray);
                float fact = 1;
                if (mean <= 1 || max < 50) {
                    fact = Math.max(100, 1 / mean);
                }
                hubnessArray = new int[sArray.length];
                for (int i = 0; i < sArray.length; i++) {
                    hubnessArray[i] = (int) (sArray[i] * fact);
                }
            }
        }
    }

    /**
     * @return Distances from the NeighborSetFinder object.
     */
    public float[][] getNSFDistances() {
        return nsf.getDistances();
    }

    /**
     * @param k Neighborhood size.
     */
    public void setK(int k) {
        this.k = k;
    }

    @Override
    public void setDistMatrix(float[][] distances) {
        this.distances = distances;
    }

    @Override
    public float[][] getDistMatrix() {
        return distances;
    }

    @Override
    public void setNSF(NeighborSetFinder nsf) {
        this.nsf = nsf;
    }

    @Override
    public NeighborSetFinder getNSF() {
        return nsf;
    }

    @Override
    public void noRecalcs() {
    } // A dummy method.

    @Override
    public void setMinIterations(int numIter) {
        probabilisticIterations = numIter;
    }

    /**
     * Sets whether to use supervised or unsupervised hubness.
     *
     * @param mode Hubness mode.
     */
    public void setHubnessMode(int mode) {
        if (mode % 2 == UNSUPERVISED) {
            unsupervisedHubness = true;
        } else {
            unsupervisedHubness = false;
        }
    }

    /**
     * @return Integer array of neighbor occurrence frequencies.
     */
    public int[] getHubnessArray() {
        return hubnessArray;
    }

    /**
     * @param history Boolean variable determining whether to keep track of all
     * hub selections.
     */
    public void keepHistory(boolean history) {
        this.history = history;
    }

    /**
     * @return An array list of int arrays containing hub selections per
     * clustering iterations.
     */
    public ArrayList<int[]> getHubHistory() {
        return historyArrayList;
    }

    /**
     * @return Error-minimizing cluster associations.
     */
    public int[] getMinimizingAssociations() {
        return bestAssociations;
    }

    @Override
    public Cluster[] getMinimizingClusters() {
        int numClusters = getNumClusters();
        Cluster[] clusters = new Cluster[numClusters];
        int[] clusterAssociations = getMinimizingAssociations();
        DataSet dset = getDataSet();
        for (int cIndex = 0; cIndex < numClusters; cIndex++) {
            int initSize = Math.max(dset.size() / numClusters, 100);
            clusters[cIndex] = new Cluster(dset, initSize);
        }
        if ((dset != null) && (clusterAssociations != null)) {
            for (int i = 0; i < dset.size(); i++) {
                clusters[clusterAssociations[i]].addInstance(i);
            }
        }
        return clusters;
    }
    
    @Override
    public int getNeighborhoodSize() {
        return k;
    }
}
