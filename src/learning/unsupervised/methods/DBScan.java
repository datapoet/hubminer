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

import combinatorial.Permutation;
import data.neighbors.NeighborSetFinder;
import data.representation.DataInstance;
import data.representation.DataSet;
import distances.primary.CombinedMetric;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Stack;
import learning.unsupervised.Cluster;
import learning.unsupervised.ClusteringAlg;
import util.ArrayUtil;
import util.AuxSort;

/**
 * This class implements the well-known density based DBScan algorithm first
 * proposed in the following paper: Martin Ester, Hans-Peter Kriegel, Jörg
 * Sander, Xiaowei Xu (1996). "A density-based algorithm for discovering
 * clusters in large spatial databases with noise"
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class DBScan extends ClusteringAlg implements
        learning.supervised.interfaces.DistMatrixUserInterface,
        data.neighbors.NSFUserInterface {

    private float[][] distances = null;
    private int[] bestAssociations = null;
    // k is used to look for the proper epsilon, according to the kdistances.
    private int k = 10;
    private NeighborSetFinder nsf;
    // We keep an array of visited points.
    private boolean[] visited;
    // minPoints is the minimum number of points in a neighborhood for the point
    // not to be considered noise.
    private int minPoints;
    private float epsilonNeighborhoodDist = Float.MAX_VALUE;
    // Noise percentage should be carefully set.
    private float noisePerc = 0.15f;

    /**
     * This method searches for a good parameter configuration. This is achieved
     * by a pre-defined threshold bias where the distances to the k-th nearest
     * neighbor are sorted and then a certain number is discarded as noise. The
     * borderline k-distance is then taken as a limit.
     *
     * @throws Exception
     */
    public void searchForGoodParameters() throws Exception {
        float[][] kdistances = nsf.getKDistances();
        float[] kthdistance = new float[kdistances.length];
        for (int i = 0; i < kdistances.length; i++) {
            kthdistance[i] = kdistances[i][k - 1];
        }
        int[] rearrIndex = AuxSort.sortIndexedValue(kthdistance, true);
        minPoints = k;
        int threshold = (int) (noisePerc * rearrIndex.length);
        epsilonNeighborhoodDist = kthdistance[threshold];
    }

    /**
     * @return Integer that is the minimal number of points a neighborhood can
     * have not to be considered noise.
     */
    public int getMinPoints() {
        return minPoints;
    }

    /**
     * @param minPoints Integer that is the minimal number of points a
     * neighborhood can have not to be considered noise.
     */
    public void setMinPoints(int minPoints) {
        this.minPoints = minPoints;
    }

    /**
     * @return Epsilon-neighborhood diameter.
     */
    public float getEpsilon() {
        return epsilonNeighborhoodDist;
    }

    /**
     * @param epsilon Float that is the epsilon-neighborhood diameter.
     */
    public void setEpsilon(float epsilon) {
        this.epsilonNeighborhoodDist = epsilon;
    }

    public DBScan() {
    }

    /**
     * @param dset DataSet object for clustering.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Neighborhood size.
     * @param minPoints Minimal number of points in neighborhoods of non-noisy
     * data points.
     * @param epsilon Diameter of the epsilon-neighborhood.
     */
    public DBScan(DataSet dset, CombinedMetric cmet, int k, int minPoints,
            float epsilon) {
        setCombinedMetric(cmet);
        setDataSet(dset);
        this.k = k;
        this.minPoints = minPoints;
        epsilonNeighborhoodDist = epsilon;
    }

    /**
     * @param dset DataSet object for clustering.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Neighborhood size.
     */
    public DBScan(DataSet dset, CombinedMetric cmet, int k) {
        setCombinedMetric(cmet);
        setDataSet(dset);
        this.k = k;
    }

    /**
     * @param dset DataSet object for clustering.
     * @param k Neighborhood size.
     * @param minPoints Minimal number of points in neighborhoods of non-noisy
     * data points.
     * @param epsilon Diameter of the epsilon-neighborhood.
     */
    public DBScan(DataSet dset, int k, int minPoints, float epsilon) {
        setCombinedMetric(CombinedMetric.EUCLIDEAN);
        setDataSet(dset);
        this.k = k;
        this.minPoints = minPoints;
        epsilonNeighborhoodDist = epsilon;
    }

    /**
     * @param dset DataSet object for clustering.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Neighborhood size.
     * @param minPoints Minimal number of points in neighborhoods of non-noisy
     * data points.
     * @param epsilon Diameter of the epsilon-neighborhood.
     * @param noisePerc Expected percentage of noise in the data.
     */
    public DBScan(DataSet dset, CombinedMetric cmet, int k, int minPoints,
            float epsilon, float noisePerc) {
        setCombinedMetric(cmet);
        setDataSet(dset);
        this.k = k;
        this.minPoints = minPoints;
        epsilonNeighborhoodDist = epsilon;
        this.noisePerc = noisePerc;
    }

    /**
     * @param dset DataSet object for clustering.
     * @param k Neighborhood size.
     * @param minPoints Minimal number of points in neighborhoods of non-noisy
     * data points.
     * @param epsilon Diameter of the epsilon-neighborhood.
     * @param noisePerc Expected percentage of noise in the data.
     */
    public DBScan(DataSet dset, int k, int minPoints, float epsilon,
            float noisePerc) {
        setCombinedMetric(CombinedMetric.EUCLIDEAN);
        setDataSet(dset);
        this.k = k;
        this.minPoints = minPoints;
        epsilonNeighborhoodDist = epsilon;
        this.noisePerc = noisePerc;
    }

    @Override
    public void cluster() throws Exception {
        performBasicChecks();
        flagAsActive();
        DataSet dset = getDataSet();
        int size = dset.size();
        visited = new boolean[size];
        Arrays.fill(visited, false);
        int cNum = 0;
        ArrayList<Cluster> clusters = new ArrayList<>(10);
        bestAssociations = new int[size];
        Arrays.fill(bestAssociations, -1);
        if (epsilonNeighborhoodDist == Float.MAX_VALUE) {
            searchForGoodParameters();
        }
        int neighbSize;
        CombinedMetric cmet = getCombinedMetric();
        // Only calculates them if the current NeighborSetFinder object doesn't
        // have them properly calculated.
        calculateNeighborSets(k, cmet);
        int[] perm = Permutation.obtainRandomPermutation(size);
        for (int i = 0; i < size; i++) {
            if (!visited[perm[i]]) {
                visited[perm[i]] = true;
                neighbSize = queryNumNPoints(perm[i], nsf);
                if (neighbSize < minPoints) {
                    bestAssociations[perm[i]] = -1; // Marked as noise.
                } else {
                    cNum++;
                    Cluster clust = new Cluster(dset, size / 10);
                    expandCluster(perm[i], neighbSize, clust, cNum - 1);
                    clusters.add(clust);
                }
            }
        }
        setClusterAssociations(bestAssociations);
    }

    /**
     * Expands the cluster around the considered core point as much as possible.
     *
     * @param index Index of the considered data point.
     * @param neighbSize Neighborhood size.
     * @param currentCluster Current cluster.
     * @param clustIndex Current cluster index.
     */
    private void expandCluster(int index, int neighbSize,
            Cluster currentCluster, int clustIndex) {
        currentCluster.addInstance(index);
        bestAssociations[index] = clustIndex;
        Stack<Integer> potentialStack = new Stack<>();
        int[][] kneighbors = nsf.getKNeighbors();
        for (int i = 0; i < neighbSize; i++) {
            potentialStack.push(kneighbors[index][i]);
        }
        while (!potentialStack.empty()) {
            int i = potentialStack.pop();
            if (!visited[i]) {
                visited[i] = true;
                int nSize2 = queryNumNPoints(i, nsf);
                if (nSize2 >= minPoints) {
                    for (int j = 0; j < nSize2; j++) {
                        potentialStack.push(kneighbors[i][j]);
                    }
                }
            }
            if (bestAssociations[i] == -1) { // Not yet assigned to a cluster.
                // Assign it to the current cluster.
                bestAssociations[i] = clustIndex;
                currentCluster.addInstance(i);
            }
        }
    }

    /**
     * Counts how many neighbor points are at a distance closer than the epsilon
     * neighborhood diameter.
     *
     * @param index Index of the considered data point.
     * @param nsf NeighborSetFinder object.
     * @return An integer count representing the number of data points closer
     * than epsilon.
     */
    private int queryNumNPoints(int index, NeighborSetFinder nsf) {
        int closePointCounter = 0;
        float[][] kdistances = nsf.getKDistances();
        while (closePointCounter < kdistances[index].length
                && kdistances[index][closePointCounter]
                < epsilonNeighborhoodDist) {
            closePointCounter++;
        }
        return closePointCounter;
    }

    /**
     * Counts how many neighbor points are at a distance closer than the epsilon
     * neighborhood diameter.
     *
     * @param index Index of the considered data point.
     * @param nsf NeighborSetFinder object.
     * @param epsilonNeighborhoodTest Epsilon neighborhood diameter to use for
     * the count.
     * @return An integer count representing the number of data points closer
     * than epsilon.
     */
    private int queryNumNPoints(int index, NeighborSetFinder nsf,
            float epsilonNeighborhoodTest) {
        int closePointCounter = 0;
        float[][] kdistances = nsf.getKDistances();
        while (closePointCounter < kdistances[index].length
                && kdistances[index][closePointCounter]
                < epsilonNeighborhoodTest) {
            closePointCounter++;
        }
        return closePointCounter;
    }

    @Override
    public int[] assignPointsToModelClusters(DataSet dsetTest,
            NeighborSetFinder nsfTest) {
        if (dsetTest == null || dsetTest.isEmpty()) {
            return null;
        } else {
            int[] clusterAssociations = new int[dsetTest.size()];
            Arrays.fill(clusterAssociations, -1);
            CombinedMetric cmet = getCombinedMetric();
            Cluster[] clusterConfiguration = getClusters();
            int numClusters = clusterConfiguration.length;
            Cluster[] clusterConfigurationCopy = new Cluster[numClusters];
            DataInstance[] centroids = new DataInstance[numClusters];
            for (int i = 0; i < centroids.length; i++) {
                try {
                    centroids[i] = clusterConfiguration[i].getCentroid();
                } catch (Exception e) {
                }
            }
            for (int i = 0; i < clusterConfigurationCopy.length; i++) {
                clusterConfigurationCopy[i] = clusterConfiguration[i].copy();
            }
            int cNum = 0;
            int neighbSize = 0;
            float minDist;
            float currentDistance;
            float[][] kdistances = nsfTest.getKDistances();
            float[] kthdistance = new float[kdistances.length];
            for (int i = 0; i < kdistances.length; i++) {
                kthdistance[i] = kdistances[i][k - 1];
            }
            int[] rearrIndex;
            try {
                rearrIndex = AuxSort.sortIndexedValue(kthdistance, true);
            } catch (Exception e) {
                return null;
            }
            int threshold = (int) (noisePerc * rearrIndex.length);
            // And here we have the epsilon diameter for the test data.
            float epsilonNeighborhoodTest = kthdistance[threshold];
            boolean[] visitedTest = new boolean[dsetTest.size()];
            Arrays.fill(visitedTest, false);
            for (int i = 0; i < dsetTest.size(); i++) {
                if (!visitedTest[i]) {
                    visitedTest[i] = true;
                    neighbSize = queryNumNPoints(i, nsfTest,
                            epsilonNeighborhoodTest);
                    if (neighbSize < minPoints) {
                        clusterAssociations[i] = -1; // Marked as noise.
                    } else {
                        // Find the closest cluster.
                        cNum = 0;
                        minDist = Float.MAX_VALUE;
                        currentDistance = Float.MAX_VALUE;
                        for (int cIndex = 0; cIndex < numClusters; cIndex++) {
                            try {
                                currentDistance = cmet.dist(
                                        dsetTest.data.get(i),
                                        centroids[cIndex]);
                            } catch (Exception e) {
                            }
                            if (currentDistance < minDist) {
                                minDist = currentDistance;
                                cNum = cIndex;
                            }
                        }
                        expandTestCluster(
                                i,
                                neighbSize,
                                clusterConfigurationCopy[cNum],
                                cNum,
                                nsfTest,
                                clusterAssociations,
                                visitedTest,
                                epsilonNeighborhoodTest);
                    }
                }
            }
            return clusterAssociations;
        }
    }

    /**
     * Expands the cluster around the considered core test point as much as
     * possible.
     *
     * @param index Index of the observed test core point.
     * @param neighbSize Neighborhood size.
     * @param currCluster Current cluster.
     * @param clustIndex Index of the current cluster.
     * @param nsfTest NeighborSetFinder object on the test set.
     * @param clusterAssociations Cluster associations.
     * @param visitedTest Boolean array indicating which test points have been
     * visited up until this point.
     * @param epsilonNeighborhoodTest Epsilon neighborhood diameter on the test
     * data.
     */
    public void expandTestCluster(
            int index,
            int neighbSize,
            Cluster currCluster,
            int clustIndex,
            NeighborSetFinder nsfTest,
            int[] clusterAssociations,
            boolean[] visitedTest,
            float epsilonNeighborhoodTest) {
        currCluster.addInstance(index);
        clusterAssociations[index] = clustIndex;
        Stack<Integer> potentialStack = new Stack<>();
        int[][] kneighbors = nsfTest.getKNeighbors();
        for (int i = 0; i < neighbSize; i++) {
            potentialStack.push(kneighbors[index][i]);
        }
        while (!potentialStack.empty()) {
            int i = potentialStack.pop();
            if (!visitedTest[i]) {
                visitedTest[i] = true;
                int nSize2 = queryNumNPoints(i, nsfTest,
                        epsilonNeighborhoodTest);
                if (nSize2 >= minPoints) {
                    for (int j = 0; j < nSize2; j++) {
                        potentialStack.push(kneighbors[i][j]);
                    }
                }
            }
            if (clusterAssociations[i] == -1) {
                // Not yet assigned to a cluster, so we assign it here.
                clusterAssociations[i] = clustIndex;
                currCluster.addInstance(i);
            }
        }
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
    } // A dummy method here, to satisfy the interface.

    /**
     * @return Best cluster associations.
     */
    public int[] getMinimizingAssociations() {
        return bestAssociations;
    }

    @Override
    public Cluster[] getMinimizingClusters() {
        int[] clusterAssociations = getMinimizingAssociations();
        int numClusters = ArrayUtil.max(clusterAssociations) + 1;
        Cluster[] clusters = new Cluster[numClusters];
        DataSet dset = getDataSet();
        for (int i = 0; i < numClusters; i++) {
            int initSize = Math.max(dset.size() / numClusters, 100);
            clusters[i] = new Cluster(dset, initSize);
        }
        if ((dset != null) && (clusterAssociations != null)) {
            for (int i = 0; i < dset.size(); i++) {
                if (clusterAssociations[i] >= 0) {
                    clusters[clusterAssociations[i]].addInstance(i);
                }
            }
        }
        return clusters;
    }

    /**
     * @return Data matrix of float distances.
     */
    public float[][] getAllDistances() {
        return nsf.getDistances();
    }

    /**
     * @param k Neighborhood size.
     * @param cmet CombinedMetric object.
     * @throws Exception
     */
    private void calculateNeighborSets(int k, CombinedMetric cmet)
            throws Exception {
        if (nsf == null) {
            if (cmet == null) {
                cmet = CombinedMetric.EUCLIDEAN;
            }
            nsf = new NeighborSetFinder(getDataSet(), cmet);
            if (distances != null) {
                nsf.setDistances(distances);
            } else {
                nsf.calculateDistances();
            }
            nsf.calculateNeighborSets(k);
        } else {
            if (!nsf.isCalculatedUpToK(k)) {
                if (nsf.getDistances() == null) {
                    if (distances != null) {
                        nsf.setDistances(distances);
                    } else {
                        nsf.calculateDistances();
                    }
                }
            }
        }
    }

    /**
     * @param k Neighborhood size.
     */
    public void setK(int k) {
        this.k = k;
    }
    
    @Override
    public int getNeighborhoodSize() {
        return k;
    }
}
