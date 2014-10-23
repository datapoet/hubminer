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
import java.util.Arrays;
import java.util.HashMap;
import java.util.Random;
import learning.unsupervised.Cluster;
import learning.unsupervised.ClusteringAlg;
import learning.unsupervised.initialization.PlusPlusSeeder;

/**
 * Local Hubness Proportional Clustering that was analyzed in the paper titled
 * "The Role of Hubness in Clustering High-dimensional Data", which was
 * presented at PAKDD in 2011. Hubness is calculated locally here, within the
 * clusters.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class LHPC extends ClusteringAlg
        implements learning.supervised.interfaces.DistMatrixUserInterface {

    private static final double ERROR_THRESHOLD = 0.001;
    private static final int MAX_ITER = 45;
    private int[] bestAssociations = null;
    private float[][] distances = null;
    private double[] cumulativeProbabilities = null;
    private int k = 5;
    DataInstance[] endCentroids;
    
    @Override
    public HashMap<String, String> getParameterNamesAndDescriptions() {
        HashMap<String, String> paramMap = new HashMap<>();
        paramMap.put("k", "Neighborhood size.");
        return paramMap;
    }

    public LHPC() {
    }

    /**
     * @param dset DataSet object for clustering.
     * @param cmet CombinedMetric object for distance calculations.
     * @param numClusters A pre-defined number of clusters.
     * @param k Neighborhood size.
     */
    public LHPC(DataSet dset, CombinedMetric cmet, int numClusters, int k) {
        setNumClusters(numClusters);
        setCombinedMetric(cmet);
        setDataSet(dset);
        this.k = k;
    }

    /**
     * @param dset
     * @param numClusters
     * @param k
     */
    public LHPC(DataSet dset, int numClusters, int k) {
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
        distances = new float[dset.size()][];
        for (int i = 0; i < distances.length; i++) {
            distances[i] = new float[distances.length - i - 1];
            for (int j = 0; j < distances[i].length; j++) {
                distances[i][j] = -1;
                // Indicating that this distance hasn't been calculated yet.
                // It's a speed up trick so that not all n^2 / 2 distances need
                // to be calculated in order to find hubs.
            }
        }
        DataInstance[] clusterHubs = new DataInstance[numClusters];
        PlusPlusSeeder seeder = new PlusPlusSeeder(numClusters,
                dset.data, cmet);
        int[] clusterHubIndexes = seeder.getCentroidIndexes();
        for (int cIndex = 0; cIndex < numClusters; cIndex++) {
            clusterAssociations[clusterHubIndexes[cIndex]] = cIndex;
            clusterHubs[cIndex] = (dset.data.get(clusterHubIndexes[cIndex]));
        }
        Cluster[] clusters;
        double errorPrevious;
        double errorCurrent = Double.MAX_VALUE;
        setIterationIndex(0);
        boolean noReassignments;
        boolean errorDifferenceSignificant = true;
        int fi, se;
        int closestHub;
        float smallestDistance;
        float currentDistance;
        int foundIndex;
        // It's best if the first assignment is done before and if the
        // assignments are done at the end of the do-while loop, therefore
        // allowing for better calculateIterationError estimates.
        for (int i = 0; i < clusterAssociations.length; i++) {
            closestHub = -1;
            smallestDistance = Float.MAX_VALUE;
            for (int cIndex = 0; cIndex < numClusters; cIndex++) {
                if (clusterHubIndexes[cIndex] > 0) {
                    if (clusterHubIndexes[cIndex] != i) {
                        fi = Math.min(i, clusterHubIndexes[cIndex]);
                        se = Math.max(i, clusterHubIndexes[cIndex]);
                        if (distances[fi][se - fi - 1] <= 0) {
                            distances[fi][se - fi - 1] =
                                    cmet.dist(dset.data.get(fi),
                                    dset.data.get(se));
                        }
                        currentDistance = distances[fi][se - fi - 1];
                    } else {
                        closestHub = cIndex;
                        break;
                    }
                } else {
                    currentDistance = cmet.dist(dset.data.get(i),
                            clusterHubs[cIndex]);
                }
                if (currentDistance < smallestDistance) {
                    smallestDistance = currentDistance;
                    closestHub = cIndex;
                }
            }
            clusterAssociations[i] = closestHub;
        }
        do {
            nextIteration();
            noReassignments = true;
            clusters = getClusters();
            int first, second;
            int[][] kneighbors;
            float[][] kdistances;
            int[] kcurrLen;
            int[] kneighborFrequencies;
            int maxFrequency;
            int maxIndex;
            int maxActualIndex;
            int currSize;
            for (int cIndex = 0; cIndex < numClusters; cIndex++) {
                currSize = clusters[cIndex].indexes.size();
                if (currSize == 1) {
                    clusterHubs[cIndex] = clusters[cIndex].getInstance(0);
                    clusterHubIndexes[cIndex] =
                            clusters[cIndex].indexes.get(0);
                    continue;
                }
                if (currSize < k + 2) {
                    clusterHubs[cIndex] = clusters[cIndex].getCentroid();
                    clusterHubIndexes[cIndex] = -1;
                    continue;
                }
                kneighbors = new int[currSize][k];
                kdistances = new float[currSize][k];
                kcurrLen = new int[currSize];
                int temp;
                float currDistance;
                for (int j = 0; j < currSize; j++) {
                    for (int l = j + 1; l < currSize; l++) {
                        first = clusters[cIndex].indexes.get(j);
                        second = clusters[cIndex].indexes.get(l);
                        if (distances[first][second - first - 1] < 0) {
                            distances[first][second - first - 1] =
                                    cmet.dist(dset.data.get(first),
                                    dset.data.get(second));
                        }
                        currDistance = distances[first][second - first - 1];
                        if (kcurrLen[j] > 0) {
                            if (kcurrLen[j] == k) {
                                if (currDistance < kdistances[j][k - 1]) {
                                    temp = k - 1;
                                    while ((temp >= 0)
                                            && currDistance <
                                            kdistances[j][temp]) {
                                        if (temp > 0) {
                                            kdistances[j][temp] =
                                                    kdistances[j][temp - 1];
                                            kneighbors[j][temp] =
                                                    kneighbors[j][temp - 1];
                                        }
                                        temp--;
                                    }
                                    kdistances[j][temp + 1] = currDistance;
                                    kneighbors[j][temp + 1] = l;
                                }
                            } else {
                                if (currDistance
                                        < kdistances[j][kcurrLen[j] - 1]) {
                                    temp = kcurrLen[j] - 1;
                                    kdistances[j][kcurrLen[j]] =
                                            kdistances[j][kcurrLen[j] - 1];
                                    kneighbors[j][kcurrLen[j]] =
                                            kneighbors[j][kcurrLen[j] - 1];
                                    while ((temp >= 0)
                                            && currDistance
                                            < kdistances[j][temp - 1]) {
                                        if (temp > 0) {
                                            kdistances[j][temp] =
                                                    kdistances[j][temp - 1];
                                            kneighbors[j][temp] =
                                                    kneighbors[j][temp - 1];
                                        }
                                        temp--;
                                    }
                                    kdistances[j][temp] = currDistance;
                                    kneighbors[j][temp] = l;
                                    kcurrLen[j]++;
                                } else {
                                    kdistances[j][kcurrLen[j]] = currDistance;
                                    kneighbors[j][kcurrLen[j]] = l;
                                    kcurrLen[j]++;
                                }
                            }
                        } else {
                            kdistances[j][0] = currDistance;
                            kneighbors[j][0] = l;
                            kcurrLen[j] = 1;
                        }
                        if (kcurrLen[l] > 0) {
                            if (kcurrLen[l] == k) {
                                if (currDistance < kdistances[l][k - 1]) {
                                    temp = k - 1;
                                    while ((temp >= 0)
                                            && currDistance <
                                            kdistances[l][temp]) {
                                        if (temp > 0) {
                                            kdistances[l][temp] =
                                                    kdistances[l][temp - 1];
                                            kneighbors[l][temp] =
                                                    kneighbors[l][temp - 1];
                                        }
                                        temp--;
                                    }
                                    kdistances[l][temp + 1] = currDistance;
                                    kneighbors[l][temp + 1] = j;
                                }
                            } else {
                                if (currDistance
                                        < kdistances[l][kcurrLen[l] - 1]) {
                                    temp = kcurrLen[l] - 1;
                                    kdistances[l][kcurrLen[l]] =
                                            kdistances[l][kcurrLen[l] - 1];
                                    kneighbors[l][kcurrLen[l]] =
                                            kneighbors[l][kcurrLen[l] - 1];
                                    while ((temp >= 0)
                                            && currDistance
                                            < kdistances[l][temp - 1]) {
                                        if (temp > 0) {
                                            kdistances[l][temp] =
                                                    kdistances[l][temp - 1];
                                            kneighbors[l][temp] =
                                                    kneighbors[l][temp - 1];
                                        }
                                        temp--;
                                    }
                                    kdistances[l][temp] = currDistance;
                                    kneighbors[l][temp] = j;
                                    kcurrLen[l]++;
                                } else {
                                    kdistances[l][kcurrLen[l]] = currDistance;
                                    kneighbors[l][kcurrLen[l]] = j;
                                    kcurrLen[l]++;
                                }
                            }
                        } else {
                            kdistances[l][0] = currDistance;
                            kneighbors[l][0] = j;
                            kcurrLen[l] = 1;
                        }
                    }
                }
                // Now calculate the locally restricted neighbor occurrence
                // frequencies.
                kneighborFrequencies = new int[kneighbors.length];
                for (int j = 0; j < kneighbors.length; j++) {
                    for (int l = 0; l < k; l++) {
                        kneighborFrequencies[kneighbors[j][l]]++;
                    }
                }
                float probDet = getProbFromSchedule();
                Random randa = new Random();
                double decision = randa.nextFloat();
                if (decision < probDet) {
                    // Deterministic k-hubs approach.
                    maxFrequency = 0;
                    maxIndex = 0;
                    for (int j = 0; j < kneighborFrequencies.length; j++) {
                        if (kneighborFrequencies[j] > maxFrequency) {
                            maxIndex = j;
                        }
                    }
                    maxActualIndex = clusters[cIndex].indexes.get(maxIndex);
                    clusterHubIndexes[cIndex] = maxActualIndex;
                    clusterHubs[cIndex] = dset.data.get(maxActualIndex);
                } else {
                    // Squared hubness proportional regime.
                    cumulativeProbabilities = new double[kneighbors.length];
                    cumulativeProbabilities[0] = 0;
                    for (int j = 1; j < kneighbors.length; j++) {
                        cumulativeProbabilities[j] =
                                cumulativeProbabilities[j - 1]
                                + kneighborFrequencies[j]
                                * kneighborFrequencies[j];
                    }
                    decision = randa.nextFloat()
                            * cumulativeProbabilities[kneighbors.length - 1];
                    foundIndex = findIndex(decision, 0, kneighbors.length - 1);
                    clusterHubIndexes[cIndex] =
                            clusters[cIndex].indexes.get(foundIndex);
                    clusterHubs[cIndex] = dset.data.get(
                            clusters[cIndex].indexes.get(foundIndex));
                }
            }
            for (int i = 0; i < clusterAssociations.length; i++) {
                closestHub = -1;
                smallestDistance = Float.MAX_VALUE;
                for (int cIndex = 0; cIndex < numClusters; cIndex++) {
                    if (clusterHubIndexes[cIndex] > 0) {
                        if (clusterHubIndexes[cIndex] != i) {
                            first = Math.min(i, clusterHubIndexes[cIndex]);
                            second = Math.max(i, clusterHubIndexes[cIndex]);
                            if (distances[first][second - first - 1] <= 0) {
                                distances[first][second - first - 1] =
                                        cmet.dist(dset.data.get(first),
                                        dset.data.get(second));
                            }
                            currentDistance =
                                    distances[first][second - first - 1];
                        } else {
                            closestHub = cIndex;
                            break;
                        }
                    } else {
                        currentDistance =
                                cmet.dist(dset.data.get(i),
                                clusterHubs[cIndex]);
                    }
                    if (currentDistance < smallestDistance) {
                        smallestDistance = currentDistance;
                        closestHub = cIndex;
                    }
                }
                if (closestHub != clusterAssociations[i]) {
                    noReassignments = false;
                }
                clusterAssociations[i] = closestHub;
            }
            errorPrevious = errorCurrent;
            errorCurrent = calculateIterationError(clusterHubs,
                    clusterHubIndexes);
            if (getIterationIndex() >= MIN_ITERATIONS) {
                if (DataMineConstants.isAcceptableDouble(errorPrevious)
                        && DataMineConstants.isAcceptableDouble(errorCurrent)
                        && (Math.abs(errorCurrent / errorPrevious) - 1f)
                        < ERROR_THRESHOLD) {
                    errorDifferenceSignificant = false;
                } else {
                    errorDifferenceSignificant = true;
                }
            }
        } while (errorDifferenceSignificant && !noReassignments &&
                getIterationIndex() < MAX_ITER);
        endCentroids = clusterHubs;
        setClusterAssociations(bestAssociations);
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
                    if (distances[first][second - first - 1] > 0) {
                        error += distances[first][second - first - 1]
                                * distances[first][second - first - 1];
                    } else {
                        error += Math.pow(cmet.dist(
                                clusterHubs[clusterAssociations[i]],
                                dset.data.get(i)), 2);
                    }
                }
            } else {
                error += Math.pow(cmet.dist(
                        clusterHubs[clusterAssociations[i]],
                        dset.data.get(i)), 2);
            }
        }
        System.out.println(error);
        return error;
    }

    /**
     * @return Probability of a deterministic iteration.
     */
    private float getProbFromSchedule() {
        int iteration = getIterationIndex();
        if (iteration < MIN_ITERATIONS) {
            return (float) ((float) iteration / (float) MIN_ITERATIONS);
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
}
