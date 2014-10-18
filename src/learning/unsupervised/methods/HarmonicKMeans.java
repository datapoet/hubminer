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
import java.util.Random;
import learning.unsupervised.ClusteringAlg;

/**
 * Clusterer that performs harmonic K-means. It is a partitional iterative
 * procedure, similar to basic K-means, where the performance function is not
 * the variance but the harmonic mean of distances to centroids. The algorithm
 * details can be found in the following paper: Zhang, Bin. "Generalized
 * k-harmonic means." Hewlett-Packard Laboratories Technical Report (2000).
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class HarmonicKMeans extends ClusteringAlg {

    // Final centroids after the clustering is done.
    private DataInstance[] endCentroids = null;
    private float p = 2; // The exponent for the generalized algorithm.
    // When the change in calculateIterationError falls below a threshold, we
    // declare convergence and end the clustering run.
    private static final double ERROR_THRESHOLD = 0.001;

    /**
     * Empty constructor.
     */
    public HarmonicKMeans() {
    }

    /**
     * @param dset DataSet object for clustering.
     * @param cmet CombinedMetric object for distance calculations.
     * @param numClusters A pre-defined number of clusters.
     */
    public HarmonicKMeans(DataSet dset, CombinedMetric cmet, int numClusters) {
        setDataSet(dset);
        setCombinedMetric(cmet);
        setNumClusters(numClusters);
    }

    /**
     * @param dset DataSet object for clustering.
     * @param numClusters A pre-defined number of clusters.
     */
    public HarmonicKMeans(DataSet dset, int numClusters) {
        setDataSet(dset);
        setCombinedMetric(CombinedMetric.EUCLIDEAN);
        setNumClusters(numClusters);
    }

    /**
     * @param dset DataSet object for clustering.
     * @param cmet CombinedMetric object for distance calculations.
     * @param numClusters A pre-defined number of clusters.
     * @param p Float value representing the generalized exponent.
     */
    public HarmonicKMeans(DataSet dset, CombinedMetric cmet, int numClusters,
            float p) {
        setDataSet(dset);
        setCombinedMetric(cmet);
        setNumClusters(numClusters);
        this.p = p;
    }

    /**
     * @param dset DataSet object for clustering.
     * @param numClusters A pre-defined number of clusters.
     * @param p Float value representing the generalized exponent.
     */
    public HarmonicKMeans(DataSet dset, int numClusters, float p) {
        setDataSet(dset);
        setCombinedMetric(CombinedMetric.EUCLIDEAN);
        setNumClusters(numClusters);
        this.p = p;
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
        DataInstance[] centroids = new DataInstance[numClusters];
        Random randa = new Random();
        int centroidIndex;
        for (int cIndex = 0; cIndex < centroids.length; cIndex++) {
            centroidIndex = randa.nextInt(dset.size());
            while (clusterAssociations[centroidIndex] != -1) {
                centroidIndex = randa.nextInt(dset.size());
            }
            clusterAssociations[centroidIndex] = cIndex;
            centroids[cIndex] = dset.getInstance(centroidIndex).copyContent();
        }
        float[][] pointToCentroidDists =
                new float[dset.size()][centroids.length];
        float[] sqInvCentDist = new float[dset.size()];
        // When there are no reassignments, we can end the clustering.
        boolean noReassignments;
        double errorPrevious;
        double errorCurrent = Double.MAX_VALUE;
        // This is initialized to true for the first iteration to go through.
        boolean errorDifferenceSignificant = true;
        setIterationIndex(0);
        do {
            nextIteration();
            noReassignments = true;
            for (int i = 0; i < dset.size(); i++) {
                float smallestDistance = Float.MAX_VALUE;
                float currentDistance;
                int closestCentroidIndex = -1;
                sqInvCentDist[i] = 0;
                for (int cIndex = 0; cIndex < centroids.length; cIndex++) {
                    currentDistance = cmet.dist(
                            dset.getInstance(i), centroids[cIndex]);
                    pointToCentroidDists[i][cIndex] = currentDistance;
                    if (currentDistance < smallestDistance) {
                        smallestDistance = currentDistance;
                        closestCentroidIndex = cIndex;
                    }
                    if (DataMineConstants.isAcceptableFloat(currentDistance)
                            && DataMineConstants.isNonZero(currentDistance)) {
                        sqInvCentDist[i] += 1 / Math.pow(currentDistance, p);
                    }
                }
                sqInvCentDist[i] = (float) Math.pow(sqInvCentDist[i], 2);
                if (closestCentroidIndex != clusterAssociations[i]) {
                    // The point has been assigned to a different cluster.
                    clusterAssociations[i] = closestCentroidIndex;
                    noReassignments = false;
                }
            }
            for (int cIndex = 0; cIndex < centroids.length; cIndex++) {
                centroids[cIndex] = new DataInstance(dset);
                float[] featureNumerator = new float[dset.getNumFloatAttr()];
                float denominator = 0;
                float factor;
                for (int i = 0; i < dset.size(); i++) {
                    factor = 1f / (sqInvCentDist[i]
                            * (float) Math.pow(
                            pointToCentroidDists[i][cIndex], p + 2));
                    if (DataMineConstants.isAcceptableFloat(factor)) {
                        denominator += factor;
                        for (int fIndex = 0; fIndex < dset.getNumFloatAttr();
                                fIndex++) {
                            featureNumerator[fIndex] +=
                                    factor * dset.getInstance(i).fAttr[fIndex];
                        }
                    }
                }
                for (int fIndex = 0; fIndex < dset.getNumFloatAttr();
                        fIndex++) {
                    if (!DataMineConstants.isAcceptableFloat(
                            featureNumerator[fIndex])
                            || !DataMineConstants.isAcceptableFloat(denominator)
                            || DataMineConstants.isZero(denominator)) {
                        continue;
                    }
                    centroids[cIndex].fAttr[fIndex] = featureNumerator[fIndex]
                            / denominator;
                }
            }
            errorPrevious = errorCurrent;
            errorCurrent = calculateIterationError(centroids);
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
        } while (errorDifferenceSignificant && !noReassignments);
        endCentroids = centroids;
        flagAsInactive();
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
     * Calculates the iteration calculateIterationError for convergence check.
     *
     * @param centroids An array of cluster centroid objects.
     * @return A harmonic mean of distances from points to centroids.
     * @throws Exception
     */
    private double calculateIterationError(DataInstance[] centroids)
            throws Exception {
        DataSet dset = getDataSet();
        CombinedMetric cmet = getCombinedMetric();
        int numClusters = getNumClusters();
        double iterationError = 0;
        float centroidDistance;
        double factor;
        for (int i = 0; i < dset.size(); i++) {
            double pointSum = 0;
            for (int cIndex = 0; cIndex < numClusters; cIndex++) {
                centroidDistance = cmet.dist(centroids[cIndex],
                        dset.getInstance(i));
                factor = Math.pow(centroidDistance, p);
                if (DataMineConstants.isAcceptableDouble(factor)
                        && DataMineConstants.isNonZero(factor)) {
                    pointSum += 1 / factor;
                }
            }
            pointSum = numClusters / pointSum;
            if (DataMineConstants.isAcceptableDouble(pointSum)) {
                iterationError += pointSum;
            }
        }
        return iterationError;
    }
}