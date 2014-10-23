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
import distances.kernel.Kernel;
import distances.kernel.KernelMatrixUserInterface;
import distances.kernel.MinKernel;
import distances.primary.CombinedMetric;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Random;
import learning.unsupervised.Cluster;
import learning.unsupervised.ClusteringAlg;

/**
 * This class implements the kernel K-means algorithm, a kernelized version of
 * the well-known partitional clustering approach.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class KernelKMeans extends ClusteringAlg implements
        KernelMatrixUserInterface {

    // Kernel matrix - it HAS diagonal entries, as k(x,x) can possibly vary -
    // not in those kernels based on (x-y), but in those based on xy. Therefore,
    // it is represented in such a way that every row i has entries for cIndex
    // >= i.
    private float[][] kmat;
    private double[] clusterKerFactors;
    private Kernel ker;
    private double error; // Calculated incrementally.
    private static final float ERROR_THRESHOLD = (float) 0.003;
    private float[] instanceWeights;
    private DataInstance[] endCentroids = null;
    
    @Override
    public HashMap<String, String> getParameterNamesAndDescriptions() {
        HashMap<String, String> paramMap = new HashMap<>();
        paramMap.put("ker", "Kernel.");
        return paramMap;
    }

    public KernelKMeans() {
    }

    /**
     * @param dset DataSet object for clustering.
     * @param ker Kernel object.
     * @param numClusters A pre-defined number of clusters.
     */
    public KernelKMeans(DataSet dset, Kernel ker, int numClusters) {
        setNumClusters(numClusters);
        this.ker = ker;
        setDataSet(dset);
    }

    /**
     * @param dset DataSet object.
     * @param ker Kernel object.
     * @param numClusters A pre-defined number of clusters.
     * @param instanceWeights A float array of instance weights.
     */
    public KernelKMeans(DataSet dset, Kernel ker, int numClusters,
            float[] instanceWeights) {
        setNumClusters(numClusters);
        this.ker = ker;
        setDataSet(dset);
        this.instanceWeights = instanceWeights;
    }

    /**
     * @param dset
     * @param numClusters
     */
    public KernelKMeans(DataSet dset, int numClusters) {
        setDataSet(dset);
        setNumClusters(numClusters);
        setCombinedMetric(CombinedMetric.EUCLIDEAN);
        ker = new MinKernel();
    }

    @Override
    public void cluster() throws Exception {
        performBasicChecks();
        flagAsActive();
        DataSet dset = getDataSet();
        int numClusters = getNumClusters();
        boolean trivial = checkIfTrivial();
        if (trivial) {
            return;
        } // Nothing needs to be done in this case.
        if (ker == null) {
            throw new Exception("No kernel was provided to the clustering"
                    + " algorithm. Unable to cluster.");
        }
        int[] clusterAssociations = new int[dset.data.size()];
        Arrays.fill(clusterAssociations, 0, dset.size(), -1);
        setClusterAssociations(clusterAssociations);
        int size = clusterAssociations.length;
        if (instanceWeights == null) {
            instanceWeights = new float[size];
            Arrays.fill(instanceWeights, 1);
        }
        if (kmat == null) {
            kmat = dset.calculateKernelMatrixMultThr(ker, 4);
        }
        DataInstance[] centroids = new DataInstance[numClusters];
        Random randa = new Random();
        int centroidIndex;
        int[] initialIndexes = new int[numClusters];
        clusterKerFactors = new double[centroids.length];
        for (int cIndex = 0; cIndex < numClusters; cIndex++) {
            centroidIndex = randa.nextInt(dset.size());
            while (clusterAssociations[centroidIndex] != -1) {
                centroidIndex = randa.nextInt(dset.size());
            }
            clusterAssociations[centroidIndex] = cIndex;
            initialIndexes[cIndex] = centroidIndex;
            centroids[cIndex] = dset.getInstance(centroidIndex).copyContent();
            clusterKerFactors[cIndex] = kmat[centroidIndex][0];
        }
        Cluster[] clusters = new Cluster[numClusters];
        // When there are no reassignments, we can end the clustering.
        boolean noReassignments;
        double errorPrevious;
        double errorCurrent = Double.MAX_VALUE;
        // This is initialized to true for the first iteration to go through.
        boolean errorDifferenceSignificant = true;
        setIterationIndex(0);
        do {
            nextIteration();
            error = 0; // Incrementally calculated.
            // Assigning points to clusters.
            noReassignments = true;
            for (int i = 0; i < clusterAssociations.length; i++) {
                int closestCentroidIndex = -1;
                double smallestDistance = Double.MAX_VALUE;
                double[] clusterDistances = new double[centroids.length];
                double currentDistance;
                if (getIterationIndex() > 1) {
                    for (int cIndex = 0; cIndex < clusters.length; cIndex++) {
                        clusterDistances[cIndex] = kmat[i][0]
                                + clusterKerFactors[cIndex];
                    }
                    for (int index = 0; index < clusterAssociations.length;
                            index++) {
                        clusterDistances[clusterAssociations[index]] -=
                                2 * instanceWeights[index] * kmat[
                                        Math.min(i, index)][Math.max(i, index)
                                - Math.min(i, index)]
                                * (1f / clusters[clusterAssociations[
                                        index]].size());
                        // Because diagonal entries are included in the kernel 
                        // matrix.
                    }
                } else {
                    int min, max;
                    for (int cIndex = 0; cIndex < clusters.length; cIndex++) {
                        min = Math.min(initialIndexes[cIndex], i);
                        max = Math.max(initialIndexes[cIndex], i);
                        clusterDistances[cIndex] = -kmat[min][max - min];
                    }
                }
                for (int cIndex = 0; cIndex < clusters.length; cIndex++) {
                    currentDistance = clusterDistances[cIndex];
                    if (currentDistance < smallestDistance) {
                        smallestDistance = currentDistance;
                        closestCentroidIndex = cIndex;
                    } else if (currentDistance == smallestDistance) {
                        if (randa.nextFloat() > 0.5) {
                            smallestDistance = currentDistance;
                            closestCentroidIndex = cIndex;
                        }
                    }
                }
                error += smallestDistance;
                if (closestCentroidIndex != clusterAssociations[i]) {
                    noReassignments = false;
                }
                clusterAssociations[i] = closestCentroidIndex;
            }
            clusters = getClusters();
            // Now recalculate the clustering kernel factors.
            int min, max;
            for (int cIndex = 0; cIndex < clusters.length; cIndex++) {
                clusterKerFactors[cIndex] = 0;
                centroids[cIndex] = clusters[cIndex].getCentroid();
                for (int i = 0; i < clusters[cIndex].size(); i++) {
                    for (int j = i; j < clusters[cIndex].size(); j++) {
                        // Including the self-kernel distance.
                        min = Math.min(
                                clusters[cIndex].indexes.get(j),
                                clusters[cIndex].indexes.get(i));
                        max = Math.max(
                                clusters[cIndex].indexes.get(j),
                                clusters[cIndex].indexes.get(i));
                        clusterKerFactors[cIndex] +=
                                instanceWeights[clusters[cIndex].indexes.get(i)]
                                * instanceWeights[clusters[cIndex].
                                indexes.get(j)] * kmat[min][max - min]
                                * (1f / (clusters[cIndex].size()
                                * Math.max(1, (clusters[cIndex].size()))));
                    }
                }
            }
            errorPrevious = errorCurrent;
            errorCurrent = error;
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
        setClusterAssociations(clusterAssociations);
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

    @Override
    public int[] assignPointsToModelClusters(
            DataSet dsetTest,
            NeighborSetFinder nsfTest,
            float[][] kernelSimToTraining,
            float[] selfKernels) {
        if (dsetTest == null || dsetTest.isEmpty()) {
            return null;
        } else {
            int[] clusterAssociations = new int[dsetTest.size()];
            double minDist;
            double dist;
            int[] modelAssociations = getClusterAssociations();
            for (int i = 0; i < dsetTest.size(); i++) {
                Cluster[] clust = this.getClusters();
                double[] clustDists = new double[clust.length];
                for (int cIndex = 0; cIndex < clustDists.length; cIndex++) {
                    clustDists[cIndex] = clusterKerFactors[cIndex];
                }
                for (int index = 0; index < modelAssociations.length; index++) {
                    if (modelAssociations[index] >= 0
                            && modelAssociations[index] < clustDists.length) {
                        clustDists[modelAssociations[index]] -=
                                2 * instanceWeights[index]
                                * kernelSimToTraining[i][index];
                    }
                }
                minDist = Double.MAX_VALUE;
                for (int c = 0; c < clustDists.length; c++) {
                    dist = clustDists[c];
                    if (dist < minDist) {
                        clusterAssociations[i] = c;
                        minDist = dist;
                    }
                }
            }
            return clusterAssociations;
        }
    }

    /**
     * @param weights A float array of instance weights.
     */
    public void setInstanceWeights(float[] weights) {
        instanceWeights = weights;
    }

    @Override
    public void setKernelMatrix(float[][] kmat) {
        this.kmat = kmat;
    }

    @Override
    public float[][] getKernelMatrix() {
        return kmat;
    }
}
