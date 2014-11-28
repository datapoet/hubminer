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

import algref.Author;
import algref.ConferencePublication;
import algref.Publication;
import algref.Publisher;
import data.neighbors.NeighborSetFinder;
import data.representation.DataInstance;
import data.representation.DataSet;
import data.representation.util.DataMineConstants;
import data.structures.KDDataNode;
import data.structures.KDTree;
import distances.primary.CombinedMetric;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Random;
import learning.unsupervised.Cluster;
import learning.unsupervised.ClusteringAlg;
import learning.unsupervised.ClusteringError;

/**
 * A fast K-means implementation described in: Alsabti, Khaled, Sanjay Ranka,
 * and Vineet Singh. "An Efficient K-Means Clustering Algorithm."
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class FastKMeans extends ClusteringAlg {

    private static final double ERROR_THRESHOLD = 0.001;
    private static final int DEFAULT_MAX_ITERATIONS = 250;
    private Cluster[] clusters = null;
    private float[] clusterSquareSums = null;
    private int[] clusterNumberOfElements = null;
    private float[][] clusterLinearIntSums = null;
    private float[][] clusterLinearFloatSums = null;
    private DataInstance[] endCentroids = null;
    private int maxIterations = DEFAULT_MAX_ITERATIONS;
    
    @Override
    public HashMap<String, String> getParameterNamesAndDescriptions() {
        HashMap<String, String> paramMap = new HashMap<>();
        paramMap.put("maxIterations", "Maximum number of iterations to run.");
        return paramMap;
    }
    
    @Override
    public Publication getPublicationInfo() {
        ConferencePublication pub = new ConferencePublication();
        pub.setConferenceName("IPPS/SPDP Workshop on High Performance Data "
                + "Mining");
        pub.addAuthor(new Author("Khaled", "Alsabti"));
        pub.addAuthor(new Author("Sanjay", "Ranka"));
        pub.addAuthor(new Author("Vineet", "Singh"));
        pub.setTitle("An Efficient K-Means Clustering Algorithm");
        pub.setYear(1998);
        return pub;
    }

    public FastKMeans() {
    }

    /**
     * @param dset DataSet object.
     * @param cmet CombinedMetric object for distance calculations.
     * @param numClusters A pre-defined number of clusters.
     */
    public FastKMeans(DataSet dset, CombinedMetric cmet, int numClusters) {
        setNumClusters(numClusters);
        setCombinedMetric(cmet);
        setDataSet(dset);
    }

    /**
     * @param dset DataSet object.
     * @param numClusters A pre-defined number of clusters.
     */
    public FastKMeans(DataSet dset, int numClusters) {
        setNumClusters(numClusters);
        setCombinedMetric(CombinedMetric.EUCLIDEAN);
        setDataSet(dset);
    }

    @Override
    public void cluster() throws Exception {
        performBasicChecks();
        flagAsActive();
        DataSet dset = getDataSet();
        int numClusters = getNumClusters();
        this.setMinIterations(5);
        boolean trivial = checkIfTrivial();
        if (trivial) {
            return;
        } // Nothing needs to be done in this case.
        int[] clusterAssociations = new int[dset.size()];
        Arrays.fill(clusterAssociations, 0, dset.size(), -1);
        setClusterAssociations(clusterAssociations);
        DataInstance[] centroids = new DataInstance[numClusters];
        // This list is used for pruning purposes.
        ArrayList<DataInstance> centroidCandidateList;
        String[] clustAttribute = new String[1];
        clustAttribute[0] = "Cluster index";
        DataSet clusterIDDSet = new DataSet(clustAttribute, null, null,
                numClusters);
        clusterIDDSet.iAttrNames = new String[1];
        clusterIDDSet.iAttrNames[0] = "clust_id";
        Random randa = new Random();
        int centroidIndex;
        int numAttempts = 0;
        boolean valid;
        do {
            numAttempts++;
            valid = true;
            try {
                for (int cIndex = 0; cIndex < centroids.length; cIndex++) {
                    centroidIndex = randa.nextInt(dset.size());
                    while (clusterAssociations[centroidIndex] != -1) {
                        centroidIndex = randa.nextInt(dset.size());
                    }
                    DataInstance ithID = new DataInstance(clusterIDDSet);
                    ithID.iAttr[0] = cIndex;
                    clusterAssociations[centroidIndex] = cIndex;
                    centroids[cIndex] =
                            dset.getInstance(centroidIndex).copyContent();
                    centroids[cIndex].setIdentifier(ithID);
                }
                KDTree dataTree = new KDTree();
                dataTree.createDataTree(dset);
                double errorPrevious;
                double errorCurrent = Double.MAX_VALUE;
                // This is initialized to true for the first iteration to go
                // through.
                boolean errorDifferenceSignificant = true;
                setIterationIndex(0);
                do {
                    nextIteration();
                    clusters = new Cluster[numClusters];
                    for (int cIndex = 0; cIndex < numClusters; cIndex++) {
                        clusters[cIndex] = new Cluster(dset,
                                (int) Math.max(dset.data.size()
                                / numClusters, 2));
                    }
                    clusterSquareSums = new float[numClusters];
                    clusterNumberOfElements = new int[numClusters];
                    clusterLinearIntSums =
                            new float[numClusters][dset.getNumIntAttr()];
                    clusterLinearFloatSums =
                            new float[numClusters][dset.getNumFloatAttr()];
                    centroidCandidateList = new ArrayList<>(numClusters);
                    for (DataInstance centroid : centroids) {
                        centroidCandidateList.add(centroid);
                    }
                    kdTreeTraverse(dataTree.getRoot(), centroidCandidateList);
                    for (int cIndex = 0; cIndex < numClusters; cIndex++) {
                        for (int j = 0; j
                                < clusterLinearIntSums[cIndex].length; j++) {
                            centroids[cIndex].iAttr[j] = (int) (
                                    clusterLinearIntSums[cIndex][j]
                                    / clusterNumberOfElements[cIndex]);
                        }
                        for (int j = 0; j
                                < clusterLinearFloatSums[cIndex].length; j++) {
                            centroids[cIndex].fAttr[j] = (
                                    clusterLinearFloatSums[cIndex][j]
                                    / clusterNumberOfElements[cIndex]);
                        }
                    }
                    errorPrevious = errorCurrent;
                    errorCurrent = calculateIterationError(
                            centroids,
                            clusterSquareSums,
                            clusterNumberOfElements,
                            clusterLinearIntSums,
                            clusterLinearFloatSums);
                    if (getIterationIndex() >= MIN_ITERATIONS) {
                        if (DataMineConstants.isAcceptableDouble(
                                errorPrevious)
                                && DataMineConstants.isAcceptableDouble(
                                errorCurrent)
                                && (Math.abs(errorCurrent / errorPrevious) - 1f)
                                < ERROR_THRESHOLD) {
                            errorDifferenceSignificant = false;
                        } else {
                            errorDifferenceSignificant = true;
                        }
                    }
                } while (errorDifferenceSignificant
                        && getIterationIndex() < maxIterations);
            } catch (ClusteringError ce) {
                if (ce.getErrorCause() == ClusteringError.EMPTY_CLUSTER) {
                    System.out.println("Empty cluster generated, reclustering");
                } else if (ce.getErrorCause()
                        == ClusteringError.UNKNOWN_PROBLEM) {
                    System.out.println("Unknown error, reclustering");
                }
                valid = false;
                if (numAttempts > ClusteringAlg.MAX_RETRIES) {
                    throw new ClusteringError(ClusteringError.UNABLE_TO_FINISH);
                }
            } catch (Exception e) {
                System.out.println(e.getMessage());
                if (numAttempts > ClusteringAlg.MAX_RETRIES) {
                    throw new ClusteringError(ClusteringError.UNABLE_TO_FINISH);
                }
            }
        } while (!valid);
        endCentroids = centroids;
        flagAsInactive();
    }

    /**
     * Traverses the KD tree and prunes the centroid candidate lists.
     *
     * @param currentNode KDDataNode that is the current node in the tree.
     * @param centroidCandidateList ArrayList of DataInstance objects that are
     * the current centroid candidates.
     * @throws Exception
     */
    private void kdTreeTraverse(KDDataNode currentNode,
            ArrayList<DataInstance> centroidCandidateList) throws Exception {
        DataSet dset = getDataSet();
        CombinedMetric cmet = getCombinedMetric();
        int[] clusterAssociations = getClusterAssociations();
        if (currentNode != null) {
            ArrayList<DataInstance> potentialCentroidsPruned =
                    currentNode.prune(centroidCandidateList, cmet);
            if (potentialCentroidsPruned == null) {
                throw new Exception("Empty cluster list caused an error.");
            }
            // First check if there is one candidate or more.
            if (potentialCentroidsPruned.size() == 1) {
                ArrayList<Integer> nodeDataIndexes =
                        currentNode.instanceIndexes;
                int winner = potentialCentroidsPruned.get(0).
                        getIdentifier().iAttr[0];
                clusterNumberOfElements[winner] += currentNode.size();
                clusterSquareSums[winner] += currentNode.squareSum;
                for (int i = 0; i < dset.getNumIntAttr(); i++) {
                    clusterLinearIntSums[winner][i] +=
                            currentNode.linearISum[i];
                }
                for (int i = 0; i < dset.getNumFloatAttr(); i++) {
                    clusterLinearFloatSums[winner][i] +=
                            currentNode.linearFSum[i];
                }
                for (int index : nodeDataIndexes) {
                    clusters[winner].addInstance(index);
                    clusterAssociations[index] = winner;
                }
            } else {
                if (currentNode.isLeaf()) {
                    ArrayList<Integer> nodeDataIndexes =
                            currentNode.instanceIndexes;
                    DataInstance closestCentroid;
                    float minDist;
                    float currDist;
                    int winner;
                    for (int index : nodeDataIndexes) {
                        minDist = Float.MAX_VALUE;
                        closestCentroid = potentialCentroidsPruned.get(0);
                        for (DataInstance centroid : potentialCentroidsPruned) {
                            currDist = cmet.dist(dset.getInstance(index),
                                    centroid);
                            if (currDist < minDist) {
                                minDist = currDist;
                                closestCentroid = centroid;
                            }
                        }
                        winner = closestCentroid.getIdentifier().iAttr[0];
                        clusters[winner].addInstance(index);
                        clusterAssociations[index] = winner;
                        clusterNumberOfElements[winner]++;
                        for (int i = 0; i < dset.getNumIntAttr(); i++) {
                            if (!DataMineConstants.isAcceptableInt(
                                    dset.getInstance(index).iAttr[i])) {
                                continue;
                            }
                            clusterSquareSums[winner] +=
                                    Math.pow(
                                    dset.getInstance(index).iAttr[i], 2);
                            clusterLinearIntSums[winner][i] +=
                                    dset.getInstance(index).iAttr[i];
                        }
                        for (int i = 0; i < dset.getNumFloatAttr(); i++) {
                            if (!DataMineConstants.isAcceptableFloat(
                                    dset.getInstance(index).fAttr[i])) {
                                continue;
                            }
                            clusterSquareSums[winner] +=
                                    Math.pow(
                                    dset.getInstance(index).fAttr[i], 2);
                            clusterLinearFloatSums[winner][i] +=
                                    dset.getInstance(index).fAttr[i];
                        }
                    }
                } else {
                    kdTreeTraverse(currentNode.left,
                            potentialCentroidsPruned);
                    kdTreeTraverse(currentNode.right,
                            potentialCentroidsPruned);
                }
            }
            setClusterAssociations(clusterAssociations);
        }
    }

    /**
     * @param centroids Centroids.
     * @param clusterSquareSums Square feature value sums per cluster.
     * @param clusterNumberOfElements Number of elements per cluster.
     * @param clusterLinearIntSums Linear integer feature value sums per
     * cluster.
     * @param clusterLinearFloatSums Linear float feature value sums per
     * cluster.
     * @return Iteration error.
     * @throws Exception
     */
    private double calculateIterationError(
            DataInstance[] centroids,
            float[] clusterSquareSums,
            int[] clusterNumberOfElements,
            float[][] clusterLinearIntSums,
            float[][] clusterLinearFloatSums) throws Exception {
        double errorTotal = 0;
        for (int cIndex = 0; cIndex < centroids.length; cIndex++) {
            if (centroids[cIndex] != null) {
                errorTotal += clusterSquareSums[cIndex];
                if (clusterLinearIntSums != null) {
                    double fact = 0;
                    for (int j = 0; j < clusterLinearIntSums[cIndex].length;
                            j++) {
                        fact += Math.pow(centroids[cIndex].iAttr[j], 2);
                        errorTotal -= (2 * centroids[cIndex].iAttr[j]
                                * clusterLinearIntSums[cIndex][j]);
                    }
                    errorTotal += (fact * clusterNumberOfElements[cIndex]);
                }
                if (clusterLinearFloatSums != null) {
                    double fact = 0;
                    for (int j = 0; j < clusterLinearFloatSums[cIndex].length;
                            j++) {
                        fact += Math.pow(centroids[cIndex].fAttr[j], 2);
                        errorTotal -= (2 * centroids[cIndex].fAttr[j]
                                * clusterLinearFloatSums[cIndex][j]);
                    }
                    errorTotal += (fact * clusterNumberOfElements[cIndex]);
                }
            }
        }
        errorTotal = Math.abs(errorTotal);
        if (!DataMineConstants.isAcceptableDouble(errorTotal)) {
            throw new ClusteringError(ClusteringError.UNKNOWN_PROBLEM);
        }
        return errorTotal;
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
    public Cluster[] getClusters() {
        return clusters;
    }
}