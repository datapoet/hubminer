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
package learning.unsupervised.methods.multithreaded;

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
 * and Vineet Singh. "An Efficient K-Means Clustering Algorithm." This is a
 * slightly modified version of the original, that multi-threads when pruning
 * centroid lists.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class MTFastKMeans extends ClusteringAlg {

    private static final double ERROR_THRESHOLD = 0.001;
    private Cluster[] clusters = null;
    private float[] clusterSquareSums = null;
    private int[] clusterNumberOfElements = null;
    private float[][] clusterLinearIntSums = null;
    private float[][] clusterLinearFloatSums = null;
    private DataInstance[] endCentroids = null;
    private int numThreads = 8;
    private volatile int threadCount = 1;
    private boolean printOutIteration = false;
    private Object[] elLock;
    private Object[] sqLock;
    private Object[] linintLock;
    private Object[] linfloatLock;
    private Object[] instanceAddLock;
    
    @Override
    public HashMap<String, String> getParameterNamesAndDescriptions() {
        HashMap<String, String> paramMap = new HashMap<>();
        return paramMap;
    }

    public MTFastKMeans() {
    }

    /**
     * Increases the count of elements in the cluster.
     *
     * @param cIndex Integer that is the cluster index.
     * @param size Integer that is the additional size.
     */
    private void increaseNumEl(int cIndex, int size) {
        synchronized (elLock[cIndex]) {
            clusterNumberOfElements[cIndex] += size;
        }
    }

    /**
     * Increases the square sums in the cluster.
     *
     * @param cIndex Integer that is the cluster index.
     * @param increment Float value that is the increment.
     */
    private void increaseSquareSums(int cIndex, float increment) {
        synchronized (sqLock[cIndex]) {
            clusterSquareSums[cIndex] += increment;
        }
    }

    /**
     * Increases the linear float sums in the cluster.
     *
     * @param cIndex Integer that is the cluster index.
     * @param attIndex Integer that is the index of the target feature.
     * @param increment Float value that is the increment.
     */
    private void increaseLinFloatSums(int cIndex, int attIndex,
            float increment) {
        synchronized (linfloatLock[cIndex]) {
            clusterLinearFloatSums[cIndex][attIndex] += increment;
        }
    }

    /**
     * Increases the linear integer sums in the cluster.
     *
     * @param cIndex Integer that is the cluster index.
     * @param attIndex Integer that is the index of the target feature.
     * @param increment Float value that is the increment.
     */
    private void increaseLinIntSums(int cIndex, int attIndex,
            float increment) {
        synchronized (linintLock[cIndex]) {
            clusterLinearIntSums[cIndex][attIndex] += increment;
        }
    }

    /**
     * Adds an instance to a cluster.
     *
     * @param cIndex Integer that is the cluster index.
     * @param index Integer that is the index of the instance to insert.
     */
    private void addInstanceToCluster(int cIndex, int index) {
        synchronized (instanceAddLock[cIndex]) {
            clusters[cIndex].addInstance(index);
        }
    }

    /**
     * @param dset DataSet object.
     * @param cmet CombinedMetric object for distance calculations.
     * @param numClusters A pre-defined number of clusters.
     */
    public MTFastKMeans(DataSet dset, CombinedMetric cmet,
            int numClusters) {
        setNumClusters(numClusters);
        setCombinedMetric(cmet);
        setDataSet(dset);
    }

    /**
     * @param dset DataSet object.
     * @param cmet CombinedMetric object for distance calculations.
     * @param numClusters A pre-defined number of clusters.
     * @param numThreads Integer that is the number of threads to use.
     */
    public MTFastKMeans(DataSet dset, CombinedMetric cmet,
            int numClusters, int numThreads) {
        setNumClusters(numClusters);
        setCombinedMetric(cmet);
        setDataSet(dset);
        this.numThreads = numThreads;
    }

    /**
     * @param dset DataSet object.
     * @param cmet CombinedMetric object for distance calculations.
     * @param numClusters A pre-defined number of clusters.
     * @param numThreads Integer that is the number of threads to use.
     * @param printOutIteration Boolean flag indicating whether to print out an
     * indicator of each completed iteration to the output stream, which can be
     * used for tracking very long clustering runs.
     */
    public MTFastKMeans(DataSet dset, CombinedMetric cmet,
            int numClusters, int numThreads, boolean printOutIteration) {
        setNumClusters(numClusters);
        setCombinedMetric(cmet);
        setDataSet(dset);
        this.numThreads = numThreads;
        this.printOutIteration = printOutIteration;
    }

    /**
     * @param dset DataSet object.
     * @param numClusters A pre-defined number of clusters.
     */
    public MTFastKMeans(DataSet dset, int numClusters) {
        setNumClusters(numClusters);
        setCombinedMetric(CombinedMetric.EUCLIDEAN);
        setDataSet(dset);
    }

    /**
     * Increases the current thread count.
     */
    private synchronized void increaseThreadCount() {
        threadCount++;
    }

    /**
     * Decreases the current thread count.
     */
    private synchronized void decreaseThreadCount() {
        threadCount--;
    }

    @Override
    public void cluster() throws Exception {
        performBasicChecks();
        flagAsActive();
        DataSet dset = getDataSet();
        int numClusters = getNumClusters();
        elLock = new Object[numClusters];
        sqLock = new Object[numClusters];
        linintLock = new Object[numClusters];
        linfloatLock = new Object[numClusters];
        instanceAddLock = new Object[numClusters];
        for (int i = 0; i < numClusters; i++) {
            elLock[i] = new Object();
            sqLock[i] = new Object();
            linintLock[i] = new Object();
            linfloatLock[i] = new Object();
            instanceAddLock[i] = new Object();
        }
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
                    if (printOutIteration) {
                        System.out.print("|");
                    }
                    clusters = new Cluster[numClusters];
                    for (int cIndex = 0; cIndex < numClusters; cIndex++) {
                        clusters[cIndex] = new Cluster(dset,
                                (int) Math.max(dset.data.size()
                                / numClusters, 2));
                    }
                    clusterSquareSums = new float[numClusters];
                    clusterNumberOfElements = new int[numClusters];
                    clusterLinearIntSums = new float[numClusters][
                            dset.getNumIntAttr()];
                    clusterLinearFloatSums = new float[numClusters][
                            dset.getNumFloatAttr()];
                    centroidCandidateList = new ArrayList<>(numClusters);
                    for (DataInstance centroid : centroids) {
                        centroidCandidateList.add(centroid);
                    }
                    Thread t = new Thread(new NodeWorker(dataTree.getRoot(),
                            centroidCandidateList));
                    t.start();
                    try {
                        t.join();
                    } catch (Throwable thr) {
                    }
                    for (int cIndex = 0; cIndex < numClusters; cIndex++) {
                        if (clusterNumberOfElements[cIndex] > 0) {
                            for (int j = 0; j
                                    < clusterLinearIntSums[cIndex].length;
                                    j++) {
                                centroids[cIndex].iAttr[j] = (int)
                                        (clusterLinearIntSums[cIndex][j]
                                        / clusterNumberOfElements[cIndex]);
                            }
                            for (int j = 0; j
                                    < clusterLinearFloatSums[cIndex].length;
                                    j++) {
                                centroids[cIndex].fAttr[j] =
                                        (clusterLinearFloatSums[cIndex][j]
                                        / clusterNumberOfElements[cIndex]);
                            }
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
                } while (errorDifferenceSignificant);
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
                if (numAttempts > ClusteringAlg.MAX_RETRIES) {
                    throw new ClusteringError(ClusteringError.UNABLE_TO_FINISH);
                }
            }
        } while (!valid);
        endCentroids = centroids;
        flagAsInactive();
    }

    public DataInstance[] getCentroids() {
        return endCentroids;
    }

    /**
     * This is a worker class for KD tree traversal.
     */
    class NodeWorker implements Runnable {

        KDDataNode currentNode;
        ArrayList<DataInstance> centroidCandidateList;

        public NodeWorker(KDDataNode currentNode,
                ArrayList<DataInstance> centroidCandidateList) {
            this.currentNode = currentNode;
            this.centroidCandidateList = centroidCandidateList;
        }

        @Override
        public void run() {
            increaseThreadCount();
            try {
                kdTreeTraverse(currentNode, centroidCandidateList);
            } catch (Exception e) {
                System.out.println(e.getMessage());
            }
            decreaseThreadCount();
        }
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
                increaseNumEl(winner, currentNode.size());
                increaseSquareSums(winner, currentNode.squareSum);
                for (int i = 0; i < dset.getNumIntAttr(); i++) {
                    increaseLinIntSums(winner, i, currentNode.linearISum[i]);
                }
                for (int i = 0; i < dset.getNumFloatAttr(); i++) {
                    increaseLinFloatSums(winner, i, currentNode.linearFSum[i]);
                }
                for (int index : nodeDataIndexes) {
                    addInstanceToCluster(winner, index);
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
                        addInstanceToCluster(winner, index);
                        clusterAssociations[index] = winner;
                        increaseNumEl(winner, 1);
                        for (int i = 0; i < dset.getNumIntAttr(); i++) {
                            if (!DataMineConstants.isAcceptableInt(
                                    dset.getInstance(index).iAttr[i])) {
                                continue;
                            }
                            increaseSquareSums(winner, (float) Math.pow(
                                    dset.getInstance(index).iAttr[i], 2));
                            increaseLinIntSums(winner, i,
                                    dset.getInstance(index).iAttr[i]);
                        }
                        for (int i = 0; i < dset.getNumFloatAttr(); i++) {
                            if (!DataMineConstants.isAcceptableFloat(
                                    dset.getInstance(index).fAttr[i])) {
                                continue;
                            }
                            increaseSquareSums(winner, (float) Math.pow(
                                    dset.getInstance(index).fAttr[i], 2));
                            increaseLinFloatSums(winner, i,
                                    dset.getInstance(index).fAttr[i]);
                        }
                    }
                } else {
                    if (this.threadCount < numThreads) {
                        Thread tLeft = new Thread(new NodeWorker(
                                currentNode.left,
                                potentialCentroidsPruned));
                        Thread tRight = new Thread(new NodeWorker(
                                currentNode.right,
                                potentialCentroidsPruned));
                        tLeft.start();
                        tRight.start();
                        try {
                            tLeft.join();
                            tRight.join();
                        } catch (Throwable thr) {
                        }
                    } else {
                        kdTreeTraverse(currentNode.left,
                                potentialCentroidsPruned);
                        kdTreeTraverse(currentNode.right,
                                potentialCentroidsPruned);
                    }
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
