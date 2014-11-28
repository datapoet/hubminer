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
import algref.JournalPublication;
import algref.Publication;
import algref.Publisher;
import data.neighbors.NeighborSetFinder;
import data.representation.DataInstance;
import data.representation.DataSet;
import data.representation.util.DataMineConstants;
import distances.primary.CombinedMetric;
import java.util.Arrays;
import java.util.HashMap;
import learning.unsupervised.Cluster;
import learning.unsupervised.ClusteringAlg;
import learning.unsupervised.initialization.PlusPlusSeeder;

/**
 * Global K-hubs Clustering algorithm that was analyzed in the paper titled "The
 * Role of Hubness in Clustering High-dimensional Data", which was presented at
 * PAKDD in 2011. Hubness is calculated locally here, within the clusters.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class GKH extends ClusteringAlg implements
        learning.supervised.interfaces.DistMatrixUserInterface,
        data.neighbors.NSFUserInterface {

    private static final double ERROR_THRESHOLD = 0.001;
    private static final int MAX_ITER = 40;
    private float smallestError = Float.MAX_VALUE;
    private int[] bestAssociations = null;
    private float[][] distances = null;
    int[] hubnessArray = null;
    private int k = 10;
    NeighborSetFinder nsf;
    DataInstance[] endCentroids = null;
    
    @Override
    public HashMap<String, String> getParameterNamesAndDescriptions() {
        HashMap<String, String> paramMap = new HashMap<>();
        paramMap.put("k", "Neighborhood size.");
        return paramMap;
    }
    
    @Override
    public Publication getPublicationInfo() {
        JournalPublication pub = new JournalPublication();
        pub.setTitle("The Role of Hubness in Clustering High-Dimensional Data");
        pub.addAuthor(Author.NENAD_TOMASEV);
        pub.addAuthor(Author.MILOS_RADOVANOVIC);
        pub.addAuthor(Author.DUNJA_MLADENIC);
        pub.addAuthor(Author.MIRJANA_IVANOVIC);
        pub.setPublisher(Publisher.IEEE);
        pub.setJournalName("IEEE Transactions on Knowledge and Data "
                + "Engineering");
        pub.setYear(2014);
        pub.setStartPage(183);
        pub.setEndPage(195);
        pub.setVolume(6634);
        pub.setDoi("10.1109/TKDE.2013.25");
        pub.setUrl("http://ieeexplore.ieee.org/xpl/articleDetails.jsp?"
                + "arnumber=6427743");
        return pub;
    }

    /**
     * The default constructor.
     */
    public GKH() {
    }

    /**
     * @param dset DataSet object.
     * @param cmet CombinedMetric object for distance calculations.
     * @param numClusters A pre-defined number of clusters.
     * @param k Neighborhood size.
     */
    public GKH(DataSet dset, CombinedMetric cmet, int numClusters, int k) {
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
    public GKH(DataSet inputSet, int numClusters, int k) {
        setNumClusters(numClusters);
        setCombinedMetric(CombinedMetric.EUCLIDEAN);
        setDataSet(inputSet);
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
        PlusPlusSeeder seeder = new PlusPlusSeeder(numClusters,
                dset.data, cmet);
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
        int closestHub;
        float smallestDistance;
        float currentDistance;
        // It's best if the first assignment is done before and if the
        // assignments are done at the end of the do-while loop, therefore
        // allowing for better calculateIterationError estimates.
        for (int i = 0; i < dset.size(); i++) {
            closestHub = -1;
            smallestDistance = Float.MAX_VALUE;
            for (int cIndex = 0; cIndex < numClusters; cIndex++) {
                if (clusterHubIndexes[cIndex] > 0) {
                    if (clusterHubIndexes[cIndex] != i) {
                        fi = Math.min(i, clusterHubIndexes[cIndex]);
                        se = Math.max(i, clusterHubIndexes[cIndex]);
                        currentDistance = distances[fi][se - fi - 1];
                    } else {
                        closestHub = cIndex;
                        break;
                    }
                } else {
                    currentDistance = cmet.dist(dset.getInstance(i),
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
            int maxFrequency;
            int maxIndex;
            int maxActualIndex = 0;
            int currSize;
            for (int cIndex = 0; cIndex < clusters.length; cIndex++) {
                currSize = clusters[cIndex].size();
                if (currSize == 1) {
                    clusterHubs[cIndex] = clusters[cIndex].getInstance(0);
                    clusterHubIndexes[cIndex] = clusters[cIndex].indexes.get(0);
                    continue;
                }
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
                clusterHubs[cIndex] = clusters[cIndex].getInstance(maxIndex);
                clusterHubIndexes[cIndex] = maxActualIndex;
            }
            for (int i = 0; i < dset.size(); i++) {
                closestHub = -1;
                smallestDistance = Float.MAX_VALUE;
                for (int cIndex = 0; cIndex < numClusters; cIndex++) {
                    if (clusterHubIndexes[cIndex] > 0) {
                        if (clusterHubIndexes[cIndex] != i) {
                            first = Math.min(i, clusterHubIndexes[cIndex]);
                            second = Math.max(i, clusterHubIndexes[cIndex]);
                            currentDistance =
                                    distances[first][second - first - 1];
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
                if (closestHub != clusterAssociations[i]) {
                    noReassignments = false;
                }
                clusterAssociations[i] = closestHub;
            }
            errorPrevious = errorCurrent;
            errorCurrent = calculateIterationError(clusterHubs,
                    clusterHubIndexes);
            if (errorCurrent < smallestError) {
                bestAssociations = clusterAssociations;
            }
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
        } while (errorDifferenceSignificant && !noReassignments
                && getIterationIndex() < MAX_ITER);
        endCentroids = clusterHubs;
        setClusterAssociations(bestAssociations);
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
        System.out.println(error);
        return error;
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
            hubnessArray = nsf.getNeighborFrequencies();
        } else {
            // If the nsf is provided, the hubness scores only need to be
            // recalculated for a smaller k - in case a larger one is given
            // in the nsf.
            if (nsf.getCurrK() > k) {
                nsf = nsf.getSubNSF(k);
            }
            hubnessArray = nsf.getNeighborFrequencies();
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
    public int getNeighborhoodSize() {
        return k;
    }
}
