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
package learning.unsupervised.tests;

import algref.Author;
import algref.JournalPublication;
import algref.Publication;
import algref.Publisher;
import data.generators.util.MultiGaussianMixForClusteringTesting;
import data.neighbors.NeighborSetFinder;
import data.representation.DataInstance;
import data.representation.DataSet;
import data.representation.util.DataMineConstants;
import distances.primary.CombinedMetric;
import ioformat.FileUtil;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import learning.unsupervised.Cluster;
import learning.unsupervised.ClusteringAlg;
import static learning.unsupervised.ClusteringAlg.MIN_ITERATIONS;
import learning.unsupervised.initialization.PlusPlusSeeder;

/**
 * Tracks the localization of hubs and medoids in K-means iterations.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class KMeansHubnessTester extends ClusteringAlg {

    private static final int MAX_ITER = 40;
    private float smallestError = Float.MAX_VALUE;
    private int[] bestAssociations = null;
    int[] hubnessArray = null;
    PrintWriter hcdWriter = null;
    // Final centroids after the clustering is done.
    private DataInstance[] endCentroids = null;
    // When the change in calculateIterationError falls below a threshold, we
    // declare convergence and end the clustering run.
    private static final double ERROR_THRESHOLD = 0.001;
    ArrayList<Float> hcMinVect = new ArrayList<>(300);
    ArrayList<Float> hcMaxVect = new ArrayList<>(300);
    ArrayList<Float> hcAvgVect = new ArrayList<>(300);
    ArrayList<Float> mAvgVect = new ArrayList<>(300);
    ArrayList<Float> mMinVect = new ArrayList<>(300);
    ArrayList<Float> mMaxVect = new ArrayList<>(300);
    
    @Override
    public HashMap<String, String> getParameterNamesAndDescriptions() {
        HashMap<String, String> paramMap = new HashMap<>();
        return paramMap;
    }

    @Override
    public Publication getPublicationInfo() {
        // The publication info is given for the paper that originally used 
        // hubness tracking in K-means iterations. For a reference on K-means 
        // itself, look up the base KMeans class.
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
     * @param hubnessArray An integer array of neighbor occurrence frequencies.
     */
    public void setHubness(int[] hubnessArray) {
        this.hubnessArray = hubnessArray;
    }

    /**
     * Calculate the neighbor occurrence frequencies of data points.
     *
     * @param k Neighborhood size to use for kNN sets.
     * @param cmet CombinedMetric object.
     * @throws Exception
     */
    public void calculateHubness(int k, CombinedMetric cmet) throws Exception {
        if (cmet == null) {
            cmet = CombinedMetric.EUCLIDEAN;
        }
        NeighborSetFinder nsf = new NeighborSetFinder(getDataSet(), cmet);
        nsf.calculateDistances();
        nsf.calculateNeighborSets(k);
        hubnessArray = nsf.getNeighborFrequencies();
    }

    /**
     * The default constructor.
     */
    public KMeansHubnessTester() {
    }

    /**
     * @param dset DataSet object for clustering.
     * @param cmet CombinedMetric object for distance calculations.
     * @param numClusters A pre-defined number of clusters.
     */
    public KMeansHubnessTester(
            DataSet dset,
            CombinedMetric cmet,
            int numClusters) {
        setDataSet(dset);
        setCombinedMetric(cmet);
        setNumClusters(numClusters);
    }

    /**
     * @param dset DataSet object for clustering.
     * @param numClusters A pre-defined number of clusters.
     */
    public KMeansHubnessTester(DataSet dset, int numClusters) {
        setDataSet(dset);
        setCombinedMetric(CombinedMetric.EUCLIDEAN);
        setNumClusters(numClusters);
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
        Cluster[] clusters;
        PlusPlusSeeder seeder =
                new PlusPlusSeeder(centroids.length, dset.data, cmet);
        int[] centroidIndexes = seeder.getCentroidIndexes();
        for (int cIndex = 0; cIndex < centroids.length; cIndex++) {
            clusterAssociations[centroidIndexes[cIndex]] = cIndex;
            centroids[cIndex] =
                    dset.getInstance(centroidIndexes[cIndex]).copyContent();
        }
        setClusterAssociations(clusterAssociations);
        DataInstance[] clusterHubs = new DataInstance[numClusters];
        // When there are no reassignments, we can end the clustering.
        boolean noReassignments;
        double errorPrevious;
        double errorCurrent = Double.MAX_VALUE;
        // This is initialized to true for the first iteration to go through.
        boolean errorDifferenceSignificant = true;
        setIterationIndex(0);
        float smallestDistance;
        float currDistance;
        float[] hubCentroidDists = new float[numClusters];
        float hcdMax;
        float hcdMin;
        float hcdAvg;
        // First iteration assignments.
        for (int i = 0; i < clusterAssociations.length; i++) {
            int closestCentroid = -1;
            smallestDistance = Float.MAX_VALUE;
            for (int j = 0; j < numClusters; j++) {
                currDistance = cmet.dist(dset.data.get(i), centroids[j]);
                if (currDistance < smallestDistance) {
                    smallestDistance = currDistance;
                    closestCentroid = j;
                }
            }
            clusterAssociations[i] = closestCentroid;
        }
        do {
            nextIteration();
            clusters = getClusters();
            int maxFrequency;
            int maxIndex;
            int currClusterSize;
            hcdMax = -Float.MAX_VALUE;
            hcdMin = Float.MAX_VALUE;
            hcdAvg = 0;
            float mAvg = 0;
            float mMin = Float.MAX_VALUE;
            float mMax = -Float.MAX_VALUE;
            for (int cIndex = 0; cIndex < numClusters; cIndex++) {
                float mDist = Float.MAX_VALUE;
                float currDist;
                currClusterSize = clusters[cIndex].size();
                if (currClusterSize == 1) {
                    // The trivial case.
                    clusterHubs[cIndex] = clusters[cIndex].getInstance(0);
                    continue;
                }
                maxFrequency = 0;
                maxIndex = 0;
                for (int j = 0; j < currClusterSize; j++) {
                    // Now look for the hubs.
                    if (hubnessArray[clusters[cIndex].
                            getWithinDataSetIndexOf(j)] > maxFrequency) {
                        maxFrequency = hubnessArray[
                                clusters[cIndex].getWithinDataSetIndexOf(j)];
                        maxIndex = j;
                    }
                    currDist = cmet.dist(
                            clusters[cIndex].getInstance(j), centroids[cIndex]);
                    // The smallest distance to the centroid is the distance
                    // from the medoid.
                    if (currDist < mDist) {
                        mDist = currDist;
                    }
                }
                if (mDist < mMin) {
                    mMin = mDist;
                }
                if (mDist > mMax) {
                    mMax = mDist;
                }
                clusterHubs[cIndex] = clusters[cIndex].getInstance(maxIndex);
                hubCentroidDists[cIndex] = cmet.dist(clusterHubs[cIndex],
                        centroids[cIndex]);
                if (hubCentroidDists[cIndex] > hcdMax) {
                    hcdMax = hubCentroidDists[cIndex];
                }
                if (hubCentroidDists[cIndex] < hcdMin) {
                    hcdMin = hubCentroidDists[cIndex];
                }
                hcdAvg += hubCentroidDists[cIndex];
                mAvg += mDist;
            }
            hcdAvg /= (float) numClusters;
            mAvg /= (float) numClusters;
            hcMinVect.add(hcdMin);
            hcMaxVect.add(hcdMax);
            hcAvgVect.add(hcdAvg);
            mAvgVect.add(mAvg);
            mMinVect.add(mMin);
            mMaxVect.add(mMax);
            hcdWriter.println(getIterationIndex() + "," + hcdMin + "," + hcdMax
                    + "," + hcdAvg + "," + mMin + "," + mMax + "," + mAvg);
            noReassignments = true;
            // Now actually assign points to closest centroids.
            for (int i = 0; i < dset.size(); i++) {
                int closestCentroidIndex = -1;
                smallestDistance = Float.MAX_VALUE;
                for (int j = 0; j < numClusters; j++) {
                    currDistance = cmet.dist(dset.data.get(i), centroids[j]);
                    if (currDistance < smallestDistance) {
                        smallestDistance = currDistance;
                        closestCentroidIndex = j;
                    }
                }
                if (closestCentroidIndex != clusterAssociations[i]) {
                    noReassignments = false;
                }
                clusterAssociations[i] = closestCentroidIndex;
            }
            clusters = getClusters();
            // Calculate new centroids.
            for (int i = 0; i < numClusters; i++) {
                centroids[i] = clusters[i].getCentroid();
            }
            errorPrevious = errorCurrent;
            errorCurrent = calculateIterationError(centroids);
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
        endCentroids = centroids;
        setClusterAssociations(bestAssociations);
        hcdWriter.close();
        flagAsInactive();
    }

    /**
     * Calculates the iteration calculateIterationError for convergence check.
     *
     * @param centroids An array of cluster centroid objects.
     * @return A sum of squared distances from points to centroids.
     * @throws Exception
     */
    private double calculateIterationError(DataInstance[] centroids)
            throws Exception {
        DataSet dset = getDataSet();
        CombinedMetric cmet = getCombinedMetric();
        int[] clusterAssociations = getClusterAssociations();
        double iterationError = 0;
        float centroidDistance;
        for (int i = 0; i < dset.size(); i++) {
            centroidDistance = cmet.dist(centroids[clusterAssociations[i]],
                    dset.getInstance(i));
            iterationError += centroidDistance * centroidDistance;
        }
        return iterationError;
    }

    /**
     * Prints out the command line parameter specification.
     */
    public static void info() {
        System.out.println("arg0: numberOfDimensions");
        System.out.println("arg1: numberOfDatasets");
        System.out.println("arg2: repetitions per dataset");
        System.out.println("arg3: neighbor set size");
        System.out.println("arg4: number of clusters");
        System.out.println("arg5: outDirectory");
    }

    /**
     * Runs the hub tracking in K-means experiment.
     *
     * @param args Command line arguments.
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        if (args.length != 6) {
            info();
            return;
        }
        int dataSize = 10000;
        boolean isPairedGaussian = true;
        int numClusters = Integer.parseInt(args[4]);
        int neighborSize = Integer.parseInt(args[3]);
        int numTimesForDataSet = Integer.parseInt(args[2]);
        int numDataSets = Integer.parseInt(args[1]);
        int numDimensions = Integer.parseInt(args[0]);
        File outDir = new File(args[5], "nDim" + args[0] + "nSS" + args[3]
                + "k" + args[4]);
        MultiGaussianMixForClusteringTesting genMix = null;
        float[] AhcMinVect = new float[300];
        float[] AhcMaxVect = new float[300];
        float[] AhcAvgVect = new float[300];
        float[] AmAvgVect = new float[300];
        float[] AmMinVect = new float[300];
        float[] AmMaxVect = new float[300];
        float[] AhcMinCounts = new float[300];
        float[] AhcMaxCounts = new float[300];
        float[] AhcAvgCounts = new float[300];
        float[] AmAvgCounts = new float[300];
        float[] AmMinCounts = new float[300];
        float[] AmMaxCounts = new float[300];
        float avgDistMin = 0;
        float avgDistMax = 0;
        float avgDistAvg = 0;
        float avgDistMinIC = 0;
        float avgDistMaxIC = 0;
        float avgDistAvgIC = 0;
        for (int i = 0; i < numDataSets; i++) {
            System.out.println("Starting dataset" + i);
            genMix = new MultiGaussianMixForClusteringTesting(
                    numClusters, numDimensions, dataSize, isPairedGaussian);
            DataSet testData = genMix.generateRandomCollection();
            CombinedMetric cmet = CombinedMetric.FLOAT_EUCLIDEAN;
            NeighborSetFinder nsf = new NeighborSetFinder(testData, cmet);
            nsf.calculateDistances();
            nsf.calculateNeighborSets(neighborSize);
            float[][] distances = nsf.getDistances();
            // Find the average, min and max distance between points.
            float distMin = Float.MAX_VALUE;
            float distMax = -Float.MAX_VALUE;
            float distAvg = 0;
            float dCounter = 0;
            for (int j = 0; j < distances.length; j++) {
                for (int l = 0; l < distances[j].length; l++) {
                    dCounter++;
                    distAvg += distances[j][l];
                    if (distances[j][l] > distMax) {
                        distMax = distances[j][l];
                    }
                    if (distances[j][l] < distMin) {
                        distMin = distances[j][l];
                    }
                }
            }
            distAvg /= dCounter;
            avgDistMin += distMin;
            avgDistMax += distMax;
            avgDistAvg += distAvg;
            for (int j = 0; j < numTimesForDataSet; j++) {
                KMeansHubnessTester kmht = new KMeansHubnessTester();
                kmht.setDataSet(testData);
                kmht.setHubness(nsf.getNeighborFrequencies());
                kmht.setCombinedMetric(cmet);
                kmht.setNumClusters(numClusters);
                File wFile = new File(
                        outDir,
                        "distLog" + (i * numTimesForDataSet + j) + ".csv");
                boolean noErrors = false;
                while (!noErrors) {
                    try {
                        kmht.cluster();
                        noErrors = true;
                    } catch (Exception e) {
                        FileUtil.createFile(wFile);
                        PrintWriter hcdWriter =
                                new PrintWriter(new FileWriter(wFile));
                        kmht.hcdWriter = hcdWriter;
                    }
                }
                Cluster[] config = kmht.getClusters();
                for (int iter = 0; iter < config.length; iter++) {
                    float distMinIC = Float.MAX_VALUE;
                    float distMaxIC = -Float.MAX_VALUE;
                    float distAvgIC = 0;
                    float TD = 0;
                    if (config[iter] == null || config[iter].isEmpty()
                            || config[iter].size() == 1) {
                        continue;
                    }
                    for (int elIndex1 = 0; elIndex1 < config[iter].size();
                            elIndex1++) {
                        for (int elIndex2 = elIndex1 + 1;
                                elIndex2 < config[iter].size(); elIndex2++) {
                            TD = distances[config[iter].
                                    getWithinDataSetIndexOf(elIndex1)][
                                    config[iter].getWithinDataSetIndexOf(
                                    elIndex2) - config[iter].
                                    getWithinDataSetIndexOf(elIndex1) - 1];
                            if (TD > distMaxIC) {
                                distMaxIC = TD;
                            }
                            if (TD < distMinIC) {
                                distMinIC = TD;
                            }
                            distAvgIC += TD;
                        }
                    }
                    distAvgIC /= 0.5f * (float) (config[iter].size()
                            * (config[iter].size() - 1));
                    avgDistMinIC += distMinIC;
                    avgDistMaxIC += distMaxIC;
                    avgDistAvgIC += distAvgIC;
                }
                for (int iter = 0; iter < kmht.hcMinVect.size(); iter++) {
                    AhcMinVect[iter] += kmht.hcMinVect.get(iter);
                    AhcMinCounts[iter]++;
                }
                for (int iter = 0; iter < kmht.hcMaxVect.size(); iter++) {
                    AhcMaxVect[iter] += kmht.hcMaxVect.get(iter);
                    AhcMaxCounts[iter]++;
                }
                for (int iter = 0; iter < kmht.hcAvgVect.size(); iter++) {
                    AhcAvgVect[iter] += kmht.hcAvgVect.get(iter);
                    AhcAvgCounts[iter]++;
                }
                for (int iter = 0; iter < kmht.mMinVect.size(); iter++) {
                    AmMinVect[iter] += kmht.mMinVect.get(iter);
                    AmMinCounts[iter]++;
                }
                for (int iter = 0; iter < kmht.mMaxVect.size(); iter++) {
                    AmMaxVect[iter] += kmht.mMaxVect.get(iter);
                    AmMaxCounts[iter]++;
                }
                for (int iter = 0; iter < kmht.mAvgVect.size(); iter++) {
                    AmAvgVect[iter] += kmht.mAvgVect.get(iter);
                    AmAvgCounts[iter]++;
                }
            }
            System.gc();
        }
        // Now average the totals.
        avgDistMin /= (float) numDataSets;
        avgDistMax /= (float) numDataSets;
        avgDistAvg /= (float) numDataSets;
        avgDistMinIC /= (float) (numDataSets * numTimesForDataSet *
                numClusters);
        avgDistMaxIC /= (float) (numDataSets * numTimesForDataSet *
                numClusters);
        avgDistAvgIC /= (float) (numDataSets * numTimesForDataSet *
                numClusters);
        for (int iter = 0; iter < 300; iter++) {
            if (AhcMinCounts[iter] > 0) {
                AhcMinVect[iter] /= AhcMinCounts[iter];
            }
            if (AhcMaxCounts[iter] > 0) {
                AhcMaxVect[iter] /= AhcMaxCounts[iter];
            }
            if (AhcAvgCounts[iter] > 0) {
                AhcAvgVect[iter] /= AhcAvgCounts[iter];
            }
            if (AmMinCounts[iter] > 0) {
                AmMinVect[iter] /= AmMinCounts[iter];
            }
            if (AmMaxCounts[iter] > 0) {
                AmMaxVect[iter] /= AmMaxCounts[iter];
            }
            if (AmAvgCounts[iter] > 0) {
                AmAvgVect[iter] /= AmAvgCounts[iter];
            }
        }
        File finalSummary = new File(outDir, "finalSummary.csv");
        FileUtil.createFile(finalSummary);
        PrintWriter pw = new PrintWriter(new FileWriter(finalSummary));
        try {
            pw.println("hMin,hMax,hAvg,mMin,mMax,mAvg");
            for (int iter = 0; iter < 300; iter++) {
                pw.println(AhcMinVect[iter] + "," + AhcMaxVect[iter] + ","
                        + AhcAvgVect[iter] + "," + AmMinVect[iter] + ","
                        + AmMaxVect[iter] + "," + AmAvgVect[iter]);
            }
        } catch (Exception e) {
            throw e;
        } finally {
            pw.close();
        }
        File finalDistanceSummary = new File(outDir,
                "finalAvgDistancesInDataset.csv");
        FileUtil.createFile(finalDistanceSummary);
        pw = new PrintWriter(new FileWriter(finalDistanceSummary));
        try {
            pw.println("dMin,dMax,dAvg,dMinIC,dMaxIC,dAvgIC");
            pw.println(avgDistMin + "," + avgDistMax + "," + avgDistAvg + ","
                    + avgDistMinIC + "," + avgDistMaxIC + "," + avgDistAvgIC);
        } catch (Exception e) {
            throw e;
        } finally {
            pw.close();
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
}
