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
package preprocessing.instance_selection;

import algref.Author;
import algref.JournalPublication;
import algref.Publication;
import algref.Publisher;
import data.neighbors.HitMissNetwork;
import data.neighbors.NSFUserInterface;
import data.neighbors.NeighborSetFinder;
import data.representation.DataSet;
import distances.primary.CombinedMetric;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import learning.supervised.methods.knn.KNN;
import util.AuxSort;

/**
 * This class implements the instance selection method based on calculating the
 * HM score in Hit-Miss networks, as proposed in the paper titled 'Class
 * Conditional Nearest Neighbor and Large Margin Instance Selection' by E.
 * Marchiori that was published in IEEE Transactions on Pattern Analysis and
 * Machine Intelligence in 2010. The method was proposed for 1-NN classification
 * but this implementation makes it possible to apply the method for kNN
 * classification with k > 1 as well. Whether that is always appropriate or not
 * remains to be seen, but it gives the users the option for experimentation.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class HMScore extends InstanceSelector implements NSFUserInterface {

    public static final int DEFAULT_NEIGHBORHOOD_SIZE = 1;
    public static final int DEFAULT_NUM_THREADS = 8;
    // Object that holds the kNN sets.
    private NeighborSetFinder nsf;
    // The upper triangular distance matrix on the data.
    private float[][] distMat;
    // Hit-Miss network on the data, used for calculating the HM scores.
    private HitMissNetwork hmNetwork;
    // The neighborhood size to use for the hit-miss network.
    private int kHM = DEFAULT_NEIGHBORHOOD_SIZE;
    private int numThreads = DEFAULT_NUM_THREADS;
    private boolean permitNoChangeInclusions = true;

    /**
     * @constructor
     */
    public HMScore() {
    }

    /**
     * Initialization.
     *
     * @param nsf Neighbor set finder object with some existing kNN info.
     * @param kHM Integer representing the neighborhood size to use for the
     * hit-miss network.
     * @constructor
     */
    public HMScore(NeighborSetFinder nsf, int kHM) {
        this.nsf = nsf;
        setOriginalDataSet(nsf.getDataSet());
        this.distMat = nsf.getDistances();
        this.kHM = kHM;
    }

    /**
     * Initialization.
     *
     * @param dset DataSet to reduce.
     * @param distMat float[][] that is the upper triangular distance matrix on
     * the data.
     * @param kHM Integer that is the neighborhood size to use for generating
     * the hit-miss network.
     */
    public HMScore(DataSet dset, float[][] distMat, int kHM) {
        setOriginalDataSet(dset);
        this.distMat = distMat;
        this.kHM = kHM;
    }

    /**
     * @param permitNoChangeInclusions Boolean flag indicating whether to
     * consider elements for incremental inclusion when they have no visible
     * negative or positive effect or to stop the process when such an element
     * is reached. If set to false, a very small number of prototypes is
     * selected. If set to true, a much lower error is achieved.
     */
    public void setInclusionPermissions(boolean permitNoChangeInclusions) {
        this.permitNoChangeInclusions = permitNoChangeInclusions;
    }

    /**
     * @param numThreads Integer that is the number of threads to use in parts
     * of the code where multi-threading is supported.
     */
    public void setNumThreads(int numThreads) {
        this.numThreads = numThreads;
    }

    @Override
    public Publication getPublicationInfo() {
        JournalPublication pub = new JournalPublication();
        pub.setTitle("Class Conditional Nearest Neighbor and Large Margin "
                + "Instance Selection");
        pub.addAuthor(new Author("E.", "Marchiori"));
        pub.setJournalName("IEEE Transactions on Pattern Analysis and Machine"
                + " Intelligence");
        pub.setYear(2010);
        pub.setVolume(32);
        pub.setIssue(2);
        pub.setStartPage(364);
        pub.setEndPage(370);
        pub.setDoi("10.1109/TPAMI.2009.164");
        pub.setPublisher(Publisher.IEEE);
        return pub;
    }

    /**
     * Calculate the number of false predictions according to the current kNN
     * sets.
     *
     * @param dset DataSet to calculate the predictions for.
     * @param numClasses Integer that is the number of classes in the data.
     * @param finder NeighborSetFinder object holding the kNN sets.
     * @return Integer value that is the count of false predictions.
     * @throws Exception
     */
    private int countFalsePredictions(DataSet dset, int numClasses,
            NeighborSetFinder finder) throws Exception {
        int numFalsePredictions = 0;
        KNN classifier = new KNN(dset, numClasses,
                finder.getCombinedMetric(), finder.getCurrK());
        for (int i = 0; i < dset.size(); i++) {
            int queryLabel = dset.getLabelOf(i);
            int predictedLabel = classifier.classify(
                    dset.getInstance(i), null,
                    finder.getKNeighbors()[i]);
            if (queryLabel != predictedLabel) {
                numFalsePredictions++;
            }
        }
        return numFalsePredictions;
    }

    @Override
    public void reduceDataSet() throws Exception {
        DataSet originalDataSet = getOriginalDataSet();
        int datasize = originalDataSet.size();
        // Initialization.
        int numClasses = getNumClasses();
        // List that will contain the selected prototype indexes.
        ArrayList<Integer> protoIndexes = new ArrayList<>(datasize / 4);
        // Initialize the HM network.
        hmNetwork = new HitMissNetwork(originalDataSet, distMat, kHM);
        if (nsf != null) {
            hmNetwork.generateNetworkFromExistingNSF(nsf);
        } else {
            hmNetwork.generateNetwork();
            // We are going to need the kNN graph on the data for leave-one-out 
            // estimates.
            nsf = new NeighborSetFinder(originalDataSet, distMat);
            nsf.calculateNeighborSets(kHM);
        }
        // Calculate the initial leave-one-out-error without reduction.
        int numFalsePredictions = countFalsePredictions(originalDataSet,
                numClasses, nsf);
        // Generate the initial core set to be expanded.
        double[] hmScores = hmNetwork.computeAllHMScores();
        int numCorePoints = Math.max(numClasses, Math.min(datasize / 2,
                kHM * numClasses));
        int[] perm = AuxSort.sortIndexedValue(hmScores, true);
        boolean[] isInitialPrototype = new boolean[datasize];
        for (int i = 0; i < numCorePoints; i++) {
            protoIndexes.add(perm[i]);
            isInitialPrototype[perm[i]] = true;
        }
        NeighborSetFinder protoNSF = new NeighborSetFinder(originalDataSet,
                distMat);
        // This is the relevant neighborhood size to use in the leave-one-out 
        // estimates.
        int k = nsf.getCurrK();
        // The initial kNN sets will contain only the initial prototypes as 
        // neighbors.
        protoNSF.calculateNeighborSetsMultiThr(k, numThreads,
                isInitialPrototype);
        // Calculate the error of the sample.
        int numFalsePredictionsSample = countFalsePredictions(originalDataSet,
                numClasses, protoNSF);
        // Calculate the number of unacceptable instances based on the HM score.
        int index = hmScores.length - 1;
        while (index >= 0 && hmScores[index] <= 0) {
            index--;
        }
        NeighborSetFinder protoExtendedNSF;
        int numUnacceptable = datasize - index - 1;
        for (int i = numCorePoints; i < datasize - numUnacceptable; i++) {
            if (numFalsePredictionsSample <= numFalsePredictions) {
                break;
            }
            protoExtendedNSF = protoNSF.copy();
            protoExtendedNSF.considerNeighbor(perm[i], false);
            int numFalsePredictionsCurr = countFalsePredictions(originalDataSet,
                    numClasses, protoExtendedNSF);
            if (permitNoChangeInclusions) {
                // Not strict inequality.
                if (numFalsePredictionsCurr <= numFalsePredictionsSample) {
                    protoNSF = protoExtendedNSF;
                    protoIndexes.add(perm[i]);
                    numFalsePredictionsSample = numFalsePredictionsCurr;
                } else {
                    break;
                }
            } else {
                // Strict inequality.
                if (numFalsePredictionsCurr < numFalsePredictionsSample) {
                    protoNSF = protoExtendedNSF;
                    protoIndexes.add(perm[i]);
                    numFalsePredictionsSample = numFalsePredictionsCurr;
                } else {
                    break;
                }
            }
        }
        // Check whether at least one instance of each class has been selected.
        int[] protoClassCounts = new int[numClasses];
        int numEmptyClasses = numClasses;
        for (int i = 0; i < protoIndexes.size(); i++) {
            int label = originalDataSet.getLabelOf(protoIndexes.get(i));
            if (protoClassCounts[label] == 0) {
                numEmptyClasses--;
            }
            protoClassCounts[label]++;
        }
        if (numEmptyClasses > 0) {
            HashMap<Integer, Integer> tabuMap =
                    new HashMap<>(protoIndexes.size() * 2);
            for (int i = 0; i < protoIndexes.size(); i++) {
                tabuMap.put(protoIndexes.get(i), i);
            }
            for (int i = 0; i < originalDataSet.size(); i++) {
                int label = originalDataSet.getLabelOf(i);
                if (!tabuMap.containsKey(i) && protoClassCounts[label] == 0) {
                    protoIndexes.add(i);
                    protoClassCounts[label]++;
                    numEmptyClasses--;
                }
                if (numEmptyClasses == 0) {
                    break;
                }
            }
        }
        // Set the selected prototype indexes and sort them.
        setPrototypeIndexes(protoIndexes);
        sortSelectedIndexes();
    }

    @Override
    public void reduceDataSet(int numPrototypes) throws Exception {
        DataSet originalDataSet = getOriginalDataSet();
        int datasize = originalDataSet.size();
        int numClasses = getNumClasses();
        ArrayList<Integer> protoIndexes = new ArrayList<>(datasize / 4);
        // Initialize the HM network.
        hmNetwork = new HitMissNetwork(originalDataSet, distMat, kHM);
        if (nsf != null) {
            hmNetwork.generateNetworkFromExistingNSF(nsf);
        } else {
            hmNetwork.generateNetwork();
        }
        // Calculate the HM scores.
        double[] hmScores = hmNetwork.computeAllHMScores();
        int[] perm = AuxSort.sortIndexedValue(hmScores, true);
        int numSelected = Math.max(Math.min(numPrototypes, datasize),
                numClasses);
        for (int i = 0; i < numSelected; i++) {
            protoIndexes.add(perm[i]);
        }
        // Check whether at least one instance of each class has been selected.
        // If not, continue considering instances in the same order and replace
        // the lowest impact replaceable instances with the ones of the classes
        // that have not been represented.
        int[] protoClassCounts = new int[numClasses];
        int numEmptyClasses = numClasses;
        for (int i = 0; i < protoIndexes.size(); i++) {
            int label = originalDataSet.getLabelOf(protoIndexes.get(i));
            if (protoClassCounts[label] == 0) {
                numEmptyClasses--;
            }
            protoClassCounts[label]++;
        }
        if (numEmptyClasses > 0) {
            for (int i = numSelected; i < originalDataSet.size(); i++) {
                int label = originalDataSet.getLabelOf(perm[i]);
                if (protoClassCounts[label] == 0) {
                    for (int j = protoIndexes.size() - 1; j >= 0; j--) {
                        int protoLabel = originalDataSet.getLabelOf(
                                protoIndexes.get(j));
                        if (protoClassCounts[protoLabel] > 1) {
                            protoIndexes.set(j, perm[i]);
                            protoClassCounts[label]++;
                            numEmptyClasses--;
                        }
                    }
                }
                if (numEmptyClasses == 0) {
                    break;
                }
            }
        }
        setPrototypeIndexes(protoIndexes);
        sortSelectedIndexes();
    }

    @Override
    public InstanceSelector copy() {
        if (nsf == null) {
            return new HMScore(getOriginalDataSet(), distMat, kHM);
        } else {
            return new HMScore(nsf, kHM);
        }
    }

    @Override
    public void setNSF(NeighborSetFinder nsf) {
        this.nsf = nsf;
        distMat = nsf.getDistances();
    }

    @Override
    public NeighborSetFinder getNSF() {
        return nsf;
    }

    @Override
    public void noRecalcs() {
    }

    @Override
    public void calculatePrototypeHubness(int k) throws Exception {
        if (nsf != null) {
            // Here we have some prior neighbor occurrence information and we
            // can re-use it to speed-up the top k prototype search.
            this.setNeighborhoodSize(k);
            if (k <= 0) {
                return;
            }
            DataSet originalDataSet = getOriginalDataSet();
            // The original k-neighbor information is used in order to speed up
            // the top-k prototype calculations, in those cases where these
            // prototypes are already known to occur as neighbors.
            // These occurrences are re-used dynamically.
            int[][] kns = nsf.getKNeighbors();
            float[][] kd = nsf.getKDistances();
            // Array that holds the kneighbors where only prototypes are allowed
            // as neighbor points.
            int[][] kneighbors = new int[originalDataSet.size()][k];
            int kNSF = kns[0].length;
            HashMap<Integer, Integer> protoMap =
                    new HashMap<>(getPrototypeIndexes().size() * 2);
            ArrayList<Integer> protoIndexes = getPrototypeIndexes();
            // Fill the HashMap with indexes of selected prototype points.
            for (int i = 0; i < protoIndexes.size(); i++) {
                protoMap.put(protoIndexes.get(i), i);
            }
            int l;
            int datasize = originalDataSet.size();
            // Distances to k-closest prototypes for each point from the
            // training data.
            float[][] kdistances = new float[datasize][k];
            // Temporary lengths of k-neighbor lists as they are being
            // calculated.
            int[] kcurrLen = new int[datasize];
            // Distance matrix.
            float[][] distMatrix = nsf.getDistances();
            // Intervals are there to help with re-using existing neighbors, in
            // order to speed up the calculations and simplify the code.
            ArrayList<Integer> intervals;
            int upper, lower;
            int min, max;
            for (int i = 0; i < datasize; i++) {
                intervals = new ArrayList(k + 2);
                // Looping is from interval indexes + 1, so -1 goes as the left
                // limit. All prototype occurrences from the original kNN sets
                // are inserted into the intervals list.
                intervals.add(-1);
                for (int j = 0; j < kNSF; j++) {
                    if (protoMap.containsKey(kns[i][j])) {
                        // What we insert as a neighbor is not the prototype
                        // index within the original data, but rather among the
                        // prototype array.
                        kneighbors[i][kcurrLen[i]] = protoMap.get(kns[i][j]);
                        kdistances[i][kcurrLen[i]] = kd[i][j];
                        kcurrLen[i]++;
                        // We insert the originalDataSet prototype index to the
                        // intervals.
                        intervals.add(kns[i][j]);
                    }
                    // If the original kNSF was larger than this neighborhood
                    // size, we need to make sure not to exceed the k limit.
                    if (kcurrLen[i] >= k) {
                        break;
                    }
                }
                intervals.add(datasize + 1);
                // We sort the known prototype neighbor indexes.
                Collections.sort(intervals);
                // If we haven't already finished, we proceed with the nearest
                // prototype search.
                if (kcurrLen[i] < k) {
                    // It needs to iterate one less than to the very end, as the
                    // last limitation is there, so there are no elements beyond
                    int iSizeRed = intervals.size() - 1;
                    for (int ind = 0; ind < iSizeRed; ind++) {
                        lower = intervals.get(ind);
                        upper = intervals.get(ind + 1);
                        for (int j = lower + 1; j < upper - 1; j++) {
                            // If the point is a prototype and different from
                            // current.
                            if (i != j && protoMap.containsKey(j)) {
                                min = Math.min(i, j);
                                max = Math.max(i, j);
                                if (kcurrLen[i] > 0) {
                                    if (kcurrLen[i] == k) {
                                        if (distMatrix[min][max - min - 1]
                                                < kdistances[i][
                                                        kcurrLen[i] - 1]) {
                                            // Search to see where to insert.
                                            l = k - 1;
                                            while ((l >= 1)
                                                    && distMatrix[min][
                                                    max - min - 1]
                                                    < kdistances[i][l - 1]) {
                                                kdistances[i][l] =
                                                        kdistances[i][l - 1];
                                                kneighbors[i][l] =
                                                        kneighbors[i][l - 1];
                                                l--;
                                            }
                                            kdistances[i][l] =
                                                    distMatrix[min][
                                                    max - min - 1];
                                            kneighbors[i][l] = protoMap.get(j);
                                        }
                                    } else {
                                        if (distMatrix[min][max - min - 1]
                                                < kdistances[i][
                                                        kcurrLen[i] - 1]) {
                                            //search to see where to insert
                                            l = kcurrLen[i] - 1;
                                            kdistances[i][kcurrLen[i]] =
                                                    kdistances[i][
                                                            kcurrLen[i] - 1];
                                            kneighbors[i][kcurrLen[i]] =
                                                    kneighbors[i][
                                                            kcurrLen[i] - 1];
                                            while ((l >= 1)
                                                    && distMatrix[min][
                                                    max - min - 1]
                                                    < kdistances[i][l - 1]) {
                                                kdistances[i][l] =
                                                        kdistances[i][l - 1];
                                                kneighbors[i][l] =
                                                        kneighbors[i][l - 1];
                                                l--;
                                            }
                                            kdistances[i][l] =
                                                    distMatrix[min][
                                                    max - min - 1];
                                            kneighbors[i][l] = protoMap.get(j);
                                            kcurrLen[i]++;
                                        } else {
                                            kdistances[i][kcurrLen[i]] =
                                                    distMatrix[min][
                                                    max - min - 1];
                                            kneighbors[i][kcurrLen[i]] =
                                                    protoMap.get(j);
                                            kcurrLen[i]++;
                                        }
                                    }
                                } else {
                                    kdistances[i][0] =
                                            distMatrix[min][max - min - 1];
                                    kneighbors[i][0] = protoMap.get(j);
                                    kcurrLen[i] = 1;
                                }
                            }
                        }
                    }
                }
            }
            int numClasses = getNumClasses();
            // Prototype occurrence frequency array.
            int[] protoHubness = new int[protoIndexes.size()];
            // Prototype good occurrence frequency array.
            int[] protoGoodHubness = new int[protoIndexes.size()];
            // Prototype detrimental occurrence frequency array.
            int[] protoBadHubness = new int[protoIndexes.size()];
            // Prototype class-conditional neighbor occurrence frequencies.
            int[][] protoClassHubness =
                    new int[numClasses][protoIndexes.size()];
            setPrototypeHubness(protoHubness);
            setPrototypeGoodHubness(protoGoodHubness);
            setPrototypeBadHubness(protoBadHubness);
            setProtoClassHubness(protoClassHubness);
            int currLabel;
            // Loop through the top-k prototype sets once.
            for (int i = 0; i < datasize; i++) {
                currLabel = originalDataSet.getLabelOf(i);
                for (int j = 0; j < k; j++) {
                    if (currLabel == originalDataSet.getLabelOf(
                            protoIndexes.get(kneighbors[i][j]))) {
                        protoGoodHubness[kneighbors[i][j]]++;
                    } else {
                        protoBadHubness[kneighbors[i][j]]++;
                    }
                    protoHubness[kneighbors[i][j]]++;
                    protoClassHubness[currLabel][kneighbors[i][j]]++;
                }
            }
            setProtoNeighborSets(kneighbors);
        } else {
            // In this case, no prior neighbor information is available, so
            // we just proceed in a simple way.
            this.setNeighborhoodSize(k);
            DataSet originalDataSet = getOriginalDataSet();
            ArrayList<Integer> prototypeIndexes = getPrototypeIndexes();
            int[][] kneighbors = new int[originalDataSet.size()][k];
            // Make a subcollection.
            DataSet tCol = originalDataSet.cloneDefinition();
            tCol.data = new ArrayList<>(prototypeIndexes.size());
            for (int index : prototypeIndexes) {
                tCol.data.add(originalDataSet.getInstance(index));
            }
            int numClasses = getNumClasses();
            int[] protoHubness = new int[prototypeIndexes.size()];
            int[] protoGoodHubness = new int[prototypeIndexes.size()];
            int[] protoBadHubness = new int[prototypeIndexes.size()];
            int[][] protoClassHubness =
                    new int[numClasses][prototypeIndexes.size()];
            setPrototypeHubness(protoHubness);
            setPrototypeGoodHubness(protoGoodHubness);
            setPrototypeBadHubness(protoBadHubness);
            setProtoClassHubness(protoClassHubness);
            int currLabel;
            int protoLabel;
            CombinedMetric cmet = this.getCombinedMetric();
            for (int i = 0; i < originalDataSet.size(); i++) {
                currLabel = originalDataSet.getLabelOf(i);
                kneighbors[i] = NeighborSetFinder.getIndexesOfNeighbors(tCol,
                        originalDataSet.getInstance(i), k, cmet);
                for (int nIndex : kneighbors[i]) {
                    protoClassHubness[currLabel][nIndex]++;
                    protoHubness[nIndex]++;
                    protoLabel = originalDataSet.getLabelOf(nIndex);
                    if (protoLabel == currLabel) {
                        protoGoodHubness[nIndex]++;
                    } else {
                        protoBadHubness[nIndex]++;
                    }
                }
            }
        }
    }
}
