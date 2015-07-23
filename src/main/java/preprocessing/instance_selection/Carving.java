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
import java.util.List;
import learning.supervised.methods.knn.KNN;

/**
 * This class implements the carving instance selection method based on 
 * calculating the HM scores in Hit-Miss networks with subsequent refinement, as 
 * proposed in the paper titled 'Class Conditional Nearest Neighbor and Large 
 * Margin Instance Selection' by E. Marchiori that was published in IEEE 
 * Transactions on Pattern Analysis and Machine Intelligence in 2010. The method 
 * was proposed for 1-NN classification but this implementation makes it 
 * possible to apply the method for kNN classification with k > 1 as well. 
 * Whether that is always appropriate or not remains to be seen, but it gives 
 * the users the option for experimentation. This method severely reduces the 
 * number of examples in practice if kHM == 1 is used and this is definitely 
 * then only good for 1-NN classification. Therefore, it is a good idea to use 
 * larger kHM values, possibly kHM == k for k-NN classification. This is not 
 * strictly enforced, in order to allow for free experimentation, but it is 
 * something that should be kept in mind while experimenting.
 * 
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class Carving extends InstanceSelector implements NSFUserInterface {
    
    public static final int DEFAULT_NEIGHBORHOOD_SIZE = 1;
    public static final int DEFAULT_NUM_THREADS = 8;
    // Object that holds the kNN sets.
    private NeighborSetFinder nsf;
    // The upper triangular distance matrix on the data.
    private float[][] distMat;
    // Hit-Miss network on the data, used for calculating the HM scores.
    private List<HitMissNetwork> hmNetworks;
    // The neighborhood size to use for the hit-miss network.
    private int kHM = DEFAULT_NEIGHBORHOOD_SIZE;
    private int numThreads = DEFAULT_NUM_THREADS;
    private boolean permitNoChangeInclusions = true;
    
    private HMScore internalReducer;
    
    /**
     * Default constructor.
     */
    public Carving() {
    }

    /**
     * Initialization.
     *
     * @param nsf Neighbor set finder object with some existing kNN info.
     * @param kHM Integer representing the neighborhood size to use for the
     * hit-miss network.
     */
    public Carving(NeighborSetFinder nsf, int kHM) {
        this.nsf = nsf;
        if (nsf == null) {
            throw new IllegalArgumentException("Null kNN object provided.");
        }
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
    public Carving(DataSet dset, float[][] distMat, int kHM) {
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
        // First obtain the superset by performing HMScore selection.
        if (nsf != null) {
            internalReducer = new HMScore(nsf, kHM);
        } else {
            internalReducer = new HMScore(originalDataSet, distMat, kHM);
        }
        internalReducer.setInclusionPermissions(permitNoChangeInclusions);
        internalReducer.reduceDataSet();
        ArrayList<Integer> superProtoIndexes =
                internalReducer.getPrototypeIndexes();
        DataSet superSet = originalDataSet.getSubsample(superProtoIndexes);
        if (superSet.countCategories() < 2) {
            // The approach can not be applied, so we return the HMScore results
            // instead.
            setPrototypeIndexes(superProtoIndexes);
            sortSelectedIndexes();
            return;
        }
        // Make the labels conform to a range if a category is missing.
        superSet = superSet.copy();
        superSet.standardizeCategories();
        // Generate the distance matrix for the initial super-set.
        float[][] superDistMat = new float[superSet.size()][];
        int minIndex, maxIndex;
        for (int i = 0; i < superDistMat.length; i++) {
            superDistMat[i] = new float[superDistMat.length - i - 1];
            for (int j = 0; j < superDistMat[i].length; j++) {
                minIndex = Math.min(superProtoIndexes.get(i),
                        superProtoIndexes.get(i + j + 1));
                maxIndex = Math.max(superProtoIndexes.get(i),
                        superProtoIndexes.get(i + j + 1));
                superDistMat[i][j] = distMat[minIndex][maxIndex - minIndex - 1];
            }
        }
        hmNetworks = new ArrayList<>(10);
        // Generate the initial hit-miss network.
        HitMissNetwork superHMNetwork = new HitMissNetwork(superSet,
                superDistMat,
                Math.max(1, Math.min(kHM, superSet.getMinClassSize())));
        superHMNetwork.generateNetwork();
        hmNetworks.add(superHMNetwork);
        // List that will contain the selected prototype indexes.
        ArrayList<Integer> protoIndexes = new ArrayList<>(datasize / 4);
        ArrayList<Integer> currNetworkIndexes, currAddedIndexes,
                newNetworkIndexes;
        // Now get the initial subsample seed based on the carving criterion.
        float[] missFreqs = superHMNetwork.getMissNeighbOccFreqs();
        currAddedIndexes = new ArrayList<>(datasize / 4);
        currNetworkIndexes = new ArrayList<>(datasize / 4);
        HashMap<Integer, Integer> backwardIndexMap =
                new HashMap<>(datasize / 4);
        for (int i = 0; i < missFreqs.length; i++) {
            if (missFreqs[i] > 0) {
                currAddedIndexes.add(superProtoIndexes.get(i));
            } else {
                backwardIndexMap.put(currNetworkIndexes.size(), i);
                currNetworkIndexes.add(superProtoIndexes.get(i));
            }
        }
        if (currAddedIndexes.isEmpty()) {
            // The approach can not be applied, so we return the HMScore results
            // instead.
            setPrototypeIndexes(superProtoIndexes);
            sortSelectedIndexes();
            return;
        }
        // Calculate the error of the core.
        // Make the initial NSF object for leave-one-out evaluations.
        boolean[] isInitialPrototype = new boolean[datasize];
        for (int i = 0; i < currAddedIndexes.size(); i++) {
            isInitialPrototype[currAddedIndexes.get(i)] = true;
        }
        NeighborSetFinder protoNSF = new NeighborSetFinder(originalDataSet,
                distMat);
        // This is the relevant neighborhood size to use in the leave-one-out 
        // estimates.
        int k = nsf != null ? nsf.getCurrK() : kHM;
        // The initial kNN sets will contain only the initial prototypes as 
        // neighbors.
        protoNSF.calculateNeighborSetsMultiThr(k, numThreads,
                isInitialPrototype);
        int numFalsePredictions = countFalsePredictions(originalDataSet,
                numClasses, protoNSF);
        protoIndexes.addAll(currAddedIndexes);
        boolean iterate = true;
        while (iterate) {
            // Ok, so first we build a network and then subsample from it to
            // find new indexes to add to the selected set.
            DataSet currDSet = originalDataSet.getSubsample(currNetworkIndexes);
            // Some classes will be lost in the subsample eventually, so it is 
            // necessary to adjust the labels because of some hit-miss network 
            // internals where there might be errors when the category indexes 
            // go out of the range of the number of categories. When we make a 
            // copy of the data and standardize the categories, this is avoided.
            currDSet = currDSet.copy();
            currDSet.standardizeCategories();
            float[][] currDistMat = new float[currDSet.size()][];
            for (int i = 0; i < currDistMat.length; i++) {
                currDistMat[i] = new float[currDistMat.length - i - 1];
                for (int j = 0; j < currDistMat[i].length; j++) {
                    minIndex = Math.min(currNetworkIndexes.get(i),
                            currNetworkIndexes.get(i + j + 1));
                    maxIndex = Math.max(currNetworkIndexes.get(i),
                            currNetworkIndexes.get(i + j + 1));
                    currDistMat[i][j] = distMat[minIndex][
                            maxIndex - minIndex - 1];
                }
            }
            if (currDSet.countCategories() < 2) {
                break;
            }
            // Generate the current hit-miss network for analysis.
            HitMissNetwork currHMNetwork = new HitMissNetwork(currDSet,
                    currDistMat, Math.max(
                    1, Math.min(kHM, currDSet.getMinClassSize())));
            currHMNetwork.generateNetwork();
            float[] missFreqsCurr = currHMNetwork.getMissNeighbOccFreqs();
            // Obtain the total freqs of the previous network.
            float[] hitFreqsPrev = hmNetworks.get(hmNetworks.size() - 1).
                    getHitNeighbOccFreqs();
            float[] missFreqsPrev = hmNetworks.get(hmNetworks.size() - 1).
                    getMissNeighbOccFreqs();
            float[] totalFreqsPrev = new float[hitFreqsPrev.length];
            for (int i = 0; i < totalFreqsPrev.length; i++) {
                totalFreqsPrev[i] = missFreqsPrev[i] + hitFreqsPrev[i];
            }
            HashMap<Integer, Integer> newBackwardIndexMap =
                    new HashMap<>(datasize / 4);
            currAddedIndexes = new ArrayList<>(datasize / 4);
            newNetworkIndexes = new ArrayList<>(datasize / 4);
            for (int i = 0; i < missFreqsCurr.length; i++) {
                if (missFreqsCurr[i] > 0 &&
                        totalFreqsPrev[backwardIndexMap.get(i)] > 0) {
                    currAddedIndexes.add(currNetworkIndexes.get(i));
                } else {
                    newBackwardIndexMap.put(newNetworkIndexes.size(), i);
                    newNetworkIndexes.add(currNetworkIndexes.get(i));
                }
            }
            hmNetworks.add(currHMNetwork);
            if (currAddedIndexes.isEmpty()) {
                break;
            }
            for (int i = 0; i < currAddedIndexes.size(); i++) {
                protoNSF.considerNeighbor(currAddedIndexes.get(i), false);
            }
            int numFalsePredictionsNew = countFalsePredictions(originalDataSet,
                    numClasses, protoNSF);
            if (permitNoChangeInclusions &&
                    numFalsePredictionsNew <= numFalsePredictions) {
                numFalsePredictions = numFalsePredictionsNew;
                protoIndexes.addAll(currAddedIndexes);
                currNetworkIndexes = newNetworkIndexes;
                backwardIndexMap = newBackwardIndexMap;
            } else if (!permitNoChangeInclusions &&
                    numFalsePredictionsNew < numFalsePredictions) {
                numFalsePredictions = numFalsePredictionsNew;
                protoIndexes.addAll(currAddedIndexes);
                currNetworkIndexes = newNetworkIndexes;
                backwardIndexMap = newBackwardIndexMap;
            } else {
                iterate = false;
            }
            if (currNetworkIndexes.size() < kHM) {
                break;
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
        // This method automatically determines the correct number of prototypes
        // and it is usually a small number, so there is no way to enforce the 
        // number of prototypes here. Automatic selection is performed instead.
        reduceDataSet();
    }
    
    @Override
    public InstanceSelector copy() {
        if (nsf == null) {
            return new Carving(getOriginalDataSet(), distMat, kHM);
        } else {
            return new Carving(nsf, kHM);
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
