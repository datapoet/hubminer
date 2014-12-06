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

/**
 * This class implements the iterative case filtering (ICF) approach that was 
 * first proposed in the paper titled 'Advances in Instance Selection for 
 * Instance-Based Learning Algorithms' by Henry Brighton and Chris Mellish in 
 * DAMI in 2002. It performs an initial ENN pass, followed by iterative 
 * filtering based on the notions of coverage and reachability.
 * 
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class IterativeCaseFiltering extends InstanceSelector
        implements NSFUserInterface {
    
    public static final int DEFAULT_NEIGHBORHOOD_SIZE = 1;
    // Object that holds the kNN sets.
    private NeighborSetFinder nsf;
    // The upper triangular distance matrix on the data.
    private float[][] distMat;
    // Hit-Miss network on the data, used for filtering purposes.
    private List<HitMissNetwork> hmNetworks;
    // Mapping the indexes in the iterations to the original indexes.
    private List<HashMap<Integer, Integer>> protoIndexMaps;
    // ENN reducer, used for the initial pass.
    private Wilson72 internalReducer;
    // One of the stopping criteria.
    private int minNumPrototypes;
    
    /**
     * Default constructor.
     */
    public IterativeCaseFiltering() {
    }

    /**
     * Initialization.
     *
     * @param nsf Neighbor set finder object with some existing kNN info.
     */
    public IterativeCaseFiltering(NeighborSetFinder nsf) {
        this.nsf = nsf;
        if (nsf == null) {
            throw new IllegalArgumentException("Null kNN object provided.");
        }
        setOriginalDataSet(nsf.getDataSet());
        this.distMat = nsf.getDistances();
    }
    
    /**
     * This recursive method performs the iterative filtering.
     * @param dset DataSet to filter.
     * @param dMat float[][] that is the upper triangular distance matrix.
     * @param hmn HitMissNetwork corresponding to dset.
     * @param protoIndexMap HashMap<Integer, Integer> mapping to the original 
     * @return ArrayList<Integer> that is the list of prototype  indexes to add 
     * to.
     * index values. 
    */
    private ArrayList<Integer> filterCases(DataSet dset, float[][] dMat,
            HitMissNetwork hmn, HashMap<Integer, Integer> protoIndexMap)
            throws Exception {
        int currentSize = dset.size();
        int minIndex, maxIndex;
        float[] nearestEnemyDist = new float[currentSize];
        for (int i = 0; i < currentSize; i++) {
            minIndex = Math.min(i, hmn.getKnMisses()[i][0]);
            maxIndex = Math.max(i, hmn.getKnMisses()[i][0]);
            nearestEnemyDist[i] = dMat[minIndex][maxIndex - minIndex - 1];
        }
        int[] coverageArr = new int[currentSize];
        int[] reachabilityArr = new int[currentSize];
        for (int i = 0; i < dMat.length; i++) {
            for (int j = 0; j < dMat[i].length; j++) {
                if (dMat[i][j] < nearestEnemyDist[i]) {
                    reachabilityArr[i]++;
                    coverageArr[i + j + 1]++;
                }
                if (dMat[i][j] < nearestEnemyDist[i + j + 1]) {
                    reachabilityArr[i + j + 1]++;
                    coverageArr[i]++;
                }
            }
        }
        HashMap<Integer, Integer> protoIndexMapNew =
                new HashMap<>(currentSize);
        ArrayList<Integer> retainedIndexes = new ArrayList<>(currentSize);
        for (int i = 0; i < currentSize; i++) {
            if (coverageArr[i] >= reachabilityArr[i]) {
                // In this case the sample is retained.
                protoIndexMapNew.put(retainedIndexes.size(),
                        protoIndexMap.get(i));
                retainedIndexes.add(i);
            }
        }
        DataSet reducedSet = dset.getSubsample(retainedIndexes);
        if (retainedIndexes.size() < minNumPrototypes ||
                retainedIndexes.size() == currentSize ||
                reducedSet.countCategories() < 2) {
            ArrayList<Integer> protoIndexes = new ArrayList<>();
            for (int i = 0; i < retainedIndexes.size(); i++) {
                protoIndexes.add(protoIndexMap.get(retainedIndexes.get(i)));
            }
            return protoIndexes;
        }
        protoIndexMaps.add(protoIndexMapNew);
        // Make the labels conform to a range if a category is missing.
        reducedSet = reducedSet.copy();
        reducedSet.standardizeCategories();
        float[][] reducedDistMat = new float[reducedSet.size()][];
        for (int i = 0; i < reducedDistMat.length; i++) {
            reducedDistMat[i] = new float[reducedDistMat.length - i - 1];
            for (int j = 0; j < reducedDistMat[i].length; j++) {
                minIndex = Math.min(retainedIndexes.get(i),
                        retainedIndexes.get(i + j + 1));
                maxIndex = Math.max(retainedIndexes.get(i),
                        retainedIndexes.get(i + j + 1));
                reducedDistMat[i][j] = dMat[minIndex][maxIndex - minIndex - 1];
            }
        }
        HitMissNetwork reducedHMNetwork = new HitMissNetwork(reducedSet,
                reducedDistMat, 1);
        reducedHMNetwork.generateNetwork();
        hmNetworks.add(reducedHMNetwork);
        return filterCases(reducedSet, reducedDistMat, reducedHMNetwork,
                protoIndexMapNew);
    }
    
    @Override
    public void reduceDataSet() throws Exception {
        DataSet originalDataSet = getOriginalDataSet();
        // Initialization.
        int numClasses = getNumClasses();
        // First obtain the superset by performing ENN.
        internalReducer = new Wilson72(nsf);
        internalReducer.reduceDataSet();
        ArrayList<Integer> superProtoIndexes =
                internalReducer.getPrototypeIndexes();
        DataSet superSet = originalDataSet.getSubsample(superProtoIndexes);
        if (superSet.countCategories() < 2) {
            // The approach can not be applied, so we return the ENN results
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
                superDistMat, 1);
        superHMNetwork.generateNetwork();
        hmNetworks.add(superHMNetwork);
        protoIndexMaps = new ArrayList<>();
        HashMap<Integer, Integer> protoIndexMap =
                new HashMap<>(superProtoIndexes.size());
        for (int i = 0; i < superProtoIndexes.size(); i++) {
            protoIndexMap.put(i, superProtoIndexes.get(i));
        }
        protoIndexMaps.add(protoIndexMap);
        minNumPrototypes = Math.min(superSet.size() / 3,
                numClasses * nsf.getCurrK());
        ArrayList<Integer> protoIndexes = filterCases(superSet, superDistMat,
                superHMNetwork, protoIndexMap);
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
    public Publication getPublicationInfo() {
        JournalPublication pub = new JournalPublication();
        pub.setTitle("Advances in Instance Selection for Instance-Based "
                + "Learning Algorithms");
        pub.addAuthor(new Author("Henry", "Brighton"));
        pub.addAuthor(new Author("Chris", "Mellish"));
        pub.setJournalName("Data Mining and Knowledge Discovery");
        pub.setYear(2002);
        pub.setVolume(6);
        pub.setIssue(2);
        pub.setStartPage(153);
        pub.setEndPage(172);
        pub.setDoi("10.1023/A:1014043630878");
        pub.setUrl("http://dx.doi.org/10.1023/A:1014043630878");
        pub.setPublisher(Publisher.KLUWER);
        return pub;
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
        return new IterativeCaseFiltering(nsf);
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
