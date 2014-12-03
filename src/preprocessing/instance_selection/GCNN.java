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
import algref.ConferencePublication;
import algref.Publication;
import data.neighbors.NeighborSetFinder;
import data.representation.DataSet;
import distances.primary.CombinedMetric;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Random;
import data.neighbors.NSFUserInterface;

/**
 * Implements the instance selection algorithm described in the paper: The
 * Generalized Condensed Nearest Neighbor Rule as A Data Reduction Method (2006)
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class GCNN extends InstanceSelector implements NSFUserInterface {

    // Absorption constant.
    private float ro = 0.99f; // This value was suggested by the authors in the
    // paper. We have determined ro = 0.75 to better fit high-dimensional data.
    private float[][] distMat;
    private NeighborSetFinder nsf;
    // Distances to nearest points from the same class.
    private float[] nearestProtoFriendDist;
    // Distances to nearest prototypes of different classes.
    private float[] nearestProtoEnemyDist;
    
    @Override
    public Publication getPublicationInfo() {
        ConferencePublication pub = new ConferencePublication();
        pub.setTitle("The Generalized Condensed Nearest Neighbor Rule as A Data"
                + " Reduction Method");
        pub.addAuthor(new Author("Chien-Hsing", "Chou"));
        pub.addAuthor(new Author("Bo-Han", "Kuo"));
        pub.addAuthor(new Author("Fu", "Chang"));
        pub.setConferenceName("International Conference on Pattern "
                + "Recognition");
        pub.setYear(2006);
        pub.setStartPage(556);
        pub.setEndPage(559);
        pub.setDoi("10.1109/ICPR.2006.1119");
        return pub;
    }

    /**
     * @constructor
     */
    public GCNN() {
    }

    /**
     * @param ro Absorption constant.
     * @constructor
     */
    public GCNN(float ro) {
        this.ro = ro;
    }

    /**
     * @param nsf Neighbor set finder object.
     * @constructor
     */
    public GCNN(NeighborSetFinder nsf) {
        this.nsf = nsf;
        setOriginalDataSet(nsf.getDataSet());
        this.distMat = nsf.getDistances();
    }

    /**
     * @param distMat Distance matrix, upper diagonal.
     * @constructor
     */
    public GCNN(float[][] distMat) {
        this.distMat = distMat;
    }

    /**
     * @param nsf Neighbor set finder object.
     * @param ro Absorption constant.
     * @constructor
     */
    public GCNN(NeighborSetFinder nsf, float ro) {
        this.nsf = nsf;
        this.distMat = nsf.getDistances();
        setOriginalDataSet(nsf.getDataSet());
        this.ro = ro;
    }

    /**
     * @param distMat Distance matrix, upper diagonal.
     * @param ro Absorption constant.
     * @constructor
     */
    public GCNN(float[][] distMat, float ro) {
        this.distMat = distMat;
        this.ro = ro;
    }

    /**
     * @param originalDataSet Original data set.
     * @param nsf Neighbor set finder object.
     * @param ro Absorption constant.
     * @constructor
     */
    public GCNN(DataSet originalDataSet, NeighborSetFinder nsf, float ro) {
        this.nsf = nsf;
        this.distMat = nsf.getDistances();
        setOriginalDataSet(originalDataSet);
        this.ro = ro;
    }

    /**
     * @param originalDataSet Original data set.
     * @param distMat Distance matrix, upper diagonal.
     * @param ro Absorption constant.
     * @constructor
     */
    public GCNN(DataSet originalDataSet, float[][] distMat, float ro) {
        setOriginalDataSet(originalDataSet);
        this.distMat = distMat;
        this.ro = ro;
    }

    /**
     * @param originalDataSet Original data set.
     * @param nsf Neighbor set finder object.
     * @constructor
     */
    public GCNN(DataSet originalDataSet, NeighborSetFinder nsf) {
        this.nsf = nsf;
        this.distMat = nsf.getDistances();
        setOriginalDataSet(originalDataSet);
    }

    /**
     * @param originalDataSet Original data set.
     * @param distMat Distance matrix, upper diagonal.
     * @constructor
     */
    public GCNN(DataSet originalDataSet, float[][] distMat) {
        setOriginalDataSet(originalDataSet);
        this.distMat = distMat;
    }

    /**
     * @param ro Absorption constant.
     */
    public void setRo(float ro) {
        this.ro = ro;
    }

    /**
     * @return Absorption constant.
     */
    public float getRo() {
        return ro;
    }

    @Override
    public InstanceSelector copy() {
        if (nsf == null) {
            return new GCNN(getOriginalDataSet(), distMat, ro);
        } else {
            return new GCNN(nsf, ro);
        }
    }

    @Override
    public void reduceDataSet() throws Exception {
        DataSet original = getOriginalDataSet();
        int datasize = original.size();
        // Initialization.
        int numClasses = getNumClasses();
        ArrayList<Integer>[] classes = new ArrayList[numClasses];
        for (int c = 0; c < numClasses; c++) {
            classes[c] = new ArrayList<>((datasize * 3) / numClasses);
        }
        int currClass, protoClass;
        for (int i = 0; i < datasize; i++) {
            currClass = original.getLabelOf(i);
            classes[currClass].add(i);
        }
        Random randa = new Random();
        // List that will contain the selected prototype indexes.
        ArrayList<Integer> pIndexes = new ArrayList<>(original.size() / 4);
        // Map that will indicate which points are prototypes.
        HashMap<Integer, Integer> protoMap = new HashMap<>(datasize);
        // Select a few points from each class randomly to represent an initial
        // prototype set that will be incrementally grown.
        int choice;
        for (int c = 0; c < numClasses; c++) {
            if (classes[c].size() > 0) {
                choice = randa.nextInt(classes[c].size());
                pIndexes.add(classes[c].get(choice));
                protoMap.put(classes[c].get(choice), c);
            }
        }
        // The first run is to get the initial nearest friend and enemy values.
        // In this context, friends have the same class label, enemies a
        // different one.
        int min, max;
        nearestProtoFriendDist = new float[datasize];
        nearestProtoEnemyDist = new float[datasize];
        Arrays.fill(nearestProtoFriendDist, Float.MAX_VALUE);
        Arrays.fill(nearestProtoEnemyDist, Float.MAX_VALUE);
        for (int i = 0; i < datasize; i++) {
            currClass = original.getLabelOf(i);
            if (!protoMap.containsKey(i)) {
                for (int pIndex : pIndexes) {
                    protoClass = original.getLabelOf(pIndex);
                    min = Math.min(i, pIndex);
                    max = Math.max(i, pIndex);
                    if (currClass == protoClass) {
                        if (distMat[min][max - min - 1]
                                < nearestProtoFriendDist[i]) {
                            nearestProtoFriendDist[i] =
                                    distMat[min][max - min - 1];
                        }
                    } else {
                        if (distMat[min][max - min - 1]
                                < nearestProtoEnemyDist[i]) {
                            nearestProtoEnemyDist[i] =
                                    distMat[min][max - min - 1];
                        }
                    }
                }
            }
        }
        // Now go through the distance matrix and find the minimal non-zero
        // heterogenous distance. The fastest way is not to check the intra-
        // class ones, so let's iterate through class-class pairs.
        int temp1, temp2;
        int c1size, c2size;
        float minHeterogenousDistance = Float.MAX_VALUE;
        for (int c1 = 0; c1 < numClasses; c1++) {
            for (int c2 = c1 + 1; c2 < numClasses; c2++) {
                c1size = classes[c1].size();
                c2size = classes[c2].size();
                if ((c1size > 0) && (c2size > 0)) {
                    for (int i = 0; i < c1size; i++) {
                        for (int j = 0; j < c2size; j++) {
                            temp1 = classes[c1].get(i);
                            temp2 = classes[c2].get(j);
                            min = Math.min(temp1, temp2);
                            max = Math.max(temp1, temp2);
                            if ((distMat[min][max - min - 1] > 0)
                                    && (distMat[min][max - min - 1]
                                    < minHeterogenousDistance)) {
                                minHeterogenousDistance =
                                        distMat[min][max - min - 1];
                            }
                        }
                    }
                }
            }
        }
        // Now perform instance selection with a strong absorption rule.
        boolean notDone = true;
        ArrayList<Integer> unAbsorbed;
        while (notDone) {
            notDone = false;
            unAbsorbed = new ArrayList<>(
                    Math.max((datasize - pIndexes.size()) / 3, 5));
            for (int i = 0; i < datasize; i++) {
                if (!protoMap.containsKey(i)) {
                    // If the enemy distance is not at least the minimum amount
                    // of separation grater than the friend distance - then
                    // this point is a candidate for prototype set extension.
                    if (!((nearestProtoEnemyDist[i] - nearestProtoFriendDist[i])
                            > (ro * minHeterogenousDistance))) { // Not absorbed
                        notDone = true;
                        unAbsorbed.add(i);
                    }
                }
            }
            // If there are prototype candidates, i.e. unabsorbed points.
            if (unAbsorbed.size() > 0) {
                // Pick a random one.
                choice = unAbsorbed.get(randa.nextInt(unAbsorbed.size()));
                protoMap.put(choice, original.getLabelOf(choice));
                pIndexes.add(choice);
                protoClass = original.getLabelOf(choice);
                // Update all the nearest friend and enemy distances.
                for (int i = 0; i < datasize; i++) {
                    if (!protoMap.containsKey(i)) {
                        min = Math.min(choice, i);
                        max = Math.max(choice, i);
                        currClass = original.getLabelOf(i);
                        if (currClass == protoClass) {
                            if (distMat[min][max - min - 1]
                                    < nearestProtoFriendDist[i]) {
                                nearestProtoFriendDist[i] =
                                        distMat[min][max - min - 1];
                            }
                        } else {
                            if (distMat[min][max - min - 1]
                                    < nearestProtoEnemyDist[i]) {
                                nearestProtoEnemyDist[i] =
                                        distMat[min][max - min - 1];
                            }
                        }
                    }
                }
            }
        }
        // Set the selected prototype indexes and sort them.
        setPrototypeIndexes(pIndexes);
        sortSelectedIndexes();
    }

    @Override
    public void reduceDataSet(int numPrototypes) throws Exception {
        // GCNN adaptively selects the proper number of prototypes. 
        // Just for the sake of consistency, we support this variant here and
        // we just return a subset of numPrototypes prototypes selected by GCNN.
        // Note that this is somewhat suboptimal and the reduceDataSet method
        // could be extended to support a cut-off criterion for a predefined
        // number of prototypes.
        reduceDataSet();
        ArrayList<Integer> pIndexes = getPrototypeIndexes();
        // The index list is sorted, so we shuffle it first to remove the bias.
        Collections.shuffle(pIndexes);
        ArrayList<Integer> pIndexesSubset = new ArrayList<>(
                pIndexes.subList(0, numPrototypes));
        setPrototypeIndexes(pIndexesSubset);
        sortSelectedIndexes();
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
}
