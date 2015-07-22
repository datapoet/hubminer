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
 * A class that implements the condensed nearest neighbor rule for instance
 * selection.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class CNN extends InstanceSelector implements NSFUserInterface {

    private float[][] distMat;
    private NeighborSetFinder nsf;
    // Arrays containing distances to nearest 'friends' and 'enemies' based on
    // matched/mismatched data labels, for all data points.
    private float[] nearestProtoFriendDist;
    private float[] nearestProtoEnemyDist;
    
    @Override
    public Publication getPublicationInfo() {
        JournalPublication pub = new JournalPublication();
        pub.setTitle("The Condensed Nearest Neighbor Rule");
        pub.addAuthor(new Author("P. E.", "Hart"));
        pub.setPublisher(Publisher.IEEE);
        pub.setJournalName("IEEE Transactions on Information Theory");
        pub.setYear(1968);
        pub.setStartPage(515);
        pub.setEndPage(516);
        pub.setVolume(14);
        return pub;
    }

    public CNN() {
    }

    /**
     * @param nsf NeighborSetFinder object.
     */
    public CNN(NeighborSetFinder nsf) {
        this.nsf = nsf;
        setOriginalDataSet(nsf.getDataSet());
        this.distMat = nsf.getDistances();
    }

    /**
     * @param distMat Upper diagonal distance matrix, each row contains only
     * distances d(i,j) for j > i. Therefore the length of the i-th row is
     * len(i_th_row) = n - i - 1
     */
    public CNN(float[][] distMat) {
        this.distMat = distMat;
    }

    /**
     * @param originalDataSet Original data set, prior to reduction.
     * @param nsf NeighborSetFinder object.
     */
    public CNN(DataSet originalDataSet, NeighborSetFinder nsf) {
        this.nsf = nsf;
        this.distMat = nsf.getDistances();
        setOriginalDataSet(originalDataSet);
    }

    /**
     * @param originalDataSet Original data set, prior to reduction.
     * @param distMat Upper diagonal distance matrix, each row contains only
     * distances d(i,j) for j > i. Therefore the length of the i-th row is
     * len(i_th_row) = n - i - 1
     */
    public CNN(DataSet original, float[][] distMat) {
        setOriginalDataSet(original);
        this.distMat = distMat;
    }

    @Override
    public void calculatePrototypeHubness(int k) throws Exception {
        if (nsf != null) {
            // We exploit the information that is available in the nsf.
            this.setNeighborhoodSize(k);
            if (k <= 0) {
                return;
            }
            DataSet originalDataSet = getOriginalDataSet();
            int[][] kns = nsf.getKNeighbors();
            float[][] kd = nsf.getKDistances();
            int[][] kneighbors = new int[originalDataSet.size()][k];
            int kNSF = kns[0].length;
            HashMap<Integer, Integer> protoMap =
                    new HashMap<>(getPrototypeIndexes().size() * 2);
            ArrayList<Integer> protoIndexes = getPrototypeIndexes();
            for (int i = 0; i < protoIndexes.size(); i++) {
                protoMap.put(protoIndexes.get(i), i);
            }
            int l;
            int datasize = originalDataSet.size();
            float[][] kdistances = new float[datasize][k];
            int[] kcurrLen = new int[datasize];
            float[][] distMatrix = nsf.getDistances();
            // Auxiliary array for fast restricted kNN search.
            ArrayList<Integer> intervals;
            int upper, lower;
            int min, max;
            for (int i = 0; i < originalDataSet.size(); i++) {
                intervals = new ArrayList(k + 2);
                intervals.add(-1);
                for (int j = 0; j < kNSF; j++) {
                    if (protoMap.containsKey(kns[i][j])) {
                        kneighbors[i][kcurrLen[i]] = protoMap.get(kns[i][j]);
                        kdistances[i][kcurrLen[i]] = kd[i][j];
                        kcurrLen[i]++;
                        intervals.add(kns[i][j]);
                    }
                    if (kcurrLen[i] >= k) {
                        break;
                    }
                }
                intervals.add(datasize + 1);
                Collections.sort(intervals);
                if (kcurrLen[i] < k) {
                    int iSizeRed = intervals.size() - 1;
                    // The loop needs to iterate one less than to the end, as
                    // the last limitation is there, so there are no elements
                    // beyond.
                    for (int ind = 0; ind < iSizeRed; ind++) {
                        lower = intervals.get(ind);
                        upper = intervals.get(ind + 1);
                        for (int j = lower + 1; j < upper - 1; j++) {
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
                                            while ((l >= 1) && distMatrix[min][
                                                    max - min - 1] < kdistances[
                                                    i][l - 1]) {
                                                kdistances[i][l] =
                                                        kdistances[i][l - 1];
                                                kneighbors[i][l] = kneighbors[
                                                        i][l - 1];
                                                l--;
                                            }
                                            kdistances[i][l] =
                                                    distMatrix[min][
                                                            max - min - 1];
                                            kneighbors[i][l] =
                                                    protoMap.get(j);
                                        }
                                    } else {
                                        if (distMatrix[min][max - min - 1]
                                                < kdistances[i][
                                                        kcurrLen[i] - 1]) {
                                            // Search to see where to insert.
                                            l = kcurrLen[i] - 1;
                                            kdistances[i][kcurrLen[i]] =
                                                    kdistances[i][
                                                            kcurrLen[i] - 1];
                                            kneighbors[i][kcurrLen[i]] =
                                                    kneighbors[i][
                                                            kcurrLen[i] - 1];
                                            while ((l >= 1) && distMatrix[min][
                                                    max - min - 1] < kdistances[
                                                            i][l - 1]) {
                                                kdistances[i][l] =
                                                        kdistances[i][l - 1];
                                                kneighbors[i][l] =
                                                        kneighbors[i][l - 1];
                                                l--;
                                            }
                                            kdistances[i][l] = distMatrix[min][
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
            // Initialize hubness-related data arrays.
            int numClasses = getNumClasses();
            int[] protoHubness = new int[protoIndexes.size()];
            int[] protoGoodHubness = new int[protoIndexes.size()];
            int[] protoBadHubness = new int[protoIndexes.size()];
            int[][] protoClassHubness =
                    new int[numClasses][protoIndexes.size()];
            setPrototypeHubness(protoHubness);
            setPrototypeGoodHubness(protoGoodHubness);
            setPrototypeBadHubness(protoBadHubness);
            setProtoClassHubness(protoClassHubness);
            int currLabel;
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
            setNeighborhoodSize(k);
            DataSet original = getOriginalDataSet();
            ArrayList<Integer> prototypeIndexes = getPrototypeIndexes();
            int[][] kneighbors = new int[original.size()][k];
            // Make a data subset.
            DataSet tCol = original.cloneDefinition();
            tCol.data = new ArrayList<>(prototypeIndexes.size());
            for (int index : prototypeIndexes) {
                tCol.data.add(original.getInstance(index));
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
            CombinedMetric cmet = getCombinedMetric();
            for (int i = 0; i < original.size(); i++) {
                currLabel = original.getLabelOf(i);
                kneighbors[i] = NeighborSetFinder.getIndexesOfNeighbors(
                        tCol,
                        original.getInstance(i),
                        k,
                        cmet);
                for (int nIndex : kneighbors[i]) {
                    protoClassHubness[currLabel][nIndex]++;
                    protoHubness[nIndex]++;
                    protoLabel = original.getLabelOf(nIndex);
                    if (protoLabel == currLabel) {
                        protoGoodHubness[nIndex]++;
                    } else {
                        protoBadHubness[nIndex]++;
                    }
                }
            }
            setProtoNeighborSets(kneighbors);
        }
    }

    @Override
    public InstanceSelector copy() {
        if (nsf == null) {
            return new CNN(getOriginalDataSet(), distMat);
        } else {
            return new CNN(nsf);
        }
    }

    @Override
    public void reduceDataSet() throws Exception {
        DataSet originalDataSet = getOriginalDataSet();
        int datasize = originalDataSet.size();
        int numClasses = getNumClasses();
        ArrayList<Integer>[] classes = new ArrayList[numClasses];
        for (int c = 0; c < numClasses; c++) {
            classes[c] = new ArrayList<>((datasize * 3) / numClasses);
        }
        int currClass, protoClass;
        for (int i = 0; i < datasize; i++) {
            currClass = originalDataSet.getLabelOf(i);
            classes[currClass].add(i);
        }
        Random randa = new Random();
        ArrayList<Integer> pIndexes =
                new ArrayList<>(originalDataSet.size() / 4);
        HashMap<Integer, Integer> protoMap = new HashMap<>(datasize);
        int choice;
        for (int c = 0; c < numClasses; c++) {
            if (classes[c].size() > 0) {
                choice = randa.nextInt(classes[c].size());
                pIndexes.add(classes[c].get(choice));
                protoMap.put(classes[c].get(choice), c);
            }
        }
        // The first run is to get the initial nearest friend and enemy values.
        int min, max;
        nearestProtoFriendDist = new float[datasize];
        nearestProtoEnemyDist = new float[datasize];
        Arrays.fill(nearestProtoFriendDist, Float.MAX_VALUE);
        Arrays.fill(nearestProtoEnemyDist, Float.MAX_VALUE);
        for (int i = 0; i < datasize; i++) {
            currClass = originalDataSet.getLabelOf(i);
            if (!protoMap.containsKey(i)) {
                for (int pIndex : pIndexes) {
                    protoClass = originalDataSet.getLabelOf(pIndex);
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
        boolean notDone = true;
        ArrayList<Integer> unAbsorbed;
        while (notDone) {
            notDone = false;
            unAbsorbed = new ArrayList<>(
                    Math.max((datasize - pIndexes.size()) / 3, 5));
            for (int i = 0; i < datasize; i++) {
                if (!protoMap.containsKey(i)) {
                    if (!((nearestProtoEnemyDist[i]
                            - nearestProtoFriendDist[i]) > 0)) {
                        // Not absorbed.
                        notDone = true;
                        unAbsorbed.add(i);
                    }
                }
            }
            if (unAbsorbed.size() > 0) {
                choice = unAbsorbed.get(randa.nextInt(unAbsorbed.size()));
                protoMap.put(choice, originalDataSet.getLabelOf(choice));
                pIndexes.add(choice);
                protoClass = originalDataSet.getLabelOf(choice);
                for (int i = 0; i < datasize; i++) {
                    if (!protoMap.containsKey(i)) {
                        min = Math.min(choice, i);
                        max = Math.max(choice, i);
                        currClass = originalDataSet.getLabelOf(i);
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
        // CNN determines the proper number automatically, so a fixed number of
        // desired prototypes is not explicitly supported. A random subset of
        // the selected CNN set might be taken here, though a question remains
        // then how to handle the case where numPrototypes > numSelectedCNN. For
        // now, this method is a dummy, as it executes the original automated
        // prototype selection method instead.
        reduceDataSet();
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
