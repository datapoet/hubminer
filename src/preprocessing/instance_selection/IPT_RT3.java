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
import algref.Publisher;
import data.neighbors.NeighborSetFinder;
import data.representation.DataSet;
import distances.primary.CombinedMetric;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import data.neighbors.NSFUserInterface;
import util.AuxSort;

/**
 * RT3 algorithm proposed in "Instance Pruning Techniques", 1997, a highly cited
 * paper on the topic. First, Wilson's rule is applied to k+1 neighbor sets.
 * Second, the instances are re-ordered by the distance to the 'nearest enemy'.
 * Third, all points whose removal would reduce the misclassification rate are
 * removed, one by one.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class IPT_RT3 extends InstanceSelector implements NSFUserInterface {

    private NeighborSetFinder nsf;
    // Neighborhood size to use in selection criteria.
    private int kSelection = 1;
    
    @Override
    public Publication getPublicationInfo() {
        ConferencePublication pub = new ConferencePublication();
        pub.setTitle("Instance Pruning Techniques");
        pub.addAuthor(new Author("D. Randall", "Wilson"));
        pub.addAuthor(new Author("Tony R.", "Martinez"));
        pub.setConferenceName("International Conference on Machine Learning");
        pub.setYear(1997);
        pub.setStartPage(404);
        pub.setEndPage(411);
        pub.setPublisher(Publisher.MORGAN_KAUFMANN);
        return pub;
    }

    public IPT_RT3() {
    }

    /**
     * @param kSelection Neighborhood size to use in selection criteria.
     */
    public IPT_RT3(int kSelection) {
        this.kSelection = kSelection;
    }

    /**
     * @param nsf NeighborSetFinder object.
     */
    public IPT_RT3(NeighborSetFinder nsf) {
        setOriginalDataSet(nsf.getDataSet());
        this.nsf = nsf;
        kSelection = nsf.getKNeighbors()[0].length;
    }

    /**
     * @param kSelection Neighborhood size to use in selection criteria.
     * @param cmet CombinedMetric object.
     */
    public IPT_RT3(int kSelection, CombinedMetric cmet) {
        this.kSelection = kSelection;
        setCombinedMetric(cmet);
    }

    /**
     * @param nsf NeighborSetFinder object.
     * @param cmet CombinedMetric object.
     */
    public IPT_RT3(NeighborSetFinder nsf, CombinedMetric cmet) {
        setOriginalDataSet(nsf.getDataSet());
        this.nsf = nsf;
        kSelection = nsf.getKNeighbors()[0].length;
        setCombinedMetric(cmet);
    }

    @Override
    public void calculatePrototypeHubness(int k) throws Exception {
        this.setNeighborhoodSize(k);
        if (k <= 0) {
            return;
        }
        DataSet originalDataSet = getOriginalDataSet();
        // Original neighbor sets and neighbor distances.
        int[][] kns = nsf.getKNeighbors();
        float[][] kd = nsf.getKDistances();
        // Neighbor sets with prototypes as neighbors.
        int[][] kneighbors = new int[originalDataSet.size()][k];
        int kNSF = kns[0].length;
        HashMap<Integer, Integer> protoMap =
                new HashMap<>(getPrototypeIndexes().size() * 2);
        ArrayList<Integer> protoIndexes = getPrototypeIndexes();
        for (int i = 0; i < protoIndexes.size(); i++) {
            protoMap.put(protoIndexes.get(i), i);
        }
        int l = 0;
        int datasize = originalDataSet.size();
        float[][] kdistances = new float[datasize][k];
        int[] kcurrLen = new int[datasize];
        float[][] distMatrix = nsf.getDistances();
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
                                            < kdistances[i][kcurrLen[i] - 1]) {
                                        // Search to see where to insert.
                                        l = k - 1;
                                        while ((l >= 1) && distMatrix[min][
                                                max - min - 1]
                                                < kdistances[i][l - 1]) {
                                            kdistances[i][l] =
                                                    kdistances[i][l - 1];
                                            kneighbors[i][l] =
                                                    kneighbors[i][l - 1];
                                            l--;
                                        }
                                        kdistances[i][l] =
                                                distMatrix[min][max - min - 1];
                                        kneighbors[i][l] = protoMap.get(j);
                                    }
                                } else {
                                    if (distMatrix[min][max - min - 1]
                                            < kdistances[i][kcurrLen[i] - 1]) {
                                        // Search to see where to insert.
                                        l = kcurrLen[i] - 1;
                                        kdistances[i][kcurrLen[i]] =
                                                kdistances[i][kcurrLen[i] - 1];
                                        kneighbors[i][kcurrLen[i]] =
                                                kneighbors[i][kcurrLen[i] - 1];
                                        while ((l >= 1) && distMatrix[min][
                                                max - min - 1]
                                                < kdistances[i][l - 1]) {
                                            kdistances[i][l] =
                                                    kdistances[i][l - 1];
                                            kneighbors[i][l] =
                                                    kneighbors[i][l - 1];
                                            l--;
                                        }
                                        kdistances[i][l] =
                                                distMatrix[min][max - min - 1];
                                        kneighbors[i][l] = protoMap.get(j);
                                        kcurrLen[i]++;
                                    } else {
                                        kdistances[i][kcurrLen[i]] =
                                                distMatrix[min][max - min - 1];
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
        // Prototype occurrence frequency array.
        int[] protoHubness = new int[protoIndexes.size()];
        // Prototype good occurrence frequency array.
        int[] protoGoodHubness = new int[protoIndexes.size()];
        // Prototype detrimental occurrence frequency array.
        int[] protoBadHubness = new int[protoIndexes.size()];
        // Prototype class-conditional neighbor occurrence frequencies.
        int[][] protoClassHubness = new int[numClasses][protoIndexes.size()];
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
    }

    @Override
    public void noRecalcs() {
    }

    @Override
    public void setNSF(NeighborSetFinder nsf) {
        this.nsf = nsf;
        kSelection = nsf.getKNeighbors()[0].length;
    }

    @Override
    public NeighborSetFinder getNSF() {
        return nsf;
    }

    @Override
    public InstanceSelector copy() {
        CombinedMetric cmet = getCombinedMetric();
        if (nsf == null) {
            return new IPT_RT3(kSelection, cmet);
        } else {
            return new IPT_RT3(nsf, cmet);
        }
    }

    @Override
    public void reduceDataSet() throws Exception {
        // Re-use the kNN set information or calculate it if not available.
        if (nsf == null) {
            nsf = new NeighborSetFinder(getOriginalDataSet(),
                    CombinedMetric.FLOAT_MANHATTAN);
            nsf.calculateDistances();
            nsf.calculateNeighborSets(kSelection + 1);
        } else {
            NeighborSetFinder nsfCopy = nsf;
            nsf = nsfCopy.copy();
            nsf.completeNeighborSets(kSelection + 1, null);
        }
        DataSet originalDataSet = getOriginalDataSet();
        int[][] kneighbors = nsf.getKNeighbors();
               

        CombinedMetric cmet = getCombinedMetric();
        // First we apply the Wilson's rule.
        Wilson72 wilReducer = new Wilson72(nsf, cmet);
        wilReducer.reduceDataSet();
        ArrayList<Integer> pIndexes = wilReducer.getPrototypeIndexes();
        // Make a map of the prototypes selected by the Wilson's rule.
        HashMap<Integer, Integer> protoMap = new HashMap<>(pIndexes.size() * 2);
        for (int i = 0; i < pIndexes.size(); i++) {
            protoMap.put(pIndexes.get(i), i);
        }
        HashMap<Integer, Integer> tabuList =
                new HashMap<>(originalDataSet.size());
        // At least 40, otherwise 2 * kSelection, unless Wilson selected less.
        int[] top2K = new int[Math.min(pIndexes.size(), Math.min(
                originalDataSet.size(), Math.max(2 * kSelection, 40)))];
        float[] top2KCriticality = new float[top2K.length];
        Arrays.fill(top2KCriticality, -Float.MAX_VALUE);
        // From this point on, only the prototypes selected by the Wilson's rule
        // are permitted as neighbors, so neighbor sets are recalculated by
        // tabu-ing the points that are not prototypes.
        for (int i = 0; i < originalDataSet.size(); i++) {
            if (!protoMap.containsKey(i)) {
                nsf.tabuANeighbor(i, i, tabuList, false);
                tabuList.put(i, i);
            }
        }
        // Re-calculate the occurrence model.
        nsf.calculateHubnessStats();
        int currClass;
        int nLabel;
        int numClasses = getNumClasses();
        int[][] kNNclassCounts = new int[originalDataSet.size()][numClasses];
        float[] nearestEnemyDistances = new float[originalDataSet.size()];

        // The first number will be the number of reverse nearest neighbors
        // where the removal would induce misclassification. The second number
        // will be the number of reverse nearest neighbors where the removal
        // would induce correct classification. The unaffected neighbor sets are
        // not counted.
        int criticalPositive;
        int criticalNegative;

        float[] classPriors = originalDataSet.getClassPriors();
        int classification;

        int criticalClass;
        boolean uniqueCritical;
        boolean correctlyClassified;
        boolean removalCriticalPositive;

        Arrays.fill(nearestEnemyDistances, Float.MAX_VALUE);

        float[][] kdistances = nsf.getKDistances();

        for (int i = 0; i < originalDataSet.size(); i++) {
            currClass = originalDataSet.getLabelOf(i);
            for (int j = 0; j < kSelection + 1; j++) {
                nLabel = originalDataSet.getLabelOf(kneighbors[i][j]);
                if (currClass != nLabel
                        && nearestEnemyDistances[i] == Float.MAX_VALUE) {
                    nearestEnemyDistances[i] = kdistances[i][j];
                }
                kNNclassCounts[i][nLabel]++;
            }
        }

        ArrayList<Integer> protoIndexes = new ArrayList<>(pIndexes.size());


        int[] reArrIndexes =
                AuxSort.sortIndexedValue(nearestEnemyDistances, true);
        ArrayList<Integer> reverseNeighbors;
        int[] hubnessArray = nsf.getNeighborFrequencies();
        int consideredClass;
        for (int i : reArrIndexes) {
            criticalPositive = 0;
            criticalNegative = 0;
            consideredClass = originalDataSet.getLabelOf(i);
            if (protoMap.containsKey(i)) {
                // If the instance is among those selected by the Wilson's
                // criterion, calculate its criticalPositive - criticalNegative
                // score and tabu/remove if necessary.
                if (hubnessArray[i] == 0) {
                    nsf.tabuANeighbor(i, 0, tabuList, false);
                }
                reverseNeighbors = nsf.getReverseNeighbors()[i];
                for (int index : reverseNeighbors) {
                    currClass = originalDataSet.getLabelOf(index);
                    classification = 0;
                    for (int c = 0; c < numClasses; c++) {
                        if (kNNclassCounts[index][c]
                                > kNNclassCounts[index][classification]) {
                            classification = c;
                        } else if (kNNclassCounts[index][c] ==
                                kNNclassCounts[index][classification] &&
                                classPriors[c] > classPriors[classification]) {
                            classification = c;
                        }
                    }
                    if (classification == currClass) {
                        correctlyClassified = true;
                    } else {
                        correctlyClassified = false;
                    }
                    removalCriticalPositive = false;
                    criticalClass = -1;
                    uniqueCritical = true;
                    for (int c = 0; c < numClasses; c++) {
                        if (correctlyClassified && (c != currClass)) {
                            if ((classPriors[c] > classPriors[currClass])
                                    && (kNNclassCounts[index][c] + 1
                                    == kNNclassCounts[index][currClass])) {
                                removalCriticalPositive = true;
                            } else if ((kNNclassCounts[c]
                                    == kNNclassCounts[currClass])) {
                                removalCriticalPositive = true;
                            }
                        } else if (!correctlyClassified && c != currClass) {
                            if ((kNNclassCounts[c] == kNNclassCounts[currClass])
                                    && (classPriors[c] > classPriors[currClass])) {
                                if (criticalClass != -1) {
                                    uniqueCritical = false;
                                    criticalClass = c;
                                } else {
                                    criticalClass = c;
                                }
                            } else if ((kNNclassCounts[index][c] - 1
                                    == kNNclassCounts[index][currClass])
                                    && (classPriors[c]
                                    <= classPriors[currClass])) {
                                if (criticalClass != -1) {
                                    uniqueCritical = false;
                                    criticalClass = c;
                                } else {
                                    criticalClass = c;
                                }
                            }
                        }
                    }
                    if (correctlyClassified && (consideredClass == currClass)
                            && removalCriticalPositive) {
                        criticalPositive++;
                    } else if (!correctlyClassified
                            && (consideredClass != currClass)) {
                        if (uniqueCritical
                                && consideredClass == criticalClass) {
                            criticalNegative++;
                        }
                    }
                }
                if (criticalPositive <= criticalNegative) {
                    ArrayList<Integer> rnnArrCopy =
                            new ArrayList<>(reverseNeighbors.size());
                    for (int p = 0; p < reverseNeighbors.size(); p++) {
                        rnnArrCopy.add(reverseNeighbors.get(p));
                    }
                    nsf.tabuANeighbor(i, 0, tabuList, false);
                    for (int index : rnnArrCopy) {
                        kNNclassCounts[index][consideredClass]--;
                        kNNclassCounts[index][originalDataSet.getLabelOf(
                                kneighbors[index][kSelection])]++;
                        // The array is a reference so it automatically gets
                        // access to the updated k-neighbor sets.
                    }
                } else {
                    protoIndexes.add(i);
                }
                float positivity = criticalPositive - criticalNegative;
                int lastIndex = top2KCriticality.length - 1;
                while (lastIndex >= 0 && positivity
                        > top2KCriticality[lastIndex]) {
                    lastIndex--;
                }
                if (lastIndex < (top2KCriticality.length - 1)) {
                    // Insertion.
                    for (int mIndex = (top2KCriticality.length - 1);
                            mIndex > lastIndex + 1; mIndex--) {
                        top2K[mIndex] = top2K[mIndex - 1];
                        top2KCriticality[mIndex] = top2KCriticality[mIndex - 1];
                    }
                    top2K[lastIndex + 1] = i;
                    top2KCriticality[lastIndex + 1] = positivity;
                }
            }
        }
        if (protoIndexes.size() < top2K.length) {
            protoIndexes = new ArrayList<>(pIndexes.size());
            for (int index : top2K) {
                protoIndexes.add(index);
            }
        }
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
        setPrototypeIndexes(protoIndexes);
        sortSelectedIndexes();
    }

    @Override
    public void reduceDataSet(int numPrototypes) throws Exception {
        reduceDataSet();
        // This class has no way of controlling how much is retained.
    }
}
