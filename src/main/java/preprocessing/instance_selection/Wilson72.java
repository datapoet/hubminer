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
import data.neighbors.NSFUserInterface;
import data.representation.DataSet;
import data.neighbors.NeighborSetFinder;
import java.util.ArrayList;
import distances.primary.CombinedMetric;
import java.util.HashMap;
import java.util.Collections;

/**
 * This class implements an old baseline algorithm described in the paper:
 * Asymptotic Properties of Nearest Neighbor Rules Using Edited Data.
 * Essentially, only those instances that agree with the majority of their
 * nearest neighbors are retained.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class Wilson72 extends InstanceSelector implements NSFUserInterface {

    private NeighborSetFinder nsf;
    // Neighborhood size to be used in selection criteria.
    private int kSelection = 1;
    
    @Override
    public Publication getPublicationInfo() {
        JournalPublication pub = new JournalPublication();
        pub.setTitle("Asymptotic Properties of Nearest Neighbor Rules Using "
                + "Edited Data");
        pub.addAuthor(new Author("D. R.", "Wilson"));
        pub.setPublisher(Publisher.IEEE);
        pub.setJournalName("IEEE Transactions on Systems, Man and Cybernetics");
        pub.setYear(1972);
        pub.setStartPage(408);
        pub.setEndPage(421);
        pub.setVolume(2);
        return pub;
    }

    public Wilson72() {
    }

    /**
     * @param kSelection Neighborhood size to be used in selection criteria.
     */
    public Wilson72(int kSelection) {
        this.kSelection = kSelection;
    }

    /**
     * @param nsf NeighborSetFinder object.
     */
    public Wilson72(NeighborSetFinder nsf) {
        setOriginalDataSet(nsf.getDataSet());
        this.nsf = nsf;
        kSelection = nsf.getKNeighbors()[0].length;
    }

    /**
     * @param kSelection Neighborhood size to be used in selection criteria.
     * @param cmet CombinedMetric object.
     */
    public Wilson72(int kSelection, CombinedMetric cmet) {
        this.kSelection = kSelection;
        setCombinedMetric(cmet);
    }

    /**
     * @param nsf NeighborSetFinder object.
     * @param cmet CombinedMetric object.
     */
    public Wilson72(NeighborSetFinder nsf, CombinedMetric cmet) {
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
    public void reduceDataSet() throws Exception {
        if (nsf == null) {
            nsf = new NeighborSetFinder(getOriginalDataSet(),
                    CombinedMetric.FLOAT_MANHATTAN);
            nsf.calculateDistances();
            nsf.calculateNeighborSets(kSelection);
        }
        DataSet originalDataSet = getOriginalDataSet();
        int[][] kneighbors = nsf.getKNeighbors();
        int currClass;
        int currClassCount;
        int threshold = kSelection / 2 + 1;
        int[] classCounts = new int[originalDataSet.countCategories()];
        ArrayList<Integer> pIndexes = new ArrayList<>(originalDataSet.size());
        for (int i = 0; i < originalDataSet.size(); i++) {
            currClass = originalDataSet.getLabelOf(i);
            currClassCount = 0;
            for (int j = 0; j < kSelection; j++) {
                if (originalDataSet.getLabelOf(kneighbors[i][j])
                        == currClass) {
                    currClassCount++;
                }
            }
            if (currClassCount >= threshold || classCounts[currClass] == 0) {
                pIndexes.add(i);
                classCounts[currClass]++;
            }
        }
        setPrototypeIndexes(pIndexes);
        sortSelectedIndexes();
    }

    @Override
    public void reduceDataSet(int numPrototypes) throws Exception {
        reduceDataSet();
        // This class has no way of controlling how much is retained.
    }

    @Override
    public InstanceSelector copy() {
        CombinedMetric cmet = getCombinedMetric();
        if (nsf == null) {
            return new Wilson72(kSelection, cmet);
        } else {
            return new Wilson72(nsf, cmet);
        }
    }
}
