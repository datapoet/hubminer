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

import algref.Publication;
import data.neighbors.NSFUserInterface;
import data.neighbors.NeighborSetFinder;
import data.representation.DataSet;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import learning.supervised.Category;
import sampling.UniformSampler;

/**
 * Random instance selection. This class implements the NSFUserInterface, which
 * might not seem logical at first - but the idea is to re-use the existing
 * kNN sets for unbiased neighbor occurrence modeling on the training data, when
 * kNN classifiers are present. When they are not, though - this might slow
 * things down a bit. However, there is also a technical catch. In general, it
 * is possible to run batch classifier testing in Hub Miner without actually
 * having the features and the metric implemented - by making dummy data files
 * and loading the pre-computed distance matrix. If the experimental framework
 * is run in that mode and unbiased hubness estimates are required, they are
 * impossible to calculate unless either the NeighborSetFinder object is
 * provided or the distance matrix itself, since the CombinedMetric objects are
 * either dummies themselves or they have no data to re-calculate the distances
 * from.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class RandomSelector extends InstanceSelector
        implements NSFUserInterface {
    
    // To use for calculating the unbiased hubness estimates.
    private NeighborSetFinder nsf;
    
    @Override
    public Publication getPublicationInfo() {
        // This is just stratified random selection. No publications associated 
        // with it.
        return new Publication();
    }

    /**
     * The default constructor.
     */
    public RandomSelector() {
    }

    /**
     * @param originalDataSet
     */
    public RandomSelector(DataSet originalDataSet) {
        setOriginalDataSet(originalDataSet);
    }
    
    @Override
    public void noRecalcs() {
    }

    @Override
    public void setNSF(NeighborSetFinder nsf) {
        this.nsf = nsf;
    }

    @Override
    public NeighborSetFinder getNSF() {
        return nsf;
    }
    
    /**
     * @param nsf NeighborSetFinder object.
     */
    public RandomSelector(NeighborSetFinder nsf) {
        setOriginalDataSet(nsf.getDataSet());
        this.nsf = nsf;
    }

    @Override
    public InstanceSelector copy() {
        if (nsf == null) {
            return new RandomSelector();
        } else {
            return new RandomSelector(nsf);
        }
    }

    @Override
    public void reduceDataSet(int numPrototypes) throws Exception {
        DataSet original = getOriginalDataSet();
        ArrayList<Integer> protoIndexes = new ArrayList<>(original.size());
        Category[] cats = original.getClassesArray(original.countCategories());
        float percRetained = (float) numPrototypes / (float) original.size();
        for (int cIndex = 0; cIndex < cats.length; cIndex++) {
            if (cats[cIndex] == null) {
                continue;
            }
            int catSize = cats[cIndex].size();
            int numClPrototypes = Math.max(1, (int)(percRetained * catSize));
            int[] sample = UniformSampler.getSample(
                    catSize, numClPrototypes);
            for (int i = 0; i < sample.length; i++) {
                protoIndexes.add(cats[cIndex].getIndex(sample[i]));
            }
        }
        setPrototypeIndexes(protoIndexes);
        sortSelectedIndexes();
    }

    @Override
    public void reduceDataSet() throws Exception {
        // There is no automatic way for determining the size of the reduced set
        // for random subsampling, so here we use a 20% retainment rate by
        // default.
        reduceDataSet(0.2f);
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
}
