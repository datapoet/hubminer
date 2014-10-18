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

import data.neighbors.NeighborSetFinder;
import data.representation.DataSet;
import distances.primary.CombinedMetric;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import data.neighbors.NSFUserInterface;
import util.AuxSort;

/**
 * INSIGHT: instance selection algorithm designed for the time-series domain,
 * but general in principle, described in the PAKDD 2011 paper by Krisztian Buza
 * INSIGHT scores the points and selects the top points according to a hubness-
 * based goodness/badness score.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class INSIGHT extends InstanceSelector implements NSFUserInterface {

    private NeighborSetFinder nsf;
    // Neighborhood size for performing instance selection.
    private int kSelection = 1;
    private int weightingMode = 0;
    private float thau = 0.7f;
    // When sampling without a predefined proportion / sample size, this is set
    // as the percentage of all occurrences that the prototypes must account for
    // Instance scores that determine what is to be retained.
    private float[] instanceScores;
    // Different selection/scoring modes.
    public static final int GOOD_HUBNESS = 0;
    public static final int GOOD_HUBNESS_RELATIVE = 1;
    public static final int XI = 2;
    public static final int GOOD_MINUS_BAD_HUBNESS_PROP = 3;

    public INSIGHT() {
    }

    /**
     * @param kSelection Neighborhood size for hubness-based selection.
     */
    public INSIGHT(int kSelection) {
        this.kSelection = kSelection;
    }

    /**
     * @param nsf NeighborSetFinder object.
     */
    public INSIGHT(NeighborSetFinder nsf) {
        this.nsf = nsf;
        kSelection = nsf.getKNeighbors()[0].length;
    }

    /**
     * @param originalDataSet Original data set, prior to reduction.
     * @param kSelection Neighborhood size for hubness-based selection.
     * @param weightingMode Integer denoting the scoring/selection mode.
     */
    public INSIGHT(DataSet originalDataSet, int kSelection, int weightingMode) {
        setOriginalDataSet(originalDataSet);
        this.kSelection = kSelection;
        this.weightingMode = weightingMode;
    }

    /**
     * @param nsf NeighborSetFinder object.
     * @param weightingMode Integer denoting the scoring/selection mode.
     */
    public INSIGHT(NeighborSetFinder nsf, int weightingMode) {
        if (nsf != null) {
            setOriginalDataSet(nsf.getDataSet());
        }
        this.nsf = nsf;
        if (nsf != null) {
            kSelection = nsf.getKNeighbors()[0].length;
        }
        this.weightingMode = weightingMode;
    }

    /**
     * @param originalDataSet Original data set, prior to reduction.
     * @param kSelection Neighborhood size for hubness-based selection.
     * @param cmet CombinedMetric object for distance calculations.
     * @param weightingMode Integer denoting the scoring/selection mode.
     */
    public INSIGHT(
            DataSet originalDataSet,
            int kSelection,
            CombinedMetric cmet,
            int weightingMode) {
        setOriginalDataSet(originalDataSet);
        this.kSelection = kSelection;
        this.weightingMode = weightingMode;
        setCombinedMetric(cmet);
    }

    /**
     * @param nsf NeighborSetFinder object.
     * @param cmet CombinedMetric object for distance calculations.
     * @param weightingMode Integer denoting the scoring/selection mode.
     */
    public INSIGHT(
            NeighborSetFinder nsf,
            CombinedMetric cmet,
            int weightingMode) {
        setOriginalDataSet(nsf.getDataSet());
        this.nsf = nsf;
        kSelection = nsf.getKNeighbors()[0].length;
        this.weightingMode = weightingMode;
        setCombinedMetric(cmet);
    }

    /**
     * @param thau The retainment percentage.
     */
    public void setThau(float thau) {
        this.thau = thau;
    }

    /**
     * @return The retainment percentage.
     */
    public float getThau() {
        return thau;
    }

    @Override
    public InstanceSelector copy() {
        CombinedMetric cmet = getCombinedMetric();
        if (nsf == null) {
            return new INSIGHT(null, kSelection, cmet, weightingMode);
        } else {
            return new INSIGHT(nsf, cmet, weightingMode);
        }
    }

    @Override
    public void calculatePrototypeHubness(int k) throws Exception {
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
        ArrayList<Integer> intervals;
        // Auxiliary array for fast restricted kNN search.
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
                                    if (distMatrix[min][max - min - 1] <
                                            kdistances[i][kcurrLen[i] - 1]) {
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
    public void reduceDataSet() throws Exception {
        if (nsf == null) {
            nsf = new NeighborSetFinder(getOriginalDataSet(),
                    CombinedMetric.FLOAT_MANHATTAN);
            nsf.calculateDistances();
            nsf.calculateNeighborSets(kSelection);
        }
        weightingMode = weightingMode % 4;
        DataSet originalDataSet = getOriginalDataSet();
        instanceScores = new float[originalDataSet.size()];
        int[] gHubness = nsf.getGoodFrequencies();
        int[] bHubness = nsf.getBadFrequencies();
        int[] totalHubness = nsf.getNeighborFrequencies();
        // Set the proper scores based on the operating mode.
        switch (weightingMode) {
            case GOOD_HUBNESS: {
                for (int i = 0; i < originalDataSet.size(); i++) {
                    instanceScores[i] = gHubness[i];
                }
                break;
            }
            case GOOD_HUBNESS_RELATIVE: {
                for (int i = 0; i < originalDataSet.size(); i++) {
                    instanceScores[i] = (float) gHubness[i]
                            / (float) (totalHubness[i] + 1);
                }
                break;
            }
            case GOOD_MINUS_BAD_HUBNESS_PROP: {
                for (int i = 0; i < originalDataSet.size(); i++) {
                    instanceScores[i] = (float) (gHubness[i] - bHubness[i])
                            / (float) (totalHubness[i] + 1);
                }
                break;
            }
            case XI: {
                for (int i = 0; i < originalDataSet.size(); i++) {
                    instanceScores[i] = gHubness[i] - 2 * bHubness[i];
                }
                break;
            }
            default: {
                for (int i = 0; i < originalDataSet.size(); i++) {
                    instanceScores[i] = (float) gHubness[i]
                            / (float) (totalHubness[i] + 1);
                }
                break;
            }
        }
        int[] indexes = AuxSort.sortIndexedValue(instanceScores, true);
        // Select the first numPrototypes instances.
        ArrayList<Integer> pIndexes =
                new ArrayList<>(originalDataSet.size() / 4);
        int sumHubness = 0;
        int thresholdHubness =
                (int) (thau * (float) (kSelection * originalDataSet.size()));
        int i = 0;
        int numClasses = originalDataSet.countCategories();
        int[] classCounts = new int[numClasses];
        do {
            pIndexes.add(indexes[i]);
            classCounts[originalDataSet.getLabelOf(indexes[i])]++;
            sumHubness += totalHubness[i];
            i++;
        } while (sumHubness < thresholdHubness);
        int numEmptyClasses = 0;
        for (int cIndex = 0; cIndex < numClasses; cIndex++) {
            if (classCounts[cIndex] == 0) {
                numEmptyClasses++;
            }
        }
        while (numEmptyClasses > 0 && i < indexes.length) {
            int label = originalDataSet.getLabelOf(indexes[i]);
            if (classCounts[label] == 0) {
                pIndexes.add(indexes[i]);
                classCounts[label]++;
                numEmptyClasses--;
            }
            i++;
        }
        setPrototypeIndexes(pIndexes);
        sortSelectedIndexes();
    }

    @Override
    public void reduceDataSet(int numPrototypes) throws Exception {
        if (nsf == null) {
            nsf = new NeighborSetFinder(getOriginalDataSet(),
                    CombinedMetric.FLOAT_MANHATTAN);
            nsf.calculateDistances();
            nsf.calculateNeighborSets(kSelection);
        }
        weightingMode = weightingMode % 4;
        DataSet originalDataSet = getOriginalDataSet();
        instanceScores = new float[originalDataSet.size()];
        int[] gHubness = nsf.getGoodFrequencies();
        int[] bHubness = nsf.getBadFrequencies();
        int[] totalHubness = nsf.getNeighborFrequencies();
        // Set the proper scores based on the operating mode.
        switch (weightingMode) {
            case GOOD_HUBNESS: {
                for (int i = 0; i < originalDataSet.size(); i++) {
                    instanceScores[i] = gHubness[i];
                }
                break;
            }
            case GOOD_HUBNESS_RELATIVE: {
                for (int i = 0; i < originalDataSet.size(); i++) {
                    instanceScores[i] = (float) gHubness[i]
                            / (float) (totalHubness[i] + 1);
                }
                break;
            }
            case XI: {
                for (int i = 0; i < originalDataSet.size(); i++) {
                    instanceScores[i] = gHubness[i] - 2 * bHubness[i];
                }
                break;
            }
            default: {
                for (int i = 0; i < originalDataSet.size(); i++) {
                    instanceScores[i] = (float) gHubness[i]
                            / (float) (totalHubness[i] + 1);
                }
                break;
            }
        }
        int[] indexes = AuxSort.sortIndexedValue(instanceScores, true);
        // Select the first numPrototypes instances.
        ArrayList<Integer> pIndexes = new ArrayList<>(numPrototypes);
        int numClasses = originalDataSet.countCategories();
        int[] classCounts = new int[numClasses];
        for (int i = 0; i < numPrototypes; i++) {
            pIndexes.add(indexes[i]);
            classCounts[originalDataSet.getLabelOf(indexes[i])]++;
        }
        int numEmptyClasses = 0;
        for (int cIndex = 0; cIndex < numClasses; cIndex++) {
            if (classCounts[cIndex] == 0) {
                numEmptyClasses++;
            }
        }
        int i = numPrototypes + 1;
        while (numEmptyClasses > 0 && i < indexes.length) {
            int label = originalDataSet.getLabelOf(indexes[i]);
            if (classCounts[label] == 0) {
                pIndexes.add(indexes[i]);
                classCounts[label]++;
                numEmptyClasses--;
            }
            i++;
        }
        setPrototypeIndexes(pIndexes);
        sortSelectedIndexes();
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
}
