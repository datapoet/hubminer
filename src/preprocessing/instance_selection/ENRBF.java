/**
 * Hub Miner: a hubness-aware machine learning experimentation library.
 * Copyright (C) 2014 Nenad Tomasev. Email: nenad.tomasev at gmail.com
 * 
* This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 * 
* This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 * 
* You should have received a copy of the GNU General Public License along with
 * this program. If not, see <http://www.gnu.org/licenses/>.
 */
package preprocessing.instance_selection;

import algref.Author;
import algref.BookChapterPublication;
import algref.Publication;
import data.neighbors.NSFUserInterface;
import data.neighbors.NeighborSetFinder;
import data.representation.DataSet;
import distances.primary.CombinedMetric;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Random;
import statistics.HigherMoments;

/**
 * This class implements the edited normalized RBF noise filter that can be used
 * for instance selection.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class ENRBF extends InstanceSelector implements NSFUserInterface {

    public static final double DEFAULT_ALPHA_VALUE = 0.9;
    private double alpha = DEFAULT_ALPHA_VALUE;
    // The upper triangular distance matrix on the data.
    private float[][] distMat;
    // Object that holds the kNN sets.
    private NeighborSetFinder nsf;

    /**
     * Default constructor.
     */
    public ENRBF() {
    }

    /**
     * Initialization.
     *
     * @param dset DataSet to reduce.
     * @param distMat float[][] that is the upper triangular distance matrix on
     * the data.
     * @param alpha Double value that is the parameter to use for determining
     * which instances to keep.
     */
    public ENRBF(DataSet dset, float[][] distMat, double alpha) {
        this.alpha = alpha;
        this.distMat = distMat;
        setOriginalDataSet(dset);
    }

    @Override
    public void reduceDataSet() throws Exception {
        DataSet originalDataSet = getOriginalDataSet();
        int dataSize = originalDataSet.size();
        // Initialization.
        int numClasses = getNumClasses();
        // First estimate the sigma value for the kernal.
        int minIndex, maxIndex, firstChoice, secondChoice;
        float[] distSample = new float[50];
        Random randa = new Random();
        for (int i = 0; i < 50; i++) {
            firstChoice = randa.nextInt(dataSize);
            secondChoice = firstChoice;
            while (firstChoice == secondChoice) {
                secondChoice = randa.nextInt(dataSize);
            }
            minIndex = Math.min(firstChoice, secondChoice);
            maxIndex = Math.max(firstChoice, secondChoice);
            distSample[i] = distMat[minIndex][maxIndex - minIndex - 1];
        }
        float distMean = HigherMoments.calculateArrayMean(distSample);
        float distSigma =
                HigherMoments.calculateArrayStDev(distMean, distSample);
        // Calculate the RBF matrix.
        double[][] rbfMat = new double[distMat.length][];
        double[] pointTotals = new double[dataSize];
        for (int i = 0; i < distMat.length; i++) {
            rbfMat[i] = new double[distMat.length - i - 1];
            for (int j = 0; j < distMat[i].length; j++) {
                rbfMat[i][j] = Math.exp(-(distMat[i][j] * distMat[i][j])
                        / distSigma);
                pointTotals[i] += rbfMat[i][j];
                pointTotals[i + j + 1] += rbfMat[i][j];
            }
        }
        // Calculate the class probabilities in points based on the RBF 
        // estimate.
        double[][] pointClassProbs = new double[dataSize][numClasses];
        int firstLabel, secondLabel;
        for (int i = 0; i < distMat.length; i++) {
            firstLabel = originalDataSet.getLabelOf(i);
            for (int j = 0; j < distMat[i].length; j++) {
                secondLabel = originalDataSet.getLabelOf(i + j + 1);
                pointClassProbs[i][secondLabel] +=
                        rbfMat[i][j] / pointTotals[i];
                pointClassProbs[i + j + 1][firstLabel] +=
                        rbfMat[i][j] / pointTotals[i + j + 1];
            }
        }
        // Now perform the filtering.
        ArrayList<Integer> protoIndexes = new ArrayList<>(dataSize);
        int label;
        boolean acceptable;
        for (int i = 0; i < dataSize; i++) {
            label = originalDataSet.getLabelOf(i);
            acceptable = true;
            for (int c = 0; c < numClasses; c++) {
                if (c != label && pointClassProbs[i][label]
                        < alpha * pointClassProbs[i][c]) {
                    acceptable = false;
                    break;
                }
            }
            if (acceptable) {
                protoIndexes.add(i);
            }
        }
        // Check whether at least one instance of each class has been selected.
        int[] protoClassCounts = new int[numClasses];
        int numEmptyClasses = numClasses;
        for (int i = 0; i < protoIndexes.size(); i++) {
            label = originalDataSet.getLabelOf(protoIndexes.get(i));
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
                label = originalDataSet.getLabelOf(i);
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
    public Publication getPublicationInfo() {
        BookChapterPublication pub = new BookChapterPublication();
        pub.setTitle("Data regularization");
        pub.addAuthor(new Author("N.", "Jankowski"));
        pub.setBookName("Neural Networks and Soft Computing");
        pub.setYear(2000);
        pub.setStartPage(209);
        pub.setEndPage(214);
        return pub;
    }

    @Override
    public InstanceSelector copy() {
        return new ENRBF(getOriginalDataSet(), distMat, alpha);
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
