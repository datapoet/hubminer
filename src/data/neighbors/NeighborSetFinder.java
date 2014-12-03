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
package data.neighbors;

import data.neighbors.approximate.AppKNNGraphLanczosBisection;
import data.representation.DataInstance;
import data.representation.DataSet;
import distances.primary.CombinedMetric;
import ioformat.FileUtil;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import util.ArrayUtil;
import util.BasicMathUtil;
import util.SOPLUtil;

/**
 * This class implements the functionality for exact kNN search and kNN graph
 * calculations, in various contexts. It also implements the functionality for
 * calculating the neighbor occurrence frequencies, good and bad occurrences,
 * reverse neighbor sets, reverse and direct neighbor set entropies and other
 * hubness-related measures. Functionally, it implements various hubness-based
 * weighting modes and the class-conditional probabilistic model estimates for
 * hubness-aware classification. It is a simple implementation in that the
 * default kNN search and graph construction methods do not rely on additional
 * spatial indexing. The reason for that, though - is that this library is meant
 * primarily for high-dimensional data analysis, where such indexes have been
 * shown to be of little use - and calculating them takes time. In case of large
 * low-to-medium dimensional datasets where spatial indexing can be very useful,
 * alternative implementations should be used. This one is meant for high-dim
 * data instead. Also, in case of large-scale datasets, approximate kNN
 * extensions are to be preferred.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class NeighborSetFinder implements Serializable {

    private static final long serialVersionUID = 1L;
    // Dataset to calculate the k-nearest neighbor sets from.
    private DataSet dset = null;
    // The upper triangular distance matrix, as used throughout the library.
    private float[][] distMatrix = null;
    // CombinedMetric object for distance calculations.
    private CombinedMetric cmet = null;
    // The k-nearest neighbor sets. Each row in the table contains the indexes
    // of the k-nearest neighbors for the given point in the provided DataSet.
    private int[][] kNeighbors = null;
    // Distances to the k-nearest neighbors for each point.
    private float[][] kDistances = null;
    // The current length of the kNN sets, used during kNN calculations.
    private int[] kCurrLen = null;
    // The neighbor occurrence frequencies.
    private int[] kNeighborFrequencies = null;
    // The bad neighbor occurrence frequencies.
    private int[] kBadFrequencies = null;
    // The good neighbor occurrence frequencies.
    private int[] kGoodFrequencies = null;
    // Reverse neighbor sets.
    private ArrayList<Integer>[] reverseNeighbors = null;
    // Boolean flag indicating whether the distance matrix was provided.
    private boolean distancesCalculated = false;
    // Variance of the distance values.
    private double distVariance = 0;
    // Mean of the distance value.
    private double distMean = 0;
    // Mean of the neighbor occurrence frequency.
    private double meanOccFreq;
    // Standard deviation of the neighbor occurrence frequency.
    private double stDevOccFreq;
    // Mean of the detrimental occurrence frequency.
    private double meanOccBadness = 0;
    // Standard deviation of the detrimental occurrence frequency.
    private double stDevOccBadness = 0;
    // Mean of the beneficial neighbor occurrence frequency.
    private double meanOccGoodness = 0;
    // Standard deviation of the beneficial neighbor occurrence frequency.
    private double stDevOccGoodness = 0;
    // Mean of the difference between the good and the bad occurrence counts.
    private double meanGoodMinusBadness = 0;
    // Mean of the normalized difference between the good and bad occurrence
    // counts.
    private double meanRelativeGoodMinusBadness = 0;
    // Standard deviation of the difference between the good and the bad
    // occurrence counts.
    private double stDevGoodMinusBadness = 0;
    // Standard deviation of the normalized difference between the good and bad
    // occurrence counts.
    private double stDevRelativeGoodMinusBadness = 0;
    // Entropies of the direct kNN sets.
    private float[] kEntropies = null;
    // Entropies of the reverse kNN sets.
    private float[] kRNNEntropies = null;
    // The currently operating neighborhood size.
    private int currK;
    // Small datasets can be extended by synthetic instances from the Gaussian
    // data model.

    /**
     * The default constructor.
     */
    public NeighborSetFinder() {
    }

    /**
     * Initialization.
     *
     * @param dset DataSet object that holds the data to calculate the kNN sets
     * for.
     * @param cmet CombinedMetric object for distance calculations.
     */
    public NeighborSetFinder(DataSet dset, CombinedMetric cmet) {
        this.dset = dset;
        this.cmet = cmet;
    }

    /**
     * Initialization.
     *
     * @param dset DataSet object that holds the data to calculate the kNN sets
     * for.
     * @param distMatrix float[][] that is the upper triangular distance matrix.
     * @param cmet CombinedMetric object for distance calculations.
     */
    public NeighborSetFinder(DataSet dset, float[][] distMatrix,
            CombinedMetric cmet) {
        this.dset = dset;
        this.distMatrix = distMatrix;
        this.cmet = cmet;
        if (distMatrix == null) {
            try {
                distMatrix = dset.calculateDistMatrix(cmet);
            } catch (Exception e) {
            }
        }
        distancesCalculated = true;
        try {
            calculateOccFreqMeanAndVariance();
        } catch (Exception e) {
            System.err.println("NSF constructor error.");
            System.err.println(e.getMessage());
        }
    }

    /**
     * Initialization.
     *
     * @param dset DataSet object that holds the data to calculate the kNN sets
     * for.
     * @param distMatrix float[][] that is the upper triangular distance matrix.
     */
    public NeighborSetFinder(DataSet dset, float[][] distMatrix) {
        this.dset = dset;
        this.distMatrix = distMatrix;
        this.cmet = CombinedMetric.FLOAT_EUCLIDEAN;
        distancesCalculated = true;
        try {
            calculateOccFreqMeanAndVariance();
        } catch (Exception e) {
            System.err.println("NSF constructor error.");
            System.err.println(e.getMessage());
        }
    }

    /**
     * @return True if the distance matrix is already available, false
     * otherwise.
     */
    public boolean distancesCalculated() {
        return distancesCalculated;
    }

    /**
     * This method persists the calculated kNN sets to a file.
     *
     * @param outFile File to save the kNN sets to.
     * @throws Exception
     */
    public void saveNeighborSets(File outFile) throws Exception {
        FileUtil.createFile(outFile);
        try (PrintWriter pw = new PrintWriter(new FileWriter(outFile));) {
            if (kNeighbors != null && kNeighbors.length > 0) {
                pw.println("size:" + kNeighbors.length);
                pw.println("k:" + kNeighbors[0].length);
                for (int i = 0; i < kNeighbors.length; i++) {
                    SOPLUtil.printArrayToStream(kNeighbors[i], pw);
                    SOPLUtil.printArrayToStream(kDistances[i], pw);
                }
            } else {
                pw.println("size:0");
                pw.println("k:0");
            }
        } catch (Exception e) {
            System.err.println(e.getMessage());
        }
    }

    /**
     * This method loads the NeighborSetFinder object from a kNN set file.
     *
     * @param inFile File where the kNN sets were persisted.
     * @param dset DataSet object that the kNN sets point to.
     * @return NeighborSet finder that was loaded from the disk.
     * @throws Exception
     */
    public static NeighborSetFinder loadNSF(File inFile, DataSet dset)
            throws Exception {
        NeighborSetFinder loadedNSF = new NeighborSetFinder();
        loadedNSF.loadNeighborSets(inFile, dset);
        return loadedNSF;
    }

    /**
     * This method loads the kNN sets from a file.
     *
     * @param inFile File where the kNN sets were persisted.
     * @param dset DataSet object that the kNN sets point to.
     * @throws Exception
     */
    public void loadNeighborSets(File inFile, DataSet dset) throws Exception {
        try (BufferedReader br = new BufferedReader(
                new InputStreamReader(new FileInputStream(inFile)));) {
            this.dset = dset;
            String line;
            String[] lineItems;
            line = br.readLine();
            // Get data size.
            int size = Integer.parseInt((line.split(":"))[1]);
            line = br.readLine();
            // Get the neighborhood size.
            int k = Integer.parseInt((line.split(":"))[1]);
            currK = k;
            // Initialize the arrays.
            kNeighbors = new int[size][k];
            kDistances = new float[size][k];
            kCurrLen = new int[size];
            kNeighborFrequencies = new int[size];
            kBadFrequencies = new int[size];
            kGoodFrequencies = new int[size];
            kEntropies = new float[size];
            kRNNEntropies = new float[size];
            reverseNeighbors = new ArrayList[size];
            // Initialize the reverse neighbor lists.
            for (int i = 0; i < size; i++) {
                reverseNeighbors[i] = new ArrayList<>(k);
            }
            int label;
            for (int i = 0; i < size; i++) {
                label = dset.getLabelOf(i);
                // The following line holds the kNN set.
                line = br.readLine();
                // The neighbor indexes are split by empty spaces.
                lineItems = line.split(" ");
                kCurrLen[i] = Math.min(lineItems.length, k);
                for (int kInd = 0; kInd < kCurrLen[i]; kInd++) {
                    // Parse the neighbor index. 
                    kNeighbors[i][kInd] = Integer.parseInt(lineItems[kInd]);
                    // Updated the good and bad occurrence counts.
                    if (label == dset.getLabelOf(kNeighbors[i][kInd])) {
                        kGoodFrequencies[kNeighbors[i][kInd]]++;
                    } else {
                        kBadFrequencies[kNeighbors[i][kInd]]++;
                    }
                    // Update the total occurrence count.
                    kNeighborFrequencies[kNeighbors[i][kInd]]++;
                    // Update the reverse neighbor list.
                    reverseNeighbors[kNeighbors[i][kInd]].add(i);
                }
                // The following line holds the distances to the previous kNN
                // set.
                line = br.readLine();
                lineItems = line.split(" ");
                for (int kInd = 0; kInd < Math.min(kCurrLen[i],
                        lineItems.length); kInd++) {
                    kDistances[i][kInd] = Float.parseFloat(lineItems[kInd]);
                }
            }
            calculateHubnessStats();
        } catch (Exception e) {
            System.out.println(e.getMessage());
        }
    }

    /**
     * @param k Integer that is the queried neighborhood size.
     * @return True if the calculated kNN sets' length exceeds k, false
     * otherwise.
     */
    public boolean isCalculatedUpToK(int k) {
        return (kNeighbors != null && kDistances != null
                && kNeighbors.length >= k && kDistances.length >= k);
    }

    /**
     * This method calculates the average distance to the k-nearest neighbors
     * for each point for the specified neighborhood size.
     *
     * @param k Integer that is the neighborhood size to calculate the average
     * k-distances for.
     * @return float[] representing the average distance to the k-nearest
     * neighbors for each point in the data.
     */
    public float[] getAvgDistToNeighbors(int k) {
        if (k <= 0) {
            return null;
        }
        if (k > kDistances[0].length) {
            k = kDistances[0].length;
        }
        float[] avgKDists = new float[kDistances.length];
        for (int i = 0; i < kDistances.length; i++) {
            avgKDists[i] = 0;
            for (int kInd = 0; kInd < k; kInd++) {
                avgKDists[i] += kDistances[i][kInd];
            }
            avgKDists[i] /= k;
        }
        return avgKDists;
    }

    /**
     * @return Integer that is the currently operating neighborhood size.
     */
    public int getCurrK() {
        return currK;
    }

    /**
     * @param distMatrix float[][] representing the upper triangular distance
     * matrix, where the length of each row i is (size - i - 1) and each row
     * contains only the entries for j > i, so that distMatrix[i][j] represents
     * the distance between i and i + j + 1.
     */
    public void setDistances(float[][] distMatrix) {
        this.distMatrix = distMatrix;
        distancesCalculated = true;
        calculateOccFreqMeanAndVariance();
    }

    /**
     * @return NeighborSetFinder that is the copy of this NeighborSetFinder
     * object.
     */
    public NeighborSetFinder copy() {
        NeighborSetFinder nsfCopy = new NeighborSetFinder();
        // Shallow copies of the DataSet, the distances and the metric object.
        nsfCopy.dset = dset;
        nsfCopy.cmet = cmet;
        nsfCopy.distMatrix = distMatrix;
        // Copy the k-nearest neighbor sets.
        if (kNeighbors != null) {
            nsfCopy.kNeighbors = new int[kNeighbors.length][];
            for (int i = 0; i < kNeighbors.length; i++) {
                nsfCopy.kNeighbors[i] = Arrays.copyOf(kNeighbors[i],
                        kNeighbors[i].length);
            }
        }
        // Copy the distances to the k-nearest neighbors.
        if (kDistances != null) {
            nsfCopy.kDistances = new float[kDistances.length][];
            for (int i = 0; i < kDistances.length; i++) {
                nsfCopy.kDistances[i] = Arrays.copyOf(kDistances[i],
                        kDistances[i].length);
            }
        }
        // Copy the current k-lengths.
        if (kCurrLen != null) {
            nsfCopy.kCurrLen = Arrays.copyOf(kCurrLen, kCurrLen.length);
        }
        // Copy the occurrence frequency and the entropy arrays.
        if (kNeighborFrequencies != null) {
            nsfCopy.kNeighborFrequencies = Arrays.copyOf(kNeighborFrequencies,
                    kNeighborFrequencies.length);
        }
        if (kBadFrequencies != null) {
            nsfCopy.kBadFrequencies = Arrays.copyOf(kBadFrequencies,
                    kBadFrequencies.length);
        }
        if (kGoodFrequencies != null) {
            nsfCopy.kGoodFrequencies = Arrays.copyOf(kGoodFrequencies,
                    kGoodFrequencies.length);
        }
        if (kEntropies != null) {
            nsfCopy.kEntropies = Arrays.copyOf(kEntropies, kEntropies.length);
        }
        if (kRNNEntropies != null) {
            nsfCopy.kRNNEntropies = Arrays.copyOf(kRNNEntropies,
                    kRNNEntropies.length);
        }
        // Copy the operating neighborhood size.
        nsfCopy.currK = currK;
        // Copy all the stats and flags.
        nsfCopy.distancesCalculated = distancesCalculated;
        nsfCopy.distVariance = distVariance;
        nsfCopy.distMean = distMean;
        nsfCopy.meanOccFreq = meanOccFreq;
        nsfCopy.stDevOccFreq = stDevOccFreq;
        nsfCopy.meanOccBadness = meanOccBadness;
        nsfCopy.stDevOccBadness = stDevOccBadness;
        nsfCopy.meanOccGoodness = meanOccGoodness;
        nsfCopy.stDevOccGoodness = stDevOccGoodness;
        nsfCopy.meanGoodMinusBadness = meanGoodMinusBadness;
        nsfCopy.meanRelativeGoodMinusBadness = meanRelativeGoodMinusBadness;
        nsfCopy.stDevGoodMinusBadness = stDevGoodMinusBadness;
        nsfCopy.stDevRelativeGoodMinusBadness = stDevRelativeGoodMinusBadness;
        // Copy the reverse neighbor lists.
        if (reverseNeighbors != null) {
            nsfCopy.reverseNeighbors = new ArrayList[reverseNeighbors.length];
            for (int i = 0; i < reverseNeighbors.length; i++) {
                if (reverseNeighbors[i] != null) {
                    nsfCopy.reverseNeighbors[i] =
                            new ArrayList<>(reverseNeighbors[i].size());
                    for (int p = 0; p < reverseNeighbors[i].size(); p++) {
                        nsfCopy.reverseNeighbors[i].add(
                                reverseNeighbors[i].get(p));
                    }

                }
            }
        }
        return nsfCopy;
    }

    /**
     * @param kcurrLen int[] that are the current kNN set lengths.
     */
    public void setKCurrLen(int[] kcurrLen) {
        this.kCurrLen = kcurrLen;
    }

    /**
     * @return int[] that are the current kNN set lengths.
     */
    public int[] getKCurrLen() {
        return kCurrLen;
    }

    /**
     * Sets the kNN set to this NeighborSetFinder object.
     *
     * @param kneighbors int[][] representing the k-nearest neighbors.
     * @param kDistances float[][] representing the k-distances.
     * @param kcurrLen int[] representing the current kNN set lengths (in case
     * some of them are not yet completed)
     */
    public void setKNeighbors(int[][] kneighbors, float[][] kDistances,
            int[] kcurrLen) {
        this.kNeighbors = kneighbors;
        this.kDistances = kDistances;
        int k = kneighbors[0].length;
        this.kCurrLen = kcurrLen;
        // Set the operating neighborhood size to the length of the kNN sets.
        currK = k;
        reverseNeighbors = new ArrayList[dset.size()];
        for (int i = 0; i < dset.size(); i++) {
            reverseNeighbors[i] = new ArrayList<>(10 * k);
        }
        kNeighborFrequencies = new int[kneighbors.length];
        kBadFrequencies = new int[kneighbors.length];
        kGoodFrequencies = new int[kneighbors.length];
        for (int i = 0; i < kneighbors.length; i++) {
            for (int kInd = 0; kInd < kcurrLen[i]; kInd++) {
                reverseNeighbors[kneighbors[i][kInd]].add(i);
                kNeighborFrequencies[kneighbors[i][kInd]]++;
                if (dset.data.get(i).getCategory() != dset.data.get(
                        kneighbors[i][kInd]).getCategory()) {
                    kBadFrequencies[kneighbors[i][kInd]]++;
                } else {
                    kGoodFrequencies[kneighbors[i][kInd]]++;
                }
            }
        }
        // Calculate the neighbor occurrence frequency stats.
        meanOccFreq = 0;
        stDevOccFreq = 0;
        meanOccBadness = 0;
        stDevOccBadness = 0;
        meanOccGoodness = 0;
        stDevOccGoodness = 0;
        meanGoodMinusBadness = 0;
        stDevGoodMinusBadness = 0;
        meanRelativeGoodMinusBadness = 0;
        stDevRelativeGoodMinusBadness = 0;
        for (int i = 0; i < kBadFrequencies.length; i++) {
            meanOccFreq += kNeighborFrequencies[i];
            meanOccBadness += kBadFrequencies[i];
            meanOccGoodness += kGoodFrequencies[i];
            meanGoodMinusBadness += kGoodFrequencies[i] - kBadFrequencies[i];
            if (kNeighborFrequencies[i] > 0) {
                meanRelativeGoodMinusBadness += ((kGoodFrequencies[i]
                        - kBadFrequencies[i]) / kNeighborFrequencies[i]);
            } else {
                meanRelativeGoodMinusBadness += 1;
            }
        }
        meanOccFreq /= (float) kNeighborFrequencies.length;
        meanOccBadness /= (float) kBadFrequencies.length;
        meanOccGoodness /= (float) kGoodFrequencies.length;
        meanGoodMinusBadness /= (float) kGoodFrequencies.length;
        meanRelativeGoodMinusBadness /= (float) kGoodFrequencies.length;
        for (int i = 0; i < kBadFrequencies.length; i++) {
            stDevOccFreq += ((meanOccFreq - kNeighborFrequencies[i])
                    * (meanOccFreq - kNeighborFrequencies[i]));
            stDevOccBadness += ((meanOccBadness - kBadFrequencies[i])
                    * (meanOccBadness - kBadFrequencies[i]));
            stDevOccGoodness += ((meanOccGoodness - kGoodFrequencies[i])
                    * (meanOccGoodness - kGoodFrequencies[i]));
            stDevGoodMinusBadness += ((meanGoodMinusBadness
                    - (kGoodFrequencies[i] - kBadFrequencies[i]))
                    * (meanGoodMinusBadness - (kGoodFrequencies[i]
                    - kBadFrequencies[i])));
            if (kNeighborFrequencies[i] > 0) {
                stDevRelativeGoodMinusBadness += (meanRelativeGoodMinusBadness
                        - ((kGoodFrequencies[i] - kBadFrequencies[i])
                        / kNeighborFrequencies[i]))
                        * (meanRelativeGoodMinusBadness - ((kGoodFrequencies[i]
                        - kBadFrequencies[i]) / kNeighborFrequencies[i]));
            } else {
                stDevRelativeGoodMinusBadness +=
                        (meanRelativeGoodMinusBadness - 1)
                        * (meanRelativeGoodMinusBadness - 1);
            }
        }
        stDevOccFreq /= (float) kNeighborFrequencies.length;
        stDevOccBadness /= (float) kBadFrequencies.length;
        stDevOccGoodness /= (float) kGoodFrequencies.length;
        stDevGoodMinusBadness /= (float) kGoodFrequencies.length;
        stDevRelativeGoodMinusBadness /= (float) kGoodFrequencies.length;
        stDevOccFreq = Math.sqrt(stDevOccFreq);
        stDevOccBadness = Math.sqrt(stDevOccBadness);
        stDevOccGoodness = Math.sqrt(stDevOccGoodness);
        stDevGoodMinusBadness = Math.sqrt(stDevGoodMinusBadness);
        stDevRelativeGoodMinusBadness =
                Math.sqrt(stDevRelativeGoodMinusBadness);
    }

    /**
     * Sets the kNN set to this NeighborSetFinder object.
     *
     * @param kneighbors int[][] representing the k-nearest neighbors.
     * @param kDistances float[][] representing the k-distances.
     */
    public void setKNeighbors(int[][] kneighbors, float[][] kDistances) {
        this.kNeighbors = kneighbors;
        this.kDistances = kDistances;
        int k = kneighbors[0].length;
        kCurrLen = new int[kneighbors.length];
        // The kNN sets are completed.
        Arrays.fill(kCurrLen, k);
        // Set the operating neighborhood size.
        currK = k;
        // Initialize the reverse neighbor lists.
        reverseNeighbors = new ArrayList[dset.size()];
        for (int i = 0; i < dset.size(); i++) {
            reverseNeighbors[i] = new ArrayList<>(10 * k);
        }
        kNeighborFrequencies = new int[kneighbors.length];
        kBadFrequencies = new int[kneighbors.length];
        kGoodFrequencies = new int[kneighbors.length];
        // Fill in the reverse neighbor lists and count the occurence
        // frequencies.
        for (int i = 0; i < kneighbors.length; i++) {
            for (int kInd = 0; kInd < k; kInd++) {
                reverseNeighbors[kneighbors[i][kInd]].add(i);
                kNeighborFrequencies[kneighbors[i][kInd]]++;
                if (dset.data.get(i).getCategory() != dset.data.get(
                        kneighbors[i][kInd]).getCategory()) {
                    kBadFrequencies[kneighbors[i][kInd]]++;
                } else {
                    kGoodFrequencies[kneighbors[i][kInd]]++;
                }
            }
        }
        // Calculate the occurrence frequency stats.
        meanOccBadness = 0;
        stDevOccBadness = 0;
        meanOccGoodness = 0;
        stDevOccGoodness = 0;
        meanGoodMinusBadness = 0;
        stDevGoodMinusBadness = 0;
        meanRelativeGoodMinusBadness = 0;
        stDevRelativeGoodMinusBadness = 0;
        for (int i = 0; i < kBadFrequencies.length; i++) {
            meanOccBadness += kBadFrequencies[i];
            meanOccGoodness += kGoodFrequencies[i];
            meanGoodMinusBadness += kGoodFrequencies[i] - kBadFrequencies[i];
            if (kNeighborFrequencies[i] > 0) {
                meanRelativeGoodMinusBadness +=
                        ((kGoodFrequencies[i] - kBadFrequencies[i])
                        / kNeighborFrequencies[i]);
            } else {
                meanRelativeGoodMinusBadness += 1;
            }
        }
        meanOccBadness /= (float) kBadFrequencies.length;
        meanOccGoodness /= (float) kGoodFrequencies.length;
        meanGoodMinusBadness /= (float) kGoodFrequencies.length;
        meanRelativeGoodMinusBadness /= (float) kGoodFrequencies.length;
        for (int i = 0; i < kBadFrequencies.length; i++) {
            stDevOccBadness += ((meanOccBadness - kBadFrequencies[i])
                    * (meanOccBadness - kBadFrequencies[i]));
            stDevOccGoodness += ((meanOccGoodness - kGoodFrequencies[i])
                    * (meanOccGoodness - kGoodFrequencies[i]));
            stDevGoodMinusBadness += ((meanGoodMinusBadness
                    - (kGoodFrequencies[i] - kBadFrequencies[i]))
                    * (meanGoodMinusBadness - (kGoodFrequencies[i]
                    - kBadFrequencies[i])));
            if (kNeighborFrequencies[i] > 0) {
                stDevRelativeGoodMinusBadness += (meanRelativeGoodMinusBadness
                        - ((kGoodFrequencies[i] - kBadFrequencies[i])
                        / kNeighborFrequencies[i]))
                        * (meanRelativeGoodMinusBadness - ((kGoodFrequencies[i]
                        - kBadFrequencies[i]) / kNeighborFrequencies[i]));
            } else {
                stDevRelativeGoodMinusBadness +=
                        (meanRelativeGoodMinusBadness - 1)
                        * (meanRelativeGoodMinusBadness - 1);
            }
        }
        // Normalize the averages.
        stDevOccBadness /= (float) kBadFrequencies.length;
        stDevOccGoodness /= (float) kGoodFrequencies.length;
        stDevGoodMinusBadness /= (float) kGoodFrequencies.length;
        stDevRelativeGoodMinusBadness /= (float) kGoodFrequencies.length;
        // Take the square root of the variances to obtain the standard
        // deviations.
        stDevOccBadness = Math.sqrt(stDevOccBadness);
        stDevOccGoodness = Math.sqrt(stDevOccGoodness);
        stDevGoodMinusBadness = Math.sqrt(stDevGoodMinusBadness);
        stDevRelativeGoodMinusBadness =
                Math.sqrt(stDevRelativeGoodMinusBadness);
    }

    /**
     * @return float[] The error-inducing neighbor occurrence counts.
     */
    public float[] getErrorInducingHubness() {
        return getErrorInducingHubness(kNeighbors[0].length);
    }

    /**
     * Calculates the error-inducing hubness counts, by counting how many of the
     * bad occurrences contribute to actual misclassification.
     *
     * @param k Integer that is the neighborhood size.
     * @param dataLabels int[] representing the external data labels.
     * @return The error-inducing neighbor occurrence counts.
     */
    public float[] getErrorInducingHubness(int k, int[] dataLabels) {
        int len = kNeighbors.length;
        int numClasses = ArrayUtil.max(dataLabels) + 1;
        float[] errOccFreqs = new float[len];
        float[] classCounts = new float[numClasses];
        float maxClassVote;
        int maxClassIndex;
        for (int i = 0; i < len; i++) {
            for (int j = 0; j < k; j++) {
                classCounts[dataLabels[kNeighbors[i][j]]]++;
            }
            maxClassVote = 0;
            maxClassIndex = 0;
            for (int c = 0; c < numClasses; c++) {
                if (classCounts[c] > maxClassVote) {
                    maxClassVote = classCounts[c];
                    maxClassIndex = c;
                }
            }
            if (maxClassIndex != dataLabels[i]) {
                // An error-inducing occurrence, as it would have contributed
                // to misclassification here.
                for (int j = 0; j < k; j++) {
                    if (dataLabels[i] != dataLabels[kNeighbors[i][j]]) {
                        errOccFreqs[kNeighbors[i][j]]++;
                    }
                }
            }
        }
        return errOccFreqs;
    }

    /**
     * Calculates the error-inducing hubness counts, by counting how many of the
     * bad occurrences contribute to actual misclassification.
     *
     * @param k Integer that is the neighborhood size.
     * @return The error-inducing neighbor occurrence counts.
     */
    public float[] getErrorInducingHubness(int k) {
        int len = kNeighbors.length;
        int numClasses = dset.countCategories();
        float[] errOccFreqs = new float[len];
        float[] classCounts = new float[numClasses];
        float maxClassVote;
        int maxClassIndex;
        for (int i = 0; i < len; i++) {
            for (int j = 0; j < k; j++) {
                classCounts[dset.getLabelOf(kNeighbors[i][j])]++;
            }
            maxClassVote = 0;
            maxClassIndex = 0;
            for (int c = 0; c < numClasses; c++) {
                if (classCounts[c] > maxClassVote) {
                    maxClassVote = classCounts[c];
                    maxClassIndex = c;
                }
            }
            if (maxClassIndex != dset.getLabelOf(i)) {
                // An error-inducing occurrence, as it would have contributed
                // to misclassification here.
                for (int j = 0; j < k; j++) {
                    if (dset.getLabelOf(i)
                            != dset.getLabelOf(kNeighbors[i][j])) {
                        errOccFreqs[kNeighbors[i][j]]++;
                    }
                }
            }
        }
        return errOccFreqs;
    }

    /**
     * @return DataInstance that is the major hub in the dataset.
     */
    public DataInstance getMajorHubInstance() {
        return dset.data.get(getHubIndex());
    }

    /**
     * @return Integer that is the index of the major hub in the dataset.
     */
    public int getHubIndex() {
        int maxFreq = 0;
        int maxIndex = -1;
        for (int i = 0; i < kNeighborFrequencies.length; i++) {
            if (kNeighborFrequencies[i] > maxFreq) {
                maxFreq = kNeighborFrequencies[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    /**
     * @return CombinedMetric object for distance calculations.
     */
    public CombinedMetric getCombinedMetric() {
        return cmet;
    }

    /**
     * This method constructs a NeighborSetFinder object from the approximate
     * NeighborSetFinder implementation via Lanczos bisections.
     *
     * @param appNSF AppKNNGraphLanczosBisection approximate kNN implementation.
     * @param calculateMissingDistances Boolean flag indicating whether to
     * calculate the missing distances or not.
     * @return NeighborSetFinder object based on the kNN sets in the approximate
     * kNN implementation.
     * @throws Exception
     */
    public static NeighborSetFinder constructFromAppFinder(
            AppKNNGraphLanczosBisection appNSF,
            boolean calculateMissingDistances) throws Exception {
        NeighborSetFinder nsf = new NeighborSetFinder();
        nsf.dset = appNSF.getDataSet();
        nsf.currK = appNSF.getK();
        nsf.distMatrix = appNSF.getDistances();
        CombinedMetric cmet = appNSF.getMetric();
        // Calculate the distance matrix, if specified.
        if (calculateMissingDistances) {
            for (int i = 0; i < nsf.dset.size(); i++) {
                for (int j = 0; j < nsf.distMatrix[i].length; j++) {
                    if (!appNSF.getDistanceFlags()[i][j]) {
                        nsf.distMatrix[i][j] = cmet.dist(nsf.dset.data.get(i),
                                nsf.dset.data.get(i + j + 1));
                    }
                }
            }
        }
        nsf.cmet = cmet;
        // Initialize the reverse neighbor lists.
        nsf.reverseNeighbors = new ArrayList[nsf.dset.size()];
        for (int i = 0; i < nsf.reverseNeighbors.length; i++) {
            nsf.reverseNeighbors[i] = new ArrayList<>(appNSF.getK() * 4);
        }
        nsf.distancesCalculated = true;
        // Get the kNN sets and the k-distances.
        nsf.kDistances = appNSF.getKdistances();
        nsf.kNeighbors = appNSF.getKneighbors();
        nsf.kNeighborFrequencies = new int[nsf.kNeighbors.length];
        nsf.kBadFrequencies = new int[nsf.kNeighbors.length];
        nsf.kGoodFrequencies = new int[nsf.kNeighbors.length];
        // Fill in the reverse neighbor lists and the occurrence frequency
        // counts.
        for (int i = 0; i < nsf.kNeighbors.length; i++) {
            for (int j = 0; j < appNSF.getK(); j++) {
                nsf.reverseNeighbors[nsf.kNeighbors[i][j]].add(i);
                nsf.kNeighborFrequencies[nsf.kNeighbors[i][j]]++;
                if (nsf.dset.data.get(i).getCategory() != nsf.dset.data.get(
                        nsf.kNeighbors[i][j]).getCategory()) {
                    nsf.kBadFrequencies[nsf.kNeighbors[i][j]]++;
                } else {
                    nsf.kGoodFrequencies[nsf.kNeighbors[i][j]]++;
                }
            }
        }
        // Calculate the occurrence frequency stats.
        nsf.meanOccFreq = 0;
        nsf.stDevOccFreq = 0;
        nsf.meanOccBadness = 0;
        nsf.stDevOccBadness = 0;
        nsf.meanOccGoodness = 0;
        nsf.stDevOccGoodness = 0;
        nsf.meanGoodMinusBadness = 0;
        nsf.stDevGoodMinusBadness = 0;
        nsf.meanRelativeGoodMinusBadness = 0;
        nsf.stDevRelativeGoodMinusBadness = 0;
        for (int i = 0; i < nsf.kBadFrequencies.length; i++) {
            nsf.meanOccFreq += nsf.kNeighborFrequencies[i];
            nsf.meanOccBadness += nsf.kBadFrequencies[i];
            nsf.meanOccGoodness += nsf.kGoodFrequencies[i];
            nsf.meanGoodMinusBadness += nsf.kGoodFrequencies[i]
                    - nsf.kBadFrequencies[i];
            if (nsf.kNeighborFrequencies[i] > 0) {
                nsf.meanRelativeGoodMinusBadness += ((nsf.kGoodFrequencies[i]
                        - nsf.kBadFrequencies[i])
                        / nsf.kNeighborFrequencies[i]);
            } else {
                nsf.meanRelativeGoodMinusBadness += 1;
            }
        }
        nsf.meanOccFreq /= (float) nsf.kNeighborFrequencies.length;
        nsf.meanOccBadness /= (float) nsf.kBadFrequencies.length;
        nsf.meanOccGoodness /= (float) nsf.kGoodFrequencies.length;
        nsf.meanGoodMinusBadness /= (float) nsf.kGoodFrequencies.length;
        nsf.meanRelativeGoodMinusBadness /= (float) nsf.kGoodFrequencies.length;
        for (int i = 0; i < nsf.kBadFrequencies.length; i++) {
            nsf.stDevOccFreq += ((nsf.meanOccFreq
                    - nsf.kNeighborFrequencies[i])
                    * (nsf.meanOccFreq - nsf.kNeighborFrequencies[i]));
            nsf.stDevOccBadness += ((nsf.meanOccBadness
                    - nsf.kBadFrequencies[i]) * (nsf.meanOccBadness
                    - nsf.kBadFrequencies[i]));
            nsf.stDevOccGoodness += ((nsf.meanOccGoodness
                    - nsf.kGoodFrequencies[i]) * (nsf.meanOccGoodness
                    - nsf.kGoodFrequencies[i]));
            nsf.stDevGoodMinusBadness += ((nsf.meanGoodMinusBadness
                    - (nsf.kGoodFrequencies[i] - nsf.kBadFrequencies[i]))
                    * (nsf.meanGoodMinusBadness - (nsf.kGoodFrequencies[i]
                    - nsf.kBadFrequencies[i])));
            if (nsf.kNeighborFrequencies[i] > 0) {
                nsf.stDevRelativeGoodMinusBadness +=
                        (nsf.meanRelativeGoodMinusBadness
                        - ((nsf.kGoodFrequencies[i] - nsf.kBadFrequencies[i])
                        / nsf.kNeighborFrequencies[i]))
                        * (nsf.meanRelativeGoodMinusBadness
                        - ((nsf.kGoodFrequencies[i] - nsf.kBadFrequencies[i])
                        / nsf.kNeighborFrequencies[i]));
            } else {
                nsf.stDevRelativeGoodMinusBadness +=
                        (nsf.meanRelativeGoodMinusBadness - 1)
                        * (nsf.meanRelativeGoodMinusBadness - 1);
            }
        }
        nsf.stDevOccFreq /= (float) nsf.kNeighborFrequencies.length;
        nsf.stDevOccBadness /= (float) nsf.kBadFrequencies.length;
        nsf.stDevOccGoodness /= (float) nsf.kGoodFrequencies.length;
        nsf.stDevGoodMinusBadness /= (float) nsf.kGoodFrequencies.length;
        nsf.stDevRelativeGoodMinusBadness /=
                (float) nsf.kGoodFrequencies.length;
        nsf.stDevOccFreq = Math.sqrt(nsf.stDevOccFreq);
        nsf.stDevOccBadness = Math.sqrt(nsf.stDevOccBadness);
        nsf.stDevOccGoodness = Math.sqrt(nsf.stDevOccGoodness);
        nsf.stDevGoodMinusBadness = Math.sqrt(nsf.stDevGoodMinusBadness);
        nsf.stDevRelativeGoodMinusBadness =
                Math.sqrt(nsf.stDevRelativeGoodMinusBadness);
        return nsf;
    }

    /**
     * @return ArrayList<Integer>[] that is an array of reverse neighbor lists
     * for all points in the data.
     */
    public ArrayList<Integer>[] getReverseNeighbors() {
        return reverseNeighbors;
    }

    public float getAvgDistToNNposition(int kPos) {
        int indexPos = kPos - 1;
        double sum = 0;
        for (int i = 0; i < kNeighbors.length; i++) {
            sum += kDistances[i][indexPos];
        }
        float result = (float) (sum / (double) kNeighbors.length);
        return result;
    }

    /**
     * Calculates and generates a NeighborSetFinder object that represents a
     * smaller k-range than the current object, on the prototype restriction in
     * case of instance selection.
     *
     * @param kSmaller Integer that is the neighborhood size to calculate the
     * kNN sets for.
     * @param prototypeIndexes ArrayList<Integer> of selected prototype indexes.
     * @param protoDistances float[][] representing the distance matrix on the
     * prototype set.
     * @param protoDSet DataSet that is the prototype data context.
     * @return NeighborSetFinder that is the calculated restriction.
     * @throws Exception
     */
    public NeighborSetFinder getSubNSF(int kSmaller,
            ArrayList<Integer> prototypeIndexes, float[][] protoDistances,
            DataSet protoDSet) throws Exception {
        // Initialize the prototype index maps.
        HashMap<Integer, Integer> originalIndexMap =
                new HashMap<>(prototypeIndexes.size() * 2);
        HashMap<Integer, Integer> protoMap =
                new HashMap<>(prototypeIndexes.size() * 2);
        int protoSize = prototypeIndexes.size();
        for (int i = 0; i < protoSize; i++) {
            originalIndexMap.put(i, prototypeIndexes.get(i));
            protoMap.put(prototypeIndexes.get(i), i);
        }
        // Initialize the resulting restriction.
        NeighborSetFinder nsfRestriction = new NeighborSetFinder(
                protoDSet, protoDistances, cmet);
        nsfRestriction.kNeighbors = new int[protoSize][kSmaller];
        nsfRestriction.kDistances = new float[protoSize][kSmaller];
        nsfRestriction.kCurrLen = new int[protoSize];
        nsfRestriction.kNeighborFrequencies = new int[protoSize];
        nsfRestriction.kBadFrequencies = new int[protoSize];
        nsfRestriction.kGoodFrequencies = new int[protoSize];
        // Intervals used for quick restriction calculations in the kNN sets.
        ArrayList<Integer> knnSetIntervals;
        int upperIndex, lowerIndex;
        int minIndVal, maxIndVal;
        // Neighborhood size.
        int k;
        // Auxiliary variable for kNN search.
        int l;
        for (int i = 0; i < protoSize; i++) {
            k = kCurrLen[prototypeIndexes.get(i)];
            nsfRestriction.kCurrLen[i] = 0;
            knnSetIntervals = new ArrayList(kSmaller + 2);
            knnSetIntervals.add(-1);
            for (int j = 0; j < k; j++) {
                if (protoMap.containsKey(
                        kNeighbors[prototypeIndexes.get(i)][j])) {
                    nsfRestriction.kNeighbors[i][nsfRestriction.kCurrLen[i]] =
                            protoMap.get(
                            kNeighbors[prototypeIndexes.get(i)][j]);
                    nsfRestriction.kDistances[i][nsfRestriction.kCurrLen[i]] =
                            kDistances[prototypeIndexes.get(i)][j];
                    knnSetIntervals.add(
                            nsfRestriction.kNeighbors[i][
                            nsfRestriction.kCurrLen[i]]);
                    nsfRestriction.kCurrLen[i]++;
                }
                if (nsfRestriction.kCurrLen[i] >= kSmaller) {
                    break;
                }
            }
            knnSetIntervals.add(protoSize + 1);
            Collections.sort(knnSetIntervals);
            if (nsfRestriction.kCurrLen[i] < kSmaller) {
                int iSizeRed = knnSetIntervals.size() - 1;
                for (int ind = 0; ind < iSizeRed; ind++) {
                    lowerIndex = knnSetIntervals.get(ind);
                    upperIndex = knnSetIntervals.get(ind + 1);
                    for (int j = lowerIndex + 1; j < upperIndex - 1; j++) {
                        if (i != j) {
                            minIndVal = Math.min(i, j);
                            maxIndVal = Math.max(i, j);

                            if (nsfRestriction.kCurrLen[i] > 0) {
                                if (nsfRestriction.kCurrLen[i] == kSmaller) {
                                    if (protoDistances[minIndVal][
                                            maxIndVal - minIndVal - 1]
                                            < nsfRestriction.kDistances[i][
                                            nsfRestriction.kCurrLen[i] - 1]) {
                                        // Search and insert.
                                        l = kSmaller - 1;
                                        while ((l >= 1) && protoDistances[
                                                minIndVal][maxIndVal - minIndVal
                                                - 1]
                                                < nsfRestriction.kDistances[i][
                                                l - 1]) {
                                            nsfRestriction.kDistances[i][l] =
                                                    nsfRestriction.kDistances[
                                                    i][l - 1];
                                            nsfRestriction.kNeighbors[i][l] =
                                                    nsfRestriction.kNeighbors[
                                                    i][l - 1];
                                            l--;
                                        }
                                        nsfRestriction.kDistances[i][l] =
                                                protoDistances[minIndVal][
                                                maxIndVal - minIndVal - 1];
                                        nsfRestriction.kNeighbors[i][l] = j;
                                    }
                                } else {
                                    if (protoDistances[minIndVal][maxIndVal
                                            - minIndVal - 1]
                                            < nsfRestriction.kDistances[i][
                                            nsfRestriction.kCurrLen[i] - 1]) {
                                        // Search and insert.
                                        l = nsfRestriction.kCurrLen[i] - 1;
                                        nsfRestriction.kDistances[i][
                                                nsfRestriction.kCurrLen[i]] =
                                                nsfRestriction.kDistances[i][
                                                nsfRestriction.kCurrLen[i] - 1];
                                        nsfRestriction.kNeighbors[i][
                                                nsfRestriction.kCurrLen[i]] =
                                                nsfRestriction.kNeighbors[i][
                                                nsfRestriction.kCurrLen[i] - 1];
                                        while ((l >= 1) && protoDistances[
                                                minIndVal][maxIndVal - minIndVal
                                                - 1]
                                                < nsfRestriction.kDistances[i][
                                                l - 1]) {
                                            nsfRestriction.kDistances[i][l] =
                                                    nsfRestriction.kDistances[
                                                    i][l - 1];
                                            nsfRestriction.kNeighbors[i][l] =
                                                    nsfRestriction.kNeighbors[
                                                    i][l - 1];
                                            l--;
                                        }
                                        nsfRestriction.kDistances[i][l] =
                                                protoDistances[minIndVal][
                                                maxIndVal - minIndVal - 1];
                                        nsfRestriction.kNeighbors[i][l] = j;
                                        nsfRestriction.kCurrLen[i]++;
                                    } else {
                                        nsfRestriction.kDistances[i][
                                                nsfRestriction.kCurrLen[i]] =
                                                protoDistances[minIndVal][
                                                maxIndVal - minIndVal - 1];
                                        nsfRestriction.kNeighbors[i][
                                                nsfRestriction.kCurrLen[i]] = j;
                                        nsfRestriction.kCurrLen[i]++;
                                    }
                                }
                            } else {
                                nsfRestriction.kDistances[i][0] =
                                        protoDistances[minIndVal][maxIndVal
                                        - minIndVal - 1];
                                nsfRestriction.kNeighbors[i][0] = j;
                                nsfRestriction.kCurrLen[i] = 1;
                            }
                        }
                    }
                }
            }
        }
        nsfRestriction.currK = kSmaller;
        // Calculate the neighbor occurrence frequencies and the reverse
        // neighbor sets for the calculated restriction.
        int currClass;
        int nClass;
        nsfRestriction.reverseNeighbors = new ArrayList[protoSize];
        for (int i = 0; i < protoSize; i++) {
            nsfRestriction.reverseNeighbors[i] = new ArrayList<>(10 * kSmaller);
        }
        for (int i = 0; i < protoSize; i++) {
            currClass = protoDSet.getLabelOf(i);

            for (int j = 0; j < kSmaller; j++) {
                nClass = protoDSet.getLabelOf(nsfRestriction.kNeighbors[i][j]);
                nsfRestriction.reverseNeighbors[
                        nsfRestriction.kNeighbors[i][j]].add(i);
                if (nClass == currClass) {
                    nsfRestriction.kGoodFrequencies[
                            nsfRestriction.kNeighbors[i][j]]++;
                } else {
                    nsfRestriction.kBadFrequencies[
                            nsfRestriction.kNeighbors[i][j]]++;
                }
                nsfRestriction.kNeighborFrequencies[
                        nsfRestriction.kNeighbors[i][j]]++;
            }
        }

        nsfRestriction.completeNeighborSets(kSmaller, null);
        return nsfRestriction;
    }

    /**
     * Calculates and generates a NeighborSetFinder object that represents a
     * smaller k-range than the current object.
     *
     * @param kSmaller Integer that is the neighborhood size to re-calculate the
     * kNN sets for.
     * @return NeighborSetFinder that is the restriction on a smaller
     * neighborhood size.
     */
    public NeighborSetFinder getSubNSF(int kSmaller) {
        NeighborSetFinder nsfRestriction =
                new NeighborSetFinder(dset, distMatrix, cmet);
        nsfRestriction.kNeighbors = new int[dset.size()][];
        nsfRestriction.kDistances = new float[dset.size()][];
        nsfRestriction.kCurrLen = new int[dset.size()];
        // Copy the full occurrence counts - we will subtract the counts for
        // the difference between the k values below.
        nsfRestriction.kNeighborFrequencies =
                Arrays.copyOfRange(kNeighborFrequencies, 0, dset.size());
        nsfRestriction.kBadFrequencies =
                Arrays.copyOfRange(kBadFrequencies, 0, dset.size());
        nsfRestriction.kGoodFrequencies =
                Arrays.copyOfRange(kGoodFrequencies, 0, dset.size());
        for (int i = 0; i < dset.size(); i++) {
            nsfRestriction.kNeighbors[i] =
                    Arrays.copyOfRange(kNeighbors[i], 0, kSmaller);
            nsfRestriction.kDistances[i] =
                    Arrays.copyOfRange(kDistances[i], 0, kSmaller);
        }
        Arrays.fill(kCurrLen, kSmaller);
        nsfRestriction.currK = kSmaller;
        nsfRestriction.reverseNeighbors = new ArrayList[dset.size()];
        for (int i = 0; i < dset.size(); i++) {
            nsfRestriction.reverseNeighbors[i] = new ArrayList<>(10 * kSmaller);
        }
        for (int i = 0; i < dset.size(); i++) {
            for (int j = 0; j < kSmaller; j++) {
                nsfRestriction.reverseNeighbors[kNeighbors[i][j]].add(i);
            }
        }
        // Subtract the additional occurrence counts in the k-range between the
        // two k values.
        for (int i = 0; i < dset.size(); i++) {
            for (int j = kSmaller; j < kNeighbors[0].length; j++) {
                nsfRestriction.kNeighborFrequencies[kNeighbors[i][j]]--;
                if (dset.data.get(i).getCategory()
                        == dset.data.get(kNeighbors[i][j]).getCategory()) {
                    nsfRestriction.kGoodFrequencies[kNeighbors[i][j]]--;
                } else {
                    nsfRestriction.kBadFrequencies[kNeighbors[i][j]]--;
                }
            }
        }
        // Calculate the neighbor occurrence frequency stats.
        nsfRestriction.meanOccFreq = 0;
        nsfRestriction.stDevOccFreq = 0;
        nsfRestriction.meanOccBadness = 0;
        nsfRestriction.stDevOccBadness = 0;
        nsfRestriction.meanOccGoodness = 0;
        nsfRestriction.stDevOccGoodness = 0;
        nsfRestriction.meanGoodMinusBadness = 0;
        nsfRestriction.meanRelativeGoodMinusBadness = 0;
        nsfRestriction.stDevGoodMinusBadness = 0;
        nsfRestriction.stDevRelativeGoodMinusBadness = 0;
        for (int i = 0; i < nsfRestriction.kBadFrequencies.length; i++) {
            nsfRestriction.meanOccFreq +=
                    nsfRestriction.kNeighborFrequencies[i];
            nsfRestriction.meanOccBadness += nsfRestriction.kBadFrequencies[i];
            nsfRestriction.meanOccGoodness +=
                    nsfRestriction.kGoodFrequencies[i];
            nsfRestriction.meanGoodMinusBadness +=
                    nsfRestriction.kGoodFrequencies[i]
                    - nsfRestriction.kBadFrequencies[i];
            if (kNeighborFrequencies[i] > 0) {
                nsfRestriction.meanRelativeGoodMinusBadness +=
                        ((nsfRestriction.kGoodFrequencies[i]
                        - nsfRestriction.kBadFrequencies[i])
                        / kNeighborFrequencies[i]);
            } else {
                nsfRestriction.meanRelativeGoodMinusBadness += 1;
            }
        }
        nsfRestriction.meanOccFreq /=
                (float) nsfRestriction.kNeighborFrequencies.length;
        nsfRestriction.meanOccBadness /=
                (float) nsfRestriction.kBadFrequencies.length;
        nsfRestriction.meanOccGoodness /=
                (float) nsfRestriction.kGoodFrequencies.length;
        nsfRestriction.meanGoodMinusBadness /=
                (float) nsfRestriction.kGoodFrequencies.length;
        nsfRestriction.meanRelativeGoodMinusBadness /=
                (float) nsfRestriction.kGoodFrequencies.length;
        for (int i = 0; i < nsfRestriction.kBadFrequencies.length; i++) {
            nsfRestriction.stDevOccFreq +=
                    ((nsfRestriction.meanOccFreq
                    - nsfRestriction.kNeighborFrequencies[i])
                    * (nsfRestriction.meanOccFreq
                    - nsfRestriction.kNeighborFrequencies[i]));
            nsfRestriction.stDevOccBadness +=
                    ((nsfRestriction.meanOccBadness
                    - nsfRestriction.kBadFrequencies[i])
                    * (nsfRestriction.meanOccBadness
                    - nsfRestriction.kBadFrequencies[i]));
            nsfRestriction.stDevOccGoodness +=
                    ((nsfRestriction.meanOccGoodness
                    - nsfRestriction.kGoodFrequencies[i])
                    * (nsfRestriction.meanOccGoodness
                    - nsfRestriction.kGoodFrequencies[i]));
            nsfRestriction.stDevGoodMinusBadness +=
                    ((nsfRestriction.meanGoodMinusBadness
                    - (nsfRestriction.kGoodFrequencies[i]
                    - nsfRestriction.kBadFrequencies[i]))
                    * (nsfRestriction.meanGoodMinusBadness
                    - (nsfRestriction.kGoodFrequencies[i]
                    - nsfRestriction.kBadFrequencies[i])));
            if (kNeighborFrequencies[i] > 0) {
                nsfRestriction.stDevRelativeGoodMinusBadness +=
                        (nsfRestriction.meanRelativeGoodMinusBadness
                        - ((nsfRestriction.kGoodFrequencies[i]
                        - nsfRestriction.kBadFrequencies[i])
                        / kNeighborFrequencies[i]))
                        * (nsfRestriction.meanRelativeGoodMinusBadness
                        - ((nsfRestriction.kGoodFrequencies[i]
                        - nsfRestriction.kBadFrequencies[i])
                        / kNeighborFrequencies[i]));
            } else {
                nsfRestriction.stDevRelativeGoodMinusBadness +=
                        (nsfRestriction.meanRelativeGoodMinusBadness)
                        * (nsfRestriction.meanRelativeGoodMinusBadness - 1);
            }
        }
        // Normalize the averages.
        nsfRestriction.stDevOccFreq /=
                (float) nsfRestriction.kNeighborFrequencies.length;
        nsfRestriction.stDevOccBadness /=
                (float) nsfRestriction.kBadFrequencies.length;
        nsfRestriction.stDevOccGoodness /=
                (float) nsfRestriction.kGoodFrequencies.length;
        nsfRestriction.stDevGoodMinusBadness /=
                (float) nsfRestriction.kGoodFrequencies.length;
        nsfRestriction.stDevRelativeGoodMinusBadness /=
                (float) nsfRestriction.kGoodFrequencies.length;
        // Take the square root of the variances to obtain the
        // standard deviations.
        nsfRestriction.stDevOccFreq =
                Math.sqrt(nsfRestriction.stDevOccFreq);
        nsfRestriction.stDevOccBadness =
                Math.sqrt(nsfRestriction.stDevOccBadness);
        nsfRestriction.stDevOccGoodness =
                Math.sqrt(nsfRestriction.stDevOccGoodness);
        nsfRestriction.stDevGoodMinusBadness =
                Math.sqrt(nsfRestriction.stDevGoodMinusBadness);
        nsfRestriction.stDevRelativeGoodMinusBadness =
                Math.sqrt(nsfRestriction.stDevRelativeGoodMinusBadness);
        return nsfRestriction;
    }

    /**
     * This method calculates the class-to-class neighbor occurrence probability
     * matrix for use in the Bayesian hubness-aware classification models.
     *
     * @param k Integer that is the neighborhood size.
     * @param numClasses Integer that is the number of classes in the data.
     * @param laplaceEstimator Float value that is the Laplace estimator for
     * distribution smoothing.
     * @param extendByElement Boolean flag indicating whether to use the query
     * point as its own 0-th nearest neighbor.
     * @return float[][] representing the class-to-class neighbor occurrence
     * probability matrix for use in the Bayesian hubness-aware classification
     * models.
     */
    public float[][] getGlobalClassToClassForKforBayerisan(int k,
            int numClasses, float laplaceEstimator, boolean extendByElement) {
        float[][] classToClassPriors = new float[numClasses][numClasses];
        float[] classPriors = new float[numClasses];
        for (int i = 0; i < dset.size(); i++) {
            int currClass = dset.data.get(i).getCategory();
            classPriors[currClass]++;
            if (extendByElement) {
                classToClassPriors[currClass][currClass]++;
            }
            for (int kInd = 0; kInd < k; kInd++) {
                classToClassPriors[dset.data.get(kNeighbors[i][kInd]).
                        getCategory()][currClass]++;
            }
        }
        float laplaceTotal = numClasses * laplaceEstimator;
        if (extendByElement) {
            for (int cFirst = 0; cFirst < numClasses; cFirst++) {
                for (int cSecond = 0; cSecond < numClasses; cSecond++) {
                    classToClassPriors[cFirst][cSecond] += laplaceEstimator;
                    classToClassPriors[cFirst][cSecond] /= ((k + 1)
                            * classPriors[cSecond] + laplaceTotal);
                }
            }
        } else {
            for (int cFirst = 0; cFirst < numClasses; cFirst++) {
                for (int cSecond = 0; cSecond < numClasses; cSecond++) {
                    classToClassPriors[cFirst][cSecond] += laplaceEstimator;
                    classToClassPriors[cFirst][cSecond] /=
                            (k * classPriors[cSecond] + laplaceTotal);
                }
            }
        }
        return classToClassPriors;
    }

    /**
     * This method calculates the class-to-class neighbor occurrence probability
     * matrix for use in the fuzzy hubness-aware classification models.
     *
     * @param k Integer that is the neighborhood size.
     * @param numClasses Integer that is the number of classes in the data.
     * @param laplaceEstimator Float value that is the Laplace estimator for
     * distribution smoothing.
     * @param extendByElement Boolean flag indicating whether to use the query
     * point as its own 0-th nearest neighbor.
     * @return float[][] representing the class-to-class neighbor occurrence
     * probability matrix for use in the fuzzy hubness-aware classification
     * models.
     */
    public float[][] getGlobalClassToClassForKforFuzzy(int k, int numClasses,
            float laplaceEstimator, boolean extendByElement) {
        float[][] classToClassPriors = new float[numClasses][numClasses];
        float[] classHubnessSums = new float[numClasses];
        for (int i = 0; i < dset.size(); i++) {
            int currClass = dset.data.get(i).getCategory();
            if (extendByElement) {
                classToClassPriors[currClass][currClass]++;
                classHubnessSums[currClass]++;
            }
            for (int kInd = 0; kInd < k; kInd++) {
                classToClassPriors[dset.data.get(
                        kNeighbors[i][kInd]).getCategory()][currClass]++;
                classHubnessSums[dset.data.get(
                        kNeighbors[i][kInd]).getCategory()]++;
            }
        }
        float laplaceTotal = numClasses * laplaceEstimator;
        for (int cFirst = 0; cFirst < numClasses; cFirst++) {
            for (int cSecond = 0; cSecond < numClasses; cSecond++) {
                classToClassPriors[cFirst][cSecond] += laplaceEstimator;
                classToClassPriors[cFirst][cSecond] /=
                        (classHubnessSums[cFirst] + laplaceTotal);
            }
        }
        return classToClassPriors;
    }

    /**
     * This method calculates the class-to-class neighbor occurrence probability
     * matrix, non-normalized.
     *
     * @param k Integer that is the neighborhood size.
     * @param numClasses Integer that is the number of classes in the data.
     * @return float[][] representing the class-to-class neighbor occurrence
     * probability matrix, non-normalized.
     */
    public float[][] getGlobalClassToClassNonNormalized(int k, int numClasses) {
        float[][] classToClass = new float[numClasses][numClasses];
        for (int i = 0; i < dset.size(); i++) {
            int currClass = dset.data.get(i).getCategory();
            for (int kInd = 0; kInd < k; kInd++) {
                classToClass[dset.data.get(
                        kNeighbors[i][kInd]).getCategory()][currClass]++;
            }
        }
        return classToClass;
    }

    /**
     * This method calculates the class-to-class neighbor occurrence probability
     * matrix for use in the fuzzy hubness-aware classification models,
     * restricted on points that have bad occurrences.
     *
     * @param k Integer that is the neighborhood size.
     * @param numClasses Integer that is the number of classes in the data.
     * @param laplaceEstimator Float value that is the Laplace estimator for
     * distribution smoothing.
     * @param extendByElement Boolean flag indicating whether to use the query
     * point as its own 0-th nearest neighbor.
     * @return float[][] representing the class-to-class neighbor occurrence
     * probability matrix for use in the fuzzy hubness-aware classification
     * models, restricted on points that have bad occurrences.
     */
    public float[][] getGlobalClassToClassForKforFuzzyRestrictOnBad(int k,
            int numClasses, float laplaceEstimator, boolean extendByElement) {
        float[][] classToClassPriors = new float[numClasses][numClasses];
        float[] classHubnessSums = new float[numClasses];
        for (int i = 0; i < dset.size(); i++) {
            int currClass = dset.data.get(i).getCategory();
            if (extendByElement) {
                classToClassPriors[currClass][currClass]++;
                classHubnessSums[currClass]++;
            }
            for (int kInd = 0; kInd < k; kInd++) {
                if (kBadFrequencies[kNeighbors[i][kInd]] > 0) {
                    classToClassPriors[dset.data.get(
                            kNeighbors[i][kInd]).getCategory()][currClass]++;
                    classHubnessSums[dset.data.get(
                            kNeighbors[i][kInd]).getCategory()]++;
                }
            }
        }
        float laplaceTotal = numClasses * laplaceEstimator;
        for (int cFirst = 0; cFirst < numClasses; cFirst++) {
            for (int cSecond = 0; cSecond < numClasses; cSecond++) {
                classToClassPriors[cFirst][cSecond] += laplaceEstimator;
                classToClassPriors[cFirst][cSecond] /=
                        (classHubnessSums[cFirst] + laplaceTotal);
            }
        }
        return classToClassPriors;
    }

    /**
     * This method calculates the class-conditional neighbor occurrence
     * probability matrix for use in the Bayesian hubness-aware classification
     * models. This means that the values are normalized to represent the
     * probability of neighbor given the class, instead of class given the
     * neighbor.
     *
     * @param k Integer that is the neighborhood size.
     * @param numClasses Integer that is the number of classes in the data.
     * @param laplaceEstimator Float value that is the Laplace estimator for
     * distribution smoothing.
     * @param extendByElement Boolean flag indicating whether to use the query
     * point as its own 0-th nearest neighbor.
     * @return float[][] representing the class-conditional neighbor occurrence
     * probability matrix for use in the Bayesian hubness-aware classification
     * models.
     */
    public float[][] getClassDataNeighborRelationForKforBayesian(int k,
            int numClasses, float laplaceEstimator, boolean extendByElement) {
        float[][] classDataKNeighborRelation =
                new float[numClasses][dset.size()];
        float[] classPriors = new float[numClasses];
        for (int i = 0; i < dset.size(); i++) {
            int currClass = dset.data.get(i).getCategory();
            classPriors[currClass]++;
            if (extendByElement) {
                classDataKNeighborRelation[currClass][i]++;
            }
            for (int kInd = 0; kInd < k; kInd++) {
                classDataKNeighborRelation[currClass][kNeighbors[i][kInd]]++;
            }
        }
        float laplaceTotal = dset.size() * laplaceEstimator;
        for (int cFirst = 0; cFirst < numClasses; cFirst++) {
            for (int cSecond = 0; cSecond < dset.size(); cSecond++) {
                classDataKNeighborRelation[cFirst][cSecond] += laplaceEstimator;
                classDataKNeighborRelation[cFirst][cSecond] /=
                        (k * classPriors[cFirst] + laplaceTotal);
            }
        }
        return classDataKNeighborRelation;
    }

    /**
     * This method calculates the class-conditional neighbor occurrence
     * probability matrix for use in the fuzzy hubness-aware classification
     * models. This means that the values are normalized to represent the
     * probability of neighbor given the class, instead of class given the
     * neighbor.
     *
     * @param k Integer that is the neighborhood size.
     * @param numClasses Integer that is the number of classes in the data.
     * @param laplaceEstimator Float value that is the Laplace estimator for
     * distribution smoothing.
     * @param extendByElement Boolean flag indicating whether to use the query
     * point as its own 0-th nearest neighbor.
     * @return float[][] representing the class-conditional neighbor occurrence
     * probability matrix for use in the fuzzy hubness-aware classification
     * models.
     */
    public float[][] getFuzzyClassDataNeighborRelation(int k, int numClasses,
            float laplaceEstimator, boolean extendByElement) {
        float[][] classDataKNeighborRelation =
                new float[numClasses][dset.size()];
        for (int i = 0; i < dset.size(); i++) {
            int currClass = dset.data.get(i).getCategory();
            if (extendByElement) {
                classDataKNeighborRelation[currClass][i]++;
            }
            for (int kInd = 0; kInd < k; kInd++) {
                classDataKNeighborRelation[currClass][kNeighbors[i][kInd]]++;
            }
        }
        float laplaceTotal = numClasses * laplaceEstimator;
        recalculateStatsForSmallerK(k);
        int[] neighbOccFreqs = getNeighborFrequencies();
        for (int cFirst = 0; cFirst < numClasses; cFirst++) {
            for (int cSecond = 0; cSecond < dset.size(); cSecond++) {
                classDataKNeighborRelation[cFirst][cSecond] += laplaceEstimator;
                classDataKNeighborRelation[cFirst][cSecond] /=
                        (neighbOccFreqs[cSecond] + 1 + laplaceTotal);
            }
        }
        recalculateStatsForSmallerK(kNeighbors[0].length);
        return classDataKNeighborRelation;
    }

    /**
     * This method calculates the class-conditional neighbor occurrence counts.
     * In the results, the cell [i,j] holds the occurrence count of point i in
     * class j.
     *
     * @param k Integer that is the neighborhood size.
     * @param numClasses Integer that is the number of classes in the data.
     * @param extendByElement Boolean flag indicating whether to use the query
     * point as its own 0-th nearest neighbor.
     * @return float[][] representing the class-conditional neighbor occurrence
     * counts. In the results, the cell [i,j] holds the occurrence count of
     * point i in class j.
     */
    public float[][] getDataClassNeighborRelationNonNormalized(int k,
            int numClasses, boolean extendByElement) {
        float[][] dataClassKNeighborRelation =
                new float[dset.size()][numClasses];
        for (int i = 0; i < dset.size(); i++) {
            int currClass = dset.data.get(i).getCategory();
            if (extendByElement) {
                dataClassKNeighborRelation[i][currClass]++;
            }
            for (int kInd = 0; kInd < k; kInd++) {
                dataClassKNeighborRelation[kNeighbors[i][kInd]][currClass]++;
            }
        }
        return dataClassKNeighborRelation;
    }

    /**
     * This method calculates the class-conditional neighbor occurrence counts.
     * In the results, the cell [i,j] holds the occurrence count of point j in
     * class i.
     *
     * @param k Integer that is the neighborhood size.
     * @param numClasses Integer that is the number of classes in the data.
     * @param extendByElement Boolean flag indicating whether to use the query
     * point as its own 0-th nearest neighbor.
     * @return float[][] representing the class-conditional neighbor occurrence
     * counts. In the results, the cell [i,j] holds the occurrence count of
     * point j in class i.
     */
    public float[][] getClassDataNeighborRelationNonNormalized(int k,
            int numClasses, boolean extendByElement) {
        float[][] classDataKNeighborRelation =
                new float[numClasses][dset.size()];
        for (int i = 0; i < dset.size(); i++) {
            int currClass = dset.data.get(i).getCategory();
            if (extendByElement) {
                classDataKNeighborRelation[currClass][i]++;
            }
            for (int kInd = 0; kInd < k; kInd++) {
                classDataKNeighborRelation[currClass][kNeighbors[i][kInd]]++;
            }
        }
        return classDataKNeighborRelation;
    }

    /**
     * @return float[][] representing an array of occurrence frequency arrays,
     * for neighborhood sizes from 1 to the maximum neighborhood size supported
     * by the current kNN sets, which is their length.
     */
    public float[][] getOccFreqsForAllK() {
        int k = kNeighbors[0].length;
        float[][] occFreqsAllK = new float[k][kNeighbors.length];
        for (int kInd = 0; kInd < k; kInd++) {
            recalculateStatsForSmallerK(kInd + 1);
            for (int i = 0; i < kNeighbors.length; i++) {
                occFreqsAllK[kInd][i] = getNeighborFrequencies()[i];
            }
        }
        return occFreqsAllK;
    }

    /**
     * This method queries the dataset to determine the neighbors of a
     * particular point from the dataset.
     *
     * @param dset DataSet object to query.
     * @param instanceIndex Integer that is the index of the data instance that
     * is the kNN query.
     * @param neighborhoodSize Integer that is the desired neighborhood size.
     * @param cmet CombinedMetric object for distance calculations.
     * @return int[] that contains the indexes of the k-nearest neighbors for
     * the query point.
     * @throws Exception
     */
    public static int[] getIndexesOfNeighbors(DataSet dset, int instanceIndex,
            int neighborhoodSize, CombinedMetric cmet) throws Exception {
        int[] neighbors = new int[neighborhoodSize];
        float currDist;
        int kCurrLen = 0;
        float[] nDists = new float[neighborhoodSize];
        Arrays.fill(nDists, Float.MAX_VALUE);
        int l;
        DataInstance instance = dset.data.get(instanceIndex);
        // Check the first half of the points, with the index value below the
        // query index value.
        for (int i = 0; i < instanceIndex; i++) {
            currDist = cmet.dist(instance, dset.data.get(i));
            if (kCurrLen > 0) {
                if (kCurrLen == neighborhoodSize) {
                    if (currDist < nDists[kCurrLen - 1]) {
                        // Search and inser.
                        l = neighborhoodSize - 1;
                        while ((l >= 1) && currDist < nDists[l - 1]) {
                            nDists[l] = nDists[l - 1];
                            neighbors[l] = neighbors[l - 1];
                            l--;
                        }
                        nDists[l] = currDist;
                        neighbors[l] = i;

                    }
                } else {
                    if (currDist < nDists[kCurrLen - 1]) {
                        // Search and insert.
                        l = kCurrLen - 1;
                        nDists[kCurrLen] = nDists[kCurrLen - 1];
                        neighbors[kCurrLen] = neighbors[kCurrLen - 1];
                        while ((l >= 1) && currDist < nDists[l]) {
                            nDists[l] = nDists[l - 1];
                            neighbors[l] = neighbors[l - 1];
                            l--;
                        }
                        nDists[l + 1] = currDist;
                        neighbors[l + 1] = i;
                        kCurrLen++;
                    } else {
                        nDists[kCurrLen] = currDist;
                        neighbors[kCurrLen] = i;
                        kCurrLen++;
                    }
                }
            } else {
                nDists[0] = currDist;
                nDists[0] = i;
                kCurrLen = 1;
            }
        }
        // Check the second half of the points, with the index value above the
        // query index value.
        for (int i = instanceIndex + 1; i < dset.size(); i++) {
            currDist = cmet.dist(instance, dset.data.get(i));
            if (kCurrLen > 0) {
                if (kCurrLen == neighborhoodSize) {
                    if (currDist < nDists[kCurrLen - 1]) {
                        // Search and insert.
                        l = neighborhoodSize - 1;
                        while ((l >= 1) && currDist < nDists[l - 1]) {
                            nDists[l] = nDists[l - 1];
                            neighbors[l] = neighbors[l - 1];
                            l--;
                        }
                        nDists[l] = currDist;
                        neighbors[l] = i;
                    }
                } else {
                    if (currDist < nDists[kCurrLen - 1]) {
                        // Search and insert.
                        l = kCurrLen - 1;
                        nDists[kCurrLen] = nDists[kCurrLen - 1];
                        neighbors[kCurrLen] = neighbors[kCurrLen - 1];
                        while ((l >= 1) && currDist < nDists[l - 1]) {
                            nDists[l] = nDists[l - 1];
                            neighbors[l] = neighbors[l - 1];
                            l--;
                        }
                        nDists[l] = currDist;
                        neighbors[l] = i;
                        kCurrLen++;
                    } else {
                        nDists[kCurrLen] = currDist;
                        neighbors[kCurrLen] = i;
                        kCurrLen++;
                    }
                }
            } else {
                nDists[0] = currDist;
                neighbors[0] = i;
                kCurrLen = 1;
            }
        }
        return neighbors;
    }

    /**
     * This method queries the dataset to determine the neighbors of a
     * particular point that does not belong to the dataset.
     *
     * @param dset DataSet object to query.
     * @param instance DataInstance that is the query point.
     * @param neighborhoodSize Integer that is the desired neighborhood size.
     * @param distances float[] representing the distances from the query point
     * to the points in the provided DataSet.
     * @return int[] that contains the indexes of the k-nearest neighbors for
     * the query point.
     * @throws Exception
     */
    public static int[] getIndexesOfNeighbors(DataSet dset,
            DataInstance instance, int neighborhoodSize, float[] distances)
            throws Exception {
        int[] neighbors = new int[neighborhoodSize];
        float currDist;
        int kCurrLen = 0;
        float[] nDists = new float[neighborhoodSize];
        Arrays.fill(nDists, Float.MAX_VALUE);
        int l;
        for (int i = 0; i < dset.size(); i++) {
            currDist = distances[i];
            if (kCurrLen > 0) {
                if (kCurrLen == neighborhoodSize) {
                    if (currDist < nDists[kCurrLen - 1]) {
                        // Search and insert.
                        l = neighborhoodSize - 1;
                        while ((l >= 1) && currDist < nDists[l - 1]) {
                            nDists[l] = nDists[l - 1];
                            neighbors[l] = neighbors[l - 1];
                            l--;
                        }
                        nDists[l] = currDist;
                        neighbors[l] = i;
                    }
                } else {
                    if (currDist < nDists[kCurrLen - 1]) {
                        // Search and insert.
                        l = kCurrLen - 1;
                        nDists[kCurrLen] = nDists[kCurrLen - 1];
                        neighbors[kCurrLen] = neighbors[kCurrLen - 1];
                        while ((l >= 1) && currDist < nDists[l - 1]) {
                            nDists[l] = nDists[l - 1];
                            neighbors[l] = neighbors[l - 1];
                            l--;
                        }
                        nDists[l] = currDist;
                        neighbors[l] = i;
                        kCurrLen++;
                    } else {
                        nDists[kCurrLen] = currDist;
                        neighbors[kCurrLen] = i;
                        kCurrLen++;
                    }
                }
            } else {
                nDists[0] = currDist;
                neighbors[0] = i;
                kCurrLen = 1;
            }
        }
        return neighbors;
    }

    /**
     * Get the indexes of neighbors from the training set for all the instances
     * in the test set.
     *
     * @param trainingDSet DataSet object that is the training data.
     * @param testDSet DataSet object that is the test data.
     * @param neighborhoodSize Integer that is the neighborhood size.
     * @param testToTrainDist Float 2D array of distances from test to training.
     * @return Integer 2D array of indexes of neighbors from the training data
     * of points in the test data.
     * @throws Exception
     */
    public static int[][] getIndexesOfNeighbors(DataSet trainingDSet,
            DataSet testDSet, int neighborhoodSize, float[][] testToTrainDist)
            throws Exception {
        if (trainingDSet == null || testDSet == null) {
            return null;
        }
        int[][] neighborIndexes = new int[testDSet.size()][];
        for (int i = 0; i < testDSet.size(); i++) {
            DataInstance instance = testDSet.getInstance(i);
            neighborIndexes[i] = getIndexesOfNeighbors(trainingDSet,
                    instance, neighborhoodSize, testToTrainDist[i]);
        }
        return neighborIndexes;
    }

    /**
     * Get the indexes of neighbors from the training set for all the instances
     * in the test set.
     *
     * @param trainingDSet DataSet object that is the training data.
     * @param testDSet DataSet object that is the test data.
     * @param neighborhoodSize Integer that is the neighborhood size.
     * @param cmet CombinedMetric object for distance calculations.
     * @return Integer 2D array of indexes of neighbors from the training data
     * of points in the test data.
     * @throws Exception
     */
    public static int[][] getIndexesOfNeighbors(DataSet trainingDSet,
            DataSet testDSet, int neighborhoodSize, CombinedMetric cmet)
            throws Exception {
        if (trainingDSet == null || testDSet == null) {
            return null;
        }
        int[][] neighborIndexes = new int[testDSet.size()][];
        for (int i = 0; i < testDSet.size(); i++) {
            DataInstance instance = testDSet.getInstance(i);
            neighborIndexes[i] = getIndexesOfNeighbors(trainingDSet,
                    instance, neighborhoodSize, cmet);
        }
        return neighborIndexes;
    }

    /**
     * This method queries the dataset to determine the neighbors of a
     * particular point that does not belong to the dataset.
     *
     * @param dset DataSet object to query.
     * @param instance DataInstance that is the query point.
     * @param neighborhoodSize Integer that is the desired neighborhood size.
     * @param cmet CombinedMetric object for distance calculations. to the
     * points in the provided DataSet.
     * @return int[] that contains the indexes of the k-nearest neighbors for
     * the query point.
     * @throws Exception
     */
    public static int[] getIndexesOfNeighbors(DataSet dset,
            DataInstance instance, int neighborhoodSize, CombinedMetric cmet)
            throws Exception {
        int[] neighbors = new int[neighborhoodSize];
        float currDist;
        int kCurrLen = 0;
        float[] nDists = new float[neighborhoodSize];
        Arrays.fill(nDists, Float.MAX_VALUE);
        int l;
        for (int i = 0; i < dset.size(); i++) {
            currDist = cmet.dist(instance, dset.data.get(i));
            if (kCurrLen > 0) {
                if (kCurrLen == neighborhoodSize) {
                    if (currDist < nDists[kCurrLen - 1]) {
                        // Search and insert.
                        l = neighborhoodSize - 1;
                        while ((l >= 1) && currDist < nDists[l - 1]) {
                            nDists[l] = nDists[l - 1];
                            neighbors[l] = neighbors[l - 1];
                            l--;
                        }
                        nDists[l] = currDist;
                        neighbors[l] = i;
                    }
                } else {
                    if (currDist < nDists[kCurrLen - 1]) {
                        // Search and insert.
                        l = kCurrLen - 1;
                        nDists[kCurrLen] = nDists[kCurrLen - 1];
                        neighbors[kCurrLen] = neighbors[kCurrLen - 1];
                        while ((l >= 1) && currDist < nDists[l - 1]) {
                            nDists[l] = nDists[l - 1];
                            neighbors[l] = neighbors[l - 1];
                            l--;
                        }
                        nDists[l] = currDist;
                        neighbors[l] = i;
                        kCurrLen++;
                    } else {
                        nDists[kCurrLen] = currDist;
                        neighbors[kCurrLen] = i;
                        kCurrLen++;
                    }
                }
            } else {
                nDists[0] = currDist;
                neighbors[0] = i;
                kCurrLen = 1;
            }
        }
        return neighbors;
    }

    /**
     * This method queries the dataset with an instance, given a tabu map of
     * points that can not be considered as potential neighbors.
     *
     * @param dset DataSet object to query.
     * @param instance DataInstance that is the query point.
     * @param neighborhoodSize Integer that is the desired neighborhood size.
     * @param cmet CombinedMetric object for distance calculations. to the
     * points in the provided DataSet.
     * @param tabuMap HashMap containing the indexes of points that can not be
     * considered as neighbors currently.
     * @return int[] that contains the indexes of the k-nearest neighbors for
     * the query point.
     * @throws Exception
     */
    public static int[] getIndexesOfNeighbors(DataSet dset,
            DataInstance instance, int neighborhoodSize, CombinedMetric cmet,
            HashMap tabuMap) throws Exception {
        int[] neighbors = new int[neighborhoodSize];
        float tempDist;
        int kCurrLen = 0;
        float[] nDists = new float[neighborhoodSize];
        Arrays.fill(nDists, Float.MAX_VALUE);
        int l;
        for (int i = 0; i < dset.size(); i++) {
            if (tabuMap.containsKey(i)) {
                continue;
            }
            tempDist = cmet.dist(instance, dset.data.get(i));
            if (kCurrLen > 0) {
                if (kCurrLen == neighborhoodSize) {
                    if (tempDist < nDists[kCurrLen - 1]) {
                        // Search and insert.
                        l = neighborhoodSize - 1;
                        while ((l >= 1) && tempDist < nDists[l - 1]) {
                            nDists[l] = nDists[l - 1];
                            neighbors[l] = neighbors[l - 1];
                            l--;
                        }
                        nDists[l] = tempDist;
                        neighbors[l] = i;
                    }
                } else {
                    if (tempDist < nDists[kCurrLen - 1]) {
                        // Search and insert.
                        l = kCurrLen - 1;
                        nDists[kCurrLen] = nDists[kCurrLen - 1];
                        neighbors[kCurrLen] = neighbors[kCurrLen - 1];
                        while ((l >= 1) && tempDist < nDists[l - 1]) {
                            nDists[l] = nDists[l - 1];
                            neighbors[l] = neighbors[l - 1];
                            l--;
                        }
                        nDists[l] = tempDist;
                        neighbors[l] = i;
                        kCurrLen++;
                    } else {
                        nDists[kCurrLen] = tempDist;
                        neighbors[kCurrLen] = i;
                        kCurrLen++;
                    }
                }
            } else {
                nDists[0] = tempDist;
                neighbors[0] = i;
                kCurrLen = 1;
            }
        }
        return neighbors;
    }
    
    /**
     * This method queries the dataset with an instance, given a tabu map of
     * points that can not be considered as potential neighbors.
     *
     * @param dMat float[][] Upper triangular distance matrix of the dataset to
     * query.
     * @param instanceIndex Integer that is the instance index.
     * @param neighborhoodSize Integer that is the desired neighborhood size.
     * @param tabuMap HashMap containing the indexes of points that can not be
     * considered as neighbors currently.
     * @return int[] that contains the indexes of the k-nearest neighbors for
     * the query point.
     * @throws Exception
     */
    public static int[] getIndexesOfNeighbors(float[][] dMat, int instanceIndex,
            int neighborhoodSize, HashMap tabuMap) throws Exception {
        int[] neighbors = new int[neighborhoodSize];
        float tempDist;
        int kCurrLen = 0;
        float[] nDists = new float[neighborhoodSize];
        Arrays.fill(nDists, Float.MAX_VALUE);
        int l;
        for (int i = 0; i < dMat.length; i++) {
            if (tabuMap.containsKey(i) || i == instanceIndex) {
                continue;
            }
            int minIndex = Math.min(i, instanceIndex);
            int maxIndex = Math.max(i, instanceIndex);
            tempDist = dMat[minIndex][maxIndex - minIndex - 1];
            if (kCurrLen > 0) {
                if (kCurrLen == neighborhoodSize) {
                    if (tempDist < nDists[kCurrLen - 1]) {
                        // Search and insert.
                        l = neighborhoodSize - 1;
                        while ((l >= 1) && tempDist < nDists[l - 1]) {
                            nDists[l] = nDists[l - 1];
                            neighbors[l] = neighbors[l - 1];
                            l--;
                        }
                        nDists[l] = tempDist;
                        neighbors[l] = i;
                    }
                } else {
                    if (tempDist < nDists[kCurrLen - 1]) {
                        // Search and insert.
                        l = kCurrLen - 1;
                        nDists[kCurrLen] = nDists[kCurrLen - 1];
                        neighbors[kCurrLen] = neighbors[kCurrLen - 1];
                        while ((l >= 1) && tempDist < nDists[l - 1]) {
                            nDists[l] = nDists[l - 1];
                            neighbors[l] = neighbors[l - 1];
                            l--;
                        }
                        nDists[l] = tempDist;
                        neighbors[l] = i;
                        kCurrLen++;
                    } else {
                        nDists[kCurrLen] = tempDist;
                        neighbors[kCurrLen] = i;
                        kCurrLen++;
                    }
                }
            } else {
                nDists[0] = tempDist;
                neighbors[0] = i;
                kCurrLen = 1;
            }
        }
        return neighbors;
    }

    /**
     * @return int[] representing the neighbor occurrence frequencies for all
     * data points.
     */
    public int[] getNeighborFrequencies() {
        return kNeighborFrequencies;
    }

    /**
     * @return int[] that contains the good neighbor occurrence frequencies for
     * all data points.
     */
    public int[] getGoodFrequencies() {
        return kGoodFrequencies;
    }

    /**
     * @return int[] that contains the bad neighbor occurrence frequencies for
     * all data points.
     */
    public int[] getBadFrequencies() {
        return kBadFrequencies;
    }

    /**
     * @return DataSet object that is being analyzed.
     */
    public DataSet getDataSet() {
        return dset;
    }

    /**
     * @return float[] representing the neighbor occurrence frequencies for all
     * data points.
     */
    public float[] getFloatOccFreqs() {
        float[] occFreqs = new float[kNeighbors.length];
        for (int i = 0; i < kNeighbors.length; i++) {
            occFreqs[i] = kNeighborFrequencies[i];
        }
        return occFreqs;
    }

    /**
     * @return float[][] representing the upper triangular distance matrix.
     */
    public float[][] getDistances() {
        return distMatrix;
    }

    /**
     * @return float[][] representing an array of arrays of k-distances for all
     * data points.
     */
    public float[][] getKDistances() {
        return kDistances;
    }

    /**
     * This method calculates an estimate of the data density in all data points
     * based on their nearest neighbors and the kNN radius.
     *
     * @param dset DataSet object that the points belong to.
     * @param cmet CombinedMetric object for distance calculations.
     * @param currK Integer that is the neighborhood size.
     * @param kNeighbors int[][] that is an array of neighbor index arrays for
     * all data points.
     * @return double[] containing the estimated point-wise densities.
     * @throws Exception
     */
    public static double[] getDataDensitiesByNormalizedRadius(DataSet dset,
            CombinedMetric cmet, int currK, int[][] kNeighbors)
            throws Exception {
        double[] radii = new double[dset.size()];
        double maxRadius = -Float.MAX_VALUE;
        for (int i = 0; i < radii.length; i++) {
            radii[i] = dset.getRadiusOfVolumeForInstances(kNeighbors[i], cmet);
            if (radii[i] > maxRadius) {
                maxRadius = radii[i];
            }
        }
        if (maxRadius == 0) {
            return null;
        }
        for (int i = 0; i < radii.length; i++) {
            if (radii[i] != 0) {
                radii[i] /= maxRadius;
                radii[i] = currK / Math.pow(radii[i], dset.getNumFloatAttr());
            }
        }
        return radii;
    }

    /**
     * This method calculates an estimate of the data density in a subset of
     * data points based on their nearest neighbors and the kNN radius. It is
     * useful for some incremental experiments.
     *
     * @param dset DataSet object that the points belong to.
     * @param cmet CombinedMetric object for distance calculations.
     * @param currK Integer that is the neighborhood size.
     * @param kNeighbors int[][] that is an array of neighbor index arrays for
     * all data points.
     * @param maxPointIndex Integer that is the maximum index to calculate the
     * densities for.
     * @return double[] containing the estimated point-wise densities.
     * @throws Exception
     */
    public static double[] getDataDensitiesByNormalizedRadiusForElementsUntil(
            DataSet dset, CombinedMetric cmet, int currK, int[][] kNeighbors,
            int maxPointIndex) throws Exception {
        double[] radii = new double[maxPointIndex];
        double maxRadius = -Float.MAX_VALUE;
        for (int i = 0; i < maxPointIndex; i++) {
            radii[i] = dset.getRadiusOfVolumeForInstances(kNeighbors[i], cmet);
            if (radii[i] > maxRadius) {
                maxRadius = radii[i];
            }
        }
        if (maxRadius == 0) {
            return null;
        }
        for (int i = 0; i < maxPointIndex; i++) {
            if (radii[i] != 0) {
                radii[i] /= maxRadius;
                radii[i] = currK / Math.pow(radii[i], dset.getNumFloatAttr());
            }
        }
        return radii;
    }

    /**
     * This method calculates an estimate of the data density in all data points
     * based on their nearest neighbors and the kNN radius.
     *
     * @return double[] containing the estimated point-wise densities.
     * @throws Exception
     */
    public double[] getDataDensitiesByNormalizedRadius() throws Exception {
        double[] radii = new double[dset.size()];
        double maxRadius = -Float.MAX_VALUE;
        for (int i = 0; i < radii.length; i++) {
            radii[i] = dset.getRadiusOfVolumeForInstances(
                    kNeighbors[i], cmet, currK);
            if (radii[i] > maxRadius) {
                maxRadius = radii[i];
            }
        }
        if (maxRadius == 0) {
            return null;
        }
        for (int i = 0; i < radii.length; i++) {
            if (radii[i] != 0) {
                radii[i] /= maxRadius;
                radii[i] = currK / Math.pow(radii[i], dset.getNumFloatAttr());
            }
        }
        return radii;
    }

    /**
     * This method calculates the distance mean and variance.
     */
    public final void calculateOccFreqMeanAndVariance() {
        for (int i = 0; i < distMatrix.length; i++) {
            for (int j = 0; j < distMatrix[i].length; j++) {
                distMean += distMatrix[i][j];
            }
        }
        distMean = distMean / (dset.size() - 1);
        distVariance = 0;
        for (int i = 0; i < distMatrix.length; i++) {
            for (int j = 0; j < distMatrix[i].length; j++) {
                distVariance += (distMean - distMatrix[i][j])
                        * (distMean - distMatrix[i][j]);
            }
        }
        distVariance = distVariance / (dset.size() - 1);
    }

    /**
     * This method calculates the distance matrix if it wasn't provided.
     *
     * @throws Exception
     */
    public void calculateDistances() throws Exception {
        if (dset == null || dset.isEmpty()) {
            return;
        }
        distMatrix = new float[dset.size()][];
        distMean = 0;
        for (int i = 0; i < dset.size(); i++) {
            distMatrix[i] = new float[distMatrix.length - i - 1];
            for (int j = i + 1; j < distMatrix.length; j++) {
                distMatrix[i][j - i - 1] = cmet.dist(dset.data.get(i),
                        dset.data.get(j));
                distMean += distMatrix[i][j - i - 1];
            }
        }
        distMean = distMean / (dset.size() - 1);
        distVariance = 0;
        for (int i = 0; i < distMatrix.length; i++) {
            for (int j = i + 1; j < distMatrix.length; j++) {
                distVariance += (distMean - distMatrix[i][j - i - 1])
                        * (distMean - distMatrix[i][j - i - 1]);
            }
        }
        distVariance = distVariance / (dset.size() - 1);
        distancesCalculated = true;
    }

    /**
     * @return Double value that is the distance variance.
     */
    public double getDistanceVariance() {
        return distVariance;
    }

    /**
     * @return Double value that is the distance mean.
     */
    public double getDistanceMean() {
        return distMean;
    }

    /**
     * This class allows for multi-threaded calculations of the kNN sets.
     */
    class ThreadNeighborCalculator implements Runnable {

        private int startRow;
        private int endRow;
        private int k;

        /**
         * Initialization.
         *
         * @param startRow Integer that is the index of the start row,
         * inclusive.
         * @param endRow Integer that is the index of the end row, inclusive.
         * @param k Integer that is the neighborhood size.
         */
        public ThreadNeighborCalculator(int startRow, int endRow, int k) {
            this.startRow = startRow;
            this.k = k;
            this.endRow = endRow;
        }

        @Override
        public void run() {
            try {
                int l;
                for (int i = startRow; i <= endRow; i++) {

                    for (int j = 0; j < i; j++) {
                        if (kCurrLen[i] > 0) {
                            if (kCurrLen[i] == k) {
                                if (distMatrix[j][i - j - 1]
                                        < kDistances[i][kCurrLen[i] - 1]) {
                                    // Search and insert.
                                    l = k - 1;
                                    while ((l >= 1) && distMatrix[j][i - j - 1]
                                            < kDistances[i][l - 1]) {
                                        kDistances[i][l] = kDistances[i][l - 1];
                                        kNeighbors[i][l] = kNeighbors[i][l - 1];
                                        l--;
                                    }
                                    kDistances[i][l] = distMatrix[j][i - j - 1];
                                    kNeighbors[i][l] = j;
                                }
                            } else {
                                if (distMatrix[j][i - j - 1] < kDistances[i][
                                        kCurrLen[i] - 1]) {
                                    //search to see where to insert
                                    l = kCurrLen[i] - 1;
                                    kDistances[i][kCurrLen[i]] = kDistances[i][
                                            kCurrLen[i] - 1];
                                    kNeighbors[i][kCurrLen[i]] = kNeighbors[i][
                                            kCurrLen[i] - 1];
                                    while ((l >= 1) && distMatrix[j][i - j - 1]
                                            < kDistances[i][l - 1]) {
                                        kDistances[i][l] = kDistances[i][l - 1];
                                        kNeighbors[i][l] = kNeighbors[i][l - 1];
                                        l--;
                                    }
                                    kDistances[i][l] = distMatrix[j][i - j - 1];
                                    kNeighbors[i][l] = j;
                                    kCurrLen[i]++;
                                } else {
                                    kDistances[i][kCurrLen[i]] = distMatrix[j][
                                            i - j - 1];
                                    kNeighbors[i][kCurrLen[i]] = j;
                                    kCurrLen[i]++;
                                }
                            }
                        } else {
                            kDistances[i][0] = distMatrix[j][i - j - 1];
                            kNeighbors[i][0] = j;
                            kCurrLen[i] = 1;
                        }
                    }

                    for (int j = 0; j < distMatrix[i].length; j++) {
                        if (kCurrLen[i] > 0) {
                            if (kCurrLen[i] == k) {
                                if (distMatrix[i][j] < kDistances[i][
                                        kCurrLen[i] - 1]) {
                                    // Search and inset.
                                    l = k - 1;
                                    while ((l >= 1) && distMatrix[i][j]
                                            < kDistances[i][l - 1]) {
                                        kDistances[i][l] = kDistances[i][l - 1];
                                        kNeighbors[i][l] = kNeighbors[i][l - 1];
                                        l--;
                                    }
                                    kDistances[i][l] = distMatrix[i][j];
                                    kNeighbors[i][l] = i + j + 1;
                                }
                            } else {
                                if (distMatrix[i][j] < kDistances[i][
                                        kCurrLen[i] - 1]) {
                                    // Search and insert.
                                    l = kCurrLen[i] - 1;
                                    kDistances[i][kCurrLen[i]] = kDistances[i][
                                            kCurrLen[i] - 1];
                                    kNeighbors[i][kCurrLen[i]] = kNeighbors[i][
                                            kCurrLen[i] - 1];
                                    while ((l >= 1) && distMatrix[i][j]
                                            < kDistances[i][l - 1]) {
                                        kDistances[i][l] = kDistances[i][l - 1];
                                        kNeighbors[i][l] = kNeighbors[i][l - 1];
                                        l--;
                                    }
                                    kDistances[i][l] = distMatrix[i][j];
                                    kNeighbors[i][l] = i + j + 1;
                                    kCurrLen[i]++;
                                } else {
                                    kDistances[i][kCurrLen[i]] =
                                            distMatrix[i][j];
                                    kNeighbors[i][kCurrLen[i]] = i + j + 1;
                                    kCurrLen[i]++;
                                }
                            }
                        } else {
                            kDistances[i][0] = distMatrix[i][j];
                            kNeighbors[i][0] = i + j + 1;
                            kCurrLen[i] = 1;
                        }
                    }
                }
            } catch (Exception e) {
                System.err.println("kNN calculation error.");
                System.err.println(e.getMessage());
            }
        }
    }

    /**
     * This method calculates the occurrence frequency (total, good, bad) means
     * and standard deviations.
     */
    public void calculateHubnessStats() {
        meanOccFreq = 0;
        stDevOccFreq = 0;
        meanOccBadness = 0;
        stDevOccBadness = 0;
        meanOccGoodness = 0;
        stDevOccGoodness = 0;
        meanGoodMinusBadness = 0;
        stDevGoodMinusBadness = 0;
        meanRelativeGoodMinusBadness = 0;
        stDevRelativeGoodMinusBadness = 0;
        for (int i = 0; i < kBadFrequencies.length; i++) {
            meanOccFreq += kNeighborFrequencies[i];
            meanOccBadness += kBadFrequencies[i];
            meanOccGoodness += kGoodFrequencies[i];
            meanGoodMinusBadness += kGoodFrequencies[i] - kBadFrequencies[i];
            if (kNeighborFrequencies[i] > 0) {
                meanRelativeGoodMinusBadness += ((kGoodFrequencies[i]
                        - kBadFrequencies[i]) / kNeighborFrequencies[i]);
            } else {
                meanRelativeGoodMinusBadness += 1;
            }
        }
        meanOccFreq /= (float) kNeighborFrequencies.length;
        meanOccBadness /= (float) kBadFrequencies.length;
        meanOccGoodness /= (float) kGoodFrequencies.length;
        meanGoodMinusBadness /= (float) kGoodFrequencies.length;
        meanRelativeGoodMinusBadness /= (float) kGoodFrequencies.length;
        for (int i = 0; i < kBadFrequencies.length; i++) {
            stDevOccFreq += ((meanOccFreq - kNeighborFrequencies[i])
                    * (meanOccFreq - kNeighborFrequencies[i]));
            stDevOccBadness += ((meanOccBadness - kBadFrequencies[i])
                    * (meanOccBadness - kBadFrequencies[i]));
            stDevOccGoodness += ((meanOccGoodness - kGoodFrequencies[i])
                    * (meanOccGoodness - kGoodFrequencies[i]));
            stDevGoodMinusBadness += ((meanGoodMinusBadness
                    - (kGoodFrequencies[i] - kBadFrequencies[i]))
                    * (meanGoodMinusBadness - (kGoodFrequencies[i]
                    - kBadFrequencies[i])));
            if (kNeighborFrequencies[i] > 0) {
                stDevRelativeGoodMinusBadness += (meanRelativeGoodMinusBadness
                        - ((kGoodFrequencies[i] - kBadFrequencies[i])
                        / kNeighborFrequencies[i]))
                        * (meanRelativeGoodMinusBadness - ((kGoodFrequencies[i]
                        - kBadFrequencies[i]) / kNeighborFrequencies[i]));
            } else {
                stDevRelativeGoodMinusBadness +=
                        (meanRelativeGoodMinusBadness - 1)
                        * (meanRelativeGoodMinusBadness - 1);
            }
        }
        stDevOccFreq /= (float) kNeighborFrequencies.length;
        stDevOccBadness /= (float) kBadFrequencies.length;
        stDevOccGoodness /= (float) kGoodFrequencies.length;
        stDevGoodMinusBadness /= (float) kGoodFrequencies.length;
        stDevRelativeGoodMinusBadness /= (float) kGoodFrequencies.length;
        stDevOccFreq = Math.sqrt(stDevOccFreq);
        stDevOccBadness = Math.sqrt(stDevOccBadness);
        stDevOccGoodness = Math.sqrt(stDevOccGoodness);
        stDevGoodMinusBadness = Math.sqrt(stDevGoodMinusBadness);
        stDevRelativeGoodMinusBadness =
                Math.sqrt(stDevRelativeGoodMinusBadness);
    }

    /**
     * This method allows the users to eliminate a particular neighbor point
     * from all calculated kNN sets. This can be used in cases of instance
     * selection, when certain points are removed from the prototype set.
     *
     * @param tNeighborIndex Integer that is the index of the neighbor point to
     * eliminate from all neighbor sets.
     * @param tVal Integer that is the value to put in the tabu hashmap
     * alongside the neighbor index.
     * @param tabuList HashMap that is the list of all banned neighbors.
     * @param calculateHubnessStatistics Boolean flag indicating whether to
     * recalculate the mean / stDev after each tabu run.
     * @throws Exception
     */
    public void tabuANeighbor(int tNeighborIndex, int tVal, HashMap tabuList,
            boolean calculateHubnessStatistics) throws Exception {
        int k = currK;
        if (!tabuList.containsKey(tNeighborIndex)) {
            tabuList.put(tNeighborIndex, tVal);
        }
        if (kNeighbors == null) {
            calculateNeighborSetsMultiThr(k, 4);
        }
        if (kCurrLen == null) {
            kDistances = new float[dset.size()][k];
            kCurrLen = new int[dset.size()];
        }
        // Eliminate the specified neighbor point.
        if (kNeighborFrequencies[tNeighborIndex] == 0) {
            // If it is an orphan, never occurs, then there are no places to
            // remove it from and there is nothing to do.
            return;
        }

        ArrayList<Integer> rnnIndexList = reverseNeighbors[tNeighborIndex];
        ArrayList<Integer> neighborIntervals;
        int upperVal, lowerVal;
        int minIndex, maxIndex;
        int datasize = dset.size();
        currK = k;
        int l;
        int remNeighborIndex;
        kNeighborFrequencies[tNeighborIndex] = 0;
        kGoodFrequencies[tNeighborIndex] = 0;
        kBadFrequencies[tNeighborIndex] = 0;
        int currClass;
        for (int tInd = 0; tInd < rnnIndexList.size(); tInd++) {
            int i = rnnIndexList.get(tInd);
            currClass = dset.getLabelOf(i);
            remNeighborIndex = 0;
            while (remNeighborIndex < kCurrLen[i]
                    && kNeighbors[i][remNeighborIndex] != tNeighborIndex) {
                remNeighborIndex++;
            }
            // Shift the remaining neighbors.
            if (remNeighborIndex < k - 1) {
                for (int m = remNeighborIndex + 1; m < k; m++) {
                    kNeighbors[i][m - 1] = kNeighbors[i][m];
                    kDistances[i][m - 1] = kDistances[i][m];
                }
            }
            kNeighbors[i][k - 1] = 0;
            kDistances[i][k - 1] = Float.MAX_VALUE;
            kCurrLen[i]--;
            // Find the replacement neighbor point.
            if (kCurrLen[i] < k) {
                neighborIntervals = new ArrayList(k + 2);
                neighborIntervals.add(-1);
                for (int j = 0; j < kCurrLen[i]; j++) {
                    neighborIntervals.add(kNeighbors[i][j]);
                }
                neighborIntervals.add(datasize + 1);
                Collections.sort(neighborIntervals);
                int iSizeRed = neighborIntervals.size() - 1;
                for (int ind = 0; ind < iSizeRed; ind++) {
                    lowerVal = neighborIntervals.get(ind);
                    upperVal = neighborIntervals.get(ind + 1);
                    for (int j = lowerVal + 1; j < upperVal - 1; j++) {
                        if ((tabuList.containsKey(j))) {
                            continue;
                        }
                        if (i != j) {
                            minIndex = Math.min(i, j);
                            maxIndex = Math.max(i, j);
                            if (kCurrLen[i] > 0) {
                                if (kCurrLen[i] == k) {
                                    if (distMatrix[minIndex][maxIndex - minIndex
                                            - 1] < kDistances[i][
                                            kCurrLen[i] - 1]) {
                                        // Search and insert.
                                        l = k - 1;
                                        while ((l >= 1) && distMatrix[
                                                minIndex][maxIndex - minIndex
                                                - 1] < kDistances[i][l - 1]) {
                                            kDistances[i][l] =
                                                    kDistances[i][l - 1];
                                            kNeighbors[i][l] =
                                                    kNeighbors[i][l - 1];
                                            l--;
                                        }
                                        kDistances[i][l] = distMatrix[
                                                minIndex][maxIndex
                                                - minIndex - 1];
                                        kNeighbors[i][l] = j;
                                    }
                                } else {
                                    if (distMatrix[minIndex][maxIndex
                                            - minIndex - 1] < kDistances[i][
                                            kCurrLen[i] - 1]) {
                                        // Search and insert.
                                        l = kCurrLen[i] - 1;
                                        kDistances[i][kCurrLen[i]] =
                                                kDistances[i][kCurrLen[i] - 1];
                                        kNeighbors[i][kCurrLen[i]] =
                                                kNeighbors[i][kCurrLen[i] - 1];
                                        while ((l >= 1) && distMatrix[
                                                minIndex][maxIndex - minIndex
                                                - 1] < kDistances[i][l - 1]) {
                                            kDistances[i][l] =
                                                    kDistances[i][l - 1];
                                            kNeighbors[i][l] =
                                                    kNeighbors[i][l - 1];
                                            l--;
                                        }
                                        kDistances[i][l] = distMatrix[
                                                minIndex][maxIndex
                                                - minIndex - 1];
                                        kNeighbors[i][l] = j;
                                        kCurrLen[i]++;
                                    } else {
                                        kDistances[i][kCurrLen[i]] = distMatrix[
                                                minIndex][maxIndex
                                                - minIndex - 1];
                                        kNeighbors[i][kCurrLen[i]] = j;
                                        kCurrLen[i]++;
                                    }
                                }
                            } else {
                                kDistances[i][0] = distMatrix[minIndex][
                                        maxIndex - minIndex - 1];
                                kNeighbors[i][0] = j;
                                kCurrLen[i] = 1;
                            }
                        }
                    }
                }
                // The last one is the new neighbor point, so update its
                // occurrence stats and the reverse neighbor list.
                if (kCurrLen[i] > 0) {
                    if (dset.getLabelOf(
                            kNeighbors[i][kCurrLen[i] - 1]) == currClass) {
                        kGoodFrequencies[kNeighbors[i][kCurrLen[i] - 1]]++;
                    } else {
                        kBadFrequencies[kNeighbors[i][kCurrLen[i] - 1]]++;
                    }
                    kNeighborFrequencies[kNeighbors[i][kCurrLen[i] - 1]]++;
                    reverseNeighbors[kNeighbors[i][kCurrLen[i] - 1]].add(i);
                }
            }
        }
        // Empty the RNN set of the tabu neighbor point.
        reverseNeighbors[tNeighborIndex] = new ArrayList<>(4);
        kNeighborFrequencies[tNeighborIndex] = 0;
        kGoodFrequencies[tNeighborIndex] = 0;
        kBadFrequencies[tNeighborIndex] = 0;
        if (calculateHubnessStatistics) {
            meanOccFreq = 0;
            stDevOccFreq = 0;
            meanOccBadness = 0;
            stDevOccBadness = 0;
            meanOccGoodness = 0;
            stDevOccGoodness = 0;
            meanGoodMinusBadness = 0;
            stDevGoodMinusBadness = 0;
            meanRelativeGoodMinusBadness = 0;
            stDevRelativeGoodMinusBadness = 0;
            for (int i = 0; i < kBadFrequencies.length; i++) {
                meanOccFreq += kNeighborFrequencies[i];
                meanOccBadness += kBadFrequencies[i];
                meanOccGoodness += kGoodFrequencies[i];
                meanGoodMinusBadness += kGoodFrequencies[i]
                        - kBadFrequencies[i];
                if (kNeighborFrequencies[i] > 0) {
                    meanRelativeGoodMinusBadness += ((kGoodFrequencies[i]
                            - kBadFrequencies[i]) / kNeighborFrequencies[i]);
                } else {
                    meanRelativeGoodMinusBadness += 1;
                }
            }
            meanOccFreq /= (float) kNeighborFrequencies.length;
            meanOccBadness /= (float) kBadFrequencies.length;
            meanOccGoodness /= (float) kGoodFrequencies.length;
            meanGoodMinusBadness /= (float) kGoodFrequencies.length;
            meanRelativeGoodMinusBadness /= (float) kGoodFrequencies.length;
            for (int i = 0; i < kBadFrequencies.length; i++) {
                stDevOccFreq += ((meanOccFreq - kNeighborFrequencies[i])
                        * (meanOccFreq - kNeighborFrequencies[i]));
                stDevOccBadness += ((meanOccBadness - kBadFrequencies[i])
                        * (meanOccBadness - kBadFrequencies[i]));
                stDevOccGoodness += ((meanOccGoodness - kGoodFrequencies[i])
                        * (meanOccGoodness - kGoodFrequencies[i]));
                stDevGoodMinusBadness += ((meanGoodMinusBadness
                        - (kGoodFrequencies[i] - kBadFrequencies[i]))
                        * (meanGoodMinusBadness - (kGoodFrequencies[i]
                        - kBadFrequencies[i])));
                if (kNeighborFrequencies[i] > 0) {
                    stDevRelativeGoodMinusBadness +=
                            (meanRelativeGoodMinusBadness
                            - ((kGoodFrequencies[i] - kBadFrequencies[i])
                            / kNeighborFrequencies[i]))
                            * (meanRelativeGoodMinusBadness
                            - ((kGoodFrequencies[i]
                            - kBadFrequencies[i]) / kNeighborFrequencies[i]));
                } else {
                    stDevRelativeGoodMinusBadness +=
                            (meanRelativeGoodMinusBadness - 1)
                            * (meanRelativeGoodMinusBadness - 1);
                }
            }
            stDevOccFreq /= (float) kNeighborFrequencies.length;
            stDevOccBadness /= (float) kBadFrequencies.length;
            stDevOccGoodness /= (float) kGoodFrequencies.length;
            stDevGoodMinusBadness /= (float) kGoodFrequencies.length;
            stDevRelativeGoodMinusBadness /= (float) kGoodFrequencies.length;
            stDevOccFreq = Math.sqrt(stDevOccFreq);
            stDevOccBadness = Math.sqrt(stDevOccBadness);
            stDevOccGoodness = Math.sqrt(stDevOccGoodness);
            stDevGoodMinusBadness = Math.sqrt(stDevGoodMinusBadness);
            stDevRelativeGoodMinusBadness =
                    Math.sqrt(stDevRelativeGoodMinusBadness);
        }
    }

    /**
     * If some neighbor sets are incomplete, after the removal of certain
     * neighbor points from neighbor sets and/or maybe wanting to increase k
     * (and after the shifting of the neighbor lists and distance lists to the
     * left) the method reads the kcurrLen array and if some kCurrLen is less
     * than k, it completes it. also, if k is greater than the length of the
     * neighbor lists, a new matrix is generated, the arrays are extended. All
     * occurrence-stats calculations are performed after the kNN recalculations.
     * This method also takes as a parameter a tabu list (or rather, a tabu map)
     * that marks certain points as 'forbidden' and they are not considered as
     * potential neighbors during re-calculations.
     *
     * @param k Integer that is the neighborhood size.
     * @param tabuList HashMap that contains the forbidden elements, those which
     * are not to be considered as potential neighbors.
     */
    public void completeNeighborSets(int k, HashMap tabuList) throws Exception {
        if (kNeighbors == null) {
            calculateNeighborSetsMultiThr(k, 4);
        }
        if (kDistances == null) {
            kDistances = new float[dset.size()][k];
            kCurrLen = new int[dset.size()];
        }
        if (kNeighbors[0].length < k) {
            int[][] kneighborsExtended = new int[dset.size()][k];
            float[][] kdistancesExtended = new float[dset.size()][k];
            for (int i = 0; i < kneighborsExtended.length; i++) {
                kneighborsExtended[i] = Arrays.copyOf(kNeighbors[i], k);
                kdistancesExtended[i] = Arrays.copyOf(kDistances[i], k);
            }
            kNeighbors = kneighborsExtended;
            kDistances = kdistancesExtended;
        }
        int upperVal, lowerVal;
        int minIndex, maxIndex;
        int datasize = dset.size();
        currK = k;
        int l;
        for (int i = 0; i < datasize; i++) {
            if (kCurrLen[i] < k) {
                ArrayList<Integer> neighborIntervals = new ArrayList<>(k + 2);
                neighborIntervals.add(-1);
                for (int j = 0; j < kCurrLen[i]; j++) {
                    neighborIntervals.add(kNeighbors[i][j]);
                }
                neighborIntervals.add(datasize + 1);
                // Ascending sort.
                Collections.sort(neighborIntervals);
                int iSizeRed = neighborIntervals.size() - 1;
                for (int ind = 0; ind < iSizeRed; ind++) {
                    lowerVal = neighborIntervals.get(ind);
                    upperVal = neighborIntervals.get(ind + 1);
                    for (int j = lowerVal + 1; j < upperVal - 1; j++) {
                        if (tabuList != null && tabuList.containsKey(j)) {
                            continue;
                        }
                        if (i != j) {
                            minIndex = Math.min(i, j);
                            maxIndex = Math.max(i, j);
                            if (kCurrLen[i] > 0) {
                                if (kCurrLen[i] == k) {
                                    if (distMatrix[minIndex][maxIndex
                                            - minIndex - 1] < kDistances[i][
                                            kCurrLen[i] - 1]) {
                                        // Search and insert.
                                        l = k - 1;
                                        while ((l >= 1) && distMatrix[minIndex][
                                                maxIndex - minIndex - 1]
                                                < kDistances[i][l - 1]) {
                                            kDistances[i][l] =
                                                    kDistances[i][l - 1];
                                            kNeighbors[i][l] =
                                                    kNeighbors[i][l - 1];
                                            l--;
                                        }
                                        kDistances[i][l] = distMatrix[minIndex][
                                                maxIndex - minIndex - 1];
                                        kNeighbors[i][l] = j;
                                    }
                                } else {
                                    if (distMatrix[minIndex][maxIndex
                                            - minIndex - 1] < kDistances[i][
                                            kCurrLen[i] - 1]) {
                                        // Search and insert.
                                        l = kCurrLen[i] - 1;
                                        kDistances[i][kCurrLen[i]] =
                                                kDistances[i][kCurrLen[i] - 1];
                                        kNeighbors[i][kCurrLen[i]] =
                                                kNeighbors[i][kCurrLen[i] - 1];
                                        while ((l >= 1) && distMatrix[minIndex][
                                                maxIndex - minIndex - 1]
                                                < kDistances[i][l - 1]) {
                                            kDistances[i][l] =
                                                    kDistances[i][l - 1];
                                            kNeighbors[i][l] =
                                                    kNeighbors[i][l - 1];
                                            l--;
                                        }
                                        kDistances[i][l] = distMatrix[minIndex][
                                                maxIndex - minIndex - 1];
                                        kNeighbors[i][l] = j;
                                        kCurrLen[i]++;
                                    } else {
                                        kDistances[i][kCurrLen[i]] = distMatrix[
                                                minIndex][maxIndex
                                                - minIndex - 1];
                                        kNeighbors[i][kCurrLen[i]] = j;
                                        kCurrLen[i]++;
                                    }
                                }
                            } else {
                                kDistances[i][0] = distMatrix[minIndex][
                                        maxIndex - minIndex - 1];
                                kNeighbors[i][0] = j;
                                kCurrLen[i] = 1;
                            }
                        }
                    }
                }
            }
        }
        reverseNeighbors = new ArrayList[datasize];
        for (int i = 0; i < dset.size(); i++) {
            reverseNeighbors[i] = new ArrayList<>(10 * k);
        }
        kNeighborFrequencies = new int[kNeighbors.length];
        kBadFrequencies = new int[kNeighbors.length];
        kGoodFrequencies = new int[kNeighbors.length];
        for (int i = 0; i < kNeighbors.length; i++) {
            for (int j = 0; j < k; j++) {
                reverseNeighbors[kNeighbors[i][j]].add(i);
                kNeighborFrequencies[kNeighbors[i][j]]++;
                if (dset.data.get(i).getCategory() != dset.data.get(
                        kNeighbors[i][j]).getCategory()) {
                    kBadFrequencies[kNeighbors[i][j]]++;
                } else {
                    kGoodFrequencies[kNeighbors[i][j]]++;
                }
            }
        }
        meanOccFreq = 0;
        stDevOccFreq = 0;
        meanOccBadness = 0;
        stDevOccBadness = 0;
        meanOccGoodness = 0;
        stDevOccGoodness = 0;
        meanGoodMinusBadness = 0;
        stDevGoodMinusBadness = 0;
        meanRelativeGoodMinusBadness = 0;
        stDevRelativeGoodMinusBadness = 0;
        for (int i = 0; i < kBadFrequencies.length; i++) {
            meanOccFreq += kNeighborFrequencies[i];
            meanOccBadness += kBadFrequencies[i];
            meanOccGoodness += kGoodFrequencies[i];
            meanGoodMinusBadness += kGoodFrequencies[i] - kBadFrequencies[i];
            if (kNeighborFrequencies[i] > 0) {
                meanRelativeGoodMinusBadness += ((kGoodFrequencies[i]
                        - kBadFrequencies[i]) / kNeighborFrequencies[i]);
            } else {
                meanRelativeGoodMinusBadness += 1;
            }
        }
        meanOccFreq /= (float) kNeighborFrequencies.length;
        meanOccBadness /= (float) kBadFrequencies.length;
        meanOccGoodness /= (float) kGoodFrequencies.length;
        meanGoodMinusBadness /= (float) kGoodFrequencies.length;
        meanRelativeGoodMinusBadness /= (float) kGoodFrequencies.length;
        for (int i = 0; i < kBadFrequencies.length; i++) {
            stDevOccFreq += ((meanOccFreq - kNeighborFrequencies[i])
                    * (meanOccFreq - kNeighborFrequencies[i]));
            stDevOccBadness += ((meanOccBadness - kBadFrequencies[i])
                    * (meanOccBadness - kBadFrequencies[i]));
            stDevOccGoodness += ((meanOccGoodness - kGoodFrequencies[i])
                    * (meanOccGoodness - kGoodFrequencies[i]));
            stDevGoodMinusBadness += ((meanGoodMinusBadness
                    - (kGoodFrequencies[i] - kBadFrequencies[i]))
                    * (meanGoodMinusBadness - (kGoodFrequencies[i]
                    - kBadFrequencies[i])));
            if (kNeighborFrequencies[i] > 0) {
                stDevRelativeGoodMinusBadness += (meanRelativeGoodMinusBadness
                        - ((kGoodFrequencies[i] - kBadFrequencies[i])
                        / kNeighborFrequencies[i]))
                        * (meanRelativeGoodMinusBadness - ((kGoodFrequencies[i]
                        - kBadFrequencies[i]) / kNeighborFrequencies[i]));
            } else {
                stDevRelativeGoodMinusBadness +=
                        (meanRelativeGoodMinusBadness - 1)
                        * (meanRelativeGoodMinusBadness - 1);
            }
        }
        stDevOccFreq /= (float) kNeighborFrequencies.length;
        stDevOccBadness /= (float) kBadFrequencies.length;
        stDevOccGoodness /= (float) kGoodFrequencies.length;
        stDevGoodMinusBadness /= (float) kGoodFrequencies.length;
        stDevRelativeGoodMinusBadness /= (float) kGoodFrequencies.length;
        stDevOccFreq = Math.sqrt(stDevOccFreq);
        stDevOccBadness = Math.sqrt(stDevOccBadness);
        stDevOccGoodness = Math.sqrt(stDevOccGoodness);
        stDevGoodMinusBadness = Math.sqrt(stDevGoodMinusBadness);
        stDevRelativeGoodMinusBadness =
                Math.sqrt(stDevRelativeGoodMinusBadness);
    }

    /**
     * This method calculates the k-nearest neighbor sets in a multi-threaded
     * way.
     *
     * @param k Integer that is the neighborhood size.
     * @param numThreads Integer that is the number of threads to use.
     */
    public void calculateNeighborSetsMultiThr(int k, int numThreads) {
        if (dset == null || dset.isEmpty() || distMatrix == null) {
            return;
        }
        currK = k;
        kNeighbors = new int[dset.size()][k];
        kDistances = new float[dset.size()][k];
        kCurrLen = new int[dset.size()];
        reverseNeighbors = new ArrayList[dset.size()];
        for (int i = 0; i < dset.size(); i++) {
            reverseNeighbors[i] = new ArrayList<>(10 * k);
        }

        int size = dset.size();
        int chunkSize = size / numThreads;
        Thread[] threads = new Thread[numThreads];
        for (int tIndex = 0; tIndex < numThreads - 1; tIndex++) {
            threads[tIndex] = new Thread(new ThreadNeighborCalculator(
                    tIndex * chunkSize, (tIndex + 1) * chunkSize - 1, k));
            threads[tIndex].start();
        }
        threads[numThreads - 1] = new Thread(new ThreadNeighborCalculator(
                (numThreads - 1) * chunkSize, size - 1, k));
        threads[numThreads - 1].start();
        for (int tIndex = 0; tIndex < numThreads; tIndex++) {
            if (threads[tIndex] != null) {
                try {
                    threads[tIndex].join();
                } catch (Throwable t) {
                    System.err.println(t.getMessage());
                }
            }
        }
        // Calculate the occurrence stats.
        kNeighborFrequencies = new int[kNeighbors.length];
        kBadFrequencies = new int[kNeighbors.length];
        kGoodFrequencies = new int[kNeighbors.length];
        for (int i = 0; i < kNeighbors.length; i++) {
            for (int j = 0; j < k; j++) {
                reverseNeighbors[kNeighbors[i][j]].add(i);
                kNeighborFrequencies[kNeighbors[i][j]]++;
                if (dset.data.get(i).getCategory() != dset.data.get(
                        kNeighbors[i][j]).getCategory()) {
                    kBadFrequencies[kNeighbors[i][j]]++;
                } else {
                    kGoodFrequencies[kNeighbors[i][j]]++;
                }
            }
        }
        meanOccFreq = 0;
        stDevOccFreq = 0;
        meanOccBadness = 0;
        stDevOccBadness = 0;
        meanOccGoodness = 0;
        stDevOccGoodness = 0;
        meanGoodMinusBadness = 0;
        stDevGoodMinusBadness = 0;
        meanRelativeGoodMinusBadness = 0;
        stDevRelativeGoodMinusBadness = 0;
        for (int i = 0; i < kBadFrequencies.length; i++) {
            meanOccFreq += kNeighborFrequencies[i];
            meanOccBadness += kBadFrequencies[i];
            meanOccGoodness += kGoodFrequencies[i];
            meanGoodMinusBadness += kGoodFrequencies[i] - kBadFrequencies[i];
            if (kNeighborFrequencies[i] > 0) {
                meanRelativeGoodMinusBadness += ((kGoodFrequencies[i]
                        - kBadFrequencies[i]) / kNeighborFrequencies[i]);
            } else {
                meanRelativeGoodMinusBadness += 1;
            }
        }
        meanOccFreq /= (float) kNeighborFrequencies.length;
        meanOccBadness /= (float) kBadFrequencies.length;
        meanOccGoodness /= (float) kGoodFrequencies.length;
        meanGoodMinusBadness /= (float) kGoodFrequencies.length;
        meanRelativeGoodMinusBadness /= (float) kGoodFrequencies.length;
        for (int i = 0; i < kBadFrequencies.length; i++) {
            stDevOccFreq += ((meanOccFreq - kNeighborFrequencies[i])
                    * (meanOccFreq - kNeighborFrequencies[i]));
            stDevOccBadness += ((meanOccBadness - kBadFrequencies[i])
                    * (meanOccBadness - kBadFrequencies[i]));
            stDevOccGoodness += ((meanOccGoodness - kGoodFrequencies[i])
                    * (meanOccGoodness - kGoodFrequencies[i]));
            stDevGoodMinusBadness += ((meanGoodMinusBadness
                    - (kGoodFrequencies[i] - kBadFrequencies[i]))
                    * (meanGoodMinusBadness - (kGoodFrequencies[i]
                    - kBadFrequencies[i])));
            if (kNeighborFrequencies[i] > 0) {
                stDevRelativeGoodMinusBadness += (meanRelativeGoodMinusBadness
                        - ((kGoodFrequencies[i] - kBadFrequencies[i])
                        / kNeighborFrequencies[i]))
                        * (meanRelativeGoodMinusBadness - ((kGoodFrequencies[i]
                        - kBadFrequencies[i]) / kNeighborFrequencies[i]));
            } else {
                stDevRelativeGoodMinusBadness +=
                        (meanRelativeGoodMinusBadness - 1)
                        * (meanRelativeGoodMinusBadness - 1);
            }
        }
        stDevOccFreq /= (float) kNeighborFrequencies.length;
        stDevOccBadness /= (float) kBadFrequencies.length;
        stDevOccGoodness /= (float) kGoodFrequencies.length;
        stDevGoodMinusBadness /= (float) kGoodFrequencies.length;
        stDevRelativeGoodMinusBadness /= (float) kGoodFrequencies.length;
        stDevOccFreq = Math.sqrt(stDevOccFreq);
        stDevOccBadness = Math.sqrt(stDevOccBadness);
        stDevOccGoodness = Math.sqrt(stDevOccGoodness);
        stDevGoodMinusBadness = Math.sqrt(stDevGoodMinusBadness);
        stDevRelativeGoodMinusBadness =
                Math.sqrt(stDevRelativeGoodMinusBadness);
    }

    /**
     * This method calculates the neighbor occurrence frequencies for some
     * neighborhood size that is less than equal to the currently calculated
     * size of the neighbor sets.
     *
     * @param kSmall Integer that is the neighborhood size to calculate the
     * neighbor occurrence frequencies for. It needs to be smaller than the
     * current length of kNN sets.
     * @return int[] representing the neighbor occurrence frequencies.
     */
    public int[] getNeighborOccFrequencies(int kSmall) {
        int[] occFreqs = new int[kNeighborFrequencies.length];
        for (int i = 0; i < occFreqs.length; i++) {
            for (int kInd = 0; kInd < kSmall; kInd++) {
                occFreqs[kNeighbors[i][kInd]]++;
            }
        }
        return occFreqs;
    }

    /**
     * This method calculates the k-nearest neighbor sets.
     *
     * @param k Integer that is the neighborhood size.
     */
    public void calculateNeighborSets(int k) {
        if (dset == null || dset.isEmpty() || distMatrix == null) {
            return;
        }
        currK = k;
        // Initialize the neighbor and reverse-neighbor lists and distance
        // arrays.
        kNeighbors = new int[dset.size()][k];
        kDistances = new float[dset.size()][k];
        kCurrLen = new int[dset.size()];
        reverseNeighbors = new ArrayList[dset.size()];
        for (int i = 0; i < dset.size(); i++) {
            reverseNeighbors[i] = new ArrayList<>(10 * k);
        }
        int l;
        // Calculate the kNN sets.
        for (int i = 0; i < dset.size(); i++) {
            for (int j = 0; j < distMatrix[i].length; j++) {
                int other = i + j + 1;
                if (kCurrLen[i] > 0) {
                    if (kCurrLen[i] == k) {
                        if (distMatrix[i][j] < kDistances[i][kCurrLen[i] - 1]) {
                            // Search and insert.
                            l = k - 1;
                            while ((l >= 1) && distMatrix[i][j]
                                    < kDistances[i][l - 1]) {
                                kDistances[i][l] = kDistances[i][l - 1];
                                kNeighbors[i][l] = kNeighbors[i][l - 1];
                                l--;
                            }
                            kDistances[i][l] = distMatrix[i][j];
                            kNeighbors[i][l] = i + j + 1;
                        }
                    } else {
                        if (distMatrix[i][j] < kDistances[i][kCurrLen[i] - 1]) {
                            // Search and insert.
                            l = kCurrLen[i] - 1;
                            kDistances[i][kCurrLen[i]] =
                                    kDistances[i][kCurrLen[i] - 1];
                            kNeighbors[i][kCurrLen[i]] =
                                    kNeighbors[i][kCurrLen[i] - 1];
                            while ((l >= 1) && distMatrix[i][j]
                                    < kDistances[i][l - 1]) {
                                kDistances[i][l] = kDistances[i][l - 1];
                                kNeighbors[i][l] = kNeighbors[i][l - 1];
                                l--;
                            }
                            kDistances[i][l] = distMatrix[i][j];
                            kNeighbors[i][l] = i + j + 1;
                            kCurrLen[i]++;
                        } else {
                            kDistances[i][kCurrLen[i]] = distMatrix[i][j];
                            kNeighbors[i][kCurrLen[i]] = i + j + 1;
                            kCurrLen[i]++;
                        }
                    }
                } else {
                    kDistances[i][0] = distMatrix[i][j];
                    kNeighbors[i][0] = i + j + 1;
                    kCurrLen[i] = 1;
                }
                if (kCurrLen[other] > 0) {
                    if (kCurrLen[other] == k) {
                        if (distMatrix[i][j] < kDistances[other][
                                kCurrLen[other] - 1]) {
                            // Search and insert.
                            l = k - 1;
                            while ((l >= 1) && distMatrix[i][j]
                                    < kDistances[other][l - 1]) {
                                kDistances[other][l] = kDistances[other][l - 1];
                                kNeighbors[other][l] = kNeighbors[other][l - 1];
                                l--;
                            }
                            kDistances[other][l] = distMatrix[i][j];
                            kNeighbors[other][l] = i;
                        }
                    } else {
                        if (distMatrix[i][j] < kDistances[other][
                                kCurrLen[other] - 1]) {
                            // Search and insert.
                            l = kCurrLen[other] - 1;
                            kDistances[other][kCurrLen[other]] =
                                    kDistances[other][kCurrLen[other] - 1];
                            kNeighbors[other][kCurrLen[other]] =
                                    kNeighbors[other][kCurrLen[other] - 1];
                            while ((l >= 1) && distMatrix[i][j]
                                    < kDistances[other][l - 1]) {
                                kDistances[other][l] = kDistances[other][l - 1];
                                kNeighbors[other][l] = kNeighbors[other][l - 1];
                                l--;
                            }
                            kDistances[other][l] = distMatrix[i][j];
                            kNeighbors[other][l] = i;
                            kCurrLen[other]++;
                        } else {
                            kDistances[other][kCurrLen[other]] =
                                    distMatrix[i][j];
                            kNeighbors[other][kCurrLen[other]] = i;
                            kCurrLen[other]++;
                        }
                    }
                } else {
                    kDistances[other][0] = distMatrix[i][j];
                    kNeighbors[other][0] = i;
                    kCurrLen[other] = 1;
                }
            }
        }
        // Count the neighbor occurrence frequencies.
        kNeighborFrequencies = new int[kNeighbors.length];
        kBadFrequencies = new int[kNeighbors.length];
        kGoodFrequencies = new int[kNeighbors.length];
        for (int i = 0; i < kNeighbors.length; i++) {
            for (int kInd = 0; kInd < k; kInd++) {
                reverseNeighbors[kNeighbors[i][kInd]].add(i);
                kNeighborFrequencies[kNeighbors[i][kInd]]++;
                if (dset.data.get(i).getCategory() != dset.data.get(
                        kNeighbors[i][kInd]).getCategory()) {
                    kBadFrequencies[kNeighbors[i][kInd]]++;
                } else {
                    kGoodFrequencies[kNeighbors[i][kInd]]++;
                }
            }
        }
        // Calculate the occurrence stats.
        meanOccFreq = 0;
        stDevOccFreq = 0;
        meanOccBadness = 0;
        stDevOccBadness = 0;
        meanOccGoodness = 0;
        stDevOccGoodness = 0;
        meanGoodMinusBadness = 0;
        stDevGoodMinusBadness = 0;
        meanRelativeGoodMinusBadness = 0;
        stDevRelativeGoodMinusBadness = 0;
        for (int i = 0; i < kBadFrequencies.length; i++) {
            meanOccFreq += kNeighborFrequencies[i];
            meanOccBadness += kBadFrequencies[i];
            meanOccGoodness += kGoodFrequencies[i];
            meanGoodMinusBadness += kGoodFrequencies[i] - kBadFrequencies[i];
            if (kNeighborFrequencies[i] > 0) {
                meanRelativeGoodMinusBadness += ((kGoodFrequencies[i]
                        - kBadFrequencies[i]) / kNeighborFrequencies[i]);
            } else {
                meanRelativeGoodMinusBadness += 1;
            }
        }
        meanOccFreq /= (float) kNeighborFrequencies.length;
        meanOccBadness /= (float) kBadFrequencies.length;
        meanOccGoodness /= (float) kGoodFrequencies.length;
        meanGoodMinusBadness /= (float) kGoodFrequencies.length;
        meanRelativeGoodMinusBadness /= (float) kGoodFrequencies.length;
        for (int i = 0; i < kBadFrequencies.length; i++) {
            stDevOccFreq += ((meanOccFreq - kNeighborFrequencies[i])
                    * (meanOccFreq - kNeighborFrequencies[i]));
            stDevOccBadness += ((meanOccBadness - kBadFrequencies[i])
                    * (meanOccBadness - kBadFrequencies[i]));
            stDevOccGoodness += ((meanOccGoodness - kGoodFrequencies[i])
                    * (meanOccGoodness - kGoodFrequencies[i]));
            stDevGoodMinusBadness += ((meanGoodMinusBadness
                    - (kGoodFrequencies[i] - kBadFrequencies[i]))
                    * (meanGoodMinusBadness - (kGoodFrequencies[i]
                    - kBadFrequencies[i])));
            if (kNeighborFrequencies[i] > 0) {
                stDevRelativeGoodMinusBadness += (meanRelativeGoodMinusBadness
                        - ((kGoodFrequencies[i] - kBadFrequencies[i])
                        / kNeighborFrequencies[i]))
                        * (meanRelativeGoodMinusBadness - ((kGoodFrequencies[i]
                        - kBadFrequencies[i]) / kNeighborFrequencies[i]));
            } else {
                stDevRelativeGoodMinusBadness +=
                        (meanRelativeGoodMinusBadness - 1)
                        * (meanRelativeGoodMinusBadness - 1);
            }
        }
        stDevOccFreq /= (float) kNeighborFrequencies.length;
        stDevOccBadness /= (float) kBadFrequencies.length;
        stDevOccGoodness /= (float) kGoodFrequencies.length;
        stDevGoodMinusBadness /= (float) kGoodFrequencies.length;
        stDevRelativeGoodMinusBadness /= (float) kGoodFrequencies.length;
        stDevOccFreq = Math.sqrt(stDevOccFreq);
        stDevOccBadness = Math.sqrt(stDevOccBadness);
        stDevOccGoodness = Math.sqrt(stDevOccGoodness);
        stDevGoodMinusBadness = Math.sqrt(stDevGoodMinusBadness);
        stDevRelativeGoodMinusBadness =
                Math.sqrt(stDevRelativeGoodMinusBadness);
    }

    /**
     * This method calculates the k-nearest neighbor sets, by also allowing for
     * distance corrections towards certain points. These corrections are
     * sometimes used in kNN variations for classification under class imbalance
     * and this method makes it possible to use them directly in kNN search
     * without having to modify the entire distance matrix. The corrections are
     * usually used for subtracting the radius to the closest 'enemy' point.
     *
     * @param k Integer that is the neighborhood size.
     * @param distanceCorrections float[] representing the distance corrections
     * to be subtracted from the actual distances to the points in kNN
     * calculations.
     */
    public void calculateNeighborSets(int k, float[] distanceCorrections) {
        if (dset == null || dset.isEmpty() || distMatrix == null) {
            return;
        }
        currK = k;
        // Initialize the kNN arrays.
        kNeighbors = new int[dset.size()][k];
        kDistances = new float[dset.size()][k];
        kCurrLen = new int[dset.size()];
        reverseNeighbors = new ArrayList[dset.size()];
        // Initialize the reverse neighbor lists.
        for (int i = 0; i < dset.size(); i++) {
            reverseNeighbors[i] = new ArrayList<>(10 * k);
        }
        int l;
        float distModified = 0;
        for (int i = 0; i < dset.size(); i++) {
            for (int j = 0; j < distMatrix[i].length; j++) {
                int other = i + j + 1;
                distModified = distMatrix[i][j] - distanceCorrections[i];
                if (kCurrLen[i] > 0) {
                    if (kCurrLen[i] == k) {
                        if (distModified < kDistances[i][kCurrLen[i] - 1]) {
                            // Search and insert.
                            l = k - 1;
                            while ((l >= 1) && distModified
                                    < kDistances[i][l - 1]) {
                                kDistances[i][l] = kDistances[i][l - 1];
                                kNeighbors[i][l] = kNeighbors[i][l - 1];
                                l--;
                            }
                            kDistances[i][l] = distModified;
                            kNeighbors[i][l] = i + j + 1;
                        }
                    } else {
                        if (distModified < kDistances[i][kCurrLen[i] - 1]) {
                            // Search and insert.
                            l = kCurrLen[i] - 1;
                            kDistances[i][kCurrLen[i]] =
                                    kDistances[i][kCurrLen[i] - 1];
                            kNeighbors[i][kCurrLen[i]] =
                                    kNeighbors[i][kCurrLen[i] - 1];
                            while ((l >= 1) && distModified
                                    < kDistances[i][l - 1]) {
                                kDistances[i][l] = kDistances[i][l - 1];
                                kNeighbors[i][l] = kNeighbors[i][l - 1];
                                l--;
                            }
                            kDistances[i][l] = distModified;
                            kNeighbors[i][l] = i + j + 1;
                            kCurrLen[i]++;
                        } else {
                            kDistances[i][kCurrLen[i]] = distModified;
                            kNeighbors[i][kCurrLen[i]] = i + j + 1;
                            kCurrLen[i]++;
                        }
                    }
                } else {
                    kDistances[i][0] = distModified;
                    kNeighbors[i][0] = i + j + 1;
                    kCurrLen[i] = 1;
                }
                distModified = distMatrix[i][j] - distanceCorrections[other];
                if (kCurrLen[other] > 0) {
                    if (kCurrLen[other] == k) {
                        if (distModified < kDistances[other][
                                kCurrLen[other] - 1]) {
                            // Search and insert.
                            l = k - 1;
                            while ((l >= 1) && distModified
                                    < kDistances[other][l - 1]) {
                                kDistances[other][l] = kDistances[other][l - 1];
                                kNeighbors[other][l] = kNeighbors[other][l - 1];
                                l--;
                            }
                            kDistances[other][l] = distModified;
                            kNeighbors[other][l] = i;
                        }
                    } else {
                        if (distModified < kDistances[other][
                                kCurrLen[other] - 1]) {
                            // Search and insert.
                            l = kCurrLen[other] - 1;
                            kDistances[other][kCurrLen[other]] =
                                    kDistances[other][kCurrLen[other] - 1];
                            kNeighbors[other][kCurrLen[other]] =
                                    kNeighbors[other][kCurrLen[other] - 1];
                            while ((l >= 1) && distModified
                                    < kDistances[other][l - 1]) {
                                kDistances[other][l] = kDistances[other][l - 1];
                                kNeighbors[other][l] = kNeighbors[other][l - 1];
                                l--;
                            }
                            kDistances[other][l] = distModified;
                            kNeighbors[other][l] = i;
                            kCurrLen[other]++;
                        } else {
                            kDistances[other][kCurrLen[other]] = distModified;
                            kNeighbors[other][kCurrLen[other]] = i;
                            kCurrLen[other]++;
                        }
                    }
                } else {
                    kDistances[other][0] = distModified;
                    kNeighbors[other][0] = i;
                    kCurrLen[other] = 1;
                }
            }
        }
        // Count the neighbor occurrence frequencies.
        kNeighborFrequencies = new int[kNeighbors.length];
        kBadFrequencies = new int[kNeighbors.length];
        kGoodFrequencies = new int[kNeighbors.length];
        for (int i = 0; i < kNeighbors.length; i++) {
            for (int kInd = 0; kInd < k; kInd++) {
                reverseNeighbors[kNeighbors[i][kInd]].add(i);
                kNeighborFrequencies[kNeighbors[i][kInd]]++;
                if (dset.data.get(i).getCategory() != dset.data.get(
                        kNeighbors[i][kInd]).getCategory()) {
                    kBadFrequencies[kNeighbors[i][kInd]]++;
                } else {
                    kGoodFrequencies[kNeighbors[i][kInd]]++;
                }
            }
        }
        // Calculate the neighbor occurrence stats.
        meanOccFreq = 0;
        stDevOccFreq = 0;
        meanOccBadness = 0;
        stDevOccBadness = 0;
        meanOccGoodness = 0;
        stDevOccGoodness = 0;
        meanGoodMinusBadness = 0;
        stDevGoodMinusBadness = 0;
        meanRelativeGoodMinusBadness = 0;
        stDevRelativeGoodMinusBadness = 0;
        for (int i = 0; i < kBadFrequencies.length; i++) {
            meanOccFreq += kNeighborFrequencies[i];
            meanOccBadness += kBadFrequencies[i];
            meanOccGoodness += kGoodFrequencies[i];
            meanGoodMinusBadness += kGoodFrequencies[i] - kBadFrequencies[i];
            if (kNeighborFrequencies[i] > 0) {
                meanRelativeGoodMinusBadness += ((kGoodFrequencies[i]
                        - kBadFrequencies[i]) / kNeighborFrequencies[i]);
            } else {
                meanRelativeGoodMinusBadness += 1;
            }
        }
        meanOccFreq /= (float) kNeighborFrequencies.length;
        meanOccBadness /= (float) kBadFrequencies.length;
        meanOccGoodness /= (float) kGoodFrequencies.length;
        meanGoodMinusBadness /= (float) kGoodFrequencies.length;
        meanRelativeGoodMinusBadness /= (float) kGoodFrequencies.length;
        for (int i = 0; i < kBadFrequencies.length; i++) {
            stDevOccFreq += ((meanOccFreq - kNeighborFrequencies[i])
                    * (meanOccFreq - kNeighborFrequencies[i]));
            stDevOccBadness += ((meanOccBadness - kBadFrequencies[i])
                    * (meanOccBadness - kBadFrequencies[i]));
            stDevOccGoodness += ((meanOccGoodness - kGoodFrequencies[i])
                    * (meanOccGoodness - kGoodFrequencies[i]));
            stDevGoodMinusBadness += ((meanGoodMinusBadness
                    - (kGoodFrequencies[i] - kBadFrequencies[i]))
                    * (meanGoodMinusBadness - (kGoodFrequencies[i]
                    - kBadFrequencies[i])));
            if (kNeighborFrequencies[i] > 0) {
                stDevRelativeGoodMinusBadness += (meanRelativeGoodMinusBadness
                        - ((kGoodFrequencies[i] - kBadFrequencies[i])
                        / kNeighborFrequencies[i]))
                        * (meanRelativeGoodMinusBadness - ((kGoodFrequencies[i]
                        - kBadFrequencies[i]) / kNeighborFrequencies[i]));
            } else {
                stDevRelativeGoodMinusBadness +=
                        (meanRelativeGoodMinusBadness - 1)
                        * (meanRelativeGoodMinusBadness - 1);
            }
        }
        stDevOccFreq /= (float) kNeighborFrequencies.length;
        stDevOccBadness /= (float) kBadFrequencies.length;
        stDevOccGoodness /= (float) kGoodFrequencies.length;
        stDevGoodMinusBadness /= (float) kGoodFrequencies.length;
        stDevRelativeGoodMinusBadness /= (float) kGoodFrequencies.length;
        stDevOccFreq = Math.sqrt(stDevOccFreq);
        stDevOccBadness = Math.sqrt(stDevOccBadness);
        stDevOccGoodness = Math.sqrt(stDevOccGoodness);
        stDevGoodMinusBadness = Math.sqrt(stDevGoodMinusBadness);
        stDevRelativeGoodMinusBadness =
                Math.sqrt(stDevRelativeGoodMinusBadness);
    }


    /**
     * This method returns the indexes of neighbor points that exceed the
     * specified occurrence frequency on the training data.
     * @param thresholdFrequency Integer that is the threshold frequency.
     * @return List of indexes of neighbor points that exceed a certain
     * occurrence frequency on the training data.
     * @throws Exception 
     */
    public ArrayList<Integer> getFrequentAtLeast(int thresholdFrequency)
            throws Exception {
        if (kNeighborFrequencies == null) {
            return null;
        } else {
            ArrayList<Integer> result = new ArrayList<>(
                    kNeighborFrequencies.length);
            for (int i = 0; i < kNeighborFrequencies.length; i++) {
                if (kNeighborFrequencies[i] >= thresholdFrequency) {
                    result.add(new Integer(i));
                }
            }
            return result;
        }
    }


    /**
     * @return Double value that is the mean of the bad neighbor occurrence
     * frequency.
     */
    public double getMeanNeighborBadness() {
        return meanOccBadness;
    }


    /**
     * @return Double value that is the standard deviation of the bad neighbor
     * occurrence frequency.
     */
    public double getNeighborBadnessStDev() {
        return stDevOccBadness;
    }


    /**
     * @return float[] representing the standardized bad neighbor occurrence
     * frequency scores.
     */
    public float[] getStandardizedBadOccFreqs() {
        float[] zScoredBadOccFreqs = new float[kBadFrequencies.length];
        for (int i = 0; i < kBadFrequencies.length; i++) {
            zScoredBadOccFreqs[i] = (kBadFrequencies[i] -
                    (float) meanOccBadness) / (float) stDevOccBadness;
        }
        return zScoredBadOccFreqs;
    }


    /**
     * @return float[] representing the exponential weights based on the
     * standardized bad neighbor occurrence frequency scores. This weighting
     * scheme was first used for vote weighting in hw-kNN, that was proposed
     * by Radovanovic and Nanopulous.
     */
    public float[] getHWKNNWeightingScheme() {
        float[] bFreqWeights = new float[kBadFrequencies.length];
        for (int i = 0; i < kBadFrequencies.length; i++) {
            bFreqWeights[i] = (float) Math.pow(Math.E, -((kBadFrequencies[i] -
                    (float) meanOccBadness) / (float) stDevOccBadness));
        }
        return bFreqWeights;
    }


    /**
     * @return float[] representing the exponential weights based on the
     * standardized bad neighbor occurrence frequency scores, with a maximum
     * value of 1 for each weight, achieved via cut-off. This is a slight
     * modification of the weighting that scheme was first used for vote
     * weighting in hw-kNN, proposed by Radovanovic and Nanopulous.
     */
    public float[] getMaxedAtOneHWKNNWeightingScheme() {
        float[] bFreqWeights = new float[kBadFrequencies.length];
        for (int i = 0; i < kBadFrequencies.length; i++) {
            if (stDevOccBadness > 0) {
                bFreqWeights[i] = Math.min((float) Math.pow(Math.E,
                        -((kBadFrequencies[i] - (float) meanOccBadness) /
                        (float) stDevOccBadness)), 1);
            } else {
                bFreqWeights[i] = 1;
            }
        }
        return bFreqWeights;
    }


    /**
     * @return float[] representing the bounded weights derived from the
     * difference between good and bad neighbor occurrence frequency for each
     * point.
     */
    public float[] getBoundedGoodMinusBadHubnessWeightingScheme() {
        return getBoundedGoodMinusBadHubnessWeightingScheme(0.2f, 1.8f);
    }
    

    /**
     * @param lowerWeightLimit Float value that is the lower weight limit.
     * @param upperWeightLimit Float value that is the upper weight limit.
     * @return float[] representing the bounded weights derived from the
     * difference between good and bad neighbor occurrence frequency for each
     * point.
     */
    public float[] getBoundedGoodMinusBadHubnessWeightingScheme(
            float lowerWeightLimit, float upperWeightLimit) {
        float[] weights = new float[kGoodFrequencies.length];
        for (int i = 0; i < kGoodFrequencies.length; i++) {
            if (stDevGoodMinusBadness > 0) {
                weights[i] = Math.min((float) Math.pow(Math.E,
                        (((kGoodFrequencies[i] - kBadFrequencies[i]) -
                        (float) meanGoodMinusBadness) /
                        (float) stDevGoodMinusBadness)), upperWeightLimit);
                weights[i] = Math.max(lowerWeightLimit, weights[i]);
            } else {
                weights[i] = 1;
            }
        }
        return weights;
    }


    /**
     * @return float[] weights based on the standardized relative difference
     * between good and bad neighbor occurrence frequencies, exponential.
     */
    public float[] getRelativeGoodMinusBadHubnessWeightingScheme() {
        float[] weights = new float[kGoodFrequencies.length];
        for (int i = 0; i < kGoodFrequencies.length; i++) {
            if (stDevGoodMinusBadness > 0) {
                if (kNeighborFrequencies[i] > 0) {
                    weights[i] = (float) Math.pow(Math.E,
                            ((((kGoodFrequencies[i] - kBadFrequencies[i]) /
                            kNeighborFrequencies[i]) -
                            (float) meanRelativeGoodMinusBadness) /
                            (float) stDevRelativeGoodMinusBadness));
                } else {
                    weights[i] = (float) Math.pow(Math.E,
                            ((1 - (float) meanRelativeGoodMinusBadness) /
                            (float) stDevRelativeGoodMinusBadness));
                }
            } else {
                weights[i] = 1;
            }
        }
        return weights;
    }


    /**
     * @return float[] representing the weights for exponential standardized
     * neighbor occurrence frequency penalization.
     */
    public float[] getPenalizeHubnessWeightingScheme() {
        float[] weights = new float[kGoodFrequencies.length];
        float stFull;
        for (int i = 0; i < kBadFrequencies.length; i++) {
            stFull = (kGoodFrequencies[i] + kBadFrequencies[i] -
                    (float) meanOccFreq) / (float) stDevOccFreq;
            weights[i] = (float) Math.pow(Math.E, -stFull);
        }
        return weights;
    }


    /**
     * @return float[] representing the weights for the exponential standardized
     * neighbor occurrence frequency emphasis.
     */
    public float[] getRewardHubnessWeightingScheme() {
        float[] weights = new float[kGoodFrequencies.length];
        float stFull;
        for (int i = 0; i < kBadFrequencies.length; i++) {
            stFull = (kGoodFrequencies[i] + kBadFrequencies[i] -
                    (float) meanOccFreq) / (float) stDevOccFreq;
            weights[i] = (float) Math.pow(Math.E, stFull);
        }
        return weights;
    }


    /**
     * Calculates k-entropies, which measure how mixed the labels are in the
     * k-neighborhoods of points
     *
     * @param numCategories Integer representing the number of classes in the
     * data.
     * @param neighborhoodSize Integer that is the neighborhood size to use
     * for calculating the k-entropies. It has to be lower than the k of the
     * current k-NN sets.
     */
    public void calculateKEntropies(int numCategories, int neighborhoodSize) {
        if (kNeighbors == null) {
            return;
        }
        float[] categoryFrequencies = new float[numCategories];
        kEntropies = new float[dset.data.size()];
        int currCat;
        float factor;
        for (int i = 0; i < kEntropies.length; i++) {
            for (int kInd = 0; kInd < Math.min(neighborhoodSize,
                    kNeighbors[0].length); kInd++) {
                currCat = dset.data.get(kNeighbors[i][kInd]).getCategory();
                if (currCat >= 0) {
                    categoryFrequencies[currCat]++;
                }
            }
            // Calculate the entropy.
            for (int cInd = 0; cInd < categoryFrequencies.length; cInd++) {
                if (categoryFrequencies[cInd] > 0) {
                    factor = categoryFrequencies[cInd] /
                            (float) neighborhoodSize;
                    kEntropies[i] -= factor * BasicMathUtil.log2(factor);
                }
            }
            // Reset the category frequency auxiliary array.
            for (int cInd = 0; cInd < numCategories; cInd++) {
                categoryFrequencies[cInd] = 0;
            }
        }
    }


    /**
     * This method calculate the entropies of the reverse neighbor sets, taking
     * into account different category weights.
     * @param numCategories Integer that is the number of classes in the data.
     * @param categoryWeights float[] representing different category weights.
     */
    public void calculateReverseNeighborEntropiesWeighted(int numCategories,
            float[] categoryWeights) {
        float[] categoryFrequencies = new float[numCategories];
        kRNNEntropies = new float[dset.data.size()];
        int currCat;
        float factor;
        for (int i = 0; i < kRNNEntropies.length; i++) {
            if (reverseNeighbors[i] == null ||
                    reverseNeighbors[i].size() <= 1) {
                for (int cInd = 0; cInd < numCategories; cInd++) {
                    categoryFrequencies[cInd] = 0;
                }
                kRNNEntropies[i] = 0;
                continue;
            } else if (reverseNeighbors[i] != null) {
                try {
                    for (int rInd = 0; rInd < reverseNeighbors[i].size();
                            rInd++) {
                        currCat = dset.data.get(
                                reverseNeighbors[i].get(rInd)).getCategory();
                        if (currCat >= 0) {
                            categoryFrequencies[currCat]++;
                        }
                    }
                    float denominator = 0;
                    for (int cInd = 0; cInd < numCategories; cInd++) {
                        denominator += Math.max(categoryWeights[cInd],
                                0.0000001f) * categoryFrequencies[cInd] /
                                (float) reverseNeighbors[i].size();
                    }
                    // Calculate eh entropy.
                    for (int cInd = 0; cInd < categoryFrequencies.length;
                            cInd++) {
                        if (categoryFrequencies[cInd] > 0) {
                            factor = (Math.max(categoryWeights[cInd],
                                    0.0000001f) * categoryFrequencies[cInd] /
                                    (float) reverseNeighbors[i].size()) /
                                    denominator;
                            kRNNEntropies[i] -=
                                    factor * BasicMathUtil.log2(factor);
                        }
                    }
                } catch (Exception e) {
                    System.err.println(e.getMessage());
                }
                // Reset the category frequency auxiliary array.
                for (int cInd = 0; cInd < numCategories; cInd++) {
                    categoryFrequencies[cInd] = 0;
                }
            }
        }
    }


    /**
     * This method calculate the entropies of the reverse neighbor sets.
     * @param numCategories Integer that is the number of classes in the data.
     */
    public void calculateReverseNeighborEntropies(int numCategories) {
        float[] categoryFrequencies = new float[numCategories];
        kRNNEntropies = new float[dset.data.size()];
        int currCat;
        float factor;
        for (int i = 0; i < kRNNEntropies.length; i++) {
            if (reverseNeighbors[i] == null ||
                    reverseNeighbors[i].size() <= 1) {
                for (int cInd = 0; cInd < numCategories; cInd++) {
                    categoryFrequencies[cInd] = 0;
                }
                kRNNEntropies[i] = 0;
                continue;
            } else if (reverseNeighbors[i] != null) {
                try {
                    for (int rInd = 0; rInd < reverseNeighbors[i].size();
                            rInd++) {
                        currCat = dset.data.get(
                                reverseNeighbors[i].get(rInd)).getCategory();
                        if (currCat >= 0) {
                            categoryFrequencies[currCat]++;
                        }
                    }
                    // Calculate the entropy.
                    for (int cInd = 0; cInd < categoryFrequencies.length;
                            cInd++) {
                        if (categoryFrequencies[cInd] > 0) {
                            factor = categoryFrequencies[cInd] /
                                    (float) reverseNeighbors[i].size();
                            kRNNEntropies[i] -=
                                    factor * BasicMathUtil.log2(factor);
                        }
                    }
                } catch (Exception e) {
                    System.err.println(e.getMessage());
                }
                // Reset the category frequency auxiliary array.
                for (int cInd = 0; cInd < numCategories; cInd++) {
                    categoryFrequencies[cInd] = 0;
                }
            }
        }
    }


    /**
     * @return float[] representing the entropies of the kNN sets.
     */
    public float[] getKEntropies() {
        return kEntropies;
    }


    /**
     * @return int[][] representing the array of arrays of indexes of k-nearest
     * neighbors of points on the training data.
     */
    public int[][] getKNeighbors() {
        return kNeighbors;
    }
    
    
    /**
     * @return float[] representing the entropies of the reverse kNN sets.
     */
    public float[] getReverseNeighborEntropies() {
        return kRNNEntropies;
    }


    /**
     * @param thresholdFreq Integer that is the occurrence frequency threshold.
     * @return Float representing the percentage of points that occur as
     * neighbors with a frequency that is greater or equal compared to the
     * provided threshold value.
     */
    public float getPercFrequentAtLeast(int thresholdFreq) {
        float count = 0;
        for (int i = 0; i < kNeighborFrequencies.length; i++) {
            if (kNeighborFrequencies[i] >= thresholdFreq) {
                count++;
            }
        }
        return count / (float) kNeighborFrequencies.length;
    }


    /**
     * @param thresholdFreq Integer that is the occurrence frequency threshold.
     * @return Float representing the percentage of points that occur as
     * neighbors with a frequency that is less or equal compared to the
     * provided threshold value.
     */
    public float getPercFrequentLessOrEqualThan(int thresholdFreq) {
        float count = 0;
        for (int i = 0; i < kNeighborFrequencies.length; i++) {
            if (kNeighborFrequencies[i] <= thresholdFreq) {
                count++;
            }
        }
        return count / (float) kNeighborFrequencies.length;
    }


    /**
     * This method re-calculates the occurrence stats for a smaller neighborhood
     * size. This way, one NeighborSetFinder object can be used to derive the
     * neighbor stats for any k-value up until the length of the calculated
     * kNN sets. The operating k value is saved to the currK variable.
     * @param kSmall Integer that is the smaller neighborhood size.
     */
    public void recalculateStatsForSmallerK(int kSmall) {
        if (kNeighbors == null) {
            return;
        }
        if (kSmall > kNeighbors[0].length) {
            // In case an inappropriately large value was provided.
            kSmall = kNeighbors[0].length;
        }
        // Indicate the new k value as the operating k value.
        currK = kSmall;
        // Re-initialize the occurrence frequency arrays and the reverse
        // neighbor lists.
        kNeighborFrequencies = new int[kNeighbors.length];
        kBadFrequencies = new int[kNeighbors.length];
        kGoodFrequencies = new int[kNeighbors.length];
        reverseNeighbors = new ArrayList[kNeighbors.length];
        for (int i = 0; i < reverseNeighbors.length; i++) {
            reverseNeighbors[i] = new ArrayList<>(10 * kSmall);
        }
        // Populate the reverse neighbor lists and count the occurrences.
        for (int i = 0; i < kNeighbors.length; i++) {
            if (reverseNeighbors[i] == null) {
                reverseNeighbors[i] = new ArrayList<>(10 * kSmall);
            }
            for (int kInd = 0; kInd < kSmall; kInd++) {
                kNeighborFrequencies[kNeighbors[i][kInd]]++;
                if (reverseNeighbors[kNeighbors[i][kInd]] == null) {
                    reverseNeighbors[kNeighbors[i][kInd]] =
                            new ArrayList<>(10 * kSmall);
                }
                reverseNeighbors[kNeighbors[i][kInd]].add(i);
                if (dset.data.get(i).getCategory() !=
                        dset.data.get(kNeighbors[i][kInd]).getCategory()) {
                    kBadFrequencies[kNeighbors[i][kInd]]++;
                } else {
                    kGoodFrequencies[kNeighbors[i][kInd]]++;
                }
            }
        }
        // Calculate the neighbor occurrence stats.
        meanOccFreq = 0;
        stDevOccFreq = 0;
        meanOccBadness = 0;
        stDevOccBadness = 0;
        meanOccGoodness = 0;
        stDevOccGoodness = 0;
        meanGoodMinusBadness = 0;
        stDevGoodMinusBadness = 0;
        meanRelativeGoodMinusBadness = 0;
        stDevRelativeGoodMinusBadness = 0;
        for (int i = 0; i < kBadFrequencies.length; i++) {
            meanOccFreq += kNeighborFrequencies[i];
            meanOccBadness += kBadFrequencies[i];
            meanOccGoodness += kGoodFrequencies[i];
            meanGoodMinusBadness += kGoodFrequencies[i] - kBadFrequencies[i];
            if (kNeighborFrequencies[i] > 0) {
                meanRelativeGoodMinusBadness += ((kGoodFrequencies[i]
                        - kBadFrequencies[i]) / kNeighborFrequencies[i]);
            } else {
                meanRelativeGoodMinusBadness += 1;
            }
        }
        meanOccFreq /= (float) kNeighborFrequencies.length;
        meanOccBadness /= (float) kBadFrequencies.length;
        meanOccGoodness /= (float) kGoodFrequencies.length;
        meanGoodMinusBadness /= (float) kGoodFrequencies.length;
        meanRelativeGoodMinusBadness /= (float) kGoodFrequencies.length;
        for (int i = 0; i < kBadFrequencies.length; i++) {
            stDevOccFreq += ((meanOccFreq - kNeighborFrequencies[i])
                    * (meanOccFreq - kNeighborFrequencies[i]));
            stDevOccBadness += ((meanOccBadness - kBadFrequencies[i])
                    * (meanOccBadness - kBadFrequencies[i]));
            stDevOccGoodness += ((meanOccGoodness - kGoodFrequencies[i])
                    * (meanOccGoodness - kGoodFrequencies[i]));
            stDevGoodMinusBadness += ((meanGoodMinusBadness
                    - (kGoodFrequencies[i] - kBadFrequencies[i]))
                    * (meanGoodMinusBadness - (kGoodFrequencies[i]
                    - kBadFrequencies[i])));
            if (kNeighborFrequencies[i] > 0) {
                stDevRelativeGoodMinusBadness += (meanRelativeGoodMinusBadness
                        - ((kGoodFrequencies[i] - kBadFrequencies[i])
                        / kNeighborFrequencies[i]))
                        * (meanRelativeGoodMinusBadness - ((kGoodFrequencies[i]
                        - kBadFrequencies[i]) / kNeighborFrequencies[i]));
            } else {
                stDevRelativeGoodMinusBadness +=
                        (meanRelativeGoodMinusBadness - 1)
                        * (meanRelativeGoodMinusBadness - 1);
            }
        }
        stDevOccFreq /= (float) kNeighborFrequencies.length;
        stDevOccBadness /= (float) kBadFrequencies.length;
        stDevOccGoodness /= (float) kGoodFrequencies.length;
        stDevGoodMinusBadness /= (float) kGoodFrequencies.length;
        stDevRelativeGoodMinusBadness /= (float) kGoodFrequencies.length;
        stDevOccFreq = Math.sqrt(stDevOccFreq);
        stDevOccBadness = Math.sqrt(stDevOccBadness);
        stDevOccGoodness = Math.sqrt(stDevOccGoodness);
        stDevGoodMinusBadness = Math.sqrt(stDevGoodMinusBadness);
        stDevRelativeGoodMinusBadness =
                Math.sqrt(stDevRelativeGoodMinusBadness);
    }

    
    /**
     * This method calculates the weights for the simhub secondary
     * shared-neighbor similarity measure.
     * 
     * @param numCategories Integer that is the number of categories in the
     * data.
     * @param theta Float parameter.
     * @return float[] representing the weights to be used for shared neighbors
     * in the simhub secondary similarity measure.
     */
    public float[] getSimhubWeightingScheme(int numCategories,
            float theta) {
        int size = dset.size();
        float[] weights = new float[size];
        float maxEntropy = (float) BasicMathUtil.log2((float) numCategories);
        if (kRNNEntropies == null) {
            calculateReverseNeighborEntropies(numCategories);
        }
        float maxWeight = 1;
        for (int i = 0; i < size; i++) {
            weights[i] = (float) BasicMathUtil.log2(((float) size) /
                    ((float) kNeighborFrequencies[i] + 1)) *
                    (maxEntropy - kRNNEntropies[i] + theta);
            maxWeight = Math.max(Math.abs(weights[i]), maxWeight);
        }
        // Normalization.
        for (int i = 0; i < size; i++) {
            weights[i] /= maxWeight;
        }
        return weights;
    }


    /**
     * This method calculates the weights for the simhub secondary
     * shared-neighbor similarity measure.
     * 
     * @param theta Float parameter.
     * @return float[] representing the weights to be used for shared neighbors
     * in the simhub secondary similarity measure.
     */
    public float[] getSimhubWeightingScheme(float theta) {
        int numCategories = dset.countCategories();
        return getSimhubWeightingScheme(numCategories, theta);
    }


    /**
     * This method calculates the weights for the simhub secondary
     * shared-neighbor similarity measure information component - an
     * unsupervised part of the total weight.
     * 
     * @param theta Float parameter.
     * @return float[] representing the weights to be used for shared neighbors
     * in the simhub secondary similarity measure information component - an
     * unsupervised part of the total weight.
     */
    public float[] getSimhubWeightingSchemeUnsupervisedComponent(float theta) {
        int size = dset.size();
        float[] weights = new float[size];
        int numCategories = dset.countCategories();
        if (kRNNEntropies == null) {
            calculateReverseNeighborEntropies(numCategories);
        }
        float maxWeight = 1;
        for (int i = 0; i < size; i++) {
            weights[i] = (float) BasicMathUtil.log2(((float) size) /
                    ((float) kNeighborFrequencies[i] + 1));
            maxWeight = Math.max(Math.abs(weights[i]), maxWeight);
        }
        for (int i = 0; i < size; i++) {
            weights[i] /= maxWeight;
        }
        return weights;
    }


    /**
     * This method calculates the weights for the simhub secondary
     * shared-neighbor similarity measure purity component - the supervised part
     * of the total weight.
     * 
     * @param theta Float parameter.
     * @return float[] representing the weights to be used for shared neighbors
     * in the simhub secondary similarity measure purity component - the
     * supervised part of the total weight.
     */
    public float[] getSimhubWeightingSchemeSupervisedComponent(float theta) {
        int size = dset.size();
        float[] weights = new float[size];
        int numCategories = dset.countCategories();
        float maxEntropy = (float) BasicMathUtil.log2((float) numCategories);
        if (kRNNEntropies == null) {
            calculateReverseNeighborEntropies(numCategories);
        }
        float maxWeight = 1;
        for (int i = 0; i < size; i++) {
            weights[i] = (maxEntropy - kRNNEntropies[i] + theta);
            maxWeight = Math.max(Math.abs(weights[i]), maxWeight);
        }
        for (int i = 0; i < size; i++) {
            weights[i] /= maxWeight;
        }
        return weights;
    }


    /**
     * This method calculates an alternative type of weights for the simhub
     * secondary shared-neighbor similarity measure, where each weight is either
     * 0 or 1, depending on how the total shared-neighbor contribution of the
     * neighbor point is judged. More details are available in the original
     * paper.
     * @return float[] representing a 0-1 type of weights for shared-neighbor
     * distances.
     */
    public float[] getSimhubAlternateWeights01() {
        int size = dset.size();
        float[] weights = new float[size];
        int numCategories = dset.countCategories();
        float[][] classConditionalCounts =
                this.getClassDataNeighborRelationNonNormalized(currK,
                numCategories, false);
        float indicator;
        float classConditionalSum;
        for (int i = 0; i < size; i++) {
            classConditionalSum = 0;
            for (int cIndex = 0; cIndex < numCategories; cIndex++) {
                classConditionalSum += classConditionalCounts[cIndex][i] *
                        classConditionalCounts[cIndex][i];
            }
            classConditionalSum *= 1.5f;
            indicator = classConditionalSum - kNeighborFrequencies[i] *
                    kNeighborFrequencies[i];
            if (indicator > 0) {
                weights[i] = 1;
            } else {
                weights[i] = 0;
            }
        }
        return weights;
    }


    /**
     * This method calculates an alternative type of weights for the simhub
     * secondary shared-neighbor similarity measure, where each weight is
     * proportional to the estimate of the goodness of the contribution of the
     * shared neighbor to the intra-class similarities and the badness of its
     * contribution to the inter-class similarities.
     * @param k Integer that is the current neighborhood size.
     * @return float[] representing a goodness-proportional shared-neighbor
     * weights.
     */
    public float[] getSimhubAlternateWeightsGoodnessProportional(int k) {
        int size = dset.size();
        float[] weights = new float[size];
        int numCategories = dset.countCategories();
        float[][] classConditionalOccurrenceCounts =
                this.getClassDataNeighborRelationNonNormalized(
                k, numCategories, false);
        float maxWeight = 0;
        float minWeight = 0;
        float indicator;
        float classConditionalSum;
        for (int i = 0; i < size; i++) {
            classConditionalSum = 0;
            for (int j = 0; j < numCategories; j++) {
                classConditionalSum +=
                        classConditionalOccurrenceCounts[j][i] *
                        classConditionalOccurrenceCounts[j][i];
            }
            classConditionalSum *= 1.5f;
            indicator = classConditionalSum - this.kNeighborFrequencies[i] *
                    this.kNeighborFrequencies[i];
            weights[i] = indicator;
            maxWeight = Math.max(weights[i], maxWeight);
            minWeight = Math.min(weights[i], minWeight);
        }
        for (int i = 0; i < size; i++) {
            weights[i] -= minWeight;
            weights[i] /= (maxWeight - minWeight);
        }
        return weights;
    }


    /**
     * This method calculates an alternative type of weights for the simhub
     * secondary shared-neighbor similarity measure, where each weight is
     * proportional to the estimate of the goodness of the contribution of the
     * shared neighbor to the intra-class similarities and the badness of its
     * contribution to the inter-class similarities.
     * @return float[] representing a goodness-proportional shared-neighbor
     * weights.
     */
    public float[] getSimhubAlternateWeightsGoodnessProportional() {
        return getSimhubAlternateWeightsGoodnessProportional(currK);
    }


    /**
     * @return float[] representing the percentages of label mismatches in
     * k-nearest neighbor sets for all k values from 1 up until the k value that
     * is the length of the current calculated k-nearest neighbor sets.
     */
    public float[] getLabelMismatchPercsAllK() {
        int kMax = kNeighbors[0].length;
        int size = dset.size();
        float mismatchCounter = 0;
        float[] bhArray = new float[kMax];
        for (int kInd = 0; kInd < kMax; kInd++) {
            for (int i = 0; i < size; i++) {
                if (dset.getLabelOf(i) != dset.getLabelOf(
                        kNeighbors[i][kInd])) {
                    mismatchCounter++;
                }
            }
            bhArray[kInd] = mismatchCounter / ((float) (size * (kInd + 1)));
        }
        return bhArray;
    }
    
    /**
     * @return float[] representing the percentages of label mismatches in
     * k-nearest neighbor sets for all k values from 1 up until the k value that
     * is the length of the current calculated k-nearest neighbor sets.
     */
    public float[] getLabelMismatchPercsAllK(int kMax) {
        int size = dset.size();
        float mismatchCounter = 0;
        float[] bhArray = new float[kMax];
        for (int kInd = 0; kInd < kMax; kInd++) {
            for (int i = 0; i < size; i++) {
                if (dset.getLabelOf(i) != dset.getLabelOf(
                        kNeighbors[i][kInd])) {
                    mismatchCounter++;
                }
            }
            bhArray[kInd] = mismatchCounter / ((float) (size * (kInd + 1)));
        }
        return bhArray;
    }


    /**
     * @param kVal Integer that is the neighborhood size to do the calculations
     * for.
     * @return float[][] representing the upper triangular matrix of neighbor
     * co-occurrence counts. The matrix value at i,j represents the
     * co-occurrence count of point i and i + j + 1. 
     */
    public float[][] getKCooccurrences(int kVal) {
        int size = kNeighbors.length;
        // Initialization.
        float[][] kCooccurrences = new float[size][];
        for (int i = 0; i < size; i++) {
            kCooccurrences[i] = new float[size - i - 1];
        }
        int minIndex, maxIndex;
        for (int i = 0; i < size; i++) {
            for (int kIndFirst = 0; kIndFirst < kVal - 1; kIndFirst++) {
                for (int kIndSecond = kIndFirst + 1; kIndSecond < kVal;
                        kIndSecond++) {
                    minIndex = Math.min(kNeighbors[i][kIndFirst],
                            kNeighbors[i][kIndSecond]);
                    maxIndex = Math.max(kNeighbors[i][kIndFirst],
                            kNeighbors[i][kIndSecond]);
                    kCooccurrences[minIndex][maxIndex - minIndex - 1]++;
                }
            }
        }
        return kCooccurrences;
    }
}
