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

import data.representation.DataSet;
import distances.primary.CombinedMetric;
import ioformat.SupervisedLoader;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import learning.supervised.Category;
import util.BasicMathUtil;
import util.CommandLineParser;
import util.SOPLUtil;

/**
 * This class implements the methods for calculating the hit-miss neighbor sets 
 * on the training data in supervised learning. Hits are those neighbor 
 * occurrences where labels match and misses are the neighbor occurrences where 
 * the labels differ. Nearest miss distances can be used to estimate the margins
 * in 1-NN classification and can be used for large-margin NN instance selection
 * and classification.
 * 
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class HitMissNetwork {
    
    public static final int DEFAULT_NEIGHBORHOOD_SIZE = 1;
    // Data to generate the network for.
    private DataSet dset;
    // The upper triangular distance matrix.
    private float[][] dMat;
    // Neighborhood size to use.
    private int k = DEFAULT_NEIGHBORHOOD_SIZE;
    
    // Tabu maps for restricted kNN search within the class or outside the 
    // class.
    private HashMap<Integer, Integer>[] tabuMapsClassQueries;
    private HashMap<Integer, Integer>[] tabuMapsClassComplementQueries;
    
    // The kNN sets for the hits.
    private int[][] knHits;
    private float[] hitNeighbOccFreqs;
    private List<Integer>[] hitReverseNNSets;
    // The kNN sets for the misses.
    private int[][] knMisses;
    private float[] missNeighbOccFreqs;
    private List<Integer>[] missReverseNNSets;
    
    /**
     * Default constructor.
     */
    public HitMissNetwork() {
    }
    
    /**
     * Initialization.
     * 
     * @param dset DataSet object to generate the network for.
     * @param dMat float[][] that is the upper triangular distance matrix.
     */
    public HitMissNetwork(DataSet dset, float[][] dMat) {
        this.dset = dset;
        this.dMat = dMat;
    }
    
    /**
     * Initialization.
     * 
     * @param dset DataSet object to generate the network for.
     * @param dMat float[][] that is the upper triangular distance matrix.
     * @param k Integer that is the neighborhood size.
     */
    public HitMissNetwork(DataSet dset, float[][] dMat, int k) {
        this.dset = dset;
        this.dMat = dMat;
        this.k = k;
    }
    
    /**
     * This method generates the hit/miss network.
     * @throws Exception 
     */
    public void generateNetwork() throws Exception {
        // First check for trivial cases.
        if (dset == null || dset.isEmpty()) {
            throw new Exception("No data provided.");
        }
        int dSize = dset.size();
        if (k <= 0 || k > dSize) {
            throw new Exception("Bad neighborhood size provided: " + k);
        }
        if (dMat == null) {
            throw new Exception("No distance matrix provided.");
        }
        int numClasses = dset.countCategories();
        if (numClasses == 1) {
            throw new Exception("Only one class detected in the data. Use "
                    + "standard kNN extraction methods instead.");
        }
        // Initialize class-specific tabu maps for kNN extraction.
        Category[] classes = dset.getClassesArray(numClasses);
        int minClassSize = dSize;
        for (int c = 0; c < numClasses; c++) {
            if (classes[c].size() < minClassSize) {
                minClassSize = classes[c].size();
            }
        }
        if (k > minClassSize) {
            throw new Exception("Specified neighborhood size exceeds minimum "
                    + "class size. Unable to form hit networks for k: " + k);
        }
        tabuMapsClassQueries = new HashMap[dSize];
        tabuMapsClassComplementQueries = new HashMap[dSize];
        for (int c = 0; c < numClasses; c++) {
            tabuMapsClassQueries[c] = new HashMap<>(dSize);
            for (int cOther = 0; cOther < c; cOther++) {
                for (int index: classes[cOther].getIndexes()) {
                    tabuMapsClassQueries[c].put(index, cOther);
                }
            }
            for (int cOther = c + 1; cOther < numClasses; cOther++) {
                for (int index: classes[cOther].getIndexes()) {
                    tabuMapsClassQueries[c].put(index, cOther);
                }
            }
            tabuMapsClassComplementQueries[c] = new HashMap<>(dSize);
            for (int index: classes[c].getIndexes()) {
                tabuMapsClassComplementQueries[c].put(index, c);
            }
        }
        // Further initialization.
        knHits = new int[dSize][k];
        hitNeighbOccFreqs = new float[dSize];
        hitReverseNNSets = new List[dSize];
        for (int i = 0; i < dSize; i++) {
            hitReverseNNSets[i] = new ArrayList<>(k);
        }
        knMisses = new int[dSize][k];
        missNeighbOccFreqs = new float[dSize];
        missReverseNNSets = new List[dSize];
        for (int i = 0; i < dSize; i++) {
            missReverseNNSets[i] = new ArrayList<>(k);
        }
        // Now calculate the hit and miss kNN sets and incrementally update the 
        // occurrence stats.
        for (int i = 0; i < dSize; i++) {
            knHits[i] = NeighborSetFinder.getIndexesOfNeighbors(
                    dMat, i, k, tabuMapsClassQueries[dset.getLabelOf(i)]);
            knMisses[i] = NeighborSetFinder.getIndexesOfNeighbors(
                    dMat, i, k, tabuMapsClassComplementQueries[
                    dset.getLabelOf(i)]);
            for (int kIndex = 0; kIndex < k; kIndex++) {
                hitNeighbOccFreqs[knHits[i][kIndex]]++;
                missNeighbOccFreqs[knMisses[i][kIndex]]++;
                hitReverseNNSets[knHits[i][kIndex]].add(i);
                missReverseNNSets[knMisses[i][kIndex]].add(i);
            }
        }
    }
    
    /**
     * This method 'recycles' some of the existing kNN information.
     * 
     * @param nsf NeighborSetFinder object holding some pre-computed kNN info.
     * @throws Exception 
     */
    public void generateNetworkFromExistingNSF(NeighborSetFinder nsf) 
            throws Exception {
        if (nsf == null) {
            generateNetwork();
            return;
        }
        int kNSF = nsf.getKNeighbors() != null ? nsf.getKNeighbors().length : 0;
        if (kNSF == 0) {
            generateNetwork();
            return;
        }
        // First check for trivial cases.
        if (dset == null || dset.isEmpty()) {
            throw new Exception("No data provided.");
        }
        int dSize = dset.size();
        if (k <= 0 || k > dSize) {
            throw new Exception("Bad neighborhood size provided: " + k);
        }
        if (dMat == null) {
            throw new Exception("No distance matrix provided.");
        }
        int numClasses = dset.countCategories();
        if (numClasses == 1) {
            throw new Exception("Only one class detected in the data. Use "
                    + "standard kNN extraction methods instead.");
        }
        // Initialize class-specific tabu maps for kNN extraction.
        Category[] classes = dset.getClassesArray(numClasses);
        int minClassSize = dSize;
        for (int c = 0; c < numClasses; c++) {
            if (classes[c].size() < minClassSize) {
                minClassSize = classes[c].size();
            }
        }
        if (k > minClassSize) {
            throw new Exception("Specified neighborhood size exceeds minimum "
                    + "class size. Unable to form hit networks for k: " + k);
        }
        tabuMapsClassQueries = new HashMap[dSize];
        tabuMapsClassComplementQueries = new HashMap[dSize];
        for (int c = 0; c < numClasses; c++) {
            tabuMapsClassQueries[c] = new HashMap<>(dSize);
            for (int cOther = 0; cOther < c; cOther++) {
                for (int index: classes[cOther].getIndexes()) {
                    tabuMapsClassQueries[c].put(index, cOther);
                }
            }
            for (int cOther = c + 1; cOther < numClasses; cOther++) {
                for (int index: classes[cOther].getIndexes()) {
                    tabuMapsClassQueries[c].put(index, cOther);
                }
            }
            tabuMapsClassComplementQueries[c] = new HashMap<>(dSize);
            for (int index: classes[c].getIndexes()) {
                tabuMapsClassComplementQueries[c].put(index, c);
            }
        }
        // Further initialization.
        knHits = new int[dSize][k];
        hitNeighbOccFreqs = new float[dSize];
        hitReverseNNSets = new List[dSize];
        for (int i = 0; i < dSize; i++) {
            hitReverseNNSets[i] = new ArrayList<>(k);
        }
        knMisses = new int[dSize][k];
        missNeighbOccFreqs = new float[dSize];
        missReverseNNSets = new List[dSize];
        for (int i = 0; i < dSize; i++) {
            missReverseNNSets[i] = new ArrayList<>(k);
        }
        // Now re-use or calculate the hit and miss kNN sets and incrementally 
        // update the occurrence stats.
        int[][] givenKNNSets = nsf.getKNeighbors();
        int[] extraHits, extraMisses;
        for (int i = 0; i < dSize; i++) {
            int queryLabel = dset.getLabelOf(i);
            int numHits = 0;
            int numMisses = 0;
            for (int kIndex = 0; kIndex < kNSF; kIndex++) {
                int neighborLabel = dset.getLabelOf(givenKNNSets[i][kIndex]);
                if (neighborLabel == queryLabel && numHits < k) {
                    knHits[i][numHits] = givenKNNSets[i][kIndex];
                } else if (neighborLabel != queryLabel && numMisses < k) {
                    knMisses[i][numMisses] = givenKNNSets[i][kIndex];
                } else if (numHits == k && numMisses == k) {
                    break;
                }
            }
            // Check if we are done or we need to calculate some more neighbors.
            if (numHits < k) {
                // Update the tabu before and after the query, to avoid 
                // returning the known neighbor points.
                for (int kIndex = 0; kIndex < numHits; kIndex++) {
                    tabuMapsClassQueries[queryLabel].put(knHits[i][kIndex],
                            dset.getLabelOf(knHits[i][kIndex]));
                }
                int kDiff = k - numHits;
                extraHits = NeighborSetFinder.getIndexesOfNeighbors(
                        dMat, i, kDiff,
                        tabuMapsClassQueries[dset.getLabelOf(i)]);
                // Remove the known neighbors from the tabu map.
                for (int kIndex = 0; kIndex < numHits; kIndex++) {
                    tabuMapsClassQueries[queryLabel].remove(knHits[i][kIndex]);
                }
                System.arraycopy(extraHits, 0, knHits[i], numHits, kDiff);
            }
            if (numMisses < k) {
                // Update the tabu before and after the query, to avoid 
                // returning the known neighbor points.
                for (int kIndex = 0; kIndex < numMisses; kIndex++) {
                    tabuMapsClassComplementQueries[queryLabel].put(
                            knMisses[i][kIndex],
                            dset.getLabelOf(knMisses[i][kIndex]));
                }
                int kDiff = k - numMisses;
                extraMisses = NeighborSetFinder.getIndexesOfNeighbors(
                        dMat, i, kDiff,
                        tabuMapsClassComplementQueries[dset.getLabelOf(i)]);
                // Remove the known neighbors from the tabu map.
                for (int kIndex = 0; kIndex < numMisses; kIndex++) {
                    tabuMapsClassComplementQueries[queryLabel].remove(
                            knMisses[i][kIndex]);
                }
                System.arraycopy(extraMisses, 0, knMisses[i], numMisses, kDiff);
            }
            for (int kIndex = 0; kIndex < k; kIndex++) {
                hitNeighbOccFreqs[knHits[i][kIndex]]++;
                missNeighbOccFreqs[knMisses[i][kIndex]]++;
                hitReverseNNSets[knHits[i][kIndex]].add(i);
                missReverseNNSets[knMisses[i][kIndex]].add(i);
            }
        }
    }

    /**
     * @return int[][] representing the k-hits.
     */
    public int[][] getKnHits() {
        return knHits;
    }

    /**
     * @param knHits int[][] representing the k-hits.
     */
    public void setKnHits(int[][] knHits) {
        this.knHits = knHits;
        if (knHits == null) {
            return;
        }
        int dSize = knHits.length;
        hitNeighbOccFreqs = new float[dSize];
        hitReverseNNSets = new List[dSize];
        for (int i = 0; i < dSize; i++) {
            hitReverseNNSets[i] = new ArrayList<>(k);
        }
        for (int i = 0; i < knHits.length; i++) {
            for (int kIndex = 0; kIndex < k; kIndex++) {
                hitNeighbOccFreqs[knHits[i][kIndex]]++;
                hitReverseNNSets[knHits[i][kIndex]].add(i);
            }
        }
    }

    /**
     * @return float[] representing the k-hit neighbor occurrence frequencies.
     */
    public float[] getHitNeighbOccFreqs() {
        return hitNeighbOccFreqs;
    }

    /**
     * @return List<Integer>[] representing the hit reverse kNN sets.
     */
    public List<Integer>[] getHitReverseNNSets() {
        return hitReverseNNSets;
    }

    /**
     * @return int[][] representing the k-misses.
     */
    public int[][] getKnMisses() {
        return knMisses;
    }

    /**
     * @param knMisses int[][] representing the k-misses.
     */
    public void setKnMisses(int[][] knMisses) {
        this.knMisses = knMisses;
        if (knMisses == null) {
            return;
        }
        int dSize = knMisses.length;
        missNeighbOccFreqs = new float[dSize];
        missReverseNNSets = new List[dSize];
        for (int i = 0; i < dSize; i++) {
            missReverseNNSets[i] = new ArrayList<>(k);
        }
        for (int i = 0; i < knMisses.length; i++) {
            for (int kIndex = 0; kIndex < k; kIndex++) {
                missNeighbOccFreqs[knMisses[i][kIndex]]++;
                missReverseNNSets[knMisses[i][kIndex]].add(i);
            }
        }
    }

    /**
     * @return float[] representing the k-hit neighbor occurrence frequencies.
     */
    public float[] getMissNeighbOccFreqs() {
        return missNeighbOccFreqs;
    }

    /**
     * @return List<Integer>[] representing the miss reverse kNN sets.
     */
    public List<Integer>[] getMissReverseNNSets() {
        return missReverseNNSets;
    }
    
    /**
     * This method computes the hit-miss score that is based on the K-divergence
     * measure of the hit and miss neighbor occurrence frequencies.
     * 
     * @param hitOccFreq Float value representing the occurrence frequency 
     * among the hit lists.
     * @param missOccFreq Float value representing the occurrence frequency 
     * among the miss lists.
     * @return Double value that is the HM-score.
     */
    public static double computeHMScore(float hitOccFreq, float missOccFreq) {
        if (hitOccFreq < 0 || missOccFreq < 0 ||
                (hitOccFreq + missOccFreq) == 0) {
            // Return the default value, lower than the minimal correct value.
            return -1;
        }
        // Normalize the two frequencies so that they sum up to one.
        float pHit = hitOccFreq / (hitOccFreq + missOccFreq);
        float pMiss = missOccFreq / (hitOccFreq + missOccFreq);
        double hmScore = 0;
        if (pHit > 0) {
            hmScore += pHit * BasicMathUtil.log2(pHit);
        }
        if (pMiss > 0) {
            hmScore -= pMiss * BasicMathUtil.log2(pMiss);
        }
        return hmScore;
    }
    
    /**
     * This method computes all the HM scores for the points in the considered 
     * dataset.
     * 
     * @return double[] representing the HM scores for the considered data 
     * points. 
     */
    public double[] computeAllHMScores() {
       if (hitNeighbOccFreqs == null || missNeighbOccFreqs == null || 
               hitNeighbOccFreqs.length != missNeighbOccFreqs.length) {
           return null;
       }
       int dSize = hitNeighbOccFreqs.length;
       double[] hmScores = new double[dSize];
       for (int i = 0; i < dSize; i++) {
           hmScores[i] = computeHMScore(hitNeighbOccFreqs[i],
                   missNeighbOccFreqs[i]);
       }
       return hmScores;
    }
    
    /**
     * This small script extracts the hit-miss network for the specified data 
     * and the Euclidean metric and persists the hit and miss occurrence 
     * frequencies, as well as the HM scores for all data points.
     * @param args
     * @throws Exception 
     */
    public static void main(String[] args) throws Exception {
        CommandLineParser clp = new CommandLineParser(true);
        clp.addParam("-inFile", "Path to the input dataset.",
                CommandLineParser.STRING, true, false);
        clp.addParam("-outFile", "Output path.", CommandLineParser.STRING,
                true, false);
        clp.addParam("-k", "Neighborhood size.",
                CommandLineParser.INTEGER, true, false);
        clp.parseLine(args);
        File inFile = new File((String) (clp.getParamValues("-inFile").get(0)));
        File outFile = new File((String) (clp.getParamValues(
                "-outFile").get(0)));
        int k = (Integer) (clp.getParamValues("-k").get(0));
        DataSet dset = SupervisedLoader.loadData(inFile.getPath(), false);
        float[][] dMat = dset.calculateDistMatrix(CombinedMetric.EUCLIDEAN);
        HitMissNetwork hmn = new HitMissNetwork(dset, dMat, k);
        hmn.generateNetwork();
        try (PrintWriter pw = new PrintWriter(new FileWriter(outFile))) {
            SOPLUtil.printArrayToStream(hmn.getHitNeighbOccFreqs(), pw, ",");
            SOPLUtil.printArrayToStream(hmn.getMissNeighbOccFreqs(), pw, ",");
            SOPLUtil.printArrayToStream(hmn.computeAllHMScores(), pw, ",");
        }
    }
    
}
