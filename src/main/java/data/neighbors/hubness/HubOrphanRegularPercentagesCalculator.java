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
package data.neighbors.hubness;

import data.neighbors.NeighborSetFinder;
import data.representation.DataInstance;
import data.representation.DataSet;
import ioformat.DistanceMatrixIO;
import java.io.File;
import java.util.ArrayList;
import statistics.HigherMoments;
import util.CommandLineParser;

/**
 * This class implements the methods for calculating the percentage of hub
 * points, regular points and orphans in the data, for a range of neighborhood
 * sizes.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class HubOrphanRegularPercentagesCalculator {

    // Object that holds the calculated kNN sets.
    private NeighborSetFinder nsf = null;
    // The upper limit on the neighborhood size to fetch the percentages for.
    private int kMax;
    // Percentages of hubs, orphans and regular points.
    private float[] hubPercs;
    private float[] orphanPercs;
    private float[] regularPercs;

    /**
     * Initialization.
     *
     * @param nsf NeighborSetFinder object for kNN calculations, with
     * pre-computed kNN sets.
     * @param kMax Integer that is the upper limit on the neighborhood size to
     * fetch the percentages for.
     */
    public HubOrphanRegularPercentagesCalculator(NeighborSetFinder nsf,
            int kMax) {
        this.nsf = nsf;
        this.kMax = kMax;
    }

    /**
     * @return float[] representing the array of hub point percentages among all
     * points for a range of neighborhood sizes.
     */
    public float[] getHubPercs() {
        return hubPercs;
    }

    /**
     * @return float[] representing the array of orphan percentages among all
     * points for a range of neighborhood sizes.
     */
    public float[] getOrphanPercs() {
        return orphanPercs;
    }

    /**
     * @return float[] representing the array of regular point percentages among
     * all points for a range of neighborhood sizes.
     */
    public float[] getRegularPercs() {
        return regularPercs;
    }

    /**
     * Calculate the percentages of different point types for the specified
     * range of neighborhood sizes.
     */
    public void calculatePTypePercs() {
        if (nsf == null) {
            return;
        }
        // Initialize the percentage result arrays.
        hubPercs = new float[kMax];
        orphanPercs = new float[kMax];
        regularPercs = new float[kMax];
        // Go through the specified k-range.
        for (int kIndex = 0; kIndex < kMax; kIndex++) {
            // Re-calculate the neighbor occurrence frequencies.
            nsf.recalculateStatsForSmallerK(kIndex + 1);
            // Fetch the neighbor occurrence frequencies.
            int[] neighbOccFreqs = nsf.getNeighborFrequencies();
            // Calculate the occurrence frequency standard deviation.
            float occFreqStDev = HigherMoments.calculateArrayStDev(kIndex + 1,
                    neighbOccFreqs);
            // Initialize the point type counts.
            float hubCount = 0, orphanCount = 0, regularCount = 0;
            for (int i = 0; i < neighbOccFreqs.length; i++) {
                if (neighbOccFreqs[i] >= kIndex + 1 + 2 * occFreqStDev) {
                    // Hub points.
                    hubCount++;
                } else if (neighbOccFreqs[i]
                        <= Math.max(0, kIndex + 1 - 2 * occFreqStDev)) {
                    // Orphan points.
                    orphanCount++;
                } else {
                    // Regular points.
                    regularCount++;
                }
            }
            // Normalize the percentages.
            hubPercs[kIndex] = hubCount / (float) neighbOccFreqs.length;
            orphanPercs[kIndex] = orphanCount / (float) neighbOccFreqs.length;
            regularPercs[kIndex] = regularCount / (float) neighbOccFreqs.length;
        }
    }
    
    /**
     * Script that loads a distance matrix and calculates the hub/orphan/regular
     * percentages for the data.
     * @param args Command line parameters, as specified.
     * @throws Exception 
     */
    public static void main(String[] args) throws Exception {
        CommandLineParser clp = new CommandLineParser(true);
        clp.addParam("-distFile", "Path to the distance matrix file.",
                CommandLineParser.STRING, true, false);
        clp.addParam("-maxK", "Maximal neighborhood size to consider.",
                CommandLineParser.INTEGER, true, false);
        clp.parseLine(args);
        File inFile = new File((String) clp.getParamValues("-distFile").get(0));
        int maxK = (Integer) clp.getParamValues("-maxK").get(0);
        float[][] dMat = DistanceMatrixIO.loadDMatFromFile(inFile);
        DataSet dummyDSet = new DataSet();
        String[] attNames = new String[1];
        attNames[0] = "dummyAtt";
        dummyDSet.fAttrNames = attNames;
        dummyDSet.data = new ArrayList<>(dMat.length);
        for (int i = 0; i < dMat.length; i++) {
            DataInstance instance = new DataInstance(dummyDSet);
            instance.embedInDataset(dummyDSet);
            dummyDSet.addDataInstance(instance);
        }
        NeighborSetFinder nsf = new NeighborSetFinder(dummyDSet, dMat);
        nsf.calculateNeighborSets(maxK);
        HubOrphanRegularPercentagesCalculator calculator =
                new HubOrphanRegularPercentagesCalculator(nsf, maxK);
        calculator.calculatePTypePercs();
        float[] hPercs = calculator.getHubPercs();
        float[] rPercs = calculator.getRegularPercs();
        float[] oPercs = calculator.getOrphanPercs();
        for (int k = 1; k <= maxK; k++) {
            System.out.print("Hubs/Regulars/Orphans for k = " + k + ": ");
            System.out.print(((int)(hPercs[k - 1] * dMat.length)) + ",");
            System.out.print(((int)(rPercs[k - 1] * dMat.length)) + ",");
            System.out.print(((int)(oPercs[k - 1] * dMat.length)));
            System.out.println();
        }
    }
}
