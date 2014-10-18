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
package data.neighbors.hubness.util;

import data.neighbors.NeighborSetFinder;
import data.representation.DataSet;
import distances.primary.CombinedMetric;
import distances.primary.DistanceMeasure;
import ioformat.SupervisedLoader;
import java.awt.Point;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import util.AuxSort;
import util.CommandLineParser;

/**
 * This utility script extracts the most frequent neighbor pairs in the given
 * dataset for the specified neighborhood size and the specified metric.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class FrequentNeighborPairsFinder {

    /**
     * Calculate and write a list of most frequent co-occurring neighbor pairs.
     *
     * @param inFilePath String that is the input file path.
     * @param outFile Output file.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Integer that is the neighborhood size.
     * @throws Exception
     */
    public static void outputTopNeighborPairs(String inFilePath, File outFile,
            CombinedMetric cmet, int k) throws Exception {
        // Load the data.
        DataSet dset = SupervisedLoader.loadData(inFilePath, false);
        int numClasses = dset.countCategories();
        // Calculate the distance matrix.
        float[][] dMat = dset.calculateDistMatrixMultThr(cmet, 4);
        // Find the kNN sets.
        NeighborSetFinder nsf = new NeighborSetFinder(dset, dMat, cmet);
        nsf.calculateNeighborSets(k);
        int[][] kneighbors = nsf.getKNeighbors();
        // Neighbor co-occuring pairs, total and within each class.
        ArrayList<Point> coOccurringPairs = new ArrayList<>(dset.size());
        HashMap<Long, Integer> coDependencyMaps = new HashMap<>(dset.size());
        HashMap<Long, Integer>[] classcoDependencyMaps =
                new HashMap[numClasses];
        for (int c = 0; c < numClasses; c++) {
            classcoDependencyMaps[c] = new HashMap<>(dset.size());
        }
        long concat;
        int minIndex;
        long maxIndex;
        int currClass;
        int currFreq;
        // Go through the data and calculate the co-occurrence frequencies.
        for (int i = 0; i < dset.size(); i++) {
            currClass = dset.getLabelOf(i);
            for (int kIndFirst = 0; kIndFirst < k; kIndFirst++) {
                for (int kIndSecond = kIndFirst + 1; kIndSecond < k;
                        kIndSecond++) {
                    // Get the appropriate pair encoding.
                    minIndex = Math.min(kneighbors[i][kIndFirst],
                            kneighbors[i][kIndSecond]);
                    maxIndex = Math.max(kneighbors[i][kIndFirst],
                            kneighbors[i][kIndSecond]);
                    concat = (maxIndex << 32) | (minIndex & 0XFFFFFFFFL);
                    if (!coDependencyMaps.containsKey(concat)) {
                        // Insert the first co-occurrence of a pair.
                        coDependencyMaps.put(concat, 1);
                        coOccurringPairs.add(new Point(minIndex,
                                (int) maxIndex));
                    } else {
                        // Increase the co-occurrence count.
                        currFreq = coDependencyMaps.get(concat);
                        coDependencyMaps.remove(concat);
                        coDependencyMaps.put(concat, currFreq + 1);
                    }
                    // Update the frequencies in the neighborhoods of the
                    // current class.
                    if (!classcoDependencyMaps[currClass].containsKey(concat)) {
                        classcoDependencyMaps[currClass].put(concat, 1);
                    } else {
                        currFreq = classcoDependencyMaps[currClass].get(concat);
                        classcoDependencyMaps[currClass].remove(concat);
                        classcoDependencyMaps[currClass].put(concat, currFreq
                                + 1);
                    }
                }
            }
        }
        // Put all the neighbor pair co-occurrence counts into an array.
        int[] pairOccurrences = new int[coOccurringPairs.size()];
        for (int i = 0; i < coOccurringPairs.size(); i++) {
            minIndex = (int) (coOccurringPairs.get(i).getX());
            maxIndex = (int) (coOccurringPairs.get(i).getY());
            concat = (maxIndex << 32) | (minIndex & 0XFFFFFFFFL);
            pairOccurrences[i] = coDependencyMaps.get(concat);
        }
        // Sort in the descending order.
        int[] sortedPermutation = AuxSort.sortIndexedValue(pairOccurrences,
                true);
        // Print out the results to a file.
        try (PrintWriter pw = new PrintWriter(new FileWriter(outFile));) {
            for (int i = 0; i < sortedPermutation.length; i++) {
                minIndex = (int) (coOccurringPairs.get(
                        sortedPermutation[i]).getX());
                maxIndex = (int) (coOccurringPairs.get(
                        sortedPermutation[i]).getY());
                concat = (maxIndex << 32) | (minIndex & 0XFFFFFFFFL);
                pw.print("(");
                pw.print(minIndex);
                pw.print(",");
                pw.print(maxIndex);
                pw.print(")");
                pw.print(" labels: (");
                pw.print(dset.getLabelOf(minIndex));
                pw.print(",");
                pw.print(dset.getLabelOf((int) maxIndex));
                pw.print(")");
                pw.print(" freq: ");
                pw.print(pairOccurrences[i]);
                pw.print(" profile: ");
                pw.print("(");
                for (int c = 0; c < numClasses; c++) {
                    if (!classcoDependencyMaps[c].containsKey(concat)) {
                        pw.print(" 0 ");
                    } else {
                        pw.print(" ");
                        pw.print(classcoDependencyMaps[c].get(concat));
                        pw.print(" ");
                    }
                }
                pw.print(")");
                pw.println();
            }
        } catch (Exception e) {
            throw e;
        }
    }

    /**
     * This utility script extracts the most frequent neighbor pairs in the
     * given dataset for the specified neighborhood size and the specified
     * metric.
     *
     * @param args Command line parameters, as specified.
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        CommandLineParser clp = new CommandLineParser(true);
        clp.addParam("-inFile", "Input data file.",
                CommandLineParser.STRING, true, false);
        clp.addParam("-outFile", "Output data file.",
                CommandLineParser.STRING, true, false);
        clp.addParam("-k", "Neighborhood size.",
                CommandLineParser.INTEGER, true, false);
        clp.addParam("-metric", "String that is the desired metric.",
                CommandLineParser.STRING, true, false);
        clp.parseLine(args);
        File inFile = new File((String) clp.getParamValues("-inFile").get(0));
        File outFile = new File((String) clp.getParamValues("-outFile").get(0));
        int k = (Integer) clp.getParamValues("-k").get(0);
        // Set the chosen metric.
        CombinedMetric cmet = new CombinedMetric();
        Class currFloatMet = Class.forName((String) clp.getParamValues(
                "-metric").get(0));
        cmet.setFloatMetric((DistanceMeasure) (currFloatMet.newInstance()));
        cmet.setCombinationMethod(CombinedMetric.DEFAULT);
        outputTopNeighborPairs(inFile.getPath(), outFile, cmet, k);
    }
}
