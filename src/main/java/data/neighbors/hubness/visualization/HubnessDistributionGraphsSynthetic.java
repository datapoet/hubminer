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
package data.neighbors.hubness.visualization;

import data.generators.MultiDimensionalSphericGaussianGenerator;
import data.neighbors.NeighborSetFinder;
import data.representation.DataInstance;
import data.representation.DataSet;
import distances.primary.CombinedMetric;
import distances.primary.MinkowskiMetric;
import ioformat.FileUtil;
import java.awt.Point;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Random;
import util.CommandLineParser;

/**
 * This utility class can be used to generate the data for some neighbor
 * occurrence and co-occurrence distribution charts on synthetic Gaussian
 * mixture data of variable dimensionality.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class HubnessDistributionGraphsSynthetic {

    private static int k = 10;
    private static final int REPETITIONS = 200;
    private static final int DATA_SIZE = 1000;
    private int[] hubnessArray;
    private int[] numCoocPointsArray;
    private int[] pairOccurrences;

    /**
     * This method generates synthetic Gaussian data of the specified size and
     * dimensionality and then calculates the neighbor occurrence and
     * co-occurrence distributions.
     *
     * @param dsize Integer that is the desired data size.
     * @param dim Integer that is the desired dimensionality.
     * @throws Exception
     */
    private void generateResults(int dsize, int dim)
            throws Exception {
        float[] meanArray = new float[dim]; // Means are a zero-array.
        float[] stDevArray = new float[dim];
        Random randa = new Random();
        // Initialize standard deviations.
        for (int i = 0; i < dim; i++) {
            stDevArray[i] = randa.nextFloat();
        }
        // Set some upper and lower bounds.
        float[] lBounds = new float[dim];
        Arrays.fill(lBounds, -100);
        float[] uBounds = new float[dim];
        Arrays.fill(uBounds, 100);
        // Generate synthetic Gaussian data.
        MultiDimensionalSphericGaussianGenerator gen =
                new MultiDimensionalSphericGaussianGenerator(
                meanArray, stDevArray, lBounds, uBounds);
        DataSet genData = new DataSet();
        String[] floatAttrNames = new String[dim];
        for (int i = 0; i < dim; i++) {
            floatAttrNames[i] = "dim" + dim;
        }
        genData.fAttrNames = floatAttrNames;
        genData.data = new ArrayList<>(dsize);
        for (int i = 0; i < dsize; i++) {
            DataInstance instance = new DataInstance(genData);
            instance.fAttr = gen.generateFloat();
            genData.addDataInstance(instance);
        }
        // Find the kNN sets.
        CombinedMetric cmet = new CombinedMetric(
                null, new MinkowskiMetric(), CombinedMetric.DEFAULT);
        NeighborSetFinder nsf = new NeighborSetFinder(genData, cmet);
        nsf.calculateDistances();
        nsf.calculateNeighborSets(k);
        int[][] kneighbors = nsf.getKNeighbors();
        // Neighbor occurrence frequencies.
        hubnessArray = nsf.getNeighborFrequencies();
        // Number of points that a point co-occurs with.
        numCoocPointsArray = new int[hubnessArray.length];
        // A list of co-occuring pairs.
        ArrayList<Point> coOccurringPairs = new ArrayList<>(dsize);
        // HashMap that maps the pair frequency.
        HashMap<Long, Integer> coDependencyMaps = new HashMap<>(dsize);
        // Hashing by concatenatin the numbers.
        long concat;
        int min;
        long max;
        int currFreq;
        for (int i = 0; i < hubnessArray.length; i++) {
            for (int kInd1 = 0; kInd1 < k; kInd1++) {
                for (int kInd2 = kInd1 + 1; kInd2 < k; kInd2++) {
                    min = Math.min(kneighbors[i][kInd1], kneighbors[i][kInd2]);
                    max = Math.max(kneighbors[i][kInd1], kneighbors[i][kInd2]);
                    concat = (max << 32) | (min & 0XFFFFFFFFL);
                    if (!coDependencyMaps.containsKey(concat)) {
                        coDependencyMaps.put(concat, 1);
                        coOccurringPairs.add(new Point(min, (int) max));
                        numCoocPointsArray[kneighbors[i][kInd1]]++;
                        numCoocPointsArray[kneighbors[i][kInd2]]++;
                    } else {
                        currFreq = coDependencyMaps.get(concat);
                        coDependencyMaps.remove(concat);
                        coDependencyMaps.put(concat, currFreq + 1);
                    }
                }
            }
        }

        // Go through all the pairs and look up their frequencies.
        pairOccurrences = new int[coOccurringPairs.size()];

        for (int i = 0; i < coOccurringPairs.size(); i++) {
            min = (int) (coOccurringPairs.get(i).getX());
            max = (int) (coOccurringPairs.get(i).getY());
            concat = (max << 32) | (min & 0XFFFFFFFFL);
            pairOccurrences[i] = coDependencyMaps.get(concat);

        }
    }

    /**
     * This script generates the neighbor occurrence and co-occurrence frequency
     * chart data from synthetic Gaussian mixtures of specified data
     * dimensionality.
     *
     * @param args Command line parameters, as specified.
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        CommandLineParser clp = new CommandLineParser(true);
        clp.addParam("-k", "Neighborhood size.", CommandLineParser.INTEGER,
                true, false);
        clp.addParam("-dim", "Dimensionality.", CommandLineParser.INTEGER,
                true, false);
        clp.addParam("-outFile", "Path to the output directory.",
                CommandLineParser.STRING, true, false);
        clp.parseLine(args);
        File outFile = new File((String) clp.getParamValues("-outFile").get(0));
        k = (Integer) clp.getParamValues("-k").get(0);
        int dim = (Integer) clp.getParamValues("-dim").get(0);
        int graphLimit = 60;
        int[] hubnessArray;
        int[] numCoocPointsArray;
        int[] pairOccurrences;
        // Distribution arrays with an upper limit on the frequency.
        float[] distributionHubness = new float[graphLimit];
        float[] distributionNumCoHubs = new float[graphLimit];
        float[] distributionCoFreqs = new float[graphLimit];
        double numPairs = 0;
        for (int r = 0; r < REPETITIONS; r++) {
            // Generate the data and calculate the frequencies for each point
            // and pair.
            HubnessDistributionGraphsSynthetic dataGen =
                    new HubnessDistributionGraphsSynthetic();
            dataGen.generateResults(DATA_SIZE, dim);
            // Fetch the results.
            hubnessArray = dataGen.hubnessArray;
            numCoocPointsArray = dataGen.numCoocPointsArray;
            pairOccurrences = dataGen.pairOccurrences;
            numPairs += pairOccurrences.length;
            // Calculate the distribution.
            for (int i = 0; i < hubnessArray.length; i++) {
                if (hubnessArray[i] < graphLimit) {
                    distributionHubness[hubnessArray[i]]++;
                }
            }
            for (int i = 0; i < numCoocPointsArray.length; i++) {
                if (numCoocPointsArray[i] < graphLimit) {
                    distributionNumCoHubs[numCoocPointsArray[i]]++;
                }
            }
            for (int i = 0; i < pairOccurrences.length; i++) {
                if (pairOccurrences[i] < graphLimit) {
                    distributionCoFreqs[pairOccurrences[i]]++;
                }
            }
            distributionCoFreqs[0] = (DATA_SIZE * (DATA_SIZE - 1) / 2)
                    - pairOccurrences.length;
            System.out.print("|");
        }
        System.out.println();
        // Normalize
        for (int i = 0; i < graphLimit; i++) {
            distributionHubness[i] /= REPETITIONS;
            distributionNumCoHubs[i] /= REPETITIONS;
            distributionCoFreqs[i] /= REPETITIONS;
        }
        numPairs /= REPETITIONS;
        FileUtil.createFile(outFile);
        // Print out the results.
        PrintWriter pw = new PrintWriter(new FileWriter(outFile));
        try {
            pw.println("Hubnesses");
            pw.print(distributionHubness[0]);
            for (int i = 1; i < distributionHubness.length; i++) {
                pw.print("," + distributionHubness[i]);
            }
            pw.println();
            pw.println("Num Co-occ Neighbors");
            pw.print(distributionNumCoHubs[0]);
            for (int i = 1; i < distributionNumCoHubs.length; i++) {
                pw.print("," + distributionNumCoHubs[i]);
            }
            pw.println();
            pw.println("Num Pair Occurrences");
            pw.print(distributionCoFreqs[0]);
            for (int i = 1; i < distributionCoFreqs.length; i++) {
                pw.print("," + distributionCoFreqs[i]);
            }
            pw.println();
            pw.println("AVG num Co-occ Pairs");
            pw.println("d=" + dim + " " + numPairs);
        } catch (Exception e) {
            throw e;
        } finally {
            pw.close();
        }
    }
}
