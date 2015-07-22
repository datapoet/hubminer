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
import data.neighbors.SharedNeighborFinder;
import data.representation.DataSet;
import distances.primary.CombinedMetric;
import distances.primary.DistanceMeasure;
import distances.secondary.snd.SharedNeighborCalculator;
import ioformat.FileUtil;
import ioformat.SupervisedLoader;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import util.CommandLineParser;

/**
 * This utility script calculates and outputs the lists of reverse neighbors for
 * each point in the data, as well as a list of class-conditional occurrence
 * profiles in a separate file.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class ReverseNeighborGetter {

    // None, simcos, simhub.
    private static int secondaryDistanceOption;
    private static final int kSND = 50;
    private static CombinedMetric cmet; // Object for distance calculations.

    /**
     * This utility script calculates and outputs the lists of reverse neighbors
     * for each point in the data, as well as a list of class-conditional
     * occurrence profiles in a separate file.
     *
     * @param args Command line parameters, as specified.
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        CommandLineParser clp = new CommandLineParser(true);
        clp.addParam("-inDir", "Path to the input data directory.",
                CommandLineParser.STRING, true, false);
        clp.addParam("-outDir", "Path to the output directory.",
                CommandLineParser.STRING, true, false);
        clp.addParam("-k", "Neighborhood size.",
                CommandLineParser.INTEGER, true, false);
        clp.addParam("-secondaryDistance", "Whether to use secondary distances."
                + " Possible values: none, simcos, simhub",
                CommandLineParser.STRING, true, false);
        clp.addParam("-metric", "String that is the desired metric.",
                CommandLineParser.STRING, true, false);
        clp.parseLine(args);
        File inDir = new File((String) clp.getParamValues("-inDir").get(0));
        File outDir = new File((String) clp.getParamValues("-outDir").get(0));
        int k = (Integer) clp.getParamValues("-k").get(0);
        String secondaryString =
                (String) clp.getParamValues("-secondaryDistance").get(0);
        secondaryDistanceOption = 0;
        switch (secondaryString.toLowerCase()) {
            case "none":
                secondaryDistanceOption = 0;
                break;
            case "simcos":
                secondaryDistanceOption = 1;
                break;
            case "simhub":
                secondaryDistanceOption = 2;
                break;
        }
        cmet = new CombinedMetric();
        Class currFloatMet = Class.forName((String) clp.getParamValues(
                "-metric").get(0));
        cmet.setFloatMetric((DistanceMeasure) (currFloatMet.newInstance()));
        cmet.setCombinationMethod(CombinedMetric.DEFAULT);
        calculateAndWriteReverseNeighbors(inDir, outDir, k);
    }

    /**
     * This method calculates the reverse neighbors and the class-conditional
     * neighbor occurrence profiles for all datasets in the specified data
     * directory.
     *
     * @param inDir Directory of input datasets.
     * @param outDir Directory for the output.
     * @param k Integer that is the neighborhood size.
     * @throws Exception
     */
    private static void calculateAndWriteReverseNeighbors(File inDir,
            File outDir, int k) throws Exception {
        File[] children = inDir.listFiles();
        // Two separate output files, one for the class-conditional neighbor
        // occurrence frequencies, one for the reverse neighbor lists.
        File occFreqsOutFile;
        File revNeighbOutFile;
        for (File child : children) {
            if (child.isFile()) {
                occFreqsOutFile = new File(outDir, child.getName().substring(0,
                        child.getName().lastIndexOf("."))
                        + "classConditionalOccurrences.txt");
                revNeighbOutFile = new File(outDir, child.getName().substring(0,
                        child.getName().lastIndexOf("."))
                        + "reverseNeighbors.txt");
                calculateAndWriteReverseNeighbors(child, occFreqsOutFile,
                        revNeighbOutFile, k);
            }
        }
    }

    /**
     * This method calculates the reverse neighbors and the class-conditional
     * neighbor occurrence profiles for the specified dataset.
     *
     * @param inFile File that holds the data.
     * @param outClassFile File to write the class-conditional neighbor
     * occurrence profiles.
     * @param outRNeighborFile File to write the reverse neighbor lists to.
     * @param k Integer that is the neighborhood size.
     * @throws Exception
     */
    private static void calculateAndWriteReverseNeighbors(File inFile,
            File outClassFile, File outRNeighborFile, int k) throws Exception {
        // Load the data.
        DataSet dset = SupervisedLoader.loadData(inFile.getPath(), false);
        // Calculate the kNN sets.
        NeighborSetFinder nsf = new NeighborSetFinder(dset, cmet);
        nsf.calculateDistances();
        nsf.calculateNeighborSetsMultiThr(k, 6);
        if (secondaryDistanceOption != 0) {
            // Calculate the secondary distances, if needed.
            boolean hubnessAware = (secondaryDistanceOption == 2);
            SharedNeighborFinder snf = new SharedNeighborFinder(nsf);
            if (hubnessAware) {
                snf.obtainWeightsFromHubnessInformation(0);
            }
            snf.countSharedNeighbors();
            // First we get the similarity matrix and then we transform it to a
            // distance matrix.
            float[][] simMat = snf.getSharedNeighborCounts();
            float[][] dMat = new float[simMat.length][];
            for (int i = 0; i < dMat.length; i++) {
                dMat[i] = new float[simMat[i].length];
                for (int j = 0; j < dMat[i].length; j++) {
                    dMat[i][j] = kSND - simMat[i][j];
                }
            }
            // Initialize the shared neighbor calculator.
            SharedNeighborCalculator snc = hubnessAware
                    ? new SharedNeighborCalculator(snf,
                    SharedNeighborCalculator.WeightingType.HUBNESS_INFORMATION)
                    : new SharedNeighborCalculator(snf,
                    SharedNeighborCalculator.WeightingType.NONE);
            // Find the kNN sets.
            nsf = new NeighborSetFinder(dset, dMat, snc);
            nsf.calculateNeighborSetsMultiThr(k, 6);
        }
        int numClasses = dset.countCategories();
        // Get the class-conditional occurrence profiles.
        float[][] classConditionalOccDist =
                nsf.getFuzzyClassDataNeighborRelation(k, numClasses, 0.05f,
                false);
        FileUtil.createFile(outClassFile);
        // Print out the class-conditional neighbor occurrence profiles.
        PrintWriter pw = new PrintWriter(new FileWriter(outClassFile));
        try {
            pw.println(numClasses + " classes");
            for (int i = 0; i < dset.size(); i++) {
                pw.print(classConditionalOccDist[0][i]);
                for (int j = 1; j < numClasses; j++) {
                    pw.print("," + classConditionalOccDist[j][i]);
                }
                pw.println();
            }
        } catch (Exception e) {
            System.err.println(e.getMessage());
        } finally {
            pw.close();
        }
        // Print out the reverse neighbor lists.
        ArrayList<Integer>[] rNeighbors = nsf.getReverseNeighbors();
        FileUtil.createFile(outRNeighborFile);
        pw = new PrintWriter(new FileWriter(outRNeighborFile));
        try {
            for (int i = 0; i < dset.size(); i++) {
                if (rNeighbors[i] != null && rNeighbors[i].size() > 0) {
                    pw.print(rNeighbors[i].get(0));
                    for (int j = 1; j < rNeighbors[i].size(); j++) {
                        pw.print("," + rNeighbors[i].get(j));
                    }
                }
                pw.println();
            }

        } catch (Exception e) {
            System.err.println(e.getMessage());
        } finally {
            pw.close();
        }
    }
}