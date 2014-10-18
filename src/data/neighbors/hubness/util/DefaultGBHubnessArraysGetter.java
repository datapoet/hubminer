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
import data.representation.sparse.BOWDataSet;
import distances.primary.CombinedMetric;
import distances.primary.MinkowskiMetric;
import distances.secondary.snd.SharedNeighborCalculator;
import distances.sparse.SparseCombinedMetric;
import distances.sparse.SparseCosineMetric;
import filters.TFIDF;
import ioformat.FileUtil;
import ioformat.IOARFF;
import ioformat.IOCSV;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import util.CommandLineParser;

/**
 * This utility script quickly extracts the good and bad neighbor occurrence
 * frequency arrays for a directory of datasets in the default metric and for
 * the specified neighborhood size. As for optional parameters, there is TFIDF
 * and the option of calculating the secondary shared neighbor distances, namely
 * simcos or simhub. The script doesn't run through the file system recursively,
 * it processes only the datasets on the first level.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class DefaultGBHubnessArraysGetter {

    private static int secondaryDistanceOption;
    private static int kSND = 50;
    private static boolean tfidfMode = true;

    /**
     * This script calculates and extracts the good and bad neighbor occurrence
     * frequencies from a directory of files and persists them to a target
     * output directory.
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
        getGBHubnessesFromDir(inDir, outDir, k);
    }

    /**
     * This method processes a directory of data files, extracts the good and
     * bad neighbor occurrence frequencies in the specified context and writes
     * the output to the target output directory.
     *
     * @param inDir Input directory.
     * @param outDir Output directory.
     * @param k Integer that is the neighborhood size.
     * @throws Exception
     */
    public static void getGBHubnessesFromDir(File inDir, File outDir, int k)
            throws Exception {
        File[] children = inDir.listFiles();
        File tmpOut;
        for (File child : children) {
            if (child.isFile()) {
                tmpOut = new File(outDir, child.getName().substring(
                        0, child.getName().lastIndexOf(".")) + "GBH.csv");
                getGBHubnessFromFile(child, tmpOut, k);
            }
        }
    }

    /**
     * Calculate and print the good and bad neighbor occurrence frequency arrays
     * for the specified context for the specified dataset.
     *
     * @param inFile Input file.
     * @param outFile Output file.
     * @param kInteger that is the neighborhood size.
     * @throws Exception
     */
    public static void getGBHubnessFromFile(File inFile, File outFile, int k)
            throws Exception {
        DataSet dset;
        File dsFile = inFile;
        String dsPath = dsFile.getPath();
        CombinedMetric cmet;
        // This code should automatically detect whether the data is in the
        // dense or sparse format and set the CombinedMetric object accordingly
        // as well.
        if (dsPath.endsWith(".csv")) {
            IOCSV reader = new IOCSV(true, ",");
            dset = reader.readData(dsFile);
            cmet = new CombinedMetric(null, new MinkowskiMetric(),
                    CombinedMetric.DEFAULT);
        } else if (dsPath.endsWith(".arff")) {
            try {
                IOARFF persister = new IOARFF();
                dset = persister.load(dsPath);
                cmet = new CombinedMetric(null, new MinkowskiMetric(),
                        CombinedMetric.DEFAULT);
            } catch (Exception e) {
                IOARFF persister = new IOARFF();
                dset = persister.loadSparse(dsPath);
                cmet = new SparseCombinedMetric(null, null,
                        new SparseCosineMetric(), CombinedMetric.DEFAULT);
            }
        } else {
            System.out.println("Error, could not read from: " + dsPath);
            return;
        }
        // Apply TF-IDF, if specified.
        if (tfidfMode) {
            if (dset instanceof BOWDataSet) {
                TFIDF.filterWords((BOWDataSet) dset);
            } else {
                TFIDF.filterFloats(dset);
            }
        }
        // Calculate the primary distance matrix.
        float[][] dMat = dset.calculateDistMatrixMultThr(cmet, 8);
        NeighborSetFinder nsf = new NeighborSetFinder(dset, dMat, cmet);
        // If in the secondary distance mode.
        if (secondaryDistanceOption != 0) {
            nsf.calculateNeighborSetsMultiThr(kSND, 8);
            boolean hubnessAware = (secondaryDistanceOption == 2);
            SharedNeighborFinder snf = new SharedNeighborFinder(nsf);
            if (hubnessAware) {
                snf.obtainWeightsFromHubnessInformation(0);
            }
            snf.countSharedNeighbors();
            // Get the similarity matrix.
            float[][] simMat = snf.getSharedNeighborCounts();
            // Transform similarities into distances.
            dMat = new float[simMat.length][];
            for (int i = 0; i < dMat.length; i++) {
                dMat[i] = new float[simMat[i].length];
                for (int j = 0; j < dMat[i].length; j++) {
                    dMat[i][j] = kSND - simMat[i][j];
                }
            }
            // Make the secondary calculator.
            SharedNeighborCalculator snc;
            if (hubnessAware) {
                snc = new SharedNeighborCalculator(snf,
                        SharedNeighborCalculator.WeightingType.
                        HUBNESS_INFORMATION);
            } else {
                snc = new SharedNeighborCalculator(snf,
                        SharedNeighborCalculator.WeightingType.NONE);
            }
            nsf = new NeighborSetFinder(dset, dMat, snc);
            nsf.calculateNeighborSetsMultiThr(k, 6);
        } else {
            nsf.calculateNeighborSetsMultiThr(k, 8);
        }
        int[] kFreq = nsf.getNeighborFrequencies();
        int[] kBadFreq = nsf.getBadFrequencies();
        int[] kGoodFreq = nsf.getGoodFrequencies();
        FileUtil.createFile(outFile);
        PrintWriter pw = new PrintWriter(new FileWriter(outFile));
        try {
            // Total, bad and good k-occurrence frequencies.
            pw.println("Nk,BNk,GNk");
            for (int i = 0; i < kFreq.length; i++) {
                pw.println(kFreq[i] + "," + kBadFreq[i] + "," + kGoodFreq[i]);
            }
        } catch (Exception e) {
            System.err.println(e.getMessage());
        } finally {
            pw.close();
        }
    }
}
