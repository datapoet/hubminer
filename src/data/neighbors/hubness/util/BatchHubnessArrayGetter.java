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
import ioformat.DistanceMatrixIO;
import ioformat.FileUtil;
import ioformat.IOARFF;
import ioformat.IOCSV;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import util.CommandLineParser;
import util.SOPLUtil;

/**
 * This script calculates and extracts the good and bad neighbor occurrence
 * frequencies from a directory of files and persists them to a target output
 * directory.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class BatchHubnessArrayGetter {

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
        clp.addParam("-inSelectionFile", "File containing a list of datasets to"
                + "process from the input directory. The specification should"
                + "have a comma-separated list in a single line, multiple lines"
                + "will be ignored.",
                CommandLineParser.STRING, true, false);
        clp.addParam("-distDir", "Path to the input distances directory. The"
                + "matrices should be in the corresponding data directories and"
                + "metric subdirectories.",
                CommandLineParser.STRING, true, false);
        clp.addParam("-outDir", "Path to the output directory.",
                CommandLineParser.STRING, true, false);
        clp.addParam("-k", "Neighborhood size.",
                CommandLineParser.INTEGER, true, false);
        clp.addParam("-metric", "String that is the desired metric.",
                CommandLineParser.STRING, true, false);
        clp.parseLine(args);
        File dataDir = new File((String) clp.getParamValues("-inDir").get(0));
        File outDir = new File((String) clp.getParamValues("-outDir").get(0));
        int k = (Integer) clp.getParamValues("-k").get(0);
        File distancesDir = new File((String) clp.getParamValues("-distDir").
                get(0));
        // Set the chosen metric.
        CombinedMetric cmet = new CombinedMetric();
        Class currFloatMet = Class.forName((String) clp.getParamValues(
                "-metric").get(0));
        cmet.setFloatMetric((DistanceMeasure) (currFloatMet.newInstance()));
        cmet.setCombinationMethod(CombinedMetric.DEFAULT);
        // Set the matrix sub-path.
        String dMatPathEnding = "NO" + File.separator
                + cmet.getFloatMetric().getClass().getName()
                + File.separator + "dMat.txt";
        // Read a list of datasets to process.
        File inSelectionFile = new File((String) clp.getParamValues(
                "-inSelectionFile").get(0));
        String dsSpec = null;
        try (BufferedReader br = new BufferedReader(new InputStreamReader(
                new FileInputStream(inSelectionFile)));) {
            dsSpec = br.readLine();
        } catch (Exception e) {
            throw e;
        }
        String[] dsNames = dsSpec.split(",");
        // For each dataset, calculate the kNN sets and extract the neighbor
        // occurrence frequencies.
        for (String dsName : dsNames) {
            dsName = dsName.trim();
            File dsFile = new File(dataDir, dsName);
            DataSet dset;
            if (dsFile.getPath().endsWith(".csv")) {
                IOCSV reader = new IOCSV(true, ",");
                dset = reader.readData(dsFile);
            } else if (dsFile.getPath().endsWith(".arff")) {
                IOARFF persister = new IOARFF();
                dset = persister.load(dsFile.getPath());
            } else {
                System.out.println("error, could not read: "
                        + dsFile.getPath());
                continue;
            }
            File dMatFile = new File(distancesDir, dsFile.getName().substring(
                    0, dsFile.getName().lastIndexOf(".")) + File.separator
                    + dMatPathEnding);
            if (!dMatFile.exists()) {
                throw new Exception("File" + dMatFile.getPath()
                        + "  does not exist");
            }
            float[][] dmat = DistanceMatrixIO.loadDMatFromFile(dMatFile);
            NeighborSetFinder nsf = new NeighborSetFinder(dset, dmat, cmet);
            nsf.calculateNeighborSets(k);
            File outHubnessFile = new File(outDir, dsFile.getName().substring(
                    0, dsFile.getName().lastIndexOf(".")) + "Hubness.txt");
            FileUtil.createFile(outHubnessFile);
            // Write the neighbor occurrence frequencies to a file.
            try (PrintWriter pw = new PrintWriter(
                    new FileWriter(outHubnessFile));) {
                SOPLUtil.printArrayToStream(nsf.getNeighborFrequencies(), pw);
                SOPLUtil.printArrayToStream(nsf.getGoodFrequencies(), pw);
                SOPLUtil.printArrayToStream(nsf.getBadFrequencies(), pw);
            } catch (Exception e) {
                throw e;
            }
        }
    }
}