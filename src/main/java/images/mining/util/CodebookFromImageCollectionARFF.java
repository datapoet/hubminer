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
package images.mining.util;

import data.representation.DataInstance;
import data.representation.DataSet;
import distances.primary.CombinedMetric;
import ioformat.FileUtil;
import ioformat.IOARFF;

import ioformat.IOCSV;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import learning.unsupervised.Cluster;
import learning.unsupervised.methods.FastKMeans;
import sampling.UniformSampler;
import util.CommandLineParser;

/**
 * This class is a utility class for calculating a codebook of image visual
 * words for an ARFF file containing all the image feature descriptors. Multiple
 * descriptors can be given per line and will be cut into pieces corresponding
 * to the provided descriptor length.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class CodebookFromImageCollectionARFF {

    /**
     * This runs the script for codebook calculations.
     *
     * @param args Command line parameters.
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        // Set the specification for the command line parameters.
        CommandLineParser clp = new CommandLineParser(true);
        clp.addParam("-inputARFF", "Path to the input arff file containing the "
                + "descriptors to calculate the codebook from.",
                CommandLineParser.STRING, true, false);
        clp.addParam("-outCodebookFile", "Path to where the codebook is to be "
                + "persisted.", CommandLineParser.STRING,
                true, false);
        clp.addParam("-descLength", "Integer that is the descriptor length.",
                CommandLineParser.INTEGER, true, false);
        clp.addParam("-codebookSize", "Integer that is the codebook size.",
                CommandLineParser.INTEGER, true, false);
        clp.addParam("-sampleSize", "Size of the sample to use for calculating"
                + "the codebook.", CommandLineParser.FLOAT, true, false);
        // Parse the provided parameters.
        clp.parseLine(args);
        // Initialize the files and parameters.
        String inFilePath = (String) clp.getParamValues("-inputARFF").get(0);
        File inputFile = new File(inFilePath);
        File outFile = new File((String) clp.getParamValues("-outCodebookFile").
                get(0));
        int descLength = (Integer) clp.getParamValues("-descLength").get(0);
        int codebookSize = (Integer) clp.getParamValues("-codebookSize").get(0);
        int sampleSize = (Integer) clp.getParamValues("-sampleSize").get(0);
        FileUtil.createFile(outFile);
        // Load the data to calculate the codebook from.
        DataSet dset = null;
        if (inFilePath.endsWith(".csv")) {
            IOCSV reader = new IOCSV(true, ",");
            dset = reader.readData(inputFile);
        } else if (inFilePath.endsWith(".arff")) {
            IOARFF persister = new IOARFF();
            dset = persister.load(inFilePath);
        }
        // Basic assertions.
        if (dset == null) {
            System.err.println("Data load failed.");
            return;
        }
        if (dset.getNumFloatAttr() % descLength != 0) {
            System.err.println(dset.getNumFloatAttr()
                    + "not divisible by" + descLength);
            return;
        }
        // Create a DataSet objects to hold the processed features.
        DataSet features = new DataSet();
        features.fAttrNames = new String[descLength];
        for (int i = 0; i < descLength; i++) {
            features.fAttrNames[i] = "f" + i;
        }
        // Get the multiple of decriptor occurrences per line if multiple
        // descriptors are given per line. This is allowed due to some backward
        // compatibility issues. One can simply provide one descriptor per line
        // and avoid the complications.
        int times = dset.getNumFloatAttr() / descLength;
        features.data = new ArrayList<>(times * dset.size());
        DataInstance tempInstance;
        for (int i = 0; i < dset.size(); i++) {
            for (int j = 0; j < times; j++) {
                tempInstance = new DataInstance(features);
                for (int k = 0; k < descLength; k++) {
                    tempInstance.fAttr[k] =
                            dset.data.get(i).fAttr[j * descLength + k];
                }
                features.addDataInstance(tempInstance);
                tempInstance.embedInDataset(features);
            }
        }
        // Take a sample prior to clustering.
        UniformSampler sampler = new UniformSampler(false);
        DataSet clusteringSample = sampler.getSample(features, sampleSize);
        // Initialize the metric.
        CombinedMetric cmet = CombinedMetric.FLOAT_MANHATTAN;
        // Cluster the feature sample.
        FastKMeans clusterer = new FastKMeans(clusteringSample,
                cmet, codebookSize);
        clusterer.cluster();
        Cluster[] clusters = clusterer.getClusters();
        // Obtain the codebook vectors.
        DataInstance[] cbVectors = new DataInstance[codebookSize];
        for (int i = 0; i < codebookSize; i++) {
            cbVectors[i] = clusters[i].getMedoid();
        }
        // Print the codebook to a file.
        try (PrintWriter pw = new PrintWriter(new FileWriter(outFile));) {
            pw.println("codebook_size:" + codebookSize);
            for (int i = 0; i < codebookSize; i++) {
                pw.print(cbVectors[i].fAttr[0]);
                for (int j = 1; j < descLength; j++) {
                    pw.print("," + cbVectors[i].fAttr[j]);
                }
                pw.println();
            }
        } catch (Exception e) {
            System.err.println(e.getMessage());
            throw e;
        }
    }
}