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
package images.mining.clustering;

import data.representation.DataInstance;
import data.representation.DataSet;
import ioformat.FileUtil;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import optimization.stochastic.algorithms.SimulatedThermicAnnealingLeap;
import optimization.stochastic.fitness.SIFTSegmentationHomogeneity;
import optimization.stochastic.operators.onFloats.HomogenousTwoDevsFloatMutator;
import util.CommandLineParser;
import util.ImageUtil;

/**
 * This utility script runs the experiments for optimizing the detection of
 * intra-image SIFT feature clusters by means of modified k-means clustering
 * that uses similarity ranking and combines different aspects of similarity -
 * spatial proximity, descriptor similarity, scale similarity and angle
 * similarity for the individual image features. The optimization is performed
 * stochastically via simulated annealing. This code corresponds to the
 * experiments that were published in the paper "Two pass k-means algorithm for
 * finding SIFT clusters in an image" in 2010 at the Slovenian KDD conference
 * which is a part of the larger Information Society multiconference.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class OptimizeSIFTClustering {

    private File inImagesDir;
    private File inSIFTDir;
    private File inSegmentDir;
    private String logPath;
    private String[] nameList;
    private int minClusters, maxClusters;
    private int numIter;
    private boolean useRank;

    /**
     * This method runs the optimization procedure for SIFT clustering.
     *
     * @throws Exception
     */
    public void runOptimization() throws Exception {
        nameList = ImageUtil.getImageNamesArray(inImagesDir, "jpg");
        // Homogeneity of SIFT features from different clusters in segments of
        // the segmented image is used as an indicator of the fitness of the
        // current solution.
        SIFTSegmentationHomogeneity fitness =
                SIFTSegmentationHomogeneity.newInstance(
                inImagesDir, inSIFTDir, inSegmentDir, nameList, minClusters,
                maxClusters, useRank);
        DataSet dset = new DataSet();
        dset.fAttrNames = new String[3];
        dset.fAttrNames[0] = "f1";
        dset.fAttrNames[1] = "f2";
        dset.fAttrNames[2] = "f3";
        DataInstance instance = new DataInstance(dset);
        instance.embedInDataset(dset);
        instance.fAttr[0] = 0.0f;
        instance.fAttr[1] = 0.05f;
        instance.fAttr[2] = 1f;
        float[] lowerBounds = new float[3];
        float[] upperBounds = new float[3];
        lowerBounds[0] = 0.00f;
        lowerBounds[1] = 0.00f;
        lowerBounds[2] = 0.00f;
        upperBounds[0] = 1f;
        upperBounds[1] = 1f;
        upperBounds[2] = 1f;
        FileUtil.createFileFromPath(logPath);
        // Set up the logging.
        PrintWriter logWriter = new PrintWriter(
                new FileWriter(logPath, true), true);
        // Initialize the mutation operator and the optimization framework.
        HomogenousTwoDevsFloatMutator mutator =
                new HomogenousTwoDevsFloatMutator(0.1f, 0.35f, 0.65f,
                lowerBounds, upperBounds);
        SimulatedThermicAnnealingLeap optimizer =
                new SimulatedThermicAnnealingLeap(
                instance, mutator, fitness, numIter);
        optimizer.setLogger(logWriter);
        optimizer.setLogging(true);
        try {
            optimizer.optimize();
            instance = (DataInstance) optimizer.getBestInstance();
            logWriter.println();
            logWriter.println("Best solution: " + instance.fAttr[0] + " "
                    + instance.fAttr[1] + " " + instance.fAttr[2]);
            logWriter.println("Best score: " + optimizer.getBestFitness());
            logWriter.println("Average score: "
                    + optimizer.getAverageFitness());
            logWriter.println("Evolution of best scores: ");
            float[] scores = optimizer.getBestScores();
            for (int i = 0; i < scores.length - 1; i++) {
                logWriter.print(scores[i] + ", ");
            }
            logWriter.println(scores[scores.length - 1]);
        } catch (Exception e) {
            System.err.println(e.getMessage());
        } finally {
            logWriter.close();
        }
    }

    /**
     * This script runs the optimization of SIFT feature clustering on images.
     *
     * @param args Command line arguments.
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        CommandLineParser clp = new CommandLineParser(true);
        clp.addParam("-imagesInput", "Path to the input folder containing the "
                + "image data.", CommandLineParser.STRING, true, false);
        clp.addParam("-siftInput", "Path to the input folder containing the "
                + "sift keyfiles.", CommandLineParser.STRING, true, false);
        clp.addParam("-segmentInput", "Path to the input folder containing the "
                + "segmented image data.", CommandLineParser.STRING, true,
                false);
        clp.addParam("-logOutput", "Path to the output log file.",
                CommandLineParser.STRING, true, false);
        clp.addParam("-numIter", "Integer that is the number of iterations.",
                CommandLineParser.INTEGER, true, false);
        clp.addParam("-minClust", "Integer that is the minimal number of "
                + "clusters.", CommandLineParser.INTEGER, true, false);
        clp.addParam("-maxClust", "Integer that is the maximal number of "
                + "clusters.", CommandLineParser.INTEGER, true, false);
        clp.addParam("-useRank", "Boolean that signifies whether to use ranks.",
                CommandLineParser.INTEGER, true, false);
        clp.parseLine(args);
        OptimizeSIFTClustering worker = new OptimizeSIFTClustering();
        worker.inImagesDir =
                new File((String) clp.getParamValues("-imagesInput").get(0));
        worker.inSIFTDir =
                new File((String) clp.getParamValues("-siftInput").get(0));
        worker.inSegmentDir = new File(
                (String) clp.getParamValues("-segmentInput").get(0));
        worker.logPath = (String) clp.getParamValues("-logOutput").get(0);
        worker.minClusters = (Integer) clp.getParamValues("-minClust").get(0);
        worker.maxClusters = (Integer) clp.getParamValues("-maxClust").get(0);
        worker.numIter = (Integer) clp.getParamValues("-numIter").get(0);
        worker.useRank = (Boolean) clp.getParamValues("-useRank").get(0);
        worker.runOptimization();
    }
}
