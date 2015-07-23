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
package learning.unsupervised.evaluation;

import data.representation.DataSet;
import data.representation.DataInstance;
import learning.unsupervised.ClusteringAlg;
import learning.unsupervised.evaluation.quality.QIndexDaviesBouldin;
import learning.unsupervised.evaluation.quality.QIndexDunn;
import learning.unsupervised.evaluation.quality.QIndexSilhouette;
import java.io.PrintWriter;
import java.io.FileWriter;
import learning.unsupervised.refinement.PantSAStar;
import distances.primary.CombinedMetric;
import learning.unsupervised.Cluster;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;
import feature.evaluation.Info;
import ioformat.SupervisedLoader;
import util.CommandLineParser;

/**
 * This class can be used for initial algorithm testing - for more detailed
 * testing and between-algorithm comparisons, BatchClusteringTester should be
 * used exclusively, since it implements more options, more functionality, etc.
 * Not everything is supported in this class.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class BasicClusteringEvaluator {

    private int numSec = 0;
    private javax.swing.Timer timeTimer;
    private DataSet dsetTest;
    private PrintWriter pw;
    private float[] dbScores;
    private float[] dunnScores;
    private float[] silScores;
    private float[] avgError;
    private float[] avgClusterEntropy;
    private float avgDB;
    private float avgDunn;
    private float avgSil;
    private float avgErr;
    private float avgTime;
    private float avgEntropy;
    private CombinedMetric cmet;

    /**
     * @param dsetTest DataSet object for testing.
     * @param pw PrintWriter for output.
     */
    public BasicClusteringEvaluator(DataSet dsetTest, PrintWriter pw) {
        this.dsetTest = dsetTest;
        this.pw = pw;
    }

    public BasicClusteringEvaluator() {
    }

    /**
     * Starts the timer.
     */
    public void startTimer() {
        timeTimer = new javax.swing.Timer(1000, timerListener);
        timeTimer.start();
    }
    ActionListener timerListener = new ActionListener() {
        @Override
        public void actionPerformed(ActionEvent e) {
            numSec++;
            try {
            } catch (Exception exc) {
            }
        }
    };

    /**
     * Stops the timer.
     */
    public void stopTimer() {
        timeTimer.stop();
        numSec = 0;
    }

    /**
     * Loads the test data from the specified path.
     *
     * @param path Path to load the data from.
     * @throws Exception
     */
    public void loadData(String path) throws Exception {
        dsetTest = SupervisedLoader.loadData(path, false);
    }

    /**
     * Sets the writer.
     *
     * @param path Output path.
     * @throws Exception
     */
    public void setWriter(String path) throws Exception {
        pw = new PrintWriter(new FileWriter(path), true);
    }

    /**
     * Perform clustering.
     *
     * @param algorithm Algorithm name.
     * @param numTimes Number of repetitions.
     * @param refinement Whether to use refinement.
     * @param numClusters Number of clusters to set, if applicable.
     * @throws Exception
     */
    public void clusterWithAlgorithm(String algorithm, int numTimes,
            boolean refinement, int numClusters) throws Exception {
        Class clustering = Class.forName(algorithm);
        Object clusterer = clustering.newInstance();
        dbScores = new float[numTimes];
        dunnScores = new float[numTimes];
        silScores = new float[numTimes];
        avgError = new float[numTimes];
        avgDB = 0;
        avgDunn = 0;
        avgSil = 0;
        avgErr = 0;
        pw.println("time, " + "DB" + ", " + "DUNN" + ", " + "SILHOUETTE" + ", "
                + "AVG_ERROR");
        for (int i = 0; i < numTimes; i++) {
            System.out.println(i + "th iteration");
            if (clusterer instanceof learning.unsupervised.ClusteringAlg) {

                if (cmet == null) {
                    cmet = CombinedMetric.FLOAT_EUCLIDEAN;
                }
                ((ClusteringAlg) clusterer).setNumClusters(numClusters);
                ((ClusteringAlg) clusterer).setCombinedMetric(cmet);
                ((ClusteringAlg) clusterer).setDataSet(dsetTest);
                startTimer();
                ((ClusteringAlg) clusterer).cluster();

                avgTime += numSec;
                pw.print(numSec + ", ");
                stopTimer();
                Cluster[] config;
                if (refinement) {
                    PantSAStar refiner = new PantSAStar(
                            numClusters,
                            ((ClusteringAlg) clusterer).
                            getClusterAssociations(),
                            dsetTest,
                            cmet);
                    refiner.calculateSilhouetteArray();
                    int[] newAssociations = refiner.refine();
                    config = Cluster.getConfigurationFromAssociations(
                            newAssociations, dsetTest);
                } else {
                    config = ((ClusteringAlg) clusterer).getClusters();
                }
                QIndexDaviesBouldin dbIndex =
                        new QIndexDaviesBouldin(config, dsetTest, cmet);
                QIndexDunn dunnIndex = new QIndexDunn(config, dsetTest, cmet);
                QIndexSilhouette silIndex =
                        new QIndexSilhouette(config, dsetTest, cmet);
                dbScores[i] = dbIndex.validity();
                dunnScores[i] = dunnIndex.validity();
                silScores[i] = silIndex.validity();
                avgDB += dbScores[i];
                avgDunn += dunnScores[i];
                avgSil += silScores[i];
                DataInstance[] centroids = new DataInstance[config.length];
                for (int j = 0; j < centroids.length; j++) {
                    if (config[j] != null) {
                        centroids[j] = config[j].getCentroid();
                        for (int p = 0; p < config[j].size(); p++) {
                            avgError[i] +=
                                    cmet.dist(centroids[j],
                                    config[j].getInstance(p));
                        }
                    }
                }
                avgError[i] /= dsetTest.size();
                avgErr += avgError[i];
                pw.println(dbScores[i] + ", " + dunnScores[i] + ", "
                        + silScores[i] + ", " + avgError[i]);
            }

        }
        avgDB /= numTimes;
        avgDunn /= numTimes;
        avgSil /= numTimes;
        avgErr /= numTimes;
        avgTime /= numTimes;
        pw.println(avgTime + ", " + avgDB + ", " + avgDunn + ", " + avgSil
                + ", " + avgErr);
        pw.close();
    }

    /**
     * Perform clustering.
     *
     * @param clusterer Clustering algorithm instance.
     * @param numTimes Number of repetitions.
     * @param refinement Whether to use refinement or not.
     * @param numClusters Number of clusters, if applicable.
     * @throws Exception
     */
    public void clusterWithAlgorithmOnLabeledData(ClusteringAlg clusterer,
            int numTimes, boolean refinement, int numClusters)
            throws Exception {
        dbScores = new float[numTimes];
        dunnScores = new float[numTimes];
        silScores = new float[numTimes];
        avgError = new float[numTimes];
        avgClusterEntropy = new float[numTimes];
        avgDB = 0;
        avgDunn = 0;
        avgSil = 0;
        avgErr = 0;
        avgEntropy = 0;
        pw.println("time, " + "DB" + ", " + "DUNN" + ", " + "SILHOUETTE"
                + ", " + "AVG_ERROR" + ", " + "AVG_CLUSTER_ENTROPY");
        for (int i = 0; i < numTimes; i++) {
            System.out.println(i + "th iteration");
            if (clusterer instanceof learning.unsupervised.ClusteringAlg) {

                boolean doneCorrectly = false;
                do {
                    if (cmet == null) {
                        cmet = CombinedMetric.FLOAT_EUCLIDEAN;
                    }
                    ((ClusteringAlg) clusterer).setNumClusters(numClusters);
                    ((ClusteringAlg) clusterer).setCombinedMetric(cmet);
                    ((ClusteringAlg) clusterer).setDataSet(dsetTest);
                    startTimer();
                    try {
                        ((ClusteringAlg) clusterer).cluster();
                        doneCorrectly = true;
                    } catch (Exception e) {
                        System.out.println("error: " + e.getMessage());
                        stopTimer();
                        numSec = 0;
                    }
                } while (!doneCorrectly);
                avgTime += numSec;
                pw.print(numSec + ", ");
                stopTimer();
                Cluster[] config;
                if (refinement) {
                    PantSAStar refiner = new PantSAStar(
                            numClusters,
                            ((ClusteringAlg) clusterer).
                            getClusterAssociations(),
                            dsetTest,
                            cmet);
                    refiner.calculateSilhouetteArray();
                    int[] newAssociations = refiner.refine();
                    config = Cluster.getConfigurationFromAssociations(
                            newAssociations, dsetTest);
                } else {
                    config = ((ClusteringAlg) clusterer).getClusters();
                }
                QIndexDaviesBouldin dbIndex = new QIndexDaviesBouldin(config,
                        dsetTest, cmet);
                QIndexDunn dunnIndex = new QIndexDunn(config, dsetTest, cmet);
                QIndexSilhouette silIndex = new QIndexSilhouette(config,
                        dsetTest, cmet);
                dbScores[i] = dbIndex.validity();
                dunnScores[i] = dunnIndex.validity();
                silScores[i] = silIndex.validity();
                avgDB += dbScores[i];
                avgDunn += dunnScores[i];
                avgSil += silScores[i];
                DataInstance[] centroids = new DataInstance[config.length];
                int numNonEmpty = 0;
                for (int j = 0; j < centroids.length; j++) {
                    if (config[j] != null && config[j].size() > 0) {
                        centroids[j] = config[j].getCentroid();
                        numNonEmpty++;
                        for (int p = 0; p < config[j].size(); p++) {
                            avgError[i] += cmet.dist(centroids[j],
                                    config[j].getInstance(p));
                        }
                    }
                }
                avgError[i] /= dsetTest.size();
                avgErr += avgError[i];
                int currIndex = -1;
                ArrayList<Integer>[] split = new ArrayList[numNonEmpty];
                for (int j = 0; j < split.length; j++) {
                    split[j] = new ArrayList(1500);
                }
                for (int j = 0; j < config.length; j++) {
                    if (config[j] != null && config[j].size() > 0) {
                        ++currIndex;
                        for (int k = 0; k < config[j].indexes.size(); k++) {
                            split[currIndex].add(
                                    (dsetTest.data.get(
                                    config[j].indexes.get(k))).getCategory());
                        }
                    }
                }
                avgClusterEntropy[i] = Info.evaluateInfoOfCategorySplit(
                        split, numClusters);
                avgEntropy += avgClusterEntropy[i];
                pw.println(dbScores[i] + ", " + dunnScores[i] + ", "
                        + silScores[i] + ", " + avgError[i] + ", "
                        + avgClusterEntropy[i]);
            }

        }
        avgDB /= numTimes;
        avgDunn /= numTimes;
        avgSil /= numTimes;
        avgErr /= numTimes;
        avgTime /= numTimes;
        avgEntropy /= numTimes;
        pw.println(avgTime + ", " + avgDB + ", " + avgDunn + ", " + avgSil
                + ", " + avgErr + ", " + avgEntropy);
        pw.close();
    }

    /**
     * Perform clustering.
     *
     * @param algorithm Algorithm name.
     * @param numTimes Number of repetitions.
     * @param refinement Whether to use refinement.
     * @param numClusters Number of clusters to set, if applicable.
     * @throws Exception
     */
    public void clusterWithAlgorithmOnLabeledData(String algorithm,
            int numTimes, boolean refinement, int numClusters)
            throws Exception {
        Class clustering = Class.forName(algorithm);
        Object clusterer = clustering.newInstance();
        clusterWithAlgorithmOnLabeledData((ClusteringAlg) clusterer,
                numTimes, refinement, numClusters);
    }

    /**
     * Executes the script.
     * @param args Command line parameters, as specified.
     * @throws Exception 
     */
    public static void main(String[] args) throws Exception {
        CommandLineParser clp = new CommandLineParser(true);
        clp.addParam("-inFile", "Path to the input file.",
                CommandLineParser.STRING, true, false);
        clp.addParam("-outFile", "Path to the output file.",
                CommandLineParser.STRING, true, false);
        clp.addParam("-algName", "Name of the algorithm to use.",
                CommandLineParser.STRING, true, false);
        clp.addParam("-numRepetitions", "Number of clustering runs.",
                CommandLineParser.INTEGER, true, false);
        clp.addParam("-numClusters", "Number of clusters.",
                CommandLineParser.INTEGER, true, false);
        clp.addParam("-noiseProb", "Probability of mutating a feature by noise",
                CommandLineParser.FLOAT, true, false);
        clp.addParam("-noiseStDev", "Standard deviation for Gaussian feature "
                + "noise", CommandLineParser.FLOAT, true, false);
        clp.addParam("-useRefinement", "Whether to use cluster refinement.",
                CommandLineParser.BOOLEAN, true, false);
        clp.parseLine(args);
        BasicClusteringEvaluator evaluator = new BasicClusteringEvaluator();
        evaluator.loadData((String) clp.getParamValues("-inFile").get(0));
        evaluator.setWriter((String) clp.getParamValues("-outFile").get(0));
        float pMutate = (Float) clp.getParamValues("-noiseProb").get(0);
        float stDev = (Float) clp.getParamValues("-noiseStDev").get(0);
        evaluator.dsetTest.addGaussianNoiseToNormalizedCollection(
                pMutate, stDev);
        try {
            evaluator.clusterWithAlgorithm(
                    (String) clp.getParamValues("-algName").get(0),
                    (Integer) clp.getParamValues("-numRepetitions").get(0),
                    (Boolean) clp.getParamValues("-useRefinement").get(0),
                    (Integer) clp.getParamValues("-numClusters").get(0));
        } catch (Exception e) {
            throw e;
        } finally {
            if (evaluator.pw != null) {
                evaluator.pw.close();
            }
        }
    }
}
