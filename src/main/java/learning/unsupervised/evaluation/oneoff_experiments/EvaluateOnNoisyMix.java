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
package learning.unsupervised.evaluation.oneoff_experiments;

import data.generators.NoisyGaussianMix;
import data.representation.DataInstance;
import data.representation.DataSet;
import distances.primary.CombinedMetric;
import feature.evaluation.Info;
import ioformat.FileUtil;
import ioformat.IOARFF;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import learning.unsupervised.Cluster;
import learning.unsupervised.evaluation.quality.QIndexSilhouette;
import learning.unsupervised.methods.GHPC;
import learning.unsupervised.methods.GHPKM;
import learning.unsupervised.methods.GKH;
import learning.unsupervised.methods.KMeansPlusPlus;

/**
 * This class tests how hubness-based clustering methods perform in synthetic
 * high-dimensional scenarios where uniform noise is slowly introduced to the
 * data in form of uniformly drawn instances around the Gaussian clusters. This
 * was one of the experiments presented in the PAKDD 2011 paper titled: "The
 * Role of Hubness in Clustering High-dimensional Data". This class is not very
 * flexible and it should be re-worked for future experiments. In its current
 * form, though - it can be used to run the same experiments as in the original
 * paper, for comparisons.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class EvaluateOnNoisyMix {

    private static final int DATA_SIZE = 10000;
    private static final int MAX_NOISY_INSTANCES = DATA_SIZE;
    private static final int STEPS = 40;
    private static final int NOISE_INCREMENT = MAX_NOISY_INSTANCES / STEPS;
    private int numSec = 0;
    private javax.swing.Timer timeTimer;
    private DataSet dsetTest;
    private PrintWriter pwKM;
    private PrintWriter pwPGKH;
    private PrintWriter pwGKH;
    private PrintWriter pwMin;
    private PrintWriter pwHPKM;
    private File writerDir;
    public int dim = 50;
    public int hubnessK = 50;
    private float[] silScores;
    private float[] avgError;
    private float[] avgClusterEntropy;
    private float avgSil;
    private float avgErr;
    private float avgTime;
    private float avgEntropy;
    private CombinedMetric cmet;

    public EvaluateOnNoisyMix(DataSet testDC) {
        this.dsetTest = testDC;
    }

    public EvaluateOnNoisyMix() {
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
        IOARFF persister = new IOARFF();
        dsetTest = persister.load(path);
    }

    /**
     * Sets the writer directory to the specified path.
     *
     * @param path File path to set the writer directory to.
     * @throws Exception
     */
    public void setWriterDir(String path) throws Exception {
        writerDir = new File(path);
    }

    /**
     * @param numTimes Repetitions.
     * @param numClusters Number of Gaussian clusters to generate.
     * @throws Exception
     */
    public void clusterWithAlgorithmOnLabeledData(
            int numTimes, int numClusters) throws Exception {
        NoisyGaussianMix genMix = new NoisyGaussianMix(numClusters, dim,
                DATA_SIZE, false, 0);
        dsetTest = genMix.generateRandomDataSet();
        System.out.println("Random data generated.");
        for (int numNoisy = 0; numNoisy <= MAX_NOISY_INSTANCES;
                numNoisy += NOISE_INCREMENT) {
            System.out.println("noise level: " + numNoisy);
            GHPC clusterer = new GHPC();
            KMeansPlusPlus clustererKM = new KMeansPlusPlus();
            GKH clustererGKH = new GKH();
            GHPKM clustererHPKM = new GHPKM();
            if (numNoisy > 0) {
                genMix.addNoiseToCollection(dsetTest, 500);
            }
            silScores = new float[numTimes];
            avgError = new float[numTimes];
            avgClusterEntropy = new float[numTimes];
            avgSil = 0;
            avgErr = 0;
            avgEntropy = 0;
            float[] silKMScores = new float[numTimes];
            float[] avgKMError = new float[numTimes];
            float[] avgKMClusterEntropy = new float[numTimes];
            float avgKMSil = 0;
            float avgKMErr = 0;
            float avgKMEntropy = 0;
            float[] silHPKMScores = new float[numTimes];
            float[] avgHPKMError = new float[numTimes];
            float[] avgHPKMClusterEntropy = new float[numTimes];
            float avgHPKMSil = 0;
            float avgHPKMErr = 0;
            float avgHPKMEntropy = 0;
            float[] silGKHScores = new float[numTimes];
            float[] avgGKHError = new float[numTimes];
            float[] avgGKHClusterEntropy = new float[numTimes];
            float avgGKHSil = 0;
            float avgGKHErr = 0;
            float avgGKHEntropy = 0;
            float[] silMinScores = new float[numTimes];
            float[] avgMinError = new float[numTimes];
            float[] avgMinClusterEntropy = new float[numTimes];
            float avgMinSil = 0;
            float avgMinErr = 0;
            float avgMinEntropy = 0;
            File currPGKHOutFile = new File(writerDir,
                    "PGKH_Noise_level" + numNoisy + "nClust"
                    + numClusters + ".csv");
            File currMinOutFile = new File(writerDir,
                    "PGKH_MIN_Noise_level" + numNoisy + "nClust"
                    + numClusters + ".csv");
            File currKMOutFile = new File(writerDir,
                    "KM_Noise_level" + numNoisy + "nClust" + numClusters
                    + ".csv");
            File currGKHOutFile = new File(writerDir,
                    "GKH_Noise_level" + numNoisy + "nClust" + numClusters
                    + ".csv");
            File currHPKMOutFile = new File(writerDir, ""
                    + "GHPKM_Noise_level" + numNoisy + "nClust" + numClusters
                    + ".csv");
            FileUtil.createFile(currPGKHOutFile);
            FileUtil.createFile(currKMOutFile);
            FileUtil.createFile(currGKHOutFile);
            FileUtil.createFile(currMinOutFile);
            FileUtil.createFile(currHPKMOutFile);
            pwPGKH = new PrintWriter(new FileWriter(currPGKHOutFile), true);
            pwHPKM = new PrintWriter(new FileWriter(currHPKMOutFile), true);
            pwKM = new PrintWriter(new FileWriter(currKMOutFile), true);
            pwGKH = new PrintWriter(new FileWriter(currGKHOutFile), true);
            pwMin = new PrintWriter(new FileWriter(currMinOutFile), true);
            pwPGKH.println("time, " + "SILHOUETTE" + ", " + "AVG_ERROR" + ", "
                    + "AVG_CLUSTER_ENTROPY");
            pwHPKM.println("time, " + "SILHOUETTE" + ", " + "AVG_ERROR" + ", "
                    + "AVG_CLUSTER_ENTROPY");
            pwKM.println("time, " + "SILHOUETTE" + ", " + "AVG_ERROR" + ", "
                    + "AVG_CLUSTER_ENTROPY");
            pwGKH.println("time, " + "SILHOUETTE" + ", " + "AVG_ERROR" + ", "
                    + "AVG_CLUSTER_ENTROPY");
            pwMin.println("time, " + "SILHOUETTE" + ", " + "AVG_ERROR" + ", "
                    + "AVG_CLUSTER_ENTROPY");
            for (int i = 0; i < numTimes; i++) {
                System.out.println("PGKH " + i + "th iteration");

                boolean doneCorrectly = false;
                do {
                    if (cmet == null) {
                        cmet = CombinedMetric.FLOAT_MANHATTAN;
                    }
                    clusterer.setNumClusters(numClusters);
                    clusterer.setCombinedMetric(cmet);
                    clusterer.setDataSet(dsetTest);
                    clusterer.setK(hubnessK);
                    clusterer.probabilisticIterations = 100;
                    avgTime = 0;
                    startTimer();
                    try {
                        clusterer.cluster();
                        doneCorrectly = true;
                    } catch (Exception e) {
                        System.out.println("error: " + e.getMessage());
                        stopTimer();
                        numSec = 0;
                    }
                } while (!doneCorrectly);
                avgTime += numSec;
                pwPGKH.print(numSec + ", ");
                stopTimer();
                Cluster[] config = clusterer.getClusters();
                Cluster[] configMin = clusterer.getMinimizingClusters();
                for (int l = 10000; l < dsetTest.size(); l++) {
                    dsetTest.data.get(l).setCategory(-1);
                }
                QIndexSilhouette silIndex = new QIndexSilhouette(
                        numClusters, clusterer.getClusterAssociations(),
                        dsetTest);
                silIndex.setDistanceMatrix(clusterer.getNSFDistances());
                silScores[i] = silIndex.validity();
                silIndex = new QIndexSilhouette(numClusters,
                        clusterer.getMinimizingAssociations(), dsetTest);
                silIndex.setDistanceMatrix(clusterer.getNSFDistances());
                silMinScores[i] = silIndex.validity();
                avgSil += silScores[i];
                avgMinSil += silMinScores[i];
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


                // And now for supervised estimates - the cluster entropies.
                // First make the split.
                int currIndex = -1;
                ArrayList<Integer>[] split = new ArrayList[numNonEmpty];
                for (int j = 0; j < split.length; j++) {
                    split[j] = new ArrayList(1500);
                }
                for (int j = 0; j < config.length; j++) {
                    if (config[j] != null && config[j].size() > 0) {
                        ++currIndex;
                        for (int k = 0; k < config[j].indexes.size(); k++) {
                            if ((dsetTest.data.get(
                                    config[j].indexes.get(k))).
                                    getCategory() != -1) {
                                split[currIndex].add(
                                        (dsetTest.data.get(
                                        config[j].indexes.get(k))).
                                        getCategory());
                            }
                        }
                    }
                }

                centroids = new DataInstance[configMin.length];
                numNonEmpty = 0;
                for (int j = 0; j < centroids.length; j++) {
                    if (configMin[j] != null && configMin[j].size() > 0) {
                        centroids[j] = configMin[j].getCentroid();
                        numNonEmpty++;
                        for (int p = 0; p < configMin[j].size(); p++) {
                            avgMinError[i] += cmet.dist(
                                    centroids[j], config[j].getInstance(p));
                        }
                    }
                }
                avgMinError[i] /= dsetTest.size();
                avgMinErr += avgMinError[i];

                avgClusterEntropy[i] = Info.evaluateInfoOfCategorySplit(
                        split, numClusters);
                avgEntropy += avgClusterEntropy[i];
                pwPGKH.println(silScores[i] + ", " + avgError[i] + ", "
                        + avgClusterEntropy[i]);

                currIndex = -1;
                split = new ArrayList[numNonEmpty];
                for (int j = 0; j < split.length; j++) {
                    split[j] = new ArrayList(1500);
                }
                for (int j = 0; j < configMin.length; j++) {
                    if (configMin[j] != null && configMin[j].size() > 0) {
                        ++currIndex;
                        for (int k = 0; k < configMin[j].indexes.size();
                                k++) {
                            if ((dsetTest.data.get(
                                    configMin[j].indexes.get(k))).
                                    getCategory() != -1) {
                                split[currIndex].add((dsetTest.data.get(
                                        configMin[j].indexes.get(k))).
                                        getCategory());
                            }
                        }
                    }
                }

                avgMinClusterEntropy[i] = Info.evaluateInfoOfCategorySplit(
                        split, numClusters);
                avgMinEntropy += avgMinClusterEntropy[i];
                pwMin.println(silMinScores[i] + ", " + avgMinError[i] + ", "
                        + avgMinClusterEntropy[i]);

            }
            avgSil /= numTimes;
            avgErr /= numTimes;
            avgTime /= numTimes;
            avgEntropy /= numTimes;
            pwPGKH.println(avgTime + ", " + avgSil + ", " + avgErr + ", "
                    + avgEntropy);
            pwPGKH.close();

            avgMinSil /= numTimes;
            avgMinErr /= numTimes;
            avgMinEntropy /= numTimes;
            pwMin.println(avgTime + ", " + avgMinSil + ", " + avgMinErr + ", "
                    + avgMinEntropy);
            pwMin.close();

            for (int i = 0; i < numTimes; i++) {
                System.out.println("HPKM " + i + "th iteration");

                boolean doneCorrectly = false;
                do {
                    if (cmet == null) {
                        cmet = CombinedMetric.FLOAT_MANHATTAN;
                    }
                    clustererHPKM.setNumClusters(numClusters);
                    clustererHPKM.setCombinedMetric(cmet);
                    clustererHPKM.setDataSet(dsetTest);
                    clustererHPKM.setK(hubnessK);
                    clustererHPKM.probabilisticIterations = 100;
                    avgTime = 0;
                    startTimer();
                    try {
                        clustererHPKM.cluster();
                        doneCorrectly = true;
                    } catch (Exception e) {
                        System.out.println("error: " + e.getMessage());
                        stopTimer();
                        numSec = 0;
                    }
                } while (!doneCorrectly);
                avgTime += numSec;
                pwHPKM.print(numSec + ", ");
                stopTimer();
                Cluster[] config = clustererHPKM.getClusters();
                Cluster[] configMin = clustererHPKM.getMinimizingClusters();
                for (int l = 10000; l < dsetTest.size(); l++) {
                    dsetTest.data.get(l).setCategory(-1);
                }
                QIndexSilhouette silIndex = new QIndexSilhouette(
                        numClusters, clustererHPKM.getClusterAssociations(),
                        dsetTest);
                silIndex.setDistanceMatrix(clustererHPKM.getNSFDistances());
                silHPKMScores[i] = silIndex.validity();
                silIndex = new QIndexSilhouette(numClusters,
                        clustererHPKM.getMinimizingAssociations(),
                        dsetTest);
                silIndex.setDistanceMatrix(clustererHPKM.getNSFDistances());
                silMinScores[i] = silIndex.validity();
                avgHPKMSil += silHPKMScores[i];
                avgMinSil += silMinScores[i];
                DataInstance[] centroids = new DataInstance[config.length];
                int numNonEmpty = 0;
                for (int j = 0; j < centroids.length; j++) {
                    if (config[j] != null && config[j].size() > 0) {
                        centroids[j] = config[j].getCentroid();
                        numNonEmpty++;
                        for (int p = 0; p < config[j].size(); p++) {
                            avgHPKMError[i] += cmet.dist(centroids[j],
                                    config[j].getInstance(p));
                        }
                    }
                }
                avgHPKMError[i] /= dsetTest.size();
                avgHPKMErr += avgHPKMError[i];


                // And now for supervised estimates - the cluster entropies.
                // First make the split.
                int currIndex = -1;
                ArrayList<Integer>[] split = new ArrayList[numNonEmpty];
                for (int j = 0; j < split.length; j++) {
                    split[j] = new ArrayList(1500);
                }
                for (int j = 0; j < config.length; j++) {
                    if (config[j] != null && config[j].size() > 0) {
                        ++currIndex;
                        for (int k = 0; k < config[j].indexes.size(); k++) {
                            if ((dsetTest.data.get(
                                    config[j].indexes.get(k))).
                                    getCategory() != -1) {
                                split[currIndex].add((dsetTest.data.get(
                                        config[j].indexes.get(k))).
                                        getCategory());
                            }
                        }
                    }
                }

                centroids = new DataInstance[configMin.length];
                numNonEmpty = 0;
                for (int j = 0; j < centroids.length; j++) {
                    if (configMin[j] != null && configMin[j].size() > 0) {
                        centroids[j] = configMin[j].getCentroid();
                        numNonEmpty++;
                        for (int p = 0; p < configMin[j].size(); p++) {
                            avgMinError[i] += cmet.dist(centroids[j],
                                    config[j].getInstance(p));
                        }
                    }
                }
                avgMinError[i] /= dsetTest.size();
                avgMinErr += avgMinError[i];

                avgHPKMClusterEntropy[i] = Info.evaluateInfoOfCategorySplit(
                        split, numClusters);
                avgHPKMEntropy += avgHPKMClusterEntropy[i];
                pwHPKM.println(silHPKMScores[i] + ", " + avgHPKMError[i]
                        + ", " + avgHPKMClusterEntropy[i]);

                currIndex = -1;
                split = new ArrayList[numNonEmpty];
                for (int j = 0; j < split.length; j++) {
                    split[j] = new ArrayList(1500);
                }
                for (int j = 0; j < configMin.length; j++) {
                    if (configMin[j] != null && configMin[j].size() > 0) {
                        ++currIndex;
                        for (int k = 0; k < configMin[j].indexes.size();
                                k++) {
                            if ((dsetTest.data.get(
                                    configMin[j].indexes.get(k))).
                                    getCategory() != -1) {
                                split[currIndex].add((dsetTest.data.get(
                                        configMin[j].indexes.get(k))).
                                        getCategory());
                            }
                        }
                    }
                }

                avgMinClusterEntropy[i] = Info.evaluateInfoOfCategorySplit(
                        split, numClusters);
                avgMinEntropy += avgMinClusterEntropy[i];

            }
            avgHPKMSil /= numTimes;
            avgHPKMErr /= numTimes;
            avgTime /= numTimes;
            avgHPKMEntropy /= numTimes;
            pwHPKM.println(avgTime + ", " + avgHPKMSil + ", " + avgHPKMErr
                    + ", " + avgHPKMEntropy);
            pwHPKM.close();

            avgMinSil /= numTimes;
            avgMinErr /= numTimes;
            avgMinEntropy /= numTimes;

            for (int i = 0; i < numTimes; i++) {
                System.out.println("KM " + i + "th iteration");

                boolean doneCorrectly = false;
                do {
                    if (cmet == null) {
                        cmet = CombinedMetric.FLOAT_MANHATTAN;
                    }
                    clustererKM.setNumClusters(numClusters);
                    clustererKM.setCombinedMetric(cmet);
                    clustererKM.setDataSet(dsetTest);
                    avgTime = 0;
                    startTimer();
                    try {
                        clustererKM.cluster();
                        doneCorrectly = true;
                    } catch (Exception e) {
                        System.out.println("error: " + e.getMessage());
                        stopTimer();
                        numSec = 0;
                    }
                } while (!doneCorrectly);
                avgTime += numSec;
                pwKM.print(numSec + ", ");
                stopTimer();
                Cluster[] config = clustererKM.getClusters();
                for (int l = 10000; l < dsetTest.size(); l++) {
                    dsetTest.data.get(l).setCategory(-1);
                }
                QIndexSilhouette silIndex = new QIndexSilhouette(
                        numClusters, clustererKM.getClusterAssociations(),
                        dsetTest);
                silIndex.setDistanceMatrix(clusterer.getNSFDistances());
                silKMScores[i] = silIndex.validity();
                avgKMSil += silKMScores[i];
                DataInstance[] centroids = new DataInstance[config.length];
                int numNonEmpty = 0;
                for (int j = 0; j < centroids.length; j++) {
                    if (config[j] != null && config[j].size() > 0) {
                        centroids[j] = config[j].getCentroid();
                        numNonEmpty++;
                        for (int p = 0; p < config[j].size(); p++) {
                            avgKMError[i] += cmet.dist(centroids[j],
                                    config[j].getInstance(p));
                        }
                    }
                }
                avgKMError[i] /= dsetTest.size();
                avgKMErr += avgKMError[i];
                // And now for supervised estimates - the cluster entropies.
                // First make the split.
                int currIndex = -1;
                ArrayList<Integer>[] split = new ArrayList[numNonEmpty];
                for (int j = 0; j < split.length; j++) {
                    split[j] = new ArrayList(1500);
                }
                for (int j = 0; j < config.length; j++) {
                    if (config[j] != null && config[j].size() > 0) {
                        ++currIndex;
                        for (int k = 0; k < config[j].indexes.size(); k++) {
                            if ((dsetTest.data.get(
                                    config[j].indexes.get(k))).
                                    getCategory() != -1) {
                                split[currIndex].add((dsetTest.data.get(
                                        config[j].indexes.get(k))).
                                        getCategory());
                            }
                        }
                    }
                }
                avgKMClusterEntropy[i] = Info.evaluateInfoOfCategorySplit(
                        split, numClusters);
                avgKMEntropy += avgKMClusterEntropy[i];
                pwKM.println(silKMScores[i] + ", " + avgKMError[i] + ", "
                        + avgKMClusterEntropy[i]);

            }
            avgKMSil /= numTimes;
            avgKMErr /= numTimes;
            avgTime /= numTimes;
            avgKMEntropy /= numTimes;
            pwKM.println(avgTime + ", " + avgKMSil + ", " + avgKMErr + ", "
                    + avgKMEntropy);
            pwKM.close();

            for (int i = 0; i < numTimes; i++) {
                System.out.println("GKH " + i + "th iteration");

                boolean doneCorrectly = false;
                do {
                    if (cmet == null) {
                        cmet = CombinedMetric.FLOAT_MANHATTAN;
                    }
                    clustererGKH.setNumClusters(numClusters);
                    clustererGKH.setCombinedMetric(cmet);
                    clustererGKH.setDataSet(dsetTest);
                    clustererGKH.setK(hubnessK);
                    avgTime = 0;
                    startTimer();
                    try {
                        clustererGKH.cluster();
                        doneCorrectly = true;
                    } catch (Exception e) {
                        System.out.println("error: " + e.getMessage());
                        stopTimer();
                        numSec = 0;
                    }
                } while (!doneCorrectly);
                avgTime += numSec;
                pwGKH.print(numSec + ", ");
                stopTimer();
                Cluster[] config = clustererGKH.getClusters();
                for (int l = 10000; l < dsetTest.size(); l++) {
                    dsetTest.data.get(l).setCategory(-1);
                }
                QIndexSilhouette silIndex = new QIndexSilhouette(
                        numClusters, clustererGKH.getClusterAssociations(),
                        dsetTest);
                silIndex.setDistanceMatrix(clusterer.getNSFDistances());
                silGKHScores[i] = silIndex.validity();
                avgGKHSil += silGKHScores[i];
                DataInstance[] centroids = new DataInstance[config.length];
                int numNonEmpty = 0;
                for (int j = 0; j < centroids.length; j++) {
                    if (config[j] != null && config[j].size() > 0) {
                        centroids[j] = config[j].getCentroid();
                        numNonEmpty++;
                        for (int p = 0; p < config[j].size(); p++) {
                            avgGKHError[i] += cmet.dist(centroids[j],
                                    config[j].getInstance(p));
                        }
                    }
                }
                avgGKHError[i] /= dsetTest.size();
                avgGKHErr += avgGKHError[i];
                // And now for supervised estimates - the cluster entropies.
                // First make the split.
                int currIndex = -1;
                ArrayList<Integer>[] split = new ArrayList[numNonEmpty];
                for (int j = 0; j < split.length; j++) {
                    split[j] = new ArrayList(1500);
                }
                for (int j = 0; j < config.length; j++) {
                    if (config[j] != null && config[j].size() > 0) {
                        ++currIndex;
                        for (int k = 0; k < config[j].indexes.size(); k++) {
                            if ((dsetTest.data.get(
                                    config[j].indexes.get(k))).
                                    getCategory() != -1) {
                                split[currIndex].add((dsetTest.data.get(
                                        config[j].indexes.get(k))).
                                        getCategory());
                            }
                        }
                    }
                }
                avgGKHClusterEntropy[i] = Info.evaluateInfoOfCategorySplit(
                        split, numClusters);
                avgGKHEntropy += avgGKHClusterEntropy[i];
                pwGKH.println(silGKHScores[i] + ", " + avgGKHError[i] + ", "
                        + avgGKHClusterEntropy[i]);

            }
            avgGKHSil /= numTimes;
            avgGKHErr /= numTimes;
            avgTime /= numTimes;
            avgGKHEntropy /= numTimes;
            pwGKH.println(avgTime + ", " + avgGKHSil + ", " + avgGKHErr + ", "
                    + avgGKHEntropy);
            pwGKH.close();
        }
    }

    /**
     * Information about the required command line parameters.
     */
    public static void info() {
        System.out.println("outDir");
        System.out.println("nTimes on dataset");
        System.out.println("nClusters");
        System.out.println("k");
        System.out.println("dim");
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 5) {
            info();
            return;
        }
        EvaluateOnNoisyMix noiser = new EvaluateOnNoisyMix();
        noiser.setWriterDir(args[0]);
        noiser.hubnessK = Integer.parseInt(args[3]);
        noiser.dim = Integer.parseInt(args[4]);
        noiser.clusterWithAlgorithmOnLabeledData(Integer.parseInt(args[1]),
                Integer.parseInt(args[2]));
    }
}
