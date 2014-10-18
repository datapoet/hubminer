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

import data.generators.util.MultiGaussianMixForClusteringTesting;
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

/**
 * Running the algorithm on its own and testing for performance gains for
 * different numbers of set probabilistic iterations.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class GPKHNumIterTweaker {

    int numSec = 0;
    javax.swing.Timer timeTimer;
    DataSet dset;
    PrintWriter pw;
    File writerDir;
    float[] dbScores;
    float[] dunnScores;
    float[] silScores;
    float[] avgError;
    float[] avgClusterEntropy;
    float avgDB;
    float avgDunn;
    float avgSil;
    float avgErr;
    float avgTime;
    float avgEntropy;
    public CombinedMetric cmet;
    boolean useRefinement = false;

    /**
     * @param dsetTest
     * @param pw
     */
    public GPKHNumIterTweaker(DataSet dsetTest, PrintWriter pw) {
        this.dset = dsetTest;
        this.pw = pw;
    }

    public GPKHNumIterTweaker() {
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
            //for testing purposes only
            try {
            } catch (Exception exc) {
            }
            //for testing purposes only
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
     * Loads data from the given path.
     *
     * @param path Path to the data file.
     * @throws Exception
     */
    public void loadData(String path) throws Exception {
        IOARFF persister = new IOARFF();
        dset = persister.load(path);
    }

    /**
     * Sets writer to output path.
     *
     * @param path Output data path.
     * @throws Exception
     */
    public void setWriterDir(String path) throws Exception {
        writerDir = new File(path);
    }

    /**
     * @param numDataSets Number of synthetic dataset.
     * @param numTimes Repetitions.
     * @param numClusters Number of clusters.
     * @throws Exception
     */
    public void clusterWithAlgorithmOnLabeledData(int numTimes, int numClusters,
            int minIter, int maxIter) throws Exception {
        GHPC clusterer = new GHPC();
        MultiGaussianMixForClusteringTesting genMix =
                new MultiGaussianMixForClusteringTesting(
                numClusters, 50, 10000, false);
        dset = genMix.generateRandomCollection();
        for (int probIter = minIter; probIter < maxIter; probIter += 5) {
            silScores = new float[numTimes];
            avgError = new float[numTimes];
            avgClusterEntropy = new float[numTimes];
            avgSil = 0;
            avgErr = 0;
            avgEntropy = 0;
            File currOutFile = new File(writerDir, "GPKH_numIter" + probIter
                    + "nClust" + numClusters + ".csv");
            FileUtil.createFile(currOutFile);
            pw = new PrintWriter(new FileWriter(currOutFile), true);
            pw.println("time, " + "SILHOUETTE" + ", " + "AVG_ERROR" + ", "
                    + "AVG_CLUSTER_ENTROPY");
            for (int i = 0; i < numTimes; i++) {
                System.out.println(i + "th iteration");

                boolean doneCorrectly = false;
                do {
                    if (cmet == null) {
                        cmet = CombinedMetric.FLOAT_MANHATTAN;
                    }
                    clusterer.setNumClusters(numClusters);
                    clusterer.setCombinedMetric(cmet);
                    clusterer.setDataSet(dset);
                    clusterer.probabilisticIterations = probIter;

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
                pw.print(numSec + ", ");
                stopTimer();
                Cluster[] config = clusterer.getClusters();
                QIndexSilhouette silIndex = new QIndexSilhouette(
                        numClusters, clusterer.getClusterAssociations(),
                        dset);
                silIndex.setDistanceMatrix(clusterer.getNSFDistances());
                silScores[i] = silIndex.validity();
                System.out.println("currSilIndex:" + silScores[i]);
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
                avgError[i] /= dset.size();
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
                                    (dset.data.get(
                                    config[j].indexes.get(
                                    k))).getCategory());
                        }
                    }
                }
                avgClusterEntropy[i] = Info.evaluateInfoOfCategorySplit(
                        split, numClusters);
                avgEntropy += avgClusterEntropy[i];
                pw.println(silScores[i] + ", " + avgError[i] + ", "
                        + avgClusterEntropy[i]);

            }
            avgSil /= numTimes;
            avgErr /= numTimes;
            avgTime /= numTimes;
            avgEntropy /= numTimes;
            pw.println(avgTime + ", " + avgSil + ", " + avgErr + ", "
                    + avgEntropy);
            pw.close();
        }
    }

    /**
     * Information about the required command line parameters.
     */
    public static void info() {
        System.out.println("outDir");
        System.out.println("nTimes on dataset");
        System.out.println("nClusters");
        System.out.println("minIter");
        System.out.println("maxIter");
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 5) {
            info();
            return;
        }
        GPKHNumIterTweaker tweaker = new GPKHNumIterTweaker();
        tweaker.setWriterDir(args[0]);
        tweaker.clusterWithAlgorithmOnLabeledData(
                Integer.parseInt(args[1]),
                Integer.parseInt(args[2]),
                Integer.parseInt(args[3]),
                Integer.parseInt(args[4]));
    }
}
