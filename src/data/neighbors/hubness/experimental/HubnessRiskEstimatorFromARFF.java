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
package data.neighbors.hubness.experimental;

import data.neighbors.NeighborSetFinder;
import data.neighbors.SharedNeighborFinder;
import data.representation.DataInstance;
import data.representation.DataSet;
import distances.primary.CombinedMetric;
import distances.secondary.MutualProximityCalculator;
import distances.secondary.NICDMCalculator;
import distances.secondary.snd.SharedNeighborCalculator;
import ioformat.FileUtil;
import ioformat.SupervisedLoader;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import learning.supervised.evaluation.ClassificationEstimator;
import learning.supervised.methods.knn.HFNN;
import learning.supervised.methods.knn.HIKNN;
import learning.supervised.methods.knn.KNN;
import learning.supervised.methods.knn.NHBNN;
import sampling.UniformSampler;
import statistics.HigherMoments;
import util.ArrayUtil;
import util.CommandLineParser;
import util.SOPLUtil;

/**
 * This class is meant to empirically estimate the distribution of the neighbor
 * occurrence skewness in synthetic data of controlled dimensionality under
 * standard metrics. It subsamples a loaded dataset and measures the hubness
 * stats. Additionally, it trains several kNN models and attempts classification
 * on a hold-out sample, measuring its robustness and stability.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class HubnessRiskEstimatorFromARFF {
    
    private File outFile;
    private int k = 5;
    private int kForSecondary = 50;
    private int numRepetitions = 500;
    private int sampleSize = 1000;
    private CombinedMetric cmet = CombinedMetric.EUCLIDEAN;
    private StatsLogger primaryLogger, nicdmLogger, simcosLogger, simhubLogger,
            mpLogger;
    private DataSet dsetTrain;
    private DataSet dsetTrainSub;
    private DataSet dsetTest;
    private float[][] dMatPrimaryTrainSub, dMatSecondaryTrainSub;
    private float[][] pointDistancesPrimary, pointDistancesSecondary;
    private int[][] pointNeighborsPrimary, pointNeighborsSecondary,
            pointNeighborsSecondaryK;
    private NeighborSetFinder nsfPrimary, nsfSecondary;
    public static final int NUM_THREADS = 8;
    
    /**
     * This method generates a new data subsample.
     * 
     * @return DataSet that is the subsample of the training data. 
     */
    private DataSet getSample() throws Exception {
        if (dsetTrain == null) {
            return null;
        }
        UniformSampler sampler = new UniformSampler(false);
        DataSet sampleData = sampler.getSample(dsetTrain, sampleSize);
        return sampleData;
    }
    
    /**
     * This method runs the script that examines the risk of hubness in
     * synthetic high-dimensional data.
     */
    private void performAllTests() throws Exception {
        primaryLogger = new StatsLogger("Euclidean");
        nicdmLogger = new StatsLogger("NICDM");
        simcosLogger = new StatsLogger("Simcos");
        simhubLogger = new StatsLogger("Simhub");
        mpLogger = new StatsLogger("MP");
        KNN knnClassifier;
        NHBNN nhbnnClassifier;
        HIKNN hiknnClassifier;
        HFNN hfnnClassifier;
        float accKNN, accNHBNN, accHIKNN, accHFNN;
        ClassificationEstimator clEstimator;
        int kPrimMax = Math.max(k, kForSecondary);
        DataInstance firstInstance, secondInstance;
        ArrayList<Integer> unitIndexesTest = new ArrayList<>(dsetTest.size());
        for (int i = 0; i < dsetTest.size(); i++) {
            unitIndexesTest.add(i);
        }
        int[] testLabels = dsetTest.obtainLabelArray();
        int numClasses = dsetTest.countCategories();
        for (int iteration = 0; iteration < numRepetitions; iteration++) {
            System.out.println("Starting iteration: " + iteration);
            do {
                dsetTrainSub = getSample();
            } while (dsetTrainSub.countCategories() !=
                    dsetTrain.countCategories());
            dsetTrainSub.orderInstancesByClasses();
            ArrayList<Integer> unitIndexes =
                    new ArrayList<>(dsetTrainSub.size());
            for (int i = 0; i < dsetTrainSub.size(); i++) {
                unitIndexes.add(i);
            }
            dMatPrimaryTrainSub =
                    dsetTrainSub.calculateDistMatrixMultThr(cmet, NUM_THREADS);
            pointDistancesPrimary = new float[dsetTest.size()][
                    dsetTrainSub.size()];
            for (int i = 0; i < dsetTest.size(); i++) {
                for (int j = 0; j < dsetTrainSub.size(); j++) {
                    firstInstance = dsetTest.getInstance(i);
                    secondInstance = dsetTrainSub.getInstance(j);
                    pointDistancesPrimary[i][j] = cmet.dist(firstInstance,
                            secondInstance);
                }
            }
            nsfPrimary = new NeighborSetFinder(dsetTrainSub,
                    dMatPrimaryTrainSub, cmet);
            nsfPrimary.calculateNeighborSets(kPrimMax);
            // We will re-calculate for the smaller k later, now we use this
            // kNN object for secondary distances, where necessary.
            // Calculate the secondary NICDM distances.
            NICDMCalculator nsc = new NICDMCalculator(nsfPrimary);
            dMatSecondaryTrainSub =
                    nsc.getTransformedDMatFromNSFPrimaryDMat();
            nsfSecondary = new NeighborSetFinder(dsetTrainSub,
                    dMatSecondaryTrainSub, nsc);
            nsfSecondary.calculateNeighborSets(k);
            nicdmLogger.updateByObservedFreqs(
                    nsfSecondary.getNeighborFrequencies());
            nicdmLogger.updateLabelMismatchPercentages(
                    nsfSecondary.getKNeighbors());
            pointDistancesSecondary = new float[dsetTest.size()][
                    dsetTrainSub.size()];
            pointNeighborsSecondary = new int[dsetTest.size()][k];
            pointNeighborsSecondaryK = new int[dsetTest.size()][kForSecondary];
            for (int index = 0; index < dsetTest.size(); index++) {
                firstInstance = dsetTest.getInstance(index);
                pointNeighborsSecondaryK[index] =
                        NeighborSetFinder.getIndexesOfNeighbors(
                        dsetTrainSub, firstInstance, kForSecondary,
                        pointDistancesPrimary[index]);
            }
            for (int indexFirst = 0; indexFirst < dsetTest.size();
                    indexFirst++) {
                for (int indexSecond = 0; indexSecond < dsetTrainSub.size();
                        indexSecond++) {
                    firstInstance = dsetTest.getInstance(indexFirst);
                    secondInstance = dsetTrainSub.getInstance(indexSecond);
                    int[] firstNeighbors = pointNeighborsSecondaryK[indexFirst];
                    float[] kDistsFirst = new float[kForSecondary];
                    float[] kDistsSecond = nsfPrimary.getKDistances()[
                            indexSecond];
                    for (int kInd = 0; kInd < kForSecondary; kInd++) {
                        kDistsFirst[kInd] = pointDistancesPrimary[indexFirst][
                                firstNeighbors[kInd]];
                    }
                    pointDistancesSecondary[indexFirst][indexSecond] =
                            nsc.distFromKDists(firstInstance, secondInstance,
                            kDistsFirst, kDistsSecond);
                }
            }
            pointNeighborsSecondary =
                    NeighborSetFinder.getIndexesOfNeighbors(dsetTrainSub,
                    dsetTest, k, pointDistancesSecondary);
            // Initialize the classifiers.
            knnClassifier = new KNN(k, nsc);
            nhbnnClassifier = new NHBNN(k, nsc, numClasses);
            hiknnClassifier = new HIKNN(k, nsc, numClasses);
            hfnnClassifier = new HFNN(k, nsc, numClasses);
            // Set the data.
            knnClassifier.setDataIndexes(unitIndexes, dsetTrainSub);
            nhbnnClassifier.setDataIndexes(unitIndexes, dsetTrainSub);
            hiknnClassifier.setDataIndexes(unitIndexes, dsetTrainSub);
            hfnnClassifier.setDataIndexes(unitIndexes, dsetTrainSub);
            // Set the distances and the kNN sets.
            nhbnnClassifier.setDistMatrix(dMatSecondaryTrainSub);
            nhbnnClassifier.setNSF(nsfSecondary);
            hiknnClassifier.setDistMatrix(dMatSecondaryTrainSub);
            hiknnClassifier.setNSF(nsfSecondary);
            hfnnClassifier.setDistMatrix(dMatSecondaryTrainSub);
            hfnnClassifier.setNSF(nsfSecondary);
            // Train the models.
            knnClassifier.train();
            nhbnnClassifier.train();
            hiknnClassifier.train();
            hfnnClassifier.train();
            // Test the classifiers.
            clEstimator = knnClassifier.test(unitIndexesTest, dsetTest,
                    testLabels, numClasses, pointDistancesSecondary,
                    pointNeighborsSecondary);
            accKNN = clEstimator.getAccuracy();
            clEstimator = nhbnnClassifier.test(unitIndexesTest, dsetTest,
                    testLabels, numClasses, pointDistancesSecondary,
                    pointNeighborsSecondary);
            accNHBNN = clEstimator.getAccuracy();
            clEstimator = hiknnClassifier.test(unitIndexesTest, dsetTest,
                    testLabels, numClasses, pointDistancesSecondary,
                    pointNeighborsSecondary);
            accHIKNN = clEstimator.getAccuracy();
            clEstimator = hfnnClassifier.test(unitIndexesTest, dsetTest,
                    testLabels, numClasses, pointDistancesSecondary,
                    pointNeighborsSecondary);
            accHFNN = clEstimator.getAccuracy();
            nicdmLogger.updateByClassifierAccuracies(accKNN, accNHBNN, accHIKNN,
                    accHFNN);
            // Calculate the secondary Simcos distances.
            SharedNeighborFinder snf =
                    new SharedNeighborFinder(nsfPrimary, k);
            snf.setNumClasses(numClasses);
            snf.countSharedNeighborsMultiThread(NUM_THREADS);
            // First fetch the similarities.
            dMatSecondaryTrainSub = snf.getSharedNeighborCounts();
            // Then transform them into distances.
            for (int indexFirst = 0; indexFirst < dMatSecondaryTrainSub.length;
                    indexFirst++) {
                for (int indexSecond = 0; indexSecond <
                        dMatSecondaryTrainSub[indexFirst].length;
                        indexSecond++) {
                    dMatSecondaryTrainSub[indexFirst][indexSecond] =
                            kForSecondary -
                            dMatSecondaryTrainSub[indexFirst][indexSecond];
                }
            }
            SharedNeighborCalculator snc =
                    new SharedNeighborCalculator(snf,SharedNeighborCalculator.
                    WeightingType.NONE);
            nsfSecondary = new NeighborSetFinder(dsetTrainSub,
                    dMatSecondaryTrainSub, snc);
            nsfSecondary.calculateNeighborSets(k);
            simcosLogger.updateByObservedFreqs(
                    nsfSecondary.getNeighborFrequencies());
            simcosLogger.updateLabelMismatchPercentages(
                    nsfSecondary.getKNeighbors());
            // Calculate the test-to-training point distances.
            pointDistancesSecondary = new float[dsetTest.size()][
                    dsetTrainSub.size()];
            pointNeighborsSecondary = new int[dsetTest.size()][k];
            pointNeighborsSecondaryK = new int[dsetTest.size()][kForSecondary];
            for (int index = 0; index < dsetTest.size(); index++) {
                firstInstance = dsetTest.getInstance(index);
                pointNeighborsSecondaryK[index] =
                        NeighborSetFinder.getIndexesOfNeighbors(
                        dsetTrainSub, firstInstance, kForSecondary,
                        pointDistancesPrimary[index]);
            }
            for (int indexFirst = 0; indexFirst < dsetTest.size();
                    indexFirst++) {
                for (int indexSecond = 0; indexSecond < dsetTrainSub.size();
                        indexSecond++) {
                    firstInstance = dsetTest.getInstance(indexFirst);
                    secondInstance = dsetTrainSub.getInstance(indexSecond);
                    pointDistancesSecondary[indexFirst][indexSecond] =
                            snc.dist(firstInstance, secondInstance,
                            pointNeighborsSecondaryK[indexFirst],
                            nsfPrimary.getKNeighbors()[indexSecond]);
                }
            }
            pointNeighborsSecondary =
                    NeighborSetFinder.getIndexesOfNeighbors(dsetTrainSub,
                    dsetTest, k, pointDistancesSecondary);
            // Initialize the classifiers.
            knnClassifier = new KNN(k, snc);
            nhbnnClassifier = new NHBNN(k, snc, numClasses);
            hiknnClassifier = new HIKNN(k, snc, numClasses);
            hfnnClassifier = new HFNN(k, snc, numClasses);
            // Set the data.
            knnClassifier.setDataIndexes(unitIndexes, dsetTrainSub);
            nhbnnClassifier.setDataIndexes(unitIndexes, dsetTrainSub);
            hiknnClassifier.setDataIndexes(unitIndexes, dsetTrainSub);
            hfnnClassifier.setDataIndexes(unitIndexes, dsetTrainSub);
            // Set the distances and the kNN sets.
            nhbnnClassifier.setDistMatrix(dMatSecondaryTrainSub);
            nhbnnClassifier.setNSF(nsfSecondary);
            hiknnClassifier.setDistMatrix(dMatSecondaryTrainSub);
            hiknnClassifier.setNSF(nsfSecondary);
            hfnnClassifier.setDistMatrix(dMatSecondaryTrainSub);
            hfnnClassifier.setNSF(nsfSecondary);
            // Train the models.
            knnClassifier.train();
            nhbnnClassifier.train();
            hiknnClassifier.train();
            hfnnClassifier.train();
            // Test the classifiers.
            clEstimator = knnClassifier.test(unitIndexesTest, dsetTest,
                    testLabels, numClasses, pointDistancesSecondary,
                    pointNeighborsSecondary);
            accKNN = clEstimator.getAccuracy();
            clEstimator = nhbnnClassifier.test(unitIndexesTest, dsetTest,
                    testLabels, numClasses, pointDistancesSecondary,
                    pointNeighborsSecondary);
            accNHBNN = clEstimator.getAccuracy();
            clEstimator = hiknnClassifier.test(unitIndexesTest, dsetTest,
                    testLabels, numClasses, pointDistancesSecondary,
                    pointNeighborsSecondary);
            accHIKNN = clEstimator.getAccuracy();
            clEstimator = hfnnClassifier.test(unitIndexesTest, dsetTest,
                    testLabels, numClasses, pointDistancesSecondary,
                    pointNeighborsSecondary);
            accHFNN = clEstimator.getAccuracy();
            simcosLogger.updateByClassifierAccuracies(accKNN, accNHBNN,
                    accHIKNN, accHFNN);
            // Calculate the secondary Simhub distances. These are actually the
            // simhub^inf variant, since there are not classes in the data.
            snf = new SharedNeighborFinder(nsfPrimary, k);
            snf.setNumClasses(numClasses);
            snf.obtainWeightsFromHubnessInformation();
            snf.countSharedNeighborsMultiThread(NUM_THREADS);
            // First fetch the similarities.
            dMatSecondaryTrainSub = snf.getSharedNeighborCounts();
            // Then transform them into distances.
            for (int indexFirst = 0; indexFirst < dMatSecondaryTrainSub.length;
                    indexFirst++) {
                for (int indexSecond = 0; indexSecond <
                        dMatSecondaryTrainSub[indexFirst].length;
                        indexSecond++) {
                    dMatSecondaryTrainSub[indexFirst][indexSecond] =
                            kForSecondary -
                            dMatSecondaryTrainSub[indexFirst][indexSecond];
                }
            }
            snc = new SharedNeighborCalculator(snf,SharedNeighborCalculator.
                    WeightingType.HUBNESS_INFORMATION);
            nsfSecondary = new NeighborSetFinder(dsetTrainSub,
                    dMatSecondaryTrainSub, snc);
            nsfSecondary.calculateNeighborSets(k);
            simhubLogger.updateByObservedFreqs(
                    nsfSecondary.getNeighborFrequencies());
            simhubLogger.updateLabelMismatchPercentages(
                    nsfSecondary.getKNeighbors());
            pointDistancesSecondary = new float[dsetTest.size()][
                    dsetTrainSub.size()];
            pointNeighborsSecondary = new int[dsetTest.size()][k];
            pointNeighborsSecondaryK = new int[dsetTest.size()][kForSecondary];
            for (int index = 0; index < dsetTest.size(); index++) {
                firstInstance = dsetTest.getInstance(index);
                pointNeighborsSecondaryK[index] =
                        NeighborSetFinder.getIndexesOfNeighbors(
                        dsetTrainSub, firstInstance, kForSecondary,
                        pointDistancesPrimary[index]);
            }
            for (int indexFirst = 0; indexFirst < dsetTest.size();
                    indexFirst++) {
                for (int indexSecond = 0; indexSecond < dsetTrainSub.size();
                        indexSecond++) {
                    firstInstance = dsetTest.getInstance(indexFirst);
                    secondInstance = dsetTrainSub.getInstance(indexSecond);
                    pointDistancesSecondary[indexFirst][indexSecond] =
                            snc.dist(firstInstance, secondInstance,
                            pointNeighborsSecondaryK[indexFirst],
                            nsfPrimary.getKNeighbors()[indexSecond]);
                }
            }
            pointNeighborsSecondary =
                    NeighborSetFinder.getIndexesOfNeighbors(dsetTrainSub,
                    dsetTest, k, pointDistancesSecondary);
            // Initialize the classifiers.
            knnClassifier = new KNN(k, snc);
            nhbnnClassifier = new NHBNN(k, snc, numClasses);
            hiknnClassifier = new HIKNN(k, snc, numClasses);
            hfnnClassifier = new HFNN(k, snc, numClasses);
            // Set the data.
            knnClassifier.setDataIndexes(unitIndexes, dsetTrainSub);
            nhbnnClassifier.setDataIndexes(unitIndexes, dsetTrainSub);
            hiknnClassifier.setDataIndexes(unitIndexes, dsetTrainSub);
            hfnnClassifier.setDataIndexes(unitIndexes, dsetTrainSub);
            // Set the distances and the kNN sets.
            nhbnnClassifier.setDistMatrix(dMatSecondaryTrainSub);
            nhbnnClassifier.setNSF(nsfSecondary);
            hiknnClassifier.setDistMatrix(dMatSecondaryTrainSub);
            hiknnClassifier.setNSF(nsfSecondary);
            hfnnClassifier.setDistMatrix(dMatSecondaryTrainSub);
            hfnnClassifier.setNSF(nsfSecondary);
            // Train the models.
            knnClassifier.train();
            nhbnnClassifier.train();
            hiknnClassifier.train();
            hfnnClassifier.train();
            // Test the classifiers.
            clEstimator = knnClassifier.test(unitIndexesTest, dsetTest,
                    testLabels, numClasses, pointDistancesSecondary,
                    pointNeighborsSecondary);
            accKNN = clEstimator.getAccuracy();
            clEstimator = nhbnnClassifier.test(unitIndexesTest, dsetTest,
                    testLabels, numClasses, pointDistancesSecondary,
                    pointNeighborsSecondary);
            accNHBNN = clEstimator.getAccuracy();
            clEstimator = hiknnClassifier.test(unitIndexesTest, dsetTest,
                    testLabels, numClasses, pointDistancesSecondary,
                    pointNeighborsSecondary);
            accHIKNN = clEstimator.getAccuracy();
            clEstimator = hfnnClassifier.test(unitIndexesTest, dsetTest,
                    testLabels, numClasses, pointDistancesSecondary,
                    pointNeighborsSecondary);
            accHFNN = clEstimator.getAccuracy();
            simhubLogger.updateByClassifierAccuracies(accKNN, accNHBNN,
                    accHIKNN, accHFNN);
            // Calculate the secondary Mutual Proximity distances.
            MutualProximityCalculator calc =
                    new MutualProximityCalculator(nsfPrimary.getDistances(),
                    nsfPrimary.getDataSet(), nsfPrimary.getCombinedMetric());
            dMatSecondaryTrainSub = calc.calculateSecondaryDistMatrixMultThr(
                    nsfPrimary, 8);
            nsfSecondary = new NeighborSetFinder(dsetTrainSub,
                    dMatSecondaryTrainSub, calc);
            nsfSecondary.calculateNeighborSets(k);
            mpLogger.updateByObservedFreqs(
                    nsfSecondary.getNeighborFrequencies());
            mpLogger.updateLabelMismatchPercentages(
                    nsfSecondary.getKNeighbors());
            pointDistancesSecondary = new float[dsetTest.size()][
                    dsetTrainSub.size()];
            pointNeighborsSecondary = new int[dsetTest.size()][k];
            pointNeighborsSecondaryK = new int[dsetTest.size()][kForSecondary];
            for (int index = 0; index < dsetTest.size(); index++) {
                firstInstance = dsetTest.getInstance(index);
                pointNeighborsSecondaryK[index] =
                        NeighborSetFinder.getIndexesOfNeighbors(
                        dsetTrainSub, firstInstance, kForSecondary,
                        pointDistancesPrimary[index]);
            }
            for (int indexFirst = 0; indexFirst < dsetTest.size();
                    indexFirst++) {
                for (int indexSecond = 0; indexSecond < dsetTrainSub.size();
                        indexSecond++) {
                    firstInstance = dsetTest.getInstance(indexFirst);
                    secondInstance = dsetTrainSub.getInstance(indexSecond);
                    int[] firstNeighbors = pointNeighborsSecondaryK[
                            indexFirst];
                    float[] kDistsFirst = new float[kForSecondary];
                    float[] kDistsSecond =
                            nsfPrimary.getKDistances()[indexSecond];
                    for (int kInd = 0; kInd < kForSecondary; kInd++) {
                        kDistsFirst[kInd] = pointDistancesPrimary[indexFirst][
                                firstNeighbors[kInd]];
                    }
                    pointDistancesSecondary[indexFirst][indexSecond] =
                            calc.dist(firstInstance, secondInstance,
                            kDistsFirst, kDistsSecond);
                }
            }
            pointNeighborsSecondary =
                    NeighborSetFinder.getIndexesOfNeighbors(dsetTrainSub,
                    dsetTest, k, pointDistancesSecondary);
            // Initialize the classifiers.
            knnClassifier = new KNN(k, calc);
            nhbnnClassifier = new NHBNN(k, calc, numClasses);
            hiknnClassifier = new HIKNN(k, calc, numClasses);
            hfnnClassifier = new HFNN(k, calc, numClasses);
            // Set the data.
            knnClassifier.setDataIndexes(unitIndexes, dsetTrainSub);
            nhbnnClassifier.setDataIndexes(unitIndexes, dsetTrainSub);
            hiknnClassifier.setDataIndexes(unitIndexes, dsetTrainSub);
            hfnnClassifier.setDataIndexes(unitIndexes, dsetTrainSub);
            // Set the distances and the kNN sets.
            nhbnnClassifier.setDistMatrix(dMatSecondaryTrainSub);
            nhbnnClassifier.setNSF(nsfSecondary);
            hiknnClassifier.setDistMatrix(dMatSecondaryTrainSub);
            hiknnClassifier.setNSF(nsfSecondary);
            hfnnClassifier.setDistMatrix(dMatSecondaryTrainSub);
            hfnnClassifier.setNSF(nsfSecondary);
            // Train the models.
            knnClassifier.train();
            nhbnnClassifier.train();
            hiknnClassifier.train();
            hfnnClassifier.train();
            // Test the classifiers.
            clEstimator = knnClassifier.test(unitIndexesTest, dsetTest,
                    testLabels, numClasses, pointDistancesSecondary,
                    pointNeighborsSecondary);
            accKNN = clEstimator.getAccuracy();
            clEstimator = nhbnnClassifier.test(unitIndexesTest, dsetTest,
                    testLabels, numClasses, pointDistancesSecondary,
                    pointNeighborsSecondary);
            accNHBNN = clEstimator.getAccuracy();
            clEstimator = hiknnClassifier.test(unitIndexesTest, dsetTest,
                    testLabels, numClasses, pointDistancesSecondary,
                    pointNeighborsSecondary);
            accHIKNN = clEstimator.getAccuracy();
            clEstimator = hfnnClassifier.test(unitIndexesTest, dsetTest,
                    testLabels, numClasses, pointDistancesSecondary,
                    pointNeighborsSecondary);
            accHFNN = clEstimator.getAccuracy();
            mpLogger.updateByClassifierAccuracies(accKNN, accNHBNN, accHIKNN,
                    accHFNN);
            // Finally the primary distances.
            nsfPrimary = nsfPrimary.getSubNSF(k);
            primaryLogger.updateByObservedFreqs(
                    nsfPrimary.getNeighborFrequencies());
            primaryLogger.updateLabelMismatchPercentages(
                    nsfPrimary.getKNeighbors());
            pointNeighborsPrimary =
                    NeighborSetFinder.getIndexesOfNeighbors(dsetTrainSub,
                    dsetTest, k, pointDistancesPrimary);
            // Initialize the classifiers.
            knnClassifier = new KNN(k, nsc);
            nhbnnClassifier = new NHBNN(k, nsc, numClasses);
            hiknnClassifier = new HIKNN(k, nsc, numClasses);
            hfnnClassifier = new HFNN(k, nsc, numClasses);
            // Set the data.
            knnClassifier.setDataIndexes(unitIndexes, dsetTrainSub);
            nhbnnClassifier.setDataIndexes(unitIndexes, dsetTrainSub);
            hiknnClassifier.setDataIndexes(unitIndexes, dsetTrainSub);
            hfnnClassifier.setDataIndexes(unitIndexes, dsetTrainSub);
            // Set the distances and the kNN sets.
            nhbnnClassifier.setDistMatrix(dMatPrimaryTrainSub);
            nhbnnClassifier.setNSF(nsfPrimary);
            hiknnClassifier.setDistMatrix(dMatPrimaryTrainSub);
            hiknnClassifier.setNSF(nsfPrimary);
            hfnnClassifier.setDistMatrix(dMatPrimaryTrainSub);
            hfnnClassifier.setNSF(nsfPrimary);
            // Train the models.
            knnClassifier.train();
            nhbnnClassifier.train();
            hiknnClassifier.train();
            hfnnClassifier.train();
            // Test the classifiers.
            clEstimator = knnClassifier.test(unitIndexesTest, dsetTest,
                    testLabels, numClasses, pointDistancesPrimary,
                    pointNeighborsPrimary);
            accKNN = clEstimator.getAccuracy();
            clEstimator = nhbnnClassifier.test(unitIndexesTest, dsetTest,
                    testLabels, numClasses, pointDistancesPrimary,
                    pointNeighborsPrimary);
            accNHBNN = clEstimator.getAccuracy();
            clEstimator = hiknnClassifier.test(unitIndexesTest, dsetTest,
                    testLabels, numClasses, pointDistancesPrimary,
                    pointNeighborsPrimary);
            accHIKNN = clEstimator.getAccuracy();
            clEstimator = hfnnClassifier.test(unitIndexesTest, dsetTest,
                    testLabels, numClasses, pointDistancesPrimary,
                    pointNeighborsPrimary);
            accHFNN = clEstimator.getAccuracy();
            primaryLogger.updateByClassifierAccuracies(accKNN, accNHBNN,
                    accHIKNN, accHFNN);
            // Try some garbage collection.
            System.gc();
        }
        // Print out the results.
        FileUtil.createFile(outFile);
        try (PrintWriter pw = new PrintWriter(new FileWriter(outFile))) {
            primaryLogger.printLoggerToStream(pw);
            pw.println();
            nicdmLogger.printLoggerToStream(pw);
            pw.println();
            simcosLogger.printLoggerToStream(pw);
            pw.println();
            simhubLogger.printLoggerToStream(pw);
            pw.println();
            mpLogger.printLoggerToStream(pw);
            pw.println();
        } catch (Exception e) {
            throw e;
        }
    }

    private class StatsLogger {
        
        // Name of the distance that this logger is logging for.
        private String distName;
        
        private ArrayList<Float> skewValues;
        private ArrayList<Float> labelMismatchPercs;
        private ArrayList<Float> kurtosisValues;
        private ArrayList<Float> knnAccuracies;
        private ArrayList<Float> nhbnnAccuracies;
        private ArrayList<Float> hiknnAccuracies;
        private ArrayList<Float> hfnnAccuracies;
        
        /**
         * Initialization.
         */
        public StatsLogger(String distName) {
            this.distName = distName;
            skewValues = new ArrayList<>(numRepetitions);
            labelMismatchPercs = new ArrayList<>(numRepetitions);
            kurtosisValues = new ArrayList<>(numRepetitions);
            knnAccuracies = new ArrayList<>(numRepetitions);
            nhbnnAccuracies = new ArrayList<>(numRepetitions);
            hiknnAccuracies = new ArrayList<>(numRepetitions);
            hfnnAccuracies = new ArrayList<>(numRepetitions);
        }
        
        /**
         * This method inserts the calculated classifier accuracies from the
         * current iteration into the log.
         * 
         * @param knnAccuracy Float value that is the kNN classification
         * accuracy.
         * @param nhbnnAccuracy Float value that is the NHBNN classification
         * accuracy.
         * @param hiknnAccuracy Float value that is the HIKNN classification
         * accuracy.
         * @param hfnnAccuracy Float value that is the hFNN classification
         * accuracy.
         */
        public void updateByClassifierAccuracies(float knnAccuracy,
                float nhbnnAccuracy, float hiknnAccuracy, float hfnnAccuracy) {
            knnAccuracies.add(knnAccuracy);
            nhbnnAccuracies.add(nhbnnAccuracy);
            hiknnAccuracies.add(hiknnAccuracy);
            hfnnAccuracies.add(hfnnAccuracy);
        }
        
        /**
         * This method calculates and updates the list of label mismatch 
         * percentages in kNN sets on the data.
         * 
         * @param kNeighbors int[][] representing the k-nearest neighbors. 
         */
        public void updateLabelMismatchPercentages(int[][] kNeighbors) {
            if (kNeighbors != null && kNeighbors.length > 0 &&
                    dsetTrainSub != null && dsetTest != null) {
                int[] testLabels = dsetTest.obtainLabelArray();
                int[] trainingLabels = dsetTrainSub.obtainLabelArray();
                float totalMismatches = 0;
                for (int i = 0; i < dsetTest.size(); i++) {
                    for (int kInd = 0; kInd < k; kInd++) {
                        if (trainingLabels[kNeighbors[i][kInd]] !=
                                testLabels[i]) {
                            totalMismatches++;
                        }
                    }
                }
                float mismatchPerc = totalMismatches / (k * dsetTest.size());
                labelMismatchPercs.add(mismatchPerc);
            }
        }
        
        /**
         * This method looks at the neighbor occurrence frequencies, calculates
         * the skewness and kurtosis and updates the stats logger object.
         * 
         * @param occFreqs int[] representing the neighbor occurrence
         * frequencies.
         */
        public void updateByObservedFreqs(int[] occFreqs) {
            float skew = HigherMoments.calculateSkewForSampleArray(occFreqs);
            float kurtosis =
                    HigherMoments.calculateKurtosisForSampleArray(occFreqs);
            skewValues.add(skew);
            kurtosisValues.add(kurtosis);
        }
        
        /**
         * This method prints the contents of the current logger to stream.
         * 
         * @param pw PrintWriter to print the logger to.
         */
        public void printLoggerToStream(PrintWriter pw) {
            // Label micmatch percentages.
            pw.println("Sampled label mismatch percentages for: " + distName);
            SOPLUtil.printArrayListToStream(labelMismatchPercs, pw, ",");
            pw.println("Label mismatch percs historgram: ");
            SOPLUtil.printArrayListToStream(getHistogram(labelMismatchPercs,
                    0.01f), pw, ",");
            // The hubness meta-skews and kurtosis.
            pw.println("Sampled hubnesses for: " + distName);
            SOPLUtil.printArrayListToStream(skewValues, pw, ",");
            pw.println("Calculated moments (mean, stdev, skew, kurtosis):");
            float sMean = HigherMoments.calculateArrayListMean(skewValues);
            float sStDev = HigherMoments.calculateArrayListStDev(sMean,
                    skewValues);
            float sSkew = HigherMoments.calculateSkewForSampleArrayList(
                    skewValues);
            float sKurtosis = HigherMoments.calculateKurtosisForSampleArrayList(
                    skewValues);
            pw.println(sMean + "," + sStDev + "," + sSkew + "," + sKurtosis);
            pw.println("Skew histogram: ");
            SOPLUtil.printArrayListToStream(getHistogram(skewValues, 0.1f),
                    pw, ",");
            pw.println("Samples occ. kurtosis for: " + distName);
            SOPLUtil.printArrayListToStream(kurtosisValues, pw, ",");
            pw.println("Calculated moments (mean, stdev, skew, kurtosis):");
            float kMean = HigherMoments.calculateArrayListMean(kurtosisValues);
            float kStDev = HigherMoments.calculateArrayListStDev(kMean,
                    kurtosisValues);
            float kSkew = HigherMoments.calculateSkewForSampleArrayList(
                    kurtosisValues);
            float kKurtosis = HigherMoments.calculateKurtosisForSampleArrayList(
                    kurtosisValues);
            pw.println(kMean + "," + kStDev + "," + kSkew + "," + kKurtosis);
            pw.println("Kurtosis histogram: ");
            SOPLUtil.printArrayListToStream(getHistogram(kurtosisValues, 0.1f),
                    pw, ",");
            // Then the classification accuracies.
            pw.println("kNN classification accuracy");
            SOPLUtil.printArrayListToStream(knnAccuracies, pw, ",");
            pw.println("Calculated moments (mean, stdev, skew, kurtosis):");
            float cMean = HigherMoments.calculateArrayListMean(knnAccuracies);
            float cStDev = HigherMoments.calculateArrayListStDev(cMean,
                    knnAccuracies);
            float cSkew = HigherMoments.calculateSkewForSampleArrayList(
                    knnAccuracies);
            float cKurtosis = HigherMoments.calculateKurtosisForSampleArrayList(
                    knnAccuracies);
            pw.println(cMean + "," + cStDev + "," + cSkew + "," + cKurtosis);
            pw.println("Accuracy histogram: ");
            SOPLUtil.printArrayListToStream(getHistogram(knnAccuracies, 0.005f),
                    pw, ",");
            pw.println("NHBNN classification accuracy");
            SOPLUtil.printArrayListToStream(nhbnnAccuracies, pw, ",");
            pw.println("Calculated moments (mean, stdev, skew, kurtosis):");
            cMean = HigherMoments.calculateArrayListMean(nhbnnAccuracies);
            cStDev = HigherMoments.calculateArrayListStDev(cMean,
                    nhbnnAccuracies);
            cSkew = HigherMoments.calculateSkewForSampleArrayList(
                    nhbnnAccuracies);
            cKurtosis = HigherMoments.calculateKurtosisForSampleArrayList(
                    nhbnnAccuracies);
            pw.println(cMean + "," + cStDev + "," + cSkew + "," + cKurtosis);
            pw.println("Accuracy histogram: ");
            SOPLUtil.printArrayListToStream(getHistogram(nhbnnAccuracies,
                    0.005f), pw, ",");
            pw.println("HIKNN classification accuracy");
            SOPLUtil.printArrayListToStream(hiknnAccuracies, pw, ",");
            pw.println("Calculated moments (mean, stdev, skew, kurtosis):");
            cMean = HigherMoments.calculateArrayListMean(hiknnAccuracies);
            cStDev = HigherMoments.calculateArrayListStDev(cMean,
                    hiknnAccuracies);
            cSkew = HigherMoments.calculateSkewForSampleArrayList(
                    hiknnAccuracies);
            cKurtosis = HigherMoments.calculateKurtosisForSampleArrayList(
                    hiknnAccuracies);
            pw.println(cMean + "," + cStDev + "," + cSkew + "," + cKurtosis);
            pw.println("Accuracy histogram: ");
            SOPLUtil.printArrayListToStream(getHistogram(hiknnAccuracies,
                    0.005f), pw, ",");
            pw.println("hFNN classification accuracy");
            SOPLUtil.printArrayListToStream(hfnnAccuracies, pw, ",");
            pw.println("Calculated moments (mean, stdev, skew, kurtosis):");
            cMean = HigherMoments.calculateArrayListMean(hfnnAccuracies);
            cStDev = HigherMoments.calculateArrayListStDev(cMean,
                    hfnnAccuracies);
            cSkew = HigherMoments.calculateSkewForSampleArrayList(
                    hfnnAccuracies);
            cKurtosis = HigherMoments.calculateKurtosisForSampleArrayList(
                    hfnnAccuracies);
            pw.println(cMean + "," + cStDev + "," + cSkew + "," + cKurtosis);
            pw.println("Accuracy histogram: ");
            SOPLUtil.printArrayListToStream(getHistogram(hfnnAccuracies,
                    0.005f), pw, ",");
        }
        
        /**
         * This method obtains a histogram from the measurements.
         * 
         * @param values ArrayList<Float> representing the values to get the
         * histogram for.
         * @param bucketWidth Integer that is the histogram bucket width.
         * @return ArrayList<Integer> that is the histogram for the provided
         * values.
         */
        private ArrayList<Integer> getHistogram(ArrayList<Float> values,
                float bucketWidth) {
            int maxValue = (int) (ArrayUtil.maxOfFloatList(values) + 1);
            int minValue = (int) Math.floor(ArrayUtil.minOfFloatList(values));
            if (minValue >= 0) {
                minValue = 0;
            }
            int numBuckets = (int) (((maxValue - minValue) / bucketWidth) + 1)
                    + 1;
            ArrayList<Integer> counts = new ArrayList<>(numBuckets);
            for (int i = 0; i < numBuckets; i++) {
                counts.add(new Integer(0));
            }
            int buckIndex;
            for (int i = 0; i < values.size(); i++) {
                if (values.get(i) < minValue) {
                    System.out.println(values.get(i) + " " + minValue);
                }
                buckIndex = (int) ((values.get(i) - minValue) / bucketWidth);
                counts.set(buckIndex, counts.get(buckIndex) + 1);
            }
            return counts;
        }
        
    }

    /**
     * This method runs the script.
     *
     * @param args Command line parameters, as specified.
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        CommandLineParser clp = new CommandLineParser(true);
        clp.addParam("-sampleSize", "Number of samples to draw in each"
                + " iteration.", CommandLineParser.INTEGER, true, false);
        clp.addParam("-testSize", "Number of test samples to use.",
                CommandLineParser.INTEGER, true, false);
        clp.addParam("-numRepetitions", "Number of repetitions.",
                CommandLineParser.INTEGER, true, false);
        clp.addParam("-k", "The neighborhood size.",
                CommandLineParser.INTEGER, true, false);
        clp.addParam("-outFile", "Output arff file path.",
                CommandLineParser.STRING, true, false);
        clp.addParam("-inFile", "Input arff data file path.",
                CommandLineParser.STRING, true, false);
        clp.parseLine(args);
        HubnessRiskEstimatorFromARFF experimenter =
                new HubnessRiskEstimatorFromARFF();
        experimenter.outFile = new File(
                (String)(clp.getParamValues("-outFile").get(0)));
        experimenter.k = (Integer) clp.getParamValues("-k").get(0);
        experimenter.sampleSize = (Integer) clp.getParamValues(
                "-sampleSize").get(0);
        int testSize = (Integer) clp.getParamValues("-sampleSize").get(0);
        experimenter.numRepetitions = (Integer) clp.getParamValues(
                "-numRepetitions").get(0);
        File inFile = new File((String)(clp.getParamValues("-inFile").get(0)));
        DataSet dsetAll = SupervisedLoader.loadData(inFile, false);
        dsetAll.normalizeFloats();
        boolean classRepresentationCondition;
        do {
            int[] testIndexes = UniformSampler.getSample(dsetAll.size(),
                    testSize);
            DataSet dsetTrain = dsetAll.cloneDefinition();
            DataSet dsetTest = dsetAll.cloneDefinition();
            dsetTrain.data = new ArrayList<>(dsetAll.size());
            dsetTest.data = new ArrayList<>(testIndexes.length);
            HashMap<Integer, Integer> testMap =
                    new HashMap<>(testIndexes.length);
            for (int i = 0; i < testIndexes.length; i++) {
                testMap.put(testIndexes[i], i);
                DataInstance instance =
                        dsetAll.getInstance(testIndexes[i]).copy();
                instance.embedInDataset(dsetTest);
                dsetTest.addDataInstance(instance);
            }
            experimenter.dsetTest = dsetTest;
            for (int i = 0; i < dsetAll.size(); i++) {
                if (!testMap.containsKey(i)) {
                    DataInstance instance = dsetAll.getInstance(i).copy();
                    instance.embedInDataset(dsetTrain);
                    dsetTrain.addDataInstance(instance);
                }
            }
            dsetTrain.orderInstancesByClasses();
            experimenter.dsetTrain = dsetTrain;
            classRepresentationCondition =
                    (dsetTrain.countCategories() == dsetAll.countCategories())
                    && (dsetTest.countCategories() ==
                    dsetAll.countCategories());
        } while (!classRepresentationCondition);
        experimenter.performAllTests();
    }
    
}
