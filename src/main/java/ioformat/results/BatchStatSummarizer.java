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
package ioformat.results;

import data.representation.util.DataMineConstants;
import ioformat.FileUtil;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import statistics.tests.TTests;
import util.AuxSort;
import util.BasicMathUtil;
import util.CommandLineParser;
import util.fileFilters.DirectoryFilter;

/**
 * This class summarizes the results produced by BatchClassifierTesterFast. The
 * testing framework outputs the per-fold and total scores for each algorithm
 * and each neighborhood size (in case of kNN methods) separately. This class
 * contains the methods for extracting that information into a single file for
 * cross-algorithm comparisons.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class BatchStatSummarizer {

    private File inDir;
    private File outDir;
    // The results for each dataset are contained in a separate directory.
    private File[] outDataDirList;
    // The results for different mislabeling and noise levels are also contained
    // in separate subdirectories. These arrays hold all the used mislabeling
    // and noise levels.
    private HashMap<String, Integer> mlNameToIndexMap;
    private HashMap<String, Integer> noiseNameToIndexMap;
    private HashMap<String, Integer> algNameToIndexMap;
    private HashMap<String, Integer> kNameToIndexMap;
    private ArrayList<String> currMLArray;
    private ArrayList<String> currNOArray;
    // The names of classifiers in the comparisons.
    ArrayList<String> classifierNames = null;
    private ArrayList<String> kVals;
    // This array holds the used neighborhood sizes.
    private ArrayList<Integer> kIntVals;
    // Indexes that hold the currently examined experimental setup during the
    // result traversal.
    private int currMLIndex = -1;
    private int currNOIndex = -1;
    private int currALGIndex = -1;
    private int currKVal = -1;
    // Structures that hold the extracted indicator and performance values. Mean
    // values are kept, as well as standard deviations.
    float[][][][][] completeAccuracyResults;
    float[][][][] accMatrices = null;
    float[][][][] precMatrices = null;
    float[][][][] recallMatrices = null;
    float[][][][] fMatrices = null;
    float[][][][] accMatricesStDev = null;
    float[][][][] precMatricesStDev = null;
    float[][][][] recallMatricesStDev = null;
    float[][][][] fMatricesStDev = null;
    // Parameters of the underlying cross-validation protocol.
    private int numTimes = 10;
    private int numFolds = 10;

    /**
     * Initialization.
     *
     * @param inDir Directory that holds the test results produced by
     * BatchClassifierTesterFast.
     * @param outDir Directory that will hold the summaries of the test results.
     */
    public BatchStatSummarizer(File inDir, File outDir) {
        this.inDir = inDir;
        this.outDir = outDir;
    }
    
    /**
     * Initialization.
     *
     * @param inDir Directory that holds the test results produced by
     * BatchClassifierTesterFast.
     * @param outDir Directory that will hold the summaries of the test results.
     * @param numTimes Integer representing the number of repetitions in the
     * cross-validation protocol that is being summarized.
     * @param numFolds Integer representing the number of folds in the
     * cross-validation protocol that is being summarized.
     */
    public BatchStatSummarizer(File inDir, File outDir, int numTimes,
            int numFolds) {
        this.inDir = inDir;
        this.outDir = outDir;
        this.numTimes = numTimes;
        this.numFolds = numFolds;
    }

    public void summarize() throws Exception {
        // Each directory corresponds to a single dataset.
        File[] dataResultsList = inDir.listFiles(new DirectoryFilter());
        outDataDirList = new File[dataResultsList.length];
        for (int dataIndex = 0; dataIndex < dataResultsList.length;
                dataIndex++) {
            // We create a file of the same name in the output directory for the
            // summaries.
            outDataDirList[dataIndex] =
                    new File(outDir, dataResultsList[dataIndex].getName());
            FileUtil.createDirectory(outDataDirList[dataIndex]);
            // This call performs the traversal of all the results for that
            // dataset and gets all the work done.
            initialRecurseForFullClassifierList(dataResultsList[dataIndex]);
            summarizeDataset(dataResultsList[dataIndex]);
            for (int methodIndex = 0; methodIndex < classifierNames.size();
                    methodIndex++) {
                // We create a summary file for each algorithm, where we will
                // write its performance over all the k values, noise and
                // mislabeling rates, etc.
                File currAlgOutFile = new File(outDataDirList[dataIndex],
                        classifierNames.get(methodIndex) + ".txt");
                FileUtil.createFile(currAlgOutFile);
                try (PrintWriter pw = new PrintWriter(
                             new FileWriter(currAlgOutFile))) {
                    // First we write the actual mislabeling and noise rates.
                    pw.print("ML: ");
                    pw.print(currMLArray.get(0).substring(2));
                    for (int ml = 1; ml < currMLArray.size(); ml++) {
                        pw.print("," + currMLArray.get(ml).substring(2));
                    }
                    pw.println();
                    pw.print("NO: ");
                    pw.print(currNOArray.get(0).substring(5));
                    for (int no = 1; no < currNOArray.size(); no++) {
                        pw.print("," + currNOArray.get(no).substring(5));
                    }
                    pw.println();
                    pw.println();
                    for (int k = 0; k < kVals.size(); k++) {
                        pw.println("k: " + kVals.get(k).substring(1));
                        pw.println();
                        pw.println("accMat: ");
                        pw.println();
                        for (int ml = 0; ml < currMLArray.size(); ml++) {
                            pw.print(BasicMathUtil.makeADecimalCutOff(
                                    accMatrices[methodIndex][k][ml][0], 3));
                            for (int no = 1; no < currNOArray.size(); no++) {
                                pw.print("," + BasicMathUtil.makeADecimalCutOff(
                                        accMatrices[methodIndex][k][ml][no],
                                        3));
                            }
                            pw.println();
                        }
                        pw.println();

                        pw.println("precMat: ");
                        pw.println();
                        for (int ml = 0; ml < currMLArray.size(); ml++) {
                            pw.print(BasicMathUtil.makeADecimalCutOff(
                                    precMatrices[methodIndex][k][ml][0], 3));
                            for (int no = 1; no < currNOArray.size(); no++) {
                                pw.print("," + BasicMathUtil.makeADecimalCutOff(
                                        precMatrices[methodIndex][k][ml][no],
                                        3));
                            }
                            pw.println();
                        }
                        pw.println();

                        pw.println("recallMat: ");
                        pw.println();
                        for (int ml = 0; ml < currMLArray.size(); ml++) {
                            pw.print(BasicMathUtil.makeADecimalCutOff(
                                    recallMatrices[methodIndex][k][ml][0], 3));
                            for (int no = 1; no < currNOArray.size(); no++) {
                                pw.print("," + BasicMathUtil.makeADecimalCutOff(
                                        recallMatrices[methodIndex][k][ml][no],
                                        3));
                            }
                            pw.println();
                        }
                        pw.println();

                        pw.println("fMat: ");
                        pw.println();
                        for (int ml = 0; ml < currMLArray.size(); ml++) {
                            pw.print(BasicMathUtil.makeADecimalCutOff(
                                    fMatrices[methodIndex][k][ml][0], 3));
                            for (int no = 1; no < currNOArray.size(); no++) {
                                pw.print("," + BasicMathUtil.makeADecimalCutOff(
                                        fMatrices[methodIndex][k][ml][no], 3));
                            }
                            pw.println();
                        }
                        pw.println();
                    }
                } catch (Exception e) {
                    System.err.println("Error writing to: "
                            + currAlgOutFile.getPath() + " , " +
                            e.getMessage());
                }
            }
            // We also create am additional summary file that will hold the
            // overall performance of all algorithms.
            File totalNoNoiseSummary = new File(outDataDirList[dataIndex],
                    "totalSummary.txt");
            FileUtil.createFile(totalNoNoiseSummary);
            PrintWriter pw = new PrintWriter(new FileWriter(
                    totalNoNoiseSummary));
            // We sort the k-values.
            int[] reArr = AuxSort.sortIIndexedValue(kIntVals, false);
            try {
                int[] accMaxIndex = new int[classifierNames.size()];
                for (int methodIndex = 0; methodIndex < classifierNames.size();
                        methodIndex++) {
                    pw.print(classifierNames.get(methodIndex) + ": ");
                    // The maximum accuracy will be prepended to a list of
                    // accuracies over the neighborhood range, so we need to
                    // first find the maximum value.
                    float accMax = 0;

                    for (int k = 0; k < kVals.size(); k++) {
                        if (!DataMineConstants.isAcceptableFloat(
                                accMatrices[methodIndex][k][0][0])) {
                            continue;
                        }
                        if (accMatrices[methodIndex][k][0][0] > accMax) {
                            accMax = accMatrices[methodIndex][k][0][0];
                            accMaxIndex[methodIndex] = k;
                        }
                    }
                    pw.print(BasicMathUtil.makeADecimalCutOff(accMax, 3));
                    pw.print("+-");
                    pw.print(BasicMathUtil.makeADecimalCutOff(
                            accMatricesStDev[methodIndex][accMaxIndex[
                            methodIndex]][0][0], 3));
                    // Now print the remaining accuracies in the sorted order.
                    for (int k = 0; k < kVals.size(); k++) {
                        pw.print("," + BasicMathUtil.makeADecimalCutOff(
                                accMatrices[methodIndex][reArr[k]][0][0], 3));
                    }
                    pw.println();
                }
                int[][] classifierToClassifier =
                        new int[classifierNames.size()][classifierNames.size()];
                // Perform statistical significance tests.
                TTests tester = new TTests();
                for (int methodIndex = 0; methodIndex < classifierNames.size();
                        methodIndex++) {
                    for (int secondMethodIndex = methodIndex + 1;
                            secondMethodIndex < classifierNames.size();
                            secondMethodIndex++) {
                        classifierToClassifier[methodIndex][secondMethodIndex] =
                                tester.pairedTwoTailedCorrectedResampled(
                                completeAccuracyResults[methodIndex][
                                accMaxIndex[methodIndex]][0][0],
                                completeAccuracyResults[secondMethodIndex][
                                accMaxIndex[secondMethodIndex]][0][0], 0.9f,
                                0.1f);
                        classifierToClassifier[secondMethodIndex][methodIndex] =
                                classifierToClassifier[methodIndex][
                                secondMethodIndex];
                    }
                }
                // Set the proper significane levels.
                for (int methodIndex = 0; methodIndex < classifierNames.size();
                        methodIndex++) {
                    if (methodIndex != 0) {
                        switch (classifierToClassifier[methodIndex][0]) {
                            case TTests.NO_SIGNIFICANCE: {
                                pw.print("  -   ");
                                break;
                            }
                            case TTests.SIGNIFICANCE_1: {
                                pw.print(" 0.01 ");
                                break;
                            }
                            case TTests.SIGNIFICANCE_5: {
                                pw.print(" 0.5  ");
                                break;
                            }
                        }
                    }
                    for (int secondMethodIndex = 1; secondMethodIndex
                            < methodIndex; secondMethodIndex++) {
                        pw.print(",");
                        switch (classifierToClassifier[methodIndex][
                                secondMethodIndex]) {
                            case TTests.NO_SIGNIFICANCE: {
                                pw.print("  -   ");
                                break;
                            }
                            case TTests.SIGNIFICANCE_1: {
                                pw.print(" 0.01 ");
                                break;
                            }
                            case TTests.SIGNIFICANCE_5: {
                                pw.print(" 0.5  ");
                                break;
                            }
                        }
                    }
                    if (methodIndex != 0) {
                        pw.print(",  -   ");
                    } else {
                        pw.print("  -   ");
                    }
                    for (int secondMethodIndex = methodIndex + 1;
                            secondMethodIndex < classifierNames.size();
                            secondMethodIndex++) {
                        pw.print(",");
                        switch (classifierToClassifier[methodIndex][
                                secondMethodIndex]) {
                            case TTests.NO_SIGNIFICANCE: {
                                pw.print("  -   ");
                                break;
                            }
                            case TTests.SIGNIFICANCE_1: {
                                pw.print(" 0.01 ");
                                break;
                            }
                            case TTests.SIGNIFICANCE_5: {
                                pw.print(" 0.5  ");
                                break;
                            }
                        }
                    }
                    pw.println();
                }
            } catch (Exception e) {
                System.err.println(e.getMessage());
            } finally {
                pw.close();
            }
            // Reset the structures for the next iteration.
            completeAccuracyResults = null;
            accMatrices = null;
            precMatrices = null;
            recallMatrices = null;
            fMatrices = null;
            accMatricesStDev = null;
            precMatricesStDev = null;
            recallMatricesStDev = null;
            fMatricesStDev = null;
            currMLArray = null;
            currNOArray = null;
            classifierNames = null;
            currMLIndex = -1;
            currNOIndex = -1;
            currALGIndex = -1;
            kVals = null;
            kIntVals = null;
            currKVal = -1;
            mlNameToIndexMap = null;
            noiseNameToIndexMap = null;
            algNameToIndexMap = null;
            // Initiate some clean-up.
            System.gc();
        }
    }
    
    /**
     * Internal method to recurse for all classifiers used in all experiments on
     * this particular dataset.
     * 
     * @param inDSDir File that is the current dataset directory.
     */
    private void initialRecurseForFullClassifierList(File inDSDir) {
        // Examine the results for all the neighborhood sizes.
        File[] fileListKValues = inDSDir.listFiles(new DirectoryFilter());
        if (kVals == null || kNameToIndexMap == null) {
            kNameToIndexMap = new HashMap<>(fileListKValues.length);
            kVals = new ArrayList<>(fileListKValues.length);
            kIntVals = new ArrayList<>(fileListKValues.length);
        }
        for (int i = 0; i < fileListKValues.length; i++) {
            if (!kNameToIndexMap.containsKey(fileListKValues[i].getName())) {
                kNameToIndexMap.put(fileListKValues[i].getName(),
                        kVals.size());
                kVals.add(fileListKValues[i].getName());
                if (kVals.get(i).length() > 1) {
                    kIntVals.add(Integer.parseInt(kVals.get(i).substring(1)));
                } else {
                    kIntVals.add(0);
                }   
            }
        }
        for (int k = 0; k < fileListKValues.length; k++) {
            initialRecurseInKDir(fileListKValues[k]);
        }
        if (currMLArray != null) {
            Collections.sort(currMLArray);
            mlNameToIndexMap = new HashMap<>(currMLArray.size());
            for (int i = 0; i < currMLArray.size(); i++) {
                mlNameToIndexMap.put(currMLArray.get(i), i);
            }
        }
        if (currNOArray != null) {
            Collections.sort(currNOArray);
            noiseNameToIndexMap = new HashMap<>(currNOArray.size());
            for (int i = 0; i < currNOArray.size(); i++) {
                noiseNameToIndexMap.put(currNOArray.get(i), i);
            }
        }
    }
    
    /**
     * Internal method to recurse for all classifiers used in all experiments on
     * this particular dataset.
     * 
     * @param inKDir File that is the current k-value subdirectory of the data
     * directory..
     */
    private void initialRecurseInKDir(File inKDir) {
        // Examine the results for all the neighborhood sizes.
        File[] mlRatesFileList = inKDir.listFiles(new DirectoryFilter());
        if (currMLArray == null || mlNameToIndexMap == null) {
            mlNameToIndexMap = new HashMap<>(mlRatesFileList.length);
            currMLArray = new ArrayList<>(mlRatesFileList.length);
        }
        for (int i = 0; i < mlRatesFileList.length; i++) {
            if (!mlNameToIndexMap.containsKey(mlRatesFileList[i].getName())) {
                mlNameToIndexMap.put(mlRatesFileList[i].getName(),
                        currMLArray.size());
                currMLArray.add(mlRatesFileList[i].getName());
            }
        }
        for (int ml = 0; ml < mlRatesFileList.length; ml++) {
            initialRecurseInMLDir(mlRatesFileList[ml]);
        }
    }
    
    /**
     * Internal method to recurse for all classifiers used in all experiments on
     * this particular dataset.
     * 
     * @param inMLDir File that is the current mislabeling rate subdirectory of
     * the k-value subdirectory of the data directory.
     */
    private void initialRecurseInMLDir(File inMLDir) {
        // Examine the results for all the neighborhood sizes.
        File[] noiseRatesFileList = inMLDir.listFiles(new DirectoryFilter());
        if (currNOArray == null || noiseNameToIndexMap == null) {
            noiseNameToIndexMap = new HashMap<>(noiseRatesFileList.length);
            currNOArray = new ArrayList<>(noiseRatesFileList.length);
        }
        for (int i = 0; i < noiseRatesFileList.length; i++) {
            if (!noiseNameToIndexMap.containsKey(
                    noiseRatesFileList[i].getName())) {
                noiseNameToIndexMap.put(noiseRatesFileList[i].getName(),
                        currNOArray.size());
                currNOArray.add(noiseRatesFileList[i].getName());
            }
        }
        for (int n = 0; n < noiseRatesFileList.length; n++) {
            initialRecurseInNoiseDir(noiseRatesFileList[n]);
        }
    }
    
    /**
     * Internal method to recurse for all classifiers used in all experiments on
     * this particular dataset.
     * 
     * @param inNoiseDir File that is the current noise-rate subdirectory of the
     * mislabeling-rate subdirectory of the k-value subdirectory of the data
     * directory.
     */
    private void initialRecurseInNoiseDir(File inNoiseDir) {
        // Examine the results for all the neighborhood sizes.
        File[] algorithmDirList = inNoiseDir.listFiles(new DirectoryFilter());
        if (classifierNames == null || algNameToIndexMap == null) {
            classifierNames = new ArrayList<>(algorithmDirList.length);
            algNameToIndexMap = new HashMap<>(algorithmDirList.length);
        }
        for (int i = 0; i < algorithmDirList.length; i++) {
            if (!algNameToIndexMap.containsKey(
                    algorithmDirList[i].getName())) {
                algNameToIndexMap.put(algorithmDirList[i].getName(),
                        classifierNames.size());
                classifierNames.add(algorithmDirList[i].getName());
            }
        }
    }

    /**
     * This method traverses the results for a given dataset.
     *
     * @param inDSDir File that is the input data result directory.
     * @throws Exception
     */
    private void summarizeDataset(File inDSDir) throws Exception {
        // Examine the results for all the neighborhood sizes.
        File[] fileListKValues = inDSDir.listFiles(new DirectoryFilter());
        for (int k = 0; k < fileListKValues.length; k++) {
            currKVal = kNameToIndexMap.get(fileListKValues[k].getName());
            summarizeKResults(fileListKValues[k]);
        }
    }

    /**
     * This method summarizes the results for a particular neighborhood size.
     *
     * @param inKDir Directory containing the results for this particular
     * neighborhood size.
     * @throws Exception
     */
    private void summarizeKResults(File inKDir) throws Exception {
        File[] mlRatesFileList = inKDir.listFiles(new DirectoryFilter());
        for (int ml = 0; ml < mlRatesFileList.length; ml++) {
            currMLIndex = mlNameToIndexMap.get(mlRatesFileList[ml].getName());
            summarizeMislabelingRun(mlRatesFileList[ml]);
        }
    }

    /**
     * This method summarizes the results in the particular mislabeling
     * experiment run.
     *
     * @param inMLDir Directory containing the experiment results for a
     * particular mislabeling rate experiment run.
     * @throws Exception
     */
    private void summarizeMislabelingRun(File inMLDir) throws Exception {
        File[] noiseRatesFileList = inMLDir.listFiles(new DirectoryFilter());
        for (int noise = 0; noise < noiseRatesFileList.length; noise++) {
            currNOIndex = noiseNameToIndexMap.get(
                    noiseRatesFileList[noise].getName());
            summarizeNoiseRun(noiseRatesFileList[noise]);
        }
    }

    /**
     * This method traverses the results of a particular noise run.
     *
     * @param inNoiseDir Directory containing the test results for a single
     * noise rate experiment run.
     * @throws Exception
     */
    private void summarizeNoiseRun(File inNoiseDir) throws Exception {
        // Each directory here corresponds to an algorithm that was tested.
        File[] algorithmDirList = inNoiseDir.listFiles(new DirectoryFilter());
        int numAlgs = 0;
        if (classifierNames != null) {
            numAlgs = classifierNames.size();
        }
        if (completeAccuracyResults == null) {
            completeAccuracyResults = new float[numAlgs][kVals.size()][
                    currMLArray.size()][currNOArray.size()][
                    numTimes * numFolds];
            for (int algIndex = 0; algIndex < numAlgs; algIndex++) {
                for (int k = 0; k < kVals.size(); k++) {
                    for (int ml = 0; ml < currMLArray.size(); ml++) {
                        for (int noise = 0; noise < currNOArray.size();
                                noise++) {
                            for (int t = 0; t < numTimes * numFolds; t++) {
                                completeAccuracyResults[algIndex][k][ml][noise][
                                        t] = Float.NaN;
                            }
                        }
                    }
                }
            }
            accMatrices = new float[numAlgs][kVals.size()][currMLArray.size()][
                    currNOArray.size()];
            for (int algIndex = 0; algIndex < numAlgs; algIndex++) {
                for (int k = 0; k < kVals.size(); k++) {
                    for (int ml = 0; ml < currMLArray.size(); ml++) {
                        for (int noise = 0; noise < currNOArray.size();
                                noise++) {
                            accMatrices[algIndex][k][ml][noise] = Float.NaN;
                        }
                    }
                }
            }
            precMatrices = new float[numAlgs][kVals.size()][currMLArray.size()][
                    currNOArray.size()];
            for (int algIndex = 0; algIndex < numAlgs; algIndex++) {
                for (int k = 0; k < kVals.size(); k++) {
                    for (int ml = 0; ml < currMLArray.size(); ml++) {
                        for (int noise = 0; noise < currNOArray.size();
                                noise++) {
                            precMatrices[algIndex][k][ml][noise] = Float.NaN;
                        }
                    }
                }
            }
            recallMatrices = new float[numAlgs][kVals.size()][
                    currMLArray.size()][currNOArray.size()];
            for (int algIndex = 0; algIndex < numAlgs; algIndex++) {
                for (int k = 0; k < kVals.size(); k++) {
                    for (int ml = 0; ml < currMLArray.size(); ml++) {
                        for (int noise = 0; noise < currNOArray.size();
                                noise++) {
                            recallMatrices[algIndex][k][ml][noise] = Float.NaN;
                        }
                    }
                }
            }
            fMatrices = new float[numAlgs][kVals.size()][currMLArray.size()][
                    currNOArray.size()];
            for (int algIndex = 0; algIndex < numAlgs; algIndex++) {
                for (int k = 0; k < kVals.size(); k++) {
                    for (int ml = 0; ml < currMLArray.size(); ml++) {
                        for (int noise = 0; noise < currNOArray.size();
                                noise++) {
                            fMatrices[algIndex][k][ml][noise] = Float.NaN;
                        }
                    }
                }
            }
            accMatricesStDev = new float[numAlgs][kVals.size()][
                    currMLArray.size()][currNOArray.size()];
            for (int algIndex = 0; algIndex < numAlgs; algIndex++) {
                for (int k = 0; k < kVals.size(); k++) {
                    for (int ml = 0; ml < currMLArray.size(); ml++) {
                        for (int noise = 0; noise < currNOArray.size();
                                noise++) {
                            accMatricesStDev[algIndex][k][ml][noise] =
                                    Float.NaN;
                        }
                    }
                }
            }
            precMatricesStDev = new float[numAlgs][kVals.size()][
                    currMLArray.size()][currNOArray.size()];
            for (int algIndex = 0; algIndex < numAlgs; algIndex++) {
                for (int k = 0; k < kVals.size(); k++) {
                    for (int ml = 0; ml < currMLArray.size(); ml++) {
                        for (int noise = 0; noise < currNOArray.size();
                                noise++) {
                            precMatricesStDev[algIndex][k][ml][noise] =
                                    Float.NaN;
                        }
                    }
                }
            }
            recallMatricesStDev = new float[numAlgs][kVals.size()][
                    currMLArray.size()][currNOArray.size()];
            for (int algIndex = 0; algIndex < numAlgs; algIndex++) {
                for (int k = 0; k < kVals.size(); k++) {
                    for (int ml = 0; ml < currMLArray.size(); ml++) {
                        for (int noise = 0; noise < currNOArray.size();
                                noise++) {
                            recallMatricesStDev[algIndex][k][ml][noise] =
                                    Float.NaN;
                        }
                    }
                }
            }
            fMatricesStDev = new float[numAlgs][kVals.size()][
                    currMLArray.size()][currNOArray.size()];
            for (int algIndex = 0; algIndex < numAlgs; algIndex++) {
                for (int k = 0; k < kVals.size(); k++) {
                    for (int ml = 0; ml < currMLArray.size(); ml++) {
                        for (int noise = 0; noise < currNOArray.size();
                                noise++) {
                            fMatricesStDev[algIndex][k][ml][noise] =
                                    Float.NaN;
                        }
                    }
                }
            }
        }
        for (int methodIndex = 0; methodIndex < algorithmDirList.length;
                methodIndex++) {
            currALGIndex = algNameToIndexMap.get(
                    algorithmDirList[methodIndex].getName());
            summarizeALG(algorithmDirList[methodIndex]);
        }
    }

    /**
     * This method summarizes the test results for a particular algorithm.
     *
     * @param inALGDir Directory containing the test result data for a
     * particular algorithm on this experimental test run.
     * @throws Exception
     */
    private void summarizeALG(File inALGDir) throws Exception {
        File resultFile = new File(inALGDir, "avg.txt");
        BufferedReader br = new BufferedReader(new InputStreamReader(
                new FileInputStream(resultFile)));
        try {
            // The first line is skipped, as it is a header.
            br.readLine();
            String line = br.readLine();
            line = line.trim();
            String[] measures = line.split(",");
            accMatrices[currALGIndex][currKVal][currMLIndex][currNOIndex] =
                    Float.parseFloat(measures[0]);
            precMatrices[currALGIndex][currKVal][currMLIndex][currNOIndex] =
                    Float.parseFloat(measures[1]);
            recallMatrices[currALGIndex][currKVal][currMLIndex][currNOIndex] =
                    Float.parseFloat(measures[2]);
            fMatrices[currALGIndex][currKVal][currMLIndex][currNOIndex] =
                    Float.parseFloat(measures[3]);
            line = br.readLine();
            while (line != null && !line.startsWith("accuracyStDev")) {
                line = br.readLine();
            }
            if (line == null) {
                accMatricesStDev[currALGIndex][currKVal][currMLIndex][
                        currNOIndex] = 0;
                precMatricesStDev[currALGIndex][currKVal][currMLIndex][
                        currNOIndex] = 0;
                recallMatricesStDev[currALGIndex][currKVal][currMLIndex][
                        currNOIndex] = 0;
                fMatricesStDev[currALGIndex][currKVal][currMLIndex][
                        currNOIndex] = 0;
            } else {
                line = br.readLine();
                line = line.trim();
                measures = line.split(",");
                accMatricesStDev[currALGIndex][currKVal][currMLIndex][
                        currNOIndex] = Float.parseFloat(measures[0]);
                precMatricesStDev[currALGIndex][currKVal][currMLIndex][
                        currNOIndex] = Float.parseFloat(measures[1]);
                recallMatricesStDev[currALGIndex][currKVal][currMLIndex][
                        currNOIndex] = Float.parseFloat(measures[2]);
                fMatricesStDev[currALGIndex][currKVal][currMLIndex][
                        currNOIndex] = Float.parseFloat(measures[3]);
            }

        } catch (IOException | NumberFormatException e) {
            System.err.println("Error reading: " + resultFile.getPath() + ","
                    + e.getMessage());
        } finally {
            br.close();
        }
        resultFile = new File(inALGDir, "allSummed.txt");
        br = new BufferedReader(new InputStreamReader(
                new FileInputStream(resultFile)));
        String[] lineItems;
        try {
            String line = br.readLine(); // We skip the header.
            for (int i = 0; i < numTimes * numFolds; i++) {
                line = br.readLine();
                line = line.trim();
                lineItems = line.split(",");
                completeAccuracyResults[currALGIndex][currKVal][currMLIndex][
                        currNOIndex][i] = Float.parseFloat(lineItems[0]);
            }
        } catch (IOException | NumberFormatException | NullPointerException e) {
            System.err.println("Error reading: " + resultFile.getPath() + ","
                    + e.getMessage());
            if (e instanceof NullPointerException) {
                System.err.println("Probably a mismatch in the number of folds "
                        + "or repetitions in the CV protocol.");
                System.err.println("All directories being summarized at the "
                        + "same time should have the same CV protocol");
            }
        } finally {
            br.close();
        }

    }

    /**
     * Initializes the result summarization process.
     *
     * @param args Command line parameters, as specified.
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        CommandLineParser clp = new CommandLineParser(true);
        clp.addParam("-testResultsDir", "Directory of test results",
                CommandLineParser.STRING, true, false);
        clp.addParam("-outSummaryDir", "Path to where to place the summaries",
                CommandLineParser.STRING, true, false);
        clp.parseLine(args);
        File inDir = new File((String) clp.getParamValues("-testResultsDir").
                get(0));
        File outDir = new File((String) clp.getParamValues("-outSummaryDir").
                get(0));
        BatchStatSummarizer btrs = new BatchStatSummarizer(inDir, outDir);
        btrs.summarize();
    }
}