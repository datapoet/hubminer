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
package data.imbalance.util;

import data.neighbors.NeighborSetFinder;
import data.neighbors.SharedNeighborFinder;
import data.representation.DataSet;
import distances.primary.CombinedMetric;
import distances.secondary.snd.SharedNeighborCalculator;
import ioformat.FileUtil;
import ioformat.IOARFF;
import ioformat.IOCSV;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import learning.supervised.evaluation.ClassificationEstimator;
import learning.supervised.evaluation.ValidateableInterface;
import learning.supervised.evaluation.cv.MultiCrossValidation;
import learning.supervised.methods.knn.ANHBNN;
import learning.supervised.methods.knn.DWHFNN;
import learning.supervised.methods.knn.HwKNN;
import learning.supervised.methods.knn.HIKNN;
import learning.supervised.methods.knn.KNN;
import learning.supervised.methods.knn.NHBNN;
import learning.supervised.methods.knn.RRKNN;
import linear.matrix.SymmetricFloatMatrix;
import statistics.HigherMoments;
import util.AuxSort;
import util.CommandLineParser;
import util.SOPLUtil;

/**
 * A utility class for analyzing some properties of class-imbalanced data. It is
 * focused on kNN properties of the data and the interplay with hubness. It is
 * essentially one big script, not meant to be re-used in other contexts.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class ImbalancedAnalyzer {

    public enum PointTypeKNN {

        SAFE(0), BORDERLINE(1), RARE(2), OUTLIER(3);
        public int id;

        PointTypeKNN(int id) {
            this.id = id;
        }
    }

    /**
     * This script reads the data, performs various forms of kNN analysis and
     * then prints out the results to a specified directory.
     *
     * @param args
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        CommandLineParser clp = new CommandLineParser(true);
        clp.addParam("-k", "Neighborhood size.", CommandLineParser.INTEGER,
                true, false);
        clp.addParam("-inFile", "Data file path.", CommandLineParser.STRING,
                true, false);
        clp.addParam("-outDir", "Output directory for generated files.",
                CommandLineParser.STRING, true, false);
        clp.addParam("-dMat", "Path to the distance matrix file, if available",
                CommandLineParser.STRING, false, false);
        clp.addParam("-normalize", "True/false. Whether to normalize the "
                + "data attributes or not.", CommandLineParser.BOOLEAN, false,
                false);
        clp.parseLine(args);
        int k = (Integer) clp.getParamValues("-k").get(0);
        if (k < 5) {
            System.out.println("Expected neighborhood size k >= 5.");
            return;
        }
        // Initialize IO handlers.
        IOARFF pers = new IOARFF();
        IOCSV reader = new IOCSV(true, ",");
        DataSet dset;
        // Read in the data.
        String inFilePath = (String) clp.getParamValues("-inFile").get(0);
        if (inFilePath.endsWith("arff")) {
            dset = pers.load(inFilePath);
        } else {
            dset = reader.readData(new File(inFilePath));
        }
        boolean normalize = (Boolean) clp.getParamValues("-normalize").get(0);
        // Normalize the features if necessary.
        if (normalize) {
            dset.normalizeFloats();
        }
        String name = getDataFileNameNoExt(inFilePath);
        // Initialize metric object.
        CombinedMetric cmet = CombinedMetric.FLOAT_MANHATTAN;
        NeighborSetFinder nsf = new NeighborSetFinder(dset, cmet);
        if (!clp.hasParamValue("-dMat")) {
            // Matrix file not provided, calculate distances now.
            nsf.calculateDistances();
        } else {
            // Load the matrix from a file.
            SymmetricFloatMatrix sfm = new SymmetricFloatMatrix();
            nsf.setDistances(sfm.loadDMatFromFileNoDiag(new File(
                    (String) clp.getParamValues("-dMat").get(0))));
        }
        float[][] distMat = nsf.getDistances();
        nsf.calculateNeighborSets(k);
        // Get hubness of different points.
        float[] hubness = nsf.getFloatOccFreqs();
        int[] badHubness = nsf.getBadFrequencies();
        // Get class priors.
        float[] classPriors = dset.getClassPriors();
        // Here we detect hubs as points that occur more than k + 2 * stDev of
        // neighbor occurrences and we use the same procedure for finding bad
        // hubs. We also take the same number of anti-hubs as hubs for analysis.
        float meanHubness = HigherMoments.calculateArrayMean(hubness);
        float stDevHubness = HigherMoments.calculateArrayStDev(meanHubness,
                hubness);
        float meanBadHubness = HigherMoments.calculateArrayMean(badHubness);
        float stDevBadHubness =
                HigherMoments.calculateArrayStDev(meanBadHubness, badHubness);
        ArrayList<Integer> hubs = new ArrayList<>(1000);
        ArrayList<Integer> badhubs = new ArrayList<>(1000);
        ArrayList<Integer> antiHubs = new ArrayList<>(1000);
        // The maps are there to quickly determine which points are hubs in
        // subsequent analysis.
        HashMap<Integer, Integer> badHubHash = new HashMap<>(1000, 1000);
        HashMap<Integer, Integer> hubHash = new HashMap<>(1000, 1000);
        int numCat = classPriors.length;
        // Count the frequencies of different point types in different classes.
        float[] hubClassCounts = new float[numCat];
        float[] badhubClassCounts = new float[numCat];
        float[] avgHubClassSize = new float[numCat];
        for (int i = 0; i < dset.size(); i++) {
            if (hubness[i] >= meanHubness + 2 * stDevHubness) {
                hubs.add(i);
                hubHash.put(i, dset.getLabelOf(i));
                hubClassCounts[dset.getLabelOf(i)]++;
                if (nsf.getReverseNeighbors()[i] != null) {
                    avgHubClassSize[dset.getLabelOf(i)] +=
                            nsf.getReverseNeighbors()[i].size();
                }
            }
            if (badHubness[i] > meanBadHubness + 2 * stDevBadHubness) {
                badhubs.add(i);
                badhubClassCounts[dset.getLabelOf(i)]++;
                badHubHash.put(i, dset.getLabelOf(i));
            }
        }
        float[] hubnessArrayCopy = Arrays.copyOf(hubness, hubness.length);
        // Ascending sort, first instances will be antihubs.
        int[] indexes = AuxSort.sortIndexedValue(hubnessArrayCopy, false);
        for (int i = 0; i < hubs.size(); i++) {
            antiHubs.add(indexes[i]);
        }
        float[] classDistrOverHubs = new float[numCat];
        float[] classDistrOverBadHubs = new float[numCat];
        float[] classDistrOverAntihubs = new float[numCat];
        for (int i = 0; i < hubs.size(); i++) {
            classDistrOverHubs[dset.data.get(hubs.get(i)).getCategory()]++;
            classDistrOverAntihubs[
                    dset.data.get(antiHubs.get(i)).getCategory()]++;
        }
        for (int i = 0; i < badhubs.size(); i++) {
            classDistrOverBadHubs[
                    dset.data.get(badhubs.get(i)).getCategory()]++;
        }
        // Normalization of class distributions.
        for (int i = 0; i < numCat; i++) {
            classDistrOverHubs[i] /= (float) hubs.size();
            classDistrOverAntihubs[i] /= (float) hubs.size();
            classDistrOverBadHubs[i] /= (float) badhubs.size();
            if (hubClassCounts[i] >= 1) {
                avgHubClassSize[i] /= hubClassCounts[i];
            }
        }
        // Examine possible applications of shared neighbor distances.
        NeighborSetFinder nsfKBig = new NeighborSetFinder(dset, cmet);
        nsfKBig.calculateDistances();
        // Calculate kNN sets for k = 50.
        nsfKBig.calculateNeighborSetsMultiThr(50, 6);
        // Simcos-based shared neighbor finder.
        SharedNeighborFinder snfSimcos = new SharedNeighborFinder(nsfKBig);
        snfSimcos.countSharedNeighbors();
        // Simhub-based shared neighbor finder.
        SharedNeighborFinder snfSimhub = new SharedNeighborFinder(nsfKBig);
        snfSimhub.obtainWeightsFromHubnessInformation(0);
        snfSimhub.countSharedNeighbors();
        // Simcos-based shared neighbor calculator.
        SharedNeighborCalculator sncSimcos = new SharedNeighborCalculator(
                snfSimcos, SharedNeighborCalculator.WeightingType.NONE);
        // Simcosub-based shared neighbor calculator.
        SharedNeighborCalculator sncSimhub = new SharedNeighborCalculator(
                snfSimcos,
                SharedNeighborCalculator.WeightingType.HUBNESS_INFORMATION);
        // Calculate the distance matrices.
        float[][] simMatSimcos = snfSimcos.getSharedNeighborCounts();
        float[][] dMatSimcos = new float[simMatSimcos.length][];
        for (int i = 0; i < dMatSimcos.length; i++) {
            dMatSimcos[i] = new float[simMatSimcos[i].length];
            for (int j = 0; j < dMatSimcos[i].length; j++) {
                dMatSimcos[i][j] = 50 - simMatSimcos[i][j];
            }
        }
        float[][] simMatSimhub = snfSimhub.getSharedNeighborCounts();
        float[][] dMatSimhub = new float[simMatSimhub.length][];
        for (int i = 0; i < dMatSimhub.length; i++) {
            dMatSimhub[i] = new float[simMatSimhub[i].length];
            for (int j = 0; j < dMatSimhub[i].length; j++) {
                dMatSimhub[i][j] = 50 - simMatSimhub[i][j];
            }
        }
        // Calculate kNN sets based on shared-neighbor distances.
        NeighborSetFinder nsfSimcos = new NeighborSetFinder(
                dset, dMatSimcos, sncSimcos);
        nsfSimcos.calculateNeighborSetsMultiThr(5, 6);
        NeighborSetFinder nsfSimhub = new NeighborSetFinder(
                dset, dMatSimhub, sncSimhub);
        nsfSimhub.calculateNeighborSetsMultiThr(5, 6);
        // Determine direct kNN-based point types: safe, borderline, rare and
        // outlier points. Calculate average hubness and bad hubness for each
        // type of points within each class.
        int[] numLabelMatch = new int[dset.size()];
        int[] numLabelMatchSimcos = new int[dset.size()];
        int[] numLabelMatchSimhub = new int[dset.size()];
        float[][] pTypeClassDistr = new float[numCat][4];
        float[][] pTypeClassDistrSimcos = new float[numCat][4];
        float[][] pTypeClassDistrSimhub = new float[numCat][4];
        // Used for normalization
        float[] pTypeNums = new float[4];
        float[] pTypeNumsSimcos = new float[4];
        float[] pTypeNumsSimhub = new float[4];
        float[] pTypeHubness = new float[4];
        float[] pTypeBadHubness = new float[4];
        float[][] pTypeClassAVGHubness = new float[numCat][4];
        float[][] pTypeClassAVGBHubness = new float[numCat][4];
        // Point type array for evaluation of point classification precision.
        PointTypeKNN[] pTypesPrimary = new PointTypeKNN[dset.size()];
        int currCat;
        for (int i = 0; i < dset.size(); i++) {
            currCat = dset.data.get(i).getCategory();
            for (int j = 0; j < 5; j++) {
                // Only the first 5 neighbors are taken into account to
                // determine the point type.
                if (dset.getLabelOf(nsf.getKNeighbors()[i][j]) == currCat) {
                    numLabelMatch[i]++;
                }
                if (dset.getLabelOf(
                        nsfSimcos.getKNeighbors()[i][j]) == currCat) {
                    numLabelMatchSimcos[i]++;
                }
                if (dset.getLabelOf(
                        nsfSimhub.getKNeighbors()[i][j]) == currCat) {
                    numLabelMatchSimhub[i]++;
                }
            }
            // Incrementally update the distributions.
            if (numLabelMatch[i] >= 4) {
                // Safe point.
                pTypeNums[0]++;
                pTypeClassDistr[currCat][0]++;
                pTypeHubness[0] += hubness[i];
                pTypeBadHubness[0] += badHubness[i];
                pTypeClassAVGHubness[currCat][0] += hubness[i];
                pTypeClassAVGBHubness[currCat][0] += badHubness[i];
                pTypesPrimary[i] = PointTypeKNN.SAFE;
            } else if (numLabelMatch[i] >= 2) {
                // Borderline.
                pTypeNums[1]++;
                pTypeClassDistr[currCat][1]++;
                pTypeHubness[1] += hubness[i];
                pTypeBadHubness[1] += badHubness[i];
                pTypeClassAVGHubness[currCat][1] += hubness[i];
                pTypeClassAVGBHubness[currCat][1] += badHubness[i];
                pTypesPrimary[i] = PointTypeKNN.BORDERLINE;
            } else if (numLabelMatch[i] == 1) {
                // Rare.
                pTypeNums[2]++;
                pTypeClassDistr[currCat][2]++;
                pTypeHubness[2] += hubness[i];
                pTypeBadHubness[2] += badHubness[i];
                pTypeClassAVGHubness[currCat][2] += hubness[i];
                pTypeClassAVGBHubness[currCat][2] += badHubness[i];
                pTypesPrimary[i] = PointTypeKNN.RARE;
            } else if (numLabelMatch[i] == 0) {
                // Outlier.
                pTypeNums[3]++;
                pTypeClassDistr[currCat][3]++;
                pTypeHubness[3] += hubness[i];
                pTypeBadHubness[3] += badHubness[i];
                pTypeClassAVGHubness[currCat][3] += hubness[i];
                pTypeClassAVGBHubness[currCat][3] += badHubness[i];
                pTypesPrimary[i] = PointTypeKNN.OUTLIER;
            }
            // Simcos case.
            if (numLabelMatchSimcos[i] >= 4) {
                // Safe point.
                pTypeNumsSimcos[0]++;
                pTypeClassDistrSimcos[currCat][0]++;
            } else if (numLabelMatchSimcos[i] >= 2) {
                // Borderline.
                pTypeNumsSimcos[1]++;
                pTypeClassDistrSimcos[currCat][1]++;
            } else if (numLabelMatchSimcos[i] == 1) {
                // Rare.
                pTypeNumsSimcos[2]++;
                pTypeClassDistrSimcos[currCat][2]++;
            } else if (numLabelMatchSimcos[i] == 0) {
                // Outlier.
                pTypeNumsSimcos[3]++;
                pTypeClassDistrSimcos[currCat][3]++;
            }
            // Simhub case.
            if (numLabelMatchSimhub[i] >= 4) {
                // Safe point.
                pTypeNumsSimhub[0]++;
                pTypeClassDistrSimhub[currCat][0]++;
            } else if (numLabelMatchSimhub[i] >= 2) {
                // Borderline.
                pTypeNumsSimhub[1]++;
                pTypeClassDistrSimhub[currCat][1]++;
            } else if (numLabelMatchSimhub[i] == 1) {
                // Rare.
                pTypeNumsSimhub[2]++;
                pTypeClassDistrSimhub[currCat][2]++;
            } else if (numLabelMatchSimhub[i] == 0) {
                // Outlier.
                pTypeNumsSimhub[3]++;
                pTypeClassDistrSimhub[currCat][3]++;
            }
        }

        for (int j = 0; j < 4; j++) {
            pTypeHubness[j] /= pTypeNums[j];
            pTypeBadHubness[j] /= pTypeNums[j];
        }
        float sum;
        float sumSimcos;
        float sumSimhub;
        for (int i = 0; i < numCat; i++) {
            for (int j = 0; j < 4; j++) {
                if (pTypeClassDistr[i][j] == 0) {
                    pTypeClassAVGHubness[i][j] = 0;
                    continue;
                }
                pTypeClassAVGHubness[i][j] /= pTypeClassDistr[i][j];
                pTypeClassAVGBHubness[i][j] /= pTypeClassDistr[i][j];
            }
            sum = pTypeClassDistr[i][0] + pTypeClassDistr[i][1]
                    + pTypeClassDistr[i][2] + pTypeClassDistr[i][3];
            sumSimcos = pTypeClassDistrSimcos[i][0]
                    + pTypeClassDistrSimcos[i][1] + pTypeClassDistrSimcos[i][2]
                    + pTypeClassDistrSimcos[i][3];
            sumSimhub = pTypeClassDistrSimhub[i][0]
                    + pTypeClassDistrSimhub[i][1] + pTypeClassDistrSimhub[i][2]
                    + pTypeClassDistrSimhub[i][3];
            for (int j = 0; j < 4; j++) {
                pTypeClassDistr[i][j] /= sum;
                pTypeClassDistrSimcos[i][j] /= sumSimcos;
                pTypeClassDistrSimhub[i][j] /= sumSimhub;
            }
        }
        // Testing the classifier performance on different point types.
        ValidateableInterface[] nonDiscreteArray = new ValidateableInterface[7];
        nonDiscreteArray[0] = new KNN(k, cmet);
        nonDiscreteArray[1] = new HwKNN(numCat, cmet, k);
        nonDiscreteArray[2] = new DWHFNN(k, cmet, numCat);
        nonDiscreteArray[3] = new NHBNN(k, cmet, numCat);
        nonDiscreteArray[4] = new HIKNN(k, cmet, numCat);
        nonDiscreteArray[5] = new ANHBNN(k, cmet, numCat);
        nonDiscreteArray[6] = new RRKNN(k, cmet, numCat);
        String[] algNames = {"KNN", "hw-kNN", "h-FNN", "NBHNN", "HIKNN",
            "ANHBNN", "RRKNN"};
        MultiCrossValidation nonDiscreteCV =
                new MultiCrossValidation(10, 10, numCat, dset, dset.data,
                nonDiscreteArray, distMat);
        nonDiscreteCV.setKValue(k);
        nonDiscreteCV.setCombinedMetric(cmet);
        int[] originalLabels = dset.obtainLabelArray();
        nonDiscreteCV.validateOnSeparateLabelArray(originalLabels);
        nonDiscreteCV.performAllTests();
        ClassificationEstimator[] avgNonDiscrete =
                nonDiscreteCV.getAverageResults();
        float[][] pointSuccess =
                nonDiscreteCV.getPerPointClassificationPrecision();
        // Improvements over kNN.
        float[][] algClassConditionalBadHubsImprovementsOverKNN =
                new float[nonDiscreteArray.length - 1][numCat];
        float[][] algClassConditionalBadHubsWorseThanKNN =
                new float[nonDiscreteArray.length - 1][numCat];
        float[][] algClassConditionalHubsImprovementsOverKNN =
                new float[nonDiscreteArray.length - 1][numCat];
        float[][] algClassConditionalHubsWorseThanKNN =
                new float[nonDiscreteArray.length - 1][numCat];
        // Precision on different types of points.
        float[][] pointTypeSuccess = new float[nonDiscreteArray.length][4];
        for (int i = 0; i < nonDiscreteArray.length; i++) {
            for (int j = 0; j < dset.size(); j++) {
                pointTypeSuccess[i][pTypesPrimary[j].id] += pointSuccess[i][j];
                if (i > 0) {
                    // If not the kNN baseline, we measure improvement over kNN. 
                    if (pointSuccess[i][j] > pointSuccess[0][j]) {
                        for (int kTmp = 0; kTmp < k; kTmp++) {
                            if (badHubHash.containsKey(
                                    nsf.getKNeighbors()[j][kTmp])) {
                                algClassConditionalBadHubsImprovementsOverKNN[
                                        i - 1][badHubHash.get(
                                        nsf.getKNeighbors()[j][kTmp])]++;
                            }
                            if (hubHash.containsKey(
                                    nsf.getKNeighbors()[j][kTmp])) {
                                algClassConditionalHubsImprovementsOverKNN[
                                        i - 1][hubHash.get(
                                        nsf.getKNeighbors()[j][kTmp])]++;
                            }
                        }
                    } else if (pointSuccess[i][j] < pointSuccess[0][j]) {
                        for (int kTmp = 0; kTmp < k; kTmp++) {
                            if (badHubHash.containsKey(
                                    nsf.getKNeighbors()[j][kTmp])) {
                                algClassConditionalBadHubsWorseThanKNN[
                                        i - 1][badHubHash.get(
                                        nsf.getKNeighbors()[j][kTmp])]++;
                            }
                            if (hubHash.containsKey(
                                    nsf.getKNeighbors()[j][kTmp])) {
                                algClassConditionalHubsWorseThanKNN[
                                        i - 1][hubHash.get(
                                        nsf.getKNeighbors()[j][kTmp])]++;
                            }
                        }
                    }
                }
            }
        }
        // Normalize the results.
        for (int i = 0; i < nonDiscreteArray.length; i++) {
            System.out.println(algNames[i] + " acc "
                    + avgNonDiscrete[i].getAccuracy());
            for (int j = 0; j < 4; j++) {
                // As we were performing 10-times 10-fold cross-validation.
                pointTypeSuccess[i][j] /= (pTypeNums[j] * 10);
            }
        }
        float tTotalSimhub = 0;
        float tTotalSimcos = 0;
        for (int i = 0; i < 4; i++) {
            tTotalSimhub += pTypeNumsSimhub[i];
            tTotalSimcos += pTypeNumsSimcos[i];
        }
        for (int i = 0; i < 4; i++) {
            pTypeNumsSimhub[i] /= tTotalSimhub;
            pTypeNumsSimcos[i] /= tTotalSimcos;
            pTypeNums[i] /= tTotalSimcos;
        }

        // Persist the results.
        File outDir = new File((String) clp.getParamValues("-outDir").get(0));
        File outFile = new File(outDir, name + ".txt");
        System.out.println("Writing output to: " + outFile.getPath());
        FileUtil.createFile(outFile);
        PrintWriter pw = new PrintWriter(new FileWriter(outFile));
        try {
            pw.println("Number of instances " + dset.size());
            pw.println("Class priors");
            SOPLUtil.printArrayToStream(classPriors, pw, ",");
            pw.println("Used k " + k);
            pw.println("Hubs class distribution");
            SOPLUtil.printArrayToStream(classDistrOverHubs, pw, ",");
            pw.println("Bad hubs class distribution");
            SOPLUtil.printArrayToStream(classDistrOverBadHubs, pw, ",");
            pw.println("Anti-hubs class distribution");
            SOPLUtil.printArrayToStream(classDistrOverAntihubs, pw, ",");
            pw.println("Different types of points: 0-safe, 1-borderline, "
                    + "2-rare. 3-outlier");
            pw.println("Distribution among the classes");
            for (int i = 0; i < numCat; i++) {
                pw.println("class " + i);
                SOPLUtil.printArrayToStream(pTypeClassDistr[i], pw, ",");
            }
            pw.println("Distribution among the classes when using simcos");
            for (int i = 0; i < numCat; i++) {
                pw.println("class " + i);
                SOPLUtil.printArrayToStream(pTypeClassDistrSimcos[i], pw, ",");
            }
            pw.println("AVG distribution among the classes for simcos");
            SOPLUtil.printArrayToStream(pTypeNumsSimcos, pw, ",");
            pw.println("Distribution among the classes when using simhub");
            for (int i = 0; i < numCat; i++) {
                pw.println("class " + i);
                SOPLUtil.printArrayToStream(pTypeClassDistrSimhub[i], pw, ",");
            }
            pw.println("AVG distribution among the classes for simhub");
            SOPLUtil.printArrayToStream(pTypeNumsSimhub, pw, ",");
            pw.println("Distribution of hubness among point types in different "
                    + "classes");
            for (int i = 0; i < numCat; i++) {
                pw.println("class " + i);
                SOPLUtil.printArrayToStream(pTypeClassAVGHubness[i], pw, ",");
            }
            pw.println("Distribution of bad hubness among point types in "
                    + "different classes");
            for (int i = 0; i < numCat; i++) {
                pw.println("class " + i);
                SOPLUtil.printArrayToStream(pTypeClassAVGBHubness[i], pw, ",");
            }
            pw.println("overall distribution of point types");
            SOPLUtil.printArrayToStream(pTypeNums, pw, ",");
            pw.println("overall distribution of hubness in point types");
            SOPLUtil.printArrayToStream(pTypeHubness, pw, ",");
            pw.println("overall distribution of bad hubness in point types");
            SOPLUtil.printArrayToStream(pTypeBadHubness, pw, ",");
            for (int i = 0; i < nonDiscreteArray.length; i++) {
                pw.print(algNames[i] + " ");
                for (int j = 0; j < 4; j++) {
                    pw.print(pointTypeSuccess[i][j] + " ");
                }
                pw.println();
            }
            pw.println();
            pw.println("Improvement over kNN, by improving classification of "
                    + "points where bad hubs of particular classes appear:");
            for (int i = 0; i < nonDiscreteArray.length - 1; i++) {
                pw.println(algNames[i + 1]);
                SOPLUtil.printArrayToStream(
                        algClassConditionalBadHubsImprovementsOverKNN[i],
                        pw, ",");
            }
            pw.println();
            pw.println("Worse than kNN, by reducing classification of points"
                    + " where bad hubs of particular classes appear:");
            for (int i = 0; i < nonDiscreteArray.length - 1; i++) {
                pw.println(algNames[i + 1]);
                SOPLUtil.printArrayToStream(
                        algClassConditionalBadHubsWorseThanKNN[i], pw, ",");
            }
            pw.println();
            pw.println("Improvement over kNN, by improving classification of"
                    + " points where bad hubs of particular classes appear");
            pw.println("(improvements - worse)/numBadHubsinClass");
            for (int i = 0; i < nonDiscreteArray.length - 1; i++) {
                pw.println(algNames[i + 1]);
                for (int j = 0; j < numCat; j++) {
                    pw.print(" "
                            + (algClassConditionalBadHubsImprovementsOverKNN[i][j]
                            - algClassConditionalBadHubsWorseThanKNN[i][j])
                            / ((float) badhubClassCounts[j] + 1) + " ");
                }
                pw.println();
            }
            pw.println();
            pw.println("Improvement over kNN, by improving classification of "
                    + "points where hubs of particular classes appear");
            pw.println("(improvements - worse)/numHubsinClass");
            for (int i = 0; i < nonDiscreteArray.length - 1; i++) {
                pw.println(algNames[i + 1]);
                for (int j = 0; j < numCat; j++) {
                    pw.print(" " + (
                            algClassConditionalHubsImprovementsOverKNN[i][j]
                            - algClassConditionalHubsWorseThanKNN[i][j])
                            / ((float) hubClassCounts[j] + 1) + " ");
                }
                pw.println();
            }
            pw.println();
            pw.println("Improvement over kNN, by improving classification of "
                    + "points where hubs of particular classes appear");
            pw.println("(improvements - worse)/(numHubsinClass*"
                    + "avgHubSizeinCLass)");
            for (int i = 0; i < nonDiscreteArray.length - 1; i++) {
                pw.println(algNames[i + 1]);
                for (int j = 0; j < numCat; j++) {
                    pw.print(" " + (
                            algClassConditionalHubsImprovementsOverKNN[i][j]
                            - algClassConditionalHubsWorseThanKNN[i][j])
                            / (((float) hubClassCounts[j] + 1)
                            * (avgHubClassSize[j] + 1)) + " ");
                }
                pw.println();
            }
            pw.println();
        } catch (Exception e) {
            System.err.println(e.getMessage());
            throw e;
        } finally {
            pw.close();
        }
    }

    /**
     * Get the data file name without extension.
     *
     * @param filePath String that is the file path.
     * @return String that is the file name with extension removed.
     */
    public static String getDataFileNameNoExt(String filePath) {
        File tmp = new File(filePath);
        String fileNameWithExtension = tmp.getName();
        if (fileNameWithExtension.endsWith(".csv")
                || fileNameWithExtension.endsWith(".tsv")) {
            return fileNameWithExtension.substring(0,
                    fileNameWithExtension.length() - 4);
        } else if (fileNameWithExtension.endsWith(".arff")) {
            return fileNameWithExtension.substring(0,
                    fileNameWithExtension.length() - 5);
        } else {
            return "Unknown";
        }
    }
}
