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
package preprocessing.instance_selection.util;

import data.neighbors.NeighborSetFinder;
import data.representation.DataInstance;
import data.representation.DataSet;
import distances.primary.CombinedMetric;
import feature.correlation.PearsonCorrelation;
import ioformat.DistanceMatrixIO;
import ioformat.FileUtil;
import ioformat.IOARFF;
import ioformat.IOCSV;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Random;
import preprocessing.instance_selection.CNN;
import preprocessing.instance_selection.GCNN;
import preprocessing.instance_selection.INSIGHT;
import preprocessing.instance_selection.IPT_RT3;
import preprocessing.instance_selection.InstanceSelector;
import preprocessing.instance_selection.RNNR_AL1;
import preprocessing.instance_selection.RandomSelector;
import preprocessing.instance_selection.Wilson72;
import statistics.HigherMoments;
import util.ArrayUtil;
import util.AuxSort;
import util.CommandLineParser;

/**
 * This class compares different instance selection methods on a dataset. The
 * reduction is either set to automatically determined subset size or to 10%
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class ReducersHubAnalyzer {

    public static final int NUM_METHODS = 9;

    /**
     * @param k Neighborhood size.
     * @param filePath Path to the input file.
     * @param dMatPath Path to the data matrix.
     * @param outDir Path to output directory.
     * @param normalize Whether to perform normalization or not.
     * @throws Exception
     */
    public static void compareMethods(
            int k,
            String filePath,
            String dMatPath,
            File outDir,
            boolean normalize) throws Exception {
        IOARFF pers = new IOARFF();
        IOCSV reader = new IOCSV(true, ",");
        DataSet dset;
        File dMatFile = new File(dMatPath);
        float[][] distMat = null;
        boolean distancesLoaded = false;
        if (dMatFile.exists()) {
            distMat = loadDMatFromFile(dMatFile);
            distancesLoaded = true;
        }
        if (!filePath.equals("null")) {
            // The filePath is "null", if no data exists, only the distance
            // matrix.
            if (filePath.endsWith("arff")) {
                dset = pers.load(filePath);
            } else {
                dset = reader.readData(new File(filePath));
            }
        } else {
            // Generate dummy data.
            dset = new DataSet();
            Random randa = new Random();
            dset.fAttrNames = new String[1];
            dset.fAttrNames[0] = "a0";
            // Here we know we must have the distance matrix, so it is not null.
            dset.data = new ArrayList(distMat.length);
            for (int i = 0; i < distMat.length; i++) {
                DataInstance instance = new DataInstance(dset);
                instance.fAttr[0] = randa.nextFloat();
                dset.addDataInstance(instance);
                instance.embedInDataset(dset);
            }
        }
        if (normalize) {
            dset.normalizeFloats();
        }
        CombinedMetric cmet = CombinedMetric.FLOAT_EUCLIDEAN;
        NeighborSetFinder nsf = new NeighborSetFinder(dset, cmet);
        if (distMat == null) {
            distMat = dset.calculateDistMatrixMultThr(cmet, 4);
            nsf.setDistances(distMat);
        } else {
            nsf.setDistances(distMat);
        }
        distMat = nsf.getDistances();
        if (!distancesLoaded) {
            try {
                DistanceMatrixIO.printDMatToFile(distMat, dMatFile);
            } catch (Exception e) {
                System.err.println(e.getMessage());
            }
        }
        nsf.calculateNeighborSets(k);
        // Make an array of reducer objects and reduce by all of them
        // separately. They do not modify the original DataSet, so this sequence
        // is possible.
        InstanceSelector[] methods = new InstanceSelector[NUM_METHODS];
        String[] methodNames = new String[NUM_METHODS];
        methods[0] = new RandomSelector(dset);
        methodNames[0] = "random";
        System.out.println("calculating: " + methodNames[0]);
        methods[0].reduceDataSet(0.1f);
        methods[1] = new INSIGHT(nsf, INSIGHT.XI);
        methodNames[1] = "INSIGHT.XI";
        System.out.println("calculating: " + methodNames[1]);
        methods[1].reduceDataSet(0.1f);
        methods[2] = new INSIGHT(nsf, INSIGHT.GOOD_HUBNESS);
        methodNames[2] = "INSIGHT.GH";
        System.out.println("calculating: " + methodNames[2]);
        methods[2].reduceDataSet(0.1f);
        methods[3] = new Wilson72(nsf, cmet);
        methodNames[3] = "Wilson72";
        System.out.println("calculating: " + methodNames[3]);
        methods[3].reduceDataSet();
        methods[4] = new IPT_RT3(nsf, cmet);
        methodNames[4] = "IPT_RT3";
        System.out.println("calculating: " + methodNames[4]);
        methods[4].reduceDataSet();
        methods[5] = new RNNR_AL1(nsf, cmet);
        methodNames[5] = "RNNR_AL1";
        System.out.println("calculating: " + methodNames[5]);
        methods[5].reduceDataSet();
        methods[6] = new GCNN(nsf);
        methodNames[6] = "GCNN_0.99";
        System.out.println("calculating: " + methodNames[6]);
        methods[6].reduceDataSet();
        methods[7] = new GCNN(nsf, .1f);
        methodNames[7] = "GCNN_0.1";
        System.out.println("calculating: " + methodNames[7]);
        methods[7].reduceDataSet();
        methods[8] = new CNN(nsf);
        methodNames[8] = "CNN";
        System.out.println("calculating: " + methodNames[8]);
        methods[8].reduceDataSet();
        // First get some stats of the initial data.
        int datasize = dset.size();
        int numClasses = dset.countCategories();
        double totalBadHubness = ArrayUtil.sum(nsf.getBadFrequencies());
        double totalOccurrences = datasize * k;
        double bhPerc = totalBadHubness / totalOccurrences;
        int[] hubnessArray = nsf.getNeighborFrequencies();
        HashMap hubHash = new HashMap(1000);
        ArrayList<Integer> hubs = new ArrayList<>(500);
        ArrayList<Integer> antiHubs = new ArrayList<>(500);
        ArrayList<Integer> regulars = new ArrayList<>(5000);
        float med = HigherMoments.calculateArrayMean(hubnessArray);
        float stDev = HigherMoments.calculateArrayStDev(med, hubnessArray);
        float up = med + 2 * stDev;
        double hubness = HigherMoments.calculateSkewForSampleArray(
                hubnessArray);
        for (int i = 0; i < hubnessArray.length; i++) {
            if (hubnessArray[i] > up) {
                hubs.add(i);
                hubHash.put(i, hubs.size());
            }
        }
        int numHubs = hubs.size();
        // Now pick numHubs antihubs.
        int[] hArrCopy = Arrays.copyOf(hubnessArray, hubnessArray.length);
        // Perform ascending sort.
        int[] reSortIndexes = AuxSort.sortIndexedValue(hArrCopy, false);
        for (int i = 0; i < numHubs; i++) {
            antiHubs.add(reSortIndexes[i]);
        }
        for (int i = numHubs; i < hubnessArray.length - numHubs; i++) {
            regulars.add(reSortIndexes[i]);
        }
        File outFile = new File(outDir,
                getDataFileName(filePath) + File.separator
                + "NSize" + k + ".txt");
        FileUtil.createFile(outFile);

        PrintWriter pw = new PrintWriter(new FileWriter(outFile));
        int min, max;
        try {
            pw.println("data: " + getDataFileName(filePath) + ",    k: " + k);
            pw.println("instances: " + datasize + ", classes: " + numClasses);
            pw.println("occ skewness: " + hubness + ", bh perc: " + bhPerc);
            pw.println("number of hubs (>k+2stDev occ): " + hubs.size());
            pw.println();
            for (int i = 0; i < methods.length; i++) {
                pw.println("-------------------------------------------------");
                pw.println(methodNames[i]);
                System.out.println("writing: " + methodNames[i]);
                ArrayList<Integer> protoIndexes =
                        methods[i].getPrototypeIndexes();
                pw.println("numPrototypes " + protoIndexes.size());
                methods[i].calculatePrototypeHubness(k);
                float[][] protoDistMatrixMc = new float[protoIndexes.size()][];
                for (int index1 = 0; index1 < protoDistMatrixMc.length;
                        index1++) {
                    protoDistMatrixMc[index1] =
                            new float[protoDistMatrixMc.length - index1 - 1];
                    for (int index2 = index1 + 1;
                            index2 < protoDistMatrixMc.length; index2++) {
                        min = Math.min(protoIndexes.get(index1),
                                protoIndexes.get(index2));
                        max = Math.max(protoIndexes.get(index1),
                                protoIndexes.get(index2));
                        protoDistMatrixMc[index1][index2 - index1 - 1] =
                                distMat[min][max - min - 1];
                    }
                }
                DataSet protoCollection = dset.getSubsample(protoIndexes);
                NeighborSetFinder protoFinder = new NeighborSetFinder(
                        protoCollection, protoDistMatrixMc, cmet);
                protoFinder.calculateNeighborSets(k);
                methods[i].calculatePrototypeHubness(k);
                // These two need to be compared between each other. A high
                // correlation between the actual and prototype-restricted
                // neighbor occurrence models means that the instance selection
                // bias has a low impact on the neighbor occurrence model. A low
                // correlation, on the other hand, would imply that kNN models
                // learned on the selected subset might be severely biased and
                // possibly perform bad on the test data.
                float[][] cDataTrue = methods[i].
                        getClassDataNeighborRelationforFuzzy(numClasses, 0.01f);
                float[][] cDataProto = protoFinder.
                        getFuzzyClassDataNeighborRelation(k, numClasses,
                        0.01f, true);
                float corrAVG = 0;

                for (int c = 0; c < numClasses; c++) {
                    corrAVG += PearsonCorrelation.correlation(cDataProto[c],
                            cDataTrue[c]);
                }
                corrAVG /= numClasses;
                int[] bhProto = protoFinder.getBadFrequencies();
                double bhPercProto = ArrayUtil.sum(bhProto);
                bhPercProto /= (protoIndexes.size() * k);
                int[] bhTrue = methods[i].getPrototypeBadHubness();
                double bhPercTrue = ArrayUtil.sum(bhTrue);
                bhPercTrue /= (datasize * k);
                float skewnessRed = HigherMoments.
                        calculateSkewForSampleArray(
                        protoFinder.getNeighborFrequencies());
                float skewnessTrue = HigherMoments.
                        calculateSkewForSampleArray(methods[i].
                        getPrototypeHubness());
                pw.println("avg class hubness correlation (protoReduced/"
                        + "protoTrue): " + corrAVG);
                pw.println("hubness correlation: "
                        + PearsonCorrelation.correlation(
                        protoFinder.getNeighborFrequencies(),
                        methods[i].getPrototypeHubness()));
                pw.println("bad hubness correlation: "
                        + PearsonCorrelation.correlation(
                        protoFinder.getBadFrequencies(),
                        methods[i].getPrototypeBadHubness()));
                // Bad and total neighbor occurrence frequencies.
                int[] bfProto = protoFinder.getBadFrequencies();
                int[] bfMethod = methods[i].getPrototypeBadHubness();
                int[] totalProto = protoFinder.getNeighborFrequencies();
                int[] totalMethod = methods[i].getPrototypeHubness();
                double avgBHPercError = 0;
                double firstPerc, secondPerc;
                for (int j = 0; j < bfProto.length; j++) {
                    if (totalProto[j] == 0) {
                        firstPerc = 0;
                    } else {
                        firstPerc = (double) bfProto[j]
                                / (double) totalProto[j];
                    }
                    if (totalMethod[j] == 0) {
                        secondPerc = 0;
                    } else {
                        secondPerc = (double) bfMethod[j]
                                / (double) totalMethod[j];
                    }
                    avgBHPercError += Math.abs(firstPerc - secondPerc);
                }
                avgBHPercError /= protoIndexes.size();
                pw.println("AVG bhPercError (pointwise): " + avgBHPercError);
                pw.println("proto-bhRed: " + bhPercProto
                        + ", proto-bhTrue: " + bhPercTrue);
                pw.println("skewnessRed: " + skewnessRed
                        + ", skewnessTrue: " + skewnessTrue);
                int hCount = 0;
                for (int index : protoIndexes) {
                    if (hubHash.containsKey(index)) {
                        hCount++;
                    }
                }
                float hPerc = (float) hCount / (float) (hubs.size());
                pw.println("perc retained hubs among the prototypes: " + hPerc);
                pw.println("-------------------------------------------------");
            }
        } catch (Exception e) {
            System.out.println(e.getMessage());
            throw e;
        } finally {
            pw.close();
        }
    }

    /**
     * @param fileNameWithExtension String representing the file name with
     * extension included.
     * @return File name without the extension.
     */
    public static String getDataFileName(String fileNameWithExtension) {
        File tmp = new File(fileNameWithExtension);
        String n = tmp.getName();
        if (n.endsWith(".csv") || n.endsWith(".tsv") || n.endsWith(".txt")) {
            return n.substring(0, n.length() - 4);
        } else if (n.endsWith(".arff")) {
            return n.substring(0, n.length() - 5);
        } else {
            return "a";
        }
    }

    /**
     * @param dMatFile File to load the matrix from.
     * @return Upper diagonal float distance matrix, each row contains only
     * distances d(i,j) for j > i. Therefore the length of the i-th row is
     * len(i_th_row) = n - i - 1.
     * @throws Exception
     */
    public static float[][] loadDMatFromFile(File dMatFile) throws Exception {
        BufferedReader br = new BufferedReader(
                new InputStreamReader(new FileInputStream(dMatFile)));
        float[][] dMatLoaded = null;
        String s;
        String[] lineParse;
        try {
            int size = Integer.parseInt(br.readLine());
            dMatLoaded = new float[size][];
            for (int i = 0; i < size - 1; i++) {
                dMatLoaded[i] = new float[size - i - 1];
                s = br.readLine();
                lineParse = s.split(",");
                for (int j = 0; j < lineParse.length; j++) {
                    dMatLoaded[i][j] = Float.parseFloat(lineParse[j]);
                }
            }
            dMatLoaded[size - 1] = new float[0];
        } catch (IOException | NumberFormatException e) {
            throw e;
        } finally {
            br.close();
        }
        return dMatLoaded;
    }

    /**
     * Runs the instance selection methods comparisons.
     *
     * @param args Command line arguments.
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        CommandLineParser clp = new CommandLineParser(true);
        clp.addParam("-inFile", "Path to the input file.",
                CommandLineParser.STRING, true, false);
        clp.addParam("-inDMat", "Path to the input distance matrix file.",
                CommandLineParser.STRING, true, false);
        clp.addParam("-outDir", "Path to the output directory.",
                CommandLineParser.STRING, true, false);
        clp.addParam("-k", "Neighborhood size.",
                CommandLineParser.INTEGER, true, false);
        clp.addParam("-normalizeFloats", "Self-descriptive.",
                CommandLineParser.BOOLEAN, true, false);
        clp.parseLine(args);
        compareMethods(
                (Integer) clp.getParamValues("-k").get(0),
                (String) clp.getParamValues("-inFile").get(0),
                (String) clp.getParamValues("-inDMat").get(0),
                new File((String) clp.getParamValues("-outDir").get(0)),
                (Boolean) clp.getParamValues("-normalizeFloats").get(0));
    }
}
