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
package images.mining.codebook;

import data.neighbors.NeighborSetFinder;
import data.representation.DataInstance;
import data.representation.DataSet;
import distances.primary.CombinedMetric;
import ioformat.FileUtil;
import ioformat.IOARFF;
import ioformat.SupervisedLoader;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import statistics.HigherMoments;
import util.AuxSort;
import util.BasicMathUtil;
import util.CommandLineParser;
import util.SOPLUtil;

/**
 * This class calculates the entropy distribution of different visual words,
 * that is - codebook vectors in images of a specified dataset. Different visual
 * words exhibit a different usefulness for classification and this helps in
 * pinpointing those codebook vectors that are more beneficial.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class VisualWordEntropyDistribution {

    public static void main(String[] args) throws Exception {
        CommandLineParser clp = new CommandLineParser(true);
        int numBuckets = 20;
        clp.addParam("-inCodebook", "Path to the codebook",
                CommandLineParser.STRING, true, false);
        clp.addParam("-inDataSet", "Path to the dataset",
                CommandLineParser.STRING, true, false);
        clp.addParam("-outFile", "Output path", CommandLineParser.STRING,
                true, false);
        clp.addParam("-outWeightFile", "Out path for feature weights",
                CommandLineParser.STRING, false, false);
        clp.addParam("-numBuckets", "Number of buckets",
                CommandLineParser.INTEGER, false, false);
        clp.addParam("-k", "neighborhood size", CommandLineParser.INTEGER,
                true, false);
        clp.parseLine(args);
        File inCodebook =
                new File((String) (clp.getParamValues("-inCodebook").get(0)));
        File inDataSet =
                new File((String) (clp.getParamValues("-inDataSet").get(0)));
        File outFile =
                new File((String) (clp.getParamValues("-outFile").get(0)));
        if (clp.hasParamValue("-numBuckets")) {
            numBuckets = (Integer) (clp.getParamValues("-numBuckets").get(0));
        }
        FileUtil.createFile(outFile);
        // Load the sample.
        DataSet dset = SupervisedLoader.loadData(inDataSet.getPath(), false);
        // Load the codebook.
        SIFTCodeBook codebook = new SIFTCodeBook();
        codebook.loadCodeBookFromFile(inCodebook);
        int numClasses = dset.countCategories();
        float[] codebookOccTotal = new float[codebook.getSize()];
        float[][] codebookOccurrenceProfiles =
                new float[codebook.getSize()][numClasses];
        int currClass;
        boolean floats = dset.hasFloatAttr();
        for (int i = 0; i < dset.size(); i++) {
            currClass = dset.getLabelOf(i);
            for (int d = 0; d < codebook.getSize(); d++) {
                if (floats) {
                    codebookOccurrenceProfiles[d][currClass] +=
                            dset.getInstance(i).fAttr[d];
                    codebookOccTotal[d] += dset.getInstance(i).fAttr[d];
                } else {
                    codebookOccurrenceProfiles[d][currClass] +=
                            dset.getInstance(i).iAttr[d];
                    codebookOccTotal[d] += dset.getInstance(i).iAttr[d];
                }
            }
        }
        int[] entropyHistogram = new int[numBuckets];
        double factor;
        double currEntropy;
        int index;
        float[] maxOccClass = new float[codebook.getSize()];
        float[] hFeatWeights = new float[codebook.getSize()];
        // Calculate the entropies.
        for (int d = 0; d < codebook.getSize(); d++) {
            currEntropy = 0;
            for (int c = 0; c < numClasses; c++) {
                factor = codebookOccurrenceProfiles[d][c] / codebookOccTotal[d];
                if (factor > 0) {
                    currEntropy -= factor * BasicMathUtil.log2(factor);
                }
                if (codebookOccurrenceProfiles[d][c] > maxOccClass[d]) {
                    maxOccClass[d] = codebookOccurrenceProfiles[d][c];
                }
            }
            currEntropy /= BasicMathUtil.log2(numClasses);
            if (codebookOccTotal[d] > 0) {
                hFeatWeights[d] = 1 - (float) currEntropy;
            }
            index = (int) (currEntropy * numBuckets);
            entropyHistogram[index]++;
        }
        // Now check how well the visual words model hubs, anti-hubs and
        // regular images.
        int neighborhoodSize = (Integer) (clp.getParamValues("-k").get(0));
        // Initialize hubness-related lists and structures.
        ArrayList<Integer> hubs = new ArrayList<>(1000);
        ArrayList<Integer> badhubs = new ArrayList<>(1000);
        ArrayList<Integer> antihubs = new ArrayList<>(1000);
        ArrayList<Integer> goodhubs = new ArrayList<>(1000);
        ArrayList<Integer> regulars = new ArrayList<>(1000);
        // Calculate the distance matrix.
        float[][] distMat = dset.calculateDistMatrix(
                CombinedMetric.FLOAT_MANHATTAN);
        NeighborSetFinder nsf = new NeighborSetFinder(
                dset, distMat, CombinedMetric.FLOAT_MANHATTAN);
        nsf.calculateNeighborSets(neighborhoodSize);
        float[] hubness = nsf.getFloatOccFreqs();
        int[] badHubness = nsf.getBadFrequencies();
        int[] goodHubness = nsf.getGoodFrequencies();
        // The difference between good and bad hubness.
        float[] gmb = new float[dset.size()];
        for (int i = 0; i < dset.size(); i++) {
            gmb[i] = goodHubness[i] - badHubness[i];
        }
        float meanH = HigherMoments.calculateArrayMean(hubness);
        float stDevH = HigherMoments.calculateArrayStDev(meanH, hubness);
        boolean[] hubOrAntiHub = new boolean[dset.size()];
        float hubWeights = 0;
        float regularWeights = 0;
        float antiHubWeights = 0;
        float goodHubWeights = 0;
        float badHubWeights = 0;
        float hubFeat = 0;
        float regularFeat = 0;
        float antiHubFeat = 0;
        float goodHubFeat = 0;
        float badHubFeat = 0;
        float[] weightedFeatureOccurrenceAverages = new float[dset.size()];
        float[] numFeatures = new float[dset.size()];
        for (int i = 0; i < dset.size(); i++) {
            weightedFeatureOccurrenceAverages[i] = 0;
            numFeatures[i] = 0;
            for (int d = 0; d < codebook.getSize(); d++) {
                if (floats) {
                    numFeatures[i] += dset.getInstance(i).fAttr[d];
                    weightedFeatureOccurrenceAverages[i] +=
                            dset.getInstance(i).fAttr[d] * hFeatWeights[d];
                } else {
                    numFeatures[i] += dset.getInstance(i).iAttr[d];
                    weightedFeatureOccurrenceAverages[i] +=
                            dset.getInstance(i).iAttr[d] * hFeatWeights[d];
                }
            }
            if (numFeatures[i] > 0) {
                weightedFeatureOccurrenceAverages[i] /= numFeatures[i];
            }
        }
        for (int i = 0; i < dset.size(); i++) {
            if (hubness[i] >= meanH + 2 * stDevH) {
                hubs.add(i);
                hubOrAntiHub[i] = true;
                hubWeights += weightedFeatureOccurrenceAverages[i];
                hubFeat += numFeatures[i];
            }
        }
        // This integer array holds the re-arranged indexes corresponding to the
        // permutation that was induced by the sorting.
        int[] reArrByHubness = AuxSort.sortIndexedValue(
                Arrays.copyOf(hubness, hubness.length), AuxSort.ASCENDING);
        for (int i = 0; i < hubs.size(); i++) {
            antihubs.add(reArrByHubness[i]);
            antiHubWeights +=
                    weightedFeatureOccurrenceAverages[reArrByHubness[i]];
            antiHubFeat += numFeatures[i];
            hubOrAntiHub[reArrByHubness[i]] = true;
        }
        // This integer array holds the re-arranged indexes corresponding to the
        // permutation that was induced by the sorting.
        int[] reArrByGoodMinusBadHubness = AuxSort.sortIndexedValue(
                Arrays.copyOf(gmb, gmb.length), AuxSort.ASCENDING);
        for (int i = 0; i < hubs.size(); i++) {
            badhubs.add(reArrByGoodMinusBadHubness[i]);
            badHubWeights += weightedFeatureOccurrenceAverages[
                    reArrByGoodMinusBadHubness[i]];
            badHubFeat += numFeatures[reArrByGoodMinusBadHubness[i]];
            goodhubs.add(reArrByGoodMinusBadHubness[gmb.length - i - 1]);
            goodHubWeights += weightedFeatureOccurrenceAverages[
                    reArrByGoodMinusBadHubness[gmb.length - i
                    - 1]];
            goodHubFeat += numFeatures[reArrByGoodMinusBadHubness[gmb.length - i
                    - 1]];
        }
        for (int i = 0; i < dset.size(); i++) {
            if (!hubOrAntiHub[i]) {
                regulars.add(i);
                regularWeights += weightedFeatureOccurrenceAverages[i];
                regularFeat += numFeatures[i];
            }
        }
        // Normalize.
        if (hubWeights > 0) {
            hubWeights /= hubs.size();
        }
        if (regularWeights > 0) {
            regularWeights /= regulars.size();
        }
        if (antiHubWeights > 0) {
            antiHubWeights /= antihubs.size();
        }
        if (goodHubWeights > 0) {
            goodHubWeights /= goodhubs.size();
        }
        if (badHubWeights > 0) {
            badHubWeights /= badhubs.size();
        }
        if (hubFeat > 0) {
            hubFeat /= hubs.size();
        }
        if (regularFeat > 0) {
            regularFeat /= regulars.size();
        }
        if (antiHubFeat > 0) {
            antiHubFeat /= antihubs.size();
        }
        if (goodHubFeat > 0) {
            goodHubFeat /= goodhubs.size();
        }
        if (badHubFeat > 0) {
            badHubFeat /= badhubs.size();
        }
        // Print the findings to a file.
        PrintWriter pw = new PrintWriter(new FileWriter(outFile));
        try {
            SOPLUtil.printArrayToStream(entropyHistogram, pw);
            pw.println();
            pw.println("Average normalized feature entropies");
            pw.println("Antihubs, Regulars, Hubs");
            pw.println((1 - antiHubWeights) + "," + (1 - regularWeights) + ","
                    + (1 - hubWeights));
            pw.println("Good hubs, Bad hubs");
            pw.println((1 - goodHubWeights) + "," + (1 - badHubWeights));
            pw.println("Average number of features");
            pw.println("Antihubs, Regulars, Hubs");
            pw.println(antiHubFeat + "," + regularFeat + "," + hubFeat);
            pw.println("Good hubs, Bad hubs");
            pw.println(goodHubFeat + "," + badHubFeat);
        } catch (Exception e) {
            throw e;
        } finally {
            pw.close();
        }
        if (clp.hasParamValue("-outWeightFile")) {
            File outFileWeights = new File((String) (clp.getParamValues(
                    "-outWeightFile").get(0)));
            IOARFF saver = new IOARFF();
            DataSet dsWeights = new DataSet();
            dsWeights.fAttrNames = new String[codebook.getSize()];
            DataInstance instance = new DataInstance(dsWeights);
            for (int d = 0; d < codebook.getSize(); d++) {
                dsWeights.fAttrNames[d] = "weight " + d;
                instance.fAttr[d] = hFeatWeights[d];
            }
            instance.embedInDataset(dsWeights);
            dsWeights.addDataInstance(instance);
            saver.save(dsWeights, outFileWeights.getPath(), null);
        }
    }
}
