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

import data.generators.DataGenerator;
import data.generators.MultiDimensionalSphericGaussianGenerator;
import data.generators.UniformGenerator;
import data.neighbors.NeighborSetFinder;
import data.neighbors.SharedNeighborFinder;
import data.representation.DataInstance;
import data.representation.DataSet;
import distances.primary.CombinedMetric;
import distances.secondary.MutualProximityCalculator;
import distances.secondary.NICDMCalculator;
import distances.secondary.snd.SharedNeighborCalculator;
import ioformat.FileUtil;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import statistics.HigherMoments;
import util.ArrayUtil;
import util.CommandLineParser;
import util.SOPLUtil;

/**
 * This class is meant to empirically estimate the distribution of the neighbor
 * occurrence skewness in synthetic data of controlled dimensionality under
 * standard metrics.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class HubnessRiskEstimatorFromGaussian {

    private File outFile;
    private int k = 5;
    private int kForSecondary = 100;
    private int dataSize = 2000;
    private int dim = 100;
    private int numRepetitions = 500;
    private CombinedMetric cmet = CombinedMetric.EUCLIDEAN;
    private DataGenerator gen;
    private StatsLogger primaryLogger, nicdmLogger, simcosLogger, simhubLogger,
            mpLogger;
    private DataSet dset;
    private float[][] dMatPrimary, dMatSecondary;
    private NeighborSetFinder nsfPrimary, nsfSecondary;
    public static final int NUM_THREADS = 8;
    
    /**
     * This method generates a dataset based on the provided generator.
     * 
     * @return DataSet of synthetic data. 
     */
    private DataSet generateDataSet() {
        if (gen == null) {
            return null;
        }
        DataSet synthSet = new DataSet();
        String[] dummyAttNames = new String[dim];
        for (int d = 0; d < dim; d++) {
            dummyAttNames[d] = "fAtt" + d;
        }
        synthSet.fAttrNames = dummyAttNames;
        synthSet.data = new ArrayList<>(dataSize);
        for (int i = 0; i < dataSize; i++) {
            DataInstance instance = new DataInstance(synthSet);
            instance.fAttr = gen.generateFloat();
            instance.embedInDataset(synthSet);
            synthSet.addDataInstance(instance);
        }
        return synthSet;
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
        int kPrimMax = Math.max(k, kForSecondary);
        for (int iteration = 0; iteration < numRepetitions; iteration++) {
            System.out.println("Starting iteration: " + iteration);
            dset = generateDataSet();
            dMatPrimary = dset.calculateDistMatrixMultThr(cmet, NUM_THREADS);
            nsfPrimary = new NeighborSetFinder(dset, dMatPrimary, cmet);
            nsfPrimary.calculateNeighborSets(kPrimMax);
            // We will re-calculate for the smaller k later, now we use this
            // kNN object for secondary distances, where necessary.
            // Calculate the secondary NICDM distances.
            NICDMCalculator nsc = new NICDMCalculator(nsfPrimary);
            dMatSecondary =
                    nsc.getTransformedDMatFromNSFPrimaryDMat();
            nsfSecondary = new NeighborSetFinder(dset, dMatSecondary, nsc);
            nsfSecondary.calculateNeighborSets(k);
            nicdmLogger.updateByObservedFreqs(
                    nsfSecondary.getNeighborFrequencies());
            // Calculate the secondary Simcos distances.
            SharedNeighborFinder snf =
                    new SharedNeighborFinder(nsfPrimary, k);
            snf.setNumClasses(1);
            snf.countSharedNeighborsMultiThread(NUM_THREADS);
            // First fetch the similarities.
            dMatSecondary = snf.getSharedNeighborCounts();
            // Then transform them into distances.
            for (int indexFirst = 0; indexFirst < dMatSecondary.length;
                    indexFirst++) {
                for (int indexSecond = 0; indexSecond <
                        dMatSecondary[indexFirst].length; indexSecond++) {
                    dMatSecondary[indexFirst][indexSecond] = kForSecondary -
                            dMatSecondary[indexFirst][indexSecond];
                }
            }
            SharedNeighborCalculator snc =
                    new SharedNeighborCalculator(snf,SharedNeighborCalculator.
                    WeightingType.NONE);
            nsfSecondary = new NeighborSetFinder(dset, dMatSecondary, snc);
            nsfSecondary.calculateNeighborSets(k);
            simcosLogger.updateByObservedFreqs(
                    nsfSecondary.getNeighborFrequencies());
            // Calculate the secondary Simhub distances. These are actually the
            // simhub^inf variant, since there are not classes in the data.
            snf = new SharedNeighborFinder(nsfPrimary, k);
            snf.setNumClasses(1);
            snf.obtainWeightsFromGeneralHubness();
            snf.countSharedNeighborsMultiThread(NUM_THREADS);
            // First fetch the similarities.
            dMatSecondary = snf.getSharedNeighborCounts();
            // Then transform them into distances.
            for (int indexFirst = 0; indexFirst < dMatSecondary.length;
                    indexFirst++) {
                for (int indexSecond = 0; indexSecond <
                        dMatSecondary[indexFirst].length; indexSecond++) {
                    dMatSecondary[indexFirst][indexSecond] = kForSecondary -
                            dMatSecondary[indexFirst][indexSecond];
                }
            }
            // Calculate the test-to-training point distances.
            snc = new SharedNeighborCalculator(snf,SharedNeighborCalculator.
                    WeightingType.HUBNESS);
            nsfSecondary = new NeighborSetFinder(dset, dMatSecondary, snc);
            nsfSecondary.calculateNeighborSets(k);
            simhubLogger.updateByObservedFreqs(
                    nsfSecondary.getNeighborFrequencies());
            // Calculate the secondary Mutual Proximity distances.
            MutualProximityCalculator calc =
                    new MutualProximityCalculator(nsfPrimary.getDistances(),
                    nsfPrimary.getDataSet(), nsfPrimary.getCombinedMetric());
            dMatSecondary = calc.calculateSecondaryDistMatrixMultThr(
                    nsfPrimary, 8);
            nsfSecondary = new NeighborSetFinder(dset, dMatSecondary, calc);
            nsfSecondary.calculateNeighborSets(k);
            mpLogger.updateByObservedFreqs(
                    nsfSecondary.getNeighborFrequencies());
            // Finally the primary distances.
            nsfPrimary = nsfPrimary.getSubNSF(k);
            primaryLogger.updateByObservedFreqs(
                    nsfPrimary.getNeighborFrequencies());
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
        
        /**
         * Initialization.
         */
        public StatsLogger(String distName) {
            this.distName = distName;
            skewValues = new ArrayList<>(numRepetitions);
            kurtosisValues = new ArrayList<>(numRepetitions);
        }

        private ArrayList<Float> skewValues;
        private ArrayList<Float> kurtosisValues;
        
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
        clp.addParam("-numDimensions", "Number of dimensions to generate.",
                CommandLineParser.INTEGER, true, false);
        clp.addParam("-numInstances", "Number of instances to generate.",
                CommandLineParser.INTEGER, true, false);
        clp.addParam("-numRepetitions", "Number of repetitions.",
                CommandLineParser.INTEGER, true, false);
        clp.addParam("-k", "The neighborhood size.",
                CommandLineParser.INTEGER, true, false);
        clp.addParam("-gaussian", "True if Gaussian, false if uniform.",
                CommandLineParser.BOOLEAN, true, false);
        clp.addParam("-outFile", "Output arff file path.",
                CommandLineParser.STRING, true, false);
        clp.parseLine(args);
        HubnessRiskEstimatorFromGaussian experimenter =
                new HubnessRiskEstimatorFromGaussian();
        experimenter.outFile = new File(
                (String)(clp.getParamValues("-outFile").get(0)));
        experimenter.k = (Integer) clp.getParamValues("-k").get(0);
        experimenter.dim = (Integer) clp.getParamValues(
                "-numDimensions").get(0);
        experimenter.numRepetitions = (Integer) clp.getParamValues(
                "-numRepetitions").get(0);
        experimenter.dataSize = (Integer) clp.getParamValues(
                "-numInstances").get(0);
        // For this experiment to be repeatable, the mean/stDev values for the
        // generating distribution are fixed and symmetrical.
        float[] distMeans = new float[experimenter.dim];
        float[] distStDevs = new float[experimenter.dim];
        Arrays.fill(distStDevs, 1);
        float[] lowerBounds = new float[experimenter.dim];
        Arrays.fill(lowerBounds, -100);
        float[] upperBounds = new float[experimenter.dim];
        Arrays.fill(upperBounds, 100);
        if ((Boolean) clp.getParamValues("-gaussian").get(0)) {
            experimenter.gen = new MultiDimensionalSphericGaussianGenerator(
                    distMeans, distStDevs, lowerBounds, upperBounds);
        } else {
            experimenter.gen = new UniformGenerator(lowerBounds, upperBounds);
        }
        experimenter.performAllTests();
    }
}
