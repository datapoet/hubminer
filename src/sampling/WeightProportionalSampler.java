/**
 * Hub Miner: a hubness-aware machine learning experimentation library.
 * Copyright (C) 2014 Nenad Tomasev. Email: nenad.tomasev at gmail.com
 * 
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 * 
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 * 
 * You should have received a copy of the GNU General Public License along with
 * this program. If not, see <http://www.gnu.org/licenses/>.
 */
package sampling;

import data.representation.DataInstance;
import data.representation.DataSet;
import data.representation.util.DataMineConstants;
import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Deque;
import java.util.HashMap;
import java.util.PriorityQueue;
import java.util.Random;
import util.ArrayUtil;
import util.DataSetJoiner;
import util.SOPLUtil;

/**
 * A class that implements weight-proportional data sampling.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class WeightProportionalSampler extends Sampler {

    float[] samplingWeights;

    /**
     * @param repetitions Boolean flag denoting whether to use repetitive
     * sampling or not.
     * @param samplingWeights float[] representing the weights for the
     * weight-proportional selection scheme.
     */
    public WeightProportionalSampler(boolean repetitions,
            float[] samplingWeights) {
        super(repetitions);
        this.samplingWeights = samplingWeights;
    }

    /**
     * Gets the indexes of a weight-proportional sample, with no repetitions.
     *
     * @param sampleSize The size of the sample.
     * @param samplingWeights float[] representing the weights for the
     * weight-proportional selection scheme. The length of this array is the
     * population size.
     * @return int[] of indexes of the weight-proportional sample.
     * @throws Exception
     */
    public static int[] getSampleNoReps(int sampleSize, float[] samplingWeights)
            throws Exception {
        double[] samplingWeightsD = new double[samplingWeights.length];
        for (int i = 0; i < samplingWeights.length; i++) {
            samplingWeightsD[i] = samplingWeights[i];
        }
        WeightedReservoirSampling sampler =
                new WeightedReservoirSampling(samplingWeightsD);
        return sampler.getRandomSampleNoReps(sampleSize);
    }

    /**
     * Gets the indexes of a weight-proportional sample, with possible
     * repetitions.
     *
     * @param samplingWeights float[] representing the weights for the
     * weight-proportional selection scheme. The length of this array is the
     * population size.
     * @return int[] of indexes of the weight-proportional sample.
     * @throws Exception
     */
    public static int[] getSampleWithReps(int sampleSize,
            float[] samplingWeights)
            throws Exception {
        double[] samplingWeightsD = new double[samplingWeights.length];
        for (int i = 0; i < samplingWeights.length; i++) {
            samplingWeightsD[i] = samplingWeights[i];
        }
        AliasSamplingMethod sampler = new AliasSamplingMethod(
                samplingWeightsD);
        int[] result = new int[sampleSize];
        for (int i = 0; i < sampleSize; i++) {
            result[i] = sampler.sampleAnIndex();
        }
        return result;
    }

    @Override
    public DataSet getSample(DataSet dset, int sampleSize) throws Exception {
        if (sampleSize <= 0 || dset == null || dset.isEmpty()) {
            return new DataSet();
        }
        DataSet sample = new DataSet(
                dset.iAttrNames,
                dset.fAttrNames,
                dset.sAttrNames, sampleSize);
        if ((!getRepetitions()) && sampleSize <= dset.size()) {
            // Take a weight-proportional stochastic sample with no repetitions.
            int[] selIndexes = getSampleNoReps(sampleSize, samplingWeights);
            for (int index : selIndexes) {
                DataInstance instanceCopy = dset.data.get(index).copy();
                sample.addDataInstance(instanceCopy);
                instanceCopy.embedInDataset(sample);
            }
        } else {
            // If repetitions are allowed, we just go and sample iteratively.
            double[] samplingWeightsD = new double[samplingWeights.length];
            for (int i = 0; i < samplingWeights.length; i++) {
                samplingWeightsD[i] = samplingWeights[i];
            }
            AliasSamplingMethod sampler = new AliasSamplingMethod(
                    samplingWeightsD);
            for (int i = 0; i < sampleSize; i++) {
                int selectedIndex = sampler.sampleAnIndex();
                sample.addDataInstance(dset.data.get(selectedIndex).copy());
                sample.data.get(i).embedInDataset(sample);
            }
        }
        return sample;
    }

    @Override
    public DataSet getSample(DataSet[] dsets, int sampleSize) throws Exception {
        if (sampleSize <= 0 || dsets == null || dsets.length == 0) {
            return new DataSet();
        }
        DataSet sample;
        // A flag for when all data sets have size zero.
        boolean allZero = true;

        // We count the non-empty data sets.
        int totalSize = 0;
        int numNonEmpty = 0;
        for (int i = 0; i < dsets.length; i++) {
            if (dsets[i] == null || dsets[i].isEmpty()) {
            } else {
                allZero = false;
                numNonEmpty++;
            }
        }
        if (allZero) {
            return new DataSet();
        }
        // Clean up the null collections so that the joiner can join later on.
        DataSet[] nonEmptyCollections = dsets;
        if (numNonEmpty < dsets.length) {
            nonEmptyCollections = new DataSet[numNonEmpty];
        }
        // Make a new collection array, where all data sets are non-null and
        // non-empty.
        int curr = -1;
        for (int i = 0; i < dsets.length; i++) {
            if (dsets[i] != null && !(dsets[i].isEmpty())) {
                nonEmptyCollections[++curr] = dsets[i];
            }
        }
        // Get sizes of all non-empty data sets into an array.
        int[] sizeArray = new int[nonEmptyCollections.length];
        for (int i = 0; i < nonEmptyCollections.length; i++) {
            sizeArray[i] = nonEmptyCollections[i].size();
            totalSize += sizeArray[i];
        }
        // As we need a sample of a fixed size, samples from individual datasets
        // need to sum up exactly to that, while maintaining proportions as well
        // as possible. Here we calculate a sample size array for all target
        // data sets.
        float[] sampleSizeArrayFloat = new float[nonEmptyCollections.length];
        int[] sampleSizeArray = new int[nonEmptyCollections.length];
        int currSize = 0;
        for (int i = 0; i < nonEmptyCollections.length; i++) {
            if (sizeArray[i] != 0) {
                sampleSizeArrayFloat[i] = ((float) sizeArray[i]
                        / (float) totalSize) * (float) sampleSize;
                sampleSizeArray[i] = (int) (sampleSizeArrayFloat[i]);
                currSize += sampleSizeArray[i];
            }
        }
        // As the integer cast loses decimal places, sampleSize will always be
        // higher than currSize.
        int bad = sampleSize - currSize;
        // Now scatter these randomly
        int numFix = 0;
        int candidate;
        Random randa = new Random();
        // We just fix individual data set sample size specifications prior to
        // extracting the sample.
        while (numFix < bad) {
            candidate = randa.nextInt(sampleSizeArray.length);
            if (sizeArray[candidate] > sampleSizeArray[candidate]) {
                numFix++;
                sampleSizeArray[candidate]++;
            }
        }
        // We go and iteratively sample through the data sets.
        DataSet[] samples = new DataSet[sampleSizeArray.length];
        for (int i = 0; i < sampleSizeArray.length; i++) {
            samples[i] = getSample(nonEmptyCollections[i], sampleSizeArray[i]);
        }
        sample = DataSetJoiner.join(samples);
        return sample;
    }

    /**
     * The following class implements the Alias method for sampling from
     * discrete probability distributions. It supports weight-proportional
     * stochastic sampling with repetitions. Based on an implementation by Keith
     * Schwarz (htiek@cs.stanford.edu), available at:
     * http://www.keithschwarz.com/interesting/code/?dir=alias-method
     */
    static class AliasSamplingMethod {

        private Random randa;
        // An array holding the alias labels to use in case the random number is
        // below the value in the probability array.
        private int[] aliasArr;
        private double[] indexProbabilityArr;

        /**
         * Initialization.
         *
         * @param weights double[] representing the weights to derive
         * probabilities from.
         */
        public AliasSamplingMethod(double[] weights) {
            randa = new Random();
            // Check the parameters.
            if (weights == null) {
                throw new NullPointerException();
            }
            if (weights.length == 0) {
                throw new IllegalArgumentException("Empty weights vector "
                        + "provided.");
            }
            // Initialize the arrays.
            int numCases = weights.length;
            indexProbabilityArr = new double[numCases];
            aliasArr = new int[numCases];
            // The average probability.
            final double averageProb = 1.0 / numCases;
            // Make a normalized weights array.
            double[] weightsNormalized = Arrays.copyOf(weights, numCases);
            double weightSum = ArrayUtil.sum(weights);
            if (!DataMineConstants.isAcceptableDouble(weightSum)
                    || DataMineConstants.isZero(weightSum)) {
                throw new IllegalArgumentException("Bad weights vector, unable "
                        + "to normalize.");
            }
            for (int i = 0; i < numCases; i++) {
                weightsNormalized[i] /= weightSum;
            }
            // Worklists for populating the tables.
            Deque<Integer> belowAvgIndexes = new ArrayDeque<>();
            Deque<Integer> aboveAvgIndexes = new ArrayDeque<>();
            // Populate the stacks with input probabilities.
            for (int i = 0; i < numCases; i++) {
                // The decision on which worklist to add a value to is based on 
                // whether it is above or below the average. This is done for 
                // easier pairing in the constructive step.
                if (weightsNormalized[i] <= averageProb) {
                    belowAvgIndexes.add(i);
                } else {
                    aboveAvgIndexes.add(i);
                }
            }
            while (!belowAvgIndexes.isEmpty() && !aboveAvgIndexes.isEmpty()) {
                int smaller = belowAvgIndexes.removeLast();
                int larger = aboveAvgIndexes.removeLast();
                // Scaling so that 1/numCases probability is given value 1 in 
                // the array.
                indexProbabilityArr[smaller] = weightsNormalized[smaller]
                        * numCases;
                aliasArr[smaller] = larger;
                // Decrease the remaining probability mass of the larger vaue.
                weightsNormalized[larger] = weightsNormalized[larger]
                        + weightsNormalized[smaller] - averageProb;
                // Update the lists.
                if (weightsNormalized[larger] >= 1.0 / numCases) {
                    aboveAvgIndexes.add(larger);
                } else {
                    belowAvgIndexes.add(larger);
                }
            }
            // After the previous loop, everything is in one list. We empty 
            // both structures (though one in principle) and finalize the 
            // process. There is no need to set the alias indexes for these 
            // points.
            while (!belowAvgIndexes.isEmpty()) {
                indexProbabilityArr[belowAvgIndexes.removeLast()] = 1.0;
            }
            while (!aboveAvgIndexes.isEmpty()) {
                indexProbabilityArr[aboveAvgIndexes.removeLast()] = 1.0;
            }
        }

        /**
         * Samples a value from the underlying distribution.
         *
         * @return A random value sampled from the underlying distribution.
         */
        public int sampleAnIndex() {
            // First randomly select a column.
            int column = randa.nextInt(indexProbabilityArr.length);
            // Perform the biased coin toss.
            boolean pickSelectedColumn = randa.nextDouble()
                    < indexProbabilityArr[column];
            return pickSelectedColumn ? column : aliasArr[column];
        }
    }

    /**
     * This class implements the weighted reservoir sampling algorithm for
     * sampling from a discrete probability distribution without repetitions.
     */
    static class WeightedReservoirSampling {

        private double[] weights;
        private double[] keys;

        /**
         * Initialization.
         *
         * @param weights double[] representing the weights to select the data
         * proportionally to.
         */
        public WeightedReservoirSampling(double[] weights) {
            this.weights = weights;
        }

        /**
         * This method obtains a random sample with no repetitions where each
         * item has a probability of being included in the sample that is
         * proportional to its provided weight.
         *
         * @param sampleSize Integer that is the size sample.
         * @return int[] that are the selected indexes.
         */
        public int[] getRandomSampleNoReps(int sampleSize) {
            // Obtain the normalized weight array.
            int numCases = weights.length;
            double[] weightsNormalized = Arrays.copyOf(weights, numCases);
            double weightSum = ArrayUtil.sum(weights);
            if (sampleSize > numCases || sampleSize < 0) {
                throw new IllegalArgumentException("Invalid sample size");
            } else if (sampleSize == numCases) {
                int[] trivialResult = new int[numCases];
                for (int i = 0; i < numCases; i++) {
                    trivialResult[i] = i;
                }
                return trivialResult;
            }
            if (!DataMineConstants.isAcceptableDouble(weightSum)
                    || DataMineConstants.isZero(weightSum)) {
                throw new IllegalArgumentException("Bad weights vector, unable "
                        + "to normalize.");
            }
            for (int i = 0; i < numCases; i++) {
                weightsNormalized[i] /= weightSum;
            }
            // Initialize the random number generator.
            Random randa = new Random();
            // Initialize the keys array.
            keys = new double[numCases];
            // Initialize the min-heap.
            PriorityQueue<Double> minHeap = new PriorityQueue<>(sampleSize);
            // Map to keep track of the indexes.
            HashMap<Double, Integer> keysToIndexes = new HashMap<>(numCases);
            // Generate the keys for all data points and keep track of the 
            // sampleSize top values.
            double minValue;
            for (int i = 0; i < numCases; i++) {
                if (DataMineConstants.isAcceptableDouble(weightsNormalized[i])
                        && DataMineConstants.isPositive(weightsNormalized[i])) {
                    keys[i] = Math.pow(randa.nextDouble(),
                            1 / weightsNormalized[i]);
                    // No keys should collide in any realistic scenario, since
                    // we are working with random double values here, but in
                    // the off-case that they do, we handle it. The following
                    // loop will probably never be executed.
                    while (keysToIndexes.containsKey(keys[i])) {
                        keys[i] = Math.pow(randa.nextDouble(),
                                1 / weightsNormalized[i]);
                    }
                    // It is important not to have the keys collide, as 
                    // insertions to the map would then overwrite the existing 
                    // key to index mappings.
                    if (minHeap.size() < sampleSize) {
                        minHeap.add(keys[i]);
                        // We will only add key-index pairs that are in the 
                        // heap to the map.
                        keysToIndexes.put(keys[i], i);
                    } else {
                        minValue = minHeap.peek();
                        if (keys[i] > minValue) {
                            // Removal of the current minimum value from the 
                            // heap, maximization process.
                            minHeap.poll();
                            if (keysToIndexes.containsKey(minValue)) {
                                keysToIndexes.remove(minValue);
                            }
                            // Insertion.
                            minHeap.add(keys[i]);
                            keysToIndexes.put(keys[i], i);
                        }
                    }
                }
            }
            // Initialize the result arrays.
            Double[] maxKeyVals = new Double[sampleSize];
            maxKeyVals = minHeap.toArray(maxKeyVals);
            int[] sampleIndexes = new int[sampleSize];
            for (int i = 0; i < sampleSize; i++) {
                sampleIndexes[i] = keysToIndexes.get(maxKeyVals[i]);
            }
            return sampleIndexes;
        }
    }
    
    public static void main(String[] args) throws Exception {
        float[] sW = {1.2f, 2.2f, 3.3f, 14.4f, 26f, 55f, 120f, 130f, 140f};
        SOPLUtil.printArray(getSampleWithReps(15, sW));
    }
}
