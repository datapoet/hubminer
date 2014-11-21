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
package sampling;

import data.representation.DataInstance;
import data.representation.DataSet;
import java.util.Arrays;
import java.util.Random;
import util.AuxSort;
import util.DataSetJoiner;

/**
 * A class that implements uniform data sampling.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class UniformSampler extends Sampler {

    /**
     * @param repetitions Boolean flag denoting whether to use repetitive
     * sampling or not.
     */
    public UniformSampler(boolean repetitions) {
        super(repetitions);
    }

    /**
     * Gets the indexes of a uniformly spread sample.
     *
     * @param popSize The size of the data.
     * @param sampleSize The size of the sample.
     * @return Integer array of indexes of a uniform sample.
     * @throws Exception
     */
    public static int[] getSample(int popSize, int sampleSize)
            throws Exception {
        float[] rVals = new float[popSize];
        Random randa = new Random();
        for (int i = 0; i < popSize; i++) {
            rVals[i] = randa.nextFloat();
        }
        int[] indexes = AuxSort.sortIndexedValue(rVals, true);
        int[] result = Arrays.copyOf(indexes, Math.min(sampleSize, popSize));
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
        Random randa = new Random();
        if ((!getRepetitions()) && sampleSize <= dset.size()) {
            // In the case where there are no repetitions, we can essentially
            // just assign a uniform random value to each array member and then
            // sort and take the first sampleSize positions.
            int[] selIndexes = getSample(dset.size(), sampleSize);
            for (int index : selIndexes) {
                DataInstance instanceCopy = dset.data.get(index).copy();
                sample.addDataInstance(instanceCopy);
                instanceCopy.embedInDataset(sample);
            }
        } else {
            // If repetitions are allowed, we just go and sample iteratively.
            int candidate;
            for (int i = 0; i < sampleSize; i++) {
                candidate = randa.nextInt(dset.size());
                sample.addDataInstance(dset.data.get(candidate).copy());
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
}
