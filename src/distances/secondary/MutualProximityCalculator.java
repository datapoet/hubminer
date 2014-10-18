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
package distances.secondary;

import data.neighbors.NeighborSetFinder;
import data.representation.DataInstance;
import data.representation.DataSet;
import distances.primary.CombinedMetric;
import java.io.Serializable;
import java.util.Arrays;
import probability.NormalDistributionCalculator;
import sampling.UniformSampler;

/**
 * This classs implements the mutual proximity similarity measure. The basic
 * formula determines the probability that x is a neighbor of y and y is a
 * neighbor of x. This class methods actually return 1 - MP, as distance
 * matrices are the default (and not similarity matrices) throughout the code.
 * The details of the procedure can be found in the following research paper:
 * "Using Mutual Proximity to Improve Content-Based Audio Similarity" that was
 * published in 2011 by Dominik Schnitzer, Arthur Flexer, Markus Schedl and
 * Gerhard Widmer.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class MutualProximityCalculator extends CombinedMetric
implements Serializable {

    private static final long serialVersionUID = 1L;
    // Means and standard deviations of distances, for the model.
    double[] distMeans;
    double[] distStDevs;
    // Primary distance matrix.
    float[][] dMatPrimary;
    DataSet dset;
    // CombinedMetric object for distance calculations.
    CombinedMetric cmet;

    /**
     * Initialization of the model.
     *
     * @param dMatPrimary Primary distance matrix.
     * @param dset DataSet object.
     * @param cmet CombinedMetric object for distance calculations.
     */
    public MutualProximityCalculator(float[][] dMatPrimary, DataSet dset,
            CombinedMetric cmet) {
        this.dMatPrimary = dMatPrimary;
        this.dset = dset;
        this.cmet = cmet;
        int other;
        if (dMatPrimary != null && dMatPrimary.length > 0) {
            distMeans = new double[dMatPrimary.length];
            distStDevs = new double[dMatPrimary.length];
            float[] numIncluded = new float[dMatPrimary.length];
            // Calculate the means of distances from each point to other points.
            for (int i = 0; i < dMatPrimary.length; i++) {
                for (int j = 0; j < dMatPrimary[i].length; j++) {
                    if (numIncluded[i] == 0) {
                        numIncluded[i] = 1;
                        distMeans[i] = dMatPrimary[i][j];
                    } else {
                        numIncluded[i]++;
                        distMeans[i] = distMeans[i]
                                * ((numIncluded[i] - 1) / numIncluded[i])
                                + dMatPrimary[i][j] * (1 / numIncluded[i]);
                    }
                    other = i + j + 1;
                    if (numIncluded[other] == 0) {
                        numIncluded[other] = 1;
                        distMeans[other] = dMatPrimary[i][j];
                    } else {
                        numIncluded[other]++;
                        distMeans[other] = distMeans[other]
                                * ((numIncluded[other] - 1)
                                / numIncluded[other])
                                + dMatPrimary[i][j] * (1 / numIncluded[other]);
                    }
                }
            }
            Arrays.fill(numIncluded, 0);
            // Calculate the standard deviation of distances from each point to
            // other points.
            for (int i = 0; i < dMatPrimary.length; i++) {
                for (int j = 0; j < dMatPrimary[i].length; j++) {
                    if (numIncluded[i] == 0) {
                        numIncluded[i] = 1;
                        distStDevs[i] = (dMatPrimary[i][j] - distMeans[i])
                                * (dMatPrimary[i][j] - distMeans[i]);
                    } else {
                        numIncluded[i]++;
                        distStDevs[i] = distStDevs[i] * ((numIncluded[i] - 1)
                                / numIncluded[i]) + (dMatPrimary[i][j]
                                - distMeans[i]) * (dMatPrimary[i][j]
                                - distMeans[i]) * (1 / numIncluded[i]);
                    }
                    other = i + j + 1;
                    if (numIncluded[other] == 0) {
                        numIncluded[other] = 1;
                        distStDevs[other] = (dMatPrimary[i][j]
                                - distMeans[other]) * (dMatPrimary[i][j]
                                - distMeans[other]);
                    } else {
                        numIncluded[other]++;
                        distStDevs[other] = distStDevs[other]
                                * ((numIncluded[other] - 1)
                                / numIncluded[other])
                                + (dMatPrimary[i][j] - distMeans[other])
                                * (dMatPrimary[i][j] - distMeans[other])
                                * (1 / numIncluded[other]);
                    }
                }
            }
            for (int i = 0; i < distStDevs.length; i++) {
                distStDevs[i] = Math.sqrt(distStDevs[i]);
            }
        }
    }

    /**
     * Calculate the secondary distance matrix on the data in a multi-threaded
     * way.
     *
     * @param nsf NeighborSetFinder object.
     * @param numThreads Number of threads to use.
     * @return float[][] representing the upper triangular secondary MP distance
     * matrix.
     * @throws Exception
     */
    public float[][] calculateSecondaryDistMatrixMultThr(NeighborSetFinder nsf,
            int numThreads) throws Exception {
        if (dset.isEmpty()) {
            return null;
        } else {
            float[][] distances = new float[dset.size()][];
            int size = dset.size();
            int chunkSize = size / numThreads;
            Thread[] threads = new Thread[numThreads];
            for (int i = 0; i < numThreads - 1; i++) {
                threads[i] = new Thread(new DmCalculator(
                        dset, i * chunkSize, (i + 1) * chunkSize - 1,
                        distances, cmet, nsf));
                threads[i].start();
            }
            threads[numThreads - 1] = new Thread(new DmCalculator(dset,
                    (numThreads - 1) * chunkSize, size - 1,
                    distances, cmet, nsf));
            threads[numThreads - 1].start();
            for (int i = 0; i < numThreads; i++) {
                if (threads[i] != null) {
                    try {
                        threads[i].join();
                    } catch (Throwable t) {
                    }
                }
            }
            return distances;
        }
    }

    /**
     * Calculate the secondary distance matrix on the data in a multi-threaded
     * way, with sampling for speed-up.
     *
     * @param numThreads Integer that is the number of threads to use.
     * @param samplingSize Integer that is the size of the sample.
     * @return float[][] representing the upper triangular secondary MP distance
     * matrix.
     * @throws Exception
     */
    public float[][] calculateSecondaryDistMatrixMultThrFast(
            int numThreads, int samplingSize) throws Exception {
        if (dset.isEmpty()) {
            return null;
        } else {
            samplingSize = Math.min(samplingSize, (int) (dset.size() * 0.8f));
            float[][] distances = new float[dset.size()][];
            int size = dset.size();
            int chunkSize = size / numThreads;
            Thread[] threads = new Thread[numThreads];
            for (int i = 0; i < numThreads - 1; i++) {
                threads[i] = new Thread(new DmCalculatorFast(
                        dset, i * chunkSize, (i + 1) * chunkSize - 1,
                        distances, cmet, samplingSize));
                threads[i].start();
            }
            threads[numThreads - 1] = new Thread(new DmCalculatorFast(dset,
                    (numThreads - 1) * chunkSize, size - 1,
                    distances, cmet, samplingSize));
            threads[numThreads - 1].start();
            for (int i = 0; i < numThreads; i++) {
                if (threads[i] != null) {
                    try {
                        threads[i].join();
                    } catch (Throwable t) {
                    }
                }
            }
            return distances;
        }
    }

    /**
     * Worker class for multi-threaded calculations of the distance matrix.
     */
    class DmCalculator implements Runnable {

        int startRow;
        int endRow;
        float[][] distances;
        CombinedMetric cmet;
        DataSet dset;
        NeighborSetFinder nsf;

        /**
         * The range is inclusive.
         *
         * @param dset DataSet object.
         * @param startRow Index of the start row.
         * @param endRow Index of the end row.
         * @param distances The primary distance matrix.
         * @param cmet The CombinedMetric object used for primary distances.
         * @param nsf The NeighborSetFinder object.
         */
        public DmCalculator(DataSet dset, int startRow, int endRow,
                float[][] distances, CombinedMetric cmet,
                NeighborSetFinder nsf) {
            this.startRow = startRow;
            this.endRow = endRow;
            this.distances = distances;
            this.cmet = cmet;
            this.dset = dset;
            this.nsf = nsf;
        }

        @Override
        public void run() {
            try {
                for (int i = startRow; i <= endRow; i++) {
                    distances[i] = new float[dset.size() - i - 1];
                    for (int j = i + 1; j < dset.size(); j++) {
                        distances[i][j - i - 1] = dist(dset.data.get(i),
                                dset.data.get(j), nsf.getKDistances()[i],
                                nsf.getKDistances()[j]);
                    }
                }
            } catch (Exception e) {
            }
        }
    }

    /**
     * Worker class for multi-threaded calculations of the distance matrix.
     */
    class DmCalculatorFast implements Runnable {

        int startRow;
        int endRow;
        float[][] distances;
        CombinedMetric cmet;
        DataSet dset;
        NeighborSetFinder nsf;
        int sampleSize;

        /**
         * The range is inclusive.
         *
         * @param dset DataSet object.
         * @param startRow Index of the start row.
         * @param endRow Index of the end row.
         * @param distances The primary distance matrix.
         * @param cmet The CombinedMetric object used for primary distances.
         * @param nsf The NeighborSetFinder object.
         * @param sampleSize Integer that is the sampling size.
         */
        public DmCalculatorFast(DataSet dset, int startRow, int endRow,
                float[][] distances, CombinedMetric cmet,
                int sampleSize) {
            this.startRow = startRow;
            this.endRow = endRow;
            this.distances = distances;
            this.cmet = cmet;
            this.dset = dset;
            this.sampleSize = sampleSize;
        }

        /**
         * Gets a distance sample.
         *
         * @param index Integer index to get the distance sample for.
         * @return float[] sample of distances.
         */
        private float[] sampleDists(int index) {
            int size = dMatPrimary.length;
            int[] sampleIndexes = null;
            try {
                sampleIndexes = UniformSampler.getSample(size, sampleSize);
            } catch (Exception e) {
            }
            float[] dSample = new float[size];
            for (int i = 0; i < sampleSize; i++) {
                int maxIndex = Math.max(sampleIndexes[i], index);
                int minIndex = Math.min(sampleIndexes[i], index);
                if (maxIndex != minIndex) {
                    dSample[i] = dMatPrimary[minIndex][maxIndex - minIndex - 1];
                } else {
                    dSample[i] = 0;
                }
            }
            return dSample;
        }

        @Override
        public void run() {
            try {
                for (int i = startRow; i <= endRow; i++) {
                    distances[i] = new float[dset.size() - i - 1];
                    for (int j = i + 1; j < dset.size(); j++) {
                        distances[i][j - i - 1] = distFast(
                                dMatPrimary[i][j - i - 1],
                                sampleDists(i),
                                sampleDists(j));
                    }
                }
            } catch (Exception e) {
            }
        }
    }

    /**
     * Calculate the secondary distance matrix. This method is single-threaded.
     *
     * @return The secondary distance matrix.
     */
    public float[][] getTransformedDMat() {
        float mp;
        if (dMatPrimary != null && dMatPrimary.length > 0) {
            float[][] dMatSecondary = new float[dMatPrimary.length][];
            for (int i = 0; i < dMatPrimary.length; i++) {
                dMatSecondary[i] = new float[dMatPrimary[i].length];
                for (int j = 0; j < dMatSecondary[i].length; j++) {
                    mp = (float) ((1
                            - NormalDistributionCalculator.PhiCumulative(
                            dMatPrimary[i][j], distMeans[i],
                            distStDevs[i])) * (1
                            - NormalDistributionCalculator.PhiCumulative(
                            dMatPrimary[i][j], distMeans[i + j + 1],
                            distStDevs[i + j + 1])));
                    dMatSecondary[i][j] = 1 - mp;
                }
            }
            return dMatSecondary;
        } else {
            return null;
        }
    }

    /**
     * Returns a secondary similarity matrix where the entries are the mutual
     * proximity scores between the points.
     *
     * @return The secondary similarity matrix.
     */
    public float[][] transformToSimilarityMat() {
        float mp;
        if (dMatPrimary != null && dMatPrimary.length > 0) {
            float[][] sMatSecondary = new float[dMatPrimary.length][];
            for (int i = 0; i < dMatPrimary.length; i++) {
                sMatSecondary[i] = new float[dMatPrimary[i].length];
                for (int j = 0; j < sMatSecondary[i].length; j++) {
                    mp = (float) ((1
                            - NormalDistributionCalculator.PhiCumulative(
                            dMatPrimary[i][j], distMeans[i], distStDevs[i]))
                            * (1 - NormalDistributionCalculator.PhiCumulative(
                            dMatPrimary[i][j], distMeans[i + j + 1],
                            distStDevs[i + j + 1])));
                    sMatSecondary[i][j] = mp;
                }
            }
            return sMatSecondary;
        } else {
            return null;
        }
    }

    @Override
    public float dist(DataInstance firstInstance,
            DataInstance secondInstance) throws Exception {
        if (dMatPrimary != null && dMatPrimary.length > 0 && cmet != null
                && dset != null && !dset.isEmpty()) {
            // First we get the basic distance statistics.
            double distMeanFirst = 0;
            double distMeanSecond = 0;
            double distStDevFirst = 0;
            double distStDevSecond = 0;
            float[] distsFirst = new float[dMatPrimary.length];
            float[] distsSecond = new float[dMatPrimary.length];
            for (int i = 0; i < distsFirst.length; i++) {
                distsFirst[i] = cmet.dist(firstInstance, dset.getInstance(i));
                distMeanFirst = ((float) i / (float) (i + 1))
                        * distMeanFirst + (1f / (float) (i + 1))
                        * distsFirst[i];
                distsSecond[i] = cmet.dist(secondInstance, dset.getInstance(i));
                distMeanSecond = ((float) i / (float) (i + 1))
                        * distMeanSecond + (1f / (float) (i + 1))
                        * distsSecond[i];
            }
            for (int i = 0; i < distsFirst.length; i++) {
                distStDevFirst = ((float) i / (float) (i + 1))
                        * distStDevFirst + (1f / (float) (i + 1))
                        * (distsFirst[i]
                        - distMeanFirst) * (distsFirst[i] - distMeanFirst);
                distStDevSecond = ((float) i / (float) (i + 1))
                        * distStDevSecond + (1f / (float) (i + 1))
                        * (distsSecond[i] - distMeanSecond) * (distsSecond[i]
                        - distMeanSecond);
            }
            distStDevFirst = Math.sqrt(distStDevFirst);
            distStDevSecond = Math.sqrt(distStDevSecond);
            float distFS = cmet.dist(firstInstance, secondInstance);
            float mp = (float) ((1
                    - NormalDistributionCalculator.PhiCumulative(
                    distFS, distMeanFirst, distStDevFirst))
                    * (1 - NormalDistributionCalculator.PhiCumulative(distFS,
                    distMeanSecond, distStDevSecond)));
            return 1 - mp;
        } else {
            return Float.MAX_VALUE;
        }
    }

    /**
     * Calculates the distance based on mutual proximity.
     *
     * @param firstInstance First DataInstance.
     * @param secondInstance Second DataInstance.
     * @param distsFirst Distances from the first point to other points in the
     * data, used to build a distance model.
     * @param distsSecond Distances from the second point to other points in the
     * data, used to build a distance model.
     * @return Float value that is the distance based on mutual proximity.
     * @throws Exception
     */
    public float dist(DataInstance firstInstance, DataInstance secondInstance,
            float[] distsFirst, float[] distsSecond) throws Exception {
        if (dMatPrimary != null && dMatPrimary.length > 0 && cmet != null
                && dset != null && !dset.isEmpty()) {
            // First we get the basic distance statistics.
            double distMeanFirst = 0;
            double distMeanSecond = 0;
            double distStDevFirst = 0;
            double distStDevSecond = 0;
            for (int i = 0; i < distsFirst.length; i++) {
                distMeanFirst = ((float) i / (float) (i + 1))
                        * distMeanFirst + (1f / (float) (i + 1))
                        * distsFirst[i];
            }
            for (int i = 0; i < distsFirst.length; i++) {
                distStDevFirst = ((float) i / (float) (i + 1))
                        * distStDevFirst + (1f / (float) (i + 1))
                        * (distsFirst[i]
                        - distMeanFirst) * (distsFirst[i] - distMeanFirst);
            }
            for (int i = 0; i < distsSecond.length; i++) {
                distMeanSecond = ((float) i / (float) (i + 1))
                        * distMeanSecond + (1f / (float) (i + 1))
                        * distsSecond[i];
            }
            for (int i = 0; i < distsSecond.length; i++) {
                distStDevSecond = ((float) i / (float) (i + 1))
                        * distStDevSecond + (1f / (float) (i + 1))
                        * (distsSecond[i] - distMeanSecond) * (distsSecond[i]
                        - distMeanSecond);
            }
            distStDevFirst = Math.sqrt(distStDevFirst);
            distStDevSecond = Math.sqrt(distStDevSecond);
            float distFS = cmet.dist(firstInstance, secondInstance);
            float mp = (float) ((1 - NormalDistributionCalculator.PhiCumulative(
                    distFS, distMeanFirst, distStDevFirst)) * (1
                    - NormalDistributionCalculator.PhiCumulative(distFS,
                    distMeanSecond, distStDevSecond)));
            return 1 - mp;
        } else {
            return Float.MAX_VALUE;
        }
    }

    /**
     * Calculates the distance based on mutual proximity.
     *
     * @param distance Float that is the distance between the instances.
     * @param distsFirst Distances from the first point to other points in the
     * data, used to build a distance model.
     * @param distsSecond Distances from the second point to other points in the
     * data, used to build a distance model.
     * @param Integer that is the sample size.
     * @return Float value that is the distance based on mutual proximity.
     * @throws Exception
     */
    public static float distFast(float distance, float[] distsFirst,
            float[] distsSecond, int sampleSize) throws Exception {
        // First we get the basic distance statistics.
        double distMeanFirst = 0;
        double distMeanSecond = 0;
        double distStDevFirst = 0;
        double distStDevSecond = 0;
        int numFirstSample = Math.min(sampleSize, distsFirst.length);
        int numSecondSample = Math.min(sampleSize, distsSecond.length);
        int[] fSample = UniformSampler.getSample(distsFirst.length,
                numFirstSample);
        int[] sSample = UniformSampler.getSample(distsSecond.length,
                numSecondSample);
        for (int i = 0; i < numFirstSample; i++) {
            distMeanFirst = ((float) i / (float) (i + 1))
                    * distMeanFirst + (1f / (float) (i + 1)) * distsFirst[
                    fSample[i]];
        }
        for (int i = 0; i < numFirstSample; i++) {
            distStDevFirst = ((float) i / (float) (i + 1))
                    * distStDevFirst + (1f / (float) (i + 1)) * (distsFirst[
                    fSample[i]]
                    - distMeanFirst) * (distsFirst[fSample[i]] - distMeanFirst);
        }
        for (int i = 0; i < numSecondSample; i++) {
            distMeanSecond = ((float) i / (float) (i + 1))
                    * distMeanSecond + (1f / (float) (i + 1)) * distsSecond[
                    sSample[i]];
        }
        for (int i = 0; i < numSecondSample; i++) {
            distStDevSecond = ((float) i / (float) (i + 1))
                    * distStDevSecond + (1f / (float) (i + 1))
                    * (distsSecond[sSample[i]] - distMeanSecond) * (distsSecond[
                    sSample[i]] - distMeanSecond);
        }
        distStDevFirst = Math.sqrt(distStDevFirst);
        distStDevSecond = Math.sqrt(distStDevSecond);
        float mp = (float) ((1 - NormalDistributionCalculator.PhiCumulative(
                distance, distMeanFirst, distStDevFirst)) * (1
                - NormalDistributionCalculator.PhiCumulative(distance,
                distMeanSecond, distStDevSecond)));
        return 1 - mp;
    }

    /**
     * Calculates the distance based on mutual proximity.
     *
     * @param distance Float that is the distance between the instances.
     * @param distsFirst Distances from the first point to other points in the
     * data, used to build a distance model.
     * @param distsSecond Distances from the second point to other points in the
     * data, used to build a distance model.
     * @return Float value that is the distance based on mutual proximity.
     * @throws Exception
     */
    public static float distFast(float distance, float[] distsFirst,
            float[] distsSecond) throws Exception {
        // First we get the basic distance statistics.
        double distMeanFirst = 0;
        double distMeanSecond = 0;
        double distStDevFirst = 0;
        double distStDevSecond = 0;
        int sampleSize = Math.min(distsFirst.length, distsSecond.length);
        for (int i = 0; i < sampleSize; i++) {
            distMeanFirst += distsFirst[i];
            distMeanSecond += distsSecond[i];
        }
        distMeanFirst /= sampleSize;
        distMeanSecond /= sampleSize;
        for (int i = 0; i < sampleSize; i++) {
            distStDevFirst += (distsFirst[i] - distMeanFirst)
                    * (distsFirst[i] - distMeanFirst);
            distStDevSecond += (distsSecond[i] - distMeanSecond)
                    * (distsSecond[i] - distMeanSecond);
        }
        distStDevFirst /= sampleSize;
        distStDevSecond /= sampleSize;
        distStDevFirst = Math.sqrt(distStDevFirst);
        distStDevSecond = Math.sqrt(distStDevSecond);
        float mp = (float) ((1 - NormalDistributionCalculator.PhiCumulative(
                distance, distMeanFirst, distStDevFirst)) * (1
                - NormalDistributionCalculator.PhiCumulative(distance,
                distMeanSecond, distStDevSecond)));
        return 1 - mp;
    }

    /**
     *
     * @param firstInstance First DataInstance.
     * @param secondInstance Second DataInstance.
     * @param distMeanFirst Mean values of the distance from the first point.
     * @param distMeanSecond Mean values of the distance from the second point.
     * @param distStDevFirst Standard deviation of distances from the first
     * point.
     * @param distStDevSecond Standard deviation of distances from the second
     * point.
     * @return Float value that is the distance based on mutual proximity.
     * @throws Exception
     */
    public float dist(DataInstance firstInstance, DataInstance secondInstance,
            float distMeanFirst, float distMeanSecond, float distStDevFirst,
            float distStDevSecond) throws Exception {
        if (cmet == null) {
            return Float.MAX_VALUE;
        }
        float distFS = cmet.dist(firstInstance, secondInstance);
        float mp = (float) ((1 - NormalDistributionCalculator.PhiCumulative(
                distFS, distMeanFirst, distStDevFirst)) * (1
                - NormalDistributionCalculator.PhiCumulative(distFS,
                distMeanSecond, distStDevSecond)));
        return 1 - mp;
    }
}
