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
package distances.secondary.snd;

import data.neighbors.NeighborSetFinder;
import data.neighbors.SharedNeighborFinder;
import data.representation.DataInstance;
import data.representation.DataSet;
import distances.primary.CombinedMetric;
import java.io.Serializable;

/**
 * This class implements a secondary metric that is based on shared nearest
 * neighbors.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class SharedNeighborCalculator extends CombinedMetric
implements Serializable {
    
    private static final long serialVersionUID = 1L;

    public enum WeightingType {

        NONE, HUBNESS, BAD_HUBNESS, HUBNESS_INFORMATION, IMBALANCE;
    }
    private WeightingType usedWeighting = WeightingType.NONE;
    // The object that calculates the kNN set intersections.
    SharedNeighborFinder snf;

    /**
     *
     * @param snf SharedNeighborFinder for kNN set intersection calculations.
     * @param USED_WEIGHTING The weighting scheme employed in snf, for the
     * cross-validation to know how to retrain
     */
    public SharedNeighborCalculator(SharedNeighborFinder snf,
            WeightingType usedWeighting) {
        this.snf = snf;
        this.usedWeighting = usedWeighting;
        switch (usedWeighting) {
            case NONE:
                break;
            case HUBNESS:
                snf.obtainWeightsFromGeneralHubness();
                break;
            case BAD_HUBNESS:
                snf.obtainWeightsFromBadHubness();
                break;
            case HUBNESS_INFORMATION: {
                snf.obtainWeightsFromHubnessInformation();
                break;
            }
            case IMBALANCE:
                snf.obtainWeightsForClassImbalancedData();
                break;
            default:
                snf.obtainWeightsFromHubnessInformation();
                break;
        }
    }

    /**
     *
     * @param snf SharedNeighborFinder for kNN set intersection calculations.
     * @param theta The theta parameter for hubness-information-based snf.
     */
    public SharedNeighborCalculator(SharedNeighborFinder snf, float theta) {
        this.snf = snf;
        this.usedWeighting = WeightingType.HUBNESS_INFORMATION;
        snf.obtainWeightsFromHubnessInformation(theta);
    }

    /**
     * @return CombinedMetric object for primary distance calculations, used in
     * finding the original kNN sets.
     */
    public CombinedMetric getCombinedMetric() {
        return snf.getCombinedMetric();
    }

    /**
     *
     * @return DataSet that is being learned from.
     */
    public DataSet getData() {
        return snf.getData();
    }

    /**
     *
     * @return WeightingType that is the currently used weighting scheme.
     */
    public WeightingType getUsedWeighting() {
        return usedWeighting;
    }

    /**
     * Calculates the secondary shared-neighbor distance matrix.
     *
     * @param cmet CombinedMetric object for primary distance calculations.
     * @param numThreads Number of threads to use.
     * @return The secondary shared-neighbor distance matrix.
     * @throws Exception
     */
    public float[][] calculateDistMatrixMultThr(CombinedMetric cmet,
            int numThreads) throws Exception {
        DataSet dset = snf.getData();
        NeighborSetFinder nsf = snf.getNSF();
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
            threads[numThreads - 1] = new Thread(new DmCalculator(
                    dset, (numThreads - 1) * chunkSize, size - 1, distances,
                    cmet, nsf));
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
                        distances[i][j - i - 1] = dist(
                                dset.data.get(i), dset.data.get(j),
                                nsf.getKNeighbors()[i], nsf.getKNeighbors()[j]);
                    }
                }
            } catch (Exception e) {
            }
        }
    }

    @Override
    public float dist(DataInstance firstInstance, DataInstance secondInstance)
            throws Exception {
        float snCount = snf.countSharedNeighborsWithRespectToDataset(
                firstInstance, secondInstance);
        int k = snf.getSNK();
        return (k - snCount);
    }

    /**
     * Calculates the secondary SNN distance.
     *
     * @param firstInstance First DataInstance.
     * @param secondInstance Second DataInstance.
     * @param distsFirst Distances from the first point to other points in the
     * data.
     * @param distsSecond Distances from the second point to other points in the
     * data.
     * @return The float value that it the secondary SNN distance.
     * @throws Exception
     */
    public float dist(DataInstance firstInstance, DataInstance secondInstance,
            float[] distsFirst, float[] distsSecond) throws Exception {
        float snCount = snf.countSharedNeighborsWithRespectToDataset(
                firstInstance, secondInstance, distsFirst, distsSecond);
        int k = snf.getSNK();
        return (k - snCount);
    }

    /**
     * Calculates the secondary SNN distance.
     *
     * @param firstInstance First DataInstance.
     * @param secondInstance Second DataInstance.
     * @param distsFirst Distances from the first point to other points in the
     * data.
     * @param neighborsSecond The kNNs of the second instance.
     * @return The float value that it the secondary SNN distance.
     * @throws Exception
     */
    public float dist(DataInstance firstInstance, DataInstance secondInstance,
            float[] distsFirst, int[] neighborsSecond) throws Exception {
        float snCount = snf.countSharedNeighborsWithRespectToDataset(
                firstInstance, secondInstance, distsFirst, neighborsSecond);
        int k = snf.getSNK();
        return (k - snCount);
    }

    /**
     * Calculates the secondary SNN distance.
     *
     * @param firstInstance First DataInstance.
     * @param secondInstance Second DataInstance.
     * @param neighborsFirst The kNNs of the first instance.
     * @param neighborsSecond The kNNs of the second instance.
     * @return The float value that it the secondary SNN distance.
     * @throws Exception
     */
    public float dist(DataInstance firstInstance, DataInstance secondInstance,
            int[] neighborsFirst, int[] neighborsSecond) throws Exception {
        float snCount = snf.countSharedNeighborsWithRespectToDataset(
                firstInstance, secondInstance, neighborsFirst, neighborsSecond);
        int k = snf.getSNK();
        return (k - snCount);
    }

    /**
     * @return Integer value that is the current neighborhood size.
     */
    public int getK() {
        return snf.getSNK();
    }
}
