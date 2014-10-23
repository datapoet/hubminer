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
import distances.primary.CombinedMetric;
import java.io.Serializable;

/**
 * This class implements a simple metric learning procedure for high-dimensional
 * data that was proposed in a paper by Zelnik-Manor and Perona in 2005. It
 * calculates local affinities, but these methods return the distance instead,
 * as 1 - similarity, for consistency.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class LocalScalingCalculator extends CombinedMetric
implements Serializable {

    private static final long serialVersionUID = 1L;
    // Object for calculating the kNN sets.
    NeighborSetFinder nsf = null;

    /**
     * @param nsf NeighborSetFinder object for calculating kNN sets.
     */
    public LocalScalingCalculator(NeighborSetFinder nsf) {
        this.nsf = nsf;
    }
    
    @Override
    public String toString() {
        if (nsf != null) {
            return "LocalScaling, k:" + nsf.getCurrK();
        } else {
            return "LocalScaling";
        }
    }

    /**
     * Transforms a primary into secondary distance matrix, based on kNN
     * information. The new matrix is returned and the original is left intact.
     *
     * @return A secondary distance matrix, after local scaling.
     */
    public float[][] getTransformedDMatFromNSFPrimaryDMat() {
        float[][] dMatPrimary = nsf.getDistances();
        float[][] dMatSecondary = new float[dMatPrimary.length][];
        float[][] kDists = nsf.getKDistances();
        int k = nsf.getCurrK();
        for (int i = 0; i < dMatPrimary.length; i++) {
            dMatSecondary[i] = new float[dMatPrimary[i].length];
            for (int j = 0; j < dMatSecondary[i].length; j++) {
                dMatSecondary[i][j] = 1 - (float) Math.exp(
                        -dMatPrimary[i][j] * dMatPrimary[i][j]
                        / (kDists[i][k - 1] * kDists[i + j + 1][k - 1]));
            }
        }
        return dMatSecondary;
    }

    @Override
    public float dist(DataInstance firstInstance,
            DataInstance secondInstance) throws Exception {
        if (nsf != null) {
            CombinedMetric cmet = nsf.getCombinedMetric();
            // Find the indexes of neighbor points.
            int[] knsFirst = NeighborSetFinder.getIndexesOfNeighbors(
                    nsf.getDataSet(), firstInstance,
                    nsf.getCurrK(), cmet);
            int[] knsSecond = NeighborSetFinder.getIndexesOfNeighbors(
                    nsf.getDataSet(), firstInstance,
                    nsf.getCurrK(), cmet);
            float affinity = (float) Math.exp(-Math.pow(
                    cmet.dist(firstInstance, secondInstance), 2)
                    / (cmet.dist(firstInstance, nsf.getDataSet().
                    getInstance(knsFirst[knsFirst.length - 1]))
                    * cmet.dist(secondInstance, nsf.getDataSet().
                    getInstance(knsSecond[knsSecond.length - 1]))));
            return 1 - affinity;
        } else {
            return Float.MAX_VALUE;
        }
    }

    /**
     * Calculates the scaled distance.
     *
     * @param firstInstance First DataInstance.
     * @param secondInstance Second DataInstance.
     * @param distsFirst Distances from the first DataInstance to other points
     * in the DataSet that the NeighborSetFinder object models.
     * @param distsSecond Distances from the second DataInstance to other points
     * in the DataSet that the NeighborSetFinder object models.
     * @return Float value that is the scaled distance between the points.
     * @throws Exception
     */
    public float dist(DataInstance firstInstance, DataInstance secondInstance,
            float[] distsFirst, float[] distsSecond) throws Exception {
        if (nsf != null) {
            CombinedMetric cmet = nsf.getCombinedMetric();
            // Find the indexes of neighbor points.
            int[] knsFirst = NeighborSetFinder.getIndexesOfNeighbors(
                    nsf.getDataSet(), firstInstance, nsf.getCurrK(),
                    distsFirst);
            int[] knsSecond = NeighborSetFinder.getIndexesOfNeighbors(
                    nsf.getDataSet(), firstInstance, nsf.getCurrK(),
                    distsSecond);
            float affinity = (float) Math.exp(-Math.pow(
                    cmet.dist(firstInstance, secondInstance), 2)
                    / (distsFirst[knsFirst[knsFirst.length - 1]]
                    * distsSecond[knsSecond[knsSecond.length - 1]]));
            return 1 - affinity;
        } else {
            return Float.MAX_VALUE;
        }
    }

    /**
     * Calculates the scaled distance.
     *
     * @param firstInstance First DataInstance.
     * @param secondDistance Second DataInstance.
     * @param distsFirst Distances from the first DataInstance to other points
     * in the DataSet that the NeighborSetFinder object models.
     * @param knsSecond Integer array of indexes to the k-nearest neighbors of
     * the second DataInstance object.
     * @return Float value that is the scaled distance between the points.
     * @throws Exception
     */
    public float dist(DataInstance firstInstance, DataInstance secondDistance,
            float[] distsFirst, int[] knsSecond) throws Exception {
        if (nsf != null) {
            CombinedMetric cmet = nsf.getCombinedMetric();
            int[] knsFirst = NeighborSetFinder.getIndexesOfNeighbors(
                    nsf.getDataSet(), firstInstance, nsf.getCurrK(),
                    cmet);
            float affinity = (float) Math.exp(-Math.pow(
                    cmet.dist(firstInstance, secondDistance), 2)
                    / (distsFirst[knsFirst[knsFirst.length - 1]]
                    * cmet.dist(secondDistance, nsf.getDataSet().
                    getInstance(knsSecond[knsSecond.length - 1]))));
            return 1 - affinity;
        } else {
            return Float.MAX_VALUE;
        }
    }

    /**
     * Calculates the scaled distance.
     *
     * @param firstInstance First DataInstance.
     * @param secondInstance Second DataInstance.
     * @param knsFirst Integer array of indexes to the k-nearest neighbors of
     * the first DataInstance object.
     * @param knsSecond Integer array of indexes to the k-nearest neighbors of
     * the second DataInstance object.
     * @return Float value that is the scaled distance between the points.
     * @throws Exception
     */
    public float dist(DataInstance firstInstance, DataInstance secondInstance,
            int[] knsFirst, int[] knsSecond) throws Exception {
        if (nsf != null) {
            CombinedMetric cmet = nsf.getCombinedMetric();
            float affinity = (float) Math.exp(-Math.pow(
                    cmet.dist(firstInstance, secondInstance), 2)
                    / (cmet.dist(firstInstance, nsf.getDataSet().
                    getInstance(knsFirst[knsFirst.length - 1]))
                    * cmet.dist(secondInstance, nsf.getDataSet().
                    getInstance(knsSecond[knsSecond.length - 1]))));
            return 1 - affinity;
        } else {
            return Float.MAX_VALUE;
        }
    }

    /**
     * Calculates the scaled distance.
     *
     * @param firstInstance First DataInstance.
     * @param secondInstance Second DataInstance.
     * @param knsFirst Integer array of indexes to the k-nearest neighbors of
     * the first DataInstance object.
     * @param knsSecond Integer array of indexes to the k-nearest neighbors of
     * the second DataInstance object.
     * @param distsFirst Distances from the first DataInstance to other points
     * in the DataSet that the NeighborSetFinder object models.
     * @param distsSecond Distances from the second DataInstance to other points
     * in the DataSet that the NeighborSetFinder object models.
     * @return Float value that is the scaled distance between the points.
     * @throws Exception
     */
    public float dist(DataInstance firstInstance, DataInstance secondInstance,
            int[] knsFirst, int[] knsSecond, float[] distsFirst,
            float[] distsSecond) throws Exception {
        if (nsf != null) {
            CombinedMetric cmet = nsf.getCombinedMetric();
            float affinity = (float) Math.exp(-Math.pow(
                    cmet.dist(firstInstance, secondInstance), 2)
                    / (distsFirst[knsFirst[knsFirst.length - 1]]
                    * distsSecond[knsSecond[knsSecond.length - 1]]));
            return 1 - affinity;
        } else {
            return Float.MAX_VALUE;
        }
    }

    /**
     * Calculates the scaled distance.
     *
     * @param firstInstance First DataInstance.
     * @param secondInstance Second DataInstance.
     * @param kDistsFirst Distances from the first DataInstance to its kNN-s.
     * @param kDistsSecond Distances from the second DataInstance to its kNN-s.
     * @return Float value that is the scaled distance between the points.
     * @throws Exception
     */
    public float distFromKDists(DataInstance firstInstance,
            DataInstance secondInstance, float[] kDistsFirst,
            float[] kDistsSecond) throws Exception {
        if (nsf != null) {
            CombinedMetric cmet = nsf.getCombinedMetric();
            float affinity = (float) Math.exp(-Math.pow(
                    cmet.dist(firstInstance, secondInstance), 2)
                    / (kDistsFirst[kDistsFirst.length - 1]
                    * kDistsSecond[kDistsSecond.length - 1]));
            return 1 - affinity;
        } else {
            return Float.MAX_VALUE;
        }
    }

    /**
     * Calculates the scaled distance.
     *
     * @param firstInstance First DataInstance.
     * @param secondInstance Second DataInstance.
     * @param knDistFirst Distance to the k-th NN of the first instance.
     * @param knDistSecond Distance to the k-th NN of the second instance.
     * @return Float value that is the scaled distance between the points.
     * @throws Exception
     */
    public float dist(DataInstance firstInstance, DataInstance secondInstance,
            float knDistFirst, float knDistSecond) throws Exception {
        if (nsf != null) {
            CombinedMetric cmet = nsf.getCombinedMetric();
            float affinity = (float) Math.exp(-Math.pow(
                    cmet.dist(firstInstance, secondInstance), 2)
                    / (knDistFirst * knDistSecond));
            return 1 - affinity;
        } else {
            return Float.MAX_VALUE;
        }
    }

    /**
     * @return Integer that is the current neighborhood size.
     */
    public int getK() {
        return nsf.getCurrK();
    }
}
