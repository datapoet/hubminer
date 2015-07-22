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

import distances.primary.CombinedMetric;
import data.neighbors.NeighborSetFinder;
import data.representation.DataInstance;
import java.io.Serializable;

/**
 * This class implements a non-iterative contextual dissimilarity measure that
 * was proposed by Jegou et al. in (2007).
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class NICDMCalculator extends CombinedMetric implements Serializable {
    
    private static final long serialVersionUID = 1L;

    private NeighborSetFinder nsf = null;
    private float[] kDistAvgs;

    /**
     * @param nsf NeighborSetFinder object for kNN set handling.
     */
    public NICDMCalculator(NeighborSetFinder nsf) {
        this.nsf = nsf;
        kDistAvgs = nsf.getAvgDistToNeighbors(getK());
    }
    
    @Override
    public String toString() {
        if (nsf != null) {
            return "NICDM, k:" + nsf.getCurrK();
        } else {
            return "NICDM";
        }
    }

    /**
     * Calculates the secondary distance matrix based on NICDM.
     *
     * @return The 2D float array corresponding to the secondary NICDM distance
     * matrix.
     */
    public float[][] getTransformedDMatFromNSFPrimaryDMat() {
        float[][] dMatPrimary = nsf.getDistances();
        float[][] dMatSecondary = new float[dMatPrimary.length][];
        for (int i = 0; i < dMatPrimary.length; i++) {
            dMatSecondary[i] = new float[dMatPrimary[i].length];
            for (int j = 0; j < dMatSecondary[i].length; j++) {
                dMatSecondary[i][j] = dMatPrimary[i][j]
                        / (float) (Math.sqrt(kDistAvgs[i] *
                        kDistAvgs[i + j + 1]));
            }
        }
        return dMatSecondary;
    }

    @Override
    public float dist(DataInstance firstInstance, DataInstance secondInstance)
            throws Exception {
        if (nsf != null) {
            CombinedMetric cmet = nsf.getCombinedMetric();
            // First get the kNN sets.
            int[] knsFirst = NeighborSetFinder.getIndexesOfNeighbors(
                    nsf.getDataSet(), firstInstance, nsf.getCurrK(),
                    cmet);
            int[] knsSecond = NeighborSetFinder.getIndexesOfNeighbors(
                    nsf.getDataSet(), firstInstance, nsf.getCurrK(),
                    cmet);
            float kDistsFirstAvg = 0;
            float kDistsSecondAvg = 0;
            // Calculate the average distance from each instance to their
            // k-nearest neighbors.
            for (int i = 0; i < nsf.getCurrK(); i++) {
                kDistsFirstAvg += cmet.dist(firstInstance,
                        nsf.getDataSet().getInstance(knsFirst[i]));
                kDistsSecondAvg += cmet.dist(secondInstance,
                        nsf.getDataSet().getInstance(knsSecond[i]));
            }
            kDistsFirstAvg /= nsf.getCurrK();
            kDistsSecondAvg /= nsf.getCurrK();
            // Scale by the roow of the product of the kNN distance averages.
            return cmet.dist(firstInstance, secondInstance)
                    / (float) (Math.sqrt(kDistsFirstAvg * kDistsSecondAvg));
        } else {
            return Float.MAX_VALUE;
        }
    }

    /**
     * Calculates the secondary NICDM distance.
     *
     * @param firstInstance First DataInstance.
     * @param secondInstance Second DataInstance.
     * @param distsFirst Distances from the first point to other points in the
     * data.
     * @param distsSecond Distances from the second point to other points in the
     * data.
     * @return The float value that it the secondary NICDM distance.
     * @throws Exception
     */
    public float dist(DataInstance firstInstance, DataInstance secondInstance,
            float[] distsFirst, float[] distsSecond) throws Exception {
        if (nsf != null) {
            CombinedMetric cmet = nsf.getCombinedMetric();
            int[] knsFirst = NeighborSetFinder.getIndexesOfNeighbors(
                    nsf.getDataSet(), firstInstance, nsf.getCurrK(),
                    distsFirst);
            int[] knsSecond = NeighborSetFinder.getIndexesOfNeighbors(
                    nsf.getDataSet(), firstInstance, nsf.getCurrK(),
                    distsSecond);
            // Calculate the average distance from each instance to their
            // k-nearest neighbors.
            float kDistsFirstAvg = 0;
            float kDistsSecondAvg = 0;
            for (int i = 0; i < nsf.getCurrK(); i++) {
                kDistsFirstAvg += distsFirst[knsFirst[i]];
                kDistsSecondAvg += distsSecond[knsSecond[i]];
            }
            kDistsFirstAvg /= nsf.getCurrK();
            kDistsSecondAvg /= nsf.getCurrK();
            // Scale by the roow of the product of the kNN distance averages.
            return cmet.dist(firstInstance, secondInstance)
                    / (float) (Math.sqrt(kDistsFirstAvg * kDistsSecondAvg));
        } else {
            return Float.MAX_VALUE;
        }
    }

    /**
     * Calculates the secondary NICDM distance.
     *
     * @param firstInstance First DataInstance.
     * @param secondInstance Second DataInstance.
     * @param distsFirst Distances from the first point to other points in the
     * data.
     * @param knsSecond The kNNs of the second instance.
     * @return The float value that it the secondary NICDM distance.
     * @throws Exception
     */
    public float dist(DataInstance firstInstance, DataInstance secondInstance,
            float[] distsFirst, int[] knsSecond) throws Exception {
        if (nsf != null) {
            CombinedMetric cmet = nsf.getCombinedMetric();
            int[] knsFirst = NeighborSetFinder.getIndexesOfNeighbors(
                    nsf.getDataSet(), firstInstance, nsf.getCurrK(),
                    distsFirst);
            // Calculate the average distance from each instance to their
            // k-nearest neighbors.
            float kDistsFirstAvg = 0;
            float kDistsSecondAvg = 0;
            for (int i = 0; i < nsf.getCurrK(); i++) {
                kDistsFirstAvg += distsFirst[knsFirst[i]];
                kDistsSecondAvg += cmet.dist(secondInstance,
                        nsf.getDataSet().getInstance(knsSecond[i]));
            }
            kDistsFirstAvg /= nsf.getCurrK();
            kDistsSecondAvg /= nsf.getCurrK();
            // Scale by the roow of the product of the kNN distance averages.
            return cmet.dist(firstInstance, secondInstance)
                    / (float) (Math.sqrt(kDistsFirstAvg * kDistsSecondAvg));
        } else {
            return Float.MAX_VALUE;
        }
    }

    /**
     * Calculates the secondary NICDM distance.
     *
     * @param firstInstance First DataInstance.
     * @param secondInstance Second DataInstance.
     * @param knsFirst The kNNs of the first instance.
     * @param knsSecond The kNNs of the second instance.
     * @return The float value that it the secondary NICDM distance.
     * @throws Exception
     */
    public float dist(DataInstance firstInstance, DataInstance secondInstance,
            int[] knsFirst, int[] knsSecond) throws Exception {
        if (nsf != null) {
            CombinedMetric cmet = nsf.getCombinedMetric();
            // Calculate the average distance from each instance to their
            // k-nearest neighbors.
            float kDistsFirstAvg = 0;
            float kDistsSecondAvg = 0;
            for (int i = 0; i < nsf.getCurrK(); i++) {
                kDistsFirstAvg += cmet.dist(firstInstance,
                        nsf.getDataSet().getInstance(knsFirst[i]));
                kDistsSecondAvg += cmet.dist(secondInstance,
                        nsf.getDataSet().getInstance(knsSecond[i]));
            }
            kDistsFirstAvg /= nsf.getCurrK();
            kDistsSecondAvg /= nsf.getCurrK();
            // Scale by the roow of the product of the kNN distance averages.
            return cmet.dist(firstInstance, secondInstance)
                    / (float) (Math.sqrt(kDistsFirstAvg * kDistsSecondAvg));
        } else {
            return Float.MAX_VALUE;
        }
    }

    /**
     * Calculates the secondary NICDM distance.
     *
     * @param firstInstance First DataInstance.
     * @param secondInstance Second DataInstance.
     * @param knsFirst The kNNs of the first instance.
     * @param knsSecond The kNNs of the second instance.
     * @param distsFirst Distances from the first point to other points in the
     * data.
     * @param distsSecond Distances from the second point to other points in the
     * data.
     * @return The float value that it the secondary NICDM distance.
     * @throws Exception
     */
    public float dist(DataInstance firstInstance, DataInstance secondInstance,
            int[] knsFirst, int[] knsSecond, float[] distsFirst,
            float[] distsSecond) throws Exception {
        if (nsf != null) {
            CombinedMetric cmet = nsf.getCombinedMetric();
            // Calculate the average distance from each instance to their
            // k-nearest neighbors.
            float kDistsFirstAvg = 0;
            float kDistsSecondAvg = 0;
            for (int i = 0; i < nsf.getCurrK(); i++) {
                kDistsFirstAvg += distsFirst[knsFirst[i]];
                kDistsSecondAvg += distsSecond[knsSecond[i]];
            }
            kDistsFirstAvg /= nsf.getCurrK();
            kDistsSecondAvg /= nsf.getCurrK();
            // Scale by the roow of the product of the kNN distance averages.
            return cmet.dist(firstInstance, secondInstance)
                    / (float) (Math.sqrt(kDistsFirstAvg * kDistsSecondAvg));
        } else {
            return Float.MAX_VALUE;
        }
    }

    /**
     * Calculates the secondary NICDM distance.
     *
     * @param firstInstance First DataInstance.
     * @param secondInstance Second DataInstance.
     * @param kDistsFirst Distances from the first point to its kNN-s.
     * @param kDistsSecond Distances from the second point to its kNN-s.
     * @return The float value that it the secondary NICDM distance.
     * @throws Exception
     */
    public float distFromKDists(DataInstance firstInstance,
            DataInstance secondInstance, float[] kDistsFirst,
            float[] kDistsSecond) throws Exception {
        if (nsf != null) {
            CombinedMetric cmet = nsf.getCombinedMetric();
            // Calculate the average distance from each instance to their
            // k-nearest neighbors.
            float kDistsFirstAvg = 0;
            float kDistsSecondAvg = 0;
            for (int i = 0; i < nsf.getCurrK(); i++) {
                kDistsFirstAvg += kDistsFirst[i];
                kDistsSecondAvg += kDistsSecond[i];
            }
            kDistsFirstAvg /= nsf.getCurrK();
            kDistsSecondAvg /= nsf.getCurrK();
            // Scale by the roow of the product of the kNN distance averages.
            return cmet.dist(firstInstance, secondInstance)
                    / (float) (Math.sqrt(kDistsFirstAvg * kDistsSecondAvg));
        } else {
            return Float.MAX_VALUE;
        }
    }

    /**
     * Calculates the secondary NICDM distance.
     *
     * @param firstInstance First DataInstance.
     * @param secondInstance Second DataInstance.
     * @param AVGknDistFirst Float value that is the average distance from the
     * first instance to its kNN-s.
     * @param AVGknDistSecond Float value that is the average distance from the
     * second instance to its kNN-s.
     * @return The float value that it the secondary NICDM distance.
     * @throws Exception
     */
    public float dist(DataInstance firstInstance, DataInstance secondInstance,
            float AVGknDistFirst, float AVGknDistSecond) throws Exception {
        if (nsf != null) {
            CombinedMetric cmet = nsf.getCombinedMetric();
            // Scale by the roow of the product of the kNN distance averages.
            return cmet.dist(firstInstance, secondInstance)
                    / (float) (Math.sqrt(AVGknDistFirst * AVGknDistSecond));
        } else {
            return Float.MAX_VALUE;
        }
    }

    /**
     * @return Integer value that is the current neighborhood size.
     */
    public final int getK() {
        return nsf.getCurrK();
    }
}
