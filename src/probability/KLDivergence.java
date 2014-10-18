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
package probability;

import data.representation.util.DataMineConstants;

/**
 * This class implements the Kullback-Leibler divergence. Implicitly does
 * absolute discounting smoothing prior when calculating the measure. This makes
 * it defined for distributions which have zero probabilities without it going
 * to infinity.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class KLDivergence {

    float unknownPrior = 0.0001f;

    public KLDivergence() {
    }

    public KLDivergence(float unknownPrior) {
        this.unknownPrior = unknownPrior;
    }

    /**
     * Calculates KL divergence between two probability distributions. Sometimes
     * the distribution will be contained in a subarray in data instances. Hence
     * , the begin and end index values.
     *
     * @param p First probability distribution, given as a float array.
     * @param q Second probability distribution, given as a float array.
     * @param beginIndex Start index of the distribution in the array.
     * @param endIndex End index of the distribution in the array.
     * @return KL divergence between the distributions.
     */
    public float dist(float[] p, float[] q, int beginIndex, int endIndex) {
        if (p == null || p.length == 0) {
            if (q == null || q.length == 0) {
                return 0;
            } else {
                return Float.MAX_VALUE;
            }
        } else {
            if (q == null || q.length == 0) {
                return Float.MAX_VALUE;
            } else {
                // Now perform actual calculations.
                // First check how many zero values are on both sides.
                int countNZ_P = 0;
                int countNZ_Q = 0;
                int countNZ_Total = 0;
                for (int i = beginIndex; i <= endIndex; i++) {
                    if (DataMineConstants.isNonZero(p[i])) {
                        countNZ_P++;
                        countNZ_Total++;
                        if (DataMineConstants.isNonZero(q[i])) {
                            countNZ_Q++;
                        }
                    } else {
                        if (DataMineConstants.isNonZero(q[i])) {
                            countNZ_Q++;
                            countNZ_Total++;
                        }
                    }
                }
                // Calculate the modifiers for non-zero values.
                float pc = ((float) (countNZ_Total - countNZ_P)
                        / (float) countNZ_P) * unknownPrior;
                float qc = ((float) (countNZ_Total - countNZ_Q)
                        / (float) countNZ_Q) * unknownPrior;
                // Calculate the final distance.
                float result = 0;
                float pHelp, qHelp;
                for (int i = beginIndex; i <= endIndex; i++) {
                    if (DataMineConstants.isNonZero(p[i])) {
                        pHelp = Math.max((p[i] - pc), unknownPrior);
                    } else {
                        pHelp = unknownPrior;
                    }
                    if (DataMineConstants.isNonZero(q[i])) {
                        qHelp = Math.max((q[i] - qc), unknownPrior);
                    } else {
                        qHelp = unknownPrior;
                    }
                    result += pHelp * Math.log(pHelp / qHelp);
                }
                if (Float.isNaN(result)) {
                    result = Float.MAX_VALUE;
                }
                return result;
            }
        }
    }

    /**
     * Calculates KL divergence between two probability distributions.
     *
     * @param p First probability distribution, given as a float array.
     * @param q Second probability distribution, given as a float array.
     * @return KL divergence between the distributions.
     */
    public float dist(float[] p, float[] q) {
        return dist(p, q, 0, p.length - 1);
    }
}
