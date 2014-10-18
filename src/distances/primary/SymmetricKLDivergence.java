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
package distances.primary;

import data.representation.util.DataMineConstants;
import java.io.Serializable;

/**
 * Calculates the symmetrized KL divergence between two probability
 * distributions.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class SymmetricKLDivergence extends DistanceMeasure
implements Serializable {

    private static final long serialVersionUID = 1L;
    private float unknownPrior = 0.0001f;
    private int beginIndex = 0;
    private int endIndex = -1;

    /**
     * The default constructor.
     */
    public SymmetricKLDivergence() {
    }

    /**
     * Initialization.
     * 
     * @param unknownPrior Float that is the prior.
     */
    public SymmetricKLDivergence(float unknownPrior) {
        this.unknownPrior = unknownPrior;
    }

    /**
     * Initialization.
     * 
     * @param unknownPrior Float that is the prior.
     * @param beginIndex Integer that marks the start index where the
     * distribution is located in the feature array.
     * @param endIndex Integer that marks the end index where the
     * distribution is located in the feature array.
     */
    public SymmetricKLDivergence(float unknownPrior, int beginIndex,
            int endIndex) {
        this.unknownPrior = unknownPrior;
        this.beginIndex = beginIndex;
        this.endIndex = endIndex;
    }

    @Override
    public float dist(float[] p, float[] q) throws MetricException {
        if (p == null && q == null) {
            return 0f;
        } else {
            if (p == null) {
                return Float.MAX_VALUE;
            }
            if (q == null) {
                return Float.MAX_VALUE;
            }
        }
        if (endIndex < 0) {
            endIndex = p.length;
        }
        int countNZ_P = 0;
        int countNZ_Q = 0;
        int countNZ_Total = 0;
        for (int i = beginIndex; i <= endIndex; i++) {
            if (DataMineConstants.isPositive(p[i])) {
                countNZ_P++;
                countNZ_Total++;
                if (DataMineConstants.isPositive(q[i])) {
                    countNZ_Q++;
                }
            } else {
                if (DataMineConstants.isPositive(q[i])) {
                    countNZ_Q++;
                    countNZ_Total++;
                }
            }
        }
        // Ok, now that we have that... calculate the modifiers for
        // non-zero values.
        float pc = ((float) (countNZ_Total - countNZ_P)
                / (float) countNZ_P) * unknownPrior;
        float qc = ((float) (countNZ_Total - countNZ_Q)
                / (float) countNZ_Q) * unknownPrior;
        // Now let's calculate the final value.
        float result = 0;
        float pHelp, qHelp;
        for (int i = beginIndex; i <= endIndex; i++) {
            if (DataMineConstants.isPositive(p[i])) {
                pHelp = Math.max((p[i] - pc), unknownPrior);
            } else {
                pHelp = unknownPrior;
            }
            if (DataMineConstants.isPositive(q[i])) {
                qHelp = Math.max((q[i] - qc), unknownPrior);
            } else {
                qHelp = unknownPrior;
            }
            result += pHelp * Math.log(pHelp / qHelp);
            result += qHelp * Math.log(qHelp / pHelp);
        }
        if (Float.isNaN(result)) {
            result = Float.MAX_VALUE;
        }
        return (result / 2f);
    }
}
