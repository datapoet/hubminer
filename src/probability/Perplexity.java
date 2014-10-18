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

import data.representation.DataInstance;
import data.representation.util.DataMineConstants;
import util.BasicMathUtil;

/**
 * Perplexity allows us to evaluate the model that we have inferred on the
 * training data by measuring our 'surprise' on the test data, that is -
 * perplexity. The lower the better.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class Perplexity {

    /**
     * @param pm ProbabilityModel instance that we wish to evaluate.
     * @param testData Test data that the ProbabilityModel was not inferred from
     * @return Model perplexity.
     */
    public static double getModelPerplexity(ProbabilityModel pm,
            DataInstance[] testData) {
        if (pm == null || testData == null || testData.length == 0) {
            return 0;
        }
        double[] testProbs = pm.calcTestDataProbabilities(testData);
        double crossEntropy = 0;
        int N = testData.length;
        int nNonZero = 0;
        for (int i = 0; i < N; i++) {
            if (DataMineConstants.isNonZero(testProbs[i])) {
                crossEntropy += BasicMathUtil.log2(testProbs[i]);
                nNonZero++;
            }
        }
        if (nNonZero > 0) {
            crossEntropy /= -nNonZero;
        } else {
            crossEntropy = Float.POSITIVE_INFINITY;
        }
        double perplexity = Math.pow(2, crossEntropy);
        return perplexity;
    }

    /**
     * When applied to discrete probability distributions, perplexity measures
     * uncertainty.
     *
     * @param probs A double array of probabilities in a discrete probability
     * distribution.
     * @return Perplexity of the probability distribution.
     */
    public static double getDistributionPerplexity(double[] probs) {
        double crossEntropy = 0;
        int N = probs.length;
        for (int i = 0; i < N; i++) {
            if (DataMineConstants.isNonZero(probs[i])) {
                crossEntropy -= probs[i] * BasicMathUtil.log2(probs[i]);
            }
        }
        double perplexity = Math.pow(2, crossEntropy);
        return perplexity;
    }

    /**
     * When applied to discrete probability distributions, perplexity measures
     * uncertainty.
     *
     * @param probs A float array of probabilities in a discrete probability
     * distribution.
     * @return Perplexity of the probability distribution.
     */
    public static double getDistributionPerplexity(float[] probs) {
        double crossEntropy = 0;
        int N = probs.length;
        for (int i = 0; i < N; i++) {
            if (DataMineConstants.isNonZero(probs[i])) {
                crossEntropy -= probs[i] * BasicMathUtil.log2(probs[i]);
            }
        }
        double perplexity = Math.pow(2, crossEntropy);
        return perplexity;
    }
}
