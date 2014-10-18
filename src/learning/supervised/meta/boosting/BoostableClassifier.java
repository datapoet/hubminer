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
package learning.supervised.meta.boosting;

import java.io.Serializable;
import learning.supervised.Classifier;

/**
 * This class extends the usual Classifier class by introducing the instance
 * weighting options to be used in re-weightable boosting procedures.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public abstract class BoostableClassifier extends Classifier
implements Serializable {

    private static final long serialVersionUID = 1L;
    /**
     * This method sets the instance weights prior to model training. The
     * weights correspond to the instance difficulty and relevance in the model.
     *
     * @param instanceWeights double[] of instance weights to be set.
     */
    public abstract void setTotalInstanceWeights(double[] instanceWeights);

    /**
     * This method sets the current label misclassification cost distribution
     * for each instance. If x has the label y, then instanceLabelWeights[x][y]
     * should be zero, as no additional cost is associated with the proper
     * classification (that information is provided in the other instance
     * weighting method, for clarity). In any case, such values will be ignored.
     * As for the remaining weights, they should sum up to 1 for each instance.
     *
     * @param instanceLabelWeights double[][] of the cost distributions over
     * labels for each DataInstance in the training data.
     */
    public abstract void setMisclassificationCostDistribution(
            double[][] instanceLabelWeights);
}
