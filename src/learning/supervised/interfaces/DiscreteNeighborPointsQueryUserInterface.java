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
package learning.supervised.interfaces;

import data.representation.discrete.DiscretizedDataInstance;

/**
 * This interface declares the methods for classification that requires kNN
 * queries. In that case, a single kNN set can be calculated and distributed to
 * all the simultaneously running classifiers that request it, instead of having
 * each classifier calculate it over and over again. Since there are many
 * kNN-based learning methods in this library, this enables the evaluation to
 * achieve a significant speed-up. This interface declares the methods for kNN
 * queries made by classifiers for discretized data instances, unlike the other
 * interface that handles continuous feature valued data instances.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public interface DiscreteNeighborPointsQueryUserInterface {

    /**
     * This method performs classification of the target data point, based on
     * its nearest neighbors and the distances to all the training points, given
     * for reference.
     *
     * @param instance DiscretizedDataInstance object that is to be classified.
     * @param distToTraining float[] representing distances to all training
     * points.
     * @param neighborsAmongTraining int[] representing the indexes of the
     * k-nearest neighbors among the training data.
     * @return float[] that is the probabilistic classification of the target
     * DiscretizedDataInstance object.
     * @throws Exception
     */
    public float[] classifyProbabilistically(DiscretizedDataInstance instance,
            float[] distToTraining, int[] neighborsAmongTraining)
            throws Exception;

    /**
     * This method performs classification of the target data point, based on
     * its nearest neighbors and the distances to all the training points, given
     * for reference.
     *
     * @param instance DiscretizedDataInstance object that is to be classified.
     * @param distToTraining float[] representing distances to all training
     * points.
     * @param neighborsAmongTraining int[] representing the indexes of the
     * k-nearest neighbors among the training data.
     * @return Integer that is the classification outcome, the prediction of the
     * class label.
     * @throws Exception
     */
    public int classify(DiscretizedDataInstance instance,
            float[] distToTraining, int[] neighborsAmongTraining)
            throws Exception;
}
