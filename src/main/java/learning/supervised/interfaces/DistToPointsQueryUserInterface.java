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

import data.representation.DataInstance;

/**
 * This interface declares the methods for classification that rely on the
 * distances to all training set points. This can also be used for some kNN
 * methods, though those methods also implement an interface that accepts kNN
 * sets directly, along with the distances to training points.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public interface DistToPointsQueryUserInterface {

    /**
     * This method performs classification of the target DataInstance.
     *
     * @param instance DataInstance that is to be classified.
     * @param distToTraining float[] representing distances to all training
     * points.
     * @return float[] that is the predicted class distribution.
     * @throws Exception
     */
    public float[] classifyProbabilistically(DataInstance instance,
            float[] distToTraining) throws Exception;

    /**
     * This method performs classification of the target DataInstance.
     *
     * @param instance DataInstance that is to be classified.
     * @param distToTraining float[] representing distances to all training
     * points.
     * @return Integer that is the classification outcome, the prediction of the
     * class label.
     * @throws Exception
     */
    public int classify(DataInstance instance, float[] distToTraining)
            throws Exception;
}
