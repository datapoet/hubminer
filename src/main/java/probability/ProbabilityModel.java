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

/**
 * This class represents a general probability model. There is only one
 * mandatory method - that calculates the probabilities for observing the test
 * data given the model, as this functionality is necessary for various
 * evaluation methods.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public abstract class ProbabilityModel {

    /**
     * @param testArray DataInstance[] that is the test data array.
     * @return double[] that are the probabilities of the test data points
     * given the model.
     */
    public abstract double[] calcTestDataProbabilities(
            DataInstance[] testArray);
}
