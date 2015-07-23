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
package optimization.stochastic.operators;

/**
 * This interface extends the basic mutation interface in that it supports
 * heterogenous mutations rates where each feature can mutate at a different
 * rate. In this particular case, all feature have the same mutation probability
 * , but different mutation magnitudes.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public interface HeterogenousMutationInterface extends MutationInterface {

    /**
     * @param stDevs An array of standard deviations for the individual features
     */
    public void setStDevs(float[] stDevs);

    /**
     * @return An array of standard deviations for the individual features.
     */
    public float[] getStDevs();
}
