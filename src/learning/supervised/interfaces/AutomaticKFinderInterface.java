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

/**
 * This interface declares the method for automatically searching for the best
 * k-value that can be implemented by kNN learning methods in order to be
 * evaluated from within the classification testing framework that relies on
 * automatic best parameter search.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public interface AutomaticKFinderInterface {

    /**
     * This method looks for the best parameter configuration in terms of the
     * neighborhood size bounded by the specified limits, as well as the
     * internal parameters of the classifier in question.
     *
     * @param kMin Integer that is the minimal neighborhood size to consider.
     * @param kMax Integer that is the maximal neighborhood size to consider.
     * @throws Exception
     */
    public void findK(int kMin, int kMax) throws Exception;
}
