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
 * Interface that sets up methods for mutating solutions.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public interface MutationInterface {

    /**
     * This method makes a new mutated solution, but keeps the original solution
     * intact.
     *
     * @param o Solution to mutate.
     * @return Object that is the newly mutated solution.
     * @throws Exception
     */
    public Object mutateNew(Object o) throws Exception;

    /**
     * This method mutates the exiting solution without generating a new one.
     *
     * @param o Solution to mutate.
     * @throws Exception
     */
    public void mutate(Object o) throws Exception;

    /**
     * A batch call for generating new mutating solutions.
     *
     * @param oArray An array of solutions to mutate.
     * @return An array of mutated solutions.
     * @throws Exception
     */
    public Object[] mutateNew(Object[] oArray) throws Exception;

    /**
     * A batch call for mutating solutions.
     *
     * @param oArray An array of solutions to mutate.
     * @throws Exception
     */
    public void mutate(Object[] oArray) throws Exception;
}
