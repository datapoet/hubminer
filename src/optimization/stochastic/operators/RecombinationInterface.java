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
 * Interface that sets up methods necessary for recombination between two or
 * more genotypes.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public interface RecombinationInterface {

    /**
     * Generates a recombined child genotype from two parent genotypes.
     *
     * @param o1 First genotype.
     * @param o2 Second genotype.
     * @return The recombined solution.
     * @throws Exception
     */
    public Object recombine(Object o1, Object o2) throws Exception;

    /**
     * Recombines a pair of parents into a pair of children. Sometimes it will
     * simply amount to two calls of the first method, but if it's swap
     * recombination, then it does something smart, because then it outputs a
     * recombined instance and its complementary one.
     *
     * @param o1 First genotype.
     * @param o2 Second genotype.
     * @return A pair of recombined solutions.
     * @throws Exception
     */
    public Object[] recombinePair(Object o1, Object o2) throws Exception;

    /**
     * A batch recombination call. Solutions from the first parameter array are
     * paired with the solutions from the second parameter array in order to
     * produce recombined children solutions.
     *
     * @param oArray1 An array of parent solutions.
     * @param oArray2 An array of parent solutions.
     * @return An array of recombined solutions.
     * @throws Exception
     */
    public Object[] recombine(Object[] oArray1, Object[] oArray2)
            throws Exception;
}
