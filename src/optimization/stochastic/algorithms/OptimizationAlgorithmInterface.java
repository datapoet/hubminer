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
package optimization.stochastic.algorithms;

import optimization.stochastic.fitness.FitnessEvaluator;

/**
 * By convention, everything will be a MINIMIZATION problem. This means that if
 * the natural fitness is not such, it needs to be changed so that it is smaller
 * for better instances.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public interface OptimizationAlgorithmInterface {

    /**
     * Performs optimization.
     *
     * @throws Exception
     */
    public void optimize() throws Exception;

    /**
     * Gets the index of the current iteration.
     *
     * @return
     */
    public int getIteration();

    /**
     * Set the number of iterations the optimization is expected to run.
     *
     * @param numIter Integer that represents the maximum number of iterations.
     */
    public void setNumIter(int numIter);

    /**
     * Get the maximum number of iterations the optimizer is to run.
     *
     * @return Integer that represents the maximum number of iterations.
     */
    public int getNumIter();

    /**
     * @return The best solution up until this point.
     */
    public Object getBestInstance();

    /**
     * @return Best fitness up until this point.
     */
    public float getBestFitness();

    /**
     * The worst solution up until this point.
     *
     * @return
     */
    public Object getWorstInstance();

    /**
     * @return Worst fitness up until this point.
     */
    public float getWorstFitness();

    /**
     * @return Average fitness up until this point.
     */
    public float getAverageFitness();

    /**
     * @return FitnessEvaluator used for estimating solution fitness within the
     * optimizer.
     */
    public FitnessEvaluator getFitnessEvaluator();

    /**
     * @param fe FitnessEvaluator used for estimating solution fitness within
     * the optimizer.
     */
    public void setFitnessEvaluator(FitnessEvaluator fe);

    /**
     * Stop the optimization process externally.
     */
    public void stop();
}
