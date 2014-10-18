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

import optimization.stochastic.operators.MutationInterface;
import optimization.stochastic.fitness.FitnessEvaluator;

/**
 * This class implements a simple hill climbing optimization procedure.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class HillClimbing implements OptimizationAlgorithmInterface {

    private int numIter = 100;
    private int iteration = 0;
    private MutationInterface mutator;
    private FitnessEvaluator fe;
    private Object instance;
    private Object previousInstance = null;
    private float previousScore = Float.MAX_VALUE;
    private Object bestInstance;
    private Object worstInstance;
    private int numEvaluatedInstances = 0;
    private double totalScore = 0;
    private float bestScore = Float.MAX_VALUE;
    private float worstScore = -Float.MAX_VALUE;
    private float score = Float.MAX_VALUE;
    private boolean stop = false;

    /**
     * @param instance Solution seed.
     * @param mutator An object responsible for inducing mutations.
     * @param fe FitnessEvaluator object for evaluating fitness of individual
     * solutions.
     * @param numIter Number of iterations to run.
     */
    public HillClimbing(
            Object instance,
            MutationInterface mutator,
            FitnessEvaluator fe,
            int numIter) {
        this.instance = instance;
        this.mutator = mutator;
        this.numIter = numIter;
        this.fe = fe;
    }

    @Override
    public void optimize() throws Exception {
        evaluate(instance);
        while (!stop && ++iteration <= numIter) {
            if (score < previousScore) {
                replacePrevious();
            }
            instance = mutator.mutateNew(previousInstance);
            evaluate(instance);
        }
    }

    /**
     * Evaluates the fitness of the current solution and updates the best/worst
     * solution fitness stats.
     *
     * @param instance
     * @return
     */
    private void evaluate(Object instance) {
        score = fe.evaluate(instance);
        numEvaluatedInstances++;
        if (score < bestScore) {
            bestScore = score;
            bestInstance = instance;
        }
        if (score > worstScore) {
            worstScore = score;
            worstInstance = instance;
        }
        totalScore += score;
    }

    /**
     * Replace previous with current score.
     */
    private void replacePrevious() {
        previousScore = score;
        previousInstance = instance;
    }

    @Override
    public void setNumIter(int numIter) {
        this.numIter = numIter;
    }

    @Override
    public int getIteration() {
        return iteration;
    }

    @Override
    public int getNumIter() {
        return numIter;
    }

    @Override
    public Object getBestInstance() {
        return bestInstance;
    }

    @Override
    public float getBestFitness() {
        return bestScore;
    }

    @Override
    public Object getWorstInstance() {
        return worstInstance;
    }

    @Override
    public float getWorstFitness() {
        return worstScore;
    }

    @Override
    public float getAverageFitness() {
        if (numEvaluatedInstances > 0) {
            return (float) (totalScore / (double) numEvaluatedInstances);
        } else {
            return worstScore;
        }
    }

    @Override
    public FitnessEvaluator getFitnessEvaluator() {
        return fe;
    }

    @Override
    public void setFitnessEvaluator(FitnessEvaluator fe) {
        this.fe = fe;
    }

    @Override
    public void stop() {
        stop = true;
    }
}
