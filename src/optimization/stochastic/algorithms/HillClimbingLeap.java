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
import optimization.stochastic.operators.TwoDevsMutationInterface;

/**
 * A class that implements a simple twist to the basic hill-climbing approach,
 * by introducing a small probability of making larger changes, 'leaps' towards
 * unexplored parts of the optimization landscape. This helps the optimization
 * escape from local optima and possibly climb more than one hill, leading to a
 * better result.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class HillClimbingLeap implements OptimizationAlgorithmInterface {

    int numIter = 100;
    int iteration = 0;
    int iterationsNoImprovement = 0;
    int noImprovementMax = 30;
    // This must be carefully set to some meaningful number, depending on
    // numIter and how long it takes to evaluate fitness.
    TwoDevsMutationInterface mutator;
    FitnessEvaluator fe;
    Object instance;
    Object previousInstance = null;
    float previousScore = Float.MAX_VALUE;
    Object bestInstance;
    Object worstInstance;
    int numEvaluatedInstances = 0;
    double totalScore = 0;
    float bestScore = Float.MAX_VALUE;
    float worstScore = -Float.MAX_VALUE;
    float score = Float.MAX_VALUE;
    boolean stop = false;
    float pSmall = 1;

    /**
     * @param instance Solution seed.
     * @param mutator An object responsible for inducing mutations.
     * @param fe FitnessEvaluator object for evaluating fitness of individual
     * solutions.
     * @param numIter Number of iterations to run.
     */
    public HillClimbingLeap(
            Object instance,
            TwoDevsMutationInterface mutator,
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
                iterationsNoImprovement = 0;
            } else {
                iterationsNoImprovement++;
            }
            updateMutationProbabilities();
            instance = mutator.mutateNew(previousInstance);
            evaluate(instance);
        }
    }

    /**
     * Update mutation probabilities.
     */
    private void updateMutationProbabilities() {
        float pLarge = (float) iterationsNoImprovement /
                (float) noImprovementMax;
        pLarge = Math.min(pLarge, 1f);
        pSmall = 1f - pLarge;
        mutator.setPSmall(pSmall);
    }

    /**
     * Sets the maximum number of iterations with no improvement. This number is
     * also used to determine the probability of making leaps.
     *
     * @param noImprovementsMax Maximum number of iterations with no improvement
     */
    public void setNoImprovementMax(int noImprovementsMax) {
        noImprovementMax = noImprovementsMax;
    }

    /**
     * Evaluates the fitness of the current solution and updates the best/worst
     * solution fitness stats.
     *
     * @param instance
     * @return
     */
    public void evaluate(Object instance) {
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
    public void replacePrevious() {
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
