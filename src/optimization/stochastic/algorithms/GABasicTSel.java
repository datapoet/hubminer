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

import java.util.Random;
import optimization.stochastic.fitness.FitnessEvaluator;
import optimization.stochastic.operators.MutationInterface;
import optimization.stochastic.operators.RecombinationInterface;
import util.AuxSort;

/**
 * This class implements a tournament selection based genetic algorithm for
 * optimization.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class GABasicTSel implements OptimizationAlgorithmInterface {

    private int numIter = 100;
    private int iteration = 0;
    private MutationInterface mutator;
    private RecombinationInterface recombiner;
    private FitnessEvaluator fe;
    private Object[] population;
    private Object[] children = null;
    private Object bestInstance;
    private Object worstInstance;
    private int numEvaluatedInstances = 0;
    private double totalScore = 0;
    private float bestScore = Float.MAX_VALUE;
    private float worstScore = -Float.MAX_VALUE;
    private float score = Float.MAX_VALUE;
    private float[] inversePopulationFitness;
    private float[] inverseOffspringFitness;
    private float[] tempFitness;
    private float totalProbs;
    private Object[] tempPopulation;
    private Object[] tempChildren;
    private int[] rearrange;
    private boolean stop = false;
    private float decision;
    private int first, second, third;

    /**
     * @param population Population of solutions to be optimized.
     * @param mutator An object responsible for inducing mutations.
     * @param recombiner An object responsible for recombining solutions.
     * @param fe FitnessEvaluator object for evaluating fitness of individual
     * solutions.
     * @param numIter Number of iterations to run.
     */
    public GABasicTSel(
            Object[] population,
            MutationInterface mutator,
            RecombinationInterface recombiner,
            FitnessEvaluator fe,
            int numIter) {
        this.population = population;
        this.mutator = mutator;
        this.numIter = numIter;
        this.fe = fe;
        this.recombiner = recombiner;
    }

    @Override
    public void optimize() throws Exception {
        inversePopulationFitness = new float[population.length];
        // First calculate the fitness of all the parents..
        for (int i = 0; i < population.length; i++) {
            inversePopulationFitness[i] = evaluate(population[i]);
        }
        rearrange = AuxSort.sortIndexedValue(inversePopulationFitness, false);
        tempPopulation = new Object[population.length];
        for (int i = 0; i < population.length; i++) {
            tempPopulation[i] = population[rearrange[i]];
        }
        population = tempPopulation;
        children = new Object[2 * population.length];
        inverseOffspringFitness = new float[children.length];
        tempFitness = new float[population.length];
        Random randa = new Random();
        while (!stop && ++iteration <= numIter) {
            // Perform mutations.
            for (int i = 0; i < population.length; i++) {
                if (!stop) {
                    children[i] = mutator.mutateNew(population[i]);
                    inverseOffspringFitness[i] = evaluate(children[i]);
                } else {
                    return;
                }
            }
            // Perform recombinations.
            totalProbs = 0;
            for (int i = 1; i < population.length; i++) {
                decision = randa.nextFloat() * totalProbs;
                first = randa.nextInt(population.length);
                second = first;
                while (second == first) {
                    second = randa.nextInt(population.length);
                }
                if (inversePopulationFitness[second]
                        < inversePopulationFitness[first]) {
                    first = second;
                }
                second = first;
                third = second;
                while (second == first || third == first
                        || third == second) {
                    second = randa.nextInt(population.length);
                    third = randa.nextInt(population.length);
                }
                if (inversePopulationFitness[third]
                        < inversePopulationFitness[second]) {
                    second = third;
                }
                children[population.length + i] =
                        recombiner.recombine(
                        population[first], population[second]);
            }
            for (int i = 0; i < children.length; i++) {
                if (!stop) {
                    inverseOffspringFitness[i] = evaluate(children[i]);
                } else {
                    return;
                }
            }
            rearrange =
                    AuxSort.sortIndexedValue(inverseOffspringFitness, false);
            tempChildren = new Object[children.length];
            for (int i = 0; i < children.length; i++) {
                tempChildren[i] = children[rearrange[i]];
            }
            children = tempChildren;
            //now merge
            int index = -1;
            first = 0;
            second = 0;
            while (++index < population.length) {
                if (inversePopulationFitness[first]
                        < inverseOffspringFitness[second]) {
                    tempPopulation[index] = population[first];
                    tempFitness[index] = inversePopulationFitness[first];
                    first++;
                } else {
                    tempPopulation[index] = children[second];
                    tempFitness[index] = inverseOffspringFitness[second];
                    second++;
                }
            }
        }
    }

    /**
     * Evaluates the fitness of the current solution and updates the best/worst
     * solution fitness stats.
     *
     * @param instance
     * @return
     */
    private float evaluate(Object instance) {
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
        return score;
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
