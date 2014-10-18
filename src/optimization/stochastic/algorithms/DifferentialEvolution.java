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

import data.representation.DataInstance;
import java.util.Random;
import optimization.stochastic.fitness.FitnessEvaluator;

/**
 * A class that implements differential evolution on float vectors.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class DifferentialEvolution implements OptimizationAlgorithmInterface {

    private int numIter = 100;
    private int iteration = 0;
    private FitnessEvaluator fe;
    private Object[] population;
    private Object[] children = null;
    private Object bestInstance;
    private Object worstInstance;
    private int numEvaluatedInstances = 0;
    private double totalScore = 0;
    private float bestScore = Float.MAX_VALUE;
    // Smaller is better in this implementation.
    private float worstScore = Float.MIN_VALUE;
    private float score = Float.MAX_VALUE;
    private float[] populationFitness;
    private boolean stop = false;
    private float decision;
    private float CR = 0.7f;
    private float F = 0.6f;

    public DifferentialEvolution() {
    }

    /**
     * Class constructor.
     *
     * @param population Population of solutions to be optimized.
     * @param numIter Number of iterations to run.
     * @param fe FitnessEvaluator object for evaluating fitness of individual
     * solutions.
     * @param CR Recombination probability.
     * @param F Linear combination parameter (see formula for details).
     */
    public DifferentialEvolution(
            Object[] population,
            int numIter,
            FitnessEvaluator fe,
            float CR,
            float F,
            float[] lBounds,
            float[] hBounds) {
        this.population = population;
        this.numIter = numIter;
        this.fe = fe;
        this.CR = CR;
        this.F = F;
    }

    @Override
    public void optimize() throws Exception {
        //first initial eval
        int r1, r2, r3;
        DataInstance instance;
        DataInstance newTemp;
        Random randa = new Random();
        int certainMutationIndex;
        populationFitness = new float[population.length];
        for (int i = 0; i < population.length; i++) {
            score = evaluate(population[i]);
            populationFitness[i] = score;
        }
        for (iteration = 1; iteration <= numIter; iteration++) {
            // Create children (the next generation).
            children = new Object[population.length];
            for (int i = 0; i < population.length; i++) {
                instance = (DataInstance) (population[i]);
                newTemp = new DataInstance();
                newTemp.sAttr = instance.sAttr;
                newTemp.iAttr = instance.iAttr;
                newTemp.fAttr = new float[instance.fAttr.length];
                certainMutationIndex = randa.nextInt(instance.fAttr.length);
                for (int j = 0; j < instance.fAttr.length; j++) {
                    // Now select three random donors.
                    do {
                        r1 = randa.nextInt(population.length);
                    } while (r1 == j);
                    do {
                        r2 = randa.nextInt(population.length);
                    } while ((r2 == j) || (r2 == r1));
                    do {
                        r3 = randa.nextInt(population.length);
                    } while ((r3 == j) || (r3 == r2) || (r3 == r1));
                    decision = randa.nextFloat();
                    if (j == certainMutationIndex || decision < CR) {
                        // Perform mutation & recombination.
                        newTemp.fAttr[j] =
                                ((DataInstance) population[r3]).fAttr[j]
                                + F * (((DataInstance) population[r2]).fAttr[j]
                                - ((DataInstance) population[r1]).fAttr[j]);
                    } else {
                        newTemp.fAttr[j] = instance.fAttr[j];
                    }
                }
                children[i] = newTemp;
            }
            // Calculate fitness for the children and replace certain parents.
            for (int i = 0; i < population.length; i++) {
                if (!stop) {
                    score = evaluate(children[i]);
                    if (score < populationFitness[i]) {
                        // Replace parent by the child.
                        population[i] = children[i];
                        populationFitness[i] = score;
                    }
                }
            }
        }
        stop = true;
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
    public int getIteration() {
        return iteration;
    }

    @Override
    public void setNumIter(int numIter) {
        this.numIter = numIter;
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
        if (numEvaluatedInstances == 0) {
            return Float.MAX_VALUE;
        } else {
            return (float) (totalScore / numEvaluatedInstances);
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
