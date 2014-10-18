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
 * This class implements the island model of GA optimization where there are
 * multiple separate solution subpopulations that occasionally communicate, so
 * that they can in principle search different parts of the data space, but also
 * horizontally transfer some of the acquired knowledge to enrich the local
 * search process.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class GAIslandModel implements OptimizationAlgorithmInterface {

    private int numIter = 100;
    private int iteration = 0;
    private int numIslands = 2;
    private float migrationProbability;
    // Probability of migrations occuring in any specific iteration.
    private int migrationSize = 10;
    // Migration size needs to be set depending on the total population size.
    private MutationInterface mutator;
    private RecombinationInterface recombiner;
    private FitnessEvaluator fe;
    private Object[][] populations;
    private Object[] children = null;
    private Object bestInstance;
    private Object worstInstance;
    private int numEvaluatedInstances = 0;
    private double totalScore = 0;
    private float bestScore = Float.MAX_VALUE;
    private float worstScore = -Float.MAX_VALUE;
    private float score = Float.MAX_VALUE;
    private float[][] inversePopulationFitness;
    private float[][] inverseOffspringFitness;
    private float[][] cumulativeProbs;
    private float[] tempFitness;
    private float[] totalProbs;
    private Object[] tempPopulation;
    private Object[] tempChildren;
    private int[] rearrange;
    private boolean stop = false;
    private float decision;
    private int first, second;

    /**
     *
     * @param population Population of solutions to be optimized.
     * @param mutator An object responsible for inducing mutations.
     * @param recombiner An object responsible for recombining solutions.
     * @param fe FitnessEvaluator object for evaluating fitness of individual
     * solutions.
     * @param numIter Number of iterations to run.
     * @param migrationProbability Float value representing the probability of
     * migration between island subpopulations.
     * @param migrationSize Migration size, once it occurs.
     */
    public GAIslandModel(
            Object[][] population,
            MutationInterface mutator,
            RecombinationInterface recombiner,
            FitnessEvaluator fe,
            int numIter,
            float migrationProbability,
            int migrationSize) {
        this.populations = population;
        this.mutator = mutator;
        this.numIter = numIter;
        this.fe = fe;
        this.recombiner = recombiner;
        this.migrationProbability = migrationProbability;
        this.migrationSize = migrationSize;
    }

    @Override
    public void optimize() throws Exception {
        numIslands = populations.length;
        cumulativeProbs = new float[numIslands][];
        totalProbs = new float[numIslands];
        inversePopulationFitness = new float[numIslands][];
        for (int islandIndex = 0; islandIndex < numIslands; islandIndex++) {
            inversePopulationFitness[islandIndex] =
                    new float[populations[islandIndex].length];
        }
        inverseOffspringFitness = new float[numIslands][];
        // Do all populations separately.
        // First calculate the fitness of all the initial parents.
        for (int islandIndex = 0; islandIndex < numIslands; islandIndex++) {
            for (int i = 0; i < populations[islandIndex].length; i++) {
                inversePopulationFitness[islandIndex][i] =
                        evaluate(populations[islandIndex][i]);
            }
            rearrange = AuxSort.sortIndexedValue(
                    inversePopulationFitness[islandIndex], false);
            tempPopulation = new Object[populations[islandIndex].length];
            for (int i = 0; i < populations[islandIndex].length; i++) {
                tempPopulation[i] = populations[islandIndex][rearrange[i]];
            }
            populations[islandIndex] = tempPopulation;
        }
        Random randa = new Random();
        while (!stop && ++iteration <= numIter) {
            for (int islandIndex = 0; islandIndex < numIslands; islandIndex++) {
                children = new Object[2 * populations[islandIndex].length];
                inverseOffspringFitness[islandIndex] =
                        new float[2 * populations[islandIndex].length];
                cumulativeProbs[islandIndex] =
                        new float[populations[islandIndex].length];
                tempFitness = new float[populations[islandIndex].length];
                // Perform mutations.
                for (int i = 0; i < populations[islandIndex].length; i++) {
                    if (!stop) {
                        children[i] =
                                mutator.mutateNew(populations[islandIndex][i]);
                        inverseOffspringFitness[islandIndex][i] =
                                evaluate(children[i]);
                    } else {
                        return;
                    }
                }
                // Perform recombinations.
                totalProbs[islandIndex] = 0;
                cumulativeProbs[islandIndex][0] =
                        (float) Math.pow(
                        Math.E,
                        -inversePopulationFitness[islandIndex][0]);
                totalProbs[islandIndex] += cumulativeProbs[islandIndex][0];
                for (int i = 1; i < populations[islandIndex].length; i++) {
                    cumulativeProbs[islandIndex][i] =
                            cumulativeProbs[islandIndex][i - 1]
                            + (float) Math.pow(Math.E,
                            -inversePopulationFitness[islandIndex][i]);
                    totalProbs[islandIndex] += cumulativeProbs[islandIndex][i];
                }
                for (int i = 1; i < populations[islandIndex].length; i++) {
                    decision = randa.nextFloat() * totalProbs[islandIndex];
                    first = findIndex(
                            islandIndex,
                            decision,
                            0,
                            populations[islandIndex].length - 1);
                    second = first;
                    while (second == first) {
                        decision = randa.nextFloat()
                                * totalProbs[islandIndex];
                        second = findIndex(
                                islandIndex,
                                decision,
                                0,
                                populations[islandIndex].length - 1);
                    }
                    children[populations[islandIndex].length + i] =
                            recombiner.recombine(
                            populations[islandIndex][first],
                            populations[islandIndex][second]);
                }
                for (int i = 0; i < children.length; i++) {
                    if (!stop) {
                        inverseOffspringFitness[islandIndex][i] =
                                evaluate(children[i]);
                    } else {
                        return;
                    }
                }
                rearrange = AuxSort.sortIndexedValue(
                        inverseOffspringFitness[islandIndex], false);
                tempChildren = new Object[children.length];
                for (int i = 0; i < children.length; i++) {
                    tempChildren[i] = children[rearrange[i]];
                }
                children = tempChildren;
                //now merge
                int index = -1;
                first = 0;
                second = 0;
                while (++index < populations[islandIndex].length) {
                    if (inversePopulationFitness[islandIndex][first]
                            < inverseOffspringFitness[islandIndex][second]) {
                        tempPopulation[index] = populations[islandIndex][first];
                        tempFitness[index] =
                                inversePopulationFitness[islandIndex][first];
                        first++;
                    } else {
                        tempPopulation[index] = children[second];
                        tempFitness[index] =
                                inverseOffspringFitness[islandIndex][second];
                        second++;
                    }
                }
            }
            // Perform migrations stochastically.
            decision = randa.nextFloat();
            if (decision < migrationProbability) {
                for (int i = 0; i < migrationSize; i++) {
                    int chosen = 0;
                    int searchSpot = -1;
                    // Randomly pick the source and the destination.
                    // The selected individual will attempt to migrate to the
                    // destination if its fitness allows it.
                    // The selection of the individual is fitness-proportional.
                    first = randa.nextInt(numIslands);
                    second = first;
                    while (second == first) {
                        second = randa.nextInt(numIslands);
                    }
                    decision = randa.nextFloat() * totalProbs[first];
                    chosen = findIndex(first, decision, 0,
                            populations[first].length - 1);
                    // Now find a place (if any) to insert it into target
                    // population.
                    while (++searchSpot < populations[second].length
                            && inversePopulationFitness[second][searchSpot]
                            < inversePopulationFitness[first][chosen]) {
                        // increment and search
                    }
                    if (searchSpot < populations[second].length) {
                        totalProbs[second] -=
                                (float) Math.pow(Math.E,
                                -inversePopulationFitness[second][
                                populations[second].length - 1]);
                        totalProbs[second] +=
                                (float) Math.pow(Math.E,
                                -inversePopulationFitness[first][chosen]);
                        for (int j = populations[second].length - 1;
                                j > searchSpot; j--) {
                            populations[second][j] = populations[second][j - 1];
                            inversePopulationFitness[second][j] =
                                    inversePopulationFitness[second][j - 1];
                        }
                        populations[second][searchSpot] =
                                populations[first][chosen];
                        inversePopulationFitness[second][searchSpot] =
                                inversePopulationFitness[first][chosen];
                        // Update the probabilities.
                        for (int j = searchSpot; j < populations[second].length;
                                j++) {
                            cumulativeProbs[second][j] =
                                    cumulativeProbs[second][j - 1]
                                    + (float) Math.pow(Math.E,
                                    -inversePopulationFitness[second][j]);
                        }
                    }
                }
            }
        }
    }

    /**
     * Binary search for the index within island subpopulations.
     *
     * @param islandIndex Island index.
     * @param searchValue Query value.
     * @param first First index.
     * @param second Second index.
     * @return
     */
    private int findIndex(int islandIndex, double searchValue, int first,
            int second) {
        if (second - first <= 1) {
            return second; //first isn't, so it must be second
        }
        int middle = (first + second) / 2;
        if (cumulativeProbs[islandIndex][middle] < searchValue) {
            return findIndex(islandIndex, searchValue, middle, second);
        } else {
            return findIndex(islandIndex, searchValue, first, middle);
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
