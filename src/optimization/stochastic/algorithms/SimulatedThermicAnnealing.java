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

/**
 * A class that implements simple simulated thermic annealing.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class SimulatedThermicAnnealing
        implements OptimizationAlgorithmInterface {

    private int numIter = 500;
    private int iteration = 0;
    private int averageSeekingIter = 25;
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
    private float p0 = 0.8f;
    private float pf = 0.05f;
    private float probUpHill = 0f;
    private float T0, TF;
    private float currTemperature;
    private float d_avg = 0; // Average fitness change.
    private float alpha = 1f; // Temperature change.
    private float decision;

    /**
     * @param instance Solution seed.
     * @param mutator An object responsible for inducing mutations.
     * @param fe FitnessEvaluator object for evaluating fitness of individual
     * solutions.
     * @param numIter Number of iterations to run.
     */
    public SimulatedThermicAnnealing(
            Object instance,
            MutationInterface mutator,
            FitnessEvaluator fe,
            int numIter) {
        this.instance = instance;
        this.mutator = mutator;
        this.numIter = numIter;
        this.fe = fe;
    }

    /**
     * Calculate the linear temperature schedule. T0 and TF are the temperature
     * constants that correspond to p0 and pf probabilities at the beginning and
     * the end of the simulated annealing run. The linear change factor alpha is
     * calculated based on the number of iterations that the procedure is
     * expected to run, so that the current temperature becomes TF in the end.
     */
    private void calculateTemperatureSchedule() {
        T0 = -d_avg / (float) Math.log(p0);
        TF = -d_avg / (float) Math.log(pf);
        currTemperature = T0;
        alpha = (float) Math.pow(
                (TF / T0),
                1f / (float) (numIter - averageSeekingIter));
    }

    /**
     * Perform temperature update - in this case, by multiplying the current
     * temperature by the linear change factor.
     */
    private void updateTemperature() {
        currTemperature *= alpha;
    }

    /**
     * Get uphill probability based on current temperature.
     *
     * @param scoreDifference Difference in fitness between the solutions.
     */
    private void getProbability(float scoreDifference) {
        probUpHill = (float) (Math.pow(
                Math.E, -scoreDifference / (currTemperature * d_avg)));
    }

    @Override
    public void optimize() throws Exception {
        Random randa = new Random();
        averageSeekingIter = Math.min(25, (int) ((float) (numIter) * 0.1f));
        averageSeekingIter = Math.max(averageSeekingIter, 5);
        while (!stop && ++iteration <= averageSeekingIter) {
            evaluate(instance);
            if (iteration > 1) {
                d_avg += Math.abs(score - previousScore);
            }
            replacePrevious();
            instance = mutator.mutateNew(previousInstance);
        }
        d_avg /= (averageSeekingIter - 1);
        calculateTemperatureSchedule();
        evaluate(instance);
        while (!stop && ++iteration <= numIter) {
            if (score < previousScore) {
                replacePrevious();
            } else {
                decision = randa.nextFloat();
                getProbability(score - previousScore);
                if (decision < probUpHill) {
                    // Up-hill movement which sometimes occurs to escape local
                    // optima.
                    replacePrevious();
                }
            }
            instance = mutator.mutateNew(previousInstance);
            evaluate(instance);
            updateTemperature();
        }
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
