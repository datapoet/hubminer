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
import data.representation.DataSet;
import java.util.ArrayList;
import java.util.Random;
import optimization.stochastic.fitness.FitnessEvaluator;

/**
 * This class implements the standard predator-prey particle swarm optimization
 * method. The methods are only applicable to float search spaces.
 * 
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class PPPSO implements OptimizationAlgorithmInterface {
    
    private int numIter = 1000;
    private int iteration = 0;
    private FitnessEvaluator fe;
    private DataInstance bestInstance;
    private DataInstance worstInstance;
    float[] upperValueLimits;
    float[] lowerValueLimits;
    ArrayList<DataInstance> preyPopulation;
    DataInstance predatorInstance;
    ArrayList<DataInstance> preyVelocities;
    DataInstance predatorVelocity;
    ArrayList<DataInstance> bestPreySolutions;
    DataInstance bestPredatorSolution;
    ArrayList<Float> bestPreyScores;
    int indexOfBestThisIteration = 0;
    float bestPredatorScore;
    DataSet populationContext;
    private int numDim = 0;
    private int numEvaluatedInstances = 0;
    private double totalScore = 0;
    private float bestScore = Float.MAX_VALUE;
    private float worstScore = -Float.MAX_VALUE;
    private float score = Float.MAX_VALUE;
    private boolean stop = false;
    private int populationSize = 10;
    private Random randa = new Random();
    // Fear parameters.
    double amplitude;
    double expParam = 10;
    // Velocity update factors.
    double factPast = 0.1;
    double factGlobal = 0.15;
    double factFear = 0.15;
    
    /**
     * Initialization.
     * 
     * @param lowerValueLimits float[] representing the lower value limits.
     * @param upperValueLimits float[] representing the upper value limits.
     * @param populationSize Integer that is the prey population size.
     */
    public PPPSO(float[] lowerValueLimits, float[] upperValueLimits,
            int populationSize) {
        this.lowerValueLimits = lowerValueLimits;
        this.upperValueLimits = upperValueLimits;
        if (lowerValueLimits != null) {
            numDim = lowerValueLimits.length;
        } else if (upperValueLimits != null) {
            numDim = upperValueLimits.length;
        }
        this.populationSize = populationSize;
    }
    
    /**
     * Initialization.
     * 
     * @param lowerValueLimits float[] representing the lower value limits.
     * @param upperValueLimits representing the upper value limits.
     * @param preyPopulation ArrayList<DataInstance> that is the initial prey
     * population.
     * @param predatorInstance DataInstance that is the initial predator
     * @param populationContext DataSet representing the data context with
     * feature definitions.
     * instance.
     */
    public PPPSO(float[] lowerValueLimits, float[] upperValueLimits,
            ArrayList<DataInstance> preyPopulation,
            DataInstance predatorInstance, DataSet populationContext) {
        this.lowerValueLimits = lowerValueLimits;
        this.upperValueLimits = upperValueLimits;
        if (lowerValueLimits != null) {
            numDim = lowerValueLimits.length;
        } else if (upperValueLimits != null) {
            numDim = upperValueLimits.length;
        }
        this.predatorInstance = predatorInstance;
        this.preyPopulation = preyPopulation;
        this.populationContext = populationContext;
        if (preyPopulation != null) {
            populationSize = preyPopulation.size();
        }
    }
    
    /**
     * Initialization.
     * 
     * @param lowerValueLimits float[] representing the lower value limits.
     * @param upperValueLimits float[] representing the upper value limits.
     * @param populationSize Integer that is the prey population size.
     * @param fe FitnessEvaluator for solution fitness evaluation.
     */
    public PPPSO(float[] lowerValueLimits, float[] upperValueLimits,
            int populationSize, FitnessEvaluator fe) {
        this.lowerValueLimits = lowerValueLimits;
        this.upperValueLimits = upperValueLimits;
        if (lowerValueLimits != null) {
            numDim = lowerValueLimits.length;
        } else if (upperValueLimits != null) {
            numDim = upperValueLimits.length;
        }
        this.populationSize = populationSize;
        this.fe = fe;
    }
    
    /**
     * Initialization.
     * 
     * @param lowerValueLimits float[] representing the lower value limits.
     * @param upperValueLimits representing the upper value limits.
     * @param preyPopulation ArrayList<DataInstance> that is the initial prey
     * population.
     * @param predatorInstance DataInstance that is the initial predator
     * instance.
     * @param populationContext DataSet representing the data context with
     * feature definitions.
     * @param fe FitnessEvaluator for solution fitness evaluation.
     */
    public PPPSO(float[] lowerValueLimits, float[] upperValueLimits,
            ArrayList<DataInstance> preyPopulation,
            DataInstance predatorInstance, DataSet populationContext,
            FitnessEvaluator fe) {
        this.lowerValueLimits = lowerValueLimits;
        this.upperValueLimits = upperValueLimits;
        if (lowerValueLimits != null) {
            numDim = lowerValueLimits.length;
        } else if (upperValueLimits != null) {
            numDim = upperValueLimits.length;
        }
        this.predatorInstance = predatorInstance;
        this.preyPopulation = preyPopulation;
        this.populationContext = populationContext;
        if (preyPopulation != null) {
            populationSize = preyPopulation.size();
        }
        this.fe = fe;
    }
    
    @Override
    public void optimize() throws Exception {
        assertValueLimits();
        initializePopulation();
        learnFearAmplitude();
        initializeVelocities();
        double[] dimPercs = new double[numDim];
        double dimSpanTotal = 0;
        for (int d = 0; d < numDim; d++) {
             dimPercs[d] = upperValueLimits[d] - lowerValueLimits[d];
             dimSpanTotal += dimPercs[d];
        }
        if (dimSpanTotal > 0) {
            for (int d = 0; d < numDim; d++) {
                dimPercs[d] /= dimSpanTotal;
            }
        }
        // The 0th iteration was already implicitly performed in
        // initializations.
        for (iteration = 1; iteration < numIter; iteration++) {
            // Check for the stop criterion.
            if (stop) {
                break;
            }
            float inertiaWeight = getInertia();
            // First update the predator velocity.
            predatorVelocity = new DataInstance(populationContext);
            float chaseFactor = randa.nextFloat();
            for (int d = 0; d < numDim; d++) {
                predatorVelocity.fAttr[d] = chaseFactor * (
                        preyPopulation.get(indexOfBestThisIteration).fAttr[d] -
                        predatorInstance.fAttr[d]);
            }
            // Update prey velocities.
            for (int i = 0; i < populationSize; i++) {
                double predatorDistance = dist(preyPopulation.get(i),
                        predatorInstance);
                for (int d = 0; d < numDim; d++) {
                    preyVelocities.get(i).fAttr[d] = (float)(Math.min((
                            preyVelocities.get(i).fAttr[d] * inertiaWeight +
                            factPast * (bestPreySolutions.get(i).fAttr[d] -
                            preyPopulation.get(i).fAttr[d]) + factGlobal * (
                            preyPopulation.get(indexOfBestThisIteration).
                            fAttr[d] - preyPopulation.get(i).fAttr[d]) +
                            factFear * dimPercs[d] * fear(predatorDistance)),
                        upperValueLimits[d] * (0.8 + 0.2 * randa.nextFloat()))
                    );
                }
            }
            // Update predator instance.
            predatorInstance =
                    applyVelocity(predatorInstance, predatorVelocity);
            // Update best predator instance.
            score = evaluate(predatorInstance);
            if (score > bestPredatorScore) {
                bestPredatorScore = score;
                bestPredatorSolution = predatorInstance;
            }
            // Update prey instances.
            float bestIterScore = -Float.MAX_VALUE;
            for (int i = 0; i < populationSize; i++) {
                DataInstance newSolution = applyVelocity(preyPopulation.get(i),
                        preyVelocities.get(i));
                ensureProperValues(newSolution);
                preyPopulation.set(i, newSolution);
                score = evaluate(predatorInstance);
                if (score > bestIterScore) {
                    bestIterScore = score;
                    indexOfBestThisIteration = i;
                }
                if (score > bestPreyScores.get(i)) {
                    bestPreyScores.set(i, score);
                    bestPreySolutions.set(i, newSolution);
                }
            }
        }
    }
    
    /**
     * This method calculates the inertia weight.
     * 
     * @return Float value that is the inertia weight. 
     */
    private float getInertia() {
        return ((float)getNumIter() - (float)getIteration()) /
                ((float)getNumIter());
    }
    
    /**
     * This method initializes the point velocities.
     */
    private void initializeVelocities() {
        double[] dimSpan = new double[numDim];
        for (int d = 0; d < numDim; d++) {
            dimSpan[d] = upperValueLimits[d] - lowerValueLimits[d];
        }
        preyVelocities = new ArrayList<>(populationSize);
        predatorVelocity = new DataInstance(populationContext);
        for (int d = 0; d < numDim; d++) {
            predatorVelocity.fAttr[d] = (float)(0.3 * dimSpan[d] *
                    randa.nextFloat());
        }
        for (int i = 0; i < populationSize; i++) {
            DataInstance preyVelocity = new DataInstance(populationContext);
            for (int d = 0; d < numDim; d++) {
                preyVelocity.fAttr[d] = (float)(0.3 * dimSpan[d] *
                        randa.nextFloat());
            }
            preyVelocities.add(preyVelocity);
        }
    }
    
    /**
     * This method applies the velocity to a DataInstance solution and returns
     * the updated solution. The original solution object is not modified.
     * 
     * @param solution DataInstance object that is the current solution.
     * @param velocity DataInstance representing the velocity.
     * @return DataInstance that is the updated solution.
     * @throws Exception 
     */
    private DataInstance applyVelocity(DataInstance solution,
            DataInstance velocity) throws Exception {
        if (solution == null) {
            return null;
        } else {
            DataInstance solutionUpdate = solution.copy();
            for (int d = 0; d < numDim; d++) {
                solutionUpdate.fAttr[d] += velocity.fAttr[d];
            }
            ensureProperValues(solutionUpdate);
            return solutionUpdate;
        }
    }
    
    /**
     * This method calculates the distance between two solutions.
     * 
     * @param first DataInstance that is the first solution.
     * @param second DataInstance that is the second solution.
     * @return 
     */
    private double dist(DataInstance first, DataInstance second) {
        double manhattanDist = 0;
        for (int d = 0; d < numDim; d++) {
            manhattanDist += Math.abs(first.fAttr[d] - second.fAttr[d]);
        }
        return manhattanDist;
    }
    
    /**
     * This method sets the maximum amplitude to the mean pairwise distance.
     */
    private void learnFearAmplitude() {
        amplitude = 0;
        for (int i = 0; i < populationSize; i++) {
            for (int j = i + 1; j < populationSize; j++) {
                amplitude += dist(preyPopulation.get(i), preyPopulation.get(j));
            }
        }
        if (populationSize > 1) {
            amplitude /= (populationSize * (populationSize - 1) / 2);
        }
    }
    
    /**
     * This method calculates the repellant force based on the provided
     * Manhattan distance.
     * 
     * @param distance Double representing the Manhattan distance between the
     * points.
     * 
     * @return Double value that is the fear of the predator at the specified
     * distance.
     */
    private double fear(double distance) {
        return 0.5 * amplitude * Math.exp(- expParam * (distance / amplitude));
    }
    
    /**
     * This method ensures that all the values within the instance are within
     * the specified range.
     * 
     * @param instance 
     */
    private void ensureProperValues(DataInstance instance) {
        if (lowerValueLimits != null && upperValueLimits != null) {
            for (int d = 0; d < numDim; d++) {
                double compensator = upperValueLimits[d] - lowerValueLimits[d];
                double iVal = instance.fAttr[d];
                while (iVal > upperValueLimits[d]) {
                    iVal -= compensator;
                }
                while (iVal < lowerValueLimits[d]) {
                    iVal += compensator;
                }
                instance.fAttr[d] = (float)iVal;
            }
        }
    }
    
    /**
     * This method checks whether the lower limits are lower than the upper
     * limits.
     * 
     * @throws Exception 
     */
    private void assertValueLimits() throws Exception {
        if (lowerValueLimits != null && upperValueLimits != null) {
            for (int d = 0; d < numDim; d++) {
                if (lowerValueLimits[d] > upperValueLimits[d]) {
                    throw new Exception("Incorrect value range specified, since"
                            + lowerValueLimits[d] + " is greater than " +
                            upperValueLimits[d] + " for dimension " + d + ".");
                }
            }
        }
    }
    
    /**
     * Initializes the predator and prey population.
     */
    private void initializePopulation() {
        if (populationContext == null) {
            populationContext = new DataSet();
            String[] fNames = new String[numDim];
            for (int d = 0; d < numDim; d++) {
                fNames[d] = "fAtt" + d;
            }
            populationContext.fAttrNames = fNames;
        }
        if (predatorInstance == null) {
            predatorInstance = generateRandomInstance();
        }
        if (preyPopulation == null) {
            preyPopulation = new ArrayList<>(populationSize);
            for (int i = 0; i < populationSize; i++) {
                preyPopulation.add(generateRandomInstance());
            }
        }
        bestPreySolutions = new ArrayList<>(populationSize);
        bestPreyScores = new ArrayList<>(populationSize);
        float currBestScore = -Float.MAX_VALUE;
        for (int i = 0; i < populationSize; i++) {
            bestPreySolutions.add(preyPopulation.get(i));
            bestPreyScores.add(evaluate(preyPopulation.get(i)));
            if (bestPreyScores.get(i) > currBestScore) {
                currBestScore = bestPreyScores.get(i);
                indexOfBestThisIteration = i;
            }
        }
        bestPredatorSolution = predatorInstance;
        bestPredatorScore = evaluate(predatorInstance);
    }
    
    /**
     * This method generates a random instance in the range.
     * 
     * @return DataInstance placed randomly in the search space. 
     */
    private DataInstance generateRandomInstance() {
        if (populationContext == null || lowerValueLimits == null ||
                upperValueLimits == null) {
            return null;
        } else {
            DataInstance instance = new DataInstance(populationContext);
            for (int d = 0; d < numDim; d++) {
                // Here we use double to avoid landing out of range if maximal
                // values for float are specified as limits.
                double rVal = lowerValueLimits[d] + (
                        upperValueLimits[d] - lowerValueLimits[d]) *
                        randa.nextFloat();
                instance.fAttr[d] = (float)rVal;
            }
            return instance;
        }
    }
    
    /**
     * Evaluates the fitness of the current solution and updates the best/worst
     * solution fitness stats.
     *
     * @param instance
     * @return
     */
    private float evaluate(DataInstance instance) {
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
