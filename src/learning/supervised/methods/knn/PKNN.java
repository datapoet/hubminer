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
package learning.supervised.methods.knn;

import data.neighbors.NeighborSetFinder;
import data.representation.DataInstance;
import data.representation.DataSet;
import distances.primary.CombinedMetric;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Random;
import learning.supervised.Category;
import learning.supervised.Classifier;
import learning.supervised.evaluation.ValidateableInterface;

/**
 * This class implements the probabilistic KNN classification algorithm that was
 * described in the paper titled "A probabilistic nearest neighbor method for
 * statistical pattern recognition that was authored by C. C. Holmes and N. M.
 * Adams from the Imperial College of Science, Technology and Medicine, London,
 * UK, 2002. In short, a Bayesian nearest neighbor model is approximated by a
 * Metropolis Markov-Chain Monte-Carlo technique. Basically, k and beta are
 * sampled in this fashion from the underlying distribution - and an averaged
 * decision is then made.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class PKNN extends Classifier implements Serializable {

    private static final long serialVersionUID = 1L;
    // The upper k-range.
    private int kMax = 500;
    // Object that does kNN calculations.
    private NeighborSetFinder nsf = null;
    private int numCategories = 0;
    // Local class counts.
    private float[] localCatCount;
    // The kNN set.
    private int[][] kneighbors = null;
    // Training data.
    private DataSet trainingData;
    // The sample chains.
    private ArrayList<Integer> kSamples;
    private ArrayList<Float> betaSamples;
    private static final int NUM_ITERATIONS = 3000;
    private static final int MIN_ITERATIONS = 1500;
    private static final int MIN_SAMPLE_SIZE = 3000;
    private static final int EARLY_STOP_K_DELTA = 3;
    private static final float EARLY_STOP_BETA_DELTA = 0.3f;
    
    @Override
    public HashMap<String, String> getParameterNamesAndDescriptions() {
        HashMap<String, String> paramMap = new HashMap<>();
        paramMap.put("kMax", "Maximum neighborhood size to consider.");
        return paramMap;
    }
    
    @Override
    public long getVersion() {
        return serialVersionUID;
    }

    @Override
    public String getName() {
        return "PNN";
    }

    /**
     * The default constructor.
     */
    public PKNN() {
    }

    /**
     * Initialization.
     *
     * @param kMax Integer that is the upper neighborhood size limit.
     */
    public PKNN(int kMax) {
        this.kMax = kMax;
    }

    /**
     * Initialization.
     *
     * @param kMax Integer that is the upper neighborhood size limit.
     * @param cmet CombinedMetric object for distance calculations.
     */
    public PKNN(int kMax, CombinedMetric cmet) {
        this.kMax = kMax;
        setCombinedMetric(cmet);
    }

    /**
     * Initialization.
     *
     * @param kMax Integer that is the upper neighborhood size limit.
     * @param cmet CombinedMetric object for distance calculations.
     * @param numCategories Integer that is the number of classes in the data.
     */
    public PKNN(int kMax, CombinedMetric cmet, int numCategories) {
        this.kMax = kMax;
        setCombinedMetric(cmet);
        this.numCategories = numCategories;
    }

    /**
     * Initialization.
     *
     * @param cmet CombinedMetric object for distance calculations.
     * @param numCategories Integer that is the number of classes in the data.
     */
    public PKNN(CombinedMetric cmet, int numCategories) {
        setCombinedMetric(cmet);
        this.numCategories = numCategories;
    }

    /**
     * This method checks if the upper k limit is reasonable given the data size
     * and updates it if necessary.
     */
    private void validateKMax() {
        kMax = Math.min(kMax, trainingData.size() / 4);
    }

    /**
     * This method calculates the probability of data labels given the current k
     * and beta choice.
     *
     * @param k Integer that is the neighborhood size.
     * @param beta Float that regulates the strength between nearest neighbors.
     * @return Double that is the probability of the labels given the current
     * parameter choice.
     */
    private double probabilityOfDataLabeling(int k, float beta) {
        if (localCatCount == null) {
            localCatCount = new float[numCategories];
        }
        double probability = 1;
        for (int i = 0; i < trainingData.size(); i++) {
            Arrays.fill(localCatCount, 0);
            // Calculate the local class frequencies.
            for (int kIndex = 0; kIndex < k; kIndex++) {
                localCatCount[trainingData.data.get(kneighbors[i][kIndex]).
                        getCategory()]++;
            }
            double denominator = 0;
            double betaRatio = beta / k;
            // Calculate the denominator.
            for (int cIndex = 0; cIndex < numCategories; cIndex++) {
                denominator += Math.exp(betaRatio * localCatCount[cIndex]);
            }
            // Calculate the estimate of the data label probability.
            probability *= Math.exp(betaRatio * localCatCount[
                    trainingData.data.get(i).getCategory()]) / denominator;
        }
        return probability;
    }

    @Override
    public void setClasses(Category[] categories) {
        int totalSize = 0;
        int indexFirstNonEmptyClass = -1;
        for (int cIndex = 0; cIndex < categories.length; cIndex++) {
            totalSize += categories[cIndex].size();
            if (indexFirstNonEmptyClass == -1 &&
                    categories[cIndex].size() > 0) {
                indexFirstNonEmptyClass = cIndex;
            }
        }
        // Instances are not embedded in the internal data context.
        trainingData = new DataSet();
        trainingData.fAttrNames = categories[indexFirstNonEmptyClass].
                getInstance(0).getEmbeddingDataset().fAttrNames;
        trainingData.iAttrNames = categories[indexFirstNonEmptyClass].
                getInstance(0).getEmbeddingDataset().iAttrNames;
        trainingData.sAttrNames = categories[indexFirstNonEmptyClass].
                getInstance(0).getEmbeddingDataset().sAttrNames;
        trainingData.data = new ArrayList<>(totalSize);
        for (int cIndex = 0; cIndex < categories.length; cIndex++) {
            for (int i = 0; i < categories[cIndex].size(); i++) {
                categories[cIndex].getInstance(i).setCategory(cIndex);
                trainingData.addDataInstance(categories[cIndex].getInstance(i));
            }
        }
        numCategories = trainingData.countCategories();
    }

    /**
     * @param nsf NeighborSetFinder object for kNN calculations.
     */
    public void setNSF(NeighborSetFinder nsf) {
        this.nsf = nsf;
    }

    /**
     * @return NeighborSetFinder object for kNN calculations.
     */
    public NeighborSetFinder getNSF() {
        return nsf;
    }

    /**
     * @return DataSet object that is the training data.
     */
    public DataSet getTrainingSet() {
        return trainingData;
    }

    /**
     * @param trainingData DataSet object that is the training data.
     */
    public void setTrainingSet(DataSet trainingData) {
        this.trainingData = trainingData;
    }

    /**
     * @param numClasses Integer that is the number of classes in the data.
     */
    public void setNumClasses(int numClasses) {
        this.numCategories = numClasses;
    }

    /**
     * @return Integer that is the number of classes in the data.
     */
    public int getNumClasses() {
        return numCategories;
    }

    /**
     * @return Integer that is the upper k limit.
     */
    public int getKMax() {
        return kMax;
    }

    /**
     * @param kMax Integer that is the upper k limit.
     */
    public void setKMax(int kMax) {
        this.kMax = kMax;
    }

    /**
     * This method calculate the kNN sets up to the upper k limit.
     *
     * @throws Exception
     */
    public void calculateNeighborSets() throws Exception {
        nsf = new NeighborSetFinder(trainingData, getCombinedMetric());
        nsf.calculateDistances();
        nsf.calculateNeighborSets(kMax);
        kneighbors = nsf.getKNeighbors();
    }

    @Override
    public ValidateableInterface copyConfiguration() {
        PKNN result = new PKNN(kMax, getCombinedMetric());
        result.setNumClasses(numCategories);
        return result;
    }

    /**
     * Mutate the beta value.
     *
     * @param betaValue Float that is the beta value to mutate.
     * @param variance Float that is the variance to use.
     * @param randa Random number generator.
     * @return The mutated beta value.
     */
    private float mutateBeta(float betaValue, float variance, Random randa) {
        float betaMutated = betaValue + variance * (float) randa.nextGaussian();
        betaMutated = Math.max(betaMutated, 0.01f);
        betaMutated = Math.min(betaMutated, 10);
        return betaMutated;
    }

    /**
     * Mutate the k value.
     *
     * @param kValue Integer that is the neighborhood size to mutate.
     * @param randa Random number generator.
     * @return The mutated k value.
     */
    private int mutateK(int kValue, Random randa) {
        int kMutated = kValue;
        float choice = randa.nextFloat();
        if (choice < 0.5f) {
            kMutated += (randa.nextInt(4) + 1);
        } else {
            kMutated -= (randa.nextInt(4) + 1);
        }
        kMutated = Math.min(kMax, kMutated);
        kMutated = Math.max(1, kMutated);
        return kMutated;
    }

    /**
     * Calculate the decision quotient based on the difference in probabilities
     * over subsequent iterations.
     *
     * @param previousProb Double that is the previous probability.
     * @param newProb Double that is the new probability.
     * @return Double that is the decision probability factor.
     */
    private double getDecisionProb(double previousProb, double newProb) {
        if (previousProb == 0) {
            return 1;
        }
        double quotient = newProb / previousProb;
        if (Double.isInfinite(quotient) || Double.isNaN(quotient)) {
            return 1;
        }
        return Math.min(1, quotient);
    }

    @Override
    public void train() throws Exception {
        validateKMax();
        calculateNeighborSets();
        // Two chains are used that start from different locations. The beta
        // variance is controlled to achieve the 30% acceptance rate after the
        // first burn. For the first thousand checks on both chains, no sample
        // points are retained. After that, beta variance is no longer modified
        // and sampling is done to obtain 1000 evaluation points.
        kSamples = new ArrayList<>(1000);
        betaSamples = new ArrayList<>(1000);
        // Neighborhood size k is increased or decreased by U(0, 1, 2, 3).
        // Beta is incremented or decremented by a gaussian random value.
        int firstChainK = 1;
        float firstChainBeta = 10;
        int secondChainK = kMax;
        float secondChainBeta = 0.01f;
        float betaMutateVariance = 0.8f;
        double dsLabelProbFirst = probabilityOfDataLabeling(firstChainK,
                firstChainBeta);
        double dsLabelProbSecond = probabilityOfDataLabeling(secondChainK,
                secondChainBeta);
        double currDSLabelProb;
        int currK;
        float currBeta;
        double choice;
        Random randa = new Random();
        // To simplify, the acceptance rate will be checked just in one chain,
        // over the last 30 samples.
        boolean[] acceptanceMemory = new boolean[30];
        Arrays.fill(acceptanceMemory, true);
        int accepted = 30;
        float firstChainKMean = 1;
        float secondChainKMean = kMax;
        float firstChainBetaMean = 10;
        float secondChainBetaMean = 0.01f;
        for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
            // First chain.
            currK = mutateK(firstChainK, randa);
            currBeta = mutateBeta(firstChainBeta, betaMutateVariance, randa);
            currDSLabelProb = probabilityOfDataLabeling(currK, currBeta);
            choice = getDecisionProb(dsLabelProbFirst, currDSLabelProb);
            dsLabelProbFirst = currDSLabelProb;
            if (randa.nextDouble() < choice) {
                // Accepted.
                if (acceptanceMemory[iteration % 30] == false) {
                    accepted++;
                }
                acceptanceMemory[iteration % 30] = true;
                firstChainK = currK;
                firstChainBeta = currBeta;
            } else {
                // Rejected.
                if (acceptanceMemory[iteration % 30] == true) {
                    accepted--;
                }
                acceptanceMemory[iteration % 30] = false;
            }
            if (iteration > 30) {
                // Check to see how many have been accepted and adjust the
                // variance accordingly.
                if (accepted < 10) {
                    betaMutateVariance = Math.max(0.15f,
                            betaMutateVariance - 0.05f);
                }
                if (accepted > 10) {
                    betaMutateVariance = Math.min(1.2f,
                            betaMutateVariance + 0.05f);
                }
            }
            // Second chain.
            currK = mutateK(secondChainK, randa);
            currBeta = mutateBeta(secondChainBeta, betaMutateVariance, randa);
            currDSLabelProb = probabilityOfDataLabeling(currK, currBeta);
            choice = getDecisionProb(dsLabelProbSecond, currDSLabelProb);
            dsLabelProbSecond = currDSLabelProb;
            if (randa.nextDouble() < choice) {
                // Accept.
                secondChainK = currK;
                secondChainBeta = currBeta;
            }
            // Update the current mean values for k and beta.
            firstChainKMean = (firstChainKMean * (iteration + 1) + firstChainK)
                    / (iteration + 2);
            secondChainKMean = (secondChainKMean
                    * (iteration + 1) + secondChainK) / (iteration + 2);
            firstChainBetaMean = (firstChainBetaMean * (iteration + 1)
                    + firstChainBeta) / (iteration + 2);
            secondChainBetaMean = (secondChainBetaMean * (iteration + 1)
                    + secondChainBeta) / (iteration + 2);

            if (iteration > MIN_ITERATIONS) {
                // Early stop check.
                if ((firstChainKMean - secondChainKMean < EARLY_STOP_K_DELTA)
                        && (firstChainBetaMean - secondChainBetaMean
                        < EARLY_STOP_BETA_DELTA)) {
                    break;
                }
            }
        }
        // After the initial burn, make the sample from both chains.
        while (kSamples.size() < MIN_SAMPLE_SIZE / 2) {
            // Sample from the first chain.
            currK = mutateK(firstChainK, randa);
            currBeta = mutateBeta(firstChainBeta, betaMutateVariance, randa);
            currDSLabelProb = probabilityOfDataLabeling(currK, currBeta);
            choice = getDecisionProb(dsLabelProbFirst, currDSLabelProb);
            dsLabelProbFirst = currDSLabelProb;
            if (randa.nextDouble() < choice) {
                // Accept.
                firstChainK = currK;
                firstChainBeta = currBeta;
            }
            kSamples.add(firstChainK);
            betaSamples.add(firstChainBeta);
            // Sample from the second. chain.
            currK = mutateK(secondChainK, randa);
            currBeta = mutateBeta(secondChainBeta, betaMutateVariance, randa);
            currDSLabelProb = probabilityOfDataLabeling(currK, currBeta);
            choice = getDecisionProb(dsLabelProbSecond, currDSLabelProb);
            dsLabelProbSecond = currDSLabelProb;
            if (randa.nextDouble() < choice) {
                // Accept.
                secondChainK = currK;
                secondChainBeta = currBeta;
            }
            kSamples.add(secondChainK);
            betaSamples.add(secondChainBeta);
        }
    }

    @Override
    public int classify(DataInstance instance) throws Exception {
        float[] classProbs = classifyProbabilistically(instance);
        float maxProb = 0;
        int maxClassIndex = 0;
        for (int cIndex = 0; cIndex < numCategories; cIndex++) {
            if (classProbs[cIndex] > maxProb) {
                maxProb = classProbs[cIndex];
                maxClassIndex = cIndex;
            }
        }
        return maxClassIndex;
    }

    @Override
    public float[] classifyProbabilistically(DataInstance instance)
            throws Exception {
        CombinedMetric cmet = getCombinedMetric();
        if (trainingData == null) {
            throw new Exception("Bad classifier initialization.");
        }
        if (instance == null) {
            return null;
        }
        // Find the kNN set up until the upper k limit.
        int[] nearestInstances = new int[kMax];
        float[] nearestDistances = new float[kMax];
        Arrays.fill(nearestInstances, -1);
        Arrays.fill(nearestDistances, Float.MAX_VALUE);
        float currDist;
        int index;
        for (int i = 0; i < trainingData.size(); i++) {
            currDist = cmet.dist(instance, trainingData.data.get(i));
            index = kMax - 1;
            while (index >= 0 && nearestDistances[index] > currDist) {
                index--;
            }
            if (index < kMax - 1) {
                for (int j = kMax - 1; j > index + 1; j--) {
                    nearestDistances[j] = nearestDistances[j - 1];
                    nearestInstances[j] = nearestInstances[j - 1];
                }
                nearestInstances[index + 1] = i;
                nearestDistances[index + 1] = currDist;
            }
        }
        // Perform the weighted vote.
        float[] classProbabilitiesFinal = new float[numCategories];
        float[] classProbsIteration = new float[numCategories];

        float[][] classFreqsIteration = new float[kMax][numCategories];
        for (int k = 0; k < kMax; k++) {
            if (k >= 1) {
                classFreqsIteration[k] = Arrays.copyOf(
                        classFreqsIteration[k - 1], numCategories);
            }
            classFreqsIteration[k][trainingData.data.get(nearestInstances[k]).
                    getCategory()]++;
        }
        float probTotal;
        for (int i = 0; i < kSamples.size(); i++) {
            int k = kSamples.get(i);
            float beta = betaSamples.get(i);
            probTotal = 0;
            for (int cIndex = 0; cIndex < numCategories; cIndex++) {
                classProbsIteration[cIndex] = (float) Math.exp((beta / k)
                        * classFreqsIteration[k - 1][cIndex]);
                probTotal += classProbsIteration[cIndex];
            }
            for (int cIndex = 0; cIndex < numCategories; cIndex++) {
                classProbsIteration[cIndex] /= probTotal;
                classProbabilitiesFinal[cIndex] += classProbsIteration[cIndex];
            }
        }
        for (int cIndex = 0; cIndex < numCategories; cIndex++) {
            classProbabilitiesFinal[cIndex] /= (float) kSamples.size();
        }
        return classProbabilitiesFinal;
    }
}
