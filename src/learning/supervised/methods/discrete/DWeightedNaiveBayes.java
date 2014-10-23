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
package learning.supervised.methods.discrete;

import data.representation.discrete.DiscretizedDataInstance;
import data.representation.discrete.DiscretizedDataSet;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import learning.supervised.DiscreteCategory;
import learning.supervised.DiscreteClassifier;
import learning.supervised.evaluation.ValidateableInterface;

/**
 * This method implements the weighted version of the Naive Bayes classifier,
 * that supports arbitrary instance weights.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class DWeightedNaiveBayes extends DiscreteClassifier
implements Serializable {
    
    private static final long serialVersionUID = 1L;

    // The conditional probability distribution arrays.
    private float[][][] floatLabelDistributionsInClasses = null;
    private float[][][] intLabelDistributionsInClasses = null;
    private float[][][] nominalLabelDistributionsInClasses = null;
    private float[] classPriors = null;
    // Laplace estimator for distribution smoothing.
    private float laplaceEstimator = 1f;
    // Instance weights.
    private float[] weights = null;
    // Whether to use the instance weights for calculating the prior class
    // distribution as well.
    private boolean isWeightedApriori = true;
    
    @Override
    public HashMap<String, String> getParameterNamesAndDescriptions() {
        HashMap<String, String> paramMap = new HashMap<>();
        paramMap.put("laplaceEstimator", "Laplace estimator for smoothing."
                + "Defaults to 1.");
        paramMap.put("isWeightedApriori", "Whether instance weights are used"
                + "for calculating class priors.");
        return paramMap;
    }

    @Override
    public long getVersion() {
        return serialVersionUID;
    }
    
    @Override
    public String getName() {
        return "Weighted Naive Bayes";
    }


    /**
     * The default constructor.
     */
    public DWeightedNaiveBayes() {
    }

    
    /**
     * Initialization.
     * 
     * @param discDSet DiscretizedDataSet that is the training data.
     */
    public DWeightedNaiveBayes(DiscretizedDataSet discDSet) {
        setDataType(discDSet);
        generateClassesFromDataType();
    }

    
    /**
     * Initialization.
     * 
     * @param discDSet DiscretizedDataSet that is the training data context.
     * @param dataClasses DiscreteCategory[] that is the training data.
     */
    public DWeightedNaiveBayes(DiscretizedDataSet discDSet,
            DiscreteCategory[] dataClasses) {
        setDataType(discDSet);
        setClasses(dataClasses);
    }

    
    /**
     * Initialization.
     * 
     * @param discDSet DiscretizedDataSet that is the training data context.
     * @param dataClasses DiscreteCategory[] that is the training data.
     * @param weights float[] representing the instance weights.
     */
    public DWeightedNaiveBayes(DiscretizedDataSet discDSet,
            DiscreteCategory[] dataClasses, float[] weights) {
        setDataType(discDSet);
        setClasses(dataClasses);
        this.weights = weights;
    }

    
    /**
     * Initialization.
     * 
     * @param discDSet DiscretizedDataSet that is the training data context.
     * @param dataClasses DiscreteCategory[] that is the training data.
     * @param laplaceEstimator Float value that is the Laplace estimator for
     * probability distribution smoothing.
     */
    public DWeightedNaiveBayes(DiscretizedDataSet discDSet,
            DiscreteCategory[] dataClasses, float laplaceEstimator) {
        setDataType(discDSet);
        setClasses(dataClasses);
        this.laplaceEstimator = laplaceEstimator;
    }

    
    /**
     * Initialization.
     * 
     * @param discDSet DiscretizedDataSet that is the training data context.
     * @param dataClasses DiscreteCategory[] that is the training data.
     * @param laplaceEstimator Float value that is the Laplace estimator for
     * probability distribution smoothing.
     * @param weights float[] representing the instance weights.
     */
    public DWeightedNaiveBayes(DiscretizedDataSet discDSet,
            DiscreteCategory[] dataClasses, float laplaceEstimator,
            float[] weights) {
        setDataType(discDSet);
        setClasses(dataClasses);
        this.laplaceEstimator = laplaceEstimator;
        this.weights = weights;
    }

    
    /**
     * @param laplaceEstimator Float value that is the Laplace estimator for
     * probability distribution smoothing.
     */
    public void setLaplaceEstimator(float laplaceEstimator) {
        this.laplaceEstimator = laplaceEstimator;
    }


    @Override
    public ValidateableInterface copyConfiguration() {
        DWeightedNaiveBayes nbCopy = new DWeightedNaiveBayes();
        nbCopy.setLaplaceEstimator(laplaceEstimator);
        nbCopy.setDataType(getDataType());
        return nbCopy;
    }

    
    /**
     * @param weights float[] representing the instance weights.
     */
    public void setWeights(float[] weights) {
        this.weights = weights;
    }


    @Override
    public void train() throws Exception {
        DiscretizedDataInstance instance;
        // Fetch the data.
        DiscretizedDataSet discDSet = getDataType();
        DiscreteCategory[] dataClasses = getClasses();
        if (weights == null) {
            weights = new float[discDSet.size()];
            Arrays.fill(weights, 1);
        }
        // Count the attributes.
        int intSize = discDSet.getNumIntAttr();
        int floatSize = discDSet.getNumFloatAttr();
        int nominalSize = discDSet.getNumNominalAttr();
        // Get the interval definitions and the vocabularies.
        ArrayList<String>[] nominalVocabularies =
                discDSet.getNominalVocabularies();
        // The intervals are left-inclusive and ordered.
        int[][] intIntervalDivisions = discDSet.getIntIntervalDivisions();
        float[][] floatIntervalDivisions = discDSet.getFloatIntervalDivisions();
        // Initialize the conditional probability distribution matrices.
        if (floatSize > 0) {
            floatLabelDistributionsInClasses = new float[dataClasses.length][
                    floatIntervalDivisions.length][];
        }
        if (intSize > 0) {
            intLabelDistributionsInClasses = new float[dataClasses.length][
                    intIntervalDivisions.length][];
        }
        if (nominalSize > 0) {
            nominalLabelDistributionsInClasses = new float[dataClasses.length][
                    nominalVocabularies.length][];
        }
        for (int attIndex = 0; attIndex < floatSize; attIndex++) {
            for (int cIndex = 0; cIndex < dataClasses.length; cIndex++) {
                if (floatIntervalDivisions[attIndex] != null &&
                        floatIntervalDivisions[attIndex].length > 0) {
                    floatLabelDistributionsInClasses[cIndex][attIndex] =
                            new float[floatIntervalDivisions[attIndex].length];
                } else {
                    floatLabelDistributionsInClasses[cIndex][attIndex] = null;
                }
            }
        }
        for (int attIndex = 0; attIndex < intSize; attIndex++) {
            for (int cIndex = 0; cIndex < dataClasses.length; cIndex++) {
                if (intIntervalDivisions[attIndex] != null &&
                        intIntervalDivisions[attIndex].length > 0) {
                    intLabelDistributionsInClasses[cIndex][attIndex] =
                            new float[intIntervalDivisions[attIndex].length];
                } else {
                    intLabelDistributionsInClasses[cIndex][attIndex] = null;
                }
            }
        }
        for (int attIndex = 0; attIndex < nominalSize; attIndex++) {
            for (int cIndex = 0; cIndex < dataClasses.length; cIndex++) {
                if (nominalVocabularies[attIndex] != null &&
                        nominalVocabularies[attIndex].size() > 0) {
                    nominalLabelDistributionsInClasses[cIndex][attIndex] =
                            new float[nominalVocabularies[attIndex].size()];
                } else {
                    nominalLabelDistributionsInClasses[cIndex][attIndex] = null;
                }
            }
        }
        float[] classTotals = new float[dataClasses.length];
        float datasetTotal = 0;
        // End of initialization. Now count the frequencies in order to estimate
        // the class-conditional probabilities.
        for (int cIndex = 0; cIndex < dataClasses.length; cIndex++) {
            if (dataClasses[cIndex] != null && dataClasses[cIndex].
                    indexes.size() > 0) {
                for (int i = 0; i < dataClasses[cIndex].indexes.size(); i++) {
                    classTotals[cIndex] += weights[dataClasses[cIndex].
                            indexes.get(i)];
                    datasetTotal += weights[dataClasses[cIndex].indexes.get(i)];
                    instance = discDSet.data.get(dataClasses[cIndex].
                            indexes.get(i));
                    for (int attIndex = 0; attIndex < floatSize; attIndex++) {
                        floatLabelDistributionsInClasses[cIndex][attIndex][
                                instance.floatIndexes[attIndex]] += weights[
                                dataClasses[cIndex].indexes.get(i)];
                    }
                    for (int attIndex = 0; attIndex < intSize; attIndex++) {
                        intLabelDistributionsInClasses[cIndex][attIndex][
                                instance.integerIndexes[attIndex]] += weights[
                                dataClasses[cIndex].indexes.get(i)];
                    }
                    for (int attIndex = 0; attIndex < nominalSize; attIndex++) {
                        nominalLabelDistributionsInClasses[cIndex][attIndex][
                                instance.nominalIndexes[attIndex]] += weights[
                                dataClasses[cIndex].indexes.get(i)];
                    }
                }
            }
        }
         // Normalization and smoothing.
        for (int cIndex = 0; cIndex < dataClasses.length; cIndex++) {
            if (dataClasses[cIndex] != null &&
                    dataClasses[cIndex].indexes.size() > 0) {
                for (int attIndex = 0; attIndex < floatSize; attIndex++) {
                    if (floatLabelDistributionsInClasses[cIndex][attIndex]
                            != null) {
                        for (int j = 0; j < floatLabelDistributionsInClasses[
                                cIndex][attIndex].length; j++) {
                            floatLabelDistributionsInClasses[cIndex][
                                    attIndex][j] += laplaceEstimator;
                            floatLabelDistributionsInClasses[cIndex][
                                    attIndex][j] /= ((float)
                                    floatLabelDistributionsInClasses[cIndex][
                                    attIndex].length * laplaceEstimator +
                                    classTotals[cIndex]);
                        }
                    }
                }
                for (int attIndex = 0; attIndex < intSize; attIndex++) {
                    if (intLabelDistributionsInClasses[cIndex][
                            attIndex] != null) {
                        for (int j = 0; j < intLabelDistributionsInClasses[
                                cIndex][attIndex].length; j++) {
                            intLabelDistributionsInClasses[cIndex][
                                    attIndex][j] += laplaceEstimator;
                            intLabelDistributionsInClasses[cIndex][
                                    attIndex][j] /= ((float)
                                    intLabelDistributionsInClasses[cIndex][
                                    attIndex].length * laplaceEstimator +
                                    classTotals[cIndex]);
                        }
                    }
                }
                for (int attIndex = 0; attIndex < nominalSize; attIndex++) {
                    if (nominalLabelDistributionsInClasses[cIndex][attIndex]
                            != null) {
                        for (int j = 0; j < nominalLabelDistributionsInClasses[
                                cIndex][attIndex].length; j++) {
                            nominalLabelDistributionsInClasses[cIndex][
                                    attIndex][j] += laplaceEstimator;
                            nominalLabelDistributionsInClasses[cIndex][
                                    attIndex][j] /= ((float)
                                    nominalLabelDistributionsInClasses[
                                    cIndex][attIndex].length *
                                    laplaceEstimator + classTotals[cIndex]);
                        }
                    }
                }
            }
        }
        // Calculate the smoothed class priors.
        classPriors = new float[dataClasses.length];
        if (!isWeightedApriori) {
            // The non-weighted priors.
            for (int cIndex = 0; cIndex < dataClasses.length; cIndex++) {
                if (dataClasses[cIndex] != null &&
                        dataClasses[cIndex].indexes.size() > 0) {
                    classPriors[cIndex] = ((float)
                            dataClasses[cIndex].indexes.size() +
                            laplaceEstimator) / (discDSet.data.size() +
                            (float) dataClasses.length * laplaceEstimator);
                } else {
                    classPriors[cIndex] = laplaceEstimator /
                            ((float) discDSet.data.size() +
                            (float) dataClasses.length * laplaceEstimator);
                }
            }
        } else {
            // The weighted priors.
            for (int cIndex = 0; cIndex < dataClasses.length; cIndex++) {
                if (dataClasses[cIndex] != null && dataClasses[cIndex].
                        indexes.size() > 0) {
                    classPriors[cIndex] = (classTotals[cIndex] +
                            laplaceEstimator) / (datasetTotal +
                            (float) dataClasses.length * laplaceEstimator);
                } else {
                    classPriors[cIndex] = laplaceEstimator /
                            (datasetTotal + (float) dataClasses.length *
                            laplaceEstimator);
                }
            }
        }

    }

    
    
    @Override
    public int classify(DiscretizedDataInstance instance) throws Exception {
        DiscreteCategory[] dataClasses = getClasses();
        double[] posteriorClassProbs = new double[dataClasses.length];
        for (int cIndex = 0; cIndex < dataClasses.length; cIndex++) {
            posteriorClassProbs[cIndex] = classPriors[cIndex];
        }
        if (instance.floatIndexes != null) {
            for (int attIndex = 0; attIndex < instance.floatIndexes.length;
                    attIndex++) {
                for (int cIndex = 0; cIndex < dataClasses.length; cIndex++) {
                    if (floatLabelDistributionsInClasses[cIndex][attIndex] !=
                            null && floatLabelDistributionsInClasses[cIndex][
                            attIndex].length > 0) {
                        posteriorClassProbs[cIndex] *=
                                floatLabelDistributionsInClasses[cIndex][
                                attIndex][instance.floatIndexes[attIndex]];
                    }
                }
            }
        }
        if (instance.integerIndexes != null) {
            for (int attIndex = 0; attIndex < instance.integerIndexes.length;
                    attIndex++) {
                for (int cIndex = 0; cIndex < dataClasses.length; cIndex++) {
                    if (intLabelDistributionsInClasses[cIndex][attIndex] !=
                            null && intLabelDistributionsInClasses[cIndex][
                            attIndex].length > 0) {
                        posteriorClassProbs[cIndex] *=
                                intLabelDistributionsInClasses[cIndex][
                                attIndex][instance.integerIndexes[attIndex]];
                    }
                }
            }
        }
        if (instance.nominalIndexes != null) {
            for (int attIndex = 0; attIndex < instance.nominalIndexes.length;
                    attIndex++) {
                for (int cIndex = 0; cIndex < dataClasses.length; cIndex++) {
                    if (nominalLabelDistributionsInClasses[cIndex][attIndex] !=
                            null && nominalLabelDistributionsInClasses[cIndex][
                            attIndex].length > 0) {
                        posteriorClassProbs[cIndex] *=
                                nominalLabelDistributionsInClasses[cIndex][
                                attIndex][instance.nominalIndexes[attIndex]];
                    }
                }
            }
        }
        double maxProb = 0;
        int maxClassIndex = 0;
        for (int cIndex = 0; cIndex < dataClasses.length; cIndex++) {
            if (posteriorClassProbs[cIndex] > maxProb) {
                maxProb = posteriorClassProbs[cIndex];
                maxClassIndex = cIndex;
            }
        }
        return maxClassIndex;
    }

    
    @Override
    public float[] classifyProbabilistically(DiscretizedDataInstance instance)
            throws Exception {
        DiscreteCategory[] dataClasses = getClasses();
        double[] posteriorClassProbs = new double[dataClasses.length];
        for (int cIndex = 0; cIndex < dataClasses.length; cIndex++) {
            posteriorClassProbs[cIndex] = classPriors[cIndex];
        }
        if (instance.floatIndexes != null) {
            for (int attIndex = 0; attIndex < instance.floatIndexes.length;
                    attIndex++) {
                for (int cIndex = 0; cIndex < dataClasses.length; cIndex++) {
                    if (floatLabelDistributionsInClasses[cIndex][attIndex] !=
                            null && floatLabelDistributionsInClasses[cIndex][
                            attIndex].length > 0) {
                        posteriorClassProbs[cIndex] *=
                                floatLabelDistributionsInClasses[cIndex][
                                attIndex][instance.floatIndexes[attIndex]];
                    }
                }
            }
        }
        if (instance.integerIndexes != null) {
            for (int attIndex = 0; attIndex < instance.integerIndexes.length;
                    attIndex++) {
                for (int cIndex = 0; cIndex < dataClasses.length; cIndex++) {
                    if (intLabelDistributionsInClasses[cIndex][attIndex] !=
                            null && intLabelDistributionsInClasses[cIndex][
                            attIndex].length > 0) {
                        posteriorClassProbs[cIndex] *=
                                intLabelDistributionsInClasses[cIndex][
                                attIndex][instance.integerIndexes[attIndex]];
                    }
                }
            }
        }
        if (instance.nominalIndexes != null) {
            for (int attIndex = 0; attIndex < instance.nominalIndexes.length;
                    attIndex++) {
                for (int cIndex = 0; cIndex < dataClasses.length; cIndex++) {
                    if (nominalLabelDistributionsInClasses[cIndex][attIndex] !=
                            null && nominalLabelDistributionsInClasses[cIndex][
                            attIndex].length > 0) {
                        posteriorClassProbs[cIndex] *=
                                nominalLabelDistributionsInClasses[cIndex][
                                attIndex][instance.nominalIndexes[attIndex]];
                    }
                }
            }
        }
        // Normalize.
        float probTotal = 0;
        for (int cIndex = 0; cIndex < dataClasses.length; cIndex++) {
            probTotal += posteriorClassProbs[cIndex];
        }
        for (int cIndex = 0; cIndex < dataClasses.length; cIndex++) {
            posteriorClassProbs[cIndex] /= probTotal;
        }
        float[] posteriorClassProbsFloat = new float[dataClasses.length];
        for (int cIndex = 0; cIndex < dataClasses.length; cIndex++) {
            posteriorClassProbsFloat[cIndex] =
                    (float)posteriorClassProbs[cIndex];
        }
        return posteriorClassProbsFloat;
    }
}
