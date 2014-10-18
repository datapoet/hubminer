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
import learning.supervised.DiscreteCategory;
import learning.supervised.DiscreteClassifier;
import learning.supervised.evaluation.ValidateableInterface;

/**
 * This class implements the standard Naive Bayes classifier.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class DNaiveBayes extends DiscreteClassifier implements Serializable {

    private static final long serialVersionUID = 1L;
    // The conditional probability distribution arrays.
    private float[][][] floatLabelDistributionsInClasses = null;
    private float[][][] intLabelDistributionsInClasses = null;
    private float[][][] nominalLabelDistributionsInClasses = null;
    private float[] classPriors = null;
    // Laplace estimator for distribution smoothing.
    private float laplaceEstimator = 1f;

    @Override
    public String getName() {
        return "Naive Bayes";
    }

    /**
     * The default constructor.
     */
    public DNaiveBayes() {
    }

    /**
     * Initialization.
     *
     * @param discDSet DiscretizedDataSet that is the training data.
     */
    public DNaiveBayes(DiscretizedDataSet discDSet) {
        setDataType(discDSet);
        generateClassesFromDataType();
    }

    /**
     * Initialization.
     *
     * @param discDSet DiscretizedDataSet that is the training data context.
     * @param dataClasses DiscreteCategory[] that is the training data.
     */
    public DNaiveBayes(DiscretizedDataSet discDSet,
            DiscreteCategory[] dataClasses) {
        setDataType(discDSet);
        setClasses(dataClasses);
    }

    /**
     * Initialization.
     *
     * @param discDSet DiscretizedDataSet that is the training data context.
     * @param dataClasses DiscreteCategory[] that is the training data.
     * @param laplaceEstimator Float value that is the Laplace estimator for
     * probability distribution smoothing.
     */
    public DNaiveBayes(DiscretizedDataSet discDSet,
            DiscreteCategory[] dataClasses, float laplaceEstimator) {
        setDataType(discDSet);
        setClasses(dataClasses);
        this.laplaceEstimator = laplaceEstimator;
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
        DNaiveBayes nbCopy = new DNaiveBayes();
        nbCopy.setLaplaceEstimator(laplaceEstimator);
        nbCopy.setDataType(getDataType());
        return nbCopy;
    }

    
    /**
     * @param classPriors float[] representing the prior class distribution.
     */
    public void setPriorsExternally(float[] classPriors) {
        this.classPriors = classPriors;
        for (int c = 0; c < classPriors.length; c++) {
            this.classPriors[c] += laplaceEstimator;
            this.classPriors[c] /= (1 + classPriors.length * laplaceEstimator);
        }
    }

    
    @Override
    public void train() throws Exception {
        DiscretizedDataInstance instance;
        // Fetch the data.
        DiscretizedDataSet discDSet = getDataType();
        DiscreteCategory[] dataClasses = getClasses();
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
        // End of initialization. Now count the frequencies in order to estimate
        // the class-conditional probabilities.
        for (int cIndex = 0; cIndex < dataClasses.length; cIndex++) {
            if (dataClasses[cIndex] != null &&
                    dataClasses[cIndex].indexes.size() > 0) {
                for (int i = 0; i < dataClasses[cIndex].indexes.size(); i++) {
                    instance = discDSet.data.get(
                            dataClasses[cIndex].indexes.get(i));
                    for (int attIndex = 0; attIndex < floatSize; attIndex++) {
                        floatLabelDistributionsInClasses[cIndex][attIndex][
                                instance.floatIndexes[attIndex]]++;
                    }
                    for (int attIndex = 0; attIndex < intSize; attIndex++) {
                        intLabelDistributionsInClasses[cIndex][attIndex][
                                instance.integerIndexes[attIndex]]++;
                    }
                    for (int attIndex = 0; attIndex < nominalSize; attIndex++) {
                        nominalLabelDistributionsInClasses[cIndex][attIndex][
                                instance.nominalIndexes[attIndex]]++;
                    }
                }
            }
        }
        // Normalization and smoothing.
        for (int cIndex = 0; cIndex < dataClasses.length; cIndex++) {
            if (dataClasses[cIndex] != null &&
                    dataClasses[cIndex].indexes.size() > 0) {
                for (int attIndex = 0; attIndex < floatSize; attIndex++) {
                    if (floatLabelDistributionsInClasses[cIndex][
                            attIndex] != null) {
                        for (int j = 0; j < floatLabelDistributionsInClasses[
                                cIndex][attIndex].length; j++) {
                            floatLabelDistributionsInClasses[cIndex][
                                    attIndex][j] += laplaceEstimator;
                            floatLabelDistributionsInClasses[cIndex][
                                    attIndex][j] /= ((float)
                                    floatLabelDistributionsInClasses[cIndex][
                                    attIndex].length * laplaceEstimator +
                                    (float) dataClasses[cIndex].indexes.size());
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
                                    (float) dataClasses[cIndex].indexes.size());
                        }
                    }
                }
                for (int attIndex = 0; attIndex < nominalSize; attIndex++) {
                    if (nominalLabelDistributionsInClasses[cIndex][
                            attIndex] != null) {
                        for (int j = 0; j < nominalLabelDistributionsInClasses[
                                cIndex][attIndex].length; j++) {
                            nominalLabelDistributionsInClasses[cIndex][
                                    attIndex][j] += laplaceEstimator;
                            nominalLabelDistributionsInClasses[cIndex][
                                    attIndex][j] /= ((float)
                                    nominalLabelDistributionsInClasses[cIndex][
                                    attIndex].length * laplaceEstimator +
                                    (float) dataClasses[cIndex].indexes.size());
                        }
                    }
                }
            }
        }
        // Calculate the smoothed class priors.
        classPriors = new float[dataClasses.length];
        for (int cIndex = 0; cIndex < dataClasses.length; cIndex++) {
            if (dataClasses[cIndex] != null &&
                    dataClasses[cIndex].indexes.size() > 0) {
                classPriors[cIndex] = ((float)
                        dataClasses[cIndex].indexes.size() + laplaceEstimator) /
                        (discDSet.data.size() + (float) dataClasses.length *
                        laplaceEstimator);
            } else {
                classPriors[cIndex] = (laplaceEstimator / ((float)
                        discDSet.data.size() + (float) dataClasses.length *
                        laplaceEstimator));
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
        // Handle the float feature values.
        if (instance.floatIndexes != null) {
            for (int i = 0; i < instance.floatIndexes.length; i++) {
                int smallCount = 0;
                for (int cIndex = 0; cIndex < dataClasses.length; cIndex++) {
                    if (posteriorClassProbs[cIndex] < 0.00001) {
                        smallCount++;
                    }
                }
                if (smallCount == dataClasses.length) {
                    for (int cIndex = 0; cIndex < dataClasses.length;
                            cIndex++) {
                        posteriorClassProbs[cIndex] *= 100000000;
                    }
                }
                for (int cIndex = 0; cIndex < dataClasses.length; cIndex++) {
                    if (floatLabelDistributionsInClasses[cIndex][i] != null &&
                            floatLabelDistributionsInClasses[cIndex][
                            i].length > 0) {
                        posteriorClassProbs[cIndex] *=
                                floatLabelDistributionsInClasses[cIndex][i][
                                instance.floatIndexes[i]];
                    }
                }
            }
        }
        // Handle the integer feature values.
        if (instance.integerIndexes != null) {
            for (int i = 0; i < instance.integerIndexes.length; i++) {
                for (int j = 0; j < dataClasses.length; j++) {
                    if (intLabelDistributionsInClasses[j][i] != null &&
                            intLabelDistributionsInClasses[j][i].length > 0) {
                        posteriorClassProbs[j] *=
                                intLabelDistributionsInClasses[j][i][
                                instance.integerIndexes[i]];
                    }
                }
            }
        }
        // Handle the nominal feature values.
        if (instance.nominalIndexes != null) {
            for (int i = 0; i < instance.nominalIndexes.length; i++) {
                for (int j = 0; j < dataClasses.length; j++) {
                    if (nominalLabelDistributionsInClasses[j][i] != null &&
                            nominalLabelDistributionsInClasses[j][
                            i].length > 0) {
                        posteriorClassProbs[j] *=
                                nominalLabelDistributionsInClasses[j][i][
                                instance.nominalIndexes[i]];
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
    public float[] classifyProbabilistically(
            DiscretizedDataInstance instance) throws Exception {
        DiscreteCategory[] dataClasses = getClasses();
        double[] posteriorClassProbs = new double[dataClasses.length];
        for (int cIndex = 0; cIndex < dataClasses.length; cIndex++) {
            posteriorClassProbs[cIndex] = classPriors[cIndex];
        }
        // Handle the float feature values.
        if (instance.floatIndexes != null) {
            for (int i = 0; i < instance.floatIndexes.length; i++) {
                int smallCount = 0;
                for (int cIndex = 0; cIndex < dataClasses.length; cIndex++) {
                    if (posteriorClassProbs[cIndex] < 0.00001) {
                        smallCount++;
                    }
                }
                if (smallCount == dataClasses.length) {
                    for (int cIndex = 0; cIndex < dataClasses.length;
                            cIndex++) {
                        posteriorClassProbs[cIndex] *= 100000000;
                    }
                }
                for (int cIndex = 0; cIndex < dataClasses.length; cIndex++) {
                    if (floatLabelDistributionsInClasses[cIndex][i] != null &&
                            floatLabelDistributionsInClasses[cIndex][
                            i].length > 0) {
                        posteriorClassProbs[cIndex] *=
                                floatLabelDistributionsInClasses[cIndex][i][
                                instance.floatIndexes[i]];
                    }
                }
            }
        }
        // Handle the integer feature values.
        if (instance.integerIndexes != null) {
            for (int i = 0; i < instance.integerIndexes.length; i++) {
                for (int cIndex = 0; cIndex < dataClasses.length; cIndex++) {
                    if (intLabelDistributionsInClasses[cIndex][i] != null &&
                            intLabelDistributionsInClasses[cIndex][
                            i].length > 0) {
                        posteriorClassProbs[cIndex] *=
                                intLabelDistributionsInClasses[cIndex][i][
                                instance.integerIndexes[i]];
                    }
                }
            }
        }
        // Handle the nominal feature values.
        if (instance.nominalIndexes != null) {
            for (int i = 0; i < instance.nominalIndexes.length; i++) {
                for (int cIndex = 0; cIndex < dataClasses.length; cIndex++) {
                    if (nominalLabelDistributionsInClasses[cIndex][i] != null &&
                            nominalLabelDistributionsInClasses[cIndex][
                            i].length > 0) {
                        posteriorClassProbs[cIndex] *=
                                nominalLabelDistributionsInClasses[cIndex][i][
                                instance.nominalIndexes[i]];
                    }
                }
            }
        }
        // Normalize.
        float probTotal = 0;
        for (int i = 0; i < dataClasses.length; i++) {
            probTotal += posteriorClassProbs[i];
        }
        for (int i = 0; i < dataClasses.length; i++) {
            posteriorClassProbs[i] /= probTotal;
        }
        float[] posteriorClassProbsFloat = new float[dataClasses.length];
        for (int i = 0; i < dataClasses.length; i++) {
            posteriorClassProbsFloat[i] = (float)posteriorClassProbs[i];
        }
        return posteriorClassProbsFloat;
    }
}
