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
package feature.evaluation;

import data.representation.discrete.DiscretizedDataSet;
import data.representation.util.DataMineConstants;
import java.util.ArrayList;

/**
 * This class implements the basic functionality for discrete attribute
 * evaluation and is extended by concrete approach implementations.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public abstract class DiscreteAttributeEvaluator {

    private DiscretizedDataSet discDSet = null;
    private float[] intEvaluations;
    private float[] floatEvaluations;
    private float[] nominalEvaluations;

    /**
     * @param discDSet Discretized dataset that is being analyzed.
     */
    public void setDiscretizedDataSet(DiscretizedDataSet discDSet) {
        this.discDSet = discDSet;
    }

    /**
     * @return Discretized dataset that is being analyzed.
     */
    public DiscretizedDataSet getDiscretizedDataSet() {
        return discDSet;
    }

    /**
     * Evaluate the specified feature.
     *
     * @param attType Integer that is the feature type, as in DataMineConstants.
     * @param attIndex Index of the feature in its feature group.
     * @return Float value that is the evaluation.
     */
    public abstract float evaluate(int attType, int attIndex);

    /**
     * Evaluate the specified feature.
     *
     * @param subset Indexes of the instance subset to evaluate the feature on.
     * @param attType Integer that is the feature type, as in DataMineConstants.
     * @param attIndex Index of the feature in its feature group.
     * @return Float value that is the evaluation.
     */
    public abstract float evaluateOnSubset(ArrayList<Integer> subset,
            int attType, int attIndex);

    public abstract float evaluateSplit(ArrayList<Integer>[] split);

    public abstract float evaluateSplitOnSubset(ArrayList<Integer> subset,
            ArrayList<Integer>[] split);

    /**
     * Get the type and index of the feature with the lowest evaluation.
     *
     * @return Integer array of two elements, the first one being the type and
     * the second one the index of the feature in its feature group.
     */
    public int[] getTypeAndIndexOfLowestEvaluatedFeature() {
        float lowInt = Float.MAX_VALUE;
        float lowFloat = Float.MAX_VALUE;
        float lowNominal = Float.MAX_VALUE;
        int intIndex = getIndexOfLowestEvaluatedFeature(
                DataMineConstants.INTEGER);
        int floatIndex = getIndexOfLowestEvaluatedFeature(
                DataMineConstants.FLOAT);
        int nominalIndex = getIndexOfLowestEvaluatedFeature(
                DataMineConstants.NOMINAL);
        if (intIndex == -1 && floatIndex == -1 && nominalIndex == -1) {
            return null;
        } else {
            int[] result = new int[2];
            if (intIndex != -1) {
                lowInt = intEvaluations[intIndex];
            }
            if (floatIndex != -1) {
                lowFloat = floatEvaluations[floatIndex];
            }
            if (nominalIndex != -1) {
                lowNominal = nominalEvaluations[nominalIndex];
            }
            // Get the minimum feature evaluation.
            float min = Math.min(lowInt, Math.min(lowFloat, lowNominal));
            // Determine the type.
            if (lowInt == min) {
                result[0] = DataMineConstants.INTEGER;
                result[1] = intIndex;
            } else if (lowFloat == min) {
                result[0] = DataMineConstants.FLOAT;
                result[1] = floatIndex;
            } else if (lowNominal == min) {
                result[0] = DataMineConstants.NOMINAL;
                result[1] = nominalIndex;
            }
            return result;
        }
    }

    /**
     * Get the type and index of the feature with the lowest evaluation. This
     * can yield best or worst features, depending on the type of estimator
     * score.
     *
     * @param subset ArrayList of integer indexes defining the subset to
     * analyze.
     * @return Integer array of two elements, the first one being the type and
     * the second one the index of the feature in its feature group.
     */
    public int[] getTypeAndIndexOfLowestEvaluatedFeatureOnSubset(
            ArrayList<Integer> subset) {
        float lowInt = Float.MAX_VALUE;
        float lowFloat = Float.MAX_VALUE;
        float lowNominal = Float.MAX_VALUE;
        int intIndex = getIndexOfLowestEvaluatedFeatureOnSubset(subset,
                DataMineConstants.INTEGER);
        int floatIndex = getIndexOfLowestEvaluatedFeatureOnSubset(subset,
                DataMineConstants.FLOAT);
        int nominalIndex = getIndexOfLowestEvaluatedFeatureOnSubset(subset,
                DataMineConstants.NOMINAL);
        if (intIndex == -1 && floatIndex == -1 && nominalIndex == -1) {
            return null;
        } else {
            int[] result = new int[2];
            if (intIndex != -1) {
                lowInt = intEvaluations[intIndex];
            }
            if (floatIndex != -1) {
                lowFloat = floatEvaluations[floatIndex];
            }
            if (nominalIndex != -1) {
                lowNominal = nominalEvaluations[nominalIndex];
            }
            // Get the minimum feature evaluation.
            float min = Math.min(lowInt, Math.min(lowFloat, lowNominal));
            // Determine the type.
            if (lowInt == min) {
                result[0] = DataMineConstants.INTEGER;
                result[1] = intIndex;
            } else if (lowFloat == min) {
                result[0] = DataMineConstants.FLOAT;
                result[1] = floatIndex;
            } else if (lowNominal == min) {
                result[0] = DataMineConstants.NOMINAL;
                result[1] = nominalIndex;
            }
            return result;
        }
    }

    /**
     * Get the type and index of the feature with the lowest evaluation. This
     * can yield best or worst features, depending on the type of estimator
     * score.
     *
     * @param subset ArrayList of integer indexes defining the subset to
     * analyze.
     * @param acceptableFloat Boolean array of acceptable float features.
     * @param acceptableInt Boolean array of acceptable integer features.
     * @param acceptableNominal Boolean array of acceptable nominal features.
     * @return Integer array of two elements, the first one being the type and
     * the second one the index of the feature in its feature group.
     */
    public int[] getTypeAndIndexOfLowestEvaluatedFeatureOnSubset(
            ArrayList<Integer> subset, boolean[] acceptableFloat,
            boolean[] acceptableInt, boolean[] acceptableNominal) {
        if (subset == null) {
            return null;
        }
        float lowInt = Float.MAX_VALUE;
        float lowFloat = Float.MAX_VALUE;
        float lowNominal = Float.MAX_VALUE;
        int intIndex = getIndexOfLowestEvaluatedFeatureOnSubset(subset,
                DataMineConstants.INTEGER, acceptableInt);
        int floatIndex = getIndexOfLowestEvaluatedFeatureOnSubset(subset,
                DataMineConstants.FLOAT, acceptableFloat);
        int nominalIndex = getIndexOfLowestEvaluatedFeatureOnSubset(subset,
                DataMineConstants.NOMINAL, acceptableNominal);
        if (intIndex == -1 && floatIndex == -1 && nominalIndex == -1) {
            return null;
        } else {
            int[] result = new int[2];
            if (intIndex != -1) {
                lowInt = intEvaluations[intIndex];
            }
            if (floatIndex != -1) {
                lowFloat = floatEvaluations[floatIndex];
            }
            if (nominalIndex != -1) {
                lowNominal = nominalEvaluations[nominalIndex];
            }
            // Get the minimum feature evaluation.
            float min = Math.min(lowInt, Math.min(lowFloat, lowNominal));
            // Determine the type.
            if (lowInt == min) {
                result[0] = DataMineConstants.INTEGER;
                result[1] = intIndex;
            } else if (lowFloat == min) {
                result[0] = DataMineConstants.FLOAT;
                result[1] = floatIndex;
            } else if (lowNominal == min) {
                result[0] = DataMineConstants.NOMINAL;
                result[1] = nominalIndex;
            }
            return result;
        }
    }

    /**
     * Get the index of the lowest evaluated feature of the specified type. This
     * can yield best or worst features, depending on the type of estimator
     * score.
     *
     * @param attType Feature type, as in DataMineConstants.
     * @return Index of the lowest evaluated feature in its feature group.
     */
    public int getIndexOfLowestEvaluatedFeature(int attType) {
        if (attType == DataMineConstants.INTEGER) {
            if (intEvaluations == null) {
                intEvaluations = evaluateAll(attType);
            }
            if (intEvaluations != null && intEvaluations.length > 0) {
                float minIndexValue = Float.MAX_VALUE;
                int minIndex = 0;
                for (int i = 0; i < intEvaluations.length; i++) {
                    if (intEvaluations[i] < minIndexValue) {
                        minIndex = i;
                        minIndexValue = intEvaluations[i];
                    }
                }
                return minIndex;
            } else {
                return -1;
            }
        } else if (attType == DataMineConstants.FLOAT) {
            if (floatEvaluations == null) {
                floatEvaluations = evaluateAll(attType);
            }
            if (floatEvaluations != null && floatEvaluations.length > 0) {
                float minIndexValue = Float.MAX_VALUE;
                int minIndex = 0;
                for (int i = 0; i < floatEvaluations.length; i++) {
                    if (floatEvaluations[i] < minIndexValue) {
                        minIndex = i;
                        minIndexValue = floatEvaluations[i];
                    }
                }
                return minIndex;
            } else {
                return -1;
            }
        } else {
            if (nominalEvaluations == null) {
                nominalEvaluations = evaluateAll(attType);
            }
            if (nominalEvaluations != null && nominalEvaluations.length > 0) {
                float minIndexValue = Float.MAX_VALUE;
                int minIndex = 0;
                for (int i = 0; i < nominalEvaluations.length; i++) {
                    if (nominalEvaluations[i] < minIndexValue) {
                        minIndex = i;
                        minIndexValue = nominalEvaluations[i];
                    }
                }
                return minIndex;
            } else {
                return -1;
            }
        }
    }

    /**
     * Get the index of the lowest evaluated feature of the specified type. This
     * can yield best or worst features, depending on the type of estimator
     * score.
     *
     * @param subset ArrayList of integer indexes defining the subset to
     * analyze.
     * @param attType Feature type, as in DataMineConstants.
     * @return Index of the lowest evaluated feature in its feature group.
     */
    public int getIndexOfLowestEvaluatedFeatureOnSubset(
            ArrayList<Integer> subset, int attType) {
        if (subset == null) {
            return -1;
        }
        if (attType == DataMineConstants.INTEGER) {
            if (intEvaluations == null) {
                intEvaluations = evaluateAll(subset, attType);
            }
            if (intEvaluations != null && intEvaluations.length > 0) {
                float minIndexValue = Float.MAX_VALUE;
                int minIndex = 0;
                for (int i = 0; i < intEvaluations.length; i++) {
                    if (intEvaluations[i] < minIndexValue) {
                        minIndex = i;
                        minIndexValue = intEvaluations[i];
                    }
                }
                return minIndex;
            } else {
                return -1;
            }
        } else if (attType == DataMineConstants.FLOAT) {
            if (floatEvaluations == null) {
                floatEvaluations = evaluateAll(subset, attType);
            }
            if (floatEvaluations != null && floatEvaluations.length > 0) {
                float minIndexValue = Float.MAX_VALUE;
                int minIndex = 0;
                for (int i = 0; i < floatEvaluations.length; i++) {
                    if (floatEvaluations[i] < minIndexValue) {
                        minIndex = i;
                        minIndexValue = floatEvaluations[i];
                    }
                }
                return minIndex;
            } else {
                if (floatEvaluations == null) {
                    System.err.println("fEval null");
                } else {
                    System.err.println("fEval zero length");
                }
                return -1;
            }
        } else {
            if (nominalEvaluations == null) {
                nominalEvaluations = evaluateAll(subset, attType);
            }
            if (nominalEvaluations != null && nominalEvaluations.length > 0) {
                float minIndexValue = Float.MAX_VALUE;
                int minIndex = 0;
                for (int i = 0; i < nominalEvaluations.length; i++) {
                    if (nominalEvaluations[i] < minIndexValue) {
                        minIndex = i;
                        minIndexValue = nominalEvaluations[i];
                    }
                }
                return minIndex;
            } else {
                return -1;
            }
        }
    }

    /**
     * Get the index of the lowest evaluated feature of the specified type. This
     * can yield best or worst features, depending on the type of estimator
     * score.
     *
     * @param subset ArrayList of integer indexes defining the subset to
     * analyze.
     * @param attType Feature type, as in DataMineConstants.
     * @param acceptable Boolean array indicating which features to consider.
     * @return Index of the lowest evaluated feature in its feature group.
     */
    public int getIndexOfLowestEvaluatedFeatureOnSubset(
            ArrayList<Integer> subset, int attType, boolean[] acceptable) {
        if (attType == DataMineConstants.INTEGER) {
            if (intEvaluations == null) {
                intEvaluations = evaluateAll(subset, attType, acceptable);
            }
            if (intEvaluations != null && intEvaluations.length > 0) {
                float minIndexValue = Float.MAX_VALUE;
                int minIndex = 0;
                for (int i = 0; i < intEvaluations.length; i++) {
                    if (acceptable[i]) {
                        if (intEvaluations[i] < minIndexValue) {
                            minIndex = i;
                            minIndexValue = intEvaluations[i];
                        }
                    }
                }
                return minIndex;
            } else {
                return -1;
            }
        } else if (attType == DataMineConstants.FLOAT) {
            if (floatEvaluations == null) {
                floatEvaluations = evaluateAll(subset, attType, acceptable);
            }
            if (floatEvaluations != null && floatEvaluations.length > 0) {
                float minIndexValue = Float.MAX_VALUE;
                int minIndex = 0;
                for (int i = 0; i < floatEvaluations.length; i++) {
                    if (acceptable[i]) {
                        if (floatEvaluations[i] < minIndexValue) {
                            minIndex = i;
                            minIndexValue = floatEvaluations[i];
                        }
                    }
                }
                return minIndex;
            } else {
                if (floatEvaluations == null) {
                    System.err.println("fEval null");
                } else {
                    System.err.println("fEval zero length");
                }
                return -1;
            }
        } else {
            if (nominalEvaluations == null) {
                nominalEvaluations = evaluateAll(subset, attType, acceptable);
            }
            if (nominalEvaluations != null && nominalEvaluations.length > 0) {
                float minIndexValue = Float.MAX_VALUE;
                int minIndex = 0;
                for (int i = 0; i < nominalEvaluations.length; i++) {
                    if (acceptable[i]) {
                        if (nominalEvaluations[i] < minIndexValue) {
                            minIndex = i;
                            minIndexValue = nominalEvaluations[i];
                        }
                    }
                }
                return minIndex;
            } else {
                return -1;
            }
        }
    }

    /**
     * Gets the type and index of the highest evaluated feature. This can yield
     * best or worst features, depending on the type of estimator score.
     *
     * @return Integer array of two elements, the first one being the type and
     * the second one the index of the feature in its feature group.
     */
    public int[] getTypeAndIndexOfHighestEvaluatedFeature() {
        float highInt = Float.MIN_VALUE;
        float highFloat = Float.MIN_VALUE;
        float highNominal = Float.MIN_VALUE;
        int intIndex = getIndexOfHighestEvaluatedFeature(
                DataMineConstants.INTEGER);
        int floatIndex = getIndexOfHighestEvaluatedFeature(
                DataMineConstants.FLOAT);
        int nominalIndex = getIndexOfHighestEvaluatedFeature(
                DataMineConstants.NOMINAL);
        if (intIndex == -1 && floatIndex == -1 && nominalIndex == -1) {
            return null;
        } else {
            int[] result = new int[2];
            if (intIndex != -1) {
                highInt = intEvaluations[intIndex];
            }
            if (floatIndex != -1) {
                highFloat = floatEvaluations[floatIndex];
            }
            if (nominalIndex != -1) {
                highNominal = nominalEvaluations[nominalIndex];
            }
            // Get the maximum feature evaluation.
            float max = Math.max(highInt, Math.max(highFloat, highNominal));
            // Determine the type.
            if (highInt == max) {
                result[0] = DataMineConstants.INTEGER;
                result[1] = intIndex;
            } else if (highFloat == max) {
                result[0] = DataMineConstants.FLOAT;
                result[1] = floatIndex;
            } else if (highNominal == max) {
                result[0] = DataMineConstants.NOMINAL;
                result[1] = nominalIndex;
            }
            return result;
        }
    }

    /**
     * Gets the type and index of the highest evaluated feature. This can yield
     * best or worst features, depending on the type of estimator score.
     *
     * @param subset ArrayList of integer indexes defining the subset to
     * analyze.
     * @return Integer array of two elements, the first one being the type and
     * the second one the index of the feature in its feature group.
     */
    public int[] getTypeAndIndexOfHighestEvaluatedFeatureOnSubset(
            ArrayList<Integer> subset) {
        if (subset == null) {
            return null;
        }
        float highInt = Float.MIN_VALUE;
        float highFloat = Float.MIN_VALUE;
        float highNominal = Float.MIN_VALUE;
        int intIndex = getIndexOfHighestEvaluatedFeatureOnSubset(subset,
                DataMineConstants.INTEGER);
        int floatIndex = getIndexOfHighestEvaluatedFeatureOnSubset(subset,
                DataMineConstants.FLOAT);
        int nominalIndex = getIndexOfHighestEvaluatedFeatureOnSubset(subset,
                DataMineConstants.NOMINAL);
        if (intIndex == -1 && floatIndex == -1 && nominalIndex == -1) {
            return null;
        } else {
            int[] result = new int[2];
            if (intIndex != -1) {
                highInt = intEvaluations[intIndex];
            }
            if (floatIndex != -1) {
                highFloat = floatEvaluations[floatIndex];
            }
            if (nominalIndex != -1) {
                highNominal = nominalEvaluations[nominalIndex];
            }
            float max = Math.max(highInt, Math.max(highFloat, highNominal));
            if (highInt == max) {
                result[0] = DataMineConstants.INTEGER;
                result[1] = intIndex;
            } else if (highFloat == max) {
                result[0] = DataMineConstants.FLOAT;
                result[1] = floatIndex;
            } else if (highNominal == max) {
                result[0] = DataMineConstants.NOMINAL;
                result[1] = nominalIndex;
            }
            return result;
        }
    }

    /**
     * Gets the highest evaluated feature index of the specified type. This can
     * yield best or worst features, depending on the type of estimator score.
     *
     * @param attType Feature type, as in DataMineConstants.
     * @return Index of the highest evaluated feature in its feature group.
     */
    public int getIndexOfHighestEvaluatedFeature(int attType) {
        if (attType == DataMineConstants.INTEGER) {
            if (intEvaluations == null) {
                intEvaluations = evaluateAll(attType);
            }
            if (intEvaluations != null && intEvaluations.length > 0) {
                float maxIndexValue = Float.MIN_VALUE;
                int maxIndex = 0;
                for (int i = 0; i < intEvaluations.length; i++) {
                    if (intEvaluations[i] > maxIndexValue) {
                        maxIndex = i;
                        maxIndexValue = intEvaluations[i];
                    }
                }
                return maxIndex;
            } else {
                return -1;
            }
        } else if (attType == DataMineConstants.FLOAT) {
            if (floatEvaluations == null) {
                floatEvaluations = evaluateAll(attType);
            }
            if (floatEvaluations != null && floatEvaluations.length > 0) {
                float maxIndexValue = Float.MIN_VALUE;
                int maxIndex = 0;
                for (int i = 0; i < floatEvaluations.length; i++) {
                    if (floatEvaluations[i] > maxIndexValue) {
                        maxIndex = i;
                        maxIndexValue = floatEvaluations[i];
                    }
                }
                return maxIndex;
            } else {
                return -1;
            }
        } else {
            if (nominalEvaluations == null) {
                nominalEvaluations = evaluateAll(attType);
            }
            if (nominalEvaluations != null && nominalEvaluations.length > 0) {
                float maxIndexValue = Float.MIN_VALUE;
                int maxIndex = 0;
                for (int i = 0; i < nominalEvaluations.length; i++) {
                    if (nominalEvaluations[i] > maxIndexValue) {
                        maxIndex = i;
                        maxIndexValue = nominalEvaluations[i];
                    }
                }
                return maxIndex;
            } else {
                return -1;
            }
        }
    }

    /**
     * Gets the highest evaluated feature index of the specified type. This can
     * yield best or worst features, depending on the type of estimator score.
     *
     * @param subset ArrayList of integer indexes defining the subset to
     * analyze.
     * @param attType Feature type, as in DataMineConstants.
     * @return Index of the highest evaluated feature in its feature group.
     */
    public int getIndexOfHighestEvaluatedFeatureOnSubset(
            ArrayList<Integer> subset, int attType) {
        if (subset == null) {
            return -1;
        }
        if (attType == DataMineConstants.INTEGER) {
            if (intEvaluations == null) {
                intEvaluations = evaluateAll(subset, attType);
            }
            if (intEvaluations != null && intEvaluations.length > 0) {
                float maxIndexValue = Float.MIN_VALUE;
                int maxIndex = 0;
                for (int i = 0; i < intEvaluations.length; i++) {
                    if (intEvaluations[i] > maxIndexValue) {
                        maxIndex = i;
                        maxIndexValue = intEvaluations[i];
                    }
                }
                return maxIndex;
            } else {
                return -1;
            }
        } else if (attType == DataMineConstants.FLOAT) {
            if (floatEvaluations == null) {
                floatEvaluations = evaluateAll(subset, attType);
            }
            if (floatEvaluations != null && floatEvaluations.length > 0) {
                float maxIndexValue = Float.MIN_VALUE;
                int maxIndex = 0;
                for (int i = 0; i < floatEvaluations.length; i++) {
                    if (floatEvaluations[i] > maxIndexValue) {
                        maxIndex = i;
                        maxIndexValue = floatEvaluations[i];
                    }
                }
                return maxIndex;
            } else {
                return -1;
            }
        } else {
            if (nominalEvaluations == null) {
                nominalEvaluations = evaluateAll(subset, attType);
            }
            if (nominalEvaluations != null && nominalEvaluations.length > 0) {
                float maxIndexValue = Float.MIN_VALUE;
                int maxIndex = 0;
                for (int i = 0; i < nominalEvaluations.length; i++) {
                    if (nominalEvaluations[i] > maxIndexValue) {
                        maxIndex = i;
                        maxIndexValue = nominalEvaluations[i];
                    }
                }
                return maxIndex;
            } else {
                return -1;
            }
        }
    }

    /**
     * @return Float array of evaluations of integer features.
     */
    public float[] getIntEvaluations() {
        return intEvaluations;
    }

    /**
     * @return Float array of evaluations of float features.
     */
    public float[] getFloatEvaluations() {
        return floatEvaluations;
    }

    /**
     * @return Float array of evaluations of nominal features.
     */
    public float[] getNominalEvaluations() {
        return nominalEvaluations;
    }

    /**
     * Evaluate all features of all feature types.
     */
    public void evaluateAll() {
        intEvaluations = evaluateAll(DataMineConstants.INTEGER);
        floatEvaluations = evaluateAll(DataMineConstants.FLOAT);
        nominalEvaluations = evaluateAll(DataMineConstants.NOMINAL);
    }

    /**
     * Evaluate all features of the specified feature type.
     *
     * @param attType Feature type, as in DataMineConstants.
     * @return Float array of feature evaluations.
     */
    public float[] evaluateAll(int attType) {
        if (discDSet != null && discDSet.size() > 0) {
            if (attType == DataMineConstants.INTEGER) {
                if (discDSet.getIntIntervalDivisions() != null
                        && discDSet.getIntIntervalDivisions().length != 0) {
                    float[] evaluations =
                            new float[
                                    discDSet.getIntIntervalDivisions().length];
                    for (int i = 0; i
                            < discDSet.getIntIntervalDivisions().length; i++) {
                        evaluations[i] = evaluate(attType, i);
                    }
                    intEvaluations = evaluations;
                    return evaluations;
                } else {
                    return null;
                }
            } else if (attType == DataMineConstants.FLOAT) {
                if (discDSet.getFloatIntervalDivisions() != null
                        && discDSet.getFloatIntervalDivisions().length != 0) {
                    float[] evaluations =
                            new float[discDSet.
                            getFloatIntervalDivisions().length];
                    for (int i = 0; i
                            < discDSet.getFloatIntervalDivisions().length;
                            i++) {
                        evaluations[i] = evaluate(attType, i);
                    }
                    floatEvaluations = evaluations;
                    return evaluations;
                } else {
                    return null;
                }
            } else {
                if (discDSet.getNominalVocabularies() != null
                        && discDSet.getNominalVocabularies().length != 0) {
                    float[] evaluations =
                            new float[discDSet.getNominalVocabularies().length];
                    for (int i = 0; i
                            < discDSet.getNominalVocabularies().length; i++) {
                        evaluations[i] = evaluate(attType, i);
                    }
                    nominalEvaluations = evaluations;
                    return evaluations;
                } else {
                    return null;
                }
            }
        } else {
            return null;
        }
    }

    /**
     * Evaluate all features of the specified feature type.
     *
     * @param subset ArrayList of integer indexes defining the subset to
     * analyze.
     * @param attType Feature type, as in DataMineConstants.
     * @return Float array of feature evaluations.
     */
    public float[] evaluateAll(ArrayList<Integer> subset, int attType) {
        if (subset == null) {
            return null;
        }
        if (discDSet != null && discDSet.size() > 0) {
            if (attType == DataMineConstants.INTEGER) {
                if (discDSet.getIntIntervalDivisions() != null
                        && discDSet.getIntIntervalDivisions().length != 0) {
                    float[] evaluations = new float[discDSet.
                            getIntIntervalDivisions().length];
                    for (int i = 0; i
                            < discDSet.getIntIntervalDivisions().length; i++) {
                        evaluations[i] = evaluateOnSubset(subset, attType, i);
                    }
                    intEvaluations = evaluations;
                    return evaluations;
                } else {
                    return null;
                }
            } else if (attType == DataMineConstants.FLOAT) {
                if (discDSet.getFloatIntervalDivisions() != null
                        && discDSet.getFloatIntervalDivisions().length != 0) {
                    float[] evaluations = new float[discDSet.
                            getFloatIntervalDivisions().length];
                    for (int i = 0; i
                            < discDSet.getFloatIntervalDivisions().length;
                            i++) {
                        evaluations[i] = evaluateOnSubset(subset, attType, i);
                    }
                    floatEvaluations = evaluations;
                    return evaluations;
                } else {
                    return null;
                }
            } else {
                if (discDSet.getNominalVocabularies() != null
                        && discDSet.getNominalVocabularies().length != 0) {
                    float[] evaluations = new float[discDSet.
                            getNominalVocabularies().length];
                    for (int i = 0; i
                            < discDSet.getNominalVocabularies().length; i++) {
                        evaluations[i] = evaluateOnSubset(subset, attType, i);
                    }
                    nominalEvaluations = evaluations;
                    return evaluations;
                } else {
                    return null;
                }
            }
        } else {
            return null;
        }
    }

    /**
     * Evaluate all designated features of the specified feature type.
     *
     * @param subset ArrayList of integer indexes defining the subset to
     * analyze.
     * @param attType Feature type, as in DataMineConstants.
     * @param acceptable Boolean array indicating which features to consider.
     * @return array of feature evaluations. The default evaluation for the
     * non-selected features is zero.
     */
    public float[] evaluateAll(ArrayList<Integer> subset, int attType,
            boolean[] acceptable) {
        if (discDSet != null && discDSet.size() > 0) {
            if (attType == DataMineConstants.INTEGER) {
                if (discDSet.getIntIntervalDivisions() != null
                        && discDSet.getIntIntervalDivisions().length != 0) {
                    float[] evaluations = new float[discDSet.
                            getIntIntervalDivisions().length];
                    for (int i = 0; i
                            < discDSet.getIntIntervalDivisions().length; i++) {
                        if (acceptable[i]) {
                            evaluations[i] =
                                    evaluateOnSubset(subset, attType, i);
                        }
                    }
                    intEvaluations = evaluations;
                    return evaluations;
                } else {
                    return null;
                }
            } else if (attType == DataMineConstants.FLOAT) {
                if (discDSet.getFloatIntervalDivisions() != null
                        && discDSet.getFloatIntervalDivisions().length != 0) {
                    float[] evaluations = new float[discDSet.
                            getFloatIntervalDivisions().length];
                    for (int i = 0; i
                            < discDSet.getFloatIntervalDivisions().length;
                            i++) {
                        if (acceptable[i]) {
                            evaluations[i] =
                                    evaluateOnSubset(subset, attType, i);
                        }
                    }
                    floatEvaluations = evaluations;
                    return evaluations;
                } else {
                    return null;
                }
            } else {
                if (discDSet.getNominalVocabularies() != null
                        && discDSet.getNominalVocabularies().length != 0) {
                    float[] evaluations = new float[discDSet.
                            getNominalVocabularies().length];
                    for (int i = 0; i
                            < discDSet.getNominalVocabularies().length; i++) {
                        if (acceptable[i]) {
                            evaluations[i] =
                                    evaluateOnSubset(subset, attType, i);
                        }
                    }
                    nominalEvaluations = evaluations;
                    return evaluations;
                } else {
                    return null;
                }
            }
        } else {
            if (discDSet == null) {
                System.err.println("Null data encountered.");
            } else {
                System.err.println("Empty data encountered.");
            }
            return null;
        }
    }
}
