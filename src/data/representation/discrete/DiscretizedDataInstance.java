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
package data.representation.discrete;

import data.representation.DataInstance;
import java.io.Serializable;
import java.util.Arrays;

/**
 * This class implements a discrete version of the basic data holder. Since
 * discrete instances are often discretized versions of existing instances with
 * continuous float feature values, DiscretizedDataInstance does not inherit but
 * rather includes the original DataInstance object, if applicable. It was also
 * possible to implement inheritance instead. However, one rarely needs to
 * operate on both types of representations at the same time and different
 * algorithm implementations are needed for different types of instances.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class DiscretizedDataInstance implements Serializable {
    
    private static final long serialVersionUID = 1L;

    private DiscretizedDataSet dataContext = null;
    private DataInstance originalDataInstance;
    private int category = -1;
    private float[] fuzzyLabels = null;
    // Discrete arrays all have discrete index values pointing to certain
    // ranges of values in the original continous spectrum. The definition of
    // the ranges can be found in the embedding dataContext.
    public int[] integerIndexes = null;
    public int[] floatIndexes = null;
    public int[] nominalIndexes = null;

    public DiscretizedDataInstance() {
    }

    /**
     * @param dataContext DiscretizedDataSet embedding object.
     */
    public DiscretizedDataInstance(DiscretizedDataSet dataContext) {
        this(dataContext, true);
    }

    /**
     * @param dataContext DiscretizedDataSet embedding object.
     * @param initializeArrays Boolean flag indicating whether to initialize all
     * the arrays at once.
     */
    public DiscretizedDataInstance(DiscretizedDataSet dataContext,
            boolean initializeArrays) {
        this.dataContext = dataContext;
        if (initializeArrays) {
            if (dataContext.getIntIntervalDivisions() != null
                    && dataContext.getIntIntervalDivisions().length > 0) {
                integerIndexes =
                        new int[dataContext.getIntIntervalDivisions().length];
            }
            if (dataContext.getFloatIntervalDivisions() != null
                    && dataContext.getFloatIntervalDivisions().length > 0) {
                floatIndexes =
                        new int[dataContext.getFloatIntervalDivisions().length];
            }
            if (dataContext.getNominalVocabularies() != null
                    && dataContext.getNominalVocabularies().length > 0) {
                nominalIndexes =
                        new int[dataContext.getNominalVocabularies().length];
            }
        }
    }

    /**
     * @param category Integer representing the label.
     */
    public void setCategory(int category) {
        this.category = category;
    }

    /**
     * @return Integer representing the label.
     */
    public int getCategory() {
        return category;
    }

    /**
     * @return Fuzzy label array of floats.
     */
    public float[] getFuzzyLabels() {
        return fuzzyLabels;
    }

    /**
     * @param fuzzyLabels Fuzzy label array of floats.
     */
    public void setFuzzyLabels(float[] fuzzyLabels) {
        this.fuzzyLabels = fuzzyLabels;
    }

    /**
     * @param dataContext DiscretizedDataSet embedding object.
     */
    public void setDataContext(DiscretizedDataSet dataContext) {
        this.dataContext = dataContext;
    }

    /**
     * @return DiscretizedDataSet embedding object.
     */
    public DiscretizedDataSet getDataContext() {
        return dataContext;
    }

    /**
     * @param originalDataInstance DataInstance that was discretized to obtain
     * this DiscretizedDataInstance.
     */
    public void setOriginalInstance(DataInstance originalDataInstance) {
        this.originalDataInstance = originalDataInstance;
    }

    /**
     * @return DataInstance that was discretized to obtain this
     * DiscretizedDataInstance.
     */
    public DataInstance getOriginalInstance() {
        return originalDataInstance;
    }

    /**
     * Copies the current instance.
     *
     * @return DiscretizedDataInstance that is the copy of the current instance.
     */
    DiscretizedDataInstance copy() {
        DiscretizedDataInstance instanceCopy =
                new DiscretizedDataInstance(dataContext, true);
        instanceCopy.setOriginalInstance(originalDataInstance);
        if (integerIndexes != null) {
            instanceCopy.integerIndexes = Arrays.copyOf(integerIndexes,
                    integerIndexes.length);
        }
        if (floatIndexes != null) {
            instanceCopy.floatIndexes = Arrays.copyOf(floatIndexes,
                    floatIndexes.length);
        }
        if (nominalIndexes != null) {
            instanceCopy.nominalIndexes = Arrays.copyOf(nominalIndexes,
                    nominalIndexes.length);
        }
        instanceCopy.setCategory(category);
        instanceCopy.fuzzyLabels = fuzzyLabels;
        return instanceCopy;
    }

    @Override
    public boolean equals(Object instance) {
        if (instance instanceof DiscretizedDataInstance) {
            if ((integerIndexes != null)) {
                if (((DiscretizedDataInstance) instance).integerIndexes ==
                        null) {
                    return false;
                } else {
                    if (integerIndexes.length
                            != ((DiscretizedDataInstance) instance).
                            integerIndexes.length) {
                        return false;
                    } else {
                        for (int i = 0; i < integerIndexes.length; i++) {
                            if (integerIndexes[i]
                                    != ((DiscretizedDataInstance) instance).
                                    integerIndexes[i]) {
                                return false;
                            }
                        }
                    }
                }
            }
            if ((floatIndexes != null)) {
                if (((DiscretizedDataInstance) instance).floatIndexes == null) {
                    return false;
                } else {
                    if (floatIndexes.length
                            != ((DiscretizedDataInstance) instance).
                            floatIndexes.length) {
                        return false;
                    } else {
                        for (int i = 0; i < floatIndexes.length; i++) {
                            if (floatIndexes[i]
                                    != ((DiscretizedDataInstance) instance).
                                    floatIndexes[i]) {
                                return false;
                            }
                        }
                    }
                }
            }
            if ((nominalIndexes != null)) {
                if (((DiscretizedDataInstance) instance).nominalIndexes ==
                        null) {
                    return false;
                } else {
                    if (nominalIndexes.length
                            != ((DiscretizedDataInstance) instance).
                            nominalIndexes.length) {
                        return false;
                    } else {
                        for (int i = 0; i < integerIndexes.length; i++) {
                            if (nominalIndexes[i]
                                    != ((DiscretizedDataInstance) instance).
                                    nominalIndexes[i]) {
                                return false;
                            }
                        }
                    }
                }
            }
            // It is possible to extend equality to original instances as well.
            // Here it is left out.
            return true;
        } else {
            return false;
        }
    }
}
