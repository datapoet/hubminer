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
package optimization.stochastic.operators.onFloats;

import data.representation.DataInstance;
import data.representation.util.DataMineConstants;
import java.util.Random;
import optimization.stochastic.operators.HeterogenousMutationInterface;
import optimization.stochastic.operators.TwoDevsMutationInterface;

/**
 * Heterogenous float mutator where each feature has a separate probability of
 * mutating by a small or a large amount.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class HeterogenousTwoDevsFloatMutator
        implements HeterogenousMutationInterface, TwoDevsMutationInterface {

    private float pMutation = 1;
    private float[] stDevSmall;
    private float[] stDevBig;
    private float pSmall = 1;
    private float[] lowerBounds;
    private float[] upperBounds;
    private int beginIndex = -1;
    private int endIndex = -1;

    /**
     * @param stDevSmall An array of standard deviations of small mutations for
     * all the features.
     * @param stDevBig An array of standard deviations of large mutations for
     * all the features.
     * @param lowerBounds Lower value bounds.
     * @param upperBounds Upper value bounds.
     * @param pSmall Probability of a small mutation.
     */
    public HeterogenousTwoDevsFloatMutator(
            float[] stDevSmall,
            float[] stDevBig,
            float[] lowerBounds,
            float[] upperBounds,
            float pSmall) {
        this.stDevSmall = stDevSmall;
        this.stDevBig = stDevBig;
        this.lowerBounds = lowerBounds;
        this.upperBounds = upperBounds;
        this.pSmall = pSmall;
    }

    /**
     * @param stDevSmall An array of standard deviations of small mutations for
     * all the features.
     * @param stDevBig An array of standard deviations of large mutations for
     * all the features.
     * @param lowerBounds Lower value bounds.
     * @param upperBounds Upper value bounds.
     * @param beginIndex The index of the first feature to mutate.
     * @param endIndex The index of the last feature to mutate.
     * @param pSmall Probability of a small mutation.
     */
    public HeterogenousTwoDevsFloatMutator(
            float[] stDevSmall,
            float[] stDevBig,
            float[] lowerBounds,
            float[] upperBounds,
            int beginIndex,
            int endIndex,
            float pSmall) {
        this.stDevSmall = stDevSmall;
        this.stDevBig = stDevBig;
        this.lowerBounds = lowerBounds;
        this.upperBounds = upperBounds;
        this.beginIndex = beginIndex;
        this.endIndex = endIndex;
        this.pSmall = pSmall;
    }

    /**
     * @param stDevSmall An array of standard deviations of small mutations for
     * all the features.
     * @param stDevBig An array of standard deviations of large mutations for
     * all the features.
     * @param lowerBounds Lower value bounds.
     * @param upperBounds Upper value bounds.
     * @param pMutation Probability of mutating a feature.
     * @param pSmall Probability of a small mutation.
     */
    public HeterogenousTwoDevsFloatMutator(
            float[] stDevSmall,
            float[] stDevBig,
            float[] lowerBounds,
            float[] upperBounds,
            float pMutation,
            float pSmall) {
        this.stDevSmall = stDevSmall;
        this.stDevBig = stDevBig;
        this.lowerBounds = lowerBounds;
        this.upperBounds = upperBounds;
        this.pMutation = pMutation;
        this.pSmall = pSmall;
    }

    /**
     * @param stDevSmall An array of standard deviations of small mutations for
     * all the features.
     * @param stDevBig An array of standard deviations of large mutations for
     * all the features.
     * @param lowerBounds Lower value bounds.
     * @param upperBounds Upper value bounds.
     * @param beginIndex The index of the first feature to mutate.
     * @param endIndex The index of the last feature to mutate.
     * @param pMutation Probability of mutating a feature.
     * @param pSmall Probability of a small mutation.
     */
    public HeterogenousTwoDevsFloatMutator(
            float[] stDevSmall,
            float[] stDevBig,
            float[] lowerBounds,
            float[] upperBounds,
            int beginIndex,
            int endIndex,
            float pMutation,
            float pSmall) {
        this.stDevSmall = stDevSmall;
        this.stDevBig = stDevBig;
        this.lowerBounds = lowerBounds;
        this.upperBounds = upperBounds;
        this.beginIndex = beginIndex;
        this.endIndex = endIndex;
        this.pMutation = pMutation;
        this.pSmall = pSmall;
    }

    @Override
    public void setPSmall(float pSmall) {
        this.pSmall = pSmall;
    }

    @Override
    public float getPSmall() {
        return pSmall;
    }

    @Override
    public void setDevSmall(float stDevSmall) {
        for (int i = 0; i < this.stDevSmall.length; i++) {
            this.stDevSmall[i] = stDevSmall;
        }
    }

    @Override
    public float getDevSmall() {
        return stDevSmall[0];
    }

    @Override
    public void setDevBig(float stDevBig) {
        for (int i = 0; i < this.stDevBig.length; i++) {
            this.stDevBig[i] = stDevBig;
        }
    }

    @Override
    public float getDevBig() {
        return stDevBig[0];
    }

    @Override
    public void setStDevs(float[] stDev) {
        this.stDevSmall = stDev;
        this.stDevBig = stDev;
    }

    @Override
    public float[] getStDevs() {
        return stDevSmall;
    }

    @Override
    public void setDevsSmall(float[] stDevSmall) {
        this.stDevSmall = stDevSmall;
    }

    @Override
    public float[] getDevsSmall() {
        return stDevSmall;
    }

    @Override
    public void setDevsBig(float[] stDevBig) {
        this.stDevBig = stDevBig;
    }

    @Override
    public float[] getDevsBig() {
        return stDevBig;
    }

    @Override
    public void mutate(Object instance) throws Exception {
        DataInstance original = (DataInstance) instance;
        Random randa = new Random();
        float decision;
        if (original.fAttr != null && original.fAttr.length > 0) {
            if (beginIndex < 0) {
                beginIndex = 0;
            }
            if (endIndex < 0) {
                endIndex = original.fAttr.length - 1;
            }
            for (int i = beginIndex; i <= endIndex; i++) {
                if (DataMineConstants.isAcceptableFloat(original.fAttr[i])) {
                    if (pMutation < 1) {
                        decision = randa.nextFloat();
                        if (decision < pMutation) {
                            decision = randa.nextFloat();
                            if (decision < 0.5f) {
                                decision = randa.nextFloat();
                                if (decision < pSmall) {
                                    original.fAttr[i] += randa.nextGaussian()
                                            * stDevSmall[i];
                                } else {
                                    original.fAttr[i] +=
                                            randa.nextGaussian() * stDevBig[i];
                                }
                            } else {
                                if (decision < pSmall) {
                                    original.fAttr[i] -= randa.nextGaussian()
                                            * stDevSmall[i];
                                } else {
                                    original.fAttr[i] -=
                                            randa.nextGaussian() * stDevBig[i];
                                }
                            }
                            // Validate the values.
                            original.fAttr[i] =
                                    Math.max(original.fAttr[i], lowerBounds[i]);
                            original.fAttr[i] =
                                    Math.min(original.fAttr[i], upperBounds[i]);
                        }
                    } else {
                        decision = randa.nextFloat();
                        if (decision < 0.5f) {
                            decision = randa.nextFloat();
                            if (decision < pSmall) {
                                original.fAttr[i] += randa.nextGaussian()
                                        * stDevSmall[i];
                            } else {
                                original.fAttr[i] += randa.nextGaussian()
                                        * stDevBig[i];
                            }
                        } else {
                            decision = randa.nextFloat();
                            if (decision < pSmall) {
                                original.fAttr[i] -= randa.nextGaussian()
                                        * stDevSmall[i];
                            } else {
                                original.fAttr[i] -= randa.nextGaussian()
                                        * stDevBig[i];
                            }
                        }
                        // Validate the values.
                        original.fAttr[i] =
                                Math.max(original.fAttr[i], lowerBounds[i]);
                        original.fAttr[i] =
                                Math.min(original.fAttr[i], upperBounds[i]);
                    }
                }
            }
        }
    }

    @Override
    public Object mutateNew(Object instance) throws Exception {
        DataInstance original = (DataInstance) instance;
        DataInstance copy = original.copy();
        copy.setCategory(original.getCategory());
        Random randa = new Random();
        float decision;
        if (original.fAttr != null && original.fAttr.length > 0) {
            if (beginIndex < 0) {
                beginIndex = 0;
            }
            if (endIndex < 0) {
                endIndex = original.fAttr.length - 1;
            }
            for (int i = beginIndex; i <= endIndex; i++) {
                if (DataMineConstants.isAcceptableFloat(original.fAttr[i])) {
                    if (pMutation < 1) {
                        decision = randa.nextFloat();
                        if (decision < pMutation) {
                            decision = randa.nextFloat();
                            if (decision < 0.5f) {
                                decision = randa.nextFloat();
                                if (decision < pSmall) {
                                    copy.fAttr[i] += randa.nextGaussian()
                                            * stDevSmall[i];
                                } else {
                                    copy.fAttr[i] += randa.nextGaussian()
                                            * stDevBig[i];
                                }
                            } else {
                                decision = randa.nextFloat();
                                if (decision < pSmall) {
                                    copy.fAttr[i] -= randa.nextGaussian()
                                            * stDevSmall[i];
                                } else {
                                    copy.fAttr[i] -= randa.nextGaussian()
                                            * stDevBig[i];
                                }
                            }
                            // Validate the values.
                            copy.fAttr[i] =
                                    Math.max(copy.fAttr[i], lowerBounds[i]);
                            copy.fAttr[i] =
                                    Math.min(copy.fAttr[i], upperBounds[i]);
                        }
                    } else {
                        decision = randa.nextFloat();
                        if (decision < 0.5f) {
                            decision = randa.nextFloat();
                            if (decision < pSmall) {
                                copy.fAttr[i] += randa.nextGaussian()
                                        * stDevSmall[i];
                            } else {
                                copy.fAttr[i] += randa.nextGaussian()
                                        * stDevBig[i];
                            }
                        } else {
                            decision = randa.nextFloat();
                            if (decision < pSmall) {
                                copy.fAttr[i] -= randa.nextGaussian()
                                        * stDevSmall[i];
                            } else {
                                copy.fAttr[i] -= randa.nextGaussian()
                                        * stDevBig[i];
                            }
                        }
                        // Validate the values.
                        copy.fAttr[i] = Math.max(copy.fAttr[i], lowerBounds[i]);
                        copy.fAttr[i] = Math.min(copy.fAttr[i], upperBounds[i]);
                    }
                }
            }
        }
        return copy;
    }

    @Override
    public Object[] mutateNew(Object[] instances) throws Exception {
        Object[] output = new Object[instances.length];
        for (int i = 0; i < instances.length; i++) {
            output[i] = mutateNew(instances[i]);
        }
        return output;
    }

    @Override
    public void mutate(Object[] instances) throws Exception {
        for (int i = 0; i < instances.length; i++) {
            mutate(instances[i]);
        }
    }
}
