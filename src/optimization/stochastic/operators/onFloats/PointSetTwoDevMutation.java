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

import java.util.Random;
import linear.matrix.DataMatrix;
import optimization.stochastic.operators.TwoDevsMutationInterface;

/**
 * Mutation class for point set configurations. It can be used in GA variants of
 * projection methods.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class PointSetTwoDevMutation implements TwoDevsMutationInterface {

    float pMutation = 1;
    float stDevSmall;
    float stDevBig;
    float pSmall;
    float[] lowerBounds;
    float[] upperBounds;
    int numDim = 2;

    /**
     * @param stDevSmall Standard deviation of small mutations.
     * @param stDevBig Standard deviation of large mutations.
     * @param pSmall Probability of small mutations.
     * @param lowerBounds Lower value bounds.
     * @param upperBounds Upper value bounds.
     * @param numDim Number of dimensions.
     */
    public PointSetTwoDevMutation(
            float stDevSmall,
            float stDevBig,
            float pSmall,
            float[] lowerBounds,
            float[] upperBounds,
            int numDim) {
        this.stDevSmall = stDevSmall;
        this.stDevBig = stDevBig;
        this.pSmall = pSmall;
        this.lowerBounds = lowerBounds;
        this.upperBounds = upperBounds;
        this.numDim = numDim;
    }

    /**
     * @param stDevSmall Standard deviation of small mutations.
     * @param stDevBig Standard deviation of large mutations.
     * @param pSmall Probability of small mutations.
     * @param lowerBounds Lower value bounds.
     * @param upperBounds Upper value bounds.
     * @param pMutation Probability of individual mutations.
     * @param numDim Number of dimensions.
     */
    public PointSetTwoDevMutation(
            float stDevSmall,
            float stDevBig,
            float pSmall,
            float[] lowerBounds,
            float[] upperBounds,
            float pMutation,
            int numDim) {
        this.stDevSmall = stDevSmall;
        this.stDevBig = stDevBig;
        this.pSmall = pSmall;
        this.lowerBounds = lowerBounds;
        this.upperBounds = upperBounds;
        this.pMutation = pMutation;
        this.numDim = numDim;
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
        this.stDevSmall = stDevSmall;
    }

    @Override
    public float getDevSmall() {
        return stDevSmall;
    }

    @Override
    public void setDevBig(float stDevBig) {
        this.stDevBig = stDevBig;
    }

    @Override
    public float getDevBig() {
        return stDevBig;
    }

    @Override
    public void setDevsSmall(float[] stDevSmall) {
        this.stDevSmall = stDevSmall[0];
    }

    @Override
    public float[] getDevsSmall() {
        float[] result = new float[1];
        result[0] = stDevSmall;
        return result;
    }

    @Override
    public void setDevsBig(float[] stDevBig) {
        this.stDevBig = stDevBig[0];
    }

    @Override
    public float[] getDevsBig() {
        float[] result = new float[1];
        result[0] = stDevBig;
        return result;
    }

    @Override
    public void mutate(Object instance) throws Exception {
        DataMatrix config = (DataMatrix) instance;
        float[][] confArr = config.getFloatMatrix();
        Random randa = new Random();
        float choice;
        choice = randa.nextFloat();
        for (int i = 0; i < confArr.length; i++) {
            if (choice < pMutation) {
                for (int j = 0; j < numDim; j++) {
                    choice = randa.nextFloat();
                    if (choice < pSmall) {
                        confArr[i][j] +=
                                (float) randa.nextGaussian() * stDevSmall;
                    } else {
                        confArr[i][j] +=
                                (float) randa.nextGaussian() * stDevBig;
                    }
                    confArr[i][j] = Math.max(lowerBounds[j], confArr[i][j]);
                    confArr[i][j] = Math.min(upperBounds[j], confArr[i][j]);
                }
            }
        }
    }

    @Override
    public Object mutateNew(Object instance) throws Exception {
        DataMatrix config = (DataMatrix) instance;
        float[][] confArr = config.getFloatMatrix();
        float[][] copyArr = new float[confArr.length][numDim];
        Random randa = new Random();
        float choice;
        choice = randa.nextFloat();
        for (int i = 0; i < confArr.length; i++) {
            if (choice < pMutation) {
                for (int j = 0; j < numDim; j++) {
                    choice = randa.nextFloat();
                    if (choice < pSmall) {
                        copyArr[i][j] = confArr[i][j]
                                + (float) randa.nextGaussian() * stDevSmall;
                    } else {
                        copyArr[i][j] = confArr[i][j]
                                + (float) randa.nextGaussian() * stDevBig;
                    }
                    copyArr[i][j] = Math.max(lowerBounds[j], copyArr[i][j]);
                    copyArr[i][j] = Math.min(upperBounds[j], copyArr[i][j]);
                }
            }
        }
        DataMatrix copy = new DataMatrix(copyArr);
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
