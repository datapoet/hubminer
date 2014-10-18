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

import optimization.stochastic.operators.RecombinationInterface;
import data.representation.DataInstance;
import data.representation.util.DataMineConstants;
import java.util.Random;

/**
 * Class that performs blend recombination of feature values.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class BlendFloatRecombiner implements RecombinationInterface {

    private float alpha;
    private float[] lowerBounds;
    private float[] upperBounds;

    /**
     * @param alpha Blend parameter.
     * @param lowerBounds Lower value bounds.
     * @param upperBounds Upper value bounds.
     */
    public BlendFloatRecombiner(
            float alpha,
            float[] lowerBounds,
            float[] upperBounds) {
        this.alpha = alpha;
        this.lowerBounds = lowerBounds;
        this.upperBounds = upperBounds;
    }

    /**
     * @return Float that is the blend parameter.
     */
    public float getAlpha() {
        return alpha;
    }

    /**
     * @param alpha Float that is the blend parameter.
     */
    public void setAlpha(float alpha) {
        this.alpha = alpha;
    }

    /**
     * @return Float array representing lower value bounds.
     */
    public float[] getLowerBounds() {
        return lowerBounds;
    }

    /**
     * @return Float array representing upper value bounds.
     */
    public float[] getUpperBounds() {
        return upperBounds;
    }

    /**
     * @param lowerBounds Float array representing lower value bounds.
     */
    public void setLowerBounds(float[] lowerBounds) {
        this.lowerBounds = lowerBounds;
    }

    /**
     * @param lowerBounds Float array representing upper value bounds.
     */
    public void setUpperBounds(float[] upperBounds) {
        this.upperBounds = upperBounds;
    }

    @Override
    public Object recombine(Object o1, Object o2) throws Exception {
        DataInstance d1 = (DataInstance) o1;
        DataInstance d2 = (DataInstance) o2;
        DataInstance result = d1.copyContent();
        float max;
        float min;
        Random randa = new Random();
        for (int i = 0; i < d1.fAttr.length; i++) {
            if (DataMineConstants.isAcceptableFloat(d1.fAttr[i])
                    && DataMineConstants.isAcceptableFloat(d2.fAttr[i])) {
                min = Math.min(d1.fAttr[i], d2.fAttr[i]);
                max = Math.max(d1.fAttr[i], d2.fAttr[i]);
                d1.fAttr[i] = min - alpha + randa.nextFloat()
                        * (max - min + 2 * alpha);
                // Validate the values.
                d1.fAttr[i] = Math.max(d1.fAttr[i], lowerBounds[i]);
                d1.fAttr[i] = Math.min(d1.fAttr[i], upperBounds[i]);
            }
        }
        return result;
    }

    @Override
    public Object[] recombinePair(Object o1, Object o2) throws Exception {
        Object[] output = new Object[2];
        output[0] = recombine(o1, o2);
        output[1] = recombine(o1, o2);
        return output;
    }

    @Override
    public Object[] recombine(Object[] oArray1, Object[] oArray2)
            throws Exception {
        Object[] output = new Object[oArray1.length];
        for (int i = 0; i < output.length; i++) {
            output[i] = recombine(oArray1[i], oArray2[i]);
        }
        return output;
    }
}
