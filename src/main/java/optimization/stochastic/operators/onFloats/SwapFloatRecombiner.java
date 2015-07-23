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
import java.util.Random;
import optimization.stochastic.operators.RecombinationInterface;

/**
 * A recombination method where parts of two genotypes are randomly swapped by a
 * certain probability.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class SwapFloatRecombiner implements RecombinationInterface {

    private float pSwap;

    /**
     * @param pSwap Probability of an allele swap.
     */
    public SwapFloatRecombiner(float pSwap) {
        this.pSwap = pSwap;
    }

    @Override
    public Object recombine(Object o1, Object o2) throws Exception {
        DataInstance d1 = (DataInstance) o1;
        DataInstance d2 = (DataInstance) o2;
        DataInstance output = d1.copyContent();
        float decision;
        if (d1.fAttr != null) {
            Random randa = new Random();
            for (int i = 0; i < d1.fAttr.length; i++) {
                decision = randa.nextFloat();
                if (decision < pSwap) {
                    output.fAttr[i] = d2.fAttr[i];
                }
            }
        }
        return output;
    }

    @Override
    public Object[] recombinePair(Object o1, Object o2) throws Exception {
        DataInstance d1 = (DataInstance) o1;
        DataInstance d2 = (DataInstance) o2;
        DataInstance[] output = new DataInstance[2];
        output[0] = d1.copyContent();
        output[1] = d2.copyContent();
        float decision;
        if (d1.fAttr != null) {
            Random randa = new Random();
            for (int i = 0; i < d1.fAttr.length; i++) {
                decision = randa.nextFloat();
                if (decision < pSwap) {
                    output[0].fAttr[i] = d2.fAttr[i];
                    output[1].fAttr[i] = d1.fAttr[i];
                }
            }
        }
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
