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
import optimization.stochastic.operators.RecombinationInterface;

/**
 * Class that performs affine recombination of float features.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class AffineFloatRecombiner implements RecombinationInterface {

    public AffineFloatRecombiner() {
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
                result.fAttr[i] = min + randa.nextFloat() * (max - min);
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
