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
package data.generators;

import java.util.Random;
import data.representation.DataInstance;

/**
 * Generates new synthetic data instances according to a uniform model.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class UniformGenerator implements DataGenerator {

    private float[] lowerBounds;
    private float[] upperBounds;

    /**
     *
     * @param lowerBounds Float array of lower feature value bounds.
     * @param upperBounds Float array of upper feature value bounds.
     */
    public UniformGenerator(float[] lowerBounds, float[] upperBounds) {
        this.lowerBounds = lowerBounds;
        this.upperBounds = upperBounds;
    }

    @Override
    public float[] generateFloat() {
        int length = upperBounds.length;
        float[] result = new float[length];
        Random randa = new Random();
        for (int i = 0; i < length; i++) {
            result[i] = (float) ((double) lowerBounds[i]
                    + (((double) upperBounds[i] - (double) lowerBounds[i])
                    * randa.nextDouble()));
        }
        return result;
    }

    @Override
    public int[] generateInt() {
        int length = upperBounds.length;
        int[] result = new int[length];
        Random randa = new Random();
        for (int i = 0; i < length; i++) {
            result[i] = (int) (lowerBounds[i]
                    + ((upperBounds[i] - lowerBounds[i]) * randa.nextFloat()));
        }
        return result;
    }

    /**
     * Generate a number of synthetic instances drawn from the specified uniform
     * data distribution.
     *
     * @param numInstances Number of instances to generate.
     * @return An array of generated DataInstance objects.
     */
    public DataInstance[] generateDataInstances(int numInstances) {
        DataInstance[] instances = new DataInstance[numInstances];
        for (int i = 0; i < instances.length; i++) {
            instances[i] = new DataInstance();
            instances[i].fAttr = generateFloat();
        }
        return instances;
    }
}
