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

import data.representation.DataInstance;

/**
 * This class is a meta-generator that acts as a mix of different underlying
 * data generators in one.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class MixtureOfFloatGenerators {

    private DataGenerator[] generators;

    public MixtureOfFloatGenerators() {
    }

    /**
     *
     * @param generators An array of underlying data generators.
     */
    public MixtureOfFloatGenerators(DataGenerator[] generators) {
        this.generators = generators;
    }

    /**
     *
     * @return An array of data generators used in this mix.
     */
    public DataGenerator[] getGenerators() {
        return generators;
    }

    /**
     * @param generators An array of data generators used in this mix.
     */
    public void setGenerators(DataGenerator[] generators) {
        this.generators = generators;
    }

    /**
     * Generate a certain number of points based on each underlying data
     * generator.
     *
     * @param amounts An integer array specifying how many instances to generate
     * by each individual generator specified in the indexes.
     * @param generatorIndexes Indexes of the generators to use.
     * @return An array of float arrays representing the generated features.
     */
    public float[][] generateData(int[] amounts, int[] generatorIndexes) {
        int totalAmount = 0;
        if (amounts == null || generatorIndexes == null || amounts.length == 0
                || generatorIndexes.length == 0) {
            return null;
        }
        for (int i = 0; i < amounts.length; i++) {
            totalAmount += amounts[i];
        }
        int currIndex = -1;
        float[][] newData = new float[totalAmount][];
        for (int i = 0; i < amounts.length; i++) {
            for (int j = 0; j < amounts[i]; j++) {
                newData[++currIndex] =
                        generators[generatorIndexes[i]].generateFloat();
            }
        }
        return newData;
    }

    /**
     * Generate a certain number of points based on each underlying data
     * generator.
     *
     * @param amounts An integer array specifying how many instances to generate
     * by each individual generator specified in the indexes.
     * @param generatorIndexes Indexes of the generators to use.
     * @return An array of generated DataInstance objects.
     */
    public DataInstance[] generateDataInstances(int[] amounts,
            int[] generatorIndexes) {
        int totalAmount = 0;
        if (amounts == null || generatorIndexes == null || amounts.length == 0
                || generatorIndexes.length == 0) {
            return null;
        }
        for (int i = 0; i < amounts.length; i++) {
            totalAmount += amounts[i];
        }
        int currIndex = -1;
        DataInstance[] newData = new DataInstance[totalAmount];
        DataInstance instance;
        for (int i = 0; i < amounts.length; i++) {
            for (int j = 0; j < amounts[i]; j++) {
                instance = new DataInstance();
                instance.fAttr =
                        generators[generatorIndexes[i]].generateFloat();
                instance.setCategory(generatorIndexes[i]);
                newData[++currIndex] = instance;
            }
        }
        return newData;
    }

    /**
     * Generate a certain number of points based on each underlying data
     * generator.
     *
     * @param amounts An integer array specifying how many instances to generate
     * by each individual generator.
     * @return An array of float arrays representing the generated features.
     */
    public float[][] generateData(int[] amounts) {
        int totalAmount = 0;
        if (amounts == null || amounts.length == 0) {
            return null;
        }
        for (int i = 0; i < amounts.length; i++) {
            totalAmount += amounts[i];
        }
        int currIndex = -1;
        float[][] newData = new float[totalAmount][];
        for (int i = 0; i < amounts.length; i++) {
            for (int j = 0; j < amounts[i]; j++) {
                newData[++currIndex] = generators[i].generateFloat();
            }
        }
        return newData;
    }

    /**
     * Generate a certain number of points based on each underlying data
     * generator.
     *
     * @param amounts An integer array specifying how many instances to generate
     * by each individual generator.
     * @return An array of generated DataInstance objects.
     */
    public DataInstance[] generateDataInstances(int[] amounts) {
        int totalAmount = 0;
        if (amounts == null || amounts.length == 0) {
            return null;
        }
        for (int i = 0; i < amounts.length; i++) {
            totalAmount += amounts[i];
        }
        int currIndex = -1;
        DataInstance[] newData = new DataInstance[totalAmount];
        DataInstance instance;
        for (int i = 0; i < amounts.length; i++) {
            for (int j = 0; j < amounts[i]; j++) {
                instance = new DataInstance();
                instance.fAttr = generators[i].generateFloat();
                instance.setCategory(i);
                newData[++currIndex] = instance;
            }
        }
        return newData;
    }
}
